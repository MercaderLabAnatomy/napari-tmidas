"""
Interactive SAM2 paths of Batch Crop Anything, driven by a scripted model.

``test_crop_anything_coverage.py`` pins the parts of ``BatchCropAnything``
that never need a model.  This file covers what happens *around* the model:
the 3D (video) setup, the click handlers that turn a mouse press into a
SAM2 prompt, propagation of a mask through a stack, and the magicgui entry
point.  SAM2 is replaced by predictors whose answers are fixed by each test,
so every assertion below checks what the widget did with that answer:
which labels were written where, which prompts were recorded, which layers
appeared or went away, and what the status bar says.

Every dialog and thread the module could start is neutralised on the
module object (see the fixtures), because a modal dialog hangs pytest and
the "remove progress layer later" thread would outlive the viewer.
"""

import os
import sys
import tempfile
import threading
import types

import numpy as np
import pytest
import tifffile

from napari_tmidas import _crop_anything as ca

pytest.importorskip("pytestqt")

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------
class FakeTensor:
    """The slice of the torch tensor API the module uses on mask logits."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def __gt__(self, other):
        return FakeTensor(self.array > other)

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def __len__(self):
        return len(self.array)

    @property
    def ndim(self):
        return self.array.ndim

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class StubOutOfMemoryError(RuntimeError):
    pass


def make_torch_stub():
    stub = types.SimpleNamespace()
    stub.float32 = "float32"
    stub.inference_mode = lambda *args, **kwargs: NullContext()
    stub.autocast = lambda *args, **kwargs: NullContext()
    stub.cuda = types.SimpleNamespace(
        OutOfMemoryError=StubOutOfMemoryError,
        is_available=lambda: False,
    )
    return stub


class FakeEvent:
    """A napari mouse event as the handlers consume it."""

    def __init__(
        self, position, button=1, modifiers=(), event_type="mouse_press"
    ):
        self.position = position
        self.button = button
        self.modifiers = modifiers
        self.type = event_type


class FakePointsLayer:
    """A points layer with only what ``_on_points_clicked`` touches."""

    def __init__(self, ndim):
        self.data = np.zeros((0, ndim))
        self.face_color = "green"
        self.name = "Points (Click to Add)"
        self.mouse_drag_callbacks = []


class IdentityLayer:
    """A layer whose world and data coordinates coincide."""

    def world_to_data(self, position):
        return np.asarray(position, dtype=float)


def logits(mask):
    """SAM2-shaped logits for one object: (1, 1, H, W), >0 inside."""
    return np.where(np.asarray(mask, dtype=bool), 5.0, -5.0)[
        np.newaxis, np.newaxis
    ]


class VideoPredictor:
    """A SAM2 video predictor with scripted answers.

    ``frame_mask`` answers ``add_new_points_or_box``; ``propagated`` maps a
    frame index to the mask ``propagate_in_video`` reports for
    ``reported_ids`` on that frame.
    """

    def __init__(self, frame_mask, propagated=None, reported_ids=(1,)):
        self.frame_mask = np.asarray(frame_mask, dtype=bool)
        self.propagated = propagated or {}
        self.reported_ids = list(reported_ids)
        self.add_calls = []
        self.init_paths = []
        self.state = object()

    def init_state(self, path):
        self.init_paths.append(path)
        return self.state

    def add_new_points_or_box(self, **kwargs):
        self.add_calls.append(kwargs)
        return None, [kwargs["obj_id"]], FakeTensor(logits(self.frame_mask))

    def propagate_in_video(self, state):
        assert state is self.state
        for frame_idx, mask in self.propagated.items():
            stacked = np.concatenate(
                [logits(mask)] * len(self.reported_ids), axis=0
            )
            yield frame_idx, list(self.reported_ids), FakeTensor(stacked)


class ImagePredictor:
    """A SAM2 image predictor whose masks are chosen by the test."""

    def __init__(self, masks, scores=None):
        self.masks = np.asarray(masks)
        self.scores = (
            np.full(len(self.masks), 0.9) if scores is None else scores
        )
        self.set_image_calls = []
        self.predict_kwargs = []

    def set_image(self, image):
        self.set_image_calls.append(image)

    def predict(self, **kwargs):
        self.predict_kwargs.append(kwargs)
        return self.masks, np.asarray(self.scores), None


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def no_sam2(monkeypatch):
    """Construct processors without loading a model."""

    def fake_init(self):
        self.predictor = None
        self.device = types.SimpleNamespace(type="cpu")

    monkeypatch.setattr(ca.BatchCropAnything, "_initialize_sam2", fake_init)


@pytest.fixture
def torch_stub(monkeypatch):
    monkeypatch.setattr(ca, "torch", make_torch_stub())


@pytest.fixture
def no_threads(monkeypatch):
    """Never start the delayed "remove progress layer" thread."""
    started = []

    class InstantThread:
        def __init__(self, target=None, **kwargs):
            self.target = target

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(threading, "Thread", InstantThread)
    return started


@pytest.fixture
def processor(make_napari_viewer, no_sam2, torch_stub):
    return ca.BatchCropAnything(make_napari_viewer())


def load_volume(proc, tmp_path, image=None, name="vol.tif"):
    """Put a (3, 8, 8) stack on disk and into the processor, 3D mode."""
    if image is None:
        image = np.full((3, 8, 8), 7, dtype=np.uint8)
    path = tmp_path / name
    tifffile.imwrite(path, image)
    proc.use_3d = True
    proc.images = [str(path)]
    proc.current_index = 0
    proc.original_image = image
    proc.current_image_for_segmentation = image
    proc.segmentation_result = np.zeros(image.shape, dtype=np.uint32)
    proc.label_info = {}
    proc.image_layer = proc.viewer.add_image(image, name="vol")
    proc.label_layer = proc.viewer.add_labels(
        proc.segmentation_result, name=f"Segmentation ({name})"
    )
    return proc


def load_scene(proc, tmp_path, image=None, name="scene.tif"):
    """Put a 16x16 image on disk and into the processor, 2D mode."""
    if image is None:
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / name
    tifffile.imwrite(path, image)
    proc.use_3d = False
    proc.images = [str(path)]
    proc.current_index = 0
    proc.original_image = image
    proc.current_image_for_segmentation = image
    proc.segmentation_result = np.zeros((16, 16), dtype=np.uint32)
    proc.label_info = {}
    proc.image_layer = proc.viewer.add_image(image, name="scene")
    proc.label_layer = proc.viewer.add_labels(
        proc.segmentation_result, name=f"Segmentation ({name})"
    )
    return proc


@pytest.fixture
def volume(processor, tmp_path):
    return load_volume(processor, tmp_path)


@pytest.fixture
def scene(processor, tmp_path):
    return load_scene(processor, tmp_path)


def square(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def layer_names(viewer):
    return [layer.name for layer in viewer.layers]


# --------------------------------------------------------------------------
# generate_segmentation_with_sensitivity
# --------------------------------------------------------------------------
class TestGenerateWithSensitivity:
    """The dispatcher between the 2D and 3D setup passes."""

    @pytest.fixture
    def recorded(self, scene, monkeypatch):
        calls = []
        monkeypatch.setattr(
            scene,
            "_generate_2d_segmentation",
            lambda threshold: calls.append(("2d", threshold)),
        )
        monkeypatch.setattr(
            scene,
            "_generate_3d_segmentation",
            lambda threshold, path: calls.append(("3d", threshold, path)),
        )
        scene.predictor = ImagePredictor(np.zeros((1, 16, 16), dtype=bool))
        return scene, calls

    def test_without_a_model_nothing_is_segmented(self, recorded):
        proc, calls = recorded
        proc.predictor = None

        proc.generate_segmentation_with_sensitivity("x.tif", sensitivity=80)

        assert proc.sensitivity == 80
        assert calls == []
        assert proc.viewer.status == (
            "SAM2 model not initialized. Cannot segment images."
        )

    def test_without_an_image_nothing_is_segmented(self, recorded):
        proc, calls = recorded
        proc.current_image_for_segmentation = None

        proc.generate_segmentation_with_sensitivity("x.tif")

        assert calls == []
        assert proc.viewer.status == "No image loaded for segmentation."

    @pytest.mark.parametrize(
        ("sensitivity", "threshold"), [(0, 0.9), (50, 0.7), (100, 0.5)]
    )
    def test_sensitivity_maps_onto_the_confidence_threshold(
        self, recorded, sensitivity, threshold
    ):
        proc, calls = recorded

        proc.generate_segmentation_with_sensitivity(
            "x.tif", sensitivity=sensitivity
        )

        ((mode, value),) = calls
        assert mode == "2d"
        assert value == pytest.approx(threshold)

    def test_3d_mode_hands_the_path_to_the_video_setup(self, recorded):
        proc, calls = recorded
        proc.use_3d = True

        proc.generate_segmentation_with_sensitivity("stack.tif")

        assert calls == [("3d", pytest.approx(0.7), "stack.tif")]

    def test_a_model_error_is_reported_not_raised(self, scene, monkeypatch):
        scene.predictor = ImagePredictor(np.zeros((1, 16, 16), dtype=bool))

        def boom(threshold):
            raise RuntimeError("CUDA went away")

        monkeypatch.setattr(scene, "_generate_2d_segmentation", boom)

        scene.generate_segmentation_with_sensitivity("x.tif")

        assert scene.viewer.status == (
            "Error generating segmentation: CUDA went away"
        )


# --------------------------------------------------------------------------
# _generate_3d_segmentation
# --------------------------------------------------------------------------
class TestGenerate3dSegmentation:
    """Setting up the SAM2 video predictor for a stack."""

    @pytest.fixture(autouse=True)
    def _restore_napari_callbacks(self, processor, monkeypatch):
        """Give napari back the label-layer callbacks the setup strips.

        ``_generate_3d_segmentation`` empties ``mouse_drag_callbacks``
        entirely, including the handler napari's labels-polygon overlay
        installed, and napari then raises when the viewer closes the layer.
        That is a bug in the module (reported, not pinned here); this
        fixture only re-adds napari's own handlers after the test so the
        viewer can be torn down.  It is a no-op once the bug is fixed.
        """
        snapshots = []
        original = processor._update_label_layer

        def recording_update():
            original()
            layer = processor.label_layer
            snapshots.append((layer, list(layer.mouse_drag_callbacks)))

        monkeypatch.setattr(processor, "_update_label_layer", recording_update)
        yield
        for layer, callbacks in snapshots:
            for callback in callbacks:
                if callback not in layer.mouse_drag_callbacks:
                    layer.mouse_drag_callbacks.append(callback)

    @pytest.fixture
    def converted(self, volume, monkeypatch, tmp_path):
        """Record MP4 conversions instead of running ffmpeg."""
        conversions = []

        def fake_tif_to_mp4(path):
            conversions.append(path)
            return str(tmp_path / "converted.mp4")

        monkeypatch.setattr(ca, "tif_to_mp4", fake_tif_to_mp4)
        volume.predictor = VideoPredictor(np.zeros((8, 8)))
        return volume, conversions

    def test_the_stack_is_converted_and_handed_to_the_predictor(
        self, converted, tmp_path
    ):
        proc, conversions = converted
        proc.segmentation_result[0, 0, 0] = 5
        proc.sam2_points_by_obj = {5: [[0, 0]]}

        result = proc._generate_3d_segmentation(0.7, proc.images[0])

        assert result is True
        assert conversions == [proc.images[0]]
        assert proc.predictor.init_paths == [str(tmp_path / "converted.mp4")]
        assert proc._sam2_state is proc.predictor.state
        # A fresh, empty uint32 volume replaces whatever was there.
        assert proc.segmentation_result.shape == (3, 8, 8)
        assert proc.segmentation_result.dtype == np.uint32
        assert not proc.segmentation_result.any()
        assert proc._sam2_next_obj_id == 1
        assert proc._sam2_prompts == {}
        assert proc.sam2_points_by_obj == {}
        assert proc.sam2_labels_by_obj == {}
        assert proc.viewer.status.startswith("3D Mode active")

    def test_clicks_on_the_labels_go_to_the_3d_handler(self, converted):
        proc, _ = converted

        proc._generate_3d_segmentation(0.7, proc.images[0])

        callbacks = proc.label_layer.mouse_drag_callbacks
        assert proc._on_3d_label_clicked in callbacks
        assert proc._on_label_clicked not in callbacks
        labels = [
            layer
            for layer in proc.viewer.layers
            if layer.name.startswith("Segmentation")
        ]
        assert labels == [proc.label_layer]
        assert proc.viewer.dims.point[0] == 0

    def test_an_existing_mp4_is_reused(self, converted, tmp_path):
        proc, conversions = converted
        existing = tmp_path / "vol.mp4"
        existing.write_bytes(b"not really a video")

        proc._generate_3d_segmentation(0.7, proc.images[0])

        assert conversions == []
        assert proc.predictor.init_paths == [str(existing)]

    def test_a_non_string_path_falls_back_to_the_current_image(
        self, converted
    ):
        proc, conversions = converted

        proc._generate_3d_segmentation(0.7, None)

        assert conversions == [proc.images[0]]

    def test_a_4d_stack_is_projected_along_z_first(
        self, processor, tmp_path, monkeypatch
    ):
        image = np.zeros((2, 3, 8, 8), dtype=np.uint8)
        image[0, 2, 1, 1] = 90  # brightest plane differs per time point
        image[1, 0, 4, 4] = 60
        load_volume(processor, tmp_path, image=image, name="tzyx.tif")
        processor.has_z_dim = True
        processor.predictor = VideoPredictor(np.zeros((8, 8)))
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))
        seen = {}

        def fake_tif_to_mp4(path):
            seen["path"] = path
            seen["data"] = tifffile.imread(path)
            return str(scratch / "projected.mp4")

        monkeypatch.setattr(ca, "tif_to_mp4", fake_tif_to_mp4)

        assert processor._generate_3d_segmentation(0.7, processor.images[0])

        assert seen["path"] == str(scratch / "temp_projected_tzyx.tif")
        np.testing.assert_array_equal(seen["data"], image.max(axis=1))
        # The projection was only needed for the conversion.
        assert not os.path.exists(seen["path"])
        assert processor.segmentation_result.shape == (2, 3, 8, 8)

    def test_a_projected_mp4_from_an_earlier_run_is_reused(
        self, processor, tmp_path, monkeypatch
    ):
        image = np.ones((2, 3, 8, 8), dtype=np.uint8)
        load_volume(processor, tmp_path, image=image, name="tzyx.tif")
        processor.has_z_dim = True
        processor.predictor = VideoPredictor(np.zeros((8, 8)))
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        cached = scratch / "temp_projected_tzyx.mp4"
        cached.write_bytes(b"cached")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))
        monkeypatch.setattr(
            ca,
            "tif_to_mp4",
            lambda path: pytest.fail("the cached MP4 should be reused"),
        )

        processor._generate_3d_segmentation(0.7, processor.images[0])

        assert processor.predictor.init_paths == [str(cached)]
        assert not (scratch / "temp_projected_tzyx.tif").exists()

    def test_a_predictor_that_cannot_open_the_video_is_reported(
        self, converted
    ):
        proc, _ = converted

        def refuse(path):
            raise RuntimeError("bad codec")

        proc.predictor.init_state = refuse
        before = proc.label_layer

        result = proc._generate_3d_segmentation(0.7, proc.images[0])

        assert result is None
        assert proc.viewer.status == (
            "Error initializing SAM2 video predictor: bad codec"
        )
        # Setup stopped before the click handler was swapped in.
        assert proc.label_layer is before
        assert proc._on_3d_label_clicked not in (
            proc.label_layer.mouse_drag_callbacks
        )

    def test_a_failed_conversion_is_reported(self, volume, monkeypatch):
        volume.predictor = VideoPredictor(np.zeros((8, 8)))

        def missing(path):
            raise FileNotFoundError("ffmpeg not found")

        monkeypatch.setattr(ca, "tif_to_mp4", missing)

        result = volume._generate_3d_segmentation(0.7, volume.images[0])

        assert result is False
        assert volume.viewer.status == (
            "Error in 3D segmentation setup: ffmpeg not found"
        )
        assert volume.predictor.init_paths == []


# --------------------------------------------------------------------------
# _on_3d_label_clicked
# --------------------------------------------------------------------------
class TestOn3dLabelClicked:
    """A click on the 3D label layer becomes a SAM2 video prompt."""

    MASK = square((8, 8), 2, 4, 2, 5)  # 6 pixels

    @pytest.fixture
    def armed(self, volume, no_threads):
        volume.predictor = VideoPredictor(
            self.MASK, propagated={0: self.MASK, 1: self.MASK, 2: self.MASK}
        )
        volume._sam2_state = volume.predictor.state
        volume._sam2_next_obj_id = 1
        volume.sam2_points_by_obj = {}
        volume.sam2_labels_by_obj = {}
        return volume

    def click(self, proc, position, **kwargs):
        proc._on_3d_label_clicked(
            proc.label_layer, FakeEvent(position, **kwargs)
        )

    def test_a_click_on_background_creates_a_propagated_object(self, armed):
        self.click(armed, (1, 3, 4))

        (call,) = armed.predictor.add_calls
        assert call["frame_idx"] == 1
        assert call["obj_id"] == 1
        assert call["inference_state"] is armed.predictor.state
        np.testing.assert_array_equal(call["points"], [[4, 3]])
        np.testing.assert_array_equal(call["labels"], [1])
        assert call["points"].dtype == np.float32
        assert call["labels"].dtype == np.int32
        for frame in range(3):
            np.testing.assert_array_equal(
                armed.segmentation_result[frame] == 1, self.MASK
            )
        assert armed._sam2_next_obj_id == 2
        assert armed.sam2_points_by_obj == {1: [[4, 3]]}
        assert armed.sam2_labels_by_obj == {1: [1]}
        assert armed.viewer.status == "Updated 3D object 1 across all frames"

    def test_the_prompt_is_shown_in_a_layer_per_object(self, armed):
        self.click(armed, (1, 3, 4))

        points = armed.viewer.layers["Points for Object 1"]
        np.testing.assert_array_equal(points.data, [[1, 3, 4]])
        # The label layer was rebuilt from the new segmentation.
        assert armed.label_layer.data is armed.segmentation_result
        assert int((armed.label_layer.data == 1).sum()) == 18

    def test_a_shift_click_refines_the_object_underneath(self, armed):
        self.click(armed, (1, 3, 4))
        armed.predictor.propagated = {}

        self.click(armed, (1, 2, 2), modifiers=("Shift",))

        first, second = armed.predictor.add_calls
        # Same object, both prompts sent, the second one negative.
        assert second["obj_id"] == 1
        np.testing.assert_array_equal(second["points"], [[4, 3], [2, 2]])
        np.testing.assert_array_equal(second["labels"], [1, -1])
        assert armed.sam2_labels_by_obj == {1: [1, -1]}
        # No new object was started and no second layer appeared.
        assert armed._sam2_next_obj_id == 2
        points = armed.viewer.layers["Points for Object 1"]
        np.testing.assert_array_equal(points.data, [[1, 3, 4], [1, 2, 2]])
        assert (
            sum(
                "Points for Object" in name
                for name in layer_names(armed.viewer)
            )
            == 1
        )

    def test_a_second_background_click_starts_a_new_object(self, armed):
        self.click(armed, (0, 3, 4))
        armed.predictor.frame_mask = square((8, 8), 6, 8, 6, 8)
        armed.predictor.propagated = {0: square((8, 8), 6, 8, 6, 8)}
        armed.predictor.reported_ids = [2]

        self.click(armed, (0, 7, 7))

        assert armed.predictor.add_calls[1]["obj_id"] == 2
        assert armed.segmentation_result[0, 7, 7] == 2
        assert armed.segmentation_result[0, 3, 3] == 1
        assert "Points for Object 2" in layer_names(armed.viewer)
        assert armed._sam2_next_obj_id == 3

    def test_a_2d_position_uses_the_current_slice(self, armed):
        armed.viewer.dims.set_current_step(0, 2)

        armed._on_3d_label_clicked(IdentityLayer(), FakeEvent((3, 4)))

        (call,) = armed.predictor.add_calls
        assert call["frame_idx"] == 2
        np.testing.assert_array_equal(call["points"], [[4, 3]])

    def test_a_4d_position_is_rejected(self, armed):
        armed._on_3d_label_clicked(IdentityLayer(), FakeEvent((0, 1, 3, 4)))

        assert armed.viewer.status.startswith(
            "Unexpected coordinate dimensions"
        )
        assert armed.predictor.add_calls == []
        assert armed.sam2_points_by_obj == {}

    def test_other_buttons_are_ignored(self, armed):
        self.click(armed, (1, 3, 4), button=2)

        assert armed.predictor.add_calls == []
        assert "Points for Object 1" not in layer_names(armed.viewer)

    def test_a_low_resolution_mask_is_upscaled_to_the_frame(self, armed):
        small = np.zeros((4, 4), dtype=bool)
        small[1, 1] = True  # -> rows/cols 2:4 at full resolution
        armed.predictor.frame_mask = small
        armed.predictor.propagated = {2: small}

        self.click(armed, (0, 3, 3))

        np.testing.assert_array_equal(
            armed.segmentation_result[0] == 1, square((8, 8), 2, 4, 2, 4)
        )
        np.testing.assert_array_equal(
            armed.segmentation_result[2] == 1, square((8, 8), 2, 4, 2, 4)
        )
        assert not armed.segmentation_result[1].any()

    def test_without_a_video_state_only_the_prompt_is_kept(self, armed):
        armed._sam2_state = None

        self.click(armed, (1, 3, 4))

        assert armed.viewer.status == "SAM2 3D state not initialized"
        assert armed.predictor.add_calls == []
        assert armed.sam2_points_by_obj == {1: [[4, 3]]}
        assert not armed.segmentation_result.any()

    def test_a_first_click_sets_up_its_own_bookkeeping(self, volume):
        """No 3D setup pass has run: the handler starts from scratch."""
        volume._sam2_state = None

        self.click(volume, (1, 3, 4))

        assert volume._sam2_next_obj_id == 2
        assert volume.sam2_points_by_obj == {1: [[4, 3]]}
        assert volume.sam2_labels_by_obj == {1: [1]}

    def test_a_points_layer_from_a_previous_image_is_extended(self, volume):
        """A leftover object layer is reused, its prompt lists started anew."""
        volume._sam2_state = None
        volume._sam2_next_obj_id = 1
        volume.viewer.add_points(
            np.array([[0, 0, 0]]), name="Points for Object 1"
        )

        self.click(volume, (1, 3, 4))

        assert volume.sam2_points_by_obj == {1: [[4, 3]]}
        assert volume.sam2_labels_by_obj == {1: [1]}
        np.testing.assert_array_equal(
            volume.viewer.layers["Points for Object 1"].data,
            [[0, 0, 0], [1, 3, 4]],
        )

    def test_the_label_table_is_refreshed(self, armed):
        table = ca.QTableWidget()
        armed.label_table_widget = table

        self.click(armed, (1, 3, 4))

        assert table.rowCount() == 1

    def test_a_model_error_is_reported_not_raised(self, armed):
        def boom(**kwargs):
            raise RuntimeError("model crashed")

        armed.predictor.add_new_points_or_box = boom

        self.click(armed, (1, 3, 4))

        assert (
            armed.viewer.status == "Error in 3D click handler: model crashed"
        )
        assert not armed.segmentation_result.any()


# --------------------------------------------------------------------------
# _propagate_mask_for_current_object
# --------------------------------------------------------------------------
class TestPropagateFallbacks:
    """Propagation when the model misbehaves or reports odd frames."""

    @pytest.fixture
    def seeded(self, volume, no_threads):
        volume.segmentation_result[1][square((8, 8), 1, 3, 1, 3)] = 4
        volume.predictor = VideoPredictor(np.zeros((8, 8)))
        volume._sam2_state = volume.predictor.state
        return volume

    def progress(self, proc):
        return proc.viewer.layers["Propagation Progress"].data

    def test_a_failing_model_copies_the_current_frame(
        self, seeded, no_threads
    ):
        def broken(state):
            raise RuntimeError("dtype mismatch")
            yield  # pragma: no cover - makes this a generator

        seeded.predictor.propagate_in_video = broken
        seeded.segmentation_result[2][1, 1] = 9  # already taken

        seeded._propagate_mask_for_current_object(4, 1)

        current = square((8, 8), 1, 3, 1, 3)
        np.testing.assert_array_equal(
            seeded.segmentation_result[0] == 4, current
        )
        # Foreign labels are never overwritten by the fallback.
        assert seeded.segmentation_result[2][1, 1] == 9
        assert int((seeded.segmentation_result[2] == 4).sum()) == 3
        progress = self.progress(seeded)
        np.testing.assert_allclose(progress[1][current], 0.8)
        np.testing.assert_allclose(progress[0][current], 0.5)
        assert seeded.viewer.status == "Propagation of object 4 complete"
        # Clean-up of the overlay is scheduled, not done inline.
        assert len(no_threads) == 1

    def test_frames_beyond_the_stack_are_skipped(self, seeded):
        seeded.predictor.propagated = {
            0: square((8, 8), 5, 7, 5, 7),
            7: np.ones((8, 8)),
        }
        seeded.predictor.reported_ids = [4]

        seeded._propagate_mask_for_current_object(4, 1)

        np.testing.assert_array_equal(
            seeded.segmentation_result[0] == 4, square((8, 8), 5, 7, 5, 7)
        )
        assert not seeded.segmentation_result[2].any()

    def test_low_resolution_masks_are_upscaled(self, seeded):
        small = np.zeros((4, 4), dtype=bool)
        small[3, 3] = True
        seeded.predictor.propagated = {2: small}
        seeded.predictor.reported_ids = [4]

        seeded._propagate_mask_for_current_object(4, 1)

        np.testing.assert_array_equal(
            seeded.segmentation_result[2] == 4, square((8, 8), 6, 8, 6, 8)
        )

    def test_an_existing_progress_overlay_is_reused(self, seeded):
        seeded.viewer.add_image(
            np.zeros((3, 8, 8)), name="Propagation Progress"
        )

        seeded._propagate_mask_for_current_object(4, 1)

        assert layer_names(seeded.viewer).count("Propagation Progress") == 1
        np.testing.assert_allclose(
            self.progress(seeded)[1][square((8, 8), 1, 3, 1, 3)], 0.8
        )

    def test_a_frame_outside_the_stack_is_reported(self, seeded):
        seeded._propagate_mask_for_current_object(4, 10)

        assert seeded.viewer.status.startswith("Error in propagation:")
        assert int((seeded.segmentation_result == 4).sum()) == 4


# --------------------------------------------------------------------------
# _add_3d_prompt and on_apply_propagate
# --------------------------------------------------------------------------
class TestAdd3dPrompt:
    @pytest.fixture
    def prompted(self, volume):
        weak = np.zeros((3, 8, 8), dtype=bool)
        weak[:, 0, 0] = True
        strong = np.zeros((3, 8, 8), dtype=bool)
        strong[1, 2:4, 2:4] = True
        volume.predictor = ImagePredictor([weak, strong], scores=[0.2, 0.9])
        volume._sam2_state = object()
        volume._sam2_next_obj_id = 3
        return volume

    def test_the_best_scoring_mask_becomes_a_new_object(self, prompted):
        prompted._add_3d_prompt((2, 3, 1))

        (call,) = prompted.predictor.predict_kwargs
        assert call["state"] is prompted._sam2_state
        np.testing.assert_array_equal(call["point_coords"], [[2, 3, 1]])
        np.testing.assert_array_equal(call["point_labels"], [1])
        assert call["multimask_output"] is True
        assert int((prompted.segmentation_result == 3).sum()) == 4
        assert prompted.segmentation_result[1, 2, 2] == 3
        assert prompted.segmentation_result[0, 0, 0] == 0
        assert prompted._sam2_next_obj_id == 4
        assert prompted.label_layer.data is prompted.segmentation_result

    def test_no_mask_leaves_the_segmentation_alone(self, prompted):
        prompted.predictor.masks = np.zeros((0, 3, 8, 8), dtype=bool)
        prompted.predictor.scores = np.zeros(0)

        prompted._add_3d_prompt((2, 3, 1))

        assert prompted.viewer.status == "No mask found for this prompt."
        assert not prompted.segmentation_result.any()
        assert prompted._sam2_next_obj_id == 3

    def test_without_a_video_state_nothing_is_asked(self, prompted):
        prompted._sam2_state = None

        prompted._add_3d_prompt((2, 3, 1))

        assert prompted.viewer.status == "SAM2 3D state not initialized."
        assert prompted.predictor.predict_kwargs == []

    def test_without_a_model_nothing_is_asked(self, prompted):
        prompted.predictor = None

        prompted._add_3d_prompt((2, 3, 1))

        assert prompted.viewer.status == "SAM2 predictor not initialized."
        assert not prompted.segmentation_result.any()


class TestApplyPropagate:
    """Rebuilding the whole segmentation from one propagation pass."""

    class MultiObjectPredictor:
        def __init__(self, frames):
            self.frames = frames

        def propagate_in_video(self, state):
            yield from self.frames

    def test_every_object_is_written_to_every_frame(self, volume):
        volume.segmentation_result[:] = 9  # stale result, must go
        one = square((8, 8), 0, 2, 0, 2)
        two = square((8, 8), 5, 8, 5, 8)
        frame_logits = FakeTensor(np.where(np.stack([one, two]), 1.0, -1.0))
        volume.predictor = self.MultiObjectPredictor(
            [
                (0, [1, 2], frame_logits),
                (2, [1, 2], frame_logits),
                (5, [1, 2], frame_logits),  # beyond the stack
            ]
        )
        volume._sam2_state = object()

        volume.on_apply_propagate()

        for frame in (0, 2):
            np.testing.assert_array_equal(
                volume.segmentation_result[frame] == 1, one
            )
            np.testing.assert_array_equal(
                volume.segmentation_result[frame] == 2, two
            )
        assert not volume.segmentation_result[1].any()
        assert (volume.segmentation_result != 9).all()
        assert volume.label_layer.data is volume.segmentation_result
        assert volume.viewer.status == "Propagation complete!"
        qt_window = volume.viewer.window._qt_window
        assert qt_window.cursor().shape() == ca.Qt.ArrowCursor


# --------------------------------------------------------------------------
# _on_points_clicked, 3D
# --------------------------------------------------------------------------
class TestOnPointsClicked3d:
    """A click on the points layer in a stack: prompt, then propagate."""

    MASK = square((8, 8), 1, 4, 1, 4)  # 9 pixels

    @pytest.fixture
    def armed(self, volume, no_threads):
        volume.predictor = VideoPredictor(
            self.MASK, propagated={0: self.MASK, 1: self.MASK, 2: self.MASK}
        )
        volume._sam2_state = volume.predictor.state
        volume.points = FakePointsLayer(ndim=3)
        return volume

    def click(self, proc, position, **kwargs):
        proc._on_points_clicked(proc.points, FakeEvent(position, **kwargs))

    def test_a_click_segments_and_propagates_an_object(
        self, armed, no_threads
    ):
        self.click(armed, (1.2, 2.4, 3.0))

        (call,) = armed.predictor.add_calls
        assert call["frame_idx"] == 1
        assert call["obj_id"] == 1
        np.testing.assert_array_equal(call["points"], [[3, 2]])
        np.testing.assert_array_equal(call["labels"], [1])
        for frame in range(3):
            np.testing.assert_array_equal(
                armed.segmentation_result[frame] == 1, self.MASK
            )
        assert armed.points_data == {1: [[3, 2]]}
        assert armed.points_labels == {1: [1]}
        assert armed._sam2_next_obj_id == 2
        np.testing.assert_array_equal(armed.points.data, [[1, 2, 3]])
        assert armed.points.face_color == ["green"]
        assert "Propagation Progress" in layer_names(armed.viewer)
        assert armed.label_layer.data is armed.segmentation_result
        assert armed.viewer.status == (
            "Object 1 segmented and propagated to all frames"
        )
        assert len(no_threads) == 1

    def test_a_shift_click_joins_the_object_underneath(self, armed):
        self.click(armed, (1, 2, 3))
        armed.predictor.propagated = {}

        self.click(armed, (1, 2, 2), modifiers=("Shift",))

        second = armed.predictor.add_calls[1]
        assert second["obj_id"] == 1
        np.testing.assert_array_equal(second["points"], [[3, 2], [2, 2]])
        np.testing.assert_array_equal(second["labels"], [1, -1])
        assert armed.points_labels == {1: [1, -1]}
        assert armed._sam2_next_obj_id == 2
        assert armed.points.face_color == ["green", "red"]
        np.testing.assert_array_equal(
            armed.points.data, [[1, 2, 3], [1, 2, 2]]
        )

    def test_a_2d_position_uses_the_current_slice(self, armed):
        armed.viewer.dims.set_current_step(0, 2)

        self.click(armed, (5, 6))

        (call,) = armed.predictor.add_calls
        assert call["frame_idx"] == 2
        np.testing.assert_array_equal(armed.points.data, [[2, 5, 6]])

    def test_a_4d_position_is_rejected(self, armed):
        self.click(armed, (0, 1, 2, 3))

        assert armed.viewer.status.startswith(
            "Unexpected coordinate dimensions"
        )
        assert len(armed.points.data) == 0
        assert armed.predictor.add_calls == []

    @pytest.mark.parametrize(
        "position", [(3, 1, 1), (0, 8, 1), (0, 1, -1), (-1, 1, 1)]
    )
    def test_a_click_outside_the_volume_is_undone(self, armed, position):
        self.click(armed, position)

        assert "out of bounds" in armed.viewer.status
        assert len(armed.points.data) == 0
        assert armed.predictor.add_calls == []

    def test_without_a_video_state_only_the_prompt_is_kept(self, armed):
        armed._sam2_state = None

        self.click(armed, (1, 2, 3))

        assert armed.points_data == {1: [[3, 2]]}
        assert armed.predictor.add_calls == []
        assert not armed.segmentation_result.any()
        assert "Propagation Progress" not in layer_names(armed.viewer)

    def test_low_resolution_masks_are_upscaled(self, armed):
        small = np.zeros((4, 4), dtype=bool)
        small[0, 0] = True
        armed.predictor.frame_mask = small
        armed.predictor.propagated = {2: small, 9: small}

        self.click(armed, (1, 0, 0))

        expected = square((8, 8), 0, 2, 0, 2)
        np.testing.assert_array_equal(
            armed.segmentation_result[1] == 1, expected
        )
        np.testing.assert_array_equal(
            armed.segmentation_result[2] == 1, expected
        )
        assert not armed.segmentation_result[0].any()

    def test_an_existing_progress_layer_is_reused(self, armed):
        armed.viewer.add_image(
            np.zeros((3, 8, 8)), name="Propagation Progress"
        )

        self.click(armed, (1, 2, 3))

        assert layer_names(armed.viewer).count("Propagation Progress") == 1

    def test_the_label_table_is_refreshed(self, armed):
        table = ca.QTableWidget()
        armed.label_table_widget = table

        self.click(armed, (1, 2, 3))

        assert table.rowCount() == 1

    def test_a_model_error_is_reported_not_raised(self, armed):
        def boom(**kwargs):
            raise ValueError("bad prompt")

        armed.predictor.add_new_points_or_box = boom

        self.click(armed, (1, 2, 3))

        assert armed.viewer.status == "Error in points handling: bad prompt"
        assert not armed.segmentation_result.any()


# --------------------------------------------------------------------------
# _on_points_clicked, 2D branches the core tests do not reach
# --------------------------------------------------------------------------
class TestOnPointsClicked2d:
    @pytest.fixture
    def armed(self, scene):
        scene.predictor = ImagePredictor([square((16, 16), 4, 8, 4, 8)])
        scene.points = FakePointsLayer(ndim=2)
        return scene

    def click(self, proc, position, **kwargs):
        proc._on_points_clicked(proc.points, FakeEvent(position, **kwargs))

    def test_a_3d_position_is_rejected(self, armed):
        self.click(armed, (1, 5, 5))

        assert armed.viewer.status.startswith(
            "Unexpected coordinate dimensions"
        )
        assert len(armed.points.data) == 0
        assert armed.predictor.predict_kwargs == []

    def test_without_an_image_nothing_is_segmented(self, armed):
        armed.current_image_for_segmentation = None

        self.click(armed, (5, 5))

        assert armed.viewer.status == "No image loaded for segmentation"
        assert armed.predictor.set_image_calls == []
        # The prompt itself is still remembered.
        assert armed.obj_points == {1: [[5, 5]]}

    def test_a_single_channel_image_is_expanded_to_rgb(self, armed):
        grey = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        armed.current_image_for_segmentation = grey

        self.click(armed, (5, 5))

        (sent,) = armed.predictor.set_image_calls
        assert sent.shape == (16, 16, 3)
        assert sent.dtype == np.uint8
        np.testing.assert_array_equal(sent[:, :, 2], grey[:, :, 0])

    def test_extra_channels_are_dropped_and_rescaled_to_uint8(self, armed):
        rgba = np.zeros((16, 16, 4), dtype=np.float32)
        rgba[..., 0] = 0.5
        rgba[..., 1] = 1.0
        rgba[..., 3] = 4.0  # alpha must not drive the scaling
        armed.current_image_for_segmentation = rgba

        self.click(armed, (5, 5))

        (sent,) = armed.predictor.set_image_calls
        assert sent.shape == (16, 16, 3)
        assert sent.dtype == np.uint8
        assert sent[0, 0].tolist() == [127, 255, 0]

    def test_a_low_resolution_mask_is_upscaled(self, armed):
        armed.predictor.masks = np.array([square((8, 8), 2, 4, 2, 4)])

        self.click(armed, (5, 5))

        np.testing.assert_array_equal(
            armed.segmentation_result == 1, square((16, 16), 4, 8, 4, 8)
        )
        assert armed.label_info[1]["area"] == 16

    def test_no_mask_means_no_label(self, armed):
        armed.predictor.masks = np.zeros((0, 16, 16), dtype=bool)
        armed.predictor.scores = np.zeros(0)

        self.click(armed, (5, 5))

        assert not armed.segmentation_result.any()
        assert 1 not in armed.label_info

    def test_the_label_table_is_refreshed(self, armed):
        table = ca.QTableWidget()
        armed.label_table_widget = table

        self.click(armed, (5, 5))

        assert table.rowCount() == 1


# --------------------------------------------------------------------------
# _add_segmentation_point, reset_sam2_state
# --------------------------------------------------------------------------
class TestAddSegmentationPoint:
    """Only the no-model path: see the module report for the model path."""

    def test_points_are_recorded_as_x_y_with_sam2_labels(self, scene):
        scene.predictor = None

        scene._add_segmentation_point(3, 7, FakeEvent((7, 3)))
        scene._add_segmentation_point(
            4, 8, FakeEvent((8, 4), modifiers=("Shift",))
        )

        assert scene.current_points == [[3, 7], [4, 8]]
        # SAM2 image predictors use 0 (not -1) for background points.
        assert scene.current_labels == [1, 0]
        assert scene.current_obj_id == 1
        assert not scene.segmentation_result.any()


class TestResetSam2State:
    @pytest.fixture
    def prepared(self, scene):
        scene.predictor = ImagePredictor(np.zeros((1, 16, 16), dtype=bool))
        scene.prepared_sam2_image = np.full((16, 16, 3), 0.5, np.float32)
        return scene

    def test_the_prepared_image_is_set_again(self, prepared):
        prepared.reset_sam2_state()

        (sent,) = prepared.predictor.set_image_calls
        assert sent is prepared.prepared_sam2_image

    def test_3d_mode_leaves_the_predictor_alone(self, prepared):
        prepared.use_3d = True

        prepared.reset_sam2_state()

        assert prepared.predictor.set_image_calls == []

    def test_nothing_to_reset_before_the_first_image(self, scene):
        scene.predictor = ImagePredictor(np.zeros((1, 16, 16), dtype=bool))

        scene.reset_sam2_state()

        assert scene.predictor.set_image_calls == []

    def test_a_video_predictor_is_skipped(self, prepared):
        prepared.predictor = VideoPredictor(np.zeros((16, 16)))

        prepared.reset_sam2_state()  # must not raise

        assert prepared.predictor.add_calls == []

    def test_a_failed_reset_reinitialises_the_model(
        self, prepared, monkeypatch
    ):
        reinitialised = []

        def broken(image):
            raise RuntimeError("stale state")

        prepared.predictor.set_image = broken
        monkeypatch.setattr(
            prepared, "_initialize_sam2", lambda: reinitialised.append(True)
        )

        prepared.reset_sam2_state()

        assert reinitialised == [True]


# --------------------------------------------------------------------------
# batch_crop_anything (magicgui entry point)
# --------------------------------------------------------------------------
class TestBatchCropAnythingEntry:
    @pytest.fixture
    def folder(self, tmp_path):
        tifffile.imwrite(
            tmp_path / "cells.tif", np.full((8, 8), 3, dtype=np.uint8)
        )
        tifffile.imwrite(
            tmp_path / "cells_labels_.tif", np.zeros((8, 8), dtype=np.uint32)
        )
        return tmp_path

    def test_declining_sam2_setup_opens_nothing(
        self, make_napari_viewer, no_sam2, folder, monkeypatch
    ):
        viewer = make_napari_viewer()
        created = []
        monkeypatch.setattr(ca, "check_or_create_sam2_env", lambda: False)
        monkeypatch.setattr(
            ca, "BatchCropAnything", lambda *a, **k: created.append(a)
        )

        ca.batch_crop_anything(
            folder_path=str(folder),
            data_dimensions="YX (2D)",
            viewer=viewer,
        )

        assert created == []
        assert "Crop Controls" not in viewer.window.dock_widgets
        assert len(viewer.layers) == 0

    @pytest.mark.parametrize(
        ("dimensions", "use_3d"),
        [("YX (2D)", False), ("TYX/ZYX (3D)", True)],
    )
    def test_the_folder_is_loaded_and_controls_are_docked(
        self,
        make_napari_viewer,
        no_sam2,
        torch_stub,
        folder,
        monkeypatch,
        dimensions,
        use_3d,
    ):
        viewer = make_napari_viewer()
        processors = []
        real_class = ca.BatchCropAnything

        def spy(*args, **kwargs):
            proc = real_class(*args, **kwargs)
            processors.append(proc)
            return proc

        monkeypatch.setattr(ca, "check_or_create_sam2_env", lambda: True)
        monkeypatch.setattr(ca, "BatchCropAnything", spy)

        ca.batch_crop_anything(
            folder_path=str(folder),
            data_dimensions=dimensions,
            viewer=viewer,
        )

        (proc,) = processors
        assert proc.use_3d is use_3d
        # The label file next to the image is filtered out.
        assert proc.images == [str(folder / "cells.tif")]
        assert "Crop Controls" in viewer.window.dock_widgets
        dock = viewer.window.dock_widgets["Crop Controls"]
        assert isinstance(dock, ca.QScrollArea)
        assert dock.widgetResizable()
        assert dock.minimumHeight() == 500

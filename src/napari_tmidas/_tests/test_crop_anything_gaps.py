"""
Batch Crop Anything: behaviour the other crop-anything suites leave out.

Covers the image preparation for odd channel layouts, the 3D box prompt
with unusual corner arrays and model outputs, label clean-up when objects
are deleted, the crop's failure modes, and the delayed removal of the
propagation overlay.  SAM2 is never loaded: the predictor and ``torch``
are replaced at the module boundary, as in the sibling suites.
"""

import os
import sys
import threading
import time
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


def make_torch_stub():
    stub = types.SimpleNamespace()
    stub.float32 = "float32"
    stub.inference_mode = lambda *args, **kwargs: NullContext()
    stub.autocast = lambda *args, **kwargs: NullContext()
    stub.cuda = types.SimpleNamespace(
        OutOfMemoryError=MemoryError, is_available=lambda: False
    )
    return stub


class ImagePredictor:
    """A 2D image predictor that records what it was handed."""

    def __init__(self, mask, score=0.9):
        self.mask = np.asarray(mask, dtype=bool)
        self.score = score
        self.set_image_calls = []
        self.predict_kwargs = []

    def set_image(self, image):
        self.set_image_calls.append(image)

    def predict(self, **kwargs):
        self.predict_kwargs.append(kwargs)
        return np.array([self.mask]), np.array([self.score]), None


class VideoPredictor:
    """A video predictor whose logits for the prompted frame are given."""

    def __init__(self, frame_logits, propagated=None):
        self.frame_logits = np.asarray(frame_logits, dtype=float)
        self.propagated = propagated or {}
        self.add_calls = []
        self.state = object()

    def add_new_points_or_box(self, **kwargs):
        self.add_calls.append(kwargs)
        return None, [kwargs["obj_id"]], FakeTensor(self.frame_logits)

    def propagate_in_video(self, state):
        for frame_idx, mask in self.propagated.items():
            yield frame_idx, [1], FakeTensor(
                np.where(mask, 5.0, -5.0)[np.newaxis, np.newaxis]
            )


class FakeEvent:
    def __init__(self, position, button=1, modifiers=()):
        self.position = position
        self.button = button
        self.modifiers = modifiers
        self.type = "mouse_press"


class FakePointsLayer:
    def __init__(self, ndim):
        self.data = np.zeros((0, ndim))
        self.face_color = "green"
        self.name = "Points (Click to Add)"
        self.mouse_drag_callbacks = []


def square(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def layer_names(viewer):
    return [layer.name for layer in viewer.layers]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def processor(make_napari_viewer, monkeypatch):
    def fake_init(self):
        self.predictor = None
        self.device = types.SimpleNamespace(type="cpu")

    monkeypatch.setattr(ca.BatchCropAnything, "_initialize_sam2", fake_init)
    monkeypatch.setattr(ca, "torch", make_torch_stub())
    return ca.BatchCropAnything(make_napari_viewer())


@pytest.fixture
def scheduled(monkeypatch):
    """Capture the delayed overlay-removal thread instead of starting it.

    The real thread sleeps two seconds and then touches the viewer from
    another thread; tests run the captured target inline instead.
    """
    targets = []

    class CapturedThread:
        def __init__(self, target=None, **kwargs):
            self.target = target

        def start(self):
            targets.append(self.target)

    monkeypatch.setattr(threading, "Thread", CapturedThread)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    return targets


def load_scene(proc, tmp_path, image=None, name="scene.tif"):
    if image is None:
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / name
    tifffile.imwrite(path, image)
    proc.use_3d = False
    proc.images = [str(path)]
    proc.current_index = 0
    proc.original_image = image
    proc.current_image_for_segmentation = image
    proc.segmentation_result = np.zeros(image.shape[:2], dtype=np.uint32)
    proc.label_info = {}
    proc.image_layer = proc.viewer.add_image(image, name="scene")
    proc.label_layer = proc.viewer.add_labels(
        proc.segmentation_result, name=f"Segmentation ({name})"
    )
    return proc


def load_volume(proc, tmp_path, name="vol.tif"):
    image = np.full((3, 8, 8), 7, dtype=np.uint8)
    path = tmp_path / name
    tifffile.imwrite(path, image, photometric="minisblack")
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


@pytest.fixture
def scene(processor, tmp_path):
    return load_scene(processor, tmp_path)


@pytest.fixture
def volume(processor, tmp_path):
    return load_volume(processor, tmp_path)


# --------------------------------------------------------------------------
# Segmentation entry point and image preparation
# --------------------------------------------------------------------------
class TestSegmentationEntryPoint:
    def test_without_a_model_the_image_is_kept_and_the_user_told(
        self, scene
    ):
        image = np.ones((16, 16), dtype=np.uint8)

        scene._generate_segmentation(image, scene.images[0])

        assert scene.current_image_for_segmentation is image
        assert scene.viewer.status == (
            "SAM2 model not initialized. Cannot segment images."
        )


class TestImagePreparationFor2d:
    """What the image predictor is handed for unusual channel layouts."""

    def prepare(self, proc, image):
        proc.current_image_for_segmentation = image
        proc.predictor = ImagePredictor(np.zeros(image.shape[:2]))
        proc._generate_2d_segmentation(0.7)
        (sent,) = proc.predictor.set_image_calls
        return sent

    def test_a_single_channel_image_is_broadcast_to_rgb(self, scene):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)

        sent = self.prepare(scene, image)

        assert sent.shape == (16, 16, 3)
        for c in range(3):
            np.testing.assert_allclose(
                sent[:, :, c], image[:, :, 0] / 255.0, atol=1e-6
            )
        assert scene.segmentation_result.shape == (16, 16)

    def test_extra_channels_are_dropped_keeping_the_first_three(
        self, scene
    ):
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        for c in range(4):
            image[:, :, c] = 50 * (c + 1)

        sent = self.prepare(scene, image)

        assert sent.shape == (16, 16, 3)
        np.testing.assert_allclose(
            sent[0, 0], [50 / 255, 100 / 255, 150 / 255], atol=1e-6
        )

    def test_a_float_image_already_in_unit_range_is_not_rescaled(
        self, scene
    ):
        image = np.full((16, 16), 0.25, dtype=np.float32)

        sent = self.prepare(scene, image)

        np.testing.assert_allclose(sent, 0.25)


class TestLoadErrorIn3d:
    def test_an_unreadable_stack_leaves_a_full_volume_error_layer(
        self, processor, tmp_path
    ):
        processor.use_3d = True
        processor.images = [str(tmp_path / "missing.tif")]
        processor.original_image = np.zeros((3, 8, 8), dtype=np.uint8)

        processor._load_current_image()

        assert "Error processing image" in processor.viewer.status
        assert processor.label_layer.name == "Error: No Segmentation"
        # In 3D mode the placeholder spans every frame, not one plane.
        assert processor.segmentation_result.shape == (3, 8, 8)


class TestShapesLayer:
    def test_an_existing_rectangles_layer_is_reused(self, scene):
        existing = scene.viewer.add_shapes(
            None, shape_type="rectangle", name="Rectangles (Draw to Segment)"
        )

        scene._ensure_shapes_layer()

        assert scene.shapes_layer is existing
        assert sum("Rectangles" in n for n in layer_names(scene.viewer)) == 1
        assert scene.viewer.layers.selection.active is existing


# --------------------------------------------------------------------------
# Box prompts
# --------------------------------------------------------------------------
class TestRectangleOnAVolume:
    """A box drawn on one slice of a stack, then propagated."""

    @pytest.fixture
    def boxed(self, volume, scheduled):
        mask = square((8, 8), 1, 4, 1, 4)
        volume.predictor = VideoPredictor(
            np.where(mask, 5.0, -5.0)[np.newaxis, np.newaxis],
            propagated={0: mask, 1: mask, 2: mask},
        )
        volume._sam2_state = volume.predictor.state
        return volume

    def test_corners_with_a_frame_column_use_only_y_and_x(self, boxed):
        corners = np.array(
            [[1, 1, 1], [1, 1, 3], [1, 3, 3], [1, 3, 1]], dtype=float
        )

        boxed._on_rectangle_added(corners)

        (call,) = boxed.predictor.add_calls
        np.testing.assert_array_equal(call["box"], [1, 1, 3, 3])
        assert int((boxed.segmentation_result == 1).sum()) == 27

    def test_corners_of_an_unknown_width_are_refused(self, boxed):
        boxed._on_rectangle_added(np.zeros((4, 4)))

        assert boxed.viewer.status == "Error: Unexpected rectangle format"
        assert boxed.predictor.add_calls == []
        assert not boxed.segmentation_result.any()

    def test_a_mask_with_a_leading_singleton_axis_is_squeezed(self, boxed):
        mask = square((8, 8), 2, 5, 2, 5)
        # (1 obj, 1, 1, H, W): one axis more than the usual logits.
        boxed.predictor.frame_logits = np.where(mask, 5.0, -5.0)[
            np.newaxis, np.newaxis, np.newaxis
        ]
        boxed.predictor.propagated = {}
        boxed.viewer.dims.set_current_step(0, 1)

        boxed._on_rectangle_added(
            np.array([[2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [4.0, 2.0]])
        )

        np.testing.assert_array_equal(
            boxed.segmentation_result[1] == 1, mask
        )

    def test_a_low_resolution_mask_is_scaled_to_the_frame(self, boxed):
        small = np.zeros((4, 4), dtype=bool)
        small[0, 0] = True
        boxed.predictor.frame_logits = np.where(small, 5.0, -5.0)[
            np.newaxis, np.newaxis
        ]
        boxed.predictor.propagated = {}
        boxed.viewer.dims.set_current_step(0, 0)

        boxed._on_rectangle_added(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        )

        np.testing.assert_array_equal(
            boxed.segmentation_result[0] == 1, square((8, 8), 0, 2, 0, 2)
        )

    def test_the_label_table_lists_the_new_object(self, boxed):
        table = boxed.create_label_table(None)
        assert table.rowCount() == 0

        boxed._on_rectangle_added(
            np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])
        )

        assert table.rowCount() == 1
        assert table.item(0, 1).text() == "1"


class TestRectangleOnAnImage:
    """A box drawn on a 2D image."""

    BOX = np.array([[2.0, 3.0], [2.0, 9.0], [7.0, 9.0], [7.0, 3.0]])

    def arm(self, proc, image=None):
        if image is not None:
            proc.current_image_for_segmentation = image
        proc.predictor = ImagePredictor(square((16, 16), 2, 8, 3, 10))
        return proc

    def test_without_an_image_nothing_is_segmented(self, scene):
        self.arm(scene)
        scene.current_image_for_segmentation = None

        scene._on_rectangle_added(self.BOX)

        assert scene.viewer.status == "No image loaded for segmentation"
        assert scene.predictor.predict_kwargs == []
        assert not scene.segmentation_result.any()

    def test_a_single_channel_image_is_sent_as_rgb(self, scene):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        self.arm(scene, image)

        scene._on_rectangle_added(self.BOX)

        (sent,) = scene.predictor.set_image_calls
        assert sent.shape == (16, 16, 3)
        np.testing.assert_array_equal(sent[:, :, 2], image[:, :, 0])

    def test_extra_channels_are_dropped(self, scene):
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        for c in range(4):
            image[:, :, c] = 10 * (c + 1)
        self.arm(scene, image)

        scene._on_rectangle_added(self.BOX)

        (sent,) = scene.predictor.set_image_calls
        np.testing.assert_array_equal(sent[0, 0], [10, 20, 30])

    def test_a_16_bit_image_is_stretched_to_8_bit(self, scene):
        image = np.zeros((16, 16), dtype=np.uint16)
        image[0, 0] = 4000
        image[0, 1] = 2000
        self.arm(scene, image)

        scene._on_rectangle_added(self.BOX)

        (sent,) = scene.predictor.set_image_calls
        assert sent.dtype == np.uint8
        assert sent[0, 0, 0] == 255
        assert sent[0, 1, 0] == 127

    def test_the_label_table_lists_the_new_object(self, scene):
        self.arm(scene)
        table = scene.create_label_table(None)

        scene._on_rectangle_added(self.BOX)

        assert table.rowCount() == 1
        assert table.item(0, 1).text() == "1"

    def test_a_model_failure_is_reported_not_raised(self, scene):
        self.arm(scene)

        def broken(**kwargs):
            raise RuntimeError("CUDA error: device-side assert")

        scene.predictor.predict = broken

        scene._on_rectangle_added(self.BOX)

        assert scene.viewer.status == (
            "Error in rectangle handling: CUDA error: device-side assert"
        )
        assert not scene.segmentation_result.any()


# --------------------------------------------------------------------------
# Clearing labels
# --------------------------------------------------------------------------
class TestClearAllLabels:
    def test_object_prompts_and_id_counters_are_reset(self, scene):
        scene.segmentation_result[0:2, 0:2] = 1
        scene.segmentation_result[5:7, 5:7] = 2
        scene.viewer.add_points(np.zeros((1, 2)), name="Points for Object 1")
        scene.viewer.add_points(np.zeros((1, 2)), name="Points for Object 2")
        scene.obj_points = {1: [[0, 0]]}
        scene.obj_labels = {1: [1]}
        scene.points_data = {2: [[5, 5]]}
        scene.points_labels = {2: [1]}
        scene.next_obj_id = 3
        scene._sam2_next_obj_id = 3

        scene.clear_selection()

        assert not scene.segmentation_result.any()
        assert not any(
            "Points for Object" in n for n in layer_names(scene.viewer)
        )
        assert scene.obj_points == {}
        assert scene.obj_labels == {}
        assert scene.points_data == {}
        assert scene.points_labels == {}
        assert scene.next_obj_id == 1
        assert scene._sam2_next_obj_id == 1
        assert scene.viewer.status == (
            "Cleared all 2 labels from segmentation"
        )


class TestClearOneLabel:
    def test_2d_delete_forgets_only_that_objects_prompts(self, scene):
        scene.segmentation_result[0:2, 0:2] = 1
        scene.segmentation_result[5:7, 5:7] = 2
        scene.obj_points = {1: [[0, 0]], 2: [[5, 5]]}
        scene.obj_labels = {1: [1], 2: [1]}

        scene.clear_label_at_position(0, 0)

        assert not (scene.segmentation_result == 1).any()
        assert (scene.segmentation_result == 2).sum() == 4
        assert scene.obj_points == {2: [[5, 5]]}
        assert scene.obj_labels == {2: [1]}
        assert scene.viewer.status == "Deleted label ID: 1"

    def test_3d_delete_drops_the_objects_prompts_and_points_layer(
        self, volume
    ):
        volume.segmentation_result[:, 0:2, 0:2] = 1
        volume.segmentation_result[1, 5:7, 5:7] = 2
        volume.viewer.add_points(np.zeros((1, 3)), name="Points for Object 1")
        volume.sam2_points_by_obj = {1: [[0, 0]], 2: [[5, 5]]}
        volume.sam2_labels_by_obj = {1: [1], 2: [1]}
        volume.points_data = {1: [[0, 0]], 2: [[5, 5]]}
        volume.points_labels = {1: [1], 2: [1]}
        table = volume.create_label_table(None)
        assert table.rowCount() == 2

        volume.clear_label_at_position_3d(2, 1, 1)

        assert not (volume.segmentation_result == 1).any()
        assert "Points for Object 1" not in layer_names(volume.viewer)
        assert volume.sam2_points_by_obj == {2: [[5, 5]]}
        assert volume.sam2_labels_by_obj == {2: [1]}
        assert volume.points_data == {2: [[5, 5]]}
        assert volume.points_labels == {2: [1]}
        assert table.rowCount() == 1
        assert table.item(0, 1).text() == "2"
        assert volume.viewer.status == (
            "Deleted label ID: 1 from all timeframes"
        )



# --------------------------------------------------------------------------
# Preview and crop
# --------------------------------------------------------------------------
class TestPreviewFailure:
    def test_a_segmentation_that_does_not_fit_the_image_is_reported(
        self, scene
    ):
        scene.segmentation_result = np.ones((4, 4), dtype=np.uint32)

        scene.preview_crop({1})

        assert scene.viewer.status.startswith("Error generating preview:")
        assert not any("Preview" in n for n in layer_names(scene.viewer))


class TestCrop:
    def test_a_colour_image_is_blanked_in_every_channel(
        self, processor, tmp_path
    ):
        rgb = np.full((16, 16, 3), 200, dtype=np.uint8)
        load_scene(processor, tmp_path, image=rgb)
        processor.segmentation_result[2:5, 2:5] = 1
        processor.selected_labels = {1}

        assert processor.crop_with_selected_labels() is True

        cropped = tifffile.imread(tmp_path / "scene_sam2_cropped.tif")
        assert cropped.shape == (16, 16, 3)
        np.testing.assert_array_equal(cropped[3, 3], [200, 200, 200])
        np.testing.assert_array_equal(cropped[0, 0], [0, 0, 0])
        assert int((cropped.any(axis=-1)).sum()) == 9

    def test_an_unwritable_destination_reports_and_returns_false(
        self, scene, tmp_path
    ):
        scene.segmentation_result[0:2, 0:2] = 1
        scene.selected_labels = {1}
        scene.images = [str(tmp_path / "gone" / "scene.tif")]

        assert scene.crop_with_selected_labels() is False

        assert scene.viewer.status.startswith("Error cropping image:")

    def test_a_broken_workflow_log_does_not_stop_the_crop(
        self, scene, tmp_path, monkeypatch, capsys
    ):
        class BrokenLog:
            def __init__(self, folder):
                raise PermissionError("read-only share")

        monkeypatch.setattr(ca, "WorkflowLog", BrokenLog)
        scene.segmentation_result[0:2, 0:2] = 1
        scene.selected_labels = {1}

        assert scene.crop_with_selected_labels() is True

        assert (tmp_path / "scene_sam2_cropped.tif").exists()
        assert (tmp_path / "scene_sam2_labels.tif").exists()
        out = capsys.readouterr().out
        assert "Workflow log: could not record crop: read-only share" in out


# --------------------------------------------------------------------------
# Propagation overlay clean-up
# --------------------------------------------------------------------------
class TestPropagationOverlayCleanup:
    """The "Propagation Progress" overlay is removed once the run is done."""

    def test_after_propagating_an_object(self, volume, scheduled):
        mask = square((8, 8), 1, 3, 1, 3)
        volume.predictor = VideoPredictor(
            np.zeros((1, 1, 8, 8)), propagated={0: mask, 1: mask}
        )
        volume._sam2_state = volume.predictor.state

        volume._propagate_mask_for_current_object(1, 0)
        assert "Propagation Progress" in layer_names(volume.viewer)

        (cleanup,) = scheduled
        cleanup()

        assert "Propagation Progress" not in layer_names(volume.viewer)
        # Only the overlay goes; the result layers stay.
        assert volume.label_layer in volume.viewer.layers

    def test_after_a_point_prompt_on_a_stack(self, volume, scheduled):
        mask = square((8, 8), 1, 4, 1, 4)
        volume.predictor = VideoPredictor(
            np.where(mask, 5.0, -5.0)[np.newaxis, np.newaxis],
            propagated={0: mask, 1: mask, 2: mask},
        )
        volume._sam2_state = volume.predictor.state
        points = FakePointsLayer(ndim=3)

        volume._on_points_clicked(points, FakeEvent((1, 2, 3)))
        assert "Propagation Progress" in layer_names(volume.viewer)

        (cleanup,) = scheduled
        cleanup()

        assert "Propagation Progress" not in layer_names(volume.viewer)
        assert int((volume.segmentation_result == 1).sum()) == 27


# --------------------------------------------------------------------------
# Control panel
# --------------------------------------------------------------------------
class TestClearPromptsButton:
    def test_a_detached_click_handler_is_reconnected(self, scene):
        widget = ca.create_crop_widget(scene)
        # A prompt layer whose handler was lost (e.g. the user re-created
        # the layer by hand).
        points = scene.viewer.add_points(
            np.array([[1.0, 1.0], [2.0, 2.0]]), name="Points (Click to Add)"
        )
        assert scene._on_points_clicked not in points.mouse_drag_callbacks

        next(
            b
            for b in widget.findChildren(ca.QPushButton)
            if b.text() == "Clear Prompts"
        ).click()

        assert len(points.data) == 0
        assert scene._on_points_clicked in points.mouse_drag_callbacks

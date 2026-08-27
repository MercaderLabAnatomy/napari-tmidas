"""
Tests for the non-SAM2 half of Batch Crop Anything (``_crop_anything``).

The model inference needs a GPU and a downloaded checkpoint, but everything
around it does not: which files get picked up as images, how navigation
moves through them, which voxels a selection covers, what the preview
masks out, and — the part that reaches disk — what the crop actually
writes.  These drive that logic against a real napari viewer with the
model initialisation stubbed out.
"""

import os
import sys
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


@pytest.fixture
def no_sam2(monkeypatch):
    """Skip model initialisation; every test here works without SAM2."""
    monkeypatch.setattr(
        ca.BatchCropAnything, "_initialize_sam2", lambda self: None
    )


@pytest.fixture
def processor(make_napari_viewer, no_sam2):
    return ca.BatchCropAnything(make_napari_viewer())


def make_scene(tmp_path, use_3d=False):
    """An image on disk plus the segmentation the processor would hold."""
    image = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16) % 251 + 1
    segmentation = np.zeros((16, 16), dtype=np.uint32)
    segmentation[0:4, 0:4] = 1
    segmentation[8:12, 8:12] = 2
    path = tmp_path / "scene.tif"
    tifffile.imwrite(path, image)
    return str(path), image, segmentation


@pytest.fixture
def loaded(processor, tmp_path):
    path, image, segmentation = make_scene(tmp_path)
    processor.images = [path]
    processor.current_index = 0
    processor.original_image = image
    processor.segmentation_result = segmentation
    processor.label_info = {
        1: {"size": 16, "bbox": (0, 0, 4, 4)},
        2: {"size": 16, "bbox": (8, 8, 12, 12)},
    }
    processor.image_layer = processor.viewer.add_image(image, name="scene")
    processor.label_layer = processor.viewer.add_labels(
        segmentation, name="Segmentation (scene.tif)"
    )
    return processor


class TestGetDevice:
    """torch is an optional dependency, so drive the branches with a stub."""

    def _torch(self, cuda=False, mps=False):
        stub = types.SimpleNamespace()
        stub.device = lambda name: f"device:{name}"
        stub.cuda = types.SimpleNamespace(
            is_available=lambda: cuda,
            get_device_name=lambda: "Stub GPU",
        )
        stub.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        )
        return stub

    def test_prefers_cuda_on_linux(self, monkeypatch):
        monkeypatch.setattr(ca.sys, "platform", "linux")
        monkeypatch.setattr(ca, "torch", self._torch(cuda=True))
        assert ca.get_device() == "device:cuda"

    def test_falls_back_to_cpu_on_linux(self, monkeypatch):
        monkeypatch.setattr(ca.sys, "platform", "linux")
        monkeypatch.setattr(ca, "torch", self._torch(cuda=False))
        assert ca.get_device() == "device:cpu"

    def test_prefers_mps_on_macos(self, monkeypatch):
        monkeypatch.setattr(ca.sys, "platform", "darwin")
        monkeypatch.setattr(ca, "torch", self._torch(mps=True))
        assert ca.get_device() == "device:mps"

    def test_macos_without_mps_uses_cpu(self, monkeypatch):
        monkeypatch.setattr(ca.sys, "platform", "darwin")
        monkeypatch.setattr(ca, "torch", self._torch(mps=False))
        assert ca.get_device() == "device:cpu"

    def test_macos_ignores_cuda(self, monkeypatch):
        # A CUDA build on darwin must still not be selected.
        monkeypatch.setattr(ca.sys, "platform", "darwin")
        monkeypatch.setattr(ca, "torch", self._torch(cuda=True, mps=False))
        assert ca.get_device() == "device:cpu"


class TestInitialState:
    def test_starts_empty(self, processor):
        assert processor.images == []
        assert processor.current_index == 0
        assert processor.selected_labels == set()
        assert processor.label_info == {}
        assert processor.segmentation_result is None
        assert processor.prompt_mode == "point"
        assert processor.use_3d is False

    def test_three_d_mode_is_recorded(self, make_napari_viewer, no_sam2):
        processor = ca.BatchCropAnything(make_napari_viewer(), use_3d=True)
        assert processor.use_3d is True


class TestLoadImages:
    @pytest.fixture(autouse=True)
    def _no_segmentation(self, monkeypatch):
        monkeypatch.setattr(
            ca.BatchCropAnything, "_load_current_image", lambda self: None
        )

    def _touch(self, folder, *names):
        for name in names:
            (folder / name).write_bytes(b"")

    def test_picks_up_tif_and_tiff(self, processor, tmp_path):
        self._touch(tmp_path, "a.tif", "b.tiff", "c.TIF")
        processor.load_images(str(tmp_path))
        assert sorted(os.path.basename(p) for p in processor.images) == [
            "a.tif",
            "b.tiff",
            "c.TIF",
        ]

    def test_ignores_other_extensions(self, processor, tmp_path):
        self._touch(tmp_path, "a.tif", "notes.txt", "b.png")
        processor.load_images(str(tmp_path))
        assert [os.path.basename(p) for p in processor.images] == ["a.tif"]

    @pytest.mark.parametrize(
        "name", ["cells_labels.tif", "a_labels_1.tif", "x_sam2_cropped.tif"]
    )
    def test_excludes_label_and_sam2_outputs(self, processor, tmp_path, name):
        # The tool writes its own results next to the inputs, so re-scanning
        # a folder must not pick them back up.
        self._touch(tmp_path, "keep.tif", name)
        processor.load_images(str(tmp_path))
        assert [os.path.basename(p) for p in processor.images] == ["keep.tif"]

    def test_picks_up_zarr_directories(self, processor, tmp_path):
        (tmp_path / "volume.zarr").mkdir()
        processor.load_images(str(tmp_path))
        assert [os.path.basename(p) for p in processor.images] == [
            "volume.zarr"
        ]

    def test_ignores_plain_directories(self, processor, tmp_path):
        (tmp_path / "subfolder").mkdir()
        processor.load_images(str(tmp_path))
        assert processor.images == []

    def test_missing_folder_is_reported(self, processor, tmp_path):
        processor.load_images(str(tmp_path / "nope"))
        assert processor.images == []
        assert "not found" in processor.viewer.status.lower()

    def test_empty_folder_is_reported(self, processor, tmp_path):
        processor.load_images(str(tmp_path))
        assert "no compatible images" in processor.viewer.status.lower()

    def test_resets_the_index(self, processor, tmp_path):
        self._touch(tmp_path, "a.tif")
        processor.current_index = 5
        processor.load_images(str(tmp_path))
        assert processor.current_index == 0


class TestNavigation:
    @pytest.fixture
    def navigable(self, processor, monkeypatch):
        monkeypatch.setattr(
            ca.BatchCropAnything, "_load_current_image", lambda self: None
        )
        processor.images = ["/a/one.tif", "/a/two.tif", "/a/three.tif"]
        return processor

    def test_next_advances(self, navigable):
        assert navigable.next_image() is True
        assert navigable.current_index == 1

    def test_next_stops_at_the_end(self, navigable):
        navigable.current_index = 2
        assert navigable.next_image() is False
        assert navigable.current_index == 2
        assert "no more images" in navigable.viewer.status.lower()

    def test_previous_goes_back(self, navigable):
        navigable.current_index = 2
        assert navigable.previous_image() is True
        assert navigable.current_index == 1

    def test_previous_stops_at_the_start(self, navigable):
        assert navigable.previous_image() is False
        assert navigable.current_index == 0
        assert "first image" in navigable.viewer.status.lower()

    @pytest.mark.parametrize("step", ["next_image", "previous_image"])
    def test_navigation_without_images_reports(self, navigable, step):
        navigable.images = []
        assert getattr(navigable, step)() is False
        assert "no images" in navigable.viewer.status.lower()

    @pytest.mark.parametrize("step", ["next_image", "previous_image"])
    def test_moving_clears_the_selection(self, navigable, step):
        navigable.current_index = 1
        navigable.selected_labels = {1, 2}
        navigable.label_table_widget = object()

        getattr(navigable, step)()

        assert navigable.selected_labels == set()
        assert navigable.label_table_widget is None


class TestSelectedLabelsMask:
    def test_single_label(self, loaded):
        mask = loaded._selected_labels_mask([1])
        assert mask.sum() == 16
        assert mask[0, 0] and not mask[8, 8]

    def test_several_labels(self, loaded):
        assert loaded._selected_labels_mask([1, 2]).sum() == 32

    def test_empty_selection_selects_nothing(self, loaded):
        assert loaded._selected_labels_mask([]).sum() == 0

    def test_unknown_label_selects_nothing(self, loaded):
        assert loaded._selected_labels_mask([99]).sum() == 0

    def test_accepts_a_set(self, loaded):
        assert loaded._selected_labels_mask({2}).sum() == 16


class TestSelectAllAndClear:
    def test_select_all_takes_every_known_label(self, loaded):
        loaded.select_all_labels()
        assert loaded.selected_labels == {1, 2}

    def test_select_all_without_labels_is_a_no_op(self, processor):
        processor.label_info = {}
        processor.select_all_labels()
        assert processor.selected_labels == set()

    def test_clear_selection_wipes_the_segmentation(self, loaded):
        loaded.selected_labels = {1, 2}
        loaded.clear_selection()

        assert not loaded.segmentation_result.any()
        assert loaded.selected_labels == set()
        assert loaded.label_info == {}

    def test_clear_selection_without_segmentation_is_reported(self, processor):
        processor.segmentation_result = None
        processor.clear_selection()
        assert "no segmentation" in processor.viewer.status.lower()

    def test_clear_selection_with_nothing_to_clear(self, loaded):
        loaded.segmentation_result[:] = 0
        loaded.clear_selection()
        assert "no labels to clear" in loaded.viewer.status.lower()


class TestClearLabelAtPosition:
    def test_removes_the_whole_label(self, loaded):
        loaded.selected_labels = {1, 2}
        loaded.clear_label_at_position(1, 1)

        assert not (loaded.segmentation_result == 1).any()
        assert (loaded.segmentation_result == 2).sum() == 16
        assert loaded.selected_labels == {2}
        assert set(loaded.label_info) == {2}
        assert "deleted label id: 1" in loaded.viewer.status.lower()

    def test_clicking_background_does_nothing(self, loaded):
        loaded.clear_label_at_position(6, 6)
        assert (loaded.segmentation_result > 0).sum() == 32
        assert "no label to delete" in loaded.viewer.status.lower()

    def test_without_segmentation_is_reported(self, processor):
        processor.segmentation_result = None
        processor.clear_label_at_position(0, 0)
        assert "no segmentation" in processor.viewer.status.lower()

    def test_removes_the_objects_point_layer(self, loaded):
        loaded.viewer.add_points(
            np.array([[1.0, 1.0]]), name="Points for Object 1"
        )
        loaded.clear_label_at_position(1, 1)
        assert not any(
            "Points for Object 1" in layer.name
            for layer in loaded.viewer.layers
        )


class TestClearLabelAtPosition3d:
    @pytest.fixture
    def volume(self, processor, tmp_path):
        segmentation = np.zeros((3, 8, 8), dtype=np.uint32)
        segmentation[:, 0:2, 0:2] = 1  # present in every timeframe
        segmentation[1, 5:7, 5:7] = 2
        image = np.zeros((3, 8, 8), dtype=np.uint8)
        path = tmp_path / "vol.tif"
        tifffile.imwrite(path, image)

        processor.use_3d = True
        processor.images = [str(path)]
        processor.original_image = image
        processor.segmentation_result = segmentation
        processor.label_info = {1: {"size": 12}, 2: {"size": 4}}
        processor.image_layer = processor.viewer.add_image(image)
        processor.label_layer = processor.viewer.add_labels(
            segmentation, name="Segmentation (vol.tif)"
        )
        return processor

    def test_removes_the_label_from_every_timeframe(self, volume):
        volume.selected_labels = {1}
        volume.clear_label_at_position_3d(0, 0, 0)

        assert not (volume.segmentation_result == 1).any()
        assert volume.selected_labels == set()

    def test_leaves_other_labels_alone(self, volume):
        volume.clear_label_at_position_3d(0, 0, 0)
        assert (volume.segmentation_result == 2).sum() == 4

    def test_background_click_is_reported(self, volume):
        volume.clear_label_at_position_3d(0, 4, 4)
        assert "no label to delete" in volume.viewer.status.lower()

    def test_without_segmentation_is_reported(self, processor):
        processor.segmentation_result = None
        processor.clear_label_at_position_3d(0, 0, 0)
        assert "no segmentation" in processor.viewer.status.lower()


class TestPreviewCrop:
    def _preview(self, processor):
        return [
            layer
            for layer in processor.viewer.layers
            if "Preview" in layer.name
        ]

    def test_adds_a_preview_layer_for_the_selection(self, loaded):
        loaded.preview_crop([1])

        previews = self._preview(loaded)
        assert len(previews) == 1
        assert "1" in previews[0].name
        # Everything outside label 1 is blanked.
        assert previews[0].data[8, 8] == 0
        assert previews[0].data[0, 0] == loaded.original_image[0, 0]

    def test_uses_the_current_selection_by_default(self, loaded):
        loaded.selected_labels = {2}
        loaded.preview_crop()

        preview = self._preview(loaded)[0]
        assert preview.data[8, 8] == loaded.original_image[8, 8]
        assert preview.data[0, 0] == 0

    def test_replaces_an_earlier_preview(self, loaded):
        loaded.preview_crop([1])
        loaded.preview_crop([2])
        assert len(self._preview(loaded)) == 1

    def test_empty_selection_removes_the_preview(self, loaded):
        loaded.preview_crop([1])
        loaded.preview_crop([])
        assert self._preview(loaded) == []

    def test_does_not_modify_the_original(self, loaded):
        before = loaded.original_image.copy()
        loaded.preview_crop([1])
        np.testing.assert_array_equal(loaded.original_image, before)

    def test_colour_images_are_masked_across_channels(
        self, processor, tmp_path
    ):
        image = np.full((8, 8, 3), 200, dtype=np.uint8)
        segmentation = np.zeros((8, 8), dtype=np.uint32)
        segmentation[0:2, 0:2] = 1
        path = tmp_path / "rgb.tif"
        tifffile.imwrite(path, image)

        processor.images = [str(path)]
        processor.original_image = image
        processor.segmentation_result = segmentation
        processor.image_layer = processor.viewer.add_image(image, rgb=True)

        processor.preview_crop([1])

        preview = self._preview(processor)[0]
        assert preview.data.shape == (8, 8, 3)
        np.testing.assert_array_equal(preview.data[0, 0], [200, 200, 200])
        np.testing.assert_array_equal(preview.data[5, 5], [0, 0, 0])

    def test_without_segmentation_is_reported(self, processor):
        processor.segmentation_result = None
        processor.preview_crop([1])
        assert "no image or segmentation" in processor.viewer.status.lower()


class TestCropWithSelectedLabels:
    def test_writes_a_cropped_image_and_a_label_mask(self, loaded, tmp_path):
        loaded.selected_labels = {1}

        assert loaded.crop_with_selected_labels() is True

        cropped = tifffile.imread(tmp_path / "scene_sam2_cropped.tif")
        labels = tifffile.imread(tmp_path / "scene_sam2_labels.tif")

        assert cropped.shape == loaded.original_image.shape
        np.testing.assert_array_equal(
            cropped[0:4, 0:4], loaded.original_image[0:4, 0:4]
        )
        assert cropped[8, 8] == 0

        assert labels.dtype == np.uint32
        assert set(np.unique(labels)) == {0, 1}
        assert (labels == 1).sum() == 16

    def test_keeps_every_selected_label(self, loaded, tmp_path):
        loaded.selected_labels = {1, 2}
        loaded.crop_with_selected_labels()

        labels = tifffile.imread(tmp_path / "scene_sam2_labels.tif")
        assert set(np.unique(labels)) == {0, 1, 2}

    def test_does_not_modify_the_original(self, loaded):
        before = loaded.original_image.copy()
        loaded.selected_labels = {1}
        loaded.crop_with_selected_labels()
        np.testing.assert_array_equal(loaded.original_image, before)

    def test_refuses_without_a_selection(self, loaded, tmp_path):
        loaded.selected_labels = set()
        assert loaded.crop_with_selected_labels() is False
        assert not (tmp_path / "scene_sam2_cropped.tif").exists()
        assert "no labels selected" in loaded.viewer.status.lower()

    def test_refuses_without_a_segmentation(self, processor):
        processor.segmentation_result = None
        processor.original_image = np.zeros((4, 4), np.uint8)
        assert processor.crop_with_selected_labels() is False
        assert "no image or segmentation" in processor.viewer.status.lower()

    def test_output_is_named_after_the_input(self, processor, tmp_path):
        image = np.ones((4, 4), dtype=np.uint8)
        segmentation = np.ones((4, 4), dtype=np.uint32)
        path = tmp_path / "embryo_07.tif"
        tifffile.imwrite(path, image)

        processor.images = [str(path)]
        processor.original_image = image
        processor.segmentation_result = segmentation
        processor.selected_labels = {1}

        assert processor.crop_with_selected_labels() is True
        assert (tmp_path / "embryo_07_sam2_cropped.tif").exists()
        assert (tmp_path / "embryo_07_sam2_labels.tif").exists()

    def test_three_d_volume_is_cropped_across_frames(
        self, processor, tmp_path
    ):
        image = np.full((3, 6, 6), 9, dtype=np.uint8)
        segmentation = np.zeros((3, 6, 6), dtype=np.uint32)
        segmentation[:, 0:2, 0:2] = 1
        path = tmp_path / "vol.tif"
        tifffile.imwrite(path, image)

        processor.use_3d = True
        processor.images = [str(path)]
        processor.original_image = image
        processor.segmentation_result = segmentation
        processor.selected_labels = {1}

        assert processor.crop_with_selected_labels() is True

        cropped = tifffile.imread(tmp_path / "vol_sam2_cropped.tif")
        assert cropped.shape == (3, 6, 6)
        assert (cropped == 9).sum() == 12  # 2x2 in each of 3 frames
        assert cropped[0, 5, 5] == 0


class TestLayerCallbacks:
    """
    napari installs its own mouse handlers on a Labels layer (the polygon
    overlay).  Clearing a layer's callback list to avoid duplicate
    handlers takes those with it, and removing the layer afterwards raises
    ``ValueError: list.remove(x): x not in list`` out of
    ``VispyLabelsPolygonOverlay.close()`` -- which is every "next image"
    click, since that replaces the segmentation layer.
    """

    def test_napari_keeps_its_own_handlers(self, loaded):
        loaded._update_label_layer()
        callbacks = loaded.label_layer.mouse_drag_callbacks

        assert loaded._on_label_clicked in callbacks
        assert len(callbacks) > 1, "napari's own handler was dropped"

    def test_our_handler_is_installed_once(self, loaded):
        loaded._update_label_layer()
        loaded._update_label_layer()

        assert (
            loaded.label_layer.mouse_drag_callbacks.count(
                loaded._on_label_clicked
            )
            == 1
        )

    def test_the_label_layer_can_be_removed(self, loaded):
        loaded._update_label_layer()
        loaded.viewer.layers.remove(loaded.label_layer)  # must not raise

    def test_replacing_the_layer_repeatedly_is_safe(self, loaded):
        for _ in range(3):
            loaded._update_label_layer()
        assert (
            len(
                [
                    layer
                    for layer in loaded.viewer.layers
                    if "Segmentation" in layer.name
                ]
            )
            == 1
        )

    def test_preview_layers_can_be_replaced(self, loaded):
        loaded.preview_crop([1])
        loaded.preview_crop([2])  # removes the first preview; must not raise
        loaded.preview_crop([])


class TestSam2EnvCheck:
    def test_returns_true_when_the_env_is_ready(self, monkeypatch):
        monkeypatch.setattr(ca.sam2_manager, "is_env_created", lambda: True)
        monkeypatch.setattr(
            ca.sam2_manager, "is_package_installed", lambda: True
        )
        assert ca.check_or_create_sam2_env() is True


def test_widget_factory_is_callable():
    assert callable(ca.batch_crop_anything_widget)

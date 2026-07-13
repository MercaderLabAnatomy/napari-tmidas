# src/napari_tmidas/_tests/test_label_inspection.py
import contextlib
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from napari_tmidas._label_inspection import (
    LabelInspector,
    label_inspector_widget,
)


class TestLabelInspector:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.viewer = Mock()

    def teardown_method(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_label_inspector_initialization(self):
        """Test LabelInspector initialization"""
        inspector = LabelInspector(self.viewer)
        assert inspector.viewer == self.viewer
        assert inspector.image_label_pairs == []
        assert inspector.current_index == 0

    def test_load_image_label_pairs_no_folder(self):
        """Test loading pairs with non-existent folder"""
        inspector = LabelInspector(self.viewer)
        inspector.load_image_label_pairs("/nonexistent/folder", "_labels")
        assert (
            self.viewer.status
            == "Folder path does not exist: /nonexistent/folder"
        )

    def test_load_image_label_pairs_no_labels(self):
        """Test loading pairs with no label files"""
        inspector = LabelInspector(self.viewer)

        # Create empty folder
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)

        inspector.load_image_label_pairs(empty_dir, "_labels")
        assert self.viewer.status == "No files found with suffix '_labels'"

    @patch("napari_tmidas._label_inspection.imread")
    def test_load_image_label_pairs_valid(self, mock_imread):
        """Test loading valid image-label pairs"""
        inspector = LabelInspector(self.viewer)

        # Create test files
        test_dir = os.path.join(self.temp_dir, "test")
        os.makedirs(test_dir)

        # Create image and label files
        image_path = os.path.join(test_dir, "test_image.tif")
        label_path = os.path.join(test_dir, "test_image_labels.tif")

        with open(image_path, "w") as f:
            f.write("dummy")
        with open(label_path, "w") as f:
            f.write("dummy")

        # Mock imread to return valid label data
        mock_imread.return_value = np.ones((10, 10), dtype=np.uint32)

        inspector.load_image_label_pairs(test_dir, "_labels")

        # Check that pairs were loaded
        assert len(inspector.image_label_pairs) == 1
        assert inspector.image_label_pairs[0] == (image_path, label_path)

    @patch("napari_tmidas._label_inspection.imread")
    @patch("napari_tmidas._label_inspection._load_image")
    def test_load_image_label_pairs_zarr_image(
        self, mock_load_image, mock_imread
    ):
        """Zarr directory is matched as raw image for a tif label file."""
        inspector = LabelInspector(self.viewer)

        test_dir = os.path.join(self.temp_dir, "zarr_test")
        os.makedirs(test_dir)

        # Raw image is a .zarr directory; label is a .tif file
        zarr_dir = os.path.join(test_dir, "test_image.zarr")
        os.makedirs(zarr_dir)
        label_path = os.path.join(test_dir, "test_image_labels.tif")
        with open(label_path, "w") as f:
            f.write("dummy")

        mock_imread.return_value = np.ones((10, 10), dtype=np.uint32)
        mock_load_image.return_value = np.ones((10, 10), dtype=np.uint8)

        inspector.load_image_label_pairs(test_dir, "_labels")

        assert len(inspector.image_label_pairs) == 1
        assert inspector.image_label_pairs[0] == (zarr_dir, label_path)

    def test_load_image_zarr_helper(self):
        """_load_image delegates to load_zarr_basic for .zarr paths."""
        from unittest.mock import patch as _patch

        from napari_tmidas._label_inspection import _load_image

        fake_data = np.zeros((5, 5), dtype=np.uint16)

        with _patch(
            "napari_tmidas._file_selector.load_zarr_basic",
            return_value=fake_data,
        ) as mock_fn:
            result = _load_image("/some/path.zarr")

        mock_fn.assert_called_once_with("/some/path.zarr")
        assert result is fake_data

    @patch("napari_tmidas._label_inspection.imread")
    def test_scale_applied_to_labels(self, mock_imread):
        """Labels get a non-unit scale when image and label shapes differ."""
        inspector = LabelInspector(self.viewer)

        test_dir = os.path.join(self.temp_dir, "scale_test")
        os.makedirs(test_dir)

        image_path = os.path.join(test_dir, "test_image.tif")
        label_path = os.path.join(test_dir, "test_image_labels.tif")
        with open(image_path, "w") as f:
            f.write("dummy")
        with open(label_path, "w") as f:
            f.write("dummy")

        # imread returns different shapes for image vs label
        def side_effect(path):
            if "labels" in path:
                return np.ones((5, 5), dtype=np.uint32)
            return np.ones((10, 10), dtype=np.uint8)

        mock_imread.side_effect = side_effect

        inspector.load_image_label_pairs(test_dir, "_labels")

        # Capture the scale passed to add_labels
        calls = self.viewer.add_labels.call_args_list
        assert calls, "add_labels was never called"
        _, kwargs = calls[0]
        assert kwargs.get("scale") == [2.0, 2.0]

    def test_delete_label_all_t_preserves_unsaved_edits(self):
        """Deleting a label across T must keep pending manual edits."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        class _FakeLabels:
            def __init__(self, data, selected_label):
                self.data = data
                self.selected_label = selected_label

            def refresh(self):
                pass

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5  # label to delete, present at every T
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))

        # Unsaved paint edit at T=1 (napari-style fancy assignment).
        wrapper[np.array([1, 1]), np.array([2, 2]), np.array([2, 3])] = 7

        layer = _FakeLabels(wrapper, selected_label=5)
        self.viewer.layers = [layer]

        inspector = LabelInspector(self.viewer)
        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            inspector.delete_label_all_timepoints()

        result = np.asarray(layer.data)
        # Label 5 is gone everywhere...
        assert not np.any(result == 5)
        # ...and the unsaved paint edit at T=1 survived.
        assert result[1, 2, 2] == 7 and result[1, 2, 3] == 7

    def test_remap_values_composes_and_remaps_pending_edits(self):
        """Consecutive remaps collapse into one LUT and apply to diffs."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        base = np.zeros((2, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        base[:, 1, 1] = 7
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))

        # Paint label 9 at T=0, then merge 7→3 and delete 5 globally.
        wrapper[np.array([0]), np.array([2]), np.array([2])] = 9
        wrapper.remap_values({7: 3})
        wrapper.remap_values({5: 0})
        # Two remaps must collapse into a single LUT (one pass at save).
        assert wrapper._lut == {7: 3, 5: 0}

        result = np.asarray(wrapper)
        assert not np.any(result == 5)
        assert not np.any(result == 7)
        assert result[0, 1, 1] == 3 and result[1, 1, 1] == 3
        assert result[0, 2, 2] == 9  # painted edit survived both remaps

        # A remap must also rewrite painted values (delete the painted 9).
        wrapper.remap_values({9: 0})
        assert np.asarray(wrapper)[0, 2, 2] == 0

    def test_click_to_delete_mode(self):
        """Toggle registers/removes the callback; click deletes, drag doesn't."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 1

            def refresh(self):
                pass

            def get_value(self, position, **kwargs):
                return 5

        class _FakeEvent:
            def __init__(self):
                self.button = 1
                self.type = "mouse_press"
                self.position = (0, 0, 0)
                self.view_direction = None
                self.dims_displayed = [1, 2]

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))
        layer = _FakeLabels(wrapper)

        self.viewer.layers = [layer]
        self.viewer.mouse_drag_callbacks = []

        inspector = LabelInspector(self.viewer)
        inspector.enable_click_delete(True)
        assert len(self.viewer.mouse_drag_callbacks) == 1

        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            # Drag (pan): callback must NOT delete.
            event = _FakeEvent()
            gen = self.viewer.mouse_drag_callbacks[0](self.viewer, event)
            next(gen)
            event.type = "mouse_move"
            next(gen)
            event.type = "mouse_release"
            with contextlib.suppress(StopIteration):
                next(gen)
            assert np.any(np.asarray(layer.data) == 5)

            # Clean click: deletes label 5 from all T.
            event = _FakeEvent()
            gen = self.viewer.mouse_drag_callbacks[0](self.viewer, event)
            next(gen)
            event.type = "mouse_release"
            with contextlib.suppress(StopIteration):
                next(gen)
            assert not np.any(np.asarray(layer.data) == 5)

        inspector.enable_click_delete(False)
        assert len(self.viewer.mouse_drag_callbacks) == 0

    def test_undo_remap(self):
        """Ctrl+Z semantics: undo restores the last deletion exactly."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        base[:, 1, 1] = 7
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))

        # Paint a 5 at T=0, then delete 5 and 7 globally.
        wrapper[np.array([0]), np.array([2]), np.array([2])] = 5
        wrapper.remap_values({5: 0})
        wrapper.remap_values({7: 0})

        # Undo last: 7 restored, 5 still deleted.
        assert wrapper.undo_remap() == {7: 0}
        result = np.asarray(wrapper)
        assert np.all(result[:, 1, 1] == 7)
        assert not np.any(result == 5)

        # Undo again: 5 restored everywhere, including the painted voxel.
        assert wrapper.undo_remap() == {5: 0}
        result = np.asarray(wrapper)
        assert np.all(result[:, 0, 0] == 5)
        assert result[0, 2, 2] == 5
        assert np.all(result[:, 1, 1] == 7)

        # Nothing left to undo.
        assert wrapper.undo_remap() is None

    def test_undo_key_handler_falls_back_to_napari_undo(self):
        """_on_undo_key uses wrapper undo first, then napari's paint undo."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        base = np.zeros((2, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.undo_called = False

            def refresh(self):
                pass

            def undo(self):
                self.undo_called = True

        layer = _FakeLabels(wrapper)
        inspector = LabelInspector(self.viewer)

        wrapper.remap_values({5: 0})
        inspector._on_undo_key(layer)
        assert np.any(np.asarray(wrapper) == 5)  # deletion undone
        assert not layer.undo_called

        inspector._on_undo_key(layer)  # nothing pending → napari undo
        assert layer.undo_called

    def test_save_over_source_file(self):
        """Saving must not truncate the file the lazy graph reads from.

        Regression test: the wrapper's dask graph reads pages from the label
        file itself; writing to that path directly destroys the source
        before the first slice is read.
        """
        import tifffile

        from napari_tmidas._label_inspection import (
            _load_label,
            _save_label_wrapper,
        )

        path = os.path.join(self.temp_dir, "labels_tracked.tif")
        base = np.zeros((4, 3, 16, 16), dtype=np.uint32)
        base[:, :, 0, 0] = 5
        base[:, :, 1, 1] = 7
        tifffile.imwrite(
            path, base, compression="zlib", photometric="minisblack"
        )

        # Load through the real lazy reader (dask graph reads from `path`).
        wrapper = _load_label(path)
        wrapper.remap_values({5: 0})
        wrapper[np.array([2]), np.array([1]), np.array([3]), np.array([3])] = 9

        # Save back to the SAME path the graph reads from.
        _save_label_wrapper(wrapper, path)

        expected = base.copy()
        expected[expected == 5] = 0
        expected[2, 1, 3, 3] = 9
        assert np.array_equal(tifffile.imread(path), expected)
        # Post-save reads through the wrapper stay consistent (idempotent).
        assert np.array_equal(np.asarray(wrapper), expected)


class TestLabelInspectorWidget:
    def test_widget_creation(self):
        """Test that the label inspector widget can be imported and called"""
        # Just test that the function exists and can be called
        # (without actually creating the widget to avoid Qt issues)
        assert callable(label_inspector_widget)

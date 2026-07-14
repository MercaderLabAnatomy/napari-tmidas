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

    def test_relabel_label_all_t_preserves_unsaved_edits(self):
        """Relabeling across T merges IDs and keeps pending manual edits."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        class _FakeLabels:
            def __init__(self, data, selected_label):
                self.data = data
                self.selected_label = selected_label

            def refresh(self):
                pass

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5  # label to relabel, present at every T
        base[:, 1, 1] = 3  # merge target
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))

        # Unsaved paint edit at T=1.
        wrapper[np.array([1]), np.array([2]), np.array([2])] = 7

        layer = _FakeLabels(wrapper, selected_label=3)
        self.viewer.layers = [layer]

        inspector = LabelInspector(self.viewer)
        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            # new_id defaults to the selected label (the pipetted ID).
            inspector.relabel_label_all_timepoints(5)

            result = np.asarray(layer.data)
            assert not np.any(result == 5)
            assert np.all(result[:, 0, 0] == 3)  # merged into 3
            assert np.all(result[:, 1, 1] == 3)
            assert result[1, 2, 2] == 7  # unsaved paint edit survived

            # No-ops: background source and identity relabel.
            inspector.relabel_label_all_timepoints(0, 9)
            inspector.relabel_label_all_timepoints(3, 3)
            result = np.asarray(layer.data)
            assert np.all(result[:, 0, 0] == 3)
            assert not np.any(result == 9)

    def test_relabel_label_all_t_numpy(self):
        """Plain numpy labels are relabeled in place."""

        class _FakeLabels:
            def __init__(self, data, selected_label):
                self.data = data
                self.selected_label = selected_label

            def refresh(self):
                pass

        data = np.zeros((4, 4), dtype=np.uint32)
        data[0, 0] = 5
        layer = _FakeLabels(data, selected_label=2)
        self.viewer.layers = [layer]

        inspector = LabelInspector(self.viewer)
        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            inspector.relabel_label_all_timepoints(5)
        assert layer.data[0, 0] == 2

    def test_click_to_relabel_mode(self):
        """Plain click relabels to the selected ID; Ctrl+click pipettes."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 3
                self.clicked_value = 5

            def refresh(self):
                pass

            def get_value(self, position, **kwargs):
                return self.clicked_value

        class _FakeEvent:
            def __init__(self, modifiers=()):
                self.button = 1
                self.type = "mouse_press"
                self.position = (0, 0, 0)
                self.view_direction = None
                self.dims_displayed = [1, 2]
                self.modifiers = modifiers

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        base[:, 1, 1] = 3
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=(1, 4, 4)))
        layer = _FakeLabels(wrapper)

        self.viewer.layers = [layer]
        self.viewer.mouse_drag_callbacks = []

        inspector = LabelInspector(self.viewer)
        inspector.enable_click_relabel(True)
        assert len(self.viewer.mouse_drag_callbacks) == 1

        def _fire(event):
            gen = self.viewer.mouse_drag_callbacks[0](self.viewer, event)
            next(gen)
            event.type = "mouse_release"
            with contextlib.suppress(StopIteration):
                next(gen)

        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            # Drag (pan): callback must NOT relabel.
            event = _FakeEvent()
            gen = self.viewer.mouse_drag_callbacks[0](self.viewer, event)
            next(gen)
            event.type = "mouse_move"
            next(gen)
            event.type = "mouse_release"
            with contextlib.suppress(StopIteration):
                next(gen)
            assert np.any(np.asarray(layer.data) == 5)

            # Ctrl+click: pipette — pick up the clicked ID, no relabel.
            layer.clicked_value = 3
            _fire(_FakeEvent(modifiers=("Control",)))
            assert layer.selected_label == 3
            assert np.any(np.asarray(layer.data) == 5)

            # Clean click on label 5: relabeled to the pipetted 3 on all T.
            layer.clicked_value = 5
            _fire(_FakeEvent())
            result = np.asarray(layer.data)
            assert not np.any(result == 5)
            assert np.all(result[:, 0, 0] == 3)

        inspector.enable_click_relabel(False)
        assert len(self.viewer.mouse_drag_callbacks) == 0

    def test_click_modes_mutually_exclusive(self):
        """Enabling one click mode disables the other."""
        self.viewer.layers = []
        self.viewer.mouse_drag_callbacks = []

        inspector = LabelInspector(self.viewer)
        inspector.enable_click_delete(True)
        inspector.enable_click_relabel(True)
        assert inspector._click_delete_cb is None
        assert inspector._click_relabel_cb is not None
        assert len(self.viewer.mouse_drag_callbacks) == 1

        inspector.enable_click_delete(True)
        assert inspector._click_relabel_cb is None
        assert inspector._click_delete_cb is not None
        assert len(self.viewer.mouse_drag_callbacks) == 1

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
        """_on_undo_key uses wrapper undo first, then napari's paint undo.

        The handler resolves the labels layer itself and ignores the
        provider argument napari passes, so it works whether Ctrl+Z is
        dispatched from the viewer or the layer keymap (i.e. regardless of
        which layer is currently active).
        """
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
        self.viewer.layers = [layer]
        inspector = LabelInspector(self.viewer)

        wrapper.remap_values({5: 0})
        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            # Provider is the viewer (as when the labels layer is NOT active);
            # the handler still resolves the labels layer and undoes.
            inspector._on_undo_key(self.viewer)
            assert np.any(np.asarray(wrapper) == 5)  # deletion undone
            assert not layer.undo_called

            inspector._on_undo_key()  # nothing pending → napari undo
            assert layer.undo_called

    def test_bind_undo_key_binds_viewer_and_layer(self):
        """Ctrl+Z is bound on the viewer (always in keymap chain) + layer."""

        class _FakeLayer:
            def __init__(self):
                self.bound = {}

            def bind_key(self, key, func, overwrite=False):
                self.bound[key] = func

        self.viewer.bind_key = Mock()
        layer = _FakeLayer()
        inspector = LabelInspector(self.viewer)

        inspector._bind_undo_key(layer, True)
        # Bound on the viewer so it fires even when another layer is active.
        # (bound methods compare equal but are not identical objects.)
        self.viewer.bind_key.assert_called_with(
            "Control-Z", inspector._on_undo_key, overwrite=True
        )
        assert layer.bound["Control-Z"] == inspector._on_undo_key

        inspector._bind_undo_key(layer, False)
        # None unbinds, restoring napari's native undo.
        self.viewer.bind_key.assert_called_with(
            "Control-Z", None, overwrite=True
        )
        assert layer.bound["Control-Z"] is None

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

    def test_detect_channel_axis_from_tiff_path(self):
        """A multi-channel TZCYX raw TIFF reports its channel axis so it can be
        aligned with a channel-less TZYX label.

        Regression: channel-axis detection was gated on zarr inputs, so a
        multi-channel raw TIFF paired with a single-channel label was overlaid
        with mismatched dimensions.
        """
        import tifffile

        from napari_tmidas._reader import detect_channel_axis_from_tiff_path

        raw = os.path.join(self.temp_dir, "raw.tif")
        # T=2, Z=3, C=2, Y=8, X=8 written with explicit axes metadata.
        data = np.zeros((2, 3, 2, 8, 8), dtype=np.uint16)
        tifffile.imwrite(
            raw, data, photometric="minisblack", metadata={"axes": "TZCYX"}
        )
        assert detect_channel_axis_from_tiff_path(raw) == 2

        # A label without a channel axis has none to report.
        lbl = os.path.join(self.temp_dir, "labels_tracked.tif")
        tifffile.imwrite(
            lbl,
            np.zeros((2, 3, 8, 8), dtype=np.uint32),
            photometric="minisblack",
            metadata={"axes": "TZYX"},
        )
        assert detect_channel_axis_from_tiff_path(lbl) is None

    def test_detect_channel_axis_matches_reader_array_across_sources(self):
        """The detected channel index must match the axis position in the array
        the reader actually hands napari — for any writer's axis order and even
        when a singleton dimension is squeezed away.

        C sits at a different position depending on the source (ImageJ/Java vs
        OME vs plain Python).  tifffile normalizes each into a consistent
        (axes, shape, array) triple, and a singleton axis (e.g. T=1) drops from
        both the axes string and the array together, shifting C's index.  The
        detector reads C from that same axes string, so its index stays aligned
        with the reader's array in every case — this test pins that invariant.
        """
        import tifffile

        from napari_tmidas._reader import (
            detect_channel_axis_from_tiff_path,
            tiff_reader_function,
        )

        cases = [
            # (filename, shape, axes, writer-kwargs, C's index in written array)
            ("ij_full.tif", (4, 3, 2, 8, 8), "TZCYX", {"imagej": True}, 2),
            ("ome_tczyx.tif", (4, 2, 3, 8, 8), "TCZYX", {"ome": True}, 1),
            ("ij_t1.tif", (1, 3, 2, 8, 8), "TZCYX", {"imagej": True}, 2),
            ("ome_t1.tif", (1, 3, 2, 8, 8), "TZCYX", {"ome": True}, 2),
        ]
        for name, shape, axes, kw, c_written in cases:
            path = os.path.join(self.temp_dir, name)
            arr = np.zeros(shape, dtype=np.uint16)
            # Give C a distinctive per-plane signature so we can locate it in
            # the reader's array regardless of how dims were squeezed/reordered.
            idx = [slice(None)] * arr.ndim
            idx[c_written] = 1
            arr[tuple(idx)] = 7
            tifffile.imwrite(path, arr, metadata={"axes": axes}, **kw)

            detected = detect_channel_axis_from_tiff_path(path)
            reader_arr = np.asarray(tiff_reader_function(path)[0][0])

            # The detected index must be valid for the reader's array and point
            # at the axis of size 2 carrying the channel signature.
            assert detected is not None, name
            assert 0 <= detected < reader_arr.ndim, name
            assert reader_arr.shape[detected] == 2, (name, reader_arr.shape)
            take1 = np.take(reader_arr, 1, axis=detected)
            assert take1.max() == 7, name

    def test_resolve_channel_axis_override(self):
        """The manual override wins over auto-detection and is range-checked."""
        from unittest.mock import Mock

        inspector = LabelInspector(Mock())
        image = np.zeros((2, 3, 2, 8, 8))  # 5D raw (TZCYX)
        label = np.zeros((2, 3, 8, 8))  # 4D label (TZYX)

        # Forced index is used verbatim (no metadata needed).
        inspector.channel_axis_override = "2"
        assert inspector._resolve_channel_axis(image, label, "raw.tif") == 2

        # "None" suppresses the channel axis even when dims differ.
        inspector.channel_axis_override = "None"
        assert inspector._resolve_channel_axis(image, label, "raw.tif") is None

        # An out-of-range manual pick degrades to None rather than misaligning.
        inspector.channel_axis_override = "9"
        assert inspector._resolve_channel_axis(image, label, "raw.tif") is None

        # "Auto" falls back to TIFF metadata detection.
        inspector.channel_axis_override = "Auto"
        with patch(
            "napari_tmidas._reader.detect_channel_axis_from_tiff_path",
            return_value=2,
        ):
            assert (
                inspector._resolve_channel_axis(image, label, "raw.tif") == 2
            )

        # Auto with equal dims (no extra axis) detects nothing.
        assert (
            inspector._resolve_channel_axis(label, label, "raw.tif") is None
        )

        # Auto falls back to the shape heuristic when metadata is unavailable
        # (e.g. a TIFF whose axes tags tifffile reports as "QQYX").
        inspector.channel_axis_override = "Auto"
        with patch(
            "napari_tmidas._reader.detect_channel_axis_from_tiff_path",
            return_value=None,
        ):
            # (2, 3, 2, 8, 8): channel axis 2 recovered from shape alone.
            assert (
                inspector._resolve_channel_axis(image, label, "raw.tif") == 2
            )


class TestLabelInspectorWidget:
    def test_widget_creation(self):
        """Test that the label inspector widget can be imported and called"""
        # Just test that the function exists and can be called
        # (without actually creating the widget to avoid Qt issues)
        assert callable(label_inspector_widget)

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

            # Real canvas events carry vispy Key objects, not strings —
            # the pipette must recognize those too (str(Key) is
            # "<Key 'Control'>", so a str()-based check silently fails).
            from vispy.util import keys

            layer.clicked_value = 5
            _fire(_FakeEvent(modifiers=(keys.CONTROL,)))
            assert layer.selected_label == 5
            assert np.any(np.asarray(layer.data) == 5)
            layer.selected_label = 3

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

    def test_delete_low_intensity_tracks_bitdepth_invariant(self):
        """Tracks below the normalized threshold are deleted on all T; the
        same threshold behaves identically for 8-bit and 16-bit raws.
        """
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 1

            def refresh(self):
                pass

        # Label 5 sits on dim raw signal, label 7 on bright signal.
        base = np.zeros((3, 8, 8), dtype=np.uint32)
        base[:, 1, 1] = 5
        base[:, 4, 4] = 7

        for dtype, dim, bright in [
            (np.uint8, 25, 240),
            (np.uint16, 6400, 61440),  # same fractions of the 16-bit range
        ]:
            raw = np.zeros((3, 8, 8), dtype=dtype)
            raw[:, 1, 1] = dim
            raw[:, 4, 4] = bright

            wrapper = _DaskFancyIndexWrapper(
                da.from_array(base.copy(), chunks=(1, 8, 8))
            )
            layer = _FakeLabels(wrapper)
            self.viewer.layers = [layer]

            inspector = LabelInspector(self.viewer)
            inspector.image_label_pairs = [("raw.tif", "lbl.tif")]
            inspector.channel_axis_override = "None"

            with patch(
                "napari_tmidas._label_inspection.Labels", _FakeLabels
            ), patch(
                "napari_tmidas._label_inspection._load_image",
                return_value=raw,
            ):
                inspector.delete_low_intensity_tracks(0.5)

            result = np.asarray(layer.data)
            assert not np.any(result == 5), dtype  # dim track deleted
            assert np.all(result[:, 4, 4] == 7), dtype  # bright track kept

    def test_delete_low_intensity_preview_undoes_previous(self):
        """Changing the threshold restores all tracks first, so previews
        reflect only the current setting instead of compounding."""
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 1

            def refresh(self):
                pass

        base = np.zeros((3, 8, 8), dtype=np.uint32)
        base[:, 1, 1] = 5  # dim
        base[:, 4, 4] = 7  # bright
        raw = np.zeros((3, 8, 8), dtype=np.uint8)
        raw[:, 0, 0] = 255  # bright background sets the normalization max
        raw[:, 1, 1] = 60  # dim track  → norm ~0.24
        raw[:, 4, 4] = 200  # bright track → norm ~0.78

        wrapper = _DaskFancyIndexWrapper(
            da.from_array(base.copy(), chunks=(1, 8, 8))
        )
        layer = _FakeLabels(wrapper)
        self.viewer.layers = [layer]
        inspector = LabelInspector(self.viewer)
        inspector.image_label_pairs = [("raw.tif", "lbl.tif")]
        inspector.channel_axis_override = "None"

        with patch(
            "napari_tmidas._label_inspection.Labels", _FakeLabels
        ), patch(
            "napari_tmidas._label_inspection._load_image", return_value=raw
        ):
            # High threshold deletes BOTH tracks.
            inspector.delete_low_intensity_tracks(0.9)
            assert not np.any(np.asarray(layer.data) == 5)
            assert not np.any(np.asarray(layer.data) == 7)

            # Lower threshold: previous deletion is undone first, so the
            # bright track returns and only the dim one is removed.
            inspector.delete_low_intensity_tracks(0.5)
            result = np.asarray(layer.data)
            assert not np.any(result == 5)
            assert np.all(result[:, 4, 4] == 7)

            # Threshold 0 restores everything.
            inspector.delete_low_intensity_tracks(0.0)
            result = np.asarray(layer.data)
            assert np.all(result[:, 1, 1] == 5)
            assert np.all(result[:, 4, 4] == 7)

    def test_delete_low_intensity_tracks_channels_and_scale(self):
        """Multi-channel raws are averaged over C, and a raw at higher
        resolution than the label is resampled onto the label grid.
        """

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 1

            def refresh(self):
                pass

        # 3 timepoints, labels at 8x8; raw at 16x16 with a channel axis (C=2).
        labels = np.zeros((3, 8, 8), dtype=np.uint32)
        labels[:, 1, 1] = 5  # dim in both channels
        labels[:, 4, 4] = 7  # bright in one channel only → mean still high
        raw = np.zeros((3, 2, 16, 16), dtype=np.uint16)
        raw[:, :, 2:4, 2:4] = 500  # label 5 footprint at 2x resolution
        raw[:, 0, 8:10, 8:10] = 65000  # label 7, channel 0 only

        layer = _FakeLabels(labels)
        self.viewer.layers = [layer]

        inspector = LabelInspector(self.viewer)
        inspector.image_label_pairs = [("raw.tif", "lbl.tif")]
        inspector.channel_axis_override = "1"

        with patch(
            "napari_tmidas._label_inspection.Labels", _FakeLabels
        ), patch(
            "napari_tmidas._label_inspection._load_image", return_value=raw
        ):
            inspector.delete_low_intensity_tracks(0.3)

        assert not np.any(layer.data == 5)  # dim track deleted in place
        assert np.all(layer.data[:, 4, 4] == 7)  # channel-mean keeps 7

    def test_delete_low_intensity_tracks_channel_selection(self):
        """The chosen channel drives the score; switching channels re-measures.

        Label 7 is bright only in channel 0. Averaging keeps it, but scoring
        on channel 1 (where it is dark) deletes it — and re-applying with a
        different channel invalidates the cached measurement.
        """

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 1

            def refresh(self):
                pass

        labels = np.zeros((2, 8, 8), dtype=np.uint32)
        labels[:, 1, 1] = 5  # bright in both channels
        labels[:, 4, 4] = 7  # bright in channel 0 only
        raw = np.zeros((2, 2, 8, 8), dtype=np.uint16)
        raw[:, :, 1, 1] = 40000  # label 5, both channels
        raw[:, 0, 4, 4] = 65000  # label 7, channel 0 only (channel 1 stays 0)

        layer = _FakeLabels(labels)
        self.viewer.layers = [layer]
        inspector = LabelInspector(self.viewer)
        inspector.image_label_pairs = [("raw.tif", "lbl.tif")]
        inspector.channel_axis_override = "1"

        with patch(
            "napari_tmidas._label_inspection.Labels", _FakeLabels
        ), patch(
            "napari_tmidas._label_inspection._load_image", return_value=raw
        ):
            # Channel 0: label 7 is bright → survives; label 5 (also bright) too.
            inspector.delete_low_intensity_tracks(0.5, channel="0")
            assert np.all(layer.data[:, 4, 4] == 7)
            assert np.all(layer.data[:, 1, 1] == 5)

            # Channel 1: label 7 is dark there → deleted; label 5 stays.
            inspector.delete_low_intensity_tracks(0.5, channel="1")
            assert not np.any(layer.data == 7)
            assert np.all(layer.data[:, 1, 1] == 5)

    def test_delete_low_intensity_tracks_none_below(self):
        """Nothing is deleted when all tracks pass; status reports it."""

        class _FakeLabels:
            def __init__(self, data):
                self.data = data
                self.selected_label = 1

            def refresh(self):
                pass

        labels = np.zeros((2, 4, 4), dtype=np.uint32)
        labels[:, 1, 1] = 3
        raw = np.zeros((2, 4, 4), dtype=np.uint8)
        raw[:, 1, 1] = 255

        layer = _FakeLabels(labels)
        self.viewer.layers = [layer]
        inspector = LabelInspector(self.viewer)
        inspector.image_label_pairs = [("raw.tif", "lbl.tif")]
        inspector.channel_axis_override = "None"

        with patch(
            "napari_tmidas._label_inspection.Labels", _FakeLabels
        ), patch(
            "napari_tmidas._label_inspection._load_image", return_value=raw
        ):
            inspector.delete_low_intensity_tracks(0.5)

        assert np.all(labels[:, 1, 1] == 3)
        assert "No tracks below" in self.viewer.status

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


class TestTrackInspection:
    """Track-inspection views: stack-T-along-Z and Z-max-projection."""

    def _wrapper(self, base):
        import dask.array as da

        from napari_tmidas._label_inspection import _DaskFancyIndexWrapper

        return _DaskFancyIndexWrapper(
            da.from_array(base, chunks=(1, *base.shape[1:]))
        )

    def test_stacked_view_geometry_and_reads(self):
        """Plane i of the stacked view is timepoint i//Z, slice i%Z."""
        from napari_tmidas._label_inspection import _StackedTrackView

        base = np.arange(3 * 2 * 4 * 4, dtype=np.uint32).reshape(3, 2, 4, 4)
        view = _StackedTrackView(self._wrapper(base))

        assert view.shape == (6, 4, 4)
        assert view.ndim == 3 and view.dtype == base.dtype
        for i in range(6):
            np.testing.assert_array_equal(view[i], base[i // 2, i % 2])
        # Scalar, sliced and full reads (the last is napari's 3-D display).
        assert view[5, 3, 3] == base[2, 1, 3, 3]
        np.testing.assert_array_equal(view[1:4], base.reshape(6, 4, 4)[1:4])
        np.testing.assert_array_equal(np.asarray(view), base.reshape(6, 4, 4))

    def test_stacked_view_fancy_read_and_write(self):
        """3-D picking reads and paint writes map back to (t, z)."""
        from napari_tmidas._label_inspection import _StackedTrackView

        base = np.zeros((3, 2, 4, 4), dtype=np.uint32)
        base[1, 0, 1, 1] = 5  # stacked plane 2
        wrapper = self._wrapper(base)
        view = _StackedTrackView(wrapper)

        # Ray-cast-style fancy read across planes.
        vals = view[
            np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([1, 1, 1])
        ]
        np.testing.assert_array_equal(vals, [0, 5, 0])

        # Paint across a timepoint boundary: planes 1 and 2 are
        # (t=0, z=1) and (t=1, z=0).
        view[np.array([1, 2]), np.array([3, 3]), np.array([0, 0])] = 9
        assert view[1, 3, 0] == 9 and view[2, 3, 0] == 9
        result = np.asarray(wrapper)
        assert result[0, 1, 3, 0] == 9 and result[1, 0, 3, 0] == 9

        # Scalar-indexed write also maps back to (t, z).
        view[5, 0, 0] = 7
        assert np.asarray(wrapper)[2, 1, 0, 0] == 7

    def test_fancy_reads_keep_numpy_shape_semantics(self):
        """Single-voxel fancy reads must return shape (1,), not 0-d —
        napari's data_setitem boolean-indexes with the result, and a 0-d
        value silently produces corrupt 2-D indices."""
        from napari_tmidas._label_inspection import _StackedTrackView

        base = np.zeros((2, 2, 4, 4), dtype=np.uint32)
        base[1, 1, 2, 2] = 6
        wrapper = self._wrapper(base)
        view = _StackedTrackView(wrapper)

        # All-constant single-voxel read on the view (stacked plane 3).
        out = view[np.array([3]), np.array([2]), np.array([2])]
        assert out.shape == (1,) and out[0] == 6
        # Same on the wrapper itself (regular 4-D layer editing path).
        out = wrapper[
            np.array([1]), np.array([1]), np.array([2]), np.array([2])
        ]
        assert out.shape == (1,) and out[0] == 6

    def test_maxproj_view_projects_and_is_readonly(self):
        """Max projection collapses Z per timepoint; paint is rejected."""
        import pytest

        from napari_tmidas._label_inspection import _MaxProjTrackView

        base = np.zeros((2, 3, 4, 4), dtype=np.uint32)
        base[0, 0, 1, 1] = 2
        base[0, 2, 1, 1] = 7  # overlaps in Z → max wins
        view = _MaxProjTrackView(self._wrapper(base))

        assert view.shape == (2, 4, 4)
        assert view[0][1, 1] == 7
        np.testing.assert_array_equal(np.asarray(view), base.max(axis=1))
        with pytest.raises(TypeError):
            view[np.array([0]), np.array([1]), np.array([1])] = 3

    def test_views_reflect_remaps(self):
        """Remaps on the source show through; the projection cache is
        stale until invalidated (the inspector invalidates on every edit)."""
        from napari_tmidas._label_inspection import (
            _MaxProjTrackView,
            _StackedTrackView,
        )

        base = np.zeros((2, 2, 4, 4), dtype=np.uint32)
        base[:, :, 0, 0] = 5
        wrapper = self._wrapper(base)
        stacked = _StackedTrackView(wrapper)
        proj = _MaxProjTrackView(wrapper)
        assert stacked[0, 0, 0] == 5 and proj[0][0, 0] == 5

        wrapper.remap_values({5: 3})
        assert stacked[0, 0, 0] == 3  # read-through, no cache
        assert proj[0][0, 0] == 5  # cached projection is stale...
        proj.invalidate()
        assert proj[0][0, 0] == 3  # ...until invalidated

    def test_views_on_tyx_source(self):
        """A TYX movie needs no stacking: both views alias the source."""
        from napari_tmidas._label_inspection import (
            _MaxProjTrackView,
            _StackedTrackView,
        )

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 2, 2] = 4
        wrapper = self._wrapper(base)
        stacked = _StackedTrackView(wrapper)
        proj = _MaxProjTrackView(wrapper)
        assert stacked.shape == base.shape and proj.shape == base.shape
        np.testing.assert_array_equal(np.asarray(stacked), base)
        np.testing.assert_array_equal(np.asarray(proj), base)

    def test_materialized_volume_serves_reads_and_remaps_in_place(self):
        """After 3-D materialization, picks cost no source I/O and
        delete/relabel updates the cached volume in place."""
        from napari_tmidas._label_inspection import _StackedTrackView

        base = np.zeros((3, 2, 4, 4), dtype=np.uint32)
        base[:, :, 0, 0] = 5
        base[:, 0, 2, 2] = 7
        wrapper = self._wrapper(base)
        view = _StackedTrackView(wrapper)

        vol = np.asarray(view)  # 3-D display path materializes...
        assert np.asarray(view) is vol  # ...and caches

        # Pick-ray reads are now served from the volume — no source I/O.
        loads = []
        orig = view._t_slice
        view._t_slice = lambda t: loads.append(t) or orig(t)
        vals = view[np.array([0, 2]), np.array([0, 2]), np.array([0, 2])]
        assert vals.tolist() == [5, 7] and loads == []

        # Delete via remap: the cached volume updates in place.
        wrapper.remap_values({5: 0})
        view.apply_mapping({5: 0})
        assert vol[0, 0, 0] == 0 and not np.any(np.asarray(view) == 5)

        # Paint through the view lands in both the source and the volume.
        view[np.array([1]), np.array([3]), np.array([3])] = 9
        assert vol[1, 3, 3] == 9 and np.asarray(wrapper)[0, 1, 3, 3] == 9

        # invalidate drops the cache; the next read rebuilds from source.
        view.invalidate()
        assert view._vol is None and loads == []
        assert view[3][0, 0] == 0  # rebuilt from the remapped source
        assert loads  # source was actually consulted again

    def test_maxproj_remap_artifact_and_exact_plane_recompute(self):
        """Volume remap is in place (documented z-overlap artifact);
        2-D planes recompute exactly and reveal occluded labels."""
        from napari_tmidas._label_inspection import _MaxProjTrackView

        base = np.zeros((2, 2, 4, 4), dtype=np.uint32)
        base[:, 0, 1, 1] = 2  # occluded by 5 (max by value)
        base[:, 1, 1, 1] = 5
        wrapper = self._wrapper(base)

        # 2-D path (no materialized volume): plane recomputes exactly.
        view = _MaxProjTrackView(wrapper)
        assert view[0][1, 1] == 5
        wrapper.remap_values({5: 0})
        view.apply_mapping({5: 0})
        assert view[0][1, 1] == 2  # occluded label revealed

        # 3-D path: the cached volume is remapped in place — background
        # where the deleted track occluded another (rebuild restores it).
        wrapper.undo_remap()
        view.invalidate()
        vol = np.asarray(view)
        assert vol[0, 1, 1] == 5
        wrapper.remap_values({5: 0})
        view.apply_mapping({5: 0})
        assert view[0][1, 1] == 0  # served from the remapped volume
        view.invalidate()
        assert view[0][1, 1] == 2  # exact after rebuild

    def test_set_track_view_mode_swaps_layers(self):
        """The dropdown swaps the view layer in/out and keeps the regular
        labels layer as the editing/saving target."""
        from napari_tmidas._label_inspection import (
            LabelInspector,
            _MaxProjTrackView,
            _StackedTrackView,
        )

        class _FakeLabels:
            def __init__(self, data, name="", scale=None):
                self.data = data
                self.name = name
                self.scale = scale or [1.0] * getattr(data, "ndim", 1)
                self.visible = True
                self.selected_label = 1

            def refresh(self):
                pass

        class _FakeViewer:
            def __init__(self):
                self.layers = []
                self.status = ""
                self.mouse_drag_callbacks = []

            def add_labels(self, data, scale=None, name=""):
                layer = _FakeLabels(data, name=name, scale=scale)
                self.layers.append(layer)
                return layer

        base = np.zeros((2, 3, 4, 4), dtype=np.uint32)
        base[:, :, 0, 0] = 5
        wrapper = self._wrapper(base)

        viewer = _FakeViewer()
        main = _FakeLabels(wrapper, scale=[1.0, 2.0, 1.0, 1.0])
        viewer.layers.append(main)

        inspector = LabelInspector(viewer)
        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            inspector.set_track_view_mode("stack")
            view_layer = inspector._track_view_layer
            assert isinstance(view_layer.data, _StackedTrackView)
            assert view_layer.data.shape == (6, 4, 4)
            # Stacked axis reuses the Z spacing; YX scale is preserved.
            assert view_layer.scale == [2.0, 1.0, 1.0]
            assert main.visible is False
            # The regular labels layer stays the editing/saving target
            # regardless of layer order; clicks resolve on the view.
            viewer.layers.reverse()
            assert inspector._find_labels_layer() is main
            viewer.layers.reverse()
            assert inspector._click_value_layer() is view_layer

            # Switching modes replaces the view layer.
            inspector.set_track_view_mode("max")
            assert view_layer not in viewer.layers
            view_layer = inspector._track_view_layer
            assert isinstance(view_layer.data, _MaxProjTrackView)

            # An all-T deletion refreshes the (cached) projection too.
            assert view_layer.data[0][0, 0] == 5
            inspector.delete_label_all_timepoints(5)
            assert view_layer.data[0][0, 0] == 0

            inspector.set_track_view_mode("off")
            assert inspector._track_view_layer is None
            assert viewer.layers == [main]
            assert main.visible is True

    def test_pick_track_view_step(self):
        """The YX step is the smallest that fits the volume in budget."""
        from napari_tmidas._label_inspection import _pick_track_view_step

        gib = 1024**3
        # Fits exactly: no subsampling.
        assert _pick_track_view_step(10, 64, 64, 4, budget=10 * 64 * 64 * 4) == 1
        # The reported crash: 33 T x 75 Z planes of 2720x2720 uint32
        # is ~68 GiB — napari uploads it as ONE 3-D texture and the GPU
        # allocation failure corrupts vispy's command queue.
        step = _pick_track_view_step(33 * 75, 2720, 2720, 4, budget=4 * gib)
        assert step > 1
        assert (
            33 * 75 * (-(-2720 // step)) ** 2 * 4 <= 4 * gib
        ), "chosen step must actually fit the budget"
        assert (
            33 * 75 * (-(-2720 // (step - 1))) ** 2 * 4 > 4 * gib
        ), "step must be the smallest that fits"

    def test_subsampled_views_geometry_reads_and_readonly(self):
        """A yx_step > 1 view strides every plane, keeps IDs and the
        edit plumbing intact, and rejects paint (not losslessly
        writable back)."""
        import pytest

        from napari_tmidas._label_inspection import (
            _MaxProjTrackView,
            _StackedTrackView,
        )

        base = np.arange(2 * 2 * 5 * 5, dtype=np.uint32).reshape(2, 2, 5, 5)
        wrapper = self._wrapper(base)
        view = _StackedTrackView(wrapper, yx_step=2)

        assert view.shape == (4, 3, 3)  # ceil(5/2) per YX axis
        for i in range(4):
            np.testing.assert_array_equal(
                view[i], base[i // 2, i % 2, ::2, ::2]
            )
        np.testing.assert_array_equal(
            np.asarray(view), base.reshape(4, 5, 5)[:, ::2, ::2]
        )
        # Pick-ray fancy reads are in view coordinates.
        vals = view[np.array([0, 3]), np.array([1, 2]), np.array([0, 1])]
        np.testing.assert_array_equal(
            vals, [base[0, 0, 2, 0], base[1, 1, 4, 2]]
        )
        with pytest.raises(TypeError):
            view[0, 0, 0] = 9
        # Remaps still update the materialized volume in place.
        first = int(base[0, 0, 0, 0])
        wrapper.remap_values({first: 42})
        view.apply_mapping({first: 42})
        assert np.asarray(view)[0, 0, 0] == 42

        proj = _MaxProjTrackView(wrapper, yx_step=2)
        assert proj.shape == (2, 3, 3)
        np.testing.assert_array_equal(
            np.asarray(proj)[1], base[1].max(axis=0)[::2, ::2]
        )

    def test_apply_track_view_subsamples_when_over_budget(self):
        """Oversized movies get an automatically subsampled view whose
        layer scale compensates, so world-coordinate clicks still hit."""
        from napari_tmidas._label_inspection import LabelInspector

        class _FakeLabels:
            def __init__(self, data, name="", scale=None):
                self.data = data
                self.name = name
                self.scale = scale or [1.0] * getattr(data, "ndim", 1)
                self.visible = True
                self.selected_label = 1

            def refresh(self):
                pass

        class _FakeViewer:
            def __init__(self):
                self.layers = []
                self.status = ""
                self.mouse_drag_callbacks = []

            def add_labels(self, data, scale=None, name=""):
                layer = _FakeLabels(data, name=name, scale=scale)
                self.layers.append(layer)
                return layer

        base = np.zeros((2, 3, 8, 8), dtype=np.uint32)
        wrapper = self._wrapper(base)
        viewer = _FakeViewer()
        main = _FakeLabels(wrapper, scale=[1.0, 2.0, 1.5, 1.5])
        viewer.layers.append(main)
        inspector = LabelInspector(viewer)

        # Full stacked volume is 2*3*8*8*4 = 1536 B; a 400 B budget
        # needs step 2 (6*4*4*4 = 384 B).
        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels), \
             patch(
                 "napari_tmidas._label_inspection._TRACK_VIEW_BUDGET_BYTES",
                 400,
             ):
            inspector.set_track_view_mode("stack")
            view_layer = inspector._track_view_layer
            assert view_layer.data.yx_step == 2
            assert view_layer.data.shape == (6, 4, 4)
            # YX scale stretched by the step; stacked axis keeps Z spacing.
            assert view_layer.scale == [2.0, 3.0, 3.0]
            assert "subsampled x2" in viewer.status
            inspector.set_track_view_mode("off")


class TestLabelInspectorWidget:
    def test_widget_creation(self):
        """Test that the label inspector widget can be imported and called"""
        # Just test that the function exists and can be called
        # (without actually creating the widget to avoid Qt issues)
        assert callable(label_inspector_widget)

    def test_save_and_continue_saves_last_pair(self, tmp_path):
        """Regression: on the LAST pair, 'Save Changes and Continue'
        returned early before next_pair(), so edits to the final label
        image were silently dropped instead of written to disk.
        """
        import tifffile
        from unittest.mock import MagicMock

        from napari_tmidas._label_inspection import label_inspector

        tifffile.imwrite(
            str(tmp_path / "a.tif"), np.zeros((8, 8), dtype=np.uint16)
        )
        label_path = tmp_path / "a_labels.tif"
        tifffile.imwrite(str(label_path), np.zeros((8, 8), dtype=np.uint32))

        viewer = MagicMock()
        label_inspector(
            folder_path=str(tmp_path),
            label_suffix="_labels.tif",
            viewer=viewer,
        )

        # The first dock widget added inside label_inspector is the
        # save-and-continue button.
        save_widget = viewer.window.add_dock_widget.call_args_list[0].args[0]

        edited = np.zeros((8, 8), dtype=np.uint32)
        edited[2:4, 2:4] = 9

        class _FakeLabels:
            data = edited

            def save(self, path):
                tifffile.imwrite(path, self.data)

        viewer.layers.__iter__.side_effect = lambda: iter([_FakeLabels()])

        with patch("napari_tmidas._label_inspection.Labels", _FakeLabels):
            save_widget()

        assert np.array_equal(tifffile.imread(str(label_path)), edited)
        assert save_widget.call_button.enabled is False

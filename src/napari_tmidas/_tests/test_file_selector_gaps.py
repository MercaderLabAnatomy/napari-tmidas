"""
Behaviour tests for branches of :mod:`napari_tmidas._file_selector` that the
other ``test_file_selector*`` modules leave unexercised.

Grouped by what a user would notice:

* the batch worker: cancelling, odd inputs (multi-layer OME-Zarr, array-likes,
  memory errors), channel extraction, and the output files it writes;
* the batch widget: a worker that dies, the dimension-order hint, and
  re-applying a recorded run;
* the results table: showing processed files and tracker output;
* the metadata probes that feed the dimension-order suggestion.

Helpers are duplicated from the sibling modules on purpose, so each file can
be run on its own.
"""

import os
import types

import numpy as np
import pytest
import tifffile

import napari_tmidas._file_selector as fs
from napari_tmidas._registry import BatchProcessingRegistry

zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------
def _write_tif(path, data, axes=None, **metadata):
    kwargs = {"photometric": "minisblack"}
    if axes is not None or metadata:
        kwargs["ome"] = True
        kwargs["metadata"] = {"axes": axes, **metadata} if axes else metadata
    tifffile.imwrite(str(path), data, **kwargs)
    return str(path)


def _worker(tmp_path, func, files, params=None, **kwargs):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return fs.ProcessingWorker(
        files,
        func,
        params or {},
        str(out),
        kwargs.pop("input_suffix", ""),
        kwargs.pop("output_suffix", "_proc"),
        output_format=kwargs.pop("output_format", "tiff"),
    )


def _write_ome_zarr_group(path, arrays, axes, dataset_paths=None):
    """A zarr v3 group with OME multiscales metadata and the given arrays."""
    root = zarr.open_group(str(path), mode="w")
    for name, data in arrays.items():
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[...] = data
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": a} for a in axes],
            "datasets": [
                {"path": p} for p in (dataset_paths or list(arrays))
            ],
        }
    ]
    return str(path)


class _Layer:
    def __init__(self, data, name="layer", **kwargs):
        self.data = data
        self.name = name
        self.kwargs = kwargs


class _LayerList(list):
    def __init__(self):
        super().__init__()
        self.moves = []

    def move(self, src, dst):
        self.moves.append((src, dst))
        self.insert(dst, self.pop(src))


class _Viewer:
    def __init__(self, open_result=None, reset_error=None):
        self.layers = _LayerList()
        self.status = ""
        self.dims = types.SimpleNamespace(ndisplay=2)
        self._open_result = open_result
        self._reset_error = reset_error
        self.tracks = []

    def add_image(self, data, channel_axis=None, **kwargs):
        layer = _Layer(data, name=kwargs.pop("name", "image"), **kwargs)
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(data, name=kwargs.pop("name", "labels"), **kwargs)
        self.layers.append(layer)
        return layer

    def add_tracks(self, data, **kwargs):
        layer = _Layer(data, name=kwargs.pop("name", "tracks"), **kwargs)
        self.tracks.append(layer)
        return layer

    def open(self, filepath, plugin=None):
        # Opened layers land at index 0, below anything already shown, so
        # the table has to lift them to the top.
        result = self._open_result
        for layer in reversed(result if isinstance(result, list) else [result]):
            if layer is not None:
                self.layers.insert(0, layer)
        return result

    def reset_view(self):
        if self._reset_error is not None:
            raise self._reset_error


class _MessageBoxStub:
    Cancel = 0x400000
    Ignore = 0x100000


@pytest.fixture
def registry(monkeypatch):
    """An empty, isolated function registry restored after the test."""
    saved = dict(BatchProcessingRegistry._processing_functions)
    BatchProcessingRegistry._processing_functions.clear()
    monkeypatch.setattr(fs, "discover_and_load_processing_functions", lambda: None)
    yield BatchProcessingRegistry
    BatchProcessingRegistry._processing_functions.clear()
    BatchProcessingRegistry._processing_functions.update(saved)


def _register(registry, name, func=None, **kwargs):
    if func is None:

        def func(image):
            return image

    registry.register(name=name, **kwargs)(func)
    return func


def _results_widget(tmp_path, files=None):
    if files is None:
        files = [
            _write_tif(tmp_path / f"img{i}.tif", np.full((4, 4), i, np.uint16))
            for i in range(2)
        ]
    viewer = _Viewer()
    widget = fs.FileResultsWidget(viewer, files, str(tmp_path), "")
    return widget, viewer, files


def _table(qapp, viewer):
    return fs.ProcessedFilesTableWidget(viewer)


# ---------------------------------------------------------------------------
# ProcessingWorker.run: cancellation
# ---------------------------------------------------------------------------
class TestWorkerRunCancellation:
    def test_file_killed_by_a_cancel_is_reported_not_raised(self, tmp_path):
        """Cancelling kills Cellpose subprocesses, so the running file fails.

        run() must turn that into a per-file error and still finish, rather
        than crash the thread.  (test_workflow_log covers the same path in a
        real QThread, which coverage does not trace; this runs it inline.)
        """
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))
        worker = None

        def killed(image):
            worker.stop_requested = True
            raise RuntimeError("Cellpose subprocess was terminated")

        worker = _worker(tmp_path, killed, [src])
        errors, finished, processed = [], [], []
        worker.error_occurred.connect(lambda f, m: errors.append((f, m)))
        worker.processing_finished.connect(lambda: finished.append(True))
        worker.file_processed.connect(processed.append)

        worker.run()

        assert errors == [(src, "Cellpose subprocess was terminated")]
        assert finished == [True]
        assert processed == []

    def test_unexpected_error_without_cancel_propagates(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def broken(image):
            raise RuntimeError("bug in the function")

        worker = _worker(tmp_path, broken, [src])
        with pytest.raises(RuntimeError, match="bug in the function"):
            worker.run()


# ---------------------------------------------------------------------------
# ProcessingWorker.process_file: inputs
# ---------------------------------------------------------------------------
class TestWorkerInputs:
    def test_image_layer_is_preferred_over_an_earlier_labels_layer(
        self, tmp_path, monkeypatch, capsys
    ):
        labels = np.ones((4, 4), np.uint32)
        image = np.full((4, 4), 9, np.uint16)
        monkeypatch.setattr(
            fs,
            "load_image_file",
            lambda p, **k: [
                (labels, {}, "labels"),
                (image, {"metadata": {"axes": "yx", "channel_axis": None}}, "image"),
            ],
        )
        seen = {}

        def record(image):
            seen["image"] = image
            return image

        src = str(tmp_path / "in.zarr")
        result = _worker(tmp_path, record, [src]).process_file(src)

        np.testing.assert_array_equal(seen["image"], image)
        saved = tifffile.imread(result["processed_file"])
        assert saved.dtype == np.uint16
        out = capsys.readouterr().out
        assert "first image layer from 2" in out
        assert "Zarr axes: yx" in out
        assert "Channel axis: None" in out

    @pytest.mark.parametrize("convertible", [True, False])
    def test_array_like_without_dtype_is_saved_as_float32(
        self, tmp_path, monkeypatch, convertible
    ):
        class ArrayLike:
            def __array__(self, dtype=None, copy=None):
                if not convertible:
                    raise ValueError("cannot convert")
                return np.ones((4, 4))

        monkeypatch.setattr(fs, "load_image_file", lambda p, **k: ArrayLike())
        src = str(tmp_path / "in.tif")
        result = _worker(
            tmp_path, lambda image: np.full((4, 4), 2.5), [src]
        ).process_file(src)

        saved = tifffile.imread(result["processed_file"])
        assert saved.dtype == np.float32
        assert saved[0, 0] == pytest.approx(2.5)

    def test_out_of_memory_while_loading_dask_input_is_raised(
        self, tmp_path, monkeypatch
    ):
        class TooBig:
            shape = (4, 4)
            dtype = np.dtype(np.uint16)
            chunks = ((4,), (4,))

            def compute(self, **kwargs):
                raise MemoryError("cannot allocate")

        monkeypatch.setattr(fs, "load_image_file", lambda p, **k: TooBig())
        src = str(tmp_path / "in.zarr")
        worker = _worker(tmp_path, lambda image: image, [src])

        with pytest.raises(MemoryError):
            worker.process_file(src)
        assert os.listdir(worker.output_folder) == []

    def test_invalid_channel_falls_back_to_the_full_image(
        self, tmp_path, capsys
    ):
        data = np.zeros((2, 8, 8), np.uint16)
        src = _write_tif(tmp_path / "a.tif", data, axes="CYX")
        seen = {}

        def record(image):
            seen["shape"] = image.shape
            return image

        _worker(tmp_path, record, [src], {"channel": "green"}).process_file(src)

        assert seen["shape"] == (2, 8, 8)
        assert "channel extraction failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ProcessingWorker.process_file: results and the files written
# ---------------------------------------------------------------------------
class TestWorkerOutputs:
    def test_out_of_memory_computing_a_dask_result_is_raised(self, tmp_path):
        da = pytest.importorskip("dask.array")
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def lazy_result(image):
            return da.from_array(np.zeros((4, 4), np.uint16)).map_blocks(
                _raise_memory_error, dtype=np.uint16
            )

        worker = _worker(tmp_path, lazy_result, [src])
        with pytest.raises(MemoryError):
            worker.process_file(src)
        assert os.listdir(worker.output_folder) == []

    def test_layer_subdivision_skips_missing_layers(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def subdivide(image):
            return (np.ones((4, 4), np.int32), None, np.full((4, 4), 3, np.int32))

        result = _worker(
            tmp_path, subdivide, [src], output_suffix="_layer"
        ).process_file(src)

        names = [os.path.basename(p) for p in result["processed_files"]]
        assert names == ["a_inner.tif", "a_outer.tif"]

    def test_multi_output_images_keep_the_source_pixel_size(self, tmp_path):
        src = _write_tif(
            tmp_path / "a.tif",
            np.zeros((4, 4), np.uint16),
            axes="YX",
            PhysicalSizeX=0.25,
            PhysicalSizeY=0.5,
        )

        def two_images(image):
            return [np.full((4, 4), 1, np.uint16), np.full((4, 4), 2, np.uint16)]

        result = _worker(
            tmp_path, two_images, [src], output_suffix="_out.tif"
        ).process_file(src)

        for path in result["processed_files"]:
            scale = fs._extract_source_physical_scale(path, "YX")
            assert scale == {"Y": pytest.approx(0.5), "X": pytest.approx(0.25)}

    def test_folder_function_with_new_data_saves_the_file(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def process_folder_offsets(image):
            return image + 5

        result = _worker(
            tmp_path, process_folder_offsets, [src]
        ).process_file(src)

        assert tifffile.imread(result["processed_file"])[0, 0] == 5

    def test_auto_split_survives_failing_channel_detection(
        self, tmp_path, monkeypatch, capsys
    ):
        """Detection failing must not stop a function's new channel axis from
        being split: the shape fallback still applies."""
        src = _write_tif(tmp_path / "a.tif", np.zeros((8, 8), np.uint16))

        def stack_variants(image):
            return np.stack([image, image + 1])

        def boom(*args, **kwargs):
            raise OSError("metadata unreadable")

        monkeypatch.setattr(fs, "detect_channels_for_file", boom)
        result = _worker(
            tmp_path, stack_variants, [src], {"dimension_order": "Auto"}
        ).process_file(src)

        names = sorted(os.path.basename(p) for p in result["processed_files"])
        assert names == ["a_proc_channel_0.tif", "a_proc_channel_1.tif"]
        assert "channel detection for Auto split failed" in capsys.readouterr().out

    def test_split_label_channels_are_saved_as_uint32(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((8, 8), np.uint16))

        def two_label_maps(image):
            return np.stack(
                [np.ones((8, 8), np.int32), np.full((8, 8), 2, np.int32)]
            )

        result = _worker(
            tmp_path, two_label_maps, [src], {"dimension_order": "Auto"}
        ).process_file(src)

        saved = [tifffile.imread(p) for p in sorted(result["processed_files"])]
        assert [s.dtype for s in saved] == [np.uint32, np.uint32]
        assert [int(s.max()) for s in saved] == [1, 2]


def _raise_memory_error(block):
    raise MemoryError("cannot allocate")


# ---------------------------------------------------------------------------
# FileResultsWidget
# ---------------------------------------------------------------------------
class TestWorkerThreadDied:
    def test_crashed_worker_re_enables_start(self, tmp_path, qapp, registry):
        """If run() raises, processing_finished never comes; the thread's
        finished signal must still put the buttons back."""
        _register(registry, "Identity")
        widget, _, files = _results_widget(tmp_path)
        worker = fs.ProcessingWorker(files, lambda image: image, {}, "", "", "")
        widget.worker = worker
        widget._live_workers.add(worker)
        widget.batch_button.setEnabled(False)
        widget.cancel_button.setEnabled(True)
        widget.sender = lambda: worker

        widget._on_worker_thread_finished()

        assert widget.worker is None
        assert widget.batch_button.isEnabled()
        assert not widget.cancel_button.isEnabled()
        assert worker not in widget._live_workers

    def test_stale_worker_does_not_touch_the_current_run(
        self, tmp_path, qapp, registry
    ):
        _register(registry, "Identity")
        widget, _, files = _results_widget(tmp_path)
        old = fs.ProcessingWorker(files, lambda image: image, {}, "", "", "")
        current = fs.ProcessingWorker(files, lambda image: image, {}, "", "", "")
        widget.worker = current
        widget.batch_button.setEnabled(False)
        widget.sender = lambda: old

        widget._on_worker_thread_finished()

        assert widget.worker is current
        assert not widget.batch_button.isEnabled()


class TestDimensionOrderHint:
    def test_explicit_order_is_echoed_when_shape_is_unknown(
        self, tmp_path, qapp, registry
    ):
        _register(registry, "Identity")
        widget, _, _ = _results_widget(
            tmp_path, files=[str(tmp_path / "missing.tif")]
        )
        widget.dimension_order.setCurrentText("TZYX")
        widget.update_dimension_order_status()
        assert widget.dimension_order_status.text() == "Axes: TZYX"

    def test_unknown_function_still_warns_about_3d_on_auto(
        self, tmp_path, qapp, registry
    ):
        files = [_write_tif(tmp_path / "z.tif", np.zeros((5, 4, 4), np.uint8))]
        widget, _, _ = _results_widget(tmp_path, files=files)
        widget.processing_selector.addItem("Not registered")
        widget.processing_selector.setCurrentText("Not registered")
        widget.dimension_order.setCurrentText("Auto")
        widget.update_dimension_order_status()
        assert widget.dimension_order_status.text().startswith("⚠")

    def test_without_message_box_the_batch_is_not_blocked(
        self, tmp_path, qapp, registry, monkeypatch
    ):
        def segment(image, dimension_order="Auto"):
            return image

        files = [_write_tif(tmp_path / "z.tif", np.zeros((5, 4, 4), np.uint8))]
        widget, _, _ = _results_widget(tmp_path, files=files)
        widget.dimension_order.setCurrentText("Auto")
        monkeypatch.setattr(fs, "QMessageBox", None)
        assert widget.confirm_dimension_order({"func": segment}) is True


class TestApplyWorkflowRun:
    def test_recorded_thread_count_is_restored(self, tmp_path, qapp, registry):
        _register(registry, "Identity")
        widget, viewer, _ = _results_widget(tmp_path)
        assert widget.thread_count.isEnabled()

        applied = widget.apply_workflow_run(
            {"run_id": "r1", "function": "Identity", "thread_count": 3}
        )

        assert applied is True
        assert widget.thread_count.value() == 3
        assert viewer.status == "Loaded the settings of run r1"


# ---------------------------------------------------------------------------
# ProcessedFilesTableWidget: showing processed files
# ---------------------------------------------------------------------------
class TestShowProcessedZarr:
    @pytest.fixture
    def zarr_path(self, tmp_path, monkeypatch):
        path = tmp_path / "p.zarr"
        path.mkdir()
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr("napari_ome_zarr.napari_get_reader", lambda p: None)
        return str(path)

    def test_single_opened_layer_is_renamed_and_raised(
        self, zarr_path, qapp
    ):
        layer = _Layer(np.zeros((4, 4), np.uint8), name="seg")
        layer.data_level = 2
        viewer = _Viewer(open_result=layer, reset_error=RuntimeError("no canvas"))
        viewer.layers.append(_Layer(np.zeros((2, 2)), name="original"))
        table = _table(qapp, viewer)

        table._load_processed_image(zarr_path)

        assert layer.name == "Processed seg"
        assert table.current_processed_images == [layer]
        # opened below the original, so it is moved to the top
        assert viewer.layers[-1] is layer
        assert "Loaded 1 processed layers" in viewer.status

    def test_pyramid_layers_are_shown_at_full_resolution(self, zarr_path, qapp):
        layers = [
            _Layer(np.zeros((4, 4), np.uint8), name=n) for n in ("a", "b")
        ]
        for layer in layers:
            layer.data_level = 2
        viewer = _Viewer(open_result=layers)
        table = _table(qapp, viewer)

        table._load_processed_image(zarr_path)

        assert [layer.name for layer in layers] == ["Processed a", "Processed b"]
        assert [layer.data_level for layer in layers] == [0, 0]

    def test_reader_with_no_layers_falls_back_to_the_loader(
        self, zarr_path, qapp, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: np.zeros((4, 4), np.uint8)
        )
        viewer = _Viewer(open_result=[])
        table = _table(qapp, viewer)

        table._load_processed_image(zarr_path)

        assert "returned no layers" in capsys.readouterr().out
        assert len(table.current_processed_images) == 1


class TestShowProcessedLayers:
    def test_tiff_reader_failure_falls_back_to_the_loader(
        self, tmp_path, qapp, monkeypatch, capsys
    ):
        path = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint8))

        def broken(p):
            raise OSError("bad OME-XML")

        monkeypatch.setattr(fs, "tiff_reader_function", broken)
        table = _table(qapp, _Viewer())
        table._load_processed_image(path)

        assert "falling back to load_image_file" in capsys.readouterr().out
        assert len(table.current_processed_images) == 1

    def test_loader_entries_of_every_shape_become_named_layers(
        self, tmp_path, qapp, monkeypatch
    ):
        path = str(tmp_path / "x.png")
        open(path, "wb").close()
        monkeypatch.setattr(
            fs,
            "load_image_file",
            lambda p, **k: [
                (np.zeros((4, 4), np.uint8), {}),
                np.zeros((4, 4), np.uint8),
                (np.zeros((4, 4), np.uint32), {"name": "cells"}, "labels"),
            ],
        )
        table = _table(qapp, _Viewer())
        table._load_processed_image(path)

        assert [layer.name for layer in table.current_processed_images] == [
            "Processed C1: x.png",
            "Processed C2: x.png",
            "Processed cells",
        ]

    def test_processed_z_stack_switches_to_3d(self, tmp_path, qapp, monkeypatch):
        path = str(tmp_path / "x.png")
        open(path, "wb").close()
        monkeypatch.setattr(
            fs,
            "load_image_file",
            lambda p, **k: [(np.zeros((12, 8, 8), np.uint8), {}, "image")],
        )
        viewer = _Viewer()
        table = _table(qapp, viewer)
        table._load_processed_image(path)
        assert viewer.dims.ndisplay == 3


    def test_unreadable_npy_is_not_shown_as_points(self, tmp_path, qapp, capsys):
        path = tmp_path / "spots.npy"
        path.write_bytes(b"not an npy file")
        viewer = _Viewer()
        table = _table(qapp, viewer)
        table._load_processed_image(str(path))
        assert "Error loading NPY file as points" in capsys.readouterr().out
        assert table.current_processed_images == []


class TestShowOriginal:
    def test_original_z_stack_switches_to_3d(self, tmp_path, qapp):
        path = _write_tif(tmp_path / "z.tif", np.zeros((12, 8, 8), np.uint8))
        viewer = _Viewer()
        table = _table(qapp, viewer)
        table._load_original_image(path)
        assert len(table.current_original_images) == 1
        assert viewer.dims.ndisplay == 3


class TestShowTracks:
    def test_2d_labels_get_no_tracks(self, tmp_path, qapp):
        from napari.layers import Labels

        viewer = _Viewer()
        table = _table(qapp, viewer)
        layers = [Labels(np.zeros((4, 4), np.uint32))]
        table._add_tracks_layer(str(tmp_path / "a_tracked.tif"), layers)
        assert viewer.tracks == [] and len(layers) == 1

    def test_tracks_that_cannot_be_loaded_are_explained(
        self, tmp_path, qapp, monkeypatch
    ):
        from napari.layers import Labels

        import napari_tmidas.processing_functions.track_qc as track_qc

        path = _write_tif(
            tmp_path / "a_tracked.tif", np.zeros((2, 4, 4), np.uint32)
        )
        monkeypatch.setattr(track_qc, "LIVE_TRACKS_MAX_BYTES", 0)
        viewer = _Viewer()
        table = _table(qapp, viewer)
        layers = [Labels(np.zeros((2, 4, 4), np.uint32))]

        table._add_tracks_layer(path, layers)

        assert viewer.tracks == []
        assert viewer.status.startswith("No tracks shown for a_tracked.tif")
        assert "too large" in viewer.status


# ---------------------------------------------------------------------------
# metadata probes behind the dimension-order suggestion
# ---------------------------------------------------------------------------
class TestZarrProbes:
    def test_info_for_a_plain_array_store(self, tmp_path):
        path = str(tmp_path / "a.zarr")
        zarr.create_array(path, shape=(3, 5), dtype="uint8", chunks=(3, 5))
        info = fs.get_zarr_info(path)
        assert info["num_arrays"] == 1
        assert info["shape"] == (3, 5)
        assert info["dtype"] == "uint8"

    def test_channel_axis_with_missing_dataset_is_single_channel(
        self, tmp_path, capsys
    ):
        path = _write_ome_zarr_group(
            tmp_path / "a.zarr", {}, "cyx", dataset_paths=["0"]
        )
        assert fs.detect_channels_from_zarr_path(path) == (1, None)
        assert "metadata read failed" in capsys.readouterr().out

    def test_malformed_multiscales_have_no_axes(self, tmp_path):
        path = tmp_path / "a.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["multiscales"] = ["not a dict"]
        assert fs.detect_axes_from_zarr_path(str(path)) is None

    def test_corrupt_zarr_json_is_ignored(self, tmp_path):
        (tmp_path / "zarr.json").write_text("{not json")
        assert fs._read_zarr_root_attrs(str(tmp_path)) == {}

    @pytest.mark.parametrize("dataset", ["0", "s0"])
    def test_shape_of_an_ome_pyramid_is_its_full_resolution(
        self, tmp_path, dataset
    ):
        path = _write_ome_zarr_group(
            tmp_path / "p.zarr",
            {dataset: np.zeros((3, 8, 8), np.uint8), "1": np.zeros((3, 4, 4), np.uint8)},
            "zyx",
            dataset_paths=[dataset, "1"],
        )
        assert fs.probe_file_shape(path) == (3, 8, 8)

    def test_shape_of_a_group_without_metadata_uses_array_0(self, tmp_path):
        root = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
        root.create_array("0", shape=(6, 7), dtype="uint8")
        assert fs.probe_file_shape(str(tmp_path / "g.zarr")) == (6, 7)

    def test_shape_of_an_empty_group_is_unknown(self, tmp_path):
        zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
        assert fs.probe_file_shape(str(tmp_path / "g.zarr")) is None

    def test_shape_of_other_formats_is_unknown(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(b"png")
        assert fs.probe_file_shape(str(path)) is None

    def test_2d_store_without_axes_is_suggested_yx(self, tmp_path):
        path = str(tmp_path / "a.zarr")
        zarr.create_array(path, shape=(8, 8), dtype="uint8")
        assert fs.suggest_dimension_order(path) == "YX"


class TestSaveAsZarrCompression:
    def test_gzip_is_used_when_zstd_is_unavailable(self, tmp_path, monkeypatch):
        pytest.importorskip("ome_zarr")
        import zarr.codecs

        def no_zstd(*args, **kwargs):
            raise TypeError("zstd not built")

        monkeypatch.setattr(zarr.codecs, "ZstdCodec", no_zstd)
        data = np.arange(64, dtype=np.uint16).reshape(8, 8)
        target = str(tmp_path / "x.zarr")

        fs.save_as_zarr(data, target)

        group = zarr.open_group(target, mode="r")
        stored = group[fs._resolve_dataset_path(group, None)]
        np.testing.assert_array_equal(stored[:], data)
        assert type(stored.compressors[0]).__name__ == "GzipCodec"


class TestLazyTiff:
    def test_small_stack_is_read_densely(self, tmp_path):
        path = _write_tif(tmp_path / "s.tif", np.zeros((3, 8, 8), np.uint8))
        assert fs._lazy_load_tiff(path) is None


# ---------------------------------------------------------------------------
# small entry points
# ---------------------------------------------------------------------------
class TestParameterConversion:
    @pytest.mark.parametrize(
        "text,expected",
        [("True", True), (" yes ", True), ("1", True), ("false", False),
         ("No", False), ("", False)],
    )
    def test_boolean_text(self, text, expected):
        assert fs.convert_parameter_value({"type": bool}, text) is expected


class TestFindingFiles:
    def test_blank_suffix_means_tif(self, tmp_path):
        (tmp_path / "a.tif").write_bytes(b"")
        (tmp_path / "b.png").write_bytes(b"")
        found = fs.find_matching_files(str(tmp_path), " , ")
        assert [os.path.basename(f) for f in found] == ["a.tif"]

    def test_invalid_folder_is_reported(self, tmp_path, qapp):
        viewer = _Viewer()
        missing = str(tmp_path / "nope")
        assert fs.file_selector(viewer, missing, ".tif") == []
        assert viewer.status == f"Invalid input folder: {missing}"


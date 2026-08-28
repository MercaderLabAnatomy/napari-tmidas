"""Coverage tests for the standalone ``_processing_worker`` module.

This is the *secondary* worker: the napari widget actually drives the
``ProcessingWorker`` defined in ``_file_selector.py``.  These tests exercise
this module on its own terms -- tiny real ``.tif`` files under ``tmp_path``
and plain local processing functions -- pinning the run loop, per-file error
capture, channel selection, output-name construction and the save helpers.
"""

import importlib.util
import inspect
import os
import sys
import types

import numpy as np
import pytest
import tifffile

import napari_tmidas._file_selector as fs
import napari_tmidas._processing_worker as pw
from napari_tmidas._processing_worker import (
    ProcessingWorker,
    _best_tiff_compression,
    is_label_image,
    load_image_file,
    load_image_file_lazy,
    save_image_file,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _write_tif(path, array):
    """Write ``array`` to ``path`` and return the path as a string."""
    tifffile.imwrite(str(path), array)
    return str(path)


def _identity(image, **kwargs):
    """Trivial processing function: returns the input as an array."""
    return np.asarray(image)


def _make_worker(tmp_path, files, func, params=None, suffix="_proc.tif"):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    worker = ProcessingWorker(
        list(files),
        func,
        params,
        str(out),
        ".tif",
        suffix,
    )
    worker.thread_count = 1
    return worker


def _record(worker):
    """Connect every worker signal to a recorder and return the buckets."""
    rec = {
        "progress": [],
        "processed": [],
        "errors": [],
        "finished": [],
    }
    worker.progress_updated.connect(rec["progress"].append)
    worker.file_processed.connect(rec["processed"].append)
    worker.error_occurred.connect(
        lambda path, msg: rec["errors"].append((path, msg))
    )
    worker.processing_finished.connect(lambda: rec["finished"].append(True))
    return rec


class _RecordingTifffile:
    """Proxy around ``tifffile`` that records how ``imwrite`` was called.

    ``save_image_file`` has two mutually exclusive write paths that produce
    byte-identical files, so the only way to tell them apart is to look at the
    call itself: the streaming path hands tifffile a *generator* plus
    ``shape``/``dtype``/``photometric``, the plain path hands it a
    materialized array positionally.
    """

    def __init__(self):
        self.calls = []
        self.TiffFileError = tifffile.TiffFileError

    def imwrite(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        return tifffile.imwrite(*args, **kwargs)


def _make_zarr_group(path, array):
    """Create a minimal zarr group holding ``array`` under key ``0``."""
    import zarr

    root = zarr.open_group(str(path), mode="w")
    try:
        arr = root.create_array("0", shape=array.shape, dtype=array.dtype)
    except AttributeError:  # pragma: no cover - zarr v2 API
        arr = root.create_dataset("0", shape=array.shape, dtype=array.dtype)
    arr[:] = array
    return str(path)


# --------------------------------------------------------------------------


class TestConstruction:
    """``__init__`` takes six positional arguments in a fixed order."""

    def test_init_stores_arguments_in_order(self):
        """Each positional argument lands on its own attribute."""
        worker = ProcessingWorker(
            ["a.tif"], _identity, {"gain": 1}, "/out", ".tif", "_p.tif"
        )

        assert worker.file_list == ["a.tif"]
        assert worker.processing_func is _identity
        assert worker.param_values == {"gain": 1}
        assert worker.output_folder == "/out"
        assert worker.input_suffix == ".tif"
        assert worker.output_suffix == "_p.tif"
        assert worker.stop_requested is False
        # Default pool size: a quarter of the CPUs, never zero.
        assert worker.thread_count == max(1, (os.cpu_count() or 4) // 4)
        assert worker.thread_count >= 1


class TestRunLoop:
    """``ProcessingWorker.run`` drives the pool and emits the four signals."""

    def test_run_emits_progress_results_and_finished(self, tmp_path):
        """Every file yields one file_processed and a progress percentage."""
        files = [
            _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8)),
            _write_tif(tmp_path / "b.tif", np.ones((4, 4), np.uint8) * 2),
        ]
        worker = _make_worker(tmp_path, files, _identity)
        rec = _record(worker)

        worker.run()

        assert rec["progress"] == [50, 100]
        assert rec["finished"] == [True]
        assert rec["errors"] == []
        originals = {r["original_file"] for r in rec["processed"]}
        assert originals == set(files)
        for result in rec["processed"]:
            assert os.path.exists(result["processed_file"])

    def test_run_captures_per_file_error(self, tmp_path):
        """A ValueError in one file becomes error_occurred, others survive."""
        good = _write_tif(tmp_path / "good.tif", np.ones((4, 4), np.uint8))
        bad = _write_tif(tmp_path / "bad.tif", np.ones((4, 4), np.uint8))

        def picky(image, **kwargs):
            if "bad" in kwargs.get("_source_filepath", ""):
                raise ValueError("boom")
            return np.asarray(image)

        worker = _make_worker(tmp_path, [good, bad], picky)
        rec = _record(worker)

        worker.run()

        assert [r["original_file"] for r in rec["processed"]] == [good]
        assert len(rec["errors"]) == 1
        assert rec["errors"][0][0] == bad
        assert "boom" in rec["errors"][0][1]
        assert rec["finished"] == [True]

    def test_run_captures_oserror(self, tmp_path):
        """OSError is one of the caught types, not a crash of the run loop."""
        path = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8))

        def exploder(image, **kwargs):
            raise OSError("disk gone")

        worker = _make_worker(tmp_path, [path], exploder)
        rec = _record(worker)

        worker.run()

        assert rec["errors"] == [(path, "disk gone")]
        assert rec["processed"] == []
        assert rec["progress"] == [100]

    def test_run_stops_when_cancelled(self, tmp_path):
        """stop() short-circuits the completion loop but still finishes."""
        files = [
            _write_tif(tmp_path / f"{i}.tif", np.ones((4, 4), np.uint8))
            for i in range(3)
        ]
        worker = _make_worker(tmp_path, files, _identity)
        rec = _record(worker)
        worker.stop()

        worker.run()

        assert worker.stop_requested is True
        assert rec["processed"] == []
        assert rec["progress"] == []
        assert rec["finished"] == [True]

    def test_run_with_empty_file_list(self, tmp_path):
        """No files: no division by zero, just the finished signal."""
        worker = _make_worker(tmp_path, [], _identity)
        rec = _record(worker)

        worker.run()

        assert rec == {
            "progress": [],
            "processed": [],
            "errors": [],
            "finished": [True],
        }


class TestLoadingBranches:
    """``process_file`` picks the right loader and normalises the input."""

    def test_loads_from_path_marker_uses_lazy_loader(
        self, tmp_path, monkeypatch
    ):
        """A ``_loads_from_path`` function is fed the lazy loader's array."""
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _write_tif(tmp_path / "a.tif", data)
        seen = []
        real_lazy = pw.load_image_file_lazy

        def spy(filepath):
            seen.append(filepath)
            return real_lazy(filepath)

        monkeypatch.setattr(pw, "load_image_file_lazy", spy)
        monkeypatch.setattr(
            pw, "load_image_file", lambda p: pytest.fail("eager loader used")
        )

        def path_func(image, **kwargs):
            return np.asarray(image)

        path_func._loads_from_path = True

        worker = _make_worker(tmp_path, [path], path_func)
        result = worker.process_file(path)

        assert seen == [path]
        # The lazy handle must still carry the real pixels through to disk.
        assert np.array_equal(tifffile.imread(result["processed_file"]), data)

    def test_multi_layer_input_picks_first_image_layer(
        self, tmp_path, monkeypatch
    ):
        """A list of napari layer tuples resolves to the first image layer."""
        labels = np.zeros((4, 4), np.uint32)
        image = np.full((4, 4), 7, np.uint8)
        monkeypatch.setattr(
            pw,
            "load_image_file",
            lambda p: [
                (labels, {}, "labels"),
                (image, {}, "image"),
            ],
        )
        path = str(tmp_path / "multi.tif")
        worker = _make_worker(tmp_path, [path], _identity)

        result = worker.process_file(path)

        saved = tifffile.imread(result["processed_file"])
        assert np.array_equal(saved, image)

    def test_multi_layer_without_image_layer_uses_first_entry(
        self, tmp_path, monkeypatch
    ):
        """With no ``image`` layer the first layer's data is used instead."""
        first = np.full((4, 4), 3, np.uint8)
        monkeypatch.setattr(
            pw,
            "load_image_file",
            lambda p: [
                (first, {}, "labels"),
                (np.zeros((4, 4), np.uint8), {}, "labels"),
            ],
        )
        path = str(tmp_path / "multi.tif")
        worker = _make_worker(tmp_path, [path], _identity)

        result = worker.process_file(path)

        saved = tifffile.imread(result["processed_file"])
        assert np.array_equal(saved, first)

    def test_input_without_dtype_defaults_to_float32(
        self, tmp_path, monkeypatch
    ):
        """Objects lacking ``.dtype``/``.shape`` are saved as float32."""
        monkeypatch.setattr(pw, "load_image_file", lambda p: ((1, 2), (3, 4)))
        path = str(tmp_path / "tuple.tif")
        worker = _make_worker(tmp_path, [path], _identity)

        result = worker.process_file(path)

        saved = tifffile.imread(result["processed_file"])
        assert saved.dtype == np.float32
        assert np.array_equal(saved, np.array([[1, 2], [3, 4]]))


class TestParameterFiltering:
    """Private ``_``-prefixed params are only passed to accepting functions."""

    def test_var_kwargs_function_receives_private_params(self, tmp_path):
        """A ``**kwargs`` function sees source path, folder and suffix.

        ``channel`` is consumed by the worker itself and must *not* reach the
        processing function, which would otherwise choke on an unexpected
        keyword or double-select the channel.
        """
        path = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8))
        seen = {}

        def grabber(image, **kwargs):
            seen.update(kwargs)
            return np.asarray(image)

        worker = _make_worker(
            tmp_path, [path], grabber, {"gain": 2, "channel": "all"}
        )
        worker.process_file(path)

        assert seen == {
            "_source_filepath": path,
            "_output_folder": worker.output_folder,
            "_output_suffix": "_proc.tif",
            "gain": 2,
        }

    def test_narrow_signature_drops_unaccepted_params(self, tmp_path):
        """A function without ``**kwargs`` only gets its declared params."""
        path = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8))
        seen = {}

        def narrow(image, gain=1, _output_suffix=""):
            seen["gain"] = gain
            seen["_output_suffix"] = _output_suffix
            return np.asarray(image) * gain

        worker = _make_worker(tmp_path, [path], narrow, {"gain": 3})
        result = worker.process_file(path)

        assert seen == {"gain": 3, "_output_suffix": "_proc.tif"}
        saved = tifffile.imread(result["processed_file"])
        assert np.array_equal(saved, np.full((4, 4), 3, np.uint8))

    def test_unsignable_callable_drops_private_params(self, tmp_path):
        """If ``inspect.signature`` fails, ``_`` params are dropped."""

        class Unsignable:
            __name__ = "unsignable"

            def __init__(self):
                self.seen = None

            @property
            def __signature__(self):
                raise ValueError("no signature available")

            def __call__(self, image, **kwargs):
                self.seen = dict(kwargs)
                return np.asarray(image)

        path = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8))
        func = Unsignable()
        worker = _make_worker(tmp_path, [path], func, {"gain": 5})

        worker.process_file(path)

        assert func.seen == {"gain": 5}


class TestChannelSelection:
    """The ``channel`` parameter selects channels before processing."""

    def test_channel_all_splits_array_channels(self, tmp_path):
        """``channel='all'`` writes one ``_ch<i>`` file per channel."""
        data = np.stack([np.full((4, 4), v, np.uint8) for v in (1, 2, 3)])
        path = _write_tif(tmp_path / "img.tif", data)
        worker = _make_worker(tmp_path, [path], _identity, {"channel": "all"})

        result = worker.process_file(path)

        assert result["original_file"] == path
        names = sorted(os.path.basename(p) for p in result["processed_files"])
        assert names == [
            "img_ch0_proc.tif",
            "img_ch1_proc.tif",
            "img_ch2_proc.tif",
        ]
        assert {os.path.dirname(p) for p in result["processed_files"]} == {
            worker.output_folder
        }
        for idx, out in enumerate(sorted(result["processed_files"])):
            saved = tifffile.imread(out)
            assert saved.shape == (4, 4)
            assert np.all(saved == idx + 1)

    def test_single_channel_index_is_extracted(self, tmp_path):
        """An in-range int channel yields one file with that channel only."""
        data = np.stack([np.full((4, 4), v, np.uint8) for v in (1, 2, 3)])
        path = _write_tif(tmp_path / "img.tif", data)
        worker = _make_worker(tmp_path, [path], _identity, {"channel": 1})

        result = worker.process_file(path)

        saved = tifffile.imread(result["processed_file"])
        assert saved.shape == (4, 4)
        assert np.all(saved == 2)

    def test_out_of_range_channel_processes_whole_image(self, tmp_path):
        """An out-of-range channel falls back to the entire image."""
        data = np.stack([np.full((4, 4), v, np.uint8) for v in (1, 2, 3)])
        path = _write_tif(tmp_path / "img.tif", data)
        worker = _make_worker(tmp_path, [path], _identity, {"channel": 9})

        result = worker.process_file(path)

        assert "processed_files" not in result
        saved = tifffile.imread(result["processed_file"])
        assert np.array_equal(saved, data)

    def test_single_channel_image_ignores_channel_param(self, tmp_path):
        """A 2D image has one channel, so the whole image is processed."""
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _write_tif(tmp_path / "img.tif", data)
        worker = _make_worker(tmp_path, [path], _identity, {"channel": "all"})

        result = worker.process_file(path)

        assert os.path.basename(result["processed_file"]) == "img_proc.tif"
        assert np.array_equal(
            tifffile.imread(result["processed_file"]), data
        )

    def test_single_channel_axis_is_not_split(self, tmp_path, monkeypatch):
        """``num_channels == 1`` must not enter the per-channel split.

        The guard is ``> 1``: with ``>= 1`` a single-channel image would be
        sliced along the channel axis and lose its leading dimension.
        """
        data = np.stack([np.full((4, 4), v, np.uint8) for v in (7,)])
        path = _write_tif(tmp_path / "img.tif", data)
        monkeypatch.setattr(fs, "detect_channels_in_image", lambda d: (1, 0))
        worker = _make_worker(tmp_path, [path], _identity, {"channel": "all"})

        result = worker.process_file(path)

        assert os.path.basename(result["processed_file"]) == "img_proc.tif"
        assert np.array_equal(
            tifffile.imread(result["processed_file"]), data
        )

    def test_channel_axis_none_processes_whole_image(
        self, tmp_path, monkeypatch
    ):
        """num_channels > 1 with no channel axis still processes one image."""
        data = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)
        path = _write_tif(tmp_path / "img.tif", data)
        monkeypatch.setattr(
            fs, "detect_channels_in_image", lambda data: (2, None)
        )
        worker = _make_worker(tmp_path, [path], _identity, {"channel": 0})

        result = worker.process_file(path)

        assert "processed_files" not in result
        assert np.array_equal(
            tifffile.imread(result["processed_file"]), data
        )

    def test_import_error_falls_back_to_whole_image(
        self, tmp_path, monkeypatch
    ):
        """If the detector cannot be imported the image is used unchanged."""
        data = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
        path = _write_tif(tmp_path / "img.tif", data)
        monkeypatch.delattr(fs, "detect_channels_in_image")
        worker = _make_worker(tmp_path, [path], _identity, {"channel": "all"})

        result = worker.process_file(path)

        # No ``_ch<i>`` split happened: one file holding the whole stack.
        assert "processed_files" not in result
        assert os.path.basename(result["processed_file"]) == "img_proc.tif"
        assert np.array_equal(
            tifffile.imread(result["processed_file"]), data
        )


class TestChannelSelectionSeparateLayers:
    """channel_axis == -1 means each image layer is its own channel."""

    @staticmethod
    def _layers():
        return [
            (np.full((4, 4), 10, np.uint8), {}, "image"),
            (np.zeros((4, 4), np.uint32), {}, "labels"),
            (np.full((4, 4), 20, np.uint8), {}, "image"),
        ]

    def _worker(self, tmp_path, monkeypatch, channel):
        layers = self._layers()
        monkeypatch.setattr(pw, "load_image_file", lambda p: layers)
        monkeypatch.setattr(
            fs, "detect_channels_in_image", lambda data: (2, -1)
        )
        path = str(tmp_path / "multi.tif")
        worker = _make_worker(
            tmp_path, [path], _identity, {"channel": channel}
        )
        return worker, path

    def test_all_layers_are_processed(self, tmp_path, monkeypatch):
        """``channel='all'`` writes one file per image layer."""
        worker, path = self._worker(tmp_path, monkeypatch, "all")

        result = worker.process_file(path)

        names = sorted(os.path.basename(p) for p in result["processed_files"])
        assert names == ["multi_ch0_proc.tif", "multi_ch2_proc.tif"]
        values = sorted(
            int(tifffile.imread(p).flat[0]) for p in result["processed_files"]
        )
        assert values == [10, 20]

    def test_selected_layer_only(self, tmp_path, monkeypatch):
        """An int channel picks the n-th *image* layer, skipping labels."""
        worker, path = self._worker(tmp_path, monkeypatch, 1)

        result = worker.process_file(path)

        saved = tifffile.imread(result["processed_file"])
        assert np.all(saved == 20)

    def test_invalid_layer_selection_uses_first_layer(
        self, tmp_path, monkeypatch
    ):
        """A bogus channel value falls back to the first image layer."""
        worker, path = self._worker(tmp_path, monkeypatch, "bogus")

        result = worker.process_file(path)

        saved = tifffile.imread(result["processed_file"])
        assert np.all(saved == 10)


class TestZarrChannelDetection:
    """Zarr inputs use the path-based channel detector."""

    def _prepare(self, tmp_path, monkeypatch, dirname):
        folder = tmp_path / dirname
        folder.mkdir()
        (folder / ".zattrs").write_text("{}")
        data = np.stack([np.full((4, 4), v, np.uint8) for v in (5, 6)])
        monkeypatch.setattr(pw, "load_image_file", lambda p: data)
        calls = []

        def fake_detect(path):
            calls.append(path)
            return (2, 0)

        monkeypatch.setattr(fs, "detect_channels_from_zarr_path", fake_detect)
        return str(folder), calls

    def test_zarr_suffix_uses_path_detector(self, tmp_path, monkeypatch):
        """A ``.zarr`` path routes to detect_channels_from_zarr_path."""
        path, calls = self._prepare(tmp_path, monkeypatch, "vol.zarr")
        worker = _make_worker(tmp_path, [path], _identity, {"channel": 1})

        result = worker.process_file(path)

        assert calls == [path]
        saved = tifffile.imread(result["processed_file"])
        assert saved.shape == (4, 4)
        assert np.all(saved == 6)

    def test_zattrs_directory_uses_path_detector(self, tmp_path, monkeypatch):
        """A directory holding ``.zattrs`` is treated as zarr too."""
        path, calls = self._prepare(tmp_path, monkeypatch, "vol.ome")
        worker = _make_worker(tmp_path, [path], _identity, {"channel": 0})

        result = worker.process_file(path)

        assert calls == [path]
        assert np.all(tifffile.imread(result["processed_file"]) == 5)


class TestMultiChannelSaving:
    """The multi-image save branch skips unsaveable per-channel results."""

    @staticmethod
    def _three_channel(tmp_path):
        data = np.stack([np.full((4, 4), v, np.uint8) for v in (1, 2, 3)])
        return _write_tif(tmp_path / "img.tif", data)

    def test_none_results_are_skipped(self, tmp_path):
        """Channels whose result is None produce no output file."""
        path = self._three_channel(tmp_path)

        def only_first(image, **kwargs):
            return None

        worker = _make_worker(tmp_path, [path], only_first, {"channel": "all"})

        result = worker.process_file(path)

        assert result["processed_files"] == []

    def test_folder_function_writes_nothing_per_channel(self, tmp_path):
        """Folder functions are recognised by name and skip saving."""
        path = self._three_channel(tmp_path)

        def track_objects(image, **kwargs):
            return np.asarray(image)

        worker = _make_worker(
            tmp_path, [path], track_objects, {"channel": "all"}
        )

        result = worker.process_file(path)

        assert result["processed_files"] == []

    def test_returned_paths_are_passed_through(self, tmp_path):
        """A per-channel result that is an existing path is not re-saved."""
        path = self._three_channel(tmp_path)
        sentinel = _write_tif(
            tmp_path / "already.tif", np.zeros((2, 2), np.uint8)
        )

        def path_returner(image, **kwargs):
            return sentinel

        worker = _make_worker(
            tmp_path, [path], path_returner, {"channel": "all"}
        )

        result = worker.process_file(path)

        assert result["processed_files"] == [sentinel] * 3


class TestOutputConstruction:
    """Single-image results, path results and multi-output results."""

    def test_single_output_uses_basename_plus_suffix(self, tmp_path):
        """The output name is ``<stem><output_suffix>`` in output_folder."""
        path = _write_tif(tmp_path / "stack.tif", np.ones((4, 4), np.uint8))
        worker = _make_worker(tmp_path, [path], _identity, suffix="_seg.tif")

        result = worker.process_file(path)

        assert os.path.basename(result["processed_file"]) == "stack_seg.tif"
        assert os.path.dirname(result["processed_file"]) == (
            worker.output_folder
        )

    def test_string_result_pointing_at_a_file_is_returned(self, tmp_path):
        """A function that writes its own file returns that path verbatim."""
        path = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8))
        written = _write_tif(tmp_path / "own.tif", np.ones((2, 2), np.uint8))

        def writer(image, **kwargs):
            return written

        worker = _make_worker(tmp_path, [path], writer)

        result = worker.process_file(path)

        assert result == {"original_file": path, "processed_file": written}

    def test_string_result_pointing_at_a_directory_is_returned(self, tmp_path):
        """Directory outputs (e.g. zarr stores) are returned unchanged."""
        path = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint8))
        outdir = tmp_path / "store.zarr"
        outdir.mkdir()

        def writer(image, **kwargs):
            return str(outdir)

        worker = _make_worker(tmp_path, [path], writer)

        result = worker.process_file(path)

        assert result["processed_file"] == str(outdir)

    def test_tuple_output_is_saved_as_numbered_channels(self, tmp_path):
        """A multi-output function writes ``_ch1``, ``_ch2``... files."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def splitter(image, **kwargs):
            return (
                np.full((4, 4), 1, np.uint8),
                np.full((4, 4), 2, np.uint8),
            )

        worker = _make_worker(tmp_path, [path], splitter)

        result = worker.process_file(path)

        names = [os.path.basename(p) for p in result["processed_files"]]
        assert names == ["img_ch1_proc.tif", "img_ch2_proc.tif"]
        assert np.all(tifffile.imread(result["processed_files"][1]) == 2)

    def test_non_array_entries_in_multi_output_are_skipped(self, tmp_path):
        """Non-ndarray members of a multi-output tuple are not written."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def splitter(image, **kwargs):
            return (np.full((4, 4), 1, np.uint8), "not an array", None)

        worker = _make_worker(tmp_path, [path], splitter)

        result = worker.process_file(path)

        assert len(result["processed_files"]) == 1
        assert os.path.basename(result["processed_files"][0]) == (
            "img_ch1_proc.tif"
        )

    def test_layer_subdivision_names_and_dtype(self, tmp_path):
        """Three outputs with the ``_layer`` suffix become inner/mid/outer."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def subdivide(image, **kwargs):
            return (
                np.full((4, 4), 1, np.uint8),
                np.full((4, 4), 2, np.uint8),
                np.full((4, 4), 3, np.uint8),
            )

        worker = _make_worker(tmp_path, [path], subdivide, suffix="_layer")

        result = worker.process_file(path)

        names = [os.path.basename(p) for p in result["processed_files"]]
        assert names == [
            "img_inner.tif",
            "img_middle.tif",
            "img_outer.tif",
        ]
        for out in result["processed_files"]:
            assert tifffile.imread(out).dtype == np.uint32

    def test_layer_subdivision_skips_non_arrays(self, tmp_path):
        """Non-array entries of a ``_layer`` triple are skipped."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def subdivide(image, **kwargs):
            return (np.zeros((4, 4), np.uint8), None, "nope")

        worker = _make_worker(tmp_path, [path], subdivide, suffix="_layer")

        result = worker.process_file(path)

        assert len(result["processed_files"]) == 1
        assert result["processed_files"][0].endswith("img_inner.tif")

    def test_folder_function_returns_no_output_file(self, tmp_path):
        """Folder-level functions report ``processed_file`` as None."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def merge_timepoints(image, **kwargs):
            return np.asarray(image)

        worker = _make_worker(tmp_path, [path], merge_timepoints)

        result = worker.process_file(path)

        assert result == {"original_file": path, "processed_file": None}
        assert os.listdir(worker.output_folder) == []

    def test_none_result_returns_no_output_file(self, tmp_path):
        """A function returning None produces no file and a None entry."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))
        worker = _make_worker(tmp_path, [path], lambda image, **kw: None)

        result = worker.process_file(path)

        assert result == {"original_file": path, "processed_file": None}
        assert os.listdir(worker.output_folder) == []

    def test_single_element_tuple_is_not_treated_as_multi_output(
        self, tmp_path
    ):
        """A length-1 tuple falls through to the single-output branch."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def one_tuple(image, **kwargs):
            return (np.full((4, 4), 9, np.uint8),)

        worker = _make_worker(tmp_path, [path], one_tuple)

        result = worker.process_file(path)

        assert "processed_files" not in result
        assert os.path.basename(result["processed_file"]) == "img_proc.tif"
        saved = tifffile.imread(result["processed_file"])
        # Saved via ``np.asarray(tuple)``, so the 1-tuple keeps a leading axis.
        assert saved.shape == (1, 4, 4)
        assert np.all(saved == 9)


class TestProcessFileErrors:
    """``process_file`` prints a traceback and re-raises."""

    def test_exception_is_reraised(self, tmp_path, capsys):
        """The original exception reaches the caller after being logged."""
        path = _write_tif(tmp_path / "img.tif", np.ones((4, 4), np.uint8))

        def boom(image, **kwargs):
            raise RuntimeError("kaboom")

        worker = _make_worker(tmp_path, [path], boom)

        with pytest.raises(RuntimeError, match="kaboom"):
            worker.process_file(path)

        assert "Error processing" in capsys.readouterr().out


class TestLoadImageFileFallback:
    """``load_image_file`` has a stand-alone fallback implementation."""

    def test_tiff_fallback(self, tmp_path, monkeypatch):
        """Without the _file_selector loader, tifffile reads the file."""
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _write_tif(tmp_path / "a.tif", data)
        monkeypatch.delattr(fs, "load_image_file")

        assert np.array_equal(load_image_file(path), data)

    def test_numpy_fallback(self, tmp_path, monkeypatch):
        """Without tifffile the last resort is ``np.load``."""
        data = np.arange(6, dtype=np.int16).reshape(2, 3)
        path = tmp_path / "a.npy"
        np.save(str(path), data)
        monkeypatch.delattr(fs, "load_image_file")
        monkeypatch.setattr(pw, "_HAS_TIFFFILE", False)

        assert np.array_equal(load_image_file(str(path)), data)

    def test_zarr_group_fallback(self, tmp_path, monkeypatch):
        """A zarr group falls back to reading its first array."""
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _make_zarr_group(tmp_path / "vol.zarr", data)
        monkeypatch.delattr(fs, "load_image_file")

        assert np.array_equal(load_image_file(path), data)

    def test_unreadable_zarr_falls_through(self, tmp_path, monkeypatch):
        """A broken zarr path swallows the error and tries the next reader."""
        data = np.arange(4, dtype=np.uint8)
        path = tmp_path / "broken.zarr"
        with open(path, "wb") as handle:
            np.save(handle, data)
        monkeypatch.delattr(fs, "load_image_file")
        monkeypatch.setattr(pw, "_HAS_TIFFFILE", False)

        assert np.array_equal(load_image_file(str(path)), data)


class TestLoadImageFileLazy:
    """The lazy loader degrades gracefully to the eager one."""

    def test_empty_reader_result_falls_back(
        self, tmp_path, monkeypatch, capsys
    ):
        """An empty reader result means the eager loader is used.

        The ``if results:`` guard has to do the work here: without it the
        ``results[0][0]`` IndexError would be swallowed by the blanket
        ``except`` and reach the same eager fallback, so the array alone
        proves nothing.  The absence of the failure message does.
        """
        import napari_tmidas._reader as reader

        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _write_tif(tmp_path / "a.tif", data)
        monkeypatch.setattr(reader, "tiff_reader_function", lambda p: [])
        eager = []
        monkeypatch.setattr(
            pw,
            "load_image_file",
            lambda p: eager.append(p) or data,
        )

        result = load_image_file_lazy(path)

        assert eager == [path]
        assert np.array_equal(np.asarray(result), data)
        assert "Lazy load failed" not in capsys.readouterr().out

    def test_reader_exception_falls_back(self, tmp_path, monkeypatch, capsys):
        """A raising lazy reader is reported and the eager loader wins."""
        import napari_tmidas._reader as reader

        data = np.ones((4, 4), np.uint8)
        path = _write_tif(tmp_path / "a.tif", data)

        def boom(filepath):
            raise ValueError("no lazy reader")

        monkeypatch.setattr(reader, "tiff_reader_function", boom)

        result = np.asarray(load_image_file_lazy(path))

        assert np.array_equal(result, data)
        assert "Lazy load failed" in capsys.readouterr().out

    def test_zarr_uses_basic_loader(self, tmp_path, monkeypatch):
        """A ``.zarr`` path is routed through ``load_zarr_basic``.

        The eager fallback reaches ``load_zarr_basic`` too, so it is barred
        outright: only the lazy branch can satisfy this test.
        """
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _make_zarr_group(tmp_path / "vol.zarr", data)
        calls = []

        def fake_basic(filepath):
            calls.append(filepath)
            return data

        monkeypatch.setattr(fs, "load_zarr_basic", fake_basic)
        monkeypatch.setattr(
            pw,
            "load_image_file",
            lambda p: pytest.fail("eager loader used for .zarr"),
        )

        assert np.array_equal(load_image_file_lazy(path), data)
        assert calls == [path]

    def test_unknown_extension_uses_eager_loader(self, tmp_path, monkeypatch):
        """Non tif/zarr paths skip the lazy branch entirely."""
        sentinel = np.arange(4, dtype=np.uint8).reshape(2, 2)
        called = []
        monkeypatch.setattr(
            pw,
            "load_image_file",
            lambda p: called.append(p) or sentinel,
        )

        result = load_image_file_lazy(str(tmp_path / "a.png"))

        assert called == [str(tmp_path / "a.png")]
        # Handed straight back, not re-wrapped or copied.
        assert result is sentinel


class TestIsLabelImage:
    """Label detection mirrors napari's dtype-based guess."""

    @pytest.mark.parametrize(
        "dtype", [np.int32, np.uint32, np.int64, np.uint64]
    )
    def test_integer_label_dtypes(self, dtype):
        """The four label dtypes are recognised."""
        assert is_label_image(np.zeros((2, 2), dtype)) is True

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
    def test_non_label_dtypes(self, dtype):
        """Other dtypes are plain images."""
        assert is_label_image(np.zeros((2, 2), dtype)) is False

    def test_object_without_dtype(self):
        """Anything without a dtype attribute is not a label image."""
        assert is_label_image([[1, 2], [3, 4]]) is False


class TestSaveImageFile:
    """``save_image_file`` dtype selection and dask streaming."""

    def test_requires_tifffile(self, tmp_path, monkeypatch):
        """Without tifffile the save raises ImportError."""
        monkeypatch.setattr(pw, "_HAS_TIFFFILE", False)

        with pytest.raises(ImportError, match="tifffile"):
            save_image_file(np.zeros((2, 2)), str(tmp_path / "a.tif"))

    def test_label_dtype_preserved_as_uint32(self, tmp_path):
        """A label-dtype input without explicit dtype is saved as uint32."""
        out = tmp_path / "labels.tif"
        save_image_file(np.arange(4, dtype=np.int64).reshape(2, 2), str(out))

        saved = tifffile.imread(str(out))
        assert saved.dtype == np.uint32
        assert np.array_equal(saved, [[0, 1], [2, 3]])

    def test_non_label_dtype_is_kept(self, tmp_path):
        """A non-label input keeps its own dtype."""
        out = tmp_path / "img.tif"
        save_image_file(np.ones((2, 2), np.float32), str(out))

        assert tifffile.imread(str(out)).dtype == np.float32

    def test_explicit_dtype_wins(self, tmp_path):
        """An explicit dtype overrides the label heuristic."""
        out = tmp_path / "img.tif"
        save_image_file(
            np.arange(4, dtype=np.uint32).reshape(2, 2), str(out), np.uint16
        )

        saved = tifffile.imread(str(out))
        assert saved.dtype == np.uint16
        assert np.array_equal(saved, [[0, 1], [2, 3]])

    def test_output_is_compressed_and_not_bigtiff(self, tmp_path):
        """Small saves are compressed classic TIFFs, not BigTIFFs."""
        out = tmp_path / "img.tif"
        save_image_file(np.zeros((8, 8), np.uint16), str(out))

        with tifffile.TiffFile(str(out)) as handle:
            assert handle.is_bigtiff is False
            assert (
                handle.pages[0].compression != tifffile.COMPRESSION.NONE
            )

    def test_unmeasurable_shape_does_not_break_saving(self, tmp_path):
        """A bogus ``.shape`` only costs the size estimate, not the write."""

        class Odd:
            shape = "not-a-shape"
            dtype = np.dtype(np.uint8)

            def __array__(self, dtype=None, copy=None):
                return np.arange(4, dtype=np.uint8).reshape(2, 2)

        out = tmp_path / "odd.tif"
        save_image_file(Odd(), str(out), np.uint8)

        assert np.array_equal(tifffile.imread(str(out)), [[0, 1], [2, 3]])

    def test_two_dimensional_dask_array_is_computed(
        self, tmp_path, monkeypatch
    ):
        """2D dask arrays take the compute-then-write path, not streaming.

        Both paths yield the same bytes, so the write call itself is
        inspected: a materialized array passed positionally, with no
        ``data=``/``shape=`` streaming keywords.
        """
        da = pytest.importorskip("dask.array")
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        out = tmp_path / "dask2d.tif"
        proxy = _RecordingTifffile()
        monkeypatch.setattr(pw, "tifffile", proxy)

        save_image_file(da.from_array(data, chunks=(2, 4)), str(out))

        (args, kwargs), = proxy.calls
        assert args[0] == str(out)
        assert isinstance(args[1], np.ndarray)
        assert np.array_equal(args[1], data)
        assert "data" not in kwargs and "shape" not in kwargs

        saved = tifffile.imread(str(out))
        assert saved.dtype == np.uint16
        assert np.array_equal(saved, data)

    def test_higher_dimensional_dask_array_is_streamed(
        self, tmp_path, monkeypatch
    ):
        """>2D dask arrays are written page by page, never materialised.

        The slab generator must stay unconsumed when it reaches tifffile,
        and ``photometric='minisblack'`` must be declared or a leading axis
        of length 3 or 4 would be misread as RGB(A).
        """
        da = pytest.importorskip("dask.array")
        data = np.arange(3 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 4, 4)
        out = tmp_path / "dask4d.tif"
        proxy = _RecordingTifffile()
        monkeypatch.setattr(pw, "tifffile", proxy)

        save_image_file(da.from_array(data, chunks=(1, 2, 4, 4)), str(out))

        (args, kwargs), = proxy.calls
        assert args == (str(out),)
        assert inspect.isgenerator(kwargs["data"])
        assert kwargs["shape"] == (3, 2, 4, 4)
        assert kwargs["dtype"] == np.uint16
        assert kwargs["photometric"] == "minisblack"

        saved = tifffile.imread(str(out))
        assert saved.shape == (3, 2, 4, 4)
        assert saved.dtype == np.uint16
        assert np.array_equal(saved, data)

    def test_streaming_generator_yields_flattened_2d_pages(
        self, tmp_path, monkeypatch
    ):
        """Each slab is flattened into individual (Y, X) pages, in order.

        tifffile pulls one page per iteration, so a TZYX slab has to be
        unrolled; the generator is drained here instead of handed on so the
        page sequence itself can be checked.
        """
        da = pytest.importorskip("dask.array")
        data = np.arange(3 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 4, 4)
        captured = {}

        class _Capturing:
            TiffFileError = tifffile.TiffFileError

            @staticmethod
            def imwrite(*args, **kwargs):
                captured["pages"] = list(kwargs["data"])
                captured["shape"] = kwargs["shape"]

        monkeypatch.setattr(pw, "tifffile", _Capturing)

        save_image_file(
            da.from_array(data, chunks=(1, 2, 4, 4)),
            str(tmp_path / "unused.tif"),
        )

        pages = captured["pages"]
        assert captured["shape"] == (3, 2, 4, 4)
        assert len(pages) == 6
        assert all(p.shape == (4, 4) for p in pages)
        assert all(p.dtype == np.uint16 for p in pages)
        assert np.array_equal(np.stack(pages), data.reshape(6, 4, 4))

    def test_streaming_slab_of_two_dims_is_yielded_whole(
        self, tmp_path, monkeypatch
    ):
        """A 3D array's slabs are already 2D and are yielded unchanged."""
        da = pytest.importorskip("dask.array")
        data = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
        captured = {}

        class _Capturing:
            TiffFileError = tifffile.TiffFileError

            @staticmethod
            def imwrite(*args, **kwargs):
                captured["pages"] = list(kwargs["data"])

        monkeypatch.setattr(pw, "tifffile", _Capturing)

        save_image_file(
            da.from_array(data, chunks=(1, 4, 4)),
            str(tmp_path / "unused.tif"),
        )

        assert np.array_equal(np.stack(captured["pages"]), data)

    def test_bare_zarr_array_fallback(self, tmp_path, monkeypatch):
        """A zarr *array* store (not a group) is read directly."""
        import zarr

        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = str(tmp_path / "bare.zarr")
        arr = zarr.open(path, mode="w", shape=data.shape, dtype=data.dtype)
        arr[:] = data
        monkeypatch.delattr(fs, "load_image_file")

        assert np.array_equal(load_image_file(path), data)


class TestCompressionChoice:
    """``_best_tiff_compression`` prefers zstd but must degrade to zlib."""

    @staticmethod
    def _fake_imagecodecs(available):
        module = types.SimpleNamespace()
        module.ZSTD = types.SimpleNamespace(available=available)
        return module

    def test_zstd_when_imagecodecs_offers_it(self, monkeypatch):
        """An imagecodecs build with ZSTD gives the zstd codec."""
        monkeypatch.setitem(
            sys.modules, "imagecodecs", self._fake_imagecodecs(True)
        )

        assert _best_tiff_compression() == "zstd"

    def test_zlib_when_zstd_unavailable(self, monkeypatch):
        """imagecodecs without ZSTD falls back to zlib."""
        monkeypatch.setitem(
            sys.modules, "imagecodecs", self._fake_imagecodecs(False)
        )

        assert _best_tiff_compression() == "zlib"

    def test_zlib_when_imagecodecs_missing(self, monkeypatch):
        """No imagecodecs at all still yields a codec tifffile can use."""
        monkeypatch.setitem(sys.modules, "imagecodecs", None)

        assert _best_tiff_compression() == "zlib"

    def test_chosen_codec_is_accepted_by_tifffile(self, tmp_path, monkeypatch):
        """Both codecs really round-trip through tifffile."""
        monkeypatch.setitem(sys.modules, "imagecodecs", None)
        data = np.arange(64, dtype=np.uint16).reshape(8, 8)
        out = tmp_path / "zlib.tif"

        save_image_file(data, str(out))

        with tifffile.TiffFile(str(out)) as handle:
            assert (
                handle.pages[0].compression != tifffile.COMPRESSION.NONE
            )
        assert np.array_equal(tifffile.imread(str(out)), data)


class TestWithoutOptionalDependencies:
    """The module still imports when dask, tifffile and qtpy are missing."""

    def test_stub_definitions_are_used(self, monkeypatch):
        """Missing optional imports leave stubs and False feature flags."""
        for name in ("dask.array", "tifffile", "qtpy.QtCore"):
            monkeypatch.setitem(sys.modules, name, None)

        spec = importlib.util.spec_from_file_location(
            "napari_tmidas_pw_no_deps", pw.__file__
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module._HAS_DASK is False
        assert module._HAS_TIFFFILE is False
        assert module._HAS_QTPY is False
        assert module.da is None
        assert module.tifffile is None
        assert module.Signal(int) is None

        worker = module.ProcessingWorker([], _identity, {}, "", "", "")
        assert worker.stop_requested is False
        worker.stop()
        assert worker.stop_requested is True
        assert module.QThread.run(worker) is None

    def test_save_requires_tifffile_in_stub_module(self, monkeypatch):
        """The stub module's save_image_file refuses to run."""
        for name in ("dask.array", "tifffile", "qtpy.QtCore"):
            monkeypatch.setitem(sys.modules, name, None)

        spec = importlib.util.spec_from_file_location(
            "napari_tmidas_pw_no_deps2", pw.__file__
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(ImportError, match="tifffile"):
            module.save_image_file(np.zeros((2, 2)), "unused.tif")

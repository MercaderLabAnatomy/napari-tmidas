"""
Coverage-focused tests for ``_file_conversion``.

The vendor readers (readlif, nd2, pylibCZIrw, tiffslide) are replaced by
in-process fakes so that every loader branch -- dtype/axis normalisation,
size-based strategy selection, metadata fallbacks and the ``except``
arms that wrap failures into ``FileFormatError``/``ConversionError`` --
runs against real module code without needing a real microscope file.

Qt modals are stubbed on the module object (they are imported *into*
``_file_conversion``), because an unstubbed ``QMessageBox`` or
``QFileDialog`` blocks the whole test session.
"""

import contextlib
import csv
import json
import os
import sys
import types

import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr

from napari_tmidas import _file_conversion as fc

pytest.importorskip("pytestqt")


# --------------------------------------------------------------------- #
# Fakes for the vendor readers
# --------------------------------------------------------------------- #
class FakeLifImage:
    """Minimal stand-in for ``readlif``'s image object."""

    def __init__(
        self,
        nt=1,
        nz=1,
        channels=1,
        x_dim=5,
        y_dim=4,
        dtype=np.uint16,
        scale=None,
        none_frames=(),
        bad_shape=(),
        errors=(),
    ):
        self.nt = nt
        self.nz = nz
        self.channels = channels
        self.dims = (x_dim, y_dim)
        self.dtype = dtype
        self.scale = scale
        self.none_frames = set(none_frames)
        self.bad_shape = set(bad_shape)
        self.errors = set(errors)
        self.requested = []

    def get_frame(self, z, t, c):
        self.requested.append((t, z, c))
        key = (t, z, c)
        if key in self.errors:
            raise OSError("unreadable frame")
        if key in self.none_frames:
            return None
        x_dim, y_dim = self.dims
        if key in self.bad_shape:
            return np.zeros((1, 1), dtype=self.dtype)
        value = 100 * t + 10 * z + c + 1
        return np.full((y_dim, x_dim), value, dtype=self.dtype)


class FakeLifFile:
    def __init__(self, images):
        self._images = images

    def get_iter_image(self):
        return iter(self._images)


def lif_factory(images):
    """Build a ``LifFile`` replacement that yields ``images``."""

    def _factory(filepath):
        return FakeLifFile(images)

    return _factory


UINT16 = np.dtype(np.uint16)


class FakeND2File:
    """Stand-in for ``nd2.ND2File`` (a context manager)."""

    def __init__(
        self,
        sizes,
        dtype=UINT16,
        data=None,
        voxel=None,
        is_rgb=False,
        dask_array=None,
        to_dask_error=None,
        support_getitem=True,
    ):
        self.sizes = sizes
        self.dtype = dtype
        self._data = data
        self._voxel = voxel
        self.is_rgb = is_rgb
        self._dask_array = dask_array
        self._to_dask_error = to_dask_error
        self._support_getitem = support_getitem

    def __enter__(self):
        if self._support_getitem:
            return self
        return _ND2NoGetItem(self)

    def __exit__(self, *exc):
        return False

    def voxel_size(self):
        if isinstance(self._voxel, Exception):
            raise self._voxel
        return self._voxel

    def to_dask(self):
        if self._to_dask_error is not None:
            raise self._to_dask_error
        return self._dask_array

    def __getitem__(self, item):
        return self._data[item]


class _ND2NoGetItem:
    """The same handle, but without ``__getitem__`` support."""

    def __init__(self, parent):
        self.sizes = parent.sizes
        self.dtype = parent.dtype
        self._parent = parent

    def to_dask(self):
        return self._parent.to_dask()


class FakeCziDoc:
    def __init__(
        self,
        total_bbox,
        scenes=None,
        metadata=None,
        plane=None,
        read_error=None,
    ):
        self.total_bounding_box = total_bbox
        self.scenes_bounding_rectangle = scenes or {}
        self._metadata = metadata
        self._plane = plane
        self._read_error = read_error
        self.reads = []

    @property
    def metadata(self):
        if isinstance(self._metadata, Exception):
            raise self._metadata
        return self._metadata

    def read(self, plane, scene=None):
        if self._read_error is not None:
            raise self._read_error
        self.reads.append((dict(plane), scene))
        if callable(self._plane):
            return self._plane(plane, scene)
        return self._plane


def czi_module(doc=None, error=None):
    """A ``pyczi`` replacement whose ``open_czi`` yields ``doc``."""

    @contextlib.contextmanager
    def open_czi(filepath):
        if error is not None:
            raise error
        yield doc

    return types.SimpleNamespace(open_czi=open_czi)


class FakeSlide:
    """Stand-in for ``tiffslide.TiffSlide``."""

    level_dimensions = [(6, 4), (3, 2)]
    properties = {
        "tiffslide.series-axes": "YXS",
        "tiffslide.mpp-x": "0.25",
        "tiffslide.mpp-y": "0.5",
    }
    open_error = None

    def __init__(self, filepath):
        if type(self).open_error is not None:
            raise type(self).open_error
        self.filepath = filepath

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read_region(self, location, level, size):
        width, height = size
        return np.full((height, width), level + 1, dtype=np.uint8)


class FakeLayers(list):
    def clear(self):
        del self[:]


class FakeViewer:
    """Just enough viewer surface for the widget's preview/load paths."""

    def __init__(self):
        self.layers = FakeLayers()
        self.status = ""
        self.added = []

    def add_image(self, data, name=None):
        self.added.append((data, name))
        self.layers.append(name)


class StubLoader(fc.FormatLoader):
    """A loader with switchable behaviour, driven per test."""

    data = np.arange(2 * 4 * 5, dtype=np.uint8).reshape(2, 4, 5)
    series_count = 1
    load_error = None
    count_error = None
    metadata = {"axes": "zyx"}

    @staticmethod
    def can_load(filepath):
        return True

    @staticmethod
    def get_series_count(filepath):
        if StubLoader.count_error is not None:
            raise StubLoader.count_error
        return StubLoader.series_count

    @staticmethod
    def load_series(filepath, series_index):
        if StubLoader.load_error is not None:
            raise StubLoader.load_error
        return StubLoader.data

    @staticmethod
    def get_metadata(filepath, series_index):
        return StubLoader.metadata


@pytest.fixture
def reset_stub_loader():
    StubLoader.series_count = 1
    StubLoader.load_error = None
    StubLoader.count_error = None
    StubLoader.metadata = {"axes": "zyx"}
    yield StubLoader
    StubLoader.series_count = 1
    StubLoader.load_error = None
    StubLoader.count_error = None
    StubLoader.metadata = {"axes": "zyx"}


@pytest.fixture
def modal_calls(monkeypatch):
    """Neutralise every modal dialog and record what would have popped."""
    calls = []

    def record(kind):
        def _record(*args, **kwargs):
            calls.append((kind, args[2] if len(args) > 2 else ""))
            return None

        return _record

    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(
            fc.QMessageBox, name, staticmethod(record(name))
        )
    return calls


@pytest.fixture
def widget(qapp, modal_calls):
    return fc.MicroscopyImageConverterWidget(FakeViewer())


@pytest.fixture
def conv_worker(qapp, tmp_path):
    return fc.ConversionWorker(
        files_to_convert=[],
        output_folder=str(tmp_path),
        use_zarr=False,
        file_loader_func=lambda filepath: None,
    )


# --------------------------------------------------------------------- #
# LIF loader
# --------------------------------------------------------------------- #
class TestLIFLoaderDispatch:
    """``LIFLoader`` picks a load strategy from the estimated size."""

    def test_can_load_accepts_a_readable_lif(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fc, "LifFile", lif_factory([FakeLifImage()]))
        assert fc.LIFLoader.can_load(str(tmp_path / "a.lif")) is True

    def test_can_load_reports_reader_failure(self, monkeypatch, tmp_path):
        def boom(filepath):
            raise ValueError("bad header")

        monkeypatch.setattr(fc, "LifFile", boom)
        assert fc.LIFLoader.can_load(str(tmp_path / "a.lif")) is False

    def test_series_count_counts_images(self, monkeypatch, tmp_path):
        images = [FakeLifImage(), FakeLifImage(), FakeLifImage()]
        monkeypatch.setattr(fc, "LifFile", lif_factory(images))
        assert fc.LIFLoader.get_series_count(str(tmp_path / "a.lif")) == 3

    def test_small_series_loads_into_a_tzcyx_numpy_array(
        self, monkeypatch, tmp_path
    ):
        image = FakeLifImage(nt=2, nz=3, channels=2, x_dim=5, y_dim=4)
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))

        result = fc.LIFLoader.load_series(str(tmp_path / "a.lif"), 0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3, 2, 4, 5)
        assert result.dtype == np.uint16
        # value == 100*t + 10*z + c + 1
        assert result[1, 2, 1, 0, 0] == 122

    def test_out_of_range_series_raises_series_index_error(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(fc, "LifFile", lif_factory([FakeLifImage()]))
        with pytest.raises(
            fc.SeriesIndexError, match=r"Series index 7 out of range \(0-0\)"
        ):
            fc.LIFLoader.load_series(str(tmp_path / "a.lif"), 7)

    def test_reader_failure_becomes_file_format_error(
        self, monkeypatch, tmp_path
    ):
        def boom(filepath):
            raise OSError("truncated")

        monkeypatch.setattr(fc, "LifFile", boom)
        with pytest.raises(fc.FileFormatError, match="Failed to load LIF"):
            fc.LIFLoader.load_series(str(tmp_path / "a.lif"), 0)

    def test_medium_series_is_routed_to_chunked_loading(
        self, monkeypatch, tmp_path
    ):
        # 1 * 1 * 3 * 20000 * 20000 * 2 bytes ~= 2.2 GB -> chunked branch.
        image = FakeLifImage(nt=1, nz=1, channels=3, x_dim=20000, y_dim=20000)
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))
        seen = {}

        def fake_chunked(img, timepoints, z_stacks, channels, y_dim, x_dim):
            seen["args"] = (timepoints, z_stacks, channels, y_dim, x_dim)
            return "chunked"

        monkeypatch.setattr(
            fc.LIFLoader, "_load_chunked_numpy", staticmethod(fake_chunked)
        )

        result = fc.LIFLoader.load_series(str(tmp_path / "a.lif"), 0)

        assert result == "chunked"
        assert seen["args"] == (1, 1, 3, 20000, 20000)

    def test_large_series_is_returned_lazily_as_dask(
        self, monkeypatch, tmp_path
    ):
        # 1 * 1 * 6 * 20000 * 20000 * 2 bytes ~= 4.5 GB -> dask branch.
        image = FakeLifImage(nt=1, nz=1, channels=6, x_dim=20000, y_dim=20000)
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))

        result = fc.LIFLoader.load_series(str(tmp_path / "a.lif"), 0)

        assert isinstance(result, da.Array)
        assert result.shape == (1, 1, 6, 20000, 20000)


class TestLIFNumpyLoading:
    """``_load_numpy`` fills a dense array and counts what it could not."""

    def test_dtype_is_taken_from_the_first_frame(self):
        image = FakeLifImage(dtype=np.uint8)
        result = fc.LIFLoader._load_numpy(image, 1, 1, 1, 4, 5)
        assert result.dtype == np.uint8

    def test_missing_and_misshaped_frames_stay_zero(self):
        image = FakeLifImage(
            nt=1,
            nz=1,
            channels=3,
            none_frames=[(0, 0, 1)],
            bad_shape=[(0, 0, 2)],
        )
        result = fc.LIFLoader._load_numpy(image, 1, 1, 3, 4, 5)

        assert result[0, 0, 0, 0, 0] == 1
        assert result[0, 0, 1].max() == 0
        assert result[0, 0, 2].max() == 0

    def test_frame_read_errors_are_tolerated(self):
        image = FakeLifImage(nt=1, nz=1, channels=2, errors=[(0, 0, 1)])
        result = fc.LIFLoader._load_numpy(image, 1, 1, 2, 4, 5)
        assert result[0, 0, 1].max() == 0

    def test_all_frames_missing_is_a_format_error(self):
        image = FakeLifImage(none_frames=[(0, 0, 0)])
        with pytest.raises(fc.FileFormatError, match="No valid frames"):
            fc.LIFLoader._load_numpy(image, 1, 1, 1, 4, 5)

    def test_many_timepoints_take_the_progress_branch(self):
        image = FakeLifImage(nt=11, x_dim=2, y_dim=2)
        result = fc.LIFLoader._load_numpy(image, 11, 1, 1, 2, 2)
        assert result.shape == (11, 1, 1, 2, 2)
        assert result[10, 0, 0, 0, 0] == 1001


class TestLIFChunkedLoading:
    """``_load_chunked_numpy`` walks timepoints in blocks."""

    def test_every_timepoint_is_filled(self):
        image = FakeLifImage(nt=6, nz=2, channels=2, x_dim=3, y_dim=2)
        result = fc.LIFLoader._load_chunked_numpy(image, 6, 2, 2, 2, 3)

        assert result.shape == (6, 2, 2, 2, 3)
        assert result[5, 1, 1, 0, 0] == 512
        assert result.dtype == np.uint16

    def test_unreadable_frames_are_zero_filled(self):
        image = FakeLifImage(
            nt=2,
            nz=1,
            channels=2,
            errors=[(1, 0, 0)],
            none_frames=[(1, 0, 1)],
        )
        result = fc.LIFLoader._load_chunked_numpy(image, 2, 1, 2, 4, 5)
        assert result[1].max() == 0
        assert result[0].max() == 2

    def test_misshaped_frames_are_counted_as_missing(self):
        image = FakeLifImage(nt=1, nz=1, channels=1, bad_shape=[(0, 0, 0)])
        result = fc.LIFLoader._load_chunked_numpy(image, 1, 1, 1, 4, 5)
        assert result.max() == 0


class TestLIFDaskLoading:
    """``_load_as_dask`` stays lazy but computes to the same pixels."""

    def test_single_chunk_is_returned_directly(self):
        image = FakeLifImage(nt=2, nz=1, channels=1, x_dim=3, y_dim=2)
        result = fc.LIFLoader._load_as_dask(image, 2, 1, 1, 2, 3)

        assert isinstance(result, da.Array)
        assert result.shape == (2, 1, 1, 2, 3)
        np.testing.assert_array_equal(
            result[1, 0, 0].compute(), np.full((2, 3), 101, np.uint16)
        )

    def test_multiple_chunks_are_concatenated_along_time(self):
        image = FakeLifImage(nt=6, nz=1, channels=1, x_dim=3, y_dim=2)
        result = fc.LIFLoader._load_as_dask(image, 6, 1, 1, 2, 3)

        assert result.shape == (6, 1, 1, 2, 3)
        computed = result.compute()
        assert computed[0, 0, 0, 0, 0] == 1
        assert computed[5, 0, 0, 0, 0] == 501

    def test_frame_errors_inside_a_chunk_leave_zeros(self):
        image = FakeLifImage(nt=2, nz=1, channels=1, errors=[(0, 0, 0)])
        result = fc.LIFLoader._load_as_dask(image, 2, 1, 1, 4, 5)
        computed = result.compute()
        assert computed[0].max() == 0
        assert computed[1].max() == 101


class TestLIFMetadata:
    """LIF metadata carries axes, resolution and z spacing when present."""

    def test_scale_becomes_resolution_and_spacing(
        self, monkeypatch, tmp_path
    ):
        image = FakeLifImage(
            nt=2, nz=3, channels=4, x_dim=5, y_dim=6, scale=[4.0, 2.0, 0.5]
        )
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))

        meta = fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 0)

        assert meta["axes"] == "TZCYX"
        assert meta["unit"] == "um"
        assert meta["resolution"] == (0.25, 0.5)
        assert meta["spacing"] == 0.5
        assert meta["timepoints"] == 2
        assert meta["z_stacks"] == 3
        assert meta["channels"] == 4
        assert meta["width"] == 5
        assert meta["height"] == 6

    def test_missing_scale_yields_no_resolution(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fc, "LifFile", lif_factory([FakeLifImage()]))
        meta = fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 0)
        # The rest of the metadata must still be there -- an empty dict would
        # satisfy the "no resolution" checks on its own.
        assert meta["axes"] == "TZCYX"
        assert meta["unit"] == "um"
        assert "resolution" not in meta
        assert "spacing" not in meta

    def test_zero_scale_is_ignored(self, monkeypatch, tmp_path):
        image = FakeLifImage(nt=2, scale=[0.0, 0.0, 0.0])
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))
        meta = fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 0)
        assert meta["axes"] == "TZCYX"
        assert meta["timepoints"] == 2
        assert "resolution" not in meta
        assert "spacing" not in meta

    def test_a_two_entry_scale_gives_resolution_but_no_spacing(
        self, monkeypatch, tmp_path
    ):
        image = FakeLifImage(scale=[4.0, 2.0])
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))
        meta = fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 0)
        assert meta["resolution"] == (0.25, 0.5)
        assert "spacing" not in meta

    def test_out_of_range_series_yields_empty_metadata(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(fc, "LifFile", lif_factory([FakeLifImage()]))
        assert fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 4) == {}

    def test_reader_failure_yields_empty_metadata(
        self, monkeypatch, tmp_path
    ):
        def boom(filepath):
            raise OSError("gone")

        monkeypatch.setattr(fc, "LifFile", boom)
        assert fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 0) == {}


# --------------------------------------------------------------------- #
# ND2 loader
# --------------------------------------------------------------------- #
def nd2_module(handle=None, imread=None):
    """A ``nd2`` replacement returning ``handle`` from ``ND2File``."""

    def nd2file(filepath):
        if isinstance(handle, Exception):
            raise handle
        return handle

    def default_imread(filepath, dask=False, xarray=False):
        raise AssertionError("imread should not be called")

    return types.SimpleNamespace(
        ND2File=nd2file, imread=imread or default_imread
    )


class FakeXarray:
    def __init__(self, array):
        self._array = array
        self.selected = None

    def isel(self, **kwargs):
        self.selected = kwargs
        return types.SimpleNamespace(data=self._array[kwargs["P"]])


class TestND2SeriesCount:
    def test_position_axis_is_the_series_count(self, monkeypatch, tmp_path):
        handle = FakeND2File({"P": 4, "Y": 2, "X": 2})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        assert fc.ND2Loader.get_series_count(str(tmp_path / "a.nd2")) == 4

    def test_missing_position_axis_defaults_to_one(
        self, monkeypatch, tmp_path
    ):
        handle = FakeND2File({"Y": 2, "X": 2})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        assert fc.ND2Loader.get_series_count(str(tmp_path / "a.nd2")) == 1


class TestND2LoadSeries:
    """Each ND2 load path -- xarray, dask, plain numpy -- is exercised."""

    def test_single_position_file_loads_via_imread(
        self, monkeypatch, tmp_path
    ):
        data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        seen = {}

        def imread(filepath, dask=False, xarray=False):
            seen["dask"] = dask
            return data

        handle = FakeND2File({"T": 2, "Y": 3, "X": 4})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 0)

        np.testing.assert_array_equal(result, data)
        assert seen["dask"] is False

    def test_multi_position_small_file_uses_direct_indexing(
        self, monkeypatch, tmp_path
    ):
        data = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
        handle = FakeND2File({"P": 3, "Y": 2, "X": 4}, data=data)
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 2)

        np.testing.assert_array_equal(result, data[2])

    def test_multi_position_without_getitem_falls_back_to_take(
        self, monkeypatch, tmp_path
    ):
        data = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)

        def imread(filepath, dask=False, xarray=False):
            return data

        handle = FakeND2File(
            {"P": 3, "Y": 2, "X": 4}, data=data, support_getitem=False
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 1)

        np.testing.assert_array_equal(result, data[1])

    def test_large_multi_position_file_prefers_the_xarray_path(
        self, monkeypatch, tmp_path
    ):
        # 1 * 2 * 30000 * 30000 * 2 bytes ~= 3.4 GB -> use_dask.
        data = np.arange(2 * 2, dtype=np.uint16).reshape(2, 2)
        holder = {}

        def imread(filepath, dask=False, xarray=False):
            holder["kwargs"] = (dask, xarray)
            return FakeXarray(data)

        handle = FakeND2File(
            {"P": 2, "T": 1, "C": 2, "Y": 30000, "X": 30000}
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 1)

        assert holder["kwargs"] == (True, True)
        np.testing.assert_array_equal(result, data[1])

    def test_xarray_failure_falls_back_to_dask_indexing(
        self, monkeypatch, tmp_path
    ):
        source = np.arange(2 * 2 * 3, dtype=np.uint16).reshape(2, 2, 3)

        def imread(filepath, dask=False, xarray=False):
            raise ValueError("no xarray support")

        handle = FakeND2File(
            {"P": 2, "C": 2, "Y": 30000, "X": 30000},
            dask_array=da.from_array(source, chunks=1),
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 1)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, source[1])

    def test_both_lazy_paths_failing_falls_back_to_numpy(
        self, monkeypatch, tmp_path
    ):
        data = np.arange(2 * 2 * 2 * 3, dtype=np.uint16).reshape(2, 2, 2, 3)

        def imread(filepath, dask=False, xarray=False):
            raise ValueError("no xarray support")

        handle = FakeND2File(
            {"P": 2, "C": 2, "Y": 30000, "X": 30000},
            data=data,
            to_dask_error=AttributeError("no to_dask"),
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 0)

        np.testing.assert_array_equal(result, data[0])

    def test_out_of_range_series_raises_series_index_error(
        self, monkeypatch, tmp_path
    ):
        handle = FakeND2File({"P": 2, "Y": 2, "X": 2})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        with pytest.raises(
            fc.SeriesIndexError, match=r"Series index 5 out of range \(0-1\)"
        ):
            fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 5)

    def test_missing_file_becomes_a_format_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            fc, "nd2", nd2_module(FileNotFoundError("no such file"))
        )
        with pytest.raises(fc.FileFormatError, match="Cannot access"):
            fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 0)

    def test_corrupt_file_becomes_a_format_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fc, "nd2", nd2_module(ValueError("bad chunk")))
        with pytest.raises(fc.FileFormatError, match="Failed to load ND2"):
            fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 0)


class TestND2Metadata:
    def test_axes_and_voxel_size_are_translated(self, monkeypatch, tmp_path):
        voxel = types.SimpleNamespace(x=0.5, y=0.25, z=2.0)
        handle = FakeND2File(
            {"P": 2, "T": 3, "Z": 4, "C": 2, "Y": 6, "X": 5},
            voxel=voxel,
            is_rgb=True,
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))

        meta = fc.ND2Loader.get_metadata(str(tmp_path / "a.nd2"), 1)

        assert meta["axes"] == "TZCYX"
        assert meta["resolution"] == (2.0, 4.0)
        assert meta["spacing"] == 2.0
        assert meta["unit"] == "um"
        assert meta["shape"] == (3, 4, 2, 6, 5)
        assert meta["is_rgb"] is True
        assert meta["dtype"] == "uint16"

    def test_absent_voxel_size_defaults_to_unit_scale(
        self, monkeypatch, tmp_path
    ):
        handle = FakeND2File({"Z": 2, "Y": 3, "X": 4}, voxel=None)
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))

        meta = fc.ND2Loader.get_metadata(str(tmp_path / "a.nd2"), 0)

        assert meta["resolution"] == (1.0, 1.0)
        assert "spacing" not in meta
        assert meta["axes"] == "ZYX"

    def test_broken_voxel_size_falls_back(self, monkeypatch, tmp_path):
        handle = FakeND2File(
            {"Y": 3, "X": 4}, voxel=AttributeError("no metadata")
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        meta = fc.ND2Loader.get_metadata(str(tmp_path / "a.nd2"), 0)
        assert meta["resolution"] == (1.0, 1.0)

    def test_out_of_range_position_yields_empty_metadata(
        self, monkeypatch, tmp_path
    ):
        handle = FakeND2File({"P": 2, "Y": 3, "X": 4})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        assert fc.ND2Loader.get_metadata(str(tmp_path / "a.nd2"), 9) == {}

    def test_nonzero_series_on_single_position_is_empty(
        self, monkeypatch, tmp_path
    ):
        handle = FakeND2File({"Y": 3, "X": 4})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        assert fc.ND2Loader.get_metadata(str(tmp_path / "a.nd2"), 1) == {}

    def test_unreadable_file_yields_empty_metadata(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(fc, "nd2", nd2_module(OSError("io error")))
        assert fc.ND2Loader.get_metadata(str(tmp_path / "a.nd2"), 0) == {}


# --------------------------------------------------------------------- #
# TIFF slide loader with tiffslide present
# --------------------------------------------------------------------- #
class TestTIFFSlideWithReader:
    """The ``tiffslide`` branch (absent in this env) driven by a fake."""

    @pytest.fixture(autouse=True)
    def _with_tiffslide(self, monkeypatch):
        FakeSlide.open_error = None
        monkeypatch.setattr(fc, "TiffSlide", FakeSlide)
        yield
        FakeSlide.open_error = None

    def test_series_count_is_the_number_of_levels(self, tmp_path):
        assert fc.TIFFSlideLoader.get_series_count(str(tmp_path / "s.ndpi")) == 2

    def test_load_series_reads_the_whole_level(self, tmp_path):
        result = fc.TIFFSlideLoader.load_series(str(tmp_path / "s.ndpi"), 1)
        assert result.shape == (2, 3)
        assert result.dtype == np.uint8
        assert (result == 2).all()

    def test_out_of_range_level_raises_series_index_error(self, tmp_path):
        with pytest.raises(
            fc.SeriesIndexError, match="Series index 9 out of range"
        ):
            fc.TIFFSlideLoader.load_series(str(tmp_path / "s.ndpi"), 9)

    def test_metadata_reads_axes_and_mpp(self, tmp_path):
        meta = fc.TIFFSlideLoader.get_metadata(str(tmp_path / "s.ndpi"), 0)
        # tiffslide reports mpp-x/mpp-y as micrometres per pixel; every
        # loader's "resolution" is pixels per micrometre (the convention
        # ConversionWorker._build_scale_transform inverts with
        # 1/resolution), so get_metadata must store the reciprocal:
        # mpp-x=0.25 -> 4.0, mpp-y=0.5 -> 2.0.
        assert meta == {
            "axes": "YXS",
            "resolution": (4.0, 2.0),
            "unit": "um",
        }

    def test_metadata_of_out_of_range_level_is_empty(self, tmp_path):
        assert fc.TIFFSlideLoader.get_metadata(str(tmp_path / "s.ndpi"), 9) == {}

    def test_unreadable_slide_metadata_is_empty(self, tmp_path):
        FakeSlide.open_error = OSError("cannot open")
        assert fc.TIFFSlideLoader.get_metadata(str(tmp_path / "s.ndpi")
                                               , 0) == {}

    def test_unreadable_slide_falls_back_to_tifffile_for_the_count(
        self, tmp_path
    ):
        FakeSlide.open_error = OSError("cannot open")
        path = tmp_path / "s.svs"
        tifffile.imwrite(path, np.zeros((4, 4), np.uint8))
        assert fc.TIFFSlideLoader.get_series_count(str(path)) == 1


# --------------------------------------------------------------------- #
# CZI loader
# --------------------------------------------------------------------- #
def plane_value(plane, scene=None, y=4, x=5, dtype=np.uint16):
    """A CZI plane whose value encodes its (T, C, Z) coordinate."""
    value = 100 * plane["T"] + 10 * plane["C"] + plane["Z"] + 1
    return np.full((y, x, 1), value, dtype=dtype)


class TestCZICanLoad:
    def test_readable_file_is_accepted(self, monkeypatch, tmp_path):
        doc = FakeCziDoc({"Y": (0, 4), "X": (0, 5)})
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        assert fc.CZILoader.can_load(str(tmp_path / "a.czi")) is True

    def test_wrong_extension_is_rejected_without_opening(self, monkeypatch):
        monkeypatch.setattr(
            fc, "pyczi", czi_module(error=AssertionError("must not open"))
        )
        assert fc.CZILoader.can_load("/nowhere/a.tif") is False

    def test_unreadable_file_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            fc, "pyczi", czi_module(error=RuntimeError("bad magic"))
        )
        assert fc.CZILoader.can_load(str(tmp_path / "a.czi")) is False


class TestCZISeriesCount:
    def test_scenes_are_counted_as_series(self, monkeypatch, tmp_path):
        doc = FakeCziDoc({"Y": (0, 4)}, scenes={0: None, 1: None, 2: None})
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        assert fc.CZILoader.get_series_count(str(tmp_path / "a.czi")) == 3

    def test_sceneless_file_has_one_series(self, monkeypatch, tmp_path):
        doc = FakeCziDoc({"Y": (0, 4)}, scenes={})
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        assert fc.CZILoader.get_series_count(str(tmp_path / "a.czi")) == 1

    def test_unreadable_file_counts_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            fc, "pyczi", czi_module(error=OSError("cannot open"))
        )
        assert fc.CZILoader.get_series_count(str(tmp_path / "a.czi")) == 0


class TestCZILoadSeries:
    """Plane-by-plane assembly into a TCZYX array."""

    def test_single_scene_assembles_tczyx(self, monkeypatch, tmp_path):
        doc = FakeCziDoc(
            {
                "T": (0, 2),
                "Z": (0, 3),
                "C": (0, 2),
                "Y": (0, 4),
                "X": (0, 5),
            },
            plane=plane_value,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        result = fc.CZILoader.load_series(str(tmp_path / "a.czi"), 0)

        assert result.shape == (2, 2, 3, 4, 5)
        assert result.dtype == np.uint16
        assert result[1, 1, 2, 0, 0] == 100 + 10 + 2 + 1
        assert len(doc.reads) == 1 + 2 * 2 * 3

    def test_scene_id_is_passed_through_to_the_reader(
        self, monkeypatch, tmp_path
    ):
        doc = FakeCziDoc(
            {"Y": (0, 4), "X": (0, 5)},
            scenes={7: (0, 0, 4, 5), 9: (0, 0, 4, 5)},
            plane=plane_value,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        result = fc.CZILoader.load_series(str(tmp_path / "a.czi"), 1)

        assert result.shape == (1, 1, 1, 4, 5)
        assert {scene for _plane, scene in doc.reads} == {9}

    def test_dimension_offsets_are_honoured(self, monkeypatch, tmp_path):
        doc = FakeCziDoc(
            {"T": (5, 2), "Z": (0, 1), "C": (0, 1), "Y": (0, 4), "X": (0, 5)},
            plane=plane_value,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        result = fc.CZILoader.load_series(str(tmp_path / "a.czi"), 0)

        assert result[0, 0, 0, 0, 0] == 501
        assert result[1, 0, 0, 0, 0] == 601

    def test_out_of_range_scene_raises_series_index_error(
        self, monkeypatch, tmp_path
    ):
        doc = FakeCziDoc(
            {"Y": (0, 4)}, scenes={0: None}, plane=plane_value
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        with pytest.raises(
            fc.SeriesIndexError, match=r"Scene index 3 out of range \(0-0\)"
        ):
            fc.CZILoader.load_series(str(tmp_path / "a.czi"), 3)

    def test_nonzero_series_on_sceneless_file_raises(
        self, monkeypatch, tmp_path
    ):
        doc = FakeCziDoc({"Y": (0, 4)}, plane=plane_value)
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        with pytest.raises(
            fc.SeriesIndexError,
            match="Single scene file only supports series index 0",
        ):
            fc.CZILoader.load_series(str(tmp_path / "a.czi"), 1)

    def test_read_failure_becomes_a_format_error(self, monkeypatch, tmp_path):
        doc = FakeCziDoc(
            {"Y": (0, 4)}, read_error=RuntimeError("Illegal data detected")
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        with pytest.raises(fc.FileFormatError, match="Failed to load CZI"):
            fc.CZILoader.load_series(str(tmp_path / "a.czi"), 0)

    def test_large_stack_is_returned_lazily(self, monkeypatch, tmp_path):
        # 40 * 2 * 4 * 2048 * 2048 * 2 bytes ~= 2.5 GB -> dask branch.
        def big_plane(plane, scene=None):
            return np.zeros((2048, 2048, 1), dtype=np.uint16)

        doc = FakeCziDoc(
            {
                "T": (0, 40),
                "Z": (0, 4),
                "C": (0, 2),
                "Y": (0, 2048),
                "X": (0, 2048),
            },
            plane=big_plane,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        result = fc.CZILoader.load_series(str(tmp_path / "a.czi"), 0)

        assert isinstance(result, da.Array)
        assert result.shape == (40, 2, 4, 2048, 2048)
        assert result.dtype == np.uint16
        # Only the touched plane is materialised.
        assert result[0, 0, 0, 0, 0].compute() == 0


class _BrokenBoundingBox(dict):
    """A bounding box whose ``get`` fails, to force the metadata fallback."""

    def get(self, *args, **kwargs):
        raise AttributeError("no get on this bounding box")


class TestCZIMetadata:
    def test_scale_xml_becomes_resolution_and_spacing(
        self, monkeypatch, tmp_path
    ):
        xml = (
            "<Scaling><Items>"
            '<Distance Id="X"><Value>2.5E-07</Value></Distance>'
            '<Distance Id="Y"><Value>2.5E-07</Value></Distance>'
            '<Distance Id="Z"><Value>1.0E-06</Value></Distance>'
            "</Items></Scaling>"
        )
        doc = FakeCziDoc(
            {"T": (0, 2), "Z": (0, 3), "C": (0, 1), "Y": (0, 4), "X": (0, 5)},
            metadata=xml,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        meta = fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 0)

        assert meta["axes"] == "TCZYX"
        assert meta["unit"] == "um"
        assert meta["resolution"] == (4.0, 4.0)
        assert meta["spacing"] == 1.0
        assert meta["has_scenes"] is False
        assert meta["scene_count"] == 1
        assert "scene_id" not in meta

    def test_scene_metadata_records_the_scene_id(self, monkeypatch, tmp_path):
        doc = FakeCziDoc(
            {"Y": (0, 4), "X": (0, 5)},
            scenes={3: None, 4: None},
            metadata=None,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        meta = fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 1)

        assert meta["scene_id"] == 4
        assert meta["has_scenes"] is True
        assert meta["scene_count"] == 2
        assert "resolution" not in meta

    def test_metadata_read_failure_still_yields_axes(
        self, monkeypatch, tmp_path
    ):
        doc = FakeCziDoc(
            {"Y": (0, 4), "X": (0, 5)},
            metadata=AttributeError("no metadata block"),
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        meta = fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 0)

        assert meta["axes"] == "TCZYX"
        assert "resolution" not in meta

    def test_out_of_range_scene_yields_empty_metadata(
        self, monkeypatch, tmp_path
    ):
        doc = FakeCziDoc({"Y": (0, 4)}, scenes={0: None})
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        assert fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 5) == {}

    def test_nonzero_series_on_sceneless_file_yields_empty(
        self, monkeypatch, tmp_path
    ):
        doc = FakeCziDoc({"Y": (0, 4)})
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))
        assert fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 2) == {}

    def test_unusable_bounding_box_falls_back_to_significant_axes(
        self, monkeypatch, tmp_path
    ):
        bbox = _BrokenBoundingBox(
            {"T": (0, 3), "Z": (0, 1), "C": (0, 2), "Y": (0, 4), "X": (0, 5)}
        )
        doc = FakeCziDoc(bbox, metadata=None)
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        meta = fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 0)

        # Z has size 1 so it is dropped by the fallback.
        assert meta["axes"] == "TCYX"

    def test_fallback_without_significant_axes_defaults_to_yx(
        self, monkeypatch, tmp_path
    ):
        bbox = _BrokenBoundingBox({"T": (0, 1), "Y": (0, 1)})
        doc = FakeCziDoc(bbox, metadata=None)
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        meta = fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 0)

        assert meta["axes"] == "YX"

    def test_unreadable_file_yields_empty_metadata(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            fc, "pyczi", czi_module(error=OSError("cannot open"))
        )
        assert fc.CZILoader.get_metadata(str(tmp_path / "a.czi"), 0) == {}


# --------------------------------------------------------------------- #
# Acquifer loader
# --------------------------------------------------------------------- #
# ``xarray`` and the Acquifer plugin are resolved lazily on purpose. A
# module-level ``importorskip`` here would silently skip *every* test in this
# file -- LIF, ND2, CZI, Zarr and the whole widget -- on any machine without
# the optional Acquifer reader, turning a missing plugin into 180+ invisible
# holes. Resolving them inside the Acquifer fixtures skips only those tests.
def _require_xarray():
    return pytest.importorskip(
        "xarray", reason="xarray backs the Acquifer dataset objects"
    )


def _require_acquifer_utils():
    return pytest.importorskip(
        "acquifer_napari_plugin.utils",
        reason="the Acquifer reader is an optional plugin",
    )


def make_well_dataset():
    xr = _require_xarray()
    values = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(2, 2, 3, 4)
    return xr.DataArray(
        values,
        dims=("Well", "Channel", "Y", "X"),
        coords={"Well": [11, 22]},
    )


def make_flat_dataset():
    xr = _require_xarray()
    values = np.arange(3 * 4, dtype=np.uint16).reshape(3, 4)
    return xr.DataArray(values, dims=("Y", "X"))


@pytest.fixture
def acquifer_dir(tmp_path):
    directory = tmp_path / "plate"
    directory.mkdir()
    (directory / "PlateLayout").write_bytes(b"")
    tifffile.imwrite(
        directory / "A01--PX0250.tif", np.zeros((4, 4), np.uint8)
    )
    fc.AcquiferLoader._dataset_cache.clear()
    yield directory
    fc.AcquiferLoader._dataset_cache.clear()


@pytest.fixture
def acquifer_reader(monkeypatch):
    """Replace the plugin's reader with a controllable stub."""
    acquifer_utils = _require_acquifer_utils()
    state = {"dataset": make_well_dataset(), "calls": 0, "error": None}

    def array_from_directory(directory):
        state["calls"] += 1
        if state["error"] is not None:
            raise state["error"]
        return state["dataset"]

    monkeypatch.setattr(
        acquifer_utils, "array_from_directory", array_from_directory
    )
    return state


class TestAcquiferDatasetLoading:
    """``_load_dataset`` caches, validates and wraps reader failures."""

    def test_dataset_is_cached_after_the_first_read(
        self, acquifer_dir, acquifer_reader
    ):
        first = fc.AcquiferLoader._load_dataset(str(acquifer_dir))
        second = fc.AcquiferLoader._load_dataset(str(acquifer_dir))

        assert first is second
        assert acquifer_reader["calls"] == 1

    def test_directory_without_images_is_rejected(
        self, tmp_path, acquifer_reader
    ):
        empty = tmp_path / "empty"
        empty.mkdir()
        fc.AcquiferLoader._dataset_cache.clear()
        with pytest.raises(fc.FileFormatError, match="No image files"):
            fc.AcquiferLoader._load_dataset(str(empty))
        assert acquifer_reader["calls"] == 0

    def test_reader_failure_becomes_a_format_error(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["error"] = ValueError("bad plate layout")
        with pytest.raises(
            fc.FileFormatError, match="Failed to load Acquifer dataset"
        ):
            fc.AcquiferLoader._load_dataset(str(acquifer_dir))

    def test_missing_plugin_becomes_a_format_error(
        self, acquifer_dir, monkeypatch
    ):
        monkeypatch.setitem(
            sys.modules, "acquifer_napari_plugin.utils", None
        )
        with pytest.raises(
            fc.FileFormatError, match="Acquifer plugin not available"
        ):
            fc.AcquiferLoader._load_dataset(str(acquifer_dir))


class TestAcquiferSeries:
    def test_series_count_is_the_number_of_wells(
        self, acquifer_dir, acquifer_reader
    ):
        assert fc.AcquiferLoader.get_series_count(str(acquifer_dir)) == 2

    def test_series_count_is_zero_when_loading_fails(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["error"] = ValueError("boom")
        assert fc.AcquiferLoader.get_series_count(str(acquifer_dir)) == 0

    def test_load_series_selects_one_well(
        self, acquifer_dir, acquifer_reader
    ):
        result = fc.AcquiferLoader.load_series(str(acquifer_dir), 1)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_equal(
            result, acquifer_reader["dataset"].values[1]
        )

    def test_out_of_range_well_raises_series_index_error(
        self, acquifer_dir, acquifer_reader
    ):
        with pytest.raises(
            fc.SeriesIndexError, match="Series index 5 out of range"
        ):
            fc.AcquiferLoader.load_series(str(acquifer_dir), 5)

    def test_single_well_dataset_returns_the_whole_array(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["dataset"] = make_flat_dataset()
        result = fc.AcquiferLoader.load_series(str(acquifer_dir), 0)
        assert result.shape == (3, 4)

    def test_single_well_dataset_rejects_other_indices(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["dataset"] = make_flat_dataset()
        with pytest.raises(
            fc.SeriesIndexError,
            match="Single well dataset only supports series index 0",
        ):
            fc.AcquiferLoader.load_series(str(acquifer_dir), 1)


class TestAcquiferMetadata:
    def test_axes_are_normalised_and_pixel_size_is_parsed(
        self, acquifer_dir, acquifer_reader
    ):
        meta = fc.AcquiferLoader.get_metadata(str(acquifer_dir), 0)

        assert meta["axes"] == "CYX"
        # --PX0250 -> 250 * 1e-4 = 0.025 micrometres per pixel; stored as
        # its reciprocal (40.0 pixels per micrometre) to match every other
        # loader's convention, which _build_scale_transform inverts back
        # with 1/resolution.
        assert meta["resolution"] == (40.0, 40.0)
        assert meta["unit"] == "um"
        assert meta["filepath"] == str(acquifer_dir)

    def test_dataset_without_wells_uses_its_own_dims(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["dataset"] = make_flat_dataset()
        meta = fc.AcquiferLoader.get_metadata(str(acquifer_dir), 0)
        assert meta["axes"] == "YX"

    def test_missing_pixel_size_defaults_to_unit_resolution(
        self, tmp_path, acquifer_reader
    ):
        directory = tmp_path / "plate2"
        directory.mkdir()
        (directory / "PlateLayout").write_bytes(b"")
        tifffile.imwrite(directory / "Image-A01.tif", np.zeros((4, 4), np.uint8))
        fc.AcquiferLoader._dataset_cache.clear()

        meta = fc.AcquiferLoader.get_metadata(str(directory), 0)

        assert meta["resolution"] == (1.0, 1.0)
        fc.AcquiferLoader._dataset_cache.clear()

    def test_unloadable_dataset_yields_empty_metadata(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["error"] = ValueError("boom")
        assert fc.AcquiferLoader.get_metadata(str(acquifer_dir), 0) == {}


# --------------------------------------------------------------------- #
# Scan worker error path
# --------------------------------------------------------------------- #
class TestScanFolderWorkerRun:
    """The scan itself -- filter matching, recursion and the progress ticks."""

    def _run(self, folder, filters):
        worker = fc.ScanFolderWorker(str(folder), filters)
        found, progress, errors = [], [], []
        worker.finished.connect(found.append)
        worker.progress.connect(lambda cur, tot: progress.append((cur, tot)))
        worker.error.connect(errors.append)
        worker.run()
        assert errors == []
        return found, progress

    def test_only_matching_extensions_are_returned(self, qapp, tmp_path):
        (tmp_path / "keep.lif").write_bytes(b"")
        (tmp_path / "KEEP.CZI").write_bytes(b"")
        (tmp_path / "drop.txt").write_bytes(b"")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "deep.lif").write_bytes(b"")

        found, progress = self._run(tmp_path, [".lif", ".czi"])

        assert len(found) == 1
        assert sorted(found[0]) == sorted(
            [
                str(tmp_path / "KEEP.CZI"),
                str(tmp_path / "keep.lif"),
                str(nested / "deep.lif"),
            ]
        )
        assert progress == [(0, 3)]

    def test_an_empty_folder_finishes_with_an_empty_list(
        self, qapp, tmp_path
    ):
        found, progress = self._run(tmp_path, [".lif"])
        assert found == [[]]
        assert progress == []

    def test_the_acquifer_filter_collects_directories(self, qapp, tmp_path):
        plate = tmp_path / "plate"
        plate.mkdir()
        (plate / "PlateLayout").write_bytes(b"")
        tifffile.imwrite(
            plate / "A01--PX0250.tif", np.zeros((2, 2), np.uint8)
        )
        (tmp_path / "ignored").mkdir()

        found, _progress = self._run(tmp_path, [".lif", "acquifer"])

        assert found[0] == [str(plate)]


class TestScanFolderWorkerErrors:
    def test_walk_failure_is_reported_on_the_error_signal(
        self, qapp, tmp_path, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise PermissionError("no access")

        monkeypatch.setattr(fc.os, "walk", boom)
        worker = fc.ScanFolderWorker(str(tmp_path), [".lif"])
        errors = []
        worker.error.connect(errors.append)

        worker.run()

        assert len(errors) == 1
        assert errors[0].startswith("Scan failed:")
        assert "no access" in errors[0]


# --------------------------------------------------------------------- #
# ConversionWorker
# --------------------------------------------------------------------- #
class NoNbytesArray:
    """An array-like without ``nbytes``, to force the size fallback."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.itemsize = array.dtype.itemsize
        self.ndim = array.ndim

    def __array__(self, dtype=None, copy=None):
        return self._array


class TestConvertSingleFileOutputPaths:
    """The output name encodes the stem and the series index."""

    def _worker(self, tmp_path, use_zarr, loader):
        return fc.ConversionWorker(
            files_to_convert=[],
            output_folder=str(tmp_path),
            use_zarr=use_zarr,
            file_loader_func=lambda filepath: loader,
        )

    def test_series_index_is_part_of_the_tif_name(
        self, qapp, tmp_path, reset_stub_loader
    ):
        worker = self._worker(tmp_path, False, reset_stub_loader)
        assert worker._convert_single_file("/data/plate.lif", 3) is True

        written = tmp_path / "plate_series3.tif"
        assert written.exists()
        # The name is only half of it -- the loaded series has to land inside.
        np.testing.assert_array_equal(
            tifffile.imread(written), reset_stub_loader.data
        )

    def test_series_index_is_part_of_the_zarr_name(
        self, qapp, tmp_path, reset_stub_loader
    ):
        worker = self._worker(tmp_path, True, reset_stub_loader)
        assert worker._convert_single_file("/data/plate.lif", 2) is True

        written = tmp_path / "plate_series2.zarr"
        assert written.is_dir()
        np.testing.assert_array_equal(
            zarr.open_group(str(written), mode="r")["0"][:],
            reset_stub_loader.data,
        )

    def test_a_file_with_no_loader_is_a_conversion_error(
        self, qapp, tmp_path
    ):
        worker = self._worker(tmp_path, False, None)
        with pytest.raises(
            fc.ConversionError, match="Conversion failed: Unsupported file format"
        ):
            worker._convert_single_file("/data/plate.xyz", 0)
        assert list(tmp_path.iterdir()) == []

    def test_loader_failure_is_wrapped_in_conversion_error(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = fc.FileFormatError("truncated file")
        worker = self._worker(tmp_path, False, reset_stub_loader)
        with pytest.raises(fc.ConversionError, match="truncated file"):
            worker._convert_single_file("/data/plate.lif", 0)

    def test_memory_error_is_wrapped_in_conversion_error(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = MemoryError("too big")
        worker = self._worker(tmp_path, False, reset_stub_loader)
        with pytest.raises(
            fc.ConversionError, match="Conversion failed: too big"
        ):
            worker._convert_single_file("/data/plate.lif", 0)

    def test_none_metadata_is_tolerated(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.metadata = None
        worker = self._worker(tmp_path, False, reset_stub_loader)
        assert worker._convert_single_file("/data/plate.lif", 0) is True
        np.testing.assert_array_equal(
            tifffile.imread(tmp_path / "plate_series0.tif"),
            reset_stub_loader.data,
        )


class TestSpatialMetadataSummary:
    def test_summary_is_skipped_without_spatial_metadata(
        self, conv_worker, capsys
    ):
        conv_worker._log_spatial_metadata_summary("/a/b.lif", {"axes": "zyx"})
        assert capsys.readouterr().out == ""

    def test_summary_names_the_file_and_values(self, conv_worker, capsys):
        conv_worker._log_spatial_metadata_summary(
            "/a/b.lif",
            {"resolution": (2.0, 2.0), "spacing": 5.0, "unit": "um"},
        )
        assert capsys.readouterr().out == (
            "Spatial metadata detected for b.lif: "
            "resolution=(2.0, 2.0), spacing=5.0, unit=um\n"
        )

    def test_a_missing_unit_is_reported_as_unknown(self, conv_worker, capsys):
        conv_worker._log_spatial_metadata_summary(
            "/a/b.lif", {"spacing": 5.0}
        )
        assert capsys.readouterr().out == (
            "Spatial metadata detected for b.lif: "
            "resolution=None, spacing=5.0, unit=unknown\n"
        )


class TestConversionWorkerRun:
    """``run`` classifies each outcome and reports it once."""

    def _worker(self, tmp_path, files, loader, use_zarr=False):
        return fc.ConversionWorker(
            files_to_convert=files,
            output_folder=str(tmp_path),
            use_zarr=use_zarr,
            file_loader_func=lambda filepath: loader,
        )

    def _collect(self, worker):
        done = []
        worker.file_done.connect(
            lambda path, ok, msg: done.append((path, ok, msg))
        )
        counts = []
        worker.finished.connect(counts.append)
        return done, counts

    def test_conversion_error_is_reported_as_a_failure(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = fc.FileFormatError("bad header")
        worker = self._worker(
            tmp_path, [("/data/a.lif", 0)], reset_stub_loader
        )
        done, counts = self._collect(worker)

        worker.run()

        assert counts == [0]
        assert done[0][1] is False
        assert "bad header" in done[0][2]

    def test_known_czi_read_errors_are_marked_skipped(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = fc.FileFormatError(
            "Invalid SubBlkDirectory-magic at offset 42"
        )
        worker = self._worker(
            tmp_path, [("/data/a.czi", 0)], reset_stub_loader
        )
        done, counts = self._collect(worker)

        worker.run()

        assert counts == [0]
        assert done[0][2] == (
            "SKIPPED: Conversion failed: "
            "Invalid SubBlkDirectory-magic at offset 42"
        )
        assert done[0][0] == "/data/a.czi"

    def test_the_second_known_czi_token_is_also_skipped(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = fc.FileFormatError(
            "Illegal data detected at offset 7"
        )
        worker = self._worker(
            tmp_path, [("/data/a.czi", 0)], reset_stub_loader
        )
        done, _counts = self._collect(worker)

        worker.run()

        assert done[0][2].startswith("SKIPPED:")

    def test_an_unrecognised_czi_error_is_a_failure_not_a_skip(
        self, qapp, tmp_path, reset_stub_loader
    ):
        """The skip list must stay a list, not swallow every CZI error."""
        reset_stub_loader.load_error = fc.FileFormatError("disk corrupted")
        worker = self._worker(
            tmp_path, [("/data/a.czi", 0)], reset_stub_loader
        )
        done, _counts = self._collect(worker)

        worker.run()

        assert done[0][1] is False
        assert not done[0][2].startswith("SKIPPED:")
        assert "disk corrupted" in done[0][2]

    def test_a_lif_never_takes_the_czi_skip_path(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = fc.FileFormatError(
            "Invalid SubBlkDirectory-magic at offset 42"
        )
        worker = self._worker(
            tmp_path, [("/data/a.lif", 0)], reset_stub_loader
        )
        done, _counts = self._collect(worker)

        worker.run()

        assert not done[0][2].startswith("SKIPPED:")

    def test_os_errors_are_reported_as_file_access_errors(
        self, qapp, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.load_error = OSError("disk went away")
        worker = self._worker(
            tmp_path, [("/data/a.lif", 0)], reset_stub_loader
        )
        done, counts = self._collect(worker)

        worker.run()

        assert counts == [0]
        assert done[0][2].startswith("File access error:")

    def test_a_falsy_conversion_result_is_reported_as_failed(
        self, qapp, tmp_path, reset_stub_loader
    ):
        class SilentlyFailingWorker(fc.ConversionWorker):
            def _convert_single_file(self, filepath, series_index):
                return False

        worker = SilentlyFailingWorker(
            files_to_convert=[("/data/a.lif", 0)],
            output_folder=str(tmp_path),
            use_zarr=False,
            file_loader_func=lambda filepath: reset_stub_loader,
        )
        done, counts = self._collect(worker)

        worker.run()

        assert counts == [0]
        assert done == [("/data/a.lif", False, "Conversion failed")]

    def test_stopping_the_worker_halts_before_the_next_file(
        self, qapp, tmp_path, reset_stub_loader
    ):
        worker = self._worker(
            tmp_path,
            [("/data/a.lif", 0), ("/data/b.lif", 0)],
            reset_stub_loader,
        )
        done, counts = self._collect(worker)
        worker.stop()

        worker.run()

        assert worker.running is False
        assert done == []
        assert counts == [0]
        # Nothing ran, so there is nothing to report on either.
        assert not (tmp_path / "conversion_report.csv").exists()

    def test_progress_is_emitted_per_file_and_report_lists_all(
        self, qapp, tmp_path, reset_stub_loader
    ):
        worker = self._worker(
            tmp_path,
            [("/data/a.lif", 0), ("/data/b.lif", 1)],
            reset_stub_loader,
        )
        progress = []
        worker.progress.connect(
            lambda cur, total, name: progress.append((cur, total, name))
        )
        _done, counts = self._collect(worker)

        worker.run()

        assert counts == [2]
        assert progress == [(1, 2, "a.lif"), (2, 2, "b.lif")]

        with open(
            tmp_path / "conversion_report.csv", newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))

        assert [
            (row["filepath"], row["series_index"], row["status"], row["message"])
            for row in rows
        ] == [
            ("/data/a.lif", "0", "success", "Conversion successful"),
            ("/data/b.lif", "1", "success", "Conversion successful"),
        ]
        assert all(row["timestamp"] for row in rows)

    def test_the_report_records_failures_alongside_successes(
        self, qapp, tmp_path, reset_stub_loader
    ):
        class HalfFailingWorker(fc.ConversionWorker):
            def _convert_single_file(self, filepath, series_index):
                if filepath.endswith("bad.lif"):
                    raise fc.ConversionError("bad header")
                return True

        worker = HalfFailingWorker(
            files_to_convert=[("/data/good.lif", 0), ("/data/bad.lif", 0)],
            output_folder=str(tmp_path),
            use_zarr=False,
            file_loader_func=lambda filepath: reset_stub_loader,
        )
        _done, counts = self._collect(worker)

        worker.run()

        assert counts == [1]
        with open(
            tmp_path / "conversion_report.csv", newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert [(row["filepath"], row["status"]) for row in rows] == [
            ("/data/good.lif", "success"),
            ("/data/bad.lif", "failed"),
        ]
        assert rows[1]["message"] == "bad header"


class NoNbytesShape:
    """Shape/itemsize only -- enough for the size estimate, not for writing."""

    def __init__(self, shape, itemsize):
        self.shape = shape
        self.itemsize = itemsize
        self.ndim = len(shape)


class TestSaveTifBranches:
    def test_size_is_estimated_without_nbytes(
        self, conv_worker, tmp_path, capsys
    ):
        data = np.arange(20, dtype=np.uint8).reshape(4, 5)
        out = tmp_path / "nonbytes.tif"

        assert conv_worker._save_tif(NoNbytesArray(data), str(out), {}) is True
        np.testing.assert_array_equal(tifffile.imread(out), data)
        assert "estimated size: 0.00GB" in capsys.readouterr().out

    def test_the_nbytes_free_estimate_is_a_real_number(
        self, conv_worker, tmp_path, capsys
    ):
        """shape * itemsize must drive the guard, not a placeholder."""
        out = tmp_path / "toobig.tif"

        with pytest.raises(MemoryError, match="File too large for TIF"):
            conv_worker._save_tif(
                NoNbytesShape((2000, 2000, 2000), 8), str(out), {}
            )

        assert "estimated size: 59.60GB" in capsys.readouterr().out
        assert not out.exists()

    def test_pixel_size_metadata_is_written_into_the_tif_tags(
        self, conv_worker, tmp_path
    ):
        out = tmp_path / "resolution.tif"

        assert (
            conv_worker._save_tif(
                np.zeros((4, 5), np.uint8),
                str(out),
                {"resolution": (4.0, 2.0)},
            )
            is True
        )

        with tifffile.TiffFile(out) as handle:
            page = handle.pages[0]
            assert page.tags["XResolution"].value == (4, 1)
            assert page.tags["YResolution"].value == (2, 1)

    def test_unparsable_resolution_is_dropped_not_fatal(
        self, conv_worker, tmp_path
    ):
        out = tmp_path / "badresolution.tif"
        data = np.arange(20, dtype=np.uint8).reshape(4, 5)

        assert (
            conv_worker._save_tif(
                data, str(out), {"resolution": ("wide", "tall")}
            )
            is True
        )
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_unwritable_target_becomes_a_conversion_error(
        self, conv_worker, tmp_path
    ):
        out = tmp_path / "missing_dir" / "x.tif"
        with pytest.raises(fc.ConversionError, match="TIF save failed"):
            conv_worker._save_tif(np.zeros((2, 2), np.uint8), str(out), {})

    def test_a_four_dimensional_dask_array_is_routed_to_chunked_writing(
        self, conv_worker, tmp_path
    ):
        source = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
        out = tmp_path / "fourd.tif"

        assert (
            conv_worker._save_tif(
                da.from_array(source, chunks=1), str(out), {}
            )
            is True
        )

        # The chunked writer streams every plane into a single series via
        # one tifffile.imwrite(data=<generator>, shape=..., dtype=...)
        # call, so the plain read recovers every timepoint, not just the
        # first. A prior version called writer.write() once per
        # leading-axis index, which starts a new *series* each time --
        # every pixel still reached disk, but tifffile.imread() only ever
        # saw the first series back.
        with tifffile.TiffFile(out) as handle:
            assert len(handle.series) == 1
        np.testing.assert_array_equal(tifffile.imread(out), source)

    def test_an_oversized_dask_array_is_pushed_towards_zarr(
        self, conv_worker, tmp_path
    ):
        # 7 GB: past the 6 GB Dask ceiling but under the 8 GB hard stop, so
        # this must be the Dask-specific guard, not the generic one.
        out = tmp_path / "hugedask.tif"
        array = da.zeros((7 * 1024**3,), dtype=np.uint8)

        with pytest.raises(MemoryError, match="Dask array too large for TIF"):
            conv_worker._save_tif(array, str(out), {})
        assert not out.exists()

    def test_three_dimensional_dask_is_computed_in_one_go(
        self, conv_worker, tmp_path
    ):
        source = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        out = tmp_path / "chunked3d.tif"

        result = conv_worker._save_tif_chunked_dask(
            da.from_array(source, chunks=1), str(out), False
        )

        assert result is True
        np.testing.assert_array_equal(tifffile.imread(out), source)

    def test_chunked_write_to_a_missing_folder_raises(
        self, conv_worker, tmp_path
    ):
        out = tmp_path / "missing_dir" / "x.tif"
        array = da.zeros((2, 2, 3, 4), dtype=np.uint8)
        with pytest.raises(
            fc.ConversionError, match="Chunked TIF writing failed"
        ):
            conv_worker._save_tif_chunked_dask(array, str(out), False)


def read_ome_multiscale(store_path):
    """Return the first multiscales block written into ``zarr.json``."""
    meta = json.loads((store_path / "zarr.json").read_text())
    return meta["attributes"], meta["attributes"]["ome"]["multiscales"][0]


class TestSaveZarrBranches:
    def test_axes_are_reordered_to_tczyx(self, conv_worker, tmp_path):
        data = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.uint8).reshape(
            2, 3, 2, 4, 5
        )  # TZCYX
        out = tmp_path / "reordered.zarr"

        assert (
            conv_worker._save_zarr(
                data, str(out), {"axes": "TZCYX"}, "sample", 0
            )
            is True
        )

        attributes, multiscale = read_ome_multiscale(out)
        assert [ax["name"] for ax in multiscale["axes"]] == [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]
        # The axis *labels* alone prove nothing: they come from the metadata
        # string, not from the array. Pin the pixels so that dropping the
        # ``transpose`` -- which would leave a TZCYX volume mislabelled as
        # TCZYX -- fails here.
        stored = zarr.open_group(str(out), mode="r")["0"]
        assert stored.shape == (2, 2, 3, 4, 5)
        np.testing.assert_array_equal(
            stored[:], np.transpose(data, (0, 2, 1, 3, 4))
        )
        assert attributes["name"] == "sample"

    def test_physical_scale_reaches_the_coordinate_transformations(
        self, conv_worker, tmp_path
    ):
        """Resolution/spacing must survive as OME-Zarr scale metadata."""
        data = np.zeros((2, 3, 2, 4, 5), dtype=np.uint8)  # TZCYX
        out = tmp_path / "scaled.zarr"

        assert (
            conv_worker._save_zarr(
                data,
                str(out),
                {
                    "axes": "TZCYX",
                    "resolution": (4.0, 2.0),  # pixels per micrometre
                    "spacing": 1.5,  # micrometres per z step
                    "unit": "um",
                },
                "scaled",
                3,
            )
            is True
        )

        _attributes, multiscale = read_ome_multiscale(out)
        (dataset,) = multiscale["datasets"]
        assert dataset["path"] == "0"
        # t and c stay unscaled; z is the spacing; y/x are 1/resolution.
        # ome-zarr may additionally emit an identity translation transform
        # alongside the scale; that's harmless, so only the scale is pinned.
        (scale_transform,) = [
            ct
            for ct in dataset["coordinateTransformations"]
            if ct["type"] == "scale"
        ]
        assert scale_transform == {
            "type": "scale",
            "scale": [1.0, 1.0, 1.5, 0.5, 0.25],
        }

    def test_the_layer_name_carries_the_series_index(
        self, conv_worker, tmp_path
    ):
        out = tmp_path / "named.zarr"
        conv_worker._save_zarr(
            np.zeros((2, 3, 4), np.uint8), str(out), {"axes": "zyx"}, "plate", 2
        )
        attributes, _multiscale = read_ome_multiscale(out)
        assert attributes["name"] == "plate_series_2"

    def test_oversized_chunks_are_rechunked_before_writing(
        self, conv_worker, tmp_path, capsys, monkeypatch
    ):
        recorded = {}

        def fake_write_image(image, group, axes, **kwargs):
            recorded["chunksize"] = image.chunksize
            recorded["shape"] = image.shape
            recorded["dtype"] = image.dtype
            recorded["axes"] = axes

        monkeypatch.setattr(fc, "write_image", fake_write_image)
        # One 2.1 GB chunk: above the 1.5 GB codec limit.
        array = da.zeros(
            (256, 2048, 2048), dtype=np.uint16, chunks=(256, 2048, 2048)
        )

        result = conv_worker._save_zarr(
            array, str(tmp_path / "big.zarr"), {"axes": "zyx"}, "big", 0
        )

        assert result is True
        assert recorded["axes"] == "zyx"
        # Only Z is a rechunk target here (no T axis) so it must absorb
        # the full 1.5e9 / 2**31 reduction alone; Y and X -- the spatial
        # dims the comment promises to keep intact -- stay at 2048.
        assert recorded["chunksize"] == (178, 2048, 2048)
        chunk_bytes = np.prod(recorded["chunksize"]) * np.dtype(
            np.uint16
        ).itemsize
        assert chunk_bytes <= 1_500_000_000
        # Rechunking must not resize or retype the data itself.
        assert recorded["shape"] == (256, 2048, 2048)
        assert recorded["dtype"] == np.uint16
        out = capsys.readouterr().out
        assert "Rechunking" in out
        assert "Rechunked to" in out

    def test_write_failure_becomes_a_conversion_error(
        self, conv_worker, tmp_path, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise OSError("store is read-only")

        monkeypatch.setattr(fc, "write_image", boom)
        with pytest.raises(fc.ConversionError, match="ZARR save failed"):
            conv_worker._save_zarr(
                np.zeros((2, 3, 4), np.uint8),
                str(tmp_path / "fail.zarr"),
                {"axes": "zyx"},
                "fail",
                0,
            )

    def test_missing_axes_metadata_defaults_to_zyx(
        self, conv_worker, tmp_path
    ):
        data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        out = tmp_path / "noaxes.zarr"
        assert (
            conv_worker._save_zarr(data, str(out), {}, "noaxes", 0) is True
        )
        _attributes, multiscale = read_ome_multiscale(out)
        assert [ax["name"] for ax in multiscale["axes"]] == ["z", "y", "x"]
        # Without axes metadata there is nothing to scale by, and the volume
        # must be stored verbatim.
        # ome-zarr may additionally emit an identity translation transform
        # alongside the scale; that's harmless, so only the scale is pinned.
        (scale_transform,) = [
            ct
            for ct in multiscale["datasets"][0]["coordinateTransformations"]
            if ct["type"] == "scale"
        ]
        assert scale_transform == {"type": "scale", "scale": [1.0, 1.0, 1.0]}
        np.testing.assert_array_equal(
            zarr.open_group(str(out), mode="r")["0"][:], data
        )


class TestBuildScaleTransform:
    """Pixel sizes are inverted into micrometres-per-pixel, per axis."""

    def test_each_axis_gets_its_own_physical_size(self, conv_worker):
        transform = conv_worker._build_scale_transform(
            {"resolution": (4.0, 2.0), "spacing": 1.5, "unit": "um"},
            "tczyx",
            (2, 2, 3, 4, 5),
        )
        assert transform == {
            "type": "scale",
            "scale": [1.0, 1.0, 1.5, 0.5, 0.25],
        }

    def test_axis_order_drives_the_placement(self, conv_worker):
        transform = conv_worker._build_scale_transform(
            {"resolution": (4.0, 2.0), "spacing": 1.5}, "zyx", (3, 4, 5)
        )
        assert transform["scale"] == [1.5, 0.5, 0.25]

    def test_absent_metadata_leaves_every_axis_unscaled(self, conv_worker):
        transform = conv_worker._build_scale_transform({}, "zyx", (3, 4, 5))
        assert transform["scale"] == [1.0, 1.0, 1.0]

    def test_nonpositive_values_are_ignored(self, conv_worker):
        transform = conv_worker._build_scale_transform(
            {"resolution": (0.0, 2.0), "spacing": 0.0}, "zyx", (3, 4, 5)
        )
        assert transform["scale"] == [1.0, 0.5, 1.0]

    def test_no_axes_means_no_transform(self, conv_worker):
        assert conv_worker._build_scale_transform(
            {"resolution": (4.0, 2.0)}, "", (3, 4, 5)
        ) is None


class TestPyramidNamingFixErrors:
    def test_corrupt_metadata_does_not_break_the_rename(
        self, conv_worker, tmp_path
    ):
        store = tmp_path / "broken.zarr"
        store.mkdir()
        (store / "s0").mkdir()
        (store / "zarr.json").write_text("{not json")

        conv_worker._fix_ome_zarr_pyramid_naming(store)

        assert (store / "0").is_dir()
        assert (store / "zarr.json").read_text() == "{not json"

    def test_a_plain_file_is_survivable(self, conv_worker, tmp_path):
        plain = tmp_path / "not_a_store"
        plain.write_bytes(b"")

        conv_worker._fix_ome_zarr_pyramid_naming(plain)

        # Handed a file rather than a store, the fixer must leave it
        # exactly as it found it instead of renaming or unlinking it.
        assert plain.is_file()
        assert plain.read_bytes() == b""
        assert list(tmp_path.iterdir()) == [plain]


# --------------------------------------------------------------------- #
# SeriesDetailWidget
# --------------------------------------------------------------------- #
class TestSeriesDetailSetFile:
    """``set_file`` populates the selector and reports the size estimate."""

    def test_series_are_listed_and_size_reported(
        self, widget, tmp_path, monkeypatch, reset_stub_loader
    ):
        sample = tmp_path / "sample.lif"
        sample.write_bytes(b"x" * 1024)
        reset_stub_loader.series_count = 3
        monkeypatch.setattr(
            widget, "get_file_loader", lambda filepath: reset_stub_loader
        )
        detail = widget.series_widget

        detail.set_file(str(sample))

        assert detail.max_series == 3
        assert detail.series_selector.count() == 3
        assert detail.series_selector.itemData(2) == 2
        assert detail.info_label.text() == (
            "File contains 3 series (estimated size: 0.00GB)"
        )
        # A small file must leave the format selection on TIF.
        assert widget.tif_radio.isChecked() is True
        assert widget.zarr_radio.isChecked() is False

    def test_a_large_file_switches_the_format_to_zarr(
        self, widget, tmp_path, monkeypatch, reset_stub_loader
    ):
        sample = tmp_path / "huge.lif"
        sample.write_bytes(b"x")
        monkeypatch.setattr(
            widget, "get_file_loader", lambda filepath: reset_stub_loader
        )
        detail = widget.series_widget
        monkeypatch.setattr(
            detail, "_estimate_file_size", lambda filepath, loader: 9.5
        )

        detail.set_file(str(sample))

        assert detail.info_label.text() == (
            "File contains 1 series (estimated size: 9.50GB)"
        )
        assert widget.zarr_radio.isChecked() is True
        assert widget.tif_radio.isChecked() is False
        assert widget.status_label.text() == (
            "Auto-selected ZARR format for large file (>4GB)"
        )

    def test_missing_loader_is_reported_in_the_label(
        self, widget, monkeypatch
    ):
        monkeypatch.setattr(widget, "get_file_loader", lambda filepath: None)
        detail = widget.series_widget

        detail.set_file("/data/mystery.xyz")

        assert detail.info_label.text().startswith("Error:")
        assert "No loader available" in detail.info_label.text()

    def test_size_estimation_failure_keeps_the_series_count(
        self, widget, tmp_path, monkeypatch, reset_stub_loader
    ):
        sample = tmp_path / "sample.lif"
        sample.write_bytes(b"x")
        reset_stub_loader.series_count = 2
        monkeypatch.setattr(
            widget, "get_file_loader", lambda filepath: reset_stub_loader
        )

        def boom(filepath):
            raise OSError("stat failed")

        monkeypatch.setattr(widget, "get_file_type", boom)
        detail = widget.series_widget

        detail.set_file(str(sample))

        assert detail.info_label.text() == "File contains 2 series"

    def test_previously_exported_file_disables_the_selector(
        self, widget, tmp_path, monkeypatch, reset_stub_loader
    ):
        sample = tmp_path / "sample.lif"
        sample.write_bytes(b"x")
        widget.export_all_series[str(sample)] = True
        monkeypatch.setattr(
            widget, "get_file_loader", lambda filepath: reset_stub_loader
        )
        detail = widget.series_widget

        detail.set_file(str(sample))

        assert detail.export_all_checkbox.isChecked() is True
        assert detail.series_selector.isEnabled() is False


class TestSeriesDetailToggleExportAll:
    def test_enabling_export_all_selects_series_zero(
        self, widget, reset_stub_loader
    ):
        detail = widget.series_widget
        detail.current_file = "/data/a.lif"

        detail.toggle_export_all(True)

        assert widget.export_all_series["/data/a.lif"] is True
        assert widget.selected_series["/data/a.lif"] == 0
        assert detail.series_selector.isEnabled() is False

    def test_disabling_export_all_reselects_the_current_index(self, widget):
        detail = widget.series_widget
        detail.current_file = "/data/a.lif"
        detail.max_series = 2
        detail.series_selector.blockSignals(True)
        detail.series_selector.addItem("Series 0", 0)
        detail.series_selector.addItem("Series 1", 1)
        detail.series_selector.setCurrentIndex(1)
        detail.series_selector.blockSignals(False)

        detail.toggle_export_all(False)

        assert widget.export_all_series["/data/a.lif"] is False
        assert widget.selected_series["/data/a.lif"] == 1
        assert detail.series_selector.isEnabled() is True

    def test_toggle_without_a_file_does_nothing(self, widget):
        detail = widget.series_widget
        detail.current_file = None

        detail.toggle_export_all(True)

        assert widget.export_all_series == {}


class TestEstimateFileSize:
    def test_nd2_size_comes_from_the_reader_dimensions(
        self, widget, monkeypatch, tmp_path
    ):
        handle = FakeND2File(
            {"T": 2, "Y": 1024, "X": 1024}, dtype=np.dtype(np.uint16)
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle))
        monkeypatch.setattr(widget, "get_file_type", lambda filepath: "ND2")
        detail = widget.series_widget

        size = detail._estimate_file_size(str(tmp_path / "a.nd2"), None)

        assert size == pytest.approx(2 * 1024 * 1024 * 2 / (1024**3))

    def test_nd2_reader_failure_falls_back_to_the_file_size(
        self, widget, monkeypatch, tmp_path
    ):
        path = tmp_path / "a.nd2"
        path.write_bytes(b"x" * 2048)
        monkeypatch.setattr(fc, "nd2", nd2_module(OSError("cannot open")))
        monkeypatch.setattr(widget, "get_file_type", lambda filepath: "ND2")
        detail = widget.series_widget

        size = detail._estimate_file_size(str(path), None)

        assert size == pytest.approx(2048 / (1024**3))

    def test_missing_file_estimates_zero(self, widget, tmp_path):
        detail = widget.series_widget
        assert detail._estimate_file_size(str(tmp_path / "gone.lif"), None) == 0.0


class TestSeriesSelection:
    def test_selecting_an_index_records_it_on_the_parent(self, widget):
        detail = widget.series_widget
        detail.current_file = "/data/a.lif"
        detail.max_series = 3
        detail.series_selector.blockSignals(True)
        detail.series_selector.addItem("Series 2", 2)
        detail.series_selector.blockSignals(False)

        detail.series_selected(0)

        assert widget.selected_series["/data/a.lif"] == 2

    def test_negative_index_is_ignored(self, widget):
        detail = widget.series_widget
        detail.current_file = "/data/a.lif"
        detail.series_selected(-1)
        assert widget.selected_series == {}

    def test_out_of_range_index_raises(self, widget):
        detail = widget.series_widget
        detail.current_file = "/data/a.lif"
        detail.max_series = 1
        detail.series_selector.blockSignals(True)
        detail.series_selector.addItem("Series 5", 5)
        detail.series_selector.blockSignals(False)

        with pytest.raises(
            fc.SeriesIndexError, match=r"Series index 5 out of range \(max: 0\)"
        ):
            detail.series_selected(0)


class TestPreviewSeries:
    def _arm(self, widget, series_data_index):
        detail = widget.series_widget
        detail.current_file = "/data/a.lif"
        detail.max_series = 2
        detail.series_selector.blockSignals(True)
        detail.series_selector.addItem("Series", series_data_index)
        detail.series_selector.setCurrentIndex(0)
        detail.series_selector.blockSignals(False)
        return detail

    def test_no_file_selected_is_a_no_op(self, widget):
        detail = widget.series_widget
        detail.current_file = None
        detail.preview_series()
        assert widget.viewer.added == []

    def test_out_of_range_index_reports_in_the_label(self, widget):
        detail = self._arm(widget, 9)
        detail.preview_series()
        assert detail.info_label.text() == "Error: Series index out of range"
        assert widget.viewer.added == []

    def test_successful_preview_adds_a_named_layer(
        self, widget, monkeypatch, reset_stub_loader
    ):
        detail = self._arm(widget, 1)
        monkeypatch.setattr(
            widget, "get_file_loader", lambda filepath: reset_stub_loader
        )

        detail.preview_series()

        assert len(widget.viewer.added) == 1
        data, name = widget.viewer.added[0]
        assert name == "a_series_1"
        np.testing.assert_array_equal(data, reset_stub_loader.data)
        assert widget.viewer.status == "Previewing a_series_1"

    def test_load_failure_warns_instead_of_raising(
        self, widget, monkeypatch, reset_stub_loader, modal_calls
    ):
        detail = self._arm(widget, 0)
        reset_stub_loader.load_error = fc.FileFormatError("cannot decode")
        monkeypatch.setattr(
            widget, "get_file_loader", lambda filepath: reset_stub_loader
        )

        detail.preview_series()

        assert widget.viewer.added == []
        assert "cannot decode" in widget.viewer.status
        assert modal_calls and modal_calls[0][0] == "warning"


class TestReorderDimensions:
    """The happy path actually permutes; the guards return the input."""

    def test_axes_metadata_drives_a_real_transpose(self, widget):
        data = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint8).reshape(
            2, 3, 4, 5, 6
        )  # TZCYX

        result = widget.series_widget._reorder_dimensions(
            data, {"axes": "TZCYX"}, target_order="CTZYX"
        )

        np.testing.assert_array_equal(
            result, np.transpose(data, (2, 0, 1, 3, 4))
        )
        assert result.shape == (4, 2, 3, 5, 6)

    def test_a_dask_array_stays_lazy(self, widget):
        source = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        lazy = da.from_array(source, chunks=1)

        result = widget.series_widget._reorder_dimensions(
            lazy, {"axes": "ZYX"}, target_order="YXZ"
        )

        assert isinstance(result, da.Array)
        np.testing.assert_array_equal(
            result.compute(), np.transpose(source, (1, 2, 0))
        )

    def test_an_axis_missing_from_the_source_returns_the_input(self, widget):
        data = np.zeros((2, 3, 4), np.uint8)
        result = widget.series_widget._reorder_dimensions(
            data, {"axes": "ZYX"}, target_order="CTZ"
        )
        assert result is data

    def test_a_length_mismatch_returns_the_input(self, widget):
        data = np.zeros((2, 3, 4), np.uint8)
        result = widget.series_widget._reorder_dimensions(
            data, {"axes": "TZCYX"}, target_order="ZYX"
        )
        assert result is data

    def test_metadata_without_axes_returns_the_input(self, widget):
        data = np.zeros((2, 3), np.uint8)
        assert (
            widget.series_widget._reorder_dimensions(data, {}, "YX") is data
        )

    def test_a_transpose_error_returns_the_input(self, widget, capsys):
        class ExplodingArray:
            shape = (2, 3)

            def transpose(self, order):
                raise ValueError("cannot transpose")

            dask = True

        data = ExplodingArray()
        detail = widget.series_widget

        result = detail._reorder_dimensions(
            data, {"axes": "YX"}, target_order="XY"
        )

        assert result is data
        assert "Dimension reordering failed" in capsys.readouterr().out


# --------------------------------------------------------------------- #
# MicroscopyImageConverterWidget
# --------------------------------------------------------------------- #
def fake_file_dialog(results):
    """A ``QFileDialog`` stand-in returning/raising from ``results``."""
    calls = []

    class FakeQFileDialog:
        ShowDirsOnly = 1
        DontResolveSymlinks = 2
        DontUseNativeDialog = 4

        @staticmethod
        def getExistingDirectory(parent, caption, directory, options):
            calls.append((caption, directory, options))
            outcome = results[min(len(calls) - 1, len(results) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    return FakeQFileDialog, calls


class TestBrowseDialogs:
    """The folder pickers write back into the line edits."""

    def test_input_folder_is_written_back(self, widget, monkeypatch):
        dialog, calls = fake_file_dialog(["/picked/input"])
        monkeypatch.setattr(fc, "QFileDialog", dialog)

        widget.browse_folder()

        assert widget.folder_edit.text() == "/picked/input"
        assert calls[0][0] == "Select Input Folder"

    def test_cancelling_leaves_the_field_alone(self, widget, monkeypatch):
        widget.folder_edit.setText("/keep/me")
        dialog, _calls = fake_file_dialog([""])
        monkeypatch.setattr(fc, "QFileDialog", dialog)

        widget.browse_folder()

        assert widget.folder_edit.text() == "/keep/me"

    def test_native_dialog_failure_retries_non_native(
        self, widget, monkeypatch
    ):
        dialog, calls = fake_file_dialog(
            [RuntimeError("native dialog crashed"), "/picked/input"]
        )
        monkeypatch.setattr(fc, "QFileDialog", dialog)

        widget.browse_folder()

        assert widget.folder_edit.text() == "/picked/input"
        assert len(calls) == 2
        assert "Browse fallback used" in widget.status_label.text()

    def test_output_folder_is_written_back(self, widget, monkeypatch):
        dialog, calls = fake_file_dialog(["/picked/output"])
        monkeypatch.setattr(fc, "QFileDialog", dialog)

        widget.browse_output()

        assert widget.output_edit.text() == "/picked/output"
        assert calls[0][0] == "Select Output Folder"

    def test_output_browse_starts_at_the_input_folder(
        self, widget, monkeypatch, tmp_path
    ):
        widget.folder_edit.setText(str(tmp_path))
        dialog, calls = fake_file_dialog([""])
        monkeypatch.setattr(fc, "QFileDialog", dialog)

        widget.browse_output()

        assert calls[0][1] == str(tmp_path)

    def test_output_browse_falls_back_to_non_native(
        self, widget, monkeypatch
    ):
        dialog, calls = fake_file_dialog(
            [TypeError("bad options"), "/picked/output"]
        )
        monkeypatch.setattr(fc, "QFileDialog", dialog)

        widget.browse_output()

        assert widget.output_edit.text() == "/picked/output"
        assert len(calls) == 2
        assert "Browse fallback used" in widget.status_label.text()


class TestScanFolder:
    def test_empty_folder_field_is_rejected(self, widget):
        widget.folder_edit.setText("")
        widget.scan_folder()
        assert widget.status_label.text() == "Please select a valid folder"
        assert widget.scan_worker is None

    def test_nonexistent_folder_is_rejected(self, widget, tmp_path):
        widget.folder_edit.setText(str(tmp_path / "nope"))
        widget.scan_folder()
        assert widget.status_label.text() == "Please select a valid folder"
        assert widget.scan_worker is None

    def test_a_worker_is_started_with_the_parsed_filters(
        self, widget, tmp_path, monkeypatch
    ):
        started = []
        monkeypatch.setattr(
            fc.ScanFolderWorker, "start", lambda self: started.append(self)
        )
        widget.folder_edit.setText(str(tmp_path))
        widget.filter_edit.setText(".lif, .czi ,")
        widget.files_table.add_file("/old/a.lif", "LIF", 1)

        widget.scan_folder()

        assert started == [widget.scan_worker]
        assert widget.scan_worker.filters == [".lif", ".czi"]
        assert widget.scan_worker.folder == str(tmp_path)
        assert widget.files_table.rowCount() == 0
        assert widget.files_table.file_data == {}
        assert widget.status_label.text() == "Scanning folder..."

    def test_blank_filters_fall_back_to_the_defaults(
        self, widget, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(fc.ScanFolderWorker, "start", lambda self: None)
        widget.folder_edit.setText(str(tmp_path))
        widget.filter_edit.setText("  ,  ")

        widget.scan_folder()

        assert widget.scan_worker.filters == [
            ".lif",
            ".nd2",
            ".ndpi",
            ".czi",
        ]

    def test_progress_is_scaled_to_percent(self, widget):
        widget.update_scan_progress(3, 4)
        assert widget.scan_progress.value() == 75

    def test_zero_total_leaves_the_bar_alone(self, widget):
        widget.scan_progress.setValue(42)
        widget.update_scan_progress(0, 0)
        assert widget.scan_progress.value() == 42


class TestFileTypeDetection:
    """``get_file_type`` is what decides whether a scan hit is offered."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a.lif", "LIF"),
            ("a.LIF", "LIF"),
            ("a.nd2", "ND2"),
            ("a.ndpi", "Slide"),
            ("a.svs", "Slide"),
            ("a.czi", "CZI"),
            ("a.tif", "Unknown"),
        ],
    )
    def test_extensions_map_to_their_loader_label(
        self, widget, tmp_path, name, expected
    ):
        assert widget.get_file_type(str(tmp_path / name)) == expected

    def test_an_acquifer_directory_is_detected_by_its_contents(
        self, widget, tmp_path
    ):
        plate = tmp_path / "plate"
        plate.mkdir()
        (plate / "PlateLayout").write_bytes(b"")
        tifffile.imwrite(
            plate / "A01--PX0250.tif", np.zeros((2, 2), np.uint8)
        )

        assert widget.get_file_type(str(plate)) == "Acquifer"
        assert widget.get_file_type(str(tmp_path)) == "Unknown"

    def test_the_first_accepting_loader_wins(self, widget, tmp_path):
        assert (
            widget.get_file_loader(str(tmp_path / "a.nd2")) is fc.ND2Loader
        )
        assert widget.get_file_loader(str(tmp_path / "a.xyz")) is None


class TestProcessFoundFiles:
    def test_each_file_gets_a_row_with_its_series_count(
        self, widget, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.series_count = 4
        widget.loaders = [reset_stub_loader]
        paths = [str(tmp_path / "a.lif"), str(tmp_path / "b.nd2")]

        widget.process_found_files(paths)

        assert widget.files_table.rowCount() == 2
        assert {
            info["series_count"]
            for info in widget.files_table.file_data.values()
        } == {4}
        assert widget.status_label.text() == "Found 2 files"

    def test_a_failing_series_count_still_adds_the_row(
        self, widget, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.count_error = ValueError("unreadable")
        widget.loaders = [reset_stub_loader]

        widget.process_found_files([str(tmp_path / "a.lif")])

        info = widget.files_table.file_data[str(tmp_path / "a.lif")]
        assert info["series_count"] == 0

    def test_files_without_a_loader_are_skipped(self, widget, tmp_path):
        # Named for what actually gates this: ``get_file_type`` returns the
        # truthy string "Unknown" for a .txt, so it is the loader lookup --
        # not the type check -- that drops the row.
        assert widget.get_file_type(str(tmp_path / "a.txt")) == "Unknown"
        widget.loaders = []

        widget.process_found_files([str(tmp_path / "a.txt")])

        assert widget.files_table.rowCount() == 0
        assert widget.files_table.file_data == {}
        assert widget.status_label.text() == "Found 1 files"


class TestErrorAndCancel:
    def test_scan_errors_reach_the_status_label_and_a_dialog(
        self, widget, modal_calls
    ):
        widget.show_error("disk unplugged")

        assert widget.status_label.text() == "Error: disk unplugged"
        assert ("critical", "disk unplugged") in modal_calls

    def test_cancel_stops_both_workers(self, widget):
        class StubWorker:
            def __init__(self):
                self.terminated = False
                self.stopped = False
                self.deleted = False

            def isRunning(self):
                return True

            def terminate(self):
                self.terminated = True

            def stop(self):
                self.stopped = True

            def deleteLater(self):
                self.deleted = True

        scan, conv = StubWorker(), StubWorker()
        widget.scan_worker = scan
        widget.conversion_worker = conv

        widget.cancel_operation()

        assert scan.terminated and scan.deleted
        assert conv.stopped and conv.deleted
        assert widget.scan_worker is None
        assert widget.conversion_worker is None
        assert widget.status_label.text() == "Operation cancelled"

    def test_cancel_without_workers_is_safe(self, widget):
        widget.cancel_operation()
        assert widget.status_label.text() == "Operation cancelled"


class TestLoadImage:
    def test_image_is_added_under_the_file_stem(
        self, widget, reset_stub_loader
    ):
        widget.loaders = [reset_stub_loader]

        widget.load_image("/data/sample.lif")

        assert len(widget.viewer.added) == 1
        data, name = widget.viewer.added[0]
        assert name == "sample"
        np.testing.assert_array_equal(data, reset_stub_loader.data)
        assert widget.viewer.status == "Loaded sample.lif"

    def test_unsupported_format_warns(self, widget, modal_calls):
        widget.loaders = []

        widget.load_image("/data/sample.xyz")

        assert widget.viewer.added == []
        assert "Unsupported file format" in widget.viewer.status
        assert modal_calls and modal_calls[0][0] == "warning"

    def test_loader_failure_warns(
        self, widget, reset_stub_loader, modal_calls
    ):
        reset_stub_loader.load_error = OSError("read error")
        widget.loaders = [reset_stub_loader]

        widget.load_image("/data/sample.lif")

        assert widget.viewer.added == []
        assert "read error" in widget.viewer.status
        assert modal_calls and modal_calls[0][0] == "warning"


class TestConvertFiles:
    @pytest.fixture
    def started(self, widget, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            widget,
            "_start_conversion_worker",
            lambda files, folder: recorded.append((files, folder)),
        )
        return recorded

    def test_no_selected_file_is_reported(self, widget, started):
        widget.series_widget.current_file = None

        widget.convert_files()

        assert started == []
        assert "select a file" in widget.status_label.text()

    def test_the_current_file_is_queued_with_its_series(
        self, widget, started, tmp_path
    ):
        widget.series_widget.current_file = "/data/a.lif"
        widget.selected_series["/data/a.lif"] = 2
        widget.output_edit.setText(str(tmp_path / "out"))

        widget.convert_files()

        assert started == [([("/data/a.lif", 2)], str(tmp_path / "out"))]

    def test_output_folder_defaults_next_to_the_input(
        self, widget, started, tmp_path
    ):
        widget.series_widget.current_file = "/data/a.lif"
        widget.folder_edit.setText(str(tmp_path))
        widget.output_edit.setText("")

        widget.convert_files()

        assert started[0][1] == str(tmp_path / "converted")
        assert widget.selected_series["/data/a.lif"] == 0

    def test_export_all_queues_every_series(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.series_count = 3
        widget.loaders = [reset_stub_loader]
        widget.series_widget.current_file = "/data/a.lif"
        widget.export_all_series["/data/a.lif"] = True
        widget.output_edit.setText(str(tmp_path))

        widget.convert_files()

        assert started[0][0] == [
            ("/data/a.lif", 0),
            ("/data/a.lif", 1),
            ("/data/a.lif", 2),
        ]

    def test_a_failing_series_count_aborts_with_a_message(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.count_error = ValueError("unreadable")
        widget.loaders = [reset_stub_loader]
        widget.series_widget.current_file = "/data/a.lif"
        widget.export_all_series["/data/a.lif"] = True
        widget.output_edit.setText(str(tmp_path))

        widget.convert_files()

        assert started == []
        assert "Error getting series count" in widget.status_label.text()

    def test_zero_series_leaves_nothing_to_convert(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.series_count = 0
        widget.loaders = [reset_stub_loader]
        widget.series_widget.current_file = "/data/a.lif"
        widget.export_all_series["/data/a.lif"] = True
        widget.output_edit.setText(str(tmp_path))

        widget.convert_files()

        assert started == []
        assert widget.status_label.text() == "No valid files to convert"

    def test_unwritable_output_folder_stops_before_starting(
        self, widget, started, tmp_path
    ):
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"")
        widget.series_widget.current_file = "/data/a.lif"
        widget.output_edit.setText(str(blocker / "sub"))

        widget.convert_files()

        assert started == []
        assert "Cannot create output folder" in widget.status_label.text()

    def test_unexpected_errors_surface_as_a_dialog(
        self, widget, monkeypatch, modal_calls, tmp_path
    ):
        widget.series_widget.current_file = "/data/a.lif"
        widget.output_edit.setText(str(tmp_path))

        def boom(folder):
            raise OSError("filesystem exploded")

        monkeypatch.setattr(widget, "_validate_output_folder", boom)

        widget.convert_files()

        assert modal_calls and modal_calls[0][0] == "critical"
        assert "filesystem exploded" in modal_calls[0][1]


class TestConvertAllFiles:
    @pytest.fixture
    def started(self, widget, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            widget,
            "_start_conversion_worker",
            lambda files, folder: recorded.append((files, folder)),
        )
        return recorded

    def test_an_empty_table_is_reported(self, widget, started):
        widget.convert_all_files()
        assert started == []
        assert widget.status_label.text() == (
            "No files available for conversion"
        )

    def test_single_series_files_are_queued_once(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        widget.loaders = [reset_stub_loader]
        widget.files_table.add_file("/data/a.ndpi", "Slide", 1)
        widget.output_edit.setText(str(tmp_path))

        widget.convert_all_files()

        assert started[0][0] == [("/data/a.ndpi", 0)]

    def test_multi_series_files_are_fully_expanded(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.series_count = 2
        widget.loaders = [reset_stub_loader]
        widget.files_table.add_file("/data/a.lif", "LIF", 2)
        widget.output_edit.setText(str(tmp_path))

        widget.convert_all_files()

        assert started[0][0] == [("/data/a.lif", 0), ("/data/a.lif", 1)]

    def test_a_failing_series_count_aborts(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.count_error = OSError("unreadable")
        widget.loaders = [reset_stub_loader]
        widget.files_table.add_file("/data/a.lif", "LIF", 2)
        widget.output_edit.setText(str(tmp_path))

        widget.convert_all_files()

        assert started == []
        assert "Error getting series count" in widget.status_label.text()

    def test_output_folder_defaults_next_to_the_input(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        widget.loaders = [reset_stub_loader]
        widget.files_table.add_file("/data/a.ndpi", "Slide", 1)
        widget.folder_edit.setText(str(tmp_path))
        widget.output_edit.setText("")

        widget.convert_all_files()

        assert started[0][1] == str(tmp_path / "converted")

    def test_unexpected_errors_surface_as_a_dialog(
        self, widget, monkeypatch, modal_calls, tmp_path
    ):
        widget.files_table.add_file("/data/a.lif", "LIF", 1)
        widget.output_edit.setText(str(tmp_path))

        def boom(folder):
            raise ValueError("bad path")

        monkeypatch.setattr(widget, "_validate_output_folder", boom)

        widget.convert_all_files()

        assert modal_calls and modal_calls[0][0] == "critical"
        assert "bad path" in modal_calls[0][1]


class TestConversionWorkerWiring:
    def test_the_worker_is_configured_from_the_widget_state(
        self, widget, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(fc.ConversionWorker, "start", lambda self: None)
        widget.zarr_radio.setChecked(True)
        files = [("/data/a.lif", 0), ("/data/a.lif", 1)]

        widget._start_conversion_worker(files, str(tmp_path))

        worker = widget.conversion_worker
        assert worker.files_to_convert == files
        assert worker.output_folder == str(tmp_path)
        assert worker.use_zarr is True
        assert worker.get_file_loader == widget.get_file_loader
        assert "Converting 2 files/series" in widget.status_label.text()

    def test_the_worker_signals_are_wired_to_the_widget(
        self, widget, monkeypatch, modal_calls, tmp_path
    ):
        """A configured-but-unconnected worker would look identical above."""
        monkeypatch.setattr(fc.ConversionWorker, "start", lambda self: None)
        widget._start_conversion_worker([("/data/a.lif", 0)], str(tmp_path))
        worker = widget.conversion_worker

        worker.progress.emit(1, 4, "a.lif")
        assert widget.conversion_progress.value() == 25
        assert "Converting a.lif (1/4)" in widget.status_label.text()

        worker.file_done.emit("/data/a.lif", False, "broken")
        assert modal_calls and modal_calls[-1][0] == "warning"

        widget.output_edit.setText(str(tmp_path))
        worker.finished.emit(2)
        assert "Successfully converted 2 files" in widget.status_label.text()
        assert widget.conversion_worker is None


class TestConversionFeedback:
    def test_progress_is_scaled_and_named(self, widget):
        widget.update_conversion_progress(1, 4, "a.lif")
        assert widget.conversion_progress.value() == 25
        assert "Converting a.lif (1/4)" in widget.status_label.text()

    def test_zero_total_leaves_the_bar_alone(self, widget):
        widget.conversion_progress.setValue(7)
        widget.update_conversion_progress(0, 0, "a.lif")
        assert widget.conversion_progress.value() == 7

    def test_success_is_only_logged(self, widget, modal_calls, capsys):
        widget.handle_conversion_result("/data/a.lif", True, "ok")
        assert modal_calls == []
        assert "Successfully converted: a.lif" in capsys.readouterr().out

    def test_skipped_files_do_not_warn(self, widget, modal_calls, capsys):
        widget.handle_conversion_result(
            "/data/a.czi", False, "SKIPPED: bad block"
        )
        assert modal_calls == []
        assert "Skipped file: a.czi" in capsys.readouterr().out

    def test_failures_warn_once(self, widget, modal_calls):
        widget.handle_conversion_result("/data/a.lif", False, "broken")
        assert modal_calls and modal_calls[0][0] == "warning"
        assert "a.lif" in modal_calls[0][1]

    def test_completion_reports_the_output_folder(self, widget, tmp_path):
        widget.output_edit.setText(str(tmp_path))

        widget.conversion_completed(3)

        assert "Successfully converted 3 files" in widget.status_label.text()
        assert str(tmp_path) in widget.status_label.text()
        assert widget.conversion_worker is None

    def test_completion_without_successes(self, widget, tmp_path):
        widget.folder_edit.setText(str(tmp_path))
        widget.output_edit.setText("")

        widget.conversion_completed(0)

        assert widget.status_label.text() == "No files were converted"

    def test_completion_cleans_up_the_worker(
        self, widget, monkeypatch, tmp_path
    ):
        class StubWorker:
            def __init__(self):
                self.deleted = False

            def deleteLater(self):
                self.deleted = True

        worker = StubWorker()
        widget.conversion_worker = worker
        widget.output_edit.setText(str(tmp_path))

        widget.conversion_completed(1)

        assert worker.deleted is True
        assert widget.conversion_worker is None


# --------------------------------------------------------------------- #
# Remaining guard branches
# --------------------------------------------------------------------- #
class TestSingleImageRowClick:
    def test_clicking_a_single_image_row_loads_it(
        self, widget, reset_stub_loader
    ):
        widget.loaders = [reset_stub_loader]
        widget.files_table.add_file("/data/slide.ndpi", "Slide", 0)

        widget.files_table.handle_cell_click(0, 0)

        assert widget.selected_series["/data/slide.ndpi"] == 0
        assert widget.viewer.added[0][1] == "slide"


class TestDebugTracebacks:
    def test_lif_failure_prints_a_traceback_in_debug_mode(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setenv("TMIDAS_CONVERSION_DEBUG", "1")

        def boom(filepath):
            raise ValueError("corrupt index")

        monkeypatch.setattr(fc, "LifFile", boom)

        with pytest.raises(fc.FileFormatError):
            fc.LIFLoader.load_series(str(tmp_path / "a.lif"), 0)

        assert "Traceback" in capsys.readouterr().err


class TestLIFDtypeProbeFailures:
    """A failing probe frame must not stop the rest of the series."""

    def test_numpy_loader_falls_back_to_uint16(self):
        image = FakeLifImage(
            nt=1, nz=1, channels=2, dtype=np.uint8, errors=[(0, 0, 0)]
        )
        result = fc.LIFLoader._load_numpy(image, 1, 1, 2, 4, 5)
        assert result.dtype == np.uint16
        assert result[0, 0, 1].max() == 2

    def test_chunked_loader_falls_back_to_uint16(self):
        image = FakeLifImage(
            nt=1, nz=1, channels=2, dtype=np.uint8, errors=[(0, 0, 0)]
        )
        result = fc.LIFLoader._load_chunked_numpy(image, 1, 1, 2, 4, 5)
        assert result.dtype == np.uint16

    def test_dask_loader_falls_back_to_uint16(self):
        image = FakeLifImage(
            nt=1, nz=1, channels=1, dtype=np.uint8, errors=[(0, 0, 0)]
        )
        result = fc.LIFLoader._load_as_dask(image, 1, 1, 1, 4, 5)
        assert result.dtype == np.uint16


class TestLIFMetadataScaleFailure:
    def test_unindexable_scale_is_ignored(self, monkeypatch, tmp_path):
        image = FakeLifImage(scale=5)  # truthy but has no len()
        monkeypatch.setattr(fc, "LifFile", lif_factory([image]))

        meta = fc.LIFLoader.get_metadata(str(tmp_path / "a.lif"), 0)

        assert meta["axes"] == "TZCYX"
        assert "resolution" not in meta


class TestND2RemainingPaths:
    def test_a_non_dask_slice_is_returned_as_is(
        self, monkeypatch, tmp_path
    ):
        source = np.arange(2 * 2 * 3, dtype=np.uint16).reshape(2, 2, 3)

        def imread(filepath, dask=False, xarray=False):
            raise ValueError("no xarray support")

        handle = FakeND2File(
            {"P": 2, "C": 2, "Y": 30000, "X": 30000}, dask_array=source
        )
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 1)

        np.testing.assert_array_equal(result, source[1])

    def test_large_single_position_file_is_read_lazily(
        self, monkeypatch, tmp_path
    ):
        seen = {}
        lazy = da.zeros((2, 4, 5), dtype=np.uint16)

        def imread(filepath, dask=False, xarray=False):
            seen["dask"] = dask
            return lazy

        handle = FakeND2File({"T": 2, "Y": 40000, "X": 40000})
        monkeypatch.setattr(fc, "nd2", nd2_module(handle, imread))

        result = fc.ND2Loader.load_series(str(tmp_path / "a.nd2"), 0)

        assert seen["dask"] is True
        assert result is lazy


class TestCZIRemainingPaths:
    def test_lazy_scene_planes_are_read_with_their_scene_id(
        self, monkeypatch, tmp_path
    ):
        def big_plane(plane, scene=None):
            return np.zeros((2048, 2048, 1), dtype=np.uint16)

        doc = FakeCziDoc(
            {
                "T": (0, 40),
                "Z": (0, 4),
                "C": (0, 2),
                "Y": (0, 2048),
                "X": (0, 2048),
            },
            scenes={5: None},
            plane=big_plane,
        )
        monkeypatch.setattr(fc, "pyczi", czi_module(doc))

        result = fc.CZILoader.load_series(str(tmp_path / "a.czi"), 0)
        assert result[0, 0, 0, 0, 0].compute() == 0
        assert {scene for _plane, scene in doc.reads} == {5}


class TestCZIScaleExtractionExtras:
    def test_non_dict_distance_entries_are_skipped(self):
        metadata = {
            "ImageDocument": {
                "Metadata": {
                    "Scaling": {
                        "Items": {
                            "Distance": [
                                "junk",
                                {"@Id": "X", "Value": "2.5E-07"},
                            ]
                        }
                    }
                }
            }
        }
        assert fc.CZILoader._extract_scale_from_xml(metadata, "X") == 0.25

    def test_legacy_nested_scaling_layout_is_understood(self):
        xml = (
            "<Scaling><Items>"
            '<Distance Name="pixel"><Sub Id="X"/>'
            "<Value>2.5E-07</Value></Distance>"
            "</Items></Scaling>"
        )
        assert fc.CZILoader._extract_scale_from_xml(xml, "X") == 0.25


class TestAcquiferRemainingPaths:
    def test_unlistable_directory_cannot_be_loaded(
        self, acquifer_dir, monkeypatch
    ):
        def boom(path):
            raise PermissionError("no access")

        monkeypatch.setattr(fc.os, "listdir", boom)
        assert fc.AcquiferLoader.can_load(str(acquifer_dir)) is False

    def test_a_dataset_without_dims_is_a_format_error(
        self, acquifer_dir, acquifer_reader
    ):
        acquifer_reader["dataset"] = object()
        with pytest.raises(
            fc.FileFormatError, match="Failed to load Acquifer series"
        ):
            fc.AcquiferLoader.load_series(str(acquifer_dir), 0)

    def test_unwalkable_directory_keeps_the_default_resolution(
        self, acquifer_dir, acquifer_reader, monkeypatch
    ):
        # Prime the cache first so only the pixel-size scan can fail.
        fc.AcquiferLoader._load_dataset(str(acquifer_dir))

        def boom(*args, **kwargs):
            raise OSError("no access")

        monkeypatch.setattr(fc.os, "walk", boom)

        meta = fc.AcquiferLoader.get_metadata(str(acquifer_dir), 0)

        assert meta["resolution"] == (1.0, 1.0)


class TestFormatToggleSender:
    def test_checking_zarr_unchecks_tif_and_back(self, widget):
        widget.zarr_radio.setChecked(True)
        assert widget.tif_radio.isChecked() is False

        widget.tif_radio.setChecked(True)
        assert widget.zarr_radio.isChecked() is False


class TestConvertAllGuards:
    @pytest.fixture
    def started(self, widget, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            widget,
            "_start_conversion_worker",
            lambda files, folder: recorded.append((files, folder)),
        )
        return recorded

    def test_an_uncreatable_output_folder_stops_the_run(
        self, widget, started, tmp_path
    ):
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"")
        widget.files_table.add_file("/data/a.lif", "LIF", 1)
        widget.output_edit.setText(str(blocker / "sub"))

        widget.convert_all_files()

        assert started == []
        assert "Cannot create output folder" in widget.status_label.text()

    def test_a_zero_series_file_leaves_nothing_to_convert(
        self, widget, started, tmp_path, reset_stub_loader
    ):
        reset_stub_loader.series_count = 0
        widget.loaders = [reset_stub_loader]
        widget.files_table.add_file("/data/a.lif", "LIF", 2)
        widget.output_edit.setText(str(tmp_path))

        widget.convert_all_files()

        assert started == []
        assert widget.status_label.text() == "No valid files to convert"


class TestValidateOutputFolder:
    def test_an_empty_path_is_rejected(self, widget):
        assert widget._validate_output_folder("") is False
        assert widget.status_label.text() == "Please specify an output folder"

    def test_a_missing_folder_is_created(self, widget, tmp_path):
        target = tmp_path / "fresh" / "nested"
        assert widget._validate_output_folder(str(target)) is True
        assert target.is_dir()

    def test_an_existing_writable_folder_is_accepted(self, widget, tmp_path):
        assert widget._validate_output_folder(str(tmp_path)) is True


class TestDockWidgetEntryPoint:
    def test_the_plugin_hook_returns_the_converter_factory(self):
        assert (
            fc.napari_experimental_provide_dock_widget()
            is fc.microscopy_converter
        )

    def test_the_factory_docks_a_converter_widget(self, qapp, modal_calls):
        docked = []

        class DockingViewer(FakeViewer):
            def __init__(self):
                super().__init__()
                self.window = types.SimpleNamespace(
                    add_dock_widget=lambda widget, name, area: docked.append(
                        (widget, name, area)
                    )
                )

        viewer = DockingViewer()

        created = fc.microscopy_converter(viewer)

        assert isinstance(created, fc.MicroscopyImageConverterWidget)
        assert docked == [
            (created, "Microscopy Image Converter", "right")
        ]


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores directory write permissions"
)
class TestOutputFolderPermissions:
    def test_a_read_only_folder_is_rejected(self, widget, tmp_path):
        folder = tmp_path / "readonly"
        folder.mkdir()
        folder.chmod(0o555)
        try:
            assert widget._validate_output_folder(str(folder)) is False
            assert widget.status_label.text() == (
                "Output folder is not writable"
            )
        finally:
            folder.chmod(0o755)

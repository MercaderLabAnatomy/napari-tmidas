# src/napari_tmidas/_tests/test_intensity_filter_compression_ome_coverage.py
"""
Branch coverage for three output/processing helpers.

* ``intensity_label_filter`` -- guard clauses, the reporting branches of the
  in-memory path and the streaming pass-1/pass-2 edge cases.
* ``file_compression`` -- the pzstd command builder and the ProcessingWorker
  monkey patch.  The module shells out and mutates two process-wide
  singletons, so every ``subprocess`` call is stubbed and the registry plus
  ``ProcessingWorker`` are snapshotted and restored around each test.
* ``ome_output_utils`` -- metadata readers, the lazy plane iterators and the
  two documented sharp edges (ome-zarr silently ignoring
  ``coordinate_transformations``, and ``photometric="minisblack"``).
"""

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import zarr

from napari_tmidas.processing_functions import intensity_label_filter as ilf
from napari_tmidas.processing_functions import ome_output_utils as ome

# ---------------------------------------------------------------------------
# ome_output_utils
# ---------------------------------------------------------------------------


class TestWriteImageAcceptsScale:
    """Pixel size only survives if the right keyword is chosen."""

    def test_missing_ome_zarr_falls_back_to_the_legacy_keyword(
        self, monkeypatch
    ):
        """No ome-zarr at all must not raise -- assume the old signature."""
        monkeypatch.setattr(ome, "_WRITE_IMAGE_ACCEPTS_SCALE", None)
        monkeypatch.setitem(sys.modules, "ome_zarr.writer", None)

        assert ome._write_image_accepts_scale() is False

        transform = {"type": "scale", "scale": [0.5, 0.5]}
        assert ome.physical_scale_kwargs(transform, "yx") == {
            "coordinate_transformations": [[transform]]
        }

    def test_answer_is_cached_after_the_first_probe(self, monkeypatch):
        monkeypatch.setattr(ome, "_WRITE_IMAGE_ACCEPTS_SCALE", True)
        monkeypatch.setitem(sys.modules, "ome_zarr.writer", None)

        # Cached: the (now broken) import is never attempted again.
        assert ome._write_image_accepts_scale() is True


class TestReadRootAttrs:
    """Both NGFF layouts are read, and neither can break on bad JSON."""

    def test_reads_zattrs(self, tmp_path):
        (tmp_path / ".zattrs").write_text('{"omero": {"channels": []}}')
        assert ome._read_root_attrs(str(tmp_path)) == {
            "omero": {"channels": []}
        }

    def test_malformed_zattrs_is_ignored(self, tmp_path):
        (tmp_path / ".zattrs").write_text("{not json")
        assert ome._read_root_attrs(str(tmp_path)) == {}

    def test_reads_zarr_json_attributes(self, tmp_path):
        (tmp_path / "zarr.json").write_text(
            json.dumps({"attributes": {"multiscales": [{"name": "x"}]}})
        )
        assert ome._read_root_attrs(str(tmp_path)) == {
            "multiscales": [{"name": "x"}]
        }

    def test_zarr_json_attributes_win_over_zattrs(self, tmp_path):
        (tmp_path / ".zattrs").write_text('{"tag": "old"}')
        (tmp_path / "zarr.json").write_text(
            json.dumps({"attributes": {"tag": "new"}})
        )
        assert ome._read_root_attrs(str(tmp_path))["tag"] == "new"

    def test_malformed_zarr_json_is_ignored(self, tmp_path):
        (tmp_path / "zarr.json").write_text("[[[")
        assert ome._read_root_attrs(str(tmp_path)) == {}

    def test_zarr_json_that_is_not_a_mapping_is_ignored(self, tmp_path):
        (tmp_path / "zarr.json").write_text("[1, 2, 3]")
        assert ome._read_root_attrs(str(tmp_path)) == {}

    def test_missing_source_returns_empty(self, tmp_path):
        assert ome._read_root_attrs(str(tmp_path / "nope")) == {}


class TestGetMultiscales:
    """v0.4 keeps multiscales at the root, v0.5 nests it under ``ome``."""

    def test_flat_key(self):
        assert ome._get_multiscales({"multiscales": [{"a": 1}]}) == [{"a": 1}]

    def test_nested_under_ome(self):
        attrs = {"ome": {"multiscales": [{"a": 2}]}}
        assert ome._get_multiscales(attrs) == [{"a": 2}]

    def test_empty_flat_key_falls_through_to_ome(self):
        attrs = {"multiscales": [], "ome": {"multiscales": [{"a": 3}]}}
        assert ome._get_multiscales(attrs) == [{"a": 3}]

    def test_neither_layout_present(self):
        assert ome._get_multiscales({"ome": {"multiscales": []}}) == []
        assert ome._get_multiscales({"ome": "not-a-dict"}) == []
        assert ome._get_multiscales({}) == []


class TestExtractSourcePhysicalScale:
    """Voxel spacing is read from OME-TIFF XML or from NGFF transforms."""

    def _zarr_source(self, tmp_path, attrs):
        source = tmp_path / "source.zarr"
        source.mkdir()
        (source / ".zattrs").write_text(json.dumps(attrs))
        return str(source)

    def test_no_source_path(self):
        assert ome._extract_source_physical_scale(None, "ZYX") == {}
        assert ome._extract_source_physical_scale("", "ZYX") == {}

    def test_tiff_source_is_dispatched_to_the_tiff_reader(self, tmp_path):
        source = tmp_path / "src.ome.tif"
        tifffile.imwrite(
            source,
            np.zeros((3, 8, 6), np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.25,
                "PhysicalSizeY": 0.25,
                "PhysicalSizeZ": 2.0,
            },
        )
        assert ome._extract_source_physical_scale(str(source), "ZYX") == {
            "Z": 2.0,
            "Y": 0.25,
            "X": 0.25,
        }

    def test_plain_tiff_has_no_ome_metadata(self, tmp_path):
        source = tmp_path / "plain.tif"
        tifffile.imwrite(source, np.zeros((4, 4), np.uint16))
        assert ome._extract_tiff_physical_scale(str(source), "YX") == {}

    def test_unreadable_tiff_is_swallowed(self, tmp_path):
        missing = tmp_path / "gone.tif"
        assert ome._extract_tiff_physical_scale(str(missing), "YX") == {}

    def test_multi_image_ome_uses_the_first_pixels_block(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "multi.ome.tif"
        tifffile.imwrite(
            source,
            np.zeros((4, 4), np.uint16),
            ome=True,
            photometric="minisblack",
        )
        monkeypatch.setattr(
            tifffile,
            "xml2dict",
            lambda *a, **k: {
                "OME": {
                    "Image": {
                        "Pixels": [
                            {"PhysicalSizeX": 1.5, "PhysicalSizeY": 1.5},
                            {"PhysicalSizeX": 9.9, "PhysicalSizeY": 9.9},
                        ]
                    }
                }
            },
        )
        assert ome._extract_tiff_physical_scale(str(source), "YX") == {
            "Y": 1.5,
            "X": 1.5,
        }

    def test_non_numeric_physical_size_is_dropped(self, tmp_path, monkeypatch):
        source = tmp_path / "bad.ome.tif"
        tifffile.imwrite(
            source,
            np.zeros((4, 4), np.uint16),
            ome=True,
            photometric="minisblack",
        )
        monkeypatch.setattr(
            tifffile,
            "xml2dict",
            lambda *a, **k: {
                "OME": {
                    "Image": {
                        "Pixels": {
                            "PhysicalSizeX": "not-a-number",
                            "PhysicalSizeY": 0.5,
                        }
                    }
                }
            },
        )
        assert ome._extract_tiff_physical_scale(str(source), "YX") == {
            "Y": 0.5
        }

    def test_pixels_that_is_not_a_mapping(self, tmp_path, monkeypatch):
        source = tmp_path / "odd.ome.tif"
        tifffile.imwrite(
            source,
            np.zeros((4, 4), np.uint16),
            ome=True,
            photometric="minisblack",
        )
        monkeypatch.setattr(
            tifffile,
            "xml2dict",
            lambda *a, **k: {"OME": {"Image": {"Pixels": "nonsense"}}},
        )
        assert ome._extract_tiff_physical_scale(str(source), "YX") == {}

    def test_zarr_transforms_are_keyed_by_axis_name(self, tmp_path):
        source = self._zarr_source(
            tmp_path,
            {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "t"},
                            {"name": "z"},
                            {"name": "y"},
                            {"name": "x"},
                        ],
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 4.0, 0.5, 0.25],
                                    }
                                ]
                            }
                        ],
                    }
                ]
            },
        )
        # Only spatial axes are reported, and T is skipped entirely.
        assert ome._extract_source_physical_scale(source, "TZYX") == {
            "Z": 4.0,
            "Y": 0.5,
            "X": 0.25,
        }

    def test_zarr_without_multiscales(self, tmp_path):
        source = self._zarr_source(tmp_path, {"omero": {}})
        assert ome._extract_source_physical_scale(source, "ZYX") == {}

    def test_zarr_multiscale_entry_is_not_a_mapping(self, tmp_path):
        source = self._zarr_source(tmp_path, {"multiscales": ["nope"]})
        assert ome._extract_source_physical_scale(source, "ZYX") == {}

    def test_zarr_without_datasets(self, tmp_path):
        source = self._zarr_source(
            tmp_path, {"multiscales": [{"axes": [], "datasets": []}]}
        )
        assert ome._extract_source_physical_scale(source, "ZYX") == {}

    def test_transform_with_the_wrong_arity_is_refused(self, tmp_path):
        """A three-axis scale on a two-axis image would be misassigned."""
        source = self._zarr_source(
            tmp_path,
            {
                "multiscales": [
                    {
                        "axes": [{"name": "y"}, {"name": "x"}],
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    "not-a-dict",
                                    {"type": "translation"},
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 2.0, 3.0],
                                    },
                                ]
                            }
                        ],
                    }
                ]
            },
        )
        assert ome._extract_source_physical_scale(source, "YX") == {}


class TestArrayNbytes:
    """Dense size must be computed without materializing anything."""

    def test_uses_the_arrays_own_dtype(self):
        array = np.zeros((3, 4, 5), dtype=np.uint16)
        assert ome._array_nbytes(array) == 3 * 4 * 5 * 2

    def test_dtype_override_wins(self):
        array = np.zeros((3, 4, 5), dtype=np.uint8)
        assert ome._array_nbytes(array, np.uint32) == 3 * 4 * 5 * 4


class _RecordingLazy:
    """Minimal stand-in for a Dask array that records compute() kwargs."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return self._data


class TestComputeBlock:
    """`max_dask_workers` has to reach the scheduler, not dask.config."""

    def test_worker_cap_is_passed_per_compute(self):
        lazy = _RecordingLazy(np.arange(6, dtype=np.uint16).reshape(2, 3))

        block = ome._compute_block(lazy, np.uint32, max_workers=2)

        assert lazy.calls == [{"scheduler": "threads", "num_workers": 2}]
        assert block.dtype == np.uint32
        np.testing.assert_array_equal(block, np.arange(6).reshape(2, 3))

    def test_without_a_cap_the_default_scheduler_is_used(self):
        da = pytest.importorskip("dask.array")
        data = np.arange(4, dtype=np.uint8).reshape(2, 2)
        lazy = da.from_array(data, chunks=(1, 2))

        block = ome._compute_block(lazy, np.uint16, max_workers=None)

        assert isinstance(block, np.ndarray)
        assert block.dtype == np.uint16
        np.testing.assert_array_equal(block, data)

    def test_plain_array_needs_no_scheduler(self):
        data = np.arange(4, dtype=np.uint8).reshape(2, 2)

        block = ome._compute_block(data, np.float32)

        assert block.dtype == np.float32
        np.testing.assert_array_equal(block, data)

    def test_a_matching_dtype_is_not_copied(self):
        """``astype(copy=False)``: a no-op cast must not duplicate a block."""
        data = np.arange(4, dtype=np.uint8).reshape(2, 2)
        assert ome._compute_block(data, np.uint8) is data


class TestIterPlanesBlockwise:
    """Lazy inputs are walked block by block, never plane by plane."""

    def _dask(self, array, chunks):
        da = pytest.importorskip("dask.array")
        return da.from_array(array, chunks=chunks)

    def test_two_dimensional_input_yields_one_plane(self):
        array = np.arange(12, dtype=np.uint16).reshape(3, 4)
        planes = list(
            ome.iter_planes_blockwise(self._dask(array, (3, 4)), np.uint32, 8)
        )
        assert len(planes) == 1
        assert planes[0].dtype == np.uint32
        np.testing.assert_array_equal(planes[0], array)

    def test_whole_array_under_budget_is_computed_once(self):
        array = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        planes = list(
            ome.iter_planes_blockwise(
                self._dask(array, (1, 3, 4, 5)), np.uint16, 10**9
            )
        )
        assert len(planes) == 6
        np.testing.assert_array_equal(
            np.stack(planes), array.reshape(-1, 4, 5)
        )
        # Copies, so tifffile may keep them queued after the block is freed.
        assert all(p.flags["C_CONTIGUOUS"] for p in planes)

    def test_oversized_array_descends_the_leading_axes(self):
        array = np.arange(4 * 3 * 4 * 5, dtype=np.uint16).reshape(4, 3, 4, 5)
        planes = list(
            ome.iter_planes_blockwise(
                self._dask(array, (1, 1, 4, 5)), np.uint16, 64
            )
        )
        assert len(planes) == 12
        np.testing.assert_array_equal(
            np.stack(planes), array.reshape(-1, 4, 5)
        )


class _ReadRecorder:
    """
    Array-like proxy that records the index of every read.

    Deliberately has no ``compute``: ``iter_planes_for_write`` must take the
    store-backed branch, and the recorded indices then show whether it really
    reads one YX plane at a time or slurps the whole stack.
    """

    def __init__(self, array):
        self._array = array
        self.shape = tuple(array.shape)
        self.dtype = array.dtype
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return self._array[index]


class TestIterPlanesForWrite:
    """Dask goes blockwise; a store-backed array stays plane by plane."""

    def test_lazy_input_is_delegated_to_the_block_iterator(self, monkeypatch):
        da = pytest.importorskip("dask.array")
        seen = {}

        def spy(array, dtype, budget_bytes, max_workers=None):
            seen["budget"] = budget_bytes
            seen["dtype"] = dtype
            seen["array"] = array
            yield np.zeros((2, 2), dtype=dtype)

        monkeypatch.setattr(ome, "iter_planes_blockwise", spy)
        array = da.zeros((2, 2, 2), chunks=(1, 2, 2))

        planes = list(ome.iter_planes_for_write(array, np.uint8, 4096))

        assert seen["budget"] == 4096
        assert seen["dtype"] is np.uint8
        assert seen["array"] is array
        assert len(planes) == 1

    def test_two_dimensional_store_array_yields_itself(self):
        array = np.arange(6, dtype=np.uint16).reshape(2, 3)
        planes = list(ome.iter_planes_for_write(array, np.uint32))
        assert len(planes) == 1
        assert planes[0].dtype == np.uint32
        np.testing.assert_array_equal(planes[0], array)

    def test_store_array_is_read_one_plane_at_a_time(self, tmp_path):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        store = zarr.open_array(
            str(tmp_path / "a.zarr"),
            mode="w",
            shape=data.shape,
            chunks=(1, 1, 4, 5),
            dtype="uint16",
        )
        store[:] = data
        recorded = _ReadRecorder(store)

        planes = list(ome.iter_planes_for_write(recorded, np.uint32))

        assert len(planes) == 6
        assert planes[0].dtype == np.uint32
        np.testing.assert_array_equal(np.stack(planes), data.reshape(-1, 4, 5))
        # One read per YX plane addressed by its leading index -- never a
        # whole-stack slice, which is the entire point of this branch.
        assert recorded.reads == [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ]


class TestStreamPlanesToTiff:
    """photometric="minisblack" is load-bearing, not decoration."""

    @pytest.mark.parametrize("lead", [3, 4, 5])
    def test_leading_axis_of_three_or_four_is_not_read_as_rgb(
        self, tmp_path, lead
    ):
        data = np.arange(lead * 8 * 6, dtype=np.uint16).reshape(lead, 8, 6)
        out = tmp_path / "stack.ome.tif"

        returned = ome.stream_planes_to_tiff(
            str(out),
            (plane for plane in data),
            data.shape,
            np.uint16,
            metadata={"axes": "ZYX"},
        )

        assert returned == str(out)
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_ome_can_be_left_to_tifffile(self, tmp_path):
        data = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
        out = tmp_path / "plain.tif"

        ome.stream_planes_to_tiff(
            str(out), iter(data), data.shape, np.uint8, ome=None
        )

        with tifffile.TiffFile(out) as handle:
            assert handle.is_ome is False
        np.testing.assert_array_equal(tifffile.imread(out), data)


class TestWriteLabelsOmeTiff:
    """The TIFF branch of write_labels_with_source_metadata."""

    def test_axes_hint_that_does_not_fit_is_replaced(self, tmp_path):
        out = tmp_path / "out.ome.tif"
        ome.write_labels_with_source_metadata(
            np.zeros((3, 8, 6), np.uint32), None, str(out), "tiff", "TCZYX"
        )
        with tifffile.TiffFile(out) as handle:
            pixels = tifffile.xml2dict(handle.ome_metadata)["OME"]["Image"][
                "Pixels"
            ]
        assert pixels["SizeZ"] == 3
        assert pixels["SizeT"] == 1

    def test_source_voxel_size_is_carried_into_the_output(self, tmp_path):
        source = tmp_path / "src.ome.tif"
        tifffile.imwrite(
            source,
            np.zeros((3, 8, 6), np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.25,
                "PhysicalSizeY": 0.25,
                "PhysicalSizeZ": 2.0,
            },
        )
        out = tmp_path / "labels.ome.tif"

        ome.write_labels_with_source_metadata(
            np.zeros((3, 8, 6), np.uint32),
            str(source),
            str(out),
            "tiff",
            "ZYX",
        )

        with tifffile.TiffFile(out) as handle:
            pixels = tifffile.xml2dict(handle.ome_metadata)["OME"]["Image"][
                "Pixels"
            ]
        assert pixels["PhysicalSizeX"] == pytest.approx(0.25)
        assert pixels["PhysicalSizeZ"] == pytest.approx(2.0)
        assert pixels["PhysicalSizeZUnit"] == "um"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestWriteLabelsZarr:
    """
    The OME-Zarr branch, including its metadata-repair block.

    Most sources here carry no ``omero`` block, simply to keep each test
    focused on one thing; ``test_a_real_repair_and_omero_both_survive_
    together`` is the one that exercises both a real transform repair and
    omero=True at once, which used to lose the repair (root.attrs["omero"]
    was set from a stale in-memory cache taken before the repair's own
    raw-JSON rewrite, so it clobbered that rewrite on disk).
    """

    def _source(self, tmp_path, datasets, omero=True):
        source = tmp_path / "source.zarr"
        source.mkdir()
        attrs = {
            "multiscales": [
                {
                    "axes": [
                        {"name": "z"},
                        {"name": "y"},
                        {"name": "x"},
                    ],
                    "datasets": datasets,
                }
            ]
        }
        if omero:
            attrs["omero"] = {"version": "0.3", "channels": []}
        (source / ".zattrs").write_text(json.dumps(attrs))
        return source

    def _written_transforms(self, out):
        attrs = json.loads((out / "zarr.json").read_text())["attributes"]
        multiscales = ome._get_multiscales(attrs)
        return [
            ds.get("coordinateTransformations")
            for ds in multiscales[0]["datasets"]
        ], multiscales[0]

    def test_unusable_source_transforms_are_left_alone(self, tmp_path):
        """A translation, a mis-sized scale, a missing entry: all skipped."""
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "translation", "translation": [0, 0, 0]}
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 2.0]}
                    ],
                },
                {"path": "2"},
            ],
        )
        out = tmp_path / "out.zarr"

        # dim_order deliberately does not match the 3-D array: the fallback
        # has to pick "zyx" or write_image would reject the axes.
        returned = ome.write_labels_with_source_metadata(
            np.zeros((4, 16, 16), np.uint32),
            str(source),
            str(out),
            "zarr",
            "TCZYX",
        )
        assert returned == str(out)

        transforms, multiscale = self._written_transforms(out)
        assert [a["name"] for a in multiscale["axes"]] == ["z", "y", "x"]
        assert len(multiscale["datasets"]) == 3

        # Nothing in the source is usable, so every level must still carry the
        # writer's own pyramid: one 3-valued scale per level, all different.
        # Copying the source anyway would show up as a 2-valued scale (the
        # mis-sized entry), a missing/translation transform, or three
        # identical levels.
        scales = []
        for ctf in transforms:
            assert ctf and ctf[0]["type"] == "scale"
            assert len(ctf[0]["scale"]) == 3
            scales.append(tuple(ctf[0]["scale"]))
        assert len(set(scales)) == 3

        # omero rides along from the source attrs.
        attrs = json.loads((out / "zarr.json").read_text())["attributes"]
        assert attrs["omero"] == {"version": "0.3", "channels": []}

    def test_a_real_repair_and_omero_both_survive_together(self, tmp_path):
        """
        The one combination every other test in this class deliberately
        avoids: a source scale that IS usable (so the repair actually
        rewrites coordinateTransformations) together with omero=True
        (the default).  root.attrs["omero"] used to be set from a stale
        in-memory cache taken right after write_image() -- before the
        repair's own raw-JSON rewrite -- so setting it re-serialised that
        stale snapshot and threw the just-applied physical scale away.
        """
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [2.0, 0.1, 0.1]}
                    ],
                }
            ],
        )
        out = tmp_path / "out.zarr"

        ome.write_labels_with_source_metadata(
            np.zeros((4, 16, 16), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        transforms, _multiscale = self._written_transforms(out)
        assert transforms[0] == [{"type": "scale", "scale": [2.0, 0.1, 0.1]}]

        attrs = json.loads((out / "zarr.json").read_text())["attributes"]
        assert attrs["omero"] == {"version": "0.3", "channels": []}

    def test_existing_output_is_replaced(self, tmp_path):
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ],
        )
        out = tmp_path / "out.zarr"
        out.mkdir()
        (out / "stale.txt").write_text("left over from an earlier run")

        ome.write_labels_with_source_metadata(
            np.zeros((2, 8, 8), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        assert not (out / "stale.txt").exists()
        assert (out / "zarr.json").exists()

    def test_zattrs_style_output_gets_the_source_scale(
        self, tmp_path, monkeypatch
    ):
        """v0.4 writers emit .zattrs; that layout must be repaired too."""
        writer = pytest.importorskip("ome_zarr.writer")
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [3.0, 0.5, 0.5]}
                    ],
                }
            ],
            omero=False,
        )
        out = tmp_path / "out.zarr"

        def fake_write_image(image=None, group=None, axes=None, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            (out / ".zattrs").write_text(
                json.dumps(
                    {
                        "multiscales": [
                            {
                                "axes": [{"name": a} for a in axes],
                                "datasets": [
                                    {
                                        "path": "0",
                                        "coordinateTransformations": [
                                            {
                                                "type": "scale",
                                                "scale": [1.0, 1.0, 1.0],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                )
            )

        monkeypatch.setattr(writer, "write_image", fake_write_image)

        ome.write_labels_with_source_metadata(
            np.zeros((4, 8, 8), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        written = json.loads((out / ".zattrs").read_text())
        transforms = written["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ]
        assert transforms == [{"type": "scale", "scale": [3.0, 0.5, 0.5]}]

    def _stub_zattrs_writer(self, monkeypatch, out, doc):
        """Make write_image emit a v0.4-style .zattrs with `doc` in it."""
        writer = pytest.importorskip("ome_zarr.writer")

        def fake_write_image(image=None, group=None, axes=None, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            (out / ".zattrs").write_text(json.dumps(doc))

        monkeypatch.setattr(writer, "write_image", fake_write_image)

    def test_output_levels_beyond_the_source_are_left_untouched(
        self, tmp_path, monkeypatch
    ):
        """No source transform to copy for that level, so nothing is copied."""
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [3.0, 0.5, 0.5]}
                    ],
                }
            ],
            omero=False,
        )
        out = tmp_path / "out.zarr"
        self._stub_zattrs_writer(
            monkeypatch,
            out,
            {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "z"},
                            {"name": "y"},
                            {"name": "x"},
                        ],
                        "datasets": [
                            {
                                "path": str(level),
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 1.0, 1.0],
                                    }
                                ],
                            }
                            for level in range(2)
                        ],
                    }
                ]
            },
        )

        ome.write_labels_with_source_metadata(
            np.zeros((4, 8, 8), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        datasets = json.loads((out / ".zattrs").read_text())["multiscales"][0][
            "datasets"
        ]
        assert datasets[0]["coordinateTransformations"] == [
            {"type": "scale", "scale": [3.0, 0.5, 0.5]}
        ]
        assert datasets[1]["coordinateTransformations"] == [
            {"type": "scale", "scale": [1.0, 1.0, 1.0]}
        ]

    def test_source_level_without_a_usable_scale_is_left_untouched(
        self, tmp_path, monkeypatch
    ):
        """
        The source level has transforms, but none of them is a scale, so
        there is nothing to align and the writer's own transform must stay.
        """
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "translation", "translation": [0, 0, 0]}
                    ],
                }
            ],
            omero=False,
        )
        out = tmp_path / "out.zarr"
        self._stub_zattrs_writer(
            monkeypatch,
            out,
            {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "z"},
                            {"name": "y"},
                            {"name": "x"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [9.0, 9.0, 9.0],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        )

        ome.write_labels_with_source_metadata(
            np.zeros((4, 8, 8), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        written = json.loads((out / ".zattrs").read_text())
        assert written["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ] == [{"type": "scale", "scale": [9.0, 9.0, 9.0]}]

    def test_unparseable_output_metadata_does_not_fail_the_write(
        self, tmp_path, monkeypatch
    ):
        """Repairing the transforms is best-effort; the data is already out."""
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [3.0, 0.5, 0.5]}
                    ],
                }
            ],
            omero=False,
        )
        out = tmp_path / "out.zarr"
        self._stub_zattrs_writer(
            monkeypatch, out, {"multiscales": ["not-a-mapping"]}
        )

        returned = ome.write_labels_with_source_metadata(
            np.zeros((4, 8, 8), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        assert returned == str(out)
        assert json.loads((out / ".zattrs").read_text()) == {
            "multiscales": ["not-a-mapping"]
        }

    def test_omero_copy_failure_does_not_abort_the_write(
        self, tmp_path, monkeypatch
    ):
        writer = pytest.importorskip("ome_zarr.writer")
        source = self._source(
            tmp_path,
            [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ],
        )
        out = tmp_path / "out.zarr"

        attempts = []

        class _RefusingAttrs(dict):
            def __setitem__(self, key, value):
                attempts.append((key, value))
                raise RuntimeError("store is read-only")

        class _FakeRoot:
            def __init__(self):
                self.attrs = _RefusingAttrs()

        monkeypatch.setattr(zarr, "group", lambda *a, **k: _FakeRoot())
        monkeypatch.setattr(writer, "write_image", lambda *a, **k: None)

        returned = ome.write_labels_with_source_metadata(
            np.zeros((2, 8, 8), np.uint32),
            str(source),
            str(out),
            "zarr",
            "zyx",
        )

        assert returned == str(out)
        # The copy really was attempted with the source payload -- without
        # this the test would pass even if the omero block were deleted.
        assert attempts == [("omero", {"version": "0.3", "channels": []})]


# ---------------------------------------------------------------------------
# file_compression
# ---------------------------------------------------------------------------

_WORKER_ATTRS = (
    "process_file",
    "_tmidas_compression_patched",
    "_tmidas_compression_original",
)

_FILE_COMPRESSION = "napari_tmidas.processing_functions.file_compression"


@pytest.fixture
def pzstd_calls(monkeypatch):
    """Record pzstd invocations instead of shelling out for real."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def pristine_worker():
    """
    A ProcessingWorker with the compression patch backed out.

    Importing ``file_compression`` mutates two process-wide singletons -- the
    registry and ``ProcessingWorker.process_file`` -- so both are snapshotted
    here and put back verbatim, or later tests in the session inherit them.
    """
    from napari_tmidas._file_selector import ProcessingWorker
    from napari_tmidas._registry import BatchProcessingRegistry

    registry = BatchProcessingRegistry._processing_functions
    registry_snapshot = dict(registry)
    worker_snapshot = {
        name: ProcessingWorker.__dict__[name]
        for name in _WORKER_ATTRS
        if name in ProcessingWorker.__dict__
    }

    ProcessingWorker.process_file = worker_snapshot.get(
        "_tmidas_compression_original", worker_snapshot["process_file"]
    )
    for name in _WORKER_ATTRS[1:]:
        if name in ProcessingWorker.__dict__:
            delattr(ProcessingWorker, name)

    try:
        yield ProcessingWorker
    finally:
        for name in _WORKER_ATTRS:
            if name in worker_snapshot:
                setattr(ProcessingWorker, name, worker_snapshot[name])
            elif name in ProcessingWorker.__dict__:
                delattr(ProcessingWorker, name)
        registry.clear()
        registry.update(registry_snapshot)


@pytest.fixture
def compression(pristine_worker, pzstd_calls):
    """``file_compression`` re-executed against a pristine worker."""
    module = importlib.reload(importlib.import_module(_FILE_COMPRESSION))
    yield module
    for attr in ("compress_after_save", "remove_source", "compression_level"):
        if hasattr(module.compress_with_zstandard, attr):
            delattr(module.compress_with_zstandard, attr)


def _flagged_func(**attrs):
    """A processing function carrying the compression hand-off attributes."""

    def func(image):
        return image

    for key, value in attrs.items():
        setattr(func, key, value)
    return func


class _FakeWorker:
    def __init__(self, processing_func):
        self.processing_func = processing_func


class TestCheckPzstdInstalled:
    """The probe must never raise, whatever the system looks like."""

    def test_probes_the_version(self, compression, pzstd_calls):
        assert compression.check_pzstd_installed() is True
        assert pzstd_calls[-1] == ["pzstd", "--version"]

    @pytest.mark.parametrize(
        "error",
        [FileNotFoundError("pzstd"), subprocess.SubprocessError("boom")],
    )
    def test_absent_or_broken_pzstd_reports_false(
        self, compression, monkeypatch, error
    ):
        def raiser(*args, **kwargs):
            raise error

        monkeypatch.setattr(subprocess, "run", raiser)
        assert compression.check_pzstd_installed() is False


class TestCompressFile:
    """The pzstd command line is assembled from the parameters."""

    def test_default_command(self, compression, pzstd_calls):
        ok, path = compression.compress_file("/data/img.tif")

        assert (ok, path) == (True, "/data/img.tif.zst")
        assert pzstd_calls[-1] == [
            "pzstd",
            "--quiet",
            "-3",
            "/data/img.tif",
        ]

    @pytest.mark.parametrize("level", [20, 22])
    def test_ultra_levels_need_the_ultra_flag(
        self, compression, pzstd_calls, level
    ):
        compression.compress_file("/data/img.tif", compression_level=level)
        assert pzstd_calls[-1] == [
            "pzstd",
            "--quiet",
            "--ultra",
            f"-{level}",
            "/data/img.tif",
        ]

    def test_level_nineteen_is_below_the_ultra_boundary(
        self, compression, pzstd_calls
    ):
        compression.compress_file("/data/img.tif", compression_level=19)
        assert pzstd_calls[-1] == [
            "pzstd",
            "--quiet",
            "-19",
            "/data/img.tif",
        ]

    def test_remove_source_adds_rm(self, compression, pzstd_calls):
        compression.compress_file("/data/img.tif", remove_source=True)
        assert "--rm" in pzstd_calls[-1]

    def test_nonzero_exit_reports_failure_but_keeps_the_name(
        self, compression, monkeypatch
    ):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1),
        )
        assert compression.compress_file("/data/img.tif") == (
            False,
            "/data/img.tif.zst",
        )

    def test_missing_binary_reports_no_path(self, compression, monkeypatch):
        def raiser(*args, **kwargs):
            raise FileNotFoundError("pzstd")

        monkeypatch.setattr(subprocess, "run", raiser)
        assert compression.compress_file("/data/img.tif") == (False, None)


class TestCompressWithZstandard:
    """The processing function only flags the file for later compression."""

    def test_flags_are_left_on_the_function(self, compression):
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)

        result = compression.compress_with_zstandard(
            image, remove_source=True, compression_level=19
        )

        assert result is image
        func = compression.compress_with_zstandard
        assert func.compress_after_save is True
        assert func.remove_source is True
        assert func.compression_level == 19

    def test_missing_pzstd_skips_compression(
        self, compression, monkeypatch, capsys
    ):
        def raiser(*args, **kwargs):
            raise FileNotFoundError("pzstd")

        monkeypatch.setattr(subprocess, "run", raiser)
        image = np.zeros((2, 2), dtype=np.uint8)

        result = compression.compress_with_zstandard(image)

        assert result is image
        assert "pzstd is not installed" in capsys.readouterr().out
        assert not hasattr(
            compression.compress_with_zstandard, "compress_after_save"
        )


class TestProcessFileWithCompression:
    """The post-save hook rewrites result paths only when asked to."""

    def _patch_pipeline(self, compression, monkeypatch, result, outcomes):
        monkeypatch.setattr(
            compression,
            "original_process_file",
            lambda self, filepath: result,
        )
        seen = []

        def fake_compress_file(path, remove_source, compression_level):
            seen.append((path, remove_source, compression_level))
            return outcomes.pop(0)

        monkeypatch.setattr(compression, "compress_file", fake_compress_file)
        return seen

    def test_non_dict_result_passes_straight_through(
        self, compression, monkeypatch
    ):
        self._patch_pipeline(compression, monkeypatch, "just-a-string", [])
        worker = _FakeWorker(_flagged_func(compress_after_save=True))

        assert (
            compression.process_file_with_compression(worker, "in.tif")
            == "just-a-string"
        )

    def test_single_file_is_compressed_with_the_declared_options(
        self, compression, monkeypatch
    ):
        seen = self._patch_pipeline(
            compression,
            monkeypatch,
            {"processed_file": "/out/a.tif"},
            [(True, "/out/a.tif.zst")],
        )
        worker = _FakeWorker(
            _flagged_func(
                compress_after_save=True,
                remove_source=True,
                compression_level=9,
            )
        )

        result = compression.process_file_with_compression(worker, "in.tif")

        assert result["processed_file"] == "/out/a.tif.zst"
        assert seen == [("/out/a.tif", True, 9)]

    def test_defaults_are_used_when_the_function_declares_none(
        self, compression, monkeypatch
    ):
        seen = self._patch_pipeline(
            compression,
            monkeypatch,
            {"processed_file": "/out/a.tif"},
            [(True, "/out/a.tif.zst")],
        )
        worker = _FakeWorker(_flagged_func(compress_after_save=True))

        compression.process_file_with_compression(worker, "in.tif")

        assert seen == [("/out/a.tif", False, 3)]

    def test_failed_compression_keeps_the_uncompressed_path(
        self, compression, monkeypatch
    ):
        self._patch_pipeline(
            compression,
            monkeypatch,
            {"processed_file": "/out/a.tif"},
            [(False, None)],
        )
        worker = _FakeWorker(_flagged_func(compress_after_save=True))

        result = compression.process_file_with_compression(worker, "in.tif")

        assert result["processed_file"] == "/out/a.tif"

    @pytest.mark.parametrize(
        "func", [_flagged_func(), _flagged_func(compress_after_save=False)]
    )
    def test_unflagged_function_is_left_alone(
        self, compression, monkeypatch, func
    ):
        seen = self._patch_pipeline(
            compression, monkeypatch, {"processed_file": "/out/a.tif"}, []
        )
        worker = _FakeWorker(func)

        result = compression.process_file_with_compression(worker, "in.tif")

        assert result["processed_file"] == "/out/a.tif"
        assert seen == []

    def test_multi_file_results_compress_each_entry(
        self, compression, monkeypatch
    ):
        seen = self._patch_pipeline(
            compression,
            monkeypatch,
            {"processed_files": ["/out/a.tif", "/out/b.tif"]},
            [(True, "/out/a.tif.zst"), (False, None)],
        )
        worker = _FakeWorker(_flagged_func(compress_after_save=True))

        result = compression.process_file_with_compression(worker, "in.tif")

        # The failed one keeps its original path rather than becoming None.
        assert result["processed_files"] == ["/out/a.tif.zst", "/out/b.tif"]
        assert [call[0] for call in seen] == ["/out/a.tif", "/out/b.tif"]

    def test_multi_file_results_of_an_unflagged_function(
        self, compression, monkeypatch
    ):
        seen = self._patch_pipeline(
            compression,
            monkeypatch,
            {"processed_files": ["/out/a.tif"]},
            [],
        )
        worker = _FakeWorker(_flagged_func())

        result = compression.process_file_with_compression(worker, "in.tif")

        assert result["processed_files"] == ["/out/a.tif"]
        assert seen == []


class TestCompressionMonkeyPatch:
    """Re-importing must not wrap the wrapper -- that recurses forever."""

    def test_patch_is_installed_once(self, pristine_worker, pzstd_calls):
        pristine = pristine_worker.process_file

        module = importlib.reload(importlib.import_module(_FILE_COMPRESSION))

        assert pristine_worker.process_file is (
            module.process_file_with_compression
        )
        assert pristine_worker._tmidas_compression_patched is True
        assert module.original_process_file is pristine

        wrapper = pristine_worker.process_file
        importlib.reload(module)

        assert module.original_process_file is pristine
        assert pristine_worker.process_file is wrapper

    def test_registers_the_processing_function(
        self, pristine_worker, pzstd_calls
    ):
        from napari_tmidas._registry import BatchProcessingRegistry

        importlib.reload(importlib.import_module(_FILE_COMPRESSION))

        info = BatchProcessingRegistry.get_function_info(
            "Compress with Zstandard"
        )
        assert info["suffix"] == "_compressed"
        assert info["parameters"]["compression_level"]["max"] == 22

    def test_no_patch_without_pzstd(self, pristine_worker, monkeypatch):
        def raiser(*args, **kwargs):
            raise FileNotFoundError("pzstd")

        monkeypatch.setattr(subprocess, "run", raiser)
        pristine = pristine_worker.process_file

        importlib.reload(importlib.import_module(_FILE_COMPRESSION))

        assert pristine_worker.process_file is pristine
        assert not getattr(
            pristine_worker, "_tmidas_compression_patched", False
        )


# ---------------------------------------------------------------------------
# intensity_label_filter
# ---------------------------------------------------------------------------


class TestLabelValueHelpers:
    """Small guards in the block-wise label scanners."""

    def test_collect_defaults_to_one_block(self):
        image = np.array([[0, 3], [3, 7]], dtype=np.uint16)
        np.testing.assert_array_equal(
            ilf._collect_label_values(image), [0, 3, 7]
        )

    def test_semantic_conversion_tolerates_none(self):
        assert ilf._convert_semantic_to_instance(None) is None

    def test_already_instance_labels_are_returned_untouched(self):
        """Two distinct ids: nothing to split, so the input comes back."""
        image = np.array([[0, 1], [2, 2]], dtype=np.uint16)
        assert ilf._convert_semantic_to_instance(image) is image


class TestMeanIntensityGuards:
    """Non-integer and empty label images must not crash the accumulator."""

    def test_float_labels_are_cast_before_binning(self):
        labels = np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        intensity = np.array([[5.0, 10.0], [20.0, 30.0]], dtype=np.float32)

        means = ilf._calculate_label_mean_intensities(labels, intensity)

        assert means == {1: pytest.approx(10.0), 2: pytest.approx(25.0)}

    def test_empty_image_has_no_labels(self):
        empty = np.zeros((0, 4), dtype=np.uint16)
        assert ilf._calculate_label_mean_intensities(empty, empty.copy()) == {}


class TestKMedoidsDegenerateBoundaries:
    """A boundary that overflows to inf empties a cluster."""

    def test_empty_cluster_keeps_its_previous_medoid(self):
        values = np.array([1e308, 1.7e308, 1.5e308, 1.6e308])

        medoids = ilf._kmedoids_1d(values, 2)

        # Midpoints overflow to +inf, so every point lands in cluster 0 and
        # the upper medoid has no members; it is carried over unchanged
        # instead of raising.  Lower medoid = the weighted median of all four
        # (1.5e308, not the seeded minimum); upper = the carried-over seed.
        np.testing.assert_array_equal(medoids, [1.5e308, 1.7e308])


class TestIntensityFilename:
    """Legacy filename derivation used by the in-memory path."""

    def test_known_suffix_is_stripped(self):
        assert (
            ilf._get_intensity_filename("cells_convpaint_labels_filtered.tif")
            == "cells.tif"
        )

    def test_custom_suffix(self):
        assert (
            ilf._get_intensity_filename("cells_masks.tif", "_masks.tif")
            == "cells.tif"
        )

    def test_unmatched_suffix_returns_the_input(self):
        assert ilf._get_intensity_filename("cells.tif") == "cells.tif"


class _FakeZarrGroup:
    """
    Group-like node with a member order the test controls.

    Only what ``_first_array`` touches: no ``shape`` (so it is treated as a
    group), ``in``/``[]`` for the 's0'/'0' lookup and ``arrays()``.
    """

    def __init__(self, members):
        self._members = list(members)

    def __contains__(self, key):
        return any(name == key for name, _ in self._members)

    def __getitem__(self, key):
        return dict(self._members)[key]

    def arrays(self):
        return list(self._members)


class TestPlaneReaderInternals:
    """Addressing schemes the reader has to pick between."""

    def test_memmap_with_a_flat_shape_is_reshaped(self, tmp_path, monkeypatch):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, data)

        flat = data.reshape(-1, 4, 5)
        monkeypatch.setattr(tifffile, "memmap", lambda *a, **k: flat)

        with ilf._PlaneReader(path) as reader:
            assert reader.shape == (2, 3, 4, 5)
            np.testing.assert_array_equal(reader.plane((1, 2)), data[1, 2])

    def test_first_array_accepts_a_bare_array(self):
        array = np.zeros((3, 4))
        assert ilf._PlaneReader._first_array(array) is array

    def test_first_array_falls_back_to_the_largest_member(self, tmp_path):
        group = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
        group.create_array("small", shape=(2, 2), dtype="uint8")
        group.create_array("big", shape=(8, 8), dtype="uint8")

        assert ilf._PlaneReader._first_array(group).shape == (8, 8)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_largest_wins_whatever_order_members_come_in(self, reverse):
        """
        ``zarr.Group.arrays()`` yields in store order, which is not sorted and
        not stable, so a real group cannot pin *which* member is chosen --
        with two members it agrees with "first" half the time by luck.  These
        drive both orders explicitly.
        """
        members = [
            ("a", np.zeros((2, 2), np.uint8)),
            ("b", np.zeros((8, 8), np.uint8)),
        ]
        node = _FakeZarrGroup(members[::-1] if reverse else members)

        assert ilf._PlaneReader._first_array(node).shape == (8, 8)

    def test_level_zero_is_preferred_over_the_largest_member(self):
        """'0' is the full-resolution level even when a member looks bigger."""
        node = _FakeZarrGroup(
            [
                ("mask", np.zeros((16, 16), np.uint8)),
                ("0", np.zeros((4, 4), np.uint8)),
            ]
        )

        assert ilf._PlaneReader._first_array(node).shape == (4, 4)

    def test_first_array_on_an_empty_group_raises(self, tmp_path):
        group = zarr.open_group(str(tmp_path / "empty.zarr"), mode="w")
        with pytest.raises(ValueError, match="No arrays found"):
            ilf._PlaneReader._first_array(group)


class TestFilterByThresholdEdges:
    """Fallbacks for label images the LUT path cannot handle."""

    def test_no_intensities_copies_into_the_requested_dtype(self):
        labels = np.array([[0, 1], [2, 2]], dtype=np.int64)

        out = ilf._filter_labels_by_threshold(
            labels, {}, 5.0, out_dtype=np.dtype(np.uint32)
        )

        assert out.dtype == np.uint32
        assert out is not labels
        np.testing.assert_array_equal(out, labels)

    def test_float_labels_use_the_per_label_pass(self):
        labels = np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float32)

        out = ilf._filter_labels_by_threshold(labels, {1: 1.0, 2: 100.0}, 50.0)

        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, [[0.0, 0.0], [2.0, 2.0]])

    def test_signed_labels_with_negatives_use_the_per_label_pass(self):
        labels = np.array([[-1, 1], [2, 2]], dtype=np.int16)

        out = ilf._filter_labels_by_threshold(labels, {1: 1.0, 2: 100.0}, 50.0)

        # The negative sentinel is untouched; only label 1 is dropped.
        np.testing.assert_array_equal(out, [[-1, 0], [2, 2]])


class TestSmallestLabelDtype:
    """uint32 is the floor, and an unrepresentable id changes nothing."""

    def test_label_beyond_every_candidate_keeps_the_current_dtype(self):
        current = np.dtype(np.uint8)
        assert ilf._smallest_label_dtype(2**64, current) == current


class TestSaveClusterStats:
    """The per-folder CSV is created once and appended to afterwards."""

    def test_creates_then_appends(self, tmp_path):
        pd = pytest.importorskip("pandas")
        label_path = tmp_path / "movie_labels.tif"

        for run in range(2):
            ilf._save_cluster_stats(
                label_path,
                label_path.name,
                n_clusters=2,
                total_labels=10 + run,
                medoids=np.array([5.0, 50.0]),
                threshold=27.5,
                n_removed=4,
            )

        stats = tmp_path / "intensity_filter_stats" / "clustering_stats.csv"
        frame = pd.read_csv(stats)
        assert len(frame) == 2
        assert list(frame["kept_labels"]) == [6, 7]
        assert frame["medoid_1"].iloc[0] == pytest.approx(50.0)

    def test_without_pandas_nothing_is_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ilf, "_HAS_PANDAS", False)

        ilf._save_cluster_stats(
            tmp_path / "movie_labels.tif",
            "movie_labels.tif",
            2,
            3,
            [1.0, 2.0],
            1.5,
            1,
        )

        assert not (tmp_path / "intensity_filter_stats").exists()


def _write_pair(tmp_path, labels, intensity, stem="movie"):
    """Write a label stack and its paired intensity image as plain TIFFs."""
    tifffile.imwrite(tmp_path / f"{stem}.tif", intensity)
    label_path = tmp_path / f"{stem}_labels.tif"
    tifffile.imwrite(label_path, labels)
    return label_path


class TestStreamingEdges:
    """Pass-1 accumulator growth, progress reporting and failure modes."""

    def test_negative_labels_are_rejected(self, tmp_path):
        labels = np.zeros((2, 4, 4), dtype=np.int16)
        labels[0, 0, 0] = -1
        label_path = _write_pair(
            tmp_path, labels, np.ones((2, 4, 4), dtype=np.uint8)
        )

        with pytest.raises(ValueError, match="negative values in plane"):
            ilf.filter_labels_by_intensity(
                image=None,
                save_stats=False,
                _source_filepath=str(label_path),
                _output_folder=str(tmp_path / "out"),
                _output_suffix="_filtered",
            )

    def test_an_all_background_stack_fails_loudly(self, tmp_path):
        label_path = _write_pair(
            tmp_path,
            np.zeros((2, 4, 4), dtype=np.uint16),
            np.ones((2, 4, 4), dtype=np.uint8),
        )

        with pytest.raises(ValueError, match="No labels found"):
            ilf.filter_labels_by_intensity(
                image=None,
                save_stats=False,
                _source_filepath=str(label_path),
                _output_folder=str(tmp_path / "out"),
                _output_suffix="_filtered",
            )

    def test_accumulators_grow_when_a_later_plane_has_more_labels(
        self, tmp_path
    ):
        """
        Every plane that raises the maximum label reallocates the count and
        sum accumulators, and the tallies so far have to be carried over.

        Label 1 is measured across planes 0 and 1 and straddles two such
        reallocations.  Only its *track* mean, 150, keeps it above the
        threshold: had plane 0 been dropped its mean would be 100, which
        lands below, and dropping both planes leaves a single intensity that
        cannot be clustered at all.
        """
        labels = np.zeros((3, 4, 4), dtype=np.uint16)
        labels[0, 0, 0] = 1
        labels[1, 0, 0] = 1
        labels[1, 1, 1] = 2  # max 1 -> 2: first reallocation
        labels[2, 2, 2] = 5  # max 2 -> 5: second reallocation
        intensity = np.zeros((3, 4, 4), dtype=np.uint8)
        intensity[0, 0, 0] = 200
        intensity[1, 0, 0] = 100
        intensity[1, 1, 1] = 5
        intensity[2, 2, 2] = 210
        label_path = _write_pair(tmp_path, labels, intensity)

        out = ilf.filter_labels_by_intensity(
            image=None,
            n_clusters=2,
            save_stats=False,
            _source_filepath=str(label_path),
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_filtered",
        )

        written = tifffile.imread(out)
        assert written.shape == labels.shape
        assert written.dtype == labels.dtype
        # Label 2 (mean 5) is the only one below the threshold.
        assert not np.any(written == 2)
        assert written[0, 0, 0] == 1
        assert written[1, 0, 0] == 1
        assert written[2, 2, 2] == 5

    def test_progress_is_reported_every_two_hundred_planes(
        self, tmp_path, capsys
    ):
        n_planes = 201
        labels = np.zeros((n_planes, 4, 4), dtype=np.uint16)
        labels[:, 0, 0] = 1
        labels[:, 2, 2] = 2
        intensity = np.zeros((n_planes, 4, 4), dtype=np.uint8)
        intensity[:, 0, 0] = 10
        intensity[:, 2, 2] = 200
        label_path = _write_pair(tmp_path, labels, intensity)

        out = ilf.filter_labels_by_intensity(
            image=None,
            n_clusters=2,
            save_stats=False,
            _source_filepath=str(label_path),
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_filtered",
        )

        printed = capsys.readouterr().out
        assert f"pass 1: 200/{n_planes} planes" in printed
        assert f"pass 2: 200/{n_planes} planes" in printed
        written = tifffile.imread(out)
        assert written.shape == labels.shape
        assert not np.any(written == 1)
        assert np.all(written[:, 2, 2] == 2)

    def test_default_output_lands_next_to_the_labels_with_stats(
        self, tmp_path
    ):
        pd = pytest.importorskip("pandas")
        labels = np.zeros((2, 8, 8), dtype=np.uint16)
        labels[:, 0:2, 0:2] = 1
        labels[:, 4:6, 0:2] = 2
        intensity = np.zeros((2, 8, 8), dtype=np.uint8)
        intensity[:, 0:2, 0:2] = 10
        intensity[:, 4:6, 0:2] = 200
        label_path = _write_pair(tmp_path, labels, intensity)

        out = ilf.filter_labels_by_intensity(
            image=None,
            n_clusters=2,
            save_stats=True,
            _source_filepath=str(label_path),
        )

        assert out == str(tmp_path / "movie_labels_intensity_filtered.tif")
        assert (tmp_path / "movie_labels_intensity_filtered.tif").exists()
        stats = pd.read_csv(
            tmp_path / "intensity_filter_stats" / "clustering_stats.csv"
        )
        assert stats["removed_labels"].iloc[0] == 1
        assert stats["kept_labels"].iloc[0] == 1


def _call_with_filepath_local(path, **kwargs):
    """Call the filter the way the worker does: filepath in a caller frame."""
    filepath = str(path)  # noqa: F841 - read back off the call stack
    return ilf.filter_labels_by_intensity(**kwargs)


class TestInMemoryPath:
    """Guards, fallbacks and the two reporting branches."""

    def _labelled(self, tmp_path, means, shape=(12, 12), dtype=np.uint16):
        """One 2x2 block per requested mean intensity, labelled 1..n."""
        labels = np.zeros(shape, dtype=dtype)
        intensity = np.zeros(shape, dtype=np.uint16)
        for i, mean in enumerate(means):
            row = 2 * (i // 3)
            col = 3 * (i % 3)
            labels[row : row + 2, col : col + 2] = i + 1
            intensity[row : row + 2, col : col + 2] = mean
        label_path = tmp_path / "cells_convpaint_labels_filtered.tif"
        tifffile.imwrite(tmp_path / "cells.tif", intensity)
        return labels, label_path

    def test_filepath_is_recovered_from_the_call_stack(self, tmp_path):
        labels, label_path = self._labelled(tmp_path, [10, 12, 200, 210])

        result = _call_with_filepath_local(
            label_path,
            image=labels.copy(),
            n_clusters=2,
            save_stats=False,
            dim_order="YX",
        )

        assert not np.any(result == 1)
        np.testing.assert_array_equal(result == 3, labels == 3)

    def test_no_filepath_anywhere_is_an_error(self, tmp_path, monkeypatch):
        # Swap the module's own reference rather than stdlib ``inspect``:
        # patching the shared module would blind pytest's own machinery too.
        monkeypatch.setattr(ilf, "inspect", SimpleNamespace(stack=list))

        with pytest.raises(ValueError, match="call stack"):
            ilf.filter_labels_by_intensity(
                image=np.zeros((4, 4), np.uint16), save_stats=False
            )

    @pytest.mark.parametrize("n_clusters", [0, 1, 4])
    def test_only_two_or_three_clusters_are_accepted(
        self, tmp_path, n_clusters
    ):
        with pytest.raises(ValueError, match="n_clusters must be 2 or 3"):
            ilf.filter_labels_by_intensity(
                image=np.zeros((4, 4), np.uint16),
                n_clusters=n_clusters,
                save_stats=False,
                _source_filepath=str(tmp_path / "x_labels.tif"),
            )

    def test_semantic_labels_are_split_before_measuring(self, tmp_path):
        """A single label value becomes one id per connected component."""
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[0:2, 0:2] = 1
        labels[5:7, 5:7] = 1
        intensity = np.zeros((8, 8), dtype=np.uint16)
        intensity[0:2, 0:2] = 10
        intensity[5:7, 5:7] = 200
        tifffile.imwrite(tmp_path / "cells.tif", intensity)
        label_path = tmp_path / "cells_convpaint_labels_filtered.tif"

        result = ilf.filter_labels_by_intensity(
            image=labels,
            n_clusters=2,
            save_stats=False,
            dim_order="YX",
            _source_filepath=str(label_path),
        )

        assert result[0, 0] == 0
        assert result[5, 5] != 0

    def test_empty_label_image_returns_zeros(self, tmp_path):
        image = np.zeros((6, 6), dtype=np.uint16)

        result = ilf.filter_labels_by_intensity(
            image=image,
            save_stats=False,
            dim_order="YX",
            _source_filepath=str(
                tmp_path / "cells_convpaint_labels_filtered.tif"
            ),
        )

        assert result.shape == image.shape
        assert not result.any()

    def test_missing_intensity_image_returns_the_input(self, tmp_path, capsys):
        image = np.zeros((6, 6), dtype=np.uint16)
        image[0:2, 0:2] = 1
        image[4:6, 4:6] = 2

        result = ilf.filter_labels_by_intensity(
            image=image,
            save_stats=False,
            dim_order="YX",
            _source_filepath=str(
                tmp_path / "cells_convpaint_labels_filtered.tif"
            ),
        )

        assert result is image
        assert "No corresponding intensity image" in capsys.readouterr().out

    def test_compressed_intensity_falls_back_from_memmap_to_imread(
        self, tmp_path, capsys
    ):
        labels = np.zeros((8, 8), dtype=np.uint16)
        labels[0:2, 0:2] = 1
        labels[4:6, 4:6] = 2
        intensity = np.zeros((8, 8), dtype=np.uint16)
        intensity[0:2, 0:2] = 10
        intensity[4:6, 4:6] = 200
        tifffile.imwrite(tmp_path / "cells.tif", intensity, compression="zlib")

        result = ilf.filter_labels_by_intensity(
            image=labels,
            save_stats=False,
            dim_order="YX",
            _source_filepath=str(
                tmp_path / "cells_convpaint_labels_filtered.tif"
            ),
        )

        assert "not memory-mappable" in capsys.readouterr().out
        assert not np.any(result == 1)
        np.testing.assert_array_equal(result == 2, labels == 2)

    def test_unreadable_intensity_image_returns_the_input(
        self, tmp_path, monkeypatch, capsys
    ):
        labels = np.zeros((6, 6), dtype=np.uint16)
        labels[0:2, 0:2] = 1
        labels[4:6, 4:6] = 2
        tifffile.imwrite(tmp_path / "cells.tif", np.zeros((6, 6), np.uint16))

        def raiser(*args, **kwargs):
            raise OSError("disk went away")

        monkeypatch.setattr(tifffile, "memmap", raiser)
        monkeypatch.setattr(tifffile, "imread", raiser)

        result = ilf.filter_labels_by_intensity(
            image=labels,
            save_stats=False,
            dim_order="YX",
            _source_filepath=str(
                tmp_path / "cells_convpaint_labels_filtered.tif"
            ),
        )

        assert result is labels
        assert "Could not read intensity image" in capsys.readouterr().out

    def test_shape_mismatch_is_rejected(self, tmp_path):
        labels = np.zeros((6, 6), dtype=np.uint16)
        labels[0:2, 0:2] = 1
        labels[4:6, 4:6] = 2
        tifffile.imwrite(tmp_path / "cells.tif", np.zeros((4, 4), np.uint16))

        with pytest.raises(ValueError, match="same shape"):
            ilf.filter_labels_by_intensity(
                image=labels,
                save_stats=False,
                dim_order="YX",
                _source_filepath=str(
                    tmp_path / "cells_convpaint_labels_filtered.tif"
                ),
            )

    def test_non_integer_labels_measure_as_background(self, tmp_path, capsys):
        """
        Fractional float ids truncate to 0 in the bincount accumulator, so
        the "no measurable labels" guard fires.  That guard is what is pinned
        here (warning + zeroed output of the input shape); the truncation
        itself is a documented limitation of float label input, not something
        this test blesses as the right answer.
        """
        labels = np.zeros((6, 6), dtype=np.float32)
        labels[0:2, 0:2] = 0.3
        labels[4:6, 4:6] = 0.6
        tifffile.imwrite(tmp_path / "cells.tif", np.ones((6, 6), np.uint16))

        result = ilf.filter_labels_by_intensity(
            image=labels,
            save_stats=False,
            dim_order="YX",
            _source_filepath=str(
                tmp_path / "cells_convpaint_labels_filtered.tif"
            ),
        )

        printed = capsys.readouterr().out
        assert "No labels found in cells_convpaint_labels_filtered.tif" in (
            printed
        )
        assert result.shape == labels.shape
        assert result.dtype == labels.dtype
        assert not result.any()

    def test_three_cluster_reporting_and_stats(self, tmp_path, capsys):
        pd = pytest.importorskip("pandas")
        labels, label_path = self._labelled(
            tmp_path, [10, 12, 100, 102, 500, 502]
        )

        for _ in range(2):  # second run appends to the existing CSV
            result = ilf.filter_labels_by_intensity(
                image=labels.copy(),
                n_clusters=3,
                save_stats=True,
                dim_order="YX",
                _source_filepath=str(label_path),
            )

        printed = capsys.readouterr().out
        assert "Medium intensity cluster" in printed
        # Only the lowest cluster is dropped.
        assert not np.any(result == 1)
        assert not np.any(result == 2)
        for kept in (3, 4, 5, 6):
            np.testing.assert_array_equal(result == kept, labels == kept)

        stats = pd.read_csv(
            tmp_path / "intensity_filter_stats" / "clustering_stats.csv"
        )
        assert len(stats) == 2
        assert stats["n_clusters"].iloc[0] == 3
        assert stats["low_cluster_count"].iloc[0] == 2
        assert stats["medium_cluster_count"].iloc[0] == 2
        assert stats["high_cluster_count"].iloc[0] == 2

    def test_three_clusters_requested_but_only_two_exist(
        self, tmp_path, capsys
    ):
        pd = pytest.importorskip("pandas")
        labels, label_path = self._labelled(tmp_path, [10, 10, 500, 500])

        for _ in range(2):  # second run appends to the existing CSV
            result = ilf.filter_labels_by_intensity(
                image=labels.copy(),
                n_clusters=3,
                save_stats=True,
                dim_order="YX",
                _source_filepath=str(label_path),
            )

        printed = capsys.readouterr().out
        assert "only separate into 2" in printed.replace("\n", " ")
        assert not np.any(result == 1)
        np.testing.assert_array_equal(result == 3, labels == 3)

        stats = pd.read_csv(
            tmp_path / "intensity_filter_stats" / "clustering_stats.csv"
        )
        assert len(stats) == 2
        # The reported cluster count is the one actually found, not asked for.
        assert stats["n_clusters"].iloc[0] == 2

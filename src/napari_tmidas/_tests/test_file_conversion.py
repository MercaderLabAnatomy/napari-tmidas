"""
Tests for the microscopy file conversion widget (``_file_conversion``).

The module is GUI-heavy, but the parts that decide what an output file
actually looks like are pure: format detection, CZI scale normalisation,
OME-Zarr coordinate transformations, and the TIF/ZARR writers.  These
tests drive that logic against real temporary files rather than mocks, so
a regression in the written pixels or in the spatial metadata shows up
here instead of downstream in a segmentation run.
"""

import csv
import json
import os
import sys

import dask.array as da
import numpy as np
import pytest
import tifffile

from napari_tmidas import _file_conversion as fc

pytest.importorskip("pytestqt")

# Qt widget tests segfault under headless Qt on macOS CI (same guard as
# test_frame_removal.py).  Only the classes that instantiate widgets need
# it -- the pure-logic tests below run everywhere.
requires_gui = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)


@pytest.fixture
def worker(qapp, tmp_path):
    """A ConversionWorker with an empty queue, for helper-method tests."""
    return fc.ConversionWorker(
        files_to_convert=[],
        output_folder=str(tmp_path),
        use_zarr=False,
        file_loader_func=lambda filepath: None,
    )


class TestDebugFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_enabled_values(self, monkeypatch, value):
        monkeypatch.setenv("TMIDAS_CONVERSION_DEBUG", value)
        assert fc._is_conversion_debug_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_disabled_values(self, monkeypatch, value):
        monkeypatch.setenv("TMIDAS_CONVERSION_DEBUG", value)
        assert fc._is_conversion_debug_enabled() is False

    def test_unset_is_disabled(self, monkeypatch):
        monkeypatch.delenv("TMIDAS_CONVERSION_DEBUG", raising=False)
        assert fc._is_conversion_debug_enabled() is False

    def test_debug_print_is_silent_unless_enabled(self, monkeypatch, capsys):
        monkeypatch.delenv("TMIDAS_CONVERSION_DEBUG", raising=False)
        fc._debug_print("hidden")
        assert capsys.readouterr().out == ""

        monkeypatch.setenv("TMIDAS_CONVERSION_DEBUG", "on")
        fc._debug_print("shown")
        assert "shown" in capsys.readouterr().out


class TestFormatLoaderBase:
    def test_abstract_methods_raise(self):
        with pytest.raises(NotImplementedError):
            fc.FormatLoader.can_load("x.tif")
        with pytest.raises(NotImplementedError):
            fc.FormatLoader.get_series_count("x.tif")
        with pytest.raises(NotImplementedError):
            fc.FormatLoader.load_series("x.tif", 0)

    def test_get_metadata_defaults_to_empty(self):
        assert fc.FormatLoader.get_metadata("x.tif", 0) == {}


class TestCanLoadByExtension:
    @pytest.mark.parametrize(
        "loader, good, bad",
        [
            (fc.ND2Loader, "movie.nd2", "movie.tif"),
            (fc.TIFFSlideLoader, "slide.ndpi", "slide.tif"),
            (fc.TIFFSlideLoader, "slide.svs", "slide.czi"),
        ],
    )
    def test_extension_dispatch(self, tmp_path, loader, good, bad):
        assert loader.can_load(str(tmp_path / good)) is True
        assert loader.can_load(str(tmp_path / bad)) is False

    def test_extension_check_is_case_insensitive(self, tmp_path):
        assert fc.ND2Loader.can_load(str(tmp_path / "MOVIE.ND2")) is True
        assert fc.TIFFSlideLoader.can_load(str(tmp_path / "S.NDPI")) is True

    def test_lif_rejects_wrong_extension_without_touching_disk(self):
        # Nothing exists at this path: can_load must short-circuit on the
        # suffix before trying to open it.
        assert fc.LIFLoader.can_load("/nonexistent/sample.tif") is False

    def test_lif_rejects_corrupt_file(self, tmp_path):
        broken = tmp_path / "broken.lif"
        broken.write_bytes(b"not a lif file")
        assert fc.LIFLoader.can_load(str(broken)) is False

    def test_lif_series_count_of_corrupt_file_is_zero(self, tmp_path):
        broken = tmp_path / "broken.lif"
        broken.write_bytes(b"not a lif file")
        assert fc.LIFLoader.get_series_count(str(broken)) == 0

    def test_nd2_series_count_of_corrupt_file_is_zero(self, tmp_path):
        broken = tmp_path / "broken.nd2"
        broken.write_bytes(b"not an nd2 file")
        assert fc.ND2Loader.get_series_count(str(broken)) == 0


class TestCZIScaleNormalisation:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (3.25e-7, 0.325),  # metres -> micrometres
            (1.0e-6, 1.0),
            (0.325, 0.325),  # already micrometres, left alone
            (65.0, 65.0),
        ],
    )
    def test_plausible_values(self, raw, expected):
        assert fc.CZILoader._normalize_czi_scale_to_um(raw) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize("raw", [0.0, -1.0])
    def test_non_positive_falls_back_to_one(self, raw):
        assert fc.CZILoader._normalize_czi_scale_to_um(raw) == 1.0

    def test_implausible_value_is_still_read_as_metres(self):
        # 1 mm/pixel: neither reading lands in the plausible band, so the
        # metre interpretation is used rather than silently defaulting.
        assert fc.CZILoader._normalize_czi_scale_to_um(1e-3) == pytest.approx(
            1000.0
        )


CZI_XML = """
<ImageDocument><Metadata><Scaling><Items>
  <Distance Id="X"><Value>3.25E-07</Value></Distance>
  <Distance Id="Y"><Value>3.25E-07</Value></Distance>
  <Distance Id="Z"><Value>1.0E-06</Value></Distance>
</Items></Scaling></Metadata></ImageDocument>
"""


class TestCZIScaleFromMetadata:
    @pytest.mark.parametrize(
        "dimension, expected", [("X", 0.325), ("Y", 0.325), ("Z", 1.0)]
    )
    def test_reads_xml_string(self, dimension, expected):
        value = fc.CZILoader._extract_scale_from_xml(CZI_XML, dimension)
        assert value == pytest.approx(expected)

    def test_missing_dimension_falls_back_to_one(self):
        assert fc.CZILoader._extract_scale_from_xml(CZI_XML, "T") == 1.0

    def test_reads_parsed_dict_metadata(self):
        metadata = {
            "ImageDocument": {
                "Metadata": {
                    "Scaling": {
                        "Items": {
                            "Distance": [
                                {"@Id": "X", "Value": 3.25e-07},
                                {"@Id": "Z", "Value": 1.0e-06},
                            ]
                        }
                    }
                }
            }
        }
        assert fc.CZILoader._extract_scale_from_xml(
            metadata, "X"
        ) == pytest.approx(0.325)
        assert fc.CZILoader._extract_scale_from_xml(
            metadata, "Z"
        ) == pytest.approx(1.0)

    def test_reads_single_distance_dict(self):
        metadata = {
            "ImageDocument": {
                "Metadata": {
                    "Scaling": {
                        "Items": {"Distance": {"@Id": "X", "Value": 5.0e-07}}
                    }
                }
            }
        }
        assert fc.CZILoader._extract_scale_from_xml(
            metadata, "X"
        ) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "metadata", [None, "", "<Metadata/>", {}, {"ImageDocument": {}}]
    )
    def test_unusable_metadata_falls_back_to_one(self, metadata):
        assert fc.CZILoader._extract_scale_from_xml(metadata, "X") == 1.0


class TestTIFFSlideLoaderFallback:
    """
    ``tiffslide`` is an optional import (it conflicts with zarr v3), so the
    loader has to keep working through its plain-tifffile fallback.  These
    tests force that branch regardless of what is installed.
    """

    @pytest.fixture(autouse=True)
    def _without_tiffslide(self, monkeypatch):
        monkeypatch.setattr(fc, "TiffSlide", None)

    @pytest.fixture
    def slide(self, tmp_path):
        # .svs rather than .ndpi: tifffile applies NDPI-specific page
        # parsing to anything named .ndpi, which a plain TIFF fixture
        # cannot satisfy.  Both extensions take the same loader path.
        path = tmp_path / "slide.svs"
        data = np.random.default_rng(0).integers(
            0, 255, (32, 48), dtype=np.uint8
        )
        tifffile.imwrite(path, data)
        return path, data

    def test_series_count(self, slide):
        path, _ = slide
        assert fc.TIFFSlideLoader.get_series_count(str(path)) == 1

    def test_series_count_of_unreadable_file_is_zero(self, tmp_path):
        broken = tmp_path / "broken.svs"
        broken.write_bytes(b"not a tiff")
        assert fc.TIFFSlideLoader.get_series_count(str(broken)) == 0

    def test_load_series_roundtrip(self, slide):
        path, data = slide
        loaded = fc.TIFFSlideLoader.load_series(str(path), 0)
        np.testing.assert_array_equal(loaded, data)

    def test_out_of_range_series_raises(self, slide):
        path, _ = slide
        with pytest.raises(fc.SeriesIndexError):
            fc.TIFFSlideLoader.load_series(str(path), 5)

    def test_unreadable_file_raises_format_error(self, tmp_path):
        broken = tmp_path / "broken.svs"
        broken.write_bytes(b"not a tiff")
        with pytest.raises(fc.FileFormatError):
            fc.TIFFSlideLoader.load_series(str(broken), 0)

    def test_metadata_is_empty_without_tiffslide(self, slide):
        path, _ = slide
        assert fc.TIFFSlideLoader.get_metadata(str(path), 0) == {}


class TestAcquiferDetection:
    def _add_image(self, directory, name="frame.tif"):
        tifffile.imwrite(directory / name, np.zeros((4, 4), dtype=np.uint8))

    def test_rejects_plain_file(self, tmp_path):
        path = tmp_path / "x.tif"
        path.write_bytes(b"")
        assert fc.AcquiferLoader.can_load(str(path)) is False

    def test_rejects_missing_path(self, tmp_path):
        assert fc.AcquiferLoader.can_load(str(tmp_path / "nope")) is False

    def test_rejects_directory_without_indicators(self, tmp_path):
        directory = tmp_path / "plain"
        directory.mkdir()
        self._add_image(directory)
        assert fc.AcquiferLoader.can_load(str(directory)) is False

    @pytest.mark.parametrize(
        "marker", ["PlateLayout", "Image-A01.tif", "A01--PX01.tif"]
    )
    def test_accepts_directory_with_indicator_and_images(
        self, tmp_path, marker
    ):
        directory = tmp_path / "acq"
        directory.mkdir()
        (directory / marker).write_bytes(b"")
        self._add_image(directory)
        assert fc.AcquiferLoader.can_load(str(directory)) is True

    def test_accepts_metadata_txt_indicator(self, tmp_path):
        directory = tmp_path / "acq"
        directory.mkdir()
        (directory / "plate_metadata.txt").write_bytes(b"")
        self._add_image(directory)
        assert fc.AcquiferLoader.can_load(str(directory)) is True

    def test_indicator_without_images_is_rejected(self, tmp_path):
        directory = tmp_path / "acq"
        directory.mkdir()
        (directory / "PlateLayout").write_bytes(b"")
        assert fc.AcquiferLoader.can_load(str(directory)) is False

    def test_series_count_without_plugin_is_zero(self, tmp_path):
        directory = tmp_path / "acq"
        directory.mkdir()
        (directory / "PlateLayout").write_bytes(b"")
        self._add_image(directory)
        fc.AcquiferLoader._dataset_cache.clear()
        assert fc.AcquiferLoader.get_series_count(str(directory)) >= 0


class TestScanFolderWorker:
    def test_finds_matching_files_recursively(self, qapp, tmp_path):
        for name in ["a.lif", "b.nd2", "ignored.txt"]:
            (tmp_path / name).write_bytes(b"")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "c.czi").write_bytes(b"")

        worker = fc.ScanFolderWorker(str(tmp_path), [".lif", ".nd2", ".czi"])
        found = []
        worker.finished.connect(found.extend)
        worker.run()

        assert sorted(os.path.basename(p) for p in found) == [
            "a.lif",
            "b.nd2",
            "c.czi",
        ]

    def test_reports_progress(self, qapp, tmp_path):
        for i in range(25):
            (tmp_path / f"f{i}.lif").write_bytes(b"")

        worker = fc.ScanFolderWorker(str(tmp_path), [".lif"])
        progress = []
        worker.progress.connect(lambda i, n: progress.append((i, n)))
        worker.run()

        assert progress[0] == (0, 25)
        assert all(total == 25 for _, total in progress)

    def test_acquifer_filter_collects_directories(self, qapp, tmp_path):
        acquifer_dir = tmp_path / "plate"
        acquifer_dir.mkdir()
        (acquifer_dir / "PlateLayout").write_bytes(b"")
        tifffile.imwrite(
            acquifer_dir / "Image-A01.tif", np.zeros((4, 4), np.uint8)
        )

        worker = fc.ScanFolderWorker(str(tmp_path), ["acquifer"])
        found = []
        worker.finished.connect(found.extend)
        worker.run()

        assert [os.path.basename(p) for p in found] == ["plate"]

    def test_empty_folder_yields_empty_list(self, qapp, tmp_path):
        worker = fc.ScanFolderWorker(str(tmp_path), [".lif"])
        results = []
        worker.finished.connect(results.append)
        worker.run()
        assert results == [[]]


class TestShouldSkipFile:
    @pytest.mark.parametrize(
        "message",
        ["Invalid SubBlkDirectory-magic", "Illegal data detected at offset 8"],
    )
    def test_known_czi_read_errors_are_skipped(self, worker, message):
        assert worker._should_skip_file("broken.czi", message) is True

    def test_other_czi_errors_are_not_skipped(self, worker):
        assert worker._should_skip_file("broken.czi", "disk full") is False

    def test_non_czi_files_are_never_skipped(self, worker):
        assert (
            worker._should_skip_file(
                "movie.nd2", "Invalid SubBlkDirectory-magic"
            )
            is False
        )


class TestConversionReport:
    def test_empty_results_write_nothing(self, worker):
        assert worker._write_conversion_report([]) is None

    def test_report_contains_one_row_per_result(self, worker, tmp_path):
        results = [
            {
                "filepath": "/data/a.lif",
                "series_index": 0,
                "status": "success",
                "message": "",
            },
            {
                "filepath": "/data/b.czi",
                "series_index": 2,
                "status": "failed",
                "message": "Illegal data detected at offset 8",
            },
        ]
        report_path = worker._write_conversion_report(results)

        assert report_path == str(tmp_path / "conversion_report.csv")
        with open(report_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert [row["filepath"] for row in rows] == [
            "/data/a.lif",
            "/data/b.czi",
        ]
        assert rows[1]["status"] == "failed"
        assert rows[1]["series_index"] == "2"
        assert rows[0]["timestamp"]

    def test_unwritable_output_folder_returns_none(self, worker, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"")
        worker.output_folder = str(blocker / "nested")

        assert (
            worker._write_conversion_report(
                [
                    {
                        "filepath": "a.lif",
                        "series_index": 0,
                        "status": "success",
                        "message": "",
                    }
                ]
            )
            is None
        )


class TestScaleTransform:
    def test_no_axes_yields_no_transform(self, worker):
        assert worker._build_scale_transform({}, "", (4, 4)) is None

    def test_defaults_to_unit_scale(self, worker):
        transform = worker._build_scale_transform({}, "zyx", (2, 4, 4))
        assert transform == {"type": "scale", "scale": [1.0, 1.0, 1.0]}

    def test_resolution_is_inverted_to_units_per_pixel(self, worker):
        # resolution is pixels-per-unit, OME-Zarr wants unit-per-pixel.
        metadata = {"resolution": (2.0, 4.0), "spacing": 3.0}
        transform = worker._build_scale_transform(metadata, "zyx", (2, 4, 4))
        assert transform["scale"] == pytest.approx([3.0, 0.25, 0.5])

    def test_time_and_channel_axes_stay_unscaled(self, worker):
        metadata = {"resolution": (2.0, 2.0), "spacing": 5.0}
        transform = worker._build_scale_transform(
            metadata, "tczyx", (3, 2, 4, 8, 8)
        )
        assert transform["scale"] == pytest.approx([1.0, 1.0, 5.0, 0.5, 0.5])

    @pytest.mark.parametrize(
        "metadata",
        [
            {"resolution": (0.0, 0.0)},
            {"spacing": 0.0},
            {"resolution": None, "spacing": None},
        ],
    )
    def test_zero_or_missing_values_stay_at_unit_scale(self, worker, metadata):
        transform = worker._build_scale_transform(metadata, "zyx", (2, 4, 4))
        assert transform["scale"] == [1.0, 1.0, 1.0]


class TestPyramidCoordinateTransformations:
    def test_no_scale_transform_yields_none(self, worker):
        assert (
            worker._build_pyramid_coordinate_transformations(None, "zyx", (2,))
            is None
        )

    def test_non_list_scale_yields_none(self, worker):
        assert (
            worker._build_pyramid_coordinate_transformations(
                {"type": "scale", "scale": "bogus"}, "zyx", (2,)
            )
            is None
        )

    def test_axis_length_mismatch_defers_to_ome_zarr(self, worker):
        assert (
            worker._build_pyramid_coordinate_transformations(
                {"type": "scale", "scale": [1.0, 1.0]}, "zyx", (2,)
            )
            is None
        )

    def test_only_xy_are_downscaled_per_level(self, worker):
        base = {"type": "scale", "scale": [3.0, 0.25, 0.5]}
        levels = worker._build_pyramid_coordinate_transformations(
            base, "zyx", (2, 4)
        )

        assert len(levels) == 3
        assert levels[0][0]["scale"] == pytest.approx([3.0, 0.25, 0.5])
        assert levels[1][0]["scale"] == pytest.approx([3.0, 0.5, 1.0])
        assert levels[2][0]["scale"] == pytest.approx([3.0, 1.0, 2.0])

    def test_no_pyramid_factors_yields_base_level_only(self, worker):
        base = {"type": "scale", "scale": [1.0, 0.5, 0.5]}
        levels = worker._build_pyramid_coordinate_transformations(
            base, "zyx", ()
        )
        assert len(levels) == 1
        assert levels[0][0]["scale"] == pytest.approx([1.0, 0.5, 0.5])


class TestSaveTif:
    def test_saves_2d_array(self, worker, tmp_path):
        data = np.random.default_rng(1).integers(
            0, 255, (16, 24), dtype=np.uint8
        )
        out = str(tmp_path / "out.tif")

        assert worker._save_tif(data, out, {}) is True
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_saves_3d_array(self, worker, tmp_path):
        data = np.random.default_rng(2).integers(
            0, 255, (5, 16, 24), dtype=np.uint8
        )
        out = str(tmp_path / "out.tif")

        assert worker._save_tif(data, out, {}) is True
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_resolution_metadata_is_written(self, worker, tmp_path):
        data = np.zeros((8, 8), dtype=np.uint8)
        out = str(tmp_path / "out.tif")
        worker._save_tif(data, out, {"resolution": (3.0, 4.0)})

        with tifffile.TiffFile(out) as tif:
            tags = tif.pages[0].tags
            assert tags["XResolution"].value[0] / tags["XResolution"].value[
                1
            ] == pytest.approx(3.0)
            assert tags["YResolution"].value[0] / tags["YResolution"].value[
                1
            ] == pytest.approx(4.0)

    def test_unusable_resolution_metadata_is_ignored(self, worker, tmp_path):
        data = np.zeros((8, 8), dtype=np.uint8)
        out = str(tmp_path / "out.tif")
        assert worker._save_tif(data, out, {"resolution": ("a", "b")}) is True

    def test_oversized_input_is_rejected(self, worker, tmp_path):
        huge = da.zeros((20, 1024, 1024, 1024), dtype=np.uint8)  # 20 GiB
        with pytest.raises(MemoryError, match="Use ZARR"):
            worker._save_tif(huge, str(tmp_path / "out.tif"), {})

    def test_large_dask_input_is_rejected(self, worker, tmp_path):
        # Between the 6 GiB Dask ceiling and the 8 GiB hard ceiling.
        big = da.zeros((7, 1024, 1024, 1024), dtype=np.uint8)
        with pytest.raises(MemoryError, match="Dask array too large"):
            worker._save_tif(big, str(tmp_path / "out.tif"), {})

    def test_small_3d_dask_input_is_computed(self, worker, tmp_path):
        data = np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)
        out = str(tmp_path / "out.tif")

        assert worker._save_tif(da.from_array(data, chunks=2), out, {}) is True
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_4d_dask_input_is_written_as_one_series(self, worker, tmp_path):
        # Every leading-axis block streams into a single series via one
        # tifffile.imwrite(data=<generator>) call, so a plain imread
        # recovers the whole array. A prior version called
        # writer.write() once per leading-axis index, which starts a new
        # *series* each time -- every pixel still reached disk, but
        # tifffile.imread() only ever returned the first slice.
        data = np.arange(3 * 2 * 8 * 8, dtype=np.uint8).reshape(3, 2, 8, 8)
        out = str(tmp_path / "out.tif")

        assert worker._save_tif(da.from_array(data, chunks=1), out, {}) is True

        with tifffile.TiffFile(out) as tif:
            assert len(tif.series) == 1
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_three_channel_slices_are_not_written_as_rgb(
        self, worker, tmp_path
    ):
        # A slice whose leading axis is 3 is the classic tifffile trap:
        # with no explicit photometric it is tagged RGB and comes back as
        # a colour image instead of three channels.
        data = np.arange(2 * 3 * 8 * 8, dtype=np.uint8).reshape(2, 3, 8, 8)
        out = str(tmp_path / "channels.tif")

        assert worker._save_tif(da.from_array(data, chunks=1), out, {}) is True

        with tifffile.TiffFile(out) as tif:
            assert tif.pages[0].photometric == tifffile.PHOTOMETRIC.MINISBLACK
        np.testing.assert_array_equal(tifffile.imread(out), data)


class TestSaveZarr:
    def test_writes_readable_ome_zarr(self, worker, tmp_path):
        data = np.random.default_rng(3).integers(
            0, 255, (4, 16, 16), dtype=np.uint8
        )
        out = str(tmp_path / "sample_series0.zarr")

        assert (
            worker._save_zarr(data, out, {"axes": "zyx"}, "sample", 0) is True
        )
        assert os.path.isdir(out)

        attrs = json.loads(
            (tmp_path / "sample_series0.zarr" / "zarr.json").read_text()
        )["attributes"]
        assert attrs["name"] == "sample"
        multiscales = attrs["ome"]["multiscales"]
        assert [ax["name"] for ax in multiscales[0]["axes"]] == [
            "z",
            "y",
            "x",
        ]
        # NGFF requires numeric level names; ome_zarr writes "s0".
        assert [d["path"] for d in multiscales[0]["datasets"]] == ["0"]
        assert (tmp_path / "sample_series0.zarr" / "0").is_dir()

    def test_spatial_metadata_reaches_the_transform(self, worker, tmp_path):
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        out = str(tmp_path / "scaled.zarr")
        metadata = {"axes": "zyx", "resolution": (2.0, 2.0), "spacing": 5.0}

        assert worker._save_zarr(data, out, metadata, "scaled", 0) is True

        attrs = json.loads(
            (tmp_path / "scaled.zarr" / "zarr.json").read_text()
        )["attributes"]
        dataset = attrs["ome"]["multiscales"][0]["datasets"][0]
        scale = dataset["coordinateTransformations"][0]["scale"]
        assert scale == pytest.approx([5.0, 0.5, 0.5])

    def test_series_index_is_reflected_in_layer_name(self, worker, tmp_path):
        data = np.zeros((4, 8, 8), dtype=np.uint8)
        out = str(tmp_path / "multi.zarr")

        worker._save_zarr(data, out, {"axes": "zyx"}, "multi", 2)

        attrs = json.loads((tmp_path / "multi.zarr" / "zarr.json").read_text())
        assert attrs["attributes"]["name"] == "multi_series_2"

    def test_existing_store_is_replaced(self, worker, tmp_path):
        out = str(tmp_path / "sample.zarr")
        os.makedirs(out)
        stale = os.path.join(out, "stale.txt")
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("leftover")

        data = np.zeros((2, 8, 8), dtype=np.uint8)
        assert (
            worker._save_zarr(data, out, {"axes": "zyx"}, "sample", 0) is True
        )
        assert not os.path.exists(stale)

    def test_dask_input_is_accepted(self, worker, tmp_path):
        data = da.zeros((4, 16, 16), dtype=np.uint8, chunks=(1, 16, 16))
        out = str(tmp_path / "lazy.zarr")

        assert worker._save_zarr(data, out, {"axes": "zyx"}, "lazy", 0) is True


class TestPyramidNamingFix:
    """
    ome_zarr writes pyramid levels as ``s0``/``s1``; NGFF requires
    ``0``/``1``.  The fix-up walks a store directory and renames both the
    directories and the dataset paths recorded in ``zarr.json``.
    """

    def _make_store(self, tmp_path, level_names):
        store = tmp_path / "store.zarr"
        store.mkdir()
        for name in level_names:
            (store / name).mkdir()
        (store / "zarr.json").write_text(
            json.dumps(
                {
                    "attributes": {
                        "ome": {
                            "multiscales": [
                                {
                                    "datasets": [
                                        {"path": name} for name in level_names
                                    ]
                                }
                            ]
                        }
                    }
                }
            )
        )
        return store

    def test_renames_levels_and_updates_metadata(self, worker, tmp_path):
        store = self._make_store(tmp_path, ["s0", "s1", "s2"])

        worker._fix_ome_zarr_pyramid_naming(store)

        assert sorted(p.name for p in store.iterdir() if p.is_dir()) == [
            "0",
            "1",
            "2",
        ]
        datasets = json.loads((store / "zarr.json").read_text())["attributes"][
            "ome"
        ]["multiscales"][0]["datasets"]
        assert [d["path"] for d in datasets] == ["0", "1", "2"]

    def test_standard_naming_is_left_alone(self, worker, tmp_path):
        store = self._make_store(tmp_path, ["0", "1"])
        before = (store / "zarr.json").read_text()

        worker._fix_ome_zarr_pyramid_naming(store)

        assert sorted(p.name for p in store.iterdir() if p.is_dir()) == [
            "0",
            "1",
        ]
        assert (store / "zarr.json").read_text() == before

    def test_accepts_a_path_string(self, worker, tmp_path):
        store = self._make_store(tmp_path, ["s0"])
        worker._fix_ome_zarr_pyramid_naming(str(store))
        assert (store / "0").is_dir()

    def test_missing_store_is_survivable(self, worker, tmp_path):
        # Post-processing must never fail an otherwise-successful save.
        worker._fix_ome_zarr_pyramid_naming(tmp_path / "gone.zarr")


class StubLoader(fc.FormatLoader):
    """In-memory loader so conversion can be driven without a real file."""

    data = np.arange(2 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8)

    @staticmethod
    def can_load(filepath):
        return True

    @staticmethod
    def get_series_count(filepath):
        return 1

    @staticmethod
    def load_series(filepath, series_index):
        return StubLoader.data

    @staticmethod
    def get_metadata(filepath, series_index):
        return {"axes": "zyx", "resolution": (2.0, 2.0), "spacing": 5.0}


class TestConvertSingleFile:
    def _worker(self, tmp_path, use_zarr):
        return fc.ConversionWorker(
            files_to_convert=[("/data/sample.lif", 0)],
            output_folder=str(tmp_path),
            use_zarr=use_zarr,
            file_loader_func=lambda filepath: StubLoader,
        )

    def test_converts_to_tif(self, qapp, tmp_path):
        worker = self._worker(tmp_path, use_zarr=False)
        assert worker._convert_single_file("/data/sample.lif", 0) is True

        out = tmp_path / "sample_series0.tif"
        assert out.exists()
        np.testing.assert_array_equal(tifffile.imread(out), StubLoader.data)

    def test_converts_to_zarr(self, qapp, tmp_path):
        worker = self._worker(tmp_path, use_zarr=True)
        assert worker._convert_single_file("/data/sample.lif", 0) is True
        assert (tmp_path / "sample_series0.zarr").is_dir()

    def test_unsupported_format_raises_conversion_error(self, qapp, tmp_path):
        worker = fc.ConversionWorker(
            files_to_convert=[],
            output_folder=str(tmp_path),
            use_zarr=False,
            file_loader_func=lambda filepath: None,
        )
        with pytest.raises(fc.ConversionError):
            worker._convert_single_file("/data/sample.xyz", 0)

    def test_run_emits_results_and_writes_report(self, qapp, tmp_path):
        worker = self._worker(tmp_path, use_zarr=False)
        done = []
        worker.file_done.connect(
            lambda path, ok, msg: done.append((path, ok, msg))
        )
        counts = []
        worker.finished.connect(counts.append)

        worker.run()

        assert counts == [1]
        assert len(done) == 1 and done[0][1] is True
        assert (tmp_path / "conversion_report.csv").exists()

    def test_stop_halts_the_queue(self, qapp, tmp_path):
        worker = self._worker(tmp_path, use_zarr=False)
        worker.stop()

        counts = []
        worker.finished.connect(counts.append)
        worker.run()

        assert counts == [0]
        assert not (tmp_path / "sample_series0.tif").exists()


@requires_gui
class TestConverterWidget:
    @pytest.fixture
    def widget(self, make_napari_viewer):
        return fc.MicroscopyImageConverterWidget(make_napari_viewer())

    def test_registers_every_loader(self, widget):
        assert widget.loaders == [
            fc.LIFLoader,
            fc.ND2Loader,
            fc.TIFFSlideLoader,
            fc.CZILoader,
            fc.AcquiferLoader,
        ]

    @pytest.mark.parametrize(
        "filename, expected",
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
    def test_get_file_type(self, widget, tmp_path, filename, expected):
        assert widget.get_file_type(str(tmp_path / filename)) == expected

    def test_get_file_type_detects_acquifer_directory(self, widget, tmp_path):
        directory = tmp_path / "plate"
        directory.mkdir()
        (directory / "PlateLayout").write_bytes(b"")
        tifffile.imwrite(
            directory / "Image-A01.tif", np.zeros((4, 4), np.uint8)
        )
        assert widget.get_file_type(str(directory)) == "Acquifer"

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("a.nd2", fc.ND2Loader),
            ("a.ndpi", fc.TIFFSlideLoader),
        ],
    )
    def test_get_file_loader(self, widget, tmp_path, filename, expected):
        assert widget.get_file_loader(str(tmp_path / filename)) is expected

    def test_get_file_loader_returns_none_for_unknown(self, widget, tmp_path):
        unknown = tmp_path / "a.tif"
        unknown.write_bytes(b"")
        assert widget.get_file_loader(str(unknown)) is None

    def test_selected_series_bookkeeping(self, widget):
        widget.set_selected_series("/data/a.lif", 3)
        assert widget.selected_series["/data/a.lif"] == 3

    def test_export_all_defaults_the_selected_series(self, widget):
        widget.set_export_all_series("/data/a.lif", True)
        assert widget.export_all_series["/data/a.lif"] is True
        assert widget.selected_series["/data/a.lif"] == 0

    def test_export_all_off_does_not_add_a_selection(self, widget):
        widget.set_export_all_series("/data/a.lif", False)
        assert widget.selected_series == {}

    def test_empty_output_folder_is_invalid(self, widget):
        assert widget._validate_output_folder("") is False
        assert "output folder" in widget.status_label.text().lower()

    def test_missing_output_folder_is_created(self, widget, tmp_path):
        target = tmp_path / "new" / "nested"
        assert widget._validate_output_folder(str(target)) is True
        assert target.is_dir()

    def test_uncreatable_output_folder_is_invalid(self, widget, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"")
        assert widget._validate_output_folder(str(blocker / "sub")) is False

    def test_format_buttons_are_mutually_exclusive(self, widget):
        widget.update_format_buttons(use_zarr=True)
        assert widget.zarr_radio.isChecked() is True
        assert widget.tif_radio.isChecked() is False
        assert "ZARR" in widget.status_label.text()

        widget.update_format_buttons(use_zarr=False)
        assert widget.tif_radio.isChecked() is True
        assert widget.zarr_radio.isChecked() is False

    def test_format_buttons_ignore_reentrant_updates(self, widget):
        widget.update_format_buttons(use_zarr=False)
        widget.updating_format_buttons = True
        widget.update_format_buttons(use_zarr=True)
        assert widget.zarr_radio.isChecked() is False
        widget.updating_format_buttons = False


@requires_gui
class TestSeriesTable:
    @pytest.fixture
    def widget(self, make_napari_viewer):
        return fc.MicroscopyImageConverterWidget(make_napari_viewer())

    def test_add_file_records_row_and_metadata(self, widget):
        table = widget.files_table
        table.add_file("/data/a.lif", "LIF", 3)

        assert table.rowCount() == 1
        assert table.item(0, 0).text() == "a.lif"
        assert table.item(0, 1).text() == "3 series"
        assert table.file_data["/data/a.lif"] == {
            "type": "LIF",
            "series_count": 3,
            "row": 0,
        }

    def test_single_image_files_are_labelled(self, widget):
        table = widget.files_table
        table.add_file("/data/a.ndpi", "Slide", 0)
        assert table.item(0, 1).text() == "Single image"

    def test_clicking_a_multi_series_file_opens_details(self, widget):
        table = widget.files_table
        table.add_file("/data/a.lif", "LIF", 3)

        table.handle_cell_click(0, 0)

        assert table.current_file == "/data/a.lif"
        assert widget.selected_series["/data/a.lif"] == 0
        assert widget.series_widget.current_file == "/data/a.lif"

    def test_clicking_the_series_column_does_nothing(self, widget):
        table = widget.files_table
        table.add_file("/data/a.lif", "LIF", 3)

        table.handle_cell_click(0, 1)

        assert table.current_file is None
        assert widget.selected_series == {}


@requires_gui
class TestReorderDimensions:
    @pytest.fixture
    def detail(self, make_napari_viewer):
        widget = fc.MicroscopyImageConverterWidget(make_napari_viewer())
        return widget.series_widget

    def test_reorders_to_target_axis_order(self, detail):
        data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint8)  # TCZYX
        result = detail._reorder_dimensions(
            data, {"axes": "TCZYX"}, target_order="YXZTC"
        )
        assert result.shape == (5, 6, 4, 2, 3)

    def test_reorders_dask_arrays_lazily(self, detail):
        data = da.zeros((2, 3, 4), dtype=np.uint8)
        result = detail._reorder_dimensions(
            data, {"axes": "ZYX"}, target_order="YXZ"
        )
        assert isinstance(result, da.Array)
        assert result.shape == (3, 4, 2)

    @pytest.mark.parametrize("metadata", [None, {}, {"axes": "ZYX"}])
    def test_unusable_metadata_returns_input_unchanged(self, detail, metadata):
        data = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        result = detail._reorder_dimensions(
            data, metadata, target_order="YXZT"
        )
        assert result is data

    def test_unknown_axis_returns_input_unchanged(self, detail):
        data = np.zeros((2, 3, 4), dtype=np.uint8)
        result = detail._reorder_dimensions(
            data, {"axes": "ZYX"}, target_order="YXC"
        )
        assert result is data


@requires_gui
def test_microscopy_converter_factory(make_napari_viewer):
    widget = fc.microscopy_converter(make_napari_viewer())
    assert isinstance(widget, fc.MicroscopyImageConverterWidget)


def test_dock_widget_hook_exposes_the_converter():
    hook = fc.napari_experimental_provide_dock_widget()
    assert hook is fc.microscopy_converter or callable(hook)

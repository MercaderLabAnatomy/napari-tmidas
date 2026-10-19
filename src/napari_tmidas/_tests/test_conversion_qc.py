"""Read-back check of converted files, and the OME-TIFF metadata it relies on."""

import csv
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import tifffile

import napari_tmidas._file_conversion as fc
from napari_tmidas._output_qc import (
    check_converted_output,
    expected_physical_sizes,
    read_physical_sizes,
)

ZYX_MD = {"axes": "ZYX", "resolution": (2.0, 4.0), "spacing": 3.0, "unit": "um"}


class _Loader:
    def __init__(self, data, metadata):
        self.data, self.metadata = data, metadata

    def load_series(self, filepath, series_index):
        return self.data

    def get_metadata(self, filepath, series_index):
        return self.metadata


def _worker(tmp_path, data, metadata, use_zarr=False):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    src = tmp_path / "src.czi"
    src.touch()
    worker = fc.ConversionWorker(
        files_to_convert=[(str(src), 0)],
        output_folder=str(out),
        use_zarr=use_zarr,
        file_loader_func=lambda filepath: _Loader(data, metadata),
    )
    return worker, str(src)


def test_expected_physical_sizes_inverts_the_resolution():
    assert expected_physical_sizes(ZYX_MD) == {"X": 0.5, "Y": 0.25, "Z": 3.0}
    assert expected_physical_sizes({}) == {}
    assert expected_physical_sizes({"resolution": ("a", "b")}) == {}
    assert expected_physical_sizes({"resolution": (0, 0), "spacing": 0}) == {}


class TestOmeTiffMetadata:
    def test_valid_axes_and_sizes(self):
        assert fc._ome_tiff_metadata(ZYX_MD, 3) == {
            "axes": "ZYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.25,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 3.0,
            "PhysicalSizeZUnit": "µm",
        }

    @pytest.mark.parametrize(
        "axes, ndim", [("MZYX", 4), ("ZYX", 4), ("ZZYX", 4), ("YXZ", 3)]
    )
    def test_axes_tifffile_cannot_store_are_left_out_with_z(self, axes, ndim):
        ome = fc._ome_tiff_metadata({**ZYX_MD, "axes": axes}, ndim)
        assert "axes" not in ome
        assert "PhysicalSizeZ" not in ome
        assert ome["PhysicalSizeX"] == 0.5


class TestConvertedFilesKeepAxesAndPixelSize:
    """The ImageJ writer stored ZYX as channels and dropped Z spacing."""

    def test_zyx_tiff(self, qapp, tmp_path):
        data = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
        worker, src = _worker(tmp_path, data, ZYX_MD)
        assert worker._convert_single_file(src, 0)
        out = fc.conversion_output_path(worker.output_folder, src, 0, False)
        with tifffile.TiffFile(out) as tif:
            assert tif.series[0].axes == "ZYX"
            np.testing.assert_array_equal(tif.series[0].asarray(), data)
        assert read_physical_sizes(out) == {"X": 0.5, "Y": 0.25, "Z": 3.0}
        assert worker.last_qc_problems == []

    def test_large_dask_tiff_takes_the_chunked_path(self, qapp, tmp_path):
        data = da.ones((2, 2, 3, 8, 8), dtype=np.uint16, chunks=(1, 2, 3, 8, 8))
        worker, src = _worker(tmp_path, data, {**ZYX_MD, "axes": "TCZYX"})
        assert worker._convert_single_file(src, 0)
        out = fc.conversion_output_path(worker.output_folder, src, 0, False)
        with tifffile.TiffFile(out) as tif:
            assert tif.series[0].axes == "TCZYX"
        assert read_physical_sizes(out) == {"X": 0.5, "Y": 0.25, "Z": 3.0}
        assert worker.last_qc_problems == []

    def test_zarr(self, qapp, tmp_path):
        data = np.ones((2, 3, 8, 8), dtype=np.uint16)
        worker, src = _worker(tmp_path, data, {**ZYX_MD, "axes": "CZYX"}, True)
        assert worker._convert_single_file(src, 0)
        assert worker.last_qc_problems == []


class TestCheckConvertedOutput:
    def test_reports_every_mismatch(self, tmp_path):
        path = str(tmp_path / "bad.tif")
        # What the old writer produced: no Z spacing, default 1/1 resolution.
        tifffile.imwrite(path, np.ones((4, 8, 8), np.uint16), imagej=True)

        problems = check_converted_output(path, (4, 8, 8), np.uint8, ZYX_MD)

        assert problems == [
            "dtype uint16 differs from source uint8",
            "X pixel size is 1 µm, source has 0.5 µm",
            "Y pixel size is 1 µm, source has 0.25 µm",
            "Z pixel size 3 µm was not written",
        ]

    def test_lost_data_and_moved_yx(self, tmp_path):
        path = str(tmp_path / "a.tif")
        tifffile.imwrite(path, np.ones((2, 8, 8), np.uint8))
        assert check_converted_output(path, (4, 8, 8), np.uint8, {}) == [
            "shape (2, 8, 8) does not hold source (4, 8, 8)"
        ]
        assert check_converted_output(path, (8, 2, 8), np.uint8, {}) == [
            "YX (8, 8) differs from source (2, 8)"
        ]

    def test_z_spacing_of_data_without_z_is_not_expected(self, tmp_path):
        path = str(tmp_path / "c.tif")
        tifffile.imwrite(
            path,
            np.ones((2, 8, 8), np.uint8),
            ome=True,
            metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.25},
        )
        md = {**ZYX_MD, "axes": "CYX"}
        assert check_converted_output(path, (2, 8, 8), np.uint8, md) == []

    def test_unreadable_output(self, tmp_path):
        path = tmp_path / "x.tif"
        path.write_bytes(b"nope")
        (problem,) = check_converted_output(str(path), (8, 8), np.uint8, {})
        assert problem.startswith("output cannot be read back")


class TestQcReachesReportAndWidget:
    def test_problems_go_into_message_and_report(self, qapp, tmp_path, monkeypatch):
        data = np.ones((4, 8, 8), np.uint16)
        worker, src = _worker(tmp_path, data, ZYX_MD)
        monkeypatch.setattr(
            fc, "check_converted_output", lambda *a: ["Z pixel size lost"]
        )
        done = []
        worker.file_done.connect(lambda *args: done.append(args))

        worker.run()

        assert done == [
            (src, True, "Conversion successful; QC: Z pixel size lost")
        ]
        with open(
            tmp_path / "out" / "conversion_report.csv", encoding="utf-8"
        ) as handle:
            (row,) = list(csv.DictReader(handle))
        assert row["qc"] == "Z pixel size lost"

    def test_clean_conversion_reports_ok(self, qapp, tmp_path):
        worker, src = _worker(tmp_path, np.ones((4, 8, 8), np.uint16), ZYX_MD)
        worker.run()
        with open(
            tmp_path / "out" / "conversion_report.csv", encoding="utf-8"
        ) as handle:
            (row,) = list(csv.DictReader(handle))
        assert row["qc"] == "ok"
        assert row["message"] == "Conversion successful"

    def test_widget_collects_and_warns_once(self, qapp, monkeypatch):
        import napari.utils.notifications as notifications

        shown = []
        monkeypatch.setattr(notifications, "show_warning", shown.append)
        # A stub viewer: the widget only stores it, and a real one needs
        # OpenGL, which segfaults on headless macOS CI.
        widget = fc.MicroscopyImageConverterWidget(SimpleNamespace(status=""))
        widget.status_label.setText("Successfully converted 2 files to /out")
        widget._conversion_qc_issues = []

        widget.handle_conversion_result(
            "/d/a.czi", True, "Conversion successful; QC: Z pixel size lost"
        )
        widget.handle_conversion_result("/d/b.czi", True, "Conversion successful")
        widget._report_conversion_qc_issues()

        assert widget._conversion_qc_issues == ["a.czi: Z pixel size lost"]
        assert len(shown) == 1 and "a.czi: Z pixel size lost" in shown[0]
        assert "1 file(s) failed the read-back check" in widget.status_label.text()

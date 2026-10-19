"""Tests for the automatic all-zero output check in batch runs."""

import numpy as np
import pytest
import tifffile

import napari_tmidas._file_selector as fs
from napari_tmidas._output_qc import (
    all_zero_outputs,
    output_is_all_zero,
    result_output_paths,
)


def _tif(path, data):
    tifffile.imwrite(
        str(path),
        data,
        ome=True,
        metadata={"axes": "TZYX"[-data.ndim :]},
        compression="zlib",
    )
    return str(path)


class TestOutputIsAllZero:
    def test_empty_stack_is_all_zero(self, tmp_path):
        path = _tif(tmp_path / "e.tif", np.zeros((3, 2, 8, 8), np.uint32))
        assert output_is_all_zero(path) is True

    def test_one_voxel_in_the_last_plane_is_enough(self, tmp_path):
        data = np.zeros((3, 2, 8, 8), np.uint32)
        data[-1, -1, 7, 7] = 4
        assert output_is_all_zero(_tif(tmp_path / "s.tif", data)) is False

    def test_2d_image(self, tmp_path):
        assert output_is_all_zero(_tif(tmp_path / "z.tif", np.zeros((8, 8), np.uint8)))
        assert not output_is_all_zero(
            _tif(tmp_path / "o.tif", np.ones((8, 8), np.uint8))
        )

    def test_zarr(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "e.zarr")
        arr = zarr.open_array(path, mode="w", shape=(2, 8, 8), dtype="u2")
        arr[:] = 0
        assert output_is_all_zero(path) is True
        arr[1, 3, 3] = 1
        assert output_is_all_zero(path) is False

    @pytest.mark.parametrize("name", ["spots.csv", "spots.npy", "missing.tif"])
    def test_non_images_and_missing_files_cannot_tell(self, tmp_path, name):
        path = tmp_path / name
        if name != "missing.tif":
            path.write_bytes(b"0,0\n")
        assert output_is_all_zero(str(path)) is None

    def test_unreadable_tiff_cannot_tell(self, tmp_path):
        path = tmp_path / "broken.tif"
        path.write_bytes(b"not a tiff")
        assert output_is_all_zero(str(path)) is None


def test_result_paths_and_basenames(tmp_path):
    empty = _tif(tmp_path / "e.tif", np.zeros((8, 8), np.uint16))
    full = _tif(tmp_path / "f.tif", np.ones((8, 8), np.uint16))
    assert result_output_paths({"processed_file": empty}) == [empty]
    assert result_output_paths({"processed_files": [empty, full]}) == [empty, full]
    assert result_output_paths({"original_file": empty}) == []
    assert all_zero_outputs([empty, full]) == ["e.tif"]


class TestWorkerFlagsEmptyOutputs:
    def _run(self, tmp_path, func):
        src = _tif(tmp_path / "in.tif", np.ones((8, 8), np.uint16))
        out = tmp_path / "out"
        out.mkdir()
        worker = fs.ProcessingWorker([src], func, {}, str(out), "", "_proc")
        results = []
        worker.file_processed.connect(results.append)
        worker.run()
        return results

    def test_all_zero_output_is_flagged(self, tmp_path):
        def wipe(image):
            return np.zeros_like(image)

        (result,) = self._run(tmp_path, wipe)
        assert result["all_zero_outputs"] == [
            result["processed_file"].rsplit("/", 1)[-1]
        ]

    def test_normal_output_is_not_flagged(self, tmp_path):
        def keep(image):
            return image

        (result,) = self._run(tmp_path, keep)
        assert "all_zero_outputs" not in result


class TestWidgetWarning:
    class _Viewer:
        status = ""

    def _widget(self, results):
        # Only what _warn_about_all_zero_outputs touches.
        widget = fs.FileResultsWidget.__new__(fs.FileResultsWidget)
        widget.viewer = self._Viewer()
        widget.processed_files_info = results
        return widget

    def test_warns_once_naming_the_files(self, monkeypatch):
        import napari.utils.notifications as notifications

        shown = []
        monkeypatch.setattr(notifications, "show_warning", shown.append)
        names = [f"f{i}_labels.tif" for i in range(7)]
        widget = self._widget(
            [{"all_zero_outputs": [n]} for n in names] + [{"processed_file": "x"}]
        )

        widget._warn_about_all_zero_outputs()

        assert len(shown) == 1
        assert shown[0].startswith("7 of the outputs contain only zeros")
        assert "f0_labels.tif" in shown[0] and "and 2 more" in shown[0]
        assert widget.viewer.status == shown[0]

    def test_silent_when_nothing_is_empty(self, monkeypatch):
        import napari.utils.notifications as notifications

        shown = []
        monkeypatch.setattr(notifications, "show_warning", shown.append)
        widget = self._widget([{"processed_file": "x"}])
        widget._warn_about_all_zero_outputs()
        assert shown == []
        assert widget.viewer.status == ""

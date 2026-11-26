"""
Edge cases across small processing modules: track lineage files with
nothing usable in them, the lazy-input signature helpers, registry
discovery when a module fails to import, and Merge Small Labels on inputs
it cannot stream.
"""

import inspect

import numpy as np
import pytest
import tifffile

import napari_tmidas.processing_functions as pf
import napari_tmidas.processing_functions.basic  # noqa: F401  registers Mirror Labels
from napari_tmidas.processing_functions import track_qc
from napari_tmidas.processing_functions._chunked import (
    independent_leading,
    lazy_capable,
)
from napari_tmidas.processing_functions.merge_small_labels import (
    merge_small_labels,
)

# --------------------------------------------------------------------------
# track_qc
# --------------------------------------------------------------------------


class TestLineageThatCannotBeUsed:
    @pytest.mark.parametrize(
        "content",
        ["", "1 0 4\n2 5 9\n"],  # nothing, or no parent column
        ids=["empty", "three-columns"],
    )
    def test_no_lineage_csv_is_written(self, tmp_path, content):
        res_track = tmp_path / "res_track.txt"
        res_track.write_text(content)
        labels = tmp_path / "cells_tracked.tif"

        assert track_qc.ctc_res_track_to_lineage(str(res_track), str(labels)) is None
        assert not (tmp_path / track_qc.lineage_csv_path(str(labels))).exists()

    def test_an_ultrack_table_without_parents_gives_no_lineage(self, tmp_path):
        labels = tmp_path / "cells_ultrack.tif"
        labels.touch()
        (tmp_path / "cells_ultrack_tracks.csv").write_text(
            "track_id,t,y,x\n1,0,3,4\n1,1,3,5\n"
        )

        assert track_qc.read_lineage(str(labels)) is None

    def test_an_ultrack_table_with_parents_gives_its_divisions(self, tmp_path):
        labels = tmp_path / "cells_ultrack.tif"
        labels.touch()
        (tmp_path / "cells_ultrack_tracks.csv").write_text(
            "track_id,parent_track_id,t\n1,-1,0\n2,1,1\n3,1,1\n3,1,2\n"
        )

        assert track_qc.read_lineage(str(labels)) == {2: 1, 3: 1}


def test_track_statistics_need_an_image_or_a_path():
    with pytest.raises(ValueError, match="No image or file path"):
        track_qc.track_statistics_qc(None)


# --------------------------------------------------------------------------
# _chunked
# --------------------------------------------------------------------------


class TestLazyCapableSignature:
    def test_source_path_is_offered_before_var_keywords(self):
        @lazy_capable
        def body(image, sigma=1.0, **kwargs):
            return image

        params = list(inspect.signature(body).parameters)

        assert params == ["image", "sigma", "_source_filepath", "kwargs"]

    def test_a_declared_source_path_is_left_as_written(self):
        def body(image, _source_filepath=None):
            return image

        wrapped = lazy_capable(body)

        assert inspect.signature(wrapped) == inspect.signature(body)


@pytest.mark.parametrize("order", ["QZYX", "TSYX"])
def test_an_order_with_unknown_axes_is_not_split(order):
    # Unknown letters mean the axes cannot be trusted: process whole.
    assert independent_leading()(order, 4) == 4


# --------------------------------------------------------------------------
# processing_functions/__init__
# --------------------------------------------------------------------------


def test_function_info_lists_every_registered_function():
    info = pf.get_processing_function_info()

    assert set(info) == set(pf.BatchProcessingRegistry.list_functions())
    entry = info["Mirror Labels"]
    assert entry["suffix"] == "_mirrored"
    assert "axis" in entry["parameters"]
    assert entry["description"]


@pytest.mark.parametrize(
    "error, expected",
    [
        (ImportError("No module named 'torch'"), "Failed to import"),
        (
            ValueError("numpy.dtype size changed, may indicate binary incompatibility"),
            "NumPy binary incompatibility",
        ),
    ],
)
def test_a_module_that_fails_to_import_is_reported_and_skipped(
    monkeypatch, capsys, error, expected
):
    def failing_import(name, *args, **kwargs):
        raise error

    monkeypatch.setattr(pf.importlib, "import_module", failing_import)
    monkeypatch.setattr(pf.importlib, "reload", failing_import)

    names = pf.discover_and_load_processing_functions()

    out = capsys.readouterr().out
    assert expected in out
    # Functions registered before stay available.
    assert "Mirror Labels" in names


# --------------------------------------------------------------------------
# merge_small_labels
# --------------------------------------------------------------------------


def test_streaming_a_1d_label_file_is_refused(tmp_path):
    path = tmp_path / "line.tif"
    tifffile.imwrite(str(path), np.array([0, 1, 1, 2], dtype=np.uint16))

    with pytest.raises(ValueError, match="2D\\+ label image"):
        merge_small_labels(
            None,
            min_size=2,
            _source_filepath=str(path),
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_merged",
        )


def test_an_unloaded_file_that_cannot_be_streamed_is_read_and_merged(tmp_path):
    from skimage.io import imsave

    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[0:4, 0:4] = 1
    labels[4, 0] = 2  # one voxel touching label 1
    path = tmp_path / "labels.png"
    imsave(str(path), labels, check_contrast=False)

    result = merge_small_labels(
        None,
        min_size=3,
        _source_filepath=str(path),
        _output_folder=str(tmp_path / "out"),
        _output_suffix="_merged",
    )

    expected = labels.copy()
    expected[4, 0] = 1
    np.testing.assert_array_equal(result, expected)

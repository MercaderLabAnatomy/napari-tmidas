"""Tests for the tracking QC: track table, statistics and the batch function."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_tmidas.processing_functions.track_qc import (
    STATS_CSV_NAME,
    TRACKS_CSV_SUFFIX,
    track_statistics,
    track_statistics_qc,
    tracks_from_labels,
)


def _movie(n_frames=5, shape=(16, 16)):
    """
    A TYX movie with four tracks of known behaviour:

    1: every frame, moving right one pixel per frame
    2: frames 0-1 and 3-4 (gap at 2)
    3: frames 2-4 (starts late)
    4: frame 0 only (short, ends early)
    """
    movie = np.zeros((n_frames,) + shape, dtype=np.uint32)
    for t in range(n_frames):
        movie[t, 2:4, t : t + 2] = 1
        if t != 2:
            movie[t, 10:12, 10:12] = 2
        if t >= 2:
            movie[t, 6, 6] = 3
    movie[0, 14, 14] = 4
    return movie


def _write(path, data, axes):
    tifffile.imwrite(
        str(path), data, ome=True, metadata={"axes": axes}, compression="zlib"
    )
    return str(path)


class TestTracksFromLabels:
    def test_table_layout_and_centroids(self):
        table, n_frames = tracks_from_labels(_movie())

        assert n_frames == 5
        assert list(table.columns) == ["track_id", "t", "y", "x", "voxels"]
        track1 = table[table.track_id == 1]
        assert track1.t.tolist() == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(track1.y, 2.5)
        np.testing.assert_allclose(track1.x, [0.5, 1.5, 2.5, 3.5, 4.5])
        assert track1.voxels.tolist() == [4] * 5
        # Sorted by track, then time: what napari's Tracks layer expects.
        assert table.track_id.is_monotonic_increasing
        assert table[table.track_id == 2].t.tolist() == [0, 1, 3, 4]

    def test_tzyx_file_streams_the_same_as_an_array(self, tmp_path):
        movie = np.zeros((3, 4, 8, 8), dtype=np.uint16)
        movie[:, 1, 2, 2] = 7
        movie[:, 3, 2, 2] = 7  # same object in two z planes
        movie[1, 0, 5, 5] = 9
        path = _write(tmp_path / "m.tif", movie, "TZYX")

        from_file, n = tracks_from_labels(path)
        from_array, _ = tracks_from_labels(movie)

        assert n == 3
        pd.testing.assert_frame_equal(from_file, from_array)
        assert list(from_file.columns) == ["track_id", "t", "z", "y", "x", "voxels"]
        track7 = from_file[from_file.track_id == 7]
        np.testing.assert_allclose(track7.z, 2.0)
        assert track7.voxels.tolist() == [2, 2, 2]

    def test_empty_movie(self):
        table, n = tracks_from_labels(np.zeros((3, 8, 8), np.uint16))
        assert n == 3 and len(table) == 0

    @pytest.mark.parametrize("shape", [(8, 8), (2, 2, 2, 8, 8)])
    def test_wrong_dimensionality_is_rejected(self, shape):
        with pytest.raises(ValueError, match="TYX or TZYX"):
            tracks_from_labels(np.zeros(shape, np.uint16))

    def test_a_non_time_dimension_order_is_rejected(self):
        with pytest.raises(ValueError, match="T first"):
            tracks_from_labels(np.zeros((3, 8, 8), np.uint16), "ZYX")


def test_track_statistics():
    table, n_frames = tracks_from_labels(_movie())

    stats = track_statistics(table, n_frames, min_track_length=3)

    assert stats == {
        "frames": 5,
        "objects_per_frame_min": 2,
        "objects_per_frame_mean": 2.6,
        "objects_per_frame_max": 3,
        "tracks": 4,
        "track_length_mean": 3.25,
        "track_length_median": 3.5,
        "tracks_full_length": 1,
        "tracks_full_length_fraction": 0.25,
        "tracks_short": 1,
        "tracks_with_gaps": 1,
        "tracks_starting_late": 1,
        "tracks_ending_early": 1,
    }


def test_statistics_of_an_empty_movie():
    table, n = tracks_from_labels(np.zeros((3, 8, 8), np.uint16))
    stats = track_statistics(table, n)
    assert stats["tracks"] == 0
    assert stats["objects_per_frame_max"] == 0
    assert stats["tracks_full_length_fraction"] == 0.0


class TestBatchFunction:
    def test_writes_stats_row_and_track_table(self, tmp_path):
        src = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        out = tmp_path / "out"

        result = track_statistics_qc(
            None, _source_filepath=src, _output_folder=str(out), _output_suffix="_track_qc"
        )

        assert result is None
        stats = pd.read_csv(out / STATS_CSV_NAME)
        assert stats.file.tolist() == ["a_tracked.tif"]
        assert stats.tracks.tolist() == [4]
        tracks = pd.read_csv(out / f"a_tracked{TRACKS_CSV_SUFFIX}")
        assert len(tracks) == 5 + 4 + 3 + 1

    def test_rerun_replaces_the_row_and_keeps_other_files(self, tmp_path):
        out = str(tmp_path / "out")
        a = _write(tmp_path / "a.tif", _movie(), "TYX")
        b = _write(tmp_path / "b.tif", _movie()[:, :, ::-1], "TYX")
        for src in (a, b, a):
            track_statistics_qc(
                None,
                save_track_table=False,
                _source_filepath=src,
                _output_folder=out,
            )
        stats = pd.read_csv(tmp_path / "out" / STATS_CSV_NAME)
        assert stats.file.tolist() == ["a.tif", "b.tif"]
        assert not list((tmp_path / "out").glob(f"*{TRACKS_CSV_SUFFIX}"))

    def test_scripting_without_output_folder_returns_stats(self):
        stats = track_statistics_qc(_movie(), min_track_length=2)
        assert stats["tracks_short"] == 1

    def test_registered_and_streams(self):
        from napari_tmidas._registry import BatchProcessingRegistry

        info = BatchProcessingRegistry.get_function_info("Track Statistics (QC)")
        assert info is not None
        assert info["func"] is track_statistics_qc
        assert track_statistics_qc.skip_load is True


class TestLoadTracksForLabels:
    def test_computed_live_from_a_small_file(self, tmp_path):
        from napari_tmidas.processing_functions.track_qc import (
            load_tracks_for_labels,
        )

        path = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        data, graph, source = load_tracks_for_labels(path, 3)
        assert source == "computed from the labels"
        assert graph == {}
        assert data.shape == (13, 4)  # track_id, t, y, x

    def test_prefers_a_table_written_by_track_statistics(self, tmp_path):
        from napari_tmidas.processing_functions.track_qc import (
            load_tracks_for_labels,
        )

        path = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        track_statistics_qc(
            None, _source_filepath=path, _output_folder=str(tmp_path / "qc")
        )
        data, _, source = load_tracks_for_labels(path, 3)
        assert source == f"a_tracked{TRACKS_CSV_SUFFIX}"
        assert data.shape == (13, 4)

    def test_ultrack_table_carries_lineage(self, tmp_path):
        from napari_tmidas.processing_functions.track_qc import (
            load_tracks_for_labels,
        )

        path = _write(tmp_path / "a_ultrack.tif", _movie(), "TYX")
        pd.DataFrame(
            {
                "track_id": [2, 1, 1, 3],
                "t": [2, 1, 0, 2],
                "y": [1.0, 1.0, 1.0, 2.0],
                "x": [1.0, 1.0, 1.0, 2.0],
                "parent_track_id": [1, -1, -1, 1],
            }
        ).to_csv(tmp_path / "a_ultrack_tracks.csv", index=False)

        data, graph, source = load_tracks_for_labels(path, 3)

        assert source == "a_ultrack_tracks.csv"
        assert graph == {2: [1], 3: [1]}
        assert data[:, 0].tolist() == [1, 1, 2, 3]  # sorted for napari
        assert data[:, 1].tolist() == [0, 1, 2, 2]

    def test_large_file_is_not_scanned(self, tmp_path, monkeypatch):
        import napari_tmidas.processing_functions.track_qc as tq

        monkeypatch.setattr(tq, "LIVE_TRACKS_MAX_BYTES", 10)
        path = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        data, graph, reason = tq.load_tracks_for_labels(path, 3)
        assert data is None and "Track Statistics (QC)" in reason

    def test_dimension_mismatch_and_empty(self, tmp_path):
        from napari_tmidas.processing_functions.track_qc import (
            load_tracks_for_labels,
        )

        path = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        assert load_tracks_for_labels(path, 4)[0] is None
        empty = _write(
            tmp_path / "e_tracked.tif", np.zeros((3, 8, 8), np.uint16), "TYX"
        )
        assert load_tracks_for_labels(empty, 3) == (
            None,
            None,
            "no objects in the tracked labels",
        )

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("a_tracked.tif", True),
            ("a_hoct_tracked.tif", True),
            ("a_ultrack.tif", True),
            ("a_tracked_merged.tif", True),
            ("a_labels.tif", False),
        ],
    )
    def test_is_tracked_output(self, name, expected):
        from napari_tmidas.processing_functions.track_qc import (
            is_tracked_output,
        )

        assert is_tracked_output(f"/x/{name}") is expected


class TestResultsTableShowsTracks:
    def test_tracked_labels_get_a_tracks_layer(self, make_napari_viewer, tmp_path):
        import napari_tmidas._file_selector as fs

        movie = np.zeros((3, 2, 16, 16), dtype=np.uint32)
        movie[:, 1, 4:6, 4:6] = 5
        tifffile.imwrite(
            str(tmp_path / "a_tracked.tif"),
            movie,
            ome=True,
            metadata={"axes": "TZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5},
        )
        viewer = make_napari_viewer()
        table = fs.ProcessedFilesTableWidget(viewer)

        table._load_processed_image(str(tmp_path / "a_tracked.tif"))
        table._load_processed_image(str(tmp_path / "a_tracked.tif"))

        kinds = [type(layer).__name__ for layer in viewer.layers]
        assert kinds == ["Labels", "Tracks"], "reloading replaces, not stacks"
        labels, tracks = viewer.layers
        assert tuple(tracks.scale) == tuple(labels.scale)
        assert "1 tracks" in viewer.status

    def test_untracked_labels_get_none(self, make_napari_viewer, tmp_path):
        import napari_tmidas._file_selector as fs

        movie = np.zeros((3, 16, 16), dtype=np.uint32)
        movie[:, 4:6, 4:6] = 5
        path = _write(tmp_path / "a_labels.tif", movie, "TYX")
        viewer = make_napari_viewer()
        fs.ProcessedFilesTableWidget(viewer)._load_processed_image(path)
        assert [type(layer).__name__ for layer in viewer.layers] == ["Labels"]

    def test_a_tracks_failure_keeps_the_labels(
        self, make_napari_viewer, tmp_path, monkeypatch
    ):
        import napari_tmidas._file_selector as fs
        import napari_tmidas.processing_functions.track_qc as tq

        def broken(*args):
            raise RuntimeError("boom")

        monkeypatch.setattr(tq, "load_tracks_for_labels", broken)
        path = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        viewer = make_napari_viewer()
        fs.ProcessedFilesTableWidget(viewer)._load_processed_image(path)
        assert [type(layer).__name__ for layer in viewer.layers] == ["Labels"]

    def test_tracked_file_in_the_original_column(self, make_napari_viewer, tmp_path):
        import napari_tmidas._file_selector as fs

        path = _write(tmp_path / "a_tracked.tif", _movie(), "TYX")
        viewer = make_napari_viewer()
        table = fs.ProcessedFilesTableWidget(viewer)
        table._load_original_image(path)
        assert [type(layer).__name__ for layer in viewer.layers] == [
            "Labels",
            "Tracks",
        ]
        assert viewer.layers[-1] in table.current_original_images

"""
Replaying a recorded frame removal from a hand-written run record: runs
that cannot be replayed are refused up front, a failed write is reported in
the outcome, and a workflow log that cannot be written never breaks a save.
"""

import numpy as np
import pytest
import tifffile

import napari_tmidas._frame_removal as fr


def _run(**overrides):
    run = {
        "run_id": "r1",
        "widget": fr.WIDGET_NAME,
        "params": {"frames_removed": [1, 3], "frame_axis": 0},
        "files": ["movie.tif"],
        "extra": {
            "axes": "TYX",
            "original_shape": [5, 4, 4],
            "output_file": "movie_cleaned.tif",
        },
    }
    run.update(overrides)
    return run


@pytest.fixture
def movie(tmp_path):
    data = np.arange(5, dtype=np.uint16)[:, None, None] * np.ones(
        (1, 4, 4), dtype=np.uint16
    )
    tifffile.imwrite(str(tmp_path / "movie.tif"), data)
    return data


def _quiet(_message):
    pass


class TestRunsThatCannotBeReplayed:
    def test_a_run_of_another_widget_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not a frame removal run"):
            fr.replay_frame_removal_run(
                _run(widget="batch_processing"), input_folder=str(tmp_path)
            )

    def test_a_run_without_a_source_file_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no source file"):
            fr.replay_frame_removal_run(
                _run(files=[]), input_folder=str(tmp_path)
            )

    def test_a_run_without_a_recorded_output_is_refused(self, tmp_path):
        run = _run()
        del run["extra"]["output_file"]

        with pytest.raises(ValueError, match="no recorded output"):
            fr.replay_frame_removal_run(run, input_folder=str(tmp_path))

    def test_a_missing_source_is_reported_by_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="movie.tif"):
            fr.replay_frame_removal_run(
                _run(), input_folder=str(tmp_path), on_progress=_quiet
            )


def test_an_output_folder_redirects_the_result_but_keeps_its_name(
    tmp_path, movie
):
    out = tmp_path / "elsewhere"

    outcome = fr.replay_frame_removal_run(
        _run(),
        input_folder=str(tmp_path),
        output_folder=str(out),
        on_progress=_quiet,
    )

    assert outcome["status"] == "finished"
    assert outcome["output_file"] == str(out / "movie_cleaned.tif")
    np.testing.assert_array_equal(
        tifffile.imread(str(out / "movie_cleaned.tif")),
        np.delete(movie, [1, 3], axis=0),
    )
    assert not (tmp_path / "movie_cleaned.tif").exists()


def test_a_failed_write_is_reported_instead_of_raised(
    tmp_path, movie, monkeypatch
):
    def disk_full(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(fr, "save_cleaned_image", disk_full)
    messages = []

    outcome = fr.replay_frame_removal_run(
        _run(), input_folder=str(tmp_path), on_progress=messages.append
    )

    assert outcome["status"] == "failed"
    assert outcome["results"] == []
    assert outcome["errors"] == [
        {"file": str(tmp_path / "movie.tif"), "error": "No space left on device"}
    ]
    assert any("No space left" in m for m in messages)


def test_a_workflow_log_that_cannot_be_written_does_not_break_the_save(
    tmp_path, movie, monkeypatch, capsys
):
    monkeypatch.delenv("NAPARI_TMIDAS_NO_WORKFLOW_LOG", raising=False)

    class BrokenLog:
        def __init__(self, *_args, **_kwargs):
            raise PermissionError("read-only folder")

    monkeypatch.setattr(fr, "WorkflowLog", BrokenLog)

    outcome = fr.replay_frame_removal_run(
        _run(), input_folder=str(tmp_path), on_progress=_quiet
    )

    assert outcome["status"] == "finished"
    assert outcome["run_id"] is None
    assert outcome["log_path"] is None
    assert (tmp_path / "movie_cleaned.tif").exists()
    assert "could not record frame removal" in capsys.readouterr().out

"""
Tests for :mod:`napari_tmidas._workflow_replay`, the ``napari-tmidas-workflow``
CLI and the batch widget's "Load settings from log…" button.
"""

import json
import os
import shutil
import types

import numpy as np
import pytest
import tifffile

import napari_tmidas._file_selector as fs
import napari_tmidas._workflow_log as wl
import napari_tmidas._workflow_replay as wr
import napari_tmidas.processing_functions as pf
from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas._workflow_cli import main


@pytest.fixture(autouse=True)
def log_on(monkeypatch):
    monkeypatch.delenv(wl.DISABLE_ENV, raising=False)


@pytest.fixture
def registry(monkeypatch):
    saved = dict(BatchProcessingRegistry._processing_functions)
    BatchProcessingRegistry._processing_functions.clear()
    monkeypatch.setattr(pf, "discover_and_load_processing_functions", lambda: None)
    monkeypatch.setattr(fs, "discover_and_load_processing_functions", lambda: None)

    def add_offset(image, offset: int = 1, mode: str = "a"):
        return image + offset

    BatchProcessingRegistry.register(
        name="Add Offset",
        suffix="_off",
        parameters={
            "offset": {"type": int, "default": 1},
            "mode": {"type": str, "default": "a", "options": ["a", "b"]},
        },
    )(add_offset)
    yield BatchProcessingRegistry
    BatchProcessingRegistry._processing_functions.clear()
    BatchProcessingRegistry._processing_functions.update(saved)


def _workspace(tmp_path, n=2):
    ws = tmp_path / "ws"
    ws.mkdir()
    files = []
    for i in range(n):
        path = str(ws / f"img{i}.tif")
        tifffile.imwrite(path, np.full((4, 4), i, np.uint16), photometric="minisblack")
        files.append(path)
    return ws, files


def _record_run(ws, files, params=None, function="Add Offset", finish=True,
                output="results"):
    log = wl.WorkflowLog(str(ws))
    run_id = log.run_started(
        widget="batch_processing",
        function=function,
        params={"offset": 10, "mode": "b"} if params is None else params,
        files=files,
        input_suffix=".tif",
        output_folder=str(ws / output),
        output_suffix="_off",
        output_format="tiff",
        thread_count=2,
    )
    if finish:
        log.run_finished(run_id, status="finished")
    return run_id


def _quiet(_message):
    pass


# ---------------------------------------------------------------------------
# load_runs / find_logs
# ---------------------------------------------------------------------------
class TestLoadRuns:
    def test_merges_and_orders(self, tmp_path):
        path = tmp_path / wl.LOG_NAME
        lines = [
            {"record": "run_started", "run_id": "b", "time": "2026-10-01T12:00:00",
             "widget": "batch_processing", "function": "Second", "replayable": True},
            {"record": "run_started", "run_id": "a", "time": "2026-10-01T11:00:00",
             "widget": "batch_processing", "function": "First", "replayable": True},
            {"record": "run_finished", "run_id": "a", "time": "2026-10-01T11:05:00",
             "status": "finished", "results": []},
            {"record": "event", "run_id": "e", "time": "2026-10-01T13:00:00",
             "widget": "label_inspection", "replayable": False},
        ]
        path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        runs = wr.load_runs(str(tmp_path))
        assert [r["run_id"] for r in runs] == ["a", "b", "e"]
        assert runs[0]["status"] == "finished"
        assert runs[0]["time"] == "2026-10-01T11:00:00"
        assert runs[0]["finished"] == "2026-10-01T11:05:00"
        assert runs[1]["status"] == "started"
        assert [wr.is_replayable(r) for r in runs] == [True, True, False]

    def test_folder_resolution(self, tmp_path):
        top = tmp_path / "exp"
        (top / "a" / "b").mkdir(parents=True)
        (top / "x").mkdir()
        (top / "x" / wl.LOG_NAME).write_text("")
        (top / "a" / wl.LOG_NAME).write_text("")
        # A folder below a log uses that log ...
        assert wr.log_paths(str(top / "a" / "b")) == [str(top / "a" / wl.LOG_NAME)]
        # ... and a folder with none above collects every log below it.
        assert wr.log_paths(str(top)) == [
            str(top / "a" / wl.LOG_NAME), str(top / "x" / wl.LOG_NAME)
        ]
        assert wr.find_logs(str(top / "x")) == [str(top / "x" / wl.LOG_NAME)]
        with pytest.raises(FileNotFoundError):
            wr.log_paths(str(tmp_path / "missing"))

    def test_find_run_by_prefix(self):
        runs = [{"run_id": "20261001T1-aa"}, {"run_id": "20261001T1-ab"}]
        assert wr.find_run(runs, "20261001T1-aa")["run_id"] == "20261001T1-aa"
        assert wr.find_run(runs, "20261001T1-ab")["run_id"] == "20261001T1-ab"
        with pytest.raises(KeyError, match="ambiguous"):
            wr.find_run(runs, "20261001T1-a")
        with pytest.raises(KeyError):
            wr.find_run(runs, "nope")


# ---------------------------------------------------------------------------
# replay_batch_run
# ---------------------------------------------------------------------------
class TestReplay:
    def test_replay_produces_outputs_and_records(self, tmp_path, registry):
        ws, files = _workspace(tmp_path)
        original = _record_run(ws, files)
        (run,) = wr.load_runs(str(ws))

        outcome = wr.replay_batch_run(run, on_progress=_quiet)

        assert outcome["status"] == "finished"
        assert outcome["errors"] == []
        assert outcome["thread_count"] == 2
        for i, src in enumerate(files):
            out = ws / "results" / f"img{i}_off.tif"
            np.testing.assert_array_equal(tifffile.imread(out), tifffile.imread(src) + 10)

        runs = wr.load_runs(str(ws))
        assert len(runs) == 2
        replay = runs[1]
        assert replay["run_id"] == outcome["run_id"] != original
        assert replay["extra"] == {"replayed_from": original}
        assert replay["status"] == "finished"
        assert replay["params"] == {"offset": 10, "mode": "b"}
        assert len(replay["results"]) == 2

    def test_moved_workspace(self, tmp_path, registry):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        (run,) = wr.load_runs(str(ws))
        moved = tmp_path / "moved"
        shutil.copytree(ws, moved)

        outcome = wr.replay_batch_run(run, input_folder=str(moved), on_progress=_quiet)
        assert outcome["output_folder"] == str(moved / "results")
        assert (moved / "results" / "img0_off.tif").is_file()
        assert not (ws / "results").exists()

    def test_missing_file_warns_and_continues(self, tmp_path, registry):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        os.remove(files[1])
        (run,) = wr.load_runs(str(ws))
        messages = []
        outcome = wr.replay_batch_run(run, on_progress=messages.append)
        assert outcome["files"] == [files[0]]
        assert any("no longer exist" in m and "img1.tif" in m for m in messages)

    def test_all_missing_rebuilds_from_suffix(self, tmp_path, registry):
        ws, files = _workspace(tmp_path)
        _record_run(ws, [str(ws / "gone_a.tif"), str(ws / "gone_b.tif")])
        (run,) = wr.load_runs(str(ws))
        messages = []
        plan = wr.replay_batch_run(run, dry_run=True, on_progress=messages.append)
        assert plan["files"] == sorted(files)
        assert any("rebuilt the list" in m for m in messages)

    def test_parameter_validation(self, registry):
        parameters = BatchProcessingRegistry.get_function_info("Add Offset")["parameters"]
        messages = []
        params = wr.resolve_params(
            {"offset": "7", "removed": 3, "dimension_order": "ZYX"},
            parameters,
            messages.append,
        )
        assert params == {"offset": 7, "dimension_order": "ZYX", "mode": "a"}
        assert any("'removed' no longer exists" in m for m in messages)
        assert any("'mode' was not recorded" in m for m in messages)
        with pytest.raises(ValueError, match="'offset'"):
            wr.resolve_params({"offset": "seven"}, parameters, messages.append)

    def test_listed_option_is_accepted_as_is(self):
        parameters = {"level": {"type": int, "default": 1, "options": ["auto", 1, 2]}}
        assert wr.resolve_params({"level": "auto"}, parameters, _quiet) == {"level": "auto"}

    def test_dry_run_writes_nothing(self, tmp_path, registry):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        before = (ws / wl.LOG_NAME).read_text()
        (run,) = wr.load_runs(str(ws))
        plan = wr.replay_batch_run(run, dry_run=True, on_progress=_quiet)
        assert plan["dry_run"] is True
        assert plan["params"] == {"offset": 10, "mode": "b"}
        assert not (ws / "results").exists()
        assert (ws / wl.LOG_NAME).read_text() == before

    def test_unknown_function(self, tmp_path, registry):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files, function="Renamed Away")
        (run,) = wr.load_runs(str(ws))
        with pytest.raises(KeyError, match="Add Offset"):
            wr.replay_batch_run(run, on_progress=_quiet)

    def test_other_widgets_are_refused(self):
        with pytest.raises(ValueError, match="only batch processing"):
            wr.replay_batch_run({"widget": "roi_colocalization", "run_id": "x"})

    def test_replay_run_dispatch(self, tmp_path, registry):
        with pytest.raises(ValueError, match="not replayable"):
            wr.replay_run({"widget": "label_inspection", "replayable": False})
        with pytest.raises(ValueError, match="not replayable"):
            wr.replay_run({"widget": "no_such_widget", "replayable": True})
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        (run,) = wr.load_runs(str(ws))
        assert wr.replay_run(run, dry_run=True, on_progress=_quiet)["dry_run"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCli:
    def test_list_and_show(self, tmp_path, registry, capsys):
        ws, files = _workspace(tmp_path)
        run_id = _record_run(ws, files)
        _record_run(ws, files, finish=False)

        assert main(["list", str(ws)]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0].split() == ["RUN", "ID", "TIME", "FUNCTION", "STATUS", "FILES"]
        assert run_id in out and "finished" in out and "started" in out

        assert main(["show", str(ws), run_id]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["run_id"] == run_id
        assert shown["params"] == {"offset": 10, "mode": "b"}

        assert main(["show", str(ws), "no-such-run"]) == 1
        assert "No run" in capsys.readouterr().err

    def test_render(self, tmp_path, registry, capsys):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        os.remove(ws / wl.MARKDOWN_NAME)
        assert main(["render", str(ws)]) == 0
        assert (ws / wl.MARKDOWN_NAME).is_file()

    def test_replay_dry_run(self, tmp_path, registry, capsys):
        ws, files = _workspace(tmp_path)
        run_id = _record_run(ws, files)
        assert main(["replay", str(ws), "--run", run_id, "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "Add Offset on 2 file(s)" in out
        assert "offset = 10" in out
        assert not (ws / "results").exists()

    def test_replay_all_skips_non_replayable(self, tmp_path, registry, capsys):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        wl.WorkflowLog(str(ws)).event(widget="label_inspection", file=files[0])
        assert main(["replay", str(ws), "--all"]) == 0
        out = capsys.readouterr().out
        assert "label_inspection): not replayable" in out
        assert (ws / "results" / "img1_off.tif").is_file()

    def test_replay_needs_a_selection(self, tmp_path, registry, capsys):
        ws, files = _workspace(tmp_path)
        _record_run(ws, files)
        assert main(["replay", str(ws)]) == 2
        assert main(["replay", str(ws), "--all", "--input-folder", str(ws)]) == 2


# ---------------------------------------------------------------------------
# Batch widget: "Load settings from log…"
# ---------------------------------------------------------------------------
class _Viewer:
    def __init__(self):
        self.layers = []
        self.status = ""
        self.dims = types.SimpleNamespace(ndisplay=2)


class TestLoadSettingsButton:
    def test_applies_recorded_settings(self, tmp_path, qapp, registry, monkeypatch):
        def identity(image, sigma: float = 1.0):
            return image

        BatchProcessingRegistry.register(
            name="Another", parameters={"sigma": {"type": float, "default": 1.0}}
        )(identity)
        ws, files = _workspace(tmp_path)
        log = wl.WorkflowLog(str(ws))
        run_id = log.run_started(
            widget="batch_processing", function="Add Offset",
            params={"offset": 42, "mode": "b", "dimension_order": "TYX"},
            files=files, input_suffix=".tif",
            output_folder=str(ws / "deep" / "out"), output_format="zarr",
        )
        log.run_finished(run_id, status="finished")
        wl.WorkflowLog(str(ws)).event(widget="label_inspection", file=files[0])

        widget = fs.FileResultsWidget(_Viewer(), files, str(ws), ".tif")
        widget.processing_selector.setCurrentText("Another")
        assert widget.workflow_run_selector.isHidden()

        monkeypatch.setattr(
            fs.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(ws / wl.LOG_NAME), "")),
        )
        widget.load_log_button.click()

        selector = widget.workflow_run_selector
        assert not selector.isHidden()
        assert selector.count() == 2, "placeholder + the one batch run (no events)"
        selector.setCurrentIndex(1)
        selector.activated.emit(1)

        assert widget.processing_selector.currentText() == "Add Offset"
        values = widget.param_widget_instance.get_parameter_values()
        assert values == {"offset": 42, "mode": "b"}
        assert widget.dimension_order.currentText() == "TYX"
        assert widget.output_folder.text() == os.path.join("deep", "out")
        assert widget.output_format.currentText() == "Zarr"
        assert run_id in widget.viewer.status

    def test_blank_output_and_missing_function(self, tmp_path, qapp, registry):
        ws, files = _workspace(tmp_path)
        widget = fs.FileResultsWidget(_Viewer(), files, str(ws), ".tif")
        widget.output_folder.setText("old")
        assert widget.apply_workflow_run(
            {"run_id": "r1", "function": "Add Offset", "params": {},
             "output_folder": str(ws), "output_format": "tiff"}
        )
        assert widget.output_folder.text() == ""
        assert widget.dimension_order.currentText() == "Auto"

        assert not widget.apply_workflow_run({"run_id": "r2", "function": "Gone"})
        assert "not available" in widget.viewer.status

    def test_log_without_batch_runs(self, tmp_path, qapp, registry):
        ws, files = _workspace(tmp_path)
        wl.WorkflowLog(str(ws)).event(widget="label_inspection", file=files[0])
        widget = fs.FileResultsWidget(_Viewer(), files, str(ws), ".tif")
        assert widget.load_workflow_log(str(ws / wl.LOG_NAME)) == 0
        assert widget.workflow_run_selector.isHidden()
        assert widget.load_workflow_log(str(tmp_path / "nope.jsonl")) == 0
        assert "Could not read" in widget.viewer.status

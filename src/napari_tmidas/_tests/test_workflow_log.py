"""
Tests for :mod:`napari_tmidas._workflow_log` and its hooks in the batch
processing widget (``FileResultsWidget``).

The suite-wide conftest switches the log off; every test here switches it
back on through the ``log_on`` fixture.
"""

import json
import os
import sys
import types

import numpy as np
import pytest
import tifffile

import napari_tmidas._file_selector as fs
import napari_tmidas._workflow_log as wl
from napari_tmidas._registry import BatchProcessingRegistry


@pytest.fixture
def log_on(monkeypatch):
    monkeypatch.delenv(wl.DISABLE_ENV, raising=False)
    monkeypatch.delenv(wl.PATH_ENV, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))
    return path


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# resolve_log_path
# ---------------------------------------------------------------------------
class TestResolveLogPath:
    def test_creates_in_input_folder(self, tmp_path, home, log_on):
        ws = _mkdir(home / "ws")
        assert wl.resolve_log_path(str(ws)) == str(ws / wl.LOG_NAME)

    @pytest.mark.parametrize("levels", [1, 2, 3, 4])
    def test_finds_ancestor_log(self, home, log_on, levels):
        ws = _mkdir(home / "ws")
        (ws / wl.LOG_NAME).write_text("")
        deep = _mkdir(ws.joinpath(*[f"d{i}" for i in range(levels)]))
        assert wl.resolve_log_path(str(deep)) == str(ws / wl.LOG_NAME)

    def test_ignores_log_five_levels_up(self, home, log_on):
        ws = _mkdir(home / "ws")
        (ws / wl.LOG_NAME).write_text("")
        deep = _mkdir(ws.joinpath(*[f"d{i}" for i in range(5)]))
        assert wl.resolve_log_path(str(deep)) == str(deep / wl.LOG_NAME)

    def test_never_uses_home(self, home, log_on):
        (home / wl.LOG_NAME).write_text("")
        ws = _mkdir(home / "ws")
        assert wl.resolve_log_path(str(ws)) == str(ws / wl.LOG_NAME)

    def test_home_as_input_falls_back(self, home, log_on, capsys):
        out = _mkdir(home / "out")
        assert wl.resolve_log_path(str(home), str(out)) == str(out / wl.LOG_NAME)

        path = wl.resolve_log_path(str(home))
        assert path.startswith(str(home / ".napari-tmidas" / "workflow_logs"))
        assert path.endswith(".jsonl")
        assert "not writable" in capsys.readouterr().out

    def test_env_override_wins(self, tmp_path, home, log_on, monkeypatch):
        target = tmp_path / "elsewhere" / "my.jsonl"
        monkeypatch.setenv(wl.PATH_ENV, str(target))
        ws = _mkdir(home / "ws")
        (ws / wl.LOG_NAME).write_text("")
        assert wl.resolve_log_path(str(ws)) == str(target)

    @pytest.mark.skipif(
        sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0,
        reason="needs POSIX permissions and a non-root user",
    )
    def test_read_only_input_falls_back_to_output(self, home, log_on):
        ws = _mkdir(home / "ws")
        out = _mkdir(home / "out")
        ws.chmod(0o500)
        try:
            assert wl.resolve_log_path(str(ws), str(out)) == str(out / wl.LOG_NAME)
        finally:
            ws.chmod(0o700)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def _sample_func(image, sigma: float = 1.0):
    return image


class TestRecords:
    def test_round_trip(self, home, log_on):
        ws = _mkdir(home / "ws")
        log = wl.WorkflowLog(str(ws), str(ws / "out"))
        run_id = log.run_started(
            widget="batch_processing",
            function="Sample",
            func=_sample_func,
            params={
                "sigma": np.float32(1.5),
                "n": np.int64(3),
                "shape": (2, 3),
                "arr": np.zeros((2, 2), np.uint8),
                "_source_filepath": str(ws / "a.tif"),
            },
            param_schema=wl.param_schema(
                {"sigma": {"type": float, "default": 1.0}, "flag": {"type": bool}}
            ),
            files=[str(ws / "a.tif"), str(ws / "sub" / "b.tif")],
            input_suffix=".tif",
            output_folder=str(ws / "out"),
            output_suffix="_s",
            output_format="tiff",
            thread_count=2,
            dimension_order="YX",
        )
        assert run_id
        log.run_finished(
            run_id,
            status="finished",
            results=[
                {
                    "original_file": str(ws / "a.tif"),
                    "processed_files": [str(ws / "out" / "a_s0.tif")],
                    "elapsed_s": 0.5,
                }
            ],
            errors=[{"file": "/elsewhere/b.tif", "error": "boom"}],
        )

        records = wl.read_records(str(ws / wl.LOG_NAME))
        assert [r["record"] for r in records] == ["run_started", "run_finished"]
        start, end = records
        assert start["run_id"] == end["run_id"] == run_id
        assert start["schema_version"] == wl.SCHEMA_VERSION
        assert start["files"] == ["a.tif", os.path.join("sub", "b.tif")]
        assert start["params"] == {
            "sigma": 1.5,
            "n": 3,
            "shape": [2, 3],
            "arr": {"__array__": [2, 2], "dtype": "uint8"},
        }
        assert start["param_schema"] == {
            "sigma": {"type": "float", "default": 1.0},
            "flag": {"type": "bool", "default": None},
        }
        assert start["function_module"] == __name__
        assert start["input_folder"] == str(ws)
        assert start["output_folder"] == str(ws / "out")
        assert start["replayable"] is True
        assert end["status"] == "finished"
        assert isinstance(end["duration_s"], float)
        assert end["results"] == [
            {
                "original_file": "a.tif",
                "processed_files": [os.path.join("out", "a_s0.tif")],
                "elapsed_s": 0.5,
            }
        ]
        # Outside the input folder: kept absolute.
        assert end["errors"] == [{"file": "/elsewhere/b.tif", "error": "boom"}]
        assert (ws / wl.MARKDOWN_NAME).is_file()

    def test_malformed_line_is_skipped(self, tmp_path, capsys):
        path = tmp_path / wl.LOG_NAME
        path.write_text('{"record": "run_started", "run_id": "a"}\nnot json\n\n'
                        '{"record": "run_finished", "run_id": "a"}\n')
        records = wl.read_records(str(path))
        assert [r["record"] for r in records] == ["run_started", "run_finished"]
        assert "malformed line 2" in capsys.readouterr().out

    def test_disabled_writes_nothing(self, home, monkeypatch):
        monkeypatch.setenv(wl.DISABLE_ENV, "1")
        ws = _mkdir(home / "ws")
        log = wl.WorkflowLog(str(ws))
        assert wl.is_disabled()
        assert log.run_started(widget="batch_processing", function="x") is None
        log.run_finished("abc", status="finished")
        log.event(widget="label_inspection", file=str(ws / "a.tif"))
        assert os.listdir(ws) == []

    @pytest.mark.parametrize("value", ["", "0", "false"])
    def test_falsy_disable_values(self, monkeypatch, value):
        monkeypatch.setenv(wl.DISABLE_ENV, value)
        assert not wl.is_disabled()

    def test_write_failure_never_raises(self, home, log_on, monkeypatch, capsys):
        ws = _mkdir(home / "ws")
        log = wl.WorkflowLog(str(ws))

        def broken(path, record):
            raise OSError("disk full")

        monkeypatch.setattr(wl, "write_record", broken)
        assert log.run_started(widget="batch_processing", function="x") is None
        log.run_finished("abc", status="finished")
        assert "disk full" in capsys.readouterr().out

    def test_event_record(self, home, log_on):
        ws = _mkdir(home / "ws")
        wl.WorkflowLog(str(ws)).event(
            widget="label_inspection",
            file=str(ws / "a_labels.tif"),
            output=str(ws / "a_labels.tif"),
        )
        (record,) = wl.read_records(str(ws / wl.LOG_NAME))
        assert record["record"] == "event"
        assert record["file"] == "a_labels.tif"
        assert record["replayable"] is False
        assert "label_inspection event" in (ws / wl.MARKDOWN_NAME).read_text()


class TestEnvironment:
    def test_basic_fields(self):
        env = wl.collect_environment()
        assert env["napari_tmidas"]
        assert env["python"].count(".") == 2
        assert env["packages"]["numpy"] == np.__version__
        assert env["dedicated_env"] is None
        assert set(env["env_vars"]) == set(wl._ENV_VARS)

    def test_unknown_package_is_none(self, monkeypatch):
        monkeypatch.setattr(
            wl, "_PACKAGES", ("numpy", "definitely-not-installed-tmidas-pkg")
        )
        wl._package_versions.cache_clear()
        try:
            packages = wl.collect_environment()["packages"]
        finally:
            wl._package_versions.cache_clear()
        assert packages["definitely-not-installed-tmidas-pkg"] is None

    def test_dedicated_env(self, home):
        env_dir = _mkdir(home / ".napari-tmidas" / "envs" / "cellpose")
        (env_dir / "pyvenv.cfg").write_text("home = /x\nversion_info = 3.11.9\n")

        def segment(image):
            return image

        segment.__module__ = "napari_tmidas.processing_functions.cellpose_segmentation"
        assert wl.collect_environment(segment)["dedicated_env"] == {
            "name": "cellpose",
            "python": "3.11.9",
        }
        assert wl.collect_environment(_sample_func)["dedicated_env"] is None


class TestMarkdown:
    def test_lists_every_run(self, home, log_on):
        ws = _mkdir(home / "ws")
        log = wl.WorkflowLog(str(ws))
        first = log.run_started(widget="batch_processing", function="Alpha",
                                params={"k": "a|b"})
        log.run_finished(first, status="finished",
                         errors=[{"file": str(ws / "x.tif"), "error": "bad"}])
        log.run_started(widget="batch_processing", function="Beta")
        md = wl.render_markdown(str(ws / wl.LOG_NAME))
        text = open(md, encoding="utf-8").read()
        assert "## 1. Alpha" in text and "## 2. Beta" in text
        assert "**finished**" in text and "**started**" in text
        assert "a\\|b" in text
        assert "`x.tif`: bad" in text


# ---------------------------------------------------------------------------
# Batch widget hooks
# ---------------------------------------------------------------------------
class _Viewer:
    def __init__(self):
        self.layers = []
        self.status = ""
        self.dims = types.SimpleNamespace(ndisplay=2)


@pytest.fixture
def registry(monkeypatch):
    saved = dict(BatchProcessingRegistry._processing_functions)
    BatchProcessingRegistry._processing_functions.clear()
    monkeypatch.setattr(fs, "discover_and_load_processing_functions", lambda: None)
    monkeypatch.setattr(fs, "cancel_cellpose_processing", None)
    yield BatchProcessingRegistry
    BatchProcessingRegistry._processing_functions.clear()
    BatchProcessingRegistry._processing_functions.update(saved)


@pytest.fixture
def no_start(monkeypatch):
    """No QThread is started; tests drive ``worker.run()`` themselves."""
    monkeypatch.setattr(fs.ProcessingWorker, "start", lambda self: None)


def _widget(ws, func, name="Add Offset"):
    BatchProcessingRegistry.register(
        name=name,
        suffix="_off",
        parameters={"offset": {"type": int, "default": 5}},
    )(func)
    files = []
    for i in range(2):
        path = str(ws / f"img{i}.tif")
        tifffile.imwrite(path, np.full((4, 4), i, np.uint16),
                         photometric="minisblack")
        files.append(path)
    widget = fs.FileResultsWidget(_Viewer(), files, str(ws), ".tif")
    widget.output_folder.setText("results")
    return widget, files


def _records(ws):
    return wl.read_records(str(ws / wl.LOG_NAME))


class TestBatchWidgetHooks:
    def test_finished_run_records_settings_and_outputs(
        self, home, log_on, qapp, registry, no_start
    ):
        def add_offset(image, offset: int = 0):
            return image + offset

        ws = _mkdir(home / "ws")
        widget, files = _widget(ws, add_offset)
        widget.param_widget_instance.param_widgets["offset"].setValue(7)

        widget.start_batch_processing()
        start = _records(ws)[0]
        assert start["record"] == "run_started"
        assert widget.worker.param_values == {"offset": 7, "dimension_order": "YX"}
        assert start["params"] == widget.worker.param_values
        assert start["function"] == "Add Offset"
        assert start["dimension_order"] == "YX"
        assert start["files"] == ["img0.tif", "img1.tif"]
        assert start["output_folder"] == str(ws / "results")
        assert start["output_suffix"] == "_off"
        assert start["input_suffix"] == ".tif"
        assert start["param_schema"] == {"offset": {"type": "int", "default": 5}}

        widget.worker.run()

        records = _records(ws)
        assert len(records) == 2
        end = records[1]
        assert end["run_id"] == start["run_id"]
        assert end["status"] == "finished"
        assert end["errors"] == []
        assert sorted(r["original_file"] for r in end["results"]) == [
            "img0.tif", "img1.tif"
        ]
        for result in end["results"]:
            assert result["processed_file"].startswith("results" + os.sep)
            assert result["elapsed_s"] >= 0
        # A second processing_finished (e.g. after a stop) adds nothing.
        widget._on_worker_thread_finished()
        assert len(_records(ws)) == 2
        assert (ws / wl.MARKDOWN_NAME).is_file()

    def test_per_file_error_stays_finished(self, home, log_on, qapp, registry, no_start):
        def picky(image, offset: int = 0):
            if image.max() == 1:
                raise ValueError("cannot handle ones")
            return image

        ws = _mkdir(home / "ws")
        widget, _ = _widget(ws, picky)
        widget.start_batch_processing()
        widget.worker.run()

        end = _records(ws)[-1]
        assert end["status"] == "finished"
        assert end["errors"] == [{"file": "img1.tif", "error": "cannot handle ones"}]
        assert [r["original_file"] for r in end["results"]] == ["img0.tif"]

    def test_cancelled_run(self, home, log_on, qapp, registry, no_start, monkeypatch):
        def identity(image, offset: int = 0):
            return image

        ws = _mkdir(home / "ws")
        widget, _ = _widget(ws, identity)
        widget.start_batch_processing()
        worker = widget.worker
        monkeypatch.setattr(worker, "isRunning", lambda: True)
        monkeypatch.setattr(worker, "wait", lambda *a: True)

        widget.cancel_processing()
        assert len(_records(ws)) == 1, "the end record waits for the worker"
        worker.run()  # delivers the queued processing_finished

        records = _records(ws)
        assert [r["record"] for r in records] == ["run_started", "run_finished"]
        assert records[1]["status"] == "cancelled"

    def test_worker_crash_is_failed(self, home, log_on, qapp, registry, no_start):
        def crash(image, offset: int = 0):
            raise RuntimeError("not a per-file error type")

        ws = _mkdir(home / "ws")
        widget, _ = _widget(ws, crash)
        widget.start_batch_processing()
        with pytest.raises(RuntimeError):
            widget.worker.run()
        assert len(_records(ws)) == 1

        widget._on_worker_thread_finished()  # QThread.finished
        assert _records(ws)[-1]["status"] == "failed"

    def test_real_thread(self, home, log_on, qtbot, registry):
        def identity(image, offset: int = 0):
            return image

        ws = _mkdir(home / "ws")
        widget, _ = _widget(ws, identity)
        widget.start_batch_processing()
        qtbot.waitUntil(lambda: len(_records(ws)) == 2, timeout=10000)
        qtbot.wait(50)  # QThread.finished arrives after processing_finished
        records = _records(ws)
        assert len(records) == 2
        assert records[1]["status"] == "finished"
        assert len(records[1]["results"]) == 2

    def test_disabled_writes_nothing(self, home, qapp, registry, no_start):
        def identity(image, offset: int = 0):
            return image

        ws = _mkdir(home / "ws")
        widget, _ = _widget(ws, identity)
        widget.start_batch_processing()
        widget.worker.run()
        assert not (ws / wl.LOG_NAME).exists()
        assert widget.processed_files_info


def test_records_are_single_json_lines(home, log_on):
    ws = _mkdir(home / "ws")
    log = wl.WorkflowLog(str(ws))
    log.run_started(widget="batch_processing", function="x",
                    params={"text": "two\nlines"})
    lines = (ws / wl.LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["params"]["text"] == "two\nlines"

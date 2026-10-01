# napari_tmidas/_workflow_log.py
"""
Workflow (provenance) log for reproducible batch runs.

Every batch run appends two JSON Lines records to ``tmidas_workflow.jsonl``
in the workspace: ``run_started`` (function, parameters, files, folders,
versions) before the worker starts, and ``run_finished`` (status, outputs,
errors) when it ends. ``tmidas_workflow.md`` next to it is a human-readable
rendering of the same records; it is derived output and never parsed.

This is not Python ``logging``: the console ``print`` output is untouched.
Nothing here may break a run, so every public method of
:class:`WorkflowLog` catches its own errors, prints one line and returns.
Set ``NAPARI_TMIDAS_NO_WORKFLOW_LOG=1`` to switch the log off, or
``NAPARI_TMIDAS_WORKFLOW_LOG=/abs/path.jsonl`` to choose the file.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import secrets
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

LOG_NAME = "tmidas_workflow.jsonl"
MARKDOWN_NAME = "tmidas_workflow.md"
SCHEMA_VERSION = 1

DISABLE_ENV = "NAPARI_TMIDAS_NO_WORKFLOW_LOG"
PATH_ENV = "NAPARI_TMIDAS_WORKFLOW_LOG"

# How many parent folders above the input folder are searched for an
# existing log, so a chained workflow (each step writing into a subfolder of
# the previous one) keeps appending to one file.
MAX_ANCESTOR_LEVELS = 4

_PACKAGES = (
    "numpy",
    "scikit-image",
    "scipy",
    "tifffile",
    "zarr",
    "dask",
    "napari",
    "torch",
)
_ENV_VARS = ("TRACKASTRA_GPUS", "HOCT_GPUS", "CUDA_VISIBLE_DEVICES")

# Function modules that run in a dedicated environment, and the environment
# name their manager uses when it cannot be read from the manager itself.
_DEDICATED_ENVS = {
    "cellpose": "cellpose",
    "sam2": "sam2-env",
    "trackastra": "trackastra",
    "hoct": "hoct",
    "spotiflow": "spotiflow",
    "careamics": "careamics",
    "viscy": "viscy",
    "convpaint": "convpaint",
    "ultrack": "ultrack",
}

_lock = threading.RLock()


def is_disabled() -> bool:
    """True when ``NAPARI_TMIDAS_NO_WORKFLOW_LOG`` is set to a true value."""
    value = os.environ.get(DISABLE_ENV, "").strip().lower()
    return value not in ("", "0", "false", "no")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)


# ---------------------------------------------------------------------------
# Where the log lives
# ---------------------------------------------------------------------------
def _is_writable_dir(folder: str) -> bool:
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        return False
    return os.access(folder, os.W_OK | os.X_OK)


def _fallback_path(input_folder: str) -> str:
    digest = hashlib.sha1(input_folder.encode("utf-8")).hexdigest()
    return os.path.join(
        os.path.expanduser("~"), ".napari-tmidas", "workflow_logs", f"{digest}.jsonl"
    )


def resolve_log_path(input_folder: str, output_folder: Optional[str] = None) -> str:
    """
    Return the log file a run on ``input_folder`` should append to.

    ``NAPARI_TMIDAS_WORKFLOW_LOG`` wins outright. Otherwise the input folder
    and up to four of its parents are searched for an existing log; the first
    one found is used, else a new one is created in the input folder. The
    search never reaches the filesystem root or the home folder, and nothing
    is ever written there. An unwritable choice falls back to the output
    folder, then to ``~/.napari-tmidas/workflow_logs/``.
    """
    override = os.environ.get(PATH_ENV, "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))

    start = os.path.abspath(input_folder)
    home = os.path.abspath(os.path.expanduser("~"))

    def forbidden(folder: str) -> bool:
        return folder == home or os.path.dirname(folder) == folder

    folder = start
    for _ in range(MAX_ANCESTOR_LEVELS + 1):
        if forbidden(folder):
            break
        candidate = os.path.join(folder, LOG_NAME)
        if os.path.isfile(candidate):
            if os.access(candidate, os.W_OK) and os.access(folder, os.W_OK):
                return candidate
            break
        folder = os.path.dirname(folder)

    for folder in (start, output_folder):
        if not folder:
            continue
        folder = os.path.abspath(folder)
        if not forbidden(folder) and _is_writable_dir(folder):
            if folder != start:
                print(f"Workflow log: {start} is not writable, using {folder}")
            return os.path.join(folder, LOG_NAME)

    path = _fallback_path(start)
    print(f"Workflow log: {start} is not writable, recording to {path}")
    return path


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def _plugin_version() -> Dict[str, Optional[str]]:
    try:
        from napari_tmidas import _version

        return {
            "napari_tmidas": _version.__version__,
            "commit": getattr(_version, "__commit_id__", None),
        }
    except Exception:
        try:
            from importlib import metadata

            return {"napari_tmidas": metadata.version("napari-tmidas"), "commit": None}
        except Exception:
            return {"napari_tmidas": None, "commit": None}


def _dedicated_env(function) -> Optional[Dict[str, Optional[str]]]:
    """Name and Python version of the environment ``function`` runs in."""
    try:
        module_name = getattr(function, "__module__", "") or ""
        short = module_name.rsplit(".", 1)[-1].lower()
        key = next((k for k in _DEDICATED_ENVS if k in short), None)
        if key is None:
            return None
        sibling = f"{module_name.rsplit('.', 1)[0]}.{key}_env_manager"

        from napari_tmidas._env_manager import (
            BaseEnvironmentManager,
            _read_env_python_version,
            env_dir_for,
        )

        # Only modules that are already loaded: logging must not import a
        # heavy env manager the run itself did not need.
        env_name = _DEDICATED_ENVS[key]
        for name in (sibling, module_name):
            module = sys.modules.get(name)
            if module is None:
                continue
            managers = [
                v for v in vars(module).values()
                if isinstance(v, BaseEnvironmentManager)
            ]
            if managers:
                env_name = managers[0].env_name
                break
        return {
            "name": env_name,
            "python": _read_env_python_version(env_dir_for(env_name)),
        }
    except Exception:
        return None


def collect_environment(function=None) -> Dict[str, Any]:
    """Versions and settings that decide what a run produces."""
    env: Dict[str, Any] = dict(_plugin_version())
    env["python"] = platform.python_version()
    env["platform"] = platform.platform()
    env["packages"] = dict(_package_versions())
    env["dedicated_env"] = _dedicated_env(function) if function is not None else None
    env["env_vars"] = {name: os.environ.get(name) for name in _ENV_VARS}
    return env


# ---------------------------------------------------------------------------
# Reading and writing records
# ---------------------------------------------------------------------------
def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return {"__array__": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, type):
        return obj.__name__
    return repr(obj)


def write_record(path: str, record: Dict[str, Any]) -> None:
    """Append one record as a line of JSON."""
    line = json.dumps(record, default=_json_default, ensure_ascii=False)
    with _lock:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def read_records(path: str) -> List[Dict[str, Any]]:
    """All records in ``path``; malformed lines are reported and skipped."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                print(f"Workflow log: skipping malformed line {number} in {path}")
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def merge_runs(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One dict per run (``run_started`` updated with ``run_finished``) and one
    per event, in log order. A run without a finish record has status
    ``"started"``.
    """
    runs: Dict[str, Dict[str, Any]] = {}
    for record in records:
        kind = record.get("record")
        run_id = record.get("run_id") or f"line-{len(runs)}"
        if kind == "run_started":
            merged = {"status": "started", **record}
            merged["started"] = record.get("time")
            runs[run_id] = merged
        elif kind == "run_finished":
            merged = runs.setdefault(run_id, {"run_id": run_id})
            merged.update({k: v for k, v in record.items() if k != "record"})
            merged["finished"] = record.get("time")
            merged["time"] = merged.get("started", record.get("time"))
        elif kind == "event":
            runs[run_id] = dict(record)
    return list(runs.values())


def _markdown_value(value) -> str:
    text = json.dumps(value, default=_json_default) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(path: str) -> str:
    """Rewrite ``tmidas_workflow.md`` next to ``path``; return its path."""
    lines = ["# napari-tmidas workflow log", "",
             f"Rendered from `{os.path.basename(path)}`. Do not edit; "
             "it is rewritten after every run.", ""]
    for n, run in enumerate(merge_runs(read_records(path)), 1):
        when = run.get("time", "?")
        if run.get("record") == "event":
            lines += [f"## {n}. {run.get('widget')} event — {when}", "",
                      f"- File: `{run.get('file')}`", f"- Output: `{run.get('output')}`",
                      "- Manual edit, not replayable", ""]
            continue
        lines += [f"## {n}. {run.get('function', run.get('widget'))} — {when}", ""]
        status = run.get("status", "?")
        duration = run.get("duration_s")
        lines.append(f"- Status: **{status}**" + (f" in {duration:.1f} s" if duration else ""))
        lines.append(f"- Run id: `{run.get('run_id')}`")
        extra = run.get("extra") or {}
        if extra.get("replayed_from"):
            lines.append(f"- Replay of `{extra['replayed_from']}`")
        lines.append(f"- Input: `{run.get('input_folder')}` ({len(run.get('files') or [])} files)")
        if run.get("output_folder"):
            lines.append(f"- Output: `{run.get('output_folder')}` "
                         f"({run.get('output_format', '')})")
        if run.get("dimension_order"):
            lines.append(f"- Dimension order: {run['dimension_order']}")
        lines.append(f"- napari-tmidas {(run.get('environment') or {}).get('napari_tmidas')}")
        if not run.get("replayable", True):
            lines.append("- Not replayable")
        if "results" in run:
            lines.append(f"- Outputs written for {len(run['results'])} input(s)")
        params = run.get("params") or {}
        if params:
            lines += ["", "| Parameter | Value |", "|---|---|"]
            lines += [f"| {k} | {_markdown_value(v)} |" for k, v in params.items()]
        errors = run.get("errors") or []
        if errors:
            lines += ["", f"Errors ({len(errors)}):"]
            lines += [f"- `{e.get('file')}`: {_markdown_value(e.get('error'))[:300]}"
                      for e in errors[:5]]
            if len(errors) > 5:
                lines.append(f"- ... and {len(errors) - 5} more")
        lines.append("")
    md_path = os.path.join(os.path.dirname(path), MARKDOWN_NAME)
    with _lock, open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# Helpers shared by the widgets
# ---------------------------------------------------------------------------
def param_schema(parameters: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """``{name: {"type", "default"}}`` from a registry ``parameters`` dict."""
    schema = {}
    for name, info in (parameters or {}).items():
        ptype = info.get("type")
        schema[name] = {
            "type": getattr(ptype, "__name__", None if ptype is None else str(ptype)),
            "default": info.get("default"),
        }
    return schema


def relative_path(path, folder: Optional[str]):
    """``path`` relative to ``folder`` when inside it, else absolute."""
    if not path or not isinstance(path, (str, os.PathLike)):
        return path
    path = os.path.abspath(os.fspath(path))
    if folder:
        folder = os.path.abspath(folder)
        try:
            if os.path.commonpath([path, folder]) == folder:
                return os.path.relpath(path, folder)
        except ValueError:  # different drives on Windows
            pass
    return path


def _relative_result(result: Dict[str, Any], folder: Optional[str]) -> Dict[str, Any]:
    out = dict(result)
    for key in ("original_file", "processed_file", "file"):
        if key in out:
            out[key] = relative_path(out[key], folder)
    if isinstance(out.get("processed_files"), (list, tuple)):
        out["processed_files"] = [relative_path(p, folder) for p in out["processed_files"]]
    return out


class WorkflowLog:
    """
    One workspace's log, written to by one widget.

    Never raises: every public method catches ``Exception``, prints one line
    and returns ``None``.
    """

    def __init__(self, input_folder, output_folder=None):
        self.input_folder = os.path.abspath(os.fspath(input_folder)) if input_folder else None
        self.path: Optional[str] = None
        self._t0: Dict[str, float] = {}
        if is_disabled() or not self.input_folder:
            return
        try:
            self.path = resolve_log_path(self.input_folder, output_folder)
        except Exception as exc:
            print(f"Workflow log disabled for this run: {exc}")

    @property
    def enabled(self) -> bool:
        return self.path is not None and not is_disabled()

    def _write(self, record: Dict[str, Any]) -> bool:
        try:
            write_record(self.path, record)
            return True
        except Exception as exc:
            print(f"Workflow log: could not write to {self.path}: {exc}")
            return False

    def run_started(
        self,
        *,
        widget: str,
        function: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        param_schema: Optional[Dict[str, Any]] = None,
        files: Iterable = (),
        input_suffix: Optional[str] = None,
        output_folder: Optional[str] = None,
        output_suffix: Optional[str] = None,
        output_format: Optional[str] = None,
        thread_count: Optional[int] = None,
        dimension_order: Optional[str] = None,
        replayable: bool = True,
        func=None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Write the ``run_started`` record; return the run id."""
        if not self.enabled:
            return None
        try:
            run_id = new_run_id()
            record = {
                "schema_version": SCHEMA_VERSION,
                "record": "run_started",
                "run_id": run_id,
                "time": _now(),
                "widget": widget,
                "function": function,
                "function_module": getattr(func, "__module__", None),
                # The worker adds per-file "_source_filepath"/"_output_*"
                # keys; they are not settings.
                "params": {
                    k: v for k, v in (params or {}).items() if not str(k).startswith("_")
                },
                "param_schema": param_schema or {},
                "dimension_order": dimension_order,
                "input_folder": self.input_folder,
                "input_suffix": input_suffix,
                "files": [relative_path(f, self.input_folder) for f in files],
                "output_folder": (
                    os.path.abspath(output_folder) if output_folder else None
                ),
                "output_suffix": output_suffix,
                "output_format": output_format,
                "thread_count": thread_count,
                "environment": collect_environment(func),
                "replayable": bool(replayable),
            }
            if extra:
                record["extra"] = dict(extra)
            if not self._write(record):
                return None
            self._t0[run_id] = time.monotonic()
            return run_id
        except Exception as exc:
            print(f"Workflow log: could not record run start: {exc}")
            return None

    def run_finished(
        self,
        run_id: Optional[str],
        *,
        status: str,
        results: Iterable[Dict[str, Any]] = (),
        errors: Iterable[Dict[str, Any]] = (),
    ) -> None:
        """Write the ``run_finished`` record and re-render the markdown."""
        if not run_id or not self.enabled:
            return
        try:
            t0 = self._t0.pop(run_id, None)
            record = {
                "schema_version": SCHEMA_VERSION,
                "record": "run_finished",
                "run_id": run_id,
                "time": _now(),
                "status": status,
                "duration_s": None if t0 is None else round(time.monotonic() - t0, 3),
                "results": [
                    _relative_result(r, self.input_folder)
                    for r in results if isinstance(r, dict)
                ],
                "errors": [
                    _relative_result(e, self.input_folder)
                    for e in errors if isinstance(e, dict)
                ],
            }
            if self._write(record):
                render_markdown(self.path)
        except Exception as exc:
            print(f"Workflow log: could not record run end: {exc}")

    def event(
        self,
        *,
        widget: str,
        file=None,
        output=None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a manual, non-replayable action such as a saved edit."""
        if not self.enabled:
            return
        try:
            record = {
                "schema_version": SCHEMA_VERSION,
                "record": "event",
                "run_id": new_run_id(),
                "time": _now(),
                "widget": widget,
                "input_folder": self.input_folder,
                "file": relative_path(file, self.input_folder),
                "output": relative_path(output, self.input_folder),
                "replayable": False,
            }
            if extra:
                record["extra"] = dict(extra)
            if self._write(record):
                render_markdown(self.path)
        except Exception as exc:
            print(f"Workflow log: could not record event: {exc}")

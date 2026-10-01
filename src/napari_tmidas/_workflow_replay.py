# napari_tmidas/_workflow_replay.py
"""
Repeat batch runs recorded in a workflow log (see ``_workflow_log``).

A replay runs the recorded function with the recorded parameters on the
recorded files, headless, through the same ``ProcessingWorker`` the batch
widget uses, and records itself in the log with ``extra.replayed_from``.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from napari_tmidas import _workflow_log as wl

BATCH_WIDGET = "batch_processing"


# ---------------------------------------------------------------------------
# Finding and reading logs
# ---------------------------------------------------------------------------
def find_logs(root: str) -> List[str]:
    """Every workflow log below ``root``, for workspaces with several."""
    found = []
    for folder, dirs, files in os.walk(root):
        # A .zarr store is one image, never a workspace.
        dirs[:] = sorted(d for d in dirs if not d.endswith(".zarr"))
        if wl.LOG_NAME in files:
            found.append(os.path.join(folder, wl.LOG_NAME))
    return found


def log_paths(path_or_folder: str) -> List[str]:
    """
    The log file(s) ``path_or_folder`` refers to: the file itself, the log
    a run in that folder would append to (it or an ancestor's), or else
    every log below it.
    """
    path = os.path.abspath(os.path.expanduser(path_or_folder))
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No such file or folder: {path_or_folder}")
    folder = path
    for _ in range(wl.MAX_ANCESTOR_LEVELS + 1):
        candidate = os.path.join(folder, wl.LOG_NAME)
        if os.path.isfile(candidate):
            return [candidate]
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return find_logs(path)


def load_runs(path_or_folder: str) -> List[Dict[str, Any]]:
    """
    One merged record per run (``run_started`` updated with its
    ``run_finished``) and per event, oldest first. A run that never
    finished has ``status: "started"``.
    """
    runs = []
    for path in log_paths(path_or_folder):
        for run in wl.merge_runs(wl.read_records(path)):
            run["log_path"] = path
            runs.append(run)
    runs.sort(key=lambda r: str(r.get("time") or ""))
    return runs


def find_run(runs: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    """The run whose id is, or uniquely starts with, ``run_id``."""
    matches = [r for r in runs if str(r.get("run_id", "")).startswith(run_id)]
    exact = [r for r in matches if r.get("run_id") == run_id]
    if exact:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No run {run_id!r} in the log")
    raise KeyError(
        f"Run id {run_id!r} is ambiguous: "
        + ", ".join(r["run_id"] for r in matches)
    )


# Widget -> "module:function" that repeats its runs. Every replayer takes
# (run, *, input_folder=None, output_folder=None, dry_run=False,
# on_progress=print), records its own run with extra.replayed_from, and
# returns a dict with at least "status" and "errors".
REPLAYERS = {
    BATCH_WIDGET: "napari_tmidas._workflow_replay:replay_batch_run",
    "frame_removal": "napari_tmidas._frame_removal:replay_frame_removal_run",
    "file_conversion": "napari_tmidas._file_conversion:replay_conversion_run",
    "roi_colocalization": "napari_tmidas._roi_colocalization:replay_colocalization_run",
}


def is_replayable(run: Dict[str, Any]) -> bool:
    return (
        run.get("record") != "event"
        and bool(run.get("replayable", False))
        and run.get("widget") in REPLAYERS
    )


def replay_run(run: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Repeat ``run`` with the replayer of the widget that recorded it."""
    target = REPLAYERS.get(run.get("widget"))
    if target is None or not run.get("replayable", False):
        raise ValueError(
            f"Run {run.get('run_id')} ({run.get('widget')}) is not replayable"
        )
    import importlib

    module_name, func_name = target.split(":")
    return getattr(importlib.import_module(module_name), func_name)(run, **kwargs)


# ---------------------------------------------------------------------------
# Resolving a recorded run against the current installation and disk
# ---------------------------------------------------------------------------
def resolve_params(
    recorded: Dict[str, Any],
    parameters: Dict[str, Dict[str, Any]],
    on_progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Check recorded parameter values against the function's current
    parameters: unknown names are dropped with a warning, missing ones take
    the current default, and a value that no longer converts raises
    ``ValueError``.
    """
    from napari_tmidas._file_selector import convert_parameter_value

    params: Dict[str, Any] = {}
    for name, value in (recorded or {}).items():
        if name == "dimension_order":
            # Set by the widget's Dimension Order selector, not a parameter.
            params[name] = value
            continue
        if name not in parameters:
            on_progress(
                f"Warning: parameter {name!r} no longer exists; ignoring "
                f"the recorded value {value!r}"
            )
            continue
        try:
            params[name] = convert_parameter_value(
                parameters[name], value, strict=True
            )
        except ValueError as exc:
            raise ValueError(f"Parameter {name!r}: {exc}") from exc
    for name, info in parameters.items():
        if name not in params:
            params[name] = info.get("default")
            on_progress(
                f"Parameter {name!r} was not recorded; using the current "
                f"default {params[name]!r}"
            )
    return params


def resolve_files(
    run: Dict[str, Any],
    input_folder: str,
    on_progress: Callable[[str], None] = print,
) -> List[str]:
    """
    The recorded input files that still exist under ``input_folder``. When
    none do, the folder is scanned with the recorded suffix instead.
    """
    from napari_tmidas._file_selector import find_matching_files

    recorded = [
        f if os.path.isabs(f) else os.path.join(input_folder, f)
        for f in run.get("files") or []
    ]
    present = [f for f in recorded if os.path.exists(f)]
    missing = [f for f in recorded if not os.path.exists(f)]
    if present:
        if missing:
            on_progress(
                f"Warning: {len(missing)} recorded file(s) no longer exist "
                "and are skipped: "
                + ", ".join(os.path.basename(f) for f in missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        return present
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    files = sorted(find_matching_files(input_folder, run.get("input_suffix") or ""))
    on_progress(
        f"None of the {len(recorded)} recorded files exist; rebuilt the list "
        f"from {input_folder} with suffix {run.get('input_suffix')!r}: "
        f"{len(files)} file(s)"
    )
    return files


def resolve_output_folder(run, input_folder, output_folder):
    if output_folder:
        return os.path.abspath(output_folder)
    recorded_out = run.get("output_folder")
    recorded_in = run.get("input_folder")
    if not recorded_out:
        return input_folder
    # The batch widget always writes below the input folder; keep that
    # relation when the input folder has moved.
    if recorded_in:
        rel = os.path.relpath(recorded_out, recorded_in)
        if rel == "." or not rel.startswith(os.pardir):
            return os.path.normpath(os.path.join(input_folder, rel))
    return recorded_out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay_batch_run(
    run: Dict[str, Any],
    *,
    input_folder: Optional[str] = None,
    output_folder: Optional[str] = None,
    dry_run: bool = False,
    on_progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Run a recorded batch-processing run again.

    ``input_folder`` / ``output_folder`` override the recorded folders (an
    output folder that was inside the input folder moves with it). With
    ``dry_run`` the run is resolved and described but nothing is written.
    Returns the plan (dry run) or the new run's outcome.
    """
    from napari_tmidas._file_selector import (
        ProcessingWorker,
        resolve_thread_count,
    )
    from napari_tmidas._registry import BatchProcessingRegistry
    from napari_tmidas.processing_functions import (
        discover_and_load_processing_functions,
    )

    if run.get("widget") != BATCH_WIDGET:
        raise ValueError(
            f"Run {run.get('run_id')} comes from {run.get('widget')!r}; only "
            "batch processing runs can be replayed"
        )

    discover_and_load_processing_functions()
    name = run.get("function")
    info = BatchProcessingRegistry.get_function_info(name)
    if info is None:
        raise KeyError(
            f"Processing function {name!r} is not available. Available: "
            + ", ".join(BatchProcessingRegistry.list_functions())
        )

    params = resolve_params(run.get("params") or {}, info.get("parameters") or {}, on_progress)
    in_folder = os.path.abspath(input_folder or run.get("input_folder") or ".")
    files = resolve_files(run, in_folder, on_progress)
    out_folder = resolve_output_folder(run, in_folder, output_folder)
    func = info["func"]
    threads, thread_note = resolve_thread_count(func, params, run.get("thread_count") or 1)
    output_format = run.get("output_format") or "tiff"

    plan = {
        "run_id": None,
        "replayed_from": run.get("run_id"),
        "function": name,
        "params": params,
        "input_folder": in_folder,
        "files": files,
        "output_folder": out_folder,
        "output_suffix": info["suffix"],
        "output_format": output_format,
        "thread_count": threads,
    }
    on_progress(
        f"Replay of {run.get('run_id')}: {name} on {len(files)} file(s) "
        f"from {in_folder} -> {out_folder} ({output_format}); {thread_note}"
    )
    if dry_run:
        for key, value in params.items():
            on_progress(f"  {key} = {value!r}")
        return {**plan, "dry_run": True}
    if not files:
        raise FileNotFoundError(f"No input files to process in {in_folder}")

    if getattr(func, "__name__", "") == "create_grid_overlay":
        from napari_tmidas.processing_functions.grid_view_overlay import (
            reset_grid_cache,
        )

        reset_grid_cache()

    os.makedirs(out_folder, exist_ok=True)
    worker = ProcessingWorker(
        files,
        func,
        dict(params),
        out_folder,
        run.get("input_suffix") or "",
        info["suffix"],
        output_format=output_format,
    )
    worker.thread_count = threads

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def on_result(result):
        results.append(result)
        on_progress(f"  done: {os.path.basename(result.get('original_file', ''))}")

    def on_error(filepath, message):
        errors.append({"file": filepath, "error": message})
        on_progress(f"  error: {os.path.basename(filepath)}: {message}")

    worker.file_processed.connect(on_result)
    worker.error_occurred.connect(on_error)

    log = wl.WorkflowLog(in_folder, out_folder)
    run_id = log.run_started(
        widget=BATCH_WIDGET,
        function=name,
        func=func,
        params=params,
        param_schema=wl.param_schema(info.get("parameters")),
        files=files,
        input_suffix=run.get("input_suffix"),
        output_folder=out_folder,
        output_suffix=info["suffix"],
        output_format=output_format,
        thread_count=threads,
        dimension_order=params.get("dimension_order"),
        extra={"replayed_from": run.get("run_id")},
    )
    status = "failed"
    try:
        # Synchronously, on this thread: the worker's signals are delivered
        # directly, no event loop needed.
        worker.run()
        status = "finished"
    finally:
        log.run_finished(run_id, status=status, results=results, errors=errors)

    on_progress(
        f"Replay finished: {len(results)} output(s), {len(errors)} error(s)"
    )
    return {
        **plan,
        "run_id": run_id,
        "status": status,
        "results": results,
        "errors": errors,
        "log_path": log.path,
    }

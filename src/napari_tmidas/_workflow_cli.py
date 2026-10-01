# napari_tmidas/_workflow_cli.py
"""
``napari-tmidas-workflow``: list, show, render and replay workflow logs.

    napari-tmidas-workflow list   <workspace>
    napari-tmidas-workflow show   <workspace> <run_id>
    napari-tmidas-workflow render <workspace>
    napari-tmidas-workflow replay <workspace> [--run ID | --all]
                                  [--input-folder P] [--output-folder P] [--dry-run]

``<workspace>`` is a ``tmidas_workflow.jsonl`` file or a folder holding one
(in it, a parent, or below it). On a machine without a display, set
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from napari_tmidas import _workflow_log as wl
from napari_tmidas import _workflow_replay as wr


def _ensure_qt_core_app():
    # The worker is a QThread; its signals need a Qt application object.
    try:
        from qtpy.QtCore import QCoreApplication

        if QCoreApplication.instance() is None:
            return QCoreApplication(sys.argv[:1])
    except Exception:
        pass
    return None


def _cmd_list(args) -> int:
    runs = wr.load_runs(args.workspace)
    if not runs:
        print("No runs recorded.")
        return 0
    rows = [("RUN ID", "TIME", "FUNCTION", "STATUS", "FILES")]
    for run in runs:
        if run.get("record") == "event":
            rows.append((run.get("run_id", ""), run.get("time", ""),
                         f"[{run.get('widget')}] {run.get('file') or ''}", "event", ""))
            continue
        name = run.get("function") or f"[{run.get('widget')}]"
        if not wr.is_replayable(run):
            name += " (not replayable)"
        rows.append((run.get("run_id", ""), run.get("time", ""), name,
                     run.get("status", ""), str(len(run.get("files") or []))))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    return 0


def _cmd_show(args) -> int:
    run = wr.find_run(wr.load_runs(args.workspace), args.run_id)
    print(json.dumps(run, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_render(args) -> int:
    for path in wr.log_paths(args.workspace):
        print(f"Wrote {wl.render_markdown(path)}")
    return 0


def _cmd_replay(args) -> int:
    if args.all and (args.input_folder or args.output_folder):
        print("--all replays each run in its own recorded folders; "
              "--input-folder/--output-folder need --run", file=sys.stderr)
        return 2
    runs = wr.load_runs(args.workspace)
    if args.all:
        selected = []
        for run in runs:
            if wr.is_replayable(run):
                selected.append(run)
            else:
                print(f"Skipping {run.get('run_id')} "
                      f"({run.get('function') or run.get('widget')}): not replayable")
    else:
        if not args.run:
            print("Give --run <run id> or --all", file=sys.stderr)
            return 2
        selected = [wr.find_run(runs, args.run)]

    _ensure_qt_core_app()
    failed = 0
    for run in selected:
        try:
            outcome = wr.replay_run(
                run,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"Replay of {run.get('run_id')} failed: {exc}", file=sys.stderr)
            failed += 1
            continue
        if outcome.get("errors"):
            failed += 1
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="napari-tmidas-workflow",
        description="Inspect and repeat runs recorded in tmidas_workflow.jsonl.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="table of recorded runs")
    p.add_argument("workspace")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show", help="one run's full record as JSON")
    p.add_argument("workspace")
    p.add_argument("run_id", help="run id, or a unique prefix of one")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("render", help="rewrite tmidas_workflow.md")
    p.add_argument("workspace")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("replay", help="run recorded batch runs again")
    p.add_argument("workspace")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--run", help="run id, or a unique prefix of one")
    which.add_argument("--all", action="store_true",
                       help="every replayable run, oldest first, in its own folders")
    p.add_argument("--input-folder", help="read the input files from here instead")
    p.add_argument("--output-folder", help="write the outputs here instead")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would run, write nothing")
    p.set_defaults(func=_cmd_replay)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

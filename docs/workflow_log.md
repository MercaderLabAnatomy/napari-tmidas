# Workflow Log

## Overview

Every batch processing run writes a record of what was done into the
workspace: which function ran, with which parameters, on which files, into
which output folder, and with which software versions. Run a chain of steps
(convert, split channels, segment, track) and the log is a step-by-step
account of how every output was made. Months later, or on another computer,
it tells you exactly how to repeat the analysis.

The log is written automatically. There is nothing to switch on.

## Where the log is written

Two files appear next to your data:

- **`tmidas_workflow.jsonl`**: the record itself, one line per entry. This
  is the file to keep with your data and to attach to a methods section or a
  data deposit.
- **`tmidas_workflow.md`**: the same content as a readable page (open it in
  any text editor or Markdown viewer). It is rewritten after every run, so
  do not edit it by hand.

The log goes into the **input folder** of the run. Because the batch widget
always writes its output into a subfolder of the input folder, a chained
workflow moves *down* the folder tree. To keep the whole workflow in one
file, each run first looks for an existing `tmidas_workflow.jsonl` in its
input folder and up to four folders above it, and appends to the first one
it finds:

```
experiment/
├── tmidas_workflow.jsonl      <- every step below is recorded here
├── tmidas_workflow.md
├── raw/                       <- step 1: input
│   └── split/                 <- step 1: output, step 2: input
│       └── labels/            <- step 2: output
```

The search never goes up to your home folder or the root of a drive, and the
log is never written there. If the input folder is read-only, the log is
written to the run's output folder instead, or as a last resort to
`~/.napari-tmidas/workflow_logs/`; the console says where.

## What is recorded

Each run adds two entries that share a run id: one when the run **starts**
and one when it **ends**. If napari crashes mid-run, the start entry is
still there, and it is the one that holds everything needed to repeat the
run.

The start entry holds:

- the processing function and every parameter value, exactly as passed to
  the function, plus the parameter defaults at the time (so you can see
  later whether a function changed)
- the **Dimension Order** setting
- the input folder, the file suffix filter and the full list of input files
- the output folder, output suffix, output format (TIFF/Zarr) and thread
  count
- the napari-tmidas version and commit, the Python version, the operating
  system, and the versions of numpy, scikit-image, scipy, tifffile, zarr,
  dask, napari and torch
- for AI methods that run in their own environment (Cellpose, SAM2,
  Trackastra, HOCT, Spotiflow, CAREamics, VisCy, Convpaint, Ultrack): the
  name of that environment and its Python version
- the GPU selection variables `TRACKASTRA_GPUS`, `HOCT_GPUS` and
  `CUDA_VISIBLE_DEVICES`

The end entry holds:

- the status: `finished`, `cancelled`, or `failed` (the run stopped on an
  unexpected error before it could finish)
- how long the run took, and for each input file the output file(s) it
  produced and how long that file took
- the files that failed, with their error messages. A run in which some
  files failed is still `finished`; the failures are listed.

File paths are stored relative to the input folder, so the log stays valid
when you move or copy the whole experiment folder.

## What a log cannot reproduce

A log describes **what was done**. Repeating it gives the same method on
the same data, which is not always the same pixels:

- **Manual edits are not repeatable.** Corrections made in Batch Label
  Inspection, objects chosen in Batch Crop Anything, and anything you did in
  the napari viewer itself are clicks, not settings.
- **AI results can differ slightly between runs.** Cellpose, SAM2,
  Trackastra, HOCT, Spotiflow, CAREamics, VisCy, Convpaint and Ultrack are
  not seeded, and the log records the name and Python version of their
  environment, not every package version installed in it. Results can also
  differ between GPU models and between GPU and CPU.
- **Files that changed after the run.** The log records file names, not
  file contents.

## Switching the log off or moving it

Set an environment variable before starting napari:

| Variable | Effect |
|---|---|
| `NAPARI_TMIDAS_NO_WORKFLOW_LOG=1` | No log is written. |
| `NAPARI_TMIDAS_WORKFLOW_LOG=/path/to/my_log.jsonl` | Every run is recorded in this one file, wherever its data are. |

Writing the log can never stop or break a run: if the file cannot be
written, the console prints one line and the run continues.

## File format

For scripts that read the log: each line of `tmidas_workflow.jsonl` is one
JSON object with `"schema_version": 1` and a `"record"` field that is
`"run_started"`, `"run_finished"` or `"event"`. Entries of one run share a
`"run_id"`. Lines are only ever appended.

```json
{"schema_version": 1, "record": "run_started", "run_id": "20261001T143012-9f3a1c",
 "time": "2026-10-01T14:30:12+02:00", "widget": "batch_processing",
 "function": "Split Color Channels", "params": {"num_channels": 3, "dimension_order": "TZYX"},
 "input_folder": "/data/exp1", "files": ["a.tif", "b.tif"],
 "output_folder": "/data/exp1/split", "output_format": "tiff", "...": "..."}
{"schema_version": 1, "record": "run_finished", "run_id": "20261001T143012-9f3a1c",
 "status": "finished", "duration_s": 83.2,
 "results": [{"original_file": "a.tif", "processed_files": ["split/a_ch0.tif", "split/a_ch1.tif"], "elapsed_s": 41.0}],
 "errors": [{"file": "b.tif", "error": "..."}]}
```

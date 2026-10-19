# Track Statistics (QC)

## Overview

Tracking can fail without any error message: tracks break apart, objects
swap identities, or a whole movie ends up as hundreds of one-frame tracks.
Two tools help you catch this:

- **Tracks in the results table**: open a tracking result from the batch
  results table and it appears as a Tracks layer on top of the labels. You
  see each object's path through time, not only labels that change colour.
- **Track Statistics (QC)**: a batch function that summarises every tracked
  file in a folder as one row of numbers, so you can spot the bad files
  without opening each one.

Both read the tracked label image (TYX or TZYX, time first), so they work
for Trackastra (`_tracked`), HOCT (`_hoct_tracked`) and Ultrack
(`_ultrack`) results, and for tracked labels you have edited since.

## Tracks in the Results Table

Double-click a tracked file in the batch results table. The labels load as
usual, and a layer named `Tracks: <file>` is added on top of them, at the
same pixel size. The positions come from, in this order:

1. Ultrack's `<file>_ultrack_tracks.csv`, next to the file. This is the only
   source that knows about divisions: daughter tracks are drawn connected to
   their mother.
2. A `<file>_label_tracks.csv` written by Track Statistics, next to the file
   or in a folder directly below it.
3. The labels themselves, read on the spot when the file is under 2 GB
   uncompressed. Larger files show a message in the status bar instead; run
   Track Statistics on them once and the table from step 2 is used.

Files whose name does not contain `_tracked` or `_ultrack` get no Tracks
layer.

## The Batch Function

1. Open **Batch Image Processing** and select the folder with the tracked
   label images (input suffix e.g. `_tracked.tif`).
2. Choose **Track Statistics (QC)**.
3. Start the run. The input files are read one plane at a time, so large
   stacks do not need to fit in memory.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_track_length` | 3 | Tracks present in fewer frames than this count as short |
| `save_track_table` | on | Also write `<file>_label_tracks.csv`, which the results table uses for its Tracks layer |

### Output

`track_statistics.csv` in the output folder, one row per file. Running the
function again on a file replaces that file's row.

| Column | Meaning | What a problem looks like |
|---|---|---|
| `frames` | Number of time points | |
| `objects_per_frame_min` / `_mean` / `_max` | Labelled objects per frame | A frame with far fewer objects than the rest: segmentation failed there |
| `tracks` | Number of distinct track ids | Much larger than objects per frame: tracks keep breaking |
| `track_length_mean` / `_median` | Frames per track | Close to 1: the tracker linked almost nothing |
| `tracks_full_length` / `_fraction` | Tracks present in every frame | Low for a field of cells that stay in view |
| `tracks_short` | Tracks shorter than `min_track_length` | Many: fragmentation or noise objects |
| `tracks_with_gaps` | Tracks that disappear and come back | Expected with gap closing; many means missed detections |
| `tracks_starting_late` | Tracks that start after the first frame | Cells entering the field, divisions, or broken tracks |
| `tracks_ending_early` | Tracks that end before the last frame | Cells leaving, dying, dividing, or broken tracks |

`<file>_label_tracks.csv` (optional) lists every object in every frame:
`track_id, t, [z,] y, x, voxels`, with positions in pixels at the object's
centre.

### Limits

- **Divisions are not counted.** A tracked label image does not record which
  track a daughter came from, so a division appears as one track ending and
  two starting. Ultrack's own CSV keeps this information, and the results
  table uses it.
- Time must be the first axis. Choose `TYX` or `TZYX` (or `Auto`) as the
  dimension order; other orders are rejected.

## See Also

- [Trackastra Tracking](trackastra_tracking.md)
- [HOCT Tracking](hoct_tracking.md)
- [Ultrack Tracking](ultrack_tracking.md)
- [Grid View](grid_view_overlay.md): one-image overview of segmentations
  before tracking

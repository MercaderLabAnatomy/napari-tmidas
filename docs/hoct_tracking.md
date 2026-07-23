# HOCT Cell Tracking

## Overview

Automatic cell tracking for time-lapse microscopy using **HOCT** (Higher-Order
Cell Tracking Transformer, [github.com/royerlab/hoct](https://github.com/royerlab/hoct)),
a transformer-based tracking framework from royerlab. This processing
function tracks cells across time points in segmented label images,
maintaining consistent object IDs throughout the time series.

## Features

- **Deep Learning-Based Tracking**: Uses pre-trained HOCT models (auto-downloaded on first use)
- **2D and 3D Support**: Handles both TYX and TZYX time-lapse data
- **Tiled Inference**: Automatically tiles large volumes to avoid GPU out-of-memory errors
- **Automatic Environment Management**: Creates a dedicated conda environment for HOCT
- **Multi-GPU Aware**: Runs one or more workers per available GPU (configurable via `workers_per_gpu`, not the CPU thread-count control) so concurrent files spread across cards, matching the Trackastra/Cellpose pattern

## Installation

HOCT runs in a dedicated conda environment that is automatically created
when first used. The environment includes:
- `hoct[bioio]` (pulls in torch, gurobipy, tracksdata, and related dependencies)

These are automatically installed into a dedicated `hoct` conda environment
when first used.

## Parameters

### `model` (string, default: "")
Checkpoint path or registered HOCT model name. Leave empty to auto-download
and use the default pretrained model.

### `device` (string, default: "cuda", options: ["cuda", "cpu", "mps"])
Compute device for inference. HOCT falls back to CPU automatically (with a
warning) if the requested device is unavailable.

### `window` (int, default: 5)
Temporal window size for the frame dataset used during prediction.

### `max_distance` (float, default: 300.0)
Maximum spatial distance (in pixels) allowed for candidate tracking edges
between objects in neighboring frames.

### `neighbors` (int, default: 5)
Maximum number of candidate neighbors considered per node when building the
tracking graph.

### `max_dt` (int, default: 3)
Maximum temporal gap (in frames) allowed for candidate edges, so tracks can
bridge across a small number of missed detections.

### `tile` (string, default: "auto", options: ["auto", "on", "off"])
Tiled inference mode for large volumes:
- **`"auto"`**: Enables tiling automatically when the candidate graph is
  dense enough to risk GPU out-of-memory.
- **`"on"`**: Always tile.
- **`"off"`**: Never tile.

### `scale` (string, default: "")
Optional physical voxel size as space-separated values, `"t y x"` for 2D+t
data or `"t z y x"` for 3D+t data (e.g. `"1 0.5 0.2 0.2"`). Leave empty to
track in pixel units.

### `gurobi_license` (string, default: "")
Path to a Gurobi license file (`.lic`) used by HOCT's ILP solver. Leave
empty to auto-detect `~/gurobi.lic` (or an already-exported
`GRB_LICENSE_FILE`); only needed to override the bundled size-limited pip
license.

### `workers_per_gpu` (int, default: 1, min: 1, max: 8)
Number of concurrent HOCT jobs to run per GPU (only relevant when
`device="cuda"`). The default runs one file at a time per card; raise it if a
single GPU has enough VRAM to run more than one tracking job simultaneously.
Batch runs use `n_gpus × workers_per_gpu` concurrent workers in total (the
"Number of threads" control in the UI is not used for this function — it is
replaced by this automatic per-GPU distribution).

### `label_pattern` (string, default: "_labels.tif")
Pattern to identify label images in filenames.

**Use cases**:
- When processing raw images: Plugin looks for corresponding label files with this pattern
- When processing label images: Plugin identifies paired raw images

**Examples**:
- `"_labels.tif"`: Matches `sample_labels.tif`
- `"_mask.tif"`: Matches `sample_mask.tif`
- `"_segmentation.tif"`: Matches `sample_segmentation.tif`

## Usage

### Prerequisites

1. **Time-series label images**: Already segmented with Cellpose, manual annotation, or other methods
2. **A matching raw image of the same shape**: HOCT requires both the raw movie and its segmentation, and their shapes must match exactly
3. **At least 2 timepoints**: Tracking requires temporal information

### In napari-tmidas

1. Open **Plugins > T-MIDAS > Image Processing**
2. Browse to your folder containing label images
3. Use the suffix filter to select label files (e.g., `_labels.tif`)
4. Select **"Track Cells with HOCT"** from the processing function dropdown
5. Configure parameters (defaults are reasonable starting points)
6. Click **"Start Batch Processing"**

### File Structure Expected

```
experiment_folder/
├── sample001.tif           # Raw time-lapse image
├── sample001_labels.tif    # Segmented labels (from Cellpose/other)
├── sample002.tif
├── sample002_labels.tif
└── ...
```

## Input Data

### Supported Dimensions

- **TYX**: 3D array (Time, Y, X) - 2D time series
- **TZYX**: 4D array (Time, Z, Y, X) - 3D time series

### Requirements

- **Minimum 2 timepoints** for tracking
- **Label images**: Instance segmentation with unique IDs per object per timepoint
- **Raw and label shapes must match**: unlike some other trackers, HOCT does not
  auto-select a channel from a multichannel raw image; pre-select a single
  channel before tracking if needed

### Input file formats

- TIFF (`.tif`, `.tiff` files)
- Zarr (`.zarr` directories, including OME-Zarr)

## Output

**Suffix**: `_hoct_tracked`

HOCT exports directly to CTC (Cell Tracking Challenge) format internally
(one labeled TIFF per timepoint plus a `res_track.txt` lineage file); this
function stitches that CTC output into a single relabeled multi-page TIFF so
the result matches the conventions of other tracking functions in this
plugin:
- Each cell maintains a consistent ID across all timepoints
- Output dimensions match input dimensions (TYX → TYX, TZYX → TZYX)
- Background remains 0

## Technical Details

### Processing Pipeline

1. **Input validation**: Checks for time dimension and minimum timepoints
2. **Environment check**: Ensures the dedicated `hoct` conda environment exists and the `hoct` CLI is usable
3. **File preparation**: Identifies label and raw image pairs
4. **Tracking**: Runs `hoct track ... -f ctc` in the dedicated environment
5. **Output assembly**: Streams the CTC mask frames into a single relabeled TIFF

### Environment Isolation

- Dedicated `hoct` conda environment
- Isolated from main napari-tmidas environment
- Uses subprocess calls (`conda run -n hoct hoct track ...`) for cross-environment execution
- Prevents dependency conflicts (e.g. HOCT pins `gurobipy<13.0.0`, which can differ from other trackers)

## Troubleshooting

### "HOCT environment not found"
- Environment is created automatically on first use
- Wait for installation to complete (may take a few minutes)
- Check terminal/console for installation progress

### "Could not find raw image for ..." / requires a matching raw image
- HOCT's CLI requires both a raw image and its segmentation, with identical shapes
- Verify a raw file exists alongside the label file with the expected naming
- Check `label_pattern` matches your label file naming convention

### Poor tracking quality / GPU out-of-memory
- Try `tile="on"` to force tiled inference on large volumes
- Reduce `neighbors` or `max_distance` to shrink the candidate graph
- Provide a full Gurobi license via `gurobi_license` if the ILP solver fails on large problems

## Credits

HOCT is developed by royerlab:
- [HOCT GitHub](https://github.com/royerlab/hoct)

## See Also

- [Cellpose Segmentation](cellpose_segmentation.md) - Segment cells before tracking
- [TrackAstra Cell Tracking](trackastra_tracking.md) - An alternative deep learning-based tracker
- [Ultrack Tracking](ultrack_tracking.md) - Ensemble-based tracking
- [Regionprops Analysis](regionprops_analysis.md) - Extract properties from tracked objects
- [All Processing Functions](all_processing_functions.md) - Label manipulation tools

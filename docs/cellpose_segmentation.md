
# Cellpose-SAM Segmentation

## Overview

Automatic instance segmentation using **Cellpose 4 (Cellpose-SAM)** for batch microscopy workflows in napari-tmidas.

## Features

- Cellpose 4 support with improved generalization
- 2D and 3D segmentation (`YX`, `ZYX`)
- Time-series support (`TZYX`)
- Automatic environment management
- GPU acceleration when available
- Distributed segmentation for large zarr volumes (`ZYX`, `TZYX`, `TCZYX`)
- Automatic non-zarr to temporary zarr conversion in distributed mode

## Installation

Cellpose is installed automatically into a dedicated environment on first use.

### First Run Behavior

If Cellpose is not detected, napari-tmidas will:

1. Create `~/.napari-tmidas/envs/cellpose-env`
2. Install Cellpose and dependencies
3. Run Cellpose jobs in that environment

### Environment Selection Behavior

There are two different environment decisions in this workflow:

- **Cellpose runtime**:
  - Uses the current environment only if Cellpose is already importable there.
  - Otherwise uses the dedicated `cellpose-env` (created automatically).

- **ConvPaint auto-mask runtime** (used only when `use_convpaint_auto_mask=True`):
  - Uses current environment if `napari-convpaint` is available and `convpaint_force_dedicated_env=False`.
  - Uses dedicated ConvPaint environment automatically when `napari-convpaint` is not available.
  - Always uses dedicated ConvPaint environment when `convpaint_force_dedicated_env=True`.

Practical note:
- `convpaint_force_dedicated_env` is an override switch, not an environment-name selector.
- At the moment, this workflow does not expose a widget parameter to provide a custom conda/mamba environment name.

## Parameters

### `channel` (string, default: `"all"`)

Select which detected channel to segment for multichannel inputs.

- `all`: process all channels
- `0`, `1`, ...: process one specific channel

### `dim_order` (string, default: `"YX"`)

Input dimension order.

- `"YX"`: 2D image
- `"ZYX"`: 3D volume
- `"CZYX"`: multichannel 3D volume
- `"TZYX"`: 4D time-lapse 3D data
- `"TCZYX"`: multichannel time-lapse 3D data
- `"TYX"`: 2D time-lapse

Recommended for usability:
- If a `C` dimension exists, include it in `dim_order` (for example `CZYX` or `TCZYX`).
- Channel selection is still controlled by the `channel` parameter.

### `timepoint_start` (int, default: `0`)

Start timepoint index (0-based, inclusive). Used only when `T` is present.

### `timepoint_end` (int, default: `-1`)

End timepoint index (0-based, inclusive).

- `-1` means the last available timepoint

### `timepoint_step` (int, default: `1`)

Process every Nth timepoint.

- `1`: all timepoints
- `2`: every other timepoint

### `diameter` (float, default: `0.0`, range: `0.0-200.0`)

Optional object diameter in pixels.

- Keep at `0.0` in most cases (recommended)
- Set only when objects are outside the usual 7.5-120 px range

### `flow_threshold` (float, default: `0.4`, range: `0.1-0.9`)

Flow threshold for detection sensitivity.

- Lower values: more permissive
- Higher values: more stringent

### `cellprob_threshold` (float, default: `0.0`, range: `-6.0 to 6.0`)

Cell probability threshold.

- Higher: fewer splits
- Lower: more splits

### `anisotropy` (float, default: `1.0`, range: `0.1-100.0`)

Rescaling factor for 3D anisotropy.

`anisotropy = z-step (um) / xy-pixel-size (um)`

### `flow3D_smooth` (int, default: `0`, range: `0-10`)

Gaussian smoothing for 3D flow fields.

### `tile_norm_blocksize` (int, default: `128`, range: `32-512`)

Tile size for Cellpose normalization.

### `batch_size` (int, default: `32`, range: `1-128`)

Number of slices/images processed together.

### `use_distributed_segmentation` (bool, default: `False`)

Enable distributed blockwise Cellpose for large zarr volumes.

- Recommended for large `ZYX`/`TZYX`/`TCZYX`
- Direct zarr processing when input is zarr
- Non-zarr inputs are auto-converted to temporary zarr

### `distributed_blocksize` (int, default: `256`, range: `64-1024`)

Distributed block edge size (voxels).

### `use_convpaint_auto_mask` (bool, default: `False`)

Generate a ConvPaint foreground mask before distributed Cellpose.

### `convpaint_model_path` (string, default: `""`)

Path to ConvPaint `.pkl` model (required if auto-mask is enabled).

### `convpaint_image_downsample` (int, default: `4`, range: `1-16`)

Downsampling used for ConvPaint mask inference.

### `convpaint_background_label` (int, default: `1`, range: `0-255`)

ConvPaint label treated as background for mask generation.

### `convpaint_mask_dilation` (int, default: `2`, range: `0-20`)

Binary dilation iterations applied to ConvPaint foreground mask.

### `convpaint_use_cpu` (bool, default: `False`)

Run ConvPaint mask generation on CPU.

### `convpaint_force_dedicated_env` (bool, default: `False`)

Force ConvPaint execution in its dedicated environment.

Use this when:
- Current environment has `napari-convpaint` but you still want isolated/consistent dedicated-env behavior.

### `convpaint_z_batch_size` (int, default: `0`, range: `0-200`)

Z-batching for ConvPaint mask generation.

- `0` disables batching

## Usage

1. Open **Plugins > T-MIDAS > Image Processing**
2. Select your input folder
3. Choose **Cellpose-SAM Segmentation**
4. Set parameters for your dataset
5. Start batch processing

## Output

- Suffix: `_labels`
- Background: `0`
- Objects: unique positive integer labels
- Output dimensionality matches the selected input dimensionality/timepoint subset

## Tips

### Start with Defaults

- Cellpose-SAM usually works well out of the box
- Tune only after checking real outputs

### Tune Sensitivity

- Too many false positives: increase `flow_threshold`
- Missing cells: decrease `flow_threshold`
- Over-splitting: increase `cellprob_threshold`

### Handle Anisotropy Carefully

- Verify Z spacing versus XY pixel size
- Set `anisotropy` accordingly for 3D data

### Manage Memory

- Reduce `batch_size` if memory is tight
- Use `use_distributed_segmentation` for large 3D volumes
- In distributed mode, non-zarr inputs are auto-converted to zarr

### QC Every Run

- Inspect several outputs before processing large batches
- Iterate parameters with side-by-side comparison in the table UI

## Troubleshooting

### Cellpose Environment Not Found

- The environment is created automatically on first run
- Wait for installation to complete

### Out of Memory

- Lower `batch_size`
- Lower `distributed_blocksize`
- Enable distributed mode for large volumetric data

### Poor Segmentation Quality

- Under-segmentation: reduce `flow_threshold`
- Over-segmentation: increase `cellprob_threshold`
- 3D artifacts: verify `anisotropy`

### Slow Time-Lapse Processing

- Use a GPU if available
- Increase `batch_size` if memory permits
- Use timepoint interval controls to process a subset first

## Technical Details

### Model

- Cellpose 4 (Cellpose-SAM)
- Trained for broad morphology generalization

### Pipeline

1. Validate dimensions and selected axes
2. Optionally subset timepoints
3. Run normalization and inference
4. Return instance labels

## Credits

- [Cellpose GitHub](https://github.com/MouseLand/cellpose)
- [Cellpose 3 Paper](https://www.nature.com/articles/s41592-024-02233-6)

## See Also

- [Basic Processing Functions](basic_processing.md)
- [Intensity-Based Label Filtering](intensity_label_filter.md)
- [Regionprops Analysis](regionprops_analysis.md)

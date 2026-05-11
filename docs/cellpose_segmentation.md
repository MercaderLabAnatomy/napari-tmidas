
# Cellpose-SAM Segmentation

## Overview

Automatic instance segmentation using **Cellpose 4 (Cellpose-SAM)** for batch microscopy workflows in napari-tmidas.

## Features

- **Cellpose 4 Support**: Uses Cellpose-SAM with strong generalization.
- **2D/3D/Time-Series Support**: Works with `YX`, `ZYX`, `TZYX`, and multichannel variants.
- **Automatic Environment Management**: Uses dedicated Cellpose environment when needed.
- **GPU Acceleration**: Uses GPU when available.
- **Distributed Zarr Segmentation**: Optional blockwise processing for large zarr volumes.
- **ConvPaint Auto-Mask Gating**: Optional foreground mask to skip clear-background blocks.
- **Fine-Mask Output Clipping**: Optional final clipping of labels to ConvPaint fine mask.
- **Small-Object Speckle Filtering**: Optional connected-component filtering in ConvPaint mask generation.

## Installation

Cellpose is installed automatically into a dedicated environment on first use.

### First Run Behavior

If Cellpose is not detected, napari-tmidas will:

1. Create `~/.napari-tmidas/envs/cellpose-env`
2. Install Cellpose and dependencies
3. Run Cellpose jobs in that environment

### Environment Selection Behavior

There are two independent environment decisions:

- **Cellpose runtime**:
  - Uses current environment only if Cellpose is importable there.
  - Otherwise uses dedicated `cellpose-env`.

- **ConvPaint auto-mask runtime** (only when `use_convpaint_auto_mask=True`):
  - Uses current environment if `napari-convpaint` is available and `convpaint_force_dedicated_env=False`.
  - Uses dedicated ConvPaint environment automatically when `napari-convpaint` is unavailable.
  - Always uses dedicated ConvPaint environment when `convpaint_force_dedicated_env=True`.

## Parameters

### `channel` (string, default: `"all"`)

Select channel(s) for multichannel inputs.

- `0`, `1`, ...: process one specific channel

For Cellpose-SAM, multichannel inputs must use one selected channel. `all` is not supported for multichannel segmentation.

### `dim_order` (string, default: `"YX"`)

Input dimension order.

- `YX`: 2D image
- `ZYX`: 3D volume
- `TZYX`: 4D time-lapse 3D data
- `TYX`: 2D time-lapse

For Cellpose-SAM, `dim_order` should describe only the non-channel axes used for segmentation. The channel axis is handled separately by the `channel` selector.

Examples:

- `TCZYX` input with one selected channel -> use `TZYX`
- `TZCYX` input with one selected channel -> use `TZYX`
- `CZYX` input with one selected channel -> use `ZYX`

### `timepoint_start` (int, default: `0`)

Start timepoint index (0-based, inclusive). Used only when `T` is present.

### `timepoint_end` (int, default: `-1`)

End timepoint index (0-based, inclusive). `-1` means last timepoint.

### `timepoint_step` (int, default: `1`)

Process every Nth timepoint.

### `diameter` (float, default: `0.0`, range: `0.0-200.0`)

Optional object diameter in pixels.

- Keep `0.0` for most cases (recommended)
- Set only when objects are outside the usual ~7.5-120 px range

### `flow_threshold` (float, default: `0.4`, range: `0.1-0.9`)

Flow threshold for detection sensitivity.

### `cellprob_threshold` (float, default: `0.0`, range: `-6.0 to 6.0`)

Cell probability threshold.

### `anisotropy` (float, default: `1.0`, range: `0.1-100.0`)

Rescaling factor for 3D anisotropy: `anisotropy = z-step / xy-pixel-size`.

### `flow3D_smooth` (int, default: `0`, range: `0-10`)

Gaussian smoothing for 3D flow fields.

### `tile_norm_blocksize` (int, default: `128`, range: `32-512`)

Tile size for Cellpose normalization.

### `batch_size` (int, default: `32`, range: `1-128`)

Number of slices/images processed together.

### `use_distributed_segmentation` (bool, default: `False`)

Enable distributed blockwise Cellpose for large zarr volumes.

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

### `convpaint_min_object_fraction_of_median` (float, default: `0.25`, range: `0.0-2.0`)

Removes ConvPaint foreground components smaller than this fraction of the slab median component size.

- `0.0`: disables filtering
- `0.25`: good starting point
- increase to `0.35-0.5` for stricter filtering

### `clip_final_labels_to_convpaint_mask` (bool, default: `True`)

After Cellpose completes, zero labels outside the fine ConvPaint mask.

### `convpaint_use_cpu` (bool, default: `False`)

Run ConvPaint mask generation on CPU.

### `convpaint_force_dedicated_env` (bool, default: `False`)

Force ConvPaint execution in dedicated environment.

### `convpaint_z_batch_size` (int, default: `0`, range: `0-200`)

Z-batching for ConvPaint mask generation (`0` disables batching).

## Usage

1. Open **Plugins > T-MIDAS > Image Processing**
2. Select your input folder
3. Choose **Cellpose-SAM Segmentation**
4. Select a single segmentation channel for multichannel data
5. Set `dim_order` for the non-channel axes, or leave the selector on `Auto`
6. Set the remaining parameters for your dataset
7. Start batch processing

## Output

- Suffix: `_labels`
- Background: `0`
- Objects: unique positive integer labels
- Output dimensionality matches selected input dimensionality/timepoint subset

## Tips

### Start with Defaults

- Cellpose-SAM usually works well out of the box
- Tune only after checking outputs

### Tune Sensitivity

- Too many false positives: increase `flow_threshold`
- Missing cells: decrease `flow_threshold`
- Over-splitting: increase `cellprob_threshold`

### Handle Anisotropy Carefully

- Verify Z spacing versus XY pixel size
- Set `anisotropy` accordingly for 3D data

### Manage Memory

- Reduce `batch_size` if memory is tight
- Use `use_distributed_segmentation` for large zarr volumes
- In distributed mode, non-zarr inputs are auto-converted to temporary zarr

### Mixed TCZYX and TZCYX Inputs

- Cellpose-SAM normalizes multichannel inputs to a channel-free axis order after channel selection.
- This means mixed folders containing `TCZYX` zarr data and `TZCYX` TIFF data can both be segmented correctly when a single channel is selected.
- In these cases, the effective Cellpose `dim_order` is `TZYX` for both layouts.

### ConvPaint Gating for Faster Distributed Runs

- Enable `use_convpaint_auto_mask` to skip empty/background blocks
- Start with `convpaint_min_object_fraction_of_median=0.25`
- Increase to `0.35-0.5` if tiny speckles still activate blocks
- Keep `clip_final_labels_to_convpaint_mask=True`

### Distributed Speed Model (Important)

Distributed mode is often used for memory safety, not guaranteed speed.
It is faster than full-volume processing only when the **active block fraction**
stays low after mask expansion.

You can think of runtime as:

$$
T_{distributed} \approx N_{active\ blocks}\cdot t_{block} + T_{overhead}
$$

where $T_{overhead}$ includes block scheduling, I/O, and merge steps.

- Small foreground fraction alone is not enough; active block fraction matters.
- If block activation is broad, distributed overhead can dominate.
- In many datasets, once active blocks exceed about 40-50%, full-volume GPU
  can be as fast or faster.

Practical tuning:

- Start `distributed_blocksize` around `128-192`; avoid very small blocks unless necessary.
- Increase `convpaint_min_object_fraction_of_median` if tiny components still activate blocks.
- Use minimal necessary mask dilation/expansion to avoid turning on large empty regions.
- If GPU memory allows and many blocks are active, try non-distributed mode.

### QC Every Run

- Inspect several outputs before large runs
- Iterate parameters with side-by-side comparison in the table UI

## Troubleshooting

### Cellpose Environment Not Found

- Environment is created automatically on first run
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

- Use GPU if available
- Increase `batch_size` if memory permits
- Use timepoint interval controls to process a subset first
- In distributed mode, check active block ratio. If many blocks are active,
  try larger block size tuning and/or full-volume GPU evaluation.

## Technical Details

### Model

- Cellpose 4 (Cellpose-SAM)
- Trained for broad morphology generalization

### Pipeline

1. Validate dimensions and selected axes
2. Optionally subset timepoints
3. Optionally generate ConvPaint mask and prune tiny components
4. Run normalization and Cellpose inference
5. Optionally clip final labels to ConvPaint fine mask
6. Return instance labels

## Credits

- [Cellpose GitHub](https://github.com/MouseLand/cellpose)
- [Cellpose 3 Paper](https://www.nature.com/articles/s41592-024-02233-6)

## See Also

- [Basic Processing Functions](basic_processing.md)


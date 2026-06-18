
# Cellpose-SAM Segmentation

## Overview

Automatic instance segmentation using **Cellpose 4 (Cellpose-SAM)** for batch microscopy image processing workflows in napari-tmidas.

## Features

- **Cellpose 4 Support**: Uses Cellpose-SAM with strong generalization.
- **Foundation Model Selection**: Supports `cpsam_v2` (default), `cpsam`, `cpdino`, and `cpdino-vitb`.
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

When available, napari-tmidas also installs `dinov3` in the Cellpose environment so
`cpdino` and `cpdino-vitb` can be selected.

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

### `model_type` (string, default: `"cpsam_v2"`)

Cellpose model variant to run.

- `cpsam_v2`: recommended default (updated Cellpose-SAM model)
- `cpsam`: legacy Cellpose-SAM model
- `cpdino`: Cellpose Dino model
- `cpdino-vitb`: smaller Dino ViT-B model

`cpdino` variants require `dinov3`:

```bash
python -m pip install git+https://github.com/facebookresearch/dinov3
```

### `diameter` (float, default: `0.0`, range: `0.0-200.0`)

Optional object diameter in pixels.

- Keep `0.0` for most cases (recommended)
- Set only when objects are outside the usual ~7.5-120 px range

### `flow_threshold` (float, default: `0.4`, range: `0.1-0.9`)

Maximum allowed flow error per mask. For 2D, increase for more ROIs and decrease to reject ill-shaped masks. In Cellpose 3D mode, this parameter is ignored by Cellpose.

### `cellprob_threshold` (float, default: `0.0`, range: `-6.0 to 6.0`)

Cell probability threshold.

### `anisotropy` (float, default: `1.0`, range: `0.1-100.0`)

Rescaling factor for 3D anisotropy: `anisotropy = z-step / xy-pixel-size`.

### `flow3D_smooth` (int, default: `0`, range: `0-10`)

Gaussian smoothing for 3D flow fields. Useful when 3D masks are fragmented or show ring-like artifacts.

### `tile_norm_blocksize` (int, default: `128`, range: `32-512`)

Tile size for Cellpose normalization.

### `batch_size` (int, default: `32`, range: `1-128`)

Number of slices/images processed together.

### `use_distributed_segmentation` (bool, default: `False`)

Enable distributed blockwise Cellpose for large zarr datasets, including large 3D volumes and large 2D planes such as whole-slide-style images.

### `distributed_blocksize_yx` (int, default: `256`, range: `64-1024`)

Distributed block edge size (voxels).

### `distributed_n_workers` (int, default: `1`, range: `1-16`)

Number of distributed workers used by Cellpose block processing.

- `1`: single-worker distributed mode (typically one GPU active)
- `2+`: enables concurrent block workers; napari-tmidas assigns one CUDA device per worker in round-robin order when GPUs are available

Note: Cellpose distributed execution is block-level parallelism. It does not split one block across multiple GPUs.

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
- Direct zarr output preserves OME multiscale depth and metadata when source metadata is available
- TIFF output is written as OME-TIFF labels

## Tips

### Start with Defaults

- Cellpose-SAM usually works well out of the box
- Tune only after checking outputs

### Tune Sensitivity

- 2D data:
  - Missing cells (too few ROIs): increase `flow_threshold` and/or decrease `cellprob_threshold`
  - Too many dim/background ROIs: increase `cellprob_threshold`
  - Too many ill-shaped or fragmented ROIs: decrease `flow_threshold`
- 3D data:
  - Cellpose ignores `flow_threshold` in 3D mode
  - If masks are fragmented or ring-like, increase `flow3D_smooth`
  - Use post-filtering by object size to remove small false positives

### Anisotropy

- Leave at default `1.0` for most cases
- Only adjust if you encounter Z-axis segmentation artifacts
- Changing anisotropy from the default can sometimes cause problems, so verify carefully before adjusting

### Manage Memory

- Reduce `batch_size` if memory is tight
- Use `use_distributed_segmentation` for large zarr volumes or very large 2D zarr planes
- Increase `distributed_n_workers` to leverage multiple GPUs with distributed mode (one worker per GPU is the usual starting point)
- In distributed mode, non-zarr inputs are auto-converted to temporary zarr (v3 by default)

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



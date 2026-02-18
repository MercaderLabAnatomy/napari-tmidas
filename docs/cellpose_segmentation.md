# Cellpose-SAM Segmentation

## Overview

Automatic instance segmentation using **Cellpose 4 (Cellpose-SAM)** with improved generalization for cellular segmentation. This processing function integrates the deep learning-based Cellpose segmentation toolkit into napari-tmidas for batch processing of microscopy images.

## Features

- **Cellpose 4 Support**: Uses the latest Cellpose-SAM model with improved generalization
- **2D and 3D Segmentation**: Handles both 2D (YX) and 3D (ZYX) data
- **Time Series Support**: Processes 4D time-lapse data (TZYX) by segmenting each timepoint
- **Automatic Environment Management**: Creates a dedicated conda environment if Cellpose is not installed
- **Flexible Dimension Orders**: Supports various dimension arrangements (YX, ZYX, TZYX)
- **GPU Acceleration**: Automatically uses GPU if available

## Installation

Cellpose will automatically install into a dedicated isolated environment when first used.

### What Happens on First Use

If Cellpose is not detected, the plugin will:
1. Create a dedicated `cellpose-env` environment at `~/.napari-tmidas/envs/cellpose-env`
2. Install Cellpose and dependencies automatically
3. Use this environment for all Cellpose operations

## Parameters

### `dim_order` (string, default: "YX")
Dimension order of the input image. Examples:
- `"YX"`: 2D image
- `"ZYX"`: 3D volumetric data
- `"TZYX"`: 4D time-lapse 3D data
- `"TYX"`: 2D time-lapse

### `diameter` (float, default: 0.0, range: 0.0-200.0)
**Optional parameter**. Typical cell/nucleus diameter in pixels.

**When to set diameter**:
- Leave at `0.0` (recommended) for most cases - Cellpose-SAM is trained for ROI diameters 7.5–120 pixels
- Only set if your objects are **outside this range** (e.g., very small <7.5 or very large >120 pixels)

**How to determine diameter**:
1. Measure a few representative objects in your image
2. Calculate the average diameter in pixels
3. Enter this value

### `flow_threshold` (float, default: 0.4, range: 0.1-0.9)
Controls cell detection sensitivity.
- **Lower values** (e.g., 0.2): More permissive, detects more cells (may include false positives)
- **Higher values** (e.g., 0.6): More stringent, only high-confidence detections

### `cellprob_threshold` (float, default: 0.0, range: -6.0 to 6.0)
Cell probability threshold.
- **Positive values**: Reduce over-segmentation (fewer splits)
- **Negative values**: Allow more splits
- Start with 0.0 and adjust based on results

### `anisotropy` (float, default: 1.0, range: 0.1-10.0)
**For 3D data only**. Rescaling factor for Z-dimension.

**Formula**: `anisotropy = Z step size (μm) / XY pixel size (μm)`

**Examples**:
- Z step = 2 μm, XY pixel = 0.5 μm → anisotropy = 4.0
- Z step = 0.3 μm, XY pixel = 0.1 μm → anisotropy = 3.0
- Z step = XY pixel size → anisotropy = 1.0 (isotropic)

### `flow3D_smooth` (int, default: 0, range: 0-10)
**For 3D data only**. Gaussian smoothing standard deviation for flow field.
- `0`: No smoothing
- `1-3`: Light smoothing (recommended for noisy data)
- `>3`: Heavy smoothing (may reduce detail)

### `tile_norm_blocksize` (int, default: 128, range: 32-512)
Block size for tile-based normalization (Cellpose 4 feature).
- Smaller values: More local normalization, better for uneven illumination
- Larger values: More global normalization

### `batch_size` (int, default: 32, range: 1-128)
Number of images/slices processed simultaneously.
- Increase for faster processing (requires more memory)
- Decrease if you encounter out-of-memory errors

## Usage

### In napari-tmidas

1. Open **Plugins > T-MIDAS > Image Processing**
2. Browse to your folder containing images
3. Select **"Cellpose-SAM Segmentation"** from the processing function dropdown
4. Configure parameters based on your data:
   - Set `dim_order` to match your image dimensions
   - Adjust `flow_threshold` if detection is too sensitive/insensitive
   - Set `anisotropy` for 3D data if Z-spacing differs from XY
5. Click **"Start Batch Processing"**

### Example Workflows

#### 2D Cell Segmentation
```
Input: 2D brightfield images (YX)
Parameters:
  - dim_order: "YX"
  - diameter: 0.0 (auto)
  - flow_threshold: 0.4
Output: Label images with instance segmentation
```

#### 3D Nucleus Segmentation
```
Input: 3D confocal stack (ZYX)
Parameters:
  - dim_order: "ZYX"
  - diameter: 0.0 (auto)
  - anisotropy: 2.5 (if Z-spacing is 2.5× larger than XY)
  - flow_threshold: 0.4
Output: 3D label image with segmented nuclei
```

#### Time-Lapse Analysis
```
Input: 4D time-lapse (TZYX)
Parameters:
  - dim_order: "TZYX"
  - diameter: 0.0 (auto)
  - flow_threshold: 0.4
Output: 4D label image with tracked instances per timepoint
```

## Output

**Suffix**: `_labels`

The function produces label images where:
- Background = 0
- Each segmented object = unique positive integer (1, 2, 3, ...)
- Output dimensions match input dimensions (2D → 2D, 3D → 3D, 4D → 4D)

## Tips and Best Practices

### 1. **Start with Default Parameters**
   - Cellpose-SAM works well out-of-the-box for most cell/nucleus images
   - Only adjust parameters if results are unsatisfactory

### 2. **Optimize Detection Sensitivity**
   - **Too many false positives**: Increase `flow_threshold` to 0.5-0.6
   - **Missing cells**: Decrease `flow_threshold` to 0.2-0.3
   - **Over-segmented cells**: Increase `cellprob_threshold` to 1.0-2.0

### 3. **Handle Anisotropic Data**
   - For 3D confocal data, always check Z-spacing vs XY pixel size
   - Use `anisotropy` parameter to account for different Z-spacing
   - Incorrect anisotropy can cause poor 3D segmentation

### 4. **Memory Management**
   - Large 3D or 4D datasets may require reducing `batch_size`
   - Monitor memory usage during processing
   - Consider splitting very large time-lapse data

### 5. **GPU Acceleration**
   - Cellpose automatically uses GPU if available
   - GPU provides 10-50× speedup for large datasets
   - For CPU-only systems, expect longer processing times

### 6. **Quality Control**
   - Always inspect results on a few samples before batch processing
   - Use the table in napari-tmidas to click and compare original vs segmented images
   - Adjust parameters based on visual inspection

## Troubleshooting

### "Cellpose environment not found"
- The plugin will automatically create the environment on first use
- Wait for installation to complete (may take several minutes)
- Alternatively, install Cellpose manually (see Installation section)

### "Out of memory" errors
- Reduce `batch_size` parameter
- Process smaller image regions
- Use a machine with more RAM or GPU memory

### Poor segmentation quality
- **Under-segmentation** (cells merged): Decrease `flow_threshold`, adjust `diameter`
- **Over-segmentation** (cells split): Increase `cellprob_threshold`
- **3D issues**: Check `anisotropy` parameter
- **Inconsistent quality**: Try adjusting `tile_norm_blocksize` for uneven illumination

### Time-lapse processing is slow
- Time-series data is processed sequentially (one timepoint at a time)
- Consider using GPU acceleration
- Increase `batch_size` if memory allows

## Technical Details

### Model Information
- Uses Cellpose 4 (Cellpose-SAM) architecture
- Pre-trained on diverse cellular and nuclear morphologies
- Supports diameters from 7.5 to 120 pixels natively

### Processing Pipeline
1. **Dimension validation**: Ensures correct dimension order
2. **Time-series handling**: Processes each timepoint independently for TZYX data
3. **Preprocessing**: Automatic normalization (1st-99th percentile)
4. **Segmentation**: Cellpose-SAM model inference
5. **Post-processing**: Instance label assignment

### Environment Management
- Dedicated conda environment: `cellpose-env`
- Isolated from main napari-tmidas environment
- Automatic installation of dependencies
- Uses subprocess calls for environment isolation

## Credits

Cellpose is developed by the Stringer and Pachitariu labs:
- [Cellpose GitHub](https://github.com/MouseLand/cellpose)
- [Cellpose 3 Paper](https://www.nature.com/articles/s41592-024-02233-6)

## See Also

- [Basic Processing Functions](basic_processing.md) - Label manipulation tools
- [Intensity-Based Label Filtering](intensity_label_filter.md) - Filter segmentation results by intensity
- [Regionprops Analysis](regionprops_analysis.md) - Extract properties from segmented objects

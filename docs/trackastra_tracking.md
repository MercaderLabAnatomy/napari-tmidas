# TrackAstra Cell Tracking

## Overview

Automatic cell tracking for time-lapse microscopy using **TrackAstra**, a deep learning-based tracking framework. This processing function tracks cells across time points in segmented label images, maintaining consistent object IDs throughout the time series.

## Features

- **Deep Learning-Based Tracking**: Uses pre-trained TrackAstra models
- **2D and 3D Support**: Handles both TYX and TZYX time-lapse data
- **Multiple Tracking Modes**: Greedy, ILP (Integer Linear Programming), and no-division modes
- **Automatic Environment Management**: Creates a dedicated conda environment for TrackAstra
- **Flexible Input**: Works with both raw images and pre-segmented label images

## Installation

TrackAstra runs in a dedicated conda environment that is automatically created when first used. The environment includes:
- TrackAstra
- ILP solver (ilpy)
- PyTorch
- scikit-image
- tifffile

These are automatically available when you install napari-tmidas with deep learning support:

```bash
pip install 'napari-tmidas[deep-learning]'
```

If you installed the basic version without deep learning, the plugin will automatically create a dedicated `trackastra` environment when first used.

## Parameters

### `model` (string, default: "ctc", options: ["general_2d", "ctc"])
TrackAstra model selection:

- **`"ctc"`**: Cell Tracking Challenge model
  - Optimized for cell tracking benchmarks
  - Good for standard cell tracking scenarios
  - Trained on diverse cell types from CTC datasets

- **`"general_2d"`**: General-purpose 2D tracking model
  - More generic, works across various cell types
  - Good fallback if ctc model doesn't work well

### `mode` (string, default: "greedy", options: ["greedy", "ilp", "greedy_nodiv"])
Tracking algorithm mode:

- **`"greedy"`**: Fast greedy linking (default)
  - Fastest option
  - Good for most applications
  - Allows cell divisions
  - May miss some complex tracking scenarios

- **`"ilp"`**: Integer Linear Programming
  - Most accurate but slower
  - Globally optimal solutions
  - Better for complex scenarios with many cells
  - Requires ILP solver (ilpy)

- **`"greedy_nodiv"`**: Greedy without divisions
  - Fast like greedy
  - Does not allow cell divisions
  - Use when cells don't divide during imaging

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
2. **At least 2 timepoints**: Tracking requires temporal information
3. **Matching dimensions**: All timepoints must have the same spatial dimensions

### In napari-tmidas

1. Open **Plugins > T-MIDAS > Image Processing**
2. Browse to your folder containing label images
3. Use the suffix filter to select label files (e.g., `_labels.tif`)
4. Select **"TrackAstra Tracking"** from the processing function dropdown
5. Configure parameters:
   - Choose `model` based on your data
   - Select `mode` (start with "greedy")
   - Set `label_pattern` to match your label file naming
6. Click **"Start Batch Processing"**

### Workflow Example

#### Complete Cell Tracking Pipeline

```
Step 1: Segment cells (Cellpose or manual)
  Input: raw_image.tif (TZYX or TYX)
  Output: raw_image_labels.tif

Step 2: Track cells (TrackAstra)
  Input: raw_image_labels.tif
  Parameters:
    - model: "ctc"
    - mode: "greedy"
    - label_pattern: "_labels.tif"
  Output: raw_image_tracked.tif
```

#### File Structure Expected

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
- **Consistent dimensions**: All timepoints must have same spatial size

### Data Types

- uint8, uint16, uint32 label images supported
- Background should be 0
- Each object should have a unique positive integer ID

## Output

**Suffix**: `_tracked`

The function produces tracked label images where:
- Each cell maintains a consistent ID across all timepoints
- Cell divisions are handled (in "greedy" and "ilp" modes)
- Output dimensions match input dimensions (TYX → TYX, TZYX → TZYX)
- Background remains 0

### Tracking Information

TrackAstra assigns consistent IDs such that:
- Cell at time t=0 with ID=5 keeps ID=5 at t=1, t=2, etc.
- Daughter cells from divisions get new unique IDs
- Lost/appeared cells get new IDs

## Tips and Best Practices

### 1. **Choose the Right Model**
   - Start with `"ctc"` for standard cell tracking
   - Try `"general_2d"` if ctc doesn't work well
   - Model choice depends on cell morphology and imaging conditions

### 2. **Select Appropriate Tracking Mode**
   - **For fast preview**: Use `"greedy"`
   - **For high accuracy**: Use `"ilp"` (slower but better)
   - **For non-dividing cells**: Use `"greedy_nodiv"`

### 3. **Ensure Good Segmentation Quality**
   - TrackAstra tracks pre-segmented objects
   - Poor segmentation = poor tracking
   - Review segmentation before tracking

### 4. **Handle Cell Divisions**
   - Use `"greedy"` or `"ilp"` modes for dividing cells
   - TrackAstra can track cell lineages through divisions
   - Use `"greedy_nodiv"` if divisions are not expected

### 5. **Process Time Series as Single Files**
   - Load time-lapse data as single TZYX or TYX files
   - Don't split into individual timepoint files before tracking
   - TrackAstra needs temporal context

### 6. **Quality Control**
   - Visualize tracked results in napari
   - Check ID consistency across timepoints
   - Verify that divisions are correctly handled

## Troubleshooting

### "TrackAstra environment not found"
- Environment is created automatically on first use
- Wait for installation to complete (may take 5-10 minutes)
- Check terminal/console for installation progress

### "No label file found"
- Verify `label_pattern` matches your label file naming convention
- Ensure label files exist in the same folder as raw images
- Check file extensions match (e.g., .tif vs .tiff)

### "Input is not a time series"
- TrackAstra requires at least 3D data (TYX)
- Ensure time dimension is present
- Check dimension order in your data

### "Need at least 2 timepoints for tracking"
- Tracking requires temporal information
- Ensure your data has multiple time points
- Single timepoints cannot be tracked

### Poor tracking quality
- **IDs jump/change**: Try `"ilp"` mode for global optimization
- **Missed tracks**: Improve segmentation quality first
- **Wrong divisions**: Check if `"greedy_nodiv"` is more appropriate
- **Model mismatch**: Try switching between "ctc" and "general_2d"

### Slow processing
- `"ilp"` mode is slower than "greedy"
- Large datasets (many cells/timepoints) take longer
- Consider using `"greedy"` for initial testing
- Processing time increases with number of objects

## Technical Details

### Processing Pipeline

1. **Input validation**: Checks for time dimension and minimum timepoints
2. **Environment check**: Ensures TrackAstra environment exists
3. **File preparation**: Identifies label and raw image pairs
4. **Tracking script generation**: Creates Python script for TrackAstra
5. **Execution**: Runs tracking in isolated environment via subprocess
6. **Output loading**: Reads tracked labels and returns results

### Environment Isolation

- Dedicated `trackastra` conda environment
- Isolated from main napari-tmidas environment
- Uses subprocess calls for cross-environment execution
- Prevents dependency conflicts

### Algorithm Details

- **Greedy**: Frame-to-frame linking with local decisions
- **ILP**: Global optimization over time using integer programming
- **No-divisions**: Simplified tracking without mitosis handling

## Credits

TrackAstra is developed by the Weigert Lab:
- [TrackAstra GitHub](https://github.com/weigertlab/trackastra)
- [TrackAstra Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf)

## See Also

- [Cellpose Segmentation](cellpose_segmentation.md) - Segment cells before tracking
- [Regionprops Analysis](regionprops_analysis.md) - Extract properties from tracked objects
- [Basic Processing Functions](basic_processing.md) - Label manipulation tools

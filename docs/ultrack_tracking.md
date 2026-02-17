# Ultrack Cell Tracking with Segmentation Ensemble

This guide explains how to use ultrack for cell tracking with segmentation ensemble in napari-tmidas.

## Overview

Ultrack is a versatile and scalable cell tracking method that addresses the challenges of tracking cells across 2D, 3D, and multichannel timelapse recordings. It's especially powerful in complex and crowded tissues where segmentation is often ambiguous.

The segmentation ensemble approach allows ultrack to evaluate multiple candidate segmentations from different methods (e.g., Cellpose, ConvPaint) and select the most consistent tracking solution.

## Features

- **Segmentation Ensemble**: Combine multiple segmentation methods for robust tracking
- **Versatile**: Supports both TYX (2D+time) and TZYX (3D+time) data
- **Optimized Solver**: Optional Gurobi integration for improved performance
- **Automatic Environment Management**: Creates and manages a dedicated conda environment
- **Auto-Repair**: Automatically detects and installs missing packages in existing environments
- **GPU Acceleration**: Full support for modern NVIDIA GPUs including Blackwell architecture

## Environment Compatibility Patches

The ultrack environment in napari-tmidas includes several custom patches to ensure compatibility with modern dependencies and GPU architectures:

### 1. Blackwell GPU Support (sm_120)

**Issue**: CuPy (used by ultrack) does not support NVIDIA Blackwell architecture GPUs (compute capability 12.0, sm_120).

**Solution**: Custom PyTorch implementation of `labels_to_contours` function
- Replaces ultrack's CuPy-based implementation with PyTorch
- PyTorch 2.5+ nightly builds support Blackwell sm_120
- Produces identical output to the original CuPy version
- Included inline in generated tracking scripts for portability

**Location**: `torch_labels_to_contours.py`

### 2. scikit-image Dev Version (0.26.1+)

**Issue 1**: scikit-image 0.26.0 and earlier have issues with read-only zarr arrays, causing "buffer source array is read-only" errors.

**Issue 2**: scikit-image 0.26.0 deprecated the `min_size` parameter in `morphology.remove_small_objects()`, replacing it with `max_size`.

**Solution**: Automatic installation of scikit-image dev version
- Environment manager checks scikit-image version during setup
- If < 0.26.1: Automatically installs dev version from GitHub with read-only array fix
- If 0.26.1+ stable is available: Auto-upgrades to stable release
- Runtime patch to ultrack's `hierarchy.py` for deprecated parameter compatibility

**Installation command** (automatic):
```bash
pip install --upgrade git+https://github.com/scikit-image/scikit-image.git@main
```

**Files patched**: 
- `/path/to/ultrack/env/lib/python3.11/site-packages/ultrack/core/segmentation/hierarchy.py`

### 3. NumPy Array Module Assignment Fix

**Issue**: When CuPy is installed but CUDA is unavailable, ultrack fails to assign `xp = np` fallback, causing undefined variable errors.

**Solution**: Runtime patch to ultrack's `cuda.py`
- Adds missing `xp = np` assignment when CUDA is unavailable
- Ensures smooth fallback to NumPy operations

**Files patched**:
- `/path/to/ultrack/env/lib/python3.11/site-packages/ultrack/utils/cuda.py`

### Impact

These patches enable:
- ✅ Full GPU acceleration on latest NVIDIA hardware (RTX 50 series, etc.)
- ✅ Compatibility with scikit-image 0.26+
- ✅ Robust CPU fallback when GPU is unavailable
- ✅ No manual intervention required - most patches auto-apply during environment creation

**Note**: For detailed patch information, see [ultrack_environment_patches.md](ultrack_environment_patches.md).

## Prerequisites

### Required Segmentation Labels

Before running ultrack tracking, you need to generate segmentation labels using one or more methods:

1. **Cellpose Segmentation**: Generate labels with suffix `_cp_labels.tif`
2. **ConvPaint Prediction**: Generate labels with suffix `_convpaint_labels.tif`
3. **Other segmentation methods**: Any method that produces labeled images

The more segmentation methods you use in the ensemble, the more robust the tracking will be.

### Data Format

Your data should be:
- **TYX**: Time series of 2D images (T=time, Y=height, X=width)
- **TZYX**: Time series of 3D images (T=time, Z=depth, Y=height, X=width)

## Usage

### Basic Workflow

1. **Segment your data** using multiple methods:
   - Run "Cellpose-SAM Segmentation" processing function
   - Run "ConvPaint Prediction" processing function
   - (Optional) Run other segmentation methods

2. **Run Ultrack Tracking**:
   - Select "Track Cells with Ultrack (Segmentation Ensemble)" from processing functions
   - Configure parameters (see below)
   - Run batch processing

### Parameters

**About Default Values**: Our parameter defaults are based on ultrack's **multi-color ensemble example** (not the class defaults). The ensemble example demonstrates real biological tracking and uses values that prevent fragmentation. The class defaults (-0.001 for weights) are often too weak and cause track fragmentation.

**Important**: These defaults work well for typical biological data with continuous cell movement. However, **parameter tuning is dataset-specific** - see the troubleshooting section below for guidance on adjusting parameters based on your results.

#### Label Suffixes (Required)
```
_cp_labels.tif,_convpaint_labels.tif
```
Comma-separated list of label file suffixes. The function will look for files with these suffixes to build the ensemble.

**Example**: If your raw image is `sample_001.tif`, ultrack will look for:
- `sample_001_cp_labels.tif`
- `sample_001_convpaint_labels.tif`

**IMPORTANT - File Loading**:
- **Order matters**: The first suffix in the list is the "primary" suffix
- **Load ONLY the first suffix**: Select files like `*_cp_labels.tif` in the file loader
- **Avoid loading all suffixes**: If you load both `*_cp_labels.tif` AND `*_convpaint_labels.tif`, they will be skipped as duplicates
- **Why**: The function automatically finds all ensemble members based on the suffix list

**Output naming**:
- Input: `sample_001_cp_labels.tif` (and `sample_001_convpaint_labels.tif`)
- Output: `sample_001_ultrack.tif` ← Label suffix is stripped, only one output per sample

#### Gurobi License (Optional)
```
(leave empty for default solver)
```
- **Empty**: Use default solver (free, slightly slower)
- **License Key**: Enter your Gurobi academic license key for faster solving

**Important**: The license key only needs to be entered **ONCE** during the first-time ultrack environment creation. After the initial setup, leave this field empty on subsequent runs.

**To get an academic Gurobi license**:
1. Register at [Gurobi's website](https://portal.gurobi.com/iam/login/) with your academic email
2. Navigate to [named academic license page](https://www.gurobi.com/features/academic-named-user-license/)
3. Follow instructions to get your license key
4. **First time only**: Enter the key in the parameter field (e.g., `abc123-def456-ghijkl`)
5. **All subsequent runs**: Leave this field empty (Gurobi will use the activated license)

#### Cell Size Parameters

****Recommended**: Set to **half the size of your smallest cell** (disregarding outliers)
- Smaller objects will be filtered out
- Too low: oversegmentation (cells split incorrectly)
- Too high: missing small cells

**Max Area** (default: 1,000,000)
- Maximum cell area in pixels
- **Recommended**: Set to **1.25-1.5× your largest cell size**
- Larger segments likely represent fused cells and will be split or removed
- **Note**: Less critical to tune than min_area
- **If tracking is fragmented**: Try lowering this (e.g., 100-150) to include smaller detections
- **Note**: Ultrack class default is 100, but the ensemble example uses 200 for better filtering
Recommended**: Set to **1.5× the maximum expected cell movement** between frames
- **How to measure**: Open your data in napari, manually track a fast-moving cell across 2 frames, measure displacement
- Too low: track fragmentation (cells can't link properly)
- Too high: incorrect long-distance connections between unrelated cells
- **Note**: Ultrack class default is 15.0,g
- Higher values = more computational cost but better handling of crowded scenes

**Max Distance** (default: 40.0, from ensemble example)
- Maximum distance (in pixels) between cells for linking across frames
- **CRITICAL FOR FRAGMENTATION**: If you see many short, disconnected tracks:
  - **Increase this value** (e.g., 50-80) bas**All values must be ≤ 0** (negative = penalty). The solver minimizes the total cost, so more negative weights create stronger penalties.

**IMPORTANT**: Our defaults come from ultrack's **multi-color ensemble example**, which prioritizes continuous tracks. However, **weights are highly dataset-specific** - you may need to adjust them based on your results (see troubleshooting below).

**Key principle**: Adjust `disappear_weight` first, then balance `division_weight` with `appear_weight`. Keep `division_weight` equal to or more negative than `appear_weight` to prevent false divisions.

**Appear Weight** (default: -0.1, from ensemble example)
- Penalizes new cell appearances (track starts)
- Range: -10.0 to 0.0, step: 0.001
- More negative → fewer new tracks (e.g., -0.5 for very restrictive)
- Less negative → allow more track starts (e.g., -0.01 for data with many appearing cells)
- **Too negative**: Missing cells that should be tracked
- **Too weak**: Many false track starts

**Disappear Weight** (default: -2.0, from ensemble example)
- Penalizes cell disappearances (track ends)
- Range: -10.0 to 0.0, step: 0.1
- More negative → tracks strongly prefer to continue (e.g., -5.0 for maximum continuity)
- Less negative → allow tracks to end more easily (e.g., -0.1 for data with cells leaving field)
- **Too negative**: Missing cells, incorrect long tracks
- **Too weak**: Fragmented tracks (many short tracks instead of continuous)
- **Tune this first** when adjusting weights

**Division Weight** (default: -0.01, from ensemble example)
- Penalizes cell divisions
- Range: -10.0 to 0.0, step: 0.001
- More negative → fewer divisions detected (e.g., -0.1 if divisions unlikely)
- Less negative → allow more divisions (e.g., -0.001 for rapidly dividing cells)
- **Keep equal to or more negative than `appear_weight`** to prevent false divisions
- **Too negative**: Missing real divisions
- **Too weak**: Many false divisionsr!), which causes severe fragmentation

**Division Weight** (default: -0.01, from ensemble example)
- Penalize cell divisions
- Range: -10.0 to 0.0, step: 0.001
- More negative = fewer divisions detected (e.g., -0.1 if divisions unlikely)
- Less negative = allow more divisions (e.g., -0.001 for rapidly dividing cells)
- **Note**: Ultrack class default is -0.001 (10× weaker)

#### GPU Acceleration

**Enable GPU** (default: ON)
- Uses PyTorch for GPU-accelerated processing
- Supports NVIDIA GPUs including Blackwell architecture (sm_120)
- Turn OFF for CPU-only mode (uses NumPy/scipy)
- GPU mode is typically 5-10× faster for large datasets

**Note**: Window size is automatically set to process the entire timelapse for best tracking quality. For very large datasets, ultrack uses out-of-memory zarr arrays to manage RAM efficiently.

## Output Files

Ultrack tracking produces two output files:

1. **`_ultrack_tracked.tif`**: Tracked label image
   - Each cell has a unique ID that persists across time
   - Cell divisions create new IDs

2. **`_tracks.csv`**: Track data in CSV format
   - Columns: `track_id`, `t`, `z` (if 3D), `y`, `x`, and other properties
   - Can be imported into analysis software

## Tips for Best Results
Understand Your Data First

**Before tuning parameters**, understand your data:

1. **Measure cell movement**: 
   - Open raw data in napari
   - Track fastest-moving cell across 2 frames
   - Measure displacement → set `max_distance` to 1.5× this value

2. **Estimate cell sizes**:
   - Find smallest cell → set `min_area` to half its size
   - Find largest cell → set `max_area` to 1.5× its size

3. **Check segmentation quality**:
   - Are most cells detected in most frames?
   - Are cells properly separated (not fused)?
   - Good segmentation → good tracking

### 4. Tune Based on Results

**Our defaults work well for typical data**, but you may need to adjust based on specific issues:

**See the Troubleshooting section below for specific guidance on**:
- Fragmented tracks (short, disconnected)
- Missing cells
- Too many/few divisions  
- Incorrect connections

**General tuning strategy**:
1. Fix hard constraints first (`max_distance`, `min_area`, `max_area`)
2. Then adjust weights starting with `disappear_weight`
3. Balance `division_weight` with `appear_weight` (keep division ≥ appear in negativity)
**For fast-moving cells** (movement > 40 pixels/frame):
- **Increase `max_distance`**: 60-100 (measure actual displacement!)
- `max_neighbors`: 8-10

**For rapidly dividing cells**:
- `division_weight`: -0.001 (less negative = allow more divisions)
- `max_neighbors`: 8-10

**For sparse data** (few cells, clear boundaries):
- `appear_weight`: -0.01 (less negative = allow new tracks)
- `disappear_weight`: -0.5 (less negative = allow track ends)
- `max_distance`: Can be larger (50-60) since less crowding

### 4. Start with a Subset
For large datasets:
1. Extract a small subset (e.g., 10 frames)
2. Tune parameters on the subset
3. Apply to full dataset once satisfied

## Example Workflow

### Example 1: 2D Timelapse with Two Segmentation Methods

```
Input files:
- embryo_t001.tif (raw image, frame 1)
- embryo_t002.tif (raw image, frame 2)
- ...
- embryo_t100.tif (raw image, frame 100)

Step 1: Run Cellpose Segmentation
Output:
- embryo_t001_cp_labels.tif
- embryo_t002_cp_labels.tif
- ...

Step 2: Run ConvPaint Prediction
Output:
- embryo_t001_convpaint_labels.tif
- embryo_t002_convpaint_labels.tif
- ...

Step 3: Run Ultrack Tracking
Parameters:
- label_suffixes: _cp_labels.tif,_convpaint_labels.tif
- min_area: 150
- max_distance: 30
- (other parameters: use defaults)

Output:
- embryo_t001_ultrack_tracked.tif
- embryo_t001_tracks.csv
```

### Example 2: 3D Timelapse with Gurobi

```
Input: 3D timelapse (TZYX)
- cells3d.tif (shape: 50, 30, 512, 512)

Step 1: Segment with multiple methods
- Run Cellpose 3D: cells3d_cp_labels.tif
- Run watershed-based: cells3d_watershed_labels.tif

Step 2: Run Ultrack Tracking (FIRST TIME with Gurobi)
Parameters:
- label_suffixes: _cp_labels.tif,_watershed_labels.tif
- gurobi_license: YOUR_LICENSE_KEY  (enter your actual key first time only)
- min_area: 500 (3D cells are larger)
- max_distance: 25
- window_size: 10 (smaller for 3D to save memory)

Step 3: Run Ultrack Tracking (SUBSEQUENT RUNS)
Parameters:
- label_suffixes: _cp_labels.tif,_watershed_labels.tif
- gurobi_license: (leave empty - already activated)
- min_area: 500
- max_distance: 25
- window_size: 10

Output:
- cells3d_ultrack_tracked.tif
- cells3d_tracks.csv
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'zarr'" (or other packages)

**Problem**: The tracking script fails with missing module errors (zarr, torch, tifffile, etc.)

**Cause**: You're using an older ultrack environment that was created before these dependencies were added.

**Solution**: The system automatically detects and installs missing packages. If you see this error:

1. **First run**: The system will automatically install missing packages
   ```
   ⚠ Found X missing package(s) in 'ultrack' environment
   Installing missing packages...
     Installing zarr...
     ✓ Installed zarr
   ```

2. **If auto-install fails**: Manually install the missing package
   ```bash
   conda run -n ultrack pip install zarr>=3.0.0
   conda run -n ultrack pip install torch torchvision
   conda run -n ultrack pip install tifffile
   ```

3. **Nuclear option**: Recreate the environment
   ```bash
   conda env remove -n ultrack
   # Then run tracking again - environment will be recreated with all packages
   ```

### "No label files found"
- Check that your label files ha, fragmented tracks

This is the **most common issue**. Try these fixes **in order**:

1. **Increase `max_distance`** (MOST IMPORTANT):
   - Measure how far cells move between frames in napari
   - Set to 2-3× the maximum displacement
   - Example: If cells move 15-20px between frames, use `max_distance: 40-60`

2. **Increase penalty weights**:
   - `appear_weight`: -0.01 to -0.5 (penalize new tracks)
   - `disappear_weight`: -0.01 to -0.5 (penalize track ends)

3. **Check segmentation quality**:
   - Are cells consistently detected across frames?
   - Large gaps in detection → fragmentation
   - Use ensemble with 2-3 segmentation methods for robustness

4. **Lower `min_area`** if cells are small:
   - Try 50-100 instead of default 100
   - Ensures small cells aren't filtered out
### "ultrack environment creation failed"
- Ensure conda or mamba is installed
- Check your internet connection
- Try manually: `conda create -n ultrack python=3.11`

### Tracking produces many short tracks
- Increase `disappear_weight` (make more negative, e.g., -0.5)
- Increase `max_distance`
- Check if segmentation is consistent across frames

### Memory errors
- Decrease `window_size` (e.g., 10 or 5)
- Process fewer frames at once
- Use a machine with more RAM

### Gurobi license activation fails
- Verify you have an internet connection during activation
- Check that your license key is correct
- Ensure you're on the network specified in your license

## References

- [ultrack GitHub](https://github.com/royerlab/ultrack)
- [ultrack Documentation](https://royerlab.github.io/ultrack/)
- [ultrack Paper](https://arxiv.org/pdf/2308.04526)
- [Segmentation Ensemble Example](https://royerlab.github.io/ultrack/examples/multi_color_ensemble/multi_color_ensemble.html)

## Citation

If you use ultrack in your research, please cite:

```
@inproceedings{bragantini2024ucmtracking,
  title={Large-scale multi-hypotheses cell tracking using ultrametric contours maps},
  author={Bragantini, Jordão and Lange, Merlin and Royer, Loïc},
  booktitle={European Conference on Computer Vision},
  pages={36--54},
  year={2024},
  organization={Springer}
}
```

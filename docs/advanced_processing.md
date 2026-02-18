# CAREamics Denoising

## Overview

Image denoising using **CAREamics** (Content-Aware Image Restoration). This processing function removes noise from microscopy images using deep learning-based methods, including Noise2Void (N2V) and CARE models.

## Features

- **Multiple Model Support**: Works with Noise2Void, CARE, and custom trained models
- **2D and 3D Data**: Handles both 2D and 3D microscopy images
- **Tile-Based Processing**: Efficiently processes large images using tiling
- **Test-Time Augmentation**: Optional TTA for improved results
- **Automatic Environment Management**: Creates dedicated environment if needed

## Installation

CAREamics will automatically install into a dedicated `careamics-env` environment when first used.

## Parameters

### `checkpoint_path` (string, required)
Path to the CAREamics model checkpoint file (.ckpt).

**How to obtain**:
- Train your own model using CAREamics
- Download pre-trained models from CAREamics model zoo
- Use models provided by your lab/collaborators

### `tile_size_x` (int, default: 32, range: 16-512)
Tile size in X dimension for processing.

### `tile_size_y` (int, default: 32, range: 16-512)
Tile size in Y dimension for processing.

### `tile_size_z` (int, default: 0, range: 0-256)
Tile size in Z dimension (for 3D data). Set to 0 for 2D images.

### `batch_size` (int, default: 1, range: 1-16)
Number of tiles to process simultaneously.
- Increase for faster processing (requires more memory)
- Decrease if you encounter out-of-memory errors

### `use_tta` (bool, default: True)
Use test-time augmentation for better denoising results.
- True: Better quality but slower
- False: Faster but potentially lower quality

### `force_dedicated_env` (bool, default: False)
Force using dedicated environment even if CAREamics is available in the main environment.

## Usage

### Prerequisites

1. **Trained CAREamics model**: You need a checkpoint file (.ckpt)
2. **Compatible image data**: Must match the dimensions the model was trained on

### In napari-tmidas

1. Open **Plugins > T-MIDAS > Image Processing**
2. Browse to your folder containing noisy images
3. Select **"CAREamics Denoise (N2V/CARE)"**
4. Configure parameters:
   - Set `checkpoint_path` to your model file
   - Adjust tile sizes based on your image dimensions
   - Enable `use_tta` for better results
5. Click **"Start Batch Processing"**

## Output

**Suffix**: `_denoised`

Produces denoised images with:
- Same dimensions as input
- Same data type as input
- Reduced noise while preserving structures

## Tips

1. **Model Selection**: Use models trained on similar data types
2. **Tile Sizes**: Match the receptive field of your model
3. **TTA**: Use for final results, disable for quick tests
4. **Memory**: Reduce batch_size and tile sizes if out of memory

## Credits

- [CAREamics GitHub](https://github.com/CAREamics/careamics)
- [CAREamics Documentation](https://careamics.github.io/)

---

# Spotiflow Spot Detection

## Overview

Accurate spot detection for fluorescence microscopy using **Spotiflow**, a deep learning-based method designed specifically for detecting fluorescent spots and puncta.

## Features

- **Pre-trained Models**: Multiple models for different microscopy types
- **2D and 3D Support**: Handles both 2D and 3D image stacks
- **High Accuracy**: Optimized for sub-pixel spot localization
- **Label Output**: Can generate label masks from detected spots
- **Automatic Environment Management**: Dedicated environment created if needed

## Installation

Spotiflow will automatically install into a dedicated `spotiflow-env` environment when first used.

## Pre-trained Models

- **`general`**: General-purpose model for various spot types
- **`smfish`**: Optimized for smFISH (single-molecule FISH)
- **`spots_3d`**: For 3D volumetric spot detection
- **Custom models**: Provide path to your trained model

## Parameters

### `pretrained_model` (string, default: "general")
Name of the pre-trained Spotiflow model or path to custom model.

### `model_path` (string, default: "")
Path to custom trained Spotiflow model (overrides pretrained_model).

### `prob_thresh` (float, default: 0.4, range: 0.0-1.0)
Probability threshold for spot detection.
- Lower: More spots detected (may include false positives)
- Higher: Fewer, higher-confidence spots

### `spot_radius` (int, default: 3, range: 1-20)
Radius for creating label masks around detected spots (in pixels).

### `force_cpu` (bool, default: False)
Force CPU processing even if GPU is available.

### `generate_labels` (bool, default: True)
Generate label masks from detected spots (vs. just coordinates).

## Usage

### In napari-tmidas

1. Open **Plugins > T-MIDAS > Image Processing**
2. Browse to folder with fluorescence images
3. Select **"Spotiflow Spot Detection"**
4. Configure parameters based on your data
5. Click **"Start Batch Processing"**

## Output

- **With labels**: Label image with each spot as unique region
- **Without labels**: Returns detected spot coordinates

## Tips

1. **Model Selection**: Choose model matching your imaging modality
2. **Threshold Adjustment**: Lower threshold for dim spots, raise for bright spots
3. **3D Data**: Use `spots_3d` model for volumetric data
4. **GPU**: Enable GPU for faster processing of large datasets

## Credits

- [Spotiflow GitHub](https://github.com/weigertlab/spotiflow)
- Developed by Weigert Lab

---

# SciPy Filters

## Overview

Image processing functions using SciPy's ndimage module for filtering and morphological operations.

## Functions

### Resize Labels (Nearest, SciPy)
**Suffix**: `_scaled`

Resize label images while preserving label integrity using nearest-neighbor interpolation.

**Parameters**:
- `scale_factor` (float, default: 1.0, range: 0.01-10.0): Scaling factor
  - < 1.0: Shrink labels
  - > 1.0: Enlarge labels
  - 1.0: No change

**Use case**: Scale segmentation masks to match differently sized images.

**Technical details**:
- Uses `scipy.ndimage.zoom` with `grid_mode=True`
- Preserves label values exactly (no interpolation)
- Centers resized objects in original array dimensions
- Maintains spatial relationships

---

### Subdivide Labels into 3 Layers
**Suffix**: `_layers`

Subdivide each labeled object into 3 concentric layers (outer, middle, inner).

**Parameters**:
- `is_half_body` (bool, default: False): Enable for objects cut in half

**Use case**: Analyze cell compartments or tissue layers separately.

**Output**: Single label image where each layer gets unique ID:
- Original object ID=1 → outer layer=1, middle layer=1001, inner layer=2001
- Original object ID=2 → outer layer=2, middle layer=1002, inner layer=2002

**Technical details**:
- Uses distance transform for layer calculation
- Layers are approximately equal volume
- Half-body mode creates layers as if object were complete

---

## Installation

```bash
pip install scipy
```

---

# Scikit-Image Filters

## Overview

Image enhancement and filtering functions using scikit-image library.

## Functions

### CLAHE (Adaptive Histogram Equalization)
**Suffix**: `_clahe`

Apply Contrast Limited Adaptive Histogram Equalization to enhance local contrast.

**Parameters**:
- `clip_limit` (float, default: 0.01): Contrast clipping limit (0.001-0.1)
  - Higher: More contrast, may amplify noise
  - Lower: Subtler enhancement
- `kernel_size` (int, default: 0): Local region size (0=auto)
  - Smaller: Enhance small features
  - Larger: Enhance large features

**Use case**: Enhance weak bright features in dark images (membranes, fine structures).

**Technical details**:
- Works locally, prevents over-brightening of background
- Auto-calculates kernel size if not specified
- Much better than global histogram equalization for microscopy

---

### Additional scikit-image Filters

Other available filters include:
- **Gaussian Blur**: Smooth images
- **Median Filter**: Remove salt-and-pepper noise
- **Edge Detection**: Sobel, Canny edge detectors
- **Morphological Operations**: Opening, closing, dilation, erosion

---

## Installation

```bash
pip install scikit-image
```

---

# File Compression

## Overview

Compress processed images using Zstandard compression for efficient storage.

## Function: Compress with Zstandard
**Suffix**: `_compressed` (file becomes `.zst`)

**Parameters**:
- `remove_source` (bool, default: False): Delete original after compression
- `compression_level` (int, default: 3, range: 1-22):
  - 1-3: Fast compression, larger files
  - 4-10: Balanced
  - 11-19: Better compression, slower
  - 20-22: Maximum compression (ultra mode)

**Use case**: Save disk space for large datasets.

## Installation

Zstandard must be installed system-wide:

### Linux
Pre-installed on most distributions

### macOS
```bash
brew install zstd
```

### Windows
```bash
pip install zstandard
```

## Usage

Run compression as a post-processing step after other processing functions.

## Tips

1. **Compression Level**: Level 3 offers good balance of speed and compression
2. **Remove Source**: Enable only after verifying compressed files are valid
3. **Batch Processing**: Compress entire folders efficiently
4. **Decompression**: Use `pzstd -d filename.zst` to decompress

---

# Colocalization Analysis

## Overview

Analyze colocalization between labeled regions across multiple image channels.

## Use Cases

- Quantify overlap between different cellular markers
- Analyze protein-protein colocalization
- Study spatial relationships in tissue sections
- Count nested structures (e.g., nuclei within cells)

## Typical Workflow

1. **Multi-channel imaging**: Acquire images with multiple fluorescent markers
2. **Segmentation**: Label regions in each channel (cells, nuclei, organelles)
3. **Colocalization analysis**: Quantify overlaps between channels

## Expected Data Structure

Multi-channel label images where:
- Channel 0: Reference channel (larger structures, e.g., cells)
- Channel 1: Secondary channel (smaller structures, e.g., nuclei)
- Channel 2: Optional tertiary channel (sub-nuclear structures)

## Outputs

Statistical tables containing:
- ROI counts per channel
- Colocalization counts
- Optional size measurements
- Median/total sizes of colocalizing regions

## Key Features

- **Boolean Masking**: Determines overlap using binary masks
- **Unique Label Counting**: Counts distinct objects in overlapping regions
- **Size Analysis**: Optional measurement of ROI sizes
- **Multi-level Analysis**: Supports 2 or 3 channels

## Technical Notes

- Reference channel typically contains larger structures
- Other channels contain smaller, potentially nested structures
- Each unique label ID represents one instance
- Zero (background) is always excluded from analysis

---

## See Also

- [Basic Processing Functions](basic_processing.md)
- [Cellpose Segmentation](cellpose_segmentation.md)
- [Regionprops Analysis](regionprops_analysis.md)
- [Grid View Overlay](grid_view_overlay.md)

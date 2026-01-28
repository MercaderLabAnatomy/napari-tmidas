# Spotiflow Spot Detection

Accurate and efficient spot detection for fluorescence microscopy using deep learning.

## Overview

Spotiflow is a deep learning-based method for detecting spots (particles, puncta, foci) in fluorescence microscopy images. It provides:

- **High accuracy**: State-of-the-art detection performance across various microscopy modalities
- **Subpixel localization**: Precise spot coordinates with sub-pixel accuracy
- **3D support**: Native 3D spot detection for volumetric data
- **Multiple pretrained models**: Ready-to-use models for various applications
- **Fast inference**: Efficient GPU-accelerated processing

This enables automated detection of spots like mRNA molecules (smFISH), protein puncta, vesicles, or any point-like structures in microscopy images.

## Requirements

- **Input image type**: Fluorescence microscopy (widefield, confocal, light-sheet)
- **Dimensionality**: 2D (YX), 3D (ZYX), or time-lapse (TYX, TZYX)
- **Spot characteristics**: Point-like structures, typically 2-10 pixels in diameter
- **Supported formats**: TIFF, Zarr

## Model Information

**Available Pretrained Models:**
- **general**: General-purpose 2D spot detector (recommended starting point)
- **hybiss**: Optimized for high-density spots (e.g., multiplexed smFISH)
- **synth_complex**: For complex backgrounds and low SNR
- **synth_3d**: 3D spot detection for volumetric data
- **smfish_3d**: Specialized for 3D single-molecule FISH

**Architecture:**
- Based on U-Net architecture with custom spot detection heads
- Outputs probability heatmaps and spot coordinates
- Subpixel localization using learned spot properties

**References:**
- Weigert et al. (2022) "Spotiflow: accurate and efficient spot detection for fluorescence microscopy with deep networks"
- https://github.com/weigertlab/spotiflow

## Installation

Spotiflow runs in a dedicated virtual environment to avoid dependency conflicts. The first time you use the function:

1. The plugin will automatically create a dedicated environment
2. Download the selected pretrained model automatically
3. Install all required dependencies (PyTorch, Spotiflow, etc.)

This one-time setup may take 5-10 minutes depending on your internet connection and hardware.

## Usage

### Basic Usage

```python
import napari
from napari_tmidas.processing_functions.spotiflow_detection import spotiflow_spot_detection

# Load your image with spots
viewer = napari.Viewer()
image_layer = viewer.open('spots_image.tif')

# Get the image data
image = image_layer.data

# Run spot detection
spot_labels = spotiflow_spot_detection(
    image,
    pretrained_model='general',
    prob_thresh=None,  # Auto threshold
    subpixel=True
)

# Add results to viewer
viewer.add_labels(spot_labels, name='Detected Spots')
```

### Using Batch Processing

1. Open napari with the tmidas plugin
2. Go to `Plugins > napari-tmidas > Batch Processing`
3. Select your input files (fluorescence TIFF images)
4. Choose function: `Spotiflow Spot Detection`
5. Configure parameters:
   - **pretrained_model**: Select appropriate model for your data
   - **prob_thresh**: Leave empty for automatic, or set manually
   - **subpixel**: Enable for precise localization
6. Click `Run Processing`

### 3D Volumetric Data

For 3D spot detection (ZYX):

```python
# Load 3D volume
volume_3d = np.random.rand(50, 512, 512)  # (Z, Y, X)

# Detect 3D spots
spot_labels_3d = spotiflow_spot_detection(
    volume_3d,
    pretrained_model='synth_3d',  # Use 3D model
    prob_thresh=0.5,
    spot_radius=3
)
```

### Time-Lapse Data

For time-lapse data (TZYX):

```python
# Process time-lapse (each timepoint independently)
timelapse = np.random.rand(20, 512, 512)  # (T, Y, X)

spot_labels_timelapse = spotiflow_spot_detection(
    timelapse,
    pretrained_model='general',
    subpixel=True
)

# Result shape: (T, Y, X) with labeled spots at each timepoint
```

## Parameters

### pretrained_model (str, default: 'general')
Select the pretrained model to use.

**Options**:
- **general**: General-purpose 2D detector (good starting point)
- **hybiss**: High-density spots, multiplexed imaging
- **synth_complex**: Complex backgrounds, low signal-to-noise
- **synth_3d**: 3D volumetric spot detection
- **smfish_3d**: 3D single-molecule FISH

**Choosing a model**:
- Start with `general` for most 2D applications
- Use `synth_3d` or `smfish_3d` for 3D data
- Try `hybiss` for very dense spot patterns
- Use `synth_complex` for challenging, noisy images

### model_path (str, default: empty)
Path to a custom trained model folder.

**When to use**:
- You've trained your own Spotiflow model
- Using a model shared by another user
- Leave empty to use pretrained models

### prob_thresh (float, default: None)
Probability threshold for spot detection.

**Guidelines**:
- **None/0.0**: Automatic threshold (recommended)
- **0.3-0.5**: Permissive, detects more spots (may include false positives)
- **0.6-0.8**: Stringent, only high-confidence spots
- Start with automatic, then adjust if needed

### subpixel (bool, default: True)
Enable subpixel localization for precise spot coordinates.

**Effect**:
- **True**: More accurate spot positions (recommended)
- **False**: Faster processing, pixel-level accuracy

### peak_mode (str, default: 'fast')
Peak detection algorithm.

**Options**:
- **fast**: Faster but less precise
- **skimage**: More accurate but slower

### normalizer (str, default: 'percentile')
Image normalization method.

**Options**:
- **percentile**: Robust to outliers (recommended)
- **minmax**: Simple min-max scaling

### normalizer_low / normalizer_high (float)
Percentile values for normalization.

**Defaults**: 1.0 and 99.8
- Adjust if image has unusual intensity distribution
- Lower `normalizer_low` for very dark backgrounds
- Lower `normalizer_high` for saturated pixels

### n_tiles (str, default: 'auto')
Number of tiles for large image processing.

**Options**:
- **'auto'**: Automatically determine tiling
- **'(2,2)'**: Process in 2x2 tiles
- **'(4,4)'**: Process in 4x4 tiles

**When to adjust**:
- Use more tiles for very large images (>2048px)
- Reduce GPU memory usage
- May affect detection at tile boundaries

### spot_radius (int, default: 3)
Approximate radius of spots in pixels.

**Guidelines**:
- Measure typical spot size in your images
- Used for generating label masks from detections
- Doesn't affect detection accuracy, only visualization

### force_cpu (bool, default: False)
Force CPU processing even if GPU is available.

**When to use**:
- Testing/debugging
- GPU memory issues
- Note: CPU is significantly slower

## Output

The output is a label image with the same spatial dimensions as input:
- **2D (YX)**: Label mask with each spot as a unique label
- **3D (ZYX)**: 3D label volume with detected spots
- **Time-lapse (TYX/TZYX)**: Labels at each timepoint

**Label values**:
- 0: Background
- 1, 2, 3, ...: Individual spots

**Additional outputs** (can be extracted if needed):
- Spot coordinates (Y, X) or (Z, Y, X)
- Detection probabilities
- Spot properties (intensity, size, etc.)

## Typical Workflow

1. **Acquire fluorescence images** with spots/puncta
2. **Run Spotiflow detection** with appropriate model
3. **Validate results** visually or with ground truth
4. **Extract spot properties** using RegionProps
5. **Perform downstream analysis** (counting, tracking, clustering)

### Example: Spot Detection + Quantification Pipeline

```python
import napari
import pandas as pd
from napari_tmidas.processing_functions.spotiflow_detection import spotiflow_spot_detection
from skimage.measure import regionprops_table

viewer = napari.Viewer()

# Step 1: Load image
image = viewer.open('spots_image.tif').data

# Step 2: Detect spots
spots = spotiflow_spot_detection(
    image,
    pretrained_model='general',
    prob_thresh=None,
    subpixel=True
)

# Step 3: Extract spot properties
props = regionprops_table(
    spots, 
    intensity_image=image,
    properties=['label', 'centroid', 'area', 'mean_intensity']
)
df = pd.DataFrame(props)

# Step 4: Analyze
print(f"Detected {len(df)} spots")
print(f"Mean intensity: {df['mean_intensity'].mean():.2f}")

# Visualize
viewer.add_image(image, name='Original')
viewer.add_labels(spots, name='Detected Spots')
```

### Example: smFISH Analysis

```python
# Detect mRNA molecules in single-molecule FISH image
mrna_spots = spotiflow_spot_detection(
    smfish_image,
    pretrained_model='smfish_3d',  # For 3D smFISH
    prob_thresh=0.5,
    spot_radius=2
)

# Count spots per cell (if you have cell segmentation)
from skimage.measure import regionprops

cell_spot_counts = {}
for region in regionprops(cell_labels):
    cell_id = region.label
    cell_mask = cell_labels == cell_id
    spots_in_cell = np.sum((mrna_spots > 0) & cell_mask)
    cell_spot_counts[cell_id] = spots_in_cell

print("mRNA counts per cell:", cell_spot_counts)
```

## Hardware Recommendations

### GPU (Recommended)
- **VRAM**: Minimum 2 GB, 4 GB or more recommended
- **Supported**: CUDA-compatible NVIDIA GPUs
- Processing speed: ~1-5 seconds per image (512x512)
- Batch processing possible with sufficient memory

### CPU (Fallback)
- Will automatically fall back to CPU if no GPU is available
- Processing speed: ~30-60 seconds per image (512x512)
- Still usable for small-scale analysis
- Recommended for batch overnight processing

## Troubleshooting

### "Spotiflow environment not found"
**Solution**: The environment will be created automatically on first use. If creation fails:
1. Check internet connection (for downloading models)
2. Ensure sufficient disk space (~2-3 GB)
3. Check conda/mamba is properly installed

### Too Many False Positives
**Solutions**:
1. Increase `prob_thresh` (e.g., from 0.4 to 0.6)
2. Try a different model (e.g., `general` → `synth_complex`)
3. Check image quality and contrast
4. Consider denoising image first (CAREamics)

### Missing True Spots
**Solutions**:
1. Decrease `prob_thresh` (e.g., from 0.5 to 0.3)
2. Set `prob_thresh=None` for automatic threshold
3. Try `hybiss` model for high-density spots
4. Check normalizer settings

### Poor Detection Quality
**Common causes**:
- Wrong model for your data type
- Incorrect normalization
- Poor image quality (low SNR)

**Solutions**:
1. Try different pretrained models
2. Adjust `normalizer_low` and `normalizer_high`
3. Denoise image first if SNR is low
4. Ensure spots are in the 2-10 pixel size range
5. Consider training custom model on your data

### Out of Memory Errors
**Solutions**:
1. Increase `n_tiles` (e.g., from 'auto' to '(4,4)')
2. Enable `force_cpu` (slower but no memory limit)
3. Process smaller image regions
4. Reduce image size (downsampling)

### Spots Detected at Tile Boundaries
**Solutions**:
1. Use fewer tiles if possible
2. Increase tile overlap (Spotiflow handles this automatically)
3. Process full image if GPU memory allows

## Tips & Best Practices

1. **Model selection**:
   - Always start with `general` for 2D data
   - Use `synth_3d` or `smfish_3d` for 3D data
   - Test multiple models and compare results
   - Consider training custom model for unusual data

2. **Parameter tuning**:
   - Leave `prob_thresh=None` initially
   - Adjust only if you get too many false positives/negatives
   - Enable `subpixel` for precise localization
   - Use default normalization for most cases

3. **Quality control**:
   - Visually inspect results on subset of images
   - Compare with manual annotations if available
   - Check detection at image edges and tile boundaries
   - Verify spot counts match expectations

4. **Performance optimization**:
   - Use GPU for interactive analysis
   - Batch process large datasets overnight
   - Increase `n_tiles` for very large images
   - Consider downsampling if spot size allows

5. **Integration with analysis**:
   - Use with RegionProps for spot quantification
   - Combine with cell segmentation for single-cell analysis
   - Track spots over time for dynamic studies
   - Export coordinates for statistical analysis

## Training Custom Models

For specialized applications, you can train custom Spotiflow models:

```python
# Training requires Spotiflow installation and training data
# See: https://github.com/weigertlab/spotiflow

from spotiflow.model import Spotiflow
from spotiflow.utils import normalize

# Prepare training data (images + spot coordinates)
# Train model
model = Spotiflow(config)
model.train(train_images, train_spots, val_images, val_spots)

# Save for use in napari-tmidas
model.save('path/to/custom_model')
```

Then use in napari-tmidas:
```python
spots = spotiflow_spot_detection(
    image,
    model_path='/path/to/custom_model',
    prob_thresh=0.5
)
```

## Model Citation

If you use Spotiflow in your research, please cite:

```bibtex
@article{weigert2022spotiflow,
  title={Spotiflow: accurate and efficient spot detection for fluorescence microscopy with deep networks},
  author={Weigert, Martin and Schmidt, Uwe and Haase, Robert and Sugawara, Ko and Myers, Gene},
  journal={bioRxiv},
  year={2022}
}
```

## Additional Resources

- **Spotiflow GitHub**: https://github.com/weigertlab/spotiflow
- **Documentation**: https://weigertlab.github.io/spotiflow/
- **Paper**: https://doi.org/10.1101/2022.XXX
- **Forum**: https://forum.image.sc/ (tag: spotiflow)

## See Also

- [CAREamics Denoising](careamics_denoising.md) - Denoise images before spot detection
- [RegionProps Analysis](regionprops_analysis.md) - Quantify detected spots
- [Trackastra Tracking](trackastra_tracking.md) - Track spots over time
- [Advanced Processing](advanced_processing.md) - More image processing functions

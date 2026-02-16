# Convpaint Prediction

## Overview

Semantic and instance segmentation using pretrained **napari-convpaint** models. This processing function enables batch inference with custom-trained convpaint models for TZYX time-series data, processing each timepoint independently.

Convpaint uses feature extractors (like DINO, VGG, Gaussian filters) to segment images based on annotations. This function allows you to apply your pretrained models to new datasets at scale.

## Features

- **Pretrained Model Support**: Load custom `.pkl` model checkpoints
- **Batch Processing**: Process entire folders of images automatically
- **Time Series Support**: Handles TZYX data by processing each timepoint independently
- **Semantic & Instance Output**: Choose between semantic classes or instance labels
- **Memory Optimization**: Configurable downsampling to reduce GPU memory usage
- **CPU Fallback**: Automatic or manual CPU execution for GPU compatibility issues
- **Background Removal**: Configurable background label removal
- **Automatic Environment Management**: Creates dedicated environment if convpaint not installed

## Installation

The plugin automatically creates a dedicated `convpaint-env` environment on first use. No manual installation required.

### Manual Installation (Optional)

If you want napari-convpaint in your main environment:

```bash
pip install napari-convpaint
```

## Parameters

### `model_path` (string, required)
Path to the pretrained convpaint model (`.pkl` file).

**Example**: `/path/to/your/model.pkl`

**How to train a model**: Use napari-convpaint to train models on annotated data. See [napari-convpaint documentation](https://github.com/guiwitz/napari-convpaint) for training instructions.

### `image_downsample` (int, default: 2, range: 1-8)
Downsampling factor for processing. Output is automatically upsampled to match input dimensions.

**Recommendations**:
- `1`: No downsampling (highest quality, most memory)
- `2`: 2x downsampling (recommended for most cases)
- `4`: 4x downsampling (for very large images or limited GPU memory)
- `8`: 8x downsampling (extreme memory constraints)

**Memory impact**: Higher downsampling reduces memory usage quadratically (2x = 4× less memory, 4x = 16× less memory).

### `output_type` (string, default: "semantic")
Type of output labels.

**Options**:
- `"semantic"`: Each class has the same label value
  - Example: All class 1 objects = 1, all class 2 objects = 2
  - Use for: Classification maps, class-based analysis
  
- `"instance"`: Each connected component gets a unique label
  - Example: Object 1 = 1, Object 2 = 2, Object 3 = 3 (even if same class)
  - Use for: Counting objects, tracking, individual object analysis
  - Processing: 3D volumes use 3D connected components, time series process per-timepoint

### `background_label` (int, default: 1, range: 0-255)
Label ID representing the background class in the model output.

All pixels with this label value will be set to 0 (standard background) in the output.

**Common values**:
- `1`: Default (most convpaint models use 1 for background)
- `0`: If your model already uses 0 for background
- Other: If your model uses a different class ID for background

### `use_cpu` (bool, default: False)
Force CPU execution even if GPU is available.

**When to enable**:
- GPU not compatible with PyTorch (e.g., very new GPUs like RTX 5000 series)
- Out of GPU memory errors
- Testing/debugging

**Note**: CPU execution is significantly slower but works on any hardware.

### `force_dedicated_env` (bool, default: False)
Force using the dedicated convpaint environment even if napari-convpaint is installed in the main environment.

## Usage Examples

### Example 1: Basic Semantic Segmentation

Process 2D or 3D images with a pretrained model:

```python
# Parameters:
# - model_path: /home/user/models/cell_segmentation.pkl
# - image_downsample: 2
# - output_type: semantic
# - background_label: 1

# Input:  sample.tif (2048×2048 pixels)
# Output: sample_convpaint_labels.tif (semantic classes)
```

### Example 2: Instance Segmentation for Cell Counting

Get individual cell labels for counting:

```python
# Parameters:
# - model_path: /home/user/models/nuclei_model.pkl
# - image_downsample: 2
# - output_type: instance
# - background_label: 1

# Input:  nuclei.tif (YX)
# Output: nuclei_convpaint_labels.tif (each nucleus has unique ID)
```

### Example 3: Time-Lapse 3D Data (TZYX)

Process 4D time series (each timepoint processed independently):

```python
# Parameters:
# - model_path: /home/user/models/organoid_3d.pkl
# - image_downsample: 2
# - output_type: instance
# - background_label: 1

# Input:  timelapse.tif (20 timepoints × 50 Z-slices × 1024×1024)
# Output: timelapse_convpaint_labels.tif (TZYX with instance labels per timepoint)
```

### Example 4: High-Resolution with Memory Constraints

Use higher downsampling for very large images:

```python
# Parameters:
# - model_path: /home/user/models/tissue.pkl
# - image_downsample: 4
# - output_type: semantic
# - background_label: 1
# - use_cpu: True (if GPU runs out of memory)

# Input:  whole_slide.tif (10000×10000 pixels)
# Output: whole_slide_convpaint_labels.tif
```

## Supported Data Formats

### Input Dimensions

| Format | Description | Example Shape |
|--------|-------------|---------------|
| **YX** | 2D image | (1024, 1024) |
| **ZYX** | 3D volume (Z-stack) | (50, 512, 512) |
| **TYX** | 2D time series | (100, 512, 512) |
| **TZYX** | 3D time series | (20, 50, 512, 512) |

### Dimension Detection

The function automatically detects dimension order using these heuristics:
- **3D data**: First dimension < 100 → ZYX (3D volume)
- **3D data**: First dimension ≥ 100 → TYX (2D time series)
- **4D data**: Always treated as TZYX

## Output Files

Output files are saved in the same directory as input with suffix `_convpaint_labels.tif`:

```
Input:  my_image.tif
Output: my_image_convpaint_labels.tif
```

**Output format**:
- Data type: `uint32` (supports up to 4 billion unique labels)
- Compression: zlib
- Metadata: Preserved from input when possible
- Background: Always 0 (after background_label removal)

## Workflow Integration

### Complete Segmentation Pipeline

```
1. File Selection → Select folder with images
2. Convpaint Prediction → Apply pretrained model
   - Set model_path to your .pkl file
   - Choose output_type (semantic or instance)
   - Adjust image_downsample if needed
3. (Optional) Post-processing:
   - Remove small labels
   - Apply morphological operations
4. Quantification → Extract regionprops
5. Analysis → Colocalization, tracking, etc.
```

### Training → Inference Workflow

1. **Training** (in napari-convpaint):
   - Load training images
   - Create annotations (paint labels)
   - Train model with desired feature extractor
   - Save model as `.pkl` file

2. **Batch Inference** (in napari-tmidas):
   - Load test images in batch processing
   - Select "Convpaint Prediction"
   - Point to trained `.pkl` model
   - Process entire folder

## Technical Details

### Processing Strategy

- **2D (YX)**: Direct processing
- **3D (ZYX)**: Processed as 3D volume (not slice-by-slice)
- **Time Series (TYX/TZYX)**: Each timepoint processed independently
  - Prevents temporal artifacts
  - Allows parallel processing
  - Matches other AI method behavior (Cellpose, CAREamics)

### Instance Segmentation Method

When `output_type="instance"`:
1. Process semantic segmentation normally
2. Apply connected components analysis:
   - **2D**: 8-connected (3×3 neighborhood)
   - **3D**: 26-connected (3×3×3 neighborhood)
3. Each class processed separately to avoid label conflicts
4. Unique labels assigned across all classes

### Environment Management

The function automatically manages a dedicated Python environment:

**Environment location**: `~/.napari-tmidas/envs/convpaint/`

**First use**:
1. Detects if napari-convpaint is available
2. If not, creates dedicated environment
3. Installs: napari-convpaint, PyTorch (with CUDA if available), dependencies

**Subsequent uses**:
- Reuses existing environment
- No reinstallation needed

**Manual environment recreation**:
```python
from napari_tmidas.processing_functions.convpaint_env_manager import recreate_convpaint_env
recreate_convpaint_env()
```

## Troubleshooting

### GPU Not Compatible Error

**Problem**: `CUDA error: no kernel image is available for execution`

**Solution**: Enable CPU mode
- Set `use_cpu=True` in parameters
- GPU is too new for PyTorch version

### Out of Memory Error

**Problem**: GPU runs out of memory during processing

**Solutions**:
1. Increase `image_downsample` (try 4 or 8)
2. Enable `use_cpu=True` (slower but no memory limit)
3. Process smaller images or crops

### Model Not Loading

**Problem**: "Model file not found" or loading errors

**Checklist**:
- Verify `.pkl` file exists at specified path
- Check file permissions
- Ensure model was saved with compatible napari-convpaint version
- Try absolute path instead of relative path

### Background Not Removed

**Problem**: Background pixels not set to 0

**Solution**: Adjust `background_label` parameter
- Check your model's class definitions
- Use napari to inspect output and identify background class ID
- Set `background_label` to that ID

### Slow Processing

**Problem**: Processing takes very long

**Optimizations**:
1. Ensure GPU is being used (`use_cpu=False`)
2. Increase `image_downsample` for faster processing
3. Check GPU utilization: `nvidia-smi`
4. Close other GPU-using applications

## Performance Tips

### Speed Optimization

- **Use GPU**: 10-100× faster than CPU
- **Increase downsampling**: 2× → 4× gives ~4× speedup
- **Process in batches**: Use batch processing widget
- **Semantic output**: Faster than instance (no connected components)

### Memory Optimization

- **Reduce image size**: Crop or bin images before processing
- **Increase downsampling**: Reduces memory quadratically
- **Process timepoints sequentially**: Automatic for TZYX
- **Use CPU**: Unlimited memory (but slower)

### Quality vs Speed Tradeoff

| Downsample | Speed | Memory | Quality |
|------------|-------|--------|---------|
| 1× | 1× (baseline) | High | Best |
| 2× | ~4× faster | 4× less | Good |
| 4× | ~16× faster | 16× less | Fair |
| 8× | ~64× faster | 64× less | Rough |

## Model Training Resources

To train custom convpaint models, see:
- [napari-convpaint GitHub](https://github.com/guiwitz/napari-convpaint)
- [Convpaint Tutorials](https://guiwitz.github.io/napari-convpaint/)

**Training tips**:
- Use diverse training data representative of your test images
- Choose appropriate feature extractor (DINO, VGG, Gaussian)
- Test on held-out validation data
- Save models with descriptive names

## Citation

If you use convpaint in your research, please cite:

```
Wigger, G. E. (2024). napari-convpaint: Interactive pixel classification with feature extractors. 
GitHub. https://github.com/guiwitz/napari-convpaint
```

## See Also

- [Cellpose Segmentation](cellpose_segmentation.md) - Deep learning instance segmentation
- [CAREamics Denoising](careamics_denoising.md) - AI-based denoising
- [Semantic to Instance](basic_processing.md#semantic-to-instance-segmentation) - Post-processing conversion
- [Batch Processing Guide](basic_processing.md) - General batch processing workflow

# CAREamics Denoising

Content-aware image restoration and enhancement for microscopy images using deep learning.

## Overview

CAREamics (Content-Aware Restoration of Microscopy ImageS) is a deep learning framework for denoising, deconvolution, and content restoration of microscopy images. It implements several state-of-the-art algorithms including:

- **Noise2Void (N2V)**: Self-supervised denoising without clean images
- **CARE**: Supervised restoration when paired training data is available
- **N2V2**: Improved Noise2Void with better performance
- **Struct N2V**: Structure-preserving Noise2Void variant

This enables you to restore noisy or low-SNR (Signal-to-Noise Ratio) microscopy images, revealing fine structures and improving downstream analysis.

## Requirements

- **Input image type**: Fluorescence microscopy (widefield, confocal, light-sheet)
- **Dimensionality**: 2D (YX), 3D (ZYX), or time-lapse (TYX, TZYX)
- **Model requirement**: Pre-trained CAREamics checkpoint file (.ckpt)
- **Supported formats**: TIFF, Zarr

## Model Information

**Supported Algorithms:**
- **Noise2Void**: Train on single noisy images, no ground truth needed
- **CARE**: Supervised learning with paired low/high quality images
- **N2V2**: Enhanced N2V with structured noise handling
- **Struct N2V**: Preserves image structures better than standard N2V

**Architecture Options:**
- U-Net based models (2D and 3D)
- Customizable depth and feature channels
- Support for various activation functions and normalization schemes

**References:**
- Krull et al. (2019) "Noise2Void - Learning Denoising from Single Noisy Images" CVPR
- Weigert et al. (2018) "Content-aware image restoration: pushing the limits of fluorescence microscopy" Nature Methods
- Höck et al. (2021) "N2V2 - Fixing Noise2Void Checkerboard Artifacts with Modified Sampling" arXiv

## Installation

CAREamics runs in a dedicated virtual environment to avoid dependency conflicts. The first time you use the function:

1. The plugin will automatically create a dedicated environment
2. Install all required dependencies (PyTorch, CAREamics, etc.)
3. Detect and configure GPU support (CUDA) if available

This one-time setup may take 5-10 minutes depending on your internet connection and hardware.

## Usage

### Training Your Own Model

Before using CAREamics for denoising, you need to train a model on your specific imaging conditions. CAREamics provides tools for training:

```python
# Training is typically done separately using CAREamics CLI or Python API
# See https://github.com/CAREamics/careamics for training documentation
```

### Basic Usage

```python
import napari
from napari_tmidas.processing_functions.careamics_denoising import careamics_denoise

# Load your noisy image
viewer = napari.Viewer()
noisy_layer = viewer.open('noisy_image.tif')

# Get the image data
noisy_image = noisy_layer.data

# Run denoising with your trained model
denoised = careamics_denoise(
    noisy_image,
    checkpoint_path='/path/to/your/model.ckpt',
    tile_size_x=128,
    tile_size_y=128,
    use_tta=True
)

# Add results to viewer
viewer.add_image(denoised, name='Denoised')
```

### Using Batch Processing

1. Train a CAREamics model on representative images from your dataset
2. Open napari with the tmidas plugin
3. Go to `Plugins > napari-tmidas > Batch Processing`
4. Select your input files (noisy TIFF images)
5. Choose function: `CAREamics Denoise (N2V/CARE)`
6. Configure parameters:
   - **checkpoint_path**: Path to your trained .ckpt file
   - **tile_size_x/y/z**: Tile dimensions for processing
   - **use_tta**: Enable test-time augmentation for better results
7. Click `Run Processing`

### 3D Volumetric Data

For 3D data (ZYX):

```python
# Load 3D volume
volume_3d = np.random.rand(50, 512, 512)  # (Z, Y, X)

# Denoise with 3D tiling
denoised_3d = careamics_denoise(
    volume_3d,
    checkpoint_path='/path/to/3d_model.ckpt',
    tile_size_z=32,
    tile_size_y=128,
    tile_size_x=128,
    tile_overlap_z=8,
    tile_overlap_y=16,
    tile_overlap_x=16
)
```

### Time-Lapse Data

For time-lapse data (TZYX):

```python
# Process time-lapse (each timepoint independently)
timelapse = np.random.rand(20, 50, 512, 512)  # (T, Z, Y, X)

denoised_timelapse = careamics_denoise(
    timelapse,
    checkpoint_path='/path/to/model.ckpt',
    tile_size_z=32,
    tile_size_y=128,
    tile_size_x=128,
    batch_size=4
)
```

## Parameters

### checkpoint_path (str, required)
Path to the trained CAREamics model checkpoint file (.ckpt).

**How to get a checkpoint**:
1. Train a model using CAREamics on your data
2. Use a pre-trained model if available for your imaging modality
3. Contact the community for shared models

### tile_size_x/y/z (int, default: varies)
Size of image tiles for processing.

**Guidelines**:
- Larger tiles: Faster but require more memory
- Smaller tiles: Slower but work with limited GPU memory
- Must match model's input size constraints
- **Typical values**: 64, 128, 256

### tile_overlap_x/y/z (int, default: 8-16)
Overlap between adjacent tiles to reduce edge artifacts.

**Guidelines**:
- Larger overlap: Better blending, slower processing
- Typical: 1/8 to 1/4 of tile size
- Minimum: 8 pixels

### batch_size (int, default: 1)
Number of tiles processed simultaneously.

**Guidelines**:
- Larger batches: Faster processing
- Limited by GPU memory
- Increase if you have spare GPU memory
- **Typical values**: 1-8 for large tiles, up to 32 for small tiles

### use_tta (bool, default: True)
Enable test-time augmentation for improved results.

**What it does**:
- Processes image with multiple augmentations (flips, rotations)
- Averages predictions for more robust denoising
- Increases processing time by ~8x
- Generally improves quality, especially near edges

### force_dedicated_env (bool, default: False)
Force using dedicated environment even if CAREamics is available in main environment.

## Output

The output has the same dimensions as the input:
- **2D (YX)**: Returns denoised 2D image
- **3D (ZYX)**: Returns denoised 3D volume
- **4D (TZYX/TYX)**: Returns denoised time-lapse

**Value range**: Preserves the input data range (not normalized to 0-1)

## Typical Workflow

1. **Acquire noisy images** with your standard imaging protocol
2. **Train a CAREamics model** on representative samples:
   - For N2V: Use noisy images only (no ground truth needed)
   - For CARE: Use paired noisy/clean images
3. **Run batch denoising** on your dataset
4. **Perform downstream analysis** (segmentation, tracking, measurements)

### Example: Denoising + Segmentation Pipeline

```python
import napari
from napari_tmidas.processing_functions.careamics_denoising import careamics_denoise
from napari_tmidas.processing_functions.cellpose_segmentation import cellpose_segmentation

viewer = napari.Viewer()

# Step 1: Load noisy image
noisy_image = viewer.open('noisy_cells.tif').data

# Step 2: Denoise with CAREamics
denoised = careamics_denoise(
    noisy_image,
    checkpoint_path='/path/to/n2v_model.ckpt',
    tile_size_x=256,
    tile_size_y=256,
    use_tta=True
)

# Step 3: Segment denoised image
labels = cellpose_segmentation(
    denoised,
    dim_order='YX',
    flow_threshold=0.4
)

# Visualize
viewer.add_image(noisy_image, name='Noisy Original')
viewer.add_image(denoised, name='Denoised')
viewer.add_labels(labels, name='Segmentation')
```

## Hardware Recommendations

### GPU (Strongly Recommended)
- **VRAM**: Minimum 4 GB, 8 GB or more recommended
- **Supported**: CUDA-compatible NVIDIA GPUs
- Processing speed: 10-50x faster than CPU depending on model size
- Larger batch sizes possible with more VRAM

### CPU (Fallback)
- Will automatically fall back to CPU if no GPU is available
- Processing is significantly slower
- Limited batch size due to RAM constraints
- May require smaller tile sizes

## Troubleshooting

### "CAREamics environment not found"
**Solution**: The environment will be created automatically on first use. If creation fails:
1. Check internet connection
2. Ensure sufficient disk space (~2-3 GB for environment)
3. Check for conflicting conda environments

### "Checkpoint file not found"
**Solution**: Verify the checkpoint path is correct
1. Use absolute path to the .ckpt file
2. Ensure the file exists and is readable
3. Check file permissions

### Out of Memory Errors
**Solutions**:
1. Reduce `tile_size_x`, `tile_size_y`, `tile_size_z`
2. Decrease `batch_size` to 1
3. Disable `use_tta` (faster but slightly lower quality)
4. Close other GPU applications
5. Use CPU if GPU memory is insufficient

### Poor Denoising Quality
**Common causes**:
- Model not trained on similar data to your images
- Incorrect tile size (doesn't match training configuration)
- Insufficient training data or epochs

**Solutions**:
1. Retrain model on your specific imaging conditions
2. Use more training data (100+ images for N2V)
3. Verify tile size matches model's expected input
4. Enable `use_tta` for better results
5. Check that input images are similar to training data

### Checkerboard Artifacts
**Cause**: Small tile overlap or edge effects

**Solutions**:
1. Increase tile overlap (e.g., from 8 to 32 pixels)
2. Use larger tile sizes if GPU memory allows
3. Enable `use_tta` to reduce artifacts
4. Consider using N2V2 which has reduced artifacts

### Processing is Very Slow
**Solutions**:
1. Ensure GPU is being used (check console output)
2. Increase batch size if GPU memory allows
3. Disable `use_tta` for 8x speedup (slight quality loss)
4. Use larger tile sizes to reduce overhead
5. Consider processing subset of frames for time-lapse data

## Tips & Best Practices

1. **Model training**:
   - Train on representative images from your dataset
   - Use at least 100 images for N2V training
   - Validate on held-out test set
   - Save multiple checkpoints and compare

2. **Tile configuration**:
   - Start with tile size matching training configuration
   - Use overlap of at least 1/8 of tile size
   - Larger tiles are more efficient if GPU memory allows

3. **Performance optimization**:
   - Enable GPU for 10-50x speedup
   - Increase batch size to utilize GPU fully
   - Disable TTA for quick previews, enable for final processing
   - Process in parallel using multiple GPUs if available

4. **Quality assessment**:
   - Compare denoised images with raw data visually
   - Check for over-smoothing or detail loss
   - Verify biological structures are preserved
   - Test on segmentation/analysis tasks to measure improvement

## Model Resources

### Pre-trained Models
- Check [CAREamics Model Zoo](https://github.com/CAREamics/careamics/wiki/Model-Zoo) for shared models
- Community-contributed models for various microscopy modalities
- Fine-tune existing models for your specific needs

### Training Resources
- **CAREamics Documentation**: https://careamics.github.io/
- **Training Tutorials**: https://github.com/CAREamics/careamics/tree/main/examples
- **Noise2Void Guide**: https://github.com/juglab/n2v

## Model Citation

If you use CAREamics in your research, please cite the relevant papers:

**Noise2Void:**
```bibtex
@inproceedings{krull2019noise2void,
  title={Noise2void-learning denoising from single noisy images},
  author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2129--2137},
  year={2019}
}
```

**CARE:**
```bibtex
@article{weigert2018content,
  title={Content-aware image restoration: pushing the limits of fluorescence microscopy},
  author={Weigert, Martin and Schmidt, Uwe and Boothe, Tobias and M{\"u}ller, Andreas and Dibrov, Alexandr and Jain, Akanksha and Wilhelm, Benjamin and Schmidt, Deborah and Broaddus, Coleman and Culley, Si{\^a}n and others},
  journal={Nature methods},
  volume={15},
  number={12},
  pages={1090--1097},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## Additional Resources

- **CAREamics GitHub**: https://github.com/CAREamics/careamics
- **Documentation**: https://careamics.github.io/
- **Paper**: Various papers for N2V, CARE, N2V2, etc.
- **Forum**: https://forum.image.sc/ (tag: careamics)

## See Also

- [Cellpose Segmentation](cellpose_segmentation.md) - Segment denoised images
- [Spotiflow Detection](spotiflow_detection.md) - Detect spots in denoised images  
- [VisCy Virtual Staining](viscy_virtual_staining.md) - Virtual staining for label-free imaging
- [Advanced Processing](advanced_processing.md) - More image processing functions

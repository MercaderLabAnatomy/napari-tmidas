# VisCy Virtual Staining

Virtual staining using the VSCyto3D deep learning model to predict nuclei and membrane channels from transmitted light (phase contrast or DIC) microscopy images.

## Overview

VisCy (Virtual Staining of Cells) is a deep learning-based method that transforms label-free transmitted light microscopy images into virtually stained fluorescence images. The VSCyto3D model specifically predicts:

- **Nuclei channel**: Virtual staining similar to nuclear dyes (e.g., Hoechst, DAPI)
- **Membrane channel**: Virtual staining similar to membrane markers (e.g., CellMask)

This enables you to obtain cell structure information without the need for fluorescent labels, avoiding phototoxicity and photobleaching concerns.

## Requirements

- **Input image type**: Phase contrast or DIC (Differential Interference Contrast) microscopy
- **Dimensionality**: 3D images (Z-stacks) with at least 15 Z-slices
- **Supported formats**: TIFF, Zarr
- **Time-lapse**: Supported (TZYX dimension order)

## Model Information

**VSCyto3D Model Specifications:**
- Architecture: fcmae-based U-Net
- Input: 1 channel (transmitted light)
- Output: 2 channels (nuclei + membrane)
- Z-stack requirement: 15 slices per batch
- Training data: 3D phase contrast images with corresponding fluorescent labels

**Reference:**
Guo et al. (2024) "Revealing architectural order with quantitative label-free imaging and deep learning"  
DOI: 10.7554/eLife.55502

## Installation

VisCy runs in a dedicated virtual environment to avoid dependency conflicts. The first time you use the function:

1. The plugin will automatically create a dedicated environment
2. Download the VSCyto3D model checkpoint (~900 MB)
3. Install all required dependencies

This one-time setup may take 5-10 minutes depending on your internet connection and hardware.

## Usage

### Basic Usage

```python
import napari
from napari_tmidas.processing_functions.viscy_virtual_staining import viscy_virtual_staining

# Load your phase contrast image
viewer = napari.Viewer()
phase_layer = viewer.open('phase_image.tif')

# Get the image data (must be ZYX or TZYX)
phase_image = phase_layer.data

# Run virtual staining
virtual_stain = viscy_virtual_staining(
    phase_image,
    dim_order='ZYX',
    output_channel='both'
)

# Add results to viewer
# virtual_stain has shape (Z, 2, Y, X)
viewer.add_image(virtual_stain[:, 0], name='Virtual Nuclei', colormap='green')
viewer.add_image(virtual_stain[:, 1], name='Virtual Membrane', colormap='magenta')
```

### Using Batch Processing

1. Open napari with the tmidas plugin
2. Go to `Plugins > napari-tmidas > Batch Processing`
3. Select your input files (phase contrast TIFF images)
4. Choose function: `VisCy Virtual Staining`
5. Configure parameters:
   - **dim_order**: Set to match your data (e.g., 'ZYX', 'TZYX')
   - **output_channel**: Choose 'both', 'nuclei', or 'membrane'
6. Click `Run Processing`

### Time-Lapse Data

For time-lapse data (TZYX):

```python
# Process time-lapse phase contrast
timelapse_phase = np.random.rand(10, 15, 512, 512)  # (T, Z, Y, X)

virtual_stain = viscy_virtual_staining(
    timelapse_phase,
    dim_order='TZYX',
    output_channel='both'
)

# Result shape: (T, Z, 2, Y, X)
# Separate channels
nuclei = virtual_stain[:, :, 0, :, :]  # (T, Z, Y, X)
membrane = virtual_stain[:, :, 1, :, :]  # (T, Z, Y, X)
```

## Parameters

### dim_order (str, default: 'ZYX')
Dimension order of the input image. Common options:
- `'ZYX'`: 3D single timepoint
- `'TZYX'`: 3D time-lapse
- `'YXZ'`: Alternative Z ordering (will be transposed)

### z_batch_size (int, default: 15)
Number of Z slices to process at once. **Must be 15** for the VSCyto3D model.
- The model architecture requires exactly 15 Z-slices as input
- Images are processed in batches of 15 slices
- If your image has more than 15 slices, it will be processed in multiple batches

### output_channel (str, default: 'both')
Which channel(s) to include in the output:
- `'both'`: Return both nuclei and membrane channels (shape: Z, 2, Y, X)
- `'nuclei'`: Return only nuclei channel (shape: Z, Y, X)
- `'membrane'`: Return only membrane channel (shape: Z, Y, X)

## Output

The output depends on the `output_channel` parameter:

**If output_channel='both':**
- Shape: `(Z, 2, Y, X)` or `(T, Z, 2, Y, X)` for time-lapse
- Channel 0: Virtual nuclei stain
- Channel 1: Virtual membrane stain

**If output_channel='nuclei' or 'membrane':**
- Shape: `(Z, Y, X)` or `(T, Z, Y, X)` for time-lapse
- Single channel with the selected virtual stain

**Value range:** 0.0 to 1.0 (normalized prediction probabilities)

## Typical Workflow

1. **Acquire phase contrast Z-stacks** (minimum 15 slices recommended)
2. **Run virtual staining** to get nuclei and membrane predictions
3. **Use nuclei channel for segmentation** (e.g., with Cellpose or watershed)
4. **Use membrane channel** for cell boundary analysis or colocalization
5. **Perform downstream analysis** (tracking, measurement, etc.)

### Example: Virtual Staining + Segmentation Pipeline

```python
import napari
from napari_tmidas.processing_functions.viscy_virtual_staining import viscy_virtual_staining
from napari_tmidas.processing_functions.cellpose_segmentation import cellpose_segmentation

viewer = napari.Viewer()

# Step 1: Load phase contrast image
phase_image = viewer.open('phase_3d.tif').data

# Step 2: Virtual staining
virtual_stain = viscy_virtual_staining(
    phase_image,
    dim_order='ZYX',
    output_channel='nuclei'  # Get only nuclei for segmentation
)

# Step 3: Segment nuclei using Cellpose
nuclei_labels = cellpose_segmentation(
    virtual_stain,
    dim_order='ZYX',
    flow_threshold=0.4
)

# Visualize results
viewer.add_image(phase_image, name='Phase Contrast')
viewer.add_image(virtual_stain, name='Virtual Nuclei', colormap='green')
viewer.add_labels(nuclei_labels, name='Nuclei Segmentation')
```

## Hardware Recommendations

### GPU (Recommended)
- **VRAM**: Minimum 4 GB, 8 GB or more recommended
- **Supported**: CUDA-compatible NVIDIA GPUs
- Processing speed: ~1-2 seconds per batch (15 slices) on modern GPU

### CPU (Fallback)
- Will automatically fall back to CPU if no GPU is available
- Processing speed: ~30-60 seconds per batch (15 slices)
- Higher memory usage (~4-8 GB RAM)

## Troubleshooting

### "VisCy environment not found"
**Solution**: The environment will be created automatically on first use. If creation fails:
1. Check internet connection (for downloading model)
2. Ensure sufficient disk space (~2 GB for environment + model)
3. Try manually creating: `Plugins > napari-tmidas > Create VisCy Environment`

### "Image has less than 15 Z slices"
**Solution**: The VSCyto3D model requires at least 15 Z-slices
- Acquire images with more Z-slices
- Use a different processing method for thin Z-stacks
- Consider using 2D methods if Z information is not critical

### "Model checkpoint not found"
**Solution**: Re-create the environment to re-download the model
1. Delete the environment folder: `~/.napari-tmidas/envs/viscy/`
2. Restart napari and run the function again

### Out of Memory Errors
**Solutions**:
1. Reduce image size (crop or downsample)
2. Process time-lapse data one timepoint at a time
3. Use CPU instead of GPU (set CUDA_VISIBLE_DEVICES="")
4. Close other applications to free up RAM/VRAM

### Poor Virtual Staining Quality
The model works best with:
- **Good contrast** in the input phase image
- **Similar imaging conditions** to the training data
- **Well-focused** Z-stacks with consistent spacing

**Optimization tips**:
1. Adjust microscope settings for optimal contrast
2. Ensure consistent Z-spacing (0.5-1.0 µm recommended)
3. Normalize input images if brightness varies significantly
4. Use appropriate numerical aperture (NA ≥ 0.75 recommended)

## Tips & Best Practices

1. **Z-stack acquisition**:
   - Use 15-30 Z-slices for optimal results
   - Maintain consistent Z-spacing (0.5-1.0 µm)
   - Ensure full cell coverage (avoid cropping nuclei)

2. **Batch processing**:
   - Process multiple files in batch mode for efficiency
   - Use consistent imaging parameters across samples
   - Output both channels initially to evaluate quality

3. **Integration with other tools**:
   - Virtual nuclei → Cellpose/Stardist for segmentation
   - Virtual membrane → Boundary analysis, cell shape metrics
   - Combine with tracking functions for dynamic analysis

4. **Performance optimization**:
   - Close other GPU applications during processing
   - Process smaller ROIs if full image is too large
   - Use GPU for real-time exploration, CPU for batch overnight jobs

## Model Citation

If you use VisCy virtual staining in your research, please cite:

```bibtex
@article{guo2024revealing,
  title={Revealing architectural order with quantitative label-free imaging and deep learning},
  author={Guo, Syuan-Ming and Yeh, Li-Hao and Folkesson, Jenny and Ivanov, Ivan E and Krishnan, Anitha P and Keefe, Matthew G and Hashemi, Ezzat and Shin, David and Chhun, Bryant B and Cho, Nathan H and others},
  journal={Elife},
  volume={9},
  pages={e55502},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```

## Additional Resources

- **VisCy GitHub**: https://github.com/mehta-lab/VisCy
- **Model Zoo**: https://github.com/mehta-lab/VisCy/wiki/Model-Zoo
- **Documentation**: https://github.com/mehta-lab/VisCy/wiki
- **Paper**: https://doi.org/10.7554/eLife.55502

## See Also

- [Cellpose Segmentation](cellpose_segmentation.md) - Segment virtual stained nuclei
- [RegionProps Analysis](regionprops_analysis.md) - Analyze segmented cells
- [Trackastra Tracking](trackastra_tracking.md) - Track cells over time

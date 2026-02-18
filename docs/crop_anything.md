# Batch Crop Anything

Batch Crop Anything is an interactive napari plugin for intelligent image cropping and object extraction. It combines SAM2 (Segment Anything Model 2) for automatic object detection with an intuitive interface for selecting and cropping specific objects from microscopy images.

## Overview

This plugin enables:
- **Interactive object segmentation** using AI-powered SAM2 model
- **2D and 3D processing** for single images and image stacks
- **Multi-frame propagation** for temporal datasets
- **Batch cropping** of selected objects across multiple images
- **GPU acceleration** (CUDA, Apple Silicon, CPU fallback)

## Installation

### Installation

Batch Crop Anything will automatically create a dedicated SAM2 environment when first used.

### SAM2 Setup

SAM2 is automatically downloaded and installed in an isolated environment on first use. However, you must manually install ffmpeg:

```bash
# Linux (usually pre-installed)
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Or via conda
mamba install -c conda-forge ffmpeg
```

Optional: Set the SAM2 path environment variable to specify a custom installation location:

```bash
export SAM2_PATH=/path/to/sam2
```

## Quick Start

1. Open napari and navigate to **Plugins → napari-tmidas → Batch Crop Anything**
2. Select your image folder containing `.tif` or `.zarr` files
3. Choose between **2D Mode** (single images) or **3D Mode** (stacks/time-series)
4. Click on objects in the image to segment them with SAM2
5. Use the interactive table to select which objects to crop
6. Save cropped regions to disk

## Modes

### 2D Mode

Perfect for single 2D images or when you want to segment individual layers.

**Interactive Workflow:**
- Click on image → Creates positive point prompt
- Shift+click → Creates negative point prompt (refine boundaries)
- SAM2 segments the object at that point
- Click existing objects → Select for cropping

**Controls:**
- Sensitivity slider: Adjust detection confidence (0-100)
  - Higher values → More aggressive segmentation
  - Lower values → Conservative segmentation
- Next/Previous buttons: Navigate through image collection

### 3D Mode

For volumetric data (Z-stacks) or time-series datasets (time-lapse videos).

**Data Format Recognition:**
The plugin automatically detects your data format:
- **TZYX**: Time-series with Z-stacks (e.g., time-lapse confocal)
- **TYX**: Time-series without Z dimension (e.g., 2D time-lapse)
- **ZYX**: Single Z-stack without time dimension

**Interactive Workflow:**
1. Navigate to the **first slice** where your object appears (using the dimension slider)
2. Click on the object in **2D view** (not 3D view)
3. SAM2 segments the object at that frame
4. **Automatic propagation**: Segmentation is propagated through all frames using video tracking

**Important:** Always click on the first frame containing your object. SAM2's video propagation then extends the segmentation forward through time.

**Controls:**
- Use dimension sliders to navigate frames/slices
- Sensitivity slider: Control propagation aggressiveness
- Objects persist across frames automatically

## Interactive Controls

### Prompt Modes

SAM2 supports two ways to specify objects:

#### Point Mode (Default)
Click on image to add point prompts. Best for complex boundaries or small objects.

| Action | Effect |
|--------|--------|
| **Click** | Add positive point (include this region) |
| **Shift+Click** | Add negative point (exclude this region) |

#### Box Mode
Draw rectangles around objects. Best for quick segmentation of simple objects.

1. Select **Box Mode** from the UI
2. Draw rectangle around object
3. SAM2 segments the region inside the box
4. Can add more rectangles for multiple objects
5. Shift+draw to refine/subtract from existing box

**When to use each:**
- **Points**: Fine details, intricate boundaries, removing noise
- **Box**: Quick segmentation, well-defined rectangular regions, speed

### Navigation

| Action | Effect |
|--------|--------|
| **Left/Right Click** | Navigate to adjacent frames (3D mode) |
| **Dimension Slider** | Jump to specific frame/slice (3D mode) |

### Table Selection

The label table displays all detected objects:
- **Checkbox**: Select objects to crop
- **Object ID**: Unique identifier in segmentation
- **Area**: Size in pixels
- **Statistics**: Min/max intensity values

### Sensitivity Control

Adjust SAM2's detection confidence:
- **Range**: 0-100
- **Default**: 50
- **Effect on 2D**: Higher values segment larger regions
- **Effect on 3D**: Higher values allow more aggressive frame-to-frame propagation

## Output Files

When you save cropped objects, the plugin creates:

```
output_folder/
├── image1_object_1.tif
├── image1_object_2.tif
├── image2_object_1.tif
└── ...
```

Each cropped region is:
- Extracted as a minimal bounding box
- Saved as a separate TIFF file
- Named with original image + object ID

## Advanced Features

### Video Conversion (3D Mode)

For 3D/4D processing, the plugin converts image stacks to MP4 format:
- **Automatic**: Conversion happens on first load
- **Cached**: MP4 files are reused if they exist
- **4D Handling**: TZYX data is projected to TYX using maximum intensity projection

### GPU Acceleration

Device selection is automatic:
- **NVIDIA GPU**: CUDA (if available)
- **Apple Silicon**: MPS (Metal Performance Shaders)
- **CPU**: Fallback for all systems

Check console output to see which device is active.

### Error Handling

If SAM2 initialization fails:
- Images still load without automatic segmentation
- You can still use manual annotation tools
- Check console for detailed error messages
- Verify SAM2_PATH environment variable if needed

## Troubleshooting

### "SAM2 not found" warning

**Solution**: SAM2 will auto-install on first use. If this fails, check console for errors.

### Segmentation not appearing

**Possible causes:**
1. SAM2 model not initialized (check console)
2. Image format incompatible (must be `.tif`, `.tiff`, or `.zarr`)
3. GPU out of memory (switch to CPU)

**Solutions:**
- Check console output for error messages
- Try reducing image size or resolution
- Enable CPU mode: `torch.device('cpu')`

### Memory errors on GPU

**Solutions:**
- Reduce image dimensions
- Switch to CPU mode
- Close other GPU-intensive applications
- Clear GPU cache: `torch.cuda.empty_cache()`

### Slow 3D processing

**Causes:**
- Large 4D volumes
- Limited GPU memory
- Network latency (SAM2 checkpoint download)

**Solutions:**
- Use 2D mode for individual slices
- Reduce image dimensions
- Pre-process images to smaller regions

## File Format Support

| Format | Dimensions | Status |
|--------|-----------|--------|
| `.tif` / `.tiff` | 2D, 3D, 4D | ✓ Fully supported |
| `.zarr` | 2D, 3D, 4D | ✓ Fully supported |
| `.png` | 2D | ✗ Not supported |
| `.jpg` | 2D | ✗ Not supported |

## Performance Tips

1. **Pre-process large images**: Downscale to < 2 megapixels for interactive use
2. **Use 2D mode**: For single large images, segment individual slices
3. **GPU selection**: CUDA > MPS > CPU (in terms of speed)
4. **Batch processing**: Process multiple small images faster than one large image
5. **Sensitivity tuning**: Start at 50, adjust based on results

## Dataset Examples

### Confocal Microscopy

```
confocal_images/
├── sample1.tif          (3D Z-stack)
├── sample2.tif          (3D Z-stack)
└── ...
```
→ Use **3D Mode**, select appropriate crop regions

### Time-Lapse Video

```
timelapse/
├── embryo_t001.tif      (2D)
├── embryo_t002.tif      (2D)
├── embryo_t003.tif      (2D)
└── ...
```
→ Process each timepoint with **2D Mode**, or stack into TYX format for 3D mode

### Multi-Channel 4D Data

```
multi_channel/
├── raw_ch1_ch2.tif      (4D: TZYX)
├── raw_ch2_ch2.tif      (4D: TZYX)
└── ...
```
→ Use **3D Mode**, plugin auto-detects dimensions

## Related Features

- **[Basic Processing](basic_processing.md)**: Image preprocessing and filtering
- **[Cellpose Segmentation](cellpose_segmentation.md)**: Alternative segmentation method
- **[Grid View Overlay](grid_view_overlay.md)**: Visualize multiple processed images
- **[Label Inspection](label_inspection.md)**: Interactive label verification and editing

## Technical Details

### SAM2 Model

- **Model**: SAM2.1 Hiera Large
- **Input**: RGB images (0-1 range)
- **Output**: Binary mask for each object
- **Inference**: Single-pass prompting + optional propagation in videos

### Device Detection

```
macOS:
  - Check for Apple Silicon (MPS) → Use MPS
  - Otherwise → Use CPU

Linux/Windows:
  - Check for CUDA → Use CUDA
  - Otherwise → Use CPU
```

## References

- [SAM2 Paper](https://arxiv.org/abs/2406.07399)
- [napari Documentation](https://napari.org/)
- [napari-tmidas Repository](https://github.com/MercaderLabAnatomy/napari-tmidas)

## Citation

If you use Batch Crop Anything in your research, please cite:

```bibtex
@software{napari_tmidas_2024,
  title = {napari-tmidas: Batch Image Processing for Microscopy},
  author = {Mercader Lab},
  year = {2024},
  url = {https://github.com/MercaderLabAnatomy/napari-tmidas}
}
```

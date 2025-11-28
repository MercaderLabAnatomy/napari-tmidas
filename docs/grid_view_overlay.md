# Grid View: Intensity + Labels Overlay

## Overview

The `Grid View: Intensity + Labels Overlay` processing function creates a comprehensive visualization showing intensity images overlaid with their corresponding label boundaries, arranged in a convenient grid layout for quick inspection.

## Purpose

This function is designed for quality control and visual inspection of segmentation results. Instead of opening each image pair individually, you can generate a single grid image showing all your intensity + label overlays at once.

## How It Works

1. **Uses Selected Files**: Processes only the label files you selected via the suffix filter in batch processing
2. **Automatic Pairing**: For each selected label file, automatically finds the corresponding intensity image
3. **Overlay Creation**: For each pair:
   - Intensity values are displayed in the **green channel**
   - Label boundaries are displayed in **magenta** (red + blue)
4. **Auto-sized Grid**: All overlays are arranged in an automatically sized grid
   - Uses square root of image count as baseline
   - Optimized for different image counts (caps at 12 columns max)

## Parameters

None - the function automatically handles everything!

## Usage

### In napari

1. Open the **Batch Image Processing** widget
2. Select a folder containing your label images
3. **Important**: Set the input suffix to match your label files
   - Example: `_convpaint_labels_filtered.tif`
   - Example: `_labels.tif`
4. Choose **"Grid View: Intensity + Labels Overlay"** from the processing function dropdown
5. Click **Start Batch Processing**

The function will only process the files matching your suffix filter, not all label files in the folder.

### Expected File Structure

The function expects label and intensity images in the same folder:

```
your_folder/
├── image1.tif                              # intensity image
├── image1_convpaint_labels_filtered.tif    # corresponding labels
├── image2.tif                              # intensity image
├── image2_labels.tif                       # corresponding labels
└── ...
```

### Supported Label Suffixes

The function recognizes these label file patterns:
- `*_convpaint_labels_filtered.tif`
- `*_labels_filtered.tif`
- `*_labels.tif`
- `*_intensity_filtered.tif`

The corresponding intensity image is found by removing these suffixes.

## Output

### File Output
- **Format**: PNG image (8-bit RGB)
- **Naming**: `_grid_overlay.png` suffix added to the first processed file
- **Location**: Saved in parent directory of input folder
- **Use case**: Ready for publications, presentations, and figures
- **Channels**: 3-channel RGB blended overlay

### Visual Interpretation
- **Grayscale background**: Normalized intensity signal (brighter = higher intensity)
- **Colored regions**: Labeled objects with 60% opacity overlay
- **Each label**: Gets a unique color for easy distinction
- **Dark areas**: Low intensity and no labeled objects

## Example

### Input Files
```
folder/
├── sample1.tif (intensity)
├── sample1_labels.tif
├── sample2.tif (intensity)
├── sample2_labels.tif
├── sample3.tif (intensity)
└── sample3_labels.tif
```

### Processing
```
Suffix filter: _labels.tif
Selected function: Grid View: Intensity + Labels Overlay
```

### Output
- Single image: `sample1_grid_overlay.png` (PNG format)
- Grid layout: Auto-sized (2 columns for 3 images)
- Each cell shows one intensity + labels overlay
- Ready to insert into papers/presentations

## Tips

1. **Grid Auto-sizing**:
   - Few images (1-4): ~2 columns
   - Medium (5-16): ~3-4 columns
   - Many (17-100): ~5-10 columns
   - Very many (100+): caps at 12 columns max

2. **3D Data**:
   - Function automatically takes max projection along first axis
   - Works with TYX, ZYX data

3. **Performance** - Optimized for Large Datasets:
   - Parallel processing with multi-threading (up to 8 workers)
   - Simple transparent overlay (no complex boundary detection)
   - Fast image resizing using OpenCV when available
   - Automatically downsamples images based on grid size
   - Target: ~10,000 pixel wide final grid (high resolution)
   - Per-image size: 200-1000px depending on total count
   - Examples:
     - 10 images: ~950px per image, <5 seconds
     - 100 images: ~830px per image, ~5-15 seconds
     - 1000 images: ~280px per image, ~20-40 seconds
   - Real-time progress bar showing completion status

4. **Quality Control Workflow**:
   ```
   1. Run segmentation → get label images
   2. Run this function → get grid overview
   3. Identify problematic segmentations
   4. Re-process specific images as needed
   ```

5. **Large Datasets (1000+ images)**:
   - Function handles thousands of image pairs efficiently
   - Batch processing in chunks of 100 images for optimal memory usage
   - Parallel processing with 4 workers for fast I/O
   - Intelligent downsampling (200-1000px per image depending on count)
   - Explicit memory cleanup between batches
   - Final grid stays high resolution (~10,000px wide)
   - Individual images scaled to maintain visibility
   - No need to manually split - handles full dataset at once
   - Typical performance: 20-30 image pairs per second
   - Memory-efficient: only ~100 overlays in RAM at a time

## Troubleshooting

### "No label files found in folder"
- Check that label files have recognized suffixes (`*_labels*.tif`)
- Ensure files are TIFF format (`.tif` or `.tiff`)

### "Dimension mismatch"
- Intensity and label images must have same dimensions
- Function attempts max projection for 3D data
- Check that pairs are correctly matched

### "No intensity image found"
- Ensure intensity image exists in same folder
- Check that intensity filename matches (after removing label suffix)
- Example: `image1_labels.tif` requires `image1.tif`

### Grid looks too dense
- Grid columns are automatically calculated
- Consider processing fewer files at once for larger individual overlays
- For 100+ images, grid caps at 12 columns to remain viewable

## Implementation Details

The function uses:
- **tifffile** for fast TIFF reading
- **scikit-image** for image resizing (handles all dtypes including uint32)
- **numpy** for image processing and alpha blending
- **concurrent.futures** for parallel processing
- Automatic normalization for display
- HSV color generation for distinct label colors
- 60% opacity blending for label visibility

## See Also

- [Intensity Label Filter](intensity_label_filter.md) - Filter labels by intensity before visualization
- [Label Inspector](../README.md#label-inspection) - Interactive single-pair inspection

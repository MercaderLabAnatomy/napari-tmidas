# Label-Based Image Cropping

## Overview

The Label-Based Image Cropping pipeline allows you to crop and mask images using user-drawn labels. This is particularly useful for:
- Extracting regions of interest from larger images
- Removing background or unwanted regions
- Preparing images for downstream analysis

## Key Features

### Interactive Label Expansion

The pipeline provides **opt-in** label expansion with immediate visual feedback:

**Expand Labels Across Z**: Copies the label from your current visible z-slice to all other z-slices in that frame. Click the checkbox to expand, then inspect and edit the labels before cropping.

**Expand Across Time Frames**: Copies the label from your current frame to all other time frames. Useful when the same region should be cropped across the entire time series.

**Manual Control**: Both expansion options are **unchecked by default**. You must explicitly enable them when needed. Labels update immediately in the viewer so you can:
- Inspect the expansion result
- Manually edit any expanded slices
- Verify before cropping

### Masking

Everything outside the drawn label is masked to zero (background value). Only pixels within the labeled region are preserved in the output.

## Usage

### Interactive Cropping (Widget)

1. **Load your data**: Open your intensity image and create a labels layer in napari
2. **Draw labels**: Use napari's built-in drawing tools to draw a label in the labels layer
   - Draw in the current visible z-slice/frame you're viewing
3. **Select layers**: In the Label-Based Image Cropping dock widget:
   - Select your Image Layer
   - Select your Label Layer (with your drawn labels)
   - Give the output a name (default: "cropped")
4. **Expand labels (optional)**:
   - Check "Expand labels across Z" to copy current slice to all z-slices
   - Check "Expand across time frames" to copy current frame to all frames
   - Labels update immediately - inspect and edit as needed
5. **Crop**: Click "Crop Image"
6. **Verify**: The cropped result will be added as a new layer

### Batch Processing

For batch processing multiple images with corresponding label files, use the Batch Image Processing widget with the "Label-Based Cropping" function.

**File naming convention**: Label files should follow one of these patterns:
- `image_labels.tif`
- `image_labels_filtered.tif`
- `image_seg.tif`
- `image_mask.tif`

Example:
- Image: `sample_001.tif`
- Label: `sample_001_labels.tif`

## Advanced Usage

### Working with 3D Data

When working with 3D volumetric data (Z, Y, X):
1. Set napari to 2D mode (View → Toggle 3D)
2. Navigate to the z-slice you want to label
3. Draw your label in that slice
4. Check "Expand labels across Z"
5. Inspect the expanded labels in other z-slices
6. Edit any slices if needed
7. Click "Crop Image"

### Working with Time-Series Data

For time-lapse data (T, Z, Y, X):

**Same label in all frames**:
1. Draw label in one frame
2. Check "Expand labels across Z" (for that frame's z-slices)
3. Check "Expand across time frames" (to copy to all frames)
4. Click "Crop Image"

**Different labels per frame**:
1. Draw label in frame 1
2. Check "Expand labels across Z"
3. Navigate to frame 2, draw different label
4. Check "Expand labels across Z" again
5. Repeat for each frame
6. Click "Crop Image" (uses current labels as-is)

## Parameters

### Widget Parameters

- **Image Layer**: The image you want to crop
- **Label Layer**: The labels defining the region to keep
- **Output Name**: Name for the cropped result layer (default: "cropped")
- **Expand labels across Z**: Copy current z-slice to all z-slices (default: unchecked)
- **Expand across time frames (T)**: Copy current frame to all frames (default: unchecked)

### Batch Processing Parameters

- **label_image_path**: Path to label image (auto-detected if not specified)
- **expand_z**: Expand 2D labels across z-slices (default: False)
- **expand_time**: Expand labels across time frames (default: False)
- **output_format**: Format for saved output
  - `"same"`: Keep original format
  - `"tif"`: Save as TIFF
  - `"npy"`: Save as NumPy array

## Examples

### Example 1: Extract a Single Cell from Microscopy Image

```python
import napari
from napari_tmidas.processing_functions.label_based_cropping import label_based_cropping

# Load your data
viewer = napari.Viewer()
image_layer = viewer.open("/path/to/cell_image.tif")
labels_layer = viewer.add_labels(...)

# Draw your label around the cell of interest...

# Crop programmatically
cropped = label_based_cropping(
    image_layer.data,
    label_image_path="/path/to/label_file.tif"
)

# Add result to viewer
viewer.add_image(cropped, name="cropped_cell")
```

### Example 2: Batch Process Multiple Images

```python
from napari_tmidas.processing_functions.label_based_cropping import batch_label_based_cropping

results = batch_label_based_cropping(
    input_folder="/data/raw_images",
    output_folder="/data/cropped_images",
    auto_detect_labels=True,
    num_workers=4
)

print(f"Processed: {len(results['successful'])} images")
print(f"Failed: {len(results['failed'])} images")
```

## Data Types Supported

The pipeline preserves the dtype of the input image:
- `uint8`, `uint16`, `uint32`: Integer types
- `float32`, `float64`: Floating-point types

## Performance Considerations

- **Memory**: Cropping is done in-memory; ensure sufficient RAM for your image sizes
- **Parallelization**: Batch processing uses 4 workers by default; adjust based on your system
- **Label Expansion**: For very large 3D/4D images, expanding labels may take some time

## Troubleshooting

### Label Not Expanding as Expected

- Ensure you're in 2D viewing mode when drawing labels
- Check that your label layer and image have the same spatial dimensions (Y, X)
- Verify the image has the correct number of z-slices or frames

### Shape Mismatch Errors

- Make sure your label file has the correct dimensions
- For batch processing, ensure label files are in the same folder as intensity images
- Check that filenames follow the expected pattern

### No Output After Cropping

- Verify that your label contains non-zero values
- Check the info panel for error messages
- Ensure the label and image have matching spatial dimensions

## See Also

- [Image Processing Overview](./basic_processing.md)
- [Batch Processing](./basic_processing.md#batch-processing)
- [Label Inspector](./batch_label_inspection.md)

## Technical Details

### Algorithm

1. **Load data**: Read intensity and label images
2. **Validate shapes**: Ensure label and image have compatible dimensions
3. **Expand labels**: If needed, expand 2D labels to match image dimensionality
4. **Create mask**: Generate binary mask from label (label > 0)
5. **Apply mask**: Multiply image by mask to preserve only labeled regions
6. **Return result**: Output masked image

### Label Expansion Strategy

- **2D to 3D**: Repeat the 2D label across all z-slices using `np.repeat`
- **2D to 4D**: First expand to 3D, then repeat across time frames
- **3D to 4D (per-frame)**: Expand z-dimension for each time frame

### Memory Usage

For batch processing:
- Each image loaded into memory sequentially
- Label expanded in-memory
- Result saved to disk before loading next image
- Overall memory usage scales with largest single image, not total dataset size

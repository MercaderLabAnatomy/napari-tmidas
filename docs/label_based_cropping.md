# Label-Based Image Cropping

## Overview

The Label-Based Image Cropping pipeline allows you to crop and mask images using user-drawn labels. This is particularly useful for:
- Extracting regions of interest from larger images
- Removing background or unwanted regions
- Preparing images for downstream analysis

## Key Features

### Automatic Label Expansion

The pipeline intelligently handles different image dimensionalities:

**2D Label to 3D Image**: If you draw a 2D label on a single z-slice and your image is 3D (Z, Y, X), the label is automatically repeated across all z-slices.

**2D Label to 4D Image (with time)**: If you draw a 2D label in 2D viewing mode and your image has both time and z-dimensions (T, Z, Y, X), the label is automatically expanded to:
- All z-slices for that time frame, then
- Repeated across all time frames

**Per-Frame Labels**: You can draw different labels in different time frames, and each will be expanded to cover all z-slices within that frame.

### Masking

Everything outside the drawn label is masked to zero (background value). Only pixels within the labeled region are preserved in the output.

## Usage

### Interactive Cropping (Widget)

1. **Load your data**: Open your intensity image and create a labels layer in napari
2. **Draw labels**: Use napari's built-in drawing tools to draw a label in the labels layer
   - For 2D images: simply draw the region
   - For 3D images: draw in one z-slice; it will automatically apply to all z-slices
   - For 4D images: draw in one time frame; it will apply to all z-slices in that frame
3. **Select layers**: In the Label-Based Image Cropping dock widget:
   - Select your Image Layer
   - Select your Label Layer (with your drawn labels)
   - Give the output a name (default: "cropped")
4. **Crop**: Click "Crop Image"
5. **Verify**: The cropped result will be added as a new layer

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
4. Click "Crop Image"
5. The label will automatically expand to all z-slices

### Working with Time-Series Data

For time-lapse data (T, Z, Y, X):
1. Enable 2D viewing mode
2. Navigate to the time frame and z-slice you want to label
3. Draw your label
4. Click "Crop Image"
5. The label expands to all z-slices in the current frame, then applies to all frames

To use different labels for different frames:
1. Go to each frame individually
2. Draw the label for that frame
3. The pipeline will preserve per-frame labels

## Parameters

### Widget Parameters

- **Image Layer**: The image you want to crop
- **Label Layer**: The labels defining the region to keep
- **Output Name**: Name for the cropped result layer (default: "cropped")
- **Auto-expand 2D labels**: Automatically expand 2D labels to 3D/4D (default: enabled)

### Batch Processing Parameters

- **label_image_path**: Path to label image (auto-detected if not specified)
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

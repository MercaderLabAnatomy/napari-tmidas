# Label-Based Image Cropping - Quick Start Guide

## What It Does

Draw a label in a 2D region, and the pipeline automatically:
1. **Expands** the 2D label to all z-slices (for 3D images)
2. **Repeats** the label in all time frames (for 4D images)
3. **Masks** everything outside the label to zero

## Quick Usage

### In Napari UI

1. Open image → Create labels layer → Draw your region
2. Go to Plugins → Label-Based Image Cropping
3. Select Image and Label layers → Click "Crop Image"
4. Result appears as a new layer

### In Python

```python
from napari_tmidas.processing_functions.label_based_cropping import label_based_cropping
import numpy as np

# Your data
image = np.load('image.npy')  # Shape: (64, 128, 128) for Z, Y, X
label = np.load('label.npy')  # Shape: (128, 128) for Y, X

# Crop (label automatically expands to 3D)
cropped = label_based_cropping(image, label_image_path='label_file.tif')

# Result has same shape as original image with masked regions as zero
print(cropped.shape)  # (64, 128, 128)
```

## Supported Image Dimensions

| Image Dim | Label Dim | Result |
|-----------|-----------|--------|
| (Y, X) | (Y, X) | 2D cropped image |
| (Z, Y, X) | (Y, X) | 3D cropped image, label repeated across Z |
| (T, Y, X) | (Y, X) | Per-frame cropping, label repeated per time frame |
| (T, Z, Y, X) | (Y, X) | Label expanded to all Z in each T frame |
| (T, Z, Y, X) | (T, Y, X) | Per-frame labels, expanded to all Z |

## Batch Processing

```python
from napari_tmidas.processing_functions.label_based_cropping import batch_label_based_cropping

results = batch_label_based_cropping(
    input_folder='/data/images',
    output_folder='/data/cropped',
    num_workers=4
)

print(f"Success: {len(results['successful'])} images")
print(f"Failed: {len(results['failed'])} images")
```

**File naming**: Label files must follow pattern like `image_labels.tif`, `image_seg.tif`, etc.

## Key Features

✓ **Automatic label expansion** - 2D labels become 3D/4D  
✓ **Batch processing** - Process multiple images in parallel  
✓ **Type preservation** - Maintains uint8, uint16, float32, etc.  
✓ **Interactive widget** - Visual UI in napari  
✓ **Memory efficient** - Processes images sequentially in batch mode  
✓ **Fully tested** - 25 unit tests covering all scenarios  

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Label not expanding | Use 2D viewing mode when drawing labels |
| Shape mismatch error | Ensure label and image spatial dims match (Y, X) |
| No output | Check that label contains non-zero values (label > 0) |
| Batch processing skips files | Check label files exist with correct naming pattern |

## See Also

- Full documentation: [label_based_cropping.md](./label_based_cropping.md)
- Batch processing: [Batch Image Processing](./basic_processing.md#batch-processing)

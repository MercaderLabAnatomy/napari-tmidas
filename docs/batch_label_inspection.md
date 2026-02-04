# Batch Label Inspection

The Batch Label Inspection widget enables interactive verification, correction, and refinement of segmentation labels across entire image datasets. Inspect and manually edit label images while automatically saving changes back to disk.

## Overview

Streamline the quality control workflow for segmentation results:

- **Side-by-side viewing**: Original image + label mask for easy comparison
- **Interactive editing**: Use napari's paint, eraser, and selection tools
- **Automatic saving**: Changes saved to disk as you proceed through pairs
- **Progress tracking**: Navigate through entire dataset with visual progress indicator
- **Batch workflow**: Process hundreds of images without manual file management
- **Format validation**: Automatic detection and validation of label image formats

## Quick Start

1. Open napari and navigate to **Plugins → napari-tmidas → Batch Label Inspection**
2. Select folder containing your image-label pairs
3. Specify label suffix (e.g., `_labels.tif`, `_segmentation.tif`)
4. Click **Load** to index image-label pairs
5. Edit labels in the viewer using napari's drawing tools
6. Click **Save and Continue** to save changes and move to next pair
7. Click **Previous** to revisit earlier pairs if needed

## Workflow

### Step 1: Prepare Your Files

Organize files with a consistent naming pattern:

```
segmentation_results/
├── sample1.tif              (original image)
├── sample1_labels.tif       (segmentation labels)
├── sample2.tif              (original image)
├── sample2_labels.tif       (segmentation labels)
└── ...
```

**File Requirements**:
- Label images must be integer type (8-bit, 16-bit, or 32-bit)
- Image and label must have matching spatial dimensions
- Any image format supported by scikit-image (TIF, PNG, etc.)

### Step 2: Configure Inspection

**Folder Path**: Select directory containing image-label pairs

**Label Suffix**: Specify the suffix that identifies label files
- Examples: `_labels`, `_segmentation`, `_mask`, `_labels_filtered`
- The suffix is used to match labels with images
- File before suffix is treated as the base image name

**Example matching**:
```
sample1.tif + sample1_labels.tif          ✓ Match
sample1_labels.tif + sample1.tif          ✓ Match (order doesn't matter)
sample1_seg.tif + sample1_seg_labels.tif  ✗ No match (_labels not in sample1_seg.tif)
sample1_labels_filtered.tif               ✓ Match (if suffix is "_labels")
```

### Step 3: Load Pairs

Click **Load** to scan the folder and create image-label pairs.

**Status Report**:
- Number of valid pairs found
- Any skipped files and reasons
- Format validation issues (if any)

### Step 4: Edit and Review

For each pair displayed:

**Viewing**:
- Left panel: Original image
- Right panel: Label layer (editable)
- Status bar: Current pair number and filename

**Editing Tools** (napari built-in):
- **Paint**: Add new labels
  - Select label ID from right panel
  - Click and drag to paint
- **Eraser**: Remove labels
  - Set label ID to 0 to erase
- **Selection tools**: Select and modify regions
- **Undo/Redo**: Ctrl+Z / Ctrl+Y

**Viewing Tips**:
- Adjust label opacity (right panel) to see image beneath
- Use different colormaps for better visibility
- Toggle layers on/off to compare

### Step 5: Save Progress

**Save and Continue**:
- Saves current label edits to disk
- Moves to next image-label pair
- Shows confirmation status

**Previous**:
- Saves current edits
- Returns to previous pair (useful for refinement)

**Stop**:
- Saves final edits and closes widget

## Features

### Automatic Pair Matching

The widget intelligently matches images with their labels:

```
Input: label suffix "_labels"

✓ Correct matches:
  image.tif ↔ image_labels.tif
  sample_001.tif ↔ sample_001_labels.tif
  data_ch1.tif ↔ data_ch1_labels.tif

✗ No match:
  image1.tif + image2_labels.tif (different base names)
  file_labels.tif (no matching image found)
```

### Format Validation

Automatic checks ensure label integrity:

- **Integer type validation**: Labels must be integer (not float/RGB)
- **File format support**: TIF, PNG, etc. (any scikit-image format)
- **Dimension matching**: Labels must match image spatial dimensions
- **Error reporting**: Detailed messages for any validation issues

### Progress Tracking

**Status Bar Display**:
```
Viewing pair 5 of 47: sample_005.tif
```

Shows:
- Current pair number
- Total number of pairs
- Current filename

Navigate using **Previous** / **Save and Continue** buttons

### Automatic Saving

**When saving**:
- Current label layer written to disk
- Original filename preserved
- Data type preserved (8/16/32-bit as original)
- File overwritten (use backup if needed)
- Status confirmed in notification

## Use Cases

### Quality Control of Automated Segmentation

After running Cellpose or another segmenter:
1. Load output label images
2. Visually compare with original images
3. Fix errors (merge split objects, remove false positives)
4. Auto-saves corrections

### Merging Split Objects

When segmentation over-splits cells:
1. Select both object IDs
2. Paint with same label to merge
3. Erase artifacts/noise
4. Save changes

### Removing False Positives

When segmentation detects spurious objects:
1. Use eraser (label = 0) to remove
2. Paint background color where needed
3. Save corrected labels

### Refining Boundaries

For inaccurate object boundaries:
1. Paint with same object ID to expand
2. Use eraser to shrink
3. Fine-tune label borders
4. Save refined masks

## Tips & Best Practices

### Organization
- Keep consistent naming scheme across dataset
- Use descriptive suffix names (`_labels_v2`, not just `_v2`)
- Backup original labels before mass editing

### Editing Efficiency
- Edit in 2D view for precise control
- Use opacity adjustment to see image beneath labels
- Zoom in for fine boundary adjustments
- Use selection tools for large regions

### Data Management
- Check "Save and Continue" status confirms write
- Verify edits saved by reloading file
- Use version suffixes for multiple iterations (`_labels_v1`, `_labels_v2`)
- Keep audit trail of manual corrections

### Performance
- For >100 pairs, consider processing in batches
- Verify label format before batch processing
- Use SSD storage for faster loading

## Troubleshooting

### "No Label Files Found"

**Cause**: Suffix doesn't match any files

**Solutions**:
- Check actual label filenames in folder
- Verify suffix spelling and case sensitivity
- Try shorter suffix (e.g., `_labels` instead of `_labels_filtered`)

### "No Valid Image-Label Pairs"

**Cause**: Labels don't match images or format issues

**Solutions**:
- Verify image and label basenames match
- Check label images are integer type (not float/RGB)
- Ensure dimensions match between image and label

### "Format Issues" Warning

**Cause**: Some label files not in expected format

**Possible Issues**:
- Label image is RGB/float instead of integer
- Label file corrupted or incompatible
- Dimension mismatch with image

**Solutions**:
- Convert labels to integer format if needed
- Regenerate problematic label files
- Verify with external tools (ImageJ, etc.)

### Edits Not Saving

**Cause**: Wrong layer selected or permission issue

**Solutions**:
- Ensure "Labels" layer (right panel) is selected
- Check folder write permissions
- Verify label filename in confirmation message

### Changes Lost After Clicking Previous

**Note**: Previous saves current edits first

If edits appear lost:
- Check file modification time
- Reload file to verify save
- Check for backup/version files

## File Format Support

| Format | Input | Output | Status |
|--------|-------|--------|--------|
| TIF/TIFF | ✓ | ✓ | Full support |
| PNG | ✓ | ✓ | Full support |
| JPEG | ✓ (8-bit only) | ✗ | Read-only |
| Zarr | ✓ | Limited | Supported |
| HDF5 | ✗ | ✗ | Not supported |

## Data Types Supported

| Type | Support |
|------|---------|
| uint8 | ✓ Full |
| uint16 | ✓ Full |
| uint32 | ✓ Full |
| int8, int16, int32 | ✓ Supported |
| float, RGB | ✗ Not supported (validation error) |

## Related Features

- **[Cellpose Segmentation](cellpose_segmentation.md)** - Generate labels to inspect
- **[Batch Processing](basic_processing.md)** - Post-process labels
- **[Label Operations](basic_processing.md#label-image-operations)** - Filter/transform labels
- **[RegionProps Analysis](regionprops_analysis.md)** - Analyze edited labels

## Technical Details

### Workflow Architecture

```
1. User selects folder + suffix
         ↓
2. Widget scans folder
         ↓
3. Matches image-label pairs
         ↓
4. Validates formats
         ↓
5. Loads first pair into napari
         ↓
6. User edits labels
         ↓
7. Click "Save and Continue"
         ↓
8. Write label file to disk
         ↓
9. Load next pair (repeat from step 5)
```

### File Matching Logic

```
Label suffix: "_labels"
Label file: sample1_labels.tif

1. Extract base: "sample1"
2. Find files starting with "sample1"
3. Find files NOT equal to label file
4. Find files with SAME extension (.tif)
5. Match first found = Image file
```

### Format Validation

```
For each label file:
  1. Read file (scikit-image imread)
  2. Check: Is dtype integer?
  3. Check: Does it load without error?
  4. Add to pairs list or report issue
```

## Citation

If you use Batch Label Inspection in your research, please cite:

```bibtex
@software{napari_tmidas_2024,
  title = {napari-tmidas: Batch Image Processing for Microscopy},
  author = {Mercader Lab},
  year = {2024},
  url = {https://github.com/MercaderLabAnatomy/napari-tmidas}
}
```

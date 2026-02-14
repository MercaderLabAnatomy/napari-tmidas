# Frame Removal Tool

The Frame Removal Tool provides a human-in-the-loop interface for interactively identifying and removing unwanted frames from time-series images in TYX or TZYX format.

## Overview

This tool is designed for cases where you need to manually inspect and remove specific time frames from your data, such as:
- Removing frames with artifacts or poor image quality
- Eliminating frames with motion blur or out-of-focus content
- Cleaning up time series by removing specific time points
- Preparing data for downstream analysis by excluding problematic frames

## Features

- **Interactive Frame Navigation**: Browse through time frames using a slider or navigation buttons
- **Visual Marking**: Mark frames for removal with a checkbox while viewing them
- **Live Preview**: Preview the cleaned result before saving
- **Batch Marking**: Mark multiple frames across the time series
- **Safe Operations**: Prevents removing all frames and warns before clearing marks
- **Multi-format Support**: Works with both TYX (3D) and TZYX (4D) images

## Usage

### 1. Open the Widget

In Napari, go to `Plugins > napari-tmidas > Frame Removal Tool`

### 2. Load Your Image

1. First, load your TYX or TZYX image into Napari (File > Open Files)
2. In the Frame Removal widget, select your image layer from the dropdown
3. The widget will display the image format and number of frames

### 3. Navigate and Mark Frames

- Use the **slider** or **Previous/Next buttons** to navigate through frames
- For each frame you want to remove, check the **"Mark current frame for REMOVAL"** checkbox
- The widget shows:
  - Current frame number (e.g., "Frame: 5 / 20")
  - List of marked frames
  - Total count of marked frames

### 4. Review Your Selections

- The **Marked frames** display shows all frames marked for removal
- You can unmark a frame by navigating to it and unchecking the box
- Use **Clear All Marks** to start over if needed

### 5. Preview (Optional)

Click **Preview Result** to:
- Create a new layer showing the cleaned time series
- Verify that the correct frames were removed
- Check the new dimensions

The preview layer is named `[original_name]_cleaned_preview` and won't affect your original data.

### 6. Save the Result

1. Click **Save Cleaned Image**
2. Choose a location and filename
3. The cleaned image will be saved as a TIFF file with:
   - Only the frames you kept
   - Original data type preserved
   - Compression applied
   - Proper metadata (axes information)

## Example Workflow

### Removing Bad Frames from a Time-Lapse

```python
# 1. Load your time-lapse image in Napari
# File > Open Files > select your_timelapse.tif

# 2. Open Frame Removal Tool
# Plugins > napari-tmidas > Frame Removal Tool

# 3. Select your image layer from the dropdown

# 4. Navigate through frames and mark bad ones:
#    - Frame 3: motion blur → Mark for removal ✓
#    - Frame 7: out of focus → Mark for removal ✓
#    - Frame 15: acquisition artifact → Mark for removal ✓

# 5. Preview the result
# Click "Preview Result" to see cleaned_preview layer

# 6. Save the cleaned time series
# Click "Save Cleaned Image"
# Save as: your_timelapse_cleaned.tif
```

## Supported Formats

### Input Formats

- **TYX (3D)**: Time series of 2D images
  - Example: (100, 512, 512) = 100 time frames of 512×512 images
  
- **TZYX (4D)**: Time series of 3D z-stacks
  - Example: (50, 20, 512, 512) = 50 time points, each with 20 z-slices

### Output Format

The tool saves cleaned images as **TIFF files** with:
- Same data type as input (uint8, uint16, float32, etc.)
- zlib compression for smaller file sizes
- Proper axes metadata for downstream tools

## Tips and Best Practices

### Navigation Tips

- **Use keyboard shortcuts**: Click on the slider and use arrow keys for quick navigation
- **Jump to frames**: Click on the slider track to jump directly to a frame
- **Review systematically**: Go through frames sequentially to avoid missing artifacts

### Marking Strategy

1. **First pass**: Quickly browse through all frames to identify obvious problems
2. **Mark frames**: Go back and mark problematic frames
3. **Review marks**: Use the marked frames list to verify your selections
4. **Preview**: Always preview before saving to catch mistakes

### Performance Notes

- The tool works with large images by keeping data in memory
- For very large datasets (>4GB), consider:
  - Closing other applications to free memory
  - Processing one region at a time using cropping tools first
  - Using the preview function to verify before saving

### Common Issues

**"Invalid Dimensions" Warning**
- The tool only works with TYX (3D) or TZYX (4D) images
- Check your image dimensions using the Napari layer controls
- Use file conversion tools if needed to get the right format

**"Insufficient Frames" Warning**
- You need at least 2 frames to use this tool
- Single-frame images cannot have frames removed

**"Cannot Remove All Frames" Warning**
- At least one frame must remain in the output
- Unmark some frames before saving

## Technical Details

### Data Handling

- The tool creates a **copy** of your data when saving
- Original image layer is **never modified**
- Removed frames are excluded by indexing along the time dimension
- No interpolation or processing is applied to remaining frames

### Memory Usage

- Full image data is kept in memory during operation
- Preview creates an additional copy of the cleaned data
- For a 1GB image with 10% frames removed:
  - Original: 1GB
  - Preview: ~0.9GB additional
  - Total: ~1.9GB

### File Format

Output TIFF files include:
```python
metadata = {
    'axes': 'TZYX'  # or 'TYX' for 3D input
}
```

This ensures compatibility with:
- ImageJ/Fiji
- Other napari plugins
- Python imaging libraries (tifffile, scikit-image, etc.)

## Integration with Other Tools

### Preprocessing Pipeline

```
Raw Data → Frame Removal Tool → Further Processing
```

Example workflow:
1. **Load raw time-lapse data**
2. **Frame Removal Tool**: Remove bad frames
3. **Intensity normalization**: Process cleaned data
4. **Segmentation**: Analyze clean time series

### Batch Processing Note

This tool is designed for **interactive, human-in-the-loop** quality control. For automated batch frame removal based on objective criteria (e.g., brightness thresholds), consider:
- Writing a custom processing function
- Using intensity-based filtering tools
- Scripting with numpy array operations

## Keyboard Shortcuts

While the widget is focused:
- **Slider + Arrow Keys**: Navigate frames
- **Space**: Toggle mark on current frame (if checkbox is focused)

## Troubleshooting

### Widget doesn't appear
- Ensure napari-tmidas is properly installed
- Restart Napari
- Check Plugins menu for the tool

### Layer selector is empty
- Load an image into Napari first
- Click the refresh button (↻) next to the dropdown
- Ensure your image is an Image layer (not Labels)

### Can't save files
- Ensure you have write permissions to the target directory
- Check that tifffile is installed: `pip install tifffile`
- Verify sufficient disk space for the output file

### Preview layer not visible
- Check if the preview layer is hidden in the layer list
- Adjust contrast limits if needed
- Verify that frames were actually marked for removal

## See Also

- [Basic Processing Functions](basic_processing.md) - For automated frame processing
- [Split TZYX into ZYX TIFs](basic_processing.md#split-tzyx-into-zyx-tifs) - For splitting time series
- [Timepoint Merger](basic_processing.md) - For combining time series

## Requirements

- napari
- numpy
- tifffile (for saving)
- qtpy (for GUI)
- magicgui (for widgets)

All dependencies are automatically installed with napari-tmidas.

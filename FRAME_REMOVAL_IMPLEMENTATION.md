# Frame Removal Feature - Implementation Summary

## Overview

I've successfully created a human-in-the-loop pipeline for removing frames from TZYX or TYX time-series images in napari-tmidas.

## What Was Created

### 1. Main Widget Module
**File:** `src/napari_tmidas/_frame_removal.py`

A Qt-based interactive widget with the following features:
- **Layer Selection**: Choose any TYX or TZYX image layer from the viewer
- **Frame Navigation**: Use slider or Previous/Next buttons to browse through time frames
- **Interactive Marking**: Checkbox to mark frames for removal while viewing them
- **Live Feedback**: Display of current frame, marked frames list, and count
- **Preview**: Generate a preview layer showing the result before saving
- **Save**: Export cleaned time series as TIFF with compression and metadata
- **Safety Features**: 
  - Prevents removing all frames
  - Confirmation before clearing marks
  - Validates image dimensions (3D/4D only)
  - Requires at least 2 frames

### 2. Plugin Registration
**File:** `src/napari_tmidas/napari.yaml`

Added:
- Command registration: `napari-tmidas._frame_removal`
- Widget entry in the Plugins menu as "Frame Removal Tool"

### 3. Comprehensive Documentation
**File:** `docs/frame_removal.md`

Complete user guide including:
- Overview and use cases (artifacts, motion blur, quality control)
- Step-by-step usage instructions
- Supported formats (TYX and TZYX)
- Tips and best practices
- Troubleshooting guide
- Technical details (memory usage, file formats)
- Integration with other tools

### 4. Example Script
**File:** `examples/frame_removal_example.py`

Demonstrates:
- Creating a sample time-lapse with simulated artifacts
- Loading data into napari
- Using the Frame Removal Tool widget
- Complete workflow from loading to saving

### 5. Comprehensive Test Suite
**File:** `src/napari_tmidas/_tests/test_frame_removal.py`

15 test cases covering:
- Widget creation and initialization
- TYX and TZYX image handling
- Frame navigation (slider, buttons)
- Frame marking/unmarking
- Clearing marks
- Creating cleaned data
- Preview functionality
- Saving results
- Invalid dimensions handling
- Insufficient frames handling
- Preventing removal of all frames
- Layer selection and filtering

### 6. Updated Package Files
- **`src/napari_tmidas/__init__.py`**: Added frame_removal_widget to exports
- **`README.md`**: Added Frame Removal to Core Workflows section

## How to Use

### Basic Workflow

1. **Launch napari** and open the plugin:
   ```
   Plugins > napari-tmidas > Frame Removal Tool
   ```

2. **Load your time-series image** into napari (File > Open Files)

3. **Select your image layer** from the dropdown in the widget

4. **Navigate through frames** using the slider or buttons

5. **Mark bad frames** by checking the "Mark current frame for REMOVAL" checkbox

6. **Preview the result** (optional) by clicking "Preview Result"

7. **Save the cleaned image** by clicking "Save Cleaned Image"

### Example Code
```python
# Run the example
python examples/frame_removal_example.py
```

This creates a sample time-lapse with artifacts in frames 6, 13, and 18, then guides you through the removal process.

## Technical Details

### Supported Formats
- **Input**: TYX (3D) or TZYX (4D) numpy arrays
- **Output**: TIFF files with zlib compression and axes metadata

### Architecture
- **Qt-based GUI**: Uses qtpy for cross-platform compatibility
- **Napari integration**: Leverages napari's viewer and layer system
- **Memory efficient**: Works directly with numpy arrays, creates copies only when saving
- **Non-destructive**: Original layers are never modified

### Key Classes and Functions

1. **`FrameRemovalWidget(QWidget)`**: Main widget class
   - Manages UI state and user interactions
   - Handles frame navigation and marking
   - Creates cleaned data and saves results

2. **`frame_removal_widget()`**: Factory function
   - Returns a widget factory for napari integration
   - Follows napari plugin conventions

## Testing

Run the tests:
```bash
pytest src/napari_tmidas/_tests/test_frame_removal.py -v
```

Expected: 15 tests pass

## Integration with Existing Features

This tool complements existing napari-tmidas features:
- **Pre-processing**: Remove bad frames before segmentation or tracking
- **Quality control**: Interactive inspection and cleaning of time-series data
- **Batch processing**: Use with other processing functions in pipeline workflows

## Use Cases

1. **Removing acquisition artifacts**: Motion blur, stage drift, focus issues
2. **Cleaning time-lapse data**: Remove frames with poor image quality
3. **Preparing data for analysis**: Ensure only high-quality frames are processed
4. **Manual quality control**: Human-in-the-loop verification of time series

## Next Steps

The feature is now fully implemented and ready to use! To test:

1. Install napari-tmidas in development mode:
   ```bash
   cd /opt/napari-tmidas
   pip install -e .
   ```

2. Launch napari:
   ```bash
   napari
   ```

3. Find "Frame Removal Tool" in:
   ```
   Plugins > napari-tmidas > Frame Removal Tool
   ```

4. Or run the example:
   ```bash
   python examples/frame_removal_example.py
   ```

## Files Modified/Created

```
napari-tmidas/
├── src/napari_tmidas/
│   ├── __init__.py                      [MODIFIED - Added export]
│   ├── _frame_removal.py                [NEW - Main widget]
│   ├── napari.yaml                      [MODIFIED - Added registration]
│   └── _tests/
│       └── test_frame_removal.py        [NEW - Test suite]
├── docs/
│   └── frame_removal.md                 [NEW - Documentation]
├── examples/
│   └── frame_removal_example.py         [NEW - Example script]
└── README.md                            [MODIFIED - Added feature]
```

## Summary

The Frame Removal Tool provides an intuitive, human-in-the-loop solution for cleaning time-series data in napari. It integrates seamlessly with the existing napari-tmidas ecosystem and follows all plugin conventions and best practices.

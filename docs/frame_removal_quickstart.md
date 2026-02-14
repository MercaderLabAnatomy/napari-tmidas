# Frame Removal Tool - Quick Start Guide

## What is it?

An interactive widget for removing unwanted frames from time-series microscopy images (TYX or TZYX format).

## When to use it?

- Remove frames with motion blur or artifacts
- Eliminate out-of-focus time points
- Clean up time-lapse data before analysis
- Manually curate your time series

## Quick Start (3 steps)

### 1. Open the tool
```
Plugins > napari-tmidas > Frame Removal Tool
```

### 2. Select your time-series image
- Use the dropdown to select your TYX or TZYX layer
- Navigate frames with the slider or buttons

### 3. Mark and save
- Check "Mark current frame for REMOVAL" for bad frames
- Click "Preview Result" to verify
- Click "Save Cleaned Image" to export

## Example

Try it out with the provided example:

```bash
cd /opt/napari-tmidas
python examples/frame_removal_example.py
```

This creates a sample time-lapse with artifacts that you can practice removing.

## Tips

✅ **DO:**
- Use the preview function before saving
- Navigate systematically through all frames
- Check the marked frames list to verify selections

❌ **DON'T:**
- Try to remove all frames (at least one must remain)
- Forget to preview before saving to a critical location
- Use on non-time-series data (only works with TYX or TZYX)

## See Full Documentation

For detailed information, see: [docs/frame_removal.md](../docs/frame_removal.md)

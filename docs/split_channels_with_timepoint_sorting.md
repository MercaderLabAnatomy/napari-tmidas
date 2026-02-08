# Split Color Channels with Timepoint Sorting

## Overview

The Split Color Channels function now supports automatic organization of images by their timepoint count before splitting channels. This is useful when you have a mixed dataset with images containing different numbers of timepoints (e.g., T=1, T=10, T=50) that need to be organized before processing.

## When to Use

Use timepoint sorting when:
- You have images with **varying or unknown timepoint counts** in the same folder
- You want to **organize images by their temporal dimension** before splitting channels
- You need to **process each timepoint group separately** to maintain temporal organization

## How It Works

### Without Timepoint Sorting (Default)

```
Input folder:
  ├── image1_T1.tif  (1 timepoint)
  ├── image2_T10.tif (10 timepoints)
  └── image3_T50.tif (50 timepoints)

Output folder after channel splitting:
  ├── image1_T1_ch1_split.tif
  ├── image1_T1_ch2_split.tif
  ├── image1_T1_ch3_split.tif
  ├── image2_T10_ch1_split.tif
  ├── image2_T10_ch2_split.tif
  ├── image2_T10_ch3_split.tif
  ├── image3_T50_ch1_split.tif
  ├── image3_T50_ch2_split.tif
  └── image3_T50_ch3_split.tif
```

### With Timepoint Sorting Enabled

```
Input folder:
  ├── image1_T1.tif  (1 timepoint)
  ├── image2_T10.tif (10 timepoints)
  └── image3_T50.tif (50 timepoints)

Output folder after channel splitting:
  ├── T1/
  │   ├── image1_T1.tif (copy)
  │   ├── image1_T1_ch1_split.tif
  │   ├── image1_T1_ch2_split.tif
  │   └── image1_T1_ch3_split.tif
  ├── T10/
  │   ├── image2_T10.tif (copy)
  │   ├── image2_T10_ch1_split.tif
  │   ├── image2_T10_ch2_split.tif
  │   └── image2_T10_ch3_split.tif
  └── T50/
      ├── image3_T50.tif (copy)
      ├── image3_T50_ch1_split.tif
      ├── image3_T50_ch2_split.tif
      └── image3_T50_ch3_split.tif
```

## Usage Instructions

### Step 1: Select Your Files

1. Open the **Batch Image Processing** widget in Napari
2. Click **Browse** to select your input folder containing mixed-timepoint images
3. The file list will show all detected image files

### Step 2: Configure Split Channels with Timepoint Sorting

1. Select **"Split Color Channels"** from the processing function dropdown
2. Configure the parameters:
   - **num_channels**: Number of color channels in your images (e.g., 3 for RGB)
   - **time_steps**: **Leave at 0** (the sorting feature auto-detects timepoints per image)
   - **output_format**: Choose "python" or "fiji" dimension ordering
3. **Enable timepoint sorting**:
   - Check the **"sort_by_timepoints"** checkbox
   - This tells the function to organize files by timepoint count first

**Important:** When using `sort_by_timepoints`, always leave `time_steps` at 0. The function will automatically detect the timepoint count for each image from its metadata.

### Step 3: Set Output Folder

1. Click **Browse** next to the output folder field
2. Choose where you want the processed files saved
3. The function will create T1/, T10/, T50/, etc. subfolders here automatically

### Step 4: Run Batch Processing

1. Click **Start Batch Processing**
2. The function will:
   - First analyze each image to determine its timepoint count
   - Create timepoint subfolders (T1/, T10/, T50/, etc.)
   - Copy original files into appropriate subfolders
   - Split channels for each file
   - Save split channels into the same timepoint subfolder

## Technical Details

### Timepoint Detection

The function determines timepoint count by reading TIFF metadata:
- Checks for explicit 'T' axis in image dimensions
- Looks for ImageJ metadata containing frame information
- Falls back to sensible defaults for ambiguous cases

### File Organization

- **Original files are copied** (not moved) to timepoint subfolders
- This preserves your original data
- Split channel outputs are saved within each timepoint subfolder
- Folder names use the format: `T{num_timepoints}` (e.g., T1, T10, T50)

### Performance Considerations

- Timepoint analysis reads **only metadata**, not full image data
- File copying happens once at the start
- Channel splitting then processes normally
- The sorting step adds minimal overhead

## Example Workflow

### Scenario: Mixed Timepoint Dataset

You have a folder with microscopy images:
- Some images are single timepoints (acquired at one moment)
- Some are time series with 10 timepoints
- Some are long time series with 50 timepoints
- Each image has 3 color channels that need to be split
- **You don't know which file has how many timepoints**

**Steps:**

1. Load the Batch Image Processing widget
2. Select your input folder
3. Choose "Split Color Channels"
4. Set parameters:
   - num_channels = 3
   - **time_steps = 0** ← Keep at 0 because timepoints vary/unknown
   - output_format = "python"
   - **sort_by_timepoints = True** ✓ ← Enable auto-detection
5. Set output folder
6. Click "Start Batch Processing"

**Result:**

Your output folder will have:
- T1/ folder with single-timepoint images and their split channels
- T10/ folder with 10-timepoint series and their split channels
- T50/ folder with 50-timepoint series and their split channels

Now you can process each timepoint group separately!

## Comparison with Standalone Sorting

### Using Split Channels with Timepoint Sorting
- ✅ One-step process: sorts and splits in single operation
- ✅ Convenient for when you always split channels after sorting
- ✅ Integrated into batch processing workflow

### Using Standalone sort_images_by_timepoints.py Script
- ✅ More flexible: sort without being tied to channel splitting
- ✅ Can be used as preprocessing before any other operation
- ✅ Supports advanced options (move instead of copy, target specific timepoints)

Both approaches are valid! Choose based on your workflow needs.

## Troubleshooting

### "Could not determine timepoints" Warning

**Cause:** The image file doesn't have clear timepoint metadata.

**Solutions:**
- Check if your images have proper TIFF metadata
- Manually organize files if metadata is unreliable
- Use the standalone sorting script with custom parameters

### Files Not Appearing in Timepoint Subfolders

**Cause:** Files may have been skipped due to unsupported format or metadata issues.

**Solutions:**
- Check the console output for specific error messages
- Ensure files are valid TIFF format
- Verify that timepoint metadata is present

### All Files Going to T1 Folder

**Cause:** Images may all be single-timepoint acquisitions, or metadata isn't indicating multiple timepoints.

**Solutions:**
- Verify that your images actually contain multiple timepoints
- Check the T dimension in your image metadata
- Consider if your files might legitimately be single-timepoint

## See Also

- [Basic Processing](basic_processing.md) - Standard channel splitting
- [Batch Processing Guide](batch_label_inspection.md) - General batch processing workflows
- `sort_images_by_timepoints.py` - Standalone timepoint sorting script

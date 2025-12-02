# Basic Image Processing Functions

This document describes the basic image processing functions available in napari-tmidas for batch processing.

## Label Image Operations

### Labels to Binary
**Suffix**: `_binary`

Convert a label image to a binary mask where all non-zero pixels become 255 and zero pixels remain 0.

**Use case**: Simplify multi-label images to binary masks for downstream analysis.

**Parameters**: None

**Example**:
- Input: Label image with values [0, 1, 2, 3, ...]
- Output: Binary image with values [0, 255]

---

### Invert Binary Labels
**Suffix**: `_inverted`

Invert a binary label image: non-zero pixels become 0, zero pixels become 255.

**Use case**: Switch foreground and background in binary masks.

**Parameters**: None

---

### Filter Label by ID
**Suffix**: `_filtered`

Keep only a specific label ID and set all other labels to background (0).

**Parameters**:
- `label_id` (int, default=1): The label ID to keep

**Use case**: Extract a single object of interest from a multi-label image.

---

### Mirror Labels
**Suffix**: `_mirrored`

Mirror labels at their largest slice area along a specified axis. The slice with the maximum area is identified and labels are mirrored relative to that position.

**Parameters**:
- `axis` (int, default=0): Axis along which to mirror the labels

**Use case**: Symmetrical tissue reconstruction or creating mirrored biological structures.

**Technical details**:
- Finds the slice with maximum non-zero area
- Mirrors labels from that position
- Offset labels to avoid ID conflicts
- Preserves original image shape

---

### Intersect Label Images
**Suffix**: `_intersected`

Compute the voxel-wise intersection of paired label images identified by suffix.

**Parameters**:
- `primary_suffix` (str, default="_a.tif"): Suffix of the primary label image
- `secondary_suffix` (str, default="_b.tif"): Suffix of the paired label image

**Use case**: Find overlapping regions between two segmentation results.

**Expected file structure**:
```
folder/
├── sample_a.tif  (primary - will be processed)
├── sample_b.tif  (secondary - paired file)
```

**Technical details**:
- Only processes files ending with `primary_suffix`
- Automatically loads the corresponding `secondary_suffix` file
- Retains label IDs from primary image where overlap occurs
- Handles dimension mismatches by centering and aligning

---

### Keep Slice Range by Area
**Suffix**: `_area_range`

Zero out label content outside the minimum and maximum area slice range, preserving image shape for alignment.

**Parameters**:
- `axis` (int, default=0): Axis index representing the slice dimension

**Use case**: Remove noise slices at the beginning or end of a 3D stack while maintaining dimensions.

**Technical details**:
- Measures non-zero pixel count per slice
- Identifies slices with minimum and maximum area
- Zeros out content outside this range
- Preserves original image dimensions

---

## Intensity Image Operations

### Gamma Correction
**Suffix**: `_gamma`

Apply gamma correction to enhance bright or dark regions of an image.

**Parameters**:
- `gamma` (float, default=1.0, range=0.1-10.0): Gamma correction factor
  - `gamma > 1.0`: Enhance bright regions
  - `gamma < 1.0`: Enhance dark regions
  - `gamma = 1.0`: No change

**Use case**: Adjust image contrast for visualization or improve feature detection.

---

### Max Z Projection
**Suffix**: `_max_z`

Maximum intensity projection along the z-axis, reducing a 3D stack to 2D.

**Parameters**: None

**Use case**: Create 2D overview of 3D volumetric data.

---

### Max Z Projection (TZYX)
**Suffix**: `_maxZ_tzyx`

Memory-efficient maximum intensity projection along the Z-axis for 4D time-series data.

**Parameters**: None

**Use case**: Reduce 4D TZYX data to 3D TYX while preserving temporal information.

**Technical details**:
- Processes time points individually to minimize memory usage
- Plane-by-plane comparison for efficient computation
- Preserves original dtype

---

## Channel Operations

### Split Color Channels
**Suffix**: `_split`

Split multi-channel images into separate single-channel arrays.

**Parameters**:
- `num_channels` (int, default=3, range=2-4): Number of color channels
- `time_steps` (int, default=0, range=0-1000): Number of time steps (0 = not time series)
- `output_format` (str, default="python", options=["python", "fiji"]): Dimension order format

**Use case**: Separate RGB or multi-channel microscopy images for individual channel analysis.

**Technical details**:
- Automatically detects channel axis
- Handles 2D (YXC), 3D (ZYXC), 4D (TZYXC) data
- "fiji" format: Reorders to TZYX for ImageJ compatibility
- "python" format: Standard numpy dimension order

---

### Merge Color Channels
**Suffix**: `_merged_colors`

Merge separate channel images from a folder into a single multi-channel image.

**Parameters**:
- `channel_substring` (str, default="_channel_"): Substring before channel number in filename

**Use case**: Combine separately processed channels back into multi-channel format.

**Expected file structure**:
```
folder/
├── sample_channel_0.tif
├── sample_channel_1.tif
├── sample_channel_2.tif
```

**Technical details**:
- Uses regex to find channel pattern: `{substring}{1-2 digit number}`
- Only processes the primary channel (lowest number)
- Stacks channels as last dimension
- All channels must have matching dimensions

---

### RGB to Labels
**Suffix**: `_labels`

Convert RGB images to label images using a color map.

**Parameters**:
- `blue_label` (int, default=1, range=0-255): Label value for blue objects
- `green_label` (int, default=2, range=0-255): Label value for green objects
- `red_label` (int, default=3, range=0-255): Label value for red objects

**Use case**: Convert color-coded annotations or segmentations to label images.

**Color mapping**:
- Pure blue (0, 0, 255) → `blue_label`
- Pure green (0, 255, 0) → `green_label`
- Pure red (255, 0, 0) → `red_label`
- All other colors → 0 (background)

---

## Time Series Operations

### Split TZYX into ZYX TIFs
**Suffix**: `_split`

Split a 4D TZYX image stack into separate 3D ZYX TIF files for each time point using parallel processing.

**Parameters**:
- `output_name_format` (str, default="{basename}_t{timepoint:03d}"): Format for output filenames
- `preserve_scale` (bool, default=True): Preserve scale/resolution metadata
- `use_compression` (bool, default=True): Apply zlib compression
- `num_workers` (int, default=4, range=1-16): Number of parallel workers

**Use case**: Split time-lapse data for individual time point analysis or downstream tools that don't support 4D data.

**Technical details**:
- Parallel file saving for efficiency
- Preserves metadata when possible
- Uses Dask for memory-efficient processing
- Automatically creates BigTIFF for large files (>4GB)

**Output naming examples**:
- With default format: `sample_t000.tif`, `sample_t001.tif`, etc.
- Custom format `{basename}_time{timepoint}`: `sample_time0.tif`, `sample_time1.tif`

**Important**: Set thread count to 1 in the batch processing interface, as this function manages its own parallelization internally.

---

## Installation Requirements

Most basic functions work with numpy only. Optional dependencies:
- `tifffile`: For TIFF file operations (intersect, split operations)
- `scikit-image`: Alternative for image I/O
- `dask`: For memory-efficient large array operations (TZYX splitting)

Install optional dependencies:
```bash
pip install tifffile scikit-image dask
```

---

## Usage Tips

1. **Batch Processing**: All functions work with the batch processing widget
2. **File Naming**: Output files automatically append the specified suffix
3. **Dimension Order**: For ambiguous cases, functions make reasonable assumptions
4. **Error Handling**: Functions validate inputs and provide informative error messages
5. **Memory Efficiency**: Large operations (TZYX splitting) use chunked processing

---

## Technical Notes

- All functions preserve the original input dtype when possible
- Label operations maintain label ID integrity
- Intensity operations normalize and rescale appropriately
- Parallel operations respect the specified worker count
- File operations create directories automatically if needed

# All Batch Processing Functions

Complete reference of all functions available in the batch processing widget.
Functions are grouped by category. AI-based functions that require deep-learning models or dedicated environments link to their own documentation.

---

## Label Image Operations

| Function | Suffix | Description |
|---|---|---|
| **Labels to Binary** | `_binary` | Convert a label image to a binary mask (non-zero → 255, zero → 0) |
| **Invert Binary Labels** | `_inverted` | Invert a binary label image (non-zero → 0, zero → 255) |
| **Binary to Labels** | `_labels` | Convert a binary image to instance labels via connected components |
| **Filter Label by ID** | `_filtered` | Keep only a specific label ID; set all others to background (0) |
| **Mirror Labels** | `_mirrored` | Mirror labels at the slice with the largest area along a chosen axis |
| **Intersect Label Images** | `_intersected` | Voxel-wise intersection of paired label images identified by suffix |
| **Keep Slice Range by Area** | `_area_range` | Zero out content outside the min/max non-zero area slice range |
| **Remove Small Labels** | `_rm_small` | Remove connected components smaller than a pixel/voxel count threshold |
| **Merge Small Labels to Neighbors** | `_merged` | Merge small labels into their largest touching neighbor |
| **Semantic to Instance Segmentation** | `_instance` | Convert a semantic mask to instance labels via connected components |
| **Resize Labels (Nearest, SciPy)** | `_scaled` | Isotropic resize of a label image using nearest-neighbor interpolation |
| **Subdivide Labels into 3 Layers** | `_layers` | Split each label object into 3 concentric layers with unique IDs |
| **RGB to Labels** | `_labels` | Convert an RGB image to a label image using a fixed color map |

---

## Intensity Image Filters

| Function | Suffix | Description |
|---|---|---|
| **Gaussian Blur** | `_blurred` | Gaussian blur (scipy). Supports dimension hints (YX, ZYX, TYX) for 2D or 3D blur |
| **Median Filter** | `_median` | Median filter for noise reduction (scipy). Supports dimension hints |
| **CLAHE (Adaptive Histogram Equalization)** | `_clahe` | Enhance local contrast using CLAHE (skimage). Good for dark images with weak features |
| **Gamma Correction** | `_gamma` | Apply gamma correction (>1 enhances bright regions, <1 enhances dark) |
| **Invert Image** | `_inverted` | Invert pixel values using skimage's dtype-aware invert |
| **Rolling Ball Background Subtraction** | `_rollingball` | Remove uneven background (like ImageJ's rolling ball algorithm) |
| **Percentile Threshold (Keep Brightest)** | `_percentile` | Zero out pixels below a brightness percentile; keep or binarize the rest |
| **Adaptive Threshold (Bright Bias)** | `_adaptive_bright` | Local adaptive thresholding biased towards bright regions |

---

## Thresholding & Segmentation

| Function | Suffix | Description |
|---|---|---|
| **Manual Thresholding (8-bit)** | `_thresh` | Fixed-value threshold to produce a binary image |
| **Otsu Thresholding (semantic)** | `_otsu_semantic` | Otsu auto-threshold → binary image. Supports per-frame/slice processing |
| **Otsu Thresholding (instance)** | `_otsu_labels` | Otsu auto-threshold → instance label image |

---

## Projections, Conversion & Resizing

| Function | Suffix | Description |
|---|---|---|
| **Max Z Projection** | `_max_z` | Maximum intensity projection along Z, reducing 3D to 2D |
| **Max Z Projection (TZYX)** | `_maxZ_tzyx` | Memory-efficient max-Z projection for 4D TZYX data → TYX |
| **Convert to 8-bit (uint8)** | `_uint8` | Rescale and convert any image to 8-bit uint8 |
| **Resize Image by YX Scale (skimage)** | `_yx_resized` | Resize YX planes by a scale factor; preserves T/Z axes. For TIFF inputs |
| **Resize Zarr by YX Scale (OME-Zarr native)** | `_yx_resized` | Same as above but reads/writes OME-Zarr natively with preserved pyramid and metadata |

---

## Channel Operations

| Function | Suffix | Description |
|---|---|---|
| **Split Color Channels** | `_split` | Split a multi-channel image into separate single-channel files |
| **Merge Color Channels** | `_merged_colors` | Merge separate channel files from a folder into one multi-channel image |

---

## Time Series Operations

| Function | Suffix | Description |
|---|---|---|
| **Split TZYX into ZYX TIFs** | `_split` | Save each time point of a 4D stack as a separate 3D ZYX TIF (parallel) |
| **Merge Timepoints** | `_merge_timeseries` | Merge per-timepoint files in a folder into a single time-series stack |

---

## Analysis

| Function | Suffix | Description |
|---|---|---|
| **Extract Regionprops to CSV** | `_regionprops` | Measure region properties (area, centroid, intensity, …) for all labels; saves one CSV per folder. See [regionprops_analysis.md](regionprops_analysis.md) |
| **Regionprops Summary Statistics** | `_regionprops_summary` | Per-file summary statistics (count, mean, std, …) of regionprops. See [regionprops_summary.md](regionprops_summary.md) |
| **Filter Labels by Intensity (K-medoids)** | `_intensity_filtered` | Remove low-intensity labels using k-medoids clustering on paired intensity images. See [intensity_label_filter.md](intensity_label_filter.md) |

---

## Visualization & Cropping

| Function | Suffix | Description |
|---|---|---|
| **Grid View: Intensity + Labels Overlay** | `_grid_overlay.tif` | Assemble selected images into a grid with optional colored label overlay. See [grid_view_overlay.md](grid_view_overlay.md) |
| **Label-Based Cropping** | `_cropped` | Crop images using user-drawn label ROIs with optional Z/time expansion. See [label_based_cropping.md](label_based_cropping.md) |

---

## Utilities

| Function | Suffix | Description |
|---|---|---|
| **Compress with Zstandard** | `_compressed` | Compress the output file with Zstandard (requires `pzstd`) |

---

## AI-Based Processing

These functions require deep-learning models or dedicated conda environments. Follow the linked documentation for setup and usage.

| Function | Suffix | Documentation |
|---|---|---|
| **Cellpose-SAM Segmentation** | `_labels` | Instance segmentation with Cellpose 4. See [cellpose_segmentation.md](cellpose_segmentation.md) |
| **Convpaint Prediction** | `_convpaint_labels` | Semantic/instance segmentation with a pretrained Convpaint model. See [convpaint_prediction.md](convpaint_prediction.md) |
| **CAREamics Denoise (N2V/CARE)** | `_denoised` | Denoising with Noise2Void or CARE models. See [careamics_denoising.md](careamics_denoising.md) |
| **Spotiflow Spot Detection** | `_spot_labels` | Fluorescence spot detection using Spotiflow. See [spotiflow_detection.md](spotiflow_detection.md) |
| **Track Cells with Trackastra** | `_tracked` | Cell tracking using the TrackAstra deep-learning tracker. See [trackastra_tracking.md](trackastra_tracking.md) |
| **Track Cells with Ultrack (Segmentation Ensemble)** | `_ultrack` | Cell tracking from a segmentation ensemble via Ultrack. See [ultrack_tracking.md](ultrack_tracking.md) |
| **VisCy Virtual Staining** | `_virtual_stain` | Virtual staining of phase/DIC images using VSCyto3D. See [viscy_virtual_staining.md](viscy_virtual_staining.md) |

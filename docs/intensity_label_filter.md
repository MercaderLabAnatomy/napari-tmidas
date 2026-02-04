# Intensity-Based Label Filtering

This module provides processing functions for filtering labels based on their intensity values using k-medoids clustering.

## Overview

When working with segmented images (label images), you often want to filter out labels with low signal-to-noise ratio. This module provides two clustering-based approaches to automatically determine intensity thresholds and filter out low-quality labels.

## Functions

### Filter Labels by Intensity (2-medoids)

**Use case:** Signal is spread over larger regions mixed with regions where signal-to-noise ratio is bad.

This function uses 2-medoids clustering to separate labels into two groups:
- **Low intensity cluster:** Labels with poor signal-to-noise ratio
- **High intensity cluster:** Labels with good signal quality

The threshold is set at the midpoint between the two cluster centers, and all labels below this threshold are removed.

**Parameters:**
- `intensity_folder`: Path to folder containing intensity images (must have same filenames as label images)
- `save_stats`: Whether to save clustering statistics to CSV (default: True)

**Output:**
- Filtered label image with low-intensity labels removed
- Statistics saved to `intensity_filter_stats/2medoids_stats.csv` (if enabled)

### Filter Labels by Intensity (3-medoids)

**Use case:** There are regions with bad signal-to-noise ratio, regions with concentrated strong intensity, or regions with signal spread over larger regions.

This function uses 3-medoids clustering to separate labels into three groups:
- **Low intensity cluster:** Labels with poor signal-to-noise ratio
- **Medium intensity cluster:** Labels with moderate signal spread
- **High intensity cluster:** Labels with concentrated strong signal

The threshold is set at the midpoint between the lowest and second-lowest cluster centers, and all labels in the low intensity cluster are removed.

**Parameters:**
- `intensity_folder`: Path to folder containing intensity images (must have same filenames as label images)
- `save_stats`: Whether to save clustering statistics to CSV (default: True)

**Output:**
- Filtered label image with low-intensity labels removed
- Statistics saved to `intensity_filter_stats/3medoids_stats.csv` (if enabled)

## How to Use

1. **Prepare your data:**
   - Label images in one folder (e.g., `segmentation_results/`)
   - Corresponding intensity images in another folder (e.g., `original_images/`)
   - Files must have matching names (e.g., `image001.tif` in both folders)

2. **Run the processing function:**
   - Open napari-tmidas
   - Select "Batch Processing" from the plugin menu
   - Choose your label folder as the input
   - Select "Filter Labels by Intensity (2-medoids)" or "Filter Labels by Intensity (3-medoids)"
   - Specify the intensity folder path in the parameters
   - Run the processing

3. **Review results:**
   - Filtered label images will be saved with suffix `_intensity_filtered_2med` or `_intensity_filtered_3med`
   - Check the statistics CSV file in the `intensity_filter_stats` subfolder
   - Statistics include:
     - Number of labels in each cluster
     - Mean intensity of each cluster
     - Threshold value used
     - Number of labels kept vs. removed

## Installation

The intensity filtering functions require the `scikit-learn-extra` package for k-medoids clustering:

The `scikit-learn-extra` package is automatically installed as part of napari-tmidas, so no additional installation is needed.

## Algorithm Details

### K-medoids Clustering

K-medoids is similar to k-means but uses actual data points (medoids) as cluster centers instead of calculated means. This makes it more robust to outliers.

**Process:**
1. Calculate mean intensity for each label using regionprops
2. Apply k-medoids clustering (k=2 or k=3) to the intensity values
3. Sort clusters by intensity (low to high)
4. Set threshold as midpoint between lowest and second-lowest cluster centers
5. Remove all labels below the threshold

### Choosing Between 2 and 3 Medoids

**Use 2-medoids when:**
- You have a simple bimodal distribution (good vs. bad labels)
- Signal is relatively uniform but with varying signal-to-noise ratio
- You want a straightforward separation between keep/discard

**Use 3-medoids when:**
- Your data has three distinct populations
- You have both diffuse signal (moderate intensity) and punctate signal (high intensity)
- You want to separate background noise, true signal, and strong features

## Example Output

```
📊 image001.tif:
   Total labels: 150
   Low intensity cluster: 45 labels (mean: 12.35)
   High intensity cluster: 105 labels (mean: 87.21)
   Threshold: 49.78
   Keeping 105 labels, removing 45 labels
```

## Technical Notes

- The functions automatically load corresponding intensity images using napari's reader
- Clustering uses PAM (Partitioning Around Medoids) algorithm
- Random state is set to 42 for reproducibility
- Original label image dtype is preserved in the output
- Empty label images return zero-filled images
- Statistics are appended to CSV files for batch processing

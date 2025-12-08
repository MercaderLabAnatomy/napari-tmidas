# Regionprops Summary Statistics

This processing function calculates aggregate statistics for region properties across all labels in each file, providing a high-level overview of your segmentation results.

## Overview

While the standard "Extract Regionprops to CSV" function provides detailed per-label measurements, the **Regionprops Summary Statistics** function provides file-level or dimension-level aggregated statistics. This is useful for:

- Getting quick overview statistics per file/timepoint/channel
- Comparing label counts across conditions
- Analyzing population-level properties (mean size, intensity distributions)
- Quality control of segmentation results

## Output

The function creates a CSV file with summary statistics including:

- **label_count**: Number of labels/regions detected
- **size_sum/mean/median/std**: Statistics for label sizes (pixel counts)
- **mean_int_sum/mean/median/std**: Statistics for mean intensity values
- **median_int_sum/mean/median/std**: Statistics for median intensity values
- **std_int_sum/mean/median/std**: Statistics for intensity standard deviations
- **max_int_sum/mean/median/std**: Statistics for maximum intensity values
- **min_int_sum/mean/median/std**: Statistics for minimum intensity values

## Parameters

### File Selection
- **label_suffix**: Suffix to identify label files (e.g., `_otsu_semantic.tif`). Only files with this suffix are processed. The suffix is removed to find matching intensity images.

### Processing Options
- **max_spatial_dims**:
  - `2`: Process as 2D slices (YX)
  - `3`: Process as 3D volumes (ZYX)

- **group_by_dimensions**:
  - `False`: One summary row per file
  - `True`: Separate summary rows for each T/C/Z dimension value

- **overwrite_existing**: Overwrite existing CSV file on first image

### Property Selection
Enable/disable statistics for specific properties:
- **size**: Label size (area/volume)
- **mean_intensity**: Mean intensity per label
- **median_intensity**: Median intensity per label
- **std_intensity**: Standard deviation of intensity per label
- **max_intensity**: Maximum intensity per label
- **min_intensity**: Minimum intensity per label

## Example Use Cases

### Case 1: Cell Count per Timepoint
**Scenario**: Track total cell count over time in a timelapse

**Settings**:
- label_suffix: `_cells.tif`
- group_by_dimensions: `True`
- size: `True`

**Output**: One row per timepoint showing cell count and size statistics

### Case 2: Intensity Distribution Summary
**Scenario**: Compare signal intensity across multiple samples

**Settings**:
- label_suffix: `_nuclei.tif`
- group_by_dimensions: `False`
- mean_intensity: `True`
- median_intensity: `True`
- std_intensity: `True`

**Output**: One row per file showing intensity statistics across all nuclei

### Case 3: Multi-Channel Analysis
**Scenario**: Analyze objects across different channels with dimensionality

**Settings**:
- label_suffix: `_semantic.tif`
- group_by_dimensions: `True`
- max_spatial_dims: `3`
- All intensity options: `True`

**Output**: Separate rows for each T/C combination with full statistics

## Example Output

Without grouping (`group_by_dimensions=False`):
```csv
filename,label_count,size_sum,size_mean,size_median,size_std,mean_int_mean,mean_int_median,mean_int_std
image1_cells.tif,152,45230,297.6,285.0,89.3,145.2,142.1,23.5
image2_cells.tif,178,53401,300.0,290.5,95.1,148.9,145.3,25.1
```

With grouping (`group_by_dimensions=True`):
```csv
filename,T,label_count,size_sum,size_mean,size_median,size_std,mean_int_mean,mean_int_median,mean_int_std
timelapse_cells.tif,0,152,45230,297.6,285.0,89.3,145.2,142.1,23.5
timelapse_cells.tif,1,178,53401,300.0,290.5,95.1,148.9,145.3,25.1
timelapse_cells.tif,2,165,49102,297.6,288.0,91.2,147.3,143.8,24.2
```

## Comparison with Extract Regionprops

| Feature | Extract Regionprops | Regionprops Summary |
|---------|-------------------|-------------------|
| Output granularity | One row per label | One row per file/dimension |
| File size | Large (all labels) | Small (aggregated) |
| Use case | Detailed analysis | Quick overview/QC |
| Processing speed | Slower (more I/O) | Faster (less I/O) |
| Statistics | Raw measurements | Sum/mean/median/std |

## Tips

1. **Use both functions**: Run summary statistics first for QC, then detailed regionprops for in-depth analysis
2. **Memory efficiency**: Summary statistics use less memory since only aggregates are stored
3. **Dimension grouping**: Enable for timelapse/multi-channel data to track changes over time/channels
4. **Property selection**: Disable unused properties to speed up processing and reduce output size

## Integration with Analysis Pipeline

```
Raw Images → Segmentation → Regionprops Summary (QC) → Regionprops Detailed (Analysis)
```

The summary function is ideal for:
- Initial quality control of segmentation
- Detecting outliers or failed segmentations
- Comparing experimental conditions at a high level
- Generating figures showing population-level trends

Use the detailed regionprops function when you need:
- Single-cell/single-object analysis
- Tracking individual objects
- Detailed spatial information
- Advanced statistical modeling on individual measurements

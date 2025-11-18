# Regionprops Analysis Documentation

## Extract Regionprops to CSV

This processing function extracts region properties (regionprops) from all label images in a folder and saves them to a single CSV file. It is **dimension-agnostic**, meaning it can handle images with different numbers of dimensions and treats non-spatial dimensions (like Time or Channel) as grouping variables.

### Features

- **Automatic dimension detection**: Works with 2D (YX), 3D (ZYX), 4D (TZYX, CZYX), and 5D (TCZYX) label images
- **Comprehensive properties**: Extracts area, centroid, bounding box, solidity, extent, and more
- **Grouping by dimensions**: For time-series or multi-channel data, adds T, C, or Z columns to track regions across frames
- **Single CSV output**: All regions from all files are saved to one convenient CSV file
- **Flexible spatial dimensions**: Can process either 2D (YX) or 3D (ZYX) as the spatial unit

### Usage in Napari

1. Open napari and go to **Plugins > T-MIDAS > Image Processing**
2. Browse to your folder containing label images (`.tif`, `.npy`, etc.)
3. Select **"Extract Regionprops to CSV"** from the processing functions dropdown
4. Configure parameters:
   - **max_spatial_dims**: Set to `3` for 3D analysis (ZYX) or `2` for 2D analysis (YX)
   - **overwrite_existing**: Set to `True` to overwrite existing CSV files
5. **IMPORTANT**: Set **thread count to 1** (this function processes entire folders at once)
6. Click **"Start Batch Processing"**

The function will create a CSV file named `<folder_name>_regionprops.csv` in the parent directory.

### Output CSV Structure

The output CSV contains one row per labeled region with the following columns:

#### Identifier Columns
- `filename`: Name of the source label image file
- `T`: Time index (for 4D/5D time-series data)
- `C`: Channel index (for multi-channel data)
- `label`: The label ID of the region

#### Spatial Properties
- `area`: Number of pixels/voxels in the region
- `centroid_x`, `centroid_y`, `centroid_z`: Center coordinates
- `bbox_min_x`, `bbox_max_x`, etc.: Bounding box coordinates

#### Shape Properties (when available)
- `eccentricity`: How elongated the region is (2D only)
- `solidity`: Ratio of region area to convex hull area
- `extent`: Ratio of region area to bounding box area
- `perimeter`: Perimeter length (2D only)

### Example: Time-Series Analysis

For a folder containing 4D label images (TZYX), the CSV will include a `T` column:

```
filename,T,label,area,centroid_z,centroid_y,centroid_x,...
cell_tracking.tif,0,1,5000,10.5,120.3,85.7,...
cell_tracking.tif,0,2,4800,12.1,200.5,150.2,...
cell_tracking.tif,1,1,5200,10.8,122.1,86.5,...
cell_tracking.tif,1,2,4900,12.3,202.3,151.0,...
```

This makes it easy to track how properties change over time in tools like pandas, R, or Excel.

### Programmatic Usage

You can also use the function directly in Python:

```python
from napari_tmidas.processing_functions.regionprops_analysis import (
    analyze_folder_regionprops,
)

# Analyze all label images in a folder
df = analyze_folder_regionprops(
    folder_path="/path/to/label/images",
    output_csv="/path/to/output.csv",
    max_spatial_dims=3,  # 3 for ZYX, 2 for YX
)

# Now analyze the results with pandas
print(f"Total regions: {len(df)}")
print(df.groupby("T")["area"].mean())  # Average area per timepoint
```

### Notes

- Requires `pandas` to be installed: `pip install pandas`
- The function uses a cache to avoid processing the same folder multiple times in one session
- Call `reset_regionprops_cache()` if you need to reprocess a folder
- Some properties (like `eccentricity` and `perimeter`) are only available for 2D regions

### Use Cases

- **Cell tracking**: Extract cell properties across time to analyze growth, movement, or division
- **Multi-channel analysis**: Compare properties of objects in different channels
- **High-throughput analysis**: Process entire experiments at once and analyze in your favorite data analysis tool
- **Quality control**: Check segmentation quality by examining region properties

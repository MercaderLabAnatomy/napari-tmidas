# Multi-Format Microscopy Image File Conversion

The File Conversion widget enables batch conversion of proprietary microscopy image formats to open-source, scalable formats (TIFF and OME-Zarr) while preserving all spatial metadata.

## Overview

Convert large microscopy datasets from manufacturer-specific formats to standardized formats with automatic metadata extraction:

- **Input Formats**: Leica LIF, Nikon ND2, Zeiss CZI, TIFF-based whole slide images (NDPI/SVS), Acquifer datasets
- **Output Formats**: TIFF (for small datasets) or OME-Zarr (recommended for large datasets)
- **Metadata Preservation**: Automatic extraction and embedding of voxel sizes, pixel spacing, and channel information
- **Multi-series Support**: Handle files with multiple acquisitions, Z-stacks, time-lapse sequences in a single file
- **Intelligent Format Selection**: Automatic recommendation based on file size

## Quick Start

1. Open napari and navigate to **Plugins → napari-tmidas → Microscopy Image Conversion**
2. Click **Browse** to select your input folder containing microscopy files
3. Click **Scan Folder** to discover all compatible files
4. Click on files/series to preview them in the napari viewer
5. Choose output format: **TIF** (small files) or **ZARR** (large files, >4GB)
6. Select output folder
7. Click **Convert Selected Files** or **Convert All Files**

## Supported Input Formats

| Format | Extension | Key Features |
|--------|-----------|--------------|
| **Leica LIF** | `.lif` | Multi-image databases with Z-stacks, time-lapse |
| **Nikon ND2** | `.nd2` | Multi-position acquisitions, time-series data |
| **Zeiss CZI** | `.czi` | Complex metadata, multiple channels, Z-stacks |
| **Whole Slide (NDPI/SVS)** | `.ndpi`, `.svs` | High-resolution pyramid/multi-resolution images |
| **Acquifer** | Directory-based | Large-scale screening datasets |

## Output Formats

### TIFF Format

**Best for:**
- Small to medium datasets (< 4GB)
- Single series from each file
- When maximum compatibility is needed

**Characteristics:**
- Standard 16-bit or 8-bit TIFF
- All metadata embedded in TIFF tags
- Compatible with most image analysis software
- Single file output per series

**File Structure:**
```
output_folder/
├── sample1_series_0.tif
├── sample2_series_0.tif
└── sample3_series_0.tif
```

### OME-Zarr Format

**Best for:**
- Large datasets (> 4GB)
- Multi-resolution requirements
- Cloud-native workflows
- Efficient parallel processing

**Characteristics:**
- [OME-Zarr specification v0.5](https://ngff.openmicroscopy.org/latest/) compliant
- Hierarchical chunk-based storage
- Automatic metadata in `.ome.xml`
- Efficient for large arrays
- Built-in multi-scale pyramids support

**File Structure:**
```
output_folder/
├── sample1_series_0.zarr/
│   ├── 0/
│   ├── .ome.xml
│   └── .zarray
├── sample2_series_0.zarr/
│   ├── 0/
│   ├── .ome.xml
│   └── .zarray
└── sample3_series_0.zarr/
```

## User Interface

### 1. File Scanning and Selection

**Input Folder**: Select directory containing microscopy files
**File Filter**: Pre-configured for `.lif, .nd2, .ndpi, .czi, acquifer`
- Click **Scan Folder** to discover all compatible files
- Files are indexed for quick preview

**Files Table** shows:
- Filename
- Number of series/acquisitions in the file
- Click to select and preview in viewer

### 2. Series Selection

Once you select a file with multiple series:

**Series Selector** dropdown:
- Choose specific series to export
- Or check **Export All Series** to convert all acquisitions at once

**Preview Button**: Load selected series into napari viewer for inspection

**Series Information**:
- Estimated file size (GB)
- Dimension information
- Format recommendation based on size

### 3. Format Selection

**TIF Checkbox**: Standard TIFF output (default)
- Automatically unchecked for files > 4GB
- Recommended for immediate compatibility

**ZARR Checkbox** (Recommended for >4GB):
- Click to enable OME-Zarr output
- Recommended for large datasets
- More efficient storage and access

**Auto-recommendation**:
- Files > 4GB automatically recommend ZARR format
- You can override if needed

### 4. Output Configuration

**Output Folder**: Select destination directory
**Convert Buttons**:
- **Convert Selected Files**: Export only checked files/series
- **Convert All Files**: Export all discovered files

## Metadata Preservation

The conversion process automatically extracts and preserves:

### Spatial Metadata
- **Pixel/Voxel Resolution**: X-Y pixel spacing (in micrometers)
- **Z Spacing**: Slice-to-slice distance for 3D data
- **Physical Units**: Recording and embedding of measurement units

### Dimensional Metadata
- **Channels**: Number of color/fluorescence channels
- **Z-stacks**: Number of optical sections
- **Time Points**: Number of frames in time-lapse sequences
- **Positions**: Multiple scanning positions in multi-acquisition files

### Format-Specific Metadata
- **LIF**: Leica objective specs, microscope configuration
- **ND2**: Nikon camera settings, acquisition parameters
- **CZI**: Zeiss imaging setup, filter configurations
- **ND2/CZI**: Physical units and scaling information

### Embedded In Output
- **TIFF**: Metadata in ImageDescription and other TIFF tags
- **OME-Zarr**: Complete metadata in embedded `.ome.xml` file (OME-NGFF compliant)

## Advanced Features

### Multi-Series Export

Files can contain multiple acquisitions (series):

```
Example: Nikon ND2 with 4 positions
- Series 0: Position 1
- Series 1: Position 2
- Series 2: Position 3
- Series 3: Position 4
```

**Export Options**:
- Select specific series individually
- Export all series at once with "Export All Series" checkbox

### Memory Management

**Adaptive Loading Strategy**:
- **Small datasets (< 1GB)**: Loaded entirely into memory (fast)
- **Medium datasets (1-4GB)**: Chunked numpy loading (balanced)
- **Large datasets (> 4GB)**: Lazy Dask loading (memory-efficient)

Automatically detects file size and chooses optimal strategy.

### Real-Time Preview

- Click on any file/series in the table
- Automatically loads in napari viewer
- Inspect dimensions, channels, and data quality before conversion
- Quick validation before batch processing

### Format Recommendations

| File Size | Recommended Format | Reason |
|-----------|-------------------|--------|
| < 1GB | TIF | Fast access, universal compatibility |
| 1-4GB | TIF or ZARR | Either works; ZARR if multiple users |
| > 4GB | ZARR (required) | TIF becomes unwieldy; ZARR is efficient |
| > 100GB | ZARR only | TIFF not recommended |

## Workflow Examples

### Single File with Multiple Series

**Scenario**: Nikon ND2 with 3 scanning positions

1. Select input folder containing the ND2 file
2. Click Scan Folder → File appears with "3 series"
3. Click on filename to show series selection
4. Option A: Check "Export All Series" → Converts all 3 positions
5. Option B: Select Series 0, 1, 2 individually → Convert separately
6. Choose output format (TIF for small, ZARR for large)
7. Click Convert

### Large Time-Lapse Dataset

**Scenario**: Zeiss CZI with 1000 time points, 3 channels, 50 Z-stacks (20GB)

1. Scan folder → CZI detected, estimated size shown as "20GB"
2. Format selector automatically recommends ZARR
3. Preview first frame to verify data
4. Click "Convert Selected Files" → Outputs as `filename.zarr`
5. Output is chunked for efficient parallel processing

### Batch Conversion of Multiple Formats

**Scenario**: Folder with mixed LIF, ND2, CZI files

1. Scan folder with default filters (`.lif,.nd2,.ndpi,.czi,acquifer`)
2. All files indexed simultaneously
3. Click each file to preview
4. Select output format once (applies to all)
5. Click "Convert All Files" → Processes in optimal order

## Troubleshooting

### "File Format Error" When Scanning

**Causes**:
- File is corrupted or truncated
- File extension doesn't match actual format
- Missing required libraries

**Solutions**:
- Verify file integrity on microscope/computer it was created on
- Check file header with external tools (e.g., `file filename.lif`)
- Reinstall napari-tmidas: `pip install --force-reinstall napari-tmidas`

### Memory Error During Conversion

**Causes**:
- File too large for available RAM
- Other applications consuming memory
- Insufficient swap space

**Solutions**:
- Convert large files individually rather than in batch
- Close other applications
- Use OME-Zarr format (memory-efficient)
- Increase system swap space if available

### Slow Conversion Speed

**Causes**:
- Large file with many series
- Slow storage device
- Data needs resizing/reordering

**Solutions**:
- Use fast SSD for output folder
- Convert files individually rather than batch
- OME-Zarr typically faster than TIFF for large files
- Check that output folder is on fast local storage, not network drive

### Series Count Shows 0

**Causes**:
- File format not recognized
- Corrupted file header
- Library version mismatch

**Solutions**:
- Verify file type is correct
- Try opening in original microscope software
- Update libraries: `pip install --upgrade readlif nd2 pylibCZIrw`

### Metadata Not Appearing in Output

**For TIFF**:
- Some viewers don't display TIFF metadata
- Use ImageJ/Fiji (Edit → Image Properties) to verify
- Metadata is present in TIFF tags even if not visible

**For OME-Zarr**:
- Metadata is in `.ome.xml` file in the `.zarr` directory
- View with: `cat output.zarr/.ome.xml`

## File Format Details

### Leica LIF

- **Structure**: Database-like container with multiple images
- **Metadata**: Excellent (objective, pixel size, Z-spacing)
- **Series**: Each image in LIF is one series
- **Special handling**: Preserves Leica microscope configuration

### Nikon ND2

- **Structure**: Multi-position acquisitions common
- **Metadata**: Complete (voxel size, camera info, settings)
- **Series**: "P" dimension represents positions
- **Special handling**: Separates positions into independent series

### Zeiss CZI

- **Structure**: Highly flexible, supports complex metadata
- **Metadata**: Very detailed (channels, objectives, filters)
- **Series**: Multiple mosaic tiles or positions
- **Special handling**: Handles RGB and multispectral data

### Whole Slide (NDPI/SVS)

- **Structure**: Pyramid format (multi-resolution)
- **Metadata**: Limited (mostly dimension info)
- **Series**: Different pyramid levels become series
- **Special handling**: High-resolution tile-based format

### Acquifer

- **Structure**: Directory-based with metadata files
- **Metadata**: Good (well-organized parameter files)
- **Series**: Each well/plate position is one series
- **Special handling**: Automatic plate layout detection

## Performance Tips

1. **Use OME-Zarr for large files**: 2-3x faster access than TIFF for > 4GB
2. **Output to local fast SSD**: Network drives significantly slower
3. **Convert batches of small files**: Faster than single large file
4. **Close other applications**: Frees memory for conversion
5. **Verify metadata first**: Preview before batch converting hundreds of files

## Related Features

- **[Batch Crop Anything](crop_anything.md)** - Extract regions from converted images
- **[Image Processing](basic_processing.md)** - Process converted images
- **[Cellpose Segmentation](cellpose_segmentation.md)** - Segment converted datasets

## Technical Specifications

### Data Types Supported
- **8-bit**: Unsigned integer (0-255)
- **16-bit**: Unsigned integer (0-65535)
- **32-bit**: Float (0-1 or wider range)

### Maximum File Size
- **TIFF**: Technically up to 4GB per file (practical limit)
- **OME-Zarr**: No practical limit (scales to terabytes)

### Chunk Sizes
- **OME-Zarr default**: 64³ or (64, Y, X) for optimal access patterns
- **Adjustable**: Via metadata configuration if needed

## References

- [OME-Zarr Specification](https://ngff.openmicroscopy.org/latest/)
- [Bioimage Format Standards](https://biop.github.io/ijp-opencv/)
- [napari Documentation](https://napari.org/)

## Citation

If you use the File Conversion feature in your research, please cite:

```bibtex
@software{napari_tmidas_2024,
  title = {napari-tmidas: Batch Image Processing for Microscopy},
  author = {Mercader Lab},
  year = {2024},
  url = {https://github.com/MercaderLabAnatomy/napari-tmidas}
}
```

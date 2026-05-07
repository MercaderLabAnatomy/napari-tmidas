
# Multichannel Processing with Channel Selection

## Overview

The napari-tmidas batch processing system now supports automatic detection of multichannel images (especially from zarr files) and allows users to select which channel(s) to process.

## Features

### 1. Automatic Channel Detection
When loading multichannel zarr files or other image formats, the system automatically detects:
- Number of channels in the image
- The axis where channels are located (C axis)
- Common patterns: CYX, CZYX, TCYX, TCZYX

### 2. Channel Selection UI
Processing functions can add a `channel` parameter with `widget_type: "channel_selector"` to enable channel selection in the UI:

```python
@BatchProcessingRegistry.register(
    name="My Processing Function",
    suffix="_processed",
    description="Process images with channel selection",
    parameters={
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to process",
        },
        # ... other parameters
    },
)
def my_function(image: np.ndarray, channel: str = "all") -> np.ndarray:
    # The processing worker handles channel extraction automatically
    # You receive the selected channel as input
    return processed_image
```

### 3. Channel Selection Options
Users can select:
- **All channels**: Process all channels separately, creating one output file per channel
- **Channel 0, 1, 2, ...**: Process only a specific channel

## How It Works

### Detection Phase
1. When files are selected, the first file is loaded
2. `detect_channels_in_image()` analyzes the image shape
3. Channels are detected based on common patterns (small dimension: 2-10 values)

### Processing Phase
1. User selects which channel(s) to process via the dropdown
2. `ProcessingWorker` extracts the selected channel(s) before processing
3. The processing function receives the extracted channel data
4. Results are saved with appropriate channel suffixes

### Output Files
- **Single channel selected**: `filename_suffix.tif`
- **All channels selected**: `filename_ch0_suffix.tif`, `filename_ch1_suffix.tif`, etc.

## Example Usage

### Adding Channel Selection to Your Function

```python
from napari_tmidas._registry import BatchProcessingRegistry

@BatchProcessingRegistry.register(
    name="Gaussian Blur with Channel Selection",
    suffix="_blurred",
    description="Apply Gaussian blur to selected channel(s)",
    parameters={
        "sigma": {
            "type": float,
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Blur strength",
        },
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select channel to process",
        },
    },
)
def gaussian_blur_multichannel(image: np.ndarray, sigma: float = 1.0, channel: str = "all") -> np.ndarray:
    from scipy import ndimage
    # Note: channel parameter is handled by the processing worker
    # You receive the already-extracted channel data
    return ndimage.gaussian_filter(image, sigma=sigma)
```

## Implementation Details

### Key Components

1. **`detect_channels_in_image()`** ([_file_selector.py](../src/napari_tmidas/_file_selector.py))
   - Detects channels in numpy arrays or OME-Zarr layer data
   - Returns (num_channels, channel_axis)
   - Handles common patterns: CYX, CZYX, TCYX, TCZYX

2. **`ParameterWidget.update_channel_selector()`** ([_file_selector.py](../src/napari_tmidas/_file_selector.py))
   - Populates the channel selector dropdown
   - Called when processing function is selected
   - Analyzes first file in the batch

3. **`ProcessingWorker.process_file()`** ([_processing_worker.py](../src/napari_tmidas/_processing_worker.py))
   - Extracts selected channel(s) before processing
   - Handles "all" channels by processing each separately
   - Manages output file naming with channel suffixes

### Supported Image Formats
- **Zarr files** (`.zarr`): Full OME-Zarr support with metadata
- **TIFF files** (`.tif`, `.tiff`): Multichannel TIFF stacks
- **Other formats**: Any format loadable by napari

### Channel Detection Patterns
The system detects channels when:
- First dimension is 2-10 in size (CYX, CZYX patterns)
- Second dimension is 2-10 in size (TCYX, TCZYX patterns)
- Dimension is clearly smaller than spatial dimensions (Y, X)
- For time-series layouts like TCYX/TCZYX, axis 1 is preferred as channel axis (axis 0 is treated as time)

### Recent Behavior Update
- TCYX and TCZYX channel detection now consistently treats axis 1 as channels when axis 0 represents time
- This avoids incorrect interpretation of timepoints as channels in multichannel time-series data

## Benefits

1. **Flexibility**: Process all channels or just one
2. **Efficiency**: Avoid processing unwanted channels
3. **Clarity**: Clear output files with channel information
4. **Automation**: No manual channel splitting needed

## Example Functions Using Channel Selection

- **Gaussian Blur**: Blur specific channels in multichannel images
- _(Add your functions here)_

## Future Enhancements

Potential improvements:
- Support for selecting multiple specific channels (e.g., channels 0 and 2)
- Channel merging after processing
- Preview of channel selection before batch processing
- Custom channel naming/labeling


# processing_functions/basic.py
"""
Basic image processing functions that don't require additional dependencies.
"""
import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry


@BatchProcessingRegistry.register(
    name="Gamma Correction",
    suffix="_gamma",
    description="Apply gamma correction to the image (>1: enhance bright regions, <1: enhance dark regions)",
    parameters={
        "gamma": {
            "type": float,
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Gamma correction factor",
        },
    },
)
def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to the image
    """
    # Determine maximum value based on dtype
    max_val = (
        np.iinfo(image.dtype).max
        if np.issubdtype(image.dtype, np.integer)
        else 1.0
    )

    # Normalize image to [0, 1]
    normalized = image.astype(np.float32) / max_val

    # Apply gamma correction
    corrected = np.power(normalized, gamma)

    # Scale back to original range and dtype
    return (corrected * max_val).clip(0, max_val).astype(image.dtype)


@BatchProcessingRegistry.register(
    name="Max Z Projection",
    suffix="_max_z",
    description="Maximum intensity projection along the z-axis",
    parameters={},
)
def max_z_projection(image: np.ndarray) -> np.ndarray:
    """
    Maximum intensity projection along the z-axis
    """
    # Determine maximum value based on dtype
    max_val = (
        np.iinfo(image.dtype).max
        if np.issubdtype(image.dtype, np.integer)
        else 1.0
    )

    # Normalize image to [0, 1]
    normalized = image.astype(np.float32) / max_val

    # Apply max z projection
    projection = np.max(normalized, axis=0)

    # Scale back to original range and dtype
    return (projection * max_val).clip(0, max_val).astype(image.dtype)


@BatchProcessingRegistry.register(
    name="Split Channels",
    suffix="_split_channels",
    description="Splits the color channels of the image",
    parameters={
        "num_channels": {
            "type": "integer",
            "default": 3,
            "description": "Number of color channels in the image",
        }
    },
)
def split_channels(image: np.ndarray, num_channels: int = 3) -> np.ndarray:
    """
    Split the image into separate channels based on the specified number of channels.

    Args:
        image: Input image array (at least 3D: XYC or higher dimensions)
        num_channels: Number of channels in the image (default: 3)

    Returns:
        Stacked array of channels with shape (num_channels, ...)
    """
    # Validate input
    if image.ndim < 3:
        raise ValueError(
            "Input must be an array with at least 3 dimensions (XYC or higher)"
        )

    print(f"Image shape: {image.shape}")
    num_channels = int(num_channels)
    # Identify the channel axis
    possible_axes = [
        axis
        for axis, dim_size in enumerate(image.shape)
        if dim_size == num_channels
    ]
    # print(f"Possible axes: {possible_axes}")
    if len(possible_axes) != 1:

        raise ValueError(
            f"Could not uniquely identify a channel axis with {num_channels} channels. "
            f"Found {len(possible_axes)} possible axes: {possible_axes}. "
            f"Image shape: {image.shape}"
        )

    channel_axis = possible_axes[0]
    print(f"Channel axis identified: {channel_axis}")

    # Split and process channels
    channels = np.split(image, num_channels, axis=channel_axis)
    # channels = [np.squeeze(ch, axis=channel_axis) for ch in channels]

    return np.stack(channels, axis=0)

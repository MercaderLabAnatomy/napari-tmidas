# processing_functions/basic.py
"""
Basic image processing functions that don't require additional dependencies.
"""
import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry


@BatchProcessingRegistry.register(
    name="Labels to Binary",
    suffix="_binary",
    description="Convert multi-label images to binary masks (all non-zero labels become 1)",
)
def labels_to_binary(image: np.ndarray) -> np.ndarray:
    """
    Convert multi-label images to binary masks.

    This function takes a label image (where different regions have different label values)
    and converts it to a binary mask (where all labeled regions have a value of 1 and
    background has a value of 0).

    Parameters:
    -----------
    image : numpy.ndarray
        Input label image array

    Returns:
    --------
    numpy.ndarray
        Binary mask with 1 for all non-zero labels and 0 for background
    """
    # Make a copy of the input image to avoid modifying the original
    binary_mask = image.copy()

    binary_mask = (binary_mask > 0).astype(np.uint32)

    return binary_mask


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
    name="Max Z Projection (TZYX)",
    suffix="_maxZ_tzyx",
    description="Maximum intensity projection along the Z-axis for TZYX data",
    parameters={},  # No parameters needed - fully automatic
)
def max_z_projection_tzyx(image: np.ndarray) -> np.ndarray:
    """
    Memory-efficient maximum intensity projection along the Z-axis for TZYX data.

    This function intelligently chooses the most memory-efficient approach
    based on the input data size and available system memory.

    Parameters:
    -----------
    image : numpy.ndarray
        Input 4D image with TZYX dimensions

    Returns:
    --------
    numpy.ndarray
        3D image with TYX dimensions after max projection
    """
    # Validate input dimensions
    if image.ndim != 4:
        raise ValueError(f"Expected 4D image (TZYX), got {image.ndim}D image")

    # Get dimensions
    t_size, z_size, y_size, x_size = image.shape

    # For Z projection, we only need one Z plane in memory at a time
    # so we can process this plane by plane to minimize memory usage

    # Create output array with appropriate dimensions and same dtype
    result = np.zeros((t_size, y_size, x_size), dtype=image.dtype)

    # Process each time point separately to minimize memory usage
    for t in range(t_size):
        # If data type allows direct max, use it
        if np.issubdtype(image.dtype, np.integer) or np.issubdtype(
            image.dtype, np.floating
        ):
            # Process Z planes efficiently
            # Start with the first Z plane
            z_max = image[t, 0].copy()

            # Compare with each subsequent Z plane
            for z in range(1, z_size):
                # Use numpy's maximum function to update max values in-place
                np.maximum(z_max, image[t, z], out=z_max)

            # Store result for this time point
            result[t] = z_max
        else:
            # For unusual data types, fall back to numpy's max function
            result[t] = np.max(image[t], axis=0)

    return result


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


@BatchProcessingRegistry.register(
    name="RGB to Labels",
    suffix="_labels",
    description="Convert RGB images to label images using a color map",
    parameters={
        "blue_label": {
            "type": int,
            "default": 1,
            "min": 0,
            "max": 255,
            "description": "Label value for blue objects",
        },
        "green_label": {
            "type": int,
            "default": 2,
            "min": 0,
            "max": 255,
            "description": "Label value for green objects",
        },
        "red_label": {
            "type": int,
            "default": 3,
            "min": 0,
            "max": 255,
            "description": "Label value for red objects",
        },
    },
)
def rgb_to_labels(
    image: np.ndarray,
    blue_label: int = 1,
    green_label: int = 2,
    red_label: int = 3,
) -> np.ndarray:
    """
    Convert RGB images to label images where each color is mapped to a specific label value.

    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image array
    blue_label : int
        Label value for blue objects (default: 1)
    green_label : int
        Label value for green objects (default: 2)
    red_label : int
        Label value for red objects (default: 3)

    Returns:
    --------
    numpy.ndarray
        Label image where each color is mapped to the specified label value
    """
    # Ensure the image is a proper RGB image
    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError("Input must be an RGB image with 3 channels")

    # Define the color mapping
    color_mapping = {
        (0, 0, 255): blue_label,  # Blue
        (0, 255, 0): green_label,  # Green
        (255, 0, 0): red_label,  # Red
    }
    # Create an empty label image
    label_image = np.zeros(image.shape[:-1], dtype=np.uint32)
    # Iterate through the color mapping and assign labels
    for color, label in color_mapping.items():
        mask = np.all(image == color, axis=-1)
        label_image[mask] = label
    # Return the label image
    return label_image

# processing_functions/cellpose_segmentation.py
"""
Processing functions for automatic instance segmentation using Cellpose.

This module provides functionality to automatically segment cells or nuclei in images
using the Cellpose deep learning-based segmentation toolkit. It supports both 2D and 3D images,
various dimension orders, and handles time series data.

Updated to support Cellpose 4 (Cellpose-SAM) which offers improved generalization
for cellular segmentation without requiring diameter parameter.

Note: This requires the cellpose library to be installed.
"""

import numpy as np

# Import the environment manager
from napari_tmidas.processing_functions.cellpose_env_manager import (
    create_cellpose_env,
    is_env_created,
    run_cellpose_in_env,
)

# Check if cellpose is directly available in this environment
try:
    from cellpose import core, models

    CELLPOSE_AVAILABLE = True
    USE_GPU = core.use_gpu()
    USE_DEDICATED_ENV = False
    print("Cellpose found in current environment. Using native import.")
except ImportError:
    CELLPOSE_AVAILABLE = False
    USE_GPU = False
    USE_DEDICATED_ENV = True
    print(
        "Cellpose not found in current environment. Will use dedicated environment."
    )

from napari_tmidas._registry import BatchProcessingRegistry


def transpose_dimensions(img, dim_order):
    """
    Transpose image dimensions to match expected Cellpose input.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    dim_order : str
        Dimension order of the input image (e.g., 'ZYX', 'TZYX', 'YXC')

    Returns:
    --------
    numpy.ndarray
        Transposed image
    str
        New dimension order
    bool
        Whether the image is 3D
    """
    # Handle time dimension if present
    has_time = "T" in dim_order

    # Determine if the image is 3D (has Z dimension)
    is_3d = "Z" in dim_order

    # Standardize dimension order
    if has_time:
        # For time series, we want to end up with TZYX
        target_dims = "TZYX"
        transpose_order = [
            dim_order.index(d) for d in target_dims if d in dim_order
        ]
        new_dim_order = "".join([dim_order[i] for i in transpose_order])
    else:
        # For single timepoints, we want ZYX
        target_dims = "ZYX"
        transpose_order = [
            dim_order.index(d) for d in target_dims if d in dim_order
        ]
        new_dim_order = "".join([dim_order[i] for i in transpose_order])

    # Perform the transpose
    img_transposed = np.transpose(img, transpose_order)

    return img_transposed, new_dim_order, is_3d


def run_cellpose(
    img,
    model,
    channels,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    dim_order="ZYX",
    max_pixels=4000000,
    tile_norm_blocksize=128,
    batch_size=32,
):
    """
    Run Cellpose segmentation on an image using Cellpose 4 (Cellpose-SAM).

    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    model : cellpose.models.Cellpose
        Cellpose model to use
    channels : list
        Channels to use for segmentation [0,0] = grayscale, [1,0] = green channel, [2,0] = red channel
    flow_threshold : float
        Flow threshold for Cellpose
    cellprob_threshold : float
        Cell probability threshold
    dim_order : str
        Dimension order of the input image
    max_pixels : int
        Maximum number of pixels to process (for 2D images)
    tile_norm_blocksize : int
        Block size for tile normalization (new parameter in Cellpose 4)
    batch_size : int
        Batch size for processing multiple images or 3D slices at once

    Returns:
    --------
    numpy.ndarray
        Segmented image with labels
    """
    # First check if the image is too large (for 2D images)
    if len(img.shape) == 2 or (len(img.shape) == 3 and "C" in dim_order):
        # For 2D images (potentially with channels)
        height, width = img.shape[:2]
        total_pixels = height * width
        if total_pixels > max_pixels:
            raise ValueError(
                f"Image size ({height}x{width}={total_pixels} pixels) exceeds the "
                f"maximum size of {max_pixels} pixels. Consider downsampling."
            )

    # Transpose to expected dimension order
    img_transposed, new_dim_order, is_3d = transpose_dimensions(img, dim_order)

    # Check if we have a time series
    has_time = "T" in new_dim_order

    # Set up normalization with tile_norm_blocksize (Cellpose 4 parameter)
    normalize = {"tile_norm_blocksize": tile_norm_blocksize}

    if has_time:
        # Handle time series - process each time point
        n_timepoints = img_transposed.shape[0]
        result = np.zeros(img_transposed.shape, dtype=np.uint32)

        # Process each time point
        for t in range(n_timepoints):
            img_t = img_transposed[t]
            masks, _, _ = model.eval(
                img_t,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize=normalize,
                z_axis=0 if is_3d else None,
                do_3D=is_3d,
                batch_size=batch_size,
            )
            result[t] = masks
    else:
        # Process single time point
        masks, _, _ = model.eval(
            img_transposed,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize=normalize,
            z_axis=0 if is_3d else None,
            do_3D=is_3d,
            batch_size=batch_size,
        )
        result = masks

    return result.astype(np.uint32)


@BatchProcessingRegistry.register(
    name="Cellpose-SAM Segmentation",
    suffix="_labels",
    description="Automatic instance segmentation using Cellpose 4 (Cellpose-SAM) with improved generalization.",
    parameters={
        "dim_order": {
            "type": str,
            "default": "YX",
            "description": "Dimension order of the input (e.g., 'YX', 'ZYX', 'TZYX')",
        },
        # "channel_1": {
        #     "type": int,
        #     "default": 0,
        #     "min": 0,
        #     "max": 3,
        #     "description": "First channel: 0=grayscale, 1=green, 2=red, 3=blue",
        # },
        # "channel_2": {
        #     "type": int,
        #     "default": 0,
        #     "min": 0,
        #     "max": 3,
        #     "description": "Second channel: 0=none, 1=green, 2=red, 3=blue",
        # },
        "flow_threshold": {
            "type": float,
            "default": 0.4,
            "min": 0.1,
            "max": 0.9,
            "description": "Flow threshold for Cellpose segmentation",
        },
        "cellprob_threshold": {
            "type": float,
            "default": 0.0,
            "min": -6.0,
            "max": 6.0,
            "description": "Cell probability threshold (Cellpose 4 parameter)",
        },
        "tile_norm_blocksize": {
            "type": int,
            "default": 128,
            "min": 32,
            "max": 512,
            "description": "Block size for tile normalization (Cellpose 4 parameter)",
        },
        "batch_size": {
            "type": int,
            "default": 32,
            "min": 1,
            "max": 128,
            "description": "Batch size for processing multiple images/slices at once",
        },
        "force_dedicated_env": {
            "type": bool,
            "default": False,
            "description": "Force using dedicated environment even if Cellpose is available",
        },
    },
)
def cellpose_segmentation(
    image: np.ndarray,
    dim_order: str = "YX",
    channel_1: int = 0,
    channel_2: int = 0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    tile_norm_blocksize: int = 128,
    batch_size: int = 32,
    force_dedicated_env: bool = False,
) -> np.ndarray:
    """
    Run Cellpose 4 (Cellpose-SAM) segmentation on an image.

    This function takes an image and performs automatic instance segmentation using
    Cellpose 4 with improved generalization for cellular segmentation. It supports
    both 2D and 3D images, various dimension orders, and handles time series data.

    If Cellpose is not available in the current environment, a dedicated virtual
    environment will be created to run Cellpose.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    dim_order : str
        Dimension order of the input (e.g., 'YX', 'ZYX', 'TZYX') (default: "YX")
    channel_1 : int
        First channel: 0=grayscale, 1=green, 2=red, 3=blue (default: 0)
    channel_2 : int
        Second channel: 0=none, 1=green, 2=red, 3=blue (default: 0)
    flow_threshold : float
        Flow threshold for Cellpose segmentation (default: 0.4)
    cellprob_threshold : float
        Cell probability threshold (Cellpose 4 parameter) (default: 0.0)
    tile_norm_blocksize : int
        Block size for tile normalization (Cellpose 4 parameter) (default: 128)
    batch_size : int
        Batch size for processing multiple images/slices at once (default: 32)
    force_dedicated_env : bool
        Force using dedicated environment even if Cellpose is available (default: False)

    Returns:
    --------
    numpy.ndarray
        Segmented image with instance labels
    """
    # Convert channel parameters to Cellpose channels list
    # channels = [channel_1, channel_2]
    channels = [0, 0]  # limit script to single channel
    # Determine whether to use dedicated environment
    use_env = force_dedicated_env or USE_DEDICATED_ENV

    if use_env:
        print("Using dedicated Cellpose environment...")

        # First check if the environment exists, create if not
        if not is_env_created():
            print(
                "Creating dedicated Cellpose environment (this may take a few minutes)..."
            )
            create_cellpose_env()
            print("Environment created successfully.")

        # Prepare arguments for the Cellpose function
        args = {
            "image": image,
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
            "batch_size": batch_size,
            "use_gpu": USE_GPU,
            "do_3D": "Z" in dim_order,
            "z_axis": 0 if "Z" in dim_order else None,
        }

        # Run Cellpose in the dedicated environment
        print("Running Cellpose model in dedicated environment...")
        result = run_cellpose_in_env("eval", args)
        print(f"Segmentation complete. Found {np.max(result)} objects.")
        return result

    else:
        print("Running Cellpose model in current environment...")
        # Initialize Cellpose model in current environment
        model = models.CellposeModel(gpu=USE_GPU)

    # Print status information
    gpu_status = "GPU" if USE_GPU else "CPU"
    print(f"Using Cellpose on {gpu_status}")
    print(
        f"Processing image with shape {image.shape}, dimension order: {dim_order}"
    )

    # Run segmentation
    try:
        result = run_cellpose(
            image,
            model,
            channels,
            flow_threshold,
            cellprob_threshold,
            dim_order,
            tile_norm_blocksize=tile_norm_blocksize,
            batch_size=batch_size,
        )

        print(f"Segmentation complete. Found {np.max(result)} objects.")
        return result

    except (Exception, MemoryError) as e:
        print(f"Error during segmentation in current environment: {str(e)}")

        # If we haven't already tried using the dedicated environment, try that as a fallback
        if not USE_DEDICATED_ENV and not force_dedicated_env:
            print("Attempting fallback to dedicated Cellpose environment...")
            try:
                args = {
                    "image": image,
                    "channels": channels,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
                    "batch_size": batch_size,
                    "use_gpu": USE_GPU,
                    "do_3D": "Z" in dim_order,
                    "z_axis": 0 if "Z" in dim_order else None,
                }

                if not is_env_created():
                    create_cellpose_env()

                result = run_cellpose_in_env("eval", args)
                print(f"Fallback successful. Found {np.max(result)} objects.")
                return result
            except (Exception, MemoryError) as fallback_e:
                print(f"Fallback also failed: {str(fallback_e)}")
                raise Exception(
                    f"Both direct and dedicated environment approaches failed: {str(e)} | {str(fallback_e)}"
                ) from fallback_e
        else:
            raise


# Update cellpose_env_manager.py to install Cellpose 4
def update_cellpose_env_manager():
    """
    Update the cellpose_env_manager to install Cellpose 4
    """
    # This function can be called to update the environment manager code
    # For example, by modifying the pip install command to install the latest version
    # or specify Cellpose 4 explicitly

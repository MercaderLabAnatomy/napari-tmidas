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
from typing import Union

import numpy as np

# Import the environment manager
from napari_tmidas.processing_functions.cellpose_env_manager import (
    create_cellpose_env,
    is_env_created,
    run_cellpose_in_env,
    cancel_cellpose_processing,  # Add cancellation import
)

# Check if cellpose is directly available in this environment
try:
    from cellpose import core

    CELLPOSE_AVAILABLE = True
    # Don't evaluate USE_GPU here - it should be evaluated in the cellpose environment
    # USE_GPU = core.use_gpu()  # This was the bug!

    print("Cellpose found in current environment. Using native import.")
except ImportError:
    CELLPOSE_AVAILABLE = False
    # USE_GPU = False  # Don't set here either

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
            "description": "Flow threshold (cell detection sensitivity)",
        },
        "cellprob_threshold": {
            "type": float,
            "default": 0.0,
            "min": -6.0,
            "max": 6.0,
            "description": "Cell probability threshold (increase to reduce splits)",
        },
        "anisotropy": {
            "type": float,
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Optional rescaling factor (3D; Z step[um] / X pixel res [um])",
        },
        "flow3D_smooth": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 10,
            "description": "Smooth flow with gaussian filter (stdev)",
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
    },
)
def cellpose_segmentation(
    image: np.ndarray,
    dim_order: str = "YX",
    channel_1: int = 0,
    channel_2: int = 0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    anisotropy: Union[float, None] = None,
    flow3D_smooth: int = 0,
    tile_norm_blocksize: int = 128,
    batch_size: int = 32,
    _source_filepath: str = None,  # Hidden parameter for zarr optimization
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

    Returns:
    --------
    numpy.ndarray
        Segmented image with instance labels
    """
    # Convert channel parameters to Cellpose channels list
    # channels = [channel_1, channel_2]
    channels = [0, 0]  # limit script to single channel

    # First check if the environment exists, create if not
    if not is_env_created():
        print(
            "Creating dedicated Cellpose environment (this may take a few minutes)..."
        )
        create_cellpose_env()
        print("Environment created successfully.")
        
    # Check if we can use zarr optimization
    use_zarr_direct = _source_filepath and _source_filepath.lower().endswith('.zarr')
    
    if use_zarr_direct:
        print(f"Using optimized zarr processing for: {_source_filepath}")
        # Prepare arguments for direct zarr processing
        args = {
            "zarr_path": _source_filepath,
            "zarr_key": None,  # Could be enhanced to support specific keys
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "flow3D_smooth": flow3D_smooth,
            "anisotropy": anisotropy,
            "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
            "batch_size": batch_size,
            "use_gpu": True,  # Let cellpose environment detect GPU
            "do_3D": "Z" in dim_order,
            "z_axis": 0 if "Z" in dim_order else None,
        }
    else:
        # Prepare arguments for the Cellpose function (legacy numpy array mode)
        args = {
            "image": image,
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "flow3D_smooth": flow3D_smooth,
            "anisotropy": anisotropy,
            "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
            "batch_size": batch_size,
            "use_gpu": True,  # Let cellpose environment detect GPU
            "do_3D": "Z" in dim_order,
            "z_axis": 0 if "Z" in dim_order else None,
        }
    # Run Cellpose in the dedicated environment
    print("Running Cellpose model in dedicated environment...")
    result = run_cellpose_in_env("eval", args)
    print(f"Segmentation complete. Found {np.max(result)} objects.")

    return result


# Update cellpose_env_manager.py to install Cellpose 4
def update_cellpose_env_manager():
    """
    Update the cellpose_env_manager to install Cellpose 4
    """
    # This function can be called to update the environment manager code
    # For example, by modifying the pip install command to install the latest version
    # or specify Cellpose 4 explicitly

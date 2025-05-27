# processing_functions/careamics_denoising.py
"""
Processing functions for denoising images using CAREamics.

This module provides functionality to denoise images using various models from CAREamics,
including Noise2Void (N2V) and CARE models. The functions support both 2D and 3D data.

The functions will automatically create and manage a dedicated environment for CAREamics
if it's not already installed in the main environment.
"""
import os

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

# Import the environment manager for CAREamics
from napari_tmidas.processing_functions.careamics_env_manager import (
    create_careamics_env,
    is_env_created,
    run_careamics_in_env,
)

# Check if CAREamics is directly available in current environment
try:
    from careamics import CAREamist

    CAREAMICS_AVAILABLE = True
    USE_DEDICATED_ENV = False
    print("CAREamics found in current environment, using direct import")
except ImportError:
    CAREAMICS_AVAILABLE = False
    USE_DEDICATED_ENV = True
    print(
        "CAREamics not found in current environment, will use dedicated environment"
    )


@BatchProcessingRegistry.register(
    name="CAREamics Denoise (N2V/CARE)",
    suffix="_denoised",
    description="Denoise images using CAREamics (Noise2Void or CARE model)",
    parameters={
        "checkpoint_path": {
            "type": str,
            "default": "",
            "description": "Path to the CAREamics model checkpoint file (.ckpt)",
        },
        "tile_size_x": {
            "type": int,
            "default": 32,
            "min": 16,
            "max": 512,
            "description": "Tile size in X dimension",
        },
        "tile_size_y": {
            "type": int,
            "default": 32,
            "min": 16,
            "max": 512,
            "description": "Tile size in Y dimension",
        },
        "tile_size_z": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 256,
            "description": "Tile size in Z dimension (for 3D data)",
        },
        "batch_size": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 16,
            "description": "Batch size for prediction",
        },
        "use_tta": {
            "type": bool,
            "default": True,
            "description": "Use test-time augmentation for better results",
        },
        "force_dedicated_env": {
            "type": bool,
            "default": False,
            "description": "Force using dedicated environment even if CAREamics is available",
        },
    },
)
def careamics_denoise(
    image: np.ndarray,
    checkpoint_path: str = "",
    tile_size_z: int = 64,
    tile_size_y: int = 64,
    tile_size_x: int = 64,
    tile_overlap_z: int = 8,
    tile_overlap_y: int = 8,
    tile_overlap_x: int = 8,
    batch_size: int = 1,
    use_tta: bool = True,
    force_dedicated_env: bool = False,
) -> np.ndarray:
    """
    Denoise images using CAREamics models.

    This function loads a CAREamics model from a checkpoint file and uses it to denoise
    the input image. The function supports both 2D and 3D data and handles tiling for
    processing large images efficiently.

    If CAREamics is not installed in the main environment, a dedicated virtual environment
    will be automatically created and managed.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image to denoise
    checkpoint_path : str
        Path to the CAREamics model checkpoint file (.ckpt)
    tile_size_z : int
        Tile size in Z dimension (for 3D data)
    tile_size_y : int
        Tile size in Y dimension
    tile_size_x : int
        Tile size in X dimension
    tile_overlap_z : int
        Tile overlap in Z dimension (for 3D data)
    tile_overlap_y : int
        Tile overlap in Y dimension
    tile_overlap_x : int
        Tile overlap in X dimension
    batch_size : int
        Batch size for prediction
    use_tta : bool
        Use test-time augmentation for better results
    force_dedicated_env : bool
        Force using dedicated environment even if CAREamics is available

    Returns:
    --------
    numpy.ndarray
        Denoised image with the same dimensions as the input
    """
    # Verify checkpoint path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        print("Please provide a valid checkpoint file path.")
        return image

    # Determine whether to use dedicated environment
    use_env = force_dedicated_env or USE_DEDICATED_ENV

    careamics_denoise.thread_safe = False

    if use_env:
        print("Using dedicated CAREamics environment...")

        # First check if the environment exists, create if not
        if not is_env_created():
            print(
                "Creating dedicated CAREamics environment (this may take a few minutes)..."
            )
            create_careamics_env()
            print("Environment created successfully.")

        # Prepare arguments for the CAREamics function
        args = {
            "image": image,
            "checkpoint_path": checkpoint_path,
            "tile_size_z": tile_size_z,
            "tile_size_y": tile_size_y,
            "tile_size_x": tile_size_x,
            "tile_overlap_z": tile_overlap_z,
            "tile_overlap_y": tile_overlap_y,
            "tile_overlap_x": tile_overlap_x,
            "batch_size": batch_size,
            "use_tta": use_tta,
        }

        # Calculate tile overlap automatically (e.g., 25% of tile size)
        def compute_overlap(tile_size, fraction=0.25):
            # Ensure overlap is at least 1 and less than tile size
            overlap = max(1, int(tile_size * fraction))
            return min(overlap, tile_size - 1)

        # Inside careamics_denoise, after parsing tile sizes:
        tile_overlap_z = (
            compute_overlap(tile_size_z) if len(image.shape) >= 3 else None
        )
        tile_overlap_y = compute_overlap(tile_size_y)
        tile_overlap_x = compute_overlap(tile_size_x)

        # Run CAREamics in the dedicated environment
        print("Running CAREamics in dedicated environment...")
        return run_careamics_in_env("predict", args)

    else:
        print("Running CAREamics in current environment...")
        # Use CAREamics directly in the current environment
        try:
            print(f"Loading CAREamics model from: {checkpoint_path}")
            # Initialize the CAREamist model
            careamist = CAREamist(
                checkpoint_path, os.path.dirname(checkpoint_path)
            )

            # Determine dimensionality
            is_3d = len(image.shape) >= 3

            if is_3d:
                print(f"Processing 3D data with shape: {image.shape}")
                # Determine axes based on dimensionality
                if len(image.shape) == 3:
                    # ZYX format
                    axes = "ZYX"
                    tile_size = (tile_size_z, tile_size_y, tile_size_x)
                    tile_overlap = (
                        tile_overlap_z,
                        tile_overlap_y,
                        tile_overlap_x,
                    )
                    print(f"Using axes configuration: {axes}")
                elif len(image.shape) == 4:
                    # Assuming TZYX format
                    axes = "TZYX"
                    tile_size = (tile_size_z, tile_size_y, tile_size_x)
                    tile_overlap = (
                        tile_overlap_z,
                        tile_overlap_y,
                        tile_overlap_x,
                    )
                    print(f"Using axes configuration: {axes}")
                else:
                    # Unknown format, try to handle it
                    print(
                        f"Warning: Unusual data shape: {image.shape}. Defaulting to 'TZYX'"
                    )
                    axes = "TZYX"
                    tile_size = (tile_size_z, tile_size_y, tile_size_x)
                    tile_overlap = (
                        tile_overlap_z,
                        tile_overlap_y,
                        tile_overlap_x,
                    )
            else:
                print(f"Processing 2D data with shape: {image.shape}")
                # 2D data
                if len(image.shape) == 2:
                    # YX format
                    axes = "YX"
                    tile_size = (tile_size_y, tile_size_x)
                    tile_overlap = (tile_overlap_y, tile_overlap_x)
                    print(f"Using axes configuration: {axes}")
                else:
                    # Unknown format, try to handle it
                    print(
                        f"Warning: Unusual data shape: {image.shape}. Defaulting to 'YX'"
                    )
                    axes = "YX"
                    tile_size = (tile_size_y, tile_size_x)
                    tile_overlap = (tile_overlap_y, tile_overlap_x)

            # Run the prediction
            print(
                f"Running prediction with tile size: {tile_size}, overlap: {tile_overlap}"
            )
            print(f"Using batch size: {batch_size}, TTA: {use_tta}")

            prediction = careamist.predict(
                source=image,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                axes=axes,
                batch_size=batch_size,
                tta=use_tta,
            )

            # # Handle output shape
            # if prediction.shape != image.shape:
            #     print(f"Warning: Prediction shape {prediction.shape} differs from input shape {image.shape}")
            #     prediction = np.squeeze(prediction)

            #     # If shapes still don't match, try to reshape
            #     if prediction.shape != image.shape:
            #         print(f"Warning: Shapes still don't match after squeezing. Using original dimensions.")
            #         try:
            #             prediction = prediction.reshape(image.shape)
            #         except ValueError:
            #             print("Error: Could not reshape prediction to match input shape.")
            #             return image

            # print(f"Denoising completed. Output shape: {prediction.shape}")
            return prediction

        except (RuntimeError, ValueError, ImportError) as e:
            import traceback

            print(
                f"Error during CAREamics denoising in current environment: {str(e)}"
            )
            traceback.print_exc()

            # If we haven't already tried using the dedicated environment, try that as a fallback
            if not force_dedicated_env:
                print(
                    "Attempting fallback to dedicated CAREamics environment..."
                )
                args = {
                    "image": image,
                    "checkpoint_path": checkpoint_path,
                    "tile_size_z": tile_size_z,
                    "tile_size_y": tile_size_y,
                    "tile_size_x": tile_size_x,
                    "tile_overlap_z": tile_overlap_z,
                    "tile_overlap_y": tile_overlap_y,
                    "tile_overlap_x": tile_overlap_x,
                    "batch_size": batch_size,
                    "use_tta": use_tta,
                }

                if not is_env_created():
                    create_careamics_env()

                return run_careamics_in_env("predict", args)

            return None

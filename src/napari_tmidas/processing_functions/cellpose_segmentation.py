# processing_functions/cellpose_segmentation.py
"""
Processing functions for automatic instance segmentation using Cellpose.

This module provides functionality to automatically segment cells or nuclei in images
using the Cellpose deep learning-based segmentation toolkit. It supports both 2D and 3D images,
various dimension orders, and handles time series data.

Note: This requires the cellpose library to be installed.
"""
import os
import sys

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
    diameter,
    flow_threshold=0.4,
    dim_order="ZYX",
    max_pixels=4000000,
):
    """
    Run Cellpose segmentation on an image.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    model : cellpose.models.Cellpose
        Cellpose model to use
    channels : list
        Channels to use for segmentation [0,0] = grayscale, [1,0] = green channel, [2,0] = red channel
    diameter : float
        Diameter of objects to segment
    flow_threshold : float
        Flow threshold for Cellpose
    dim_order : str
        Dimension order of the input image
    max_pixels : int
        Maximum number of pixels to process (for 2D images)

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

    if has_time:
        # Handle time series - process each time point
        n_timepoints = img_transposed.shape[0]
        result = np.zeros(img_transposed.shape, dtype=np.uint32)

        # Process each time point
        for t in range(n_timepoints):
            img_t = img_transposed[t]
            mask, _, _, _ = model.eval(
                img_t,
                diameter=diameter,
                flow_threshold=flow_threshold,
                channels=channels,
                z_axis=0 if is_3d else None,
                do_3D=is_3d,
                niter=2000,  # Maximum iterations
            )
            result[t] = mask
    else:
        # Process single time point
        result, _, _, _ = model.eval(
            img_transposed,
            diameter=diameter,
            flow_threshold=flow_threshold,
            channels=channels,
            z_axis=0 if is_3d else None,
            do_3D=is_3d,
            niter=2000,  # Maximum iterations
        )

    return result.astype(np.uint32)


@BatchProcessingRegistry.register(
    name="Segment cells or nuclei (Cellpose)",
    suffix="_labels",
    description="Automatic instance segmentation using Cellpose 3.0 (dedicated environment)",
    parameters={
        "model_type": {
            "type": str,
            "default": "cyto3",
            "description": "Cellpose model type: 'cyto'/'cyto2'/'cyto3' for cells, 'nuclei' for nuclei",
        },
        "diameter": {
            "type": float,
            "default": 40.0,
            "min": 5.0,
            "max": 1000.0,
            "description": "Approximate diameter of objects to segment (pixels)",
        },
        "dim_order": {
            "type": str,
            "default": "YX",
            "description": "Dimension order of the input (e.g., 'YX', 'ZYX', 'TZYX')",
        },
        "channel_1": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 3,
            "description": "First channel: 0=grayscale, 1=green, 2=red, 3=blue",
        },
        "channel_2": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 3,
            "description": "Second channel: 0=none, 1=green, 2=red, 3=blue",
        },
        "flow_threshold": {
            "type": float,
            "default": 0.4,
            "min": 0.1,
            "max": 0.9,
            "description": "Flow threshold for Cellpose segmentation",
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
    model_type: str = "cyto3",
    diameter: float = 40.0,
    dim_order: str = "YX",
    channel_1: int = 0,
    channel_2: int = 0,
    flow_threshold: float = 0.4,
    force_dedicated_env: bool = False,
) -> np.ndarray:
    """
    Run Cellpose segmentation on an image.

    This function takes an image and performs automatic instance segmentation using
    Cellpose. It supports both 2D and 3D images, various dimension orders, and handles
    time series data.

    If Cellpose is not available in the current environment, a dedicated virtual
    environment will be created to run Cellpose.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    model_type : str
        Cellpose model type: 'cyto'/'cyto2'/'cyto3' for cells, 'nuclei' for nuclei (default: "cyto3")
    diameter : float
        Approximate diameter of objects to segment in pixels (default: 40.0)
    dim_order : str
        Dimension order of the input (e.g., 'YX', 'ZYX', 'TZYX') (default: "YX")
    channel_1 : int
        First channel: 0=grayscale, 1=green, 2=red, 3=blue (default: 0)
    channel_2 : int
        Second channel: 0=none, 1=green, 2=red, 3=blue (default: 0)
    flow_threshold : float
        Flow threshold for Cellpose segmentation (default: 0.4)
    force_dedicated_env : bool
        Force using dedicated environment even if Cellpose is available (default: False)

    Returns:
    --------
    numpy.ndarray
        Segmented image with instance labels
    """
    # Convert channel parameters to Cellpose channels list
    channels = [channel_1, channel_2]

    # Validate parameters
    valid_models = ["cyto", "cyto2", "cyto3", "nuclei"]
    if model_type not in valid_models:
        raise ValueError(
            f"Invalid model_type: {model_type}. "
            f"Must be one of: {', '.join(valid_models)}"
        )

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
            "model_type": model_type,
            "diameter": diameter,
            "channels": channels,
            "flow_threshold": flow_threshold,
            "use_gpu": USE_GPU,
            "do_3D": "Z" in dim_order,
            "z_axis": 0 if "Z" in dim_order else None,
        }

        # Run Cellpose in the dedicated environment
        print(f"Running Cellpose ({model_type}) in dedicated environment...")
        result = run_cellpose_in_env("eval", args)
        print(f"Segmentation complete. Found {np.max(result)} objects.")
        return result

    else:
        print(f"Running Cellpose ({model_type}) in current environment...")
        # Initialize Cellpose model in current environment
        model = models.Cellpose(gpu=USE_GPU, model_type=model_type)

    # Print status information
    gpu_status = "GPU" if USE_GPU else "CPU"
    print(f"Using Cellpose {model_type} model on {gpu_status}")
    print(
        f"Processing image with shape {image.shape}, dimension order: {dim_order}"
    )

    # Run segmentation
    try:
        result = run_cellpose(
            image, model, channels, diameter, flow_threshold, dim_order
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
                    "model_type": model_type,
                    "diameter": diameter,
                    "channels": channels,
                    "flow_threshold": flow_threshold,
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


# Add a command-line function to run cellpose segmentation
def run_cellpose_segmentation():
    """Run Cellpose segmentation from the command line."""
    import argparse

    from skimage.io import imread
    from tifffile import imwrite
    from tqdm import tqdm

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Runs automatic mask generation on images using Cellpose."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input images."
    )
    parser.add_argument(
        "--diameter", type=float, default=40.0, help="Diameter of objects."
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0, 0],
        help="Channels to use.",
    )
    parser.add_argument(
        "--dim_order",
        type=str,
        default="ZYX",
        help="Dimension order of the input images.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cyto3",
        choices=["cyto", "cyto2", "cyto3", "nuclei"],
        help="Model type: 'cyto'/'cyto2'/'cyto3' for cells, 'nuclei' for nuclei",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.4,
        help="Flow threshold for Cellpose (default: 0.4)",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=4000000,
        help="Maximum number of pixels to process for 2D images (default: 4000000)",
    )

    args = parser.parse_args()

    # Validate input folder
    input_folder = args.input
    if not os.path.isdir(input_folder):
        print(
            f"Error: The input folder '{input_folder}' does not exist or is not accessible."
        )
        return 1

    # Find input files
    input_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith(".tif") and not f.endswith("_labels.tif")
    ]

    if not input_files:
        print(f"No .tif files found in {input_folder}")
        return 1

    print(f"Found {len(input_files)} files to process")

    # Check if Cellpose is available
    if not CELLPOSE_AVAILABLE:
        print(
            "Error: Cellpose is not installed. Please install it with: pip install cellpose"
        )
        return 1

    # Initialize model
    model = models.Cellpose(gpu=USE_GPU, model_type=args.model_type)

    # Print status
    gpu_status = "GPU" if USE_GPU else "CPU"
    print(f"Using Cellpose {args.model_type} model on {gpu_status}")

    # Process each file
    for input_file in tqdm(input_files, desc="Processing images"):
        try:
            # Check image size
            img = imread(os.path.join(input_folder, input_file))

            if len(img.shape) == 2 or (
                len(img.shape) == 3 and "C" in args.dim_order
            ):
                # For 2D images (potentially with channels)
                height, width = img.shape[:2]
                total_pixels = height * width
                if total_pixels > args.max_pixels:
                    print(
                        f"Skipping {input_file} as it exceeds the maximum size of {args.max_pixels} pixels."
                    )
                    continue

            print(
                "\nCheck if image shape corresponds to the dim order that you have given:"
            )
            print(
                f"Image shape: {img.shape}, dimension order: {args.dim_order}\n"
            )

            # Run segmentation
            result = run_cellpose(
                img,
                model,
                args.channels,
                args.diameter,
                args.flow_threshold,
                args.dim_order,
                args.max_pixels,
            )

            # Save result
            output_file = os.path.join(
                input_folder, input_file.replace(".tif", "_labels.tif")
            )
            imwrite(output_file, result, compression="zlib")

            print(
                f"Saved segmentation with {np.max(result)} objects to {output_file}"
            )

        except (Exception, MemoryError) as e:
            print(f"Error processing {input_file}: {str(e)}")

    print("\nProcessing complete.")
    return 0


# Run the command-line function if this script is run directly
if __name__ == "__main__":
    import sys

    sys.exit(run_cellpose_segmentation())

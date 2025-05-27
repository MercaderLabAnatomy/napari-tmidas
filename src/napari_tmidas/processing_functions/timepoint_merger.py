# processing_functions/timepoint_merger.py
"""
Processing function for merging timepoint folders into time series stacks.

This module provides functionality to merge folders containing individual timepoint images
into single time series stacks (TZYX or TYX format). It can handle both 2D (YX) and 3D (ZYX)
input images and automatically sorts them by filename to maintain temporal order.

The function works by detecting when it's processing the first file from a folder,
then loading ALL files from that folder to create a merged time series.
"""

import os
import re
from typing import List, Tuple

import numpy as np
import tifffile
from skimage.io import imread

from napari_tmidas._registry import BatchProcessingRegistry


def natural_sort_key(filename: str) -> List:
    """
    Generate a key for natural sorting of filenames containing numbers.

    This ensures that files are sorted in the correct order:
    file1.tif, file2.tif, ..., file10.tif, file11.tif
    instead of: file1.tif, file10.tif, file11.tif, file2.tif
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", filename)
    ]


def find_timepoint_images(
    folder_path: str, file_extensions: List[str] = None
) -> List[str]:
    """
    Find and sort image files in a folder.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing timepoint images
    file_extensions : List[str], optional
        List of file extensions to look for (default: ['.tif', '.tiff', '.png', '.jpg'])

    Returns:
    --------
    List[str]
        Sorted list of image file paths
    """
    if file_extensions is None:
        file_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    # Find all image files
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in file_extensions):
            image_files.append(os.path.join(folder_path, file))

    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")

    # Sort files naturally (handling numbers correctly)
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

    return image_files


def load_and_validate_images(image_files: List[str]) -> Tuple[np.ndarray, str]:
    """
    Load all images and validate they have consistent dimensions.

    Parameters:
    -----------
    image_files : List[str]
        List of image file paths

    Returns:
    --------
    Tuple[np.ndarray, str]
        Tuple of (stacked images array, dimension order string)
    """
    print(f"Loading {len(image_files)} timepoint images...")

    # Load first image to determine dimensions and data type
    first_image = imread(image_files[0])
    print(
        f"First image shape: {first_image.shape}, dtype: {first_image.dtype}"
    )

    # Determine dimension order
    if len(first_image.shape) == 2:
        # 2D image (YX)
        dimension_order = "TYX"
        expected_shape = first_image.shape
    elif len(first_image.shape) == 3:
        # 3D image (ZYX) - assuming Z is the first dimension
        dimension_order = "TZYX"
        expected_shape = first_image.shape
    else:
        raise ValueError(
            f"Unsupported image dimensionality: {first_image.shape}"
        )

    # Pre-allocate array for all timepoints
    stack_shape = (len(image_files),) + expected_shape
    print(
        f"Creating time series with shape: {stack_shape} ({dimension_order})"
    )

    # Use the same dtype as the first image
    time_series = np.zeros(stack_shape, dtype=first_image.dtype)

    # Load all images
    time_series[0] = first_image

    for i, image_file in enumerate(image_files[1:], 1):
        try:
            image = imread(image_file)

            # Validate shape consistency
            if image.shape != expected_shape:
                raise ValueError(
                    f"Image {os.path.basename(image_file)} has shape {image.shape}, "
                    f"expected {expected_shape}. All images must have the same dimensions."
                )

            # Validate dtype consistency
            if image.dtype != first_image.dtype:
                print(
                    f"Warning: Converting {os.path.basename(image_file)} from {image.dtype} to {first_image.dtype}"
                )
                image = image.astype(first_image.dtype)

            time_series[i] = image

            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Loaded {i + 1}/{len(image_files)} images")

        except Exception as e:
            raise ValueError(f"Error loading {image_file}: {str(e)}") from e

    print(f"Successfully loaded all {len(image_files)} timepoints")
    return time_series, dimension_order


# Global variable to track which folders have been processed
_processed_folders = set()


# Advanced version with more options
@BatchProcessingRegistry.register(
    name="Merge Timepoints",
    suffix="_merge_timeseries",
    description="Advanced timepoint merging with subsampling and memory optimization. IMPORTANT: Set thread count to 1!",
    parameters={
        "subsample_factor": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 10,
            "description": "Take every Nth timepoint (1 = all timepoints, 2 = every other, etc.)",
        },
        "max_timepoints": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 10000,
            "description": "Maximum number of timepoints to include (0 = no limit)",
        },
        "start_timepoint": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 1000,
            "description": "Starting timepoint index (0-based)",
        },
        "memory_efficient": {
            "type": bool,
            "default": False,
            "description": "Use memory-efficient loading for very large datasets",
        },
    },
)
def merge_timepoint_folder_advanced(
    image: np.ndarray,
    subsample_factor: int = 1,
    max_timepoints: int = 0,
    start_timepoint: int = 0,
    memory_efficient: bool = False,
) -> np.ndarray:
    """
    Advanced timepoint merging with additional options for large datasets.

    This function provides additional control over the merging process, including
    subsampling, time range selection, and memory-efficient processing for large datasets.

    IMPORTANT: This function should be run with thread count = 1 in the batch processing
    widget, as it processes entire folders at once.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image (used to determine the current file being processed)
    subsample_factor : int
        Take every Nth timepoint (1 = all, 2 = every other, etc.)
    max_timepoints : int
        Maximum number of timepoints to include (0 = no limit)
    start_timepoint : int
        Starting timepoint index (0-based)
    memory_efficient : bool
        Use memory-efficient loading (loads images one at a time)

    Returns:
    --------
    numpy.ndarray
        Time series array with selected timepoints
    """
    global _processed_folders

    # Get folder path and file suffix from batch processing context
    import inspect

    current_file = None
    output_folder = None
    input_suffix = None

    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "filepath" in frame_locals:
            current_file = frame_locals["filepath"]
        if "self" in frame_locals:
            obj = frame_locals["self"]
            if hasattr(obj, "output_folder") and hasattr(obj, "input_suffix"):
                output_folder = obj.output_folder
                input_suffix = obj.input_suffix
                break

    if current_file is None:
        raise ValueError("Could not determine current file path")

    folder_path = os.path.dirname(current_file)
    folder_name = os.path.basename(folder_path)

    if output_folder is None:
        output_folder = folder_path

    if input_suffix is None:
        input_suffix = os.path.splitext(current_file)[1]

    # Check if already processed
    advanced_key = f"{folder_path}_advanced"
    if advanced_key in _processed_folders:
        print(
            f"Advanced processing for {folder_name} already completed, skipping..."
        )
        return image

    _processed_folders.add(advanced_key)

    print(f"🔄 ADVANCED PROCESSING FOLDER: {folder_name}")
    print(f"Using file suffix: {input_suffix}")

    # Use the same suffix from the batch processing widget
    extensions = [input_suffix]

    # Find all timepoint images
    try:
        image_files = find_timepoint_images(folder_path, extensions)
        image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

        print(f"Found {len(image_files)} total timepoints")

        # Show all filenames for verification
        print("📁 Complete file list:")
        for i, file_path in enumerate(image_files):
            print(f"  {i:2d}: {os.path.basename(file_path)}")
            # Show a break after 20 files to avoid too much output
            if i == 19 and len(image_files) > 25:
                print(
                    f"  ... (showing first 20 and last 5 of {len(image_files)} files)"
                )
                break

        # If we had a break, show the last few files
        if len(image_files) > 25:
            for i, file_path in enumerate(
                image_files[-5:], len(image_files) - 5
            ):
                print(f"  {i:2d}: {os.path.basename(file_path)}")

        # Apply timepoint selection
        if start_timepoint > 0:
            if start_timepoint >= len(image_files):
                _processed_folders.discard(advanced_key)
                raise ValueError(
                    f"start_timepoint ({start_timepoint}) >= total timepoints ({len(image_files)})"
                )
            image_files = image_files[start_timepoint:]
            print(
                f"Starting from timepoint {start_timepoint}: {len(image_files)} remaining"
            )

        # Apply subsampling
        if subsample_factor > 1:
            image_files = image_files[::subsample_factor]
            print(
                f"Subsampling by factor {subsample_factor}: {len(image_files)} timepoints selected"
            )

        # Apply maximum timepoints limit
        if max_timepoints > 0 and len(image_files) > max_timepoints:
            image_files = image_files[:max_timepoints]
            print(f"Limited to {max_timepoints} timepoints")

        if len(image_files) < 1:
            _processed_folders.discard(advanced_key)
            raise ValueError("No timepoints selected after applying filters")

        print(f"Final selection: {len(image_files)} timepoints")

        # Show final selection if filtering was applied
        if subsample_factor > 1 or max_timepoints > 0 or start_timepoint > 0:
            print("📁 Final selected files:")
            for i, file_path in enumerate(image_files):
                print(f"  {i:2d}: {os.path.basename(file_path)}")

        # Load images based on memory efficiency setting
        if memory_efficient and len(image_files) > 100:
            print("Using memory-efficient loading...")

            # Load first image to determine shape and dtype
            first_image = imread(image_files[0])
            stack_shape = (len(image_files),) + first_image.shape

            # Create memory-mapped array if possible, otherwise regular array
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    time_series = np.memmap(
                        temp_file.name,
                        dtype=first_image.dtype,
                        mode="w+",
                        shape=stack_shape,
                    )
                    print(f"Created memory-mapped array: {stack_shape}")
            except (OSError, ValueError):
                time_series = np.zeros(stack_shape, dtype=first_image.dtype)
                print(f"Created regular array: {stack_shape}")

            time_series[0] = first_image

            # Load remaining images one by one
            for i, image_file in enumerate(image_files[1:], 1):
                if i % 50 == 0:
                    print(f"Loading timepoint {i+1}/{len(image_files)}")

                img = imread(image_file)
                if img.shape != first_image.shape:
                    _processed_folders.discard(advanced_key)
                    raise ValueError(
                        f"Shape mismatch at timepoint {i}: {img.shape} vs {first_image.shape}"
                    )

                time_series[i] = img

            # Convert back to regular array if using memmap
            if isinstance(time_series, np.memmap):
                result = np.array(time_series)
                del time_series  # Clean up memmap
                time_series = result
        else:
            # Use standard loading
            time_series = load_and_validate_images(image_files)[0]

        # Save the advanced time series
        output_filename = f"{folder_name}_merged_timepoints.tif"
        output_path = os.path.join(output_folder, output_filename)

        print(f"💾 Saving advanced time series to: {output_path}")

        size_gb = time_series.nbytes / (1024**3)
        use_bigtiff = size_gb > 2.0

        tifffile.imwrite(
            output_path,
            time_series,
            compression="zlib",
            bigtiff=use_bigtiff,
        )

        print("✅ Successfully saved advanced time series!")
        print(f"📁 Output file: {output_filename}")
        print(f"📊 File size: {size_gb:.2f} GB")
        print(f"📐 Final shape: {time_series.shape}")

        # IMPORTANT: Return the original image unchanged so the batch processor
        # doesn't save individual processed files. The merged file is already saved above.
        return image

    except Exception as e:
        _processed_folders.discard(advanced_key)
        raise ValueError(
            f"Error in advanced timepoint merging: {str(e)}"
        ) from e


# Command-line utility function
def merge_timepoints_cli():
    """Command-line interface for merging timepoint folders."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge timepoint images into time series"
    )
    parser.add_argument(
        "input_folder", help="Folder containing timepoint images"
    )
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument(
        "--extensions",
        default=".tif,.tiff,.png,.jpg",
        help="File extensions to include (comma-separated)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor (take every Nth timepoint)",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=0,
        help="Maximum number of timepoints (0 = no limit)",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting timepoint index"
    )

    args = parser.parse_args()

    try:
        # Parse extensions
        extensions = [
            ext.strip() for ext in args.extensions.split(",") if ext.strip()
        ]

        # Find and sort files
        image_files = find_timepoint_images(args.input_folder, extensions)

        # Apply filters
        if args.start > 0:
            image_files = image_files[args.start :]
        if args.subsample > 1:
            image_files = image_files[:: args.subsample]
        if args.max_timepoints > 0:
            image_files = image_files[: args.max_timepoints]

        # Load and save
        result = load_and_validate_images(image_files)[0]
        tifffile.imwrite(args.output_file, result, compression="zlib")

        print(f"Successfully saved time series to {args.output_file}")
        print(f"Final shape: {result.shape}")
        print(f"Data type: {result.dtype}")
        print(
            f"File size: {os.path.getsize(args.output_file) / (1024**2):.1f} MB"
        )

    except (ValueError, OSError, RuntimeError) as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(merge_timepoints_cli())

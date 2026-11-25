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
from skimage.io import imread as _skimage_imread

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions.ome_output_utils import (
    _extract_source_physical_scale,
)


def imread(path):
    """
    Read one timepoint, keeping its axis order intact.

    ``skimage.io.imread`` interprets a leading axis of length 3 or 4 as RGB(A)
    samples and moves it last, so a 3-slice Z-stack came back as (Y, X, Z)
    while the merged stack was still labelled TZYX -- transposed, mislabelled
    and silent.  tifffile reads a TIFF's own shape metadata instead, so the
    axes survive; anything else still goes through skimage.
    """
    if str(path).lower().endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    return _skimage_imread(path)


# Global set to track which folders have been processed in the current session
# This prevents redundant processing when the function is called for each file
_PROCESSED_FOLDERS = set()


def reset_timepoint_merger_cache():
    """
    Reset the cache of processed folders.

    Call this function if you want to reprocess folders that were already
    processed in the current session. This is automatically managed during
    normal batch processing, but can be called manually if needed.
    """
    global _PROCESSED_FOLDERS
    _PROCESSED_FOLDERS.clear()
    print("🔄 Timepoint merger cache cleared")


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


def _resolve_merge_order(
    first_shape: Tuple[int, ...], dimension_order: str = "auto"
) -> Tuple[str, str, bool]:
    """
    How files of `first_shape` merge: (input order, output order, joins_time).

    ``joins_time`` means the files already carry a T axis and are
    concatenated along it (TZYX); otherwise each file becomes one timepoint
    on a new leading T axis.  "Auto" is matched case-insensitively: the batch
    widget's dropdown sends "Auto", and comparing it against "auto" made the
    default setting fail every 3D series as a 4-letter order "AUTO".
    """
    ndim = len(first_shape)
    if str(dimension_order or "auto").lower() == "auto":
        if ndim == 2:
            return "YX", "TYX", False
        if ndim == 3:
            print("⚠️  3D images detected - assuming ZYX (Z-stack)")
            print(
                "   If this is CYX (color channels), set dimension_order='CYX'"
            )
            return "ZYX", "TZYX", False
        if ndim == 4:
            print("⚠️  4D images detected - assuming TZYX (time series)")
            print(
                "   If this is CZYX (color Z-stack), set dimension_order='CZYX'"
            )
            return "TZYX", "TZYX", True
        raise ValueError(f"Unsupported image dimensionality: {first_shape}")

    detected_order = dimension_order.upper()
    print(f"Using specified dimension order: {detected_order}")
    if ndim != len(detected_order):
        raise ValueError(
            f"Dimension order '{detected_order}' expects {len(detected_order)}D data, "
            f"but images have shape {first_shape} ({ndim}D)"
        )
    if detected_order == "YX":
        return detected_order, "TYX", False
    if detected_order in ("ZYX", "CYX"):
        return detected_order, "T" + detected_order, False  # TZYX or TCYX
    if detected_order == "TZYX":
        # Files are joined along the T axis they already have, so the result
        # is still 4D TZYX.  Prefixing another "T" labelled a 4-axis array as
        # 5-axis ("TTZYX"), which every consumer of this string mis-parsed.
        return detected_order, "TZYX", True
    if detected_order == "CZYX":
        return detected_order, "TCZYX", False  # one timepoint per file
    raise ValueError(f"Unsupported dimension order: {detected_order}")


def _merged_shape(
    first_shape: Tuple[int, ...], n_files: int, joins_time: bool
) -> Tuple[int, ...]:
    if joins_time:
        return (n_files * first_shape[0],) + tuple(first_shape[1:])
    return (n_files,) + tuple(first_shape)


def _iter_validated_images(image_files: List[str], first_image: np.ndarray):
    """
    Yield each file's array in order, `first_image` standing in for the
    first: same shape as it or ValueError, cast to its dtype with a warning.
    Only one file is held at a time.
    """
    yield first_image
    for i, image_file in enumerate(image_files[1:], 1):
        try:
            image = imread(image_file)
            if image.shape != first_image.shape:
                raise ValueError(
                    f"Image {os.path.basename(image_file)} has shape {image.shape}, "
                    f"expected {first_image.shape}. All images must have the same dimensions."
                )
            if image.dtype != first_image.dtype:
                print(
                    f"Warning: Converting {os.path.basename(image_file)} from {image.dtype} to {first_image.dtype}"
                )
                image = image.astype(first_image.dtype)
        except Exception as e:
            raise ValueError(f"Error loading {image_file}: {str(e)}") from e
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"Loaded {i + 1}/{len(image_files)} files")
        yield image


def load_and_validate_images(
    image_files: List[str], dimension_order: str = "auto"
) -> Tuple[np.ndarray, str]:
    """
    Load all images and validate they have consistent dimensions.

    Holds the whole series in memory; the batch function streams it to disk
    instead (see `_write_time_series`).  Kept for the CLI and for callers
    that want the array.

    Parameters:
    -----------
    image_files : List[str]
        List of image file paths
    dimension_order : str
        Dimension order of input files: "auto", "YX", "ZYX", "CYX", "CZYX", or "TZYX"

    Returns:
    --------
    Tuple[np.ndarray, str]
        Tuple of (stacked images array, dimension order string)
    """
    print(f"Loading {len(image_files)} timepoint images...")
    first_image = imread(image_files[0])
    print(
        f"First image shape: {first_image.shape}, dtype: {first_image.dtype}"
    )
    _, output_order, joins_time = _resolve_merge_order(
        first_image.shape, dimension_order
    )
    stack_shape = _merged_shape(first_image.shape, len(image_files), joins_time)
    print(f"Creating time series with shape: {stack_shape} ({output_order})")

    time_series = np.zeros(stack_shape, dtype=first_image.dtype)
    step = first_image.shape[0] if joins_time else 1
    for i, image in enumerate(_iter_validated_images(image_files, first_image)):
        if joins_time:
            time_series[i * step : (i + 1) * step] = image
        else:
            time_series[i] = image

    print(f"Successfully loaded all {len(image_files)} timepoints")
    return time_series, output_order


def _write_time_series(
    image_files: List[str], dimension_order: str, output_path: str
) -> Tuple[Tuple[int, ...], str, float]:
    """
    Merge `image_files` into an OME-TIFF at `output_path`, one YX plane at a
    time: only one input file is in memory at once, never the series.

    The previous path allocated the whole series before writing it -- and
    its "memory efficient" variant filled a memmap, then copied it back into
    RAM with np.array() before saving.  The file is written beside the
    target and renamed into place, so a failure partway (a shape mismatch in
    file 900 of 1000) leaves no partial output.

    Returns (shape, axes, size in GB).
    """
    from napari_tmidas.processing_functions.ome_output_utils import (
        stream_planes_to_tiff,
    )

    first_image = imread(image_files[0])
    print(
        f"First image shape: {first_image.shape}, dtype: {first_image.dtype}"
    )
    _, output_order, joins_time = _resolve_merge_order(
        first_image.shape, dimension_order
    )
    stack_shape = _merged_shape(first_image.shape, len(image_files), joins_time)
    print(f"Creating time series with shape: {stack_shape} ({output_order})")

    # The axes are the merge's own output order: deriving them from ndim
    # alone labelled a merged CYX series (TCYX) as TZYX.
    ome_metadata = {"axes": output_order}
    # Timepoint files in a series share a physical scale; the first file is
    # representative.  Without this the merged output reset to isotropic.
    for ax_name, ax_scale in _extract_source_physical_scale(
        image_files[0], output_order
    ).items():
        ome_metadata[f"PhysicalSize{ax_name}"] = ax_scale
        ome_metadata[f"PhysicalSize{ax_name}Unit"] = "um"

    def planes():
        for image in _iter_validated_images(image_files, first_image):
            for index in np.ndindex(*image.shape[:-2]):
                yield image[index]

    size_gb = (
        int(np.prod(stack_shape, dtype=np.int64))
        * first_image.dtype.itemsize
        / (1024**3)
    )
    tmp_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        f".{os.path.basename(output_path)}.tmp-{os.getpid()}",
    )
    try:
        stream_planes_to_tiff(
            tmp_path,
            planes(),
            stack_shape,
            first_image.dtype,
            metadata=ome_metadata,
            bigtiff=size_gb > 2.0,
        )
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"Successfully merged all {len(image_files)} timepoints")
    return stack_shape, output_order, size_gb


@BatchProcessingRegistry.register(
    name="Merge Timepoints",
    suffix="_merge_timeseries",
    description="Merge folder timepoints into time series. Processes each folder ONCE (skips redundant calls). Set thread count to 1!",
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
            "description": "No longer needed: every merge streams to disk one timepoint at a time. Kept so saved workflows still load.",
        },
        "overwrite_existing": {
            "type": bool,
            "default": False,
            "description": "Overwrite existing merged file if it exists",
        },
        # NOTE: the dimension order is not declared here on purpose --
        # it comes from the batch widget's "Dimension Order" dropdown,
        # which applies to every function. Declaring it again made the
        # user set the same thing twice, in two places that could disagree.
    },
)
def merge_timepoint_folder_advanced(
    image: np.ndarray,
    subsample_factor: int = 1,
    max_timepoints: int = 0,
    start_timepoint: int = 0,
    memory_efficient: bool = False,
    overwrite_existing: bool = False,
    dimension_order: str = "auto",
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
    overwrite_existing : bool
        Overwrite existing merged file if it exists

    Returns:
    --------
    numpy.ndarray
        Time series array with selected timepoints
    """
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

    # Generate output filename with parameters in the name for uniqueness
    param_suffix = ""
    if subsample_factor > 1:
        param_suffix += f"_sub{subsample_factor}"
    if start_timepoint > 0:
        param_suffix += f"_start{start_timepoint}"
    if max_timepoints > 0:
        param_suffix += f"_max{max_timepoints}"

    output_filename = f"{folder_name}_merged_timepoints{param_suffix}.tif"
    output_path = os.path.join(output_folder, output_filename)

    # Create a unique key for this processing task (folder + parameters)
    processing_key = f"{folder_path}|{param_suffix}|{dimension_order}"

    # Check if this folder has already been processed in this session
    if processing_key in _PROCESSED_FOLDERS:
        print(f"✅ Folder already processed in this session: {folder_name}")
        print("   Skipping to avoid redundant processing")
        return image

    # Check if output file already exists
    if os.path.exists(output_path) and not overwrite_existing:
        print(f"🔵 Merged file already exists: {output_filename}")
        print(f"   Full path: {output_path}")
        print("   Skipping this folder. To reprocess:")
        print("   - Delete the existing file, or")
        print("   - Use a different output folder, or")
        print("   - Enable 'overwrite_existing' parameter")
        # Mark as processed so we don't check again for other files in this folder
        _PROCESSED_FOLDERS.add(processing_key)
        return image

    # If we're here and the file exists, we're overwriting
    if os.path.exists(output_path):
        print(f"⚠️  Overwriting existing file: {output_filename}")

    print(f"🔄 PROCESSING FOLDER: {folder_name}")
    print(f"📁 Output will be: {output_filename}")
    print(f"Using file suffix: {input_suffix}")

    # Use the same suffix from the batch processing widget
    # Split comma-separated suffixes into a list
    if isinstance(input_suffix, str):
        extensions = [s.strip() for s in input_suffix.split(",") if s.strip()]
    else:
        extensions = [input_suffix]

    # Find all timepoint images
    try:
        image_files = find_timepoint_images(folder_path, extensions)

        # Exclude the output file if it exists in the folder (BEFORE sorting)
        image_files = [f for f in image_files if f != output_path]

        # Now sort the remaining files
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
            raise ValueError("No timepoints selected after applying filters")

        print(f"Final selection: {len(image_files)} timepoints")

        # Show final selection if filtering was applied
        if subsample_factor > 1 or max_timepoints > 0 or start_timepoint > 0:
            print("📁 Final selected files:")
            for i, file_path in enumerate(image_files):
                print(f"  {i:2d}: {os.path.basename(file_path)}")

        # Stream the series to disk one plane at a time.  This used to load
        # it whole (and `memory_efficient` copied its memmap back into RAM
        # before saving), so the parameter is now a no-op: every merge is
        # memory efficient.
        print(f"💾 Saving time series to: {output_path}")
        stack_shape, _, size_gb = _write_time_series(
            image_files, dimension_order, output_path
        )

        print("✅ Successfully saved time series!")
        print(f"📁 Output file: {output_filename}")
        print(f"📊 File size: {size_gb:.2f} GB")
        print(f"📐 Final shape: {stack_shape}")

        # Mark this folder as processed to avoid redundant processing
        _PROCESSED_FOLDERS.add(processing_key)

        # IMPORTANT: Return the original image unchanged so the batch processor
        # doesn't save individual processed files. The merged file is already saved above.
        return image

    except Exception as e:
        raise ValueError(f"Error in timepoint merging: {str(e)}") from e


# Command-line utility function
# skip_load: the worker must not load the input for this function.  It reads
# the files it needs itself and only ever returns the input as a "nothing to
# save" signal -- so every call (one per file in the folder) loaded a file
# just to discard it.  With skip_load it gets image=None, and returning None
# skips the save the same way.
merge_timepoint_folder_advanced.skip_load = True


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
        result = load_and_validate_images(image_files, "auto")[0]
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

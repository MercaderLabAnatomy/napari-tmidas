# processing_functions/basic.py
"""
Basic image processing functions
"""
import concurrent.futures
import inspect
import os
import traceback
import warnings

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

# Lazy imports for optional heavy dependencies
try:
    import dask.array as da

    _HAS_DASK = True
except ImportError:
    da = None
    _HAS_DASK = False

try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False


def _to_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError("Label images must have at least one dimension")
    return arr


def _nonzero_bounds(arr: np.ndarray) -> list[tuple[int, int]]:
    if not np.any(arr):
        return [(0, size) for size in arr.shape]

    bounds: list[tuple[int, int]] = []
    dims = range(arr.ndim)
    for axis in dims:
        reduction_axes = tuple(i for i in dims if i != axis)
        axis_any = np.any(arr, axis=reduction_axes)
        nonzero_idx = np.flatnonzero(axis_any)
        if nonzero_idx.size == 0:
            bounds.append((0, arr.shape[axis]))
        else:
            bounds.append((int(nonzero_idx[0]), int(nonzero_idx[-1]) + 1))
    return bounds


def _match_ndim(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    result = np.asarray(candidate)
    if result.ndim > reference.ndim:
        axes = [idx for idx, size in enumerate(result.shape) if size == 1]
        while axes and result.ndim > reference.ndim:
            result = np.squeeze(result, axis=axes.pop())
    if result.ndim > reference.ndim:
        raise ValueError(
            "Unable to align label images with differing numbers of non-singleton dimensions"
        )
    while result.ndim < reference.ndim:
        result = result[np.newaxis, ...]
    return result


def _align_candidate(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    candidate = _match_ndim(reference, candidate)
    if candidate.shape == reference.shape:
        return candidate

    aligned = np.zeros_like(reference)
    if not np.any(candidate):
        return aligned

    ref_bounds = _nonzero_bounds(reference)
    cand_bounds = _nonzero_bounds(candidate)

    shifts: list[int] = []
    for axis in range(reference.ndim):
        ref_min, ref_max = ref_bounds[axis]
        cand_min, cand_max = cand_bounds[axis]
        ref_center = (ref_min + ref_max) / 2.0
        cand_center = (cand_min + cand_max) / 2.0
        shifts.append(int(round(ref_center - cand_center)))

    target_slices: list[slice] = []
    source_slices: list[slice] = []
    for axis, shift in enumerate(shifts):
        ref_len = reference.shape[axis]
        cand_len = candidate.shape[axis]
        ref_start = max(0, shift)
        cand_start = max(0, -shift)
        length = min(ref_len - ref_start, cand_len - cand_start)
        if length <= 0:
            return aligned
        target_slices.append(slice(ref_start, ref_start + length))
        source_slices.append(slice(cand_start, cand_start + length))

    aligned[tuple(target_slices)] = candidate[tuple(source_slices)]
    return aligned


@BatchProcessingRegistry.register(
    name="Labels to Binary",
    suffix="_binary",
    description="Convert a label image to a binary mask (255 for non-zero, 0 otherwise)",
    parameters={},
)
def labels_to_binary(image: np.ndarray) -> np.ndarray:
    arr = _to_array(image)
    result = np.zeros(arr.shape, dtype=np.uint8)
    np.copyto(result, 255, where=arr != 0)
    return result


@BatchProcessingRegistry.register(
    name="Invert Binary Labels",
    suffix="_inverted",
    description="Invert a binary label image (non-zero becomes 0, zero becomes 255)",
    parameters={},
)
def invert_binary_labels(image: np.ndarray) -> np.ndarray:
    arr = _to_array(image)
    result = np.zeros(arr.shape, dtype=np.uint8)
    np.copyto(result, 255, where=arr == 0)
    return result


@BatchProcessingRegistry.register(
    name="Filter Label by ID",
    suffix="_filtered",
    description="Keep only the specified label ID, set all other labels to background (0)",
    parameters={
        "label_id": {
            "type": int,
            "default": 1,
            "min": 1,
            "description": "Label ID to keep (all others become background)",
        }
    },
)
def filter_label_by_id(image: np.ndarray, label_id: int = 1) -> np.ndarray:
    """
    Filter a label image to keep only the specified label ID.
    All other label IDs are set to background (0).

    Parameters
    ----------
    image : np.ndarray
        Input label image
    label_id : int
        The label ID to keep (default: 1)

    Returns
    -------
    np.ndarray
        Filtered label image with only the specified label ID preserved
    """
    arr = _to_array(image)
    result = np.where(arr == label_id, arr, 0).astype(arr.dtype)
    return result


@BatchProcessingRegistry.register(
    name="Mirror Labels",
    suffix="_mirrored",
    description="Mirror labels at their largest slice area along an axis, keeping original image shape",
    parameters={
        "axis": {
            "type": int,
            "default": 0,
            "description": "Axis along which to mirror the labels",
        }
    },
)
def mirror_labels(image: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = _to_array(image)
    if arr.ndim == 0:
        raise ValueError("Cannot mirror a scalar")
    if not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    if axis >= arr.ndim or axis < -arr.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for an image with {arr.ndim} dimensions"
        )

    axis = axis % arr.ndim

    # Find the slice with the largest area (most non-zero pixels)
    reduction_axes = tuple(i for i in range(arr.ndim) if i != axis)
    slice_areas = np.sum(arr != 0, axis=reduction_axes, dtype=np.int64)

    if slice_areas.size == 0 or np.max(slice_areas) == 0:
        # No labels to mirror, return copy
        return arr.copy()

    # Find the index of the slice with maximum area
    max_area_idx = int(np.argmax(slice_areas))

    # Create result array (same shape as input)
    result = np.zeros_like(arr)

    # Get max label value for offset
    max_label = int(np.max(arr)) if arr.size else 0

    # Mirror labels from the max_area_idx position
    for i in range(arr.shape[axis]):
        # Calculate the mirrored position relative to max_area_idx
        mirrored_i = 2 * max_area_idx - i

        # Create slicers for current position i and mirrored position
        slicer_i = [slice(None)] * arr.ndim
        slicer_i[axis] = i
        slicer_i = tuple(slicer_i)

        # If mirrored position is within bounds, copy and offset the labels
        if 0 <= mirrored_i < arr.shape[axis]:
            slicer_mirrored = [slice(None)] * arr.ndim
            slicer_mirrored[axis] = mirrored_i
            slicer_mirrored = tuple(slicer_mirrored)

            # Copy mirrored slice with offset labels
            mirrored_slice = arr[slicer_mirrored]
            result[slicer_i] = np.where(
                mirrored_slice != 0, mirrored_slice + max_label, 0
            )

    return result.astype(arr.dtype, copy=False)


@BatchProcessingRegistry.register(
    name="Intersect Label Images",
    suffix="_intersected",
    description="Compute the voxel-wise intersection of paired label images identified by suffix",
    parameters={
        "primary_suffix": {
            "type": str,
            "default": "_a.tif",
            "description": "Suffix (including extension) of the primary label image",
        },
        "secondary_suffix": {
            "type": str,
            "default": "_b.tif",
            "description": "Suffix (including extension) of the paired label image",
        },
    },
)
def intersect_label_images(
    image: np.ndarray,
    primary_suffix: str = "_a.tif",
    secondary_suffix: str = "_b.tif",
) -> np.ndarray:
    """Return the intersection of two paired label images.

    The function expects two label files that share a base name but differ by suffix.
    When overlap occurs, the label IDs from the primary image are retained anywhere both
    volumes contain non-zero voxels. Each pair is processed exactly once by treating the
    file with ``primary_suffix`` as the active entry point; secondary label files are
    skipped and left unchanged.
    """

    if not primary_suffix or not secondary_suffix:
        raise ValueError(
            "Both primary_suffix and secondary_suffix must be provided"
        )

    def _load_label_file(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            return np.load(path)
        if _HAS_TIFFFILE and ext in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
            return tifffile.imread(path)

        try:
            from skimage.io import imread
        except (
            ImportError
        ) as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Install 'tifffile' or 'scikit-image' to load paired label images"
            ) from exc

        return imread(path)

    current_file = None
    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "filepath" in frame_locals:
            current_file = frame_locals["filepath"]
            break

    if current_file is None:
        raise ValueError(
            "Could not determine current file path for paired label lookup"
        )

    current_file = os.fspath(current_file)
    folder_path = os.path.dirname(current_file)
    filename = os.path.basename(current_file)

    if filename.endswith(primary_suffix):
        base_name = filename[: -len(primary_suffix)]
        paired_suffix = secondary_suffix
    elif filename.endswith(secondary_suffix):
        warnings.warn(
            (
                f"Skipping secondary label image '{filename}'; only files ending with "
                f"'{primary_suffix}' are processed by the 'Intersect Label Images' function."
            ),
            stacklevel=2,
        )
        return None
    else:
        raise ValueError(
            f"Filename '{filename}' does not end with either '{primary_suffix}' or '{secondary_suffix}'"
        )

    paired_name = base_name + paired_suffix
    paired_path = os.path.abspath(os.path.join(folder_path, paired_name))

    if not os.path.exists(paired_path):
        raise FileNotFoundError(f"Paired label image not found: {paired_path}")

    current_array = _to_array(image)
    paired_array = _to_array(_load_label_file(paired_path))

    paired_aligned = _align_candidate(current_array, paired_array)

    overlap_mask = (current_array != 0) & (paired_aligned != 0)
    if not np.any(overlap_mask):
        result_dtype = np.promote_types(
            current_array.dtype, paired_aligned.dtype
        )
        return np.zeros(current_array.shape, dtype=result_dtype)

    result_dtype = np.promote_types(current_array.dtype, paired_aligned.dtype)
    result = np.zeros(current_array.shape, dtype=result_dtype)
    np.copyto(
        result,
        current_array.astype(result_dtype, copy=False),
        where=overlap_mask,
    )

    return result


@BatchProcessingRegistry.register(
    name="Keep Slice Range by Area",
    suffix="_area_range",
    description="Zero out label content outside the min/max area slice range (preserves image shape for alignment)",
    parameters={
        "axis": {
            "type": int,
            "default": 0,
            "description": "Axis index representing the slice dimension (negative values count from the end)",
        }
    },
)
def keep_slice_range_by_area(image: np.ndarray, axis: int = 0) -> np.ndarray:
    """Keep label content only between the minimum-area and maximum-area slices (inclusive).

    The per-slice area is measured as the number of non-zero pixels in the slice. When all slices
    share the same area, the original volume is returned unchanged. This function preserves the
    original image dimensions but zeros out label content outside the detected range, ensuring
    alignment with corresponding image data is maintained.

    Parameters
    ----------
    image:
        3D (or higher dimensional) label image as a NumPy array.
    axis:
        Axis index corresponding to the slice dimension that should be evaluated.

    Returns
    -------
    numpy.ndarray
        Volume with the same shape as input, but with label content zeroed outside the
        minimum and maximum area slice range (inclusive).
    """

    if image.ndim < 3:
        raise ValueError(
            "Slice range trimming requires an array with at least 3 dimensions"
        )
    if not isinstance(axis, int):
        raise TypeError("axis must be provided as an integer")
    if axis >= image.ndim or axis < -image.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for an image with {image.ndim} dimensions"
        )

    axis = axis % image.ndim

    if image.shape[axis] == 0:
        raise ValueError(
            "Cannot determine slice range on an axis with zero length"
        )

    reduction_axes = tuple(i for i in range(image.ndim) if i != axis)
    # Count non-zero pixels per slice to determine occupied area per slice
    slice_areas = np.sum(image != 0, axis=reduction_axes, dtype=np.int64)

    if slice_areas.size == 0:
        return image.copy()

    if slice_areas.min() == slice_areas.max():
        return image.copy()

    min_idx = int(np.argmin(slice_areas))
    max_idx = int(np.argmax(slice_areas))

    start = min(min_idx, max_idx)
    end = max(min_idx, max_idx)

    # Create a copy of the full image to preserve shape
    result = image.copy()

    # Zero out slices before the start
    if start > 0:
        before_slicer = [slice(None)] * image.ndim
        before_slicer[axis] = slice(0, start)
        result[tuple(before_slicer)] = 0

    # Zero out slices after the end
    if end < image.shape[axis] - 1:
        after_slicer = [slice(None)] * image.ndim
        after_slicer[axis] = slice(end + 1, None)
        result[tuple(after_slicer)] = 0

    return result


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
    name="Split Color Channels",
    suffix="_split",
    description="Splits the color channels of the image",
    parameters={
        "num_channels": {
            "type": int,
            "default": 3,
            "min": 2,
            "max": 4,
            "description": "Number of color channels in the image",
        },
        "time_steps": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 1000,
            "description": "Number of time steps (leave 0 if not a time series)",
        },
        "output_format": {
            "type": str,
            "default": "python",
            "options": ["python", "fiji"],
            "description": "Output dimension order: python (standard) or fiji (ImageJ/Fiji compatible)",
        },
    },
)
def split_channels(
    image: np.ndarray,
    num_channels: int = 3,
    time_steps: int = 0,
    output_format: str = "python",
) -> np.ndarray:
    """
    Split the image into separate channels based on the specified number of channels.
    Can handle various dimensional orderings including time series data.

    Args:
        image: Input image array (at least 3D: XYC or higher dimensions)
        num_channels: Number of channels in the image (default: 3)
        time_steps: Number of time steps if time series (default: 0, meaning not a time series)
        output_format: Dimension order format, either "python" (standard) or "fiji" (ImageJ compatible)

    Returns:
        Stacked array of channels with shape (num_channels, ...)
    """
    # Validate input
    if image.ndim < 3:
        raise ValueError(
            "Input must be an array with at least 3 dimensions (XYC or higher)"
        )

    print(f"Image shape: {image.shape}")
    is_timelapse = time_steps > 0
    is_3d = (
        image.ndim > 3
    )  # More than 3 dimensions likely means 3D + channels or time series

    # Find channel axis based on provided channel count
    channel_axis = None
    for axis, dim_size in enumerate(image.shape):
        if dim_size == num_channels:
            # Found a dimension matching the specified channel count
            channel_axis = axis
            # If we have multiple matching dimensions, prefer the one that's not likely spatial
            if (
                axis < image.ndim - 2
            ):  # Not one of the last two dimensions (likely spatial)
                break

    # If channel axis is not found with exact match, look for other possibilities
    if channel_axis is None:
        # Try to infer channel axis using heuristics
        ndim = image.ndim

        # Check dimensions for a small value (1-16) that could be channels
        for i, dim_size in enumerate(image.shape):
            # Skip dimensions that are likely spatial (Y,X) - typically the last two
            if i >= ndim - 2:
                continue
            # Skip first dimension if this is a time series
            if is_timelapse and i == 0:
                continue
            # A dimension with size 1-16 is likely channels
            if 1 <= dim_size <= 16:
                channel_axis = i
                break

        # If still not found, check even the spatial dimensions (for RGB images)
        if channel_axis is None and image.shape[-1] <= 16:
            channel_axis = ndim - 1

    if channel_axis is None:
        raise ValueError(
            f"Could not identify a channel axis. Please check if the number of channels ({num_channels}) "
            f"matches any dimension in your image shape {image.shape}"
        )

    print(f"Channel axis identified: {channel_axis}")

    # Generate dimensional understanding for better handling
    # Create axes string to understand dimension ordering
    axes = [""] * image.ndim

    # Assign channel axis
    axes[channel_axis] = "C"

    # Assign time axis if present
    if is_timelapse and 0 not in [
        channel_axis
    ]:  # If channel is not at position 0
        axes[0] = "T"

    # Assign remaining spatial dimensions
    remaining_dims = [i for i in range(image.ndim) if axes[i] == ""]
    spatial_axes = []
    if is_3d and len(remaining_dims) > 2:
        # We have Z dimension
        spatial_axes.append("Z")

    # Add Y and X
    spatial_axes.extend(["Y", "X"])

    # Assign remaining dimensions
    for i, dim in enumerate(remaining_dims):
        if i < len(spatial_axes):
            axes[dim] = spatial_axes[i]
        else:
            axes[dim] = "A"  # Anonymous dimension

    axes_str = "".join(axes)
    print(f"Inferred dimension order: {axes_str}")

    # Split along the channel axis
    actual_channels = image.shape[channel_axis]
    if actual_channels != num_channels:
        print(
            f"Warning: Specified {num_channels} channels but found {actual_channels} in the data. Using {actual_channels}."
        )
        num_channels = actual_channels

    # Split channels
    channels = np.split(image, num_channels, axis=channel_axis)

    # Process output format
    result_channels = []
    for i, channel_img in enumerate(channels):
        # Get original axes without channel
        axes_without_channel = axes.copy()
        del axes_without_channel[channel_axis]
        axes_without_channel_str = "".join(axes_without_channel)

        # For fiji format, reorganize dimensions to TZYX order
        if output_format.lower() == "fiji":
            # Map dimensions to positions
            dim_indices = {
                dim: i for i, dim in enumerate(axes_without_channel_str)
            }

            # Build target order and transpose indices
            target_order = ""
            transpose_indices = []

            # Add T if exists
            if "T" in dim_indices:
                target_order += "T"
                transpose_indices.append(dim_indices["T"])

            # Add Z if exists
            if "Z" in dim_indices:
                target_order += "Z"
                transpose_indices.append(dim_indices["Z"])

            # Add Y and X (should always exist)
            if "Y" in dim_indices and "X" in dim_indices:
                target_order += "YX"
                transpose_indices.append(dim_indices["Y"])
                transpose_indices.append(dim_indices["X"])

            # Only transpose if order is different and we have enough dimensions
            if (
                axes_without_channel_str != target_order
                and len(transpose_indices) > 1
                and len(transpose_indices) == len(axes_without_channel)
            ):
                print(
                    f"Channel {i}: Transposing from {axes_without_channel_str} to {target_order}"
                )
                result_channels.append(
                    np.transpose(channel_img, transpose_indices)
                )
            else:
                # Keep as is
                result_channels.append(channel_img)
        else:
            # For python format, keep as is
            result_channels.append(channel_img)

    # Stack channels along a new first dimension
    return np.stack(result_channels, axis=0)


@BatchProcessingRegistry.register(
    name="Merge Color Channels",
    suffix="_merged_colors",
    description="Merges separate channel images from a folder into a single multi-channel image",
    parameters={
        "channel_substring": {
            "type": str,
            "default": "_channel_",
            "description": "Substring before channel number",
        },
    },
)
def merge_channels(
    image: np.ndarray,
    channel_substring: str = "_channel_",
) -> np.ndarray:
    """
    Merge multiple single-channel images from a folder into one multi-channel image.

    Identifies channel files by finding a substring followed by a 1-2 digit number.
    Adds channels as the last dimension regardless of input dimensionality.

    Args:
        image: Current image being processed
        channel_substring: Substring that appears before channel number in filenames

    Returns:
        Multi-channel image with channels as last dimension
    """
    # Get file context from batch processing
    import inspect
    import re

    from skimage.io import imread

    current_file = None

    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "filepath" in frame_locals:
            current_file = frame_locals["filepath"]
            break

    if current_file is None:
        raise ValueError("Could not determine current file path")

    folder_path = os.path.dirname(current_file)
    filename = os.path.basename(current_file)

    # Create regex pattern to find channel substring followed by 1-2 digits
    pattern = re.compile(rf"({re.escape(channel_substring)})(\d{{1,2}})")
    match = pattern.search(filename)

    if not match:
        print(
            f"⚠️  No channel pattern '{channel_substring}[number]' found in filename"
        )
        return image

    # Extract base name and channel number
    channel_num = int(match.group(2))
    base_name = filename[: match.start()] + filename[match.end() :]

    print(f"\n📐 Current image shape: {image.shape}")
    print(f"✅ Found channel {channel_num} in file: {filename}")

    # Find all related channel files
    all_files = os.listdir(folder_path)
    channel_files = {}

    for file in all_files:
        file_match = pattern.search(file)
        if file_match:
            # Check if base name matches (excluding channel part)
            file_base = file[: file_match.start()] + file[file_match.end() :]
            if file_base == base_name:
                ch_num = int(file_match.group(2))
                channel_files[ch_num] = os.path.join(folder_path, file)

    # Sort by channel number
    sorted_channels = sorted(channel_files.keys())
    num_channels = len(sorted_channels)

    if num_channels < 2:
        print(
            f"⚠️  Only found {num_channels} channel(s). Need at least 2 for merging."
        )
        return image

    # Determine which channel acts as the primary trigger for merging
    primary_channel = sorted_channels[0]

    if channel_num != primary_channel:
        print(
            f"ℹ️  Channel {channel_num} is not the primary channel ({primary_channel}); skipping merge for this file."
        )
        return image

    print(f"📊 Found {num_channels} channels: {sorted_channels}")

    # Load all channels in order
    # First channel is the current image
    channels = []
    for ch_num in sorted_channels:
        if ch_num == channel_num:
            # Use the already loaded image for current channel
            channels.append(image)
        else:
            # Load other channel files
            channel_path = channel_files[ch_num]
            channel_data = imread(channel_path)

            if channel_data.shape != image.shape:
                raise ValueError(
                    f"Channel {ch_num} has different shape: {channel_data.shape} vs {image.shape}"
                )

            channels.append(channel_data)

        print(
            f"   Channel {ch_num}: {os.path.basename(channel_files[ch_num])}"
        )

    # Stack channels as last dimension
    merged = np.stack(channels, axis=-1)

    print(
        f"✨ Merged shape: {merged.shape} (channels added as last dimension)"
    )

    # Return merged array so downstream steps receive the combined channels
    return merged


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


@BatchProcessingRegistry.register(
    name="Split TZYX into ZYX TIFs",
    suffix="_split",
    description="Splits a 4D TZYX image stack into separate 3D ZYX TIFs for each time point using parallel processing",
    parameters={
        "output_name_format": {
            "type": str,
            "default": "{basename}_t{timepoint:03d}",
            "description": "Format for output filenames. Use {basename} and {timepoint} as placeholders",
        },
        "preserve_scale": {
            "type": bool,
            "default": True,
            "description": "Preserve scale/resolution metadata when saving",
        },
        "use_compression": {
            "type": bool,
            "default": True,
            "description": "Apply zlib compression to output files",
        },
        "num_workers": {
            "type": int,
            "default": 4,
            "min": 1,
            "max": 16,
            "description": "Number of worker processes for parallel processing",
        },
    },
)
def split_tzyx_stack(
    image: np.ndarray,
    output_name_format: str = "{basename}_t{timepoint:03d}",
    preserve_scale: bool = True,
    use_compression: bool = True,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Split a 4D TZYX stack into separate 3D ZYX TIF files using parallel processing.

    This function takes a 4D TZYX image stack and saves each time point as
    a separate 3D ZYX TIF file. Files are processed in parallel for better performance.
    The original 4D stack is returned unchanged.

    Parameters:
    -----------
    image : numpy.ndarray
        Input 4D TZYX image stack
    output_name_format : str
        Format string for output filenames. Use {basename} and {timepoint} as placeholders.
        Default: "{basename}_t{timepoint:03d}"
    preserve_scale : bool
        Whether to preserve scale/resolution metadata when saving
    use_compression : bool
        Whether to apply zlib compression to output files
    num_workers : int
        Number of worker processes for parallel file saving

    Returns:
    --------
    numpy.ndarray
        The original image (unchanged)
    """
    # Validate input dimensions
    if image.ndim != 4:
        print(
            f"Warning: Expected 4D TZYX input, got {image.ndim}D. Returning original image."
        )
        return image

    # Use dask array to optimize memory usage when processing slices
    chunks = (1,) + image.shape[1:]  # Each timepoint is a chunk
    dask_image = da.from_array(image, chunks=chunks)

    # Store processing parameters for post-processing
    split_tzyx_stack.dask_image = dask_image
    split_tzyx_stack.output_name_format = output_name_format
    split_tzyx_stack.preserve_scale = preserve_scale
    split_tzyx_stack.use_compression = use_compression
    split_tzyx_stack.num_workers = min(
        num_workers, image.shape[0]
    )  # Limit workers to number of timepoints

    # Mark for post-processing with multiple output files
    split_tzyx_stack.requires_post_processing = True
    split_tzyx_stack.produces_multiple_files = True
    # Tell the processing system to skip creating the original output file
    split_tzyx_stack.skip_original_output = True

    # Get dimensions for informational purposes
    t_size, z_size, y_size, x_size = image.shape
    print(f"TZYX stack dimensions: {image.shape}, dtype: {image.dtype}")
    print(f"Will generate {t_size} separate ZYX files")
    print(f"Parallelization: {split_tzyx_stack.num_workers} workers")

    # The actual file saving will happen in the post-processing step
    return image


# Monkey patch ProcessingWorker.process_file to handle parallel TZYX splitting
try:
    # Import tifffile here to ensure it's available for the monkey patch
    if not _HAS_TIFFFILE:
        try:
            import tifffile

            _HAS_TIFFFILE = True
        except ImportError:
            pass  # tifffile not available, skip monkey patch

    from napari_tmidas._file_selector import ProcessingWorker

    # Define function to save a single timepoint
    def save_timepoint(
        t: int,
        data: np.ndarray,
        output_filepath: str,
        resolution=None,
        use_compression=True,
    ) -> str:
        """
        Save a single timepoint to disk.

        Parameters:
        -----------
        t : int
            Timepoint index for logging
        data : np.ndarray
            3D ZYX data to save
        output_filepath : str
            Path to save the file
        resolution : tuple, optional
            Resolution metadata to preserve
        use_compression : bool
            Whether to use compression

        Returns:
        --------
        str
            Path to the saved file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            # Determine the appropriate compression parameter
            # Note: tifffile uses 'compression', not 'compress'
            compression_arg = "zlib" if use_compression else None

            # Calculate approximate file size for BigTIFF decision
            size_gb = (data.size * data.itemsize) / (1024**3)
            use_bigtiff = size_gb > 4.0

            # Save the file with proper parameters
            tifffile.imwrite(
                output_filepath,
                data,
                resolution=resolution,
                compression=compression_arg,
                bigtiff=use_bigtiff,
            )

            print(f"✓ Saved timepoint {t} to {output_filepath}")
            return output_filepath
        except Exception as e:
            print(f"✘ Error saving timepoint {t}: {str(e)}")
            traceback.print_exc()
            raise

    # Store the original process_file function
    original_process_file = ProcessingWorker.process_file

    # Define the custom process_file function
    def process_file_with_tzyx_splitting(self, filepath):
        """Modified process_file function that handles parallel TZYX splitting."""
        # First call the original function to get the initial result
        result = original_process_file(self, filepath)

        # Skip further processing if there's no result or no processed_file
        if not isinstance(result, dict) or "processed_file" not in result:
            return result

        # Get the output path from the original processing
        output_path = result["processed_file"]
        processing_func = self.processing_func

        # Check if our function has the required attributes for TZYX splitting
        if (
            hasattr(processing_func, "requires_post_processing")
            and processing_func.requires_post_processing
            and hasattr(processing_func, "dask_image")
            and hasattr(processing_func, "produces_multiple_files")
            and processing_func.produces_multiple_files
        ):
            try:
                # Get the Dask image and processing parameters
                dask_image = processing_func.dask_image
                output_name_format = processing_func.output_name_format
                preserve_scale = processing_func.preserve_scale
                use_compression = processing_func.use_compression
                num_workers = processing_func.num_workers

                # Extract base filename without extension
                basename = os.path.splitext(os.path.basename(output_path))[0]
                dirname = os.path.dirname(output_path)

                # Try to get scale info from original file if needed
                resolution = None
                if preserve_scale:
                    try:
                        with tifffile.TiffFile(filepath) as tif:
                            if hasattr(tif, "pages") and tif.pages:
                                page = tif.pages[0]
                                if hasattr(page, "resolution"):
                                    resolution = page.resolution
                    except (OSError, AttributeError, KeyError) as e:

                        print(
                            f"Warning: Could not read original resolution: {e}"
                        )

                # Get number of timepoints
                t_size = dask_image.shape[0]
                print(f"Processing {t_size} timepoints in parallel...")

                # Prepare output paths for each timepoint
                output_filepaths = []
                for t in range(t_size):
                    # Format the output filename
                    output_filename = output_name_format.format(
                        basename=basename, timepoint=t
                    )
                    # Add extension
                    output_filepath = os.path.join(
                        dirname, f"{output_filename}.tif"
                    )
                    output_filepaths.append(output_filepath)

                # Process timepoints in parallel
                processed_files = []

                # Use ThreadPoolExecutor for parallel file saving
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    # Submit tasks for each timepoint
                    future_to_timepoint = {}
                    for t in range(t_size):
                        # Extract this timepoint's data using Dask
                        timepoint_array = dask_image[t].compute()

                        # Submit the task to save this timepoint
                        future = executor.submit(
                            save_timepoint,
                            t,
                            timepoint_array,
                            output_filepaths[t],
                            resolution,
                            use_compression,
                        )
                        future_to_timepoint[future] = t

                    total = len(future_to_timepoint)
                    for completed, future in enumerate(
                        concurrent.futures.as_completed(future_to_timepoint),
                        start=1,
                    ):
                        t = future_to_timepoint[future]
                        try:
                            output_filepath = future.result()
                            processed_files.append(output_filepath)
                        except (OSError, concurrent.futures.TimeoutError) as e:
                            print(f"Failed to save timepoint {t}: {e}")

                        # Update progress
                        if completed % 5 == 0 or completed == total:
                            percent = int(completed * 100 / total)
                            print(
                                f"Progress: {completed}/{total} timepoints ({percent}%)"
                            )

                # Update the result with the list of processed files
                if processed_files:
                    print(
                        f"Successfully generated {len(processed_files)} ZYX files from TZYX stack"
                    )
                    result["processed_files"] = processed_files

                    # Skip creating the original consolidated output file if requested
                    if (
                        hasattr(processing_func, "skip_original_output")
                        and processing_func.skip_original_output
                    ):
                        # Remove the original file if it was already created
                        if os.path.exists(output_path):
                            try:
                                os.remove(output_path)
                                print(
                                    f"Removed unnecessary consolidated file: {output_path}"
                                )
                            except OSError as e:
                                print(
                                    f"Warning: Could not remove consolidated file: {e}"
                                )

                        # Remove the entry from the result to prevent its display
                        if "processed_file" in result:
                            del result["processed_file"]

                else:
                    print("Warning: No ZYX files were successfully generated")

            except (OSError, ValueError, RuntimeError) as e:

                traceback.print_exc()
                print(f"Error in TZYX splitting post-processing: {e}")

        return result

    # Apply the monkey patch
    ProcessingWorker.process_file = process_file_with_tzyx_splitting

except (NameError, AttributeError) as e:
    print(f"Warning: Could not apply TZYX splitting patch: {e}")

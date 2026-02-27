# processing_functions/label_based_cropping.py
"""
Label-based image cropping pipeline.

This module provides functionality to crop images based on user-drawn labels.
The pipeline supports:
- 2D drawing in a label image
- Automatic repetition across z-slices for 3D images
- Automatic repetition in each frame for time-series images
- Masking of everything outside the drawn label
"""

import concurrent.futures
import inspect
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from napari_tmidas._registry import BatchProcessingRegistry

# Lazy imports for optional heavy dependencies
try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False


def _get_label_image_filename(intensity_filename: str) -> str:
    """
    Derive the label image filename from the intensity image filename.

    This function looks for common patterns and tries to find a corresponding
    label file in the same directory.

    Parameters
    ----------
    intensity_filename : str
        Path to the intensity image file

    Returns
    -------
    str
        Path to the label image file, or None if not found
    """
    intensity_path = Path(intensity_filename)
    parent_dir = intensity_path.parent

    # Common label suffixes
    label_suffixes = [
        "_labels.tif",
        "_labels.tiff",
        "_labels_filtered.tif",
        "_labels_filtered.tiff",
        "_convpaint_labels.tif",
        "_convpaint_labels.tiff",
        "_convpaint_labels_filtered.tif",
        "_convpaint_labels_filtered.tiff",
        "_seg.tif",
        "_seg.tiff",
        "_mask.tif",
        "_mask.tiff",
    ]

    stem = intensity_path.stem
    for suffix in label_suffixes:
        label_path = parent_dir / (stem + suffix)
        if label_path.exists():
            return str(label_path)

    return None


def _load_image(filepath: str) -> tuple:
    """
    Load image from file and return image data with metadata.

    Parameters
    ----------
    filepath : str
        Path to the image file

    Returns
    -------
    tuple
        (image_data, axes_string) where axes_string describes the dimension order
    """
    if not _HAS_TIFFFILE:
        raise ImportError("tifffile is required for image loading")

    with tifffile.TiffFile(filepath) as tif:
        # Try to get image from series first
        if len(tif.series) > 0:
            series = tif.series[0]
            image = series.asarray()
            axes = series.axes
        else:
            # Fallback to reading all pages
            image = tif.asarray()
            axes = ""

            # Try to infer axes from metadata
            if hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
                metadata = tif.imagej_metadata
                if "frames" in metadata:
                    # Has time dimension
                    if image.ndim >= 3:
                        axes = "TYX" if image.ndim == 3 else "TZYX"

        return image, axes


def _expand_label_to_3d(label_2d: np.ndarray, z_size: int) -> np.ndarray:
    """
    Expand a 2D label to 3D by repeating across z-slices.

    Parameters
    ----------
    label_2d : np.ndarray
        2D label image (Y, X)
    z_size : int
        Number of z-slices to create

    Returns
    -------
    np.ndarray
        3D label image (Z, Y, X)
    """
    return np.repeat(label_2d[np.newaxis, :, :], z_size, axis=0)


def _expand_label_to_time(
    label_2d: np.ndarray, t_size: int, z_size: Optional[int] = None
) -> np.ndarray:
    """
    Expand a 2D label to time series, optionally with z-dimension.

    Parameters
    ----------
    label_2d : np.ndarray
        2D label image (Y, X)
    t_size : int
        Number of time frames
    z_size : int, optional
        Number of z-slices. If provided, creates (T, Z, Y, X), else (T, Y, X)

    Returns
    -------
    np.ndarray
        Time series label image
    """
    if z_size is None:
        # Expand to (T, Y, X)
        return np.repeat(label_2d[np.newaxis, :, :], t_size, axis=0)
    else:
        # Expand to (T, Z, Y, X)
        label_3d = np.repeat(label_2d[np.newaxis, :, :], z_size, axis=0)
        return np.repeat(label_3d[np.newaxis, :, :, :], t_size, axis=0)


def _crop_image_with_label(
    image: np.ndarray,
    label: np.ndarray,
    image_axes: str = "",
    label_axes: str = "",
) -> np.ndarray:
    """
    Crop and mask an image based on a label image.

    Everything outside the label mask is set to 0 (or the minimum value for the dtype).

    Parameters
    ----------
    image : np.ndarray
        Image data to crop and mask
    label : np.ndarray
        Label image where non-zero values define the region to keep
    image_axes : str, optional
        Axes string for the image (e.g., "ZYX", "TZYX")
    label_axes : str, optional
        Axes string for the label (e.g., "ZYX", "TZYX")

    Returns
    -------
    np.ndarray
        Masked image with the same shape as the input image
    """
    # Ensure label has same shape as image
    if image.shape != label.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match label shape {label.shape}"
        )

    # Create output image
    cropped = image.copy()

    # Apply mask: set values to 0 where label is 0
    mask = label > 0
    cropped[~mask] = 0

    return cropped


def _infer_axes(image: np.ndarray, metadata: Optional[dict] = None) -> str:
    """
    Infer axis order from image shape and metadata.

    Parameters
    ----------
    image : np.ndarray
        Image data
    metadata : dict, optional
        Image metadata that might contain axis information

    Returns
    -------
    str
        Axes string (e.g., "YX", "ZYX", "TYX", "TZYX")
    """
    if metadata is None:
        metadata = {}

    ndim = image.ndim

    if ndim == 2:
        return "YX"
    elif ndim == 3:
        # Could be ZYX or TYX - check metadata
        if "frames" in metadata:
            return "TYX"
        return "ZYX"
    elif ndim == 4:
        return "TZYX"
    else:
        return "".join(["D" + str(i) for i in range(ndim)])


@BatchProcessingRegistry.register(
    name="Label-Based Cropping",
    suffix="_cropped",
    description="Crop images using user-drawn labels. Optional expansion across Z and/or time.",
    parameters={
        "label_image_path": {
            "type": str,
            "default": "",
            "description": "Path to the label image file. If empty, auto-detects based on intensity image name.",
        },
        "expand_z": {
            "type": bool,
            "default": False,
            "description": "If True, expand 2D labels across Z for 3D/4D images and per-frame labels across Z.",
        },
        "expand_time": {
            "type": bool,
            "default": False,
            "description": "If True, expand a single 2D label across all time frames (T) for 4D images.",
        },
        "output_format": {
            "type": str,
            "default": "same",
            "description": "Output format: 'same' (keep original), 'tif' (save as TIFF), or 'npy' (save as NumPy)",
        },
    },
)
def label_based_cropping(
    image_array: np.ndarray,
    label_image_path: str = "",
    expand_z: bool = False,
    expand_time: bool = False,
    output_format: str = "same",
    metadata: Optional[dict] = None,
) -> np.ndarray:
    """
    Crop and mask an image based on a label image.

    This function supports:
    - Optional 2D label expansion across z-slices for 3D images (expand_z)
    - Optional 2D label expansion across time frames for 4D images (expand_time)
    - Optional expansion of per-frame labels across z-slices for 4D images (expand_z)
    - Masking of everything outside the label region

    Parameters
    ----------
    image_array : np.ndarray
        The intensity image to crop
    label_image_path : str, optional
        Path to the label image. If empty, attempts to auto-detect.
    expand_z : bool, optional
        If True, expand 2D labels across Z and per-frame labels across Z.
    expand_time : bool, optional
        If True, expand a single 2D label across all time frames for 4D images.
    output_format : str, optional
        Output format specification (for future use)
    metadata : dict, optional
        Additional metadata about the image

    Returns
    -------
    np.ndarray
        Cropped and masked image array with same shape as input
    """
    if metadata is None:
        metadata = {}

    # Determine label image path
    if not label_image_path:
        # Auto-detect would happen at a higher level
        raise ValueError(
            "Label image path must be provided or auto-detected by the calling function"
        )

    # Check if label file exists
    if not os.path.exists(label_image_path):
        raise FileNotFoundError(f"Label image not found: {label_image_path}")

    # Load label image
    label_array, label_axes = _load_image(label_image_path)

    # Get image axes
    image_axes = _infer_axes(image_array, metadata)

    # Handle dimension mismatch
    if image_array.ndim != label_array.ndim:
        # Label is 2D
        if label_array.ndim == 2 and image_array.ndim == 3:
            # Image is (Z, Y, X)
            if not expand_z:
                raise ValueError(
                    "2D label provided for 3D image. Enable expand_z to repeat across Z."
                )
            z_size = image_array.shape[0]
            label_array = _expand_label_to_3d(label_array, z_size)

        elif label_array.ndim == 2 and image_array.ndim == 4:
            # Image is (T, Z, Y, X)
            if not expand_time:
                raise ValueError(
                    "2D label provided for 4D image. Provide per-frame labels (T, Y, X) or enable expand_time."
                )
            t_size, z_size = image_array.shape[0], image_array.shape[1]
            if not expand_z:
                raise ValueError(
                    "2D label expansion across time requires expand_z for 4D images. Enable expand_z."
                )
            label_array = _expand_label_to_time(label_array, t_size, z_size)

        elif label_array.ndim == 3 and image_array.ndim == 4:
            # Expect per-frame labels (T, Y, X)
            if not image_axes.startswith("T"):
                raise ValueError(
                    f"Cannot match label shape {label_array.shape} to image shape {image_array.shape}"
                )
            t_size = image_array.shape[0]
            z_size = image_array.shape[1]
            if label_array.shape[0] != t_size:
                raise ValueError(
                    f"Label shape {label_array.shape} must match expected (T, Y, X) for image shape {image_array.shape}"
                )
            if not expand_z:
                raise ValueError(
                    "Per-frame labels provided for 4D image. Enable expand_z to apply labels across Z."
                )
            label_array = np.repeat(
                label_array[:, np.newaxis, :, :], z_size, axis=1
            )

    # Check that spatial dimensions match
    if (
        image_array.shape[-2:] != label_array.shape[-2:]
    ):  # Last 2 dims are Y, X
        raise ValueError(
            f"Label spatial dimensions {label_array.shape[-2:]} don't match image {image_array.shape[-2:]}"
        )

    # Crop and mask
    cropped = _crop_image_with_label(
        image_array, label_array, image_axes, label_axes
    )

    return cropped


def batch_label_based_cropping(
    input_folder: str,
    output_folder: str,
    auto_detect_labels: bool = True,
    label_suffix_priority: Optional[List[str]] = None,
    expand_z: bool = True,
    expand_time: bool = False,
    output_format: str = "same",
    num_workers: int = max(1, (os.cpu_count() or 4) // 4),
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Batch process images with label-based cropping.

    Parameters
    ----------
    input_folder : str
        Folder containing intensity images
    output_folder : str
        Folder where cropped images will be saved
    auto_detect_labels : bool, optional
        If True, automatically detect label images based on filename patterns
    label_suffix_priority : list of str, optional
        Priority order for label suffixes when auto-detecting
    expand_z : bool, optional
        If True, expand 2D labels across Z and per-frame labels across Z.
    expand_time : bool, optional
        If True, expand a single 2D label across all time frames for 4D images.
    output_format : str, optional
        Output format: 'same', 'tif', or 'npy'
    num_workers : int, optional
        Number of parallel workers for batch processing
    verbose : bool, optional
        Print progress information

    Returns
    -------
    dict
        Dictionary with processing results:
        {
            'successful': [list of processed file paths],
            'failed': [list of files that failed with reasons],
            'skipped': [list of files that were skipped]
        }
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if label_suffix_priority is None:
        label_suffix_priority = [
            "_labels.tif",
            "_labels_filtered.tif",
            "_seg.tif",
            "_mask.tif",
        ]

    # Find all image files
    image_extensions = [".tif", ".tiff", ".png", ".jpg", ".npy"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if verbose:
        print(f"\nFound {len(image_files)} image files in {input_folder}")

    results = {"successful": [], "failed": [], "skipped": []}

    def process_single_file(image_path: Path) -> tuple:
        """Process a single image file."""
        try:
            # Find corresponding label file
            label_path = _get_label_image_filename(str(image_path))

            if label_path is None:
                if verbose:
                    print(f"Skipped {image_path.name}: No label file found")
                return ("skipped", str(image_path), None)

            # Load images
            image, _ = _load_image(str(image_path))
            label, _ = _load_image(label_path)

            # Process
            cropped = label_based_cropping(
                image,
                label_image_path=label_path,
                expand_z=expand_z,
                expand_time=expand_time,
                output_format=output_format,
            )

            # Save output
            output_file = output_path / image_path.name.replace(
                image_path.suffix, "_cropped" + image_path.suffix
            )

            if _HAS_TIFFFILE and output_format in ["same", "tif"]:
                tifffile.imwrite(str(output_file), cropped)
            elif output_format == "npy" or image_path.suffix == ".npy":
                np.save(str(output_file).replace(image_path.suffix, ".npy"), cropped)
            else:
                # Fallback: try tifffile or numpy
                try:
                    if _HAS_TIFFFILE:
                        tifffile.imwrite(str(output_file), cropped)
                    else:
                        np.save(str(output_file).replace(image_path.suffix, ".npy"), cropped)
                except Exception:
                    np.save(str(output_file).replace(image_path.suffix, ".npy"), cropped)

            if verbose:
                print(f"Processed {image_path.name} -> {output_file.name}")

            return ("successful", str(image_path), str(output_file))

        except Exception as e:
            error_msg = f"{str(e)}"
            if verbose:
                print(f"Failed {image_path.name}: {error_msg}")
            return ("failed", str(image_path), error_msg)

    # Process files
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_file, img_path) for img_path in image_files
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing images",
            disable=not verbose,
        ):
            try:
                status, file_path, result = future.result()
                if status == "successful":
                    results["successful"].append(result)
                elif status == "failed":
                    results["failed"].append(f"{file_path}: {result}")
                else:  # skipped
                    results["skipped"].append(file_path)
            except Exception as e:
                results["failed"].append(f"Worker error: {str(e)}")
                if verbose:
                    print(f"Worker error: {e}")
                    traceback.print_exc()

    if verbose:
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")
        print(f"{'='*60}\n")

    return results

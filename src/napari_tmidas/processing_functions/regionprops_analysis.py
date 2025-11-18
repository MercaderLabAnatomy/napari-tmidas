# processing_functions/regionprops_analysis.py
# ruff: noqa: SIM105, BLE001
"""
Processing function for calculating region properties of label images.

This module provides functionality to extract region properties (regionprops) from
label images in a folder and save them to a single CSV file. The function is
dimension-agnostic and treats dimensions like T (time) or C (channel) as grouping
variables, adding corresponding columns to the output.
"""

import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage import measure

from napari_tmidas._registry import BatchProcessingRegistry

# Lazy import for pandas (optional dependency)
try:  # noqa: SIM105
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

# Global dict to track CSV files created per folder (for header management)
_REGIONPROPS_CSV_FILES = {}


def reset_regionprops_cache():
    """
    Reset the cache of CSV files.

    Call this function if you want to reprocess folders that were already
    processed in the current session.
    """
    global _REGIONPROPS_CSV_FILES
    _REGIONPROPS_CSV_FILES.clear()
    print("🔄 Regionprops analysis cache cleared")


def get_current_filepath() -> Optional[str]:
    """
    Extract the current file path from the call stack.

    Returns:
        str or None: The filepath being processed, or None if not found
    """
    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "filepath" in frame_locals:
            return frame_locals["filepath"]
    return None


def load_label_image(filepath: str) -> np.ndarray:
    """
    Load a label image from file.

    Parameters:
    -----------
    filepath : str
        Path to the label image file

    Returns:
    --------
    np.ndarray
        Loaded label image
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npy":
        return np.load(filepath)
    elif ext in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
        try:  # noqa: SIM105
            import tifffile

            return tifffile.imread(filepath)
        except ImportError:
            from skimage.io import imread

            return imread(filepath)
    else:
        from skimage.io import imread

        return imread(filepath)


def find_label_images(
    folder_path: str,
    extensions: List[str] = None,
    intensity_suffix: str = None,
) -> List[str]:
    """
    Find all label image files in a folder.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing label images
    extensions : List[str], optional
        List of file extensions to look for
    intensity_suffix : str, optional
        If provided, only return files that contain this suffix in their name
        This prevents finding both label and intensity images when they're in the same folder

    Returns:
    --------
    List[str]
        Sorted list of label image file paths
    """
    if extensions is None:
        extensions = [".tif", ".tiff", ".npy", ".png"]

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    # Find all label image files
    label_files = []
    for ext in extensions:
        label_files.extend(folder.glob(f"*{ext}"))

    # Filter to only label images if intensity_suffix provided
    if intensity_suffix:
        label_files = [f for f in label_files if intensity_suffix in f.name]

    if not label_files:
        raise ValueError(
            f"No label image files found in folder: {folder_path}"
        )

    # Sort files by name
    label_files.sort(key=lambda x: x.name)

    return [str(f) for f in label_files]


def parse_dimensions_from_shape(
    shape: Tuple[int, ...], ndim: int
) -> Dict[str, int]:
    """
    Parse dimension information from image shape.

    For images with more than 3 dimensions, tries to infer which dimensions
    correspond to T (time), C (channel), Z (depth), Y (height), X (width).

    Parameters:
    -----------
    shape : Tuple[int, ...]
        Shape of the image array
    ndim : int
        Number of dimensions

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping dimension names to their sizes
    """
    dim_info = {}

    if ndim == 2:
        # YX
        dim_info["Y"] = shape[0]
        dim_info["X"] = shape[1]
    elif ndim == 3:
        # Assume ZYX (could also be TYX or CYX)
        dim_info["Z"] = shape[0]
        dim_info["Y"] = shape[1]
        dim_info["X"] = shape[2]
    elif ndim == 4:
        # Assume TZYX or CZYX
        dim_info["T"] = shape[0]
        dim_info["Z"] = shape[1]
        dim_info["Y"] = shape[2]
        dim_info["X"] = shape[3]
    elif ndim == 5:
        # Assume TCZYX
        dim_info["T"] = shape[0]
        dim_info["C"] = shape[1]
        dim_info["Z"] = shape[2]
        dim_info["Y"] = shape[3]
        dim_info["X"] = shape[4]
    else:
        # For other dimensions, just number them
        for i, size in enumerate(shape):
            dim_info[f"dim_{i}"] = size

    return dim_info


def extract_regionprops_recursive(
    image: np.ndarray,
    intensity_image: np.ndarray = None,
    prefix_dims: Dict[str, int] = None,
    current_dim: int = 0,
    max_spatial_dims: int = 3,
    dimension_order: str = "Auto",
    properties: List[str] = None,
) -> List[Dict]:
    """
    Recursively extract regionprops from a multi-dimensional label image.

    This function handles images with arbitrary dimensions by recursively
    processing each slice along non-spatial dimensions.

    Parameters:
    -----------
    image : np.ndarray
        Label image array
    intensity_image : np.ndarray, optional
        Intensity image for measuring mean/max/min intensity values
    prefix_dims : Dict[str, int], optional
        Dictionary of dimension indices processed so far (for grouping)
    current_dim : int
        Current dimension being processed
    max_spatial_dims : int
        Maximum number of spatial dimensions to process as a single unit
    dimension_order : str
        Dimension order string (e.g., "TZYX", "CZYX", "Auto")
    properties : List[str], optional
        List of properties to extract

    Returns:
    --------
    List[Dict]
        List of dictionaries containing regionprops for each label
    """
    if prefix_dims is None:
        prefix_dims = {}

    if properties is None:
        properties = [
            "label",
            "area",
            "centroid",
            "bbox",
            "mean_intensity",
            "median_intensity",
            "std_intensity",
            "max_intensity",
            "min_intensity",
        ]

    results = []

    # Determine if we should process this as spatial data
    ndim = image.ndim

    # If we have 2 or 3 dimensions left, treat as spatial and extract regionprops
    if ndim <= max_spatial_dims:
        # Skip empty images (huge performance boost)
        if image.max() == 0:
            return results

        # Extract regionprops for this spatial slice
        # Use cache=False for better memory efficiency with large datasets
        try:  # noqa: SIM105
            # Pass intensity image if provided
            regions = measure.regionprops(
                image.astype(int), intensity_image=intensity_image, cache=False
            )

            for region in regions:
                props = prefix_dims.copy()

                # Always include label and area (renamed to 'size' for output)
                props["label"] = int(region.label)
                if "area" in properties:
                    props["size"] = int(region.area)

                # Add centroid coordinates if requested
                if "centroid" in properties:
                    centroid = region.centroid
                    if ndim == 2:
                        props["centroid_y"] = float(centroid[0])
                        props["centroid_x"] = float(centroid[1])
                    elif ndim == 3:
                        props["centroid_z"] = float(centroid[0])
                        props["centroid_y"] = float(centroid[1])
                        props["centroid_x"] = float(centroid[2])

                # Add bounding box if requested
                if "bbox" in properties:
                    bbox = region.bbox
                    if ndim == 2:
                        props["bbox_min_y"] = int(bbox[0])
                        props["bbox_min_x"] = int(bbox[1])
                        props["bbox_max_y"] = int(bbox[2])
                        props["bbox_max_x"] = int(bbox[3])
                    elif ndim == 3:
                        props["bbox_min_z"] = int(bbox[0])
                        props["bbox_min_y"] = int(bbox[1])
                        props["bbox_min_x"] = int(bbox[2])
                        props["bbox_max_z"] = int(bbox[3])
                        props["bbox_max_y"] = int(bbox[4])
                        props["bbox_max_x"] = int(bbox[5])

                # Add other properties if requested (only for 2D, as some aren't available for 3D)
                if ndim == 2:
                    if "perimeter" in properties:
                        try:  # noqa: SIM105
                            props["perimeter"] = float(region.perimeter)
                        except (
                            NotImplementedError,
                            AttributeError,
                        ):
                            pass

                    if "eccentricity" in properties:
                        try:  # noqa: SIM105
                            props["eccentricity"] = float(region.eccentricity)
                        except (
                            NotImplementedError,
                            AttributeError,
                        ):
                            pass

                    if "solidity" in properties:
                        try:  # noqa: SIM105
                            props["solidity"] = float(region.solidity)
                        except (
                            NotImplementedError,
                            AttributeError,
                        ):
                            pass

                    if "major_axis_length" in properties:
                        try:  # noqa: SIM105
                            props["major_axis_length"] = float(
                                region.major_axis_length
                            )
                        except (
                            NotImplementedError,
                            AttributeError,
                        ):
                            pass

                    if "minor_axis_length" in properties:
                        try:  # noqa: SIM105
                            props["minor_axis_length"] = float(
                                region.minor_axis_length
                            )
                        except (
                            NotImplementedError,
                            AttributeError,
                        ):
                            pass

                    if "orientation" in properties:
                        try:  # noqa: SIM105
                            props["orientation"] = float(region.orientation)
                        except (
                            NotImplementedError,
                            AttributeError,
                        ):
                            pass

                # Add extent if requested (available for both 2D and 3D)
                if "extent" in properties:
                    try:  # noqa: SIM105
                        props["extent"] = float(region.extent)
                    except (
                        NotImplementedError,
                        AttributeError,
                    ):
                        pass

                # Add intensity measurements if intensity image was provided and requested
                if intensity_image is not None:
                    if "mean_intensity" in properties:
                        try:  # noqa: SIM105
                            props["mean_intensity"] = float(
                                region.mean_intensity
                            )
                        except (NotImplementedError, AttributeError) as e:
                            print(f"⚠️  Could not extract mean_intensity: {e}")

                    if "median_intensity" in properties:
                        try:  # noqa: SIM105
                            # Median intensity requires accessing the intensity values
                            props["median_intensity"] = float(
                                np.median(region.intensity_image[region.image])
                            )
                        except (
                            NotImplementedError,
                            AttributeError,
                            Exception,
                        ) as e:
                            print(
                                f"⚠️  Could not extract median_intensity: {e}"
                            )

                    if "std_intensity" in properties:
                        try:  # noqa: SIM105
                            # Standard deviation of intensity
                            props["std_intensity"] = float(
                                np.std(region.intensity_image[region.image])
                            )
                        except (
                            NotImplementedError,
                            AttributeError,
                            Exception,
                        ) as e:
                            print(f"⚠️  Could not extract std_intensity: {e}")

                    if "max_intensity" in properties:
                        try:  # noqa: SIM105
                            props["max_intensity"] = float(
                                region.max_intensity
                            )
                        except (NotImplementedError, AttributeError) as e:
                            print(f"⚠️  Could not extract max_intensity: {e}")

                    if "min_intensity" in properties:
                        try:  # noqa: SIM105
                            props["min_intensity"] = float(
                                region.min_intensity
                            )
                        except (NotImplementedError, AttributeError) as e:
                            print(f"⚠️  Could not extract min_intensity: {e}")

                results.append(props)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Error extracting regionprops: {e}")

    else:
        # Recurse along the first dimension
        dim_name = None

        # Use dimension_order to determine dimension names
        if dimension_order != "Auto":
            # Parse the dimension order string to identify T, C, Z dimensions
            # The dimension_order describes the FULL shape of the original image
            # We need to map current_dim to the correct position in that string
            dim_order_upper = dimension_order.upper()

            # The current_dim tells us which dimension index we're at
            # Simply use it to index into the dimension_order string
            if current_dim < len(dim_order_upper):
                dim_char = dim_order_upper[current_dim]
                # Use the actual dimension character (T, C, Z, Y, X)
                if dim_char in "TCZ":
                    dim_name = dim_char
                else:
                    # Y or X - shouldn't reach here as these are spatial
                    dim_name = f"dim_{current_dim}"
            else:
                dim_name = f"dim_{current_dim}"
        else:
            # Auto mode: Try to infer dimension name based on position
            # Assume common conventions: TYX, ZYX, TZYX, CZYX, TCZYX, CYX
            total_dims = current_dim + ndim
            if total_dims == 3:
                # 3D: likely TYX, ZYX, or CYX - assume T at position 0
                # (T is most common for timelapse data)
                dim_name = "T" if current_dim == 0 else f"dim_{current_dim}"
            elif total_dims == 4:
                # 4D: likely TZYX or CZYX - assume T at position 0
                dim_name = "T" if current_dim == 0 else f"dim_{current_dim}"
            elif total_dims == 5:
                # 5D: likely TCZYX
                if current_dim == 0:
                    dim_name = "T"
                elif current_dim == 1:
                    dim_name = "C"
                else:
                    dim_name = f"dim_{current_dim}"
            else:
                dim_name = f"dim_{current_dim}"

        # Process each slice along this dimension
        for idx in range(image.shape[0]):
            slice_dims = prefix_dims.copy()
            slice_dims[dim_name] = idx

            slice_data = image[idx]
            slice_intensity = (
                intensity_image[idx] if intensity_image is not None else None
            )

            slice_results = extract_regionprops_recursive(
                slice_data,
                intensity_image=slice_intensity,
                prefix_dims=slice_dims,
                current_dim=current_dim + 1,
                max_spatial_dims=max_spatial_dims,
                dimension_order=dimension_order,
                properties=properties,
            )
            results.extend(slice_results)

    return results


def analyze_folder_regionprops(
    folder_path: str,
    output_csv: str,
    max_spatial_dims: int = 3,
    dimension_order: str = "Auto",
    properties: List[str] = None,
    intensity_suffix: str = None,
):
    """
    Analyze all label images in a folder and save regionprops to CSV.

    Parameters:
    -----------
    folder_path : str
        Path to folder containing label images
    output_csv : str
        Path to output CSV file
    max_spatial_dims : int
        Maximum number of spatial dimensions (2 or 3)
    dimension_order : str
        Dimension order string (e.g., "TZYX", "CZYX", "Auto")
    properties : List[str], optional
        List of properties to extract
    intensity_suffix : str, optional
        Suffix to replace in label filename to find matching intensity image.
        E.g., if label is "image_semantic_otsu.tif" and intensity is "image.tif",
        use intensity_suffix="_semantic_otsu.tif" (replaces with ".tif")

    Returns:
    --------
    DataFrame
        DataFrame containing all regionprops
    """
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas is required for regionprops analysis. "
            "Install it with: pip install pandas"
        )

    print(f"🔍 Analyzing label images in: {folder_path}")
    if dimension_order != "Auto":
        print(f"   Using dimension order: {dimension_order}")

    # Find all label image files
    # Pass intensity_suffix to filter only label images (not intensity images)
    label_files = find_label_images(
        folder_path, intensity_suffix=intensity_suffix
    )
    print(f"Found {len(label_files)} label image files")

    all_results = []

    for file_idx, filepath in enumerate(label_files):
        filename = os.path.basename(filepath)
        print(
            f"Processing {file_idx + 1}/{len(label_files)}: {filename}",
            end="",
            flush=True,
        )

        try:  # noqa: SIM105
            # Load label image
            label_image = load_label_image(filepath)

            # Skip completely empty images
            if label_image.max() == 0:
                print(" - empty, skipped")
                continue

            # Load intensity image if suffix provided
            intensity_image = None
            if intensity_suffix:
                # Find matching intensity image by replacing suffix
                label_path = Path(filepath)
                label_filename = label_path.name

                if intensity_suffix in label_filename:
                    # Replace the suffix (e.g., "_semantic_otsu.tif" -> ".tif")
                    intensity_filename = label_filename.replace(
                        intensity_suffix, ".tif"
                    )
                    intensity_path = label_path.parent / intensity_filename

                    if intensity_path.exists():
                        try:  # noqa: SIM105
                            intensity_image = load_label_image(
                                str(intensity_path)
                            )
                            # Verify shapes match
                            if intensity_image.shape != label_image.shape:
                                print(
                                    f" - WARNING: intensity image shape {intensity_image.shape} != label shape {label_image.shape}, skipping intensity"
                                )
                                intensity_image = None
                        except Exception as e:  # noqa: BLE001
                            print(
                                f" - WARNING: could not load intensity image: {e}"
                            )
                    else:
                        print(
                            f" - WARNING: intensity image not found: {intensity_path.name}"
                        )

            # Extract regionprops recursively
            import time

            start_time = time.time()
            file_results = extract_regionprops_recursive(
                label_image,
                intensity_image=intensity_image,
                prefix_dims={"filename": filename},
                current_dim=0,
                max_spatial_dims=max_spatial_dims,
                dimension_order=dimension_order,
                properties=properties,
            )
            elapsed = time.time() - start_time

            all_results.extend(file_results)
            print(f" - {len(file_results)} regions in {elapsed:.2f}s")

        except Exception as e:  # noqa: BLE001
            print(f"\n  Error processing {filename}: {e}")
            import traceback

            traceback.print_exc()

    # Convert to DataFrame
    if all_results:
        df = pd.DataFrame(all_results)

        # Reorder columns to put identifiers first
        id_cols = ["filename"]
        if "T" in df.columns:
            id_cols.append("T")
        if "C" in df.columns:
            id_cols.append("C")
        if "Z" in df.columns:
            id_cols.append("Z")

        id_cols.append("label")

        # Get remaining columns
        other_cols = [col for col in df.columns if col not in id_cols]

        # Reorder
        df = df[id_cols + other_cols]

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved regionprops to: {output_csv}")
        print(f"   Total regions: {len(df)}")
        print(f"   Columns: {', '.join(df.columns)}")

        return df
    else:
        print("⚠️  No regions found in any label images")
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=["filename", "label", "area"])
        df.to_csv(output_csv, index=False)
        return df


@BatchProcessingRegistry.register(
    name="Extract Regionprops to CSV",
    suffix="_regionprops",
    description="Extract region properties from label images to single CSV. Set label_suffix (e.g., '_otsu_semantic.tif') to filter only label files and pair with intensity images. All results saved to one CSV per folder.",
    parameters={
        "max_spatial_dims": {
            "type": int,
            "default": 2,
            "min": 2,
            "max": 3,
            "description": "Spatial dimensions: 2=2D slices (YX), 3=3D volumes (ZYX)",
        },
        "overwrite_existing": {
            "type": bool,
            "default": False,
            "description": "Overwrite existing CSV file if it exists",
        },
        "label_suffix": {
            "type": str,
            "default": "",
            "description": "Label suffix to remove for finding intensity image (e.g., '_otsu_semantic.tif'). Only files with this suffix are processed. Removes suffix to find intensity image. Leave empty to skip intensity.",
        },
        "size": {
            "type": bool,
            "default": True,
            "description": "Size (pixel count)",
        },
        "centroid": {
            "type": bool,
            "default": True,
            "description": "Centroid Y,X coords",
        },
        "bbox": {
            "type": bool,
            "default": True,
            "description": "Bounding box coords",
        },
        "perimeter": {
            "type": bool,
            "default": False,
            "description": "Perimeter (2D)",
        },
        "eccentricity": {
            "type": bool,
            "default": False,
            "description": "Eccentricity (2D)",
        },
        "extent": {
            "type": bool,
            "default": False,
            "description": "Extent (area/bbox ratio)",
        },
        "solidity": {
            "type": bool,
            "default": False,
            "description": "Solidity (2D, SLOW)",
        },
        "major_axis": {
            "type": bool,
            "default": False,
            "description": "Major axis (2D)",
        },
        "minor_axis": {
            "type": bool,
            "default": False,
            "description": "Minor axis (2D)",
        },
        "orientation": {
            "type": bool,
            "default": False,
            "description": "Orientation angle (2D)",
        },
        "mean_intensity": {
            "type": bool,
            "default": True,
            "description": "Mean intensity",
        },
        "median_intensity": {
            "type": bool,
            "default": True,
            "description": "Median intensity",
        },
        "std_intensity": {
            "type": bool,
            "default": True,
            "description": "Std intensity",
        },
        "max_intensity": {
            "type": bool,
            "default": False,
            "description": "Max intensity",
        },
        "min_intensity": {
            "type": bool,
            "default": False,
            "description": "Min intensity",
        },
    },
)
def extract_regionprops_folder(
    image: np.ndarray,
    max_spatial_dims: int = 2,
    overwrite_existing: bool = False,
    label_suffix: str = "",
    size: bool = True,
    centroid: bool = True,
    bbox: bool = True,
    perimeter: bool = False,
    eccentricity: bool = False,
    extent: bool = False,
    solidity: bool = False,
    major_axis: bool = False,
    minor_axis: bool = False,
    orientation: bool = False,
    mean_intensity: bool = True,
    median_intensity: bool = True,
    std_intensity: bool = True,
    max_intensity: bool = False,
    min_intensity: bool = False,
    dimension_order: str = "Auto",
) -> None:
    """
    Extract region properties from a label image and append to CSV file.

    This function processes a single label image and extracts region properties
    (area, centroid, bounding box, etc.) for each labeled region. Results are
    appended to a single CSV file per folder (created in parent directory).

    **Output:** Creates ONLY a CSV file, no image files are generated.

    The function uses dimension_order (from file selector dropdown) to properly identify
    T (time) and C (channel) dimensions, which are treated as grouping variables in the
    output CSV.

    **Intensity Measurements:** If label_suffix is provided, the function will find
    matching intensity images by replacing the suffix in label filenames. For example:
    - Label: "image_otsu_semantic.tif", Intensity: "image.tif" → use label_suffix="_otsu_semantic.tif"
    - This enables mean/max/min intensity measurements for each region.

    Parameters:
    -----------
    image : numpy.ndarray
        Input label image
    max_spatial_dims : int
        Maximum number of spatial dimensions to process as a unit (2=YX, 3=ZYX)
    overwrite_existing : bool
        Overwrite existing CSV file if it exists (only applies to first image)
    label_suffix : str
        Label file suffix to remove for finding matching intensity image
        (e.g., "_otsu_semantic.tif"). Leave empty to skip intensity measurements.
    size, centroid, bbox, ... : bool
        Enable/disable specific region properties
    dimension_order : str
        Dimension order string (e.g., "TZYX", "CZYX", "TYX", "Auto")
        This parameter is automatically provided by the file selector dropdown

    Returns:
    --------
    None
        This function only generates CSV output, no image is returned
    """
    global _REGIONPROPS_CSV_FILES

    # Get the current file path from the call stack
    current_file = get_current_filepath()

    if current_file is None:
        print("⚠️  Could not determine current file path")
        return None

    # This is the label image file
    label_path = Path(current_file)

    # IMPORTANT: Only process files that have the label_suffix (label images)
    # Skip files without the suffix (those are the intensity images)
    if label_suffix and label_suffix.strip():
        # The suffix should include the extension, e.g., "_otsu_semantic.tif"
        # Check if filename ends with this suffix
        filename_str = str(label_path.name)
        if not filename_str.endswith(label_suffix):
            # This file doesn't have the label suffix - skip it (it's an intensity image)
            return None

    # Generate output CSV path (one per folder)
    folder_name = label_path.parent.name
    parent_dir = label_path.parent.parent
    output_csv = str(parent_dir / f"{folder_name}_regionprops.csv")

    # Convert checkbox parameters to properties list
    properties_list = []
    if size:
        properties_list.append("area")
    if centroid:
        properties_list.append("centroid")
    if bbox:
        properties_list.append("bbox")
    if perimeter:
        properties_list.append("perimeter")
    if eccentricity:
        properties_list.append("eccentricity")
    if extent:
        properties_list.append("extent")
    if solidity:
        properties_list.append("solidity")
    if major_axis:
        properties_list.append("major_axis_length")
    if minor_axis:
        properties_list.append("minor_axis_length")
    if orientation:
        properties_list.append("orientation")
    if mean_intensity:
        properties_list.append("mean_intensity")
    if median_intensity:
        properties_list.append("median_intensity")
    if std_intensity:
        properties_list.append("std_intensity")
    if max_intensity:
        properties_list.append("max_intensity")
    if min_intensity:
        properties_list.append("min_intensity")

    # Always include label
    if "label" not in properties_list:
        properties_list.insert(0, "label")

    # Debug: Print properties to extract (only on first file)
    if output_csv not in _REGIONPROPS_CSV_FILES:
        print(f"📋 Properties to extract: {', '.join(properties_list)}")

    # Check if CSV already exists and overwrite_existing is False
    csv_path = Path(output_csv)
    if (
        csv_path.exists()
        and not overwrite_existing
        and output_csv not in _REGIONPROPS_CSV_FILES
    ):
        # CSV exists, don't overwrite, and we haven't tracked it yet in this session
        # Skip processing (user wants to keep existing file)
        return None

    # Determine if this is the first image in this folder
    write_header = False
    if output_csv not in _REGIONPROPS_CSV_FILES:
        # First time seeing this CSV in this session
        if overwrite_existing or not csv_path.exists():
            write_header = True
        _REGIONPROPS_CSV_FILES[output_csv] = True

    # Process this single image
    try:  # noqa: SIM105
        # Load intensity image if suffix provided
        intensity_image = None
        if label_suffix and label_suffix.strip():
            intensity_path_str = str(label_path).replace(
                label_suffix, Path(label_path).suffix
            )
            intensity_path = Path(intensity_path_str)
            if intensity_path.exists():
                intensity_image = load_label_image(str(intensity_path))
                print(f"📊 Loaded intensity image: {intensity_path.name}")
            else:
                print(f"⚠️  Intensity image not found: {intensity_path.name}")

        # Extract regionprops for this image
        results = extract_regionprops_recursive(
            image=image,
            intensity_image=intensity_image,
            properties=properties_list,
            max_spatial_dims=max_spatial_dims,
            dimension_order=dimension_order,
        )

        # Add filename to each result
        for row in results:
            row["filename"] = label_path.name

        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)

            # Write to CSV (header only on first write)
            if write_header:
                df.to_csv(output_csv, index=False, mode="w")
                print(f"✅ Created CSV with header: {output_csv}")
            else:
                df.to_csv(output_csv, index=False, mode="a", header=False)
                print(f"✅ Appended {len(df)} rows to: {output_csv}")

            return None
        else:
            print(f"⚠️  No regions found in {label_path.name}")
            return None

    except Exception as e:  # noqa: BLE001
        print(f"❌ Error processing {label_path.name}: {e}")
        import traceback

        traceback.print_exc()
        return None

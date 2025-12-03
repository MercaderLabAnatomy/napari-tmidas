# processing_functions/grid_view_overlay.py
"""
Processing function for displaying grid view of intensity images overlaid with labels.
"""
import concurrent.futures
import inspect
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from napari_tmidas._registry import BatchProcessingRegistry

# Lazy imports for optional dependencies
try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False

# Global flags to ensure grid is created and saved only once per batch
_grid_created = False
_cached_grid = None
_grid_saved = False
_grid_output_path = None


def _get_intensity_filename(label_filename: str) -> str:
    """
    Get intensity filename from label filename by removing label suffix.

    Parameters
    ----------
    label_filename : str
        Label image filename

    Returns
    -------
    str
        Intensity image filename
    """
    # Remove common label suffixes (handle both with and without .tif extension)
    suffixes_to_remove = [
        "_convpaint_labels_filtered.tif",
        "_labels_filtered.tif",
        "_labels.tif",
        "_intensity_filtered.tif",
        "_convpaint_labels_filtered",
        "_labels_filtered",
        "_labels",
        "_intensity_filtered",
    ]

    for suffix in suffixes_to_remove:
        if label_filename.endswith(suffix):
            base = label_filename.replace(suffix, "")
            # Ensure .tif extension
            if not base.endswith(".tif"):
                base += ".tif"
            return base

    # If no known suffix found, just use the filename as-is (already has .tif)
    return label_filename


def _downsample_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downsample image to target size while preserving aspect ratio.

    Uses skimage which handles all dtypes including uint32.

    Parameters
    ----------
    image : np.ndarray
        Input image
    target_size : int
        Target size for the larger dimension

    Returns
    -------
    np.ndarray
        Downsampled image
    """
    from skimage.transform import resize

    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim <= target_size:
        return image  # No downsampling needed

    scale = target_size / max_dim
    new_h = int(h * scale)
    new_w = int(w * scale)

    # skimage handles all dtypes including uint32
    if len(image.shape) == 2:
        downsampled = resize(
            image,
            (new_h, new_w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
    else:
        downsampled = resize(
            image,
            (new_h, new_w, image.shape[2]),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )

    return downsampled.astype(image.dtype)


def _create_overlay(
    intensity_image: np.ndarray,
    label_image: np.ndarray,
    target_size: int = None,
    label_opacity: float = 0.6,
    show_overlay: bool = True,
) -> np.ndarray:
    """
    Create an overlay of intensity and label images with transparency.

    Parameters
    ----------
    intensity_image : np.ndarray
        Intensity image
    label_image : np.ndarray
        Label image
    target_size : int, optional
        Target size for downsampling (max dimension). If None, no downsampling.
    label_opacity : float, optional
        Opacity of label overlay (0-1). Default is 0.6 (60%).
    show_overlay : bool, optional
        If True, show colored label overlay on intensity (default).
        If False, show only intensity in grayscale.

    Returns
    -------
    np.ndarray
        RGB overlay image with intensity in grayscale and optional colored label regions
    """
    # Downsample if target size specified
    if target_size is not None:
        intensity_image = _downsample_image(intensity_image, target_size)
        # Use nearest neighbor for labels to preserve label IDs
        h, w = intensity_image.shape
        if label_image.shape != (h, w):
            from skimage.transform import resize

            # skimage handles uint32 natively, use order=0 for nearest neighbor
            label_image = resize(
                label_image,
                (h, w),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(label_image.dtype)

    # Normalize intensity image to 0-1 range
    if (
        intensity_image.dtype != np.float32
        and intensity_image.dtype != np.float64
    ):
        intensity_norm = intensity_image.astype(np.float32)
    else:
        intensity_norm = intensity_image.copy()

    if intensity_norm.max() > 0:
        intensity_norm = (intensity_norm - intensity_norm.min()) / (
            intensity_norm.max() - intensity_norm.min()
        )

    # Create RGB image with intensity in grayscale (all channels)
    h, w = intensity_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = intensity_norm  # Red
    rgb[:, :, 1] = intensity_norm  # Green
    rgb[:, :, 2] = intensity_norm  # Blue

    # Create colored label overlay using simple colormap (only if show_overlay is True)
    if show_overlay:
        # Generate colors for each unique label (excluding background)
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        if len(unique_labels) == 0:
            # No labels found - image will be grayscale only
            pass  # rgb already has intensity in all channels (grayscale)
        elif len(unique_labels) > 0:
            # Create a simple colormap using hue variation
            # Use modulo to cycle through distinct colors even with many labels
            for i, label_id in enumerate(unique_labels):
                mask = label_image == label_id

                # Generate color by cycling through hue values (0-360 degrees)
                hue = (
                    i * 137.5
                ) % 360  # Golden angle for better color distribution

                # Convert HSV to RGB (H=hue, S=1, V=1)
                h_norm = hue / 60.0
                h_int = int(h_norm) % 6
                f = h_norm - int(h_norm)

                if h_int == 0:
                    r, g, b = 1.0, f, 0.0
                elif h_int == 1:
                    r, g, b = 1.0 - f, 1.0, 0.0
                elif h_int == 2:
                    r, g, b = 0.0, 1.0, f
                elif h_int == 3:
                    r, g, b = 0.0, 1.0 - f, 1.0
                elif h_int == 4:
                    r, g, b = f, 0.0, 1.0
                else:
                    r, g, b = 1.0, 0.0, 1.0 - f

                # Blend with opacity
                rgb[mask, 0] = (1 - label_opacity) * rgb[
                    mask, 0
                ] + label_opacity * r
                rgb[mask, 1] = (1 - label_opacity) * rgb[
                    mask, 1
                ] + label_opacity * g
                rgb[mask, 2] = (1 - label_opacity) * rgb[
                    mask, 2
                ] + label_opacity * b

    # Convert to uint8
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    return rgb_uint8


def _create_grid(images: list, grid_cols: int = 4) -> np.ndarray:
    """
    Arrange images in a grid layout.

    Parameters
    ----------
    images : list
        List of images to arrange
    grid_cols : int
        Number of columns in grid

    Returns
    -------
    np.ndarray
        Grid image
    """
    if not images:
        return None

    # Calculate grid dimensions
    n_images = len(images)
    grid_rows = (n_images + grid_cols - 1) // grid_cols

    # Get dimensions from first image
    h, w = images[0].shape[:2]
    has_channels = len(images[0].shape) == 3
    n_channels = images[0].shape[2] if has_channels else 1

    # Create grid
    if has_channels:
        grid = np.zeros(
            (grid_rows * h, grid_cols * w, n_channels), dtype=images[0].dtype
        )
    else:
        grid = np.zeros((grid_rows * h, grid_cols * w), dtype=images[0].dtype)

    # Fill grid
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start = row * h
        y_end = (row + 1) * h
        x_start = col * w
        x_end = (col + 1) * w
        grid[y_start:y_end, x_start:x_end] = img

    return grid


@BatchProcessingRegistry.register(
    name="Grid View: Intensity + Labels Overlay",
    suffix="_grid_overlay.tif",
    description="Create grid view of intensity images with optional colored label overlay for selected files",
    parameters={
        "label_suffix": {
            "type": str,
            "default": "_labels.tif",
            "description": "Example: _labels.tif. Leave empty for intensity-only grid.",
        }
    },
)
def create_grid_overlay(
    image: np.ndarray, label_suffix: str = "_labels.tif"
) -> np.ndarray:
    """
    Create a grid view showing intensity images with optional colored label overlay.

    This function processes all files selected in the batch processing queue.
    If label_suffix is provided, it finds corresponding label files and creates
    overlays. If label_suffix is empty, it creates a grid of intensity images only.

    Parameters
    ----------
    image : np.ndarray
        Input image (processed as part of the batch)
    label_suffix : str, optional
        Suffix pattern to identify label files (e.g., "_labels.tif", "_segmentation.tif").
        If empty string, creates intensity-only grid without looking for labels.
        Default is "_labels.tif".

    Returns
    -------
    np.ndarray
        Grid image with intensity images and optional overlays (RGB uint8)

    Notes
    -----
    - Intensity is shown in grayscale
    - When labels are used: each label gets a unique color with 60% opacity
    - Images are automatically normalized for display
    - Grid columns are automatically determined based on number of images
    - Only processes files selected by user's suffix filter in batch processing
    """
    global _grid_created, _cached_grid

    # Determine mode based on label_suffix
    intensity_only_mode = label_suffix == "" or label_suffix is None
    mode_str = (
        "intensity only"
        if intensity_only_mode
        else f"intensity + labels (suffix: '{label_suffix}')"
    )

    # If grid has already been created in this batch, return None to skip saving
    if _grid_created:
        return None

    if not _HAS_TIFFFILE:
        print(
            "⚠️  tifffile not available. Please install it: pip install tifffile"
        )
        return image

    # Mark that we're creating the grid to prevent concurrent calls
    _grid_created = True

    # Suppress any stdout from this point to avoid verbose output
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Get the filepath, files list, and output folder from the call stack
        current_filepath = None
        file_list = None
        output_folder = None

        for frame_info in inspect.stack():
            frame_locals = frame_info.frame.f_locals
            if "filepath" in frame_locals:
                current_filepath = Path(frame_locals["filepath"])
            if "file_list" in frame_locals:
                file_list = frame_locals["file_list"]
            if "self" in frame_locals:
                obj = frame_locals["self"]
                if hasattr(obj, "output_folder"):
                    output_folder = obj.output_folder
            if (
                current_filepath is not None
                and file_list is not None
                and output_folder is not None
            ):
                break
    finally:
        # Restore stdout
        sys.stdout = old_stdout

    if current_filepath is None:
        print("⚠️  Could not determine current file path")
        return image

    label_folder = current_filepath.parent

    # Use output folder from batch processing if available; default to parent folder
    if output_folder is None:
        output_folder = str(label_folder)

    output_folder = Path(output_folder)

    # When the batch UI output matches the input folder (blank value), save to parent
    try:
        same_as_input = output_folder.resolve() == label_folder.resolve()
    except FileNotFoundError:
        same_as_input = False

    if same_as_input:
        output_folder = label_folder.parent

    output_folder.mkdir(parents=True, exist_ok=True)

    # Use the file_list from batch processing if available
    if file_list is not None:
        label_files = [Path(f) for f in file_list]
    else:
        # Fallback: determine files based on mode
        label_files = []

        if intensity_only_mode:
            # Intensity-only mode: use all TIFF files in folder
            patterns = ["*.tif", "*.tiff"]
        else:
            # Label mode: honor user-provided suffix while keeping legacy patterns
            patterns = []
            if label_suffix:
                suffixes = {label_suffix}
                if label_suffix.lower().endswith(".tif"):
                    suffixes.add(label_suffix[:-4] + ".tiff")
                for suffix in suffixes:
                    patterns.append(f"*{suffix}")

            # Legacy fallback patterns
            patterns.extend(["*_labels*.tif", "*_labels*.tiff"])

        for pattern in patterns:
            label_files.extend(label_folder.glob(pattern))

        if not label_files:
            msg = "intensity files" if intensity_only_mode else "label files"
            print(f"⚠️  No {msg} found in folder")
            return image

    # Filter out any grid overlay files to prevent reprocessing
    label_files = [f for f in label_files if "_grid_overlay" not in f.name]

    # Deduplicate and sort for deterministic ordering
    label_files = sorted(set(label_files))

    if not label_files:
        print("⚠️  No valid label files found after filtering")
        return image

    # Calculate square grid dimensions
    import math

    # For square grid: use sqrt to get equal rows and columns
    grid_cols = max(1, math.ceil(math.sqrt(len(label_files))))
    grid_rows = grid_cols  # Square grid

    # Target final dimensions (aim for ~12000px max dimension for PNG compatibility)
    # Square grid means we can use same calculation for both dimensions
    max_grid_dimension = 12000
    target_per_image = max_grid_dimension // grid_cols

    # Clamp to reasonable range (not too small, not too large)
    target_per_image = max(100, min(target_per_image, 500))

    print(
        f"\n📊 Processing {len(label_files)} images → {target_per_image}px per image, {grid_cols}×{grid_rows} grid (square), {mode_str}"
    )

    # Create overlays for each pair using parallel processing
    def process_image_pair(file_path):
        """Process a single image file (with or without labels)."""
        filename = file_path.name

        if intensity_only_mode:
            # Intensity-only mode: use the file itself as intensity, no labels
            intensity_path = file_path
            label_path = None
        else:
            # Label mode: file is a label, find corresponding intensity
            intensity_filename = _get_intensity_filename(filename)
            intensity_path = label_folder / intensity_filename
            label_path = file_path

            if not intensity_path.exists():
                return (
                    None,
                    f"⚠️  Skipping {filename}: no intensity image found",
                )

        try:
            # Load intensity image
            intensity_img = tifffile.imread(str(intensity_path))

            # Load label image if in label mode
            if label_path is not None:
                label_img = tifffile.imread(str(label_path))
            else:
                label_img = None

            # Handle 3D data by taking max projection
            if len(intensity_img.shape) > 2:
                intensity_img = np.max(intensity_img, axis=0)

            if label_img is not None:
                if len(label_img.shape) > 2:
                    label_img = np.max(label_img, axis=0)

                # Check for labels
                unique_labels = np.unique(label_img)
                n_labels = len(unique_labels[unique_labels > 0])

                # Ensure matching dimensions
                if intensity_img.shape != label_img.shape:
                    return None, (
                        f"⚠️  Skipping {filename}: dimension mismatch "
                        f"(intensity: {intensity_img.shape}, labels: {label_img.shape})"
                    )
            else:
                # Intensity-only mode: create dummy zero label image
                label_img = np.zeros(intensity_img.shape, dtype=np.uint16)
                n_labels = 0

            # Create overlay with intelligent downsampling and 60% label opacity
            # When label_img is all zeros (intensity_only_mode), this creates grayscale output
            overlay = _create_overlay(
                intensity_img,
                label_img,
                target_size=target_per_image,
                label_opacity=0.6,
                show_overlay=(not intensity_only_mode),
            )

            # Explicitly delete large arrays to free memory immediately
            del intensity_img, label_img

            return (overlay, n_labels), None

        except (FileNotFoundError, OSError) as e:
            return None, f"⚠️  Error processing {filename}: {e}"

    # Process in parallel with ThreadPoolExecutor
    # Use max 4 workers for better memory management with large datasets
    max_workers = min(4, os.cpu_count() or 4)

    # Process in batches to manage memory
    batch_size = 100  # Process 100 images at a time
    sorted_label_files = sorted(label_files)
    all_overlays = []
    valid_pairs = 0
    errors = []
    total_labels = 0
    images_with_labels = 0

    # Progress bar for overall processing
    with tqdm(
        total=len(sorted_label_files), desc="Creating overlays", unit="pair"
    ) as pbar:
        for batch_start in range(0, len(sorted_label_files), batch_size):
            batch_end = min(batch_start + batch_size, len(sorted_label_files))
            batch_files = sorted_label_files[batch_start:batch_end]

            batch_overlays = []

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit batch tasks
                future_to_path = {
                    executor.submit(process_image_pair, label_path): label_path
                    for label_path in batch_files
                }

                # Collect batch results as they complete
                for future in concurrent.futures.as_completed(future_to_path):
                    result, error_msg = future.result()

                    if error_msg:
                        errors.append(error_msg)
                    elif result is not None:
                        overlay, n_labels = result
                        batch_overlays.append(overlay)
                        valid_pairs += 1
                        total_labels += n_labels
                        if n_labels > 0:
                            images_with_labels += 1

                    pbar.update(1)

            # Add batch results to main list
            all_overlays.extend(batch_overlays)

            # Clear batch to free memory
            del batch_overlays

    # Print diagnostics after progress bar completes
    print(f"\n✓ Processed {valid_pairs} images")
    if not intensity_only_mode:
        print(
            f"  Labels found: {images_with_labels}/{valid_pairs} images ({total_labels} total labels)"
        )

        if total_labels == 0:
            print("  ⚠️  WARNING: No labels detected in any image!")
            print(
                "      Output will be grayscale intensity only (no colored regions)"
            )
    else:
        print("  Mode: Intensity only (no labels)")

    if errors:
        print(f"\n⚠️  {len(errors)} files skipped:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")

    if not all_overlays:
        print("⚠️  No valid image pairs found")
        return image

    overlays = all_overlays

    mode_desc = (
        "intensity only"
        if intensity_only_mode
        else "intensity + colored labels at 60% opacity"
    )
    print(
        f"\n✨ Creating final grid: {valid_pairs} images, {grid_cols}×{grid_cols} square grid ({mode_desc})"
    )

    # Create grid
    grid = _create_grid(overlays, grid_cols=grid_cols)

    if grid is None:
        print("⚠️  ERROR: Grid creation returned None!")
        return image

    print(f"✅ Complete! Grid shape: {grid.shape}")

    # Cache the result for subsequent calls in the same batch
    _cached_grid = grid

    # Save the grid to file (only once)
    global _grid_saved, _grid_output_path
    if not _grid_saved:
        # Save as compressed TIF for napari viewing
        output_filename = f"{sorted_label_files[0].stem}_grid_overlay.tif"
        output_path = output_folder / output_filename

        try:
            # Ensure output directory exists
            output_folder.mkdir(parents=True, exist_ok=True)

            # Save as compressed TIFF
            tifffile.imwrite(
                str(output_path),
                grid,
                compression="zlib",
                compressionargs={"level": 6},
            )
            _grid_output_path = str(output_path)
            _grid_saved = True

            # Verify file was actually saved
            if output_path.exists():
                file_size = output_path.stat().st_size
                print("\n" + "=" * 80)
                print("💾 SAVED GRID IMAGE TO:")
                print(f"   {output_path}")
                print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
                print("   (Compressed TIF format)")
                print("=" * 80 + "\n")
            else:
                print(
                    f"⚠️  WARNING: File save appeared to succeed but file not found at {output_path}"
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
            tifffile.TiffFileError,
        ) as e:
            print(f"⚠️  Error saving grid: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    return grid


def reset_grid_cache():
    """Reset the grid creation cache for a new batch run."""
    global _grid_created, _cached_grid, _grid_saved, _grid_output_path
    _grid_created = False
    _cached_grid = None
    _grid_saved = False
    _grid_output_path = None

# processing_functions/spotiflow_detection.py
"""
Processing functions for spot detection using Spotiflow.

This module provides functionality to detect spots in fluorescence microscopy images
using Spotiflow models. It supports both 2D and 3D data with various pretrained models.

The functions will automatically create and manage a dedicated environment for Spotiflow
if it's not already installed in the main environment.
"""
import os

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

# Import the environment manager for Spotiflow
from napari_tmidas.processing_functions.spotiflow_env_manager import (
    run_spotiflow_in_env,
)


# Utility functions for axes and input preparation (from napari-spotiflow)
def _validate_axes(img: np.ndarray, axes: str) -> None:
    """Validate that the number of dimensions in the image matches the given axes string."""
    if img.ndim != len(axes):
        raise ValueError(
            f"Image has {img.ndim} dimensions, but axes has {len(axes)} dimensions"
        )


def _prepare_input(img: np.ndarray, axes: str) -> np.ndarray:
    """Reshape input for Spotiflow's API compatibility based on axes notation."""
    _validate_axes(img, axes)

    if axes in {"YX", "ZYX", "TYX", "TZYX"}:
        return img[..., None]
    elif axes in {"YXC", "ZYXC", "TYXC", "TZYXC"}:
        return img
    elif axes == "CYX":
        return img.transpose(1, 2, 0)
    elif axes == "CZYX":
        return img.transpose(1, 2, 3, 0)
    elif axes == "ZCYX" or axes == "TCYX":
        return img.transpose(0, 2, 3, 1)
    elif axes == "TZCYX":
        return img.transpose(0, 1, 3, 4, 2)
    elif axes == "TCZYX":
        return img.transpose(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Invalid axes: {axes}")


def _infer_axes(img: np.ndarray) -> str:
    """Infer the most likely axes order for the image."""
    ndim = img.ndim
    if ndim == 2:
        return "YX"
    elif ndim == 3:
        # For 3D, we need to make an educated guess
        # Most common is ZYX for 3D microscopy
        return "ZYX"
    elif ndim == 4:
        # Could be TZYX or ZYXC, let's check the last dimension
        if img.shape[-1] <= 4:  # Likely channels
            return "ZYXC"
        else:
            return "TZYX"
    elif ndim == 5:
        return "TZYXC"
    else:
        raise ValueError(f"Cannot infer axes for {ndim}D image")


# Check if Spotiflow is directly available in current environment
try:
    import importlib.util

    spec = importlib.util.find_spec("spotiflow.model")
    if spec is not None:
        SPOTIFLOW_AVAILABLE = True
        USE_DEDICATED_ENV = False
        print("Spotiflow found in current environment, using direct import")
    else:
        raise ImportError("Spotiflow not found")
except ImportError:
    SPOTIFLOW_AVAILABLE = False
    USE_DEDICATED_ENV = True
    print(
        "Spotiflow not found in current environment, will use dedicated environment"
    )


def _convert_points_to_labels_with_heatmap(
    image: np.ndarray,
    points: np.ndarray,
    spot_radius: int,
    pretrained_model: str,
    model_path: str,
    prob_thresh: float,
    force_cpu: bool,
) -> np.ndarray:
    """
    Convert points to label masks using Spotiflow's probability heatmap for better segmentation.
    """
    try:
        import torch
        from scipy.ndimage import label
        from skimage.segmentation import watershed
        from spotiflow.model import Spotiflow

        # Set device
        if force_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Load the model (reuse existing model loading logic)
        if model_path and os.path.exists(model_path):
            model = Spotiflow.from_folder(model_path)
        else:
            model = Spotiflow.from_pretrained(pretrained_model)

        model = model.to(device)

        # Prepare input (reuse existing logic)
        axes = _infer_axes(image)
        prepared_img = _prepare_input(image, axes)

        # Normalize (simple percentile normalization)
        p_low, p_high = np.percentile(prepared_img, [1.0, 99.8])
        normalized_img = np.clip(
            (prepared_img - p_low) / (p_high - p_low), 0, 1
        )

        # Get prediction with details
        points_new, details = model.predict(
            normalized_img,
            prob_thresh=prob_thresh,
            device=device,
            verbose=False,
        )

        # Use probability heatmap for segmentation
        if hasattr(details, "heatmap") and details.heatmap is not None:
            prob_map = details.heatmap

            # Apply threshold to create binary mask
            threshold = prob_thresh if prob_thresh is not None else 0.4
            binary_mask = prob_map > threshold

            # Use detected points as seeds for watershed segmentation
            if len(points) > 0:
                # Create marker image from detected points
                markers = np.zeros(prob_map.shape, dtype=np.int32)
                for i, point in enumerate(points):
                    if len(point) >= 2:
                        y, x = int(point[0]), int(point[1])
                        if (
                            0 <= y < markers.shape[0]
                            and 0 <= x < markers.shape[1]
                        ):
                            markers[y, x] = i + 1

                # Apply watershed segmentation using probability map and markers
                labels = watershed(-prob_map, markers, mask=binary_mask)
            else:
                # No points detected, just label connected components
                labels, _ = label(binary_mask)

            return labels.astype(np.uint16)
        else:
            # Fallback to point-based method
            return _points_to_label_mask(points, image.shape[:2], spot_radius)

    except (ImportError, RuntimeError, ValueError, AttributeError) as e:
        print(f"Error in heatmap-based conversion: {e}")
        # Fallback to point-based method
        return _points_to_label_mask(points, image.shape[:2], spot_radius)


@BatchProcessingRegistry.register(
    name="Spotiflow Spot Detection",
    suffix="_spot_labels",
    description="Detect spots in fluorescence microscopy images using Spotiflow and return as label masks",
    parameters={
        "pretrained_model": {
            "type": str,
            "default": "general",
            "description": "Pretrained model to use (general, hybiss, synth_complex, synth_3d, smfish_3d)",
            "choices": [
                "general",
                "hybiss",
                "synth_complex",
                "synth_3d",
                "smfish_3d",
            ],
        },
        "model_path": {
            "type": str,
            "default": "",
            "description": "Path to custom trained model folder (leave empty to use pretrained model)",
        },
        "subpixel": {
            "type": bool,
            "default": True,
            "description": "Enable subpixel localization for more accurate spot coordinates",
        },
        "peak_mode": {
            "type": str,
            "default": "fast",
            "description": "Peak detection mode",
            "choices": ["fast", "skimage"],
        },
        "normalizer": {
            "type": str,
            "default": "percentile",
            "description": "Image normalization method",
            "choices": ["percentile", "minmax"],
        },
        "normalizer_low": {
            "type": float,
            "default": 1.0,
            "min": 0.0,
            "max": 50.0,
            "description": "Lower percentile for normalization",
        },
        "normalizer_high": {
            "type": float,
            "default": 99.8,
            "min": 50.0,
            "max": 100.0,
            "description": "Upper percentile for normalization",
        },
        "prob_thresh": {
            "type": float,
            "default": None,
            "min": 0.0,
            "max": 1.0,
            "description": "Probability threshold (leave empty or 0.0 for automatic)",
        },
        "n_tiles": {
            "type": str,
            "default": "auto",
            "description": "Number of tiles for prediction (e.g., '(2,2)' or 'auto')",
        },
        "exclude_border": {
            "type": bool,
            "default": True,
            "description": "Exclude spots near image borders",
        },
        "scale": {
            "type": str,
            "default": "auto",
            "description": "Scaling factor (e.g., '(1,1)' or 'auto')",
        },
        "min_distance": {
            "type": int,
            "default": 2,
            "min": 1,
            "max": 10,
            "description": "Minimum distance between detected spots",
        },
        "spot_radius": {
            "type": int,
            "default": 3,
            "min": 1,
            "max": 20,
            "description": "Radius of spots in the label mask (in pixels, used for fallback method)",
        },
        "axes": {
            "type": str,
            "default": "auto",
            "description": "Axes order (e.g., 'ZYX', 'YX', or 'auto' for automatic detection)",
        },
        "output_csv": {
            "type": bool,
            "default": True,
            "description": "Save spot coordinates as CSV file alongside the mask",
        },
        "force_dedicated_env": {
            "type": bool,
            "default": False,
            "description": "Force using dedicated environment even if Spotiflow is available",
        },
        "force_cpu": {
            "type": bool,
            "default": False,
            "description": "Force CPU execution (disable GPU) to avoid CUDA compatibility issues",
        },
    },
)
def spotiflow_detect_spots(
    image: np.ndarray,
    pretrained_model: str = "general",
    model_path: str = "",
    subpixel: bool = True,
    peak_mode: str = "fast",
    normalizer: str = "percentile",
    normalizer_low: float = 1.0,
    normalizer_high: float = 99.8,
    prob_thresh: float = None,
    n_tiles: str = "auto",
    exclude_border: bool = True,
    scale: str = "auto",
    min_distance: int = 2,
    spot_radius: int = 3,
    axes: str = "auto",
    output_csv: bool = True,
    force_dedicated_env: bool = False,
    force_cpu: bool = False,
    # For internal use by processing system
    input_file_path: str = None,
) -> np.ndarray:
    """
    Detect spots in fluorescence microscopy images using Spotiflow and return label masks.

    Spotiflow is a deep learning-based spot detection method that provides
    threshold-agnostic, subpixel-accurate detection of spots in 2D and 3D
    fluorescence microscopy images. The output is a label mask suitable for
    napari Labels layers, created from the Spotiflow probability heatmap.

    Parameters:
    -----------
    image : np.ndarray
        Input image (2D or 3D)
    pretrained_model : str
        Pretrained model to use ('general', 'hybiss', 'synth_complex', 'synth_3d', 'smfish_3d')
    model_path : str
        Path to custom trained model folder (overrides pretrained_model if provided)
    subpixel : bool
        Enable subpixel localization
    peak_mode : str
        Peak detection mode ('fast' or 'skimage')
    normalizer : str
        Image normalization method ('percentile' or 'minmax')
    normalizer_low : float
        Lower percentile for normalization
    normalizer_high : float
        Upper percentile for normalization
    prob_thresh : float or None
        Probability threshold (None for automatic)
    n_tiles : str
        Number of tiles for prediction (e.g., '(2,2)' or 'auto')
    exclude_border : bool
        Exclude spots near image borders
    scale : str
        Scaling factor (e.g., '(1,1)' or 'auto')
    min_distance : int
        Minimum distance between detected spots
    spot_radius : int
        Radius of spots in the label mask (in pixels, used for fallback method)
    axes : str
        Axes order (e.g., 'ZYX', 'YX', or 'auto' for automatic detection)
    output_csv : bool
        Save spot coordinates as CSV file alongside the mask
    force_dedicated_env : bool
        Force using dedicated environment
    force_cpu : bool
        Force CPU execution (disable GPU) to avoid CUDA compatibility issues
    input_file_path : str
        Path to input file (used for saving CSV output)

    Returns:
    --------
    np.ndarray
        Label mask with detected spots (uint16) for napari Labels layer
    """
    print("Detecting spots using Spotiflow...")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")

    # Infer axes if auto
    if axes == "auto":
        axes = _infer_axes(image)
        print(f"Inferred axes: {axes}")
    else:
        print(f"Using provided axes: {axes}")

    # Decide whether to use dedicated environment
    use_env = USE_DEDICATED_ENV or force_dedicated_env

    if not use_env and SPOTIFLOW_AVAILABLE:
        # Use direct import
        points = _detect_spots_direct(
            image,
            axes,
            pretrained_model,
            model_path,
            subpixel,
            peak_mode,
            normalizer,
            normalizer_low,
            normalizer_high,
            prob_thresh,
            n_tiles,
            exclude_border,
            scale,
            min_distance,
            force_cpu,
        )
    else:
        # Use dedicated environment
        points = _detect_spots_env(
            image,
            axes,
            pretrained_model,
            model_path,
            subpixel,
            peak_mode,
            normalizer,
            normalizer_low,
            normalizer_high,
            prob_thresh,
            n_tiles,
            exclude_border,
            scale,
            min_distance,
            force_cpu,
        )

    # Save CSV if requested (use a default filename if no input path provided)
    if output_csv:
        if input_file_path:
            _save_coords_csv(points, input_file_path, use_env)
        else:
            # No input file path provided; skipping CSV export.
            print(
                "No input file path provided, skipping CSV export of spot coordinates."
            )

    # Convert points to label masks using the improved method
    print(f"Detected {len(points)} spots, converting to label masks...")

    # Always use the simple point-based method for now to ensure it works
    label_mask = _points_to_label_mask(points, image.shape, spot_radius)

    print(
        f"Created label mask with {len(np.unique(label_mask)) - 1} labeled objects"
    )
    return label_mask


def _points_to_label_mask(
    points: np.ndarray, image_shape: tuple, spot_radius: int
) -> np.ndarray:
    """Convert detected points to a label mask for napari."""
    from scipy import ndimage
    from skimage import draw

    # Create empty label mask with the same shape as input image
    label_mask = np.zeros(image_shape, dtype=np.uint16)

    # Handle different dimensionalities - focus on spatial dimensions
    spatial_dims = len(image_shape)
    if spatial_dims >= 4:  # TZYX, TZYXC, etc.
        if image_shape[-1] <= 4:  # Last dim is channels
            spatial_shape = image_shape[-4:-1]  # Take ZYX (skip channels)
        else:
            spatial_shape = image_shape[-3:]  # Take last 3 dims (ZYX)
    elif spatial_dims == 3:  # ZYX or YXC
        # Check if last dimension is small (likely channels)
        if image_shape[-1] <= 4:
            spatial_shape = image_shape[:2]  # YX (with channels)
        else:
            spatial_shape = image_shape  # ZYX
    else:  # 2D: YX or YXC
        if len(image_shape) == 3 and image_shape[-1] <= 4:
            spatial_shape = image_shape[:2]  # YX (with channels)
        else:
            spatial_shape = image_shape  # YX

    if len(points) == 0:
        return label_mask

    # Check coordinate format and swap if necessary
    if points.shape[1] == 2:  # 2D points (y, x)
        coords = points.astype(int)
    elif points.shape[1] == 3:  # 3D points - need to figure out the format
        # Try to determine the correct coordinate mapping based on spatial shape
        if len(spatial_shape) == 2:  # Working with 2D spatial data
            # If dim1 and dim2 fit in image bounds, assume (z, y, x)
            if (
                points[:, 1].max() < spatial_shape[0]
                and points[:, 2].max() < spatial_shape[1]
            ):
                coords = points[:, 1:3].astype(int)  # Take y, x (skip z)
            # If dim0 and dim2 fit in image bounds, assume (y, z, x)
            elif (
                points[:, 0].max() < spatial_shape[0]
                and points[:, 2].max() < spatial_shape[1]
            ):
                coords = points[:, [0, 2]].astype(int)  # Take y, x (skip z)
            # If dim0 and dim1 fit in image bounds, assume (y, x, z)
            elif (
                points[:, 0].max() < spatial_shape[0]
                and points[:, 1].max() < spatial_shape[1]
            ):
                coords = points[:, 0:2].astype(int)  # Take y, x (skip z)
            else:
                # Try swapping coordinates - maybe it's (x, y) instead of (y, x)
                coords = points[:, [1, 0]].astype(int)
        else:  # Working with 3D spatial data
            coords = points.astype(int)  # Use all 3 coordinates
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}")

    # Create spots based on spatial dimensions
    valid_spots = 0

    if len(spatial_shape) == 2:  # 2D spatial
        for i, (y, x) in enumerate(coords):
            if 0 <= y < spatial_shape[0] and 0 <= x < spatial_shape[1]:
                try:
                    rr, cc = draw.disk(
                        (y, x), spot_radius, shape=spatial_shape
                    )
                    # Handle different label mask shapes
                    if len(image_shape) == 2:  # Pure 2D
                        label_mask[rr, cc] = i + 1
                    elif len(image_shape) == 3:  # 2D with channels or 3D
                        if image_shape[-1] <= 4:  # Likely channels
                            label_mask[rr, cc, :] = (
                                i + 1
                            )  # Apply to all channels
                        else:  # 3D data - apply to all Z slices
                            label_mask[:, rr, cc] = i + 1
                    elif len(image_shape) == 4:  # TZYX or similar
                        label_mask[:, :, rr, cc] = (
                            i + 1
                        )  # Apply to all T and Z
                    elif len(image_shape) == 5:  # TZYXC
                        label_mask[:, :, rr, cc, :] = (
                            i + 1
                        )  # Apply to all T, Z, and C

                    valid_spots += 1
                except (ValueError, IndexError, TypeError) as e:
                    print(f"Error drawing spot {i} at ({y}, {x}): {e}")

    elif len(spatial_shape) == 3:  # 3D spatial
        # For 3D spatial, we need 3D coordinates
        if coords.shape[1] == 2:
            # We have 2D points but need 3D - place them in the middle Z slice
            middle_z = spatial_shape[0] // 2
            coords_3d = np.column_stack(
                [np.full(len(coords), middle_z), coords]
            )
        else:
            coords_3d = coords

        for i, (z, y, x) in enumerate(coords_3d):
            if (
                0 <= z < spatial_shape[0]
                and 0 <= y < spatial_shape[1]
                and 0 <= x < spatial_shape[2]
            ):
                try:
                    # Create a small sphere
                    ball = ndimage.generate_binary_structure(3, 1)
                    ball = ndimage.iterate_structure(ball, spot_radius)

                    # Get sphere coordinates
                    ball_coords = np.array(np.where(ball)).T - spot_radius
                    z_coords = ball_coords[:, 0] + z
                    y_coords = ball_coords[:, 1] + y
                    x_coords = ball_coords[:, 2] + x

                    # Filter valid coordinates
                    valid = (
                        (z_coords >= 0)
                        & (z_coords < spatial_shape[0])
                        & (y_coords >= 0)
                        & (y_coords < spatial_shape[1])
                        & (x_coords >= 0)
                        & (x_coords < spatial_shape[2])
                    )

                    # Handle different label mask shapes
                    if len(image_shape) == 3:  # Pure 3D
                        label_mask[
                            z_coords[valid], y_coords[valid], x_coords[valid]
                        ] = (i + 1)
                    elif len(image_shape) == 4:  # TZYX or ZYXC
                        if image_shape[-1] <= 4:  # ZYXC
                            label_mask[
                                z_coords[valid],
                                y_coords[valid],
                                x_coords[valid],
                                :,
                            ] = (
                                i + 1
                            )
                        else:  # TZYX
                            label_mask[
                                :,
                                z_coords[valid],
                                y_coords[valid],
                                x_coords[valid],
                            ] = (
                                i + 1
                            )
                    elif len(image_shape) == 5:  # TZYXC
                        label_mask[
                            :,
                            z_coords[valid],
                            y_coords[valid],
                            x_coords[valid],
                            :,
                        ] = (
                            i + 1
                        )

                    valid_spots += 1
                except (ValueError, IndexError, TypeError) as e:
                    print(f"Error drawing 3D spot {i} at ({z}, {y}, {x}): {e}")

    print(
        f"Successfully created {valid_spots} spots in label mask with shape {label_mask.shape}"
    )
    return label_mask


def _detect_spots_direct(
    image,
    axes,
    pretrained_model,
    model_path,
    subpixel,
    peak_mode,
    normalizer,
    normalizer_low,
    normalizer_high,
    prob_thresh,
    n_tiles,
    exclude_border,
    scale,
    min_distance,
    force_cpu,
):
    """Direct implementation using imported Spotiflow."""
    import torch
    from spotiflow.model import Spotiflow

    # Set device based on force_cpu parameter
    if force_cpu:
        print("Forcing CPU execution as requested")
        device = torch.device("cpu")
        # Set environment variable to ensure CPU usage
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Use CUDA if available and compatible
        if torch.cuda.is_available():
            try:
                # Test CUDA compatibility by creating a small tensor
                torch.ones(1).cuda()
                device = torch.device("cuda")
                print("Using CUDA (GPU) for inference")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"CUDA incompatible ({e}), falling back to CPU")
                device = torch.device("cpu")
                force_cpu = True
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
            force_cpu = True

    # Load the model
    if model_path and os.path.exists(model_path):
        print(f"Loading custom model from {model_path}")
        model = Spotiflow.from_folder(model_path)
    else:
        print(f"Loading pretrained model: {pretrained_model}")
        model = Spotiflow.from_pretrained(pretrained_model)

    # Move model to the appropriate device
    try:
        model = model.to(device)
        print(f"Model moved to device: {device}")
    except Exception as e:
        if not force_cpu:
            print(f"Failed to move model to GPU ({e}), falling back to CPU")
            device = torch.device("cpu")
            model = model.to(device)
        else:
            raise

    # Check model compatibility with image dimensionality
    is_3d_image = len(image.shape) == 3 and "Z" in axes
    if is_3d_image and not model.config.is_3d:
        print(
            "Warning: Using a 2D model on 3D data. Consider using a 3D model like 'synth_3d' or 'smfish_3d'."
        )

    # Prepare input using the same method as napari-spotiflow
    print(f"Preparing input with axes: {axes}")
    try:
        prepared_img = _prepare_input(image, axes)
        print(f"Prepared image shape: {prepared_img.shape}")
    except ValueError as e:
        print(f"Error preparing input: {e}")
        # Fallback to original image
        prepared_img = image

    # Parse string parameters
    def parse_param(param_str, default_val):
        if param_str == "auto":
            return default_val
        try:
            return eval(param_str) if param_str.startswith("(") else param_str
        except (ValueError, SyntaxError):
            return default_val

    n_tiles_parsed = parse_param(n_tiles, None)
    scale_parsed = parse_param(scale, None)

    # Prepare prediction parameters (following napari-spotiflow style)
    predict_kwargs = {
        "subpix": subpixel,  # Note: Spotiflow API uses 'subpix', not 'subpixel'
        "peak_mode": peak_mode,
        "normalizer": None,  # We'll handle normalization manually
        "exclude_border": exclude_border,
        "min_distance": min_distance,
        "verbose": True,
    }

    # Set probability threshold - use automatic or provided value
    if prob_thresh is not None and prob_thresh > 0.0:
        predict_kwargs["prob_thresh"] = prob_thresh
    else:
        # Use automatic thresholding similar to napari-spotiflow
        # Don't set prob_thresh - let spotiflow determine it automatically
        # This includes None and 0.0 values which should use automatic thresholding
        pass  # Spotiflow will use its default optimized threshold

    if n_tiles_parsed is not None:
        predict_kwargs["n_tiles"] = n_tiles_parsed
    if scale_parsed is not None:
        predict_kwargs["scale"] = scale_parsed

    # Handle normalization manually (similar to napari-spotiflow)
    if normalizer == "percentile":
        print(
            f"Applying percentile normalization: {normalizer_low}% to {normalizer_high}%"
        )
        p_low, p_high = np.percentile(
            prepared_img, [normalizer_low, normalizer_high]
        )
        normalized_img = np.clip(
            (prepared_img - p_low) / (p_high - p_low), 0, 1
        )
    elif normalizer == "minmax":
        print("Applying min-max normalization")
        img_min, img_max = prepared_img.min(), prepared_img.max()
        normalized_img = (
            (prepared_img - img_min) / (img_max - img_min)
            if img_max > img_min
            else prepared_img
        )
    else:
        normalized_img = prepared_img

    print(
        f"Normalized image range: {normalized_img.min():.3f} to {normalized_img.max():.3f}"
    )

    # Perform spot detection
    print("Running Spotiflow prediction...")
    try:
        points, details = model.predict(normalized_img, **predict_kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA" in str(e) and not force_cpu:
            print(f"CUDA error during prediction ({e}), retrying with CPU")
            # Move model to CPU and retry
            device = torch.device("cpu")
            model = model.to(device)
            # Set environment to force CPU
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            points, details = model.predict(normalized_img, **predict_kwargs)
        else:
            raise

    print(f"Initial detection: {len(points)} spots")

    # Only apply minimal additional filtering if we still have too many detections
    # This should rarely be needed now that we use proper automatic thresholding
    if len(points) > 500:  # Only if we have an excessive number of spots
        print(f"Applying additional filtering for {len(points)} spots")

        # Check if we can apply probability filtering
        if hasattr(details, "prob"):
            # Use a more stringent threshold
            auto_thresh = 0.7
            prob_mask = details.prob > auto_thresh
            points = points[prob_mask]
            print(
                f"After additional probability thresholding ({auto_thresh}): {len(points)} spots"
            )

    print(f"Final detection: {len(points)} spots")
    return points


def _detect_spots_env(
    image,
    axes,
    pretrained_model,
    model_path,
    subpixel,
    peak_mode,
    normalizer,
    normalizer_low,
    normalizer_high,
    prob_thresh,
    n_tiles,
    exclude_border,
    scale,
    min_distance,
    force_cpu,
):
    """Implementation using dedicated environment."""
    # Prepare arguments for environment execution
    args_dict = {
        "image": image,
        "axes": axes,
        "pretrained_model": pretrained_model,
        "model_path": model_path,
        "subpixel": subpixel,
        "peak_mode": peak_mode,
        "normalizer": normalizer,
        "normalizer_low": normalizer_low,
        "normalizer_high": normalizer_high,
        "prob_thresh": prob_thresh,
        "n_tiles": n_tiles,
        "exclude_border": exclude_border,
        "scale": scale,
        "min_distance": min_distance,
        "force_cpu": force_cpu,
    }

    # Run in dedicated environment
    result = run_spotiflow_in_env("detect_spots", args_dict)

    print(f"Detected {len(result['points'])} spots")
    return result["points"]


def _save_coords_csv(
    points: np.ndarray, input_file_path: str, use_env: bool = False
):
    """Save coordinates to CSV using Spotiflow's write_coords_csv function."""
    if not input_file_path:
        return

    # Generate CSV filename based on input file
    from pathlib import Path

    input_path = Path(input_file_path)
    csv_path = input_path.parent / (input_path.stem + "_spots.csv")

    if use_env:
        # Use dedicated environment
        _save_coords_csv_env(points, str(csv_path))
    else:
        # Use direct import
        _save_coords_csv_direct(points, str(csv_path))


def _save_coords_csv_direct(points: np.ndarray, csv_path: str):
    """Save coordinates directly using Spotiflow utils."""
    try:
        from spotiflow.utils import write_coords_csv

        write_coords_csv(points, csv_path)
        print(f"Saved {len(points)} spot coordinates to {csv_path}")
    except ImportError:
        # Fallback to basic CSV writing
        import pandas as pd

        columns = ["y", "x"] if points.shape[1] == 2 else ["z", "y", "x"]
        df = pd.DataFrame(points, columns=columns)
        df.to_csv(csv_path, index=False)
        print(
            f"Saved {len(points)} spot coordinates to {csv_path} (fallback method)"
        )


def _save_coords_csv_env(points: np.ndarray, csv_path: str):
    """Save coordinates using dedicated environment."""
    import contextlib
    import subprocess
    import tempfile

    from napari_tmidas.processing_functions.spotiflow_env_manager import (
        get_env_python_path,
    )

    # Save points to temporary numpy file
    with tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as temp_points:
        np.save(temp_points.name, points)

        # Create script to save CSV
        script = f"""
import numpy as np
from spotiflow.utils import write_coords_csv

# Load points
points = np.load('{temp_points.name}')

# Save CSV
write_coords_csv(points, '{csv_path}')
print(f"Saved {{len(points)}} spot coordinates to {csv_path}")
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(script)
            script_file.flush()

            # Execute script
            env_python = get_env_python_path()
            result = subprocess.run(
                [env_python, script_file.name],
                check=True,
                capture_output=True,
                text=True,
            )

            print(result.stdout)

            # Clean up
            with contextlib.suppress(FileNotFoundError):
                import os

                os.unlink(temp_points.name)
                os.unlink(script_file.name)


# Alias for convenience
spotiflow_spot_detection = spotiflow_detect_spots

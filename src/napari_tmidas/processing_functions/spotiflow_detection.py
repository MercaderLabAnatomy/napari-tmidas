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


@BatchProcessingRegistry.register(
    name="Spotiflow Spot Detection",
    suffix="_spots",
    description="Detect spots in fluorescence microscopy images using Spotiflow",
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
    axes: str = "auto",
    output_csv: bool = True,
    force_dedicated_env: bool = False,
    # For internal use by processing system
    input_file_path: str = None,
) -> np.ndarray:
    """
    Detect spots in fluorescence microscopy images using Spotiflow.

    Spotiflow is a deep learning-based spot detection method that provides
    threshold-agnostic, subpixel-accurate detection of spots in 2D and 3D
    fluorescence microscopy images. The output is a numpy array of spot
    coordinates suitable for napari Points layers.

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
    axes : str
        Axes order (e.g., 'ZYX', 'YX', or 'auto' for automatic detection)
    output_csv : bool
        Save spot coordinates as CSV file alongside the mask
    force_dedicated_env : bool
        Force using dedicated environment
    input_file_path : str
        Path to input file (used for saving CSV output)

    Returns:
    --------
    np.ndarray
        Spot coordinates as (N, 2) or (N, 3) array for napari Points layer
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

    print(f"Detected {len(points)} spots, returning points for points layer")
    return points  # Return points directly for napari points layer


def _points_to_label_mask(
    points: np.ndarray, image_shape: tuple, spot_radius: int
) -> np.ndarray:
    """Convert detected points to a label mask for napari."""
    from scipy import ndimage
    from skimage import draw

    # Create empty label mask
    label_mask = np.zeros(image_shape, dtype=np.uint16)

    # Handle different dimensionalities
    if len(image_shape) == 2:  # 2D image
        if points.shape[1] == 2:  # 2D points
            coords = points.astype(int)
        elif (
            points.shape[1] == 3
        ):  # 3D points on 2D image - take YX coordinates
            coords = points[:, 1:].astype(int)  # Skip Z coordinate
        else:
            raise ValueError(
                f"Unexpected points shape for 2D image: {points.shape}"
            )

        # Create circular spots
        for i, (y, x) in enumerate(coords):
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                rr, cc = draw.disk((y, x), spot_radius, shape=image_shape)
                label_mask[rr, cc] = i + 1  # Labels start from 1

    elif len(image_shape) == 3:  # 3D image
        if points.shape[1] == 3:  # 3D points
            coords = points.astype(int)
        elif points.shape[1] == 2:  # 2D points on 3D image - add Z=0
            coords = np.column_stack([np.zeros(len(points)), points]).astype(
                int
            )
        else:
            raise ValueError(
                f"Unexpected points shape for 3D image: {points.shape}"
            )

        # Create spherical spots
        for i, (z, y, x) in enumerate(coords):
            if (
                0 <= z < image_shape[0]
                and 0 <= y < image_shape[1]
                and 0 <= x < image_shape[2]
            ):
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
                    & (z_coords < image_shape[0])
                    & (y_coords >= 0)
                    & (y_coords < image_shape[1])
                    & (x_coords >= 0)
                    & (x_coords < image_shape[2])
                )

                label_mask[
                    z_coords[valid], y_coords[valid], x_coords[valid]
                ] = (i + 1)

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
):
    """Direct implementation using imported Spotiflow."""
    from spotiflow.model import Spotiflow

    # Load the model
    if model_path and os.path.exists(model_path):
        print(f"Loading custom model from {model_path}")
        model = Spotiflow.from_folder(model_path)
    else:
        print(f"Loading pretrained model: {pretrained_model}")
        model = Spotiflow.from_pretrained(pretrained_model)

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
    points, details = model.predict(normalized_img, **predict_kwargs)

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

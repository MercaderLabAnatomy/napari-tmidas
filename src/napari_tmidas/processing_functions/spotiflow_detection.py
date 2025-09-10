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
            "description": "Probability threshold (leave empty for automatic)",
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
            "default": 1,
            "min": 1,
            "max": 10,
            "description": "Minimum distance between detected spots",
        },
        "spot_radius": {
            "type": int,
            "default": 2,
            "min": 1,
            "max": 10,
            "description": "Radius of spots in the output label mask",
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
    min_distance: int = 1,
    spot_radius: int = 2,
    output_csv: bool = True,
    force_dedicated_env: bool = False,
    # For internal use by processing system
    input_file_path: str = None,
) -> np.ndarray:
    """
    Detect spots in fluorescence microscopy images using Spotiflow.

    Spotiflow is a deep learning-based spot detection method that provides
    threshold-agnostic, subpixel-accurate detection of spots in 2D and 3D
    fluorescence microscopy images. The output is a label mask where each
    spot is represented as a labeled region.

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
        Radius of spots in the output label mask
    output_csv : bool
        Save spot coordinates as CSV file alongside the mask
    force_dedicated_env : bool
        Force using dedicated environment
    input_file_path : str
        Path to input file (used for saving CSV output)

    Returns:
    --------
    np.ndarray
        Label mask with detected spots as labeled regions
    """
    print("Detecting spots using Spotiflow...")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")

    # Decide whether to use dedicated environment
    use_env = USE_DEDICATED_ENV or force_dedicated_env

    if not use_env and SPOTIFLOW_AVAILABLE:
        # Use direct import
        points = _detect_spots_direct(
            image,
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

    # Save CSV if requested and file path is available
    if output_csv and input_file_path:
        _save_coords_csv(points, input_file_path, use_env)

    print(f"Detected {len(points)} spots, returning points for points layer")
    return points  # Return points directly for napari points layer


def _detect_spots_direct(
    image,
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

    # Prepare normalizer function
    if normalizer == "percentile":
        from csbdeep.utils import normalize_mi_ma

        # Create a normalizer function that uses the specified percentiles
        def normalizer_func(img):
            p_low, p_high = np.percentile(
                img, [normalizer_low, normalizer_high]
            )
            return normalize_mi_ma(img, p_low, p_high)

        actual_normalizer = normalizer_func
    elif normalizer == "minmax":
        from csbdeep.utils import normalize_mi_ma

        def normalizer_func(img):
            return normalize_mi_ma(img, img.min(), img.max())

        actual_normalizer = normalizer_func
    else:
        actual_normalizer = "auto"

    # Prepare prediction parameters
    predict_kwargs = {
        "subpix": subpixel,  # Note: Spotiflow API uses 'subpix', not 'subpixel'
        "peak_mode": peak_mode,
        "normalizer": actual_normalizer,
        "exclude_border": exclude_border,
        "min_distance": min_distance,
        "device": "cpu",  # Force CPU for now to avoid GPU compatibility issues
    }

    # Add optional parameters
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if n_tiles_parsed is not None:
        predict_kwargs["n_tiles"] = n_tiles_parsed
    if scale_parsed is not None:
        predict_kwargs["scale"] = scale_parsed

    # Perform spot detection
    points, details = model.predict(image, **predict_kwargs)

    print(f"Detected {len(points)} spots")
    return points


def _detect_spots_env(
    image,
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

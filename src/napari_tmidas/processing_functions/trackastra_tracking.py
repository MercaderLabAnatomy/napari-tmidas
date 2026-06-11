#!/usr/bin/env python3
"""
TrackAstra Cell Tracking Module for napari-tmidas

This module integrates TrackAstra deep learning-based cell tracking into the
napari-tmidas batch processing framework. It uses a dedicated conda environment
to manage TrackAstra dependencies separately from the main environment.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imread

# Add the registry import
from napari_tmidas._registry import BatchProcessingRegistry


class TrackAstraEnvManager:
    """Manages the TrackAstra conda environment."""

    ENV_NAME = "trackastra"
    REQUIRED_VERSIONS = {
        "python": "3.10",
        "gurobipy": "13.0.0",
        "ilpy": "0.5.1",
        "motile": "0.4.0",
        "trackastra": "0.5.3",
    }

    @staticmethod
    def get_conda_cmd():
        """Get the conda/mamba command available on the system."""
        # Try mamba first (faster)
        if shutil.which("mamba"):
            return "mamba"
        elif shutil.which("conda"):
            return "conda"
        else:
            raise RuntimeError(
                "Neither conda nor mamba found. Please install Anaconda/Miniconda/Miniforge."
            )

    @staticmethod
    def check_env_exists():
        conda_cmd = TrackAstraEnvManager.get_conda_cmd()
        try:
            # Try running python --version in the env
            result = subprocess.run(
                [conda_cmd, "run", "-n", TrackAstraEnvManager.ENV_NAME, "python", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return False

    @staticmethod
    def _version_tuple(version_str):
        """Convert a version string to a comparable integer tuple."""
        if version_str is None:
            return tuple()
        nums = re.findall(r"\d+", str(version_str))
        if not nums:
            return tuple()
        return tuple(int(n) for n in nums)

    @staticmethod
    def _version_at_least(found, required):
        found_tuple = TrackAstraEnvManager._version_tuple(found)
        required_tuple = TrackAstraEnvManager._version_tuple(required)
        if not found_tuple or not required_tuple:
            return False

        # Compare only up to the required precision (e.g. 3.10, 0.5.1)
        found_norm = found_tuple[: len(required_tuple)]
        if len(found_norm) < len(required_tuple):
            found_norm = found_norm + (0,) * (len(required_tuple) - len(found_norm))
        return found_norm >= required_tuple

    @staticmethod
    def get_env_status():
        """Return package/version status for the TrackAstra environment."""
        conda_cmd = TrackAstraEnvManager.get_conda_cmd()
        check_script = r'''
import importlib.util
import json
import sys

status = {
    "python": sys.version.split()[0],
    "packages": {},
}

for name in ["gurobipy", "ilpy", "motile", "trackastra"]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        status["packages"][name] = {"present": False, "version": None}
        continue

    mod = __import__(name)
    if name == "gurobipy":
        version = ".".join(str(x) for x in mod.gurobi.version())
    else:
        version = getattr(mod, "__version__", None)
    status["packages"][name] = {"present": True, "version": version}

print(json.dumps(status))
'''
        try:
            result = subprocess.run(
                [
                    conda_cmd,
                    "run",
                    "-n",
                    TrackAstraEnvManager.ENV_NAME,
                    "python",
                    "-c",
                    check_script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            import json

            return json.loads(result.stdout.strip())
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def env_needs_repair(status):
        """Determine whether TrackAstra env should be repaired/upgraded."""
        if not status or "error" in status:
            return True, ["Could not determine environment status"]

        reasons = []

        python_version = status.get("python")
        if not TrackAstraEnvManager._version_at_least(
            python_version, TrackAstraEnvManager.REQUIRED_VERSIONS["python"]
        ):
            reasons.append(
                f"Python {python_version} < required {TrackAstraEnvManager.REQUIRED_VERSIONS['python']}"
            )

        packages = status.get("packages", {})
        for pkg in ["gurobipy", "ilpy", "motile", "trackastra"]:
            info = packages.get(pkg, {})
            if not info.get("present"):
                reasons.append(f"Missing package: {pkg}")
                continue
            found = info.get("version")
            required = TrackAstraEnvManager.REQUIRED_VERSIONS[pkg]
            if not TrackAstraEnvManager._version_at_least(found, required):
                reasons.append(f"{pkg} version {found} < required {required}")

        return (len(reasons) > 0), reasons

    @staticmethod
    def repair_env():
        """Repair/upgrade TrackAstra environment using upstream ILP recipe."""
        print("Repairing TrackAstra environment to required package versions...")
        conda_cmd = TrackAstraEnvManager.get_conda_cmd()
        try:
            # Keep solver stack aligned with TrackAstra ILP requirements.
            solver_cmd = [
                conda_cmd,
                "install",
                "-n",
                TrackAstraEnvManager.ENV_NAME,
                "-c",
                "conda-forge",
                "-c",
                "gurobi",
                "-c",
                "funkelab",
                "ilpy",
                "gurobi",
                "-y",
            ]
            subprocess.run(solver_cmd, check=True)

            pip_cmd = [
                conda_cmd,
                "run",
                "-n",
                TrackAstraEnvManager.ENV_NAME,
                "pip",
                "install",
                "--upgrade",
                "trackastra[ilp]",
                "motile",
            ]
            subprocess.run(pip_cmd, check=True)

            print("TrackAstra environment repair completed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error repairing TrackAstra environment: {e}")
            return False

    @staticmethod
    def create_env():
        """Create the TrackAstra conda environment if it doesn't exist."""
        if TrackAstraEnvManager.check_env_exists():
            print("TrackAstra environment already exists.")
            return True

        print("Creating TrackAstra conda environment...")
        conda_cmd = TrackAstraEnvManager.get_conda_cmd()

        # Create environment with Python 3.10 (required for TrackAstra)
        env_create_cmd = [
            conda_cmd,
            "create",
            "-n",
            TrackAstraEnvManager.ENV_NAME,
            "python=3.10",
            "--no-default-packages",
            "-y",
        ]

        try:
            subprocess.run(env_create_cmd, check=True)

            # Install ilpy first from conda-forge
            ilpy_cmd = [
                conda_cmd,
                "install",
                "-n",
                TrackAstraEnvManager.ENV_NAME,
                "-c",
                "conda-forge",
                "-c",
                "gurobi",
                "-c",
                "funkelab",
                "ilpy",
                "gurobi",
                "-y",
            ]
            subprocess.run(ilpy_cmd, check=True)

            # Install TrackAstra ILP extras via pip (upstream recipe).
            pip_packages = [
                "trackastra[ilp]",
                "motile",
            ]

            pip_cmd = [
                conda_cmd,
                "run",
                "-n",
                TrackAstraEnvManager.ENV_NAME,
                "pip",
                "install",
            ] + pip_packages

            subprocess.run(pip_cmd, check=True)

            print("TrackAstra environment created successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error creating TrackAstra environment: {e}")
            return False

    @staticmethod
    def ensure_env_ready():
        """Ensure environment exists and required package versions are present."""
        if not TrackAstraEnvManager.check_env_exists():
            print("TrackAstra environment not found. Creating it now...")
            if not TrackAstraEnvManager.create_env():
                return False

        status = TrackAstraEnvManager.get_env_status()
        needs_repair, reasons = TrackAstraEnvManager.env_needs_repair(status)
        if needs_repair:
            print("TrackAstra environment drift detected:")
            for reason in reasons:
                print(f" - {reason}")

            if not TrackAstraEnvManager.repair_env():
                return False

            # Recheck after attempted repair.
            status = TrackAstraEnvManager.get_env_status()
            needs_repair, reasons = TrackAstraEnvManager.env_needs_repair(status)
            if needs_repair:
                print("TrackAstra environment is still not healthy after repair:")
                for reason in reasons:
                    print(f" - {reason}")
                return False

        print("TrackAstra environment is ready.")
        return True


def create_trackastra_script(img_path, mask_path, model, mode, output_path, channel="all"):
    """Create a Python script to run TrackAstra in the dedicated environment."""
    
    # Determine if inputs are zarr files
    img_is_zarr = str(img_path).lower().endswith(".zarr")
    mask_is_zarr = str(mask_path).lower().endswith(".zarr")
    
    # Build image loading code based on file type
    if img_is_zarr:
        img_load_code = f"""
import zarr
print('Loading zarr image: {img_path}')
img_zarr = zarr.open('{img_path}', mode='r')
if hasattr(img_zarr, 'shape'):
    img = np.asarray(img_zarr)
else:
    # Multi-array zarr group, get first array
    arrays = list(img_zarr.array_keys())
    print(f'Available arrays: {{arrays}}')
    img = np.asarray(img_zarr[arrays[0]])
"""
    else:
        img_load_code = f"img = imread('{img_path}')"
    
    # Build mask loading code based on file type
    if mask_is_zarr:
        mask_load_code = f"""
import zarr
print('Loading zarr mask: {mask_path}')
mask_zarr = zarr.open('{mask_path}', mode='r')
if hasattr(mask_zarr, 'shape'):
    mask = np.asarray(mask_zarr)
else:
    # Multi-array zarr group, get first array
    arrays = list(mask_zarr.array_keys())
    print(f'Available arrays: {{arrays}}')
    mask = np.asarray(mask_zarr[arrays[0]])
"""
    else:
        mask_load_code = f"mask = imread('{mask_path}')"
    
    script_content = f"""
import sys
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks


# Load images
print('Loading images...')
{img_load_code}
{mask_load_code}
print(f'Img shape: {{img.shape}}, Mask shape: {{mask.shape}}')


# Validate and fix image dimensions
if mask.ndim not in [3, 4]:
    raise ValueError(f'Expected 3D (TYX) or 4D (TZYX) mask, got {{mask.ndim}}D')

if mask.shape[0] < 2:
    raise ValueError(f'Need at least 2 timepoints, got {{mask.shape[0]}}')

# Handle multichannel images (TCZYX) by selecting the specified channel
# If channel is "all" but the image is multichannel, take the first channel
channel_param = "{channel}"
if img.ndim == 5:
    # Input is TCZYX
    print(f'Input image is 5D {{img.shape}}, treating as TCZYX')
    if channel_param == "all":
        print('Channel="all" with multichannel image, taking first channel...')
        img = img[:, :, 0, :, :]  # Take first channel -> TZYX
    else:
        try:
            ch_idx = int(channel_param)
            print(f'Extracting channel {{ch_idx}}...')
            img = img[:, :, ch_idx, :, :]  # Take specified channel -> TZYX
        except (ValueError, IndexError):
            print(f'Invalid channel {{channel_param}}, taking first channel...')
            img = img[:, :, 0, :, :]
    print(f'Image shape after channel selection: {{img.shape}}')
elif img.ndim == 4 and mask.ndim == 4:
    # Both are 4D, check if dimensions match
    if img.shape != mask.shape:
        print(f'Warning: img and mask shapes differ: {{img.shape}} vs {{mask.shape}}')
        # If img is TCZYX and mask is TZYX, take first channel
        if img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1] and img.shape[3] == mask.shape[2]:
            print('Detected img as TCZYX, taking first channel...')
            img = img[:, :, 0, :, :]
elif img.ndim == 3 and mask.ndim == 3:
    # Both are 3D TYX, this is fine
    pass
else:
    print(f'Image ndim: {{img.ndim}}, Mask ndim: {{mask.ndim}}')

print(f'Final shapes - Img: {{img.shape}}, Mask: {{mask.shape}}')

model = Trackastra.from_pretrained('{model}', device="automatic")
try:
    track_result = model.track(img, mask, mode='{mode}')
except Exception as exc:
    if '{mode}' == 'ilp':
        print(f'Warning: TrackAstra ILP mode failed ({{exc}}). Retrying with greedy mode...')
        track_result = model.track(img, mask, mode='greedy')
    else:
        raise

# Trackastra API compatibility:
# - newer versions may return (track_graph, masks_tracked)
# - older versions return only track_graph
# Always run graph_to_ctc to enforce consistent track-based relabeling.
if isinstance(track_result, tuple):
    if len(track_result) < 1:
        raise RuntimeError('Unexpected empty tuple returned by Trackastra.track().')
    track_graph = track_result[0]
else:
    track_graph = track_result

if not hasattr(track_graph, 'nodes'):
    raise RuntimeError(
        f'Unexpected Trackastra.track() return type for graph: {{type(track_graph)}}'
    )

_, masks_tracked = graph_to_ctc(track_graph, mask, outdir=None)

same_as_input = np.array_equal(mask, masks_tracked)
in_ids = np.unique(mask)
out_ids = np.unique(masks_tracked)
print(
    f'Relabel check: identical_to_input={{same_as_input}}, '
    f'unique_ids_in={{len(in_ids)}}, unique_ids_out={{len(out_ids)}}'
)

# Save the tracked masks
imwrite('{output_path}', masks_tracked.astype(np.uint32), compression='zlib')
print(f'Saved tracked masks to: {output_path}')

"""

    return script_content


@BatchProcessingRegistry.register(
    name="Track Cells with Trackastra",
    suffix="_tracked",
    description="Track cells across time using TrackAstra deep learning (expects TYX or TZYX label images). Supports TIFF and zarr inputs with optional channel selection for multichannel images.",
    parameters={
        "model": {
            "type": str,
            "default": "ctc",
            "description": "general_2d (nuclei/cells/particles) or ctc (Cell Tracking Challenge; 2D/3D)",
        },
        "mode": {
            "type": str,
            "default": "greedy",
            "description": "greedy (fast), ilp (accurate with divisions), greedy_nodiv",
        },
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to process (automatically detected from multichannel images)",
        },
        "dimension_order": {
            "type": str,
            "default": "Auto",
            "options": ["Auto", "TYX", "TZYX", "TCZYX", "TCYX"],
            "description": "Dimension order hint for raw images (e.g., TCZYX for time-Z-channel-Y-X). Helps with channel detection when loading label files.",
        },
        "label_pattern": {
            "type": str,
            "default": "_labels.tif",
            "description": " ",
        },
    },
)
def trackastra_tracking(
    image: np.ndarray,
    model: str = "ctc",
    mode: str = "greedy",
    channel: str = "all",
    dimension_order: str = "Auto",
    label_pattern: str = "_labels.tif",
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = "_tracked",
) -> np.ndarray:
    """
    Track cells in time-lapse label images using TrackAstra.

    This function takes a time series of segmentation masks and performs
    automatic cell tracking using TrackAstra deep learning framework.

    Supports both TIFF and zarr input files. For multichannel images (TCZYX),
    automatically extracts the specified channel before tracking.

    Expected input dimensions:
    - TYX: Time series of 2D label images
    - TZYX: Time series of 3D label images (will process each Z-slice separately)
    - TCZYX: Multichannel time series (channel selection via 'channel' parameter)

    Input file formats:
    - TIFF (.tif, .tiff files)
    - Zarr (.zarr directories, including OME-Zarr with multiple arrays)

    Parameters:
    -----------
    image : np.ndarray
        Input label image array with time as first dimension
    model : str
        TrackAstra model: 'general_2d' or 'ctc' (default: "ctc")
    mode : str
        Tracking mode: 'greedy', 'ilp', or 'greedy_nodiv' (default: "greedy")
    channel : str
        Channel selection: "all" or specific channel number (default: "all")
    dimension_order : str
        Dimension order hint for raw images (e.g., "TCZYX", "TZYX"). If "Auto",
        dimensions are auto-detected. This helps clarify channel detection when
        processing label files.
    label_pattern : str
        To identify label images

    Returns:
    --------
    np.ndarray
        Tracked label image with consistent IDs across time
    """
    print(f"Input shape: {image.shape}, dtype: {image.dtype}")

    # Validate input
    if image.ndim < 3:
        print(
            "Input is not a time series (needs at least 3 dimensions). Returning unchanged."
        )
        return image

    if image.shape[0] < 2:
        print(
            "Input has only one timepoint. Need at least 2 for tracking. Returning unchanged."
        )
        return image

    # Ensure TrackAstra environment exists and has compatible package versions.
    if not TrackAstraEnvManager.ensure_env_ready():
        print("Failed to prepare TrackAstra environment. Returning unchanged.")
        return image

    # Get source file path. Prefer explicit worker-provided path.
    img_path = _source_filepath
    if img_path is None:
        import inspect

        for frame_info in inspect.stack():
            frame_locals = frame_info.frame.f_locals
            if "filepath" in frame_locals:
                img_path = frame_locals["filepath"]
                break

    if img_path is None:
        print("Could not determine input file path. Returning unchanged.")

    temp_dir = Path(os.path.dirname(img_path))

    # Create the tracking script
    script_path = temp_dir / "run_tracking.py"
    # Save the mask data
    # For label images, use the original path as mask_path
    if label_pattern in os.path.basename(img_path):
        mask_path = img_path
        # Find corresponding raw image by removing the label pattern
        raw_base = os.path.basename(img_path).replace(label_pattern, "")
        raw_path = os.path.join(os.path.dirname(img_path), raw_base + ".tif")
        if not os.path.exists(raw_path):
            print(f"Warning: Could not find raw image for {img_path}")
            raw_path = img_path  # Fallback to using label as input
        else:
            # Reload the raw image instead of the label image that was passed in
            # This ensures channel detection/extraction happens on the correct file
            print(f"Processing label file: reloading raw image for channel handling")
            try:
                image = imread(raw_path)
                print(f"Raw image reloaded: shape={image.shape}, dtype={image.dtype}")
            except Exception as e:
                print(f"Warning: failed to reload raw image ({e}), using original image")
    else:
        # For raw images, find the corresponding label image
        raw_path = img_path
        base_name = os.path.basename(img_path).replace(".tif", "")
        mask_path = os.path.join(
            os.path.dirname(img_path), base_name + label_pattern
        )
        if not os.path.exists(mask_path):
            print(f"No label file found for {img_path}")
            return image

    mask_filename = os.path.basename(mask_path)
    if mask_filename.endswith(label_pattern):
        output_filename = (
            mask_filename[: -len(label_pattern)] + f"{_output_suffix}.tif"
        )
    else:
        output_filename = (
            os.path.splitext(mask_filename)[0] + f"{_output_suffix}.tif"
        )

    output_dir = Path(_output_folder) if _output_folder else temp_dir
    output_path = output_dir / output_filename

    script_content = create_trackastra_script(
        str(raw_path), str(mask_path), model, mode, str(output_path), channel
    )

    with open(script_path, "w") as f:
        f.write(script_content)

    # Run TrackAstra in the dedicated environment
    conda_cmd = TrackAstraEnvManager.get_conda_cmd()
    cmd = [
        conda_cmd,
        "run",
        "-n",
        TrackAstraEnvManager.ENV_NAME,
        "python",
        str(script_path),
    ]
    print(f"Running TrackAstra with model='{model}', mode='{mode}'...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("TrackAstra error:")
        print(result.stdout)
        print(result.stderr)
        print("Returning original image unchanged.")
        return image

    print(result.stdout)

    # Return the produced path so the processing worker does not save a
    # second suffixed copy based on the current input filename.
    if output_path.exists():
        print(f"Tracking completed. Output saved at: {output_path}")
        os.remove(script_path)
        return str(output_path)
    else:
        print("TrackAstra did not produce output. Returning unchanged.")
        return image

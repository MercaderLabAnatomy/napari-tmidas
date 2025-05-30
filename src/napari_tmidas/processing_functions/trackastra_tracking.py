#!/usr/bin/env python3
"""
TrackAstra Cell Tracking Module for napari-tmidas

This module integrates TrackAstra deep learning-based cell tracking into the
napari-tmidas batch processing framework. It uses a dedicated conda environment
to manage TrackAstra dependencies separately from the main environment.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imread

# Add the registry import
from napari_tmidas._registry import BatchProcessingRegistry


class TrackAstraEnvManager:
    """Manages the TrackAstra conda environment."""

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
                [conda_cmd, "run", "-n", "trackastra", "python", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
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
            "trackastra",
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
                "trackastra",
                "-c",
                "conda-forge",
                "-c",
                "gurobi",
                "-c",
                "funkelab",
                "ilpy",
                "-y",
            ]
            subprocess.run(ilpy_cmd, check=True)

            # Install TrackAstra and other dependencies via pip
            pip_packages = [
                "trackastra[napari]",
                "scikit-image",
                "tifffile",
                "torch",
                "torchvision",
            ]

            pip_cmd = [
                conda_cmd,
                "run",
                "-n",
                "trackastra",
                "pip",
                "install",
            ] + pip_packages

            subprocess.run(pip_cmd, check=True)

            print("TrackAstra environment created successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error creating TrackAstra environment: {e}")
            return False


def create_trackastra_script(img_path, mask_path, model, mode, output_path):
    """Create a Python script to run TrackAstra in the dedicated environment."""
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
img = imread('{img_path}')
mask = imread('{mask_path}')
print(f'Img shape: {{img.shape}}, Mask shape: {{mask.shape}}')


# Validate dimensions
if mask.ndim not in [3, 4]:
    raise ValueError(f'Expected 3D (TYX) or 4D (TZYX) mask, got {{mask.ndim}}D')

if mask.shape[0] < 2:
    raise ValueError(f'Need at least 2 timepoints, got {{mask.shape[0]}}')

model = Trackastra.from_pretrained('{model}', device="automatic")
track_graph = model.track(img, mask, mode='{mode}')
_, masks_tracked = graph_to_ctc(track_graph, mask, outdir=None)

# Save the tracked masks
imwrite('{output_path}', masks_tracked.astype(np.uint32), compression='zlib')
print(f'Saved tracked masks to: {output_path}')

"""

    return script_content


@BatchProcessingRegistry.register(
    name="Track Cells with Trackastra",
    suffix="_tracked",
    description="Track cells across time using TrackAstra deep learning (expects TYX or TZYX label images)",
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
    label_pattern: str = "_labels.tif",
) -> np.ndarray:
    """
    Track cells in time-lapse label images using TrackAstra.

    This function takes a time series of segmentation masks and performs
    automatic cell tracking using TrackAstra deep learning framework.

    Expected input dimensions:
    - TYX: Time series of 2D label images
    - TZYX: Time series of 3D label images (will process each Z-slice separately)

    Parameters:
    -----------
    image : np.ndarray
        Input label image array with time as first dimension
    model : str
        TrackAstra model: 'general_2d' or 'ctc' (default: "ctc")
    mode : str
        Tracking mode: 'greedy', 'ilp', or 'greedy_nodiv' (default: "greedy")
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

    # Ensure TrackAstra environment exists
    if not TrackAstraEnvManager.check_env_exists():
        print("TrackAstra environment not found. Creating it now...")
        if not TrackAstraEnvManager.create_env():
            print(
                "Failed to create TrackAstra environment. Returning unchanged."
            )
            return image

    # Get the current file path from the processing context
    import inspect

    img_path = None

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
        # For raw images, find the corresponding label image
        raw_path = img_path
        base_name = os.path.basename(img_path).replace(".tif", "")
        mask_path = os.path.join(
            os.path.dirname(img_path), base_name + label_pattern
        )
        if not os.path.exists(mask_path):
            print(f"No label file found for {img_path}")
            return image

    output_path = temp_dir / os.path.basename(mask_path).replace(
        label_pattern, "_tracked.tif"
    )

    script_content = create_trackastra_script(
        str(raw_path), str(mask_path), model, mode, str(output_path)
    )

    with open(script_path, "w") as f:
        f.write(script_content)

    if label_pattern in img_path:
        pass
    else:
        # Run TrackAstra in the dedicated environment
        conda_cmd = TrackAstraEnvManager.get_conda_cmd()
        cmd = [
            conda_cmd,
            "run",
            "-n",
            "trackastra",
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

    # Load and return the tracked result
    if output_path.exists():
        tracked = imread(str(output_path))
        print(f"Tracking completed. Output shape: {tracked.shape}")
        os.remove(script_path)
        return tracked
    else:
        print("TrackAstra did not produce output. Returning unchanged.")
        return image

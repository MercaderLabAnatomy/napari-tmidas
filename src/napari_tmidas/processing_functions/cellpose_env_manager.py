# processing_functions/cellpose_env_manager.py
"""
This module manages a dedicated virtual environment for Cellpose.
Updated to support Cellpose 4 (Cellpose-SAM) installation.
"""

import os
import subprocess
import tempfile
from contextlib import suppress

import tifffile

from napari_tmidas._env_manager import BaseEnvironmentManager


class CellposeEnvironmentManager(BaseEnvironmentManager):
    """Environment manager for Cellpose."""

    def __init__(self):
        super().__init__("cellpose")

    def _install_dependencies(self, env_python: str) -> None:
        """Install Cellpose-specific dependencies."""
        # Install cellpose 4 and other dependencies
        print(
            "Installing Cellpose 4 (Cellpose-SAM) in the dedicated environment..."
        )
        subprocess.check_call([env_python, "-m", "pip", "install", "cellpose"])

        # Check if installation was successful
        try:
            # Run a command to check if cellpose can be imported and GPU is available
            result = subprocess.run(
                [
                    env_python,
                    "-c",
                    "from cellpose import core; print(f'GPU available: {core.use_gpu()}')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print("Cellpose installation successful:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not verify Cellpose installation: {e}")

    def is_package_installed(self) -> bool:
        """Check if cellpose is installed in the current environment."""
        try:
            import importlib.util

            return importlib.util.find_spec("cellpose") is not None
        except ImportError:
            return False


# Global instance for backward compatibility
manager = CellposeEnvironmentManager()


def is_cellpose_installed():
    """Check if cellpose is installed in the current environment."""
    return manager.is_package_installed()


def is_env_created():
    """Check if the dedicated environment exists."""
    return manager.is_env_created()


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    return manager.get_env_python_path()


def create_cellpose_env():
    """Create a dedicated virtual environment for Cellpose."""
    return manager.create_env()


def run_cellpose_in_env(func_name, args_dict):
    """
    Run Cellpose in a dedicated environment with minimal complexity.

    Parameters:
    -----------
    func_name : str
        Name of the Cellpose function to run (currently unused)
    args_dict : dict
        Dictionary of arguments for Cellpose segmentation

    Returns:
    --------
    numpy.ndarray
        Segmentation masks
    """
    # Ensure the environment exists
    if not is_env_created():
        create_cellpose_env()

    # Prepare temporary files
    with tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as input_file, tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as output_file, tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:

        # Save input image
        tifffile.imwrite(input_file.name, args_dict["image"])

        # Prepare a temporary script to run Cellpose
        # Updated to use Cellpose 4 parameters
        script = f"""
import numpy as np
from cellpose import models, core
import tifffile

# Load image
image = tifffile.imread('{input_file.name}')

# Create and run model
model = models.CellposeModel(
    gpu={args_dict.get('use_gpu', True)})

# Prepare normalization parameters (Cellpose 4)
normalize = {args_dict.get('normalize', {'tile_norm_blocksize': 128})}

# Perform segmentation with Cellpose 4 parameters
masks, flows, styles = model.eval(
    image,
    channels={args_dict.get('channels', [0, 0])},
    flow_threshold={args_dict.get('flow_threshold', 0.4)},
    cellprob_threshold={args_dict.get('cellprob_threshold', 0.0)},
    batch_size={args_dict.get('batch_size', 32)},
    normalize=normalize,
    do_3D={args_dict.get('do_3D', False)},
    flow3D_smooth={args_dict.get('flow3D_smooth', 0)},
    anisotropy={args_dict.get('anisotropy', None)},
    z_axis={args_dict.get('z_axis', 0)} if {args_dict.get('do_3D', False)} else None
)

# Save results
tifffile.imwrite('{output_file.name}', masks)
"""

        # Write script
        script_file.write(script)
        script_file.flush()

    try:
        # Run the script in the dedicated environment
        env_python = get_env_python_path()
        result = subprocess.run(
            [env_python, script_file.name], capture_output=True, text=True
        )
        print("Stdout:", result.stdout)
        # Check for errors
        if result.returncode != 0:
            print("Stderr:", result.stderr)
            raise RuntimeError(
                f"Cellpose segmentation failed: {result.stderr}"
            )

        # Read and return the results
        return tifffile.imread(output_file.name)

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error in Cellpose segmentation: {e}")
        raise

    finally:
        # Clean up temporary files
        for fname in [input_file.name, output_file.name, script_file.name]:
            with suppress(OSError, FileNotFoundError):
                os.unlink(fname)

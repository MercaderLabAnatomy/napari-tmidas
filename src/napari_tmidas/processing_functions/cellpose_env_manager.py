# processing_functions/cellpose_env_manager.py
"""
This module manages a dedicated virtual environment for Cellpose.
"""

import os
import platform
import shutil
import subprocess
import tempfile
import venv

import tifffile

# Define the environment directory in user's home folder
ENV_DIR = os.path.join(
    os.path.expanduser("~"), ".napari-tmidas", "envs", "cellpose"
)


def is_cellpose_installed():
    """Check if cellpose is installed in the current environment."""
    try:
        import importlib.util

        return importlib.util.find_spec("cellpose") is not None
    except ImportError:
        return False


def is_env_created():
    """Check if the dedicated environment exists."""
    env_python = get_env_python_path()
    return os.path.exists(env_python)


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    if platform.system() == "Windows":
        return os.path.join(ENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(ENV_DIR, "bin", "python")


def create_cellpose_env():
    """Create a dedicated virtual environment for Cellpose."""
    # Ensure the environment directory exists
    os.makedirs(os.path.dirname(ENV_DIR), exist_ok=True)

    # Remove existing environment if it exists
    if os.path.exists(ENV_DIR):
        shutil.rmtree(ENV_DIR)

    print(f"Creating Cellpose environment at {ENV_DIR}...")

    # Create a new virtual environment
    venv.create(ENV_DIR, with_pip=True)

    # Path to the Python executable in the new environment
    env_python = get_env_python_path()

    # Install numpy first to ensure correct version
    print("Installing NumPy...")
    subprocess.check_call(
        [env_python, "-m", "pip", "install", "numpy>=1.24,<1.25"]
    )

    # Install cellpose and other dependencies
    print("Installing Cellpose in the dedicated environment...")
    subprocess.check_call([env_python, "-m", "pip", "install", "cellpose"])

    # Install tifffile for image handling
    subprocess.check_call([env_python, "-m", "pip", "install", "tifffile"])

    print("Cellpose environment created successfully.")
    return env_python


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
        script = f"""
import numpy as np
from cellpose import models
import tifffile

# Load image
image = tifffile.imread('{input_file.name}')

# Create and run model
model = models.Cellpose(
    gpu={args_dict.get('use_gpu', True)},
    model_type='{args_dict.get('model_type', 'cyto3')}'
)

# Perform segmentation
masks, *_ = model.eval(
    image,
    diameter={args_dict.get('diameter', 30.0)},
    flow_threshold={args_dict.get('flow_threshold', 0.4)},
    channels={args_dict.get('channels', [0, 0])},
    do_3D={args_dict.get('do_3D', False)},
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

        # Check for errors
        if result.returncode != 0:
            print("Stdout:", result.stdout)
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
        # Clean up temporary files using contextlib.suppress
        from contextlib import suppress

        for fname in [input_file.name, output_file.name, script_file.name]:
            with suppress(OSError, FileNotFoundError):
                os.unlink(fname)

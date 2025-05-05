"""
processing_functions/sam2_env_manager.py

This module manages a dedicated virtual environment for SAM2.
"""

import os
import platform
import shutil
import subprocess
import venv

# Define the environment directory in user's home folder
ENV_DIR = os.path.join(
    os.path.expanduser("~"), ".napari-tmidas", "envs", "sam2-env"
)


def is_sam2_installed():
    """Check if SAM2 is installed in the current environment."""
    try:
        import importlib.util

        return importlib.util.find_spec("sam2-env") is not None
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


def create_sam2_env():
    """Create a dedicated virtual environment for SAM2."""
    # Ensure the environment directory exists
    os.makedirs(os.path.dirname(ENV_DIR), exist_ok=True)

    # Remove existing environment if it exists
    if os.path.exists(ENV_DIR):
        shutil.rmtree(ENV_DIR)

    print(f"Creating SAM2 environment at {ENV_DIR}...")

    # Create a new virtual environment
    venv.create(ENV_DIR, with_pip=True)

    # Path to the Python executable in the new environment
    env_python = get_env_python_path()

    # Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call(
        [env_python, "-m", "pip", "install", "--upgrade", "pip"]
    )

    # Install numpy and torch first for compatibility
    print("Installing torch and torchvision...")
    subprocess.check_call(
        [env_python, "-m", "pip", "install", "torch", "torchvision"]
    )

    # Install sam2 from GitHub
    print("Installing SAM2 from GitHub...")
    subprocess.check_call(
        [
            env_python,
            "-m",
            "pip",
            "install",
            "git+https://github.com/facebookresearch/sam2.git",
        ]
    )

    subprocess.run(
        [
            env_python,
            "-c",
            "import torch; import torchvision; print('PyTorch version:', torch.__version__); print('Torchvision version:', torchvision.__version__); print('CUDA is available:', torch.cuda.is_available())",
        ]
    )

    print("SAM2 environment created successfully.")
    return env_python


def run_sam2_in_env(func_name, args_dict):
    """
    Run SAM2 in a dedicated environment with minimal complexity.

    Parameters:
    -----------
    func_name : str
        Name of the SAM2 function to run (currently unused)
    args_dict : dict
        Dictionary of arguments for SAM2

    Returns:
    --------
    numpy.ndarray
        Segmentation masks
    """

"""
processing_functions/sam2_env_manager.py

This module manages a dedicated virtual environment for SAM2.
"""

import subprocess

from napari_tmidas._env_manager import BaseEnvironmentManager


class SAM2EnvironmentManager(BaseEnvironmentManager):
    """Environment manager for SAM2."""

    def __init__(self):
        super().__init__("sam2-env")

    def _install_dependencies(self, env_python: str) -> None:
        """Install SAM2-specific dependencies."""
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

    def is_package_installed(self) -> bool:
        """Check if SAM2 is installed in the current environment."""
        try:
            import importlib.util

            return importlib.util.find_spec("sam2") is not None
        except ImportError:
            return False


# Global instance for backward compatibility
manager = SAM2EnvironmentManager()


def is_sam2_installed():
    """Check if SAM2 is installed in the current environment."""
    return manager.is_package_installed()


def is_env_created():
    """Check if the dedicated environment exists."""
    return manager.is_env_created()


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    return manager.get_env_python_path()


def create_sam2_env():
    """Create a dedicated virtual environment for SAM2."""
    return manager.create_env()


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

"""
processing_functions/sam2_env_manager.py

This module manages a dedicated virtual environment for SAM2.
"""

import os
import subprocess

from napari_tmidas._env_manager import BaseEnvironmentManager


class SAM2EnvironmentManager(BaseEnvironmentManager):
    """Environment manager for SAM2."""

    def __init__(self):
        super().__init__("sam2-env")
        self.sam2_repo_dir = os.path.join(self.env_dir, "sam2_repo")
        self.checkpoints_dir = os.path.join(self.sam2_repo_dir, "checkpoints")

    def _install_dependencies(self, env_python: str) -> None:
        """Install SAM2-specific dependencies including cloning the repository."""
        # Install numpy and torch first for compatibility
        print("Installing torch and torchvision...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "torch", "torchvision"]
        )

        # Clone SAM2 repository into the environment directory
        print(f"Cloning SAM2 repository to {self.sam2_repo_dir}...")
        if os.path.exists(self.sam2_repo_dir):
            print("SAM2 repository already exists, pulling latest changes...")
            subprocess.check_call(
                ["git", "-C", self.sam2_repo_dir, "pull"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://github.com/facebookresearch/sam2.git",
                    self.sam2_repo_dir,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Install SAM2 from the cloned repository
        print("Installing SAM2 package from cloned repository...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "-e", self.sam2_repo_dir]
        )

        # Download model checkpoint
        self._download_model_checkpoint(env_python)

        # Verify installation
        subprocess.run(
            [
                env_python,
                "-c",
                "import torch; import torchvision; import sam2; "
                "print('PyTorch version:', torch.__version__); "
                "print('Torchvision version:', torchvision.__version__); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('SAM2 version:', sam2.__version__ if hasattr(sam2, '__version__') else 'installed')",
            ]
        )

    def _download_model_checkpoint(self, env_python: str) -> None:
        """Download SAM2 model checkpoint if not already present."""
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        model_path = os.path.join(
            self.checkpoints_dir, "sam2.1_hiera_large.pt"
        )

        if os.path.exists(model_path):
            print(f"Model checkpoint already exists at {model_path}")
            return

        print(
            f"Downloading SAM2 model checkpoint (~850 MB) to {model_path}..."
        )

        # Use Python to download with progress
        download_script = f"""
import urllib.request
import sys

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    sys.stdout.write(f'\\rDownload progress: {{percent:.1f}}%')
    sys.stdout.flush()

urllib.request.urlretrieve(
    '{model_url}',
    '{model_path}',
    reporthook=report_progress
)
print('\\nModel download complete!')
"""

        try:
            subprocess.check_call([env_python, "-c", download_script])
        except subprocess.CalledProcessError:
            print("Failed to download model. You can manually download from:")
            print(f"  {model_url}")
            print(f"  to {model_path}")

    def is_package_installed(self) -> bool:
        """Check if SAM2 is installed in the environment."""
        env_python = self.get_env_python_path()
        if not os.path.exists(env_python):
            return False

        try:
            result = subprocess.run(
                [env_python, "-c", "import sam2"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
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

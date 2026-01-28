# processing_functions/viscy_env_manager.py
"""
This module manages a dedicated virtual environment for VisCy (Virtual Staining of Cells using deep learning).
"""

import contextlib
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from napari_tmidas._env_manager import BaseEnvironmentManager

try:
    import tifffile
except ImportError:
    tifffile = None


class ViscyEnvironmentManager(BaseEnvironmentManager):
    """Environment manager for VisCy."""

    def __init__(self):
        super().__init__("viscy")
        # Model directory in the environment
        self.model_dir = os.path.join(self.env_dir, "models")

    def _install_dependencies(self, env_python: str) -> None:
        """Install VisCy-specific dependencies."""
        # Try to detect if CUDA is available
        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print("CUDA is available in main environment")
                if torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"GPU detected: {gpu_name}")
            else:
                print("CUDA is not available in main environment")
        except ImportError:
            print("PyTorch not detected in main environment")
            # Try to detect CUDA from nvidia-smi
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    cuda_available = True
                    print("NVIDIA GPU detected via nvidia-smi")
                else:
                    cuda_available = False
                    print("No NVIDIA GPU detected")
            except FileNotFoundError:
                cuda_available = False
                print("nvidia-smi not found, assuming no CUDA support")

        if cuda_available:
            print("Attempting PyTorch installation with CUDA support...")
            try:
                subprocess.check_call(
                    [
                        env_python,
                        "-m",
                        "pip",
                        "install",
                        "torch==2.0.1",
                        "torchvision==0.15.2",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu118",
                    ]
                )
                print("✓ PyTorch with CUDA 11.8 installed successfully")

                # Test CUDA compatibility
                test_script = """
import torch
try:
    if torch.cuda.is_available():
        test_tensor = torch.ones(1).cuda()
        print("CUDA compatibility test passed")
    else:
        print("CUDA not available in PyTorch")
        exit(1)
except Exception as e:
    print(f"CUDA compatibility test failed: {e}")
    exit(1)
"""
                result = subprocess.run(
                    [env_python, "-c", test_script],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print(
                        "⚠ CUDA test failed, falling back to CPU-only installation"
                    )
                    print(result.stdout)
                    print(result.stderr)
                    cuda_available = False

            except subprocess.CalledProcessError as e:
                print(
                    f"⚠ PyTorch CUDA installation failed: {e}, falling back to CPU-only"
                )
                cuda_available = False

        if not cuda_available:
            print("Installing PyTorch (CPU-only version)...")
            subprocess.check_call(
                [
                    env_python,
                    "-m",
                    "pip",
                    "install",
                    "torch==2.0.1",
                    "torchvision==0.15.2",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )
            print("✓ PyTorch (CPU-only) installed successfully")

        # Install VisCy and dependencies
        print("Installing VisCy and dependencies...")
        subprocess.check_call(
            [
                env_python,
                "-m",
                "pip",
                "install",
                "viscy",
                "iohub",
                "tifffile",
                "numpy",
            ]
        )

        print("✓ VisCy and dependencies installed successfully")

        # Download the VSCyto3D model checkpoint
        print("Downloading VSCyto3D model checkpoint...")
        self._download_model()

    def _download_model(self):
        """Download the VSCyto3D checkpoint if not already present."""
        os.makedirs(self.model_dir, exist_ok=True)

        checkpoint_path = os.path.join(self.model_dir, "VSCyto3D.ckpt")

        if not os.path.exists(checkpoint_path):
            print("Downloading VSCyto3D model...")
            url = "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto3D/epoch=83-step=14532-loss=0.492.ckpt"
            try:
                urllib.request.urlretrieve(url, checkpoint_path)
                print(f"✓ Model checkpoint downloaded to {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Failed to download model checkpoint: {e}")
                print(
                    "You can manually download it from the URL above and place it in:"
                )
                print(f"  {checkpoint_path}")
        else:
            print(f"✓ Model checkpoint already exists at {checkpoint_path}")

    def is_package_installed(self) -> bool:
        """Check if VisCy is installed in the environment."""
        if not self.is_env_created():
            return False

        env_python = self.get_env_python_path()
        try:
            subprocess.check_call(
                [env_python, "-c", "import viscy"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_model_path(self) -> str:
        """Get the path to the VSCyto3D model checkpoint."""
        return os.path.join(self.model_dir, "VSCyto3D.ckpt")


# Singleton instance
_viscy_env_manager = ViscyEnvironmentManager()


def create_viscy_env() -> str:
    """Create the VisCy environment."""
    return _viscy_env_manager.create_env()


def is_env_created() -> bool:
    """Check if the VisCy environment exists."""
    return _viscy_env_manager.is_env_created()


def is_viscy_installed() -> bool:
    """Check if VisCy is installed."""
    return _viscy_env_manager.is_package_installed()


def get_model_path() -> str:
    """Get the path to the VSCyto3D model checkpoint."""
    return _viscy_env_manager.get_model_path()


def run_viscy_in_env(image: np.ndarray, z_batch_size: int = 15) -> np.ndarray:
    """
    Run VisCy virtual staining in the dedicated environment.

    Parameters:
    -----------
    image : np.ndarray
        Input image with shape (Z, Y, X)
    z_batch_size : int
        Number of Z slices to process at once (default: 15, required by VSCyto3D)

    Returns:
    --------
    np.ndarray
        Virtual stained output with shape (Z, 2, Y, X) where channels are:
        - Channel 0: Nuclei
        - Channel 1: Membrane
    """
    if not is_env_created():
        raise RuntimeError(
            "VisCy environment not created. Please create it first."
        )

    if not is_viscy_installed():
        raise RuntimeError(
            "VisCy not installed in environment. Please create the environment first."
        )

    env_python = _viscy_env_manager.get_env_python_path()
    model_path = get_model_path()

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model checkpoint not found at {model_path}. Please re-create the environment."
        )

    # Create a temporary file for input
    with tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as input_file:
        input_path = input_file.name
        if tifffile is not None:
            tifffile.imwrite(input_path, image)
        else:
            raise ImportError("tifffile is required but not available")

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as output_file:
        output_path = output_file.name

    try:
        # Create Python script to run in the environment
        script = f"""
import numpy as np
import torch
import tifffile
from viscy.translation.engine import VSUNet

# Load the model
model = VSUNet.load_from_checkpoint(
    "{model_path}",
    architecture="fcmae",
    model_config={{
        "in_channels": 1,
        "out_channels": 2,
        "encoder_blocks": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
        "decoder_conv_blocks": 2,
        "stem_kernel_size": [5, 4, 4],
        "in_stack_depth": 15,
        "pretraining": False
    }}
)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

# Load input image
image = tifffile.imread("{input_path}")
n_z = image.shape[0]
z_batch_size = {z_batch_size}
n_batches = (n_z + z_batch_size - 1) // z_batch_size

all_predictions = []

for batch_idx in range(n_batches):
    start_z = batch_idx * z_batch_size
    end_z = min((batch_idx + 1) * z_batch_size, n_z)
    
    # Get batch
    batch_data = image[start_z:end_z]
    actual_size = batch_data.shape[0]
    
    # Pad if necessary
    if actual_size < z_batch_size:
        pad_size = z_batch_size - actual_size
        batch_data = np.pad(batch_data, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    
    # Normalize
    p_low, p_high = np.percentile(batch_data, [1, 99])
    batch_data = np.clip((batch_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
    
    # Convert to tensor: (Z, Y, X) -> (1, 1, Z, Y, X)
    batch_tensor = torch.from_numpy(batch_data.astype(np.float32))[None, None, :, :, :]
    if torch.cuda.is_available():
        batch_tensor = batch_tensor.cuda()
    
    # Run prediction
    with torch.no_grad():
        pred = model(batch_tensor)  # (1, 2, Z, Y, X)
    
    # Process output: (2, Z, Y, X) -> (Z, 2, Y, X)
    pred_np = pred[0].cpu().numpy().transpose(1, 0, 2, 3)[:actual_size]
    all_predictions.append(pred_np)
    
    # Free memory
    del batch_data, batch_tensor, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Concatenate all predictions: (Z, 2, Y, X)
full_prediction = np.concatenate(all_predictions, axis=0)

# Save output
tifffile.imwrite("{output_path}", full_prediction)
"""

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_path = script_file.name
            script_file.write(script)

        # Run the script in the environment
        result = subprocess.run(
            [env_python, script_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"VisCy processing failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        # Load the output
        if tifffile is not None:
            output_image = tifffile.imread(output_path)
        else:
            raise ImportError("tifffile is required but not available")

        return output_image

    finally:
        # Clean up temporary files
        with contextlib.suppress(Exception):
            os.unlink(input_path)
        with contextlib.suppress(Exception):
            os.unlink(output_path)
        with contextlib.suppress(Exception):
            os.unlink(script_path)

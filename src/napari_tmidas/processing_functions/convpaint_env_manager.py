# processing_functions/convpaint_env_manager.py
"""
This module manages a dedicated virtual environment for napari-convpaint.
"""

import os
import subprocess
import tempfile

import tifffile

from napari_tmidas._env_manager import BaseEnvironmentManager


class ConvpaintEnvironmentManager(BaseEnvironmentManager):
    """Environment manager for napari-convpaint."""

    def __init__(self):
        super().__init__("convpaint")

    def _install_dependencies(self, env_python: str) -> None:
        """Install napari-convpaint-specific dependencies."""
        # Try to detect if CUDA is available
        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            print(
                f"CUDA is {'available' if cuda_available else 'not available'}"
            )
        except ImportError:
            print("PyTorch not detected in main environment")

        if cuda_available:
            # Install PyTorch with CUDA support
            print("Installing PyTorch with CUDA support...")
            subprocess.check_call(
                [
                    env_python,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ]
            )
        else:
            # Install PyTorch without CUDA
            print("Installing PyTorch without CUDA support...")
            subprocess.check_call(
                [env_python, "-m", "pip", "install", "torch", "torchvision"]
            )

        # Install napari-convpaint and dependencies
        print("Installing napari-convpaint in the dedicated environment...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "napari-convpaint"]
        )

        # Install tifffile and other dependencies for image handling
        subprocess.check_call(
            [
                env_python,
                "-m",
                "pip",
                "install",
                "tifffile",
                "numpy",
                "scikit-image",
            ]
        )

        # Check if installation was successful
        self._verify_installation(env_python)

    def _verify_installation(self, env_python: str) -> None:
        """Verify napari-convpaint installation."""
        check_script = """
import sys
try:
    from napari_convpaint.conv_paint_model import ConvpaintModel
    print("ConvpaintModel imported successfully")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("SUCCESS: napari-convpaint environment is working correctly")
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp:
            temp.write(check_script)
            temp_path = temp.name

        try:
            result = subprocess.run(
                [env_python, temp_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if "SUCCESS" in result.stdout:
                print(
                    "napari-convpaint environment created and verified successfully."
                )
            else:
                print(
                    "WARNING: napari-convpaint environment created but verification uncertain."
                )
        finally:
            os.unlink(temp_path)

    def is_package_installed(self) -> bool:
        """Check if napari-convpaint is installed in the current environment."""
        try:
            import importlib.util

            return importlib.util.find_spec("napari_convpaint") is not None
        except ImportError:
            return False


# Global instance for backward compatibility
manager = ConvpaintEnvironmentManager()


def is_convpaint_installed():
    """Check if napari-convpaint is installed in the current environment."""
    return manager.is_package_installed()


def is_env_created():
    """Check if the dedicated environment exists."""
    return manager.is_env_created()


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    return manager.get_env_python_path()


def create_convpaint_env():
    """Create a dedicated virtual environment for napari-convpaint."""
    return manager.create_env()


def recreate_convpaint_env():
    """Delete and recreate the napari-convpaint environment (e.g., for version updates)."""
    print("Recreating napari-convpaint environment...")
    manager.delete_env()
    return manager.create_env()


def run_convpaint_in_env(image, model_path, image_downsample=2, use_cpu=False):
    """
    Run convpaint prediction in the dedicated environment.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image to segment
    model_path : str
        Path to the pretrained convpaint model (.pkl file)
    image_downsample : int
        Downsampling factor for processing (default: 2)
    use_cpu : bool
        Force CPU execution even if GPU is available (default: False)

    Returns:
    --------
    numpy.ndarray
        Segmentation labels
    """
    if not is_env_created():
        print("Creating dedicated napari-convpaint environment...")
        create_convpaint_env()

    env_python = get_env_python_path()

    # Create temporary files for input/output
    with tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as input_file:
        input_path = input_file.name
        tifffile.imwrite(input_path, image)

    output_path = input_path.replace(".tif", "_output.tif")

    # Create the script to run in the environment
    script = f"""
import numpy as np
import tifffile
from napari_convpaint.conv_paint_model import ConvpaintModel
import gc
import torch
import os

# Force CPU mode if requested
use_cpu = {use_cpu}
if use_cpu:
    print("Forcing CPU execution (GPU disabled)")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    fe_use_gpu = False
else:
    # Check if GPU is available and compatible
    fe_use_gpu = torch.cuda.is_available()
    if fe_use_gpu:
        try:
            # Try a simple CUDA operation to check compatibility
            test_tensor = torch.randn(2, 2, device='cuda')
            _ = test_tensor * 2
            print("GPU available and compatible")
        except RuntimeError as e:
            if "no kernel image is available" in str(e) or "CUDA" in str(e):
                print(f"GPU detected but not compatible with PyTorch: {{e}}")
                print("Falling back to CPU execution")
                fe_use_gpu = False
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            else:
                raise

# Load model with appropriate GPU setting
print("Loading model from: {model_path}")
model = ConvpaintModel(model_path="{model_path}")
print("Model loaded successfully")

# If forcing CPU, update the model's GPU setting
if use_cpu:
    model._param.fe_use_gpu = False
    # Move model to CPU if it's on GPU
    if hasattr(model, 'fe_model') and hasattr(model.fe_model, 'device'):
        if 'cuda' in str(model.fe_model.device):
            model.fe_model.device = torch.device('cpu')
            model.fe_model.model = model.fe_model.model.cpu()
            print("  Moved feature extractor model to CPU")

print(f"  Model has classifier: {{model.classifier is not None}}")
print(f"  Model device: {{model.fe_model.device}}")
print(f"  GPU enabled: {{model._param.fe_use_gpu}}")

# Set downsampling if needed
if {image_downsample} > 1:
    model.set_params(
        image_downsample={image_downsample},
        tile_annotations=False,
        ignore_warnings=True
    )
    print(f"Downsampling set to: {{model._param.image_downsample}}x")

# Load image
print("Loading image from: {input_path}")
image = tifffile.imread("{input_path}")
print(f"Image shape: {{image.shape}}, dtype: {{image.dtype}}")

# Segment
print("Running segmentation...")
segmentation = model.segment(image)
print(f"Segmentation shape: {{segmentation.shape}}")

# Remove singleton dimensions if present
segmentation = np.squeeze(segmentation)
print(f"Final segmentation shape: {{segmentation.shape}}")

# Save output
print("Saving output to: {output_path}")
tifffile.imwrite("{output_path}", segmentation.astype(np.uint32), compression="zlib")

# Clear memory
del image, segmentation, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

print("Segmentation complete")
"""

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_path = script_file.name
        script_file.write(script)

    try:
        # Run the script in the environment
        result = subprocess.run(
            [env_python, script_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Convpaint processing failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        # Print output for debugging
        if result.stdout:
            print(result.stdout)

        # Load the output
        output_image = tifffile.imread(output_path)

        return output_image

    finally:
        # Clean up temporary files
        try:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(script_path)
        except (OSError, FileNotFoundError):
            pass

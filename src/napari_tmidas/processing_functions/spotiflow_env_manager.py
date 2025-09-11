# processing_functions/spotiflow_env_manager.py
"""
This module manages a dedicated virtual environment for Spotiflow.
"""

import contextlib
import os
import subprocess
import tempfile

import numpy as np

from napari_tmidas._env_manager import BaseEnvironmentManager

try:
    import tifffile
except ImportError:
    tifffile = None


class SpotiflowEnvironmentManager(BaseEnvironmentManager):
    """Environment manager for Spotiflow."""

    def __init__(self):
        super().__init__("spotiflow")

    def _install_dependencies(self, env_python: str) -> None:
        """Install Spotiflow-specific dependencies."""
        # Install PyTorch first for compatibility
        # Try to detect if CUDA is available and GPU architecture
        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print("CUDA is available in main environment")
                # Try to get GPU info
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
            # Install PyTorch with CUDA support for older GPUs (includes sm_61 for GTX 1080 Ti)
            print(
                "Installing PyTorch with CUDA support for older GPU architectures..."
            )
            # Use PyTorch with CUDA 11.8 which supports sm_61 (GTX 1080 Ti) and other older GPUs
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
        else:
            # Install PyTorch without CUDA
            print("Installing PyTorch without CUDA support...")
            subprocess.check_call(
                [
                    env_python,
                    "-m",
                    "pip",
                    "install",
                    "torch==2.0.1",
                    "torchvision==0.15.2",
                ]
            )

        # Install Spotiflow with all dependencies, but force CPU usage to avoid GPU issues
        print("Installing Spotiflow in the dedicated environment...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "spotiflow"]
        )

        # Install additional dependencies for image handling
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "tifffile", "numpy"]
        )

        # Check if installation was successful
        self._verify_installation(env_python)

    def _verify_installation(self, env_python: str) -> None:
        """Verify Spotiflow installation."""
        check_script = """
import sys
try:
    import spotiflow
    print(f"Spotiflow version: {spotiflow.__version__}")
    from spotiflow.model import Spotiflow
    print("Spotiflow model imported successfully")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("SUCCESS: Spotiflow environment is working correctly")
except Exception as e:
    print(f"ERROR: {str(e)}")
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
                    "Spotiflow environment created and verified successfully."
                )
            else:
                raise RuntimeError(
                    "Spotiflow environment verification failed."
                )
        except subprocess.CalledProcessError as e:
            print(f"Verification failed: {e.stderr}")
            raise
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)

    def is_package_installed(self) -> bool:
        """Check if spotiflow is installed in the current environment."""
        try:
            import importlib.util

            return importlib.util.find_spec("spotiflow") is not None
        except ImportError:
            return False


# Global instance for backward compatibility
manager = SpotiflowEnvironmentManager()


def is_spotiflow_installed():
    """Check if spotiflow is installed in the current environment."""
    return manager.is_package_installed()


def is_env_created():
    """Check if the dedicated environment exists."""
    return manager.is_env_created()


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    return manager.get_env_python_path()


def create_spotiflow_env():
    """Create a dedicated virtual environment for Spotiflow."""
    return manager.create_env()


def run_spotiflow_in_env(func_name, args_dict):
    """
    Run Spotiflow in a dedicated environment.

    Parameters:
    -----------
    func_name : str
        Name of the Spotiflow function to run
    args_dict : dict
        Dictionary of arguments for Spotiflow prediction

    Returns:
    --------
    numpy.ndarray or tuple
        Detection results (points coordinates and optionally heatmap/flow)
    """
    # Ensure the environment exists
    if not is_env_created():
        create_spotiflow_env()

    # Prepare temporary files
    with tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as input_file, tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as output_file, tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:

        # Save input image
        tifffile.imwrite(input_file.name, args_dict["image"])

        # Prepare a temporary script to run Spotiflow
        script = f"""
import numpy as np
import os
import sys
print("Starting Spotiflow detection script...")
print(f"Python version: {{sys.version}}")

try:
    from spotiflow.model import Spotiflow
    print("✓ Spotiflow model imported successfully")
except Exception as e:
    print(f"✗ Failed to import Spotiflow model: {{e}}")
    sys.exit(1)

try:
    import tifffile
    print("✓ tifffile imported successfully")
except Exception as e:
    print(f"✗ Failed to import tifffile: {{e}}")
    sys.exit(1)

try:
    # Load image
    print(f"Loading image from: {input_file.name}")
    image = tifffile.imread('{input_file.name}')
    print(f"✓ Image loaded successfully, shape: {{image.shape}}, dtype: {{image.dtype}}")
except Exception as e:
    print(f"✗ Failed to load image: {{e}}")
    sys.exit(1)

try:
    # Load the model
    if '{args_dict.get('model_path', '')}' and os.path.exists('{args_dict.get('model_path', '')}'):
        # Load custom model from folder
        print(f"Loading custom model from {args_dict.get('model_path', '')}")
        model = Spotiflow.from_folder('{args_dict.get('model_path', '')}')
    else:
        # Load pretrained model
        print(f"Loading pretrained model: {args_dict.get('pretrained_model', 'general')}")
        model = Spotiflow.from_pretrained('{args_dict.get('pretrained_model', 'general')}')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {{e}}")
    sys.exit(1)

# Utility functions for input preparation
def _validate_axes(img, axes):
    if img.ndim != len(axes):
        raise ValueError(f"Image has {{img.ndim}} dimensions, but axes has {{len(axes)}} dimensions")

def _prepare_input(img, axes):
    _validate_axes(img, axes)
    if axes in {{"YX", "ZYX", "TYX", "TZYX"}}:
        return img[..., None]
    elif axes in {{"YXC", "ZYXC", "TYXC", "TZYXC"}}:
        return img
    elif axes == "CYX":
        return img.transpose(1, 2, 0)
    elif axes == "CZYX":
        return img.transpose(1, 2, 3, 0)
    elif axes == "ZCYX":
        return img.transpose(0, 2, 3, 1)
    elif axes == "TCYX":
        return img.transpose(0, 2, 3, 1)
    elif axes == "TZCYX":
        return img.transpose(0, 1, 3, 4, 2)
    elif axes == "TCZYX":
        return img.transpose(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Invalid axes: {{axes}}")

try:
    # Handle axes and input preparation
    axes = '{args_dict.get('axes', 'auto')}'
    if axes == 'auto':
        # Auto-infer axes
        ndim = image.ndim
        if ndim == 2:
            axes = "YX"
        elif ndim == 3:
            axes = "ZYX"
        elif ndim == 4:
            if image.shape[-1] <= 4:
                axes = "ZYXC"
            else:
                axes = "TZYX"
        elif ndim == 5:
            axes = "TZYXC"
        else:
            raise ValueError(f"Cannot infer axes for {{ndim}}D image")

    print(f"Using axes: {{axes}}")

    # Prepare input
    prepared_img = _prepare_input(image, axes)
    print(f"Prepared image shape: {{prepared_img.shape}}")

    # Check model compatibility
    is_3d_image = len(image.shape) == 3 and "Z" in axes
    if is_3d_image and not model.config.is_3d:
        print("Warning: Using a 2D model on 3D data. Consider using a 3D model.")

except Exception as e:
    print(f"✗ Failed to prepare input: {{e}}")
    # Fallback to original image
    prepared_img = image
    axes = "YX" if image.ndim == 2 else "ZYX"

try:
    # Parse string parameters
    def parse_param(param_str, default_val):
        if param_str == "auto":
            return default_val
        try:
            return eval(param_str) if param_str.startswith("(") else param_str
        except:
            return default_val

    n_tiles_parsed = parse_param('{args_dict.get('n_tiles', 'auto')}', None)
    scale_parsed = parse_param('{args_dict.get('scale', 'auto')}', None)

    # Handle normalization manually (similar to napari-spotiflow)
    normalizer_type = '{args_dict.get('normalizer', 'percentile')}'
    if normalizer_type == "percentile":
        normalizer_low = {args_dict.get('normalizer_low', 1.0)}
        normalizer_high = {args_dict.get('normalizer_high', 99.8)}
        print(f"Applying percentile normalization: {{normalizer_low}}% to {{normalizer_high}}%")
        p_low, p_high = np.percentile(prepared_img, [normalizer_low, normalizer_high])
        normalized_img = np.clip((prepared_img - p_low) / (p_high - p_low), 0, 1)
    elif normalizer_type == "minmax":
        print("Applying min-max normalization")
        img_min, img_max = prepared_img.min(), prepared_img.max()
        normalized_img = (prepared_img - img_min) / (img_max - img_min) if img_max > img_min else prepared_img
    else:
        normalized_img = prepared_img

    print(f"Normalized image range: {{normalized_img.min():.3f}} to {{normalized_img.max():.3f}}")

    # Prepare prediction parameters (following napari-spotiflow style)
    predict_kwargs = {{
        'subpix': {args_dict.get('subpixel', True)},  # Note: Spotiflow API uses 'subpix', not 'subpixel'
        'peak_mode': '{args_dict.get('peak_mode', 'fast')}',
        'normalizer': None,  # We handle normalization manually
        'exclude_border': {args_dict.get('exclude_border', True)},
        'min_distance': {args_dict.get('min_distance', 2)},
        'verbose': True,
    }}

    # Set probability threshold - use automatic or provided value
    prob_thresh = {args_dict.get('prob_thresh', None)}
    if prob_thresh is not None and prob_thresh > 0.0:
        predict_kwargs['prob_thresh'] = prob_thresh
    # If prob_thresh is None or 0.0, don't set it - let spotiflow use automatic threshold

    if n_tiles_parsed is not None:
        predict_kwargs['n_tiles'] = n_tiles_parsed
    if scale_parsed is not None:
        predict_kwargs['scale'] = scale_parsed

    print(f"Prediction parameters: {{predict_kwargs}}")
except Exception as e:
    print(f"✗ Failed to prepare parameters: {{e}}")
    sys.exit(1)

try:
    # Perform spot detection
    print("Running Spotiflow prediction...")
    points, details = model.predict(normalized_img, **predict_kwargs)
    print(f"✓ Initial detection: {{len(points)}} spots")

    # Only apply minimal additional filtering if we still have too many detections
    # This should rarely be needed now that we use proper automatic thresholding
    if len(points) > 500:  # Only if we have an excessive number of spots
        print(f"Applying additional filtering for {{len(points)}} spots")

        # Check if we can apply probability filtering
        if hasattr(details, 'prob'):
            # Use a more stringent threshold
            auto_thresh = 0.7
            prob_mask = details.prob > auto_thresh
            points = points[prob_mask]
            print(f"After additional probability thresholding ({{auto_thresh}}): {{len(points)}} spots")

    print(f"Final detection: {{len(points)}} spots")

    if len(points) > 0:
        print(f"✓ Points shape: {{points.shape}}")
        print(f"✓ Points dtype: {{points.dtype}}")
        print(f"✓ First few points: {{points[:3]}}")

except Exception as e:
    print(f"✗ Failed during spot detection: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Prepare output data
    output_data = {{
        'points': points,
    }}

    # Save results
    print(f"Saving results to: {output_file.name}")
    np.save('{output_file.name}', output_data)
    print(f"✓ Results saved successfully")
    print(f"Detected {{len(points)}} spots")
except Exception as e:
    print(f"✗ Failed to save results: {{e}}")
    sys.exit(1)
"""

        # Write script
        script_file.write(script)
        script_file.flush()

        # Execute the script in the dedicated environment
        env_python = get_env_python_path()
        result = subprocess.run(
            [env_python, script_file.name],
            capture_output=True,
            text=True,
        )

        # Check for errors
        if result.returncode != 0:
            print("Error in Spotiflow environment execution:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )

        print(result.stdout)

        # Load and return results
        output_data = np.load(output_file.name, allow_pickle=True).item()

        # Clean up temporary files
        with contextlib.suppress(FileNotFoundError):
            os.unlink(input_file.name)
            os.unlink(output_file.name)
            os.unlink(script_file.name)

        return output_data

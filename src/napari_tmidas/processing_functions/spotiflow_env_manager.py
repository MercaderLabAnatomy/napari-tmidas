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

    # Prepare normalizer function
    normalizer_type = '{args_dict.get('normalizer', 'percentile')}'
    if normalizer_type == "percentile":
        normalizer_low = {args_dict.get('normalizer_low', 1.0)}
        normalizer_high = {args_dict.get('normalizer_high', 99.8)}
        from csbdeep.utils import normalize_mi_ma
        import numpy as np
        # Create a normalizer function that uses the specified percentiles
        def normalizer_func(img):
            p_low, p_high = np.percentile(img, [normalizer_low, normalizer_high])
            return normalize_mi_ma(img, p_low, p_high)
        actual_normalizer = normalizer_func
    elif normalizer_type == "minmax":
        from csbdeep.utils import normalize_mi_ma
        def normalizer_func(img):
            return normalize_mi_ma(img, img.min(), img.max())
        actual_normalizer = normalizer_func
    else:
        actual_normalizer = "auto"

    # Prepare prediction parameters
    predict_kwargs = {{
        'subpix': {args_dict.get('subpixel', True)},  # Note: Spotiflow API uses 'subpix', not 'subpixel'
        'peak_mode': '{args_dict.get('peak_mode', 'fast')}',
        'normalizer': actual_normalizer,
        'exclude_border': {args_dict.get('exclude_border', True)},
        'min_distance': {args_dict.get('min_distance', 1)},
        'device': 'cpu',  # Force CPU for now to avoid GPU compatibility issues
    }}

    # Add optional parameters
    prob_thresh = {args_dict.get('prob_thresh', None)}
    if prob_thresh is not None:
        predict_kwargs['prob_thresh'] = prob_thresh
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
    print("Starting spot detection...")
    points, details = model.predict(image, **predict_kwargs)
    print(f"✓ Spot detection completed successfully")
    print(f"✓ Detected {{len(points)}} spots")

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

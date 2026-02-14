# processing_functions/careamics_env_manager.py
"""
This module manages a dedicated virtual environment for CAREamics.
"""

import contextlib
import os
import subprocess
import tempfile

import tifffile

from napari_tmidas._env_manager import BaseEnvironmentManager


class CAREamicsEnvironmentManager(BaseEnvironmentManager):
    """Environment manager for CAREamics."""

    def __init__(self):
        super().__init__("careamics")

    def _install_dependencies(self, env_python: str) -> None:
        """Install CAREamics-specific dependencies."""
        # Install PyTorch first for compatibility
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
                [env_python, "-m", "pip", "install", "torch", "torchvision"]
            )
        else:
            # Install PyTorch without CUDA
            print("Installing PyTorch without CUDA support...")
            subprocess.check_call(
                [env_python, "-m", "pip", "install", "torch", "torchvision"]
            )

        # Install CAREamics and dependencies
        print("Installing CAREamics in the dedicated environment...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "careamics[tensorboard]"]
        )

        # Install tifffile for image handling
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "tifffile", "numpy"]
        )

        # Check if installation was successful
        self._verify_installation(env_python)

    def _verify_installation(self, env_python: str) -> None:
        """Verify CAREamics installation."""
        check_script = """
import sys
try:
    import careamics
    print(f"CAREamics version: {careamics.__version__}")
    from careamics import CAREamist
    print("CAREamist imported successfully")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("SUCCESS: CAREamics environment is working correctly")
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
                    "CAREamics environment created and verified successfully."
                )
            else:
                print(
                    "WARNING: CAREamics environment created but verification uncertain."
                )
        finally:
            os.unlink(temp_path)

    def is_package_installed(self) -> bool:
        """Check if CAREamics is installed in the current environment."""
        try:
            import importlib.util

            return importlib.util.find_spec("careamics") is not None
        except ImportError:
            return False


# Global instance for backward compatibility
manager = CAREamicsEnvironmentManager()


def is_careamics_installed():
    """Check if CAREamics is installed in the current environment."""
    return manager.is_package_installed()


def is_env_created():
    """Check if the dedicated environment exists."""
    return manager.is_env_created()


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    return manager.get_env_python_path()


def create_careamics_env():
    """Create a dedicated virtual environment for CAREamics."""
    return manager.create_env()


def recreate_careamics_env():
    """Delete and recreate the CAREamics environment (e.g., for version updates)."""
    print("Recreating CAREamics environment...")
    manager.delete_env()
    return manager.create_env()


def run_careamics_in_env(func_name, args_dict, retry_count=0):
    """
    Run CAREamics in a dedicated environment.

    Parameters:
    -----------
    func_name : str
        Name of the CAREamics function to run (e.g., 'predict')
    args_dict : dict
        Dictionary of arguments for CAREamics function
    retry_count : int
        Number of retry attempts (used internally)

    Returns:
    --------
    numpy.ndarray
        Denoised image
    """
    # Ensure the environment exists
    if not is_env_created():
        create_careamics_env()

    # Prepare temporary files
    with tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as input_file, tempfile.NamedTemporaryFile(
        suffix=".tif", delete=False
    ) as output_file, tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:

        # Save input image to temp file
        tifffile.imwrite(
            input_file.name, args_dict["image"], compression="zlib"
        )

        # Create a temporary script to run CAREamics
        script = f"""
import sys
import os
import numpy as np
import tifffile
from pathlib import Path

try:
    from careamics import CAREamist

    # Load input image
    print(f"Loading image from {{os.path.basename('{input_file.name}')}}")
    image = tifffile.imread('{input_file.name}')
    print(f"Input image shape: {{image.shape}}, dtype: {{image.dtype}}")

    # Load model
    print(f"Loading model from {{os.path.basename('{args_dict['checkpoint_path']}')}}")
    model = CAREamist('{args_dict['checkpoint_path']}')

    # Determine dimensionality and format
    dims = len(image.shape)
    print(f"Input: {{{{image.shape}}}} ({{'TZYX' if dims == 4 else 'TYX' if dims == 3 and image.shape[0] <= 10 else 'ZYX' if dims == 3 else 'YX'}})")

    # Handle different data formats with progress reporting
    if dims == 2:
        # YX - 2D image
        tile_size = ({args_dict.get('tile_size_y', 64)},
                     {args_dict.get('tile_size_x', 64)})
        tile_overlap = ({args_dict.get('tile_overlap_y', 32)},
                       {args_dict.get('tile_overlap_x', 32)})
        
        print("Processing 2D image...")
        prediction = model.predict(
            source=image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size={args_dict.get('batch_size', 1)},
            tta={args_dict.get('use_tta', True)},
        )
        prediction = np.squeeze(prediction)
        
    elif dims == 3 and image.shape[0] <= 10:
        # TYX - 2D with time (iterate through timepoints)
        t_size = image.shape[0]
        print(f"Processing 2D+time (T={{{{t_size}}}})...")
        
        tile_size = ({args_dict.get('tile_size_y', 64)},
                     {args_dict.get('tile_size_x', 64)})
        tile_overlap = ({args_dict.get('tile_overlap_y', 32)},
                       {args_dict.get('tile_overlap_x', 32)})
        
        # Pre-allocate result array
        prediction = np.zeros_like(image)
        
        # Process each timepoint
        for t in range(t_size):
            print(f"  T={{t+1}}/{{t_size}}...", end=" ", flush=True)
            
            pred_t = model.predict(
                source=image[t],
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                batch_size={args_dict.get('batch_size', 1)},
                tta={args_dict.get('use_tta', True)},
            )
            prediction[t] = np.squeeze(pred_t)
            print("\u2713")
            
    elif dims == 3:
        # ZYX - 3D image
        print(f"Processing 3D (Z={{{{image.shape[0]}}}})...")
        
        tile_size = ({args_dict.get('tile_size_z', 64)},
                     {args_dict.get('tile_size_y', 64)},
                     {args_dict.get('tile_size_x', 64)})
        tile_overlap = ({args_dict.get('tile_overlap_z', 32)},
                       {args_dict.get('tile_overlap_y', 32)},
                       {args_dict.get('tile_overlap_x', 32)})
        
        prediction = model.predict(
            source=image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size={args_dict.get('batch_size', 1)},
            tta={args_dict.get('use_tta', True)},
        )
        prediction = np.squeeze(prediction)
        
    elif dims == 4:
        # TZYX - 3D with time (iterate through timepoints)
        t_size = image.shape[0]
        print(f"Processing 3D+time (T={{{{t_size}}}}, Z={{{{image.shape[1]}}}})...")
        
        tile_size = ({args_dict.get('tile_size_z', 64)},
                     {args_dict.get('tile_size_y', 64)},
                     {args_dict.get('tile_size_x', 64)})
        tile_overlap = ({args_dict.get('tile_overlap_z', 32)},
                       {args_dict.get('tile_overlap_y', 32)},
                       {args_dict.get('tile_overlap_x', 32)})
        
        # Pre-allocate result array
        prediction = np.zeros_like(image)
        
        # Process each timepoint
        for t in range(t_size):
            print(f"  T={{{{t+1}}}}/{{{{t_size}}}}...", end=" ", flush=True)
            
            pred_t = model.predict(
                source=image[t],
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                batch_size={args_dict.get('batch_size', 1)},
                tta={args_dict.get('use_tta', True)},
            )
            prediction[t] = np.squeeze(pred_t)
            print("\u2713")
    else:
        raise ValueError(f"Unsupported data shape: {{{{image.shape}}}}")

    print(f"Output: {{{{prediction.shape}}}}")

    # Save result
    print(f"Saving result to {{os.path.basename('{output_file.name}')}}")
    tifffile.imwrite('{output_file.name}', prediction, compression="zlib")
    print("Done!")

    # Exit with success code
    sys.exit(0)

except Exception as e:
    import traceback
    print(f"Error in CAREamics processing: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""

        # Write script
        script_file.write(script)
        script_file.flush()

    try:
        # Run the script in the dedicated environment with real-time output
        env_python = get_env_python_path()
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            [env_python, script_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output while streaming it
        stdout_lines = []
        stderr_lines = []
        
        # Stream stdout in real-time
        for line in process.stdout:
            print(line, end='')
            stdout_lines.append(line)
        
        # Wait for process to complete and get stderr
        process.wait()
        stderr_output = process.stderr.read()
        if stderr_output:
            stderr_lines.append(stderr_output)
        
        # Check for errors
        if process.returncode != 0:
            error_output = stderr_output + ''.join(stdout_lines)
            
            # Check if this is a version mismatch error
            is_version_mismatch = (
                "Lightning v" in error_output and "newer than" in error_output
            ) or (
                "size mismatch" in error_output and retry_count == 0
            )
            
            if is_version_mismatch and retry_count == 0:
                print("\n⚠️  Detected version mismatch between checkpoint and environment")
                print("🔄 Recreating environment with updated dependencies...\n")
                recreate_careamics_env()
                print("\n✓ Environment recreated. Retrying...\n")
                # Retry once with the new environment
                return run_careamics_in_env(func_name, args_dict, retry_count=1)
            
            print("Error in CAREamics processing:")
            if stderr_output:
                print(stderr_output)
            raise RuntimeError(
                f"CAREamics denoising failed with error code {process.returncode}"
            )

        # Read and return the results
        denoised_image = tifffile.imread(output_file.name)
        return denoised_image

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error in CAREamics processing: {e}")
        # Return original image in case of error
        return args_dict["image"]

    finally:
        # Clean up temporary files
        for file_path in [input_file.name, output_file.name, script_file.name]:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                os.unlink(file_path)

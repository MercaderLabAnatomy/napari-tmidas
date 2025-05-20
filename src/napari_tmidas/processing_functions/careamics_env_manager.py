# processing_functions/careamics_env_manager.py
"""
This module manages a dedicated virtual environment for CAREamics.
"""

import contextlib
import os
import platform
import shutil
import subprocess
import tempfile
import venv

import tifffile

# Define the environment directory in user's home folder
ENV_DIR = os.path.join(
    os.path.expanduser("~"), ".napari-tmidas", "envs", "careamics"
)


def is_careamics_installed():
    """Check if CAREamics is installed in the current environment."""
    try:
        import importlib.util

        return importlib.util.find_spec("careamics") is not None
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


def create_careamics_env():
    """Create a dedicated virtual environment for CAREamics."""
    # Ensure the environment directory exists
    os.makedirs(os.path.dirname(ENV_DIR), exist_ok=True)

    # Remove existing environment if it exists
    if os.path.exists(ENV_DIR):
        shutil.rmtree(ENV_DIR)

    print(f"Creating CAREamics environment at {ENV_DIR}...")

    # Create a new virtual environment
    venv.create(ENV_DIR, with_pip=True)

    # Path to the Python executable in the new environment
    env_python = get_env_python_path()

    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.check_call(
        [env_python, "-m", "pip", "install", "--upgrade", "pip"]
    )

    # Install PyTorch first for compatibility
    # Try to detect if CUDA is available
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA is {'available' if cuda_available else 'not available'}")
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
    try:
        # Run a simple script to verify CAREamics is installed and working
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

    except subprocess.CalledProcessError as e:
        print(f"Error during CAREamics installation: {e}")
        print(f"Output: {e.output}")
        print(f"Errors: {e.stderr}")
        raise RuntimeError("Failed to create CAREamics environment") from e

    return env_python


def run_careamics_in_env(func_name, args_dict):
    """
    Run CAREamics in a dedicated environment.

    Parameters:
    -----------
    func_name : str
        Name of the CAREamics function to run (e.g., 'predict')
    args_dict : dict
        Dictionary of arguments for CAREamics function

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
    model = CAREamist('{args_dict['checkpoint_path']}',os.path.dirname('{args_dict['checkpoint_path']}'))

    # Determine dimensionality
    dims = len(image.shape)
    is_3d = dims >= 3

    if is_3d:
        print(f"Processing 3D data with shape: {{image.shape}}")
        # Determine axes based on dimensionality
        if dims == 3:
            # ZYX format
            axes = "ZYX"
            tile_size = ({args_dict.get('tile_size_z', 64)},
                         {args_dict.get('tile_size_y', 64)},
                         {args_dict.get('tile_size_x', 64)})
            tile_overlap = ({args_dict.get('tile_overlap_z', 32)},
                           {args_dict.get('tile_overlap_y', 32)},
                           {args_dict.get('tile_overlap_x', 32)})
        elif dims == 4:
            # Assuming TZYX format
            axes = "TZYX"
            tile_size = ({args_dict.get('tile_size_z', 64)},
                         {args_dict.get('tile_size_y', 64)},
                         {args_dict.get('tile_size_x', 64)})
            tile_overlap = ({args_dict.get('tile_overlap_z', 32)},
                           {args_dict.get('tile_overlap_y', 32)},
                           {args_dict.get('tile_overlap_x', 32)})
        else:
            print(f"Warning: Unusual data shape: {{image.shape}}. Defaulting to 'TZYX'")
            axes = "TZYX"
            tile_size = ({args_dict.get('tile_size_z', 64)},
                         {args_dict.get('tile_size_y', 64)},
                         {args_dict.get('tile_size_x', 64)})
            tile_overlap = ({args_dict.get('tile_overlap_z', 32)},
                           {args_dict.get('tile_overlap_y', 32)},
                           {args_dict.get('tile_overlap_x', 32)})
    else:
        print(f"Processing 2D data with shape: {{image.shape}}")
        # 2D data
        if dims == 2:
            # YX format
            axes = "YX"
            tile_size = ({args_dict.get('tile_size_y', 64)},
                         {args_dict.get('tile_size_x', 64)})
            tile_overlap = ({args_dict.get('tile_overlap_y', 32)},
                           {args_dict.get('tile_overlap_x', 32)})
        else:
            print(f"Warning: Unusual data shape: {{image.shape}}. Defaulting to 'YX'")
            axes = "YX"
            tile_size = ({args_dict.get('tile_size_y', 64)},
                         {args_dict.get('tile_size_x', 64)})
            tile_overlap = ({args_dict.get('tile_overlap_y', 32)},
                           {args_dict.get('tile_overlap_x', 32)})

    print(f"Using axes: {{axes}}, tile_size: {{tile_size}}, tile_overlap: {{tile_overlap}}")

    # Run the prediction
    print("Starting CAREamics prediction...")
    prediction = model.predict(
        source=image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes=axes,
        batch_size={args_dict.get('batch_size', 1)},
        tta={args_dict.get('use_tta', True)},
    )

    # # Handle output shape
    # if prediction.shape != image.shape:
    #     print(f"Warning: Prediction shape {{prediction.shape}} differs from input shape {{image.shape}}")
    #     prediction = np.squeeze(prediction)

    #     # If shapes still don't match, try to reshape
    #     if prediction.shape != image.shape:
    #         print(f"Warning: Shapes still don't match after squeezing. Using original dimensions.")

    # print(f"Denoising completed. Output shape: {{prediction.shape}}")

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
        # Run the script in the dedicated environment
        env_python = get_env_python_path()
        result = subprocess.run(
            [env_python, script_file.name], capture_output=True, text=True
        )

        # Print output
        print(result.stdout)

        # Check for errors
        if result.returncode != 0:
            print("Error in CAREamics processing:")
            print(result.stderr)
            raise RuntimeError(
                f"CAREamics denoising failed with error code {result.returncode}"
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

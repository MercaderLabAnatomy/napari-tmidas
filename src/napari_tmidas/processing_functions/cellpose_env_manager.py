# processing_functions/cellpose_env_manager.py
"""
This module manages a dedicated virtual environment for Cellpose.
Updated to support Cellpose 4 (Cellpose-SAM) installation.
"""

import os
import subprocess
import sys
import tempfile
import threading
from contextlib import suppress

import tifffile

from napari_tmidas._env_manager import BaseEnvironmentManager

# Global variable to track running processes for cancellation
_running_processes = []
_process_lock = threading.Lock()


def cancel_all_processes():
    """Cancel all running cellpose processes."""
    with _process_lock:
        for process in _running_processes[
            :
        ]:  # Copy list to avoid modification during iteration
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        process.kill()
                        process.wait()
                _running_processes.remove(process)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"Error terminating process: {e}")


def _add_process(process):
    """Add a process to the tracking list."""
    with _process_lock:
        _running_processes.append(process)


def _remove_process(process):
    """Remove a process from the tracking list."""
    with _process_lock:
        if process in _running_processes:
            _running_processes.remove(process)


class CellposeEnvironmentManager(BaseEnvironmentManager):
    """Environment manager for Cellpose."""

    def __init__(self):
        super().__init__("cellpose")

    def _install_dependencies(self, env_python: str) -> None:
        """Install Cellpose-specific dependencies."""
        # Install cellpose 4 and other dependencies
        print(
            "Installing Cellpose 4 (Cellpose-SAM) in the dedicated environment..."
        )

        # Install packages one by one with error checking
        packages = ["cellpose", "zarr", "tifffile"]
        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call(
                    [env_python, "-m", "pip", "install", package]
                )
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                raise

        # Verify installations
        print("Verifying installations...")

        # Check cellpose
        try:
            result = subprocess.run(
                [
                    env_python,
                    "-c",
                    "from cellpose import core; print(f'GPU available: {core.use_gpu()}')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print("✓ Cellpose installation verified:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Cellpose verification failed: {e}")
            raise

        # Check zarr
        try:
            result = subprocess.run(
                [
                    env_python,
                    "-c",
                    "import zarr; print(f'Zarr version: {zarr.__version__}')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print("✓ Zarr installation verified:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Zarr verification failed: {e}")
            raise

        # Check tifffile
        try:
            result = subprocess.run(
                [
                    env_python,
                    "-c",
                    "import tifffile; print(f'Tifffile version: {tifffile.__version__}')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print("✓ Tifffile installation verified:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Tifffile verification failed: {e}")
            raise

    def is_package_installed(self) -> bool:
        """Check if cellpose is installed in the current environment."""
        try:
            import importlib.util

            return importlib.util.find_spec("cellpose") is not None
        except ImportError:
            return False

    def are_all_packages_installed(self) -> bool:
        """Check if all required packages are installed in the dedicated environment."""
        if not self.is_env_created():
            return False

        env_python = self.get_env_python_path()
        required_packages = ["cellpose", "zarr", "tifffile"]

        for package in required_packages:
            try:
                subprocess.run(
                    [env_python, "-c", f"import {package}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                print(f"Missing package in cellpose environment: {package}")
                return False

        return True

    def reinstall_packages(self) -> None:
        """Force reinstall all packages in the dedicated environment."""
        if not self.is_env_created():
            print("Environment not created. Creating new environment...")
            self.create_env()
            return

        env_python = self.get_env_python_path()
        print("Force reinstalling packages in cellpose environment...")
        self._install_dependencies(env_python)


# Global instance for backward compatibility
manager = CellposeEnvironmentManager()


def is_cellpose_installed():
    """Check if cellpose is installed in the current environment."""
    return manager.is_package_installed()


def is_env_created():
    """Check if the dedicated environment exists."""
    return manager.is_env_created()


def get_env_python_path():
    """Get the path to the Python executable in the environment."""
    return manager.get_env_python_path()


def create_cellpose_env():
    """Create a dedicated virtual environment for Cellpose."""
    return manager.create_env()


def check_cellpose_packages():
    """Check if all required packages are installed in the cellpose environment."""
    return manager.are_all_packages_installed()


def reinstall_cellpose_packages():
    """Force reinstall all packages in the cellpose environment."""
    return manager.reinstall_packages()


def cancel_cellpose_processing():
    """Cancel all running cellpose processes."""
    cancel_all_processes()


def run_cellpose_in_env(func_name, args_dict):
    """
    Run Cellpose in a dedicated environment with optimized zarr support.
    """
    # Ensure the environment exists
    if not is_env_created():
        create_cellpose_env()

    # Check if all required packages are installed
    if not manager.are_all_packages_installed():
        print("Missing packages detected. Reinstalling...")
        manager.reinstall_packages()

    # Check for zarr optimization
    use_zarr_direct = "zarr_path" in args_dict

    if use_zarr_direct:
        zarr_path = args_dict["zarr_path"]
        print(f"Using optimized zarr processing for: {zarr_path}")
        return run_zarr_processing(zarr_path, args_dict)
    else:
        return run_legacy_processing(args_dict)


def run_zarr_processing(zarr_path, args_dict):
    """Process zarr files directly without temporary input files."""

    with (
        tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ) as output_file,
        tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file,
    ):

        # Create zarr processing script (similar to working TIFF script)
        script = f"""
import numpy as np
import sys
from cellpose import models, core
import tifffile
import zarr

# Force output to flush immediately for real-time progress
import sys
sys.stdout.flush()

print("=== Cellpose Environment Info ===")
print(f"GPU available in dedicated environment: {{core.use_gpu()}}")
sys.stdout.flush()

def process_volume(image, name=""):
    print(f"\\nProcessing {{name}}: shape={{image.shape}}, range={{np.min(image):.1f}}-{{np.max(image):.1f}}")
    sys.stdout.flush()

    # GPU detection IN THE DEDICATED ENVIRONMENT (not main environment)
    gpu_available = core.use_gpu()
    use_gpu_requested = {str(args_dict.get('use_gpu', True))}  # Convert to string for f-string
    actual_use_gpu = use_gpu_requested and gpu_available

    print(f"GPU: requested={{use_gpu_requested}}, available={{gpu_available}}, using={{actual_use_gpu}}")
    sys.stdout.flush()

    # Create model with explicit GPU setting
    print("Creating Cellpose model...")
    sys.stdout.flush()
    model = models.CellposeModel(gpu=actual_use_gpu)
    print(f"Model created (GPU={{model.gpu}})")
    sys.stdout.flush()

    print("Running segmentation...")
    sys.stdout.flush()

    # Remove deprecated channels parameter for Cellpose v4.0.1+
    masks, flows, styles = model.eval(
        image,
        # channels parameter removed - deprecated in v4.0.1+
        flow_threshold={args_dict.get('flow_threshold', 0.4)},
        cellprob_threshold={args_dict.get('cellprob_threshold', 0.0)},
        batch_size={args_dict.get('batch_size', 32)},
        normalize={args_dict.get('normalize', {'tile_norm_blocksize': 128})},
        do_3D={args_dict.get('do_3D', False)},
        flow3D_smooth={args_dict.get('flow3D_smooth', 0)},
        anisotropy={args_dict.get('anisotropy', None)},
        z_axis={args_dict.get('z_axis', 0)} if {args_dict.get('do_3D', False)} else None
    )

    object_count = np.max(masks)
    print(f"Found {{object_count}} objects in {{name}}")
    sys.stdout.flush()
    return masks

print("Opening zarr: {zarr_path}")
sys.stdout.flush()
zarr_root = zarr.open('{zarr_path}', mode='r')

if hasattr(zarr_root, 'shape'):
    image = np.array(zarr_root)
    result = process_volume(image, "zarr")
else:
    arrays = list(zarr_root.array_keys())
    print(f"Arrays: {{arrays}}")
    sys.stdout.flush()
    zarr_array = zarr_root[arrays[0]]
    print(f"Selected: {{arrays[0]}}, shape={{zarr_array.shape}}")
    sys.stdout.flush()

    if len(zarr_array.shape) == 5:  # TCZYX
        T, C, Z, Y, X = zarr_array.shape
        print(f"5D TCZYX: T={{T}}, C={{C}}, Z={{Z}}, Y={{Y}}, X={{X}}")
        print(f"Will process {{T*C}} T,C combinations")
        sys.stdout.flush()
        result = np.zeros((T, C, Z, Y, X), dtype=np.uint32)

        for t in range(T):
            for c in range(C):
                print(f"\\n=== T={{t+1}}/{{T}}, C={{c+1}}/{{C}} ===")
                sys.stdout.flush()
                image = np.array(zarr_array[t, c, :, :, :])
                masks = process_volume(image, f"T{{t+1}}C{{c+1}}")
                result[t, c] = masks

    elif len(zarr_array.shape) == 4:  # 4D
        dim1, Z, Y, X = zarr_array.shape
        print(f"4D array: dim1={{dim1}}, Z={{Z}}, Y={{Y}}, X={{X}}")
        sys.stdout.flush()
        result = np.zeros((dim1, Z, Y, X), dtype=np.uint32)

        for i in range(dim1):
            print(f"\\n=== Volume {{i+1}}/{{dim1}} ===")
            sys.stdout.flush()
            image = np.array(zarr_array[i, :, :, :])
            masks = process_volume(image, f"Vol{{i+1}}")
            result[i] = masks
    else:
        image = np.array(zarr_array)
        result = process_volume(image, "3D")

print(f"Saving results: shape={{result.shape}}, total objects={{np.max(result)}}")
sys.stdout.flush()
tifffile.imwrite('{output_file.name}', result.astype(np.uint32))
print("Complete!")
sys.stdout.flush()
"""

        script_file.write(script)
        script_file.flush()

        try:
            # Run with REAL-TIME output (don't capture, let it stream)
            env_python = get_env_python_path()
            print("=== Starting Cellpose Processing ===")

            # Use Popen for real-time progress and cancellation support
            process = subprocess.Popen(
                [env_python, script_file.name],
                cwd=os.getcwd(),
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )

            # Track the process for cancellation
            _add_process(process)

            try:
                # Wait for completion
                returncode = process.wait()

                if returncode != 0:
                    raise RuntimeError(
                        f"Cellpose failed with return code {returncode}"
                    )

            finally:
                # Remove from tracking regardless of outcome
                _remove_process(process)

            # Check if output file was created
            if not os.path.exists(output_file.name):
                raise RuntimeError("Output file was not created")

            return tifffile.imread(output_file.name)

        finally:
            # Cleanup
            for fname in [output_file.name, script_file.name]:
                with suppress(OSError, FileNotFoundError):
                    os.unlink(fname)


def run_legacy_processing(args_dict):
    """Legacy processing for numpy arrays (original working TIFF approach)."""

    with (
        tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as input_file,
        tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ) as output_file,
        tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file,
    ):

        # Save input image (exactly like original working code)
        tifffile.imwrite(input_file.name, args_dict["image"])

        # Create script (exactly like original working code)
        script = f"""
import numpy as np
from cellpose import models, core
import tifffile

# Load image
image = tifffile.imread('{input_file.name}')

# Create and run model (exactly like original working code)
model = models.CellposeModel(
    gpu={args_dict.get('use_gpu', True)})

# Prepare normalization parameters (Cellpose 4)
normalize = {args_dict.get('normalize', {'tile_norm_blocksize': 128})}

# Perform segmentation with Cellpose 4 parameters
masks, flows, styles = model.eval(
    image,
    channels={args_dict.get('channels', [0, 0])},
    flow_threshold={args_dict.get('flow_threshold', 0.4)},
    cellprob_threshold={args_dict.get('cellprob_threshold', 0.0)},
    batch_size={args_dict.get('batch_size', 32)},
    normalize=normalize,
    do_3D={args_dict.get('do_3D', False)},
    flow3D_smooth={args_dict.get('flow3D_smooth', 0)},
    anisotropy={args_dict.get('anisotropy', None)},
    z_axis={args_dict.get('z_axis', 0)} if {args_dict.get('do_3D', False)} else None
)

# Save results
tifffile.imwrite('{output_file.name}', masks)
"""

        # Write script
        script_file.write(script)
        script_file.flush()

    try:
        # Run the script with cancellation support
        env_python = get_env_python_path()

        # Use Popen for cancellation support
        process = subprocess.Popen(
            [env_python, script_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Track the process for cancellation
        _add_process(process)

        try:
            # Wait for completion and get output
            stdout, stderr = process.communicate()
            print("Stdout:", stdout)

            # Check for errors
            if process.returncode != 0:
                print("Stderr:", stderr)
                raise RuntimeError(f"Cellpose segmentation failed: {stderr}")

        finally:
            # Remove from tracking regardless of outcome
            _remove_process(process)

        # Read and return the results
        return tifffile.imread(output_file.name)

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error in Cellpose segmentation: {e}")
        raise

    finally:
        # Clean up temporary files
        for fname in [input_file.name, output_file.name, script_file.name]:
            with suppress(OSError, FileNotFoundError):
                os.unlink(fname)

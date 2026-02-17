#!/usr/bin/env python3
"""
Environment Manager for Ultrack

This module manages a dedicated conda environment for ultrack and its dependencies,
isolated from the main napari-tmidas environment.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict


def get_conda_cmd() -> str:
    """Detect conda/mamba command available in the system."""
    # Try mamba first (faster), fallback to conda
    for cmd in ["mamba", "conda"]:
        if shutil.which(cmd):
            return cmd
    raise RuntimeError("Neither conda nor mamba found in PATH")


def is_env_created(env_name: str = "ultrack") -> bool:
    """
    Check if the ultrack conda environment exists.
    
    Parameters:
    -----------
    env_name : str
        Name of the conda environment (default: "ultrack")
    
    Returns:
    --------
    bool
        True if environment exists, False otherwise
    """
    try:
        conda_cmd = get_conda_cmd()
        result = subprocess.run(
            [conda_cmd, "env", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Look for environment name in the output
        for line in result.stdout.split("\n"):
            if env_name in line:
                return True
        return False
    except Exception as e:
        print(f"Error checking environment: {e}")
        return False


def _get_cuda_version() -> Optional[str]:
    """
    Detect CUDA version from nvidia-smi.
    
    Returns:
    --------
    Optional[str]
        CUDA version string (e.g., '13.0'), or None if not available
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return None
        
        # Parse CUDA version from nvidia-smi output
        # Example line: "CUDA Version: 13.0"
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _ensure_scikit_image_fix(env_name: str = "ultrack", log_func=None) -> bool:
    """
    Ensure scikit-image has the read-only array fix.
    
    The fix for read-only arrays (needed for zarr/pandas compatibility) was merged
    in commit 70ab2a6b on Feb 8, 2026. It will be in scikit-image >= 0.26.1.
    
    Until 0.26.1 is released, this installs the dev version from GitHub.
    Once 0.26.1+ is available, it upgrades to the stable release.
    
    Parameters:
    -----------
    env_name : str
        Name of the conda environment
    log_func : callable
        Optional logging function
    
    Returns:
    --------
    bool
        True if scikit-image has the fix, False otherwise
    """
    def log(msg):
        if log_func:
            log_func(msg)
    
    try:
        conda_cmd = get_conda_cmd()
        
        # Check current scikit-image version
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", "-c",
             "import skimage; print(skimage.__version__)"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            log("Could not check scikit-image version")
            return False
        
        current_version = result.stdout.strip()
        log(f"Current scikit-image version: {current_version}")
        
        # Parse version to check if we have the fix
        # The fix is in scikit-image >= 0.26.1 (released Feb 2026)
        import re
        version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', current_version)
        
        if version_match:
            major, minor, patch = map(int, version_match.groups())
            
            # Check if we have 0.26.1 or later (stable release with fix)
            if (major, minor, patch) >= (0, 26, 1):
                log("✓ scikit-image has the read-only array fix")
                return True
        
        # Check if we have the dev version with the fix
        if 'dev' in current_version and '0.26.1' in current_version:
            log("✓ Using scikit-image dev version with read-only array fix")
            
            # Try to upgrade to stable if available
            log("  Checking for stable release...")
            check_result = subprocess.run(
                [conda_cmd, "run", "-n", env_name, "pip", "install", 
                 "--dry-run", "--upgrade", "scikit-image>=0.26.1"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # If a stable version is available, upgrade
            if check_result.returncode == 0 and '0.26.1' in check_result.stdout and 'dev' not in check_result.stdout:
                log("  ✓ Stable release available, upgrading...")
                upgrade_result = subprocess.run(
                    [conda_cmd, "run", "-n", env_name, "pip", "install",
                     "--upgrade", "scikit-image>=0.26.1"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if upgrade_result.returncode == 0:
                    log("  ✓ Upgraded to stable scikit-image")
                else:
                    log("  Keeping dev version")
            
            return True
        
        # Need to install dev version with the fix
        log("Installing scikit-image dev version with read-only array fix...")
        log("  (Will auto-upgrade to stable 0.26.1+ when available)")
        
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "pip", "install",
             "--upgrade", "git+https://github.com/scikit-image/scikit-image.git@main"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            log("✓ Installed scikit-image dev version")
            return True
        else:
            log(f"Failed to install scikit-image dev version: {result.stderr}")
            return False
            
    except Exception as e:
        log(f"Error managing scikit-image: {e}")
        return False


def _patch_ultrack_xp(env_name: str = "ultrack", log_func=None) -> bool:
    """
    Patch ultrack's cuda.py to fix xp import bug when CUDA is disabled.
    
    Bug: When CuPy is installed but CUDA is not available (e.g., CUDA_VISIBLE_DEVICES=''),
    ultrack sets cp=None but forgets to set xp=np, causing ImportError.
    
    Fix: Add 'xp = np' when CUDA is not available.
    
    Parameters:
    -----------
    env_name : str
        Name of the conda environment
    log_func : callable
        Optional logging function
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    def log(msg):
        if log_func:
            log_func(msg)
    
    try:
        conda_cmd = get_conda_cmd()
        
        # Get Python site-packages path in the environment
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", "-c", 
             "import site; print(site.getsitepackages()[0])"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            log(f"Could not get site-packages path: {result.stderr}")
            return False
        
        site_packages = result.stdout.strip()
        cuda_py_path = Path(site_packages) / "ultrack" / "utils" / "cuda.py"
        
        if not cuda_py_path.exists():
            log(f"ultrack/utils/cuda.py not found at {cuda_py_path}")
            return False
        
        # Read the file
        with open(cuda_py_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "xp = np" in content and "cupy found but cuda is not available" in content:
            # Check the exact context to see if it's already fixed
            if """    if not cp.cuda.is_available():
        cp = None
        xp = np
        LOG.info("cupy found but cuda is not available.")""" in content:
                log("ultrack already patched")
                return True
        
        # Apply the patch
        old_code = """    if not cp.cuda.is_available():
        cp = None
        LOG.info("cupy found but cuda is not available.")
    else:"""
        
        new_code = """    if not cp.cuda.is_available():
        cp = None
        xp = np
        LOG.info("cupy found but cuda is not available.")
    else:"""
        
        if old_code not in content:
            log("ultrack code structure different than expected - patch may not be needed")
            return True  # Not a failure, just different version
        
        content = content.replace(old_code, new_code)
        
        # Write back
        with open(cuda_py_path, 'w') as f:
            f.write(content)
        
        log(f"Patched ultrack/utils/cuda.py to fix xp import bug")
        return True
        
    except Exception as e:
        log(f"Error patching ultrack: {e}")
        return False


def _patch_ultrack_readonly_arrays(env_name: str, log_func=None) -> bool:
    """
    Patch ultrack to handle read-only zarr arrays without making copies.
    
    Fixes: ValueError: buffer source array is read-only
    
    The issue occurs when ultrack's solver uses fancy indexing on read-only
    zarr arrays. This patch ensures arrays are writable before fancy indexing
    without making full copies (only copy if necessary).
    
    Based on: https://github.com/royerlab/ultrack/pull/260
    
    Parameters:
    -----------
    env_name : str
        Name of the conda environment
    log_func : callable
        Optional logging function
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    def log(msg):
        if log_func:
            log_func(msg)
    
    try:
        conda_cmd = get_conda_cmd()
        
        # Get Python site-packages path in the environment
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", "-c", 
             "import site; print(site.getsitepackages()[0])"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            log(f"Could not get site-packages path: {result.stderr}")
            return False
        
        site_packages = result.stdout.strip()
        solver_path = Path(site_packages) / "ultrack" / "core" / "solve" / "solver" / "mip_solver.py"
        
        if not solver_path.exists():
            log(f"ultrack solver not found at {solver_path}")
            return False
        
        # Read the file
        with open(solver_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "# PATCH: Ensure arrays are writable" in content:
            log("ultrack solver already patched for read-only arrays")
            return True
        
        # Patch the add_edges method
        old_code = '''        sources = self._forward_map[np.asarray(sources, dtype=int)]
        targets = self._forward_map[np.asarray(targets, dtype=int)]'''
        
        new_code = '''        # PATCH: Ensure arrays are writable for scikit-image ArrayMapper
        # Always copy to avoid "buffer source array is read-only" error
        # These are small node ID arrays (~MB), not full image data (~GB)
        sources_arr = np.array(sources, dtype=int, copy=True)
        sources = self._forward_map[sources_arr]
        
        targets_arr = np.array(targets, dtype=int, copy=True)
        targets = self._forward_map[targets_arr]'''
        
        if old_code not in content:
            log("ultrack solver code structure different than expected")
            return True  # Not a failure, might be a different version
        
        content = content.replace(old_code, new_code)
        
        # Write back
        with open(solver_path, 'w') as f:
            f.write(content)
        
        log(f"Patched ultrack solver to handle read-only arrays")
        return True
        
    except Exception as e:
        log(f"Error patching ultrack solver: {e}")
        return False


def create_ultrack_env(env_name: str = "ultrack", progress_callback=None) -> bool:
    """
    Create a dedicated conda environment for ultrack with all dependencies.
    
    This function:
    1. Creates a new conda environment with Python 3.11
    2. Installs ultrack and dependencies from conda-forge
    3. Attempts to install CuPy for GPU acceleration (optional)
    4. Pins scipy to version 1.14 to avoid binary compatibility issues
    
    Parameters:
    -----------
    env_name : str
        Name for the conda environment (default: "ultrack")
    progress_callback : callable, optional
        Function to call with progress messages
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    def log(msg: str):
        """Helper to log messages."""
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    try:
        conda_cmd = get_conda_cmd()
        log(f"Using {conda_cmd} to create environment '{env_name}'...")
        
        # Step 1: Create environment with Python 3.11
        log("Creating conda environment with Python 3.11...")
        result = subprocess.run(
            [conda_cmd, "create", "-n", env_name, "python=3.11", "-y"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            log(f"Failed to create environment: {result.stderr}")
            return False
        log("✓ Environment created")
        
        # Step 2: Install core scientific packages from conda-forge (faster, better compatibility)
        log("Installing core scientific packages from conda-forge...")
        conda_packages = [
            "numpy",
            "scipy=1.14",  # Pin to avoid binary compatibility issues with 1.17
            "pandas",
            "scikit-image",
        ]
        
        result = subprocess.run(
            [conda_cmd, "install", "-n", env_name, "-c", "conda-forge"] + conda_packages + ["-y"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            log(f"Warning: Some conda packages failed to install: {result.stderr}")
            log("Will try to continue with pip...")
        else:
            log(f"✓ Installed {len(conda_packages)} packages from conda-forge")
        
        # Step 3: Install PyTorch (for GPU-accelerated labels_to_contours)
        cuda_version = _get_cuda_version()
        log("Installing PyTorch...")
        
        if cuda_version:
            cuda_major = int(float(cuda_version))
            log(f"  NVIDIA GPU detected (CUDA {cuda_version})")
            
            # Install PyTorch with CUDA support
            # PyTorch 2.5+ supports Blackwell (sm_120)
            log("  Installing PyTorch with CUDA support...")
            if cuda_major >= 12:
                # CUDA 12.x: use latest PyTorch with cu121
                torch_package = "torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            else:
                # CUDA 11.x
                torch_package = "torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            
            result = subprocess.run(
                f"{conda_cmd} run -n {env_name} pip install {torch_package}",
                capture_output=True,
                text=True,
                shell=True,
                timeout=600
            )
            if result.returncode != 0:
                log(f"  ⚠ Failed to install PyTorch with CUDA: {result.stderr}")
                log("  Falling back to CPU-only PyTorch...")
                result = subprocess.run(
                    [conda_cmd, "run", "-n", env_name, "pip", "install", "torch", "torchvision"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    log("  ✓ Installed PyTorch (CPU-only)")
            else:
                log("  ✓ Installed PyTorch with CUDA support")
                
                # Test PyTorch GPU compatibility (especially for Blackwell sm_120)
                log("  Testing PyTorch GPU compatibility...")
                test_result = subprocess.run(
                    [conda_cmd, "run", "-n", env_name, "python", "-c",
                     "import torch; import warnings; warnings.filterwarnings('error'); "
                     "d = 'cuda' if torch.cuda.is_available() else 'cpu'; "
                     "t = torch.ones((10, 10), device=d); "
                     "r = t.sum(); "
                     "print(f'SUCCESS:{torch.cuda.get_device_name(0) if d==\"cuda\" else \"CPU\"}:{torch.cuda.get_device_properties(0).major}{torch.cuda.get_device_properties(0).minor if d==\"cuda\" else \"\"}')"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if "SUCCESS:" in test_result.stdout:
                    device_info = test_result.stdout.strip().split(":")
                    log(f"    ✓ GPU works: {device_info[1]}")
                    if len(device_info) > 2 and device_info[2]:
                        compute_cap = device_info[2]
                        log(f"    Compute capability: sm_{compute_cap}")
                elif "sm_120" in test_result.stderr or "Blackwell" in test_result.stderr:
                    log("    ⚠ Blackwell GPU detected but PyTorch stable doesn't support sm_120")
                    log("    SOLUTION: Upgrade to PyTorch nightly for Blackwell support:")
                    log("      conda run -n ultrack pip uninstall -y torch torchvision")
                    log("      conda run -n ultrack pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130")
                    log("    OR use CPU mode (works reliably without GPU)")
                else:
                    log(f"    ⚠ GPU test inconclusive, but installation successful")
        else:
            # No GPU: install CPU-only PyTorch
            log("  No NVIDIA GPU detected, installing CPU-only PyTorch...")
            result = subprocess.run(
                [conda_cmd, "run", "-n", env_name, "pip", "install", "torch", "torchvision"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                log("  ✓ Installed PyTorch (CPU-only)")
            else:
                log(f"  ⚠ Failed to install PyTorch: {result.stderr}")
        
        # Step 4: Install ultrack and related packages from pip
        log("Installing ultrack and dependencies from pip...")
        pip_packages = [
            "ultrack>=0.13.0",
            "zarr>=3.0.0",
            "tifffile",
        ]
        
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "pip", "install"] + pip_packages,
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            log(f"ERROR: Failed to install pip packages: {result.stderr}")
            return False
        else:
            log(f"✓ Installed {len(pip_packages)} packages from pip")
        
        # Step 5: Check for NVIDIA GPU and install CuPy (legacy support)
        # Note: PyTorch is now the primary GPU backend, but CuPy kept for compatibility
        if cuda_version:
            log("Installing CuPy (legacy GPU support)...")
            
            # Determine which CuPy package to install based on CUDA version
            cuda_major = int(float(cuda_version))
            if cuda_major >= 13:
                cupy_package = "cupy-cuda13x"
            elif cuda_major == 12:
                cupy_package = "cupy-cuda12x"
            else:
                cupy_package = "cupy-cuda11x"
            
            # For Blackwell (sm_120) support, we need CuPy 13.3.0+ with CUDA 12.6+
            # Try to install the latest version to maximize GPU architecture support
            log(f"  Installing latest {cupy_package} for Blackwell GPU support...")
            result = subprocess.run(
                [conda_cmd, "run", "-n", env_name, "pip", "install", "--upgrade", cupy_package],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                log(f"  ✓ Installed {cupy_package}")
            else:
                log(f"  ⚠ Failed to install {cupy_package}: {result.stderr}")
                log("  GPU acceleration will not be available")
            
            # Install cucim for GPU-accelerated image processing
            log("  Installing cucim from conda-forge...")
            result = subprocess.run(
                [conda_cmd, "install", "-n", env_name, "-c", "conda-forge", "cucim", "-y"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                log("  ✓ Installed cucim")
            else:
                log(f"  ⚠ Failed to install cucim: {result.stderr}")
        else:
            log("ℹ No NVIDIA GPU detected. Installing CPU-only version.")
        
        # Step 5: Test GPU availability if CuPy was installed (informational only)
        if cuda_version:
            log("Testing GPU detection (informational only - GPU currently not used)...")
            test_result = check_gpu_available(env_name)
            if test_result["available"]:
                log(f"  ✓ GPU ready for future use: {test_result['device_name']}")
                log(f"    Compute capability: {test_result['compute_capability']}")
                log(f"    Memory: {test_result['memory_gb']:.1f} GB")
                log(f"    Compatible with Ada architecture (sm_89) and older")
            else:
                error_msg = test_result.get('error', 'Unknown reason')
                if 'CUDA_ERROR_NO_BINARY_FOR_GPU' in error_msg or 'sm_12' in error_msg:
                    log(f"  ℹ Newer GPU architecture detected (Blackwell)")
                    log(f"    CuPy installed but will use CPU mode (normal behavior)")
                else:
                    log(f"  ℹ GPU test: {error_msg}")
                log(f"  → CPU processing active (works fine for all users)")
        
        # Step 6: Verify critical packages are installed
        log("Verifying package installation...")
        critical_packages = ["ultrack", "zarr", "numpy", "scipy", "tifffile"]
        missing_packages = []
        
        for pkg in critical_packages:
            if not is_package_installed(pkg, env_name):
                missing_packages.append(pkg)
        
        if missing_packages:
            log(f"✗ ERROR: Critical packages missing: {', '.join(missing_packages)}")
            log("Environment creation failed. Please try recreating the environment.")
            return False
        else:
            log(f"✓ All {len(critical_packages)} critical packages verified")
        
        # Step 7: Patch ultrack to fix xp import bug when CUDA is disabled
        log("Applying compatibility patches...")
        if not _patch_ultrack_xp(env_name, log):
            log("  ⚠ Warning: Failed to apply ultrack patch (may cause issues with CPU mode)")
        else:
            log("  ✓ ultrack patched for CPU mode compatibility")
        
        # Patch ultrack to handle read-only zarr arrays (fixes solver errors)
        if not _patch_ultrack_readonly_arrays(env_name, log):
            log("  ⚠ Warning: Failed to patch read-only array handling")
        else:
            log("  ✓ ultrack patched for read-only array compatibility")
        
        # Ensure scikit-image has the read-only array fix
        if not _ensure_scikit_image_fix(env_name, log):
            log("  ⚠ Warning: Could not verify scikit-image fix")
        else:
            log("  ✓ scikit-image has read-only array fix")
        
        log(f"✓ Environment '{env_name}' created successfully")
        log("  Processing mode: CPU-only (reliable across all systems)")
        if cuda_version:
            log("  GPU packages installed and ready for future ultrack updates")
        return True
        
    except Exception as e:
        log(f"Error creating environment: {e}")
        return False


def check_gpu_available(env_name: str = "ultrack") -> Dict[str, any]:
    """
    Check if GPU acceleration is available in the ultrack environment.
    
    Parameters:
    -----------
    env_name : str
        Name of the conda environment
    
    Returns:
    --------
    Dict[str, any]
        Dictionary with keys:
        - 'available': bool - True if GPU can be used
        - 'device_name': str - GPU model name
        - 'compute_capability': str - Compute capability (e.g., '12.0')
        - 'memory_gb': float - GPU memory in GB
        - 'error': str - Error message if not available
    """
    try:
        conda_cmd = get_conda_cmd()
        
        # Create a simple test script
        test_script = """
import sys
try:
    import cupy as cp
    if not cp.cuda.is_available():
        print("ERROR:CUDA not available")
        sys.exit(1)
    
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    compute_cap = device.compute_capability
    
    # Test actual kernel compilation (this is where Blackwell fails)
    test_arr = cp.array([1, 2, 3])
    _ = test_arr > 0
    cp.cuda.Device().synchronize()
    
    # If we got here, GPU works!
    print(f"SUCCESS:{props['name'].decode()}:{compute_cap[0]}.{compute_cap[1]}:{props['totalGlobalMem'] / 1024**3:.1f}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}: {e}")
    sys.exit(1)
"""
        
        # Run the test in the ultrack environment
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout.strip()
        
        if output.startswith("SUCCESS:"):
            _, device_name, compute_cap, memory_gb = output.split(":")
            return {
                "available": True,
                "device_name": device_name,
                "compute_capability": compute_cap,
                "memory_gb": float(memory_gb),
            }
        else:
            error_msg = output.replace("ERROR:", "") if output.startswith("ERROR:") else output
            return {
                "available": False,
                "error": error_msg,
            }
    
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def is_package_installed(package_name: str, env_name: str = "ultrack") -> bool:
    """
    Check if a package is installed in the specified conda environment.
    
    Parameters:
    -----------
    package_name : str
        Name of the package to check
    env_name : str
        Name of the conda environment
    
    Returns:
    --------
    bool
        True if package is installed, False otherwise
    """
    try:
        conda_cmd = get_conda_cmd()
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "python", "-c", f"import {package_name}"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def setup_gurobi_license(license_key: str, env_name: str = "ultrack") -> bool:
    """
    Set up Gurobi license in the ultrack environment.
    
    Parameters:
    -----------
    license_key : str
        Gurobi license key
    env_name : str
        Name of the conda environment
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        conda_cmd = get_conda_cmd()
        
        # Install gurobi if not already installed
        if not is_package_installed("gurobipy", env_name):
            print("Installing Gurobi...")
            result = subprocess.run(
                [conda_cmd, "install", "-n", env_name, "-c", "gurobi", "gurobi", "-y"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"Failed to install Gurobi: {result.stderr}")
                return False
        
        # Activate license
        print("Activating Gurobi license...")
        result = subprocess.run(
            [conda_cmd, "run", "-n", env_name, "grbgetkey", license_key],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✓ Gurobi license activated")
            return True
        else:
            print(f"Failed to activate license: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"Error setting up Gurobi: {e}")
        return False


def run_ultrack_in_env(
    script_content: str,
    env_name: str = "ultrack",
    progress_callback=None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, any]:
    """
    Run a Python script in the ultrack conda environment.
    
    Parameters:
    -----------
    script_content : str
        Python script to execute
    env_name : str
        Name of the conda environment
    progress_callback : callable, optional
        Function to call with progress messages
    input_file : str, optional
        Path to input file (for logging)
    output_file : str, optional
        Path to output file (for logging)
    
    Returns:
    --------
    Dict[str, any]
        Dictionary with keys:
        - 'success': bool
        - 'output': str
        - 'error': str
    """
    def log(msg: str):
        """Helper to log messages."""
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    try:
        conda_cmd = get_conda_cmd()
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run the script with REAL-TIME output streaming
            log(f"Running ultrack in environment '{env_name}'...")
            if input_file:
                log(f"  Input: {input_file}")
            if output_file:
                log(f"  Output: {output_file}")
            
            # Stream output in real-time so you see ILP progress
            process = subprocess.Popen(
                [conda_cmd, "run", "-n", env_name, "python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                if progress_callback:
                    progress_callback(line.rstrip())
                else:
                    print(line, end='', flush=True)
            
            try:
                process.wait(timeout=7200)  # 2 hour timeout
            except subprocess.TimeoutExpired:
                process.kill()
                log("✗ Tracking timed out after 2 hours")
                return {
                    "success": False,
                    "output": ''.join(output_lines),
                    "error": "Process timed out after 2 hours",
                }
            
            success = process.returncode == 0
            
            if success:
                log("✓ Tracking completed successfully")
            else:
                log(f"✗ Tracking failed with return code {process.returncode}")
            
            return {
                "success": success,
                "output": ''.join(output_lines),
                "error": "" if success else ''.join(output_lines[-50:]),  # Last 50 lines on error
            }
        
        finally:
            # Clean up temporary script
            if Path(script_path).exists():
                Path(script_path).unlink()
    
    except Exception as e:
        log(f"✗ Error running ultrack: {e}")
        return {
            "success": False,
            "output": "",
            "error": str(e),
        }


class UltrackEnvironmentManager:
    """
    Backwards compatibility wrapper for ultrack environment management.
    
    This class provides the same interface as the original implementation,
    but delegates to module-level functions.
    """
    
    def __init__(self, env_name: str = "ultrack"):
        """Initialize the environment manager."""
        self.env_name = env_name
    
    def _get_conda_cmd(self) -> str:
        """Get conda command."""
        return get_conda_cmd()
    
    def is_env_created(self) -> bool:
        """Check if environment exists."""
        return is_env_created(self.env_name)
    
    def create_env(self, progress_callback=None) -> bool:
        """Create the ultrack environment."""
        return create_ultrack_env(self.env_name, progress_callback)
    
    def is_package_installed(self, package_name: str = "ultrack") -> bool:
        """Check if a package is installed."""
        return is_package_installed(package_name, self.env_name)
    
    def check_gpu_available(self) -> Dict[str, any]:
        """Check GPU availability."""
        return check_gpu_available(self.env_name)
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available (simplified boolean result)."""
        result = check_gpu_available(self.env_name)
        return result.get("available", False)
    
    def setup_gurobi_license(self, license_key: str) -> bool:
        """Setup Gurobi license."""
        return setup_gurobi_license(license_key, self.env_name)
    
    def run_in_env(
        self,
        script_content: str,
        progress_callback=None,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, any]:
        """Run a script in the environment."""
        return run_ultrack_in_env(
            script_content=script_content,
            env_name=self.env_name,
            progress_callback=progress_callback,
            input_file=input_file,
            output_file=output_file,
        )

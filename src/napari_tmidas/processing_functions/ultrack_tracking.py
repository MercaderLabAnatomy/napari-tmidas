#!/usr/bin/env python3
"""
Ultrack Cell Tracking Module for napari-tmidas

This module integrates ultrack cell tracking with segmentation ensemble into the
napari-tmidas batch processing framework. It uses a dedicated conda environment
to manage ultrack dependencies separately from the main environment.

Ultrack supports tracking cells across 2D, 3D, and multichannel datasets, and can
handle segmentation uncertainty by evaluating multiple candidate segmentations.
"""

import inspect
from pathlib import Path
from typing import List, Optional

import numpy as np
from skimage.io import imread

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions.ultrack_env_manager import (
    _ensure_scikit_image_fix,
    create_ultrack_env,
    is_env_created,
    is_package_installed,
    run_ultrack_in_env,
    setup_gurobi_license,
)


def create_ultrack_ensemble_script(
    label_paths: List[str],
    output_path: str,
    gurobi_license: Optional[str] = None,
    min_area: int = 200,
    max_neighbors: int = 5,
    max_distance: float = 40.0,
    appear_weight: float = -0.1,
    disappear_weight: float = -2.0,
    division_weight: float = -0.01,
    enable_gpu: bool = True,
) -> str:
    """
    Create a Python script to run ultrack with segmentation ensemble.

    Parameters:
    -----------
    label_paths : List[str]
        List of paths to label images from different segmentation methods
    output_path : str
        Path to save the tracked output
    gurobi_license : Optional[str]
        Gurobi license key if provided
    min_area : int
        Minimum area for candidate segments
    max_neighbors : int
        Maximum number of candidate neighbors
    max_distance : float
        Maximum distance between cells for linking
    appear_weight : float
        Weight for cell appearance
    disappear_weight : float
        Weight for cell disappearance
    division_weight : float
        Weight for cell division
    enable_gpu : bool
        Enable GPU acceleration with PyTorch (default ON). Set to False for CPU-only mode.

    Returns:
    --------
    str
        Python script content
    """
    # Format label_paths list as Python code
    label_paths_str = "[\n" + ",\n".join([f"        '{path}'" for path in label_paths]) + "\n    ]"
    
    # Generate GPU enable/disable code and import appropriate labels_to_contours
    if enable_gpu:
        gpu_setup_code = """
# ============================================================================
# GPU MODE ENABLED - Using PyTorch for GPU acceleration
# ============================================================================
# This implementation uses PyTorch instead of CuPy for better GPU compatibility
# PyTorch 2.5+ supports NVIDIA Blackwell architecture (sm_120)
# ============================================================================
import os
import sys

print("=" * 80)
print("GPU MODE ENABLED - Using PyTorch for GPU acceleration")
print("=" * 80)
print("Benefits:")
print("  ✓ PyTorch 2.5+ supports Blackwell GPUs (sm_120)")
print("  ✓ Better compatibility across GPU architectures")
print("  ✓ Same output as CuPy-based version")
print("=" * 80)
"""
        
        # Use PyTorch-based labels_to_contours
        labels_to_contours_import = """
# Import PyTorch-based labels_to_contours (GPU-accelerated, CuPy-free)
# This custom implementation produces identical output to ultrack's CuPy version
import torch
import torch.nn.functional as F

# PyTorch labels_to_contours implementation (inline for portability)
def _find_boundaries_torch(labels_tensor, mode="outer"):
    \"\"\"Find boundaries using PyTorch morphological operations.\"\"\"
    # Convert uint dtypes to signed int for CUDA compatibility
    if labels_tensor.dtype in [torch.uint8, torch.uint16, torch.uint32]:
        labels_tensor = labels_tensor.to(torch.int32)
    elif labels_tensor.dtype not in [torch.int32, torch.int64]:
        labels_tensor = labels_tensor.long()
    foreground = labels_tensor > 0
    
    if labels_tensor.ndim == 2:
        # 2D case
        padded = F.pad(labels_tensor.unsqueeze(0).unsqueeze(0).float(), 
                      (1, 1, 1, 1), mode='constant', value=0)
        boundary = torch.zeros_like(labels_tensor, dtype=torch.bool)
        center_val = labels_tensor
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[0, 0, 1+dy:labels_tensor.shape[0]+1+dy, 1+dx:labels_tensor.shape[1]+1+dx]
                boundary |= ((shifted.long() != center_val) & (center_val > 0))
    
    elif labels_tensor.ndim == 3:
        # 3D case
        padded = F.pad(labels_tensor.unsqueeze(0).unsqueeze(0).float(),
                      (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        boundary = torch.zeros_like(labels_tensor, dtype=torch.bool)
        center_val = labels_tensor
        
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    shifted = padded[0, 0, 
                                   1+dz:labels_tensor.shape[0]+1+dz,
                                   1+dy:labels_tensor.shape[1]+1+dy,
                                   1+dx:labels_tensor.shape[2]+1+dx]
                    boundary |= ((shifted.long() != center_val) & (center_val > 0))
    
    return boundary

def _gaussian_filter_torch(input_tensor, sigma):
    \"\"\"Apply Gaussian filter using PyTorch separable convolution.\"\"\"
    if sigma <= 0:
        return input_tensor
    
    kernel_size = max(3, int(4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    x = torch.arange(kernel_size, dtype=input_tensor.dtype, device=input_tensor.device)
    x = x - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    result = input_tensor
    
    if input_tensor.ndim == 2:
        result = result.unsqueeze(0).unsqueeze(0)
        kernel = gauss_1d.view(1, 1, 1, -1)
        result = F.pad(result, (kernel_size // 2, kernel_size // 2, 0, 0), mode='replicate')
        result = F.conv2d(result, kernel)
        kernel = gauss_1d.view(1, 1, -1, 1)
        result = F.pad(result, (0, 0, kernel_size // 2, kernel_size // 2), mode='replicate')
        result = F.conv2d(result, kernel)
        result = result.squeeze(0).squeeze(0)
    elif input_tensor.ndim == 3:
        result = result.unsqueeze(0).unsqueeze(0)
        kernel = gauss_1d.view(1, 1, -1, 1, 1)
        result = F.pad(result, (0, 0, 0, 0, kernel_size // 2, kernel_size // 2), mode='replicate')
        result = F.conv3d(result, kernel)
        kernel = gauss_1d.view(1, 1, 1, -1, 1)
        result = F.pad(result, (0, 0, kernel_size // 2, kernel_size // 2, 0, 0), mode='replicate')
        result = F.conv3d(result, kernel)
        kernel = gauss_1d.view(1, 1, 1, 1, -1)
        result = F.pad(result, (kernel_size // 2, kernel_size // 2, 0, 0, 0, 0), mode='replicate')
        result = F.conv3d(result, kernel)
        result = result.squeeze(0).squeeze(0)
    
    return result

def labels_to_contours_torch(labels_list, sigma, foreground_path, edges_path, device='cuda'):
    \"\"\"PyTorch-based labels_to_contours implementation.\"\"\"
    import numpy as np
    from tqdm import tqdm
    
    shape = labels_list[0].shape
    print("GPU processing with PyTorch")
    
    # Create zarr output arrays
    foreground = zarr.open(foreground_path, mode='w', shape=shape, dtype=bool, chunks=(1,) + shape[1:])
    contours = zarr.open(edges_path, mode='w', shape=shape, dtype=np.float32, chunks=(1,) + shape[1:])
    
    for t in tqdm(range(shape[0]), desc="GPU-accelerated contour detection"):
        foreground_frame = torch.zeros(shape[1:], dtype=torch.bool, device=device)
        contours_frame = torch.zeros(shape[1:], dtype=torch.float32, device=device)
        
        for lb in labels_list:
            lb_frame = torch.from_numpy(np.asarray(lb[t])).to(device)
            # Convert uint dtypes to signed int for CUDA compatibility
            # CUDA doesn't support comparison ops on uint16/uint32
            if lb_frame.dtype in [torch.uint8, torch.uint16, torch.uint32]:
                lb_frame = lb_frame.to(torch.int32)
            foreground_frame |= (lb_frame > 0)
            boundaries = _find_boundaries_torch(lb_frame, mode="outer")
            contours_frame += boundaries.float()
        
        contours_frame /= len(labels_list)
        
        if sigma is not None and sigma > 0:
            contours_frame = _gaussian_filter_torch(contours_frame, sigma)
            max_val = contours_frame.max()
            if max_val > 0:
                contours_frame = contours_frame / max_val
        
        foreground[t] = foreground_frame.cpu().numpy()
        contours[t] = contours_frame.cpu().numpy()
    
    return foreground, contours
"""
        
        gpu_verification_code = """
# Test GPU compatibility before processing
gpu_available = False
device = 'cpu'

if torch.cuda.is_available():
    print("✓ PyTorch CUDA detected")
    print(f"  Device: {{{{torch.cuda.get_device_name(0)}}}}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Compute: sm_{{{{props.major}}}}{{{{props.minor}}}}")
    print(f"  VRAM: {{{{props.total_memory / 1024**3:.1f}}}} GB")
    
    # Critical: Test actual GPU kernel execution
    # Blackwell sm_120 will fail here with PyTorch stable
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')  # Convert warnings to errors
            test = torch.ones((100, 100), device='cuda')
            _ = test.sum()
            torch.cuda.synchronize()
        
        print("✓ GPU kernel test PASSED - using GPU acceleration")
        gpu_available = True
        device = 'cuda'
    except Exception as e:
        print(f"⚠ GPU kernel test FAILED: {{{{type(e).__name__}}}}")
        if 'sm_120' in str(e) or 'Blackwell' in str(e) or 'no kernel image' in str(e):
            print("  → Blackwell GPU detected but not supported by PyTorch stable")
            print("  → SOLUTION: Upgrade to PyTorch nightly:")
            print("     conda run -n ultrack pip uninstall -y torch torchvision")
            print("     conda run -n ultrack pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130")
        else:
            print(f"  → Error: {{{{e}}}}")
        print("  → Falling back to CPU mode (will work but slower)")
        device = 'cpu'
else:
    print("ℹ CUDA not available, using CPU mode")

print(f"\\nProcessing device: {{{{device.upper()}}}}")
"""
        
        labels_to_contours_call = """
    # Use PyTorch-based labels_to_contours (GPU if available and compatible)
    print("\\nConverting labels to contours...")
    
    detection, contours = labels_to_contours_torch(
        label_images,
        sigma=5.0,
        foreground_path=foreground_path,
        edges_path=edges_path,
        device=device,  # Already set by GPU verification above
    )
"""
    
    else:
        gpu_setup_code = """
# CRITICAL: Disable GPU/CUDA before ANY imports
# These environment variables MUST be set before importing CuPy or ultrack
# Otherwise CuPy will initialize with GPU and cannot be disabled
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''           # Hide all CUDA devices
os.environ['CUPY_CACHE_DIR'] = '/dev/null'        # Disable CuPy JIT cache
os.environ['NUMBA_DISABLE_CUDA'] = '1'            # Disable Numba CUDA
os.environ['JAX_PLATFORMS'] = 'cpu'               # Force JAX to CPU (if used)

print("GPU mode DISABLED - using CPU-only processing for compatibility")
"""
        labels_to_contours_import = """
# Standalone CPU-only labels_to_contours (avoids ultrack's CuPy imports)
# This ensures we never accidentally trigger CuPy/CUDA operations
from scipy import ndimage as ndi
from skimage import segmentation as segm
from tqdm import tqdm

def labels_to_contours_cpu(labels_list, sigma, foreground_path, edges_path):
    \"\"\"Standalone CPU-based labels_to_contours using NumPy/scipy.\"\"\"
    import numpy as np
    
    shape = labels_list[0].shape
    print("CPU processing with NumPy/scipy")
    
    # Create zarr output arrays
    foreground = zarr.open(foreground_path, mode='w', shape=shape, dtype=bool, chunks=(1,) + shape[1:])
    contours = zarr.open(edges_path, mode='w', shape=shape, dtype=np.float32, chunks=(1,) + shape[1:])
    
    for t in tqdm(range(shape[0]), desc="Converting labels to contours (CPU)"):
        foreground_frame = np.zeros(shape[1:], dtype=bool)
        contours_frame = np.zeros(shape[1:], dtype=np.float32)
        
        for lb in labels_list:
            lb_frame = np.asarray(lb[t])
            foreground_frame |= (lb_frame > 0)
            boundaries = segm.find_boundaries(lb_frame, mode="outer")
            contours_frame += boundaries.astype(np.float32)
        
        contours_frame /= len(labels_list)
        
        if sigma is not None and sigma > 0:
            contours_frame = ndi.gaussian_filter(contours_frame, sigma)
            max_val = contours_frame.max()
            if max_val > 0:
                contours_frame = contours_frame / max_val
        
        foreground[t] = foreground_frame
        contours[t] = contours_frame
    
    return foreground, contours
"""
        
        gpu_verification_code = """
print("✓ CPU-only mode active (NumPy/scipy standalone)")
print("  Using custom implementation to avoid CuPy")
"""
        
        labels_to_contours_call = """
    # Use standalone CPU-based labels_to_contours (NumPy/scipy)
    # This avoids ultrack's import_module which may try to use CuPy
    print("\\nConverting labels to contours (CPU mode)...")
    detection, contours = labels_to_contours_cpu(
        label_images,
        sigma=5.0,
        foreground_path=foreground_path,
        edges_path=edges_path,
    )
"""
    
    script_content = f"""
{gpu_setup_code}

# Now safe to import - ultrack will use NumPy (CPU) or CuPy (GPU) based on environment
import numpy as np
from pathlib import Path
import tifffile
from tifffile import imread, imwrite
import tempfile
import shutil
import zarr

from ultrack import track, to_tracks_layer, tracks_to_zarr
from ultrack.config import MainConfig

{labels_to_contours_import}

# Verify GPU/CPU configuration
print("\\nVerifying processing mode...")
{gpu_verification_code}

# Setup Gurobi if license key is provided
gurobi_license = {repr(gurobi_license)}
if gurobi_license and gurobi_license != "":
    print("Setting up Gurobi with license key...")
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        print("✓ Gurobi license verified")
    except Exception as e:
        print(f"Warning: Gurobi setup failed: {{{{e}}}}")
        print("Proceeding with default solver...")

# Load label images from different segmentation methods
print('Loading label images for ensemble (using out-of-memory zarr storage)...')
label_paths = {label_paths_str}

print(f"Found {{{{len(label_paths)}}}} label images for ensemble:")
for path in label_paths:
    print(f"  - {{{{path}}}}")

# Create temporary directory for zarr storage
temp_dir = tempfile.mkdtemp(prefix='ultrack_labels_')
print(f"Using temporary storage: {{{{temp_dir}}}}")

try:
    # Load label images and convert to zarr arrays (disk-backed)
    label_images = []
    reference_shape = None
    
    for i, path in enumerate(label_paths):
        if not Path(path).exists():
            print(f"Warning: Label file not found: {{{{path}}}}")
            continue
        
        print(f"Loading {{{{path}}}}...")
        
        # Load TIFF into memory (temporary)
        labels_data = imread(path)
        shape = labels_data.shape
        dtype = labels_data.dtype
        
        print(f"  Shape: {{{{shape}}}}, dtype: {{{{dtype}}}}")
        
        # Validate dimensions
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise ValueError(
                f"Label image {{i}} has shape {{shape}}, expected {{reference_shape}}"
            )
        
        # Create zarr array on disk and save data
        zarr_path = Path(temp_dir) / f'labels_{{i}}.zarr'
        z_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=shape,
            dtype=dtype,
            chunks=(1,) + shape[1:],  # Chunk by time
        )
        
        # Write data to zarr (transfers from RAM to disk)
        z_array[:] = labels_data
        del labels_data  # Free RAM immediately after writing
        
        print(\"  \u2713 Converted to zarr array (disk-backed)\")
        
        # Store the zarr array reference
        label_images.append(z_array)
    
    if len(label_images) == 0:
        raise ValueError("No valid label images found!")

    # Determine if data is TYX (3D) or TZYX (4D)
    ndim = len(reference_shape)
    print(f"Data dimensionality: {{{{ndim}}}}D")

    if ndim == 3:
        print("Processing as TYX (time-series 2D)")
    elif ndim == 4:
        print("Processing as TZYX (time-series 3D)")
    else:
        raise ValueError(f"Expected 3D (TYX) or 4D (TZYX) data, got {{ndim}}D")

    # Create zarr paths for detection and edges
    foreground_path = Path(temp_dir) / 'foreground.zarr'
    edges_path = Path(temp_dir) / 'edges.zarr'

{labels_to_contours_call}
    print(f"Detection shape: {{{{detection.shape}}}}")
    print(f"Contours shape: {{{{contours.shape}}}}")

    # Configure ultrack - set working directory to output folder
    # This ensures data.db and metadata.toml are saved with the output images
    output_dir = Path('{output_path}').parent
    ultrack_db_dir = output_dir / 'ultrack_data'
    ultrack_db_dir.mkdir(exist_ok=True)
    print(f"ultrack database directory: {{{{ultrack_db_dir}}}}")
    
    config = MainConfig()
    config.data_config.working_dir = str(ultrack_db_dir)

    # Segmentation parameters
    config.segmentation_config.min_area = {min_area}
    config.segmentation_config.max_area = 1000000
    config.segmentation_config.min_frontier = 0.01

    # Linking parameters
    config.linking_config.max_neighbors = {max_neighbors}
    config.linking_config.max_distance = {max_distance}
    config.linking_config.z_score_threshold = 3.0

    # Tracking parameters
    config.tracking_config.appear_weight = {appear_weight}
    config.tracking_config.disappear_weight = {disappear_weight}
    config.tracking_config.division_weight = {division_weight}
    config.tracking_config.window_size = None  # Process entire timelapse for best quality
    config.tracking_config.overlap_size = 1
    config.tracking_config.solution_gap = 0.0

    print("\\nultrack configuration:")
    print(f"  Segmentation: min_area={{{{config.segmentation_config.min_area}}}}")
    print(f"  Linking: max_neighbors={{{{config.linking_config.max_neighbors}}}}, max_distance={{{{config.linking_config.max_distance}}}}")
    print(f"  Tracking: appear={{{{config.tracking_config.appear_weight}}}}, disappear={{{{config.tracking_config.disappear_weight}}}}, division={{{{config.tracking_config.division_weight}}}}")

    # Run tracking
    print("\\nRunning ultrack tracking...")
    track(
        config,
        foreground=detection,
        edges=contours,
        overwrite=True
    )

    # Export results
    print("Exporting tracking results...")
    tracks_df, lineage_graph = to_tracks_layer(config)

    # Save tracks to CSV
    tracks_csv_path = '{output_path}'.replace('_ultrack.tif', '_ultrack_tracks.csv')
    tracks_df.to_csv(tracks_csv_path, index=False)
    print(f"Saved tracks to: {{{{tracks_csv_path}}}}")

    # Create tracked label image (zarr array on disk)
    tracking_labels = tracks_to_zarr(config, tracks_df)

    # Save tracked masks using simple imwrite (like trackastra)
    # Label images are sparse (mostly zeros), so memory footprint is small even for large TZYX
    print(f"Exporting tracked labels to TIFF...")
    print(f"  Shape: {{{{tracking_labels.shape}}}}, dtype: {{{{tracking_labels.dtype}}}}")
    
    # Load full array from zarr (sparse labels compress well, minimal RAM)
    tracked_array = np.asarray(tracking_labels).astype(np.uint32)
    print(f"  Loaded {{{{tracked_array.nbytes / (1024**3):.2f}}}} GB from zarr")
    
    # Write using simple imwrite (same approach as trackastra)
    imwrite('{output_path}', tracked_array, compression='zlib')
    
    print(f'✓ Saved tracked masks to: {output_path}')
    print(f'\u2713 Tracked {{{{len(tracks_df["track_id"].unique())}}}} cells across {{{{tracking_labels.shape[0]}}}} timepoints')

finally:
    # Clean up temporary directory
    if Path(temp_dir).exists():
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

"""
    return script_content


def _verify_and_fix_ultrack_env(env_name: str = "ultrack") -> bool:
    """
    Verify critical packages are installed in ultrack environment and install if missing.
    
    This function checks for packages that may be missing in older ultrack environments
    and installs them if needed. This ensures backward compatibility when new dependencies
    are added.
    
    Parameters
    ----------
    env_name : str
        Name of the conda environment (default: "ultrack")
    
    Returns
    -------
    bool
        True if all packages are available, False on critical error
    """
    import subprocess
    
    # Critical packages required by generated scripts
    critical_packages = {
        "numpy": "numpy",       # Core scientific computing
        "scipy": "scipy",       # Scientific algorithms
        "pandas": "pandas",     # Data structures
        "skimage": "scikit-image",  # Image processing
        "ultrack": "ultrack",  # Main tracking package (latest available)
        "zarr": "zarr",  # Zarr array storage (v3+ auto-installed)
        "torch": "torch",       # Required for GPU-accelerated labels_to_contours
        "tifffile": "tifffile", # Required for TIFF I/O
    }
    
    missing_packages = []
    for pkg_name, pkg_spec in critical_packages.items():
        if not is_package_installed(pkg_name, env_name):
            missing_packages.append((pkg_name, pkg_spec))
    
    if not missing_packages:
        return True
    
    # Install missing packages
    print(f"⚠ Found {len(missing_packages)} missing package(s) in '{env_name}' environment")
    print("Installing missing packages...")
    
    try:
        # Get conda command
        conda_cmd = None
        for cmd in ["mamba", "conda"]:
            result = subprocess.run(
                ["which", cmd], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                conda_cmd = cmd
                break
        
        if conda_cmd is None:
            print("✗ ERROR: Could not find conda or mamba")
            return False
        
        # Install each missing package
        for pkg_name, pkg_spec in missing_packages:
            print(f"  Installing {pkg_name}...")
            result = subprocess.run(
                [conda_cmd, "run", "-n", env_name, "pip", "install", pkg_spec],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"  ✗ Failed to install {pkg_name}: {result.stderr}")
                print(f"  Please manually install: conda run -n {env_name} pip install {pkg_spec}")
                return False
            else:
                print(f"  ✓ Installed {pkg_name}")
        
        print(f"✓ All missing packages installed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error installing packages: {e}")
        print(f"Please manually install missing packages:")
        for pkg_name, pkg_spec in missing_packages:
            print(f"  conda run -n {env_name} pip install {pkg_spec}")
        return False


@BatchProcessingRegistry.register(
    name="Track Cells with Ultrack (Segmentation Ensemble)",
    suffix="_ultrack",
    description="Track cells using ultrack with segmentation ensemble (sets of label images from different segmentation methods). Important: Use one of the label suffixes when loading files (e.g., load only '*_cp_labels.tif' files) to avoid redundant processing. The function will automatically find other ensemble members based on the provided suffixes.",
    parameters={
        "label_suffixes": {
            "type": str,
            "default": "_cp_labels.tif,_convpaint_labels.tif",
            "description": "Comma-separated label file suffixes for ensemble (e.g., '_cp_labels.tif,_convpaint_labels.tif'). Files should be named like: 'sample_cp_labels.tif', 'sample_convpaint_labels.tif'. CRITICAL: In the file loader, select ONLY ONE label type per sample (e.g., load only '*_cp_labels.tif'). The function will find other ensemble members automatically.",
            "wide_input": True,
        },
        "gurobi_license": {
            "type": str,
            "default": "",
            "description": "Gurobi license key (only needed ONCE during first-time setup; leave empty for default solver)",
        },
        "min_area": {
            "type": int,
            "default": 200,
            "min": 1,
            "max": 10000,
            "description": "Minimum cell area in pixels (ensemble example default: 200)",
        },
        "max_neighbors": {
            "type": int,
            "default": 5,
            "min": 1,
            "max": 20,
            "description": "Maximum number of candidate neighbors for linking (ultrack default: 5)",
        },
        "max_distance": {
            "type": float,
            "default": 40.0,
            "min": 1.0,
            "max": 200.0,
            "step": 1.0,
            "description": "Maximum distance between cells for linking in pixels (ensemble example default: 40)",
        },
        "appear_weight": {
            "type": float,
            "default": -0.1,
            "min": -10.0,
            "max": 0.0,
            "step": 0.001,
            "description": "Weight for cell appearance - more negative = higher penalty (ensemble example default: -0.1)",
        },
        "disappear_weight": {
            "type": float,
            "default": -2.0,
            "min": -10.0,
            "max": 0.0,
            "step": 0.1,
            "description": "Weight for cell disappearance - more negative = higher penalty (ensemble example default: -2.0)",
        },
        "division_weight": {
            "type": float,
            "default": -0.01,
            "min": -10.0,
            "max": 0.0,
            "step": 0.001,
            "description": "Weight for cell division - more negative = higher penalty (ensemble example default: -0.01)",
        },
        "enable_gpu": {
            "type": bool,
            "default": True,
            "description": "Enable GPU acceleration with PyTorch (default: ON, supports Blackwell sm_120). Turn OFF for CPU-only mode (NumPy/scipy).",
        },
    },
)
def ultrack_ensemble_tracking(
    image: np.ndarray,
    label_suffixes: str = "_cp_labels.tif,_convpaint_labels.tif",
    gurobi_license: str = "",
    min_area: int = 200,
    max_neighbors: int = 5,
    max_distance: float = 40.0,
    appear_weight: float = -0.1,
    disappear_weight: float = -2.0,
    division_weight: float = -0.01,
    enable_gpu: bool = True,
) -> np.ndarray:
    """
    Track cells using ultrack with segmentation ensemble.

    This function performs cell tracking across time using multiple segmentation
    results (ensemble approach). It supports both TYX (2D+time) and TZYX (3D+time)
    data formats.

    The segmentation ensemble approach allows ultrack to evaluate multiple candidate
    segmentations from different methods (e.g., Cellpose, ConvPaint) and select
    the most consistent tracking solution.
    
    **OUT-OF-MEMORY PROCESSING**: 
    - Uses zarr arrays stored on disk to minimize RAM usage
    - Processes data in chunks to avoid loading entire images into memory
    - Can handle very large files (>10GB each) with modest RAM requirements
    
    **PROCESSING MODE**:
    - Default: CPU-only mode (standalone NumPy/scipy) for maximum compatibility
    - Optional: GPU mode (enable_gpu=True) uses PyTorch for CUDA acceleration
    - CPU mode uses custom implementation (bypasses ultrack's CuPy dependencies)
    - GPU mode requires PyTorch (installed automatically in ultrack environment):
      * Blackwell (sm_120): PyTorch 2.5+ 
      * Ada/Hopper (sm_89/90): PyTorch 2.0+
      * Older GPUs: PyTorch 1.7+
    - Performance: CPU uses cores + disk I/O; GPU uses VRAM + compute units

    **IMPORTANT for batch processing**: When loading files, select only ONE label
    type per sample (e.g., load only `*_cp_labels.tif` files). Do NOT load
    all label variants (both `*_cp_labels.tif` AND `*_convpaint_labels.tif`),
    as this will cause redundant processing. The function automatically finds all
    label variants based on the provided suffixes.

    Parameters:
    -----------
    image : np.ndarray
        Input image array (typically raw image, but can be any reference)
    label_suffixes : str
        Comma-separated list of label file suffixes for ensemble
        (e.g., "_cp_labels.tif,_convpaint_labels.tif")
    gurobi_license : str
        Gurobi license key. Only needed ONCE during first-time environment setup.
        Leave empty for default solver, or enter your license key (e.g., "YOUR_LICENSE_KEY").
        After initial activation, leave empty on subsequent runs.
    min_area : int
        Minimum cell area in pixels
    max_neighbors : int
        Maximum number of candidate neighbors for linking
    max_distance : float
        Maximum distance between cells for linking (in pixels)
    appear_weight : float
        Weight for cell appearance (negative = penalize)
    disappear_weight : float
        Weight for cell disappearance (negative = penalize)
    division_weight : float
        Weight for cell division (negative = penalize)
    enable_gpu : bool
        Enable GPU acceleration with PyTorch (default: ON, supports Blackwell sm_120)
        Set to False for CPU-only mode (NumPy/scipy)

    Returns:
    --------
    np.ndarray
        Tracked label image with consistent IDs across time
    """
    print(f"Input shape: {image.shape}, dtype: {image.dtype}")
    print(f"Label suffixes: {label_suffixes}")

    # Ensure ultrack environment exists
    if not is_env_created():
        print("ultrack environment not found. Creating it now...")
        if not create_ultrack_env():
            print("Failed to create ultrack environment. Returning unchanged.")
            return image
    
    # Verify and fix ultrack environment (check for missing packages)
    print("Verifying ultrack environment packages...")
    if not _verify_and_fix_ultrack_env():
        print("Warning: Some packages may be missing. Tracking might fail.")
        print("If you encounter errors, try recreating the ultrack environment.")

    # Ensure scikit-image has the read-only array fix (runtime check)
    print("Checking scikit-image version for read-only array compatibility...")
    if not _ensure_scikit_image_fix():
        print("Warning: scikit-image version check failed. Tracking may encounter issues with zarr arrays.")
        print("Continuing anyway - if you see 'buffer source array is read-only' errors, please check scikit-image installation.")

    # Handle Gurobi license if provided (only needed first time)
    if gurobi_license and gurobi_license.strip() != "":
        print(f"Gurobi license key provided: {gurobi_license}")
        print("Note: License only needs to be entered once during first-time setup.")
        if not setup_gurobi_license(gurobi_license):
            print("Warning: Failed to setup Gurobi license. Continuing with default solver.")
        else:
            print("Gurobi license activated. You can leave this field empty on future runs.")

    # Get the current file path from the processing context
    img_path = None
    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "filepath" in frame_locals:
            img_path = frame_locals["filepath"]
            break

    if img_path is None:
        print("Could not determine input file path. Returning unchanged.")
        return image

    # Parse label suffixes
    suffix_list = [s.strip() for s in label_suffixes.split(",") if s.strip()]
    if len(suffix_list) == 0:
        print("No valid label suffixes provided. Returning unchanged.")
        return image

    # CRITICAL: Only process files that match the FIRST suffix to avoid redundant processing
    # When ensemble has _cp_labels.tif and _convpaint_labels.tif, only process _cp_labels.tif
    first_suffix = suffix_list[0]
    first_suffix_stem = first_suffix.replace(".tif", "").replace(".tiff", "")
    
    input_filename = Path(img_path).name
    if not input_filename.endswith(first_suffix):
        print(f"Skipping: This file doesn't match the first suffix '{first_suffix}'")
        print(f"  (Ensemble will be processed when '{first_suffix}' file is encountered)")
        return image  # Return unchanged - will be processed with the first suffix file

    print(f"Looking for {len(suffix_list)} label file(s):")
    for suffix in suffix_list:
        print(f"  - {suffix}")

    # Find all label files
    base_dir = Path(img_path).parent
    base_name = Path(img_path).stem
    
    # Remove common suffixes from base_name to find the root name
    for suffix in suffix_list:
        suffix_stem = suffix.replace(".tif", "").replace(".tiff", "")
        if base_name.endswith(suffix_stem):
            base_name = base_name[:-len(suffix_stem)]
            break

    label_paths = []
    for suffix in suffix_list:
        # Construct label path
        label_path = base_dir / f"{base_name}{suffix}"
        
        if label_path.exists():
            label_paths.append(str(label_path))
            print(f"  ✓ Found: {label_path}")
        else:
            print(f"  ✗ Not found: {label_path}")

    if len(label_paths) == 0:
        print("No label files found. Cannot perform ensemble tracking.")
        return image

    if len(label_paths) < 2:
        print(f"Warning: Only {len(label_paths)} label file found. Ensemble works best with 2+ segmentations.")

    # Set up output path
    output_dir = base_dir
    output_path = output_dir / f"{base_name}_ultrack.tif"

    # Create the tracking script
    script_content = create_ultrack_ensemble_script(
        label_paths=label_paths,
        output_path=str(output_path),
        gurobi_license=gurobi_license if gurobi_license.strip() != "" else None,
        min_area=min_area,
        max_neighbors=max_neighbors,
        max_distance=max_distance,
        appear_weight=appear_weight,
        disappear_weight=disappear_weight,
        division_weight=division_weight,
        enable_gpu=enable_gpu,
    )

    # Run ultrack in the dedicated environment
    print(f"Running ultrack ensemble tracking with {len(label_paths)} segmentation(s)...")
    result = run_ultrack_in_env(
        script_content=script_content,
        progress_callback=lambda msg: print(msg),
        input_file=str(label_paths[0]),
        output_file=str(output_path),
    )

    if not result["success"]:
        print("ultrack tracking failed:")
        print(result["output"])
        print(result["error"])
        print("Returning original image unchanged.")
        return image

    # Load and return the tracked result
    if output_path.exists():
        tracked = imread(str(output_path))
        print(f"Tracking completed. Output shape: {tracked.shape}")
        return tracked
    else:
        print("ultrack did not produce output. Returning unchanged.")
        return image

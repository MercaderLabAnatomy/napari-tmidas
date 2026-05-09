# processing_functions/cellpose_env_manager.py
"""
This module manages a dedicated virtual environment for Cellpose.
Updated to support Cellpose 4 (Cellpose-SAM) installation.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import suppress

import numpy as np
import tifffile
import zarr

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

        # Prefer stable PyTorch first. If it does not include sm_120 kernels,
        # fall back to nightly for Blackwell compatibility.
        print("Installing stable PyTorch with CUDA 12.8...")
        try:
            subprocess.check_call(
                [
                    env_python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ]
            )
            print("✓ Stable PyTorch CUDA 12.8 installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install stable PyTorch: {e}")
            raise

        sm120_supported = self._torch_supports_sm120(env_python)
        if sm120_supported:
            print("✓ Installed PyTorch build supports sm_120")
        else:
            print(
                "Stable PyTorch build does not include sm_120; "
                "installing nightly CUDA 12.8 build..."
            )
            try:
                subprocess.check_call(
                    [
                        env_python,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--pre",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/nightly/cu128",
                    ]
                )
                print("✓ Nightly PyTorch CUDA 12.8 installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install nightly PyTorch: {e}")
                raise

            if not self._torch_supports_sm120(env_python):
                raise RuntimeError(
                    "Installed PyTorch does not report sm_120 support. "
                    "Please verify the selected wheel channel and GPU driver compatibility."
                )

        # Install packages one by one with error checking
        packages = [
            "cellpose",
            "zarr<3",
            "tifffile",
            "dask[distributed]",
            "dask-jobqueue",
            "dask-image",
        ]
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

    def _torch_supports_sm120(self, env_python: str) -> bool:
        """Return True when torch in the env reports Blackwell (sm_120) support."""
        probe = (
            "import torch; "
            "print('sm_120' in (torch.cuda.get_arch_list() if torch.cuda.is_available() else []))"
        )
        try:
            result = subprocess.run(
                [env_python, "-c", probe],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip().endswith("True")
        except Exception:
            return False

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
        required_packages = [
            ("cellpose", "import cellpose"),
            (
                "zarr<3",
                "import zarr; assert int(zarr.__version__.split('.')[0]) < 3",
            ),
            ("tifffile", "import tifffile"),
            ("dask", "import dask"),
            ("distributed", "import distributed"),
            ("dask-image", "import dask_image"),
        ]

        for package_name, check_code in required_packages:
            try:
                subprocess.run(
                    [env_python, "-c", check_code],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                print(
                    f"Missing package in cellpose environment: {package_name}"
                )
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

    args_dict = dict(args_dict)

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

    source_parent = os.path.dirname(os.path.abspath(zarr_path))
    tmp_root = os.path.join(source_parent, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    if not os.access(tmp_root, os.W_OK):
        raise PermissionError(
            f"Temporary folder is not writable: {tmp_root}. "
            "Please adjust folder permissions."
        )

    # If the caller provides a stable zarr output path (e.g. per-timepoint cache),
    # write directly there. Otherwise create a temporary zarr dir that we delete
    # after reading the result back.
    persist_output_zarr_path = args_dict.get("persist_output_zarr_path")
    if persist_output_zarr_path:
        output_zarr_path = persist_output_zarr_path
        os.makedirs(
            os.path.dirname(os.path.abspath(output_zarr_path)), exist_ok=True
        )
        _temp_output_zarr = None
    else:
        _temp_output_zarr = tempfile.mkdtemp(
            suffix="_cellpose_out.zarr", dir=tmp_root
        )
        output_zarr_path = _temp_output_zarr

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tmp_root
    ) as script_file:

        # Create zarr processing script (similar to working TIFF script)
        script = f"""
import numpy as np
import sys
from cellpose import models, core
import tifffile
import zarr
import os as _os

# Force output to flush immediately for real-time progress
import sys
sys.stdout.flush()

print("=== Cellpose Environment Info ===")
print(f"GPU available in dedicated environment: {{core.use_gpu()}}")
sys.stdout.flush()

import shutil as _shutil
import tempfile as _tempfile


def _get_writable_tmp_dir():
    # Enforce all temp artifacts under input-folder/tmp only.
    _source_parent = _os.path.dirname(_os.path.abspath('{zarr_path}'))
    _tmp_root = _os.path.join(_source_parent, 'tmp')
    _os.makedirs(_tmp_root, exist_ok=True)
    if not _os.access(_tmp_root, _os.W_OK):
        raise PermissionError(
            f"Temporary folder is not writable: {{_tmp_root}}. "
            "Please adjust folder permissions."
        )
    return _tmp_root

USE_DISTRIBUTED = {args_dict.get('use_distributed_segmentation', False)}
BLOCKSIZE = {args_dict.get('distributed_blocksize_yx', args_dict.get('distributed_blocksize', 256))}
BLOCKSIZE_Z = {args_dict.get('distributed_blocksize_z', args_dict.get('distributed_blocksize', 256))}
MASK_PATH = {repr(args_dict.get('distributed_mask_path', None))}
MASK_ZARR_PATH = {repr(args_dict.get('distributed_mask_zarr_path', None))}
TIMEPOINT_INDEX = {repr(args_dict.get('timepoint_index', None))}

_distributed_eval_fn = None
if USE_DISTRIBUTED:
    try:
        from cellpose.contrib.distributed_segmentation import distributed_eval as _distributed_eval_fn
        print(f"distributed_eval loaded (blocksize={{BLOCKSIZE}})")
    except ImportError as _e:
        print(f"WARNING: distributed_eval not available ({{_e}}), falling back to in-memory processing")
sys.stdout.flush()

# Shared eval kwargs used by both distributed and in-memory paths
_EVAL_KWARGS = {{
    'flow_threshold': {args_dict.get('flow_threshold', 0.4)},
    'cellprob_threshold': {args_dict.get('cellprob_threshold', 0.0)},
    'do_3D': {args_dict.get('do_3D', False)},
    'z_axis': 0,
    'normalize': {args_dict.get('normalize', {'tile_norm_blocksize': 128})},
    'batch_size': {args_dict.get('batch_size', 32)},
    'flow3D_smooth': {args_dict.get('flow3D_smooth', 0)},
}}
_DIAMETER = {args_dict.get('diameter', 0.0)}
if _DIAMETER and _DIAMETER > 0:
    _EVAL_KWARGS['diameter'] = _DIAMETER
_ANISOTROPY = {args_dict.get('anisotropy', None)}
if _ANISOTROPY and _ANISOTROPY != 1.0:
    _EVAL_KWARGS['anisotropy'] = _ANISOTROPY

_MODEL_KWARGS = {{'gpu': {args_dict.get('use_gpu', True)}}}
_DASK_LOCAL_DIR = _get_writable_tmp_dir()
_TMP_BASE_DIR = _DASK_LOCAL_DIR
_CLUSTER_KWARGS = {{
    'n_workers': 1,
    'ncpus': 4,
    'memory_limit': '32GB',
    'threads_per_worker': 1,
    'local_directory': _DASK_LOCAL_DIR,
    # cellpose.contrib.myLocalCluster sets dask config temporary-directory
    # from os.getcwd(); override explicitly to a writable location.
    'config': {{'temporary-directory': _DASK_LOCAL_DIR}},
}}
print(f"Dask local_directory: {{_DASK_LOCAL_DIR}}")
print(f"Temporary artifacts base dir: {{_TMP_BASE_DIR}}")
sys.stdout.flush()

_DISTRIBUTED_MASK = None
if MASK_ZARR_PATH:
    print(f"Loading distributed mask zarr: {{MASK_ZARR_PATH}}")
    _DISTRIBUTED_MASK = zarr.open(MASK_ZARR_PATH, mode='r')
    print(f"Distributed mask zarr loaded: shape={{_DISTRIBUTED_MASK.shape}}")
    sys.stdout.flush()
elif MASK_PATH:
    print(f"Loading distributed mask: {{MASK_PATH}}")
    _DISTRIBUTED_MASK = tifffile.imread(MASK_PATH)
    print(
        "Distributed mask loaded: "
        f"shape={{_DISTRIBUTED_MASK.shape}}, "
        f"foreground_fraction={{float(np.mean(_DISTRIBUTED_MASK > 0)):.4f}}"
    )
    sys.stdout.flush()


def _select_mask_for_volume(mask, t_idx=None, c_idx=None):
    if mask is None:
        return None
    if mask.ndim == 5:
        if t_idx is None or c_idx is None:
            return None
        return mask[t_idx, c_idx]
    if mask.ndim == 4:
        # Axis-0 may represent either time or channel depending on source axes.
        if t_idx is not None:
            return mask[t_idx]
        if c_idx is not None:
            return mask[c_idx]
        return None
    if mask.ndim == 3:
        return mask
    return None


def run_distributed_on_slab(slab_zarr, name="", slab_mask=None):
    \"\"\"Run distributed_eval on a zarr.Array (ZYX). Returns numpy uint32 masks.\"\"\"
    tmp_out = _tempfile.mkdtemp(dir=_TMP_BASE_DIR, suffix='.zarr')
    try:
        _blocksize_z = min(BLOCKSIZE_Z, slab_zarr.shape[0]) if slab_zarr.ndim >= 3 else BLOCKSIZE_Z
        print(f"\\nDistributed_eval {{name}}: shape={{slab_zarr.shape}}, blocksize=(z={{_blocksize_z}}, yx={{BLOCKSIZE}})")
        sys.stdout.flush()
        try:
            if slab_mask is not None and slab_mask.shape != slab_zarr.shape:
                print(
                    f"Warning: mask shape {{slab_mask.shape}} does not match "
                    f"slab shape {{slab_zarr.shape}} for {{name}}; ignoring mask"
                )
                slab_mask = None
            segments, boxes = _distributed_eval_fn(
                input_zarr=slab_zarr,
                blocksize=(_blocksize_z, BLOCKSIZE, BLOCKSIZE),
                write_path=tmp_out,
                mask=slab_mask,
                model_kwargs=_MODEL_KWARGS,
                eval_kwargs=_EVAL_KWARGS,
                cluster_kwargs=_CLUSTER_KWARGS,
                temporary_directory=_TMP_BASE_DIR,
            )
        except ValueError as exc:
            # Cellpose 4 distributed_eval can fail on empty/background-only slabs
            # when its merge step reduces over an empty label set.
            exc_msg = str(exc)
            known_empty_merge_errors = (
                "zero-size array to reduction operation maximum",
                "need at least one array to concatenate",
            )
            if not any(msg in exc_msg for msg in known_empty_merge_errors):
                raise

            print(
                f"Distributed_eval {{name}} encountered empty-merge case "
                f"({{exc_msg}}). Falling back to non-distributed eval for this slab."
            )
            sys.stdout.flush()

            try:
                fallback_masks = process_volume(
                    np.array(slab_zarr),
                    f"{{name}} (fallback)",
                )
                return np.asarray(fallback_masks, dtype=np.uint32)
            except Exception as fallback_exc:
                print(
                    f"Fallback eval failed for {{name}} ({{fallback_exc}}); "
                    "returning all-zero mask"
                )
                sys.stdout.flush()
                return np.zeros(slab_zarr.shape, dtype=np.uint32)
        masks = np.array(segments).astype(np.uint32)
        print(f"Distributed_eval {{name}}: found {{np.max(masks)}} objects")
        sys.stdout.flush()
        return masks
    finally:
        _shutil.rmtree(tmp_out, ignore_errors=True)


def slab_to_disk_zarr(source_5d_or_4d, index_slices, shape_zyx, chunks, tmp_dir):
    \"\"\"
    Stream a ZYX slab from a source zarr array to an on-disk temp zarr without
    loading the whole slab into RAM. Reads and writes one Z-chunk row at a time.
    Returns (zarr.Array, zarr_dir) — caller must delete zarr_dir when done.
    \"\"\"
    zarr_dir = _tempfile.mkdtemp(dir=tmp_dir, suffix='_slab.zarr')
    dest = zarr.open_array(zarr_dir, mode='w', shape=shape_zyx,
                           chunks=chunks, dtype=source_5d_or_4d.dtype)
    Z, Y, X = shape_zyx
    cz = chunks[0]
    for z0 in range(0, Z, cz):
        z1 = min(z0 + cz, Z)
        # Build the full index tuple for the source (higher dims + z slice + :, :)
        src_idx = index_slices + (slice(z0, z1), slice(None), slice(None))
        dest[z0:z1, :, :] = source_5d_or_4d[src_idx]
    return dest, zarr_dir


def process_volume(image, name=""):
    print(f"\\nProcessing {{name}}: shape={{image.shape}}, range={{np.min(image):.1f}}-{{np.max(image):.1f}}")
    sys.stdout.flush()

    gpu_available = core.use_gpu()
    use_gpu_requested = {str(args_dict.get('use_gpu', True))}
    actual_use_gpu = use_gpu_requested and gpu_available
    print(f"GPU: requested={{use_gpu_requested}}, available={{gpu_available}}, using={{actual_use_gpu}}")
    sys.stdout.flush()

    print("Creating Cellpose model...")
    sys.stdout.flush()
    model = models.CellposeModel(gpu=actual_use_gpu)
    print(f"Model created (GPU={{model.gpu}})")
    sys.stdout.flush()

    print("Running segmentation...")
    sys.stdout.flush()
    masks, flows, styles = model.eval(
        image,
        flow_threshold={args_dict.get('flow_threshold', 0.4)},
        cellprob_threshold={args_dict.get('cellprob_threshold', 0.0)},
        batch_size={args_dict.get('batch_size', 32)},
        normalize={args_dict.get('normalize', {'tile_norm_blocksize': 128})},
        do_3D={args_dict.get('do_3D', False)},
        flow3D_smooth={args_dict.get('flow3D_smooth', 0)},
        anisotropy={args_dict.get('anisotropy', None)},
        z_axis={args_dict.get('z_axis', 0)} if {args_dict.get('do_3D', False)} else None,
        channel_axis={args_dict.get('channel_axis', None)}
    )
    print(f"Found {{np.max(masks)}} objects in {{name}}")
    sys.stdout.flush()
    return masks


def _create_output_array(shape):
    # Create output zarr at the final destination path with safe chunking.
    shape = tuple(int(s) for s in shape)
    if len(shape) >= 3:
        chunks = (
            (1,) * (len(shape) - 3)
            + (
                max(1, min(BLOCKSIZE, shape[-3])),
                max(1, min(BLOCKSIZE, shape[-2])),
                max(1, min(BLOCKSIZE, shape[-1])),
            )
        )
    else:
        chunks = tuple(max(1, min(BLOCKSIZE, s)) for s in shape)
    return zarr.open_array(
        '{output_zarr_path}',
        mode='w',
        shape=shape,
        chunks=chunks,
        dtype=np.uint32,
    )

def main():
    # Channel selection: "all" → process every channel; integer → process only that channel
    selected_channel = {repr(args_dict.get('channel', 'all'))}

    print("Opening zarr: {zarr_path}")
    sys.stdout.flush()
    zarr_root = zarr.open('{zarr_path}', mode='r')

    if hasattr(zarr_root, 'shape'):
        zarr_source = zarr_root
        if _distributed_eval_fn is not None:
            # zarr_root is itself a zarr.Array — pass directly, no copy needed
            result = run_distributed_on_slab(
                zarr_source,
                "zarr",
                slab_mask=_select_mask_for_volume(_DISTRIBUTED_MASK),
            )
        else:
            result = process_volume(np.array(zarr_source), "zarr")
    else:
        arrays = list(zarr_root.array_keys())
        print(f"Arrays: {{arrays}}")
        sys.stdout.flush()
        zarr_array = zarr_root[arrays[0]]
        print(f"Selected: {{arrays[0]}}, shape={{zarr_array.shape}}")
        sys.stdout.flush()

        chunk_3d = (min(BLOCKSIZE, zarr_array.shape[-3]),
                    min(BLOCKSIZE, zarr_array.shape[-2]),
                    min(BLOCKSIZE, zarr_array.shape[-1]))
        _tmp_slabs = _tempfile.mkdtemp(dir=_TMP_BASE_DIR, suffix='_tmidas_slabs') if _distributed_eval_fn is not None else None

        try:
            if len(zarr_array.shape) == 5:  # TCZYX
                T, C, Z, Y, X = zarr_array.shape
                print(f"5D TCZYX: T={{T}}, C={{C}}, Z={{Z}}, Y={{Y}}, X={{X}}")

                if selected_channel == "all" or selected_channel is None:
                    channels_to_process = list(range(C))
                else:
                    ch = int(selected_channel)
                    if ch < 0 or ch >= C:
                        raise ValueError(f"Selected channel {{ch}} out of range (0-{{C-1}})")
                    channels_to_process = [ch]

                print(f"Processing channels: {{channels_to_process}} ({{len(channels_to_process)}} of {{C}})")
                if TIMEPOINT_INDEX is None:
                    t_indices = list(range(T))
                else:
                    t_req = int(TIMEPOINT_INDEX)
                    if t_req < 0 or t_req >= T:
                        raise ValueError(f"Requested timepoint {{t_req}} out of range (0-{{T-1}})")
                    t_indices = [t_req]
                print(f"Will process {{len(t_indices) * len(channels_to_process)}} T,C combinations")
                sys.stdout.flush()

                n_out_channels = len(channels_to_process)
                if TIMEPOINT_INDEX is not None:
                    if n_out_channels == 1:
                        result_shape = (Z, Y, X)
                    else:
                        result_shape = (n_out_channels, Z, Y, X)
                elif n_out_channels == 1:
                    result_shape = (len(t_indices), Z, Y, X)
                else:
                    result_shape = (len(t_indices), n_out_channels, Z, Y, X)
                result = _create_output_array(result_shape)

                for out_t, t in enumerate(t_indices):
                    for out_c, c in enumerate(channels_to_process):
                        print(f"\\n=== T={{t+1}}/{{T}}, C={{c+1}}/{{C}} ===")
                        sys.stdout.flush()
                        if _distributed_eval_fn is not None:
                            slab_z, slab_dir = slab_to_disk_zarr(
                                zarr_array, (t, c), (Z, Y, X), chunk_3d, _tmp_slabs)
                            slab_mask = _select_mask_for_volume(
                                _DISTRIBUTED_MASK, t_idx=t, c_idx=c
                            )
                            try:
                                masks = run_distributed_on_slab(
                                    slab_z,
                                    f"T{{t+1}}C{{c+1}}",
                                    slab_mask=slab_mask,
                                )
                            finally:
                                _shutil.rmtree(slab_dir, ignore_errors=True)
                        else:
                            masks = process_volume(np.array(zarr_array[t, c, :, :, :]), f"T{{t+1}}C{{c+1}}")

                        if TIMEPOINT_INDEX is not None:
                            if n_out_channels == 1:
                                result[:, :, :] = masks
                            else:
                                result[out_c, :, :, :] = masks
                        elif n_out_channels == 1:
                            result[out_t, :, :, :] = masks
                        else:
                            result[out_t, out_c, :, :, :] = masks

            elif len(zarr_array.shape) == 4:  # 4D (T or C)ZYX
                dim1, Z, Y, X = zarr_array.shape
                print(f"4D array: dim1={{dim1}}, Z={{Z}}, Y={{Y}}, X={{X}}")
                sys.stdout.flush()

                # Determine whether axis-0 is time or channel from OME metadata.
                axis0_role = "time"
                try:
                    root_attrs = dict(zarr_root.attrs) if hasattr(zarr_root, "attrs") else {{}}
                    root_multiscales = root_attrs.get("multiscales", [])
                    if root_multiscales:
                        root_axes = root_multiscales[0].get("axes", [])
                        if root_axes:
                            axis0 = root_axes[0]
                            axis0_name = (
                                axis0.get("name", "").lower()
                                if isinstance(axis0, dict)
                                else str(axis0).lower()
                            )
                            axis0_type = (
                                axis0.get("type", "").lower()
                                if isinstance(axis0, dict)
                                else ""
                            )
                            if axis0_name in ("c", "channel", "ch") or axis0_type == "channel":
                                axis0_role = "channel"
                            elif axis0_name in ("t", "time") or axis0_type == "time":
                                axis0_role = "time"
                except Exception as _axis_exc:
                    print(
                        f"Warning: Could not parse 4D axis metadata ({{_axis_exc}}); "
                        "assuming axis-0 is time"
                    )

                print(f"4D axis-0 interpretation: {{axis0_role}}")
                sys.stdout.flush()

                if axis0_role == "time":
                    if TIMEPOINT_INDEX is None:
                        indices = list(range(dim1))
                    else:
                        t_req = int(TIMEPOINT_INDEX)
                        if t_req < 0 or t_req >= dim1:
                            raise ValueError(
                                f"Requested timepoint {{t_req}} out of range (0-{{dim1-1}})"
                            )
                        indices = [t_req]

                    if TIMEPOINT_INDEX is not None:
                        result_shape = (Z, Y, X)
                    else:
                        result_shape = (len(indices), Z, Y, X)
                    result = _create_output_array(result_shape)
                    for out_i, i in enumerate(indices):
                        print(f"\\n=== Timepoint {{i+1}}/{{dim1}} ===")
                        sys.stdout.flush()
                        if _distributed_eval_fn is not None:
                            slab_z, slab_dir = slab_to_disk_zarr(
                                zarr_array, (i,), (Z, Y, X), chunk_3d, _tmp_slabs
                            )
                            slab_mask = _select_mask_for_volume(
                                _DISTRIBUTED_MASK, t_idx=i
                            )
                            try:
                                masks = run_distributed_on_slab(
                                    slab_z,
                                    f"T{{i+1}}",
                                    slab_mask=slab_mask,
                                )
                            finally:
                                _shutil.rmtree(slab_dir, ignore_errors=True)
                        else:
                            masks = process_volume(
                                np.array(zarr_array[i, :, :, :]), f"T{{i+1}}"
                            )

                        if TIMEPOINT_INDEX is not None:
                            result[:, :, :] = masks
                        else:
                            result[out_i, :, :, :] = masks

                else:
                    if TIMEPOINT_INDEX is not None:
                        print(
                            "TIMEPOINT_INDEX provided for channel-first 4D data; "
                            "ignoring timepoint filter"
                        )

                    if selected_channel == "all" or selected_channel is None:
                        indices = list(range(dim1))
                    else:
                        c_req = int(selected_channel)
                        if c_req < 0 or c_req >= dim1:
                            raise ValueError(
                                f"Selected channel {{c_req}} out of range (0-{{dim1-1}})"
                            )
                        indices = [c_req]

                    if len(indices) == 1:
                        result_shape = (Z, Y, X)
                    else:
                        result_shape = (len(indices), Z, Y, X)
                    result = _create_output_array(result_shape)
                    for out_i, i in enumerate(indices):
                        print(f"\\n=== Channel {{i+1}}/{{dim1}} ===")
                        sys.stdout.flush()
                        if _distributed_eval_fn is not None:
                            slab_z, slab_dir = slab_to_disk_zarr(
                                zarr_array, (i,), (Z, Y, X), chunk_3d, _tmp_slabs
                            )
                            slab_mask = _select_mask_for_volume(
                                _DISTRIBUTED_MASK, c_idx=i
                            )
                            try:
                                masks = run_distributed_on_slab(
                                    slab_z,
                                    f"C{{i+1}}",
                                    slab_mask=slab_mask,
                                )
                            finally:
                                _shutil.rmtree(slab_dir, ignore_errors=True)
                        else:
                            masks = process_volume(
                                np.array(zarr_array[i, :, :, :]), f"C{{i+1}}"
                            )

                        if len(indices) == 1:
                            result[:, :, :] = masks
                        else:
                            result[out_i, :, :, :] = masks

            else:  # bare 3D ZYX — zarr_array is already a zarr.Array on disk
                if _distributed_eval_fn is not None:
                    result = run_distributed_on_slab(
                        zarr_array,
                        "3D",
                        slab_mask=_select_mask_for_volume(_DISTRIBUTED_MASK),
                    )
                else:
                    result = process_volume(np.array(zarr_array), "3D")

        finally:
            if _tmp_slabs is not None:
                _shutil.rmtree(_tmp_slabs, ignore_errors=True)

    print(f"Saving results: shape={{result.shape}}, total objects={{np.max(result)}}")
    sys.stdout.flush()
    if isinstance(result, np.ndarray):
        zarr.save('{output_zarr_path}', result.astype(np.uint32))
    print("Complete!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
"""

        script_file.write(script)
        script_file.flush()

        try:
            # Run with REAL-TIME output (don't capture, let it stream)
            env_python = get_env_python_path()
            print("=== Starting Cellpose Processing ===")

            # Keep cellpose/dask scratch creation under input-folder/tmp.
            subprocess_cwd = tmp_root

            # Use Popen for real-time progress and cancellation support
            process = subprocess.Popen(
                [env_python, script_file.name],
                cwd=subprocess_cwd,
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

            # Check if output zarr was created
            if not os.path.exists(output_zarr_path):
                raise RuntimeError("Output zarr was not created")

            print(f"Reading result from: {output_zarr_path}")
            result = np.array(zarr.open(output_zarr_path, mode="r"))
            return result

        finally:
            # Cleanup script; delete temp zarr only (stable cache is kept by caller)
            with suppress(OSError, FileNotFoundError):
                os.unlink(script_file.name)
            if _temp_output_zarr is not None:
                shutil.rmtree(_temp_output_zarr, ignore_errors=True)


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
    z_axis={args_dict.get('z_axis', 0)} if {args_dict.get('do_3D', False)} else None,
    channel_axis={args_dict.get('channel_axis', None)}
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

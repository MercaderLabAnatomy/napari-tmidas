#!/usr/bin/env python3
"""
TrackAstra Cell Tracking Module for napari-tmidas

This module integrates TrackAstra deep learning-based cell tracking into the
napari-tmidas batch processing framework. It uses a dedicated conda environment
to manage TrackAstra dependencies separately from the main environment.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imread

# Add the registry import
from napari_tmidas._registry import BatchProcessingRegistry


_SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".zarr")


def _strip_known_image_suffix(name: str) -> str:
    """Strip known image suffixes from a basename."""
    lower_name = name.lower()
    for suffix in _SUPPORTED_IMAGE_SUFFIXES:
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _raw_candidates_from_label_name(label_name: str, label_pattern: str):
    """Build possible raw basenames from a label basename and label pattern."""
    if label_name.endswith(label_pattern):
        raw_base = label_name[: -len(label_pattern)]
    else:
        raw_base = label_name.replace(label_pattern, "", 1)

    if raw_base.lower().endswith(_SUPPORTED_IMAGE_SUFFIXES):
        return raw_base, [raw_base]

    return raw_base, [raw_base + suffix for suffix in _SUPPORTED_IMAGE_SUFFIXES]


def _find_matching_raw_path(label_path: str, label_pattern: str):
    """Find a raw image path (tif/tiff/zarr) corresponding to a label image path."""
    label_name = os.path.basename(label_path)
    raw_base, candidates = _raw_candidates_from_label_name(label_name, label_pattern)
    label_dir = os.path.dirname(label_path)

    for candidate in candidates:
        candidate_path = os.path.join(label_dir, candidate)
        if os.path.exists(candidate_path):
            return raw_base, candidates, candidate_path

    return raw_base, candidates, None


def _load_zarr_array(path: str) -> np.ndarray:
    """Load a zarr array/group as a NumPy array, with robust fallbacks."""
    import zarr

    try:
        zobj = zarr.open(path, mode="r")
    except Exception:
        # Some datasets keep multiscale levels (e.g., 0, 1, 2, ...) without
        # opening cleanly at the root. Fall back to level 0 if present.
        level0 = os.path.join(path, "0")
        if os.path.exists(level0):
            zobj = zarr.open(level0, mode="r")
        else:
            raise

    if hasattr(zobj, "shape"):
        return np.asarray(zobj)

    if not hasattr(zobj, "array_keys"):
        # Some zarr-like objects are array-convertible without exposing group APIs.
        return np.asarray(zobj)

    array_keys = list(zobj.array_keys())
    if "0" in array_keys:
        return np.asarray(zobj["0"])
    if array_keys:
        return np.asarray(zobj[array_keys[0]])

    raise ValueError(f"No arrays found in zarr group: {path}")


class TrackAstraEnvManager:
    """Manages the TrackAstra conda environment."""

    ENV_NAME = "trackastra"
    REQUIRED_VERSIONS = {
        "python": "3.11",
        "gurobipy": "13.0.0",
        "ilpy": "0.5.1",
        "motile": "0.4.0",
        "trackastra": "0.5.3",
        "zarr": "3.0.0",
    }

    @staticmethod
    def get_conda_cmd():
        """Get the conda/mamba command available on the system."""
        # Try mamba first (faster)
        if shutil.which("mamba"):
            return "mamba"
        elif shutil.which("conda"):
            return "conda"
        else:
            raise RuntimeError(
                "Neither conda nor mamba found. Please install Anaconda/Miniconda/Miniforge."
            )

    @classmethod
    def check_env_exists(cls):
        conda_cmd = cls.get_conda_cmd()
        try:
            # Try running python --version in the env
            result = subprocess.run(
                [conda_cmd, "run", "-n", cls.ENV_NAME, "python", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return False

    @staticmethod
    def _version_tuple(version_str):
        """Convert a version string to a comparable integer tuple."""
        if version_str is None:
            return tuple()
        nums = re.findall(r"\d+", str(version_str))
        if not nums:
            return tuple()
        return tuple(int(n) for n in nums)

    @classmethod
    def _version_at_least(cls, found, required):
        found_tuple = cls._version_tuple(found)
        required_tuple = cls._version_tuple(required)
        if not found_tuple or not required_tuple:
            return False

        # Compare only up to the required precision (e.g. 3.10, 0.5.1)
        found_norm = found_tuple[: len(required_tuple)]
        if len(found_norm) < len(required_tuple):
            found_norm = found_norm + (0,) * (len(required_tuple) - len(found_norm))
        return found_norm >= required_tuple


    @classmethod
    def get_env_status(cls):
        """Return package/version status for the TrackAstra environment."""
        conda_cmd = cls.get_conda_cmd()
        check_script = r'''
import importlib.util
import json
import sys

status = {
    "python": sys.version.split()[0],
    "packages": {},
}

for name in ["gurobipy", "ilpy", "motile", "trackastra", "zarr"]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        status["packages"][name] = {"present": False, "version": None}
        continue

    mod = __import__(name)
    if name == "gurobipy":
        version = ".".join(str(x) for x in mod.gurobi.version())
    else:
        version = getattr(mod, "__version__", None)
    status["packages"][name] = {"present": True, "version": version}

print(json.dumps(status))
'''
        try:
            result = subprocess.run(
                [
                    conda_cmd,
                    "run",
                    "-n",
                    cls.ENV_NAME,
                    "python",
                    "-c",
                    check_script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            import json

            return json.loads(result.stdout.strip())
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def env_needs_repair(cls, status):
        """Determine whether TrackAstra env should be repaired/upgraded."""
        if not status or "error" in status:
            return True, ["Could not determine environment status"]

        reasons = []

        python_version = status.get("python")
        if not cls._version_at_least(python_version, cls.REQUIRED_VERSIONS["python"]):
            reasons.append(
                f"Python {python_version} < required {cls.REQUIRED_VERSIONS['python']}"
            )

        packages = status.get("packages", {})
        for pkg in ["gurobipy", "ilpy", "motile", "trackastra", "zarr"]:
            info = packages.get(pkg, {})
            if not info.get("present"):
                reasons.append(f"Missing package: {pkg}")
                continue
            found = info.get("version")
            required = cls.REQUIRED_VERSIONS[pkg]
            if not cls._version_at_least(found, required):
                reasons.append(f"{pkg} version {found} < required {required}")

        return (len(reasons) > 0), reasons

    @classmethod
    def repair_env(cls):
        """Repair/upgrade TrackAstra environment using upstream ILP recipe."""
        print("Repairing TrackAstra environment to required package versions...")
        conda_cmd = cls.get_conda_cmd()
        try:
            # Keep solver stack aligned with TrackAstra ILP requirements.
            solver_cmd = [
                conda_cmd,
                "install",
                "-n",
                cls.ENV_NAME,
                "-c",
                "conda-forge",
                "-c",
                "gurobi",
                "-c",
                "funkelab",
                "ilpy",
                "gurobi",
                "-y",
            ]
            subprocess.run(solver_cmd, check=True)

            pip_cmd = [
                conda_cmd,
                "run",
                "-n",
                cls.ENV_NAME,
                "pip",
                "install",
                "--upgrade",
                "trackastra[ilp]",
                "motile",
                "zarr>=3",
            ]
            subprocess.run(pip_cmd, check=True)

            print("TrackAstra environment repair completed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error repairing TrackAstra environment: {e}")
            return False

    @classmethod
    def create_env(cls):
        """Create the TrackAstra conda environment if it doesn't exist."""
        if cls.check_env_exists():
            print("TrackAstra environment already exists.")
            return True

        print("Creating TrackAstra conda environment...")
        conda_cmd = cls.get_conda_cmd()

        # Create environment with Python 3.11+ (required for zarr>=3; TrackAstra supports 3.10-3.13)
        env_create_cmd = [
            conda_cmd,
            "create",
            "-n",
            cls.ENV_NAME,
            "python=3.11",
            "--no-default-packages",
            "-y",
        ]

        try:
            subprocess.run(env_create_cmd, check=True)

            # Install ilpy first from conda-forge
            ilpy_cmd = [
                conda_cmd,
                "install",
                "-n",
                cls.ENV_NAME,
                "-c",
                "conda-forge",
                "-c",
                "gurobi",
                "-c",
                "funkelab",
                "ilpy",
                "gurobi",
                "-y",
            ]
            subprocess.run(ilpy_cmd, check=True)

            # Install TrackAstra ILP extras via pip (upstream recipe).
            pip_packages = [
                "trackastra[ilp]",
                "motile",
                "zarr>=3",
            ]

            pip_cmd = [
                conda_cmd,
                "run",
                "-n",
                cls.ENV_NAME,
                "pip",
                "install",
            ] + pip_packages

            subprocess.run(pip_cmd, check=True)

            print("TrackAstra environment created successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error creating TrackAstra environment: {e}")
            return False

    @classmethod
    def ensure_env_ready(cls):
        """Ensure environment exists and required package versions are present."""
        if not cls.check_env_exists():
            print("TrackAstra environment not found. Creating it now...")
            if not cls.create_env():
                return False

        status = cls.get_env_status()
        needs_repair, reasons = cls.env_needs_repair(status)
        if needs_repair:
            print("TrackAstra environment drift detected:")
            for reason in reasons:
                print(f" - {reason}")

            # If the Python version itself is wrong the only fix is a full
            # env rebuild — pip installs cannot change the interpreter.
            python_version = (status or {}).get("python", "")
            python_ok = cls._version_at_least(python_version, cls.REQUIRED_VERSIONS["python"])
            if not python_ok:
                print(
                    f"Python {python_version} < required "
                    f"{cls.REQUIRED_VERSIONS['python']}; "
                    "recreating environment..."
                )
                conda_cmd = cls.get_conda_cmd()
                try:
                    subprocess.run(
                        [conda_cmd, "env", "remove", "-n", cls.ENV_NAME, "-y"],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error removing old environment: {e}")
                    return False
                if not cls.create_env():
                    return False
            else:
                if not cls.repair_env():
                    return False

            # Recheck after attempted repair/recreate.
            status = cls.get_env_status()
            needs_repair, reasons = cls.env_needs_repair(status)
            if needs_repair:
                print("TrackAstra environment is still not healthy after repair:")
                for reason in reasons:
                    print(f" - {reason}")
                return False

        print("TrackAstra environment is ready.")
        return True


def create_trackastra_script(
    img_path,
    mask_path,
    model,
    mode,
    output_path,
    channel="all",
    dimension_order="Auto",
):
    """Create a Python script to run TrackAstra in the dedicated environment."""
    
    # Determine if inputs are zarr files
    img_is_zarr = str(img_path).lower().endswith(".zarr")
    mask_is_zarr = str(mask_path).lower().endswith(".zarr")
    
    # Build image loading code based on file type
    if img_is_zarr:
        img_load_code = f"""
import zarr
import os
print('Loading zarr image: {img_path}')
def _zarr_group_to_array(zgroup):
    if hasattr(zgroup, 'shape'):
        return np.asarray(zgroup)

    if hasattr(zgroup, 'array_keys'):
        array_keys = list(zgroup.array_keys())
        if '0' in array_keys:
            return np.asarray(zgroup['0'])
        if array_keys:
            return np.asarray(zgroup[array_keys[0]])

    if hasattr(zgroup, 'group_keys'):
        subgroup_keys = sorted(list(zgroup.group_keys()), key=lambda k: (k != '0', k))
        for key in subgroup_keys:
            try:
                arr = _zarr_group_to_array(zgroup[key])
                if arr is not None:
                    return arr
            except Exception:
                continue

    return None

def _load_zarr_any(path):
    attempts = []

    def _open_with_kind(kind, location):
        if kind == 'open':
            return zarr.open(location, mode='r')
        if kind == 'open_array':
            return zarr.open_array(store=location, mode='r')
        if kind == 'open_group':
            return zarr.open_group(store=location, mode='r')
        raise ValueError(f'Unknown zarr opener kind: {{kind}}')

    for kind in ('open', 'open_array', 'open_group'):
        try:
            obj = _open_with_kind(kind, path)
            arr = _zarr_group_to_array(obj)
            if arr is not None:
                return arr
        except Exception as exc:
            attempts.append(f"{{kind}}({{path}}): {{exc}}")

    level0 = os.path.join(path, '0')
    for kind in ('open', 'open_array', 'open_group'):
        try:
            obj = _open_with_kind(kind, level0)
            arr = _zarr_group_to_array(obj)
            if arr is not None:
                print(f'Loaded zarr via level 0 fallback: {{level0}}')
                return arr
        except Exception as exc:
            attempts.append(f"{{kind}}({{level0}}): {{exc}}")

    # Last resort: look for a direct child directory named "0".
    try:
        for child in sorted(os.listdir(path)):
            child_path = os.path.join(path, child)
            child_level0 = os.path.join(child_path, '0')
            if os.path.isdir(child_level0):
                for kind in ('open', 'open_array', 'open_group'):
                    try:
                        obj = _open_with_kind(kind, child_level0)
                        arr = _zarr_group_to_array(obj)
                        if arr is not None:
                            print(f'Loaded zarr via nested level 0 fallback: {{child_level0}}')
                            return arr
                    except Exception as exc:
                        attempts.append(f"{{kind}}({{child_level0}}): {{exc}}")
    except Exception as exc:
        attempts.append(f"os.listdir({{path}}): {{exc}}")

    raise RuntimeError(
        'Unable to load zarr image. Attempts:\\n' + '\\n'.join(attempts)
    )

img = _load_zarr_any('{img_path}')
"""
    else:
        img_load_code = f"img = imread('{img_path}')"
    
    # Build mask loading code based on file type
    if mask_is_zarr:
        mask_load_code = f"""
import zarr
import os
print('Loading zarr mask: {mask_path}')
def _zarr_group_to_array(zgroup):
    if hasattr(zgroup, 'shape'):
        return np.asarray(zgroup)

    if hasattr(zgroup, 'array_keys'):
        array_keys = list(zgroup.array_keys())
        if '0' in array_keys:
            return np.asarray(zgroup['0'])
        if array_keys:
            return np.asarray(zgroup[array_keys[0]])

    if hasattr(zgroup, 'group_keys'):
        subgroup_keys = sorted(list(zgroup.group_keys()), key=lambda k: (k != '0', k))
        for key in subgroup_keys:
            try:
                arr = _zarr_group_to_array(zgroup[key])
                if arr is not None:
                    return arr
            except Exception:
                continue

    return None

def _load_zarr_any(path):
    attempts = []

    def _open_with_kind(kind, location):
        if kind == 'open':
            return zarr.open(location, mode='r')
        if kind == 'open_array':
            return zarr.open_array(store=location, mode='r')
        if kind == 'open_group':
            return zarr.open_group(store=location, mode='r')
        raise ValueError(f'Unknown zarr opener kind: {{kind}}')

    for kind in ('open', 'open_array', 'open_group'):
        try:
            obj = _open_with_kind(kind, path)
            arr = _zarr_group_to_array(obj)
            if arr is not None:
                return arr
        except Exception as exc:
            attempts.append(f"{{kind}}({{path}}): {{exc}}")

    level0 = os.path.join(path, '0')
    for kind in ('open', 'open_array', 'open_group'):
        try:
            obj = _open_with_kind(kind, level0)
            arr = _zarr_group_to_array(obj)
            if arr is not None:
                print(f'Loaded zarr via level 0 fallback: {{level0}}')
                return arr
        except Exception as exc:
            attempts.append(f"{{kind}}({{level0}}): {{exc}}")

    try:
        for child in sorted(os.listdir(path)):
            child_path = os.path.join(path, child)
            child_level0 = os.path.join(child_path, '0')
            if os.path.isdir(child_level0):
                for kind in ('open', 'open_array', 'open_group'):
                    try:
                        obj = _open_with_kind(kind, child_level0)
                        arr = _zarr_group_to_array(obj)
                        if arr is not None:
                            print(f'Loaded zarr via nested level 0 fallback: {{child_level0}}')
                            return arr
                    except Exception as exc:
                        attempts.append(f"{{kind}}({{child_level0}}): {{exc}}")
    except Exception as exc:
        attempts.append(f"os.listdir({{path}}): {{exc}}")

    raise RuntimeError(
        'Unable to load zarr mask. Attempts:\\n' + '\\n'.join(attempts)
    )

mask = _load_zarr_any('{mask_path}')
"""
    else:
        mask_load_code = f"mask = imread('{mask_path}')"
    
    script_content = f"""
import sys
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks


# Load images
print('Loading images...')
{img_load_code}
{mask_load_code}
print(f'Img shape: {{img.shape}}, Mask shape: {{mask.shape}}')


# Validate and fix image dimensions
if mask.ndim not in [3, 4]:
    raise ValueError(f'Expected 3D (TYX) or 4D (TZYX) mask, got {{mask.ndim}}D')

if mask.shape[0] < 2:
    raise ValueError(f'Need at least 2 timepoints, got {{mask.shape[0]}}')

# Handle multichannel images by selecting a channel on the axis that makes
# image and mask shapes compatible.
channel_param = "{channel}"
dimension_order_param = "{dimension_order}"
if img.ndim == mask.ndim + 1:
    # Find candidate channel axes whose removal makes img shape match mask shape.
    candidate_axes = [
        ax for ax in range(1, img.ndim)
        if img.shape[:ax] + img.shape[ax + 1:] == mask.shape
    ]
    requested_axis = None
    if dimension_order_param and str(dimension_order_param).upper() not in ("AUTO", "NONE"):
        axes = str(dimension_order_param).upper()
        if len(axes) == img.ndim and "C" in axes:
            requested_axis = axes.index("C")
            print(f'Using dimension_order={{axes}} requested channel axis {{requested_axis}}')
        else:
            print(
                f'Ignoring dimension_order={{axes}} for img.ndim={{img.ndim}}; '
                'expected one C and matching length'
            )

    if requested_axis is not None and requested_axis in candidate_axes:
        channel_axis = requested_axis
        print(f'Using requested channel axis {{channel_axis}} validated by mask shape')
    elif candidate_axes:
        channel_axis = candidate_axes[0]
        print(f'Inferred channel axis {{channel_axis}} from mask shape compatibility')
    else:
        # Default convention for 5D is TCZYX -> channel axis 1.
        channel_axis = 1 if img.ndim == 5 else img.ndim - 1
        print(
            f'Could not infer channel axis from shapes {{img.shape}} and {{mask.shape}}; '
            f'using fallback axis {{channel_axis}}'
        )

    n_channels = img.shape[channel_axis]
    if channel_param in ("", "all", "None"):
        ch_idx = 0
        print('No channel specified for multichannel image, taking first channel...')
    else:
        try:
            ch_idx = int(channel_param)
        except ValueError:
            print(f'Invalid channel {{channel_param}}, taking first channel...')
            ch_idx = 0

    if ch_idx < 0 or ch_idx >= n_channels:
        print(f'Channel index {{ch_idx}} out of bounds for {{n_channels}} channels; using 0')
        ch_idx = 0

    print(f'Extracting channel {{ch_idx}} on axis {{channel_axis}}...')
    img = np.take(img, ch_idx, axis=channel_axis)
    print(f'Image shape after channel selection: {{img.shape}}')
elif img.ndim == mask.ndim:
    if img.shape != mask.shape:
        raise ValueError(
            f'Image and mask shapes must match when dimensions are equal: '
            f'img={{img.shape}}, mask={{mask.shape}}'
        )
else:
    raise ValueError(
        f'Unexpected dimensionality relationship: img={{img.shape}} (ndim={{img.ndim}}), '
        f'mask={{mask.shape}} (ndim={{mask.ndim}})'
    )

print(f'Final shapes - Img: {{img.shape}}, Mask: {{mask.shape}}')

model = Trackastra.from_pretrained('{model}', device="automatic")
try:
    track_result = model.track(img, mask, mode='{mode}')
except Exception as exc:
    if '{mode}' == 'ilp':
        print(f'Warning: TrackAstra ILP mode failed ({{exc}}). Retrying with greedy mode...')
        track_result = model.track(img, mask, mode='greedy')
    else:
        raise

# Trackastra API compatibility:
# - newer versions may return (track_graph, masks_tracked)
# - older versions return only track_graph
# Always run graph_to_ctc to enforce consistent track-based relabeling.
if isinstance(track_result, tuple):
    if len(track_result) < 1:
        raise RuntimeError('Unexpected empty tuple returned by Trackastra.track().')
    track_graph = track_result[0]
else:
    track_graph = track_result

if not hasattr(track_graph, 'nodes'):
    raise RuntimeError(
        f'Unexpected Trackastra.track() return type for graph: {{type(track_graph)}}'
    )

_, masks_tracked = graph_to_ctc(track_graph, mask, outdir=None)

same_as_input = np.array_equal(mask, masks_tracked)
in_ids = np.unique(mask)
out_ids = np.unique(masks_tracked)
print(
    f'Relabel check: identical_to_input={{same_as_input}}, '
    f'unique_ids_in={{len(in_ids)}}, unique_ids_out={{len(out_ids)}}'
)

# Save the tracked masks
imwrite('{output_path}', masks_tracked.astype(np.uint32), compression='zlib')
print(f'Saved tracked masks to: {output_path}')

"""

    return script_content


@BatchProcessingRegistry.register(
    name="Track Cells with Trackastra",
    suffix="_tracked",
    description="Track cells across time using TrackAstra deep learning (expects TYX or TZYX label images). Supports TIFF and zarr inputs with optional channel selection for multichannel images.",
    parameters={
        "model": {
            "type": str,
            "default": "ctc",
            "description": "general_2d (nuclei/cells/particles) or ctc (Cell Tracking Challenge; 2D/3D)",
        },
        "mode": {
            "type": str,
            "default": "greedy",
            "description": "greedy (fast), ilp (accurate with divisions), greedy_nodiv",
        },
        "channel": {
            "type": str,
            "default": "",
            "description": "Optional raw-image channel index for multichannel input. Leave empty to use the default first channel.",
        },
        "dimension_order": {
            "type": str,
            "default": "Auto",
            "options": ["Auto", "TYX", "TZYX", "TCZYX", "TCYX"],
            "description": "Dimension order hint for raw images (e.g., TCZYX for time-Z-channel-Y-X). Helps with channel detection when loading label files.",
        },
        "label_pattern": {
            "type": str,
            "default": "_labels.tif",
            "description": " ",
        },
    },
)
def trackastra_tracking(
    image: np.ndarray,
    model: str = "ctc",
    mode: str = "greedy",
    channel: str = "",
    dimension_order: str = "Auto",
    label_pattern: str = "_labels.tif",
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = "_tracked",
) -> np.ndarray:
    """
    Track cells in time-lapse label images using TrackAstra.

    This function takes a time series of segmentation masks and performs
    automatic cell tracking using TrackAstra deep learning framework.

    Supports both TIFF and zarr input files. For multichannel images (TCZYX),
    optionally extracts a user-specified raw-image channel before tracking.

    Expected input dimensions:
    - TYX: Time series of 2D label images
    - TZYX: Time series of 3D label images (will process each Z-slice separately)
    - TCZYX: Multichannel time series (optional channel selection via 'channel' parameter)

    Input file formats:
    - TIFF (.tif, .tiff files)
    - Zarr (.zarr directories, including OME-Zarr with multiple arrays)

    Parameters:
    -----------
    image : np.ndarray
        Input label image array with time as first dimension
    model : str
        TrackAstra model: 'general_2d' or 'ctc' (default: "ctc")
    mode : str
        Tracking mode: 'greedy', 'ilp', or 'greedy_nodiv' (default: "greedy")
    channel : str
        Optional channel index for multichannel raw images. Leave empty to use
        the default first channel when needed.
    dimension_order : str
        Dimension order hint for raw images (e.g., "TCZYX", "TZYX"). If "Auto",
        dimensions are auto-detected. This helps clarify channel detection when
        processing label files.
    label_pattern : str
        To identify label images

    Returns:
    --------
    np.ndarray
        Tracked label image with consistent IDs across time
    """
    print(f"Input shape: {image.shape}, dtype: {image.dtype}")

    # Validate input
    if image.ndim < 3:
        print(
            "Input is not a time series (needs at least 3 dimensions). Returning unchanged."
        )
        return image

    if image.shape[0] < 2:
        print(
            "Input has only one timepoint. Need at least 2 for tracking. Returning unchanged."
        )
        return image

    # Ensure TrackAstra environment exists and has compatible package versions.
    if not TrackAstraEnvManager.ensure_env_ready():
        print("Failed to prepare TrackAstra environment. Returning unchanged.")
        return image

    # Get source file path. Prefer explicit worker-provided path.
    img_path = _source_filepath
    if img_path is None:
        import inspect

        for frame_info in inspect.stack():
            frame_locals = frame_info.frame.f_locals
            if "filepath" in frame_locals:
                img_path = frame_locals["filepath"]
                break

    if img_path is None:
        print("Could not determine input file path. Returning unchanged.")

    temp_dir = Path(os.path.dirname(img_path))

    # Create the tracking script
    script_path = temp_dir / "run_tracking.py"
    # Save the mask data
    # For label images, use the original path as mask_path
    basename = os.path.basename(img_path)
    
    # Check if this matches the configured label pattern
    if label_pattern in basename:
        mask_path = img_path
        # Find corresponding raw image by removing the label pattern
        raw_base, raw_candidates, raw_path = _find_matching_raw_path(
            img_path, label_pattern
        )
        if raw_path is None:
            print(f"Warning: Could not find raw image for {img_path}")
            print(
                f"  Tried removing '{label_pattern}' to get base '{raw_base}' and checking: {raw_candidates}"
            )
            raw_path = img_path  # Fallback to using label as input
        else:
            print("Processing label file: using matched raw-label pair for tracking")
    else:
        # For raw images, find the corresponding label image
        raw_path = img_path
        base_name = _strip_known_image_suffix(os.path.basename(img_path))
        mask_path = os.path.join(
            os.path.dirname(img_path), base_name + label_pattern
        )
        if not os.path.exists(mask_path):
            print(f"No label file found for {img_path}")
            return image

    mask_filename = os.path.basename(mask_path)
    if mask_filename.endswith(label_pattern):
        output_filename = (
            mask_filename[: -len(label_pattern)] + f"{_output_suffix}.tif"
        )
    else:
        output_filename = (
            os.path.splitext(mask_filename)[0] + f"{_output_suffix}.tif"
        )

    output_dir = Path(_output_folder) if _output_folder else temp_dir
    output_path = output_dir / output_filename

    script_content = create_trackastra_script(
        str(raw_path),
        str(mask_path),
        model,
        mode,
        str(output_path),
        channel,
        dimension_order,
    )

    with open(script_path, "w") as f:
        f.write(script_content)

    # Run TrackAstra in the dedicated environment
    conda_cmd = TrackAstraEnvManager.get_conda_cmd()
    cmd = [
        conda_cmd,
        "run",
        "-n",
        TrackAstraEnvManager.ENV_NAME,
        "python",
        str(script_path),
    ]
    print(f"Running TrackAstra with model='{model}', mode='{mode}'...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("TrackAstra error:")
        print(result.stdout)
        print(result.stderr)
        print("Returning original image unchanged.")
        if script_path.exists():
            os.remove(script_path)
        return image

    print(result.stdout)

    # Return the produced path so the processing worker does not save a
    # second suffixed copy based on the current input filename.
    if output_path.exists():
        print(f"Tracking completed. Output saved at: {output_path}")
        if script_path.exists():
            os.remove(script_path)
        return str(output_path)
    else:
        print("TrackAstra did not produce output. Returning unchanged.")
        if script_path.exists():
            os.remove(script_path)
        return image

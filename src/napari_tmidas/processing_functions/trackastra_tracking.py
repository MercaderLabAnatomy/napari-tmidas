#!/usr/bin/env python3
"""
TrackAstra Cell Tracking Module for napari-tmidas

This module integrates TrackAstra deep learning-based cell tracking into the
napari-tmidas batch processing framework. It uses a dedicated conda environment
to manage TrackAstra dependencies separately from the main environment.
"""

import os
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from skimage.io import imread

# Add the registry import
from napari_tmidas._registry import BatchProcessingRegistry


_SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".zarr")


# --- Multi-GPU distribution -------------------------------------------------
# When several movies are tracked in one batch run, spread the per-file
# Trackastra subprocesses across the available GPUs. Each job acquires one GPU
# id from a shared pool (pinned via CUDA_VISIBLE_DEVICES); the pool size bounds
# concurrency to one job per GPU, which also prevents two jobs colliding on the
# same card and running it out of memory.
_GPU_POOL_LOCK = threading.Lock()
_GPU_POOL = None  # queue.Queue of GPU id strings (built lazily)
_GPU_IDS = None  # list of detected GPU id strings ([] = don't pin)
_GPU_POOL_WORKERS_PER_GPU = None  # repeat count baked into the current pool
_GPU_POOL_KEY = None  # (workers_per_gpu, gpus_override) baked into the current pool


def _detect_gpu_ids(gpus_override: str = None):
    """Detect GPU ids to distribute across. Honours overrides.

    - ``gpus_override`` (the function's own ``gpus`` parameter), e.g. "0" or
      "0,1", or "none"/"cpu"/"" to disable pinning
    - else ``TRACKASTRA_GPUS`` env var, same syntax
    - else ``CUDA_VISIBLE_DEVICES`` if already set
    - else counts physical GPUs via ``nvidia-smi -L``
    Returns a list of id strings; empty means do not pin a device.
    """
    if gpus_override is not None and gpus_override.strip() != "":
        if gpus_override.strip().lower() in ("none", "cpu"):
            return []
        return [g.strip() for g in gpus_override.split(",") if g.strip() != ""]

    override = os.environ.get("TRACKASTRA_GPUS")
    if override is not None:
        if override.strip().lower() in ("", "none", "cpu"):
            return []
        return [g.strip() for g in override.split(",") if g.strip() != ""]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        return [g.strip() for g in cvd.split(",") if g.strip() != ""]

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            n = len(
                [
                    line
                    for line in out.stdout.splitlines()
                    if line.strip().startswith("GPU ")
                ]
            )
            if n > 0:
                return [str(i) for i in range(n)]
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def _get_gpu_pool(workers_per_gpu: int = 1, gpus_override: str = None):
    """Return (pool, gpu_ids), (re)building the shared pool as needed.

    Each GPU id is enqueued ``workers_per_gpu`` times, so up to that many
    concurrent jobs can share a single card (useful when a GPU has enough
    VRAM to run several Trackastra inferences at once). The pool is rebuilt
    if a batch run requests a different ``workers_per_gpu``/``gpus_override``
    than the cached one; this only happens between runs, since every file in
    one batch shares the same parameter values.
    """
    global _GPU_POOL, _GPU_IDS, _GPU_POOL_WORKERS_PER_GPU, _GPU_POOL_KEY
    workers_per_gpu = max(1, int(workers_per_gpu))
    cache_key = (workers_per_gpu, gpus_override)
    with _GPU_POOL_LOCK:
        if _GPU_POOL is None or _GPU_POOL_KEY != cache_key:
            _GPU_IDS = _detect_gpu_ids(gpus_override)
            _GPU_POOL = queue.Queue()
            for gpu_id in _GPU_IDS:
                for _ in range(workers_per_gpu):
                    _GPU_POOL.put(gpu_id)
            _GPU_POOL_WORKERS_PER_GPU = workers_per_gpu
            _GPU_POOL_KEY = cache_key
        return _GPU_POOL, _GPU_IDS


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


def _resolve_gurobi_license(gurobi_license: str = ""):
    """Resolve which Gurobi license file the ilp solver should use.

    Trackastra's ``ilp`` mode solves via motile/ilpy → Gurobi. The pip
    ``gurobipy`` package ships a bundled *size-limited* license
    (``TYPE=PIP``) inside the conda env (``.../envs/trackastra/lib/gurobi.lic``)
    and prioritises it, so it shadows a full/academic license placed in the
    home directory. Setting ``GRB_LICENSE_FILE`` explicitly overrides that,
    since it takes precedence over every default search location.

    Resolution order (first hit wins):
      1. explicit ``gurobi_license`` path argument (from the widget),
      2. an already-exported ``GRB_LICENSE_FILE``,
      3. ``~/gurobi.lic`` (the standard academic/named-user location).

    Returns the resolved license path as a string, or ``None`` to leave the
    environment untouched (bundled size-limited license is then used).
    """
    candidate = (gurobi_license or "").strip()
    if candidate:
        lic = os.path.expanduser(candidate)
        if os.path.isfile(lic):
            return lic
        print(
            f"Warning: gurobi_license path '{candidate}' not found; "
            "falling back to auto-detection."
        )

    existing = os.environ.get("GRB_LICENSE_FILE", "").strip()
    if existing and os.path.isfile(os.path.expanduser(existing)):
        return os.path.expanduser(existing)

    home_lic = os.path.join(os.path.expanduser("~"), "gurobi.lic")
    if os.path.isfile(home_lic):
        return home_lic

    return None


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
    # Exclusive upper bounds: versions >= these break compatibility.
    MAX_VERSIONS = {
        "motile": "1.0.0",  # 1.0.0 renamed NodeSelection -> NodeSelectedCost
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
            max_ver = cls.MAX_VERSIONS.get(pkg)
            if max_ver and cls._version_at_least(found, max_ver):
                reasons.append(
                    f"{pkg} version {found} >= max allowed {max_ver} (API incompatibility)"
                )

        return (len(reasons) > 0), reasons

    @classmethod
    def repair_env(cls):
        """Repair/upgrade TrackAstra environment using upstream ILP recipe."""
        print("Repairing TrackAstra environment to required package versions...")
        conda_cmd = cls.get_conda_cmd()
        try:
            # Clear corrupted/stale cached packages before installing.
            subprocess.run(
                [conda_cmd, "clean", "--packages", "--index-cache", "-y"],
                check=False,
            )
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
                "motile==0.4.0",
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
            "-y",
        ]
        # `--no-default-packages` is a conda-only flag; mamba rejects it.
        if os.path.basename(conda_cmd) == "conda":
            env_create_cmd.insert(-1, "--no-default-packages")

        try:
            # Clear corrupted/stale cached packages before installing.
            subprocess.run(
                [conda_cmd, "clean", "--packages", "--index-cache", "-y"],
                check=False,
            )
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
                "motile==0.4.0",
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
    batch_size="Auto",
):
    """Create a Python script to run TrackAstra in the dedicated environment.

    The generated script loads images/masks as lazy dask arrays (one task per
    TIFF page, or ``da.from_zarr`` for zarr) so Trackastra streams feature
    extraction frame-by-frame, and writes the tracked output one frame at a
    time. This keeps peak RAM at a few frames instead of the whole TZYX volume.

    GPU prediction memory is the other bottleneck: on CUDA OOM the script
    shrinks ``batch_size`` and finally falls back to CPU. ``batch_size``
    accepts "Auto" (model default, 4 on CUDA) or a positive integer.
    """

    # Resolve batch_size into a literal injected into the script (None = Auto).
    batch_size_literal = "None"
    if batch_size is not None and str(batch_size).strip().lower() not in ("", "auto"):
        try:
            parsed = int(batch_size)
            if parsed > 0:
                batch_size_literal = str(parsed)
        except (TypeError, ValueError):
            pass

    # img/mask loads use a single embedded lazy loader (see _LAZY_LOADERS).
    img_load_code = f"img = _lazy_load({str(img_path)!r})"
    mask_load_code = f"mask = _lazy_load({str(mask_path)!r})"

    script_content = f"""
import os
# Reduce CUDA fragmentation; must be set before torch initializes CUDA.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import gc
import numpy as np
import dask.array as da
from dask import delayed
import tifffile
from tifffile import imwrite
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc


# ----- Lazy loaders: stream frames instead of loading the whole volume -----
def _lazy_load_tiff(path):
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        dtype = series.dtype
        n_pages = len(series.pages)

    # 2D images or single-page files: tiny, just read directly.
    if n_pages <= 1 or len(shape) <= 2:
        return da.from_array(tifffile.imread(path))

    page_shape = shape[-2:]

    def _read_page(fpath, idx):
        with tifffile.TiffFile(fpath) as _tf:
            return _tf.series[0].pages[idx].asarray()

    dask_pages = [
        da.from_delayed(delayed(_read_page)(path, i), shape=page_shape, dtype=dtype)
        for i in range(n_pages)
    ]
    return da.stack(dask_pages).reshape(shape)


def _zarr_resolve(path):
    import zarr

    def _node_to_array(zobj):
        if hasattr(zobj, 'shape'):
            return zobj
        if hasattr(zobj, 'array_keys'):
            keys = list(zobj.array_keys())
            if '0' in keys:
                return zobj['0']
            if keys:
                return zobj[keys[0]]
        if hasattr(zobj, 'group_keys'):
            for key in sorted(zobj.group_keys(), key=lambda k: (k != '0', k)):
                try:
                    arr = _node_to_array(zobj[key])
                    if arr is not None:
                        return arr
                except Exception:
                    continue
        return None

    candidates = [path, os.path.join(path, '0')]
    try:
        for child in sorted(os.listdir(path)):
            candidates.append(os.path.join(path, child, '0'))
    except Exception:
        pass

    openers = (
        lambda p: zarr.open(p, mode='r'),
        lambda p: zarr.open_group(store=p, mode='r'),
        lambda p: zarr.open_array(store=p, mode='r'),
    )
    attempts = []
    for loc in candidates:
        for opener in openers:
            try:
                arr = _node_to_array(opener(loc))
                if arr is not None:
                    return arr
            except Exception as exc:
                attempts.append(f'{{loc}}: {{exc}}')
    raise RuntimeError('Unable to open zarr array. Attempts:\\n' + '\\n'.join(attempts))


def _lazy_load(path):
    if str(path).lower().endswith('.zarr'):
        return da.from_zarr(_zarr_resolve(path))
    return _lazy_load_tiff(path)


def _take_axis(arr, idx, axis):
    # np.take does not dispatch to dask; build an explicit slice instead.
    sl = [slice(None)] * arr.ndim
    sl[axis] = idx
    return arr[tuple(sl)]


# Load images lazily (dask arrays)
print('Lazily loading images...')
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
    img = _take_axis(img, ch_idx, channel_axis)
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

# User-requested prediction batch size ('Auto' -> model default, 4 on CUDA).
requested_batch = {batch_size_literal}


def _is_cuda_oom(exc):
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', ())
    return isinstance(exc, oom_cls) or 'out of memory' in str(exc).lower()


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _track_once(track_model, m, batch_size):
    kwargs = {{}} if batch_size is None else {{'batch_size': batch_size}}
    return track_model.track(img, mask, mode=m, **kwargs)


def _track_mode(m):
    # Prediction (transformer attention) is the GPU-memory bottleneck and is
    # identical across tracking modes. On CUDA OOM, shrink the batch size, then
    # fall back to CPU (slow but bounded by host RAM) rather than failing.
    ladder = [requested_batch]
    if requested_batch != 1:
        ladder.append(1)
    for batch_size in ladder:
        try:
            return _track_once(model, m, batch_size)
        except Exception as exc:
            if _is_cuda_oom(exc):
                print(
                    f'CUDA OOM (mode={{m}}, batch_size={{batch_size}}); '
                    'freeing GPU and retrying with a smaller batch...'
                )
                _free_gpu()
                continue
            raise
    print('CUDA still OOM; loading model on CPU and predicting there (slow)...')
    _free_gpu()
    cpu_model = Trackastra.from_pretrained('{model}', device='cpu')
    return cpu_model.track(img, mask, mode=m, batch_size=1)


try:
    track_result = _track_mode('{mode}')
except Exception as exc:
    # ilp/greedy only differ in the post-prediction solve, so a non-OOM failure
    # in '{mode}' may be solver-specific — retry once with greedy.
    if '{mode}' != 'greedy' and not _is_cuda_oom(exc):
        print(f'Warning: TrackAstra {mode} mode failed ({{exc}}). Retrying with greedy mode...')
        _free_gpu()
        track_result = _track_mode('greedy')
    else:
        raise

# Trackastra API compatibility:
# - newer versions may return (track_graph, masks_tracked)
# - older versions return only track_graph
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

out_shape = tuple(int(s) for s in mask.shape)
n_timepoints = out_shape[0]
out_dtype = np.uint32
out_axes = "TZYX" if len(out_shape) == 4 else "TYX"
bytes_total = int(np.prod(out_shape, dtype=np.int64)) * np.dtype(out_dtype).itemsize
use_bigtiff = bytes_total > 2 * 1024**3


def _best_tiff_compression():
    # zstd is faster and compresses label data well; needs imagecodecs.
    try:
        import imagecodecs
        if imagecodecs.ZSTD.available:
            return 'zstd'
    except Exception:
        pass
    return 'zlib'


tiff_compression = _best_tiff_compression()
print(f'Using TIFF compression: {{tiff_compression}}')


# Build a per-frame relabel map identical to graph_to_ctc, then stream output.
# graph_to_ctc allocates the whole TZYX output in RAM (np.zeros_like stack);
# replicating only its label assignment lets us write frame-by-frame instead.
frame_relabel = None
try:
    try:
        from trackastra.tracking.utils import ctc_tracklets
    except Exception:
        from trackastra.tracking import ctc_tracklets

    tracklets = ctc_tracklets(track_graph, frame_attribute='time')
    frame_relabel = {{}}
    for i, _tracklet in enumerate(sorted(tracklets)):
        label = i + 1
        for _n in _tracklet.nodes:
            node = track_graph.nodes[_n]
            t = int(node['time'])
            lab = int(node['label'])
            frame_relabel.setdefault(t, {{}})[lab] = label
    print(f'Built streaming relabel map for {{len(frame_relabel)}} frames '
          f'({{sum(len(v) for v in frame_relabel.values())}} objects).')
except Exception as exc:
    print(f'Streaming relabel unavailable ({{exc}}); falling back to in-memory graph_to_ctc.')
    frame_relabel = None


if frame_relabel is not None:
    def _relabel_frame(t):
        # Relabel one timepoint slab (Z,Y,X) or (Y,X) via a small lookup table.
        frame = np.asarray(mask[t])
        relmap = frame_relabel.get(t)
        if not relmap:
            return np.zeros(frame.shape, dtype=out_dtype)
        maxl = int(frame.max())
        lut = np.zeros(maxl + 1, dtype=out_dtype)
        for orig, new in relmap.items():
            if 0 <= orig <= maxl:
                lut[orig] = new
        return lut[frame].astype(out_dtype, copy=False)

    def _iter_relabeled_pages():
        # tifffile pulls one (Y,X) page per iteration for a TZYX `shape`, so we
        # flatten each timepoint's leading axes (e.g. Z) into individual pages.
        for t in range(n_timepoints):
            relabeled = _relabel_frame(t)
            if relabeled.ndim <= 2:
                yield relabeled
            else:
                for page in relabeled.reshape((-1,) + relabeled.shape[-2:]):
                    yield page

    imwrite(
        '{output_path}',
        data=_iter_relabeled_pages(),
        shape=out_shape,
        dtype=out_dtype,
        ome=True,
        metadata={{'axes': out_axes}},
        compression=tiff_compression,
        photometric='minisblack',
        bigtiff=use_bigtiff,
    )
    print(f'Saved tracked masks (streamed) to: {output_path}')
else:
    # Fallback: original in-memory path (materializes the full mask + output).
    mask_np = np.asarray(mask)
    _, masks_tracked = graph_to_ctc(track_graph, mask_np, outdir=None)
    in_ids = np.unique(mask_np)
    out_ids = np.unique(masks_tracked)
    print(f'Relabel check: unique_ids_in={{len(in_ids)}}, unique_ids_out={{len(out_ids)}}')
    imwrite(
        '{output_path}',
        masks_tracked.astype(out_dtype),
        ome=True,
        metadata={{'axes': out_axes}},
        compression=tiff_compression,
        bigtiff=use_bigtiff,
    )
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
        "batch_size": {
            "type": str,
            "default": "Auto",
            "description": "GPU prediction batch size. 'Auto' uses the model default (4 on CUDA); lower it (e.g. 1) if you hit CUDA out-of-memory. On OOM the run auto-shrinks the batch and finally falls back to CPU.",
        },
        "gurobi_license": {
            "type": str,
            "default": "",
            "description": "Path to a Gurobi license file (.lic) for the 'ilp' mode solver. Leave empty to auto-detect ~/gurobi.lic; only needed to override the bundled size-limited pip license. Ignored by 'greedy' modes.",
        },
        "gpus": {
            "type": str,
            "default": "",
            "description": "Comma-separated GPU ids to use (e.g. '0' or '0,1'). Leave empty to auto-detect and use all available GPUs. Set to 'cpu' or 'none' to disable GPU pinning. Each pinned GPU runs its own subprocess that loads the full raw+segmentation arrays into RAM, so restrict this on large datasets to limit memory use, not just VRAM.",
        },
        "workers_per_gpu": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 8,
            "description": "Number of concurrent Trackastra jobs to run per GPU. Increase if a single card has enough VRAM to run more than one tracking job at once (multi-GPU workstations benefit most).",
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
    batch_size: str = "Auto",
    gurobi_license: str = "",
    gpus: str = "",
    workers_per_gpu: int = 1,
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
    gurobi_license : str
        Optional path to a Gurobi license file (.lic) used by the 'ilp' mode
        solver. Leave empty to auto-detect ~/gurobi.lic (or an already-exported
        GRB_LICENSE_FILE); only needed to override the bundled size-limited pip
        license that ships in the conda env. Ignored by the greedy modes.
    gpus : str
        Comma-separated GPU ids to pin to (e.g. '0' or '0,1'). Empty
        auto-detects and uses all available GPUs; 'cpu'/'none' disables
        pinning. Restricting this bounds how many concurrent Trackastra
        subprocesses run, which bounds RAM use as well as VRAM, since each
        pinned GPU loads its own full copy of the raw + segmentation arrays.
    workers_per_gpu : int
        Number of concurrent Trackastra jobs to run per GPU (default: 1).
    label_pattern : str
        To identify label images

    Returns:
    --------
    np.ndarray
        Tracked label image with consistent IDs across time
    """
    # When the worker honours `skip_load` (see below) the array is never loaded
    # into RAM and `image` is None; the subprocess validates dimensions from the
    # file itself. Only run the in-memory checks when an array is present.
    if image is not None:
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
    else:
        _src_tag = os.path.basename(_source_filepath) if _source_filepath else "?"
        print(
            f"[{_src_tag}] Input array not loaded (skip_load); "
            "reading dimensions from file."
        )

    # Normalize the tracking mode up front so an invalid value does not waste a
    # full (multi-minute) feature-extraction run before failing in the subprocess.
    valid_modes = ("greedy", "ilp", "greedy_nodiv")
    if mode not in valid_modes:
        # Tolerate the common 'lip' typo for 'ilp'; otherwise fall back to greedy.
        corrected = "ilp" if mode == "lip" else "greedy"
        print(
            f"Warning: invalid tracking mode '{mode}'. "
            f"Using '{corrected}' instead (valid: {', '.join(valid_modes)})."
        )
        mode = corrected

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

    # Create the tracking script. Use a per-file unique name so concurrent jobs
    # (multi-GPU batches) sharing an input directory never clobber each other.
    script_path = temp_dir / f"run_tracking_{Path(img_path).stem}_{os.getpid()}.py"
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
            print(
                f"[{os.path.basename(img_path)}] Processing label file: "
                "using matched raw-label pair for tracking"
            )
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
        batch_size,
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

    # Acquire a GPU from the shared pool so concurrent files spread across cards
    # (blocks until one is free; no-op when no GPUs are detected/pinning is off).
    # Each GPU appears `workers_per_gpu` times in the pool, so that many jobs
    # can share it.
    pool, gpu_ids = _get_gpu_pool(workers_per_gpu, gpus)
    gpu_id = pool.get() if gpu_ids else None
    run_env = os.environ.copy()

    # Point Gurobi at the user's (academic/named-user) license so the ilp
    # solver isn't capped by the bundled size-limited pip license. Only
    # relevant for mode='ilp'; harmless for greedy modes.
    license_path = _resolve_gurobi_license(gurobi_license)
    if license_path:
        run_env["GRB_LICENSE_FILE"] = license_path
        if mode == "ilp":
            print(f"Using Gurobi license: {license_path}")
    elif mode == "ilp":
        print(
            "No Gurobi license file found (checked gurobi_license arg, "
            "GRB_LICENSE_FILE, ~/gurobi.lic); the ilp solver will use the "
            "bundled size-limited pip license and may fail on large problems. "
            "Provide a license via the 'gurobi_license' parameter."
        )

    label_tag = os.path.basename(mask_path)
    if gpu_id is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(
            f"[{label_tag}] Running TrackAstra with model='{model}', "
            f"mode='{mode}' on GPU {gpu_id}..."
        )
    else:
        print(
            f"[{label_tag}] Running TrackAstra with model='{model}', "
            f"mode='{mode}'..."
        )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env
        )
    finally:
        if gpu_id is not None:
            pool.put(gpu_id)
            print(f"[{label_tag}] Released GPU {gpu_id}")

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
        print(f"[{label_tag}] Tracking completed. Output saved at: {output_path}")
        if script_path.exists():
            os.remove(script_path)
        return str(output_path)
    else:
        print("TrackAstra did not produce output. Returning unchanged.")
        if script_path.exists():
            os.remove(script_path)
        return image


# TrackAstra reads its inputs directly from `_source_filepath` inside a
# dedicated subprocess and ignores the in-memory array passed by the worker.
# `skip_load` tells the napari widget's ProcessingWorker
# (_file_selector.ProcessingWorker) to pass image=None and never allocate the
# full TZYX volume. `_loads_from_path` is the equivalent hint for the secondary
# _processing_worker.ProcessingWorker (lazy dask load there). Either way the
# parent process avoids materializing the ~tens-of-GB array just to discard it.
trackastra_tracking.skip_load = True
trackastra_tracking._loads_from_path = True

# Each file is tracked in its own subprocess pinned to one GPU (see the GPU
# pool above), so multiple files can run concurrently across GPUs (and, via
# the `workers_per_gpu` parameter, multiple concurrent files per GPU). This
# marker lets the batch widget raise the worker thread count to
# n_gpus * workers_per_gpu instead of forcing single-threaded execution.
trackastra_tracking.supports_gpu_distribution = True

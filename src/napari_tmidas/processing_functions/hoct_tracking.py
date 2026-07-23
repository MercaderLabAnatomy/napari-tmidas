#!/usr/bin/env python3
"""
HOCT Cell Tracking Module for napari-tmidas

This module integrates HOCT (Higher-Order Cell Tracking Transformer,
https://github.com/royerlab/hoct) deep learning-based cell tracking into the
napari-tmidas batch processing framework. It uses a dedicated conda
environment to manage HOCT dependencies separately from the main environment,
and drives the ``hoct`` CLI directly (rather than a generated Python script)
since HOCT already ships a CTC (Cell Tracking Challenge) exporter.
"""

import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import tifffile
from tifffile import imwrite

from napari_tmidas._registry import BatchProcessingRegistry

_SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".zarr")


# --- Multi-GPU distribution -------------------------------------------------
# When several movies are tracked in one batch run, spread the per-file
# HOCT subprocesses across the available GPUs (one worker per GPU, not one
# worker per CPU core) instead of the widget's default CPU thread count.
# Each job acquires one GPU id from a shared pool (pinned via
# CUDA_VISIBLE_DEVICES); the pool size bounds concurrency to one job per GPU,
# which also prevents two jobs colliding on the same card and running it out
# of memory.
_GPU_POOL_LOCK = threading.Lock()
_GPU_POOL = None  # queue.Queue of GPU id strings (built lazily)
_GPU_IDS = None  # list of detected GPU id strings ([] = don't pin)
_GPU_POOL_WORKERS_PER_GPU = None  # repeat count baked into the current pool


def _detect_gpu_ids():
    """Detect GPU ids to distribute across. Honours overrides.

    - ``HOCT_GPUS`` (e.g. "0,1", or "none"/"" to disable pinning)
    - else ``CUDA_VISIBLE_DEVICES`` if already set
    - else counts physical GPUs via ``nvidia-smi -L``
    Returns a list of id strings; empty means do not pin a device.
    """
    override = os.environ.get("HOCT_GPUS")
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


def _get_gpu_pool(workers_per_gpu: int = 1):
    """Return (pool, gpu_ids), (re)building the shared pool as needed.

    Each GPU id is enqueued ``workers_per_gpu`` times, so up to that many
    concurrent jobs can share a single card (useful when a GPU has enough
    VRAM to run several HOCT inferences at once). The pool is rebuilt if a
    batch run requests a different ``workers_per_gpu`` than the cached one;
    this only happens between runs, since every file in one batch shares the
    same parameter value.
    """
    global _GPU_POOL, _GPU_IDS, _GPU_POOL_WORKERS_PER_GPU
    workers_per_gpu = max(1, int(workers_per_gpu))
    with _GPU_POOL_LOCK:
        if _GPU_POOL is None or _GPU_POOL_WORKERS_PER_GPU != workers_per_gpu:
            _GPU_IDS = _detect_gpu_ids()
            _GPU_POOL = queue.Queue()
            for gpu_id in _GPU_IDS:
                for _ in range(workers_per_gpu):
                    _GPU_POOL.put(gpu_id)
            _GPU_POOL_WORKERS_PER_GPU = workers_per_gpu
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
    """Resolve which Gurobi license file HOCT's ILP solver should use.

    HOCT's tracking step solves an ILP via tracksdata/gurobipy. The pip
    ``gurobipy`` package ships a bundled *size-limited* license inside the
    conda env and prioritises it, so it shadows a full/academic license
    placed in the home directory. Setting ``GRB_LICENSE_FILE`` explicitly
    overrides that, since it takes precedence over every default search
    location.

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


class HoctEnvManager:
    """Manages the dedicated conda environment for HOCT."""

    ENV_NAME = "hoct"
    REQUIRED_PYTHON = "3.11"

    @staticmethod
    def get_conda_cmd():
        """Get the conda/mamba command available on the system."""
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
            result = subprocess.run(
                [conda_cmd, "run", "-n", cls.ENV_NAME, "python", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            return False

    @classmethod
    def _hoct_cli_ready(cls):
        """Check that the ``hoct`` CLI entry point is importable/runnable."""
        conda_cmd = cls.get_conda_cmd()
        try:
            result = subprocess.run(
                [conda_cmd, "run", "-n", cls.ENV_NAME, "hoct", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            return False

    @classmethod
    def create_env(cls):
        """Create the HOCT conda environment if it doesn't exist."""
        if cls.check_env_exists():
            print("HOCT environment already exists.")
        else:
            print("Creating HOCT conda environment...")
            conda_cmd = cls.get_conda_cmd()

            env_create_cmd = [
                conda_cmd,
                "create",
                "-n",
                cls.ENV_NAME,
                f"python={cls.REQUIRED_PYTHON}",
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
            except subprocess.CalledProcessError as e:
                print(f"Error creating HOCT environment: {e}")
                return False

        conda_cmd = cls.get_conda_cmd()
        try:
            # HOCT's own dependency pins (e.g. gurobipy<13.0.0) are resolved
            # automatically by pip; no extra conda-forge packages are needed.
            pip_cmd = [
                conda_cmd,
                "run",
                "-n",
                cls.ENV_NAME,
                "pip",
                "install",
                "hoct[bioio]",
            ]
            subprocess.run(pip_cmd, check=True)
            print("HOCT environment is ready.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing HOCT: {e}")
            return False

    @classmethod
    def ensure_env_ready(cls):
        """Ensure the environment exists and the ``hoct`` CLI is usable."""
        if not cls.check_env_exists():
            print("HOCT environment not found. Creating it now...")
            if not cls.create_env():
                return False

        if not cls._hoct_cli_ready():
            print("HOCT CLI not available in environment; (re)installing...")
            conda_cmd = cls.get_conda_cmd()
            try:
                subprocess.run(
                    [
                        conda_cmd,
                        "run",
                        "-n",
                        cls.ENV_NAME,
                        "pip",
                        "install",
                        "--upgrade",
                        "hoct[bioio]",
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error installing HOCT: {e}")
                return False

            if not cls._hoct_cli_ready():
                print("HOCT environment is still not healthy after repair.")
                return False

        print("HOCT environment is ready.")
        return True


def _assemble_ctc_output(ctc_dir: Path, output_path: Path) -> tuple:
    """Stitch a HOCT CTC output directory (mask###.tif frames) into one TIFF.

    HOCT's CTC exporter (``tracksdata.io.to_ctc``) writes one
    ``mask{t:0{n_digits}d}.tif`` file per timepoint (each file itself being a
    single (Y, X) page or a (Z, Y, X) stack) plus a ``res_track.txt`` lineage
    file. Streaming the frames straight into a single multi-page TIFF keeps
    peak RAM at a few frames instead of the whole (T, [Z,] Y, X) volume.
    """
    mask_files = sorted(ctc_dir.glob("mask*.tif"))
    if not mask_files:
        raise RuntimeError(f"No CTC mask files found in {ctc_dir}")

    first_frame = tifffile.imread(str(mask_files[0]))
    frame_shape = first_frame.shape
    n_timepoints = len(mask_files)
    out_shape = (n_timepoints,) + frame_shape
    out_dtype = first_frame.dtype
    bytes_total = int(np.prod(out_shape, dtype=np.int64)) * np.dtype(out_dtype).itemsize
    use_bigtiff = bytes_total > 2 * 1024**3

    def _iter_pages():
        for mask_file in mask_files:
            frame = tifffile.imread(str(mask_file))
            if frame.ndim <= 2:
                yield frame
            else:
                yield from frame

    imwrite(
        str(output_path),
        data=_iter_pages(),
        shape=out_shape,
        dtype=out_dtype,
        compression="zlib",
        photometric="minisblack",
        bigtiff=use_bigtiff,
    )
    return out_shape


@BatchProcessingRegistry.register(
    name="Track Cells with HOCT",
    suffix="_hoct_tracked",
    description=(
        "Track cells across time using HOCT (Higher-Order Cell Tracking "
        "Transformer, github.com/royerlab/hoct), a transformer-based tracker. "
        "Expects TYX or TZYX label images with a matching raw image of the "
        "same shape. Supports TIFF and zarr inputs."
    ),
    parameters={
        "model": {
            "type": str,
            "default": "",
            "description": "Checkpoint path or registered HOCT model name. Leave empty to auto-download the default pretrained model.",
        },
        "device": {
            "type": str,
            "default": "cuda",
            "options": ["cuda", "cpu", "mps"],
            "description": "Compute device for inference. HOCT falls back to CPU automatically if the requested device is unavailable.",
        },
        "window": {
            "type": int,
            "default": 5,
            "min": 1,
            "max": 50,
            "description": "Temporal window size for the frame dataset (HOCT --window).",
        },
        "max_distance": {
            "type": float,
            "default": 300.0,
            "min": 1.0,
            "max": 5000.0,
            "step": 1.0,
            "description": "Maximum spatial distance (pixels) for candidate tracking edges (HOCT --max-distance).",
        },
        "neighbors": {
            "type": int,
            "default": 5,
            "min": 1,
            "max": 50,
            "description": "Maximum number of candidate neighbors per node (HOCT --neighbors).",
        },
        "max_dt": {
            "type": int,
            "default": 3,
            "min": 1,
            "max": 20,
            "description": "Maximum temporal gap in frames for candidate edges, allowing bridging of missed detections (HOCT --max-dt).",
        },
        "tile": {
            "type": str,
            "default": "auto",
            "options": ["auto", "on", "off"],
            "description": "Tiled inference mode for large volumes. 'auto' enables tiling when the candidate graph is dense enough to risk GPU OOM.",
        },
        "scale": {
            "type": str,
            "default": "",
            "description": "Optional physical voxel size as space-separated 't y x' or 't z y x' (e.g. '1 0.5 0.2 0.2'). Leave empty to track in pixel units.",
        },
        "gurobi_license": {
            "type": str,
            "default": "",
            "description": "Path to a Gurobi license file (.lic) for HOCT's ILP solver. Leave empty to auto-detect ~/gurobi.lic; only needed to override the bundled size-limited pip license.",
        },
        "workers_per_gpu": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 8,
            "description": "Number of concurrent HOCT jobs to run per GPU. Increase if a single card has enough VRAM to run more than one tracking job at once (multi-GPU workstations benefit most). Only used when device='cuda'.",
        },
        "label_pattern": {
            "type": str,
            "default": "_labels.tif",
            "description": " ",
        },
    },
)
def hoct_tracking(
    image: np.ndarray,
    model: str = "",
    device: str = "cuda",
    window: int = 5,
    max_distance: float = 300.0,
    neighbors: int = 5,
    max_dt: int = 3,
    tile: str = "auto",
    scale: str = "",
    gurobi_license: str = "",
    workers_per_gpu: int = 1,
    label_pattern: str = "_labels.tif",
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = "_hoct_tracked",
) -> np.ndarray:
    """
    Track cells in time-lapse label images using HOCT.

    This function takes a time series of segmentation masks and a matching
    raw image (same shape) and performs automatic cell tracking using HOCT
    (https://github.com/royerlab/hoct), a transformer-based tracker from
    royerlab. Tracking is run via the ``hoct`` CLI in a dedicated conda
    environment, exporting directly to CTC (Cell Tracking Challenge) format,
    which is then stitched into a single relabeled TIFF.

    Expected input dimensions:
    - TYX: Time series of 2D label images
    - TZYX: Time series of 3D label images

    Input file formats:
    - TIFF (.tif, .tiff files)
    - Zarr (.zarr directories, including OME-Zarr)

    Parameters:
    -----------
    image : np.ndarray
        Input label image array with time as first dimension
    model : str
        Checkpoint path or registered HOCT model name. Empty uses the
        default pretrained model (auto-downloaded on first use).
    device : str
        Compute device: 'cuda', 'cpu', or 'mps' (default: "cuda")
    window : int
        Temporal window size for the frame dataset
    max_distance : float
        Maximum spatial distance (pixels) for candidate tracking edges
    neighbors : int
        Maximum number of candidate neighbors per node
    max_dt : int
        Maximum temporal gap (frames) for candidate edges
    tile : str
        Tiled inference mode: 'auto', 'on', or 'off'
    scale : str
        Optional physical voxel size, space-separated 't y x' or 't z y x'
    gurobi_license : str
        Optional path to a Gurobi license file (.lic) used by HOCT's ILP
        solver. Leave empty to auto-detect ~/gurobi.lic (or an
        already-exported GRB_LICENSE_FILE).
    workers_per_gpu : int
        Number of concurrent HOCT jobs to run per GPU (default: 1). Only
        used when device='cuda'.
    label_pattern : str
        To identify label images

    Returns:
    --------
    np.ndarray
        Tracked label image with consistent IDs across time
    """
    # When the worker honours `skip_load` (see below) the array is never
    # loaded into RAM and `image` is None; the subprocess validates
    # dimensions from the file itself.
    if image is not None:
        print(f"Input shape: {image.shape}, dtype: {image.dtype}")

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

    if device not in ("cuda", "cpu", "mps"):
        print(f"Warning: invalid device '{device}'. Using 'cuda' instead.")
        device = "cuda"

    if tile not in ("auto", "on", "off"):
        print(f"Warning: invalid tile mode '{tile}'. Using 'auto' instead.")
        tile = "auto"

    # Ensure HOCT environment exists and the CLI is usable.
    if not HoctEnvManager.ensure_env_ready():
        print("Failed to prepare HOCT environment. Returning unchanged.")
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
        return image

    temp_dir = Path(os.path.dirname(img_path))
    basename = os.path.basename(img_path)

    # Check if this matches the configured label pattern
    if label_pattern in basename:
        mask_path = img_path
        raw_base, raw_candidates, raw_path = _find_matching_raw_path(
            img_path, label_pattern
        )
        if raw_path is None:
            print(f"Warning: Could not find raw image for {img_path}")
            print(
                f"  Tried removing '{label_pattern}' to get base '{raw_base}' and checking: {raw_candidates}"
            )
            print("HOCT requires a matching raw image. Returning unchanged.")
            return image
        print(
            f"[{os.path.basename(img_path)}] Processing label file: "
            "using matched raw-label pair for tracking"
        )
    else:
        # For raw images, find the corresponding label image
        raw_path = img_path
        base_name = _strip_known_image_suffix(basename)
        mask_path = os.path.join(
            os.path.dirname(img_path), base_name + label_pattern
        )
        if not os.path.exists(mask_path):
            print(f"No label file found for {img_path}")
            return image

    mask_filename = os.path.basename(mask_path)
    if mask_filename.endswith(label_pattern):
        output_stem = mask_filename[: -len(label_pattern)]
    else:
        output_stem = os.path.splitext(mask_filename)[0]

    output_dir = Path(_output_folder) if _output_folder else temp_dir
    output_path = output_dir / f"{output_stem}{_output_suffix}.tif"

    # HOCT's CTC exporter writes into a fresh directory (it refuses to write
    # into a non-empty one without --overwrite). Use a per-file unique name
    # so concurrent jobs (multi-GPU batches) never collide.
    ctc_dir = temp_dir / f"hoct_ctc_{output_stem}_{os.getpid()}"
    if ctc_dir.exists():
        shutil.rmtree(ctc_dir)

    conda_cmd = HoctEnvManager.get_conda_cmd()
    cmd = [
        conda_cmd,
        "run",
        "-n",
        HoctEnvManager.ENV_NAME,
        "hoct",
        "track",
        str(raw_path),
        str(mask_path),
        "-o",
        str(ctc_dir),
        "-f",
        "ctc",
        "-d",
        device,
        "-w",
        str(window),
        "--max-distance",
        str(max_distance),
        "--neighbors",
        str(neighbors),
        "--max-dt",
        str(max_dt),
        "--tile",
        tile,
        "--overwrite",
    ]
    if model.strip():
        cmd += ["-m", model.strip()]

    scale_values = [v for v in scale.replace(",", " ").split() if v.strip()]
    for value in scale_values:
        cmd += ["--scale", value]

    # Resolve the Gurobi license file for HOCT's ILP solver. We point
    # GRB_LICENSE_FILE at it rather than running grbgetkey: an existing
    # .lic file is read directly by Gurobi and needs no activation step.
    run_env = os.environ.copy()
    license_path = _resolve_gurobi_license(gurobi_license)
    if license_path:
        run_env["GRB_LICENSE_FILE"] = license_path
        print(f"Using Gurobi license: {license_path}")
    else:
        print(
            "No Gurobi license file found (checked gurobi_license arg, "
            "GRB_LICENSE_FILE, ~/gurobi.lic); HOCT's ILP solver will use the "
            "bundled size-limited pip license and may fail on large problems. "
            "Provide a license via the 'gurobi_license' parameter."
        )

    # Acquire a GPU from the shared pool so concurrent files spread across
    # cards (blocks until one is free; no-op when no GPUs are detected/
    # pinning is off, or when running on CPU/MPS). Each GPU appears
    # `workers_per_gpu` times in the pool, so that many jobs can share it.
    gpu_id = None
    pool = None
    if device == "cuda":
        pool, gpu_ids = _get_gpu_pool(workers_per_gpu)
        gpu_id = pool.get() if gpu_ids else None

    label_tag = os.path.basename(mask_path)
    if gpu_id is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"[{label_tag}] Running HOCT tracking on GPU {gpu_id}...")
    else:
        print(f"[{label_tag}] Running HOCT tracking on device '{device}'...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    finally:
        if gpu_id is not None:
            pool.put(gpu_id)
            print(f"[{label_tag}] Released GPU {gpu_id}")

    if result.returncode != 0:
        print("HOCT error:")
        print(result.stdout)
        print(result.stderr)
        print("Returning original image unchanged.")
        shutil.rmtree(ctc_dir, ignore_errors=True)
        return image

    print(result.stdout)

    try:
        out_shape = _assemble_ctc_output(ctc_dir, output_path)
    except Exception as exc:
        print(f"Failed to assemble HOCT CTC output: {exc}")
        return image
    finally:
        shutil.rmtree(ctc_dir, ignore_errors=True)

    print(
        f"[{label_tag}] HOCT tracking completed (shape {out_shape}). "
        f"Output saved at: {output_path}"
    )
    return str(output_path)


# HOCT reads its inputs directly from `_source_filepath` inside a dedicated
# subprocess and ignores the in-memory array passed by the worker.
# `skip_load` tells the napari widget's ProcessingWorker
# (_file_selector.ProcessingWorker) to pass image=None and never allocate the
# full TZYX volume. `_loads_from_path` is the equivalent hint for the
# secondary _processing_worker.ProcessingWorker (lazy dask load there).
hoct_tracking.skip_load = True
hoct_tracking._loads_from_path = True

# Each file is tracked in its own subprocess pinned to one GPU (see the GPU
# pool above), so multiple files can run concurrently across GPUs (and, via
# the `workers_per_gpu` parameter, multiple concurrent files per GPU). This
# marker lets the batch widget raise the worker thread count to
# n_gpus * workers_per_gpu instead of using the UI's CPU thread-count
# slider, matching the pattern used by Trackastra and Cellpose.
hoct_tracking.supports_gpu_distribution = True

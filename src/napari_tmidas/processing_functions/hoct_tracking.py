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
import uuid
from pathlib import Path

import numpy as np
import tifffile
from tifffile import imwrite

from napari_tmidas._registry import BatchProcessingRegistry

_SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".zarr")

# --- Lazy input staging -----------------------------------------------------
# HOCT's loader (``hoct._io.load_array``) only returns *lazy* dask arrays for
# Zarr stores and folders of single-frame TIFFs; a single multi-page TIFF is
# read eagerly with ``tifffile.asarray()``, so the whole (T, [Z,] Y, X) volume
# is resident for the entire run. Downstream, tracksdata's RegionPropsNodes
# reads one timepoint at a time (``np.asarray(labels[t])``), so a Zarr-backed
# input keeps peak RAM at roughly one timepoint instead of the full movie.
# Therefore we stage large TIFF inputs into temporary Zarr stores (chunked one
# timepoint at a time, written by streaming, never fully materialised) before
# handing them to the CLI.
_STAGE_MIN_BYTES = 1 * 1024**3  # below this an eager TIFF load is harmless
_STAGE_CHUNK_HINT = (16, 512, 512)  # per-axis chunk cap for (Z, Y, X)
# Napari only shows unsigned-integer label images as a Labels layer, and
# uint32 is the narrowest dtype the rest of the plugin uses for labels, so
# staged label volumes are cast down to it (int64 label TIFFs are common and
# cost twice the RAM for no extra label capacity in practice).
_STAGE_LABEL_DTYPE = np.uint32


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
_GPU_POOL_KEY = None  # (workers_per_gpu, gpus_override) baked into the current pool


def _detect_gpu_ids(gpus_override: str = None):
    """Detect GPU ids to distribute across. Honours overrides.

    - ``gpus_override`` (the function's own ``gpus`` parameter), e.g. "0" or
      "0,1", or "none"/"cpu"/"" to disable pinning
    - else ``HOCT_GPUS`` env var, same syntax
    - else ``CUDA_VISIBLE_DEVICES`` if already set
    - else counts physical GPUs via ``nvidia-smi -L``
    Returns a list of id strings; empty means do not pin a device.
    """
    if gpus_override is not None and gpus_override.strip() != "":
        if gpus_override.strip().lower() in ("none", "cpu"):
            return []
        return [g.strip() for g in gpus_override.split(",") if g.strip() != ""]

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


def _get_gpu_pool(workers_per_gpu: int = 1, gpus_override: str = None):
    """Return (pool, gpu_ids), (re)building the shared pool as needed.

    Each GPU id is enqueued ``workers_per_gpu`` times, so up to that many
    concurrent jobs can share a single card (useful when a GPU has enough
    VRAM to run several HOCT inferences at once). The pool is rebuilt if a
    batch run requests a different ``workers_per_gpu``/``gpus_override``
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
    # Napari only auto-detects a Labels layer for certain (unsigned) integer
    # dtypes; HOCT's CTC exporter writes int64 masks, so cast to uint32 to
    # match the convention used by write_labels_with_source_metadata and the
    # Trackastra tracker.
    out_dtype = np.uint32
    axes = "TZYX" if len(frame_shape) == 3 else "TYX"
    bytes_total = int(np.prod(out_shape, dtype=np.int64)) * np.dtype(out_dtype).itemsize
    use_bigtiff = bytes_total > 2 * 1024**3

    def _iter_pages():
        for mask_file in mask_files:
            frame = tifffile.imread(str(mask_file)).astype(out_dtype, copy=False)
            if frame.ndim <= 2:
                yield frame
            else:
                yield from frame

    imwrite(
        str(output_path),
        data=_iter_pages(),
        shape=out_shape,
        dtype=out_dtype,
        ome=True,
        metadata={"axes": axes},
        compression="zlib",
        photometric="minisblack",
        bigtiff=use_bigtiff,
    )
    return out_shape


def _open_zarr_array(zarr_path: str):
    """Open the full-resolution array of a Zarr store (OME group or plain)."""
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if hasattr(root, "shape"):  # plain array store
        return root

    # OME-NGFF: the multiscales metadata names the highest-resolution level
    # (v0.4 keeps it at the top level, v0.5 nests it under "ome").
    attrs = dict(root.attrs)
    multiscales = attrs.get("multiscales")
    if multiscales is None and isinstance(attrs.get("ome"), dict):
        multiscales = attrs["ome"].get("multiscales")
    if multiscales:
        try:
            return root[multiscales[0]["datasets"][0]["path"]]
        except (KeyError, IndexError, TypeError):
            pass

    for candidate in ("0", "s0", "data"):
        try:
            return root[candidate]
        except (KeyError, IndexError):
            continue
    raise ValueError(f"Could not find an array in Zarr store {zarr_path}")


def _peek_raw_shape(raw_path: str):
    """Read only shape metadata for a raw TIFF/Zarr image (no pixel data)."""
    if raw_path.lower().endswith(".zarr"):
        try:
            return _open_zarr_array(raw_path).shape
        except Exception:
            return None

    with tifffile.TiffFile(raw_path) as tif:
        return tif.series[0].shape


def _movie_shape(shape, axes: str = None):
    """Frame shape HOCT sees after dropping channel and length-1 axes.

    Mirrors ``hoct._io._reduce_to_movie``: a channel/sample axis is reduced to
    its first index and remaining length-1 axes (except the trailing Y, X) are
    collapsed. Used to check whether an input can be handed to HOCT as-is.
    """
    if shape is None:
        return None
    dims = list(shape)
    if axes:
        names = [a.lower() for a in axes]
        if len(names) == len(dims):
            for channel in ("c", "s"):
                if channel in names:
                    index = names.index(channel)
                    names.pop(index)
                    dims.pop(index)
    keep_from = len(dims) - 2
    return tuple(d for i, d in enumerate(dims) if i >= keep_from or d != 1)


def _zarr_axes(zarr_path: str):
    """Axes string ('TCZYX') of a Zarr store, from OME metadata if present."""
    from napari_tmidas._file_selector import detect_axes_from_zarr_path

    try:
        return detect_axes_from_zarr_path(zarr_path)
    except Exception:
        return None


def _stage_chunks(frame_shape):
    """Chunk shape for a staged store: one timepoint, capped spatial blocks."""
    hint = _STAGE_CHUNK_HINT[-len(frame_shape) :]
    return (1,) + tuple(min(s, c) for s, c in zip(frame_shape, hint))


def _create_stage_array(dest: Path, shape, dtype):
    """Create the destination Zarr array for a staged input."""
    import zarr

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    return zarr.create_array(
        store=str(dest),
        shape=tuple(shape),
        chunks=_stage_chunks(tuple(shape)[1:]),
        dtype=dtype,
        overwrite=True,
    )


def _stage_tiff_as_zarr(
    src_path: str,
    dest: Path,
    out_dtype=None,
    channel_axis: int = None,
    ch_idx: int = 0,
) -> Path:
    """Stream a multi-page TIFF into a Zarr store, one timepoint at a time.

    Peak RAM is one timepoint (optionally including its channels), not the
    whole movie, because pages are read per timepoint via
    ``TiffPageSeries.asarray(key=...)`` rather than in one ``asarray()`` call.
    ``channel_axis`` (in *source* axis numbering) selects a single channel on
    the way out, so a multichannel raw image needs no separate extraction pass.
    """
    with tifffile.TiffFile(src_path) as tif:
        series = tif.series[0]
        src_shape = tuple(series.shape)
        n_timepoints = src_shape[0]
        block_shape = src_shape[1:]

        # How many TIFF pages make up one timepoint. This is *not* simply
        # prod(shape[1:-2]): tifffile writes some stacks as volumetric pages
        # that already hold (Z, Y, X), so ask the series how many pages it has.
        # A layout whose pages don't split evenly across time (e.g. the whole
        # stack in one page) can't be read per timepoint; read it in one go
        # instead, which still writes a store HOCT can then read lazily.
        n_pages = len(series.pages)
        pages_per_t = (
            n_pages // n_timepoints
            if n_timepoints >= 1 and n_pages % n_timepoints == 0
            else None
        )
        if pages_per_t is None:
            print(
                f"{os.path.basename(src_path)}: {n_pages} TIFF pages do not "
                f"split evenly across {n_timepoints} timepoints; reading the "
                "whole array once to stage it."
            )
            whole = series.asarray()

        if channel_axis is not None:
            # Index within a single timepoint block (source axis 0 is time).
            block_channel_axis = channel_axis - 1
            frame_shape = tuple(
                d for i, d in enumerate(block_shape) if i != block_channel_axis
            )
        else:
            block_channel_axis = None
            frame_shape = block_shape

        out_shape = (n_timepoints,) + frame_shape
        out_dtype = out_dtype if out_dtype is not None else series.dtype
        zdst = _create_stage_array(dest, out_shape, out_dtype)

        for t in range(n_timepoints):
            if pages_per_t is None:
                block = whole[t]
            else:
                block = series.asarray(
                    key=slice(t * pages_per_t, (t + 1) * pages_per_t)
                ).reshape(block_shape)
            if block_channel_axis is not None:
                block = np.take(block, ch_idx, axis=block_channel_axis)
            zdst[t] = block.astype(out_dtype, copy=False)
    return dest


def _stage_zarr_as_zarr(
    src_path: str,
    dest: Path,
    out_dtype=None,
    channel_axis: int = None,
    ch_idx: int = 0,
) -> Path:
    """Copy a Zarr store timepoint by timepoint, optionally taking one channel.

    Used when a Zarr input cannot be handed to HOCT unchanged (e.g. a channel
    other than the first is requested). Only one timepoint is in RAM at a time.
    """
    src = _open_zarr_array(src_path)
    src_shape = tuple(src.shape)
    n_timepoints = src_shape[0]

    index = [slice(None)] * len(src_shape)
    index[0] = 0
    if channel_axis is not None:
        index[channel_axis] = ch_idx
    # Frame shape after dropping time and channel, then collapsing length-1
    # axes the same way HOCT's reader would.
    frame_shape = _movie_shape(
        tuple(d for i, d in enumerate(src_shape) if i not in (0, channel_axis))
    )

    out_shape = (n_timepoints,) + frame_shape
    out_dtype = out_dtype if out_dtype is not None else src.dtype
    zdst = _create_stage_array(dest, out_shape, out_dtype)

    for t in range(n_timepoints):
        index[0] = t
        block = np.asarray(src[tuple(index)]).reshape(frame_shape)
        zdst[t] = block.astype(out_dtype, copy=False)
    return dest


def _stage_input_as_zarr(
    src_path: str,
    dest: Path,
    out_dtype=None,
    channel_axis: int = None,
    ch_idx: int = 0,
) -> Path:
    """Stage a TIFF or Zarr input into a temporary Zarr store (streaming)."""
    if str(src_path).lower().endswith(".zarr"):
        return _stage_zarr_as_zarr(
            src_path, dest, out_dtype, channel_axis, ch_idx
        )
    return _stage_tiff_as_zarr(src_path, dest, out_dtype, channel_axis, ch_idx)


def _uncompressed_bytes(shape, dtype) -> int:
    """In-RAM size of an array, i.e. what an eager TIFF load would cost."""
    if shape is None:
        return 0
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def _stage_label_input(
    mask_path: str, work_dir: Path, stage_mode: str, tag: str
):
    """Give HOCT a lazily-readable segmentation input.

    A label TIFF is streamed into a temporary uint32 Zarr store when an eager
    load would be expensive; Zarr inputs and small TIFFs are passed through
    untouched. Returns (path_for_hoct, staged_path_or_None).
    """
    if stage_mode == "off" or mask_path.lower().endswith(".zarr"):
        return mask_path, None

    with tifffile.TiffFile(mask_path) as tif:
        series = tif.series[0]
        shape, dtype = tuple(series.shape), series.dtype

    eager_bytes = _uncompressed_bytes(shape, dtype)
    if stage_mode == "auto" and eager_bytes < _STAGE_MIN_BYTES:
        return mask_path, None

    dest = work_dir / f"hoct_labels_{_unique_tag()}.zarr"
    staged_bytes = _uncompressed_bytes(shape, _STAGE_LABEL_DTYPE)
    print(
        f"[{tag}] Staging segmentation to Zarr for lazy per-timepoint reads "
        f"({dtype} {shape}: {eager_bytes / 1024**3:.1f} GB in RAM if loaded "
        f"eagerly, {staged_bytes / 1024**3:.1f} GB as uint32)..."
    )
    _stage_input_as_zarr(mask_path, dest, out_dtype=_STAGE_LABEL_DTYPE)
    print(f"[{tag}] Segmentation staged at: {dest}")
    return str(dest), str(dest)


def _unique_tag() -> str:
    """Unique suffix for temporary files.

    The PID alone is not enough: a batch run tracks several files concurrently
    from *threads of one process*, so PID-named temporaries collide and jobs
    overwrite each other's staged raw image.
    """
    return f"{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _cleanup_paths(paths) -> None:
    """Delete temporary staged inputs (files or Zarr directories)."""
    for path in paths:
        target = Path(path)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)


def _prepare_raw_input(
    raw_path: str,
    channel_param: str,
    dimension_order_param: str,
    work_dir: Path,
    stage_mode: str = "auto",
    label_shape=None,
    tag: str = "",
):
    """
    Give HOCT a raw image it can read lazily, with no channel axis.

    HOCT expects the raw image to have the same shape as the label image
    (TYX or TZYX). Three cases, cheapest first:

    * the file already reduces to the label shape on HOCT's side (a
      single-channel input, or an OME-Zarr whose first channel is the one
      requested) — pass the path straight through, nothing is read;
    * otherwise stream the requested channel into a temporary Zarr store, one
      timepoint at a time;
    * with staging disabled, fall back to the previous behaviour (load the
      whole array, slice the channel, write a temporary TIFF).

    Returns (path_to_use, temp_path_or_None); the caller must delete
    temp_path_or_None (if not None) once the HOCT subprocess has finished.
    """
    channel_axis = None
    num_channels = 1

    # An explicit dimension_order hint (e.g. "TCZYX") is authoritative when
    # its length matches the raw array's ndim, mirroring Trackastra's
    # channel-axis resolution.
    if dimension_order_param and str(dimension_order_param).upper() not in (
        "AUTO",
        "NONE",
        "",
    ):
        axes = str(dimension_order_param).upper()
        if "C" in axes:
            shape = _peek_raw_shape(raw_path)
            if shape is not None and len(axes) == len(shape):
                channel_axis = axes.index("C")
                num_channels = shape[channel_axis]
                print(
                    f"Using dimension_order={axes} requested channel axis {channel_axis}"
                )

    if channel_axis is None:
        from napari_tmidas._file_selector import detect_channels_for_file

        num_channels, channel_axis = detect_channels_for_file(raw_path)

    is_zarr = raw_path.lower().endswith(".zarr")
    raw_shape = _peek_raw_shape(raw_path)

    if num_channels <= 1 or channel_axis is None:
        # Single-channel input: HOCT can open it directly. A big TIFF is still
        # staged, since HOCT would otherwise read the whole movie into RAM.
        if (
            stage_mode != "off"
            and not is_zarr
            and raw_shape is not None
            and (
                stage_mode == "on"
                or _uncompressed_bytes(raw_shape, _peek_raw_dtype(raw_path))
                >= _STAGE_MIN_BYTES
            )
        ):
            return _stage_raw_to_zarr(raw_path, work_dir, None, 0, tag)
        return raw_path, None

    print(
        f"Raw image {os.path.basename(raw_path)} has {num_channels} channels "
        f"(axis {channel_axis}); extracting one channel for HOCT."
    )

    if channel_param in ("", "all", "None"):
        ch_idx = 0
        print(
            "No channel specified for multichannel raw image, using channel 0..."
        )
    else:
        try:
            ch_idx = int(channel_param)
        except ValueError:
            print(f"Invalid channel '{channel_param}', using channel 0...")
            ch_idx = 0

    if ch_idx < 0 or ch_idx >= num_channels:
        print(
            f"Channel index {ch_idx} out of bounds for {num_channels} channels; using 0"
        )
        ch_idx = 0

    # HOCT's Zarr reader already keeps only the *first* channel and reads the
    # store lazily, so when channel 0 is what we want and the reduced shape
    # matches the segmentation, the store can be handed over untouched — no
    # extraction pass, no temporary copy, no full array in RAM.
    if is_zarr and ch_idx == 0 and label_shape is not None:
        reduced = _movie_shape(raw_shape, _zarr_axes(raw_path))
        if reduced is not None and tuple(reduced) == tuple(label_shape):
            print(
                f"[{tag}] Raw Zarr store reduces to {reduced} (channel 0) and "
                "matches the segmentation; passing it to HOCT directly "
                "(read lazily, no channel extraction)."
            )
            return raw_path, None

    if stage_mode != "off":
        return _stage_raw_to_zarr(raw_path, work_dir, channel_axis, ch_idx, tag)

    # stage_mode="off": original eager path (whole array in RAM at once).
    if is_zarr:
        arr = _open_zarr_array(raw_path)
        channel_data = np.asarray(np.take(arr, ch_idx, axis=channel_axis))
    else:
        # tifffile has no cheap partial-page read for an arbitrary channel
        # axis, so the full array is loaded once here before slicing.
        full = tifffile.imread(raw_path)
        channel_data = np.take(full, ch_idx, axis=channel_axis)

    temp_raw_path = work_dir / f"hoct_raw_ch{ch_idx}_{_unique_tag()}.tif"
    imwrite(str(temp_raw_path), channel_data)
    print(
        f"Extracted channel {ch_idx} to temporary raw file: {temp_raw_path} "
        f"(shape {channel_data.shape})"
    )
    return str(temp_raw_path), str(temp_raw_path)


def _peek_raw_dtype(raw_path: str):
    """Read only the dtype of a raw TIFF/Zarr image (no pixel data)."""
    if raw_path.lower().endswith(".zarr"):
        return _open_zarr_array(raw_path).dtype
    with tifffile.TiffFile(raw_path) as tif:
        return tif.series[0].dtype


def _stage_raw_to_zarr(
    raw_path: str, work_dir: Path, channel_axis, ch_idx: int, tag: str
):
    """Stream a raw image (one channel of it) into a temporary Zarr store."""
    dest = work_dir / f"hoct_raw_ch{ch_idx}_{_unique_tag()}.zarr"
    print(
        f"[{tag}] Staging raw image to Zarr for lazy per-timepoint reads "
        f"(channel {ch_idx})..."
    )
    _stage_input_as_zarr(
        raw_path, dest, out_dtype=None, channel_axis=channel_axis, ch_idx=ch_idx
    )
    from_zarr = _open_zarr_array(str(dest))
    print(f"[{tag}] Raw image staged at: {dest} (shape {tuple(from_zarr.shape)})")
    return str(dest), str(dest)


@BatchProcessingRegistry.register(
    name="Track Cells with HOCT",
    suffix="_hoct_tracked",
    description=(
        "Track cells across time using HOCT (Higher-Order Cell Tracking "
        "Transformer, github.com/royerlab/hoct), a transformer-based tracker. "
        "Expects TYX or TZYX label images with a matching raw image of the "
        "same shape (or a multichannel raw image, with optional channel "
        "selection). Supports TIFF and zarr inputs."
    ),
    parameters={
        "channel": {
            "type": str,
            "default": "",
            "description": "Optional raw-image channel index for multichannel input. Leave empty to use the default first channel.",
        },
        # NOTE: the dimension order is not declared here on purpose --
        # it comes from the batch widget's "Dimension Order" dropdown,
        # which applies to every function. Declaring it again made the
        # user set the same thing twice, in two places that could disagree.
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
        "gpus": {
            "type": str,
            "default": "",
            "description": "Comma-separated GPU ids to use (e.g. '0' or '0,1'). Leave empty to auto-detect and use all available GPUs. Set to 'cpu' or 'none' to disable GPU pinning. Each pinned GPU runs its own subprocess, so this also bounds how many movies are held in RAM at once (see 'stage_inputs', which is what keeps that per-movie cost small).",
        },
        "workers_per_gpu": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 8,
            "description": "Number of concurrent HOCT jobs to run per GPU. Increase if a single card has enough VRAM to run more than one tracking job at once (multi-GPU workstations benefit most). Only used when device='cuda'.",
        },
        "stage_inputs": {
            "type": str,
            "default": "auto",
            "options": ["auto", "on", "off"],
            "description": (
                "Stage large TIFF inputs into temporary Zarr stores so HOCT "
                "reads them one timepoint at a time instead of loading the "
                "whole movie into RAM (a 4D label TIFF can be tens of GB). "
                "'auto' stages inputs above 1 GB, 'off' restores the old "
                "load-everything behaviour."
            ),
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
    channel: str = "",
    dimension_order: str = "Auto",
    model: str = "",
    device: str = "cuda",
    window: int = 5,
    max_distance: float = 300.0,
    neighbors: int = 5,
    max_dt: int = 3,
    tile: str = "auto",
    scale: str = "",
    gurobi_license: str = "",
    gpus: str = "",
    workers_per_gpu: int = 1,
    stage_inputs: str = "auto",
    label_pattern: str = "_labels.tif",
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = "_hoct_tracked",
) -> np.ndarray:
    """
    Track cells in time-lapse label images using HOCT.

    This function takes a time series of segmentation masks and a matching
    raw image (same shape, or multichannel with an optional 'channel' index)
    and performs automatic cell tracking using HOCT
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
    channel : str
        Optional raw-image channel index for multichannel input. Leave
        empty to use the default first channel.
    dimension_order : str
        Dimension order hint for raw images (e.g., "TCZYX"). Helps with
        channel detection when the raw image is multichannel.
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
    gpus : str
        Comma-separated GPU ids to pin to (e.g. '0' or '0,1'). Empty
        auto-detects and uses all available GPUs; 'cpu'/'none' disables
        pinning. Restricting this bounds how many concurrent HOCT
        subprocesses run, which bounds RAM use as well as VRAM.
    workers_per_gpu : int
        Number of concurrent HOCT jobs to run per GPU (default: 1). Only
        used when device='cuda'.
    stage_inputs : str
        'auto' (default), 'on' or 'off'. HOCT reads a single multi-page TIFF
        eagerly (the whole T×Z×Y×X volume stays in RAM for the entire run)
        but reads Zarr stores lazily, one timepoint at a time. Staging
        streams large TIFF inputs into temporary Zarr stores first, which is
        what keeps big movies from being OOM-killed.
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
                "Input is not a time series (needs at least 3 dimensions). Skipping."
            )
            return None

        if image.shape[0] < 2:
            print(
                "Input has only one timepoint. Need at least 2 for tracking. Skipping."
            )
            return None
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

    if stage_inputs not in ("auto", "on", "off"):
        print(
            f"Warning: invalid stage_inputs '{stage_inputs}'. Using 'auto' instead."
        )
        stage_inputs = "auto"

    # Ensure HOCT environment exists and the CLI is usable.
    if not HoctEnvManager.ensure_env_ready():
        print("Failed to prepare HOCT environment. Skipping.")
        return None

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
        print("Could not determine input file path. Skipping.")
        return None

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
            print("HOCT requires a matching raw image. Skipping.")
            return None
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
            return None

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
    ctc_dir = temp_dir / f"hoct_ctc_{output_stem}_{_unique_tag()}"
    if ctc_dir.exists():
        shutil.rmtree(ctc_dir)

    label_tag = os.path.basename(mask_path)

    # Both inputs are prepared so the HOCT subprocess can read them one
    # timepoint at a time (see _STAGE_MIN_BYTES): the segmentation is streamed
    # into a temporary uint32 Zarr store when it is large, and the raw image
    # is either passed through (already lazily readable and channel-free) or
    # streamed into a Zarr store with the requested channel selected.
    staged_paths = []
    try:
        mask_path, staged_mask = _stage_label_input(
            str(mask_path), temp_dir, stage_inputs, label_tag
        )
        if staged_mask:
            staged_paths.append(staged_mask)

        label_shape = _peek_raw_shape(str(mask_path))
        raw_path, temp_raw_path = _prepare_raw_input(
            str(raw_path),
            channel,
            dimension_order,
            temp_dir,
            stage_mode=stage_inputs,
            label_shape=_movie_shape(label_shape),
            tag=label_tag,
        )
        if temp_raw_path:
            staged_paths.append(temp_raw_path)
    except Exception as exc:
        print(f"[{label_tag}] Failed to prepare HOCT inputs: {exc}")
        _cleanup_paths(staged_paths)
        return None

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
        pool, gpu_ids = _get_gpu_pool(workers_per_gpu, gpus)
        gpu_id = pool.get() if gpu_ids else None

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
        _cleanup_paths(staged_paths)

    if result.returncode != 0:
        print(f"HOCT error (exit code {result.returncode}):")
        print(result.stdout)
        print(result.stderr)
        print("Skipping — no output will be saved.")
        shutil.rmtree(ctc_dir, ignore_errors=True)
        return None

    print(result.stdout)

    try:
        out_shape = _assemble_ctc_output(ctc_dir, output_path)
    except Exception as exc:
        print(f"Failed to assemble HOCT CTC output: {exc}")
        return None
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

# processing_functions/cellpose_segmentation.py
"""
Processing functions for automatic instance segmentation using Cellpose.

This module provides functionality to automatically segment cells or nuclei in images
using the Cellpose deep learning-based segmentation toolkit. It supports both 2D and 3D images,
various dimension orders, and handles time series data.

Updated to support Cellpose 4 (Cellpose-SAM) which offers improved generalization
for cellular segmentation without requiring diameter parameter.

Note: This requires the cellpose library to be installed.
"""
import os
import shutil
import time
from contextlib import suppress
from tempfile import NamedTemporaryFile
from typing import Union
import json
import hashlib

import numpy as np
import tifffile
import zarr

# Import the environment manager
from napari_tmidas.processing_functions.cellpose_env_manager import (
    create_cellpose_env,
    is_env_created,
    run_cellpose_in_env,
)

# Check if cellpose is directly available in this environment
try:
    import cellpose  # noqa: F401

    CELLPOSE_AVAILABLE = True
    # Don't evaluate USE_GPU here - it should be evaluated in the cellpose environment

    print("Cellpose found in current environment. Using native import.")
except ImportError:
    CELLPOSE_AVAILABLE = False

    print(
        "Cellpose not found in current environment. Will use dedicated environment."
    )

from napari_tmidas._registry import BatchProcessingRegistry


def transpose_dimensions(img, dim_order):
    """
    Transpose image dimensions to match expected Cellpose input.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    dim_order : str
        Dimension order of the input image (e.g., 'ZYX', 'TZYX', 'YXC')

    Returns:
    --------
    numpy.ndarray
        Transposed image
    str
        New dimension order
    bool
        Whether the image is 3D
    """
    # Handle time dimension if present
    has_time = "T" in dim_order

    # Determine if the image is 3D (has Z dimension)
    is_3d = "Z" in dim_order

    # Standardize dimension order
    if has_time:
        # For time series, we want to end up with TZYX
        target_dims = "TZYX"
        transpose_order = [
            dim_order.index(d) for d in target_dims if d in dim_order
        ]
        new_dim_order = "".join([dim_order[i] for i in transpose_order])
    else:
        # For single timepoints, we want ZYX
        target_dims = "ZYX"
        transpose_order = [
            dim_order.index(d) for d in target_dims if d in dim_order
        ]
        new_dim_order = "".join([dim_order[i] for i in transpose_order])

    # Perform the transpose
    img_transposed = np.transpose(img, transpose_order)

    return img_transposed, new_dim_order, is_3d


@BatchProcessingRegistry.register(
    name="Cellpose-SAM Segmentation",
    suffix="_labels",
    description="Automatic instance segmentation using Cellpose 4 (Cellpose-SAM) with improved generalization. For multichannel images, select which channel to segment.",
    parameters={
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to segment (automatically detected from multichannel images)",
        },
        "dim_order": {
            "type": str,
            "default": "YX",
            "description": "Dimension order of the input (e.g., 'YX', 'ZYX', 'TZYX')",
        },
        "timepoint_start": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 100000,
            "description": "Start timepoint index (0-based, inclusive). Only used when T is present.",
        },
        "timepoint_end": {
            "type": int,
            "default": -1,
            "min": -1,
            "max": 100000,
            "description": "End timepoint index (0-based, inclusive). Use -1 for the last timepoint.",
        },
        "timepoint_step": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 100,
            "description": "Take every Nth timepoint (1 = all, 2 = every other, etc.).",
        },
        # "channel_1": {
        #     "type": int,
        #     "default": 0,
        #     "min": 0,
        #     "max": 3,
        #     "description": "First channel: 0=grayscale, 1=green, 2=red, 3=blue",
        # },
        # "channel_2": {
        #     "type": int,
        #     "default": 0,
        #     "min": 0,
        #     "max": 3,
        #     "description": "Second channel: 0=none, 1=green, 2=red, 3=blue",
        # },
        "diameter": {
            "type": float,
            "default": 0.0,
            "min": 0.0,
            "max": 200.0,
            "description": "Optional. Only required if your ROI diameter is outside the range 7.5–120. Set to 0 to leave unset (recommended for most users). Cellpose-SAM is trained for diameters in this range.",
        },
        "flow_threshold": {
            "type": float,
            "default": 0.4,
            "min": 0.1,
            "max": 0.9,
            "description": "Flow threshold (cell detection sensitivity)",
        },
        "cellprob_threshold": {
            "type": float,
            "default": 0.0,
            "min": -6.0,
            "max": 6.0,
            "description": "Cell probability threshold (increase to reduce splits)",
        },
        "anisotropy": {
            "type": float,
            "default": 1.0,
            "min": 0.1,
            "max": 100.0,
            "description": "Optional rescaling factor (3D; Z step[um] / X pixel res [um]). Highly anisotropic datasets may require values >10.",
        },
        "flow3D_smooth": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 10,
            "description": "Smooth flow with gaussian filter (stdev)",
        },
        "tile_norm_blocksize": {
            "type": int,
            "default": 128,
            "min": 32,
            "max": 512,
            "description": "Block size for tile normalization (Cellpose 4 parameter)",
        },
        "batch_size": {
            "type": int,
            "default": 32,
            "min": 1,
            "max": 128,
            "description": "Batch size for processing multiple images/slices at once",
        },
        "use_distributed_segmentation": {
            "type": bool,
            "default": False,
            "description": "Use distributed blockwise segmentation (cellpose.contrib.distributed_segmentation) for large zarr volumes (for example ZYX, TZYX, TCZYX). Reduces peak memory use during inference. Non-zarr inputs are auto-converted to temporary zarr when possible.",
        },
        "distributed_blocksize": {
            "type": int,
            "default": 256,
            "min": 64,
            "max": 1024,
            "description": "Block edge length (voxels) for distributed segmentation. Only used when use_distributed_segmentation=True.",
        },
        "use_convpaint_auto_mask": {
            "type": bool,
            "default": False,
            "description": "Use ConvPaint model to generate a foreground mask before distributed Cellpose. Blocks outside mask are skipped.",
        },
        "convpaint_model_path": {
            "type": str,
            "default": "",
            "description": "Path to ConvPaint .pkl model used for auto-mask generation (required when use_convpaint_auto_mask=True).",
        },
        "convpaint_image_downsample": {
            "type": int,
            "default": 4,
            "min": 1,
            "max": 16,
            "description": "Downsampling for ConvPaint auto-mask inference (higher is faster, lower keeps detail).",
        },
        "convpaint_background_label": {
            "type": int,
            "default": 1,
            "min": 0,
            "max": 255,
            "description": "ConvPaint label id treated as background for foreground-mask conversion.",
        },
        "convpaint_mask_dilation": {
            "type": int,
            "default": 2,
            "min": 0,
            "max": 20,
            "description": "Binary dilation iterations applied to ConvPaint foreground mask to avoid false negatives.",
        },
        "convpaint_min_object_fraction_of_median": {
            "type": float,
            "default": 0.25,
            "min": 0.0,
            "max": 2.0,
            "description": "Remove ConvPaint foreground components smaller than this fraction of the slab median component size (0 disables). Helps suppress tiny speckles that would otherwise activate full distributed blocks.",
        },
        "clip_final_labels_to_convpaint_mask": {
            "type": bool,
            "default": True,
            "description": "After Cellpose finishes, zero labels outside the fine ConvPaint mask. Keeps the final result from inheriting block-level gating artifacts.",
        },
        "convpaint_use_cpu": {
            "type": bool,
            "default": False,
            "description": "Run ConvPaint mask generation on CPU.",
        },
        "convpaint_force_dedicated_env": {
            "type": bool,
            "default": False,
            "description": "Force ConvPaint mask generation in dedicated environment.",
        },
        "convpaint_z_batch_size": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 200,
            "description": "Z-batching for ConvPaint auto-mask generation (0 disables batching).",
        },
    },
)
def cellpose_segmentation(
    image: np.ndarray,
    channel: str = "all",
    dim_order: str = "YX",
    channel_1: int = 0,
    channel_2: int = 0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    anisotropy: Union[float, None] = None,
    flow3D_smooth: int = 0,
    tile_norm_blocksize: int = 128,
    batch_size: int = 32,
    diameter: float = 0.0,
    use_distributed_segmentation: bool = False,
    distributed_blocksize: int = 256,
    use_convpaint_auto_mask: bool = False,
    convpaint_model_path: str = "",
    convpaint_image_downsample: int = 4,
    convpaint_background_label: int = 1,
    convpaint_mask_dilation: int = 2,
    convpaint_min_object_fraction_of_median: float = 0.25,
    clip_final_labels_to_convpaint_mask: bool = True,
    convpaint_use_cpu: bool = False,
    convpaint_force_dedicated_env: bool = False,
    convpaint_z_batch_size: int = 0,
    timepoint_start: int = 0,
    timepoint_end: int = -1,
    timepoint_step: int = 1,
    _source_filepath: str = None,  # Hidden parameter for zarr optimization
    _output_folder: str = None,
    _output_suffix: str = None,
    _output_format: str = "tiff",
) -> np.ndarray:
    """
    Run Cellpose 4 (Cellpose-SAM) segmentation on an image.

    This function takes an image and performs automatic instance segmentation using
    Cellpose 4 with improved generalization for cellular segmentation. It supports
    both 2D and 3D images, various dimension orders, and handles time series data.

    If Cellpose is not available in the current environment, a dedicated virtual
    environment will be created to run Cellpose.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    dim_order : str
        Dimension order of the input (e.g., 'YX', 'ZYX', 'TZYX') (default: "YX")
    channel_1 : int
        First channel: 0=grayscale, 1=green, 2=red, 3=blue (default: 0)
    channel_2 : int
        Second channel: 0=none, 1=green, 2=red, 3=blue (default: 0)
    flow_threshold : float
        Flow threshold for Cellpose segmentation (default: 0.4)
    cellprob_threshold : float
        Cell probability threshold (Cellpose 4 parameter) (default: 0.0)
    tile_norm_blocksize : int
        Block size for tile normalization (Cellpose 4 parameter) (default: 128)
    batch_size : int
        Batch size for processing multiple images/slices at once (default: 32)

    Returns:
    --------
    numpy.ndarray
        Segmented image with instance labels
    """
    # Diameter parameter guidance:
    # Cellpose-SAM is trained for ROI diameters 7.5–120. Only set diameter if your images have objects outside this range (e.g., diameter <7.5 or >120). Otherwise, leave as None.
    # Cellpose 4 handles normalization internally via percentile-based normalization
    # It accepts uint8, uint16, float32, float64 - no pre-conversion needed!
    # The normalize=True parameter (default) will convert to float and normalize
    # to 1st-99th percentile range internally

    print(
        "Cellpose runtime options: "
        f"distributed_requested={use_distributed_segmentation}, "
        f"source_path={_source_filepath}"
    )

    def _direct_output_path() -> Union[str, None]:
        if not (_output_folder and _output_suffix and _source_filepath):
            return None

        source_base = os.path.splitext(
            os.path.basename(_source_filepath)
        )[0]
        output_ext = ".zarr" if _output_format == "zarr" else ".tif"
        return os.path.join(
            _output_folder,
            f"{source_base}{_output_suffix}{output_ext}",
        )

    def _write_interleaved_checkpoint_output(
        checkpoint: zarr.Array, checkpoint_path: str
    ) -> Union[str, None]:
        output_path = _direct_output_path()
        if not output_path:
            return None

        os.makedirs(_output_folder, exist_ok=True)
        if _output_format == "zarr":
            shutil.rmtree(output_path, ignore_errors=True)
            shutil.copytree(checkpoint_path, output_path)
            print(f"Saved interleaved checkpoint zarr to: {output_path}")
            return output_path

        shape = tuple(int(s) for s in checkpoint.shape)
        size_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.uint32).itemsize
        use_bigtiff = (size_bytes / (1024**3)) > 2.0
        axes_by_ndim = {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}
        axes = axes_by_ndim.get(len(shape), "YX")

        if os.path.exists(output_path):
            os.remove(output_path)

        if len(shape) <= 2:
            tifffile.imwrite(
                output_path,
                np.asarray(checkpoint, dtype=np.uint32),
                compression="zlib",
                bigtiff=use_bigtiff,
                metadata={"axes": axes},
            )
            print(f"Saved interleaved checkpoint TIFF to: {output_path}")
            return output_path

        def _iter_checkpoint_slabs():
            for i in range(shape[0]):
                yield np.asarray(checkpoint[i], dtype=np.uint32)

        tifffile.imwrite(
            output_path,
            data=_iter_checkpoint_slabs(),
            shape=shape,
            dtype=np.uint32,
            compression="zlib",
            photometric="minisblack",
            bigtiff=use_bigtiff,
            metadata={"axes": axes},
        )
        print(f"Saved interleaved checkpoint TIFF to: {output_path}")
        return output_path

    # Prefer direct zarr processing when a source zarr path is available.
    # This keeps distributed segmentation enabled for full TCZYX workflows.
    use_zarr_direct = bool(_source_filepath) and (
        _source_filepath.lower().endswith(".zarr")
        or (
            os.path.isdir(_source_filepath)
            and os.path.exists(os.path.join(_source_filepath, ".zattrs"))
        )
    )

    # Distributed Cellpose relies on zarr-backed slabs. If the user requested
    # distributed mode on a non-zarr source, automatically convert using the
    # existing microscopy conversion helper and continue in zarr-direct mode.
    if (
        use_distributed_segmentation
        and not use_zarr_direct
        and bool(_source_filepath)
    ):
        try:
            from napari_tmidas._file_selector import save_as_zarr

            source_abs = os.path.abspath(_source_filepath)
            source_parent = os.path.dirname(source_abs)
            tmp_root = os.path.join(source_parent, "tmp")
            os.makedirs(tmp_root, exist_ok=True)

            auto_root = os.path.join(tmp_root, "cellpose_auto_zarr")
            os.makedirs(auto_root, exist_ok=True)

            cache_payload = {
                "source": source_abs,
                "source_mtime": os.path.getmtime(source_abs),
                "shape": tuple(int(s) for s in image.shape),
                "dtype": str(getattr(image, "dtype", "unknown")),
                "dim_order": str(dim_order),
                "channel": str(channel),
            }
            cache_key = hashlib.sha1(
                json.dumps(cache_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]

            source_base = os.path.splitext(os.path.basename(source_abs))[0]
            auto_zarr_path = os.path.join(
                auto_root, f"{source_base}_cellpose_{cache_key}.zarr"
            )

            if not os.path.exists(auto_zarr_path):
                axes = str(dim_order).upper() if dim_order else ""
                if len(axes) != image.ndim:
                    axes = "TCZYX"

                print(
                    "Distributed segmentation requested on non-zarr input; "
                    "auto-converting source to zarr..."
                )
                save_as_zarr(
                    data=np.asarray(image),
                    filepath=auto_zarr_path,
                    axes=axes,
                )
            else:
                print(
                    "Distributed segmentation: reusing cached auto-converted "
                    f"zarr: {auto_zarr_path}"
                )

            _source_filepath = auto_zarr_path
            use_zarr_direct = True
            print(
                "Distributed segmentation: using auto-converted zarr source: "
                f"{_source_filepath}"
            )
        except Exception as exc:
            print(
                "Warning: auto zarr conversion for distributed segmentation "
                f"failed ({exc}). Falling back to standard Cellpose evaluation."
            )

    def _resolve_timepoint_indices(total_timepoints: int):
        """Resolve selected timepoints from start/end/step controls."""
        if total_timepoints <= 0:
            return []

        tp_start = int(timepoint_start)
        tp_end = int(timepoint_end)
        tp_step = int(timepoint_step)

        if tp_step <= 0:
            raise ValueError("timepoint_step must be >= 1")

        # Support Python-like negative indexing for start/end.
        if tp_start < 0:
            tp_start = total_timepoints + tp_start
        tp_start = max(0, min(tp_start, total_timepoints - 1))

        if tp_end < 0:
            tp_end = total_timepoints - 1
        else:
            tp_end = max(0, min(tp_end, total_timepoints - 1))

        if tp_end < tp_start:
            raise ValueError(
                "timepoint_end must be >= timepoint_start after bounds handling "
                f"(got start={tp_start}, end={tp_end})"
            )

        indices = list(range(tp_start, tp_end + 1, tp_step))
        if not indices:
            raise ValueError("No timepoints selected. Check start/end/step settings.")

        return indices

    generated_mask_path = None
    generated_mask_zarr_path = None
    convpaint_mask = None

    if use_convpaint_auto_mask:
        print("ConvPaint auto-mask requested: generating foreground mask...")
        if not convpaint_model_path or not convpaint_model_path.strip():
            raise ValueError(
                "convpaint_model_path is required when use_convpaint_auto_mask=True"
            )
        try:
            from napari_tmidas.processing_functions.convpaint_prediction import (
                convpaint_predict,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import ConvPaint prediction function for auto-mask generation"
            ) from exc

        # Guardrail: if ConvPaint mask is too sparse for distributed block-wise
        # processing, disable mask pruning for that slab to avoid missing whole
        # neighboring blocks that still contain cells.
        sparse_mask_fraction_threshold = 1e-4

        def _compute_active_block_grid(mask: np.ndarray, blocksize: int):
            """Return boolean grid of active distributed blocks for a ZYX mask."""
            if mask.ndim != 3 or blocksize <= 0:
                return None

            bs = int(blocksize)
            z, y, x = mask.shape
            gz = (z + bs - 1) // bs
            gy = (y + bs - 1) // bs
            gx = (x + bs - 1) // bs
            active = np.zeros((gz, gy, gx), dtype=bool)

            for iz in range(gz):
                z0, z1 = iz * bs, min((iz + 1) * bs, z)
                for iy in range(gy):
                    y0, y1 = iy * bs, min((iy + 1) * bs, y)
                    for ix in range(gx):
                        x0, x1 = ix * bs, min((ix + 1) * bs, x)
                        active[iz, iy, ix] = bool(
                            np.any(mask[z0:z1, y0:y1, x0:x1])
                        )

            return active

        def _expand_mask_to_neighbor_blocks(
            mask: np.ndarray, blocksize: int, margin_blocks: int = 1
        ) -> np.ndarray:
            """Expand mask at distributed-block granularity in YX."""
            active = _compute_active_block_grid(mask, blocksize)
            if active is None:
                return mask

            if margin_blocks > 0:
                try:
                    from scipy import ndimage as ndi

                    # Expand only in YX neighbors for same Z block.
                    structure = np.ones(
                        (1, 2 * margin_blocks + 1, 2 * margin_blocks + 1),
                        dtype=bool,
                    )
                    active = ndi.binary_dilation(active, structure=structure)
                except Exception as exc:
                    print(
                        "Warning: block-level mask expansion failed "
                        f"({exc}); using original block activity"
                    )

            bs = int(blocksize)
            z, y, x = mask.shape
            expanded = np.zeros_like(mask, dtype=np.uint8)

            for iz in range(active.shape[0]):
                z0, z1 = iz * bs, min((iz + 1) * bs, z)
                for iy in range(active.shape[1]):
                    y0, y1 = iy * bs, min((iy + 1) * bs, y)
                    for ix in range(active.shape[2]):
                        if not active[iz, iy, ix]:
                            continue
                        x0, x1 = ix * bs, min((ix + 1) * bs, x)
                        expanded[z0:z1, y0:y1, x0:x1] = 1

            return expanded

        def _summarize_mask(mask: np.ndarray, blocksize: int):
            """Return (foreground_fraction, active_blocks, total_blocks)."""
            fg = float(np.mean(mask > 0))
            active_grid = _compute_active_block_grid(mask > 0, blocksize)
            if active_grid is None:
                return fg, None, None
            return fg, int(np.count_nonzero(active_grid)), int(active_grid.size)

        def _labels_to_mask(labels: np.ndarray) -> np.ndarray:
            mask = (labels > 0).astype(np.uint8)
            if convpaint_mask_dilation > 0:
                try:
                    from scipy import ndimage as ndi

                    mask = ndi.binary_dilation(
                        mask, iterations=convpaint_mask_dilation
                    ).astype(np.uint8)
                except Exception as exc:
                    print(
                        "Warning: ConvPaint mask dilation failed "
                        f"({exc}); using undilated mask"
                    )

            if convpaint_min_object_fraction_of_median > 0:
                try:
                    from scipy import ndimage as ndi

                    cc, n_cc = ndi.label(mask > 0)
                    if n_cc > 0:
                        counts = np.bincount(cc.ravel())
                        comp_sizes = counts[1:]
                        if comp_sizes.size > 0:
                            median_size = float(np.median(comp_sizes))
                            # Keep this floor tiny: just enough to remove
                            # single-voxel noise when median is very small.
                            min_keep_size = max(
                                8,
                                int(
                                    np.ceil(
                                        median_size
                                        * float(
                                            convpaint_min_object_fraction_of_median
                                        )
                                    )
                                ),
                            )
                            keep_ids = np.where(comp_sizes >= min_keep_size)[0] + 1
                            filtered = np.isin(cc, keep_ids)
                            removed = int(n_cc - keep_ids.size)
                            if removed > 0:
                                print(
                                    "ConvPaint auto-mask small-object filter: "
                                    f"removed {removed}/{int(n_cc)} components "
                                    f"(min_size={min_keep_size} voxels, "
                                    f"median_size={median_size:.1f})"
                                )
                            mask = filtered.astype(np.uint8)
                except Exception as exc:
                    print(
                        "Warning: ConvPaint small-object filtering failed "
                        f"({exc}); using unfiltered mask"
                    )

            return mask

        def _prepare_runtime_distributed_mask(
            mask: np.ndarray,
            cached_mask_path: str,
            mask_cache_root: str,
            timepoint_index: int,
            blocksize: int,
        ):
            """Build the runtime block-gating mask used by distributed Cellpose."""
            runtime_mask = _expand_mask_to_neighbor_blocks(
                (mask > 0).astype(np.uint8),
                blocksize=blocksize,
                margin_blocks=1,
            ).astype(np.uint8)
            runtime_mask_path = cached_mask_path
            if not np.array_equal(runtime_mask > 0, mask > 0):
                runtime_mask_path = os.path.join(
                    mask_cache_root,
                    f"t{timepoint_index:04d}_mask_runtime_bs{int(blocksize)}.tif",
                )
                tifffile.imwrite(runtime_mask_path, runtime_mask)

            runtime_fg, runtime_active_blocks, runtime_total_blocks = _summarize_mask(
                runtime_mask, blocksize
            )
            return (
                runtime_mask,
                runtime_mask_path,
                runtime_fg,
                runtime_active_blocks,
                runtime_total_blocks,
            )

        # For zarr-direct workflows, build a mask zarr slab-by-slab (per timepoint)
        # to avoid loading the full dataset into RAM.
        if (
            use_zarr_direct
            and use_distributed_segmentation
            and image.ndim == 4
        ):
            # For distributed TZYX zarr workflows, defer mask generation to
            # an interleaved per-timepoint loop below: mask(T) -> segment(T).
            print(
                "ConvPaint auto-mask: using interleaved per-timepoint "
                "mask->segment flow for distributed zarr processing"
            )
        elif use_zarr_direct:
            source_parent = os.path.dirname(os.path.abspath(_source_filepath))
            tmp_root = os.path.join(source_parent, "tmp")
            os.makedirs(tmp_root, exist_ok=True)
            generated_mask_zarr_path = os.path.join(
                tmp_root, "convpaint_auto_mask.zarr"
            )
            if os.path.exists(generated_mask_zarr_path):
                shutil.rmtree(generated_mask_zarr_path, ignore_errors=True)

            image_shape = tuple(int(s) for s in image.shape)
            if hasattr(image, "chunks"):
                chunks = tuple(
                    int(c[0]) if isinstance(c, tuple) else int(c)
                    for c in image.chunks
                )
            else:
                chunks = tuple(max(1, min(16, s)) for s in image_shape)

            mask_zarr = zarr.open_array(
                generated_mask_zarr_path,
                mode="w",
                shape=image_shape,
                chunks=chunks,
                dtype=np.uint8,
            )

            if image.ndim == 4:
                t_count = image_shape[0]
                print(
                    "ConvPaint auto-mask: generating slab-wise mask "
                    f"for {t_count} timepoints..."
                )
                for t in range(t_count):
                    print(f"ConvPaint auto-mask slab {t+1}/{t_count}")
                    slab = np.asarray(image[t])
                    labels = convpaint_predict(
                        slab,
                        channel="all",
                        model_path=convpaint_model_path,
                        image_downsample=convpaint_image_downsample,
                        output_type="semantic",
                        background_label=convpaint_background_label,
                        use_cpu=convpaint_use_cpu,
                        force_dedicated_env=convpaint_force_dedicated_env,
                        z_batch_size=convpaint_z_batch_size,
                        tmp_dir=tmp_root,
                    )
                    mask_zarr[t] = _labels_to_mask(labels)
            elif image.ndim == 5:
                t_count, c_count = image_shape[:2]
                print(
                    "ConvPaint auto-mask: generating slab-wise mask "
                    f"for {t_count} timepoints x {c_count} channels..."
                )
                for t in range(t_count):
                    for c in range(c_count):
                        print(
                            "ConvPaint auto-mask slab "
                            f"T{t+1}/{t_count}, C{c+1}/{c_count}"
                        )
                        slab = np.asarray(image[t, c])
                        labels = convpaint_predict(
                            slab,
                            channel="all",
                            model_path=convpaint_model_path,
                            image_downsample=convpaint_image_downsample,
                            output_type="semantic",
                            background_label=convpaint_background_label,
                            use_cpu=convpaint_use_cpu,
                            force_dedicated_env=convpaint_force_dedicated_env,
                            z_batch_size=convpaint_z_batch_size,
                            tmp_dir=tmp_root,
                        )
                        mask_zarr[t, c] = _labels_to_mask(labels)
            elif image.ndim == 3:
                print("ConvPaint auto-mask: generating slab mask for 3D volume...")
                slab = np.asarray(image)
                labels = convpaint_predict(
                    slab,
                    channel="all",
                    model_path=convpaint_model_path,
                    image_downsample=convpaint_image_downsample,
                    output_type="semantic",
                    background_label=convpaint_background_label,
                    use_cpu=convpaint_use_cpu,
                    force_dedicated_env=convpaint_force_dedicated_env,
                    z_batch_size=convpaint_z_batch_size,
                    tmp_dir=tmp_root,
                )
                mask_zarr[:] = _labels_to_mask(labels)
            else:
                # Non-3D/4D/5D masks are unsupported in distributed zarr mode.
                raise ValueError(
                    "ConvPaint auto-mask with distributed zarr expects 3D, 4D, or 5D "
                    f"image data after channel selection, got shape {image_shape}."
                )

            print(
                "Saved ConvPaint slab-wise mask zarr for distributed processing: "
                f"{generated_mask_zarr_path}"
            )
        else:
            convpaint_input = np.asarray(image)
            convpaint_labels = convpaint_predict(
                convpaint_input,
                channel="all",
                model_path=convpaint_model_path,
                image_downsample=convpaint_image_downsample,
                output_type="semantic",
                background_label=convpaint_background_label,
                use_cpu=convpaint_use_cpu,
                force_dedicated_env=convpaint_force_dedicated_env,
                z_batch_size=convpaint_z_batch_size,
            )
            convpaint_mask = _labels_to_mask(convpaint_labels)
            foreground_fraction = float(convpaint_mask.mean())
            print(
                "ConvPaint auto-mask ready: "
                f"shape={convpaint_mask.shape}, foreground_fraction={foreground_fraction:.4f}"
            )

    # Handle any time-series data by processing each timepoint separately.
    # This includes TYX (2D time-lapse) and TZYX (3D time-lapse).
    if "T" in dim_order and not use_zarr_direct:
        t_axis = dim_order.index("T")
        if t_axis >= image.ndim:
            raise ValueError(
                "dim_order contains 'T' but image does not have a matching T axis "
                f"(dim_order={dim_order}, shape={image.shape})."
            )

        print(
            f"Detected time-series data with shape {image.shape}. Processing each timepoint separately..."
        )
        num_timepoints = image.shape[t_axis]
        selected_timepoints = _resolve_timepoint_indices(int(num_timepoints))

        print(
            "Selected timepoints: "
            f"{selected_timepoints[0]}..{selected_timepoints[-1]} "
            f"(count={len(selected_timepoints)}, step={int(timepoint_step)})"
        )

        # Move T axis to front if not already there
        if t_axis != 0:
            axes_order = list(range(image.ndim))
            axes_order.insert(0, axes_order.pop(t_axis))
            image = np.transpose(image, axes_order)

        # Process each timepoint
        results = []
        for out_idx, t in enumerate(selected_timepoints):
            print(
                f"Processing timepoint {t+1}/{num_timepoints} "
                f"(selected {out_idx+1}/{len(selected_timepoints)})..."
            )
            timepoint_image = image[t]
            # Remove 'T' from dim_order for single timepoint
            timepoint_dim_order = dim_order.replace("T", "")

            # Recursively call this function for the single timepoint
            timepoint_result = cellpose_segmentation(
                timepoint_image,
                channel=channel,
                dim_order=timepoint_dim_order,
                timepoint_start=0,
                timepoint_end=-1,
                timepoint_step=1,
                channel_1=channel_1,
                channel_2=channel_2,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                anisotropy=anisotropy,
                flow3D_smooth=flow3D_smooth,
                tile_norm_blocksize=tile_norm_blocksize,
                batch_size=batch_size,
                diameter=diameter,
                use_distributed_segmentation=use_distributed_segmentation,
                distributed_blocksize=distributed_blocksize,
                use_convpaint_auto_mask=use_convpaint_auto_mask,
                convpaint_model_path=convpaint_model_path,
                convpaint_image_downsample=convpaint_image_downsample,
                convpaint_background_label=convpaint_background_label,
                convpaint_mask_dilation=convpaint_mask_dilation,
                convpaint_min_object_fraction_of_median=(
                    convpaint_min_object_fraction_of_median
                ),
                convpaint_use_cpu=convpaint_use_cpu,
                convpaint_force_dedicated_env=convpaint_force_dedicated_env,
                convpaint_z_batch_size=convpaint_z_batch_size,
                _source_filepath=None,  # Don't use zarr optimization for timepoints
            )
            results.append(timepoint_result)

        # Stack results back together
        result = np.stack(results, axis=0)
        print(
            "Completed processing selected timepoints: "
            f"{len(selected_timepoints)}/{num_timepoints}."
        )
        return result

    # Convert channel parameters to Cellpose channels list
    # channels = [channel_1, channel_2]
    channels = [0, 0]  # limit script to single channel

    # First check if the environment exists, create if not
    if not is_env_created():
        print(
            "Creating dedicated Cellpose environment (this may take a few minutes)..."
        )
        create_cellpose_env()
        print("Environment created successfully.")

    if use_zarr_direct:
        print(f"Using optimized zarr processing for: {_source_filepath}")

        # Interleaved mode: for each timepoint, generate ConvPaint foreground mask
        # and immediately run distributed Cellpose for that same slab.
        if (
            use_distributed_segmentation
            and use_convpaint_auto_mask
            and image.ndim == 4
        ):
            source_parent = os.path.dirname(os.path.abspath(_source_filepath))
            tmp_root = os.path.join(source_parent, "tmp")
            os.makedirs(tmp_root, exist_ok=True)

            t_count = int(image.shape[0])
            selected_timepoints = _resolve_timepoint_indices(t_count)
            selected_count = len(selected_timepoints)
            source_base = os.path.splitext(
                os.path.basename(_source_filepath)
            )[0]
            print(
                "Distributed+ConvPaint interleaved mode: "
                f"processing {selected_count} selected timepoints "
                f"out of {t_count} (mask->segment per slab)"
            )

            # Cache ConvPaint masks on disk so reruns can reuse them.
            mask_cache_key_payload = {
                "source": os.path.abspath(_source_filepath),
                "source_mtime": os.path.getmtime(_source_filepath),
                "mask_cache_version": 3,
                "channel": channel,
                "convpaint_model_path": convpaint_model_path,
                "convpaint_image_downsample": int(convpaint_image_downsample),
                "convpaint_background_label": int(convpaint_background_label),
                "convpaint_mask_dilation": int(convpaint_mask_dilation),
                "convpaint_min_object_fraction_of_median": float(
                    convpaint_min_object_fraction_of_median
                ),
                "convpaint_use_cpu": bool(convpaint_use_cpu),
                "convpaint_force_dedicated_env": bool(
                    convpaint_force_dedicated_env
                ),
                "convpaint_z_batch_size": int(convpaint_z_batch_size),
                "shape": tuple(int(s) for s in image.shape),
            }
            mask_cache_key = hashlib.sha1(
                json.dumps(mask_cache_key_payload, sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            mask_cache_root = os.path.join(
                tmp_root,
                "convpaint_mask_cache",
                f"{source_base}_{mask_cache_key}",
            )
            os.makedirs(mask_cache_root, exist_ok=True)
            print(f"ConvPaint mask cache: {mask_cache_root}")

            # Persist interleaved slab results so failed runs can resume from
            # the last completed slab on restart.
            channel_tag = str(channel).replace(os.sep, "_")
            checkpoint_path = os.path.join(
                tmp_root,
                f"{source_base}_cellpose_interleaved_ch{channel_tag}.zarr",
            )

            slab_shape = tuple(int(s) for s in image.shape[1:])
            checkpoint_shape = (selected_count, *slab_shape)
            z_chunk = max(1, min(16, slab_shape[0]))
            y_chunk = max(1, min(512, slab_shape[1]))
            x_chunk = max(1, min(512, slab_shape[2]))
            checkpoint_chunks = (1, z_chunk, y_chunk, x_chunk)

            run_signature = {
                "source": os.path.abspath(_source_filepath),
                "channel": channel,
                "distributed_blocksize": int(distributed_blocksize),
                "flow_threshold": float(flow_threshold),
                "cellprob_threshold": float(cellprob_threshold),
                "anisotropy": (
                    None if anisotropy is None else float(anisotropy)
                ),
                "flow3D_smooth": int(flow3D_smooth),
                "tile_norm_blocksize": int(tile_norm_blocksize),
                "batch_size": int(batch_size),
                "diameter": float(diameter),
                "convpaint_model_path": convpaint_model_path,
                "convpaint_image_downsample": int(
                    convpaint_image_downsample
                ),
                "convpaint_background_label": int(
                    convpaint_background_label
                ),
                "convpaint_mask_dilation": int(convpaint_mask_dilation),
                "convpaint_min_object_fraction_of_median": float(
                    convpaint_min_object_fraction_of_median
                ),
                "clip_final_labels_to_convpaint_mask": bool(
                    clip_final_labels_to_convpaint_mask
                ),
                # Increment when mask gating behavior changes so stale
                # checkpoints are not silently reused.
                "mask_strategy_version": 6,
                "sparse_mask_fraction_threshold": float(
                    sparse_mask_fraction_threshold
                ),
                "timepoint_start": int(timepoint_start),
                "timepoint_end": int(timepoint_end),
                "timepoint_step": int(timepoint_step),
                "selected_timepoints": [int(v) for v in selected_timepoints],
            }
            run_signature_json = json.dumps(run_signature, sort_keys=True)

            # Per-slab zarr cache: each completed timepoint is stored as an
            # individual zarr file so results survive process restarts and can
            # be inspected without loading the whole checkpoint array.
            slab_output_cache_key = hashlib.sha1(
                run_signature_json.encode("utf-8")
            ).hexdigest()[:16]
            slab_output_root = os.path.join(
                tmp_root,
                "cellpose_timepoint_cache",
                f"{source_base}_interleaved_ch{channel_tag}_{slab_output_cache_key}",
            )
            os.makedirs(slab_output_root, exist_ok=True)
            print(f"Interleaved slab output cache: {slab_output_root}")

            if os.path.exists(checkpoint_path):
                try:
                    existing = zarr.open_array(checkpoint_path, mode="r")
                    existing_sig = existing.attrs.get("run_signature", "")
                    if (
                        tuple(existing.shape) != checkpoint_shape
                        or existing_sig != run_signature_json
                    ):
                        print(
                            "Interleaved checkpoint exists but is incompatible; "
                            "starting a new checkpoint."
                        )
                        shutil.rmtree(checkpoint_path, ignore_errors=True)
                except Exception as exc:
                    print(
                        "Warning: could not inspect existing interleaved "
                        f"checkpoint ({exc}); recreating it."
                    )
                    shutil.rmtree(checkpoint_path, ignore_errors=True)

            try:
                # Prefer v2 metadata where supported; zarr v3 may ignore or
                # reject v2-only kwargs like zarr_format.
                checkpoint_open_kwargs = {
                    "mode": "a",
                    "shape": checkpoint_shape,
                    "chunks": checkpoint_chunks,
                    "dtype": np.uint32,
                }
                try:
                    zarr_major = int(str(getattr(zarr, "__version__", "2")).split(".")[0])
                except Exception:
                    zarr_major = 2
                if zarr_major < 3:
                    checkpoint_open_kwargs["zarr_format"] = 2
                    checkpoint_open_kwargs["dimension_separator"] = "/"

                checkpoint = zarr.open_array(
                    checkpoint_path,
                    **checkpoint_open_kwargs,
                )
            except TypeError:
                # Older/newer zarr APIs may not accept zarr_format or
                # dimension_separator in this call signature.
                checkpoint = zarr.open_array(
                    checkpoint_path,
                    mode="a",
                    shape=checkpoint_shape,
                    chunks=checkpoint_chunks,
                    dtype=np.uint32,
                )
            except ValueError as exc:
                # zarr v3 raises when dimension_separator is provided without
                # V2 format; retry with default chunk key encoding.
                if "dimension_separator" in str(exc):
                    checkpoint = zarr.open_array(
                        checkpoint_path,
                        mode="a",
                        shape=checkpoint_shape,
                        chunks=checkpoint_chunks,
                        dtype=np.uint32,
                    )
                else:
                    raise
            checkpoint.attrs["run_signature"] = run_signature_json
            completed_slabs = int(checkpoint.attrs.get("completed_slabs", 0))
            completed_slabs = max(0, min(completed_slabs, selected_count))

            if completed_slabs > 0:
                print(
                    "Resuming interleaved run from checkpoint: "
                    f"{completed_slabs}/{selected_count} slabs already completed"
                )
            else:
                print(
                    "Starting new interleaved checkpoint at: "
                    f"{checkpoint_path}"
                )

            is_3d = "Z" in dim_order
            slab_perf_records = []
            current_distributed_blocksize = int(distributed_blocksize)
            tuning_history = []
            safe_non_distributed_max_voxels = 80_000_000
            memory_safe_blocksize_cap = 768
            for out_idx in range(completed_slabs, selected_count):
                t = int(selected_timepoints[out_idx])
                cached_mask_path = os.path.join(
                    mask_cache_root, f"t{t:04d}_mask.tif"
                )
                cached_output_zarr = os.path.join(
                    slab_output_root, f"t{t:04d}_labels.zarr"
                )
                slab_start = time.perf_counter()
                slab_shape = tuple(int(s) for s in image.shape[1:])
                slab_voxels = int(np.prod(slab_shape, dtype=np.int64))

                slab_mask = None
                loaded_mask_from_cache = False
                slab_mask_path = cached_mask_path
                if os.path.exists(cached_mask_path):
                    try:
                        slab_mask = tifffile.imread(cached_mask_path).astype(
                            np.uint8
                        )
                        expected_shape = tuple(int(s) for s in image.shape[1:])
                        if tuple(slab_mask.shape) != expected_shape:
                            print(
                                f"Interleaved slab {out_idx+1}/{selected_count} "
                                f"(source t={t}): "
                                "cached mask shape mismatch, regenerating"
                            )
                            slab_mask = None
                        else:
                            print(
                                f"\nInterleaved slab {out_idx+1}/{selected_count} "
                                f"(source t={t}): "
                                "reusing cached ConvPaint mask"
                            )
                            loaded_mask_from_cache = True
                    except Exception as exc:
                        print(
                            f"Interleaved slab {out_idx+1}/{selected_count} "
                            f"(source t={t}): "
                            f"failed to load cached mask ({exc}), regenerating"
                        )
                        slab_mask = None

                if slab_mask is None:
                    print(
                        f"\nInterleaved slab {out_idx+1}/{selected_count} "
                        f"(source t={t}): "
                        "generating ConvPaint mask"
                    )
                    convpaint_start = time.perf_counter()
                    slab = np.asarray(image[t])
                    slab_labels = convpaint_predict(
                        slab,
                        channel="all",
                        model_path=convpaint_model_path,
                        image_downsample=convpaint_image_downsample,
                        output_type="semantic",
                        background_label=convpaint_background_label,
                        use_cpu=convpaint_use_cpu,
                        force_dedicated_env=convpaint_force_dedicated_env,
                        z_batch_size=convpaint_z_batch_size,
                        tmp_dir=tmp_root,
                    )
                    slab_mask = _labels_to_mask(slab_labels).astype(np.uint8)
                    tifffile.imwrite(cached_mask_path, slab_mask)
                    convpaint_elapsed = time.perf_counter() - convpaint_start
                    print(
                        f"Interleaved slab {out_idx+1}/{selected_count} "
                        f"(source t={t}): "
                        f"cached ConvPaint mask -> {cached_mask_path}"
                    )
                else:
                    convpaint_elapsed = 0.0

                foreground_fraction, fine_active_blocks, fine_total_blocks = _summarize_mask(
                    slab_mask,
                    current_distributed_blocksize,
                )
                if (
                    fine_active_blocks is not None
                    and fine_total_blocks is not None
                ):
                    print(
                        f"Interleaved slab {out_idx+1}/{selected_count} "
                        f"(source t={t}): "
                        f"fine_mask_foreground_fraction={foreground_fraction:.4f}, "
                        f"fine_mask_active_blocks={fine_active_blocks}/{fine_total_blocks}"
                    )
                else:
                    print(
                        f"Interleaved slab {out_idx+1}/{selected_count} "
                        f"(source t={t}): "
                        f"fine_mask_foreground_fraction={foreground_fraction:.4f}"
                    )

                use_slab_mask = True
                run_distributed_for_slab = True
                runtime_slab_mask = slab_mask
                runtime_active_blocks = fine_active_blocks
                runtime_total_blocks = fine_total_blocks
                runtime_fg = foreground_fraction
                if foreground_fraction < sparse_mask_fraction_threshold:
                    print(
                        f"Warning: ConvPaint mask is very sparse "
                        f"(fg={foreground_fraction:.6f} < {sparse_mask_fraction_threshold}). "
                        "Disabling foreground-based block pruning for this slab "
                        "to avoid missing neighboring regions."
                    )
                    use_slab_mask = False
                elif (
                    use_distributed_segmentation
                    and slab_mask is not None
                    and slab_mask.ndim == 3
                ):
                    (
                        runtime_slab_mask,
                        slab_mask_path,
                        runtime_fg,
                        runtime_active_blocks,
                        runtime_total_blocks,
                    ) = _prepare_runtime_distributed_mask(
                        slab_mask,
                        cached_mask_path,
                        mask_cache_root,
                        t,
                        current_distributed_blocksize,
                    )
                    if loaded_mask_from_cache:
                        if (
                            runtime_active_blocks is not None
                            and runtime_total_blocks is not None
                        ):
                            print(
                                f"Interleaved slab {out_idx+1}/{selected_count} "
                                f"(source t={t}): "
                                "runtime gating mask, "
                                f"foreground_fraction={runtime_fg:.4f}, "
                                f"active_blocks={runtime_active_blocks}/{runtime_total_blocks}"
                            )
                        else:
                            print(
                                f"Interleaved slab {out_idx+1}/{selected_count} "
                                f"(source t={t}): "
                                "runtime gating mask, "
                                f"foreground_fraction={runtime_fg:.4f}"
                            )

                if (
                    use_slab_mask
                    and runtime_active_blocks is not None
                    and runtime_total_blocks is not None
                ):
                    active_ratio = float(runtime_active_blocks) / float(
                        runtime_total_blocks
                    )
                    # If runtime gating does not reduce distributed work, only
                    # switch to non-distributed mode when the slab itself is
                    # small enough to be safe in limited-RAM environments.
                    if runtime_total_blocks <= 1 or active_ratio >= 0.95:
                        use_slab_mask = False
                        if slab_voxels <= safe_non_distributed_max_voxels:
                            run_distributed_for_slab = False
                            print(
                                f"Interleaved slab {out_idx+1}/{selected_count} "
                                f"(source t={t}): "
                                "auto-optimization -> using non-distributed Cellpose "
                                f"(runtime_active_blocks={runtime_active_blocks}/{runtime_total_blocks}, "
                                f"active_ratio={active_ratio:.3f}, "
                                f"slab_voxels={slab_voxels})"
                            )
                        else:
                            current_distributed_blocksize = min(
                                current_distributed_blocksize,
                                memory_safe_blocksize_cap,
                            )
                            print(
                                f"Interleaved slab {out_idx+1}/{selected_count} "
                                f"(source t={t}): "
                                "runtime mask pruning ineffective, but slab is large; "
                                "keeping distributed mode for RAM safety "
                                f"(runtime_active_blocks={runtime_active_blocks}/{runtime_total_blocks}, "
                                f"active_ratio={active_ratio:.3f}, "
                                f"slab_voxels={slab_voxels}, "
                                f"blocksize={current_distributed_blocksize})"
                            )

                args_t = {
                    "zarr_path": _source_filepath,
                    "zarr_key": None,
                    "timepoint_index": t,
                    "channel": channel,
                    "channels": channels,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "flow3D_smooth": flow3D_smooth,
                    "anisotropy": anisotropy,
                    "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
                    "batch_size": batch_size,
                    "diameter": diameter,
                    "use_gpu": True,
                    "do_3D": is_3d,
                    "z_axis": 0 if is_3d else None,
                    "channel_axis": None,
                    "use_distributed_segmentation": run_distributed_for_slab,
                    "distributed_blocksize": current_distributed_blocksize,
                    "distributed_mask_path": (
                        slab_mask_path if use_slab_mask else None
                    ),
                    "distributed_mask_zarr_path": None,
                }

                print(
                    f"Interleaved slab {out_idx+1}/{selected_count} "
                    f"(source t={t}): "
                    f"running {'distributed' if run_distributed_for_slab else 'non-distributed'} Cellpose"
                )
                if os.path.exists(cached_output_zarr):
                    print(
                        f"Interleaved slab {out_idx+1}/{selected_count} "
                        f"(source t={t}): "
                        f"reusing persisted slab output -> {cached_output_zarr}"
                    )
                    cellpose_start = time.perf_counter()
                    slab_result = np.array(zarr.open(cached_output_zarr, mode="r"))
                    cellpose_mode = "reused-cached-output"
                else:
                    args_t["persist_output_zarr_path"] = cached_output_zarr
                    try:
                        cellpose_start = time.perf_counter()
                        slab_result = run_cellpose_in_env("eval", args_t)
                        cellpose_mode = (
                            "distributed"
                            if run_distributed_for_slab
                            else "non-distributed"
                        )
                    except RuntimeError as exc:
                        if not args_t.get("use_distributed_segmentation", False):
                            raise

                        print(
                            f"Warning: distributed Cellpose failed for "
                            f"slab {out_idx+1}/{selected_count} "
                            f"(source t={t}) ({exc}). "
                            "Retrying this slab in non-distributed mode."
                        )
                        args_t_fallback = dict(args_t)
                        args_t_fallback["use_distributed_segmentation"] = False
                        args_t_fallback["distributed_mask_path"] = None
                        args_t_fallback["distributed_mask_zarr_path"] = None
                        cellpose_start = time.perf_counter()
                        slab_result = run_cellpose_in_env(
                            "eval", args_t_fallback
                        )
                        cellpose_mode = "fallback-non-distributed"
                cellpose_elapsed = time.perf_counter() - cellpose_start

                if slab_result.dtype != np.uint32:
                    slab_result = slab_result.astype(np.uint32)
                if (
                    clip_final_labels_to_convpaint_mask
                    and slab_mask is not None
                    and slab_mask.shape == slab_result.shape
                ):
                    slab_result = np.where(slab_mask > 0, slab_result, 0).astype(
                        np.uint32, copy=False
                    )
                checkpoint[out_idx] = slab_result
                checkpoint.attrs["completed_slabs"] = out_idx + 1

                slab_elapsed = time.perf_counter() - slab_start
                slab_record = {
                    "selected_index": int(out_idx),
                    "source_timepoint": int(t),
                    "mode": cellpose_mode,
                    "distributed_blocksize": int(current_distributed_blocksize),
                    "convpaint_sec": float(convpaint_elapsed),
                    "cellpose_sec": float(cellpose_elapsed),
                    "total_sec": float(slab_elapsed),
                    "foreground_fraction": float(foreground_fraction),
                    "fine_active_blocks": (
                        None
                        if fine_active_blocks is None
                        else int(fine_active_blocks)
                    ),
                    "fine_total_blocks": (
                        None if fine_total_blocks is None else int(fine_total_blocks)
                    ),
                    "runtime_foreground_fraction": float(runtime_fg),
                    "active_blocks": (
                        None
                        if runtime_active_blocks is None
                        else int(runtime_active_blocks)
                    ),
                    "total_blocks": (
                        None
                        if runtime_total_blocks is None
                        else int(runtime_total_blocks)
                    ),
                }
                slab_perf_records.append(slab_record)
                print(
                    f"Interleaved slab {out_idx+1}/{selected_count} "
                    f"(source t={t}) timing: "
                    f"convpaint={convpaint_elapsed:.1f}s, "
                    f"cellpose={cellpose_elapsed:.1f}s, "
                    f"total={slab_elapsed:.1f}s, mode={cellpose_mode}, "
                    f"blocksize={current_distributed_blocksize}"
                )

                # Auto-tune distributed blocksize after a few distributed samples.
                if (
                    cellpose_mode == "distributed"
                    and runtime_active_blocks is not None
                    and runtime_total_blocks is not None
                    and runtime_total_blocks > 0
                ):
                    tuning_history.append(
                        {
                            "cellpose_sec": float(cellpose_elapsed),
                            "active_ratio": float(runtime_active_blocks)
                            / float(runtime_total_blocks),
                            "blocksize": int(current_distributed_blocksize),
                        }
                    )

                    if len(tuning_history) >= 3:
                        recent = tuning_history[-3:]
                        avg_cellpose_recent = sum(r["cellpose_sec"] for r in recent) / 3.0
                        avg_active_ratio_recent = sum(r["active_ratio"] for r in recent) / 3.0

                        tuned_blocksize = int(current_distributed_blocksize)

                        # High active ratio means little pruning benefit; favor smaller tiles.
                        if (
                            avg_active_ratio_recent >= 0.85
                            and avg_cellpose_recent >= 90.0
                            and tuned_blocksize > 512
                        ):
                            tuned_blocksize = max(512, int(round(tuned_blocksize * 0.75)))
                        # Strong pruning + slow runtime can benefit from moderately larger tiles.
                        elif (
                            avg_active_ratio_recent <= 0.35
                            and avg_cellpose_recent >= 90.0
                            and tuned_blocksize < 1024
                        ):
                            tuned_blocksize = min(1024, int(round(tuned_blocksize * 1.25)))

                        # Keep blocksize aligned to multiples of 32 for stability.
                        tuned_blocksize = max(256, (tuned_blocksize // 32) * 32)

                        if tuned_blocksize != current_distributed_blocksize:
                            print(
                                "Auto-tuner: updating distributed blocksize "
                                f"{current_distributed_blocksize} -> {tuned_blocksize} "
                                f"(avg_cellpose={avg_cellpose_recent:.1f}s, "
                                f"avg_active_ratio={avg_active_ratio_recent:.3f})"
                            )
                            current_distributed_blocksize = tuned_blocksize

            if slab_perf_records:
                total_convpaint = sum(r["convpaint_sec"] for r in slab_perf_records)
                total_cellpose = sum(r["cellpose_sec"] for r in slab_perf_records)
                total_elapsed = sum(r["total_sec"] for r in slab_perf_records)
                distributed_count = sum(
                    1 for r in slab_perf_records if r["mode"] == "distributed"
                )
                non_dist_count = sum(
                    1 for r in slab_perf_records if r["mode"] == "non-distributed"
                )
                fallback_count = sum(
                    1
                    for r in slab_perf_records
                    if r["mode"] == "fallback-non-distributed"
                )
                checkpoint.attrs["slab_perf_records"] = json.dumps(
                    slab_perf_records
                )
                checkpoint.attrs["slab_perf_summary"] = json.dumps(
                    {
                        "processed_slabs_this_run": len(slab_perf_records),
                        "convpaint_total_sec": total_convpaint,
                        "cellpose_total_sec": total_cellpose,
                        "total_sec": total_elapsed,
                        "distributed_count": distributed_count,
                        "non_distributed_count": non_dist_count,
                        "fallback_non_distributed_count": fallback_count,
                        "final_distributed_blocksize": int(current_distributed_blocksize),
                    }
                )
                print(
                    "Interleaved run timing summary: "
                    f"slabs={len(slab_perf_records)}, "
                    f"convpaint={total_convpaint:.1f}s, "
                    f"cellpose={total_cellpose:.1f}s, "
                    f"total={total_elapsed:.1f}s, "
                    f"modes(distributed/non-distributed/fallback)="
                    f"{distributed_count}/{non_dist_count}/{fallback_count}, "
                    f"final_blocksize={current_distributed_blocksize}"
                )

            direct_output = _write_interleaved_checkpoint_output(
                checkpoint, checkpoint_path
            )
            if direct_output:
                return direct_output

            result = np.asarray(checkpoint)
            print(
                "Interleaved distributed Cellpose complete. "
                f"Output shape: {result.shape}, max label: {int(np.max(result))}"
            )
            return result

        if use_convpaint_auto_mask:
            if convpaint_mask is not None:
                source_parent = os.path.dirname(os.path.abspath(_source_filepath))
                tmp_root = os.path.join(source_parent, "tmp")
                os.makedirs(tmp_root, exist_ok=True)
                with NamedTemporaryFile(
                    suffix="_convpaint_mask.tif",
                    delete=False,
                    dir=tmp_root,
                ) as mask_file:
                    generated_mask_path = mask_file.name
                tifffile.imwrite(generated_mask_path, convpaint_mask.astype(np.uint8))
                print(
                    "Saved ConvPaint auto-mask for distributed processing: "
                    f"{generated_mask_path}"
                )
        if use_distributed_segmentation:
            print(
                "Distributed segmentation requested: enabled "
                f"(blocksize={distributed_blocksize})"
            )
            print(
                "Status: preparing blockwise Cellpose processing. "
                "Detailed block progress will appear from the dedicated environment."
            )
        # Prepare arguments for direct zarr processing
        is_3d = "Z" in dim_order
        args = {
            "zarr_path": _source_filepath,
            "zarr_key": None,  # Could be enhanced to support specific keys
            "channel": channel,  # Pass channel selection for filtering
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "flow3D_smooth": flow3D_smooth,
            "anisotropy": anisotropy,
            "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
            "batch_size": batch_size,
            "diameter": diameter,
            "use_gpu": True,  # Let cellpose environment detect GPU
            "do_3D": is_3d,
            "z_axis": 0 if is_3d else None,
            "channel_axis": None,  # No channel axis for single-channel grayscale images
            "use_distributed_segmentation": use_distributed_segmentation,
            "distributed_blocksize": distributed_blocksize,
            "distributed_mask_path": generated_mask_path,
            "distributed_mask_zarr_path": generated_mask_zarr_path,
        }

        selected_timepoints = None
        t_axis = None
        timepoint_cache_root = None
        use_timepoint_cache = False

        if "T" in dim_order:
            t_axis = dim_order.index("T")
            if t_axis >= image.ndim:
                t_axis = 0

            total_timepoints = int(image.shape[t_axis])
            selected_timepoints = _resolve_timepoint_indices(total_timepoints)
            use_timepoint_cache = True

            source_parent = os.path.dirname(os.path.abspath(_source_filepath))
            tmp_root = os.path.join(source_parent, "tmp")
            os.makedirs(tmp_root, exist_ok=True)
            source_base = os.path.splitext(os.path.basename(_source_filepath))[0]

            # Cache key intentionally excludes selected_timepoints so different
            # intervals can reuse already completed per-timepoint outputs.
            tp_cache_signature = {
                "source": os.path.abspath(_source_filepath),
                "source_mtime": os.path.getmtime(_source_filepath),
                "channel": channel,
                "flow_threshold": float(flow_threshold),
                "cellprob_threshold": float(cellprob_threshold),
                "flow3D_smooth": int(flow3D_smooth),
                "anisotropy": None if anisotropy is None else float(anisotropy),
                "tile_norm_blocksize": int(tile_norm_blocksize),
                "batch_size": int(batch_size),
                "diameter": float(diameter),
                "use_distributed_segmentation": bool(use_distributed_segmentation),
                "distributed_blocksize": int(distributed_blocksize),
                "use_convpaint_auto_mask": bool(use_convpaint_auto_mask),
                "convpaint_model_path": convpaint_model_path,
                "convpaint_image_downsample": int(convpaint_image_downsample),
                "convpaint_background_label": int(convpaint_background_label),
                "convpaint_mask_dilation": int(convpaint_mask_dilation),
                "convpaint_min_object_fraction_of_median": float(
                    convpaint_min_object_fraction_of_median
                ),
                "convpaint_use_cpu": bool(convpaint_use_cpu),
                "convpaint_force_dedicated_env": bool(convpaint_force_dedicated_env),
                "convpaint_z_batch_size": int(convpaint_z_batch_size),
            }
            tp_cache_key = hashlib.sha1(
                json.dumps(tp_cache_signature, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            timepoint_cache_root = os.path.join(
                tmp_root,
                "cellpose_timepoint_cache",
                f"{source_base}_{tp_cache_key}",
            )
            os.makedirs(timepoint_cache_root, exist_ok=True)

            print(
                "Timepoint interval enabled for zarr-direct Cellpose: "
                f"selected {len(selected_timepoints)} / {total_timepoints} "
                f"timepoints (start={selected_timepoints[0]}, "
                f"end={selected_timepoints[-1]}, step={int(timepoint_step)})"
            )
            print(f"Per-timepoint cache: {timepoint_cache_root}")

        if use_distributed_segmentation and not is_3d:
            print(
                "Distributed segmentation was requested, but dim_order has no Z axis. "
                "Falling back to standard Cellpose evaluation."
            )
        elif use_distributed_segmentation:
            print(
                "Distributed segmentation is eligible for this zarr input."
            )
    else:
        if use_distributed_segmentation:
            print(
                "Distributed segmentation was requested but no zarr source is available. "
                "Falling back to standard Cellpose evaluation."
            )
        # Prepare arguments for the Cellpose function (legacy numpy array mode)
        is_3d = "Z" in dim_order
        args = {
            "image": image,
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "flow3D_smooth": flow3D_smooth,
            "anisotropy": anisotropy,
            "normalize": {"tile_norm_blocksize": tile_norm_blocksize},
            "batch_size": batch_size,
            "diameter": diameter,
            "use_gpu": True,  # Let cellpose environment detect GPU
            "do_3D": is_3d,
            "z_axis": 0 if is_3d else None,
            "channel_axis": None,  # No channel axis for single-channel grayscale images
        }
    # Run Cellpose in the dedicated environment
    print("Running Cellpose model in dedicated environment...")
    try:
        if use_zarr_direct and use_timepoint_cache and selected_timepoints is not None:
            slab_results = []
            selected_count = len(selected_timepoints)
            existing_outputs = 0
            for t in selected_timepoints:
                tp_existing = os.path.join(
                    timepoint_cache_root,
                    f"t{int(t):04d}_labels.tif",
                )
                if os.path.exists(tp_existing):
                    existing_outputs += 1

            skipped_existing = 0
            newly_processed = 0
            for idx, t in enumerate(selected_timepoints):
                t = int(t)
                tp_output_path = os.path.join(
                    timepoint_cache_root,
                    f"t{t:04d}_labels.tif",
                )

                completed = skipped_existing + newly_processed
                remaining = selected_count - completed
                print(
                    "Timepoint progress: "
                    f"completed={completed}/{selected_count}, "
                    f"skipped_existing={skipped_existing}, "
                    f"newly_processed={newly_processed}, "
                    f"remaining={remaining}, "
                    f"preexisting_total={existing_outputs}"
                )

                if os.path.exists(tp_output_path):
                    print(
                        f"Using existing timepoint output {idx+1}/{len(selected_timepoints)} "
                        f"(source t={t})"
                    )
                    slab_result = tifffile.imread(tp_output_path)
                    skipped_existing += 1
                else:
                    print(
                        f"Running selected timepoint {idx+1}/{len(selected_timepoints)} "
                        f"(source t={t})"
                    )
                    args_t = dict(args)
                    args_t["timepoint_index"] = t
                    slab_result = run_cellpose_in_env("eval", args_t)

                    if slab_result.dtype != np.uint32:
                        slab_result = slab_result.astype(np.uint32)
                    tifffile.imwrite(tp_output_path, slab_result)
                    newly_processed += 1

                if slab_result.dtype != np.uint32:
                    slab_result = slab_result.astype(np.uint32)
                slab_results.append(slab_result)

            print(
                "Timepoint progress: "
                f"completed={selected_count}/{selected_count}, "
                f"skipped_existing={skipped_existing}, "
                f"newly_processed={newly_processed}, "
                "remaining=0"
            )

            result = np.stack(slab_results, axis=0)
        else:
            result = run_cellpose_in_env("eval", args)
    finally:
        if generated_mask_path:
            with suppress(OSError, FileNotFoundError):
                os.unlink(generated_mask_path)
        if generated_mask_zarr_path:
            shutil.rmtree(generated_mask_zarr_path, ignore_errors=True)
    print(f"Segmentation complete. Found {np.max(result)} objects.")

    # Ensure result is uint32 for proper label detection in napari
    # Cellpose typically returns int32, but uint32 is preferred for labels
    if result.dtype != np.uint32:
        result = result.astype(np.uint32)

    return result


# Update cellpose_env_manager.py to install Cellpose 4
def update_cellpose_env_manager():
    """
    Update the cellpose_env_manager to install Cellpose 4
    """
    # This function can be called to update the environment manager code
    # For example, by modifying the pip install command to install the latest version
    # or specify Cellpose 4 explicitly


# Cellpose distributed mode launches internal workers and can be very memory-heavy.
# Keep the outer batch scheduler single-file-at-a-time for stability.
cellpose_segmentation.thread_safe = False

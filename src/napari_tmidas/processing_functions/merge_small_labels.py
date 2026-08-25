# processing_functions/merge_small_labels.py
"""
Processing function to merge small (fragmented) labels into touching neighbors.

Only labels whose voxel count is below a user-defined size threshold are
merged; large labels are left untouched.  Each small label is reassigned the
ID of its largest touching neighbor.  If a small label has no touching
neighbor it is removed (set to 0).
"""

import os

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry


def _merge_single_frame(
    frame: np.ndarray, min_size: int, copy: bool = True
) -> np.ndarray:
    """Merge small labels within a single 2-D or 3-D label frame.

    This is the core implementation.  The public ``merge_small_labels``
    function dispatches here after stripping any leading T dimension(s).

    ``copy=False`` merges in place and returns ``frame`` itself.  The
    streaming path owns its buffer and uses this to avoid a second
    frame-sized allocation; callers passing a user's array must not.
    """
    from scipy.ndimage import binary_dilation, find_objects

    result = frame.copy() if copy else frame
    original_dtype = frame.dtype
    ndim = result.ndim

    # Full connectivity kernel (8-connected 2-D / 26-connected 3-D)
    struct = np.ones((3,) * ndim, dtype=bool)

    for _ in range(50):  # generous cap; 1–2 passes suffice in practice
        max_label = int(result.max())
        if max_label == 0:
            break

        # np.bincount is O(N + max_label) with no sort — faster than np.unique.
        # It upcasts its input to int64 internally, though, so counting the
        # whole volume at once transiently doubles a uint32 stack (3.4 GB on
        # top of a 1.7 GB timepoint).  Counting slice by slice caps that
        # intermediate at one plane; the totals are identical.
        sizes = np.zeros(max_label + 1, dtype=np.int64)
        for sl in result if result.ndim > 2 else (result,):
            sizes += np.bincount(sl.ravel(), minlength=max_label + 1)

        # Boolean lookup table: is_small[label_id] → True/False.
        # Used for O(1)-per-element filtering instead of np.isin O(K).
        is_small = np.zeros(max_label + 1, dtype=bool)
        is_small[1:] = (sizes[1:] > 0) & (sizes[1:] < min_size)
        small_ids = np.nonzero(is_small)[0]

        if len(small_ids) == 0:
            break

        # find_objects: one O(N) C-level pass → tight bbox for every label.
        # Replaces per-label np.where(result == sid) which was O(N × K).
        bboxes = find_objects(result)

        changed = False
        for sid in small_ids.tolist():
            bbox = bboxes[sid - 1]  # find_objects: label k → index k-1
            if bbox is None:
                continue  # already absorbed during this pass

            # Expand bbox by one voxel in every direction, clamped to bounds
            expanded = tuple(
                slice(max(0, sl.start - 1), min(result.shape[i], sl.stop + 1))
                for i, sl in enumerate(bbox)
            )
            sub = result[expanded]  # view — no copy

            local_mask = sub == sid
            dilated = binary_dilation(local_mask, structure=struct)
            border_ids = sub[dilated & ~local_mask & (sub != 0)]

            if border_ids.size == 0:
                result[expanded][local_mask] = 0
                changed = True
                continue

            # Boolean-array lookup: O(1) per element vs O(K) for np.isin
            large_border = border_ids[~is_small[border_ids]]
            candidates = large_border if large_border.size > 0 else border_ids

            unique_n, contact_counts = np.unique(candidates, return_counts=True)
            result[expanded][local_mask] = int(unique_n[np.argmax(contact_counts)])
            changed = True

        if not changed:
            break

    # ``result`` only ever receives scalar assignments, so its dtype already
    # is ``original_dtype``; copy=False keeps this from being an unconditional
    # frame-sized memcpy (1.7 GB per timepoint on a real tracked stack).
    return result.astype(original_dtype, copy=False)


def _block_ndim(ndim: int, dim_hint: str) -> int:
    """Trailing axes that form one independently-merged block.

    Mirrors the dispatch in ``merge_small_labels``: 2-D frames for 2-D input
    and for 3-D input explicitly marked TYX, otherwise 3-D volumes, with any
    remaining leading axes (T, or T and C) iterated over.
    """
    if ndim <= 2:
        return 2
    if ndim == 3 and dim_hint == "TYX":
        return 2
    return 3


def _stream_merge_small_labels(
    source_path, output_path, min_size: int, dim_order: str = "ZYX"
) -> str:
    """
    Merge small labels one block at a time, never holding the whole stack.

    Merging is inherently spatial — a label needs its touching neighbors — so
    unlike a pure per-label filter this cannot go plane by plane.  It can go
    *block* by block: each timepoint (and channel) is already independent, so
    only one 3-D volume is ever resident.  On a 52 GB tracked stack that is
    1.7 GB instead of 52 GB in, 52 GB out.

    The volume buffer is allocated once and reused across blocks, and merging
    runs in place inside it.

    Returns the path written.
    """
    from napari_tmidas.processing_functions.intensity_label_filter import (
        _PlaneReader,
    )
    from napari_tmidas.processing_functions.ome_output_utils import (
        stream_planes_to_tiff,
    )

    dim_hint = str(dim_order).upper()
    if dim_hint == "AUTO":
        dim_hint = "ZYX"

    with _PlaneReader(source_path) as labels:
        shape = labels.shape
        ndim = len(shape)
        if ndim < 2:
            raise ValueError(
                f"{source_path}: expected a 2D+ label image, got shape {shape}"
            )

        block_ndim = _block_ndim(ndim, dim_hint)
        group_shape = shape[:-block_ndim]
        block_shape = shape[-block_ndim:]
        inner_shape = block_shape[:-2]
        n_blocks = int(np.prod(group_shape)) if group_shape else 1
        planes_per_block = int(np.prod(inner_shape)) if inner_shape else 1
        n_planes = n_blocks * planes_per_block

        # One reused buffer for the whole run — this is the peak allocation.
        buffer = np.empty(block_shape, dtype=labels.dtype)
        print(
            f"🔎 Streaming {os.path.basename(str(source_path))} "
            f"{shape} {labels.dtype}: {n_blocks} block(s) of {block_shape}, "
            f"merging labels < {min_size} voxels "
            f"({buffer.nbytes / 1e9:.2f} GB resident at a time)"
        )

        def inner_indices():
            return np.ndindex(*inner_shape) if inner_shape else [()]

        done = 0

        def plane_iterator():
            nonlocal done
            for n_block, group in enumerate(
                np.ndindex(*group_shape) if group_shape else [()], start=1
            ):
                for inner in inner_indices():
                    buffer[inner] = labels.plane(tuple(group) + inner)

                merged = _merge_single_frame(buffer, min_size, copy=False)

                for inner in inner_indices():
                    # Copy: the buffer is refilled for the next block, and
                    # tifffile may still hold queued chunks for compression.
                    yield np.ascontiguousarray(merged[inner]).copy()
                    done += 1
                print(
                    f"   block {n_block}/{n_blocks} "
                    f"({done}/{n_planes} planes)",
                    flush=True,
                )

        axes = {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}.get(ndim)
        if ndim == 3 and block_ndim == 2:
            axes = "TYX"
        print(
            f"💾 Writing {os.path.basename(str(output_path))} ({labels.dtype})"
        )
        stream_planes_to_tiff(
            str(output_path),
            plane_iterator(),
            shape,
            labels.dtype,
            metadata={"axes": axes} if axes else None,
            bigtiff=True,
            ome=None,
        )

        print(
            f"✅ {os.path.basename(str(output_path))}: "
            f"{n_blocks} block(s) merged"
        )
    return str(output_path)


@BatchProcessingRegistry.register(
    name="Merge Small Labels to Neighbors",
    suffix="_merged",
    description="Merge fragmented labels smaller than a size threshold into their largest touching neighbor. Labels with no touching neighbor are removed.",
    parameters={
        "min_size": {
            "type": int,
            "default": 100,
            "min": 1,
            "max": 10_000_000,
            "description": "Labels with fewer voxels than this are merged into their largest touching neighbor.",
        },
    },
)
def merge_small_labels(
    label_image: np.ndarray,
    min_size: int = 100,
    dim_order: str = "ZYX",
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = None,
) -> np.ndarray:
    """Merge small labels into their largest touching neighbor.

    When the batch widget supplies a TIFF/Zarr source path and an output
    folder, the stack is streamed one block at a time and the written path is
    returned (``skip_load`` keeps the worker from loading it densely).
    Otherwise the in-memory array path below is used unchanged.
    """
    min_size = int(min_size)

    # --- Streaming path: never materialise the stack ----------------------
    if _source_filepath and _output_folder and _output_suffix:
        suffix = os.path.splitext(_source_filepath)[1].lower()
        if suffix in (".tif", ".tiff", ".zarr") or os.path.isdir(
            _source_filepath
        ):
            os.makedirs(_output_folder, exist_ok=True)
            stem = os.path.splitext(
                os.path.basename(_source_filepath.rstrip("/"))
            )[0]
            output_path = os.path.join(
                _output_folder, f"{stem}{_output_suffix}.tif"
            )
            return _stream_merge_small_labels(
                _source_filepath, output_path, min_size, dim_order
            )

    if label_image is None:
        # skip_load left the array unloaded and the format isn't one we can
        # stream — read it here rather than silently doing nothing.
        from napari_tmidas._file_selector import load_image_file

        label_image = np.asarray(load_image_file(_source_filepath))

    ndim = label_image.ndim
    # Treat "Auto" the same as the default "ZYX" (single 3-D volume)
    dim_hint = str(dim_order).upper()
    if dim_hint == "AUTO":
        dim_hint = "ZYX"

    if ndim <= 2:
        return _merge_single_frame(label_image, min_size)

    if ndim == 3:
        # Ambiguous: ZYX (single 3-D volume) or TYX (2-D time series).
        if dim_hint == "TYX":
            t_size = label_image.shape[0]
            print(f"Merge small labels: processing {t_size} timepoints (TYX)...")
            result = np.empty_like(label_image)
            for t in range(t_size):
                print(f"  T={t + 1}/{t_size}", end="\r", flush=True)
                result[t] = _merge_single_frame(label_image[t], min_size)
            print()
            return result
        # ZYX (default) — single 3-D volume
        return _merge_single_frame(label_image, min_size)

    if ndim == 4:
        # TZYX (or TYX with extra dim) — iterate per timepoint
        t_size = label_image.shape[0]
        print(f"Merge small labels: processing {t_size} timepoints (TZYX)...")
        result = np.empty_like(label_image)
        for t in range(t_size):
            print(f"  T={t + 1}/{t_size}", end="\r", flush=True)
            result[t] = _merge_single_frame(label_image[t], min_size)
        print()
        return result

    # ndim == 5: TCZYX — iterate over T and C independently
    t_size, c_size = label_image.shape[0], label_image.shape[1]
    print(
        f"Merge small labels: processing {t_size} timepoints × "
        f"{c_size} channels (TCZYX)..."
    )
    result = np.empty_like(label_image)
    for t in range(t_size):
        for c in range(c_size):
            print(f"  T={t + 1}/{t_size}  C={c + 1}/{c_size}", end="\r", flush=True)
            result[t, c] = _merge_single_frame(label_image[t, c], min_size)
    print()
    return result


# skip_load=True: the worker must NOT call load_image_file for this function.
# A tracked stack that is 90 MB compressed can be 70 GB dense, and a dense load
# plus the same-sized output allocation is what gets the process OOM-killed.
# The function streams the file itself instead.
merge_small_labels.skip_load = True

# processing_functions/merge_small_labels.py
"""
Processing function to merge small (fragmented) labels into touching neighbors.

Only labels whose voxel count is below a user-defined size threshold are
merged; large labels are left untouched.  Each small label is reassigned the
ID of its largest touching neighbor.  If a small label has no touching
neighbor it is removed (set to 0).
"""

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry


def _merge_single_frame(frame: np.ndarray, min_size: int) -> np.ndarray:
    """Merge small labels within a single 2-D or 3-D label frame.

    This is the core implementation.  The public ``merge_small_labels``
    function dispatches here after stripping any leading T dimension(s).
    """
    from scipy.ndimage import binary_dilation, find_objects

    result = frame.copy()
    original_dtype = frame.dtype
    ndim = result.ndim

    # Full connectivity kernel (8-connected 2-D / 26-connected 3-D)
    struct = np.ones((3,) * ndim, dtype=bool)

    for _ in range(50):  # generous cap; 1–2 passes suffice in practice
        max_label = int(result.max())
        if max_label == 0:
            break

        # np.bincount is O(N + max_label) with no sort — faster than np.unique
        sizes = np.bincount(result.ravel(), minlength=max_label + 1)

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

    return result.astype(original_dtype)


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
) -> np.ndarray:
    """Merge small labels into their largest touching neighbor."""
    min_size = int(min_size)
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

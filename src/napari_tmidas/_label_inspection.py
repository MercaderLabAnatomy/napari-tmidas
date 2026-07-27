"""
Batch Label Inspection for Napari
---------------------------------
This module provides a widget for Napari that allows users to inspect image-label pairs in a folder.
The widget loads image-label pairs from a folder and displays them in the Napari viewer.
Users can make and save changes to the labels, and proceed to the next pair.


"""

import contextlib
import os
import re
import sys
from collections import OrderedDict

import numpy as np

# Lazy imports for optional heavy dependencies
try:
    from magicgui import magicgui

    _HAS_MAGICGUI = True
except ImportError:
    # Create stub decorator
    def magicgui(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    _HAS_MAGICGUI = False

try:
    from napari.layers import Labels
    from napari.viewer import Viewer

    _HAS_NAPARI = True
except ImportError:
    Labels = None
    Viewer = None
    _HAS_NAPARI = False

try:
    from qtpy.QtWidgets import QFileDialog, QMessageBox, QPushButton

    _HAS_QTPY = True
except ImportError:
    QFileDialog = QMessageBox = QPushButton = None
    _HAS_QTPY = False

try:
    from skimage.io import imread

    _HAS_SKIMAGE = True
except ImportError:
    imread = None
    _HAS_SKIMAGE = False

sys.path.append("src/napari_tmidas")


def _is_zarr(path: str) -> bool:
    return path.lower().endswith(".zarr") or (
        os.path.isdir(path) and os.path.exists(os.path.join(path, ".zattrs"))
    )


def _load_image(path: str):
    """Load an image from *path* as a lazy dask array where possible.

    Zarr files are opened via *load_zarr_basic* which returns a dask array
    backed by the on-disk store — no data is read into RAM until napari
    requests a specific slice.

    TIFF files are loaded via the lazy *tiff_reader_function* which builds
    one dask task per IFD page so only the pages currently in view are
    decompressed.
    """
    if _is_zarr(path):
        try:
            from napari_tmidas._file_selector import load_zarr_basic

            return load_zarr_basic(
                path
            )  # already a dask array — do NOT compute()
        except Exception as exc:
            raise OSError(f"Could not load zarr file {path}: {exc}") from exc

    # Lazy TIFF path
    try:
        from napari_tmidas._reader import tiff_reader_function

        results = tiff_reader_function(path)
        if results:
            return results[0][0]  # (data, kwargs, layer_type) → data
    except Exception:
        pass

    # Final fallback: eager skimage read
    if imread is None:
        raise ImportError("scikit-image is required to read non-zarr images.")
    return imread(path)


def _is_dask_array(arr) -> bool:
    """Return True if *arr* is a dask array, without importing dask eagerly."""
    try:
        import dask.array as da

        return isinstance(arr, da.Array)
    except ImportError:
        return False


def _apply_value_map(block, mapping=None):
    """Return *block* with the {old_id: new_id} *mapping* applied.

    All masks derive from the original block, so chained mappings like
    ``{5: 0, 0: 5}`` swap correctly (simultaneous, not sequential).
    """
    out = block.copy()
    for k, v in mapping.items():
        out[block == k] = v
    return out


def _apply_value_map_inplace(arr, mapping):
    """In-place variant of :func:`_apply_value_map` (simultaneous semantics)."""
    masks = [(arr == k) for k in mapping]
    for mask, v in zip(masks, mapping.values()):
        arr[mask] = v


class _DaskFancyIndexWrapper:
    """Wrap a dask label array for efficient interactive editing in napari.

    Edits are stored as *descriptions*, never as materialized copies of the
    data; slices are assembled on demand as ``base → LUT → diffs``:

    * **Value-remap LUT** (:meth:`remap_values`) — global operations such as
      "delete label 5 everywhere" or "merge 7 into 3" are one dict entry.
      All accumulated remaps are baked into ``_arr`` as a *single* lazy
      ``map_blocks`` pass over the base array, so k remaps cost the same as
      one at read/save time and zero memory or I/O when issued.
    * **Sparse diffs** — napari paint / fill / erase edits are recorded per
      T-slice as (coordinates, values), typically kilobytes per stroke.  A
      slice whose accumulated diffs would outgrow a full copy is upgraded to
      a **dense snapshot** (the full edited slice, never evicted).
    * **Multi-entry LRU T-slice cache** — up to ``_CACHE_MAX_SLICES``
      assembled timepoint volumes are kept as numpy arrays so repeated
      reads and edits at the same timepoint skip disk I/O entirely.
    * **nd fancy-indexing** — napari's ray-cast picking calls
      ``data[t_arr, z_arr, y_arr, x_arr]`` simultaneously; dask raises
      ``NotImplementedError`` for this; the wrapper serves it from the
      T-slice cache.
    * **``__array__``** — called by numpy / tifffile for conversion; rebuilds
      the array T-by-T so peak RAM stays at one T-slice.
    """

    # Max number of T-slices to keep in the LRU read cache.
    _CACHE_MAX_SLICES: int = 4

    def __init__(self, arr):
        self._base = arr  # immutable disk-backed dask array
        self._arr = arr  # base with the value-remap LUT baked in (lazy)
        self._lut: dict = {}  # accumulated global remaps {old_id: new_id}
        # Undo log: [(mapping, inverse_patches), ...] — the LUT above is
        # always the composition of the mappings in this log.
        self._op_log: list = []
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.ndim = arr.ndim
        # LRU read cache: OrderedDict {(outer_dim, outer_val) -> numpy_array}
        self._cache: OrderedDict = OrderedDict()
        # Pending local edits per T-slice:
        #   ("sparse", [coords_tuple, ...], [values_array, ...])  — replayed
        #   in order on top of the freshly loaded slice, or
        #   ("dense", full_slice)  — authoritative snapshot, never evicted.
        self._diffs: dict = {}

    # Forward every attribute napari or dask may look up to the wrapped array.
    def __getattr__(self, name):
        return getattr(self._arr, name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_outer_slice(self, outer_dim: int, outer_val: int) -> np.ndarray:
        """Assemble a T-slice as numpy: dense snapshot → cache → dask+diffs."""
        key = (outer_dim, outer_val)
        entry = self._diffs.get(key)
        if entry is not None and entry[0] == "dense":
            return entry[1]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # Load from dask one T-slice at a time (LUT already baked into _arr).
        idx = tuple(
            outer_val if i == outer_dim else slice(None)
            for i in range(self.ndim)
        )
        sliced = self._arr[idx]
        if hasattr(sliced, "compute"):
            sliced = sliced.compute()
        sliced = np.ascontiguousarray(sliced)
        if entry is not None:  # replay sparse edits in order
            for coords, vals in zip(entry[1], entry[2]):
                sliced[coords] = vals
        while len(self._cache) >= self._CACHE_MAX_SLICES:
            self._cache.popitem(last=False)  # evict LRU
        self._cache[key] = sliced
        return sliced

    def _inner_to_coords(self, inner, t_slice):
        """Return point coordinates for the *inner* index, or None when
        enumerating them would cost more than a dense snapshot."""
        # napari paint/fill: tuple of equal-length 1-D int arrays.
        if (
            len(inner) == t_slice.ndim
            and all(
                isinstance(ix, np.ndarray) and ix.ndim == 1 for ix in inner
            )
            and len({ix.size for ix in inner}) == 1
        ):
            return tuple(np.ascontiguousarray(ix) for ix in inner)
        # Mixed fancy + slice indexing has outer-product-vs-broadcast
        # pitfalls; only enumerate pure int/slice tuples.
        if not all(isinstance(ix, (int, np.integer, slice)) for ix in inner):
            return None
        padded = list(inner) + [slice(None)] * (t_slice.ndim - len(inner))
        per_dim = []
        n_total = 1
        for d, ix in enumerate(padded):
            if isinstance(ix, slice):
                r = np.arange(*ix.indices(t_slice.shape[d]))
            else:
                r = np.array([int(ix)])
            n_total *= r.size
            per_dim.append(r)
        if n_total * (t_slice.ndim + 1) * 8 > t_slice.nbytes:
            return None  # coordinate storage would outgrow a snapshot
        grids = np.meshgrid(*per_dim, indexing="ij")
        return tuple(g.ravel() for g in grids)

    def _to_dense(self, key, t_slice):
        """Promote a slice's diff record to an authoritative dense snapshot."""
        self._diffs[key] = ("dense", t_slice)
        self._cache.pop(key, None)  # avoid aliasing with the cache

    def remap_values(self, mapping: dict) -> None:
        """Apply {old_id: new_id} to the entire array (all timepoints).

        Zero I/O and O(1) memory at call time: the mapping is appended to
        the operation log and the composed LUT is baked into ``_arr`` as one
        lazy ``map_blocks`` pass over the base array.  Pending local edits
        and warm cache entries are remapped in place so the viewer reflects
        the change immediately.  Undoable via :meth:`undo_remap`.
        """
        mapping = {
            int(k): int(v) for k, v in mapping.items() if int(k) != int(v)
        }
        if not mapping:
            return
        # Inverse patches: where this mapping changes pending local edits,
        # remember the positions and the pre-remap value (small — only the
        # remapped label's voxels within edited slices).
        patches = []
        for key, entry in self._diffs.items():
            if entry[0] == "dense":
                for k in mapping:
                    coords = np.nonzero(entry[1] == k)
                    if coords[0].size:
                        patches.append((key, "dense", None, coords, k))
            else:
                for i, vals in enumerate(entry[2]):
                    for k in mapping:
                        idx = np.nonzero(vals == k)[0]
                        if idx.size:
                            patches.append((key, "sparse", i, idx, k))
        self._op_log.append((mapping, patches))
        self._rebuild_arr()
        # Keep pending edits and warm cache slices consistent.
        for entry in self._diffs.values():
            if entry[0] == "dense":
                _apply_value_map_inplace(entry[1], mapping)
            else:
                for vals in entry[2]:
                    _apply_value_map_inplace(vals, mapping)
        for cached in self._cache.values():
            _apply_value_map_inplace(cached, mapping)

    def undo_remap(self):
        """Undo the most recent :meth:`remap_values` call.

        Pops the operation log, recomposes the LUT from the remaining ops
        (the base array is untouched, so nothing needs restoring there),
        and reverts the inverse patches on pending local edits.  Returns
        the undone mapping, or None if there is nothing to undo.
        """
        if not self._op_log:
            return None
        mapping, patches = self._op_log.pop()
        self._rebuild_arr()
        for key, kind, i, coords_or_idx, k in patches:
            entry = self._diffs.get(key)
            if entry is None or entry[0] != kind:
                # Diff record restructured since (e.g. dense promotion) —
                # its content already reflects the remap; skip the patch.
                continue
            if kind == "dense":
                entry[1][coords_or_idx] = k
            else:
                entry[2][i][coords_or_idx] = k
        # Cached slices have the mapping baked in and can't be un-applied
        # in place; drop them — they re-assemble from disk on next read.
        self._cache.clear()
        return mapping

    def _rebuild_arr(self):
        """Recompose the LUT from the op log and re-bake the lazy view."""
        import dask.array as da

        lut: dict = {}
        for mapping, _ in self._op_log:
            # Compose: displayed = mapping(lut(base)); base values already
            # remapped by the LUT must not be remapped again from raw keys.
            new = {k: mapping.get(v, v) for k, v in lut.items()}
            for k, v in mapping.items():
                new.setdefault(k, v)
            lut = {k: v for k, v in new.items() if k != v}
        self._lut = lut
        self._arr = (
            da.map_blocks(
                _apply_value_map,
                self._base,
                mapping=dict(self._lut),
                dtype=self._base.dtype,
            )
            if self._lut
            else self._base
        )

    def _find_constant_outer(self, index):
        """Return (dim, val) of the first constant outer dim in *index*."""
        for i, idx in enumerate(index):
            if isinstance(idx, (int, np.integer)):
                return i, int(idx)
            if isinstance(idx, np.ndarray) and idx.ndim == 1:
                unique = np.unique(idx)
                if len(unique) == 1:
                    return i, int(unique[0])
        return None, None

    # ── Public interface ──────────────────────────────────────────────────────

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        # ── Scalar outer dim + trailing slices/scalars → T-slice cache ─────────
        # This covers (T,:,:,:), (T,Z,:,:), (T,Z,Y,X), etc.
        # Always route through _get_outer_slice so pending edits are respected.
        if all(isinstance(i, (int, np.integer, slice)) for i in index):
            for dim, idx in enumerate(index):
                if isinstance(idx, (int, np.integer)):
                    rest = index[dim + 1 :]
                    t_slice = self._get_outer_slice(dim, int(idx))
                    if rest:
                        return t_slice[rest]
                    return t_slice
            # All slices — return a lazy dask result (avoid OOM for large arrays).
            result = self._arr[index]
            if hasattr(result, "compute") and getattr(result, "ndim", 1) == 0:
                return result.compute()
            return result

        # ── Fancy indexing path ───────────────────────────────────────────────
        array_positions = [
            i
            for i, idx in enumerate(index)
            if isinstance(idx, np.ndarray) and idx.ndim == 1
        ]

        if len(array_positions) <= 1:
            result = self._arr[index]
            return result.compute() if hasattr(result, "compute") else result

        # Collapse constant array-dims to scalars.
        new_index = list(index)
        constant_dims: dict = {}
        for i in array_positions:
            unique = np.unique(index[i])
            if len(unique) == 1:
                new_index[i] = int(unique[0])
                constant_dims[i] = int(unique[0])

        # ── Fast path: serve from T-slice cache ───────────────────────────────
        if constant_dims:
            outer_dim = min(constant_dims.keys())
            outer_val = constant_dims[outer_dim]
            t_slice = self._get_outer_slice(outer_dim, outer_val)
            # Keep the non-outer index arrays as arrays (from the original
            # index): collapsing them all to scalars would return a 0-d
            # value where numpy fancy-indexing semantics require shape
            # (n,) — napari's data_setitem relies on this for single-voxel
            # edits.
            reduced = tuple(
                idx for i, idx in enumerate(index) if i != outer_dim
            )
            return t_slice[reduced]

        # ── Fallback A: ≤1 remaining array dim — dask handles natively ────────
        remaining = [
            i
            for i, idx in enumerate(new_index)
            if isinstance(idx, np.ndarray) and idx.ndim == 1
        ]
        if len(remaining) <= 1:
            result = self._arr[tuple(new_index)]
            return result.compute() if hasattr(result, "compute") else result

        # ── Fallback B: point-by-point (multiple genuinely varying dims) ───────
        n = len(new_index[remaining[0]])
        values = []
        for j in range(n):
            pt = tuple(
                (
                    idx[j]
                    if (isinstance(idx, np.ndarray) and idx.ndim == 1)
                    else idx
                )
                for idx in new_index
            )
            val = self._arr[pt]
            if hasattr(val, "compute"):
                val = val.compute()
            values.append(
                int(val)
                if (
                    np.isscalar(val)
                    or (hasattr(val, "ndim") and val.ndim == 0)
                )
                else val
            )
        return np.array(values, dtype=self.dtype)

    def __setitem__(self, index, value):
        """Support napari paint / fill / erase by editing one T-slice at a time.

        Applies the edit to the assembled (cached) T-slice so the viewer sees
        it immediately, and records it as a sparse (coords, values) diff —
        kilobytes per brush stroke.  If a slice's accumulated diffs would
        outgrow a full copy, the slice is promoted to a dense snapshot.
        """
        if not isinstance(index, tuple):
            index = (index,)

        outer_dim, outer_val = self._find_constant_outer(index)

        if outer_dim is None:
            # No constant outer dim — materialize everything (rare; only
            # full-nd edits hit this), bake in, and reset edit state.
            import dask.array as da

            full = np.asarray(self)
            full[index] = value
            self._base = self._arr = da.from_array(full)
            self._lut = {}
            self._op_log.clear()
            self._diffs.clear()
            self._cache.clear()
            return

        key = (outer_dim, outer_val)
        t_slice = self._get_outer_slice(outer_dim, outer_val)
        # Inner index: all dims except the fixed outer dim.
        inner = tuple(idx for i, idx in enumerate(index) if i != outer_dim)
        t_slice[inner] = value  # mutates the cached / dense slice in place

        entry = self._diffs.get(key)
        if entry is not None and entry[0] == "dense":
            return  # the dense snapshot itself is the durable record

        coords = self._inner_to_coords(inner, t_slice)
        if coords is None:
            self._to_dense(key, t_slice)
            return
        # Read the values back from the slice: correct for any broadcasting.
        vals = np.ascontiguousarray(t_slice[coords])
        if entry is None:
            entry = ("sparse", [], [])
            self._diffs[key] = entry
        entry[1].append(coords)
        entry[2].append(vals)
        n_points = sum(v.size for v in entry[2])
        if n_points * (t_slice.ndim + 1) * 8 > t_slice.nbytes:
            self._to_dense(key, t_slice)

    def __array__(self, dtype=None):
        """Reconstruct as numpy T-by-T, applying all modifications.

        Called by ``np.asarray(wrapper)`` during saving.  Peak RAM stays at
        one T-slice rather than the full dataset size.
        """
        outer_dim = 0
        n_outer = self.shape[outer_dim]
        slices = []
        for t in range(n_outer):
            slices.append(self._get_outer_slice(outer_dim, t))
        result = np.stack(slices, axis=outer_dim)
        return result.astype(dtype) if dtype is not None else result


# A changed-region tuple covering nothing (an edit that matched no voxel).
_EMPTY_REGION = (slice(0, 0), slice(0, 0), slice(0, 0))


def _bbox_union(a, b):
    """Smallest region containing both slice-tuples (either may be None)."""
    if a is None:
        return b
    if b is None:
        return a
    return tuple(
        slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(a, b)
    )


class _TrackView:
    """Read-through 3-D "track volume" view over a TZYX/TYX label source.

    Base class for the track-inspection views: every plane is assembled on
    demand from the underlying source (a :class:`_DaskFancyIndexWrapper` or
    a numpy array), so the view never copies the movie and always reflects
    pending remaps and paint edits.  Label IDs are unchanged, so the
    ID-based tools (click-to-delete / click-to-relabel, Ctrl+Z undo, save)
    work identically through the view.

    ``yx_step`` > 1 subsamples every plane ``[::step, ::step]`` so the
    materialized 3-D volume fits in GPU memory for large movies (napari
    uploads it as a single 3-D texture; see :func:`_pick_track_view_step`).
    Nearest-neighbor striding keeps label IDs intact, and the layer scale
    compensates, so world-coordinate clicking still resolves the right
    ID — only painting is disabled (a strided pixel cannot be written
    back losslessly).  The underlying data stays full resolution.
    """

    # Largest label ID for which the per-label bounding-box index is built
    # (find_objects walks 1..max_id, so a sparse ID space is not worth it).
    _BBOX_MAX_ID: int = 2_000_000

    def __init__(self, source, yx_step: int = 1):
        if getattr(source, "ndim", 0) not in (3, 4):
            raise ValueError(
                "Track view needs a 3-D (TYX) or 4-D (TZYX) label source."
            )
        self._source = source
        self.dtype = source.dtype
        self.ndim = 3
        self.n_t = source.shape[0]
        self.yx_step = max(1, int(yx_step))
        # ceil(size / step) per YX axis — what `plane[::step]` yields.
        self._yx_shape = tuple(
            -(-int(s) // self.yx_step) for s in source.shape[-2:]
        )
        # A TYX movie has an implicit Z of size 1.
        self.n_z = source.shape[1] if source.ndim == 4 else 1
        # Materialized full view, built lazily the first time napari's
        # 3-D display asks for more than one plane.  While present, ALL
        # reads (including 3-D click pick-rays, which would otherwise
        # re-read one timepoint from disk per plane they cross) are
        # served from it, and delete/relabel remaps update it in place —
        # so 3-D clicking and editing stay interactive on movies of any
        # size.
        self._vol = None
        # {label_id: (slice, slice, slice)} bounding each label in _vol,
        # so a remap only touches the label's own sub-volume instead of
        # scanning the whole thing.  Built lazily (one find_objects pass)
        # and maintained as a *superset* across edits; None = not built,
        # and _bbox_ok False = don't build (too many IDs to be worth it).
        self._bboxes = None
        self._bbox_ok = True

    def _t_slice(self, t: int) -> np.ndarray:
        """Timepoint *t* of the source as numpy, pending edits included."""
        if isinstance(self._source, _DaskFancyIndexWrapper):
            return self._source._get_outer_slice(0, int(t))
        s = self._source[int(t)]
        if hasattr(s, "compute"):
            s = s.compute()
        return np.asarray(s)

    def _plane(self, i: int) -> np.ndarray:
        raise NotImplementedError

    def _stride(self, plane: np.ndarray) -> np.ndarray:
        """Apply the YX subsampling step to a full-resolution plane."""
        if self.yx_step == 1:
            return plane
        return plane[:: self.yx_step, :: self.yx_step]

    def invalidate(self) -> None:
        """Drop all derived caches after an arbitrary source edit."""
        self._vol = None
        self._bboxes = None

    def _planes_of_t(self, t: int):
        """The view plane indices that show timepoint *t*."""
        raise NotImplementedError

    def refresh_timepoint(self, t: int):
        """Recompute cached planes for timepoint *t* only.

        The cheap alternative to :meth:`invalidate` when a source edit is
        confined to one timepoint: the materialized volume is patched with
        that timepoint's planes (served from the source's warm slice
        cache), so the 3-D display stays interactive.

        Returns the changed region as a tuple of slices into the view (for
        a partial canvas redraw), or None when the whole view may differ.
        """
        if self._vol is None:
            return None
        planes = list(self._planes_of_t(int(t)))
        for i in planes:
            self._vol[i] = self._plane(i)
        # Painted / merged voxels can carry IDs anywhere in these planes.
        self._bboxes = None
        if not planes:
            return _EMPTY_REGION
        return (
            slice(min(planes), max(planes) + 1),
            slice(0, self.shape[1]),
            slice(0, self.shape[2]),
        )

    def _bbox_index(self):
        """``{label_id: region}`` bounding every voxel of each label in the
        materialized volume, or None when no index is available.

        One ``find_objects`` pass builds the whole index; it is then kept
        as a *superset* across edits (a merged label inherits the union of
        both boxes, a shrunk label keeps its old box), which is all the
        remap fast path needs.  Volumes whose largest ID dwarfs the label
        count would need a huge intermediate list, so they opt out and fall
        back to whole-volume passes.
        """
        if self._bboxes is not None:
            return self._bboxes
        vol = self._vol
        if vol is None or not self._bbox_ok:
            return None
        max_id = int(vol.max())
        if max_id <= 0 or max_id > self._BBOX_MAX_ID:
            self._bbox_ok = False
            return None
        from scipy import ndimage as ndi

        self._bboxes = {
            i + 1: sl
            for i, sl in enumerate(ndi.find_objects(vol))
            if sl is not None
        }
        return self._bboxes

    def _grow_bboxes(self, index, value) -> None:
        """Widen the bbox index to cover a paint write into the volume.

        Only the written IDs' boxes grow, and only ever outward, so the
        index stays the superset the remap fast path relies on.  Anything
        this cannot describe cheaply drops the index instead (it rebuilds
        on the next remap).
        """
        if self._bboxes is None:
            return
        # Read the written extent straight off the index: napari paints
        # with 1-D coordinate arrays, and slices/scalars bound trivially.
        padded = list(index) + [slice(None)] * (3 - len(index))
        box = []
        for dim, ix in enumerate(padded):
            if isinstance(ix, (int, np.integer)):
                lo, hi = int(ix), int(ix) + 1
            elif isinstance(ix, slice):
                rng = range(*ix.indices(self.shape[dim]))
                if not len(rng):
                    return  # nothing written
                lo, hi = min(rng.start, rng[-1]), max(rng.start, rng[-1]) + 1
            elif isinstance(ix, np.ndarray) and ix.ndim == 1 and ix.size:
                lo, hi = int(ix.min()), int(ix.max()) + 1
            else:
                self._bboxes = None  # can't bound it — rebuild on next remap
                return
            box.append(slice(lo, hi))
        box = tuple(box)
        for v in np.unique(np.asarray(value)):
            v = int(v)
            if v:
                self._bboxes[v] = _bbox_union(self._bboxes.get(v), box)

    def _remap_vol(self, mapping: dict):
        """Apply *mapping* to the materialized volume, touching only the
        remapped labels' bounding boxes when an index is available.

        Returns the changed region, or None when the whole volume was
        scanned (and so may have changed anywhere).
        """
        vol = self._vol
        index = self._bbox_index()
        # Background has no bounding box (it is everywhere), so remapping
        # it — which the click tools never do — takes the whole-volume pass.
        if index is None or any(int(k) == 0 for k in mapping):
            _apply_value_map_inplace(vol, mapping)
            self._bboxes = None
            return None
        # Masks first, writes second, so chained mappings stay simultaneous.
        hits = []
        region = None
        for k, v in mapping.items():
            k, v = int(k), int(v)
            box = index.get(k)
            if box is None:
                continue
            sub = vol[box]
            hits.append((sub, sub == k, v))
            del index[k]
            if v:
                index[v] = _bbox_union(index.get(v), box)
            region = _bbox_union(region, box)
        for sub, mask, v in hits:
            sub[mask] = v
        return region if region is not None else _EMPTY_REGION

    def apply_mapping(self, mapping: dict):
        """Update caches in place for a global {old_id: new_id} remap.

        The cheap alternative to :meth:`invalidate` for delete / relabel:
        the materialized volume is remapped where the affected labels live
        instead of being rebuilt from disk, so the 3-D display and its
        pick-rays stay interactive.

        Returns the changed region as a tuple of slices into the view (for
        a partial canvas redraw), or None when the whole view may differ.
        """
        if self._vol is None:
            return None
        return self._remap_vol(mapping)

    def _ensure_vol(self) -> np.ndarray:
        """Materialize (and cache) the full view volume plane-by-plane."""
        if self._vol is None:
            out = np.empty(self.shape, dtype=self.dtype)
            for i in range(self.shape[0]):
                out[i] = self._plane(i)
            self._vol = out
            self._bboxes = None
        return self._vol

    def __getitem__(self, index):
        # Serve everything from the materialized volume while it exists —
        # it already reflects all edits (kept up to date in place).
        if self._vol is not None:
            return self._vol[index]
        if not isinstance(index, tuple):
            index = (index,)
        first = index[0] if index else slice(None)
        rest = index[1:]
        # Scalar outer plane — napari's 2-D slicing path, kept lazy so
        # scrubbing a movie never materializes the whole volume.
        if isinstance(first, (int, np.integer)):
            plane = self._plane(int(first))
            return plane[rest] if rest else plane
        # A range of planes — napari's 3-D slicing path, which re-reads the
        # *entire* view on every redraw.  Materialize once and serve this
        # and all later reads from the volume; edits patch it in place, so
        # a delete no longer costs a full re-read of the movie.
        if all(isinstance(ix, (int, np.integer, slice)) for ix in index):
            return self._ensure_vol()[index]
        # Fancy indexing (napari's 3-D picking): 1-D coord arrays, possibly
        # mixed with scalars (broadcast to the arrays' length), served
        # plane-by-plane so the source's T-slice cache is reused.
        if len(index) == 3 and all(
            isinstance(ix, (int, np.integer))
            or (isinstance(ix, np.ndarray) and ix.ndim == 1)
            for ix in index
        ):
            n = next(ix.size for ix in index if isinstance(ix, np.ndarray))
            pp, yy, xx = (
                (
                    np.asarray(ix, dtype=np.intp)
                    if isinstance(ix, np.ndarray)
                    else np.full(n, int(ix), dtype=np.intp)
                )
                for ix in index
            )
            out = np.zeros(pp.shape, dtype=self.dtype)
            for i in np.unique(pp):
                m = pp == i
                out[m] = self._plane(int(i))[yy[m], xx[m]]
            return out
        raise NotImplementedError(
            f"Track view does not support this index: {index!r}"
        )

    def __array__(self, dtype=None):
        """Materialize the full view plane-by-plane (used by napari's 3-D
        display and only then — 2-D slicing stays lazy).

        The result is cached on the view so subsequent refreshes and 3-D
        pick-rays cost no I/O; edits keep the cache current in place.
        """
        vol = self._ensure_vol()
        if dtype is not None and dtype != vol.dtype:
            return vol.astype(dtype)
        return vol


class _StackedTrackView(_TrackView):
    """All timepoints concatenated along Z: shape ``(T*Z, Y, X)``.

    Plane ``i`` shows timepoint ``i // Z``, slice ``i % Z``, so a track
    (one label ID through time) is a single connected object through the
    stack.  Fully editable: paint / fill writes are translated back to
    ``(t, z)`` and forwarded to the source, where they are recorded (and
    saved) like any other edit.
    """

    def __init__(self, source, yx_step: int = 1):
        super().__init__(source, yx_step)
        self.shape = (self.n_t * self.n_z, *self._yx_shape)

    def _plane(self, i: int) -> np.ndarray:
        t, z = divmod(int(i), self.n_z)
        t_slice = self._t_slice(t)
        return self._stride(t_slice[z] if self._source.ndim == 4 else t_slice)

    def _planes_of_t(self, t: int):
        return range(t * self.n_z, (t + 1) * self.n_z)

    def __setitem__(self, index, value):
        if self.yx_step != 1:
            raise TypeError(
                "This track view is YX-subsampled to fit in GPU memory "
                "and therefore read-only: a strided pixel cannot be "
                "written back losslessly. Use the ID-based click tools, "
                "or paint in the normal view (Track view 'Off')."
            )
        if not isinstance(index, tuple):
            index = (index,)
        # Keep the materialized 3-D volume in sync with paint edits.
        if self._vol is not None:
            self._vol[index] = value
            self._grow_bboxes(index, value)
        if self._source.ndim == 3:  # TYX: the view aliases the source
            self._source[index] = value
            return
        first = index[0] if index else None
        if isinstance(first, (int, np.integer)) and all(
            isinstance(ix, (int, np.integer, slice)) for ix in index[1:]
        ):
            t, z = divmod(int(first), self.n_z)
            self._source[(t, z, *index[1:])] = value
            return
        # napari paint / fill: tuple of equal-length 1-D coord arrays.
        if len(index) == 3 and all(
            isinstance(ix, np.ndarray) and ix.ndim == 1 for ix in index
        ):
            pp, yy, xx = (np.asarray(ix, dtype=np.intp) for ix in index)
            tt, zz = np.divmod(pp, self.n_z)
            vals = None if np.ndim(value) == 0 else np.asarray(value)
            for t in np.unique(tt):
                m = tt == t
                # Constant-t coordinate arrays hit the source wrapper's
                # sparse-diff fast path.
                self._source[
                    np.full(int(m.sum()), int(t), dtype=np.intp),
                    zz[m],
                    yy[m],
                    xx[m],
                ] = (
                    value if vals is None else vals[m]
                )
            return
        raise NotImplementedError(
            f"Track view does not support writing with this index: {index!r}"
        )


class _MaxProjTrackView(_TrackView):
    """One Z-max-projected plane per timepoint: shape ``(T, Y, X)``.

    Tracks read as clean tubes in 3-D, at the cost of Z information —
    where labels overlap along Z the higher ID wins.  Read-only for paint
    (a projected pixel has no unique ``(t, z)`` origin to write back to);
    the ID-based click tools still work because they only need the label
    ID under the cursor.  Projected planes are cached (LRU) and dropped
    via :meth:`invalidate` whenever the source is edited.
    """

    _CACHE_MAX_PLANES = 32

    def __init__(self, source, yx_step: int = 1):
        super().__init__(source, yx_step)
        self.shape = (self.n_t, *self._yx_shape)
        self._cache: OrderedDict = OrderedDict()

    def _plane(self, i: int) -> np.ndarray:
        i = int(i)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        t_slice = self._t_slice(i)
        plane = t_slice.max(axis=0) if self._source.ndim == 4 else t_slice
        # Copy so the cache doesn't pin the full-resolution plane alive.
        plane = np.ascontiguousarray(self._stride(plane))
        while len(self._cache) >= self._CACHE_MAX_PLANES:
            self._cache.popitem(last=False)
        self._cache[i] = plane
        return plane

    def invalidate(self) -> None:
        super().invalidate()
        self._cache.clear()

    def _planes_of_t(self, t: int):
        return (t,)

    def refresh_timepoint(self, t: int):
        # Unlike apply_mapping, this re-projects the timepoint exactly, so
        # labels that were occluded along Z reappear correctly.
        self._cache.pop(int(t), None)
        return super().refresh_timepoint(t)

    def apply_mapping(self, mapping: dict):
        """Remap caches in place; projected planes are dropped instead.

        2-D planes recompute exactly on demand (one timepoint of I/O), so
        they are dropped.  The materialized 3-D volume is remapped in
        place to stay interactive; where the remapped track occluded
        another label along Z the true projection would reveal that
        label, but recomputing it means re-reading the whole movie — the
        volume shows background there instead until the view is rebuilt
        (toggle the mode off/on).  The underlying data is always exact.
        """
        region = super().apply_mapping(mapping)
        self._cache.clear()
        return region

    def __setitem__(self, index, value):
        raise TypeError(
            "The Z-max-projection track view is read-only: a projected "
            "pixel has no unique (t, z) origin to write back to. Use "
            "click-to-delete / click-to-relabel (they work on label IDs), "
            "or switch the track view to 'Stack T along Z' to paint."
        )


# Budget for the materialized track-view volume.  napari's 3-D display
# uploads the whole (planes, Y, X) volume as ONE 3-D GPU texture; past
# GPU memory the upload fails mid-frame and vispy's command queue is
# left corrupted (every later frame dies with "Cannot SIZE object ...
# does not exist").  4 GiB fits comfortably on common 8 GB GPUs;
# override with the NAPARI_TMIDAS_TRACK_VIEW_GB env var.
_TRACK_VIEW_BUDGET_BYTES = int(
    float(os.environ.get("NAPARI_TMIDAS_TRACK_VIEW_GB", 4)) * 1024**3
)


def _pick_track_view_step(n_planes, n_y, n_x, itemsize, budget=None):
    """Smallest YX subsampling step that fits the track-view volume in
    *budget* bytes (the plane count is never reduced — T continuity is
    the whole point of the view)."""
    budget = _TRACK_VIEW_BUDGET_BYTES if budget is None else int(budget)
    full = int(n_planes) * int(n_y) * int(n_x) * int(itemsize)
    if full <= budget or budget <= 0:
        return 1
    # The step shrinks two axes, so start at ceil(sqrt(excess)); ceil
    # rounding in the strided shape can leave it one step short.
    step = max(1, int(np.ceil(np.sqrt(full / budget))))
    while (
        int(n_planes)
        * -(-int(n_y) // step)
        * -(-int(n_x) // step)
        * int(itemsize)
        > budget
    ):
        step += 1
    return step


def _load_label(path: str):
    """Load a label TIFF lazily.  Napari handles dask Labels natively."""
    try:
        from napari_tmidas._reader import tiff_reader_function

        results = tiff_reader_function(path)
        if results:
            arr = results[0][0]
            # Wrap dask arrays so napari's nd fancy indexing (used when
            # n_edit_dimensions > 2) doesn't raise NotImplementedError, and
            # so that __setitem__ / __array__ work correctly for editing.
            try:
                import dask.array as da

                if isinstance(arr, da.Array):
                    return _DaskFancyIndexWrapper(arr)
            except ImportError:
                pass
            return arr
    except Exception:
        pass
    if imread is None:
        raise ImportError("scikit-image is required to read label images.")
    return imread(path)


def _save_label_wrapper(wrapper: "_DaskFancyIndexWrapper", path: str) -> None:
    """Write a *_DaskFancyIndexWrapper* to a TIFF file one T-slice at a time.

    Peak RAM is bounded by one T-slice (e.g. ~390 MB for a 40×1600×1600
    uint32 volume) regardless of the total dataset size.  Each slice is
    assembled on demand as base → value-remap LUT → pending local diffs.

    The write goes to a temporary file in the same directory, which
    atomically replaces *path* only after the write fully succeeds.  This is
    essential: the wrapper's dask graph lazily reads its pages from *path*
    itself, so writing to *path* directly would truncate the source before
    the first slice is ever read — destroying the data.

    On success the edit state is committed (cleared): the new file contains
    the LUT and diffs, the path-backed graph re-reads it, and the undo log
    is emptied — a deletion that has been written to disk is permanent.
    """
    import tempfile

    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required to save label data.") from exc

    # Same directory → same filesystem → os.replace is atomic.
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tif", prefix=".tmp_save_", dir=os.path.dirname(path) or "."
    )
    os.close(fd)
    try:
        if wrapper.ndim < 3:
            # Small / 2-D array — compute in one shot.
            tifffile.imwrite(
                tmp_path,
                np.asarray(wrapper),
                compression="zlib",
                photometric="minisblack",
            )
        else:
            outer_dim = 0
            n_outer = wrapper.shape[outer_dim]
            bigtiff = (
                int(np.prod(wrapper.shape)) * wrapper.dtype.itemsize
                > 4 * 1024**3
            )

            def _iter_pages():
                # tifffile's iterator contract wants YX pages; assemble one
                # T-slice at a time (base → LUT → diffs) and emit its pages.
                page_shape = wrapper.shape[-2:]
                for t in range(n_outer):
                    t_slice = np.ascontiguousarray(
                        wrapper._get_outer_slice(outer_dim, t)
                    )
                    yield from t_slice.reshape((-1, *page_shape))

            # A single write() with a page generator streams the data (peak
            # RAM = one T-slice) while producing one series and allowing
            # compression — write(contiguous=True) would forbid compression.
            with tifffile.TiffWriter(tmp_path, bigtiff=bigtiff) as tif:
                tif.write(
                    _iter_pages(),
                    shape=wrapper.shape,
                    dtype=wrapper.dtype,
                    compression="zlib",
                    # Labels are never RGB; without this a size-3 axis is
                    # misinterpreted as color samples.
                    photometric="minisblack",
                )
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Drop the shared read handle for this path: it still points at the
    # replaced (deleted) inode.  Re-reads reopen the new file.
    try:
        from napari_tmidas._reader import invalidate_tiff_cache

        invalidate_tiff_cache(path)
    except ImportError:
        pass

    # Commit the edit state: the saved file now contains LUT + diffs.
    wrapper._op_log.clear()
    wrapper._lut = {}
    wrapper._arr = wrapper._base
    wrapper._diffs.clear()
    wrapper._cache.clear()


def _resample_nearest(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Nearest-neighbor resample *arr* to *shape* (same ndim, no interpolation).

    Used to align a raw-image slice with a lower/higher-resolution label
    slice: for each target pixel the spatially closest source pixel is
    sampled, mirroring how the viewer overlays the two layers via scale.
    """
    if arr.shape == tuple(shape):
        return arr
    idx = np.ix_(
        *[
            np.minimum(((np.arange(ts) + 0.5) * (ss / ts)).astype(int), ss - 1)
            for ts, ss in zip(shape, arr.shape)
        ]
    )
    return arr[idx]


def _label_dtype_is_integer(path: str) -> bool:
    """Return True if the TIFF at *path* stores integer data.

    Reads only the file header (no pixel data) via tifffile so large files
    are not loaded into RAM during validation.
    """
    try:
        import tifffile

        with tifffile.TiffFile(path) as tf:
            dtype = tf.series[0].dtype
        return np.issubdtype(dtype, np.integer)
    except Exception:
        # Fallback: load a tiny sample
        if imread is not None:
            try:
                return np.issubdtype(imread(path).dtype, np.integer)
            except Exception:
                pass
    return False


class LabelInspector:
    # Cap on per-track intensity-histogram bins; wider integer ranges (and
    # non-integer raws) fall back to a mean instead of a percentile.
    _INTENSITY_MAX_BINS = 1 << 16

    def __init__(self, viewer: Viewer):
        self.viewer = viewer
        self.image_label_pairs = []
        self.current_index = 0
        self._click_delete_cb = None  # click-to-delete mouse callback
        self._click_relabel_cb = None  # click-to-relabel mouse callback
        # Scope of the click tools: "all" applies to every timepoint where
        # the clicked ID exists, "current" only to the clicked timepoint.
        self.delete_scope = "all"
        self.relabel_scope = "all"
        # Merge defaults to the clicked timepoint: which labels touch is a
        # per-frame fact, so spreading it over the movie is opt-in.
        self.merge_scope = "current"
        self._click_split_cb = None  # click-to-split mouse callback
        self._click_merge_cb = None  # click-to-merge-neighbors mouse callback
        self._click_grow_cb = None  # click-to-grow-to-signal mouse callback
        # Click-to-grow settings (see grow_label_at_timepoint, which segments
        # with SAM2): grow_max_px is both the context SAM2 is shown and the
        # cap on how far the label may travel, grow_smooth_px polishes the
        # returned contour, grow_channel picks which raw channel defines it.
        self.grow_max_px = 30
        self.grow_smooth_px = 1
        self.grow_channel = "mean"
        self._click_add_cb = None  # click-to-add-missed-cell mouse callback
        # Click-to-add settings (see add_label_at_timepoint, which segments
        # with SAM2 too): add_max_px is the largest radius the new label may
        # reach from the clicked pixel and the context SAM2 is shown,
        # add_propagate_z continues the object onto the neighboring Z-planes.
        self.add_max_px = 30
        self.add_smooth_px = 1
        self.add_channel = "mean"
        self.add_propagate_z = True
        # Seeds accumulated for a pending split, or None:
        # {"label_id": int, "t": int, "coords": [tuple, ...]}.  Plain-click
        # commits once two or more seeds exist; Ctrl+click adds more first.
        self._split_seeds = None
        # Self-managed, globally-unique ID allocator for split-off regions,
        # (re)initialized per pair to global_max + 1.  napari's own "next
        # free ID" ('m') maxes over the whole array but no-ops on wrapped /
        # dask labels, so a split allocates its new ID here instead — global
        # so a fragment never collides with a real track in another frame.
        self._next_free_id = None
        # Most recent single-timepoint click edit, for Ctrl+Z:
        # {"t": int, "restores": [(coords, old_id), ...], "desc": str}.
        # Cleared by any later all-T remap so undo order stays correct.
        self._single_t_last = None
        # Cached per-track raw-intensity stats for the current pair, and the
        # last low-intensity deletion (for live-preview undo). See
        # delete_low_intensity_tracks.
        self._track_stats = None
        self._low_intensity_last = None
        # Manual channel-axis override for the raw image: "auto" runs metadata
        # detection, "none" forces no channel axis, an int string (e.g. "2")
        # pins that axis.  Fallback for TIFFs with missing/ambiguous axes tags.
        self.channel_axis_override = "auto"
        # Track-inspection view: "off" | "stack" | "max" (see _TrackView).
        # Like the click modes, the chosen mode persists across pairs.
        self.track_view_mode = "off"
        self._track_view_layer = None
        self._track_hidden = []  # [(layer, was_visible)] to restore on exit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _can_show_message(self) -> bool:
        """Return True if it's (probably) safe to show a QMessageBox.

        On Windows CI (headless) creating a modal dialog without a running
        QApplication or with a mocked viewer can cause access violations.
        We suppress dialogs when:
          * No QApplication instance exists
          * Running under pytest (detected via env var)
          * The provided viewer is a mock (has no 'window' attr)
        """
        try:
            from qtpy.QtWidgets import QApplication

            if QApplication.instance() is None:
                return False
        except (ImportError, RuntimeError):
            return False
        if "PYTEST_CURRENT_TEST" in os.environ:
            return False
        return hasattr(self.viewer, "window")

    def _show_message(self, level: str, title: str, text: str):
        """Safely show a QMessageBox if environment allows, otherwise noop."""
        if not self._can_show_message():
            return
        try:
            if level == "warning":
                QMessageBox.warning(None, title, text)
            else:
                QMessageBox.information(None, title, text)
        except (RuntimeError, ValueError, OSError):
            # Never let common GUI/runtime issues crash tests
            pass

    def load_image_label_pairs(self, folder_path: str, label_suffix: str):
        """
        Load image-label pairs from a folder.
        Finds all files with the given suffix and matches them with their corresponding image files.
        Validates that label files are in the correct format.
        """
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            self.viewer.status = f"Folder path does not exist: {folder_path}"
            return

        files = os.listdir(folder_path)

        # Find all files that contain the label suffix
        # Using "in" instead of "endswith" for more flexibility
        potential_label_files = [
            file for file in files if label_suffix in file
        ]

        if not potential_label_files:
            self.viewer.status = f"No files found with suffix '{label_suffix}'"
            self._show_message(
                "warning",
                "No Label Files Found",
                f"No files containing '{label_suffix}' were found in {folder_path}.",
            )
            return

        # Process all potential label files
        self.image_label_pairs = []
        skipped_files = []
        format_issues = []

        # All recognised image extensions / directory types
        _IMAGE_EXTS = {
            ".tif",
            ".tiff",
            ".zarr",
            ".czi",
            ".nd2",
            ".png",
            ".jpg",
            ".jpeg",
            ".npy",
        }

        for label_file in potential_label_files:
            label_path = os.path.join(folder_path, label_file)

            # Try to find a matching image file (everything before the label suffix)
            base_name = label_file.split(label_suffix)[0]

            # Look for potential images matching the base name with any known extension.
            # Normalise runs of whitespace so filenames that differ only in
            # single- vs double-space still match (e.g. "Image 14  #2" vs "Image 14 #2").
            base_name_norm = re.sub(r" +", " ", base_name)
            potential_images = [
                file
                for file in files
                if re.sub(r" +", " ", file).startswith(base_name_norm)
                and file != label_file
                and os.path.splitext(file)[1].lower() in _IMAGE_EXTS
            ]

            # If we found at least one potential image
            if potential_images:
                image_path = os.path.join(folder_path, potential_images[0])

                # Validate label file format
                try:
                    # Validate dtype via header only — no pixel data loaded
                    if not _label_dtype_is_integer(label_path):
                        format_issues.append(
                            (label_file, "not an integer type")
                        )
                        continue

                    # Add valid pair
                    self.image_label_pairs.append((image_path, label_path))

                except (
                    FileNotFoundError,
                    OSError,
                    ValueError,
                    Exception,
                ) as e:
                    skipped_files.append((label_file, str(e)))
            else:
                skipped_files.append((label_file, "no matching image found"))

        # Report results
        if self.image_label_pairs:
            self.viewer.status = (
                f"Found {len(self.image_label_pairs)} valid image-label pairs."
            )
            self.current_index = 0
            self._load_current_pair()
        else:
            self.viewer.status = "No valid image-label pairs found."

        # Show detailed report if there were issues
        if skipped_files or format_issues:
            msg = ""
            if skipped_files:
                msg += "Skipped files:\n"
                for file, reason in skipped_files:
                    msg += f"- {file}: {reason}\n"

            if format_issues:
                msg += "\nFormat issues:\n"
                for file, issue in format_issues:
                    msg += f"- {file}: {issue}\n"

            self._show_message("info", "Loading Report", msg)

    def _resolve_channel_axis(self, image, label_image, image_path):
        """Return the raw image's channel-axis index, or ``None``.

        Honors :attr:`channel_axis_override` first (``"none"`` → no axis, an
        int string → that axis); ``"auto"`` falls back to metadata detection
        for zarr and TIFF inputs.  The returned index is always validated
        against the loaded array's dimensionality, so a stale or out-of-range
        value (e.g. from a squeezed array or a wrong manual pick) degrades to
        ``None`` rather than corrupting the overlay.
        """
        override = str(self.channel_axis_override).strip().lower()
        channel_axis = None
        if override == "none":
            return None
        if override not in ("", "auto"):
            try:
                channel_axis = int(override)
            except ValueError:
                channel_axis = None
        elif image.ndim > label_image.ndim:
            try:
                if _is_zarr(image_path):
                    from napari_tmidas._file_selector import (
                        detect_channels_from_zarr_path,
                    )

                    _n_ch, channel_axis = detect_channels_from_zarr_path(
                        image_path
                    )
                else:
                    # Metadata first, then the shared shape heuristic — the
                    # same two-stage strategy the zarr detector uses, so a
                    # TIFF with missing/ambiguous axes tags (e.g. tifffile
                    # reporting "QQYX") still resolves the channel axis from
                    # the array shape instead of silently giving up.
                    from napari_tmidas._reader import (
                        detect_channel_axis_from_tiff_path,
                    )

                    channel_axis = detect_channel_axis_from_tiff_path(
                        image_path
                    )
                    if channel_axis is None:
                        from napari_tmidas._file_selector import (
                            _detect_channels_from_shape,
                        )

                        _n_ch, channel_axis = _detect_channels_from_shape(
                            image.shape
                        )
            except Exception:
                channel_axis = None
        # Guard: a stale/out-of-range index would misalign the overlay.
        if channel_axis is not None and not (0 <= channel_axis < image.ndim):
            channel_axis = None
        return channel_axis

    def _load_current_pair(self):
        """
        Load the current image-label pair into the Napari viewer.
        Automatically scales both layers so they share the same spatial extent,
        enabling correct overlay even when image and label have different resolutions.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to inspect."
            return

        # New pair → previous pair's intensity stats and preview no longer apply.
        self._invalidate_track_stats()

        # Layers are about to be cleared — drop stale track-view references;
        # the view is rebuilt over the new pair below if the mode is on.
        self._track_view_layer = None
        self._track_hidden = []

        # Pending split seeds and the ID allocator belong to the old pair.
        self._split_seeds = None
        self._next_free_id = None

        image_path, label_path = self.image_label_pairs[self.current_index]
        image = _load_image(image_path)
        label_image = _load_label(label_path)

        # --- Resolve the raw image's channel axis ---------------------------
        # A channel axis lets napari (a) split the raw into per-channel layers
        # and (b) be excluded from the label spatial-scale computation.  The
        # manual override wins when set; otherwise it is auto-detected for both
        # zarr (via .zattrs) and multi-channel TIFFs (e.g. a TZCYX raw paired
        # with a TZYX label, read from series axes metadata).
        channel_axis = self._resolve_channel_axis(
            image, label_image, image_path
        )

        # Build the image shape *without* the channel axis for scale comparison
        if channel_axis is not None:
            img_spatial = np.array(
                [s for i, s in enumerate(image.shape) if i != channel_axis],
                dtype=float,
            )
        else:
            img_spatial = np.array(image.shape, dtype=float)

        # Compute per-axis scale so label covers the same spatial extent.
        lbl_shape = np.array(label_image.shape, dtype=float)
        n_shared = min(len(img_spatial), len(lbl_shape))
        label_scale = (
            img_spatial[-n_shared:] / lbl_shape[-n_shared:]
        ).tolist()
        if len(lbl_shape) > n_shared:
            label_scale = [1.0] * (len(lbl_shape) - n_shared) + label_scale

        # Clear existing layers
        self.viewer.layers.clear()

        # Add image; split into per-channel layers when a channel axis is known
        img_kwargs = {"name": f"Image ({os.path.basename(image_path)})"}
        if channel_axis is not None:
            img_kwargs["channel_axis"] = channel_axis
        self.viewer.add_image(image, **img_kwargs)

        new_labels_layer = self.viewer.add_labels(
            label_image,
            scale=label_scale,
            name=f"Labels ({os.path.basename(label_path)})",
        )

        # Click-to-delete / click-to-relabel modes persist across pairs;
        # rebind their shared Ctrl+Z undo on the freshly created layer.
        if (
            self._click_delete_cb is not None
            or self._click_relabel_cb is not None
            or self._click_split_cb is not None
            or self._click_merge_cb is not None
            or self._click_grow_cb is not None
            or self._click_add_cb is not None
        ):
            self._bind_undo_key(new_labels_layer, True)

        # Track view persists across pairs; rebuild it over the new labels.
        if self.track_view_mode != "off":
            self._apply_track_view()

        # Show progress
        total = len(self.image_label_pairs)
        ch_info = (
            f", channel_axis={channel_axis}"
            if channel_axis is not None
            else ""
        )
        self.viewer.status = (
            f"Viewing pair {self.current_index + 1} of {total}: "
            f"{os.path.basename(image_path)}"
            f"{ch_info} (label scale {label_scale})"
        )

    def save_current_labels(self):
        """
        Save the current labels back to the original file.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to save."
            return

        _, label_path = self.image_label_pairs[self.current_index]

        # Find the labels layer in the viewer (never the track view — its
        # 3-D shape must not be written over the TZYX label file).
        labels_layer = self._find_labels_layer()

        if labels_layer is None:
            self.viewer.status = "No labels found."
            return

        # Save the labels layer data to the original file path.
        # For dask-backed wrappers, write T-by-T to avoid loading the full
        # dataset into RAM.
        if isinstance(labels_layer.data, _DaskFancyIndexWrapper):
            _save_label_wrapper(labels_layer.data, label_path)
        else:
            labels_layer.save(label_path)
        self.viewer.status = f"Saved labels to {label_path}."

    def _find_labels_layer(self):
        """The regular (source-backed) labels layer, never a track view."""
        return next(
            (
                layer
                for layer in self.viewer.layers
                if isinstance(layer, Labels)
                and not isinstance(getattr(layer, "data", None), _TrackView)
            ),
            None,
        )

    def _click_value_layer(self):
        """Layer that resolves what the user clicked: the track view when
        active (it is the visible one), else the regular labels layer."""
        if self._track_view_layer is not None:
            return self._track_view_layer
        return self._find_labels_layer()

    def _remap_all_timepoints(self, labels_layer, mapping: dict) -> None:
        """Apply {old_id: new_id} *mapping* to the layer across **all** timepoints.

        For dask-backed labels this is a value-remap LUT entry
        (:meth:`_DaskFancyIndexWrapper.remap_values`): zero I/O at call time,
        one lazy LUT pass at read/save regardless of how many remaps have
        accumulated, and pending manual edits are remapped too.  The current
        view refreshes instantly from the in-place-updated slice cache.
        """
        data = labels_layer.data

        try:
            import dask.array as da
        except ImportError:
            da = None

        if isinstance(data, _DaskFancyIndexWrapper):
            data.remap_values(mapping)
        elif da is not None and isinstance(data, da.Array):
            wrapper = _DaskFancyIndexWrapper(data)
            wrapper.remap_values(mapping)
            labels_layer.data = wrapper
        else:
            # Plain numpy: in-place.
            _apply_value_map_inplace(data, mapping)

        labels_layer.refresh()
        self._refresh_track_view(mapping)
        # An all-T remap supersedes any pending single-T undo.
        self._single_t_last = None

    def delete_label_all_timepoints(self, label_id: int = None) -> None:
        """Delete *label_id* (default: the selected label) from **all** timepoints."""
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return

        if label_id is None:
            label_id = int(labels_layer.selected_label)
        label_id = int(label_id)
        if label_id == 0:
            self.viewer.status = "Select a non-background label first."
            return

        self._remap_all_timepoints(labels_layer, {label_id: 0})
        # A manual edit changes track geometry — re-measure on next preview.
        self._invalidate_track_stats()
        self.viewer.status = (
            f"Label {label_id} removed from all timepoints. "
            "Save to write changes to disk."
        )

    def relabel_label_all_timepoints(
        self, old_id: int, new_id: int = None
    ) -> None:
        """Give *old_id* the value *new_id* (default: the selected label)
        across **all** timepoints.

        Relabeling onto an existing ID merges the two labels; relabeling to
        0 is equivalent to deletion.
        """
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return

        if new_id is None:
            new_id = int(labels_layer.selected_label)
        old_id, new_id = int(old_id), int(new_id)
        if old_id == 0:
            self.viewer.status = (
                "Cannot relabel background — click a label instead."
            )
            return
        if old_id == new_id:
            self.viewer.status = f"Label {old_id} already has this ID."
            return

        self._remap_all_timepoints(labels_layer, {old_id: new_id})
        # A manual edit changes track geometry — re-measure on next preview.
        self._invalidate_track_stats()
        self.viewer.status = (
            f"Label {old_id} relabeled to {new_id} on all timepoints. "
            "Save to write changes to disk."
        )

    def _remap_one_timepoint(self, labels_layer, mapping: dict, t: int):
        """Apply {old_id: new_id} *mapping* at timepoint *t* only.

        Unlike the all-T LUT path, the edit is recorded like a paint
        stroke — a sparse per-slice diff on wrapper-backed labels, an
        in-place write on numpy — so it saves normally and stays
        consistent with later all-T remaps (which patch pending diffs).

        Returns the [(coords, old_values)] restore record for the
        single-level Ctrl+Z undo (empty when no ID was present at *t*),
        or the string ``"all"`` when the labels have no time axis and the
        remap fell back to the whole array.
        """
        data = labels_layer.data
        if getattr(data, "ndim", 0) < 3:
            # No time axis to restrict to — the whole array is the one
            # timepoint (undo goes through the LUT op log instead).
            self._remap_all_timepoints(labels_layer, mapping)
            return "all"

        try:
            import dask.array as da
        except ImportError:
            da = None
        if da is not None and isinstance(data, da.Array):
            data = _DaskFancyIndexWrapper(data)
            labels_layer.data = data

        t = int(t)
        if isinstance(data, _DaskFancyIndexWrapper):
            t_slice = data._get_outer_slice(0, t)
        else:
            t_slice = data[t]
        # One pass over the slice for the whole mapping: merging an
        # over-segmented cell remaps a dozen neighbors at once, and a
        # mask per neighbor would re-scan the volume a dozen times.
        items = [
            (int(k), int(v)) for k, v in mapping.items() if int(k) != int(v)
        ]
        restores = []
        if items:
            keys = np.array([k for k, _ in items], dtype=t_slice.dtype)
            targets = np.array([v for _, v in items], dtype=t_slice.dtype)
            hit = (
                (t_slice == keys[0])
                if len(items) == 1
                else np.isin(t_slice, keys)
            )
            coords = np.nonzero(hit)
            if coords[0].size:
                # Old values are read before any write, so a chained
                # mapping stays simultaneous, and they double as the undo
                # record for exactly the voxels that changed.
                old = t_slice[coords]
                order = np.argsort(keys)
                new = targets[order][np.searchsorted(keys[order], old)]
                if isinstance(data, _DaskFancyIndexWrapper):
                    data[(t, *coords)] = new
                else:
                    t_slice[coords] = new
                restores.append((coords, old))
        if restores:
            labels_layer.refresh()
            self._refresh_track_view(timepoint=t)
        return restores

    def delete_label_at_timepoint(self, label_id: int, t: int) -> None:
        """Delete *label_id* from timepoint *t* only."""
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return
        label_id = int(label_id)
        if label_id == 0:
            self.viewer.status = "Select a non-background label first."
            return
        restores = self._remap_one_timepoint(labels_layer, {label_id: 0}, t)
        self._invalidate_track_stats()
        if restores == "all":
            self.viewer.status = (
                f"Label {label_id} removed (no time axis — whole image). "
                "Save to write changes to disk."
            )
            return
        if not restores:
            self.viewer.status = (
                f"Label {label_id} not present at timepoint {t}."
            )
            return
        self._single_t_last = {
            "t": int(t),
            "restores": restores,
            "desc": f"{label_id} restored (deletion undone)",
        }
        self.viewer.status = (
            f"Label {label_id} removed from timepoint {t}. "
            "Save to write changes to disk."
        )

    def relabel_label_at_timepoint(
        self, old_id: int, new_id: int, t: int
    ) -> None:
        """Give *old_id* the value *new_id* at timepoint *t* only."""
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return
        old_id, new_id = int(old_id), int(new_id)
        if old_id == 0:
            self.viewer.status = (
                "Cannot relabel background — click a label instead."
            )
            return
        if old_id == new_id:
            self.viewer.status = f"Label {old_id} already has this ID."
            return
        restores = self._remap_one_timepoint(labels_layer, {old_id: new_id}, t)
        self._invalidate_track_stats()
        if restores == "all":
            self.viewer.status = (
                f"Label {old_id} relabeled to {new_id} (no time axis — "
                "whole image). Save to write changes to disk."
            )
            return
        if not restores:
            self.viewer.status = (
                f"Label {old_id} not present at timepoint {t}."
            )
            return
        self._single_t_last = {
            "t": int(t),
            "restores": restores,
            "desc": f"{old_id}→{new_id} reverted",
        }
        self.viewer.status = (
            f"Label {old_id} relabeled to {new_id} at timepoint {t}. "
            "Save to write changes to disk."
        )

    def _ray_points(self, layer, event, dims_displayed):
        """Integer data coordinates sampled along the 3-D pick ray, ``(N, ndim)``.

        The same ray napari casts for ``get_value``, marched at ~2 samples per
        voxel of its length; non-displayed axes (T for a 4-D layer) come along
        as their constant slider value, so each row is a full data index.
        Returns None when the ray misses the layer.
        """
        try:
            # Same world→layer conversions napari's get_value performs
            # before its own ray cast: dims_displayed arrives in world
            # dims, position/ray in world coordinates.
            from napari.layers.utils.layer_utils import (
                dims_displayed_world_to_layer,
            )

            # Layer._world_to_data_ray is private (proxy-blocked for
            # plugins from napari 0.7), so transform the direction as
            # the difference of two transformed points instead.
            view_dir = np.asarray(event.view_direction, dtype=float)
            r1 = np.asarray(layer.world_to_data(view_dir), dtype=float)
            r0 = np.asarray(
                layer.world_to_data(np.zeros_like(view_dir)), dtype=float
            )
            data_ray = (r1 - r0) / np.linalg.norm(r1 - r0)

            start, end = layer.get_ray_intersections(
                layer.world_to_data(event.position),
                data_ray,
                dims_displayed_world_to_layer(
                    dims_displayed,
                    ndim_world=len(event.position),
                    ndim_layer=layer.ndim,
                ),
                world=False,
            )
        except Exception:
            return None
        if start is None or end is None:
            return None
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        # Same sampling convention as napari's _get_value_ray: the ray
        # endpoints are voxel-corner-based (voxel i spans [i, i+1)), so
        # indices come from truncation, at ~2 samples per voxel of length.
        n = max(int(2 * np.linalg.norm(end - start)), 2)
        pts = np.floor(np.linspace(start, end, n)).astype(np.intp)
        shape = np.asarray(layer.data.shape, dtype=np.intp)
        np.clip(pts, 0, shape - 1, out=pts)
        return pts

    def _ray_hit_voxel(self, layer, label_id, event, dims_displayed):
        """Full data coordinate of the first voxel carrying *label_id* along
        the 3-D pick ray (a 1-D int array), or None.

        napari's ``get_value`` already resolved the click to *label_id*
        via the same ray, so marching it again (one vectorized fancy-index
        read served from the view's caches) recovers *where* along the ray it
        was hit — the precise voxel is needed both for the timepoint and as a
        split seed.
        """
        pts = self._ray_points(layer, event, dims_displayed)
        if pts is None:
            return None
        vals = np.asarray(layer.data[tuple(pts.T)])
        hit = np.nonzero(vals == label_id)[0]
        if hit.size == 0:
            return None
        return pts[hit[0]]

    def _ray_signal_voxel(self, layer, event, dims_displayed, raw_t):
        """``(voxel, blocker)`` for a background click in 3-D display.

        The 3-D counterpart of picking a pixel.  napari's own pick reports
        only where the ray hits a label, and a cell that was missed has none —
        but the user is aiming at something they can see, and napari draws the
        raw volume by maximum intensity along exactly this ray.  So the voxel
        under the cursor is the ray's *brightest* sample, and that is the seed.

        Deliberately the brightest overall, not the brightest unlabeled one:
        when the ray passes through a cell that already has a label, the
        brightest thing no label owns is that cell's own halo rim, a pixel
        ahead of it — seeding there would add a sliver in front of a cell the
        user was pointing at.  A labeled peak instead comes back as *blocker*,
        which the caller reports, since the object drawn under the cursor is
        that label and it needs the grow tool, not this one.

        *raw_t* is the (label-aligned) raw volume at the clicked timepoint;
        its axes are the ray's trailing ones.  The voxel is None when the ray
        misses the layer.
        """
        pts = self._ray_points(layer, event, dims_displayed)
        if pts is None:
            return None, 0
        spatial = pts[:, pts.shape[1] - np.ndim(raw_t) :]
        intensity = np.asarray(raw_t[tuple(spatial.T)], dtype=float)
        peak = int(np.argmax(intensity))
        owner = int(np.asarray(layer.data[tuple(pts[peak])]))
        if owner:
            return None, owner
        return pts[peak], 0

    def _ray_hit_plane(self, layer, label_id, event, dims_displayed):
        """Axis-0 index of the first voxel carrying *label_id* along the
        3-D pick ray, or None."""
        vox = self._ray_hit_voxel(layer, label_id, event, dims_displayed)
        if vox is None:
            return None
        return int(vox[0])

    def _clicked_timepoint(self, layer, label_id, event):
        """Timepoint of the clicked voxel, or None when unresolvable.

        In 2-D display (and for a 4-D layer in 3-D display, where T is
        the non-displayed axis) the click position carries the axis-0
        coordinate directly.  For a 3-D layer in 3-D display the pick
        ray is marched to the voxel that was hit.  Track-view plane
        indices are mapped back to source timepoints.
        """
        data = getattr(layer, "data", None)
        dims_displayed = list(getattr(event, "dims_displayed", None) or [])
        if len(dims_displayed) == 3 and getattr(data, "ndim", 0) == 3:
            plane = self._ray_hit_plane(layer, label_id, event, dims_displayed)
        else:
            try:
                plane = int(
                    round(float(layer.world_to_data(event.position)[0]))
                )
            except Exception:
                return None
        if plane is None:
            return None
        if isinstance(data, _StackedTrackView):
            t = plane // data.n_z
        else:  # max-projection view (plane == t) or the regular layer
            t = plane
        n_t = data.n_t if isinstance(data, _TrackView) else data.shape[0]
        return min(max(t, 0), n_t - 1)

    def _iter_label_raw_slices(self, labels_data, raw):
        """Yield aligned (label_slice, raw_slice) numpy pairs, outer-dim-wise.

        Streams one outer (T) slice at a time so peak RAM stays at one
        slice for arbitrarily long movies.  Wrapper-backed labels are read
        via :meth:`_DaskFancyIndexWrapper._get_outer_slice` so pending
        unsaved edits are measured too.  Raw slices are nearest-neighbor
        resampled onto the label grid when resolutions differ.
        """
        if labels_data.ndim < 3:
            outer = [None]
        else:
            outer = range(labels_data.shape[0])
        for t in outer:
            if t is None:
                lbl_t, raw_t = labels_data, raw
            elif isinstance(labels_data, _DaskFancyIndexWrapper):
                lbl_t, raw_t = labels_data._get_outer_slice(0, t), raw[t]
            else:
                lbl_t, raw_t = labels_data[t], raw[t]
            if hasattr(lbl_t, "compute"):
                lbl_t = lbl_t.compute()
            if hasattr(raw_t, "compute"):
                raw_t = raw_t.compute()
            lbl_t = np.asarray(lbl_t)
            raw_t = _resample_nearest(np.asarray(raw_t), lbl_t.shape)
            yield lbl_t, raw_t

    def _raw_slice_at(self, labels_data, t, channel="mean"):
        """The raw image's spatial slice at timepoint *t*, label-aligned.

        The single-timepoint counterpart of :meth:`_iter_label_raw_slices`:
        the raw is opened lazily, its channel axis resolved (and reduced to
        one plane per *channel* — ``"mean"`` averages all channels), and only
        the requested timepoint is computed, so cost is one slice however
        long the movie is.  The result is nearest-neighbor resampled onto the
        label grid.  Returns ``None`` (with an explanatory viewer status) when
        the raw cannot be loaded or aligned; pass ``t=None`` for labels with
        no time axis.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No raw image loaded."
            return None
        image_path, _ = self.image_label_pairs[self.current_index]
        try:
            raw = _load_image(image_path)
        except Exception as exc:
            self.viewer.status = f"Could not load raw image: {exc}"
            return None

        channel = str(channel).strip().lower()
        channel_axis = self._resolve_channel_axis(raw, labels_data, image_path)
        if channel_axis is not None:
            n_ch = raw.shape[channel_axis]
            ci = None
            if channel not in ("", "mean", "auto"):
                with contextlib.suppress(ValueError):
                    ci = int(channel)
            if ci is not None and 0 <= ci < n_ch:
                raw = np.take(raw, ci, axis=channel_axis)
            else:
                raw = raw.mean(axis=channel_axis)
        while raw.ndim > labels_data.ndim and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != labels_data.ndim:
            self.viewer.status = (
                f"Cannot read the raw image: shape {raw.shape} does not "
                f"align with label shape {labels_data.shape}."
            )
            return None

        raw_t = raw if t is None else raw[int(t)]
        if hasattr(raw_t, "compute"):
            raw_t = raw_t.compute()
        spatial = tuple(
            labels_data.shape if t is None else labels_data.shape[1:]
        )
        return _resample_nearest(np.asarray(raw_t), spatial)

    def _measure_track_intensities(self, labels_layer, channel="mean"):
        """Build and cache per-track raw-intensity statistics for the pair.

        In one streaming pass (one T-slice at a time, so peak RAM stays at a
        single slice) this accumulates, for every non-zero label ID:

        * a **histogram** of its voxel intensities — indexed by integer raw
          value (bounded to ``_INTENSITY_MAX_BINS`` bins), enabling any
          percentile to be read back later without re-touching the image, and
        * a **sum and count** as a fallback average for non-integer or
          wide-range raws where a histogram is impractical.

        *channel* selects which channel of a multi-channel raw supplies the
        intensity: ``"mean"`` averages all channels, an int string picks that
        channel index along the detected channel axis.

        Also records the raw image's global min/max for bit-depth-independent
        normalization.  The result is cached and reused across threshold
        changes; it is re-measured when *channel* differs from the cached run,
        and :meth:`_invalidate_track_stats` drops it when the pair changes or
        the labels are edited by another tool.

        Returns the stats dict, or ``None`` with an explanatory viewer status
        when the raw image cannot be loaded or aligned.
        """
        channel = str(channel).strip().lower()
        if (
            self._track_stats is not None
            and self._track_stats.get("channel") == channel
        ):
            return self._track_stats

        image_path, _ = self.image_label_pairs[self.current_index]
        try:
            raw = _load_image(image_path)
        except Exception as exc:
            self.viewer.status = f"Could not load raw image: {exc}"
            return None

        labels_data = labels_layer.data
        channel_axis = self._resolve_channel_axis(raw, labels_data, image_path)
        if channel_axis is not None:
            n_ch = raw.shape[channel_axis]
            ci = None
            if channel not in ("", "mean", "auto"):
                try:
                    ci = int(channel)
                except ValueError:
                    ci = None
            if ci is not None and 0 <= ci < n_ch:
                raw = np.take(raw, ci, axis=channel_axis)
            else:
                # "mean" or an out-of-range pick → average all channels.
                raw = raw.mean(axis=channel_axis)
        # Squeeze leading singleton dims (e.g. a stray T=1 axis).
        while raw.ndim > labels_data.ndim and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != labels_data.ndim:
            self.viewer.status = (
                f"Cannot measure intensity: raw shape {raw.shape} does not "
                f"align with label shape {labels_data.shape}."
            )
            return None

        # Integer raws with a modest range get exact per-track histograms;
        # anything else falls back to a plain mean.
        raw_dtype = (
            np.asarray(raw).dtype if not hasattr(raw, "dtype") else raw.dtype
        )
        n_bins = 0
        if np.issubdtype(raw_dtype, np.integer):
            info_max = int(np.iinfo(raw_dtype).max)
            if 0 <= info_max < self._INTENSITY_MAX_BINS:
                n_bins = info_max + 1

        hist: dict = {}
        sums: dict = {}
        counts: dict = {}
        raw_min, raw_max = np.inf, -np.inf
        for lbl_t, raw_t in self._iter_label_raw_slices(labels_data, raw):
            raw_min = min(raw_min, float(raw_t.min()))
            raw_max = max(raw_max, float(raw_t.max()))
            flat_l = lbl_t.ravel()
            fg = flat_l != 0
            if not np.any(fg):
                continue
            flat_l = flat_l[fg]
            flat_r = raw_t.ravel()[fg]
            uniq, inv = np.unique(flat_l, return_inverse=True)
            s = np.bincount(inv, weights=flat_r.astype(np.float64))
            c = np.bincount(inv)
            for i, (u, sv, cv) in enumerate(
                zip(uniq.tolist(), s.tolist(), c.tolist())
            ):
                sums[u] = sums.get(u, 0.0) + sv
                counts[u] = counts.get(u, 0) + cv
                if n_bins:
                    vals = np.clip(
                        flat_r[inv == i].astype(np.int64), 0, n_bins - 1
                    )
                    h = hist.get(u)
                    if h is None:
                        h = np.zeros(n_bins, dtype=np.int64)
                        hist[u] = h
                    h += np.bincount(vals, minlength=n_bins)

        self._track_stats = {
            "channel": channel,
            "hist": hist if n_bins else None,
            "sums": sums,
            "counts": counts,
            "raw_min": raw_min,
            "raw_max": raw_max,
        }
        return self._track_stats

    def _track_brightness(self, stats: dict, percentile: float) -> dict:
        """Return ``{label_id: normalized_brightness}`` for the given percentile.

        Each track's brightness is the *percentile*-th percentile of its voxel
        intensities (50 = median, the robust analog of the mean; lower values
        weight the track's dimmer voxels so partially-dim tracks are flagged),
        linearly normalized to [0, 1] via the raw image's global min/max.
        Falls back to the mean when no histogram was built.
        """
        raw_min, raw_max = stats["raw_min"], stats["raw_max"]
        span = raw_max - raw_min
        hist = stats["hist"]
        out = {}
        for label_id, count in stats["counts"].items():
            if hist is not None:
                h = hist[label_id]
                total = h.sum()
                rank = np.searchsorted(
                    np.cumsum(h), percentile / 100.0 * total, side="left"
                )
                value = float(min(rank, len(h) - 1))
            else:
                value = stats["sums"][label_id] / count
            out[int(label_id)] = (value - raw_min) / span if span > 0 else 0.0
        return out

    def delete_low_intensity_tracks(
        self, threshold: float, channel="mean", percentile: float = 50.0
    ) -> None:
        """Delete every track whose raw-image brightness, normalized to
        [0, 1], is below *threshold* — across **all** timepoints.

        A track's brightness is the median of its voxel intensities
        (*percentile* defaults to 50; robust to a few bright pixels).
        Normalization uses the raw image's own global min/max, so the same
        threshold behaves identically for 8-bit and 16-bit inputs (and for
        data that only occupies part of its dtype range, e.g. a 12-bit
        camera).  For a multi-channel raw, *channel* selects which channel
        supplies the intensity (``"mean"`` averages all channels, an int
        string picks that channel index).

        Re-applying is safe: the previous low-intensity deletion (if any) is
        undone first, so each Apply reflects only the current settings rather
        than compounding.  ``threshold <= 0`` deletes nothing and simply
        restores all tracks.

        The deletion itself is a single remap operation: instant on
        dask-backed labels, also undoable with Ctrl+Z while a click mode is
        active, and staged in memory until saved.
        """
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return
        if not self.image_label_pairs:
            self.viewer.status = "No image-label pair loaded."
            return

        # Undo the previous preview so each call reflects only current settings.
        self._undo_low_intensity(labels_layer)

        if threshold <= 0:
            self.viewer.status = "Showing all tracks (threshold 0)."
            return

        stats = self._measure_track_intensities(labels_layer, channel)
        if stats is None:
            return  # status already set
        if not stats["counts"]:
            self.viewer.status = "No labels found to measure."
            return

        brightness = self._track_brightness(stats, percentile)
        mapping = {
            label_id: 0
            for label_id, norm in brightness.items()
            if norm < threshold
        }
        n_tracks = len(brightness)

        if not mapping:
            self.viewer.status = (
                f"No tracks below normalized intensity {threshold:.2f} "
                f"({n_tracks} track(s) measured)."
            )
            return

        self._apply_low_intensity(labels_layer, mapping)
        ids = sorted(mapping)
        id_info = f" (IDs {ids})" if len(ids) <= 8 else ""
        self.viewer.status = (
            f"Deleted {len(ids)} of {n_tracks} track(s) with median "
            f"intensity < {threshold:.2f}{id_info}. "
            "Save to write changes to disk."
        )

    def _apply_low_intensity(self, labels_layer, mapping: dict) -> None:
        """Apply a low-intensity deletion and remember how to reverse it."""
        data = labels_layer.data
        backup = (
            np.array(data, copy=True)
            if not isinstance(data, _DaskFancyIndexWrapper)
            and not _is_dask_array(data)
            else None
        )
        self._remap_all_timepoints(labels_layer, mapping)
        self._low_intensity_last = {"mapping": mapping, "backup": backup}

    def _undo_low_intensity(self, labels_layer=None) -> None:
        """Reverse the most recent low-intensity deletion, if still pending.

        Dask-backed labels are reverted via the wrapper's undo log, but only
        when the deletion is still the most recent operation — if the user
        made other edits since, reversing it could corrupt them, so it is
        left in place.  Plain-numpy labels are restored from the snapshot
        taken when the deletion was applied.
        """
        info = self._low_intensity_last
        if info is None:
            return
        self._low_intensity_last = None
        layer = labels_layer or self._find_labels_layer()
        if layer is None:
            return
        data = layer.data
        if isinstance(data, _DaskFancyIndexWrapper):
            if data._op_log and data._op_log[-1][0] == info["mapping"]:
                data.undo_remap()
                layer.refresh()
                # Reversing a deletion is not itself a value map (0 maps
                # back to many IDs), so the view rebuilds from the source.
                self._refresh_track_view()
        elif info["backup"] is not None:
            data[...] = info["backup"]
            layer.refresh()
            self._refresh_track_view()

    def _invalidate_track_stats(self) -> None:
        """Drop cached per-track intensity stats and any pending preview.

        Called when the loaded pair changes or the labels are edited by
        another tool, so a subsequent low-intensity preview re-measures.
        Also drops the pending single-timepoint undo record — its coords
        would be stale on another pair.  (The single-T click tools set
        their record *after* calling this.)
        """
        self._track_stats = None
        self._low_intensity_last = None
        self._single_t_last = None

    def _on_undo_key(self, _provider=None):
        """Ctrl+Z while click-to-delete / click-to-relabel mode is on.

        Undoes the most recent all-T deletion or relabel; when there is
        none pending, falls through to napari's own paint undo.  Saved
        operations cannot be undone (the save is the undo barrier).

        Bound on both the viewer and the Labels layer, so it receives the
        key regardless of which layer is currently active; the labels layer
        is resolved here rather than taken from the provider argument.
        """
        layer = self._find_labels_layer()
        if layer is None:
            return
        data = getattr(layer, "data", None)
        # Single-timepoint click edits are recorded like paint strokes, not
        # LUT ops; when one is the most recent operation (all-T remaps clear
        # the record), write its voxels back the same way.
        info = self._single_t_last
        if info is not None and data is not None:
            self._single_t_last = None
            t = info["t"]
            for coords, old_id in info["restores"]:
                # t is None only for labels with no time axis, where the
                # spatial coords already address the whole array.
                data[coords if t is None else (t, *coords)] = old_id
            layer.refresh()
            self._refresh_track_view(timepoint=t)
            self.viewer.status = f"Undo: label {info['desc']}" + (
                "." if t is None else f" at timepoint {t}."
            )
            return
        if isinstance(data, _DaskFancyIndexWrapper):
            mapping = data.undo_remap()
            if mapping:
                layer.refresh()
                # The undone mapping is not its own inverse, so the track
                # view rebuilds from the (already reverted) source.
                self._refresh_track_view()
                desc = ", ".join(
                    (
                        f"{k} restored (deletion undone)"
                        if v == 0
                        else f"{k}→{v} reverted"
                    )
                    for k, v in mapping.items()
                )
                self.viewer.status = f"Undo: label {desc}."
                return
        # Paint history lives on the layer that was painted — the track
        # view when it is active, the regular labels layer otherwise.
        undo_layer = (
            self._track_view_layer
            if self._track_view_layer is not None
            else layer
        )
        undo = getattr(undo_layer, "undo", None)
        if callable(undo):
            undo()

    def _bind_undo_key(self, layer, bind: bool) -> None:
        """(Un)bind Ctrl+Z for delete-undo.

        napari only dispatches a key to the *active* layer's keymap, so a
        layer-only binding silently stops working whenever another layer
        (e.g. an image channel) becomes active.  Bind on the viewer too —
        it is always in the keymap chain — while keeping the layer binding
        so it shadows napari's native paint-undo when Labels is active.
        Passing ``None`` unbinds, restoring napari's default undo.
        """
        handler = self._on_undo_key if bind else None
        with contextlib.suppress(Exception):
            self.viewer.bind_key("Control-Z", handler, overwrite=True)
        for lyr in (layer, self._track_view_layer):
            if lyr is not None and hasattr(lyr, "bind_key"):
                with contextlib.suppress(Exception):
                    lyr.bind_key("Control-Z", handler, overwrite=True)

    def _make_click_callback(self, on_click):
        """Build a mouse-drag callback that fires *on_click(layer, label_id,
        event)* only for a clean left-click (no drag — panning is unaffected),
        with the clicked label value already resolved (None for off-canvas).
        """

        def _cb(viewer, event):
            if event.button != 1:  # left button only
                return
            dragged = False
            yield  # wait out the drag: only a clean click fires
            while event.type == "mouse_move":
                dragged = True
                yield
            if dragged:
                return
            layer = self._click_value_layer()
            if layer is None:
                return
            try:
                val = layer.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                )
            except TypeError:  # older napari signature
                val = layer.get_value(event.position, world=True)
            on_click(layer, None if val is None else int(val), event)

        return _cb

    def _on_click_delete(self, layer, label_id, event):
        if not label_id:  # None (off-canvas) or 0 (background)
            self.viewer.status = (
                "Click-to-delete: background clicked, nothing removed."
            )
            return
        if self.delete_scope != "current":
            self.delete_label_all_timepoints(label_id)
            return
        t = self._clicked_timepoint(layer, label_id, event)
        if t is None:
            self.viewer.status = (
                "Click-to-delete: could not resolve the clicked timepoint."
            )
            return
        self.delete_label_at_timepoint(label_id, t)

    def _on_click_relabel(self, layer, label_id, event):
        # Real events carry vispy Key objects, whose str() is "<Key 'Control'>";
        # `in` compares via Key.__eq__, which matches both Keys and plain strings.
        modifiers = getattr(event, "modifiers", None) or ()
        if "Control" in modifiers:
            # Pipette: pick up the clicked label as the target ID.
            if not label_id:
                self.viewer.status = (
                    "Click-to-relabel: background clicked, no ID picked."
                )
                return
            layer.selected_label = label_id
            self.viewer.status = (
                f"Picked label {label_id} — plain-click other labels "
                f"to relabel them to {label_id}."
            )
            return
        if not label_id:
            self.viewer.status = (
                "Click-to-relabel: background clicked, nothing relabeled."
            )
            return
        if self.relabel_scope != "current":
            self.relabel_label_all_timepoints(
                label_id, int(layer.selected_label)
            )
            return
        t = self._clicked_timepoint(layer, label_id, event)
        if t is None:
            self.viewer.status = (
                "Click-to-relabel: could not resolve the clicked timepoint."
            )
            return
        self.relabel_label_at_timepoint(label_id, int(layer.selected_label), t)

    def enable_click_delete(self, enabled: bool) -> None:
        """Toggle click-to-delete mode (mutually exclusive with relabel).

        While enabled, a plain left-click (no drag — panning is unaffected)
        on a label deletes that label — from **all** timepoints via the fast
        LUT path, or only the clicked timepoint when ``delete_scope`` is
        ``"current"`` — and Ctrl+Z undoes the last deletion.  This replaces
        napari's bucket-with-ndim-4 workflow, which materializes the entire
        array before filling.
        """
        if enabled and self._click_delete_cb is None:
            self.enable_click_relabel(False)
            self.enable_click_split(False)
            self.enable_click_merge(False)
            self.enable_click_grow(False)
            self.enable_click_add(False)
            self._click_delete_cb = self._make_click_callback(
                self._on_click_delete
            )
            self.viewer.mouse_drag_callbacks.append(self._click_delete_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-delete ON: click a label to remove it — from all "
                "timepoints or only the clicked one, per 'Apply to' "
                "(Ctrl+Z undoes the last deletion)."
            )
        elif not enabled and self._click_delete_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_delete_cb)
            self._click_delete_cb = None
            if (
                self._click_relabel_cb is None
                and self._click_split_cb is None
                and self._click_merge_cb is None
                and self._click_grow_cb is None
                and self._click_add_cb is None
            ):
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-delete OFF."

    def enable_click_relabel(self, enabled: bool) -> None:
        """Toggle click-to-relabel mode (mutually exclusive with delete).

        While enabled, a plain left-click on a label assigns it the layer's
        currently selected label ID (merging it into any existing label
        with that ID) — on **all** timepoints via the fast LUT path, or
        only the clicked timepoint when ``relabel_scope`` is ``"current"``.
        Ctrl+left-click pipettes: it sets the selected label ID from the
        clicked label instead of relabeling.  Ctrl+Z undoes the last
        relabel.
        """
        if enabled and self._click_relabel_cb is None:
            self.enable_click_delete(False)
            self.enable_click_split(False)
            self.enable_click_merge(False)
            self.enable_click_grow(False)
            self.enable_click_add(False)
            self._click_relabel_cb = self._make_click_callback(
                self._on_click_relabel
            )
            self.viewer.mouse_drag_callbacks.append(self._click_relabel_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-relabel ON: Ctrl+click a label to pick up its ID, "
                "then click labels to relabel them to it — on all "
                "timepoints or only the clicked one, per 'Apply to' "
                "(Ctrl+Z undoes the last relabel)."
            )
        elif not enabled and self._click_relabel_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_relabel_cb)
            self._click_relabel_cb = None
            if (
                self._click_delete_cb is None
                and self._click_split_cb is None
                and self._click_merge_cb is None
                and self._click_grow_cb is None
                and self._click_add_cb is None
            ):
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-relabel OFF."

    # ------------------------------------------------------------------
    # Click-to-split (spatial split of an under-segmented label)
    # ------------------------------------------------------------------
    def _global_max_id(self, labels_layer) -> int:
        """Largest label ID anywhere in the current pair's labels.

        Read from the base data (all timepoints); for wrapper-backed
        labels this maxes over the LUT-baked dask array, which is the
        original geometry — split IDs are tracked in ``_next_free_id``,
        so they never need re-scanning.
        """
        data = labels_layer.data
        if isinstance(data, _DaskFancyIndexWrapper):
            base = data._arr
            if hasattr(base, "compute"):
                return int(base.max().compute())
            return int(np.max(np.asarray(base)))
        try:
            import dask.array as da
        except ImportError:
            da = None
        if da is not None and isinstance(data, da.Array):
            return int(data.max().compute())
        return int(np.max(data))

    def _allocate_new_id(self, labels_layer) -> int:
        """Return the next globally-unique label ID and advance the counter.

        Shared by every tool that creates a label (split, add).  Initialized
        lazily per pair to ``global_max + 1`` so the first allocation costs
        one scan and later ones are O(1); monotonic, so intervening deletes /
        relabels (which only lower the max) can never make a later
        allocation reuse a live ID.
        """
        if self._next_free_id is None:
            self._next_free_id = self._global_max_id(labels_layer) + 1
        new_id = self._next_free_id
        self._next_free_id += 1
        return new_id

    def _click_data_coord(self, layer, label_id, event):
        """(t, spatial_coord) of the clicked voxel, or None.

        Only for labels with a time axis (axis 0), as the split writes and
        single-timepoint undo are keyed by ``t``.  In 2-D display (the usual
        way to split touching cells) ``world_to_data`` resolves the click
        exactly; the non-displayed axes come from the dims sliders.  In 3-D
        display ``world_to_data`` of the cursor lands on the near plane, not
        the clicked voxel, so the pick ray is marched to the first voxel of
        *label_id* — the same voxel napari's ``get_value`` returned.
        """
        data = getattr(layer, "data", None)
        if getattr(data, "ndim", 0) < 3:
            return None
        shape = np.asarray(data.shape, dtype=np.intp)
        dims_displayed = list(getattr(event, "dims_displayed", None) or [])
        if len(dims_displayed) == 3:
            vox = self._ray_hit_voxel(layer, label_id, event, dims_displayed)
            if vox is None:
                return None
            coord = np.clip(np.asarray(vox, dtype=np.intp), 0, shape - 1)
        else:
            try:
                full = np.asarray(
                    layer.world_to_data(event.position), dtype=float
                )
            except Exception:
                return None
            if full.shape[0] != shape.shape[0]:
                return None
            coord = np.clip(np.rint(full).astype(np.intp), 0, shape - 1)
        return int(coord[0]), tuple(int(c) for c in coord[1:])

    def split_label_at_timepoint(self, label_id, t, seeds) -> None:
        """Split the *label_id* region at timepoint *t* by *seeds*.

        A seeded watershed on the distance transform of the label's mask
        (its full spatial sub-volume at *t*) divides it into one region per
        seed voxel, cut at the constrictions between them: the first seed's
        region keeps *label_id*, each remaining seed's region becomes a
        fresh globally-unique ID.  Two or more distinct seeds are required
        — pass several to separate a cluster of merged cells in one go.  The
        edit is a single-timepoint pixel write, recorded like a paint stroke
        so it saves normally and Ctrl+Z merges every region back.
        """
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return
        label_id = int(label_id)
        if label_id == 0:
            self.viewer.status = "Select a non-background label to split."
            return

        seeds = list(dict.fromkeys(tuple(s) for s in seeds))  # unique, ordered
        if len(seeds) < 2:
            self.viewer.status = (
                "Split: place at least two distinct points on the label."
            )
            return

        data = labels_layer.data
        try:
            import dask.array as da
        except ImportError:
            da = None
        if da is not None and isinstance(data, da.Array):
            data = _DaskFancyIndexWrapper(data)
            labels_layer.data = data

        t = int(t)
        if isinstance(data, _DaskFancyIndexWrapper):
            sub = data._get_outer_slice(0, t)
        else:
            sub = np.asarray(data[t])

        if any(sub[s] != label_id for s in seeds):
            self.viewer.status = (
                f"Split: every point must land on label {label_id} at "
                f"timepoint {t} — one missed it."
            )
            return

        n_new = len(seeds) - 1
        if self._next_free_id is None:
            self._next_free_id = self._global_max_id(labels_layer) + 1
        if (
            np.issubdtype(data.dtype, np.integer)
            and self._next_free_id + n_new - 1 > np.iinfo(data.dtype).max
        ):
            self.viewer.status = (
                f"Split: label dtype {np.dtype(data.dtype)} cannot hold "
                f"{n_new} more ID(s) (max {np.iinfo(data.dtype).max}). "
                "Convert the labels to a wider integer type first."
            )
            return

        from scipy import ndimage as ndi
        from skimage.segmentation import watershed

        blob = sub == label_id
        dist = ndi.distance_transform_edt(blob)
        markers = np.zeros(sub.shape, dtype=np.int32)
        for i, s in enumerate(seeds):
            markers[s] = i + 1
        ws = watershed(-dist, markers, mask=blob)

        # Marker 1 keeps label_id; markers 2..N each become a new ID.
        restores = []
        new_ids = []
        for marker in range(2, len(seeds) + 1):
            coords = np.nonzero(ws == marker)
            if not coords[0].size:
                continue
            new_id = self._allocate_new_id(labels_layer)
            if isinstance(data, _DaskFancyIndexWrapper):
                data[(t, *coords)] = new_id
            else:
                sub[coords] = new_id
            restores.append((coords, label_id))
            new_ids.append(new_id)

        if not new_ids:
            self.viewer.status = (
                "Split: watershed produced no new regions — try seeds "
                "farther apart."
            )
            return

        labels_layer.refresh()
        self._refresh_track_view(timepoint=t)
        # A manual edit changes track geometry — re-measure on next preview.
        # (Also clears any older single-T undo, so set our record after.)
        self._invalidate_track_stats()
        ids_str = ", ".join(str(i) for i in new_ids)
        self._single_t_last = {
            "t": t,
            "restores": restores,
            "desc": (
                f"{len(new_ids)} region(s) ({ids_str}) merged back into "
                f"{label_id} (split undone)"
            ),
        }
        self.viewer.status = (
            f"Split label {label_id} at timepoint {t} into "
            f"{len(new_ids) + 1} region(s): new label(s) {ids_str}. "
            "Ctrl+Z merges them back; 'Save Changes and Continue' writes "
            "them to disk."
        )

    def _on_click_split(self, layer, label_id, event):
        """Place (or remove) a split seed; the split runs on 'Apply split'.

        Plain left-click adds one seed per cell of a merged label;
        Ctrl+left-click removes the most recently placed seed.  A click on a
        different label or timepoint starts a fresh seed set.  Nothing is
        written until :meth:`commit_split` (the Apply button) runs.
        """
        if self._track_view_layer is not None:
            self._split_seeds = None
            self.viewer.status = (
                "Split works only in the normal frame view — set Track "
                "view to 'Off' first."
            )
            return
        modifiers = getattr(event, "modifiers", None) or ()
        if "Control" in modifiers:
            active = self._split_seeds
            if active and active["coords"]:
                active["coords"].pop()
                if not active["coords"]:
                    self._split_seeds = None
                    self.viewer.status = "Split: all seeds removed."
                else:
                    self.viewer.status = (
                        f"Split: removed last seed, {len(active['coords'])} "
                        "left."
                    )
            else:
                self.viewer.status = "Split: no seed to remove."
            return
        if not label_id:  # None (off-canvas) or 0 (background)
            self.viewer.status = "Split: background clicked, no seed placed."
            return
        resolved = self._click_data_coord(layer, label_id, event)
        if resolved is None:
            self.viewer.status = (
                "Split: could not resolve the clicked voxel (needs a label "
                "movie with a time axis)."
            )
            return
        t, coord = resolved

        active = self._split_seeds
        if (
            active is None
            or active["label_id"] != label_id
            or active["t"] != t
        ):
            active = {"label_id": label_id, "t": t, "coords": []}
            self._split_seeds = active
        if coord not in active["coords"]:
            active["coords"].append(coord)
        self.viewer.status = (
            f"Split: {len(active['coords'])} seed(s) on label {label_id} "
            f"(t={t}). Click one point per cell, then press 'Apply split' "
            "(Ctrl+click removes the last seed)."
        )

    def commit_split(self) -> None:
        """Run the watershed split on the seeds placed so far (Apply button)."""
        active = self._split_seeds
        if not active or len(active["coords"]) < 2:
            self.viewer.status = (
                "Split: click at least two points (one per cell) on a "
                "label first."
            )
            return
        if self._track_view_layer is not None:
            self.viewer.status = (
                "Split works only in the normal frame view — set Track "
                "view to 'Off' first."
            )
            return
        label_id = active["label_id"]
        t = active["t"]
        seeds = list(active["coords"])
        self._split_seeds = None
        self.split_label_at_timepoint(label_id, t, seeds)

    def enable_click_split(self, enabled: bool) -> None:
        """Toggle click-to-split mode (mutually exclusive with delete /
        relabel).

        While enabled, click two points inside one under-segmented label to
        divide it — at the clicked timepoint only — with a seeded watershed;
        the split-off part gets a fresh globally-unique ID.  Ctrl+Z merges
        the last split back.  Track view must be off (a split needs precise
        source voxels, which the projected views don't provide).
        """
        if enabled and self._click_split_cb is None:
            self.enable_click_delete(False)
            self.enable_click_relabel(False)
            self.enable_click_merge(False)
            self.enable_click_grow(False)
            self.enable_click_add(False)
            self._split_seeds = None
            self._click_split_cb = self._make_click_callback(
                self._on_click_split
            )
            self.viewer.mouse_drag_callbacks.append(self._click_split_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-split ON: click one point inside each cell of a "
                "merged label, then press 'Apply split' (Ctrl+click removes "
                "the last seed). Each part after the first gets a new ID; "
                "Ctrl+Z merges the split back. Turn Track view off to use it."
            )
        elif not enabled and self._click_split_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_split_cb)
            self._click_split_cb = None
            self._split_seeds = None
            if (
                self._click_delete_cb is None
                and self._click_relabel_cb is None
                and self._click_merge_cb is None
                and self._click_grow_cb is None
                and self._click_add_cb is None
            ):
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-split OFF."

    # ------------------------------------------------------------------
    # Click-to-merge-neighbors (fuse an over-segmented cell's fragments)
    # ------------------------------------------------------------------
    def _touching_neighbors(self, labels_layer, label_id, t) -> list:
        """IDs sharing a border with *label_id* on the spatial slice at *t*.

        Adjacency is full connectivity — a shared face, edge or corner all
        count as touching.  Empty when the layer, the label or any
        neighbor is missing; the viewer status says which.
        """
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return []
        label_id = int(label_id)
        if label_id == 0:
            self.viewer.status = "Select a non-background label to merge into."
            return []

        data = labels_layer.data
        try:
            import dask.array as da
        except ImportError:
            da = None
        if da is not None and isinstance(data, da.Array):
            data = _DaskFancyIndexWrapper(data)
            labels_layer.data = data

        t = int(t)
        if isinstance(data, _DaskFancyIndexWrapper):
            sub = data._get_outer_slice(0, t)
        elif getattr(data, "ndim", 0) < 3:
            sub = np.asarray(data)  # no time axis — whole array is the slice
        else:
            sub = np.asarray(data[t])

        mask = sub == label_id
        if not mask.any():
            self.viewer.status = (
                f"Label {label_id} not present at timepoint {t}."
            )
            return []

        from scipy import ndimage as ndi

        # The one-pixel dilation band around the label collects every
        # neighboring ID in a single vectorized read — and every voxel
        # that can touch the label lies inside its bounding box grown by
        # one, so the dilation runs on that box rather than on the whole
        # (possibly gigavoxel) spatial slice.
        where = np.nonzero(mask)
        box = tuple(
            slice(max(int(c.min()) - 1, 0), min(int(c.max()) + 2, n))
            for c, n in zip(where, mask.shape)
        )
        sub_mask = mask[box]
        structure = ndi.generate_binary_structure(mask.ndim, mask.ndim)
        border = ndi.binary_dilation(sub_mask, structure=structure) & ~sub_mask
        neighbors = [
            int(n)
            for n in np.unique(sub[box][border])
            if int(n) != 0 and int(n) != label_id
        ]
        if not neighbors:
            self.viewer.status = (
                f"Label {label_id} has no touching neighbors at "
                f"timepoint {t}."
            )
        return neighbors

    def merge_neighbors_all_timepoints(self, label_id, t) -> None:
        """Merge the labels touching *label_id* at *t* into it, on **all**
        timepoints.

        Which labels touch is still decided on one slice — the clicked
        timepoint — and those IDs are then merged across the whole movie.
        That is the right unit for tracked data, where a label ID is one
        object's trajectory: an over-segmented cell keeps the same
        fragment IDs frame to frame, so one click repairs the entire
        track instead of one frame of it.  It is deliberately *not* a
        per-frame re-detection: neighbors are only ever taken from the
        frame you looked at, so a merge can never quietly swallow a cell
        that happens to brush past at some other timepoint.

        Runs through the value-remap LUT, so it costs no I/O however long
        the movie is, and Ctrl+Z reverts the whole merge in one step.
        """
        labels_layer = self._find_labels_layer()
        neighbors = self._touching_neighbors(labels_layer, label_id, t)
        if not neighbors:
            return
        label_id, t = int(label_id), int(t)

        self._remap_all_timepoints(
            labels_layer, dict.fromkeys(neighbors, label_id)
        )
        self._invalidate_track_stats()
        ids_str = ", ".join(str(n) for n in neighbors)
        self._single_t_last = None
        self.viewer.status = (
            f"Merged {len(neighbors)} neighbor(s) ({ids_str}) into label "
            f"{label_id} on all timepoints (touching measured at t={t}). "
            "Ctrl+Z reverts; 'Save Changes and Continue' writes it to disk."
        )

    def merge_neighbors_at_timepoint(self, label_id, t) -> None:
        """Merge every label touching *label_id* at timepoint *t* into it.

        For over-segmented cells split across several IDs: click one
        fragment and all labels sharing a border with it — at the clicked
        timepoint only — are relabeled to the clicked ID.  Adjacency is
        full connectivity (a shared face, edge or corner counts), computed
        on the label's spatial slice at *t*.  Only direct neighbors merge,
        not a neighbor's neighbors, so re-clicking the (now larger) label
        grows the region another ring outward.  The edit is a
        single-timepoint pixel write, recorded like a paint stroke so it
        saves normally and Ctrl+Z reverts the merge.

        See :meth:`merge_neighbors_all_timepoints` for the same merge
        applied to the whole movie.
        """
        labels_layer = self._find_labels_layer()
        neighbors = self._touching_neighbors(labels_layer, label_id, t)
        if not neighbors:
            return
        label_id, t = int(label_id), int(t)

        mapping = dict.fromkeys(neighbors, label_id)
        restores = self._remap_one_timepoint(labels_layer, mapping, t)
        self._invalidate_track_stats()
        ids_str = ", ".join(str(n) for n in neighbors)
        if restores == "all":
            self.viewer.status = (
                f"Merged neighbor(s) {ids_str} into label {label_id} "
                "(no time axis — whole image). Save to write changes to disk."
            )
            return
        self._single_t_last = {
            "t": int(t),
            "restores": restores,
            "desc": (
                f"{len(neighbors)} neighbor(s) ({ids_str}) split back out "
                f"of {label_id} (merge undone)"
            ),
        }
        self.viewer.status = (
            f"Merged {len(neighbors)} neighbor(s) ({ids_str}) into label "
            f"{label_id} at timepoint {t}. Ctrl+Z reverts; 'Save Changes "
            "and Continue' writes it to disk."
        )

    def _on_click_merge(self, layer, label_id, event):
        """Merge every label touching the clicked one — at the clicked
        timepoint, or across the movie when ``merge_scope`` is ``"all"``.

        Works in the track views too: only the clicked ID and timepoint
        come from the view, and adjacency is then computed on the source's
        full-resolution spatial slice at that timepoint — so a subsampled
        or Z-projected view never affects which labels are found to touch.
        """
        if not label_id:  # None (off-canvas) or 0 (background)
            self.viewer.status = (
                "Merge neighbors: background clicked, nothing merged."
            )
            return
        data = getattr(layer, "data", None)
        if getattr(data, "ndim", 0) < 3:
            t = 0  # no time axis — merge acts on the whole array
        else:
            t = self._clicked_timepoint(layer, label_id, event)
            if t is None:
                self.viewer.status = (
                    "Merge neighbors: could not resolve the clicked "
                    "timepoint."
                )
                return
        if self.merge_scope != "current":
            self.merge_neighbors_all_timepoints(label_id, t)
            return
        self.merge_neighbors_at_timepoint(label_id, t)

    def enable_click_merge(self, enabled: bool) -> None:
        """Toggle click-to-merge-neighbors mode (mutually exclusive with
        delete / relabel / split).

        While enabled, a plain left-click on a label merges every label
        touching it into the clicked ID, fusing an over-segmented cell's
        fragments in one click — at the clicked timepoint, or across the
        whole movie when ``merge_scope`` is ``"all"``.  Ctrl+Z reverts the
        last merge.  Works in the track views as well: they supply the
        clicked ID and timepoint, and adjacency is measured on the
        source's full-resolution slice at that timepoint.
        """
        if enabled and self._click_merge_cb is None:
            self.enable_click_delete(False)
            self.enable_click_relabel(False)
            self.enable_click_split(False)
            self.enable_click_grow(False)
            self.enable_click_add(False)
            self._click_merge_cb = self._make_click_callback(
                self._on_click_merge
            )
            self.viewer.mouse_drag_callbacks.append(self._click_merge_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-merge-neighbors ON: click a label to merge every "
                "label touching it into it — at the clicked timepoint or "
                "across the movie, per 'Apply to'. Re-click to grow "
                "another ring outward; Ctrl+Z reverts."
            )
        elif not enabled and self._click_merge_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_merge_cb)
            self._click_merge_cb = None
            if (
                self._click_delete_cb is None
                and self._click_relabel_cb is None
                and self._click_split_cb is None
                and self._click_grow_cb is None
                and self._click_add_cb is None
            ):
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-merge-neighbors OFF."

    # ------------------------------------------------------------------
    # Click-to-grow (extend an under-segmented label out to the signal)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_uint8_rgb(plane: np.ndarray) -> np.ndarray:
        """Contrast-stretch a raw plane into the uint8 RGB SAM2 expects.

        SAM2 was trained on 8-bit natural images, so a 12- or 16-bit
        microscopy plane has to be mapped into that range first.  The stretch
        is a 1st–99.5th percentile window over the crop being segmented, which
        keeps a dim cell legible without letting a few hot pixels flatten it.
        """
        arr = np.asarray(plane, dtype=np.float32)
        lo, hi = (float(v) for v in np.percentile(arr, (1.0, 99.5)))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape + (3,), dtype=np.uint8)
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    @staticmethod
    def _prompt_points(seed, lbl, label_id, max_negative: int = 6):
        """Point prompts describing "this object, not its neighbors".

        Positive points are the most interior pixels of *seed* — the distance
        transform's peaks, spread out so they sample it rather than crowding
        one spot.  *seed* is the existing label when growing one, and the
        clicked pixel when segmenting a cell that has no label yet.  Each
        neighboring label contributes one negative point, nearest first, which
        is what stops SAM2 from returning a whole clump of touching cells as a
        single object.  Pass ``label_id=0`` to make *every* label in the crop
        a neighbor.

        Note there is deliberately no box prompt: a box drawn around an
        under-segmented label would tell SAM2 the object ends at the label's
        current edge, re-creating the very problem this tool exists to fix.

        Returns ``(coords, labels)`` in SAM2's (x, y) order.
        """
        from scipy import ndimage as ndi

        coords, labels = [], []
        dist = ndi.distance_transform_edt(seed)
        yy, xx = np.ogrid[: seed.shape[0], : seed.shape[1]]
        for _ in range(3):
            peak = float(dist.max())
            if peak <= 0:
                break
            y, x = np.unravel_index(np.argmax(dist), dist.shape)
            coords.append([float(x), float(y)])
            labels.append(1)
            # Blank a disc around the accepted point so the next peak is a
            # genuinely different part of the label.
            radius = max(peak, 3.0)
            dist = np.where(
                (yy - y) ** 2 + (xx - x) ** 2 <= radius**2, 0.0, dist
            )

        others = np.unique(lbl[(lbl != 0) & (lbl != label_id)])
        if others.size > max_negative:
            # A crowded crop holds more neighbors than the prompt should
            # carry; the ones worth spending a point on are those the object
            # could be confused with, i.e. the closest ones.
            to_seed = ndi.distance_transform_edt(~seed)
            gap = np.asarray(ndi.minimum(to_seed, lbl, others), dtype=float)
            others = others[np.argsort(gap, kind="stable")]
        for other in others[:max_negative]:
            other_mask = lbl == other
            other_dist = ndi.distance_transform_edt(other_mask)
            y, x = np.unravel_index(np.argmax(other_dist), other_dist.shape)
            coords.append([float(x), float(y)])
            labels.append(0)

        return (
            np.asarray(coords, dtype=np.float32),
            np.asarray(labels, dtype=np.int32),
        )

    def _sam2_plane_mask(
        self, worker, seed, lbl, raw, label_id, max_px, must_cover=None
    ):
        """SAM2's mask for the object *seed* belongs to, on one plane.

        *seed* is whatever is already known to lie inside the object — the
        existing label when growing one, a single clicked pixel (with
        ``label_id=0``) when segmenting a cell that has none yet, the
        previous plane's cross-section when tracing one through Z.  It both
        supplies the prompt points and, by default, decides which candidate
        won; pass *must_cover* to judge the candidates against something
        smaller than the prompt, which is what a tapering cell needs — its
        next cross-section can be well under half of the prompt's area
        without the mask having drifted anywhere.

        The plane is cropped to the seed plus *max_px* of context before
        being sent.  That is not only cheaper: SAM2 rescales whatever it is
        given to 1024x1024, so cropping to the cell raises its effective
        resolution considerably compared with sending a full frame.

        Two passes are used.  The first offers three candidate masks at
        different scales; the one that best covers the seed wins.  The second
        feeds that candidate back as a mask prompt, which is what pulls the
        contour out to the true edge — on a synthetic cell it took the
        recovered area from 93%% to 99.6%% of ground truth, and
        under-reaching is exactly the failure being fixed here.
        """
        ys, xs = np.nonzero(seed)
        pad = int(max_px) + 8
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, seed.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, seed.shape[1])
        crop = (slice(y0, y1), slice(x0, x1))

        sub_seed = seed[crop]
        image = self._to_uint8_rgb(raw[crop])
        coords, point_labels = self._prompt_points(
            sub_seed, lbl[crop], label_id
        )
        if not coords.size:
            return np.zeros_like(seed)

        target = sub_seed if must_cover is None else must_cover[crop]
        masks, scores = worker.segment(image, coords, point_labels)
        target_area = float(target.sum())
        cover = np.array(
            [float((m & target).sum()) / target_area for m in masks]
        )
        # Only a candidate that actually contains what is known to be inside
        # the object can be that object; among those, trust SAM2's own score.
        usable = cover >= 0.5
        index = int(
            np.argmax(np.where(usable, scores, -np.inf))
            if usable.any()
            else np.argmax(cover)
        )

        best = masks[index]
        refined, _score = worker.refine(index, coords, point_labels)
        # A refinement that loses the object has drifted onto something else;
        # keep the unrefined candidate rather than the wrong object.
        if float((refined & target).sum()) >= 0.5 * target_area:
            best = refined

        out = np.zeros_like(seed)
        out[crop] = best
        return out

    def _constrain_region(self, region, seed, free, max_px, smooth_px):
        """Make a raw model mask safe to write: no theft, no islands, no shrink.

        SAM2 knows nothing about the other labels, so its mask is intersected
        with *free* (background plus this label), clipped to *max_px* of the
        label as a runaway guard, reduced to the component holding the label,
        and unioned back with it — the tool only ever grows.
        """
        from scipy import ndimage as ndi

        region = (region | seed) & free
        if max_px > 0:
            region &= ndi.distance_transform_edt(~seed) <= max_px
        if smooth_px > 0:
            structure = ndi.generate_binary_structure(region.ndim, 1)
            it = int(round(smooth_px))
            # Closing then opening, spelled out because the erosions need
            # ``border_value=1``: the cell continues past the edge of the
            # crop, so treating the outside as background would shave the
            # boundary. The dilations keep the default 0 so they cannot
            # invent foreground at the border either.
            region = ndi.binary_dilation(
                region, structure=structure, iterations=it
            )
            region = ndi.binary_erosion(
                region, structure=structure, iterations=2 * it, border_value=1
            )
            region = ndi.binary_dilation(
                region, structure=structure, iterations=it
            )
            region = (region | seed) & free
        region = ndi.binary_fill_holes(region) & free

        labeled, _n = ndi.label(region)
        keep = np.unique(labeled[seed])
        keep = keep[keep != 0]
        return np.isin(labeled, keep) if keep.size else seed

    def grow_label_at_timepoint(self, label_id, t) -> None:
        """Extend *label_id* out to the boundaries of the cell at *t*.

        For under-segmented cells — the label covers only part of the object,
        so it needs to be pushed outward rather than split or merged.  SAM2
        supplies the boundary: the existing label becomes a point prompt and
        the model returns the whole object it belongs to (see
        :meth:`_sam2_plane_mask`), which is then constrained so it can never
        take pixels from a neighboring label (:meth:`_constrain_region`).

        SAM2 segments one plane at a time, so this runs independently on every
        Z-plane where the label appears — each plane gets its boundary from
        its own in-plane image, which suits anisotropic stacks.  The label
        keeps one ID, so the planes need no stitching.

        Only background pixels are ever claimed, and only at the clicked
        timepoint.  The edit is a single-timepoint pixel write, recorded like
        a paint stroke, so it saves normally and Ctrl+Z removes exactly the
        pixels it added.
        """
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return
        label_id = int(label_id)
        if label_id == 0:
            self.viewer.status = "Select a non-background label to grow."
            return

        from napari_tmidas._sam2_worker import Sam2Unavailable, Sam2Worker

        data = labels_layer.data
        try:
            import dask.array as da
        except ImportError:
            da = None
        if da is not None and isinstance(data, da.Array):
            data = _DaskFancyIndexWrapper(data)
            labels_layer.data = data

        has_time = getattr(data, "ndim", 0) >= 3
        t = int(t) if has_time else None
        if isinstance(data, _DaskFancyIndexWrapper) and has_time:
            sub = data._get_outer_slice(0, t)
        elif has_time:
            sub = np.asarray(data[t])  # a view: writes reach the layer
        else:
            sub = np.asarray(data)

        mask = sub == label_id
        if not mask.any():
            self.viewer.status = f"Label {label_id} is not present" + (
                "." if t is None else f" at timepoint {t}."
            )
            return

        raw_t = self._raw_slice_at(data, t, self.grow_channel)
        if raw_t is None:
            return  # status already set

        max_px = max(int(self.grow_max_px), 1)
        smooth_px = max(int(self.grow_smooth_px), 0)
        # Work in a window around the label only — everything the growth can
        # reach lies within max_px of it, plus a margin so the crop SAM2 sees
        # carries context on every side.
        pad = max_px + smooth_px + 8
        where = np.nonzero(mask)
        box = tuple(
            slice(max(int(c.min()) - pad, 0), min(int(c.max()) + pad + 1, n))
            for c, n in zip(where, mask.shape)
        )
        lbl = sub[box]
        raw = np.asarray(raw_t[box], dtype=np.float32)
        seed = lbl == label_id
        free = (lbl == 0) | seed  # never claim another label's pixels

        # Starting the worker loads a ~900 MB checkpoint, so the first click
        # of a session pauses for a few seconds; later ones are ~0.2 s/plane.
        try:
            self.viewer.status = "Grow: contacting SAM2..."
            worker = Sam2Worker.instance()
        except Sam2Unavailable as exc:
            self.viewer.status = f"Grow: {exc}"
            return

        planes = (
            [z for z in range(lbl.shape[0]) if seed[z].any()]
            if lbl.ndim == 3
            else [None]
        )
        grown = np.zeros_like(seed)
        try:
            for i, z in enumerate(planes, start=1):
                if z is None:
                    region = self._sam2_plane_mask(
                        worker, seed, lbl, raw, label_id, max_px
                    )
                    grown[...] = self._constrain_region(
                        region, seed, free, max_px, smooth_px
                    )
                else:
                    self.viewer.status = (
                        f"Grow: SAM2 on Z-plane {i}/{len(planes)}..."
                    )
                    region = self._sam2_plane_mask(
                        worker, seed[z], lbl[z], raw[z], label_id, max_px
                    )
                    grown[z] = self._constrain_region(
                        region, seed[z], free[z], max_px, smooth_px
                    )
        except Sam2Unavailable as exc:
            self.viewer.status = f"Grow: {exc}"
            return

        new = grown & ~seed
        if not new.any():
            self.viewer.status = (
                f"Grow: SAM2 returned nothing beyond label {label_id}'s "
                "current extent — nothing added. Raise 'Max growth' if the "
                "cap is what stopped it."
            )
            return

        offsets = [b.start for b in box]
        coords = tuple(c + o for c, o in zip(np.nonzero(new), offsets))
        index = coords if t is None else (t, *coords)
        if isinstance(data, _DaskFancyIndexWrapper):
            data[index] = label_id
        else:
            sub[coords] = label_id

        labels_layer.refresh()
        self._refresh_track_view(timepoint=t)
        # A manual edit changes the label's geometry — re-measure intensity
        # stats on the next preview. (Also clears any older single-T undo,
        # so our record has to be set after it.)
        self._invalidate_track_stats()
        n_px = int(new.sum())
        self._single_t_last = {
            "t": t,
            "restores": [(coords, 0)],
            "desc": f"{label_id} shrunk back by {n_px} pixel(s) (grow undone)",
        }
        n_planes = len(planes)
        self.viewer.status = (
            f"Grew label {label_id} by {n_px} pixel(s) (SAM2 on "
            f"{n_planes} plane(s)"
            + ("" if t is None else f", timepoint {t}")
            + "). Ctrl+Z undoes it; 'Save Changes and Continue' writes it "
            "to disk."
        )

    def _on_click_grow(self, layer, label_id, event):
        """Grow the clicked label to the signal boundary at its timepoint.

        Like the merge tool this works in the track views as well: only the
        clicked ID and timepoint are taken from the view, and the growth
        itself is computed on the source's full-resolution slice and raw
        image at that timepoint.
        """
        if not label_id:  # None (off-canvas) or 0 (background)
            self.viewer.status = "Grow: background clicked, nothing to grow."
            return
        data = getattr(layer, "data", None)
        if getattr(data, "ndim", 0) < 3:
            t = 0  # no time axis — the whole array is the one timepoint
        else:
            t = self._clicked_timepoint(layer, label_id, event)
            if t is None:
                self.viewer.status = (
                    "Grow: could not resolve the clicked timepoint."
                )
                return
        self.grow_label_at_timepoint(label_id, t)

    def enable_click_grow(self, enabled: bool) -> None:
        """Toggle click-to-grow mode (mutually exclusive with the other
        click tools).

        While enabled, a plain left-click on an under-segmented label
        hands it to SAM2 as a prompt and extends it to the full cell, at the
        clicked timepoint.  Ctrl+Z removes the pixels the last grow added.
        """
        if enabled and self._click_grow_cb is None:
            self.enable_click_delete(False)
            self.enable_click_relabel(False)
            self.enable_click_split(False)
            self.enable_click_merge(False)
            self.enable_click_add(False)
            self._click_grow_cb = self._make_click_callback(
                self._on_click_grow
            )
            self.viewer.mouse_drag_callbacks.append(self._click_grow_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-grow ON: click an under-segmented label and SAM2 "
                "extends it to the whole cell at that timepoint. The first "
                "click loads the model (a few seconds). Ctrl+Z undoes the "
                "last grow."
            )
        elif not enabled and self._click_grow_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_grow_cb)
            self._click_grow_cb = None
            if (
                self._click_delete_cb is None
                and self._click_relabel_cb is None
                and self._click_split_cb is None
                and self._click_merge_cb is None
                and self._click_add_cb is None
            ):
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-grow OFF."

    # ------------------------------------------------------------------
    # Click-to-add (segment a missed cell into a brand-new label)
    # ------------------------------------------------------------------
    # Fewer pixels than this is not a cell — a click on a featureless patch
    # of background is reported rather than written.
    _MIN_NEW_AREA = 4
    # Z propagation stops where the cross-section falls below this fraction
    # of the clicked plane's: a cell tapers off, and what lies past the taper
    # is noise, not more cell.
    _Z_TAPER_STOP = 0.1
    # ... and a plane that suddenly grows by this factor has leaked into its
    # surroundings, so it is dropped and the walk ends there.
    _Z_LEAK_STOP = 3.0
    # The one that actually ends most walks: a plane counts as part of the
    # cell only while its brightness above the local background is at least
    # this fraction of the clicked plane's. Past the cell's last plane a
    # Z-stack still shows its out-of-focus halo, which is real signal and
    # which SAM2 will happily segment plane after plane — area alone shrinks
    # far too gently to tell that cone apart from a tapering cell, while the
    # contrast drops off a cliff (measured on a blurred synthetic sphere:
    # 0.36 of the clicked plane's on its last true plane, 0.19 on the first
    # empty one).
    _Z_FAINT_STOP = 0.3

    def _add_plane_mask(
        self, worker, prompt, point, lbl, raw, max_px, smooth_px
    ):
        """One plane of a new object: prompt SAM2, then make the mask safe.

        *prompt* is what SAM2 is told lies inside the object — the clicked
        pixel on the clicked plane, the previous plane's cross-section while
        propagating through Z — while *point* is the single pixel the result
        must contain and stay within *max_px* of.  Keeping the two apart is
        what lets a propagated plane be prompted with the whole previous
        cross-section without simply inheriting it, since
        :meth:`_constrain_region` unions its seed back in.
        """
        region = self._sam2_plane_mask(
            worker, prompt, lbl, raw, 0, max_px, must_cover=point
        )
        # Only background is free: a missed cell is by definition unlabeled,
        # so anything already carrying an ID belongs to another cell.
        return self._constrain_region(
            region, point, lbl == 0, max_px, smooth_px
        )

    @staticmethod
    def _plane_contrast(raw, lbl, mask):
        """How far *mask* stands out from the unlabeled background around it.

        Medians, so a few saturated pixels or a ragged edge cannot move it,
        and measured per plane, so a stack that dims with depth is judged
        against its own background rather than the clicked plane's.
        """
        background = raw[(lbl == 0) & ~mask]
        if not mask.any() or not background.size:
            return 0.0
        return float(np.median(raw[mask]) - np.median(background))

    def _next_plane_mask(
        self, worker, prev, lbl, raw, max_px, smooth_px, first_area, contrast
    ):
        """The new object's cross-section on the next Z-plane, or None to stop.

        The previous plane's mask, minus whatever is labeled here, is the
        prompt; its most interior pixel anchors the result.  ``None`` ends the
        walk — the plane is only the cell's out-of-focus halo, the cell has
        tapered out, its footprint is taken by another label, or the mask
        leaked.
        """
        from scipy import ndimage as ndi

        prompt = prev & (lbl == 0)
        if not prompt.any():
            return None  # another label owns this plane's footprint
        point = np.zeros_like(prompt)
        dist = ndi.distance_transform_edt(prompt)
        point[np.unravel_index(np.argmax(dist), dist.shape)] = True

        mask = self._add_plane_mask(
            worker, prompt, point, lbl, raw, max_px, smooth_px
        )
        area = float(mask.sum())
        if area < self._Z_TAPER_STOP * first_area:
            return None
        if area > self._Z_LEAK_STOP * float(prev.sum()):
            return None
        here = self._plane_contrast(raw, lbl, mask)
        if here < self._Z_FAINT_STOP * contrast:
            return None  # only the cell's halo reaches this plane
        return mask

    def add_label_at_timepoint(self, coord, t, raw_t=None) -> None:
        """Segment the unlabeled cell at spatial *coord* into a new label at *t*.

        For false negatives — a cell the segmentation missed entirely, so
        there is no label to correct and nothing for the other tools to act
        on.  The clicked pixel is the whole prompt: SAM2 returns the object
        around it, every label in the crop contributes a negative point
        (nearest first), and the result is intersected with background, so a
        cell that got missed *because* it touches a labeled one still cannot
        take a single pixel from its neighbor.  The mask is also confined to
        ``add_max_px`` of the click, which bounds what one click can create.

        On a Z-stack the click resolves one plane only, so the object is
        propagated up and down from there: each plane is prompted with the
        previous plane's cross-section and segmented from its own in-plane
        image, exactly as the grow tool treats an anisotropic stack.  The
        walk stops once a plane holds only the cell's out-of-focus halo
        rather than the cell — by brightness, since the halo is real signal
        that segments perfectly well (see :meth:`_next_plane_mask`).  Set
        ``add_propagate_z`` False to label the clicked plane alone.

        The new cell gets a fresh globally-unique ID, and the write is a
        single-timepoint pixel edit recorded like a paint stroke, so Ctrl+Z
        removes it and 'Save Changes and Continue' writes it to disk.

        Pass *raw_t* to reuse a raw slice the caller has already read (a 3-D
        click needs it to resolve the clicked voxel in the first place), so
        one click costs one read however big the timepoint is.
        """
        labels_layer = self._find_labels_layer()
        if labels_layer is None:
            self.viewer.status = "No labels layer found."
            return

        from napari_tmidas._sam2_worker import Sam2Unavailable, Sam2Worker

        data = labels_layer.data
        try:
            import dask.array as da
        except ImportError:
            da = None
        if da is not None and isinstance(data, da.Array):
            data = _DaskFancyIndexWrapper(data)
            labels_layer.data = data

        has_time = getattr(data, "ndim", 0) >= 3
        t = int(t) if has_time else None
        if isinstance(data, _DaskFancyIndexWrapper) and has_time:
            sub = data._get_outer_slice(0, t)
        elif has_time:
            sub = np.asarray(data[t])  # a view: writes reach the layer
        else:
            sub = np.asarray(data)

        coord = tuple(int(c) for c in coord)
        if len(coord) != sub.ndim or any(
            not 0 <= c < n for c, n in zip(coord, sub.shape)
        ):
            self.viewer.status = (
                "Add: the clicked point is outside the label image."
            )
            return
        occupant = int(sub[coord])
        if occupant:
            self.viewer.status = (
                f"Add: that pixel already belongs to label {occupant} — click "
                "a cell that has no label, or use click-to-grow to extend "
                "this one."
            )
            return

        if raw_t is None:
            raw_t = self._raw_slice_at(data, t, self.add_channel)
        if raw_t is None:
            return  # status already set

        # One more ID has to fit in the label dtype; check before spending
        # any time on inference.
        if self._next_free_id is None:
            self._next_free_id = self._global_max_id(labels_layer) + 1
        if (
            np.issubdtype(data.dtype, np.integer)
            and self._next_free_id > np.iinfo(data.dtype).max
        ):
            self.viewer.status = (
                f"Add: label dtype {np.dtype(data.dtype)} cannot hold another "
                f"ID (max {np.iinfo(data.dtype).max}). Convert the labels to "
                "a wider integer type first."
            )
            return

        max_px = max(int(self.add_max_px), 1)
        smooth_px = max(int(self.add_smooth_px), 0)
        # Work in a window around the click only — the new label cannot reach
        # beyond max_px of it, plus a margin so the crop SAM2 sees carries
        # context on every side. Z is kept whole when propagating, since that
        # is the axis the object is about to be traced along.
        pad = max_px + smooth_px + 8
        box = tuple(
            slice(max(c - pad, 0), min(c + pad + 1, n))
            for c, n in zip(coord[-2:], sub.shape[-2:])
        )
        if sub.ndim == 3:
            z = coord[0]
            z_range = (
                slice(0, sub.shape[0])
                if self.add_propagate_z
                else slice(z, z + 1)
            )
            box = (z_range, *box)
        lbl = np.asarray(sub[box])
        raw = np.asarray(raw_t[box], dtype=np.float32)
        local = tuple(c - b.start for c, b in zip(coord, box))

        # Starting the worker loads a ~900 MB checkpoint, so the first click
        # of a session pauses for a few seconds; later ones are ~0.2 s/plane.
        try:
            self.viewer.status = "Add: contacting SAM2..."
            worker = Sam2Worker.instance()
        except Sam2Unavailable as exc:
            self.viewer.status = f"Add: {exc}"
            return

        z0 = local[0] if lbl.ndim == 3 else None
        seed = np.zeros(lbl.shape[-2:], dtype=bool)
        seed[local[-2:]] = True
        new = np.zeros(lbl.shape, dtype=bool)
        try:
            first = self._add_plane_mask(
                worker,
                seed,
                seed,
                lbl if z0 is None else lbl[z0],
                raw if z0 is None else raw[z0],
                max_px,
                smooth_px,
            )
            first_area = float(first.sum())
            if first_area < self._MIN_NEW_AREA:
                self.viewer.status = (
                    "Add: SAM2 found no object at that point — click nearer "
                    "the middle of the cell, or raise 'Max radius' if the "
                    "cell is bigger than that."
                )
                return
            # How far this cell stands out here is the yardstick every
            # further plane is held to.
            contrast = self._plane_contrast(
                raw if z0 is None else raw[z0],
                lbl if z0 is None else lbl[z0],
                first,
            )
            if z0 is None:
                new[...] = first
            else:
                new[z0] = first
                if self.add_propagate_z:
                    for step in (-1, 1):
                        prev, zi = first, z0 + step
                        while 0 <= zi < lbl.shape[0]:
                            self.viewer.status = (
                                f"Add: SAM2 on Z-plane {zi}..."
                            )
                            mask = self._next_plane_mask(
                                worker,
                                prev,
                                lbl[zi],
                                raw[zi],
                                max_px,
                                smooth_px,
                                first_area,
                                contrast,
                            )
                            if mask is None:
                                break
                            new[zi] = mask
                            prev, zi = mask, zi + step
        except Sam2Unavailable as exc:
            self.viewer.status = f"Add: {exc}"
            return

        new_id = self._allocate_new_id(labels_layer)
        offsets = [b.start for b in box]
        coords = tuple(c + o for c, o in zip(np.nonzero(new), offsets))
        index = coords if t is None else (t, *coords)
        if isinstance(data, _DaskFancyIndexWrapper):
            data[index] = new_id
        else:
            sub[coords] = new_id

        labels_layer.refresh()
        self._refresh_track_view(timepoint=t)
        # A new label changes what there is to measure — re-measure intensity
        # stats on the next preview. (Also clears any older single-T undo,
        # so our record has to be set after it.)
        self._invalidate_track_stats()
        n_px = int(new.sum())
        self._single_t_last = {
            "t": t,
            "restores": [(coords, 0)],
            "desc": f"new label {new_id} removed ({n_px} pixel(s))",
        }
        n_planes = int(new.any(axis=(-2, -1)).sum()) if new.ndim == 3 else 1
        self.viewer.status = (
            f"Added label {new_id}: {n_px} pixel(s) on {n_planes} plane(s)"
            + ("" if t is None else f", timepoint {t}")
            + ". Ctrl+Z removes it; 'Save Changes and Continue' writes it "
            "to disk."
        )

    def _on_click_add(self, layer, label_id, event):
        """Segment the missed cell under the click into a new label.

        Unlike the other click tools this one wants a *background* click, so
        the clicked voxel itself has to be resolved rather than an ID.  In 2-D
        display ``world_to_data`` gives it exactly.  In 3-D display it takes
        the raw volume: napari's pick reports only where the ray hits a label
        and a missed cell has none, so the ray is marched and the brightest
        unlabeled voxel on it — the one napari's maximum-intensity rendering
        put under the cursor — becomes the seed (:meth:`_ray_signal_voxel`).
        The track views project or restack Z, so they are still refused.
        """
        if self._track_view_layer is not None:
            self.viewer.status = (
                "Add works only in the normal frame view — set Track view to "
                "'Off' first."
            )
            return
        occupied = (
            f"Add: that pixel already belongs to label {label_id} — click "
            "a cell that has no label, or use click-to-grow to extend "
            "this one."
        )
        dims_displayed = list(getattr(event, "dims_displayed", None) or [])
        if len(dims_displayed) == 3:
            # In 3-D a ray that meets no label returns None, which here is
            # the ordinary case — a missed cell has no label to hit — so it
            # goes to the ray path rather than being read as off-canvas.
            if label_id:
                self.viewer.status = occupied
            else:
                self._add_from_ray(layer, event, dims_displayed)
            return
        if label_id is None:
            return  # off-canvas
        if label_id:
            self.viewer.status = occupied
            return
        data = getattr(layer, "data", None)
        shape = np.asarray(getattr(data, "shape", ()), dtype=np.intp)
        try:
            full = np.asarray(layer.world_to_data(event.position), dtype=float)
        except Exception:
            full = np.empty(0)
        if full.shape[0] != shape.shape[0]:
            self.viewer.status = "Add: could not resolve the clicked voxel."
            return
        coord = np.clip(np.rint(full).astype(np.intp), 0, shape - 1)
        if data.ndim >= 3:  # axis 0 is time (see grow_label_at_timepoint)
            self.add_label_at_timepoint(tuple(coord[1:]), int(coord[0]))
        else:
            self.add_label_at_timepoint(tuple(coord), 0)

    def _add_from_ray(self, layer, event, dims_displayed) -> None:
        """Add a missed cell from a click in napari's 3-D display.

        Only a 4-D (T, Z, Y, X) label has a Z axis to click into: T is then
        the non-displayed axis, so the timepoint comes from the slider and
        the ray runs through the Z-stack of that frame.  A 3-D (T, Y, X)
        movie shown in 3-D has time as its volume axis, where a "voxel" spans
        frames rather than planes, so that case is sent back to 2-D display.
        """
        data = getattr(layer, "data", None)
        if getattr(data, "ndim", 0) < 4:
            self.viewer.status = (
                "Add: this label has no Z axis, so napari's 3D display is "
                "showing time as the third axis — switch back to 2D display "
                "to add a cell."
            )
            return
        try:
            t = int(round(float(layer.world_to_data(event.position)[0])))
        except Exception:
            self.viewer.status = "Add: could not resolve the clicked frame."
            return
        t = min(max(t, 0), data.shape[0] - 1)

        raw_t = self._raw_slice_at(data, t, self.add_channel)
        if raw_t is None:
            return  # status already set
        voxel, blocker = self._ray_signal_voxel(
            layer, event, dims_displayed, raw_t
        )
        if blocker:
            self.viewer.status = (
                f"Add: the brightest thing along that view ray is label "
                f"{blocker}, so that is the cell under the cursor — use "
                "click-to-grow for it, or rotate the view to reach a cell "
                "behind it."
            )
            return
        if voxel is None:
            self.viewer.status = (
                "Add: the view ray missed the volume — click over the data."
            )
            return
        self.add_label_at_timepoint(
            tuple(int(c) for c in voxel[1:]), t, raw_t=raw_t
        )

    def enable_click_add(self, enabled: bool) -> None:
        """Toggle click-to-add mode (mutually exclusive with the other click
        tools).

        While enabled, a plain left-click on a cell the segmentation missed
        hands that pixel to SAM2 as a prompt and writes the object it returns
        as a new label with a fresh ID, at the clicked timepoint.  Ctrl+Z
        removes it again.
        """
        if enabled and self._click_add_cb is None:
            self.enable_click_delete(False)
            self.enable_click_relabel(False)
            self.enable_click_split(False)
            self.enable_click_merge(False)
            self.enable_click_grow(False)
            self._click_add_cb = self._make_click_callback(self._on_click_add)
            self.viewer.mouse_drag_callbacks.append(self._click_add_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-add ON: click a cell that has no label and SAM2 "
                "segments it into a new one at that timepoint. The first "
                "click loads the model (a few seconds). Ctrl+Z removes the "
                "last added label. Turn Track view off to use it."
            )
        elif not enabled and self._click_add_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_add_cb)
            self._click_add_cb = None
            if (
                self._click_delete_cb is None
                and self._click_relabel_cb is None
                and self._click_split_cb is None
                and self._click_merge_cb is None
                and self._click_grow_cb is None
            ):
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-add OFF."

    def set_track_view_mode(self, mode: str) -> None:
        """Switch the track-inspection view: ``"off"``, ``"stack"`` or ``"max"``.

        ``"stack"`` concatenates all timepoints along Z into one
        ``(T*Z, Y, X)`` volume — a track (label ID) becomes a single
        connected object through the stack, and paint edits map back to
        ``(t, z)``.  ``"max"`` shows one Z-max-projected plane per
        timepoint, ``(T, Y, X)`` — tracks read as clean tubes in 3-D but
        the view is read-only for paint.  Both views are lazy for 2-D
        scrubbing and share the underlying label data, so the ID-based
        click tools, Ctrl+Z undo and saving keep working unchanged.  The
        mode persists across pairs, like the click modes.

        Movies whose full view volume would not fit in GPU memory (napari's
        3-D display uploads it as one 3-D texture) are automatically YX
        subsampled — see :func:`_pick_track_view_step`.
        """
        mode = str(mode).strip().lower()
        if mode not in ("off", "stack", "max"):
            mode = "off"
        self.track_view_mode = mode
        self._apply_track_view()

    def _apply_track_view(self) -> None:
        """(Re)build the track-view layer for the current mode."""
        self._remove_track_view()
        if self.track_view_mode == "off":
            self.viewer.status = "Track view off."
            return
        labels_layer = self._find_labels_layer()
        data = getattr(labels_layer, "data", None)
        if labels_layer is None or getattr(data, "ndim", 0) not in (3, 4):
            self.viewer.status = (
                "Track view needs a loaded 3-D (TYX) or 4-D (TZYX) "
                "labels layer."
            )
            return
        n_z = data.shape[1] if data.ndim == 4 else 1
        n_planes = (
            data.shape[0] * n_z
            if self.track_view_mode == "stack"
            else data.shape[0]
        )
        step = _pick_track_view_step(
            n_planes,
            data.shape[-2],
            data.shape[-1],
            np.dtype(data.dtype).itemsize,
        )
        if self.track_view_mode == "stack":
            view = _StackedTrackView(data, yx_step=step)
            desc = "T stacked along Z"
        else:
            view = _MaxProjTrackView(data, yx_step=step)
            desc = "Z max-projected per T"
        # Reuse the labels layer's spatial scale so the view keeps the
        # right YX aspect; the stacked axis reuses the Z spacing.  A
        # subsampled view stretches its YX scale by the step, so world
        # coordinates (and the click tools) are unaffected.
        try:
            lscale = [float(s) for s in labels_layer.scale]
        except Exception:
            lscale = [1.0] * data.ndim
        if data.ndim == 4:
            axis0 = lscale[1] if self.track_view_mode == "stack" else 1.0
            view_scale = [axis0, lscale[2] * step, lscale[3] * step]
        else:
            view_scale = [1.0, lscale[1] * step, lscale[2] * step]
        # Hide the regular layers while the track view is active — mixing
        # a (T*Z)YX volume with TZYX layers in one dims model is confusing.
        for layer in list(self.viewer.layers):
            self._track_hidden.append(
                (layer, bool(getattr(layer, "visible", True)))
            )
            with contextlib.suppress(Exception):
                layer.visible = False
        self._track_view_layer = self.viewer.add_labels(
            view, scale=view_scale, name=f"Track view ({desc})"
        )
        if self.track_view_mode == "max" or step > 1:
            # Best effort — napari may re-enable this on display changes;
            # the view's read-only __setitem__ is the hard backstop.
            with contextlib.suppress(Exception):
                self._track_view_layer.editable = False
        if (
            self._click_delete_cb is not None
            or self._click_relabel_cb is not None
        ):
            self._bind_undo_key(self._find_labels_layer(), True)
        downsample_note = (
            (
                f" YX subsampled x{step} so 3D fits in GPU memory "
                f"(full volume would be "
                f"{n_planes * data.shape[-2] * data.shape[-1] * np.dtype(data.dtype).itemsize / 1024**3:.1f}"
                " GiB); click tools work as usual, painting is off, the "
                "file stays full resolution."
            )
            if step > 1
            else ""
        )
        self.viewer.status = (
            f"Track view ON ({desc}, {view.shape[0]} planes)."
            f"{downsample_note} Toggle napari's 3D display to see whole "
            "tracks; click modes and Ctrl+Z work as usual. Set 'Off' to "
            "restore the normal view."
        )

    def _remove_track_view(self) -> None:
        """Remove the view layer and restore the hidden regular layers."""
        if self._track_view_layer is not None:
            with contextlib.suppress(Exception):
                self.viewer.layers.remove(self._track_view_layer)
            self._track_view_layer = None
        for layer, was_visible in self._track_hidden:
            with contextlib.suppress(Exception):
                layer.visible = was_visible
        self._track_hidden = []

    def _refresh_track_view(
        self, mapping: dict = None, timepoint: int = None
    ) -> None:
        """Redraw the track-view layer after a label edit.

        When the edit is a global {old_id: new_id} *mapping* (delete /
        relabel), the view's caches are updated in place — no I/O, so 3-D
        stays interactive.  When it is confined to one *timepoint*, only
        that timepoint's planes are recomputed.  For arbitrary edits
        (undo, restores) the caches are dropped and rebuilt lazily.
        """
        layer = self._track_view_layer
        if layer is None:
            return
        data = getattr(layer, "data", None)
        region = None
        if isinstance(data, _TrackView):
            if mapping:
                region = data.apply_mapping(mapping)
            elif timepoint is not None:
                region = data.refresh_timepoint(timepoint)
            else:
                data.invalidate()
        if region is not None and self._partial_track_redraw(layer, region):
            return
        with contextlib.suppress(Exception):
            layer.refresh()

    def _partial_track_redraw(self, layer, region) -> bool:
        """Redraw only *region* of the 3-D track-view layer.

        ``layer.refresh()`` re-slices the whole view and re-uploads the
        whole GPU texture — seconds on a multi-GiB track volume, for every
        click.  napari's own paint tools instead push just the changed box
        (``_partial_labels_refresh``); an edit whose extent we know can do
        the same.  Returns False when that isn't possible, so the caller
        falls back to the full refresh.
        """
        try:
            if int(self.viewer.dims.ndisplay) != 3:
                return False  # 2-D slicing is one plane — already cheap
            if getattr(layer, "contour", 0):
                return False  # contours redraw beyond the changed box
            if any(s.stop <= s.start for s in region):
                return True  # nothing changed — nothing to redraw
            vol = getattr(layer.data, "_vol", None)
            image = layer._slice.image
            # The displayed slice must be the very buffer we just edited,
            # or the canvas would show a stale copy of the rest of it.
            if vol is None or not np.shares_memory(image.raw, vol):
                return False
            image.view[region] = layer.colormap._data_to_texture(vol[region])
            layer._updated_slice = _bbox_union(
                getattr(layer, "_updated_slice", None), region
            )
            layer._partial_labels_refresh()
            return True
        except Exception:
            return False

    def _proceed(self, save: bool):
        """
        Advance to the next image-label pair, optionally saving first.

        Shared by :meth:`next_pair` (save=True) and :meth:`skip_pair`
        (save=False, discarding any in-memory edits to the current pair).
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to inspect."
            return

        if save:
            self.save_current_labels()

        # Check if we're already at the last pair
        if self.current_index >= len(self.image_label_pairs) - 1:
            self.viewer.status = (
                "No more pairs to inspect. Inspection complete."
            )
            # should also clear the viewer
            self.viewer.layers.clear()
            return False  # Return False to indicate we're at the end

        # Move to the next pair
        self.current_index += 1

        # Load the next pair
        self._load_current_pair()
        return (
            True  # Return True to indicate successful navigation to next pair
        )

    def next_pair(self):
        """
        Save changes and proceed to the next image-label pair.
        """
        return self._proceed(save=True)

    def skip_pair(self):
        """
        Discard any in-memory (unsaved) changes to the current pair and
        proceed to the next image-label pair.
        """
        return self._proceed(save=False)


@magicgui(
    call_button="Start Label Inspection",
    folder_path={"label": "Folder Path", "widget_type": "LineEdit"},
    label_suffix={"label": "Label Suffix (e.g., _labels.tif)"},
    channel_axis={
        "label": "Raw channel axis (index)",
        "widget_type": "ComboBox",
        "choices": ["Auto", "None", "0", "1", "2", "3", "4"],
        "tooltip": (
            "0-based position of the channel dimension in the RAW image's "
            "dimension order (the dimension the label lacks). E.g. for a "
            "TZCYX raw paired with a TZYX label, C is at position 2 "
            "(T=0, Z=1, C=2). 'Auto' reads it from the file's axes metadata "
            "and is correct for well-formed OME-TIFF/zarr. Pick a number to "
            "force it when metadata is missing/ambiguous, or 'None' if the "
            "raw has no channel dimension."
        ),
    },
)
def label_inspector(
    folder_path: str,
    label_suffix: str,
    viewer: Viewer,
    channel_axis: str = "Auto",
):
    """
    MagicGUI widget for starting label inspection.
    """
    inspector = LabelInspector(viewer)
    inspector.channel_axis_override = channel_axis
    inspector.load_image_label_pairs(folder_path, label_suffix)

    # Add buttons for saving (or skipping) and continuing to the next pair
    @magicgui(call_button="Save Changes and Continue")
    def save_and_continue():
        at_last_pair = (
            inspector.current_index >= len(inspector.image_label_pairs) - 1
        )
        # next_pair() saves the current labels before checking for the end,
        # so the last pair's edits are written too.
        inspector.next_pair()
        if at_last_pair:
            save_and_continue.call_button.enabled = False
            skip_and_continue.call_button.enabled = False

    @magicgui(call_button="Skip (Discard Changes)")
    def skip_and_continue():
        at_last_pair = (
            inspector.current_index >= len(inspector.image_label_pairs) - 1
        )
        inspector.skip_pair()
        if at_last_pair:
            save_and_continue.call_button.enabled = False
            skip_and_continue.call_button.enabled = False

    # The click tools are mutually exclusive by nature, so one selector
    # drives all of them instead of six checkboxes that had to switch each
    # other off, and a mode's settings are shown only while it is chosen.
    # That keeps the dock two or three rows tall instead of seventeen, which
    # is the difference between the napari window fitting on a laptop screen
    # and not. Each mode explains itself in the status bar when picked.
    modes = {
        "Off": "off",
        "Delete label": "delete",
        "Relabel label": "relabel",
        "Split merged label": "split",
        "Merge touching neighbors": "merge",
        "Grow label to cell (SAM2)": "grow",
        "Add missed cell (SAM2)": "add",
    }

    @magicgui(
        auto_call=True,
        mode={
            "label": "Click mode",
            "widget_type": "ComboBox",
            "choices": list(modes),
            "tooltip": (
                "What a left-click in the viewer does. One mode at a time; "
                "its settings appear below it. Click-drag (pan/zoom) is "
                "never affected, Ctrl+Z undoes the last edit of any mode, "
                "and every edit is staged in memory until 'Save Changes and "
                "Continue' writes it to the file.\n\n"
                "• Delete — remove the clicked label, from the whole movie "
                "or just the clicked timepoint.\n"
                "• Relabel — Ctrl+click a label to pipette its ID, then "
                "click others to reassign them to it (merging them into "
                "it).\n"
                "• Split — for one label spanning several touching cells: "
                "click one point per cell, then 'Apply split'. A seeded "
                "watershed divides it at the constrictions; each part after "
                "the first gets a new ID. Ctrl+click removes the last seed. "
                "Needs Track view off.\n"
                "• Merge touching neighbors — for one cell broken into "
                "several IDs: click one fragment and every label sharing a "
                "border with it merges into it.\n"
                "• Grow label to cell — for a label covering only part of "
                "its cell: SAM2 segments the cell and extends the label to "
                "it, only into background, at the clicked timepoint.\n"
                "• Add missed cell — for a cell with no label at all: click "
                "inside it and SAM2 segments it into a new label with a "
                "fresh ID. In 3D display the click takes the brightest "
                "unlabeled voxel along the view ray. Needs Track view "
                "off.\n\n"
                "The two SAM2 modes need the SAM2 environment, which the "
                "'Batch Crop Anything' widget installs; the first click of a "
                "session loads the model (a few seconds)."
            ),
        },
    )
    def click_mode(mode: str = "Off"):
        """Choose which click tool is active (they are mutually exclusive)."""
        chosen = modes.get(mode, "off")
        for name, settings in mode_settings.items():
            settings.visible = name == chosen
        setters = {
            "delete": inspector.enable_click_delete,
            "relabel": inspector.enable_click_relabel,
            "split": inspector.enable_click_split,
            "merge": inspector.enable_click_merge,
            "grow": inspector.enable_click_grow,
            "add": inspector.enable_click_add,
        }
        # Off first, so the tools' own mutual exclusion never has to fight
        # this one, then on — a no-op when the mode did not really change.
        for name, setter in setters.items():
            if name != chosen:
                setter(False)
        if chosen in setters:
            setters[chosen](True)

    scope_tip = (
        "'All timepoints' applies to the clicked ID across the whole movie "
        "(the fast track-wide path). 'Clicked timepoint only' applies just "
        "to the timepoint you clicked — in the track views the click's plane "
        "determines that timepoint."
    )

    @magicgui(
        auto_call=True,
        scope={
            "label": "Apply to",
            "widget_type": "ComboBox",
            "choices": ["All timepoints", "Clicked timepoint only"],
            "tooltip": scope_tip,
        },
    )
    def delete_settings(scope: str = "All timepoints"):
        """Scope of click-to-delete."""
        inspector.delete_scope = (
            "current" if scope.startswith("Clicked") else "all"
        )

    @magicgui(
        auto_call=True,
        scope={
            "label": "Apply to",
            "widget_type": "ComboBox",
            "choices": ["All timepoints", "Clicked timepoint only"],
            "tooltip": (
                scope_tip + " The target ID is napari's selected label, so "
                "you can also pick it with napari's pipette (color picker) "
                "tool — switch back to the pan/zoom tool before clicking — "
                "or type it into the label spinbox."
            ),
        },
    )
    def relabel_settings(scope: str = "All timepoints"):
        """Scope of click-to-relabel."""
        inspector.relabel_scope = (
            "current" if scope.startswith("Clicked") else "all"
        )

    @magicgui(
        auto_call=True,
        scope={
            "label": "Apply to",
            "widget_type": "ComboBox",
            "choices": ["Clicked timepoint only", "All timepoints"],
            "tooltip": (
                "'Clicked timepoint only' merges the neighbors just in the "
                "frame you clicked. 'All timepoints' merges those same "
                "neighbor IDs across the whole movie — the right unit for "
                "tracked data, where an over-segmented cell keeps the same "
                "fragment IDs frame to frame, so one click repairs the "
                "entire track (the fast LUT path, no extra I/O). Which "
                "labels count as touching is always decided on the clicked "
                "frame alone, so an all-timepoint merge can never swallow a "
                "cell that only brushes past at some other timepoint."
            ),
        },
    )
    def merge_settings(scope: str = "Clicked timepoint only"):
        """Scope of click-to-merge-neighbors."""
        inspector.merge_scope = (
            "current" if scope.startswith("Clicked") else "all"
        )

    @magicgui(call_button="Apply split", labels=False)
    def apply_split():
        """Run the watershed split on the seeds placed in the viewer."""
        inspector.commit_split()

    @magicgui(
        auto_call=True,
        max_growth={
            "label": "Max growth (px)",
            "widget_type": "SpinBox",
            "min": 1,
            "max": 400,
            "tooltip": (
                "Does double duty: how far, in pixels, the label may travel "
                "from its current boundary, and how much context around the "
                "label SAM2 is shown. Too small clips a genuinely larger "
                "cell; too large makes the crop mostly background, which "
                "gives SAM2 less resolution on the cell itself (the crop is "
                "rescaled to 1024x1024 whatever its size). Roughly the "
                "radius of one cell is a good starting point."
            ),
        },
        smoothing={
            "label": "Smoothing (px)",
            "widget_type": "SpinBox",
            "min": 0,
            "max": 20,
            "tooltip": (
                "Polishes the returned contour with a closing then an "
                "opening of this radius, filling small bays and removing "
                "thin spurs. SAM2 masks are already fairly smooth, so 0-2 is "
                "usually right; 0 keeps the model's boundary exactly. "
                "Smoothing can never shrink the label below its original "
                "extent, nor push it onto a neighbor."
            ),
        },
        channel={
            "label": "Signal channel",
            "widget_type": "ComboBox",
            "choices": ["Mean", "0", "1", "2", "3", "4"],
            "tooltip": (
                "Which channel of a multi-channel raw image is shown to "
                "SAM2. 'Mean' averages all channels; pick a channel index "
                "(0-based, along the raw's channel axis) to grow to one "
                "marker's extent — e.g. a membrane or cytoplasm channel "
                "rather than a nuclear one. Ignored for single-channel raws."
            ),
        },
    )
    def grow_settings(
        max_growth: int = 30,
        smoothing: int = 1,
        channel: str = "Mean",
    ):
        """Settings for growing an under-segmented label to its cell."""
        inspector.grow_max_px = int(max_growth)
        inspector.grow_smooth_px = int(smoothing)
        inspector.grow_channel = channel.strip().lower()

    @magicgui(
        auto_call=True,
        max_radius={
            "label": "Max radius (px)",
            "widget_type": "SpinBox",
            "min": 1,
            "max": 400,
            "tooltip": (
                "Does double duty: how far, in pixels, the new label may "
                "reach from the pixel you clicked, and how much context "
                "around it SAM2 is shown. Too small clips a large cell (or "
                "one clicked off-center); too large makes the crop mostly "
                "background, which gives SAM2 less resolution on the cell "
                "itself (the crop is rescaled to 1024x1024 whatever its "
                "size). Roughly the diameter of one cell is a good start, "
                "since the click rarely lands dead center."
            ),
        },
        smoothing={
            "label": "Smoothing (px)",
            "widget_type": "SpinBox",
            "min": 0,
            "max": 20,
            "tooltip": (
                "Polishes the returned contour with a closing then an "
                "opening of this radius, filling small bays and removing thin "
                "spurs. SAM2 masks are already fairly smooth, so 0-2 is "
                "usually right; 0 keeps the model's boundary exactly."
            ),
        },
        channel={
            "label": "Signal channel",
            "widget_type": "ComboBox",
            "choices": ["Mean", "0", "1", "2", "3", "4"],
            "tooltip": (
                "Which channel of a multi-channel raw image is shown to "
                "SAM2. 'Mean' averages all channels; pick a channel index "
                "(0-based, along the raw's channel axis) to outline the cell "
                "in one marker — e.g. a membrane or cytoplasm channel rather "
                "than a nuclear one. Ignored for single-channel raws."
            ),
        },
        propagate_z={
            "label": "Continue through Z",
            "tooltip": (
                "For Z-stacks: after segmenting the plane you clicked, "
                "continue the same object onto the planes above and below, "
                "each prompted with the previous plane's outline and "
                "segmented from its own image (which suits anisotropic "
                "stacks). The walk stops once a plane shows only the cell's "
                "dimmer out-of-focus halo, or where the cell tapers out or "
                "the mask leaks, and every plane keeps the one new ID, so no "
                "stitching is needed. Turn off to label the clicked plane "
                "alone. Ignored for data without a Z axis."
            ),
        },
    )
    def add_settings(
        max_radius: int = 30,
        smoothing: int = 1,
        channel: str = "Mean",
        propagate_z: bool = True,
    ):
        """Settings for segmenting a missed cell into a new label."""
        inspector.add_max_px = int(max_radius)
        inspector.add_smooth_px = int(smoothing)
        inspector.add_channel = channel.strip().lower()
        inspector.add_propagate_z = bool(propagate_z)

    mode_settings = {
        "delete": delete_settings,
        "relabel": relabel_settings,
        "split": apply_split,
        "merge": merge_settings,
        "grow": grow_settings,
        "add": add_settings,
    }
    # Push each panel's defaults into the inspector once, so a mode works
    # off what its settings show even if they are never touched.
    for settings in mode_settings.values():
        if settings is not apply_split:
            settings()

    @magicgui(
        call_button="Apply",
        threshold={
            "label": "Intensity threshold (0–1)",
            "widget_type": "FloatSpinBox",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": (
                "Deletes every track (label ID) whose median raw-image "
                "intensity is below this threshold, on all timepoints. "
                "Intensities are normalized to 0–1 using the raw image's own "
                "min/max, so the same threshold works for 8-bit and 16-bit "
                "images regardless of their dtype range. Set the threshold "
                "and press 'Apply' to preview; re-applying restores all "
                "tracks first and re-applies only the current threshold, so "
                "nothing compounds (0 shows all tracks). Deletions are "
                "staged in memory; press 'Save Changes and Continue' to "
                "write them to the file."
            ),
        },
        channel={
            "label": "Measure channel",
            "widget_type": "ComboBox",
            "choices": ["Mean", "0", "1", "2", "3", "4"],
            "tooltip": (
                "Which channel of a multi-channel raw image supplies the "
                "intensity used to score each track. 'Mean' averages all "
                "channels; pick a channel index (0-based, along the raw's "
                "channel axis) to use just that marker. Ignored for "
                "single-channel raws."
            ),
        },
    )
    def delete_low_intensity(threshold: float = 0.0, channel: str = "Mean"):
        """Delete all tracks whose normalized median raw intensity < threshold."""
        inspector.delete_low_intensity_tracks(threshold, channel=channel)

    @magicgui(
        auto_call=True,
        mode={
            "label": "Track view",
            "widget_type": "ComboBox",
            "choices": ["Off", "Stack T along Z", "Max-project Z per T"],
            "tooltip": (
                "Show the whole movie as a single 3-D volume so each track "
                "(label ID) appears as one connected object — switch napari "
                "to 3D display to see entire tracks, and use the click "
                "modes to delete or relabel a whole track with one click. "
                "'Stack T along Z' concatenates the timepoints (plane i = "
                "timepoint i//Z, slice i%Z) and stays fully editable: "
                "paint and fill map back to the right timepoint. "
                "'Max-project Z per T' shows one Z-projected plane per "
                "timepoint, so tracks read as clean tubes, but painting "
                "is disabled (a projected pixel has no unique Z origin) "
                "and where labels overlap in Z the higher ID wins. Both "
                "views load lazily while scrubbing in 2D; napari's 3D "
                "display loads the whole volume into RAM and GPU memory, "
                "so very large movies are automatically shown YX-"
                "downsampled (label IDs, click tools and the saved file "
                "are unaffected; painting is disabled then). Edits, Ctrl+Z "
                "and 'Save Changes and Continue' work exactly as in the "
                "normal view. 'Off' restores the normal layers."
            ),
        },
    )
    def track_view(mode: str = "Off"):
        """Inspect whole tracks by viewing the movie as one 3-D stack."""
        inspector.set_track_view_mode(
            {
                "Off": "off",
                "Stack T along Z": "stack",
                "Max-project Z per T": "max",
            }.get(mode, "off")
        )

    from magicgui.widgets import Container

    # One dock per functional group, rather than one per tool.
    save_widget = Container(
        widgets=[save_and_continue, skip_and_continue], labels=False
    )
    viewer.window.add_dock_widget(save_widget, name="Save / Skip")

    label_manip_widget = Container(
        widgets=[click_mode, *mode_settings.values()],
        labels=False,
    )
    viewer.window.add_dock_widget(
        label_manip_widget, name="Label manipulations"
    )
    # Only the chosen mode's settings are shown; with "Off" that leaves one
    # row. Hiding has to happen after the container is built, since adding a
    # widget to a Container makes it visible again.
    click_mode()

    # Track-level bulk operations (act across all timepoints of a track);
    # more tools land here as they're added.
    track_manip_widget = Container(
        widgets=[delete_low_intensity], labels=False
    )
    viewer.window.add_dock_widget(
        track_manip_widget, name="Track manipulations"
    )

    viewer.window.add_dock_widget(track_view, name="Track inspection")


def label_inspector_widget():
    """
    Provide the label inspector widget to Napari
    """
    # Create the magicgui widget
    widget = label_inspector

    # Create and add browse button
    browse_button = QPushButton("Browse...")

    def on_browse_clicked():
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            # Update the folder_path field
            widget.folder_path.value = folder

    browse_button.clicked.connect(on_browse_clicked)

    # Insert the browse button next to the folder_path field
    # Find the folder_path widget and its layout
    folder_layout = widget.folder_path.native.parent().layout()
    folder_layout.addWidget(browse_button)

    return widget

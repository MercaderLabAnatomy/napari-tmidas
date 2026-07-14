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


class _TrackView:
    """Read-through 3-D "track volume" view over a TZYX/TYX label source.

    Base class for the track-inspection views: every plane is assembled on
    demand from the underlying source (a :class:`_DaskFancyIndexWrapper` or
    a numpy array), so the view never copies the movie and always reflects
    pending remaps and paint edits.  Label IDs are unchanged, so the
    ID-based tools (click-to-delete / click-to-relabel, Ctrl+Z undo, save)
    work identically through the view.
    """

    def __init__(self, source):
        if getattr(source, "ndim", 0) not in (3, 4):
            raise ValueError(
                "Track view needs a 3-D (TYX) or 4-D (TZYX) label source."
            )
        self._source = source
        self.dtype = source.dtype
        self.ndim = 3
        self._n_t = source.shape[0]
        # A TYX movie has an implicit Z of size 1.
        self._n_z = source.shape[1] if source.ndim == 4 else 1
        # Materialized full view, built lazily by __array__ when napari's
        # 3-D display first asks for it.  While present, ALL reads
        # (including 3-D click pick-rays, which would otherwise re-read
        # one timepoint from disk per plane they cross) are served from
        # it, and delete/relabel remaps update it in place — so 3-D
        # clicking stays interactive on movies of any size.
        self._vol = None

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

    def invalidate(self) -> None:
        """Drop all derived caches after an arbitrary source edit."""
        self._vol = None

    def apply_mapping(self, mapping: dict) -> None:
        """Update caches in place for a global {old_id: new_id} remap.

        The cheap alternative to :meth:`invalidate` for delete / relabel:
        the materialized volume is remapped with one numpy pass instead of
        being rebuilt from disk, so the 3-D display and its pick-rays stay
        interactive.
        """
        if self._vol is not None:
            _apply_value_map_inplace(self._vol, mapping)

    def __getitem__(self, index):
        # Serve everything from the materialized volume while it exists —
        # it already reflects all edits (kept up to date in place).
        if self._vol is not None:
            return self._vol[index]
        if not isinstance(index, tuple):
            index = (index,)
        first = index[0] if index else slice(None)
        rest = index[1:]
        # Scalar outer plane — napari's 2-D slicing path.
        if isinstance(first, (int, np.integer)):
            plane = self._plane(int(first))
            return plane[rest] if rest else plane
        # Pure int/slice indexing — assemble the requested planes.
        if all(isinstance(ix, (int, np.integer, slice)) for ix in index):
            planes = [
                self._plane(i)[rest] if rest else self._plane(i)
                for i in range(*first.indices(self.shape[0]))
            ]
            if planes:
                return np.stack(planes)
            probe = self._plane(0)[rest] if rest else self._plane(0)
            return np.empty((0, *np.shape(probe)), dtype=self.dtype)
        # Fancy indexing (napari's 3-D picking): 1-D coord arrays, possibly
        # mixed with scalars (broadcast to the arrays' length), served
        # plane-by-plane so the source's T-slice cache is reused.
        if len(index) == 3 and all(
            isinstance(ix, (int, np.integer))
            or (isinstance(ix, np.ndarray) and ix.ndim == 1)
            for ix in index
        ):
            n = next(
                ix.size for ix in index if isinstance(ix, np.ndarray)
            )
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
        if self._vol is None:
            out = np.empty(self.shape, dtype=self.dtype)
            for i in range(self.shape[0]):
                out[i] = self._plane(i)
            self._vol = out
        if dtype is not None and dtype != self._vol.dtype:
            return self._vol.astype(dtype)
        return self._vol


class _StackedTrackView(_TrackView):
    """All timepoints concatenated along Z: shape ``(T*Z, Y, X)``.

    Plane ``i`` shows timepoint ``i // Z``, slice ``i % Z``, so a track
    (one label ID through time) is a single connected object through the
    stack.  Fully editable: paint / fill writes are translated back to
    ``(t, z)`` and forwarded to the source, where they are recorded (and
    saved) like any other edit.
    """

    def __init__(self, source):
        super().__init__(source)
        self.shape = (self._n_t * self._n_z, *source.shape[-2:])

    def _plane(self, i: int) -> np.ndarray:
        t, z = divmod(int(i), self._n_z)
        t_slice = self._t_slice(t)
        return t_slice[z] if self._source.ndim == 4 else t_slice

    def __setitem__(self, index, value):
        if not isinstance(index, tuple):
            index = (index,)
        # Keep the materialized 3-D volume in sync with paint edits.
        if self._vol is not None:
            self._vol[index] = value
        if self._source.ndim == 3:  # TYX: the view aliases the source
            self._source[index] = value
            return
        first = index[0] if index else None
        if isinstance(first, (int, np.integer)) and all(
            isinstance(ix, (int, np.integer, slice)) for ix in index[1:]
        ):
            t, z = divmod(int(first), self._n_z)
            self._source[(t, z, *index[1:])] = value
            return
        # napari paint / fill: tuple of equal-length 1-D coord arrays.
        if len(index) == 3 and all(
            isinstance(ix, np.ndarray) and ix.ndim == 1 for ix in index
        ):
            pp, yy, xx = (np.asarray(ix, dtype=np.intp) for ix in index)
            tt, zz = np.divmod(pp, self._n_z)
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
                ] = (value if vals is None else vals[m])
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

    def __init__(self, source):
        super().__init__(source)
        self.shape = (self._n_t, *source.shape[-2:])
        self._cache: OrderedDict = OrderedDict()

    def _plane(self, i: int) -> np.ndarray:
        i = int(i)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        t_slice = self._t_slice(i)
        plane = t_slice.max(axis=0) if self._source.ndim == 4 else t_slice
        while len(self._cache) >= self._CACHE_MAX_PLANES:
            self._cache.popitem(last=False)
        self._cache[i] = plane
        return plane

    def invalidate(self) -> None:
        super().invalidate()
        self._cache.clear()

    def apply_mapping(self, mapping: dict) -> None:
        """Remap caches in place; projected planes are dropped instead.

        2-D planes recompute exactly on demand (one timepoint of I/O), so
        they are dropped.  The materialized 3-D volume is remapped in
        place to stay interactive; where the remapped track occluded
        another label along Z the true projection would reveal that
        label, but recomputing it means re-reading the whole movie — the
        volume shows background there instead until the view is rebuilt
        (toggle the mode off/on).  The underlying data is always exact.
        """
        super().apply_mapping(mapping)
        self._cache.clear()

    def __setitem__(self, index, value):
        raise TypeError(
            "The Z-max-projection track view is read-only: a projected "
            "pixel has no unique (t, z) origin to write back to. Use "
            "click-to-delete / click-to-relabel (they work on label IDs), "
            "or switch the track view to 'Stack T along Z' to paint."
        )


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
            np.minimum(
                ((np.arange(ts) + 0.5) * (ss / ts)).astype(int), ss - 1
            )
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

        image_path, label_path = self.image_label_pairs[self.current_index]
        image = _load_image(image_path)
        label_image = _load_label(label_path)

        # --- Resolve the raw image's channel axis ---------------------------
        # A channel axis lets napari (a) split the raw into per-channel layers
        # and (b) be excluded from the label spatial-scale computation.  The
        # manual override wins when set; otherwise it is auto-detected for both
        # zarr (via .zattrs) and multi-channel TIFFs (e.g. a TZCYX raw paired
        # with a TZYX label, read from series axes metadata).
        channel_axis = self._resolve_channel_axis(image, label_image, image_path)

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
        channel_axis = self._resolve_channel_axis(
            raw, labels_data, image_path
        )
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
        raw_dtype = np.asarray(raw).dtype if not hasattr(raw, "dtype") else raw.dtype
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
                    vals = np.clip(flat_r[inv == i].astype(np.int64), 0, n_bins - 1)
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
            out[int(label_id)] = (
                (value - raw_min) / span if span > 0 else 0.0
            )
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
                self._refresh_track_view()
        elif info["backup"] is not None:
            data[...] = info["backup"]
            layer.refresh()
            self._refresh_track_view()

    def _invalidate_track_stats(self) -> None:
        """Drop cached per-track intensity stats and any pending preview.

        Called when the loaded pair changes or the labels are edited by
        another tool, so a subsequent low-intensity preview re-measures.
        """
        self._track_stats = None
        self._low_intensity_last = None

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
        if isinstance(data, _DaskFancyIndexWrapper):
            mapping = data.undo_remap()
            if mapping:
                layer.refresh()
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
        self.delete_label_all_timepoints(label_id)

    def _on_click_relabel(self, layer, label_id, event):
        modifiers = {
            str(m) for m in (getattr(event, "modifiers", None) or ())
        }
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
        self.relabel_label_all_timepoints(
            label_id, int(layer.selected_label)
        )

    def enable_click_delete(self, enabled: bool) -> None:
        """Toggle click-to-delete mode (mutually exclusive with relabel).

        While enabled, a plain left-click (no drag — panning is unaffected)
        on a label deletes that label from **all** timepoints via the fast
        LUT path, and Ctrl+Z undoes the last deletion.  This replaces
        napari's bucket-with-ndim-4 workflow, which materializes the entire
        array before filling.
        """
        if enabled and self._click_delete_cb is None:
            self.enable_click_relabel(False)
            self._click_delete_cb = self._make_click_callback(
                self._on_click_delete
            )
            self.viewer.mouse_drag_callbacks.append(self._click_delete_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-delete ON: click a label to remove it "
                "from all timepoints (Ctrl+Z undoes the last deletion)."
            )
        elif not enabled and self._click_delete_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_delete_cb)
            self._click_delete_cb = None
            if self._click_relabel_cb is None:
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-delete OFF."

    def enable_click_relabel(self, enabled: bool) -> None:
        """Toggle click-to-relabel mode (mutually exclusive with delete).

        While enabled, a plain left-click on a label assigns it the layer's
        currently selected label ID on **all** timepoints via the fast LUT
        path (merging it into any existing label with that ID).
        Ctrl+left-click pipettes: it sets the selected label ID from the
        clicked label instead of relabeling.  Ctrl+Z undoes the last
        relabel.
        """
        if enabled and self._click_relabel_cb is None:
            self.enable_click_delete(False)
            self._click_relabel_cb = self._make_click_callback(
                self._on_click_relabel
            )
            self.viewer.mouse_drag_callbacks.append(self._click_relabel_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-relabel ON: Ctrl+click a label to pick up its ID, "
                "then click labels to relabel them to it on all timepoints "
                "(Ctrl+Z undoes the last relabel)."
            )
        elif not enabled and self._click_relabel_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(
                    self._click_relabel_cb
                )
            self._click_relabel_cb = None
            if self._click_delete_cb is None:
                self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-relabel OFF."

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
        if self.track_view_mode == "stack":
            view = _StackedTrackView(data)
            desc = "T stacked along Z"
        else:
            view = _MaxProjTrackView(data)
            desc = "Z max-projected per T"
        # Reuse the labels layer's spatial scale so the view keeps the
        # right YX aspect; the stacked axis reuses the Z spacing.
        try:
            lscale = [float(s) for s in labels_layer.scale]
        except Exception:
            lscale = [1.0] * data.ndim
        if data.ndim == 4:
            axis0 = lscale[1] if self.track_view_mode == "stack" else 1.0
            view_scale = [axis0, lscale[2], lscale[3]]
        else:
            view_scale = [1.0, lscale[1], lscale[2]]
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
        if self.track_view_mode == "max":
            # Best effort — napari may re-enable this on display changes;
            # the view's read-only __setitem__ is the hard backstop.
            with contextlib.suppress(Exception):
                self._track_view_layer.editable = False
        if (
            self._click_delete_cb is not None
            or self._click_relabel_cb is not None
        ):
            self._bind_undo_key(self._find_labels_layer(), True)
        self.viewer.status = (
            f"Track view ON ({desc}, {view.shape[0]} planes). Toggle "
            "napari's 3D display to see whole tracks; click modes and "
            "Ctrl+Z work as usual. Set 'Off' to restore the normal view."
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

    def _refresh_track_view(self, mapping: dict = None) -> None:
        """Redraw the track-view layer after a label edit.

        When the edit is a global {old_id: new_id} *mapping* (delete /
        relabel), the view's caches are updated in place — no I/O, so 3-D
        stays interactive.  For arbitrary edits (undo, restores) the
        caches are dropped and rebuilt lazily.
        """
        layer = self._track_view_layer
        if layer is None:
            return
        data = getattr(layer, "data", None)
        if isinstance(data, _TrackView):
            if mapping:
                data.apply_mapping(mapping)
            else:
                data.invalidate()
        with contextlib.suppress(Exception):
            layer.refresh()

    def next_pair(self):
        """
        Save changes and proceed to the next image-label pair.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to inspect."
            return

        # Save current labels before proceeding
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

    # Add buttons for saving and continuing to the next pair
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

    @magicgui(
        auto_call=True,
        enabled={
            "label": "Click a label to delete it from all timepoints",
            "tooltip": (
                "While enabled, left-click any label in the viewer to "
                "remove it from every timepoint instantly. Ctrl+Z undoes "
                "the last deletion. Click-drag (pan/zoom) and clicks on "
                "background do nothing. Deletions are staged in memory; "
                "press 'Save Changes and Continue' to write them to the "
                "file — saved deletions can no longer be undone."
            ),
        },
    )
    def click_to_delete(enabled: bool = False):
        """While on, left-click a label in the viewer to delete it from all T."""
        # NB: index by name — `.enabled` is FunctionGui's own bool property
        # (widget enabled state) and shadows the parameter's CheckBox.
        if enabled and click_to_relabel["enabled"].value:
            click_to_relabel["enabled"].value = False
        inspector.enable_click_delete(enabled)

    @magicgui(
        auto_call=True,
        enabled={
            "label": "Click a label to relabel it (Ctrl+click picks up an ID)",
            "tooltip": (
                "While enabled, Ctrl+left-click a label to pipette its ID, "
                "then plain left-click other labels to relabel them to that "
                "ID on every timepoint (merging them into it). The target "
                "ID is napari's selected label, so you can also pick it "
                "with napari's pipette (color picker) tool — switch back "
                "to the pan/zoom tool (camera symbol) before clicking — or "
                "type it into the label spinbox. Ctrl+Z undoes the last "
                "relabel. Click-drag (pan/zoom) and clicks on background "
                "do nothing. Relabels are staged in memory; press 'Save "
                "Changes and Continue' to write them to the file — saved "
                "relabels can no longer be undone."
            ),
        },
    )
    def click_to_relabel(enabled: bool = False):
        """While on, left-click a label to give it the pipetted ID on all T."""
        if enabled and click_to_delete["enabled"].value:
            click_to_delete["enabled"].value = False
        inspector.enable_click_relabel(enabled)

    @magicgui(
        call_button="Apply",
        threshold={
            "label": "Intensity threshold (0–1)",
            "widget_type": "FloatSlider",
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
                "display loads the whole volume into RAM. Edits, Ctrl+Z "
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

    viewer.window.add_dock_widget(save_and_continue)
    viewer.window.add_dock_widget(click_to_delete, name="Delete label (all T)")
    viewer.window.add_dock_widget(
        click_to_relabel, name="Relabel label (all T)"
    )
    viewer.window.add_dock_widget(
        delete_low_intensity, name="Delete low-intensity tracks"
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

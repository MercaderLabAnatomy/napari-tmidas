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
            reduced = tuple(
                idx for i, idx in enumerate(new_index) if i != outer_dim
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
    def __init__(self, viewer: Viewer):
        self.viewer = viewer
        self.image_label_pairs = []
        self.current_index = 0
        self._click_delete_cb = None  # click-to-delete mouse callback

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

    def _load_current_pair(self):
        """
        Load the current image-label pair into the Napari viewer.
        Automatically scales both layers so they share the same spatial extent,
        enabling correct overlay even when image and label have different resolutions.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to inspect."
            return

        image_path, label_path = self.image_label_pairs[self.current_index]
        image = _load_image(image_path)
        label_image = _load_label(label_path)

        # --- Detect channel axis for zarr images ---
        # If the raw image has a channel dimension that the label lacks we need
        # to (a) tell napari which axis is channels so it splits layers, and
        # (b) exclude that axis when computing the spatial scale.
        channel_axis = None
        if _is_zarr(image_path) and image.ndim > label_image.ndim:
            try:
                from napari_tmidas._file_selector import (
                    detect_channels_from_zarr_path,
                )

                _n_ch, channel_axis = detect_channels_from_zarr_path(
                    image_path
                )
            except Exception:
                channel_axis = None

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

        # Click-to-delete mode persists across pairs; rebind its Ctrl+Z
        # undo on the freshly created layer.
        if self._click_delete_cb is not None:
            self._bind_undo_key(new_labels_layer, True)

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

        # Find the labels layer in the viewer
        labels_layer = next(
            (
                layer
                for layer in self.viewer.layers
                if isinstance(layer, Labels)
            ),
            None,
        )

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
        return next(
            (
                layer
                for layer in self.viewer.layers
                if isinstance(layer, Labels)
            ),
            None,
        )

    def delete_label_all_timepoints(self, label_id: int = None) -> None:
        """Delete *label_id* (default: the selected label) from **all** timepoints.

        For dask-backed labels this is a value-remap LUT entry
        (:meth:`_DaskFancyIndexWrapper.remap_values`): zero I/O at call time,
        one lazy LUT pass at read/save regardless of how many remaps have
        accumulated, and pending manual edits are remapped too.  The current
        view refreshes instantly from the in-place-updated slice cache.
        """
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

        data = labels_layer.data

        try:
            import dask.array as da
        except ImportError:
            da = None

        if isinstance(data, _DaskFancyIndexWrapper):
            data.remap_values({label_id: 0})
        elif da is not None and isinstance(data, da.Array):
            wrapper = _DaskFancyIndexWrapper(data)
            wrapper.remap_values({label_id: 0})
            labels_layer.data = wrapper
        else:
            # Plain numpy: in-place.
            data[data == label_id] = 0

        labels_layer.refresh()
        self.viewer.status = (
            f"Label {label_id} removed from all timepoints. "
            "Save to write changes to disk."
        )

    def _on_undo_key(self, layer):
        """Ctrl+Z while click-to-delete mode is on.

        Undoes the most recent all-T deletion; when there is none pending,
        falls through to napari's own paint undo.  Saved deletions cannot
        be undone (the save is the undo barrier).
        """
        data = getattr(layer, "data", None)
        if isinstance(data, _DaskFancyIndexWrapper):
            mapping = data.undo_remap()
            if mapping:
                layer.refresh()
                ids = ", ".join(str(k) for k in mapping)
                self.viewer.status = f"Restored label {ids} (deletion undone)."
                return
        undo = getattr(layer, "undo", None)
        if callable(undo):
            undo()

    def _bind_undo_key(self, layer, bind: bool) -> None:
        if layer is None or not hasattr(layer, "bind_key"):
            return
        with contextlib.suppress(Exception):
            layer.bind_key(
                "Control-Z",
                self._on_undo_key if bind else None,
                overwrite=True,
            )

    def enable_click_delete(self, enabled: bool) -> None:
        """Toggle click-to-delete mode.

        While enabled, a plain left-click (no drag — panning is unaffected)
        on a label deletes that label from **all** timepoints via the fast
        LUT path, and Ctrl+Z undoes the last deletion.  This replaces
        napari's bucket-with-ndim-4 workflow, which materializes the entire
        array before filling.
        """
        if enabled and self._click_delete_cb is None:

            def _cb(viewer, event):
                if event.button != 1:  # left button only
                    return
                dragged = False
                yield  # wait out the drag: only a clean click deletes
                while event.type == "mouse_move":
                    dragged = True
                    yield
                if dragged:
                    return
                layer = self._find_labels_layer()
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
                if val is None or int(val) == 0:
                    self.viewer.status = (
                        "Click-to-delete: background clicked, nothing removed."
                    )
                    return
                self.delete_label_all_timepoints(int(val))

            self._click_delete_cb = _cb
            self.viewer.mouse_drag_callbacks.append(_cb)
            self._bind_undo_key(self._find_labels_layer(), True)
            self.viewer.status = (
                "Click-to-delete ON: click a label to remove it "
                "from all timepoints (Ctrl+Z undoes the last deletion)."
            )
        elif not enabled and self._click_delete_cb is not None:
            with contextlib.suppress(ValueError):
                self.viewer.mouse_drag_callbacks.remove(self._click_delete_cb)
            self._click_delete_cb = None
            self._bind_undo_key(self._find_labels_layer(), False)
            self.viewer.status = "Click-to-delete OFF."

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
)
def label_inspector(
    folder_path: str,
    label_suffix: str,
    viewer: Viewer,
):
    """
    MagicGUI widget for starting label inspection.
    """
    inspector = LabelInspector(viewer)
    inspector.load_image_label_pairs(folder_path, label_suffix)

    # Add buttons for saving and continuing to the next pair
    @magicgui(call_button="Save Changes and Continue")
    def save_and_continue():
        # Check if we're at the last pair before proceeding
        if inspector.current_index >= len(inspector.image_label_pairs) - 1:
            save_and_continue.call_button.enabled = False
            inspector.viewer.status = (
                "All pairs processed. Inspection complete."
            )
            return
        inspector.next_pair()

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
        inspector.enable_click_delete(enabled)

    viewer.window.add_dock_widget(save_and_continue)
    viewer.window.add_dock_widget(click_to_delete, name="Delete label (all T)")


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

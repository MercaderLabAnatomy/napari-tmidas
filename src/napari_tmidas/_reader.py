"""
This module implements a reader plugin for napari.
"""

import contextlib
import threading
from collections import OrderedDict

import numpy as np

# Cache of open TiffFile handles so lazy per-page dask tasks don't reopen
# and re-parse the file's page index on every read (O(pages²) otherwise —
# dominant cost for files with thousands of pages).
_TIFF_HANDLES: OrderedDict = OrderedDict()
_TIFF_HANDLES_LOCK = threading.Lock()
_TIFF_HANDLES_MAX = 8


def _get_cached_tiff(path):
    """Return a (shared, thread-safe) open TiffFile for *path*.

    The cached handle is validated against the file's current identity
    (inode, mtime, size) so an overwritten or replaced file is reopened
    instead of served stale from the old inode.
    """
    import os

    import tifffile

    st = os.stat(path)
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _TIFF_HANDLES_LOCK:
        cached = _TIFF_HANDLES.get(path)
        if cached is not None:
            tf, cached_sig = cached
            if cached_sig == sig and not tf.filehandle.closed:
                _TIFF_HANDLES.move_to_end(path)
                return tf
            with contextlib.suppress(Exception):
                tf.close()
        tf = tifffile.TiffFile(path)
        # Serialize file seeks/reads across dask threads (decode still
        # runs in parallel) — same mechanism tifffile's zarr store uses.
        tf.filehandle.lock = True
        _TIFF_HANDLES[path] = (tf, sig)
        while len(_TIFF_HANDLES) > _TIFF_HANDLES_MAX:
            _, (old, _) = _TIFF_HANDLES.popitem(last=False)
            with contextlib.suppress(Exception):
                old.close()
        return tf


def invalidate_tiff_cache(path=None):
    """Close cached TIFF handle(s) — call after overwriting a file on disk,
    otherwise reads keep coming from the old (replaced) inode."""
    with _TIFF_HANDLES_LOCK:
        paths = [path] if path is not None else list(_TIFF_HANDLES)
        for p in paths:
            entry = _TIFF_HANDLES.pop(p, None)
            if entry is not None:
                with contextlib.suppress(Exception):
                    entry[0].close()


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # Support .npy files
    if path.endswith(".npy"):
        return reader_function

    # Support .tif/.tiff: use tifffile series[0] shape + lazy dask page loading.
    # napari's built-in reader calls iio.imread() which fully materializes the
    # array into RAM.  For large TZYX volumes (e.g. 68 GB uncompressed) this
    # causes OOM.  This reader builds a dask array shaped by series[0].shape
    # so napari decompresses only the pages currently in view (~one YX plane).
    if path.lower().endswith((".tif", ".tiff")):
        return tiff_reader_function

    # if we know we cannot read the file, we immediately return None.
    return None


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def tiff_reader_function(path):
    """Lazy dask reader for TIFF files preserving correct n-dimensional shape.

    Rebuilds the array as series[0].shape (e.g. TZYX) via one dask task per
    IFD page, so only viewed slices are ever decompressed into RAM.
    """
    import dask.array as da
    import tifffile
    from dask import delayed

    paths = [path] if isinstance(path, str) else path
    results = []

    for p in paths:
        tf = _get_cached_tiff(p)
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        dtype = series.dtype
        n_pages = len(series.pages)

        if n_pages <= 1 or len(shape) <= 2:
            arr = tifffile.imread(p)
            import os

            basename = os.path.basename(p).lower()
            layer_type = "labels" if "label" in basename else "image"
            results.append((arr, {}, layer_type))
            continue

        page_shape = shape[-2:]

        def _read_page(fpath, idx):
            # Shared cached handle: the page index is parsed once per file,
            # not once per page read.  The whole fetch runs under the
            # handle's reentrant lock: lazy page-IFD instantiation in
            # `pages[idx]` mutates shared state and is not thread-safe.
            tf = _get_cached_tiff(fpath)
            with tf.filehandle.lock:
                return tf.series[0].pages[idx].asarray()

        dask_pages = [
            da.from_delayed(
                delayed(_read_page)(p, i),
                shape=page_shape,
                dtype=dtype,
            )
            for i in range(n_pages)
        ]

        flat = da.stack(dask_pages)
        arr = flat.reshape(shape)

        # Detect label files by filename convention
        import os

        basename = os.path.basename(p).lower()
        layer_type = "labels" if "label" in basename else "image"
        results.append((arr, {}, layer_type))

    return results

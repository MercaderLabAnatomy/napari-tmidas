"""
This module implements a reader plugin for napari.
"""

import numpy as np


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
    import tifffile
    import dask.array as da
    from dask import delayed

    paths = [path] if isinstance(path, str) else path
    results = []

    for p in paths:
        with tifffile.TiffFile(p) as tf:
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
            with tifffile.TiffFile(fpath) as _tf:
                return _tf.series[0].pages[idx].asarray()

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

"""
Automatic checks on files a batch run has just written.

Kept cheap enough to run on every output: the checks stream one YX plane at
a time and stop as soon as the answer is known. They never raise; a file
they cannot read gets ``None`` ("cannot tell"), not an error.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# Outputs the plane reader understands. Anything else (CSV, NPY points,
# JSON) is not an image and is never reported as empty.
_IMAGE_SUFFIXES = (".tif", ".tiff", ".zarr")


def output_is_all_zero(path: str) -> Optional[bool]:
    """
    Whether every pixel of the image at ``path`` is zero.

    Returns False at the first non-zero plane, so a normal output costs about
    one plane read. Returns None when ``path`` is not an image this can read.
    """
    try:
        if not path or not os.path.exists(path):
            return None
        if not str(path).lower().rstrip("/").endswith(_IMAGE_SUFFIXES):
            return None

        from napari_tmidas.processing_functions.intensity_label_filter import (
            _PlaneReader,
        )

        with _PlaneReader(Path(path)) as reader:
            for index in np.ndindex(*reader.leading_shape):
                if np.any(reader.plane(index)):
                    return False
        return True
    except Exception as exc:  # noqa: BLE001 - a QC check must never break a run
        print(f"Empty-output check skipped for {path}: {exc}")
        return None


def expected_physical_sizes(metadata: dict) -> Dict[str, float]:
    """
    Pixel size in µm per axis that a converter loader reported.

    Loaders give ``resolution`` as (x, y) pixels per µm and ``spacing`` as
    the Z step in µm. Axes the loader said nothing about are left out.
    """
    sizes = {}
    resolution = metadata.get("resolution")
    try:
        if resolution and len(resolution) >= 2:
            for axis, value in zip("XY", resolution):
                if float(value) > 0:
                    sizes[axis] = 1.0 / float(value)
    except (TypeError, ValueError):
        pass
    try:
        spacing = metadata.get("spacing")
        if spacing and float(spacing) > 0:
            sizes["Z"] = float(spacing)
    except (TypeError, ValueError):
        pass
    return sizes


def _read_shape_dtype(path: str):
    """Shape and dtype of an image file's full-resolution data."""
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            return tuple(int(s) for s in series.shape), np.dtype(series.dtype)

    import zarr

    from napari_tmidas.processing_functions.intensity_label_filter import (
        _PlaneReader,
    )

    array = _PlaneReader._first_array(zarr.open(str(path), mode="r"))
    return tuple(int(s) for s in array.shape), np.dtype(array.dtype)


def read_physical_sizes(path: str) -> Dict[str, float]:
    """Pixel size in µm per axis (X, Y, Z) as stored in an output file."""
    from napari_tmidas.processing_functions.ome_output_utils import (
        _extract_source_physical_scale,
    )

    sizes = _extract_source_physical_scale(path, "XYZ")
    if sizes or not str(path).lower().endswith((".tif", ".tiff")):
        return {k: float(v) for k, v in sizes.items()}

    # Not OME: an ImageJ TIFF keeps X/Y in the resolution tags (pixels per
    # unit) and Z in its own metadata.
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        for axis, tag in (("X", "XResolution"), ("Y", "YResolution")):
            value = page.tags.get(tag)
            if value is not None:
                numerator, denominator = value.value
                if numerator:
                    sizes[axis] = denominator / numerator
        spacing = (tif.imagej_metadata or {}).get("spacing")
        if spacing:
            sizes["Z"] = float(spacing)
    return sizes


def check_converted_output(
    path: str, shape: tuple, dtype, metadata: dict
) -> List[str]:
    """
    Read a converted file back and list how it differs from its source.

    ``shape`` and ``dtype`` are those of the array handed to the writer and
    ``metadata`` is the loader's metadata. An empty list means the file
    holds what was meant to be written. Never raises.
    """
    try:
        written_shape, written_dtype = _read_shape_dtype(path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return [f"output cannot be read back ({exc})"]

    problems = []
    shape = tuple(int(s) for s in shape)
    # Writers may reorder leading axes (OME-Zarr wants TCZYX) but never
    # drop data or move Y and X from the end.
    if int(np.prod(written_shape)) != int(np.prod(shape)):
        problems.append(f"shape {written_shape} does not hold source {shape}")
    elif written_shape[-2:] != shape[-2:]:
        problems.append(f"YX {written_shape[-2:]} differs from source {shape[-2:]}")
    if written_dtype != np.dtype(dtype):
        problems.append(f"dtype {written_dtype} differs from source {np.dtype(dtype)}")

    expected = expected_physical_sizes(metadata)
    axes = str(metadata.get("axes") or "").upper()
    if len(shape) < 3 or (axes and "Z" not in axes):
        # Some loaders report a Z step for data that has no Z axis.
        expected.pop("Z", None)
    if expected:
        try:
            written = read_physical_sizes(path)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            written = {}
            problems.append(f"pixel size cannot be read back ({exc})")
        for axis, value in sorted(expected.items()):
            got = written.get(axis)
            if got is None:
                problems.append(f"{axis} pixel size {value:.4g} µm was not written")
            elif not np.isclose(got, value, rtol=1e-3):
                problems.append(
                    f"{axis} pixel size is {got:.4g} µm, source has {value:.4g} µm"
                )
    return problems


def result_output_paths(result: dict) -> List[str]:
    """Every output path named in a batch worker result dict."""
    paths = []
    if result.get("processed_file"):
        paths.append(result["processed_file"])
    processed = result.get("processed_files")
    if isinstance(processed, (list, tuple)):
        paths.extend(p for p in processed if p)
    return [str(p) for p in paths]


def all_zero_outputs(paths: Iterable[str]) -> List[str]:
    """Basenames of the outputs in ``paths`` that are entirely zero."""
    return [
        os.path.basename(str(p).rstrip("/"))
        for p in paths
        if output_is_all_zero(p) is True
    ]

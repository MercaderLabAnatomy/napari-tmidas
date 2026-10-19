"""
Automatic checks on files a batch run has just written.

Kept cheap enough to run on every output: the checks stream one YX plane at
a time and stop as soon as the answer is known. They never raise; a file
they cannot read gets ``None`` ("cannot tell"), not an error.
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional

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

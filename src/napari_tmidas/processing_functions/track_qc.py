# processing_functions/track_qc.py
"""
Quality control for tracking results.

The trackers (Trackastra, HOCT, Ultrack) all write a tracked label image: a
TYX or TZYX stack in which an object keeps its label id from frame to frame.
Everything here is derived from that image alone, so it works the same for
every tracker and for tracked stacks edited by hand afterwards.

What labels alone cannot tell is lineage: a tracker that splits a track at a
division gives the daughters new ids, and nothing in the image says whose
daughters they are. Divisions therefore show up here as tracks that start
after the first frame, not as divisions.
"""

import os
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

# Written into the batch output folder, one row per tracked file.
STATS_CSV_NAME = "track_statistics.csv"
# Per-file table of every object in every frame, in napari tracks layout.
TRACKS_CSV_SUFFIX = "_label_tracks.csv"

# Several files of one batch can finish at once; they share the stats CSV.
_STATS_LOCK = threading.Lock()


def _resolve_layout(shape: tuple, dim_order: str = "Auto") -> bool:
    """
    Whether a tracked stack of ``shape`` has a Z axis (TZYX) or not (TYX).

    Every tracker in this package writes T first, so the layout follows from
    the number of dimensions; a dimension hint is only used to reject input
    that is not a time series.
    """
    hint = str(dim_order or "Auto").upper()
    if hint not in ("AUTO", "") and not hint.startswith("T"):
        raise ValueError(
            f"Track statistics need a time series with T first; the "
            f"dimension order is set to {hint}"
        )
    if len(shape) == 3:
        return False
    if len(shape) == 4:
        return True
    raise ValueError(
        f"Expected a tracked TYX or TZYX label image, got shape {shape}"
    )


def _plane_objects(plane: np.ndarray, z: Optional[int]):
    """Label id, voxel count and coordinate sums of every object in a plane."""
    flat = plane.ravel()
    nonzero = np.flatnonzero(flat)
    if nonzero.size == 0:
        return None
    ids = flat[nonzero]
    y, x = np.divmod(nonzero, plane.shape[1])
    # Count over the unique ids, not the raw ids: np.bincount on raw ids
    # allocates (max id + 1) int64s, which for sparse uint32 ids is huge.
    unique, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = [
        np.full(unique.size, float(z)) * counts if z is not None else None,
        np.bincount(inverse, weights=y),
        np.bincount(inverse, weights=x),
    ]
    return unique, counts, sums


def _reduce(parts):
    """Merge per-plane object sums of one frame into one row per id."""
    ids = np.concatenate([p[0] for p in parts])
    counts = np.concatenate([p[1] for p in parts])
    unique, inverse = np.unique(ids, return_inverse=True)
    total = np.bincount(inverse, weights=counts)
    sums = []
    for axis in range(3):
        if parts[0][2][axis] is None:
            sums.append(None)
            continue
        values = np.concatenate([p[2][axis] for p in parts])
        sums.append(np.bincount(inverse, weights=values))
    return unique, total, sums


def tracks_from_labels(source, dim_order: str = "Auto"):
    """
    Every object in every frame of a tracked label image.

    ``source`` is a file path (read one YX plane at a time) or an array.
    Returns ``(table, n_frames)``: a DataFrame with ``track_id, t, [z,] y, x,
    voxels``, sorted by track then time (the order napari's Tracks layer
    wants), and the number of frames in the stack.
    """
    if isinstance(source, (str, os.PathLike)):
        from napari_tmidas.processing_functions.intensity_label_filter import (
            _PlaneReader,
        )

        reader = _PlaneReader(Path(source))
    else:
        reader = _ArrayPlanes(np.asarray(source))

    try:
        has_z = _resolve_layout(reader.shape, dim_order)
        n_frames = int(reader.shape[0])
        n_z = int(reader.shape[1]) if has_z else 1
        columns = {"track_id": [], "t": [], "z": [], "y": [], "x": [], "voxels": []}
        for t in range(n_frames):
            parts = []
            for z in range(n_z):
                plane = reader.plane((t, z) if has_z else (t,))
                found = _plane_objects(plane, z if has_z else None)
                if found is not None:
                    parts.append(found)
            if not parts:
                continue
            ids, voxels, sums = _reduce(parts)
            columns["track_id"].append(ids.astype(np.int64))
            columns["t"].append(np.full(ids.size, t, dtype=np.int64))
            for name, total in zip("zyx", sums):
                if total is not None:
                    columns[name].append(total / voxels)
            columns["voxels"].append(voxels.astype(np.int64))
    finally:
        reader.close()

    names = ["track_id", "t"] + (["z"] if has_z else []) + ["y", "x", "voxels"]
    data = {
        name: (
            np.concatenate(columns[name])
            if columns[name]
            else np.empty(0, dtype=np.float64 if name in ("z", "y", "x") else np.int64)
        )
        for name in names
    }
    table = pd.DataFrame(data, columns=names)
    table = table.sort_values(["track_id", "t"], kind="stable")
    return table.reset_index(drop=True), n_frames


class _ArrayPlanes:
    """The slice of the _PlaneReader interface tracks_from_labels uses."""

    def __init__(self, array: np.ndarray):
        self._array = array
        self.shape = tuple(array.shape)

    def plane(self, index: tuple) -> np.ndarray:
        return np.asarray(self._array[index])

    def close(self):
        pass


def track_statistics(table, n_frames: int, min_track_length: int = 3) -> Dict:
    """Summary numbers that show whether a tracking run went wrong."""
    stats = {"frames": int(n_frames)}
    per_frame = np.zeros(max(int(n_frames), 1), dtype=np.int64)
    if len(table):
        frame_counts = table.groupby("t").size()
        per_frame[frame_counts.index.to_numpy()] = frame_counts.to_numpy()
    stats["objects_per_frame_min"] = int(per_frame.min())
    stats["objects_per_frame_mean"] = round(float(per_frame.mean()), 3)
    stats["objects_per_frame_max"] = int(per_frame.max())

    tracks = table.groupby("track_id")["t"].agg(["size", "min", "max"])
    n_tracks = len(tracks)
    length = tracks["size"].to_numpy()
    span = (tracks["max"] - tracks["min"] + 1).to_numpy()
    stats["tracks"] = n_tracks
    stats["track_length_mean"] = round(float(length.mean()), 3) if n_tracks else 0.0
    stats["track_length_median"] = float(np.median(length)) if n_tracks else 0.0
    full = int(np.count_nonzero(length == n_frames))
    stats["tracks_full_length"] = full
    stats["tracks_full_length_fraction"] = (
        round(full / n_tracks, 4) if n_tracks else 0.0
    )
    stats["tracks_short"] = int(np.count_nonzero(length < int(min_track_length)))
    stats["tracks_with_gaps"] = int(np.count_nonzero(span > length))
    stats["tracks_starting_late"] = int(np.count_nonzero(tracks["min"] > 0))
    stats["tracks_ending_early"] = int(
        np.count_nonzero(tracks["max"] < n_frames - 1)
    )
    return stats


# File names the trackers give their output (`_hoct_tracked` contains
# `_tracked`). Only these get a Tracks layer in the results table.
TRACKED_NAME_MARKERS = ("_tracked", "_ultrack")
# Above this dense size a tracks table is not computed while the user waits.
LIVE_TRACKS_MAX_BYTES = 2 * 1024**3


def is_tracked_output(filepath: str) -> bool:
    name = os.path.basename(str(filepath).rstrip("/")).lower()
    return any(marker in name for marker in TRACKED_NAME_MARKERS)


def _ultrack_tracks_csv(filepath: str) -> Optional[str]:
    """The tracks table Ultrack writes next to its `_ultrack.tif`."""
    path = str(filepath)
    if not path.endswith("_ultrack.tif"):
        return None
    candidate = path[: -len("_ultrack.tif")] + "_ultrack_tracks.csv"
    return candidate if os.path.exists(candidate) else None


def _label_tracks_csv(filepath: str) -> Optional[str]:
    """A table Track Statistics wrote for this file: next to it or one level down."""
    folder, name = os.path.split(str(filepath).rstrip("/"))
    filename = os.path.splitext(name)[0] + TRACKS_CSV_SUFFIX
    candidates = [os.path.join(folder, filename)]
    candidates += sorted(
        str(p) for p in Path(folder or ".").glob(f"*/{filename}")
    )
    return next((c for c in candidates if os.path.exists(c)), None)


def load_tracks_for_labels(filepath: str, labels_ndim: int):
    """
    Tracks layer data for a tracked label file, from the cheapest source.

    In order: Ultrack's own tracks CSV (the only one with lineage), a table
    written by Track Statistics, or the label file itself when it is small
    enough to scan now. Returns ``(data, graph, source)`` with ``data`` as
    napari wants it (``track_id, t, [z,] y, x``), or ``(None, None,
    reason)`` when there is nothing to show.
    """
    graph = {}
    csv_path = _ultrack_tracks_csv(filepath)
    if csv_path:
        table = pd.read_csv(csv_path)
        source = os.path.basename(csv_path)
        if "parent_track_id" in table:
            links = table[["track_id", "parent_track_id"]].drop_duplicates()
            links = links[links.parent_track_id >= 0]
            graph = {
                int(child): [int(parent)]
                for child, parent in zip(links.track_id, links.parent_track_id)
            }
    else:
        csv_path = _label_tracks_csv(filepath)
        if csv_path:
            table = pd.read_csv(csv_path)
            source = os.path.basename(csv_path)
        else:
            from napari_tmidas._output_qc import _read_shape_dtype

            shape, dtype = _read_shape_dtype(filepath)
            dense = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if dense > LIVE_TRACKS_MAX_BYTES:
                return None, None, (
                    "too large to scan now; run 'Track Statistics (QC)' on it "
                    "to write a tracks table"
                )
            table, _ = tracks_from_labels(filepath)
            source = "computed from the labels"

    columns = ["track_id"] + [c for c in ("t", "z", "y", "x") if c in table]
    if len(columns) - 1 != labels_ndim:
        return None, None, (
            f"tracks have {len(columns) - 1} dimensions, labels have {labels_ndim}"
        )
    if table.empty:
        return None, None, "no objects in the tracked labels"
    table = table.sort_values(["track_id", "t"], kind="stable")
    data = table[columns].to_numpy(dtype=np.float64)
    return data, graph, source


def _upsert_stats_row(csv_path: str, row: Dict) -> None:
    """Write ``row`` into the stats CSV, replacing an older row for its file."""
    with _STATS_LOCK:
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            existing = existing[existing["file"] != row["file"]]
            frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            frame = pd.DataFrame([row])
        frame = frame.sort_values("file", kind="stable")
        frame.to_csv(csv_path, index=False)


@BatchProcessingRegistry.register(
    name="Track Statistics (QC)",
    suffix="_track_qc",
    description=(
        "Check tracking results: for each tracked label image (TYX or TZYX, "
        "as written by Trackastra, HOCT or Ultrack) count tracks, their "
        "lengths, gaps, and tracks that start late or end early. Writes one "
        "row per file to track_statistics.csv in the output folder and, "
        "optionally, a per-file table of track positions that the results "
        "table shows as a Tracks layer."
    ),
    parameters={
        "min_track_length": {
            "type": int,
            "default": 3,
            "min": 1,
            "max": 100000,
            "description": "Tracks present in fewer frames than this are counted as short",
        },
        "save_track_table": {
            "type": bool,
            "default": True,
            "description": f"Also write <file>{TRACKS_CSV_SUFFIX} with every object's position per frame",
        },
    },
)
def track_statistics_qc(
    image: np.ndarray,
    min_track_length: int = 3,
    save_track_table: bool = True,
    dim_order: str = "Auto",
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = None,
):
    """
    Batch entry point: statistics for one tracked label image.

    Returns None; the results are CSV files, there is no image to save.
    Without an output folder (scripting), returns the statistics dict.
    """
    if not _HAS_PANDAS:
        raise ImportError("Track statistics need pandas")

    source = _source_filepath if _source_filepath else image
    if source is None:
        raise ValueError("No image or file path to compute track statistics for")
    table, n_frames = tracks_from_labels(source, dim_order)
    stats = track_statistics(table, n_frames, min_track_length)

    if not _output_folder:
        return stats

    os.makedirs(_output_folder, exist_ok=True)
    name = os.path.basename(str(_source_filepath or "array").rstrip("/"))
    _upsert_stats_row(
        os.path.join(_output_folder, STATS_CSV_NAME), {"file": name, **stats}
    )
    if save_track_table:
        stem = os.path.splitext(name)[0]
        table.to_csv(
            os.path.join(_output_folder, f"{stem}{TRACKS_CSV_SUFFIX}"),
            index=False,
        )
    print(
        f"📈 {name}: {stats['tracks']} tracks over {n_frames} frames, "
        f"{stats['tracks_full_length']} full length, "
        f"{stats['tracks_with_gaps']} with gaps, "
        f"{stats['tracks_short']} shorter than {min_track_length} frames"
    )
    return None


# The worker must not load the stack: tracked stacks are often tens of GB
# dense and this reads them one plane at a time.
track_statistics_qc.skip_load = True

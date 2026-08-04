# processing_functions/intensity_label_filter.py
"""
Processing functions for filtering labels based on intensity using k-medoids clustering.
"""
import inspect
from pathlib import Path
from typing import Dict

import numpy as np
from skimage import measure

from napari_tmidas._registry import BatchProcessingRegistry


def _resolve_spatial_ndim(dim_order: str, ndim: int) -> int:
    """
    Number of trailing axes that make up one spatial block (2 for YX, 3 for ZYX).

    Any axes in front of that block (T, C, ...) are iterated over instead of
    being handed to routines that only understand 2-D/3-D data.

    Parameters
    ----------
    dim_order : str
        Dimension order hint from the widget ("Auto", "TYX", "TZYX", ...)
    ndim : int
        Number of dimensions of the actual image

    Returns
    -------
    int
        2 or 3 (never more than ``ndim``)
    """
    hint = str(dim_order or "Auto").upper()
    if hint == "AUTO":
        # 2-D → YX, 3-D → ZYX, 4-D/5-D → leading axes are T/C
        return min(ndim, 3)
    # Only a trailing "ZYX" gives a contiguous 3-D block; orders such as
    # ZCYX interleave the channel axis, so their spatial block is just YX.
    spatial = 3 if hint.endswith("ZYX") else 2
    return min(spatial, ndim)


def _iter_spatial_blocks(image: np.ndarray, spatial_ndim: int):
    """
    Yield ``(index, block)`` for every 2-D/3-D block of a possibly nD image.

    ``index`` is a tuple addressing the leading (T/C) axes and can be used to
    write results back into an output array of the same shape.
    """
    leading_shape = image.shape[: image.ndim - spatial_ndim]
    if not leading_shape:
        yield (), image
        return
    for index in np.ndindex(*leading_shape):
        yield index, image[index]


def _collect_label_values(
    image: np.ndarray, spatial_ndim: int = None
) -> np.ndarray:
    """
    Sorted array of the distinct values in ``image``, computed block-wise.

    ``np.unique(image[image != 0])`` is the obvious way to do this, but on a
    large stack it allocates a full boolean mask *plus* a compressed copy of
    every foreground voxel *plus* a sorted copy of that — several times the
    image size in transient memory.  Going block by block bounds the temporary
    to one 2-D/3-D block, and the running result is bounded by the number of
    distinct labels.
    """
    if spatial_ndim is None:
        spatial_ndim = image.ndim
    seen = None
    for _, block in _iter_spatial_blocks(image, spatial_ndim):
        block_values = np.unique(block)
        seen = (
            block_values
            if seen is None
            else np.union1d(seen, block_values)
        )
    return seen if seen is not None else np.zeros(0, dtype=image.dtype)


def _convert_semantic_to_instance(
    image: np.ndarray,
    spatial_ndim: int = None,
    label_values: np.ndarray = None,
) -> np.ndarray:
    """
    Convert semantic labels (where all objects have the same value) to instance labels.

    Parameters
    ----------
    image : np.ndarray
        Label image that may contain semantic labels
    spatial_ndim : int, optional
        Number of trailing spatial axes.  When given and smaller than
        ``image.ndim``, each block is labelled independently (so objects are
        not connected across time/channels) and IDs are offset to stay unique.
    label_values : np.ndarray, optional
        Pre-computed distinct values (see :func:`_collect_label_values`).
        Passing them avoids a second full scan of the image.

    Returns
    -------
    np.ndarray
        Image with instance labels (each connected component gets unique label)
    """
    if image is None:
        return image

    if label_values is None:
        label_values = _collect_label_values(image, spatial_ndim)
    unique_labels = label_values[label_values != 0]

    # All background, or already instance labels — nothing to do
    if len(unique_labels) != 1:
        return image

    # Single semantic label - convert to instance labels
    if spatial_ndim is None or spatial_ndim >= image.ndim:
        return measure.label(image > 0, connectivity=None)

    # nD: label each spatial block on its own so components are not merged
    # across T/C, offsetting IDs so every object stays uniquely addressable.
    result = np.zeros(image.shape, dtype=np.int32)
    offset = 0
    for index, block in _iter_spatial_blocks(image, spatial_ndim):
        block_labels = measure.label(block > 0, connectivity=None)
        n_labels = int(block_labels.max())
        if n_labels:
            block_labels[block_labels > 0] += offset
            offset += n_labels
        result[index] = block_labels
    return result


try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False


def _calculate_label_mean_intensities(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    spatial_ndim: int = None,
) -> Dict[int, float]:
    """
    Calculate mean intensity for each label.

    Works for any dimensionality: sums and voxel counts are accumulated per
    label with ``np.bincount``, one spatial block at a time, so nothing ever
    sees more than 3 dimensions and no full-size float copy of the intensity
    image is created.  For tracked data (a label ID shared across timepoints)
    this yields the mean over the whole track.

    Parameters
    ----------
    label_image : np.ndarray
        Label image with integer labels
    intensity_image : np.ndarray
        Intensity image corresponding to the label image
    spatial_ndim : int, optional
        Number of trailing spatial axes; leading axes are iterated over.
        Defaults to treating the whole array as one block.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping label IDs to mean intensities
    """
    if label_image.dtype.kind not in "iub":
        label_image = label_image.astype(np.int64)
    if label_image.size == 0:
        return {}

    max_label = int(label_image.max())
    min_label = int(label_image.min())
    if min_label < 0:
        raise ValueError(
            f"Label image contains negative values (min: {min_label})"
        )
    if max_label == 0:
        return {}

    if spatial_ndim is None:
        spatial_ndim = label_image.ndim

    n_bins = max_label + 1
    counts = np.zeros(n_bins, dtype=np.int64)
    sums = np.zeros(n_bins, dtype=np.float64)

    for index, label_block in _iter_spatial_blocks(label_image, spatial_ndim):
        flat_labels = label_block.ravel()
        flat_intensity = np.asarray(intensity_image[index]).ravel()
        counts += np.bincount(flat_labels, minlength=n_bins)
        sums += np.bincount(
            flat_labels, weights=flat_intensity, minlength=n_bins
        )

    present = np.nonzero(counts)[0]
    present = present[present > 0]  # drop background
    means = sums[present] / counts[present]

    return {
        int(label_id): float(mean) for label_id, mean in zip(present, means)
    }


def _kmedoids_1d(
    values: np.ndarray, n_clusters: int, max_iter: int = 300
) -> np.ndarray:
    """
    K-medoids for one-dimensional data.

    General-purpose k-medoids libraries need an n×n distance matrix, which is
    both wasteful and a hard scaling limit here — we cluster a single scalar
    per label, and a densely tracked movie can have tens of thousands of them.
    In 1-D the problem collapses: within a cluster, the point minimising the
    summed absolute distance to all members is exactly the *weighted median*
    snapped to an actual data point.  So this runs in O(n log n) time and O(n)
    memory via Voronoi iteration, with no distance matrix at all.

    Initialisation is quantile-spaced (not random), so results are
    deterministic and need no ``random_state``.

    Parameters
    ----------
    values : np.ndarray
        1-D array of values to cluster
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum number of Voronoi iterations

    Returns
    -------
    np.ndarray
        Sorted medoids, each an actual element of ``values``
    """
    # Collapse duplicates and carry their multiplicity as weights, so the
    # medoid still minimises distance over *all* points, not distinct ones.
    unique, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)

    if len(unique) <= n_clusters:
        # Fewer distinct values than clusters — each value is its own medoid
        return unique

    # Deterministic quantile-spaced seeding
    seed_idx = np.unique(
        np.linspace(0, len(unique) - 1, n_clusters).round().astype(int)
    )
    medoids = unique[seed_idx]

    for _ in range(max_iter):
        # Assign every distinct value to its nearest medoid.  Medoids are
        # sorted, so the boundaries are the midpoints between consecutive ones.
        boundaries = (medoids[:-1] + medoids[1:]) / 2.0
        assignment = np.searchsorted(boundaries, unique)

        new_medoids = []
        for k in range(len(medoids)):
            member_mask = assignment == k
            if not member_mask.any():
                new_medoids.append(medoids[k])
                continue
            members = unique[member_mask]
            weights = counts[member_mask]
            # Weighted median = first point where the cumulative weight
            # reaches half the cluster's total weight
            cumulative = np.cumsum(weights)
            half = cumulative[-1] / 2.0
            new_medoids.append(members[np.searchsorted(cumulative, half)])

        new_medoids = np.unique(new_medoids)  # sorted, duplicates collapsed
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    return medoids


def _cluster_intensities(
    intensities: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Cluster intensities using k-medoids and determine threshold.

    Parameters
    ----------
    intensities : np.ndarray
        Array of intensity values to cluster
    n_clusters : int
        Number of clusters (2 or 3)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Cluster labels, cluster centers (medoids), and threshold value
    """
    intensities = np.asarray(intensities, dtype=np.float64)

    # Medoids come back sorted low → high, so cluster index 0 is already the
    # low-intensity cluster and no relabelling is needed.
    sorted_medoids = _kmedoids_1d(intensities, n_clusters)

    if len(sorted_medoids) < 2:
        raise ValueError(
            "Cannot separate intensity clusters: all labels share the same "
            f"mean intensity ({sorted_medoids[0]:.4g}). Nothing to filter."
        )

    boundaries = (sorted_medoids[:-1] + sorted_medoids[1:]) / 2.0
    sorted_labels = np.searchsorted(boundaries, intensities)

    # Threshold between the lowest and second-lowest clusters
    threshold = float((sorted_medoids[0] + sorted_medoids[1]) / 2.0)

    return sorted_labels, sorted_medoids, threshold


def _get_intensity_filename(
    label_filename: str, label_suffix: str = "_convpaint_labels_filtered.tif"
) -> str:
    """
    Convert label filename to intensity filename by removing suffix.

    Parameters
    ----------
    label_filename : str
        Filename of the label image
    label_suffix : str
        Suffix to remove from label filename (default: "_convpaint_labels_filtered.tif")

    Returns
    -------
    str
        Intensity image filename
    """
    if label_filename.endswith(label_suffix):
        # Remove the label suffix and add .tif
        base_name = label_filename[: -len(label_suffix)]
        return base_name + ".tif"
    else:
        # If suffix doesn't match, assume same filename
        return label_filename


# Suffixes tried when label_suffix="auto".  Longest first so that a more
# specific suffix wins over a prefix of itself.
_KNOWN_LABEL_SUFFIXES = (
    "_convpaint_labels_filtered",
    "_hoct_tracked",
    "_trackastra_tracked",
    "_ultrack_tracked",
    "_tracked",
    "_labels",
    "_masks",
)

# Extensions an intensity source may have, in preference order
_INTENSITY_EXTENSIONS = (".zarr", ".tif", ".tiff", ".ome.tif", ".ome.tiff")


def _resolve_intensity_source(label_path: Path, label_suffix: str = "auto"):
    """
    Locate the intensity image that pairs with a label image.

    Parameters
    ----------
    label_path : Path
        Path of the label image being processed
    label_suffix : str
        Suffix to strip from the label filename before searching, or "auto"
        to try a list of known conventions.

    Returns
    -------
    Path
        Path of the intensity source (a .tif or a .zarr)

    Raises
    ------
    FileNotFoundError
        If no candidate exists.  This deliberately does *not* fall back to the
        label image itself: doing so silently clusters label IDs instead of
        intensities and produces plausible-looking nonsense.
    """
    stem = label_path.name
    for ext in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    if label_suffix and label_suffix.lower() != "auto":
        candidates_stems = [
            stem[: -len(label_suffix)]
            if stem.endswith(label_suffix)
            else stem
        ]
    else:
        candidates_stems = [
            stem[: -len(suffix)]
            for suffix in _KNOWN_LABEL_SUFFIXES
            if stem.endswith(suffix)
        ]

    tried = []
    for base in candidates_stems:
        for ext in _INTENSITY_EXTENSIONS:
            candidate = label_path.parent / (base + ext)
            tried.append(candidate.name)
            if candidate.exists() and candidate != label_path:
                return candidate

    suffix_hint = (
        f"none of the known suffixes {list(_KNOWN_LABEL_SUFFIXES)} matched"
        if label_suffix.lower() == "auto"
        else f"suffix {label_suffix!r} did not resolve to an existing file"
    )
    raise FileNotFoundError(
        f"No intensity image found for label image {label_path.name!r} "
        f"({suffix_hint}).\n"
        f"  Tried: {tried if tried else '(no candidates)'}\n"
        f"  Set the 'label_suffix' parameter to the suffix that turns the "
        f"label filename into the intensity filename."
    )


class _PlaneReader:
    """
    Uniform per-YX-plane reader over a TIFF or a zarr array.

    A tracked stack is typically >90% background, so materialising it densely
    costs orders of magnitude more RAM than the information it carries.  Both
    formats this package produces are already chunked per plane — a compressed
    TIFF stores one IFD per plane, and an OME-Zarr chunks (1, 1, 1, Y, X) — so
    reading a plane at a time keeps peak memory at a few MB regardless of how
    many timepoints and z-slices the file has.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tif = None
        self._array = None

        if self.path.suffix.lower() == ".zarr" or self.path.is_dir():
            import zarr

            node = zarr.open(str(self.path), mode="r")
            self._array = self._first_array(node)
            self.shape = tuple(self._array.shape)
            self.dtype = np.dtype(self._array.dtype)
        else:
            import tifffile

            self._tif = tifffile.TiffFile(str(self.path))
            series = self._tif.series[0]
            self.shape = tuple(series.shape)
            self.dtype = np.dtype(series.dtype)
            self._pages = None

            # 1. A memory map is the best case: full nD indexing, paged by the
            #    OS, no RAM cost.  Works for uncompressed TIFFs, including
            #    ImageJ hyperstacks that store the whole stack in a single IFD.
            try:
                self._array = tifffile.memmap(str(self.path), mode="r")
                if tuple(self._array.shape) != self.shape:
                    self._array = self._array.reshape(self.shape)
            except (ValueError, MemoryError, OSError, NotImplementedError):
                self._array = None

            # 2. Otherwise fall back to per-page reads, which is what makes
            #    compressed stacks streamable — but only when pages really do
            #    map one-to-one onto YX planes.
            if self._array is None:
                if len(series.pages) == int(np.prod(self.shape[:-2]) or 1):
                    self._pages = series.pages
                else:
                    # 3. Neither addressing scheme applies (e.g. a compressed
                    #    ImageJ hyperstack). Read once and keep it; correctness
                    #    first, and say so rather than failing.
                    print(
                        f"⚠️  {self.path.name}: {len(series.pages)} page(s) for "
                        f"shape {self.shape} and not memory-mappable — reading "
                        "it into RAM (cannot stream this layout)"
                    )
                    self._array = series.asarray()

    @staticmethod
    def _first_array(node):
        """Highest-resolution array in a (possibly multiscale) zarr node."""
        if hasattr(node, "shape"):
            return node
        # Multiscale group: prefer 's0'/'0', else the largest array present
        for key in ("s0", "0"):
            if key in node:
                return node[key]
        arrays = [a for _, a in node.arrays()]
        if not arrays:
            raise ValueError(f"No arrays found in zarr group {node}")
        return max(arrays, key=lambda a: int(np.prod(a.shape)))

    @property
    def leading_shape(self):
        return self.shape[:-2]

    def plane(self, index: tuple) -> np.ndarray:
        """Read the YX plane addressed by ``index`` over the leading axes."""
        if self._array is not None:
            return np.asarray(self._array[index])
        flat = (
            int(np.ravel_multi_index(index, self.leading_shape))
            if index
            else 0
        )
        return self._pages[flat].asarray()

    def close(self):
        if self._tif is not None:
            self._tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _locate_channel_axis(label_shape: tuple, intensity_shape: tuple):
    """
    Index of the intensity axis that has no counterpart in the label image.

    Returns None when the shapes already match.  Raises when they cannot be
    reconciled by dropping a single axis.
    """
    label_shape = tuple(label_shape)
    intensity_shape = tuple(intensity_shape)
    if intensity_shape == label_shape:
        return None
    if len(intensity_shape) == len(label_shape) + 1:
        for axis in range(len(intensity_shape)):
            if (
                intensity_shape[:axis] + intensity_shape[axis + 1 :]
                == label_shape
            ):
                return axis
    raise ValueError(
        f"Label and intensity images cannot be matched. "
        f"Label: {label_shape}, Intensity: {intensity_shape}. "
        "They must have the same shape, or the intensity image may have one "
        "extra (channel) axis."
    )


def _filter_labels_by_threshold(
    label_image: np.ndarray,
    label_intensities: Dict[int, float],
    threshold: float,
    out_dtype: np.dtype = None,
    spatial_ndim: int = None,
) -> np.ndarray:
    """
    Filter labels based on intensity threshold.

    Parameters
    ----------
    label_image : np.ndarray
        Label image with integer labels
    label_intensities : Dict[int, float]
        Dictionary mapping label IDs to mean intensities
    threshold : float
        Intensity threshold - labels below this are removed
    out_dtype : np.dtype, optional
        dtype for the result.  Defaults to the input dtype.  Passing a
        narrower integer type avoids carrying e.g. int64 labels that only
        need uint16, which on a large stack saves several GB.
    spatial_ndim : int, optional
        Number of trailing spatial axes.  The lookup is applied block-wise so
        no full-size temporary is created on top of the output array.

    Returns
    -------
    np.ndarray
        Filtered label image
    """
    if out_dtype is None:
        out_dtype = label_image.dtype

    if not label_intensities:
        return label_image.astype(out_dtype, copy=True)

    if label_image.dtype.kind not in "iu" or label_image.min() < 0:
        # Fallback for float/signed label images: one pass per label
        filtered_image = label_image.astype(out_dtype, copy=True)
        for label_id, intensity in label_intensities.items():
            if intensity < threshold:
                filtered_image[label_image == label_id] = 0
        return filtered_image

    # Lookup table: one pass over the image regardless of label count, which
    # matters for tracked stacks with thousands of labels.
    n_bins = max(int(label_image.max()), max(label_intensities)) + 1
    lut = np.arange(n_bins, dtype=np.int64)
    for label_id, intensity in label_intensities.items():
        if intensity < threshold:
            lut[label_id] = 0
    lut = lut.astype(out_dtype)

    # Fill a pre-allocated output block by block.  `lut[label_image]` in one
    # shot would materialise an extra full-size intermediate when the output
    # dtype differs from the LUT's.
    if spatial_ndim is None:
        spatial_ndim = label_image.ndim
    filtered_image = np.empty(label_image.shape, dtype=out_dtype)
    for index, block in _iter_spatial_blocks(label_image, spatial_ndim):
        np.take(lut, block, out=filtered_image[index])

    return filtered_image


def _smallest_label_dtype(max_label: int, current: np.dtype) -> np.dtype:
    """
    Narrowest unsigned integer dtype that still holds ``max_label``.

    Tracking tools routinely emit int64 label images; with a few thousand
    labels that is twice the memory uint32 needs, for both the array in RAM and
    the file on disk.  Narrowing is lossless for label data.

    **uint32 is the floor.**  Both napari's ``guess_labels`` and this package's
    ``is_label_image`` treat only int32/uint32/int64/uint64 as label data —
    anything narrower is auto-loaded as a grayscale *image* layer instead of
    coloured labels.  Narrowing past uint32 would therefore break display.

    An input that is already narrower than uint32 is left alone rather than
    widened; that is the caller's choice and widening only costs memory.
    """
    if current.kind not in "iu":
        return current

    if max_label <= np.iinfo(current).max:
        # Current dtype already holds every ID — only ever narrow, never below
        # uint32, and never touch types that are already uint32 or smaller.
        if current.itemsize <= np.dtype(np.uint32).itemsize:
            return current
        if max_label <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        return current

    # Current dtype would overflow: widen to the smallest type that fits
    for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_label <= np.iinfo(candidate).max:
            return np.dtype(candidate)
    return current


def _save_cluster_stats(
    label_path: Path,
    label_filename: str,
    n_clusters: int,
    total_labels: int,
    medoids,
    threshold: float,
    n_removed: int,
) -> None:
    """Append one row of clustering statistics to the per-folder CSV."""
    if not _HAS_PANDAS:
        return
    stats = {
        "filename": label_filename,
        "n_clusters": n_clusters,
        "total_labels": total_labels,
        "removed_labels": n_removed,
        "kept_labels": total_labels - n_removed,
        "threshold": threshold,
    }
    for i, medoid in enumerate(medoids):
        stats[f"medoid_{i}"] = float(medoid)

    stats_dir = label_path.parent / "intensity_filter_stats"
    stats_dir.mkdir(exist_ok=True)
    stats_file = stats_dir / "clustering_stats.csv"
    df = pd.DataFrame([stats])
    if stats_file.exists():
        df.to_csv(stats_file, mode="a", header=False, index=False)
    else:
        df.to_csv(stats_file, index=False)


def _stream_filter_labels_by_intensity(
    label_path: Path,
    intensity_path: Path,
    n_clusters: int,
    intensity_channel: int,
    output_path: Path,
) -> tuple:
    """
    Filter labels by intensity without ever holding the stack in memory.

    Two streaming passes over the data, one YX plane at a time:

    1. accumulate per-label voxel counts and intensity sums (``np.bincount``)
    2. apply the resulting lookup table and write each plane straight out

    Peak memory is one label plane plus one intensity plane — a few MB — no
    matter how large the stack is.

    Returns
    -------
    tuple
        (label_intensities, medoids, threshold, out_dtype, n_labels)
    """
    import tifffile

    with _PlaneReader(label_path) as labels, _PlaneReader(
        intensity_path
    ) as intensity:
        channel_axis = _locate_channel_axis(labels.shape, intensity.shape)
        if channel_axis is not None:
            n_channels = intensity.shape[channel_axis]
            if not 0 <= intensity_channel < n_channels:
                raise ValueError(
                    f"intensity_channel={intensity_channel} is out of range: "
                    f"{intensity_path.name} has {n_channels} channels "
                    f"(axis {channel_axis})."
                )
            print(
                f"🎨 Intensity image has {n_channels} channels on axis "
                f"{channel_axis}; using channel {intensity_channel}"
            )

        def intensity_index(index):
            if channel_axis is None:
                return index
            return (
                index[:channel_axis]
                + (intensity_channel,)
                + index[channel_axis:]
            )

        leading = labels.leading_shape
        n_planes = int(np.prod(leading)) if leading else 1
        plane_mb = (
            np.prod(labels.shape[-2:]) * labels.dtype.itemsize / 1e6
        )
        print(
            f"🌊 Streaming {n_planes} planes of "
            f"{labels.shape[-2]}x{labels.shape[-1]} "
            f"({plane_mb:.1f} MB per label plane) — the {np.prod(labels.shape) * labels.dtype.itemsize / 1e9:.1f} GB "
            "stack is never held in memory"
        )

        # --- Pass 1: per-label counts and intensity sums -------------------
        counts = None
        sums = None
        max_label = 0
        for n_done, index in enumerate(
            np.ndindex(*leading) if leading else [()], start=1
        ):
            label_plane = labels.plane(index)
            if label_plane.min() < 0:
                raise ValueError(
                    f"Label image contains negative values in plane {index}"
                )
            plane_max = int(label_plane.max())
            if plane_max > max_label:
                grown = np.zeros(plane_max + 1, dtype=np.int64)
                grown_sums = np.zeros(plane_max + 1, dtype=np.float64)
                if counts is not None:
                    grown[: counts.size] = counts
                    grown_sums[: sums.size] = sums
                counts, sums = grown, grown_sums
                max_label = plane_max
            if max_label == 0:
                continue

            flat = label_plane.ravel()
            intensity_plane = np.asarray(
                intensity.plane(intensity_index(index))
            ).ravel()
            counts += np.bincount(flat, minlength=counts.size)
            sums += np.bincount(
                flat, weights=intensity_plane, minlength=sums.size
            )
            if n_done % 200 == 0 or n_done == n_planes:
                print(
                    f"   pass 1: {n_done}/{n_planes} planes", flush=True
                )

        if counts is None or max_label == 0:
            raise ValueError(f"No labels found in {label_path.name}")

        present = np.nonzero(counts)[0]
        present = present[present > 0]
        label_intensities = {
            int(lab): float(s / c)
            for lab, s, c in zip(present, sums[present], counts[present])
        }
        print(f"📋 Found {len(label_intensities)} labels across the stack")

        # --- Cluster and build the lookup table ---------------------------
        intensities = np.fromiter(
            label_intensities.values(), dtype=np.float64
        )
        _, medoids, threshold = _cluster_intensities(intensities, n_clusters)

        out_dtype = _smallest_label_dtype(int(present.max()), labels.dtype)
        lut = np.arange(max_label + 1, dtype=np.int64)
        for label_id, mean_intensity in label_intensities.items():
            if mean_intensity < threshold:
                lut[label_id] = 0
        lut = lut.astype(out_dtype)

        # --- Pass 2: apply the LUT and stream straight to disk -------------
        def plane_iterator():
            for n_done, index in enumerate(
                np.ndindex(*leading) if leading else [()], start=1
            ):
                yield np.take(lut, labels.plane(index))
                if n_done % 200 == 0 or n_done == n_planes:
                    print(
                        f"   pass 2: {n_done}/{n_planes} planes", flush=True
                    )

        axes = {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}.get(
            len(labels.shape)
        )
        print(
            f"💾 Writing {output_path.name} as {out_dtype} "
            f"(compressed, streamed plane by plane)"
        )
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as writer:
            writer.write(
                plane_iterator(),
                shape=labels.shape,
                dtype=out_dtype,
                compression="zlib",
                # Without this, tifffile reads a leading axis of length 3 or 4
                # (e.g. a 4-timepoint stack) as RGB samples, stores separate
                # component planes, and the plane iterator raises on the first
                # write.
                photometric="minisblack",
                metadata={"axes": axes} if axes else None,
            )

        return (
            label_intensities,
            medoids,
            threshold,
            out_dtype,
            len(label_intensities),
        )


@BatchProcessingRegistry.register(
    name="Filter Labels by Intensity (K-medoids)",
    suffix="_intensity_filtered",
    description="Filter out labels with low intensity using k-medoids clustering. Streams the stack plane by plane, so memory stays flat regardless of size. Finds the matching intensity image (.tif or .zarr) in the same folder. Choose 2 clusters for simple low/high separation, or 3 clusters when you have distinct noise/signal/strong-signal populations.",
    parameters={
        "n_clusters": {
            "type": int,
            "default": 2,
            "description": "Number of clusters (2 or 3). Use 2 for simple low/high separation, 3 for noise/diffuse/strong separation.",
        },
        "label_suffix": {
            "type": str,
            "default": "auto",
            "description": "Suffix that turns the label filename into the intensity filename, e.g. '_hoct_tracked'. 'auto' tries known conventions. Fails loudly if no intensity image is found.",
        },
        "intensity_channel": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 100,
            "description": "Channel to measure when the intensity image has a channel axis the label image does not.",
        },
        "save_stats": {
            "type": bool,
            "default": True,
            "description": "Save clustering statistics to CSV file",
        },
    },
)
def filter_labels_by_intensity(
    image: np.ndarray = None,
    n_clusters: int = 2,
    save_stats: bool = True,
    dim_order: str = "Auto",
    label_suffix: str = "auto",
    intensity_channel: int = 0,
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = None,
) -> np.ndarray:
    """
    Filter labels based on intensity using k-medoids clustering.

    For each label image the matching intensity image is located in the same
    folder (see ``label_suffix``), the mean intensity of every label is
    measured, k-medoids splits those means into groups, and labels in the
    lowest group are removed.

    Use n_clusters=2 for simple separation (bad vs. good signal).
    Use n_clusters=3 when you have distinct populations (noise, diffuse signal, strong signal).

    Two execution paths
    -------------------
    *Streaming* (default in the widget): the label and intensity files are read
    one YX plane at a time and the result is written straight to disk, so peak
    memory is a few MB regardless of stack size.  Tracked stacks are typically
    >90% background, so this is orders of magnitude cheaper than the dense
    array — a 30 GB stack streams in about a minute.

    *In-memory*: used when an array is passed directly (tests, scripting).

    Parameters
    ----------
    image : np.ndarray, optional
        Label image.  ``None`` selects the streaming path (the widget passes
        ``None`` because this function sets ``skip_load``).
    n_clusters : int
        Number of clusters (2 or 3)
    save_stats : bool
        Whether to save clustering statistics to CSV
    dim_order : str
        Dimension order hint from the widget.  Determines which trailing axes
        are spatial; leading T/C axes are handled by iteration.  Images of any
        dimensionality (YX up to TCZYX) are supported.
    label_suffix : str
        Suffix that turns the label filename into the intensity filename, e.g.
        "_hoct_tracked".  "auto" tries a list of known conventions.  If no
        intensity image is found the run fails rather than silently using the
        label image as its own intensity.
    intensity_channel : int
        Channel to measure when the intensity image has a channel axis that the
        label image does not.

    Returns
    -------
    np.ndarray or str
        The filtered label image, or — on the streaming path — the path of the
        file that was written.
    """
    # Extract current filepath: explicit parameter first, call stack as fallback
    current_filepath = _source_filepath
    if current_filepath is None:
        for frame_info in inspect.stack():
            frame_locals = frame_info.frame.f_locals
            if "filepath" in frame_locals:
                current_filepath = frame_locals["filepath"]
                break

    if current_filepath is None:
        raise ValueError(
            "Could not determine current file path from call stack"
        )

    if n_clusters not in [2, 3]:
        raise ValueError(f"n_clusters must be 2 or 3, got {n_clusters}")

    label_path = Path(current_filepath)

    # ------------------------------------------------------------------
    # Streaming path: never materialise the stack
    # ------------------------------------------------------------------
    if image is None:
        intensity_path = _resolve_intensity_source(label_path, label_suffix)
        print(f"🔗 Intensity source: {intensity_path.name}")

        if _output_folder and _output_suffix:
            output_path = Path(_output_folder) / (
                label_path.stem + _output_suffix + ".tif"
            )
            Path(_output_folder).mkdir(parents=True, exist_ok=True)
        else:
            output_path = label_path.parent / (
                label_path.stem + "_intensity_filtered.tif"
            )

        (
            label_intensities,
            medoids,
            threshold,
            out_dtype,
            n_labels,
        ) = _stream_filter_labels_by_intensity(
            label_path,
            intensity_path,
            n_clusters=n_clusters,
            intensity_channel=intensity_channel,
            output_path=output_path,
        )

        n_removed = sum(
            1 for v in label_intensities.values() if v < threshold
        )
        print(f"📊 {label_path.name}:")
        print(f"   Total labels: {n_labels}")
        print(f"   Medoids: {[round(float(m), 2) for m in medoids]}")
        print(f"   Threshold: {threshold:.2f}")
        print(
            f"   Keeping {n_labels - n_removed} labels, removing {n_removed}"
        )

        if save_stats and _HAS_PANDAS:
            _save_cluster_stats(
                label_path,
                label_path.name,
                n_clusters,
                n_labels,
                medoids,
                threshold,
                n_removed,
            )

        print(f"✅ Wrote {output_path}")
        return str(output_path)

    # Work out how many trailing axes are spatial; anything in front (T, C)
    # is iterated over so that any shape from the dim-order dropdown works.
    spatial_ndim = _resolve_spatial_ndim(dim_order, image.ndim)
    if image.ndim > spatial_ndim:
        print(
            f"🧭 Image shape {image.shape} (dim_order={dim_order}): "
            f"iterating over {image.ndim - spatial_ndim} leading axis/axes, "
            f"{spatial_ndim}-D spatial blocks"
        )

    # One block-wise scan for the distinct values, reused for the semantic
    # check and the label inventory.  Scanning the whole array with
    # `image[image != 0]` would allocate several times the image size.
    original_dtype = image.dtype
    label_values = _collect_label_values(image, spatial_ndim)

    # Convert semantic labels to instance labels if needed
    converted = _convert_semantic_to_instance(
        image, spatial_ndim=spatial_ndim, label_values=label_values
    )
    if converted is not image:
        image = converted
        label_values = _collect_label_values(image, spatial_ndim)

    unique_labels = label_values[label_values != 0]
    if len(unique_labels) == 0:
        print("⚠️  No labels found in image, returning empty image")
        return np.zeros_like(image)

    print(f"📋 Found {len(unique_labels)} labels in the image")

    # Find corresponding intensity image in same folder
    label_path = Path(current_filepath)
    label_filename = label_path.name
    intensity_filename = _get_intensity_filename(label_filename)
    intensity_path = label_path.parent / intensity_filename

    if not intensity_path.exists():
        print(
            f"⚠️  No corresponding intensity image found for {label_filename}"
        )
        print(f"   Expected: {intensity_filename}")
        print(f"   Full path: {intensity_path}")
        print("   Skipping this file...")
        return image  # Return original image unchanged

    # Load the intensity image.  It is only ever read block-wise, so prefer a
    # memory map — that keeps a second full-size array out of RAM.  Memory
    # mapping requires an uncompressed, contiguously stored TIFF; compressed
    # files fall back to a normal read.
    try:
        import tifffile

        try:
            intensity_image = tifffile.memmap(str(intensity_path), mode="r")
            print(f"🗺️  Memory-mapped intensity image {intensity_path.name}")
        except (ValueError, MemoryError, OSError):
            intensity_image = tifffile.imread(str(intensity_path))
            print(
                f"📖 Read intensity image {intensity_path.name} into RAM "
                f"({intensity_image.nbytes / 1e9:.1f} GB) — not memory-mappable "
                "(compressed or non-contiguous TIFF)"
            )
    except (FileNotFoundError, OSError) as e:
        print(f"⚠️  Could not read intensity image: {intensity_path}")
        print(f"   Error: {e}")
        print("   Skipping this file...")
        return image  # Return original if can't read intensity image

    # Validate dimensions match
    if image.shape != intensity_image.shape:
        raise ValueError(
            f"Label and intensity images must have same shape. "
            f"Label: {image.shape}, Intensity: {intensity_image.shape}"
        )

    # Calculate mean intensity for each label (across all timepoints/channels;
    # for tracked labels this is the mean over the whole track)
    label_intensities = _calculate_label_mean_intensities(
        image, intensity_image, spatial_ndim=spatial_ndim
    )

    if len(label_intensities) == 0:
        print(f"⚠️  No labels found in {label_filename}, returning empty image")
        return np.zeros_like(image)

    # Perform k-medoids clustering
    intensities = np.array(list(label_intensities.values()))
    cluster_labels, medoids, threshold = _cluster_intensities(
        intensities, n_clusters=n_clusters
    )

    # Print results based on the clusters actually found.  Fewer medoids than
    # requested means the data did not support that many distinct populations.
    print(f"📊 {label_filename}:")
    print(f"   Total labels: {len(label_intensities)}")

    if len(medoids) < n_clusters:
        print(
            f"   ⚠️  Requested {n_clusters} clusters but the intensities only "
            f"separate into {len(medoids)}; reporting {len(medoids)}."
        )
        n_clusters = len(medoids)

    if n_clusters == 2:
        n_low = np.sum(cluster_labels == 0)
        n_high = np.sum(cluster_labels == 1)
        print(
            f"   Low intensity cluster: {n_low} labels (medoid: {medoids[0]:.2f})"
        )
        print(
            f"   High intensity cluster: {n_high} labels (medoid: {medoids[1]:.2f})"
        )
        print(f"   Threshold: {threshold:.2f}")
        print(f"   Keeping {n_high} labels, removing {n_low} labels")

        # Save statistics if requested
        if save_stats and _HAS_PANDAS:
            stats = {
                "filename": label_filename,
                "n_clusters": n_clusters,
                "total_labels": len(label_intensities),
                "low_cluster_count": n_low,
                "high_cluster_count": n_high,
                "low_cluster_medoid": medoids[0],
                "high_cluster_medoid": medoids[1],
                "threshold": threshold,
            }

            stats_dir = (
                Path(current_filepath).parent / "intensity_filter_stats"
            )
            stats_dir.mkdir(exist_ok=True)
            stats_file = stats_dir / "clustering_stats.csv"

            df = pd.DataFrame([stats])
            if stats_file.exists():
                df.to_csv(stats_file, mode="a", header=False, index=False)
            else:
                df.to_csv(stats_file, index=False)

    else:  # n_clusters == 3
        n_low = np.sum(cluster_labels == 0)
        n_medium = np.sum(cluster_labels == 1)
        n_high = np.sum(cluster_labels == 2)
        print(
            f"   Low intensity cluster: {n_low} labels (medoid: {medoids[0]:.2f})"
        )
        print(
            f"   Medium intensity cluster: {n_medium} labels (medoid: {medoids[1]:.2f})"
        )
        print(
            f"   High intensity cluster: {n_high} labels (medoid: {medoids[2]:.2f})"
        )
        print(f"   Threshold: {threshold:.2f}")
        print(
            f"   Keeping {n_medium + n_high} labels, removing {n_low} labels"
        )

        # Save statistics if requested
        if save_stats and _HAS_PANDAS:
            stats = {
                "filename": label_filename,
                "n_clusters": n_clusters,
                "total_labels": len(label_intensities),
                "low_cluster_count": n_low,
                "medium_cluster_count": n_medium,
                "high_cluster_count": n_high,
                "low_cluster_medoid": medoids[0],
                "medium_cluster_medoid": medoids[1],
                "high_cluster_medoid": medoids[2],
                "threshold": threshold,
            }

            stats_dir = (
                Path(current_filepath).parent / "intensity_filter_stats"
            )
            stats_dir.mkdir(exist_ok=True)
            stats_file = stats_dir / "clustering_stats.csv"

            df = pd.DataFrame([stats])
            if stats_file.exists():
                df.to_csv(stats_file, mode="a", header=False, index=False)
            else:
                df.to_csv(stats_file, index=False)

    # Pick the output dtype before allocating anything.  Tracking tools often
    # emit int64 labels; narrowing to the smallest type that still holds every
    # ID is lossless and, on a large stack, saves several GB in RAM and on disk.
    out_dtype = _smallest_label_dtype(int(unique_labels.max()), original_dtype)
    if out_dtype != original_dtype:
        saved_gb = (
            image.size * (original_dtype.itemsize - out_dtype.itemsize) / 1e9
        )
        print(
            f"🗜️  Narrowing labels {original_dtype} → {out_dtype} "
            f"(max label {int(unique_labels.max())}), saving {saved_gb:.1f} GB"
        )

    # Filter labels straight into an output of that dtype
    filtered_image = _filter_labels_by_threshold(
        image,
        label_intensities,
        threshold,
        out_dtype=out_dtype,
        spatial_ndim=spatial_ndim,
    )

    return filtered_image


# skip_load=True: the worker must NOT call load_image_file for this function.
# A tracked stack is >90% background, so materialising it densely (30 GB for a
# 50 MB file) is exactly the cost the streaming path exists to avoid.  With
# image=None the function opens the label and intensity files itself and reads
# them one plane at a time.
filter_labels_by_intensity.skip_load = True

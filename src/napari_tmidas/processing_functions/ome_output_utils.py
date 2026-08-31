import inspect
import json
import os
import time
from contextlib import suppress
from typing import Any, Optional

import numpy as np
import tifffile

# Upper bound on how much of a lazy result is materialized at once when
# streaming it to disk.  Peak RAM for such a write is a small multiple of this
# block -- the threaded scheduler decompresses several source chunks
# concurrently to fill one, measured at ~7x -- but it stays independent of how
# large the full output is.  Size it against available RAM with that in mind.
STREAM_BLOCK_BYTES = 256 * 1024 * 1024


_WRITE_IMAGE_ACCEPTS_SCALE: Optional[bool] = None


def _write_image_accepts_scale() -> bool:
    """Whether the installed ome-zarr takes pixel size via ``scale``."""
    global _WRITE_IMAGE_ACCEPTS_SCALE
    if _WRITE_IMAGE_ACCEPTS_SCALE is None:
        try:
            from ome_zarr.writer import write_image

            _WRITE_IMAGE_ACCEPTS_SCALE = (
                "scale" in inspect.signature(write_image).parameters
            )
        except (ImportError, TypeError, ValueError):
            _WRITE_IMAGE_ACCEPTS_SCALE = False
    return _WRITE_IMAGE_ACCEPTS_SCALE


def physical_scale_kwargs(scale_transform: Optional[dict], axes: Any) -> dict:
    """
    Build the ``write_image()`` keyword that carries physical pixel size.

    ome-zarr 0.18 deprecated ``coordinate_transformations`` in favour of an
    axis-keyed ``scale`` dict, and it *silently ignores* the old argument
    rather than raising -- every axis lands in the file as 1.0, so converted
    images lose their pixel size with nothing in the output to say so.
    Releases up to 0.16 have no ``scale`` parameter at all, so neither
    argument works everywhere; pick whichever the installed version honours.
    """
    if not scale_transform:
        return {}
    scales = scale_transform.get("scale")
    if not scales:
        return {}

    legacy = {"coordinate_transformations": [[scale_transform]]}
    if not _write_image_accepts_scale():
        return legacy

    axis_names = [str(axis).lower() for axis in (axes or "")]
    if len(axis_names) != len(scales):
        # Without one name per value the axes cannot be keyed reliably, and a
        # misassigned scale is worse than a dropped one.
        return legacy
    return {"scale": dict(zip(axis_names, (float(s) for s in scales)))}


def _read_root_attrs(source_path: str) -> dict:
    attrs = {}
    zattrs_path = os.path.join(source_path, ".zattrs")
    if os.path.exists(zattrs_path):
        try:
            with open(zattrs_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                attrs.update(loaded)
        except Exception:
            pass

    zarr_json_path = os.path.join(source_path, "zarr.json")
    if os.path.exists(zarr_json_path):
        try:
            with open(zarr_json_path, encoding="utf-8") as f:
                loaded = json.load(f)
            zarr_attrs = loaded.get("attributes", {}) if isinstance(loaded, dict) else {}
            if isinstance(zarr_attrs, dict):
                attrs.update(zarr_attrs)
        except Exception:
            pass
    return attrs


def save_root_attrs(group_path: str, attrs: dict) -> bool:
    """
    Write a zarr group's attributes back, for either metadata layout.

    The counterpart of :func:`_read_root_attrs`: zarr v2 keeps them in
    ``.zattrs``, v3 nests them under ``attributes`` in ``zarr.json``.  Code
    that patches metadata after ``write_image`` must go through here -- a
    plain ``.zattrs`` write is a silent no-op on a v3 store, and assigning to
    ``group.attrs`` re-serialises a stale in-memory document that clobbers
    what ``write_image`` just wrote.

    Returns whether anything was written.
    """
    zattrs_path = os.path.join(group_path, ".zattrs")
    if os.path.exists(zattrs_path):
        with open(zattrs_path, "w", encoding="utf-8") as f:
            json.dump(attrs, f, indent=2)
        return True

    zarr_json_path = os.path.join(group_path, "zarr.json")
    if os.path.exists(zarr_json_path):
        with open(zarr_json_path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            return False
        doc["attributes"] = attrs
        with open(zarr_json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        return True

    return False


def _get_multiscales(attrs: dict) -> list:
    multiscales = attrs.get("multiscales", [])
    if isinstance(multiscales, list) and multiscales:
        return multiscales
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        multiscales = ome.get("multiscales", [])
        if isinstance(multiscales, list) and multiscales:
            return multiscales
    return []


def _extract_tiff_physical_scale(source_path: str, axes: str) -> dict:
    """TIFF counterpart of the zarr branch in _extract_source_physical_scale
    below: reads OME PhysicalSize{X,Y,Z} from an OME-TIFF's own XML metadata
    (mirrors _reader.py's _ome_scale_for_series, but from a path rather than
    an already-open TiffFile, and keyed by axis letter to match this
    module's dict contract)."""
    try:
        with tifffile.TiffFile(source_path) as tf:
            if not getattr(tf, "is_ome", False) or not tf.ome_metadata:
                return {}
            pixels = tifffile.xml2dict(tf.ome_metadata)["OME"]["Image"][
                "Pixels"
            ]
            if isinstance(pixels, list):
                pixels = pixels[0]
    except Exception:
        return {}
    if not isinstance(pixels, dict):
        return {}
    result = {}
    for ax_char in axes:
        ax_upper = ax_char.upper()
        if ax_upper in ("X", "Y", "Z"):
            val = pixels.get(f"PhysicalSize{ax_upper}")
            if val is not None:
                with suppress(TypeError, ValueError):
                    result[ax_upper] = float(val)
    return result


def _extract_source_physical_scale(source_path: Optional[str], axes: str) -> dict:
    """
    Return {"X": scale, "Y": scale, "Z": scale} (only for axes present in
    `axes`) read from the source's level-0 OME metadata, or {} if
    unavailable. Used to embed correct voxel spacing in OME-TIFF output so
    it displays at the same physical extent as the source in viewers that
    don't otherwise know the two files should share a scale.
    """
    if not source_path:
        return {}
    if source_path.lower().endswith((".tif", ".tiff")):
        return _extract_tiff_physical_scale(source_path, axes)
    attrs = _read_root_attrs(source_path)
    multiscales = _get_multiscales(attrs)
    if not multiscales:
        return {}
    src_ms = multiscales[0]
    if not isinstance(src_ms, dict):
        return {}
    src_axis_names = [
        str(a.get("name") if isinstance(a, dict) else a).lower()
        for a in src_ms.get("axes", [])
    ]
    src_datasets = src_ms.get("datasets", [])
    if not src_datasets:
        return {}
    for ctf in src_datasets[0].get("coordinateTransformations", []):
        if not isinstance(ctf, dict) or ctf.get("type") != "scale":
            continue
        scale = ctf.get("scale")
        if not (isinstance(scale, list) and len(scale) == len(src_axis_names)):
            continue
        result = {}
        for ax_char in axes:
            ax_lower = ax_char.lower()
            if ax_lower in ("x", "y", "z") and ax_lower in src_axis_names:
                result[ax_char.upper()] = scale[src_axis_names.index(ax_lower)]
        return result
    return {}


def _array_nbytes(array, dtype=None) -> int:
    """Dense size of `array` in bytes, without materializing it."""
    itemsize = np.dtype(dtype if dtype is not None else array.dtype).itemsize
    return int(np.prod(array.shape, dtype=np.int64)) * itemsize


def _compute_block(array, dtype, max_workers=None):
    """
    Materialize one lazy block, optionally bounding scheduler concurrency.

    A byte budget bounds the *result* of a block, not what producing it costs.
    Where a task inflates its input -- CLAHE's equalize_adapthist works in
    float64 and peaks at 30-150x its block -- Dask's threaded scheduler
    running one task per core multiplies that by the core count.  Functions
    that expensive declare `max_dask_workers` and get it passed through here.
    `num_workers` is passed per-compute rather than set on the global
    dask.config, which several files processing concurrently would race on.
    """
    if max_workers and hasattr(array, "compute"):
        block = array.compute(scheduler="threads", num_workers=max_workers)
    else:
        block = np.asarray(array)
    return block.astype(dtype, copy=False)


def iter_planes_blockwise(array, dtype, budget_bytes, max_workers=None):
    """
    Yield `array` as contiguous YX planes in C order, materializing at most
    ~`budget_bytes` at a time.

    Descends the leading axes until a sub-block fits the budget, computes that
    block in one go, then yields its planes.  Computing whole blocks rather
    than individual planes keeps the number of Dask graph executions down --
    slicing one plane at a time re-executes the whole containing block for
    every plane in it, so a 4-block/32-plane array runs the graph 33 times
    instead of 5 -- while still bounding resident memory by the block size.
    """
    if array.ndim <= 2:
        yield np.ascontiguousarray(_compute_block(array, dtype, max_workers))
        return

    if _array_nbytes(array, dtype) <= budget_bytes:
        block = _compute_block(array, dtype, max_workers)
        for plane in block.reshape(-1, *block.shape[-2:]):
            # Copy: the block is released once this loop ends, and tifffile
            # may still hold queued planes for compression at that point.
            # `copy` defaults to C order, so the result is contiguous.
            yield plane.copy()
        return

    for i in range(array.shape[0]):
        yield from iter_planes_blockwise(
            array[i], dtype, budget_bytes, max_workers
        )


def iter_planes_for_write(array, dtype, budget_bytes=STREAM_BLOCK_BYTES):
    """
    Yield `array` as YX planes, traversing it the way its backend wants.

    A Dask input has to go block by block: slicing it one plane at a time
    re-executes the whole containing block for every plane in it, so a stack
    chunked one timepoint per block would relabel that timepoint once per Z
    slice.  A store-backed array (zarr) has no such recompute -- reading a
    plane is just reading its chunks -- so it stays plane by plane, which
    bounds peak by a single plane instead of by the block budget.
    """
    if hasattr(array, "compute"):
        yield from iter_planes_blockwise(array, dtype, budget_bytes)
        return

    lead_shape = tuple(array.shape)[:-2]
    if not lead_shape:
        yield np.asarray(array, dtype=dtype)
        return
    for lead_idx in np.ndindex(*lead_shape):
        yield np.asarray(array[lead_idx], dtype=dtype)


def stream_planes_to_tiff(
    output_path: str,
    planes,
    shape,
    dtype,
    metadata: Optional[dict] = None,
    bigtiff: bool = False,
    ome: Optional[bool] = True,
) -> str:
    """
    Write a TIFF from an iterator of YX planes, never holding it whole.

    `planes` is consumed lazily, so a result far larger than RAM can be
    written as long as the caller yields one plane (or block of planes) at a
    time.  `shape` and `dtype` describe the *full* output, since tifffile
    cannot infer them from a generator.  `ome=None` leaves the choice to
    tifffile, which turns OME on for a ``.ome.tif`` extension.

    Two of the kwargs below are load-bearing and must not be dropped:

    ``photometric="minisblack"`` -- without it tifffile reads an axis of
    length 3 or 4 (4 z-slices, say) as RGB samples and stores the stack as
    separate component planes, which breaks the plane iterator.

    ``maxworkers=1`` -- threaded compression drains the iterator as fast as it
    can and queues the *encoded* segments with no backpressure, so peak scales
    with the whole output rather than one block: measured 545 MB vs 3 MB on a
    48-block write, and unbounded on the multi-GB stacks this path exists for.
    tifffile enables threading by heuristic once a write looks big enough (the
    cliff falls between 32 and 36 blocks), which is exactly when it hurts.
    Costs ~2x write time.
    """
    tifffile.imwrite(
        output_path,
        data=planes,
        shape=tuple(int(s) for s in shape),
        dtype=np.dtype(dtype),
        ome=ome,
        metadata=metadata,
        compression="zlib",
        photometric="minisblack",
        bigtiff=bigtiff,
        maxworkers=1,
    )
    return output_path


def write_labels_with_source_metadata(
    labels: Any,
    source_path: Optional[str],
    output_path: str,
    output_format: str,
    dim_order: str,
) -> str:
    """Write labels while preserving source OME metadata/pyramid when possible."""
    output_format = str(output_format or "tiff").lower()
    labels_dtype = np.uint32

    if output_format == "zarr":
        from ome_zarr.io import parse_url
        from ome_zarr.scale import Scaler
        from ome_zarr.writer import write_image
        import zarr

        # write_image() falls back to da.from_array() for anything that is not
        # already a Dask array, and that auto-chunks to ~128 MiB with no
        # regard for the array's own on-disk layout.  Peak is then fixed at
        # roughly one such chunk per Dask thread (~4 GB on a large stack) and
        # the caller has no way to bound the write, however finely the source
        # is chunked; on an array below the auto target it is a single
        # whole-stack chunk, which is what the 1.61x-vs-0.37x measurement saw.
        # Wrapping with the array's own chunking restores that control, so the
        # write goes through da.store block by block.  numpy input is left
        # alone: it is already resident, and one chunk is right for it.
        if (
            not isinstance(labels, np.ndarray)
            and hasattr(labels, "chunks")
            and not hasattr(labels, "compute")
        ):
            import dask.array as da

            labels = da.from_array(labels, chunks=labels.chunks)

        attrs = _read_root_attrs(source_path) if source_path else {}
        multiscales = _get_multiscales(attrs)
        src_ms = multiscales[0] if multiscales else {}
        src_datasets = src_ms.get("datasets", []) if isinstance(src_ms, dict) else []
        src_n_levels = max(int(len(src_datasets)), 1)

        axes = str(dim_order or "YX").lower()
        labels_ndim = labels.ndim if hasattr(labels, "ndim") else np.asarray(labels).ndim
        if len(axes) != labels_ndim:
            fallback = {2: "yx", 3: "zyx", 4: "tzyx", 5: "tczyx"}
            axes = fallback.get(labels_ndim, "yx")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if os.path.exists(output_path):
            import shutil

            shutil.rmtree(output_path, ignore_errors=True)

        store = parse_url(output_path, mode="w").store
        root = zarr.group(store=store, zarr_format=3)

        write_image(
            image=labels,
            group=root,
            axes=axes,
            coordinate_transformations=None,
            scaler=Scaler(max_layer=src_n_levels - 1, method="nearest", downscale=2),
            compute=True,
            storage_options={
                "chunks": tuple(
                    (1 if i < labels_ndim - 2 else max(1, min(512, int(s))))
                    for i, s in enumerate(labels.shape if hasattr(labels, "shape") else np.asarray(labels).shape)
                )
            },
        )

        # Copy the omero block (channel/rendering info) BEFORE the raw
        # JSON rewrite below.  root.attrs is a stale in-memory cache of
        # the attrs written by write_image() above, so setting a key on
        # it re-serialises that whole (now-outdated) document to disk --
        # done after the coordinate-transformations rewrite, it silently
        # clobbers the fix that block just persisted. Done first, the
        # rewrite below reads this omero key back off disk along with
        # everything else and keeps it.
        if "omero" in attrs:
            try:
                root.attrs["omero"] = attrs["omero"]
            except Exception:
                pass

        # Align output per-level coordinate transforms to source metadata.
        # Source and output axes can differ (e.g. a channel axis dropped
        # during processing), so transforms are rebuilt per-axis by name
        # rather than copied wholesale — copying a 5-value TCZYX scale onto
        # a 4-axis TZYX output would silently misassign Z's scale to the
        # dropped channel's slot.
        out_zattrs_path = os.path.join(output_path, ".zattrs")
        out_zarr_json_path = os.path.join(output_path, "zarr.json")
        if (os.path.exists(out_zattrs_path) or os.path.exists(out_zarr_json_path)) and src_datasets:
            try:
                out_attrs = {}
                target_path = None
                write_back_as_zarr_json = False
                if os.path.exists(out_zattrs_path):
                    with open(out_zattrs_path, encoding="utf-8") as f:
                        out_attrs = json.load(f)
                    target_path = out_zattrs_path
                elif os.path.exists(out_zarr_json_path):
                    with open(out_zarr_json_path, encoding="utf-8") as f:
                        zarr_doc = json.load(f)
                    if isinstance(zarr_doc, dict):
                        out_attrs = zarr_doc.get("attributes", {})
                    target_path = out_zarr_json_path
                    write_back_as_zarr_json = True

                # _get_multiscales handles both the flat ``multiscales`` key
                # and the NGFF-v0.5-style ``ome.multiscales`` nesting that
                # zarr_format=3 output actually uses; a plain
                # out_attrs.get("multiscales") misses the latter and makes
                # this whole block a silent no-op.
                out_ms_list = _get_multiscales(out_attrs)
                if out_ms_list:
                    out_ms = out_ms_list[0]
                    out_ds = out_ms.get("datasets", [])

                    src_axis_names = [
                        str(
                            a.get("name") if isinstance(a, dict) else a
                        ).lower()
                        for a in (
                            src_ms.get("axes", [])
                            if isinstance(src_ms, dict)
                            else []
                        )
                    ]

                    def _aligned_scale(src_ctf_list):
                        for ctf in src_ctf_list:
                            if (
                                not isinstance(ctf, dict)
                                or ctf.get("type") != "scale"
                            ):
                                continue
                            src_scale = ctf.get("scale")
                            if not (
                                isinstance(src_scale, list)
                                and len(src_scale) == len(src_axis_names)
                            ):
                                continue
                            return [
                                src_scale[src_axis_names.index(ax_char)]
                                if ax_char in src_axis_names
                                else 1.0
                                for ax_char in axes
                            ]
                        return None

                    for i, ds in enumerate(out_ds):
                        if i >= len(src_datasets):
                            break
                        src_ctf = src_datasets[i].get("coordinateTransformations")
                        if not (isinstance(src_ctf, list) and src_ctf):
                            continue
                        aligned_scale = _aligned_scale(src_ctf)
                        if aligned_scale is not None:
                            ds["coordinateTransformations"] = [
                                {"type": "scale", "scale": aligned_scale}
                            ]

                    if write_back_as_zarr_json:
                        with open(target_path, encoding="utf-8") as f:
                            zarr_doc = json.load(f)
                        zarr_doc["attributes"] = out_attrs
                        with open(target_path, "w", encoding="utf-8") as f:
                            json.dump(zarr_doc, f, indent=2)
                    else:
                        with open(target_path, "w", encoding="utf-8") as f:
                            json.dump(out_attrs, f, indent=2)
            except Exception:
                pass

        return output_path

    # OME-TIFF path
    labels_shape = tuple(
        int(s) for s in (labels.shape if hasattr(labels, "shape") else np.asarray(labels).shape)
    )
    labels_ndim = len(labels_shape)
    size_bytes = int(np.prod(labels_shape, dtype=np.int64)) * np.dtype(labels_dtype).itemsize
    use_bigtiff = (size_bytes / (1024**3)) > 2.0
    axes = str(dim_order or "YX").upper()
    if len(axes) != labels_ndim:
        fallback = {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}
        axes = fallback.get(labels_ndim, "YX")

    ome_metadata = {"axes": axes}
    for ax_name, ax_scale in _extract_source_physical_scale(
        source_path, axes
    ).items():
        ome_metadata[f"PhysicalSize{ax_name}"] = ax_scale
        ome_metadata[f"PhysicalSize{ax_name}Unit"] = "um"

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    tmp_output_path = os.path.join(
        output_dir,
        f".{os.path.basename(output_path)}.tmp-{os.getpid()}-{time.time_ns()}",
    )

    try:
        if isinstance(labels, np.ndarray):
            tifffile.imwrite(
                tmp_output_path,
                np.asarray(labels, dtype=labels_dtype),
                dtype=labels_dtype,
                ome=True,
                metadata=ome_metadata,
                compression="zlib",
                photometric="minisblack",
                bigtiff=use_bigtiff,
            )
        else:
            # For array-like backends (zarr.Array, Dask) stream YX planes so
            # the full volume is never materialized in RAM.
            stream_planes_to_tiff(
                tmp_output_path,
                iter_planes_for_write(labels, labels_dtype),
                labels_shape,
                labels_dtype,
                metadata=ome_metadata,
                bigtiff=use_bigtiff,
            )

        os.replace(tmp_output_path, output_path)
    except Exception:
        with suppress(OSError, FileNotFoundError):
            os.unlink(tmp_output_path)
        raise

    return output_path
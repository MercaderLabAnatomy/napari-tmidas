import json
import os
import time
from contextlib import suppress
from typing import Any, Optional

import numpy as np
import tifffile


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

        if "omero" in attrs:
            try:
                root.attrs["omero"] = attrs["omero"]
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
            # For array-like backends (e.g. zarr.Array) stream YX planes via a
            # generator so the full volume is never materialized in RAM.
            def _iter_planes():
                lead_shape = labels_shape[:-2]
                if not lead_shape:
                    yield np.asarray(labels, dtype=labels_dtype)
                    return
                for lead_idx in np.ndindex(*lead_shape):
                    yield np.asarray(labels[lead_idx], dtype=labels_dtype)

            tifffile.imwrite(
                tmp_output_path,
                data=_iter_planes(),
                shape=labels_shape,
                dtype=labels_dtype,
                ome=True,
                metadata=ome_metadata,
                compression="zlib",
                photometric="minisblack",
                bigtiff=use_bigtiff,
            )

        os.replace(tmp_output_path, output_path)
    except Exception:
        with suppress(OSError, FileNotFoundError):
            os.unlink(tmp_output_path)
        raise

    return output_path
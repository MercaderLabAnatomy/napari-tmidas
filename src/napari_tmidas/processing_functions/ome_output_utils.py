import json
import os
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
        if len(axes) != getattr(labels, "ndim", np.asarray(labels).ndim):
            fallback = {2: "yx", 3: "zyx", 4: "tzyx", 5: "tczyx"}
            axes = fallback.get(getattr(labels, "ndim", np.asarray(labels).ndim), "yx")

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
                    (1 if i < getattr(labels, "ndim", np.asarray(labels).ndim) - 2 else max(1, min(512, int(s))))
                    for i, s in enumerate(getattr(labels, "shape", np.asarray(labels).shape))
                )
            },
        )

        # Align output per-level coordinate transforms to source metadata.
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

                out_ms_list = out_attrs.get("multiscales", [])
                if isinstance(out_ms_list, list) and out_ms_list:
                    out_ms = out_ms_list[0]
                    out_ds = out_ms.get("datasets", [])

                    for i, ds in enumerate(out_ds):
                        if i >= len(src_datasets):
                            break
                        src_ctf = src_datasets[i].get("coordinateTransformations")
                        if isinstance(src_ctf, list) and src_ctf:
                            ds["coordinateTransformations"] = src_ctf

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
    arr = np.asarray(labels, dtype=labels_dtype)
    shape = tuple(int(s) for s in arr.shape)
    size_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(labels_dtype).itemsize
    use_bigtiff = (size_bytes / (1024**3)) > 2.0
    axes = str(dim_order or "YX").upper()
    if len(axes) != arr.ndim:
        fallback = {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}
        axes = fallback.get(arr.ndim, "YX")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tifffile.imwrite(
        output_path,
        arr,
        dtype=labels_dtype,
        ome=True,
        metadata={"axes": axes},
        compression="zlib",
        bigtiff=use_bigtiff,
    )
    return output_path
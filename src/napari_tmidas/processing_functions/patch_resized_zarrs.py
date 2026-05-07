"""
Patch existing resized zarrs in-place:
- Fix coordinate transforms (copy physical scale from source, adjust for new YX size)
- Add omero window metadata with contrast limits computed from level-0 data
- No pixel data is touched; only .zattrs files are rewritten.

Usage:
    python patch_resized_zarrs.py <resized_dir> [source_dir]

If source_dir is omitted it is inferred by removing the last path component
from resized_dir (i.e. the parent of the resized folder).
"""

import sys
import os
import json
import glob
import numpy as np
import zarr

# ──────────────────────────────────────────────────────────────────────────────
def _axes_name_indices(axes):
    """Return (t_idx, c_idx, z_idx, y_idx, x_idx) from OME axes list."""
    mapping = {}
    for i, ax in enumerate(axes):
        name = (ax.get("name", "") if isinstance(ax, dict) else str(ax)).lower()
        atype = (ax.get("type", "") if isinstance(ax, dict) else "").lower()
        if name == "t" or atype == "time":
            mapping["t"] = i
        elif name in ("c", "channel", "ch") or atype == "channel":
            mapping["c"] = i
        elif name == "z":
            mapping["z"] = i
        elif name == "y":
            mapping["y"] = i
        elif name == "x":
            mapping["x"] = i
    return mapping


def patch_zarr(resized_path, source_path):
    print(f"\n=== Patching: {os.path.basename(resized_path)} ===")

    # ── load metadata ─────────────────────────────────────────────────────────
    dst_attrs_path = os.path.join(resized_path, ".zattrs")
    src_attrs_path = os.path.join(source_path, ".zattrs")

    if not os.path.exists(dst_attrs_path):
        print(f"  SKIP: no .zattrs found in {resized_path}")
        return
    if not os.path.exists(src_attrs_path):
        print(f"  SKIP: source zarr not found at {source_path}")
        return

    dst_attrs = json.load(open(dst_attrs_path))
    src_attrs = json.load(open(src_attrs_path))

    src_ms  = src_attrs.get("multiscales", [{}])[0]
    dst_ms  = dst_attrs.get("multiscales", [{}])[0]
    src_axes= src_ms.get("axes", [])
    dst_datasets = dst_ms.get("datasets", [])

    if not dst_datasets:
        print(f"  SKIP: no datasets in resized .zattrs")
        return

    # ── Fix coordinate transforms ──────────────────────────────────────────────
    src_ctf0 = (src_ms.get("datasets") or [{}])[0].get("coordinateTransformations", [{}])
    src_scale_l0 = None
    if src_ctf0 and src_ctf0[0].get("type") == "scale":
        src_scale_l0 = src_ctf0[0]["scale"]

    if src_scale_l0:
        axis_map = _axes_name_indices(src_axes)
        y_idx = axis_map.get("y", len(src_scale_l0) - 2)
        x_idx = axis_map.get("x", len(src_scale_l0) - 1)

        # Read level-0 shapes
        src_arr0 = zarr.open_array(os.path.join(source_path, "0"), mode="r")
        dst_arr0 = zarr.open_array(os.path.join(resized_path, "0"), mode="r")

        src_y = src_arr0.shape[y_idx]
        src_x = src_arr0.shape[x_idx]
        dst_y = dst_arr0.shape[y_idx]
        dst_x = dst_arr0.shape[x_idx]

        new_y_scale = src_scale_l0[y_idx] * (src_y / dst_y)
        new_x_scale = src_scale_l0[x_idx] * (src_x / dst_x)

        lvl0_scale = list(src_scale_l0)
        lvl0_scale[y_idx] = new_y_scale
        lvl0_scale[x_idx] = new_x_scale

        # Check if C was dropped (resized has fewer dims than source)
        if dst_arr0.ndim < src_arr0.ndim:
            c_idx = axis_map.get("c")
            if c_idx is not None:
                lvl0_scale = [v for i, v in enumerate(lvl0_scale) if i != c_idx]
                y_idx = y_idx - (1 if c_idx < y_idx else 0)
                x_idx = x_idx - (1 if c_idx < x_idx else 0)

        for n, ds in enumerate(dst_datasets):
            level_scale = list(lvl0_scale)
            level_scale[y_idx] = lvl0_scale[y_idx] * (2 ** n)
            level_scale[x_idx] = lvl0_scale[x_idx] * (2 ** n)
            ds["coordinateTransformations"] = [{"type": "scale", "scale": level_scale}]

        print(f"  Coordinate transforms updated: level-0 Y/X = {new_y_scale:.4f}")
    else:
        print(f"  WARNING: no source scale found, skipping coord transform fix")

    # ── Build omero window metadata ────────────────────────────────────────────
    try:
        lv0_arr = zarr.open_array(os.path.join(resized_path, "0"), mode="r")
        ndim = lv0_arr.ndim
        dst_axes = dst_ms.get("axes", src_axes)
        ax_map_dst = _axes_name_indices(dst_axes)

        T_out = lv0_arr.shape[0]
        z_dim = ax_map_dst.get("z", ndim - 3)
        Z_out = lv0_arr.shape[z_dim] if ndim >= 4 else 1
        c_dim = ax_map_dst.get("c")

        t_idxs = sorted(set(int(i) for i in np.linspace(0, T_out - 1, min(5, T_out))))
        z_idxs = sorted(set(int(i) for i in np.linspace(0, Z_out - 1, min(15, Z_out))))

        n_out_channels = lv0_arr.shape[c_dim] if (c_dim is not None and ndim == 5) else 1

        src_omero   = src_attrs.get("omero", {})
        src_ch_list = src_omero.get("channels", [])

        _default_colors = ["FFFFFF", "00FF00", "FF00FF", "00FFFF", "FF0000", "0000FF", "FFFF00"]

        omero_channels = []
        for oc in range(n_out_channels):
            samples = []
            for t in t_idxs:
                for z in z_idxs:
                    if c_dim is not None and ndim == 5:
                        sl = tuple(
                            t if i == 0
                            else oc if i == c_dim
                            else z if i == z_dim
                            else slice(None)
                            for i in range(ndim)
                        )
                    elif ndim == 5:
                        sl = (t, slice(None), z, slice(None), slice(None))
                    else:
                        sl = (t, z, slice(None), slice(None))
                    samples.append(lv0_arr[sl].ravel())
            flat = np.concatenate(samples)
            lo  = int(np.percentile(flat, 0.5))
            mn  = int(flat.min())
            mx  = int(flat.max())
            hi  = mx
            p99 = int(np.percentile(flat, 99.0))
            if hi > 10 * p99 and p99 > 0:
                hi = 10 * p99

            base_ch = src_ch_list[oc] if oc < len(src_ch_list) else {}
            ch_entry = dict(base_ch)
            ch_entry["window"] = {"min": mn, "max": mx, "start": lo, "end": hi}
            ch_entry.setdefault("label", f"Channel {oc}")
            ch_entry["active"] = True
            ch_entry.setdefault("color", _default_colors[oc % len(_default_colors)])
            omero_channels.append(ch_entry)

        omero_out = dict(src_omero)
        omero_out["channels"] = omero_channels
        omero_out.setdefault("version", "0.3")
        dst_attrs["omero"] = omero_out
        print(f"  Omero windows: {[(c['window']['start'], c['window']['end']) for c in omero_channels]}")

    except Exception as e:
        print(f"  WARNING: omero metadata failed: {e}")
        import traceback; traceback.print_exc()

    # ── Write updated .zattrs ─────────────────────────────────────────────────
    json.dump(dst_attrs, open(dst_attrs_path, "w"), indent=2)
    print(f"  .zattrs written ✓")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    resized_dir = sys.argv[1].rstrip("/")
    # Auto-discover source dir: sibling of resized_dir, or explicit arg
    if len(sys.argv) >= 3:
        source_dir = sys.argv[2].rstrip("/")
        # source_dir is a directory containing source zarrs
        source_root = source_dir
    else:
        source_root = os.path.dirname(resized_dir)

    resized_zarrs = sorted(glob.glob(os.path.join(resized_dir, "*.zarr")))
    if not resized_zarrs:
        print(f"No .zarr found in {resized_dir}")
        sys.exit(1)

    print(f"Found {len(resized_zarrs)} resized zarrs to patch")
    print(f"Source root: {source_root}")

    for rzarr in resized_zarrs:
        basename = os.path.basename(rzarr)
        # Strip suffix variations like _yx_resized, _resized, _resize
        stem = basename
        for suffix in ["_yx_resized.zarr", "_resized.zarr", "_resize.zarr"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)] + ".zarr"
                break
        src = os.path.join(source_root, stem)
        if not os.path.isdir(src):
            # Try without .zarr extension
            stem2 = stem[:-5] if stem.endswith(".zarr") else stem
            for candidate in [stem2 + ".zarr", stem2]:
                c = os.path.join(source_root, candidate)
                if os.path.isdir(c):
                    src = c
                    break
        patch_zarr(rzarr, src)

    print("\nDone.")


if __name__ == "__main__":
    main()

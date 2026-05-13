import json

import numpy as np
import pytest
import tifffile
import zarr

from napari_tmidas.processing_functions.ome_output_utils import (
    write_labels_with_source_metadata,
)


def _write_source_zattrs(source_path, n_levels=3):
    datasets = []
    for level in range(n_levels):
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, float(2**level), float(2**level)],
                    }
                ],
            }
        )

    attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": datasets,
            }
        ],
        "omero": {
            "version": "0.3",
            "channels": [
                {
                    "label": "labels",
                    "color": "FFFFFF",
                    "window": {
                        "start": 0,
                        "end": 10,
                        "min": 0,
                        "max": 10,
                    },
                }
            ],
        },
    }

    with open(source_path / ".zattrs", "w", encoding="utf-8") as f:
        json.dump(attrs, f)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_write_labels_with_source_metadata_preserves_pyramid_and_omero(tmp_path):
    pytest.importorskip("ome_zarr")

    source_path = tmp_path / "source.zarr"
    source_path.mkdir()
    _write_source_zattrs(source_path, n_levels=3)

    labels = np.zeros((8, 64, 64), dtype=np.uint32)
    labels[:, 10:20, 10:20] = 3

    output_path = tmp_path / "out_labels.zarr"
    returned = write_labels_with_source_metadata(
        labels=labels,
        source_path=str(source_path),
        output_path=str(output_path),
        output_format="zarr",
        dim_order="ZYX",
    )

    assert returned == str(output_path)
    assert output_path.exists()

    out_zattrs = output_path / ".zattrs"
    out_zarr_json = output_path / "zarr.json"
    if out_zattrs.exists():
        with open(out_zattrs, encoding="utf-8") as f:
            out_attrs = json.load(f)
    else:
        with open(out_zarr_json, encoding="utf-8") as f:
            out_doc = json.load(f)
        out_attrs = out_doc.get("attributes", {})

    out_multiscales = out_attrs.get("multiscales", [])
    if not out_multiscales and isinstance(out_attrs.get("ome"), dict):
        out_multiscales = out_attrs["ome"].get("multiscales", [])

    assert out_multiscales
    out_datasets = out_multiscales[0].get("datasets", [])
    assert len(out_datasets) == 3
    assert out_attrs.get("omero", {}).get("version") == "0.3"


def test_write_labels_with_source_metadata_writes_ome_tiff(tmp_path):
    labels = np.zeros((32, 32), dtype=np.uint32)
    labels[5:10, 5:10] = 7

    output_path = tmp_path / "labels.ome.tif"
    returned = write_labels_with_source_metadata(
        labels=labels,
        source_path=None,
        output_path=str(output_path),
        output_format="tiff",
        dim_order="YX",
    )

    assert returned == str(output_path)
    assert output_path.exists()

    with tifffile.TiffFile(output_path) as tif:
        assert tif.is_ome
        arr = tif.asarray()
        assert arr.dtype == np.uint32
        assert arr.shape == labels.shape


def test_write_labels_with_source_metadata_tiff_failure_is_atomic(
    tmp_path, monkeypatch
):
    labels = np.ones((4, 8, 8), dtype=np.uint32)
    output_path = tmp_path / "labels.ome.tif"

    def _failing_imwrite(path, *args, **kwargs):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(tifffile, "imwrite", _failing_imwrite)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        write_labels_with_source_metadata(
            labels=labels,
            source_path=None,
            output_path=str(output_path),
            output_format="tiff",
            dim_order="ZYX",
        )

    assert not output_path.exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_write_labels_with_source_metadata_streams_zarr_array_to_ome_tiff(
    tmp_path,
):
    labels_path = tmp_path / "labels_cache.zarr"
    labels = zarr.open_array(
        str(labels_path),
        mode="w",
        shape=(2, 3, 16, 16),
        chunks=(1, 1, 16, 16),
        dtype=np.uint32,
    )
    labels[:] = np.arange(2 * 3 * 16 * 16, dtype=np.uint32).reshape(
        2, 3, 16, 16
    )

    output_path = tmp_path / "labels_streamed.ome.tif"
    returned = write_labels_with_source_metadata(
        labels=labels,
        source_path=None,
        output_path=str(output_path),
        output_format="tiff",
        dim_order="TZYX",
    )

    assert returned == str(output_path)
    assert output_path.exists()

    with tifffile.TiffFile(output_path) as tif:
        assert tif.is_ome
        arr = tif.asarray()
        assert arr.dtype == np.uint32
        assert arr.shape == (2, 3, 16, 16)

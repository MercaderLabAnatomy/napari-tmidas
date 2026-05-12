import importlib
import json
import sys

import numpy as np
import pytest


cellpose_mod = importlib.import_module(
    "napari_tmidas.processing_functions.cellpose_segmentation"
)


def test_tyx_timepoint_interval_non_zarr(monkeypatch):
    """TYX inputs should honor start/end/step in non-zarr mode."""
    calls = []

    def fake_run_cellpose_in_env(command, args):
        assert command == "eval"
        calls.append(dict(args))
        return np.zeros(args["image"].shape, dtype=np.uint32)

    monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: True)
    monkeypatch.setattr(cellpose_mod, "create_cellpose_env", lambda: None)
    monkeypatch.setattr(
        cellpose_mod, "run_cellpose_in_env", fake_run_cellpose_in_env
    )

    image = np.random.rand(5, 16, 16).astype(np.float32)
    result = cellpose_mod.cellpose_segmentation(
        image,
        dim_order="TYX",
        timepoint_start=1,
        timepoint_end=4,
        timepoint_step=2,
    )

    assert result.shape == (2, 16, 16)
    assert len(calls) == 2
    assert [c["image"].shape for c in calls] == [(16, 16), (16, 16)]


def test_tyx_timepoint_interval_zarr_direct(tmp_path, monkeypatch):
    """TYX zarr-direct path should honor start/end/step via timepoint_index calls."""
    calls = []

    def fake_run_cellpose_in_env(command, args):
        assert command == "eval"
        calls.append(dict(args))
        return np.zeros((16, 16), dtype=np.uint32)

    monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: True)
    monkeypatch.setattr(cellpose_mod, "create_cellpose_env", lambda: None)
    monkeypatch.setattr(
        cellpose_mod, "run_cellpose_in_env", fake_run_cellpose_in_env
    )

    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zattrs").write_text("{}", encoding="utf-8")

    image = np.random.rand(5, 16, 16).astype(np.float32)
    result = cellpose_mod.cellpose_segmentation(
        image,
        dim_order="TYX",
        timepoint_start=0,
        timepoint_end=4,
        timepoint_step=2,
        _source_filepath=str(zarr_path),
    )

    assert result.shape == (3, 16, 16)
    assert [c.get("timepoint_index") for c in calls] == [0, 2, 4]
    assert all(c.get("zarr_path") == str(zarr_path) for c in calls)


def test_distributed_kept_for_non_z_time_series(tmp_path, monkeypatch):
    """Distributed mode should remain enabled for non-Z large 2D workflows."""
    calls = []

    def fake_run_cellpose_in_env(command, args):
        assert command == "eval"
        calls.append(dict(args))
        return np.zeros((16, 16), dtype=np.uint32)

    monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: True)
    monkeypatch.setattr(cellpose_mod, "create_cellpose_env", lambda: None)
    monkeypatch.setattr(
        cellpose_mod, "run_cellpose_in_env", fake_run_cellpose_in_env
    )

    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zattrs").write_text("{}", encoding="utf-8")

    image = np.random.rand(3, 16, 16).astype(np.float32)
    result = cellpose_mod.cellpose_segmentation(
        image,
        dim_order="TYX",
        use_distributed_segmentation=True,
        timepoint_start=0,
        timepoint_end=2,
        timepoint_step=1,
        _source_filepath=str(zarr_path),
    )

    assert result.shape == (3, 16, 16)
    assert len(calls) == 3
    assert all(
        c.get("use_distributed_segmentation") is True for c in calls
    )


def test_distributed_auto_conversion_requests_zarr_v3(tmp_path, monkeypatch):
    """Non-zarr distributed auto-conversion should request zarr v3 output."""
    conversion_calls = []

    fake_module = type(sys)("napari_tmidas._file_selector")

    def fake_save_as_zarr(**kwargs):
        conversion_calls.append(dict(kwargs))

    fake_module.save_as_zarr = fake_save_as_zarr
    monkeypatch.setitem(sys.modules, "napari_tmidas._file_selector", fake_module)

    monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: True)
    monkeypatch.setattr(cellpose_mod, "create_cellpose_env", lambda: None)
    monkeypatch.setattr(
        cellpose_mod,
        "run_cellpose_in_env",
        lambda _command, _args: np.zeros((8, 8, 8), dtype=np.uint32),
    )

    source_file = tmp_path / "input.tif"
    source_file.write_bytes(b"placeholder")

    image = np.random.rand(8, 8, 8).astype(np.float32)
    _ = cellpose_mod.cellpose_segmentation(
        image,
        dim_order="ZYX",
        use_distributed_segmentation=True,
        _source_filepath=str(source_file),
    )

    assert len(conversion_calls) == 1
    assert conversion_calls[0].get("zarr_format") == 3


def test_direct_zarr_output_preserves_source_multiscales_lightweight(
    tmp_path, monkeypatch
):
    """Lightweight integration test for direct Cellpose->OME-Zarr output writing."""
    pytest.importorskip("ome_zarr")

    monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: True)
    monkeypatch.setattr(cellpose_mod, "create_cellpose_env", lambda: None)

    def fake_run_cellpose_in_env(command, args):
        assert command == "eval"
        assert args.get("zarr_path")
        return np.ones((4, 16, 16), dtype=np.uint32)

    monkeypatch.setattr(cellpose_mod, "run_cellpose_in_env", fake_run_cellpose_in_env)

    source_path = tmp_path / "source.zarr"
    source_path.mkdir()
    source_attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                        ],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 2.0, 2.0]}
                        ],
                    },
                ],
            }
        ],
        "omero": {"version": "0.3", "channels": [{"label": "src"}]},
    }
    (source_path / ".zattrs").write_text(json.dumps(source_attrs), encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image = np.random.rand(4, 16, 16).astype(np.float32)

    out = cellpose_mod.cellpose_segmentation(
        image,
        dim_order="ZYX",
        _source_filepath=str(source_path),
        _output_folder=str(output_dir),
        _output_suffix="_labels",
        _output_format="zarr",
    )

    assert isinstance(out, str)
    assert out.endswith(".zarr")

    out_path = tmp_path / "out" / "source_labels.zarr"
    assert out_path.exists()

    out_zattrs = out_path / ".zattrs"
    out_zarr_json = out_path / "zarr.json"
    if out_zattrs.exists():
        out_attrs = json.loads(out_zattrs.read_text(encoding="utf-8"))
    else:
        out_doc = json.loads(out_zarr_json.read_text(encoding="utf-8"))
        out_attrs = out_doc.get("attributes", {})

    out_multiscales = out_attrs.get("multiscales", [])
    if not out_multiscales and isinstance(out_attrs.get("ome"), dict):
        out_multiscales = out_attrs["ome"].get("multiscales", [])
    assert out_multiscales
    assert len(out_multiscales[0].get("datasets", [])) == 2

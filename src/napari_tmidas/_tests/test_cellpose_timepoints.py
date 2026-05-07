import importlib

import numpy as np


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

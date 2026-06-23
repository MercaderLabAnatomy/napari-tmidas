import numpy as np

from napari_tmidas.processing_functions import trackastra_tracking as trackastra_mod
from napari_tmidas.processing_functions.trackastra_tracking import (
    TrackAstraEnvManager,
    _find_matching_raw_path,
    _load_zarr_array,
    _strip_known_image_suffix,
)


def test_strip_known_image_suffix_supports_zarr_and_tiff():
    assert _strip_known_image_suffix("sample.zarr") == "sample"
    assert _strip_known_image_suffix("sample.tiff") == "sample"
    assert _strip_known_image_suffix("sample.tif") == "sample"
    assert _strip_known_image_suffix("sample") == "sample"


def test_find_matching_raw_path_detects_zarr_for_tif_label(tmp_path):
    raw_zarr = tmp_path / "movie.zarr"
    raw_zarr.mkdir()
    label_tif = tmp_path / "movie_convpaint_labels_rm_small.tif"
    label_tif.write_text("", encoding="utf-8")

    raw_base, candidates, raw_path = _find_matching_raw_path(
        str(label_tif), "_convpaint_labels_rm_small.tif"
    )

    assert raw_base == "movie"
    assert "movie.zarr" in candidates
    assert raw_path == str(raw_zarr)


def test_find_matching_raw_path_preserves_extension_if_in_base(tmp_path):
    raw_zarr = tmp_path / "movie.zarr"
    raw_zarr.mkdir()
    label_tif = tmp_path / "movie.zarr_labels.tif"
    label_tif.write_text("", encoding="utf-8")

    _, candidates, raw_path = _find_matching_raw_path(str(label_tif), "_labels.tif")

    assert candidates == ["movie.zarr"]
    assert raw_path == str(raw_zarr)


def test_trackastra_tracking_uses_zarr_raw_for_tif_label(tmp_path, monkeypatch):
    raw_zarr = tmp_path / "movie.zarr"
    raw_zarr.mkdir()
    label_tif = tmp_path / "movie_convpaint_labels_rm_small.tif"
    label_tif.write_text("", encoding="utf-8")

    captured = {}

    def fake_create_trackastra_script(
        img_path,
        mask_path,
        model,
        mode,
        output_path,
        channel,
        dimension_order,
        batch_size="Auto",
    ):
        captured["img_path"] = img_path
        captured["mask_path"] = mask_path
        captured["output_path"] = output_path
        captured["dimension_order"] = dimension_order
        return "print('mock trackastra script')\n"

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_subprocess_run(cmd, *args, **kwargs):
        # Simulate a successful TrackAstra run and create expected output.
        out = tmp_path / "movie_tracked.tif"
        out.write_bytes(b"TIFF")
        return _Completed()

    monkeypatch.setattr(trackastra_mod.TrackAstraEnvManager, "ensure_env_ready", lambda: True)
    monkeypatch.setattr(trackastra_mod.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba")
    monkeypatch.setattr(trackastra_mod, "create_trackastra_script", fake_create_trackastra_script)
    monkeypatch.setattr(trackastra_mod.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        trackastra_mod,
        "imread",
        lambda path: np.zeros((2, 8, 8), dtype=np.uint16),
    )

    image = np.zeros((2, 8, 8), dtype=np.uint32)
    result = trackastra_mod.trackastra_tracking(
        image,
        label_pattern="_convpaint_labels_rm_small.tif",
        _source_filepath=str(label_tif),
        _output_folder=str(tmp_path),
        _output_suffix="_tracked",
    )

    assert captured["img_path"] == str(raw_zarr)
    assert captured["mask_path"] == str(label_tif)
    assert captured["output_path"] == str(tmp_path / "movie_tracked.tif")
    assert captured["dimension_order"] == "Auto"
    assert result == str(tmp_path / "movie_tracked.tif")


def test_load_zarr_array_falls_back_to_level0(tmp_path, monkeypatch):
    calls = []

    class _FakeArray:
        def __array__(self, dtype=None):
            return np.zeros((2, 3), dtype=np.uint16)

    def fake_zarr_open(path, mode="r"):
        calls.append(path)
        if path.endswith("/0"):
            return _FakeArray()
        raise RuntimeError("root open failed")

    monkeypatch.setattr(trackastra_mod.os.path, "exists", lambda p: str(p).endswith("/0"))
    import zarr

    monkeypatch.setattr(zarr, "open", fake_zarr_open)

    out = _load_zarr_array(str(tmp_path / "sample.zarr"))
    assert out.shape == (2, 3)
    assert calls[0].endswith("sample.zarr")
    assert calls[1].endswith("sample.zarr/0")


def test_create_trackastra_script_uses_mask_aware_channel_axis_selection():
    script = trackastra_mod.create_trackastra_script(
        "raw.zarr",
        "mask.tif",
        "ctc",
        "greedy",
        "out.tif",
        channel="0",
        dimension_order="TCZYX",
    )

    assert "candidate_axes" in script
    assert "dimension_order_param" in script
    assert "requested_axis" in script
    # Channel selection now uses a dask-safe slice helper instead of np.take.
    assert "_take_axis(img, ch_idx, channel_axis)" in script


def test_create_trackastra_script_uses_lazy_dask_loaders():
    """Generated script must stream input via dask, not full imread/np.asarray."""
    script = trackastra_mod.create_trackastra_script(
        "raw.tif", "mask_labels.tif", "ctc", "greedy", "out_tracked.tif",
        channel="", dimension_order="Auto",
    )
    # Valid Python.
    compile(script, "gen.py", "exec")
    # Lazy, per-page dask loading rather than eager imread of the whole volume.
    assert "import dask.array as da" in script
    assert "_lazy_load" in script
    assert "da.from_delayed" in script
    assert "da.from_zarr" in script
    # The whole-volume eager load must be gone.
    assert "mask = imread(" not in script
    assert "img = imread(" not in script


def test_create_trackastra_script_streams_output_relabel():
    """Output must be written frame-by-frame, bypassing graph_to_ctc's full alloc."""
    script = trackastra_mod.create_trackastra_script(
        "raw.tif", "mask_labels.tif", "ctc", "greedy", "out_tracked.tif",
        channel="", dimension_order="Auto",
    )
    # Replicates graph_to_ctc label assignment for identical IDs...
    assert "ctc_tracklets" in script
    assert "frame_relabel" in script
    # ...and streams pages into the output TIFF with a generator + shape/dtype.
    assert "data=_iter_relabeled_pages()" in script
    assert "shape=out_shape" in script
    assert "bigtiff=use_bigtiff" in script
    # Per-page yielding (flatten leading axes) so 4D TZYX writes correctly.
    assert "reshape((-1,) + relabeled.shape[-2:])" in script
    # Keeps a graph_to_ctc fallback for older/odd trackastra versions.
    assert "graph_to_ctc" in script


def test_create_trackastra_script_has_cuda_oom_recovery():
    """Script must shrink batch size on CUDA OOM and fall back to CPU."""
    script = trackastra_mod.create_trackastra_script(
        "raw.tif", "mask_labels.tif", "ctc", "ilp", "out_tracked.tif",
        channel="", dimension_order="Auto", batch_size="Auto",
    )
    compile(script, "gen.py", "exec")
    assert "_is_cuda_oom" in script
    assert "torch.cuda.empty_cache()" in script
    assert "device='cpu'" in script
    assert "expandable_segments" in script
    # ilp/greedy retry must not fire on OOM (prediction is mode-independent).
    assert "not _is_cuda_oom(exc)" in script


def test_create_trackastra_script_resolves_batch_size():
    """batch_size param maps to the injected `requested_batch` literal."""
    cases = {"Auto": "requested_batch = None", "": "requested_batch = None",
             "bogus": "requested_batch = None", "1": "requested_batch = 1",
             "4": "requested_batch = 4", "0": "requested_batch = None"}
    for value, expected in cases.items():
        script = trackastra_mod.create_trackastra_script(
            "raw.tif", "mask_labels.tif", "ctc", "greedy", "out.tif",
            channel="", dimension_order="Auto", batch_size=value,
        )
        assert expected in script, (value, expected)


def test_detect_gpu_ids_honours_override(monkeypatch):
    """TRACKASTRA_GPUS controls multi-GPU distribution; 'none' disables pinning."""
    monkeypatch.setenv("TRACKASTRA_GPUS", "0,1")
    assert trackastra_mod._detect_gpu_ids() == ["0", "1"]

    monkeypatch.setenv("TRACKASTRA_GPUS", "none")
    assert trackastra_mod._detect_gpu_ids() == []

    # Falls back to CUDA_VISIBLE_DEVICES when no explicit override is given.
    monkeypatch.delenv("TRACKASTRA_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert trackastra_mod._detect_gpu_ids() == ["3"]


def test_gpu_pool_size_matches_detected_ids(monkeypatch):
    """The shared pool holds one slot per detected GPU."""
    monkeypatch.setenv("TRACKASTRA_GPUS", "0,1")
    # Reset the lazily-built singleton so the override takes effect.
    monkeypatch.setattr(trackastra_mod, "_GPU_POOL", None)
    monkeypatch.setattr(trackastra_mod, "_GPU_IDS", None)

    pool, ids = trackastra_mod._get_gpu_pool()
    assert ids == ["0", "1"]
    assert pool.qsize() == 2


def test_trackastra_marked_as_path_loading():
    """Workers must know Trackastra reads from disk so they skip the eager load."""
    # skip_load is honoured by the napari widget worker (image=None, no alloc).
    assert getattr(trackastra_mod.trackastra_tracking, "skip_load", False) is True
    # _loads_from_path is the equivalent hint for the secondary worker.
    assert getattr(trackastra_mod.trackastra_tracking, "_loads_from_path", False) is True


def test_trackastra_normalizes_invalid_mode(tmp_path, monkeypatch):
    """Invalid modes are corrected before the (expensive) subprocess runs."""
    label_tif = tmp_path / "movie_labels.tif"
    label_tif.write_text("", encoding="utf-8")

    captured = {}

    def fake_create_trackastra_script(
        img_path, mask_path, model, mode, output_path, channel, dimension_order,
        batch_size="Auto",
    ):
        captured["mode"] = mode
        return "print('mock')\n"

    monkeypatch.setattr(
        trackastra_mod.TrackAstraEnvManager, "ensure_env_ready", lambda: True
    )
    monkeypatch.setattr(
        trackastra_mod.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
    )
    monkeypatch.setattr(
        trackastra_mod, "create_trackastra_script", fake_create_trackastra_script
    )

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(
        trackastra_mod.subprocess, "run", lambda *a, **k: _Completed()
    )

    # 'lip' is the common typo for 'ilp'.
    trackastra_mod.trackastra_tracking(
        None,
        mode="lip",
        label_pattern="_labels.tif",
        _source_filepath=str(label_tif),
        _output_folder=str(tmp_path),
        _output_suffix="_tracked",
    )
    assert captured["mode"] == "ilp"

    # Any other unknown mode falls back to greedy.
    trackastra_mod.trackastra_tracking(
        None,
        mode="bogus",
        label_pattern="_labels.tif",
        _source_filepath=str(label_tif),
        _output_folder=str(tmp_path),
        _output_suffix="_tracked",
    )
    assert captured["mode"] == "greedy"


def test_trackastra_tolerates_skip_load_none_image(tmp_path, monkeypatch):
    """With skip_load, image is None; the function must not crash on validation."""
    label_tif = tmp_path / "movie_labels.tif"
    label_tif.write_text("", encoding="utf-8")

    # Env not ready → returns early (None) without touching `image.shape`.
    monkeypatch.setattr(
        trackastra_mod.TrackAstraEnvManager, "ensure_env_ready", lambda: False
    )

    result = trackastra_mod.trackastra_tracking(
        None,
        label_pattern="_labels.tif",
        _source_filepath=str(label_tif),
        _output_folder=str(tmp_path),
        _output_suffix="_tracked",
    )
    # No crash; returns the (None) input unchanged when env is unavailable.
    assert result is None


def test_trackastra_env_requires_zarr_for_py311():
    assert "zarr" in TrackAstraEnvManager.REQUIRED_VERSIONS
    assert TrackAstraEnvManager.REQUIRED_VERSIONS["zarr"] == "3.0.0"
    assert TrackAstraEnvManager.REQUIRED_VERSIONS["python"] == "3.11"


def test_ensure_env_ready_recreates_env_on_python_version_mismatch(
    tmp_path, monkeypatch
):
    """When Python is too old, ensure_env_ready must delete+recreate rather than repair."""
    old_status = {
        "python": "3.10.0",
        "packages": {
            "gurobipy": {"present": True, "version": "13.0.0"},
            "ilpy": {"present": True, "version": "0.5.1"},
            "motile": {"present": True, "version": "0.4.0"},
            "trackastra": {"present": True, "version": "0.5.3"},
            "zarr": {"present": True, "version": "2.18.3"},
        },
    }
    new_status = {
        "python": "3.11.0",
        "packages": {
            "gurobipy": {"present": True, "version": "13.0.0"},
            "ilpy": {"present": True, "version": "0.5.1"},
            "motile": {"present": True, "version": "0.4.0"},
            "trackastra": {"present": True, "version": "0.5.3"},
            "zarr": {"present": True, "version": "3.0.0"},
        },
    }

    calls = []
    statuses = [old_status, new_status]

    monkeypatch.setattr(TrackAstraEnvManager, "check_env_exists", lambda: True)
    monkeypatch.setattr(TrackAstraEnvManager, "get_env_status", lambda: statuses.pop(0))
    monkeypatch.setattr(TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba")
    monkeypatch.setattr(
        trackastra_mod.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd) or type("R", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(TrackAstraEnvManager, "create_env", lambda: True)

    result = TrackAstraEnvManager.ensure_env_ready()

    assert result is True
    remove_calls = [c for c in calls if "remove" in c]
    assert remove_calls, "Expected env remove call when Python version is wrong"


def test_trackastra_env_needs_repair_when_zarr_missing():
    status = {
        "python": "3.10.0",
        "packages": {
            "gurobipy": {"present": True, "version": "13.0.0"},
            "ilpy": {"present": True, "version": "0.5.1"},
            "motile": {"present": True, "version": "0.4.0"},
            "trackastra": {"present": True, "version": "0.5.3"},
            "zarr": {"present": False, "version": None},
        },
    }

    needs_repair, reasons = TrackAstraEnvManager.env_needs_repair(status)
    assert needs_repair is True
    assert any("zarr" in reason for reason in reasons)
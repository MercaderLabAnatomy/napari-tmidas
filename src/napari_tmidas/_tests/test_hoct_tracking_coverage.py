"""Coverage tests for the HOCT tracking integration.

HOCT is not installed here, so every test drives the module the way the
plugin does at runtime -- through mocked ``subprocess`` calls and real
files on disk -- and pins the surrounding glue: GPU-pool distribution,
conda environment management, name/path matching, Zarr introspection,
CTC output assembly and the registry-registered entry point itself.

The fakes stop at ``subprocess.run``: everything above it (staging real
TIFF/Zarr inputs, building the CLI argv, stitching the CTC frames the fake
writes to disk, cleaning up temporaries) is the real module code, and the
assertions pin values rather than shapes so a mutation is caught rather
than merely covered.
"""

import os
import queue
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

import napari_tmidas.processing_functions.hoct_tracking as mod
from napari_tmidas._registry import BatchProcessingRegistry

zarr = pytest.importorskip("zarr")

T, Z, Y, X, C = 4, 3, 12, 16, 2


class _Completed:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode=0, stdout="out", stderr="err"):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _guard_run(*args, **kwargs):
    raise AssertionError(f"unexpected subprocess call: {args!r}")


@pytest.fixture(autouse=True)
def _isolated_module_state(monkeypatch, tmp_path):
    """No real subprocesses, no inherited env, no leaked GPU pool.

    The module shells out with a blocking ``subprocess.run`` (no ``Popen``,
    no reader thread), so every test that does not install its own fake gets
    a guard that turns an unnoticed real launch into a failure. ``Popen`` is
    guarded too, so a future switch to it cannot silently run ``conda``.
    """
    monkeypatch.setattr(mod.subprocess, "run", _guard_run)
    monkeypatch.setattr(mod.subprocess, "Popen", _guard_run)
    for name in ("HOCT_GPUS", "CUDA_VISIBLE_DEVICES", "GRB_LICENSE_FILE"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(mod, "_GPU_POOL", None)
    monkeypatch.setattr(mod, "_GPU_IDS", None)
    monkeypatch.setattr(mod, "_GPU_POOL_WORKERS_PER_GPU", None)
    monkeypatch.setattr(mod, "_GPU_POOL_KEY", None)


def _write_tif(path, data, axes=None):
    kwargs = {"photometric": "minisblack"}
    if axes is not None:
        kwargs["metadata"] = {"axes": axes}
    tifffile.imwrite(str(path), data, **kwargs)
    return path


def _plain_zarr(path, data):
    arr = zarr.create_array(
        store=str(path), shape=data.shape, dtype=data.dtype, overwrite=True
    )
    arr[:] = data
    return path


class TestDetectGpuIds:
    """GPU discovery honours overrides before ever shelling out."""

    def test_explicit_override_is_split_and_stripped(self):
        assert mod._detect_gpu_ids(" 0 , 1 ,, 2 ") == ["0", "1", "2"]

    @pytest.mark.parametrize("value", ["none", "CPU", " None "])
    def test_override_can_disable_pinning(self, value):
        assert mod._detect_gpu_ids(value) == []

    def test_env_override_used_when_argument_blank(self, monkeypatch):
        monkeypatch.setenv("HOCT_GPUS", "3,4")
        assert mod._detect_gpu_ids("   ") == ["3", "4"]

    def test_env_override_can_disable_pinning(self, monkeypatch):
        monkeypatch.setenv("HOCT_GPUS", "cpu")
        assert mod._detect_gpu_ids(None) == []

    def test_empty_env_override_disables_pinning(self, monkeypatch):
        monkeypatch.setenv("HOCT_GPUS", "")
        assert mod._detect_gpu_ids(None) == []

    def test_cuda_visible_devices_is_next_in_line(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
        assert mod._detect_gpu_ids(None) == ["1", "2"]

    def test_nvidia_smi_lines_are_counted(self, monkeypatch):
        listing = "GPU 0: NVIDIA A100\nGPU 1: NVIDIA A100\nnoise\n"
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            lambda *a, **k: _Completed(0, stdout=listing),
        )
        assert mod._detect_gpu_ids(None) == ["0", "1"]

    def test_nvidia_smi_without_gpu_lines_yields_nothing(self, monkeypatch):
        monkeypatch.setattr(
            mod.subprocess, "run", lambda *a, **k: _Completed(0, stdout="")
        )
        assert mod._detect_gpu_ids(None) == []

    def test_nvidia_smi_failure_yields_nothing(self, monkeypatch):
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            lambda *a, **k: _Completed(2, stdout="GPU 0"),
        )
        assert mod._detect_gpu_ids(None) == []

    def test_missing_nvidia_smi_is_swallowed(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(mod.subprocess, "run", boom)
        assert mod._detect_gpu_ids(None) == []

    def test_nvidia_smi_timeout_is_swallowed(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(["nvidia-smi"], 10)

        monkeypatch.setattr(mod.subprocess, "run", boom)
        assert mod._detect_gpu_ids(None) == []


class TestGpuPool:
    """The shared pool bounds concurrency to workers_per_gpu per card."""

    def test_pool_holds_each_gpu_workers_per_gpu_times(self, monkeypatch):
        monkeypatch.setattr(mod, "_detect_gpu_ids", lambda o: ["0", "1"])
        pool, gpu_ids = mod._get_gpu_pool(3, "0,1")
        assert gpu_ids == ["0", "1"]
        assert pool.qsize() == 6
        assert sorted(pool.get() for _ in range(6)) == [
            "0",
            "0",
            "0",
            "1",
            "1",
            "1",
        ]

    def test_pool_is_reused_for_the_same_key(self, monkeypatch):
        detected = []

        def fake_detect(override):
            detected.append(override)
            return ["0"]

        monkeypatch.setattr(mod, "_detect_gpu_ids", fake_detect)
        first, _ = mod._get_gpu_pool(1, "0")
        second, _ = mod._get_gpu_pool(1, "0")
        assert first is second
        assert detected == ["0"]

    def test_pool_is_rebuilt_when_the_key_changes(self, monkeypatch):
        monkeypatch.setattr(mod, "_detect_gpu_ids", lambda o: ["0"])
        first, _ = mod._get_gpu_pool(1, "0")
        second, _ = mod._get_gpu_pool(2, "0")
        assert first is not second
        assert second.qsize() == 2

    def test_workers_per_gpu_is_clamped_to_at_least_one(self, monkeypatch):
        monkeypatch.setattr(mod, "_detect_gpu_ids", lambda o: ["7"])
        pool, _ = mod._get_gpu_pool(0, None)
        assert pool.qsize() == 1
        assert mod._GPU_POOL_WORKERS_PER_GPU == 1

    def test_no_gpus_gives_an_empty_pool(self, monkeypatch):
        monkeypatch.setattr(mod, "_detect_gpu_ids", lambda o: [])
        pool, gpu_ids = mod._get_gpu_pool(2, "none")
        assert gpu_ids == []
        assert isinstance(pool, queue.Queue)
        assert pool.qsize() == 0


class TestNameMatching:
    """Raw/label filename pairing, the entry point's file discovery."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("movie.tif", "movie"),
            ("movie.TIFF", "movie"),
            ("movie.zarr", "movie"),
            ("movie.png", "movie.png"),
            ("movie", "movie"),
        ],
    )
    def test_strip_known_image_suffix(self, name, expected):
        assert mod._strip_known_image_suffix(name) == expected

    def test_candidates_from_trailing_pattern(self):
        raw_base, candidates = mod._raw_candidates_from_label_name(
            "movie_labels.tif", "_labels.tif"
        )
        assert raw_base == "movie"
        assert candidates == ["movie.tif", "movie.tiff", "movie.zarr"]

    def test_candidates_from_embedded_pattern_keep_suffix(self):
        raw_base, candidates = mod._raw_candidates_from_label_name(
            "movie_labels_v2.tif", "_labels"
        )
        # The pattern sits mid-name, so it is removed in place and the
        # resulting basename already carries a known image suffix.
        assert raw_base == "movie_v2.tif"
        assert candidates == ["movie_v2.tif"]

    def test_only_the_first_embedded_pattern_is_removed(self):
        # A doubled pattern must not collapse both occurrences, or the
        # derived raw name stops matching the file that is actually there.
        raw_base, candidates = mod._raw_candidates_from_label_name(
            "movie_labels_a_labels_b.png", "_labels"
        )
        assert raw_base == "movie_a_labels_b.png"
        assert candidates == [
            "movie_a_labels_b.png.tif",
            "movie_a_labels_b.png.tiff",
            "movie_a_labels_b.png.zarr",
        ]

    def test_find_matching_raw_path_prefers_existing_candidate(self, tmp_path):
        _write_tif(tmp_path / "movie.tiff", np.zeros((2, 4, 4), np.uint8))
        label = tmp_path / "movie_labels.tif"
        raw_base, candidates, found = mod._find_matching_raw_path(
            str(label), "_labels.tif"
        )
        assert raw_base == "movie"
        assert "movie.tif" in candidates
        assert found == str(tmp_path / "movie.tiff")

    def test_candidates_are_probed_in_suffix_order(self, tmp_path):
        # With several partners on disk the first supported suffix wins;
        # a reversed scan would hand HOCT the Zarr store instead.
        _write_tif(tmp_path / "movie.tif", np.zeros((2, 4, 4), np.uint8))
        _write_tif(tmp_path / "movie.tiff", np.zeros((2, 4, 4), np.uint8))
        _plain_zarr(tmp_path / "movie.zarr", np.zeros((2, 4, 4), np.uint8))
        _, candidates, found = mod._find_matching_raw_path(
            str(tmp_path / "movie_labels.tif"), "_labels.tif"
        )
        assert candidates == ["movie.tif", "movie.tiff", "movie.zarr"]
        assert found == str(tmp_path / "movie.tif")

    def test_find_matching_raw_path_reports_no_match(self, tmp_path):
        raw_base, candidates, found = mod._find_matching_raw_path(
            str(tmp_path / "movie_labels.tif"), "_labels.tif"
        )
        assert found is None
        assert raw_base == "movie"
        assert len(candidates) == 3


class TestResolveGurobiLicense:
    """First hit wins: argument, then env var, then ~/gurobi.lic."""

    def test_explicit_path_wins(self, tmp_path):
        lic = tmp_path / "explicit.lic"
        lic.write_text("KEY")
        assert mod._resolve_gurobi_license(str(lic)) == str(lic)

    def test_missing_explicit_path_warns_and_falls_back(
        self, monkeypatch, tmp_path, capsys
    ):
        env_lic = tmp_path / "env.lic"
        env_lic.write_text("KEY")
        monkeypatch.setenv("GRB_LICENSE_FILE", str(env_lic))
        resolved = mod._resolve_gurobi_license(str(tmp_path / "gone.lic"))
        assert resolved == str(env_lic)
        assert "gone.lic' not found" in capsys.readouterr().out

    def test_env_var_pointing_at_a_missing_file_is_ignored(
        self, monkeypatch, tmp_path
    ):
        home = tmp_path / "home3"
        home.mkdir()
        (home / "gurobi.lic").write_text("KEY")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("GRB_LICENSE_FILE", str(tmp_path / "vanished.lic"))
        assert mod._resolve_gurobi_license("") == str(home / "gurobi.lic")

    def test_home_license_is_the_last_resort(self, monkeypatch, tmp_path):
        home = tmp_path / "home2"
        home.mkdir()
        (home / "gurobi.lic").write_text("KEY")
        monkeypatch.setenv("HOME", str(home))
        assert mod._resolve_gurobi_license("") == str(home / "gurobi.lic")

    def test_nothing_found_returns_none(self):
        assert mod._resolve_gurobi_license("") is None


class TestHoctEnvManager:
    """Conda environment bootstrap, with every subprocess mocked."""

    def test_get_conda_cmd_prefers_mamba(self, monkeypatch):
        monkeypatch.setattr(
            mod.shutil,
            "which",
            lambda name: "/x/mamba" if name == "mamba" else None,
        )
        assert mod.HoctEnvManager.get_conda_cmd() == "mamba"

    def test_get_conda_cmd_falls_back_to_conda(self, monkeypatch):
        monkeypatch.setattr(
            mod.shutil,
            "which",
            lambda name: "/x/conda" if name == "conda" else None,
        )
        assert mod.HoctEnvManager.get_conda_cmd() == "conda"

    def test_get_conda_cmd_raises_without_a_package_manager(self, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="Neither conda nor mamba"):
            mod.HoctEnvManager.get_conda_cmd()

    @pytest.mark.parametrize(
        ("method", "binary", "timeout"),
        # The two probes differ only in *which* executable they run, so the
        # binary (not the shared ``--version`` flag) is what must be pinned:
        # `check_env_exists` only proves the env resolves, `_hoct_cli_ready`
        # proves the HOCT entry point itself is installed.
        [("check_env_exists", "python", 10), ("_hoct_cli_ready", "hoct", 30)],
    )
    def test_probe_returns_true_on_zero_exit(
        self, monkeypatch, method, binary, timeout
    ):
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append((list(cmd), kwargs))
            return _Completed(0)

        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        assert getattr(mod.HoctEnvManager, method)() is True
        assert len(seen) == 1
        cmd, kwargs = seen[0]
        assert cmd == ["conda", "run", "-n", "hoct", binary, "--version"]
        assert kwargs["timeout"] == timeout
        assert kwargs["capture_output"] is True
        # `check=True` would turn a bad exit into an exception instead of the
        # False that both callers rely on.
        assert kwargs.get("check", False) is False

    @pytest.mark.parametrize("method", ["check_env_exists", "_hoct_cli_ready"])
    def test_probe_returns_false_on_nonzero_exit(self, monkeypatch, method):
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(
            mod.subprocess, "run", lambda *a, **k: _Completed(127)
        )
        assert getattr(mod.HoctEnvManager, method)() is False

    @pytest.mark.parametrize("method", ["check_env_exists", "_hoct_cli_ready"])
    def test_probe_swallows_launch_errors(self, monkeypatch, method):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(["conda"], 10)

        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(mod.subprocess, "run", boom)
        assert getattr(mod.HoctEnvManager, method)() is False

    def test_create_env_builds_env_then_pip_installs(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _Completed(0)

        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(mod.subprocess, "run", fake_run)

        assert mod.HoctEnvManager.create_env() is True
        assert calls[0][:2] == ["conda", "clean"]
        # `--no-default-packages` is conda-only and must stay before `-y`.
        assert calls[1] == [
            "conda",
            "create",
            "-n",
            "hoct",
            "python=3.11",
            "--no-default-packages",
            "-y",
        ]
        assert calls[2][-3:] == ["pip", "install", "hoct[bioio]"]

    def test_create_env_omits_conda_only_flag_for_mamba(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            lambda cmd, **k: (calls.append(list(cmd)), _Completed(0))[1],
        )
        mod.HoctEnvManager.create_env()
        assert "--no-default-packages" not in calls[1]

    def test_create_env_skips_creation_when_env_exists(
        self, monkeypatch, capsys
    ):
        calls = []
        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            lambda cmd, **k: (calls.append(list(cmd)), _Completed(0))[1],
        )
        assert mod.HoctEnvManager.create_env() is True
        assert len(calls) == 1  # only the pip install
        assert "already exists" in capsys.readouterr().out

    def test_create_env_returns_false_when_creation_fails(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[1] == "create":
                raise subprocess.CalledProcessError(1, cmd)
            return _Completed(0)

        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        assert mod.HoctEnvManager.create_env() is False

    def test_create_env_returns_false_when_pip_fails(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if "pip" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return _Completed(0)

        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        assert mod.HoctEnvManager.create_env() is False

    def test_ensure_env_ready_short_circuits_when_healthy(self, monkeypatch):
        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "_hoct_cli_ready", lambda: True
        )
        assert mod.HoctEnvManager.ensure_env_ready() is True

    def test_ensure_env_ready_fails_when_creation_fails(self, monkeypatch):
        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(mod.HoctEnvManager, "create_env", lambda: False)
        assert mod.HoctEnvManager.ensure_env_ready() is False

    def test_ensure_env_ready_repairs_a_missing_cli(self, monkeypatch):
        states = iter([False, True])
        calls = []
        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "_hoct_cli_ready", lambda: next(states)
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            lambda cmd, **k: (calls.append(list(cmd)), _Completed(0))[1],
        )
        assert mod.HoctEnvManager.ensure_env_ready() is True
        assert calls[0][-3:] == ["install", "--upgrade", "hoct[bioio]"]

    def test_ensure_env_ready_gives_up_if_repair_does_not_help(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "_hoct_cli_ready", lambda: False
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(
            mod.subprocess, "run", lambda *a, **k: _Completed(0)
        )
        assert mod.HoctEnvManager.ensure_env_ready() is False

    def test_ensure_env_ready_fails_when_repair_install_raises(
        self, monkeypatch
    ):
        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(
            mod.HoctEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "_hoct_cli_ready", lambda: False
        )
        monkeypatch.setattr(
            mod.HoctEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(mod.subprocess, "run", boom)
        assert mod.HoctEnvManager.ensure_env_ready() is False


class TestZarrIntrospection:
    """Finding the full-resolution array inside plain and OME stores."""

    def test_plain_array_store_is_returned_directly(self, tmp_path):
        data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        path = _plain_zarr(tmp_path / "plain.zarr", data)
        opened = mod._open_zarr_array(str(path))
        assert tuple(opened.shape) == (2, 3, 4)
        assert np.array_equal(np.asarray(opened[:]), data)

    def test_broken_multiscales_falls_back_to_a_named_array(self, tmp_path):
        path = tmp_path / "broken.zarr"
        root = zarr.create_group(store=str(path), overwrite=True)
        arr = root.create_array("0", shape=(2, 2), dtype="uint8")
        arr[:] = 5
        root.attrs["multiscales"] = [{"datasets": [{"path": "missing"}]}]
        found = mod._open_zarr_array(str(path))
        assert tuple(found.shape) == (2, 2)
        assert int(np.asarray(found[:]).max()) == 5

    def test_group_without_metadata_scans_known_names(self, tmp_path):
        path = tmp_path / "named.zarr"
        root = zarr.create_group(store=str(path), overwrite=True)
        arr = root.create_array("data", shape=(3, 3), dtype="uint8")
        arr[:] = 1
        found = mod._open_zarr_array(str(path))
        assert tuple(found.shape) == (3, 3)
        assert np.array_equal(np.asarray(found[:]), np.ones((3, 3), np.uint8))

    @staticmethod
    def _pyramid(path, attrs_key):
        """A two-level pyramid whose levels are *not* named "0"/"s0"/"data".

        The unusual names are deliberate: if the multiscales metadata were
        ignored, the fallback name scan could not rescue the lookup, so
        these tests fail loudly instead of passing by accident.
        """
        root = zarr.create_group(store=str(path), overwrite=True)
        full = root.create_array("full", shape=(2, 6, 6), dtype="uint8")
        full[:] = 7
        half = root.create_array("half", shape=(2, 3, 3), dtype="uint8")
        half[:] = 1
        meta = [{"datasets": [{"path": "full"}, {"path": "half"}]}]
        if attrs_key is None:
            root.attrs["multiscales"] = meta
        else:
            root.attrs[attrs_key] = {"multiscales": meta}
        return path

    def test_ome_v04_multiscales_pick_the_first_dataset(self, tmp_path):
        path = self._pyramid(tmp_path / "ome04.zarr", None)
        found = mod._open_zarr_array(str(path))
        # The first dataset is the full-resolution level; picking "half"
        # instead would silently downsample every tracked movie.
        assert tuple(found.shape) == (2, 6, 6)
        assert int(np.asarray(found[:]).min()) == 7

    def test_ome_v05_nests_multiscales_under_an_ome_key(self, tmp_path):
        """v0.5 stores put ``multiscales`` inside an ``ome`` attribute."""
        path = self._pyramid(tmp_path / "ome05.zarr", "ome")
        found = mod._open_zarr_array(str(path))
        assert tuple(found.shape) == (2, 6, 6)
        assert int(np.asarray(found[:]).min()) == 7

    def test_group_with_no_array_raises(self, tmp_path):
        path = tmp_path / "empty.zarr"
        zarr.create_group(store=str(path), overwrite=True)
        with pytest.raises(ValueError, match="Could not find an array"):
            mod._open_zarr_array(str(path))

    def test_peek_raw_shape_returns_none_for_unreadable_zarr(self, tmp_path):
        assert mod._peek_raw_shape(str(tmp_path / "absent.zarr")) is None

    def test_peek_raw_shape_reads_tiff_series_metadata(self, tmp_path):
        path = _write_tif(
            tmp_path / "m.tif", np.zeros((T, Y, X), np.uint16), axes="TYX"
        )
        assert tuple(mod._peek_raw_shape(str(path))) == (T, Y, X)

    def test_peek_raw_dtype_for_tiff_and_zarr(self, tmp_path):
        tif = _write_tif(
            tmp_path / "d.tif", np.zeros((T, Y, X), np.uint16), axes="TYX"
        )
        zpath = _plain_zarr(
            tmp_path / "d.zarr", np.zeros((T, Y, X), np.float32)
        )
        assert mod._peek_raw_dtype(str(tif)) == np.uint16
        assert mod._peek_raw_dtype(str(zpath)) == np.float32

    def test_zarr_axes_delegates_to_the_file_selector(self, monkeypatch):
        from napari_tmidas import _file_selector as fs

        seen = []

        def fake_detect(path):
            seen.append(path)
            return "TCZYX"

        monkeypatch.setattr(fs, "detect_axes_from_zarr_path", fake_detect)
        assert mod._zarr_axes("/data/whatever.zarr") == "TCZYX"
        assert seen == ["/data/whatever.zarr"]

    def test_zarr_axes_returns_none_when_detection_raises(self, monkeypatch):
        from napari_tmidas import _file_selector as fs

        def boom(path):
            raise RuntimeError("no metadata")

        monkeypatch.setattr(fs, "detect_axes_from_zarr_path", boom)
        assert mod._zarr_axes("whatever.zarr") is None


class TestMovieShape:
    """``_movie_shape`` mirrors HOCT's own ``_reduce_to_movie``.

    It decides whether a raw store can be handed to HOCT untouched, so a
    wrong answer here silently ships a mismatched raw/label pair.
    """

    def test_channel_axis_is_dropped_when_axes_are_known(self):
        assert mod._movie_shape((T, C, Z, Y, X), "TCZYX") == (T, Z, Y, X)
        assert mod._movie_shape((T, Z, C, Y, X), "TZCYX") == (T, Z, Y, X)

    def test_sample_axis_counts_as_a_channel_axis(self):
        assert mod._movie_shape((T, 3, Y, X), "TSYX") == (T, Y, X)

    def test_leading_length_one_axes_collapse(self):
        assert mod._movie_shape((T, 1, Y, X)) == (T, Y, X)

    def test_trailing_yx_survive_even_when_length_one(self):
        assert mod._movie_shape((T, 1, 1)) == (T, 1, 1)

    def test_axes_of_the_wrong_length_are_ignored(self):
        # A stale hint must not silently delete a real axis.
        assert mod._movie_shape((T, C, Y, X), "TCZYX") == (T, C, Y, X)

    def test_axes_without_a_channel_letter_change_nothing(self):
        assert mod._movie_shape((T, Z, Y, X), "TZYX") == (T, Z, Y, X)

    def test_none_passes_through(self):
        assert mod._movie_shape(None) is None


class TestStagingHelpers:
    """Chunking, destination creation and temp-path cleanup."""

    def test_tiff_pages_that_do_not_split_across_time_are_read_whole(
        self, tmp_path, capsys
    ):
        """A volumetric single-page stack cannot be read per timepoint."""
        data = (
            np.arange(3 * Y * X, dtype=np.uint16).reshape(3, Y, X) % 97
        ).astype(np.uint16)
        src = tmp_path / "volumetric.tif"
        tifffile.imwrite(
            str(src), data, photometric="minisblack", volumetric=True
        )
        with tifffile.TiffFile(str(src)) as tif:
            assert len(tif.series[0].pages) == 1  # the premise of the branch

        dest = mod._stage_tiff_as_zarr(str(src), tmp_path / "staged.zarr")
        out = capsys.readouterr().out
        assert "do not split evenly across 3 timepoints" in out
        staged = zarr.open(str(dest), mode="r")
        assert tuple(staged.shape) == (3, Y, X)
        assert np.array_equal(np.asarray(staged[:]), data)

    def test_stage_chunks_caps_spatial_axes_and_keeps_one_timepoint(self):
        assert mod._stage_chunks((3, 1024, 700)) == (1, 3, 512, 512)
        assert mod._stage_chunks((1024, 700)) == (1, 512, 512)

    def test_create_stage_array_replaces_an_existing_destination(
        self, tmp_path
    ):
        dest = tmp_path / "stage.zarr"
        dest.mkdir()
        (dest / "stale.txt").write_text("junk")
        arr = mod._create_stage_array(dest, (T, Y, X), np.uint32)
        assert tuple(arr.shape) == (T, Y, X)
        assert arr.dtype == np.uint32
        # One timepoint per chunk is what makes HOCT read the store lazily;
        # a default (whole-array) chunking would defeat the staging.
        assert tuple(arr.chunks) == (1, Y, X)
        assert not (dest / "stale.txt").exists()

    def test_create_stage_array_caps_large_spatial_blocks(self, tmp_path):
        arr = mod._create_stage_array(
            tmp_path / "big.zarr", (2, 40, 1024, 700), np.uint16
        )
        assert tuple(arr.chunks) == (1, 16, 512, 512)

    def test_unique_tag_differs_between_calls_in_one_process(self):
        # A batch run stages several movies from threads of ONE process, so
        # a PID-only tag would make concurrent jobs overwrite each other.
        tags = {mod._unique_tag() for _ in range(20)}
        assert len(tags) == 20
        assert all(t.startswith(f"{os.getpid()}_") for t in tags)

    def test_uncompressed_bytes(self):
        assert mod._uncompressed_bytes(None, np.uint8) == 0
        assert mod._uncompressed_bytes((2, 3, 4), np.uint32) == 96

    def test_cleanup_removes_files_directories_and_ignores_absent(
        self, tmp_path
    ):
        a_file = tmp_path / "temp.tif"
        a_file.write_bytes(b"x")
        a_dir = tmp_path / "temp.zarr"
        a_dir.mkdir()
        (a_dir / "chunk").write_bytes(b"x")
        mod._cleanup_paths(
            [str(a_file), str(a_dir), str(tmp_path / "never.tif")]
        )
        assert not a_file.exists()
        assert not a_dir.exists()

    def test_stage_label_input_off_mode_passes_through(self, tmp_path):
        path, cleanup = mod._stage_label_input(
            str(tmp_path / "x_labels.tif"), tmp_path, "off", "tag"
        )
        assert (path, cleanup) == (str(tmp_path / "x_labels.tif"), None)

    def test_stage_label_input_leaves_zarr_alone(self, tmp_path):
        store = str(tmp_path / "labels.zarr")
        path, cleanup = mod._stage_label_input(store, tmp_path, "on", "tag")
        assert (path, cleanup) == (store, None)


class TestAssembleCtcOutput:
    """Stitching HOCT's per-frame CTC masks into one multi-page TIFF."""

    def test_missing_masks_raise(self, tmp_path):
        ctc = tmp_path / "ctc"
        ctc.mkdir()
        with pytest.raises(RuntimeError, match="No CTC mask files"):
            mod._assemble_ctc_output(ctc, tmp_path / "out.tif")

    def test_two_dimensional_frames_become_a_tyx_stack(self, tmp_path):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 40, size=(T, Y, X)).astype(np.int64)
        ctc = tmp_path / "ctc"
        ctc.mkdir()
        for i, frame in enumerate(frames):
            _write_tif(ctc / f"mask{i:03d}.tif", frame)

        out = tmp_path / "tracked.tif"
        shape = mod._assemble_ctc_output(ctc, out)

        assert shape == (T, Y, X)
        written = tifffile.imread(str(out))
        assert written.shape == (T, Y, X)
        # int64 CTC masks are cast down so napari shows them as labels.
        assert written.dtype == np.uint32
        assert np.array_equal(written, frames)

    def test_three_dimensional_frames_become_a_tzyx_stack(self, tmp_path):
        rng = np.random.default_rng(1)
        frames = rng.integers(0, 9, size=(T, Z, Y, X)).astype(np.int64)
        ctc = tmp_path / "ctc3"
        ctc.mkdir()
        for i, frame in enumerate(frames):
            _write_tif(ctc / f"mask{i:03d}.tif", frame)

        out = tmp_path / "tracked3.tif"
        shape = mod._assemble_ctc_output(ctc, out)

        assert shape == (T, Z, Y, X)
        with tifffile.TiffFile(str(out)) as tif:
            assert tif.series[0].axes == "TZYX"
        written = tifffile.imread(str(out))
        assert written.dtype == np.uint32
        assert np.array_equal(written, frames)


class TestPrepareRawInputBranches:
    """Channel resolution and staging decisions for the raw image."""

    def _multichannel_tif(self, tmp_path, seed=0):
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 500, size=(T, C, Y, X)).astype(np.uint16)
        return raw, _write_tif(tmp_path / "raw.tif", raw, axes="TCYX")

    def test_small_single_channel_tiff_is_passed_through(
        self, tmp_path, monkeypatch
    ):
        from napari_tmidas import _file_selector as fs

        monkeypatch.setattr(
            fs, "detect_channels_for_file", lambda p: (1, None)
        )
        raw = np.zeros((T, Y, X), np.uint16)
        path = _write_tif(tmp_path / "single.tif", raw, axes="TYX")
        used, cleanup = mod._prepare_raw_input(
            str(path), "", "Auto", tmp_path, "auto", (T, Y, X), "tag"
        )
        assert used == str(path)
        assert cleanup is None

    def test_single_channel_tiff_is_staged_when_forced(
        self, tmp_path, monkeypatch
    ):
        from napari_tmidas import _file_selector as fs

        monkeypatch.setattr(
            fs, "detect_channels_for_file", lambda p: (1, None)
        )
        rng = np.random.default_rng(2)
        raw = rng.integers(0, 300, size=(T, Y, X)).astype(np.uint16)
        path = _write_tif(tmp_path / "single_on.tif", raw, axes="TYX")
        used, cleanup = mod._prepare_raw_input(
            str(path), "", "Auto", tmp_path, "on", (T, Y, X), "tag"
        )
        assert used.endswith(".zarr")
        assert cleanup == used
        assert np.array_equal(np.asarray(zarr.open(used, mode="r")[:]), raw)
        mod._cleanup_paths([cleanup])

    def test_blank_channel_on_multichannel_input_uses_channel_zero(
        self, tmp_path, capsys
    ):
        raw, path = self._multichannel_tif(tmp_path, seed=3)
        used, cleanup = mod._prepare_raw_input(
            str(path), "", "TCYX", tmp_path, "on", (T, Y, X), "tag"
        )
        assert "using channel 0" in capsys.readouterr().out
        assert np.array_equal(
            np.asarray(zarr.open(used, mode="r")[:]), raw[:, 0]
        )
        mod._cleanup_paths([cleanup])

    def test_unparsable_channel_falls_back_to_zero(self, tmp_path, capsys):
        raw, path = self._multichannel_tif(tmp_path, seed=4)
        used, cleanup = mod._prepare_raw_input(
            str(path), "left", "TCYX", tmp_path, "on", (T, Y, X), "tag"
        )
        assert "Invalid channel 'left'" in capsys.readouterr().out
        assert np.array_equal(
            np.asarray(zarr.open(used, mode="r")[:]), raw[:, 0]
        )
        mod._cleanup_paths([cleanup])

    def test_out_of_range_channel_falls_back_to_zero(self, tmp_path, capsys):
        raw, path = self._multichannel_tif(tmp_path, seed=5)
        used, cleanup = mod._prepare_raw_input(
            str(path), "9", "TCYX", tmp_path, "on", (T, Y, X), "tag"
        )
        assert "out of bounds" in capsys.readouterr().out
        assert np.array_equal(
            np.asarray(zarr.open(used, mode="r")[:]), raw[:, 0]
        )
        mod._cleanup_paths([cleanup])

    def test_zarr_channel_is_streamed_into_a_staged_store(self, tmp_path):
        rng = np.random.default_rng(21)
        raw = rng.integers(0, 500, size=(T, C, Y, X)).astype(np.uint16)
        path = _plain_zarr(tmp_path / "multi.zarr", raw)
        used, cleanup = mod._prepare_raw_input(
            str(path), "1", "TCYX", tmp_path, "on", (T, Y, X), "tag"
        )
        assert used.endswith(".zarr")
        assert cleanup == used
        staged = zarr.open(used, mode="r")
        assert tuple(staged.shape) == (T, Y, X)
        assert staged.dtype == np.uint16  # source dtype is preserved
        assert np.array_equal(np.asarray(staged[:]), raw[:, 1])
        mod._cleanup_paths([cleanup])

    def test_matching_zarr_channel_zero_is_handed_over_untouched(
        self, tmp_path, monkeypatch, capsys
    ):
        from napari_tmidas import _file_selector as fs

        monkeypatch.setattr(
            fs, "detect_axes_from_zarr_path", lambda p: "TCYX"
        )
        rng = np.random.default_rng(22)
        raw = rng.integers(0, 500, size=(T, C, Y, X)).astype(np.uint16)
        path = _plain_zarr(tmp_path / "pass.zarr", raw)
        used, cleanup = mod._prepare_raw_input(
            str(path), "0", "TCYX", tmp_path, "on", (T, Y, X), "tag"
        )
        # HOCT's own reader keeps channel 0 and reads lazily, so no copy.
        assert (used, cleanup) == (str(path), None)
        assert "passing it to HOCT directly" in capsys.readouterr().out
        assert list(tmp_path.glob("hoct_raw_*")) == []

    def test_zarr_channel_zero_is_staged_when_the_shape_disagrees(
        self, tmp_path, monkeypatch
    ):
        """The passthrough is conditional on the reduced shape matching."""
        from napari_tmidas import _file_selector as fs

        monkeypatch.setattr(
            fs, "detect_axes_from_zarr_path", lambda p: "TCYX"
        )
        rng = np.random.default_rng(23)
        raw = rng.integers(0, 500, size=(T, C, Y, X)).astype(np.uint16)
        path = _plain_zarr(tmp_path / "mismatch.zarr", raw)
        used, cleanup = mod._prepare_raw_input(
            str(path), "0", "TCYX", tmp_path, "on", (T, Y + 1, X), "tag"
        )
        assert used.endswith(".zarr")
        assert used != str(path)
        assert np.array_equal(
            np.asarray(zarr.open(used, mode="r")[:]), raw[:, 0]
        )
        mod._cleanup_paths([cleanup])

    def test_staging_off_writes_a_temporary_tiff_from_a_tiff(self, tmp_path):
        raw, path = self._multichannel_tif(tmp_path, seed=6)
        used, cleanup = mod._prepare_raw_input(
            str(path), "1", "TCYX", tmp_path, "off", (T, Y, X), "tag"
        )
        assert used.endswith(".tif")
        assert cleanup == used
        assert np.array_equal(tifffile.imread(used), raw[:, 1])
        mod._cleanup_paths([cleanup])

    def test_staging_off_writes_a_temporary_tiff_from_a_zarr(self, tmp_path):
        rng = np.random.default_rng(7)
        raw = rng.integers(0, 500, size=(T, C, Y, X)).astype(np.uint16)
        path = _plain_zarr(tmp_path / "raw.zarr", raw)
        used, cleanup = mod._prepare_raw_input(
            str(path), "1", "TCYX", tmp_path, "off", (T, Y, X), "tag"
        )
        assert used.endswith(".tif")
        assert np.array_equal(tifffile.imread(used), raw[:, 1])
        mod._cleanup_paths([cleanup])


class _RunRecorder:
    """Fake ``subprocess.run`` that plays the role of the ``hoct`` CLI."""

    def __init__(
        self,
        frames=None,
        returncode=0,
        stdout="HOCT-STDOUT",
        stderr="HOCT-STDERR",
    ):
        self.frames = frames
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.cmds = []
        self.envs = []

    def __call__(self, cmd, **kwargs):
        self.cmds.append(list(cmd))
        self.envs.append(dict(kwargs.get("env") or {}))
        if self.frames is not None:
            ctc = Path(cmd[cmd.index("-o") + 1])
            ctc.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(self.frames):
                _write_tif(ctc / f"mask{i:03d}.tif", frame)
        return _Completed(self.returncode, self.stdout, self.stderr)


@pytest.fixture
def hoct_pair(tmp_path):
    """A raw/label TIFF pair on disk, plus the frames HOCT would return."""
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 6, size=(T, Y, X)).astype(np.uint16)
    raw = rng.integers(0, 500, size=(T, Y, X)).astype(np.uint16)
    tracked = rng.integers(0, 6, size=(T, Y, X)).astype(np.int64)
    return types.SimpleNamespace(
        labels=labels,
        raw=raw,
        tracked=tracked,
        label_path=_write_tif(
            tmp_path / "movie_labels.tif", labels, axes="TYX"
        ),
        raw_path=_write_tif(tmp_path / "movie.tif", raw, axes="TYX"),
        out_dir=tmp_path,
    )


@pytest.fixture
def ready_env(monkeypatch):
    """Pretend the dedicated conda env exists and inputs are 1-channel."""
    from napari_tmidas import _file_selector as fs

    monkeypatch.setattr(mod.HoctEnvManager, "ensure_env_ready", lambda: True)
    monkeypatch.setattr(mod.HoctEnvManager, "get_conda_cmd", lambda: "conda")
    monkeypatch.setattr(
        fs, "detect_channels_for_file", lambda p, image_data=None: (1, None)
    )


class TestHoctTrackingGuards:
    """Cheap rejections before any environment or subprocess work."""

    def test_two_dimensional_input_is_skipped(self, capsys):
        assert mod.hoct_tracking(np.zeros((8, 8), np.uint16)) is None
        assert "not a time series" in capsys.readouterr().out

    def test_single_timepoint_is_skipped(self, capsys):
        assert mod.hoct_tracking(np.zeros((1, 8, 8), np.uint16)) is None
        assert "only one timepoint" in capsys.readouterr().out

    def test_invalid_options_are_normalised(self, monkeypatch, capsys):
        monkeypatch.setattr(
            mod.HoctEnvManager, "ensure_env_ready", lambda: False
        )
        result = mod.hoct_tracking(
            np.zeros((2, 8, 8), np.uint16),
            device="tpu",
            tile="sometimes",
            stage_inputs="maybe",
        )
        out = capsys.readouterr().out
        assert result is None
        assert "invalid device 'tpu'" in out
        assert "invalid tile mode 'sometimes'" in out
        assert "invalid stage_inputs 'maybe'" in out
        assert "Failed to prepare HOCT environment" in out

    def test_unknown_source_path_is_skipped(
        self, monkeypatch, ready_env, capsys
    ):
        import inspect

        monkeypatch.setattr(inspect, "stack", list)  # no caller frames
        result = mod.hoct_tracking(
            np.zeros((2, 8, 8), np.uint16), device="cpu"
        )
        assert result is None
        assert "Could not determine input file path" in capsys.readouterr().out

    def test_skip_load_reports_reading_dimensions_from_file(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)
        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        out = capsys.readouterr().out
        assert "Input array not loaded (skip_load)" in out
        assert "movie_labels.tif" in out
        # image=None must not short-circuit the run: the subprocess is still
        # driven from the file and the stitched output is still produced.
        assert result == str(hoct_pair.out_dir / "movie_hoct_tracked.tif")
        assert np.array_equal(
            tifffile.imread(result), hoct_pair.tracked
        )
        assert run.cmds[0][6:8] == [
            str(hoct_pair.raw_path),
            str(hoct_pair.label_path),
        ]

    def test_label_file_without_a_raw_partner_is_skipped(
        self, tmp_path, ready_env, capsys
    ):
        orphan = _write_tif(
            tmp_path / "orphan_labels.tif",
            np.zeros((T, Y, X), np.uint16),
            axes="TYX",
        )
        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(orphan)
        )
        out = capsys.readouterr().out
        assert result is None
        assert "Could not find raw image" in out
        assert "requires a matching raw image" in out

    def test_raw_file_without_a_label_partner_is_skipped(
        self, tmp_path, ready_env, capsys
    ):
        lonely = _write_tif(
            tmp_path / "lonely.tif",
            np.zeros((T, Y, X), np.uint16),
            axes="TYX",
        )
        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(lonely)
        )
        assert result is None
        assert "No label file found" in capsys.readouterr().out

    def test_input_preparation_failure_is_reported(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        def boom(*args, **kwargs):
            raise ValueError("staging blew up")

        monkeypatch.setattr(mod, "_stage_label_input", boom)
        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        assert result is None
        assert "Failed to prepare HOCT inputs: staging blew up" in (
            capsys.readouterr().out
        )

    def test_input_preparation_failure_deletes_already_staged_stores(
        self, monkeypatch, ready_env, hoct_pair, tmp_path, capsys
    ):
        """The label store is staged first; a later failure must remove it."""
        staged = {}
        real_stage = mod._stage_label_input

        def record(mask_path, work_dir, stage_mode, tag):
            path, cleanup = real_stage(mask_path, work_dir, stage_mode, tag)
            staged["path"] = cleanup
            return path, cleanup

        def boom(*args, **kwargs):
            raise ValueError("raw prep blew up")

        monkeypatch.setattr(mod, "_stage_label_input", record)
        monkeypatch.setattr(mod, "_prepare_raw_input", boom)

        result = mod.hoct_tracking(
            None,
            device="cpu",
            stage_inputs="on",
            _source_filepath=str(hoct_pair.label_path),
        )
        assert result is None
        assert "Failed to prepare HOCT inputs: raw prep blew up" in (
            capsys.readouterr().out
        )
        # Staging really happened, and the store it created is gone again.
        assert staged["path"] is not None
        assert staged["path"].endswith(".zarr")
        assert not Path(staged["path"]).exists()
        assert list(tmp_path.glob("*.zarr")) == []


class TestHoctTrackingRun:
    """The CLI invocation, its environment and the assembled output."""

    def test_successful_run_writes_a_stitched_uint32_tiff(
        self, monkeypatch, ready_env, hoct_pair, tmp_path
    ):
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None,
            device="cpu",
            window=7,
            max_distance=12.5,
            neighbors=4,
            max_dt=2,
            tile="off",
            _source_filepath=str(hoct_pair.label_path),
            _output_folder=str(out_dir),
        )

        expected = out_dir / "movie_hoct_tracked.tif"
        assert result == str(expected)
        written = tifffile.imread(str(expected))
        assert written.dtype == np.uint32
        assert np.array_equal(written, hoct_pair.tracked)

        cmd = run.cmds[0]
        assert cmd[:6] == ["conda", "run", "-n", "hoct", "hoct", "track"]
        assert cmd[6] == str(hoct_pair.raw_path)
        assert cmd[7] == str(hoct_pair.label_path)
        assert cmd[cmd.index("-d") + 1] == "cpu"
        assert cmd[cmd.index("-w") + 1] == "7"
        assert cmd[cmd.index("--max-distance") + 1] == "12.5"
        assert cmd[cmd.index("--neighbors") + 1] == "4"
        assert cmd[cmd.index("--max-dt") + 1] == "2"
        assert cmd[cmd.index("--tile") + 1] == "off"
        assert "--overwrite" in cmd
        assert "-m" not in cmd  # empty model => let HOCT pick its default
        # The CTC scratch directory is removed once it has been stitched.
        assert not Path(cmd[cmd.index("-o") + 1]).exists()

    def test_model_and_scale_are_forwarded_to_the_cli(
        self, monkeypatch, ready_env, hoct_pair
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        mod.hoct_tracking(
            None,
            device="cpu",
            model="  my-model  ",
            scale="1, 0.5 0.5",
            _source_filepath=str(hoct_pair.label_path),
        )

        cmd = run.cmds[0]
        assert cmd[cmd.index("-m") + 1] == "my-model"
        # Commas and spaces both separate values; each gets its own flag.
        scales = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--scale"]
        assert scales == ["1", "0.5", "0.5"]

    def test_output_lands_next_to_the_input_without_an_output_folder(
        self, monkeypatch, ready_env, hoct_pair, tmp_path
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None,
            device="cpu",
            _output_suffix="_tracks",
            _source_filepath=str(hoct_pair.label_path),
        )
        assert result == str(tmp_path / "movie_tracks.tif")

    def test_source_path_is_recovered_from_the_calling_frame(
        self, monkeypatch, ready_env, hoct_pair, tmp_path
    ):
        import inspect

        frame = types.SimpleNamespace(
            frame=types.SimpleNamespace(
                f_locals={"filepath": str(hoct_pair.raw_path)}
            )
        )
        monkeypatch.setattr(inspect, "stack", lambda: [frame])
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        # Called with the *raw* path, the label partner is derived from it.
        result = mod.hoct_tracking(hoct_pair.labels, device="cpu")

        assert result == str(tmp_path / "movie_hoct_tracked.tif")
        assert run.cmds[0][7] == str(hoct_pair.label_path)

    def test_stale_ctc_directory_is_removed_before_the_run(
        self, monkeypatch, ready_env, hoct_pair, tmp_path
    ):
        monkeypatch.setattr(mod, "_unique_tag", lambda: "fixed")
        stale = tmp_path / "hoct_ctc_movie_fixed"
        stale.mkdir()
        (stale / "mask999.tif").write_bytes(b"garbage")
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        # A surviving mask999.tif would be sorted last and stitched as a
        # fifth timepoint (and it is not even a readable TIFF).
        assert result is not None
        written = tifffile.imread(result)
        assert written.shape == (T, Y, X)
        assert np.array_equal(written, hoct_pair.tracked)
        assert not stale.exists()

    def test_mid_name_label_pattern_keeps_the_whole_stem(
        self, monkeypatch, ready_env, tmp_path
    ):
        rng = np.random.default_rng(12)
        labels = rng.integers(0, 4, size=(T, Y, X)).astype(np.uint16)
        tracked = rng.integers(0, 4, size=(T, Y, X)).astype(np.int64)
        label_path = _write_tif(
            tmp_path / "movie_labels_v2.tif", labels, axes="TYX"
        )
        _write_tif(tmp_path / "movie_v2.tif", labels, axes="TYX")
        run = _RunRecorder(frames=tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None,
            device="cpu",
            label_pattern="_labels",
            _source_filepath=str(label_path),
        )

        # The pattern is not a suffix here, so the extension is stripped
        # instead of the pattern and nothing is cut out of the middle.
        assert result == str(tmp_path / "movie_labels_v2_hoct_tracked.tif")
        assert run.cmds[0][6] == str(tmp_path / "movie_v2.tif")

    def test_nonzero_exit_discards_the_output(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        run = _RunRecorder(frames=hoct_pair.tracked, returncode=3)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        out = capsys.readouterr().out
        assert result is None
        assert "HOCT error (exit code 3)" in out
        assert "no output will be saved" in out
        # The CLI's own diagnostics are surfaced, not swallowed.
        assert "HOCT-STDOUT" in out
        assert "HOCT-STDERR" in out
        assert not Path(run.cmds[0][run.cmds[0].index("-o") + 1]).exists()
        # HOCT wrote usable masks; a non-zero exit must still discard them.
        assert not (
            hoct_pair.out_dir / "movie_hoct_tracked.tif"
        ).exists()

    def test_empty_ctc_directory_reports_an_assembly_failure(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        seen = {}

        def run(cmd, **kwargs):
            ctc = Path(cmd[cmd.index("-o") + 1])
            ctc.mkdir(parents=True)
            seen["ctc"] = ctc
            return _Completed(0)

        monkeypatch.setattr(mod.subprocess, "run", run)
        result = mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        out = capsys.readouterr().out
        assert result is None
        assert "Failed to assemble HOCT CTC output" in out
        assert "No CTC mask files found" in out
        assert not seen["ctc"].exists()
        assert not (hoct_pair.out_dir / "movie_hoct_tracked.tif").exists()

    def test_staged_temporary_stores_are_deleted_after_the_run(
        self, monkeypatch, ready_env, hoct_pair, tmp_path
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None,
            device="cpu",
            stage_inputs="on",
            _source_filepath=str(hoct_pair.label_path),
        )

        cmd = run.cmds[0]
        assert cmd[6].endswith(".zarr")  # raw staged
        assert cmd[7].endswith(".zarr")  # labels staged
        assert result is not None
        assert list(tmp_path.glob("*.zarr")) == []

    def test_gurobi_license_is_exported_to_the_subprocess(
        self, monkeypatch, ready_env, hoct_pair, tmp_path, capsys
    ):
        lic = tmp_path / "site.lic"
        lic.write_text("KEY")
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        mod.hoct_tracking(
            None,
            device="cpu",
            gurobi_license=str(lic),
            _source_filepath=str(hoct_pair.label_path),
        )
        assert run.envs[0]["GRB_LICENSE_FILE"] == str(lic)
        assert "Using Gurobi license" in capsys.readouterr().out

    def test_missing_gurobi_license_only_warns(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        assert "GRB_LICENSE_FILE" not in run.envs[0]
        assert "No Gurobi license file found" in capsys.readouterr().out

    def test_invalid_device_and_tile_reach_the_cli_normalised(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        result = mod.hoct_tracking(
            None,
            device="tpu",
            tile="sometimes",
            gpus="none",  # keeps the fallback to 'cuda' from pinning a card
            _source_filepath=str(hoct_pair.label_path),
        )
        out = capsys.readouterr().out
        assert "invalid device 'tpu'" in out
        assert "invalid tile mode 'sometimes'" in out
        # Warning without substitution would hand HOCT an unusable "-d tpu".
        cmd = run.cmds[0]
        assert cmd[cmd.index("-d") + 1] == "cuda"
        assert cmd[cmd.index("--tile") + 1] == "auto"
        assert result is not None

    def test_cpu_device_never_pins_a_gpu(
        self, monkeypatch, ready_env, hoct_pair
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        mod.hoct_tracking(
            None, device="cpu", _source_filepath=str(hoct_pair.label_path)
        )
        assert "CUDA_VISIBLE_DEVICES" not in run.envs[0]
        assert mod._GPU_POOL is None  # the pool is never even built

    def test_cuda_run_pins_one_gpu_and_returns_it_to_the_pool(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        mod.hoct_tracking(
            None,
            device="cuda",
            gpus="0,1",
            _source_filepath=str(hoct_pair.label_path),
        )
        out = capsys.readouterr().out
        assert run.envs[0]["CUDA_VISIBLE_DEVICES"] == "0"
        assert "Running HOCT tracking on GPU 0" in out
        assert "Released GPU 0" in out
        # Both cards are available again for the next file in the batch.
        assert mod._GPU_POOL.qsize() == 2

    def test_cuda_without_detected_gpus_runs_unpinned(
        self, monkeypatch, ready_env, hoct_pair, capsys
    ):
        run = _RunRecorder(frames=hoct_pair.tracked)
        monkeypatch.setattr(mod.subprocess, "run", run)

        mod.hoct_tracking(
            None,
            device="cuda",
            gpus="none",
            _source_filepath=str(hoct_pair.label_path),
        )
        assert "CUDA_VISIBLE_DEVICES" not in run.envs[0]
        assert "on device 'cuda'" in capsys.readouterr().out


class TestRegistration:
    """The entry point the batch widget actually discovers and calls."""

    def test_registered_under_its_display_name(self):
        info = BatchProcessingRegistry.get_function_info(
            "Track Cells with HOCT"
        )
        assert info is not None
        assert info["func"] is mod.hoct_tracking
        assert info["suffix"] == "_hoct_tracked"

    def test_registered_parameters_match_the_signature_defaults(self):
        import inspect

        info = BatchProcessingRegistry.get_function_info(
            "Track Cells with HOCT"
        )
        signature = inspect.signature(mod.hoct_tracking)
        for name, spec in info["parameters"].items():
            assert name in signature.parameters
            assert signature.parameters[name].default == spec["default"]
        # ...and nothing user-facing is missing from the widget: an
        # unregistered parameter is silently unreachable from the UI.
        public = {
            name
            for name in signature.parameters
            if name != "image" and not name.startswith("_")
        }
        # "dimension_order" is deliberately unregistered: it is supplied by
        # the batch widget's global "Dimension Order" dropdown, so declaring
        # it here too would ask the user for the same thing twice.
        assert public - {"dimension_order"} == set(info["parameters"])
        assert "dimension_order" not in info["parameters"]

    def test_registered_option_lists_contain_their_defaults(self):
        info = BatchProcessingRegistry.get_function_info(
            "Track Cells with HOCT"
        )
        with_options = {
            name: spec
            for name, spec in info["parameters"].items()
            if "options" in spec
        }
        assert set(with_options) == {
            "device",
            "tile",
            "stage_inputs",
        }
        for name, spec in with_options.items():
            assert spec["default"] in spec["options"], name

    def test_worker_hints_keep_the_volume_out_of_ram(self):
        # The subprocess reads the file itself, so the widget must not
        # materialise the whole TZYX array before calling in.
        assert mod.hoct_tracking.skip_load is True
        assert mod.hoct_tracking._loads_from_path is True
        assert mod.hoct_tracking.supports_gpu_distribution is True

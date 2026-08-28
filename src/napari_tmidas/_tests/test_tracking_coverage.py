"""Coverage-focused tests for the two subprocess-driven tracking modules.

Neither ``trackastra`` nor ``ultrack`` is installed in this environment, so
every test here exercises the *orchestration* layer that lives in the main
env: conda environment bootstrap/repair, raw/label file pairing, GPU
pinning, the subprocess invocation and its failure handling, and result
collection.  Heavy collaborators (``subprocess``, ``shutil.which``, the
ultrack env-manager helpers, ``imread``) are stubbed on the module object so
nothing ever shells out for real.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import zarr

from napari_tmidas.processing_functions import trackastra_tracking as ta
from napari_tmidas.processing_functions import ultrack_tracking as ut


class _FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def gpu_pool_reset():
    """Save/restore the process-wide Trackastra GPU pool globals."""
    saved = (
        ta._GPU_POOL,
        ta._GPU_IDS,
        ta._GPU_POOL_WORKERS_PER_GPU,
        ta._GPU_POOL_KEY,
    )
    yield
    (
        ta._GPU_POOL,
        ta._GPU_IDS,
        ta._GPU_POOL_WORKERS_PER_GPU,
        ta._GPU_POOL_KEY,
    ) = saved


@pytest.fixture()
def no_gpu_env(monkeypatch):
    """Remove the env vars ``_detect_gpu_ids`` consults."""
    monkeypatch.delenv("TRACKASTRA_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


# ---------------------------------------------------------------------------
# trackastra: GPU detection / pool
# ---------------------------------------------------------------------------


class TestDetectGpuIds:
    """Pin the precedence order of the Trackastra GPU-id resolution."""

    @pytest.mark.parametrize("override", ["none", "cpu", "NONE", " CPU "])
    def test_explicit_override_disables_pinning(self, override, no_gpu_env):
        """'none'/'cpu' (any case) means: do not pin a device at all."""
        assert ta._detect_gpu_ids(override) == []

    def test_explicit_override_is_split_and_stripped(self, no_gpu_env):
        """A comma list wins over every env var and is whitespace-trimmed."""
        assert ta._detect_gpu_ids(" 0 , 1 ,, 3 ") == ["0", "1", "3"]

    def test_env_var_override_disables_pinning(self, monkeypatch):
        """TRACKASTRA_GPUS=none short-circuits detection."""
        monkeypatch.setenv("TRACKASTRA_GPUS", "none")
        assert ta._detect_gpu_ids() == []

    def test_env_var_override_lists_ids(self, monkeypatch):
        """TRACKASTRA_GPUS is parsed like the explicit override."""
        monkeypatch.setenv("TRACKASTRA_GPUS", "2,5")
        assert ta._detect_gpu_ids() == ["2", "5"]

    def test_cuda_visible_devices_is_honoured(self, monkeypatch):
        """An already-set CUDA_VISIBLE_DEVICES is reused verbatim."""
        monkeypatch.delenv("TRACKASTRA_GPUS", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 2")
        assert ta._detect_gpu_ids() == ["1", "2"]

    def test_nvidia_smi_lines_are_counted(self, monkeypatch, no_gpu_env):
        """With no overrides, `nvidia-smi -L` output is counted."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeCompleted(
                0, "GPU 0: NVIDIA A100\nGPU 1: NVIDIA A100\nnoise\n"
            )

        monkeypatch.setattr(ta.subprocess, "run", fake_run)
        assert ta._detect_gpu_ids() == ["0", "1"]
        assert calls == [["nvidia-smi", "-L"]]

    def test_nvidia_smi_nonzero_exit_means_no_gpus(
        self, monkeypatch, no_gpu_env
    ):
        """A failing nvidia-smi yields an empty id list, not an error."""
        monkeypatch.setattr(
            ta.subprocess, "run", lambda *a, **k: _FakeCompleted(1, "")
        )
        assert ta._detect_gpu_ids() == []

    def test_nvidia_smi_zero_gpus_means_no_gpus(self, monkeypatch, no_gpu_env):
        """rc==0 but no 'GPU ' lines still means no pinning."""
        monkeypatch.setattr(
            ta.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "nothing")
        )
        assert ta._detect_gpu_ids() == []

    def test_missing_nvidia_smi_is_swallowed(self, monkeypatch, no_gpu_env):
        """OSError from a missing binary must not propagate."""

        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(ta.subprocess, "run", boom)
        assert ta._detect_gpu_ids() == []

    def test_subprocess_error_is_swallowed(self, monkeypatch, no_gpu_env):
        """subprocess.SubprocessError from a timeout is swallowed too."""

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(["nvidia-smi"], 10)

        monkeypatch.setattr(ta.subprocess, "run", boom)
        assert ta._detect_gpu_ids() == []


class TestGpuPool:
    """The shared pool is rebuilt whenever the requested shape changes."""

    def test_pool_repeats_each_gpu_per_worker(self, gpu_pool_reset):
        """workers_per_gpu copies of every id are enqueued."""
        pool, ids = ta._get_gpu_pool(workers_per_gpu=3, gpus_override="0,1")
        assert ids == ["0", "1"]
        assert pool.qsize() == 6
        drained = sorted(pool.get() for _ in range(6))
        assert drained == ["0", "0", "0", "1", "1", "1"]

    def test_pool_is_cached_for_the_same_key(self, gpu_pool_reset):
        """Repeat calls with the same key return the identical queue."""
        first, _ = ta._get_gpu_pool(1, "0")
        second, _ = ta._get_gpu_pool(1, "0")
        assert first is second

    def test_pool_is_rebuilt_when_key_changes(self, gpu_pool_reset):
        """A different override rebuilds the pool with new ids."""
        first, first_ids = ta._get_gpu_pool(1, "0")
        second, second_ids = ta._get_gpu_pool(1, "cpu")
        assert first is not second
        assert first_ids == ["0"]
        assert second_ids == []
        assert second.qsize() == 0

    def test_workers_per_gpu_is_clamped_to_one(self, gpu_pool_reset):
        """Zero/negative worker counts still yield one slot per GPU."""
        pool, ids = ta._get_gpu_pool(workers_per_gpu=0, gpus_override="7")
        assert ids == ["7"]
        assert pool.qsize() == 1
        assert ta._GPU_POOL_WORKERS_PER_GPU == 1


# ---------------------------------------------------------------------------
# trackastra: filename helpers
# ---------------------------------------------------------------------------


class TestRawCandidates:
    """Raw/label basename pairing, including the non-suffix pattern."""

    def test_pattern_in_the_middle_uses_replace(self):
        """A pattern that is not a trailing suffix is replaced once."""
        raw_base, candidates = ta._raw_candidates_from_label_name(
            "movie_labels.tif", "_labels"
        )
        assert raw_base == "movie.tif"
        # base already carries a known image suffix -> single candidate
        assert candidates == ["movie.tif"]

    def test_trailing_pattern_expands_to_all_suffixes(self):
        """Stripping a trailing pattern leaves a bare stem to expand."""
        raw_base, candidates = ta._raw_candidates_from_label_name(
            "movie_labels.tif", "_labels.tif"
        )
        assert raw_base == "movie"
        assert candidates == ["movie.tif", "movie.tiff", "movie.zarr"]

    def test_no_match_returns_none_path(self, tmp_path):
        """When nothing on disk matches, the resolved path is None."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        raw_base, candidates, found = ta._find_matching_raw_path(
            str(label), "_labels.tif"
        )
        assert raw_base == "movie"
        assert found is None
        assert len(candidates) == 3


# ---------------------------------------------------------------------------
# trackastra: gurobi license resolution
# ---------------------------------------------------------------------------


class TestTrackastraGurobiLicense:
    """First-hit-wins resolution of GRB_LICENSE_FILE."""

    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        """An existing explicit path is returned untouched."""
        lic = tmp_path / "mine.lic"
        lic.write_text("KEY", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
        assert ta._resolve_gurobi_license(str(lic)) == str(lic)

    def test_missing_explicit_path_falls_back(self, tmp_path, monkeypatch):
        """A bogus explicit path warns and falls through to auto-detect."""
        monkeypatch.setenv("HOME", str(tmp_path))
        env_lic = tmp_path / "env.lic"
        env_lic.write_text("KEY", encoding="utf-8")
        monkeypatch.setenv("GRB_LICENSE_FILE", str(env_lic))
        assert ta._resolve_gurobi_license("/does/not/exist.lic") == str(
            env_lic
        )

    def test_home_license_is_the_last_resort(self, tmp_path, monkeypatch):
        """~/gurobi.lic is used when nothing else is configured."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
        home_lic = tmp_path / "gurobi.lic"
        home_lic.write_text("KEY", encoding="utf-8")
        assert ta._resolve_gurobi_license("") == str(home_lic)

    def test_nothing_found_returns_none(self, tmp_path, monkeypatch):
        """No license anywhere leaves the environment untouched."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
        assert ta._resolve_gurobi_license("") is None


# ---------------------------------------------------------------------------
# trackastra: zarr loading
# ---------------------------------------------------------------------------


class TestLoadZarrArray:
    """``_load_zarr_array`` unwraps arrays, groups and multiscale roots."""

    def test_plain_array_root(self, tmp_path):
        """A root-level zarr array is converted straight to NumPy."""
        path = tmp_path / "a.zarr"
        arr = zarr.open(str(path), mode="w", shape=(2, 3), dtype="uint8")
        arr[:] = 5
        out = ta._load_zarr_array(str(path))
        assert out.shape == (2, 3)
        assert out.dtype == np.uint8
        assert np.all(out == 5)

    def test_group_prefers_level_zero(self, tmp_path):
        """A group holding '0' resolves to that array."""
        path = tmp_path / "g.zarr"
        grp = zarr.open_group(store=str(path), mode="w")
        zero = grp.create_array("0", shape=(2, 2), dtype="uint16")
        zero[:] = 9
        one = grp.create_array("1", shape=(1, 1), dtype="uint16")
        one[:] = 1
        out = ta._load_zarr_array(str(path))
        assert out.shape == (2, 2)
        assert np.all(out == 9)

    def test_group_without_level_zero_takes_first_array(self, tmp_path):
        """Any other single array in the group is accepted."""
        path = tmp_path / "g2.zarr"
        grp = zarr.open_group(store=str(path), mode="w")
        only = grp.create_array("lvl", shape=(3,), dtype="uint8")
        only[:] = 2
        out = ta._load_zarr_array(str(path))
        assert out.tolist() == [2, 2, 2]

    def test_empty_group_raises(self, tmp_path):
        """A group with no arrays is an explicit ValueError."""
        path = tmp_path / "empty.zarr"
        zarr.open_group(store=str(path), mode="w")
        with pytest.raises(ValueError, match="No arrays found"):
            ta._load_zarr_array(str(path))

    def test_unopenable_root_falls_back_to_level_zero(self, tmp_path):
        """A bare directory containing a '0' array still loads."""
        root = tmp_path / "multiscale.zarr"
        root.mkdir()
        level0 = zarr.open(
            str(root / "0"), mode="w", shape=(4,), dtype="uint8"
        )
        level0[:] = 3
        out = ta._load_zarr_array(str(root))
        assert out.tolist() == [3, 3, 3, 3]

    def test_missing_path_reraises(self, tmp_path):
        """No root and no '0' subdirectory -> the original error escapes."""
        missing = tmp_path / "nope.zarr"
        with pytest.raises(FileNotFoundError) as excinfo:
            ta._load_zarr_array(str(missing))
        # the message must name the path we asked for, so the test cannot be
        # satisfied by an unrelated FileNotFoundError from elsewhere
        assert "nope.zarr" in str(excinfo.value)
        # the failed fallback must not have created the store on the way out
        assert not missing.exists()


# ---------------------------------------------------------------------------
# trackastra: TrackAstraEnvManager
# ---------------------------------------------------------------------------


class TestTrackAstraCondaCmd:
    """`get_conda_cmd` prefers mamba and fails loudly when neither exists."""

    def test_mamba_preferred(self, monkeypatch):
        """mamba wins when both are on PATH."""
        monkeypatch.setattr(ta.shutil, "which", lambda name: "/bin/" + name)
        assert ta.TrackAstraEnvManager.get_conda_cmd() == "mamba"

    def test_conda_fallback(self, monkeypatch):
        """conda is used when mamba is absent."""
        monkeypatch.setattr(
            ta.shutil,
            "which",
            lambda name: "/bin/conda" if name == "conda" else None,
        )
        assert ta.TrackAstraEnvManager.get_conda_cmd() == "conda"

    def test_neither_raises_runtime_error(self, monkeypatch):
        """Missing both is a RuntimeError telling the user to install one."""
        monkeypatch.setattr(ta.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="Neither conda nor mamba"):
            ta.TrackAstraEnvManager.get_conda_cmd()


class TestTrackAstraCheckEnvExists:
    """`check_env_exists` maps `conda run python --version` onto a bool."""

    def test_returns_true_on_zero_exit(self, monkeypatch):
        """rc == 0 means the env exists."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["timeout"] = kwargs.get("timeout")
            return _FakeCompleted(0, "Python 3.11.9")

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(ta.subprocess, "run", fake_run)
        assert ta.TrackAstraEnvManager.check_env_exists() is True
        assert seen["cmd"][:5] == [
            "mamba",
            "run",
            "-n",
            "trackastra",
            "python",
        ]
        assert seen["timeout"] == 10

    def test_returns_false_on_nonzero_exit(self, monkeypatch):
        """rc != 0 means the env is missing."""
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(
            ta.subprocess, "run", lambda *a, **k: _FakeCompleted(1)
        )
        assert ta.TrackAstraEnvManager.check_env_exists() is False

    def test_timeout_is_treated_as_missing(self, monkeypatch):
        """A hung conda call is reported as 'env not there'."""

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(["conda"], 10)

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(ta.subprocess, "run", boom)
        assert ta.TrackAstraEnvManager.check_env_exists() is False


class TestTrackAstraVersionHelpers:
    """Version parsing/comparison corner cases."""

    def test_version_tuple_of_none_is_empty(self):
        """None has no numeric components."""
        assert ta.TrackAstraEnvManager._version_tuple(None) == ()

    def test_version_tuple_without_digits_is_empty(self):
        """A digit-free string yields an empty tuple."""
        assert ta.TrackAstraEnvManager._version_tuple("unknown") == ()

    def test_version_tuple_extracts_all_numbers(self):
        """Digits are extracted in order, separators ignored."""
        assert ta.TrackAstraEnvManager._version_tuple("0.5.1rc2") == (
            0,
            5,
            1,
            2,
        )

    def test_unparseable_version_is_never_at_least(self):
        """A missing/garbage version can never satisfy a requirement."""
        assert ta.TrackAstraEnvManager._version_at_least(None, "3.11") is False
        assert ta.TrackAstraEnvManager._version_at_least("3.11", None) is False

    def test_short_version_is_zero_padded(self):
        """'3' is compared as (3, 0) against a two-part requirement."""
        assert ta.TrackAstraEnvManager._version_at_least("3", "3.11") is False
        assert ta.TrackAstraEnvManager._version_at_least("4", "3.11") is True

    def test_extra_precision_is_ignored(self):
        """Comparison is truncated to the required precision."""
        assert (
            ta.TrackAstraEnvManager._version_at_least("3.11.9", "3.11") is True
        )


class TestTrackAstraGetEnvStatus:
    """`get_env_status` shells into the env and parses its JSON report."""

    def test_parses_json_payload(self, monkeypatch):
        """Valid JSON on stdout is returned as a dict."""
        payload = {
            "python": "3.11.9",
            "packages": {"zarr": {"present": True, "version": "3.0.1"}},
        }
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["check"] = kwargs.get("check")
            return _FakeCompleted(0, json.dumps(payload) + "\n")

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(ta.subprocess, "run", fake_run)
        assert ta.TrackAstraEnvManager.get_env_status() == payload
        assert seen["check"] is True
        assert seen["cmd"][:4] == ["mamba", "run", "-n", "trackastra"]

    def test_failure_is_reported_as_error_dict(self, monkeypatch):
        """Any exception becomes {'error': ...} instead of propagating."""

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, ["mamba"])

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(ta.subprocess, "run", boom)
        status = ta.TrackAstraEnvManager.get_env_status()
        assert list(status) == ["error"]
        assert "mamba" in status["error"]

    def test_invalid_json_is_reported_as_error_dict(self, monkeypatch):
        """Garbage stdout also lands in the error dict."""
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(
            ta.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "not json")
        )
        assert "error" in ta.TrackAstraEnvManager.get_env_status()


class TestTrackAstraRepairEnv:
    """`repair_env` runs clean + solver install + pip upgrade."""

    def test_runs_the_three_expected_commands(self, monkeypatch):
        """Order: cache clean (check=False), ilpy/gurobi, pip upgrade."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), kwargs.get("check")))
            return _FakeCompleted(0)

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(ta.subprocess, "run", fake_run)
        assert ta.TrackAstraEnvManager.repair_env() is True
        assert len(calls) == 3
        assert calls[0][0][1] == "clean"
        assert calls[0][1] is False
        assert "ilpy" in calls[1][0]
        assert calls[1][1] is True
        assert "trackastra[ilp]" in calls[2][0]
        assert "motile==0.4.0" in calls[2][0]

    def test_install_failure_returns_false(self, monkeypatch):
        """A failing install is caught and reported as False."""

        def fake_run(cmd, **kwargs):
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, list(cmd))
            return _FakeCompleted(0)

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(ta.subprocess, "run", fake_run)
        assert ta.TrackAstraEnvManager.repair_env() is False


class TestTrackAstraCreateEnv:
    """`create_env` is a no-op when the env exists, else builds it."""

    def test_existing_env_short_circuits(self, monkeypatch):
        """No subprocess is spawned when the env is already there."""

        def explode(*a, **k):
            raise AssertionError("should not shell out")

        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "check_env_exists", lambda: True
        )
        monkeypatch.setattr(ta.subprocess, "run", explode)
        assert ta.TrackAstraEnvManager.create_env() is True

    def test_conda_gets_no_default_packages_flag(self, monkeypatch):
        """`--no-default-packages` is conda-only; mamba rejects it."""
        calls = []
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "conda"
        )
        monkeypatch.setattr(
            ta.subprocess,
            "run",
            lambda cmd, **k: calls.append(list(cmd)) or _FakeCompleted(0),
        )
        assert ta.TrackAstraEnvManager.create_env() is True
        create_cmd = calls[1]
        assert create_cmd[:4] == ["conda", "create", "-n", "trackastra"]
        assert "--no-default-packages" in create_cmd
        assert create_cmd[-1] == "-y"
        assert "python=3.11" in create_cmd

    def test_mamba_omits_no_default_packages_flag(self, monkeypatch):
        """mamba builds the same env without the conda-only flag."""
        calls = []
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )
        monkeypatch.setattr(
            ta.subprocess,
            "run",
            lambda cmd, **k: calls.append(list(cmd)) or _FakeCompleted(0),
        )
        assert ta.TrackAstraEnvManager.create_env() is True
        assert len(calls) == 4
        assert "--no-default-packages" not in calls[1]
        assert "trackastra[ilp]" in calls[3]
        assert "zarr>=3" in calls[3]

    def test_failure_returns_false(self, monkeypatch):
        """A CalledProcessError during creation is caught."""
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "check_env_exists", lambda: False
        )
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
        )

        def fake_run(cmd, **kwargs):
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(2, list(cmd))
            return _FakeCompleted(0)

        monkeypatch.setattr(ta.subprocess, "run", fake_run)
        assert ta.TrackAstraEnvManager.create_env() is False


_HEALTHY_STATUS = {
    "python": "3.11.9",
    "packages": {
        "gurobipy": {"present": True, "version": "13.0.0"},
        "ilpy": {"present": True, "version": "0.5.1"},
        "motile": {"present": True, "version": "0.4.0"},
        "trackastra": {"present": True, "version": "0.5.3"},
        "zarr": {"present": True, "version": "3.0.1"},
    },
}


class TestTrackAstraEnvNeedsRepair:
    """`env_needs_repair` accumulates one reason per drifted package."""

    def test_missing_status_needs_repair(self):
        """An empty/failed status is unconditionally unhealthy."""
        needs, reasons = ta.TrackAstraEnvManager.env_needs_repair({})
        assert needs is True
        assert reasons == ["Could not determine environment status"]
        needs, reasons = ta.TrackAstraEnvManager.env_needs_repair(
            {"error": "boom"}
        )
        assert needs is True

    def test_healthy_status_needs_no_repair(self):
        """A fully pinned env reports no reasons."""
        needs, reasons = ta.TrackAstraEnvManager.env_needs_repair(
            _HEALTHY_STATUS
        )
        assert needs is False
        assert reasons == []

    def test_old_python_is_a_reason(self):
        """A too-old interpreter is called out explicitly."""
        status = json.loads(json.dumps(_HEALTHY_STATUS))
        status["python"] = "3.9.18"
        needs, reasons = ta.TrackAstraEnvManager.env_needs_repair(status)
        assert needs is True
        assert any(r.startswith("Python 3.9.18") for r in reasons)

    def test_outdated_package_is_a_reason(self):
        """A below-minimum package version is reported."""
        status = json.loads(json.dumps(_HEALTHY_STATUS))
        status["packages"]["trackastra"]["version"] = "0.4.0"
        needs, reasons = ta.TrackAstraEnvManager.env_needs_repair(status)
        assert needs is True
        assert any("trackastra version 0.4.0" in r for r in reasons)


class TestTrackAstraEnsureEnvReady:
    """`ensure_env_ready` is the create -> inspect -> repair state machine."""

    def _patch(self, monkeypatch, **attrs):
        for name, value in attrs.items():
            monkeypatch.setattr(ta.TrackAstraEnvManager, name, value)

    def test_create_failure_aborts(self, monkeypatch):
        """If the env is missing and cannot be created, give up."""
        self._patch(
            monkeypatch,
            check_env_exists=lambda: False,
            create_env=lambda: False,
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is False

    def test_healthy_env_is_ready(self, monkeypatch):
        """An existing, healthy env needs neither repair nor rebuild."""
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_env_status=lambda: _HEALTHY_STATUS,
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is True

    def test_package_drift_triggers_repair_only(self, monkeypatch):
        """Package drift repairs in place; the interpreter is left alone."""
        drifted = json.loads(json.dumps(_HEALTHY_STATUS))
        drifted["packages"]["zarr"]["present"] = False
        statuses = [drifted, _HEALTHY_STATUS]
        calls = []
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_env_status=lambda: statuses.pop(0),
            repair_env=lambda: calls.append("repair") or True,
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is True
        assert calls == ["repair"]

    def test_repair_failure_aborts(self, monkeypatch):
        """A failed repair short-circuits before the recheck."""
        drifted = json.loads(json.dumps(_HEALTHY_STATUS))
        drifted["packages"]["motile"]["version"] = "1.1.0"
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_env_status=lambda: drifted,
            repair_env=lambda: False,
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is False

    def test_still_unhealthy_after_repair_aborts(self, monkeypatch):
        """A repair that does not fix the drift reports failure."""
        drifted = json.loads(json.dumps(_HEALTHY_STATUS))
        drifted["packages"]["ilpy"]["present"] = False
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_env_status=lambda: drifted,
            repair_env=lambda: True,
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is False

    def test_python_drift_removes_and_recreates(self, monkeypatch):
        """A wrong interpreter forces `conda env remove` + create."""
        drifted = json.loads(json.dumps(_HEALTHY_STATUS))
        drifted["python"] = "3.9.18"
        statuses = [drifted, _HEALTHY_STATUS]
        removed = []
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_conda_cmd=lambda: "mamba",
            get_env_status=lambda: statuses.pop(0),
            create_env=lambda: True,
        )
        monkeypatch.setattr(
            ta.subprocess,
            "run",
            lambda cmd, **k: removed.append(list(cmd)) or _FakeCompleted(0),
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is True
        assert removed == [
            ["mamba", "env", "remove", "-n", "trackastra", "-y"]
        ]

    def test_env_remove_failure_aborts(self, monkeypatch):
        """If the old env cannot be removed, bail out."""
        drifted = json.loads(json.dumps(_HEALTHY_STATUS))
        drifted["python"] = "3.9.18"
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_conda_cmd=lambda: "mamba",
            get_env_status=lambda: drifted,
        )

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, ["mamba", "env", "remove"])

        monkeypatch.setattr(ta.subprocess, "run", boom)
        assert ta.TrackAstraEnvManager.ensure_env_ready() is False

    def test_recreate_failure_aborts(self, monkeypatch):
        """Removal succeeds but recreation fails -> False."""
        drifted = json.loads(json.dumps(_HEALTHY_STATUS))
        drifted["python"] = "3.9.18"
        self._patch(
            monkeypatch,
            check_env_exists=lambda: True,
            get_conda_cmd=lambda: "mamba",
            get_env_status=lambda: drifted,
            create_env=lambda: False,
        )
        monkeypatch.setattr(
            ta.subprocess, "run", lambda *a, **k: _FakeCompleted(0)
        )
        assert ta.TrackAstraEnvManager.ensure_env_ready() is False


# ---------------------------------------------------------------------------
# trackastra: the registered processing function
# ---------------------------------------------------------------------------


def _stub_trackastra(monkeypatch, *, returncode=0, produce=True):
    """Stub env bootstrap, script generation and the subprocess call.

    Returns a recorder dict capturing what the real module handed to its
    collaborators.
    """
    rec = {"script_args": None, "cmd": None, "env": {}, "runs": 0}

    monkeypatch.setattr(
        ta.TrackAstraEnvManager, "ensure_env_ready", lambda: True
    )
    monkeypatch.setattr(
        ta.TrackAstraEnvManager, "get_conda_cmd", lambda: "mamba"
    )

    def fake_script(*args):
        rec["script_args"] = args
        return "print('stub')\n"

    monkeypatch.setattr(ta, "create_trackastra_script", fake_script)

    def fake_run(cmd, **kwargs):
        rec["runs"] += 1
        rec["cmd"] = list(cmd)
        rec["env"] = dict(kwargs.get("env") or {})
        if produce:
            Path(rec["script_args"][4]).write_text("tracked", encoding="utf-8")
        return _FakeCompleted(returncode, "SUBPROC-OUT", "SUBPROC-ERR")

    monkeypatch.setattr(ta.subprocess, "run", fake_run)
    return rec


def _call_trackastra_through_worker_frame(filepath, **kwargs):
    """Emulate the worker frame that exposes a local named ``filepath``."""
    return ta.trackastra_tracking(None, **kwargs)


class TestTrackastraInputValidation:
    """Cheap in-memory guards run before any environment work."""

    def test_non_time_series_returns_unchanged(self):
        """A 2D array is not a time series; nothing is spawned."""
        image = np.zeros((4, 4), dtype=np.uint16)
        assert ta.trackastra_tracking(image) is image

    def test_single_timepoint_returns_unchanged(self):
        """One timepoint cannot be tracked."""
        image = np.zeros((1, 4, 4), dtype=np.uint16)
        assert ta.trackastra_tracking(image) is image

    def test_env_not_ready_returns_unchanged(self, monkeypatch):
        """A broken conda env aborts before touching the filesystem."""
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "ensure_env_ready", lambda: False
        )
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        assert ta.trackastra_tracking(image) is image


class TestTrackastraPathPairing:
    """Raw/label pairing and the derived output filename."""

    def test_raw_input_without_label_returns_unchanged(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """A raw image with no sibling label file is skipped."""
        raw = tmp_path / "movie.tif"
        raw.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)

        out = ta.trackastra_tracking(
            image,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(raw),
        )
        assert out is image
        assert rec["runs"] == 0

    def test_raw_input_pairs_with_sibling_label(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """A raw image is paired with ``<stem><label_pattern>``."""
        raw = tmp_path / "movie.tif"
        raw.write_text("", encoding="utf-8")
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        out = ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(raw),
        )
        raw_arg, mask_arg, model, mode, output_arg = rec["script_args"][:5]
        assert raw_arg == str(raw)
        assert mask_arg == str(label)
        assert model == "ctc"
        assert mode == "greedy"
        assert Path(output_arg).name == "movie_tracked.tif"
        assert out == str(tmp_path / "movie_tracked.tif")

    def test_label_input_without_raw_falls_back_to_itself(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """With no raw partner the label file is used as the image too."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        raw_arg, mask_arg = rec["script_args"][:2]
        assert raw_arg == str(label)
        assert mask_arg == str(label)

    def test_infix_label_pattern_uses_splitext_output_name(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """A pattern that is not a trailing suffix keeps the full stem."""
        raw = tmp_path / "movie.tif"
        raw.write_text("", encoding="utf-8")
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        out = ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels",
            _source_filepath=str(label),
        )
        raw_arg, mask_arg, _, _, output_arg = rec["script_args"][:5]
        # '_labels' is an infix here, so the raw candidate is 'movie.tif'.
        assert raw_arg == str(raw)
        assert mask_arg == str(label)
        assert Path(output_arg).name == "movie_labels_tracked.tif"
        assert out == str(tmp_path / "movie_labels_tracked.tif")

    def test_output_folder_overrides_input_directory(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """``_output_folder`` redirects the produced tif."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        out = ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
            _output_folder=str(out_dir),
            _output_suffix="_trk",
        )
        assert out == str(out_dir / "movie_trk.tif")
        assert Path(rec["script_args"][4]).parent == out_dir

    def test_filepath_is_recovered_from_the_caller_frame(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """Without ``_source_filepath`` the stack is scanned for ``filepath``."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        out = _call_trackastra_through_worker_frame(
            str(label),
            gpus="cpu",
            label_pattern="_labels.tif",
        )
        assert rec["script_args"][1] == str(label)
        assert out == str(tmp_path / "movie_tracked.tif")

    def test_undeterminable_filepath_returns_the_image_unchanged(
        self, monkeypatch, capsys
    ):
        """The "cannot determine path" branch must actually return.

        It used to log "Could not determine input file path. Returning
        unchanged." and then fall through to
        ``Path(os.path.dirname(img_path))`` with ``img_path is None``,
        raising TypeError instead of honouring what it just printed.
        """
        monkeypatch.setattr(
            ta.TrackAstraEnvManager, "ensure_env_ready", lambda: True
        )

        def no_shell_out(*a, **k):
            raise AssertionError("must not spawn a subprocess")

        monkeypatch.setattr(ta.subprocess, "run", no_shell_out)
        image = np.zeros((2, 4, 4), dtype=np.uint16)

        result = ta.trackastra_tracking(image, gpus="cpu")

        assert result is image
        assert (
            "Could not determine input file path" in capsys.readouterr().out
        )


class TestTrackastraSubprocess:
    """Command construction, GPU pinning and exit-code handling."""

    def test_command_targets_the_trackastra_env(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """The generated script is run via ``<conda> run -n trackastra``."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert rec["cmd"][:5] == [
            "mamba",
            "run",
            "-n",
            "trackastra",
            "python",
        ]
        script_name = Path(rec["cmd"][5]).name
        assert script_name.startswith("run_tracking_movie_labels_")
        assert script_name.endswith(f"_{os.getpid()}.py")

    def test_script_is_removed_after_success(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """The per-file runner script does not linger in the input folder."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert list(tmp_path.glob("run_tracking_*.py")) == []

    def test_nonzero_exit_returns_input_and_cleans_up(
        self, tmp_path, monkeypatch, gpu_pool_reset, capsys
    ):
        """A failed subprocess logs both streams and returns the input."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        _stub_trackastra(monkeypatch, returncode=3, produce=False)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)
        image = np.zeros((3, 4, 4), dtype=np.uint16)

        out = ta.trackastra_tracking(
            image,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert out is image
        printed = capsys.readouterr().out
        assert "TrackAstra error:" in printed
        assert "SUBPROC-OUT" in printed
        assert "SUBPROC-ERR" in printed
        assert list(tmp_path.glob("run_tracking_*.py")) == []

    def test_missing_output_returns_input_and_cleans_up(
        self, tmp_path, monkeypatch, gpu_pool_reset, capsys
    ):
        """rc == 0 but no output file still returns the input unchanged."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        _stub_trackastra(monkeypatch, returncode=0, produce=False)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)
        image = np.zeros((3, 4, 4), dtype=np.uint16)

        out = ta.trackastra_tracking(
            image,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert out is image
        assert "did not produce output" in capsys.readouterr().out
        assert list(tmp_path.glob("run_tracking_*.py")) == []

    def test_gpu_is_pinned_and_returned_to_the_pool(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """The acquired GPU id is exported and released in the finally."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        ta.trackastra_tracking(
            None,
            gpus="2",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert rec["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        # released again, so the next file in the batch can take it
        assert ta._GPU_POOL.qsize() == 1
        assert ta._GPU_POOL.get() == "2"

    def test_gpu_is_released_even_when_the_subprocess_raises(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """The pool must not leak a slot when the run blows up."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        def boom(*a, **k):
            raise OSError("conda vanished")

        monkeypatch.setattr(ta.subprocess, "run", boom)
        with pytest.raises(OSError, match="conda vanished"):
            ta.trackastra_tracking(
                None,
                gpus="4",
                label_pattern="_labels.tif",
                _source_filepath=str(label),
            )
        assert ta._GPU_POOL.qsize() == 1

    def test_cpu_mode_leaves_cuda_visible_devices_alone(
        self, tmp_path, monkeypatch, gpu_pool_reset
    ):
        """gpus='cpu' means no pinning is injected by this module."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        ta.trackastra_tracking(
            None,
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert "CUDA_VISIBLE_DEVICES" not in rec["env"]

    def test_license_is_exported_to_the_subprocess(
        self, tmp_path, monkeypatch, gpu_pool_reset, capsys
    ):
        """A resolved .lic overrides the bundled size-limited pip license."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(
            ta, "_resolve_gurobi_license", lambda _x: "/lic/gurobi.lic"
        )

        ta.trackastra_tracking(
            None,
            mode="ilp",
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert rec["env"]["GRB_LICENSE_FILE"] == "/lic/gurobi.lic"
        assert (
            "Using Gurobi license: /lic/gurobi.lic" in capsys.readouterr().out
        )

    def test_ilp_without_license_warns(
        self, tmp_path, monkeypatch, gpu_pool_reset, capsys
    ):
        """ilp mode without a license warns about the bundled cap."""
        label = tmp_path / "movie_labels.tif"
        label.write_text("", encoding="utf-8")
        rec = _stub_trackastra(monkeypatch)
        monkeypatch.setattr(ta, "_resolve_gurobi_license", lambda _x: None)

        ta.trackastra_tracking(
            None,
            mode="ilp",
            gpus="cpu",
            label_pattern="_labels.tif",
            _source_filepath=str(label),
        )
        assert "GRB_LICENSE_FILE" not in rec["env"]
        assert "No Gurobi license file found" in capsys.readouterr().out


class TestTrackastraMarkers:
    """Worker-facing markers on the registered callable."""

    def test_streaming_markers_stay_set(self):
        """Both worker implementations must keep skipping the eager load."""
        func = ta.trackastra_tracking
        assert func.skip_load is True
        assert func._loads_from_path is True

    def test_gpu_distribution_marker_stays_set(self):
        """The batch widget relies on this to raise the thread count."""
        assert ta.trackastra_tracking.supports_gpu_distribution is True


# ---------------------------------------------------------------------------
# ultrack: gurobi license resolution
# ---------------------------------------------------------------------------


class TestUltrackGurobiLicense:
    """Same first-hit-wins order as Trackastra, separate implementation."""

    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        """An existing explicit .lic short-circuits detection."""
        lic = tmp_path / "explicit.lic"
        lic.write_text("KEY", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
        assert ut._resolve_gurobi_license(str(lic)) == str(lic)

    def test_missing_explicit_path_falls_back_to_env(
        self, tmp_path, monkeypatch
    ):
        """A bogus path warns and then honours GRB_LICENSE_FILE."""
        env_lic = tmp_path / "env.lic"
        env_lic.write_text("KEY", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("GRB_LICENSE_FILE", str(env_lic))
        assert ut._resolve_gurobi_license("/nope.lic") == str(env_lic)

    def test_home_license_is_the_last_resort(self, tmp_path, monkeypatch):
        """~/gurobi.lic is picked up automatically."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
        home_lic = tmp_path / "gurobi.lic"
        home_lic.write_text("KEY", encoding="utf-8")
        assert ut._resolve_gurobi_license("") == str(home_lic)

    def test_nothing_found_returns_none(self, tmp_path, monkeypatch):
        """Without any license the default solver is used."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
        assert ut._resolve_gurobi_license("   ") is None


# ---------------------------------------------------------------------------
# ultrack: generated script
# ---------------------------------------------------------------------------


class TestUltrackScriptGeneration:
    """The GPU and CPU variants of the generated worker script differ."""

    def test_cpu_variant_disables_cuda_and_inlines_scipy_path(self):
        """enable_gpu=False must never import torch or leave CUDA on."""
        script = ut.create_ultrack_ensemble_script(
            label_paths=["/data/a_cp_labels.tif"],
            output_path="/data/a_ultrack.tif",
            enable_gpu=False,
        )
        # The script is executed by a subprocess, so syntax errors from a
        # mis-escaped brace in the f-string template are a real regression.
        compile(script, "<ultrack_cpu>", "exec")
        assert "GPU mode DISABLED" in script
        assert "os.environ['CUDA_VISIBLE_DEVICES'] = ''" in script
        assert "labels_to_contours_cpu" in script
        assert "NUMBA_DISABLE_CUDA" in script
        assert "import torch" not in script

    def test_gpu_variant_uses_torch(self):
        """enable_gpu=True inlines the PyTorch contour implementation."""
        script = ut.create_ultrack_ensemble_script(
            label_paths=["/data/a_cp_labels.tif"],
            output_path="/data/a_ultrack.tif",
            enable_gpu=True,
        )
        compile(script, "<ultrack_gpu>", "exec")
        assert "GPU MODE ENABLED" in script
        assert "_find_boundaries_torch" in script
        assert "labels_to_contours_cpu" not in script

    def test_solver_parameters_are_injected(self):
        """Widget parameters land in the generated ultrack config."""
        script = ut.create_ultrack_ensemble_script(
            label_paths=["/data/a.tif", "/data/b.tif"],
            output_path="/data/a_ultrack.tif",
            gurobi_license="/lic/gurobi.lic",
            min_area=321,
            window_size=None,
            solution_gap=0.05,
            enable_gpu=False,
        )
        compile(script, "<ultrack_params>", "exec")
        assert "config.segmentation_config.min_area = 321" in script
        assert "config.tracking_config.window_size = None" in script
        assert "config.tracking_config.solution_gap = 0.05" in script
        assert "gurobi_license = '/lic/gurobi.lic'" in script
        assert "'/data/a.tif'" in script
        assert "'/data/b.tif'" in script

    def test_temp_dir_is_cleaned_up_in_a_finally(self):
        """The staged label zarrs are removed on success and on failure."""
        script = ut.create_ultrack_ensemble_script(
            label_paths=["/data/a.tif"],
            output_path="/data/a_ultrack.tif",
            enable_gpu=False,
        )
        assert "finally:" in script
        assert "shutil.rmtree(temp_dir)" in script
        assert "tempfile.mkdtemp(prefix='ultrack_labels_')" in script


# ---------------------------------------------------------------------------
# ultrack: environment repair
# ---------------------------------------------------------------------------


class TestVerifyAndFixUltrackEnv:
    """`_verify_and_fix_ultrack_env` back-fills packages added over time."""

    def test_complete_env_needs_no_installs(self, monkeypatch):
        """Nothing is spawned when every critical package is present."""
        checked = []

        def fake_installed(pkg, env_name="ultrack"):
            checked.append((pkg, env_name))
            return True

        monkeypatch.setattr(ut, "is_package_installed", fake_installed)

        def explode(*a, **k):
            raise AssertionError("should not shell out")

        monkeypatch.setattr(subprocess, "run", explode)
        assert ut._verify_and_fix_ultrack_env() is True
        assert ("ultrack", "ultrack") in checked
        assert ("torch", "ultrack") in checked

    def test_missing_packages_are_pip_installed(self, monkeypatch):
        """Only the missing packages get a `pip install` in the env."""
        monkeypatch.setattr(
            ut,
            "is_package_installed",
            lambda pkg, env_name="ultrack": pkg != "tifffile",
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _FakeCompleted(0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ut._verify_and_fix_ultrack_env("myenv") is True
        assert calls[0] == ["which", "mamba"]
        assert calls[-1] == [
            "mamba",
            "run",
            "-n",
            "myenv",
            "pip",
            "install",
            "tifffile",
        ]

    def test_conda_is_used_when_mamba_is_absent(self, monkeypatch):
        """`which mamba` failing falls through to conda."""
        monkeypatch.setattr(
            ut,
            "is_package_installed",
            lambda pkg, env_name="ultrack": pkg != "zarr",
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd[:1] == ["which"]:
                return _FakeCompleted(0 if cmd[1] == "conda" else 1)
            return _FakeCompleted(0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ut._verify_and_fix_ultrack_env() is True
        assert calls[-1][0] == "conda"

    def test_no_conda_at_all_returns_false(self, monkeypatch):
        """Neither mamba nor conda on PATH is a hard failure."""
        monkeypatch.setattr(
            ut,
            "is_package_installed",
            lambda pkg, env_name="ultrack": False,
        )
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: _FakeCompleted(1)
        )
        assert ut._verify_and_fix_ultrack_env() is False

    def test_failed_install_returns_false(self, monkeypatch):
        """A non-zero pip exit stops the loop and reports failure."""
        monkeypatch.setattr(
            ut,
            "is_package_installed",
            lambda pkg, env_name="ultrack": pkg != "scipy",
        )

        def fake_run(cmd, **kwargs):
            if cmd[:1] == ["which"]:
                return _FakeCompleted(0)
            return _FakeCompleted(1, "", "no such package")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ut._verify_and_fix_ultrack_env() is False

    def test_unexpected_exception_returns_false(self, monkeypatch, capsys):
        """Any error while installing degrades to a manual-install hint."""
        monkeypatch.setattr(
            ut,
            "is_package_installed",
            lambda pkg, env_name="ultrack": pkg != "pandas",
        )

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(["which"], 5)

        monkeypatch.setattr(subprocess, "run", boom)
        assert ut._verify_and_fix_ultrack_env() is False
        assert "pip install pandas" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ultrack: the registered processing function
# ---------------------------------------------------------------------------


def _stub_ultrack_env(
    monkeypatch,
    *,
    env_created=True,
    created_ok=True,
    verify_ok=True,
    scikit_ok=True,
    license_path=None,
):
    """Neutralise every conda/env collaborator of ultrack_ensemble_tracking."""
    monkeypatch.setattr(ut, "is_env_created", lambda: env_created)
    monkeypatch.setattr(ut, "create_ultrack_env", lambda: created_ok)
    monkeypatch.setattr(ut, "_verify_and_fix_ultrack_env", lambda: verify_ok)
    monkeypatch.setattr(ut, "_ensure_scikit_image_fix", lambda: scikit_ok)
    monkeypatch.setattr(ut, "_resolve_gurobi_license", lambda _x: license_path)


def _stub_ultrack_run(monkeypatch, *, success=True, produce=True):
    """Record the call into ``run_ultrack_in_env`` and fake its result."""
    rec = {"kwargs": None, "runs": 0}

    def fake_run(**kwargs):
        rec["runs"] += 1
        rec["kwargs"] = kwargs
        if produce and kwargs.get("output_file"):
            Path(kwargs["output_file"]).write_text("out", encoding="utf-8")
        return {
            "success": success,
            "output": "ULTRACK-OUT",
            "error": "ULTRACK-ERR",
        }

    monkeypatch.setattr(ut, "run_ultrack_in_env", fake_run)
    return rec


def _forbid_ultrack_run(monkeypatch):
    """Make it impossible for a test to reach the real subprocess helper."""

    def explode(**kwargs):
        raise AssertionError("run_ultrack_in_env must not be reached")

    monkeypatch.setattr(ut, "run_ultrack_in_env", explode)


def _call_ultrack_through_worker_frame(filepath, image, **kwargs):
    """Emulate the worker frame that exposes a local named ``filepath``."""
    return ut.ultrack_ensemble_tracking(image, **kwargs)


class TestUltrackEnvironmentBootstrap:
    """The env checks run before anything touches the filesystem."""

    def test_failed_env_creation_returns_unchanged(self, monkeypatch, capsys):
        """A missing env that cannot be built aborts the run."""
        _stub_ultrack_env(monkeypatch, env_created=False, created_ok=False)
        _forbid_ultrack_run(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        assert ut.ultrack_ensemble_tracking(image) is image
        assert (
            "Failed to create ultrack environment" in capsys.readouterr().out
        )

    def test_degraded_env_only_warns(self, monkeypatch, capsys):
        """Missing packages / old scikit-image warn but do not abort."""
        _stub_ultrack_env(monkeypatch, verify_ok=False, scikit_ok=False)
        _forbid_ultrack_run(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        # No `filepath` in any caller frame -> falls through to the path guard.
        assert ut.ultrack_ensemble_tracking(image) is image
        printed = capsys.readouterr().out
        assert "Some packages may be missing" in printed
        assert "scikit-image version check failed" in printed
        assert "Could not determine input file path" in printed

    def test_missing_license_falls_back_to_default_solver(
        self, monkeypatch, capsys
    ):
        """No .lic anywhere is announced, not fatal."""
        _stub_ultrack_env(monkeypatch, license_path=None)
        _forbid_ultrack_run(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        assert ut.ultrack_ensemble_tracking(image) is image
        assert "using default solver" in capsys.readouterr().out


class TestUltrackLabelDiscovery:
    """Ensemble members are found from the first-suffix file only."""

    def test_empty_suffix_list_returns_unchanged(self, tmp_path, monkeypatch):
        """A blank `label_suffixes` cannot address any ensemble member."""
        _stub_ultrack_env(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        out = _call_ultrack_through_worker_frame(
            str(tmp_path / "sample_cp_labels.tif"),
            image,
            label_suffixes="  ,  ",
        )
        assert out is image

    def test_non_first_suffix_file_is_skipped(
        self, tmp_path, monkeypatch, capsys
    ):
        """Only the first suffix drives a run; the rest are no-ops."""
        _stub_ultrack_env(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        out = _call_ultrack_through_worker_frame(
            str(tmp_path / "sample_convpaint_labels.tif"),
            image,
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert out is image
        assert "doesn't match the first suffix" in capsys.readouterr().out

    def test_no_label_files_on_disk_returns_unchanged(
        self, tmp_path, monkeypatch, capsys
    ):
        """Nothing to ensemble -> unchanged image."""
        _stub_ultrack_env(monkeypatch)
        image = np.zeros((3, 4, 4), dtype=np.uint16)
        out = _call_ultrack_through_worker_frame(
            str(tmp_path / "sample_cp_labels.tif"),
            image,
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert out is image
        assert "No label files found" in capsys.readouterr().out

    def test_single_member_warns_but_still_runs(
        self, tmp_path, monkeypatch, capsys
    ):
        """One segmentation is allowed, with a quality warning."""
        first = tmp_path / "sample_cp_labels.tif"
        first.write_text("", encoding="utf-8")
        _stub_ultrack_env(monkeypatch)
        rec = _stub_ultrack_run(monkeypatch)
        monkeypatch.setattr(
            ut, "imread", lambda p: np.ones((3, 4, 4), dtype=np.uint16)
        )

        out = _call_ultrack_through_worker_frame(
            str(first),
            np.zeros((3, 4, 4), dtype=np.uint16),
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert rec["runs"] == 1
        assert np.all(out == 1)
        assert "Ensemble works best with 2+" in capsys.readouterr().out


class TestUltrackRun:
    """Script hand-off, result collection and failure handling."""

    def _make_pair(self, tmp_path):
        first = tmp_path / "sample_cp_labels.tif"
        second = tmp_path / "sample_convpaint_labels.tif"
        first.write_text("", encoding="utf-8")
        second.write_text("", encoding="utf-8")
        return first, second

    def test_successful_run_returns_the_tracked_array(
        self, tmp_path, monkeypatch
    ):
        """The produced tif is read back and returned to the worker."""
        first, second = self._make_pair(tmp_path)
        _stub_ultrack_env(monkeypatch, license_path="/lic/gurobi.lic")
        rec = _stub_ultrack_run(monkeypatch)
        tracked = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)
        read = []
        monkeypatch.setattr(ut, "imread", lambda p: read.append(p) or tracked)

        out = _call_ultrack_through_worker_frame(
            str(first),
            np.zeros((3, 2, 2), dtype=np.uint16),
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert out is tracked
        expected_output = str(tmp_path / "sample_ultrack.tif")
        assert read == [expected_output]
        kwargs = rec["kwargs"]
        assert kwargs["input_file"] == str(first)
        assert kwargs["output_file"] == expected_output
        assert kwargs["extra_env"] == {"GRB_LICENSE_FILE": "/lic/gurobi.lic"}
        assert "'" + str(second) + "'" in kwargs["script_content"]

    def test_no_license_means_no_extra_env(self, tmp_path, monkeypatch):
        """Without a license the subprocess env is left untouched."""
        first, _ = self._make_pair(tmp_path)
        _stub_ultrack_env(monkeypatch, license_path=None)
        rec = _stub_ultrack_run(monkeypatch)
        monkeypatch.setattr(
            ut, "imread", lambda p: np.zeros((3, 2, 2), dtype=np.uint16)
        )

        _call_ultrack_through_worker_frame(
            str(first),
            np.zeros((3, 2, 2), dtype=np.uint16),
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert rec["kwargs"]["extra_env"] is None

    def test_failed_run_returns_unchanged(self, tmp_path, monkeypatch, capsys):
        """A failed subprocess logs both streams and returns the input."""
        first, _ = self._make_pair(tmp_path)
        _stub_ultrack_env(monkeypatch)
        _stub_ultrack_run(monkeypatch, success=False, produce=False)
        image = np.zeros((3, 2, 2), dtype=np.uint16)

        out = _call_ultrack_through_worker_frame(
            str(first),
            image,
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert out is image
        printed = capsys.readouterr().out
        assert "ULTRACK-OUT" in printed
        assert "ULTRACK-ERR" in printed

    def test_missing_output_returns_unchanged(
        self, tmp_path, monkeypatch, capsys
    ):
        """A 'successful' run without an output file is still a no-op."""
        first, _ = self._make_pair(tmp_path)
        _stub_ultrack_env(monkeypatch)
        _stub_ultrack_run(monkeypatch, success=True, produce=False)
        image = np.zeros((3, 2, 2), dtype=np.uint16)

        out = _call_ultrack_through_worker_frame(
            str(first),
            image,
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
        )
        assert out is image
        assert "did not produce output" in capsys.readouterr().out

    def test_window_size_zero_disables_windowing(self, tmp_path, monkeypatch):
        """window_size=0 is translated to None ('whole timelapse')."""
        first, second = self._make_pair(tmp_path)
        _stub_ultrack_env(monkeypatch)
        _stub_ultrack_run(monkeypatch, produce=False)
        seen = {}

        def fake_script(**kwargs):
            seen.update(kwargs)
            return "print('stub')\n"

        monkeypatch.setattr(ut, "create_ultrack_ensemble_script", fake_script)

        _call_ultrack_through_worker_frame(
            str(first),
            np.zeros((3, 2, 2), dtype=np.uint16),
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
            window_size=0,
            min_area=42,
            enable_gpu=False,
        )
        assert seen["window_size"] is None
        assert seen["min_area"] == 42
        assert seen["enable_gpu"] is False
        assert seen["label_paths"] == [str(first), str(second)]

    def test_nonzero_window_size_is_passed_through(
        self, tmp_path, monkeypatch
    ):
        """Any other window size reaches the generated config verbatim."""
        first, _ = self._make_pair(tmp_path)
        _stub_ultrack_env(monkeypatch)
        _stub_ultrack_run(monkeypatch, produce=False)
        seen = {}
        monkeypatch.setattr(
            ut,
            "create_ultrack_ensemble_script",
            lambda **kwargs: seen.update(kwargs) or "print('stub')\n",
        )

        _call_ultrack_through_worker_frame(
            str(first),
            np.zeros((3, 2, 2), dtype=np.uint16),
            label_suffixes="_cp_labels.tif,_convpaint_labels.tif",
            window_size=7,
            solution_gap=0.02,
        )
        assert seen["window_size"] == 7
        assert seen["solution_gap"] == 0.02

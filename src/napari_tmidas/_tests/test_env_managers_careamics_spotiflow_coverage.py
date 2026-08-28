"""
Coverage tests for the CAREamics / Spotiflow / SAM2 environment managers
and for the CAREamics denoising processing function.

Every test in this file replaces the ``subprocess`` module *object* on the
module under test with a stand-in namespace, so no test ever shells out.
Temporary files are redirected into ``tmp_path`` by pointing
``tempfile.tempdir`` at it, so nothing is written into the real system
temp directory or the repository tree.
"""

import io
import os
import re
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest
import tifffile

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import (
    careamics_denoising as cd,
)
from napari_tmidas.processing_functions import (
    careamics_env_manager as cem,
)
from napari_tmidas.processing_functions import (
    sam2_env_manager as sem,
)
from napari_tmidas.processing_functions import (
    spotiflow_env_manager as spm,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _forbidden(name):
    """Build a stub that fails loudly if a test really shells out."""

    def _call(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess.{name}() call")

    return _call


def fake_subprocess(**overrides):
    """A stand-in for the ``subprocess`` module.

    Real exception/constant objects are re-exported so that ``except
    subprocess.CalledProcessError`` inside the module under test still
    matches, while every process-spawning entry point is replaced.
    """
    ns = types.SimpleNamespace(
        PIPE=subprocess.PIPE,
        DEVNULL=subprocess.DEVNULL,
        CalledProcessError=subprocess.CalledProcessError,
        TimeoutExpired=subprocess.TimeoutExpired,
        CompletedProcess=subprocess.CompletedProcess,
    )
    for name in ("run", "check_call", "check_output", "call", "Popen"):
        setattr(ns, name, _forbidden(name))
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def completed(cmd, returncode=0, stdout="", stderr=""):
    """Build a real CompletedProcess so attribute access is realistic."""
    return subprocess.CompletedProcess(
        args=list(cmd), returncode=returncode, stdout=stdout, stderr=stderr
    )


def install_fake_torch(monkeypatch, *, cuda, device_count=1):
    """Make ``import torch`` inside a module resolve to a stub."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        device_count=lambda: device_count,
        get_device_name=lambda index: f"FakeGPU{index}",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def block_torch_import(monkeypatch):
    """Make ``import torch`` raise ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)


class FakeProcess:
    """Minimal stand-in for a ``subprocess.Popen`` handle."""

    def __init__(self, returncode=0, stdout_lines=(), stderr=""):
        self.returncode = returncode
        self.stdout = iter(list(stdout_lines))
        self.stderr = io.StringIO(stderr)
        self.waited = False

    def wait(self):
        self.waited = True
        return self.returncode


@pytest.fixture()
def temp_root(tmp_path, monkeypatch):
    """Redirect every ``tempfile`` allocation into ``tmp_path``."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class StubManager:
    """Records delegation from the module-level convenience wrappers."""

    def __init__(self):
        self.calls = []

    def is_package_installed(self):
        self.calls.append("is_package_installed")
        return True

    def is_env_created(self):
        self.calls.append("is_env_created")
        return False

    def get_env_python_path(self):
        self.calls.append("get_env_python_path")
        return "/stub/bin/python"

    def create_env(self):
        self.calls.append("create_env")
        return "/stub/bin/python"

    def delete_env(self):
        self.calls.append("delete_env")


# ===========================================================================
# careamics_env_manager
# ===========================================================================


class TestCareamicsInstallDependencies:
    """Pins the pip invocations issued when building the CAREamics env."""

    def test_installs_torch_careamics_and_tifffile_with_cuda(
        self, monkeypatch, capsys
    ):
        install_fake_torch(monkeypatch, cuda=True)
        calls = []
        verified = []
        monkeypatch.setattr(
            cem,
            "subprocess",
            fake_subprocess(
                check_call=lambda cmd, **kw: calls.append(list(cmd))
            ),
        )
        manager = cem.CAREamicsEnvironmentManager()
        monkeypatch.setattr(manager, "_verify_installation", verified.append)

        manager._install_dependencies("/env/bin/python")

        assert [cmd[4] for cmd in calls] == [
            "torch",
            "careamics[tensorboard]",
            "tifffile",
        ]
        assert all(cmd[0] == "/env/bin/python" for cmd in calls)
        assert verified == ["/env/bin/python"]
        out = capsys.readouterr().out
        assert "CUDA is available" in out
        assert "Installing PyTorch with CUDA support" in out

    def test_missing_torch_falls_back_to_cpu_message(
        self, monkeypatch, capsys
    ):
        block_torch_import(monkeypatch)
        calls = []
        monkeypatch.setattr(
            cem,
            "subprocess",
            fake_subprocess(
                check_call=lambda cmd, **kw: calls.append(list(cmd))
            ),
        )
        manager = cem.CAREamicsEnvironmentManager()
        monkeypatch.setattr(
            manager, "_verify_installation", lambda env_python: None
        )

        manager._install_dependencies("/env/bin/python")

        out = capsys.readouterr().out
        assert "PyTorch not detected in main environment" in out
        assert "Installing PyTorch without CUDA support" in out
        assert len(calls) == 3


class TestCareamicsVerifyInstallation:
    """Pins the generated check script and its temp-file lifecycle."""

    def _patch_run(self, monkeypatch, seen, **kwargs):
        def fake_run(cmd, **kw):
            seen["cmd"] = list(cmd)
            with open(cmd[1]) as handle:
                seen["script"] = handle.read()
            seen["kwargs"] = kw
            if "raise_exc" in kwargs:
                raise kwargs["raise_exc"]
            return completed(cmd, stdout=kwargs.get("stdout", ""))

        monkeypatch.setattr(cem, "subprocess", fake_subprocess(run=fake_run))

    def test_success_prints_confirmation_and_removes_script(
        self, monkeypatch, temp_root, capsys
    ):
        seen = {}
        self._patch_run(monkeypatch, seen, stdout="SUCCESS: CAREamics ok\n")
        cem.CAREamicsEnvironmentManager()._verify_installation(
            "/env/bin/python"
        )

        assert seen["cmd"][0] == "/env/bin/python"
        assert seen["cmd"][1].startswith(str(temp_root))
        assert "import careamics" in seen["script"]
        assert "CAREamist" in seen["script"]
        assert seen["kwargs"]["check"] is True
        assert not os.path.exists(seen["cmd"][1])
        assert "verified successfully" in capsys.readouterr().out

    def test_missing_success_marker_warns(
        self, monkeypatch, temp_root, capsys
    ):
        seen = {}
        self._patch_run(monkeypatch, seen, stdout="CAREamics version: 0.1\n")
        cem.CAREamicsEnvironmentManager()._verify_installation(
            "/env/bin/python"
        )
        assert "verification uncertain" in capsys.readouterr().out
        assert not os.path.exists(seen["cmd"][1])

    def test_subprocess_failure_propagates_but_still_unlinks(
        self, monkeypatch, temp_root
    ):
        seen = {}
        self._patch_run(
            monkeypatch,
            seen,
            raise_exc=subprocess.CalledProcessError(1, ["python"]),
        )
        with pytest.raises(subprocess.CalledProcessError):
            cem.CAREamicsEnvironmentManager()._verify_installation(
                "/env/bin/python"
            )
        assert not os.path.exists(seen["cmd"][1])


class TestCareamicsModuleWrappers:
    """The module-level helpers must delegate to the global manager."""

    def test_wrappers_delegate_to_manager(self, monkeypatch):
        stub = StubManager()
        monkeypatch.setattr(cem, "manager", stub)

        assert cem.is_careamics_installed() is True
        assert cem.is_env_created() is False
        assert cem.get_env_python_path() == "/stub/bin/python"
        assert cem.create_careamics_env() == "/stub/bin/python"
        assert stub.calls == [
            "is_package_installed",
            "is_env_created",
            "get_env_python_path",
            "create_env",
        ]

    def test_recreate_deletes_before_creating(self, monkeypatch, capsys):
        stub = StubManager()
        monkeypatch.setattr(cem, "manager", stub)

        assert cem.recreate_careamics_env() == "/stub/bin/python"

        assert stub.calls == ["delete_env", "create_env"]
        assert "Recreating CAREamics environment" in capsys.readouterr().out

    def test_is_package_installed_reports_a_bool(self):
        assert isinstance(
            cem.CAREamicsEnvironmentManager().is_package_installed(), bool
        )


class TestRunCareamicsInEnv:
    """Drives the generated-script / subprocess / read-back round trip."""

    @staticmethod
    def _args(image, **extra):
        args = {
            "image": image,
            "checkpoint_path": "/models/last.ckpt",
            "tile_size_z": 16,
            "tile_size_y": 64,
            "tile_size_x": 32,
            "tile_overlap_z": 4,
            "tile_overlap_y": 8,
            "tile_overlap_x": 2,
            "batch_size": 3,
            # Deliberately the opposite of the script template's own
            # default (True) so a broken args_dict['use_tta'] lookup -
            # which would silently fall back to the default - shows up
            # as a mismatch instead of an accidental pass.
            "use_tta": False,
        }
        args.update(extra)
        return args

    @staticmethod
    def _popen(
        record, *, result=None, returncode=0, stdout_lines=(), stderr=""
    ):
        def _factory(cmd, **kwargs):
            with open(cmd[1]) as handle:
                script = handle.read()
            record.append(
                {
                    "cmd": list(cmd),
                    "script": script,
                    "kwargs": kwargs,
                }
            )
            if result is not None:
                target = re.search(
                    r"tifffile\.imwrite\('([^']+)'", script
                ).group(1)
                tifffile.imwrite(target, result)
            return FakeProcess(returncode, stdout_lines, stderr)

        return _factory

    def _patch_env(self, monkeypatch, created=True, popen=None):
        monkeypatch.setattr(cem, "is_env_created", lambda: created)
        monkeypatch.setattr(
            cem, "get_env_python_path", lambda: "/env/bin/python"
        )
        monkeypatch.setattr(cem, "subprocess", fake_subprocess(Popen=popen))

    def test_returns_image_written_by_the_child_process(
        self, monkeypatch, temp_root, capsys
    ):
        image = np.arange(16, dtype=np.uint16).reshape(4, 4)
        expected = (image * 3).astype(np.uint16)
        record = []
        self._patch_env(
            monkeypatch,
            popen=self._popen(
                record, result=expected, stdout_lines=["Done!\n"]
            ),
        )

        out = cem.run_careamics_in_env("predict", self._args(image))

        np.testing.assert_array_equal(out, expected)
        assert out.dtype == np.uint16
        assert record[0]["cmd"][0] == "/env/bin/python"
        assert record[0]["kwargs"]["text"] is True
        script = record[0]["script"]
        assert "/models/last.ckpt" in script
        assert "batch_size=3" in script
        assert "tta=False" in script
        assert "Done!" in capsys.readouterr().out
        # every temp file created for the run is cleaned up again
        assert not os.path.exists(record[0]["cmd"][1])
        assert list(temp_root.iterdir()) == []

    def test_creates_environment_when_missing(self, monkeypatch, temp_root):
        image = np.ones((2, 2), dtype=np.uint8)
        created = []
        record = []
        self._patch_env(
            monkeypatch,
            created=False,
            popen=self._popen(record, result=image),
        )
        monkeypatch.setattr(
            cem, "create_careamics_env", lambda: created.append(True)
        )

        cem.run_careamics_in_env("predict", self._args(image))

        assert created == [True]

    def test_child_failure_returns_the_original_image(
        self, monkeypatch, temp_root, capsys
    ):
        image = np.full((3, 3), 7, dtype=np.uint16)
        record = []
        self._patch_env(
            monkeypatch,
            popen=self._popen(
                record, returncode=1, stderr="traceback: boom\n"
            ),
        )

        out = cem.run_careamics_in_env("predict", self._args(image))

        assert out is image
        captured = capsys.readouterr().out
        assert "Error in CAREamics processing" in captured
        assert "traceback: boom" in captured

    def test_version_mismatch_recreates_env_and_retries_once(
        self, monkeypatch, temp_root
    ):
        image = np.arange(4, dtype=np.uint16).reshape(2, 2)
        expected = (image + 10).astype(np.uint16)
        attempts = []
        recreated = []

        def popen(cmd, **kwargs):
            with open(cmd[1]) as handle:
                script = handle.read()
            attempts.append(script)
            if len(attempts) == 1:
                return FakeProcess(
                    1,
                    (),
                    "Lightning v2.1 checkpoint is newer than the "
                    "installed version",
                )
            target = re.search(r"tifffile\.imwrite\('([^']+)'", script).group(
                1
            )
            tifffile.imwrite(target, expected)
            return FakeProcess(0)

        self._patch_env(monkeypatch, popen=popen)
        monkeypatch.setattr(
            cem, "recreate_careamics_env", lambda: recreated.append(True)
        )

        out = cem.run_careamics_in_env("predict", self._args(image))

        assert recreated == [True]
        assert len(attempts) == 2
        np.testing.assert_array_equal(out, expected)

    def test_size_mismatch_on_a_retry_is_not_retried_again(
        self, monkeypatch, temp_root
    ):
        image = np.zeros((2, 2), dtype=np.uint16)
        record = []
        self._patch_env(
            monkeypatch,
            popen=self._popen(
                record, returncode=1, stderr="size mismatch for layer.0"
            ),
        )
        monkeypatch.setattr(
            cem,
            "recreate_careamics_env",
            _forbidden("recreate_careamics_env"),
        )

        out = cem.run_careamics_in_env(
            "predict", self._args(image), retry_count=1
        )

        assert out is image
        assert len(record) == 1


# ===========================================================================
# spotiflow_env_manager
# ===========================================================================


class TestSpotiflowInstallDependencies:
    """Pins CUDA detection and the resulting pip command sequence."""

    def _manager(self, monkeypatch, **subprocess_overrides):
        manager = spm.SpotiflowEnvironmentManager()
        monkeypatch.setattr(
            manager, "_verify_installation", lambda env_python: None
        )
        monkeypatch.setattr(
            spm, "subprocess", fake_subprocess(**subprocess_overrides)
        )
        return manager

    def test_cuda_path_installs_cu118_wheels_and_passes_the_test(
        self, monkeypatch, capsys
    ):
        install_fake_torch(monkeypatch, cuda=True, device_count=2)
        calls = []
        manager = self._manager(
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=lambda cmd, **kw: completed(cmd, returncode=0),
        )

        manager._install_dependencies("/env/bin/python")

        assert "--index-url" in calls[0]
        assert calls[0][-1].endswith("/cu118")
        assert [cmd[4] for cmd in calls[1:]] == ["spotiflow", "tifffile"]
        out = capsys.readouterr().out
        assert "GPU detected: FakeGPU0" in out
        assert "CUDA compatibility test passed" in out

    def test_failed_cuda_check_reinstalls_cpu_wheels(
        self, monkeypatch, capsys
    ):
        install_fake_torch(monkeypatch, cuda=True)
        calls = []
        manager = self._manager(
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=lambda cmd, **kw: completed(cmd, returncode=1),
        )

        manager._install_dependencies("/env/bin/python")

        assert [cmd[3] for cmd in calls[:3]] == [
            "install",
            "uninstall",
            "install",
        ]
        assert "--index-url" not in calls[2]
        assert "Switched to CPU-only PyTorch" in capsys.readouterr().out

    def test_cuda_wheel_install_error_falls_back_to_cpu(
        self, monkeypatch, capsys
    ):
        install_fake_torch(monkeypatch, cuda=True)
        calls = []

        def check_call(cmd, **kwargs):
            calls.append(list(cmd))
            if "--index-url" in cmd:
                raise subprocess.CalledProcessError(1, list(cmd))

        manager = self._manager(monkeypatch, check_call=check_call)

        manager._install_dependencies("/env/bin/python")

        assert "--index-url" in calls[0]
        assert calls[1][4:6] == ["torch==2.0.1", "torchvision==0.15.2"]
        out = capsys.readouterr().out
        assert "CUDA PyTorch installation failed" in out
        assert "CPU-only PyTorch installed as fallback" in out

    def test_nvidia_smi_detects_gpu_when_torch_is_missing(
        self, monkeypatch, capsys
    ):
        block_torch_import(monkeypatch)
        calls = []
        seen = []

        def run(cmd, **kwargs):
            seen.append(list(cmd))
            return completed(cmd, returncode=0)

        manager = self._manager(
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=run,
        )

        manager._install_dependencies("/env/bin/python")

        assert seen[0] == ["nvidia-smi"]
        assert "--index-url" in calls[0]
        assert "NVIDIA GPU detected via nvidia-smi" in capsys.readouterr().out

    def test_nvidia_smi_without_gpu_installs_cpu_wheels(
        self, monkeypatch, capsys
    ):
        block_torch_import(monkeypatch)
        calls = []
        manager = self._manager(
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=lambda cmd, **kw: completed(cmd, returncode=1),
        )

        manager._install_dependencies("/env/bin/python")

        assert len(calls) == 3
        assert "--index-url" not in calls[0]
        assert "No NVIDIA GPU detected" in capsys.readouterr().out

    def test_missing_nvidia_smi_installs_cpu_wheels(self, monkeypatch, capsys):
        block_torch_import(monkeypatch)
        calls = []

        def run(cmd, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        manager = self._manager(
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=run,
        )

        manager._install_dependencies("/env/bin/python")

        assert len(calls) == 3
        assert "--index-url" not in calls[0]
        assert calls[0][4:6] == ["torch==2.0.1", "torchvision==0.15.2"]
        out = capsys.readouterr().out
        assert "nvidia-smi not found" in out
        assert "Installing PyTorch without CUDA support" in out

    def test_torch_without_cuda_installs_cpu_wheels(self, monkeypatch, capsys):
        install_fake_torch(monkeypatch, cuda=False)
        calls = []
        manager = self._manager(
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
        )

        manager._install_dependencies("/env/bin/python")

        assert len(calls) == 3
        assert "CUDA is not available" in capsys.readouterr().out


class TestSpotiflowVerifyInstallation:
    """Verification succeeds, warns loudly, or re-raises - never silently."""

    def _patch_run(self, monkeypatch, seen, **kwargs):
        def fake_run(cmd, **kw):
            seen["cmd"] = list(cmd)
            with open(cmd[1]) as handle:
                seen["script"] = handle.read()
            if "raise_exc" in kwargs:
                raise kwargs["raise_exc"]
            return completed(cmd, stdout=kwargs.get("stdout", ""))

        monkeypatch.setattr(spm, "subprocess", fake_subprocess(run=fake_run))

    def test_success_marker_accepted(self, monkeypatch, temp_root, capsys):
        seen = {}
        self._patch_run(monkeypatch, seen, stdout="SUCCESS: Spotiflow ok")
        spm.SpotiflowEnvironmentManager()._verify_installation(
            "/env/bin/python"
        )
        assert "import spotiflow" in seen["script"]
        assert "verified successfully" in capsys.readouterr().out
        assert not os.path.exists(seen["cmd"][1])

    def test_missing_success_marker_raises_runtime_error(
        self, monkeypatch, temp_root
    ):
        seen = {}
        self._patch_run(monkeypatch, seen, stdout="Spotiflow version: 0.5")
        with pytest.raises(RuntimeError, match="verification failed"):
            spm.SpotiflowEnvironmentManager()._verify_installation(
                "/env/bin/python"
            )
        assert not os.path.exists(seen["cmd"][1])

    def test_called_process_error_is_reported_and_reraised(
        self, monkeypatch, temp_root, capsys
    ):
        seen = {}
        self._patch_run(
            monkeypatch,
            seen,
            raise_exc=subprocess.CalledProcessError(
                2, ["python"], stderr="no module named spotiflow"
            ),
        )
        with pytest.raises(subprocess.CalledProcessError):
            spm.SpotiflowEnvironmentManager()._verify_installation(
                "/env/bin/python"
            )
        assert "no module named spotiflow" in capsys.readouterr().out
        assert not os.path.exists(seen["cmd"][1])


class TestSpotiflowModuleWrappers:
    """Delegation of the module-level Spotiflow helpers."""

    def test_wrappers_delegate_to_manager(self, monkeypatch):
        stub = StubManager()
        monkeypatch.setattr(spm, "manager", stub)

        assert spm.is_spotiflow_installed() is True
        assert spm.is_env_created() is False
        assert spm.get_env_python_path() == "/stub/bin/python"
        assert spm.create_spotiflow_env() == "/stub/bin/python"
        assert stub.calls == [
            "is_package_installed",
            "is_env_created",
            "get_env_python_path",
            "create_env",
        ]

    def test_is_package_installed_reports_a_bool(self):
        assert isinstance(
            spm.SpotiflowEnvironmentManager().is_package_installed(), bool
        )


class TestRunSpotiflowInEnv:
    """The Spotiflow child-process round trip and its failure mode."""

    @staticmethod
    def _args(image, **extra):
        args = {
            "image": image,
            "model_path": "",
            # Deliberately not the script template's own default
            # ("general") so a broken args_dict['pretrained_model']
            # lookup - which would silently fall back to the default -
            # shows up as a mismatch instead of an accidental pass.
            "pretrained_model": "test_pretrained_v2",
            "prob_thresh": 0.4,
            "min_distance": 3,
            "subpixel": True,
            "peak_mode": "fast",
        }
        args.update(extra)
        return args

    def test_round_trip_returns_saved_points(
        self, monkeypatch, temp_root, capsys
    ):
        image = np.zeros((8, 8), dtype=np.uint16)
        image[2, 3] = 900
        points = np.array([[2.0, 3.0], [5.0, 6.0]])
        record = []
        created = []

        def run(cmd, **kwargs):
            with open(cmd[1]) as handle:
                script = handle.read()
            record.append({"cmd": list(cmd), "script": script})
            target = re.search(r"np\.save\('([^']+)'", script).group(1)
            np.save(target, {"points": points})
            return completed(cmd, stdout="Detected 2 spots\n")

        monkeypatch.setattr(spm, "is_env_created", lambda: False)
        monkeypatch.setattr(
            spm, "create_spotiflow_env", lambda: created.append(True)
        )
        monkeypatch.setattr(
            spm, "get_env_python_path", lambda: "/env/bin/python"
        )
        monkeypatch.setattr(spm, "subprocess", fake_subprocess(run=run))

        result = spm.run_spotiflow_in_env("predict", self._args(image))

        assert created == [True]
        assert set(result) == {"points"}
        np.testing.assert_array_equal(result["points"], points)
        script = record[0]["script"]
        assert "Spotiflow.from_pretrained('test_pretrained_v2')" in script
        assert "'min_distance': 3" in script
        assert "prob_thresh = 0.4" in script
        assert "Detected 2 spots" in capsys.readouterr().out
        assert not os.path.exists(record[0]["cmd"][1])
        # every temp file created for the run is cleaned up again
        assert list(temp_root.iterdir()) == []

    def test_custom_model_path_is_embedded_in_the_script(
        self, monkeypatch, temp_root
    ):
        image = np.zeros((4, 4), dtype=np.uint16)
        record = []

        def run(cmd, **kwargs):
            with open(cmd[1]) as handle:
                script = handle.read()
            record.append(script)
            target = re.search(r"np\.save\('([^']+)'", script).group(1)
            np.save(target, {"points": np.empty((0, 2))})
            return completed(cmd)

        monkeypatch.setattr(spm, "is_env_created", lambda: True)
        monkeypatch.setattr(
            spm, "get_env_python_path", lambda: "/env/bin/python"
        )
        monkeypatch.setattr(spm, "subprocess", fake_subprocess(run=run))

        result = spm.run_spotiflow_in_env(
            "predict", self._args(image, model_path="/models/mine")
        )

        assert result["points"].shape == (0, 2)
        assert "Spotiflow.from_folder('/models/mine')" in record[0]
        # every temp file created for the run is cleaned up again
        assert list(temp_root.iterdir()) == []

    def test_nonzero_exit_raises_called_process_error(
        self, monkeypatch, temp_root, capsys
    ):
        image = np.zeros((4, 4), dtype=np.uint16)

        monkeypatch.setattr(spm, "is_env_created", lambda: True)
        monkeypatch.setattr(
            spm, "get_env_python_path", lambda: "/env/bin/python"
        )
        monkeypatch.setattr(
            spm,
            "subprocess",
            fake_subprocess(
                run=lambda cmd, **kw: completed(
                    cmd, returncode=3, stdout="partial", stderr="kaboom"
                )
            ),
        )

        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            spm.run_spotiflow_in_env("predict", self._args(image))

        assert excinfo.value.returncode == 3
        assert "kaboom" in capsys.readouterr().out


# ===========================================================================
# sam2_env_manager
# ===========================================================================


class TestSam2Manager:
    """Repository cloning, checkpoint download and install detection."""

    def _manager(self, tmp_path, monkeypatch, **subprocess_overrides):
        manager = sem.SAM2EnvironmentManager()
        manager.env_dir = str(tmp_path / "env")
        manager.sam2_repo_dir = str(tmp_path / "env" / "sam2_repo")
        manager.checkpoints_dir = os.path.join(
            manager.sam2_repo_dir, "checkpoints"
        )
        monkeypatch.setattr(
            sem, "subprocess", fake_subprocess(**subprocess_overrides)
        )
        return manager

    def test_init_derives_repo_and_checkpoint_paths(self):
        manager = sem.SAM2EnvironmentManager()
        assert manager.env_name == "sam2-env"
        assert manager.sam2_repo_dir == os.path.join(
            manager.env_dir, "sam2_repo"
        )
        assert manager.checkpoints_dir == os.path.join(
            manager.sam2_repo_dir, "checkpoints"
        )

    def test_install_clones_repository_when_absent(
        self, tmp_path, monkeypatch, capsys
    ):
        calls = []
        runs = []
        manager = self._manager(
            tmp_path,
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=lambda cmd, **kw: runs.append(list(cmd)),
        )
        downloaded = []
        monkeypatch.setattr(
            manager, "_download_model_checkpoint", downloaded.append
        )

        manager._install_dependencies("/env/bin/python")

        assert calls[0][4:6] == ["torch", "torchvision"]
        assert calls[1][:2] == ["git", "clone"]
        assert calls[1][-1] == manager.sam2_repo_dir
        assert calls[2][4:6] == ["-e", manager.sam2_repo_dir]
        assert downloaded == ["/env/bin/python"]
        assert "import sam2" in runs[0][2]
        assert "Cloning SAM2 repository" in capsys.readouterr().out

    def test_install_pulls_when_repository_already_present(
        self, tmp_path, monkeypatch, capsys
    ):
        calls = []
        manager = self._manager(
            tmp_path,
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
            run=lambda cmd, **kw: None,
        )
        os.makedirs(manager.sam2_repo_dir)
        monkeypatch.setattr(
            manager, "_download_model_checkpoint", lambda env: None
        )

        manager._install_dependencies("/env/bin/python")

        assert calls[1] == ["git", "-C", manager.sam2_repo_dir, "pull"]
        assert "already exists" in capsys.readouterr().out

    def test_existing_checkpoint_short_circuits_download(
        self, tmp_path, monkeypatch, capsys
    ):
        manager = self._manager(tmp_path, monkeypatch)
        os.makedirs(manager.checkpoints_dir)
        checkpoint = os.path.join(
            manager.checkpoints_dir, "sam2.1_hiera_large.pt"
        )
        with open(checkpoint, "wb") as handle:
            handle.write(b"weights")

        manager._download_model_checkpoint("/env/bin/python")

        assert "already exists" in capsys.readouterr().out

    def test_download_script_targets_the_checkpoint_url(
        self, tmp_path, monkeypatch
    ):
        calls = []
        manager = self._manager(
            tmp_path,
            monkeypatch,
            check_call=lambda cmd, **kw: calls.append(list(cmd)),
        )

        manager._download_model_checkpoint("/env/bin/python")

        assert os.path.isdir(manager.checkpoints_dir)
        script = calls[0][2]
        assert "sam2.1_hiera_large.pt" in script
        assert "urllib.request.urlretrieve" in script
        assert manager.checkpoints_dir in script

    def test_download_failure_prints_manual_instructions(
        self, tmp_path, monkeypatch, capsys
    ):
        def check_call(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, list(cmd))

        manager = self._manager(tmp_path, monkeypatch, check_call=check_call)

        manager._download_model_checkpoint("/env/bin/python")

        out = capsys.readouterr().out
        assert "Failed to download model" in out
        assert "dl.fbaipublicfiles.com" in out

    def test_package_check_false_without_interpreter(
        self, tmp_path, monkeypatch
    ):
        manager = self._manager(tmp_path, monkeypatch)
        assert manager.is_package_installed() is False

    def test_package_check_uses_child_return_code(self, tmp_path, monkeypatch):
        seen = []

        def run(cmd, **kwargs):
            seen.append((list(cmd), kwargs))
            return completed(cmd, returncode=0)

        manager = self._manager(tmp_path, monkeypatch, run=run)
        env_python = manager.get_env_python_path()
        os.makedirs(os.path.dirname(env_python))
        with open(env_python, "w") as handle:
            handle.write("")

        assert manager.is_package_installed() is True
        assert seen[0][0][1:] == ["-c", "import sam2"]
        assert seen[0][1]["timeout"] == 5

        monkeypatch.setattr(
            sem,
            "subprocess",
            fake_subprocess(run=lambda cmd, **kw: completed(cmd, 1)),
        )
        assert manager.is_package_installed() is False

    def test_package_check_false_on_timeout(self, tmp_path, monkeypatch):
        def run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(list(cmd), 5)

        manager = self._manager(tmp_path, monkeypatch, run=run)
        env_python = manager.get_env_python_path()
        os.makedirs(os.path.dirname(env_python))
        with open(env_python, "w") as handle:
            handle.write("")

        assert manager.is_package_installed() is False

    def test_wrappers_delegate_to_manager(self, monkeypatch):
        stub = StubManager()
        monkeypatch.setattr(sem, "manager", stub)

        assert sem.is_sam2_installed() is True
        assert sem.is_env_created() is False
        assert sem.get_env_python_path() == "/stub/bin/python"
        assert sem.create_sam2_env() == "/stub/bin/python"
        assert stub.calls == [
            "is_package_installed",
            "is_env_created",
            "get_env_python_path",
            "create_env",
        ]

    def test_run_sam2_in_env_is_an_unimplemented_stub(self):
        # Documented here because callers must not rely on a return value:
        # the function body is only a docstring.
        assert sem.run_sam2_in_env("predict", {"image": None}) is None


# ===========================================================================
# careamics_denoising
# ===========================================================================


class FakeCAREamist:
    """Stand-in model: doubles the input and adds a leading singleton axis."""

    instances = []

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.calls = []
        FakeCAREamist.instances.append(self)

    def predict(self, source, tile_size, tile_overlap, batch_size, tta):
        self.calls.append(
            {
                "shape": source.shape,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "batch_size": batch_size,
                "tta": tta,
            }
        )
        return (source * 2)[None, ...]


@pytest.fixture()
def direct_careamist(monkeypatch):
    """Run ``careamics_denoise`` against an in-process fake model."""
    FakeCAREamist.instances = []
    monkeypatch.setattr(cd, "USE_DEDICATED_ENV", False)
    monkeypatch.setattr(cd, "CAREamist", FakeCAREamist, raising=False)
    return FakeCAREamist


class TestCareamicsDenoiseRegistration:
    """The processing function must stay registered with its metadata."""

    def test_registered_with_expected_metadata(self):
        info = BatchProcessingRegistry.get_function_info(
            "CAREamics Denoise (N2V/CARE)"
        )
        assert info is not None
        assert info["suffix"] == "_denoised"
        assert info["func"] is cd.careamics_denoise
        assert set(info["parameters"]) == {
            "channel",
            "checkpoint_path",
            "tile_size",
            "tile_overlap",
            "batch_size",
            "use_tta",
            "force_dedicated_env",
        }
        assert info["parameters"]["tile_size"]["default"] == "128,128,32"


class TestCareamicsDenoiseGuards:
    """Argument parsing and the missing-checkpoint guard."""

    def test_missing_checkpoint_returns_input_unchanged(self, capsys):
        image = np.arange(9, dtype=np.uint16).reshape(3, 3)
        out = cd.careamics_denoise(image, checkpoint_path="")
        assert out is image
        assert "No model checkpoint provided" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("tile_size", "expected"),
        [
            ("100,50", (100, 50, 32)),
            ("100,50,20", (100, 50, 20)),
            ("1,2,3,4", (128, 128, 32)),
            ("a,b", (128, 128, 32)),
            (None, (128, 128, 32)),
        ],
    )
    def test_tile_size_parsing(self, monkeypatch, tile_size, expected):
        captured = {}
        monkeypatch.setattr(cd, "USE_DEDICATED_ENV", True)
        monkeypatch.setattr(cd, "is_env_created", lambda: True)
        monkeypatch.setattr(
            cd,
            "run_careamics_in_env",
            lambda name, args: captured.update(args) or "done",
        )

        image = np.zeros((2, 2), dtype=np.uint8)
        assert (
            cd.careamics_denoise(
                image, checkpoint_path="/m.ckpt", tile_size=tile_size
            )
            == "done"
        )
        assert (
            captured["tile_size_x"],
            captured["tile_size_y"],
            captured["tile_size_z"],
        ) == expected

    @pytest.mark.parametrize(
        ("tile_overlap", "expected"),
        [
            ("10,5", (10, 5, 8)),
            ("10,5,2", (10, 5, 2)),
            ("1,2,3,4", (48, 48, 8)),
            ("x", (48, 48, 8)),
            (None, (48, 48, 8)),
        ],
    )
    def test_tile_overlap_parsing(self, monkeypatch, tile_overlap, expected):
        captured = {}
        monkeypatch.setattr(cd, "USE_DEDICATED_ENV", True)
        monkeypatch.setattr(cd, "is_env_created", lambda: True)
        monkeypatch.setattr(
            cd,
            "run_careamics_in_env",
            lambda name, args: captured.update(args) or "done",
        )

        image = np.zeros((2, 2), dtype=np.uint8)
        cd.careamics_denoise(
            image, checkpoint_path="/m.ckpt", tile_overlap=tile_overlap
        )
        assert (
            captured["tile_overlap_x"],
            captured["tile_overlap_y"],
            captured["tile_overlap_z"],
        ) == expected


class TestCareamicsDenoiseDedicatedEnv:
    """The dedicated-environment branch and its argument marshalling."""

    def test_creates_env_and_forwards_all_arguments(self, monkeypatch, capsys):
        image = np.ones((4, 4), dtype=np.uint16)
        created = []
        captured = {}
        monkeypatch.setattr(cd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(cd, "is_env_created", lambda: False)
        monkeypatch.setattr(
            cd, "create_careamics_env", lambda: created.append(True)
        )
        monkeypatch.setattr(
            cd,
            "run_careamics_in_env",
            lambda name, args: (
                captured.update({"name": name, "args": args}) or "denoised"
            ),
        )

        out = cd.careamics_denoise(
            image,
            checkpoint_path="/models/last.ckpt",
            tile_size="128,64,16",
            tile_overlap="48,24,4",
            batch_size=5,
            use_tta=True,
            force_dedicated_env=True,
        )

        assert out == "denoised"
        assert created == [True]
        assert captured["name"] == "predict"
        args = captured["args"]
        assert args["image"] is image
        assert args["checkpoint_path"] == "/models/last.ckpt"
        assert args["tile_size_x"] == 128
        assert args["tile_size_y"] == 64
        assert args["tile_size_z"] == 16
        assert args["tile_overlap_x"] == 48
        assert args["tile_overlap_y"] == 24
        assert args["tile_overlap_z"] == 4
        assert args["batch_size"] == 5
        assert args["use_tta"] is True
        assert cd.careamics_denoise.thread_safe is False
        assert "dedicated CAREamics environment" in capsys.readouterr().out


class TestCareamicsDenoiseDirect:
    """In-process CAREamics execution for each supported layout."""

    def test_two_dimensional_image(self, direct_careamist):
        image = np.arange(64, dtype=np.uint16).reshape(8, 8)
        out = cd.careamics_denoise(
            image,
            checkpoint_path="/m.ckpt",
            tile_size="32,16",
            tile_overlap="8,4",
        )
        np.testing.assert_array_equal(out, image * 2)
        assert out.shape == image.shape
        call = direct_careamist.instances[0].calls[0]
        assert call["tile_size"] == (16, 32)
        assert call["tile_overlap"] == (4, 8)

    def test_tyx_iterates_over_time(self, direct_careamist, capsys):
        image = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
        out = cd.careamics_denoise(image, checkpoint_path="/m.ckpt")
        np.testing.assert_array_equal(out, image * 2)
        assert out.dtype == image.dtype
        model = direct_careamist.instances[0]
        assert len(model.calls) == 3
        assert {call["shape"] for call in model.calls} == {(4, 4)}
        assert "Format: TYX" in capsys.readouterr().out

    def test_zyx_processed_in_one_shot(self, direct_careamist, capsys):
        image = np.ones((12, 4, 4), dtype=np.uint16)
        out = cd.careamics_denoise(
            image, checkpoint_path="/m.ckpt", tile_size="16,8,4"
        )
        np.testing.assert_array_equal(out, image * 2)
        model = direct_careamist.instances[0]
        assert len(model.calls) == 1
        assert model.calls[0]["tile_size"] == (4, 8, 16)
        assert "Format: 3D (ZYX" in capsys.readouterr().out

    def test_tzyx_iterates_over_time(self, direct_careamist, capsys):
        image = np.ones((2, 3, 4, 4), dtype=np.uint16)
        out = cd.careamics_denoise(image, checkpoint_path="/m.ckpt")
        np.testing.assert_array_equal(out, image * 2)
        model = direct_careamist.instances[0]
        assert len(model.calls) == 2
        assert {call["shape"] for call in model.calls} == {(3, 4, 4)}
        assert "Format: TZYX" in capsys.readouterr().out

    def test_five_dimensional_input_is_rejected(
        self, direct_careamist, capsys
    ):
        image = np.ones((2, 2, 2, 2, 2), dtype=np.uint16)
        out = cd.careamics_denoise(image, checkpoint_path="/m.ckpt")
        assert out is image
        assert "Unsupported image dimensionality" in capsys.readouterr().out

    def test_model_load_failure_returns_input(self, monkeypatch, capsys):
        def boom(checkpoint_path):
            raise OSError("corrupt checkpoint")

        monkeypatch.setattr(cd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(cd, "CAREamist", boom, raising=False)

        image = np.ones((4, 4), dtype=np.uint16)
        out = cd.careamics_denoise(image, checkpoint_path="/m.ckpt")

        assert out is image
        out_text = capsys.readouterr().out
        assert "Error loading model" in out_text
        assert "Troubleshooting" in out_text

    def test_prediction_error_falls_back_to_dedicated_env(
        self, monkeypatch, capsys
    ):
        class Exploding(FakeCAREamist):
            def predict(self, **kwargs):
                raise RuntimeError("CUDA out of memory")

        captured = {}
        created = []
        monkeypatch.setattr(cd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(cd, "CAREamist", Exploding, raising=False)
        monkeypatch.setattr(cd, "is_env_created", lambda: False)
        monkeypatch.setattr(
            cd, "create_careamics_env", lambda: created.append(True)
        )
        monkeypatch.setattr(
            cd,
            "run_careamics_in_env",
            lambda name, args: captured.update(args) or "fallback",
        )

        image = np.ones((4, 4), dtype=np.uint16)
        out = cd.careamics_denoise(image, checkpoint_path="/m.ckpt")

        assert out == "fallback"
        assert created == [True]
        assert captured["checkpoint_path"] == "/m.ckpt"
        assert "fallback to dedicated CAREamics environment" in (
            capsys.readouterr().out
        )

    def test_prediction_error_skips_env_creation_when_present(
        self, monkeypatch
    ):
        class Exploding(FakeCAREamist):
            def predict(self, **kwargs):
                raise ValueError("bad tile size")

        monkeypatch.setattr(cd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(cd, "CAREamist", Exploding, raising=False)
        monkeypatch.setattr(cd, "is_env_created", lambda: True)
        monkeypatch.setattr(
            cd, "create_careamics_env", _forbidden("create_careamics_env")
        )
        monkeypatch.setattr(
            cd, "run_careamics_in_env", lambda name, args: "fallback"
        )

        image = np.ones((4, 4), dtype=np.uint16)
        assert (
            cd.careamics_denoise(image, checkpoint_path="/m.ckpt")
            == "fallback"
        )

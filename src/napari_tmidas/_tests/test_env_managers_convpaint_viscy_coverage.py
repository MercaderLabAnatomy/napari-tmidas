# src/napari_tmidas/_tests/test_env_managers_convpaint_viscy_coverage.py
"""Coverage tests for the convpaint and VisCy environment managers.

Neither dedicated virtual environment may be built during a test run, so
every test here swaps the target module's ``subprocess`` handle for a
stub and redirects ``tempfile`` into ``tmp_path``.  Nothing shells out,
nothing downloads, and nothing touches ``~/.napari-tmidas``.
"""

import os
import re
import subprocess
import sys
import tempfile
import types
import urllib.request

import numpy as np
import pytest
import tifffile

import napari_tmidas.processing_functions.convpaint_env_manager as cpm
import napari_tmidas.processing_functions.viscy_env_manager as vem

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


class FakeCompleted:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SubprocessStub:
    """Stand-in for the ``subprocess`` module inside a target module.

    Records every ``run``/``check_call`` invocation so tests can assert
    on the argv the module actually builds.
    """

    CalledProcessError = subprocess.CalledProcessError
    SubprocessError = subprocess.SubprocessError
    CompletedProcess = subprocess.CompletedProcess
    DEVNULL = subprocess.DEVNULL

    def __init__(self, run=None, check_call=None):
        self.run_calls = []
        self.check_call_calls = []
        self._run = run
        self._check_call = check_call

    def run(self, cmd, **kwargs):
        self.run_calls.append((cmd, kwargs))
        if self._run is None:
            return FakeCompleted()
        return self._run(cmd, **kwargs)

    def check_call(self, cmd, **kwargs):
        self.check_call_calls.append((cmd, kwargs))
        if self._check_call is not None:
            return self._check_call(cmd, **kwargs)
        return 0

    @property
    def check_call_argvs(self):
        return [cmd for cmd, _ in self.check_call_calls]


def make_fake_torch(cuda=True, device_count=1, name="FakeGPU"):
    """Build a stub ``torch`` module for the CUDA-probe branches."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        device_count=lambda: device_count,
        get_device_name=lambda idx: name,
    )
    torch.__version__ = "0.0.0-fake"
    return torch


def make_env_python(env_dir):
    """Create the ``bin/python`` marker ``is_env_created`` looks for."""
    bindir = os.path.join(env_dir, "bin")
    os.makedirs(bindir, exist_ok=True)
    env_python = os.path.join(bindir, "python")
    with open(env_python, "w") as handle:
        handle.write("#!/bin/false\n")
    return env_python


@pytest.fixture()
def temp_root(tmp_path, monkeypatch):
    """Point ``tempfile`` at ``tmp_path`` for the whole test."""
    root = tmp_path / "systmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture()
def cp_manager(tmp_path):
    """A convpaint manager whose env lives under ``tmp_path``."""
    manager = cpm.ConvpaintEnvironmentManager(cuda_version="cpu")
    manager.env_dir = str(tmp_path / "envs" / "convpaint")
    return manager


@pytest.fixture()
def viscy_manager(tmp_path, monkeypatch):
    """Redirect the viscy singleton at ``tmp_path`` and hand it back."""
    manager = vem._viscy_env_manager
    env_dir = str(tmp_path / "envs" / "viscy")
    monkeypatch.setattr(manager, "env_dir", env_dir)
    monkeypatch.setattr(manager, "model_dir", os.path.join(env_dir, "models"))
    return manager


# ---------------------------------------------------------------------
# convpaint: generated script text
# ---------------------------------------------------------------------


class TestConvpaintScriptBuilders:
    """The helper-script text is what the child interpreter executes."""

    def test_import_statement_has_both_fallbacks(self):
        stmt = cpm._get_convpaint_import_statement()
        assert "from napari_convpaint.convpaint_model import" in stmt
        assert "except ImportError:" in stmt
        # dedented and stripped: no leading blank line / indentation
        assert stmt.startswith("try:")
        assert not stmt.endswith("\n")

    def test_check_script_is_valid_python_and_probes_cuda(self):
        script = cpm._build_convpaint_check_script()
        compile(script, "<check>", "exec")
        assert "ConvpaintModel imported successfully" in script
        assert "SUCCESS: napari-convpaint environment is working" in script
        assert "torch.cuda.is_available()" in script
        # the f-string braces must survive as literal format calls
        assert "{torch.__version__}" in script
        assert "sys.exit(1)" in script


# ---------------------------------------------------------------------
# convpaint: CUDA detection
# ---------------------------------------------------------------------


class TestDetectCudaVersion:
    """nvidia-smi output is mapped onto a PyTorch wheel index."""

    @pytest.mark.parametrize(
        ("reported", "expected"),
        [
            ("13.0", "cu130"),
            ("12.6", "cu124"),
            ("12.4", "cu124"),
            ("12.2", "cu121"),
            ("12.0", "cu118"),
            ("11.8", "cu118"),
        ],
    )
    def test_version_mapping(
        self, cp_manager, monkeypatch, reported, expected
    ):
        stdout = f"| NVIDIA-SMI 550.0  Driver  CUDA Version: {reported}   |\n"
        stub = SubprocessStub(
            run=lambda cmd, **kw: FakeCompleted(0, stdout=stdout)
        )
        monkeypatch.setattr(cpm, "subprocess", stub)
        assert cp_manager._detect_cuda_version() == expected
        assert stub.run_calls[0][0] == ["nvidia-smi"]
        assert stub.run_calls[0][1]["timeout"] == 5

    def test_unknown_major_falls_back_to_cu124(self, cp_manager, monkeypatch):
        stdout = "CUDA Version: 10.2\n"
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(run=lambda cmd, **kw: FakeCompleted(0, stdout)),
        )
        assert cp_manager._detect_cuda_version() == "cu124"

    def test_no_cuda_line_in_output_means_cpu(self, cp_manager, monkeypatch):
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(
                run=lambda cmd, **kw: FakeCompleted(0, "no gpu here\n")
            ),
        )
        assert cp_manager._detect_cuda_version() == "cpu"

    def test_unparseable_version_string_means_cpu(
        self, cp_manager, monkeypatch
    ):
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(
                run=lambda cmd, **kw: FakeCompleted(0, "CUDA Version: N/A\n")
            ),
        )
        assert cp_manager._detect_cuda_version() == "cpu"

    def test_nonzero_returncode_means_cpu(self, cp_manager, monkeypatch):
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(
                run=lambda cmd, **kw: FakeCompleted(1, "CUDA Version: 12.4\n")
            ),
        )
        assert cp_manager._detect_cuda_version() == "cpu"

    def test_missing_nvidia_smi_means_cpu(
        self, cp_manager, monkeypatch, capsys
    ):
        def boom(cmd, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(cpm, "subprocess", SubprocessStub(run=boom))
        assert cp_manager._detect_cuda_version() == "cpu"
        assert "Could not detect CUDA version" in capsys.readouterr().out


# ---------------------------------------------------------------------
# convpaint: dependency installation
# ---------------------------------------------------------------------


class TestConvpaintInstallDependencies:
    """Which pip argv the manager builds for each CUDA decision."""

    @staticmethod
    def _install(manager, monkeypatch, stub):
        monkeypatch.setattr(cpm, "subprocess", stub)
        monkeypatch.setattr(
            manager, "_verify_installation", lambda env_python: None
        )
        manager._install_dependencies("/fake/env/bin/python")

    def test_auto_detect_with_cuda_torch_uses_index_url(
        self, cp_manager, monkeypatch, capsys
    ):
        cp_manager.cuda_version = "auto"
        monkeypatch.setattr(
            cp_manager, "_detect_cuda_version", lambda: "cu130"
        )
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=True))
        stub = SubprocessStub()
        self._install(cp_manager, monkeypatch, stub)

        first = stub.check_call_argvs[0]
        assert "torch" in first and "torchvision" in first
        assert first[-1] == "https://download.pytorch.org/whl/cu130"
        out = capsys.readouterr().out
        assert "Auto-detected CUDA version: cu130" in out
        assert "CUDA is available" in out

    def test_explicit_cpu_skips_index_url(self, cp_manager, monkeypatch):
        cp_manager.cuda_version = "cpu"
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=True))
        stub = SubprocessStub()
        self._install(cp_manager, monkeypatch, stub)

        argvs = stub.check_call_argvs
        assert argvs[0][-2:] == ["torch", "torchvision"]
        assert not any("--index-url" in argv for argv in argvs)

    def test_torch_absent_still_installs_cuda_wheels(
        self, cp_manager, monkeypatch, capsys
    ):
        cp_manager.cuda_version = "cu124"
        monkeypatch.setitem(sys.modules, "torch", None)
        stub = SubprocessStub()
        self._install(cp_manager, monkeypatch, stub)

        assert "PyTorch not detected in main environment" in (
            capsys.readouterr().out
        )
        assert stub.check_call_argvs[0][-1].endswith("/cu124")

    def test_torch_without_cuda_downgrades_to_cpu_wheels(
        self, cp_manager, monkeypatch
    ):
        cp_manager.cuda_version = "cu124"
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=False))
        stub = SubprocessStub()
        self._install(cp_manager, monkeypatch, stub)
        assert not any("--index-url" in argv for argv in stub.check_call_argvs)

    def test_installs_convpaint_qt_and_imaging_stack(
        self, cp_manager, monkeypatch
    ):
        cp_manager.cuda_version = "cpu"
        monkeypatch.setitem(sys.modules, "torch", None)
        stub = SubprocessStub()
        self._install(cp_manager, monkeypatch, stub)

        packages = [argv[4:] for argv in stub.check_call_argvs]
        assert ["napari-convpaint"] in packages
        assert ["PyQt5"] in packages
        assert ["tifffile", "numpy", "scikit-image"] in packages
        for argv in stub.check_call_argvs:
            assert argv[:4] == [
                "/fake/env/bin/python",
                "-m",
                "pip",
                "install",
            ]

    def test_verification_runs_last(self, cp_manager, monkeypatch):
        cp_manager.cuda_version = "cpu"
        monkeypatch.setitem(sys.modules, "torch", None)
        seen = []
        monkeypatch.setattr(cpm, "subprocess", SubprocessStub())
        monkeypatch.setattr(cp_manager, "_verify_installation", seen.append)
        cp_manager._install_dependencies("/fake/env/bin/python")
        assert seen == ["/fake/env/bin/python"]


# ---------------------------------------------------------------------
# convpaint: verification / repair / ensure_env_ready
# ---------------------------------------------------------------------


class TestConvpaintVerification:
    """The smoke test writes a temp script, runs it, then deletes it."""

    def test_verify_installation_success_path(
        self, cp_manager, monkeypatch, temp_root, capsys
    ):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = list(cmd)
            seen["kwargs"] = kwargs
            with open(cmd[1]) as handle:
                seen["script"] = handle.read()
            return FakeCompleted(0, "SUCCESS: napari-convpaint ok\n")

        monkeypatch.setattr(cpm, "subprocess", SubprocessStub(run=fake_run))
        cp_manager._verify_installation("/fake/py")

        assert seen["cmd"][0] == "/fake/py"
        assert seen["kwargs"]["check"] is True
        assert "ConvpaintModel" in seen["script"]
        assert "created and verified successfully" in capsys.readouterr().out
        # the temp script is removed even on the happy path
        assert not os.path.exists(seen["cmd"][1])
        assert list(temp_root.iterdir()) == []

    def test_verify_installation_uncertain_when_no_success_token(
        self, cp_manager, monkeypatch, temp_root, capsys
    ):
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(run=lambda cmd, **kw: FakeCompleted(0, "hmm\n")),
        )
        cp_manager._verify_installation("/fake/py")
        assert "verification uncertain" in capsys.readouterr().out

    def test_verify_installation_deletes_script_on_failure(
        self, cp_manager, monkeypatch, temp_root
    ):
        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(cpm, "subprocess", SubprocessStub(run=boom))
        with pytest.raises(subprocess.CalledProcessError):
            cp_manager._verify_installation("/fake/py")
        assert list(temp_root.iterdir()) == []

    def test_run_verification_returns_completed_process(
        self, cp_manager, monkeypatch, temp_root
    ):
        scripts = []

        def fake_run(cmd, **kwargs):
            with open(cmd[1]) as handle:
                scripts.append(handle.read())
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True
            assert "check" not in kwargs
            return FakeCompleted(3, "out", "err")

        monkeypatch.setattr(cpm, "subprocess", SubprocessStub(run=fake_run))
        result = cp_manager._run_verification("/fake/py")

        assert result.returncode == 3
        assert result.stderr == "err"
        assert "SUCCESS: napari-convpaint" in scripts[0]
        assert list(temp_root.iterdir()) == []

    def test_repair_environment_installs_pyqt5(
        self, cp_manager, monkeypatch, capsys
    ):
        stub = SubprocessStub()
        monkeypatch.setattr(cpm, "subprocess", stub)
        cp_manager._repair_environment("/fake/py")
        assert stub.check_call_argvs == [
            ["/fake/py", "-m", "pip", "install", "PyQt5"]
        ]
        assert "Repairing napari-convpaint" in capsys.readouterr().out


class TestConvpaintEnsureEnvReady:
    """Create / verify / repair / recreate decision table."""

    def test_missing_env_is_created(self, cp_manager, monkeypatch, capsys):
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: "/created/python",
        )
        assert cp_manager.ensure_env_ready() == "/created/python"
        assert "Creating dedicated" in capsys.readouterr().out

    def test_healthy_env_is_reused(self, cp_manager, monkeypatch):
        env_python = make_env_python(cp_manager.env_dir)
        monkeypatch.setattr(
            cp_manager, "_run_verification", lambda py: FakeCompleted(0)
        )
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: pytest.fail("must not recreate a healthy env"),
        )
        assert cp_manager.ensure_env_ready() == env_python

    def test_repair_rescues_a_broken_env(
        self, cp_manager, monkeypatch, capsys
    ):
        env_python = make_env_python(cp_manager.env_dir)
        results = [
            FakeCompleted(1, "stdout detail", "stderr detail"),
            FakeCompleted(0),
        ]
        repaired = []
        monkeypatch.setattr(
            cp_manager, "_run_verification", lambda py: results.pop(0)
        )
        monkeypatch.setattr(cp_manager, "_repair_environment", repaired.append)
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: pytest.fail("repair succeeded, must not recreate"),
        )

        assert cp_manager.ensure_env_ready() == env_python
        assert repaired == [env_python]
        out = capsys.readouterr().out
        assert "failed verification" in out
        assert "stdout detail" in out
        assert "stderr detail" in out
        assert "repaired successfully" in out

    def test_failed_repair_recreates_the_env(
        self, cp_manager, monkeypatch, capsys
    ):
        make_env_python(cp_manager.env_dir)
        monkeypatch.setattr(
            cp_manager, "_run_verification", lambda py: FakeCompleted(1)
        )
        monkeypatch.setattr(cp_manager, "_repair_environment", lambda py: None)
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: "/recreated/python",
        )
        assert cp_manager.ensure_env_ready() == "/recreated/python"
        assert "Repair attempt failed" in capsys.readouterr().out


# ---------------------------------------------------------------------
# convpaint: module-level wrappers
# ---------------------------------------------------------------------


class TestConvpaintModuleFunctions:
    """The thin module-level wrappers delegate to the singleton."""

    def test_is_package_installed_reports_absent_convpaint(self):
        # napari-convpaint is genuinely not installed in this env
        assert cpm.is_convpaint_installed() is False

    def test_is_package_installed_swallows_import_error(self, monkeypatch):
        import importlib.util as importlib_util

        def boom(name):
            raise ImportError(name)

        monkeypatch.setattr(importlib_util, "find_spec", boom)
        assert cpm.manager.is_package_installed() is False

    def test_is_package_installed_reports_present_package(self, monkeypatch):
        # The "found" branch (find_spec(...) is not None) is otherwise never
        # exercised: both other tests in this trio expect False, so a
        # regression that hardcoded "return False" would slip past them.
        import importlib.util as importlib_util

        monkeypatch.setattr(
            importlib_util, "find_spec", lambda name: object()
        )
        assert cpm.manager.is_package_installed() is True

    def test_is_env_created_and_python_path(self, monkeypatch, tmp_path):
        env_dir = str(tmp_path / "convpaint")
        monkeypatch.setattr(cpm.manager, "env_dir", env_dir)
        assert cpm.is_env_created() is False
        expected = make_env_python(env_dir)
        assert cpm.get_env_python_path() == expected
        assert cpm.is_env_created() is True

    def test_ensure_convpaint_env_ready_delegates(self, monkeypatch):
        monkeypatch.setattr(
            cpm.manager, "ensure_env_ready", lambda: "/singleton/python"
        )
        assert cpm.ensure_convpaint_env_ready() == "/singleton/python"

    def test_create_env_auto_uses_the_singleton(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: seen.append(self) or "/py",
        )
        assert cpm.create_convpaint_env() == "/py"
        assert seen == [cpm.manager]

    def test_create_env_with_explicit_cuda_builds_new_manager(
        self, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: seen.append(self.cuda_version) or "/py",
        )
        assert cpm.create_convpaint_env(cuda_version="cu130") == "/py"
        assert seen == ["cu130"]
        assert cpm.manager.cuda_version == "auto"

    def test_recreate_env_auto_uses_the_singleton(self, monkeypatch, capsys):
        seen = []
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: seen.append(self) or "/py",
        )
        assert cpm.recreate_convpaint_env() == "/py"
        assert seen == [cpm.manager]
        assert "Recreating napari-convpaint" in capsys.readouterr().out

    def test_recreate_env_with_explicit_cuda(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            cpm.ConvpaintEnvironmentManager,
            "create_env",
            lambda self: seen.append(self.cuda_version) or "/py",
        )
        assert cpm.recreate_convpaint_env(cuda_version="cpu") == "/py"
        assert seen == ["cpu"]


# ---------------------------------------------------------------------
# convpaint: GPU enumeration
# ---------------------------------------------------------------------


class TestDetectGpuIds:
    """CUDA_VISIBLE_DEVICES wins over nvidia-smi; UUIDs pin nothing."""

    def test_visible_devices_are_parsed_in_order(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 0 ,1")
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(
                run=lambda *a, **k: pytest.fail("nvidia-smi must not run")
            ),
        )
        assert cpm.detect_gpu_ids() == [2, 0, 1]

    def test_empty_visible_devices_pins_nothing(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert cpm.detect_gpu_ids() == []

    def test_uuid_visible_devices_pins_nothing(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,GPU-deadbeef")
        assert cpm.detect_gpu_ids() == []

    def test_nvidia_smi_indices_are_parsed(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(
                run=lambda cmd, **kw: FakeCompleted(0, "0\n1\n\nweird\n2\n")
            ),
        )
        assert cpm.detect_gpu_ids() == [0, 1, 2]

    def test_nvidia_smi_failure_returns_empty(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            cpm,
            "subprocess",
            SubprocessStub(run=lambda cmd, **kw: FakeCompleted(9, "0\n")),
        )
        assert cpm.detect_gpu_ids() == []

    @pytest.mark.parametrize(
        "exc",
        [OSError("no binary"), subprocess.TimeoutExpired("nvidia-smi", 10)],
    )
    def test_nvidia_smi_exceptions_return_empty(
        self, monkeypatch, exc, capsys
    ):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        def boom(cmd, **kwargs):
            raise exc

        monkeypatch.setattr(cpm, "subprocess", SubprocessStub(run=boom))
        assert cpm.detect_gpu_ids() == []
        assert "Could not enumerate GPUs" in capsys.readouterr().out


# ---------------------------------------------------------------------
# convpaint: the tiff staging round-trip
# ---------------------------------------------------------------------


def _convpaint_paths_from_script(script):
    """Pull the staged input/output tiff paths out of a generated script."""
    in_match = re.search(r'tifffile\.imread\("([^"]+)"\)', script)
    out_match = re.search(r'tifffile\.imwrite\("([^"]+)"', script)
    assert in_match and out_match
    return in_match.group(1), out_match.group(1)


class TestRunConvpaintInEnv:
    """Stage a tiff, generate a script, run it, read the result back."""

    @staticmethod
    def _arrange(monkeypatch, labels, returncode=0, stdout="ran\n"):
        monkeypatch.setattr(
            cpm, "ensure_convpaint_env_ready", lambda: "/fake/env/bin/python"
        )
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = kwargs.get("env")
            with open(cmd[1]) as handle:
                script = handle.read()
            captured["script"] = script
            in_path, out_path = _convpaint_paths_from_script(script)
            captured["input_path"] = in_path
            captured["output_path"] = out_path
            captured["staged"] = tifffile.imread(in_path)
            if returncode == 0 and labels is not None:
                tifffile.imwrite(out_path, labels)
            return FakeCompleted(returncode, stdout, "boom\n")

        monkeypatch.setattr(cpm, "subprocess", SubprocessStub(run=fake_run))
        return captured

    def test_round_trip_returns_the_child_output(
        self, monkeypatch, tmp_path, capsys
    ):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(8, 6), dtype=np.uint8)
        labels = np.arange(48, dtype=np.uint32).reshape(8, 6)
        work = tmp_path / "work"
        captured = self._arrange(monkeypatch, labels)

        result = cpm.run_convpaint_in_env(
            image,
            "/models/demo.pkl",
            image_downsample=3,
            tmp_dir=str(work),
        )

        np.testing.assert_array_equal(result, labels)
        np.testing.assert_array_equal(captured["staged"], image)
        assert captured["cmd"][0] == "/fake/env/bin/python"
        assert work.is_dir()
        assert captured["input_path"].startswith(str(work))
        assert "ran" in capsys.readouterr().out

    def test_generated_script_carries_the_parameters(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((4, 4), dtype=np.uint16)
        captured = self._arrange(
            monkeypatch, np.zeros((4, 4), dtype=np.uint32)
        )
        cpm.run_convpaint_in_env(
            image,
            "/models/demo.pkl",
            image_downsample=4,
            use_cpu=True,
            tmp_dir=str(tmp_path / "w"),
        )

        script = captured["script"]
        compile(script, "<convpaint>", "exec")
        assert 'ConvpaintModel(model_path="/models/demo.pkl")' in script
        assert "use_cpu = True" in script
        assert "image_downsample=4" in script
        assert "if 4 > 1:" in script
        assert captured["output_path"].endswith("_output.tif")

    def test_use_cpu_blanks_cuda_visible_devices(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        captured = self._arrange(
            monkeypatch, np.zeros((2, 2), dtype=np.uint32)
        )
        cpm.run_convpaint_in_env(
            np.zeros((2, 2), dtype=np.uint8),
            "/m.pkl",
            use_cpu=True,
            tmp_dir=str(tmp_path / "w"),
        )
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""

    def test_gpu_id_pins_the_child(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        captured = self._arrange(
            monkeypatch, np.zeros((2, 2), dtype=np.uint32)
        )
        cpm.run_convpaint_in_env(
            np.zeros((2, 2), dtype=np.uint8),
            "/m.pkl",
            gpu_id=3,
            tmp_dir=str(tmp_path / "w"),
        )
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"

    def test_no_gpu_id_leaves_the_inherited_environment(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
        captured = self._arrange(
            monkeypatch, np.zeros((2, 2), dtype=np.uint32)
        )
        cpm.run_convpaint_in_env(
            np.zeros((2, 2), dtype=np.uint8),
            "/m.pkl",
            tmp_dir=str(tmp_path / "w"),
        )
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "7"

    def test_nonzero_exit_raises_and_cleans_up(self, monkeypatch, tmp_path):
        work = tmp_path / "w"
        captured = self._arrange(monkeypatch, None, returncode=2)
        with pytest.raises(RuntimeError) as excinfo:
            cpm.run_convpaint_in_env(
                np.zeros((2, 2), dtype=np.uint8),
                "/m.pkl",
                tmp_dir=str(work),
            )
        message = str(excinfo.value)
        assert "Convpaint processing failed" in message
        assert "boom" in message
        assert not os.path.exists(captured["output_path"])
        assert not os.path.exists(captured["input_path"])
        assert list(work.iterdir()) == []

    def test_temp_files_removed_after_success(self, monkeypatch, tmp_path):
        work = tmp_path / "w"
        self._arrange(monkeypatch, np.zeros((2, 2), dtype=np.uint32))
        cpm.run_convpaint_in_env(
            np.zeros((2, 2), dtype=np.uint8), "/m.pkl", tmp_dir=str(work)
        )
        assert list(work.iterdir()) == []

    def test_missing_temp_files_do_not_mask_the_result(
        self, monkeypatch, tmp_path
    ):
        """Cleanup swallows the OSError when the child removed the input."""
        labels = np.ones((2, 2), dtype=np.uint32)
        captured = self._arrange(monkeypatch, labels, stdout="")
        work = tmp_path / "w"
        original = cpm.subprocess.run

        def wrapper(cmd, **kwargs):
            result = original(cmd, **kwargs)
            os.unlink(captured["input_path"])
            return result

        monkeypatch.setattr(cpm.subprocess, "run", wrapper)
        result = cpm.run_convpaint_in_env(
            np.zeros((2, 2), dtype=np.uint8), "/m.pkl", tmp_dir=str(work)
        )
        np.testing.assert_array_equal(result, labels)

    def test_system_tempdir_is_used_when_no_tmp_dir(
        self, monkeypatch, temp_root
    ):
        labels = np.full((3, 3), 5, dtype=np.uint32)
        captured = self._arrange(monkeypatch, labels)
        result = cpm.run_convpaint_in_env(
            np.zeros((3, 3), dtype=np.uint8), "/m.pkl"
        )
        np.testing.assert_array_equal(result, labels)
        assert captured["input_path"].startswith(str(temp_root))
        assert list(temp_root.iterdir()) == []


# ---------------------------------------------------------------------
# viscy: dependency installation
# ---------------------------------------------------------------------


class TestViscyInstallDependencies:
    """CUDA probing decides between the cu118 and cpu wheel indices."""

    @staticmethod
    def _install(manager, monkeypatch, stub):
        monkeypatch.setattr(vem, "subprocess", stub)
        downloaded = []
        monkeypatch.setattr(
            manager, "_download_model", lambda: downloaded.append(True)
        )
        manager._install_dependencies("/fake/env/bin/python")
        return downloaded

    def test_cuda_torch_and_passing_probe_keeps_cu118(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=True))
        stub = SubprocessStub(run=lambda cmd, **kw: FakeCompleted(0, "ok"))
        downloaded = self._install(viscy_manager, monkeypatch, stub)

        argvs = stub.check_call_argvs
        assert argvs[0][-1] == "https://download.pytorch.org/whl/cu118"
        assert not any(argv[-1].endswith("/whl/cpu") for argv in argvs)
        assert argvs[-1][4:] == ["viscy", "iohub", "tifffile", "numpy"]
        assert downloaded == [True]
        out = capsys.readouterr().out
        assert "CUDA is available in main environment" in out
        assert "GPU detected: FakeGPU" in out

    def test_zero_device_count_skips_the_name_lookup(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(
            sys.modules, "torch", make_fake_torch(cuda=True, device_count=0)
        )
        stub = SubprocessStub(run=lambda cmd, **kw: FakeCompleted(0))
        self._install(viscy_manager, monkeypatch, stub)
        assert "GPU detected:" not in capsys.readouterr().out

    def test_failing_cuda_probe_falls_back_to_cpu(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=True))
        stub = SubprocessStub(
            run=lambda cmd, **kw: FakeCompleted(1, "bad", "worse")
        )
        self._install(viscy_manager, monkeypatch, stub)

        argvs = stub.check_call_argvs
        assert argvs[0][-1].endswith("/whl/cu118")
        assert argvs[1][-1].endswith("/whl/cpu")
        out = capsys.readouterr().out
        assert "falling back to CPU-only installation" in out
        assert "worse" in out
        # the probe script really is Python
        compile(stub.run_calls[0][0][2], "<probe>", "exec")

    def test_cuda_install_error_falls_back_to_cpu(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=True))

        def check_call(cmd, **kwargs):
            if "cu118" in cmd[-1]:
                raise subprocess.CalledProcessError(1, cmd)
            return 0

        stub = SubprocessStub(check_call=check_call)
        self._install(viscy_manager, monkeypatch, stub)
        assert stub.check_call_argvs[1][-1].endswith("/whl/cpu")
        assert "PyTorch CUDA installation failed" in capsys.readouterr().out

    def test_torch_without_cuda_installs_cpu_wheels(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda=False))
        stub = SubprocessStub()
        self._install(viscy_manager, monkeypatch, stub)
        assert stub.check_call_argvs[0][-1].endswith("/whl/cpu")
        assert "CUDA is not available" in capsys.readouterr().out

    def test_no_torch_but_nvidia_smi_present_means_cuda(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", None)
        stub = SubprocessStub(run=lambda cmd, **kw: FakeCompleted(0))
        self._install(viscy_manager, monkeypatch, stub)
        assert stub.run_calls[0][0] == ["nvidia-smi"]
        assert stub.check_call_argvs[0][-1].endswith("/whl/cu118")
        assert "NVIDIA GPU detected via nvidia-smi" in capsys.readouterr().out

    def test_no_torch_and_failing_nvidia_smi_means_cpu(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", None)
        stub = SubprocessStub(run=lambda cmd, **kw: FakeCompleted(1))
        self._install(viscy_manager, monkeypatch, stub)
        assert stub.check_call_argvs[0][-1].endswith("/whl/cpu")
        assert "No NVIDIA GPU detected" in capsys.readouterr().out

    def test_no_torch_and_no_nvidia_smi_binary_means_cpu(
        self, viscy_manager, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", None)

        def boom(cmd, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        stub = SubprocessStub(run=boom)
        self._install(viscy_manager, monkeypatch, stub)
        assert stub.check_call_argvs[0][-1].endswith("/whl/cpu")
        assert "nvidia-smi not found" in capsys.readouterr().out


class TestViscyDownloadModel:
    """The VSCyto3D checkpoint download is best-effort."""

    def test_downloads_when_missing(self, viscy_manager, monkeypatch, capsys):
        seen = {}

        def fake_urlretrieve(url, path):
            seen["url"] = url
            seen["path"] = path
            with open(path, "wb") as handle:
                handle.write(b"ckpt")

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
        viscy_manager._download_model()

        assert seen["url"].startswith("https://public.czbiohub.org/")
        assert seen["path"] == viscy_manager.get_model_path()
        assert os.path.exists(seen["path"])
        assert "Model checkpoint downloaded" in capsys.readouterr().out

    def test_existing_checkpoint_is_not_redownloaded(
        self, viscy_manager, monkeypatch, capsys
    ):
        os.makedirs(viscy_manager.model_dir, exist_ok=True)
        with open(viscy_manager.get_model_path(), "wb") as handle:
            handle.write(b"already here")

        def fail(url, path):
            pytest.fail("must not re-download an existing checkpoint")

        monkeypatch.setattr(urllib.request, "urlretrieve", fail)
        viscy_manager._download_model()
        assert "already exists" in capsys.readouterr().out

    def test_download_failure_is_reported_not_raised(
        self, viscy_manager, monkeypatch, capsys
    ):
        def boom(url, path):
            raise OSError("network down")

        monkeypatch.setattr(urllib.request, "urlretrieve", boom)
        viscy_manager._download_model()

        out = capsys.readouterr().out
        assert "Failed to download model checkpoint" in out
        assert "network down" in out
        assert not os.path.exists(viscy_manager.get_model_path())


class TestViscyPackageChecks:
    """``is_package_installed`` needs the env before it probes it."""

    def test_missing_env_short_circuits(self, viscy_manager, monkeypatch):
        monkeypatch.setattr(
            vem,
            "subprocess",
            SubprocessStub(
                check_call=lambda *a, **k: pytest.fail("no env to probe")
            ),
        )
        assert vem.is_env_created() is False
        assert vem.is_viscy_installed() is False

    def test_import_probe_success(self, viscy_manager, monkeypatch):
        env_python = make_env_python(viscy_manager.env_dir)
        stub = SubprocessStub()
        monkeypatch.setattr(vem, "subprocess", stub)

        assert vem.is_env_created() is True
        assert vem.is_viscy_installed() is True
        cmd, kwargs = stub.check_call_calls[0]
        assert cmd == [env_python, "-c", "import viscy"]
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL

    def test_import_probe_failure(self, viscy_manager, monkeypatch):
        make_env_python(viscy_manager.env_dir)

        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(vem, "subprocess", SubprocessStub(check_call=boom))
        assert vem.is_viscy_installed() is False

    def test_model_path_lives_under_the_env(self, viscy_manager):
        expected = os.path.join(viscy_manager.model_dir, "VSCyto3D.ckpt")
        assert vem.get_model_path() == expected

    def test_create_viscy_env_delegates(self, monkeypatch):
        # Assert identity, not just the return value: a class-level patch
        # of create_env would return "/viscy/python" for *any* instance, so
        # only capturing `self` proves the module function reaches the
        # singleton rather than building a throwaway manager.
        seen = []
        monkeypatch.setattr(
            vem.ViscyEnvironmentManager,
            "create_env",
            lambda self: seen.append(self) or "/viscy/python",
        )
        assert vem.create_viscy_env() == "/viscy/python"
        assert seen == [vem._viscy_env_manager]


# ---------------------------------------------------------------------
# viscy: run_viscy_in_env
# ---------------------------------------------------------------------


class TestRunViscyInEnv:
    """Guard clauses, the generated script, and the tiff round-trip."""

    @staticmethod
    def _ready(viscy_manager, with_model=True):
        env_python = make_env_python(viscy_manager.env_dir)
        if with_model:
            os.makedirs(viscy_manager.model_dir, exist_ok=True)
            with open(viscy_manager.get_model_path(), "wb") as handle:
                handle.write(b"ckpt")
        return env_python

    def test_missing_env_raises(self, viscy_manager, monkeypatch):
        monkeypatch.setattr(vem, "subprocess", SubprocessStub())
        with pytest.raises(RuntimeError, match="environment not created"):
            vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))

    def test_env_without_viscy_raises(self, viscy_manager, monkeypatch):
        make_env_python(viscy_manager.env_dir)

        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(vem, "subprocess", SubprocessStub(check_call=boom))
        with pytest.raises(RuntimeError, match="not installed in environment"):
            vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))

    def test_missing_checkpoint_raises(self, viscy_manager, monkeypatch):
        self._ready(viscy_manager, with_model=False)
        monkeypatch.setattr(vem, "subprocess", SubprocessStub())
        with pytest.raises(RuntimeError, match="Model checkpoint not found"):
            vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))

    def test_missing_tifffile_raises_import_error(
        self, viscy_manager, monkeypatch, temp_root
    ):
        self._ready(viscy_manager)
        monkeypatch.setattr(vem, "subprocess", SubprocessStub())
        monkeypatch.setattr(vem, "tifffile", None)
        with pytest.raises(ImportError, match="tifffile is required"):
            vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))

    def _arrange(self, viscy_manager, monkeypatch, returncode=0, output=None):
        env_python = self._ready(viscy_manager)
        captured = {"env_python": env_python}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            with open(cmd[1]) as handle:
                script = handle.read()
            captured["script"] = script
            in_match = re.search(r'tifffile\.imread\("([^"]+)"\)', script)
            out_match = re.search(r'tifffile\.imwrite\("([^"]+)"', script)
            captured["input_path"] = in_match.group(1)
            captured["output_path"] = out_match.group(1)
            captured["staged"] = tifffile.imread(in_match.group(1))
            if returncode == 0 and output is not None:
                tifffile.imwrite(out_match.group(1), output)
            return FakeCompleted(returncode, "out\n", "err\n")

        monkeypatch.setattr(vem, "subprocess", SubprocessStub(run=fake_run))
        return captured

    def test_round_trip(self, viscy_manager, monkeypatch, temp_root):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 4096, size=(6, 5, 4)).astype(np.uint16)
        stained = rng.random((6, 2, 5, 4)).astype(np.float32)
        captured = self._arrange(viscy_manager, monkeypatch, output=stained)

        result = vem.run_viscy_in_env(image, z_batch_size=3)

        assert result.shape == (6, 2, 5, 4)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, stained)
        np.testing.assert_array_equal(captured["staged"], image)
        assert captured["cmd"][0] == captured["env_python"]

    def test_generated_script_carries_model_and_batch_size(
        self, viscy_manager, monkeypatch, temp_root
    ):
        captured = self._arrange(
            viscy_manager,
            monkeypatch,
            output=np.zeros((2, 2, 2, 2), dtype=np.float32),
        )
        vem.run_viscy_in_env(
            np.zeros((2, 2, 2), dtype=np.uint16), z_batch_size=7
        )

        script = captured["script"]
        compile(script, "<viscy>", "exec")
        assert "z_batch_size = 7" in script
        assert vem.get_model_path() in script
        assert "from viscy.translation.engine import VSUNet" in script
        assert '"in_stack_depth": 15' in script

    def test_nonzero_exit_raises_with_both_streams(
        self, viscy_manager, monkeypatch, temp_root
    ):
        self._arrange(viscy_manager, monkeypatch, returncode=5)
        with pytest.raises(RuntimeError) as excinfo:
            vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))
        message = str(excinfo.value)
        assert "VisCy processing failed" in message
        assert "STDOUT: out" in message
        assert "STDERR: err" in message

    def test_temp_files_are_always_removed(
        self, viscy_manager, monkeypatch, temp_root
    ):
        self._arrange(
            viscy_manager,
            monkeypatch,
            output=np.zeros((2, 2, 2, 2), dtype=np.float32),
        )
        vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))
        assert list(temp_root.iterdir()) == []

    def test_tifffile_lost_before_readback_raises(
        self, viscy_manager, monkeypatch, temp_root
    ):
        """The readback guard fires if tifffile vanishes mid-run."""
        self._ready(viscy_manager)

        def fake_run(cmd, **kwargs):
            monkeypatch.setattr(vem, "tifffile", None)
            return FakeCompleted(0, "", "")

        monkeypatch.setattr(vem, "subprocess", SubprocessStub(run=fake_run))
        with pytest.raises(ImportError, match="tifffile is required"):
            vem.run_viscy_in_env(np.zeros((2, 2, 2), dtype=np.uint16))
        assert list(temp_root.iterdir()) == []

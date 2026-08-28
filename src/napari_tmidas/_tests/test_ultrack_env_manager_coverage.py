"""
Branch-coverage tests for
``napari_tmidas.processing_functions.ultrack_env_manager``.

The module shells out to conda/mamba for every operation.  Neither the
``ultrack`` conda environment nor the ``ultrack`` package exists in the
test environment, so every test here replaces the module's *collaborators*
(``shutil.which``, ``subprocess.run``, ``subprocess.Popen`` and the sibling
helpers looked up as module globals) and then exercises the real control
flow of the function under test.

Nothing ever shells out for real and no file is written outside ``tmp_path``.
"""

import subprocess

import pytest

from napari_tmidas.processing_functions import ultrack_env_manager as uem

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _flatten(cmd):
    """Render a run() command (list or shell string) as one string."""
    if isinstance(cmd, str):
        return cmd
    return " ".join(str(part) for part in cmd)


class RunRouter:
    """Dispatch ``subprocess.run`` calls to canned results by substring.

    Rules are ``(needle, FakeCompleted)`` pairs checked in order, so more
    specific needles must come first.  Every command is recorded in
    ``self.calls`` for call-argument assertions.
    """

    def __init__(self, rules=(), default=None):
        self.rules = list(rules)
        self.default = default or FakeCompleted(0, "", "")
        self.calls = []

    def __call__(self, cmd, *args, **kwargs):
        text = _flatten(cmd)
        self.calls.append(text)
        for needle, result in self.rules:
            if needle in text:
                return result
        return self.default

    def matching(self, needle):
        return [c for c in self.calls if needle in c]


class Logger:
    """Collect log messages passed to ``log_func`` / ``progress_callback``."""

    def __init__(self):
        self.messages = []

    def __call__(self, msg):
        self.messages.append(msg)

    def text(self):
        return "\n".join(self.messages)


@pytest.fixture()
def conda(monkeypatch):
    """Pin ``get_conda_cmd`` so no real conda lookup happens."""
    monkeypatch.setattr(uem, "get_conda_cmd", lambda: "conda")
    return "conda"


def _write_module(root, parts, content):
    """Create ``root/<parts...>`` with *content* and return the path."""
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


CUDA_UNPATCHED = """import numpy as np

try:
    import cupy as cp

    if not cp.cuda.is_available():
        cp = None
        LOG.info("cupy found but cuda is not available.")
    else:
        xp = cp
except ImportError:
    cp = None
"""

CUDA_PATCHED = """import numpy as np

try:
    import cupy as cp

    if not cp.cuda.is_available():
        cp = None
        xp = np
        LOG.info("cupy found but cuda is not available.")
    else:
        xp = cp
except ImportError:
    cp = None
"""

SOLVER_UNPATCHED = """    def add_edges(self, sources, targets, weights):
        sources = self._forward_map[np.asarray(sources, dtype=int)]
        targets = self._forward_map[np.asarray(targets, dtype=int)]
        return sources, targets
"""


# ---------------------------------------------------------------------------
# get_conda_cmd
# ---------------------------------------------------------------------------


class TestGetCondaCmd:
    """Pin the mamba-before-conda preference and the not-found error."""

    def test_prefers_mamba(self, monkeypatch):
        seen = []

        def which(cmd):
            seen.append(cmd)
            return "/usr/bin/mamba" if cmd == "mamba" else None

        monkeypatch.setattr(uem.shutil, "which", which)
        assert uem.get_conda_cmd() == "mamba"
        # conda is never probed once mamba is found.
        assert seen == ["mamba"]

    def test_falls_back_to_conda(self, monkeypatch):
        monkeypatch.setattr(
            uem.shutil,
            "which",
            lambda cmd: "/usr/bin/conda" if cmd == "conda" else None,
        )
        assert uem.get_conda_cmd() == "conda"

    def test_raises_when_neither_present(self, monkeypatch):
        monkeypatch.setattr(uem.shutil, "which", lambda cmd: None)
        with pytest.raises(RuntimeError, match="Neither conda nor mamba"):
            uem.get_conda_cmd()


# ---------------------------------------------------------------------------
# is_env_created
# ---------------------------------------------------------------------------


class TestIsEnvCreated:
    """Environment listing is parsed line-wise; failures degrade to False."""

    def test_true_when_name_in_env_list(self, monkeypatch, conda):
        router = RunRouter(
            default=FakeCompleted(
                0, "# conda environments:\nbase   *  /opt/c\nultrack  /opt/u\n"
            )
        )
        monkeypatch.setattr(uem.subprocess, "run", router)

        assert uem.is_env_created("ultrack") is True
        assert router.matching("conda env list")

    def test_false_when_name_absent(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(0, "base  *  /opt/c\n")),
        )
        assert uem.is_env_created("ultrack") is False

    def test_custom_env_name_is_used(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(0, "myenv  /opt/myenv\n")),
        )
        assert uem.is_env_created("myenv") is True
        assert uem.is_env_created("ultrack") is False

    def test_exception_is_swallowed(self, monkeypatch, capsys):
        def boom():
            raise RuntimeError("no conda")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        assert uem.is_env_created() is False
        assert (
            "Error checking environment: no conda" in capsys.readouterr().out
        )


# ---------------------------------------------------------------------------
# _get_cuda_version
# ---------------------------------------------------------------------------


class TestGetCudaVersion:
    """nvidia-smi output parsing and every failure mode."""

    def test_parses_version(self, monkeypatch):
        out = (
            "NVIDIA-SMI 580.00   Driver Version: 580.00   "
            "CUDA Version: 13.0  |\n"
        )
        router = RunRouter(default=FakeCompleted(0, out))
        monkeypatch.setattr(uem.subprocess, "run", router)

        assert uem._get_cuda_version() == "13.0"
        # A bare `nvidia-smi` is the contract: `-q`/`-x` output does not
        # carry the "CUDA Version:" line the regex above depends on.
        assert router.calls == ["nvidia-smi"]

    def test_returns_none_on_nonzero_returncode(self, monkeypatch):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(1, "CUDA Version: 12.4")),
        )
        assert uem._get_cuda_version() is None

    def test_returns_none_when_pattern_absent(self, monkeypatch):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(0, "no gpu info here")),
        )
        assert uem._get_cuda_version() is None

    def test_returns_none_when_nvidia_smi_missing(self, monkeypatch):
        def raiser(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(uem.subprocess, "run", raiser)
        assert uem._get_cuda_version() is None

    def test_returns_none_on_timeout(self, monkeypatch):
        def raiser(*args, **kwargs):
            raise subprocess.TimeoutExpired("nvidia-smi", 10)

        monkeypatch.setattr(uem.subprocess, "run", raiser)
        assert uem._get_cuda_version() is None


# ---------------------------------------------------------------------------
# _ensure_scikit_image_fix
# ---------------------------------------------------------------------------


class TestEnsureScikitImageFix:
    """Version gate around the read-only-array (const buffer) fix."""

    def test_returns_false_when_version_probe_fails(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess, "run", RunRouter(default=FakeCompleted(1, ""))
        )
        log = Logger()
        assert uem._ensure_scikit_image_fix("ultrack", log) is False
        assert "Could not check scikit-image version" in log.text()

    def test_recent_release_reports_fix_present(self, monkeypatch, conda):
        router = RunRouter(default=FakeCompleted(0, "0.26.1\n"))
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()

        assert uem._ensure_scikit_image_fix("myenv", log) is True
        assert "has the read-only array fix" in log.text()
        assert "Current scikit-image version: 0.26.1" in log.text()
        # The version must be read from the target env, not from the
        # interpreter running the tests.
        assert router.matching("conda run -n myenv python -c import skimage")

    def test_older_release_falls_back_to_runtime_shim(
        self, monkeypatch, conda
    ):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(0, "0.25.2\n")),
        )
        log = Logger()
        assert uem._ensure_scikit_image_fix("ultrack", log) is True
        assert "lacks the const-buffer fix" in log.text()
        assert "map_array shim" in log.text()

    def test_unparseable_version_still_reports_true(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(0, "unknown\n")),
        )
        assert uem._ensure_scikit_image_fix("ultrack", None) is True

    def test_no_log_func_is_tolerated(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(0, "0.26.1\n")),
        )
        assert uem._ensure_scikit_image_fix("ultrack") is True

    def test_dev_branch_upgrades_when_stable_available(
        self, monkeypatch, conda
    ):
        # NOTE: the 'dev' branch is only reachable with a version string that
        # does not start with a PEP440 release triple >= 0.26.1 -- see the
        # module's own re.match early-return.  A leading-text version is used
        # here purely to execute that branch.
        router = RunRouter(
            rules=[
                ("--dry-run", FakeCompleted(0, "Would install 0.26.1")),
                ("pip install --upgrade", FakeCompleted(0, "ok")),
            ],
            default=FakeCompleted(0, "dev build 0.26.1\n"),
        )
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()
        assert uem._ensure_scikit_image_fix("ultrack", log) is True
        assert "Upgraded to stable scikit-image" in log.text()
        assert router.matching(
            "pip install --dry-run --upgrade scikit-image>=0.26.1"
        )
        # The real upgrade keeps the >=0.26.1 pin (the dry-run call does
        # not contain this substring, so this matches only the upgrade).
        assert router.matching("pip install --upgrade scikit-image>=0.26.1")

    def test_dev_branch_keeps_dev_when_upgrade_fails(self, monkeypatch, conda):
        router = RunRouter(
            rules=[
                ("--dry-run", FakeCompleted(0, "Would install 0.26.1")),
                ("pip install --upgrade", FakeCompleted(1, "", "boom")),
            ],
            default=FakeCompleted(0, "dev build 0.26.1\n"),
        )
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()
        assert uem._ensure_scikit_image_fix("ultrack", log) is True
        assert "Keeping dev version" in log.text()

    def test_dev_branch_skips_upgrade_when_no_stable(self, monkeypatch, conda):
        router = RunRouter(
            rules=[("--dry-run", FakeCompleted(1, "", "nope"))],
            default=FakeCompleted(0, "dev build 0.26.1\n"),
        )
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()
        assert uem._ensure_scikit_image_fix("ultrack", log) is True
        assert "Upgraded to stable" not in log.text()
        assert not router.matching("pip install --upgrade")

    def test_exception_returns_false(self, monkeypatch):
        def boom():
            raise RuntimeError("no conda")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        log = Logger()
        assert uem._ensure_scikit_image_fix("ultrack", log) is False
        assert "Error managing scikit-image: no conda" in log.text()


# ---------------------------------------------------------------------------
# _patch_ultrack_xp
# ---------------------------------------------------------------------------


class TestPatchUltrackXp:
    """In-place edit of ultrack/utils/cuda.py for the xp=np bug."""

    @staticmethod
    def _site(monkeypatch, tmp_path):
        router = RunRouter(default=FakeCompleted(0, f"{tmp_path}\n"))
        monkeypatch.setattr(uem.subprocess, "run", router)
        return router

    def test_returns_false_when_site_packages_probe_fails(
        self, monkeypatch, conda
    ):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(1, "", "no env")),
        )
        log = Logger()
        assert uem._patch_ultrack_xp("ultrack", log) is False
        assert "Could not get site-packages path" in log.text()

    def test_returns_false_when_cuda_py_missing(
        self, monkeypatch, tmp_path, conda
    ):
        self._site(monkeypatch, tmp_path)
        log = Logger()
        assert uem._patch_ultrack_xp("ultrack", log) is False
        assert "cuda.py not found" in log.text()

    def test_applies_patch_and_rewrites_file(
        self, monkeypatch, tmp_path, conda
    ):
        path = _write_module(
            tmp_path, ("ultrack", "utils", "cuda.py"), CUDA_UNPATCHED
        )
        router = self._site(monkeypatch, tmp_path)
        log = Logger()

        assert uem._patch_ultrack_xp("myenv", log) is True

        new = path.read_text()
        assert "        cp = None\n        xp = np\n" in new
        assert new.count("xp = np") == 1
        assert "Patched ultrack/utils/cuda.py" in log.text()
        # site-packages is resolved inside the target env; probing the
        # wrong env would patch the wrong installation.
        assert router.matching("conda run -n myenv python -c import site;")

    def test_already_patched_is_a_noop(self, monkeypatch, tmp_path, conda):
        path = _write_module(
            tmp_path, ("ultrack", "utils", "cuda.py"), CUDA_PATCHED
        )
        before = path.read_text()
        self._site(monkeypatch, tmp_path)
        log = Logger()

        assert uem._patch_ultrack_xp("ultrack", log) is True
        assert path.read_text() == before
        assert "ultrack already patched" in log.text()

    def test_unknown_structure_is_reported_but_not_a_failure(
        self, monkeypatch, tmp_path, conda
    ):
        path = _write_module(
            tmp_path,
            ("ultrack", "utils", "cuda.py"),
            "import numpy as np\ncp = None\n",
        )
        self._site(monkeypatch, tmp_path)
        log = Logger()

        assert uem._patch_ultrack_xp("ultrack", log) is True
        assert path.read_text() == "import numpy as np\ncp = None\n"
        assert "code structure different than expected" in log.text()

    def test_partial_marker_still_applies_patch(
        self, monkeypatch, tmp_path, conda
    ):
        # 'xp = np' and the log message both appear, but not in the exact
        # patched block, so the patch is applied anyway.
        content = "xp = np  # elsewhere\n" + CUDA_UNPATCHED
        path = _write_module(
            tmp_path, ("ultrack", "utils", "cuda.py"), content
        )
        self._site(monkeypatch, tmp_path)

        assert uem._patch_ultrack_xp("ultrack", None) is True
        assert path.read_text().count("xp = np") == 2

    def test_exception_returns_false(self, monkeypatch):
        def boom():
            raise RuntimeError("kaput")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        log = Logger()
        assert uem._patch_ultrack_xp("ultrack", log) is False
        assert "Error patching ultrack: kaput" in log.text()


# ---------------------------------------------------------------------------
# _patch_ultrack_readonly_arrays
# ---------------------------------------------------------------------------


SOLVER_PARTS = ("ultrack", "core", "solve", "solver", "mip_solver.py")


class TestPatchUltrackReadonlyArrays:
    """In-place edit of the MIP solver for read-only zarr arrays."""

    @staticmethod
    def _site(monkeypatch, tmp_path):
        router = RunRouter(default=FakeCompleted(0, f"{tmp_path}\n"))
        monkeypatch.setattr(uem.subprocess, "run", router)
        return router

    def test_returns_false_when_site_packages_probe_fails(
        self, monkeypatch, conda
    ):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(1, "", "no env")),
        )
        log = Logger()
        assert uem._patch_ultrack_readonly_arrays("ultrack", log) is False
        assert "Could not get site-packages path" in log.text()

    def test_returns_false_when_solver_missing(
        self, monkeypatch, tmp_path, conda
    ):
        self._site(monkeypatch, tmp_path)
        log = Logger()
        assert uem._patch_ultrack_readonly_arrays("ultrack", log) is False
        assert "ultrack solver not found" in log.text()

    def test_applies_patch_and_rewrites_file(
        self, monkeypatch, tmp_path, conda
    ):
        path = _write_module(tmp_path, SOLVER_PARTS, SOLVER_UNPATCHED)
        router = self._site(monkeypatch, tmp_path)
        log = Logger()

        assert uem._patch_ultrack_readonly_arrays("myenv", log) is True
        assert router.matching("conda run -n myenv python -c import site;")

        new = path.read_text()
        assert "# PATCH: Ensure arrays are writable" in new
        assert "np.array(sources, dtype=int, copy=True)" in new
        assert "np.array(targets, dtype=int, copy=True)" in new
        assert "np.asarray(sources, dtype=int)" not in new
        assert "Patched ultrack solver" in log.text()

    def test_already_patched_is_a_noop(self, monkeypatch, tmp_path, conda):
        content = "# PATCH: Ensure arrays are writable\n" + SOLVER_UNPATCHED
        path = _write_module(tmp_path, SOLVER_PARTS, content)
        self._site(monkeypatch, tmp_path)
        log = Logger()

        assert uem._patch_ultrack_readonly_arrays("ultrack", log) is True
        assert path.read_text() == content
        assert "already patched for read-only arrays" in log.text()

    def test_unknown_structure_is_reported_but_not_a_failure(
        self, monkeypatch, tmp_path, conda
    ):
        path = _write_module(
            tmp_path, SOLVER_PARTS, "class MIPSolver:\n    pass\n"
        )
        self._site(monkeypatch, tmp_path)
        log = Logger()

        assert uem._patch_ultrack_readonly_arrays("ultrack", log) is True
        assert "# PATCH" not in path.read_text()
        assert "code structure different than expected" in log.text()

    def test_exception_returns_false(self, monkeypatch):
        def boom():
            raise RuntimeError("kaput")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        log = Logger()
        assert uem._patch_ultrack_readonly_arrays("ultrack", log) is False
        assert "Error patching ultrack solver: kaput" in log.text()


# ---------------------------------------------------------------------------
# is_package_installed
# ---------------------------------------------------------------------------


class TestIsPackageInstalled:
    """Import probe run inside the target environment."""

    def test_true_on_zero_returncode(self, monkeypatch, conda):
        router = RunRouter(default=FakeCompleted(0))
        monkeypatch.setattr(uem.subprocess, "run", router)

        assert uem.is_package_installed("zarr", "ultrack") is True
        assert router.matching("conda run -n ultrack python -c import zarr")

    def test_false_on_nonzero_returncode(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess, "run", RunRouter(default=FakeCompleted(1))
        )
        assert uem.is_package_installed("nope") is False

    def test_false_on_exception(self, monkeypatch, conda):
        def raiser(*args, **kwargs):
            raise subprocess.TimeoutExpired("conda", 10)

        monkeypatch.setattr(uem.subprocess, "run", raiser)
        assert uem.is_package_installed("zarr") is False


# ---------------------------------------------------------------------------
# check_gpu_available
# ---------------------------------------------------------------------------


class TestCheckGpuAvailable:
    """The probe script's stdout contract: SUCCESS:... vs ERROR:..."""

    def test_success_output_is_parsed(self, monkeypatch, conda):
        router = RunRouter(
            default=FakeCompleted(0, "SUCCESS:NVIDIA RTX A6000:8.6:47.5\n")
        )
        monkeypatch.setattr(uem.subprocess, "run", router)

        info = uem.check_gpu_available("myenv")
        assert info == {
            "available": True,
            "device_name": "NVIDIA RTX A6000",
            "compute_capability": "8.6",
            "memory_gb": 47.5,
        }
        assert isinstance(info["memory_gb"], float)
        # The cupy probe runs inside the target env and really is the
        # cupy availability script.
        assert router.matching("conda run -n myenv python -c")
        assert "cp.cuda.is_available()" in router.calls[0]

    def test_error_prefix_is_stripped(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(1, "ERROR:CUDA not available\n")),
        )
        info = uem.check_gpu_available()
        assert info == {"available": False, "error": "CUDA not available"}

    def test_unrecognised_output_is_passed_through(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(default=FakeCompleted(1, "garbage\n")),
        )
        assert uem.check_gpu_available() == {
            "available": False,
            "error": "garbage",
        }

    def test_exception_becomes_error_dict(self, monkeypatch):
        def boom():
            raise RuntimeError("Neither conda nor mamba found in PATH")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        info = uem.check_gpu_available()
        assert info["available"] is False
        assert "Neither conda nor mamba" in info["error"]


# ---------------------------------------------------------------------------
# setup_gurobi_license
# ---------------------------------------------------------------------------


class TestSetupGurobiLicense:
    """Optional gurobi install followed by grbgetkey activation."""

    def test_skips_install_when_already_present(self, monkeypatch, conda):
        monkeypatch.setattr(uem, "is_package_installed", lambda pkg, env: True)
        router = RunRouter(default=FakeCompleted(0))
        monkeypatch.setattr(uem.subprocess, "run", router)

        assert uem.setup_gurobi_license("KEY-123", "ultrack") is True
        assert not router.matching("install -n ultrack -c gurobi")
        assert router.matching("grbgetkey KEY-123")

    def test_installs_then_activates(self, monkeypatch, conda):
        monkeypatch.setattr(
            uem, "is_package_installed", lambda pkg, env: False
        )
        router = RunRouter(default=FakeCompleted(0))
        monkeypatch.setattr(uem.subprocess, "run", router)

        assert uem.setup_gurobi_license("KEY-123") is True
        assert router.matching("conda install -n ultrack -c gurobi gurobi -y")
        assert router.matching("grbgetkey KEY-123")

    def test_install_failure_returns_false(self, monkeypatch, conda, capsys):
        monkeypatch.setattr(
            uem, "is_package_installed", lambda pkg, env: False
        )
        router = RunRouter(
            rules=[("-c gurobi", FakeCompleted(1, "", "solver conflict"))],
            default=FakeCompleted(0),
        )
        monkeypatch.setattr(uem.subprocess, "run", router)

        assert uem.setup_gurobi_license("KEY-123") is False
        assert "Failed to install Gurobi" in capsys.readouterr().out
        # grbgetkey is never reached.
        assert not router.matching("grbgetkey")

    def test_activation_failure_returns_false(
        self, monkeypatch, conda, capsys
    ):
        monkeypatch.setattr(uem, "is_package_installed", lambda pkg, env: True)
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(
                rules=[("grbgetkey", FakeCompleted(1, "", "bad key"))],
                default=FakeCompleted(0),
            ),
        )
        assert uem.setup_gurobi_license("BAD") is False
        assert "Failed to activate license" in capsys.readouterr().out

    def test_exception_returns_false(self, monkeypatch, capsys):
        def boom():
            raise RuntimeError("no conda")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        assert uem.setup_gurobi_license("KEY") is False
        assert "Error setting up Gurobi: no conda" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_ultrack_in_env
# ---------------------------------------------------------------------------


class FakePopen:
    """Minimal ``subprocess.Popen`` stand-in with line-iterable stdout."""

    instances = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = list(self.lines)
        self.returncode = self.rc
        self.killed = False
        self.wait_timeout = None
        type(self).instances.append(self)

    lines = []
    rc = 0
    raise_timeout = False

    def wait(self, timeout=None):
        self.wait_timeout = timeout
        if self.raise_timeout:
            raise subprocess.TimeoutExpired("conda", timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def _popen_factory(monkeypatch, lines, rc=0, raise_timeout=False):
    """Install a FakePopen subclass and return the instance registry."""
    instances = []

    class _P(FakePopen):
        pass

    _P.instances = instances
    _P.lines = lines
    _P.rc = rc
    _P.raise_timeout = raise_timeout
    monkeypatch.setattr(uem.subprocess, "Popen", _P)
    return instances


class TestRunUltrackInEnv:
    """Streaming subprocess runner: success, failure, timeout, cleanup."""

    @pytest.fixture(autouse=True)
    def _temp_dir(self, monkeypatch, tmp_path):
        # Keep the module's NamedTemporaryFile inside tmp_path.
        monkeypatch.setattr(uem.tempfile, "tempdir", str(tmp_path))

    def test_success_streams_output_to_callback(
        self, monkeypatch, tmp_path, conda
    ):
        instances = _popen_factory(monkeypatch, ["hello\n", "world\n"], rc=0)
        log = Logger()

        result = uem.run_ultrack_in_env(
            "print('hi')",
            env_name="ultrack",
            progress_callback=log,
            input_file="/data/in.tif",
            output_file="/data/out.tif",
        )

        assert result["success"] is True
        assert result["output"] == "hello\nworld\n"
        assert result["error"] == ""
        assert "hello" in log.messages
        assert "world" in log.messages
        assert "  Input: /data/in.tif" in log.messages
        assert "  Output: /data/out.tif" in log.messages
        assert "✓ Tracking completed successfully" in log.messages

        proc = instances[0]
        assert proc.cmd[:4] == ["conda", "run", "-n", "ultrack"]
        assert proc.cmd[4] == "python"
        assert proc.cmd[5].endswith(".py")
        assert proc.kwargs["env"] is None
        assert proc.wait_timeout == 7200
        # stderr is folded into the single streamed pipe -- with a
        # separate PIPE the child's tracebacks would never be read.
        assert proc.kwargs["stdout"] is subprocess.PIPE
        assert proc.kwargs["stderr"] is subprocess.STDOUT
        assert proc.kwargs["text"] is True
        assert proc.kwargs["bufsize"] == 1

    def test_script_is_written_then_deleted(
        self, monkeypatch, tmp_path, conda
    ):
        seen = {}
        instances = _popen_factory(monkeypatch, [], rc=0)
        orig = uem.subprocess.Popen

        def spy(cmd, **kwargs):
            seen["path"] = cmd[-1]
            seen["content"] = uem.Path(cmd[-1]).read_text()
            return orig(cmd, **kwargs)

        monkeypatch.setattr(uem.subprocess, "Popen", spy)

        result = uem.run_ultrack_in_env(
            "print('body')", progress_callback=Logger()
        )

        assert result["success"] is True
        assert seen["content"] == "print('body')"
        assert seen["path"].startswith(str(tmp_path))
        assert not uem.Path(seen["path"]).exists()
        assert instances  # the underlying FakePopen really ran

    def test_failure_returns_last_lines_as_error(
        self, monkeypatch, tmp_path, conda
    ):
        lines = [f"line{i}\n" for i in range(60)]
        _popen_factory(monkeypatch, lines, rc=3)
        log = Logger()

        result = uem.run_ultrack_in_env("x", progress_callback=log)

        assert result["success"] is False
        assert result["output"] == "".join(lines)
        assert result["error"] == "".join(lines[-50:])
        assert result["error"].startswith("line10\n")
        assert "✗ Tracking failed with return code 3" in log.messages

    def test_timeout_kills_process(self, monkeypatch, tmp_path, conda):
        instances = _popen_factory(
            monkeypatch, ["partial\n"], rc=0, raise_timeout=True
        )
        log = Logger()

        result = uem.run_ultrack_in_env("x", progress_callback=log)

        assert result["success"] is False
        assert result["error"] == "Process timed out after 2 hours"
        assert result["output"] == "partial\n"
        assert instances[0].killed is True
        assert "✗ Tracking timed out after 2 hours" in log.messages

    def test_extra_env_is_layered_on_os_environ(
        self, monkeypatch, tmp_path, conda
    ):
        monkeypatch.setenv("TMIDAS_SENTINEL", "keep-me")
        instances = _popen_factory(monkeypatch, [], rc=0)

        uem.run_ultrack_in_env(
            "x",
            progress_callback=Logger(),
            extra_env={"GRB_LICENSE_FILE": "/lic/gurobi.lic"},
        )

        env = instances[0].kwargs["env"]
        assert env["GRB_LICENSE_FILE"] == "/lic/gurobi.lic"
        assert env["TMIDAS_SENTINEL"] == "keep-me"

    def test_without_callback_output_is_printed(
        self, monkeypatch, tmp_path, conda, capsys
    ):
        _popen_factory(monkeypatch, ["streamed\n"], rc=0)

        result = uem.run_ultrack_in_env("x")

        captured = capsys.readouterr().out
        assert "streamed" in captured
        assert "Running ultrack in environment 'ultrack'" in captured
        assert result["success"] is True

    def test_exception_before_popen_is_reported(self, monkeypatch, capsys):
        def boom():
            raise RuntimeError("no conda")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)

        result = uem.run_ultrack_in_env("x")
        assert result == {
            "success": False,
            "output": "",
            "error": "no conda",
        }
        assert "✗ Error running ultrack: no conda" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# create_ultrack_env
# ---------------------------------------------------------------------------


def _happy_env(monkeypatch):
    """Neutralise everything create_ultrack_env delegates to."""
    monkeypatch.setattr(uem, "get_conda_cmd", lambda: "conda")
    monkeypatch.setattr(uem, "is_package_installed", lambda pkg, env: True)
    monkeypatch.setattr(uem, "_patch_ultrack_xp", lambda env, log: True)
    monkeypatch.setattr(
        uem, "_patch_ultrack_readonly_arrays", lambda env, log: True
    )
    monkeypatch.setattr(uem, "_ensure_scikit_image_fix", lambda env, log: True)
    monkeypatch.setattr(
        uem,
        "check_gpu_available",
        lambda env: {
            "available": True,
            "device_name": "RTX",
            "compute_capability": "8.9",
            "memory_gb": 24.0,
        },
    )


class TestCreateUltrackEnvFailures:
    """Early exits: no conda, env creation failure, pip failure."""

    def test_conda_missing_returns_false(self, monkeypatch):
        def boom():
            raise RuntimeError("Neither conda nor mamba found in PATH")

        monkeypatch.setattr(uem, "get_conda_cmd", boom)
        log = Logger()
        assert uem.create_ultrack_env("ultrack", log) is False
        assert "Error creating environment" in log.text()

    def test_env_creation_failure_returns_false(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        router = RunRouter(
            rules=[("create -n", FakeCompleted(1, "", "disk full"))],
            default=FakeCompleted(0),
        )
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is False
        assert "Failed to create environment: disk full" in log.text()
        # Nothing after step 1 runs.
        assert not router.matching("scipy=1.14")

    def test_pip_install_failure_returns_false(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(
                rules=[
                    (
                        "ultrack zarr tifffile",
                        FakeCompleted(1, "", "no matching dist"),
                    )
                ],
                default=FakeCompleted(0),
            ),
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is False
        assert "Failed to install pip packages" in log.text()

    def test_missing_critical_packages_returns_false(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        monkeypatch.setattr(
            uem, "is_package_installed", lambda pkg, env: pkg != "ultrack"
        )
        monkeypatch.setattr(
            uem.subprocess, "run", RunRouter(default=FakeCompleted(0))
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is False
        assert "Critical packages missing: ultrack" in log.text()

    def test_conda_package_failure_only_warns(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        router = RunRouter(
            rules=[("scipy=1.14", FakeCompleted(1, "", "conflict"))],
            default=FakeCompleted(0),
        )
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "Some conda packages failed to install" in log.text()
        assert "Will try to continue with pip" in log.text()
        # Execution continues into the pip step.
        assert router.matching("ultrack zarr tifffile")


class TestCreateUltrackEnvCpuPath:
    """No CUDA detected: CPU-only torch, no CuPy, no cucim."""

    def test_cpu_only_success(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        router = RunRouter(default=FakeCompleted(0))
        monkeypatch.setattr(uem.subprocess, "run", router)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True

        assert router.matching("pip install torch torchvision")
        assert not router.matching("download.pytorch.org")
        assert not router.matching("cupy-cuda")
        assert not router.matching("cucim")
        assert "No NVIDIA GPU detected, installing CPU-only PyTorch" in (
            log.text()
        )
        assert "✓ Installed PyTorch (CPU-only)" in log.text()
        assert "created successfully" in log.text()

    def test_cpu_torch_failure_is_only_a_warning(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        monkeypatch.setattr(
            uem.subprocess,
            "run",
            RunRouter(
                rules=[
                    (
                        "pip install torch torchvision",
                        FakeCompleted(1, "", "wheel missing"),
                    )
                ],
                default=FakeCompleted(0),
            ),
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "⚠ Failed to install PyTorch: wheel missing" in log.text()

    def test_progress_callback_is_optional(self, monkeypatch, capsys):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        monkeypatch.setattr(
            uem.subprocess, "run", RunRouter(default=FakeCompleted(0))
        )

        assert uem.create_ultrack_env("ultrack") is True
        assert "Using conda to create environment" in capsys.readouterr().out

    def test_patch_failures_are_warnings_not_errors(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: None)
        monkeypatch.setattr(uem, "_patch_ultrack_xp", lambda env, log: False)
        monkeypatch.setattr(
            uem, "_patch_ultrack_readonly_arrays", lambda env, log: False
        )
        monkeypatch.setattr(
            uem, "_ensure_scikit_image_fix", lambda env, log: False
        )
        monkeypatch.setattr(
            uem.subprocess, "run", RunRouter(default=FakeCompleted(0))
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "Failed to apply ultrack patch" in log.text()
        assert "Failed to patch read-only array handling" in log.text()
        assert "Could not verify scikit-image fix" in log.text()


class TestCreateUltrackEnvGpuPath:
    """CUDA detected: index-url selection, GPU smoke test, CuPy variant."""

    def _router(self, monkeypatch, rules=()):
        router = RunRouter(rules=list(rules), default=FakeCompleted(0))
        monkeypatch.setattr(uem.subprocess, "run", router)
        return router

    def test_cuda12_uses_cu121_and_cupy_cuda12x(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.4")
        router = self._router(
            monkeypatch,
            [
                (
                    "warnings.filterwarnings",
                    FakeCompleted(0, "SUCCESS:RTX 4090:89\n"),
                )
            ],
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert router.matching("whl/cu121")
        assert not router.matching("whl/cu118")
        assert router.matching("pip install --upgrade cupy-cuda12x")
        assert router.matching("cucim -y")
        assert "NVIDIA GPU detected (CUDA 12.4)" in log.text()
        assert "✓ GPU works: RTX 4090" in log.text()
        assert "Compute capability: sm_89" in log.text()
        assert "✓ Installed cucim" in log.text()
        assert "GPU ready for future use: RTX" in log.text()

    def test_cuda11_uses_cu118_and_cupy_cuda11x(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "11.8")
        router = self._router(monkeypatch)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert router.matching("whl/cu118")
        assert router.matching("pip install --upgrade cupy-cuda11x")

    def test_cuda13_uses_cupy_cuda13x(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "13.0")
        router = self._router(monkeypatch)

        assert uem.create_ultrack_env("ultrack", Logger()) is True
        assert router.matching("pip install --upgrade cupy-cuda13x")

    def test_cuda_torch_failure_falls_back_to_cpu(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.4")
        router = self._router(
            monkeypatch,
            [("download.pytorch.org", FakeCompleted(1, "", "404 wheel"))],
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "⚠ Failed to install PyTorch with CUDA: 404 wheel" in log.text()
        assert "Falling back to CPU-only PyTorch" in log.text()
        assert "✓ Installed PyTorch (CPU-only)" in log.text()
        # The GPU smoke test is skipped on the fallback path.
        assert not router.matching("warnings.filterwarnings")

    def test_blackwell_stderr_triggers_nightly_advice(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.8")
        self._router(
            monkeypatch,
            [
                (
                    "warnings.filterwarnings",
                    FakeCompleted(1, "", "no kernel image sm_120"),
                )
            ],
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "Blackwell GPU detected" in log.text()
        assert "nightly/cu130" in log.text()

    def test_inconclusive_gpu_test_is_tolerated(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.1")
        self._router(
            monkeypatch,
            [("warnings.filterwarnings", FakeCompleted(1, "", "weird"))],
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "GPU test inconclusive" in log.text()

    def test_cupy_and_cucim_failures_are_warnings(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.4")
        self._router(
            monkeypatch,
            [
                ("cupy-cuda", FakeCompleted(1, "", "no wheel")),
                ("cucim", FakeCompleted(1, "", "no package")),
            ],
        )
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "⚠ Failed to install cupy-cuda12x: no wheel" in log.text()
        assert "GPU acceleration will not be available" in log.text()
        assert "⚠ Failed to install cucim: no package" in log.text()

    def test_blackwell_gpu_check_reports_cpu_mode(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.8")
        monkeypatch.setattr(
            uem,
            "check_gpu_available",
            lambda env: {
                "available": False,
                "error": "CUDA_ERROR_NO_BINARY_FOR_GPU",
            },
        )
        self._router(monkeypatch)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "Newer GPU architecture detected (Blackwell)" in log.text()
        assert "CPU processing active" in log.text()

    def test_other_gpu_failure_is_logged_verbatim(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.8")
        monkeypatch.setattr(
            uem,
            "check_gpu_available",
            lambda env: {"available": False, "error": "driver too old"},
        )
        self._router(monkeypatch)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "GPU test: driver too old" in log.text()

    def test_gpu_check_without_error_key_uses_default(self, monkeypatch):
        _happy_env(monkeypatch)
        monkeypatch.setattr(uem, "_get_cuda_version", lambda: "12.8")
        monkeypatch.setattr(
            uem, "check_gpu_available", lambda env: {"available": False}
        )
        self._router(monkeypatch)
        log = Logger()

        assert uem.create_ultrack_env("ultrack", log) is True
        assert "GPU test: Unknown reason" in log.text()


# ---------------------------------------------------------------------------
# UltrackEnvironmentManager
# ---------------------------------------------------------------------------


class TestUltrackEnvironmentManagerWrapper:
    """The class is a thin delegator; pin that it forwards env_name."""

    def test_custom_env_name(self):
        assert uem.UltrackEnvironmentManager("other").env_name == "other"

    def test_get_conda_cmd_delegates(self, monkeypatch):
        monkeypatch.setattr(uem, "get_conda_cmd", lambda: "mamba")
        assert uem.UltrackEnvironmentManager()._get_conda_cmd() == "mamba"

    def test_is_env_created_forwards_env_name(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            uem, "is_env_created", lambda name: seen.append(name) or True
        )
        assert uem.UltrackEnvironmentManager("myenv").is_env_created() is True
        assert seen == ["myenv"]

    def test_create_env_forwards_callback(self, monkeypatch):
        seen = {}

        def fake(name, cb):
            seen["name"] = name
            seen["cb"] = cb
            return True

        monkeypatch.setattr(uem, "create_ultrack_env", fake)
        log = Logger()
        assert uem.UltrackEnvironmentManager("e").create_env(log) is True
        assert seen == {"name": "e", "cb": log}

    def test_is_package_installed_defaults_to_ultrack(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            uem,
            "is_package_installed",
            lambda pkg, env: seen.append((pkg, env)) or False,
        )
        mgr = uem.UltrackEnvironmentManager("e")
        assert mgr.is_package_installed() is False
        assert mgr.is_package_installed("zarr") is False
        assert seen == [("ultrack", "e"), ("zarr", "e")]

    def test_check_gpu_available_returns_full_dict(self, monkeypatch):
        info = {"available": True, "device_name": "A100"}
        monkeypatch.setattr(uem, "check_gpu_available", lambda env: info)
        assert uem.UltrackEnvironmentManager().check_gpu_available() is info

    def test_check_gpu_available_boolean_variant(self, monkeypatch):
        monkeypatch.setattr(
            uem, "check_gpu_available", lambda env: {"available": True}
        )
        assert uem.UltrackEnvironmentManager()._check_gpu_available() is True

        monkeypatch.setattr(uem, "check_gpu_available", lambda env: {})
        assert uem.UltrackEnvironmentManager()._check_gpu_available() is False

    def test_setup_gurobi_license_forwards(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            uem,
            "setup_gurobi_license",
            lambda key, env: seen.append((key, env)) or True,
        )
        mgr = uem.UltrackEnvironmentManager("e")
        assert mgr.setup_gurobi_license("KEY") is True
        assert seen == [("KEY", "e")]

    def test_run_in_env_forwards_all_kwargs(self, monkeypatch):
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return {"success": True, "output": "", "error": ""}

        monkeypatch.setattr(uem, "run_ultrack_in_env", fake)
        log = Logger()
        result = uem.UltrackEnvironmentManager("e").run_in_env(
            "print(1)",
            progress_callback=log,
            input_file="in.tif",
            output_file="out.tif",
        )

        assert result["success"] is True
        assert seen == {
            "script_content": "print(1)",
            "env_name": "e",
            "progress_callback": log,
            "input_file": "in.tif",
            "output_file": "out.tif",
        }

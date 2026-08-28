"""Coverage-focused tests for the Cellpose environment manager.

Everything that would shell out, build a venv or touch a GPU is replaced
by an in-process double, so only real module code executes. The doubles
stand in for ``subprocess`` and for the generated worker script; the
module's own parsing, branching, script generation and cleanup logic is
exercised for real.
"""

import os
import re
import subprocess
import types

import numpy as np
import pytest
import tifffile
import zarr

import napari_tmidas.processing_functions.cellpose_env_manager as cem

# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


_UNSET = object()


class _Completed:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _FakeProcess:
    """Minimal ``Popen`` double: iterable stdout plus wait/poll/kill."""

    def __init__(self, lines=(), returncode=0, poll_value=None):
        self.stdout = None if lines is None else iter(list(lines))
        self._returncode = returncode
        self._poll_value = poll_value
        self.terminated = False
        self.killed = False
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self._returncode

    def poll(self):
        return self._poll_value

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _StubManager:
    """Records the manager calls made by ``run_cellpose_in_env``."""

    def __init__(self, packages_ok=True):
        self.packages_ok = packages_ok
        self.calls = []

    def are_all_packages_installed(self):
        self.calls.append("check")
        return self.packages_ok

    def reinstall_packages(self):
        self.calls.append("reinstall")

    def ensure_minimum_cellpose_version(self, minimum):
        self.calls.append(("ensure", minimum))


def _install_fake_subprocess(monkeypatch, **overrides):
    """Swap ``cem.subprocess`` for a namespace of harmless doubles."""
    namespace = types.SimpleNamespace(
        run=lambda *a, **k: _Completed(),
        check_call=lambda *a, **k: 0,
        Popen=lambda *a, **k: _FakeProcess(),
        CalledProcessError=subprocess.CalledProcessError,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError,
        PIPE=subprocess.PIPE,
        STDOUT=subprocess.STDOUT,
    )
    for key, value in overrides.items():
        setattr(namespace, key, value)
    monkeypatch.setattr(cem, "subprocess", namespace)
    return namespace


def _recording_check_call(calls, fail_on=None):
    def _check_call(cmd, **kwargs):
        calls.append(list(cmd))
        if fail_on is not None and fail_on in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return 0

    return _check_call


def _probe_run(sm120_results, fail_marker=None):
    """Fake ``subprocess.run`` for the sm_120 probe and verifications."""
    pending = list(sm120_results)

    def _run(cmd, **kwargs):
        code = cmd[2] if len(cmd) > 2 else ""
        if "get_arch_list" in code:
            value = pending.pop(0) if pending else False
            return _Completed(stdout=("True" if value else "False") + "\n")
        if fail_marker is not None and fail_marker in code:
            raise subprocess.CalledProcessError(1, cmd)
        return _Completed(stdout="verified\n")

    return _run


@pytest.fixture(autouse=True)
def _clean_process_registry():
    """Keep the module-global process registry isolated per test."""
    saved = list(cem._running_processes)
    cem._running_processes.clear()
    yield
    cem._running_processes.clear()
    cem._running_processes.extend(saved)


# ---------------------------------------------------------------------
# Log line helpers
# ---------------------------------------------------------------------


class TestShouldEmitCellposeLogLine:
    """Pins which worker log lines reach the terminal."""

    # Pinned literally rather than read from the module, so shrinking the
    # module's prefix tuple fails here instead of silently collecting fewer
    # (or zero) parametrised cases.
    EXPECTED_NOISY_PREFIXES = (
        "[GUI INFO] : WRITING LOG OUTPUT TO",
        "RUNNING BLOCK:",
        "cellpose version:",
        "platform:",
        "python version:",
        "torch version:",
        "=== Cellpose Environment Info ===",
    )

    def test_noisy_prefix_list_is_exactly_the_pinned_set(self):
        assert cem._NOISY_CELLPOSE_PREFIXES == self.EXPECTED_NOISY_PREFIXES

    @pytest.mark.parametrize("line", ["", "   ", "\n", None])
    def test_blank_lines_are_dropped(self, line):
        assert cem._should_emit_cellpose_log_line(line) is False

    @pytest.mark.parametrize("prefix", EXPECTED_NOISY_PREFIXES)
    def test_every_noisy_prefix_is_dropped(self, prefix):
        assert cem._should_emit_cellpose_log_line(prefix + " tail") is False
        # Leading whitespace is stripped before the prefix match.
        assert (
            cem._should_emit_cellpose_log_line("   " + prefix + " tail")
            is False
        )

    @pytest.mark.parametrize("prefix", EXPECTED_NOISY_PREFIXES)
    def test_prefix_only_matches_at_the_start(self, prefix):
        # A line that merely *contains* a noisy marker must still be shown,
        # otherwise real errors quoting the marker would be swallowed.
        assert (
            cem._should_emit_cellpose_log_line(f"ERROR while {prefix} x")
            is True
        )

    def test_ordinary_line_is_kept(self):
        assert cem._should_emit_cellpose_log_line("  segmenting  ") is True


class TestTransformCellposeLogLine:
    """Pins the raw-log -> progress-message translation."""

    def test_blank_line_is_swallowed(self):
        assert cem._transform_cellpose_log_line("  ", {}) == (True, None)

    def test_total_line_primes_state_and_renders_zero_percent(self):
        state = {}
        handled, rendered = cem._transform_cellpose_log_line(
            "DISTRIBUTED_PROGRESS_TOTAL=8", state
        )
        assert handled is True
        assert rendered == "Distributed Cellpose progress: 0/8 blocks (0.0%)"
        assert state == {"total": 8, "done": 0}

    def test_zero_total_renders_nothing(self):
        state = {}
        assert cem._transform_cellpose_log_line(
            "DISTRIBUTED_PROGRESS_TOTAL=0", state
        ) == (True, None)
        assert state["total"] == 0

    def test_negative_total_is_clamped_to_zero(self):
        state = {}
        assert cem._transform_cellpose_log_line(
            "DISTRIBUTED_PROGRESS_TOTAL=-4", state
        ) == (True, None)
        assert state == {"total": 0, "done": 0}

    def test_unparsable_total_falls_back_to_zero(self):
        state = {"total": 5}
        assert cem._transform_cellpose_log_line(
            "DISTRIBUTED_PROGRESS_TOTAL=oops", state
        ) == (True, None)
        assert state["total"] == 0
        assert state["done"] == 0

    def test_running_block_reports_percentage(self):
        state = {"total": 4, "done": 1}
        handled, rendered = cem._transform_cellpose_log_line(
            "RUNNING BLOCK: 2", state
        )
        assert handled is True
        assert rendered == (
            "Distributed Cellpose progress: 2/4 blocks (50.0%)"
        )
        assert state["done"] == 2

    def test_running_block_percentage_is_capped_at_100(self):
        state = {"total": 2, "done": 5}
        handled, rendered = cem._transform_cellpose_log_line(
            "RUNNING BLOCK: x", state
        )
        assert handled is True
        assert rendered == (
            "Distributed Cellpose progress: 6/2 blocks (100.0%)"
        )
        assert state["done"] == 6

    def test_running_block_without_total_counts_blocks(self):
        state = {}
        _, rendered = cem._transform_cellpose_log_line(
            "RUNNING BLOCK: x", state
        )
        assert rendered == (
            "Distributed Cellpose progress: 1 blocks processed"
        )

    def test_cellpose_progress_renders_percentage(self):
        state = {}
        handled, rendered = cem._transform_cellpose_log_line(
            "CELLPOSE_PROGRESS=42.9", state
        )
        assert (handled, rendered) == (True, "Cellpose progress: 42%")
        assert state["cellpose_last_pct"] == 42

    def test_repeated_cellpose_progress_is_deduplicated(self):
        state = {}
        cem._transform_cellpose_log_line("CELLPOSE_PROGRESS=42", state)
        assert cem._transform_cellpose_log_line(
            "CELLPOSE_PROGRESS=42", state
        ) == (True, None)

    @pytest.mark.parametrize(("raw", "expected"), [("150", 100), ("-5", 0)])
    def test_cellpose_progress_is_clamped(self, raw, expected):
        state = {}
        _, rendered = cem._transform_cellpose_log_line(
            f"CELLPOSE_PROGRESS={raw}", state
        )
        assert rendered == f"Cellpose progress: {expected}%"

    def test_unparsable_cellpose_progress_is_swallowed(self):
        state = {}
        assert cem._transform_cellpose_log_line(
            "CELLPOSE_PROGRESS=nan%", state
        ) == (True, None)
        assert "cellpose_last_pct" not in state

    def test_unrelated_line_is_left_to_the_caller(self):
        assert cem._transform_cellpose_log_line("plain text", {}) == (
            False,
            None,
        )


class TestVersionHelpers:
    """Pins the numeric version parsing/comparison edge cases."""

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_version_yields_empty_tuple(self, value):
        assert cem._version_tuple(value) == ()

    def test_non_numeric_version_yields_empty_tuple(self):
        assert cem._version_tuple("unknown") == ()

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("4.2.1.1", (4, 2, 1, 1)),
            ("v4.2.1.1", (4, 2, 1, 1)),
            ("V4.3", (4, 3)),
            ("  4.0.1.dev3  ", (4, 0, 1, 3)),
            ("4.10", (4, 10)),
        ],
    )
    def test_numeric_components_are_extracted(self, raw, expected):
        assert cem._version_tuple(raw) == expected

    def test_missing_component_makes_comparison_false(self):
        assert cem._is_version_at_least("", "4.2") is False
        assert cem._is_version_at_least("4.2", "") is False

    @pytest.mark.parametrize(
        ("current", "minimum", "expected"),
        [
            ("4.2.1.1", "4.2.1.1", True),
            ("4.2.1.2", "4.2.1.1", True),
            ("4.2.1.0", "4.2.1.1", False),
            ("4.10", "4.9", True),  # numeric, not lexicographic
            ("4.9", "4.10", False),
            ("v4.3.0", "4.2.1.1", True),
            ("3.9.9", "4.2.1.1", False),
        ],
    )
    def test_comparison_is_numeric_per_component(
        self, current, minimum, expected
    ):
        assert cem._is_version_at_least(current, minimum) is expected

    def test_shorter_version_is_zero_padded(self):
        assert cem._is_version_at_least("4.3", "4.3.0.0") is True
        assert cem._is_version_at_least("4.3", "4.3.0.1") is False


# ---------------------------------------------------------------------
# Process registry / cancellation
# ---------------------------------------------------------------------


class TestProcessRegistry:
    """Pins process bookkeeping and the cancellation fallbacks."""

    def test_add_and_remove_round_trip(self):
        proc = _FakeProcess()
        cem._add_process(proc)
        assert cem._running_processes == [proc]
        cem._remove_process(proc)
        assert cem._running_processes == []

    def test_remove_unknown_process_leaves_the_registry_intact(self):
        # A registered sentinel makes this fail if _remove_process were to
        # clear the list (or be emptied out); an empty registry would pass
        # even with no implementation at all.
        registered = _FakeProcess()
        cem._add_process(registered)
        cem._remove_process(_FakeProcess())
        assert cem._running_processes == [registered]

    def test_add_keeps_insertion_order_and_allows_duplicates(self):
        first, second = _FakeProcess(), _FakeProcess()
        cem._add_process(first)
        cem._add_process(second)
        cem._add_process(first)
        assert cem._running_processes == [first, second, first]
        cem._remove_process(first)
        # Only the first occurrence is dropped.
        assert cem._running_processes == [second, first]

    def test_cancel_terminates_a_running_process(self):
        proc = _FakeProcess(poll_value=None)
        cem._add_process(proc)
        cem.cancel_all_processes()
        assert proc.terminated is True
        assert proc.killed is False
        assert proc.waits == [5]
        assert cem._running_processes == []

    def test_cancel_kills_a_process_that_ignores_terminate(self):
        proc = _FakeProcess(poll_value=None)

        def _wait(timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("cellpose", timeout)
            return 0

        proc.wait = _wait
        cem._add_process(proc)
        cem.cancel_all_processes()
        assert proc.killed is True
        assert cem._running_processes == []

    def test_cancel_only_deregisters_a_finished_process(self):
        proc = _FakeProcess(poll_value=0)
        cem._add_process(proc)
        cem.cancel_all_processes()
        assert proc.terminated is False
        assert cem._running_processes == []

    def test_cancel_reports_but_survives_an_os_error(self, capsys):
        proc = _FakeProcess(poll_value=None)

        def _boom():
            raise OSError("gone")

        proc.terminate = _boom
        cem._add_process(proc)
        cem.cancel_all_processes()  # must not propagate
        assert "Error terminating process: gone" in capsys.readouterr().out
        assert proc.killed is False

    def test_one_failing_process_does_not_stop_the_others(self, capsys):
        failing = _FakeProcess(poll_value=None)

        def _boom():
            raise OSError("gone")

        failing.terminate = _boom
        healthy = _FakeProcess(poll_value=None)
        cem._add_process(failing)
        cem._add_process(healthy)

        cem.cancel_all_processes()

        # The loop continues past the failure...
        assert healthy.terminated is True
        # ...and every process is deregistered even when terminating it
        # raised, so a process that is already gone does not stay tracked
        # forever and get retried (and reprinted) on every later cancel.
        assert cem._running_processes == []
        assert "Error terminating process: gone" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------


class TestInstallDependencies:
    """Pins the PyTorch/cellpose install sequence and its failure modes."""

    def _manager(self):
        return cem.CellposeEnvironmentManager()

    def test_stable_torch_path_installs_every_package_once(
        self, monkeypatch, capsys
    ):
        calls = []
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call(calls),
            run=_probe_run([True]),
        )
        self._manager()._install_dependencies("/env/python")

        joined = [" ".join(c) for c in calls]
        assert len(calls) == 8
        assert "whl/cu128" in joined[0]
        assert not any("--pre" in c for c in joined)
        for package in [
            "cellpose",
            "zarr>=3",
            "tifffile",
            "dask[distributed]",
            "dask-jobqueue",
            "dask-image",
        ]:
            assert any(c.endswith(package) for c in joined)
        assert "dinov3" in joined[-1]
        out = capsys.readouterr().out
        assert "Installed PyTorch build supports sm_120" in out

    def test_stable_torch_failure_propagates(self, monkeypatch, capsys):
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call([], fail_on="torch"),
            run=_probe_run([True]),
        )
        with pytest.raises(subprocess.CalledProcessError):
            self._manager()._install_dependencies("/env/python")
        assert "Failed to install stable PyTorch" in capsys.readouterr().out

    def test_missing_sm120_triggers_nightly_install(self, monkeypatch, capsys):
        calls = []
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call(calls),
            run=_probe_run([False, True]),
        )
        self._manager()._install_dependencies("/env/python")

        joined = [" ".join(c) for c in calls]
        assert len(calls) == 9
        assert "--pre" in joined[1]
        assert "nightly/cu128" in joined[1]
        assert "Nightly PyTorch CUDA 12.8 installed" in capsys.readouterr().out

    def test_nightly_install_failure_propagates(self, monkeypatch, capsys):
        def _check_call(cmd, **kwargs):
            if "--pre" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return 0

        _install_fake_subprocess(
            monkeypatch,
            check_call=_check_call,
            run=_probe_run([False]),
        )
        with pytest.raises(subprocess.CalledProcessError):
            self._manager()._install_dependencies("/env/python")
        assert "Failed to install nightly PyTorch" in capsys.readouterr().out

    def test_nightly_without_sm120_raises_runtime_error(self, monkeypatch):
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call([]),
            run=_probe_run([False, False]),
        )
        with pytest.raises(RuntimeError, match="sm_120 support"):
            self._manager()._install_dependencies("/env/python")

    def test_package_install_failure_stops_before_dinov3(
        self, monkeypatch, capsys
    ):
        calls = []
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call(calls, fail_on="zarr>=3"),
            run=_probe_run([True]),
        )
        with pytest.raises(subprocess.CalledProcessError):
            self._manager()._install_dependencies("/env/python")
        assert not any("dinov3" in " ".join(c) for c in calls)
        assert "Failed to install zarr>=3" in capsys.readouterr().out

    def test_dinov3_failure_is_only_a_warning(self, monkeypatch, capsys):
        calls = []
        dep = "git+https://github.com/facebookresearch/dinov3"
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call(calls, fail_on=dep),
            run=_probe_run([True]),
        )
        self._manager()._install_dependencies("/env/python")
        out = capsys.readouterr().out
        assert "Failed to install optional dinov3 dependency" in out
        assert "Tifffile installation verified" in out

    @pytest.mark.parametrize(
        ("marker", "message"),
        [
            ("from cellpose import core", "Cellpose verification failed"),
            ("import zarr;", "Zarr verification failed"),
            ("import tifffile;", "Tifffile verification failed"),
        ],
    )
    def test_verification_failures_propagate(
        self, monkeypatch, capsys, marker, message
    ):
        _install_fake_subprocess(
            monkeypatch,
            check_call=_recording_check_call([]),
            run=_probe_run([True], fail_marker=marker),
        )
        with pytest.raises(subprocess.CalledProcessError):
            self._manager()._install_dependencies("/env/python")
        assert message in capsys.readouterr().out


class TestTorchSupportsSm120:
    """Pins the sm_120 probe's parsing of the child process output."""

    def test_true_output_is_recognised(self, monkeypatch):
        _install_fake_subprocess(
            monkeypatch, run=lambda *a, **k: _Completed(stdout="True\n")
        )
        mgr = cem.CellposeEnvironmentManager()
        assert mgr._torch_supports_sm120("/env/python") is True

    def test_false_output_is_recognised(self, monkeypatch):
        _install_fake_subprocess(
            monkeypatch, run=lambda *a, **k: _Completed(stdout="False\n")
        )
        mgr = cem.CellposeEnvironmentManager()
        assert mgr._torch_supports_sm120("/env/python") is False

    def test_probe_exception_is_swallowed(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("no interpreter")

        _install_fake_subprocess(monkeypatch, run=_boom)
        mgr = cem.CellposeEnvironmentManager()
        assert mgr._torch_supports_sm120("/env/python") is False


# ---------------------------------------------------------------------
# Environment inspection
# ---------------------------------------------------------------------


class TestIsPackageInstalled:
    """Pins the in-process cellpose availability check."""

    def test_returns_true_when_spec_found(self, monkeypatch):
        import importlib.util

        asked = []

        def _find_spec(name):
            asked.append(name)
            return object()

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec)
        assert cem.CellposeEnvironmentManager().is_package_installed() is True
        assert asked == ["cellpose"]

    def test_returns_false_when_spec_missing(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert cem.CellposeEnvironmentManager().is_package_installed() is False

    def test_import_error_is_swallowed(self, monkeypatch):
        import importlib.util

        def _boom(name):
            raise ImportError(name)

        monkeypatch.setattr(importlib.util, "find_spec", _boom)
        assert not cem.CellposeEnvironmentManager().is_package_installed()


class TestAreAllPackagesInstalled:
    """Pins the per-package probe of the dedicated environment."""

    def _manager(self, env_created=True):
        mgr = cem.CellposeEnvironmentManager()
        mgr.is_env_created = lambda: env_created
        mgr.get_env_python_path = lambda: "/env/python"
        return mgr

    def test_missing_env_short_circuits(self, monkeypatch):
        def _unexpected(*a, **k):
            raise AssertionError("should not probe a missing env")

        _install_fake_subprocess(monkeypatch, run=_unexpected)
        assert (
            self._manager(env_created=False).are_all_packages_installed()
            is False
        )

    def test_all_probes_succeeding_returns_true(self, monkeypatch):
        probed = []

        def _run(cmd, **kwargs):
            probed.append(cmd)
            return _Completed()

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().are_all_packages_installed() is True

        # The full probe set is the contract of "all packages installed";
        # dropping or reordering one must fail here.
        assert [cmd[2] for cmd in probed] == [
            "import cellpose",
            "import zarr; assert int(zarr.__version__.split('.')[0]) >= 3",
            "import tifffile",
            "import dask",
            "import distributed",
            "import dask_image",
        ]
        # Every probe runs inside the dedicated env, not the host interpreter.
        assert {tuple(cmd[:2]) for cmd in probed} == {("/env/python", "-c")}

    @pytest.mark.parametrize(
        ("failing_import", "reported_name", "expected_probes"),
        [
            ("import cellpose", "cellpose", 1),
            ("import zarr;", "zarr>=3", 2),
            ("import tifffile", "tifffile", 3),
            ("import dask_image", "dask-image", 6),
        ],
    )
    def test_first_failing_probe_returns_false(
        self,
        monkeypatch,
        capsys,
        failing_import,
        reported_name,
        expected_probes,
    ):
        probed = []

        def _run(cmd, **kwargs):
            probed.append(cmd[2])
            if failing_import in cmd[2]:
                raise subprocess.CalledProcessError(1, cmd)
            return _Completed()

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().are_all_packages_installed() is False
        assert (
            f"Missing package in cellpose environment: {reported_name}"
            in capsys.readouterr().out
        )
        # The check short-circuits on the first missing package.
        assert len(probed) == expected_probes


class TestReinstallPackages:
    """Pins the create-vs-reinstall decision."""

    def test_missing_env_is_created_instead(self, capsys):
        mgr = cem.CellposeEnvironmentManager()
        mgr.is_env_created = lambda: False
        created = []
        mgr.create_env = lambda: created.append(True)
        installed = []
        mgr._install_dependencies = lambda p: installed.append(p)

        mgr.reinstall_packages()
        assert created == [True]
        assert installed == []
        assert "Environment not created" in capsys.readouterr().out

    def test_existing_env_reinstalls_in_place(self):
        mgr = cem.CellposeEnvironmentManager()
        mgr.is_env_created = lambda: True
        mgr.get_env_python_path = lambda: "/env/python"
        installed = []
        mgr._install_dependencies = lambda p: installed.append(p)

        mgr.reinstall_packages()
        assert installed == ["/env/python"]


class TestGetCellposeVersionInEnv:
    """Pins the import probe and the pip-metadata fallback."""

    def _manager(self, env_created=True):
        mgr = cem.CellposeEnvironmentManager()
        mgr.is_env_created = lambda: env_created
        mgr.get_env_python_path = lambda: "/env/python"
        return mgr

    def test_missing_env_returns_none(self, monkeypatch):
        _install_fake_subprocess(monkeypatch)
        assert (
            self._manager(env_created=False).get_cellpose_version_in_env()
            is None
        )

    def test_import_probe_supplies_the_version(self, monkeypatch):
        commands = []

        def _run(cmd, **kwargs):
            commands.append(list(cmd))
            return _Completed(stdout=" 4.3.0 \n")

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().get_cellpose_version_in_env() == "4.3.0"
        # A successful import probe must not fall through to pip.
        assert len(commands) == 1
        assert commands[0][:2] == ["/env/python", "-c"]
        assert "import cellpose" in commands[0][2]

    def test_empty_import_probe_falls_back_to_pip_show(self, monkeypatch):
        def _run(cmd, **kwargs):
            if "pip" in cmd:
                return _Completed(stdout="Name: cellpose\nVersion: 4.2.1.1\n")
            return _Completed(stdout="\n")

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().get_cellpose_version_in_env() == "4.2.1.1"

    def test_failing_import_probe_falls_back_to_pip_show(self, monkeypatch):
        def _run(cmd, **kwargs):
            if "pip" in cmd:
                return _Completed(stdout="version: 9.9\n")
            raise subprocess.CalledProcessError(1, cmd)

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().get_cellpose_version_in_env() == "9.9"

    def test_pip_show_failure_returns_none(self, monkeypatch):
        def _run(cmd, **kwargs):
            if "pip" in cmd:
                return _Completed(stdout="whatever", returncode=1)
            return _Completed(stdout="")

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().get_cellpose_version_in_env() is None

    def test_pip_show_without_version_line_returns_none(self, monkeypatch):
        def _run(cmd, **kwargs):
            if "pip" in cmd:
                return _Completed(stdout="Name: cellpose\nVersion:\n")
            return _Completed(stdout="")

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().get_cellpose_version_in_env() is None

    def test_pip_show_exception_returns_none(self, monkeypatch):
        def _run(cmd, **kwargs):
            if "pip" in cmd:
                raise OSError("boom")
            return _Completed(stdout="")

        _install_fake_subprocess(monkeypatch, run=_run)
        assert self._manager().get_cellpose_version_in_env() is None


class TestEnsureMinimumCellposeVersion:
    """Pins the rebuild decision and the failure report."""

    def _manager(self, versions, rebuilds=None):
        mgr = cem.CellposeEnvironmentManager()
        mgr.is_env_created = lambda: True
        mgr.get_env_python_path = lambda: "/env/python"
        mgr.create_env = lambda: (
            rebuilds.append("create_env") if rebuilds is not None else None
        )
        mgr.get_cellpose_version_in_env = lambda: versions.pop(0)
        return mgr

    def test_sufficient_version_does_not_rebuild_the_environment(
        self, monkeypatch, capsys
    ):
        # create_env() wipes the existing env, so an unnecessary rebuild is
        # destructive and slow: prove it is skipped.
        def _unexpected(*a, **k):
            raise AssertionError("no subprocess should run on the happy path")

        _install_fake_subprocess(monkeypatch, run=_unexpected)
        rebuilds = []
        mgr = self._manager(["4.3.0"], rebuilds)

        assert mgr.ensure_minimum_cellpose_version("4.2.1.1") is None
        assert rebuilds == []
        assert (
            "Cellpose version check passed: 4.3.0 >= 4.2.1.1"
            in capsys.readouterr().out
        )

    def test_outdated_version_is_rebuilt_once_and_accepted(
        self, monkeypatch, capsys
    ):
        _install_fake_subprocess(monkeypatch)
        rebuilds = []
        mgr = self._manager(["4.0.0", "4.2.1.1"], rebuilds)

        assert mgr.ensure_minimum_cellpose_version("4.2.1.1") is None
        assert rebuilds == ["create_env"]
        out = capsys.readouterr().out
        assert "'4.0.0' < 4.2.1.1" in out
        assert (
            "Cellpose environment upgraded successfully: 4.2.1.1 >= 4.2.1.1"
            in out
        )

    def test_default_minimum_is_the_module_constant(self, monkeypatch):
        _install_fake_subprocess(
            monkeypatch, run=lambda *a, **k: _Completed(stdout="")
        )
        mgr = self._manager(["1.0", "1.0"])
        with pytest.raises(RuntimeError) as excinfo:
            mgr.ensure_minimum_cellpose_version()
        # Pinned literally: an accidental downgrade of the module constant
        # must be visible here.
        assert cem.MIN_CELLPOSE_VERSION == "4.2.1.1"
        assert "required minimum 4.2.1.1" in str(excinfo.value)

    def test_missing_version_is_treated_as_too_old(self, monkeypatch):
        _install_fake_subprocess(
            monkeypatch, run=lambda *a, **k: _Completed(stdout="")
        )
        rebuilds = []
        mgr = self._manager([None, "4.3.0"], rebuilds)
        mgr.ensure_minimum_cellpose_version("4.2.1.1")
        assert rebuilds == ["create_env"]

    def test_still_outdated_after_rebuild_raises_with_pip_debug(
        self, monkeypatch
    ):
        _install_fake_subprocess(
            monkeypatch,
            run=lambda *a, **k: _Completed(
                stdout="Version: 4.0.0\n", stderr="pip warning\n"
            ),
        )
        mgr = self._manager(["4.0.0", "4.0.0"])
        with pytest.raises(RuntimeError) as excinfo:
            mgr.ensure_minimum_cellpose_version("4.2.1.1")

        message = str(excinfo.value)
        assert "still below" in message
        assert "pip show cellpose output:" in message
        assert "pip warning" in message

    def test_pip_probe_exception_is_reported_in_the_error(self, monkeypatch):
        def _run(*a, **k):
            raise OSError("no pip")

        _install_fake_subprocess(monkeypatch, run=_run)
        mgr = self._manager([None, None])
        with pytest.raises(RuntimeError) as excinfo:
            mgr.ensure_minimum_cellpose_version("4.2.1.1")
        assert "pip show probe failed: no pip" in str(excinfo.value)

    def test_empty_pip_output_is_rendered_as_placeholder(self, monkeypatch):
        _install_fake_subprocess(
            monkeypatch, run=lambda *a, **k: _Completed(stdout="  ")
        )
        mgr = self._manager(["1.0", "1.0"])
        with pytest.raises(RuntimeError) as excinfo:
            mgr.ensure_minimum_cellpose_version("4.2.1.1")
        assert "<empty>" in str(excinfo.value)


class TestModuleLevelWrappers:
    """Pins that the thin module functions delegate to the manager."""

    def test_wrappers_delegate(self, monkeypatch):
        recorded = []

        class _Recorder:
            def is_package_installed(self):
                recorded.append("is_package_installed")
                return True

            def is_env_created(self):
                recorded.append("is_env_created")
                return False

            def get_env_python_path(self):
                recorded.append("get_env_python_path")
                return "/env/python"

            def create_env(self):
                recorded.append("create_env")
                return "/env"

            def are_all_packages_installed(self):
                recorded.append("are_all_packages_installed")
                return True

            def reinstall_packages(self):
                recorded.append("reinstall_packages")
                return None

        monkeypatch.setattr(cem, "manager", _Recorder())

        assert cem.is_cellpose_installed() is True
        assert cem.is_env_created() is False
        assert cem.get_env_python_path() == "/env/python"
        assert cem.create_cellpose_env() == "/env"
        assert cem.check_cellpose_packages() is True
        assert cem.reinstall_cellpose_packages() is None
        assert recorded == [
            "is_package_installed",
            "is_env_created",
            "get_env_python_path",
            "create_env",
            "are_all_packages_installed",
            "reinstall_packages",
        ]

    def test_cancel_wrapper_calls_cancel_all_processes(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            cem, "cancel_all_processes", lambda: seen.append(True)
        )
        cem.cancel_cellpose_processing()
        assert seen == [True]


class TestRunCellposeInEnv:
    """Pins the setup gate and the zarr-vs-legacy dispatch."""

    def test_missing_env_is_created_first(self, monkeypatch):
        order = []
        monkeypatch.setattr(cem, "is_env_created", lambda: False)
        monkeypatch.setattr(
            cem, "create_cellpose_env", lambda: order.append("create")
        )
        stub = _StubManager()
        monkeypatch.setattr(cem, "manager", stub)
        seen = {}

        def _legacy(args):
            seen["args"] = args
            return "legacy"

        monkeypatch.setattr(cem, "run_legacy_processing", _legacy)

        original = {"image": 1, "model_type": "cyto3"}
        assert cem.run_cellpose_in_env("eval", original) == "legacy"
        assert order == ["create"]
        # Packages were reported present, so no reinstall must happen.
        assert stub.calls == [
            "check",
            ("ensure", "4.2.1.1"),
        ]
        # Args reach the legacy path unchanged (but as a defensive copy).
        assert seen["args"] == original
        assert seen["args"] is not original

    def test_existing_env_is_not_recreated(self, monkeypatch):
        monkeypatch.setattr(cem, "is_env_created", lambda: True)

        def _unexpected():
            raise AssertionError("must not recreate an existing env")

        monkeypatch.setattr(cem, "create_cellpose_env", _unexpected)
        monkeypatch.setattr(cem, "manager", _StubManager())
        monkeypatch.setattr(
            cem, "run_legacy_processing", lambda args: "legacy"
        )
        assert cem.run_cellpose_in_env("eval", {"image": 1}) == "legacy"

    def test_missing_packages_trigger_reinstall(self, monkeypatch, capsys):
        monkeypatch.setattr(cem, "is_env_created", lambda: True)
        stub = _StubManager(packages_ok=False)
        monkeypatch.setattr(cem, "manager", stub)
        monkeypatch.setattr(
            cem, "run_legacy_processing", lambda args: "legacy"
        )

        assert cem.run_cellpose_in_env("eval", {"image": 1}) == "legacy"
        # Reinstall happens before the version gate, not after.
        assert stub.calls == [
            "check",
            "reinstall",
            ("ensure", "4.2.1.1"),
        ]
        assert "Missing packages detected" in capsys.readouterr().out

    def test_zarr_path_dispatches_to_zarr_processing(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(cem, "is_env_created", lambda: True)
        monkeypatch.setattr(cem, "manager", _StubManager())

        def _unexpected(args):
            raise AssertionError("zarr input must not take the legacy path")

        monkeypatch.setattr(cem, "run_legacy_processing", _unexpected)
        seen = {}

        def _zarr(path, args):
            seen["path"] = path
            seen["args"] = args
            return "zarr-result"

        monkeypatch.setattr(cem, "run_zarr_processing", _zarr)

        original = {"zarr_path": "/data/img.zarr", "use_gpu": False}
        assert cem.run_cellpose_in_env("eval", original) == "zarr-result"
        assert seen["path"] == "/data/img.zarr"
        assert seen["args"] == original
        assert seen["args"] is not original
        assert (
            "Using optimized zarr processing for: /data/img.zarr"
            in capsys.readouterr().out
        )

    def test_version_gate_failure_aborts_before_any_processing(
        self, monkeypatch
    ):
        monkeypatch.setattr(cem, "is_env_created", lambda: True)

        class _Failing(_StubManager):
            def ensure_minimum_cellpose_version(self, minimum):
                raise RuntimeError("cellpose too old")

        monkeypatch.setattr(cem, "manager", _Failing())

        def _unexpected(*a, **k):
            raise AssertionError("processing must not start")

        monkeypatch.setattr(cem, "run_legacy_processing", _unexpected)
        monkeypatch.setattr(cem, "run_zarr_processing", _unexpected)

        with pytest.raises(RuntimeError, match="cellpose too old"):
            cem.run_cellpose_in_env("eval", {"image": 1})


# ---------------------------------------------------------------------
# Zarr / legacy processing drivers
# ---------------------------------------------------------------------


def _zarr_popen(capture, result=None, returncode=0, lines=(), stdout=_UNSET):
    """Popen double that plays the role of the generated zarr script."""

    def _popen(cmd, **kwargs):
        capture["cmd"] = list(cmd)
        capture["kwargs"] = kwargs
        with open(cmd[1]) as handle:
            script = handle.read()
        capture["script"] = script
        # The generated worker never runs here, so nothing else would catch a
        # broken f-string template: prove it is valid Python.
        compile(script, cmd[1], "exec")
        match = re.search(r"zarr\.save\('([^']+)'", script)
        capture["output_path"] = match.group(1)
        if result is not None:
            zarr.save(match.group(1), result)
        proc = _FakeProcess(lines=lines, returncode=returncode)
        if stdout is not _UNSET:
            proc.stdout = stdout
        return proc

    return _popen


class TestRunZarrProcessing:
    """Pins staging, script generation, streaming and cleanup."""

    def _prepare(self, tmp_path):
        source = tmp_path / "input.zarr"
        zarr.save(str(source), np.zeros((2, 2), dtype=np.uint16))
        return source, tmp_path / "tmp"

    def test_temporary_output_is_read_back_then_deleted(
        self, tmp_path, monkeypatch, capsys
    ):
        source, tmp_root = self._prepare(tmp_path)
        expected = np.arange(4, dtype=np.uint32).reshape(2, 2)
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_zarr_popen(
                capture,
                result=expected,
                lines=[
                    "DISTRIBUTED_PROGRESS_TOTAL=2\n",
                    "\n",
                    "RUNNING BLOCK: 1\n",
                    "cellpose version: 4.2\n",
                    "segmentation done\n",
                ],
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        result = cem.run_zarr_processing(str(source), {})

        assert np.array_equal(result, expected)
        assert result.dtype == np.uint32
        assert capture["kwargs"]["cwd"] == str(tmp_root)
        assert capture["cmd"][0] == "/env/py"
        assert capture["output_path"].endswith("_cellpose_out.zarr")
        # Temporary zarr and script are cleaned up.
        assert not os.path.exists(capture["output_path"])
        assert not os.path.exists(capture["cmd"][1])
        assert cem._running_processes == []

        out = capsys.readouterr().out
        assert "Distributed Cellpose progress: 0/2 blocks (0.0%)" in out
        assert "Distributed Cellpose progress: 1/2 blocks (50.0%)" in out
        assert "cellpose version: 4.2" not in out
        assert "segmentation done" in out

    def test_missing_stdout_pipe_is_tolerated(self, tmp_path, monkeypatch):
        source, _ = self._prepare(tmp_path)
        expected = np.full((2, 2), 7, dtype=np.uint32)
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_zarr_popen(capture, result=expected, lines=None),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        result = cem.run_zarr_processing(str(source), {})
        assert np.array_equal(result, expected)

    def test_persisted_output_path_is_used_and_kept(
        self, tmp_path, monkeypatch
    ):
        source, _ = self._prepare(tmp_path)
        persist = tmp_path / "cache" / "t000.zarr"
        expected = np.ones((2, 2), dtype=np.uint32)
        capture = {}
        _install_fake_subprocess(
            monkeypatch, Popen=_zarr_popen(capture, result=expected)
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        result = cem.run_zarr_processing(
            str(source),
            {
                "persist_output_zarr_path": str(persist),
                "distributed_n_workers": 3,
                "distributed_blocksize_yx": 512,
                "model_type": "cyto3",
            },
        )

        assert np.array_equal(result, expected)
        assert capture["output_path"] == str(persist)
        assert persist.exists()
        assert "'n_workers': 3," in capture["script"]
        assert "BLOCKSIZE = 512" in capture["script"]
        assert "_MODEL_TYPE = 'cyto3'" in capture["script"]

    def test_default_parameters_are_baked_into_the_script(
        self, tmp_path, monkeypatch
    ):
        source, _ = self._prepare(tmp_path)
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_zarr_popen(
                capture, result=np.zeros((1, 1), dtype=np.uint32)
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_zarr_processing(str(source), {})
        script = capture["script"]
        for expected in [
            "USE_DISTRIBUTED = False",
            "BLOCKSIZE = 256",
            "BLOCKSIZE_Z = 256",
            "MASK_PATH = None",
            "MASK_ZARR_PATH = None",
            "TIMEPOINT_INDEX = None",
            "'flow_threshold': 0.4,",
            "'cellprob_threshold': 0.0,",
            "'do_3D': False,",
            "'z_axis': None,",
            "'normalize': {'tile_norm_blocksize': 128},",
            "'batch_size': 32,",
            "'flow3D_smooth': 0,",
            "_DIAMETER = 0.0",
            "_ANISOTROPY = None",
            "_MODEL_KWARGS = {'gpu': True}",
            "_MODEL_TYPE = 'cpsam_v2'",
            "selected_channel = 'all'",
            f"zarr.open('{source}', mode='r')",
        ]:
            assert expected in script, expected

    def test_caller_parameters_are_baked_into_the_script(
        self, tmp_path, monkeypatch
    ):
        source, _ = self._prepare(tmp_path)
        mask = tmp_path / "mask.tif"
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_zarr_popen(
                capture, result=np.zeros((1, 1), dtype=np.uint32)
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_zarr_processing(
            str(source),
            {
                "use_distributed_segmentation": True,
                "distributed_blocksize": 128,  # fallback for both axes
                "distributed_mask_path": str(mask),
                "timepoint_index": 4,
                "channel": 2,
                "flow_threshold": 0.9,
                "cellprob_threshold": -1.5,
                "do_3D": True,
                "z_axis": 0,
                "batch_size": 8,
                "flow3D_smooth": 3,
                "diameter": 30.0,
                "anisotropy": 2.5,
                "use_gpu": False,
                "model_type": "cpdino",
            },
        )
        script = capture["script"]
        for expected in [
            "USE_DISTRIBUTED = True",
            "BLOCKSIZE = 128",
            "BLOCKSIZE_Z = 128",
            f"MASK_PATH = {str(mask)!r}",
            "TIMEPOINT_INDEX = 4",
            "selected_channel = 2",
            "'flow_threshold': 0.9,",
            "'cellprob_threshold': -1.5,",
            "'do_3D': True,",
            "'z_axis': 0,",
            "'batch_size': 8,",
            "'flow3D_smooth': 3,",
            "_DIAMETER = 30.0",
            "_ANISOTROPY = 2.5",
            "_MODEL_KWARGS = {'gpu': False}",
            "_MODEL_TYPE = 'cpdino'",
            "use_gpu_requested = False",
        ]:
            assert expected in script, expected

    @pytest.mark.parametrize("workers", ["abc", None, 0, -4])
    def test_worker_count_is_sanitised(self, tmp_path, monkeypatch, workers):
        source, _ = self._prepare(tmp_path)
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_zarr_popen(
                capture, result=np.zeros((1, 1), dtype=np.uint32)
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_zarr_processing(
            str(source), {"distributed_n_workers": workers}
        )
        assert "'n_workers': 1," in capture["script"]

    def test_unwritable_tmp_folder_raises_permission_error(
        self, tmp_path, monkeypatch
    ):
        source, tmp_root = self._prepare(tmp_path)
        real_access = os.access

        def _access(path, mode, **kwargs):
            if str(path) == str(tmp_root):
                return False
            return real_access(path, mode, **kwargs)

        monkeypatch.setattr(cem.os, "access", _access)
        with pytest.raises(PermissionError, match="not writable"):
            cem.run_zarr_processing(str(source), {})

    def test_non_zero_return_code_raises(self, tmp_path, monkeypatch):
        source, _ = self._prepare(tmp_path)
        capture = {}
        _install_fake_subprocess(
            monkeypatch, Popen=_zarr_popen(capture, returncode=3)
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        with pytest.raises(RuntimeError, match="return code 3"):
            cem.run_zarr_processing(str(source), {})
        assert not os.path.exists(capture["cmd"][1])
        # The scratch output dir must not survive a failed run.
        assert not os.path.exists(capture["output_path"])
        assert cem._running_processes == []

    def test_process_is_tracked_while_it_streams(self, tmp_path, monkeypatch):
        source, _ = self._prepare(tmp_path)
        capture = {}
        seen_during_stream = []

        def _lines():
            seen_during_stream.append(list(cem._running_processes))
            yield "working\n"

        _install_fake_subprocess(
            monkeypatch,
            Popen=_zarr_popen(
                capture,
                result=np.zeros((1, 1), dtype=np.uint32),
                stdout=_lines(),
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_zarr_processing(str(source), {})
        # Registered for cancellation for the whole streaming phase...
        assert len(seen_during_stream) == 1
        assert len(seen_during_stream[0]) == 1
        # ...and deregistered afterwards.
        assert cem._running_processes == []

    def test_missing_output_zarr_raises(self, tmp_path, monkeypatch):
        source, _ = self._prepare(tmp_path)
        capture = {}
        _install_fake_subprocess(
            monkeypatch, Popen=_zarr_popen(capture, result=None)
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        with pytest.raises(RuntimeError, match="Output zarr was not created"):
            cem.run_zarr_processing(
                str(source),
                {"persist_output_zarr_path": str(tmp_path / "c" / "o.zarr")},
            )


def _legacy_popen(capture, masks=None, returncode=0, lines=()):
    """Popen double that plays the role of the generated TIFF script."""

    def _popen(cmd, **kwargs):
        capture["cmd"] = list(cmd)
        with open(cmd[2]) as handle:
            script = handle.read()
        capture["script"] = script
        # Nothing executes the generated worker in these tests, so a broken
        # f-string template would otherwise go unnoticed.
        compile(script, cmd[2], "exec")
        capture["input_path"] = re.search(
            r"tifffile\.imread\('([^']+)'\)", script
        ).group(1)
        capture["input_image"] = tifffile.imread(capture["input_path"])
        capture["output_path"] = re.search(
            r"tifffile\.imwrite\('([^']+)', masks\)", script
        ).group(1)
        if masks is not None:
            tifffile.imwrite(capture["output_path"], masks)
        return _FakeProcess(lines=lines, returncode=returncode)

    return _popen


class TestRunLegacyProcessing:
    """Pins the TIFF round trip, script parameters and cleanup."""

    def test_round_trip_writes_input_and_returns_masks(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(cem.tempfile, "tempdir", str(tmp_path))
        image = np.arange(9, dtype=np.uint16).reshape(3, 3)
        masks = (np.arange(9, dtype=np.uint32) * 2).reshape(3, 3)
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_legacy_popen(
                capture,
                masks=masks,
                lines=[
                    "CELLPOSE_PROGRESS=50\n",
                    "CELLPOSE_PROGRESS=50\n",
                    "[GUI INFO] : WRITING LOG OUTPUT TO /x\n",
                    "eval finished\n",
                ],
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        result = cem.run_legacy_processing(
            {
                "image": image,
                "use_gpu": False,
                "model_type": "cyto3",
                "flow_threshold": 0.7,
            }
        )

        assert np.array_equal(result, masks)
        assert np.array_equal(capture["input_image"], image)
        assert capture["cmd"][:2] == ["/env/py", "-u"]
        assert "gpu=False," in capture["script"]
        assert "model_type='cyto3')" in capture["script"]
        assert "flow_threshold=0.7," in capture["script"]
        # Every temporary file is removed once the result is read.
        assert not os.path.exists(capture["input_path"])
        assert not os.path.exists(capture["output_path"])
        assert not os.path.exists(capture["cmd"][2])
        assert cem._running_processes == []

        out = capsys.readouterr().out
        assert out.count("Cellpose progress: 50%") == 1
        assert "WRITING LOG OUTPUT TO" not in out
        assert "eval finished" in out

    def test_defaults_are_baked_into_the_script(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cem.tempfile, "tempdir", str(tmp_path))
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_legacy_popen(
                capture, masks=np.zeros((2, 2), dtype=np.uint32)
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_legacy_processing({"image": np.zeros((2, 2), dtype=np.uint16)})
        script = capture["script"]
        assert "gpu=True," in script
        assert "model_type='cpsam_v2')" in script
        assert "normalize = {'tile_norm_blocksize': 128}" in script
        assert "batch_size=32," in script
        assert "do_3D=False," in script

    def test_missing_stdout_pipe_is_tolerated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cem.tempfile, "tempdir", str(tmp_path))
        masks = np.full((2, 2), 5, dtype=np.uint32)
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_legacy_popen(capture, masks=masks, lines=None),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        result = cem.run_legacy_processing(
            {"image": np.zeros((2, 2), dtype=np.uint16)}
        )
        assert np.array_equal(result, masks)

    def test_non_zero_return_code_raises_and_cleans_up(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(cem.tempfile, "tempdir", str(tmp_path))
        capture = {}
        _install_fake_subprocess(
            monkeypatch, Popen=_legacy_popen(capture, returncode=2)
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        with pytest.raises(RuntimeError, match="return code 2"):
            cem.run_legacy_processing(
                {"image": np.zeros((2, 2), dtype=np.uint16)}
            )

        assert not os.path.exists(capture["input_path"])
        assert not os.path.exists(capture["output_path"])
        assert not os.path.exists(capture["cmd"][2])
        assert cem._running_processes == []
        assert "Error in Cellpose segmentation" in capsys.readouterr().out

    def test_caller_parameters_are_baked_into_the_script(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cem.tempfile, "tempdir", str(tmp_path))
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_legacy_popen(
                capture, masks=np.zeros((2, 2), dtype=np.uint32)
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_legacy_processing(
            {
                "image": np.zeros((2, 2), dtype=np.uint16),
                "use_gpu": False,
                "model_type": "cyto3",
                "channels": [1, 2],
                "flow_threshold": 0.9,
                "cellprob_threshold": -1.5,
                "batch_size": 8,
                "normalize": {"tile_norm_blocksize": 0},
                "do_3D": True,
                "flow3D_smooth": 3,
                "anisotropy": 2.5,
                "z_axis": 1,
                "channel_axis": 0,
            }
        )
        script = capture["script"]
        for expected in [
            "gpu=False,",
            "model_type='cyto3')",
            "channels=[1, 2],",
            "flow_threshold=0.9,",
            "cellprob_threshold=-1.5,",
            "batch_size=8,",
            "normalize = {'tile_norm_blocksize': 0}",
            "do_3D=True,",
            "flow3D_smooth=3,",
            "anisotropy=2.5,",
            # z_axis is only forwarded for 3D runs.
            "z_axis=1 if True else None,",
            "channel_axis=0,",
        ]:
            assert expected in script, expected

    def test_z_axis_is_suppressed_for_2d_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cem.tempfile, "tempdir", str(tmp_path))
        capture = {}
        _install_fake_subprocess(
            monkeypatch,
            Popen=_legacy_popen(
                capture, masks=np.zeros((2, 2), dtype=np.uint32)
            ),
        )
        monkeypatch.setattr(cem, "get_env_python_path", lambda: "/env/py")

        cem.run_legacy_processing(
            {
                "image": np.zeros((2, 2), dtype=np.uint16),
                "z_axis": 2,
                "do_3D": False,
            }
        )
        assert "z_axis=2 if False else None," in capture["script"]

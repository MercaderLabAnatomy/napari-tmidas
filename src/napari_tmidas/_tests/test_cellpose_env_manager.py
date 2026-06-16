import napari_tmidas.processing_functions.cellpose_env_manager as cem
import pytest


def test_version_tuple_and_comparison_helpers():
    assert cem._version_tuple("v4.2.1.1") == (4, 2, 1, 1)
    assert cem._version_tuple("4.2.1") == (4, 2, 1)
    assert cem._is_version_at_least("4.2.1.1", "4.2.1.1")
    assert cem._is_version_at_least("4.2.2", "4.2.1.1")
    assert not cem._is_version_at_least("4.2.1", "4.2.1.1")


def test_ensure_minimum_cellpose_version_recreates_when_outdated(monkeypatch):
    manager = cem.CellposeEnvironmentManager()

    versions = ["4.1.0", "4.2.1.1"]
    created = {"count": 0}

    def fake_get_version():
        return versions.pop(0)

    def fake_create_env():
        created["count"] += 1

    monkeypatch.setattr(manager, "get_cellpose_version_in_env", fake_get_version)
    monkeypatch.setattr(manager, "create_env", fake_create_env)

    manager.ensure_minimum_cellpose_version("4.2.1.1")
    assert created["count"] == 1


def test_ensure_minimum_cellpose_version_skips_recreate_when_ok(monkeypatch):
    manager = cem.CellposeEnvironmentManager()

    created = {"count": 0}
    monkeypatch.setattr(manager, "get_cellpose_version_in_env", lambda: "4.2.1.1")
    monkeypatch.setattr(
        manager, "create_env", lambda: created.__setitem__("count", created["count"] + 1)
    )

    manager.ensure_minimum_cellpose_version("4.2.1.1")
    assert created["count"] == 0


def test_run_cellpose_in_env_enforces_version_check(monkeypatch):
    called = {"version_check": 0}

    monkeypatch.setattr(cem, "is_env_created", lambda: True)
    monkeypatch.setattr(cem.manager, "are_all_packages_installed", lambda: True)
    monkeypatch.setattr(
        cem.manager,
        "ensure_minimum_cellpose_version",
        lambda _min: called.__setitem__("version_check", called["version_check"] + 1),
    )
    monkeypatch.setattr(cem, "run_legacy_processing", lambda _args: "ok")

    result = cem.run_cellpose_in_env("eval", {"image": "dummy"})
    assert result == "ok"
    assert called["version_check"] == 1


def test_get_cellpose_version_falls_back_to_pip_show(monkeypatch):
    manager = cem.CellposeEnvironmentManager()

    class _Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    calls = {"n": 0}

    def fake_run(*_args, **_kwargs):
        calls["n"] += 1
        # First probe (import cellpose) fails, second probe (pip show) succeeds.
        if calls["n"] == 1:
            raise cem.subprocess.CalledProcessError(1, "python -c")
        return _Result(stdout="Name: cellpose\nVersion: 4.2.1.1\n")

    monkeypatch.setattr(manager, "is_env_created", lambda: True)
    monkeypatch.setattr(manager, "get_env_python_path", lambda: "python")
    monkeypatch.setattr(cem.subprocess, "run", fake_run)

    assert manager.get_cellpose_version_in_env() == "4.2.1.1"


def test_ensure_minimum_cellpose_version_raises_with_debug_info(monkeypatch):
    manager = cem.CellposeEnvironmentManager()

    monkeypatch.setattr(manager, "get_cellpose_version_in_env", lambda: None)
    monkeypatch.setattr(manager, "create_env", lambda: None)
    monkeypatch.setattr(manager, "get_env_python_path", lambda: "python")

    class _Result:
        def __init__(self):
            self.returncode = 0
            self.stdout = "Name: cellpose\nVersion: 4.0.0\n"
            self.stderr = ""

    monkeypatch.setattr(cem.subprocess, "run", lambda *_a, **_k: _Result())

    with pytest.raises(RuntimeError) as excinfo:
        manager.ensure_minimum_cellpose_version("4.2.1.1")

    msg = str(excinfo.value)
    assert "pip show cellpose output" in msg

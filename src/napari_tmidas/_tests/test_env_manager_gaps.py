"""
Locating uv, and deciding whether a dedicated environment can be reused,
when uv is missing or an environment's pyvenv.cfg is.
"""

import builtins
import os
import sys

import pytest

import napari_tmidas._env_manager as em


@pytest.fixture
def uv_allowed(monkeypatch):
    monkeypatch.delenv("NAPARI_TMIDAS_NO_UV", raising=False)


def _block_uv_package(monkeypatch):
    real_import = builtins.__import__

    def no_uv(name, *args, **kwargs):
        if name == "uv":
            raise ImportError("No module named 'uv'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "uv", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_uv)


def test_without_the_uv_package_a_uv_on_path_is_used(uv_allowed, monkeypatch):
    _block_uv_package(monkeypatch)
    monkeypatch.setattr(em.shutil, "which", lambda name: f"/opt/bin/{name}")

    assert em.find_uv() == "/opt/bin/uv"


def test_without_uv_anywhere_pip_is_used(uv_allowed, monkeypatch):
    _block_uv_package(monkeypatch)
    monkeypatch.setattr(em.shutil, "which", lambda name: None)

    assert em.find_uv() is None
    assert em.pip_command("/env/bin/python", "install", "x") == [
        "/env/bin/python",
        "-m",
        "pip",
        "install",
        "x",
    ]


def _fake_env(tmp_path, cfg=None):
    env = tmp_path / "env"
    python = em.env_python_path(str(env))
    os.makedirs(os.path.dirname(python))
    open(python, "w").close()
    if cfg is not None:
        (env / "pyvenv.cfg").write_text(cfg)
    return str(env)


class TestEnvIsUsable:
    @pytest.fixture(autouse=True)
    def _uv_present(self, monkeypatch):
        monkeypatch.setattr(em, "find_uv", lambda: "/usr/bin/uv")

    def test_an_env_without_pyvenv_cfg_is_trusted(self, tmp_path):
        assert em.env_is_usable(_fake_env(tmp_path), "3.11") is True

    def test_an_env_whose_cfg_names_no_version_is_trusted(self, tmp_path):
        env = _fake_env(tmp_path, cfg="home = /usr/bin\n")

        assert em.env_is_usable(env, "3.11") is True

    def test_an_env_on_another_python_is_rebuilt(self, tmp_path):
        env = _fake_env(tmp_path, cfg="version_info = 3.12.4\n")

        assert em.env_is_usable(env, "3.11") is False

    def test_a_patch_release_of_the_wanted_python_is_usable(self, tmp_path):
        env = _fake_env(tmp_path, cfg="version = 3.11.9\n")

        assert em.env_is_usable(env, "3.11") is True


class TestCreateVenvWithoutUv:
    @pytest.fixture
    def calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(em, "find_uv", lambda: None)
        monkeypatch.setattr(
            em.venv, "create", lambda path, with_pip: calls.append(("venv", path))
        )
        monkeypatch.setattr(
            em.subprocess, "check_call", lambda cmd: calls.append(("run", cmd))
        )
        return calls

    def test_an_existing_env_is_replaced(self, tmp_path, calls):
        env = tmp_path / "envs" / "old"
        env.mkdir(parents=True)
        (env / "stale.txt").write_text("from an old install")

        python = em.create_venv(str(env))

        assert not (env / "stale.txt").exists()
        assert calls[0] == ("venv", str(env))
        assert calls[1] == (
            "run",
            [python, "-m", "pip", "install", "--upgrade", "pip"],
        )

    def test_a_requested_python_that_cannot_be_provided_is_warned_about(
        self, tmp_path, calls, capsys
    ):
        em.create_venv(str(tmp_path / "envs" / "cellpose"), python_version="3.9")

        out = capsys.readouterr().out
        assert "cellpose needs Python 3.9" in out
        assert "uv is unavailable" in out

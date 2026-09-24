# src/napari_tmidas/_tests/test_env_manager.py
import os
import sys
import tempfile
from unittest.mock import Mock, patch

from napari_tmidas._env_manager import (
    BaseEnvironmentManager,
    find_uv,
    pip_command,
)


class MockEnvironmentManager(BaseEnvironmentManager):
    """Test implementation of BaseEnvironmentManager."""

    def __init__(self):
        super().__init__("test-env")

    def _install_dependencies(self, env_python: str) -> None:
        """Mock installation."""

    def is_package_installed(self) -> bool:
        """Mock package check."""
        return True


class TestBaseEnvironmentManager:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = MockEnvironmentManager()

    def teardown_method(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test manager initialization"""
        assert self.manager.env_name == "test-env"
        assert "test-env" in self.manager.env_dir

    def test_is_env_created_false(self):
        """Test is_env_created returns False when env doesn't exist"""
        assert not self.manager.is_env_created()

    @patch("napari_tmidas._env_manager.venv.create")
    @patch("napari_tmidas._env_manager.subprocess.check_call")
    def test_create_env(self, mock_subprocess, mock_venv):
        """Test environment creation"""
        env_python = self.manager.create_env()

        # Check that venv.create was called
        mock_venv.assert_called_once()

        # Check that pip upgrade was called
        mock_subprocess.assert_called()

        # Check that the returned path is correct
        assert env_python == self.manager.get_env_python_path()

    def test_get_env_python_path_linux(self):
        """Test getting Python path on Linux"""
        with patch(
            "napari_tmidas._env_manager.platform.system", return_value="Linux"
        ):
            path = self.manager.get_env_python_path()
            import os

            norm = os.path.normpath(path)
            assert os.path.join("bin", "python") in norm

    def test_get_env_python_path_windows(self):
        """Test getting Python path on Windows"""
        with patch(
            "napari_tmidas._env_manager.platform.system",
            return_value="Windows",
        ):
            path = self.manager.get_env_python_path()
            assert "Scripts" in path and "python.exe" in path

    def test_is_package_installed(self):
        """Test package installation check"""
        assert self.manager.is_package_installed()

    @patch("napari_tmidas._env_manager.subprocess.run")
    def test_run_in_env(self, mock_subprocess):
        """Test running command in environment"""
        mock_subprocess.return_value = Mock()
        result = self.manager.run_in_env("print('test')")

        mock_subprocess.assert_called_once()
        assert result is not None


class Py311EnvironmentManager(MockEnvironmentManager):
    python_version = "3.11"


def _fake_env(env_dir, cfg_line):
    """Lay out an environment with a python file and a pyvenv.cfg."""
    os.makedirs(os.path.join(env_dir, "bin"))
    open(os.path.join(env_dir, "bin", "python"), "w").close()
    with open(os.path.join(env_dir, "pyvenv.cfg"), "w") as cfg:
        cfg.write(f"home = /usr/bin\n{cfg_line}\n")


class TestUvIntegration:
    def test_no_uv_variable_disables_uv(self):
        # conftest sets NAPARI_TMIDAS_NO_UV for every test
        assert find_uv() is None

    def test_pip_command_without_uv(self):
        assert pip_command("/env/python", "install", "torch") == [
            "/env/python",
            "-m",
            "pip",
            "install",
            "torch",
        ]

    def test_pip_command_with_uv_targets_the_env(self):
        with patch(
            "napari_tmidas._env_manager.find_uv", return_value="/bin/uv"
        ):
            assert pip_command("/env/python", "uninstall", "-y", "x") == [
                "/bin/uv",
                "pip",
                "uninstall",
                "--python",
                "/env/python",
                "-y",
                "x",
            ]

    def test_find_uv_uses_bundled_binary(self, monkeypatch):
        monkeypatch.delenv("NAPARI_TMIDAS_NO_UV")
        uv = find_uv()
        assert uv is not None and os.path.exists(uv)

    @patch("napari_tmidas._env_manager.venv.create")
    @patch("napari_tmidas._env_manager.subprocess.check_call")
    def test_create_env_with_uv_uses_napari_python(
        self, mock_call, mock_venv, tmp_path
    ):
        manager = MockEnvironmentManager()
        manager.env_dir = str(tmp_path / "env")
        with patch(
            "napari_tmidas._env_manager.find_uv", return_value="/bin/uv"
        ):
            manager.create_env()

        mock_venv.assert_not_called()
        mock_call.assert_called_once_with(
            [
                "/bin/uv",
                "venv",
                "--seed",
                "--python",
                sys.executable,
                manager.env_dir,
            ]
        )

    @patch("napari_tmidas._env_manager.subprocess.check_call")
    def test_create_env_with_uv_requests_pinned_python(
        self, mock_call, tmp_path
    ):
        manager = Py311EnvironmentManager()
        manager.env_dir = str(tmp_path / "env")
        with patch(
            "napari_tmidas._env_manager.find_uv", return_value="/bin/uv"
        ):
            manager.create_env()

        assert mock_call.call_args[0][0][3:5] == ["--python", "3.11"]

    def test_env_on_wrong_python_counts_as_missing(self, tmp_path):
        manager = Py311EnvironmentManager()
        manager.env_dir = str(tmp_path / "env")
        _fake_env(manager.env_dir, "version = 3.12.13")
        with patch(
            "napari_tmidas._env_manager.find_uv", return_value="/bin/uv"
        ):
            assert not manager.is_env_created()

    def test_env_on_pinned_python_is_reused(self, tmp_path):
        manager = Py311EnvironmentManager()
        manager.env_dir = str(tmp_path / "env")
        _fake_env(manager.env_dir, "version_info = 3.11.9")
        with patch(
            "napari_tmidas._env_manager.find_uv", return_value="/bin/uv"
        ):
            assert manager.is_env_created()

    def test_without_uv_a_wrong_python_env_is_kept(self, tmp_path):
        # Rebuilding could not change the Python, so it would loop forever.
        manager = Py311EnvironmentManager()
        manager.env_dir = str(tmp_path / "env")
        _fake_env(manager.env_dir, "version = 3.12.13")
        assert manager.is_env_created()

# src/napari_tmidas/_tests/test_env_manager.py
import tempfile
from unittest.mock import Mock, patch

from napari_tmidas._env_manager import BaseEnvironmentManager


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

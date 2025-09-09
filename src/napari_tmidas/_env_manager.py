"""
Base environment manager for handling virtual environments.
"""

import os
import platform
import shutil
import subprocess
import venv
from abc import ABC, abstractmethod


class BaseEnvironmentManager(ABC):
    """Base class for managing virtual environments for different packages."""

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.env_dir = os.path.join(
            os.path.expanduser("~"), ".napari-tmidas", "envs", env_name
        )

    def is_env_created(self) -> bool:
        """Check if the dedicated environment exists."""
        env_python = self.get_env_python_path()
        return os.path.exists(env_python)

    def get_env_python_path(self) -> str:
        """Get the path to the Python executable in the environment."""
        if platform.system() == "Windows":
            return os.path.join(self.env_dir, "Scripts", "python.exe")
        else:
            return os.path.join(self.env_dir, "bin", "python")

    def create_env(self) -> str:
        """Create a dedicated virtual environment."""
        # Ensure the environment directory exists
        os.makedirs(os.path.dirname(self.env_dir), exist_ok=True)

        # Remove existing environment if it exists
        if os.path.exists(self.env_dir):
            shutil.rmtree(self.env_dir)

        print(f"Creating {self.env_name} environment at {self.env_dir}...")

        # Create a new virtual environment
        venv.create(self.env_dir, with_pip=True)

        # Path to the Python executable in the new environment
        env_python = self.get_env_python_path()

        # Upgrade pip
        print("Upgrading pip...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "--upgrade", "pip"]
        )

        # Install package-specific dependencies
        self._install_dependencies(env_python)

        print(f"{self.env_name} environment created successfully.")
        return env_python

    @abstractmethod
    def _install_dependencies(self, env_python: str) -> None:
        """Install package-specific dependencies."""

    @abstractmethod
    def is_package_installed(self) -> bool:
        """Check if the package is installed."""

    def run_in_env(
        self, command: str, **kwargs
    ) -> subprocess.CompletedProcess:
        """Run a command in the environment."""
        env_python = self.get_env_python_path()
        return subprocess.run([env_python, "-c", command], **kwargs)

"""
Base environment manager for handling virtual environments.

Environments are created and populated with uv, a dependency of this
package, which installs faster than pip and can provide a Python version
other than the one napari runs on. Without uv, or with
NAPARI_TMIDAS_NO_UV set (uv does not read pip.conf, so a pip-only
package mirror needs it), the venv module and pip are used instead.
"""

import os
import platform
import shutil
import subprocess
import sys
import venv
from abc import ABC, abstractmethod


def find_uv() -> str | None:
    """Return the path to the uv executable, or None to fall back to pip."""
    if os.environ.get("NAPARI_TMIDAS_NO_UV"):
        return None
    try:
        from uv import find_uv_bin

        return find_uv_bin()
    except (ImportError, FileNotFoundError):
        return shutil.which("uv")


def pip_command(env_python: str, subcommand: str, *args: str) -> list[str]:
    """Build a pip command that acts on the environment of ``env_python``.

    Only the command is built, not run: callers pass it to their own
    ``subprocess``, so tests that patch a manager module's ``subprocess``
    still intercept it.
    """
    uv = find_uv()
    if uv:
        return [uv, "pip", subcommand, "--python", env_python, *args]
    return [env_python, "-m", "pip", subcommand, *args]


def _read_env_python_version(env_dir: str) -> str | None:
    """Return the Python version recorded in an environment's pyvenv.cfg."""
    try:
        with open(os.path.join(env_dir, "pyvenv.cfg")) as cfg:
            for line in cfg:
                key, _, value = line.partition("=")
                # venv writes "version", uv writes "version_info".
                if key.strip() in ("version", "version_info"):
                    return value.strip()
    except OSError:
        pass
    return None


def env_dir_for(env_name: str) -> str:
    """Directory of the dedicated environment named ``env_name``."""
    return os.path.join(
        os.path.expanduser("~"), ".napari-tmidas", "envs", env_name
    )


def env_python_path(env_dir: str) -> str:
    """Path to the Python executable of the environment at ``env_dir``."""
    if platform.system() == "Windows":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def env_is_usable(env_dir: str, python_version: str | None = None) -> bool:
    """Whether the environment exists and, where it matters, is on
    ``python_version``.

    An environment on another Python counts as unusable, so callers rebuild
    it instead of reusing one whose pinned packages never installed. Without
    uv a rebuild could not change the Python, so the version is not checked.
    """
    if not os.path.exists(env_python_path(env_dir)):
        return False
    if python_version is None or find_uv() is None:
        return True
    version = _read_env_python_version(env_dir)
    if version is None:
        return True
    return version == python_version or version.startswith(
        python_version + "."
    )


def create_venv(env_dir: str, python_version: str | None = None) -> str:
    """Create an empty virtual environment at ``env_dir``, replacing any
    existing one, and return its Python path. Raises on failure."""
    os.makedirs(os.path.dirname(env_dir), exist_ok=True)
    if os.path.exists(env_dir):
        shutil.rmtree(env_dir)

    env_python = env_python_path(env_dir)
    uv = find_uv()
    if uv:
        # uv downloads the requested Python if it is not installed.
        # --seed adds pip, for anyone maintaining the env by hand.
        python = python_version or sys.executable
        subprocess.check_call(
            [uv, "venv", "--seed", "--python", python, env_dir]
        )
    else:
        if python_version:
            print(
                f"Warning: {os.path.basename(env_dir)} needs Python "
                f"{python_version}, but uv is unavailable; using "
                f"Python {platform.python_version()} instead."
            )
        venv.create(env_dir, with_pip=True)
        print("Upgrading pip...")
        subprocess.check_call(
            [env_python, "-m", "pip", "install", "--upgrade", "pip"]
        )
    return env_python


class BaseEnvironmentManager(ABC):
    """Base class for managing virtual environments for different packages."""

    # Python version for the environment, e.g. "3.11"; None uses the
    # interpreter napari runs on. Only uv can provide another version.
    python_version: str | None = None

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.env_dir = env_dir_for(env_name)

    def is_env_created(self) -> bool:
        """Check if the dedicated environment exists (see env_is_usable)."""
        return env_is_usable(self.env_dir, self.python_version)

    def get_env_python_path(self) -> str:
        """Get the path to the Python executable in the environment."""
        return env_python_path(self.env_dir)

    def create_env(self) -> str:
        """Create a dedicated virtual environment."""
        print(f"Creating {self.env_name} environment at {self.env_dir}...")
        env_python = create_venv(self.env_dir, self.python_version)

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

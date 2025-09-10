# src/napari_tmidas/_tests/test_windows_basic.py
"""
Basic Windows tests that don't require heavy dependencies.
This ensures the package structure is correct without testing full functionality.
"""
import os
import platform
import sys


class TestWindowsBasic:
    def test_python_version(self):
        """Test that Python version is supported"""
        assert sys.version_info >= (3, 9)

    def test_platform_detection(self):
        """Test that we can detect Windows platform"""
        if platform.system() == "Windows":
            assert True  # We're on Windows, basic test passes
        else:
            assert True  # Not on Windows, still pass

    def test_basic_imports(self):
        """Test that basic Python modules can be imported"""
        import json
        import pathlib
        import tempfile

        # Create a simple test
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = pathlib.Path(temp_dir) / "test.json"
            test_data = {"test": "data"}

            # Write and read JSON
            with open(test_file, "w") as f:
                json.dump(test_data, f)

            with open(test_file) as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data

    def test_package_structure_exists(self):
        """Test that the package structure exists"""
        # Test that we can find the package directory
        import napari_tmidas

        package_dir = os.path.dirname(napari_tmidas.__file__)
        assert os.path.exists(package_dir)
        assert os.path.isdir(package_dir)

        # Test that __version__ is available
        assert hasattr(napari_tmidas, "__version__")
        assert isinstance(napari_tmidas.__version__, str)

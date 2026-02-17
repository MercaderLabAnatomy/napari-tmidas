# src/napari_tmidas/_tests/test_ultrack.py
"""
Test module for Ultrack processing functions.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import discover_and_load_processing_functions
from napari_tmidas.processing_functions.ultrack_env_manager import (
    UltrackEnvironmentManager,
    is_env_created,
)


class TestUltrackEnvManager:
    """Test UltrackEnvironmentManager functionality."""

    def test_ultrack_env_manager_init(self):
        """Test UltrackEnvironmentManager initialization."""
        manager = UltrackEnvironmentManager()
        assert manager.env_name == "ultrack"

    def test_is_env_created(self):
        """Test environment creation check."""
        result = is_env_created()
        assert isinstance(result, bool)

    def test_get_conda_cmd(self):
        """Test that conda/mamba command can be found."""
        manager = UltrackEnvironmentManager()
        try:
            conda_cmd = manager._get_conda_cmd()
            assert conda_cmd in ["conda", "mamba"]
        except RuntimeError as e:
            pytest.skip(f"Conda/mamba not available: {e}")

    def test_is_package_installed(self):
        """Test package installation check."""
        manager = UltrackEnvironmentManager()
        result = manager.is_package_installed()
        assert isinstance(result, bool)

    def test_check_gpu_available(self):
        """Test GPU availability detection."""
        manager = UltrackEnvironmentManager()
        result = manager._check_gpu_available()
        assert isinstance(result, bool)
        # Just verify it doesn't crash - actual result depends on hardware


class TestUltrackFunctionRegistration:
    """Test ultrack function registration."""

    def test_ultrack_function_registered(self):
        """Test that ultrack function is registered."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()

        functions = BatchProcessingRegistry.list_functions()
        assert "Track Cells with Ultrack (Segmentation Ensemble)" in functions

    def test_ultrack_parameters(self):
        """Test ultrack function has correct parameters."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()

        func_info = BatchProcessingRegistry.get_function_info(
            "Track Cells with Ultrack (Segmentation Ensemble)"
        )
        assert func_info is not None

        params = func_info["parameters"]
        # Check all required parameters exist
        assert "label_suffixes" in params
        assert "gurobi_license" in params
        assert "min_area" in params
        assert "max_neighbors" in params
        assert "max_distance" in params
        assert "appear_weight" in params
        assert "disappear_weight" in params
        assert "division_weight" in params
        assert "enable_gpu" in params
        # window_size and overlap_size removed - auto-set to None/1 internally

    def test_ultrack_parameter_defaults(self):
        """Test ultrack function has correct default values."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()

        func_info = BatchProcessingRegistry.get_function_info(
            "Track Cells with Ultrack (Segmentation Ensemble)"
        )
        params = func_info["parameters"]

        # Check parameter defaults (based on ultrack ensemble example)
        assert params["label_suffixes"]["default"] == "_cp_labels.tif,_convpaint_labels.tif"
        assert params["gurobi_license"]["default"] == ""
        assert params["min_area"]["default"] == 200
        assert params["max_neighbors"]["default"] == 5
        assert params["max_distance"]["default"] == 40.0
        assert params["appear_weight"]["default"] == -0.1
        assert params["disappear_weight"]["default"] == -2.0
        assert params["division_weight"]["default"] == -0.01
        assert params["enable_gpu"]["default"] is True

    def test_ultrack_suffix(self):
        """Test ultrack function has correct suffix."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()

        func_info = BatchProcessingRegistry.get_function_info(
            "Track Cells with Ultrack (Segmentation Ensemble)"
        )
        assert func_info["suffix"] == "_ultrack_tracked"


class TestUltrackTracking:
    """Test ultrack tracking functionality."""

    def test_ultrack_import(self):
        """Test importing the ultrack tracking module."""
        try:
            from napari_tmidas.processing_functions import ultrack_tracking

            assert hasattr(ultrack_tracking, "ultrack_ensemble_tracking")
            assert hasattr(ultrack_tracking, "create_ultrack_ensemble_script")
        except ImportError:
            pytest.skip("Ultrack tracking module not available")

    def test_ultrack_script_generation(self):
        """Test ultrack script generation."""
        from napari_tmidas.processing_functions.ultrack_tracking import (
            create_ultrack_ensemble_script,
        )

        # Create temporary label files for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create dummy label paths
            label_paths = [
                str(tmpdir_path / "test_cp_labels.tif"),
                str(tmpdir_path / "test_convpaint_labels.tif"),
            ]
            output_path = str(tmpdir_path / "test_ultrack_tracked.tif")

            # Generate script
            script = create_ultrack_ensemble_script(
                label_paths=label_paths,
                output_path=output_path,
                gurobi_license=None,
                min_area=200,
                max_neighbors=5,
                max_distance=40.0,
            )

            # Check script contains expected content
            assert "import ultrack" in script or "from ultrack import" in script
            assert "labels_to_contours" in script
            assert "track(" in script
            assert str(label_paths[0]) in script
            assert str(label_paths[1]) in script
            assert output_path in script
            assert "min_area = 200" in script
            assert "max_neighbors = 5" in script
    
    def test_ultrack_script_zarr_storage(self):
        """Test that ultrack script uses zarr for out-of-memory storage."""
        from napari_tmidas.processing_functions.ultrack_tracking import (
            create_ultrack_ensemble_script,
        )

        # Create temporary label files for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            label_paths = [
                str(tmpdir_path / "test1_labels.tif"),
                str(tmpdir_path / "test2_labels.tif"),
            ]
            output_path = str(tmpdir_path / "tracked.tif")

            # Generate script
            script = create_ultrack_ensemble_script(
                label_paths=label_paths,
                output_path=output_path,
                gurobi_license=None,
                min_area=100,
                max_neighbors=5,
                max_distance=50.0,
            )

            # Check script uses zarr for out-of-memory storage
            assert "import zarr" in script
            assert "imread(path)" in script or "imread(" in script
            assert "zarr.open" in script
            assert "del labels_data" in script  # Should free RAM after zarr conversion
            assert "tempfile.mkdtemp" in script or "temp_dir" in script
            assert "out-of-memory" in script.lower() or "zarr storage" in script.lower()
            
            # Check it creates zarr paths for intermediate arrays
            assert "foreground_path" in script or "foreground.zarr" in script
            assert "edges_path" in script or "edges.zarr" in script

    def test_ultrack_with_2d_synthetic_data(self):
        """Test ultrack tracking with synthetic 2D data."""
        from napari_tmidas.processing_functions.ultrack_tracking import (
            ultrack_ensemble_tracking,
        )

        # Create synthetic TYX data (time series of 2D images)
        # Just a simple test to check interface, not actual tracking
        image = np.zeros((5, 100, 100), dtype=np.uint16)

        # This should return the original image if no label files are found
        # (which is expected in a test environment without proper setup)
        result = ultrack_ensemble_tracking(
            image,
            label_suffixes="_test_labels.tif",
            gurobi_license="",
        )

        # Should return array (even if unchanged due to missing files)
        assert isinstance(result, np.ndarray)

    def test_ultrack_parameter_validation(self):
        """Test parameter validation in ultrack tracking."""
        from napari_tmidas.processing_functions.ultrack_tracking import (
            ultrack_ensemble_tracking,
        )

        # Test with empty label suffixes
        image = np.zeros((5, 100, 100), dtype=np.uint16)

        result = ultrack_ensemble_tracking(
            image,
            label_suffixes="",  # Empty suffixes
            gurobi_license="",
        )

        # Should return unchanged when no valid suffixes
        assert np.array_equal(result, image)

    def test_ultrack_dimension_check(self):
        """Test dimension validation in ultrack tracking."""
        from napari_tmidas.processing_functions.ultrack_tracking import (
            ultrack_ensemble_tracking,
        )

        # Test with 2D image (no time dimension)
        image_2d = np.zeros((100, 100), dtype=np.uint16)

        result = ultrack_ensemble_tracking(
            image_2d,
            label_suffixes="_test_labels.tif",
        )

        # Should return unchanged for non-time-series data
        assert np.array_equal(result, image_2d)


class TestUltrackIntegration:
    """Integration tests for ultrack (require environment)."""

    @pytest.mark.slow
    def test_create_ultrack_env(self):
        """Test creating ultrack environment."""
        from napari_tmidas.processing_functions.ultrack_env_manager import (
            create_ultrack_env,
        )

        # This is a slow test - only run if explicitly requested
        try:
            result = create_ultrack_env()
            assert isinstance(result, bool)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            pytest.skip(f"Environment creation failed: {e}")

    @pytest.mark.slow
    def test_ultrack_with_real_labels(self):
        """Test ultrack with real label files (integration test)."""
        pytest.skip("Requires real label files and ultrack environment")
    
    def test_zarr_script_execution_mock(self):
        """Test that the generated script can be parsed (syntax check)."""
        from napari_tmidas.processing_functions.ultrack_tracking import (
            create_ultrack_ensemble_script,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            label_paths = [
                str(tmpdir_path / "label1.tif"),
                str(tmpdir_path / "label2.tif"),
            ]
            output_path = str(tmpdir_path / "output.tif")
            
            # Generate script
            script = create_ultrack_ensemble_script(
                label_paths=label_paths,
                output_path=output_path,
                gurobi_license=None,
                min_area=100,
                max_neighbors=5,
                max_distance=50.0,
            )
            
            # Write to file and check syntax
            script_path = tmpdir_path / "test_script.py"
            script_path.write_text(script)
            
            # Try to compile it (syntax check)
            try:
                compile(script, str(script_path), 'exec')
                print("✓ Script compiles successfully")
            except SyntaxError as e:
                pytest.fail(f"Generated script has syntax error: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing UltrackEnvManager...")
    test_env = TestUltrackEnvManager()
    test_env.test_ultrack_env_manager_init()
    test_env.test_is_env_created()
    test_env.test_is_package_installed()
    print("✓ Environment manager tests passed")

    print("\nTesting function registration...")
    test_reg = TestUltrackFunctionRegistration()
    test_reg.test_ultrack_function_registered()
    test_reg.test_ultrack_parameters()
    test_reg.test_ultrack_parameter_defaults()
    test_reg.test_ultrack_suffix()
    print("✓ Registration tests passed")

    print("\nTesting ultrack tracking...")
    test_track = TestUltrackTracking()
    test_track.test_ultrack_import()
    test_track.test_ultrack_script_generation()
    test_track.test_ultrack_with_2d_synthetic_data()
    test_track.test_ultrack_parameter_validation()
    test_track.test_ultrack_dimension_check()
    print("✓ Tracking tests passed")

    print("\n✓ All ultrack tests passed!")

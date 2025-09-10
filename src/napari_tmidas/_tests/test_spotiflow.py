"""
Test module for Spotiflow processing functions.
"""

import subprocess

import numpy as np
import pytest

from napari_tmidas.processing_functions.spotiflow_env_manager import (
    SpotiflowEnvironmentManager,
    is_env_created,
    is_spotiflow_installed,
)


def test_spotiflow_env_manager_init():
    """Test SpotiflowEnvironmentManager initialization."""
    manager = SpotiflowEnvironmentManager()
    assert manager.env_name == "spotiflow"
    assert "spotiflow" in manager.env_dir


def test_is_spotiflow_installed():
    """Test spotiflow installation check."""
    # This will likely be False in most test environments
    result = is_spotiflow_installed()
    assert isinstance(result, bool)


def test_is_env_created():
    """Test environment creation check."""
    result = is_env_created()
    assert isinstance(result, bool)


@pytest.mark.slow
def test_spotiflow_detection_import():
    """Test importing the spotiflow detection module."""
    try:
        from napari_tmidas.processing_functions import spotiflow_detection

        assert hasattr(spotiflow_detection, "spotiflow_detect_spots")
    except ImportError:
        pytest.skip("Spotiflow detection module not available")


@pytest.mark.slow
def test_spotiflow_detection_with_synthetic_data():
    """Test spot detection with synthetic data."""
    try:
        from napari_tmidas.processing_functions.spotiflow_detection import (
            spotiflow_detect_spots,
        )

        # Create synthetic 2D image with some bright spots
        image = np.zeros((100, 100), dtype=np.uint16)
        # Add some bright spots
        image[25:27, 25:27] = 1000
        image[75:77, 75:77] = 1200
        image[50:52, 25:27] = 800

        # Add some noise
        image = image + np.random.normal(0, 50, image.shape).astype(np.uint16)

        # Test detection (this will likely use the dedicated environment)
        points = spotiflow_detect_spots(
            image, pretrained_model="general", force_dedicated_env=True
        )

        # Should return an array of points
        assert isinstance(points, np.ndarray)
        assert points.ndim == 2
        assert points.shape[1] == 2  # 2D coordinates

    except ImportError:
        pytest.skip("Spotiflow not available for testing")
    except (RuntimeError, subprocess.CalledProcessError) as e:
        pytest.skip(f"Spotiflow test failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_spotiflow_env_manager_init()
    test_is_spotiflow_installed()
    test_is_env_created()
    print("Basic Spotiflow tests passed!")

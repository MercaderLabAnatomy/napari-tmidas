# Test VisCy virtual staining integration
"""
Tests for VisCy virtual staining processing function.
"""
import numpy as np
import pytest

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import discover_and_load_processing_functions


def test_viscy_registered():
    """Test that VisCy function is registered."""
    # Ensure processing functions are loaded
    discover_and_load_processing_functions()
    
    functions = BatchProcessingRegistry.list_functions()
    assert "VisCy Virtual Staining" in functions


def test_viscy_function_info():
    """Test that VisCy function has correct metadata."""
    # Ensure processing functions are loaded
    discover_and_load_processing_functions()
    
    info = BatchProcessingRegistry.get_function_info("VisCy Virtual Staining")
    
    assert info is not None
    assert "description" in info
    assert "parameters" in info
    assert info["suffix"] == "_virtual_stain"
    
    # Check parameters
    params = info["parameters"]
    assert "dim_order" in params
    assert "z_batch_size" in params
    assert "output_channel" in params
    
    # Check parameter defaults
    assert params["dim_order"]["default"] == "ZYX"
    assert params["z_batch_size"]["default"] == 15
    assert params["output_channel"]["default"] == "both"


def test_viscy_import():
    """Test that VisCy modules can be imported."""
    from napari_tmidas.processing_functions import viscy_env_manager
    from napari_tmidas.processing_functions import viscy_virtual_staining
    
    assert hasattr(viscy_env_manager, "ViscyEnvironmentManager")
    assert hasattr(viscy_virtual_staining, "viscy_virtual_staining")


def test_viscy_dimension_validation():
    """Test that VisCy validates input dimensions."""
    from napari_tmidas.processing_functions.viscy_virtual_staining import (
        viscy_virtual_staining,
    )
    
    # Test with 2D image (should fail)
    image_2d = np.random.rand(512, 512)
    with pytest.raises(ValueError, match="requires 3D images with Z dimension"):
        viscy_virtual_staining(image_2d, dim_order="YX")
    
    # Test with insufficient Z slices (should fail)
    image_3d_small = np.random.rand(10, 512, 512)  # Only 10 slices
    with pytest.raises(ValueError, match="at least 15 Z slices"):
        viscy_virtual_staining(image_3d_small, dim_order="ZYX")


def test_viscy_transpose_dimensions():
    """Test dimension transposition."""
    from napari_tmidas.processing_functions.viscy_virtual_staining import (
        transpose_dimensions,
    )
    
    # Test ZYX (no change needed)
    img = np.random.rand(15, 100, 100)
    transposed, new_order, has_time = transpose_dimensions(img, "ZYX")
    assert transposed.shape == (15, 100, 100)
    assert new_order == "ZYX"
    assert has_time is False
    
    # Test YXZ (should transpose)
    img = np.random.rand(100, 100, 15)
    transposed, new_order, has_time = transpose_dimensions(img, "YXZ")
    assert transposed.shape == (15, 100, 100)
    assert new_order == "ZYX"
    assert has_time is False
    
    # Test TZYX (no change needed)
    img = np.random.rand(5, 15, 100, 100)
    transposed, new_order, has_time = transpose_dimensions(img, "TZYX")
    assert transposed.shape == (5, 15, 100, 100)
    assert new_order == "TZYX"
    assert has_time is True


def test_viscy_env_manager():
    """Test VisCy environment manager."""
    from napari_tmidas.processing_functions.viscy_env_manager import (
        ViscyEnvironmentManager,
    )
    
    manager = ViscyEnvironmentManager()
    assert manager.env_name == "viscy"
    assert "viscy" in manager.env_dir
    assert "models" in manager.model_dir


def test_viscy_output_channel_options():
    """Test that output channel options are correctly defined."""
    # Ensure processing functions are loaded
    discover_and_load_processing_functions()
    
    info = BatchProcessingRegistry.get_function_info("VisCy Virtual Staining")
    params = info["parameters"]
    
    assert "output_channel" in params
    assert "options" in params["output_channel"]
    
    options = params["output_channel"]["options"]
    assert "both" in options
    assert "nuclei" in options
    assert "membrane" in options


if __name__ == "__main__":
    # Run basic tests
    test_viscy_registered()
    test_viscy_function_info()
    test_viscy_import()
    test_viscy_dimension_validation()
    test_viscy_transpose_dimensions()
    test_viscy_env_manager()
    test_viscy_output_channel_options()
    
    print("✓ All tests passed!")

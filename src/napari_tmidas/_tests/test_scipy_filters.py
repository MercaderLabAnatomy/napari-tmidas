# src/napari_tmidas/_tests/test_scipy_filters.py
import numpy as np

from napari_tmidas.processing_functions.scipy_filters import gaussian_blur


class TestScipyFilters:
    def test_gaussian_blur_basic(self):
        """Test basic gaussian blur functionality"""
        image = np.random.rand(100, 100)

        # Test with default parameters
        result = gaussian_blur(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_gaussian_blur_with_sigma(self):
        """Test gaussian blur with custom sigma"""
        image = np.random.rand(50, 50)

        # Test with sigma parameter
        result = gaussian_blur(image, sigma=2.0)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_gaussian_blur_3d(self):
        """Test gaussian blur on 3D image"""
        image = np.random.rand(20, 20, 20)

        result = gaussian_blur(image, sigma=1.0)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

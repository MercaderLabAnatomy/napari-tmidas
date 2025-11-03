# src/napari_tmidas/_tests/test_scipy_filters.py
import numpy as np
import pytest

from napari_tmidas.processing_functions import scipy_filters
from napari_tmidas.processing_functions.scipy_filters import gaussian_blur


class TestScipyFilters:
    def test_resize_labels(self):
        """Test resizing a label image by scale factor preserves label values and shape."""
        from napari_tmidas.processing_functions.scipy_filters import (
            resize_labels,
        )

        label_image = np.zeros((10, 10), dtype=np.uint8)
        label_image[2:8, 2:8] = 3
        # Test with float
        scale_factor = 0.5
        scaled = resize_labels(label_image, scale_factor=scale_factor)
        expected_shape = tuple(
            (np.array(label_image.shape) * scale_factor).astype(int)
        )
        assert scaled.shape == expected_shape
        assert set(np.unique(scaled)).issubset({0, 3})
        assert np.sum(scaled == 3) > 0

        # Test with string
        scale_factor_str = "0.5"
        scaled_str = resize_labels(label_image, scale_factor=scale_factor_str)
        assert scaled_str.shape == expected_shape
        assert set(np.unique(scaled_str)).issubset({0, 3})
        assert np.sum(scaled_str == 3) > 0

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

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_combined_output(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        label_image = np.zeros((9, 9), dtype=np.uint16)
        label_image[2:7, 2:7] = 1

        result = subdivide_labels_3layers(label_image)

        assert result.shape == label_image.shape
        unique_ids = set(np.unique(result))
        assert unique_ids.issubset({0, 1, 2, 3})
        assert {1, 2, 3}.issubset(unique_ids)

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_dtype_promotion(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        label_image = np.zeros((9, 9), dtype=np.uint8)
        label_image[2:7, 2:7] = 200

        result = subdivide_labels_3layers(label_image)

        assert result.dtype in (np.uint32, np.uint64)
        assert result.max() == 200 + 2 * 200

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_empty(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        label_image = np.zeros((5, 5, 5), dtype=np.uint16)

        result = subdivide_labels_3layers(label_image)

        np.testing.assert_array_equal(result, label_image)

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

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_half_body(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        # Create a simple half-spheroid-like object
        label_image = np.zeros((20, 20, 20), dtype=np.uint16)
        # Fill upper half with a sphere-like object
        for z in range(10, 20):
            for y in range(20):
                for x in range(20):
                    if (z - 15) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 25:
                        label_image[z, y, x] = 1

        # Test with half-body mode disabled (default)
        result_normal = subdivide_labels_3layers(
            label_image, is_half_body=False
        )
        assert result_normal.shape == label_image.shape
        unique_normal = np.unique(result_normal)
        assert len(unique_normal) > 1  # Should have background + layers

        # Test with half-body mode enabled (cut along Z-axis = 0)
        result_half_body = subdivide_labels_3layers(
            label_image, is_half_body=True, cut_axis=0
        )
        assert result_half_body.shape == label_image.shape
        unique_half = np.unique(result_half_body)
        assert len(unique_half) > 1  # Should have background + layers

        # Half-body mode should create different layer distributions
        # The cut surface (z=10 plane) should show multiple layers
        cut_surface_normal = result_normal[10, :, :]
        cut_surface_half = result_half_body[10, :, :]

        # In half-body mode, the cut surface should have more variety of layers
        assert len(np.unique(cut_surface_half[cut_surface_half > 0])) >= len(
            np.unique(cut_surface_normal[cut_surface_normal > 0])
        )

        # Test invalid cut_axis
        with pytest.raises(ValueError):
            subdivide_labels_3layers(
                label_image, is_half_body=True, cut_axis=5
            )

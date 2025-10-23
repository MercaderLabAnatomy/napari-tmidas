# src/napari_tmidas/_tests/test_skimage_filters.py
import numpy as np

from napari_tmidas.processing_functions.skimage_filters import (
    invert_image,
    simple_thresholding,
)


class TestSkimageFilters:
    def test_invert_image_basic(self):
        """Test basic image inversion functionality"""
        image = np.random.rand(100, 100)

        # Test with default parameters
        result = invert_image(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_invert_image_binary(self):
        """Test image inversion on binary image"""
        image = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        result = invert_image(image)
        # skimage.util.invert inverts all bits, so 0->255, 1->254 for uint8
        expected = np.array([[255, 254], [254, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_invert_image_3d(self):
        """Test image inversion on 3D image"""
        image = np.random.rand(20, 20, 20)

        result = invert_image(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_simple_thresholding_returns_uint32(self):
        """Test that manual thresholding returns uint8 with value 255 for proper display"""
        image = np.array([[0, 100, 200], [50, 150, 255]], dtype=np.uint8)

        result = simple_thresholding(image, threshold=128)

        # Check dtype is uint8
        assert result.dtype == np.uint8

        # Check values are binary (0 or 255)
        assert set(np.unique(result)).issubset({0, 255})

        # Check correct thresholding
        expected = np.array([[0, 0, 255], [0, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_simple_thresholding_different_thresholds(self):
        """Test manual thresholding with different threshold values"""
        image = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)

        # Test with low threshold
        result_low = simple_thresholding(image, threshold=50)
        assert result_low.dtype == np.uint8
        assert (
            np.sum(result_low == 255) > np.prod(result_low.shape) * 0.8
        )  # Most pixels above 50

        # Test with high threshold
        result_high = simple_thresholding(image, threshold=200)
        assert result_high.dtype == np.uint8
        assert (
            np.sum(result_high == 255) < np.prod(result_high.shape) * 0.3
        )  # Most pixels below 200

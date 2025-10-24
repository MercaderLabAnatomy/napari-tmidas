# src/napari_tmidas/_tests/test_skimage_filters.py
import numpy as np

from napari_tmidas.processing_functions.skimage_filters import (
    adaptive_threshold_bright,
    invert_image,
    percentile_threshold,
    rolling_ball_background,
    simple_thresholding,
)


class TestSkimageFilters:

    def test_resize_labels(self):
        """Test resizing a label image by scale factor preserves label values and shape."""
        from napari_tmidas.processing_functions.skimage_filters import (
            resize_labels,
        )

        label_image = np.zeros((10, 10), dtype=np.uint8)
        label_image[2:8, 2:8] = 3
        scale_factor = 0.5
        scaled = resize_labels(label_image, scale_factor=scale_factor)
        expected_shape = tuple(
            (np.array(label_image.shape) * scale_factor).astype(int)
        )
        assert scaled.shape == expected_shape
        # Should only contain 0 and 3
        assert set(np.unique(scaled)).issubset({0, 3})
        # Check that the central region is still present
        assert np.sum(scaled == 3) > 0

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


class TestBrightRegionExtraction:
    """Test suite for bright region extraction functions"""

    def test_percentile_threshold_original(self):
        """Test percentile thresholding with original values"""
        # Create image with gradient
        image = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)

        result = percentile_threshold(
            image, percentile=90, output_type="original"
        )

        # Only top 10% should remain
        assert result.shape == image.shape
        assert np.sum(result > 0) < image.size * 0.15  # Allow some margin
        assert result.max() == image.max()  # Original max value preserved

    def test_percentile_threshold_binary(self):
        """Test percentile thresholding with binary output"""
        image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)

        result = percentile_threshold(
            image, percentile=80, output_type="binary"
        )

        # Should be binary
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_rolling_ball_background_subtraction(self):
        """Test rolling ball background subtraction"""
        # Create image with uneven background and bright spot
        x, y = np.meshgrid(np.arange(100), np.arange(100))
        background = (50 + 30 * np.sin(x / 20) + 30 * np.sin(y / 20)).astype(
            np.uint8
        )
        image = background.copy()
        image[40:60, 40:60] += 150  # Add bright feature

        result = rolling_ball_background(image, radius=30)

        # Background should be reduced
        assert result.shape == image.shape
        # Center of bright spot should be brighter in result than in corners
        assert result[50, 50] > result[10, 10]

    def test_adaptive_threshold_bright(self):
        """Test adaptive thresholding with bright bias"""
        # Create image with varying brightness
        image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

        result = adaptive_threshold_bright(image, block_size=35, offset=-10.0)

        # Should be binary
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})
        assert result.shape == image.shape

    def test_adaptive_threshold_even_blocksize(self):
        """Test that even block size is handled correctly"""
        image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)

        # Should handle even block size by making it odd
        result = adaptive_threshold_bright(image, block_size=34, offset=0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

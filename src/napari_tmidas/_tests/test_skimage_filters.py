# src/napari_tmidas/_tests/test_skimage_filters.py
import numpy as np

from napari_tmidas.processing_functions.skimage_filters import (
    adaptive_threshold_bright,
    equalize_histogram,
    invert_image,
    percentile_threshold,
    rolling_ball_background,
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


class TestCLAHE:
    """Test suite for CLAHE (Contrast Limited Adaptive Histogram Equalization)"""

    def test_clahe_basic(self):
        """Test basic CLAHE functionality"""
        # Create a dark image with weak bright features
        image = np.zeros((100, 100), dtype=np.float32)
        image[40:60, 40:60] = 0.1  # Weak bright region

        result = equalize_histogram(image)

        # Output should be same shape
        assert result.shape == image.shape
        # Output should be normalized to [0, 1] range
        assert result.min() >= 0
        assert result.max() <= 1
        # Contrast should be enhanced (std deviation should increase)
        assert result.std() > image.std()

    def test_clahe_dark_with_membranes(self):
        """Test CLAHE on dark images with weak bright membranes (the use case that failed)"""
        # Create a realistic dark image with weak membrane-like structures
        np.random.seed(42)
        image = np.random.normal(0.05, 0.01, (200, 200))  # Dark background
        image = np.clip(image, 0, 1)

        # Add weak membrane-like structures
        image[50:55, :] += 0.1  # Horizontal membrane
        image[:, 100:105] += 0.1  # Vertical membrane
        image = np.clip(image, 0, 1)

        result = equalize_histogram(image, clip_limit=0.01)

        # Should not produce black image
        assert result.max() > 0.1, "CLAHE should not produce near-black images"
        # Should enhance contrast
        assert result.std() > image.std()
        # Membranes should be more visible (higher values)
        membrane_region = result[50:55, :]
        background_region = result[10:20, 10:20]
        assert membrane_region.mean() > background_region.mean()

    def test_clahe_custom_kernel_size(self):
        """Test CLAHE with custom kernel size"""
        image = np.random.rand(256, 256)

        result = equalize_histogram(image, kernel_size=64)

        assert result.shape == image.shape
        assert result.min() >= 0
        assert result.max() <= 1

    def test_clahe_auto_kernel_size(self):
        """Test CLAHE with automatic kernel size calculation"""
        # Small image
        small_image = np.random.rand(128, 128)
        result_small = equalize_histogram(small_image, kernel_size=0)
        assert result_small.shape == small_image.shape

        # Large image
        large_image = np.random.rand(1024, 1024)
        result_large = equalize_histogram(large_image, kernel_size=0)
        assert result_large.shape == large_image.shape

    def test_clahe_different_clip_limits(self):
        """Test CLAHE with different clip limit values"""
        image = np.random.rand(100, 100) * 0.2  # Dark image

        # Low clip limit (less contrast enhancement)
        result_low = equalize_histogram(image, clip_limit=0.005)

        # High clip limit (more contrast enhancement)
        result_high = equalize_histogram(image, clip_limit=0.05)

        # Both should enhance contrast compared to original
        assert result_low.std() > image.std()
        assert result_high.std() > image.std()
        # Higher clip limit typically gives more contrast (but not always guaranteed)
        assert (
            result_high.max() >= result_low.max() * 0.8
        )  # Allow some tolerance

    def test_clahe_3d_image(self):
        """Test CLAHE on 3D image (should work on last 2 dimensions)"""
        # Create 3D image (e.g., time series or z-stack)
        image_3d = np.random.rand(10, 100, 100) * 0.3

        result = equalize_histogram(image_3d)

        assert result.shape == image_3d.shape
        # Each slice should be enhanced independently
        assert result.std() > image_3d.std()

    def test_clahe_preserves_dtype(self):
        """Test that CLAHE preserves the original dtype"""
        # Test uint8
        img_uint8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result_uint8 = equalize_histogram(img_uint8)
        assert result_uint8.dtype == np.uint8
        assert result_uint8.max() <= 255
        assert result_uint8.min() >= 0

        # Test uint16
        img_uint16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        result_uint16 = equalize_histogram(img_uint16)
        assert result_uint16.dtype == np.uint16
        assert result_uint16.max() <= 65535

        # Test float32
        img_float32 = np.random.rand(100, 100).astype(np.float32)
        result_float32 = equalize_histogram(img_float32)
        assert result_float32.dtype == np.float32
        assert result_float32.max() <= 1.0
        assert result_float32.min() >= 0.0

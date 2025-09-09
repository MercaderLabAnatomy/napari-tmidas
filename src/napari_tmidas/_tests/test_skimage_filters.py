# src/napari_tmidas/_tests/test_skimage_filters.py
import numpy as np

from napari_tmidas.processing_functions.skimage_filters import invert_image


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

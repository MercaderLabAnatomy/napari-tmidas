# _tests/test_label_based_cropping.py
"""
Tests for label-based image cropping functionality.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from napari_tmidas.processing_functions.label_based_cropping import (
    _crop_image_with_label,
    _expand_label_to_3d,
    _expand_label_to_time,
    _get_label_image_filename,
    _infer_axes,
    label_based_cropping,
)

# Check if tifffile is available
try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


class TestLabelExpansion:
    """Test label expansion functions"""

    def test_expand_label_to_3d_simple(self):
        """Test expanding a 2D label to 3D"""
        label_2d = np.array([[1, 0], [0, 1]], dtype=np.uint32)
        z_size = 3

        result = _expand_label_to_3d(label_2d, z_size)

        assert result.shape == (3, 2, 2)
        # All z-slices should be identical
        assert np.allclose(result[0], label_2d)
        assert np.allclose(result[1], label_2d)
        assert np.allclose(result[2], label_2d)

    def test_expand_label_to_3d_zeros(self):
        """Test expanding a 2D label with zeros"""
        label_2d = np.zeros((4, 5), dtype=np.uint32)
        label_2d[1:3, 2:4] = 1

        result = _expand_label_to_3d(label_2d, 2)

        assert result.shape == (2, 4, 5)
        assert np.array_equal(result[0], label_2d)
        assert np.array_equal(result[1], label_2d)

    def test_expand_label_to_time_2d_to_3d(self):
        """Test expanding 2D label to time series (T, Y, X)"""
        label_2d = np.ones((4, 5), dtype=np.uint32)
        t_size = 3

        result = _expand_label_to_time(label_2d, t_size)

        assert result.shape == (3, 4, 5)
        for t in range(t_size):
            assert np.array_equal(result[t], label_2d)

    def test_expand_label_to_time_2d_to_4d(self):
        """Test expanding 2D label to 4D with z-dimension (T, Z, Y, X)"""
        label_2d = np.zeros((4, 5), dtype=np.uint32)
        label_2d[1:3, 2:4] = 2
        t_size = 2
        z_size = 3

        result = _expand_label_to_time(label_2d, t_size, z_size)

        assert result.shape == (2, 3, 4, 5)
        # Check all time points and z-slices have the same label
        for t in range(t_size):
            for z in range(z_size):
                assert np.array_equal(result[t, z], label_2d)


class TestImageCropping:
    """Test image cropping with labels"""

    def test_crop_2d_simple(self):
        """Test simple 2D cropping"""
        image = np.ones((4, 5), dtype=np.uint8) * 255
        label = np.zeros((4, 5), dtype=np.uint32)
        label[1:3, 2:4] = 1

        result = _crop_image_with_label(image, label)

        assert result.shape == image.shape
        # Check that masked region is zero
        assert np.all(result[0, :] == 0)
        assert np.all(result[3, :] == 0)
        assert np.all(result[:, 0] == 0)
        assert np.all(result[:, 1] == 0)
        assert np.all(result[:, 4] == 0)
        # Check that labeled region is preserved
        assert np.all(result[1:3, 2:4] == 255)

    def test_crop_3d_simple(self):
        """Test 3D image cropping"""
        image = np.ones((3, 4, 5), dtype=np.uint16) * 1000
        label = np.zeros((3, 4, 5), dtype=np.uint32)
        label[:, 1:3, 2:4] = 1

        result = _crop_image_with_label(image, label)

        assert result.shape == image.shape
        assert np.all(result[:, 1:3, 2:4] == 1000)
        # Check background
        assert np.all(result[:, 0, :] == 0)

    def test_crop_with_multiple_labels(self):
        """Test cropping keeps only regions with label > 0"""
        image = np.arange(16, dtype=np.uint8).reshape((4, 4))
        label = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint32,
        )

        result = _crop_image_with_label(image, label)

        # Verify labeled region is preserved
        assert np.array_equal(result[1, 1:3], image[1, 1:3])
        assert np.array_equal(result[2, 1:3], image[2, 1:3])
        # Verify background is zero
        assert np.all(result[0, :] == 0)
        assert np.all(result[3, :] == 0)

    def test_crop_preserves_dtype(self):
        """Test that cropping preserves image dtype"""
        for dtype in [np.uint8, np.uint16, np.uint32, np.float32]:
            image = np.ones((3, 3), dtype=dtype)
            label = np.ones((3, 3), dtype=np.uint32)

            result = _crop_image_with_label(image, label)

            assert result.dtype == dtype


class TestAxisInference:
    """Test axis inference from image shape"""

    def test_infer_axes_2d(self):
        """Test 2D axis inference"""
        image = np.zeros((10, 20))
        axes = _infer_axes(image)
        assert axes == "YX"

    def test_infer_axes_3d_default(self):
        """Test 3D axis inference (defaults to ZYX)"""
        image = np.zeros((5, 10, 20))
        axes = _infer_axes(image)
        assert axes == "ZYX"

    def test_infer_axes_3d_with_metadata(self):
        """Test 3D axis inference with metadata indicating time"""
        image = np.zeros((5, 10, 20))
        metadata = {"frames": 5}
        axes = _infer_axes(image, metadata)
        assert axes == "TYX"

    def test_infer_axes_4d(self):
        """Test 4D axis inference"""
        image = np.zeros((3, 5, 10, 20))
        axes = _infer_axes(image)
        assert axes == "TZYX"


class TestLabelFilenameDetection:
    """Test label image filename detection"""

    def test_detect_label_file_simple(self):
        """Test detection of label file with common suffix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            intensity_path = Path(tmpdir) / "image_intensity.tif"
            label_path = Path(tmpdir) / "image_intensity_labels.tif"

            # Create dummy files
            intensity_path.touch()
            label_path.touch()

            result = _get_label_image_filename(str(intensity_path))
            assert result == str(label_path)

    def test_detect_label_file_filtered(self):
        """Test detection with filtered suffix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            intensity_path = Path(tmpdir) / "image.tif"
            label_path = Path(tmpdir) / "image_labels_filtered.tif"

            intensity_path.touch()
            label_path.touch()

            result = _get_label_image_filename(str(intensity_path))
            assert result == str(label_path)

    def test_detect_label_file_not_found(self):
        """Test when label file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            intensity_path = Path(tmpdir) / "image.tif"
            intensity_path.touch()

            result = _get_label_image_filename(str(intensity_path))
            assert result is None

    def test_detect_label_file_priority(self):
        """Test that correct suffix is found among multiple options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            intensity_path = Path(tmpdir) / "image.tif"
            label_path_1 = Path(tmpdir) / "image_seg.tif"
            label_path_2 = Path(tmpdir) / "image_labels.tif"

            intensity_path.touch()
            label_path_1.touch()
            label_path_2.touch()

            result = _get_label_image_filename(str(intensity_path))
            # Should find one of them (the exact one depends on iteration order)
            assert result in [str(label_path_1), str(label_path_2)]


@pytest.mark.skipif(not _HAS_TIFFFILE, reason="tifffile not available")
class TestLabelBasedCroppingIntegration:
    """Integration tests for the full cropping pipeline"""

    def test_crop_2d_image_with_2d_label(self):
        """Test cropping 2D image with 2D label"""
        # Create sample data
        image = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
        label = np.zeros((64, 64), dtype=np.uint32)
        label[10:30, 15:45] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.tif"
            label_path = Path(tmpdir) / "image_labels.tif"

            tifffile.imwrite(str(image_path), image)
            tifffile.imwrite(str(label_path), label)

            result = label_based_cropping(
                image, label_image_path=str(label_path)
            )

            assert result.shape == image.shape
            # Check that cropping worked
            assert np.all(result[9, :] == 0)
            assert np.all(result[30:, :] == 0)

    def test_crop_3d_image_with_2d_label(self):
        """Test cropping 3D image with 2D label (should expand to 3D)"""
        # Create sample data
        image = np.random.randint(0, 256, size=(5, 64, 64), dtype=np.uint8)
        label = np.zeros((64, 64), dtype=np.uint32)
        label[10:30, 15:45] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.tif"
            label_path = Path(tmpdir) / "image_labels.tif"

            # Create 3D image file
            tifffile.imwrite(str(image_path), image)
            tifffile.imwrite(str(label_path), label)

            result = label_based_cropping(
                image, label_image_path=str(label_path), expand_z=True
            )

            assert result.shape == image.shape
            # Check that masking applied to all z-slices
            for z in range(image.shape[0]):
                assert np.all(result[z, 9, :] == 0)
                assert np.all(result[z, 30:, :] == 0)

    def test_crop_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises appropriate error"""
        image = np.ones((64, 64), dtype=np.uint8)
        label = np.ones((32, 32), dtype=np.uint32)

        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / "label.tif"
            tifffile.imwrite(str(label_path), label)

            with pytest.raises(ValueError):
                label_based_cropping(
                    image, label_image_path=str(label_path)
                )

    def test_crop_missing_label_raises_error(self):
        """Test that missing label file raises error"""
        image = np.ones((64, 64), dtype=np.uint8)

        with pytest.raises(FileNotFoundError):
            label_based_cropping(image, label_image_path="/nonexistent/label.tif")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_crop_all_background(self):
        """Test cropping when label is all background"""
        image = np.ones((10, 10), dtype=np.uint8) * 100
        label = np.zeros((10, 10), dtype=np.uint32)

        result = _crop_image_with_label(image, label)

        # Everything should be zero
        assert np.all(result == 0)

    def test_crop_all_label(self):
        """Test cropping when label is all foreground"""
        image = np.ones((10, 10), dtype=np.uint8) * 100
        label = np.ones((10, 10), dtype=np.uint32)

        result = _crop_image_with_label(image, label)

        # Everything should be preserved
        assert np.array_equal(result, image)

    def test_crop_single_pixel_label(self):
        """Test cropping with single-pixel label"""
        image = np.arange(25, dtype=np.uint8).reshape((5, 5))
        label = np.zeros((5, 5), dtype=np.uint32)
        label[2, 2] = 1

        result = _crop_image_with_label(image, label)

        # Only center pixel should remain
        assert result[2, 2] == image[2, 2]
        assert np.sum(result > 0) == 1

    def test_expand_label_single_z_slice(self):
        """Test expanding to single z-slice"""
        label_2d = np.ones((5, 5), dtype=np.uint32)
        result = _expand_label_to_3d(label_2d, 1)

        assert result.shape == (1, 5, 5)
        assert np.array_equal(result[0], label_2d)

    def test_expand_label_large_z_size(self):
        """Test expanding to many z-slices"""
        label_2d = np.ones((3, 3), dtype=np.uint32)
        result = _expand_label_to_3d(label_2d, 100)

        assert result.shape == (100, 3, 3)
        for z in range(100):
            assert np.array_equal(result[z], label_2d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

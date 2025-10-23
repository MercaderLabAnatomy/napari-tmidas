# src/napari_tmidas/_tests/test_processing_basic.py
import numpy as np

from napari_tmidas.processing_functions.basic import (
    invert_binary_labels,
    labels_to_binary,
)


class TestBasicProcessing:
    def test_labels_to_binary(self):
        """Test converting labels to binary mask"""
        # Create test label image
        labels = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.uint32)

        # Process
        result = labels_to_binary(labels)

        # Check result - now expects 255 instead of 1
        expected = np.array(
            [[0, 255, 255], [255, 255, 0], [255, 0, 255]], dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8

    def test_labels_to_binary_all_zeros(self):
        """Test with all zero labels"""
        labels = np.zeros((3, 3), dtype=np.uint32)
        result = labels_to_binary(labels)
        np.testing.assert_array_equal(result, labels)

    def test_labels_to_binary_all_nonzero(self):
        """Test with all non-zero labels"""
        labels = np.ones((3, 3), dtype=np.uint32) * 5
        result = labels_to_binary(labels)
        np.testing.assert_array_equal(
            result, np.ones((3, 3), dtype=np.uint8) * 255
        )

    def test_labels_to_binary_empty_image(self):
        """Test with empty image"""
        labels = np.zeros((0, 0), dtype=np.uint32)
        result = labels_to_binary(labels)
        assert result.shape == (0, 0)
        assert result.dtype == np.uint8

    def test_labels_to_binary_3d_image(self):
        """Test with 3D image"""
        labels = np.array(
            [[[0, 1], [1, 2]], [[2, 0], [1, 1]]], dtype=np.uint32
        )
        result = labels_to_binary(labels)
        expected = np.array(
            [[[0, 255], [255, 255]], [[255, 0], [255, 255]]], dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_labels_to_binary_float_input(self):
        """Test with float input (should still work)"""
        labels = np.array([[0.0, 1.5, 2.7]], dtype=np.float32)
        result = labels_to_binary(labels)
        expected = np.array([[0, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_invert_binary_labels_basic(self):
        """Test basic inversion of binary mask"""
        # Create test binary image
        binary = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=np.uint32)

        # Process
        result = invert_binary_labels(binary)

        # Check result - zeros become 255, non-zeros become 0
        expected = np.array(
            [[255, 0, 0], [0, 255, 255], [0, 0, 255]], dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8

    def test_invert_binary_labels_all_zeros(self):
        """Test inversion with all zeros"""
        binary = np.zeros((3, 3), dtype=np.uint32)
        result = invert_binary_labels(binary)
        # All zeros should become 255
        np.testing.assert_array_equal(
            result, np.ones((3, 3), dtype=np.uint8) * 255
        )

    def test_invert_binary_labels_all_ones(self):
        """Test inversion with all ones"""
        binary = np.ones((3, 3), dtype=np.uint32)
        result = invert_binary_labels(binary)
        # All ones should become zeros
        np.testing.assert_array_equal(
            result, np.zeros((3, 3), dtype=np.uint32)
        )

    def test_invert_binary_labels_with_labels(self):
        """Test inversion with multi-label image"""
        # Create label image with different values
        labels = np.array([[0, 1, 2], [3, 0, 5], [7, 8, 0]], dtype=np.uint32)

        # Process
        result = invert_binary_labels(labels)

        # Check result - zeros become 255, all non-zero values become 0
        expected = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_invert_binary_labels_3d(self):
        """Test inversion with 3D image"""
        binary = np.array(
            [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint32
        )
        result = invert_binary_labels(binary)
        expected = np.array(
            [[[255, 0], [0, 255]], [[0, 255], [255, 0]]], dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_invert_binary_labels_empty(self):
        """Test with empty image"""
        binary = np.zeros((0, 0), dtype=np.uint32)
        result = invert_binary_labels(binary)
        assert result.shape == (0, 0)
        assert result.dtype == np.uint8

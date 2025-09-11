# src/napari_tmidas/_tests/test_processing_basic.py
import pytest
import numpy as np

# Skip entire module if dask is not available
dask = pytest.importorskip("dask")

from napari_tmidas.processing_functions.basic import labels_to_binary


class TestBasicProcessing:
    def test_labels_to_binary(self):
        """Test converting labels to binary mask"""
        # Create test label image
        labels = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.uint32)

        # Process
        result = labels_to_binary(labels)

        # Check result
        expected = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint32

    def test_labels_to_binary_all_zeros(self):
        """Test with all zero labels"""
        labels = np.zeros((3, 3), dtype=np.uint32)
        result = labels_to_binary(labels)
        np.testing.assert_array_equal(result, labels)

    def test_labels_to_binary_all_nonzero(self):
        """Test with all non-zero labels"""
        labels = np.ones((3, 3), dtype=np.uint32) * 5
        result = labels_to_binary(labels)
        np.testing.assert_array_equal(result, np.ones((3, 3), dtype=np.uint32))

    def test_labels_to_binary_empty_image(self):
        """Test with empty image"""
        labels = np.zeros((0, 0), dtype=np.uint32)
        result = labels_to_binary(labels)
        assert result.shape == (0, 0)
        assert result.dtype == np.uint32

    def test_labels_to_binary_3d_image(self):
        """Test with 3D image"""
        labels = np.array(
            [[[0, 1], [1, 2]], [[2, 0], [1, 1]]], dtype=np.uint32
        )
        result = labels_to_binary(labels)
        expected = np.array(
            [[[0, 1], [1, 1]], [[1, 0], [1, 1]]], dtype=np.uint32
        )
        np.testing.assert_array_equal(result, expected)

    def test_labels_to_binary_float_input(self):
        """Test with float input (should still work)"""
        labels = np.array([[0.0, 1.5, 2.7]], dtype=np.float32)
        result = labels_to_binary(labels)
        expected = np.array([[0, 1, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

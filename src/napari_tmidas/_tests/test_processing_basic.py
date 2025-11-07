# src/napari_tmidas/_tests/test_processing_basic.py
import numpy as np
import pytest

from napari_tmidas.processing_functions.basic import (
    intersect_label_images,
    invert_binary_labels,
    keep_slice_range_by_area,
    labels_to_binary,
    mirror_labels,
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
        np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=np.uint8))

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
        np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=np.uint8))

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

    def test_mirror_labels_double_size_default_axis(self):
        """Mirroring keeps the same shape and mirrors around largest area slice"""
        image = np.zeros((4, 2, 2), dtype=np.uint16)
        image[0, 0, 0] = 5  # slice 0 has 1 pixel
        image[1, :, :] = 3  # slice 1 has 4 pixels (largest area)

        result = mirror_labels(image)

        # Shape should remain the same
        assert result.shape == (4, 2, 2)
        # Mirror around slice 1 (largest area)
        # slice 0 gets from slice 2 (2*1 - 0 = 2), which is empty
        # slice 1 gets from slice 1 (2*1 - 1 = 1), which has value 3
        # slice 2 gets from slice 0 (2*1 - 2 = 0), which has value 5 at [0,0]
        # slice 3 gets from slice -1 (2*1 - 3 = -1, out of bounds)
        expected = np.zeros((4, 2, 2), dtype=np.uint16)
        expected[0] = 0  # mirrored from empty slice 2
        expected[1] = 3 + 5  # mirrored from slice 1 (value 3) with offset
        expected[2, 0, 0] = (
            5 + 5
        )  # mirrored from slice 0 (value 5 at [0,0]) with offset
        expected[3] = 0  # out of bounds
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == image.dtype

    def test_mirror_labels_other_axis(self):
        """Mirroring along a non-zero axis keeps shape and mirrors around largest area"""
        image = np.zeros((1, 4, 4), dtype=np.int32)
        image[0, 0, :] = 1  # slice 0: 4 pixels
        image[0, 1, :] = (
            2  # slice 1: 4 pixels (will be selected as max_area_idx)
        )
        image[0, 2, 0] = 3  # slice 2: 1 pixel
        image[0, 3, 0] = 4  # slice 3: 1 pixel

        result = mirror_labels(image, axis=1)

        # Shape should remain the same
        assert result.shape == (1, 4, 4)
        # Mirror around slice 0 (first slice with max area)
        # slice 0 gets from slice 0 (2*0 - 0 = 0), which has value 1
        # slice 1 gets from slice -1 (2*0 - 1 = -1, out of bounds)
        # slice 2 gets from slice -2 (2*0 - 2 = -2, out of bounds)
        # slice 3 gets from slice -3 (2*0 - 3 = -3, out of bounds)
        expected = np.zeros((1, 4, 4), dtype=np.int32)
        expected[0, 0, :] = (
            1 + 4
        )  # mirrored from slice 0 (value 1) with offset
        expected[0, 1:, :] = 0  # out of bounds
        np.testing.assert_array_equal(result, expected)

    def test_mirror_labels_prefers_larger_end(self):
        """Mirrors around the slice with the largest area"""
        image = np.zeros((4, 3, 3), dtype=np.uint8)
        image[0, :2, :2] = 1  # slice 0: 4 pixels (largest area)
        image[3, 0, 0] = 1  # slice 3: 1 pixel

        result = mirror_labels(image)

        # Shape should remain the same
        assert result.shape == (4, 3, 3)
        # Mirror around slice 0 (largest area)
        # slice 0 mirrors slice 0 (2*0 - 0 = 0)
        # slice 1 mirrors slice -1 (2*0 - 1 = -1, out of bounds)
        # slice 2 mirrors slice -2 (2*0 - 2 = -2, out of bounds)
        # slice 3 mirrors slice -3 (2*0 - 3 = -3, out of bounds)
        expected = np.zeros((4, 3, 3), dtype=np.uint8)
        expected[0, :2, :2] = 1 + 1  # mirrored from slice 0 itself
        expected[1:] = 0  # out of bounds
        np.testing.assert_array_equal(result, expected)

    def test_mirror_labels_uniform(self):
        """Mirroring uniform labels creates offset mirrored labels"""
        image = np.ones((3, 3, 3), dtype=np.uint8)

        result = mirror_labels(image)

        # Shape should remain the same
        assert result.shape == (3, 3, 3)
        # All slices have equal area (9 pixels), so slice 0 is chosen
        # Mirror around slice 0 (first slice with max area)
        # slice 0 mirrors slice 0 (2*0 - 0 = 0)
        # slice 1 mirrors slice -1 (2*0 - 1 = -1, out of bounds)
        # slice 2 mirrors slice -2 (2*0 - 2 = -2, out of bounds)
        expected = np.zeros((3, 3, 3), dtype=np.uint8)
        expected[0] = 2  # mirrored from slice 0 (1 + 1)
        expected[1:] = 0  # out of bounds
        np.testing.assert_array_equal(result, expected)

    def test_mirror_labels_invalid_axis(self):
        """Invalid axis should raise an error"""
        image = np.zeros((3, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            mirror_labels(image, axis=2)

    def test_keep_slice_range_by_area_basic(self):
        """Keep label content between minimum and maximum area, preserving shape"""
        volume = np.zeros((5, 4, 4), dtype=np.int32)
        volume[0, 0, 0] = 1  # area 1 (min)
        volume[1, :2, :2] = 1  # area 4
        volume[2, :3, :3] = 1  # area 9 (max)
        volume[3, :1, :3] = 1  # area 3
        volume[4, :2, :1] = 1  # area 2

        result = keep_slice_range_by_area(volume)

        # Shape should be preserved
        assert result.shape == (5, 4, 4)
        # Content between min (slice 0) and max (slice 2) should be kept
        np.testing.assert_array_equal(result[0:3], volume[0:3])
        # Content after max should be zeroed
        np.testing.assert_array_equal(result[3:], np.zeros((2, 4, 4)))

    def test_keep_slice_range_by_area_with_axis(self):
        """Axis parameter allows zeroing content along any dimension while preserving shape"""
        # Create volume with different areas along axis 1
        volume = np.zeros((4, 5, 3), dtype=np.uint16)
        volume[:2, 0, :2] = 1  # slice 0: area = 2*2 = 4
        volume[:, 1, :] = 1  # slice 1: area = 4*3 = 12 (max)
        volume[:3, 2, :] = 1  # slice 2: area = 3*3 = 9
        volume[:2, 3, :2] = 1  # slice 3: area = 2*2 = 4
        volume[0, 4, 0] = 1  # slice 4: area = 1 (min)

        result = keep_slice_range_by_area(volume, axis=1)

        # Shape should be preserved
        assert result.shape == volume.shape
        # Min area is at slice 4, max area is at slice 1, so range is 1-4 (inclusive)
        # Slice 0 should be zeroed (before the range)
        np.testing.assert_array_equal(
            result[:, 0, :], np.zeros((4, 3), dtype=np.uint16)
        )
        # Slices 1-4 should be kept
        np.testing.assert_array_equal(result[:, 1:5, :], volume[:, 1:5, :])

    def test_keep_slice_range_by_area_uniform(self):
        """Uniform area returns the original volume"""
        volume = np.ones((3, 4, 4), dtype=np.uint8)

        result = keep_slice_range_by_area(volume)

        np.testing.assert_array_equal(result, volume)

    def test_keep_slice_range_by_area_shape_preserved(self):
        """Verify that output shape matches input shape (critical for image-label alignment)"""
        # Simulate a label volume with 100 z-slices where labels exist in slices 20-80
        volume = np.zeros((100, 50, 50), dtype=np.uint32)
        volume[20, :10, :10] = 1  # Sparse content at slice 20 (min area)
        for i in range(21, 80):
            volume[i, :30, :30] = i  # Denser content in middle slices
        volume[79, :, :] = 100  # Maximum content at slice 79 (max area)
        # Slices 0-19 and 80-99 should be empty and get zeroed

        result = keep_slice_range_by_area(volume, axis=0)

        # Critical: shape must be preserved to maintain alignment with image data
        assert result.shape == (
            100,
            50,
            50,
        ), "Output shape must match input shape"

        # Slices before min (0-19) should be zeroed
        assert np.all(
            result[:20] == 0
        ), "Slices before min-area slice should be zeroed"

        # Slices between min and max (20-79) should be preserved
        np.testing.assert_array_equal(
            result[20:80],
            volume[20:80],
            err_msg="Label content in range should be preserved",
        )

        # Slices after max (80-99) should be zeroed
        assert np.all(
            result[80:] == 0
        ), "Slices after max-area slice should be zeroed"

    def test_keep_slice_range_by_area_invalid_dims(self):
        """At least 3 dimensions are required"""
        image = np.ones((4, 4), dtype=np.uint8)

        with pytest.raises(ValueError):
            keep_slice_range_by_area(image)

    def test_intersect_label_images_basic(self, tmp_path):
        """Primary file intersects with its paired secondary"""
        label_a = np.array([[0, 5], [2, 0]], dtype=np.uint8)
        label_b = np.array([[1, 5], [0, 0]], dtype=np.uint8)

        primary_path = tmp_path / "sample_a.npy"
        secondary_path = tmp_path / "sample_b.npy"
        np.save(primary_path, label_a)
        np.save(secondary_path, label_b)

        def call_primary() -> np.ndarray:
            filepath = str(primary_path)
            assert filepath
            return intersect_label_images(
                label_a,
                primary_suffix="_a.npy",
                secondary_suffix="_b.npy",
            )

        result = call_primary()
        expected = np.array([[0, 5], [0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

        def call_secondary() -> np.ndarray:
            filepath = str(secondary_path)
            assert filepath
            return intersect_label_images(
                label_b,
                primary_suffix="_a.npy",
                secondary_suffix="_b.npy",
            )

        with pytest.warns(UserWarning, match="Skipping secondary label image"):
            secondary_result = call_secondary()
        assert secondary_result is None

    def test_intersect_label_images_retains_primary_labels(self, tmp_path):
        label_a = np.zeros((4, 4), dtype=np.uint8)
        label_b = np.zeros((4, 4), dtype=np.uint8)
        label_a[1:3, 1:3] = 1
        label_b[1:2, 1:3] = 2
        label_b[2:3, 1:3] = 3

        primary_path = tmp_path / "detail_a.npy"
        secondary_path = tmp_path / "detail_b.npy"
        np.save(primary_path, label_a)
        np.save(secondary_path, label_b)

        def call_primary():
            filepath = str(primary_path)
            assert filepath
            return intersect_label_images(
                label_a,
                primary_suffix="_a.npy",
                secondary_suffix="_b.npy",
            )

        result = call_primary()
        expected = np.zeros_like(label_a)
        expected[1:3, 1:3] = 1
        np.testing.assert_array_equal(result, expected)

    def test_intersect_label_images_preserve_primary_detail(self, tmp_path):
        label_a = np.zeros((4, 4), dtype=np.uint8)
        label_b = np.zeros((4, 4), dtype=np.uint8)
        label_a[1:2, 1:3] = 4
        label_a[2:3, 1:3] = 5
        label_b[1:3, 1:3] = 7

        primary_path = tmp_path / "detail_a.npy"
        secondary_path = tmp_path / "detail_b.npy"
        np.save(primary_path, label_a)
        np.save(secondary_path, label_b)

        def call_primary():
            filepath = str(primary_path)
            assert filepath
            return intersect_label_images(
                label_a,
                primary_suffix="_a.npy",
                secondary_suffix="_b.npy",
            )

        result = call_primary()
        expected = np.zeros_like(label_a)
        expected[1:2, 1:3] = 4
        expected[2:3, 1:3] = 5
        np.testing.assert_array_equal(result, expected)

    def test_intersect_label_images_missing_pair(self, tmp_path):
        label_a = np.ones((2, 2), dtype=np.uint16)
        primary_path = tmp_path / "orphan_a.npy"
        np.save(primary_path, label_a)

        def call_primary():
            filepath = str(primary_path)
            assert filepath
            return intersect_label_images(
                label_a,
                primary_suffix="_a.npy",
                secondary_suffix="_b.npy",
            )

        with pytest.raises(FileNotFoundError):
            call_primary()

    def test_intersect_label_images_shape_mismatch(self, tmp_path):
        label_a = np.ones((2, 2), dtype=np.uint16)
        label_b = np.ones((3, 3), dtype=np.uint16)

        primary_path = tmp_path / "sample_a.npy"
        secondary_path = tmp_path / "sample_b.npy"
        np.save(primary_path, label_a)
        np.save(secondary_path, label_b)

        def call_primary():
            filepath = str(primary_path)
            assert filepath
            return intersect_label_images(
                label_a,
                primary_suffix="_a.npy",
                secondary_suffix="_b.npy",
            )

        result = call_primary()
        expected = np.ones_like(label_a)
        np.testing.assert_array_equal(result, expected)

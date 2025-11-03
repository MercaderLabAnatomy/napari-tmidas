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
        """Mirroring doubles the size along the default axis"""
        image = np.zeros((2, 2, 2), dtype=np.uint16)
        image[0, 0, 0] = 5

        result = mirror_labels(image)

        assert result.shape == (4, 2, 2)
        mirrored_expected = np.where(
            np.flip(image, axis=0) > 0,
            np.flip(image, axis=0) + image.max(),
            0,
        )
        np.testing.assert_array_equal(result[:2], mirrored_expected)
        np.testing.assert_array_equal(result[2:], image)
        assert result.dtype == image.dtype

    def test_mirror_labels_other_axis(self):
        """Mirroring along a non-zero axis doubles that axis length"""
        image = np.arange(12, dtype=np.int32).reshape(1, 3, 4)

        result = mirror_labels(image, axis=1)

        assert result.shape == (1, 6, 4)
        mirrored_expected = np.where(
            np.flip(image, axis=1) > 0,
            np.flip(image, axis=1) + image.max(),
            0,
        )
        np.testing.assert_array_equal(result[:, :3], image)
        np.testing.assert_array_equal(result[:, 3:], mirrored_expected)

    def test_mirror_labels_prefers_larger_end(self):
        """The side with more labels sits near the center after mirroring"""
        image = np.zeros((4, 3, 3), dtype=np.uint8)
        image[0, :2, :2] = 1  # larger area at the beginning
        image[3, 0, 0] = 1

        result = mirror_labels(image)

        mirrored_expected = np.where(
            np.flip(image, axis=0) > 0,
            np.flip(image, axis=0) + image.max(),
            0,
        )
        np.testing.assert_array_equal(result[:4], mirrored_expected)
        np.testing.assert_array_equal(result[4:], image)

    def test_mirror_labels_uniform(self):
        """Mirroring uniform labels offsets mirrored half"""
        image = np.ones((2, 3, 3), dtype=np.uint8)

        result = mirror_labels(image)

        assert result.shape == (4, 3, 3)
        np.testing.assert_array_equal(result[:2], image)
        np.testing.assert_array_equal(
            result[2:], np.full((2, 3, 3), 2, dtype=np.uint8)
        )

    def test_mirror_labels_invalid_axis(self):
        """Invalid axis should raise an error"""
        image = np.zeros((3, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            mirror_labels(image, axis=2)

    def test_keep_slice_range_by_area_basic(self):
        """Keep slices between minimum and maximum area along default axis"""
        volume = np.zeros((5, 4, 4), dtype=np.int32)
        volume[0, 0, 0] = 1  # area 1
        volume[1, :2, :2] = 1  # area 4
        volume[2, :3, :3] = 1  # area 9 (max)
        volume[3, :1, :3] = 1  # area 3
        volume[4, :2, :1] = 1  # area 2

        result = keep_slice_range_by_area(volume)

        assert result.shape == (3, 4, 4)
        np.testing.assert_array_equal(result, volume[0:3])

    def test_keep_slice_range_by_area_with_axis(self):
        """Axis parameter allows trimming along any dimension"""
        base = np.zeros((5, 4, 4), dtype=np.uint16)
        base[0, 0, 0] = 1
        base[1, :2, :2] = 1
        base[2, :3, :3] = 1
        reordered = base.transpose(1, 0, 2)

        result = keep_slice_range_by_area(reordered, axis=1)

        expected = reordered[:, 0:3, :]
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)

    def test_keep_slice_range_by_area_uniform(self):
        """Uniform area returns the original volume"""
        volume = np.ones((3, 4, 4), dtype=np.uint8)

        result = keep_slice_range_by_area(volume)

        np.testing.assert_array_equal(result, volume)

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

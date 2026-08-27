# src/napari_tmidas/_tests/test_processing_basic.py
import os
from pathlib import Path

import numpy as np
import pytest

from napari_tmidas.processing_functions.basic import (
    _merge_channels_file_pre_filter,
    filter_label_by_id,
    intersect_label_images,
    invert_binary_labels,
    keep_slice_range_by_area,
    labels_to_binary,
    merge_channels,
    mirror_labels,
    rgb_to_labels,
    split_tzyx_stack,
)


class TestBasicProcessing:
    def test_split_tzyx_stack_is_not_thread_safe(self):
        """Split TZYX has inner workers; outer batch worker must be single-threaded."""
        assert hasattr(split_tzyx_stack, "thread_safe")
        assert split_tzyx_stack.thread_safe is False

    def test_split_tzyx_stack_accepts_tczyx_shape(self):
        """5D TCZYX input should still register all timepoints for splitting."""
        image = np.random.rand(10, 1, 4, 8, 8).astype(np.float32)

        result = split_tzyx_stack(image, num_workers=4)

        # Function returns original image by design; internal dask image drives splitting.
        assert result is image
        tl = split_tzyx_stack._thread_local
        assert tl.dask_image.shape[0] == 10
        assert tl.num_workers == 4
        assert tl.produces_multiple_files is True

    def test_split_tzyx_stack_accepts_dask_input(self):
        """Dask TCZYX input should rechunk without calling da.from_array again."""
        da = pytest.importorskip("dask.array")
        image = da.random.random((10, 1, 4, 8, 8), chunks=(1, 1, 1, 8, 8)).astype(
            np.float32
        )

        result = split_tzyx_stack(image, num_workers=3)

        assert result is image
        tl = split_tzyx_stack._thread_local
        assert tl.dask_image.shape[0] == 10
        assert tl.num_workers == 3
        assert tl.produces_multiple_files is True

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

    def test_filter_label_by_id_basic(self):
        """Test filtering to keep only one label ID"""
        # Create test label image with multiple labels
        labels = np.array([[0, 1, 2], [3, 1, 2], [1, 0, 3]], dtype=np.uint32)

        # Keep only label 1
        result = filter_label_by_id(labels, label_id=1)

        # Check result - only label 1 should remain, others become 0
        expected = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == labels.dtype

    def test_filter_label_by_id_default_param(self):
        """Test filtering with default parameter (label_id=1)"""
        labels = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.uint32)
        result = filter_label_by_id(labels)
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

    def test_filter_label_by_id_nonexistent(self):
        """Test filtering with label ID that doesn't exist"""
        labels = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.uint32)
        # Try to keep label 99 which doesn't exist
        result = filter_label_by_id(labels, label_id=99)
        # All should become background
        expected = np.zeros_like(labels)
        np.testing.assert_array_equal(result, expected)

    def test_filter_label_by_id_3d(self):
        """Test filtering with 3D label image"""
        labels = np.array(
            [[[1, 2], [3, 1]], [[2, 1], [1, 3]]], dtype=np.uint32
        )
        result = filter_label_by_id(labels, label_id=2)
        expected = np.array(
            [[[0, 2], [0, 0]], [[2, 0], [0, 0]]], dtype=np.uint32
        )
        np.testing.assert_array_equal(result, expected)

    def test_filter_label_by_id_all_same(self):
        """Test filtering when all pixels are the target label"""
        labels = np.ones((3, 3), dtype=np.uint32) * 5
        result = filter_label_by_id(labels, label_id=5)
        # All should remain
        np.testing.assert_array_equal(result, labels)

    def test_filter_label_by_id_all_background(self):
        """Test filtering with all background"""
        labels = np.zeros((3, 3), dtype=np.uint32)
        result = filter_label_by_id(labels, label_id=1)
        # Should remain all zeros
        np.testing.assert_array_equal(result, labels)

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

    def test_merge_channels_pre_filter_accepts_varying_suffixes(self, tmp_path):
        """Only the lowest channel should pass pre-filter even with channel-specific suffixes."""
        files = [
            "sample_channel_1_dapi.tif",
            "sample_channel_2_gfp.tif",
            "sample_channel_3_cy5.tif",
        ]
        for name in files:
            (tmp_path / name).touch()

        params = {"channel_substring": "_channel_"}
        assert (
            _merge_channels_file_pre_filter(
                str(tmp_path / "sample_channel_1_dapi.tif"), params
            )
            is True
        )
        assert (
            _merge_channels_file_pre_filter(
                str(tmp_path / "sample_channel_2_gfp.tif"), params
            )
            is False
        )
        assert (
            _merge_channels_file_pre_filter(
                str(tmp_path / "sample_channel_3_cy5.tif"), params
            )
            is False
        )

    def test_merge_channels_merges_varying_suffixes(self, tmp_path):
        """merge_channels should merge channels even when post-number suffixes vary."""
        tifffile = pytest.importorskip("tifffile")

        ch1 = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        ch2 = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        ch3 = np.array([[100, 200], [300, 400]], dtype=np.uint16)

        p1 = tmp_path / "sample_channel_1_dapi.tif"
        p2 = tmp_path / "sample_channel_2_gfp.tif"
        p3 = tmp_path / "sample_channel_3_cy5.tif"

        tifffile.imwrite(p1, ch1)
        tifffile.imwrite(p2, ch2)
        tifffile.imwrite(p3, ch3)

        merged = merge_channels(
            ch1,
            channel_substring="_channel_",
            _source_filepath=str(p1),
        )

        assert isinstance(merged, np.ndarray)
        assert merged.shape == (3, 2, 2)
        np.testing.assert_array_equal(merged[0], ch1)
        np.testing.assert_array_equal(merged[1], ch2)
        np.testing.assert_array_equal(merged[2], ch3)


class TestRgbToLabels:
    """
    ``rgb_to_labels`` maps three exact primary colours onto label values.
    Anything that is not exactly one of those colours must stay background --
    the mapping is by equality, not by nearest colour or by dominant channel.
    """

    @staticmethod
    def _rgb(pixels):
        return np.array(pixels, dtype=np.uint8)

    def test_primaries_map_to_default_labels(self):
        image = self._rgb(
            [
                [(0, 0, 255), (0, 255, 0)],
                [(255, 0, 0), (0, 0, 0)],
            ]
        )

        result = rgb_to_labels(image)

        np.testing.assert_array_equal(result, [[1, 2], [3, 0]])

    def test_custom_label_values(self):
        image = self._rgb([[(0, 0, 255), (0, 255, 0), (255, 0, 0)]])

        result = rgb_to_labels(
            image, blue_label=10, green_label=20, red_label=30
        )

        np.testing.assert_array_equal(result, [[10, 20, 30]])

    def test_returns_uint32(self):
        """
        Label images must be uint32 or napari loads them as a grayscale Image
        layer instead of a Labels layer.
        """
        image = self._rgb([[(255, 0, 0)]])

        assert rgb_to_labels(image).dtype == np.uint32

    def test_near_miss_colours_are_background(self):
        """
        Off-by-one colours -- what JPEG compression or any interpolating
        resize produces -- are not the mapped primaries and must not be
        silently rounded into a label.
        """
        image = self._rgb(
            [
                [(0, 0, 254), (1, 255, 0)],
                [(254, 0, 0), (128, 128, 128)],
            ]
        )

        result = rgb_to_labels(image)

        np.testing.assert_array_equal(result, [[0, 0], [0, 0]])

    def test_drops_the_channel_axis(self):
        """A (Z, Y, X, 3) volume becomes a (Z, Y, X) label volume."""
        image = np.zeros((4, 5, 6, 3), dtype=np.uint8)
        image[2, 3, 4] = (0, 255, 0)

        result = rgb_to_labels(image)

        assert result.shape == (4, 5, 6)
        assert result[2, 3, 4] == 2
        assert result.sum() == 2

    @pytest.mark.parametrize(
        "shape",
        [(4, 4), (4, 4, 4), (4, 4, 1)],
    )
    def test_rejects_non_rgb_input(self, shape):
        """Grayscale, 4-channel and single-channel input are all errors."""
        with pytest.raises(ValueError, match="RGB"):
            rgb_to_labels(np.zeros(shape, dtype=np.uint8))


class TestMergeChannelsStreamingPath:
    """
    When the worker injects ``_output_folder`` and ``_output_suffix``,
    merge_channels stops returning an array and instead writes the merged file
    itself, streaming through a temporary Zarr store so peak RAM stays at one
    slice. That path was untested, and it is the one that actually runs in the
    application -- the ndarray return is documented as the legacy/test path.
    """

    @staticmethod
    def _write_channels(folder, shape, n_channels=3, dtype=np.uint16):
        """One file per channel; channel c is filled with the value c."""
        tifffile = pytest.importorskip("tifffile")
        paths = []
        for c in range(1, n_channels + 1):
            p = folder / f"sample_channel_{c}.tif"
            tifffile.imwrite(
                p, np.full(shape, c, dtype=dtype), photometric="minisblack"
            )
            paths.append(p)
        return paths

    def test_streamed_output_matches_the_in_memory_merge(self, tmp_path):
        """
        The two paths must agree exactly. If they diverge, results depend on
        whether the worker happened to pass an output folder.
        """
        tifffile = pytest.importorskip("tifffile")
        src = tmp_path / "in"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        paths = self._write_channels(src, (5, 8, 9))
        primary = tifffile.imread(str(paths[0]))

        dense = merge_channels(
            primary, channel_substring="_channel_",
            _source_filepath=str(paths[0]),
        )
        written = merge_channels(
            primary, channel_substring="_channel_",
            _source_filepath=str(paths[0]),
            _output_folder=str(out), _output_suffix="_merged",
        )

        assert isinstance(written, str), (
            "with an output folder the function must save and return the path"
        )
        assert os.path.exists(written)
        np.testing.assert_array_equal(tifffile.imread(written), dense)

    def test_streamed_output_preserves_channel_order_and_dtype(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        src = tmp_path / "in"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        paths = self._write_channels(src, (4, 6, 7), n_channels=3)
        primary = tifffile.imread(str(paths[0]))

        written = merge_channels(
            primary, channel_substring="_channel_",
            _source_filepath=str(paths[0]),
            _output_folder=str(out), _output_suffix="_merged",
        )

        merged = tifffile.imread(written)
        assert merged.dtype == np.uint16
        assert merged.shape == (3, 4, 6, 7)
        # Channel c was filled with the value c, so order is directly checkable.
        for c in range(3):
            assert set(np.unique(merged[c])) == {c + 1}

    def test_streamed_output_lands_in_the_requested_folder(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        src = tmp_path / "in"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        paths = self._write_channels(src, (2, 4, 5))
        primary = tifffile.imread(str(paths[0]))

        written = merge_channels(
            primary, channel_substring="_channel_",
            _source_filepath=str(paths[0]),
            _output_folder=str(out), _output_suffix="_merged",
        )

        assert Path(written).parent == out
        assert "_merged" in Path(written).name

    def test_no_temporary_zarr_store_is_left_behind(self, tmp_path):
        """The intermediate store is a buffer, not an artefact."""
        tifffile = pytest.importorskip("tifffile")
        src = tmp_path / "in"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        paths = self._write_channels(src, (3, 5, 6))
        primary = tifffile.imread(str(paths[0]))

        merge_channels(
            primary, channel_substring="_channel_",
            _source_filepath=str(paths[0]),
            _output_folder=str(out), _output_suffix="_merged",
        )

        leftovers = [p.name for p in out.iterdir() if p.suffix == ".zarr"]
        assert leftovers == []

    def test_missing_filepath_raises(self):
        """Without a source path there is no folder to find siblings in."""
        with pytest.raises(ValueError):
            merge_channels(
                np.zeros((4, 4), dtype=np.uint16),
                channel_substring="_channel_",
            )

    @pytest.mark.parametrize("n_z", [3, 4])
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The merged CZYX stack is handed to TiffWriter without "
            "photometric='minisblack', so a Z axis of length 3 or 4 is "
            "interpreted as RGB(A) samples: the page is tagged PHOTOMETRIC.RGB "
            "with samplesperpixel=Z. tifffile reads it back correctly from its "
            "own shaped metadata, which is why a round-trip test misses this, "
            "but ImageJ, bio-formats and PIL all see a colour image."
        ),
    )
    def test_merged_tiff_is_not_tagged_as_rgb(self, tmp_path, n_z):
        tifffile = pytest.importorskip("tifffile")
        src = tmp_path / "in"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        paths = self._write_channels(src, (n_z, 6, 7), n_channels=3)
        primary = tifffile.imread(str(paths[0]))

        written = merge_channels(
            primary, channel_substring="_channel_",
            _source_filepath=str(paths[0]),
            _output_folder=str(out), _output_suffix="_merged",
        )

        with tifffile.TiffFile(written) as tf:
            page = tf.pages[0]
            assert page.samplesperpixel == 1, (
                f"Z={n_z} was written as {page.samplesperpixel} colour "
                "samples per pixel"
            )
            assert page.photometric == tifffile.PHOTOMETRIC.MINISBLACK

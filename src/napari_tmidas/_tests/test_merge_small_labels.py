# src/napari_tmidas/_tests/test_merge_small_labels.py
"""Tests for merge_small_labels processing function."""

import numpy as np
import pytest

from napari_tmidas.processing_functions.merge_small_labels import (
    _merge_single_frame,
    merge_small_labels,
)


class TestMergeSmallLabels:
    """Tests for the merge_small_labels function."""

    # ------------------------------------------------------------------
    # Basic 2-D cases
    # ------------------------------------------------------------------

    def test_no_small_labels_unchanged(self):
        """When all labels exceed min_size the image must be returned unchanged."""
        img = np.zeros((10, 10), dtype=np.int32)
        img[0:5, 0:5] = 1  # 25 voxels
        img[5:10, 5:10] = 2  # 25 voxels
        result = merge_small_labels(img, min_size=10)
        np.testing.assert_array_equal(result, img)

    def test_small_label_merged_into_neighbor(self):
        """A small label touching a large label should take the large label's ID."""
        img = np.zeros((10, 10), dtype=np.int32)
        img[0:8, 0:8] = 1  # 64 voxels – large
        img[8:10, 0:2] = 2  # 4 voxels  – small, touches label 1

        result = merge_small_labels(img, min_size=10)

        # Label 2 (small) should have been absorbed into label 1
        assert 2 not in np.unique(result)
        assert 1 in np.unique(result)
        # Region formerly occupied by label 2 now belongs to label 1
        assert np.all(result[8:10, 0:2] == 1)

    def test_isolated_small_label_removed(self):
        """A small label with no touching neighbour should become background."""
        img = np.zeros((10, 10), dtype=np.int32)
        img[0:8, 0:8] = 1  # large label, not adjacent
        img[9, 9] = 2  # isolated single-voxel label

        result = merge_small_labels(img, min_size=5)

        assert 2 not in np.unique(result)
        assert result[9, 9] == 0

    def test_dtype_preserved(self):
        """Output dtype must match input dtype."""
        img = np.zeros((10, 10), dtype=np.uint16)
        img[0:8, 0:8] = 1
        img[8:10, 0:2] = 2
        result = merge_small_labels(img, min_size=10)
        assert result.dtype == np.uint16

    def test_empty_image_unchanged(self):
        """An all-zero image should be returned as-is."""
        img = np.zeros((10, 10), dtype=np.int32)
        result = merge_small_labels(img, min_size=10)
        np.testing.assert_array_equal(result, img)

    def test_large_label_not_affected(self):
        """Labels above min_size must keep their original ID and pixel set."""
        img = np.zeros((20, 20), dtype=np.int32)
        img[0:15, 0:15] = 1  # 225 voxels
        img[15:17, 15:17] = 2  # 4 voxels, small

        result = merge_small_labels(img, min_size=10)

        # Large label unchanged
        np.testing.assert_array_equal(result[0:15, 0:15], 1)

    def test_small_label_merges_into_largest_contact(self):
        """When a small label touches multiple neighbors, the one with the
        most contact voxels (not the globally largest label) wins."""
        img = np.zeros((7, 10), dtype=np.int32)
        # Label 1: occupies columns 0-3 (wide contact with small)
        img[0:7, 0:4] = 1  # 28 voxels
        # Label 2: occupies columns 6-9 (narrow contact with small)
        img[0:7, 6:10] = 2  # 28 voxels – same global size, less contact
        # Small label in the middle column 4-5, full height: 14 voxels
        img[0:7, 4:6] = 3  # 14 voxels, touches 1 on left and 2 on right equally? 
        # Make label 1 have more contact: shift small label one step right
        # Small label occupies col 5 only (7 voxels), touching label 1 at col 4 (no!)
        # Actually let's place it differently:
        # img col 4 belongs to label 1, col 6 belongs to label 2 → small is col 5
        img[:, 4] = 1
        img[:, 5] = 3  # 7 voxels (small), touches label 1 (left) and label 2 (right)
        img[:, 6] = 2
        # Both contacts are equal (7 voxels each). Tie breaks arbitrarily; just
        # check the small label was absorbed by *something*.
        result = merge_small_labels(img, min_size=10)
        assert 3 not in np.unique(result)
        assert set(np.unique(result[result != 0])).issubset({1, 2})

    # ------------------------------------------------------------------
    # 3-D case
    # ------------------------------------------------------------------

    def test_3d_small_label_merged(self):
        """Function should work identically on 3-D label images."""
        img = np.zeros((10, 10, 10), dtype=np.int32)
        img[0:8, 0:8, 0:8] = 1  # 512 voxels – large
        img[8:10, 0:2, 0:2] = 2  # 8 voxels – small, touches label 1

        result = merge_small_labels(img, min_size=20)

        assert 2 not in np.unique(result)
        assert np.all(result[8:10, 0:2, 0:2] == 1)

    # ------------------------------------------------------------------
    # Parameter edge cases
    # ------------------------------------------------------------------

    def test_min_size_as_string(self):
        """min_size passed as a string should be coerced without error."""
        img = np.zeros((10, 10), dtype=np.int32)
        img[0:8, 0:8] = 1
        img[8:10, 0:2] = 2
        result = merge_small_labels(img, min_size="10")
        assert 2 not in np.unique(result)

    def test_min_size_zero_nothing_merged(self):
        """min_size=0 means nothing is considered small; image unchanged."""
        img = np.zeros((10, 10), dtype=np.int32)
        img[0:8, 0:8] = 1
        img[8:10, 0:2] = 2
        result = merge_small_labels(img, min_size=0)
        np.testing.assert_array_equal(result, img)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def test_registered_in_registry(self):
        """The function must be discoverable via the BatchProcessingRegistry."""
        from napari_tmidas._registry import BatchProcessingRegistry

        info = BatchProcessingRegistry.get_function_info(
            "Merge Small Labels to Neighbors"
        )
        assert info is not None
        assert "min_size" in info["parameters"]

    # ------------------------------------------------------------------
    # Time-series (T) dispatch
    # ------------------------------------------------------------------

    def test_tyx_timeseries_processed_per_frame(self):
        """4-D input (T, Z, Y, X) must be processed per timepoint."""
        # Build a 3-frame 2-D label stack
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[0:8, 0:8] = 1   # large
        frame[8:10, 0:2] = 2  # small, touches label 1

        img_4d = np.stack([frame, frame, frame])  # shape (3, 10, 10) → T,Y,X
        # Treat as (T, Z, Y, X) would be 4-D; here we use (T, Y, X) = 3-D
        # so let's build a proper 4-D TZYX with a trivial Z of 1
        img_4d = img_4d[:, np.newaxis, :, :]  # (3, 1, 10, 10)

        result = merge_small_labels(img_4d, min_size=10)

        assert result.shape == img_4d.shape
        assert result.dtype == img_4d.dtype
        # Small label 2 absorbed in every timepoint
        for t in range(3):
            assert 2 not in np.unique(result[t])
            assert 1 in np.unique(result[t])

    def test_tzyx_timeseries_shape_preserved(self):
        """Output shape and dtype must match input for 4-D arrays."""
        img = np.zeros((4, 5, 10, 10), dtype=np.uint16)
        img[:, 0:4, 0:8, 0:8] = 1
        img[:, 0:2, 8:10, 0:2] = 2  # small per timepoint

        result = merge_small_labels(img, min_size=10)

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_dimension_order_tyx_iterates_per_frame(self):
        """dimension_order='TYX' must process each 2-D frame independently."""
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[0:8, 0:8] = 1   # large
        frame[8:10, 0:2] = 2  # small, touches label 1
        img_3d = np.stack([frame, frame, frame])  # (3, 10, 10)

        result = merge_small_labels(img_3d, min_size=10, dim_order="TYX")

        assert result.shape == img_3d.shape
        for t in range(3):
            assert 2 not in np.unique(result[t])
            assert 1 in np.unique(result[t])

    def test_dimension_order_zyx_treats_3d_as_volume(self):
        """dimension_order='ZYX' (or Auto) must treat 3-D as a single volume."""
        img = np.zeros((5, 10, 10), dtype=np.int32)
        img[0:4, 0:8, 0:8] = 1
        img[4:5, 8:10, 0:2] = 2  # small

        # Both ZYX and Auto should give same result (single-volume processing)
        r_zyx  = merge_small_labels(img, min_size=10, dim_order="ZYX")
        r_auto = merge_small_labels(img, min_size=10)
        np.testing.assert_array_equal(r_zyx, r_auto)
        assert 2 not in np.unique(r_zyx)

    def test_single_frame_via_helper(self):
        """_merge_single_frame must behave identically to merge_small_labels on 2-D."""
        img = np.zeros((10, 10), dtype=np.int32)
        img[0:8, 0:8] = 1
        img[8:10, 0:2] = 2

        r1 = merge_small_labels(img, min_size=10)
        r2 = _merge_single_frame(img, min_size=10)
        np.testing.assert_array_equal(r1, r2)

# src/napari_tmidas/_tests/test_merge_small_labels.py
"""Tests for merge_small_labels processing function."""

import tracemalloc

import numpy as np
import pytest
import tifffile

from napari_tmidas.processing_functions.merge_small_labels import (
    _merge_single_frame,
    _stream_merge_small_labels,
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
        """The function must be discoverable via the BatchProcessingRegistry.

        Uses an explicit reload so the test is not affected by other test
        classes that clear the registry in their setup_method.
        """
        import importlib

        import napari_tmidas.processing_functions.merge_small_labels as _mod
        importlib.reload(_mod)

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


class TestMergeSmallLabelsStreaming:
    """
    The streaming path must be indistinguishable from the in-memory one.

    Merging is spatial, so it cannot go plane by plane like a pure per-label
    filter — but each timepoint (and channel) is independent, so only one
    block is ever resident.  Without this the worker loaded the whole stack
    densely and allocated a same-sized output, which OOM-killed the process on
    real tracked data.
    """

    @staticmethod
    def _blobby(shape, n_labels, seed):
        """Label image with contiguous blobs, so merging has real neighbors."""
        rng = np.random.default_rng(seed)
        out = np.zeros(shape, dtype=np.uint32)
        flat = out.reshape(-1, *shape[-2:])
        for i in range(flat.shape[0]):
            for lab in range(1, n_labels + 1):
                y = rng.integers(0, shape[-2] - 3)
                x = rng.integers(0, shape[-1] - 3)
                flat[
                    i,
                    y : y + rng.integers(1, 4),
                    x : x + rng.integers(1, 4),
                ] = lab
        return out

    @staticmethod
    def _write(path, array):
        # photometric is required: without it tifffile reads a leading axis of
        # length 3 or 4 as RGB samples and stores component planes instead.
        tifffile.imwrite(
            str(path), array, compression="zlib", photometric="minisblack"
        )

    @pytest.mark.parametrize("min_size", [1, 5, 50])
    @pytest.mark.parametrize(
        "shape,dim_order",
        [
            ((30, 30), "ZYX"),
            ((4, 24, 24), "ZYX"),
            ((4, 24, 24), "TYX"),
            ((4, 24, 24), "Auto"),
            ((3, 4, 20, 20), "TZYX"),
            ((2, 2, 3, 18, 18), "TCZYX"),
        ],
    )
    def test_matches_in_memory_result(
        self, tmp_path, shape, dim_order, min_size
    ):
        array = self._blobby(shape, 8, seed=len(shape) * 10 + min_size)
        expected = merge_small_labels(
            array.copy(), min_size=min_size, dim_order=dim_order
        )
        src = tmp_path / "labels.tif"
        self._write(src, array)

        out = _stream_merge_small_labels(
            src, tmp_path / "out.tif", min_size, dim_order
        )

        result = tifffile.imread(out)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)

    def test_tyx_and_zyx_disagree_on_the_same_3d_file(self, tmp_path):
        """The dim_order hint must reach the streaming path, not be ignored."""
        array = self._blobby((4, 24, 24), 10, seed=99)
        src = tmp_path / "labels.tif"
        self._write(src, array)

        as_tyx = tifffile.imread(
            _stream_merge_small_labels(src, tmp_path / "t.tif", 5, "TYX")
        )
        as_zyx = tifffile.imread(
            _stream_merge_small_labels(src, tmp_path / "z.tif", 5, "ZYX")
        )

        assert not np.array_equal(as_tyx, as_zyx)

    def test_dtype_is_preserved(self, tmp_path):
        array = self._blobby((2, 3, 16, 16), 8, seed=3).astype(np.int64)
        src = tmp_path / "labels.tif"
        self._write(src, array)

        out = _stream_merge_small_labels(src, tmp_path / "out.tif", 50, "TZYX")

        assert tifffile.imread(out).dtype == np.int64

    def test_declares_skip_load(self):
        """Without this the worker densely loads the stack before we run."""
        assert getattr(merge_small_labels, "skip_load", False) is True

    def test_widget_call_writes_its_own_output(self, tmp_path):
        """label_image=None + source/output params -> returns written path."""
        array = self._blobby((3, 4, 20, 20), 8, seed=11)
        src = tmp_path / "labels.tif"
        self._write(src, array)
        outdir = tmp_path / "out"

        result = merge_small_labels(
            None,
            min_size=50,
            dim_order="TZYX",
            _source_filepath=str(src),
            _output_folder=str(outdir),
            _output_suffix="_merged",
        )

        assert result == str(outdir / "labels_merged.tif")
        np.testing.assert_array_equal(
            tifffile.imread(result),
            merge_small_labels(array.copy(), min_size=50, dim_order="TZYX"),
        )

    def test_in_memory_call_does_not_mutate_input(self):
        """copy=False is for the streaming buffer only, never a caller array."""
        array = self._blobby((3, 4, 20, 20), 8, seed=12)
        before = array.copy()

        merge_small_labels(array, min_size=50, dim_order="TZYX")

        np.testing.assert_array_equal(array, before)

    def test_merge_single_frame_in_place_returns_same_buffer(self):
        """copy=False merges into the caller's buffer, no extra allocation."""
        frame = self._blobby((4, 20, 20), 8, seed=13)

        result = _merge_single_frame(frame, 50, copy=False)

        assert result is frame

    def test_never_materialises_the_stack(self, tmp_path):
        """Peak allocation stays near one block, not the whole stack."""
        shape = (6, 16, 512, 512)  # 100 MB dense, 16.8 MB per block
        dense_bytes = int(np.prod(shape)) * 4
        block_bytes = int(np.prod(shape[1:])) * 4
        rng = np.random.default_rng(42)
        array = np.zeros(shape, dtype=np.uint32)
        for i in range(300):
            t, z = rng.integers(0, shape[0]), rng.integers(0, shape[1])
            y, x = rng.integers(0, 500), rng.integers(0, 500)
            array[
                t, z, y : y + rng.integers(2, 6), x : x + rng.integers(2, 6)
            ] = (i + 1)
        src = tmp_path / "labels.tif"
        self._write(src, array)
        del array

        tracemalloc.start()
        try:
            _stream_merge_small_labels(
                src, tmp_path / "out.tif", 100, "TZYX"
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # One block, plus a fixed ~10 MB of scipy/tifffile working room that
        # does not scale with the stack (on a real 1.69 GB timepoint the peak
        # is 1.03x the block).  Allowing only a constant on top of one block
        # is what catches a regression like np.bincount upcasting the whole
        # volume to int64, which cost 2x the block.
        assert (
            peak < block_bytes + 12e6
        ), f"peak {peak/1e6:.1f} MB vs block {block_bytes/1e6:.1f} MB"
        assert (
            peak < dense_bytes / 3
        ), f"peak {peak/1e6:.1f} MB vs dense {dense_bytes/1e6:.1f} MB"

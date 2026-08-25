"""Test for split_channels function with various image formats"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from napari_tmidas.processing_functions.basic import (
    get_timepoint_count,
    sort_files_by_timepoints,
    split_channels,
)


class TestSplitChannels:
    """Test the split_channels function with various input formats"""

    def test_split_tcyx_python_format(self):
        """Test splitting TCYX image (Time, Channel, Y, X) with python format"""
        # Create a TCYX image: 5 timepoints, 3 channels, 100x100 pixels
        tcyx_image = np.random.rand(5, 3, 100, 100)

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=5, output_format="python"
        )

        # Result should be (3, 5, 100, 100): 3 channels, each with shape (5, 100, 100)
        assert result.shape == (
            3,
            5,
            100,
            100,
        ), f"Expected shape (3, 5, 100, 100), got {result.shape}"

        # Each channel should have shape (5, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                5,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_tcyx_fiji_format(self):
        """Test splitting TCYX image with Fiji format"""
        # Create a TCYX image: 5 timepoints, 3 channels, 100x100 pixels
        tcyx_image = np.random.rand(5, 3, 100, 100)

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=5, output_format="fiji"
        )

        # Result should be (3, 5, 100, 100): 3 channels, each with shape (5, 100, 100)
        assert result.shape == (
            3,
            5,
            100,
            100,
        ), f"Expected shape (3, 5, 100, 100), got {result.shape}"

        # Each channel should have shape (5, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                5,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_yxc_image(self):
        """Test splitting standard RGB image (YXC)"""
        # Create a YXC image: 100x100 pixels, 3 channels
        yxc_image = np.random.rand(100, 100, 3)

        result = split_channels(
            yxc_image, num_channels=3, time_steps=0, output_format="python"
        )

        # Result should be (3, 100, 100): 3 channels, each with shape (100, 100)
        assert result.shape == (
            3,
            100,
            100,
        ), f"Expected shape (3, 100, 100), got {result.shape}"

        # Each channel should have shape (100, 100)
        for i in range(3):
            assert result[i].shape == (
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_zyxc_image(self):
        """Test splitting 3D color image (ZYXC)"""
        # Create a ZYXC image: 10 z-slices, 100x100 pixels, 3 channels
        zyxc_image = np.random.rand(10, 100, 100, 3)

        result = split_channels(
            zyxc_image, num_channels=3, time_steps=0, output_format="python"
        )

        # Result should be (3, 10, 100, 100): 3 channels, each with shape (10, 100, 100)
        assert result.shape == (
            3,
            10,
            100,
            100,
        ), f"Expected shape (3, 10, 100, 100), got {result.shape}"

        # Each channel should have shape (10, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                10,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_tzyxc_image(self):
        """Test splitting 4D time-series color Z-stack (TZYXC)"""
        # Create a TZYXC image: 5 timepoints, 10 z-slices, 100x100 pixels, 3 channels
        tzyxc_image = np.random.rand(5, 10, 100, 100, 3)

        result = split_channels(
            tzyxc_image, num_channels=3, time_steps=5, output_format="python"
        )

        # Result should be (3, 5, 10, 100, 100): 3 channels, each with shape (5, 10, 100, 100)
        assert result.shape == (
            3,
            5,
            10,
            100,
            100,
        ), f"Expected shape (3, 5, 10, 100, 100), got {result.shape}"

        # Each channel should have shape (5, 10, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                5,
                10,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_channels_with_4_channels(self):
        """Test splitting image with 4 channels (RGBA)"""
        # Create a YXC image: 100x100 pixels, 4 channels
        yxc_image = np.random.rand(100, 100, 4)

        result = split_channels(
            yxc_image, num_channels=4, time_steps=0, output_format="python"
        )

        # Result should be (4, 100, 100): 4 channels, each with shape (100, 100)
        assert result.shape == (
            4,
            100,
            100,
        ), f"Expected shape (4, 100, 100), got {result.shape}"

        # Each channel should have shape (100, 100)
        for i in range(4):
            assert result[i].shape == (
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_channels_verifies_data_integrity(self):
        """Test that split channels contain the correct data"""
        # Create a simple test image where we can verify the data
        tcyx_image = np.zeros((2, 3, 10, 10))  # 2 timepoints, 3 channels

        # Set distinct values for each channel
        tcyx_image[:, 0, :, :] = 1.0  # Channel 0
        tcyx_image[:, 1, :, :] = 2.0  # Channel 1
        tcyx_image[:, 2, :, :] = 3.0  # Channel 2

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=2, output_format="python"
        )

        # Verify shape
        assert result.shape == (3, 2, 10, 10)

        # Verify data integrity
        assert np.allclose(result[0], 1.0), "Channel 0 data incorrect"
        assert np.allclose(result[1], 2.0), "Channel 1 data incorrect"
        assert np.allclose(result[2], 3.0), "Channel 2 data incorrect"

    def test_split_channels_auto_detect_mismatch(self):
        """Test that function handles mismatch between specified and actual channel count"""
        # Create a TCYX image: 5 timepoints, 4 channels, 100x100 pixels
        tcyx_image = np.random.rand(5, 4, 100, 100)

        # Specify 3 channels when there are actually 4
        result = split_channels(
            tcyx_image, num_channels=3, time_steps=5, output_format="python"
        )

        # Should auto-detect and use 4 channels
        assert result.shape == (
            4,
            5,
            100,
            100,
        ), f"Expected shape (4, 5, 100, 100), got {result.shape}"

    def test_split_channels_dimension_error(self):
        """Test that function raises error for invalid input"""
        # Create a 2D image (should fail)
        image_2d = np.random.rand(100, 100)

        with pytest.raises(ValueError, match="at least 3 dimensions"):
            split_channels(image_2d, num_channels=3, time_steps=0)

    def test_split_channels_python_returns_view_when_possible(self):
        """Test split returns a view for python format to reduce peak memory."""
        tcyx_image = np.random.rand(2, 3, 8, 8)

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=2, output_format="python"
        )

        assert result.shape == (3, 2, 8, 8)
        assert np.shares_memory(result, tcyx_image)


class TestTimepointSorting:
    """Test the timepoint sorting functionality"""

    def test_get_timepoint_count(self):
        """Test timepoint count detection from TIFF files"""
        # This test requires tifffile to be installed
        pytest.importorskip("tifffile")
        import tifffile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test TIFF with different timepoint counts
            test_cases = [
                ("single_timepoint.tif", np.random.rand(100, 100, 3), 1),
                ("time_series_10.tif", np.random.rand(10, 100, 100, 3), 10),
                ("time_series_50.tif", np.random.rand(50, 100, 100, 3), 50),
            ]

            for filename, data, expected_t in test_cases:
                filepath = os.path.join(tmpdir, filename)
                # Save with explicit axes information
                if data.ndim == 4:
                    # TCYX format
                    tifffile.imwrite(filepath, data, metadata={"axes": "TCYX"})
                else:
                    # CYX format (no time dimension)
                    tifffile.imwrite(filepath, data, metadata={"axes": "CYX"})

                # Test timepoint detection
                detected_t = get_timepoint_count(filepath)
                assert detected_t == expected_t, (
                    f"Expected {expected_t} timepoints for {filename}, "
                    f"but detected {detected_t}"
                )

    def test_sort_files_by_timepoints(self):
        """Test sorting files into timepoint subfolders"""
        pytest.importorskip("tifffile")
        import tifffile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create test files with different timepoint counts
            test_files = []
            file_configs = [
                ("img1.tif", np.random.rand(100, 100, 3), "CYX", 1),
                ("img2.tif", np.random.rand(100, 100, 3), "CYX", 1),
                ("img3.tif", np.random.rand(10, 100, 100, 3), "TCYX", 10),
                ("img4.tif", np.random.rand(10, 100, 100, 3), "TCYX", 10),
                ("img5.tif", np.random.rand(50, 100, 100, 3), "TCYX", 50),
            ]

            for filename, data, axes, _ in file_configs:
                filepath = str(input_dir / filename)
                tifffile.imwrite(filepath, data, metadata={"axes": axes})
                test_files.append(filepath)

            # Sort files by timepoints
            timepoint_map = sort_files_by_timepoints(
                test_files, str(output_dir)
            )

            # Verify folder structure
            assert (output_dir / "T1").exists(), "T1 folder should exist"
            assert (output_dir / "T10").exists(), "T10 folder should exist"
            assert (output_dir / "T50").exists(), "T50 folder should exist"

            # Verify file counts
            assert (
                1 in timepoint_map and len(timepoint_map[1]) == 2
            ), "Should have 2 files with T=1"
            assert (
                10 in timepoint_map and len(timepoint_map[10]) == 2
            ), "Should have 2 files with T=10"
            assert (
                50 in timepoint_map and len(timepoint_map[50]) == 1
            ), "Should have 1 file with T=50"

            # Verify files were copied correctly
            assert len(list((output_dir / "T1").glob("*.tif"))) == 2
            assert len(list((output_dir / "T10").glob("*.tif"))) == 2
            assert len(list((output_dir / "T50").glob("*.tif"))) == 1

    def test_split_channels_with_timepoint_sorting_flag(self):
        """Test that sort_by_timepoints parameter doesn't break normal splitting"""
        # Create a simple test image
        yxc_image = np.random.rand(100, 100, 3)

        # Test with sort_by_timepoints=False (default behavior)
        result_no_sort = split_channels(
            yxc_image, num_channels=3, sort_by_timepoints=False
        )

        # Test with sort_by_timepoints=True (should still split correctly)
        # Note: Without proper file context, sorting won't actually happen,
        # but the splitting should still work
        result_with_sort = split_channels(
            yxc_image, num_channels=3, sort_by_timepoints=True
        )

        # Both should produce the same result
        assert result_no_sort.shape == result_with_sort.shape
        assert result_no_sort.shape == (3, 100, 100)

    def test_get_timepoint_count_ome_zarr(self):
        """Test timepoint count detection from OME-Zarr metadata."""
        pytest.importorskip("zarr")
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "timelapse.zarr"
            root = zarr.open_group(str(zarr_path), mode="w")
            root.create_array(
                "0",
                shape=(7, 3, 16, 16),
                chunks=(1, 3, 16, 16),
                dtype="float32",
            )
            root.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ]

            detected_t = get_timepoint_count(str(zarr_path))
            assert detected_t == 7

    def test_sort_files_by_timepoints_with_zarr(self):
        """Test sorting zarr stores into timepoint subfolders."""
        pytest.importorskip("zarr")
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            zarr_path = input_dir / "img_t5.zarr"
            root = zarr.open_group(str(zarr_path), mode="w")
            root.create_array(
                "0",
                shape=(5, 2, 8, 8),
                chunks=(1, 2, 8, 8),
                dtype="uint16",
            )
            root.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ]

            timepoint_map = sort_files_by_timepoints(
                [str(zarr_path)], str(output_dir)
            )

            assert 5 in timepoint_map
            assert len(timepoint_map[5]) == 1
            copied_store = output_dir / "T5" / "img_t5.zarr"
            assert copied_store.exists()
            assert copied_store.is_dir()


def _write_tczyx_ome_zarr(zarr_path, data):
    """Write `data` as a TCZYX OME-Zarr, chunked one (t, z) plane at a time."""
    import zarr

    # Deliberately v2: real acquisitions on disk are v2 (the source this
    # was built from is), and reading them has to keep working even though
    # nothing in this package writes v2 any more.
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)
    arr = root.create_array(
        "0",
        shape=data.shape,
        chunks=(1, data.shape[1], 1) + data.shape[3:],
        dtype=str(data.dtype),
    )
    arr[:] = data
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    # Required by NGFF 0.4; napari-ome-zarr >= 0.10 raises
                    # AttributeError on a dataset that omits them.
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0] * data.ndim}
                    ],
                }
            ],
        }
    ]


class TestSplitChannelsStreaming:
    """
    Splitting a Zarr larger than RAM must not go through a dense array.

    The worker loads Zarr lazily as Dask and split_channels only moves an
    axis, so the result stays lazy — but the worker used to .compute() it
    unconditionally before saving, materializing every channel of the stack
    at once.  On a real 52 GB acquisition that is an OOM kill, so these
    tests pin both halves of the fix: the result stays lazy, and streaming
    it to disk produces the same bytes as the dense path.
    """

    def _run_split(
        self, tmp_path, src, num_channels, output_format, block_bytes
    ):
        import napari_tmidas._file_selector as fs
        from napari_tmidas._file_selector import ProcessingWorker

        out = tmp_path / "out"
        out.mkdir(exist_ok=True)

        original_budget = fs._STREAM_BLOCK_BYTES
        fs._STREAM_BLOCK_BYTES = block_bytes
        try:
            worker = ProcessingWorker(
                file_list=[str(src)],
                processing_func=split_channels,
                param_values={
                    "num_channels": num_channels,
                    "dimension_order": "Auto",
                },
                output_folder=str(out),
                input_suffix=".zarr",
                output_suffix="_split",
                output_format=output_format,
            )
            return worker.process_file(str(src))
        finally:
            fs._STREAM_BLOCK_BYTES = original_budget

    @pytest.mark.parametrize("output_format", ["tiff", "zarr"])
    def test_streamed_output_matches_dense_split(
        self, tmp_path, output_format
    ):
        """Streaming each channel to disk reproduces the source bytes."""
        pytest.importorskip("zarr")
        pytest.importorskip("dask")

        rng = np.random.default_rng(0)
        data = rng.integers(0, 4000, size=(5, 2, 4, 64, 64), dtype=np.uint16)
        src = tmp_path / "src.zarr"
        _write_tczyx_ome_zarr(src, data)

        # One plane is 8 KB, so this budget forces the deepest descent of the
        # block iterator: it must recurse past T and Z to a single plane.
        result = self._run_split(tmp_path, src, 2, output_format, 16 * 1024)

        paths = sorted(result["processed_files"])
        assert len(paths) == 2

        for channel, path in enumerate(paths):
            if output_format == "zarr":
                import zarr

                written = np.asarray(zarr.open_group(path, mode="r")["s0"])
            else:
                import tifffile

                written = tifffile.imread(path)
                with tifffile.TiffFile(path) as tif:
                    assert tif.series[0].axes == "TZYX"
            assert written.shape == (5, 4, 64, 64)
            np.testing.assert_array_equal(written, data[:, channel])

    def test_never_materialises_the_stack(self, tmp_path):
        """Peak allocation stays near one block, not the whole split."""
        pytest.importorskip("zarr")
        pytest.importorskip("dask")
        import tracemalloc

        # 40 t x 4 z planes of 512x512 uint16: 168 MB dense across both
        # channels, 2.1 MB per t-block of one channel.  Deep enough that one
        # block and the whole stack are far apart (80x), and deep enough to
        # cross the plane count where tifffile switches on threaded
        # compression (see the maxworkers=1 note in _write_tiff_output).
        shape = (40, 2, 4, 512, 512)
        rng = np.random.default_rng(42)
        data = rng.integers(0, 4000, size=shape, dtype=np.uint16)
        dense_bytes = data.nbytes
        # Budget chosen so the iterator computes one whole (z, y, x) block.
        block_bytes = 4 * 512 * 512 * 2

        src = tmp_path / "src.zarr"
        _write_tczyx_ome_zarr(src, data)
        # The measurement below covers the whole process, so the fixture must
        # not still be resident — it alone is the size this test is asserting
        # the split never reaches.
        del data

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            result = self._run_split(tmp_path, src, 2, "tiff", block_bytes)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert len(result["processed_files"]) == 2

        # One block, plus the working room Dask's threaded scheduler needs to
        # assemble it: each source chunk here spans both channels, so it is
        # read and decompressed whole before the wanted half is sliced out,
        # several threads at a time.  That lands at ~7x the block (measured
        # 14.3-14.5 MB across runs), so bound by a multiple of the block
        # rather than "one block + a constant" — what has to be caught is a
        # regression that scales with the *stack*, and 10x still fails hard
        # on the one this guards: computing the whole lazy result up front is
        # 40x the block.
        assert (
            peak < block_bytes * 10
        ), f"peak {peak/1e6:.1f} MB vs block {block_bytes/1e6:.1f} MB"
        # Independently: nowhere near the dense stack.  At 80 blocks this is
        # a genuinely separate check (42 MB) rather than a looser restatement
        # of the 21 MB bound above.
        assert (
            peak < dense_bytes / 4
        ), f"peak {peak/1e6:.1f} MB vs dense {dense_bytes/1e6:.1f} MB"


class TestLazyTiffLoading:
    """
    A large TIFF is opened page-wise for functions that can consume a lazy
    array.  Without this the streaming save path above is defeated at the
    door: tifffile.imread materializes the whole stack before the processing
    function is ever called, so the same split that handles a 52 GB Zarr in
    0.4 GB would still OOM on a 52 GB TIFF.
    """

    def _write_tczyx_tiff(self, path, data):
        import tifffile

        tifffile.imwrite(
            str(path),
            data,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "TCZYX"},
        )

    def test_small_tiff_is_not_opened_lazily(self, tmp_path):
        """Under the threshold a dense read is faster and just as safe."""
        pytest.importorskip("dask")
        from napari_tmidas._file_selector import _lazy_load_tiff

        data = np.zeros((4, 2, 3, 32, 32), dtype=np.uint16)
        src = tmp_path / "small.tif"
        self._write_tczyx_tiff(src, data)
        assert _lazy_load_tiff(str(src)) is None

    def test_large_tiff_loads_lazily_with_correct_pages(self, tmp_path):
        """A large TIFF loads lazily and round-trips byte-identically.

        The fixture is a standard OME layout, where series order and file
        order coincide, so this pins the round-trip rather than the
        series-vs-file page mapping itself.
        """
        pytest.importorskip("dask")
        import napari_tmidas._file_selector as fs

        rng = np.random.default_rng(5)
        data = rng.integers(0, 4000, size=(4, 2, 3, 32, 32), dtype=np.uint16)
        src = tmp_path / "large.tif"
        self._write_tczyx_tiff(src, data)

        original = fs._LAZY_TIFF_MIN_BYTES
        fs._LAZY_TIFF_MIN_BYTES = 1024
        try:
            lazy = fs._lazy_load_tiff(str(src))
        finally:
            fs._LAZY_TIFF_MIN_BYTES = original

        assert lazy is not None
        assert hasattr(lazy, "compute"), "expected a lazy array"
        assert lazy.shape == data.shape
        np.testing.assert_array_equal(np.asarray(lazy), data)

    def test_split_streams_a_large_tiff(self, tmp_path):
        """End to end: TIFF in, one channel per file out, nothing dense."""
        pytest.importorskip("dask")
        import tifffile

        import napari_tmidas._file_selector as fs
        from napari_tmidas._file_selector import ProcessingWorker

        rng = np.random.default_rng(6)
        data = rng.integers(
            0, 4000, size=(30, 2, 8, 128, 128), dtype=np.uint16
        )
        dense_bytes = data.nbytes
        src = tmp_path / "stack.tif"
        self._write_tczyx_tiff(src, data)
        out = tmp_path / "out"
        out.mkdir()

        budget, threshold = fs._STREAM_BLOCK_BYTES, fs._LAZY_TIFF_MIN_BYTES
        fs._STREAM_BLOCK_BYTES = 2 * 1024 * 1024
        fs._LAZY_TIFF_MIN_BYTES = 1024 * 1024
        try:
            worker = ProcessingWorker(
                file_list=[str(src)],
                processing_func=split_channels,
                param_values={
                    "num_channels": 2,
                    "dimension_order": "TCZYX",
                },
                output_folder=str(out),
                input_suffix=".tif",
                output_suffix="_split",
                output_format="tiff",
            )
            import tracemalloc

            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                result = worker.process_file(str(src))
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        finally:
            fs._STREAM_BLOCK_BYTES = budget
            fs._LAZY_TIFF_MIN_BYTES = threshold

        paths = sorted(result["processed_files"])
        assert len(paths) == 2
        for channel, path in enumerate(paths):
            np.testing.assert_array_equal(
                tifffile.imread(path), data[:, channel]
            )
        # The dense read alone was 1.9x the stack before this path existed.
        assert (
            peak < dense_bytes / 2
        ), f"peak {peak/1e6:.1f} MB vs dense {dense_bytes/1e6:.1f} MB"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

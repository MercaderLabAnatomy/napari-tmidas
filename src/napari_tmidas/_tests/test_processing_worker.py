# src/napari_tmidas/_tests/test_processing_worker.py
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import tifffile

from napari_tmidas._processing_worker import (
    ProcessingWorker,
    load_image_file_lazy,
    save_image_file,
)


class TestProcessingWorker:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_worker_initialization(self):
        """Test ProcessingWorker initialization"""
        file_list = ["/path/to/file1.tif", "/path/to/file2.tif"]
        processing_func = Mock()
        param_values = {"param1": "value1"}
        output_folder = "/output"
        input_suffix = ".tif"
        output_suffix = "_processed.tif"

        worker = ProcessingWorker(
            file_list,
            processing_func,
            param_values,
            output_folder,
            input_suffix,
            output_suffix,
        )

        assert worker.file_list == file_list
        assert worker.processing_func == processing_func
        assert worker.param_values == param_values
        assert worker.output_folder == output_folder
        assert worker.input_suffix == input_suffix
        assert worker.output_suffix == output_suffix
        assert not worker.stop_requested
        assert worker.thread_count >= 1

    def test_worker_stop(self):
        """Test stopping the worker"""
        worker = ProcessingWorker([], Mock(), {}, "", "", "")
        assert not worker.stop_requested
        worker.stop()
        assert worker.stop_requested

    @patch(
        "napari_tmidas._processing_worker.concurrent.futures.ThreadPoolExecutor"
    )
    @patch("napari_tmidas._processing_worker.load_image_file")
    def test_process_file_single_output(self, mock_load, mock_executor):
        """Test processing a file with single output"""
        # Mock the executor and future
        mock_future = Mock()
        mock_future.result.return_value = np.random.rand(100, 100)
        mock_executor.return_value.__enter__.return_value.submit.return_value = (
            mock_future
        )
        mock_executor.return_value.__enter__.return_value.as_completed.return_value = [
            mock_future
        ]

        # Mock image loading
        mock_load.return_value = np.random.rand(100, 100)

        # Create worker
        worker = ProcessingWorker(
            ["/test/file.tif"],
            Mock(return_value=np.random.rand(100, 100)),
            {},
            self.temp_dir,
            ".tif",
            "_processed.tif",
        )

        # Mock the run method to avoid threading issues
        worker.run = Mock()

        # Test process_file method
        result = worker.process_file("/test/file.tif")

        assert result is not None
        assert "original_file" in result
        assert "processed_file" in result

    @patch("napari_tmidas._processing_worker.load_image_file")
    def test_process_file_multiple_outputs(self, mock_load):
        """Test processing a file with multiple outputs"""
        # Mock image loading
        mock_load.return_value = np.random.rand(100, 100)

        # Create worker with function that returns multiple outputs
        def multi_output_func(image):
            return [image, image * 2, image * 3]

        worker = ProcessingWorker(
            ["/test/file.tif"],
            multi_output_func,
            {},
            self.temp_dir,
            ".tif",
            "_processed.tif",
        )

        result = worker.process_file("/test/file.tif")

        assert result is not None
        assert "original_file" in result
        assert "processed_files" in result
        assert len(result["processed_files"]) == 3

    @patch("napari_tmidas._processing_worker.load_image_file")
    def test_process_file_folder_function(self, mock_load):
        """Test processing with folder function that returns None"""
        # Mock image loading
        mock_load.return_value = np.random.rand(100, 100)

        # Create worker with folder function
        def folder_func(image):
            return None  # Folder functions don't return processed images

        worker = ProcessingWorker(
            ["/test/file.tif"],
            folder_func,
            {},
            self.temp_dir,
            ".tif",
            "_processed.tif",
        )

        result = worker.process_file("/test/file.tif")

        assert result is not None
        assert result["processed_file"] is None

    def test_save_image_file_dask_streaming(self):
        """Test that Dask arrays can be saved to TIFF without eager full-array conversion."""
        da = pytest.importorskip("dask.array")

        arr = da.random.random((4, 32, 32), chunks=(1, 32, 32)).astype(
            np.float32
        )
        out_path = f"{self.temp_dir}/dask_stream.tif"

        save_image_file(arr, out_path, np.float32)

        saved = tifffile.imread(out_path)
        assert saved.shape == (4, 32, 32)
        assert saved.dtype == np.float32

    def test_best_tiff_compression_falls_back_to_zlib(self, monkeypatch):
        """zstd when imagecodecs is present, zlib otherwise."""
        import builtins

        from napari_tmidas import _processing_worker as pw

        assert pw._best_tiff_compression() in ("zstd", "zlib")

        real_import = builtins.__import__

        def no_imagecodecs(name, *args, **kwargs):
            if name == "imagecodecs":
                raise ImportError("simulated: imagecodecs missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_imagecodecs)
        assert pw._best_tiff_compression() == "zlib"

    def test_save_image_file_dask_streaming_4d_tzyx(self):
        """4D TZYX dask arrays must stream page-by-page (regression: reshape error)."""
        da = pytest.importorskip("dask.array")

        arr_np = np.arange(2 * 3 * 8 * 8, dtype=np.uint32).reshape(2, 3, 8, 8)
        arr = da.from_array(arr_np, chunks=(1, 1, 8, 8))
        out_path = f"{self.temp_dir}/dask_tzyx.tif"

        save_image_file(arr, out_path, np.uint32)

        saved = tifffile.imread(out_path)
        assert saved.shape == (2, 3, 8, 8)
        assert saved.dtype == np.uint32
        np.testing.assert_array_equal(saved, arr_np)

    def test_load_image_file_lazy_returns_dask_for_tiff(self):
        """Large multi-page TIFFs must load lazily (dask), not fully into RAM."""
        da = pytest.importorskip("dask.array")

        arr = np.arange(5 * 16 * 16, dtype=np.uint32).reshape(5, 16, 16)
        path = f"{self.temp_dir}/movie_labels.tif"
        tifffile.imwrite(path, arr, photometric="minisblack")

        lazy = load_image_file_lazy(path)

        assert isinstance(lazy, da.Array)
        assert lazy.shape == (5, 16, 16)
        # Lazy handle still yields correct data when materialized.
        np.testing.assert_array_equal(np.asarray(lazy), arr)

    def test_path_loading_function_skips_eager_load(self):
        """A function marked _loads_from_path must trigger the lazy loader."""
        arr = np.zeros((2, 8, 8), dtype=np.uint32)
        path = f"{self.temp_dir}/movie_labels.tif"
        tifffile.imwrite(path, arr)

        def fake_func(image, **kwargs):
            # Path-based functions return an output path string.
            return path

        fake_func._loads_from_path = True

        worker = ProcessingWorker(
            [path], fake_func, {}, self.temp_dir, ".tif", "_tracked.tif"
        )

        with patch(
            "napari_tmidas._processing_worker.load_image_file_lazy"
        ) as lazy_loader, patch(
            "napari_tmidas._processing_worker.load_image_file"
        ) as eager_loader:
            lazy_loader.return_value = arr
            worker.process_file(path)

        lazy_loader.assert_called_once_with(path)
        eager_loader.assert_not_called()

# src/napari_tmidas/_tests/test_processing_worker.py
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from napari_tmidas._processing_worker import ProcessingWorker


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

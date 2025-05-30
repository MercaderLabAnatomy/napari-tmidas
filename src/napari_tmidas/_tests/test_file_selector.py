# src/napari_tmidas/_tests/test_file_selector.py
import os
import tempfile
from unittest.mock import Mock

import numpy as np

from napari_tmidas._file_selector import ProcessingWorker, file_selector
from napari_tmidas._registry import BatchProcessingRegistry


class TestProcessingWorker:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        BatchProcessingRegistry._processing_functions.clear()

        # Register a test function
        @BatchProcessingRegistry.register(name="Test Process", suffix="_proc")
        def test_process(image):
            return image * 2

        self.test_func = BatchProcessingRegistry.get_function_info(
            "Test Process"
        )["func"]

    def teardown_method(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_process_file(self):
        """Test processing a single file"""
        # Create test image
        test_image = np.random.rand(100, 100)
        input_path = os.path.join(self.temp_dir, "test.tif")

        import tifffile

        tifffile.imwrite(input_path, test_image)

        # Create worker
        worker = ProcessingWorker(
            [input_path], self.test_func, {}, self.temp_dir, "", "_proc"
        )

        # Process file
        result = worker.process_file(input_path)

        assert result is not None
        assert "original_file" in result
        assert "processed_file" in result
        assert os.path.exists(result["processed_file"])

    def test_multi_channel_output(self):
        """Test processing that outputs multiple channels"""

        @BatchProcessingRegistry.register(
            name="Split Channels", suffix="_split"
        )
        def split_channels(image):
            return np.stack([image, image * 2, image * 3])

        test_image = np.random.rand(100, 100)
        input_path = os.path.join(self.temp_dir, "test.tif")

        import tifffile

        tifffile.imwrite(input_path, test_image)

        func_info = BatchProcessingRegistry.get_function_info("Split Channels")
        worker = ProcessingWorker(
            [input_path], func_info["func"], {}, self.temp_dir, "", "_split"
        )

        result = worker.process_file(input_path)

        assert "processed_files" in result
        assert len(result["processed_files"]) == 3


class TestFileSelector:
    def test_file_selector_widget_creation(self):
        """Test that file selector widget is created properly"""
        viewer_mock = Mock()

        # Test the widget can be called
        result = file_selector(viewer_mock, "/tmp", ".tif")
        assert isinstance(result, list)

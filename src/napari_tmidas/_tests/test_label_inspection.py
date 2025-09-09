# src/napari_tmidas/_tests/test_label_inspection.py
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from napari_tmidas._label_inspection import (
    LabelInspector,
    label_inspector_widget,
)


class TestLabelInspector:
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.viewer = Mock()

    def teardown_method(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_label_inspector_initialization(self):
        """Test LabelInspector initialization"""
        inspector = LabelInspector(self.viewer)
        assert inspector.viewer == self.viewer
        assert inspector.image_label_pairs == []
        assert inspector.current_index == 0

    def test_load_image_label_pairs_no_folder(self):
        """Test loading pairs with non-existent folder"""
        inspector = LabelInspector(self.viewer)
        inspector.load_image_label_pairs("/nonexistent/folder", "_labels")
        assert (
            self.viewer.status
            == "Folder path does not exist: /nonexistent/folder"
        )

    def test_load_image_label_pairs_no_labels(self):
        """Test loading pairs with no label files"""
        inspector = LabelInspector(self.viewer)

        # Create empty folder
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)

        inspector.load_image_label_pairs(empty_dir, "_labels")
        assert self.viewer.status == "No files found with suffix '_labels'"

    @patch("napari_tmidas._label_inspection.imread")
    def test_load_image_label_pairs_valid(self, mock_imread):
        """Test loading valid image-label pairs"""
        inspector = LabelInspector(self.viewer)

        # Create test files
        test_dir = os.path.join(self.temp_dir, "test")
        os.makedirs(test_dir)

        # Create image and label files
        image_path = os.path.join(test_dir, "test_image.tif")
        label_path = os.path.join(test_dir, "test_image_labels.tif")

        with open(image_path, "w") as f:
            f.write("dummy")
        with open(label_path, "w") as f:
            f.write("dummy")

        # Mock imread to return valid label data
        mock_imread.return_value = np.ones((10, 10), dtype=np.uint32)

        inspector.load_image_label_pairs(test_dir, "_labels")

        # Check that pairs were loaded
        assert len(inspector.image_label_pairs) == 1
        assert inspector.image_label_pairs[0] == (image_path, label_path)


class TestLabelInspectorWidget:
    def test_widget_creation(self):
        """Test that the label inspector widget can be imported and called"""
        # Just test that the function exists and can be called
        # (without actually creating the widget to avoid Qt issues)
        assert callable(label_inspector_widget)

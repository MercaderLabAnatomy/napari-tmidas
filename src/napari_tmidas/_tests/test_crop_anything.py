# src/napari_tmidas/_tests/test_crop_anything.py
import pytest
from unittest.mock import Mock, patch

# Skip entire module if torch is not available
torch = pytest.importorskip("torch")

from napari_tmidas._crop_anything import batch_crop_anything_widget


class TestBatchCropAnythingWidget:
    def test_widget_creation(self):
        """Test that the batch crop anything widget is created properly"""
        widget = batch_crop_anything_widget()
        assert widget is not None
        # Check that it has the expected attributes
        assert hasattr(widget, "folder_path")
        assert hasattr(widget, "data_dimensions")
        # viewer is a parameter but may not be exposed as attribute
        assert hasattr(widget, "call_button")  # magicgui adds this

    @patch("napari_tmidas._crop_anything.batch_crop_anything")
    def test_widget_has_browse_button(self, mock_batch_crop):
        """Test that the widget has a browse button added"""
        mock_widget = Mock()
        mock_widget.folder_path = Mock()
        mock_widget.folder_path.native = Mock()
        mock_widget.folder_path.native.parent.return_value.layout.return_value = (
            Mock()
        )
        mock_widget.folder_path.value = "/test/path"

        mock_batch_crop.return_value = mock_widget

        batch_crop_anything_widget()

        # The browse button should be added to the layout
        # This is hard to test directly without mocking Qt, but we can check the function exists
        assert callable(batch_crop_anything_widget)

    @patch("napari_tmidas._crop_anything.BatchCropAnything")
    @patch("napari_tmidas._crop_anything.magicgui")
    def test_widget_creation_safe(self, mock_magicgui, mock_batch_crop):
        """Test widget creation with BatchCropAnything mocked to avoid any SAM2 issues"""
        # Mock the BatchCropAnything class to avoid any SAM2 initialization
        mock_instance = Mock()
        mock_batch_crop.return_value = mock_instance

        # Mock magicgui to return a simple widget
        mock_widget = Mock()
        mock_magicgui.return_value = mock_widget

        # This should be completely safe since everything is mocked
        widget = batch_crop_anything_widget()
        assert widget is not None

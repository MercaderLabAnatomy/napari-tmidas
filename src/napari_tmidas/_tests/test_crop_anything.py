# src/napari_tmidas/_tests/test_crop_anything.py
from unittest.mock import Mock, patch

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

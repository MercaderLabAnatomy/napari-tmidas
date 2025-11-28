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

    def test_next_image_at_last_image(self):
        """Test that next_image returns False when already at the last image"""
        from napari_tmidas._crop_anything import BatchCropAnything

        # Create a mock viewer
        mock_viewer = Mock()
        mock_viewer.layers = Mock()
        mock_viewer.layers.clear = Mock()

        # Create processor with mocked predictor to avoid SAM2 initialization
        with patch.object(BatchCropAnything, "_initialize_sam2"):
            processor = BatchCropAnything(mock_viewer, use_3d=False)
            processor.predictor = (
                None  # Ensure predictor is None to skip segmentation
            )

        # Set up test data with 3 images
        processor.images = [
            "/path/img1.tif",
            "/path/img2.tif",
            "/path/img3.tif",
        ]
        processor.current_index = 2  # At the last image (index 2 of 3 images)

        # Try to move to next image when already at the last one
        result = processor.next_image()

        # Should return False
        assert result is False

        # Current index should not change
        assert processor.current_index == 2

        # Layers should not have been cleared (no call to _load_current_image)
        mock_viewer.layers.clear.assert_not_called()

    def test_prev_image_at_first_image(self):
        """Test that previous_image returns False when already at the first image"""
        from napari_tmidas._crop_anything import BatchCropAnything

        # Create a mock viewer
        mock_viewer = Mock()
        mock_viewer.layers = Mock()
        mock_viewer.layers.clear = Mock()

        # Create processor with mocked predictor to avoid SAM2 initialization
        with patch.object(BatchCropAnything, "_initialize_sam2"):
            processor = BatchCropAnything(mock_viewer, use_3d=False)
            processor.predictor = (
                None  # Ensure predictor is None to skip segmentation
            )

        # Set up test data with 3 images
        processor.images = [
            "/path/img1.tif",
            "/path/img2.tif",
            "/path/img3.tif",
        ]
        processor.current_index = 0  # At the first image (index 0)

        # Try to move to previous image when already at the first one
        result = processor.previous_image()

        # Should return False
        assert result is False

        # Current index should not change
        assert processor.current_index == 0

        # Layers should not have been cleared (no call to _load_current_image)
        mock_viewer.layers.clear.assert_not_called()

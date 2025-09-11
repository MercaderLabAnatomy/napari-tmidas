# src/napari_tmidas/_tests/test_ui_utils.py
import pytest
from unittest.mock import Mock, patch

# Skip entire module if qtpy is not available (GUI dependencies)
qtpy = pytest.importorskip("qtpy")

from napari_tmidas._ui_utils import add_browse_button_to_folder_field


@pytest.mark.skip(reason="GUI tests require QApplication and can crash in headless environment")
class TestUIUtils:
    def test_add_browse_button_to_folder_field(self):
        """Test adding browse button to a folder field"""
        # Create mock widget
        mock_widget = Mock()
        mock_folder_field = Mock()
        mock_folder_field.value = "/test/path"
        mock_folder_field.native = Mock()
        mock_parent = Mock()
        mock_layout = Mock()
        mock_parent.layout.return_value = mock_layout
        mock_folder_field.native.parent.return_value = mock_parent

        mock_widget.folder_path = mock_folder_field

        # Call the function
        result = add_browse_button_to_folder_field(mock_widget, "folder_path")
        assert result == mock_widget

    def test_add_browse_button_with_existing_value(self):
        """Test adding browse button when folder field has existing value"""
        # Create mock widget
        mock_widget = Mock()
        mock_folder_field = Mock()
        mock_folder_field.value = "/existing/path"
        mock_folder_field.native = Mock()
        mock_parent = Mock()
        mock_layout = Mock()
        mock_parent.layout.return_value = mock_layout
        mock_folder_field.native.parent.return_value = mock_parent

        mock_widget.existing_path = mock_folder_field

        # Call the function
        result = add_browse_button_to_folder_field(
            mock_widget, "existing_path"
        )

        # Check that the button was added to the layout
        mock_layout.addWidget.assert_called_once()
        assert result == mock_widget

    def test_add_browse_button_with_empty_value(self):
        """Test adding browse button when folder field is empty"""
        # Create mock widget
        mock_widget = Mock()
        mock_folder_field = Mock()
        mock_folder_field.value = ""  # Empty value
        mock_folder_field.native = Mock()
        mock_parent = Mock()
        mock_layout = Mock()
        mock_parent.layout.return_value = mock_layout
        mock_folder_field.native.parent.return_value = mock_parent

        mock_widget.empty_path = mock_folder_field

        # Call the function
        result = add_browse_button_to_folder_field(mock_widget, "empty_path")

        # Check that the button was added to the layout
        mock_layout.addWidget.assert_called_once()
        assert result == mock_widget

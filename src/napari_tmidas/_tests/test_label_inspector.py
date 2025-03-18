import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from napari.layers import Image, Labels
from napari.viewer import Viewer

# Import the module to test
from napari_tmidas.label_inspector import LabelInspector


@pytest.fixture
def mock_viewer():
    """Create a mock Napari viewer for testing."""
    viewer = MagicMock(spec=Viewer)
    viewer.layers = MagicMock()
    viewer.layers.clear = MagicMock()
    viewer.add_image = MagicMock(return_value=MagicMock(spec=Image))
    viewer.add_labels = MagicMock(return_value=MagicMock(spec=Labels))
    return viewer


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory with test image and label files."""
    # Create test directory
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Create test files
    # Case 1: Standard naming pattern
    img1 = test_dir / "sample1.tif"
    lbl1 = test_dir / "sample1_labels.tif"
    np.random.seed(42)
    np.save(img1, np.random.rand(10, 10))
    np.save(lbl1, np.random.randint(0, 5, (10, 10)))

    # Case 2: Different base name for image and label
    img2 = test_dir / "image2.tif"
    lbl2 = test_dir / "image2_otsu_labels.tif"
    np.save(img2, np.random.rand(10, 10))
    np.save(lbl2, np.random.randint(0, 5, (10, 10)))

    # Case 3: Multiple images that could match the label prefix
    img3a = test_dir / "sample3.tif"
    img3b = test_dir / "sample3_filtered.tif"
    lbl3 = test_dir / "sample3_labels.tif"
    np.save(img3a, np.random.rand(10, 10))
    np.save(img3b, np.random.rand(10, 10))
    np.save(lbl3, np.random.randint(0, 5, (10, 10)))

    return str(test_dir)


def test_initialization(mock_viewer):
    """Test that the LabelInspector initializes correctly."""
    inspector = LabelInspector(mock_viewer)
    assert inspector.viewer == mock_viewer
    assert inspector.image_label_pairs == []
    assert inspector.current_index == 0


@patch("napari_tmidas.label_inspector.imread")
def test_load_image_label_pairs(mock_imread, mock_viewer, temp_test_dir):
    """Test loading image-label pairs from a directory."""
    inspector = LabelInspector(mock_viewer)

    # Test loading with standard suffix
    inspector.load_image_label_pairs(temp_test_dir, "_labels.tif")
    assert (
        len(inspector.image_label_pairs) >= 2
    )  # Should find at least 2 pairs

    # Verify paths in the pairs
    paths = [
        os.path.basename(path)
        for pair in inspector.image_label_pairs
        for path in pair
    ]
    assert "sample1.tif" in paths
    assert "sample1_labels.tif" in paths

    # Test with a more specific suffix
    inspector.load_image_label_pairs(temp_test_dir, "_otsu_labels.tif")
    assert len(inspector.image_label_pairs) == 1
    assert os.path.basename(inspector.image_label_pairs[0][0]) == "image2.tif"
    assert (
        os.path.basename(inspector.image_label_pairs[0][1])
        == "image2_otsu_labels.tif"
    )


@patch("napari_tmidas.label_inspector.imread")
def test_load_current_pair(mock_imread, mock_viewer, temp_test_dir):
    """Test loading a pair into the viewer."""
    inspector = LabelInspector(mock_viewer)

    # Mock the imread function to return dummy data
    mock_imread.return_value = np.zeros((10, 10))

    # Load pairs and test _load_current_pair
    inspector.load_image_label_pairs(temp_test_dir, "_labels.tif")
    inspector._load_current_pair()

    # Check if layers were cleared and new layers added
    mock_viewer.layers.clear.assert_called_once()
    assert mock_viewer.add_image.call_count == 1
    assert mock_viewer.add_labels.call_count == 1


def test_next_pair(mock_viewer):
    """Test navigation to the next pair."""
    inspector = LabelInspector(mock_viewer)

    # Setup mock image-label pairs
    inspector.image_label_pairs = [
        ("/path/to/img1.tif", "/path/to/lbl1.tif"),
        ("/path/to/img2.tif", "/path/to/lbl2.tif"),
        ("/path/to/img3.tif", "/path/to/lbl3.tif"),
    ]

    # Mock the _load_current_pair and save_current_labels methods
    inspector._load_current_pair = MagicMock()
    inspector.save_current_labels = MagicMock()

    # Test navigation to next pair
    assert inspector.current_index == 0
    result = inspector.next_pair()
    assert result is True
    assert inspector.current_index == 1
    inspector.save_current_labels.assert_called_once()

    # Navigate to the last pair
    inspector.current_index = len(inspector.image_label_pairs) - 1
    result = inspector.next_pair()
    assert result is False
    assert inspector.current_index == len(inspector.image_label_pairs) - 1
    mock_viewer.layers.clear.assert_called_once()


def test_save_current_labels(mock_viewer):
    """Test saving the current labels."""
    inspector = LabelInspector(mock_viewer)

    # Setup mock image-label pairs
    inspector.image_label_pairs = [
        ("/path/to/img1.tif", "/path/to/lbl1.tif"),
    ]

    # Create a mock Labels layer
    mock_labels_layer = MagicMock(spec=Labels)
    mock_labels_layer.data = np.zeros((10, 10), dtype=np.uint32)
    mock_labels_layer.save = MagicMock()

    # Mock the viewer.layers to return our mock Labels layer
    mock_viewer.layers = [mock_labels_layer]

    # Test saving labels
    inspector.save_current_labels()
    mock_labels_layer.save.assert_called_once_with("/path/to/lbl1.tif")


def test_empty_conditions(mock_viewer):
    """Test behavior when no image-label pairs are found."""
    inspector = LabelInspector(mock_viewer)

    # Test with empty image_label_pairs
    inspector.image_label_pairs = []

    # These should not raise exceptions but set status messages
    inspector._load_current_pair()
    assert mock_viewer.status == "No pairs to inspect."

    inspector.save_current_labels()
    assert mock_viewer.status == "No pairs to save."

    result = inspector.next_pair()
    assert result is None
    assert mock_viewer.status == "No pairs to inspect."


@patch("napari_tmidas.label_inspector.LabelInspector")
def test_label_inspector_widget(mock_label_inspector_class, mock_viewer):
    """Test the label_inspector MagicGUI widget."""
    from napari_tmidas.label_inspector import label_inspector

    # Call the widget function with test parameters
    mock_inspector = MagicMock()
    mock_label_inspector_class.return_value = mock_inspector

    label_inspector(
        folder_path="/test/path",
        label_suffix="_labels.tif",
        viewer=mock_viewer,
    )

    # Verify LabelInspector was instantiated and methods called
    mock_label_inspector_class.assert_called_once_with(mock_viewer)
    mock_inspector.load_image_label_pairs.assert_called_once_with(
        "/test/path", "_labels.tif"
    )

"""
Tests for the Frame Removal Widget
"""

import os
import sys

import numpy as np
import pytest

# Skip all Qt widget tests on macOS CI due to segfaults with headless Qt/OpenGL
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless environment)",
)


def test_frame_removal_widget_import():
    """Test that the frame removal widget can be imported."""
    from napari_tmidas._frame_removal import frame_removal_widget

    assert frame_removal_widget is not None


def test_frame_removal_widget_creation(make_napari_viewer):
    """Test that the frame removal widget can be created."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    assert widget is not None
    assert widget.viewer == viewer
    assert widget.original_data is None
    assert widget.frames_to_remove == []


def test_frame_removal_widget_with_tyx_image(make_napari_viewer):
    """Test frame removal widget with TYX image."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create a TYX image (10 time frames, 64x64)
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test_tyx")

    # Select the layer
    widget._on_layer_selected(layer)

    assert widget.original_data is not None
    assert widget.original_data.shape == (10, 64, 64)
    assert widget.is_tzyx is False
    assert widget.frame_slider.maximum() == 9


def test_frame_removal_widget_with_tzyx_image(make_napari_viewer):
    """Test frame removal widget with TZYX image."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create a TZYX image (5 time frames, 8 z-slices, 32x32)
    test_data = np.random.randint(0, 255, (5, 8, 32, 32), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test_tzyx")

    # Select the layer
    widget._on_layer_selected(layer)

    assert widget.original_data is not None
    assert widget.original_data.shape == (5, 8, 32, 32)
    assert widget.is_tzyx is True
    assert widget.frame_slider.maximum() == 4


def test_frame_removal_navigation(make_napari_viewer):
    """Test frame navigation functionality."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create test data
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Test navigation
    assert widget.current_frame == 0

    widget._next_frame()
    assert widget.current_frame == 1

    widget._next_frame()
    assert widget.current_frame == 2

    widget._prev_frame()
    assert widget.current_frame == 1

    # Test slider
    widget._on_slider_changed(5)
    assert widget.current_frame == 5


def test_frame_marking(make_napari_viewer):
    """Test frame marking functionality."""
    from napari_tmidas._frame_removal import FrameRemovalWidget
    from qtpy.QtCore import Qt

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create test data
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Mark frame 0
    widget._on_mark_changed(Qt.Checked)
    assert 0 in widget.frames_to_remove

    # Navigate to frame 3 and mark it
    widget.current_frame = 3
    widget._on_mark_changed(Qt.Checked)
    assert 3 in widget.frames_to_remove

    # Unmark frame 0
    widget.current_frame = 0
    widget._on_mark_changed(Qt.Unchecked)
    assert 0 not in widget.frames_to_remove
    assert 3 in widget.frames_to_remove


def test_clear_marks(make_napari_viewer, monkeypatch):
    """Test clearing all marks."""
    from napari_tmidas._frame_removal import FrameRemovalWidget
    from qtpy.QtWidgets import QMessageBox

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Mock QMessageBox.question to always return Yes
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )

    # Create test data and mark some frames
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    widget.frames_to_remove = [1, 3, 5]
    widget._clear_marks()

    assert widget.frames_to_remove == []


def test_create_cleaned_data(make_napari_viewer):
    """Test creation of cleaned data."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create test data with distinct values per frame
    test_data = np.zeros((10, 64, 64), dtype=np.uint8)
    for i in range(10):
        test_data[i] = i  # Each frame has value equal to its index

    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Mark frames 2, 5, 7 for removal
    widget.frames_to_remove = [2, 5, 7]

    # Create cleaned data
    cleaned_data = widget._create_cleaned_data()

    # Should have 7 frames remaining (10 - 3)
    assert cleaned_data.shape[0] == 7

    # Check that remaining frames are correct
    expected_frames = [0, 1, 3, 4, 6, 8, 9]
    for i, expected_val in enumerate(expected_frames):
        assert np.all(cleaned_data[i] == expected_val)


def test_create_cleaned_data_tzyx(make_napari_viewer):
    """Test creation of cleaned data for TZYX images."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create TZYX test data
    test_data = np.zeros((5, 8, 32, 32), dtype=np.uint8)
    for i in range(5):
        test_data[i] = i

    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Mark frame 1 and 3 for removal
    widget.frames_to_remove = [1, 3]

    cleaned_data = widget._create_cleaned_data()

    # Should have 3 frames remaining
    assert cleaned_data.shape[0] == 3
    assert cleaned_data.shape[1:] == (8, 32, 32)  # Other dimensions unchanged

    # Check correct frames
    assert np.all(cleaned_data[0] == 0)
    assert np.all(cleaned_data[1] == 2)
    assert np.all(cleaned_data[2] == 4)


def test_preview_result(make_napari_viewer):
    """Test preview functionality."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create test data
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Mark some frames
    widget.frames_to_remove = [2, 5, 7]

    # Create preview
    widget._preview_result()

    # Check that preview layer was created
    preview_layer = [
        layer for layer in viewer.layers if "cleaned_preview" in layer.name
    ]
    assert len(preview_layer) == 1
    assert preview_layer[0].data.shape[0] == 7  # 10 - 3 removed


def test_invalid_dimensions(make_napari_viewer, monkeypatch):
    """Test handling of invalid image dimensions."""
    from napari_tmidas._frame_removal import FrameRemovalWidget
    from qtpy.QtWidgets import QMessageBox

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Mock QMessageBox to prevent actual dialog
    warning_called = False

    def mock_warning(*args, **kwargs):
        nonlocal warning_called
        warning_called = True

    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    # Create 2D image (invalid)
    test_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test_2d")

    widget._on_layer_selected(layer)

    assert warning_called
    assert widget.original_data is None


def test_insufficient_frames(make_napari_viewer, monkeypatch):
    """Test handling of images with insufficient frames."""
    from napari_tmidas._frame_removal import FrameRemovalWidget
    from qtpy.QtWidgets import QMessageBox

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Mock QMessageBox
    warning_called = False

    def mock_warning(*args, **kwargs):
        nonlocal warning_called
        warning_called = True

    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    # Create image with only 1 frame
    test_data = np.random.randint(0, 255, (1, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test_single")

    widget._on_layer_selected(layer)

    assert warning_called
    assert widget.original_data is None


@pytest.mark.skipif(
    not os.environ.get("TIFFFILE_AVAILABLE", True),
    reason="tifffile not available",
)
def test_save_result(make_napari_viewer, tmp_path, monkeypatch):
    """Test saving cleaned image."""
    from napari_tmidas._frame_removal import FrameRemovalWidget
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Create test data
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Mark some frames
    widget.frames_to_remove = [2, 5, 7]

    # Mock file dialog to return a test path
    output_path = tmp_path / "test_cleaned.tif"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output_path), ""),
    )

    # Mock QMessageBox to prevent dialog
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    # Save the result
    widget._save_result()

    # Check that file was created
    assert output_path.exists()

    # Verify saved data
    try:
        import tifffile

        saved_data = tifffile.imread(str(output_path))
        assert saved_data.shape[0] == 7  # 10 - 3 removed
        assert saved_data.shape[1:] == (64, 64)
    except ImportError:
        pytest.skip("tifffile not available for verification")


def test_get_image_layers(make_napari_viewer):
    """Test getting image layers from viewer."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Add different layer types
    viewer.add_image(np.random.rand(10, 64, 64), name="image1")
    viewer.add_image(np.random.rand(10, 64, 64), name="image2")
    viewer.add_labels(np.zeros((10, 64, 64), dtype=int), name="labels")

    layers = widget._get_image_layers()

    # Should only return image layers, not labels
    assert len(layers) == 2
    assert all(layer.name in ["image1", "image2"] for layer in layers)


def test_reset_state(make_napari_viewer):
    """Test resetting widget state."""
    from napari_tmidas._frame_removal import FrameRemovalWidget

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Set up some state
    test_data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)
    widget.frames_to_remove = [1, 2, 3]
    widget.current_frame = 5

    # Reset
    widget._reset_state()

    # Check all state is cleared
    assert widget.image_layer is None
    assert widget.original_data is None
    assert widget.frames_to_remove == []
    assert widget.current_frame == 0
    assert widget.is_tzyx is False
    assert not widget.frame_slider.isEnabled()
    assert not widget.mark_checkbox.isEnabled()


def test_prevent_removing_all_frames(make_napari_viewer, monkeypatch):
    """Test that removing all frames is prevented."""
    from napari_tmidas._frame_removal import FrameRemovalWidget
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMessageBox

    viewer = make_napari_viewer()
    widget = FrameRemovalWidget(viewer)

    # Mock QMessageBox
    warning_called = False

    def mock_warning(*args, **kwargs):
        nonlocal warning_called
        warning_called = True

    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    # Create small test data
    test_data = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    layer = viewer.add_image(test_data, name="test")
    widget._on_layer_selected(layer)

    # Mark 2 frames
    widget.frames_to_remove = [0, 1]

    # Try to mark the last frame (frame 2)
    widget.current_frame = 2
    widget._on_mark_changed(Qt.Checked)

    # Should show warning and not add frame 2
    assert warning_called
    assert 2 not in widget.frames_to_remove
    assert len(widget.frames_to_remove) == 2

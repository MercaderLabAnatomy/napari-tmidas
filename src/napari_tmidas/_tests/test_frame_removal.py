"""
Tests for the Frame Removal Widget
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pytestqt")

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


# ---------------------------------------------------------------------------
# Workflow log and replay
# ---------------------------------------------------------------------------
@pytest.fixture
def log_on(monkeypatch):
    monkeypatch.delenv("NAPARI_TMIDAS_NO_WORKFLOW_LOG", raising=False)
    monkeypatch.delenv("NAPARI_TMIDAS_WORKFLOW_LOG", raising=False)


def _save_with_widget(viewer, monkeypatch, output_path, source=None,
                      frames=(2, 5)):
    """Load a (10, 8, 8) movie into the widget, mark ``frames`` and save."""
    import tifffile
    from napari.layers._source import layer_source

    import napari_tmidas._frame_removal as fr

    data = np.arange(10)[:, None, None] * np.ones((1, 8, 8), dtype=np.uint16)
    if source is not None:
        tifffile.imwrite(str(source), data)
        with layer_source(path=str(source)):
            layer = viewer.add_image(data, name="movie")
    else:
        layer = viewer.add_image(data, name="movie")

    widget = fr.FrameRemovalWidget(viewer)
    widget._on_layer_selected(layer)
    widget.frames_to_remove = list(frames)

    monkeypatch.setattr(
        fr.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(output_path), ""),
    )
    monkeypatch.setattr(fr.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(fr.QMessageBox, "critical", lambda *a, **k: None)
    widget._save_result()
    return data


def test_save_records_workflow_run(make_napari_viewer, tmp_path, monkeypatch,
                                   log_on):
    import napari_tmidas._workflow_replay as wr

    ws = tmp_path / "ws"
    ws.mkdir()
    _save_with_widget(
        make_napari_viewer(), monkeypatch, ws / "movie_cleaned.tif",
        source=ws / "movie.tif", frames=(5, 2),
    )

    (run,) = wr.load_runs(str(ws))
    assert run["widget"] == "frame_removal"
    assert run["status"] == "finished"
    assert run["replayable"] is True
    assert run["input_folder"] == str(ws)
    assert run["files"] == ["movie.tif"]
    assert run["params"] == {"frames_removed": [2, 5], "frame_axis": 0}
    assert run["extra"]["layer"] == "movie"
    assert run["extra"]["original_shape"] == [10, 8, 8]
    assert run["extra"]["output_file"] == "movie_cleaned.tif"
    assert run["results"][0]["processed_file"] == "movie_cleaned.tif"
    assert run["results"][0]["original_file"] == "movie.tif"
    assert run["results"][0]["shape"] == [8, 8, 8]


def test_layer_without_source_is_not_replayable(make_napari_viewer, tmp_path,
                                                monkeypatch, log_on):
    import napari_tmidas._workflow_replay as wr

    out = tmp_path / "out"
    out.mkdir()
    _save_with_widget(make_napari_viewer(), monkeypatch, out / "clean.tif")

    (run,) = wr.load_runs(str(out))
    assert run["replayable"] is False
    assert run["files"] == []
    assert run["input_folder"] == str(out)
    assert run["status"] == "finished"
    assert not wr.is_replayable(run)


def test_failed_save_is_recorded(make_napari_viewer, tmp_path, monkeypatch,
                                 log_on):
    import napari_tmidas._workflow_replay as wr

    ws = tmp_path / "ws"
    ws.mkdir()
    _save_with_widget(
        make_napari_viewer(), monkeypatch, ws / "missing" / "clean.tif",
        source=ws / "movie.tif",
    )

    (run,) = wr.load_runs(str(ws))
    assert run["status"] == "failed"
    assert run["errors"][0]["file"] == "movie.tif"
    assert run["errors"][0]["error"]


def test_replay_frame_removal_in_moved_folder(make_napari_viewer, tmp_path,
                                              monkeypatch, log_on):
    import shutil

    import tifffile

    import napari_tmidas._workflow_replay as wr
    from napari_tmidas._frame_removal import replay_frame_removal_run

    ws = tmp_path / "ws"
    ws.mkdir()
    data = _save_with_widget(
        make_napari_viewer(), monkeypatch, ws / "movie_cleaned.tif",
        source=ws / "movie.tif",
    )
    (run,) = wr.load_runs(str(ws))

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(ws / "movie.tif", moved / "movie.tif")
    outcome = replay_frame_removal_run(
        run, input_folder=str(moved), on_progress=lambda m: None
    )

    assert outcome["status"] == "finished"
    assert outcome["errors"] == []
    out = moved / "movie_cleaned.tif"
    np.testing.assert_array_equal(
        tifffile.imread(str(out)), np.delete(data, [2, 5], axis=0)
    )
    (new,) = wr.load_runs(str(moved))
    assert new["extra"]["replayed_from"] == run["run_id"]
    assert new["params"] == run["params"]
    assert new["status"] == "finished"
    assert new["replayable"] is True


def test_replay_frame_removal_dry_run_writes_nothing(make_napari_viewer,
                                                     tmp_path, monkeypatch,
                                                     log_on):
    import napari_tmidas._workflow_log as wl
    import napari_tmidas._workflow_replay as wr
    from napari_tmidas._frame_removal import replay_frame_removal_run

    ws = tmp_path / "ws"
    ws.mkdir()
    _save_with_widget(
        make_napari_viewer(), monkeypatch, ws / "movie_cleaned.tif",
        source=ws / "movie.tif",
    )
    (run,) = wr.load_runs(str(ws))
    (ws / "movie_cleaned.tif").unlink()
    before = (ws / wl.LOG_NAME).read_text()

    outcome = replay_frame_removal_run(
        run, dry_run=True, on_progress=lambda m: None
    )

    assert outcome["dry_run"] is True
    assert outcome["output_file"] == str(ws / "movie_cleaned.tif")
    assert outcome["params"]["frames_removed"] == [2, 5]
    assert not (ws / "movie_cleaned.tif").exists()
    assert (ws / wl.LOG_NAME).read_text() == before


def test_replay_frame_removal_refuses_changed_source(make_napari_viewer,
                                                     tmp_path, monkeypatch,
                                                     log_on):
    import tifffile

    import napari_tmidas._workflow_replay as wr
    from napari_tmidas._frame_removal import replay_frame_removal_run

    ws = tmp_path / "ws"
    ws.mkdir()
    # Saving over the source is allowed; replaying must not cut it again.
    _save_with_widget(
        make_napari_viewer(), monkeypatch, ws / "movie.tif",
        source=ws / "movie.tif",
    )
    (run,) = wr.load_runs(str(ws))
    assert tifffile.imread(str(ws / "movie.tif")).shape[0] == 8

    with pytest.raises(ValueError, match="shape"):
        replay_frame_removal_run(run, on_progress=lambda m: None)

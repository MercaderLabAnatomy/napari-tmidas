"""
Frame Removal Widget for Napari
--------------------------------
This module provides a human-in-the-loop widget for interactively removing frames
from TYX or TZYX time-series images in Napari.

Users can:
- Load a TYX or TZYX image from the Napari viewer
- Navigate through time frames
- Mark frames for removal
- Preview the result
- Save the cleaned time series
"""

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from napari_tmidas._workflow_log import WorkflowLog, relative_path

# Lazy imports for optional heavy dependencies
try:
    from magicgui import magicgui
    from magicgui.widgets import Container, PushButton, create_widget

    _HAS_MAGICGUI = True
except ImportError:
    # Create stub decorator
    def magicgui(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    Container = PushButton = create_widget = None
    _HAS_MAGICGUI = False

try:
    from napari.layers import Image
    from napari.viewer import Viewer

    _HAS_NAPARI = True
except ImportError:
    Image = None
    Viewer = None
    _HAS_NAPARI = False

try:
    from qtpy.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from qtpy.QtCore import Qt

    _HAS_QTPY = True
except ImportError:
    (
        QCheckBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
        Qt,
    ) = (None, None, None, None, None, None, None, None, None, None)
    _HAS_QTPY = False

try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False

if TYPE_CHECKING:
    import napari

WIDGET_NAME = "frame_removal"
FRAME_AXIS = 0


def remove_frames(data, frames, axis: int = FRAME_AXIS):
    """``data`` without the frames whose indices along ``axis`` are in ``frames``."""
    drop = {int(f) for f in frames}
    keep = [i for i in range(data.shape[axis]) if i not in drop]
    return data[(slice(None),) * axis + (keep,)]


def save_cleaned_image(filepath: str, data, is_tzyx: bool) -> None:
    tifffile.imwrite(
        filepath,
        data,
        compression="zlib",
        metadata={"axes": "TZYX" if is_tzyx else "TYX"},
    )


def _record_frame_removal(
    *,
    source: Optional[str],
    output: str,
    frames: List[int],
    is_tzyx: bool,
    original_shape,
    result_shape=None,
    error: Optional[str] = None,
    input_folder: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Write one saved frame removal to the workflow log as a finished (or
    failed) run. Returns ``(run_id, log_path)``; never raises.
    """
    try:
        source = os.fspath(source) if source else None
        output = os.path.abspath(output)
        folder = input_folder or os.path.dirname(os.path.abspath(source or output))
        log = WorkflowLog(folder, os.path.dirname(output))
        run_id = log.run_started(
            widget=WIDGET_NAME,
            params={
                "frames_removed": sorted(int(f) for f in frames),  # 0-based
                "frame_axis": FRAME_AXIS,
            },
            files=[source] if source else [],
            output_folder=os.path.dirname(output),
            output_format="tiff",
            # The replayer reads the source with tifffile.
            replayable=bool(source)
            and source.lower().endswith((".tif", ".tiff")),
            extra={
                "axes": "TZYX" if is_tzyx else "TYX",
                "original_shape": [int(n) for n in original_shape],
                "output_file": relative_path(output, log.input_folder),
                **(extra or {}),
            },
        )
        if error is None:
            result = {"processed_file": output}
            if source:
                result["original_file"] = source
            if result_shape is not None:
                result["shape"] = [int(n) for n in result_shape]
            log.run_finished(run_id, status="finished", results=[result])
        else:
            log.run_finished(
                run_id,
                status="failed",
                errors=[{"file": source or output, "error": error}],
            )
        return run_id, log.path
    except Exception as exc:
        print(f"Workflow log: could not record frame removal: {exc}")
        return None, None


def replay_frame_removal_run(
    run: Dict[str, Any],
    *,
    input_folder: Optional[str] = None,
    output_folder: Optional[str] = None,
    dry_run: bool = False,
    on_progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Repeat a recorded save: remove the same frames from the same source
    file and write the result under the recorded output name. An output
    that was inside the recorded input folder moves with ``input_folder``.
    """
    if run.get("widget") != WIDGET_NAME:
        raise ValueError(f"Run {run.get('run_id')} is not a frame removal run")
    params = run.get("params") or {}
    extra = run.get("extra") or {}
    frames = [int(f) for f in params.get("frames_removed") or []]
    axis = int(params.get("frame_axis", FRAME_AXIS))
    in_folder = os.path.abspath(input_folder or run.get("input_folder") or ".")
    if not run.get("files"):
        raise ValueError(f"Run {run.get('run_id')} has no source file")
    source = os.path.join(in_folder, run["files"][0])
    output = extra.get("output_file")
    if not output:
        raise ValueError(f"Run {run.get('run_id')} has no recorded output file")
    if output_folder:
        output = os.path.join(
            os.path.abspath(output_folder), os.path.basename(output)
        )
    output = os.path.abspath(os.path.join(in_folder, output))

    plan = {
        "run_id": None,
        "replayed_from": run.get("run_id"),
        "widget": WIDGET_NAME,
        "params": {"frames_removed": frames, "frame_axis": axis},
        "input_folder": in_folder,
        "files": [source],
        "output_file": output,
    }
    on_progress(
        f"Replay of {run.get('run_id')}: remove frames {frames} from "
        f"{source} -> {output}"
    )
    if dry_run:
        return {**plan, "dry_run": True}
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    data = tifffile.imread(source)
    expected = extra.get("original_shape")
    if expected and list(data.shape) != list(expected):
        raise ValueError(
            f"{source} has shape {data.shape}, but the recorded run removed "
            f"frames from shape {tuple(expected)}"
        )
    is_tzyx = extra.get("axes", "TZYX" if data.ndim == 4 else "TYX") == "TZYX"
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    cleaned = None
    try:
        cleaned = remove_frames(data, frames, axis)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        save_cleaned_image(output, cleaned, is_tzyx)
        results.append({"original_file": source, "processed_file": output})
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append({"file": source, "error": str(exc)})
        on_progress(f"  error: {exc}")
    run_id, log_path = _record_frame_removal(
        source=source,
        output=output,
        frames=frames,
        is_tzyx=is_tzyx,
        original_shape=data.shape,
        result_shape=None if cleaned is None else cleaned.shape,
        error=errors[0]["error"] if errors else None,
        input_folder=in_folder,
        extra={"replayed_from": run.get("run_id")},
    )
    status = "failed" if errors else "finished"
    on_progress(f"Replay {status}: {output}")
    return {
        **plan,
        "run_id": run_id,
        "status": status,
        "results": results,
        "errors": errors,
        "log_path": log_path,
    }


class FrameRemovalWidget(QWidget):
    """
    Interactive widget for removing frames from TYX or TZYX time-series images.
    """

    def __init__(self, viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.image_layer: Optional[Image] = None
        self.original_data: Optional[np.ndarray] = None
        self.frames_to_remove: List[int] = []
        self.current_frame: int = 0
        self.is_tzyx: bool = False

        self._init_ui()
        
        # Connect to viewer events for automatic layer list updates
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("<h2>Frame Removal Tool</h2>")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Select a TYX or TZYX image layer, navigate frames, "
            "and mark frames for removal."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Image layer selector
        layer_layout = QHBoxLayout()
        layer_label = QLabel("Image Layer:")
        layer_layout.addWidget(layer_label)

        self.layer_selector = create_widget(
            annotation=Image, label="", options={"choices": self._get_image_layers}
        )
        self.layer_selector.changed.connect(self._on_layer_selected)
        layer_layout.addWidget(self.layer_selector.native)

        layout.addLayout(layer_layout)

        # Info display
        self.info_label = QLabel("No image selected")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Frame navigation
        nav_layout = QVBoxLayout()

        # Current frame display
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.frame_label)

        # Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self.frame_slider)

        # Navigation buttons
        nav_buttons_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._prev_frame)
        nav_buttons_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._next_frame)
        nav_buttons_layout.addWidget(self.next_btn)

        nav_layout.addLayout(nav_buttons_layout)
        layout.addLayout(nav_layout)

        # Mark frame checkbox
        self.mark_checkbox = QCheckBox("Mark current frame for REMOVAL")
        self.mark_checkbox.setEnabled(False)
        self.mark_checkbox.setStyleSheet("QCheckBox { color: red; font-weight: bold; }")
        self.mark_checkbox.stateChanged.connect(self._on_mark_changed)
        layout.addWidget(self.mark_checkbox)

        # Marked frames display
        self.marked_label = QLabel("Marked frames: None")
        self.marked_label.setWordWrap(True)
        layout.addWidget(self.marked_label)

        # Action buttons
        action_layout = QVBoxLayout()

        self.clear_marks_btn = QPushButton("Clear All Marks (Unmark Frames)")
        self.clear_marks_btn.setEnabled(False)
        self.clear_marks_btn.setToolTip("Remove all marks and start over - does NOT create cleaned image")
        self.clear_marks_btn.clicked.connect(self._clear_marks)
        action_layout.addWidget(self.clear_marks_btn)

        self.preview_btn = QPushButton("1. Preview Result (Create New Layer)")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip("Create a new layer showing the image with marked frames removed")
        self.preview_btn.clicked.connect(self._preview_result)
        action_layout.addWidget(self.preview_btn)

        self.save_btn = QPushButton("2. Save Cleaned Image")
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Save the cleaned image with marked frames removed to a file")
        self.save_btn.clicked.connect(self._save_result)
        action_layout.addWidget(self.save_btn)

        layout.addLayout(action_layout)

        # Add stretch at the end
        layout.addStretch()

    def _get_image_layers(self, widget=None):
        """Get list of image layers from viewer."""
        return [layer for layer in self.viewer.layers if isinstance(layer, Image)]

    def _on_layers_changed(self, event):
        """Handle layer list changes (added/removed)."""
        # Reset choices to update the dropdown
        self.layer_selector.reset_choices()
        
        # Auto-select the most recently added image layer if none selected
        if self.image_layer is None:
            image_layers = self._get_image_layers()
            if image_layers:
                # Select the last added image layer
                self.layer_selector.value = image_layers[-1]

    def _on_layer_selected(self, layer: Optional[Image]):
        """Handle layer selection."""
        if layer is None:
            self._reset_state()
            return

        # Remove any existing preview layers from previous selections
        self._remove_preview_layers()

        # Store the layer and data
        self.image_layer = layer
        self.original_data = layer.data

        # Validate dimensions
        ndim = self.original_data.ndim
        shape = self.original_data.shape

        if ndim == 3:
            # TYX format
            self.is_tzyx = False
            num_frames = shape[0]
            info = f"Format: TYX\nShape: {shape}\nFrames: {num_frames}"
        elif ndim == 4:
            # TZYX format
            self.is_tzyx = True
            num_frames = shape[0]
            info = f"Format: TZYX\nShape: {shape}\nFrames: {num_frames} (T dimension)"
        else:
            QMessageBox.warning(
                self,
                "Invalid Dimensions",
                f"Selected layer has {ndim} dimensions. "
                "This tool only supports TYX (3D) or TZYX (4D) images.",
            )
            self._reset_state()
            return

        # Check if we have enough frames
        if num_frames < 2:
            QMessageBox.warning(
                self,
                "Insufficient Frames",
                f"Image has only {num_frames} frame(s). "
                "At least 2 frames are needed for removal.",
            )
            self._reset_state()
            return

        # Update UI
        self.info_label.setText(info)
        self.frames_to_remove = []
        self.current_frame = 0

        # Enable controls
        self.frame_slider.setEnabled(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(num_frames - 1)
        self.frame_slider.setValue(0)

        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.mark_checkbox.setEnabled(True)
        self.clear_marks_btn.setEnabled(True)

        self._update_display()

    def _reset_state(self):
        """Reset widget to initial state."""
        self.image_layer = None
        self.original_data = None
        self.frames_to_remove = []
        self.current_frame = 0
        self.is_tzyx = False

        self.info_label.setText("No image selected")
        self.frame_label.setText("Frame: 0 / 0")
        self.marked_label.setText("Marked frames: None")

        self.frame_slider.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.mark_checkbox.setEnabled(False)
        self.mark_checkbox.setChecked(False)
        self.clear_marks_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        if self.original_data is None:
            return

        self.current_frame = value
        self._update_display()

    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)

    def _next_frame(self):
        """Go to next frame."""
        if self.original_data is not None:
            num_frames = self.original_data.shape[0]
            if self.current_frame < num_frames - 1:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)

    def _update_display(self):
        """Update the display for the current frame."""
        if self.original_data is None or self.image_layer is None:
            return

        num_frames = self.original_data.shape[0]

        # Update frame label
        self.frame_label.setText(f"Frame: {self.current_frame + 1} / {num_frames}")

        # Update checkbox state
        self.mark_checkbox.blockSignals(True)
        self.mark_checkbox.setChecked(self.current_frame in self.frames_to_remove)
        self.mark_checkbox.blockSignals(False)

        # Update marked frames display
        if self.frames_to_remove:
            marked_str = ", ".join([str(f + 1) for f in sorted(self.frames_to_remove)])
            self.marked_label.setText(
                f"Marked frames ({len(self.frames_to_remove)}): {marked_str}"
            )
            self.preview_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
        else:
            self.marked_label.setText("Marked frames: None")
            self.preview_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

        # Navigate to the frame in the viewer
        if self.is_tzyx:
            # For TZYX, set the time dimension slider
            self.viewer.dims.set_point(0, self.current_frame)
        else:
            # For TYX, set the first dimension slider
            self.viewer.dims.set_point(0, self.current_frame)

    def _on_mark_changed(self, state: int):
        """Handle mark checkbox state change."""
        if self.original_data is None:
            return

        num_frames = self.original_data.shape[0]

        # stateChanged delivers a plain int, but under PyQt6 Qt.Checked is
        # a Python enum that never compares equal to one.  Normalise first.
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            if self.current_frame not in self.frames_to_remove:
                self.frames_to_remove.append(self.current_frame)
        else:
            if self.current_frame in self.frames_to_remove:
                self.frames_to_remove.remove(self.current_frame)

        # Check if we're marking all frames
        if len(self.frames_to_remove) >= num_frames:
            QMessageBox.warning(
                self,
                "Cannot Remove All Frames",
                "You cannot mark all frames for removal. "
                "At least one frame must remain.",
            )
            self.frames_to_remove.remove(self.current_frame)
            self.mark_checkbox.blockSignals(True)
            self.mark_checkbox.setChecked(False)
            self.mark_checkbox.blockSignals(False)

        self._update_display()

    def _clear_marks(self):
        """Clear all marked frames."""
        if not self.frames_to_remove:
            QMessageBox.information(
                self,
                "No Marks",
                "There are no marked frames to clear.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Clear Marks",
            "Are you sure you want to clear all marked frames?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.frames_to_remove = []
            self._update_display()
            
            # Remove any preview layers
            original_name = self.image_layer.name
            preview_name = f"{original_name}_cleaned_preview"
            existing_preview = [
                layer for layer in self.viewer.layers if layer.name == preview_name
            ]
            for layer in existing_preview:
                self.viewer.layers.remove(layer)
            
            self.viewer.status = "All marked frames cleared"

    def _remove_preview_layers(self):
        """Remove all preview layers from the viewer."""
        # Remove any layers with "_cleaned_preview" in the name
        preview_layers = [
            layer for layer in self.viewer.layers 
            if "_cleaned_preview" in layer.name
        ]
        for layer in preview_layers:
            self.viewer.layers.remove(layer)

    def _create_cleaned_data(self) -> np.ndarray:
        """Create the cleaned data by removing marked frames."""
        if self.original_data is None or not self.frames_to_remove:
            return self.original_data

        return remove_frames(self.original_data, self.frames_to_remove)

    def _preview_result(self):
        """Preview the result by creating a new layer."""
        if self.original_data is None or not self.frames_to_remove:
            return

        # Remove any existing preview layers first
        self._remove_preview_layers()

        # Create cleaned data
        cleaned_data = self._create_cleaned_data()

        # Create a preview layer name
        original_name = self.image_layer.name
        preview_name = f"{original_name}_cleaned_preview"

        # Add the preview layer
        self.viewer.add_image(
            cleaned_data,
            name=preview_name,
            colormap=self.image_layer.colormap.name,
            contrast_limits=self.image_layer.contrast_limits,
        )

        self.viewer.status = (
            f"Preview created: removed {len(self.frames_to_remove)} frame(s). "
            f"New shape: {cleaned_data.shape}"
        )

    def _save_result(self):
        """Save the cleaned image to a file."""
        if self.original_data is None or not self.frames_to_remove:
            return

        if not _HAS_TIFFFILE:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "tifffile package is required for saving. "
                "Please install it: pip install tifffile",
            )
            return

        # Get save path from user
        default_name = f"{self.image_layer.name}_cleaned.tif"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cleaned Image",
            default_name,
            "TIFF Files (*.tif *.tiff);;All Files (*)",
        )

        if not filepath:
            return

        try:
            # Create cleaned data
            cleaned_data = self._create_cleaned_data()

            save_cleaned_image(filepath, cleaned_data, self.is_tzyx)
            self._record_save(filepath, result_shape=cleaned_data.shape)

            QMessageBox.information(
                self,
                "Save Successful",
                f"Cleaned image saved successfully!\n\n"
                f"File: {os.path.basename(filepath)}\n"
                f"Original frames: {self.original_data.shape[0]}\n"
                f"Removed frames: {len(self.frames_to_remove)}\n"
                f"Remaining frames: {cleaned_data.shape[0]}\n"
                f"Shape: {cleaned_data.shape}",
            )

            self.viewer.status = f"Saved cleaned image to {filepath}"

        except (OSError, ValueError, RuntimeError) as e:
            self._record_save(filepath, error=str(e))
            QMessageBox.critical(
                self, "Save Failed", f"Failed to save image:\n{str(e)}"
            )

    def _record_save(self, filepath: str, result_shape=None, error=None):
        """Record one save in the workflow log; never raises."""
        source = getattr(getattr(self.image_layer, "source", None), "path", None)
        _record_frame_removal(
            source=source,
            output=filepath,
            frames=self.frames_to_remove,
            is_tzyx=self.is_tzyx,
            original_shape=self.original_data.shape,
            result_shape=result_shape,
            error=error,
            extra={"layer": getattr(self.image_layer, "name", None)},
        )


@magicgui(call_button="Open Frame Removal Tool")
def frame_removal_tool(viewer: "napari.Viewer"):
    """
    MagicGUI widget for starting frame removal.
    
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    """
    # Create the widget
    widget = FrameRemovalWidget(viewer)
    
    # Add it as a dock widget
    viewer.window.add_dock_widget(widget, name="Frame Removal", area="right")


def frame_removal_widget():
    """
    Provide the frame removal widget to Napari.
    
    Returns
    -------
    magicgui widget
        The frame removal tool widget
    """
    return frame_removal_tool

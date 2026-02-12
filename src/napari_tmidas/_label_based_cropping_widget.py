"""
Label-Based Image Cropping Widget for Napari

This widget provides an interactive interface for cropping images using user-drawn labels.
It supports:
- 2D label drawing in 2D mode
- Automatic expansion of 2D labels to 3D/4D images
- Preview of cropped results
- Batch and single-image processing
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

# Lazy imports for optional heavy dependencies
try:
    from magicgui import magic_factory, magicgui
    from magicgui.widgets import Container, create_widget

    _HAS_MAGICGUI = True
except ImportError:
    # Create stub decorator and stubs
    def magic_factory(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    def magicgui(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    class Container:
        pass

    create_widget = None
    _HAS_MAGICGUI = False

try:
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _HAS_QTPY = True
except ImportError:
    Qt = QThread = Signal = None
    QCheckBox = QComboBox = QFormLayout = QHBoxLayout = QLabel = QLineEdit = None
    QMessageBox = QPushButton = QSpinBox = QTextEdit = QVBoxLayout = QWidget = None
    _HAS_QTPY = False

try:
    import napari
    from napari.layers import Image, Labels

    _HAS_NAPARI = True
except ImportError:
    napari = None
    Image = None
    Labels = None
    _HAS_NAPARI = False

try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False

if TYPE_CHECKING:
    import napari

from napari_tmidas.processing_functions.label_based_cropping import (
    _crop_image_with_label,
    _expand_label_to_3d,
    _expand_label_to_time,
    _infer_axes,
    label_based_cropping,
)


if not _HAS_QTPY:

    class QWidget:
        pass

    class QThread:
        def start(self):
            pass

    QVBoxLayout = QHBoxLayout = QFormLayout = QLineEdit = None
    QPushButton = QLabel = QCheckBox = QComboBox = QSpinBox = QTextEdit = (
        QMessageBox
    ) = None

    def Signal(*args):
        return None


class LabelBasedCroppingWorker(QThread if _HAS_QTPY else object):
    """Worker thread for processing images in the background"""

    if _HAS_QTPY:
        progress = Signal(str)
        finished = Signal(bool, str, object)  # success, message, cropped_data

    def __init__(
        self,
        image_data: np.ndarray,
        label_data: np.ndarray,
    ):
        super().__init__()
        self.image_data = image_data
        self.label_data = label_data

    def run(self):
        """Run the cropping operation"""
        try:
            image = self.image_data
            label = self.label_data

            self._emit_progress("Cropping image...")

            # Verify shapes match
            if image.shape != label.shape:
                raise ValueError(
                    f"Image shape {image.shape} does not match label shape {label.shape}. "
                    f"Please use the expansion checkboxes to expand labels first."
                )

            # Perform cropping
            cropped = _crop_image_with_label(image, label)

            self._emit_progress("Cropping completed!")

            # Return result via signal
            self._emit_finished(True, "Cropping completed successfully!", cropped)

        except Exception as e:
            error_msg = f"Error during cropping: {str(e)}"
            self._emit_finished(False, error_msg, None)

    def _emit_progress(self, msg: str):
        """Safely emit progress signal"""
        if _HAS_QTPY:
            try:
                self.progress.emit(msg)
            except Exception:
                pass

    def _emit_finished(self, success: bool, msg: str, cropped_data):
        """Safely emit finished signal"""
        if _HAS_QTPY:
            try:
                self.finished.emit(success, msg, cropped_data)
            except Exception:
                pass


class LabelBasedCroppingWidget(QWidget if _HAS_QTPY else object):
    """Widget for interactive label-based image cropping"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._worker = None

        if not _HAS_QTPY or not _HAS_NAPARI:
            return

        # Create main layout
        layout = QVBoxLayout()

        # Create form layout for layer selection
        form_layout = QFormLayout()

        # Image layer selector
        self._image_layer_combo = QComboBox()
        self._update_image_layers()
        form_layout.addRow(QLabel("Image Layer:"), self._image_layer_combo)

        # Label layer selector
        self._label_layer_combo = QComboBox()
        self._update_label_layers()
        form_layout.addRow(QLabel("Label Layer:"), self._label_layer_combo)

        # Output name input
        self._crop_name_input = QLineEdit("cropped")
        form_layout.addRow(QLabel("Output Name:"), self._crop_name_input)

        layout.addLayout(form_layout)

        # Auto-expand checkboxes
        self._expand_z_checkbox = QCheckBox("Expand labels across Z")
        self._expand_z_checkbox.setChecked(False)
        self._expand_z_checkbox.stateChanged.connect(self._on_expand_z_changed)
        layout.addWidget(self._expand_z_checkbox)

        self._expand_time_checkbox = QCheckBox(
            "Expand a single 2D label across all time frames (T)"
        )
        self._expand_time_checkbox.setChecked(False)
        self._expand_time_checkbox.stateChanged.connect(self._on_expand_time_changed)
        layout.addWidget(self._expand_time_checkbox)

        # Info text box
        layout.addWidget(QLabel("Status:"))
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(100)
        layout.addWidget(self._info_text)

        # Crop button
        self._crop_button = QPushButton("Crop Image")
        self._crop_button.clicked.connect(self._on_crop_clicked)
        layout.addWidget(self._crop_button)

        # Add stretch
        layout.addStretch()

        self.setLayout(layout)

        # Connect viewer events to update layer lists
        self._viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_layers_changed)

    def _on_layers_changed(self, event=None):
        """Handle when layers are added or removed"""
        self._update_image_layers()
        self._update_label_layers()

    def _update_image_layers(self):
        """Update the image layer combo box"""
        self._image_layer_combo.clear()
        for i, layer in enumerate(self._viewer.layers):
            if isinstance(layer, Image):
                self._image_layer_combo.addItem(layer.name, i)

    def _update_label_layers(self):
        """Update the label layer combo box"""
        self._label_layer_combo.clear()
        for i, layer in enumerate(self._viewer.layers):
            if isinstance(layer, Labels):
                self._label_layer_combo.addItem(layer.name, i)

    def _update_info(self, message: str):
        """Update info text"""
        current = self._info_text.toPlainText()
        if current:
            self._info_text.setText(current + "\n" + message)
        else:
            self._info_text.setText(message)
        # Auto-scroll to bottom
        scrollbar = self._info_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_expand_z_changed(self, state):
        """Handle expand Z checkbox change"""
        if state == 0:  # Unchecked
            return

        # Get selected layers
        image_idx = self._image_layer_combo.currentData()
        label_idx = self._label_layer_combo.currentData()

        if image_idx is None or label_idx is None:
            self._update_info("⚠ Please select both image and label layers first")
            self._expand_z_checkbox.setChecked(False)
            return

        image_layer = self._viewer.layers[image_idx]
        label_layer = self._viewer.layers[label_idx]

        try:
            image_data = image_layer.data
            label_data = label_layer.data

            # Get current step/slice position from viewer
            current_step = list(self._viewer.dims.current_step)

            # Handle different dimensionality cases
            if image_data.ndim == 3:  # (Z, Y, X)
                if label_data.ndim == 2:
                    # Simple 2D label -> expand to 3D
                    expanded = _expand_label_to_3d(label_data, image_data.shape[0])
                    label_layer.data = expanded
                    self._update_info(
                        f"✓ Expanded 2D label to 3D: {expanded.shape}"
                    )
                elif label_data.ndim == 3:
                    # Get current z-slice
                    z_idx = int(current_step[0]) if len(current_step) > 0 else 0
                    current_slice = label_data[z_idx]
                    # Expand current slice to all z
                    expanded = _expand_label_to_3d(current_slice, image_data.shape[0])
                    label_layer.data = expanded
                    self._update_info(
                        f"✓ Copied slice {z_idx} to all Z slices: {expanded.shape}"
                    )
                else:
                    self._update_info("⚠ Unexpected label dimensionality")
                    self._expand_z_checkbox.setChecked(False)

            elif image_data.ndim == 4:  # (T, Z, Y, X)
                if label_data.ndim == 2:
                    # Single 2D label -> expand to current frame's Z slices
                    expanded = _expand_label_to_3d(label_data, image_data.shape[1])
                    # Optionally expand across T
                    if self._expand_time_checkbox.isChecked():
                        expanded = np.repeat(
                            expanded[np.newaxis, :, :, :], image_data.shape[0], axis=0
                        )
                    label_layer.data = expanded
                    self._update_info(
                        f"✓ Expanded 2D label: {expanded.shape}"
                    )
                elif label_data.ndim == 3:
                    # Check if it's (T, Y, X) per-frame labels
                    if label_data.shape[0] == image_data.shape[0]:
                        # Per-frame 2D labels -> expand each frame's Z
                        expanded = np.repeat(
                            label_data[:, np.newaxis, :, :], image_data.shape[1], axis=1
                        )
                        label_layer.data = expanded
                        self._update_info(
                            f"✓ Expanded per-frame labels across Z: {expanded.shape}"
                        )
                    else:
                        # Get current slice and expand
                        z_idx = int(current_step[1]) if len(current_step) > 1 else 0
                        current_slice = label_data[z_idx]
                        expanded = _expand_label_to_3d(current_slice, image_data.shape[1])
                        label_layer.data = expanded
                        self._update_info(
                            f"✓ Copied slice {z_idx} across Z: {expanded.shape}"
                        )
                elif label_data.ndim == 4:
                    # Get current frame and z-slice
                    t_idx = int(current_step[0]) if len(current_step) > 0 else 0
                    z_idx = int(current_step[1]) if len(current_step) > 1 else 0
                    current_slice = label_data[t_idx, z_idx]
                    # Expand within current frame
                    expanded_frame = _expand_label_to_3d(current_slice, image_data.shape[1])
                    # Update only current frame
                    new_data = label_data.copy()
                    new_data[t_idx] = expanded_frame
                    label_layer.data = new_data
                    self._update_info(
                        f"✓ Copied frame {t_idx} slice {z_idx} to all Z in frame {t_idx}"
                    )
                else:
                    self._update_info("⚠ Unexpected label dimensionality")
                    self._expand_z_checkbox.setChecked(False)
            else:
                self._update_info("ℹ Image must be 3D or 4D for Z expansion")
                self._expand_z_checkbox.setChecked(False)

        except Exception as e:
            self._update_info(f"❌ Error expanding labels: {str(e)}")
            self._expand_z_checkbox.setChecked(False)

    def _on_expand_time_changed(self, state):
        """Handle expand time checkbox change"""
        if state == 0:  # Unchecked
            return

        # Get selected layers
        image_idx = self._image_layer_combo.currentData()
        label_idx = self._label_layer_combo.currentData()

        if image_idx is None or label_idx is None:
            self._update_info("⚠ Please select both image and label layers first")
            self._expand_time_checkbox.setChecked(False)
            return

        image_layer = self._viewer.layers[image_idx]
        label_layer = self._viewer.layers[label_idx]

        try:
            image_data = image_layer.data
            label_data = label_layer.data

            # Only applies to 4D images
            if image_data.ndim != 4:
                self._update_info("⚠ Time expansion only applies to 4D images")
                self._expand_time_checkbox.setChecked(False)
                return

            # Get current step/slice position from viewer
            current_step = list(self._viewer.dims.current_step)
            t_idx = int(current_step[0]) if len(current_step) > 0 else 0

            # Handle different label dimensionalities
            if label_data.ndim == 2:
                # Single 2D label -> expand to 3D first, then across time
                if not self._expand_z_checkbox.isChecked():
                    self._update_info(
                        "⚠ For 4D images, enable 'Expand across Z' before expanding time"
                    )
                    self._expand_time_checkbox.setChecked(False)
                    return
                expanded = _expand_label_to_time(
                    label_data, image_data.shape[0], image_data.shape[1]
                )
                label_layer.data = expanded
                self._update_info(
                    f"✓ Copied 2D label to all {image_data.shape[0]} time frames"
                )
            elif label_data.ndim == 3:
                # Could be (T, Y, X) or (Z, Y, X)
                if label_data.shape[0] == image_data.shape[0]:
                    # Already per-frame (T, Y, X) - copy current frame to all frames
                    current_frame = label_data[t_idx]
                    expanded = np.repeat(
                        current_frame[np.newaxis, :, :], image_data.shape[0], axis=0
                    )
                    label_layer.data = expanded
                    self._update_info(
                        f"✓ Copied frame {t_idx} to all {image_data.shape[0]} time frames"
                    )
                else:
                    # Single 3D (Z, Y, X) label - expand across time
                    expanded = np.repeat(
                        label_data[np.newaxis, :, :, :], image_data.shape[0], axis=0
                    )
                    label_layer.data = expanded
                    self._update_info(
                        f"✓ Copied 3D label to all {image_data.shape[0]} time frames"
                    )
            elif label_data.ndim == 4:
                # Already 4D (T, Z, Y, X) - copy current frame to all frames
                current_frame = label_data[t_idx]
                expanded = np.repeat(
                    current_frame[np.newaxis, :, :, :], image_data.shape[0], axis=0
                )
                label_layer.data = expanded
                self._update_info(
                    f"✓ Copied frame {t_idx} to all {image_data.shape[0]} time frames"
                )
            else:
                self._update_info("⚠ Unexpected label dimensionality")
                self._expand_time_checkbox.setChecked(False)

        except Exception as e:
            self._update_info(f"❌ Error expanding labels: {str(e)}")
            self._expand_time_checkbox.setChecked(False)

    def _on_crop_clicked(self):
        """Handle crop button click"""
        # Get selected layer indices
        image_idx = self._image_layer_combo.currentData()
        label_idx = self._label_layer_combo.currentData()

        if image_idx is None:
            QMessageBox.warning(self, "Warning", "Please select an image layer")
            return

        if label_idx is None:
            QMessageBox.warning(self, "Warning", "Please select a label layer")
            return

        image_layer = self._viewer.layers[image_idx]
        label_layer = self._viewer.layers[label_idx]

        if not isinstance(image_layer, Image):
            QMessageBox.warning(
                self, "Warning", "Selected image layer is not an Image layer"
            )
            return

        if not isinstance(label_layer, Labels):
            QMessageBox.warning(
                self, "Warning", "Selected label layer is not a Labels layer"
            )
            return

        self._crop_name = self._crop_name_input.text() or "cropped"

        # Store source layer for adding result back (in main thread)
        self._source_layer = image_layer

        # Disable button during processing
        self._crop_button.setEnabled(False)
        self._update_info("Starting cropping operation...")

        # Create and start worker with data arrays (not layer objects)
        self._worker = LabelBasedCroppingWorker(
            image_layer.data,
            label_layer.data,
        )
        self._worker.progress.connect(self._update_info)
        self._worker.finished.connect(self._on_crop_finished)
        self._worker.start()

    def _on_crop_finished(self, success: bool, message: str, cropped_data):
        """Handle worker completion"""
        self._update_info(message)
        self._crop_button.setEnabled(True)

        if success and cropped_data is not None:
            # Add result to viewer (safe - we're in main thread)
            if self._crop_name in self._viewer.layers:
                self._viewer.layers[self._crop_name].data = cropped_data
            else:
                self._viewer.add_image(
                    cropped_data,
                    name=self._crop_name,
                    colormap=self._source_layer.colormap,
                    blending=self._source_layer.blending,
                )
            self._update_info(f"Added result as '{self._crop_name}'")
        elif not success:
            QMessageBox.critical(self, "Error", message)


@magicgui(call_button="Open Label-Based Cropping", layout="vertical")
def label_based_cropping_widget(viewer: napari.Viewer):
    """Open the label-based image cropping widget"""
    widget = LabelBasedCroppingWidget(viewer)
    viewer.window.add_dock_widget(
        widget, name="Label-Based Cropping", area="right"
    )
    return widget


def napari_experimental_provide_dock_widget():
    """Provide the label-based cropping widget to Napari"""
    return label_based_cropping_widget

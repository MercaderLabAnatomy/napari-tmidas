"""
Batch Crop Anything - A Napari plugin for interactive image cropping

This plugin combines Segment Anything Model (SAM) for automatic object detection with
an interactive interface for selecting and cropping objects from images.
"""

import os

import numpy as np
import torch
from magicgui import magicgui
from napari.layers import Labels
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread
from skimage.transform import resize  # Added import for resize function
from tifffile import imwrite


class BatchCropAnything:
    """
    Class for processing images with Segment Anything and cropping selected objects.
    """

    def __init__(self, viewer: Viewer):
        """Initialize the BatchCropAnything processor."""
        # Core components
        self.viewer = viewer
        self.images = []
        self.current_index = 0

        # Image and segmentation data
        self.original_image = None
        self.segmentation_result = None
        self.current_image_for_segmentation = None
        self.current_scale_factor = 1.0  # Added scale factor tracking

        # UI references
        self.image_layer = None
        self.label_layer = None
        self.label_table_widget = None

        # State tracking
        self.selected_labels = set()
        self.label_info = {}

        # Segmentation parameters
        self.sensitivity = 50  # Default sensitivity (0-100 scale)

        # Initialize the SAM model
        self._initialize_sam()

    # --------------------------------------------------
    # Model Initialization
    # --------------------------------------------------

    def _initialize_sam(self):
        """Initialize the Segment Anything Model."""
        try:
            # Import required modules
            from mobile_sam import (
                SamAutomaticMaskGenerator,
                sam_model_registry,
            )

            # Setup device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_type = "vit_t"

            # Find the model weights file
            checkpoint_path = self._find_sam_checkpoint()
            if checkpoint_path is None:
                self.mobile_sam = None
                self.mask_generator = None
                return

            # Initialize the model
            self.mobile_sam = sam_model_registry[model_type](
                checkpoint=checkpoint_path
            )
            self.mobile_sam.to(device=self.device)
            self.mobile_sam.eval()

            # Create mask generator with default parameters
            self.mask_generator = SamAutomaticMaskGenerator(self.mobile_sam)
            self.viewer.status = f"Initialized SAM model from {checkpoint_path} on {self.device}"

        except (ImportError, Exception) as e:
            self.viewer.status = f"Error initializing SAM: {str(e)}"
            self.mobile_sam = None
            self.mask_generator = None

    def _find_sam_checkpoint(self):
        """Find the SAM model checkpoint file."""
        try:
            import importlib.util

            # Find the mobile_sam package location
            mobile_sam_spec = importlib.util.find_spec("mobile_sam")
            if mobile_sam_spec is None:
                raise ImportError("mobile_sam package not found")

            mobile_sam_path = os.path.dirname(mobile_sam_spec.origin)

            # Check common locations for the model file
            checkpoint_paths = [
                os.path.join(mobile_sam_path, "weights", "mobile_sam.pt"),
                os.path.join(mobile_sam_path, "mobile_sam.pt"),
                os.path.join(
                    os.path.dirname(mobile_sam_path),
                    "weights",
                    "mobile_sam.pt",
                ),
                os.path.join(
                    os.path.expanduser("~"), "models", "mobile_sam.pt"
                ),
                "/opt/T-MIDAS/models/mobile_sam.pt",
                os.path.join(os.getcwd(), "mobile_sam.pt"),
            ]

            for path in checkpoint_paths:
                if os.path.exists(path):
                    return path

            # If model not found, ask user
            QMessageBox.information(
                None,
                "Model Not Found",
                "Mobile-SAM model weights not found. Please select the mobile_sam.pt file.",
            )

            checkpoint_path, _ = QFileDialog.getOpenFileName(
                None, "Select Mobile-SAM model file", "", "Model Files (*.pt)"
            )

            return checkpoint_path if checkpoint_path else None

        except (ImportError, Exception) as e:
            self.viewer.status = f"Error finding SAM checkpoint: {str(e)}"
            return None

    # --------------------------------------------------
    # Image Loading and Navigation
    # --------------------------------------------------

    def load_images(self, folder_path: str):
        """Load images from the specified folder path."""
        if not os.path.exists(folder_path):
            self.viewer.status = f"Folder not found: {folder_path}"
            return

        files = os.listdir(folder_path)
        self.images = [
            os.path.join(folder_path, file)
            for file in files
            if file.lower().endswith(
                (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            )
            and not file.endswith(("_labels.tif", "_cropped.tif", "_cropped_"))
        ]

        if not self.images:
            self.viewer.status = "No compatible images found in the folder."
            return

        self.viewer.status = f"Found {len(self.images)} images."
        self.current_index = 0
        self._load_current_image()

    def next_image(self):
        """Move to the next image."""
        if not self.images:
            self.viewer.status = "No images to process."
            return False

        # Check if we're already at the last image
        if self.current_index >= len(self.images) - 1:
            self.viewer.status = "No more images. Processing complete."
            return False

        # Move to the next image
        self.current_index += 1

        # Clear selected labels
        self.selected_labels = set()

        # Clear the table reference (will be recreated)
        self.label_table_widget = None

        # Load the next image
        self._load_current_image()
        return True

    def previous_image(self):
        """Move to the previous image."""
        if not self.images:
            self.viewer.status = "No images to process."
            return False

        # Check if we're already at the first image
        if self.current_index <= 0:
            self.viewer.status = "Already at the first image."
            return False

        # Move to the previous image
        self.current_index -= 1

        # Clear selected labels
        self.selected_labels = set()

        # Clear the table reference (will be recreated)
        self.label_table_widget = None

        # Load the previous image
        self._load_current_image()
        return True

    def _load_current_image(self):
        """Load the current image and generate segmentation."""
        if not self.images:
            self.viewer.status = "No images to process."
            return

        if self.mobile_sam is None or self.mask_generator is None:
            self.viewer.status = (
                "SAM model not initialized. Cannot segment images."
            )
            return

        image_path = self.images[self.current_index]
        self.viewer.status = f"Processing {os.path.basename(image_path)}"

        try:
            # Clear existing layers
            self.viewer.layers.clear()

            # Load and process image
            self.original_image = imread(image_path)

            # Ensure image is 8-bit for SAM display (keeping original for saving)
            if self.original_image.dtype != np.uint8:
                image_for_display = (
                    self.original_image / np.amax(self.original_image) * 255
                ).astype(np.uint8)
            else:
                image_for_display = self.original_image

            # Add image to viewer
            self.image_layer = self.viewer.add_image(
                image_for_display,
                name=f"Image ({os.path.basename(image_path)})",
            )

            # Generate segmentation
            self._generate_segmentation(image_for_display)

        except (Exception, ValueError) as e:
            import traceback

            self.viewer.status = f"Error processing image: {str(e)}"
            traceback.print_exc()
            # Create empty segmentation in case of error
            if (
                hasattr(self, "original_image")
                and self.original_image is not None
            ):
                self.segmentation_result = np.zeros(
                    self.original_image.shape[:2], dtype=np.uint32
                )
                self.label_layer = self.viewer.add_labels(
                    self.segmentation_result, name="Error: No Segmentation"
                )

    # --------------------------------------------------
    # Segmentation Generation and Control
    # --------------------------------------------------

    def _generate_segmentation(self, image):
        """Generate segmentation for the current image."""
        # Prepare for SAM (add color channel if needed)
        if len(image.shape) == 2:
            image_for_sam = image[:, :, np.newaxis].repeat(3, axis=2)
        else:
            image_for_sam = image

        # Store the current image for later regeneration if sensitivity changes
        self.current_image_for_segmentation = image_for_sam

        # Generate segmentation with current sensitivity
        self.generate_segmentation_with_sensitivity()

    def generate_segmentation_with_sensitivity(self, sensitivity=None):
        """Generate segmentation with the specified sensitivity."""
        if sensitivity is not None:
            self.sensitivity = sensitivity

        if self.mobile_sam is None or self.mask_generator is None:
            self.viewer.status = (
                "SAM model not initialized. Cannot segment images."
            )
            return

        if self.current_image_for_segmentation is None:
            self.viewer.status = "No image loaded for segmentation."
            return

        try:
            # Map sensitivity (0-100) to SAM parameters
            # Higher sensitivity (100) = lower thresholds = more objects detected
            # Lower sensitivity (0) = higher thresholds = fewer objects detected

            # pred_iou_thresh range: 0.92 (low sensitivity) to 0.75 (high sensitivity)
            pred_iou = 0.92 - (self.sensitivity / 100) * 0.17

            # stability_score_thresh range: 0.97 (low sensitivity) to 0.85 (high sensitivity)
            stability = 0.97 - (self.sensitivity / 100) * 0.12

            # min_mask_region_area range: 300 (low sensitivity) to 30 (high sensitivity)
            min_area = 300 - (self.sensitivity / 100) * 270

            # Configure mask generator with sensitivity-adjusted parameters
            self.mask_generator.pred_iou_thresh = pred_iou
            self.mask_generator.stability_score_thresh = stability
            self.mask_generator.min_mask_region_area = min_area

            # Apply gamma correction based on sensitivity
            # Low sensitivity: gamma > 1 (brighten image)
            # High sensitivity: gamma < 1 (darken image)
            gamma = (
                1.5 - (self.sensitivity / 100) * 1.0
            )  # Range from 1.5 to 0.5

            # Apply gamma correction to the input image
            image_for_processing = self.current_image_for_segmentation.copy()

            # Convert to float for proper gamma correction
            image_float = image_for_processing.astype(np.float32) / 255.0

            # Apply gamma correction
            image_gamma = np.power(image_float, gamma)

            # Convert back to uint8
            image_gamma = (image_gamma * 255).astype(np.uint8)

            # Check if the image is very large and needs downscaling
            orig_shape = image_gamma.shape[:2]  # (height, width)

            # Calculate image size in megapixels
            image_mp = (orig_shape[0] * orig_shape[1]) / 1e6

            # If image is larger than 2 megapixels, downscale it
            max_mp = 2.0  # Maximum image size in megapixels
            scale_factor = 1.0

            if image_mp > max_mp:
                scale_factor = np.sqrt(max_mp / image_mp)
                new_height = int(orig_shape[0] * scale_factor)
                new_width = int(orig_shape[1] * scale_factor)

                self.viewer.status = f"Downscaling image from {orig_shape} to {(new_height, new_width)} for processing (scale: {scale_factor:.2f})"

                # Resize the image for processing
                image_gamma_resized = resize(
                    image_gamma,
                    (new_height, new_width),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.uint8)

                # Store scale factor for later use
                self.current_scale_factor = scale_factor
            else:
                image_gamma_resized = image_gamma
                self.current_scale_factor = 1.0

            self.viewer.status = f"Generating segmentation with sensitivity {self.sensitivity} (gamma={gamma:.2f})..."

            # Generate masks with gamma-corrected and potentially resized image
            masks = self.mask_generator.generate(image_gamma_resized)
            self.viewer.status = f"Generated {len(masks)} masks"

            if not masks:
                self.viewer.status = (
                    "No segments detected. Try increasing the sensitivity."
                )
                # Create empty label layer
                shape = self.current_image_for_segmentation.shape[:2]
                self.segmentation_result = np.zeros(shape, dtype=np.uint32)

                # Remove existing label layer if exists
                for layer in list(self.viewer.layers):
                    if (
                        isinstance(layer, Labels)
                        and "Segmentation" in layer.name
                    ):
                        self.viewer.layers.remove(layer)

                # Add new empty label layer
                self.label_layer = self.viewer.add_labels(
                    self.segmentation_result,
                    name=f"Segmentation ({os.path.basename(self.images[self.current_index])})",
                    opacity=0.7,
                )

                # Make the label layer active
                self.viewer.layers.selection.active = self.label_layer
                return

            # Process segmentation masks
            # If image was downscaled, we need to ensure masks are upscaled correctly
            if self.current_scale_factor < 1.0:
                # Upscale the segmentation masks to match the original image dimensions
                self._process_segmentation_masks_with_scaling(
                    masks, self.current_image_for_segmentation.shape[:2]
                )
            else:
                self._process_segmentation_masks(
                    masks, self.current_image_for_segmentation.shape[:2]
                )

            # Clear selected labels since segmentation has changed
            self.selected_labels = set()

            # Update table if it exists
            if self.label_table_widget:
                self._populate_label_table(self.label_table_widget)

        except (Exception, ValueError) as e:
            import traceback

            self.viewer.status = f"Error generating segmentation: {str(e)}"
            traceback.print_exc()

    def _process_segmentation_masks(self, masks, shape):
        """Process segmentation masks and create label layer."""
        # Create label image from masks
        labels = np.zeros(shape, dtype=np.uint32)
        self.label_info = {}  # Reset label info

        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            label_id = i + 1  # Start label IDs from 1
            labels[mask] = label_id

            # Calculate label information
            area = np.sum(mask)
            y_indices, x_indices = np.where(mask)
            center_y = np.mean(y_indices) if len(y_indices) > 0 else 0
            center_x = np.mean(x_indices) if len(x_indices) > 0 else 0

            # Store label info
            self.label_info[label_id] = {
                "area": area,
                "center_y": center_y,
                "center_x": center_x,
                "score": mask_data.get("stability_score", 0),
            }

        # Sort labels by area (largest first)
        self.label_info = dict(
            sorted(
                self.label_info.items(),
                key=lambda item: item[1]["area"],
                reverse=True,
            )
        )

        # Save segmentation result
        self.segmentation_result = labels

        # Remove existing label layer if exists
        for layer in list(self.viewer.layers):
            if isinstance(layer, Labels) and "Segmentation" in layer.name:
                self.viewer.layers.remove(layer)

        # Add label layer to viewer
        self.label_layer = self.viewer.add_labels(
            labels,
            name=f"Segmentation ({os.path.basename(self.images[self.current_index])})",
            opacity=0.7,
        )

        # Make the label layer active by default
        self.viewer.layers.selection.active = self.label_layer

        # Disconnect existing callbacks if any
        if (
            hasattr(self, "label_layer")
            and self.label_layer is not None
            and hasattr(self.label_layer, "mouse_drag_callbacks")
        ):
            # Remove old callbacks
            for callback in list(self.label_layer.mouse_drag_callbacks):
                self.label_layer.mouse_drag_callbacks.remove(callback)

        # Connect mouse click event to label selection
        self.label_layer.mouse_drag_callbacks.append(self._on_label_clicked)

        # image_name = os.path.basename(self.images[self.current_index])
        self.viewer.status = f"Loaded image {self.current_index + 1}/{len(self.images)} - Found {len(masks)} segments"

    # New method for handling scaled segmentation masks
    def _process_segmentation_masks_with_scaling(self, masks, original_shape):
        """Process segmentation masks with scaling to match the original image size."""
        # Create label image from masks
        # First determine the size of the mask predictions (which are at the downscaled resolution)
        if not masks:
            return

        mask_shape = masks[0]["segmentation"].shape

        # Create an empty label image at the downscaled resolution
        downscaled_labels = np.zeros(mask_shape, dtype=np.uint32)
        self.label_info = {}  # Reset label info

        # Fill in the downscaled labels
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            label_id = i + 1  # Start label IDs from 1
            downscaled_labels[mask] = label_id

            # Store basic label info
            area = np.sum(mask)
            y_indices, x_indices = np.where(mask)
            center_y = np.mean(y_indices) if len(y_indices) > 0 else 0
            center_x = np.mean(x_indices) if len(x_indices) > 0 else 0

            # Scale centers to original image coordinates
            center_y_orig = center_y / self.current_scale_factor
            center_x_orig = center_x / self.current_scale_factor

            # Store label info at original scale
            self.label_info[label_id] = {
                "area": area
                / (
                    self.current_scale_factor**2
                ),  # Approximate area in original scale
                "center_y": center_y_orig,
                "center_x": center_x_orig,
                "score": mask_data.get("stability_score", 0),
            }

        # Upscale the labels to the original image size
        upscaled_labels = resize(
            downscaled_labels,
            original_shape,
            order=0,  # Nearest neighbor interpolation
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint32)

        # Sort labels by area (largest first)
        self.label_info = dict(
            sorted(
                self.label_info.items(),
                key=lambda item: item[1]["area"],
                reverse=True,
            )
        )

        # Save segmentation result
        self.segmentation_result = upscaled_labels

        # Remove existing label layer if exists
        for layer in list(self.viewer.layers):
            if isinstance(layer, Labels) and "Segmentation" in layer.name:
                self.viewer.layers.remove(layer)

        # Add label layer to viewer
        self.label_layer = self.viewer.add_labels(
            upscaled_labels,
            name=f"Segmentation ({os.path.basename(self.images[self.current_index])})",
            opacity=0.7,
        )

        # Make the label layer active by default
        self.viewer.layers.selection.active = self.label_layer

        # Disconnect existing callbacks if any
        if (
            hasattr(self, "label_layer")
            and self.label_layer is not None
            and hasattr(self.label_layer, "mouse_drag_callbacks")
        ):
            # Remove old callbacks
            for callback in list(self.label_layer.mouse_drag_callbacks):
                self.label_layer.mouse_drag_callbacks.remove(callback)

        # Connect mouse click event to label selection
        self.label_layer.mouse_drag_callbacks.append(self._on_label_clicked)

        self.viewer.status = f"Loaded image {self.current_index + 1}/{len(self.images)} - Found {len(masks)} segments"

    # --------------------------------------------------
    # Label Selection and UI Elements
    # --------------------------------------------------

    def _on_label_clicked(self, layer, event):
        """Handle label selection on mouse click."""
        try:
            # Only process clicks, not drags
            if event.type != "mouse_press":
                return

            # Get coordinates of mouse click
            coords = np.round(event.position).astype(int)

            # Make sure coordinates are within bounds
            shape = self.segmentation_result.shape
            if (
                coords[0] < 0
                or coords[1] < 0
                or coords[0] >= shape[0]
                or coords[1] >= shape[1]
            ):
                return

            # Get the label ID at the clicked position
            label_id = self.segmentation_result[coords[0], coords[1]]

            # Skip if background (0) is clicked
            if label_id == 0:
                return

            # Toggle the label selection
            if label_id in self.selected_labels:
                self.selected_labels.remove(label_id)
                self.viewer.status = f"Deselected label ID: {label_id} | Selected labels: {self.selected_labels}"
            else:
                self.selected_labels.add(label_id)
                self.viewer.status = f"Selected label ID: {label_id} | Selected labels: {self.selected_labels}"

            # Update table if it exists
            self._update_label_table()

            # Update preview after selection changes
            self.preview_crop()

        except (Exception, ValueError) as e:
            self.viewer.status = f"Error selecting label: {str(e)}"

    def create_label_table(self, parent_widget):
        """Create a table widget displaying all detected labels."""
        # Create table widget
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Select", "Label ID"])

        # Set up the table
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)

        # Turn off alternating colors to avoid coloring issues
        table.setAlternatingRowColors(False)

        # Column sizing
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        table.horizontalHeader().setMinimumSectionSize(80)

        # Fill the table with label information
        self._populate_label_table(table)

        # Store reference to the table
        self.label_table_widget = table

        # Connect signal to make segmentation layer active when table is clicked
        table.clicked.connect(lambda: self._ensure_segmentation_layer_active())

        return table

    def _ensure_segmentation_layer_active(self):
        """Ensure the segmentation layer is the active layer."""
        if self.label_layer is not None:
            self.viewer.layers.selection.active = self.label_layer

    def _populate_label_table(self, table):
        """Populate the table with label information."""
        if not self.label_info:
            table.setRowCount(0)
            return

        # Set row count
        table.setRowCount(len(self.label_info))

        # Sort labels by size (largest first)
        sorted_labels = sorted(
            self.label_info.items(),
            key=lambda item: item[1]["area"],
            reverse=True,
        )

        # Fill table with data
        for row, (label_id, _info) in enumerate(sorted_labels):
            # Checkbox for selection
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(5, 0, 5, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)

            checkbox = QCheckBox()
            checkbox.setChecked(label_id in self.selected_labels)

            # Connect checkbox to label selection
            def make_checkbox_callback(lid):
                def callback(state):
                    if state == Qt.Checked:
                        self.selected_labels.add(lid)
                    else:
                        self.selected_labels.discard(lid)
                    self.preview_crop()

                return callback

            checkbox.stateChanged.connect(make_checkbox_callback(label_id))

            checkbox_layout.addWidget(checkbox)
            table.setCellWidget(row, 0, checkbox_widget)

            # Label ID as plain text with transparent background
            item = QTableWidgetItem(str(label_id))
            item.setTextAlignment(Qt.AlignCenter)

            # Set the background color to transparent
            brush = item.background()
            brush.setStyle(Qt.NoBrush)
            item.setBackground(brush)

            table.setItem(row, 1, item)

    def _update_label_table(self):
        """Update the label selection table if it exists."""
        if self.label_table_widget is None:
            return

        # Block signals during update
        self.label_table_widget.blockSignals(True)

        # Update checkboxes
        for row in range(self.label_table_widget.rowCount()):
            # Get label ID from the visible column
            label_id_item = self.label_table_widget.item(row, 1)
            if label_id_item is None:
                continue

            label_id = int(label_id_item.text())

            # Find checkbox cell
            checkbox_item = self.label_table_widget.cellWidget(row, 0)
            if checkbox_item is None:
                continue

            # Update checkbox state
            checkbox = checkbox_item.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(label_id in self.selected_labels)

        # Unblock signals
        self.label_table_widget.blockSignals(False)

    def select_all_labels(self):
        """Select all labels."""
        if not self.label_info:
            return

        self.selected_labels = set(self.label_info.keys())
        self._update_label_table()
        self.preview_crop()
        self.viewer.status = f"Selected all {len(self.selected_labels)} labels"

    def clear_selection(self):
        """Clear all selected labels."""
        self.selected_labels = set()
        self._update_label_table()
        self.preview_crop()
        self.viewer.status = "Cleared all selections"

    # --------------------------------------------------
    # Image Processing and Export
    # --------------------------------------------------

    def preview_crop(self, label_ids=None):
        """Preview the crop result with the selected label IDs."""
        if self.segmentation_result is None or self.image_layer is None:
            self.viewer.status = (
                "No image or segmentation available for preview."
            )
            return

        try:
            # Use provided label IDs or default to selected labels
            if label_ids is None:
                label_ids = self.selected_labels

            # Skip if no labels are selected
            if not label_ids:
                # Remove previous preview if exists
                for layer in list(self.viewer.layers):
                    if "Preview" in layer.name:
                        self.viewer.layers.remove(layer)

                # Make sure the segmentation layer is active again
                if self.label_layer is not None:
                    self.viewer.layers.selection.active = self.label_layer
                return

            # Get current image
            image = self.original_image.copy()

            # Create mask from selected label IDs
            mask = np.zeros_like(self.segmentation_result, dtype=bool)
            for label_id in label_ids:
                mask |= self.segmentation_result == label_id

            # Apply mask to image for preview (set everything outside mask to 0)
            if len(image.shape) == 2:
                # Grayscale image
                preview_image = image.copy()
                preview_image[~mask] = 0
            else:
                # Color image
                preview_image = image.copy()
                for c in range(preview_image.shape[2]):
                    preview_image[:, :, c][~mask] = 0

            # Remove previous preview if exists
            for layer in list(self.viewer.layers):
                if "Preview" in layer.name:
                    self.viewer.layers.remove(layer)

            # Add preview layer
            if label_ids:
                label_str = ", ".join(str(lid) for lid in sorted(label_ids))
                self.viewer.add_image(
                    preview_image,
                    name=f"Preview (Labels: {label_str})",
                    opacity=0.55,
                )

            # Make sure the segmentation layer is active again
            if self.label_layer is not None:
                self.viewer.layers.selection.active = self.label_layer

        except (Exception, ValueError) as e:
            self.viewer.status = f"Error generating preview: {str(e)}"

    def crop_with_selected_labels(self):
        """Crop the current image using all selected label IDs."""
        if self.segmentation_result is None or self.original_image is None:
            self.viewer.status = (
                "No image or segmentation available for cropping."
            )
            return False

        if not self.selected_labels:
            self.viewer.status = "No labels selected for cropping."
            return False

        try:
            # Get current image
            image = self.original_image

            # Create mask from all selected label IDs
            mask = np.zeros_like(self.segmentation_result, dtype=bool)
            for label_id in self.selected_labels:
                mask |= self.segmentation_result == label_id

            # Apply mask to image (set everything outside mask to 0)
            if len(image.shape) == 2:
                # Grayscale image
                cropped_image = image.copy()
                cropped_image[~mask] = 0
            else:
                # Color image
                cropped_image = image.copy()
                for c in range(cropped_image.shape[2]):
                    cropped_image[:, :, c][~mask] = 0

            # Save cropped image
            image_path = self.images[self.current_index]
            base_name, ext = os.path.splitext(image_path)
            label_str = "_".join(
                str(lid) for lid in sorted(self.selected_labels)
            )
            output_path = f"{base_name}_cropped_{label_str}{ext}"

            # Save using appropriate method based on file type
            if output_path.lower().endswith((".tif", ".tiff")):
                imwrite(output_path, cropped_image, compression="zlib")
            else:
                from skimage.io import imsave

                imsave(output_path, cropped_image)

            self.viewer.status = f"Saved cropped image to {output_path}"

            # Make sure the segmentation layer is active again
            if self.label_layer is not None:
                self.viewer.layers.selection.active = self.label_layer

            return True

        except (Exception, ValueError) as e:
            self.viewer.status = f"Error cropping image: {str(e)}"
            return False


# --------------------------------------------------
# UI Creation Functions
# --------------------------------------------------


def create_crop_widget(processor):
    """Create the crop control widget."""
    crop_widget = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(10)  # Add more space between elements
    layout.setContentsMargins(
        10, 10, 10, 10
    )  # Add margins around all elements

    # Instructions
    instructions_label = QLabel(
        "Select objects to keep in the cropped image.\n"
        "You can select labels using the table below or by clicking directly on objects "
        "in the image (make sure the Segmentation layer is active)."
    )
    instructions_label.setWordWrap(True)
    layout.addWidget(instructions_label)

    # Sensitivity slider
    sensitivity_layout = QVBoxLayout()

    # Header label
    sensitivity_header_layout = QHBoxLayout()
    sensitivity_label = QLabel("Segmentation Sensitivity:")
    sensitivity_value_label = QLabel(f"{processor.sensitivity}")
    sensitivity_header_layout.addWidget(sensitivity_label)
    sensitivity_header_layout.addStretch()
    sensitivity_header_layout.addWidget(sensitivity_value_label)
    sensitivity_layout.addLayout(sensitivity_header_layout)

    # Slider
    slider_layout = QHBoxLayout()
    sensitivity_slider = QSlider(Qt.Horizontal)
    sensitivity_slider.setMinimum(0)
    sensitivity_slider.setMaximum(100)
    sensitivity_slider.setValue(processor.sensitivity)
    sensitivity_slider.setTickPosition(QSlider.TicksBelow)
    sensitivity_slider.setTickInterval(10)
    slider_layout.addWidget(sensitivity_slider)

    apply_sensitivity_button = QPushButton("Apply")
    apply_sensitivity_button.setToolTip(
        "Apply sensitivity changes to regenerate segmentation"
    )
    slider_layout.addWidget(apply_sensitivity_button)
    sensitivity_layout.addLayout(slider_layout)

    # Description label
    sensitivity_description = QLabel(
        "Medium sensitivity - Balanced detection (γ=1.00)"
    )
    sensitivity_description.setStyleSheet("font-style: italic; color: #666;")
    sensitivity_layout.addWidget(sensitivity_description)

    layout.addLayout(sensitivity_layout)

    # Create label table
    label_table = processor.create_label_table(crop_widget)
    label_table.setMinimumHeight(150)  # Reduce minimum height to save space
    label_table.setMaximumHeight(
        300
    )  # Set maximum height to prevent taking too much space
    layout.addWidget(label_table)

    # Remove "Focus on Segmentation Layer" button as it's now redundant
    selection_layout = QHBoxLayout()
    select_all_button = QPushButton("Select All")
    clear_selection_button = QPushButton("Clear Selection")
    selection_layout.addWidget(select_all_button)
    selection_layout.addWidget(clear_selection_button)
    layout.addLayout(selection_layout)

    # Crop button
    crop_button = QPushButton("Crop with Selected Objects")
    layout.addWidget(crop_button)

    # Navigation buttons
    nav_layout = QHBoxLayout()
    prev_button = QPushButton("Previous Image")
    next_button = QPushButton("Next Image")
    nav_layout.addWidget(prev_button)
    nav_layout.addWidget(next_button)
    layout.addLayout(nav_layout)

    # Status label
    status_label = QLabel(
        "Ready to process images. Select objects using the table or by clicking on them."
    )
    status_label.setWordWrap(True)
    layout.addWidget(status_label)

    # Set layout
    crop_widget.setLayout(layout)

    # Function to completely replace the table widget
    def replace_table_widget():
        nonlocal label_table
        # Remove old table
        layout.removeWidget(label_table)
        label_table.setParent(None)
        label_table.deleteLater()

        # Create new table
        label_table = processor.create_label_table(crop_widget)
        label_table.setMinimumHeight(200)
        layout.insertWidget(3, label_table)  # Insert after sensitivity slider
        return label_table

    # Connect button signals
    def on_sensitivity_changed(value):
        sensitivity_value_label.setText(f"{value}")
        # Update description based on sensitivity
        if value < 25:
            gamma = (
                1.5 - (value / 100) * 1.0
            )  # Higher gamma for low sensitivity
            description = f"Low sensitivity - Seeks large, distinct objects (γ={gamma:.2f})"
        elif value < 75:
            gamma = 1.5 - (value / 100) * 1.0
            description = (
                f"Medium sensitivity - Balanced detection (γ={gamma:.2f})"
            )
        else:
            gamma = (
                1.5 - (value / 100) * 1.0
            )  # Lower gamma for high sensitivity
            description = f"High sensitivity - Detects subtle, small objects (γ={gamma:.2f})"
        sensitivity_description.setText(description)

    def on_apply_sensitivity_clicked():
        new_sensitivity = sensitivity_slider.value()
        processor.generate_segmentation_with_sensitivity(new_sensitivity)
        replace_table_widget()
        status_label.setText(
            f"Regenerated segmentation with sensitivity {new_sensitivity}"
        )

    def on_select_all_clicked():
        processor.select_all_labels()
        status_label.setText(
            f"Selected all {len(processor.selected_labels)} objects"
        )

    def on_clear_selection_clicked():
        processor.clear_selection()
        status_label.setText("Selection cleared")

    def on_crop_clicked():
        success = processor.crop_with_selected_labels()
        if success:
            labels_str = ", ".join(
                str(label) for label in sorted(processor.selected_labels)
            )
            status_label.setText(
                f"Cropped image with {len(processor.selected_labels)} objects (IDs: {labels_str})"
            )

    def on_next_clicked():
        if not processor.next_image():
            next_button.setEnabled(False)
        else:
            prev_button.setEnabled(True)
            replace_table_widget()
            # Reset sensitivity slider to default
            sensitivity_slider.setValue(processor.sensitivity)
            sensitivity_value_label.setText(f"{processor.sensitivity}")
            status_label.setText(
                f"Showing image {processor.current_index + 1}/{len(processor.images)}"
            )

    def on_prev_clicked():
        if not processor.previous_image():
            prev_button.setEnabled(False)
        else:
            next_button.setEnabled(True)
            replace_table_widget()
            # Reset sensitivity slider to default
            sensitivity_slider.setValue(processor.sensitivity)
            sensitivity_value_label.setText(f"{processor.sensitivity}")
            status_label.setText(
                f"Showing image {processor.current_index + 1}/{len(processor.images)}"
            )

    sensitivity_slider.valueChanged.connect(on_sensitivity_changed)
    apply_sensitivity_button.clicked.connect(on_apply_sensitivity_clicked)
    select_all_button.clicked.connect(on_select_all_clicked)
    clear_selection_button.clicked.connect(on_clear_selection_clicked)
    crop_button.clicked.connect(on_crop_clicked)
    next_button.clicked.connect(on_next_clicked)
    prev_button.clicked.connect(on_prev_clicked)

    return crop_widget


# --------------------------------------------------
# Napari Plugin Functions
# --------------------------------------------------


@magicgui(
    call_button="Start Batch Crop Anything",
    folder_path={"label": "Folder Path", "widget_type": "LineEdit"},
)
def batch_crop_anything(
    folder_path: str,
    viewer: Viewer = None,
):
    """MagicGUI widget for starting Batch Crop Anything."""
    # Check if Mobile-SAM is available
    try:
        # import torch
        # from mobile_sam import sam_model_registry

        # Check if the required files are included with the package
        try:
            import importlib.util
            import os

            mobile_sam_spec = importlib.util.find_spec("mobile_sam")
            if mobile_sam_spec is None:
                raise ImportError("mobile_sam package not found")

            mobile_sam_path = os.path.dirname(mobile_sam_spec.origin)

            # Check for model file in package
            model_found = False
            checkpoint_paths = [
                os.path.join(mobile_sam_path, "weights", "mobile_sam.pt"),
                os.path.join(mobile_sam_path, "mobile_sam.pt"),
                os.path.join(
                    os.path.dirname(mobile_sam_path),
                    "weights",
                    "mobile_sam.pt",
                ),
                os.path.join(
                    os.path.expanduser("~"), "models", "mobile_sam.pt"
                ),
                "/opt/T-MIDAS/models/mobile_sam.pt",
                os.path.join(os.getcwd(), "mobile_sam.pt"),
            ]

            for path in checkpoint_paths:
                if os.path.exists(path):
                    model_found = True
                    break

            if not model_found:
                QMessageBox.warning(
                    None,
                    "Model File Missing",
                    "Mobile-SAM model weights (mobile_sam.pt) not found. You'll be prompted to locate it when starting the tool.\n\n"
                    "You can download it from: https://github.com/ChaoningZhang/MobileSAM/tree/master/weights",
                )
        except (ImportError, AttributeError) as e:
            print(f"Warning checking for model file: {str(e)}")

    except ImportError:
        QMessageBox.critical(
            None,
            "Missing Dependency",
            "Mobile-SAM not found. Please install with:\n"
            "pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n\n"
            "You'll also need to download the model weights file (mobile_sam.pt) from:\n"
            "https://github.com/ChaoningZhang/MobileSAM/tree/master/weights",
        )
        return

    # Initialize processor and load images
    processor = BatchCropAnything(viewer)
    processor.load_images(folder_path)

    # Create UI
    crop_widget = create_crop_widget(processor)

    # Wrap the widget in a scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidget(crop_widget)
    scroll_area.setWidgetResizable(
        True
    )  # This allows the widget to resize with the scroll area
    scroll_area.setFrameShape(QScrollArea.NoFrame)  # Hide the frame
    scroll_area.setMinimumHeight(
        500
    )  # Set a minimum height to ensure visibility

    # Add scroll area to viewer
    viewer.window.add_dock_widget(scroll_area, name="Crop Controls")


def batch_crop_anything_widget():
    """Provide the batch crop anything widget to Napari."""
    # Create the magicgui widget
    widget = batch_crop_anything

    # Create and add browse button for folder path
    folder_browse_button = QPushButton("Browse...")

    def on_folder_browse_clicked():
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            # Update the folder_path field
            widget.folder_path.value = folder

    folder_browse_button.clicked.connect(on_folder_browse_clicked)

    # Insert the browse button next to the folder_path field
    folder_layout = widget.folder_path.native.parent().layout()
    folder_layout.addWidget(folder_browse_button)

    return widget

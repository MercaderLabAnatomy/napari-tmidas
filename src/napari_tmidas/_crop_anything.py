"""
Batch Crop Anything - A Napari plugin for interactive image cropping

This plugin combines SAM2 for automatic object detection with
an interactive interface for selecting and cropping objects from images.
The plugin supports both 2D (YX) and 3D (TYX/ZYX) data.
"""

import contextlib
import os
import sys

import numpy as np
import requests
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread
from skimage.transform import resize
from tifffile import imwrite

from napari_tmidas.processing_functions.sam2_mp4 import tif_to_mp4

sam2_paths = [
    os.environ.get("SAM2_PATH"),
    "/opt/sam2",
    os.path.expanduser("~/sam2"),
    "./sam2",
]

for path in sam2_paths:
    if path and os.path.exists(path):
        sys.path.append(path)
        break
else:
    print(
        "Warning: SAM2 not found in common locations. Please set SAM2_PATH environment variable."
    )


def get_device():
    if sys.platform == "darwin":
        # MacOS: Only check for MPS
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        # Other platforms: check for CUDA, then CPU
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    return device


class BatchCropAnything:
    """Class for processing images with SAM2 and cropping selected objects."""

    def __init__(self, viewer: Viewer, use_3d=False):
        """Initialize the BatchCropAnything processor."""
        # Core components
        self.viewer = viewer
        self.images = []
        self.current_index = 0
        self.use_3d = use_3d

        # Image and segmentation data
        self.original_image = None
        self.segmentation_result = None
        self.current_image_for_segmentation = None
        self.current_scale_factor = 1.0

        # UI references
        self.image_layer = None
        self.label_layer = None
        self.label_table_widget = None

        # State tracking
        self.selected_labels = set()
        self.label_info = {}

        # Segmentation parameters
        self.sensitivity = 50  # Default sensitivity (0-100 scale)

        # Initialize the SAM2 model
        self._initialize_sam2()

    def _initialize_sam2(self):
        """Initialize the SAM2 model based on dimension mode."""

        def download_checkpoint(url, dest_folder):
            import os

            os.makedirs(dest_folder, exist_ok=True)
            filename = os.path.join(dest_folder, url.split("/")[-1])
            if not os.path.exists(filename):
                print(f"Downloading checkpoint to {filename}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            else:
                print(f"Checkpoint already exists at {filename}.")
            return filename

        try:
            # import torch

            self.device = get_device()

            # Download checkpoint if needed
            checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            checkpoint_path = download_checkpoint(
                checkpoint_url, "/opt/sam2/checkpoints/"
            )
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

            if self.use_3d:
                from sam2.build_sam import build_sam2_video_predictor

                self.predictor = build_sam2_video_predictor(
                    model_cfg, checkpoint_path, device=self.device
                )
                self.viewer.status = (
                    f"Initialized SAM2 Video Predictor on {self.device}"
                )
            else:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                self.predictor = SAM2ImagePredictor(
                    build_sam2(model_cfg, checkpoint_path)
                )
                self.viewer.status = (
                    f"Initialized SAM2 Image Predictor on {self.device}"
                )

        except (
            ImportError,
            RuntimeError,
            ValueError,
            FileNotFoundError,
            requests.RequestException,
        ) as e:
            import traceback

            self.viewer.status = f"Error initializing SAM2: {str(e)}"
            self.predictor = None
            print(traceback.format_exc())

    def load_images(self, folder_path: str):
        """Load images from the specified folder path."""
        if not os.path.exists(folder_path):
            self.viewer.status = f"Folder not found: {folder_path}"
            return

        files = os.listdir(folder_path)
        self.images = [
            os.path.join(folder_path, file)
            for file in files
            if file.lower().endswith(".tif")
            or file.lower().endswith(".tiff")
            and "label" not in file.lower()
            and "cropped" not in file.lower()
            and "_labels_" not in file.lower()
            and "_cropped_" not in file.lower()
        ]

        if not self.images:
            self.viewer.status = "No compatible images found in the folder."
            return

        self.viewer.status = f"Found {len(self.images)} .tif images."
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

        if self.predictor is None:
            self.viewer.status = (
                "SAM2 model not initialized. Cannot segment images."
            )
            return

        image_path = self.images[self.current_index]
        self.viewer.status = f"Processing {os.path.basename(image_path)}"

        try:
            # Clear existing layers
            self.viewer.layers.clear()

            # Load and process image
            self.original_image = imread(image_path)

            # For 3D/4D data, determine dimensions
            if self.use_3d and len(self.original_image.shape) >= 3:
                # Check shape to identify dimensions
                if len(self.original_image.shape) == 4:  # TZYX or similar
                    # Identify time dimension as first dim with size > 4 and < 400
                    # This is a heuristic to differentiate time from channels/small Z stacks
                    time_dim_idx = -1
                    for i, dim_size in enumerate(self.original_image.shape):
                        if 4 < dim_size < 400:
                            time_dim_idx = i
                            break

                    if time_dim_idx == 0:  # TZYX format
                        # Keep as is, T is already the first dimension
                        self.image_layer = self.viewer.add_image(
                            self.original_image,
                            name=f"Image ({os.path.basename(image_path)})",
                        )
                        # Store time dimension info
                        self.time_dim_size = self.original_image.shape[0]
                        self.has_z_dim = True
                    elif (
                        time_dim_idx > 0
                    ):  # Unusual format, we need to transpose
                        # Transpose to move T to first dimension
                        # Create permutation order that puts time_dim_idx first
                        perm_order = list(
                            range(len(self.original_image.shape))
                        )
                        perm_order.remove(time_dim_idx)
                        perm_order.insert(0, time_dim_idx)

                        transposed_image = np.transpose(
                            self.original_image, perm_order
                        )
                        self.original_image = (
                            transposed_image  # Replace with transposed version
                        )

                        self.image_layer = self.viewer.add_image(
                            self.original_image,
                            name=f"Image ({os.path.basename(image_path)})",
                        )
                        # Store time dimension info
                        self.time_dim_size = self.original_image.shape[0]
                        self.has_z_dim = True
                    else:
                        # No time dimension found, treat as ZYX
                        self.image_layer = self.viewer.add_image(
                            self.original_image,
                            name=f"Image ({os.path.basename(image_path)})",
                        )
                        self.time_dim_size = 1
                        self.has_z_dim = True
                elif (
                    len(self.original_image.shape) == 3
                ):  # Could be TYX or ZYX
                    # Check if first dimension is likely time (> 4, < 400)
                    if 4 < self.original_image.shape[0] < 400:
                        # Likely TYX format
                        self.image_layer = self.viewer.add_image(
                            self.original_image,
                            name=f"Image ({os.path.basename(image_path)})",
                        )
                        self.time_dim_size = self.original_image.shape[0]
                        self.has_z_dim = False
                    else:
                        # Likely ZYX format or another 3D format
                        self.image_layer = self.viewer.add_image(
                            self.original_image,
                            name=f"Image ({os.path.basename(image_path)})",
                        )
                        self.time_dim_size = 1
                        self.has_z_dim = True
                else:
                    # Should not reach here with use_3d=True, but just in case
                    self.image_layer = self.viewer.add_image(
                        self.original_image,
                        name=f"Image ({os.path.basename(image_path)})",
                    )
                    self.time_dim_size = 1
                    self.has_z_dim = False
            else:
                # Handle 2D data as before
                if self.original_image.dtype != np.uint8:
                    image_for_display = (
                        self.original_image
                        / np.amax(self.original_image)
                        * 255
                    ).astype(np.uint8)
                else:
                    image_for_display = self.original_image

                # Add image to viewer
                self.image_layer = self.viewer.add_image(
                    image_for_display,
                    name=f"Image ({os.path.basename(image_path)})",
                )

            # Generate segmentation
            self._generate_segmentation(self.original_image, image_path)

        except (FileNotFoundError, ValueError, TypeError, OSError) as e:
            import traceback

            self.viewer.status = f"Error processing image: {str(e)}"
            traceback.print_exc()

            # Create empty segmentation in case of error
            if (
                hasattr(self, "original_image")
                and self.original_image is not None
            ):
                if self.use_3d:
                    shape = self.original_image.shape
                else:
                    shape = self.original_image.shape[:2]

                self.segmentation_result = np.zeros(shape, dtype=np.uint32)
                self.label_layer = self.viewer.add_labels(
                    self.segmentation_result, name="Error: No Segmentation"
                )

    def _generate_segmentation(self, image, image_path: str):
        """Generate segmentation for the current image using SAM2."""
        # Store the current image for later processing
        self.current_image_for_segmentation = image

        # Generate segmentation with current sensitivity
        self.generate_segmentation_with_sensitivity(image_path)

    def generate_segmentation_with_sensitivity(
        self, image_path: str, sensitivity=None
    ):
        """Generate segmentation with the specified sensitivity."""
        if sensitivity is not None:
            self.sensitivity = sensitivity

        if self.predictor is None:
            self.viewer.status = (
                "SAM2 model not initialized. Cannot segment images."
            )
            return

        if self.current_image_for_segmentation is None:
            self.viewer.status = "No image loaded for segmentation."
            return

        try:
            # Map sensitivity (0-100) to SAM2 parameters
            # For SAM2, adjust confidence threshold based on sensitivity
            confidence_threshold = (
                0.9 - (self.sensitivity / 100) * 0.4
            )  # Range from 0.9 to 0.5

            # Process based on dimension mode
            if self.use_3d:
                # Process 3D data
                self._generate_3d_segmentation(
                    confidence_threshold, image_path
                )
            else:
                # Process 2D data
                self._generate_2d_segmentation(confidence_threshold)

        except (
            ValueError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
            TypeError,
        ) as e:
            import traceback

            self.viewer.status = f"Error generating segmentation: {str(e)}"
            traceback.print_exc()

    def _generate_2d_segmentation(self, confidence_threshold):
        """Generate 2D segmentation using SAM2 Image Predictor."""
        # Ensure image is in the correct format for SAM2
        image = self.current_image_for_segmentation

        # Handle resizing for very large images
        orig_shape = image.shape[:2]
        image_mp = (orig_shape[0] * orig_shape[1]) / 1e6
        max_mp = 2.0  # Maximum image size in megapixels

        if image_mp > max_mp:
            scale_factor = np.sqrt(max_mp / image_mp)
            new_height = int(orig_shape[0] * scale_factor)
            new_width = int(orig_shape[1] * scale_factor)

            self.viewer.status = f"Downscaling image from {orig_shape} to {(new_height, new_width)} for processing"

            # Resize image
            resized_image = resize(
                image,
                (new_height, new_width),
                anti_aliasing=True,
                preserve_range=True,
            ).astype(
                np.float32
            )  # Convert to float32

            self.current_scale_factor = scale_factor
        else:
            # Convert to float32 format
            if image.dtype != np.float32:
                resized_image = image.astype(np.float32)
            else:
                resized_image = image
            self.current_scale_factor = 1.0

        # Ensure image is in RGB format for SAM2
        if len(resized_image.shape) == 2:
            # Convert grayscale to RGB
            resized_image = np.stack([resized_image] * 3, axis=-1)
        elif len(resized_image.shape) == 3 and resized_image.shape[2] == 1:
            # Convert single channel to RGB
            resized_image = np.concatenate([resized_image] * 3, axis=2)
        elif len(resized_image.shape) == 3 and resized_image.shape[2] > 3:
            # Use first 3 channels
            resized_image = resized_image[:, :, :3]

        # Normalize the image to [0,1] range if it's not already
        if resized_image.max() > 1.0:
            resized_image = resized_image / 255.0

        # Set SAM2 prediction parameters based on sensitivity
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float32
        ):
            # Set the image in the predictor
            self.predictor.set_image(resized_image)

            # Use automatic points generation with confidence threshold
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True,
            )

            # Filter masks by confidence threshold
            valid_masks = scores > confidence_threshold
            masks = masks[valid_masks]
            scores = scores[valid_masks]

        # Convert masks to label image
        labels = np.zeros(resized_image.shape[:2], dtype=np.uint32)
        self.label_info = {}  # Reset label info

        for i, mask in enumerate(masks):
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
                "score": float(scores[i]),
            }

        # Handle upscaling if needed
        if self.current_scale_factor < 1.0:
            labels = resize(
                labels,
                orig_shape,
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
        self.segmentation_result = labels

        # Update the label layer
        self._update_label_layer()

    def _generate_3d_segmentation(self, confidence_threshold, image_path):
        """
        Initialize 3D segmentation using SAM2 Video Predictor.
        This correctly sets up interactive segmentation following SAM2's video approach.
        """
        try:
            # Handle image_path - make sure it's a string
            if not isinstance(image_path, str):
                image_path = self.images[self.current_index]

            # Initialize empty segmentation
            volume_shape = self.current_image_for_segmentation.shape
            labels = np.zeros(volume_shape, dtype=np.uint32)
            self.segmentation_result = labels

            # Create a temp directory for the MP4 conversion if needed
            import os
            import tempfile

            temp_dir = tempfile.gettempdir()
            mp4_path = os.path.join(
                temp_dir, f"temp_volume_{os.path.basename(image_path)}.mp4"
            )

            # If we need to save a modified version for MP4 conversion
            need_temp_tif = False
            temp_tif_path = None

            # Check if we have a 4D volume with Z dimension
            if (
                hasattr(self, "has_z_dim")
                and self.has_z_dim
                and len(self.current_image_for_segmentation.shape) == 4
            ):
                # We need to convert the 4D TZYX to a 3D TYX for proper video conversion
                # by taking maximum intensity projection of Z for each time point
                self.viewer.status = (
                    "Converting 4D TZYX volume to 3D TYX for SAM2..."
                )

                # Create maximum intensity projection along Z axis (axis 1 in TZYX)
                projected_volume = np.max(
                    self.current_image_for_segmentation, axis=1
                )

                # Save this as a temporary TIF for MP4 conversion
                temp_tif_path = os.path.join(
                    temp_dir, f"temp_projected_{os.path.basename(image_path)}"
                )
                imwrite(temp_tif_path, projected_volume)
                need_temp_tif = True

                # Convert the projected TIF to MP4
                self.viewer.status = (
                    "Converting projected 3D volume to MP4 format for SAM2..."
                )
                mp4_path = tif_to_mp4(temp_tif_path)
            else:
                # Convert original volume to video format for SAM2
                self.viewer.status = (
                    "Converting 3D volume to MP4 format for SAM2..."
                )
                mp4_path = tif_to_mp4(image_path)

            # Initialize SAM2 state with the video
            self.viewer.status = "Initializing SAM2 Video Predictor..."
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                self._sam2_state = self.predictor.init_state(mp4_path)

            # Store needed state for 3D processing
            self._sam2_next_obj_id = 1
            self._sam2_prompts = (
                {}
            )  # Store prompts for each object (points, labels, box)

            # Update the label layer with empty segmentation
            self._update_label_layer()

            # Replace the click handler for interactive 3D segmentation
            if self.label_layer is not None and hasattr(
                self.label_layer, "mouse_drag_callbacks"
            ):
                for callback in list(self.label_layer.mouse_drag_callbacks):
                    self.label_layer.mouse_drag_callbacks.remove(callback)

                # Add 3D-specific click handler
                self.label_layer.mouse_drag_callbacks.append(
                    self._on_3d_label_clicked
                )

            # Set the viewer to show the first frame
            if hasattr(self.viewer, "dims") and self.viewer.dims.ndim > 2:
                self.viewer.dims.set_point(
                    0, 0
                )  # Set the first dimension (typically time/z) to 0

            # Clean up temporary file if we created one
            if (
                need_temp_tif
                and temp_tif_path
                and os.path.exists(temp_tif_path)
            ):
                with contextlib.suppress(Exception):
                    os.remove(temp_tif_path)

            # Show instructions
            self.viewer.status = (
                "3D Mode active: Navigate to the first frame where object appears, then click. "
                "Use Shift+click for negative points (to remove areas). "
                "Segmentation will be propagated to all frames automatically."
            )

            return True

        except (
            FileNotFoundError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
            ValueError,
            OSError,
        ) as e:
            import traceback

            self.viewer.status = f"Error in 3D segmentation setup: {str(e)}"
            traceback.print_exc()
            return False

    def _on_3d_label_clicked(self, layer, event):
        """Handle click on 3D label layer to add a prompt for segmentation."""
        try:
            if event.button != 1:
                return

            coords = layer.world_to_data(event.position)
            if len(coords) == 3:
                z, y, x = map(int, coords)
            elif len(coords) == 2:
                z = int(self.viewer.dims.current_step[0])
                y, x = map(int, coords)
            else:
                self.viewer.status = (
                    f"Unexpected coordinate dimensions: {coords}"
                )
                return

            # Check if Shift key is pressed
            is_negative = "Shift" in event.modifiers
            point_label = -1 if is_negative else 1

            # Initialize a unique object ID for this click
            if not hasattr(self, "_sam2_next_obj_id"):
                self._sam2_next_obj_id = 1

            # Get current object ID (or create new one)
            label_id = self.segmentation_result[z, y, x]
            if is_negative and label_id > 0:
                # Use existing object ID for negative points
                ann_obj_id = label_id
            else:
                # Create new object for positive points on background
                ann_obj_id = self._sam2_next_obj_id
                if point_label > 0 and label_id == 0:
                    self._sam2_next_obj_id += 1

            # Find or create points layer for this object
            points_layer = None
            for layer in list(self.viewer.layers):
                if f"Points for Object {ann_obj_id}" in layer.name:
                    points_layer = layer
                    break

            if points_layer is None:
                # Create new points layer for this object
                points_layer = self.viewer.add_points(
                    np.array([[z, y, x]]),
                    name=f"Points for Object {ann_obj_id}",
                    size=10,
                    face_color="green" if point_label > 0 else "red",
                    border_color="white",
                    border_width=1,
                    opacity=0.8,
                )

                with contextlib.suppress(AttributeError, ValueError):
                    points_layer.mouse_drag_callbacks.remove(
                        self._on_points_clicked
                    )
                    points_layer.mouse_drag_callbacks.append(
                        self._on_points_clicked
                    )

                # Initialize points for this object
                if not hasattr(self, "sam2_points_by_obj"):
                    self.sam2_points_by_obj = {}
                    self.sam2_labels_by_obj = {}

                self.sam2_points_by_obj[ann_obj_id] = [[x, y]]
                self.sam2_labels_by_obj[ann_obj_id] = [point_label]
            else:
                # Add to existing points layer
                current_points = points_layer.data
                new_points = np.vstack([current_points, [z, y, x]])
                points_layer.data = new_points

                # Add to existing point lists
                if not hasattr(self, "sam2_points_by_obj"):
                    self.sam2_points_by_obj = {}
                    self.sam2_labels_by_obj = {}

                if ann_obj_id not in self.sam2_points_by_obj:
                    self.sam2_points_by_obj[ann_obj_id] = []
                    self.sam2_labels_by_obj[ann_obj_id] = []

                self.sam2_points_by_obj[ann_obj_id].append([x, y])
                self.sam2_labels_by_obj[ann_obj_id].append(point_label)

            # Perform SAM2 segmentation
            if hasattr(self, "_sam2_state") and self._sam2_state is not None:
                points = np.array(
                    self.sam2_points_by_obj[ann_obj_id], dtype=np.float32
                )
                labels = np.array(
                    self.sam2_labels_by_obj[ann_obj_id], dtype=np.int32
                )

                self.viewer.status = f"Processing object at frame {z}..."

                _, out_obj_ids, out_mask_logits = (
                    self.predictor.add_new_points_or_box(
                        inference_state=self._sam2_state,
                        frame_idx=z,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )
                )

                # Convert logits to mask and update segmentation
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()

                # Fix mask dimensions if needed
                if mask.ndim > 2:
                    mask = mask.squeeze()

                # Check mask dimensions and resize if needed
                if mask.shape != self.segmentation_result[z].shape:
                    from skimage.transform import resize

                    mask = resize(
                        mask.astype(float),
                        self.segmentation_result[z].shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(bool)

                # Apply the mask to current frame
                # For negative points, only remove from the current object
                if point_label < 0:
                    # Remove only from current object
                    self.segmentation_result[z][
                        (self.segmentation_result[z] == ann_obj_id) & mask
                    ] = 0
                else:
                    # Add to current object (only overwrite background)
                    self.segmentation_result[z][
                        mask & (self.segmentation_result[z] == 0)
                    ] = ann_obj_id

                # Automatically propagate to other frames
                self._propagate_mask_for_current_object(ann_obj_id, z)

                # Update label layer
                self._update_label_layer()

                # Update label table if needed
                if (
                    hasattr(self, "label_table_widget")
                    and self.label_table_widget is not None
                ):
                    self._populate_label_table(self.label_table_widget)

                self.viewer.status = (
                    f"Updated 3D object {ann_obj_id} across all frames"
                )
            else:
                self.viewer.status = "SAM2 3D state not initialized"

        except (
            IndexError,
            KeyError,
            ValueError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
        ) as e:
            import traceback

            self.viewer.status = f"Error in 3D click handler: {str(e)}"
            traceback.print_exc()

    def _propagate_mask_for_current_object(self, obj_id, current_frame_idx):
        """
        Propagate the mask for the current object from the given frame to all other frames.
        Uses SAM2's video propagation with proper error handling.

        Parameters:
            obj_id: The ID of the object to propagate
            current_frame_idx: The frame index where the object was identified
        """
        try:
            if not hasattr(self, "_sam2_state") or self._sam2_state is None:
                self.viewer.status = (
                    "SAM2 3D state not initialized for propagation"
                )
                return

            total_frames = self.segmentation_result.shape[0]
            self.viewer.status = f"Propagating object {obj_id} through all {total_frames} frames..."

            # Create a progress layer for visualization
            progress_layer = None
            for layer in list(self.viewer.layers):
                if "Propagation Progress" in layer.name:
                    progress_layer = layer
                    break

            if progress_layer is None:
                progress_data = np.zeros_like(
                    self.segmentation_result, dtype=float
                )
                progress_layer = self.viewer.add_image(
                    progress_data,
                    name="Propagation Progress",
                    colormap="magma",
                    opacity=0.3,
                    visible=True,
                )

            # Update current frame in the progress layer
            progress_data = progress_layer.data
            current_mask = (
                self.segmentation_result[current_frame_idx] == obj_id
            )
            progress_data[current_frame_idx] = current_mask.astype(float) * 0.8
            progress_layer.data = progress_data

            # Try to perform SAM2 propagation with error handling
            try:
                # Use torch.inference_mode() and torch.autocast to ensure consistent dtypes
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float32
                ):
                    # Attempt to run SAM2 propagation - this will iterate through all frames
                    for (
                        frame_idx,
                        object_ids,
                        mask_logits,
                    ) in self.predictor.propagate_in_video(self._sam2_state):
                        if frame_idx >= total_frames:
                            continue

                        # Find our object ID in the results
                        # obj_mask = None
                        for i, prop_obj_id in enumerate(object_ids):
                            if prop_obj_id == obj_id:
                                # Get the mask for our object
                                mask = (mask_logits[i] > 0.0).cpu().numpy()

                                # Fix dimensions if needed
                                if mask.ndim > 2:
                                    mask = mask.squeeze()

                                # Resize if needed
                                if (
                                    mask.shape
                                    != self.segmentation_result[
                                        frame_idx
                                    ].shape
                                ):
                                    from skimage.transform import resize

                                    mask = resize(
                                        mask.astype(float),
                                        self.segmentation_result[
                                            frame_idx
                                        ].shape,
                                        order=0,
                                        preserve_range=True,
                                        anti_aliasing=False,
                                    ).astype(bool)

                                # Update segmentation - only replacing background pixels
                                self.segmentation_result[frame_idx][
                                    mask
                                    & (
                                        self.segmentation_result[frame_idx]
                                        == 0
                                    )
                                ] = obj_id

                                # Update progress visualization
                                progress_data = progress_layer.data
                                progress_data[frame_idx] = (
                                    mask.astype(float) * 0.8
                                )
                                progress_layer.data = progress_data

                        # Update status occasionally
                        if frame_idx % 10 == 0:
                            self.viewer.status = f"Propagating: frame {frame_idx+1}/{total_frames}"

            except RuntimeError as e:
                # If we get a dtype mismatch or other error, the current frame's mask to other frames
                self.viewer.status = f"SAM2 propagation failed with error: {str(e)}. Falling back to alternative method."

                # Use the current frame's mask for propagation
                for frame_idx in range(total_frames):
                    if (
                        frame_idx != current_frame_idx
                    ):  # Skip current frame as it's already done
                        # Only replace background pixels with the current frame's object
                        self.segmentation_result[frame_idx][
                            current_mask
                            & (self.segmentation_result[frame_idx] == 0)
                        ] = obj_id

                        # Update progress layer
                        progress_data = progress_layer.data
                        progress_data[frame_idx] = (
                            current_mask.astype(float) * 0.5
                        )  # Different intensity to indicate fallback
                        progress_layer.data = progress_data

                    # Update status occasionally
                    if frame_idx % 10 == 0:
                        self.viewer.status = f"Fallback propagation: frame {frame_idx+1}/{total_frames}"

            # Remove progress layer after 2 seconds
            import threading

            def remove_progress():
                import time

                time.sleep(2)
                for layer in list(self.viewer.layers):
                    if "Propagation Progress" in layer.name:
                        self.viewer.layers.remove(layer)

            threading.Thread(target=remove_progress).start()

            self.viewer.status = f"Propagation of object {obj_id} complete"

        except (
            IndexError,
            ValueError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
            TypeError,
        ) as e:
            import traceback

            self.viewer.status = f"Error in propagation: {str(e)}"
            traceback.print_exc()

    def _add_3d_prompt(self, prompt_coords):
        """
        Given a 3D coordinate (x, y, z), run SAM2 video predictor to segment the object at that point,
        update the segmentation result and label layer.
        """
        if not hasattr(self, "_sam2_state") or self._sam2_state is None:
            self.viewer.status = "SAM2 3D state not initialized."
            return

        if self.predictor is None:
            self.viewer.status = "SAM2 predictor not initialized."
            return

        # Prepare prompt for SAM2: point_coords is [[x, y, t]], point_labels is [1]
        x, y, z = prompt_coords
        point_coords = np.array([[x, y, z]])
        point_labels = np.array([1])  # 1 = foreground

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            masks, scores, _ = self.predictor.predict(
                state=self._sam2_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # Pick the best mask (highest score)
        if masks is not None and len(masks) > 0:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            obj_id = self._sam2_next_obj_id
            self.segmentation_result[mask] = obj_id
            self._sam2_next_obj_id += 1
            self.viewer.status = (
                f"Added object {obj_id} at (x={x}, y={y}, z={z})"
            )
            self._update_label_layer()
        else:
            self.viewer.status = "No mask found for this prompt."

    def on_apply_propagate(self):
        """Propagate masks across the video and update the segmentation layer."""
        self.viewer.status = "Propagating masks across all frames..."
        self.viewer.window._qt_window.setCursor(Qt.WaitCursor)

        self.segmentation_result[:] = 0

        for (
            frame_idx,
            object_ids,
            mask_logits,
        ) in self.predictor.propagate_in_video(self._sam2_state):
            masks = (mask_logits > 0.0).cpu().numpy()
            if frame_idx >= self.segmentation_result.shape[0]:
                print(
                    f"Warning: frame_idx {frame_idx} out of bounds for segmentation_result with shape {self.segmentation_result.shape}"
                )
                continue
            for i, obj_id in enumerate(object_ids):
                self.segmentation_result[frame_idx][masks[i]] = obj_id
            self.viewer.status = f"Propagating: frame {frame_idx+1}"

        self._update_label_layer()
        self.viewer.status = "Propagation complete!"
        self.viewer.window._qt_window.setCursor(Qt.ArrowCursor)

    def _update_label_layer(self):
        """Update the label layer in the viewer."""
        # Remove existing label layer if it exists
        for layer in list(self.viewer.layers):
            if isinstance(layer, Labels) and "Segmentation" in layer.name:
                self.viewer.layers.remove(layer)

        # Add label layer to viewer
        self.label_layer = self.viewer.add_labels(
            self.segmentation_result,
            name=f"Segmentation ({os.path.basename(self.images[self.current_index])})",
            opacity=0.7,
        )

        # Create points layer for interaction if it doesn't exist
        points_layer = None
        for layer in list(self.viewer.layers):
            if "Points" in layer.name:
                points_layer = layer
                break

        if points_layer is None:
            # Initialize an empty points layer
            points_layer = self.viewer.add_points(
                np.zeros((0, 2 if not self.use_3d else 3)),
                name="Points (Click to Add)",
                size=10,
                face_color="green",
                border_color="white",
                border_width=1,
                opacity=0.8,
            )

            with contextlib.suppress(AttributeError, ValueError):
                points_layer.mouse_drag_callbacks.remove(
                    self._on_points_clicked
                )
                points_layer.mouse_drag_callbacks.append(
                    self._on_points_clicked
                )

            # Connect points layer mouse click event
            points_layer.mouse_drag_callbacks.append(self._on_points_clicked)

        # Make the points layer active to encourage interaction with it
        self.viewer.layers.selection.active = points_layer

        # Update status
        n_labels = len(np.unique(self.segmentation_result)) - (
            1 if 0 in np.unique(self.segmentation_result) else 0
        )
        self.viewer.status = f"Loaded image {self.current_index + 1}/{len(self.images)} - Found {n_labels} segments"

    def _on_points_clicked(self, layer, event):
        """Handle clicks on the points layer for adding/removing points."""
        try:
            # Only process clicks, not drags
            if event.type != "mouse_press":
                return

            # Get coordinates of mouse click
            coords = np.round(event.position).astype(int)

            # Check if Shift is pressed for negative points
            is_negative = "Shift" in event.modifiers
            point_label = -1 if is_negative else 1

            # Handle 2D vs 3D coordinates
            if self.use_3d:
                if len(coords) == 3:
                    t, y, x = map(int, coords)
                elif len(coords) == 2:
                    t = int(self.viewer.dims.current_step[0])
                    y, x = map(int, coords)
                else:
                    self.viewer.status = (
                        f"Unexpected coordinate dimensions: {coords}"
                    )
                    return

                # Add point to the layer immediately for visual feedback
                new_point = np.array([[t, y, x]])
                if len(layer.data) == 0:
                    layer.data = new_point
                else:
                    layer.data = np.vstack([layer.data, new_point])

                # Update point colors
                colors = layer.face_color
                if isinstance(colors, list):
                    colors.append("red" if is_negative else "green")
                else:
                    n_points = len(layer.data)
                    colors = ["green"] * (n_points - 1)
                    colors.append("red" if is_negative else "green")
                layer.face_color = colors

                # Get the object ID
                # If clicking on existing segmentation with negative point
                label_id = self.segmentation_result[t, y, x]
                if is_negative and label_id > 0:
                    obj_id = label_id
                else:
                    # For new objects or negative on background
                    if not hasattr(self, "_sam2_next_obj_id"):
                        self._sam2_next_obj_id = 1
                    obj_id = self._sam2_next_obj_id
                    if point_label > 0 and label_id == 0:
                        self._sam2_next_obj_id += 1

                # Store point information
                if not hasattr(self, "points_data"):
                    self.points_data = {}
                    self.points_labels = {}

                if obj_id not in self.points_data:
                    self.points_data[obj_id] = []
                    self.points_labels[obj_id] = []

                self.points_data[obj_id].append(
                    [x, y]
                )  # Note: SAM2 expects [x,y] format
                self.points_labels[obj_id].append(point_label)

                # Perform segmentation
                if (
                    hasattr(self, "_sam2_state")
                    and self._sam2_state is not None
                ):
                    # Prepare points
                    points = np.array(
                        self.points_data[obj_id], dtype=np.float32
                    )
                    labels = np.array(
                        self.points_labels[obj_id], dtype=np.int32
                    )

                    # Create progress layer for visual feedback
                    progress_layer = None
                    for existing_layer in self.viewer.layers:
                        if "Propagation Progress" in existing_layer.name:
                            progress_layer = existing_layer
                            break

                    if progress_layer is None:
                        progress_data = np.zeros_like(self.segmentation_result)
                        progress_layer = self.viewer.add_image(
                            progress_data,
                            name="Propagation Progress",
                            colormap="magma",
                            opacity=0.5,
                            visible=True,
                        )

                    # First update the current frame immediately
                    self.viewer.status = f"Processing object at frame {t}..."

                    # Run SAM2 on current frame
                    _, out_obj_ids, out_mask_logits = (
                        self.predictor.add_new_points_or_box(
                            inference_state=self._sam2_state,
                            frame_idx=t,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )
                    )

                    # Update current frame
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    if mask.ndim > 2:
                        mask = mask.squeeze()

                    # Resize if needed
                    if mask.shape != self.segmentation_result[t].shape:
                        from skimage.transform import resize

                        mask = resize(
                            mask.astype(float),
                            self.segmentation_result[t].shape,
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        ).astype(bool)

                    # Update segmentation for this frame
                    if point_label < 0:
                        # For negative points, only remove from this object
                        self.segmentation_result[t][
                            (self.segmentation_result[t] == obj_id) & mask
                        ] = 0
                    else:
                        # For positive points, only replace background
                        self.segmentation_result[t][
                            mask & (self.segmentation_result[t] == 0)
                        ] = obj_id

                    # Update progress layer for this frame
                    progress_data = progress_layer.data
                    progress_data[t] = (
                        mask.astype(float) * 0.5
                    )  # Highlight current frame
                    progress_layer.data = progress_data

                    # Now propagate to all frames with visual feedback
                    self.viewer.status = "Propagating to all frames..."

                    # Run propagation
                    frame_count = self.segmentation_result.shape[0]
                    for (
                        frame_idx,
                        prop_obj_ids,
                        mask_logits,
                    ) in self.predictor.propagate_in_video(self._sam2_state):
                        if frame_idx >= frame_count:
                            continue

                        # Find our object
                        obj_mask = None
                        for i, prop_obj_id in enumerate(prop_obj_ids):
                            if prop_obj_id == obj_id:
                                obj_mask = (mask_logits[i] > 0.0).cpu().numpy()
                                if obj_mask.ndim > 2:
                                    obj_mask = obj_mask.squeeze()

                                # Resize if needed
                                if (
                                    obj_mask.shape
                                    != self.segmentation_result[
                                        frame_idx
                                    ].shape
                                ):
                                    obj_mask = resize(
                                        obj_mask.astype(float),
                                        self.segmentation_result[
                                            frame_idx
                                        ].shape,
                                        order=0,
                                        preserve_range=True,
                                        anti_aliasing=False,
                                    ).astype(bool)

                                # Update segmentation
                                self.segmentation_result[frame_idx][
                                    obj_mask
                                    & (
                                        self.segmentation_result[frame_idx]
                                        == 0
                                    )
                                ] = obj_id

                                # Update progress visualization
                                progress_data = progress_layer.data
                                progress_data[frame_idx] = (
                                    obj_mask.astype(float) * 0.8
                                )  # Show as processed
                                progress_layer.data = progress_data

                                # Update status
                                if frame_idx % 5 == 0:
                                    self.viewer.status = f"Propagating: frame {frame_idx+1}/{frame_count}"
                                    # Remove the viewer.update() call as it's causing errors

                    # Process any missing frames
                    processed_frames = set(range(frame_count))
                    for frame_idx in range(frame_count):
                        if (
                            progress_data[frame_idx].max() == 0
                        ):  # Frame not processed yet
                            # Use nearest processed frame's mask
                            nearest_idx = min(
                                processed_frames,
                                key=lambda x: abs(x - frame_idx),
                            )
                            if progress_data[nearest_idx].max() > 0:
                                self.segmentation_result[frame_idx][
                                    (self.segmentation_result[frame_idx] == 0)
                                    & (
                                        self.segmentation_result[nearest_idx]
                                        == obj_id
                                    )
                                ] = obj_id

                                # Update progress visualization
                                progress_data[frame_idx] = (
                                    progress_data[nearest_idx] * 0.6
                                )  # Mark as copied

                    # Final update of progress layer
                    progress_layer.data = progress_data

                    # Remove progress layer after 2 seconds
                    import threading

                    def remove_progress():
                        import time

                        time.sleep(2)
                        for layer in list(self.viewer.layers):
                            if "Propagation Progress" in layer.name:
                                self.viewer.layers.remove(layer)

                    threading.Thread(target=remove_progress).start()

                    # Update UI
                    self._update_label_layer()
                    if (
                        hasattr(self, "label_table_widget")
                        and self.label_table_widget is not None
                    ):
                        self._populate_label_table(self.label_table_widget)

                    self.viewer.status = f"Object {obj_id} segmented and propagated to all frames"

            else:
                # 2D case
                if len(coords) == 2:
                    y, x = map(int, coords)
                else:
                    self.viewer.status = (
                        f"Unexpected coordinate dimensions: {coords}"
                    )
                    return

                # Add point to the layer immediately for visual feedback
                new_point = np.array([[y, x]])
                if len(layer.data) == 0:
                    layer.data = new_point
                else:
                    layer.data = np.vstack([layer.data, new_point])

                # Update point colors
                colors = layer.face_color
                if isinstance(colors, list):
                    colors.append("red" if is_negative else "green")
                else:
                    n_points = len(layer.data)
                    colors = ["green"] * (n_points - 1)
                    colors.append("red" if is_negative else "green")
                layer.face_color = colors

                # Get object ID
                label_id = self.segmentation_result[y, x]
                if is_negative and label_id > 0:
                    obj_id = label_id
                else:
                    if not hasattr(self, "next_obj_id"):
                        self.next_obj_id = 1
                    obj_id = self.next_obj_id
                    if point_label > 0 and label_id == 0:
                        self.next_obj_id += 1

                # Store point information
                if not hasattr(self, "obj_points"):
                    self.obj_points = {}
                    self.obj_labels = {}

                if obj_id not in self.obj_points:
                    self.obj_points[obj_id] = []
                    self.obj_labels[obj_id] = []

                self.obj_points[obj_id].append(
                    [x, y]
                )  # SAM2 expects [x,y] format
                self.obj_labels[obj_id].append(point_label)

                # Perform segmentation
                if hasattr(self, "predictor") and self.predictor is not None:
                    # Make sure image is loaded
                    if self.current_image_for_segmentation is None:
                        self.viewer.status = "No image loaded for segmentation"
                        return

                    # Prepare image for SAM2
                    image = self.current_image_for_segmentation
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)
                    elif len(image.shape) == 3 and image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=2)
                    elif len(image.shape) == 3 and image.shape[2] > 3:
                        image = image[:, :, :3]

                    if image.dtype != np.uint8:
                        image = (image / np.max(image) * 255).astype(np.uint8)

                    # Set the image in the predictor
                    self.predictor.set_image(image)

                    # Use only points for current object
                    points = np.array(
                        self.obj_points[obj_id], dtype=np.float32
                    )
                    labels = np.array(self.obj_labels[obj_id], dtype=np.int32)

                    self.viewer.status = f"Segmenting object {obj_id} with {len(points)} points..."

                    with torch.inference_mode(), torch.autocast("cuda"):
                        masks, scores, _ = self.predictor.predict(
                            point_coords=points,
                            point_labels=labels,
                            multimask_output=True,
                        )

                        # Get best mask
                        if len(masks) > 0:
                            best_mask = masks[0]

                            # Update segmentation result
                            if (
                                best_mask.shape
                                != self.segmentation_result.shape
                            ):
                                from skimage.transform import resize

                                best_mask = resize(
                                    best_mask.astype(float),
                                    self.segmentation_result.shape,
                                    order=0,
                                    preserve_range=True,
                                    anti_aliasing=False,
                                ).astype(bool)

                            # Apply mask based on point type
                            if point_label < 0:
                                # Remove only from current object
                                mask_condition = np.logical_and(
                                    self.segmentation_result == obj_id,
                                    best_mask,
                                )
                                self.segmentation_result[mask_condition] = 0
                            else:
                                # Add to current object (only overwrite background)
                                mask_condition = np.logical_and(
                                    best_mask, (self.segmentation_result == 0)
                                )
                                self.segmentation_result[mask_condition] = (
                                    obj_id
                                )

                            # Update label info
                            area = np.sum(self.segmentation_result == obj_id)
                            y_indices, x_indices = np.where(
                                self.segmentation_result == obj_id
                            )
                            center_y = (
                                np.mean(y_indices) if len(y_indices) > 0 else 0
                            )
                            center_x = (
                                np.mean(x_indices) if len(x_indices) > 0 else 0
                            )

                            self.label_info[obj_id] = {
                                "area": area,
                                "center_y": center_y,
                                "center_x": center_x,
                                "score": float(scores[0]),
                            }

                            self.viewer.status = f"Updated object {obj_id}"
                        else:
                            self.viewer.status = "No valid mask produced"

                    # Update the UI
                    self._update_label_layer()
                    if (
                        hasattr(self, "label_table_widget")
                        and self.label_table_widget is not None
                    ):
                        self._populate_label_table(self.label_table_widget)

        except (
            IndexError,
            KeyError,
            ValueError,
            RuntimeError,
            TypeError,
        ) as e:
            import traceback

            self.viewer.status = f"Error in points handling: {str(e)}"
            traceback.print_exc()

    def _on_label_clicked(self, layer, event):
        """Handle label selection and user prompts on mouse click."""
        try:
            # Only process clicks, not drags
            if event.type != "mouse_press":
                return

            # Get coordinates of mouse click
            coords = np.round(event.position).astype(int)

            # Check if Shift is pressed (negative point)
            is_negative = "Shift" in event.modifiers
            point_label = -1 if is_negative else 1

            # For 2D data
            if not self.use_3d:
                if len(coords) == 2:
                    y, x = map(int, coords)
                else:
                    self.viewer.status = (
                        f"Unexpected coordinate dimensions: {coords}"
                    )
                    return

                # Check if within image bounds
                shape = self.segmentation_result.shape
                if y < 0 or x < 0 or y >= shape[0] or x >= shape[1]:
                    self.viewer.status = "Click is outside image bounds"
                    return

                # Get the label ID at the clicked position
                label_id = self.segmentation_result[y, x]

                # Initialize a unique object ID for this click (if needed)
                if not hasattr(self, "next_obj_id"):
                    # Start with highest existing ID + 1
                    if self.segmentation_result.max() > 0:
                        self.next_obj_id = (
                            int(self.segmentation_result.max()) + 1
                        )
                    else:
                        self.next_obj_id = 1

                # If clicking on background or using negative click, handle segmentation
                if label_id == 0 or is_negative:
                    # Find or create points layer for the current object we're working on
                    current_obj_id = None

                    # If negative point on existing label, use that label's ID
                    if is_negative and label_id > 0:
                        current_obj_id = label_id
                    # For positive clicks on background, create a new object
                    elif point_label > 0 and label_id == 0:
                        current_obj_id = self.next_obj_id
                        self.next_obj_id += 1
                    # For negative on background, try to find most recent object
                    elif point_label < 0 and label_id == 0:
                        # Use most recently created object if available
                        if hasattr(self, "obj_points") and self.obj_points:
                            current_obj_id = max(self.obj_points.keys())
                        else:
                            self.viewer.status = "No existing object to modify with negative point"
                            return

                    if current_obj_id is None:
                        self.viewer.status = (
                            "Could not determine which object to modify"
                        )
                        return

                    # Find or create points layer for this object
                    points_layer = None
                    for layer in list(self.viewer.layers):
                        if f"Points for Object {current_obj_id}" in layer.name:
                            points_layer = layer
                            break

                    # Initialize object tracking if needed
                    if not hasattr(self, "obj_points"):
                        self.obj_points = {}
                        self.obj_labels = {}

                    if current_obj_id not in self.obj_points:
                        self.obj_points[current_obj_id] = []
                        self.obj_labels[current_obj_id] = []

                    # Create or update points layer for this object
                    if points_layer is None:
                        # First point for this object
                        points_layer = self.viewer.add_points(
                            np.array([[y, x]]),
                            name=f"Points for Object {current_obj_id}",
                            size=10,
                            face_color=["green" if point_label > 0 else "red"],
                            border_color="white",
                            border_width=1,
                            opacity=0.8,
                        )
                        with contextlib.suppress(AttributeError, ValueError):
                            points_layer.mouse_drag_callbacks.remove(
                                self._on_points_clicked
                            )
                            points_layer.mouse_drag_callbacks.append(
                                self._on_points_clicked
                            )

                        self.obj_points[current_obj_id] = [[x, y]]
                        self.obj_labels[current_obj_id] = [point_label]
                    else:
                        # Add point to existing layer
                        current_points = points_layer.data
                        current_colors = points_layer.face_color

                        # Add new point
                        new_points = np.vstack([current_points, [y, x]])
                        new_color = "green" if point_label > 0 else "red"

                        # Update points layer
                        points_layer.data = new_points

                        # Update colors
                        if isinstance(current_colors, list):
                            current_colors.append(new_color)
                            points_layer.face_color = current_colors
                        else:
                            # If it's an array, create a list of colors
                            colors = []
                            for i in range(len(new_points)):
                                if i < len(current_points):
                                    colors.append(
                                        "green" if point_label > 0 else "red"
                                    )
                                else:
                                    colors.append(new_color)
                            points_layer.face_color = colors

                        # Update object tracking
                        self.obj_points[current_obj_id].append([x, y])
                        self.obj_labels[current_obj_id].append(point_label)

                    # Now do the actual segmentation using SAM2
                    if (
                        hasattr(self, "predictor")
                        and self.predictor is not None
                    ):
                        try:
                            # Make sure image is loaded
                            if self.current_image_for_segmentation is None:
                                self.viewer.status = (
                                    "No image loaded for segmentation"
                                )
                                return

                            # Prepare image for SAM2
                            image = self.current_image_for_segmentation
                            if len(image.shape) == 2:
                                image = np.stack([image] * 3, axis=-1)
                            elif len(image.shape) == 3 and image.shape[2] == 1:
                                image = np.concatenate([image] * 3, axis=2)
                            elif len(image.shape) == 3 and image.shape[2] > 3:
                                image = image[:, :, :3]

                            if image.dtype != np.uint8:
                                image = (image / np.max(image) * 255).astype(
                                    np.uint8
                                )

                            # Set the image in the predictor
                            self.predictor.set_image(image)

                            # Only use the points for the current object being segmented
                            points = np.array(
                                self.obj_points[current_obj_id],
                                dtype=np.float32,
                            )
                            labels = np.array(
                                self.obj_labels[current_obj_id], dtype=np.int32
                            )

                            self.viewer.status = f"Segmenting object {current_obj_id} with {len(points)} points..."

                            with torch.inference_mode(), torch.autocast(
                                "cuda"
                            ):
                                masks, scores, _ = self.predictor.predict(
                                    point_coords=points,
                                    point_labels=labels,
                                    multimask_output=True,
                                )

                                # Get best mask
                                if len(masks) > 0:
                                    best_mask = masks[0]

                                    # Update segmentation result
                                    if (
                                        best_mask.shape
                                        != self.segmentation_result.shape
                                    ):
                                        from skimage.transform import resize

                                        best_mask = resize(
                                            best_mask.astype(float),
                                            self.segmentation_result.shape,
                                            order=0,
                                            preserve_range=True,
                                            anti_aliasing=False,
                                        ).astype(bool)

                                    # CRITICAL FIX: For negative points, only remove from this object's mask
                                    # For positive points, add to this object's mask without removing other objects
                                    if point_label < 0:
                                        # Remove only from current object's mask
                                        self.segmentation_result[
                                            (
                                                self.segmentation_result
                                                == current_obj_id
                                            )
                                            & best_mask
                                        ] = 0
                                    else:
                                        # Add to current object's mask without affecting other objects
                                        # Only overwrite background (value 0)
                                        self.segmentation_result[
                                            best_mask
                                            & (self.segmentation_result == 0)
                                        ] = current_obj_id

                                    # Update label info
                                    area = np.sum(
                                        self.segmentation_result
                                        == current_obj_id
                                    )
                                    y_indices, x_indices = np.where(
                                        self.segmentation_result
                                        == current_obj_id
                                    )
                                    center_y = (
                                        np.mean(y_indices)
                                        if len(y_indices) > 0
                                        else 0
                                    )
                                    center_x = (
                                        np.mean(x_indices)
                                        if len(x_indices) > 0
                                        else 0
                                    )

                                    self.label_info[current_obj_id] = {
                                        "area": area,
                                        "center_y": center_y,
                                        "center_x": center_x,
                                        "score": float(scores[0]),
                                    }

                                    self.viewer.status = (
                                        f"Updated object {current_obj_id}"
                                    )
                                else:
                                    self.viewer.status = (
                                        "No valid mask produced"
                                    )

                            # Update the UI
                            self._update_label_layer()
                            if (
                                hasattr(self, "label_table_widget")
                                and self.label_table_widget is not None
                            ):
                                self._populate_label_table(
                                    self.label_table_widget
                                )

                        except (
                            IndexError,
                            KeyError,
                            ValueError,
                            AttributeError,
                            TypeError,
                        ) as e:
                            import traceback

                            self.viewer.status = (
                                f"Error in SAM2 processing: {str(e)}"
                            )
                            traceback.print_exc()

                # If clicking on an existing label, toggle selection
                elif label_id > 0:
                    # Toggle the label selection
                    if label_id in self.selected_labels:
                        self.selected_labels.remove(label_id)
                        self.viewer.status = f"Deselected label ID: {label_id} | Selected labels: {self.selected_labels}"
                    else:
                        self.selected_labels.add(label_id)
                        self.viewer.status = f"Selected label ID: {label_id} | Selected labels: {self.selected_labels}"

                    # Update table and preview
                    self._update_label_table()
                    self.preview_crop()

            # 3D case (handle differently)
            else:
                if len(coords) == 3:
                    t, y, x = map(int, coords)
                elif len(coords) == 2:
                    t = int(self.viewer.dims.current_step[0])
                    y, x = map(int, coords)
                else:
                    self.viewer.status = (
                        f"Unexpected coordinate dimensions: {coords}"
                    )
                    return

                # Check if within bounds
                shape = self.segmentation_result.shape
                if (
                    t < 0
                    or t >= shape[0]
                    or y < 0
                    or y >= shape[1]
                    or x < 0
                    or x >= shape[2]
                ):
                    self.viewer.status = "Click is outside volume bounds"
                    return

                # Get the label ID at the clicked position
                label_id = self.segmentation_result[t, y, x]

                # If background or shift is pressed, handle in _on_3d_label_clicked
                if label_id == 0 or is_negative:
                    # This will be handled by _on_3d_label_clicked already attached
                    pass
                # If clicking on an existing label, handle selection
                elif label_id > 0:
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

        except (
            IndexError,
            KeyError,
            ValueError,
            AttributeError,
            TypeError,
        ) as e:
            import traceback

            self.viewer.status = f"Error in click handling: {str(e)}"
            traceback.print_exc()

    def _add_point_marker(self, coords, label_type):
        """Add a visible marker for where the user clicked."""
        # Remove previous point markers
        for layer in list(self.viewer.layers):
            if "Point Prompt" in layer.name:
                self.viewer.layers.remove(layer)

        # Create points layer
        color = (
            "red" if label_type < 0 else "green"
        )  # Red for negative, green for positive
        self.viewer.add_points(
            [coords],
            name="Point Prompt",
            size=10,
            face_color=color,
            edge_color="white",
            edge_width=2,
            opacity=0.8,
        )

        with contextlib.suppress(AttributeError, ValueError):
            self.points_layer.mouse_drag_callbacks.remove(
                self._on_points_clicked
            )
            self.points_layer.mouse_drag_callbacks.append(
                self._on_points_clicked
            )

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
        try:
            # Get all unique non-zero labels from the segmentation result safely
            if self.segmentation_result is None:
                # No segmentation yet
                table.setRowCount(0)
                self.viewer.status = "No segmentation available"
                return

            # Get unique labels, safely handling None values
            unique_labels = []
            for val in np.unique(self.segmentation_result):
                if val is not None and val > 0:
                    unique_labels.append(val)

            if len(unique_labels) == 0:
                table.setRowCount(0)
                self.viewer.status = "No labeled objects found"
                return

            # Set row count
            table.setRowCount(len(unique_labels))

            # Fill in label info for any missing labels
            for label_id in unique_labels:
                if label_id not in self.label_info:
                    # Calculate basic info for this label
                    mask = self.segmentation_result == label_id
                    area = np.sum(mask)

                    # Add info to label_info dictionary
                    self.label_info[label_id] = {
                        "area": area,
                        "score": 1.0,  # Default score
                    }

            # Fill table with data
            for row, label_id in enumerate(unique_labels):
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

        except (KeyError, TypeError, ValueError, AttributeError) as e:
            import traceback

            self.viewer.status = f"Error populating table: {str(e)}"
            traceback.print_exc()
            # Set empty table as fallback
            table.setRowCount(0)

    def _update_label_table(self):
        """Update the label selection table if it exists."""
        if self.label_table_widget is None:
            return

        # Block signals during update
        self.label_table_widget.blockSignals(True)

        # Completely repopulate the table to ensure it's up to date
        self._populate_label_table(self.label_table_widget)

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
            if self.use_3d:
                # For 3D data
                mask = np.zeros_like(self.segmentation_result, dtype=bool)
                for label_id in label_ids:
                    mask |= self.segmentation_result == label_id

                # Apply mask
                preview_image = image.copy()
                preview_image[~mask] = 0
            else:
                # For 2D data
                mask = np.zeros_like(self.segmentation_result, dtype=bool)
                for label_id in label_ids:
                    mask |= self.segmentation_result == label_id

                # Apply mask
                if len(image.shape) == 2:
                    preview_image = image.copy()
                    preview_image[~mask] = 0
                else:
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
            if self.use_3d:
                # For 3D data, create a 3D mask
                mask = np.zeros_like(self.segmentation_result, dtype=bool)
                for label_id in self.selected_labels:
                    mask |= self.segmentation_result == label_id

                # Apply mask to image (set everything outside mask to 0)
                cropped_image = image.copy()
                cropped_image[~mask] = 0

                # Save label image with same dimensions as original
                label_image = np.zeros_like(
                    self.segmentation_result, dtype=np.uint32
                )
                for label_id in self.selected_labels:
                    label_image[self.segmentation_result == label_id] = (
                        label_id
                    )
            else:
                # For 2D data, handle as before
                mask = np.zeros_like(self.segmentation_result, dtype=bool)
                for label_id in self.selected_labels:
                    mask |= self.segmentation_result == label_id

                # Apply mask to image (set everything outside mask to 0)
                if len(image.shape) == 2:
                    # Grayscale image
                    cropped_image = image.copy()
                    cropped_image[~mask] = 0

                    # Create label image with same dimensions
                    label_image = np.zeros_like(
                        self.segmentation_result, dtype=np.uint32
                    )
                    for label_id in self.selected_labels:
                        label_image[self.segmentation_result == label_id] = (
                            label_id
                        )
                else:
                    # Color image - mask must be expanded to match channel dimension
                    cropped_image = image.copy()
                    for c in range(cropped_image.shape[2]):
                        cropped_image[:, :, c][~mask] = 0

                    # Create label image with 2D dimensions (without channels)
                    label_image = np.zeros_like(
                        self.segmentation_result, dtype=np.uint32
                    )
                    for label_id in self.selected_labels:
                        label_image[self.segmentation_result == label_id] = (
                            label_id
                        )

            # Save cropped image
            image_path = self.images[self.current_index]
            base_name, ext = os.path.splitext(image_path)
            label_str = "_".join(
                str(lid) for lid in sorted(self.selected_labels)
            )
            output_path = f"{base_name}_cropped_{label_str}.tif"

            # Save using tifffile with explicit parameters for best compatibility
            imwrite(output_path, cropped_image, compression="zlib")
            self.viewer.status = f"Saved cropped image to {output_path}"

            # Save the label image with exact same dimensions as original
            label_output_path = f"{base_name}_labels_{label_str}.tif"
            imwrite(label_output_path, label_image, compression="zlib")
            self.viewer.status += f"\nSaved label mask to {label_output_path}"

            # Make sure the segmentation layer is active again
            if self.label_layer is not None:
                self.viewer.layers.selection.active = self.label_layer

            return True

        except (Exception, ValueError) as e:
            self.viewer.status = f"Error cropping image: {str(e)}"
            return False


def create_crop_widget(processor):
    """Create the crop control widget."""
    crop_widget = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(10)
    layout.setContentsMargins(10, 10, 10, 10)

    # Instructions
    dimension_type = "3D (TYX/ZYX)" if processor.use_3d else "2D (YX)"
    instructions_label = QLabel(
        f"<b>Processing {dimension_type} data</b><br><br>"
        "To create/edit objects:<br>"
        "1. <b>Click on the POINTS layer</b> to add positive points<br>"
        "2. Use Shift+click for negative points to refine segmentation<br>"
        "3. Click on existing objects in the Segmentation layer to select them<br>"
        "4. Press 'Crop' to save the selected objects to disk"
    )
    instructions_label.setWordWrap(True)
    layout.addWidget(instructions_label)

    # Add a button to ensure points layer is active
    activate_button = QPushButton("Make Points Layer Active")
    activate_button.clicked.connect(
        lambda: processor._ensure_points_layer_active()
    )
    layout.addWidget(activate_button)

    # Add a "Clear Points" button to reset prompts
    clear_points_button = QPushButton("Clear Points")
    layout.addWidget(clear_points_button)

    # Create label table
    label_table = processor.create_label_table(crop_widget)
    label_table.setMinimumHeight(150)
    label_table.setMaximumHeight(300)
    layout.addWidget(label_table)

    # Selection buttons
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
        "Ready to process images. Click on POINTS layer to add segmentation points."
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
        layout.insertWidget(3, label_table)  # Insert after clear points button
        return label_table

    # Add helper method to ensure points layer is active
    def _ensure_points_layer_active():
        points_layer = None
        for layer in list(processor.viewer.layers):
            if "Points" in layer.name:
                points_layer = layer
                break

        if points_layer is not None:
            processor.viewer.layers.selection.active = points_layer
            status_label.setText(
                "Points layer is now active - click to add points"
            )
        else:
            status_label.setText(
                "No points layer found. Please load an image first."
            )

    processor._ensure_points_layer_active = _ensure_points_layer_active

    # Connect button signals
    def on_clear_points_clicked():
        # Remove all point layers
        for layer in list(processor.viewer.layers):
            if "Points" in layer.name:
                processor.viewer.layers.remove(layer)

        # Reset point tracking attributes
        if hasattr(processor, "points_data"):
            processor.points_data = {}
            processor.points_labels = {}

        if hasattr(processor, "obj_points"):
            processor.obj_points = {}
            processor.obj_labels = {}

        # Re-create empty points layer
        processor._update_label_layer()
        processor._ensure_points_layer_active()

        status_label.setText(
            "Cleared all points. Click on Points layer to add new points."
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
        # Clear points before moving to next image
        on_clear_points_clicked()

        if not processor.next_image():
            next_button.setEnabled(False)
        else:
            prev_button.setEnabled(True)
            replace_table_widget()
            status_label.setText(
                f"Showing image {processor.current_index + 1}/{len(processor.images)}"
            )
            processor._ensure_points_layer_active()

    def on_prev_clicked():
        # Clear points before moving to previous image
        on_clear_points_clicked()

        if not processor.previous_image():
            prev_button.setEnabled(False)
        else:
            next_button.setEnabled(True)
            replace_table_widget()
            status_label.setText(
                f"Showing image {processor.current_index + 1}/{len(processor.images)}"
            )
            processor._ensure_points_layer_active()

    clear_points_button.clicked.connect(on_clear_points_clicked)
    select_all_button.clicked.connect(on_select_all_clicked)
    clear_selection_button.clicked.connect(on_clear_selection_clicked)
    crop_button.clicked.connect(on_crop_clicked)
    next_button.clicked.connect(on_next_clicked)
    prev_button.clicked.connect(on_prev_clicked)
    activate_button.clicked.connect(_ensure_points_layer_active)

    return crop_widget


@magicgui(
    call_button="Start Batch Crop Anything",
    folder_path={"label": "Folder Path", "widget_type": "LineEdit"},
    data_dimensions={
        "label": "Data Dimensions",
        "choices": ["YX (2D)", "TYX/ZYX (3D)"],
    },
)
def batch_crop_anything(
    folder_path: str,
    data_dimensions: str,
    viewer: Viewer = None,
):
    """MagicGUI widget for starting Batch Crop Anything using SAM2."""
    # Check if SAM2 is available
    try:
        import importlib.util

        sam2_spec = importlib.util.find_spec("sam2")
        if sam2_spec is None:
            QMessageBox.critical(
                None,
                "Missing Dependency",
                "SAM2 not found. Please follow installation instructions at:\n"
                "https://github.com/MercaderLabAnatomy/napari-tmidas?tab=readme-ov-file#dependencies\n",
            )
            return
    except ImportError:
        QMessageBox.critical(
            None,
            "Missing Dependency",
            "SAM2 package cannot be imported. Please follow installation instructions at\n"
            "https://github.com/MercaderLabAnatomy/napari-tmidas?tab=readme-ov-file#dependencies",
        )
        return

    # Initialize processor with the selected dimensions mode
    use_3d = "TYX/ZYX" in data_dimensions
    processor = BatchCropAnything(viewer, use_3d=use_3d)
    processor.load_images(folder_path)

    # Create UI
    crop_widget = create_crop_widget(processor)

    # Wrap the widget in a scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidget(crop_widget)
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QScrollArea.NoFrame)
    scroll_area.setMinimumHeight(500)

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

"""
Batch Label Inspection for Napari
---------------------------------
This module provides a widget for Napari that allows users to inspect image-label pairs in a folder.
The widget loads image-label pairs from a folder and displays them in the Napari viewer.
Users can make and save changes to the labels, and proceed to the next pair.


"""

import os
import sys

from magicgui import magicgui
from napari.layers import Labels
from napari.viewer import Viewer
from qtpy.QtWidgets import QFileDialog, QPushButton
from skimage.io import imread  # , imsave

sys.path.append("src/napari_tmidas")


class LabelInspector:
    def __init__(self, viewer: Viewer):
        self.viewer = viewer
        self.image_label_pairs = []
        self.current_index = 0

    def load_image_label_pairs(self, folder_path: str, label_suffix: str):
        """
        Load image-label pairs from a folder.
        Finds label files with the given suffix and matches them with their corresponding image files.
        """
        files = os.listdir(folder_path)
        label_files = [file for file in files if file.endswith(label_suffix)]

        # Extract the file extension (e.g., .tif)
        file_extension = (
            os.path.splitext(label_suffix)[-1] if "." in label_suffix else ""
        )

        # Modified matching logic
        self.image_label_pairs = []
        for lbl in label_files:
            # Remove the label suffix to get the base name
            label_prefix = lbl[: -len(label_suffix)]

            # Potential corresponding image file
            img = f"{label_prefix}{file_extension}"
            img_path = os.path.join(folder_path, img)

            # Check if the image file exists
            if os.path.exists(img_path):
                self.image_label_pairs.append(
                    (
                        img_path,
                        os.path.join(folder_path, lbl),
                    )
                )
                continue

            # If not found, try finding any file that starts with the base name
            potential_images = [
                file
                for file in files
                if file.startswith(label_prefix)
                and file.endswith(file_extension)
                and file != lbl
            ]

            if potential_images:
                # Use the first matching image
                self.image_label_pairs.append(
                    (
                        os.path.join(folder_path, potential_images[0]),
                        os.path.join(folder_path, lbl),
                    )
                )

        if not self.image_label_pairs:
            self.viewer.status = "No matching image-label pairs found."
            return

        self.viewer.status = (
            f"Found {len(self.image_label_pairs)} image-label pairs."
        )
        self.current_index = 0
        self._load_current_pair()

    def _load_current_pair(self):
        """
        Load the current image-label pair into the Napari viewer.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to inspect."
            return

        image_path, label_path = self.image_label_pairs[self.current_index]
        image = imread(image_path)
        label_image = imread(label_path)

        # Clear existing layers
        self.viewer.layers.clear()

        # Add the new layers
        self.viewer.add_image(
            image, name=f"Image ({os.path.basename(image_path)})"
        )
        self.viewer.add_labels(
            label_image, name=f"Labels ({os.path.basename(label_path)})"
        )

    def save_current_labels(self):
        """
        Save the current labels back to the original file.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to save."
            return

        _, label_path = self.image_label_pairs[self.current_index]

        # Find the labels layer in the viewer
        labels_layer = next(
            (
                layer
                for layer in self.viewer.layers
                if isinstance(layer, Labels)
            ),
            None,
        )

        if labels_layer is None:
            self.viewer.status = "No labels found."
            return

        # Save the labels layer data to the original file path
        # imsave(label_path, labels_layer.data.astype("uint32"))
        labels_layer.save(label_path)
        self.viewer.status = f"Saved labels to {label_path}."

    def next_pair(self):
        """
        Save changes and proceed to the next image-label pair.
        """
        if not self.image_label_pairs:
            self.viewer.status = "No pairs to inspect."
            return

        # Save current labels before proceeding
        self.save_current_labels()

        # Check if we're already at the last pair
        if self.current_index >= len(self.image_label_pairs) - 1:
            self.viewer.status = (
                "No more pairs to inspect. Inspection complete."
            )
            # should also clear the viewer
            self.viewer.layers.clear()
            return False  # Return False to indicate we're at the end

        # Move to the next pair
        self.current_index += 1

        # Load the next pair
        self._load_current_pair()
        return (
            True  # Return True to indicate successful navigation to next pair
        )


@magicgui(
    call_button="Start Label Inspection",
    folder_path={"label": "Folder Path", "widget_type": "LineEdit"},
    label_suffix={"label": "Label Suffix (e.g., _otsu_labels.tif)"},
)
def label_inspector(
    folder_path: str,
    label_suffix: str,
    viewer: Viewer,
):
    """
    MagicGUI widget for starting label inspection.
    """
    inspector = LabelInspector(viewer)
    inspector.load_image_label_pairs(folder_path, label_suffix)

    # Add buttons for saving and continuing to the next pair
    @magicgui(call_button="Save Changes and Continue")
    def save_and_continue():
        # Check if we're at the last pair before proceeding
        if inspector.current_index >= len(inspector.image_label_pairs) - 1:
            save_and_continue.call_button.enabled = False
            inspector.viewer.status = (
                "All pairs processed. Inspection complete."
            )
            return
        inspector.next_pair()

    viewer.window.add_dock_widget(save_and_continue)


def label_inspector_widget():
    """
    Provide the label inspector widget to Napari
    """
    # Create the magicgui widget
    widget = label_inspector

    # Create and add browse button
    browse_button = QPushButton("Browse...")

    def on_browse_clicked():
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            # Update the folder_path field
            widget.folder_path.value = folder

    browse_button.clicked.connect(on_browse_clicked)

    # Insert the browse button next to the folder_path field
    # Find the folder_path widget and its layout
    folder_layout = widget.folder_path.native.parent().layout()
    folder_layout.addWidget(browse_button)

    return widget

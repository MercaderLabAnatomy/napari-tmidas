import os
import sys

from magicgui import magicgui
from napari.layers import Labels
from napari.viewer import Viewer
from skimage.io import imread, imsave

sys.path.append("src/napari_tmidas")


class LabelInspector:
    def __init__(self, viewer: Viewer):
        self.viewer = viewer
        self.image_label_pairs = []
        self.current_index = 0

    def load_image_label_pairs(
        self, folder_path: str, image_suffix: str, label_suffix: str
    ):
        """
        Load image-label pairs from a folder.
        """
        files = os.listdir(folder_path)
        image_files = [file for file in files if file.endswith(image_suffix)]
        label_files = [file for file in files if file.endswith(label_suffix)]

        # Modified matching logic
        self.image_label_pairs = []
        for img in image_files:
            img_base = img[: -len(image_suffix)]  # Remove image suffix
            for lbl in label_files:
                if lbl.startswith(
                    img_base
                ):  # Check if label starts with image base
                    self.image_label_pairs.append(
                        (
                            os.path.join(folder_path, img),
                            os.path.join(folder_path, lbl),
                        )
                    )
                    break  # Match found, move to next image

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
        imsave(label_path, labels_layer.data.astype("uint16"))
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
            self.viewer.status = "No more pairs to inspect."
            # should also clear the viewer
            self.viewer.layers.clear()
            return

        # Move to the next pair
        self.current_index += 1

        # Load the next pair
        self._load_current_pair()


@magicgui(
    call_button="Start Label Inspection",
    folder_path={"label": "Folder Path"},
    image_suffix={"label": "Image Suffix (e.g., .tif)"},
    label_suffix={"label": "Label Suffix (e.g., _labels.tif)"},
)
def label_inspector(
    folder_path: str,
    image_suffix: str,
    label_suffix: str,
    viewer: Viewer,
):
    """
    MagicGUI widget for starting label inspection.
    """
    inspector = LabelInspector(viewer)
    inspector.load_image_label_pairs(folder_path, image_suffix, label_suffix)

    # Add buttons for saving and continuing to the next pair
    @magicgui(call_button="Save Changes and Continue")
    def save_and_continue():
        inspector.next_pair()

    viewer.window.add_dock_widget(save_and_continue)


def label_inspector_widget():
    """
    Provide the label inspector widget to Napari
    """
    return label_inspector

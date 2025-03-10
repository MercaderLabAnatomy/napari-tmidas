import contextlib
import os
from typing import List, Optional

import napari
import numpy as np
import tifffile
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class BatchProcessingRegistry:
    """
    A registry to manage and track available processing functions
    """

    _processing_functions = {}

    @classmethod
    def register(cls, name: str, suffix: str = "_processed"):
        """
        Decorator to register processing functions
        """

        def decorator(func):
            cls._processing_functions[name] = {"func": func, "suffix": suffix}
            return func

        return decorator

    @classmethod
    def get_function_info(cls, name: str) -> Optional[dict]:
        """
        Retrieve a registered processing function and its metadata
        """
        return cls._processing_functions.get(name)

    @classmethod
    def list_functions(cls) -> List[str]:
        """
        List all registered processing function names
        """
        return list(cls._processing_functions.keys())


# Example processing functions (kept from previous implementation)
@BatchProcessingRegistry.register("Simple Normalization", suffix="_normalized")
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization
    """
    return (image - image.min()) / (image.max() - image.min())


@BatchProcessingRegistry.register("Gaussian Blur", suffix="_blurred")
def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to the image
    """
    from scipy import ndimage

    return ndimage.gaussian_filter(image, sigma=sigma)


class ProcessedFilesTableWidget(QTableWidget):
    """
    Custom table widget with lazy loading and processing capabilities
    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # Configure table
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Original Files", "Processed Files"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Track file mappings
        self.file_pairs = {}

        # Currently loaded images
        self.current_original_image = None
        self.current_processed_image = None

    def add_initial_files(self, file_list: List[str]):
        """
        Add initial files to the table
        """
        # Clear existing rows
        self.setRowCount(0)
        self.file_pairs.clear()

        # Add files
        for filepath in file_list:
            row = self.rowCount()
            self.insertRow(row)

            # Original file item
            original_item = QTableWidgetItem(os.path.basename(filepath))
            original_item.setData(Qt.UserRole, filepath)
            self.setItem(row, 0, original_item)

            # Initially empty processed file column
            processed_item = QTableWidgetItem("")
            self.setItem(row, 1, processed_item)

            # Store file pair
            self.file_pairs[filepath] = {
                "original": filepath,
                "processed": None,
                "row": row,
            }

    def update_processed_files(self, processing_info: dict):
        """
        Update table with processed files

        processing_info: {
            'original_file': original filepath,
            'processed_file': processed filepath
        }
        """
        for item in processing_info:
            original_file = item["original_file"]
            processed_file = item["processed_file"]

            # Find the corresponding row
            if original_file in self.file_pairs:
                row = self.file_pairs[original_file]["row"]

                # Update processed file column
                processed_item = QTableWidgetItem(
                    os.path.basename(processed_file)
                )
                processed_item.setData(Qt.UserRole, processed_file)
                self.setItem(row, 1, processed_item)

                # Update file pairs
                self.file_pairs[original_file]["processed"] = processed_file

    def mousePressEvent(self, event):
        """
        Load image when clicked
        """
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item:
                filepath = item.data(Qt.UserRole)
                if filepath:
                    # Determine which column was clicked
                    column = self.columnAt(event.pos().x())
                    if column == 0:
                        # Original image clicked
                        self._load_original_image(filepath)
                    elif column == 1 and filepath:
                        # Processed image clicked
                        self._load_processed_image(filepath)

        super().mousePressEvent(event)

    def _load_original_image(self, filepath: str):
        """
        Load original image into viewer
        """
        # Remove existing original layer if it exists
        if self.current_original_image is not None:
            with contextlib.suppress(Exception):
                self.viewer.layers.remove(self.current_original_image)

        # Load new image
        try:
            image = tifffile.imread(filepath)
            self.current_original_image = self.viewer.add_image(
                image, name=f"Original: {os.path.basename(filepath)}"
            )
        except (ValueError, TypeError) as e:
            print(f"Error loading original image {filepath}: {e}")

    def _load_processed_image(self, filepath: str):
        """
        Load processed image into viewer
        """
        # Remove existing processed layer if it exists
        if self.current_processed_image is not None:
            with contextlib.suppress(Exception):
                self.viewer.layers.remove(self.current_processed_image)

        # Load new image
        try:
            image = tifffile.imread(filepath)
            self.current_processed_image = self.viewer.add_image(
                image, name=f"Processed: {os.path.basename(filepath)}"
            )
        except (ValueError, TypeError) as e:
            print(f"Error loading processed image {filepath}: {e}")

    def _load_image(self, filepath: str):
        """
        Legacy method kept for compatibility
        """
        self._load_original_image(filepath)


@magicgui(
    call_button="Find and Index Image Files",
    input_folder={"label": "Select Folder"},
    input_suffix={"label": "File Suffix (Example: _labels.tif)", "value": ""},
)
def file_selector(
    viewer: napari.Viewer, input_folder: str, input_suffix: str = "_labels.tif"
) -> List[str]:
    """
    Find files in a specified input folder with a given suffix and prepare for batch processing.
    """
    # Validate input_folder
    if not os.path.isdir(input_folder):
        viewer.status = f"Invalid input folder: {input_folder}"
        return []

    # Find matching files
    matching_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(input_suffix)
    ]

    # Create a results widget with batch processing option
    results_widget = FileResultsWidget(
        viewer,
        matching_files,
        input_folder=input_folder,
        input_suffix=input_suffix,
    )

    # Add the results widget to the Napari viewer
    viewer.window.add_dock_widget(
        results_widget, name="Matching Files", area="right"
    )

    # Update viewer status
    viewer.status = f"Found {len(matching_files)} files"

    return matching_files


class FileResultsWidget(QWidget):
    """
    Custom widget to display matching files and enable batch processing
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        file_list: List[str],
        input_folder: str,
        input_suffix: str,
    ):
        super().__init__()

        # Store viewer and file list
        self.viewer = viewer
        self.file_list = file_list
        self.input_folder = input_folder
        self.input_suffix = input_suffix

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create table of files
        self.table = ProcessedFilesTableWidget(viewer)
        self.table.add_initial_files(file_list)

        # Add table to layout
        layout.addWidget(self.table)

        # Create processing function selector
        processing_layout = QVBoxLayout()
        processing_label = QLabel("Select Processing Function:")
        processing_layout.addWidget(processing_label)

        self.processing_selector = QComboBox()
        self.processing_selector.addItems(
            BatchProcessingRegistry.list_functions()
        )
        processing_layout.addWidget(self.processing_selector)

        # Optional output folder selector
        output_layout = QVBoxLayout()
        output_label = QLabel("Output Folder (optional):")
        output_layout.addWidget(output_label)

        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText(
            "Leave blank to use source folder"
        )
        output_layout.addWidget(self.output_folder)

        layout.addLayout(processing_layout)
        layout.addLayout(output_layout)

        # Add batch processing button
        self.batch_button = QPushButton("Start Batch Processing")
        self.batch_button.clicked.connect(self.start_batch_processing)
        layout.addWidget(self.batch_button)

    def start_batch_processing(self):
        """
        Initiate batch processing of selected files
        """
        # Get selected processing function
        selected_function_name = self.processing_selector.currentText()
        function_info = BatchProcessingRegistry.get_function_info(
            selected_function_name
        )

        if not function_info:
            self.viewer.status = "No processing function selected"
            return

        processing_func = function_info["func"]
        output_suffix = function_info["suffix"]

        # Determine output folder
        output_folder = self.output_folder.text().strip()
        if not output_folder:
            output_folder = os.path.dirname(self.file_list[0])

        # make output folder a subfolder of the input folder
        output_folder = os.path.join(self.input_folder, output_folder)

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Track processed files
        processed_files_info = []

        # Process each file
        for filepath in self.file_list:
            try:
                # Load the image
                image = tifffile.imread(filepath)

                # Apply processing
                processed_image = processing_func(image)

                # Generate new filename
                filename = os.path.basename(filepath)
                name, ext = os.path.splitext(filename)
                new_filename = (
                    filename.replace(self.input_suffix, output_suffix) + ext
                )
                new_filepath = os.path.join(output_folder, new_filename)

                # Save processed image
                tifffile.imwrite(new_filepath, processed_image)

                # Track processed file
                processed_files_info.append(
                    {"original_file": filepath, "processed_file": new_filepath}
                )

            except (ValueError, TypeError) as e:
                print(f"Error processing {filepath}: {e}")

        # Update table with processed files
        self.table.update_processed_files(processed_files_info)

        # Update viewer status
        self.viewer.status = f"Processed files with {selected_function_name}"


def napari_experimental_provide_dock_widget():
    """
    Provide the file selector widget to Napari
    """
    return file_selector

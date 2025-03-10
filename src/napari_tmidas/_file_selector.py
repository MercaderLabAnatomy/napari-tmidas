import contextlib
import os
import sys
from typing import Any, Dict, List

import napari
import tifffile
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Import registry and processing functions
from napari_tmidas._registry import BatchProcessingRegistry

sys.path.append("src/napari_tmidas")
from napari_tmidas.processing_functions import (
    discover_and_load_processing_functions,
)


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
            with contextlib.suppress(KeyError):
                self.viewer.layers.remove(self.current_original_image)

        # Load new image
        try:
            image = tifffile.imread(filepath)
            self.current_original_image = self.viewer.add_image(
                image, name=f"Original: {os.path.basename(filepath)}"
            )
        except (ValueError, TypeError, OSError, tifffile.TiffFileError) as e:
            print(f"Error loading original image {filepath}: {e}")
            self.viewer.status = f"Error processing {filepath}: {e}"

    def _load_processed_image(self, filepath: str):
        """
        Load processed image into viewer
        """
        # Remove existing processed layer if it exists
        if self.current_processed_image is not None:
            with contextlib.suppress(KeyError):
                self.viewer.layers.remove(self.current_processed_image)

        # Load new image
        try:
            image = tifffile.imread(filepath)
            self.current_processed_image = self.viewer.add_image(
                image, name=f"Processed: {os.path.basename(filepath)}"
            )
        except (ValueError, TypeError, OSError, tifffile.TiffFileError) as e:
            print(f"Error loading processed image {filepath}: {e}")
            self.viewer.status = f"Error processing {filepath}: {e}"

    def _load_image(self, filepath: str):
        """
        Legacy method kept for compatibility
        """
        self._load_original_image(filepath)


class ParameterWidget(QWidget):
    """
    Widget to display and edit processing function parameters
    """

    def __init__(self, parameters: Dict[str, Dict[str, Any]]):
        super().__init__()

        self.parameters = parameters
        self.param_widgets = {}

        layout = QFormLayout()
        self.setLayout(layout)

        # Create widgets for each parameter
        for param_name, param_info in parameters.items():
            param_type = param_info.get("type")
            default_value = param_info.get("default")
            min_value = param_info.get("min")
            max_value = param_info.get("max")
            description = param_info.get("description", "")

            # Create appropriate widget based on parameter type
            if param_type is int:
                widget = QSpinBox()
                if min_value is not None:
                    widget.setMinimum(min_value)
                if max_value is not None:
                    widget.setMaximum(max_value)
                if default_value is not None:
                    widget.setValue(default_value)
            elif param_type is float:
                widget = QDoubleSpinBox()
                if min_value is not None:
                    widget.setMinimum(min_value)
                if max_value is not None:
                    widget.setMaximum(max_value)
                widget.setDecimals(3)
                if default_value is not None:
                    widget.setValue(default_value)
            else:
                # Default to text input for other types
                widget = QLineEdit(
                    str(default_value) if default_value is not None else ""
                )

            # Add widget to layout with label
            layout.addRow(f"{param_name} ({description}):", widget)
            self.param_widgets[param_name] = widget

    def get_parameter_values(self) -> Dict[str, Any]:
        """
        Get current parameter values from widgets
        """
        values = {}
        for param_name, widget in self.param_widgets.items():
            param_type = self.parameters[param_name]["type"]

            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[param_name] = widget.value()
            else:
                # For text inputs, try to convert to the appropriate type
                try:
                    values[param_name] = param_type(widget.text())
                except (ValueError, TypeError):
                    # Fall back to string if conversion fails
                    values[param_name] = widget.text()

        return values


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

        # Load all processing functions
        print("Calling discover_and_load_processing_functions")
        discover_and_load_processing_functions()
        # print what is found by discover_and_load_processing_functions
        print("Available processing functions:")
        for func_name in BatchProcessingRegistry.list_functions():
            print(func_name)

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

        # Add description label
        self.function_description = QLabel("")
        processing_layout.addWidget(self.function_description)

        # Create parameters section (will be populated when function is selected)
        self.parameters_widget = QWidget()
        processing_layout.addWidget(self.parameters_widget)

        # Connect function selector to update parameters
        self.processing_selector.currentTextChanged.connect(
            self.update_function_info
        )

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

        # Initialize parameters for the first function
        if self.processing_selector.count() > 0:
            self.update_function_info(self.processing_selector.currentText())

    def update_function_info(self, function_name: str):
        """
        Update the function description and parameters when a new function is selected
        """
        function_info = BatchProcessingRegistry.get_function_info(
            function_name
        )
        if not function_info:
            return

        # Update description
        description = function_info.get("description", "")
        self.function_description.setText(description)

        # Update parameters
        parameters = function_info.get("parameters", {})

        # Remove old parameters widget if it exists
        if hasattr(self, "param_widget_instance"):
            self.parameters_widget.layout().removeWidget(
                self.param_widget_instance
            )
            self.param_widget_instance.deleteLater()

        # Create new layout if needed
        if self.parameters_widget.layout() is None:
            self.parameters_widget.setLayout(QVBoxLayout())

        # Create and add new parameters widget
        if parameters:
            self.param_widget_instance = ParameterWidget(parameters)
            self.parameters_widget.layout().addWidget(
                self.param_widget_instance
            )
        else:
            # Create empty widget if no parameters
            self.param_widget_instance = QLabel(
                "No parameters for this function"
            )
            self.parameters_widget.layout().addWidget(
                self.param_widget_instance
            )

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

        # Get parameter values if available
        param_values = {}
        if hasattr(self, "param_widget_instance") and hasattr(
            self.param_widget_instance, "get_parameter_values"
        ):
            param_values = self.param_widget_instance.get_parameter_values()

        # Determine output folder
        output_folder = self.output_folder.text().strip()
        if not output_folder:
            output_folder = os.path.dirname(self.file_list[0])
        else:
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

                # Apply processing with parameters
                processed_image = processing_func(image, **param_values)

                # Generate new filename
                filename = os.path.basename(filepath)
                name, ext = os.path.splitext(filename)
                new_filename = (
                    name.replace(self.input_suffix, "") + output_suffix + ext
                )
                new_filepath = os.path.join(output_folder, new_filename)

                # Save processed image
                tifffile.imwrite(new_filepath, processed_image)

                # Track processed file
                processed_files_info.append(
                    {
                        "original_file": filepath,
                        "processed_file": new_filepath,
                    }
                )
                self.viewer.status = f"Processed and saved: {new_filename}"

            except (
                ValueError,
                TypeError,
                OSError,
                tifffile.TiffFileError,
            ) as e:
                self.viewer.status = f"Error processing {filepath}: {e}"
                print(f"Error processing {filepath}: {e}")

        # Update the table with the processed files
        self.table.update_processed_files(processed_files_info)

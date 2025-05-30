"""
Batch Image Processing with Napari
----------------------------------
This module provides a collection of functions for batch processing of image files.
It includes a Napari widget for selecting files and processing functions, and a
custom widget for displaying and processing the selected files.

New functions can be added to the processing registry by decorating them with
`@register_batch_processing_function`. Each function should accept an image array
as the first argument, and any additional keyword arguments for parameters.
"""

import concurrent.futures
import os
import sys
from typing import Any, Dict, List

import napari
import numpy as np
import tifffile
from magicgui import magicgui
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread

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

        # For tracking multi-output files
        self.multi_output_files = {}

        # Connect the cellDoubleClicked signal
        self.cellDoubleClicked.connect(self._handle_cell_double_click)

    def add_initial_files(self, file_list: List[str]):
        """
        Add initial files to the table
        """
        # Clear existing rows
        self.setRowCount(0)
        self.file_pairs.clear()
        self.multi_output_files.clear()

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

    def update_processed_files(self, processing_info: List[Dict]):
        """
        Update table with processed files

        Args:
            processing_info: List of dictionaries containing:
                {
                    'original_file': original filepath,
                    'processed_file': processed filepath (single output)
                    - OR -
                    'processed_files': list of processed filepaths (multi-output)
                }
        """
        for item in processing_info:
            original_file = item["original_file"]

            # Handle single processed file case
            if "processed_file" in item:
                processed_file = item["processed_file"]

                # Find the corresponding row
                if original_file in self.file_pairs:
                    row = self.file_pairs[original_file]["row"]

                    # Create a single item with the processed file
                    file_name = os.path.basename(processed_file)
                    processed_item = QTableWidgetItem(file_name)
                    processed_item.setData(Qt.UserRole, processed_file)
                    processed_item.setToolTip("Double-click to view")
                    self.setItem(row, 1, processed_item)

                    # Update file pairs
                    self.file_pairs[original_file][
                        "processed"
                    ] = processed_file

            # Handle multi-file output case
            elif "processed_files" in item and item["processed_files"]:
                processed_files = item["processed_files"]

                # Store all processed files for this original file
                self.multi_output_files[original_file] = processed_files

                # Find the corresponding row
                if original_file in self.file_pairs:
                    row = self.file_pairs[original_file]["row"]

                    # Create a ComboBox for selecting outputs
                    combo = QComboBox()
                    for i, file_path in enumerate(processed_files):
                        file_name = os.path.basename(file_path)
                        combo.addItem(f"Channel {i}: {file_name}", file_path)

                    # Connect the combo box to load the selected processed file
                    combo.currentIndexChanged.connect(
                        lambda idx, files=processed_files: self._load_processed_image(
                            files[idx]
                        )
                    )

                    # Add the ComboBox directly to the table cell
                    self.setCellWidget(row, 1, combo)

                    # Update file pairs with first file as default
                    self.file_pairs[original_file]["processed"] = (
                        processed_files[0]
                    )

    def mousePressEvent(self, event):
        """
        Handle mouse click events on the table to load appropriate images
        """
        if event.button() == Qt.LeftButton:
            # Get the item at the click position
            item = self.itemAt(event.pos())
            column = self.columnAt(event.pos().x())
            row = self.rowAt(event.pos().y())

            # Load original image when clicking on first column
            if column == 0 and item:
                filepath = item.data(Qt.UserRole)
                if filepath:
                    self._load_original_image(filepath)

            # Load processed image when clicking on second column (for single output files)
            elif column == 1:
                # Check if this cell has a non-combo-box item (single output)
                cell_item = self.item(row, column)
                if cell_item and cell_item.data(Qt.UserRole):
                    filepath = cell_item.data(Qt.UserRole)
                    if filepath:
                        self._load_processed_image(filepath)
                # Combo boxes are handled by their own event handlers

        super().mousePressEvent(event)

    def _handle_cell_double_click(self, row, column):
        """
        Handle double-click events on cells, particularly for single processed files
        """
        if column == 1:
            item = self.item(row, column)
            if (
                item
            ):  # This means it's a single processed file, not a combo box
                filepath = item.data(Qt.UserRole)
                if filepath:
                    self._load_processed_image(filepath)

    def _load_original_image(self, filepath: str):
        """
        Load original image into viewer
        """
        # Ensure filepath is valid
        if not filepath or not os.path.exists(filepath):
            print(f"Error: File does not exist: {filepath}")
            self.viewer.status = f"Error: File not found: {filepath}"
            return

        # Remove existing original layer if it exists
        if self.current_original_image is not None:
            try:
                # Check if the layer is still in the viewer
                if self.current_original_image in self.viewer.layers:
                    self.viewer.layers.remove(self.current_original_image)
                else:
                    # If not found by reference, try by name
                    layer_names = [layer.name for layer in self.viewer.layers]
                    if self.current_original_image.name in layer_names:
                        self.viewer.layers.remove(
                            self.current_original_image.name
                        )
            except (KeyError, ValueError) as e:
                print(
                    f"Warning: Could not remove previous original layer: {e}"
                )

            # Reset the current original image reference
            self.current_original_image = None

        # Load new image
        try:
            # Display status while loading
            self.viewer.status = f"Loading {os.path.basename(filepath)}..."

            image = imread(filepath)
            # remove singletons
            image = np.squeeze(image)
            # check if label image by checking file name
            is_label = "labels" in os.path.basename(
                filepath
            ) or "semantic" in os.path.basename(filepath)
            if is_label:
                image = image.astype(np.uint32)
                self.current_original_image = self.viewer.add_labels(
                    image, name=f"Labels: {os.path.basename(filepath)}"
                )
            else:
                self.current_original_image = self.viewer.add_image(
                    image, name=f"Original: {os.path.basename(filepath)}"
                )

            # Update status with success message
            self.viewer.status = f"Loaded {os.path.basename(filepath)}"

        except (ValueError, TypeError, OSError, tifffile.TiffFileError) as e:
            print(f"Error loading original image {filepath}: {e}")
            self.viewer.status = f"Error processing {filepath}: {e}"

    def _load_processed_image(self, filepath: str):
        """
        Load processed image into viewer, distinguishing labels by filename pattern
        and ensure it's always shown on top
        """
        # Ensure filepath is valid
        if not filepath or not os.path.exists(filepath):
            print(f"Error: File does not exist: {filepath}")
            self.viewer.status = f"Error: File not found: {filepath}"
            return

        # Remove existing processed layer if it exists
        if self.current_processed_image is not None:
            try:
                # Check if the layer is still in the viewer
                if self.current_processed_image in self.viewer.layers:
                    self.viewer.layers.remove(self.current_processed_image)
                else:
                    # If not found by reference, try by name
                    layer_names = [layer.name for layer in self.viewer.layers]
                    if self.current_processed_image.name in layer_names:
                        self.viewer.layers.remove(
                            self.current_processed_image.name
                        )
            except (KeyError, ValueError) as e:
                print(
                    f"Warning: Could not remove previous processed layer: {e}"
                )

            # Reset the current processed image reference
            self.current_processed_image = None

        # Load new image
        try:
            # Display status while loading
            self.viewer.status = f"Loading {os.path.basename(filepath)}..."

            image = imread(filepath)
            # remove singletons
            image = np.squeeze(image)
            filename = os.path.basename(filepath)

            # Check if filename contains label indicators
            is_label = "labels" in filename or "semantic" in filename

            # Add the layer using the appropriate method
            if is_label:
                # Ensure it's an appropriate dtype for labels
                if not np.issubdtype(image.dtype, np.integer):
                    image = image.astype(np.uint32)

                self.current_processed_image = self.viewer.add_labels(
                    image, name=f"Labels: {filename}"
                )
            else:
                self.current_processed_image = self.viewer.add_image(
                    image, name=f"Processed: {filename}"
                )

            # Move the processed layer to the top of the stack
            # Get the index of the current processed layer
            layer_index = self.viewer.layers.index(
                self.current_processed_image
            )
            # Move it to the top (last position in the list)
            if layer_index < len(self.viewer.layers) - 1:
                self.viewer.layers.move(
                    layer_index, len(self.viewer.layers) - 1
                )

            # Update status with success message
            self.viewer.status = f"Loaded {filename} (moved to top layer)"

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
    input_folder={
        "widget_type": "LineEdit",
        "label": "Select Folder",
        "value": "",
    },
    input_suffix={"label": "File Suffix (Example: .tif)", "value": ""},
)
def file_selector(
    viewer: napari.Viewer, input_folder: str, input_suffix: str = ".tif"
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


# Modify the file_selector widget to add a browse button after it's created
def _add_browse_button_to_selector(file_selector_widget):
    """
    Add a browse button to the file selector widget
    """
    # Get the container widget that holds the input_folder widget
    container = file_selector_widget.native

    # Create a browse button
    browse_button = QPushButton("Browse...")

    # Get access to the input_folder widget
    input_folder_widget = file_selector_widget.input_folder.native

    # Get the parent of the input_folder widget
    parent_layout = input_folder_widget.parentWidget().layout()

    # Create a container for input field and browse button
    container_widget = QWidget()
    h_layout = QHBoxLayout(container_widget)
    h_layout.setContentsMargins(0, 0, 0, 0)

    # Add the input field to our container
    h_layout.addWidget(input_folder_widget)

    # Add the browse button
    h_layout.addWidget(browse_button)

    # Replace the input field with our container
    layout_index = parent_layout.indexOf(input_folder_widget)
    parent_layout.removeWidget(input_folder_widget)
    parent_layout.insertWidget(layout_index, container_widget)

    # Connect button to browse action
    def browse_folder():
        folder = QFileDialog.getExistingDirectory(
            container,
            "Select Folder",
            file_selector_widget.input_folder.value or os.path.expanduser("~"),
        )
        if folder:
            file_selector_widget.input_folder.value = folder

    browse_button.clicked.connect(browse_folder)

    return file_selector_widget


# Create a modified file_selector with browse button
file_selector = _add_browse_button_to_selector(file_selector)


# Processing worker for multithreading
class ProcessingWorker(QThread):
    """
    Worker thread for processing images in the background
    """

    # Signals to communicate with the main thread
    progress_updated = Signal(int)
    file_processed = Signal(dict)
    processing_finished = Signal()
    error_occurred = Signal(str, str)  # filepath, error message

    def __init__(
        self,
        file_list,
        processing_func,
        param_values,
        output_folder,
        input_suffix,
        output_suffix,
    ):
        super().__init__()
        self.file_list = file_list
        self.processing_func = processing_func
        self.param_values = param_values
        self.output_folder = output_folder
        self.input_suffix = input_suffix
        self.output_suffix = output_suffix
        self.stop_requested = False
        self.thread_count = max(1, (os.cpu_count() or 4) - 1)  # Default value

    def stop(self):
        """Request the worker to stop processing"""
        self.stop_requested = True

    def run(self):
        """Process files in a separate thread"""
        # Track processed files
        processed_files_info = []
        total_files = len(self.file_list)

        # Create a thread pool for concurrent processing with specified thread count
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread_count
        ) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(self.process_file, filepath): filepath
                for filepath in self.file_list
            }

            # Process as they complete
            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_file)
            ):
                # Check if cancellation was requested
                if self.stop_requested:
                    break

                filepath = future_to_file[future]
                try:
                    result = future.result()
                    # Only process result if it's not None (folder functions may return None)
                    if result is not None:
                        processed_files_info.append(result)
                        self.file_processed.emit(result)
                except (
                    ValueError,
                    TypeError,
                    OSError,
                    tifffile.TiffFileError,
                ) as e:
                    self.error_occurred.emit(filepath, str(e))

                # Update progress
                self.progress_updated.emit(int((i + 1) / total_files * 100))

        # Signal that processing is complete
        self.processing_finished.emit()

    def process_file(self, filepath):
        """Process a single file with support for large TIFF files and removal of all singleton dimensions"""
        try:
            # Load the image
            image = imread(filepath)
            image_dtype = image.dtype

            print(f"Original image shape: {image.shape}, dtype: {image_dtype}")

            # Check if this is a folder-processing function that shouldn't save individual files
            function_name = getattr(
                self.processing_func, "__name__", "unknown"
            )
            is_folder_function = (
                "timepoint" in function_name.lower()
                or "merge" in function_name.lower()
                or "folder" in function_name.lower()
            )

            # Apply processing with parameters
            processed_image = self.processing_func(image, **self.param_values)

            print(
                f"Processed image shape before removing singletons: {processed_image.shape}, dtype: {processed_image.dtype}"
            )

            # For folder functions, check if the output is the same as input (indicating no individual file should be saved)
            if is_folder_function:
                # If the function returns the original image unchanged, it means it handled saving internally
                if np.array_equal(processed_image, image):
                    print(
                        "Folder function returned unchanged image - skipping individual file save"
                    )
                    return None  # Return None to indicate no file should be created
                else:
                    print(
                        "Folder function returned different data - will save individual file"
                    )

            # Remove ALL singleton dimensions from the processed image
            # This will keep only dimensions with size > 1
            processed_image = np.squeeze(processed_image)

            print(
                f"Processed image shape after removing singletons: {processed_image.shape}"
            )

            # Generate new filename base
            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            if name.endswith(self.input_suffix):
                new_filename_base = (
                    name[: -len(self.input_suffix)] + self.output_suffix
                )
            else:
                new_filename_base = name + self.output_suffix

            # Check if the first dimension should be treated as channels
            # If processed_image has more dimensions than the original image,
            # assume the first dimension represents channels
            is_multi_channel = (processed_image.ndim > image.ndim - 1) or (
                processed_image.ndim == image.ndim
                and processed_image.shape[0] <= 10
            )

            if (
                is_multi_channel and processed_image.shape[0] <= 10
            ):  # Reasonable number of channels
                # Save each channel as a separate image
                processed_files = []

                num_channels = processed_image.shape[0]
                print(
                    f"Treating first dimension as channels. Saving {num_channels} separate channel files"
                )

                for i in range(num_channels):
                    channel_filename = f"{new_filename_base}_channel_{i}{ext}"
                    channel_filepath = os.path.join(
                        self.output_folder, channel_filename
                    )

                    # Extract channel data and remove any remaining singleton dimensions
                    channel_image = np.squeeze(processed_image[i])

                    print(f"Channel {i} shape: {channel_image.shape}")

                    # Calculate approx file size in GB
                    size_gb = (
                        channel_image.size * channel_image.itemsize / (1024**3)
                    )
                    print(f"Estimated file size: {size_gb:.2f} GB")

                    # Check data range
                    data_min = (
                        np.min(channel_image) if channel_image.size > 0 else 0
                    )
                    data_max = (
                        np.max(channel_image) if channel_image.size > 0 else 0
                    )
                    print(f"Channel {i} data range: {data_min} to {data_max}")

                    # For very large files, we need to use BigTIFF format
                    use_bigtiff = (
                        size_gb > 2.0
                    )  # Use BigTIFF for files over 2GB

                    if (
                        "labels" in channel_filename
                        or "semantic" in channel_filename
                    ):
                        # Choose appropriate integer type based on data range
                        if data_max <= 255:
                            save_dtype = np.uint8
                        elif data_max <= 65535:
                            save_dtype = np.uint16
                        else:
                            save_dtype = np.uint32

                        print(
                            f"Label image detected, saving as {save_dtype.__name__} with bigtiff={use_bigtiff}"
                        )
                        tifffile.imwrite(
                            channel_filepath,
                            channel_image.astype(save_dtype),
                            compression="zlib",
                            bigtiff=use_bigtiff,
                        )
                    else:
                        # Handle large images with bigtiff format
                        print(
                            f"Regular image channel, saving with dtype {image_dtype} and bigtiff={use_bigtiff}"
                        )

                        # Save with original dtype and bigtiff format if needed
                        tifffile.imwrite(
                            channel_filepath,
                            channel_image.astype(image_dtype),
                            compression="zlib",
                            bigtiff=use_bigtiff,
                        )

                    processed_files.append(channel_filepath)

                # Return processing info
                return {
                    "original_file": filepath,
                    "processed_files": processed_files,
                }
            else:
                # Save as a single image
                new_filepath = os.path.join(
                    self.output_folder, new_filename_base + ext
                )

                print(f"Single output image shape: {processed_image.shape}")

                # Calculate approx file size in GB
                size_gb = (
                    processed_image.size * processed_image.itemsize / (1024**3)
                )
                print(f"Estimated file size: {size_gb:.2f} GB")

                # For very large files, we need to use BigTIFF format
                use_bigtiff = size_gb > 2.0  # Use BigTIFF for files over 2GB

                # Check data range
                data_min = (
                    np.min(processed_image) if processed_image.size > 0 else 0
                )
                data_max = (
                    np.max(processed_image) if processed_image.size > 0 else 0
                )
                print(f"Data range: {data_min} to {data_max}")

                if (
                    "labels" in new_filename_base
                    or "semantic" in new_filename_base
                ):

                    save_dtype = np.uint32

                    print(
                        f"Saving label image as {save_dtype.__name__} with bigtiff={use_bigtiff}"
                    )
                    tifffile.imwrite(
                        new_filepath,
                        processed_image.astype(save_dtype),
                        compression="zlib",
                        bigtiff=use_bigtiff,
                    )
                else:
                    print(
                        f"Saving image with dtype {image_dtype} and bigtiff={use_bigtiff}"
                    )
                    tifffile.imwrite(
                        new_filepath,
                        processed_image.astype(image_dtype),
                        compression="zlib",
                        bigtiff=use_bigtiff,
                    )

                # Return processing info
                return {
                    "original_file": filepath,
                    "processed_file": new_filepath,
                }

        except Exception as e:
            # Log the error and re-raise to be caught by the executor
            print(f"Error processing {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            # Explicit cleanup to help with memory management
            if "image" in locals():
                del image
            if "processed_image" in locals():
                del processed_image


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
        self.worker = None  # Will hold the processing worker

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create table of files
        self.table = ProcessedFilesTableWidget(viewer)
        self.table.add_initial_files(file_list)

        # Add table to layout
        layout.addWidget(self.table)

        # Load all processing functions
        print("Calling discover_and_load_processing_functions")
        discover_and_load_processing_functions()
        # print what is found by discover_and_load_processing_functions
        print("Available processing functions:")
        for func_name in BatchProcessingRegistry.list_functions():
            print(func_name)

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

        # Thread count selector
        thread_layout = QHBoxLayout()
        thread_label = QLabel("Number of threads:")
        thread_layout.addWidget(thread_label)

        self.thread_count = QSpinBox()
        self.thread_count.setMinimum(1)
        self.thread_count.setMaximum(
            os.cpu_count() or 4
        )  # Default to CPU count or 4
        self.thread_count.setValue(
            max(1, (os.cpu_count() or 4) - 1)
        )  # Default to CPU count - 1
        thread_layout.addWidget(self.thread_count)

        output_layout.addLayout(thread_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hide initially

        layout.addLayout(processing_layout)
        layout.addLayout(output_layout)
        layout.addWidget(self.progress_bar)

        # Add batch processing and cancel buttons
        button_layout = QHBoxLayout()

        self.batch_button = QPushButton("Start Batch Processing")
        self.batch_button.clicked.connect(self.start_batch_processing)
        button_layout.addWidget(self.batch_button)

        self.cancel_button = QPushButton("Cancel Processing")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)  # Disabled initially
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Initialize parameters for the first function
        if self.processing_selector.count() > 0:
            self.update_function_info(self.processing_selector.currentText())

        # Container for tracking processed files during batch operation
        self.processed_files_info = []

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

        # Check if this is a folder-processing function that needs single threading
        is_folder_function = (
            "folder" in function_name.lower()
            or "timepoint" in function_name.lower()
            or "merge" in function_name.lower()
            or "folder" in description.lower()
            or "cellpose" in description.lower()
            or "careamics" in description.lower()
            or "trackastra" in description.lower()
        )

        # Disable threading controls for folder functions
        if is_folder_function:
            self.thread_count.setValue(1)
            self.thread_count.setEnabled(False)
            self.thread_count.setToolTip(
                "This function processes entire folders and must run with 1 thread only."
            )

            # Add warning to description if not already present
            if (
                "IMPORTANT:" not in description
                and "WARNING:" not in description
            ):
                description += "\nThis function has to run single-threaded."

            self.function_description.setText(description)

            # Change the description color to make it more prominent
            self.function_description.setStyleSheet(
                "QLabel { color: #ff6b00; font-weight: bold; }"
            )
        else:
            # Re-enable threading controls for normal functions
            self.thread_count.setEnabled(True)
            self.thread_count.setToolTip(
                "Number of threads to use for parallel processing"
            )
            self.function_description.setStyleSheet("")  # Reset styling
            self.function_description.setText(description)

        # Get parameters
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
        Initiate multithreaded batch processing of selected files
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

        # Reset progress tracking
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.processed_files_info = []

        # Update UI
        self.batch_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Set thread count based on function properties
        worker_thread_count = self.thread_count.value()

        # Check if function should run single-threaded
        if (
            hasattr(processing_func, "thread_safe")
            and not processing_func.thread_safe
        ):
            worker_thread_count = 1
            self.viewer.status = (
                "Processing with a single thread (function is not thread-safe)"
            )
        else:
            self.viewer.status = (
                f"Processing with {worker_thread_count} threads"
            )

        # Create and start the worker thread
        self.worker = ProcessingWorker(
            self.file_list,
            processing_func,
            param_values,
            output_folder,
            self.input_suffix,
            output_suffix,
        )

        # Set the thread count from the UI or function attribute
        self.worker.thread_count = worker_thread_count

        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.file_processed.connect(self.file_processed)
        self.worker.processing_finished.connect(self.processing_finished)
        self.worker.error_occurred.connect(self.processing_error)

        # Start processing
        self.worker.start()

        # Update status
        self.viewer.status = f"Processing {len(self.file_list)} files with {selected_function_name} using {worker_thread_count} threads"

    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)

    def file_processed(self, result):
        """Handle a processed file result"""
        self.processed_files_info.append(result)
        # Update table with this single processed file
        self.table.update_processed_files([result])

    def processing_finished(self):
        """Handle processing completion"""
        # Update UI
        self.progress_bar.setValue(100)
        self.batch_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        # Clean up worker
        self.worker = None

        # Update status
        self.viewer.status = (
            f"Completed processing {len(self.processed_files_info)} files"
        )

    def processing_error(self, filepath, error_msg):
        """Handle processing errors"""
        print(f"Error processing {filepath}: {error_msg}")
        self.viewer.status = f"Error processing {filepath}: {error_msg}"

    def cancel_processing(self):
        """Cancel the current processing operation"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()  # Wait for the thread to finish

            # Update UI
            self.batch_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.viewer.status = "Processing cancelled"


def napari_experimental_provide_dock_widget():
    """
    Provide the file selector widget to Napari
    """
    return file_selector

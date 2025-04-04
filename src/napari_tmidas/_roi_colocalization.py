"""
ROI Colocalization Analysis for Napari
-------------------------------------
This module provides a GUI for analyzing colocalization between ROIs in multiple channel label images.
It can process images with 2 or 3 channels and generate statistics about their overlap.

The colocalization analysis counts how many labels from one channel overlap with regions in another channel,
and can optionally calculate sizes of these overlapping regions.
"""

import concurrent.futures

# contextlib is used to suppress exceptions
import contextlib
import csv
import os
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import tifffile
from magicgui import magic_factory
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
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
from skimage import measure


def longest_common_substring(s1, s2):
    """Finds the longest common substring between two strings."""
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    substring = s1[match.a : match.a + match.size]
    print(f"Longest common substring between '{s1}' and '{s2}': '{substring}'")
    return substring


def group_files_by_common_substring(file_lists, channels):
    """
    Groups files across channels based on the longest common substring in their filenames.

    Args:
        file_lists (dict): A dictionary where keys are channel names and values are lists of file paths.
        channels (list): A list of channel names corresponding to the keys in file_lists.

    Returns:
        dict: A dictionary where keys are common substrings (without suffixes) and values are lists of file paths grouped by substring.
    """
    # Extract the base filenames for each channel
    base_files = {
        channel: [os.path.basename(file) for file in file_lists[channel]]
        for channel in channels
    }

    # Create a dictionary to store groups
    groups = defaultdict(lambda: {channel: None for channel in channels})

    # Iterate over all files in the first channel
    for file1 in base_files[channels[0]]:
        # Start with the first file as the "common substring"
        common_substring = file1

        # Iterate over the other channels to find matching files
        matched_files = {channels[0]: file1}
        for channel in channels[1:]:
            best_match = None
            best_common = ""

            # Compare the current common substring with files in the current channel
            for file2 in base_files[channel]:
                current_common = longest_common_substring(
                    common_substring, file2
                )
                if len(current_common) > len(best_common):
                    best_match = file2
                    best_common = current_common

            # If a match is found, update the common substring and store the match
            if best_match:
                common_substring = best_common
                matched_files[channel] = best_match
            else:
                # If no match is found, skip this file
                break

        # If matches were found for all channels, add them to the group
        if len(matched_files) == len(channels):
            # Strip suffixes from the common substring
            stripped_common_substring = common_substring.rsplit("_", 1)[0]
            groups[stripped_common_substring] = {
                channel: file_lists[channel][
                    base_files[channel].index(matched_files[channel])
                ]
                for channel in channels
            }

    # Filter out incomplete groups (e.g., missing files for required channels)
    valid_groups = {
        key: list(group.values())
        for key, group in groups.items()
        if all(group[channel] for channel in channels)
    }

    return valid_groups


class ColocalizationWorker(QThread):
    """Worker thread for processing label images"""

    progress_updated = Signal(int)  # Current progress
    file_processed = Signal(dict)  # Results for a processed file
    processing_finished = Signal()  # Signal when all processing is done
    error_occurred = Signal(str, str)  # filepath, error message

    def __init__(
        self,
        file_pairs,
        channel_names,
        get_sizes=False,
        size_method="median",
        output_folder=None,
    ):
        super().__init__()
        self.file_pairs = file_pairs
        self.channel_names = channel_names
        self.get_sizes = get_sizes
        self.size_method = size_method
        self.output_folder = output_folder
        self.stop_requested = False
        self.thread_count = max(1, (os.cpu_count() or 4) - 1)  # Default value

    def run(self):
        """Process files in a separate thread"""
        # Track processed files
        processed_files_info = []
        total_files = len(self.file_pairs)

        # Create output folder if it doesn't exist
        csv_path = None
        if self.output_folder:
            try:
                # Make sure the directory exists with all parent directories
                os.makedirs(self.output_folder, exist_ok=True)

                # Set up CSV path
                channels_str = "_".join(self.channel_names)
                csv_path = os.path.join(
                    self.output_folder, f"{channels_str}_colocalization.csv"
                )

                # Create CSV header
                header = [
                    "Filename",
                    f"{self.channel_names[0]}_label_id",
                    f"{self.channel_names[1]}_in_{self.channel_names[0]}_count",
                ]

                if self.get_sizes:
                    header.extend(
                        [
                            f"{self.channel_names[0]}_size",
                            f"{self.channel_names[1]}_in_{self.channel_names[0]}_size",
                        ]
                    )

                if len(self.channel_names) == 3:
                    header.extend(
                        [
                            f"{self.channel_names[2]}_in_{self.channel_names[1]}_in_{self.channel_names[0]}_count",
                            f"{self.channel_names[2]}_not_in_{self.channel_names[1]}_but_in_{self.channel_names[0]}_count",
                        ]
                    )

                    if self.get_sizes:
                        header.extend(
                            [
                                f"{self.channel_names[2]}_in_{self.channel_names[1]}_in_{self.channel_names[0]}_size",
                                f"{self.channel_names[2]}_not_in_{self.channel_names[1]}_but_in_{self.channel_names[0]}_size",
                            ]
                        )

                # print(f"CSV Header: {header}")

                # check if the file already exists and overwrite it
                if os.path.exists(csv_path):
                    # If it exists, remove it
                    os.remove(csv_path)  # this
                    # if it fails, tell the user to delete it manually:
                    if os.path.exists(csv_path):
                        raise Exception(
                            f"Failed to remove existing CSV file: {csv_path}"
                        )

                # Try to create and initialize CSV file
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)

            except (Exception, FileNotFoundError) as e:
                import traceback

                traceback.print_exc()
                csv_path = None
                self.error_occurred.emit(
                    "CSV file", f"Failed to set up CSV file: {str(e)}"
                )

        # Create a thread pool for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread_count
        ) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(self.process_file_pair, file_pair): file_pair
                for file_pair in self.file_pairs
            }

            # Process as they complete
            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_file)
            ):
                # Check if cancellation was requested
                if self.stop_requested:
                    break

                file_pair = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        processed_files_info.append(result)
                        self.file_processed.emit(result)

                        # Write to CSV if output folder is specified and CSV setup worked
                        if csv_path and "csv_rows" in result:
                            try:
                                with open(
                                    csv_path, "a", newline=""
                                ) as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerows(result["csv_rows"])
                            except (Exception, FileNotFoundError) as e:
                                # Log the error but continue processing
                                print(f"Error writing to CSV file: {str(e)}")

                except (Exception, ValueError) as e:
                    import traceback

                    traceback.print_exc()
                    self.error_occurred.emit(str(file_pair), str(e))

                # Update progress
                self.progress_updated.emit(int((i + 1) / total_files * 100))

        # Signal that processing is complete
        self.processing_finished.emit()

    def process_file_pair(self, file_pair):
        """Process a pair of files containing label images"""
        try:
            # Extract file paths
            filepath_c1 = file_pair[0]  # Channel 1
            filepath_c2 = file_pair[1]  # Channel 2
            filepath_c3 = (
                file_pair[2] if len(file_pair) > 2 else None
            )  # Channel 3 (optional)

            # Load label images
            image_c1 = tifffile.imread(filepath_c1)
            image_c2 = tifffile.imread(filepath_c2)
            image_c3 = tifffile.imread(filepath_c3) if filepath_c3 else None

            # Ensure all images have the same shape
            if image_c1.shape != image_c2.shape:
                raise ValueError(
                    f"Image shapes don't match: {image_c1.shape} vs {image_c2.shape}"
                )
            if filepath_c3 and image_c1.shape != image_c3.shape:
                raise ValueError(
                    f"Image shapes don't match: {image_c1.shape} vs {image_c3.shape}"
                )

            # Get base filename for the output
            base_filename = os.path.basename(filepath_c1)

            # Process colocalization
            results = self.process_colocalization(
                base_filename, image_c1, image_c2, image_c3
            )

            # Generate output image if needed
            if self.output_folder:
                self.save_output_image(results, file_pair)

            return results

        except (Exception, ValueError) as e:
            import traceback

            traceback.print_exc()
            raise ValueError(f"Error processing {file_pair}: {str(e)}") from e

    def process_colocalization(
        self, filename, image_c1, image_c2, image_c3=None
    ):
        """Process colocalization between channels"""
        # Get unique label IDs in image_c1
        label_ids = self.get_nonzero_labels(image_c1)

        # Pre-calculate sizes for image_c1 if needed
        roi_sizes = {}
        if self.get_sizes:
            roi_sizes = self.calculate_all_rois_size(image_c1)

        # Process each label
        csv_rows = []
        results = []

        for label_id in label_ids:
            row = self.process_single_roi(
                filename, label_id, image_c1, image_c2, image_c3, roi_sizes
            )
            csv_rows.append(row)

            # Extract results as dictionary
            result_dict = {"label_id": label_id, "ch2_in_ch1_count": row[2]}

            idx = 3
            if self.get_sizes:
                result_dict["ch1_size"] = row[idx]
                result_dict["ch2_in_ch1_size"] = row[idx + 1]
                idx += 2

            if image_c3 is not None:
                result_dict["ch3_in_ch2_in_ch1_count"] = row[idx]
                result_dict["ch3_not_in_ch2_but_in_ch1_count"] = row[idx + 1]
                idx += 2

                if self.get_sizes:
                    result_dict["ch3_in_ch2_in_ch1_size"] = row[idx]
                    result_dict["ch3_not_in_ch2_but_in_ch1_size"] = row[
                        idx + 1
                    ]

            results.append(result_dict)

        # Create output
        output = {
            "filename": filename,
            "results": results,
            "csv_rows": csv_rows,
        }

        return output

    def process_single_roi(
        self, filename, label_id, image_c1, image_c2, image_c3, roi_sizes
    ):
        """Process a single ROI for colocalization analysis."""
        # Create masks once
        mask_roi = image_c1 == label_id
        mask_c2 = image_c2 != 0

        # Calculate counts
        c2_in_c1_count = self.count_unique_nonzero(
            image_c2, mask_roi & mask_c2
        )

        # Build the result row
        row = [filename, int(label_id), c2_in_c1_count]

        # Add size information if requested
        if self.get_sizes:
            size = roi_sizes.get(int(label_id), 0)
            c2_in_c1_size = self.calculate_coloc_size(
                image_c1, image_c2, label_id
            )
            row.extend([size, c2_in_c1_size])

        # Handle third channel if present
        if image_c3 is not None:
            mask_c3 = image_c3 != 0

            # Calculate third channel statistics
            c3_in_c2_in_c1_count = self.count_unique_nonzero(
                image_c3, mask_roi & mask_c2 & mask_c3
            )
            c3_not_in_c2_but_in_c1_count = self.count_unique_nonzero(
                image_c3, mask_roi & ~mask_c2 & mask_c3
            )

            row.extend([c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count])

            # Add size information for third channel if requested
            if self.get_sizes:
                c3_in_c2_in_c1_size = self.calculate_coloc_size(
                    image_c1,
                    image_c2,
                    label_id,
                    mask_c2=True,
                    image_c3=image_c3,
                )
                c3_not_in_c2_but_in_c1_size = self.calculate_coloc_size(
                    image_c1,
                    image_c2,
                    label_id,
                    mask_c2=False,
                    image_c3=image_c3,
                )
                row.extend([c3_in_c2_in_c1_size, c3_not_in_c2_but_in_c1_size])

        return row

    def save_output_image(self, results, file_pair):
        """Generate and save visualization of colocalization results"""
        if not self.output_folder:
            return

        try:
            # Load images again to avoid memory issues
            filepath_c1 = file_pair[0]  # Channel 1
            image_c1 = tifffile.imread(filepath_c1)

            # Try to load channel 2 as well
            try:
                # filepath_c2 = file_pair[1]  # Channel 2
                # image_c2 = tifffile.imread(filepath_c2)
                has_c2 = True
            except (FileNotFoundError, IndexError):
                has_c2 = False

            # Try to load channel 3 if available
            has_c3 = False
            if len(file_pair) > 2:
                contextlib.suppress(FileNotFoundError, IndexError)

            # Create output filename
            channels_str = "_".join(self.channel_names)
            base_name = os.path.splitext(os.path.basename(filepath_c1))[0]
            output_path = os.path.join(
                self.output_folder, f"{base_name}_{channels_str}_coloc.tif"
            )

            # Create a more informative visualization
            # Start with the original first channel labels
            output_image = np.zeros((3,) + image_c1.shape, dtype=np.uint32)

            # First layer: original labels from channel 1
            output_image[0] = image_c1.copy()

            # Process results to create visualization
            if "results" in results:
                # Second layer: labels that have overlap with channel 2
                if has_c2:
                    ch2_overlap = np.zeros_like(image_c1)
                    for result in results["results"]:
                        label_id = result["label_id"]
                        if result["ch2_in_ch1_count"] > 0:
                            # This label has overlap with channel 2
                            mask = image_c1 == label_id
                            ch2_overlap[mask] = label_id
                    output_image[1] = ch2_overlap

                # Third layer: labels that have overlap with channel 3
                if has_c3:
                    ch3_overlap = np.zeros_like(image_c1)
                    for result in results["results"]:
                        label_id = result["label_id"]
                        if (
                            "ch3_in_ch2_in_ch1_count" in result
                            and result["ch3_in_ch2_in_ch1_count"] > 0
                        ):
                            # This label has overlap with channel 3
                            mask = image_c1 == label_id
                            ch3_overlap[mask] = label_id
                    output_image[2] = ch3_overlap

            # Save the visualization output
            tifffile.imwrite(output_path, output_image, compression="zlib")

            # Add the output path to the results
            results["output_path"] = output_path

        except (Exception, FileNotFoundError) as e:
            print(f"Error saving output image: {str(e)}")
            import traceback

            traceback.print_exc()

    # Helper functions
    def get_nonzero_labels(self, image):
        """Get unique, non-zero labels from an image."""
        mask = image != 0
        labels = np.unique(image[mask])
        return [int(x) for x in labels]

    def count_unique_nonzero(self, array, mask):
        """Count unique non-zero values in array where mask is True."""
        unique_vals = np.unique(array[mask])
        count = len(unique_vals)

        # Remove 0 from count if present
        if count > 0 and 0 in unique_vals:
            count -= 1

        return count

    def calculate_all_rois_size(self, image):
        """Calculate sizes of all ROIs in the given image."""
        sizes = {}
        try:
            # Convert to int32 to avoid potential overflow issues with regionprops
            image_int = image.astype(np.uint32)
            for prop in measure.regionprops(image_int):
                label = int(prop.label)
                sizes[label] = int(prop.area)
        except (Exception, ValueError) as e:
            print(f"Error calculating ROI sizes: {str(e)}")
        return sizes

    def calculate_coloc_size(
        self, image_c1, image_c2, label_id, mask_c2=None, image_c3=None
    ):
        """Calculate the size of colocalization between channels."""
        # Create mask for current ROI
        mask = image_c1 == int(label_id)

        # Handle mask_c2 parameter
        if mask_c2 is not None:
            if mask_c2:
                # sizes where c2 is present
                mask = mask & (image_c2 != 0)
                target_image = image_c3 if image_c3 is not None else image_c2
            else:
                # sizes where c2 is NOT present
                mask = mask & (image_c2 == 0)
                if image_c3 is None:
                    # If no image_c3, just return count of mask pixels
                    return np.count_nonzero(mask)
                target_image = image_c3
        else:
            target_image = image_c2

        # Calculate size of overlap
        masked_image = target_image * mask
        size = np.count_nonzero(masked_image)

        return int(size)

    def stop(self):
        """Request worker to stop processing"""
        self.stop_requested = True


class ColocalizationResultsWidget(QWidget):
    """Widget to display colocalization results"""

    def __init__(self, viewer, channel_names):
        super().__init__()
        self.viewer = viewer
        self.channel_names = channel_names
        self.file_results = {}  # Store results by filename

        # Create layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add information label at top
        info_label = QLabel(
            "Click on a result to view it in the viewer. For more detailed results please check the generated CSV file."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic;")
        self.layout.addWidget(info_label)

        # Create results table
        self.table = QTableWidget()
        self.table.setColumnCount(2)  # Just two columns
        self.table.setHorizontalHeaderLabels(["Identifier", "Coloc Count"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellClicked.connect(
            self.on_table_clicked
        )  # Connect cell click event
        self.layout.addWidget(self.table)

        # Add explanation for coloc count
        count_explanation = QLabel(
            "Coloc Count: Number of objects with colocalization"
        )
        count_explanation.setStyleSheet("font-style: italic;")
        self.layout.addWidget(count_explanation)

    def add_result(self, result):
        """Add a result to the table."""
        filename = result["filename"]
        self.file_results[filename] = result

        # Add to table
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Use the common substring as the identifier
        identifier = result.get("common_substring", filename)
        id_item = QTableWidgetItem(identifier)
        id_item.setToolTip(filename)  # Show full filename on hover
        id_item.setData(Qt.UserRole, filename)  # Store for reference
        self.table.setItem(row, 0, id_item)

        # Label count for colocalization
        if "csv_rows" in result and result["csv_rows"]:
            ch2_in_ch1_counts = [r[2] for r in result["csv_rows"]]
            total_coloc = sum(1 for c in ch2_in_ch1_counts if c > 0)
            count_item = QTableWidgetItem(f"{total_coloc} ")
        else:
            count_item = QTableWidgetItem("0 ")
        self.table.setItem(row, 1, count_item)

        # If there's an output file, store it with the row
        if "output_path" in result:
            # Store output path as data in all cells
            for col in range(2):
                item = self.table.item(row, col)
                if item:
                    item.setData(Qt.UserRole + 1, result["output_path"])

    def _extract_identifier(self, filename):
        """
        Extract the identifier for the given filename.

        This method assumes that the longest common substring (used as the key in
        `group_files_by_common_substring`) is already available in the results.
        """
        # Check if the filename exists in the results
        if filename in self.file_results:
            # Use the common substring (key) as the identifier
            return self.file_results[filename].get(
                "common_substring", filename
            )

        # Fallback to the base filename if no common substring is available
        return os.path.splitext(os.path.basename(filename))[0]

    def on_table_clicked(self, row, column):
        """Handle clicking on a table cell"""
        # Get the filename from the row
        filename_item = self.table.item(row, 0)
        if not filename_item:
            return

        filename = filename_item.data(Qt.UserRole)
        if filename not in self.file_results:
            return

        # Get the result object
        # result = self.file_results[filename]

        # Get output path if available (stored in UserRole+1)
        item = self.table.item(row, column)
        output_path = item.data(Qt.UserRole + 1) if item else None

        # Display result visualization
        if output_path and os.path.exists(output_path):
            # Clear existing layers
            self.viewer.layers.clear()

            # Load and display the visualization
            try:
                image = tifffile.imread(output_path)
                self.viewer.add_labels(
                    image,
                    name=f"Colocalization: {os.path.basename(output_path)}",
                )
                self.viewer.status = (
                    f"Loaded visualization for {os.path.basename(filename)}"
                )
            except (Exception, FileNotFoundError) as e:
                self.viewer.status = f"Error loading visualization: {str(e)}"
        else:
            self.viewer.status = "No visualization available for this result"


class ColocalizationAnalysisWidget(QWidget):
    """
    Widget for ROI colocalization analysis
    """

    def __init__(
        self, viewer: Viewer, channel_folders=None, channel_patterns=None
    ):
        super().__init__()
        self.viewer = viewer
        self.channel_folders = channel_folders or []
        self.channel_patterns = channel_patterns or []
        self.file_pairs = []  # Will hold matched files for analysis
        self.file_results = {}  # Store results by filename
        self.worker = None

        # Ensure default channel names are set
        self.channel_names = ["CH1", "CH2", "CH3"][
            : len(self.channel_folders) or 3
        ]

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Channel selection section
        # channels_layout = QFormLayout()

        # Channel 1 (primary/reference channel)
        self.ch1_label = QLabel("Channel 1 (Reference):")
        self.ch1_folder = QLineEdit()
        self.ch1_pattern = QLineEdit()
        self.ch1_pattern.setPlaceholderText("*_labels.tif")
        self.ch1_browse = QPushButton("Browse...")
        self.ch1_browse.clicked.connect(lambda: self.browse_folder(0))

        ch1_layout = QHBoxLayout()
        ch1_layout.addWidget(self.ch1_label)
        ch1_layout.addWidget(self.ch1_folder)
        ch1_layout.addWidget(self.ch1_pattern)
        ch1_layout.addWidget(self.ch1_browse)
        layout.addLayout(ch1_layout)

        # Channel 2
        self.ch2_label = QLabel("Channel 2:")
        self.ch2_folder = QLineEdit()
        self.ch2_pattern = QLineEdit()
        self.ch2_pattern.setPlaceholderText("*_labels.tif")
        self.ch2_browse = QPushButton("Browse...")
        self.ch2_browse.clicked.connect(lambda: self.browse_folder(1))

        ch2_layout = QHBoxLayout()
        ch2_layout.addWidget(self.ch2_label)
        ch2_layout.addWidget(self.ch2_folder)
        ch2_layout.addWidget(self.ch2_pattern)
        ch2_layout.addWidget(self.ch2_browse)
        layout.addLayout(ch2_layout)

        # Channel 3 (optional)
        self.ch3_label = QLabel("Channel 3 (Optional):")
        self.ch3_folder = QLineEdit()
        self.ch3_pattern = QLineEdit()
        self.ch3_pattern.setPlaceholderText("*_labels.tif")
        self.ch3_browse = QPushButton("Browse...")
        self.ch3_browse.clicked.connect(lambda: self.browse_folder(2))

        ch3_layout = QHBoxLayout()
        ch3_layout.addWidget(self.ch3_label)
        ch3_layout.addWidget(self.ch3_folder)
        ch3_layout.addWidget(self.ch3_pattern)
        ch3_layout.addWidget(self.ch3_browse)
        layout.addLayout(ch3_layout)

        # Analysis options
        options_layout = QFormLayout()

        # Get sizes option
        self.get_sizes_checkbox = QCheckBox("Calculate Region Sizes")
        options_layout.addRow(self.get_sizes_checkbox)

        # Size calculation method
        self.size_method_layout = QHBoxLayout()
        self.size_method_label = QLabel("Size Calculation Method:")
        self.size_method_median = QCheckBox("Median")
        self.size_method_median.setChecked(True)
        self.size_method_sum = QCheckBox("Sum")

        # Connect to make them mutually exclusive
        self.size_method_median.toggled.connect(
            lambda checked: (
                self.size_method_sum.setChecked(not checked)
                if checked
                else None
            )
        )
        self.size_method_sum.toggled.connect(
            lambda checked: (
                self.size_method_median.setChecked(not checked)
                if checked
                else None
            )
        )

        self.size_method_layout.addWidget(self.size_method_label)
        self.size_method_layout.addWidget(self.size_method_median)
        self.size_method_layout.addWidget(self.size_method_sum)
        options_layout.addRow(self.size_method_layout)

        layout.addLayout(options_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        self.output_folder = QLineEdit()
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self.browse_output)

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_folder)
        output_layout.addWidget(output_browse)
        layout.addLayout(output_layout)

        # Thread count selector
        thread_layout = QHBoxLayout()
        thread_label = QLabel("Number of threads:")
        thread_layout.addWidget(thread_label)

        self.thread_count = QSpinBox()
        self.thread_count.setMinimum(1)
        self.thread_count.setMaximum(os.cpu_count() or 4)
        self.thread_count.setValue(max(1, (os.cpu_count() or 4) - 1))
        thread_layout.addWidget(self.thread_count)

        layout.addLayout(thread_layout)

        # Find matching files button
        find_button = QPushButton("Find Matching Files")
        find_button.clicked.connect(self.find_matching_files)
        layout.addWidget(find_button)

        # Match results label
        self.match_label = QLabel("No files matched yet")
        layout.addWidget(self.match_label)

        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Start/cancel buttons
        button_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Start Colocalization Analysis")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)  # Disabled until files are found

        self.cancel_button = QPushButton("Cancel Analysis")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setEnabled(False)  # Disabled initially

        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Results widget (will be created when needed)
        self.results_widget = None

        # Fill in values if provided
        if self.channel_folders:
            if len(self.channel_folders) > 0:
                self.ch1_folder.setText(self.channel_folders[0])
            if len(self.channel_folders) > 1:
                self.ch2_folder.setText(self.channel_folders[1])
            if len(self.channel_folders) > 2:
                self.ch3_folder.setText(self.channel_folders[2])

        if self.channel_patterns:
            if len(self.channel_patterns) > 0:
                self.ch1_pattern.setText(self.channel_patterns[0])
            if len(self.channel_patterns) > 1:
                self.ch2_pattern.setText(self.channel_patterns[1])
            if len(self.channel_patterns) > 2:
                self.ch3_pattern.setText(self.channel_patterns[2])

    def browse_folder(self, channel_index):
        """Browse for a channel folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Channel Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            if channel_index == 0:
                self.ch1_folder.setText(folder)
            elif channel_index == 1:
                self.ch2_folder.setText(folder)
            elif channel_index == 2:
                self.ch3_folder.setText(folder)

    def browse_output(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            self.output_folder.setText(folder)

    def find_matching_files(self):
        """Find matching files across channels using the updated grouping function."""
        # Get channel folders and patterns
        ch1_folder = self.ch1_folder.text().strip()
        ch1_pattern = self.ch1_pattern.text().strip() or "*_labels.tif"

        ch2_folder = self.ch2_folder.text().strip()
        ch2_pattern = self.ch2_pattern.text().strip() or "*_labels.tif"

        ch3_folder = self.ch3_folder.text().strip()
        ch3_pattern = self.ch3_pattern.text().strip() or "*_labels.tif"

        # Validate required folders
        if not ch1_folder or not os.path.isdir(ch1_folder):
            self.status_label.setText(
                "Channel 1 folder is required and must exist"
            )
            return

        if not ch2_folder or not os.path.isdir(ch2_folder):
            self.status_label.setText(
                "Channel 2 folder is required and must exist"
            )
            return

        # Find files in each folder
        import glob

        ch1_files = sorted(glob.glob(os.path.join(ch1_folder, ch1_pattern)))
        ch2_files = sorted(glob.glob(os.path.join(ch2_folder, ch2_pattern)))

        # Check if third channel is provided
        use_ch3 = bool(ch3_folder and os.path.isdir(ch3_folder))
        if use_ch3:
            ch3_files = sorted(
                glob.glob(os.path.join(ch3_folder, ch3_pattern))
            )
        else:
            ch3_files = []

        # Prepare file lists for grouping
        file_lists = {
            "CH1": ch1_files,
            "CH2": ch2_files,
        }
        if use_ch3:
            file_lists["CH3"] = ch3_files

        # Group files by common substring
        grouped_files = group_files_by_common_substring(
            file_lists, list(file_lists.keys())
        )

        # Convert grouped files into file pairs/triplets and store the common substring
        self.file_pairs = []
        for common_substring, files in grouped_files.items():
            print(f"Group key (common substring): {common_substring}")
            self.file_pairs.append(tuple(files))
            for file in files:
                # Store the stripped common substring in the results
                self.file_results[file] = {
                    "common_substring": common_substring
                }
                print(f"Stored {file} with group key: {common_substring}")

        # Update status
        if self.file_pairs:
            count = len(self.file_pairs)
            channels = 3 if use_ch3 else 2
            self.match_label.setText(
                f"Found {count} matching file sets across {channels} channels"
            )
            self.analyze_button.setEnabled(True)
            self.status_label.setText("Ready to analyze")
        else:
            self.match_label.setText("No matching files found across channels")
            self.analyze_button.setEnabled(False)
            self.status_label.setText("No files to analyze")

    def start_analysis(self):
        """Start the colocalization analysis"""
        if not self.file_pairs:
            self.status_label.setText("No file pairs to analyze")
            return

        # Get settings
        get_sizes = self.get_sizes_checkbox.isChecked()
        size_method = (
            "median" if self.size_method_median.isChecked() else "sum"
        )
        output_folder = self.output_folder.text().strip()

        # Create output folder if it doesn't exist and is specified
        if output_folder:
            try:
                # Create all necessary directories
                os.makedirs(output_folder, exist_ok=True)

                # Try to create a test file to check write permissions
                test_path = os.path.join(output_folder, ".test_write")
                try:
                    with open(test_path, "w") as f:
                        f.write("test")
                    os.remove(test_path)  # Clean up after test
                except (PermissionError, OSError) as e:
                    self.status_label.setText(
                        f"Cannot write to output folder: {str(e)}"
                    )
                    return

            except (OSError, PermissionError) as e:
                self.status_label.setText(
                    f"Error creating output folder: {str(e)}"
                )
                return

        # Update UI
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.analyze_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Create worker thread
        self.worker = ColocalizationWorker(
            self.file_pairs,
            self.channel_names,
            get_sizes,
            size_method,
            output_folder,
        )

        # Set thread count
        self.worker.thread_count = self.thread_count.value()

        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.file_processed.connect(self.file_processed)
        self.worker.processing_finished.connect(self.processing_finished)
        self.worker.error_occurred.connect(self.processing_error)

        # Start processing
        self.worker.start()

        # Update status
        self.status_label.setText(
            f"Processing {len(self.file_pairs)} file pairs with {self.thread_count.value()} threads"
        )

        # Create results widget if needed
        if not self.results_widget:
            self.results_widget = ColocalizationResultsWidget(
                self.viewer, self.channel_names
            )
            self.viewer.window.add_dock_widget(
                self.results_widget,
                name="Colocalization Results",
                area="right",
            )

    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)

    def file_processed(self, result):
        """Handle a processed file result"""
        if self.results_widget:
            self.results_widget.add_result(result)

    def processing_finished(self):
        """Handle processing completion"""
        # Update UI
        self.progress_bar.setValue(100)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        # Clean up worker - safely
        if self.worker:
            if self.worker.isRunning():
                # This shouldn't happen, but just in case
                self.worker.stop()
                self.worker.wait()
            self.worker = None

        # Update status
        self.status_label.setText("Analysis complete")

        # Hide progress bar after a delay - use QTimer instead of threading
        from qtpy.QtCore import QTimer

        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    def processing_error(self, filepath, error_msg):
        """Handle processing errors"""
        print(f"Error processing {filepath}: {error_msg}")
        self.status_label.setText(f"Error: {error_msg}")

    def cancel_analysis(self):
        """Cancel the current processing operation"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # Wait for the worker to finish with timeout
            if not self.worker.wait(1000):  # Wait up to 1 second
                # Force termination if it doesn't respond
                self.worker.terminate()
                self.worker.wait()

            # Clear the worker reference
            self.worker = None

            # Update UI
            self.analyze_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Analysis cancelled")
            self.progress_bar.setVisible(False)


# This is the key change: use magic_factory to create a widget that Napari can understand
@magic_factory(call_button="Start ROI Colocalization Analysis")
def roi_colocalization_analyzer(viewer: Viewer):
    """
    Analyze colocalization between ROIs in multiple channel label images.

    This tool helps find and measure overlaps between labeled regions across
    different channels, generating statistics such as overlap counts and sizes.
    """
    # Create the analysis widget
    analysis_widget = ColocalizationAnalysisWidget(viewer)

    # Add to viewer
    viewer.window.add_dock_widget(
        analysis_widget, name="ROI Colocalization Analysis", area="right"
    )

    # Instead of using destroyed signal which doesn't exist,
    # we can use the removed event from napari's dock widget
    def _on_widget_removed(event):
        if hasattr(analysis_widget, "closeEvent"):
            # Call closeEvent to properly clean up
            analysis_widget.closeEvent(None)

    # Make sure we clean up on our own closeEvent as well
    original_close = getattr(analysis_widget, "closeEvent", lambda x: None)

    def enhanced_close_event(event):
        # Make sure worker threads are stopped
        if (
            hasattr(analysis_widget, "worker")
            and analysis_widget.worker
            and analysis_widget.worker.isRunning()
        ):
            analysis_widget.worker.stop()
            if not analysis_widget.worker.wait(1000):
                analysis_widget.worker.terminate()
                analysis_widget.worker.wait()
            analysis_widget.worker = None

        # Call original closeEvent
        original_close(event)

    # Replace the closeEvent
    analysis_widget.closeEvent = enhanced_close_event

    return analysis_widget

"""
Batch Microscopy Image File Conversion
=======================================
This module provides a GUI for batch conversion of microscopy image files to a common format.
The user can select a folder containing microscopy image files, preview the images, and convert them to an open format for image processing.
The supported input formats include Leica LIF, Nikon ND2, Zeiss CZI, and TIFF-based whole slide images (NDPI, etc.).

"""

import concurrent.futures
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask.array as da
import napari
import nd2  # https://github.com/tlambert03/nd2
import numpy as np
import tifffile
import zarr
from dask.diagnostics import ProgressBar
from magicgui import magicgui
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from pylibCZIrw import czi  # https://github.com/ZEISS/pylibczirw
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Format-specific readers
from readlif.reader import (
    LifFile,  # https://github.com/Arcadia-Science/readlif
)
from tiffslide import TiffSlide  # https://github.com/Bayer-Group/tiffslide


class SeriesTableWidget(QTableWidget):
    """
    Custom table widget to display original files and their series
    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # Configure table
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Original Files", "Series"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Track file mappings
        self.file_data = (
            {}
        )  # {filepath: {type: file_type, series: [list_of_series]}}

        # Currently loaded images
        self.current_file = None
        self.current_series = None

        # Connect selection signals
        self.cellClicked.connect(self.handle_cell_click)

    def add_file(self, filepath: str, file_type: str, series_count: int):
        """Add a file to the table with series information"""
        row = self.rowCount()
        self.insertRow(row)

        # Original file item
        original_item = QTableWidgetItem(os.path.basename(filepath))
        original_item.setData(Qt.UserRole, filepath)
        self.setItem(row, 0, original_item)

        # Series info
        series_info = (
            f"{series_count} series"
            if series_count >= 0
            else "Not a series file"
        )
        series_item = QTableWidgetItem(series_info)
        self.setItem(row, 1, series_item)

        # Store file info
        self.file_data[filepath] = {
            "type": file_type,
            "series_count": series_count,
            "row": row,
        }

    def handle_cell_click(self, row: int, column: int):
        """Handle cell click to show series details or load image"""
        if column == 0:
            # Get filepath from the clicked cell
            item = self.item(row, 0)
            if item:
                filepath = item.data(Qt.UserRole)
                file_info = self.file_data.get(filepath)

                if file_info and file_info["series_count"] > 0:
                    # Update the current file
                    self.current_file = filepath

                    # Signal to show series details
                    self.parent().show_series_details(filepath)
                else:
                    # Not a series file, just load the image
                    self.parent().load_image(filepath)


class SeriesDetailWidget(QWidget):
    """Widget to display and select series from a file"""

    def __init__(self, parent, viewer: napari.Viewer):
        super().__init__()
        self.parent = parent
        self.viewer = viewer
        self.current_file = None
        self.max_series = 0

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Series selection widgets
        self.series_label = QLabel("Select Series:")
        layout.addWidget(self.series_label)

        self.series_selector = QComboBox()
        layout.addWidget(self.series_selector)

        # Add "Export All Series" checkbox
        self.export_all_checkbox = QCheckBox("Export All Series")
        self.export_all_checkbox.toggled.connect(self.toggle_export_all)
        layout.addWidget(self.export_all_checkbox)

        # Connect series selector
        self.series_selector.currentIndexChanged.connect(self.series_selected)

        # Add preview button
        preview_button = QPushButton("Preview Selected Series")
        preview_button.clicked.connect(self.preview_series)
        layout.addWidget(preview_button)

        # Add info label
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

    def toggle_export_all(self, checked):
        """Handle toggle of export all checkbox"""
        if self.current_file and checked:
            # Disable series selector when exporting all
            self.series_selector.setEnabled(not checked)
            # Update parent with export all setting
            self.parent.set_export_all_series(self.current_file, checked)
        elif self.current_file:
            # Re-enable series selector
            self.series_selector.setEnabled(True)
            # Update parent with currently selected series only
            self.series_selected(self.series_selector.currentIndex())
            # Update parent to not export all
            self.parent.set_export_all_series(self.current_file, False)

    def set_file(self, filepath: str):
        """Set the current file and update series list"""
        self.current_file = filepath
        self.series_selector.clear()

        # Reset export all checkbox
        self.export_all_checkbox.setChecked(False)
        self.series_selector.setEnabled(True)

        # Try to get series information
        file_loader = self.parent.get_file_loader(filepath)
        if file_loader:
            try:
                series_count = file_loader.get_series_count(filepath)
                self.max_series = series_count
                for i in range(series_count):
                    self.series_selector.addItem(f"Series {i}", i)

                # Set info text
                if series_count > 0:
                    # metadata = file_loader.get_metadata(filepath, 0)

                    # Estimate file size and set appropriate format radio button
                    file_type = self.parent.get_file_type(filepath)
                    if file_type == "ND2":
                        try:
                            with nd2.ND2File(filepath) as nd2_file:
                                dims = dict(nd2_file.sizes)
                                pixel_size = nd2_file.dtype.itemsize
                                total_elements = np.prod(
                                    [dims[dim] for dim in dims]
                                )
                                size_GB = (total_elements * pixel_size) / (
                                    1024**3
                                )

                                self.info_label.setText(
                                    f"File contains {series_count} series (size: {size_GB:.2f}GB)"
                                )

                                # Update format buttons
                                self.parent.update_format_buttons(size_GB > 4)
                        except (ValueError, FileNotFoundError) as e:
                            print(f"Error estimating file size: {e}")
            except FileNotFoundError:
                self.info_label.setText("File not found.")
            except PermissionError:
                self.info_label.setText(
                    "Permission denied when accessing the file."
                )
            except ValueError as e:
                self.info_label.setText(f"Invalid data in file: {str(e)}")
            except OSError as e:
                self.info_label.setText(f"I/O error occurred: {str(e)}")

    def series_selected(self, index: int):
        """Handle series selection"""
        if index >= 0 and self.current_file:
            series_index = self.series_selector.itemData(index)

            # Validate series index
            if series_index >= self.max_series:
                self.info_label.setText(
                    f"Error: Series index {series_index} out of range (max: {self.max_series-1})"
                )
                return

            # Update parent with selected series
            self.parent.set_selected_series(self.current_file, series_index)

            # Automatically set the appropriate format radio button based on file size
            file_loader = self.parent.get_file_loader(self.current_file)
            if file_loader:
                try:
                    # Estimate file size based on metadata
                    # metadata = file_loader.get_metadata(
                    #    self.current_file, series_index
                    # )

                    # For ND2 files, we can directly check the size
                    if self.parent.get_file_type(self.current_file) == "ND2":
                        with nd2.ND2File(self.current_file) as nd2_file:
                            dims = dict(nd2_file.sizes)
                            pixel_size = nd2_file.dtype.itemsize
                            total_elements = np.prod(
                                [dims[dim] for dim in dims]
                            )
                            size_GB = (total_elements * pixel_size) / (1024**3)

                            # Automatically set the appropriate radio button based on size
                            self.parent.update_format_buttons(size_GB > 4)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Error estimating file size: {e}")

    def preview_series(self):
        """Preview the selected series in Napari"""
        if self.current_file and self.series_selector.currentIndex() >= 0:
            series_index = self.series_selector.itemData(
                self.series_selector.currentIndex()
            )

            # Validate series index
            if series_index >= self.max_series:
                self.info_label.setText(
                    f"Error: Series index {series_index} out of range (max: {self.max_series-1})"
                )
                return

            file_loader = self.parent.get_file_loader(self.current_file)

            try:
                # First get metadata to understand dimensions
                metadata = file_loader.get_metadata(
                    self.current_file, series_index
                )

                # Load the series
                image_data = file_loader.load_series(
                    self.current_file, series_index
                )

                # Reorder dimensions for Napari based on metadata
                if metadata and "axes" in metadata:
                    print(f"File has dimension order: {metadata['axes']}")
                    # Target dimension order for Napari
                    napari_order = "CTZYX"[: len(image_data.shape)]
                    image_data = self._reorder_dimensions(
                        image_data, metadata, napari_order
                    )

                # Clear existing layers and display the image
                self.viewer.layers.clear()
                self.viewer.add_image(
                    image_data,
                    name=f"{Path(self.current_file).stem} - Series {series_index}",
                )

                # Update status
                self.viewer.status = f"Previewing {Path(self.current_file).name} - Series {series_index}"
            except (ValueError, FileNotFoundError) as e:
                self.viewer.status = f"Error loading series: {str(e)}"
                QMessageBox.warning(
                    self, "Error", f"Could not load series: {str(e)}"
                )

    def _reorder_dimensions(self, image_data, metadata, target_order="YXZTC"):
        """Reorder dimensions based on metadata axes information

        Args:
            image_data: The numpy or dask array to reorder
            metadata: Metadata dictionary containing axes information
            target_order: Target dimension order (e.g., "YXZTC")

        Returns:
            Reordered array
        """
        # Early exit if no metadata or no axes information
        if not metadata or "axes" not in metadata:
            print("No axes information in metadata - returning original")
            return image_data

        # Get source order from metadata
        source_order = metadata["axes"]

        # Ensure dimensions match
        ndim = len(image_data.shape)
        if len(source_order) != ndim:
            print(
                f"Dimension mismatch - array has {ndim} dims but axes metadata indicates {len(source_order)}"
            )
            return image_data

        # Ensure target order has the same number of dimensions
        if len(target_order) != ndim:
            print(
                f"Target order {target_order} doesn't match array dimensions {ndim}"
            )
            return image_data

        # Create reordering index list
        reorder_indices = []
        for axis in target_order:
            if axis in source_order:
                reorder_indices.append(source_order.index(axis))
            else:
                print(f"Axis {axis} not found in source order {source_order}")
                return image_data

        # Reorder the array using appropriate method
        try:
            print(f"Reordering from {source_order} to {target_order}")

            # Check if using Dask array
            if hasattr(image_data, "dask"):
                # Use Dask's transpose to preserve lazy computation
                reordered = image_data.transpose(reorder_indices)
            else:
                # Use numpy transpose
                reordered = np.transpose(image_data, reorder_indices)

            print(f"Reordered shape: {reordered.shape}")
            return reordered
        except (ValueError, IndexError) as e:
            print(f"Error reordering dimensions: {e}")
            return image_data


class FormatLoader:
    """Base class for format loaders"""

    @staticmethod
    def can_load(filepath: str) -> bool:
        raise NotImplementedError()

    @staticmethod
    def get_series_count(filepath: str) -> int:
        raise NotImplementedError()

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        return {}


class LIFLoader(FormatLoader):
    """Loader for Leica LIF files"""

    @staticmethod
    def can_load(filepath: str) -> bool:
        return filepath.lower().endswith(".lif")

    @staticmethod
    def get_series_count(filepath: str) -> int:
        try:
            lif_file = LifFile(filepath)
            # Directly use the iterator, no need to load all images into a list
            return sum(1 for _ in lif_file.get_iter_image())
        except (ValueError, FileNotFoundError):
            return 0

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        lif_file = LifFile(filepath)
        image = lif_file.get_image(series_index)

        # Extract dimensions
        channels = image.channels
        z_stacks = image.nz
        timepoints = image.nt
        x_dim, y_dim = image.dims[0], image.dims[1]

        # Create an array to hold the entire series
        series_shape = (
            timepoints,
            z_stacks,
            channels,
            y_dim,
            x_dim,
        )  # Corrected shape
        series_data = np.zeros(series_shape, dtype=np.uint16)

        # Populate the array
        missing_frames = 0
        for t in range(timepoints):
            for z in range(z_stacks):
                for c in range(channels):
                    # Get the frame and convert to numpy array
                    frame = image.get_frame(z=z, t=t, c=c)
                    if frame:
                        series_data[t, z, c, :, :] = np.array(frame)
                    else:
                        missing_frames += 1
                        series_data[t, z, c, :, :] = np.zeros(
                            (y_dim, x_dim), dtype=np.uint16
                        )

        if missing_frames > 0:
            print(
                f"Warning: {missing_frames} frames were missing and filled with zeros."
            )

        return series_data

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        try:
            lif_file = LifFile(filepath)
            image = lif_file.get_image(series_index)
            axes = "".join(image.dims._fields).upper()
            channels = image.channels
            if channels > 1:
                # add C to end of string
                axes += "C"

            metadata = {
                # "channels": image.channels,
                # "z_stacks": image.nz,
                # "timepoints": image.nt,
                "axes": "TZCYX",
                "unit": "um",
                "resolution": image.scale[:2],
            }
            if image.scale[2] is not None:
                metadata["spacing"] = image.scale[2]
            return metadata
        except (ValueError, FileNotFoundError):
            return {}


class ND2Loader(FormatLoader):
    """Loader for Nikon ND2 files"""

    @staticmethod
    def can_load(filepath: str) -> bool:
        return filepath.lower().endswith(".nd2")

    @staticmethod
    def get_series_count(filepath: str) -> int:
        # ND2 files typically have a single series with multiple channels/dimensions
        return 1

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        if series_index != 0:
            raise ValueError("ND2 files only support series index 0")

        # First open the file to check metadata
        with nd2.ND2File(filepath) as nd2_file:
            # Calculate size in GB
            total_size = np.prod(
                [nd2_file.sizes[dim] for dim in nd2_file.sizes]
            )
            pixel_size = nd2_file.dtype.itemsize
            size_GB = (total_size * pixel_size) / (1024**3)

            print(f"ND2 file dimensions: {nd2_file.sizes}")
            print(f"Pixel type: {nd2_file.dtype}, size: {pixel_size} bytes")
            print(f"Estimated file size: {size_GB:.2f} GB")

        if size_GB > 4:
            print("Using Dask for large file")
            # Load as Dask array for large files
            data = nd2.imread(filepath, dask=True)
            return data
        else:
            print("Using direct loading for smaller file")
            # Load directly into memory for smaller files
            data = nd2.imread(filepath)
            return data

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        if series_index != 0:
            return {}

        with nd2.ND2File(filepath) as nd2_file:
            # Get all dimensions and their sizes
            dims = dict(nd2_file.sizes)

            # Create a more detailed axes representation
            axes = "".join(dims.keys())

            print(f"ND2 metadata - dims: {dims}")
            print(f"ND2 metadata - axes: {axes}")

            # Get spatial dimensions and convert to resolution
            try:
                voxel = nd2_file.voxel_size()
                x_res = 1 / voxel.x if voxel.x > 0 else 1.0
                y_res = 1 / voxel.y if voxel.y > 0 else 1.0
                z_spacing = 1 / voxel.z if voxel.z > 0 else 1.0

                print(f"Voxel size: x={voxel.x}, y={voxel.y}, z={voxel.z}")
            except (ValueError, AttributeError) as e:
                print(f"Error getting voxel size: {e}")
                x_res, y_res, z_spacing = 1.0, 1.0, 1.0

            # Create scale information for all dimensions
            scales = {}
            if "T" in dims:
                scales["scale_t"] = 1.0
            if "Z" in dims:
                scales["scale_z"] = z_spacing
            if "C" in dims:
                scales["scale_c"] = 1.0
            if "Y" in dims:
                scales["scale_y"] = y_res
            if "X" in dims:
                scales["scale_x"] = x_res

            # Build comprehensive metadata
            metadata = {
                "axes": axes,
                "dimensions": dims,
                "resolution": (x_res, y_res),
                "unit": "um",
                "spacing": z_spacing,
                **scales,  # Add all scale information
            }

            # Add extra metadata for zarr transformations
            ordered_scales = []
            for ax in axes:
                scale_key = f"scale_{ax.lower()}"
                ordered_scales.append(scales.get(scale_key, 1.0))

            metadata["scales"] = ordered_scales

            return metadata


class TIFFSlideLoader(FormatLoader):
    """Loader for whole slide TIFF images (NDPI, etc.)"""

    @staticmethod
    def can_load(filepath: str) -> bool:
        ext = filepath.lower()
        return ext.endswith(".ndpi")

    @staticmethod
    def get_series_count(filepath: str) -> int:
        try:
            with TiffSlide(filepath) as slide:
                # NDPI typically has a main image and several levels (pyramid)
                return len(slide.level_dimensions)
        except (ValueError, FileNotFoundError):
            # Try standard tifffile if TiffSlide fails
            try:
                with tifffile.TiffFile(filepath) as tif:
                    return len(tif.series)
            except (ValueError, FileNotFoundError):
                return 0

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        try:
            # First try TiffSlide for whole slide images
            with TiffSlide(filepath) as slide:
                if series_index < 0 or series_index >= len(
                    slide.level_dimensions
                ):
                    raise ValueError(
                        f"Series index {series_index} out of range"
                    )

                # Get dimensions for the level
                width, height = slide.level_dimensions[series_index]
                # Read the entire level
                return np.array(
                    slide.read_region((0, 0), series_index, (width, height))
                )
        except (ValueError, FileNotFoundError):
            # Fall back to tifffile
            with tifffile.TiffFile(filepath) as tif:
                if series_index < 0 or series_index >= len(tif.series):
                    raise ValueError(
                        f"Series index {series_index} out of range"
                    ) from None

                return tif.series[series_index].asarray()

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        try:
            with TiffSlide(filepath) as slide:
                if series_index < 0 or series_index >= len(
                    slide.level_dimensions
                ):
                    return {}

                return {
                    "axes": slide.properties["tiffslide.series-axes"],
                    "resolution": (
                        slide.properties["tiffslide.mpp-x"],
                        slide.properties["tiffslide.mpp-y"],
                    ),
                    "unit": "um",
                }
        except (ValueError, FileNotFoundError):
            # Fall back to tifffile
            with tifffile.TiffFile(filepath) as tif:
                if series_index < 0 or series_index >= len(tif.series):
                    return {}

                series = tif.series[series_index]
                return {
                    "shape": series.shape,
                    "dtype": str(series.dtype),
                    "axes": series.axes,
                }


class CZILoader(FormatLoader):
    """Loader for Zeiss CZI files
    https://github.com/ZEISS/pylibczirw
    """

    @staticmethod
    def can_load(filepath: str) -> bool:
        return filepath.lower().endswith(".czi")

    @staticmethod
    def get_series_count(filepath: str) -> int:
        try:
            with czi.open_czi(filepath) as czi_file:
                scenes = czi_file.scenes_bounding_rectangle
                return len(scenes)
        except (ValueError, FileNotFoundError):
            return 0

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        try:
            with czi.open_czi(filepath) as czi_file:
                scenes = czi_file.scenes_bounding_rectangle

                if series_index < 0 or series_index >= len(scenes):
                    raise ValueError(
                        f"Scene index {series_index} out of range"
                    )

                scene_keys = list(scenes.keys())
                scene_index = scene_keys[series_index]

                # You might need to specify pixel_type if automatic detection fails
                image = czi_file.read(scene=scene_index)
                return image
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading series: {e}")
            raise  # Re-raise the exception after logging

    @staticmethod
    def get_scales(metadata_xml, dim):
        pattern = re.compile(
            r'<Distance[^>]*Id="'
            + re.escape(dim)
            + r'"[^>]*>.*?<Value[^>]*>(.*?)</Value>',
            re.DOTALL,
        )
        match = pattern.search(metadata_xml)

        if match:
            scale = float(match.group(1))
            # convert to microns
            scale = scale * 1e6
            return scale
        else:
            return None  # Fixed: return a single None value instead of (None, None, None)

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        try:
            with czi.open_czi(filepath) as czi_file:
                scenes = czi_file.scenes_bounding_rectangle

                if series_index < 0 or series_index >= len(scenes):
                    return {}

                # scene_keys = list(scenes.keys())
                # scene_index = scene_keys[series_index]
                # scene = scenes[scene_index]

                dims = czi_file.total_bounding_box

                # Extract the raw metadata as an XML string
                metadata_xml = czi_file.raw_metadata

                # Initialize metadata with default values
                try:
                    # scales are in meters, convert to microns
                    scale_x = CZILoader.get_scales(metadata_xml, "X") * 1e6
                    scale_y = CZILoader.get_scales(metadata_xml, "Y") * 1e6

                    filtered_dims = {
                        k: v for k, v in dims.items() if v != (0, 1)
                    }
                    axes = "".join(filtered_dims.keys())
                    metadata = {
                        "axes": axes,
                        "resolution": (scale_x, scale_y),
                        "unit": "um",
                    }

                    if dims["Z"] != (0, 1):
                        scale_z = CZILoader.get_scales(metadata_xml, "Z")
                        metadata["spacing"] = scale_z
                except ValueError as e:
                    print(f"Error getting scale metadata: {e}")

                return metadata

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"Error getting metadata: {e}")
            return {}

    @staticmethod
    def get_physical_pixel_size(
        filepath: str, series_index: int
    ) -> Dict[str, float]:
        try:
            with czi.open_czi(filepath) as czi_file:
                scenes = czi_file.scenes_bounding_rectangle

                if series_index < 0 or series_index >= len(scenes):
                    raise ValueError(
                        f"Scene index {series_index} out of range"
                    )

                # scene_keys = list(scenes.keys())
                # scene_index = scene_keys[series_index]

                # Get scale information
                scale_x = czi_file.scale_x
                scale_y = czi_file.scale_y
                scale_z = czi_file.scale_z

                return {"X": scale_x, "Y": scale_y, "Z": scale_z}
        except (ValueError, FileNotFoundError) as e:
            print(f"Error getting pixel size: {str(e)}")
            return {}


class AcquiferLoader(FormatLoader):
    """Loader for Acquifer datasets using the acquifer_napari_plugin utility"""

    # Cache for loaded datasets to avoid reloading the same directory multiple times
    _dataset_cache = {}  # {directory_path: xarray_dataset}

    @staticmethod
    def can_load(filepath: str) -> bool:
        """
        Check if this is a directory that can be loaded as an Acquifer dataset
        """
        if not os.path.isdir(filepath):
            return False

        try:

            # Check if directory contains files
            image_files = []
            for root, _, files in os.walk(filepath):
                for file in files:
                    if file.lower().endswith(
                        (".tif", ".tiff", ".png", ".jpg", ".jpeg")
                    ):
                        image_files.append(os.path.join(root, file))

            return bool(image_files)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error checking Acquifer dataset: {e}")
            return False

    @staticmethod
    def _load_dataset(directory):
        """Load the dataset using array_from_directory and cache it"""
        if directory in AcquiferLoader._dataset_cache:
            return AcquiferLoader._dataset_cache[directory]

        try:
            from acquifer_napari_plugin.utils import array_from_directory

            # Check if directory contains files before trying to load
            image_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(
                        (".tif", ".tiff", ".png", ".jpg", ".jpeg")
                    ):
                        image_files.append(os.path.join(root, file))

            if not image_files:
                raise ValueError(
                    f"No image files found in directory: {directory}"
                )

            dataset = array_from_directory(directory)
            AcquiferLoader._dataset_cache[directory] = dataset
            return dataset
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading Acquifer dataset: {e}")
            raise ValueError(f"Failed to load Acquifer dataset: {e}") from e

    @staticmethod
    def get_series_count(filepath: str) -> int:
        """
        Return the number of wells as series count
        """
        try:
            dataset = AcquiferLoader._load_dataset(filepath)

            # Check for Well dimension
            if "Well" in dataset.dims:
                return len(dataset.coords["Well"])
            else:
                # Single series for the whole dataset
                return 1
        except (ValueError, FileNotFoundError) as e:
            print(f"Error getting series count: {e}")
            return 0

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        """
        Load a specific well as a series
        """
        try:
            dataset = AcquiferLoader._load_dataset(filepath)

            # If the dataset has a Well dimension, select the specific well
            if "Well" in dataset.dims:
                if series_index < 0 or series_index >= len(
                    dataset.coords["Well"]
                ):
                    raise ValueError(
                        f"Series index {series_index} out of range"
                    )

                # Get the well value at this index
                well_value = dataset.coords["Well"].values[series_index]

                # Select the data for this well
                well_data = dataset.sel(Well=well_value)
                # squeeze out singleton dimensions
                well_data = well_data.squeeze()
                # Convert to numpy array and return
                return well_data.values
            else:
                # No Well dimension, return the entire dataset
                return dataset.values

        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading series: {e}")
            import traceback

            traceback.print_exc()
            raise ValueError(f"Failed to load series: {e}") from e

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        """
        Extract metadata for a specific well
        """
        try:
            dataset = AcquiferLoader._load_dataset(filepath)

            # Initialize with default values
            axes = ""
            resolution = (1.0, 1.0)  # Default resolution

            if "Well" in dataset.dims:
                well_value = dataset.coords["Well"].values[series_index]
                well_data = dataset.sel(Well=well_value)
                well_data = well_data.squeeze()  # remove singleton dimensions

                # Get dimensions
                dims = list(well_data.dims)
                dims = [
                    item.replace("Channel", "C").replace("Time", "T")
                    for item in dims
                ]
                axes = "".join(dims)

                # Try to get the first image file in the directory for metadata
                image_files = []
                for root, _, files in os.walk(filepath):
                    for file in files:
                        if file.lower().endswith((".tif", ".tiff")):
                            image_files.append(os.path.join(root, file))

                if image_files:
                    sample_file = image_files[0]
                    try:
                        # acquifer_metadata.getPixelSize_um(sample_file) is deprecated, get values after --PX in filename
                        pattern = re.compile(r"--PX(\d+)")
                        match = pattern.search(sample_file)
                        if match:
                            pixel_size = float(match.group(1)) * 10**-4

                        resolution = (pixel_size, pixel_size)
                    except (ValueError, FileNotFoundError) as e:
                        print(f"Warning: Could not get pixel size: {e}")
            else:
                # If no Well dimension, use dimensions from the dataset
                dims = list(dataset.dims)
                dims = [
                    item.replace("Channel", "C").replace("Time", "T")
                    for item in dims
                ]
                axes = "".join(dims)

            metadata = {
                "axes": axes,
                "resolution": resolution,
                "unit": "um",
                "filepath": filepath,
            }
            print(f"Extracted metadata: {metadata}")
            return metadata

        except (ValueError, FileNotFoundError) as e:
            print(f"Error getting metadata: {e}")
            return {}


class ScanFolderWorker(QThread):
    """Worker thread for scanning folders"""

    progress = Signal(int, int)  # current, total
    finished = Signal(list)  # list of found files
    error = Signal(str)  # error message

    def __init__(self, folder: str, filters: List[str]):
        super().__init__()
        self.folder = folder
        self.filters = filters

    def run(self):
        try:
            found_files = []
            all_items = []

            # Get both files and potential Acquifer directories
            include_directories = "acquifer" in [
                f.lower() for f in self.filters
            ]

            # Count items to scan
            for root, dirs, files in os.walk(self.folder):
                for file in files:
                    if any(
                        file.lower().endswith(f)
                        for f in self.filters
                        if f.lower() != "acquifer"
                    ):
                        all_items.append(os.path.join(root, file))

                # Add potential Acquifer directories
                if include_directories:
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if AcquiferLoader.can_load(dir_path):
                            all_items.append(dir_path)

            # Scan all items
            total_items = len(all_items)
            for i, item_path in enumerate(all_items):
                if i % 10 == 0:
                    self.progress.emit(i, total_items)

                found_files.append(item_path)

            self.finished.emit(found_files)
        except (ValueError, FileNotFoundError) as e:
            self.error.emit(str(e))


class ConversionWorker(QThread):
    """Worker thread for file conversion"""

    progress = Signal(int, int, str)  # current, total, filename
    file_done = Signal(str, bool, str)  # filepath, success, error message
    finished = Signal(int)  # number of successfully converted files

    def __init__(
        self,
        files_to_convert: List[Tuple[str, int]],
        output_folder: str,
        use_zarr: bool,
        file_loader_func,
    ):
        super().__init__()
        self.files_to_convert = files_to_convert
        self.output_folder = output_folder
        self.use_zarr = use_zarr
        self.get_file_loader = file_loader_func
        self.running = True

    def run(self):
        success_count = 0
        for i, (filepath, series_index) in enumerate(self.files_to_convert):
            if not self.running:
                break

            # Update progress
            self.progress.emit(
                i + 1, len(self.files_to_convert), Path(filepath).name
            )

            try:
                # Get loader
                loader = self.get_file_loader(filepath)
                if not loader:
                    self.file_done.emit(
                        filepath, False, "Unsupported file format"
                    )
                    continue

                # Load series - this is the critical part that must succeed
                try:
                    image_data = loader.load_series(filepath, series_index)
                except (ValueError, FileNotFoundError) as e:
                    self.file_done.emit(
                        filepath, False, f"Failed to load image: {str(e)}"
                    )
                    continue

                # Try to extract metadata - but don't fail if this doesn't work
                metadata = None
                try:
                    metadata = (
                        loader.get_metadata(filepath, series_index) or {}
                    )
                    print(f"Extracted metadata keys: {list(metadata.keys())}")
                except (ValueError, FileNotFoundError) as e:
                    print(f"Warning: Failed to extract metadata: {str(e)}")
                    metadata = {}

                # Generate output filename
                base_name = Path(filepath).stem

                # Determine format based on size and settings
                estimated_size_bytes = (
                    np.prod(image_data.shape) * image_data.itemsize
                )
                file_size_GB = estimated_size_bytes / (1024**3)

                # Determine format
                use_zarr = self.use_zarr
                # If file is very large (>4GB) and user didn't explicitly choose TIF,
                # auto-switch to ZARR format
                if file_size_GB > 4 and not self.use_zarr:
                    # Recommend ZARR format but respect user's choice by still allowing TIF
                    print(
                        f"File size ({file_size_GB:.2f}GB) exceeds 4GB, ZARR format is recommended but using TIF with BigTIFF format"
                    )
                    self.file_done.emit(
                        filepath,
                        True,
                        f"File size ({file_size_GB:.2f}GB) exceeds 4GB, using TIF with BigTIFF format",
                    )

                # Set up the output path
                if use_zarr:
                    output_path = os.path.join(
                        self.output_folder,
                        f"{base_name}_series{series_index}.zarr",
                    )
                else:
                    output_path = os.path.join(
                        self.output_folder,
                        f"{base_name}_series{series_index}.tif",
                    )

                # The crucial part - save the file with separate try/except for each save method
                save_success = False
                error_message = ""

                try:
                    if use_zarr:
                        save_success = self._save_zarr(
                            image_data, output_path, metadata
                        )
                    else:
                        self._save_tif(image_data, output_path, metadata)
                        save_success = os.path.exists(output_path)

                    if save_success:
                        success_count += 1
                        self.file_done.emit(
                            filepath, True, f"Saved to {output_path}"
                        )
                    else:
                        error_message = "Failed to save file - unknown error"
                        self.file_done.emit(filepath, False, error_message)

                except (ValueError, FileNotFoundError) as e:
                    error_message = f"Failed to save file: {str(e)}"
                    print(f"Error in save operation: {error_message}")
                    self.file_done.emit(filepath, False, error_message)

            except (ValueError, FileNotFoundError) as e:
                print(f"Unexpected error during conversion: {str(e)}")
                self.file_done.emit(
                    filepath, False, f"Unexpected error: {str(e)}"
                )

        self.finished.emit(success_count)

    def stop(self):
        self.running = False

    def _save_tif(
        self, image_data: np.ndarray, output_path: str, metadata: dict = None
    ):
        """Enhanced TIF saving with proper dimension handling and BigTIFF support"""
        import tifffile

        print(f"Saving TIF file: {output_path}")
        print(f"Image data shape: {image_data.shape}")

        # Check if this is a large file that needs BigTIFF
        estimated_size_bytes = np.prod(image_data.shape) * image_data.itemsize
        file_size_GB = estimated_size_bytes / (1024**3)
        use_bigtiff = file_size_GB > 4

        if use_bigtiff:
            print(
                f"File size ({file_size_GB:.2f}GB) exceeds 4GB, using BigTIFF format"
            )

        if metadata:
            print(f"Metadata keys: {list(metadata.keys())}")
            if "axes" in metadata:
                print(f"Original axes: {metadata['axes']}")

        # Handle Dask arrays
        if hasattr(image_data, "compute"):
            print("Computing Dask array before saving")
            # For large arrays, compute block by block
            try:
                # Convert to numpy array in memory
                image_data = image_data.compute()
            except (MemoryError, ValueError) as e:
                print(f"Error computing dask array: {e}")
                # Alternative: write block by block
                # This would require custom implementation
                raise

        # Basic save if no metadata
        if metadata is None:
            print("No metadata provided, using basic save")
            tifffile.imwrite(
                output_path,
                image_data,
                compression="zlib",
                bigtiff=use_bigtiff,
            )
            return

        # Get image dimensions and axis order
        ndim = len(image_data.shape)
        axes = metadata.get("axes", "")

        print(f"Number of dimensions: {ndim}")
        if axes:
            print(f"Axes from metadata: {axes}")

        # Handle ImageJ compatibility for dimensions
        if ndim > 2:
            # Get target order for ImageJ
            imagej_order = "TZCYX"

            # If axes information is incomplete, try to infer from shape
            if len(axes) != ndim:
                print(
                    f"Warning: Axes length ({len(axes)}) doesn't match dimensions ({ndim})"
                )
                # For your specific case with shape (45, 101, 4, 1024, 1024)
                # Infer TZCYX if shape matches
                if (
                    ndim == 5
                    and image_data.shape[2] <= 10
                    and image_data.shape[3] > 100
                    and image_data.shape[4] > 100
                ):
                    print("Inferring TZCYX from shape")
                    axes = "TZCYX"

            if axes and axes != imagej_order:
                print(f"Reordering: {axes} -> {imagej_order}")

                # Map dimensions from original to target order
                dim_map = {}
                for i, ax in enumerate(axes):
                    if ax in imagej_order:
                        dim_map[ax] = i

                # Handle missing dimensions
                for ax in imagej_order:
                    if ax not in dim_map:
                        print(f"Adding missing dimension: {ax}")
                        image_data = np.expand_dims(image_data, axis=0)
                        dim_map[ax] = image_data.shape[0] - 1

                # Create reordering indices
                source_idx = [dim_map[ax] for ax in imagej_order]
                target_idx = list(range(len(imagej_order)))

                print(f"Reordering dimensions: {source_idx} -> {target_idx}")

                # Reorder dimensions
                try:
                    image_data = np.moveaxis(
                        image_data, source_idx, target_idx
                    )
                except (ValueError, IndexError) as e:
                    print(f"Error reordering dimensions: {e}")
                    # Fall back to simple save without reordering
                    tifffile.imwrite(
                        output_path,
                        image_data,
                        compression="zlib",
                        bigtiff=use_bigtiff,
                    )
                    return

                # Update axes information
                metadata["axes"] = imagej_order

        # Extract resolution information for ImageJ
        resolution = None
        if "resolution" in metadata:
            try:
                res_x, res_y = metadata["resolution"]
                resolution = (float(res_x), float(res_y))
                print(f"Using resolution: {resolution}")
            except (ValueError, TypeError) as e:
                print(f"Error processing resolution: {e}")

        # Handle saving with metadata
        try:
            if ndim <= 2:
                # 2D case - simpler saving
                print("Saving as 2D image")
                tifffile.imwrite(
                    output_path,
                    image_data,
                    resolution=resolution,
                    compression="zlib",
                    bigtiff=use_bigtiff,
                )
            else:
                # Hyperstack case
                print("Saving as hyperstack with ImageJ metadata")

                # Create clean metadata dict with only needed keys
                imagej_metadata = {}
                if "unit" in metadata:
                    imagej_metadata["unit"] = metadata["unit"]
                if "spacing" in metadata:
                    imagej_metadata["spacing"] = float(metadata["spacing"])

                tifffile.imwrite(
                    output_path,
                    image_data,
                    imagej=True,
                    resolution=resolution,
                    metadata=imagej_metadata,
                    compression="zlib",
                    bigtiff=use_bigtiff,
                )

            print(f"Successfully saved TIF file: {output_path}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error saving TIF file: {e}")
            # Try simple save as fallback
            tifffile.imwrite(output_path, image_data, bigtiff=use_bigtiff)

    def _save_zarr(
        self, image_data: np.ndarray, output_path: str, metadata: dict = None
    ):
        """Enhanced ZARR saving with proper metadata storage and specific exceptions"""
        print(f"Saving ZARR file: {output_path}")
        print(f"Image data shape: {image_data.shape}")

        metadata = metadata or {}
        print(
            f"Metadata keys: {list(metadata.keys()) if metadata else 'No metadata'}"
        )

        # Handle overwriting by deleting the directory if it exists
        if os.path.exists(output_path):
            print(f"Deleting existing Zarr directory: {output_path}")
            shutil.rmtree(output_path)

        # Explicitly create a DirectoryStore
        store = parse_url(output_path, mode="w").store

        ndim = len(image_data.shape)

        axes = metadata.get("axes").lower() if metadata else None

        # Standardize axes order to 'ctzyx' if possible, regardless of Z presence
        target_axes = "tczyx"
        if axes != target_axes[:ndim]:
            print(f"Reordering axes from {axes} to {target_axes[:ndim]}")
            try:
                # Create a mapping from original axes to target axes
                axes_map = {ax: i for i, ax in enumerate(axes)}
                reorder_list = []
                for _i, target_ax in enumerate(target_axes[:ndim]):
                    if target_ax in axes_map:
                        reorder_list.append(axes_map[target_ax])
                    else:
                        print(f"Axis {target_ax} not found in original axes")
                        reorder_list.append(None)

                # Filter out None values (missing axes)
                reorder_list = [i for i in reorder_list if i is not None]

                if len(reorder_list) != len(axes):
                    raise ValueError(
                        "Reordering failed: Mismatch between original and reordered dimensions."
                    )
                image_data = np.moveaxis(
                    image_data, range(len(axes)), reorder_list
                )
                axes = "".join(
                    [axes[i] for i in reorder_list]
                )  # Update axes to reflect new order
                print(f"New axes order after reordering: {axes}")
            except (ValueError, IndexError) as e:
                print(f"Error during reordering: {e}")
                raise

        # Convert to Dask array
        if not hasattr(image_data, "dask"):
            print("Converting to dask array with auto chunks...")
            image_data = da.from_array(image_data, chunks="auto")
        else:
            print("Using existing dask array")

        # Write the image data as OME-Zarr
        try:
            print("Writing image data using ome_zarr.writer.write_image...")
            with ProgressBar():
                root = zarr.group(store=store)
                write_image(
                    image_data,
                    group=root,
                    axes=axes,
                    scaler=None,
                    storage_options={"compression": "zstd"},
                )

            # Add basic OME-Zarr metadata
            root = zarr.open(store)
            root.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "datasets": [{"path": "0"}],
                    "axes": [
                        {
                            "name": ax,
                            "type": (
                                "space"
                                if ax in "xyz"
                                else "time" if ax == "t" else "channel"
                            ),
                        }
                        for ax in axes
                    ],
                }
            ]

            print("OME-Zarr file saved successfully.")
            return True

        except (ValueError, FileNotFoundError) as e:
            print(f"Error during Zarr writing: {e}")
            import traceback

            traceback.print_exc()
            return False


class MicroscopyImageConverterWidget(QWidget):
    """Main widget for microscopy image conversion to TIF/ZARR"""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # Register format loaders
        self.loaders = [
            LIFLoader,
            ND2Loader,
            TIFFSlideLoader,
            CZILoader,
            AcquiferLoader,
        ]

        # Selected series for conversion
        self.selected_series = {}  # {filepath: series_index}

        # Track files that should export all series
        self.export_all_series = {}  # {filepath: boolean}

        # Working threads
        self.scan_worker = None
        self.conversion_worker = None

        # Flag to prevent recursive radio button updates
        self.updating_format_buttons = False

        # Create layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # File selection widgets
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Input Folder:")
        self.folder_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_folder)

        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_edit)
        folder_layout.addWidget(browse_button)
        main_layout.addLayout(folder_layout)

        # File filter widgets
        filter_layout = QHBoxLayout()
        filter_label = QLabel("File Filter:")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            ".lif, .nd2, .ndpi, .czi, acquifer (comma separated)"
        )
        self.filter_edit.setText(".lif,.nd2,.ndpi,.czi, acquifer")
        scan_button = QPushButton("Scan Folder")
        scan_button.clicked.connect(self.scan_folder)

        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_edit)
        filter_layout.addWidget(scan_button)
        main_layout.addLayout(filter_layout)

        # Progress bar for scanning
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        main_layout.addWidget(self.scan_progress)

        # Files and series tables
        tables_layout = QHBoxLayout()

        # Files table
        self.files_table = SeriesTableWidget(viewer)
        tables_layout.addWidget(self.files_table)

        # Series details widget
        self.series_widget = SeriesDetailWidget(self, viewer)
        tables_layout.addWidget(self.series_widget)

        main_layout.addLayout(tables_layout)

        # Conversion options
        options_layout = QVBoxLayout()

        # Output format selection
        format_layout = QHBoxLayout()
        format_label = QLabel("Output Format:")
        self.tif_radio = QCheckBox("TIF (< 4GB)")
        self.tif_radio.setChecked(True)
        self.zarr_radio = QCheckBox("ZARR (> 4GB)")

        # Make checkboxes mutually exclusive like radio buttons
        self.tif_radio.toggled.connect(self.handle_format_toggle)
        self.zarr_radio.toggled.connect(self.handle_format_toggle)

        format_layout.addWidget(format_label)
        format_layout.addWidget(self.tif_radio)
        format_layout.addWidget(self.zarr_radio)
        options_layout.addLayout(format_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        self.output_edit = QLineEdit()
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self.browse_output)

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        options_layout.addLayout(output_layout)

        main_layout.addLayout(options_layout)

        # Conversion progress bar
        self.conversion_progress = QProgressBar()
        self.conversion_progress.setVisible(False)
        main_layout.addWidget(self.conversion_progress)

        # Conversion and cancel buttons
        button_layout = QHBoxLayout()
        convert_button = QPushButton("Convert Selected Files")
        convert_button.clicked.connect(self.convert_files)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.setVisible(False)

        button_layout.addWidget(convert_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

    def cancel_operation(self):
        """Cancel current operation"""
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.terminate()
            self.scan_worker = None
            self.status_label.setText("Scanning cancelled")

        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.stop()
            self.status_label.setText("Conversion cancelled")

        self.scan_progress.setVisible(False)
        self.conversion_progress.setVisible(False)
        self.cancel_button.setVisible(False)

    def browse_folder(self):
        """Open a folder browser dialog"""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.folder_edit.setText(folder)

    def browse_output(self):
        """Open a folder browser dialog for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def scan_folder(self):
        """Scan the selected folder for image files"""
        folder = self.folder_edit.text()
        if not folder or not os.path.isdir(folder):
            self.status_label.setText("Please select a valid folder")
            return

        # Get file filters
        filters = [
            f.strip() for f in self.filter_edit.text().split(",") if f.strip()
        ]
        if not filters:
            filters = [".lif", ".nd2", ".ndpi", ".czi"]

        # Clear existing files
        self.files_table.setRowCount(0)
        self.files_table.file_data.clear()

        # Set up and start the worker thread
        self.scan_worker = ScanFolderWorker(folder, filters)
        self.scan_worker.progress.connect(self.update_scan_progress)
        self.scan_worker.finished.connect(self.process_found_files)
        self.scan_worker.error.connect(self.show_error)

        # Show progress bar and start worker
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        self.cancel_button.setVisible(True)
        self.status_label.setText("Scanning folder...")
        self.scan_worker.start()

    def update_scan_progress(self, current, total):
        """Update the scan progress bar"""
        if total > 0:
            self.scan_progress.setValue(int(current * 100 / total))

    def process_found_files(self, found_files):
        """Process the list of found files after scanning is complete"""
        # Hide progress bar
        self.scan_progress.setVisible(False)
        self.cancel_button.setVisible(False)

        # Process files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process files in parallel to get series counts
            futures = {}
            for filepath in found_files:
                file_type = self.get_file_type(filepath)
                if file_type:
                    loader = self.get_file_loader(filepath)
                    if loader:
                        future = executor.submit(
                            loader.get_series_count, filepath
                        )
                        futures[future] = (filepath, file_type)

            # Process results as they complete
            file_count = len(found_files)
            processed = 0

            for i, future in enumerate(
                concurrent.futures.as_completed(futures)
            ):
                processed = i + 1
                filepath, file_type = futures[future]

                try:
                    series_count = future.result()
                    # Add file to table
                    self.files_table.add_file(
                        filepath, file_type, series_count
                    )
                except (ValueError, FileNotFoundError) as e:
                    print(f"Error processing {filepath}: {str(e)}")
                    # Add file with error indication
                    self.files_table.add_file(filepath, file_type, -1)

                # Update status periodically
                if processed % 5 == 0 or processed == file_count:
                    self.status_label.setText(
                        f"Processed {processed}/{file_count} files..."
                    )
                    QApplication.processEvents()

        self.status_label.setText(f"Found {len(found_files)} files")

    def show_error(self, error_message):
        """Show error message"""
        self.status_label.setText(f"Error: {error_message}")
        self.scan_progress.setVisible(False)
        self.cancel_button.setVisible(False)
        QMessageBox.critical(self, "Error", error_message)

    def get_file_type(self, filepath: str) -> str:
        """Determine the file type based on extension or directory type"""
        if os.path.isdir(filepath) and AcquiferLoader.can_load(filepath):
            return "Acquifer"
        ext = filepath.lower()
        if ext.endswith(".lif"):
            return "LIF"
        elif ext.endswith(".nd2"):
            return "ND2"
        elif ext.endswith((".ndpi", ".svs")):
            return "Slide"
        elif ext.endswith(".czi"):
            return "CZI"
        return "Unknown"

    def get_file_loader(self, filepath: str) -> Optional[FormatLoader]:
        """Get the appropriate loader for the file type"""
        for loader in self.loaders:
            if loader.can_load(filepath):
                return loader
        return None

    def show_series_details(self, filepath: str):
        """Show details for the series in the selected file"""
        self.series_widget.set_file(filepath)

    def set_selected_series(self, filepath: str, series_index: int):
        """Set the selected series for a file"""
        self.selected_series[filepath] = series_index

    def set_export_all_series(self, filepath: str, export_all: bool):
        """Set whether to export all series for a file"""
        self.export_all_series[filepath] = export_all

        # If exporting all, we still need a default series in selected_series
        # for files that are marked for export all
        if export_all and filepath not in self.selected_series:
            self.selected_series[filepath] = 0

    def load_image(self, filepath: str):
        """Load an image file into the viewer"""
        loader = self.get_file_loader(filepath)
        if not loader:
            self.viewer.status = f"Unsupported file format: {filepath}"
            return

        try:
            # For non-series files, just load the first series
            series_index = 0
            image_data = loader.load_series(filepath, series_index)

            # Clear existing layers and display the image
            self.viewer.layers.clear()
            self.viewer.add_image(image_data, name=f"{Path(filepath).stem}")

            # Update status
            self.viewer.status = f"Loaded {Path(filepath).name}"
        except (ValueError, FileNotFoundError) as e:
            self.viewer.status = f"Error loading image: {str(e)}"
            QMessageBox.warning(
                self, "Error", f"Could not load image: {str(e)}"
            )

    def is_output_folder_valid(self, folder):
        """Check if the output folder is valid and writable"""
        if not folder:
            self.status_label.setText("Please specify an output folder")
            return False

        # Check if folder exists, if not try to create it
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except (FileNotFoundError, PermissionError) as e:
                self.status_label.setText(
                    f"Cannot create output folder: {str(e)}"
                )
                return False

        # Check if folder is writable
        if not os.access(folder, os.W_OK):
            self.status_label.setText("Output folder is not writable")
            return False

        return True

    def convert_files(self):
        """Convert selected files to TIF or ZARR"""
        # Check if any files are selected
        if not self.selected_series:
            self.status_label.setText("No files selected for conversion")
            return

        # Check output folder
        output_folder = self.output_edit.text()
        if not output_folder:
            output_folder = os.path.join(self.folder_edit.text(), "converted")

        # Validate output folder
        if not self.is_output_folder_valid(output_folder):
            return

        # Create files to convert list
        files_to_convert = []

        for filepath, series_index in self.selected_series.items():
            # Check if we should export all series for this file
            if self.export_all_series.get(filepath, False):
                # Get the number of series for this file
                loader = self.get_file_loader(filepath)
                if loader:
                    try:
                        series_count = loader.get_series_count(filepath)
                        # Add all series for this file
                        for i in range(series_count):
                            files_to_convert.append((filepath, i))
                    except (ValueError, FileNotFoundError) as e:
                        self.status_label.setText(
                            f"Error getting series count: {str(e)}"
                        )
                        QMessageBox.warning(
                            self,
                            "Error",
                            f"Could not get series count for {Path(filepath).name}: {str(e)}",
                        )
            else:
                # Just add the selected series
                files_to_convert.append((filepath, series_index))

        if not files_to_convert:
            self.status_label.setText("No valid files to convert")
            return

        # Set up and start the conversion worker thread
        self.conversion_worker = ConversionWorker(
            files_to_convert=files_to_convert,
            output_folder=output_folder,
            use_zarr=self.zarr_radio.isChecked(),
            file_loader_func=self.get_file_loader,
        )

        # Connect signals
        self.conversion_worker.progress.connect(
            self.update_conversion_progress
        )
        self.conversion_worker.file_done.connect(
            self.handle_file_conversion_result
        )
        self.conversion_worker.finished.connect(self.conversion_completed)

        # Show progress bar and start worker
        self.conversion_progress.setVisible(True)
        self.conversion_progress.setValue(0)
        self.cancel_button.setVisible(True)
        self.status_label.setText(
            f"Starting conversion of {len(files_to_convert)} files/series..."
        )

        # Start conversion
        self.conversion_worker.start()

    def update_conversion_progress(self, current, total, filename):
        """Update conversion progress bar and status"""
        if total > 0:
            self.conversion_progress.setValue(int(current * 100 / total))
            self.status_label.setText(
                f"Converting {filename} ({current}/{total})..."
            )

    def handle_file_conversion_result(self, filepath, success, message):
        """Handle result of a single file conversion"""
        filename = Path(filepath).name
        if success:
            print(f"Successfully converted: {filename} - {message}")
        else:
            print(f"Failed to convert: {filename} - {message}")
            QMessageBox.warning(
                self,
                "Conversion Warning",
                f"Error converting {filename}: {message}",
            )

    def conversion_completed(self, success_count):
        """Handle completion of all conversions"""
        self.conversion_progress.setVisible(False)
        self.cancel_button.setVisible(False)

        output_folder = self.output_edit.text()
        if not output_folder:
            output_folder = os.path.join(self.folder_edit.text(), "converted")
        if success_count > 0:
            self.status_label.setText(
                f"Successfully converted {success_count} files to {output_folder}"
            )
        else:
            self.status_label.setText("No files were converted")

    def update_format_buttons(self, use_zarr=False):
        """Update format radio buttons based on file size

        Args:
            use_zarr: True if file size > 4GB, otherwise False
        """
        if self.updating_format_buttons:
            return

        self.updating_format_buttons = True
        try:
            if use_zarr:
                self.zarr_radio.setChecked(True)
                self.tif_radio.setChecked(False)
                print("Auto-selected ZARR format for large file (>4GB)")
            else:
                self.tif_radio.setChecked(True)
                self.zarr_radio.setChecked(False)
                print("Auto-selected TIF format for smaller file (<4GB)")
        finally:
            self.updating_format_buttons = False

    def handle_format_toggle(self, checked):
        """Handle format radio button toggle"""
        if self.updating_format_buttons:
            return

        self.updating_format_buttons = True
        try:
            # Make checkboxes mutually exclusive like radio buttons
            sender = self.sender()
            if sender == self.tif_radio and checked:
                self.zarr_radio.setChecked(False)
            elif sender == self.zarr_radio and checked:
                self.tif_radio.setChecked(False)
        finally:
            self.updating_format_buttons = False


# Create a MagicGUI widget that creates and returns the converter widget
@magicgui(
    call_button="Start Microscopy Image Converter",
    layout="vertical",
)
def microscopy_converter(viewer: napari.Viewer):
    """
    Start the microscopy image converter tool
    """
    # Create the converter widget
    converter_widget = MicroscopyImageConverterWidget(viewer)

    # Add to viewer
    viewer.window.add_dock_widget(
        converter_widget, name="Microscopy Image Converter", area="right"
    )

    return converter_widget


# This is what napari calls to get the widget
def napari_experimental_provide_dock_widget():
    """Provide the converter widget to Napari"""
    return microscopy_converter

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
from typing import Any, Dict, List, Optional, Tuple, Union

import napari
import numpy as np
import tifffile
import zarr
from magicgui import magicgui
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
from napari_tmidas._ui_utils import add_browse_button_to_folder_field

sys.path.append("src/napari_tmidas")
from napari_tmidas.processing_functions import (
    discover_and_load_processing_functions,
)

# Check for OME-Zarr support
try:
    from napari_ome_zarr import napari_get_reader

    OME_ZARR_AVAILABLE = True
    print("napari-ome-zarr found - enhanced Zarr support enabled")
except ImportError:
    OME_ZARR_AVAILABLE = False
    print(
        "Tip: Install napari-ome-zarr for better Zarr support: pip install napari-ome-zarr"
    )

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print(
        "Tip: Install dask for better performance with large datasets: pip install dask"
    )


def load_zarr_with_napari_ome_zarr(
    filepath: str, verbose: bool = True
) -> Optional[List[Tuple]]:
    """
    Load zarr using napari-ome-zarr reader with enhanced error handling
    """
    if not OME_ZARR_AVAILABLE:
        return None

    try:
        # Try multiple approaches to get the reader
        reader_func = napari_get_reader(filepath)
        if reader_func is None:
            if verbose:
                print(f"napari-ome-zarr: No reader available for {filepath}")
            return None

        # Try to read the data
        layer_data_list = reader_func(filepath)

        if layer_data_list and len(layer_data_list) > 0:
            if verbose:
                print(
                    f"napari-ome-zarr: Successfully loaded {len(layer_data_list)} layers"
                )

            # Enhance layer metadata
            enhanced_layers = []
            for i, (data, add_kwargs, layer_type) in enumerate(
                layer_data_list
            ):
                # Ensure proper naming
                if "name" not in add_kwargs or not add_kwargs["name"]:
                    basename = os.path.basename(filepath)
                    if layer_type == "image":
                        add_kwargs["name"] = f"C{i+1}: {basename}"
                    elif layer_type == "labels":
                        add_kwargs["name"] = f"Labels{i+1}: {basename}"
                    else:
                        add_kwargs["name"] = (
                            f"{layer_type.title()}{i+1}: {basename}"
                        )

                # Set appropriate blending for multi-channel images
                if layer_type == "image" and len(layer_data_list) > 1:
                    add_kwargs["blending"] = "additive"

                # Ensure proper colormap assignment for multi-channel
                if layer_type == "image" and "colormap" not in add_kwargs:
                    channel_colormaps = [
                        "red",
                        "green",
                        "blue",
                        "cyan",
                        "magenta",
                        "yellow",
                    ]
                    add_kwargs["colormap"] = channel_colormaps[
                        i % len(channel_colormaps)
                    ]

                enhanced_layers.append((data, add_kwargs, layer_type))

            return enhanced_layers
        else:
            if verbose:
                print(
                    f"napari-ome-zarr: Reader returned empty layer list for {filepath}"
                )
            return None

    except (ImportError, ValueError, TypeError, OSError) as e:
        if verbose:
            print(f"napari-ome-zarr: Failed to load {filepath}: {e}")
            import traceback

            traceback.print_exc()
        return None


def load_zarr_basic(filepath: str) -> Union[np.ndarray, Any]:
    """
    Basic zarr loading with dask support as fallback
    """
    try:
        root = zarr.open(filepath, mode="r")

        # Handle zarr groups vs single arrays
        if hasattr(root, "arrays"):
            arrays_list = list(root.arrays())
            if not arrays_list:
                raise ValueError(f"No arrays found in zarr group: {filepath}")

            # Try to find the main data array
            # Look for arrays named '0', 'data', or take the first one
            main_array = None
            for name, array in arrays_list:
                if name in ["0", "data"]:
                    main_array = array
                    break

            if main_array is None:
                main_array = arrays_list[0][1]

            zarr_array = main_array
        else:
            zarr_array = root

        # Convert to dask array for lazy loading if available
        if DASK_AVAILABLE:
            print(f"Loading zarr as dask array with shape: {zarr_array.shape}")
            return da.from_zarr(zarr_array)
        else:
            print(
                f"Loading zarr as numpy array with shape: {zarr_array.shape}"
            )
            return np.array(zarr_array)

    except (ValueError, TypeError, OSError) as e:
        print(f"Error in basic zarr loading for {filepath}: {e}")
        raise


def is_ome_zarr(filepath: str) -> bool:
    """
    Check if a zarr file is OME-Zarr format by looking for OME metadata
    """
    try:
        if not os.path.exists(filepath):
            return False

        root = zarr.open(filepath, mode="r")

        if hasattr(root, "attrs") and (
            "ome" in root.attrs
            or "omero" in root.attrs
            or "multiscales" in root.attrs
        ):
            return True

        # Check for .zattrs file with OME metadata
        zattrs_path = os.path.join(filepath, ".zattrs")
        if os.path.exists(zattrs_path):
            import json

            try:
                with open(zattrs_path) as f:
                    attrs = json.load(f)
                if (
                    "ome" in attrs
                    or "omero" in attrs
                    or "multiscales" in attrs
                ):
                    return True
            except (OSError, json.JSONDecodeError):
                pass

        return False

    except (ValueError, TypeError, OSError):
        return False


def get_zarr_info(filepath: str) -> dict:
    """Get detailed information about a zarr dataset"""
    info = {
        "is_ome_zarr": False,
        "is_multiscale": False,
        "num_arrays": 0,
        "arrays": [],
        "shape": None,
        "dtype": None,
        "chunks": None,
        "has_labels": False,
        "resolution_levels": 0,
    }

    try:
        root = zarr.open(filepath, mode="r")
        info["is_ome_zarr"] = is_ome_zarr(filepath)

        if hasattr(root, "arrays"):
            arrays_list = list(root.arrays())
            info["num_arrays"] = len(arrays_list)
            info["arrays"] = [name for name, _ in arrays_list]

            if (
                info["is_ome_zarr"]
                and hasattr(root, "attrs")
                and "multiscales" in root.attrs
            ):
                info["is_multiscale"] = True
                multiscales = root.attrs["multiscales"]
                if multiscales and len(multiscales) > 0:
                    datasets = multiscales[0].get("datasets", [])
                    info["resolution_levels"] = len(datasets)

            if arrays_list:
                first_array = arrays_list[0][1]
                info["shape"] = first_array.shape
                info["dtype"] = str(first_array.dtype)
                info["chunks"] = first_array.chunks

            info["has_labels"] = "labels" in info["arrays"]

        else:
            info["num_arrays"] = 1
            info["shape"] = root.shape
            info["dtype"] = str(root.dtype)
            info["chunks"] = root.chunks

    except (ValueError, TypeError, OSError) as e:
        print(f"Error getting zarr info for {filepath}: {e}")

    return info


def load_image_file(filepath: str) -> Union[np.ndarray, List, Any]:
    """
    Load image from file, supporting both TIFF and Zarr formats with proper metadata handling
    """
    if filepath.lower().endswith(".zarr"):

        # Try to use napari-ome-zarr reader first for proper metadata handling
        if OME_ZARR_AVAILABLE:
            try:
                layer_data_list = load_zarr_with_napari_ome_zarr(filepath)
                if layer_data_list:
                    print(
                        f"Loaded {len(layer_data_list)} layers from OME-Zarr"
                    )
                    return layer_data_list
            except (ImportError, ValueError, TypeError, OSError) as e:
                print(
                    f"napari-ome-zarr reader failed: {e}, falling back to basic zarr loading"
                )

        # Fallback to basic zarr loading with dask
        return load_zarr_basic(filepath)
    else:
        return imread(filepath)


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

        # Currently loaded images (can be multiple for multi-channel)
        self.current_original_images = []
        self.current_processed_images = []

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

    def _clear_current_images(self, image_list):
        """Helper to clear a list of current images"""
        for img_layer in image_list:
            try:
                if img_layer in self.viewer.layers:
                    self.viewer.layers.remove(img_layer)
                else:
                    # Try by name if reference doesn't work
                    layer_names = [layer.name for layer in self.viewer.layers]
                    if img_layer.name in layer_names:
                        self.viewer.layers.remove(img_layer.name)
            except (KeyError, ValueError, AttributeError) as e:
                print(f"Warning: Could not remove layer: {e}")
        image_list.clear()

    def _should_enable_3d_view(self, data):
        """Check if 3D view should be enabled based on data dimensions"""
        if not hasattr(data, "shape") or len(data.shape) < 3:
            return False

        # Check if we have meaningful 3D data (excluding potential channel dimensions)
        shape = data.shape

        # If first dimension is small (<=10), it's likely channels, check remaining dims
        meaningful_dims = shape[1:] if shape[0] <= 10 else shape

        # Enable 3D if we have at least 3 spatial dimensions with substantial size
        if len(meaningful_dims) >= 3:
            # Check that we have meaningful Z depth (not just singleton)
            z_dim = meaningful_dims[0] if len(meaningful_dims) >= 3 else 1
            return z_dim > 1

        return False

    def _load_original_image(self, filepath: str):
        """
        Load original image into viewer with proper multi-channel support using napari-ome-zarr
        """
        # Ensure filepath is valid
        if not filepath or not os.path.exists(filepath):
            print(f"Error: File does not exist: {filepath}")
            self.viewer.status = f"Error: File not found: {filepath}"
            return

        # Remove existing original layers
        self._clear_current_images(self.current_original_images)

        # Load new image
        try:
            # Display status while loading
            self.viewer.status = f"Loading {os.path.basename(filepath)}..."

            # For zarr files, use viewer.open() with the napari-ome-zarr plugin directly
            if filepath.lower().endswith(".zarr") and OME_ZARR_AVAILABLE:
                print("Using viewer.open() with napari-ome-zarr plugin")

                # Use napari's built-in open method with the plugin
                # This is exactly what napari does when you open a zarr file
                try:
                    layers = self.viewer.open(
                        filepath, plugin="napari-ome-zarr"
                    )

                    # Track the added layers
                    if layers:
                        if isinstance(layers, list):
                            self.current_original_images.extend(layers)
                        else:
                            self.current_original_images.append(layers)

                        # Check if we should enable 3D view
                        if len(self.current_original_images) > 0:
                            first_layer = self.current_original_images[0]
                            if hasattr(
                                first_layer, "data"
                            ) and self._should_enable_3d_view(
                                first_layer.data
                            ):
                                self.viewer.dims.ndisplay = 3
                                print(
                                    f"Switched to 3D view for data with shape: {first_layer.data.shape}"
                                )

                        self.viewer.status = f"Loaded {len(self.current_original_images)} layers from {os.path.basename(filepath)}"
                        return
                    else:
                        print(
                            "napari-ome-zarr returned no layers, falling back to manual loading"
                        )
                except (ImportError, ValueError, TypeError, OSError) as e:
                    print(
                        f"napari-ome-zarr failed: {e}, falling back to manual loading"
                    )

            # Fallback for non-zarr files or if napari-ome-zarr fails
            # Load image using the unified loader function
            image_data = load_image_file(filepath)

            # Handle multi-layer data from OME-Zarr or enhanced basic loading
            if isinstance(image_data, list):
                # Channel-specific colormaps: R, G, B, then additional colors
                channel_colormaps = [
                    "red",
                    "green",
                    "blue",
                    "cyan",
                    "magenta",
                    "yellow",
                    "orange",
                    "purple",
                    "pink",
                    "gray",
                ]

                # This is from napari-ome-zarr reader or enhanced basic loading - add each layer separately
                for layer_idx, layer_info in enumerate(image_data):
                    # Handle different formats of layer_info
                    if isinstance(layer_info, tuple) and len(layer_info) == 3:
                        # Format: (data, add_kwargs, layer_type)
                        data, add_kwargs, layer_type = layer_info
                    elif (
                        isinstance(layer_info, tuple) and len(layer_info) == 2
                    ):
                        # Format: (data, add_kwargs) - assume image type
                        data, add_kwargs = layer_info
                        layer_type = "image"
                    else:
                        # Just data - create minimal kwargs
                        data = layer_info
                        add_kwargs = {}
                        layer_type = "image"

                    base_filename = os.path.basename(filepath)

                    if layer_type == "image":
                        # Check if this is a multi-channel image that needs to be split using channel_axis
                        if hasattr(data, "shape") and len(data.shape) >= 3:
                            # Look for a channel dimension (small dimension, typically <= 10)
                            potential_channel_dims = []
                            for dim_idx, dim_size in enumerate(data.shape):
                                if dim_size <= 10 and dim_size > 1:
                                    potential_channel_dims.append(
                                        (dim_idx, dim_size)
                                    )

                            # If we found a potential channel dimension, use napari's channel_axis
                            if potential_channel_dims:
                                # Use the first potential channel dimension
                                channel_axis, num_channels = (
                                    potential_channel_dims[0]
                                )
                                print(
                                    f"Using napari channel_axis={channel_axis} for {num_channels} channels"
                                )

                                # Let napari handle channel splitting automatically with proper colormaps
                                layers = self.viewer.add_image(
                                    data,
                                    channel_axis=channel_axis,
                                    name=f"Original: {base_filename}",
                                    blending="additive",
                                )

                                # Track all the layers napari created
                                if isinstance(layers, list):
                                    self.current_original_images.extend(layers)
                                else:
                                    self.current_original_images.append(layers)

                                continue  # Skip the normal single-layer processing

                        # Normal single-layer processing (no channel splitting needed)
                        # Override/set colormap for proper channel assignment
                        if "colormap" not in add_kwargs:
                            add_kwargs["colormap"] = (
                                channel_colormaps[layer_idx]
                                if layer_idx < len(channel_colormaps)
                                else "gray"
                            )

                        if "blending" not in add_kwargs:
                            add_kwargs["blending"] = (
                                "additive"  # Enable proper multi-channel blending
                            )

                        # Ensure proper naming
                        if "name" not in add_kwargs or not add_kwargs["name"]:
                            add_kwargs["name"] = (
                                f"C{layer_idx+1}: {base_filename}"
                            )

                        layer = self.viewer.add_image(data, **add_kwargs)
                        self.current_original_images.append(layer)

                    elif layer_type == "labels":
                        if "name" not in add_kwargs or not add_kwargs["name"]:
                            add_kwargs["name"] = (
                                f"Labels{layer_idx+1}: {base_filename}"
                            )

                        layer = self.viewer.add_labels(data, **add_kwargs)
                        self.current_original_images.append(layer)

                # Switch to 3D view if data has meaningful 3D dimensions
                if len(self.current_original_images) > 0:
                    # Get the first layer's data safely
                    first_layer = self.current_original_images[0]
                    if hasattr(first_layer, "data"):
                        first_layer_data = first_layer.data
                        if self._should_enable_3d_view(first_layer_data):
                            self.viewer.dims.ndisplay = 3
                            print(
                                f"Switched to 3D view for data with shape: {first_layer_data.shape}"
                            )

                self.viewer.status = f"Loaded {len(self.current_original_images)} channels from {os.path.basename(filepath)}"
                return

            # Handle single image data (TIFF or simple zarr)
            image = image_data

            # Remove singletons if it's a numpy array
            if hasattr(image, "squeeze") and not hasattr(image, "chunks"):
                image = np.squeeze(image)

            # Check for multi-channel data in single array
            if (
                hasattr(image, "shape")
                and len(image.shape) > 2
                and image.shape[0] <= 10
                and image.shape[0] > 1
            ):  # Likely channels first
                print(
                    f"Using napari channel_axis=0 for {image.shape[0]} channels"
                )

                # Use napari's channel_axis to automatically split channels with proper colormaps
                layers = self.viewer.add_image(
                    image,
                    channel_axis=0,
                    name=f"Original: {os.path.basename(filepath)}",
                    blending="additive",
                )

                # Track all the layers napari created
                if isinstance(layers, list):
                    self.current_original_images.extend(layers)
                else:
                    self.current_original_images.append(layers)

                # Switch to 3D view if data has meaningful 3D dimensions (excluding channel dim)
                if len(self.current_original_images) > 0:
                    first_layer = self.current_original_images[0]
                    if hasattr(first_layer, "data"):
                        channel_data = first_layer.data
                        if self._should_enable_3d_view(channel_data):
                            self.viewer.dims.ndisplay = 3
                            print(
                                f"Switched to 3D view for multi-channel data with shape: {channel_data.shape}"
                            )

                self.viewer.status = f"Loaded {len(self.current_original_images)} channels from {os.path.basename(filepath)}"
                return

            # Single channel image
            base_filename = os.path.basename(filepath)
            # check if label image by checking file name
            is_label = "labels" in base_filename or "semantic" in base_filename

            if is_label:
                if hasattr(image, "astype"):
                    image = image.astype(np.uint32)
                layer = self.viewer.add_labels(
                    image, name=f"Labels: {base_filename}"
                )
            else:
                layer = self.viewer.add_image(
                    image, name=f"Original: {base_filename}"
                )

            self.current_original_images.append(layer)

            # Switch to 3D view if data has meaningful 3D dimensions
            if hasattr(layer, "data") and self._should_enable_3d_view(
                layer.data
            ):
                self.viewer.dims.ndisplay = 3
                print(
                    f"Switched to 3D view for single-channel data with shape: {layer.data.shape}"
                )

            self.viewer.status = f"Loaded {base_filename}"

        except (ValueError, TypeError, OSError, ImportError) as e:
            print(f"Error loading original image {filepath}: {e}")
            import traceback

            traceback.print_exc()
            self.viewer.status = f"Error processing {filepath}: {e}"

    def _load_processed_image(self, filepath: str):
        """
        Load processed image into viewer with multi-channel support and ensure it's always shown on top
        Also handles points data from spot detection functions.
        """
        # Ensure filepath is valid
        if not filepath or not os.path.exists(filepath):
            print(f"Error: File does not exist: {filepath}")
            self.viewer.status = f"Error: File not found: {filepath}"
            return

        # Remove existing processed layers
        self._clear_current_images(self.current_processed_images)

        # Special handling for .npy files (likely points data from spot detection)
        if filepath.lower().endswith(".npy"):
            try:
                data = np.load(filepath)

                # Check if this is points data
                if (
                    isinstance(data, np.ndarray)
                    and data.ndim == 2
                    and data.shape[1] in [2, 3]  # 2D or 3D coordinates
                    and data.dtype in [np.float32, np.float64]
                ):  # Coordinate data

                    print(f"Loading points data: {data.shape} points")

                    # Determine if 2D or 3D points
                    is_3d = data.shape[1] == 3

                    # Set appropriate point properties
                    point_properties = {
                        "size": 8,
                        "symbol": "ring",
                        "opacity": 1,
                        "face_color": [1.0, 0.5, 0.2],
                        "border_color": [1.0, 0.5, 0.2],
                    }

                    if is_3d:
                        point_properties["out_of_slice_display"] = True

                    # Add points layer
                    points_layer = self.viewer.add_points(
                        data,
                        name=f"Spots ({os.path.basename(filepath)})",
                        **point_properties,
                    )

                    # Track the layer
                    self.current_processed_images = [points_layer]

                    self.viewer.status = f"Loaded {len(data)} spots from {os.path.basename(filepath)}"
                    print(
                        f"Successfully loaded {len(data)} spots as points layer"
                    )
                    return

                else:
                    print(
                        "NPY file doesn't contain points data, treating as image"
                    )
                    # Fall through to regular image loading

            except (OSError, ValueError, AttributeError) as e:
                print(f"Error loading NPY file as points: {e}")
                # Fall through to regular image loading

        # Load new image (original logic)
        try:
            # Display status while loading
            self.viewer.status = f"Loading {os.path.basename(filepath)}..."

            # For zarr files, use viewer.open() with the napari-ome-zarr plugin directly
            if filepath.lower().endswith(".zarr") and OME_ZARR_AVAILABLE:
                print(
                    "Using viewer.open() with napari-ome-zarr plugin for processed image"
                )

                # Use napari's built-in open method with the plugin
                try:
                    layers = self.viewer.open(
                        filepath, plugin="napari-ome-zarr"
                    )

                    # Track the added layers and rename them as processed
                    if layers:
                        if isinstance(layers, list):
                            for layer in layers:
                                layer.name = f"Processed {layer.name}"
                                self.current_processed_images.append(layer)
                        else:
                            layers.name = f"Processed {layers.name}"
                            self.current_processed_images.append(layers)

                        # Switch to 3D view if data has meaningful 3D dimensions
                        if len(self.current_processed_images) > 0:
                            first_layer = self.current_processed_images[0]
                            if hasattr(first_layer, "data"):
                                first_layer_data = first_layer.data
                                if self._should_enable_3d_view(
                                    first_layer_data
                                ):
                                    self.viewer.dims.ndisplay = 3
                                    print(
                                        f"Switched to 3D view for processed data with shape: {first_layer_data.shape}"
                                    )

                        # Move all processed layers to top
                        for layer in self.current_processed_images:
                            if layer in self.viewer.layers:
                                layer_index = self.viewer.layers.index(layer)
                                if layer_index < len(self.viewer.layers) - 1:
                                    self.viewer.layers.move(
                                        layer_index,
                                        len(self.viewer.layers) - 1,
                                    )

                        self.viewer.status = f"Loaded {len(self.current_processed_images)} processed layers from {os.path.basename(filepath)}"
                        return
                    else:
                        print(
                            "napari-ome-zarr returned no layers for processed image, falling back"
                        )
                except (ImportError, ValueError, TypeError, OSError) as e:
                    print(
                        f"napari-ome-zarr failed for processed image: {e}, falling back"
                    )

            # Fallback for non-zarr files or if napari-ome-zarr fails
            # Load image using the unified loader function
            image_data = load_image_file(filepath)

            # Handle multi-layer data from OME-Zarr or enhanced basic loading
            if isinstance(image_data, list):
                # Channel-specific colormaps: R, G, B, then additional colors
                channel_colormaps = [
                    "red",
                    "green",
                    "blue",
                    "cyan",
                    "magenta",
                    "yellow",
                    "orange",
                    "purple",
                    "pink",
                    "gray",
                ]

                # This is from napari-ome-zarr reader or enhanced basic loading - add each layer separately
                for layer_idx, layer_info in enumerate(image_data):
                    # Handle different formats of layer_info
                    if isinstance(layer_info, tuple) and len(layer_info) == 3:
                        # Format: (data, add_kwargs, layer_type)
                        data, add_kwargs, layer_type = layer_info
                    elif (
                        isinstance(layer_info, tuple) and len(layer_info) == 2
                    ):
                        # Format: (data, add_kwargs) - assume image type
                        data, add_kwargs = layer_info
                        layer_type = "image"
                    else:
                        # Just data - create minimal kwargs
                        data = layer_info
                        add_kwargs = {}
                        layer_type = "image"

                    # Ensure proper naming and colormaps for processed images
                    filename = os.path.basename(filepath)

                    if layer_type == "image":
                        # Check if this is a multi-channel image that needs to be split using channel_axis
                        if hasattr(data, "shape") and len(data.shape) >= 3:
                            # Look for a channel dimension (small dimension, typically <= 10)
                            potential_channel_dims = []
                            for dim_idx, dim_size in enumerate(data.shape):
                                if dim_size <= 10 and dim_size > 1:
                                    potential_channel_dims.append(
                                        (dim_idx, dim_size)
                                    )

                            # If we found a potential channel dimension, use napari's channel_axis
                            if potential_channel_dims:
                                # Use the first potential channel dimension
                                channel_axis, num_channels = (
                                    potential_channel_dims[0]
                                )
                                print(
                                    f"Using napari channel_axis={channel_axis} for {num_channels} processed channels"
                                )

                                # Let napari handle channel splitting automatically with proper colormaps
                                layers = self.viewer.add_image(
                                    data,
                                    channel_axis=channel_axis,
                                    name=f"Processed: {filename}",
                                    blending="additive",
                                )

                                # Track all the layers napari created
                                if isinstance(layers, list):
                                    self.current_processed_images.extend(
                                        layers
                                    )
                                else:
                                    self.current_processed_images.append(
                                        layers
                                    )

                                continue  # Skip the normal single-layer processing

                        # Normal single-layer processing (no channel splitting needed)
                        # Override/set colormap for proper channel assignment
                        if "colormap" not in add_kwargs:
                            add_kwargs["colormap"] = (
                                channel_colormaps[layer_idx]
                                if layer_idx < len(channel_colormaps)
                                else "gray"
                            )

                        if "blending" not in add_kwargs:
                            add_kwargs["blending"] = "additive"

                        # Ensure proper naming for processed images
                        if "name" not in add_kwargs or not add_kwargs["name"]:
                            add_kwargs["name"] = (
                                f"Processed C{layer_idx+1}: {filename}"
                            )
                        elif not add_kwargs["name"].startswith("Processed"):
                            add_kwargs["name"] = (
                                f"Processed {add_kwargs['name']}"
                            )

                        layer = self.viewer.add_image(data, **add_kwargs)
                        self.current_processed_images.append(layer)

                    elif layer_type == "labels":
                        if "name" not in add_kwargs or not add_kwargs["name"]:
                            add_kwargs["name"] = (
                                f"Processed Labels{layer_idx+1}: {filename}"
                            )
                        elif not add_kwargs["name"].startswith("Processed"):
                            add_kwargs["name"] = (
                                f"Processed {add_kwargs['name']}"
                            )

                        layer = self.viewer.add_labels(data, **add_kwargs)
                        self.current_processed_images.append(layer)

                # Switch to 3D view if data has meaningful 3D dimensions
                if len(self.current_processed_images) > 0:
                    # Get the first layer's data safely
                    first_layer = self.current_processed_images[0]
                    if hasattr(first_layer, "data"):
                        first_layer_data = first_layer.data
                        if self._should_enable_3d_view(first_layer_data):
                            self.viewer.dims.ndisplay = 3
                            print(
                                f"Switched to 3D view for processed data with shape: {first_layer_data.shape}"
                            )

                # Move all processed layers to top
                for layer in self.current_processed_images:
                    if layer in self.viewer.layers:
                        layer_index = self.viewer.layers.index(layer)
                        if layer_index < len(self.viewer.layers) - 1:
                            self.viewer.layers.move(
                                layer_index, len(self.viewer.layers) - 1
                            )

                self.viewer.status = f"Loaded {len(self.current_processed_images)} processed channels from {os.path.basename(filepath)}"
                return

            # Handle single image data
            image = image_data

            # Remove singletons if it's a numpy array
            if hasattr(image, "squeeze") and not hasattr(image, "chunks"):
                image = np.squeeze(image)

            filename = os.path.basename(filepath)

            # Check for multi-channel data in single array
            if (
                hasattr(image, "shape")
                and len(image.shape) > 2
                and image.shape[0] <= 10
                and image.shape[0] > 1
            ):  # Likely channels first
                print(
                    f"Using napari channel_axis=0 for {image.shape[0]} processed channels"
                )

                # Use napari's channel_axis to automatically split channels with proper colormaps
                layers = self.viewer.add_image(
                    image,
                    channel_axis=0,
                    name=f"Processed: {filename}",
                    blending="additive",
                )

                # Track all the layers napari created
                if isinstance(layers, list):
                    self.current_processed_images.extend(layers)
                else:
                    self.current_processed_images.append(layers)

                # Switch to 3D view if data has meaningful 3D dimensions (excluding channel dim)
                if len(self.current_processed_images) > 0:
                    first_layer = self.current_processed_images[0]
                    if hasattr(first_layer, "data"):
                        channel_data = first_layer.data
                        if self._should_enable_3d_view(channel_data):
                            self.viewer.dims.ndisplay = 3
                            print(
                                f"Switched to 3D view for processed multi-channel data with shape: {channel_data.shape}"
                            )

                # Move all processed layers to top
                for layer in self.current_processed_images:
                    if layer in self.viewer.layers:
                        layer_index = self.viewer.layers.index(layer)
                        if layer_index < len(self.viewer.layers) - 1:
                            self.viewer.layers.move(
                                layer_index, len(self.viewer.layers) - 1
                            )

                self.viewer.status = f"Loaded {len(self.current_processed_images)} processed channels from {filename}"
                return

            # Single channel processed image
            filename = os.path.basename(filepath)
            # Check if filename contains label indicators
            is_label = "labels" in filename or "semantic" in filename

            # Add the layer using the appropriate method
            if is_label:
                # Ensure it's an appropriate dtype for labels
                if hasattr(image, "astype") and not np.issubdtype(
                    image.dtype, np.integer
                ):
                    image = image.astype(np.uint32)

                layer = self.viewer.add_labels(
                    image, name=f"Processed Labels: {filename}"
                )
            else:
                layer = self.viewer.add_image(
                    image, name=f"Processed: {filename}"
                )

            self.current_processed_images.append(layer)

            # Switch to 3D view if data has meaningful 3D dimensions
            if hasattr(layer, "data") and self._should_enable_3d_view(
                layer.data
            ):
                self.viewer.dims.ndisplay = 3
                print(
                    f"Switched to 3D view for processed single-channel data with shape: {layer.data.shape}"
                )

            # Move the processed layer to the top of the stack
            if layer in self.viewer.layers:
                layer_index = self.viewer.layers.index(layer)
                if layer_index < len(self.viewer.layers) - 1:
                    self.viewer.layers.move(
                        layer_index, len(self.viewer.layers) - 1
                    )

            # Update status with success message
            self.viewer.status = f"Loaded {filename} (moved to top layer)"

        except (ValueError, TypeError, OSError, ImportError) as e:
            print(f"Error loading processed image {filepath}: {e}")
            import traceback

            traceback.print_exc()
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
    input_suffix={
        "label": "File Suffix (Example: .tif,.zarr)",
        "value": ".tif,.zarr",
    },
)
def file_selector(
    viewer: napari.Viewer, input_folder: str, input_suffix: str = ".tif,.zarr"
) -> List[str]:
    """
    Find files in a specified input folder with a given suffix and prepare for batch processing.
    """
    # Validate input_folder
    if not os.path.isdir(input_folder):
        viewer.status = f"Invalid input folder: {input_folder}"
        return []

    # Parse multiple suffixes
    suffixes = [s.strip() for s in input_suffix.split(",") if s.strip()]
    if not suffixes:
        suffixes = [".tif"]  # Fallback to tif if no valid suffixes

    # Find matching files with multiple suffix support
    matching_files = []
    for f in os.listdir(input_folder):
        if any(f.endswith(suffix) for suffix in suffixes):
            matching_files.append(os.path.join(input_folder, f))

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


# Create a modified file_selector with browse button
file_selector = add_browse_button_to_folder_field(
    file_selector, "input_folder"
)


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
        """Process a single file with support for large TIFF and Zarr files"""
        try:
            # Load the image using the unified loader
            image_data = load_image_file(filepath)

            # Handle multi-layer data from OME-Zarr - extract first layer for processing
            if isinstance(image_data, list):
                print(
                    f"Processing first layer of multi-layer file: {filepath}"
                )
                # Take the first image layer
                for data, _add_kwargs, layer_type in image_data:
                    if layer_type == "image":
                        image = data
                        break
                else:
                    # No image layer found, take first available
                    image = image_data[0][0]
            else:
                image = image_data

            # Store original dtype for saving
            if hasattr(image, "dtype"):
                image_dtype = image.dtype
            else:
                image_dtype = np.float32

            print(
                f"Original image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}, dtype: {image_dtype}"
            )

            # Check if this is a folder-processing function that shouldn't save individual files
            function_name = getattr(
                self.processing_func, "__name__", "unknown"
            )
            is_folder_function = (
                "timepoint" in function_name.lower()
                or "merge" in function_name.lower()
                or "folder" in function_name.lower()
            )

            # Convert dask array to numpy for processing functions that don't support dask
            if hasattr(image, "chunks") and hasattr(image, "compute"):
                print("Converting dask array to numpy for processing...")
                # For very large arrays, we might want to process in chunks
                try:
                    image = image.compute()
                except MemoryError:
                    print(
                        "Memory error computing dask array, trying chunked processing..."
                    )
                    # Could implement chunked processing here if needed
                    raise

            # Apply processing with parameters
            processed_result = self.processing_func(image, **self.param_values)

            # Check if result is points data (for spot detection functions)
            if (
                isinstance(processed_result, np.ndarray)
                and processed_result.ndim == 2
                and processed_result.shape[1] in [2, 3]  # 2D or 3D coordinates
                and processed_result.dtype in [np.float32, np.float64]
            ):  # Coordinate data

                print(f"Detected points data: {processed_result.shape} points")

                # Save points as numpy array
                filename = os.path.basename(filepath)
                name, _ = os.path.splitext(filename)
                points_filename = f"{name}_spots.npy"
                points_filepath = os.path.join(
                    self.output_folder, points_filename
                )

                np.save(points_filepath, processed_result)
                print(f"Saved points to: {points_filepath}")

                # Also save as CSV if requested
                if hasattr(self, "param_values") and self.param_values.get(
                    "output_csv", False
                ):
                    csv_filename = f"{name}_spots.csv"
                    csv_filepath = os.path.join(
                        self.output_folder, csv_filename
                    )

                    try:
                        # Try to save as CSV with pandas
                        import pandas as pd

                        columns = (
                            ["y", "x"]
                            if processed_result.shape[1] == 2
                            else ["z", "y", "x"]
                        )
                        df = pd.DataFrame(processed_result, columns=columns)
                        df.to_csv(csv_filepath, index=False)
                        print(f"Saved CSV to: {csv_filepath}")
                    except ImportError:
                        # Fallback to numpy if pandas not available
                        np.savetxt(
                            csv_filepath,
                            processed_result,
                            delimiter=",",
                            header=(
                                "y,x"
                                if processed_result.shape[1] == 2
                                else "z,y,x"
                            ),
                            comments="",
                        )
                        print(f"Saved CSV (numpy fallback) to: {csv_filepath}")

                return {
                    "original_file": filepath,
                    "processed_file": points_filepath,
                }

            # Handle as image data (original logic)
            processed_image = processed_result

            print(
                f"Processed image shape before removing singletons: {processed_image.shape}, dtype: {processed_image.dtype}"
            )

            # For folder functions, check if the output is the same as input
            if is_folder_function:
                if np.array_equal(processed_image, image):
                    print(
                        "Folder function returned unchanged image - skipping individual file save"
                    )
                    return None
                else:
                    print(
                        "Folder function returned different data - will save individual file"
                    )

            # Remove ALL singleton dimensions from the processed image
            processed_image = np.squeeze(processed_image)

            print(
                f"Processed image shape after removing singletons: {processed_image.shape}"
            )

            # Generate new filename base
            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)

            # Handle multiple input suffixes for filename generation
            input_suffixes = [
                s.strip() for s in self.input_suffix.split(",") if s.strip()
            ]
            matched_suffix = ""
            for suffix in input_suffixes:
                suffix_clean = suffix.replace(
                    ".", ""
                )  # Remove dot for comparison
                if name.endswith(suffix_clean):
                    matched_suffix = suffix_clean
                    break

            if matched_suffix:
                new_filename_base = (
                    name[: -len(matched_suffix)] + self.output_suffix
                )
            else:
                new_filename_base = name + self.output_suffix

            # For zarr input, default to .tif output unless processing function specifies otherwise
            if filepath.lower().endswith(".zarr") and ext == ".zarr":
                ext = ".tif"

            # Check if the first dimension should be treated as channels
            is_multi_channel = (
                processed_image.ndim > 2 and processed_image.shape[0] <= 10
            )

            if is_multi_channel:
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

                    # For very large files, use BigTIFF format
                    use_bigtiff = size_gb > 2.0

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
                        print(
                            f"Regular image channel, saving with dtype {image_dtype} and bigtiff={use_bigtiff}"
                        )
                        tifffile.imwrite(
                            channel_filepath,
                            channel_image.astype(image_dtype),
                            compression="zlib",
                            bigtiff=use_bigtiff,
                        )

                    processed_files.append(channel_filepath)

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

                # For very large files, use BigTIFF format
                use_bigtiff = size_gb > 2.0

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

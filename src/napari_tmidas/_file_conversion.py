"""
Enhanced Batch Microscopy Image File Conversion
===============================================
This module provides batch conversion of microscopy image files to a common format.

Supported formats: Leica LIF, Nikon ND2, Zeiss CZI, TIFF-based whole slide images (NDPI), Acquifer datasets
"""

import concurrent.futures
import contextlib
import gc
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import napari
import nd2
import numpy as np
import tifffile
import zarr
from dask.diagnostics import ProgressBar
from magicgui import magicgui
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from pylibCZIrw import czi as pyczi
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
from readlif.reader import LifFile
from tiffslide import TiffSlide


# Custom exceptions for better error handling
class FileFormatError(Exception):
    """Raised when file format is not supported or corrupted"""


class SeriesIndexError(Exception):
    """Raised when series index is out of range"""


class ConversionError(Exception):
    """Raised when file conversion fails"""


class SeriesTableWidget(QTableWidget):
    """Custom table widget to display original files and their series"""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Original Files", "Series"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.file_data = {}  # {filepath: {type, series_count, row}}
        self.current_file = None
        self.current_series = None

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
            f"{series_count} series" if series_count > 0 else "Single image"
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
            item = self.item(row, 0)
            if item:
                filepath = item.data(Qt.UserRole)
                file_info = self.file_data.get(filepath)

                if file_info and file_info["series_count"] > 0:
                    self.current_file = filepath
                    self.parent().set_selected_series(filepath, 0)
                    self.parent().show_series_details(filepath)
                else:
                    self.parent().set_selected_series(filepath, 0)
                    self.parent().load_image(filepath)


class SeriesDetailWidget(QWidget):
    """Widget to display and select series from a file"""

    def __init__(self, parent, viewer: napari.Viewer):
        super().__init__()
        self.parent = parent
        self.viewer = viewer
        self.current_file = None
        self.max_series = 0

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Series selection
        self.series_label = QLabel("Select Series:")
        layout.addWidget(self.series_label)

        self.series_selector = QComboBox()
        layout.addWidget(self.series_selector)

        # Export all series option
        self.export_all_checkbox = QCheckBox("Export All Series")
        self.export_all_checkbox.toggled.connect(self.toggle_export_all)
        layout.addWidget(self.export_all_checkbox)

        self.series_selector.currentIndexChanged.connect(self.series_selected)

        # Preview button
        preview_button = QPushButton("Preview Selected Series")
        preview_button.clicked.connect(self.preview_series)
        layout.addWidget(preview_button)

        # Info label
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

    def toggle_export_all(self, checked):
        """Handle toggle of export all checkbox"""
        if self.current_file:
            self.series_selector.setEnabled(not checked)
            self.parent.set_export_all_series(self.current_file, checked)
            if not checked:
                self.series_selected(self.series_selector.currentIndex())

    def set_file(self, filepath: str):
        """Set the current file and update series list"""
        self.current_file = filepath
        self.series_selector.clear()
        self.export_all_checkbox.setChecked(False)
        self.series_selector.setEnabled(True)

        try:
            file_loader = self.parent.get_file_loader(filepath)
            if not file_loader:
                raise FileFormatError(f"No loader available for {filepath}")

            series_count = file_loader.get_series_count(filepath)
            self.max_series = series_count

            for i in range(series_count):
                self.series_selector.addItem(f"Series {i}", i)

            # Estimate file size for format recommendation
            if series_count > 0:
                try:
                    size_gb = self._estimate_file_size(filepath, file_loader)
                    self.info_label.setText(
                        f"File contains {series_count} series (estimated size: {size_gb:.2f}GB)"
                    )
                    self.parent.update_format_buttons(size_gb > 4)
                except (MemoryError, OverflowError, OSError) as e:
                    self.info_label.setText(
                        f"File contains {series_count} series"
                    )
                    print(f"Size estimation failed: {e}")

        except (FileNotFoundError, PermissionError, FileFormatError) as e:
            self.info_label.setText(f"Error: {str(e)}")

    def _estimate_file_size(self, filepath: str, file_loader) -> float:
        """Estimate file size in GB"""
        file_type = self.parent.get_file_type(filepath)

        if file_type == "ND2":
            try:
                with nd2.ND2File(filepath) as nd2_file:
                    dims = dict(nd2_file.sizes)
                    pixel_size = nd2_file.dtype.itemsize
                    total_elements = np.prod([dims[dim] for dim in dims])
                    return (total_elements * pixel_size) / (1024**3)
            except (OSError, AttributeError, ValueError):
                pass

        # Fallback estimation based on file size
        try:
            file_size = os.path.getsize(filepath)
            return file_size / (1024**3)
        except OSError:
            return 0.0

    def series_selected(self, index: int):
        """Handle series selection"""
        if index >= 0 and self.current_file:
            series_index = self.series_selector.itemData(index)

            if series_index >= self.max_series:
                raise SeriesIndexError(
                    f"Series index {series_index} out of range (max: {self.max_series-1})"
                )

            self.parent.set_selected_series(self.current_file, series_index)

    def preview_series(self):
        """Preview the selected series in Napari"""
        if not self.current_file or self.series_selector.currentIndex() < 0:
            return

        series_index = self.series_selector.itemData(
            self.series_selector.currentIndex()
        )

        if series_index >= self.max_series:
            self.info_label.setText("Error: Series index out of range")
            return

        try:
            file_loader = self.parent.get_file_loader(self.current_file)
            metadata = file_loader.get_metadata(
                self.current_file, series_index
            )
            image_data = file_loader.load_series(
                self.current_file, series_index
            )

            # Reorder dimensions for Napari if needed
            if metadata and "axes" in metadata:
                napari_order = "CTZYX"[: len(image_data.shape)]
                image_data = self._reorder_dimensions(
                    image_data, metadata, napari_order
                )

            self.viewer.layers.clear()
            layer_name = (
                f"{Path(self.current_file).stem}_series_{series_index}"
            )
            self.viewer.add_image(image_data, name=layer_name)
            self.viewer.status = f"Previewing {layer_name}"

        except (
            FileNotFoundError,
            SeriesIndexError,
            MemoryError,
            FileFormatError,
        ) as e:
            error_msg = f"Error loading series: {str(e)}"
            self.viewer.status = error_msg
            QMessageBox.warning(self, "Preview Error", error_msg)

    def _reorder_dimensions(self, image_data, metadata, target_order="YXZTC"):
        """Reorder dimensions based on metadata axes information"""
        if not metadata or "axes" not in metadata:
            return image_data

        source_order = metadata["axes"]
        ndim = len(image_data.shape)

        if len(source_order) != ndim or len(target_order) != ndim:
            return image_data

        try:
            reorder_indices = []
            for axis in target_order:
                if axis in source_order:
                    reorder_indices.append(source_order.index(axis))
                else:
                    return image_data

            if hasattr(image_data, "dask"):
                return image_data.transpose(reorder_indices)
            else:
                return np.transpose(image_data, reorder_indices)

        except (ValueError, IndexError) as e:
            print(f"Dimension reordering failed: {e}")
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
    def load_series(
        filepath: str, series_index: int
    ) -> Union[np.ndarray, da.Array]:
        raise NotImplementedError()

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        return {}


class LIFLoader(FormatLoader):
    """
    Leica LIF loader based on readlif API

    """

    @staticmethod
    def can_load(filepath: str) -> bool:
        """Check if file can be loaded as LIF"""
        if not filepath.lower().endswith(".lif"):
            return False

        try:
            # Quick validation by attempting to open
            lif_file = LifFile(filepath)
            # Check if we can at least get the image list
            list(lif_file.get_iter_image())
            return True
        except (OSError, ValueError, ImportError, AttributeError) as e:
            print(f"Cannot load LIF file {filepath}: {e}")
            return False

    @staticmethod
    def get_series_count(filepath: str) -> int:
        """Get number of series in LIF file with better error handling"""
        try:
            lif_file = LifFile(filepath)
            # Count images more safely
            count = 0
            for _ in lif_file.get_iter_image():
                count += 1
            return count
        except (OSError, ValueError, ImportError, AttributeError) as e:
            print(f"Error counting series in {filepath}: {e}")
            return 0

    @staticmethod
    def load_series(
        filepath: str, series_index: int
    ) -> Union[np.ndarray, da.Array]:
        """
        Load LIF series with improved memory management and error handling
        """
        lif_file = None
        try:
            print(f"Loading LIF series {series_index} from {filepath}")
            lif_file = LifFile(filepath)

            # Get the specific image
            images = list(lif_file.get_iter_image())
            if series_index >= len(images):
                raise SeriesIndexError(
                    f"Series index {series_index} out of range (0-{len(images)-1})"
                )

            image = images[series_index]

            # Get image properties
            channels = image.channels
            z_stacks = image.nz
            timepoints = image.nt
            x_dim, y_dim = image.dims[0], image.dims[1]

            print(
                f"LIF Image dimensions: T={timepoints}, Z={z_stacks}, C={channels}, Y={y_dim}, X={x_dim}"
            )

            # Calculate memory requirements
            total_frames = timepoints * z_stacks * channels
            estimated_size_gb = (total_frames * x_dim * y_dim * 2) / (
                1024**3
            )  # Assuming 16-bit

            print(
                f"Estimated memory: {estimated_size_gb:.2f} GB for {total_frames} frames"
            )

            # Choose loading strategy based on size
            if estimated_size_gb > 4.0:
                print("Large dataset detected, using Dask lazy loading")
                return LIFLoader._load_as_dask(
                    image, timepoints, z_stacks, channels, y_dim, x_dim
                )
            elif estimated_size_gb > 1.0:
                print("Medium dataset, using chunked numpy loading")
                return LIFLoader._load_chunked_numpy(
                    image, timepoints, z_stacks, channels, y_dim, x_dim
                )
            else:
                print("Small dataset, using standard numpy loading")
                return LIFLoader._load_numpy(
                    image, timepoints, z_stacks, channels, y_dim, x_dim
                )

        except (OSError, IndexError, ValueError, AttributeError) as e:
            print("Full error traceback for LIF loading:")
            traceback.print_exc()
            raise FileFormatError(
                f"Failed to load LIF series {series_index}: {str(e)}"
            ) from e
        finally:
            # Cleanup
            if lif_file is not None:
                with contextlib.suppress(Exception):
                    # readlif doesn't have explicit close, but we can delete the reference
                    del lif_file
            gc.collect()

    @staticmethod
    def _load_numpy(
        image,
        timepoints: int,
        z_stacks: int,
        channels: int,
        y_dim: int,
        x_dim: int,
    ) -> np.ndarray:
        """Load small datasets directly into numpy array"""

        # Determine data type from first available frame
        dtype = np.uint16  # Default
        test_frame = None
        for t in range(min(1, timepoints)):
            for z in range(min(1, z_stacks)):
                for c in range(min(1, channels)):
                    try:
                        test_frame = image.get_frame(z=z, t=t, c=c)
                        if test_frame is not None:
                            dtype = np.array(test_frame).dtype
                            break
                    except (OSError, ValueError, AttributeError):
                        continue
                if test_frame is not None:
                    break
            if test_frame is not None:
                break

        # Pre-allocate array
        series_shape = (timepoints, z_stacks, channels, y_dim, x_dim)
        series_data = np.zeros(series_shape, dtype=dtype)

        # Load frames with better error handling
        missing_frames = 0
        loaded_frames = 0

        for t in range(timepoints):
            for z in range(z_stacks):
                for c in range(channels):
                    try:
                        frame = image.get_frame(z=z, t=t, c=c)
                        if frame is not None:
                            frame_array = np.array(frame, dtype=dtype)
                            # Ensure correct dimensions
                            if frame_array.shape == (y_dim, x_dim):
                                series_data[t, z, c, :, :] = frame_array
                                loaded_frames += 1
                            else:
                                print(
                                    f"Warning: Frame shape mismatch at T={t}, Z={z}, C={c}: "
                                    f"expected {(y_dim, x_dim)}, got {frame_array.shape}"
                                )
                                missing_frames += 1
                        else:
                            missing_frames += 1
                    except (OSError, ValueError, AttributeError) as e:
                        print(f"Error loading frame T={t}, Z={z}, C={c}: {e}")
                        missing_frames += 1

            # Progress feedback for large datasets
            if timepoints > 10 and (t + 1) % max(1, timepoints // 10) == 0:
                print(f"Loaded {t + 1}/{timepoints} timepoints")

        print(
            f"Loading complete: {loaded_frames} frames loaded, {missing_frames} frames missing"
        )

        if loaded_frames == 0:
            raise FileFormatError(
                "No valid frames could be loaded from LIF file"
            )

        return series_data

    @staticmethod
    def _load_chunked_numpy(
        image,
        timepoints: int,
        z_stacks: int,
        channels: int,
        y_dim: int,
        x_dim: int,
    ) -> np.ndarray:
        """Load medium datasets with memory management"""

        print("Using chunked loading strategy")

        # Load in chunks to manage memory
        chunk_size = max(
            1, min(10, timepoints // 2)
        )  # Process multiple timepoints at once

        # Determine data type
        dtype = np.uint16
        for t in range(min(1, timepoints)):
            for z in range(min(1, z_stacks)):
                for c in range(min(1, channels)):
                    try:
                        frame = image.get_frame(z=z, t=t, c=c)
                        if frame is not None:
                            dtype = np.array(frame).dtype
                            break
                    except (OSError, ValueError, AttributeError):
                        continue

        # Pre-allocate final array
        series_shape = (timepoints, z_stacks, channels, y_dim, x_dim)
        series_data = np.zeros(series_shape, dtype=dtype)

        missing_frames = 0

        for t_start in range(0, timepoints, chunk_size):
            t_end = min(t_start + chunk_size, timepoints)
            print(f"Loading timepoints {t_start} to {t_end-1}")

            for t in range(t_start, t_end):
                for z in range(z_stacks):
                    for c in range(channels):
                        try:
                            frame = image.get_frame(z=z, t=t, c=c)
                            if frame is not None:
                                frame_array = np.array(frame, dtype=dtype)
                                if frame_array.shape == (y_dim, x_dim):
                                    series_data[t, z, c, :, :] = frame_array
                                else:
                                    missing_frames += 1
                            else:
                                missing_frames += 1
                        except (OSError, ValueError, AttributeError):
                            missing_frames += 1

            # Force garbage collection after each chunk
            gc.collect()

        if missing_frames > 0:
            print(
                f"Warning: {missing_frames} frames were missing and filled with zeros"
            )

        return series_data

    @staticmethod
    def _load_as_dask(
        image,
        timepoints: int,
        z_stacks: int,
        channels: int,
        y_dim: int,
        x_dim: int,
    ) -> da.Array:
        """Load large datasets as dask arrays for lazy evaluation"""

        print("Creating Dask array for lazy loading")

        # Determine data type
        dtype = np.uint16
        for t in range(min(1, timepoints)):
            for z in range(min(1, z_stacks)):
                for c in range(min(1, channels)):
                    try:
                        frame = image.get_frame(z=z, t=t, c=c)
                        if frame is not None:
                            dtype = np.array(frame).dtype
                            break
                    except (OSError, ValueError, AttributeError):
                        continue

        # Define chunk size for dask array
        # Chunk by timepoints to make it memory efficient
        time_chunk = (
            max(1, min(5, timepoints // 4)) if timepoints > 4 else timepoints
        )

        def load_chunk(block_id):
            """Load a specific chunk of the data"""
            t_start = block_id[0] * time_chunk
            t_end = min(t_start + time_chunk, timepoints)

            chunk_shape = (t_end - t_start, z_stacks, channels, y_dim, x_dim)
            chunk_data = np.zeros(chunk_shape, dtype=dtype)

            for t_idx, t in enumerate(range(t_start, t_end)):
                for z in range(z_stacks):
                    for c in range(channels):
                        try:
                            frame = image.get_frame(z=z, t=t, c=c)
                            if frame is not None:
                                frame_array = np.array(frame, dtype=dtype)
                                if frame_array.shape == (y_dim, x_dim):
                                    chunk_data[t_idx, z, c, :, :] = frame_array
                        except (OSError, ValueError, AttributeError) as e:
                            print(
                                f"Error in chunk loading T={t}, Z={z}, C={c}: {e}"
                            )

            return chunk_data

        # Use da.from_delayed for custom loading function
        from dask import delayed

        # Create delayed objects for each chunk
        delayed_chunks = []
        for t_chunk_idx in range((timepoints + time_chunk - 1) // time_chunk):
            delayed_chunk = delayed(load_chunk)((t_chunk_idx,))
            delayed_chunks.append(delayed_chunk)

        # Convert to dask arrays and concatenate
        dask_chunks = []
        for i, delayed_chunk in enumerate(delayed_chunks):
            t_start = i * time_chunk
            t_end = min(t_start + time_chunk, timepoints)
            chunk_shape = (t_end - t_start, z_stacks, channels, y_dim, x_dim)

            dask_chunk = da.from_delayed(
                delayed_chunk, shape=chunk_shape, dtype=dtype
            )
            dask_chunks.append(dask_chunk)

        # Concatenate along time axis
        if len(dask_chunks) == 1:
            return dask_chunks[0]
        else:
            return da.concatenate(dask_chunks, axis=0)

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        """Extract metadata with better error handling"""
        try:
            lif_file = LifFile(filepath)
            images = list(lif_file.get_iter_image())

            if series_index >= len(images):
                return {}

            image = images[series_index]

            metadata = {
                "axes": "TZCYX",  # Standard microscopy order
                "unit": "um",
            }

            # Try to get resolution information
            try:
                if hasattr(image, "scale") and image.scale:
                    # scale is typically [x_res, y_res, z_res] in micrometers per pixel
                    if len(image.scale) >= 2:
                        x_scale, y_scale = image.scale[0], image.scale[1]
                        if x_scale and y_scale and x_scale > 0 and y_scale > 0:
                            metadata["resolution"] = (
                                1.0 / x_scale,
                                1.0 / y_scale,
                            )  # Convert to pixels per micrometer

                    if (
                        len(image.scale) >= 3
                        and image.scale[2]
                        and image.scale[2] > 0
                    ):
                        metadata["spacing"] = image.scale[
                            2
                        ]  # Z spacing in micrometers
            except (AttributeError, TypeError, IndexError):
                pass

            # Add image dimensions info
            with contextlib.suppress(AttributeError, IndexError):
                metadata.update(
                    {
                        "timepoints": image.nt,
                        "z_stacks": image.nz,
                        "channels": image.channels,
                        "width": image.dims[0],
                        "height": image.dims[1],
                    }
                )

            return metadata

        except (OSError, IndexError, AttributeError, ImportError) as e:
            print(f"Warning: Could not extract metadata from {filepath}: {e}")
            return {}


class ND2Loader(FormatLoader):
    """
    Loader for Nikon ND2 files based on nd2 API
    """

    @staticmethod
    def can_load(filepath: str) -> bool:
        return filepath.lower().endswith(".nd2")

    @staticmethod
    def get_series_count(filepath: str) -> int:
        """Get number of series (positions) in ND2 file"""
        try:
            with nd2.ND2File(filepath) as nd2_file:
                # The 'P' dimension represents positions/series
                return nd2_file.sizes.get("P", 1)
        except (OSError, ValueError, ImportError) as e:
            print(
                f"Warning: Could not determine series count for {filepath}: {e}"
            )
            return 0

    @staticmethod
    def load_series(
        filepath: str, series_index: int
    ) -> Union[np.ndarray, da.Array]:
        """
        Load a specific series from ND2 file
        """
        try:
            # First, get basic info about the file
            with nd2.ND2File(filepath) as nd2_file:
                dims = nd2_file.sizes
                max_series = dims.get("P", 1)

                if series_index >= max_series:
                    raise SeriesIndexError(
                        f"Series index {series_index} out of range (0-{max_series-1})"
                    )

                # Calculate memory requirements for decision making
                total_voxels = np.prod([dims[k] for k in dims if k != "P"])
                pixel_size = np.dtype(nd2_file.dtype).itemsize
                size_gb = (total_voxels * pixel_size) / (1024**3)

                print(f"ND2 file dimensions: {dims}")
                print(f"Single series estimated size: {size_gb:.2f} GB")

            # Now load the data using the appropriate method
            use_dask = size_gb > 2.0

            if "P" in dims and dims["P"] > 1:
                # Multi-position file
                return ND2Loader._load_multi_position(
                    filepath, series_index, use_dask, dims
                )
            else:
                # Single position file
                if series_index != 0:
                    raise SeriesIndexError(
                        "Single position file only supports series index 0"
                    )
                return ND2Loader._load_single_position(filepath, use_dask)

        except (FileNotFoundError, PermissionError) as e:
            raise FileFormatError(
                f"Cannot access ND2 file {filepath}: {str(e)}"
            ) from e
        except (
            OSError,
            ValueError,
            AttributeError,
            ImportError,
            KeyError,
        ) as e:
            raise FileFormatError(
                f"Failed to load ND2 series {series_index}: {str(e)}"
            ) from e

    @staticmethod
    def _load_multi_position(
        filepath: str, series_index: int, use_dask: bool, dims: dict
    ):
        """Load specific position from multi-position file"""

        if use_dask:
            # METHOD 1: Use nd2.imread with xarray for better indexing
            try:
                print("Loading multi-position file as dask-xarray...")
                data_xr = nd2.imread(filepath, dask=True, xarray=True)

                # Use xarray's isel to extract position - this stays lazy!
                series_data = data_xr.isel(P=series_index)

                # Return the underlying dask array
                return (
                    series_data.data
                    if hasattr(series_data, "data")
                    else series_data.values
                )

            except (
                OSError,
                ValueError,
                AttributeError,
                MemoryError,
                FileFormatError,
            ) as e:
                print(f"xarray method failed: {e}, trying alternative...")

            # METHOD 2: Fallback - use direct indexing on ResourceBackedDaskArray
            try:
                print(
                    "Loading multi-position file with direct dask indexing..."
                )
                # We need to keep the file open for the duration of the dask operations
                # This is tricky - we'll compute immediately for now to avoid file closure issues

                with nd2.ND2File(filepath) as nd2_file:
                    dask_array = nd2_file.to_dask()

                    # Find position axis
                    axis_names = list(dims.keys())
                    p_axis = axis_names.index("P")

                    # Create slice tuple to extract the specific position
                    # This is the CORRECTED approach for ResourceBackedDaskArray
                    slices = [slice(None)] * len(dask_array.shape)
                    slices[p_axis] = series_index

                    # Extract the series - but we need to compute it while file is open
                    series_data = dask_array[tuple(slices)]

                    # For large arrays, we compute immediately to avoid file closure issues
                    # This is not ideal but necessary due to ResourceBackedDaskArray limitations
                    if hasattr(series_data, "compute"):
                        print(
                            "Computing dask array immediately due to file closure limitations..."
                        )
                        return series_data.compute()
                    else:
                        return series_data

            except (
                OSError,
                ValueError,
                AttributeError,
                MemoryError,
                FileFormatError,
            ) as e:
                print(f"Dask method failed: {e}, falling back to numpy...")

        # METHOD 3: Load as numpy array (for small files or as fallback)
        print("Loading multi-position file as numpy array...")
        with nd2.ND2File(filepath) as nd2_file:
            # Use direct indexing on the ND2File object
            if hasattr(nd2_file, "__getitem__"):
                axis_names = list(dims.keys())
                p_axis = axis_names.index("P")
                slices = [slice(None)] * len(dims)
                slices[p_axis] = series_index
                return nd2_file[tuple(slices)]
            else:
                # Final fallback: load entire array and slice
                full_data = nd2.imread(filepath, dask=False)
                axis_names = list(dims.keys())
                p_axis = axis_names.index("P")
                return np.take(full_data, series_index, axis=p_axis)

    @staticmethod
    def _load_single_position(filepath: str, use_dask: bool):
        """Load single position file"""
        if use_dask:
            # For single position, we can use imread directly
            return nd2.imread(filepath, dask=True)
        else:
            return nd2.imread(filepath, dask=False)

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        """Extract metadata with proper handling of series information"""
        try:
            with nd2.ND2File(filepath) as nd2_file:
                dims = nd2_file.sizes

                # For multi-position files, get dimensions without P axis
                if "P" in dims:
                    if series_index >= dims["P"]:
                        return {}
                    # Remove P dimension for series-specific metadata
                    series_dims = {k: v for k, v in dims.items() if k != "P"}
                else:
                    if series_index != 0:
                        return {}
                    series_dims = dims

                # Create axis string (standard microscopy order: TZCYX)
                axis_order = "TZCYX"
                axes = "".join([ax for ax in axis_order if ax in series_dims])

                # Get voxel/pixel size information
                try:
                    voxel = nd2_file.voxel_size()
                    if voxel:
                        # Convert from micrometers (nd2 default) to resolution
                        x_res = 1 / voxel.x if voxel.x > 0 else 1.0
                        y_res = 1 / voxel.y if voxel.y > 0 else 1.0
                        z_spacing = voxel.z if voxel.z > 0 else 1.0
                    else:
                        x_res, y_res, z_spacing = 1.0, 1.0, 1.0
                except (AttributeError, ValueError, TypeError):
                    x_res, y_res, z_spacing = 1.0, 1.0, 1.0

                metadata = {
                    "axes": axes,
                    "resolution": (x_res, y_res),
                    "unit": "um",
                }

                # Add Z spacing if Z dimension exists
                if "Z" in series_dims and z_spacing != 1.0:
                    metadata["spacing"] = z_spacing

                # Add additional useful metadata
                metadata.update(
                    {
                        "dtype": str(nd2_file.dtype),
                        "shape": tuple(
                            series_dims[ax] for ax in axes if ax in series_dims
                        ),
                        "is_rgb": getattr(nd2_file, "is_rgb", False),
                    }
                )

                return metadata

        except (OSError, AttributeError, ImportError) as e:
            print(f"Warning: Could not extract metadata from {filepath}: {e}")
            return {}


class TIFFSlideLoader(FormatLoader):
    """Loader for whole slide TIFF images (NDPI, etc.)"""

    @staticmethod
    def can_load(filepath: str) -> bool:
        return filepath.lower().endswith((".ndpi", ".svs"))

    @staticmethod
    def get_series_count(filepath: str) -> int:
        try:
            with TiffSlide(filepath) as slide:
                return len(slide.level_dimensions)
        except (OSError, ImportError, ValueError):
            try:
                with tifffile.TiffFile(filepath) as tif:
                    return len(tif.series)
            except (OSError, ValueError, ImportError):
                return 0

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        try:
            with TiffSlide(filepath) as slide:
                if series_index >= len(slide.level_dimensions):
                    raise SeriesIndexError(
                        f"Series index {series_index} out of range"
                    )

                width, height = slide.level_dimensions[series_index]
                return np.array(
                    slide.read_region((0, 0), series_index, (width, height))
                )
        except (OSError, ImportError, AttributeError):
            try:
                with tifffile.TiffFile(filepath) as tif:
                    if series_index >= len(tif.series):
                        raise SeriesIndexError(
                            f"Series index {series_index} out of range"
                        )
                    return tif.series[series_index].asarray()
            except (OSError, IndexError, ValueError, ImportError) as e:
                raise FileFormatError(
                    f"Failed to load TIFF slide series {series_index}: {str(e)}"
                ) from e

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        try:
            with TiffSlide(filepath) as slide:
                if series_index >= len(slide.level_dimensions):
                    return {}

                return {
                    "axes": slide.properties.get(
                        "tiffslide.series-axes", "YX"
                    ),
                    "resolution": (
                        float(slide.properties.get("tiffslide.mpp-x", 1.0)),
                        float(slide.properties.get("tiffslide.mpp-y", 1.0)),
                    ),
                    "unit": "um",
                }
        except (OSError, ImportError, ValueError, KeyError):
            return {}


class CZILoader(FormatLoader):
    """
    Loader for Zeiss CZI files using pylibCZIrw API

    """

    @staticmethod
    def can_load(filepath: str) -> bool:
        if not filepath.lower().endswith(".czi"):
            return False

        # Test if we can actually open the file
        try:
            with pyczi.open_czi(filepath) as czidoc:
                # Try to get basic info to validate the file
                _ = czidoc.total_bounding_box
                return True
        except (
            OSError,
            ImportError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as e:
            print(f"Cannot load CZI file {filepath}: {e}")
            return False

    @staticmethod
    def get_series_count(filepath: str) -> int:
        """
        Get number of series in CZI file

        For CZI files:
        - If scenes exist, each scene is a series
        - If no scenes, there's 1 series (the whole image)
        """
        try:
            with pyczi.open_czi(filepath) as czidoc:
                scenes_bbox = czidoc.scenes_bounding_rectangle

                if scenes_bbox:
                    # File has scenes - each scene is a series
                    scene_count = len(scenes_bbox)
                    print(f"CZI file has {scene_count} scenes")
                    return scene_count
                else:
                    # No scenes - single series
                    print("CZI file has no scenes - treating as single series")
                    return 1

        except (
            OSError,
            ImportError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as e:
            print(f"Error getting series count for {filepath}: {e}")
            return 0

    @staticmethod
    def load_series(
        filepath: str, series_index: int
    ) -> Union[np.ndarray, da.Array]:
        """
        Load a specific series from CZI file using correct pylibCZIrw API
        """
        try:
            print(f"Loading CZI series {series_index} from {filepath}")

            with pyczi.open_czi(filepath) as czidoc:
                # Get file information
                total_bbox = czidoc.total_bounding_box
                scenes_bbox = czidoc.scenes_bounding_rectangle

                print(f"Total bounding box: {total_bbox}")
                print(f"Scenes: {len(scenes_bbox) if scenes_bbox else 0}")

                # Determine if we're dealing with scenes or single image
                if scenes_bbox:
                    # Multi-scene file
                    scene_indices = list(scenes_bbox.keys())
                    if series_index >= len(scene_indices):
                        raise SeriesIndexError(
                            f"Scene index {series_index} out of range (0-{len(scene_indices)-1})"
                        )

                    # Get the actual scene ID (may not be sequential 0,1,2...)
                    scene_id = scene_indices[series_index]
                    print(f"Loading scene ID: {scene_id}")

                    # Read the specific scene
                    # The scene parameter in read() expects the actual scene ID
                    image_data = czidoc.read(scene=scene_id)

                else:
                    # Single scene file
                    if series_index != 0:
                        raise SeriesIndexError(
                            f"Single scene file only supports series index 0, got {series_index}"
                        )

                    print("Loading single scene CZI")
                    # Read without specifying scene
                    image_data = czidoc.read()

                print(
                    f"Raw CZI data shape: {image_data.shape}, dtype: {image_data.dtype}"
                )

                # Simply squeeze out all singleton dimensions
                if hasattr(image_data, "dask"):
                    image_data = da.squeeze(image_data)
                else:
                    image_data = np.squeeze(image_data)

                print(f"Final CZI data shape: {image_data.shape}")

                # Check if we need to use Dask for large arrays
                size_gb = (
                    image_data.nbytes
                    if hasattr(image_data, "nbytes")
                    else np.prod(image_data.shape) * 4
                ) / (1024**3)

                if size_gb > 2.0 and not hasattr(image_data, "dask"):
                    print(
                        f"Large CZI data ({size_gb:.2f}GB), converting to Dask array"
                    )
                    return da.from_array(image_data, chunks="auto")
                else:
                    return image_data

        except (
            OSError,
            ImportError,
            AttributeError,
            ValueError,
            RuntimeError,
        ) as e:
            raise FileFormatError(
                f"Failed to load CZI series {series_index}: {str(e)}"
            ) from e

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        """Extract metadata using correct pylibCZIrw API"""
        try:
            with pyczi.open_czi(filepath) as czidoc:
                scenes_bbox = czidoc.scenes_bounding_rectangle
                total_bbox = czidoc.total_bounding_box

                # Validate series index
                if scenes_bbox:
                    if series_index >= len(scenes_bbox):
                        return {}
                    scene_indices = list(scenes_bbox.keys())
                    scene_id = scene_indices[series_index]
                    print(f"Getting metadata for scene {scene_id}")
                else:
                    if series_index != 0:
                        return {}
                    scene_id = None
                    print("Getting metadata for single scene CZI")

                # Get basic metadata
                metadata = {}

                try:
                    # Get raw metadata XML
                    raw_metadata = czidoc.metadata
                    if raw_metadata:
                        # Extract scale information from XML metadata
                        scale_x = CZILoader._extract_scale_from_xml(
                            raw_metadata, "X"
                        )
                        scale_y = CZILoader._extract_scale_from_xml(
                            raw_metadata, "Y"
                        )
                        scale_z = CZILoader._extract_scale_from_xml(
                            raw_metadata, "Z"
                        )

                        if scale_x and scale_y:
                            metadata["resolution"] = (scale_x, scale_y)
                        if scale_z:
                            metadata["spacing"] = scale_z

                except (AttributeError, RuntimeError):
                    print(
                        "Warning: Could not extract scale information from metadata"
                    )

                # Get actual data to determine final dimensions after squeezing
                try:
                    if scenes_bbox:
                        scene_indices = list(scenes_bbox.keys())
                        scene_id = scene_indices[series_index]
                        sample_data = czidoc.read(scene=scene_id)
                    else:
                        sample_data = czidoc.read()

                    # Squeeze to match what load_series() returns
                    if hasattr(sample_data, "dask"):
                        sample_data = da.squeeze(sample_data)
                    else:
                        sample_data = np.squeeze(sample_data)

                    actual_shape = sample_data.shape
                    actual_ndim = len(actual_shape)
                    print(
                        f"Actual squeezed shape for metadata: {actual_shape}"
                    )

                    # Create axes based on actual squeezed dimensions
                    if actual_ndim == 2:
                        axes = "YX"
                    elif actual_ndim == 3:
                        # Check which dimension survived the squeeze
                        unsqueezed_dims = []
                        for dim, (_start, size) in total_bbox.items():
                            if size > 1 and dim in ["T", "Z", "C"]:
                                unsqueezed_dims.append(dim)

                        if unsqueezed_dims:
                            axes = f"{unsqueezed_dims[0]}YX"  # First non-singleton dim + YX
                        else:
                            axes = "ZYX"  # Default fallback
                    elif actual_ndim == 4:
                        axes = "TCYX"  # Most common 4D case
                    elif actual_ndim == 5:
                        axes = "TZCYX"
                    else:
                        # Fallback: just use YX and pad with standard dims
                        standard_dims = ["T", "Z", "C"]
                        axes = "".join(standard_dims[: actual_ndim - 2]) + "YX"

                    # Ensure axes length matches actual dimensions
                    axes = axes[:actual_ndim]

                    print(
                        f"Final axes for squeezed data: '{axes}' (length: {len(axes)})"
                    )

                except (AttributeError, RuntimeError) as e:
                    print(f"Could not get sample data for metadata: {e}")
                    # Fallback to original logic
                    filtered_dims = {}
                    for dim, (_start, size) in total_bbox.items():
                        if size > 1:  # Only include dimensions with size > 1
                            filtered_dims[dim] = size

                    # Standard microscopy axis order: TZCYX
                    axis_order = "TZCYX"
                    axes = "".join(
                        [ax for ax in axis_order if ax in filtered_dims]
                    )

                    # Fallback to YX if no significant dimensions found
                    if not axes:
                        axes = "YX"

                metadata.update(
                    {
                        "axes": axes,
                        "unit": "um",
                        "total_bounding_box": total_bbox,
                        "has_scenes": bool(scenes_bbox),
                        "scene_count": len(scenes_bbox) if scenes_bbox else 1,
                    }
                )

                # Add scene-specific info if applicable
                if scene_id is not None:
                    metadata["scene_id"] = scene_id

                return metadata

        except (OSError, ImportError, AttributeError, RuntimeError) as e:
            print(f"Warning: Could not extract metadata from {filepath}: {e}")
            return {}

    @staticmethod
    def _extract_scale_from_xml(metadata_xml: str, dimension: str) -> float:
        """
        Extract scale information from CZI XML metadata

        This looks for Distance elements with the specified dimension ID
        """
        try:
            # Pattern to find Distance elements with specific dimension
            pattern = re.compile(
                rf'<Distance[^>]*Id="{re.escape(dimension)}"[^>]*>.*?<Value[^>]*>(.*?)</Value>',
                re.DOTALL | re.IGNORECASE,
            )

            match = pattern.search(metadata_xml)
            if match:
                value = float(match.group(1))
                # CZI typically stores in meters, convert to micrometers
                return value * 1e6

            # Alternative pattern for older CZI format
            pattern2 = re.compile(
                rf'<Scaling>.*?<Items>.*?<Distance.*?Id="{re.escape(dimension)}".*?>.*?<Value>(.*?)</Value>',
                re.DOTALL | re.IGNORECASE,
            )

            match2 = pattern2.search(metadata_xml)
            if match2:
                value = float(match2.group(1))
                return value * 1e6

            return 1.0  # Default fallback

        except (ValueError, TypeError, AttributeError):
            return 1.0


class AcquiferLoader(FormatLoader):
    """Enhanced loader for Acquifer datasets with better detection"""

    _dataset_cache = {}

    @staticmethod
    def can_load(filepath: str) -> bool:
        """Check if directory contains Acquifer-specific patterns"""
        if not os.path.isdir(filepath):
            return False

        try:
            dir_contents = os.listdir(filepath)

            # Check for Acquifer-specific indicators
            acquifer_indicators = [
                "PlateLayout" in dir_contents,
                any(f.startswith("Image") for f in dir_contents),
                any("--PX" in f for f in dir_contents),
                any(f.endswith("_metadata.txt") for f in dir_contents),
                "Well" in str(dir_contents).upper(),
            ]

            if not any(acquifer_indicators):
                return False

            # Verify it contains image files
            image_files = []
            for _root, _, files in os.walk(filepath):
                for file in files:
                    if file.lower().endswith(
                        (".tif", ".tiff", ".png", ".jpg", ".jpeg")
                    ):
                        image_files.append(file)

            return len(image_files) > 0

        except (OSError, PermissionError):
            return False

    @staticmethod
    def _load_dataset(directory):
        """Load and cache Acquifer dataset"""
        if directory in AcquiferLoader._dataset_cache:
            return AcquiferLoader._dataset_cache[directory]

        try:
            from acquifer_napari_plugin.utils import array_from_directory

            # Verify image files exist
            image_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(
                        (".tif", ".tiff", ".png", ".jpg", ".jpeg")
                    ):
                        image_files.append(os.path.join(root, file))

            if not image_files:
                raise FileFormatError(
                    f"No image files found in Acquifer directory: {directory}"
                )

            dataset = array_from_directory(directory)
            AcquiferLoader._dataset_cache[directory] = dataset
            return dataset

        except ImportError as e:
            raise FileFormatError(
                f"Acquifer plugin not available: {str(e)}"
            ) from e
        except (OSError, ValueError, AttributeError) as e:
            raise FileFormatError(
                f"Failed to load Acquifer dataset: {str(e)}"
            ) from e

    @staticmethod
    def get_series_count(filepath: str) -> int:
        try:
            dataset = AcquiferLoader._load_dataset(filepath)
            return len(dataset.coords.get("Well", [1]))
        except (FileFormatError, AttributeError, KeyError):
            return 0

    @staticmethod
    def load_series(filepath: str, series_index: int) -> np.ndarray:
        try:
            dataset = AcquiferLoader._load_dataset(filepath)

            if "Well" in dataset.dims:
                if series_index >= len(dataset.coords["Well"]):
                    raise SeriesIndexError(
                        f"Series index {series_index} out of range"
                    )

                well_value = dataset.coords["Well"].values[series_index]
                well_data = dataset.sel(Well=well_value).squeeze()
                return well_data.values
            else:
                if series_index != 0:
                    raise SeriesIndexError(
                        "Single well dataset only supports series index 0"
                    )
                return dataset.values

        except (AttributeError, KeyError, IndexError) as e:
            raise FileFormatError(
                f"Failed to load Acquifer series {series_index}: {str(e)}"
            ) from e

    @staticmethod
    def get_metadata(filepath: str, series_index: int) -> Dict:
        try:
            dataset = AcquiferLoader._load_dataset(filepath)

            if "Well" in dataset.dims:
                well_value = dataset.coords["Well"].values[series_index]
                well_data = dataset.sel(Well=well_value).squeeze()
                dims = list(well_data.dims)
            else:
                dims = list(dataset.dims)

            # Normalize dimension names
            dims = [
                dim.replace("Channel", "C").replace("Time", "T")
                for dim in dims
            ]
            axes = "".join(dims)

            # Try to extract pixel size from filenames
            resolution = (1.0, 1.0)
            try:
                for _root, _, files in os.walk(filepath):
                    for file in files:
                        if file.lower().endswith((".tif", ".tiff")):
                            match = re.search(r"--PX(\d+)", file)
                            if match:
                                pixel_size = float(match.group(1)) * 1e-4
                                resolution = (pixel_size, pixel_size)
                                break
                    if resolution != (1.0, 1.0):
                        break
            except (OSError, ValueError, TypeError):
                pass

            return {
                "axes": axes,
                "resolution": resolution,
                "unit": "um",
                "filepath": filepath,
            }
        except (FileFormatError, AttributeError, KeyError):
            return {}


class ScanFolderWorker(QThread):
    """Worker thread for scanning folders"""

    progress = Signal(int, int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, folder: str, filters: List[str]):
        super().__init__()
        self.folder = folder
        self.filters = filters

    def run(self):
        try:
            found_files = []
            all_items = []

            include_acquifer = "acquifer" in [f.lower() for f in self.filters]

            # Collect files and directories
            for root, dirs, files in os.walk(self.folder):
                # Add matching files
                for file in files:
                    if any(
                        file.lower().endswith(f)
                        for f in self.filters
                        if f.lower() != "acquifer"
                    ):
                        all_items.append(os.path.join(root, file))

                # Add Acquifer directories
                if include_acquifer:
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if AcquiferLoader.can_load(dir_path):
                            all_items.append(dir_path)

            # Process items
            total_items = len(all_items)
            for i, item_path in enumerate(all_items):
                if i % 10 == 0:
                    self.progress.emit(i, total_items)
                found_files.append(item_path)

            self.finished.emit(found_files)

        except (OSError, PermissionError) as e:
            self.error.emit(f"Scan failed: {str(e)}")


class ConversionWorker(QThread):
    """Enhanced worker thread for file conversion"""

    progress = Signal(int, int, str)
    file_done = Signal(str, bool, str)
    finished = Signal(int)

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

            filename = Path(filepath).name
            self.progress.emit(i + 1, len(self.files_to_convert), filename)

            try:
                # Load and convert file
                success = self._convert_single_file(filepath, series_index)
                if success:
                    success_count += 1
                    self.file_done.emit(
                        filepath, True, "Conversion successful"
                    )
                else:
                    self.file_done.emit(filepath, False, "Conversion failed")

            except (
                FileFormatError,
                SeriesIndexError,
                ConversionError,
                MemoryError,
            ) as e:
                self.file_done.emit(filepath, False, str(e))
            except (OSError, PermissionError) as e:
                self.file_done.emit(
                    filepath, False, f"File access error: {str(e)}"
                )

        self.finished.emit(success_count)

    def stop(self):
        self.running = False

    def _convert_single_file(self, filepath: str, series_index: int) -> bool:
        """Convert a single file to the target format"""
        image_data = None
        try:
            # Get loader and load data
            loader = self.get_file_loader(filepath)
            if not loader:
                raise FileFormatError("Unsupported file format")

            image_data = loader.load_series(filepath, series_index)
            metadata = loader.get_metadata(filepath, series_index) or {}

            # Generate output path
            base_name = Path(filepath).stem
            if self.use_zarr:
                output_path = os.path.join(
                    self.output_folder,
                    f"{base_name}_series{series_index}.zarr",
                )
                result = self._save_zarr(
                    image_data, output_path, metadata, base_name, series_index
                )
            else:
                output_path = os.path.join(
                    self.output_folder, f"{base_name}_series{series_index}.tif"
                )
                result = self._save_tif(image_data, output_path, metadata)

            return result

        except (FileFormatError, SeriesIndexError, MemoryError) as e:
            raise ConversionError(f"Conversion failed: {str(e)}") from e
        finally:
            # Free up memory after conversion
            if image_data is not None:
                del image_data
            import gc

            gc.collect()

    def _save_tif(
        self,
        image_data: Union[np.ndarray, da.Array],
        output_path: str,
        metadata: dict,
    ) -> bool:
        """Save image data as TIF with memory-efficient handling"""
        try:
            # Estimate file size
            if hasattr(image_data, "nbytes"):
                size_gb = image_data.nbytes / (1024**3)
            else:
                size_gb = (
                    np.prod(image_data.shape)
                    * getattr(image_data, "itemsize", 8)
                ) / (1024**3)

            print(
                f"Saving TIF: {output_path}, estimated size: {size_gb:.2f}GB"
            )

            # For very large files, reject TIF format
            if size_gb > 8:
                raise MemoryError(
                    "File too large for TIF format. Use ZARR instead."
                )

            use_bigtiff = size_gb > 4

            # Handle Dask arrays efficiently
            if hasattr(image_data, "dask"):
                if size_gb > 6:  # Conservative threshold for Dask->TIF
                    raise MemoryError(
                        "Dask array too large for TIF. Use ZARR instead."
                    )

                # For large Dask arrays, use chunked writing
                if len(image_data.shape) > 3:
                    return self._save_tif_chunked_dask(
                        image_data, output_path, use_bigtiff
                    )
                else:
                    # Compute smaller arrays
                    image_data = image_data.compute()

            # Standard TIF saving
            save_kwargs = {"bigtiff": use_bigtiff, "compression": "zlib"}

            if len(image_data.shape) > 2:
                save_kwargs["imagej"] = True

            if metadata.get("resolution"):
                try:
                    res_x, res_y = metadata["resolution"]
                    save_kwargs["resolution"] = (float(res_x), float(res_y))
                except (ValueError, TypeError):
                    pass

            tifffile.imwrite(output_path, image_data, **save_kwargs)
            return os.path.exists(output_path)

        except (OSError, PermissionError) as e:
            raise ConversionError(f"TIF save failed: {str(e)}") from e

    def _save_tif_chunked_dask(
        self, dask_array: da.Array, output_path: str, use_bigtiff: bool
    ) -> bool:
        """Save large Dask array to TIF using chunked writing"""
        try:
            print(
                f"Using chunked Dask TIF writing for shape {dask_array.shape}"
            )

            # Write timepoints/slices individually for multi-dimensional data
            if len(dask_array.shape) >= 4:
                with tifffile.TiffWriter(
                    output_path, bigtiff=use_bigtiff
                ) as writer:
                    for i in range(dask_array.shape[0]):
                        slice_data = dask_array[i].compute()
                        writer.write(slice_data, compression="zlib")
            else:
                # For 3D or smaller, compute and save normally
                computed_data = dask_array.compute()
                tifffile.imwrite(
                    output_path,
                    computed_data,
                    bigtiff=use_bigtiff,
                    compression="zlib",
                )

            return True

        except (OSError, PermissionError, MemoryError) as e:
            raise ConversionError(
                f"Chunked TIF writing failed: {str(e)}"
            ) from e
        finally:
            # Clean up temporary data
            if "slice_data" in locals():
                del slice_data
            if "computed_data" in locals():
                del computed_data

    def _save_zarr(
        self,
        image_data: Union[np.ndarray, da.Array],
        output_path: str,
        metadata: dict,
        base_name: str,
        series_index: int,
    ) -> bool:
        """Save image data as ZARR with proper OME-ZARR structure for napari-ome-zarr"""
        try:
            print(f"Saving ZARR: {output_path}")

            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            store = parse_url(output_path, mode="w").store

            # Convert to Dask array if needed
            if not hasattr(image_data, "dask"):
                image_data = da.from_array(image_data, chunks="auto")

            # Handle axes reordering for proper OME-ZARR structure
            axes = metadata.get("axes", "").lower()
            if axes:
                ndim = len(image_data.shape)
                has_time = "t" in axes
                target_axes = "tczyx" if has_time else "czyx"
                target_axes = target_axes[:ndim]

                if axes != target_axes and len(axes) == ndim:
                    try:
                        reorder_indices = [
                            axes.index(ax) for ax in target_axes if ax in axes
                        ]
                        if len(reorder_indices) == len(axes):
                            image_data = image_data.transpose(reorder_indices)
                            axes = target_axes
                    except (ValueError, IndexError):
                        pass

            # Create proper layer name for napari
            layer_name = (
                f"{base_name}_series_{series_index}"
                if series_index > 0
                else base_name
            )

            # Save with OME-ZARR - let napari-ome-zarr handle colormaps
            with ProgressBar():
                root = zarr.group(store=store)

                # Set minimal metadata - let napari-ome-zarr reader handle the rest
                root.attrs["name"] = layer_name

                # Write the image with proper OME-ZARR structure
                write_image(
                    image_data,
                    group=root,
                    axes=axes or "zyx",
                    scaler=None,
                    storage_options={"compression": "zstd"},
                )

            return True

        except (OSError, PermissionError, ImportError) as e:
            raise ConversionError(f"ZARR save failed: {str(e)}") from e
        finally:
            # Force cleanup of any large intermediate arrays
            import gc

            gc.collect()


class MicroscopyImageConverterWidget(QWidget):
    """Enhanced main widget for microscopy image conversion"""

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

        # Conversion state
        self.selected_series = {}
        self.export_all_series = {}
        self.scan_worker = None
        self.conversion_worker = None
        self.updating_format_buttons = False

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Input folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Input Folder:"))
        self.folder_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_folder)

        folder_layout.addWidget(self.folder_edit)
        folder_layout.addWidget(browse_button)
        main_layout.addLayout(folder_layout)

        # File filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("File Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            ".lif, .nd2, .ndpi, .czi, acquifer"
        )
        self.filter_edit.setText(".lif,.nd2,.ndpi,.czi,acquifer")
        scan_button = QPushButton("Scan Folder")
        scan_button.clicked.connect(self.scan_folder)

        filter_layout.addWidget(self.filter_edit)
        filter_layout.addWidget(scan_button)
        main_layout.addLayout(filter_layout)

        # Progress bars
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        main_layout.addWidget(self.scan_progress)

        # Tables layout
        tables_layout = QHBoxLayout()
        self.files_table = SeriesTableWidget(self.viewer)
        self.series_widget = SeriesDetailWidget(self, self.viewer)
        tables_layout.addWidget(self.files_table)
        tables_layout.addWidget(self.series_widget)
        main_layout.addLayout(tables_layout)

        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.tif_radio = QCheckBox("TIF")
        self.tif_radio.setChecked(True)
        self.zarr_radio = QCheckBox("ZARR (Recommended for >4GB)")

        self.tif_radio.toggled.connect(self.handle_format_toggle)
        self.zarr_radio.toggled.connect(self.handle_format_toggle)

        format_layout.addWidget(self.tif_radio)
        format_layout.addWidget(self.zarr_radio)
        main_layout.addLayout(format_layout)

        # Output folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_edit = QLineEdit()
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self.browse_output)

        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        main_layout.addLayout(output_layout)

        # Conversion progress
        self.conversion_progress = QProgressBar()
        self.conversion_progress.setVisible(False)
        main_layout.addWidget(self.conversion_progress)

        # Control buttons
        button_layout = QHBoxLayout()
        convert_button = QPushButton("Convert Selected Files")
        convert_button.clicked.connect(self.convert_files)
        convert_all_button = QPushButton("Convert All Files")
        convert_all_button.clicked.connect(self.convert_all_files)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.setVisible(False)

        button_layout.addWidget(convert_button)
        button_layout.addWidget(convert_all_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        # Status
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

    def browse_folder(self):
        """Browse for input folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.folder_edit.setText(folder)

    def browse_output(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def scan_folder(self):
        """Scan folder for image files"""
        folder = self.folder_edit.text()
        if not folder or not os.path.isdir(folder):
            self.status_label.setText("Please select a valid folder")
            return

        filters = [
            f.strip() for f in self.filter_edit.text().split(",") if f.strip()
        ]
        if not filters:
            filters = [".lif", ".nd2", ".ndpi", ".czi"]

        # Clear existing data and force garbage collection
        self.files_table.setRowCount(0)
        self.files_table.file_data.clear()

        # Clear any cached datasets
        AcquiferLoader._dataset_cache.clear()

        # Force memory cleanup before starting scan
        import gc

        gc.collect()

        # Start scan worker
        self.scan_worker = ScanFolderWorker(folder, filters)
        self.scan_worker.progress.connect(self.update_scan_progress)
        self.scan_worker.finished.connect(self.process_found_files)
        self.scan_worker.error.connect(self.show_error)

        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        self.cancel_button.setVisible(True)
        self.status_label.setText("Scanning folder...")
        self.scan_worker.start()

    def update_scan_progress(self, current: int, total: int):
        """Update scan progress"""
        if total > 0:
            self.scan_progress.setValue(int(current * 100 / total))

    def process_found_files(self, found_files: List[str]):
        """Process found files and add to table"""
        self.scan_progress.setVisible(False)
        self.cancel_button.setVisible(False)

        with concurrent.futures.ThreadPoolExecutor() as executor:
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

            for i, future in enumerate(
                concurrent.futures.as_completed(futures)
            ):
                filepath, file_type = futures[future]
                try:
                    series_count = future.result()
                    self.files_table.add_file(
                        filepath, file_type, series_count
                    )
                except (OSError, FileFormatError, ValueError) as e:
                    print(f"Error processing {filepath}: {e}")
                    self.files_table.add_file(filepath, file_type, 0)

                # Update status periodically
                if i % 5 == 0:
                    self.status_label.setText(
                        f"Processed {i+1}/{len(futures)} files..."
                    )
                    QApplication.processEvents()

        self.status_label.setText(f"Found {len(found_files)} files")

    def show_error(self, error_message: str):
        """Show error message"""
        self.status_label.setText(f"Error: {error_message}")
        self.scan_progress.setVisible(False)
        self.cancel_button.setVisible(False)
        QMessageBox.critical(self, "Error", error_message)

    def cancel_operation(self):
        """Cancel current operation"""
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.terminate()
            self.scan_worker.deleteLater()
            self.scan_worker = None

        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.stop()
            self.conversion_worker.deleteLater()
            self.conversion_worker = None

        # Force memory cleanup after cancellation
        import gc

        gc.collect()

        self.scan_progress.setVisible(False)
        self.conversion_progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.status_label.setText("Operation cancelled")

    def get_file_type(self, filepath: str) -> str:
        """Determine file type"""
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
        """Get appropriate loader for file"""
        for loader in self.loaders:
            if loader.can_load(filepath):
                return loader
        return None

    def show_series_details(self, filepath: str):
        """Show series details"""
        self.series_widget.set_file(filepath)

    def set_selected_series(self, filepath: str, series_index: int):
        """Set selected series for file"""
        self.selected_series[filepath] = series_index

    def set_export_all_series(self, filepath: str, export_all: bool):
        """Set export all series flag"""
        self.export_all_series[filepath] = export_all
        if export_all and filepath not in self.selected_series:
            self.selected_series[filepath] = 0

    def load_image(self, filepath: str):
        """Load image into viewer"""
        try:
            loader = self.get_file_loader(filepath)
            if not loader:
                raise FileFormatError("Unsupported file format")

            image_data = loader.load_series(filepath, 0)
            self.viewer.layers.clear()
            layer_name = f"{Path(filepath).stem}"
            self.viewer.add_image(image_data, name=layer_name)
            self.viewer.status = f"Loaded {Path(filepath).name}"

        except (OSError, FileFormatError, MemoryError) as e:
            error_msg = f"Error loading image: {str(e)}"
            self.viewer.status = error_msg
            QMessageBox.warning(self, "Load Error", error_msg)

    def update_format_buttons(self, use_zarr: bool = False):
        """Update format buttons based on file size"""
        if self.updating_format_buttons:
            return

        self.updating_format_buttons = True
        try:
            if use_zarr:
                self.zarr_radio.setChecked(True)
                self.tif_radio.setChecked(False)
                self.status_label.setText(
                    "Auto-selected ZARR format for large file (>4GB)"
                )
            else:
                self.tif_radio.setChecked(True)
                self.zarr_radio.setChecked(False)
        finally:
            self.updating_format_buttons = False

    def handle_format_toggle(self, checked: bool):
        """Handle format toggle"""
        if self.updating_format_buttons:
            return

        self.updating_format_buttons = True
        try:
            sender = self.sender()
            if sender == self.tif_radio and checked:
                self.zarr_radio.setChecked(False)
            elif sender == self.zarr_radio and checked:
                self.tif_radio.setChecked(False)
        finally:
            self.updating_format_buttons = False

    def convert_files(self):
        """Convert selected files"""
        try:
            # Prepare conversion list
            if not self.selected_series:
                all_files = list(self.files_table.file_data.keys())
                if not all_files:
                    self.status_label.setText(
                        "No files available for conversion"
                    )
                    return
                for filepath in all_files:
                    self.selected_series[filepath] = 0

            # Validate output folder
            output_folder = self.output_edit.text()
            if not output_folder:
                output_folder = os.path.join(
                    self.folder_edit.text(), "converted"
                )

            if not self._validate_output_folder(output_folder):
                return

            # Build conversion list
            files_to_convert = []
            for filepath, series_index in self.selected_series.items():
                if self.export_all_series.get(filepath, False):
                    loader = self.get_file_loader(filepath)
                    if loader:
                        try:
                            series_count = loader.get_series_count(filepath)
                            for i in range(series_count):
                                files_to_convert.append((filepath, i))
                        except (OSError, FileFormatError, ValueError) as e:
                            self.status_label.setText(
                                f"Error getting series count: {str(e)}"
                            )
                            return
                else:
                    files_to_convert.append((filepath, series_index))

            if not files_to_convert:
                self.status_label.setText("No valid files to convert")
                return

            # Start conversion
            self.conversion_worker = ConversionWorker(
                files_to_convert=files_to_convert,
                output_folder=output_folder,
                use_zarr=self.zarr_radio.isChecked(),
                file_loader_func=self.get_file_loader,
            )

            self.conversion_worker.progress.connect(
                self.update_conversion_progress
            )
            self.conversion_worker.file_done.connect(
                self.handle_conversion_result
            )
            self.conversion_worker.finished.connect(self.conversion_completed)

            self.conversion_progress.setVisible(True)
            self.conversion_progress.setValue(0)
            self.cancel_button.setVisible(True)
            self.status_label.setText(
                f"Converting {len(files_to_convert)} files/series..."
            )

            self.conversion_worker.start()

        except (OSError, PermissionError, ValueError) as e:
            QMessageBox.critical(
                self,
                "Conversion Error",
                f"Failed to start conversion: {str(e)}",
            )

    def convert_all_files(self):
        """Convert all files with default settings"""
        self.selected_series.clear()
        self.export_all_series.clear()

        all_files = list(self.files_table.file_data.keys())
        if not all_files:
            self.status_label.setText("No files available for conversion")
            return

        for filepath in all_files:
            self.selected_series[filepath] = 0
            file_info = self.files_table.file_data.get(filepath)
            if file_info and file_info.get("series_count", 0) > 1:
                self.export_all_series[filepath] = True

        self.convert_files()

    def _validate_output_folder(self, folder: str) -> bool:
        """Validate output folder"""
        if not folder:
            self.status_label.setText("Please specify an output folder")
            return False

        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except (OSError, PermissionError) as e:
                self.status_label.setText(
                    f"Cannot create output folder: {str(e)}"
                )
                return False

        if not os.access(folder, os.W_OK):
            self.status_label.setText("Output folder is not writable")
            return False

        return True

    def update_conversion_progress(
        self, current: int, total: int, filename: str
    ):
        """Update conversion progress"""
        if total > 0:
            self.conversion_progress.setValue(int(current * 100 / total))
            self.status_label.setText(
                f"Converting {filename} ({current}/{total})..."
            )

    def handle_conversion_result(
        self, filepath: str, success: bool, message: str
    ):
        """Handle single file conversion result"""
        filename = Path(filepath).name
        if success:
            print(f"Successfully converted: {filename}")
        else:
            print(f"Failed to convert: {filename} - {message}")
            QMessageBox.warning(
                self,
                "Conversion Warning",
                f"Error converting {filename}: {message}",
            )

    def conversion_completed(self, success_count: int):
        """Handle conversion completion"""
        self.conversion_progress.setVisible(False)
        self.cancel_button.setVisible(False)

        # Clean up conversion worker
        if self.conversion_worker:
            self.conversion_worker.deleteLater()
            self.conversion_worker = None

        # Force memory cleanup
        import gc

        gc.collect()

        output_folder = self.output_edit.text()
        if not output_folder:
            output_folder = os.path.join(self.folder_edit.text(), "converted")

        if success_count > 0:
            self.status_label.setText(
                f"Successfully converted {success_count} files to {output_folder}"
            )
        else:
            self.status_label.setText("No files were converted")


@magicgui(call_button="Start Microscopy Image Converter", layout="vertical")
def microscopy_converter(viewer: napari.Viewer):
    """Start the enhanced microscopy image converter tool"""
    converter_widget = MicroscopyImageConverterWidget(viewer)
    viewer.window.add_dock_widget(
        converter_widget, name="Microscopy Image Converter", area="right"
    )
    return converter_widget


def napari_experimental_provide_dock_widget():
    """Provide the converter widget to Napari"""
    return microscopy_converter

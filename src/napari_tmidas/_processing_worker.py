"""
Processing worker for batch image processing.
"""

import concurrent.futures
import os
from typing import Any, List, Union

import numpy as np

try:
    import dask.array as da

    _HAS_DASK = True
except ImportError:
    da = None
    _HAS_DASK = False

# Lazy imports for optional heavy dependencies
try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False

try:
    from qtpy.QtCore import QThread, Signal

    _HAS_QTPY = True
except ImportError:
    # Create stubs to allow class definitions
    class QThread:
        def __init__(self):
            pass

        def run(self):
            pass

    def Signal(*args):
        return None

    _HAS_QTPY = False


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
        self.thread_count = max(1, (os.cpu_count() or 4) // 4)  # Default value

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

            # Handle multi-layer data from OME-Zarr
            is_multi_layer = isinstance(image_data, list)
            if is_multi_layer:
                print(
                    f"Processing multi-layer file ({len(image_data)} layers): {filepath}"
                )
                # Take the first image layer as default
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

            # Check if this is a folder-processing function that shouldn't save individual files
            function_name = getattr(
                self.processing_func, "__name__", "unknown"
            )
            is_folder_function = function_name in [
                "merge_timepoints",
                "track_objects",
                "create_grid_overlay",  # Grid overlay processes all files at once
            ]

            # Only print verbose output for non-folder functions
            if not is_folder_function:
                print(
                    f"Original image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}, dtype: {image_dtype}"
                )

            # Handle channel selection if specified in parameters
            channel_param = self.param_values.get("channel", None) if self.param_values else None
            images_to_process = []
            channel_indices = []
            
            if channel_param is not None:
                try:
                    from napari_tmidas._file_selector import (
                        detect_channels_in_image,
                        detect_channels_from_zarr_path,
                    )
                    # For zarr files use metadata-based detection (same logic as the UI)
                    if filepath.lower().endswith(".zarr") or (
                        os.path.isdir(filepath) and
                        os.path.exists(os.path.join(filepath, ".zattrs"))
                    ):
                        num_channels, channel_axis = detect_channels_from_zarr_path(filepath)
                    else:
                        num_channels, channel_axis = detect_channels_in_image(image_data)
                    
                    if num_channels > 1:
                        if channel_axis == -1:
                            # Channels are in separate layers (multi-layer OME-Zarr)
                            print(f"Detected {num_channels} channels in separate layers")
                            
                            if channel_param == "all":
                                # Process all layers
                                print(f"Processing all {num_channels} layers")
                                for i, (data, _kwargs, layer_type) in enumerate(image_data):
                                    if layer_type == "image":
                                        images_to_process.append(data)
                                        channel_indices.append(i)
                            elif isinstance(channel_param, int) and 0 <= channel_param < num_channels:
                                # Process only the selected layer
                                print(f"Processing layer {channel_param}")
                                image_idx = 0
                                for data, _kwargs, layer_type in image_data:
                                    if layer_type == "image":
                                        if image_idx == channel_param:
                                            images_to_process.append(data)
                                            channel_indices.append(channel_param)
                                            break
                                        image_idx += 1
                            else:
                                # Invalid channel selection, process first layer
                                print(f"Invalid channel selection: {channel_param}, processing first layer")
                                images_to_process.append(image)
                                channel_indices.append(None)
                        elif channel_axis is not None:
                            # Channels are in a dimension within the array
                            if channel_param == "all":
                                # Process all channels separately
                                print(f"Processing all {num_channels} channels separately")
                                for i in range(num_channels):
                                    channel_img = np.take(image, i, axis=channel_axis)
                                    images_to_process.append(channel_img)
                                    channel_indices.append(i)
                            elif isinstance(channel_param, int) and 0 <= channel_param < num_channels:
                                # Process only the selected channel
                                print(f"Processing channel {channel_param}")
                                channel_img = np.take(image, channel_param, axis=channel_axis)
                                images_to_process.append(channel_img)
                                channel_indices.append(channel_param)
                            else:
                                # Invalid channel selection, process entire image
                                print(f"Invalid channel selection: {channel_param}, processing entire image")
                                images_to_process.append(image)
                                channel_indices.append(None)
                        else:
                            # Single channel image, process normally
                            images_to_process.append(image)
                            channel_indices.append(None)
                    else:
                        # Single channel image, process normally
                        images_to_process.append(image)
                        channel_indices.append(None)
                except ImportError:
                    # Fallback if detection function not available
                    images_to_process.append(image)
                    channel_indices.append(None)
            else:
                # No channel parameter specified, process entire image
                images_to_process.append(image)
                channel_indices.append(None)

            # Process each image in the list
            all_processed_images = []
            for img_idx, (img_to_process, ch_idx) in enumerate(zip(images_to_process, channel_indices)):
                # Apply the processing function with parameters
                if self.param_values:
                    # Remove channel parameter before passing to processing function
                    proc_params = {k: v for k, v in self.param_values.items() if k != "channel"}
                    processed_image = self.processing_func(img_to_process, **proc_params)
                else:
                    processed_image = self.processing_func(img_to_process)
                
                all_processed_images.append((processed_image, ch_idx))

            # Handle saving based on number of processed images
            if len(all_processed_images) > 1:
                # Multiple channels were processed separately
                processed_files = []
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                
                for processed_image, ch_idx in all_processed_images:
                    if processed_image is None or is_folder_function:
                        continue
                    
                    # Generate output filename with channel suffix
                    output_filename = f"{base_name}_ch{ch_idx}{self.output_suffix}"
                    output_path = os.path.join(self.output_folder, output_filename)
                    
                    # Save the processed image
                    save_image_file(processed_image, output_path, image_dtype)
                    processed_files.append(output_path)
                
                return {
                    "original_file": filepath,
                    "processed_files": processed_files,
                }
            else:
                # Single image processed (or single channel)
                processed_image, ch_idx = all_processed_images[0]

            # Handle functions that return multiple outputs (e.g., channel splitting)
            if (
                isinstance(processed_image, (list, tuple))
                and len(processed_image) > 1
            ):
                # Multiple outputs - save each as separate file
                processed_files = []
                base_name = os.path.splitext(os.path.basename(filepath))[0]

                # Check if this is a layer subdivision function (returns 3 outputs)
                if (
                    len(processed_image) == 3
                    and self.output_suffix == "_layer"
                ):
                    layer_names = [
                        "_inner",
                        "_middle",
                        "_outer",
                    ]
                    for img, layer_name in zip(processed_image, layer_names):
                        if not isinstance(img, np.ndarray):
                            continue

                        # Generate output filename with layer name
                        output_filename = f"{base_name}{layer_name}.tif"
                        output_path = os.path.join(
                            self.output_folder, output_filename
                        )

                        # Save as uint32 to ensure Napari auto-detects as labels
                        save_image_file(img, output_path, np.uint32)
                        processed_files.append(output_path)
                else:
                    # Default behavior for other multi-output functions (e.g., channel splitting)
                    for idx, img in enumerate(processed_image):
                        if not isinstance(img, np.ndarray):
                            continue

                        # Generate output filename
                        output_filename = (
                            f"{base_name}_ch{idx + 1}{self.output_suffix}"
                        )
                        output_path = os.path.join(
                            self.output_folder, output_filename
                        )

                        # Save the processed image
                        save_image_file(img, output_path, image_dtype)
                        processed_files.append(output_path)

                return {
                    "original_file": filepath,
                    "processed_files": processed_files,
                }

            elif processed_image is not None and not is_folder_function:
                # Single output - save as single file
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                output_filename = f"{base_name}{self.output_suffix}"
                output_path = os.path.join(self.output_folder, output_filename)

                # Save the processed image
                save_image_file(processed_image, output_path, image_dtype)

                return {
                    "original_file": filepath,
                    "processed_file": output_path,
                }

            else:
                # Folder function or no output to save
                return {
                    "original_file": filepath,
                    "processed_file": None,
                }

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise


def load_image_file(filepath: str) -> Union[np.ndarray, List, Any]:
    """
    Load image from file, supporting both TIFF and Zarr formats with proper metadata handling
    """
    # Import the actual implementation from _file_selector
    try:
        from napari_tmidas._file_selector import load_image_file as _load_impl
        return _load_impl(filepath)
    except ImportError:
        # Fallback implementation
        if filepath.lower().endswith(".zarr"):
            try:
                import zarr
                root = zarr.open(filepath, mode="r")
                if hasattr(root, "arrays"):
                    arrays_list = list(root.arrays())
                    if arrays_list:
                        return np.array(arrays_list[0][1])
                return np.array(root)
            except Exception:
                pass
        
        # Try tifffile
        if _HAS_TIFFFILE:
            return tifffile.imread(filepath)
        
        # Last resort: numpy
        return np.load(filepath)


def is_label_image(image: np.ndarray) -> bool:
    """
    Determine if an image should be treated as a label image based on its dtype.

    This function uses the same logic as Napari's guess_labels() function,
    checking if the dtype is one of the integer types commonly used for labels.
    """
    if hasattr(image, "dtype"):
        return image.dtype in (np.int32, np.uint32, np.int64, np.uint64)
    return False


def save_image_file(image: np.ndarray, filepath: str, dtype=None):
    """
    Save image to file with proper format detection.

    Label images are saved as uint32 to ensure napari recognizes them as labels.
    Napari automatically detects int32/uint32/int64/uint64 dtypes as labels.
    """
    if not _HAS_TIFFFILE:
        raise ImportError("tifffile is required to save images")

    def _is_dask_array(obj: Any) -> bool:
        return _HAS_DASK and isinstance(obj, da.Array)

    def _estimate_size_gb(obj: Any) -> float:
        try:
            shape = tuple(int(x) for x in obj.shape)
            itemsize = np.dtype(obj.dtype).itemsize
            total = int(np.prod(shape, dtype=np.int64))
            return (total * itemsize) / (1024**3)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    # Calculate approx file size in GB
    size_gb = _estimate_size_gb(image)

    # For very large files, use BigTIFF format
    use_bigtiff = size_gb > 2.0

    # Determine save dtype
    if dtype is not None:
        # Use explicitly provided dtype
        save_dtype = dtype
    elif is_label_image(image):
        # Input is already a label dtype, preserve as uint32
        # uint32 is the standard for label images and is automatically
        # recognized by napari
        save_dtype = np.uint32
    else:
        # Use image's dtype
        save_dtype = image.dtype

    # Stream dask arrays per leading-axis slab to avoid full materialization.
    if _is_dask_array(image):
        if image.ndim <= 2:
            arr = np.asarray(image.compute(), dtype=save_dtype)
            tifffile.imwrite(
                filepath,
                arr,
                compression="zlib",
                bigtiff=use_bigtiff,
            )
            return

        expected_shape = tuple(int(x) for x in image.shape)

        def _iter_slabs():
            for i in range(expected_shape[0]):
                yield np.asarray(image[i].compute(), dtype=save_dtype)

        tifffile.imwrite(
            filepath,
            data=_iter_slabs(),
            shape=expected_shape,
            dtype=save_dtype,
            compression="zlib",
            photometric="minisblack",
            bigtiff=use_bigtiff,
        )
        return

    tifffile.imwrite(
        filepath,
        np.asarray(image, dtype=save_dtype),
        compression="zlib",
        bigtiff=use_bigtiff,
    )

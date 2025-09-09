"""
Processing worker for batch image processing.
"""

import concurrent.futures
import os
from typing import Any, List, Union

import numpy as np
import tifffile
from qtpy.QtCore import QThread, Signal


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
            is_folder_function = function_name in [
                "merge_timepoints",
                "track_objects",
            ]

            # Apply the processing function with parameters
            if self.param_values:
                processed_image = self.processing_func(
                    image, **self.param_values
                )
            else:
                processed_image = self.processing_func(image)

            # Handle functions that return multiple outputs (e.g., channel splitting)
            if (
                isinstance(processed_image, (list, tuple))
                and len(processed_image) > 1
            ):
                # Multiple outputs - save each as separate file
                processed_files = []
                base_name = os.path.splitext(os.path.basename(filepath))[0]

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
    # This is a placeholder - the actual implementation would be moved from _file_selector.py
    # For now, return a dummy array
    return np.random.rand(100, 100)


def save_image_file(image: np.ndarray, filepath: str, dtype=None):
    """
    Save image to file with proper format detection
    """
    # This is a placeholder - the actual implementation would be moved from _file_selector.py

# processing_functions/file_compression.py
"""
Processing functions for compressing files using pzstd.

This module provides a function to compress files using the Zstandard compression algorithm
via the pzstd tool. It compresses image files after they have been processed by other functions
in the batch processing pipeline.

Note: This requires the pzstd tool to be installed on the system.
"""
import subprocess

import numpy as np

from napari_tmidas._file_selector import ProcessingWorker
from napari_tmidas._registry import BatchProcessingRegistry


def check_pzstd_installed():
    """Check if pzstd is installed on the system."""
    try:
        subprocess.run(["pzstd", "--version"], capture_output=True, text=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def compress_file(file_path, remove_source=False, compression_level=3):
    """
    Compress a file using pzstd.

    Parameters:
    -----------
    file_path : str
        Path to the file to compress
    remove_source : bool
        Whether to remove the source file after compression
    compression_level : int
        Compression level (1-22)

    Returns:
    --------
    tuple
        (success, compressed_file_path)
    """
    compressed_file = f"{file_path}.zst"
    command = ["pzstd", "--quiet"]

    # Set compression level
    if compression_level >= 20:
        command.extend(["--ultra", f"-{compression_level}"])
    else:
        command.append(f"-{compression_level}")

    # Remove source if requested
    if remove_source:
        command.append("--rm")

    command.append(file_path)

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode == 0, compressed_file
    except (subprocess.SubprocessError, FileNotFoundError):
        return False, None


@BatchProcessingRegistry.register(
    name="Compress with Zstandard",
    suffix="_compressed",
    description="Compress the processed image file using Zstandard (requires pzstd to be installed)",
    parameters={
        "remove_source": {
            "type": bool,
            "default": False,
            "description": "Remove the source file after compression",
        },
        "compression_level": {
            "type": int,
            "default": 3,
            "min": 1,
            "max": 22,
            "description": "Compression level (1-22, higher = better compression but slower)",
        },
    },
)
def compress_with_zstandard(
    image: np.ndarray, remove_source: bool = False, compression_level: int = 3
) -> np.ndarray:
    """
    Process an image and compress the output file using Zstandard.

    This function:
    1. Takes an image array as input
    2. Returns the original image unchanged (compression happens to the saved file)
    3. The batch processing system saves the file
    4. This function then compresses the saved file using pzstd

    Parameters:
    -----------
    image : numpy.ndarray
        Input image array
    remove_source : bool
        Whether to remove the source file after compression (default: False)
    compression_level : int
        Compression level (1-22) (default: 3)

    Returns:
    --------
    numpy.ndarray
        The original image (unchanged)
    """
    # Check if pzstd is installed
    if not check_pzstd_installed():
        print("Warning: pzstd is not installed. Compression will be skipped.")
        return image

    # Instead of trying to modify the array, set attributes on the processing function itself
    compress_with_zstandard.compress_after_save = True
    compress_with_zstandard.remove_source = remove_source
    compress_with_zstandard.compression_level = compression_level

    # Return the image unchanged - compression happens after saving
    return image


# Monkey patch the batch processing system to compress files after saving
# This is a bit of a hack, but it allows us to compress files after they've been saved
# by the batch processing system

# Store the original save_file function


original_process_file = ProcessingWorker.process_file


# Replace it with our modified version that compresses after saving
def process_file_with_compression(self, filepath):
    """Modified process_file function that compresses files after saving."""
    result = original_process_file(self, filepath)

    # Check if there's a result and if we should compress it
    if isinstance(result, dict):
        # Single output file
        if "processed_file" in result:
            output_path = result["processed_file"]
            # Check if the processed image had compression metadata
            if (
                hasattr(self.processing_func, "compress_after_save")
                and self.processing_func.compress_after_save
            ):
                # Get compression parameters
                remove_source = getattr(
                    self.processing_func, "remove_source", False
                )
                compression_level = getattr(
                    self.processing_func, "compression_level", 3
                )

                # Compress the file
                success, compressed_path = compress_file(
                    output_path, remove_source, compression_level
                )

                if success:
                    # Update the result with the compressed file path
                    result["processed_file"] = compressed_path

        # Multiple output files
        elif "processed_files" in result:
            output_paths = result["processed_files"]
            # Check if the processed image had compression metadata
            if (
                hasattr(self.processing_func, "compress_after_save")
                and self.processing_func.compress_after_save
            ):
                # Get compression parameters
                remove_source = getattr(
                    self.processing_func, "remove_source", False
                )
                compression_level = getattr(
                    self.processing_func, "compression_level", 3
                )

                # Compress each file
                compressed_paths = []
                for output_path in output_paths:
                    success, compressed_path = compress_file(
                        output_path, remove_source, compression_level
                    )

                    if success:
                        compressed_paths.append(compressed_path)
                    else:
                        compressed_paths.append(output_path)

                # Update the result with the compressed file paths
                result["processed_files"] = compressed_paths

    return result


# Apply the monkey patch if pzstd is available
if check_pzstd_installed():
    ProcessingWorker.process_file = process_file_with_compression

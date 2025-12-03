#!/usr/bin/env python
"""
Test script to run grid overlay function directly on a folder.
"""
import sys
from pathlib import Path

import tifffile

# Import the grid overlay function
from napari_tmidas.processing_functions.grid_view_overlay import (
    create_grid_overlay,
    reset_grid_cache,
)


def test_grid_overlay(folder_path, output_folder=None):
    """
    Test the grid overlay function on a folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing label files
    output_folder : str, optional
        Path to output folder (if None, uses input folder)
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"ERROR: Folder does not exist: {folder}")
        return

    # Find label files or any .tif files
    label_files = list(folder.glob("*_labels*.tif")) + list(
        folder.glob("*_labels*.tiff")
    )

    if not label_files:
        print("No label files found, looking for any .tif files...")
        label_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))

    if not label_files:
        print(f"ERROR: No .tif files found in {folder}")
        return

    print(f"Found {len(label_files)} label files in {folder}")
    print(f"First file: {label_files[0]}")

    # Reset cache to ensure clean run
    reset_grid_cache()

    # Create a mock processing context
    class MockWorker:
        def __init__(self, output_folder):
            self.output_folder = output_folder

    # Set up the context that the function expects

    # We need to simulate the batch processing context
    # The function looks for 'filepath', 'file_list', and 'self.output_folder' in the call stack

    # Create the mock worker
    if output_folder is None:
        output_folder = str(folder)

    mock_worker = MockWorker(output_folder)
    filepath = str(label_files[0])
    file_list = [str(f) for f in label_files]

    print("\nTest configuration:")
    print(f"  Input folder: {folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Number of files: {len(file_list)}")
    print(f"  First filepath for stack context: {filepath}")

    # Load the first label image to pass as dummy input
    try:
        dummy_image = tifffile.imread(str(label_files[0]))
        print(f"  Dummy image shape: {dummy_image.shape}")
    except (OSError, ValueError, tifffile.TiffFileError) as e:
        print(f"ERROR loading dummy image: {e}")
        return

    # Call the function with the mock context
    print("\n" + "=" * 80)
    print("CALLING create_grid_overlay...")
    print("=" * 80 + "\n")

    try:

        def _run_with_context(worker, filepath, file_list, dummy):
            """Call processing while keeping expected variables in scope."""

            self = worker  # noqa: F841 - accessed through inspect
            _ = (self.output_folder, filepath, len(file_list))
            return create_grid_overlay(dummy, label_suffix="_labels.tif")

        result = _run_with_context(
            mock_worker, filepath, file_list, dummy_image
        )

        print("\n" + "=" * 80)
        print("FUNCTION COMPLETED")
        print("=" * 80)

        if result is not None:
            print(f"Result type: {type(result)}")
            print(
                f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}"
            )
        else:
            print("Result: None (expected for subsequent calls in batch)")

    except (OSError, ValueError, tifffile.TiffFileError, RuntimeError) as e:
        print(f"\nERROR during execution: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python test_grid_overlay.py <folder_path> [output_folder]"
        )
        print("\nExample:")
        print(
            "  python test_grid_overlay.py /mnt/disk2/Asli/Acquifer_quantification/converted"
        )
        print(
            "  python test_grid_overlay.py /mnt/disk2/Asli/Acquifer_quantification/converted /tmp/output"
        )
        sys.exit(1)

    folder_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    test_grid_overlay(folder_path, output_folder)

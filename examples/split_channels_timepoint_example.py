#!/usr/bin/env python3
"""
Example: Split Color Channels with Timepoint Sorting

This example demonstrates how to use the split_channels function with
automatic timepoint sorting for datasets with mixed timepoint counts.

Usage:
    python examples/split_channels_timepoint_example.py <input_folder> <output_folder>
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import napari_tmidas
sys.path.insert(0, str(Path(__file__).parent.parent))

from napari_tmidas.processing_functions.basic import (
    get_timepoint_count,
    sort_files_by_timepoints,
)


def demonstrate_timepoint_detection(input_folder):
    """
    Demonstrate timepoint detection for images in a folder

    Parameters
    ----------
    input_folder : str or Path
        Folder containing image files
    """
    print("\n" + "=" * 70)
    print("TIMEPOINT DETECTION DEMONSTRATION")
    print("=" * 70)

    input_folder = Path(input_folder)
    extensions = [".tif", ".tiff", ".TIF", ".TIFF"]

    image_files = []
    for ext in extensions:
        image_files.extend(input_folder.glob(f"*{ext}"))

    if not image_files:
        print(f"No TIFF files found in {input_folder}")
        return

    print(f"Found {len(image_files)} image files:\n")

    # Analyze each file
    timepoint_counts = {}
    for img_path in sorted(image_files):
        num_timepoints = get_timepoint_count(str(img_path))
        if num_timepoints:
            print(f"  {img_path.name:40s} -> {num_timepoints:3d} timepoints")
            timepoint_counts[num_timepoints] = (
                timepoint_counts.get(num_timepoints, 0) + 1
            )
        else:
            print(f"  {img_path.name:40s} -> Unable to determine")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for t_count, num_files in sorted(timepoint_counts.items()):
        print(f"  {num_files} file(s) with {t_count} timepoint(s)")
    print(f"{'='*70}\n")


def demonstrate_sorting(input_folder, output_folder):
    """
    Demonstrate sorting files by timepoint count

    Parameters
    ----------
    input_folder : str or Path
        Folder containing image files to sort
    output_folder : str or Path
        Output folder where timepoint subfolders will be created
    """
    print("\n" + "=" * 70)
    print("SORTING FILES BY TIMEPOINTS")
    print("=" * 70)

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    extensions = [".tif", ".tiff", ".TIF", ".TIFF"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_folder.glob(f"*{ext}"))

    if not image_files:
        print(f"No TIFF files found in {input_folder}")
        return

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Total files: {len(image_files)}\n")

    # Sort files into timepoint subfolders
    timepoint_map = sort_files_by_timepoints(
        [str(f) for f in image_files], str(output_folder)
    )

    print("\nFiles have been organized into timepoint subfolders:")
    for t_count in sorted(timepoint_map.keys()):
        subfolder = output_folder / f"T{t_count}"
        print(f"  {subfolder}: {len(timepoint_map[t_count])} files")


def demonstrate_split_with_sorting(
    input_folder, output_folder, num_channels=3
):
    """
    Demonstrate split_channels with automatic timepoint sorting

    This shows how the function would be used in batch processing context.
    Note: When using sort_by_timepoints, time_steps should be left at 0
    since each image may have a different timepoint count.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing image files
    output_folder : str or Path
        Output folder for processed files
    num_channels : int
        Number of color channels to split
    """
    print("\n" + "=" * 70)
    print("SPLIT CHANNELS WITH TIMEPOINT SORTING")
    print("=" * 70)

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    extensions = [".tif", ".tiff", ".TIF", ".TIFF"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_folder.glob(f"*{ext}"))

    if not image_files:
        print(f"No TIFF files found in {input_folder}")
        return

    print(f"Processing {len(image_files)} files")
    print(f"Number of channels: {num_channels}")
    print("Timepoint sorting: ENABLED")
    print(f"Output folder: {output_folder}\n")

    # Note: In actual batch processing, this would be called by the
    # ProcessingWorker for each file. Here we demonstrate the concept.

    # First, sort files by timepoints
    print("Step 1: Sorting files by timepoints...")
    timepoint_map = sort_files_by_timepoints(
        [str(f) for f in image_files], str(output_folder)
    )

    print("\nStep 2: Split channels would now process each file...")
    print("(In actual usage, this happens automatically in batch processing)")

    # Show what would happen
    for t_count, files in sorted(timepoint_map.items()):
        print(f"\nProcessing T{t_count} folder ({len(files)} files):")
        for filepath in files[:2]:  # Show first 2 as example
            filename = Path(filepath).name
            base_name = filename.rsplit(".", 1)[0]
            for ch in range(1, num_channels + 1):
                output_name = f"{base_name}_ch{ch}_split.tif"
                print(f"  - Would create: T{t_count}/{output_name}")
        if len(files) > 2:
            print(f"  - ... and {len(files) - 2} more files")


def main():
    """Main function to run examples"""
    if len(sys.argv) < 2:
        print(
            "Usage: python split_channels_timepoint_example.py <input_folder> [output_folder]"
        )
        print("\nThis script demonstrates:")
        print("  1. Detecting timepoint counts in TIFF files")
        print("  2. Sorting files into timepoint subfolders")
        print("  3. How split_channels integrates with timepoint sorting")
        print("\nExample:")
        print(
            "  python split_channels_timepoint_example.py ./my_images ./processed"
        )
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = (
        sys.argv[2] if len(sys.argv) > 2 else "./output_with_timepoint_sorting"
    )

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SPLIT CHANNELS WITH TIMEPOINT SORTING - DEMONSTRATION")
    print("=" * 70)
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")
    print("=" * 70)

    # Run demonstrations
    demonstrate_timepoint_detection(input_folder)

    response = input("\nProceed with sorting files? (y/n): ")
    if response.lower() == "y":
        demonstrate_sorting(input_folder, output_folder)
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\nCheck the output folder to see the organized structure:")
        print(f"  {output_folder}/")
        print("    ├── T1/     (single timepoint images)")
        print("    ├── T10/    (10-timepoint series)")
        print("    └── T50/    (50-timepoint series)")
        print("\nIn the Napari Batch Processing widget:")
        print("  1. Select your input files")
        print("  2. Choose 'Split Color Channels'")
        print("  3. Set parameters:")
        print("     - num_channels: 3 (or your channel count)")
        print("     - time_steps: 0 (leave at 0 - auto-detected)")
        print("     - sort_by_timepoints: ✓ ENABLED")
        print("  4. Run batch processing")
        print(
            "\nThe plugin will automatically organize and split your images!"
        )
    else:
        print("\nSkipped sorting demonstration.")


if __name__ == "__main__":
    main()

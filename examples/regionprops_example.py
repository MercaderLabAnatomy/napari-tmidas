"""
Example: Using the Regionprops Analysis Function

This example demonstrates how to use the regionprops analysis function
to extract properties from label images in a folder.
"""

from pathlib import Path

import numpy as np


# Create example label images
def create_example_data():
    """Create example label images for testing."""
    output_dir = Path("example_labels")
    output_dir.mkdir(exist_ok=True)

    # Create a simple 2D label image
    label_2d = np.zeros((200, 200), dtype=np.uint16)
    label_2d[50:100, 50:100] = 1  # Square region
    label_2d[120:160, 120:160] = 2  # Another square region
    np.save(output_dir / "example_2d.npy", label_2d)

    # Create a 3D label image (ZYX)
    label_3d = np.zeros((20, 200, 200), dtype=np.uint16)
    label_3d[5:15, 50:100, 50:100] = 1  # Cube region
    label_3d[10:18, 120:160, 120:160] = 2  # Another cube region
    np.save(output_dir / "example_3d.npy", label_3d)

    # Create a 4D time series (TZYX)
    label_4d = np.zeros((3, 20, 200, 200), dtype=np.uint16)
    for t in range(3):
        # Objects that move over time
        offset = t * 10
        label_4d[t, 5:15, 50 + offset : 100 + offset, 50:100] = 1
        label_4d[t, 10:18, 120:160, 120 + offset : 160 + offset] = 2
    np.save(output_dir / "example_4d_timeseries.npy", label_4d)

    print(f"Created example label images in: {output_dir}")
    return output_dir


# Method 1: Use the function directly
def analyze_with_function():
    """Analyze label images using the function directly."""
    from napari_tmidas.processing_functions.regionprops_analysis import (
        analyze_folder_regionprops,
    )

    # Create example data
    folder = create_example_data()

    # Analyze all label images in the folder
    output_csv = folder.parent / f"{folder.name}_regionprops.csv"

    df = analyze_folder_regionprops(
        folder_path=str(folder),
        output_csv=str(output_csv),
        max_spatial_dims=3,  # Treat up to 3D as spatial (2D: YX, 3D: ZYX)
    )

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_csv}")
    print("\nDataFrame preview:")
    print(df.head(10))
    print(f"\nTotal regions found: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")

    # Show some statistics
    if "T" in df.columns:
        print("\nRegions per timepoint:")
        print(df.groupby("T").size())

    if "area" in df.columns:
        print("\nArea statistics:")
        print(df["area"].describe())


# Method 2: Use through batch processing (requires napari)
def analyze_with_batch_processing():
    """
    Analyze label images using the batch processing widget.

    To use this in napari:
    1. Open napari
    2. Go to Plugins > T-MIDAS > Image Processing
    3. Select your folder containing label images
    4. Choose "Extract Regionprops to CSV" from the processing functions
    5. Set parameters:
       - max_spatial_dims: 3 for 3D (ZYX) or 2 for 2D (YX)
       - overwrite_existing: True to overwrite existing CSV files
    6. IMPORTANT: Set thread count to 1 (this processes entire folders at once)
    7. Click "Start Batch Processing"

    The function will:
    - Process all label images in the folder
    - Extract region properties for each labeled region
    - Handle multi-dimensional data (T, C, Z dimensions)
    - Save results to a single CSV file in the parent directory
    """
    print("To use in napari, follow these steps:")
    print("1. Open napari and go to Plugins > T-MIDAS > Image Processing")
    print("2. Browse to your folder containing label images")
    print("3. Select 'Extract Regionprops to CSV' from processing functions")
    print("4. Set max_spatial_dims (2 for 2D, 3 for 3D)")
    print("5. Set thread count to 1")
    print("6. Click 'Start Batch Processing'")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Regionprops Analysis Example")
    print("=" * 60)

    # Check if pandas is available
    try:
        import pandas as pd  # noqa: F401

        print("\n✓ pandas is available")

        # Run the direct analysis
        print("\n" + "=" * 60)
        print("Method 1: Direct Function Call")
        print("=" * 60)
        analyze_with_function()

    except ImportError:
        print("\n✗ pandas is not installed")
        print("Install it with: pip install pandas")
        sys.exit(1)

    # Show batch processing instructions
    print("\n" + "=" * 60)
    print("Method 2: Batch Processing in Napari")
    print("=" * 60)
    analyze_with_batch_processing()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

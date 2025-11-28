"""
Example: Create Grid View of Intensity + Labels Overlay

This example demonstrates how to create a grid visualization showing
intensity images overlaid with their corresponding label boundaries.
"""


def main():
    """Run the grid overlay example."""
    print("Grid View Overlay Example")
    print("=" * 50)

    # Note: In actual usage, this function is called by the batch processing
    # system which provides the filepath context automatically.

    # For demonstration purposes only
    # In real usage, the function will process all label images selected
    # in the batch processing queue

    print("\nUsage:")
    print("1. Use the Batch Image Processing widget in napari")
    print("2. Select a folder and suffix for label images")
    print("   Example: suffix = '_labels_filtered.tif'")
    print('3. Choose "Grid View: Intensity + Labels Overlay" function')
    print("4. Run batch processing")
    print("\nOutput:")
    print("- Single RGB image showing all selected overlays in a grid")
    print("- Green channel: intensity values")
    print("- Magenta boundaries: label edges")
    print("- Grid columns automatically sized based on image count")

    print("\nExpected file structure:")
    print("  folder/")
    print("    image1.tif                           # intensity")
    print("    image1_labels.tif                    # labels")
    print("    image2.tif                           # intensity")
    print("    image2_convpaint_labels_filtered.tif # labels")
    print("    ...")

    # Show what the function does
    print("\nFunction behavior:")
    print("- Scans folder for all label files (*_labels*.tif)")
    print("- Finds corresponding intensity images (removes label suffix)")
    print(
        "- Creates overlay for each pair (green=intensity, magenta=boundaries)"
    )
    print("- Arranges overlays in a grid")
    print("- Returns single RGB image for easy inspection")


if __name__ == "__main__":
    main()

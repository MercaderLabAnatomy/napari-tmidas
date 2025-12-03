"""
Example: Create Grid View of Intensity + Labels Overlay

This example demonstrates how to create a grid visualization showing
intensity images with optional label overlay.
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
    print("2. Select a folder and suffix for your files")
    print(
        "   - Overlay mode: suffix matches label files (e.g., '_labels_filtered.tif')"
    )
    print("   - Intensity only: suffix matches intensity files (e.g., '.tif')")
    print('3. Choose "Grid View: Intensity + Labels Overlay" function')
    print("4. In the parameter panel (right side):")
    print(
        "   - Set `label_suffix` to match label files (default '_labels.tif')"
    )
    print("   - Clear `label_suffix` to create an intensity-only grid")
    print("5. Run batch processing")
    print("\nOutput:")
    print("- Single RGB image showing all selected images in a grid")
    print(
        "- With overlay: grayscale intensity + colored label regions (60% opacity)"
    )
    print("- Without overlay: grayscale intensity images only")
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
    print("- Scans folder for files matching the suffix filter")
    print(
        "- If `label_suffix` is set: finds labels + matching intensity images"
    )
    print("- If `label_suffix` is empty: uses intensity files directly")
    print("- Creates visualization for each file/pair:")
    print("  * Overlay mode: grayscale intensity + colored label regions")
    print("  * Intensity-only mode: grayscale intensity only")
    print("- Arranges images in a grid")
    print("- Returns single RGB image for easy inspection")


if __name__ == "__main__":
    main()

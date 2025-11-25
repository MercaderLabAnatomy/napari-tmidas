"""
Example: Intensity-based label filtering using k-medoids clustering

This example demonstrates how to filter labels based on their intensity values
using 2-medoids or 3-medoids clustering.
"""

from pathlib import Path

import numpy as np

# Import the filtering functions
try:
    from napari_tmidas.processing_functions.intensity_label_filter import (
        filter_labels_by_intensity_2medoids,
        filter_labels_by_intensity_3medoids,
    )

    HAS_FUNCTIONS = True
except ImportError:
    print("Install scikit-learn-extra to use these functions:")
    print("  pip install scikit-learn-extra")
    HAS_FUNCTIONS = False


def create_example_data(output_dir: Path):
    """
    Create example label and intensity images for testing.

    This creates:
    - 3 label images with different numbers of labels
    - 3 corresponding intensity images where some labels have low intensity
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir = output_dir / "labels"
    intensity_dir = output_dir / "intensity"
    label_dir.mkdir(exist_ok=True)
    intensity_dir.mkdir(exist_ok=True)

    # Create 3 example images
    for i in range(3):
        # Create label image with 10 labels
        label_image = np.zeros((200, 200), dtype=np.uint16)
        np.random.seed(42 + i)

        # Place 10 rectangular labels at random positions
        for label_id in range(1, 11):
            x = np.random.randint(0, 150)
            y = np.random.randint(0, 150)
            label_image[y : y + 30, x : x + 30] = label_id

        # Create intensity image
        # Labels 1-4 have low intensity (noise), 5-10 have high intensity (signal)
        intensity_image = np.zeros((200, 200), dtype=np.float32)
        for label_id in range(1, 11):
            mask = label_image == label_id
            if label_id <= 4:
                # Low intensity + noise
                intensity_image[mask] = np.random.normal(20, 5, np.sum(mask))
            else:
                # High intensity + noise
                intensity_image[mask] = np.random.normal(100, 10, np.sum(mask))

        # Add background noise
        intensity_image[label_image == 0] = np.random.normal(
            5, 2, np.sum(label_image == 0)
        )

        # Save images
        import tifffile

        tifffile.imwrite(label_dir / f"image_{i:03d}.tif", label_image)
        tifffile.imwrite(intensity_dir / f"image_{i:03d}.tif", intensity_image)

    print(f"✅ Created example data in {output_dir}")
    print(f"   Labels: {label_dir}")
    print(f"   Intensity: {intensity_dir}")
    return label_dir, intensity_dir


def run_filtering_example(label_dir: Path, intensity_dir: Path):
    """
    Run both 2-medoids and 3-medoids filtering on example data.
    """
    if not HAS_FUNCTIONS:
        return

    output_dir_2med = label_dir.parent / "filtered_2medoids"
    output_dir_3med = label_dir.parent / "filtered_3medoids"
    output_dir_2med.mkdir(exist_ok=True)
    output_dir_3med.mkdir(exist_ok=True)

    # Process each label image
    label_files = sorted(label_dir.glob("*.tif"))

    print("\n" + "=" * 60)
    print("Running 2-medoids filtering...")
    print("=" * 60)

    for label_file in label_files:
        import tifffile

        label_image = tifffile.imread(label_file)

        # Apply 2-medoids filtering
        filtered_2med = filter_labels_by_intensity_2medoids(
            label_image,
            intensity_folder=str(intensity_dir),
            save_stats=True,
            current_filepath=str(label_file),
        )

        # Save result
        output_file = output_dir_2med / label_file.name
        tifffile.imwrite(output_file, filtered_2med)

    print("\n" + "=" * 60)
    print("Running 3-medoids filtering...")
    print("=" * 60)

    for label_file in label_files:
        import tifffile

        label_image = tifffile.imread(label_file)

        # Apply 3-medoids filtering
        filtered_3med = filter_labels_by_intensity_3medoids(
            label_image,
            intensity_folder=str(intensity_dir),
            save_stats=True,
            current_filepath=str(label_file),
        )

        # Save result
        output_file = output_dir_3med / label_file.name
        tifffile.imwrite(output_file, filtered_3med)

    print("\n✅ Filtering complete!")
    print(f"   2-medoids results: {output_dir_2med}")
    print(f"   3-medoids results: {output_dir_3med}")
    print(f"   Statistics: {label_dir.parent / 'intensity_filter_stats'}")


def compare_results():
    """
    Quick visual comparison of original vs filtered labels.
    """
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print("\n2-medoids clustering:")
    print("  - Separates labels into LOW and HIGH intensity groups")
    print("  - Removes labels in LOW group")
    print("  - Best for: signal spread over regions with varying S/N ratio")
    print("\n3-medoids clustering:")
    print("  - Separates labels into LOW, MEDIUM, and HIGH intensity groups")
    print("  - Removes labels in LOW group only")
    print(
        "  - Best for: distinct populations (noise, diffuse signal, strong signal)"
    )
    print(
        "\nCheck the statistics CSV files for detailed clustering information!"
    )


if __name__ == "__main__":
    # Set up example data directory
    example_dir = Path("./intensity_filter_example")

    # Create example data
    label_dir, intensity_dir = create_example_data(example_dir)

    # Run filtering
    if HAS_FUNCTIONS:
        run_filtering_example(label_dir, intensity_dir)
        compare_results()
    else:
        print("\nTo run this example, install scikit-learn-extra:")
        print("  pip install scikit-learn-extra")

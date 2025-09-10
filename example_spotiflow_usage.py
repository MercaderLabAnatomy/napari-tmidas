#!/usr/bin/env python3
"""
Example script demonstrating Spotiflow spot detection functionality.
This script shows how to use the new Spotiflow integration in napari-tmidas.
"""

import os
import subprocess
import sys

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from napari_tmidas.processing_functions.spotiflow_detection import (
    spotiflow_detect_spots,
)
from napari_tmidas.processing_functions.spotiflow_env_manager import (
    is_env_created,
    is_spotiflow_installed,
)


def create_synthetic_spot_image():
    """Create a synthetic image with spots for testing."""
    print("Creating synthetic test image...")

    # Create a 256x256 image
    image = np.zeros((256, 256), dtype=np.float32)

    # Add some bright spots at known locations
    spots = [
        (50, 50),
        (150, 100),
        (75, 200),
        (200, 180),
        (100, 75),
    ]

    # Add Gaussian-like spots
    for y, x in spots:
        # Create a small Gaussian-like spot
        y_coords, x_coords = np.ogrid[y - 3 : y + 4, x - 3 : x + 4]
        center_y, center_x = y, x

        # Gaussian-like intensity
        spot_intensity = 1000 * np.exp(
            -((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
            / (2 * 1.5**2)
        )

        # Ensure we don't go out of bounds
        y_start, y_end = max(0, y - 3), min(256, y + 4)
        x_start, x_end = max(0, x - 3), min(256, x + 4)
        spot_y_start, spot_y_end = max(0, 3 - (y - y_start)), min(
            7, 7 - (y_end - y - 3)
        )
        spot_x_start, spot_x_end = max(0, 3 - (x - x_start)), min(
            7, 7 - (x_end - x - 3)
        )

        image[y_start:y_end, x_start:x_end] += spot_intensity[
            spot_y_start:spot_y_end, spot_x_start:spot_x_end
        ]

    # Add some background noise
    image += np.random.normal(100, 20, image.shape)

    # Ensure positive values and convert to uint16
    image = np.clip(image, 0, 65535).astype(np.uint16)

    print(
        f"Created test image with shape {image.shape} and {len(spots)} spots"
    )
    return image, spots


def main():
    """Main example function."""
    print("=== Spotiflow Integration Example ===\n")

    # Check current status
    print("1. Checking Spotiflow status:")
    print(f"   Spotiflow installed in main env: {is_spotiflow_installed()}")
    print(f"   Dedicated environment exists: {is_env_created()}")
    print()

    # Create synthetic test data
    test_image, true_spots = create_synthetic_spot_image()

    print("2. Testing Spotiflow spot detection:")
    print("   Parameters:")
    print("   - Model: general (pretrained)")
    print("   - Subpixel: True")
    print("   - Exclude border: True")
    print()

    try:
        # Run spot detection
        print("   Running detection... (this may take a while on first run)")
        print(
            "   Note: First run will create the dedicated environment and download the model"
        )

        detected_mask = spotiflow_detect_spots(
            test_image,
            pretrained_model="general",
            subpixel=True,
            exclude_border=True,
            normalizer="percentile",
            normalizer_low=1.0,
            normalizer_high=99.8,
            spot_radius=3,
            output_csv=False,  # Don't save CSV in example
        )

        print("   ✓ Detection completed successfully!")
        print(f"   ✓ Output is a label mask with shape: {detected_mask.shape}")
        print(f"   ✓ Number of detected regions: {np.max(detected_mask)}")
        print(f"   ✓ Expected ~{len(true_spots)} spots")

        if np.max(detected_mask) > 0:
            print("   ✓ Label mask statistics:")
            print(
                f"      - Min label: {np.min(detected_mask[detected_mask > 0])}"
            )
            print(f"      - Max label: {np.max(detected_mask)}")
            print(f"      - Total labeled pixels: {np.sum(detected_mask > 0)}")

    except (ImportError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"   ✗ Detection failed: {e}")
        print("   This might be due to:")
        print("   - Network issues (downloading model)")
        print("   - Environment creation issues")
        print("   - Missing system dependencies")
        return False

    print("\n3. Advanced usage example:")
    try:
        # Test with different parameters
        detected_mask_2 = spotiflow_detect_spots(
            test_image,
            pretrained_model="general",
            spot_radius=2,
            prob_thresh=0.5,
            output_csv=False,
        )

        print("   ✓ Advanced detection with different parameters:")
        print(f"      - Label mask shape: {detected_mask_2.shape}")
        print(f"      - Number of detected regions: {np.max(detected_mask_2)}")

    except (ImportError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"   ✗ Advanced detection failed: {e}")

    print("\n=== Example completed successfully! ===")
    print("\nNotes:")
    print(
        "- The dedicated environment is created at: ~/.napari-tmidas/envs/spotiflow"
    )
    print("- Pretrained models are cached for future use")
    print("- Output is a label mask where each spot is a labeled region")
    print(
        "- CSV coordinates are automatically saved alongside processed images"
    )
    print(
        "- Available pretrained models: general, hybiss, synth_complex, synth_3d, smfish_3d"
    )
    print("- For 3D data, use synth_3d or smfish_3d models")
    print(
        "- Custom trained models can be loaded using the model_path parameter"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

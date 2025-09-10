#!/usr/bin/env python3
"""
Test script to debug the Spotiflow environment issue.
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


def create_test_3d_image():
    """Create a simple 3D test image."""
    # Create a small 3D volume for testing
    image = np.zeros((10, 64, 64), dtype=np.uint16)

    # Add some bright spots
    image[5, 32, 32] = 1000
    image[3, 20, 40] = 800
    image[7, 45, 25] = 1200

    # Add some background noise
    image += np.random.normal(100, 20, image.shape).astype(np.uint16)
    image = np.clip(image, 0, 65535)

    return image


def main():
    print("=== Debugging Spotiflow Environment Issue ===")

    # Create test image
    test_image = create_test_3d_image()
    print(
        f"Created test 3D image: shape={test_image.shape}, dtype={test_image.dtype}"
    )

    try:
        print("\nTesting Spotiflow spot detection with smfish_3d model...")
        result = spotiflow_detect_spots(
            test_image,
            pretrained_model="smfish_3d",
            subpixel=True,
            peak_mode="fast",
            normalizer="percentile",
            normalizer_low=1.0,
            normalizer_high=99.8,
            prob_thresh=0.5,  # Set a specific threshold to avoid None issues
            n_tiles="auto",
            exclude_border=True,
            scale="auto",
            min_distance=1,
            spot_radius=2,
            output_csv=False,  # Don't save CSV for test
            force_dedicated_env=True,
        )

        print(f"✓ Success! Result shape: {result.shape}")
        print(f"✓ Result dtype: {result.dtype}")
        print(f"✓ Max label: {np.max(result)}")

    except (ImportError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"✗ Failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Simple Spotiflow test with CPU fallback.
"""
import subprocess

import numpy as np

from src.napari_tmidas.processing_functions.spotiflow_detection import (
    spotiflow_detect_spots,
)


def main():
    print("=== Testing Spotiflow with CPU fallback ===")

    # Create test 3D image
    test_image = np.random.randint(0, 1000, size=(10, 64, 64), dtype=np.uint16)
    print(
        f"Created test 3D image: shape={test_image.shape}, dtype={test_image.dtype}"
    )

    try:
        print("\nTesting Spotiflow spot detection with CPU device...")
        result = spotiflow_detect_spots(
            image=test_image,
            pretrained_model="smfish_3d",
            model_path="",
            subpixel=True,
            peak_mode="fast",
            normalizer="percentile",
            normalizer_low=1.0,
            normalizer_high=99.8,
            prob_thresh=0.5,
            n_tiles="auto",
            exclude_border=True,
            scale="auto",
            min_distance=1,
            spot_radius=2,
            force_cpu=True,  # Add this parameter to force CPU
        )

        if result is not None:
            print(f"✓ Success! Detected spots shape: {result.shape}")
            print(f"✓ Number of spots: {len(result)}")
            if len(result) > 0:
                print(f"✓ First few spots: {result[:3]}")
        else:
            print("✗ No result returned")

    except (ImportError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"✗ Failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

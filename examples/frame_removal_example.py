"""
Example: Using the Frame Removal Tool

This example demonstrates how to use the Frame Removal Tool widget
to interactively remove frames from a time-series image.

The Frame Removal Tool provides a human-in-the-loop interface for
quality control of time-series data.
"""

import numpy as np
import napari


def create_sample_timelapse():
    """
    Create a sample time-lapse with some simulated artifacts.
    
    Returns
    -------
    np.ndarray
        TYX image with 20 time frames
    """
    # Create 20 frames of 256x256 images
    frames = []
    
    for t in range(20):
        # Base image with moving circles
        img = np.zeros((256, 256), dtype=np.uint8)
        
        # Add moving spots
        y, x = np.ogrid[:256, :256]
        
        # Spot 1: moves from left to right
        spot1_x = 50 + t * 10
        spot1_y = 100
        mask1 = ((x - spot1_x)**2 + (y - spot1_y)**2) < 400
        img[mask1] = 200
        
        # Spot 2: moves from top to bottom
        spot2_x = 180
        spot2_y = 30 + t * 10
        mask2 = ((x - spot2_x)**2 + (y - spot2_y)**2) < 300
        img[mask2] = 150
        
        # Simulate artifacts in specific frames
        if t == 5:
            # Frame 5: Add horizontal lines (motion artifact)
            img[100:110, :] = 255
            img[150:155, :] = 255
        
        if t == 12:
            # Frame 12: Add random noise (acquisition glitch)
            noise = np.random.randint(0, 100, (256, 256))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        if t == 17:
            # Frame 17: Reduce brightness dramatically (out of focus)
            img = (img * 0.2).astype(np.uint8)
        
        frames.append(img)
    
    return np.stack(frames, axis=0)


def main():
    """
    Main function to run the example.
    """
    print("Creating sample time-lapse data...")
    timelapse = create_sample_timelapse()
    
    print(f"Created time-lapse with shape: {timelapse.shape}")
    print(f"Data type: {timelapse.dtype}")
    print(f"Number of frames: {timelapse.shape[0]}")
    print("\nArtifacts added to frames:")
    print("  - Frame 6 (index 5): Motion artifacts (horizontal lines)")
    print("  - Frame 13 (index 12): Acquisition glitch (noise)")
    print("  - Frame 18 (index 17): Out of focus (low brightness)")
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    # Add the time-lapse image
    viewer.add_image(
        timelapse,
        name="Sample Time-lapse",
        colormap="gray",
        contrast_limits=[0, 255]
    )
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. The time-lapse is now loaded in Napari")
    print("2. Open the Frame Removal Tool:")
    print("   Plugins > napari-tmidas > Frame Removal Tool")
    print("3. Select 'Sample Time-lapse' from the layer dropdown")
    print("4. Navigate through frames using the slider or buttons")
    print("5. Mark bad frames (6, 13, 18) by checking the removal box")
    print("6. Click 'Preview Result' to see the cleaned time series")
    print("7. Click 'Save Cleaned Image' to save the result")
    print("="*60)
    
    # Start napari
    napari.run()


if __name__ == "__main__":
    main()

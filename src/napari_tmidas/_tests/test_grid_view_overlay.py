"""Tests for grid view overlay processing function."""

import numpy as np
import pytest

try:
    from napari_tmidas.processing_functions.grid_view_overlay import (
        _create_grid,
        _create_overlay,
        _get_intensity_filename,
    )

    GRID_OVERLAY_AVAILABLE = True
except ImportError:
    GRID_OVERLAY_AVAILABLE = False


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_get_intensity_filename():
    """Test intensity filename extraction from label filenames."""
    assert (
        _get_intensity_filename("test_convpaint_labels_filtered.tif")
        == "test.tif"
    )
    assert _get_intensity_filename("test_labels.tif") == "test.tif"
    assert _get_intensity_filename("test_labels_filtered.tif") == "test.tif"
    assert _get_intensity_filename("test_intensity_filtered.tif") == "test.tif"
    assert _get_intensity_filename("unknown.tif") == "unknown.tif"


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_overlay():
    """Test overlay creation with intensity and labels."""
    # Create simple test images
    intensity = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    labels = np.zeros((100, 100), dtype=np.uint16)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:80] = 2

    # Create overlay without downsampling (with overlay enabled)
    overlay = _create_overlay(intensity, labels, show_overlay=True)

    # Check output
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_overlay_with_downsampling():
    """Test overlay creation with downsampling."""
    # Create larger test images
    intensity = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    labels = np.zeros((1000, 1000), dtype=np.uint16)
    labels[200:400, 200:400] = 1
    labels[600:800, 600:800] = 2

    # Create overlay with downsampling to 300px
    overlay = _create_overlay(
        intensity, labels, target_size=300, show_overlay=True
    )

    # Check output is downsampled
    assert overlay.shape[0] <= 300
    assert overlay.shape[1] <= 300
    assert overlay.shape[2] == 3
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_grid():
    """Test grid creation from multiple images."""
    # Create test images
    images = [
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        for _ in range(6)
    ]

    # Create grid with 3 columns (should be 2 rows)
    grid = _create_grid(images, grid_cols=3)

    # Check output
    assert grid.shape == (
        100,
        150,
        3,
    )  # 2 rows * 50px, 3 cols * 50px, 3 channels
    assert grid.dtype == np.uint8


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_grid_grayscale():
    """Test grid creation with grayscale images."""
    # Create grayscale test images
    images = [
        np.random.randint(0, 255, (50, 50), dtype=np.uint8) for _ in range(4)
    ]

    # Create grid with 2 columns
    grid = _create_grid(images, grid_cols=2)

    # Check output
    assert grid.shape == (100, 100)  # 2 rows * 50px, 2 cols * 50px
    assert grid.dtype == np.uint8


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_grid_empty():
    """Test grid creation with empty list."""
    grid = _create_grid([], grid_cols=4)
    assert grid is None


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_overlay_intensity_only():
    """Test overlay creation with intensity only (no label overlay)."""
    # Create simple test images
    intensity = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    labels = np.zeros((100, 100), dtype=np.uint16)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:80] = 2

    # Create overlay with overlay disabled (intensity only)
    overlay = _create_overlay(intensity, labels, show_overlay=False)

    # Check output - should be grayscale RGB (all channels equal)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255

    # Check that all channels are equal (grayscale)
    assert np.array_equal(overlay[:, :, 0], overlay[:, :, 1])
    assert np.array_equal(overlay[:, :, 1], overlay[:, :, 2])


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
def test_create_overlay_with_and_without_labels():
    """Test that overlay mode creates different outputs with labels."""
    # Create test images with labels
    intensity = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    labels = np.zeros((100, 100), dtype=np.uint16)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:80] = 2

    # Create overlay with labels
    overlay_with_labels = _create_overlay(intensity, labels, show_overlay=True)

    # Create overlay without labels (intensity only)
    overlay_intensity_only = _create_overlay(
        intensity, labels, show_overlay=False
    )

    # Both should have same shape
    assert overlay_with_labels.shape == overlay_intensity_only.shape

    # But different content (overlay should have colored regions)
    # In the intensity-only version, all channels should be equal
    assert np.array_equal(
        overlay_intensity_only[:, :, 0], overlay_intensity_only[:, :, 1]
    )

    # In the overlay version, channels should differ in labeled regions
    # Check the labeled region [20:40, 20:40]
    labeled_region_r = overlay_with_labels[20:40, 20:40, 0]
    labeled_region_g = overlay_with_labels[20:40, 20:40, 1]
    labeled_region_b = overlay_with_labels[20:40, 20:40, 2]

    # At least one channel pair should differ in the labeled region
    assert (
        not np.array_equal(labeled_region_r, labeled_region_g)
        or not np.array_equal(labeled_region_g, labeled_region_b)
        or not np.array_equal(labeled_region_r, labeled_region_b)
    )

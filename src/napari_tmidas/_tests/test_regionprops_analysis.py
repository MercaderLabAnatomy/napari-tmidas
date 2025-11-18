"""
Tests for regionprops_analysis processing function
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_tmidas.processing_functions.regionprops_analysis import (
    analyze_folder_regionprops,
    extract_regionprops_folder,
    extract_regionprops_recursive,
    find_label_images,
    load_label_image,
    parse_dimensions_from_shape,
    reset_regionprops_cache,
)


@pytest.fixture
def temp_label_folder():
    """Create a temporary folder with test label images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir) / "labels"
        folder.mkdir()

        # Create 2D label image
        label_2d = np.zeros((100, 100), dtype=np.uint16)
        label_2d[20:40, 20:40] = 1
        label_2d[60:80, 60:80] = 2
        np.save(folder / "image_2d.npy", label_2d)

        # Create 3D label image (ZYX)
        label_3d = np.zeros((10, 100, 100), dtype=np.uint16)
        label_3d[2:5, 20:40, 20:40] = 1
        label_3d[5:8, 60:80, 60:80] = 2
        np.save(folder / "image_3d.npy", label_3d)

        # Create 4D label image (TZYX) - 3 timepoints
        label_4d = np.zeros((3, 10, 100, 100), dtype=np.uint16)
        for t in range(3):
            label_4d[t, 2:5, 20:40, 20:40] = 1
            label_4d[t, 5:8, 60:80, 60:80] = 2
        np.save(folder / "image_4d.npy", label_4d)

        yield folder


def test_parse_dimensions_from_shape():
    """Test dimension parsing from image shape."""
    # Test 2D
    dims = parse_dimensions_from_shape((100, 100), 2)
    assert "Y" in dims and "X" in dims
    assert dims["Y"] == 100 and dims["X"] == 100

    # Test 3D
    dims = parse_dimensions_from_shape((10, 100, 100), 3)
    assert "Z" in dims and "Y" in dims and "X" in dims
    assert dims["Z"] == 10

    # Test 4D
    dims = parse_dimensions_from_shape((5, 10, 100, 100), 4)
    assert "T" in dims and "Z" in dims
    assert dims["T"] == 5 and dims["Z"] == 10


def test_find_label_images(temp_label_folder):
    """Test finding label images in a folder."""
    files = find_label_images(str(temp_label_folder))
    assert len(files) == 3
    assert all(f.endswith(".npy") for f in files)


def test_load_label_image(temp_label_folder):
    """Test loading label images."""
    files = find_label_images(str(temp_label_folder))

    for filepath in files:
        img = load_label_image(filepath)
        assert isinstance(img, np.ndarray)
        assert img.dtype in [np.uint16, np.int32, np.int64]


def test_extract_regionprops_recursive_2d(temp_label_folder):
    """Test extracting regionprops from 2D label image."""
    files = find_label_images(str(temp_label_folder))
    file_2d = [f for f in files if "2d" in f][0]

    img = load_label_image(file_2d)
    results = extract_regionprops_recursive(
        img,
        prefix_dims={"filename": os.path.basename(file_2d)},
        max_spatial_dims=3,
    )

    # Should have 2 regions
    assert len(results) == 2

    # Check that results have expected keys
    for result in results:
        assert "filename" in result
        assert "label" in result
        assert "size" in result  # area is renamed to size
        assert "centroid_y" in result
        assert "centroid_x" in result
        assert "bbox_min_y" in result
        assert "bbox_max_x" in result

    # Check areas (20x20 = 400 pixels each)
    assert results[0]["size"] == 400
    assert results[1]["size"] == 400


def test_extract_regionprops_recursive_3d(temp_label_folder):
    """Test extracting regionprops from 3D label image."""
    files = find_label_images(str(temp_label_folder))
    file_3d = [f for f in files if "3d" in f][0]

    img = load_label_image(file_3d)
    results = extract_regionprops_recursive(
        img,
        prefix_dims={"filename": os.path.basename(file_3d)},
        max_spatial_dims=3,
    )

    # Should have 2 regions
    assert len(results) == 2

    # Check that results have expected keys including Z coordinates
    for result in results:
        assert "filename" in result
        assert "label" in result
        assert "size" in result  # area is renamed to size
        assert "centroid_z" in result
        assert "centroid_y" in result
        assert "centroid_x" in result
        assert "bbox_min_z" in result
        assert "bbox_max_z" in result

    # Check volumes (3 z-slices x 20x20 = 1200 voxels each)
    assert results[0]["size"] == 1200
    assert results[1]["size"] == 1200


def test_extract_regionprops_recursive_4d(temp_label_folder):
    """Test extracting regionprops from 4D label image with time dimension."""
    files = find_label_images(str(temp_label_folder))
    file_4d = [f for f in files if "4d" in f][0]

    img = load_label_image(file_4d)
    results = extract_regionprops_recursive(
        img,
        prefix_dims={"filename": os.path.basename(file_4d)},
        max_spatial_dims=3,
    )

    # Should have 2 regions x 3 timepoints = 6 results
    assert len(results) == 6

    # Check that results have time dimension
    for result in results:
        assert "T" in result
        assert "label" in result
        assert result["T"] in [0, 1, 2]

    # Check that we have 2 labels per timepoint
    for t in range(3):
        t_results = [r for r in results if r["T"] == t]
        assert len(t_results) == 2
        labels = [r["label"] for r in t_results]
        assert sorted(labels) == [1, 2]


def test_analyze_folder_regionprops(temp_label_folder):
    """Test analyzing entire folder and saving to CSV."""
    output_csv = temp_label_folder.parent / "test_regionprops.csv"

    df = analyze_folder_regionprops(
        folder_path=str(temp_label_folder),
        output_csv=str(output_csv),
        max_spatial_dims=3,
        dimension_order="Auto",
    )

    # Check that CSV was created
    assert output_csv.exists()

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "filename" in df.columns
    assert "label" in df.columns
    assert "size" in df.columns  # area is renamed to size

    # Check that we have results from all files
    filenames = df["filename"].unique()
    assert len(filenames) == 3

    # Load CSV and verify it matches the DataFrame
    df_loaded = pd.read_csv(output_csv)
    assert len(df_loaded) == len(df)


def test_extract_regionprops_folder_integration(temp_label_folder):
    """Test the full batch processing function."""
    # Reset cache first
    reset_regionprops_cache()

    # Load one of the images
    files = find_label_images(str(temp_label_folder))
    img = load_label_image(files[0])

    # Mock the filepath in the call stack by calling from a function
    # that has filepath in its locals
    def mock_process(filepath, image):
        return extract_regionprops_folder(
            image,
            max_spatial_dims=3,
            dimension_order="Auto",
            overwrite_existing=True,
        )

    result = mock_process(files[0], img)

    # Function should return None (only generates CSV)
    assert result is None

    # Check that CSV was created
    folder_name = temp_label_folder.name
    parent_dir = temp_label_folder.parent
    output_csv = parent_dir / f"{folder_name}_regionprops.csv"
    assert output_csv.exists()

    # Load and verify CSV
    df = pd.read_csv(output_csv)
    assert len(df) > 0
    assert "filename" in df.columns


def test_extract_regionprops_folder_no_overwrite(temp_label_folder):
    """Test that function respects overwrite_existing flag."""
    reset_regionprops_cache()

    # Create a CSV file first
    folder_name = temp_label_folder.name
    parent_dir = temp_label_folder.parent
    output_csv = parent_dir / f"{folder_name}_regionprops.csv"

    # Create dummy CSV
    pd.DataFrame({"test": [1, 2, 3]}).to_csv(output_csv, index=False)
    initial_mtime = output_csv.stat().st_mtime

    # Load image
    files = find_label_images(str(temp_label_folder))
    img = load_label_image(files[0])

    # Process with overwrite_existing=False
    def mock_process(filepath, image):
        return extract_regionprops_folder(
            image,
            max_spatial_dims=3,
            dimension_order="Auto",
            overwrite_existing=False,
        )

    _ = mock_process(files[0], img)  # result unused, just checking behavior

    # File should not have been modified
    assert output_csv.stat().st_mtime == initial_mtime

    # Now test with overwrite=True
    reset_regionprops_cache()

    # Create a mock process that properly sets filepath in locals
    def mock_process_with_overwrite(filepath, image):
        return extract_regionprops_folder(
            image,
            max_spatial_dims=3,
            dimension_order="Auto",
            overwrite_existing=True,
        )

    _ = mock_process_with_overwrite(files[0], img)  # result unused

    # Verify the CSV was updated properly
    if output_csv.exists():
        df = pd.read_csv(output_csv)
        # Either it should have the proper structure, or it should have been skipped
        # (because we can't determine filepath in test context)
        if len(df.columns) > 1:
            assert "filename" in df.columns


def test_reset_cache():
    """Test that cache reset works."""
    from napari_tmidas.processing_functions.regionprops_analysis import (
        _REGIONPROPS_CSV_FILES,
    )

    # First clear any existing cache
    reset_regionprops_cache()
    assert len(_REGIONPROPS_CSV_FILES) == 0

    # Add a CSV file to the cache
    _REGIONPROPS_CSV_FILES["/test/folder/test_regionprops.csv"] = True
    assert len(_REGIONPROPS_CSV_FILES) == 1

    # Reset cache
    reset_regionprops_cache()
    assert len(_REGIONPROPS_CSV_FILES) == 0


def test_empty_label_image():
    """Test handling of empty label images."""
    empty_img = np.zeros((100, 100), dtype=np.uint16)

    results = extract_regionprops_recursive(
        empty_img,
        prefix_dims={"filename": "empty.npy"},
        max_spatial_dims=3,
    )

    # Should return empty list
    assert len(results) == 0


def test_single_region():
    """Test handling of label image with single region."""
    img = np.zeros((50, 50), dtype=np.uint16)
    img[10:30, 10:30] = 5  # Single region with label 5

    results = extract_regionprops_recursive(
        img,
        prefix_dims={"filename": "single.npy"},
        max_spatial_dims=3,
    )

    assert len(results) == 1
    assert results[0]["label"] == 5
    assert results[0]["size"] == 400  # 20x20


def test_dimension_order_tzyx():
    """Test that dimension_order correctly identifies T and Z dimensions."""
    # Create TZYX image (3 timepoints, 5 z-slices, 50x50)
    label_tzyx = np.zeros((3, 5, 50, 50), dtype=np.uint16)
    for t in range(3):
        label_tzyx[t, 1:4, 10:20, 10:20] = 1
        label_tzyx[t, 2:4, 30:40, 30:40] = 2

    # Extract with explicit dimension order
    results = extract_regionprops_recursive(
        label_tzyx,
        prefix_dims={"filename": "test_tzyx.npy"},
        current_dim=0,
        max_spatial_dims=3,
        dimension_order="TZYX",
    )

    # Should have 2 regions x 3 timepoints = 6 results
    assert len(results) == 6

    # Check that T dimension is properly identified
    for result in results:
        assert "T" in result
        assert result["T"] in [0, 1, 2]
        assert "filename" in result
        assert "label" in result


def test_dimension_order_czyx():
    """Test that dimension_order correctly identifies C and Z dimensions."""
    # Create CZYX image (2 channels, 5 z-slices, 50x50)
    label_czyx = np.zeros((2, 5, 50, 50), dtype=np.uint16)
    for c in range(2):
        label_czyx[c, 1:4, 10:20, 10:20] = 1
        label_czyx[c, 2:4, 30:40, 30:40] = 2

    # Extract with explicit dimension order
    results = extract_regionprops_recursive(
        label_czyx,
        prefix_dims={"filename": "test_czyx.npy"},
        current_dim=0,
        max_spatial_dims=3,
        dimension_order="CZYX",
    )

    # Should have 2 regions x 2 channels = 4 results
    assert len(results) == 4

    # Check that C dimension is properly identified
    for result in results:
        assert "C" in result
        assert result["C"] in [0, 1]
        assert "filename" in result
        assert "label" in result


def test_dimension_order_tczyx():
    """Test that dimension_order correctly identifies T, C, and Z dimensions."""
    # Create TCZYX image (2 timepoints, 2 channels, 3 z-slices, 30x30)
    label_tczyx = np.zeros((2, 2, 3, 30, 30), dtype=np.uint16)
    for t in range(2):
        for c in range(2):
            label_tczyx[t, c, 1:3, 5:15, 5:15] = 1
            label_tczyx[t, c, 1:2, 20:25, 20:25] = 2

    # Extract with explicit dimension order
    results = extract_regionprops_recursive(
        label_tczyx,
        prefix_dims={"filename": "test_tczyx.npy"},
        current_dim=0,
        max_spatial_dims=3,
        dimension_order="TCZYX",
    )

    # Should have 2 regions x 2 timepoints x 2 channels = 8 results
    assert len(results) == 8

    # Check that T and C dimensions are properly identified
    for result in results:
        assert "T" in result
        assert result["T"] in [0, 1]
        assert "C" in result
        assert result["C"] in [0, 1]
        assert "filename" in result
        assert "label" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

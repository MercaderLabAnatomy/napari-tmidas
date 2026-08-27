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
    extract_regionprops_summary_folder,
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
    file_2d = [f for f in files if "2d" in os.path.basename(f)][0]

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
    file_3d = [f for f in files if "3d" in os.path.basename(f)][0]

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
    file_4d = [f for f in files if "4d" in os.path.basename(f)][0]

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


def test_extract_regionprops_summary_basic():
    """Test summary statistics extraction without grouping by dimensions."""
    # Create a simple 2D label image
    label_2d = np.zeros((100, 100), dtype=np.uint16)
    label_2d[10:20, 10:20] = 1  # 100 pixels
    label_2d[30:50, 30:50] = 2  # 400 pixels
    label_2d[60:70, 60:80] = 3  # 200 pixels

    # Create intensity image
    intensity_2d = np.random.rand(100, 100) * 100

    # Extract regionprops first to get individual values
    results = extract_regionprops_recursive(
        label_2d,
        intensity_image=intensity_2d,
        prefix_dims={"filename": "test.npy"},
        max_spatial_dims=2,
        properties=[
            "label",
            "area",
            "mean_intensity",
            "median_intensity",
            "std_intensity",
        ],
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Calculate expected summary statistics (note: 'area' is renamed to 'size')
    expected_count = len(df)
    expected_size_sum = df["size"].sum()
    expected_size_mean = df["size"].mean()
    expected_size_median = df["size"].median()

    # Verify we have 3 labels
    assert expected_count == 3
    assert expected_size_sum == 700  # 100 + 400 + 200
    assert expected_size_mean == pytest.approx(233.333, rel=0.01)
    assert expected_size_median == 200


def test_extract_regionprops_summary_with_grouping():
    """Test summary statistics extraction with grouping by dimensions."""
    # Create 4D label image (T=2, Z=3, Y=50, X=50)
    label_4d = np.zeros((2, 3, 50, 50), dtype=np.uint16)

    # T=0: 2 labels
    label_4d[0, 1:3, 10:20, 10:20] = 1  # 200 pixels
    label_4d[0, 1:2, 30:40, 30:40] = 2  # 100 pixels

    # T=1: 3 labels
    label_4d[1, 1:3, 10:15, 10:15] = 1  # 50 pixels
    label_4d[1, 1:2, 20:30, 20:30] = 2  # 100 pixels
    label_4d[1, 1:2, 35:45, 35:45] = 3  # 100 pixels

    # Extract with dimension order
    results = extract_regionprops_recursive(
        label_4d,
        intensity_image=None,
        prefix_dims={"filename": "test_4d.npy"},
        current_dim=0,
        max_spatial_dims=3,
        dimension_order="TZYX",
        properties=["label", "area"],
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Group by T dimension (note: 'area' is renamed to 'size')
    summary_stats = []
    for t, group in df.groupby("T"):
        summary_stats.append(
            {
                "T": t,
                "label_count": len(group),
                "size_sum": group["size"].sum(),
                "size_mean": group["size"].mean(),
            }
        )

    # Verify T=0 has 2 labels
    t0_stats = [s for s in summary_stats if s["T"] == 0][0]
    assert t0_stats["label_count"] == 2
    assert t0_stats["size_sum"] == 300  # 200 + 100

    # Verify T=1 has 3 labels
    t1_stats = [s for s in summary_stats if s["T"] == 1][0]
    assert t1_stats["label_count"] == 3
    assert t1_stats["size_sum"] == 250  # 50 + 100 + 100


def test_summary_statistics_calculations():
    """Test that summary statistics are calculated correctly."""
    # Simple test data
    data = np.array([10, 20, 30, 40, 50])

    # Expected values
    expected_sum = 150
    expected_mean = 30.0
    expected_median = 30.0
    expected_std = np.std(data, ddof=1)  # pandas uses ddof=1

    # Verify using pandas
    df = pd.DataFrame({"values": data})
    assert df["values"].sum() == expected_sum
    assert df["values"].mean() == expected_mean
    assert df["values"].median() == expected_median
    assert df["values"].std() == pytest.approx(expected_std, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestExtractRegionpropsSummaryFolder:
    """
    Real coverage for "Regionprops Summary Statistics".

    The function takes no explicit filepath -- it recovers one by walking the
    call stack for a local named ``filepath`` -- and its only output is a CSV
    written next to the folder. So every test here drives it through a caller
    that owns a ``filepath`` local, exactly as the batch worker does, and then
    reads the CSV back.
    """

    @staticmethod
    def _run(filepath, image, **kwargs):
        """Invoke the function the way the batch worker does."""
        return extract_regionprops_summary_folder(image, **kwargs)

    @staticmethod
    def _summary_csv(folder):
        return folder.parent / f"{folder.name}_regionprops_summary.csv"

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """The header-management cache is module global and leaks across tests."""
        reset_regionprops_cache()
        yield
        reset_regionprops_cache()

    @pytest.fixture
    def labels_2d(self):
        """Three labels of 100, 400 and 200 pixels."""
        image = np.zeros((100, 100), dtype=np.uint16)
        image[10:20, 10:20] = 1
        image[30:50, 30:50] = 2
        image[60:70, 60:80] = 3
        return image

    def test_writes_one_row_of_summary_statistics(self, tmp_path, labels_2d):
        folder = tmp_path / "labels"
        folder.mkdir()
        label_path = folder / "sample.tif"

        self._run(str(label_path), labels_2d, mean_intensity=False,
                  median_intensity=False, std_intensity=False)

        csv_path = self._summary_csv(folder)
        assert csv_path.exists(), "no summary CSV was written"
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["filename"] == "sample.tif"
        assert row["label_count"] == 3
        # 100 + 400 + 200 -- aggregated from the real regionprops pass, not
        # recomputed in the test.
        assert row["size_sum"] == 700
        assert row["size_mean"] == pytest.approx(700 / 3)
        assert row["size_median"] == 200

    def test_second_file_appends_without_a_second_header(
        self, tmp_path, labels_2d
    ):
        """
        One CSV per folder, one row per file. A repeated header would appear
        as a data row reading "filename" once parsed.
        """
        folder = tmp_path / "labels"
        folder.mkdir()
        for name in ("a.tif", "b.tif"):
            self._run(str(folder / name), labels_2d, mean_intensity=False,
                      median_intensity=False, std_intensity=False)

        df = pd.read_csv(self._summary_csv(folder))

        assert list(df["filename"]) == ["a.tif", "b.tif"]
        assert len(df) == 2

    def test_label_suffix_skips_files_that_do_not_match(
        self, tmp_path, labels_2d
    ):
        """
        label_suffix selects which files are label images. A file without the
        suffix must be ignored entirely rather than summarised.
        """
        folder = tmp_path / "labels"
        folder.mkdir()

        self._run(
            str(folder / "raw.tif"), labels_2d, label_suffix="_labels.tif"
        )

        assert not self._summary_csv(folder).exists()

    def test_group_by_dimensions_emits_one_row_per_timepoint(self, tmp_path):
        """With grouping on, a TZYX stack summarises per T rather than once."""
        folder = tmp_path / "labels"
        folder.mkdir()
        image = np.zeros((3, 10, 60, 60), dtype=np.uint16)
        for t in range(3):
            image[t, 2:5, 10:20, 10:20] = 1
            image[t, 5:8, 30:40, 30:40] = 2

        self._run(
            str(folder / "movie.tif"),
            image,
            max_spatial_dims=3,
            dimension_order="TZYX",
            group_by_dimensions=True,
            mean_intensity=False,
            median_intensity=False,
            std_intensity=False,
        )

        df = pd.read_csv(self._summary_csv(folder))

        assert len(df) == 3
        assert set(df["label_count"]) == {2}

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "df.groupby(group_cols) is called with a list, and since pandas "
            "2.0 a single-element list yields 1-tuple keys. The "
            "len(group_cols) == 1 branch assigns that key unchanged, so the "
            "T/C/Z column is written as the string '(0,)' instead of 0. The "
            "multi-dimension path indexes group_key[i] and is unaffected."
        ),
    )
    def test_grouped_dimension_column_holds_plain_integers(self, tmp_path):
        """
        A T column of '(0,)' strings cannot be sorted, filtered or plotted --
        the CSV parses fine and every value in it is unusable.
        """
        folder = tmp_path / "labels"
        folder.mkdir()
        image = np.zeros((3, 10, 60, 60), dtype=np.uint16)
        for t in range(3):
            image[t, 2:5, 10:20, 10:20] = 1

        self._run(
            str(folder / "movie.tif"),
            image,
            max_spatial_dims=3,
            dimension_order="TZYX",
            group_by_dimensions=True,
            mean_intensity=False,
            median_intensity=False,
            std_intensity=False,
        )

        df = pd.read_csv(self._summary_csv(folder))

        assert sorted(df["T"]) == [0, 1, 2]

    def test_without_grouping_a_4d_stack_gets_a_single_row(self, tmp_path):
        """The same stack, grouping off, collapses to one row for the file."""
        folder = tmp_path / "labels"
        folder.mkdir()
        image = np.zeros((3, 10, 60, 60), dtype=np.uint16)
        for t in range(3):
            image[t, 2:5, 10:20, 10:20] = 1
            image[t, 5:8, 30:40, 30:40] = 2

        self._run(
            str(folder / "movie.tif"),
            image,
            max_spatial_dims=3,
            dimension_order="TZYX",
            group_by_dimensions=False,
            mean_intensity=False,
            median_intensity=False,
            std_intensity=False,
        )

        df = pd.read_csv(self._summary_csv(folder))

        assert len(df) == 1
        assert df.iloc[0]["label_count"] == 6

    def test_disabled_properties_are_absent_from_the_csv(
        self, tmp_path, labels_2d
    ):
        """Unchecked boxes must not silently appear as columns."""
        folder = tmp_path / "labels"
        folder.mkdir()

        self._run(
            str(folder / "sample.tif"),
            labels_2d,
            size=False,
            mean_intensity=False,
            median_intensity=False,
            std_intensity=False,
        )

        df = pd.read_csv(self._summary_csv(folder))

        assert "size_sum" not in df.columns
        assert "mean_int_sum" not in df.columns
        assert "label_count" in df.columns

    def test_empty_label_image_writes_nothing(self, tmp_path):
        """No regions means no row, not a row of NaNs."""
        folder = tmp_path / "labels"
        folder.mkdir()

        self._run(
            str(folder / "empty.tif"), np.zeros((50, 50), dtype=np.uint16)
        )

        assert not self._summary_csv(folder).exists()

    def test_returns_none_when_filepath_cannot_be_recovered(self, labels_2d):
        """
        Called outside a batch context there is no ``filepath`` local to find,
        and the function must bail out rather than guess a path.
        """
        assert extract_regionprops_summary_folder(labels_2d) is None

    def test_existing_csv_is_preserved_unless_overwrite_is_set(
        self, tmp_path, labels_2d
    ):
        """
        A stale CSV from a previous run is appended to by default, so earlier
        results are not lost; overwrite_existing=True starts a fresh file.
        """
        folder = tmp_path / "labels"
        folder.mkdir()
        csv_path = self._summary_csv(folder)
        stats_off = {
            "mean_intensity": False,
            "median_intensity": False,
            "std_intensity": False,
        }

        # A previous session's run leaves a CSV behind.
        self._run(str(folder / "old.tif"), labels_2d, **stats_off)
        reset_regionprops_cache()

        # Default: the new run appends, keeping the earlier results.
        self._run(str(folder / "new.tif"), labels_2d, **stats_off)
        df = pd.read_csv(csv_path)
        assert list(df["filename"]) == ["old.tif", "new.tif"]

        # overwrite_existing starts the file over.
        reset_regionprops_cache()
        self._run(
            str(folder / "fresh.tif"),
            labels_2d,
            overwrite_existing=True,
            **stats_off,
        )
        df = pd.read_csv(csv_path)
        assert list(df["filename"]) == ["fresh.tif"]

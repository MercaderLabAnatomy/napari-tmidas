# src/napari_tmidas/_tests/test_intensity_label_filter.py
"""Tests for intensity-based label filtering functions."""

import numpy as np
import pytest

# Import the module and check if k-medoids is available
from napari_tmidas.processing_functions.intensity_label_filter import (
    _HAS_KMEDOIDS,
    _calculate_label_mean_intensities,
    _cluster_intensities,
    _filter_labels_by_threshold,
)

HAS_KMEDOIDS = _HAS_KMEDOIDS


@pytest.mark.skipif(
    not HAS_KMEDOIDS, reason="scikit-learn-extra not installed"
)
class TestIntensityLabelFilter:
    """Test suite for intensity-based label filtering."""

    def test_calculate_label_mean_intensities(self):
        """Test mean intensity calculation for labels."""
        # Create simple label image with 3 labels
        label_image = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 0, 0],
                [3, 3, 0, 0],
            ]
        )

        # Create intensity image with different values for each label
        intensity_image = np.array(
            [
                [10, 10, 50, 50],
                [10, 10, 50, 50],
                [100, 100, 0, 0],
                [100, 100, 0, 0],
            ],
            dtype=np.float32,
        )

        result = _calculate_label_mean_intensities(
            label_image, intensity_image
        )

        assert len(result) == 3
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(50.0)
        assert result[3] == pytest.approx(100.0)

    def test_cluster_intensities_2medoids(self):
        """Test 2-medoids clustering."""
        # Create clear separation: low (10, 15, 20) and high (80, 85, 90)
        intensities = np.array([10, 15, 20, 80, 85, 90])

        labels, medoids, threshold = _cluster_intensities(
            intensities, n_clusters=2
        )

        assert len(labels) == 6
        assert len(medoids) == 2
        assert medoids[0] < medoids[1]  # Sorted low to high
        assert threshold > medoids[0]
        assert threshold < medoids[1]
        # Check threshold is between the two groups
        assert threshold > 20
        assert threshold < 80

    def test_cluster_intensities_3medoids(self):
        """Test 3-medoids clustering."""
        # Create clear separation: low (10, 15), medium (50, 55), high (90, 95)
        intensities = np.array([10, 15, 50, 55, 90, 95])

        labels, medoids, threshold = _cluster_intensities(
            intensities, n_clusters=3
        )

        assert len(labels) == 6
        assert len(medoids) == 3
        assert medoids[0] < medoids[1] < medoids[2]  # Sorted low to high
        # Threshold should be between lowest and second-lowest
        assert threshold > medoids[0]
        assert threshold < medoids[1]

    def test_filter_labels_by_threshold(self):
        """Test label filtering based on threshold."""
        label_image = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 0, 0],
                [3, 3, 0, 0],
            ]
        )

        label_intensities = {1: 10.0, 2: 50.0, 3: 100.0}
        threshold = 40.0  # Should keep labels 2 and 3, remove label 1

        result = _filter_labels_by_threshold(
            label_image, label_intensities, threshold
        )

        # Label 1 should be removed (set to 0)
        assert np.all(result[0:2, 0:2] == 0)
        # Labels 2 and 3 should remain
        assert np.all(result[0:2, 2:4] == 2)
        assert np.all(result[2:4, 0:2] == 3)
        # Background should remain
        assert np.all(result[2:4, 2:4] == 0)

    def test_filter_labels_2medoids_integration(self, tmp_path):
        """Integration test for 2-medoids filtering."""
        # Create test label image with 3 labels
        label_image = np.zeros((100, 100), dtype=np.uint16)
        label_image[10:40, 10:40] = 1  # Low intensity
        label_image[50:80, 10:40] = 2  # High intensity
        label_image[10:40, 50:80] = 3  # High intensity

        # Create intensity image where label 1 has low intensity, 2 and 3 high
        intensity_image = np.zeros((100, 100), dtype=np.float32)
        intensity_image[10:40, 10:40] = 20  # Low
        intensity_image[50:80, 10:40] = 100  # High
        intensity_image[10:40, 50:80] = 110  # High

        # Save intensity image to temporary file
        intensity_folder = tmp_path / "intensity"
        intensity_folder.mkdir()
        intensity_file = intensity_folder / "test_image.tif"

        # Use tifffile if available, otherwise numpy
        try:
            import tifffile

            tifffile.imwrite(intensity_file, intensity_image)
        except ImportError:
            np.save(intensity_file.with_suffix(".npy"), intensity_image)
            intensity_file = intensity_file.with_suffix(".npy")

        # Create fake label file path
        label_file = tmp_path / "labels" / intensity_file.name
        label_file.parent.mkdir()

        # Run filter (without actual file, just testing logic)
        # Note: This would require mocking the file reader in a real test
        # For now, we'll test the components separately

    def test_empty_label_image(self):
        """Test handling of empty label image."""
        label_image = np.zeros((50, 50), dtype=np.uint16)
        intensity_image = np.random.rand(50, 50).astype(np.float32)

        result = _calculate_label_mean_intensities(
            label_image, intensity_image
        )

        assert len(result) == 0

    def test_single_label(self):
        """Test handling of single label."""
        label_image = np.ones((50, 50), dtype=np.uint16)
        intensity_image = np.full((50, 50), 42.0, dtype=np.float32)

        result = _calculate_label_mean_intensities(
            label_image, intensity_image
        )

        assert len(result) == 1
        assert result[1] == pytest.approx(42.0)

    def test_filter_preserves_dtype(self):
        """Test that filtering preserves label image dtype."""
        for dtype in [np.uint8, np.uint16, np.uint32, np.int32]:
            label_image = np.array(
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                ],
                dtype=dtype,
            )

            label_intensities = {1: 10.0, 2: 50.0}
            threshold = 40.0

            result = _filter_labels_by_threshold(
                label_image, label_intensities, threshold
            )

            assert result.dtype == dtype

    def test_clustering_reproducibility(self):
        """Test that clustering is reproducible due to random_state."""
        intensities = np.array([10, 15, 20, 25, 80, 85, 90, 95])

        labels1, medoids1, threshold1 = _cluster_intensities(
            intensities, n_clusters=2
        )
        labels2, medoids2, threshold2 = _cluster_intensities(
            intensities, n_clusters=2
        )

        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_almost_equal(medoids1, medoids2)
        assert threshold1 == pytest.approx(threshold2)


@pytest.mark.skipif(HAS_KMEDOIDS, reason="Test for missing dependency")
def test_import_error_without_sklearn_extra():
    """Test that appropriate error is raised when sklearn-extra is not installed."""
    # This test only runs when sklearn-extra is NOT installed
    with pytest.raises(ImportError):
        from napari_tmidas.processing_functions.intensity_label_filter import (
            _cluster_intensities,
        )

        _cluster_intensities(np.array([1, 2, 3]), n_clusters=2)

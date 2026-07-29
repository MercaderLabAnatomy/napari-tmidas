# src/napari_tmidas/_tests/test_intensity_label_filter.py
"""Tests for intensity-based label filtering functions."""

import numpy as np
import pytest
import tifffile

# Import the module and check if k-medoids is available
from napari_tmidas.processing_functions.intensity_label_filter import (
    _calculate_label_mean_intensities,
    _cluster_intensities,
    _collect_label_values,
    _convert_semantic_to_instance,
    _filter_labels_by_threshold,
    _kmedoids_1d,
    _locate_channel_axis,
    _PlaneReader,
    _resolve_intensity_source,
    _resolve_spatial_ndim,
    _smallest_label_dtype,
    filter_labels_by_intensity,
)

# Every shape the widget's dimension-order dropdown can produce
DIM_ORDER_SHAPES = [
    ("Auto", (8, 6)),
    ("YX", (8, 6)),
    ("CYX", (2, 8, 6)),
    ("TYX", (3, 8, 6)),
    ("ZYX", (4, 8, 6)),
    ("TCYX", (3, 2, 8, 6)),
    ("TZYX", (3, 4, 8, 6)),
    ("ZCYX", (4, 2, 8, 6)),
    ("TZCYX", (2, 3, 2, 8, 6)),
    ("TCZYX", (2, 2, 3, 8, 6)),
]


def _make_label_and_intensity(shape):
    """Three labels present in every frame, with low/high/high intensities."""
    labels = np.zeros(shape, dtype=np.uint16)
    intensity = np.zeros(shape, dtype=np.float32)
    labels[..., 0:2, 0:2] = 1
    labels[..., 3:5, 0:2] = 2
    labels[..., 0:2, 3:5] = 3
    intensity[..., 0:2, 0:2] = 10.0
    intensity[..., 3:5, 0:2] = 100.0
    intensity[..., 0:2, 3:5] = 110.0
    return labels, intensity


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


class TestDimensionOrderSupport:
    """All shapes offered by the dimension-order dropdown must be accepted."""

    @pytest.mark.parametrize("dim_order,shape", DIM_ORDER_SHAPES)
    def test_mean_intensities_any_shape(self, dim_order, shape):
        """Mean intensities are computed for 2-D up to 5-D images."""
        labels, intensity = _make_label_and_intensity(shape)
        spatial_ndim = _resolve_spatial_ndim(dim_order, len(shape))

        result = _calculate_label_mean_intensities(
            labels, intensity, spatial_ndim=spatial_ndim
        )

        assert set(result) == {1, 2, 3}
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(100.0)
        assert result[3] == pytest.approx(110.0)

    @pytest.mark.parametrize("dim_order,shape", DIM_ORDER_SHAPES)
    def test_filter_any_shape(self, dim_order, shape):
        """Filtering removes the low-intensity label and preserves shape/dtype."""
        labels, _ = _make_label_and_intensity(shape)

        result = _filter_labels_by_threshold(
            labels, {1: 10.0, 2: 100.0, 3: 110.0}, threshold=50.0
        )

        assert result.shape == labels.shape
        assert result.dtype == labels.dtype
        assert not np.any(result == 1)
        np.testing.assert_array_equal(result == 2, labels == 2)
        np.testing.assert_array_equal(result == 3, labels == 3)

    @pytest.mark.parametrize(
        "dim_order,ndim,expected",
        [
            ("Auto", 2, 2),
            ("Auto", 3, 3),
            ("Auto", 4, 3),
            ("Auto", 5, 3),
            ("YX", 2, 2),
            ("TYX", 3, 2),
            ("ZYX", 3, 3),
            ("TZYX", 4, 3),
            ("TCYX", 4, 2),
            # C sits between Z and YX, so the spatial block is only YX
            ("ZCYX", 4, 2),
            ("TZCYX", 5, 2),
            ("TCZYX", 5, 3),
            # Hint wider than the actual image (e.g. channel already extracted)
            ("TZYX", 2, 2),
        ],
    )
    def test_resolve_spatial_ndim(self, dim_order, ndim, expected):
        assert _resolve_spatial_ndim(dim_order, ndim) == expected

    def test_semantic_labels_not_merged_across_leading_axes(self):
        """Semantic → instance conversion labels each frame independently."""
        # One blob per frame, identical position: nD labelling would fuse them
        image = np.zeros((3, 8, 6), dtype=np.uint16)
        image[:, 0:2, 0:2] = 1

        result = _convert_semantic_to_instance(image, spatial_ndim=2)

        ids = np.unique(result[result != 0])
        assert len(ids) == 3  # one distinct ID per timepoint
        for t in range(3):
            assert len(np.unique(result[t][result[t] != 0])) == 1

    def test_semantic_labels_merged_within_spatial_block(self):
        """Within a 3-D block, connected voxels stay a single object."""
        image = np.zeros((4, 8, 6), dtype=np.uint16)
        image[:, 0:2, 0:2] = 1

        result = _convert_semantic_to_instance(image, spatial_ndim=3)

        assert len(np.unique(result[result != 0])) == 1

    def test_negative_labels_rejected(self):
        """Signed label images with negative values raise a clear error."""
        labels = np.array([[-1, 1], [1, 0]], dtype=np.int32)
        intensity = np.ones((2, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="negative"):
            _calculate_label_mean_intensities(labels, intensity)

    def test_label_missing_from_intensities_is_kept(self):
        """Labels without an intensity entry are left untouched."""
        labels = np.array([[1, 2], [3, 0]], dtype=np.uint16)

        result = _filter_labels_by_threshold(labels, {1: 10.0}, threshold=50.0)

        assert result[0, 0] == 0  # below threshold
        assert result[0, 1] == 2  # no entry → untouched
        assert result[1, 0] == 3


class TestMemoryBehaviour:
    """Large stacks must not allocate multiples of the image size."""

    @pytest.mark.parametrize(
        "max_label,current,expected",
        [
            # int64 tracking output with a few thousand labels → uint32 (2x)
            (1583, np.int64, np.uint32),
            (1583, np.uint64, np.uint32),
            (70000, np.int64, np.uint32),
            # uint32 is the floor: napari only recognises int32/uint32/
            # int64/uint64 as labels, so never narrow past it
            (3, np.int64, np.uint32),
            (1583, np.uint32, np.uint32),
            (1583, np.int32, np.int32),
            # IDs too large for uint32 keep the wide dtype
            (2**40, np.int64, np.int64),
            # already narrower than uint32 → left alone, never widened
            (1583, np.uint16, np.uint16),
            (3, np.uint8, np.uint8),
            # ...but must widen when the IDs no longer fit
            (300, np.uint8, np.uint16),
            (70000, np.uint16, np.uint32),
            # float labels are left untouched
            (5, np.float32, np.float32),
        ],
    )
    def test_smallest_label_dtype(self, max_label, current, expected):
        assert _smallest_label_dtype(max_label, np.dtype(current)) == np.dtype(
            expected
        )

    def test_narrowed_dtype_still_recognised_as_labels(self):
        """The narrowed output must survive napari's labels-vs-image guess."""
        from napari_tmidas._file_selector import is_label_image

        for current in (np.int64, np.uint64, np.int32, np.uint32):
            out_dtype = _smallest_label_dtype(1583, np.dtype(current))
            probe = np.zeros((2, 2), dtype=out_dtype)
            assert is_label_image(
                probe
            ), f"{current.__name__} → {out_dtype} is no longer seen as labels"

    def test_narrowing_preserves_label_ids(self):
        """Narrowing int64 → uint16 must not alter any label ID."""
        labels = np.zeros((2, 40, 40), dtype=np.int64)
        labels[:, 0:10, 0:10] = 1
        labels[:, 20:30, 0:10] = 60000

        result = _filter_labels_by_threshold(
            labels,
            {1: 100.0, 60000: 100.0},
            threshold=1.0,
            out_dtype=np.dtype(np.uint16),
            spatial_ndim=2,
        )

        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, labels.astype(np.uint16))

    def test_collect_label_values_matches_full_scan(self):
        """Block-wise scan must equal the naive np.unique it replaces."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 50, size=(4, 3, 20, 20), dtype=np.int64)

        for spatial_ndim in (2, 3):
            np.testing.assert_array_equal(
                _collect_label_values(image, spatial_ndim), np.unique(image)
            )

    def test_collect_label_values_all_background(self):
        image = np.zeros((3, 8, 8), dtype=np.uint16)

        np.testing.assert_array_equal(
            _collect_label_values(image, spatial_ndim=2), [0]
        )

    def test_filter_allocates_only_the_output(self):
        """Peak RSS growth must stay near one output array, not a multiple."""
        shape = (40, 200, 200)  # 1.6 M voxels, int64 = 12.8 MB
        labels = np.empty(shape, dtype=np.int64)
        labels.fill(0)
        for lab in range(1, 200):
            labels[:, (lab % 20) * 10 : (lab % 20) * 10 + 6, lab % 190] = lab
        intensities = {lab: float(lab) for lab in range(1, 200)}

        def rss():
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * 4096

        before = rss()
        result = _filter_labels_by_threshold(
            labels,
            intensities,
            threshold=100.0,
            out_dtype=np.dtype(np.uint16),
            spatial_ndim=2,
        )
        growth = rss() - before

        # Output is 2 bytes/voxel; allow generous headroom but far below the
        # 8 bytes/voxel a full-size int64 intermediate would add.
        assert result.dtype == np.uint16
        assert growth < labels.size * 4, (
            f"grew {growth / 1e6:.1f} MB for a {result.nbytes / 1e6:.1f} MB output"
        )


class TestIntensitySourceResolution:
    """Locating the intensity image must never silently pick the label file."""

    def test_finds_tif_by_known_suffix(self, tmp_path):
        (tmp_path / "movie.tif").write_bytes(b"x")
        label = tmp_path / "movie_hoct_tracked.tif"
        label.write_bytes(b"x")

        assert _resolve_intensity_source(label).name == "movie.tif"

    def test_prefers_zarr_over_tif(self, tmp_path):
        (tmp_path / "movie.zarr").mkdir()
        (tmp_path / "movie.tif").write_bytes(b"x")
        label = tmp_path / "movie_hoct_tracked.tif"
        label.write_bytes(b"x")

        assert _resolve_intensity_source(label).name == "movie.zarr"

    def test_explicit_suffix(self, tmp_path):
        (tmp_path / "sample.tif").write_bytes(b"x")
        label = tmp_path / "sample_my_custom_labels.tif"
        label.write_bytes(b"x")

        resolved = _resolve_intensity_source(
            label, label_suffix="_my_custom_labels"
        )
        assert resolved.name == "sample.tif"

    def test_raises_instead_of_using_label_as_its_own_intensity(
        self, tmp_path
    ):
        """The bug this replaces: clustering label IDs and calling it signal."""
        label = tmp_path / "movie_hoct_tracked.tif"
        label.write_bytes(b"x")  # no matching intensity file exists

        with pytest.raises(FileNotFoundError, match="No intensity image"):
            _resolve_intensity_source(label)

    def test_raises_when_suffix_does_not_match(self, tmp_path):
        (tmp_path / "movie.tif").write_bytes(b"x")
        label = tmp_path / "movie_unknown_convention.tif"
        label.write_bytes(b"x")

        with pytest.raises(FileNotFoundError, match="No intensity image"):
            _resolve_intensity_source(label)


class TestPlaneReader:
    """Every TIFF layout the reader may meet must address planes correctly."""

    def _check_all_planes(self, path, expected):
        reader = _PlaneReader(path)
        try:
            assert reader.shape == expected.shape
            for index in np.ndindex(*expected.shape[:-2]):
                np.testing.assert_array_equal(
                    reader.plane(index), expected[index]
                )
            return reader
        finally:
            reader.close()

    def test_imagej_hyperstack_5d(self, tmp_path):
        """A 5-D TZCYX ImageJ hyperstack, the layout seen in the wild."""
        rng = np.random.default_rng(0)
        data = (rng.random((3, 2, 2, 8, 8)) * 1000).astype(np.uint16)
        path = tmp_path / "hyperstack.tif"
        tifffile.imwrite(path, data, imagej=True)

        reader = self._check_all_planes(path, data)
        assert reader._array is not None  # uncompressed → memmap path

    def test_compressed_page_per_plane(self, tmp_path):
        """Compressed stacks cannot be memory-mapped; pages must be used."""
        rng = np.random.default_rng(1)
        data = (rng.random((3, 5, 16, 16)) * 1000).astype(np.uint16)
        path = tmp_path / "compressed.tif"
        tifffile.imwrite(
            path, data, compression="zlib", photometric="minisblack"
        )

        reader = _PlaneReader(path)
        try:
            assert reader._array is None, "compressed file should not memmap"
            assert reader._pages is not None, "should use the page path"
            for index in np.ndindex(*data.shape[:-2]):
                np.testing.assert_array_equal(
                    reader.plane(index), data[index]
                )
        finally:
            reader.close()

    def test_falls_back_when_neither_scheme_applies(
        self, tmp_path, monkeypatch
    ):
        """Pages not mapping 1:1 and no memmap must not reach the old path.

        This is the exact situation that raised
        ``IndexError: list index out of range`` — the reader used to hand an
        nD index tuple to tifffile's ``key=``, which expects page numbers.
        """
        import napari_tmidas.processing_functions.intensity_label_filter as mod

        rng = np.random.default_rng(2)
        data = (rng.random((3, 2, 2, 8, 8)) * 1000).astype(np.uint16)
        path = tmp_path / "awkward.tif"
        tifffile.imwrite(path, data, imagej=True)

        def no_memmap(*args, **kwargs):
            raise ValueError("memmap unavailable")

        monkeypatch.setattr(tifffile, "memmap", no_memmap)
        # Force the page count to disagree with the plane count
        monkeypatch.setattr(
            mod.np, "prod", lambda *a, **k: np.multiply.reduce(*a, **k) + 1
        )

        reader = _PlaneReader(path)
        try:
            for index in np.ndindex(*data.shape[:-2]):
                np.testing.assert_array_equal(
                    reader.plane(index), data[index]
                )
        finally:
            reader.close()

    def test_plain_2d(self, tmp_path):
        data = (np.arange(64).reshape(8, 8)).astype(np.uint16)
        path = tmp_path / "flat.tif"
        tifffile.imwrite(path, data)

        reader = _PlaneReader(path)
        try:
            np.testing.assert_array_equal(reader.plane(()), data)
        finally:
            reader.close()

    def test_zarr_source(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        rng = np.random.default_rng(2)
        data = (rng.random((2, 3, 8, 8)) * 255).astype(np.uint8)
        path = tmp_path / "vol.zarr"
        root = zarr.open(str(path), mode="w")
        root.create_array("s0", shape=data.shape, dtype=data.dtype)
        root["s0"][:] = data

        self._check_all_planes(path, data)


class TestChannelAxis:
    """Matching a TCZYX intensity image to a TZYX label image."""

    @pytest.mark.parametrize(
        "label_shape,intensity_shape,expected",
        [
            ((55, 40, 130, 130), (55, 2, 40, 130, 130), 1),  # the real case
            ((10, 20, 20), (10, 20, 20), None),  # already matching
            ((5, 8, 8), (3, 5, 8, 8), 0),  # channel first
            ((4, 6, 6), (4, 6, 6, 3), 3),  # channel last
        ],
    )
    def test_locate_channel_axis(
        self, label_shape, intensity_shape, expected
    ):
        assert (
            _locate_channel_axis(label_shape, intensity_shape) == expected
        )

    def test_irreconcilable_shapes_raise(self):
        with pytest.raises(ValueError, match="cannot be matched"):
            _locate_channel_axis((10, 20, 20), (10, 30, 30))

    def test_too_many_extra_axes_raise(self):
        with pytest.raises(ValueError, match="cannot be matched"):
            _locate_channel_axis((8, 8), (2, 3, 8, 8))


class TestStreamingPath:
    """End-to-end streaming, including the TZYX labels + TCZYX zarr case."""

    def _write_pair(self, tmp_path, with_channel_axis):
        """Labels TZYX, intensity either TZYX or TCZYX; label 1 low, 2/3 high."""
        shape = (3, 2, 16, 16)
        labels = np.zeros(shape, dtype=np.int64)
        labels[..., 0:4, 0:4] = 1
        labels[..., 6:10, 0:4] = 2
        labels[..., 0:4, 8:12] = 3

        if with_channel_axis:
            intensity = np.zeros((3, 2, 2, 16, 16), dtype=np.uint8)
            intensity[:, 0][..., 0:4, 0:4] = 10
            intensity[:, 0][..., 6:10, 0:4] = 200
            intensity[:, 0][..., 0:4, 8:12] = 220
            intensity[:, 1] = 99  # decoy channel, must not be measured
        else:
            intensity = np.zeros(shape, dtype=np.uint8)
            intensity[..., 0:4, 0:4] = 10
            intensity[..., 6:10, 0:4] = 200
            intensity[..., 0:4, 8:12] = 220

        tifffile.imwrite(tmp_path / "movie.tif", intensity)
        label_path = tmp_path / "movie_hoct_tracked.tif"
        tifffile.imwrite(label_path, labels)
        return label_path, labels

    @pytest.mark.parametrize("with_channel_axis", [False, True])
    def test_streaming_matches_expected(self, tmp_path, with_channel_axis):
        label_path, labels = self._write_pair(tmp_path, with_channel_axis)
        out_dir = tmp_path / "out"

        result = filter_labels_by_intensity(
            image=None,
            n_clusters=2,
            save_stats=False,
            intensity_channel=0,
            _source_filepath=str(label_path),
            _output_folder=str(out_dir),
            _output_suffix="_filtered",
        )

        assert isinstance(result, str)
        written = tifffile.imread(result)
        assert written.shape == labels.shape
        # Label 1 (intensity 10) removed; 2 and 3 (200/220) kept, IDs intact
        assert not np.any(written == 1)
        np.testing.assert_array_equal(written == 2, labels == 2)
        np.testing.assert_array_equal(written == 3, labels == 3)

    def test_streaming_matches_in_memory_path(self, tmp_path):
        """Streaming and in-memory must agree on the same data."""
        label_path, labels = self._write_pair(tmp_path, with_channel_axis=False)

        streamed = tifffile.imread(
            filter_labels_by_intensity(
                image=None,
                n_clusters=2,
                save_stats=False,
                _source_filepath=str(label_path),
                _output_folder=str(tmp_path / "out"),
                _output_suffix="_filtered",
            )
        )
        in_memory = filter_labels_by_intensity(
            labels.copy(),
            n_clusters=2,
            save_stats=False,
            dim_order="TZYX",
            _source_filepath=str(label_path),
        )

        np.testing.assert_array_equal(streamed, in_memory)

    def test_out_of_range_channel_raises(self, tmp_path):
        label_path, _ = self._write_pair(tmp_path, with_channel_axis=True)

        with pytest.raises(ValueError, match="out of range"):
            filter_labels_by_intensity(
                image=None,
                save_stats=False,
                intensity_channel=7,
                _source_filepath=str(label_path),
                _output_folder=str(tmp_path / "out"),
                _output_suffix="_filtered",
            )

    def test_output_dtype_is_recognised_as_labels(self, tmp_path):
        from napari_tmidas._file_selector import is_label_image

        label_path, _ = self._write_pair(tmp_path, with_channel_axis=False)
        result = filter_labels_by_intensity(
            image=None,
            save_stats=False,
            _source_filepath=str(label_path),
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_filtered",
        )

        assert is_label_image(tifffile.imread(result))

    def test_skip_load_is_declared(self):
        """Without this the worker would densely load the stack first."""
        assert getattr(filter_labels_by_intensity, "skip_load", False) is True


class TestKMedoids1D:
    """The built-in 1-D k-medoids replaces the unmaintained scikit-learn-extra."""

    def test_medoids_are_real_data_points(self):
        """Medoids must be actual observed values, not means."""
        values = np.array([10.0, 11.0, 12.0, 90.0, 91.0, 92.0])

        medoids = _kmedoids_1d(values, n_clusters=2)

        assert len(medoids) == 2
        for m in medoids:
            assert m in values

    def test_weighted_by_multiplicity(self):
        """Repeated values pull the medoid, as they do in true k-medoids."""
        # 100 copies of 10, one 20; plus a well-separated high cluster
        values = np.concatenate(
            [np.full(100, 10.0), [20.0], np.full(100, 90.0)]
        )

        medoids = _kmedoids_1d(values, n_clusters=2)

        assert medoids[0] == pytest.approx(10.0)
        assert medoids[1] == pytest.approx(90.0)

    def test_deterministic(self):
        """No random_state needed: seeding is quantile-based."""
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(10, 1, 200), rng.normal(90, 1, 200)])

        first = _kmedoids_1d(values, n_clusters=2)
        second = _kmedoids_1d(values, n_clusters=2)

        np.testing.assert_array_equal(first, second)

    def test_recovers_known_clusters(self):
        """Medoids land inside their true populations."""
        rng = np.random.default_rng(1)
        values = np.concatenate(
            [rng.normal(10, 1, 300), rng.normal(50, 1, 300), rng.normal(90, 1, 300)]
        )

        medoids = _kmedoids_1d(values, n_clusters=3)

        assert len(medoids) == 3
        assert abs(medoids[0] - 10) < 2
        assert abs(medoids[1] - 50) < 2
        assert abs(medoids[2] - 90) < 2

    def test_fewer_distinct_values_than_clusters(self):
        """Degenerate input returns the distinct values rather than erroring."""
        medoids = _kmedoids_1d(np.array([5.0, 5.0, 7.0]), n_clusters=3)

        np.testing.assert_array_equal(medoids, [5.0, 7.0])

    def test_scales_without_distance_matrix(self):
        """100k values would need an 80 GB n×n matrix; this must stay cheap."""
        rng = np.random.default_rng(2)
        values = np.concatenate(
            [rng.normal(10, 1, 50_000), rng.normal(90, 1, 50_000)]
        )

        medoids = _kmedoids_1d(values, n_clusters=2)

        assert abs(medoids[0] - 10) < 1
        assert abs(medoids[1] - 90) < 1

    def test_single_intensity_population_raises(self):
        """All labels identical → nothing to separate, with a clear message."""
        with pytest.raises(ValueError, match="same"):
            _cluster_intensities(np.full(10, 42.0), n_clusters=2)

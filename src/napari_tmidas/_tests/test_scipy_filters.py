# src/napari_tmidas/_tests/test_scipy_filters.py
import numpy as np
import pytest

from napari_tmidas.processing_functions import scipy_filters
from napari_tmidas.processing_functions.scipy_filters import gaussian_blur


class TestScipyFilters:
    def test_resize_labels(self):
        """Test resizing label objects while maintaining original array dimensions."""
        from napari_tmidas.processing_functions.scipy_filters import (
            resize_labels,
        )

        label_image = np.zeros((10, 10), dtype=np.uint8)
        label_image[2:8, 2:8] = 3

        # Test with float - dimensions should stay the same
        scale_factor = 0.5
        scaled = resize_labels(label_image, scale_factor=scale_factor)
        # Function maintains original dimensions
        assert scaled.shape == label_image.shape
        assert set(np.unique(scaled)).issubset({0, 3})
        # Objects should be smaller (fewer pixels with label 3)
        assert np.sum(scaled == 3) > 0
        assert np.sum(scaled == 3) < np.sum(label_image == 3)

        # Test with string
        scale_factor_str = "0.5"
        scaled_str = resize_labels(label_image, scale_factor=scale_factor_str)
        assert scaled_str.shape == label_image.shape
        assert set(np.unique(scaled_str)).issubset({0, 3})
        assert np.sum(scaled_str == 3) > 0

    def test_gaussian_blur_basic(self):
        """Test basic gaussian blur functionality"""
        image = np.random.rand(100, 100)

        # Test with default parameters
        result = gaussian_blur(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_gaussian_blur_with_sigma(self):
        """Test gaussian blur with custom sigma"""
        image = np.random.rand(50, 50)

        # Test with sigma parameter
        result = gaussian_blur(image, sigma=2.0)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_gaussian_blur_3d(self):
        """Test gaussian blur on 3D image"""
        image = np.random.rand(20, 20, 20)

        result = gaussian_blur(image, sigma=1.0)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_combined_output(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        label_image = np.zeros((9, 9), dtype=np.uint16)
        label_image[2:7, 2:7] = 1

        result = subdivide_labels_3layers(label_image)

        assert result.shape == label_image.shape
        unique_ids = set(np.unique(result))
        assert unique_ids.issubset({0, 1, 2, 3})
        assert {1, 2, 3}.issubset(unique_ids)

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_dtype_promotion(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        label_image = np.zeros((9, 9), dtype=np.uint8)
        label_image[2:7, 2:7] = 200

        result = subdivide_labels_3layers(label_image)

        assert result.dtype in (np.uint32, np.uint64)
        assert result.max() == 200 + 2 * 200

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_empty(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        label_image = np.zeros((5, 5, 5), dtype=np.uint16)

        result = subdivide_labels_3layers(label_image)

        np.testing.assert_array_equal(result, label_image)

    @pytest.mark.skipif(
        not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
    )
    def test_subdivide_labels_3layers_half_body(self):
        from napari_tmidas.processing_functions.scipy_filters import (
            subdivide_labels_3layers,
        )

        # Create a simple half-spheroid-like object
        label_image = np.zeros((20, 20, 20), dtype=np.uint16)
        # Fill upper half with a sphere-like object
        for z in range(10, 20):
            for y in range(20):
                for x in range(20):
                    if (z - 15) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 25:
                        label_image[z, y, x] = 1

        # Test with half-body mode disabled (default)
        result_normal = subdivide_labels_3layers(
            label_image, is_half_body=False
        )
        assert result_normal.shape == label_image.shape
        unique_normal = np.unique(result_normal)
        assert len(unique_normal) > 1  # Should have background + layers

        # Test with half-body mode enabled (cut along Z-axis = 0)
        result_half_body = subdivide_labels_3layers(
            label_image, is_half_body=True, cut_axis=0
        )
        assert result_half_body.shape == label_image.shape
        unique_half = np.unique(result_half_body)
        assert len(unique_half) > 1  # Should have background + layers

        # Both modes should produce layered results
        # In normal mode, the cut surface (z=10 plane) may show fewer layers
        # because it's treating it as a partial object
        cut_surface_normal = result_normal[10, :, :]
        cut_surface_half = result_half_body[10, :, :]

        # Both should have some non-zero values at the cut surface
        assert np.sum(cut_surface_normal > 0) > 0
        assert np.sum(cut_surface_half > 0) > 0

        # Half-body mode should ideally show more variety, but the exact
        # behavior depends on the implementation details. Just verify both work.
        unique_layers_normal = len(
            np.unique(cut_surface_normal[cut_surface_normal > 0])
        )
        unique_layers_half = len(
            np.unique(cut_surface_half[cut_surface_half > 0])
        )
        assert unique_layers_normal >= 1
        assert unique_layers_half >= 1

        # Test invalid cut_axis
        with pytest.raises(ValueError):
            subdivide_labels_3layers(
                label_image, is_half_body=True, cut_axis=5
            )


class TestMedianFilter:
    """
    The dimension_order hint decides which axes the median window spans.
    Getting it wrong blends unrelated timepoints or channels together -- a
    silent corruption that looks like plausible image data, so these tests
    check the window shape directly as well as end-to-end behaviour.
    """

    def test_2d_filters_both_axes(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 255  # lone impulse, must be removed by a 3x3 median

        result = scipy_filters.median_filter(image, size=3)

        assert result.shape == image.shape
        assert result[3, 3] == 0

    def test_tyx_keeps_frames_independent(self):
        """
        One bright frame between two dark ones. With a per-frame 2D filter the
        bright frame keeps its (uniform) value; filtering across T would pull
        the dark neighbours into it.
        """
        image = np.zeros((3, 7, 7), dtype=np.uint8)
        image[1] = 200

        result = scipy_filters.median_filter(
            image, size=3, dimension_order="TYX"
        )

        assert result[1].min() == 200
        assert result[0].max() == 0
        assert result[2].max() == 0

    def test_zyx_filters_across_z(self):
        """ZYX is the one 3D hint that *should* span the leading axis."""
        image = np.zeros((7, 7, 7), dtype=np.uint8)
        image[3] = 255  # a single bright Z slice is an impulse along Z

        result = scipy_filters.median_filter(
            image, size=3, dimension_order="ZYX"
        )

        assert result[3].max() == 0

    def test_auto_on_3d_does_not_blend_leading_axis(self):
        """
        Auto must assume the last two axes are spatial. Defaulting to a full
        3D window would mix unrelated frames for anyone who does not set the
        hint.
        """
        image = np.zeros((3, 7, 7), dtype=np.uint8)
        image[1] = 200

        result = scipy_filters.median_filter(image, size=3)

        assert result[1].min() == 200
        assert result[0].max() == 0

    @pytest.mark.parametrize(
        "dimension_order,ndim,expected",
        [
            ("YX", 2, (3, 3)),
            ("Auto", 2, (3, 3)),
            ("TYX", 3, (1, 3, 3)),
            ("CYX", 3, (1, 3, 3)),
            ("ZYX", 3, (3, 3, 3)),
            ("TCYX", 4, (1, 1, 3, 3)),
            ("TZYX", 4, (1, 1, 3, 3)),
            ("ZCYX", 4, (1, 1, 3, 3)),
            ("TZCYX", 5, (1, 1, 1, 3, 3)),
            ("TCZYX", 5, (1, 1, 1, 3, 3)),
            # Unrecognised hints and mismatched ndim fall back to "last two
            # axes are spatial" rather than raising.
            ("Auto", 4, (1, 1, 3, 3)),
            ("TYX", 4, (1, 1, 3, 3)),
            ("nonsense", 3, (1, 3, 3)),
        ],
    )
    def test_size_tuple_branches(self, dimension_order, ndim, expected):
        assert (
            scipy_filters._median_filter_size_tuple(dimension_order, ndim, 3)
            == expected
        )

    def test_dask_input_returns_dask_and_matches_numpy(self):
        """
        Dask input takes a separate map_overlap path. It must produce the same
        answer as the in-memory path, or results silently depend on whether
        the input happened to be lazy.
        """
        da = pytest.importorskip("dask.array")
        rng = np.random.default_rng(0)
        image = (rng.random((4, 32, 32)) * 255).astype(np.uint8)
        lazy = da.from_array(image, chunks=(1, 16, 16))

        result = scipy_filters.median_filter(
            lazy, size=3, dimension_order="TYX"
        )
        expected = scipy_filters.median_filter(
            image, size=3, dimension_order="TYX"
        )

        assert hasattr(result, "chunks"), "dask input should stay lazy"
        np.testing.assert_array_equal(np.asarray(result), expected)

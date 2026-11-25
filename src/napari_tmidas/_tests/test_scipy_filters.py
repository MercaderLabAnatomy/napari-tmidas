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

        result = subdivide_labels_3layers(
            label_image, dimension_order="ZYX"
        )

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
            label_image, is_half_body=False, dimension_order="ZYX"
        )
        assert result_normal.shape == label_image.shape
        unique_normal = np.unique(result_normal)
        assert len(unique_normal) > 1  # Should have background + layers

        # Test with half-body mode enabled (cut along Z-axis = 0)
        result_half_body = subdivide_labels_3layers(
            label_image, is_half_body=True, cut_axis=0, dimension_order="ZYX"
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
                label_image,
                is_half_body=True,
                cut_axis=5,
                dimension_order="ZYX",
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


@pytest.mark.skipif(
    not scipy_filters.SCIPY_AVAILABLE, reason="SciPy is required"
)
class TestZoomScalesSpatialAxesOnly:
    """
    Resize Labels and Subdivide Labels rescale objects with ndimage.zoom,
    which scales every axis it is handed.  Given a whole TZYX stack it
    scaled time too: at 0.5, half the timepoints vanished and each surviving
    frame held another frame's object -- with the output shape unchanged and
    no error.  The contract pinned here: T and C are never scaled, and each
    timepoint comes out exactly as if it had been processed on its own.
    """

    @staticmethod
    def _per_frame_objects(shape):
        labels = np.zeros(shape, dtype=np.uint32)
        for t in range(shape[0]):
            labels[t, ..., 6:26, 6:26] = t + 1
        return labels

    @pytest.mark.parametrize("scale", [0.5, 2.0])
    @pytest.mark.parametrize(
        "order,shape,block_order",
        [
            ("TZYX", (6, 8, 32, 32), "ZYX"),
            ("TYX", (6, 32, 32), "YX"),
            ("ZCYX", (8, 3, 32, 32), None),  # C independent, not leading
            ("CZYX", (3, 8, 32, 32), "ZYX"),
            ("TCZYX", (6, 8, 32, 32), "ZYX"),  # stale C after extraction
        ],
    )
    def test_resize_matches_each_frame_on_its_own(
        self, scale, order, shape, block_order
    ):
        labels = self._per_frame_objects(shape)

        out = scipy_filters.resize_labels(
            labels, scale_factor=scale, dimension_order=order
        )

        assert out.shape == labels.shape
        assert out.dtype == labels.dtype
        if block_order is None:
            # ZCYX: each channel is a ZYX volume at out[:, c].
            for c in range(shape[1]):
                np.testing.assert_array_equal(
                    out[:, c],
                    scipy_filters.resize_labels(
                        labels[:, c], scale_factor=scale, dimension_order="ZYX"
                    ),
                )
            return
        for t in range(shape[0]):
            np.testing.assert_array_equal(
                out[t],
                scipy_filters.resize_labels(
                    labels[t], scale_factor=scale, dimension_order=block_order
                ),
            )
            # Every frame keeps its own object, and only its own.  (At 2x
            # the object can fill the frame, so background is not assured.)
            assert set(np.unique(out[t]).tolist()) - {0} == {t + 1}

    def test_subdivide_matches_each_frame_with_a_global_offset(self):
        """
        Layer IDs are n, n + max and n + 2*max.  `max` must be the stack's,
        not the frame's, or one tracked object gets different shell IDs in
        different frames.  An empty frame stays empty.
        """
        labels = self._per_frame_objects((6, 8, 32, 32))
        labels[3] = 0
        top = int(labels.max())

        out = scipy_filters.subdivide_labels_3layers(
            labels, dimension_order="TZYX"
        )

        assert out.shape == labels.shape
        assert not out[3].any()
        for t in (0, 1, 2, 4, 5):
            n = t + 1
            assert set(np.unique(out[t]).tolist()) == {
                0,
                n,
                n + top,
                n + 2 * top,
            }
        # Frame 5 holds the stack maximum, so its own max equals the global
        # one and its result must match processing that volume alone.
        np.testing.assert_array_equal(
            out[5],
            scipy_filters.subdivide_labels_3layers(
                labels[5], dimension_order="ZYX"
            ),
        )

    def test_subdivide_half_body_per_frame(self):
        labels = np.zeros((4, 10, 24, 24), dtype=np.uint16)
        labels[:, 0:5, 4:20, 4:20] = 1

        out = scipy_filters.subdivide_labels_3layers(
            labels, is_half_body=True, cut_axis=0, dimension_order="TZYX"
        )

        alone = scipy_filters.subdivide_labels_3layers(
            labels[0], is_half_body=True, cut_axis=0, dimension_order="ZYX"
        )
        for t in range(4):
            np.testing.assert_array_equal(out[t], alone)

    def test_cut_axis_counts_within_the_spatial_block(self):
        """cut_axis is documented as 0=Z, 1=Y, 2=X; on TZYX there is no
        fourth spatial axis to cut along."""
        labels = np.ones((2, 4, 8, 8), dtype=np.uint16)
        with pytest.raises(ValueError, match="cut_axis must be between 0 and 2"):
            scipy_filters.subdivide_labels_3layers(
                labels, is_half_body=True, cut_axis=3, dimension_order="TZYX"
            )

    @pytest.mark.parametrize("shape", [(4, 16, 16), (3, 4, 16, 16)])
    @pytest.mark.parametrize("order", ["Auto", "", None, "TZYX"])
    def test_an_unresolvable_order_is_refused(self, shape, order):
        """3D may be ZYX or TYX and 4D always hides a T or C axis, so
        without a matching hint there is nothing safe to guess."""
        if order == "TZYX" and len(shape) == 4:
            pytest.skip("TZYX resolves a 4D stack")
        labels = np.ones(shape, dtype=np.uint16)
        for call in (
            lambda: scipy_filters.resize_labels(
                labels, scale_factor=0.5, dimension_order=order
            ),
            lambda: scipy_filters.subdivide_labels_3layers(
                labels, dimension_order=order
            ),
        ):
            with pytest.raises(ValueError, match="dimension order"):
                call()

    def test_every_dropdown_order_is_resolvable(self):
        """The dropdown and the set of orders the label functions accept
        are separate lists; CZYX was offered by one and missing from the
        other, so picking it failed every run."""
        from napari_tmidas._file_selector import DIMENSION_ORDER_OPTIONS
        from napari_tmidas.processing_functions.skimage_filters import (
            _DIMENSION_ORDER_AXES,
        )

        assert set(DIMENSION_ORDER_OPTIONS) - {"Auto"} == _DIMENSION_ORDER_AXES

    @pytest.mark.parametrize(
        "order,shape",
        [("TZYX", (1, 8, 32, 32)), ("TYX", (1, 32, 32)), ("CZYX", (1, 8, 32, 32))],
    )
    def test_a_single_timepoint_keeps_its_axis(self, order, shape):
        """One timepoint is one spatial block, but stack[0] -- the T axis
        indexed away.  Returning that block's result as the whole answer
        saved a (1, Z, Y, X) stack as (Z, Y, X)."""
        labels = self._per_frame_objects(shape)
        resized = scipy_filters.resize_labels(
            labels, scale_factor=0.5, dimension_order=order
        )
        layered = scipy_filters.subdivide_labels_3layers(
            labels, dimension_order=order
        )
        assert resized.shape == labels.shape
        assert layered.shape == labels.shape
        np.testing.assert_array_equal(
            resized[0],
            scipy_filters.resize_labels(
                labels[0], scale_factor=0.5, dimension_order=order[1:]
            ),
        )

    def test_2d_needs_no_hint(self):
        labels = np.zeros((16, 16), dtype=np.uint16)
        labels[4:12, 4:12] = 2
        out = scipy_filters.resize_labels(labels, scale_factor=0.5)
        assert 0 < (out == 2).sum() < (labels == 2).sum()

    @pytest.mark.parametrize(
        "func",
        [
            scipy_filters.resize_labels,
            scipy_filters.subdivide_labels_3layers,
        ],
    )
    def test_the_worker_passes_the_dropdown_hint(self, func):
        """The worker forwards the Dimension Order dropdown only to functions
        that declare it, and the widget warns on 'Auto' only for those."""
        from napari_tmidas._file_selector import function_uses_dimension_order

        assert function_uses_dimension_order(func)

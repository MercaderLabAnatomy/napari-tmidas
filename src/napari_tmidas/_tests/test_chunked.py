"""
Tests for the @chunked decorator that makes processing functions block-wise.

The failure mode being guarded against is silent: a function that stops being
lazy still produces correct output, it just materialises the whole stack again
and OOMs on a big file.  So these check the wiring (does the worker see it as
lazy-capable, does a lazy input stay lazy, is it actually split into more than
one block) as well as the results.
"""

import inspect

import dask.array as da
import numpy as np
import pytest

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import (
    discover_and_load_processing_functions,
)
from napari_tmidas.processing_functions._chunked import (
    _block_chunks,
    chunked,
    independent_leading,
    plane_wise_only,
)

discover_and_load_processing_functions()

# (registry name, fixture kind, params) for every function converted so far.
CONVERTED = [
    ("Labels to Binary", "labels", {}),
    ("Invert Binary Labels", "labels", {}),
    ("Filter Label by ID", "labels", {"label_id": 3}),
    ("Gamma Correction", "intensity", {"gamma": 1.4}),
    ("Invert Image", "intensity", {}),
    ("Manual Thresholding (8-bit)", "intensity", {}),
    ("Rolling Ball Background Subtraction", "intensity", {"radius": 8}),
]

SHAPE = (6, 3, 64, 64)


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    intensity = rng.integers(0, 4000, SHAPE, dtype=np.uint16)
    labels = np.zeros(SHAPE, dtype=np.uint32)
    for label in range(1, 9):
        t, z = rng.integers(0, SHAPE[0]), rng.integers(0, SHAPE[1])
        y, x = rng.integers(0, 40), rng.integers(0, 40)
        labels[t, z, y : y + 20, x : x + 20] = label
    return {"intensity": intensity, "labels": labels}


def _func(name):
    return BatchProcessingRegistry.get_function_info(name)["func"]


class TestConvertedFunctions:
    @pytest.mark.parametrize("name,kind,params", CONVERTED)
    def test_lazy_result_is_identical_to_dense(
        self, name, kind, params, images
    ):
        """
        Blocks are cut along leading axes only, so each block holds whole YX
        planes and the body sees exactly what it saw densely.  That makes
        byte-identity the correct expectation, not approximate equality.

        dtype is asserted separately because assert_array_equal ignores it,
        and the lazy dtype is only *declared* from a one-block probe.  A
        divergence there would silently narrow a label result past the uint32
        floor -- which loads as grayscale instead of labels -- with the value
        comparison still green.
        """
        dense_in = images[kind]
        expected = _func(name)(dense_in, **params)

        lazy = da.from_array(dense_in, chunks=(1, 1, -1, -1))
        result = _func(name)(lazy, _source_filepath="x.zarr", **params)

        assert hasattr(result, "compute"), "lazy input must stay lazy"
        assert result.dtype == expected.dtype, "declared dtype must match"
        computed = np.asarray(result.compute())
        assert computed.dtype == expected.dtype, "computed dtype must match"
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.parametrize("name,kind,params", CONVERTED)
    def test_worker_sees_it_as_lazy_capable(self, name, kind, params):
        """
        The worker decides whether to keep the input lazy by looking for
        _source_filepath in the signature.  functools.wraps sets __wrapped__,
        which inspect.signature follows, so without an explicit __signature__
        the decorator would be invisible here and the input densified.
        """
        assert "_source_filepath" in inspect.signature(_func(name)).parameters

    @pytest.mark.parametrize("name,kind,params", CONVERTED)
    def test_dense_input_passes_straight_through(
        self, name, kind, params, images
    ):
        """Small inputs must keep the fast dense path, not build a graph."""
        result = _func(name)(images[kind], **params)
        assert not hasattr(result, "compute")

    @pytest.mark.parametrize("name,kind,params", CONVERTED)
    def test_result_is_split_into_several_blocks(
        self, name, kind, params, images
    ):
        """
        A single-block result would be correct but would still materialise
        everything at once -- the bug this decorator exists to prevent.
        """
        lazy = da.from_array(images[kind], chunks=(1, 1, -1, -1))
        result = _func(name)(lazy, _source_filepath="x.zarr", **params)

        assert result.numblocks[0] > 1
        # The spatial plane must never be split.
        assert result.numblocks[-2:] == (1, 1)


class TestDimensionOrderSensitive:
    """
    Functions whose behaviour depends on the dimension_order hint may only be
    split along axes that hint says are independent -- and the sets differ
    per function.  Otsu treats ZYX as per-plane (it thresholds each YX slice
    of a Z stack); Gaussian treats the same hint as a 3D blur, which couples
    Z.  Getting this wrong changes results rather than crashing, so each case
    asserts both the laziness decision and byte-identity.
    """

    # (name, ndim, dimension_order, must_be_lazy)
    CASES = [
        ("Gaussian Blur", 4, "TZYX", True),
        ("Gaussian Blur", 3, "TYX", True),
        ("Gaussian Blur", 3, "ZYX", False),  # 3D blur couples Z
        ("Gaussian Blur", 4, "Auto", False),  # couples every axis
        ("Otsu Thresholding (semantic)", 4, "TZYX", True),
        ("Otsu Thresholding (semantic)", 3, "ZYX", True),  # per-plane here
        ("Otsu Thresholding (semantic)", 4, "Auto", False),  # global threshold
        ("Otsu Thresholding (instance)", 4, "TZYX", True),
        ("Otsu Thresholding (instance)", 3, "ZYX", False),  # Z kept whole
        ("Binary to Labels", 4, "TZYX", True),
        ("Semantic to Instance Segmentation", 4, "TZYX", True),
    ]

    @pytest.mark.parametrize("name,ndim,order,must_be_lazy", CASES)
    def test_matches_dense_and_makes_the_right_choice(
        self, name, ndim, order, must_be_lazy
    ):
        rng = np.random.default_rng(1)
        shape = (6, 3, 48, 48) if ndim == 4 else (6, 48, 48)
        image = rng.integers(0, 4000, shape, dtype=np.uint16)

        expected = _func(name)(image, dimension_order=order)
        lazy = da.from_array(image, chunks=(1,) * (ndim - 2) + (-1, -1))
        result = _func(name)(
            lazy, _source_filepath="x.zarr", dimension_order=order
        )

        assert hasattr(result, "compute") is must_be_lazy
        got = np.asarray(
            result.compute() if hasattr(result, "compute") else result
        )
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)

    def test_hint_of_the_wrong_rank_falls_back(self):
        """A stale 5D hint on a 4D array must not be trusted."""
        rng = np.random.default_rng(2)
        image = rng.integers(0, 255, (4, 2, 32, 32), dtype=np.uint32)
        lazy = da.from_array(image, chunks=(1, 1, -1, -1))

        result = _func("Binary to Labels")(
            lazy, _source_filepath="x.zarr", dimension_order="TCZYX"
        )
        assert not hasattr(result, "compute")


class TestShapeChangingFunctions:
    """
    Two different shapes of "the output isn't the input shape", which need
    two different mechanisms.

    A per-plane *resize* still computes each block independently, so it maps
    block-for-block — only the output chunking has to state the new trailing
    extent.  A *projection* reduces along an axis blocks are cut on, so its
    answer combines blocks; that one needs Dask's own tree reduction, which
    is why those functions are only opted in to a lazy input rather than
    wrapped.
    """

    @pytest.mark.parametrize("scale", [0.5, 2.0])
    def test_resize_matches_dense_and_stays_lazy(self, scale):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 4000, (4, 3, 32, 32), dtype=np.uint16)
        name = "Resize Image by YX Scale (skimage)"

        expected = _func(name)(image, scale_factor=scale)
        lazy = da.from_array(image, chunks=(1, 1, -1, -1))
        result = _func(name)(
            lazy, _source_filepath="x.zarr", scale_factor=scale
        )

        assert hasattr(result, "compute")
        assert result.shape == expected.shape
        # Leading axes stay split; the resized plane stays in one piece.
        assert result.numblocks[0] > 1
        assert result.numblocks[-2:] == (1, 1)
        assert result.dtype == expected.dtype
        computed = np.asarray(result.compute())
        assert computed.dtype == expected.dtype
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.parametrize(
        "name,shape",
        [
            ("Max Z Projection", (5, 32, 32)),
            ("Max Z Projection (TZYX)", (4, 5, 32, 32)),
        ],
    )
    def test_projection_reduces_lazily(self, name, shape):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 4000, shape, dtype=np.uint16)

        expected = _func(name)(image)
        lazy = da.from_array(image, chunks=(1,) * (len(shape) - 2) + (-1, -1))
        result = _func(name)(lazy, _source_filepath="x.zarr")

        assert hasattr(result, "compute"), "projection must stay lazy"
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        computed = np.asarray(result.compute())
        assert computed.dtype == expected.dtype
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.parametrize(
        "name",
        [
            "Max Z Projection",
            "Max Z Projection (TZYX)",
            "Resize Image by YX Scale (skimage)",
        ],
    )
    def test_worker_sees_it_as_lazy_capable(self, name):
        assert "_source_filepath" in inspect.signature(_func(name)).parameters

    def test_axis_dropping_body_still_falls_back(self):
        """
        @chunked must not try to map a body that removes an axis: it cannot
        know *which* axis went, and guessing would mis-assemble the result.
        """

        @chunked(trailing_whole=3)
        def drop_one(image):
            return image.max(axis=-3)

        lazy = da.from_array(np.ones((8, 4, 8, 8), dtype=np.uint8), chunks=1)
        assert not hasattr(drop_one(lazy, _source_filepath="x"), "compute")


class TestIndependentLeading:
    """Mirrors _iter_dimension_blocks: T/C independent, Z stays with YX."""

    @pytest.mark.parametrize(
        "order,ndim,expected",
        [
            ("TZYX", 4, 3),  # split T only
            ("TCZYX", 5, 3),  # split T and C
            ("TYX", 3, 2),
            ("CYX", 3, 2),
            ("ZYX", 3, 3),  # Z is not independent -> dense
            ("ZCYX", 4, 4),  # C is independent but not leading -> dense
            ("Auto", 4, 4),
        ],
    )
    def test_resolves(self, order, ndim, expected):
        assert independent_leading()(order, ndim) == expected

    def test_never_splits_z(self):
        """The regression this guards: one label per Z slice instead of one
        label per 3D object."""
        assert independent_leading()("TZYX", 4) >= 3


class TestBlockChunks:
    def test_trailing_axes_are_kept_whole(self):
        chunks = _block_chunks((10, 8, 256, 256), 2, keep=2, block_bytes=10**6)
        assert chunks[-2:] == (-1, -1)

    def test_keep_three_leaves_z_intact(self):
        chunks = _block_chunks((10, 8, 256, 256), 2, keep=3, block_bytes=10**6)
        assert chunks[-3:] == (-1, -1, -1)
        assert chunks[0] >= 1

    def test_block_fits_the_budget(self):
        shape, itemsize = (100, 8, 256, 256), 2
        budget = 4 * 1024 * 1024
        chunks = _block_chunks(shape, itemsize, keep=2, block_bytes=budget)
        plane = 256 * 256 * itemsize
        assert chunks[0] * chunks[1] * plane <= budget

    def test_single_leading_axis_is_not_collapsed_into_one_block(self):
        """
        With only one leading axis to split (keep=3 on TZYX), sizing purely
        by the byte budget grew that axis to cover the whole array whenever
        the array was smaller than the budget.  One block materialises
        everything at once and leaves Dask a single task to schedule, which
        is the opposite of the point.
        """
        shape, itemsize = (40, 8, 128, 128), 4
        chunks = _block_chunks(
            shape, itemsize, keep=3, block_bytes=64 * 1024 * 1024
        )
        assert chunks[0] < shape[0]
        assert shape[0] // chunks[0] >= 8

    def test_budget_still_caps_large_inputs(self):
        """The >=8 blocks target must never override the memory budget."""
        shape, itemsize = (4000, 8, 512, 512), 2
        budget = 16 * 1024 * 1024
        chunks = _block_chunks(shape, itemsize, keep=3, block_bytes=budget)
        volume = 8 * 512 * 512 * itemsize
        assert chunks[0] * volume <= max(budget, volume)

    def test_oversized_unit_still_yields_one_unit(self):
        """A single plane larger than the budget cannot be split further."""
        chunks = _block_chunks((10, 4096, 4096), 2, keep=2, block_bytes=1024)
        assert chunks == (1, -1, -1)


class TestPlaneWiseOnly:
    ORDERS = frozenset({"TYX", "TZYX"})

    def test_per_plane_hint_allows_splitting(self):
        assert plane_wise_only(self.ORDERS)("TZYX", 4) == 2

    @pytest.mark.parametrize("order", ["ZYX", "Auto", "", None])
    def test_coupled_or_unknown_hint_keeps_everything(self, order):
        """
        A 3D or unresolved hint means the filter may couple the axes it would
        be split along, so it must fall back to the dense path rather than
        silently change the result.
        """
        assert plane_wise_only(self.ORDERS)(order, 4) == 4

    def test_hint_of_the_wrong_rank_keeps_everything(self):
        assert plane_wise_only(self.ORDERS)("TZYX", 3) == 3


class TestChunkedFallbacks:
    def test_shape_changing_body_falls_back_to_dense(self):
        """
        map_blocks would reassemble a projection into the wrong shape, so a
        body whose output shape differs must not be mapped block-wise.
        """

        @chunked(trailing_whole=2)
        def project(image):
            return image.max(axis=0)

        lazy = da.from_array(np.ones((4, 8, 8), dtype=np.uint8), chunks=1)
        result = project(lazy, _source_filepath="x.zarr")

        assert not hasattr(result, "compute")
        assert result.shape == (8, 8)

    def test_body_is_still_callable_directly(self):
        @chunked(trailing_whole=2)
        def double(image, factor=2):
            return image * factor

        assert double.__chunked_body__(np.ones((2, 2)), factor=3)[0, 0] == 3

    def test_keeping_every_axis_computes_densely(self):
        @chunked(trailing_whole=3)
        def identity(image):
            return image

        lazy = da.from_array(np.ones((4, 8, 8), dtype=np.uint8), chunks=1)
        assert not hasattr(identity(lazy, _source_filepath="x"), "compute")

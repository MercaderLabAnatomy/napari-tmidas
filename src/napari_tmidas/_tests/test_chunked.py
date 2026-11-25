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


class TestWholeStackRemaps:
    """
    Functions that need one whole-stack quantity (per-slice areas, a max
    label) and are otherwise an index remap or mask.  They stay lazy by
    computing those few numbers in one pass and expressing the rest as Dask
    ops, so the cost of getting it wrong is an off-by-one slice -- which
    byte-identity against the dense path catches.
    """

    @staticmethod
    def _labels(shape, dtype=np.uint32, seed=0):
        rng = np.random.default_rng(seed)
        labels = np.zeros(shape, dtype=dtype)
        # Objects of different sizes in different slices along every axis, so
        # the largest-area slice is unique and off-centre.
        for label in range(1, 7):
            starts = [int(rng.integers(0, s - 2)) for s in shape]
            sizes = [int(rng.integers(1, max(2, s // 2))) for s in shape]
            index = tuple(slice(a, a + b) for a, b in zip(starts, sizes))
            labels[index] = label
        return labels

    @staticmethod
    def _run(name, dense, **params):
        expected = _func(name)(dense, **params)
        lazy = da.from_array(
            dense, chunks=(1,) * (dense.ndim - 2) + (-1, -1)
        )
        result = _func(name)(lazy, _source_filepath="x.zarr", **params)
        assert hasattr(result, "compute"), "lazy input must stay lazy"
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        computed = np.asarray(result.compute())
        assert computed.dtype == expected.dtype
        np.testing.assert_array_equal(computed, expected)
        return lazy, result

    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mirror_matches_dense(self, axis, seed):
        self._run(
            "Mirror Labels",
            self._labels((7, 5, 24, 20), seed=seed),
            axis=axis,
        )

    @pytest.mark.parametrize("k", [0, 1, 4, 8, 9])
    def test_mirror_at_every_pivot(self, k):
        """The pivot slice decides how much of the output is zero padding on
        each side; the edges (k=0, k=n-1) leave padding on one side only."""
        labels = np.zeros((10, 16, 16), dtype=np.uint16)
        labels[k, 2:14, 2:14] = 1
        labels[(k + 3) % 10, 4:6, 4:6] = 2
        self._run("Mirror Labels", labels, axis=0)

    def test_mirror_offset_wraps_like_dense(self):
        """The label offset is added in the label dtype; uint8 overflow must
        wrap the same way on both paths rather than promote."""
        labels = np.zeros((6, 8, 8), dtype=np.uint8)
        labels[2, :, :] = 200
        labels[3, :2, :2] = 100
        self._run("Mirror Labels", labels, axis=0)

    def test_mirror_of_empty_labels(self):
        self._run("Mirror Labels", np.zeros((4, 8, 8), np.uint32), axis=0)

    def test_mirror_keeps_input_chunking(self):
        """
        A reversed slice keeps one block per input block; da.take with an
        out-of-order index would not, and zero padding built as one block
        would materialise every padded slice at once when written.
        """
        labels = np.zeros((40, 16, 16), dtype=np.uint32)
        labels[5, 2:14, 2:14] = 1  # pivot near the start: 30 padded slices
        lazy, result = self._run("Mirror Labels", labels, axis=0)
        assert max(result.chunks[0]) <= max(lazy.chunks[0])
        assert result.numblocks[-2:] == (1, 1)

    @pytest.mark.parametrize("axis", [0, 1, -3])
    @pytest.mark.parametrize("seed", [0, 3])
    def test_keep_slice_range_matches_dense(self, axis, seed):
        self._run(
            "Keep Slice Range by Area",
            self._labels((8, 6, 20, 20), seed=seed),
            axis=axis,
        )

    @pytest.mark.parametrize(
        "areas",
        [
            [3, 1, 4, 9, 2, 5],  # min before max: keep slices 1..3
            [3, 9, 4, 1, 2, 5],  # max before min: same range, other way
        ],
    )
    def test_keep_slice_range_edges(self, areas):
        """Content on both sides of the kept range, so an off-by-one at
        either end changes the result.  Random fixtures rarely do this."""
        labels = np.zeros((len(areas), 4, 4), dtype=np.uint32)
        for z, area in enumerate(areas):
            labels[z].flat[:area] = z + 1
        self._run("Keep Slice Range by Area", labels, axis=0)

    def test_keep_slice_range_with_uniform_areas(self):
        labels = np.ones((5, 8, 8), dtype=np.uint32)
        self._run("Keep Slice Range by Area", labels, axis=0)

    @pytest.mark.parametrize("rgb_chunk", [-1, 1])
    def test_rgb_to_labels_matches_dense(self, rgb_chunk):
        """rgb_chunk=1 splits the colour axis, so the per-pixel all() has
        to combine blocks rather than see each pixel whole."""
        rng = np.random.default_rng(0)
        palette = np.array(
            [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]],
            dtype=np.uint8,
        )
        image = palette[rng.integers(0, len(palette), (4, 3, 24, 24))]
        params = {"blue_label": 5, "green_label": 0, "red_label": 255}

        expected = _func("RGB to Labels")(image, **params)
        lazy = da.from_array(image, chunks=(1, 1, -1, -1, rgb_chunk))
        result = _func("RGB to Labels")(
            lazy, _source_filepath="x.zarr", **params
        )
        assert hasattr(result, "compute")
        assert result.numblocks[0] > 1
        assert result.dtype == expected.dtype
        computed = np.asarray(result.compute())
        assert computed.dtype == expected.dtype
        np.testing.assert_array_equal(computed, expected)

    def test_rgb_to_labels_still_rejects_non_rgb(self):
        lazy = da.zeros((4, 8, 8, 4), dtype=np.uint8, chunks=1)
        with pytest.raises(ValueError, match="RGB"):
            _func("RGB to Labels")(lazy, _source_filepath="x.zarr")

    def test_compression_passes_a_lazy_image_through(self):
        lazy = da.zeros((4, 8, 8), dtype=np.uint16, chunks=1)
        func = _func("Compress with Zstandard")
        assert func(lazy, _source_filepath="x.zarr") is lazy

    @pytest.mark.parametrize(
        "name",
        [
            "Mirror Labels",
            "Keep Slice Range by Area",
            "RGB to Labels",
            "Compress with Zstandard",
        ],
    )
    def test_worker_sees_it_as_lazy_capable(self, name):
        assert "_source_filepath" in inspect.signature(_func(name)).parameters


class TestWorkerStreamsRemaps:
    """
    End to end through ProcessingWorker: a large TIFF must be opened
    lazily, stay lazy through the function, and be streamed to disk.  A
    function that is lazy in isolation can still be densified by the worker
    (a name it mistakes for a folder function, a signature it cannot read),
    so the unit tests above do not prove this on their own.

    Sizing matters for the memory check to mean anything.  The lazy TIFF is
    one Dask task per page at ~28 KB of graph each, so with tiny pages that
    per-task overhead dominates and grows with page count.  256x256 uint32
    pages (256 KB) keep the measurement on pixel data, and Dask is held to
    two threads so pages in flight -- roughly one or two per thread -- do
    not depend on the machine's core count.
    """

    @staticmethod
    def _process(tmp_path, name, data, axes, params):
        import gc
        import tracemalloc

        import dask
        import tifffile

        import napari_tmidas._file_selector as fs
        import napari_tmidas.processing_functions.ome_output_utils as ou
        from napari_tmidas._file_selector import ProcessingWorker

        src = tmp_path / "stack.tif"
        tifffile.imwrite(
            str(src),
            data,
            ome=True,
            photometric="rgb" if axes.endswith("S") else "minisblack",
            compression="zlib",
            metadata={"axes": axes},
        )
        out = tmp_path / "out"
        out.mkdir()

        # Label results are written by ome_output_utils, not the worker, so
        # its budget has to shrink too or a stack this small is one block.
        budget, threshold = fs._STREAM_BLOCK_BYTES, fs._LAZY_TIFF_MIN_BYTES
        label_budget = ou.STREAM_BLOCK_BYTES
        fs._STREAM_BLOCK_BYTES = ou.STREAM_BLOCK_BYTES = 2 * 1024 * 1024
        fs._LAZY_TIFF_MIN_BYTES = 1024 * 1024
        try:
            worker = ProcessingWorker(
                file_list=[str(src)],
                processing_func=_func(name),
                param_values=params,
                output_folder=str(out),
                input_suffix=".tif",
                output_suffix="_out",
                output_format="tiff",
            )
            with dask.config.set(num_workers=2):
                tracemalloc.start()
                try:
                    gc.collect()
                    tracemalloc.reset_peak()
                    baseline, _ = tracemalloc.get_traced_memory()
                    result = worker.process_file(str(src))
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
        finally:
            fs._STREAM_BLOCK_BYTES = budget
            fs._LAZY_TIFF_MIN_BYTES = threshold
            ou.STREAM_BLOCK_BYTES = label_budget

        return tifffile.imread(result["processed_file"]), peak - baseline

    @pytest.mark.parametrize(
        "name,params",
        [
            ("Mirror Labels", {"axis": 1}),
            ("Mirror Labels", {"axis": 0}),
            ("Keep Slice Range by Area", {"axis": 1}),
        ],
    )
    def test_label_remaps_stream(self, tmp_path, name, params):
        rng = np.random.default_rng(7)
        data = np.zeros((24, 8, 256, 256), dtype=np.uint32)
        for t in range(24):
            for z in range(8):
                side = int(rng.integers(8, 250))
                data[t, z, :side, :side] = z + 1

        got, peak = self._process(tmp_path, name, data, "TZYX", params)

        np.testing.assert_array_equal(got, _func(name)(data, **params))
        # Dense would be >= 1x (the input alone); streaming measures ~0.1x.
        assert (
            peak < data.nbytes * 0.3
        ), f"peak {peak/1e6:.1f} MB vs dense {data.nbytes/1e6:.1f} MB"

    def test_rgb_to_labels_streams(self, tmp_path):
        """Also pins the lazy TIFF loader on RGB pages: it used to assume
        every page was one YX plane and read RGB stacks densely."""
        rng = np.random.default_rng(8)
        palette = np.array(
            [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]], np.uint8
        )
        data = palette[rng.integers(0, 4, (120, 256, 256))]

        got, peak = self._process(
            tmp_path, "RGB to Labels", data, "TYXS", {}
        )

        np.testing.assert_array_equal(got, _func("RGB to Labels")(data))
        # The uint32 result is 4/3 the size of the RGB input, so budget
        # against the larger of the two.
        dense = max(data.nbytes, got.nbytes)
        assert (
            peak < dense * 0.3
        ), f"peak {peak/1e6:.1f} MB vs {dense/1e6:.1f} MB"


class TestWholeStackStatistic:
    """
    Functions whose only coupling across planes is one whole-stack number
    -- the min and max for Convert to 8-bit, a percentile for Percentile
    Threshold.  They compute it in a reduction pass, then map per block.
    The number has to be *exactly* the dense one: a threshold one grey level
    off changes which pixels survive, with nothing to flag it.
    """

    @staticmethod
    def _lazy_vs_dense(name, dense, chunks, **params):
        expected = _func(name)(dense, **params)
        lazy = da.from_array(dense, chunks=chunks)
        result = _func(name)(lazy, _source_filepath="x.zarr", **params)
        assert hasattr(result, "compute"), "lazy input must stay lazy"
        assert result.dtype == expected.dtype
        computed = np.asarray(result.compute())
        assert computed.dtype == expected.dtype
        np.testing.assert_array_equal(computed, expected)
        return result

    @pytest.mark.parametrize(
        "dtype,low,high",
        [
            (np.uint16, 100, 4000),
            (np.int16, -3000, 3000),
            (np.uint8, 3, 200),
            (np.float32, -1.5, 7.25),
        ],
    )
    @pytest.mark.parametrize(
        "chunks", [(1, 1, -1, -1), (2, 3, 16, 64)], ids=["planes", "tiles"]
    )
    def test_uint8_conversion_matches_dense(self, dtype, low, high, chunks):
        """'tiles' also splits YX: the arithmetic is per pixel, so blocks
        need not hold whole planes once min and max are global."""
        rng = np.random.default_rng(0)
        image = rng.uniform(low, high, (4, 3, 32, 64)).astype(dtype)
        result = self._lazy_vs_dense(
            "Convert to 8-bit (uint8)", image, chunks
        )
        assert result.numblocks[0] > 1

    @pytest.mark.parametrize("value,expected", [(0, 0), (-5, 0), (7, 255)])
    def test_uint8_conversion_of_a_constant_image(self, value, expected):
        image = np.full((4, 16, 16), value, dtype=np.int16)
        result = self._lazy_vs_dense(
            "Convert to 8-bit (uint8)", image, (1, -1, -1)
        )
        assert int(result[0, 0, 0].compute()) == expected

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int8, np.int16])
    @pytest.mark.parametrize("percentile", [0.0, 12.5, 50.0, 90.0, 99.9, 100.0])
    @pytest.mark.parametrize("output_type", ["original", "binary"])
    def test_percentile_matches_dense(self, dtype, percentile, output_type):
        info = np.iinfo(dtype)
        rng = np.random.default_rng(1)
        image = rng.integers(
            info.min, int(info.max) + 1, (5, 3, 24, 24), dtype=dtype
        )
        self._lazy_vs_dense(
            "Percentile Threshold (Keep Brightest)",
            image,
            (1, 1, -1, -1),
            percentile=percentile,
            output_type=output_type,
        )

    def test_percentile_on_ties_and_rank_boundaries(self):
        """
        Few distinct values make most percentiles fall exactly on a tie or
        between two equal order statistics -- where an off-by-one rank, or
        interpolating from the wrong side, moves the cut.  n = 11 makes
        (n - 1) * 0.9 a float that is not exactly 9.
        """
        from napari_tmidas.processing_functions.skimage_filters import (
            _exact_percentile_lazy,
        )

        rng = np.random.default_rng(2)
        for n in (1, 2, 11, 101, 4097):
            data = rng.integers(0, 4, n).astype(np.uint16)
            lazy = da.from_array(data, chunks=max(1, n // 4))
            for q in (0, 10, 25, 33.3, 50, 66.6, 90, 100):
                assert _exact_percentile_lazy(lazy, q) == np.percentile(
                    data, q
                ), (n, q)

    def test_percentile_wraps_like_numpy_at_int8_extremes(self):
        """numpy takes b - a in the input dtype, so -128..127 wraps; the
        exact path must reproduce that, not 'fix' it."""
        from napari_tmidas.processing_functions.skimage_filters import (
            _exact_percentile_lazy,
        )

        data = np.array([-128, 127] * 8, dtype=np.int8)
        lazy = da.from_array(data, chunks=4)
        with np.errstate(over="ignore"):
            for q in (10, 40, 50, 60, 90):
                assert _exact_percentile_lazy(lazy, q) == np.percentile(
                    data, q
                )

    def test_percentile_rejects_out_of_range(self):
        lazy = da.zeros((4, 8, 8), dtype=np.uint16, chunks=1)
        with pytest.raises(ValueError, match="range"):
            _func("Percentile Threshold (Keep Brightest)")(
                lazy, _source_filepath="x.zarr", percentile=101.0
            )

    @pytest.mark.parametrize("dtype", [np.float32, np.uint32])
    def test_percentile_without_an_exact_lazy_path_stays_correct(self, dtype):
        """Floats and wide integers have no exact bounded-memory percentile,
        so they take the dense path -- correct, just not streamed."""
        rng = np.random.default_rng(3)
        image = (rng.random((4, 16, 16)) * 1000).astype(dtype)
        name = "Percentile Threshold (Keep Brightest)"
        lazy = da.from_array(image, chunks=(1, -1, -1))
        result = _func(name)(lazy, _source_filepath="x.zarr", percentile=75.0)
        assert not hasattr(result, "compute")
        np.testing.assert_array_equal(
            result, _func(name)(image, percentile=75.0)
        )

    @pytest.mark.parametrize(
        "name",
        ["Convert to 8-bit (uint8)", "Percentile Threshold (Keep Brightest)"],
    )
    def test_worker_sees_it_as_lazy_capable(self, name):
        assert "_source_filepath" in inspect.signature(_func(name)).parameters

    @pytest.mark.parametrize(
        "name,params",
        [
            ("Convert to 8-bit (uint8)", {}),
            ("Percentile Threshold (Keep Brightest)", {"percentile": 90.0}),
            (
                "Percentile Threshold (Keep Brightest)",
                {"percentile": 90.0, "output_type": "binary"},
            ),
        ],
    )
    def test_worker_streams_it(self, tmp_path, name, params):
        rng = np.random.default_rng(4)
        data = rng.integers(0, 4000, (24, 8, 256, 256), dtype=np.uint16)

        got, peak = TestWorkerStreamsRemaps._process(
            tmp_path, name, data, "TZYX", params
        )

        np.testing.assert_array_equal(got, _func(name)(data, **params))
        assert (
            peak < data.nbytes * 0.3
        ), f"peak {peak/1e6:.1f} MB vs dense {data.nbytes/1e6:.1f} MB"

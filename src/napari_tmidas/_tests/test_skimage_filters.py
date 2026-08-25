# src/napari_tmidas/_tests/test_skimage_filters.py
import tracemalloc

import numpy as np
import pytest
import skimage.morphology
import tifffile

from napari_tmidas.processing_functions.skimage_filters import (
    adaptive_threshold_bright,
    equalize_histogram,
    invert_image,
    _stream_remove_small_labels,
    percentile_threshold,
    remove_small_objects,
    resize_image_fixed_yx,
    rolling_ball_background,
    simple_thresholding,
)


class TestSkimageFilters:

    def test_invert_image_basic(self):
        """Test basic image inversion functionality"""
        image = np.random.rand(100, 100)

        # Test with default parameters
        result = invert_image(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_invert_image_binary(self):
        """Test image inversion on binary image"""
        image = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        result = invert_image(image)
        # skimage.util.invert inverts all bits, so 0->255, 1->254 for uint8
        expected = np.array([[255, 254], [254, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_invert_image_3d(self):
        """Test image inversion on 3D image"""
        image = np.random.rand(20, 20, 20)

        result = invert_image(image)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_simple_thresholding_returns_uint32(self):
        """Manual thresholding returns a uint32 binary label image (1=foreground,
        0=background) so it's recognized as a Labels layer, not an Image layer."""
        image = np.array([[0, 100, 200], [50, 150, 255]], dtype=np.uint8)

        result = simple_thresholding(image, threshold=128)

        # Check dtype is uint32 (label-typed, see is_label_image)
        assert result.dtype == np.uint32

        # Check values are binary (0 or 1)
        assert set(np.unique(result)).issubset({0, 1})

        # Check correct thresholding
        expected = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

    def test_simple_thresholding_different_thresholds(self):
        """Test manual thresholding with different threshold values"""
        image = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)

        # Test with low threshold
        result_low = simple_thresholding(image, threshold=50)
        assert result_low.dtype == np.uint32
        assert (
            np.sum(result_low == 1) > np.prod(result_low.shape) * 0.8
        )  # Most pixels above 50

        # Test with high threshold
        result_high = simple_thresholding(image, threshold=200)
        assert result_high.dtype == np.uint32
        assert (
            np.sum(result_high == 1) < np.prod(result_high.shape) * 0.3
        )  # Most pixels below 200


class TestBrightRegionExtraction:
    """Test suite for bright region extraction functions"""

    def test_percentile_threshold_original(self):
        """Test percentile thresholding with original values"""
        # Create image with gradient
        image = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)

        result = percentile_threshold(
            image, percentile=90, output_type="original"
        )

        # Only top 10% should remain
        assert result.shape == image.shape
        assert np.sum(result > 0) < image.size * 0.15  # Allow some margin
        assert result.max() == image.max()  # Original max value preserved

    def test_percentile_threshold_binary(self):
        """Test percentile thresholding with binary output"""
        image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)

        result = percentile_threshold(
            image, percentile=80, output_type="binary"
        )

        # Should be binary
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_rolling_ball_background_subtraction(self):
        """Test rolling ball background subtraction"""
        # Create image with uneven background and bright spot
        x, y = np.meshgrid(np.arange(100), np.arange(100))
        background = (50 + 30 * np.sin(x / 20) + 30 * np.sin(y / 20)).astype(
            np.uint8
        )
        image = background.copy()
        image[40:60, 40:60] += 150  # Add bright feature

        result = rolling_ball_background(image, radius=30)

        # Background should be reduced
        assert result.shape == image.shape
        # Center of bright spot should be brighter in result than in corners
        assert result[50, 50] > result[10, 10]

    def test_adaptive_threshold_bright(self):
        """Test adaptive thresholding with bright bias"""
        # Create image with varying brightness
        image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

        result = adaptive_threshold_bright(image, block_size=35, offset=-10.0)

        # Should be binary
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})
        assert result.shape == image.shape

    def test_adaptive_threshold_even_blocksize(self):
        """Test that even block size is handled correctly"""
        image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)

        # Should handle even block size by making it odd
        result = adaptive_threshold_bright(image, block_size=34, offset=0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8


class TestRollingBallPerPlane:
    """
    rolling_ball's kernel spans every axis it is handed, so passing a whole
    stack rolled a ball with as many dimensions as the stack: at the default
    radius=50 a 5 MB TZYX input ran 25 minutes at 13.7 GB RSS without
    finishing, and the background it estimated was blended across T/Z/C.
    The background must be estimated one YX plane at a time.
    """

    @staticmethod
    def _reference_plane(plane, radius):
        """Rolling ball on a single 2D plane, where it was always correct."""
        from skimage.restoration import rolling_ball

        corrected = plane.astype(np.float32) - rolling_ball(
            plane, radius=radius
        )
        return np.clip(corrected, 0, 65535).astype(plane.dtype)

    @pytest.mark.parametrize("shape", [(3, 40, 40), (2, 3, 32, 32)])
    def test_matches_per_plane_reference(self, shape):
        """A stack must equal the 2D result computed plane by plane."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 4000, shape, dtype=np.uint16)

        result = rolling_ball_background(image, radius=5)

        expected = np.empty_like(image)
        for index in np.ndindex(*shape[:-2]):
            expected[index] = self._reference_plane(image[index], 5)
        assert np.array_equal(result, expected)

    def test_planes_are_independent(self):
        """
        Editing one plane must not change any other plane's output.  An n-D
        ball reaches across the leading axes, so this fails outright when the
        whole stack is handed to rolling_ball at once.
        """
        rng = np.random.default_rng(1)
        image = rng.integers(0, 4000, (3, 40, 40), dtype=np.uint16)

        before = rolling_ball_background(image, radius=5)

        edited = image.copy()
        edited[0] = rng.integers(0, 4000, (40, 40), dtype=np.uint16)
        after = rolling_ball_background(edited, radius=5)

        assert np.array_equal(before[1:], after[1:])

    @pytest.mark.parametrize(
        "dtype", [np.uint8, np.uint16, np.float32]
    )
    def test_dtype_is_preserved(self, dtype):
        rng = np.random.default_rng(2)
        if dtype is np.float32:
            image = rng.random((2, 32, 32)).astype(np.float32)
        else:
            high = np.iinfo(dtype).max
            image = rng.integers(0, high, (2, 32, 32), dtype=dtype)

        assert rolling_ball_background(image, radius=5).dtype == dtype

    def test_peak_memory_is_bounded(self):
        """
        Peak must stay near input + output.  The old version also held a
        full-size background plus two full-size float copies on top.
        """
        rng = np.random.default_rng(3)
        image = rng.integers(0, 4000, (8, 4, 128, 128), dtype=np.uint16)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            rolling_ball_background(image, radius=10)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        # input + output is 2.0x; allow one plane and change on top.
        assert peak < 2.5 * image.nbytes, (
            f"peak {peak/1e6:.1f} MB on a {image.nbytes/1e6:.1f} MB input"
        )


class TestCLAHE:
    """Test suite for CLAHE (Contrast Limited Adaptive Histogram Equalization)"""

    def test_clahe_basic(self):
        """Test basic CLAHE functionality"""
        # Create a dark image with weak bright features
        image = np.zeros((100, 100), dtype=np.float32)
        image[40:60, 40:60] = 0.1  # Weak bright region

        result = equalize_histogram(image)

        # Output should be same shape
        assert result.shape == image.shape
        # Output should be normalized to [0, 1] range
        assert result.min() >= 0
        assert result.max() <= 1
        # Contrast should be enhanced (std deviation should increase)
        assert result.std() > image.std()

    def test_clahe_dark_with_membranes(self):
        """Test CLAHE on dark images with weak bright membranes (the use case that failed)"""
        # Create a realistic dark image with weak membrane-like structures
        np.random.seed(42)
        image = np.random.normal(0.05, 0.01, (200, 200))  # Dark background
        image = np.clip(image, 0, 1)

        # Add weak membrane-like structures
        image[50:55, :] += 0.1  # Horizontal membrane
        image[:, 100:105] += 0.1  # Vertical membrane
        image = np.clip(image, 0, 1)

        result = equalize_histogram(image, clip_limit=0.01)

        # Should not produce black image
        assert result.max() > 0.1, "CLAHE should not produce near-black images"
        # Should enhance contrast
        assert result.std() > image.std()
        # Membranes should be more visible (higher values)
        membrane_region = result[50:55, :]
        background_region = result[10:20, 10:20]
        assert membrane_region.mean() > background_region.mean()

    def test_clahe_custom_kernel_size(self):
        """Test CLAHE with custom kernel size"""
        image = np.random.rand(256, 256)

        result = equalize_histogram(image, kernel_size=64)

        assert result.shape == image.shape
        assert result.min() >= 0
        assert result.max() <= 1

    def test_clahe_auto_kernel_size(self):
        """Test CLAHE with automatic kernel size calculation"""
        # Small image
        small_image = np.random.rand(128, 128)
        result_small = equalize_histogram(small_image, kernel_size=0)
        assert result_small.shape == small_image.shape

        # Large image
        large_image = np.random.rand(1024, 1024)
        result_large = equalize_histogram(large_image, kernel_size=0)
        assert result_large.shape == large_image.shape

    def test_clahe_different_clip_limits(self):
        """Test CLAHE with different clip limit values"""
        image = np.random.rand(100, 100) * 0.2  # Dark image

        # Low clip limit (less contrast enhancement)
        result_low = equalize_histogram(image, clip_limit=0.005)

        # High clip limit (more contrast enhancement)
        result_high = equalize_histogram(image, clip_limit=0.05)

        # Both should enhance contrast compared to original
        assert result_low.std() > image.std()
        assert result_high.std() > image.std()
        # Higher clip limit typically gives more contrast (but not always guaranteed)
        assert (
            result_high.max() >= result_low.max() * 0.8
        )  # Allow some tolerance

    def test_clahe_3d_image(self):
        """Test CLAHE on 3D image (should work on last 2 dimensions)"""
        # Create 3D image (e.g., time series or z-stack)
        image_3d = np.random.rand(10, 100, 100) * 0.3

        result = equalize_histogram(image_3d)

        assert result.shape == image_3d.shape
        # Each slice should be enhanced independently
        assert result.std() > image_3d.std()

    def test_clahe_preserves_dtype(self):
        """Test that CLAHE preserves the original dtype"""
        # Test uint8
        img_uint8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result_uint8 = equalize_histogram(img_uint8)
        assert result_uint8.dtype == np.uint8
        assert result_uint8.max() <= 255
        assert result_uint8.min() >= 0

        # Test uint16
        img_uint16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        result_uint16 = equalize_histogram(img_uint16)
        assert result_uint16.dtype == np.uint16
        assert result_uint16.max() <= 65535

        # Test float32
        img_float32 = np.random.rand(100, 100).astype(np.float32)
        result_float32 = equalize_histogram(img_float32)
        assert result_float32.dtype == np.float32
        assert result_float32.max() <= 1.0
        assert result_float32.min() >= 0.0

    def test_clahe_max_workers_parameter(self):
        """Test that max_workers parameter is respected"""
        # Create a 4D image that will trigger parallel processing
        image_4d = np.random.rand(10, 50, 100, 100) * 0.5

        # Test with different worker counts
        result_1 = equalize_histogram(image_4d, max_workers=1)
        result_4 = equalize_histogram(image_4d, max_workers=4)
        result_8 = equalize_histogram(image_4d, max_workers=8)

        # All should produce same shape
        assert result_1.shape == image_4d.shape
        assert result_4.shape == image_4d.shape
        assert result_8.shape == image_4d.shape

        # Results should be nearly identical (some floating point differences are OK)
        np.testing.assert_allclose(result_1, result_4, rtol=1e-5)
        np.testing.assert_allclose(result_1, result_8, rtol=1e-5)


class TestResizeImageFixedYX:
    """Test suite for fixed YX resizing."""

    def test_resize_2d(self):
        image = np.random.randint(0, 65535, (300, 500), dtype=np.uint16)
        result = resize_image_fixed_yx(image, scale_factor=2.0)
        assert result.shape == (600, 1000)
        assert result.dtype == image.dtype

    def test_resize_zyx(self):
        image = np.random.rand(7, 256, 384).astype(np.float32)
        result = resize_image_fixed_yx(image, scale_factor=2.0, dim_order="ZYX")
        assert result.shape == (7, 512, 768)
        assert result.dtype == image.dtype

    def test_resize_tyx(self):
        image = np.random.rand(5, 420, 360).astype(np.float32)
        result = resize_image_fixed_yx(image, scale_factor=2.0, dim_order="TYX")
        assert result.shape == (5, 840, 720)
        assert result.dtype == image.dtype

    def test_resize_tzyx(self):
        image = np.random.randint(0, 255, (3, 4, 128, 256), dtype=np.uint8)
        result = resize_image_fixed_yx(image, scale_factor=2.0)
        assert result.shape == (3, 4, 256, 512)
        assert result.dtype == image.dtype

    def test_resize_with_scale_factor(self):
        image = np.random.rand(5, 420, 360).astype(np.float32)
        result = resize_image_fixed_yx(
            image,
            scale_factor=0.5,
            dim_order="TYX",
        )
        assert result.shape == (5, 210, 180)
        assert result.dtype == image.dtype

    def test_resize_rejects_non_positive_scale_factor(self):
        image = np.random.rand(100, 120).astype(np.float32)
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            resize_image_fixed_yx(image, scale_factor=0)

    def test_invalid_dim_order(self):
        """Orders that do not end in YX are rejected."""
        image = np.random.rand(100, 120).astype(np.float32)
        with pytest.raises(ValueError, match="Unsupported dim_order"):
            resize_image_fixed_yx(image, dim_order="ZXY")

    def test_dim_order_ndim_mismatch(self):
        """A valid order that does not match the image's ndim is rejected."""
        image = np.random.rand(100, 120).astype(np.float32)
        with pytest.raises(ValueError, match="incompatible with image.ndim"):
            resize_image_fixed_yx(image, dim_order="TCZYX")

    @pytest.mark.parametrize(
        "dim_order,shape",
        [
            ("YX", (20, 30)),
            ("CYX", (2, 20, 30)),
            ("TYX", (3, 20, 30)),
            ("ZYX", (3, 20, 30)),
            ("TCYX", (2, 2, 20, 30)),
            ("TZYX", (2, 3, 20, 30)),
            ("ZCYX", (3, 2, 20, 30)),
            ("TZCYX", (2, 2, 2, 20, 30)),
            ("TCZYX", (2, 2, 2, 20, 30)),
        ],
    )
    def test_resize_accepts_every_dim_order(self, dim_order, shape):
        """Every order offered by the dimension-order dropdown is supported."""
        image = np.random.rand(*shape).astype(np.float32)

        result = resize_image_fixed_yx(
            image, scale_factor=2.0, dim_order=dim_order
        )

        assert result.shape == shape[:-2] + (40, 60)
        assert result.dtype == image.dtype


class TestRemoveSmallLabelsStreaming:
    """
    The streaming path must be indistinguishable from the in-memory one.

    A tracked stack is >90% background, so a ~90 MB compressed TIFF can be
    70 GB dense; loading it densely (as the worker did before ``skip_load``)
    got the process OOM-killed.  These tests pin both the equivalence and the
    fact that the stack is never materialised.
    """

    @staticmethod
    def _reference(image, min_size):
        if image.ndim > 3:
            out = np.zeros_like(image)
            for i in range(image.shape[0]):
                out[i] = TestRemoveSmallLabelsStreaming._reference(
                    image[i], min_size
                )
            return out
        try:
            return skimage.morphology.remove_small_objects(
                image, max_size=min_size
            )
        except TypeError:
            return skimage.morphology.remove_small_objects(
                image, min_size=min_size + 1
            )

    @staticmethod
    def _write(path, array):
        # photometric is required: without it tifffile reads a leading axis of
        # length 3 or 4 as RGB samples and stores component planes instead.
        tifffile.imwrite(
            str(path), array, compression="zlib", photometric="minisblack"
        )

    @pytest.mark.parametrize("min_size", [1, 50, 200])
    @pytest.mark.parametrize(
        "shape",
        [(40, 40), (5, 30, 30), (3, 4, 25, 25), (2, 2, 3, 20, 20)],
    )
    def test_matches_in_memory_result(self, tmp_path, shape, min_size):
        rng = np.random.default_rng(0)
        array = rng.integers(0, 8, size=shape, dtype=np.uint32)
        src = tmp_path / "labels.tif"
        self._write(src, array)

        out = _stream_remove_small_labels(src, tmp_path / "out.tif", min_size)

        result = tifffile.imread(out)
        expected = self._reference(array, min_size)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)

    def test_dtype_is_preserved(self, tmp_path):
        array = np.zeros((2, 3, 10, 10), dtype=np.int64)
        array[0, 0, :5, :5] = 7
        src = tmp_path / "labels.tif"
        self._write(src, array)

        out = _stream_remove_small_labels(src, tmp_path / "out.tif", 100)

        assert tifffile.imread(out).dtype == np.int64

    def test_declares_skip_load(self):
        """Without this the worker densely loads the stack before we run."""
        assert getattr(remove_small_objects, "skip_load", False) is True

    def test_widget_call_writes_its_own_output(self, tmp_path):
        """image=None + source/output params -> returns the written path."""
        rng = np.random.default_rng(1)
        array = rng.integers(0, 8, size=(3, 4, 25, 25), dtype=np.uint32)
        src = tmp_path / "labels.tif"
        self._write(src, array)
        outdir = tmp_path / "out"

        result = remove_small_objects(
            None,
            min_size=50,
            _source_filepath=str(src),
            _output_folder=str(outdir),
            _output_suffix="_rm_small",
        )

        assert result == str(outdir / "labels_rm_small.tif")
        np.testing.assert_array_equal(
            tifffile.imread(result), self._reference(array, 50)
        )

    def test_in_memory_call_still_returns_an_array(self):
        """Calling with a plain array (tests, scripts) is unchanged."""
        rng = np.random.default_rng(2)
        array = rng.integers(0, 8, size=(3, 4, 25, 25), dtype=np.uint32)

        result = remove_small_objects(array, min_size=50)

        np.testing.assert_array_equal(result, self._reference(array, 50))

    def test_never_materialises_the_stack(self, tmp_path):
        """Peak allocation stays near one plane, not the whole stack."""
        shape = (16, 16, 512, 512)
        dense_bytes = int(np.prod(shape)) * 4  # 268 MB
        src = tmp_path / "labels.tif"

        # Write the source streamed too, so the test itself never holds the
        # dense stack and the measured peak is the function's alone.
        def planes():
            plane = np.zeros(shape[-2:], dtype=np.uint32)
            plane[10:60, 10:60] = 3  # one label large enough to survive
            for _ in range(int(np.prod(shape[:-2]))):
                yield plane

        with tifffile.TiffWriter(str(src), bigtiff=True) as writer:
            writer.write(
                planes(),
                shape=shape,
                dtype=np.uint32,
                compression="zlib",
                photometric="minisblack",
            )

        tracemalloc.start()
        try:
            _stream_remove_small_labels(src, tmp_path / "out.tif", 100)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # One plane is 1 MB; allow generous slack but stay far below dense.
        assert (
            peak < dense_bytes / 20
        ), f"peak {peak/1e6:.1f} MB vs dense {dense_bytes/1e6:.1f} MB"

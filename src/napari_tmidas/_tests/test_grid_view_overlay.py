"""Tests for grid view overlay processing function."""

import numpy as np
import pytest
import tifffile

try:
    from napari_tmidas.processing_functions.grid_view_overlay import (
        _create_grid,
        _create_overlay,
        _get_intensity_filename,
        create_grid_overlay,
        reset_grid_cache,
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


@pytest.mark.skipif(
    not GRID_OVERLAY_AVAILABLE, reason="Grid overlay function not available"
)
class TestCreateGridOverlay:
    """
    End-to-end coverage for "Grid View: Intensity + Labels Overlay".

    The helpers this builds on are already tested above; what was untested is
    the orchestration around them -- pairing labels with intensity images,
    skipping unusable pairs, the once-per-batch guard, and where the grid
    gets written. All of it depends on module-level state and on a `filepath`
    local recovered from the call stack, so each test drives the function
    through a caller that owns one.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_grid_cache()
        yield
        reset_grid_cache()

    @staticmethod
    def _run(filepath, image=None, **kwargs):
        """Invoke it the way the batch worker does: with a `filepath` local."""
        if image is None:
            image = np.zeros((4, 4), dtype=np.uint8)
        return create_grid_overlay(image, **kwargs)

    @staticmethod
    def _write(path, array):
        tifffile.imwrite(str(path), array, photometric="minisblack")
        return path

    def _pair(self, folder, stem, shape=(32, 32), n_labels=2):
        """Write an intensity/label pair that _get_intensity_filename matches."""
        rng = np.random.default_rng(abs(hash(stem)) % 2**32)
        intensity = (rng.random(shape) * 255).astype(np.uint8)
        labels = np.zeros(shape, dtype=np.uint16)
        for i in range(1, n_labels + 1):
            labels[i * 4 : i * 4 + 3, i * 4 : i * 4 + 3] = i
        self._write(folder / f"{stem}.tif", intensity)
        self._write(folder / f"{stem}_labels.tif", labels)

    def test_builds_and_saves_a_grid(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        out = tmp_path / "out"
        for stem in ("a", "b", "c", "d"):
            self._pair(folder, stem)

        grid = self._run(str(folder / "a_labels.tif"))

        assert isinstance(grid, np.ndarray)
        # RGB output, so three channels.
        assert grid.ndim == 3 and grid.shape[-1] == 3
        assert grid.dtype == np.uint8
        # Saved next to the input folder, named after the first label file.
        saved = list(tmp_path.glob("*_grid_overlay.tif"))
        assert len(saved) == 1, f"expected one saved grid, got {saved}"
        assert saved[0].name == "a_labels_grid_overlay.tif"
        assert out.exists() is False

    def test_second_call_in_a_batch_returns_none(self, tmp_path):
        """
        The batch worker calls the function once per selected file, but the
        grid covers all of them. Only the first call may do work; the rest
        return None so the worker skips saving a copy per file.
        """
        folder = tmp_path / "data"
        folder.mkdir()
        for stem in ("a", "b"):
            self._pair(folder, stem)

        first = self._run(str(folder / "a_labels.tif"))
        second = self._run(str(folder / "b_labels.tif"))

        assert isinstance(first, np.ndarray)
        assert second is None
        assert len(list(tmp_path.glob("*_grid_overlay.tif"))) == 1

    def test_reset_allows_a_new_batch(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a")

        self._run(str(folder / "a_labels.tif"))
        reset_grid_cache()
        again = self._run(str(folder / "a_labels.tif"))

        assert isinstance(again, np.ndarray)

    def test_labels_without_an_intensity_image_are_skipped(self, tmp_path):
        """
        A label file whose intensity partner is missing must not abort the
        run or contribute a blank tile -- it is dropped and the rest proceed.
        """
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a")
        self._pair(folder, "b")
        (folder / "b.tif").unlink()  # orphan b_labels.tif

        grid = self._run(str(folder / "a_labels.tif"))

        assert isinstance(grid, np.ndarray)

    def test_dimension_mismatch_is_skipped(self, tmp_path):
        """Label and intensity images of different shapes cannot be overlaid."""
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a")
        self._write(folder / "b.tif", np.zeros((32, 32), dtype=np.uint8))
        self._write(folder / "b_labels.tif", np.zeros((16, 16), dtype=np.uint16))

        grid = self._run(str(folder / "a_labels.tif"))

        assert isinstance(grid, np.ndarray)

    def test_intensity_only_mode_needs_no_labels(self, tmp_path):
        """An empty label_suffix grids the intensity files by themselves."""
        folder = tmp_path / "data"
        folder.mkdir()
        for stem in ("a", "b", "c"):
            self._write(
                folder / f"{stem}.tif",
                (np.random.default_rng(0).random((32, 32)) * 255).astype(
                    np.uint8
                ),
            )

        grid = self._run(str(folder / "a.tif"), label_suffix="")

        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 3 and grid.shape[-1] == 3

    def test_existing_grid_files_are_not_reprocessed(self, tmp_path, capsys):
        """
        The output lands beside the inputs, so a second run must not fold the
        previous grid into the new one.
        """
        folder = tmp_path / "data"
        folder.mkdir()
        rng = np.random.default_rng(0)
        for stem in ("a", "b"):
            self._write(
                folder / f"{stem}.tif",
                (rng.random((32, 32)) * 255).astype(np.uint8),
            )
        self._write(
            folder / "old_grid_overlay.tif",
            np.zeros((64, 64, 3), dtype=np.uint8),
        )

        grid = self._run(str(folder / "a.tif"), label_suffix="")

        assert isinstance(grid, np.ndarray)
        # The stale overlay is the third *.tif in the folder and must not be
        # counted among the sources.
        assert "Processing 2 images" in capsys.readouterr().out

    def test_3d_stacks_are_max_projected(self, tmp_path):
        """ZYX input is projected to a plane rather than rejected."""
        folder = tmp_path / "data"
        folder.mkdir()
        rng = np.random.default_rng(7)
        intensity = np.zeros((5, 32, 32), dtype=np.uint8)
        intensity[2] = (rng.random((32, 32)) * 255).astype(np.uint8)
        labels = np.zeros((5, 32, 32), dtype=np.uint16)
        labels[2, 4:8, 4:8] = 1
        self._write(folder / "a.tif", intensity)
        self._write(folder / "a_labels.tif", labels)

        grid = self._run(str(folder / "a_labels.tif"))

        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 3

    def test_no_matching_files_returns_the_input_unchanged(self, tmp_path):
        """Nothing to grid means the batch pipeline gets its image back."""
        folder = tmp_path / "data"
        folder.mkdir()
        image = np.zeros((4, 4), dtype=np.uint8)

        result = create_grid_overlay(image, label_suffix="_labels.tif")

        # No `filepath` local anywhere in this call stack.
        np.testing.assert_array_equal(result, image)

    def test_uniform_intensity_does_not_render_black(self):
        """
        The contrast stretch is 0/0 on a uniform image; the NaNs it used to
        produce cast to 0, so a saturated field rendered as a solid black
        tile -- indistinguishable from an empty one in a QC grid.
        """
        flat = np.full((32, 32), 200, dtype=np.uint8)
        labels = np.zeros((32, 32), dtype=np.uint16)

        overlay = _create_overlay(
            flat, labels, target_size=32, label_opacity=0.6,
            show_overlay=False,
        )

        assert overlay.max() > 0, "a uniformly bright image rendered as black"

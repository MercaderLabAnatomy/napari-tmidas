"""
Edge cases of processing_functions/basic.py not reached by the main tests:
the lazy (Dask) Mirror Labels guards, and Intersect Label Images when the
primary is lazy and its pair is a non-TIFF file.
"""

import dask.array as da
import numpy as np
import pytest

from napari_tmidas.processing_functions.basic import (
    intersect_label_images,
    mirror_labels,
)


class TestLazyMirrorLabelsGuards:
    """A lazy stack is checked the same way a dense one is."""

    def test_a_scalar_cannot_be_mirrored(self):
        with pytest.raises(ValueError, match="scalar"):
            mirror_labels(da.zeros((), dtype=np.uint16))

    def test_a_non_integer_axis_is_refused(self):
        with pytest.raises(TypeError, match="integer"):
            mirror_labels(da.zeros((4, 4), dtype=np.uint16), axis=1.0)

    @pytest.mark.parametrize("axis", [2, -3])
    def test_an_axis_outside_the_image_is_refused(self, axis):
        with pytest.raises(ValueError, match="out of bounds"):
            mirror_labels(da.zeros((4, 4), dtype=np.uint16), axis=axis)

    def test_an_empty_stack_comes_back_unchanged_and_lazy(self):
        empty = da.zeros((0, 5, 6), dtype=np.uint16, chunks=(1, 5, 6))

        result = mirror_labels(empty, axis=0)

        assert isinstance(result, da.Array)
        assert result.shape == (0, 5, 6)
        assert result.dtype == np.uint16


class TestIntersectLazyPrimaryWithPngPair:
    def test_primary_ids_are_kept_where_both_images_have_labels(self, tmp_path):
        from skimage.io import imsave

        primary = np.zeros((6, 8), dtype=np.uint8)
        primary[1:5, 1:4] = 3
        primary[1:5, 5:7] = 9
        secondary = np.zeros((6, 8), dtype=np.uint8)
        secondary[2:4, 0:6] = 1
        imsave(str(tmp_path / "cells_b.png"), secondary, check_contrast=False)

        result = intersect_label_images(
            da.from_array(primary, chunks=(3, 4)),
            primary_suffix="_a.tif",
            secondary_suffix="_b.png",
            _source_filepath=str(tmp_path / "cells_a.tif"),
        )

        assert isinstance(result, da.Array), "a lazy primary stays lazy"
        expected = np.where(secondary != 0, primary, 0)
        np.testing.assert_array_equal(result.compute(), expected)

    def test_a_lazy_scalar_primary_is_refused(self, tmp_path):
        from skimage.io import imsave

        imsave(
            str(tmp_path / "x_b.png"),
            np.ones((2, 2), dtype=np.uint8),
            check_contrast=False,
        )

        with pytest.raises(ValueError, match="at least one dimension"):
            intersect_label_images(
                da.zeros((), dtype=np.uint8),
                primary_suffix="_a.tif",
                secondary_suffix="_b.png",
                _source_filepath=str(tmp_path / "x_a.tif"),
            )

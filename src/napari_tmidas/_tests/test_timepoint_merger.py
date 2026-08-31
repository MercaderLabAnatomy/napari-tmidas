# src/napari_tmidas/_tests/test_timepoint_merger.py
"""
Tests for the "Merge Timepoints" processing function.

Merging is the step that turns a folder of per-timepoint files into a single
time series, so two things must hold and neither produces an error when it
breaks: the frames have to be ordered the way a human would order them, and
the axis label the merger reports has to match the array it actually built.
"""
import os

import numpy as np
import pytest
import tifffile

from napari_tmidas.processing_functions import timepoint_merger
from napari_tmidas.processing_functions.timepoint_merger import (
    find_timepoint_images,
    load_and_validate_images,
    natural_sort_key,
    reset_timepoint_merger_cache,
)


def _write_stack(path, shape, value, dtype=np.uint16):
    """
    Write a test TIFF.

    ``photometric="minisblack"`` is not optional here: tifffile interprets a
    leading axis of length 3 or 4 as RGB(A) samples and silently moves it last
    on read, which would make a (3, Y, X) Z-stack come back as (Y, X, 3).
    """
    tifffile.imwrite(
        str(path),
        np.full(shape, value, dtype=dtype),
        photometric="minisblack",
    )
    return str(path)


def _write_series(folder, shape, count, dtype=np.uint16, prefix="f"):
    """Write `count` files whose pixel values encode their 1-based index."""
    return [
        _write_stack(
            folder / f"{prefix}{i}.tif", shape, i, dtype=dtype
        )
        for i in range(1, count + 1)
    ]


class TestNaturalSortKey:
    """
    Timepoint order comes from filenames alone. Plain lexicographic sorting
    puts t10 before t2, which silently reorders the time series -- the merge
    still succeeds and the resulting movie is simply wrong.
    """

    def test_numbers_sort_numerically_not_lexicographically(self):
        names = ["t10.tif", "t2.tif", "t1.tif", "t20.tif", "t3.tif"]

        assert sorted(names, key=natural_sort_key) == [
            "t1.tif",
            "t2.tif",
            "t3.tif",
            "t10.tif",
            "t20.tif",
        ]

    def test_zero_padded_numbers(self):
        names = ["t007.tif", "t10.tif", "t0.tif"]

        assert sorted(names, key=natural_sort_key) == [
            "t0.tif",
            "t007.tif",
            "t10.tif",
        ]

    def test_multiple_number_groups(self):
        """Later number groups break ties in earlier ones."""
        names = ["p2_t10.tif", "p10_t1.tif", "p2_t2.tif"]

        assert sorted(names, key=natural_sort_key) == [
            "p2_t2.tif",
            "p2_t10.tif",
            "p10_t1.tif",
        ]

    def test_case_insensitive_for_text(self):
        assert natural_sort_key("ABC") == natural_sort_key("abc")

    def test_filenames_without_digits(self):
        names = ["beta.tif", "alpha.tif"]

        assert sorted(names, key=natural_sort_key) == [
            "alpha.tif",
            "beta.tif",
        ]


class TestFindTimepointImages:
    def test_returns_naturally_sorted_paths(self, tmp_path):
        for name in ["img10.tif", "img2.tif", "img1.tif"]:
            _write_stack(tmp_path / name, (4, 5), 1)

        found = find_timepoint_images(str(tmp_path))

        assert [os.path.basename(p) for p in found] == [
            "img1.tif",
            "img2.tif",
            "img10.tif",
        ]

    def test_ignores_non_image_files(self, tmp_path):
        _write_stack(tmp_path / "a.tif", (4, 5), 1)
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "results.csv").write_text("also not an image")

        found = find_timepoint_images(str(tmp_path))

        assert [os.path.basename(p) for p in found] == ["a.tif"]

    def test_extension_matching_is_case_insensitive(self, tmp_path):
        _write_stack(tmp_path / "a.TIF", (4, 5), 1)
        _write_stack(tmp_path / "b.Tiff", (4, 5), 1)

        assert len(find_timepoint_images(str(tmp_path))) == 2

    def test_custom_extensions(self, tmp_path):
        _write_stack(tmp_path / "a.tif", (4, 5), 1)
        (tmp_path / "b.png").write_bytes(b"")

        found = find_timepoint_images(str(tmp_path), file_extensions=[".png"])

        assert [os.path.basename(p) for p in found] == ["b.png"]

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            find_timepoint_images(str(tmp_path / "nope"))

    def test_folder_without_images_raises(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")

        with pytest.raises(ValueError, match="No image files found"):
            find_timepoint_images(str(tmp_path))


class TestLoadAndValidateImages:
    def test_2d_files_become_tyx(self, tmp_path):
        files = _write_series(tmp_path, (6, 7), 3)

        stack, order = load_and_validate_images(files)

        assert stack.shape == (3, 6, 7)
        assert order == "TYX"

    def test_frames_keep_file_order(self, tmp_path):
        """
        Each file is filled with its own index, so the merged stack must read
        1, 2, 3 along T. This is the assertion that catches a silent reorder.
        """
        files = _write_series(tmp_path, (6, 7), 4)

        stack, _ = load_and_validate_images(files)

        np.testing.assert_array_equal(stack[:, 0, 0], [1, 2, 3, 4])

    def test_3d_files_become_tzyx(self, tmp_path):
        files = _write_series(tmp_path, (5, 6, 7), 3)

        stack, order = load_and_validate_images(files)

        assert stack.shape == (3, 5, 6, 7)
        assert order == "TZYX"

    def test_explicit_cyx_becomes_tcyx(self, tmp_path):
        """
        3D data is ambiguous -- ZYX and CYX have the same shape -- so the
        explicit hint must win over the ZYX assumption.
        """
        files = _write_series(tmp_path, (5, 6, 7), 3)

        stack, order = load_and_validate_images(files, dimension_order="CYX")

        assert stack.shape == (3, 5, 6, 7)
        assert order == "TCYX"

    def test_4d_files_concatenate_along_time(self, tmp_path):
        """
        4D input is treated as a time series per file, so merging three
        2-timepoint files yields six timepoints -- not a new leading axis.
        """
        files = _write_series(tmp_path, (2, 5, 6, 7), 3)

        stack, order = load_and_validate_images(files)

        assert stack.shape == (6, 5, 6, 7)
        assert order == "TZYX"
        np.testing.assert_array_equal(
            stack[:, 0, 0, 0], [1, 1, 2, 2, 3, 3]
        )

    def test_explicit_czyx_stacks_instead_of_concatenating(self, tmp_path):
        """CZYX is a single timepoint of channels, so files stack on a new T."""
        files = _write_series(tmp_path, (2, 5, 6, 7), 3)

        stack, order = load_and_validate_images(files, dimension_order="CZYX")

        assert stack.shape == (3, 2, 5, 6, 7)
        assert order == "TCZYX"

    def test_dtype_of_first_file_wins(self, tmp_path):
        """
        A mismatched dtype is converted rather than rejected. The merged stack
        must come back in the first file's dtype with values intact.
        """
        files = [
            _write_stack(tmp_path / "f1.tif", (6, 7), 1, dtype=np.uint16),
            _write_stack(tmp_path / "f2.tif", (6, 7), 2, dtype=np.uint8),
        ]

        stack, _ = load_and_validate_images(files)

        assert stack.dtype == np.uint16
        np.testing.assert_array_equal(stack[:, 0, 0], [1, 2])

    def test_inconsistent_shapes_raise(self, tmp_path):
        files = [
            _write_stack(tmp_path / "f1.tif", (6, 7), 1),
            _write_stack(tmp_path / "f2.tif", (6, 8), 2),
        ]

        with pytest.raises(ValueError, match="must have the same dimensions"):
            load_and_validate_images(files)

    def test_hint_that_contradicts_the_data_raises(self, tmp_path):
        files = _write_series(tmp_path, (5, 6, 7), 2)

        with pytest.raises(ValueError, match="expects 2D data"):
            load_and_validate_images(files, dimension_order="YX")

    def test_unsupported_dimensionality_raises(self, tmp_path):
        files = _write_series(tmp_path, (2, 3, 5, 6, 7), 2)

        with pytest.raises(ValueError, match="Unsupported image dimension"):
            load_and_validate_images(files)

    def test_unrecognised_dimension_order_raises(self, tmp_path):
        """A 5-letter hint matches the ndim check but is still not supported."""
        files = _write_series(tmp_path, (2, 3, 5, 6, 7), 2)

        with pytest.raises(ValueError, match="Unsupported dimension order"):
            load_and_validate_images(files, dimension_order="QRSTU")

    def test_dimension_order_hint_is_case_insensitive(self, tmp_path):
        files = _write_series(tmp_path, (5, 6, 7), 2)

        _, order = load_and_validate_images(files, dimension_order="cyx")

        assert order == "TCYX"

    def test_explicit_tzyx_reports_four_axes(self, tmp_path):
        """
        Files that are already time series are concatenated along their own
        T axis, so the result stays 4D.  The explicit path used to label it
        "TTZYX" while auto-detection of the same data said "TZYX".
        """
        files = _write_series(tmp_path, (2, 5, 6, 7), 3)

        stack, order = load_and_validate_images(files, dimension_order="TZYX")

        assert stack.shape == (6, 5, 6, 7)
        assert len(order) == stack.ndim
        assert order == "TZYX"


class TestResetCache:
    def test_reset_clears_processed_folders(self):
        """
        Reached through the module rather than a from-import: the registry
        reloads processing_functions modules, which rebinds this global to a
        fresh set and would leave a captured reference pointing at the old one.
        """
        timepoint_merger._PROCESSED_FOLDERS.add("/some/folder")

        reset_timepoint_merger_cache()

        assert len(timepoint_merger._PROCESSED_FOLDERS) == 0

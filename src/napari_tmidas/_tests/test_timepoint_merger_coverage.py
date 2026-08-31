# src/napari_tmidas/_tests/test_timepoint_merger_coverage.py
"""
Coverage tests for the "Merge Timepoints" batch entry point and CLI.

``merge_timepoint_folder_advanced`` has no explicit context argument: it
discovers the file it is working on by walking the call stack for a frame
holding a ``filepath`` local and, optionally, a worker object exposing
``output_folder`` and ``input_suffix``. These tests reproduce that frame
layout instead of stubbing the function's internals, so every assertion is
about a real merge - files written under ``tmp_path``, shapes, dtypes, OME
axis labels and physical scale, and the skip/overwrite bookkeeping that
decides whether a folder is processed at all.
"""

import os
import runpy
import sys
import tempfile

import numpy as np
import pytest
import tifffile

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import timepoint_merger
from napari_tmidas.processing_functions.timepoint_merger import (
    find_timepoint_images,
    load_and_validate_images,
    merge_timepoint_folder_advanced,
    merge_timepoints_cli,
    reset_timepoint_merger_cache,
)


def _write(path, shape, value, dtype=np.uint16, **ome):
    """
    Write a TIFF filled with ``value``.

    ``photometric="minisblack"`` is load-bearing: tifffile reads a leading
    axis of length 3 or 4 back as RGB(A) samples otherwise, which would
    silently reshape the fixtures.
    """
    kwargs = {"photometric": "minisblack"}
    if ome:
        kwargs["ome"] = True
        kwargs["metadata"] = dict(ome)
    tifffile.imwrite(str(path), np.full(shape, value, dtype=dtype), **kwargs)
    return str(path)


def _series(folder, shape, count, dtype=np.uint16, prefix="t"):
    """Write ``count`` files whose pixel value encodes their 1-based index."""
    folder.mkdir(parents=True, exist_ok=True)
    return [
        _write(folder / f"{prefix}{i}.tif", shape, i, dtype=dtype)
        for i in range(1, count + 1)
    ]


def _ome_pixels(path):
    """Return the OME ``Pixels`` dict of a written OME-TIFF."""
    with tifffile.TiffFile(str(path)) as tif:
        return tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]["Pixels"]


class _FakeWorker:
    """
    Stand-in for the batch worker whose frame the merger inspects.

    Only the two attributes the stack walk looks for are needed; the merger
    breaks out of the walk as soon as it finds an object exposing both.
    """

    def __init__(self, output_folder, input_suffix):
        self.output_folder = output_folder
        self.input_suffix = input_suffix

    def process(self, filepath, image, **kwargs):
        return merge_timepoint_folder_advanced(image, **kwargs)


def _merge(folder, output_folder=None, suffix=".tif", image=None, **kwargs):
    """Invoke the merger through a worker-shaped frame."""
    target = folder if output_folder is None else output_folder
    worker = _FakeWorker(str(target), suffix)
    if image is None:
        image = np.zeros((2, 2), dtype=np.uint16)
    return worker.process(os.path.join(str(folder), "t1.tif"), image, **kwargs)


def _merge_without_worker(folder, image, **kwargs):
    """
    Invoke the merger from a frame that has ``filepath`` but no worker.

    This is the fallback path where the output folder and the file suffix
    have to be derived from the current file itself.
    """
    filepath = os.path.join(str(folder), "t1.tif")
    assert filepath  # keep the local alive for the stack walk
    return merge_timepoint_folder_advanced(image, **kwargs)


@pytest.fixture(autouse=True)
def _clear_processed_folders():
    """
    The merger keeps a module-level set of folders already handled; without
    clearing it, a later test merging the same key would silently no-op.
    """
    reset_timepoint_merger_cache()
    yield
    reset_timepoint_merger_cache()


class TestLoadAndValidateExplicitOrders:
    """
    The per-order branches of ``load_and_validate_images`` that the hint
    (rather than auto-detection) selects, plus the per-file validation that
    happens inside the 4D loading loops.
    """

    def test_explicit_yx_hint_reports_tyx(self, tmp_path):
        files = _series(tmp_path / "s", (6, 7), 3)

        stack, order = load_and_validate_images(files, dimension_order="YX")

        assert stack.shape == (3, 6, 7)
        assert order == "TYX"
        np.testing.assert_array_equal(stack[:, 0, 0], [1, 2, 3])

    def test_tzyx_concatenation_rejects_a_shape_mismatch(self, tmp_path):
        """
        The offending file is named in the message: with 4D concatenation
        the writes are positional, so a wrong shape would otherwise corrupt
        the time axis rather than fail.
        """
        folder = tmp_path / "s"
        folder.mkdir()
        files = [
            _write(folder / "t1.tif", (2, 2, 3, 5), 1),
            _write(folder / "t2.tif", (2, 2, 3, 6), 2),
        ]

        with pytest.raises(ValueError) as excinfo:
            load_and_validate_images(files, dimension_order="TZYX")

        message = str(excinfo.value)
        assert "Error loading" in message
        assert "t2.tif" in message

    def test_tzyx_concatenation_converts_a_mismatched_dtype(self, tmp_path):
        """A differing dtype is converted to the first file's, not rejected."""
        folder = tmp_path / "s"
        folder.mkdir()
        files = [
            _write(folder / "t1.tif", (2, 2, 3, 5), 1, dtype=np.uint16),
            _write(folder / "t2.tif", (2, 2, 3, 5), 2, dtype=np.uint8),
        ]

        stack, _ = load_and_validate_images(files, dimension_order="TZYX")

        assert stack.shape == (4, 2, 3, 5)
        assert stack.dtype == np.uint16
        np.testing.assert_array_equal(
            stack[:, 0, 0, 0], np.array([1, 1, 2, 2], dtype=np.uint16)
        )

    def test_czyx_stacking_rejects_a_shape_mismatch(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        files = [
            _write(folder / "t1.tif", (2, 2, 3, 5), 1),
            _write(folder / "t2.tif", (2, 2, 3, 6), 2),
        ]

        with pytest.raises(ValueError) as excinfo:
            load_and_validate_images(files, dimension_order="CZYX")

        assert "t2.tif" in str(excinfo.value)

    def test_czyx_stacking_converts_a_mismatched_dtype(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        files = [
            _write(folder / "t1.tif", (2, 2, 3, 5), 1, dtype=np.uint16),
            _write(folder / "t2.tif", (2, 2, 3, 5), 2, dtype=np.uint8),
        ]

        stack, order = load_and_validate_images(files, dimension_order="CZYX")

        assert stack.shape == (2, 2, 2, 3, 5)
        assert stack.dtype == np.uint16
        assert order == "TCZYX"
        np.testing.assert_array_equal(stack[:, 0, 0, 0, 0], [1, 2])

    def test_standard_loop_converts_a_mismatched_dtype(self, tmp_path):
        """
        Same conversion behaviour as the 4D branches above, but for the
        plain 2D/3D per-timepoint loop that most folders actually use -
        that loop has its own separate dtype check, untested until now.
        """
        folder = tmp_path / "s"
        folder.mkdir()
        files = [
            _write(folder / "t1.tif", (6, 7), 1, dtype=np.uint16),
            _write(folder / "t2.tif", (6, 7), 2, dtype=np.uint8),
        ]

        stack, order = load_and_validate_images(files)

        assert stack.dtype == np.uint16
        assert order == "TYX"
        np.testing.assert_array_equal(stack[:, 0, 0], [1, 2])

    def test_explicit_zyx_hint_reports_tzyx(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        files = [
            _write(folder / "t1.tif", (2, 3, 4), 1),
            _write(folder / "t2.tif", (2, 3, 4), 2),
        ]

        stack, order = load_and_validate_images(files, dimension_order="ZYX")

        assert stack.shape == (2, 2, 3, 4)
        assert order == "TZYX"
        np.testing.assert_array_equal(stack[:, 0, 0, 0], [1, 2])

    def test_explicit_hint_ndim_mismatch_raises(self, tmp_path):
        """Declaring a 2D order for a 3D file is a user error, not a shape bug."""
        folder = tmp_path / "s"
        folder.mkdir()
        files = [_write(folder / "t1.tif", (2, 3, 4), 1)]

        with pytest.raises(ValueError, match="expects 2D data"):
            load_and_validate_images(files, dimension_order="YX")

    def test_unrecognised_explicit_hint_raises(self, tmp_path):
        """A hint that matches the input's ndim but none of the five orders."""
        folder = tmp_path / "s"
        folder.mkdir()
        files = [_write(folder / "t1.tif", (2, 3), 1)]

        with pytest.raises(ValueError, match="Unsupported dimension order"):
            load_and_validate_images(files, dimension_order="zz")


class TestFindTimepointImages:
    """
    ``find_timepoint_images`` on its own: the entry points always pass an
    explicit extension list, so its ``None`` default is otherwise never
    exercised anywhere in this module.
    """

    def test_default_extensions_match_the_documented_list(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        _write(folder / "a.tif", (2, 2), 1)
        _write(folder / "b.png", (2, 2), 2, dtype=np.uint8)
        _write(folder / "c.bmp", (2, 2), 3, dtype=np.uint8)

        found = find_timepoint_images(str(folder))

        assert sorted(os.path.basename(f) for f in found) == ["a.tif", "b.png"]


class TestMergeFolderHappyPath:
    """
    What the batch entry point actually produces: one merged OME-TIFF per
    folder, the unchanged input array handed back so the batch worker does
    not also write a per-file result, and axis/scale metadata that matches
    the array written.
    """

    def test_merges_folder_and_returns_the_input_unchanged(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (6, 7), 3)
        out = tmp_path / "out"
        out.mkdir()
        image = np.zeros((2, 2), dtype=np.uint16)

        returned = _merge(folder, output_folder=out, image=image)

        assert returned is image
        merged = out / "series_merged_timepoints.tif"
        assert merged.exists()
        data = tifffile.imread(str(merged))
        assert data.shape == (3, 6, 7)
        assert data.dtype == np.uint16
        np.testing.assert_array_equal(data[:, 0, 0], [1, 2, 3])

    def test_frames_are_written_in_natural_order(self, tmp_path):
        """t10 must land after t2, not after t1."""
        folder = tmp_path / "series"
        _series(folder, (4, 5), 11)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        np.testing.assert_array_equal(data[:, 0, 0], list(range(1, 12)))

    def test_2d_input_is_labelled_tyx(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (6, 7), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        pixels = _ome_pixels(out / "series_merged_timepoints.tif")
        assert pixels["SizeT"] == 2
        assert pixels["SizeZ"] == 1
        assert pixels["SizeY"] == 6
        assert pixels["SizeX"] == 7

    def test_3d_input_is_labelled_tzyx(self, tmp_path):
        """A Z of 5, not 3: skimage reads a leading axis of 3 as RGB."""
        folder = tmp_path / "series"
        _series(folder, (5, 6, 7), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        merged = out / "series_merged_timepoints.tif"
        assert tifffile.imread(str(merged)).shape == (2, 5, 6, 7)
        pixels = _ome_pixels(merged)
        assert pixels["SizeT"] == 2
        assert pixels["SizeZ"] == 5

    def test_4d_input_with_no_hint_is_auto_detected_as_tzyx(self, tmp_path):
        """
        A per-file 4D array with no dimension_order hint is assumed to be
        TZYX and concatenated along time through the standard (non-
        memory-efficient) path - the branch real callers hit, since
        passing an explicit hint is the exception, not the rule.
        """
        folder = tmp_path / "series"
        _series(folder, (2, 2, 3, 5), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (4, 2, 3, 5)
        np.testing.assert_array_equal(data[:, 0, 0, 0], [1, 1, 2, 2])

    def test_three_slice_zstacks_keep_z_second(self, tmp_path):
        """
        A leading axis of length 3 or 4 is exactly what skimage.io.imread
        mistakes for RGB(A) samples and moves last, which silently merged a
        3-slice Z-stack as (T, Y, X, Z) under TZYX metadata.
        """
        folder = tmp_path / "series"
        _series(folder, (3, 6, 7), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        merged = out / "series_merged_timepoints.tif"
        assert tifffile.imread(str(merged)).shape == (2, 3, 6, 7)

    def test_physical_scale_of_the_first_file_is_carried_over(self, tmp_path):
        """
        Without the passthrough the merged stack would come back isotropic
        and display at a different physical extent than its own inputs.

        t1 and t2 are given different scales rather than the same values
        twice: with two identical sources the assertion below would pass
        just as well if the code read the *last* file, or an arbitrary
        one, as it would if it genuinely used the first.
        """
        folder = tmp_path / "series"
        folder.mkdir()
        _write(
            folder / "t1.tif",
            (6, 7),
            1,
            axes="YX",
            PhysicalSizeX=0.25,
            PhysicalSizeXUnit="um",
            PhysicalSizeY=0.5,
            PhysicalSizeYUnit="um",
        )
        _write(
            folder / "t2.tif",
            (6, 7),
            2,
            axes="YX",
            PhysicalSizeX=9.0,
            PhysicalSizeXUnit="um",
            PhysicalSizeY=9.0,
            PhysicalSizeYUnit="um",
        )
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        pixels = _ome_pixels(out / "series_merged_timepoints.tif")
        assert float(pixels["PhysicalSizeX"]) == pytest.approx(0.25)
        assert float(pixels["PhysicalSizeY"]) == pytest.approx(0.5)
        assert pixels["PhysicalSizeXUnit"] == "um"

    def test_sources_without_scale_get_no_physical_size(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (6, 7), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        pixels = _ome_pixels(out / "series_merged_timepoints.tif")
        assert float(pixels.get("PhysicalSizeX", 1.0)) == pytest.approx(1.0)

    def test_comma_separated_suffix_matches_several_extensions(self, tmp_path):
        """The widget hands the suffix through as one comma-joined string."""
        folder = tmp_path / "series"
        folder.mkdir()
        _write(folder / "t1.tif", (6, 7), 1)
        _write(folder / "t2.tiff", (6, 7), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out, suffix=".tif, .tiff")

        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (2, 6, 7)

    def test_non_string_suffix_is_used_as_a_single_extension(self, tmp_path):
        """A tuple suffix is wrapped rather than split."""
        folder = tmp_path / "series"
        _series(folder, (6, 7), 2)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out, suffix=(".tif",))

        merged = out / "series_merged_timepoints.tif"
        assert merged.exists()
        data = tifffile.imread(str(merged))
        assert data.shape == (2, 6, 7)
        np.testing.assert_array_equal(data[:, 0, 0], [1, 2])

    def test_without_a_worker_the_source_folder_is_the_output(self, tmp_path):
        """
        Called outside the batch worker there is no output folder and no
        configured suffix, so both are derived from the current file.
        """
        folder = tmp_path / "series"
        _series(folder, (6, 7), 2)
        image = np.zeros((2, 2), dtype=np.uint16)

        returned = _merge_without_worker(folder, image)

        assert returned is image
        merged = folder / "series_merged_timepoints.tif"
        assert merged.exists()
        assert tifffile.imread(str(merged)).shape == (2, 6, 7)

    def test_long_file_lists_are_truncated_in_the_listing(
        self, tmp_path, capsys
    ):
        """Above 25 files only the first 20 and last 5 are echoed."""
        folder = tmp_path / "series"
        _series(folder, (2, 2), 26, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)

        printed = capsys.readouterr().out
        assert "showing first 20 and last 5 of 26 files" in printed
        assert "t26.tif" in printed
        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (26, 2, 2)


class TestMergeFolderSelection:
    """
    Timepoint selection: the chosen range is reflected both in the stack
    that is written and in the output filename, so two different selections
    of the same folder cannot overwrite each other.
    """

    def test_start_subsample_and_max_are_applied_in_order(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 8)
        out = tmp_path / "out"
        out.mkdir()

        _merge(
            folder,
            output_folder=out,
            start_timepoint=1,
            subsample_factor=2,
            max_timepoints=3,
        )

        merged = out / "series_merged_timepoints_sub2_start1_max3.tif"
        assert merged.exists()
        data = tifffile.imread(str(merged))
        assert data.shape == (3, 4, 5)
        np.testing.assert_array_equal(data[:, 0, 0], [2, 4, 6])

    def test_subsample_only_names_the_file_after_the_factor(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 4)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out, subsample_factor=2)

        data = tifffile.imread(str(out / "series_merged_timepoints_sub2.tif"))
        np.testing.assert_array_equal(data[:, 0, 0], [1, 3])

    def test_max_timepoints_larger_than_the_series_is_a_no_op(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 3)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out, max_timepoints=99)

        data = tifffile.imread(str(out / "series_merged_timepoints_max99.tif"))
        assert data.shape == (3, 4, 5)

    def test_start_beyond_the_series_raises(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 3)
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError) as excinfo:
            _merge(folder, output_folder=out, start_timepoint=5)

        assert "start_timepoint" in str(excinfo.value)
        assert not list(out.iterdir())

    def test_empty_selection_raises_instead_of_writing(self, tmp_path):
        """
        A folder whose only matching file is a previous merge result: the
        result is excluded from its own input, leaving nothing to merge.
        """
        folder = tmp_path / "series"
        folder.mkdir()
        _write(folder / "series_merged_timepoints.tif", (2, 4, 5), 9)

        with pytest.raises(ValueError, match="No timepoints selected"):
            _merge(folder, overwrite_existing=True)

    def test_unmatched_suffix_raises(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 2)
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="No image files found"):
            _merge(folder, output_folder=out, suffix=".zarr")


class TestMergeFolderSkipping:
    """
    The merger is called once per file by the batch worker but must run once
    per folder; the session cache and the existing-output check are what
    keep it from redoing (or silently clobbering) work.
    """

    def test_second_call_for_the_same_folder_is_skipped(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 3)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)
        merged = out / "series_merged_timepoints.tif"
        merged.unlink()
        _merge(folder, output_folder=out)

        assert not merged.exists()

    def test_a_different_selection_is_not_skipped(self, tmp_path):
        """The cache key includes the parameters, not just the folder."""
        folder = tmp_path / "series"
        _series(folder, (4, 5), 4)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out)
        _merge(folder, output_folder=out, subsample_factor=2)

        assert sorted(p.name for p in out.iterdir()) == [
            "series_merged_timepoints.tif",
            "series_merged_timepoints_sub2.tif",
        ]

    def test_existing_output_is_left_alone_by_default(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 3)
        out = tmp_path / "out"
        out.mkdir()
        merged = out / "series_merged_timepoints.tif"
        _write(merged, (2, 2), 7)

        image = np.zeros((2, 2), dtype=np.uint16)
        returned = _merge(folder, output_folder=out, image=image)

        assert returned is image
        assert tifffile.imread(str(merged)).shape == (2, 2)
        assert timepoint_merger._PROCESSED_FOLDERS

    def test_overwrite_replaces_the_existing_output(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (4, 5), 3)
        out = tmp_path / "out"
        out.mkdir()
        merged = out / "series_merged_timepoints.tif"
        _write(merged, (2, 2), 7)

        _merge(folder, output_folder=out, overwrite_existing=True)

        assert tifffile.imread(str(merged)).shape == (3, 4, 5)

    def test_a_previous_result_in_the_folder_is_not_re_merged(self, tmp_path):
        """
        With the output written next to its inputs, re-running must exclude
        the previous result - otherwise every rerun grows the stack.
        """
        folder = tmp_path / "series"
        _series(folder, (4, 5), 3)

        _merge(folder)
        reset_timepoint_merger_cache()
        _merge(folder, overwrite_existing=True)

        merged = folder / "series_merged_timepoints.tif"
        assert tifffile.imread(str(merged)).shape == (3, 4, 5)

    def test_missing_file_context_raises(self):
        """
        Called with no ``filepath`` anywhere on the stack there is nothing
        to locate the folder with, and the merger must say so.
        """
        with pytest.raises(
            ValueError, match="Could not determine current file path"
        ):
            merge_timepoint_folder_advanced(np.zeros((2, 2), dtype=np.uint16))


class TestMemoryEfficientLoading:
    """
    Above 100 files the merger switches to one-at-a-time loading into a
    memory-mapped buffer. The result has to be indistinguishable from the
    standard path, and the same shape validation has to apply.
    """

    @pytest.fixture(autouse=True)
    def _temp_dir_inside_tmp_path(self, monkeypatch, tmp_path):
        """
        The memmap buffer is created with ``delete=False``, so redirect
        tempfile at tmp_path rather than leaking into the system temp dir.
        """
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
        (tmp_path / "scratch").mkdir()

    def test_large_2d_series_is_stacked_via_memmap(self, tmp_path, capsys):
        folder = tmp_path / "series"
        _series(folder, (2, 3), 101, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out, memory_efficient=True)

        # Confirms the memmap branch actually ran rather than the regular-
        # array fallback exercised by the next test - both produce the
        # same shape/values, so the print is what tells them apart.
        assert "Created memory-mapped array" in capsys.readouterr().out
        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (101, 2, 3)
        assert data.dtype == np.uint8
        np.testing.assert_array_equal(
            data[:, 0, 0], np.arange(1, 102, dtype=np.uint8)
        )

    def test_memmap_failure_falls_back_to_a_plain_array(
        self, tmp_path, capsys, monkeypatch
    ):
        folder = tmp_path / "series"
        _series(folder, (2, 3), 101, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        class _UnavailableMemmap(np.memmap):
            """Still a type, so the later isinstance() check keeps working."""

            def __new__(cls, *args, **kwargs):
                raise OSError("no mmap here")

        monkeypatch.setattr(np, "memmap", _UnavailableMemmap)
        _merge(folder, output_folder=out, memory_efficient=True)

        assert "Created regular array" in capsys.readouterr().out
        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (101, 2, 3)
        np.testing.assert_array_equal(
            data[:, 0, 0], np.arange(1, 102, dtype=np.uint8)
        )

    def test_large_4d_series_concatenates_along_time(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (2, 2, 3, 5), 101, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        _merge(folder, output_folder=out, memory_efficient=True)

        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (202, 2, 3, 5)
        np.testing.assert_array_equal(data[:4, 0, 0, 0], [1, 1, 2, 2])

    def test_explicit_tzyx_hint_forces_time_concatenation(self, tmp_path):
        """
        The hint, not the ndim, decides in the memory-efficient path; a 4D
        stack declared CZYX gets a new time axis instead.
        """
        folder = tmp_path / "series"
        _series(folder, (2, 2, 3, 5), 101, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        _merge(
            folder,
            output_folder=out,
            memory_efficient=True,
            dimension_order="CZYX",
        )

        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (101, 2, 2, 3, 5)
        np.testing.assert_array_equal(data[:2, 0, 0, 0, 0], [1, 2])

    def test_shape_mismatch_in_a_large_2d_series_raises(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (2, 3), 100, dtype=np.uint8)
        _write(folder / "t101.tif", (2, 4), 101, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="Shape mismatch at timepoint"):
            _merge(folder, output_folder=out, memory_efficient=True)

        assert not list(out.iterdir())

    def test_shape_mismatch_in_a_large_4d_series_raises(self, tmp_path):
        folder = tmp_path / "series"
        _series(folder, (2, 2, 3, 5), 100, dtype=np.uint8)
        _write(folder / "t101.tif", (2, 2, 3, 6), 101, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="Shape mismatch at file"):
            _merge(folder, output_folder=out, memory_efficient=True)

    def test_memory_efficient_is_ignored_below_the_threshold(
        self, tmp_path, monkeypatch
    ):
        """
        The flag only switches paths for series longer than 100 files.

        4 files is small enough that the standard and memory-efficient
        loaders agree on the output byte-for-byte, so shape/value checks
        alone would pass even if the ``len(image_files) > 100`` guard were
        deleted outright. np.memmap is only ever called from the
        memory-efficient branch, so failing loudly if it is touched is
        what actually proves that branch was skipped.
        """
        folder = tmp_path / "series"
        _series(folder, (2, 3), 4, dtype=np.uint8)
        out = tmp_path / "out"
        out.mkdir()

        def _memmap_should_not_be_called(*args, **kwargs):
            raise AssertionError(
                "np.memmap was called below the memory-efficient threshold"
            )

        monkeypatch.setattr(np, "memmap", _memmap_should_not_be_called)

        _merge(folder, output_folder=out, memory_efficient=True)

        data = tifffile.imread(str(out / "series_merged_timepoints.tif"))
        assert data.shape == (4, 2, 3)
        np.testing.assert_array_equal(data[:, 0, 0], [1, 2, 3, 4])


class TestCommandLineInterface:
    """
    The CLI is a separate entry point onto the same loading code: it takes
    an explicit output path and reports failures with a return code rather
    than an exception.
    """

    def test_cli_writes_the_merged_series(self, tmp_path, monkeypatch):
        folder = tmp_path / "series"
        _series(folder, (6, 7), 4)
        out = tmp_path / "movie.tif"
        monkeypatch.setattr(sys, "argv", ["merge", str(folder), str(out)])

        assert merge_timepoints_cli() == 0
        data = tifffile.imread(str(out))
        assert data.shape == (4, 6, 7)
        np.testing.assert_array_equal(data[:, 0, 0], [1, 2, 3, 4])

    def test_cli_applies_start_subsample_and_max(self, tmp_path, monkeypatch):
        folder = tmp_path / "series"
        _series(folder, (6, 7), 9)
        out = tmp_path / "movie.tif"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "merge",
                str(folder),
                str(out),
                "--start",
                "1",
                "--subsample",
                "2",
                "--max-timepoints",
                "3",
            ],
        )

        assert merge_timepoints_cli() == 0
        data = tifffile.imread(str(out))
        np.testing.assert_array_equal(data[:, 0, 0], [2, 4, 6])

    def test_cli_honours_the_extensions_option(self, tmp_path, monkeypatch):
        folder = tmp_path / "series"
        folder.mkdir()
        _write(folder / "a.tif", (6, 7), 1)
        _write(folder / "b.tiff", (6, 7), 2)
        out = tmp_path / "movie.tif"
        monkeypatch.setattr(
            sys,
            "argv",
            ["merge", str(folder), str(out), "--extensions", ".tiff"],
        )

        assert merge_timepoints_cli() == 0
        # A single timepoint still gets the leading time axis.
        data = tifffile.imread(str(out))
        assert data.shape == (1, 6, 7)
        # It must be b.tiff (value 2) that matched, not a.tif (value 1) -
        # shape alone can't tell a correctly-filtered file from a
        # wrongly-filtered one when both folders would contain one file.
        assert data[0, 0, 0] == 2

    def test_cli_reports_a_missing_folder(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / "movie.tif"
        monkeypatch.setattr(
            sys, "argv", ["merge", str(tmp_path / "nope"), str(out)]
        )

        assert merge_timepoints_cli() == 1
        assert "Error:" in capsys.readouterr().out
        assert not out.exists()

    def test_cli_reports_inconsistent_shapes(
        self, tmp_path, monkeypatch, capsys
    ):
        folder = tmp_path / "series"
        folder.mkdir()
        _write(folder / "a.tif", (6, 7), 1)
        _write(folder / "b.tif", (6, 8), 2)
        out = tmp_path / "movie.tif"
        monkeypatch.setattr(sys, "argv", ["merge", str(folder), str(out)])

        assert merge_timepoints_cli() == 1
        assert "same dimensions" in capsys.readouterr().out


class TestModuleEntryPoint:
    """Running the module as a script exits with the CLI's return code."""

    def test_script_execution_exits_with_the_cli_status(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "timepoint_merger",
                str(tmp_path / "nope"),
                str(tmp_path / "out.tif"),
            ],
        )
        registered = BatchProcessingRegistry._processing_functions.get(
            "Merge Timepoints"
        )

        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(timepoint_merger.__file__, run_name="__main__")
        finally:
            # Re-executing the module re-registers a second copy of the
            # merger; put the original entry back so nothing downstream
            # ends up talking to a module object with its own caches.
            if registered is not None:
                BatchProcessingRegistry._processing_functions[
                    "Merge Timepoints"
                ] = registered

        assert excinfo.value.code == 1

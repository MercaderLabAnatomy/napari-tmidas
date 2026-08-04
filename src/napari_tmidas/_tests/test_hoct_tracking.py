"""Tests for the HOCT tracker's input staging.

HOCT reads a single multi-page TIFF eagerly (whole movie in RAM) but reads
Zarr stores one timepoint at a time, so large inputs are streamed into
temporary Zarr stores before the CLI is invoked. These tests pin down that
the staged stores hold exactly the same pixels, in the shape HOCT expects.
"""

import numpy as np
import pytest
import tifffile

from napari_tmidas.processing_functions.hoct_tracking import (
    _cleanup_paths,
    _movie_shape,
    _prepare_raw_input,
    _stage_label_input,
    _unique_tag,
)

zarr = pytest.importorskip("zarr")

T, Z, Y, X, C = 4, 3, 12, 16, 2


def _read_zarr(path):
    return np.asarray(zarr.open(str(path), mode="r")[:])


def _write_ome_zarr(path, data, axes=("t", "c", "z", "y", "x")):
    root = zarr.create_group(store=str(path), overwrite=True)
    array = root.create_array(
        "s0", shape=data.shape, chunks=(1,) * (data.ndim - 2) + data.shape[-2:],
        dtype=data.dtype,
    )
    array[:] = data
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "datasets": [{"path": "s0"}],
                "axes": [{"name": name} for name in axes],
            }
        ],
    }
    return path


def test_movie_shape_matches_hoct_reduction():
    # Channel axis dropped, length-1 axes collapsed, trailing Y/X kept.
    assert _movie_shape((5, 2, 3, 16, 20), "TCZYX") == (5, 3, 16, 20)
    assert _movie_shape((5, 2, 1, 16, 20), "TCZYX") == (5, 16, 20)
    assert _movie_shape((5, 3, 16, 20)) == (5, 3, 16, 20)
    assert _movie_shape((5, 1, 1)) == (5, 1, 1)
    assert _movie_shape(None) is None


def test_unique_tag_differs_between_calls_in_one_process():
    # Concurrent files are tracked from threads of one process, so a
    # PID-based name would make two jobs share (and clobber) a temp file.
    assert _unique_tag() != _unique_tag()


def test_stage_label_input_streams_tiff_to_uint32_zarr(tmp_path):
    labels = np.random.default_rng(0).integers(0, 7, size=(T, Z, Y, X))
    label_tif = tmp_path / "movie_labels.tif"
    tifffile.imwrite(
        label_tif, labels.astype(np.int64), compression="zlib",
        photometric="minisblack",
    )

    staged_path, cleanup = _stage_label_input(
        str(label_tif), tmp_path, "on", "labels"
    )

    staged = _read_zarr(staged_path)
    assert staged.dtype == np.uint32  # int64 costs twice the RAM for nothing
    assert np.array_equal(staged, labels)
    assert cleanup == staged_path

    _cleanup_paths([cleanup])
    assert not (tmp_path / cleanup).exists()


def test_stage_label_input_passes_small_tiff_through(tmp_path):
    label_tif = tmp_path / "small_labels.tif"
    tifffile.imwrite(label_tif, np.zeros((T, Y, X), dtype=np.uint16))

    staged_path, cleanup = _stage_label_input(
        str(label_tif), tmp_path, "auto", "labels"
    )

    # Under the staging threshold an eager load is harmless, so no copy.
    assert staged_path == str(label_tif)
    assert cleanup is None


def test_stage_label_input_handles_2d_time_series(tmp_path):
    labels = np.random.default_rng(1).integers(0, 5, size=(T, Y, X))
    label_tif = tmp_path / "flat_labels.tif"
    tifffile.imwrite(
        label_tif, labels.astype(np.uint16), photometric="minisblack"
    )

    staged_path, _ = _stage_label_input(str(label_tif), tmp_path, "on", "l")

    assert np.array_equal(_read_zarr(staged_path), labels)


def test_stage_label_input_handles_pages_not_split_by_time(tmp_path):
    # Written without `photometric`, tifffile stores this as one RGB-style
    # page holding every timepoint, so per-timepoint page reads are
    # impossible; staging must fall back rather than fail.
    labels = np.random.default_rng(6).integers(0, 5, size=(T, Y, X))
    label_tif = tmp_path / "onepage_labels.tif"
    tifffile.imwrite(label_tif, labels.astype(np.uint16))

    staged_path, _ = _stage_label_input(str(label_tif), tmp_path, "on", "l")

    assert np.array_equal(_read_zarr(staged_path), labels)


@pytest.mark.parametrize("channel", [0, 1])
def test_prepare_raw_input_extracts_channel_from_tiff(tmp_path, channel):
    raw = np.random.default_rng(2).integers(0, 500, size=(T, C, Z, Y, X))
    raw = raw.astype(np.uint16)
    raw_tif = tmp_path / "movie.tif"
    tifffile.imwrite(raw_tif, raw, metadata={"axes": "TCZYX"})

    path, cleanup = _prepare_raw_input(
        str(raw_tif), str(channel), "TCZYX", tmp_path, "on", (T, Z, Y, X), "raw"
    )

    assert np.array_equal(_read_zarr(path), raw[:, channel])
    _cleanup_paths([cleanup])


def test_prepare_raw_input_passes_ome_zarr_channel_zero_through(tmp_path):
    raw = np.random.default_rng(3).integers(0, 500, size=(T, C, Z, Y, X))
    raw_zarr = _write_ome_zarr(tmp_path / "movie.zarr", raw.astype(np.uint16))

    path, cleanup = _prepare_raw_input(
        str(raw_zarr), "0", "Auto", tmp_path, "auto", (T, Z, Y, X), "raw"
    )

    # HOCT keeps the first channel itself and reads Zarr lazily, so handing
    # the store over untouched avoids an extraction pass entirely.
    assert path == str(raw_zarr)
    assert cleanup is None


def test_prepare_raw_input_streams_non_zero_channel_from_zarr(tmp_path):
    raw = np.random.default_rng(4).integers(0, 500, size=(T, C, Z, Y, X))
    raw = raw.astype(np.uint16)
    raw_zarr = _write_ome_zarr(tmp_path / "movie.zarr", raw)

    path, cleanup = _prepare_raw_input(
        str(raw_zarr), "1", "Auto", tmp_path, "auto", (T, Z, Y, X), "raw"
    )

    assert path != str(raw_zarr)
    assert np.array_equal(_read_zarr(path), raw[:, 1])
    _cleanup_paths([cleanup])


def test_prepare_raw_input_collapses_singleton_z_like_hoct(tmp_path):
    raw = np.random.default_rng(5).integers(0, 500, size=(T, C, 1, Y, X))
    raw = raw.astype(np.uint16)
    raw_zarr = _write_ome_zarr(tmp_path / "flat.zarr", raw)

    path, cleanup = _prepare_raw_input(
        str(raw_zarr), "1", "Auto", tmp_path, "on", (T, Y, X), "raw"
    )

    staged = _read_zarr(path)
    assert staged.shape == (T, Y, X)
    assert np.array_equal(staged, raw[:, 1, 0])
    _cleanup_paths([cleanup])

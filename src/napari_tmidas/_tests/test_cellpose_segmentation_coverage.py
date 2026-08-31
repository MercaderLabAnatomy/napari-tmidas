"""Coverage tests for the Cellpose-SAM segmentation processing function.

Cellpose is never installed in this environment and normally runs in a
dedicated out-of-process venv.  Every test here stubs that boundary
(``is_env_created`` / ``create_cellpose_env`` / ``run_cellpose_in_env``)
plus the ConvPaint collaborator, so nothing shells out, no model weights
are needed, and all assertions are about this module's own logic.

Most of the module's logic lives in closures defined inside
``cellpose_segmentation``.  ``_capture_locals`` runs the real function
with a stub runner and snapshots its frame, which is the only way to
exercise those closures directly instead of re-implementing the flow
around them.
"""

import glob
import importlib
import json
import os
import sys

import numpy as np
import pytest
import tifffile
import zarr

from napari_tmidas._registry import BatchProcessingRegistry

cellpose_mod = importlib.import_module(
    "napari_tmidas.processing_functions.cellpose_segmentation"
)
convpaint_mod = importlib.import_module(
    "napari_tmidas.processing_functions.convpaint_prediction"
)


# ---------------------------------------------------------------- helpers


def _stub_env(monkeypatch, runner):
    """Replace the dedicated-environment boundary with *runner*."""
    monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: True)
    monkeypatch.setattr(cellpose_mod, "create_cellpose_env", lambda: None)
    monkeypatch.setattr(cellpose_mod, "run_cellpose_in_env", runner)


def _capture_locals(monkeypatch, **kwargs):
    """Run cellpose_segmentation and snapshot its frame locals."""
    grabbed = {}

    def fake_run(command, args):
        grabbed.update(sys._getframe(1).f_locals)
        image = args.get("image")
        shape = np.shape(image) if image is not None else (4, 4)
        return np.zeros(shape, dtype=np.uint32)

    _stub_env(monkeypatch, fake_run)
    kwargs.setdefault("image", np.zeros((4, 4), dtype=np.float32))
    kwargs.setdefault("dim_order", "YX")
    assert "T" not in kwargs["dim_order"], (
        "time series recurse, so the captured frame would be an inner call"
    )
    cellpose_mod.cellpose_segmentation(**kwargs)
    assert grabbed, "stub runner was never invoked"
    return grabbed


def _make_zarr_source(tmp_path, name="src.zarr", shape=(2, 2, 8, 8)):
    """Create a real on-disk zarr array usable as ``_source_filepath``."""
    path = str(tmp_path / name)
    chunks = tuple(max(1, min(int(s), 8)) for s in shape)
    arr = zarr.open_array(
        path, mode="w", shape=shape, chunks=chunks, dtype=np.uint16
    )
    arr[:] = 1
    return path


def _write_uint32_tif(path, shape):
    tifffile.imwrite(str(path), np.zeros(shape, dtype=np.uint32))
    return str(path)


# ------------------------------------------------------ transpose helper


class TestTransposeDimensions:
    """Pin the axis-reordering helper used to normalise dim orders."""

    def test_yx_is_unchanged(self):
        img = np.arange(6, dtype=np.uint8).reshape(2, 3)
        out, order, is_3d = cellpose_mod.transpose_dimensions(img, "YX")
        assert order == "YX"
        assert is_3d is False
        np.testing.assert_array_equal(out, img)

    def test_zyx_is_marked_3d(self):
        img = np.zeros((3, 4, 5), dtype=np.uint8)
        out, order, is_3d = cellpose_mod.transpose_dimensions(img, "ZYX")
        assert (out.shape, order, is_3d) == ((3, 4, 5), "ZYX", True)

    def test_xyz_is_reordered_to_zyx(self):
        img = np.zeros((5, 4, 3), dtype=np.uint8)
        out, order, is_3d = cellpose_mod.transpose_dimensions(img, "XYZ")
        assert order == "ZYX"
        assert out.shape == (3, 4, 5)
        assert is_3d is True

    def test_tzyx_keeps_time_first(self):
        img = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        out, order, is_3d = cellpose_mod.transpose_dimensions(img, "TZYX")
        assert (out.shape, order, is_3d) == ((2, 3, 4, 5), "TZYX", True)

    def test_ztyx_moves_time_in_front_of_z(self):
        img = np.zeros((3, 2, 4, 5), dtype=np.uint8)
        out, order, is_3d = cellpose_mod.transpose_dimensions(img, "ZTYX")
        assert order == "TZYX"
        assert out.shape == (2, 3, 4, 5)

    def test_time_2d_stack_is_not_3d(self):
        img = np.zeros((7, 4, 5), dtype=np.uint8)
        out, order, is_3d = cellpose_mod.transpose_dimensions(img, "TYX")
        assert (out.shape, order, is_3d) == ((7, 4, 5), "TYX", False)

    def test_values_follow_the_transpose(self):
        img = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        out, _, _ = cellpose_mod.transpose_dimensions(img, "XZY")
        np.testing.assert_array_equal(out, np.transpose(img, [1, 2, 0]))


# ------------------------------------------------------------ validation


class TestParameterValidation:
    """Guard rails that reject impossible parameter combinations."""

    def test_non_string_model_type_is_rejected(self, monkeypatch):
        _stub_env(monkeypatch, lambda c, a: np.zeros((4, 4), np.uint32))
        with pytest.raises(ValueError, match="model_type must be a string"):
            cellpose_mod.cellpose_segmentation(
                np.zeros((4, 4), np.float32), model_type=7
            )

    def test_unknown_model_type_is_rejected(self, monkeypatch):
        _stub_env(monkeypatch, lambda c, a: np.zeros((4, 4), np.uint32))
        with pytest.raises(ValueError, match="Unsupported model_type"):
            cellpose_mod.cellpose_segmentation(
                np.zeros((4, 4), np.float32), model_type="stardist"
            )

    def test_model_type_whitespace_is_stripped(self, monkeypatch):
        seen = {}

        def fake_run(command, args):
            seen.update(args)
            return np.zeros(args["image"].shape, np.uint32)

        _stub_env(monkeypatch, fake_run)
        cellpose_mod.cellpose_segmentation(
            np.zeros((4, 4), np.float32), model_type="  cpdino  "
        )
        assert seen["model_type"] == "cpdino"

    @pytest.mark.parametrize(
        "given, expected",
        [(0, 1), (-4, 1), ("3", 3), (2.9, 2), (None, 1), ("many", 1)],
    )
    def test_worker_count_is_coerced_to_a_positive_int(
        self, monkeypatch, given, expected
    ):
        # A diagnostic print used to call int(distributed_n_workers)
        # *before* this exact try/except guard, so a non-numeric value
        # crashed on the print instead of being coerced by the guard it
        # was written to fall into.  None and "many" now exercise that
        # guard for real.
        grabbed = _capture_locals(
            monkeypatch, distributed_n_workers=given
        )
        assert grabbed["distributed_n_workers"] == expected

    def test_time_axis_beyond_image_rank_is_rejected(self, monkeypatch):
        _stub_env(monkeypatch, lambda c, a: np.zeros((4, 4), np.uint32))
        with pytest.raises(ValueError, match="does not have a matching T"):
            cellpose_mod.cellpose_segmentation(
                np.zeros((4, 4), np.float32), dim_order="YXT"
            )

    def test_convpaint_mask_without_model_path_is_rejected(
        self, monkeypatch
    ):
        _stub_env(monkeypatch, lambda c, a: np.zeros((4, 4), np.uint32))
        with pytest.raises(ValueError, match="convpaint_model_path"):
            cellpose_mod.cellpose_segmentation(
                np.zeros((4, 4), np.float32),
                use_convpaint_auto_mask=True,
                convpaint_model_path="   ",
            )

    def test_convpaint_import_failure_becomes_runtime_error(
        self, monkeypatch
    ):
        _stub_env(monkeypatch, lambda c, a: np.zeros((4, 4), np.uint32))
        monkeypatch.setitem(
            sys.modules,
            "napari_tmidas.processing_functions.convpaint_prediction",
            None,
        )
        with pytest.raises(RuntimeError, match="Failed to import ConvPaint"):
            cellpose_mod.cellpose_segmentation(
                np.zeros((4, 4), np.float32),
                use_convpaint_auto_mask=True,
                convpaint_model_path="model.pkl",
            )


class TestResolveTimepointIndices:
    """Start/end/step resolution for time-series selection."""

    def _resolver(self, monkeypatch, **kwargs):
        return _capture_locals(monkeypatch, **kwargs)[
            "_resolve_timepoint_indices"
        ]

    def test_empty_series_returns_no_indices(self, monkeypatch):
        assert self._resolver(monkeypatch)(0) == []

    def test_defaults_select_every_timepoint(self, monkeypatch):
        assert self._resolver(monkeypatch)(4) == [0, 1, 2, 3]

    def test_step_selects_every_nth(self, monkeypatch):
        resolve = self._resolver(
            monkeypatch, timepoint_start=1, timepoint_end=6, timepoint_step=2
        )
        assert resolve(10) == [1, 3, 5]

    def test_end_is_clamped_to_last_index(self, monkeypatch):
        resolve = self._resolver(monkeypatch, timepoint_end=99)
        assert resolve(3) == [0, 1, 2]

    def test_negative_start_counts_from_the_end(self, monkeypatch):
        resolve = self._resolver(monkeypatch, timepoint_start=-2)
        assert resolve(5) == [3, 4]

    def test_start_is_clamped_into_range(self, monkeypatch):
        resolve = self._resolver(monkeypatch, timepoint_start=50)
        assert resolve(3) == [2]

    def test_zero_step_is_rejected(self, monkeypatch):
        resolve = self._resolver(monkeypatch, timepoint_step=0)
        with pytest.raises(ValueError, match="timepoint_step must be >= 1"):
            resolve(5)

    def test_end_before_start_is_rejected(self, monkeypatch):
        resolve = self._resolver(
            monkeypatch, timepoint_start=3, timepoint_end=1
        )
        with pytest.raises(ValueError, match="timepoint_end must be >="):
            resolve(5)


# ------------------------------------------------- source signature bits


class TestSourceSignatureHelpers:
    """Path-identity helpers that let runs resume across mount aliases."""

    def test_signature_id_is_the_basename(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        sig = _capture_locals(monkeypatch, _source_filepath=src)[
            "_source_signature_id"
        ]
        assert sig() == "movie.tif"
        assert sig("/a/b/c/other.zarr/") == "other.zarr"
        assert sig("") == ""
        assert sig(None) == "movie.tif"

    def test_signature_id_is_empty_without_a_source(self, monkeypatch):
        sig = _capture_locals(monkeypatch)["_source_signature_id"]
        assert sig() == ""

    def test_matches_identical_and_aliased_paths(
        self, monkeypatch, tmp_path
    ):
        real = tmp_path / "real"
        real.mkdir()
        src = _write_uint32_tif(real / "movie.tif", (2, 2))
        link = tmp_path / "link"
        link.symlink_to(real)
        matches = _capture_locals(monkeypatch, _source_filepath=src)[
            "_source_signature_matches"
        ]
        assert matches(src) is True
        assert matches(str(link / "movie.tif")) is True
        assert matches("/elsewhere/movie.tif") is True
        assert matches("/elsewhere/different.tif") is False

    def test_matches_is_false_without_either_side(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        with_src = _capture_locals(monkeypatch, _source_filepath=src)
        assert with_src["_source_signature_matches"]("") is False
        no_src = _capture_locals(monkeypatch)
        assert no_src["_source_signature_matches"]("movie.tif") is False

    def test_abspath_equality_alone_is_enough_to_match(
        self, monkeypatch, tmp_path
    ):
        # Basenames differ, so the basename fallback would say False and
        # realpath is never reached: only the abspath comparison can match.
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        matches = _capture_locals(monkeypatch, _source_filepath=src)[
            "_source_signature_matches"
        ]
        monkeypatch.setattr(os.path, "abspath", lambda _p: "/collapsed")
        assert matches("/somewhere/else/other-name.tif") is True

    def test_realpath_equality_alone_is_enough_to_match(
        self, monkeypatch, tmp_path
    ):
        # abspath keeps the two apart and the basenames differ, so only the
        # realpath comparison can match.
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        matches = _capture_locals(monkeypatch, _source_filepath=src)[
            "_source_signature_matches"
        ]
        monkeypatch.setattr(os.path, "realpath", lambda _p: "/collapsed")
        assert matches("/somewhere/else/other-name.tif") is True


class TestRunSignatureCompatibility:
    """Resume is allowed when only the source path prefix differs."""

    def _compat(self, monkeypatch, src):
        return _capture_locals(monkeypatch, _source_filepath=src)[
            "_run_signatures_compatible"
        ]

    def test_identical_json_is_compatible(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "m.tif", (2, 2))
        compat = self._compat(monkeypatch, src)
        assert compat('{"a": 1}', '{"a": 1}') is True

    def test_unparsable_json_is_incompatible(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "m.tif", (2, 2))
        compat = self._compat(monkeypatch, src)
        assert compat("not json", '{"a": 1}') is False

    def test_non_dict_payloads_are_incompatible(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "m.tif", (2, 2))
        compat = self._compat(monkeypatch, src)
        assert compat("[1, 2]", "[1, 2, 3]") is False

    def test_differing_parameters_are_incompatible(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "m.tif", (2, 2))
        compat = self._compat(monkeypatch, src)
        a = json.dumps({"source": "m.tif", "flow": 0.4})
        b = json.dumps({"source": "m.tif", "flow": 0.9})
        assert compat(a, b) is False

    def test_same_parameters_and_source_basename_are_compatible(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "m.tif", (2, 2))
        compat = self._compat(monkeypatch, src)
        a = json.dumps({"source": "/mnt/a/m.tif", "flow": 0.4})
        b = json.dumps({"source": "/run/mnt/a/m.tif", "flow": 0.4})
        assert compat(a, b) is True

    def test_same_parameters_but_other_source_is_incompatible(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "m.tif", (2, 2))
        compat = self._compat(monkeypatch, src)
        a = json.dumps({"source": "/x/other.tif", "flow": 0.4})
        b = json.dumps({"source": "/y/another.tif", "flow": 0.4})
        assert compat(a, b) is False


class TestNormalizeLoadedPath:
    """Saved paths are re-resolved across /media and /run/media mounts."""

    def _normalize(self, monkeypatch):
        return _capture_locals(monkeypatch)["_normalize_loaded_path"]

    def test_empty_path_uses_the_fallback(self, monkeypatch):
        normalize = self._normalize(monkeypatch)
        assert normalize("", fallback="/kept") == "/kept"

    def test_existing_path_is_returned_unchanged(
        self, monkeypatch, tmp_path
    ):
        normalize = self._normalize(monkeypatch)
        real = str(tmp_path / "model.pkl")
        (tmp_path / "model.pkl").write_text("x")
        assert normalize(real) == real

    def test_unresolvable_path_is_returned_as_is(self, monkeypatch):
        normalize = self._normalize(monkeypatch)
        missing = "/media/disk/model.pkl"
        assert normalize(missing) == missing

    def test_media_prefix_is_swapped_when_the_alias_exists(
        self, monkeypatch
    ):
        normalize = self._normalize(monkeypatch)
        wanted = "/run/media/disk/model.pkl"
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path,
            "exists",
            lambda p: True if p == wanted else real_exists(p),
        )
        assert normalize("/media/disk/model.pkl") == wanted

    def test_run_media_prefix_is_swapped_back(self, monkeypatch):
        normalize = self._normalize(monkeypatch)
        wanted = "/media/disk/model.pkl"
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path,
            "exists",
            lambda p: True if p == wanted else real_exists(p),
        )
        assert normalize("/run/media/disk/model.pkl") == wanted


# ------------------------------------------------------- output plumbing


class TestOutputPathHelpers:
    """Direct/legacy output naming and existing-output validation."""

    def test_no_source_means_no_direct_output(self, monkeypatch):
        grabbed = _capture_locals(monkeypatch)
        assert grabbed["_direct_output_path"]() is None

    def test_direct_output_falls_back_to_source_folder(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        grabbed = _capture_locals(monkeypatch, _source_filepath=src)
        assert grabbed["_direct_output_path"]() == str(
            tmp_path / "movie_cp_labels.tif"
        )

    def test_direct_output_honours_folder_and_suffix(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        out = tmp_path / "out"
        out.mkdir()
        grabbed = _capture_locals(
            monkeypatch,
            _source_filepath=src,
            _output_folder=str(out),
            _output_suffix="_labels",
        )
        assert grabbed["_direct_output_path"]() == str(
            out / "movie_labels.tif"
        )

    def test_direct_output_uses_zarr_extension(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        grabbed = _capture_locals(
            monkeypatch,
            _source_filepath=src,
            _output_folder=str(tmp_path),
            _output_suffix="_labels",
            _output_format="zarr",
        )
        assert grabbed["_direct_output_path"]().endswith(
            "movie_labels.zarr"
        )

    def test_legacy_candidates_need_folder_and_suffix(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        grabbed = _capture_locals(monkeypatch, _source_filepath=src)
        assert grabbed["_legacy_output_candidates"]() == []

    def test_legacy_candidates_exclude_the_direct_output(
        self, monkeypatch, tmp_path
    ):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        out = tmp_path / "out"
        out.mkdir()
        direct = _write_uint32_tif(out / "movie_labels.tif", (2, 2))
        older = _write_uint32_tif(out / "movie_v1_labels.tif", (2, 2))
        newer = _write_uint32_tif(out / "movie_v2_labels.tif", (2, 2))
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        grabbed = _capture_locals(
            monkeypatch,
            _source_filepath=src,
            _output_folder=str(out),
            _output_suffix="_labels",
        )
        candidates = grabbed["_legacy_output_candidates"]()
        assert direct not in candidates
        assert candidates == [newer, older]


class TestExistingOutputIsValid:
    """Reuse of already-written outputs must not accept partial files."""

    def _validator(self, monkeypatch, output_format="tiff"):
        return _capture_locals(
            monkeypatch, _output_format=output_format
        )["_existing_output_is_valid"]

    def test_missing_tiff_is_invalid(self, monkeypatch, tmp_path):
        valid = self._validator(monkeypatch)
        assert valid(str(tmp_path / "nope.tif")) is False

    def test_complete_uint32_tiff_is_valid(self, monkeypatch, tmp_path):
        path = _write_uint32_tif(tmp_path / "ok.tif", (3, 4, 5))
        valid = self._validator(monkeypatch)
        assert valid(path) is True
        assert valid(path, expected_shape=(3, 4, 5)) is True

    def test_shape_mismatch_is_invalid(self, monkeypatch, tmp_path):
        path = _write_uint32_tif(tmp_path / "ok.tif", (3, 4, 5))
        valid = self._validator(monkeypatch)
        assert valid(path, expected_shape=(9, 4, 5)) is False

    def test_wrong_dtype_is_invalid(self, monkeypatch, tmp_path):
        path = str(tmp_path / "u16.tif")
        tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))
        valid = self._validator(monkeypatch)
        assert valid(path) is False

    def test_unreadable_tiff_is_invalid(self, monkeypatch, tmp_path):
        path = tmp_path / "broken.tif"
        path.write_bytes(b"not a tiff at all")
        valid = self._validator(monkeypatch)
        assert valid(str(path)) is False

    def test_zarr_output_only_needs_a_directory(
        self, monkeypatch, tmp_path
    ):
        valid = self._validator(monkeypatch, output_format="zarr")
        (tmp_path / "out.zarr").mkdir()
        assert valid(str(tmp_path / "out.zarr")) is True
        assert valid(str(tmp_path / "absent.zarr")) is False

    def test_unknown_format_falls_back_to_existence(
        self, monkeypatch, tmp_path
    ):
        valid = self._validator(monkeypatch, output_format="npy")
        target = tmp_path / "out.npy"
        assert valid(str(target)) is False
        target.write_bytes(b"0")
        assert valid(str(target)) is True


# --------------------------------------------------- cached-slab recovery


def _write_slab_cache(root, timepoints, slab_shape, value=7):
    root.mkdir(parents=True, exist_ok=True)
    for t in timepoints:
        arr = zarr.open_array(
            str(root / f"t{t:04d}_labels.zarr"),
            mode="w",
            shape=slab_shape,
            chunks=slab_shape,
            dtype=np.uint32,
        )
        arr[:] = value + t
    return str(root)


class TestRecoverTiffFromCachedZarr:
    """Fallback that rebuilds a TIFF from per-timepoint zarr slabs."""

    def _recover(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "src.tif", (2, 2))
        return _capture_locals(monkeypatch, _source_filepath=src)[
            "_recover_tiff_from_cached_timepoint_zarr"
        ]

    def test_missing_cache_directory_raises(self, monkeypatch, tmp_path):
        recover = self._recover(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match="No slab cache directory"):
            recover(str(tmp_path / "absent"), str(tmp_path / "o.tif"))

    def test_empty_cache_directory_raises(self, monkeypatch, tmp_path):
        recover = self._recover(monkeypatch, tmp_path)
        empty = tmp_path / "cache"
        empty.mkdir()
        (empty / "unrelated.txt").write_text("x")
        with pytest.raises(RuntimeError, match="No cached timepoint zarr"):
            recover(str(empty), str(tmp_path / "o.tif"))

    def test_two_dimensional_slabs_stack_into_tyx(
        self, monkeypatch, tmp_path
    ):
        recover = self._recover(monkeypatch, tmp_path)
        root = _write_slab_cache(tmp_path / "cache", [0, 1, 2], (4, 5))
        out = str(tmp_path / "recovered.tif")
        assert recover(root, out) == out
        data = tifffile.imread(out)
        assert data.shape == (3, 4, 5)
        assert data.dtype == np.uint32
        assert set(np.unique(data)) == {7, 8, 9}

    def test_three_dimensional_slabs_stack_into_tzyx(
        self, monkeypatch, tmp_path
    ):
        recover = self._recover(monkeypatch, tmp_path)
        root = _write_slab_cache(tmp_path / "cache", [0, 1], (2, 4, 5))
        out = str(tmp_path / "recovered.tif")
        recover(root, out)
        data = tifffile.imread(out)
        assert data.shape == (2, 2, 4, 5)
        assert data.dtype == np.uint32
        # Slab t carries the value 7 + t, in cache order.
        assert set(np.unique(data[0])) == {7}
        assert set(np.unique(data[1])) == {8}

    def test_selected_timepoints_pick_a_subset(self, monkeypatch, tmp_path):
        recover = self._recover(monkeypatch, tmp_path)
        root = _write_slab_cache(tmp_path / "cache", [0, 1, 2, 3], (4, 5))
        out = str(tmp_path / "recovered.tif")
        recover(root, out, selected_timepoints=[1, 3])
        data = tifffile.imread(out)
        assert data.shape == (2, 4, 5)
        assert set(np.unique(data)) == {8, 10}

    def test_missing_selected_timepoint_raises(self, monkeypatch, tmp_path):
        recover = self._recover(monkeypatch, tmp_path)
        root = _write_slab_cache(tmp_path / "cache", [0, 1], (4, 5))
        with pytest.raises(RuntimeError, match=r"Missing cached slabs"):
            recover(
                root,
                str(tmp_path / "o.tif"),
                selected_timepoints=[0, 5],
            )

    def test_unsupported_slab_rank_raises(self, monkeypatch, tmp_path):
        recover = self._recover(monkeypatch, tmp_path)
        root = _write_slab_cache(tmp_path / "cache", [0], (2, 2, 3, 3))
        with pytest.raises(RuntimeError, match="2D or 3D shapes"):
            recover(root, str(tmp_path / "o.tif"))

    def test_inconsistent_slab_shapes_raise(self, monkeypatch, tmp_path):
        recover = self._recover(monkeypatch, tmp_path)
        root = tmp_path / "cache"
        _write_slab_cache(root, [0], (4, 5))
        _write_slab_cache(root, [1], (6, 7))
        with pytest.raises(RuntimeError, match="Cached slab shape mismatch"):
            recover(str(root), str(tmp_path / "o.tif"))


class TestWriteInterleavedCheckpointOutput:
    """Final write of the interleaved checkpoint, incl. skip and rescue."""

    def _writer(self, monkeypatch, tmp_path, **kwargs):
        src = _write_uint32_tif(tmp_path / "src.tif", (2, 2))
        kwargs.setdefault("_source_filepath", src)
        grabbed = _capture_locals(monkeypatch, **kwargs)
        return src, grabbed["_write_interleaved_checkpoint_output"]

    def _checkpoint(self, tmp_path, shape=(2, 4, 5)):
        arr = zarr.open_array(
            str(tmp_path / "ckpt.zarr"),
            mode="w",
            shape=shape,
            chunks=shape,
            dtype=np.uint32,
        )
        arr[:] = 3
        return arr

    def test_returns_none_without_a_source(self, monkeypatch, tmp_path):
        grabbed = _capture_locals(monkeypatch)
        write = grabbed["_write_interleaved_checkpoint_output"]
        assert write(self._checkpoint(tmp_path), "unused") is None

    def test_delegates_to_the_metadata_writer(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: calls.append(kw) or kw["output_path"],
        )
        src, write = self._writer(
            monkeypatch,
            tmp_path,
            image=np.zeros((2, 4, 4), dtype=np.float32),
            dim_order="ZYX",
            _output_folder=str(tmp_path),
            _output_suffix="_lbl",
        )
        checkpoint = self._checkpoint(tmp_path)
        out = write(checkpoint, "ckpt-path")
        assert out == str(tmp_path / "src_lbl.tif")
        assert len(calls) == 1
        assert calls[0]["source_path"] == src
        assert calls[0]["dim_order"] == "ZYX"
        assert calls[0]["output_format"] == "tiff"
        assert calls[0]["labels"] is checkpoint

    def test_valid_existing_output_is_not_rewritten(
        self, monkeypatch, tmp_path
    ):
        calls = []
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: calls.append(kw),
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_uint32_tif(out_dir / "src_lbl.tif", (2, 4, 5))
        _, write = self._writer(
            monkeypatch,
            tmp_path,
            _output_folder=str(out_dir),
            _output_suffix="_lbl",
        )
        assert write(self._checkpoint(tmp_path), "ckpt") == str(
            out_dir / "src_lbl.tif"
        )
        assert calls == []

    def test_valid_legacy_output_is_reused(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: calls.append(kw),
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        legacy = _write_uint32_tif(out_dir / "src_old_lbl.tif", (2, 4, 5))
        _, write = self._writer(
            monkeypatch,
            tmp_path,
            _output_folder=str(out_dir),
            _output_suffix="_lbl",
        )
        assert write(self._checkpoint(tmp_path), "ckpt") == legacy
        assert calls == []

    def test_skip_disabled_forces_a_rewrite(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: calls.append(kw),
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_uint32_tif(out_dir / "src_lbl.tif", (2, 4, 5))
        _, write = self._writer(
            monkeypatch,
            tmp_path,
            _output_folder=str(out_dir),
            _output_suffix="_lbl",
            skip_overwrite_existing_valid_output=False,
        )
        write(self._checkpoint(tmp_path), "ckpt")
        assert len(calls) == 1
        assert calls[0]["output_path"] == str(out_dir / "src_lbl.tif")

    def test_failed_tiff_write_recovers_from_slab_cache(
        self, monkeypatch, tmp_path
    ):
        def boom(**kw):
            raise OSError("disk full")

        monkeypatch.setattr(
            cellpose_mod, "write_labels_with_source_metadata", boom
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _, write = self._writer(
            monkeypatch,
            tmp_path,
            _output_folder=str(out_dir),
            _output_suffix="_lbl",
        )
        root = _write_slab_cache(tmp_path / "slabs", [0, 1], (4, 5))
        out = write(
            self._checkpoint(tmp_path),
            "ckpt",
            slab_output_root=root,
            selected_timepoints=[0, 1],
        )
        assert out == str(out_dir / "src_lbl.tif")
        rescued = tifffile.imread(out)
        assert rescued.shape == (2, 4, 5)
        assert set(np.unique(rescued[0])) == {7}
        assert set(np.unique(rescued[1])) == {8}

    def test_failed_zarr_write_is_not_rescued(self, monkeypatch, tmp_path):
        def boom(**kw):
            raise OSError("disk full")

        monkeypatch.setattr(
            cellpose_mod, "write_labels_with_source_metadata", boom
        )
        _, write = self._writer(
            monkeypatch,
            tmp_path,
            _output_folder=str(tmp_path),
            _output_suffix="_lbl",
            _output_format="zarr",
        )
        with pytest.raises(OSError, match="disk full"):
            write(self._checkpoint(tmp_path), "ckpt")


# ------------------------------------------------------ ConvPaint helpers


def _stub_convpaint(monkeypatch, factory):
    """Replace the ConvPaint predictor with a deterministic stub."""
    calls = []

    def fake_predict(image, **kwargs):
        calls.append({"shape": np.shape(image), **kwargs})
        return factory(np.asarray(image))

    monkeypatch.setattr(convpaint_mod, "convpaint_predict", fake_predict)
    return calls


class TestConvpaintMaskHelpers:
    """Mask post-processing and distributed block gating."""

    def _helpers(self, monkeypatch, **kwargs):
        _stub_convpaint(
            monkeypatch, lambda img: np.ones(img.shape, dtype=np.uint8)
        )
        kwargs.setdefault("use_convpaint_auto_mask", True)
        kwargs.setdefault("convpaint_model_path", "model.pkl")
        kwargs.setdefault("image", np.zeros((2, 8, 8), dtype=np.float32))
        kwargs.setdefault("dim_order", "ZYX")
        return _capture_locals(monkeypatch, **kwargs)

    def test_labels_to_mask_binarises(self, monkeypatch):
        helpers = self._helpers(
            monkeypatch,
            convpaint_mask_dilation=0,
            convpaint_min_object_fraction_of_median=0.0,
        )
        labels = np.array([[0, 2], [5, 0]], dtype=np.int32)
        mask = helpers["_labels_to_mask"](labels)
        assert mask.dtype == np.uint8
        np.testing.assert_array_equal(mask, [[0, 1], [1, 0]])

    def test_labels_to_mask_dilates(self, monkeypatch):
        helpers = self._helpers(
            monkeypatch,
            convpaint_mask_dilation=1,
            convpaint_min_object_fraction_of_median=0.0,
        )
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[2, 2] = 1
        mask = helpers["_labels_to_mask"](labels)
        assert mask.sum() == 5

    def test_labels_to_mask_drops_small_components(self, monkeypatch):
        helpers = self._helpers(
            monkeypatch,
            convpaint_mask_dilation=0,
            convpaint_min_object_fraction_of_median=0.5,
        )
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[0:6, 0:6] = 1
        labels[19, 19] = 1
        mask = helpers["_labels_to_mask"](labels)
        assert mask[19, 19] == 0
        assert mask[0:6, 0:6].sum() == 36

    def test_block_grid_is_none_for_non_3d(self, monkeypatch):
        helpers = self._helpers(monkeypatch)
        grid = helpers["_compute_active_block_grid"]
        assert grid(np.ones((4, 4), dtype=np.uint8), 2) is None
        assert grid(np.ones((2, 4, 4), dtype=np.uint8), 0) is None

    def test_block_grid_marks_occupied_blocks(self, monkeypatch):
        helpers = self._helpers(monkeypatch)
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[0, 0:2, 0:2] = 1
        grid = helpers["_compute_active_block_grid"](mask, 4)
        assert grid.shape == (1, 2, 2)
        assert grid.sum() == 1
        assert grid[0, 0, 0]

    def test_summarize_mask_reports_blocks(self, monkeypatch):
        helpers = self._helpers(monkeypatch)
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[:, 0:4, 0:4] = 1
        fg, active, total = helpers["_summarize_mask"](mask, 4)
        assert total == 4
        assert active == 1
        assert fg == pytest.approx(0.25)

    def test_summarize_mask_without_blocks_for_2d(self, monkeypatch):
        helpers = self._helpers(monkeypatch)
        fg, active, total = helpers["_summarize_mask"](
            np.ones((4, 4), dtype=np.uint8), 4
        )
        assert (active, total) == (None, None)
        assert fg == 1.0

    def test_expand_returns_input_when_grid_is_unavailable(
        self, monkeypatch
    ):
        helpers = self._helpers(monkeypatch)
        mask = np.ones((4, 4), dtype=np.uint8)
        out = helpers["_expand_mask_to_neighbor_blocks"](mask, 2)
        np.testing.assert_array_equal(out, mask)

    def test_expand_fills_whole_blocks(self, monkeypatch):
        helpers = self._helpers(monkeypatch)
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[0, 0, 0] = 1
        out = helpers["_expand_mask_to_neighbor_blocks"](
            mask, 4, margin_blocks=0
        )
        assert out[:, 0:4, 0:4].all()
        assert out[:, 4:, :].sum() == 0

    def test_expand_dilates_to_neighbor_blocks(self, monkeypatch):
        helpers = self._helpers(monkeypatch)
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[0, 0, 0] = 1
        out = helpers["_expand_mask_to_neighbor_blocks"](
            mask, 4, margin_blocks=1
        )
        assert out.all()

    def test_runtime_mask_is_cached_when_it_differs(
        self, monkeypatch, tmp_path
    ):
        helpers = self._helpers(monkeypatch)
        cache = tmp_path / "mask_cache"
        cache.mkdir()
        cached = str(cache / "t0003_mask.tif")
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[0, 0, 0] = 1
        tifffile.imwrite(cached, mask)
        (
            runtime_mask,
            runtime_path,
            fg,
            active,
            total,
        ) = helpers["_prepare_runtime_distributed_mask"](
            mask, cached, str(cache), 3, 4
        )
        assert runtime_path != cached
        assert os.path.exists(runtime_path)
        assert runtime_mask[:, 0:4, 0:4].all()
        assert (active, total) == (1, 4)
        assert fg == pytest.approx(0.25)

    def test_runtime_mask_reuses_path_when_unchanged(
        self, monkeypatch, tmp_path
    ):
        helpers = self._helpers(monkeypatch)
        cache = tmp_path / "mask_cache"
        cache.mkdir()
        cached = str(cache / "t0000_mask.tif")
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[:, 0:4, 0:4] = 1
        tifffile.imwrite(cached, mask)
        _, runtime_path, _, _, _ = helpers[
            "_prepare_runtime_distributed_mask"
        ](mask, cached, str(cache), 0, 4)
        assert runtime_path == cached


class TestConvpaintNonZarrMask:
    """Whole-image ConvPaint mask generation for non-zarr sources.

    This pins a SOURCE DEFECT rather than a desirable design.  With a
    non-zarr source the module still runs the (expensive) ConvPaint
    prediction, but ``convpaint_mask`` is only ever consumed inside the
    ``if use_zarr_direct:`` branch -- which by construction cannot be
    reached when the source is not a zarr.  So the mask is computed and
    then dropped, and ``use_convpaint_auto_mask=True`` is a no-op for
    TIFF inputs.  Asserting it keeps the waste visible instead of silent.
    """

    def test_mask_is_computed_but_never_reaches_cellpose(
        self, monkeypatch, tmp_path
    ):
        calls = _stub_convpaint(
            monkeypatch, lambda img: np.ones(img.shape, dtype=np.uint8)
        )
        src = _write_uint32_tif(tmp_path / "movie.tif", (8, 8))
        seen = {}

        def fake_run(command, args):
            seen.update(args)
            return np.zeros((8, 8), dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        result = cellpose_mod.cellpose_segmentation(
            np.zeros((8, 8), dtype=np.float32),
            dim_order="YX",
            use_convpaint_auto_mask=True,
            convpaint_model_path="model.pkl",
            _source_filepath=src,
        )
        assert result.dtype == np.uint32
        assert len(calls) == 1
        assert calls[0]["model_path"] == "model.pkl"
        assert calls[0]["output_type"] == "semantic"
        # ConvPaint ran on the whole image ...
        assert calls[0]["shape"] == (8, 8)
        # ... and then went nowhere: no mask argument, no mask file.
        assert "distributed_mask_path" not in seen
        assert "distributed_mask_zarr_path" not in seen
        assert glob.glob(str(tmp_path / "tmp" / "*_convpaint_mask.tif")) == []


# ------------------------------------------------ zarr-direct mask zarrs


class TestConvpaintMaskZarrGeneration:
    """Slab-wise ConvPaint mask zarr built for zarr-direct workflows."""

    def _run(self, monkeypatch, tmp_path, image, dim_order):
        calls = _stub_convpaint(
            monkeypatch, lambda img: np.ones(img.shape, dtype=np.uint8)
        )
        src = _make_zarr_source(tmp_path, shape=image.shape)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            return np.zeros(image.shape[1:], dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: kw["output_path"],
        )
        out = cellpose_mod.cellpose_segmentation(
            image,
            dim_order=dim_order,
            use_convpaint_auto_mask=True,
            convpaint_model_path="model.pkl",
            convpaint_mask_dilation=0,
            convpaint_min_object_fraction_of_median=0.0,
            _source_filepath=src,
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_lbl",
        )
        return calls, seen, out, tmp_path / "tmp"

    def test_three_dimensional_volume_makes_one_mask(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 8, 8), dtype=np.float32)
        calls, seen, out, tmp_root = self._run(
            monkeypatch, tmp_path, image, "ZYX"
        )
        assert len(calls) == 1
        assert calls[0]["shape"] == (2, 8, 8)
        assert seen[0]["do_3D"] is True
        assert seen[0]["distributed_mask_zarr_path"] is not None
        # The mask zarr is removed once the run finishes.
        assert not os.path.exists(seen[0]["distributed_mask_zarr_path"])
        assert out == str(tmp_path / "out" / "src_lbl.tif")

    def test_four_dimensional_stack_masks_each_timepoint(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((3, 2, 8, 8), dtype=np.float32)
        calls, seen, _, _ = self._run(
            monkeypatch, tmp_path, image, "TZYX"
        )
        assert len(calls) == 3
        assert {c["shape"] for c in calls} == {(2, 8, 8)}
        assert [a["timepoint_index"] for a in seen] == [0, 1, 2]

    def test_five_dimensional_stack_masks_each_channel(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 2, 8, 8), dtype=np.float32)
        calls, seen, _, _ = self._run(
            monkeypatch, tmp_path, image, "TCZYX"
        )
        assert len(calls) == 4
        assert {c["shape"] for c in calls} == {(2, 8, 8)}
        # One mask per (T, C) slab, but still one Cellpose run per timepoint.
        assert [a["timepoint_index"] for a in seen] == [0, 1]

    def test_two_dimensional_zarr_mask_is_rejected(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="expects 3D, 4D, or 5D"):
            self._run(monkeypatch, tmp_path, image, "YX")


# --------------------------------------------- auto zarr conversion path


class TestAutoZarrConversion:
    """Distributed mode auto-converts non-zarr sources to zarr."""

    def _prepare(self, monkeypatch, tmp_path, image):
        src = _write_uint32_tif(tmp_path / "movie.tif", image.shape)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            return np.zeros(image.shape, dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: kw["output_path"],
        )
        return src, seen

    def test_converted_zarr_becomes_the_source(self, monkeypatch, tmp_path):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        src, seen = self._prepare(monkeypatch, tmp_path, image)
        converted = []

        def fake_save(data, filepath, axes):
            converted.append({"axes": axes, "shape": data.shape})
            zarr.open_array(
                filepath,
                mode="w",
                shape=data.shape,
                chunks=data.shape,
                dtype=data.dtype,
            )[:] = data

        import napari_tmidas._file_selector as fs

        monkeypatch.setattr(fs, "save_as_zarr", fake_save)
        out = cellpose_mod.cellpose_segmentation(
            image,
            dim_order="ZYX",
            use_distributed_segmentation=True,
            _source_filepath=src,
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_lbl",
        )
        assert converted == [{"axes": "ZYX", "shape": (2, 8, 8)}]
        auto = tmp_path / "tmp" / "cellpose_auto_zarr"
        made = sorted(p.name for p in auto.iterdir())
        assert len(made) == 1
        assert made[0].startswith("movie_cellpose_chall_")
        assert seen[0]["zarr_path"].endswith(made[0])
        assert seen[0]["use_distributed_segmentation"] is True
        assert out == str(tmp_path / "out" / "movie_lbl.tif")

    def test_axes_fall_back_to_tczyx_on_rank_mismatch(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        src, _ = self._prepare(monkeypatch, tmp_path, image)
        converted = []

        def fake_save(data, filepath, axes):
            converted.append(axes)
            zarr.open_array(
                filepath,
                mode="w",
                shape=data.shape,
                chunks=data.shape,
                dtype=data.dtype,
            )[:] = data

        import napari_tmidas._file_selector as fs

        monkeypatch.setattr(fs, "save_as_zarr", fake_save)
        cellpose_mod.cellpose_segmentation(
            image,
            dim_order="YX",
            use_distributed_segmentation=True,
            _source_filepath=src,
        )
        assert converted == ["TCZYX"]

    def test_existing_cache_is_reused_without_reconverting(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        src, seen = self._prepare(monkeypatch, tmp_path, image)
        calls = []

        def fake_save(data, filepath, axes):
            calls.append(filepath)
            zarr.open_array(
                filepath,
                mode="w",
                shape=data.shape,
                chunks=data.shape,
                dtype=data.dtype,
            )[:] = data

        import napari_tmidas._file_selector as fs

        monkeypatch.setattr(fs, "save_as_zarr", fake_save)
        kwargs = {
            "dim_order": "ZYX",
            "use_distributed_segmentation": True,
            "_source_filepath": src,
        }
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert len(calls) == 1
        assert seen[0]["zarr_path"] == seen[1]["zarr_path"]

    def test_legacy_cache_with_matching_shape_is_adopted(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        src, seen = self._prepare(monkeypatch, tmp_path, image)
        auto_root = tmp_path / "tmp" / "cellpose_auto_zarr"
        auto_root.mkdir(parents=True)
        legacy = str(auto_root / "movie_cellpose_chall_deadbeef1234.zarr")
        zarr.open_array(
            legacy,
            mode="w",
            shape=image.shape,
            chunks=image.shape,
            dtype=image.dtype,
        )[:] = image

        def fake_save(data, filepath, axes):
            raise AssertionError("should have reused the legacy cache")

        import napari_tmidas._file_selector as fs

        monkeypatch.setattr(fs, "save_as_zarr", fake_save)
        cellpose_mod.cellpose_segmentation(
            image,
            dim_order="ZYX",
            use_distributed_segmentation=True,
            _source_filepath=src,
        )
        assert seen[0]["zarr_path"] == legacy

    def test_conversion_failure_falls_back_to_plain_evaluation(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        src, seen = self._prepare(monkeypatch, tmp_path, image)

        def boom(data, filepath, axes):
            raise RuntimeError("conversion exploded")

        import napari_tmidas._file_selector as fs

        monkeypatch.setattr(fs, "save_as_zarr", boom)
        result = cellpose_mod.cellpose_segmentation(
            image,
            dim_order="ZYX",
            use_distributed_segmentation=True,
            _source_filepath=src,
        )
        assert isinstance(result, np.ndarray)
        assert "zarr_path" not in seen[0]
        assert seen[0]["image"].shape == (2, 8, 8)


# ----------------------------------------------- zarr-direct timepoints


class TestZarrDirectTimepointCache:
    """Per-timepoint TIFF cache for zarr-direct time-series runs."""

    def test_second_run_reuses_cached_timepoint_outputs(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((3, 2, 8, 8), dtype=np.float32)
        src = _make_zarr_source(tmp_path, shape=image.shape)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            return np.full((2, 8, 8), 5, dtype=np.int32)

        _stub_env(monkeypatch, fake_run)
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: kw["output_path"],
        )
        kwargs = {
            "dim_order": "TZYX",
            "timepoint_start": 0,
            "timepoint_end": 2,
            "_source_filepath": src,
            "_output_folder": str(tmp_path / "out"),
            "_output_suffix": "_lbl",
        }
        first = cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert first == str(tmp_path / "out" / "src_lbl.tif")
        assert len(seen) == 3

        cache_root = glob.glob(
            str(tmp_path / "tmp" / "cellpose_timepoint_cache" / "src_*")
        )
        assert len(cache_root) == 1
        cached = sorted(os.listdir(cache_root[0]))
        assert cached == [
            "t0000_labels.tif",
            "t0001_labels.tif",
            "t0002_labels.tif",
        ]
        assert tifffile.imread(
            os.path.join(cache_root[0], "t0000_labels.tif")
        ).dtype == np.uint32

        cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert len(seen) == 3, "cached timepoints must not be recomputed"


# ------------------------------------------- interleaved distributed run


def _interleaved_kwargs(src, tmp_path, **overrides):
    kwargs = {
        "dim_order": "TZYX",
        "use_distributed_segmentation": True,
        "use_convpaint_auto_mask": True,
        "convpaint_model_path": "model.pkl",
        "convpaint_mask_dilation": 0,
        "convpaint_min_object_fraction_of_median": 0.0,
        "_source_filepath": src,
        "_output_folder": str(tmp_path / "out"),
        "_output_suffix": "_lbl",
    }
    kwargs.update(overrides)
    return kwargs


class TestInterleavedDistributedRun:
    """mask -> segment per timepoint, with checkpointing and caching."""

    def _setup(self, monkeypatch, tmp_path, image, mask_factory=None):
        src = _make_zarr_source(tmp_path, shape=image.shape)
        if mask_factory is None:

            def mask_factory(img):
                return np.ones(img.shape, dtype=np.uint8)

        calls = _stub_convpaint(monkeypatch, mask_factory)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            out = np.full(image.shape[1:], 4, dtype=np.int32)
            persist = args.get("persist_output_zarr_path")
            if persist:
                zarr.open_array(
                    persist,
                    mode="w",
                    shape=out.shape,
                    chunks=out.shape,
                    dtype=np.uint32,
                )[:] = out
            return out

        _stub_env(monkeypatch, fake_run)
        return src, calls, seen

    def test_small_slabs_run_non_distributed_and_write_a_tiff(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, seen = self._setup(monkeypatch, tmp_path, image)
        out = cellpose_mod.cellpose_segmentation(
            image, **_interleaved_kwargs(src, tmp_path)
        )
        assert len(calls) == 2
        assert len(seen) == 2
        # A single block covers the slab, so gating is dropped entirely.
        assert seen[0]["use_distributed_segmentation"] is False
        assert seen[0]["distributed_mask_path"] is None
        assert seen[0]["distributed_blocksize_z"] == 2
        assert [a["timepoint_index"] for a in seen] == [0, 1]
        assert out == str(tmp_path / "out" / "src_lbl.tif")
        written = tifffile.imread(out)
        assert written.shape == (2, 2, 16, 16)
        assert written.dtype == np.uint32
        assert int(written.max()) == 4

    def test_checkpoint_and_slab_caches_are_populated(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, _ = self._setup(monkeypatch, tmp_path, image)
        cellpose_mod.cellpose_segmentation(
            image, **_interleaved_kwargs(src, tmp_path)
        )
        tmp_root = tmp_path / "tmp"
        ckpt = tmp_root / "src_cellpose_interleaved_chall.zarr"
        assert ckpt.is_dir()
        checkpoint = zarr.open_array(str(ckpt), mode="r")
        assert checkpoint.shape == (2, 2, 16, 16)
        assert checkpoint.attrs["completed_slabs"] == 2
        summary = json.loads(checkpoint.attrs["slab_perf_summary"])
        assert summary["processed_slabs_this_run"] == 2
        assert summary["non_distributed_count"] == 2

        slab_roots = glob.glob(
            str(tmp_root / "cellpose_timepoint_cache" / "src_interleaved_*")
        )
        assert len(slab_roots) == 1
        names = sorted(os.listdir(slab_roots[0]))
        assert "run_settings.json" in names
        assert "t0000_labels.zarr" in names
        settings = json.loads(
            (
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / os.path.basename(slab_roots[0])
                / "run_settings.json"
            ).read_text()
        )
        assert settings["run_signature"]["source"] == "src.zarr"
        assert settings["run_signature"]["selected_timepoints"] == [0, 1]

    def test_rerun_skips_when_the_output_is_already_valid(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, seen = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        first = cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert len(seen) == 2
        second = cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert second == first
        assert len(seen) == 2, "existing valid output must short-circuit"

    def test_rerun_reuses_masks_and_persisted_slabs(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, seen = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert len(calls) == 2 and len(seen) == 2

        # Drop the checkpoint so the slab loop runs again from scratch.
        import shutil as _shutil

        _shutil.rmtree(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr")
        )
        rerun = dict(kwargs)
        rerun["skip_overwrite_existing_valid_output"] = False
        cellpose_mod.cellpose_segmentation(image, **rerun)
        assert len(calls) == 2, "cached ConvPaint masks must be reused"
        assert len(seen) == 2, "persisted slab outputs must be reused"

    def test_resume_from_a_complete_checkpoint_skips_the_loop(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, seen = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert len(seen) == 2 and len(calls) == 2

        # Delete every *other* reuse route -- the cached ConvPaint masks and
        # the persisted per-slab zarrs -- so that only the checkpoint's
        # ``completed_slabs`` counter can keep the loop from running again.
        import shutil as _shutil

        for stale_mask in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "convpaint_mask_cache"
                / "src_*"
                / "t*_mask.tif"
            )
        ):
            os.unlink(stale_mask)
        for slab in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "t*_labels.zarr"
            )
        ):
            _shutil.rmtree(slab)

        rerun = dict(kwargs)
        rerun["skip_overwrite_existing_valid_output"] = False
        out = cellpose_mod.cellpose_segmentation(image, **rerun)
        assert len(seen) == 2, "the checkpoint must skip the slab loop"
        assert len(calls) == 2, "the checkpoint must skip mask generation"
        # The checkpoint still holds the labels, so the TIFF is rewritten.
        assert int(tifffile.imread(out).max()) == 4

    def test_incompatible_checkpoint_is_discarded(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, seen = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        rerun = dict(kwargs)
        rerun["flow_threshold"] = 0.75
        rerun["auto_load_saved_interleaved_settings"] = False
        rerun["skip_overwrite_existing_valid_output"] = False
        cellpose_mod.cellpose_segmentation(image, **rerun)
        checkpoint = zarr.open_array(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr"),
            mode="r",
        )
        signature = json.loads(checkpoint.attrs["run_signature"])
        assert signature["flow_threshold"] == 0.75
        # The signature attribute is rewritten either way.  What proves the
        # incompatible checkpoint was thrown out is that both slabs had to
        # be segmented again: a kept checkpoint reports completed_slabs=2
        # up front and the loop body never runs.
        assert len(seen) == 4
        assert checkpoint.attrs["completed_slabs"] == 2
        assert len(calls) == 2, "mask cache key is independent of flow_threshold"

    def test_unreadable_checkpoint_is_recreated(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, seen = self._setup(monkeypatch, tmp_path, image)
        ckpt = tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr"
        ckpt.mkdir(parents=True)
        (ckpt / "junk.txt").write_text("not zarr")
        cellpose_mod.cellpose_segmentation(
            image, **_interleaved_kwargs(src, tmp_path)
        )
        assert len(seen) == 2
        # The whole directory was removed and rebuilt, so the junk is gone.
        assert not (ckpt / "junk.txt").exists()
        assert zarr.open_array(str(ckpt), mode="r").shape == (2, 2, 16, 16)

    def test_saved_settings_are_auto_loaded_on_restart(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, seen = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(
            src,
            tmp_path,
            flow_threshold=0.55,
            batch_size=8,
            anisotropy=2.0,
        )
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        os.unlink(str(tmp_path / "out" / "src_lbl.tif"))
        import shutil as _shutil

        _shutil.rmtree(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr")
        )
        for slab in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "t*_labels.zarr"
            )
        ):
            _shutil.rmtree(slab)
        restart = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **restart)
        # The restart passed defaults; saved settings win.
        assert seen[-1]["flow_threshold"] == 0.55
        assert seen[-1]["batch_size"] == 8
        assert seen[-1]["anisotropy"] == 2.0

    def test_saved_settings_are_ignored_for_another_model(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, seen = self._setup(monkeypatch, tmp_path, image)
        cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(src, tmp_path, flow_threshold=0.55),
        )
        restart = _interleaved_kwargs(
            src,
            tmp_path,
            model_type="cpsam",
            skip_overwrite_existing_valid_output=False,
        )
        cellpose_mod.cellpose_segmentation(image, **restart)
        assert seen[-1]["model_type"] == "cpsam"
        assert seen[-1]["flow_threshold"] == 0.4

    def test_corrupt_saved_settings_are_survived(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, seen = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        for path in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "run_settings.json"
            )
        ):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
        rerun = dict(kwargs)
        rerun["skip_overwrite_existing_valid_output"] = False
        out = cellpose_mod.cellpose_segmentation(image, **rerun)
        assert len(seen) == 2
        assert int(tifffile.imread(out).max()) == 4

    def test_non_dict_saved_run_signature_is_ignored(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, seen = self._setup(monkeypatch, tmp_path, image)
        cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(src, tmp_path, flow_threshold=0.55),
        )
        for path in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "run_settings.json"
            )
        ):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"run_signature": ["not", "a", "dict"]}, handle)
        restart = _interleaved_kwargs(
            src, tmp_path, skip_overwrite_existing_valid_output=False
        )
        cellpose_mod.cellpose_segmentation(image, **restart)
        # Nothing usable in the cache: the caller's own value survives, and
        # the now-incompatible checkpoint forces both slabs to be redone.
        assert seen[-1]["flow_threshold"] == 0.4
        assert len(seen) == 4

    def test_unreadable_cached_mask_is_regenerated(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, _ = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        masks = glob.glob(
            str(
                tmp_path
                / "tmp"
                / "convpaint_mask_cache"
                / "src_*"
                / "t*_mask.tif"
            )
        )
        assert masks
        for path in masks:
            with open(path, "wb") as handle:
                handle.write(b"junk")
        import shutil as _shutil

        _shutil.rmtree(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr")
        )
        for slab in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "t*_labels.zarr"
            )
        ):
            _shutil.rmtree(slab)
        rerun = dict(kwargs)
        rerun["skip_overwrite_existing_valid_output"] = False
        cellpose_mod.cellpose_segmentation(image, **rerun)
        assert len(calls) == 4, "corrupt masks must be regenerated"

    def test_cached_mask_with_wrong_shape_is_regenerated(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, _ = self._setup(monkeypatch, tmp_path, image)
        kwargs = _interleaved_kwargs(src, tmp_path)
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        for path in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "convpaint_mask_cache"
                / "src_*"
                / "t*_mask.tif"
            )
        ):
            tifffile.imwrite(path, np.ones((3, 3), dtype=np.uint8))
        import shutil as _shutil

        _shutil.rmtree(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr")
        )
        for slab in glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "t*_labels.zarr"
            )
        ):
            _shutil.rmtree(slab)
        rerun = dict(kwargs)
        rerun["skip_overwrite_existing_valid_output"] = False
        cellpose_mod.cellpose_segmentation(image, **rerun)
        assert len(calls) == 4

    def test_sparse_mask_disables_block_pruning(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((1, 2, 64, 64), dtype=np.float32)

        def empty_mask(img):
            return np.zeros(img.shape, dtype=np.uint8)

        src, _, seen = self._setup(
            monkeypatch, tmp_path, image, mask_factory=empty_mask
        )
        cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src, tmp_path, distributed_blocksize_yx=64
            ),
        )
        assert seen[0]["distributed_mask_path"] is None
        assert seen[0]["use_distributed_segmentation"] is True

    def test_partial_mask_keeps_distributed_mode_and_gating(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((3, 2, 128, 128), dtype=np.float32)

        def corner_mask(img):
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[:, :64, :64] = 3
            return mask

        src, _, seen = self._setup(
            monkeypatch, tmp_path, image, mask_factory=corner_mask
        )
        cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src, tmp_path, distributed_blocksize_yx=64
            ),
        )
        assert len(seen) == 3
        assert all(a["use_distributed_segmentation"] for a in seen)
        assert all(
            a["distributed_mask_path"] is not None for a in seen
        )
        checkpoint = zarr.open_array(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr"),
            mode="r",
        )
        summary = json.loads(checkpoint.attrs["slab_perf_summary"])
        assert summary["distributed_count"] == 3
        assert summary["final_distributed_blocksize"] == 64
        records = json.loads(checkpoint.attrs["slab_perf_records"])
        assert [r["mode"] for r in records] == ["distributed"] * 3
        assert records[0]["total_blocks"] == 4
        assert records[0]["active_blocks"] == 1

    def test_labels_are_clipped_to_the_convpaint_mask(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((1, 2, 128, 128), dtype=np.float32)

        def corner_mask(img):
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[:, :64, :64] = 1
            return mask

        src, _, _ = self._setup(
            monkeypatch, tmp_path, image, mask_factory=corner_mask
        )
        out = cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src, tmp_path, distributed_blocksize_yx=64
            ),
        )
        written = tifffile.imread(out)
        assert written[..., :64, :64].min() == 4
        assert written[..., 64:, :].max() == 0

    def test_clipping_can_be_disabled(self, monkeypatch, tmp_path):
        image = np.zeros((1, 2, 128, 128), dtype=np.float32)

        def corner_mask(img):
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[:, :64, :64] = 1
            return mask

        src, _, _ = self._setup(
            monkeypatch, tmp_path, image, mask_factory=corner_mask
        )
        out = cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src,
                tmp_path,
                distributed_blocksize_yx=64,
                clip_final_labels_to_convpaint_mask=False,
            ),
        )
        written = tifffile.imread(out)
        assert written.min() == 4

    def test_distributed_failure_retries_without_distribution(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((1, 2, 128, 128), dtype=np.float32)

        def corner_mask(img):
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[:, :64, :64] = 1
            return mask

        src = _make_zarr_source(tmp_path, shape=image.shape)
        _stub_convpaint(monkeypatch, corner_mask)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            if args.get("use_distributed_segmentation"):
                raise RuntimeError("dask cluster died")
            return np.full(image.shape[1:], 6, dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        out = cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src, tmp_path, distributed_blocksize_yx=64
            ),
        )
        assert len(seen) == 2
        assert seen[1]["use_distributed_segmentation"] is False
        assert seen[1]["distributed_mask_path"] is None
        checkpoint = zarr.open_array(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr"),
            mode="r",
        )
        summary = json.loads(checkpoint.attrs["slab_perf_summary"])
        assert summary["fallback_non_distributed_count"] == 1
        assert tifffile.imread(out)[..., :64, :64].min() == 6

    def test_non_distributed_failure_is_not_retried(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((1, 2, 16, 16), dtype=np.float32)
        src = _make_zarr_source(tmp_path, shape=image.shape)
        _stub_convpaint(
            monkeypatch, lambda img: np.ones(img.shape, dtype=np.uint8)
        )
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            raise RuntimeError("cellpose crashed")

        _stub_env(monkeypatch, fake_run)
        with pytest.raises(RuntimeError, match="cellpose crashed"):
            cellpose_mod.cellpose_segmentation(
                image, **_interleaved_kwargs(src, tmp_path)
            )
        assert len(seen) == 1


# ------------------------------------------------------------- registry


class TestRegistryWrapper:
    """The registered entry the batch widget actually calls."""

    def test_function_is_registered_with_its_parameters(self):
        info = BatchProcessingRegistry.get_function_info(
            "Cellpose-SAM Segmentation"
        )
        assert info is not None
        assert info["suffix"] == "_labels"
        assert info["func"] is cellpose_mod.cellpose_segmentation
        params = info["parameters"]
        assert params["model_type"]["default"] == "cpsam_v2"
        assert (
            params["model_type"]["options"]
            == cellpose_mod.SUPPORTED_CELLPOSE_MODELS
        )
        # The dimension order comes from the batch widget's global
        # "Dimension Order" dropdown (resolved by
        # ``resolve_cellpose_dim_order``), so it is deliberately not a
        # per-function parameter here.
        assert "dim_order" not in params

    def test_function_is_marked_not_thread_safe(self):
        assert cellpose_mod.cellpose_segmentation.thread_safe is False

    def test_env_manager_updater_is_a_no_op(self):
        assert cellpose_mod.update_cellpose_env_manager() is None


# ------------------------------------------------------- narrow branches


class _FakeClock:
    """Monotonic stand-in for ``time`` that makes every step cost 100 s."""

    def __init__(self, step=100.0):
        self._now = 0.0
        self._step = step

    def perf_counter(self):
        self._now += self._step
        return self._now

    def time(self):
        return 1_700_000_000.0

    def time_ns(self):
        return 1_700_000_000_000_000_000


class TestSignatureMatchingFallbacks:
    """Path comparison degrades to basenames when os.path blows up."""

    def _matches(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "movie.tif", (2, 2))
        return _capture_locals(monkeypatch, _source_filepath=src)[
            "_source_signature_matches"
        ]

    def test_abspath_and_realpath_errors_fall_back_to_basename(
        self, monkeypatch, tmp_path
    ):
        matches = self._matches(monkeypatch, tmp_path)

        def boom(_path):
            raise OSError("path resolution failed")

        monkeypatch.setattr(os.path, "abspath", boom)
        monkeypatch.setattr(os.path, "realpath", boom)
        assert matches("/somewhere/else/movie.tif") is True
        assert matches("/somewhere/else/other.tif") is False


class TestScipyFailurePaths:
    """ConvPaint mask post-processing degrades when scipy misbehaves."""

    def _helpers(self, monkeypatch, **kwargs):
        _stub_convpaint(
            monkeypatch, lambda img: np.ones(img.shape, dtype=np.uint8)
        )
        kwargs.setdefault("use_convpaint_auto_mask", True)
        kwargs.setdefault("convpaint_model_path", "model.pkl")
        kwargs.setdefault("image", np.zeros((2, 8, 8), dtype=np.float32))
        kwargs.setdefault("dim_order", "ZYX")
        return _capture_locals(monkeypatch, **kwargs)

    def test_failed_dilation_keeps_the_raw_mask(self, monkeypatch):
        import scipy.ndimage

        def boom(*args, **kwargs):
            raise MemoryError("no room for dilation")

        monkeypatch.setattr(scipy.ndimage, "binary_dilation", boom)
        helpers = self._helpers(
            monkeypatch,
            convpaint_mask_dilation=2,
            convpaint_min_object_fraction_of_median=0.0,
        )
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[2, 2] = 1
        mask = helpers["_labels_to_mask"](labels)
        assert mask.sum() == 1

    def test_failed_component_labelling_keeps_every_pixel(
        self, monkeypatch
    ):
        import scipy.ndimage

        def boom(*args, **kwargs):
            raise MemoryError("no room for labelling")

        monkeypatch.setattr(scipy.ndimage, "label", boom)
        helpers = self._helpers(
            monkeypatch,
            convpaint_mask_dilation=0,
            convpaint_min_object_fraction_of_median=0.5,
        )
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[0:6, 0:6] = 1
        labels[19, 19] = 1
        mask = helpers["_labels_to_mask"](labels)
        assert mask[19, 19] == 1

    def test_failed_block_dilation_keeps_original_activity(
        self, monkeypatch
    ):
        helpers = self._helpers(
            monkeypatch,
            convpaint_mask_dilation=0,
            convpaint_min_object_fraction_of_median=0.0,
        )
        import scipy.ndimage

        def boom(*args, **kwargs):
            raise MemoryError("no room for block dilation")

        monkeypatch.setattr(scipy.ndimage, "binary_dilation", boom)
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[0, 0, 0] = 1
        out = helpers["_expand_mask_to_neighbor_blocks"](
            mask, 4, margin_blocks=1
        )
        assert out[:, 0:4, 0:4].all()
        assert out[:, 4:, :].sum() == 0


class TestRecoveryMetadata:
    """Recovered TIFFs keep the source's physical pixel size."""

    def test_physical_scale_is_embedded(self, monkeypatch, tmp_path):
        src = _write_uint32_tif(tmp_path / "src.tif", (2, 2))
        monkeypatch.setattr(
            cellpose_mod,
            "_extract_source_physical_scale",
            lambda path, axes: {"X": 0.25, "Y": 0.25},
        )
        recover = _capture_locals(monkeypatch, _source_filepath=src)[
            "_recover_tiff_from_cached_timepoint_zarr"
        ]
        root = _write_slab_cache(tmp_path / "cache", [0, 1], (4, 5))
        out = str(tmp_path / "recovered.tif")
        recover(root, out)
        with tifffile.TiffFile(out) as tif:
            assert tif.is_ome
            assert 'PhysicalSizeX="0.25"' in tif.ome_metadata


class TestEnvironmentBootstrap:
    """A missing Cellpose environment is created before evaluation."""

    def test_missing_environment_is_created(self, monkeypatch):
        created = []
        monkeypatch.setattr(cellpose_mod, "is_env_created", lambda: False)
        monkeypatch.setattr(
            cellpose_mod,
            "create_cellpose_env",
            lambda: created.append("built"),
        )
        monkeypatch.setattr(
            cellpose_mod,
            "run_cellpose_in_env",
            lambda command, args: np.zeros((4, 4), dtype=np.uint32),
        )
        cellpose_mod.cellpose_segmentation(
            np.zeros((4, 4), dtype=np.float32), dim_order="YX"
        )
        assert created == ["built"]


class TestResultDtypeNormalisation:
    """Whatever Cellpose returns, callers get uint32 labels."""

    def test_int32_results_are_converted(self, monkeypatch):
        _stub_env(
            monkeypatch,
            lambda command, args: np.full((4, 4), 3, dtype=np.int32),
        )
        result = cellpose_mod.cellpose_segmentation(
            np.zeros((4, 4), dtype=np.float32), dim_order="YX"
        )
        assert result.dtype == np.uint32
        assert int(result.max()) == 3


class TestTimeAxisPlacement:
    """A trailing time axis is moved to the front before slicing."""

    def test_trailing_time_axis_is_rotated_to_front(self, monkeypatch):
        shapes = []

        def fake_run(command, args):
            shapes.append(args["image"].shape)
            return np.zeros(args["image"].shape, dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        rng = np.random.default_rng(0)
        image = rng.random((4, 5, 3)).astype(np.float32)
        result = cellpose_mod.cellpose_segmentation(
            image, dim_order="YXT"
        )
        assert result.shape == (3, 4, 5)
        assert shapes == [(4, 5)] * 3


class TestZarrDirectTimeAxisFallback:
    """An out-of-range T index in dim_order is rejected, not guessed at."""

    def test_time_axis_beyond_rank_is_rejected(self, monkeypatch, tmp_path):
        """
        dim_order names a T axis the array does not have. The zarr-direct
        path used to silently reinterpret axis 0 (the user's Z) as T
        instead, while the in-memory path already raised for the same
        input (see
        TestParameterValidation.test_time_axis_beyond_image_rank_is_rejected)
        -- now both do.
        """
        image = np.zeros((3, 8, 8), dtype=np.float32)
        src = _make_zarr_source(tmp_path, shape=image.shape)

        def fake_run(command, args):
            raise AssertionError(
                "must not shell out once the dim_order guard rejects input"
            )

        _stub_env(monkeypatch, fake_run)

        with pytest.raises(ValueError, match="does not have a matching T"):
            cellpose_mod.cellpose_segmentation(
                image,
                dim_order="ZYXT",
                _source_filepath=src,
                _output_folder=str(tmp_path / "out"),
                _output_suffix="_lbl",
            )


class TestStaleTimepointCacheDtype:
    """Cached timepoint TIFFs are re-normalised to uint32 on reuse."""

    def test_uint16_cache_entries_are_converted(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 8, 8), dtype=np.float32)
        src = _make_zarr_source(tmp_path, shape=image.shape)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            return np.full((8, 8), 2, dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        written = {}
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: written.update(kw) or kw["output_path"],
        )
        kwargs = {
            "dim_order": "TYX",
            "_source_filepath": src,
            "_output_folder": str(tmp_path / "out"),
            "_output_suffix": "_lbl",
        }
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        cached = glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_*"
                / "t*_labels.tif"
            )
        )
        assert len(cached) == 2
        for path in cached:
            tifffile.imwrite(
                path, np.full((8, 8), 2, dtype=np.uint16)
            )
        cellpose_mod.cellpose_segmentation(image, **kwargs)
        assert len(seen) == 2, "cached timepoints must not be recomputed"
        assert written["labels"].dtype == np.uint32


class TestMaskZarrEdgeCases:
    """Mask zarr creation reuses chunking and clears stale output."""

    def _run(self, monkeypatch, tmp_path, image, dim_order="ZYX"):
        _stub_convpaint(
            monkeypatch, lambda img: np.ones(img.shape, dtype=np.uint8)
        )
        src = _make_zarr_source(tmp_path, shape=tuple(image.shape))
        seen = []
        probes = []

        def fake_run(command, args):
            seen.append(dict(args))
            # The run deletes the mask zarr in its ``finally`` block, so the
            # only place it can be inspected is from inside the stub.
            mask_path = args.get("distributed_mask_zarr_path")
            probe = {"leftover": None, "chunks": None, "values": None}
            if mask_path:
                probe["leftover"] = os.path.exists(
                    os.path.join(mask_path, "leftover.txt")
                )
                mask_arr = zarr.open_array(mask_path, mode="r")
                probe["chunks"] = tuple(int(c) for c in mask_arr.chunks)
                probe["values"] = {
                    int(v) for v in np.unique(np.asarray(mask_arr))
                }
            probes.append(probe)
            return np.zeros(tuple(image.shape), dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        monkeypatch.setattr(
            cellpose_mod,
            "write_labels_with_source_metadata",
            lambda **kw: kw["output_path"],
        )
        cellpose_mod.cellpose_segmentation(
            image,
            dim_order=dim_order,
            use_convpaint_auto_mask=True,
            convpaint_model_path="model.pkl",
            convpaint_mask_dilation=0,
            convpaint_min_object_fraction_of_median=0.0,
            _source_filepath=src,
            _output_folder=str(tmp_path / "out"),
            _output_suffix="_lbl",
        )
        return seen, probes

    def test_stale_mask_zarr_is_removed_first(self, monkeypatch, tmp_path):
        stale = tmp_path / "tmp" / "convpaint_auto_mask.zarr"
        stale.mkdir(parents=True)
        (stale / "leftover.txt").write_text("stale")
        image = np.zeros((2, 8, 8), dtype=np.float32)
        seen, probes = self._run(monkeypatch, tmp_path, image)
        assert seen[0]["distributed_mask_zarr_path"] == str(stale)
        # The leftover is already gone while Cellpose runs, which only
        # happens if the stale tree is deleted before the mask is rebuilt.
        # (Checking after the run proves nothing: the ``finally`` block
        # removes the directory either way.)
        assert probes[0]["leftover"] is False
        # A plain ndarray has no .chunks, so the 16-capped fallback applies.
        assert probes[0]["chunks"] == (2, 8, 8)
        assert probes[0]["values"] == {1}
        assert not stale.exists()

    def test_chunked_input_drives_the_mask_chunking(
        self, monkeypatch, tmp_path
    ):
        chunked = zarr.open_array(
            str(tmp_path / "chunked.zarr"),
            mode="w",
            shape=(4, 8, 8),
            chunks=(2, 4, 4),
            dtype=np.uint8,
        )
        chunked[:] = 1
        seen, probes = self._run(monkeypatch, tmp_path, chunked)
        mask_path = seen[0]["distributed_mask_zarr_path"]
        assert mask_path is not None
        # Chunking is copied from the input array; the fallback for this
        # shape would have been (4, 8, 8), so this pins the .chunks branch.
        assert probes[0]["chunks"] == (2, 4, 4)
        assert probes[0]["values"] == {1}
        assert not os.path.exists(mask_path)


class TestAutoZarrLegacyCacheErrors:
    """Unreadable legacy caches are skipped instead of adopted."""

    def test_broken_legacy_cache_is_ignored(self, monkeypatch, tmp_path):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        src = _write_uint32_tif(tmp_path / "movie.tif", image.shape)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            return np.zeros(image.shape, dtype=np.uint32)

        _stub_env(monkeypatch, fake_run)
        auto_root = tmp_path / "tmp" / "cellpose_auto_zarr"
        auto_root.mkdir(parents=True)
        broken = auto_root / "movie_cellpose_chall_broken.zarr"
        broken.mkdir()
        (broken / "not-zarr.txt").write_text("garbage")
        made = []

        def fake_save(data, filepath, axes):
            made.append(filepath)
            zarr.open_array(
                filepath,
                mode="w",
                shape=data.shape,
                chunks=data.shape,
                dtype=data.dtype,
            )[:] = data

        import napari_tmidas._file_selector as fs

        monkeypatch.setattr(fs, "save_as_zarr", fake_save)
        cellpose_mod.cellpose_segmentation(
            image,
            dim_order="ZYX",
            use_distributed_segmentation=True,
            _source_filepath=src,
        )
        assert len(made) == 1
        assert not made[0].endswith("broken.zarr")
        assert seen[0]["zarr_path"] == made[0]


class TestInterleavedNarrowBranches:
    """Remaining interleaved branches: legacy skip, settings, tuning."""

    def _setup(self, monkeypatch, tmp_path, image, mask_factory):
        src = _make_zarr_source(tmp_path, shape=tuple(image.shape))
        calls = _stub_convpaint(monkeypatch, mask_factory)
        seen = []

        def fake_run(command, args):
            seen.append(dict(args))
            return np.full(
                tuple(image.shape[1:]), 4, dtype=np.uint32
            )

        _stub_env(monkeypatch, fake_run)
        return src, calls, seen

    def test_valid_legacy_output_short_circuits_the_run(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, calls, seen = self._setup(
            monkeypatch,
            tmp_path,
            image,
            lambda img: np.ones(img.shape, dtype=np.uint8),
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        legacy = _write_uint32_tif(
            out_dir / "src_old_lbl.tif", (2, 2, 16, 16)
        )
        result = cellpose_mod.cellpose_segmentation(
            image, **_interleaved_kwargs(src, tmp_path)
        )
        assert result == legacy
        assert seen == []
        assert calls == []

    def test_unwritable_run_settings_do_not_stop_the_run(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((2, 2, 16, 16), dtype=np.float32)
        src, _, seen = self._setup(
            monkeypatch,
            tmp_path,
            image,
            lambda img: np.ones(img.shape, dtype=np.uint8),
        )
        real_replace = os.replace

        def flaky_replace(source, dest, **kwargs):
            if str(dest).endswith("run_settings.json"):
                raise OSError("read-only cache")
            return real_replace(source, dest, **kwargs)

        monkeypatch.setattr(os, "replace", flaky_replace)
        out = cellpose_mod.cellpose_segmentation(
            image, **_interleaved_kwargs(src, tmp_path)
        )
        assert len(seen) == 2
        assert tifffile.imread(out).shape == (2, 2, 16, 16)
        assert not glob.glob(
            str(
                tmp_path
                / "tmp"
                / "cellpose_timepoint_cache"
                / "src_interleaved_*"
                / "run_settings.json"
            )
        )

    def test_slow_pruned_slabs_grow_the_block_size(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((3, 2, 256, 256), dtype=np.uint8)

        def corner_mask(img):
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[:, :128, :128] = 1
            return mask

        src, _, seen = self._setup(
            monkeypatch, tmp_path, image, corner_mask
        )
        monkeypatch.setattr(cellpose_mod, "time", _FakeClock())
        cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src, tmp_path, distributed_blocksize_yx=128
            ),
        )
        assert len(seen) == 3
        checkpoint = zarr.open_array(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr"),
            mode="r",
        )
        summary = json.loads(checkpoint.attrs["slab_perf_summary"])
        assert summary["distributed_count"] == 3
        assert summary["final_distributed_blocksize"] == 160

    def test_slow_crowded_slabs_do_not_grow_past_the_floor(
        self, monkeypatch, tmp_path
    ):
        image = np.zeros((3, 1, 1632, 1632), dtype=np.uint8)

        def mostly_full_mask(img):
            mask = np.ones(img.shape, dtype=np.uint8)
            mask[:, 1088:, 1088:] = 0
            return mask

        src, _, seen = self._setup(
            monkeypatch, tmp_path, image, mostly_full_mask
        )
        monkeypatch.setattr(cellpose_mod, "time", _FakeClock())
        cellpose_mod.cellpose_segmentation(
            image,
            **_interleaved_kwargs(
                src, tmp_path, distributed_blocksize_yx=544
            ),
        )
        assert len(seen) == 3
        checkpoint = zarr.open_array(
            str(tmp_path / "tmp" / "src_cellpose_interleaved_chall.zarr"),
            mode="r",
        )
        records = json.loads(checkpoint.attrs["slab_perf_records"])
        assert [r["total_blocks"] for r in records] == [9, 9, 9]
        assert [r["active_blocks"] for r in records] == [8, 8, 8]
        summary = json.loads(checkpoint.attrs["slab_perf_summary"])
        assert summary["final_distributed_blocksize"] == 544

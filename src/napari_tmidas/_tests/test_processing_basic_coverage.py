# src/napari_tmidas/_tests/test_processing_basic_coverage.py
"""Breadth coverage for napari_tmidas.processing_functions.basic.

Each class pins one area of the module: the zarr metadata helpers, the
label-array alignment helpers, the registered processing functions and
their guard branches, and the TZYX post-processing helpers that the
module monkey-patches onto ProcessingWorker.

Two pieces of process-wide state live on function objects in ``basic``
(``split_channels._timepoint_sorted_output`` and
``split_tzyx_stack._thread_local``).  The module-level autouse fixture
below restores both after every test so nothing leaks into the rest of
the pytest session.
"""
import json
import os

import numpy as np
import pytest
import tifffile

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import basic

# Attributes split_tzyx_stack writes onto its thread-local scratch space.
_THREAD_LOCAL_ATTRS = (
    "dask_image",
    "output_name_format",
    "preserve_scale",
    "use_compression",
    "num_workers",
    "requires_post_processing",
    "produces_multiple_files",
    "skip_original_output",
)


@pytest.fixture(autouse=True)
def _restore_basic_module_state():
    """Undo the module-level state the functions under test write."""
    had_sorted_flag = hasattr(basic.split_channels, "_timepoint_sorted_output")
    previous_flag = getattr(
        basic.split_channels, "_timepoint_sorted_output", None
    )

    yield

    if had_sorted_flag:
        basic.split_channels._timepoint_sorted_output = previous_flag
    elif hasattr(basic.split_channels, "_timepoint_sorted_output"):
        del basic.split_channels._timepoint_sorted_output

    thread_local = getattr(basic.split_tzyx_stack, "_thread_local", None)
    if thread_local is not None:
        for name in _THREAD_LOCAL_ATTRS:
            if hasattr(thread_local, name):
                delattr(thread_local, name)


def _write_tif(path, array, **kwargs):
    """Write ``array`` to ``path`` and return the path as a string."""
    tifffile.imwrite(str(path), array, **kwargs)
    return str(path)


def _make_ome_zarr(path, shape, axes, dtype="uint16", dataset="0"):
    """Create a minimal OME-Zarr group and return its path as a string."""
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array(dataset, shape=shape, dtype=dtype)
    arr[...] = 1
    root.attrs["multiscales"] = [
        {"axes": axes, "datasets": [{"path": dataset}]}
    ]
    return str(path)


def _slice_counts(array):
    """Non-zero pixel count of every slice along the leading axis."""
    return [int(np.count_nonzero(plane)) for plane in array]


class _FakeRoot:
    """Mapping-ish stand-in for a zarr group with a controllable key set."""

    def __init__(self, keys=(), arrays=None):
        self._keys = set(keys)
        self._arrays = arrays

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return f"array:{key}"

    def arrays(self):
        if self._arrays is None:
            raise AttributeError("no arrays")
        return list(self._arrays)


class TestReadZarrRootAttrs:
    """Pins how root attrs are merged from a group, .zattrs and zarr.json."""

    def test_reads_v2_zattrs_from_disk(self, tmp_path):
        (tmp_path / ".zattrs").write_text(json.dumps({"a": 1}))

        assert basic._read_zarr_root_attrs(str(tmp_path)) == {"a": 1}

    def test_reads_v3_zarr_json_attributes(self, tmp_path):
        (tmp_path / "zarr.json").write_text(
            json.dumps({"attributes": {"b": 2}, "node_type": "group"})
        )

        assert basic._read_zarr_root_attrs(str(tmp_path)) == {"b": 2}

    def test_zarr_json_wins_over_zattrs_for_shared_keys(self, tmp_path):
        (tmp_path / ".zattrs").write_text(json.dumps({"k": "v2", "only2": 1}))
        (tmp_path / "zarr.json").write_text(
            json.dumps({"attributes": {"k": "v3"}})
        )

        attrs = basic._read_zarr_root_attrs(str(tmp_path))

        assert attrs == {"k": "v3", "only2": 1}

    def test_non_dict_zattrs_payload_is_ignored(self, tmp_path):
        (tmp_path / ".zattrs").write_text(json.dumps([1, 2, 3]))

        assert basic._read_zarr_root_attrs(str(tmp_path)) == {}

    def test_corrupt_json_is_swallowed(self, tmp_path):
        (tmp_path / ".zattrs").write_text("{not json")
        (tmp_path / "zarr.json").write_text("{also not json")

        assert basic._read_zarr_root_attrs(str(tmp_path)) == {}

    def test_zarr_json_without_attributes_key_yields_nothing(self, tmp_path):
        (tmp_path / "zarr.json").write_text(json.dumps({"attributes": 7}))

        assert basic._read_zarr_root_attrs(str(tmp_path)) == {}

    def test_root_attrs_are_merged_first(self, tmp_path):
        class _Root:
            attrs = {"from_root": True}

        (tmp_path / ".zattrs").write_text(json.dumps({"from_disk": True}))

        attrs = basic._read_zarr_root_attrs(str(tmp_path), root=_Root())

        assert attrs == {"from_root": True, "from_disk": True}

    def test_unusable_root_attrs_do_not_abort_the_read(self, tmp_path):
        class _BadAttrs:
            def keys(self):
                raise TypeError("nope")

        class _Root:
            attrs = _BadAttrs()

        (tmp_path / ".zattrs").write_text(json.dumps({"from_disk": True}))

        attrs = basic._read_zarr_root_attrs(str(tmp_path), root=_Root())

        assert attrs == {"from_disk": True}


class TestGetOmeMultiscales:
    """Pins multiscales lookup at the root and under the 'ome' namespace."""

    def test_top_level_multiscales_win(self):
        attrs = {"multiscales": [{"axes": []}], "ome": {"multiscales": [{}]}}

        assert basic._get_ome_multiscales(attrs) == [{"axes": []}]

    def test_nested_ome_multiscales_are_used_as_fallback(self):
        attrs = {"ome": {"multiscales": [{"axes": ["t"]}]}}

        assert basic._get_ome_multiscales(attrs) == [{"axes": ["t"]}]

    def test_empty_attrs_give_empty_list(self):
        assert basic._get_ome_multiscales({}) == []

    def test_non_dict_ome_entry_is_ignored(self):
        assert basic._get_ome_multiscales({"ome": "not-a-dict"}) == []

    def test_empty_nested_multiscales_give_empty_list(self):
        assert basic._get_ome_multiscales({"ome": {"multiscales": []}}) == []


class TestResolveDatasetPath:
    """Pins the 0 / s0 / data candidate order and the arrays() fallback."""

    def test_preferred_path_is_tried_first(self):
        root = _FakeRoot(keys={"0", "custom"})

        assert basic._resolve_dataset_path(root, "custom") == "custom"

    def test_falls_back_to_zero(self):
        root = _FakeRoot(keys={"0"})

        assert basic._resolve_dataset_path(root, "missing") == "0"

    def test_supports_s0_naming(self):
        root = _FakeRoot(keys={"s0"})

        assert basic._resolve_dataset_path(root) == "s0"

    def test_supports_data_naming(self):
        root = _FakeRoot(keys={"data"})

        assert basic._resolve_dataset_path(root) == "data"

    def test_uses_first_array_name_when_no_candidate_matches(self):
        root = _FakeRoot(keys=set(), arrays=[("lvl0", object())])

        assert basic._resolve_dataset_path(root) == "lvl0"

    def test_returns_none_when_nothing_resolves(self):
        root = _FakeRoot(keys=set(), arrays=[])

        assert basic._resolve_dataset_path(root) is None


class TestGetTimepointCount:
    """Pins timepoint discovery for zarr stores and TIFF files."""

    def test_ome_zarr_with_time_axis_reports_t_size(self, tmp_path):
        path = _make_ome_zarr(
            tmp_path / "movie.zarr",
            shape=(7, 2, 4, 4),
            axes=[
                {"name": "t", "type": "time"},
                {"name": "z"},
                {"name": "y"},
                {"name": "x"},
            ],
        )

        assert basic.get_timepoint_count(path) == 7

    def test_string_axes_named_time_are_recognised(self, tmp_path):
        path = _make_ome_zarr(
            tmp_path / "movie.zarr",
            shape=(3, 4, 4),
            axes=["time", "y", "x"],
        )

        assert basic.get_timepoint_count(path) == 3

    def test_ome_zarr_without_time_axis_reports_one(self, tmp_path):
        path = _make_ome_zarr(
            tmp_path / "still.zarr",
            shape=(2, 4, 4),
            axes=[{"name": "z"}, {"name": "y"}, {"name": "x"}],
        )

        assert basic.get_timepoint_count(path) == 1

    def test_s0_dataset_naming_is_resolved(self, tmp_path):
        path = _make_ome_zarr(
            tmp_path / "s0named.zarr",
            shape=(5, 4, 4),
            axes=[{"name": "t", "type": "time"}, {"name": "y"}, {"name": "x"}],
            dataset="s0",
        )

        assert basic.get_timepoint_count(path) == 5

    def test_ome_metadata_without_any_dataset_reports_one(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = tmp_path / "empty.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["multiscales"] = [
            {"axes": [{"name": "t", "type": "time"}], "datasets": []}
        ]

        assert basic.get_timepoint_count(str(path)) == 1

    def test_plain_zarr_group_uses_first_array_leading_axis(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = tmp_path / "plain.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.create_array("frames", shape=(6, 4, 4), dtype="uint8")

        assert basic.get_timepoint_count(str(path)) == 6

    def test_plain_zarr_group_with_singleton_leading_axis(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = tmp_path / "single.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.create_array("frames", shape=(1, 4, 4), dtype="uint8")

        assert basic.get_timepoint_count(str(path)) == 1

    def test_bare_zarr_array_uses_its_leading_axis(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = tmp_path / "bare.zarr"
        zarr.open_array(str(path), mode="w", shape=(9, 4, 4), dtype="uint8")

        assert basic.get_timepoint_count(str(path)) == 9

    def test_empty_zarr_group_reports_one(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = tmp_path / "nothing.zarr"
        zarr.open_group(str(path), mode="w")

        assert basic.get_timepoint_count(str(path)) == 1

    def test_imagej_frames_metadata_is_used_when_axes_lack_time(
        self, tmp_path
    ):
        path = tmp_path / "ij.tif"
        tifffile.imwrite(
            str(path),
            np.zeros((3, 5, 5), np.uint8),
            photometric="minisblack",
            description="ImageJ=1.53t\nimages=3\nframes=3\nmode=grayscale\n",
        )

        assert basic.get_timepoint_count(str(path)) == 3

    def test_unreadable_zarr_returns_none(self, tmp_path):
        path = tmp_path / "broken.zarr"
        path.mkdir()

        assert basic.get_timepoint_count(str(path)) is None

    def test_directory_with_zattrs_takes_the_zarr_branch(self, tmp_path):
        path = tmp_path / "looks_like_zarr"
        path.mkdir()
        (path / ".zattrs").write_text(json.dumps({}))

        # zarr cannot open it, so the error path reports "unknown".
        assert basic.get_timepoint_count(str(path)) is None

    def test_tiff_with_declared_time_axis(self, tmp_path):
        data = np.zeros((4, 2, 5, 5), dtype=np.uint8)
        path = _write_tif(
            tmp_path / "t.tif",
            data,
            imagej=True,
            metadata={"axes": "TZYX"},
        )

        assert basic.get_timepoint_count(path) == 4

    def test_single_page_tiff_reports_one(self, tmp_path):
        path = _write_tif(tmp_path / "plane.tif", np.zeros((5, 5), np.uint8))

        assert basic.get_timepoint_count(path) == 1

    def test_multi_page_tiff_without_time_axis_reports_one(self, tmp_path):
        path = _write_tif(
            tmp_path / "stack.tif",
            np.zeros((3, 5, 5), np.uint8),
            photometric="minisblack",
        )

        assert basic.get_timepoint_count(path) == 1

    def test_missing_file_returns_none(self, tmp_path):
        # tifffile raises FileNotFoundError (an OSError), which the
        # function turns into the documented "unknown" answer.
        path = tmp_path / "does_not_exist.tif"

        assert basic.get_timepoint_count(str(path)) is None

    def test_without_tifffile_the_count_defaults_to_one(
        self, tmp_path, monkeypatch
    ):
        # A 4-timepoint TIFF: the answer is 1 only because the tifffile
        # branch is skipped, not because the file says so.
        path = _write_tif(
            tmp_path / "x.tif",
            np.zeros((4, 2, 5, 5), np.uint8),
            imagej=True,
            metadata={"axes": "TZYX"},
        )
        assert basic.get_timepoint_count(path) == 4

        monkeypatch.setattr(basic, "_HAS_TIFFFILE", False)

        assert basic.get_timepoint_count(path) == 1

    def test_accepts_a_path_object(self, tmp_path):
        path = tmp_path / "plane.tif"
        _write_tif(path, np.zeros((5, 5), np.uint8))

        assert basic.get_timepoint_count(path) == 1


class TestSortFilesByTimepoints:
    """Pins the T<n>/ foldering, the skip path and the copy-error path."""

    def test_files_are_grouped_into_timepoint_folders(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        single = np.arange(25, dtype=np.uint8).reshape(5, 5)
        movie = np.arange(4 * 2 * 5 * 5, dtype=np.uint8).reshape(4, 2, 5, 5)
        a = _write_tif(src / "a.tif", single)
        b = _write_tif(
            src / "b.tif",
            movie,
            imagej=True,
            metadata={"axes": "TZYX"},
        )

        mapping = basic.sort_files_by_timepoints([a, b], str(out))

        assert mapping == {
            1: [str(out / "T1" / "a.tif")],
            4: [str(out / "T4" / "b.tif")],
        }
        # The copies must be byte-faithful, not just present.
        np.testing.assert_array_equal(
            tifffile.imread(str(out / "T1" / "a.tif")), single
        )
        np.testing.assert_array_equal(
            tifffile.imread(str(out / "T4" / "b.tif")), movie
        )

    def test_undeterminable_files_are_skipped(self, tmp_path):
        missing = tmp_path / "src" / "gone.tif"

        mapping = basic.sort_files_by_timepoints([str(missing)], str(tmp_path))

        assert mapping == {}
        # Nothing may be created for a file that could not be inspected.
        assert not (tmp_path / "T1").exists()

    def test_directory_inputs_are_copied_as_trees(self, tmp_path):
        store = _make_ome_zarr(
            tmp_path / "movie.zarr",
            shape=(3, 4, 4),
            axes=[{"name": "t", "type": "time"}, {"name": "y"}, {"name": "x"}],
        )
        out = tmp_path / "out"

        mapping = basic.sort_files_by_timepoints([store], str(out))

        assert mapping == {3: [str(out / "T3" / "movie.zarr")]}
        assert (out / "T3" / "movie.zarr").is_dir()
        # The copied store must still be a readable zarr with the same data.
        assert basic.get_timepoint_count(str(out / "T3" / "movie.zarr")) == 3

    def test_existing_destination_tree_is_replaced(self, tmp_path):
        store = _make_ome_zarr(
            tmp_path / "movie.zarr",
            shape=(3, 4, 4),
            axes=[{"name": "t", "type": "time"}, {"name": "y"}, {"name": "x"}],
        )
        out = tmp_path / "out"
        stale = out / "T3" / "movie.zarr"
        stale.mkdir(parents=True)
        (stale / "stale.txt").write_text("old")

        basic.sort_files_by_timepoints([store], str(out))

        assert not (stale / "stale.txt").exists()
        assert (stale / "zarr.json").exists()

    def test_copy_errors_are_reported_and_skipped(
        self, tmp_path, monkeypatch, capsys
    ):
        src = _write_tif(tmp_path / "a.tif", np.zeros((5, 5), np.uint8))

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(basic.shutil, "copy2", _boom)

        mapping = basic.sort_files_by_timepoints([src], str(tmp_path / "o"))

        assert mapping == {}
        assert "ERROR: disk full" in capsys.readouterr().out


class TestArrayHelpers:
    """Pins the shared array coercion / alignment helpers."""

    def test_to_array_rejects_scalars(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            basic._to_array(np.float32(3.0))

    def test_to_array_passes_lists_through_as_arrays(self):
        arr = basic._to_array([[1, 2], [3, 4]])

        assert isinstance(arr, np.ndarray)
        assert arr.tolist() == [[1, 2], [3, 4]]

    def test_to_array_returns_the_same_buffer_for_an_ndarray(self):
        source = np.ones((2, 2), dtype=np.uint8)

        assert basic._to_array(source) is source

    def test_nonzero_bounds_on_empty_content_covers_full_shape(self):
        arr = np.zeros((3, 4), dtype=np.uint8)

        assert basic._nonzero_bounds(arr) == [(0, 3), (0, 4)]

    def test_nonzero_bounds_are_inclusive_of_the_last_index(self):
        arr = np.zeros((5, 5), dtype=np.uint8)
        arr[1:3, 2:5] = 1

        assert basic._nonzero_bounds(arr) == [(1, 3), (2, 5)]

    def test_match_ndim_squeezes_singleton_axes(self):
        reference = np.zeros((4, 4), dtype=np.uint8)
        candidate = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)

        result = basic._match_ndim(reference, candidate)

        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, candidate[0])

    def test_match_ndim_rejects_extra_non_singleton_axes(self):
        reference = np.zeros((4, 4), dtype=np.uint8)
        candidate = np.ones((2, 4, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unable to align"):
            basic._match_ndim(reference, candidate)

    def test_match_ndim_prepends_axes_when_candidate_is_flatter(self):
        reference = np.zeros((1, 4, 4), dtype=np.uint8)
        candidate = np.arange(16, dtype=np.uint8).reshape(4, 4)

        result = basic._match_ndim(reference, candidate)

        assert result.shape == (1, 4, 4)
        np.testing.assert_array_equal(result[0], candidate)

    def test_align_candidate_returns_identical_shapes_untouched(self):
        reference = np.ones((3, 3), dtype=np.uint8)
        candidate = np.arange(9, dtype=np.uint8).reshape(3, 3)

        result = basic._align_candidate(reference, candidate)

        np.testing.assert_array_equal(result, candidate)

    def test_align_candidate_on_empty_candidate_returns_zeros(self):
        reference = np.ones((6, 6), dtype=np.uint8)
        candidate = np.zeros((3, 3), dtype=np.uint8)

        result = basic._align_candidate(reference, candidate)

        assert result.shape == (6, 6)
        assert not result.any()

    def test_align_candidate_recentres_a_smaller_block(self):
        reference = np.zeros((8, 8), dtype=np.uint8)
        reference[3:5, 3:5] = 1
        candidate = np.zeros((4, 4), dtype=np.uint8)
        candidate[0:2, 0:2] = 7

        result = basic._align_candidate(reference, candidate)

        assert result.shape == (8, 8)
        # The candidate block lands exactly on the reference's centre and
        # nothing is left anywhere else.
        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[3:5, 3:5] = 7
        np.testing.assert_array_equal(result, expected)

    def test_align_candidate_shifts_an_off_centre_block_back(self):
        # Reference content bottom-right, candidate content top-left: the
        # candidate must be translated, not merely copied in place.
        reference = np.zeros((6, 6), dtype=np.uint8)
        reference[4:6, 4:6] = 1
        candidate = np.zeros((8, 8), dtype=np.uint8)
        candidate[0:2, 0:2] = 3

        result = basic._align_candidate(reference, candidate)

        expected = np.zeros((6, 6), dtype=np.uint8)
        expected[4:6, 4:6] = 3
        np.testing.assert_array_equal(result, expected)


class TestChunkedLabelFunctions:
    """Pins the three @chunked label transforms end to end."""

    def test_labels_to_binary_maps_every_non_zero_label_to_255(self):
        arr = np.array([[0, 3], [7, 0]], dtype=np.uint16)

        result = basic.labels_to_binary(arr)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[0, 255], [255, 0]])

    def test_labels_to_binary_rejects_scalars(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            basic.labels_to_binary(np.uint8(1))

    def test_invert_binary_labels_is_the_complement(self):
        arr = np.array([[0, 3], [7, 0]], dtype=np.uint16)

        result = basic.invert_binary_labels(arr)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[255, 0], [0, 255]])

    def test_binary_and_inverted_never_overlap(self):
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 4, size=(3, 5, 5), dtype=np.uint16)

        both = basic.labels_to_binary(arr).astype(
            np.uint16
        ) + basic.invert_binary_labels(arr).astype(np.uint16)

        np.testing.assert_array_equal(both, np.full(arr.shape, 255))

    def test_filter_label_by_id_keeps_only_the_requested_id(self):
        arr = np.array([[0, 3], [7, 3]], dtype=np.uint16)

        result = basic.filter_label_by_id(arr, label_id=3)

        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, [[0, 3], [0, 3]])

    def test_filter_label_by_id_defaults_to_label_one(self):
        arr = np.array([[1, 2], [3, 1]], dtype=np.uint8)

        np.testing.assert_array_equal(
            basic.filter_label_by_id(arr), [[1, 0], [0, 1]]
        )

    def test_filter_label_by_id_on_an_absent_id_clears_everything(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        result = basic.filter_label_by_id(arr, label_id=99)

        assert not result.any()
        assert result.dtype == np.uint8


class TestRgbToLabels:
    """Pins the RGB colour -> label-value mapping."""

    def test_primary_colours_map_to_their_default_labels(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, 0] = (0, 0, 255)  # blue -> 1
        rgb[0, 1] = (0, 255, 0)  # green -> 2
        rgb[0, 2] = (255, 0, 0)  # red -> 3
        rgb[1, 0] = (10, 10, 10)  # unmapped -> 0

        result = basic.rgb_to_labels(rgb)

        assert result.dtype == np.uint32
        np.testing.assert_array_equal(result, [[1, 2, 3], [0, 0, 0]])

    def test_label_values_are_configurable(self):
        rgb = np.zeros((1, 3, 3), dtype=np.uint8)
        rgb[0, 0] = (0, 0, 255)
        rgb[0, 1] = (0, 255, 0)
        rgb[0, 2] = (255, 0, 0)

        result = basic.rgb_to_labels(
            rgb, blue_label=5, green_label=6, red_label=7
        )

        np.testing.assert_array_equal(result, [[5, 6, 7]])

    def test_near_misses_are_not_mapped(self):
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        rgb[0, 0] = (0, 0, 254)  # not quite blue
        rgb[0, 1] = (0, 0, 255)

        np.testing.assert_array_equal(basic.rgb_to_labels(rgb), [[0, 1]])

    def test_non_rgb_input_is_rejected(self):
        with pytest.raises(ValueError, match="RGB image with 3 channels"):
            basic.rgb_to_labels(np.zeros((4, 4), np.uint8))

    def test_a_four_channel_trailing_axis_is_rejected(self):
        with pytest.raises(ValueError, match="RGB image with 3 channels"):
            basic.rgb_to_labels(np.zeros((4, 4, 4), np.uint8))


class TestMirrorLabelsGuards:
    """Pins mirror_labels' argument validation and mirroring arithmetic."""

    def test_non_integer_axis_is_a_type_error(self):
        with pytest.raises(TypeError, match="axis must be an integer"):
            basic.mirror_labels(np.zeros((3, 3), np.uint8), axis="0")

    def test_axis_below_negative_ndim_is_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            basic.mirror_labels(np.zeros((3, 3), np.uint8), axis=-3)

    def test_axis_at_ndim_is_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            basic.mirror_labels(np.zeros((3, 3), np.uint8), axis=2)

    def test_empty_label_image_is_returned_as_a_copy(self):
        arr = np.zeros((3, 4), dtype=np.uint16)

        result = basic.mirror_labels(arr)

        assert result is not arr
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.uint16

    def test_labels_are_mirrored_about_the_largest_slice(self):
        # Slice areas along axis 0 are [0, 2, 9, 1]; slice 2 is the pivot
        # and every kept label is offset by max_label (6).
        arr = np.zeros((4, 3, 3), dtype=np.uint8)
        arr[1, 0, 0] = 4
        arr[1, 0, 1] = 4
        arr[2, :, :] = 5
        arr[3, 2, 2] = 6

        result = basic.mirror_labels(arr)

        expected = np.zeros((4, 3, 3), dtype=np.uint8)
        expected[1, 2, 2] = 12  # slice 3 reflected onto slice 1 (6 + 6)
        expected[2, :, :] = 11  # pivot reflects onto itself (5 + 6)
        expected[3, 0, 0] = 10  # slice 1 reflected onto slice 3 (4 + 6)
        expected[3, 0, 1] = 10
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8
        # The input must not be touched.
        assert int(arr[2, 0, 0]) == 5

    def test_negative_axis_is_normalised(self):
        # axis=-2 on a 2D array must behave exactly like axis=0.
        arr = np.zeros((5, 4), dtype=np.uint8)
        arr[2, :] = 1  # largest slice -> pivot at index 2
        arr[1, 0] = 1

        result = basic.mirror_labels(arr, axis=-2)

        expected = np.zeros((5, 4), dtype=np.uint8)
        expected[2, :] = 2  # pivot reflected onto itself (1 + max_label)
        expected[3, 0] = 2  # slice 1 reflected across the pivot to slice 3
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(
            result, basic.mirror_labels(arr, axis=0)
        )


class TestKeepSliceRangeGuards:
    """Pins keep_slice_range_by_area's validation and trimming branches."""

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            basic.keep_slice_range_by_area(np.zeros((3, 3), np.uint8))

    def test_non_integer_axis_is_a_type_error(self):
        with pytest.raises(TypeError, match="axis must be provided"):
            basic.keep_slice_range_by_area(
                np.zeros((2, 3, 3), np.uint8), axis=1.0
            )

    def test_out_of_range_axis_is_a_value_error(self):
        with pytest.raises(ValueError, match="out of bounds"):
            basic.keep_slice_range_by_area(
                np.zeros((2, 3, 3), np.uint8), axis=5
            )

    def test_zero_length_axis_is_rejected(self):
        with pytest.raises(ValueError, match="zero length"):
            basic.keep_slice_range_by_area(np.zeros((0, 3, 3), np.uint8))

    def test_uniform_areas_return_an_untouched_copy(self):
        arr = np.ones((3, 2, 2), dtype=np.uint8)

        result = basic.keep_slice_range_by_area(arr)

        assert result is not arr
        np.testing.assert_array_equal(result, arr)

    def test_content_outside_the_range_is_zeroed(self):
        # Areas [5, 1, 16, 8, 2]: min at index 1, max at index 2, so only
        # slices 1..2 survive.  Slices 0, 3 and 4 all carry content, so a
        # function body that just copied the input would fail here.
        arr = np.zeros((5, 4, 4), dtype=np.uint8)
        arr[0, 0, :4] = 1
        arr[0, 1, 0] = 1
        arr[1, 0, 0] = 3
        arr[2, :, :] = 7
        arr[3, 0:2, :] = 2
        arr[4, 0, 0:2] = 9
        assert _slice_counts(arr) == [5, 1, 16, 8, 2]

        result = basic.keep_slice_range_by_area(arr)

        assert result.shape == arr.shape
        assert result.dtype == arr.dtype
        assert _slice_counts(result) == [0, 1, 16, 0, 0]
        assert int(result[1, 0, 0]) == 3
        np.testing.assert_array_equal(result[2], np.full((4, 4), 7))
        # The input is left untouched (the function copies).
        assert _slice_counts(arr) == [5, 1, 16, 8, 2]

    def test_trimming_follows_the_requested_axis(self):
        # Same content, transposed onto axis 2.
        arr = np.zeros((4, 4, 5), dtype=np.uint8)
        arr[0, :4, 0] = 1
        arr[1, 0, 0] = 1
        arr[0, 0, 1] = 3
        arr[:, :, 2] = 7
        arr[0:2, :, 3] = 2
        arr[0, 0:2, 4] = 9
        counts = [int(np.count_nonzero(arr[:, :, i])) for i in range(5)]
        assert counts == [5, 1, 16, 8, 2]

        result = basic.keep_slice_range_by_area(arr, axis=2)

        kept = [int(np.count_nonzero(result[:, :, i])) for i in range(5)]
        assert kept == [0, 1, 16, 0, 0]

    def test_negative_axis_matches_the_positive_one(self):
        arr = np.zeros((3, 3, 4), dtype=np.uint8)
        arr[:, :, 0] = 1
        arr[:, 0, 1] = 1
        arr[:, :, 3] = 1

        np.testing.assert_array_equal(
            basic.keep_slice_range_by_area(arr, axis=-1),
            basic.keep_slice_range_by_area(arr, axis=2),
        )


class TestIntersectLabelImagesBranches:
    """Pins intersect_label_images' guard and pairing branches."""

    def test_blank_suffix_is_rejected(self):
        with pytest.raises(ValueError, match="must be provided"):
            basic.intersect_label_images(
                np.zeros((2, 2), np.uint8), primary_suffix=""
            )

    def test_missing_worker_context_is_reported(self):
        # No caller frame carries a `filepath` local, so the paired lookup
        # cannot be resolved.
        with pytest.raises(ValueError, match="Could not determine current"):
            basic.intersect_label_images(np.zeros((2, 2), np.uint8))

    def test_npy_pairs_are_loaded_with_numpy(self, tmp_path):
        primary = np.zeros((4, 4), dtype=np.uint16)
        primary[1:3, 1:3] = 5
        secondary = np.zeros((4, 4), dtype=np.uint16)
        secondary[2:4, 1:3] = 9
        np.save(tmp_path / "pair_b.npy", secondary)
        # Read out of this frame by intersect_label_images' inspect.stack()
        # walk; it has no other reader here.
        filepath = str(tmp_path / "pair_a.npy")  # noqa: F841

        result = basic.intersect_label_images(
            primary,
            primary_suffix="_a.npy",
            secondary_suffix="_b.npy",
        )

        # Overlap is exactly row 2, cols 1:2, carrying the PRIMARY ids.
        expected = np.zeros((4, 4), dtype=np.uint16)
        expected[2, 1:3] = 5
        np.testing.assert_array_equal(result, expected)

    def test_png_pairs_fall_back_to_skimage(self, tmp_path):
        pytest.importorskip("skimage.io")
        primary = np.zeros((4, 4), dtype=np.uint8)
        primary[0:2, 0:2] = 3
        secondary = np.zeros((4, 4), dtype=np.uint8)
        secondary[1:3, 1:3] = 4
        from skimage.io import imsave

        imsave(str(tmp_path / "pair_b.png"), secondary, check_contrast=False)
        filepath = str(tmp_path / "pair_a.png")  # noqa: F841

        result = basic.intersect_label_images(
            primary,
            primary_suffix="_a.png",
            secondary_suffix="_b.png",
        )

        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1, 1] = 3  # the only voxel non-zero in both images
        np.testing.assert_array_equal(result, expected)

    def test_secondary_files_are_skipped_with_a_warning(self, tmp_path):
        filepath = str(tmp_path / "pair_b.tif")  # noqa: F841

        with pytest.warns(UserWarning, match="Skipping secondary"):
            result = basic.intersect_label_images(np.zeros((2, 2), np.uint8))

        assert result is None

    def test_unrecognised_suffix_is_rejected(self, tmp_path):
        filepath = str(tmp_path / "lonely.tif")  # noqa: F841

        with pytest.raises(ValueError, match="does not end with"):
            basic.intersect_label_images(np.zeros((2, 2), np.uint8))

    def test_a_missing_partner_file_is_reported(self, tmp_path):
        filepath = str(tmp_path / "solo_a.tif")  # noqa: F841

        with pytest.raises(
            FileNotFoundError, match=r"Paired label image not found.*solo_b"
        ):
            basic.intersect_label_images(np.zeros((2, 2), np.uint8))

    def test_disjoint_labels_give_an_all_zero_result(self, tmp_path):
        primary = np.zeros((4, 4), dtype=np.uint8)
        primary[0, 0] = 1
        secondary = np.zeros((4, 4), dtype=np.uint16)
        secondary[3, 3] = 1
        _write_tif(tmp_path / "d_b.tif", secondary)
        filepath = str(tmp_path / "d_a.tif")  # noqa: F841

        result = basic.intersect_label_images(primary)

        assert not result.any()
        assert result.shape == (4, 4)
        assert result.dtype == np.promote_types(np.uint8, np.uint16)

    def test_mismatched_partner_shapes_are_recentred_before_intersecting(
        self, tmp_path
    ):
        primary = np.zeros((6, 6), dtype=np.uint16)
        primary[2:4, 2:4] = 8
        secondary = np.zeros((2, 2), dtype=np.uint16)
        secondary[:, :] = 9
        _write_tif(tmp_path / "r_b.tif", secondary)
        filepath = str(tmp_path / "r_a.tif")  # noqa: F841

        result = basic.intersect_label_images(primary)

        # The 2x2 partner is centred onto the primary's block, so the
        # whole block survives carrying the primary's id.
        expected = np.zeros((6, 6), dtype=np.uint16)
        expected[2:4, 2:4] = 8
        np.testing.assert_array_equal(result, expected)


class TestGammaCorrection:
    """Pins gamma correction for integer and float inputs."""

    def test_gamma_one_is_a_no_op_for_uint8(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(6, 6), dtype=np.uint8)

        result = basic.gamma_correction(image, gamma=1.0)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, image)

    def test_gamma_below_one_brightens_midtones(self):
        image = np.full((4, 4), 64, dtype=np.uint8)

        result = basic.gamma_correction(image, gamma=0.5)

        assert result.dtype == np.uint8
        # sqrt(64/255) * 255 == 127.7 -> 127 after the uint8 store.
        np.testing.assert_array_equal(result, np.full((4, 4), 127, np.uint8))

    def test_float_images_use_a_unit_maximum(self):
        image = np.full((3, 4, 4), 0.25, dtype=np.float32)

        result = basic.gamma_correction(image, gamma=2.0)

        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 0.0625, atol=1e-6)

    def test_leading_axes_are_iterated_plane_by_plane(self):
        image = np.zeros((2, 3, 4, 4), dtype=np.uint8)
        image[1, 2] = 255
        image[0, 1] = 128

        result = basic.gamma_correction(image, gamma=2.0)

        assert result.shape == image.shape
        # Every plane gets the same curve: 255 -> 255, 128 -> 64, 0 -> 0.
        np.testing.assert_array_equal(result[1, 2], np.full((4, 4), 255))
        np.testing.assert_array_equal(result[0, 1], np.full((4, 4), 64))
        assert not result[0, 0].any()
        assert not result[1, 0].any()

    def test_uint16_uses_its_own_dtype_maximum(self):
        image = np.full((2, 2), 32768, dtype=np.uint16)

        result = basic.gamma_correction(image, gamma=2.0)

        assert result.dtype == np.uint16
        # (32768/65535)^2 * 65535 == 16384.5 -> 16384 stored.
        assert int(result[0, 0]) == 16384


class TestMaxZProjections:
    """Pins both Z-projection entry points, including their guards."""

    def test_max_z_projection_drops_the_leading_axis(self):
        image = np.zeros((3, 4, 5), dtype=np.uint16)
        image[1, 2, 3] = 42
        image[0, 0, 0] = 7

        result = basic.max_z_projection(image)

        assert result.shape == (4, 5)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, image.max(axis=0))
        assert int(result[2, 3]) == 42

    def test_tzyx_projection_requires_four_dimensions(self):
        with pytest.raises(ValueError, match="Expected 4D image"):
            basic.max_z_projection_tzyx(np.zeros((3, 4, 5), np.uint8))

    def test_tzyx_projection_reduces_the_z_axis(self):
        rng = np.random.default_rng(11)
        image = rng.integers(0, 500, size=(2, 3, 4, 4), dtype=np.uint16)

        result = basic.max_z_projection_tzyx(image)

        assert result.shape == (2, 4, 4)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, image.max(axis=1))

    def test_boolean_input_uses_the_generic_max_fallback(self):
        image = np.zeros((2, 3, 4, 4), dtype=bool)
        image[1, 2, 3, 3] = True

        result = basic.max_z_projection_tzyx(image)

        assert result.dtype == bool
        np.testing.assert_array_equal(result, image.max(axis=1))
        assert result[1, 3, 3]
        assert not result[0].any()

    def test_dask_input_is_reduced_lazily(self):
        da = pytest.importorskip("dask.array")
        dense = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
        image = da.from_array(dense, chunks=(1, 1, 4, 4))

        result = basic.max_z_projection_tzyx(image)

        # Still lazy...
        assert hasattr(result, "compute")
        assert result.shape == (2, 4, 4)
        # ...and correct once realised.
        np.testing.assert_array_equal(result.compute(), dense.max(axis=1))


class TestSplitChannels:
    """Pins channel-axis discovery, dimension inference and output order."""

    def test_fewer_than_three_dimensions_is_rejected(self):
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            basic.split_channels(np.zeros((4, 4), np.uint8))

    def test_leading_channel_axis_is_moved_to_the_front(self, capsys):
        image = np.arange(3 * 6 * 7, dtype=np.uint8).reshape(3, 6, 7)

        result = basic.split_channels(image, num_channels=3)

        assert result.shape == (3, 6, 7)
        np.testing.assert_array_equal(result, image)
        assert "Channel axis identified: 0" in capsys.readouterr().out

    def test_trailing_rgb_axis_is_detected(self, capsys):
        image = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)

        result = basic.split_channels(image, num_channels=3)

        assert result.shape == (3, 6, 7)
        np.testing.assert_array_equal(result, np.moveaxis(image, 2, 0))
        assert "Channel axis identified: 2" in capsys.readouterr().out

    def test_channel_count_mismatch_falls_back_to_the_data(self, capsys):
        image = np.arange(2 * 6 * 7, dtype=np.uint8).reshape(2, 6, 7)

        result = basic.split_channels(image, num_channels=3)

        assert result.shape == (2, 6, 7)
        np.testing.assert_array_equal(result, image)
        assert (
            "Warning: Specified 3 channels but found 2 in the data. Using 2."
            in capsys.readouterr().out
        )

    def test_time_axis_is_never_mistaken_for_the_channel_axis(self, capsys):
        # No dimension equals num_channels, so the heuristic runs; axis 0 is
        # skipped because time_steps marks it as the time axis, leaving
        # axis 1 (size 5) as the channel axis.
        image = np.zeros((20, 5, 12, 13), dtype=np.uint8)
        image[7, 3] = 11

        result = basic.split_channels(image, num_channels=3, time_steps=20)

        assert result.shape == (5, 20, 12, 13)
        np.testing.assert_array_equal(result, np.moveaxis(image, 1, 0))
        assert int(result[3, 7].max()) == 11
        out = capsys.readouterr().out
        assert "Channel axis identified: 1" in out
        assert "Inferred dimension order: TCYX" in out

    def test_trailing_axis_is_the_last_resort_channel_axis(self, capsys):
        image = np.zeros((20, 30, 8), dtype=np.uint8)
        image[..., 7] = 2

        result = basic.split_channels(image, num_channels=3)

        assert result.shape == (8, 20, 30)
        np.testing.assert_array_equal(result, np.moveaxis(image, 2, 0))
        assert int(result[7].max()) == 2
        assert "Channel axis identified: 2" in capsys.readouterr().out

    def test_unidentifiable_channel_axis_raises(self):
        image = np.zeros((20, 30, 40), dtype=np.uint8)

        with pytest.raises(ValueError, match="Could not identify a channel"):
            basic.split_channels(image, num_channels=3)

    def test_five_dimensional_input_auto_detects_time(self, capsys):
        image = np.zeros((20, 3, 2, 5, 5), dtype=np.uint8)
        image[4, 2] = 6

        result = basic.split_channels(image, num_channels=3)

        assert result.shape == (3, 20, 2, 5, 5)
        np.testing.assert_array_equal(result, np.moveaxis(image, 1, 0))
        assert int(result[2, 4].max()) == 6
        out = capsys.readouterr().out
        assert "Auto-detected time dimension: T=20" in out
        assert "Inferred dimension order: TCZYX" in out

    def test_explicit_time_steps_mark_the_series_as_a_timelapse(self, capsys):
        image = np.zeros((4, 3, 2, 5, 5), dtype=np.uint8)
        image[1, 2] = 8

        result = basic.split_channels(image, num_channels=3, time_steps=4)

        assert result.shape == (3, 4, 2, 5, 5)
        np.testing.assert_array_equal(result, np.moveaxis(image, 1, 0))
        assert int(result[2, 1].max()) == 8
        out = capsys.readouterr().out
        assert "Inferred dimension order: TCZYX" in out
        # time_steps was given, so the auto-detection must stay quiet.
        assert "Auto-detected time dimension" not in out

    def test_fiji_output_format_keeps_tzyx_order(self, capsys):
        image = np.zeros((4, 3, 2, 5, 5), dtype=np.uint8)
        image[1, 2] = 8

        result = basic.split_channels(
            image, num_channels=3, time_steps=4, output_format="fiji"
        )

        # The non-channel axes are already TZYX, so fiji mode is a no-op
        # and must produce exactly the same array as python mode.
        assert result.shape == (3, 4, 2, 5, 5)
        np.testing.assert_array_equal(result, np.moveaxis(image, 1, 0))
        out = capsys.readouterr().out
        assert "Inferred dimension order: TCZYX" in out
        assert "Transposing channels" not in out

    def test_anonymous_trailing_axes_are_labelled(self, capsys):
        image = np.zeros((3, 4, 5, 6, 7), dtype=np.uint8)
        image[2] = 4

        result = basic.split_channels(
            image, num_channels=3, output_format="fiji"
        )

        assert result.shape == (3, 4, 5, 6, 7)
        np.testing.assert_array_equal(result, image)
        # The fifth axis has no spatial name left, so it is labelled "A".
        assert "Inferred dimension order: CZYXA" in capsys.readouterr().out

    def test_timepoint_sorting_runs_once_per_output_folder(self, tmp_path):
        if hasattr(basic.split_channels, "_timepoint_sorted_output"):
            del basic.split_channels._timepoint_sorted_output
        src = tmp_path / "src"
        src.mkdir()
        plane = np.arange(25, dtype=np.uint8).reshape(5, 5)
        first = _write_tif(src / "a.tif", plane)
        # Both names are picked out of this frame by split_channels'
        # inspect.stack() walk; nothing else reads them.
        file_list = [first]  # noqa: F841
        output_folder = str(tmp_path / "out")  # noqa: F841

        result = basic.split_channels(
            np.zeros((3, 6, 7), np.uint8),
            num_channels=3,
            sort_by_timepoints=True,
        )

        assert result.shape == (3, 6, 7)
        sorted_copy = tmp_path / "out" / "T1" / "a.tif"
        np.testing.assert_array_equal(tifffile.imread(str(sorted_copy)), plane)
        assert basic.split_channels._timepoint_sorted_output == str(
            tmp_path / "out"
        )

    def test_sorting_is_not_repeated_for_the_same_output_folder(
        self, tmp_path
    ):
        src = tmp_path / "src"
        src.mkdir()
        first = _write_tif(src / "a.tif", np.zeros((5, 5), np.uint8))
        file_list = [first]  # noqa: F841
        output_folder = str(tmp_path / "out")  # noqa: F841
        # Pretend a previous file in the batch already sorted this folder.
        basic.split_channels._timepoint_sorted_output = str(tmp_path / "out")

        basic.split_channels(
            np.zeros((3, 6, 7), np.uint8),
            num_channels=3,
            sort_by_timepoints=True,
        )

        assert not (tmp_path / "out").exists()

    def test_timepoint_sorting_is_skipped_without_worker_context(
        self, tmp_path
    ):
        if hasattr(basic.split_channels, "_timepoint_sorted_output"):
            del basic.split_channels._timepoint_sorted_output

        result = basic.split_channels(
            np.zeros((3, 6, 7), np.uint8),
            num_channels=3,
            sort_by_timepoints=True,
        )

        assert result.shape == (3, 6, 7)
        # The flag is initialised but never advanced, and no folder is made.
        assert basic.split_channels._timepoint_sorted_output is None
        assert list(tmp_path.iterdir()) == []


class TestMergeChannelsBranches:
    """Pins merge_channels' early returns and the in-memory merge path."""

    def test_missing_channel_marker_returns_the_input(self, tmp_path, capsys):
        image = np.ones((4, 4), dtype=np.uint8)

        result = basic.merge_channels(
            image, _source_filepath=str(tmp_path / "plain.tif")
        )

        assert result is image
        assert "No channel pattern" in capsys.readouterr().out

    def test_no_source_path_at_all_is_reported(self):
        # Nothing in the call stack carries a `filepath` local here.
        with pytest.raises(
            ValueError, match="Could not determine current file path"
        ):
            basic.merge_channels(np.ones((2, 2), np.uint8))

    def test_a_lone_channel_is_returned_unmerged(self, tmp_path, capsys):
        image = np.ones((4, 4), dtype=np.uint8)
        path = _write_tif(tmp_path / "s_channel_0.tif", image)

        result = basic.merge_channels(image, _source_filepath=path)

        assert result is image
        assert "Only found 1 channel(s)" in capsys.readouterr().out

    def test_non_primary_channels_are_skipped(self, tmp_path, capsys):
        image = np.ones((4, 4), dtype=np.uint8)
        _write_tif(tmp_path / "s_channel_0.tif", image)
        second = _write_tif(tmp_path / "s_channel_1.tif", image)

        result = basic.merge_channels(image, _source_filepath=second)

        assert result is image
        assert "is not the primary channel (0)" in capsys.readouterr().out

    def test_header_shape_is_read_when_the_image_is_not_loaded(
        self, tmp_path
    ):
        # skip_load=True means the worker hands merge_channels image=None,
        # so shape and dtype have to come from the TIFF header.
        rng = np.random.default_rng(0)
        c0 = rng.integers(0, 100, size=(5, 6), dtype=np.uint16)
        c1 = rng.integers(0, 100, size=(5, 6), dtype=np.uint16)
        first = _write_tif(tmp_path / "h_channel_0.tif", c0)
        _write_tif(tmp_path / "h_channel_1.tif", c1)
        out = tmp_path / "out"

        written = basic.merge_channels(
            None,
            _source_filepath=first,
            _output_folder=str(out),
            _output_suffix="_merged",
        )

        assert written == str(out / "h_merged.tif")
        merged = tifffile.imread(written)
        assert merged.shape == (2, 5, 6)
        assert merged.dtype == np.uint16
        np.testing.assert_array_equal(merged[0], c0)
        np.testing.assert_array_equal(merged[1], c1)

    def test_unreadable_header_raises_a_value_error(self, tmp_path):
        bad = tmp_path / "b_channel_0.tif"
        bad.write_text("not a tiff")
        _write_tif(tmp_path / "b_channel_1.tif", np.zeros((3, 3), np.uint8))

        with pytest.raises(ValueError, match="cannot read header"):
            basic.merge_channels(None, _source_filepath=str(bad))

    def test_in_memory_merge_of_zyx_channels(self, tmp_path):
        rng = np.random.default_rng(1)
        c0 = rng.integers(0, 50, size=(3, 4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(3, 4, 5), dtype=np.uint16)
        first = _write_tif(
            tmp_path / "z_channel_0.tif", c0, photometric="minisblack"
        )
        _write_tif(tmp_path / "z_channel_1.tif", c1, photometric="minisblack")

        result = basic.merge_channels(c0, _source_filepath=first)

        assert result.shape == (2, 3, 4, 5)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result[0], c0)
        np.testing.assert_array_equal(result[1], c1)

    def test_in_memory_merge_of_tzyx_channels(self, tmp_path):
        rng = np.random.default_rng(2)
        c0 = rng.integers(0, 50, size=(2, 3, 4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(2, 3, 4, 5), dtype=np.uint16)
        first = _write_tif(
            tmp_path / "t_channel_0.tif", c0, photometric="minisblack"
        )
        _write_tif(tmp_path / "t_channel_1.tif", c1, photometric="minisblack")

        result = basic.merge_channels(c0, _source_filepath=first)

        # 4D input gets C inserted at axis 1 (TCZYX), not at the front.
        assert result.shape == (2, 2, 3, 4, 5)
        np.testing.assert_array_equal(result[:, 0], c0)
        np.testing.assert_array_equal(result[:, 1], c1)

    def test_compressed_siblings_are_read_through_skimage(self, tmp_path):
        pytest.importorskip("skimage.io")
        rng = np.random.default_rng(3)
        c0 = rng.integers(0, 50, size=(4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(4, 5), dtype=np.uint16)
        first = _write_tif(tmp_path / "k_channel_0.tif", c0)
        _write_tif(tmp_path / "k_channel_1.tif", c1, compression="zlib")

        result = basic.merge_channels(c0, _source_filepath=first)

        assert result.shape == (2, 4, 5)
        np.testing.assert_array_equal(result[0], c0)
        np.testing.assert_array_equal(result[1], c1)

    def test_shape_mismatch_between_channels_is_rejected(self, tmp_path):
        pytest.importorskip("skimage.io")
        c0 = np.zeros((4, 5), dtype=np.uint16)
        first = _write_tif(tmp_path / "m_channel_0.tif", c0)
        _write_tif(tmp_path / "m_channel_1.tif", np.zeros((6, 7), np.uint16))

        with pytest.raises(ValueError, match=r"different shape.*\(6, 7\)"):
            basic.merge_channels(c0, _source_filepath=first)

    def test_source_path_is_discovered_from_the_calling_frame(self, tmp_path):
        rng = np.random.default_rng(6)
        c0 = rng.integers(0, 50, size=(4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(4, 5), dtype=np.uint16)
        # No _source_filepath: merge_channels must find this local by
        # walking the stack.  Two channels exist, so a real merge proves
        # it resolved the *right* path rather than bailing out early.
        filepath = str(tmp_path / "f_channel_0.tif")  # noqa: F841
        _write_tif(tmp_path / "f_channel_0.tif", c0)
        _write_tif(tmp_path / "f_channel_1.tif", c1)

        result = basic.merge_channels(c0)

        assert result.shape == (2, 4, 5)
        np.testing.assert_array_equal(result[0], c0)
        np.testing.assert_array_equal(result[1], c1)

    def test_streaming_write_of_a_tzyx_pair(self, tmp_path):
        rng = np.random.default_rng(4)
        c0 = rng.integers(0, 50, size=(2, 3, 4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(2, 3, 4, 5), dtype=np.uint16)
        first = _write_tif(
            tmp_path / "w_channel_0.tif", c0, photometric="minisblack"
        )
        _write_tif(tmp_path / "w_channel_1.tif", c1, photometric="minisblack")
        out = tmp_path / "out"

        written = basic.merge_channels(
            c0,
            _source_filepath=first,
            _output_folder=str(out),
            _output_suffix="_merged",
        )

        assert written == str(out / "w_merged.tif")
        merged = tifffile.imread(written)
        assert merged.shape == (2, 2, 3, 4, 5)
        assert merged.dtype == np.uint16
        np.testing.assert_array_equal(merged[:, 0], c0)
        np.testing.assert_array_equal(merged[:, 1], c1)

    def test_streaming_write_of_a_zyx_pair(self, tmp_path):
        rng = np.random.default_rng(8)
        c0 = rng.integers(0, 50, size=(3, 4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(3, 4, 5), dtype=np.uint16)
        first = _write_tif(
            tmp_path / "v_channel_0.tif", c0, photometric="minisblack"
        )
        _write_tif(tmp_path / "v_channel_1.tif", c1, photometric="minisblack")
        out = tmp_path / "out"

        written = basic.merge_channels(
            c0,
            _source_filepath=first,
            _output_folder=str(out),
            _output_suffix="_merged",
        )

        assert written == str(out / "v_merged.tif")
        merged = tifffile.imread(written)
        # 3D input puts C in front (CZYX), unlike the 4D TCZYX case.
        assert merged.shape == (2, 3, 4, 5)
        np.testing.assert_array_equal(merged[0], c0)
        np.testing.assert_array_equal(merged[1], c1)

    def test_streaming_write_reads_compressed_siblings(self, tmp_path):
        # A compressed sibling cannot be memmapped, so the streaming path
        # has to fall back to a full imread for that channel.
        rng = np.random.default_rng(9)
        c0 = rng.integers(0, 50, size=(3, 4, 5), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(3, 4, 5), dtype=np.uint16)
        first = _write_tif(
            tmp_path / "c_channel_0.tif", c0, photometric="minisblack"
        )
        _write_tif(
            tmp_path / "c_channel_1.tif",
            c1,
            photometric="minisblack",
            compression="zlib",
        )
        out = tmp_path / "out"

        written = basic.merge_channels(
            c0,
            _source_filepath=first,
            _output_folder=str(out),
            _output_suffix="_merged",
        )

        merged = tifffile.imread(written)
        np.testing.assert_array_equal(merged[0], c0)
        np.testing.assert_array_equal(merged[1], c1)

    def test_a_failed_rollback_does_not_mask_the_copy_error(
        self, tmp_path, monkeypatch
    ):
        c0 = np.zeros((4, 5), dtype=np.uint16)
        first = _write_tif(tmp_path / "j_channel_0.tif", c0)
        _write_tif(tmp_path / "j_channel_1.tif", c0)
        out = tmp_path / "out"
        out.mkdir()

        real_memmap = basic.tifffile.memmap
        calls = []

        def _flaky(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                return real_memmap(*args, **kwargs)
            raise ValueError("source unreadable")

        def _unremovable(*_args, **_kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(basic.tifffile, "memmap", _flaky)
        monkeypatch.setattr(
            basic.tifffile,
            "imread",
            lambda *_a, **_k: (_ for _ in ()).throw(
                ValueError("source unreadable")
            ),
        )
        monkeypatch.setattr(basic.os, "remove", _unremovable)

        # The rollback fails too, but the caller must still see the
        # original copy error rather than the cleanup one.
        with pytest.raises(ValueError, match="source unreadable"):
            basic.merge_channels(
                c0,
                _source_filepath=first,
                _output_folder=str(out),
                _output_suffix="_merged",
            )

        assert (out / "j_merged.tif").exists()

    def test_legacy_tzyx_merge_falls_back_to_skimage(self, tmp_path):
        pytest.importorskip("skimage.io")
        rng = np.random.default_rng(5)
        c0 = rng.integers(0, 50, size=(2, 5, 6, 7), dtype=np.uint16)
        c1 = rng.integers(0, 50, size=(2, 5, 6, 7), dtype=np.uint16)
        first = _write_tif(
            tmp_path / "q_channel_0.tif", c0, photometric="minisblack"
        )
        _write_tif(
            tmp_path / "q_channel_1.tif",
            c1,
            photometric="minisblack",
            compression="zlib",
        )

        result = basic.merge_channels(c0, _source_filepath=first)

        assert result.shape == (2, 2, 5, 6, 7)
        np.testing.assert_array_equal(result[:, 0], c0)
        np.testing.assert_array_equal(result[:, 1], c1)

    def test_streaming_write_cleans_up_on_a_memmap_failure(
        self, tmp_path, monkeypatch
    ):
        c0 = np.zeros((4, 5), dtype=np.uint16)
        first = _write_tif(tmp_path / "e_channel_0.tif", c0)
        _write_tif(tmp_path / "e_channel_1.tif", c0)
        out = tmp_path / "out"
        out.mkdir()
        # A stale output from a previous run must be removed on failure.
        (out / "e_merged.tif").write_text("stale")

        def _boom(*_args, **_kwargs):
            raise RuntimeError("no memmap for you")

        monkeypatch.setattr(basic.tifffile, "memmap", _boom)

        with pytest.raises(RuntimeError, match="no memmap"):
            basic.merge_channels(
                c0,
                _source_filepath=first,
                _output_folder=str(out),
                _output_suffix="_merged",
            )

        assert not (out / "e_merged.tif").exists()

    def test_an_undeletable_output_path_still_surfaces_the_error(
        self, tmp_path
    ):
        # A directory sitting on the output path makes both the memmap
        # creation and the follow-up cleanup fail; the original error
        # must still reach the caller.
        c0 = np.zeros((4, 5), dtype=np.uint16)
        first = _write_tif(tmp_path / "u_channel_0.tif", c0)
        _write_tif(tmp_path / "u_channel_1.tif", c0)
        out = tmp_path / "out"
        (out / "u_merged.tif").mkdir(parents=True)

        with pytest.raises(IsADirectoryError, match="u_merged.tif"):
            basic.merge_channels(
                c0,
                _source_filepath=first,
                _output_folder=str(out),
                _output_suffix="_merged",
            )

        assert (out / "u_merged.tif").is_dir()

    def test_streaming_write_cleans_up_when_a_copy_fails(
        self, tmp_path, monkeypatch
    ):
        c0 = np.zeros((4, 5), dtype=np.uint16)
        first = _write_tif(tmp_path / "g_channel_0.tif", c0)
        _write_tif(tmp_path / "g_channel_1.tif", c0)
        out = tmp_path / "out"
        out.mkdir()

        real_memmap = basic.tifffile.memmap
        calls = []

        def _flaky(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                # The output memmap succeeds; every source read fails.
                return real_memmap(*args, **kwargs)
            raise ValueError("source unreadable")

        def _no_read(*_args, **_kwargs):
            raise ValueError("source unreadable")

        monkeypatch.setattr(basic.tifffile, "memmap", _flaky)
        monkeypatch.setattr(basic.tifffile, "imread", _no_read)

        # The copy failure must reach the caller unchanged.  It used to
        # surface as UnboundLocalError, because the except branch deleted
        # out_mm and the finally branch then deleted it a second time.
        with pytest.raises(ValueError, match="source unreadable"):
            basic.merge_channels(
                c0,
                _source_filepath=first,
                _output_folder=str(out),
                _output_suffix="_merged",
            )

        # The output memmap was created, then rolled back.
        assert len(calls) >= 2
        assert not (out / "g_merged.tif").exists()


class TestMergeChannelsPreFilter:
    """Pins the pre-load filter that keeps only primary channel files."""

    def test_files_without_a_channel_marker_are_rejected(self, tmp_path):
        path = str(tmp_path / "plain.tif")

        assert basic._merge_channels_file_pre_filter(path, {}) is False

    def test_primary_channel_is_accepted(self, tmp_path):
        _write_tif(tmp_path / "p_channel_1.tif", np.zeros((2, 2), np.uint8))
        _write_tif(tmp_path / "p_channel_2.tif", np.zeros((2, 2), np.uint8))

        accepted = basic._merge_channels_file_pre_filter(
            str(tmp_path / "p_channel_1.tif"), {}
        )
        rejected = basic._merge_channels_file_pre_filter(
            str(tmp_path / "p_channel_2.tif"), {}
        )

        assert accepted is True
        assert rejected is False

    def test_unlistable_folder_lets_the_file_through(self, tmp_path):
        missing = str(tmp_path / "nope" / "x_channel_0.tif")

        assert basic._merge_channels_file_pre_filter(missing, {}) is True

    def test_an_empty_folder_lets_the_file_through(self, tmp_path):
        # The folder exists but holds no sibling channels, so the primary
        # cannot be determined and the file must not be dropped.
        orphan = str(tmp_path / "y_channel_4.tif")

        assert basic._merge_channels_file_pre_filter(orphan, {}) is True

    def test_unrelated_siblings_do_not_join_the_group(self, tmp_path):
        _write_tif(tmp_path / "a_channel_2.tif", np.zeros((2, 2), np.uint8))
        _write_tif(tmp_path / "notes.txt.tif", np.zeros((2, 2), np.uint8))
        (tmp_path / "other_channel_1.txt").write_text("ignored")

        # a_channel_2 is alone in its (prefix, extension) group, so despite
        # the lower-numbered .txt sibling it counts as the primary.
        assert (
            basic._merge_channels_file_pre_filter(
                str(tmp_path / "a_channel_2.tif"), {}
            )
            is True
        )

    def test_a_different_prefix_does_not_join_the_group(self, tmp_path):
        _write_tif(tmp_path / "one_channel_0.tif", np.zeros((2, 2), np.uint8))
        _write_tif(tmp_path / "two_channel_1.tif", np.zeros((2, 2), np.uint8))

        # Different prefixes are different groups, so each file is its own
        # primary and both must be accepted.
        assert (
            basic._merge_channels_file_pre_filter(
                str(tmp_path / "two_channel_1.tif"), {}
            )
            is True
        )

    def test_custom_channel_substring_is_honoured(self, tmp_path):
        _write_tif(tmp_path / "s_ch3.tif", np.zeros((2, 2), np.uint8))
        _write_tif(tmp_path / "s_ch4.tif", np.zeros((2, 2), np.uint8))
        params = {"channel_substring": "_ch"}

        assert (
            basic._merge_channels_file_pre_filter(
                str(tmp_path / "s_ch3.tif"), params
            )
            is True
        )
        assert (
            basic._merge_channels_file_pre_filter(
                str(tmp_path / "s_ch4.tif"), params
            )
            is False
        )
        # ...and the default substring finds nothing in those names.
        assert (
            basic._merge_channels_file_pre_filter(
                str(tmp_path / "s_ch3.tif"), {}
            )
            is False
        )


class TestSplitTzyxStackAxisDiscovery:
    """Pins time-axis discovery, normalisation and the early returns."""

    def test_three_dimensional_input_is_returned_unchanged(self, capsys):
        image = np.zeros((3, 4, 5), dtype=np.uint8)

        result = basic.split_tzyx_stack(image)

        assert result is image
        assert "Expected at least 4D" in capsys.readouterr().out
        # No post-processing state may be published for a rejected input.
        thread_local = getattr(basic.split_tzyx_stack, "_thread_local", None)
        assert thread_local is None or not hasattr(thread_local, "dask_image")

    def test_plain_four_dimensional_input_uses_axis_zero(self):
        image = np.arange(6 * 2 * 4 * 4, dtype=np.uint16).reshape(6, 2, 4, 4)

        result = basic.split_tzyx_stack(image)

        assert result is image
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (6, 2, 4, 4)
        np.testing.assert_array_equal(lazy.compute(), image)
        assert basic.split_tzyx_stack._thread_local.requires_post_processing

    def test_zarr_metadata_moves_a_non_leading_time_axis(self, tmp_path):
        store = _make_ome_zarr(
            tmp_path / "cz.zarr",
            shape=(2, 5, 4, 4),
            axes=[
                {"name": "c"},
                {"name": "t", "type": "time"},
                {"name": "y"},
                {"name": "x"},
            ],
        )
        image = np.arange(2 * 5 * 4 * 4, dtype=np.uint16).reshape(2, 5, 4, 4)

        result = basic.split_tzyx_stack(image, _source_filepath=store)

        assert result is image
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (5, 2, 4, 4)
        np.testing.assert_array_equal(
            lazy.compute(), np.moveaxis(image, 1, 0)
        )

    def test_a_singleton_channel_axis_is_squeezed_away(self, tmp_path):
        image = np.arange(3 * 1 * 2 * 4 * 4, dtype=np.uint16).reshape(
            3, 1, 2, 4, 4
        )

        result = basic.split_tzyx_stack(image)

        assert result is image
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (3, 2, 4, 4)
        np.testing.assert_array_equal(lazy.compute(), image[:, 0])

    def test_a_singleton_channel_axis_is_squeezed_away_for_dask(self):
        da = pytest.importorskip("dask.array")
        dense = np.arange(3 * 1 * 2 * 4 * 4, dtype=np.uint16).reshape(
            3, 1, 2, 4, 4
        )
        image = da.from_array(dense, chunks=(1, 1, 1, 4, 4))

        result = basic.split_tzyx_stack(image)

        assert result is image
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (3, 2, 4, 4)
        np.testing.assert_array_equal(lazy.compute(), dense[:, 0])

    def test_time_axis_beyond_the_data_aborts(self, tmp_path, capsys):
        store = _make_ome_zarr(
            tmp_path / "bad.zarr",
            shape=(2, 2, 2, 2),
            axes=[
                {"name": "c"},
                {"name": "z"},
                {"name": "y"},
                {"name": "x"},
                {"name": "t", "type": "time"},
            ],
        )
        image = np.zeros((2, 2, 2, 2), dtype=np.uint8)

        result = basic.split_tzyx_stack(image, _source_filepath=store)

        assert result is image
        assert "Could not identify time axis" in capsys.readouterr().out
        thread_local = getattr(basic.split_tzyx_stack, "_thread_local", None)
        assert thread_local is None or not hasattr(thread_local, "dask_image")

    def test_unreadable_zarr_metadata_falls_back_to_axis_zero(
        self, tmp_path, capsys
    ):
        store = tmp_path / "empty.zarr"
        store.mkdir()
        image = np.arange(4 * 2 * 3 * 3, dtype=np.uint16).reshape(4, 2, 3, 3)

        result = basic.split_tzyx_stack(image, _source_filepath=str(store))

        assert result is image
        assert "Could not parse zarr axes" in capsys.readouterr().out
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (4, 2, 3, 3)
        np.testing.assert_array_equal(lazy.compute(), image)

    def test_dask_input_with_a_moved_time_axis(self, tmp_path):
        da = pytest.importorskip("dask.array")
        store = _make_ome_zarr(
            tmp_path / "dz.zarr",
            shape=(2, 5, 4, 4),
            axes=[
                {"name": "c"},
                {"name": "t", "type": "time"},
                {"name": "y"},
                {"name": "x"},
            ],
        )
        dense = np.arange(2 * 5 * 4 * 4, dtype=np.uint16).reshape(2, 5, 4, 4)
        image = da.from_array(dense, chunks=(1, 1, 4, 4))

        result = basic.split_tzyx_stack(image, _source_filepath=store)

        assert result is image
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (5, 2, 4, 4)
        # A lazy input must stay lazy, and rechunk to one timepoint each.
        assert lazy.chunks[0] == (1,) * 5
        np.testing.assert_array_equal(
            lazy.compute(), np.moveaxis(dense, 1, 0)
        )

    def test_string_axis_names_in_zarr_metadata_are_understood(
        self, tmp_path
    ):
        store = _make_ome_zarr(
            tmp_path / "sz.zarr",
            shape=(2, 5, 4, 4),
            axes=["c", "time", "y", "x"],
        )
        image = np.arange(2 * 5 * 4 * 4, dtype=np.uint16).reshape(2, 5, 4, 4)

        result = basic.split_tzyx_stack(image, _source_filepath=store)

        assert result is image
        lazy = basic.split_tzyx_stack._thread_local.dask_image
        assert lazy.shape == (5, 2, 4, 4)
        np.testing.assert_array_equal(
            lazy.compute(), np.moveaxis(image, 1, 0)
        )

    def test_worker_count_is_capped_at_the_timepoint_count(self):
        image = np.zeros((2, 3, 4, 4), dtype=np.uint8)

        basic.split_tzyx_stack(image, num_workers=16)

        assert basic.split_tzyx_stack._thread_local.num_workers == 2

    def test_worker_count_below_the_cap_is_kept(self):
        image = np.zeros((8, 3, 4, 4), dtype=np.uint8)

        basic.split_tzyx_stack(image, num_workers=3)

        assert basic.split_tzyx_stack._thread_local.num_workers == 3

    def test_parameters_are_published_for_post_processing(self):
        image = np.zeros((2, 3, 4, 4), dtype=np.uint8)

        basic.split_tzyx_stack(
            image,
            output_name_format="{basename}-{timepoint}",
            preserve_scale=False,
            use_compression=False,
        )

        thread_local = basic.split_tzyx_stack._thread_local
        assert thread_local.output_name_format == "{basename}-{timepoint}"
        assert thread_local.preserve_scale is False
        assert thread_local.use_compression is False
        assert thread_local.produces_multiple_files is True
        assert thread_local.skip_original_output is True


class TestSaveTimepoint:
    """Pins the per-timepoint writer used by the TZYX post-processing."""

    def test_writes_one_zyx_tif_per_timepoint(self, tmp_path):
        da = pytest.importorskip("dask.array")
        data = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
        lazy = da.from_array(data, chunks=(1, 3, 4, 4))
        out = str(tmp_path / "nested" / "t1.tif")

        written = basic.save_timepoint(1, lazy, out, None, False)

        assert written == out
        # Only the requested timepoint is written, and it is the 3D slab.
        stored = tifffile.imread(out)
        assert stored.shape == (3, 4, 4)
        np.testing.assert_array_equal(stored, data[1])

    def test_compression_is_applied_when_requested(self, tmp_path):
        da = pytest.importorskip("dask.array")
        data = np.zeros((1, 4, 32, 32), dtype=np.uint16)
        lazy = da.from_array(data, chunks=(1, 4, 32, 32))
        plain = str(tmp_path / "plain.tif")
        packed = str(tmp_path / "packed.tif")

        basic.save_timepoint(0, lazy, plain, None, False)
        basic.save_timepoint(0, lazy, packed, None, True)

        with tifffile.TiffFile(packed) as tif:
            assert tif.pages[0].compression != 1  # not COMPRESSION.NONE
        assert os.path.getsize(packed) < os.path.getsize(plain)
        np.testing.assert_array_equal(tifffile.imread(packed), data[0])

    def test_physical_scale_is_written_as_ome_metadata(self, tmp_path):
        da = pytest.importorskip("dask.array")
        data = np.zeros((1, 2, 3, 3), dtype=np.uint8)
        lazy = da.from_array(data, chunks=(1, 2, 3, 3))
        out = str(tmp_path / "scaled.tif")

        basic.save_timepoint(
            0, lazy, out, {"X": 0.5, "Y": 0.5, "Z": 2.0}, True
        )

        with tifffile.TiffFile(out) as tif:
            description = tif.pages[0].description

        for fragment in (
            'PhysicalSizeX="0.5"',
            'PhysicalSizeXUnit="um"',
            'PhysicalSizeY="0.5"',
            'PhysicalSizeZ="2.0"',
            'PhysicalSizeZUnit="um"',
            'DimensionOrder="XYCZT"',
        ):
            assert fragment in description

    def test_no_scale_writes_no_physical_size(self, tmp_path):
        da = pytest.importorskip("dask.array")
        lazy = da.zeros((1, 2, 3, 3), chunks=(1, 2, 3, 3), dtype="uint8")
        out = str(tmp_path / "unscaled.tif")

        basic.save_timepoint(0, lazy, out, None, False)

        with tifffile.TiffFile(out) as tif:
            assert "PhysicalSize" not in tif.pages[0].description

    def test_write_failures_are_reported_and_re_raised(self, tmp_path, capsys):
        da = pytest.importorskip("dask.array")
        lazy = da.zeros((1, 2, 3, 3), chunks=(1, 2, 3, 3), dtype="uint8")
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory")

        with pytest.raises(FileExistsError, match="blocker"):
            basic.save_timepoint(0, lazy, str(blocker / "x.tif"), None, True)

        assert "Error saving timepoint 0" in capsys.readouterr().out


class _FakeWorker:
    """Minimal stand-in for ProcessingWorker.process_file's ``self``."""

    def __init__(self, processing_func):
        self.processing_func = processing_func


class TestProcessFileWithTzyxSplitting:
    """Pins the ProcessingWorker.process_file wrapper installed here.

    ``basic.original_process_file`` is the *dependency* the wrapper calls,
    and it is resolved from the module namespace at call time.  Patching it
    substitutes the real (Qt-bound) worker step, so the wrapper's own
    post-processing logic is what actually runs in every test below.
    """

    def test_non_dict_results_pass_straight_through(self, monkeypatch):
        monkeypatch.setattr(
            basic, "original_process_file", lambda _s, _f: "not-a-dict"
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(object()), "x.tif"
        )

        assert result == "not-a-dict"

    def test_the_wrapper_forwards_its_arguments(self, monkeypatch):
        seen = []

        def _record(worker, filepath):
            seen.append((worker, filepath))
            return None

        monkeypatch.setattr(basic, "original_process_file", _record)
        worker = _FakeWorker(object())

        assert (
            basic.process_file_with_tzyx_splitting(worker, "a/b.tif") is None
        )
        assert seen == [(worker, "a/b.tif")]

    def test_results_without_a_processed_file_pass_through(self, monkeypatch):
        monkeypatch.setattr(
            basic, "original_process_file", lambda _s, _f: {"skipped": True}
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(object()), "x.tif"
        )

        assert result == {"skipped": True}

    def test_functions_without_thread_local_state_are_left_alone(
        self, monkeypatch, tmp_path
    ):
        out = str(tmp_path / "o.tif")
        _write_tif(out, np.zeros((2, 2), np.uint8))
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": out},
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(lambda x: x), "x.tif"
        )

        assert result == {"processed_file": out}
        # The consolidated output must survive untouched.
        assert os.path.exists(out)

    def test_each_timepoint_becomes_its_own_file(self, monkeypatch, tmp_path):
        source = _write_tif(
            tmp_path / "src.tif",
            np.zeros((6, 2, 4, 4), np.uint8),
            photometric="minisblack",
        )
        consolidated = str(tmp_path / "out" / "src_split.tif")
        os.makedirs(os.path.dirname(consolidated), exist_ok=True)
        _write_tif(
            consolidated,
            np.zeros((6, 2, 4, 4), np.uint8),
            photometric="minisblack",
        )
        image = np.arange(6 * 2 * 4 * 4, dtype=np.uint16).reshape(6, 2, 4, 4)
        basic.split_tzyx_stack(image, num_workers=2)
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": consolidated},
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), source
        )

        expected = [
            str(tmp_path / "out" / f"src_split_t{t:03d}.tif")
            for t in range(6)
        ]
        assert sorted(result["processed_files"]) == expected
        assert "processed_file" not in result
        assert not os.path.exists(consolidated)
        # Every timepoint must hold its own slab, not timepoint 0 six times.
        for t, path in enumerate(expected):
            np.testing.assert_array_equal(tifffile.imread(path), image[t])

    def test_scale_preservation_can_be_switched_off(
        self, monkeypatch, tmp_path
    ):
        consolidated = str(tmp_path / "out" / "n_split.tif")
        os.makedirs(os.path.dirname(consolidated), exist_ok=True)
        image = np.arange(2 * 2 * 3 * 3, dtype=np.uint16).reshape(2, 2, 3, 3)
        basic.split_tzyx_stack(image, preserve_scale=False, num_workers=1)

        # A spy that hands back *real-looking* scale data if it is ever
        # called, rather than raising.  A raising sentinel here would be
        # a no-op regression test: filepath points at a file that does not
        # exist, so even the real _extract_source_physical_scale raises on
        # it and that is swallowed by the function's own broad
        # ``except Exception`` — a raising sentinel would be silently
        # absorbed the same way whether or not the ``if preserve_scale``
        # guard actually skipped the call, and every assertion below would
        # still pass. Recording the call (and returning data that would
        # show up in the written metadata) catches the guard being removed
        # even though the source path is unreadable.
        calls = []

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return {"X": 0.5, "Y": 0.5, "Z": 0.5}

        monkeypatch.setattr(basic, "_extract_source_physical_scale", _spy)
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": consolidated},
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), str(tmp_path / "missing.tif")
        )

        expected = [
            str(tmp_path / "out" / f"n_split_t{t:03d}.tif") for t in range(2)
        ]
        assert sorted(result["processed_files"]) == expected
        # preserve_scale=False must skip the extraction call outright, not
        # merely tolerate it failing.
        assert calls == []
        for t, path in enumerate(expected):
            np.testing.assert_array_equal(tifffile.imread(path), image[t])
            with tifffile.TiffFile(path) as tif:
                assert "PhysicalSize" not in tif.pages[0].description

    def test_the_source_scale_is_carried_onto_every_timepoint(
        self, monkeypatch, tmp_path
    ):
        consolidated = str(tmp_path / "out" / "p_split.tif")
        os.makedirs(os.path.dirname(consolidated), exist_ok=True)
        image = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        basic.split_tzyx_stack(image, preserve_scale=True, num_workers=1)
        monkeypatch.setattr(
            basic,
            "_extract_source_physical_scale",
            lambda *_a, **_k: {"X": 0.25, "Y": 0.25, "Z": 1.5},
        )
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": consolidated},
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), str(tmp_path / "src.tif")
        )

        assert len(result["processed_files"]) == 2
        for path in result["processed_files"]:
            with tifffile.TiffFile(path) as tif:
                description = tif.pages[0].description
            assert 'PhysicalSizeX="0.25"' in description
            assert 'PhysicalSizeZ="1.5"' in description

    def test_unreadable_source_scale_does_not_stop_the_split(
        self, monkeypatch, tmp_path, capsys
    ):
        consolidated = str(tmp_path / "out" / "s_split.tif")
        os.makedirs(os.path.dirname(consolidated), exist_ok=True)
        image = np.arange(2 * 2 * 3 * 3, dtype=np.uint16).reshape(2, 2, 3, 3)
        basic.split_tzyx_stack(image, preserve_scale=True)
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": consolidated},
        )

        def _no_scale(*_args, **_kwargs):
            raise RuntimeError("no metadata here")

        monkeypatch.setattr(
            basic, "_extract_source_physical_scale", _no_scale
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), str(tmp_path / "src.tif")
        )

        assert len(result["processed_files"]) == 2
        # The pixels still land on disk; only the scale metadata is lost.
        for t, path in enumerate(sorted(result["processed_files"])):
            np.testing.assert_array_equal(tifffile.imread(path), image[t])
            with tifffile.TiffFile(path) as tif:
                assert "PhysicalSize" not in tif.pages[0].description
        assert "Could not read original physical scale" in (
            capsys.readouterr().out
        )

    def test_undeletable_consolidated_output_is_reported(
        self, monkeypatch, tmp_path, capsys
    ):
        # A directory at the consolidated path makes os.remove fail.
        consolidated = tmp_path / "out" / "d_split.tif"
        consolidated.mkdir(parents=True)
        image = np.arange(1 * 2 * 3 * 3, dtype=np.uint16).reshape(1, 2, 3, 3)
        basic.split_tzyx_stack(image, preserve_scale=False)
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": str(consolidated)},
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), str(tmp_path / "src.tif")
        )

        # The split still succeeds and the unremovable path is dropped
        # from the result, with a warning.
        assert result["processed_files"] == [
            str(tmp_path / "out" / "d_split_t000.tif")
        ]
        assert "processed_file" not in result
        assert consolidated.is_dir()
        assert "Could not remove consolidated file" in capsys.readouterr().out

    def test_failed_timepoints_leave_no_processed_files(
        self, monkeypatch, tmp_path, capsys
    ):
        consolidated = str(tmp_path / "out" / "f_split.tif")
        os.makedirs(os.path.dirname(consolidated), exist_ok=True)
        image = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        basic.split_tzyx_stack(image, preserve_scale=False)
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": consolidated},
        )

        def _fail(*_args, **_kwargs):
            raise OSError("write refused")

        monkeypatch.setattr(basic, "save_timepoint", _fail)

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), str(tmp_path / "src.tif")
        )

        # The consolidated output is kept as the only result.
        assert "processed_files" not in result
        assert result["processed_file"] == consolidated
        out = capsys.readouterr().out
        assert "Failed to save timepoint" in out
        assert "No ZYX files" in out

    def test_a_broken_name_format_is_caught_and_reported(
        self, monkeypatch, tmp_path, capsys
    ):
        consolidated = str(tmp_path / "out" / "b_split.tif")
        os.makedirs(os.path.dirname(consolidated), exist_ok=True)
        image = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        basic.split_tzyx_stack(image, preserve_scale=False)
        basic.split_tzyx_stack._thread_local.output_name_format = (
            "{basename}_t{timepoint:s}"
        )
        monkeypatch.setattr(
            basic,
            "original_process_file",
            lambda _s, _f: {"processed_file": consolidated},
        )

        result = basic.process_file_with_tzyx_splitting(
            _FakeWorker(basic.split_tzyx_stack), str(tmp_path / "src.tif")
        )

        # The formatting error is swallowed and the original result stands.
        assert result == {"processed_file": consolidated}
        assert not list((tmp_path / "out").glob("b_split_t*.tif"))
        assert "Error in TZYX splitting" in capsys.readouterr().out


class TestRegistryEntries:
    """Every function this module registers must resolve by name."""

    # name -> (suffix, module attribute, parameter names)
    EXPECTED = {
        "Labels to Binary": ("_binary", "labels_to_binary", []),
        "Invert Binary Labels": (
            "_inverted",
            "invert_binary_labels",
            [],
        ),
        "Filter Label by ID": (
            "_filtered",
            "filter_label_by_id",
            ["label_id"],
        ),
        "Mirror Labels": ("_mirrored", "mirror_labels", ["axis"]),
        "Intersect Label Images": (
            "_intersected",
            "intersect_label_images",
            ["primary_suffix", "secondary_suffix"],
        ),
        "Keep Slice Range by Area": (
            "_area_range",
            "keep_slice_range_by_area",
            ["axis"],
        ),
        "Gamma Correction": (
            "_gamma",
            "gamma_correction",
            ["channel", "gamma"],
        ),
        "Max Z Projection": ("_max_z", "max_z_projection", ["channel"]),
        "Max Z Projection (TZYX)": (
            "_maxZ_tzyx",
            "max_z_projection_tzyx",
            ["channel"],
        ),
        "Split Color Channels": (
            "_split",
            "split_channels",
            [
                "num_channels",
                "output_format",
                "sort_by_timepoints",
                "time_steps",
            ],
        ),
        "Merge Color Channels": (
            "_merged_colors",
            "merge_channels",
            ["channel_substring"],
        ),
        "RGB to Labels": (
            "_labels",
            "rgb_to_labels",
            ["blue_label", "green_label", "red_label"],
        ),
        "Split TZYX into ZYX TIFs": (
            "_split",
            "split_tzyx_stack",
            [
                "num_workers",
                "output_name_format",
                "preserve_scale",
                "use_compression",
            ],
        ),
    }

    @pytest.mark.parametrize(
        ("name", "expected"), sorted(EXPECTED.items())
    )
    def test_registry_entry_matches_the_module(self, name, expected):
        suffix, attribute, parameters = expected

        info = BatchProcessingRegistry.get_function_info(name)

        assert info is not None, f"{name} is not registered"
        assert info["suffix"] == suffix
        # The registered callable must be the very object the module
        # exposes, not some other function that happens to be callable.
        assert info["func"] is getattr(basic, attribute)
        assert sorted(info["parameters"]) == parameters
        assert info["description"]

    def test_every_registered_name_is_accounted_for(self):
        registered = {
            name
            for name in BatchProcessingRegistry.list_functions()
            if getattr(
                BatchProcessingRegistry.get_function_info(name)["func"],
                "__module__",
                "",
            )
            == basic.__name__
        }

        assert registered == set(self.EXPECTED)

    def test_merge_channels_declares_its_worker_contract(self):
        assert basic.merge_channels.skip_load is True
        assert (
            basic.merge_channels.file_pre_filter
            is basic._merge_channels_file_pre_filter
        )

    def test_split_tzyx_stack_is_not_thread_safe(self):
        assert basic.split_tzyx_stack.thread_safe is False

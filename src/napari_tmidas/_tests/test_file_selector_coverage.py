"""
Coverage-oriented tests for :mod:`napari_tmidas._file_selector`.

The module mixes pure helpers (channel/axes detection, path and dtype
resolution, zarr/TIFF IO) with Qt widgets and a QThread worker.  The pure
helpers are driven directly against tiny real files under ``tmp_path``;
the Qt-bound methods are exercised either against real widgets (with a
QApplication from pytest-qt's ``qapp`` fixture) or as unbound methods
against hand-built stand-ins, so that the real module lines run without a
napari viewer.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import napari_tmidas._file_selector as fs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_tif(path, data, axes=None):
    """Write a small OME-TIFF, optionally stamping explicit axes metadata."""
    kwargs = {"photometric": "minisblack"}
    if axes is not None:
        kwargs["ome"] = True
        kwargs["metadata"] = {"axes": axes}
    tifffile.imwrite(str(path), data, **kwargs)
    return str(path)


class _FakeLayerList(list):
    """List with the couple of napari LayerList methods the module uses."""

    def __init__(self):
        super().__init__()
        self.moves = []

    def move(self, src, dst):
        self.moves.append((src, dst))
        layer = self.pop(src)
        self.insert(dst, layer)

    def remove(self, item):
        if isinstance(item, str):
            for layer in list(self):
                if layer.name == item:
                    list.remove(self, layer)
                    return
            raise KeyError(item)
        list.remove(self, item)


class _FakeLayer:
    def __init__(self, data, name="layer", **kwargs):
        self.data = data
        self.name = name
        self.kwargs = kwargs


class _FakeViewer:
    """Minimal stand-in for napari.Viewer used by the table widget."""

    def __init__(self, open_result=None, open_error=None, prepend=False):
        self.layers = _FakeLayerList()
        self.status = ""
        self.dims = SimpleNamespace(ndisplay=2)
        self._open_result = open_result
        self._open_error = open_error
        self._prepend = prepend
        self.reset_view_calls = 0
        self.added_points = []

    def _track(self, layer):
        if self._prepend:
            self.layers.insert(0, layer)
        else:
            self.layers.append(layer)
        return layer

    def add_image(self, data, channel_axis=None, **kwargs):
        name = kwargs.pop("name", "image")
        if channel_axis is not None:
            return [
                self._track(
                    _FakeLayer(
                        np.take(data, i, axis=channel_axis),
                        name=f"{name} [{i}]",
                        **kwargs,
                    )
                )
                for i in range(data.shape[channel_axis])
            ]
        return self._track(_FakeLayer(data, name=name, **kwargs))

    def add_labels(self, data, **kwargs):
        return self._track(
            _FakeLayer(data, name=kwargs.pop("name", "labels"), **kwargs)
        )

    def add_points(self, data, **kwargs):
        layer = _FakeLayer(data, name=kwargs.pop("name", "points"), **kwargs)
        self.added_points.append(layer)
        return self._track(layer)

    def _add_layer_from_data(self, data, kwargs, layer_type):
        adder = self.add_labels if layer_type == "labels" else self.add_image
        return [adder(data, **kwargs)]

    def open(self, filepath, plugin=None):
        if self._open_error is not None:
            raise self._open_error
        result = self._open_result
        if isinstance(result, list):
            for layer in result:
                self.layers.append(layer)
        elif result is not None:
            self.layers.append(result)
        return result

    def reset_view(self):
        self.reset_view_calls += 1


def _table_stub(viewer):
    """An object carrying the table widget's state, but no Qt base class."""
    stub = SimpleNamespace(
        viewer=viewer,
        current_original_images=[],
        current_processed_images=[],
        file_pairs={},
        multi_output_files={},
    )
    stub._clear_current_images = (
        lambda lst: fs.ProcessedFilesTableWidget._clear_current_images(
            stub, lst
        )
    )
    stub._should_enable_3d_view = (
        lambda data: fs.ProcessedFilesTableWidget._should_enable_3d_view(
            stub, data
        )
    )
    stub._promote_label_dtype_layers = (
        lambda layers: fs.ProcessedFilesTableWidget._promote_label_dtype_layers(
            stub, layers
        )
    )
    stub._load_processed_image = (
        lambda path: fs.ProcessedFilesTableWidget._load_processed_image(
            stub, path
        )
    )
    stub._load_original_image = (
        lambda path: fs.ProcessedFilesTableWidget._load_original_image(
            stub, path
        )
    )
    return stub


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------
class TestSmallHelpers:
    """Env-gated logging, error enrichment and label-dtype detection."""

    def test_verbose_flag_reads_env(self, monkeypatch):
        monkeypatch.delenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", raising=False)
        assert fs._channel_detection_verbose_enabled() is False
        for value in ("1", "true", "YES", " on "):
            monkeypatch.setenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", value)
            assert fs._channel_detection_verbose_enabled() is True
        monkeypatch.setenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", "0")
        assert fs._channel_detection_verbose_enabled() is False

    def test_verbose_only_log_is_suppressed(self, monkeypatch, capsys):
        monkeypatch.delenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", raising=False)
        fs._channel_detection_log("quiet", verbose_only=True)
        fs._channel_detection_log("loud")
        out = capsys.readouterr().out
        assert "quiet" not in out
        assert "loud" in out

        monkeypatch.setenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", "1")
        fs._channel_detection_log("now visible", verbose_only=True)
        assert "now visible" in capsys.readouterr().out

    def test_sigkill_message_gets_oom_hint(self):
        enhanced = fs._enhance_processing_error_message(
            "Cellpose failed with return code -9"
        )
        assert "out-of-memory" in enhanced
        assert enhanced.startswith("Cellpose failed with return code -9")

    def test_unrelated_message_is_untouched(self):
        assert (
            fs._enhance_processing_error_message("boom") == "boom"
        ), "only the SIGKILL signature should be annotated"

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (np.uint32, True),
            (np.int32, True),
            (np.int64, True),
            (np.uint64, True),
            (np.uint8, False),
            (np.uint16, False),
            (np.float32, False),
        ],
    )
    def test_is_label_image_by_dtype(self, dtype, expected):
        assert fs.is_label_image(np.zeros((2, 2), dtype=dtype)) is expected

    def test_is_label_image_without_dtype(self):
        assert fs.is_label_image(object()) is False


class TestShapeHeuristics:
    """_detect_channels_from_shape's ordered fallbacks."""

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((64, 64), (1, None)),
            ((3, 64, 64), (3, 0)),
            ((10, 3, 64, 64), (3, 1)),
            ((4, 5, 3, 64, 64), (3, 2)),
            ((1, 10, 64, 64), (10, 1)),
            ((100, 64, 64), (1, None)),
        ],
    )
    def test_shape_variants(self, shape, expected):
        assert fs._detect_channels_from_shape(shape) == expected


class TestDetectChannelsInImage:
    """Layer-list handling on top of the shape heuristic."""

    def test_multiple_image_layers_become_channels(self):
        layers = [
            (np.zeros((4, 4)), {}, "image"),
            (np.zeros((4, 4)), {}, "image"),
            (np.zeros((4, 4)), {}, "labels"),
        ]
        assert fs.detect_channels_in_image(layers) == (2, -1)

    def test_channel_axis_kwarg_is_trusted(self):
        layers = [(np.zeros((2, 4, 4)), {"channel_axis": 0}, "image")]
        assert fs.detect_channels_in_image(layers) == (2, 0)

    def test_single_layer_falls_back_to_shape(self):
        layers = [(np.zeros((3, 4, 4)), {}, "image")]
        assert fs.detect_channels_in_image(layers) == (3, 0)

    def test_channel_axis_beyond_ndim_is_ignored(self):
        layers = [(np.zeros((4, 4)), {"channel_axis": 5}, "image")]
        assert fs.detect_channels_in_image(layers) == (1, None)

    def test_list_without_image_layers(self):
        layers = [(np.zeros((4, 4)), {}, "labels"), ("malformed", "entry")]
        assert fs.detect_channels_in_image(layers) == (1, None)

    def test_object_without_shape(self):
        assert fs.detect_channels_in_image(object()) == (1, None)


# ---------------------------------------------------------------------------
# zarr metadata plumbing
# ---------------------------------------------------------------------------
class TestZarrAttrHelpers:
    """_read_zarr_root_attrs / _get_ome_multiscales / path resolution."""

    def test_reads_zattrs_and_zarr_json(self, tmp_path):
        root = tmp_path / "meta.zarr"
        root.mkdir()
        (root / ".zattrs").write_text(json.dumps({"a": 1, "shared": "v2"}))
        (root / "zarr.json").write_text(
            json.dumps({"attributes": {"b": 2, "shared": "v3"}})
        )
        attrs = fs._read_zarr_root_attrs(str(root))
        assert attrs["a"] == 1
        assert attrs["b"] == 2
        # zarr.json is merged last, so v3 metadata wins.
        assert attrs["shared"] == "v3"

    def test_corrupt_metadata_is_ignored(self, tmp_path):
        root = tmp_path / "bad.zarr"
        root.mkdir()
        (root / ".zattrs").write_text("{not json")
        (root / "zarr.json").write_text("[]")
        assert fs._read_zarr_root_attrs(str(root)) == {}

    def test_root_attrs_are_merged(self, tmp_path):
        root = SimpleNamespace(attrs={"from_root": True})
        attrs = fs._read_zarr_root_attrs(str(tmp_path / "nope"), root=root)
        assert attrs == {"from_root": True}

    def test_unusable_root_attrs_are_skipped(self, tmp_path):
        # attrs that cannot be coerced to a mapping must not propagate.
        root = SimpleNamespace(attrs=12345)
        assert fs._read_zarr_root_attrs(str(tmp_path), root=root) == {}

    def test_multiscales_top_level(self):
        assert fs._get_ome_multiscales({"multiscales": [{"x": 1}]}) == [
            {"x": 1}
        ]

    def test_multiscales_nested_under_ome(self):
        attrs = {"multiscales": [], "ome": {"multiscales": [{"y": 2}]}}
        assert fs._get_ome_multiscales(attrs) == [{"y": 2}]

    def test_multiscales_absent(self):
        assert fs._get_ome_multiscales({"ome": "not-a-dict"}) == []

    def test_resolve_dataset_path_prefers_hint(self):
        root = {"0": "a", "s0": "b"}
        assert fs._resolve_dataset_path(root, "s0") == "s0"
        assert fs._resolve_dataset_path(root, None) == "0"

    def test_resolve_dataset_path_uses_arrays_listing(self):
        class _Root:
            def __getitem__(self, key):
                raise KeyError(key)

            def arrays(self):
                return [("only", object())]

        assert fs._resolve_dataset_path(_Root(), None) == "only"

    def test_resolve_dataset_path_gives_up(self):
        # `is None` on its own would hold just as well for an empty body,
        # so pin the candidate names the resolver has to probe first.
        probed = []

        class _Root:
            def __getitem__(self, key):
                probed.append(key)
                raise KeyError(key)

        assert fs._resolve_dataset_path(_Root(), None) is None
        assert probed == ["0", "s0", "data"]

        probed.clear()
        assert fs._resolve_dataset_path(_Root(), "s1") is None
        assert probed == ["s1", "0", "s0", "data"]

    def test_resolve_first_array_on_array_root(self):
        arr = np.zeros((2, 2))
        assert fs._resolve_first_array(arr) is arr

    def test_resolve_first_array_on_group(self):
        class _Root(dict):
            def arrays(self):
                return list(self.items())

        root = _Root({"0": np.zeros((3, 3))})
        assert fs._resolve_first_array(root).shape == (3, 3)

    def test_resolve_first_array_none(self):
        # As above: prove the probing happened rather than only that None
        # came back, which a deleted body would produce too.
        probed = []

        class _Root:
            def __getitem__(self, key):
                probed.append(key)
                raise KeyError(key)

        assert fs._resolve_first_array(_Root()) is None
        assert probed == ["0", "s0", "data"]

    def test_resolve_first_array_when_dataset_cannot_be_opened(self):
        # arrays() names a dataset, but opening it blows up: that has to be
        # swallowed into None, not propagated to the caller.
        class _Root:
            def __getitem__(self, key):
                raise RuntimeError(f"corrupt chunk in {key}")

            def arrays(self):
                return [("lvl0", object())]

        assert fs._resolve_first_array(_Root()) is None


class TestZarrDetection:
    """Channel/axes detection driven by real OME-Zarr metadata."""

    @staticmethod
    def _ome_zarr(tmp_path, data, axes):
        path = str(tmp_path / "src.zarr")
        fs.save_as_zarr(data, path, axes=axes)
        return path

    def test_axes_and_channels_from_ome_metadata(self, tmp_path):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        path = self._ome_zarr(tmp_path, data, "TCZY")
        assert fs.detect_axes_from_zarr_path(path) == "TCZY"
        assert fs.detect_channels_from_zarr_path(path) == (3, 1)
        assert fs.detect_channels_for_file(path) == (3, 1)
        assert fs.detect_axes_for_file(path) == "TCZY"

    def test_axes_without_channel_axis_mean_single_channel(self, tmp_path):
        data = np.zeros((6, 4, 5), dtype=np.uint16)
        path = self._ome_zarr(tmp_path, data, "ZYX")
        assert fs.detect_axes_from_zarr_path(path) == "ZYX"
        assert fs.detect_channels_from_zarr_path(path) == (1, None)
        # The authoritative axes string wins over the shape heuristic,
        # which would otherwise call the 6-slice Z axis "channels".
        assert fs.detect_channels_for_file(path) == (1, None)

    def test_metadata_only_folder_resolves_channel_axis(self, tmp_path):
        root = tmp_path / "meta.zarr"
        root.mkdir()
        (root / ".zattrs").write_text(
            json.dumps(
                {
                    "multiscales": [
                        {
                            "axes": [{"name": n} for n in "tczyx"],
                            "datasets": [{"path": "0"}],
                        }
                    ]
                }
            )
        )
        # No array on disk: the channel axis is known, the count is not.
        assert fs.detect_channels_from_zarr_path(str(root)) == (1, 1)
        assert fs.detect_axes_from_zarr_path(str(root)) == "TCZYX"

    def test_axes_given_as_bare_strings(self, tmp_path):
        root = tmp_path / "bare.zarr"
        root.mkdir()
        (root / ".zattrs").write_text(
            json.dumps({"multiscales": [{"axes": ["c", "y", "x"]}]})
        )
        assert fs.detect_axes_from_zarr_path(str(root)) == "CYX"
        assert fs.detect_channels_from_zarr_path(str(root)) == (1, 0)

    def test_no_metadata_falls_back_to_shape(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "plain.zarr")
        root = zarr.open_group(path, mode="w")
        arr = root.create_array(
            "0", shape=(3, 8, 8), chunks=(3, 8, 8), dtype="uint8"
        )
        arr[:] = 1
        assert fs.detect_channels_from_zarr_path(path) == (3, 0)
        assert fs.detect_axes_from_zarr_path(path) is None

    def test_missing_path_is_single_channel(self, tmp_path):
        missing = str(tmp_path / "nothing.zarr")
        assert fs.detect_channels_from_zarr_path(missing) == (1, None)
        assert fs.detect_axes_from_zarr_path(missing) is None

    def test_is_ome_zarr_and_info(self, tmp_path):
        data = np.zeros((2, 4, 5), dtype=np.uint16)
        path = self._ome_zarr(tmp_path, data, "ZYX")
        assert fs.is_ome_zarr(path) is True
        assert fs.is_ome_zarr(str(tmp_path / "absent.zarr")) is False

        info = fs.get_zarr_info(path)
        assert info["is_ome_zarr"] is True
        assert info["is_multiscale"] is True
        assert info["resolution_levels"] == 1
        assert info["num_arrays"] == 1
        assert info["shape"] == (2, 4, 5)
        assert info["dtype"] == "uint16"
        assert info["has_labels"] is False

    def test_get_zarr_info_on_broken_path(self, tmp_path, capsys):
        info = fs.get_zarr_info(str(tmp_path / "missing.zarr"))
        assert info["num_arrays"] == 0
        assert info["shape"] is None
        assert "Error getting zarr info" in capsys.readouterr().out


class TestZarrLoading:
    """load_zarr_basic and its group/array/empty branches."""

    def test_group_with_named_array(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "g.zarr")
        root = zarr.open_group(path, mode="w")
        arr = root.create_array(
            "0", shape=(2, 4, 4), chunks=(1, 4, 4), dtype="uint8"
        )
        arr[:] = 7
        loaded = fs.load_zarr_basic(path)
        assert np.asarray(loaded).shape == (2, 4, 4)
        assert np.asarray(loaded)[0, 0, 0] == 7

    def test_group_without_conventional_name(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "odd.zarr")
        root = zarr.open_group(path, mode="w")
        arr = root.create_array(
            "weird", shape=(3, 3), chunks=(3, 3), dtype="uint8"
        )
        arr[:] = 2
        assert np.asarray(fs.load_zarr_basic(path)).shape == (3, 3)

    def test_bare_array_root(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "bare.zarr")
        arr = zarr.open_array(
            path, mode="w", shape=(5, 5), chunks=(5, 5), dtype="uint8"
        )
        arr[:] = 3
        assert np.asarray(fs.load_zarr_basic(path)).shape == (5, 5)

    def test_empty_group_raises(self, tmp_path, capsys):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "empty.zarr")
        zarr.open_group(path, mode="w")
        with pytest.raises(ValueError, match="No arrays found in zarr"):
            fs.load_zarr_basic(path)
        assert "Error in basic zarr loading" in capsys.readouterr().out

    def test_numpy_path_when_dask_absent(self, tmp_path, monkeypatch):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "nd.zarr")
        arr = zarr.open_array(
            path, mode="w", shape=(4, 4), chunks=(4, 4), dtype="uint8"
        )
        arr[:] = 9
        monkeypatch.setattr(fs, "DASK_AVAILABLE", False)
        loaded = fs.load_zarr_basic(path)
        assert isinstance(loaded, np.ndarray)
        assert loaded[0, 0] == 9


class TestOmeZarrReaderWrapper:
    """load_zarr_with_napari_ome_zarr's metadata enrichment and guards."""

    def test_returns_none_when_plugin_missing(self, monkeypatch, tmp_path):
        # Returning None is not the claim -- short-circuiting *before* the
        # reader lookup is, and only a live reader stub can prove it.
        def _must_not_run(_path):
            raise AssertionError("the availability guard did not fire")

        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", False)
        monkeypatch.setattr(fs, "napari_get_reader", _must_not_run)
        target = str(tmp_path / "x.zarr")
        assert fs.load_zarr_with_napari_ome_zarr(target) is None

    def test_returns_none_when_no_reader(self, monkeypatch, capsys):
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr(fs, "napari_get_reader", lambda p: None)
        assert fs.load_zarr_with_napari_ome_zarr("/x.zarr") is None
        assert "No reader available" in capsys.readouterr().out

    def test_names_colormaps_and_blending(self, monkeypatch):
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        payload = [
            (np.zeros((2, 2)), {}, "image"),
            (np.zeros((2, 2)), {}, "image"),
            (np.zeros((2, 2), np.uint32), {}, "labels"),
            (np.zeros((2, 2)), {}, "points"),
        ]
        monkeypatch.setattr(
            fs, "napari_get_reader", lambda p: (lambda q: payload)
        )
        layers = fs.load_zarr_with_napari_ome_zarr("/tmp/img.zarr")
        names = [kw["name"] for _, kw, _ in layers]
        assert names[0] == "C1: img.zarr"
        assert names[2] == "Labels3: img.zarr"
        assert names[3] == "Points4: img.zarr"
        assert layers[0][1]["blending"] == "additive"
        assert layers[0][1]["colormap"] == "red"
        assert layers[1][1]["colormap"] == "green"
        # Non-image layers must not be given an image colormap.
        assert "colormap" not in layers[2][1]

    def test_empty_layer_list(self, monkeypatch, capsys):
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr(
            fs, "napari_get_reader", lambda p: (lambda q: [])
        )
        assert fs.load_zarr_with_napari_ome_zarr("/x.zarr") is None
        assert "empty layer list" in capsys.readouterr().out

    def test_reader_exception_is_swallowed(self, monkeypatch, capsys):
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)

        def _boom(_path):
            def _read(_p):
                raise OSError("disk gone")

            return _read

        monkeypatch.setattr(fs, "napari_get_reader", _boom)
        assert fs.load_zarr_with_napari_ome_zarr("/x.zarr") is None
        assert "Failed to load" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# TIFF metadata + generic file dispatch
# ---------------------------------------------------------------------------
class TestTiffDetection:
    def test_channel_axis_from_axes_metadata(self, tmp_path):
        path = _write_tif(
            tmp_path / "cyx.tif", np.zeros((3, 8, 8), np.uint16), axes="CYX"
        )
        assert fs.detect_axes_from_tiff_path(path) == "CYX"
        assert fs.detect_channels_from_tiff_path(path) == (3, 0)
        assert fs.detect_channels_for_file(path) == (3, 0)
        assert fs.detect_axes_for_file(path) == "CYX"

    def test_axes_without_channel_are_authoritative(self, tmp_path):
        path = _write_tif(
            tmp_path / "zyx.tif", np.zeros((5, 8, 8), np.uint16), axes="ZYX"
        )
        assert fs.detect_channels_from_tiff_path(path) == (1, None)
        assert fs.detect_channels_for_file(path) == (1, None)

    def test_unreadable_tiff_is_single_channel(self, tmp_path, capsys):
        path = tmp_path / "broken.tif"
        path.write_bytes(b"definitely not a tiff")
        assert fs.detect_channels_from_tiff_path(str(path)) == (1, None)
        assert fs.detect_axes_from_tiff_path(str(path)) is None
        assert "TIFF metadata read failed" in capsys.readouterr().out

    def test_tifffile_absent_short_circuits(self, monkeypatch):
        monkeypatch.setattr(fs, "_HAS_TIFFFILE", False)
        assert fs.detect_channels_from_tiff_path("/x.tif") == (1, None)
        assert fs.detect_axes_from_tiff_path("/x.tif") is None

    def test_shape_fallback_when_axes_mismatch(self, tmp_path, monkeypatch):
        path = _write_tif(tmp_path / "plain.tif", np.zeros((3, 8, 8), np.uint8))

        real_tiff_file = tifffile.TiffFile

        class _Wrapped:
            def __init__(self, fp):
                self._tif = real_tiff_file(fp)

            def __enter__(self):
                series = self._tif.series[0]
                self.series = [
                    SimpleNamespace(axes="ABCDEF", shape=series.shape)
                ]
                return self

            def __exit__(self, *exc):
                self._tif.close()
                return False

        monkeypatch.setattr(fs.tifffile, "TiffFile", _Wrapped)
        # Axes length does not match the shape, so the heuristic runs.
        assert fs.detect_channels_from_tiff_path(path) == (3, 0)

    def test_axes_for_file_from_data_ndim(self):
        assert fs.detect_axes_for_file("x.png", np.zeros((2, 3, 4))) == "ZYX"
        assert fs.detect_axes_for_file("x.png", np.zeros((4, 4))) == "YX"
        assert fs.detect_axes_for_file("x.png", None) is None
        assert fs.detect_axes_for_file("x.png", np.zeros((2,) * 6)) is None

    def test_channels_for_file_loads_when_needed(self, tmp_path, monkeypatch):
        path = str(tmp_path / "custom.dat")
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: np.zeros((3, 8, 8))
        )
        assert fs.detect_channels_for_file(path) == (3, 0)


class TestCellposeDimOrder:
    def test_hint_wins_and_channel_is_stripped(self, tmp_path):
        path = _write_tif(
            tmp_path / "z.tif", np.zeros((5, 8, 8), np.uint16), axes="ZYX"
        )
        assert (
            fs.resolve_cellpose_dim_order(path, None, None, "TCZYX") == "TZYX"
        )

    def test_auto_uses_file_metadata(self, tmp_path):
        path = _write_tif(
            tmp_path / "z.tif", np.zeros((5, 8, 8), np.uint16), axes="ZYX"
        )
        assert fs.resolve_cellpose_dim_order(path, None, None, "Auto") == "ZYX"

    def test_multichannel_without_selection_raises(self, tmp_path):
        path = _write_tif(
            tmp_path / "c.tif", np.zeros((3, 8, 8), np.uint16), axes="CYX"
        )
        with pytest.raises(ValueError, match="specific channel selection"):
            fs.resolve_cellpose_dim_order(path, None, "all", "Auto")

    def test_unsupported_axes_fall_back_to_ndim(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", "1")
        path = str(tmp_path / "broken.tif")
        with open(path, "wb") as fh:
            fh.write(b"nope")
        resolved = fs.resolve_cellpose_dim_order(
            path, np.zeros((4, 4, 4)), "0", "QQQ"
        )
        assert resolved == "ZYX"
        assert "ignoring unsupported metadata axes 'QQQ'" in (
            capsys.readouterr().out
        )

    def test_default_when_nothing_is_known(self, tmp_path):
        # No axes metadata anywhere and data with no ndim -> the 2D default.
        path = str(tmp_path / "mystery.dat")
        assert fs.resolve_cellpose_dim_order(path, [], "0", "Auto") == "YX"


# ---------------------------------------------------------------------------
# output resolution + writers
# ---------------------------------------------------------------------------
class TestOutputResolution:
    @pytest.mark.parametrize(
        "hint,extracted,ndim,expected",
        [
            ("TCZYX", True, 4, "TZYX"),
            ("TCZYX", False, 5, "TCZYX"),
            ("CYX", True, 2, "YX"),
            ("Auto", False, 3, "ZYX"),
            ("Auto", False, 2, "YX"),
            ("Auto", False, 4, "TZYX"),
            ("Auto", False, 5, "TCZYX"),
            # Hint length does not match ndim -> fall back to the table.
            ("TCZYX", False, 3, "ZYX"),
            # ndim outside the table -> conservative default.
            ("None", False, 7, "YX"),
        ],
    )
    def test_dim_order(self, hint, extracted, ndim, expected):
        assert (
            fs._resolve_output_dim_order(hint, extracted, ndim) == expected
        )

    def test_scale_is_axis_aligned(self, tmp_path):
        path = str(tmp_path / "scaled.tif")
        tifffile.imwrite(
            path,
            np.zeros((4, 8, 8), np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "PhysicalSizeZ": 2.0,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeX": 0.5,
            },
        )
        # T has no physical size and defaults to 1.0; Z/Y/X carry the
        # OME PhysicalSize values, laid out in the requested axis order.
        assert fs._resolve_output_scale(path, "TZYX") == pytest.approx(
            (1.0, 2.0, 0.5, 0.5)
        )
        # Reordering the axes string reorders the scale with it -- a scale
        # that merely had the right values would not survive this.
        assert fs._resolve_output_scale(path, "XYZ") == pytest.approx(
            (0.5, 0.5, 2.0)
        )

    def test_scale_defaults_without_source(self):
        assert fs._resolve_output_scale(None, "ZYX") == (1.0, 1.0, 1.0)


class TestLazyArrayHelpers:
    def test_is_lazy_array(self):
        da = pytest.importorskip("dask.array")
        assert fs._is_lazy_array(da.zeros((4, 4), chunks=(2, 2))) is True
        assert fs._is_lazy_array(np.zeros((4, 4))) is False
        assert fs._is_lazy_array(None) is False

    def test_iter_lazy_planes_uses_module_budget(self, monkeypatch):
        da = pytest.importorskip("dask.array")
        data = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
        lazy = da.from_array(data, chunks=(1, 4, 4))
        monkeypatch.setattr(fs, "_STREAM_BLOCK_BYTES", 64)
        planes = list(fs._iter_lazy_planes(lazy, np.uint16))
        assert len(planes) == 3
        np.testing.assert_array_equal(np.stack(planes), data)

    def test_dense_write_round_trips(self, tmp_path):
        data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        out = str(tmp_path / "dense.tif")
        fs._write_tiff_output(data, out, np.uint16, {"axes": "ZYX"}, False)
        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_lazy_write_streams_planes(self, tmp_path, capsys):
        da = pytest.importorskip("dask.array")
        data = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
        lazy = da.from_array(data, chunks=(1, 5, 6))
        out = str(tmp_path / "streamed.tif")
        fs._write_tiff_output(lazy, out, np.uint16, {"axes": "ZYX"}, False)
        np.testing.assert_array_equal(tifffile.imread(out), data)
        assert "Streaming" in capsys.readouterr().out


class TestChunkProgressCallback:
    def test_reports_milestones(self, capsys):
        da = pytest.importorskip("dask.array")
        callback = fs._ChunkProgressCallback("demo", milestone_pct=25)
        result = da.ones((8, 8), chunks=(4, 4)).sum().compute(
            callbacks=[callback.pair]
        )
        assert result == 64
        out = capsys.readouterr().out
        assert "[demo] processing" in out
        assert "chunks)" in out
        assert callback._total > 0  # else the equality below is vacuous
        assert callback._done == callback._total

    def test_posttask_without_start_is_a_noop(self, capsys):
        callback = fs._ChunkProgressCallback("demo")
        # No _start_state yet, so _total is 0: the counter must not move
        # (and the percentage must not divide by it).
        callback._posttask("k", None, None, None, None)
        assert callback._done == 0
        assert capsys.readouterr().out == ""
        # Paired with the live case, so a _posttask that did nothing at
        # all would not pass for a _posttask that correctly does nothing
        # *yet*.
        callback._start_state(None, {"ready": [1, 2], "waiting": []})
        assert callback._total == 2
        callback._posttask("k", None, None, None, None)
        assert callback._done == 1
        assert "[demo] 50% (1/2 chunks)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# save_as_zarr / load_image_file
# ---------------------------------------------------------------------------
class TestSaveAsZarr:
    def test_round_trip_with_scale(self, tmp_path):
        pytest.importorskip("ome_zarr")
        data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
        out = str(tmp_path / "saved")  # no extension on purpose
        fs.save_as_zarr(data, out, axes="ZYX", scale=(2.0, 0.5, 0.5))
        assert os.path.isdir(out + ".zarr")
        loaded = np.asarray(fs.load_zarr_basic(out + ".zarr"))
        np.testing.assert_array_equal(loaded, data)
        assert fs.detect_axes_from_zarr_path(out + ".zarr") == "ZYX"

    @pytest.mark.parametrize(
        "shape,expected_axes",
        [
            ((4, 5), "YX"),
            ((2, 4, 5), "ZYX"),
            ((2, 3, 4, 5), "CZYX"),
            ((2, 2, 3, 4, 5), "TCZYX"),
        ],
    )
    def test_axes_inferred_from_ndim(self, tmp_path, shape, expected_axes):
        pytest.importorskip("ome_zarr")
        data = np.zeros(shape, dtype=np.uint8)
        out = str(tmp_path / "inferred.zarr")
        # A wrong-length axes string is discarded and re-inferred.
        fs.save_as_zarr(data, out, axes="Q")
        assert fs.detect_axes_from_zarr_path(out) == expected_axes

    def test_explicit_chunks_are_used(self, tmp_path):
        pytest.importorskip("ome_zarr")
        data = np.zeros((4, 8, 8), dtype=np.uint8)
        out = str(tmp_path / "chunked.zarr")
        fs.save_as_zarr(data, out, axes="ZYX", chunks=(2, 4, 4))
        info = fs.get_zarr_info(out)
        assert info["chunks"] == (2, 4, 4)

    def test_write_failure_raises_valueerror(self, tmp_path, monkeypatch):
        pytest.importorskip("ome_zarr")
        import ome_zarr.writer

        def _boom(**kwargs):
            raise RuntimeError("writer exploded")

        monkeypatch.setattr(ome_zarr.writer, "write_image", _boom)
        with pytest.raises(ValueError, match="Failed to save Zarr"):
            fs.save_as_zarr(
                np.zeros((2, 2), np.uint8), str(tmp_path / "x.zarr")
            )


class TestLoadImageFile:
    def test_tiff_dense(self, tmp_path):
        data = np.arange(64, dtype=np.uint16).reshape(8, 8)
        path = _write_tif(tmp_path / "a.tif", data)
        np.testing.assert_array_equal(fs.load_image_file(path), data)

    def test_tiff_lazy_when_requested(self, tmp_path, monkeypatch):
        pytest.importorskip("dask.array")
        data = np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8)
        path = _write_tif(tmp_path / "stack.tif", data)
        monkeypatch.setattr(fs, "_LAZY_TIFF_MIN_BYTES", 0)
        lazy = fs.load_image_file(path, prefer_lazy=True)
        assert fs._is_lazy_array(lazy)
        np.testing.assert_array_equal(np.asarray(lazy), data)

    def test_small_tiff_is_never_lazy(self, tmp_path, monkeypatch):
        pytest.importorskip("dask.array")
        data = np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8)
        path = _write_tif(tmp_path / "stack.tif", data)
        # Under the module's byte budget the file is read densely...
        assert data.nbytes < fs._LAZY_TIFF_MIN_BYTES
        assert fs._lazy_load_tiff(path) is None
        # ...and it really is the budget deciding, not the page layout:
        # the identical file goes lazy once the budget drops to its size.
        monkeypatch.setattr(fs, "_LAZY_TIFF_MIN_BYTES", data.nbytes)
        lazy = fs._lazy_load_tiff(path)
        assert fs._is_lazy_array(lazy)
        np.testing.assert_array_equal(np.asarray(lazy), data)

    def test_lazy_open_failure_returns_none(self, tmp_path, capsys):
        path = tmp_path / "broken.tif"
        path.write_bytes(b"nope")
        assert fs._lazy_load_tiff(str(path)) is None
        assert "Lazy TIFF open failed" in capsys.readouterr().out

    def test_zarr_goes_through_ome_reader(self, tmp_path, monkeypatch, capsys):
        payload = [(np.zeros((4, 4)), {"name": "c"}, "image")]
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr(
            fs, "load_zarr_with_napari_ome_zarr", lambda p: payload
        )
        assert fs.load_image_file(str(tmp_path / "x.zarr")) is payload
        assert "Loaded 1 layer from OME-Zarr" in capsys.readouterr().out

    def test_zarr_reader_error_falls_back(self, tmp_path, monkeypatch, capsys):
        pytest.importorskip("zarr")
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = str(tmp_path / "fallback.zarr")
        fs.save_as_zarr(data, path, axes="YX")

        def _boom(_p):
            raise ValueError("reader down")

        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr(fs, "load_zarr_with_napari_ome_zarr", _boom)
        loaded = fs.load_image_file(path)
        np.testing.assert_array_equal(np.asarray(loaded), data)
        assert "falling back to basic zarr loading" in capsys.readouterr().out

    def test_non_tiff_uses_skimage(self, tmp_path, monkeypatch):
        called = {}

        def _fake_imread(path):
            called["path"] = path
            return np.zeros((2, 2))

        monkeypatch.setattr(fs, "imread", _fake_imread)
        out = fs.load_image_file(str(tmp_path / "img.png"))
        assert out.shape == (2, 2)
        assert called["path"].endswith("img.png")


# ---------------------------------------------------------------------------
# ProcessingWorker
# ---------------------------------------------------------------------------
def _worker(tmp_path, func, files, **kwargs):
    """Build the worker the FileResultsWidget uses, with sane defaults."""
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return fs.ProcessingWorker(
        files,
        func,
        kwargs.pop("param_values", {}),
        str(out),
        kwargs.pop("input_suffix", ""),
        kwargs.pop("output_suffix", "_proc"),
        output_format=kwargs.pop("output_format", "tiff"),
    )


class TestProcessingWorkerEarlyExits:
    """process_file's guard clauses before any array is written."""

    def test_pre_filter_declines_file(self, tmp_path, capsys):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def never_needed(image):
            raise AssertionError("must not be called")

        never_needed.file_pre_filter = lambda path, params: False
        worker = _worker(tmp_path, never_needed, [src])
        assert worker.process_file(src) is None
        assert "Skipping" in capsys.readouterr().out

    def test_function_returning_none_skips_save(self, tmp_path, capsys):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def nothing(image):
            return None

        worker = _worker(tmp_path, nothing, [src])
        assert worker.process_file(src) is None
        assert "returned None" in capsys.readouterr().out

    def test_skip_load_function_owns_its_io(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))
        seen = {}

        def merge_everything(image, _output_folder=None, **kwargs):
            seen["image"] = image
            written = os.path.join(_output_folder, "merged.tif")
            tifffile.imwrite(written, np.ones((2, 2), np.uint8))
            return written

        merge_everything.skip_load = True
        worker = _worker(tmp_path, merge_everything, [src])
        result = worker.process_file(src)

        assert seen["image"] is None, "skip_load must not load the array"
        assert result["original_file"] == src
        assert os.path.isfile(result["processed_file"])

    def test_folder_function_returning_input_saves_nothing(
        self, tmp_path, capsys
    ):
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        src = _write_tif(tmp_path / "a.tif", data)

        def merge_timepoints_folder(image):
            return image

        worker = _worker(tmp_path, merge_timepoints_folder, [src])
        assert worker.process_file(src) is None
        assert "unchanged image" in capsys.readouterr().out

    def test_error_is_enhanced_and_reraised(self, tmp_path, capsys):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def exploding(image):
            raise RuntimeError("Cellpose failed with return code -9")

        worker = _worker(tmp_path, exploding, [src])
        with pytest.raises(RuntimeError, match="return code -9"):
            worker.process_file(src)
        out = capsys.readouterr().out
        assert "Diagnostic:" in out
        assert "out-of-memory" in out


class TestProcessingWorkerOutputs:
    """The various shapes of result process_file knows how to persist."""

    def test_points_result_written_as_npy_and_csv(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))
        coords = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )

        def spots(image):
            return coords

        worker = _worker(
            tmp_path, spots, [src], param_values={"output_csv": True}
        )
        result = worker.process_file(src)

        assert result["processed_file"].endswith("a_spots.npy")
        np.testing.assert_array_equal(
            np.load(result["processed_file"]), coords
        )
        csv_path = result["processed_file"].replace(".npy", ".csv")
        assert os.path.isfile(csv_path)
        with open(csv_path) as handle:
            assert handle.readline().strip() == "y,x"

    def test_points_result_without_csv(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))
        coords = np.zeros((2, 3), dtype=np.float64)

        def spots3d(image):
            return coords

        worker = _worker(tmp_path, spots3d, [src])
        result = worker.process_file(src)
        # "_spots" is a fixed suffix on the input stem, not the function
        # name -- the output_suffix ("_proc") plays no part here.
        assert os.path.basename(result["processed_file"]) == "a_spots.npy"
        np.testing.assert_array_equal(
            np.load(result["processed_file"]), coords
        )
        assert not os.path.isfile(
            result["processed_file"].replace(".npy", ".csv")
        )

    def test_three_way_layer_split(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def subdivide(image):
            return (
                np.ones((4, 4), np.uint16),
                np.full((4, 4), 2, np.uint16),
                np.full((4, 4), 3, np.uint16),
            )

        worker = _worker(tmp_path, subdivide, [src], output_suffix="_layer")
        result = worker.process_file(src)
        names = [os.path.basename(p) for p in result["processed_files"]]
        assert names == ["a_inner.tif", "a_middle.tif", "a_outer.tif"]
        # Layer outputs are forced to uint32 so napari treats them as labels.
        assert tifffile.imread(result["processed_files"][0]).dtype == np.uint32

    def test_multi_output_mixes_labels_and_images(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def two_outputs(image):
            return [
                np.arange(16, dtype=np.uint32).reshape(4, 4),
                np.full((4, 4), 7, np.uint16),
            ]

        worker = _worker(
            tmp_path, two_outputs, [src], output_suffix="_out.tif"
        )
        result = worker.process_file(src)
        names = [os.path.basename(p) for p in result["processed_files"]]
        assert names == ["a_ch1_out.tif", "a_ch2_out.tif"]
        assert tifffile.imread(result["processed_files"][0]).dtype == np.uint32
        assert tifffile.imread(result["processed_files"][1])[0, 0] == 7

    def test_multi_output_skips_non_arrays(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def mixed(image):
            return [np.zeros((4, 4), np.uint16), "not an array"]

        worker = _worker(tmp_path, mixed, [src], output_suffix="_out.tif")
        result = worker.process_file(src)
        # The string is dropped, and numbering does not skip a beat: the
        # array keeps the _ch1_ slot it would have had on its own.
        assert [os.path.basename(p) for p in result["processed_files"]] == [
            "a_ch1_out.tif"
        ]
        saved = tifffile.imread(result["processed_files"][0])
        assert saved.dtype == np.uint16
        np.testing.assert_array_equal(saved, np.zeros((4, 4)))

    def test_label_output_is_saved_as_uint32(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def segment(image):
            return np.arange(16, dtype=np.int64).reshape(4, 4)

        worker = _worker(tmp_path, segment, [src])
        result = worker.process_file(src)
        saved = tifffile.imread(result["processed_file"])
        assert saved.dtype == np.uint32
        np.testing.assert_array_equal(saved, np.arange(16).reshape(4, 4))

    def test_zarr_output_format(self, tmp_path):
        pytest.importorskip("ome_zarr")
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        src = _write_tif(tmp_path / "a.tif", data)

        def double(image):
            return image * 2

        worker = _worker(tmp_path, double, [src], output_format="zarr")
        result = worker.process_file(src)
        assert result["processed_file"].endswith("a_proc.zarr")
        assert os.path.isdir(result["processed_file"])
        np.testing.assert_array_equal(
            np.asarray(fs.load_zarr_basic(result["processed_file"])), data * 2
        )

    def test_input_suffix_is_stripped_from_output_name(self, tmp_path):
        src = _write_tif(tmp_path / "sample_raw.tif", np.zeros((4, 4), np.uint16))

        def identity(image):
            return image + 1

        worker = _worker(
            tmp_path, identity, [src], input_suffix="_raw", output_suffix="_p"
        )
        result = worker.process_file(src)
        assert os.path.basename(result["processed_file"]) == "sample_p.tif"


class TestProcessingWorkerChannelSplitting:
    """dimension_order hints and the Auto heuristic decide channel splits."""

    def test_hint_forces_split(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def triple(image):
            return np.stack([image, image + 1, image + 2])

        worker = _worker(
            tmp_path,
            triple,
            [src],
            param_values={"dimension_order": "CYX"},
        )
        result = worker.process_file(src)
        names = [os.path.basename(p) for p in result["processed_files"]]
        assert names == [
            "a_proc_channel_0.tif",
            "a_proc_channel_1.tif",
            "a_proc_channel_2.tif",
        ]

    def test_time_hint_prevents_split(self, tmp_path, capsys):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def triple(image):
            return np.stack([image, image, image])

        worker = _worker(
            tmp_path,
            triple,
            [src],
            param_values={"dimension_order": "TYX"},
        )
        result = worker.process_file(src)
        assert "processed_file" in result
        assert tifffile.imread(result["processed_file"]).shape == (3, 4, 4)
        assert "will NOT split channels" in capsys.readouterr().out

    def test_auto_splits_on_gained_leading_axis(self, tmp_path, capsys):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))

        def variants(image):
            return np.stack([image, image + 1, image + 2])

        worker = _worker(tmp_path, variants, [src])
        result = worker.process_file(src)
        assert len(result["processed_files"]) == 3
        assert "gained a new leading axis" in capsys.readouterr().out

    def test_auto_splits_when_metadata_says_channels(self, tmp_path, capsys):
        src = _write_tif(
            tmp_path / "c.tif", np.zeros((3, 4, 4), np.uint16), axes="CYX"
        )

        def identity(image):
            return image

        worker = _worker(tmp_path, identity, [src])
        result = worker.process_file(src)
        assert len(result["processed_files"]) == 3
        assert "metadata indicates 3 channels" in capsys.readouterr().out

    def test_auto_keeps_z_stack_intact(self, tmp_path):
        src = _write_tif(
            tmp_path / "z.tif", np.zeros((3, 4, 4), np.uint16), axes="ZYX"
        )

        def identity(image):
            return image

        worker = _worker(tmp_path, identity, [src])
        result = worker.process_file(src)
        # No new leading axis, and metadata says Z, so nothing is split.
        assert "processed_file" in result
        assert tifffile.imread(result["processed_file"]).shape == (3, 4, 4)


class TestProcessingWorkerChannelExtraction:
    """The `channel` parameter slices the input before the function runs."""

    def test_selected_channel_is_extracted(self, tmp_path, capsys):
        data = np.stack(
            [np.full((4, 4), i, np.uint16) for i in range(3)]
        )
        src = _write_tif(tmp_path / "c.tif", data, axes="CYX")
        seen = {}

        def identity(image, channel=None):
            seen["shape"] = image.shape
            seen["value"] = int(np.asarray(image).flat[0])
            return image

        worker = _worker(
            tmp_path, identity, [src], param_values={"channel": "1"}
        )
        result = worker.process_file(src)

        assert seen["shape"] == (4, 4)
        assert seen["value"] == 1
        assert "processed_file" in result
        assert "Channel 1 extracted" in capsys.readouterr().out

    def test_blank_channel_string_is_ignored(self, tmp_path):
        data = np.zeros((3, 4, 4), np.uint16)
        src = _write_tif(tmp_path / "c.tif", data, axes="CYX")
        seen = {}

        def identity(image, channel=None):
            seen["shape"] = image.shape
            return image

        worker = _worker(
            tmp_path, identity, [src], param_values={"channel": "   "}
        )
        worker.process_file(src)
        assert seen["shape"] == (3, 4, 4)

    def test_out_of_range_channel_warns_and_keeps_array(
        self, tmp_path, capsys
    ):
        data = np.zeros((3, 4, 4), np.uint16)
        src = _write_tif(tmp_path / "c.tif", data, axes="CYX")
        seen = {}

        def identity(image, channel=None):
            seen["shape"] = image.shape
            return image

        worker = _worker(
            tmp_path, identity, [src], param_values={"channel": "9"}
        )
        worker.process_file(src)
        assert seen["shape"] == (3, 4, 4)
        assert "out of range" in capsys.readouterr().out

    def test_cellpose_gets_a_channel_free_dim_order(self, tmp_path):
        src = _write_tif(
            tmp_path / "z.tif", np.zeros((3, 4, 4), np.uint16), axes="ZYX"
        )
        seen = {}

        def cellpose_segmentation(image, **kwargs):
            seen.update(kwargs)
            return image

        worker = _worker(
            tmp_path,
            cellpose_segmentation,
            [src],
            param_values={"dimension_order": "TCZYX"},
        )
        worker.process_file(src)
        assert seen["dim_order"] == "TZYX"
        assert "dimension_order" not in seen

    def test_dimension_order_is_remapped_to_dim_order(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))
        seen = {}

        def takes_dim_order(image, dim_order="Auto"):
            seen["dim_order"] = dim_order
            return image

        worker = _worker(
            tmp_path,
            takes_dim_order,
            [src],
            param_values={"dimension_order": "YX"},
        )
        worker.process_file(src)
        assert seen["dim_order"] == "YX"


class TestProcessingWorkerLazyPaths:
    """Dask inputs/outputs are computed or streamed, never silently densified."""

    def test_dask_input_is_computed_for_eager_functions(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("dask.array")
        data = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        src = _write_tif(tmp_path / "z.tif", data, axes="ZYX")
        monkeypatch.setattr(fs, "_LAZY_TIFF_MIN_BYTES", 0)
        seen = {}

        def eager(image, _source_filepath=None):
            # Declaring _source_filepath opts into a lazy load, but this
            # function is then handed the lazy array as-is.
            seen["lazy"] = fs._is_lazy_array(image)
            return np.asarray(image) + 1

        worker = _worker(tmp_path, eager, [src])
        result = worker.process_file(src)
        assert seen["lazy"] is True
        np.testing.assert_array_equal(
            tifffile.imread(result["processed_file"]), data + 1
        )

    def test_dask_input_densified_for_plain_functions(
        self, tmp_path, monkeypatch, capsys
    ):
        pytest.importorskip("dask.array")
        data = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        src = _write_tif(tmp_path / "z.tif", data, axes="ZYX")
        monkeypatch.setattr(fs, "_LAZY_TIFF_MIN_BYTES", 0)
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: fs._lazy_load_tiff(p)
        )
        seen = {}

        def plain(image):
            seen["type"] = type(image)
            return image

        worker = _worker(tmp_path, plain, [src])
        worker.process_file(src)
        assert seen["type"] is np.ndarray
        assert "Converting dask array to numpy" in capsys.readouterr().out

    def test_small_dask_result_is_computed(self, tmp_path, capsys):
        da = pytest.importorskip("dask.array")
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        src = _write_tif(tmp_path / "a.tif", data)

        def lazy_double(image):
            return da.from_array(image, chunks=(2, 2)) * 2

        worker = _worker(tmp_path, lazy_double, [src])
        result = worker.process_file(src)
        np.testing.assert_array_equal(
            tifffile.imread(result["processed_file"]), data * 2
        )
        assert "Dask computation complete" in capsys.readouterr().out

    def test_large_dask_result_is_streamed(
        self, tmp_path, monkeypatch, capsys
    ):
        da = pytest.importorskip("dask.array")
        data = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
        src = _write_tif(tmp_path / "z.tif", data, axes="ZYX")

        def lazy_identity(image):
            return da.from_array(image, chunks=(1, 4, 4))

        monkeypatch.setattr(fs, "_STREAM_BLOCK_BYTES", 32)
        worker = _worker(tmp_path, lazy_identity, [src])
        result = worker.process_file(src)

        out = capsys.readouterr().out
        assert "Keeping Dask result lazy" in out
        np.testing.assert_array_equal(
            tifffile.imread(result["processed_file"]), data
        )

    def test_max_dask_workers_attribute_is_honoured(self, tmp_path):
        da = pytest.importorskip("dask.array")
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        src = _write_tif(tmp_path / "a.tif", data)

        def capped(image):
            return da.from_array(image, chunks=(2, 2)) + 1

        capped.max_dask_workers = 2
        worker = _worker(tmp_path, capped, [src])
        result = worker.process_file(src)
        np.testing.assert_array_equal(
            tifffile.imread(result["processed_file"]), data + 1
        )


class TestProcessingWorkerZarrInput:
    """Multi-layer OME-Zarr input is unwrapped down to a single array."""

    def test_layer_list_input_is_unwrapped(self, tmp_path, capsys):
        pytest.importorskip("ome_zarr")
        data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
        src = str(tmp_path / "in.zarr")
        fs.save_as_zarr(data, src, axes="ZYX")
        seen = {}

        def identity(image):
            seen["shape"] = tuple(np.asarray(image).shape)
            return np.asarray(image)

        worker = _worker(tmp_path, identity, [src])
        result = worker.process_file(src)
        assert seen["shape"] == (2, 4, 5)
        # A .zarr input with TIFF output falls back to a .tif extension.
        assert result["processed_file"].endswith(".tif")

    def test_layer_list_without_image_layer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        payload = [(np.full((4, 4), 5, np.uint16), {}, "labels")]
        monkeypatch.setattr(fs, "load_image_file", lambda p, **k: payload)
        seen = {}

        def identity(image):
            seen["value"] = int(image[0, 0])
            return image

        src = str(tmp_path / "labels.zarr")
        worker = _worker(tmp_path, identity, [src])
        worker.process_file(src)
        assert seen["value"] == 5

    def test_empty_layer_list_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fs, "load_image_file", lambda p, **k: [])

        def identity(image):
            return image

        src = str(tmp_path / "empty.zarr")
        worker = _worker(tmp_path, identity, [src])
        with pytest.raises(ValueError, match="No image data found"):
            worker.process_file(src)


class TestProcessingWorkerRun:
    """run() fans files out to a pool and reports through its signals."""

    def test_run_emits_progress_and_results(self, tmp_path, qapp):
        files = [
            _write_tif(tmp_path / f"f{i}.tif", np.zeros((4, 4), np.uint16))
            for i in range(3)
        ]

        def identity(image):
            return image + 1

        worker = _worker(tmp_path, identity, files)
        progress, results, finished = [], [], []
        worker.progress_updated.connect(progress.append)
        worker.file_processed.connect(results.append)
        worker.processing_finished.connect(lambda: finished.append(True))

        worker.run()

        assert progress == [33, 66, 100]
        assert sorted(r["original_file"] for r in results) == sorted(files)
        for res in results:
            base = os.path.basename(res["original_file"])
            assert os.path.basename(res["processed_file"]) == base.replace(
                ".tif", "_proc.tif"
            )
            np.testing.assert_array_equal(
                tifffile.imread(res["processed_file"]),
                np.ones((4, 4), np.uint16),
            )
        assert finished == [True]

    def test_run_reports_per_file_errors(self, tmp_path, qapp):
        good = _write_tif(tmp_path / "good.tif", np.zeros((4, 4), np.uint16))
        bad = _write_tif(tmp_path / "bad.tif", np.zeros((4, 4), np.uint16))

        def picky(image, _source_filepath=None):
            if os.path.basename(_source_filepath).startswith("bad"):
                raise ValueError("cannot handle this one")
            return np.asarray(image)

        worker = _worker(tmp_path, picky, [good, bad])
        errors = []
        worker.error_occurred.connect(lambda p, m: errors.append((p, m)))
        worker.run()

        assert len(errors) == 1
        assert errors[0][0] == bad
        assert "cannot handle this one" in errors[0][1]

    def test_stop_halts_the_loop(self, tmp_path, qapp):
        files = [
            _write_tif(tmp_path / f"f{i}.tif", np.zeros((4, 4), np.uint16))
            for i in range(3)
        ]

        def identity(image):
            return image

        worker = _worker(tmp_path, identity, files)
        worker.stop()
        assert worker.stop_requested is True

        results = []
        worker.file_processed.connect(results.append)
        worker.run()
        assert results == [], "a stop request must short-circuit the loop"


# ---------------------------------------------------------------------------
# ProcessedFilesTableWidget
# ---------------------------------------------------------------------------
class TestTableWidget:
    """Row bookkeeping for original/processed file pairs."""

    def test_initial_files_populate_rows(self, tmp_path, qapp):
        files = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
        table = fs.ProcessedFilesTableWidget(_FakeViewer())
        table.add_initial_files(files)

        assert table.rowCount() == 2
        assert table.item(0, 0).text() == "a.tif"
        assert table.item(0, 1).text() == ""
        assert table.file_pairs[files[0]]["row"] == 0
        assert table.file_pairs[files[1]]["processed"] is None

        # Re-adding clears the previous contents rather than appending.
        table.add_initial_files(files[:1])
        assert table.rowCount() == 1
        assert set(table.file_pairs) == {files[0]}

    def test_single_processed_file_lands_in_column_two(self, tmp_path, qapp):
        src = str(tmp_path / "a.tif")
        table = fs.ProcessedFilesTableWidget(_FakeViewer())
        table.add_initial_files([src])
        table.update_processed_files(
            [{"original_file": src, "processed_file": str(tmp_path / "a_p.tif")}]
        )
        assert table.item(0, 1).text() == "a_p.tif"
        assert table.file_pairs[src]["processed"].endswith("a_p.tif")

    def test_multi_output_becomes_a_combobox(self, tmp_path, qapp):
        src = str(tmp_path / "a.tif")
        outputs = [str(tmp_path / f"a_ch{i}.tif") for i in range(3)]
        table = fs.ProcessedFilesTableWidget(_FakeViewer())
        table.add_initial_files([src])
        table.update_processed_files(
            [{"original_file": src, "processed_files": outputs}]
        )

        combo = table.cellWidget(0, 1)
        assert combo is not None
        assert combo.count() == 3
        assert combo.itemData(1) == outputs[1]
        assert table.multi_output_files[src] == outputs
        assert table.file_pairs[src]["processed"] == outputs[0]

        loaded = []
        table._load_processed_image = loaded.append
        combo.setCurrentIndex(2)
        assert loaded == [outputs[2]]

    def test_unknown_original_file_is_ignored(self, tmp_path, qapp):
        table = fs.ProcessedFilesTableWidget(_FakeViewer())
        table.add_initial_files([str(tmp_path / "a.tif")])
        src = str(tmp_path / "a.tif")
        table.update_processed_files(
            [{"original_file": "/elsewhere/x.tif", "processed_file": "/y.tif"}]
        )
        assert table.item(0, 1).text() == ""
        assert table.rowCount() == 1
        assert table.file_pairs[src]["processed"] is None
        assert table.multi_output_files == {}

    def test_double_click_loads_processed_column(self, tmp_path, qapp):
        src = str(tmp_path / "a.tif")
        table = fs.ProcessedFilesTableWidget(_FakeViewer())
        table.add_initial_files([src])
        table.update_processed_files(
            [{"original_file": src, "processed_file": str(tmp_path / "p.tif")}]
        )
        loaded = []
        table._load_processed_image = loaded.append

        table._handle_cell_double_click(0, 1)
        assert loaded == [str(tmp_path / "p.tif")]

        # Column 0 is the original file; the handler ignores it.
        table._handle_cell_double_click(0, 0)
        assert len(loaded) == 1


class TestClearAndPromoteLayers:
    def test_clear_removes_by_reference(self):
        viewer = _FakeViewer()
        layer = _FakeLayer(np.zeros((2, 2)), name="one")
        viewer.layers.append(layer)
        tracked = [layer]
        fs.ProcessedFilesTableWidget._clear_current_images(
            SimpleNamespace(viewer=viewer), tracked
        )
        assert viewer.layers == []
        assert tracked == []

    def test_clear_falls_back_to_name_lookup(self):
        viewer = _FakeViewer()
        present = _FakeLayer(np.zeros((2, 2)), name="shared")
        viewer.layers.append(present)
        stale = _FakeLayer(np.zeros((2, 2)), name="shared")
        tracked = [stale]
        fs.ProcessedFilesTableWidget._clear_current_images(
            SimpleNamespace(viewer=viewer), tracked
        )
        # The live layer went, matched by name rather than identity...
        assert viewer.layers == []
        # ...and the caller's tracking list was emptied with it.
        assert tracked == []

    def test_clear_survives_removal_errors(self, capsys):
        class _Angry(list):
            def remove(self, item):
                raise KeyError("gone")

        viewer = SimpleNamespace(layers=_Angry())
        layer = _FakeLayer(np.zeros((2, 2)), name="x")
        viewer.layers.append(layer)
        fs.ProcessedFilesTableWidget._clear_current_images(
            SimpleNamespace(viewer=viewer), [layer]
        )
        assert "Could not remove layer" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((8, 8), False),
            ((12, 8, 8), True),
            ((5, 8, 8), False),
            ((3, 12, 8, 8), True),
            ((2, 5, 10, 8, 8), True),
        ],
    )
    def test_should_enable_3d_view(self, shape, expected):
        result = fs.ProcessedFilesTableWidget._should_enable_3d_view(
            None, np.zeros(shape, np.uint8)
        )
        # `is`, not bool(): a method that answered a constant None would
        # otherwise sail through every False case here.
        assert result is expected

    def test_should_enable_3d_view_channels_plus_2d(self):
        # (C, Y, X) leaves only two meaningful dims -> no 3D view.  The
        # source falls off the end of the method here and answers None
        # instead of False (its `return False` is stranded, unreachable,
        # at the bottom of the *next* method), so this can only assert
        # falsiness -- paired with a truthy case, so the pair still fails
        # if the method ever degenerates to a constant.
        enable = fs.ProcessedFilesTableWidget._should_enable_3d_view
        assert not enable(None, np.zeros((3, 8, 8), np.uint8))
        assert enable(None, np.zeros((3, 12, 8, 8), np.uint8)) is True

    def test_should_enable_3d_view_without_shape(self):
        assert (
            fs.ProcessedFilesTableWidget._should_enable_3d_view(None, object())
            is False
        )

    def test_label_dtype_image_layer_is_promoted(self):
        napari = pytest.importorskip("napari")
        viewer = _FakeViewer()
        image_layer = napari.layers.Image(np.zeros((4, 4), np.uint32))
        plain_layer = napari.layers.Image(np.zeros((4, 4), np.float32))
        viewer.layers.extend([image_layer, plain_layer])

        promoted = fs.ProcessedFilesTableWidget._promote_label_dtype_layers(
            SimpleNamespace(viewer=viewer), [image_layer, plain_layer]
        )

        assert promoted[1] is plain_layer
        assert promoted[0] is not image_layer, "uint32 layer was not promoted"
        # It went back in through add_labels (the fake viewer records the
        # adder by producing a _FakeLayer), carrying name/scale/translate.
        assert isinstance(promoted[0], _FakeLayer)
        np.testing.assert_array_equal(promoted[0].data, image_layer.data)
        assert promoted[0].name == image_layer.name
        np.testing.assert_array_equal(
            promoted[0].kwargs["scale"], image_layer.scale
        )
        np.testing.assert_array_equal(
            promoted[0].kwargs["translate"], image_layer.translate
        )
        # The swap happens in place: the replacement keeps the original's
        # stacking position rather than landing on top.
        assert viewer.layers.index(promoted[0]) == 0
        assert viewer.layers.moves == [(1, 0)]


class TestLoadOriginalImage:
    """_load_original_image's reader selection and layer bookkeeping."""

    def test_missing_file_sets_status(self, tmp_path, capsys):
        stub = _table_stub(_FakeViewer())
        stub._load_original_image(str(tmp_path / "gone.tif"))
        assert "File not found" in stub.viewer.status
        assert "does not exist" in capsys.readouterr().out

    def test_zarr_uses_viewer_open(self, tmp_path, monkeypatch):
        path = tmp_path / "src.zarr"
        path.mkdir()
        layer = _FakeLayer(np.zeros((12, 8, 8), np.uint8), name="L")
        viewer = _FakeViewer(open_result=[layer])
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        stub = _table_stub(viewer)

        stub._load_original_image(str(path))

        assert stub.current_original_images == [layer]
        assert viewer.dims.ndisplay == 3
        assert "Loaded 1 layers" in viewer.status

    def test_zarr_single_layer_result(self, tmp_path, monkeypatch):
        path = tmp_path / "src.zarr"
        path.mkdir()
        layer = _FakeLayer(np.zeros((4, 4), np.uint8), name="L")
        viewer = _FakeViewer(open_result=layer)
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        stub = _table_stub(viewer)
        viewer.dims.ndisplay = 2

        stub._load_original_image(str(path))
        assert stub.current_original_images == [layer]
        assert viewer.status == "Loaded 1 layers from src.zarr"
        # ndisplay is only ever *raised* to 3 by this module, so on its own
        # "still 2" would just be reading the fixture back; it earns its
        # keep next to the status line, as the 2-D no-op case.
        assert viewer.dims.ndisplay == 2

    def test_zarr_open_failure_falls_back(self, tmp_path, monkeypatch, capsys):
        pytest.importorskip("ome_zarr")
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = str(tmp_path / "src.zarr")
        fs.save_as_zarr(data, path, axes="YX")
        viewer = _FakeViewer(open_error=ValueError("plugin broke"))
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr(fs, "load_image_file", lambda p, **k: data)
        stub = _table_stub(viewer)

        stub._load_original_image(path)
        assert "napari-ome-zarr failed" in capsys.readouterr().out
        assert len(stub.current_original_images) == 1

    def test_zarr_open_returns_nothing_falls_back(
        self, tmp_path, monkeypatch, capsys
    ):
        path = tmp_path / "src.zarr"
        path.mkdir()
        viewer = _FakeViewer(open_result=[])
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: np.zeros((4, 4), np.uint8)
        )
        stub = _table_stub(viewer)
        stub._load_original_image(str(path))
        assert "falling back to manual loading" in capsys.readouterr().out
        assert len(stub.current_original_images) == 1

    def test_tiff_multichannel_uses_channel_axis(self, tmp_path):
        data = np.zeros((3, 8, 8), np.uint8)
        path = _write_tif(tmp_path / "c.tif", data)
        viewer = _FakeViewer()
        stub = _table_stub(viewer)

        stub._load_original_image(path)
        assert len(stub.current_original_images) == 3
        assert "Loaded 3 channels" in viewer.status

    def test_tiff_label_dtype_becomes_labels_layer(self, tmp_path):
        # Filename deliberately has no "label" substring: tiff_reader_function
        # independently guesses layer_type="labels" from a "label" in the
        # filename, which would make this pass even with the dtype-based
        # promotion in _load_original_image (`is_label_image(data)`) deleted.
        # Only the dtype can be driving the result here.
        data = np.arange(9, dtype=np.uint32).reshape(3, 3)
        path = _write_tif(tmp_path / "seg.tif", data)
        viewer = _FakeViewer()
        stub = _table_stub(viewer)

        stub._load_original_image(path)
        assert len(stub.current_original_images) == 1
        assert stub.current_original_images[0].name.startswith("Labels1:")

    def test_tiff_plain_2d_gets_colormap_and_name(self, tmp_path):
        path = _write_tif(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
        viewer = _FakeViewer()
        stub = _table_stub(viewer)

        stub._load_original_image(path)
        layer = stub.current_original_images[0]
        assert layer.name == "C1: a.tif"
        assert layer.kwargs["colormap"] == "red"
        assert layer.kwargs["blending"] == "additive"

    def test_reader_failure_falls_back_to_loader(
        self, tmp_path, monkeypatch, capsys
    ):
        data = np.arange(16, dtype=np.uint8).reshape(4, 4)
        path = _write_tif(tmp_path / "a.tif", data)

        def _boom(_p):
            raise RuntimeError("reader down")

        monkeypatch.setattr(fs, "tiff_reader_function", _boom)
        viewer = _FakeViewer()
        stub = _table_stub(viewer)
        stub._load_original_image(path)

        assert "Scale-aware TIFF reader failed" in capsys.readouterr().out
        assert stub.current_original_images[0].name == "Original: a.tif"

    def test_single_label_array_is_cast_to_uint32(self, tmp_path, monkeypatch):
        path = str(tmp_path / "x.png")
        with open(path, "wb") as fh:
            fh.write(b"stub")
        monkeypatch.setattr(
            fs,
            "load_image_file",
            lambda p, **k: np.arange(16, dtype=np.int64).reshape(4, 4),
        )
        viewer = _FakeViewer()
        stub = _table_stub(viewer)
        stub._load_original_image(path)

        layer = stub.current_original_images[0]
        assert layer.name == "Labels: x.png"
        assert layer.data.dtype == np.uint32

    def test_loader_exception_sets_error_status(self, tmp_path, monkeypatch):
        path = str(tmp_path / "x.png")
        with open(path, "wb") as fh:
            fh.write(b"stub")

        def _boom(_p, **_k):
            raise OSError("unreadable")

        monkeypatch.setattr(fs, "load_image_file", _boom)
        stub = _table_stub(_FakeViewer())
        stub._load_original_image(path)
        assert "Error processing" in stub.viewer.status

    def test_legacy_load_image_delegates(self, tmp_path):
        path = _write_tif(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
        stub = _table_stub(_FakeViewer())
        fs.ProcessedFilesTableWidget._load_image(stub, path)
        # Same layer _load_original_image builds for this file, so the
        # delegation is what is being observed, not merely "something ran".
        assert len(stub.current_original_images) == 1
        assert stub.current_original_images[0].name == "C1: a.tif"
        assert stub.viewer.status == "Loaded 1 channels from a.tif"


class TestLoadProcessedImage:
    """_load_processed_image adds a 'Processed' layer and lifts it to the top."""

    def test_missing_file_sets_status(self, tmp_path):
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(str(tmp_path / "gone.tif"))
        assert "File not found" in stub.viewer.status

    def test_npy_points_become_a_points_layer(self, tmp_path):
        coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        path = str(tmp_path / "spots.npy")
        np.save(path, coords)
        viewer = _FakeViewer()
        stub = _table_stub(viewer)

        stub._load_processed_image(path)
        assert len(viewer.added_points) == 1
        assert viewer.added_points[0].kwargs["symbol"] == "ring"
        assert "out_of_slice_display" not in viewer.added_points[0].kwargs
        assert "Loaded 2 spots" in viewer.status

    def test_npy_3d_points_enable_out_of_slice(self, tmp_path):
        coords = np.zeros((2, 3), dtype=np.float64)
        path = str(tmp_path / "spots3d.npy")
        np.save(path, coords)
        viewer = _FakeViewer()
        stub = _table_stub(viewer)

        stub._load_processed_image(path)
        points = viewer.added_points[0]
        assert points.kwargs["out_of_slice_display"] is True
        assert points.kwargs["symbol"] == "ring"
        np.testing.assert_array_equal(points.data, coords)
        assert "Loaded 2 spots" in viewer.status

    def test_npy_without_coordinates_falls_through(self, tmp_path, capsys):
        path = str(tmp_path / "img.npy")
        np.save(path, np.zeros((8, 8), np.uint8))
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(path)
        out = capsys.readouterr().out
        assert "doesn't contain points data" in out

    def test_zarr_reader_kwargs_are_sanitised(self, tmp_path, monkeypatch):
        path = tmp_path / "p.zarr"
        path.mkdir()
        data_levels = [np.zeros((4, 4), np.uint8), np.zeros((2, 2), np.uint8)]
        payload = [
            (data_levels, {"colormap": [], "multiscale": True}, "image")
        ]
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        # _load_processed_image imports the reader from napari_ome_zarr
        # directly, so that is the name that has to be replaced.
        monkeypatch.setattr(
            "napari_ome_zarr.napari_get_reader",
            lambda p: (lambda q: payload),
        )
        viewer = _FakeViewer()
        stub = _table_stub(viewer)

        stub._load_processed_image(str(path))

        layer = stub.current_processed_images[0]
        assert layer.name.startswith("Processed")
        assert layer.kwargs["multiscale"] is False
        assert layer.kwargs["interpolation3d"] == "nearest"
        assert "colormap" not in layer.kwargs
        assert viewer.reset_view_calls == 1

    def test_zarr_without_reader_uses_viewer_open(self, tmp_path, monkeypatch):
        path = tmp_path / "p.zarr"
        path.mkdir()
        layer = _FakeLayer(np.zeros((12, 8, 8), np.uint8), name="L")
        viewer = _FakeViewer(open_result=[layer])
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)
        monkeypatch.setattr("napari_ome_zarr.napari_get_reader", lambda p: None)
        stub = _table_stub(viewer)

        stub._load_processed_image(str(path))
        assert layer.name == "Processed L"
        assert viewer.dims.ndisplay == 3

    def test_zarr_reader_failure_falls_back(
        self, tmp_path, monkeypatch, capsys
    ):
        path = tmp_path / "p.zarr"
        path.mkdir()
        monkeypatch.setattr(fs, "OME_ZARR_AVAILABLE", True)

        def _boom(_p):
            raise ValueError("no reader")

        monkeypatch.setattr("napari_ome_zarr.napari_get_reader", _boom)
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: np.zeros((4, 4), np.uint8)
        )
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(str(path))
        assert "falling back" in capsys.readouterr().out
        assert len(stub.current_processed_images) == 1

    def test_tiff_multichannel_split(self, tmp_path):
        path = _write_tif(tmp_path / "c.tif", np.zeros((3, 8, 8), np.uint8))
        viewer = _FakeViewer()
        stub = _table_stub(viewer)
        stub._load_processed_image(path)
        assert len(stub.current_processed_images) == 3
        assert "processed channels" in viewer.status

    def test_tiff_labels_are_named_processed(self, tmp_path):
        path = _write_tif(
            tmp_path / "l.tif", np.arange(9, dtype=np.uint32).reshape(3, 3)
        )
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(path)
        name = stub.current_processed_images[0].name
        assert name == "Processed Labels1: l.tif"

    def test_existing_names_get_processed_prefix(self, tmp_path, monkeypatch):
        payload = [(np.zeros((8, 8), np.uint8), {"name": "raw"}, "image")]
        monkeypatch.setattr(fs, "tiff_reader_function", lambda p: payload)
        path = _write_tif(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(path)
        assert stub.current_processed_images[0].name == "Processed raw"

    def test_single_array_is_moved_to_top(self, tmp_path, monkeypatch):
        path = str(tmp_path / "x.png")
        with open(path, "wb") as fh:
            fh.write(b"stub")
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: np.zeros((4, 4), np.float32)
        )
        viewer = _FakeViewer(prepend=True)
        viewer.layers.append(_FakeLayer(np.zeros((2, 2)), name="below"))
        stub = _table_stub(viewer)

        stub._load_processed_image(path)
        # The new layer lands at index 0 and must be lifted to the top.
        assert viewer.layers.moves == [(0, 1)]
        assert viewer.layers[-1].name == "Processed: x.png"
        assert "moved to top layer" in viewer.status

    def test_float_labels_are_cast(self, tmp_path, monkeypatch):
        # is_label_image() only ever recognizes the four integer label
        # dtypes (see TestSmallHelpers), so a *float* array can only reach
        # this branch if something upstream already decided it is a label
        # image by some other means; force that decision here so the
        # `image.astype(np.uint32)` cast itself -- not just the layer
        # naming, which any label-dtype input would already produce -- is
        # actually exercised and checked.
        path = str(tmp_path / "x.png")
        with open(path, "wb") as fh:
            fh.write(b"stub")
        monkeypatch.setattr(
            fs,
            "load_image_file",
            lambda p, **k: np.arange(4, dtype=np.float64).reshape(2, 2),
        )
        monkeypatch.setattr(fs, "is_label_image", lambda img: True)
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(path)
        layer = stub.current_processed_images[0]
        assert layer.name.startswith("Processed Labels:")
        assert layer.data.dtype == np.uint32, "float array was not cast"

    def test_loader_exception_sets_error_status(self, tmp_path, monkeypatch):
        path = str(tmp_path / "x.png")
        with open(path, "wb") as fh:
            fh.write(b"stub")

        def _boom(_p, **_k):
            raise OSError("unreadable")

        monkeypatch.setattr(fs, "load_image_file", _boom)
        stub = _table_stub(_FakeViewer())
        stub._load_processed_image(path)
        assert "Error processing" in stub.viewer.status


# ---------------------------------------------------------------------------
# dimension order: probing, suggesting and warning
# ---------------------------------------------------------------------------
class _ComboStub:
    """The bit of QComboBox behaviour the dimension-order code relies on."""

    def __init__(self, text="Auto", items=None):
        self.items = list(items or fs.DIMENSION_ORDER_OPTIONS)
        self.text = text

    def currentText(self):
        return self.text

    def findText(self, value):
        return self.items.index(value) if value in self.items else -1

    def setCurrentIndex(self, index):
        self.text = self.items[index]


class _LabelStub:
    def __init__(self):
        self.text = ""
        self.style = ""

    def setText(self, value):
        self.text = value

    def setStyleSheet(self, value):
        self.style = value


def _dim_order_stub(file_list, selected="Auto", func=None):
    """A stand-in carrying just the widget state the methods below touch."""
    stub = SimpleNamespace(
        file_list=list(file_list),
        viewer=_FakeViewer(),
        dimension_order=_ComboStub(selected),
        dimension_order_status=_LabelStub(),
        processing_selector=None,
        _dimension_order_user_set=False,
        _probed_shape_cache={},
    )
    if func is not None:
        stub.processing_selector = SimpleNamespace(currentText=lambda: "F")
    for name in (
        "probed_shape",
        "update_dimension_order_status",
        "apply_dimension_order_suggestion",
        "_selected_function_uses_dimension_order",
        "_on_dimension_order_selected",
    ):
        setattr(
            stub,
            name,
            (
                lambda *a, _n=name: getattr(fs.FileResultsWidget, _n)(
                    stub, *a
                )
            ),
        )
    stub.confirm_dimension_order = (
        lambda info: fs.FileResultsWidget.confirm_dimension_order(stub, info)
    )
    return stub


class TestProbeFileShape:
    def test_reads_tiff_shape_without_loading(self, tmp_path):
        path = _write_tif(
            tmp_path / "s.tif", np.zeros((3, 4, 8, 8), np.uint8)
        )
        assert fs.probe_file_shape(path) == (3, 4, 8, 8)

    def test_unreadable_file_returns_none(self, tmp_path):
        path = tmp_path / "broken.tif"
        path.write_bytes(b"not a tiff")
        assert fs.probe_file_shape(str(path)) is None

    def test_empty_path_returns_none(self):
        assert fs.probe_file_shape("") is None

    def test_reads_zarr_array_shape(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        path = str(tmp_path / "a.zarr")
        arr = zarr.open(path, mode="w", shape=(2, 5, 6), dtype="uint8")
        arr[:] = 0
        assert fs.probe_file_shape(path) == (2, 5, 6)


class TestSuggestDimensionOrder:
    def test_trustworthy_metadata_is_used(self, tmp_path):
        path = _write_tif(
            tmp_path / "t.tif", np.zeros((2, 3, 8, 8), np.uint8), axes="TZYX"
        )
        assert fs.suggest_dimension_order(path) == "TZYX"

    def test_placeholder_axes_are_ignored(self, tmp_path):
        # tifffile stamps unknown axes as "Q"; a 4D file like that carries no
        # usable information, so the user must be asked rather than guessed at.
        path = _write_tif(tmp_path / "q.tif", np.zeros((2, 3, 8, 8), np.uint8))
        assert fs.probe_file_shape(path) == (2, 3, 8, 8)
        assert fs.suggest_dimension_order(path) is None

    def test_two_dimensional_file_is_unambiguous(self, tmp_path):
        path = _write_tif(tmp_path / "p.tif", np.zeros((8, 8), np.uint8))
        assert fs.suggest_dimension_order(path) == "YX"

    def test_axes_whose_rank_disagrees_with_shape_are_rejected(
        self, tmp_path, monkeypatch
    ):
        path = _write_tif(tmp_path / "m.tif", np.zeros((2, 8, 8), np.uint8))
        monkeypatch.setattr(fs, "detect_axes_for_file", lambda p, **k: "TZYX")
        assert fs.suggest_dimension_order(path) is None


class TestFunctionUsesDimensionOrder:
    def test_detects_both_parameter_spellings(self):
        assert fs.function_uses_dimension_order(
            lambda image, dimension_order="Auto": image
        )
        assert fs.function_uses_dimension_order(
            lambda image, dim_order="Auto": image
        )

    def test_plain_function_does_not_use_it(self):
        assert not fs.function_uses_dimension_order(lambda image: image)

    def test_uninspectable_object_is_not_assumed_to_use_it(self):
        assert not fs.function_uses_dimension_order(None)


class TestDimensionOrderStatus:
    def test_multidimensional_auto_warns(self, tmp_path):
        path = _write_tif(tmp_path / "a.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path])
        stub.update_dimension_order_status()
        assert "⚠" in stub.dimension_order_status.text
        assert "4D (2, 3, 8, 8)" in stub.dimension_order_status.text
        assert "TZYX" in stub.dimension_order_status.text
        assert "#ff6b00" in stub.dimension_order_status.style

    def test_two_dimensional_auto_is_fine(self, tmp_path):
        path = _write_tif(tmp_path / "b.tif", np.zeros((8, 8), np.uint8))
        stub = _dim_order_stub([path])
        stub.update_dimension_order_status()
        assert "⚠" not in stub.dimension_order_status.text
        assert "'Auto' is fine" in stub.dimension_order_status.text

    def test_explicit_order_is_spelled_out_per_axis(self, tmp_path):
        path = _write_tif(tmp_path / "c.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path], selected="TZYX")
        stub.update_dimension_order_status()
        assert "T=2" in stub.dimension_order_status.text
        assert "X=8" in stub.dimension_order_status.text
        assert "⚠" not in stub.dimension_order_status.text

    def test_order_with_wrong_rank_warns(self, tmp_path):
        path = _write_tif(tmp_path / "d.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path], selected="ZYX")
        stub.update_dimension_order_status()
        assert "⚠" in stub.dimension_order_status.text
        assert "3 axes" in stub.dimension_order_status.text

    def test_unprobeable_files_still_nudge(self, tmp_path):
        stub = _dim_order_stub([str(tmp_path / "missing.tif")])
        stub.update_dimension_order_status()
        assert "Auto" in stub.dimension_order_status.text

    def test_suggestion_is_applied_but_never_overrides_the_user(
        self, tmp_path
    ):
        path = _write_tif(
            tmp_path / "e.tif", np.zeros((2, 3, 8, 8), np.uint8), axes="TZYX"
        )
        stub = _dim_order_stub([path])
        stub.apply_dimension_order_suggestion()
        assert stub.dimension_order.currentText() == "TZYX"

        stub.dimension_order.text = "ZCYX"
        stub._on_dimension_order_selected(0)
        stub.apply_dimension_order_suggestion()
        assert stub.dimension_order.currentText() == "ZCYX"

    def test_probe_is_cached_per_file(self, tmp_path, monkeypatch):
        path = _write_tif(tmp_path / "f.tif", np.zeros((8, 8), np.uint8))
        stub = _dim_order_stub([path])
        assert stub.probed_shape() == (8, 8)
        monkeypatch.setattr(
            fs,
            "probe_file_shape",
            lambda p: pytest.fail("shape was re-read from disk"),
        )
        assert stub.probed_shape() == (8, 8)


class TestConfirmDimensionOrder:
    """The pre-flight guard that catches a forgotten dimension order."""

    @staticmethod
    def _stub_message_box(monkeypatch, answer):
        calls = []

        class _MessageBox:
            Cancel = 1
            Ignore = 2

            @staticmethod
            def warning(*args, **kwargs):
                calls.append(args)
                return answer

        monkeypatch.setattr(fs, "QMessageBox", _MessageBox)
        return calls

    def test_auto_on_4d_data_asks_and_cancel_stops_the_batch(
        self, tmp_path, monkeypatch
    ):
        calls = self._stub_message_box(monkeypatch, 1)  # Cancel
        path = _write_tif(tmp_path / "a.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path])
        info = {"func": lambda image, dimension_order="Auto": image}
        assert stub.confirm_dimension_order(info) is False
        assert len(calls) == 1

    def test_ignoring_the_warning_runs_anyway(self, tmp_path, monkeypatch):
        self._stub_message_box(monkeypatch, 2)  # Ignore
        path = _write_tif(tmp_path / "b.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path])
        info = {"func": lambda image, dimension_order="Auto": image}
        assert stub.confirm_dimension_order(info) is True

    def test_explicit_order_never_prompts(self, tmp_path, monkeypatch):
        calls = self._stub_message_box(monkeypatch, 1)
        path = _write_tif(tmp_path / "c.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path], selected="TZYX")
        info = {"func": lambda image, dimension_order="Auto": image}
        assert stub.confirm_dimension_order(info) is True
        assert calls == []

    def test_function_that_ignores_the_order_never_prompts(
        self, tmp_path, monkeypatch
    ):
        calls = self._stub_message_box(monkeypatch, 1)
        path = _write_tif(tmp_path / "d.tif", np.zeros((2, 3, 8, 8), np.uint8))
        stub = _dim_order_stub([path])
        assert stub.confirm_dimension_order({"func": lambda image: image})
        assert calls == []

    def test_two_dimensional_data_never_prompts(self, tmp_path, monkeypatch):
        calls = self._stub_message_box(monkeypatch, 1)
        path = _write_tif(tmp_path / "e.tif", np.zeros((8, 8), np.uint8))
        stub = _dim_order_stub([path])
        info = {"func": lambda image, dimension_order="Auto": image}
        assert stub.confirm_dimension_order(info) is True
        assert calls == []

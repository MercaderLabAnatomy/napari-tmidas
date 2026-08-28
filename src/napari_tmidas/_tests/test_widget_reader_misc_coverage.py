"""
Coverage for the small, long-tail modules of the plugin.

Nine modules are exercised here: the napari-contract stubs (``_writer``,
``_sample_data``, ``_widget``), the package's lazy ``__init__`` surface, the
TIFF/npy reader, the browse-button helper, and the guard branches left
uncovered in ``_frame_removal``, ``grid_view_overlay`` and ``scipy_filters``.

Two shared hazards are handled once, in ``_shared_module_state``:

* ``BatchProcessingRegistry._processing_functions`` is a process-wide dict.
* ``_reader._TIFF_HANDLES`` and ``grid_view_overlay``'s ``_grid_*`` globals
  are module-level caches.

Both are snapshotted and restored so this file passes on its own and as part
of the full suite.
"""

import importlib
import importlib.util
import os
import sys
import types

import numpy as np
import pytest
import tifffile

import napari_tmidas
import napari_tmidas._frame_removal as frame_removal
import napari_tmidas._reader as reader
import napari_tmidas._ui_utils as ui_utils
import napari_tmidas._widget as widget
import napari_tmidas.processing_functions.grid_view_overlay as grid_mod
import napari_tmidas.processing_functions.scipy_filters as scipy_mod
from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas._sample_data import make_sample_data
from napari_tmidas._writer import write_multiple, write_single_image

pytest.importorskip("pytestqt", reason="Qt widgets need pytest-qt's qapp")


# Qt widget tests segfault the macOS CI runner and take the whole session
# down. Same guard as test_label_based_cropping_widget.py / test_frame_removal.py.
requires_gui = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)


@pytest.fixture(autouse=True)
def _shared_module_state():
    """Protect the process-wide singletons these modules keep."""
    registry = dict(BatchProcessingRegistry._processing_functions)
    yield
    BatchProcessingRegistry._processing_functions.clear()
    BatchProcessingRegistry._processing_functions.update(registry)
    reader.invalidate_tiff_cache()
    grid_mod.reset_grid_cache()


def _exec_module_without(module, probe_name, blocked):
    """Re-run ``module``'s source with ``blocked`` imports failing.

    Every one of these modules wraps its optional imports in
    ``try/except ImportError`` and defines stand-ins in the handler; the
    handlers are unreachable in an environment where the dependency is
    installed. Executing the same file under a throwaway module name with
    ``None`` planted in ``sys.modules`` (which makes ``import x`` raise
    ``ImportError``) runs the fallback for real.
    """
    saved = {name: sys.modules.get(name, ...) for name in blocked}
    for name in blocked:
        sys.modules[name] = None
    try:
        spec = importlib.util.spec_from_file_location(
            probe_name, module.__file__
        )
        probe = importlib.util.module_from_spec(spec)
        sys.modules[probe_name] = probe
        spec.loader.exec_module(probe)
        return probe
    finally:
        for name, value in saved.items():
            if value is ...:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        sys.modules.pop(probe_name, None)


class TestWriterContract:
    """The napari writer contribution returns the paths it was given."""

    def test_write_single_image_returns_the_path(self, tmp_path):
        target = str(tmp_path / "image.tif")
        assert write_single_image(target, np.zeros((2, 2)), {"name": "x"}) == [
            target
        ]

    def test_write_multiple_returns_the_path(self, tmp_path):
        target = str(tmp_path / "layers.tif")
        layers = [
            (np.zeros((2, 2)), {"name": "a"}, "image"),
            (np.ones((2, 2), dtype=np.uint8), {"name": "b"}, "labels"),
        ]
        assert write_multiple(target, layers) == [target]


class TestSampleData:
    """The sample-data provider yields one add_image-ready layer tuple."""

    def test_returns_a_single_float_image_layer(self):
        sample = make_sample_data()

        assert len(sample) == 1
        data, add_kwargs = sample[0]
        assert data.shape == (512, 512)
        assert np.issubdtype(data.dtype, np.floating)
        assert data.min() >= 0.0 and data.max() < 1.0
        # It is noise, not a constant fill (the bounds above allow zeros).
        assert data.std() > 0.1
        assert add_kwargs == {}


class TestPackageSurface:
    """``napari_tmidas/__init__`` degrades to ``None`` per missing module."""

    def test_all_public_names_are_importable(self):
        for name in napari_tmidas.__all__:
            assert getattr(napari_tmidas, name) is not None, name

    def test_every_optional_import_falls_back_to_none(self):
        """
        Each export sits behind its own ``try/except ImportError`` so that a
        partial install (the Windows case the guards were written for) still
        yields an importable package. Planting empty stand-in modules makes
        every ``from ._x import y`` raise, which is what those handlers catch.
        """
        submodules = [
            "_version",
            "_file_selector",
            "_reader",
            "_sample_data",
            "_writer",
            "_label_inspection",
            "_roi_colocalization",
            "_crop_anything",
            "_frame_removal",
        ]
        saved = {}
        for short in submodules:
            full = f"napari_tmidas.{short}"
            saved[full] = sys.modules.get(full)
            sys.modules[full] = types.ModuleType(full)
        try:
            importlib.reload(napari_tmidas)
            assert napari_tmidas.__version__ == "unknown"
            missing = {
                name: getattr(napari_tmidas, name)
                for name in napari_tmidas.__all__
            }
            assert set(missing.values()) == {None}, missing
        finally:
            for full, module in saved.items():
                if module is None:
                    sys.modules.pop(full, None)
                else:
                    sys.modules[full] = module
            importlib.reload(napari_tmidas)

        # The package must be whole again for every later test.
        assert napari_tmidas.__version__ != "unknown"
        assert napari_tmidas.file_selector is not None


class TestReaderDispatch:
    """``napari_get_reader`` picks a loader from the path alone."""

    def test_npy_gets_the_numpy_reader(self, tmp_path):
        path = tmp_path / "stack.npy"
        np.save(path, np.zeros((2, 2)))
        assert reader.napari_get_reader(str(path)) is reader.reader_function

    def test_a_list_is_judged_by_its_first_entry(self, tmp_path):
        paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.npy")]
        assert reader.napari_get_reader(paths) is reader.tiff_reader_function

    @pytest.mark.parametrize(
        "name", ["a.tif", "b.tiff", "C.TIF", "d.OME.TIFF"]
    )
    def test_tiff_extensions_are_case_insensitive(self, name):
        assert reader.napari_get_reader(name) is reader.tiff_reader_function

    @pytest.mark.parametrize("name", ["a.png", "b.zarr", "c.npz", "noext"])
    def test_unsupported_extensions_are_declined(self, name):
        assert reader.napari_get_reader(name) is None

    def test_npy_reader_stacks_and_squeezes_multiple_files(self, tmp_path):
        rng = np.random.default_rng(0)
        arrays = [rng.random((1, 4, 5)) for _ in range(3)]
        paths = []
        for i, array in enumerate(arrays):
            path = tmp_path / f"p{i}.npy"
            np.save(path, array)
            paths.append(str(path))

        ((data, add_kwargs, layer_type),) = reader.reader_function(paths)

        # (3, 1, 4, 5) with the length-1 axis squeezed away.
        assert data.shape == (3, 4, 5)
        assert add_kwargs == {}
        assert layer_type == "image"
        np.testing.assert_allclose(data[2], arrays[2][0])


def _write_tiff(path, array, **kwargs):
    kwargs.setdefault("photometric", "minisblack")
    tifffile.imwrite(str(path), array, **kwargs)
    return path


class TestTiffHandleCache:
    """``_get_cached_tiff`` reuses handles but never serves a stale inode."""

    def test_repeated_reads_share_one_handle(self, tmp_path):
        path = str(_write_tiff(tmp_path / "a.tif", np.zeros((4, 4), np.uint8)))

        first = reader._get_cached_tiff(path)
        second = reader._get_cached_tiff(path)

        assert first is second
        # tifffile's default is a NullContext, which is also "not False";
        # what the reader installs is a real reentrant lock so the per-page
        # dask tasks serialise their seeks.
        assert hasattr(first.filehandle.lock, "acquire")
        assert hasattr(first.filehandle.lock, "release")

    def test_overwriting_the_file_reopens_it(self, tmp_path):
        path = tmp_path / "a.tif"
        _write_tiff(path, np.zeros((4, 4), np.uint8))
        first = reader._get_cached_tiff(str(path))

        # A different size guarantees a different identity signature.
        _write_tiff(path, np.ones((16, 16), np.uint8))
        second = reader._get_cached_tiff(str(path))

        assert second is not first
        assert first.filehandle.closed
        assert second.series[0].shape == (16, 16)

    def test_the_cache_evicts_the_least_recently_used_handle(self, tmp_path):
        paths = [
            str(
                _write_tiff(tmp_path / f"f{i}.tif", np.zeros((2, 2), np.uint8))
            )
            for i in range(reader._TIFF_HANDLES_MAX + 1)
        ]
        oldest = reader._get_cached_tiff(paths[0])
        for path in paths[1:]:
            reader._get_cached_tiff(path)

        assert len(reader._TIFF_HANDLES) == reader._TIFF_HANDLES_MAX
        assert paths[0] not in reader._TIFF_HANDLES
        assert oldest.filehandle.closed

    def test_invalidate_closes_one_path(self, tmp_path):
        a = str(_write_tiff(tmp_path / "a.tif", np.zeros((2, 2), np.uint8)))
        b = str(_write_tiff(tmp_path / "b.tif", np.zeros((2, 2), np.uint8)))
        handle_a = reader._get_cached_tiff(a)
        reader._get_cached_tiff(b)

        reader.invalidate_tiff_cache(a)

        assert handle_a.filehandle.closed
        assert a not in reader._TIFF_HANDLES
        assert b in reader._TIFF_HANDLES

    def test_invalidate_without_a_path_closes_everything(self, tmp_path):
        for i in range(3):
            reader._get_cached_tiff(
                str(
                    _write_tiff(tmp_path / f"f{i}.tif", np.zeros((2, 2), "u1"))
                )
            )

        reader.invalidate_tiff_cache()

        assert len(reader._TIFF_HANDLES) == 0

    def test_invalidating_an_unknown_path_is_a_no_op(self, tmp_path):
        kept = str(_write_tiff(tmp_path / "kept.tif", np.zeros((2, 2), "u1")))
        handle = reader._get_cached_tiff(kept)

        reader.invalidate_tiff_cache(str(tmp_path / "never-opened.tif"))

        assert list(reader._TIFF_HANDLES) == [kept]
        assert handle.filehandle.closed is False


class TestChannelAxisDetection:
    """The channel axis comes from the series' axes string."""

    def test_c_axis_is_located(self, tmp_path):
        path = _write_tiff(
            tmp_path / "tcyx.tif",
            np.zeros((3, 2, 8, 8), np.uint8),
            metadata={"axes": "TCYX"},
        )
        assert reader.detect_channel_axis_from_tiff_path(str(path)) == 1

    def test_no_c_axis_returns_none(self, tmp_path):
        path = _write_tiff(
            tmp_path / "zyx.tif",
            np.zeros((3, 8, 8), np.uint8),
            metadata={"axes": "ZYX"},
        )
        assert reader.detect_channel_axis_from_tiff_path(str(path)) is None

    def test_an_unreadable_path_returns_none(self, tmp_path):
        assert (
            reader.detect_channel_axis_from_tiff_path(
                str(tmp_path / "missing.tif")
            )
            is None
        )


class TestOmeScale:
    """OME PhysicalSize metadata becomes a per-axis napari ``scale``."""

    def test_physical_sizes_map_onto_the_axes(self, tmp_path):
        path = _write_tiff(
            tmp_path / "vol.ome.tif",
            np.zeros((2, 3, 8, 8), np.uint8),
            metadata={
                "axes": "TZYX",
                "PhysicalSizeX": 0.65,
                "PhysicalSizeY": 0.65,
                "PhysicalSizeZ": 2.0,
            },
        )
        tf = reader._get_cached_tiff(str(path))

        # T has no physical size, so it stays 1.0 rather than vetoing scale.
        assert reader._ome_scale_for_series(tf, "TZYX") == (
            1.0,
            2.0,
            0.65,
            0.65,
        )

    def test_a_plain_tiff_has_no_scale(self, tmp_path):
        path = _write_tiff(tmp_path / "plain.tif", np.zeros((4, 8, 8), "u1"))
        tf = reader._get_cached_tiff(str(path))

        assert tf.is_ome is False
        assert reader._ome_scale_for_series(tf, tf.series[0].axes) is None

    def test_ome_without_any_physical_size_returns_none(self, tmp_path):
        path = _write_tiff(
            tmp_path / "nosize.ome.tif",
            np.zeros((3, 8, 8), np.uint8),
            metadata={"axes": "ZYX"},
        )
        tf = reader._get_cached_tiff(str(path))

        assert tf.is_ome is True
        assert reader._ome_scale_for_series(tf, "ZYX") is None

    def test_unparsable_ome_xml_returns_none(self):
        broken = types.SimpleNamespace(is_ome=True, ome_metadata="<not-ome>")
        assert reader._ome_scale_for_series(broken, "YX") is None

    def test_a_list_of_pixels_uses_the_first_entry(self, monkeypatch):
        monkeypatch.setattr(
            tifffile,
            "xml2dict",
            lambda _xml: {
                "OME": {
                    "Image": {
                        "Pixels": [
                            {"PhysicalSizeX": 3.0, "PhysicalSizeY": 4.0},
                            {"PhysicalSizeX": 9.0},
                        ]
                    }
                }
            },
        )
        stub = types.SimpleNamespace(is_ome=True, ome_metadata="<ome/>")

        assert reader._ome_scale_for_series(stub, "YX") == (4.0, 3.0)

    def test_a_non_mapping_pixels_entry_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            tifffile,
            "xml2dict",
            lambda _xml: {"OME": {"Image": {"Pixels": "unexpected"}}},
        )
        stub = types.SimpleNamespace(is_ome=True, ome_metadata="<ome/>")

        assert reader._ome_scale_for_series(stub, "YX") is None


class TestTiffReaderFunction:
    """The TIFF reader stays lazy and keeps the series' n-d shape."""

    def test_multipage_stacks_load_lazily_page_by_page(self, tmp_path):
        data = (
            np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8) % 900
        )
        path = _write_tiff(tmp_path / "vol.tif", data)

        ((array, add_kwargs, layer_type),) = reader.tiff_reader_function(
            str(path)
        )

        assert hasattr(array, "compute"), "expected a lazy dask array"
        assert array.shape == data.shape
        assert array.dtype == data.dtype
        assert add_kwargs == {}
        assert layer_type == "image"
        # Force the per-page tasks on this thread so the page reader runs.
        np.testing.assert_array_equal(
            np.asarray(array.compute(scheduler="synchronous")), data
        )

    def test_a_single_plane_is_read_eagerly(self, tmp_path):
        data = np.arange(64, dtype=np.uint8).reshape(8, 8)
        path = _write_tiff(tmp_path / "plane.tif", data)

        ((array, _kwargs, layer_type),) = reader.tiff_reader_function(
            str(path)
        )

        assert isinstance(array, np.ndarray)
        np.testing.assert_array_equal(array, data)
        assert layer_type == "image"

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("cells_labels.tif", "labels"),
            ("CELLS_LABELS.tif", "labels"),
            ("cells.tif", "image"),
        ],
    )
    def test_label_files_are_recognised_by_name(
        self, tmp_path, name, expected
    ):
        path = _write_tiff(tmp_path / name, np.zeros((8, 8), np.uint8))

        ((_array, _kwargs, layer_type),) = reader.tiff_reader_function(
            str(path)
        )

        assert layer_type == expected

    def test_label_naming_also_applies_to_lazy_stacks(self, tmp_path):
        path = _write_tiff(
            tmp_path / "seg_labels.tif", np.zeros((3, 4, 8, 8), np.uint8)
        )

        ((array, _kwargs, layer_type),) = reader.tiff_reader_function(
            str(path)
        )

        assert hasattr(array, "compute")
        assert layer_type == "labels"

    def test_a_list_of_paths_yields_one_tuple_each(self, tmp_path):
        paths = [
            str(_write_tiff(tmp_path / f"s{i}.tif", np.full((8, 8), i, "u1")))
            for i in range(3)
        ]

        results = reader.tiff_reader_function(paths)

        assert len(results) == 3
        assert [int(np.asarray(r[0]).max()) for r in results] == [0, 1, 2]

    def test_ome_voxel_spacing_is_passed_through_as_scale(self, tmp_path):
        path = _write_tiff(
            tmp_path / "vol.ome.tif",
            np.zeros((4, 8, 8), np.uint8),
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeZ": 3.0,
            },
        )

        ((_array, add_kwargs, _type),) = reader.tiff_reader_function(str(path))

        assert add_kwargs == {"scale": (3.0, 0.5, 0.5)}


@requires_gui
class TestBrowseButton:
    """``add_browse_button_to_folder_field`` wires a real QPushButton."""

    class _Field:
        def __init__(self, value, layout):
            self.value = value
            self._layout = layout
            self.native = self

        def parent(self):
            return self

        def layout(self):
            return self._layout

    class _Layout:
        def __init__(self):
            self.widgets = []

        def addWidget(self, w):  # noqa: N802 - Qt naming
            self.widgets.append(w)

    def _build(self, qtbot, value):
        layout = self._Layout()
        field = self._Field(value, layout)
        holder = types.SimpleNamespace(folder_path=field)

        result = ui_utils.add_browse_button_to_folder_field(
            holder, "folder_path"
        )

        assert result is holder
        assert len(layout.widgets) == 1
        button = layout.widgets[0]
        qtbot.addWidget(button)
        return field, button

    def test_choosing_a_folder_updates_the_field(
        self, qtbot, monkeypatch, tmp_path
    ):
        field, button = self._build(qtbot, str(tmp_path))
        seen = {}

        class _Dialog:
            ShowDirsOnly = 1
            DontResolveSymlinks = 2

            @staticmethod
            def getExistingDirectory(parent, caption, start_dir, options):
                seen["start_dir"] = start_dir
                seen["caption"] = caption
                return str(tmp_path / "chosen")

        monkeypatch.setattr(ui_utils, "QFileDialog", _Dialog)
        button.click()

        assert field.value == str(tmp_path / "chosen")
        assert seen["start_dir"] == str(tmp_path)
        assert seen["caption"] == "Select Folder"

    def test_an_empty_field_starts_the_dialog_at_home(
        self, qtbot, monkeypatch
    ):
        field, button = self._build(qtbot, "")
        seen = {}

        class _Dialog:
            ShowDirsOnly = 1
            DontResolveSymlinks = 2

            @staticmethod
            def getExistingDirectory(parent, caption, start_dir, options):
                seen["start_dir"] = start_dir
                return ""

        monkeypatch.setattr(ui_utils, "QFileDialog", _Dialog)
        button.click()

        assert seen["start_dir"] == os.path.expanduser("~")
        # A cancelled dialog must leave the field alone.
        assert field.value == ""

    def test_it_survives_a_field_without_a_layout(self, qtbot):
        field = self._Field("", None)
        holder = types.SimpleNamespace(folder_path=field)

        assert (
            ui_utils.add_browse_button_to_folder_field(holder, "folder_path")
            is holder
        )

    def test_the_module_imports_without_qt(self):
        """The Qt import is guarded so the helper module stays importable."""
        probe = _exec_module_without(
            ui_utils, "_probe_ui_utils_no_qt", ["qtpy.QtWidgets"]
        )

        assert probe._HAS_QTPY is False
        assert probe.QFileDialog is None and probe.QPushButton is None


@requires_gui
class TestWidgetExamples:
    """The example widgets from the plugin template, driven headlessly."""

    def test_autogenerate_widget_thresholds_in_place(self):
        image = np.linspace(0, 1, 64).reshape(8, 8)
        image[0, 0] = 0.5  # exactly on the threshold: excluded, not included

        out = widget.threshold_autogenerate_widget(image, 0.5)

        assert out.dtype == bool
        assert out.shape == image.shape
        np.testing.assert_array_equal(out, image > 0.5)

    def test_magic_widget_thresholds_a_layer(self, make_napari_viewer):
        viewer = make_napari_viewer()
        data = np.linspace(0, 1, 100).reshape(10, 10)
        data[0, 0] = 0.25  # exactly on the threshold
        layer = viewer.add_image(data)

        gui = widget.threshold_magic_widget()
        out = gui(layer, 0.25)

        np.testing.assert_array_equal(out, data > 0.25)
        assert out[0, 0] == False  # noqa: E712 - the boundary is exclusive

    def test_image_threshold_container_adds_then_updates_a_layer(
        self, make_napari_viewer
    ):
        viewer = make_napari_viewer()
        data = np.linspace(0, 1, 100).reshape(10, 10)
        # Two samples sit exactly on the two thresholds used below, so a
        # `>=` comparison would show up as an extra True.
        data[0, 0] = 0.5
        data[0, 1] = 0.9
        layer = viewer.add_image(data, name="src")
        container = ImageThresholdHelper.build(viewer, layer, 0.5)

        container._threshold_im()

        assert "src_thresholded" in viewer.layers
        result = viewer.layers["src_thresholded"]
        np.testing.assert_array_equal(result.data, data > 0.5)

        # A second run must reuse the layer, not stack up copies.
        container._threshold_slider.value = 0.9
        container._threshold_im()
        assert len(viewer.layers) == 2
        np.testing.assert_array_equal(result.data, data > 0.9)

    def test_the_invert_checkbox_flips_the_comparison(
        self, make_napari_viewer
    ):
        viewer = make_napari_viewer()
        data = np.linspace(0, 1, 100).reshape(10, 10)
        data[0, 0] = 0.5  # exactly on the threshold: excluded either way
        layer = viewer.add_image(data, name="src")
        container = ImageThresholdHelper.build(viewer, layer, 0.5)
        container._invert_checkbox.value = True

        container._threshold_im()

        np.testing.assert_array_equal(
            viewer.layers["src_thresholded"].data, data < 0.5
        )

    def test_no_selected_layer_is_a_no_op(self, make_napari_viewer):
        viewer = make_napari_viewer()
        container = widget.ImageThreshold(viewer)
        # With no image layers in the viewer the combo has no value at all.
        assert container._image_layer_combo.value is None

        container._threshold_im()

        assert len(viewer.layers) == 0

    def test_example_qwidget_button_reports_the_layer_count(
        self, make_napari_viewer, capsys
    ):
        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((4, 4)))
        viewer.add_image(np.zeros((4, 4)))
        example = widget.ExampleQWidget(viewer)

        # The click has to travel through the layout the widget built.
        button = example.layout().itemAt(0).widget()
        button.click()

        assert capsys.readouterr().out == "napari has 2 layers\n"

    def test_without_its_optional_dependencies_it_still_imports(self):
        """
        magicgui, qtpy and skimage each sit behind an ImportError guard so a
        bare install can still import the module; the stubs those handlers
        install are what this pins.
        """
        probe = _exec_module_without(
            widget,
            "_probe_widget_no_deps",
            ["magicgui", "magicgui.widgets", "qtpy.QtWidgets", "skimage.util"],
        )

        assert probe._HAS_MAGICGUI is False
        assert probe._HAS_QTPY is False
        assert probe._HAS_SKIMAGE is False
        assert probe.img_as_float is None
        assert probe.CheckBox is None and probe.create_widget is None
        assert probe.QHBoxLayout is None and probe.QPushButton is None

        # The stub magic_factory is a no-op in both calling styles.
        def _f():
            return 1

        assert probe.magic_factory(_f) is _f
        assert probe.magic_factory(auto_call=True)(_f) is _f
        assert probe.threshold_magic_widget.__name__ == (
            "threshold_magic_widget"
        )


class ImageThresholdHelper:
    """Build an ``ImageThreshold`` already pointed at a layer."""

    @staticmethod
    def build(viewer, layer, threshold):
        container = widget.ImageThreshold(viewer)
        container._image_layer_combo.value = layer
        container._threshold_slider.value = threshold
        return container


@requires_gui
class TestFrameRemovalGuards:
    """The frame-removal widget's early returns and dialog branches."""

    @pytest.fixture()
    def loaded(self, make_napari_viewer, qtbot):
        viewer = make_napari_viewer()
        data = np.arange(6 * 4 * 4, dtype=np.uint8).reshape(6, 4, 4)
        layer = viewer.add_image(data, name="movie")
        panel = frame_removal.FrameRemovalWidget(viewer)
        qtbot.addWidget(panel)
        panel._on_layer_selected(layer)
        return panel

    def test_deselecting_the_layer_resets_the_widget(self, loaded):
        loaded._on_layer_selected(None)

        assert loaded.image_layer is None
        assert loaded.original_data is None
        assert loaded.frames_to_remove == []
        assert loaded.frame_slider.isEnabled() is False
        assert loaded.info_label.text() == "No image selected"

    def test_controls_are_inert_before_a_layer_is_chosen(
        self, make_napari_viewer, qtbot
    ):
        viewer = make_napari_viewer()
        panel = frame_removal.FrameRemovalWidget(viewer)
        qtbot.addWidget(panel)

        panel._on_slider_changed(4)
        panel._update_display()
        panel._on_mark_changed(2)
        panel._preview_result()

        assert panel.current_frame == 0
        assert panel.frames_to_remove == []
        assert panel.frame_label.text() == "Frame: 0 / 0"
        assert len(viewer.layers) == 0

    def test_create_cleaned_data_without_marks_returns_the_original(
        self, loaded
    ):
        assert loaded._create_cleaned_data() is loaded.original_data

    def test_clearing_with_nothing_marked_only_informs(
        self, loaded, monkeypatch
    ):
        shown = []
        monkeypatch.setattr(
            frame_removal.QMessageBox,
            "information",
            lambda *args, **kwargs: shown.append(args[1]),
        )
        monkeypatch.setattr(
            frame_removal.QMessageBox,
            "question",
            lambda *args, **kwargs: pytest.fail("must not ask to clear"),
        )

        loaded._clear_marks()

        assert shown == ["No Marks"]

    def test_clearing_marks_drops_the_preview_layer(self, loaded, monkeypatch):
        loaded.current_frame = 2
        loaded.frames_to_remove = [2]
        loaded._preview_result()
        assert "movie_cleaned_preview" in loaded.viewer.layers

        monkeypatch.setattr(
            frame_removal.QMessageBox,
            "question",
            lambda *args, **kwargs: frame_removal.QMessageBox.Yes,
        )
        loaded._clear_marks()

        assert loaded.frames_to_remove == []
        assert "movie_cleaned_preview" not in loaded.viewer.layers
        assert loaded.viewer.status == "All marked frames cleared"

    def test_preview_replaces_rather_than_stacks(self, loaded):
        loaded.current_frame = 1
        loaded.frames_to_remove = [1]

        loaded._preview_result()
        loaded.frames_to_remove = [1, 3]
        loaded._preview_result()

        previews = [
            layer
            for layer in loaded.viewer.layers
            if "_cleaned_preview" in layer.name
        ]
        assert len(previews) == 1
        assert previews[0].data.shape == (4, 4, 4)

    def test_saving_without_marks_never_opens_a_dialog(
        self, loaded, monkeypatch
    ):
        opened = []

        class _Dialog:
            @staticmethod
            def getSaveFileName(*args, **kwargs):  # noqa: N802 - Qt naming
                opened.append(args[2])
                return ("", "")

        monkeypatch.setattr(frame_removal, "QFileDialog", _Dialog)

        loaded._save_result()

        assert opened == []
        # Positive control: an empty mark list is the only thing holding the
        # dialog back, so the same call with one mark must reach it.
        loaded.frames_to_remove = [0]
        loaded._save_result()
        assert opened == ["movie_cleaned.tif"]

    def test_saving_without_tifffile_reports_the_missing_dependency(
        self, loaded, monkeypatch
    ):
        loaded.frames_to_remove = [0]
        critical = []
        monkeypatch.setattr(frame_removal, "_HAS_TIFFFILE", False)
        monkeypatch.setattr(
            frame_removal.QMessageBox,
            "critical",
            lambda *args, **kwargs: critical.append(args[1]),
        )
        monkeypatch.setattr(
            frame_removal,
            "QFileDialog",
            _fail_dialog("dependency check must come first"),
        )

        loaded._save_result()

        assert critical == ["Missing Dependency"]

    def test_a_cancelled_save_dialog_writes_nothing(
        self, loaded, monkeypatch, tmp_path
    ):
        loaded.frames_to_remove = [0]
        monkeypatch.setattr(
            frame_removal, "QFileDialog", _stub_dialog(("", ""))
        )
        monkeypatch.setattr(
            frame_removal,
            "tifffile",
            types.SimpleNamespace(
                imwrite=lambda *a, **k: pytest.fail("nothing to write")
            ),
        )
        _forbid_message_boxes(monkeypatch)
        status_before = loaded.viewer.status

        loaded._save_result()

        assert list(tmp_path.iterdir()) == []
        # The success path sets a "Saved ..." status; cancelling must not.
        assert loaded.viewer.status == status_before

    def test_a_failing_write_is_reported_not_raised(
        self, loaded, monkeypatch, tmp_path
    ):
        loaded.frames_to_remove = [0]
        target = str(tmp_path / "out.tif")
        monkeypatch.setattr(
            frame_removal, "QFileDialog", _stub_dialog((target, ""))
        )

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            frame_removal, "tifffile", types.SimpleNamespace(imwrite=_boom)
        )
        critical = []
        monkeypatch.setattr(
            frame_removal.QMessageBox,
            "critical",
            lambda *args, **kwargs: critical.append(args[2]),
        )
        monkeypatch.setattr(
            frame_removal.QMessageBox,
            "information",
            lambda *args, **kwargs: pytest.fail("the write failed"),
        )

        loaded._save_result()

        assert len(critical) == 1
        assert "disk full" in critical[0]
        assert not os.path.exists(target)

    def test_the_dock_widget_entry_point_builds_the_panel(
        self, make_napari_viewer
    ):
        viewer = make_napari_viewer()

        assert (
            frame_removal.frame_removal_widget()
            is frame_removal.frame_removal_tool
        )
        frame_removal.frame_removal_tool(viewer)

        assert "Frame Removal" in viewer.window.dock_widgets

    def test_without_its_optional_dependencies_it_still_imports(self):
        probe = _exec_module_without(
            frame_removal,
            "_probe_frame_removal_no_deps",
            [
                "magicgui",
                "magicgui.widgets",
                "napari.layers",
                "napari.viewer",
                "tifffile",
            ],
        )

        assert probe._HAS_MAGICGUI is False
        assert probe._HAS_NAPARI is False
        assert probe._HAS_TIFFFILE is False
        assert probe.tifffile is None
        assert probe.create_widget is None
        # The magicgui stub leaves the decorated function untouched, whether
        # it is applied bare or with keyword arguments.
        assert probe.frame_removal_widget() is probe.frame_removal_tool
        assert callable(probe.frame_removal_tool)

        def _f():
            return 1

        assert probe.magicgui(_f) is _f
        assert probe.magicgui(call_button="go")(_f) is _f


def _forbid_message_boxes(monkeypatch):
    """Make every modal box a loud failure instead of a silent hang."""
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(
            frame_removal.QMessageBox,
            name,
            lambda *a, _n=name, **k: pytest.fail(f"unexpected {_n} box"),
        )


def _stub_dialog(result):
    class _Dialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):  # noqa: N802 - Qt naming
            return result

    return _Dialog


def _fail_dialog(message):
    class _Dialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):  # noqa: N802 - Qt naming
            pytest.fail(message)

    return _Dialog


class _BatchCaller:
    """Stand-in for the batch worker frame ``create_grid_overlay`` inspects.

    The function recovers the file being processed, the batch's file list and
    the output folder from its caller's frame locals, so a test can only reach
    those branches through a caller that owns locals with those names.
    """

    def __init__(self, output_folder):
        self.output_folder = str(output_folder)

    def run(self, filepath, file_list, image, **kwargs):
        return grid_mod.create_grid_overlay(image, **kwargs)


def _run_with_filepath(filepath, image, **kwargs):
    """Call it with only a ``filepath`` local, as the simple path does."""
    return grid_mod.create_grid_overlay(image, **kwargs)


class TestGridOverlayBranches:
    """Guard branches of the grid-overlay batch function."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        grid_mod.reset_grid_cache()
        yield
        grid_mod.reset_grid_cache()

    @staticmethod
    def _pair(folder, stem, shape=(16, 16), labelled=True):
        rng = np.random.default_rng(len(stem))
        _write_tiff(
            folder / f"{stem}.tif", (rng.random(shape) * 255).astype(np.uint8)
        )
        labels = np.zeros(shape, dtype=np.uint16)
        if labelled:
            labels[2:5, 2:5] = 1
        _write_tiff(folder / f"{stem}_labels.tif", labels)

    def test_downsampling_keeps_the_channel_axis(self):
        rgb = np.zeros((40, 20, 3), dtype=np.uint8)
        rgb[:, :, 1] = 255

        out = grid_mod._downsample_image(rgb, 20)

        assert out.shape == (20, 10, 3)
        assert out.dtype == np.uint8

    def test_a_small_image_is_returned_untouched(self):
        small = np.zeros((8, 8), dtype=np.uint8)
        assert grid_mod._downsample_image(small, 32) is small

    def test_float_intensity_is_not_recast(self):
        # Deliberately outside 0..1: a float image is copied rather than
        # re-cast, then min-max normalised, so the ramp has to land on the
        # full 8-bit range.  A 0..1 ramp would make normalising a no-op and
        # the test blind to it.
        intensity = np.linspace(2.0, 6.0, 64, dtype=np.float32).reshape(8, 8)
        labels = np.zeros((8, 8), dtype=np.uint16)

        overlay = grid_mod._create_overlay(
            intensity, labels, show_overlay=False
        )

        assert overlay.shape == (8, 8, 3)
        assert overlay.dtype == np.uint8
        assert overlay[0, 0, 0] == 0
        assert overlay[-1, -1, 0] == 255
        assert intensity.dtype == np.float32  # input left untouched

    def test_an_empty_label_image_stays_grayscale(self):
        rng = np.random.default_rng(0)
        intensity = (rng.random((8, 8)) * 255).astype(np.uint8)
        labels = np.zeros((8, 8), dtype=np.uint16)

        overlay = grid_mod._create_overlay(
            intensity, labels, show_overlay=True
        )

        np.testing.assert_array_equal(overlay[:, :, 0], overlay[:, :, 1])
        np.testing.assert_array_equal(overlay[:, :, 1], overlay[:, :, 2])

    def test_the_grid_fills_row_by_row_and_pads_the_remainder(self):
        tiles = [np.full((4, 6, 3), i + 1, dtype=np.uint8) for i in range(5)]

        grid = grid_mod._create_grid(tiles, grid_cols=2)

        # 5 tiles over 2 columns is 3 rows; the sixth cell is padding.
        assert grid.shape == (3 * 4, 2 * 6, 3)
        assert grid.dtype == np.uint8
        cells = [
            np.unique(grid[r * 4 : (r + 1) * 4, c * 6 : (c + 1) * 6, 0])
            for r in range(3)
            for c in range(2)
        ]
        assert [cell.tolist() for cell in cells] == [
            [1],
            [2],
            [3],
            [4],
            [5],
            [0],
        ]

    def test_a_grid_of_plain_2d_tiles_has_no_channel_axis(self):
        tiles = [np.full((3, 3), i + 1, dtype=np.uint16) for i in range(4)]

        grid = grid_mod._create_grid(tiles, grid_cols=2)

        assert grid.shape == (6, 6)
        assert grid.dtype == np.uint16
        assert grid[0, 0] == 1 and grid[0, 3] == 2
        assert grid[3, 0] == 3 and grid[3, 3] == 4

    def test_an_empty_image_list_has_no_grid(self):
        assert grid_mod._create_grid([], grid_cols=4) is None

    def test_hiding_the_overlay_leaves_pure_grayscale(self):
        """``show_overlay=False`` must not tint a *labelled* image either."""
        rng = np.random.default_rng(1)
        intensity = (rng.random((8, 8)) * 255).astype(np.uint8)
        labels = np.zeros((8, 8), dtype=np.uint16)
        labels[2:6, 2:6] = 4

        hidden = grid_mod._create_overlay(
            intensity, labels, show_overlay=False
        )
        shown = grid_mod._create_overlay(intensity, labels, show_overlay=True)

        np.testing.assert_array_equal(hidden[:, :, 0], hidden[:, :, 1])
        np.testing.assert_array_equal(hidden[:, :, 1], hidden[:, :, 2])
        # With the overlay on, only the labelled block changes.
        assert not np.array_equal(shown[2:6, 2:6], hidden[2:6, 2:6])
        background = np.ones((8, 8), dtype=bool)
        background[2:6, 2:6] = False
        np.testing.assert_array_equal(shown[background], hidden[background])

    def test_the_batch_file_list_and_output_folder_are_honoured(
        self, tmp_path
    ):
        folder = tmp_path / "data"
        folder.mkdir()
        out = tmp_path / "results"
        for stem in ("a", "b"):
            self._pair(folder, stem)
        caller = _BatchCaller(out)

        grid = caller.run(
            filepath=str(folder / "a_labels.tif"),
            file_list=[str(folder / "a_labels.tif")],
            image=np.zeros((4, 4), dtype=np.uint8),
        )

        assert isinstance(grid, np.ndarray)
        saved = list(out.glob("*_grid_overlay.tif"))
        assert [p.name for p in saved] == ["a_labels_grid_overlay.tif"]
        # Only the one file from file_list was gridded, not b as well.
        assert grid.shape[:2] == (16, 16)

    def test_an_empty_folder_returns_the_input_image(self, tmp_path, capsys):
        folder = tmp_path / "data"
        folder.mkdir()
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)

        out = _run_with_filepath(str(folder / "a_labels.tif"), image)

        np.testing.assert_array_equal(out, image)
        # The glob found nothing at all -- a different guard from the one
        # that fires when files exist but are all filtered away.
        assert "No label files found in folder" in capsys.readouterr().out

    def test_a_folder_of_only_old_grids_returns_the_input_image(
        self, tmp_path, capsys
    ):
        folder = tmp_path / "data"
        folder.mkdir()
        _write_tiff(
            folder / "old_grid_overlay.tif", np.zeros((8, 8), np.uint8)
        )
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)

        out = _run_with_filepath(
            str(folder / "old_grid_overlay.tif"), image, label_suffix=""
        )

        np.testing.assert_array_equal(out, image)
        printed = capsys.readouterr().out
        assert "No valid label files found after filtering" in printed

    def test_unreadable_inputs_are_collected_and_summarised(
        self, tmp_path, capsys
    ):
        """
        More than ten skipped files must not print one line each, and a batch
        where every file fails hands the pipeline its image back.
        """
        folder = tmp_path / "data"
        folder.mkdir()
        for i in range(12):
            self._pair(folder, f"s{i:02d}")
            (folder / f"s{i:02d}.tif").unlink()  # orphan every label file
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)

        out = _run_with_filepath(str(folder / "s00_labels.tif"), image)

        np.testing.assert_array_equal(out, image)
        printed = capsys.readouterr().out
        assert "12 files skipped" in printed
        assert "... and 2 more" in printed
        assert "No valid image pairs found" in printed

    def test_an_unreadable_intensity_file_is_skipped(self, tmp_path, capsys):
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "good")
        labels = np.zeros((16, 16), dtype=np.uint16)
        _write_tiff(folder / "bad_labels.tif", labels)
        (folder / "bad.tif").mkdir()  # unreadable stand-in for the intensity

        grid = _run_with_filepath(
            str(folder / "good_labels.tif"), np.zeros((4, 4), np.uint8)
        )

        assert isinstance(grid, np.ndarray)
        assert "Error processing bad_labels.tif" in capsys.readouterr().out

    def test_a_batch_without_any_labels_warns(self, tmp_path, capsys):
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a", labelled=False)

        grid = _run_with_filepath(
            str(folder / "a_labels.tif"), np.zeros((4, 4), np.uint8)
        )

        assert isinstance(grid, np.ndarray)
        assert "No labels detected in any image" in capsys.readouterr().out

    def test_a_failed_grid_assembly_returns_the_input(
        self, tmp_path, monkeypatch, capsys
    ):
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a")
        monkeypatch.setattr(grid_mod, "_create_grid", lambda *a, **k: None)
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)

        out = _run_with_filepath(str(folder / "a_labels.tif"), image)

        np.testing.assert_array_equal(out, image)
        assert "Grid creation returned None" in capsys.readouterr().out

    def test_a_silently_missing_output_file_is_flagged(
        self, tmp_path, monkeypatch, capsys
    ):
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a")
        monkeypatch.setattr(grid_mod.tifffile, "imwrite", lambda *a, **k: None)

        grid = _run_with_filepath(
            str(folder / "a_labels.tif"), np.zeros((4, 4), np.uint8)
        )

        assert isinstance(grid, np.ndarray)
        assert "file not found at" in capsys.readouterr().out

    def test_a_write_error_does_not_lose_the_grid(
        self, tmp_path, monkeypatch, capsys
    ):
        folder = tmp_path / "data"
        folder.mkdir()
        self._pair(folder, "a")

        def _boom(*args, **kwargs):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(grid_mod.tifffile, "imwrite", _boom)

        grid = _run_with_filepath(
            str(folder / "a_labels.tif"), np.zeros((4, 4), np.uint8)
        )

        # The caller still receives the assembled grid.
        assert isinstance(grid, np.ndarray)
        printed = capsys.readouterr().out
        assert "Error saving grid: OSError: read-only filesystem" in printed

    def test_without_tifffile_the_input_is_returned(self, tmp_path, capsys):
        """Reading and writing the grid both need tifffile; without it the
        batch function must bow out instead of raising."""
        probe = _exec_module_without(
            grid_mod, "_probe_grid_no_tifffile", ["tifffile"]
        )
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)

        out = probe.create_grid_overlay(image)

        assert probe._HAS_TIFFFILE is False
        assert probe.tifffile is None
        np.testing.assert_array_equal(out, image)
        assert "tifffile not available" in capsys.readouterr().out


class TestScipyFilterBranches:
    """Dimension-order and dtype branches of the SciPy processing helpers."""

    def test_resize_by_one_copies_without_zooming(self):
        labels = np.zeros((8, 8), dtype=np.uint16)
        labels[2:5, 2:5] = 7

        out = scipy_mod.resize_labels(labels, scale_factor=1.0)

        assert out is not labels
        np.testing.assert_array_equal(out, labels)

    def test_upscaling_crops_back_to_the_original_shape(self):
        labels = np.zeros((16, 16), dtype=np.uint16)
        labels[6:10, 6:10] = 3

        out = scipy_mod.resize_labels(labels, scale_factor=1.5)

        assert out.shape == labels.shape
        assert out.dtype == labels.dtype
        # The object grew, so it covers more pixels than before.
        assert (out == 3).sum() > (labels == 3).sum()
        # ...and it grew about the centre rather than off one corner.
        rows, cols = np.nonzero(out == 3)
        assert rows.min() + rows.max() == labels.shape[0] - 1
        assert cols.min() + cols.max() == labels.shape[1] - 1

    def test_downscaling_keeps_the_object_centred(self):
        labels = np.zeros((16, 16), dtype=np.uint16)
        labels[4:12, 4:12] = 2

        out = scipy_mod.resize_labels(labels, scale_factor=0.5)

        assert out.shape == labels.shape
        assert 0 < (out == 2).sum() < (labels == 2).sum()
        # Shrunk about the array centre: the bounding box stays symmetric.
        rows, cols = np.nonzero(out == 2)
        assert rows.min() + rows.max() == labels.shape[0] - 1
        assert cols.min() + cols.max() == labels.shape[1] - 1

    def test_a_float_label_image_is_promoted_to_uint32(self):
        labels = np.zeros((12, 12), dtype=np.float32)
        labels[3:9, 3:9] = 1.0

        out = scipy_mod.subdivide_labels_3layers(labels)

        assert out.dtype == np.uint32
        # max_label is 1, so the shells are 1, 1 + 1 and 1 + 2 -- all three
        # must actually be present, and nothing else.
        assert set(np.unique(out).tolist()) == {0, 1, 2, 3}
        # The core sits inside the middle shell, which sits inside the outer.
        assert (out == 1).sum() < (out == 2).sum() < (out == 3).sum()
        assert np.array_equal(out > 0, labels > 0)

    def test_an_empty_half_body_returns_zeros(self):
        empty = np.zeros((6, 6, 6), dtype=np.uint16)

        out = scipy_mod.subdivide_labels_3layers(
            empty, is_half_body=True, cut_axis=0
        )

        assert out.shape == empty.shape
        assert not out.any()

    def test_a_half_body_cut_at_the_start_is_mirrored_forward(self):
        """
        The widest slice marks the cut surface. Here it is the first slice of
        the object, so the mirror is prepended and the original half is read
        back out of the second half of the working volume.
        """
        volume = np.zeros((12, 12, 12), dtype=np.uint16)
        for i in range(5):
            half = 5 - i
            volume[2 + i, 6 - half : 6 + half, 6 - half : 6 + half] = 1

        out = scipy_mod.subdivide_labels_3layers(
            volume, is_half_body=True, cut_axis=0
        )

        assert out.shape == volume.shape
        # The object starts at z=2, so the result has to be written back at
        # that offset -- nothing may leak outside its bounding box.
        assert not out[:2].any()
        assert not out[7:].any()
        # max_label is 1, so the shells are 1 (core), 2 and 3 (outer).  The
        # cut face is the widest slice and must expose all three; the tip of
        # the object is outer shell only.  Returning the mirrored half
        # instead would swap these two.
        assert set(np.unique(out[2]).tolist()) == {0, 1, 2, 3}
        assert set(np.unique(out[6]).tolist()) == {0, 3}

    def test_an_invalid_cut_axis_is_rejected(self):
        volume = np.ones((4, 4, 4), dtype=np.uint16)

        with pytest.raises(ValueError, match="cut_axis must be between"):
            scipy_mod.subdivide_labels_3layers(
                volume, is_half_body=True, cut_axis=3
            )

    @pytest.mark.parametrize(
        "order, shape",
        [
            ("TYX", (3, 8, 8)),
            ("CYX", (3, 8, 8)),
            ("TCYX", (2, 2, 8, 8)),
            ("TZYX", (2, 2, 8, 8)),
            ("ZCYX", (2, 2, 8, 8)),
            ("TZCYX", (2, 2, 2, 8, 8)),
            ("TCZYX", (2, 2, 2, 8, 8)),
        ],
    )
    def test_leading_axes_stay_independent(self, order, shape):
        """Only one leading slice carries signal; the rest must stay empty."""
        image = np.zeros(shape, dtype=np.float32)
        index = (0,) * (len(shape) - 2)
        image[index + (4, 4)] = 100.0

        out = scipy_mod.gaussian_blur(image, sigma=1.0, dimension_order=order)

        assert out.shape == image.shape
        assert out[index].sum() > 0
        empty = (slice(1, None),) + (slice(None),) * (len(shape) - 1)
        assert out[empty].sum() == 0

    def test_zyx_blurs_across_the_volume(self):
        volume = np.zeros((5, 8, 8), dtype=np.float32)
        volume[2, 4, 4] = 100.0

        out = scipy_mod.gaussian_blur(volume, sigma=1.0, dimension_order="ZYX")

        # A 3D blur spreads signal into the neighbouring planes.
        assert out[1].sum() > 0 and out[3].sum() > 0

    def test_a_mismatched_hint_falls_back_to_a_plain_blur(self):
        image = np.zeros((4, 8, 8), dtype=np.float32)
        image[1, 4, 4] = 100.0

        out = scipy_mod.gaussian_blur(image, sigma=1.0, dimension_order="TCYX")

        # The hint does not match the rank, so the whole array is blurred.
        assert out[0].sum() > 0

    def test_an_oversized_window_clamps_the_dask_halo(self, capsys):
        da = pytest.importorskip("dask.array")
        image = da.from_array(
            np.arange(100, dtype=np.uint8).reshape(10, 10), chunks=(2, 2)
        )

        out = scipy_mod.median_filter(image, size=9, dimension_order="YX")

        assert hasattr(out, "chunks")
        assert out.compute(scheduler="synchronous").shape == (10, 10)
        printed = capsys.readouterr().out
        assert "clamping to 1" in printed

    def test_without_scipy_the_filters_refuse_to_run(self):
        probe = _exec_module_without(
            scipy_mod,
            "_probe_scipy_filters_no_scipy",
            ["scipy", "scipy.ndimage"],
        )

        assert probe.SCIPY_AVAILABLE is False
        with pytest.raises(ImportError, match="SciPy is not available"):
            probe.gaussian_blur(np.zeros((2, 2)))
        with pytest.raises(ImportError, match="SciPy is not available"):
            probe.median_filter(np.zeros((2, 2)))
        # The whole `if SCIPY_AVAILABLE:` block is skipped, so the two
        # functions that exist only inside it are absent entirely.
        assert not hasattr(probe, "resize_labels")
        assert not hasattr(probe, "subdivide_labels_3layers")
        # ...and nothing was registered, so the live registry still points at
        # the real, SciPy-backed implementations.
        registered = BatchProcessingRegistry._processing_functions
        assert registered["Gaussian Blur"]["func"] is scipy_mod.gaussian_blur
        assert registered["Median Filter"]["func"] is scipy_mod.median_filter

# src/napari_tmidas/_tests/test_label_based_cropping_combined_coverage.py
"""
Coverage-driven tests for the label-based cropping pair of modules.

``_label_based_cropping_widget`` (the Qt dock widget) and
``processing_functions.label_based_cropping`` (the array/file level
pipeline) are exercised together because the widget imports the
expansion helpers straight out of the processing module -- a change to
either side shows up here.

Two things shape how these tests are written:

* every guard in ``_on_crop_clicked`` / ``_on_crop_finished`` opens a
  modal ``QMessageBox``, which would block the whole session, so the
  dialogs are replaced on the *widget module object* (they are imported
  names there, patching ``qtpy.QtWidgets`` is too late).
* the widget only ever touches ``viewer.layers``, ``viewer.dims`` and
  ``viewer.add_image``.  A tiny stand-in viewer holding *real*
  ``napari.layers`` objects therefore drives every branch of the
  expansion handlers without building an OpenGL canvas, and lets the
  odd-dimensionality guards be reached at all.

Nothing here registers or clears anything in
``BatchProcessingRegistry``, so the process-wide singleton is untouched.
"""

import importlib.util
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest
from napari.layers import Image, Labels

import napari_tmidas._label_based_cropping_widget as widget_mod
import napari_tmidas.processing_functions.label_based_cropping as proc_mod
from napari_tmidas._label_based_cropping_widget import (
    LabelBasedCroppingWidget,
    LabelBasedCroppingWorker,
)
from napari_tmidas._registry import BatchProcessingRegistry

tifffile = pytest.importorskip(
    "tifffile", reason="the pipeline reads and writes TIFFs"
)


# --------------------------------------------------------------------
# stand-in viewer
# --------------------------------------------------------------------
# Qt widget tests segfault the macOS CI runner and take the whole session
# down. Same guard as test_label_based_cropping_widget.py / test_frame_removal.py.
requires_gui = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)

class _FakeEventEmitter:
    """Records ``connect`` calls the way napari's EventEmitter accepts."""

    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)
        return callback


class _FakeLayerList(list):
    """A list that also indexes by layer name, like napari's LayerList."""

    def __init__(self, layers=()):
        super().__init__(layers)
        self.events = types.SimpleNamespace(
            inserted=_FakeEventEmitter(), removed=_FakeEventEmitter()
        )

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(la, "name", None) == item for la in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for layer in self:
                if getattr(layer, "name", None) == key:
                    return layer
            raise KeyError(key)
        return list.__getitem__(self, key)


class _FakeWindow:
    def __init__(self):
        self.docked = []

    def add_dock_widget(self, widget, **kwargs):
        self.docked.append((widget, kwargs))
        return widget


class FakeViewer:
    """Minimal duck-typed viewer: layers, dims, add_image, window."""

    def __init__(self, layers=(), current_step=(0, 0, 0, 0)):
        self.layers = _FakeLayerList(layers)
        self.dims = types.SimpleNamespace(current_step=current_step)
        self.window = _FakeWindow()
        self.added = []

    def add_image(self, data, **kwargs):
        self.added.append((data, kwargs))
        layer = types.SimpleNamespace(
            data=data, name=kwargs.get("name", "image")
        )
        self.layers.append(layer)
        return layer


def _image(shape, name="raw", dtype=np.uint8):
    rng = np.random.default_rng(0)
    return Image((rng.random(shape) * 200).astype(dtype), name=name)


def _labels(shape, name="mask"):
    data = np.zeros(shape, dtype=np.uint32)
    data[..., 1:3, 1:4] = 1
    return Labels(data, name=name)


@pytest.fixture
def no_dialogs(monkeypatch):
    """Replace both modal dialogs with a recorder (HANG HAZARD)."""
    calls = []
    monkeypatch.setattr(
        widget_mod.QMessageBox,
        "warning",
        lambda *a, **k: calls.append(a[2] if len(a) > 2 else ""),
    )
    monkeypatch.setattr(
        widget_mod.QMessageBox,
        "critical",
        lambda *a, **k: calls.append(a[2] if len(a) > 2 else ""),
    )
    return calls


def _select_all(widget):
    """Point both combos at index 0/1 of the fake viewer's layer list."""
    widget._image_layer_combo.setCurrentIndex(0)
    widget._label_layer_combo.setCurrentIndex(0)


def _module_copy_without(module, missing_names):
    """
    Execute *module*'s own source again as a throwaway copy, with
    *missing_names* poisoned in ``sys.modules`` so the module's
    ``except ImportError`` fallbacks run.

    ``importlib.reload`` is deliberately not used: it rebinds the
    classes inside the shared module object, so every
    ``from ... import X`` that another test module already performed
    would silently stop being ``module.X``.  Executing the file under a
    private name leaves ``sys.modules`` untouched, and coverage is keyed
    on the file path, so the fallback lines still count.
    """
    spec = importlib.util.spec_from_file_location(
        module.__name__ + "__probe", module.__file__
    )
    copy = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, dict.fromkeys(missing_names, None)):
        spec.loader.exec_module(copy)
    return copy


# ====================================================================
# widget: construction guards
# ====================================================================
@requires_gui
class TestWidgetConstructionGuards:
    """The early-out that keeps the widget importable without napari."""

    def test_returns_before_building_ui_without_napari(
        self, qtbot, monkeypatch
    ):
        monkeypatch.setattr(widget_mod, "_HAS_NAPARI", False)

        widget = LabelBasedCroppingWidget(FakeViewer())
        qtbot.addWidget(widget)

        assert widget._worker is None
        assert not hasattr(widget, "_image_layer_combo")
        assert widget.layout() is None

    def test_layer_change_event_refreshes_both_combos(self, qtbot):
        viewer = FakeViewer([_image((3, 5, 6))])
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)
        assert widget._label_layer_combo.count() == 0

        viewer.layers.append(_labels((3, 5, 6)))
        widget._on_layers_changed()

        assert widget._image_layer_combo.count() == 1
        assert widget._label_layer_combo.count() == 1
        assert widget._label_layer_combo.itemData(0) == 1


# ====================================================================
# widget: _on_crop_clicked guards
# ====================================================================
@requires_gui
class TestCropClickedGuards:
    """Every refusal path must warn and start no worker."""

    def test_missing_label_layer_warns(self, qtbot, no_dialogs):
        viewer = FakeViewer([_image((3, 5, 6))])
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)

        widget._on_crop_clicked()

        assert no_dialogs == ["Please select a label layer"]
        assert widget._worker is None

    def test_missing_image_layer_warns(self, qtbot, no_dialogs):
        viewer = FakeViewer([_labels((3, 5, 6))])
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)

        widget._on_crop_clicked()

        assert no_dialogs == ["Please select an image layer"]
        assert widget._worker is None

    def test_image_slot_pointing_at_a_labels_layer_is_refused(
        self, qtbot, no_dialogs
    ):
        """
        The combos are index-carrying; a stale index that resolves to a
        Labels layer must be caught before the worker masks the wrong
        array.
        """
        viewer = FakeViewer([_labels((3, 5, 6)), _labels((3, 5, 6), "m2")])
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)
        widget._image_layer_combo.addItem("m2", 1)
        _select_all(widget)

        widget._on_crop_clicked()

        assert no_dialogs == ["Selected image layer is not an Image layer"]
        assert widget._worker is None

    def test_label_slot_pointing_at_an_image_layer_is_refused(
        self, qtbot, no_dialogs
    ):
        viewer = FakeViewer([_image((3, 5, 6)), _image((3, 5, 6), "raw2")])
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)
        widget._label_layer_combo.addItem("raw2", 1)
        _select_all(widget)

        widget._on_crop_clicked()

        assert no_dialogs == ["Selected label layer is not a Labels layer"]
        assert widget._worker is None

    def test_valid_selection_hands_the_arrays_to_the_worker(
        self, qtbot, no_dialogs, monkeypatch
    ):
        """
        ``start()`` is stubbed so the assertions do not race a real
        thread; everything up to and including the hand-off is real.
        """
        monkeypatch.setattr(
            widget_mod.LabelBasedCroppingWorker, "start", lambda self: None
        )
        image_layer = _image((3, 5, 6))
        label_layer = _labels((3, 5, 6))
        widget = LabelBasedCroppingWidget(
            FakeViewer([image_layer, label_layer])
        )
        qtbot.addWidget(widget)
        widget._crop_name_input.setText("")

        widget._on_crop_clicked()

        assert no_dialogs == []
        assert isinstance(widget._worker, LabelBasedCroppingWorker)
        assert widget._worker.image_data is image_layer.data
        assert widget._worker.label_data is label_layer.data
        # an empty name box falls back to the default
        assert widget._crop_name == "cropped"
        assert widget._source_layer is image_layer
        assert widget._crop_button.isEnabled() is False
        assert "Starting cropping" in widget._info_text.toPlainText()

    def test_custom_output_name_is_kept(self, qtbot, no_dialogs, monkeypatch):
        monkeypatch.setattr(
            widget_mod.LabelBasedCroppingWorker, "start", lambda self: None
        )
        widget = LabelBasedCroppingWidget(
            FakeViewer([_image((3, 5, 6)), _labels((3, 5, 6))])
        )
        qtbot.addWidget(widget)
        widget._crop_name_input.setText("my_crop")

        widget._on_crop_clicked()

        assert widget._crop_name == "my_crop"


# ====================================================================
# widget: _on_crop_finished
# ====================================================================
@requires_gui
class TestCropFinished:
    """What the main thread does with the worker's result."""

    def _ready_widget(self, qtbot, extra_layers=()):
        image_layer = _image((3, 5, 6))
        label_layer = _labels((3, 5, 6))
        viewer = FakeViewer([image_layer, label_layer, *extra_layers])
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)
        widget._crop_name = "cropped"
        widget._source_layer = image_layer
        widget._crop_button.setEnabled(False)
        return widget, viewer, image_layer

    def test_success_adds_a_new_layer_with_source_display_settings(
        self, qtbot, no_dialogs
    ):
        widget, viewer, image_layer = self._ready_widget(qtbot)
        cropped = np.ones((3, 5, 6), dtype=np.uint8)

        widget._on_crop_finished(True, "done", cropped)

        assert len(viewer.added) == 1
        data, kwargs = viewer.added[0]
        assert data is cropped
        assert kwargs["name"] == "cropped"
        assert kwargs["colormap"] is image_layer.colormap
        assert kwargs["blending"] == image_layer.blending
        assert widget._crop_button.isEnabled() is True
        # the worker's message and the follow-up both survive: the status
        # box appends, it does not overwrite
        status = widget._info_text.toPlainText()
        assert "done" in status
        assert "Added result as 'cropped'" in status

    def test_success_reuses_an_existing_layer_of_the_same_name(
        self, qtbot, no_dialogs
    ):
        """
        Re-cropping must overwrite the previous result instead of piling
        up duplicate layers.
        """
        existing = types.SimpleNamespace(
            data=np.zeros((3, 5, 6), np.uint8), name="cropped"
        )
        widget, viewer, _ = self._ready_widget(qtbot, extra_layers=[existing])
        cropped = np.full((3, 5, 6), 7, dtype=np.uint8)

        widget._on_crop_finished(True, "done", cropped)

        assert viewer.added == []
        assert existing.data is cropped

    def test_failure_raises_a_critical_dialog_and_re_enables_the_button(
        self, qtbot, no_dialogs
    ):
        widget, viewer, _ = self._ready_widget(qtbot)

        widget._on_crop_finished(False, "Error during cropping: boom", None)

        assert no_dialogs == ["Error during cropping: boom"]
        assert viewer.added == []
        assert widget._crop_button.isEnabled() is True

    def test_success_with_no_data_neither_adds_nor_warns(
        self, qtbot, no_dialogs
    ):
        widget, viewer, _ = self._ready_widget(qtbot)

        widget._on_crop_finished(True, "nothing to do", None)

        assert viewer.added == []
        assert no_dialogs == []
        assert widget._crop_button.isEnabled() is True


# ====================================================================
# widget: Z expansion handler
# ====================================================================
@requires_gui
class TestExpandZHandler:
    """``_on_expand_z_changed`` rewrites the label layer in place."""

    def _widget(self, qtbot, image_layer, label_layer, current_step=(0,) * 4):
        viewer = FakeViewer([image_layer, label_layer], current_step)
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)
        _select_all(widget)
        return widget, viewer

    def test_unchecking_is_a_no_op(self, qtbot):
        label_layer = _labels((5, 6))
        widget, _ = self._widget(qtbot, _image((3, 5, 6)), label_layer)
        before = label_layer.data

        widget._on_expand_z_changed(0)

        assert label_layer.data is before

    def test_2d_label_is_repeated_over_z_of_a_3d_image(self, qtbot):
        label_layer = _labels((5, 6))
        original = label_layer.data.copy()
        widget, _ = self._widget(qtbot, _image((3, 5, 6)), label_layer)

        widget._on_expand_z_changed(2)

        assert label_layer.data.shape == (3, 5, 6)
        for z in range(3):
            np.testing.assert_array_equal(label_layer.data[z], original)
        assert "Expanded 2D label to 3D" in widget._info_text.toPlainText()

    def test_3d_label_copies_the_current_z_slice_everywhere(self, qtbot):
        label_data = np.zeros((3, 5, 6), dtype=np.uint32)
        label_data[1, 2, 2] = 9
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(
            qtbot, _image((3, 5, 6)), label_layer, current_step=(1, 0, 0)
        )

        widget._on_expand_z_changed(2)

        assert label_layer.data.shape == (3, 5, 6)
        for z in range(3):
            assert label_layer.data[z, 2, 2] == 9
        assert "Copied slice 1" in widget._info_text.toPlainText()

    def test_unexpected_label_ndim_for_3d_image_unticks(self, qtbot):
        label_layer = _labels((2, 2, 2, 5, 6))
        widget, _ = self._widget(qtbot, _image((3, 5, 6)), label_layer)
        widget._expand_z_checkbox.setChecked(True)
        widget._info_text.setText("")

        widget._on_expand_z_changed(2)

        assert widget._expand_z_checkbox.isChecked() is False
        assert "Unexpected label" in widget._info_text.toPlainText()

    def test_2d_label_on_4d_image_expands_z_only(self, qtbot):
        label_layer = _labels((5, 6))
        original = label_layer.data.copy()
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)

        widget._on_expand_z_changed(2)

        # only the image's Z extent (3), not its T extent (2)
        assert label_layer.data.shape == (3, 5, 6)
        for z in range(3):
            np.testing.assert_array_equal(label_layer.data[z], original)

    def test_2d_label_on_4d_image_also_expands_t_when_ticked(self, qtbot):
        label_layer = _labels((5, 6))
        original = label_layer.data.copy()
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        widget._expand_time_checkbox.blockSignals(True)
        widget._expand_time_checkbox.setChecked(True)
        widget._expand_time_checkbox.blockSignals(False)

        widget._on_expand_z_changed(2)

        assert label_layer.data.shape == (2, 3, 5, 6)
        np.testing.assert_array_equal(label_layer.data[1, 2], original)

    def test_per_frame_3d_label_is_broadcast_across_z(self, qtbot):
        label_data = np.zeros((2, 5, 6), dtype=np.uint32)
        label_data[1, 3, 3] = 4
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)

        widget._on_expand_z_changed(2)

        assert label_layer.data.shape == (2, 3, 5, 6)
        assert np.all(label_layer.data[1, :, 3, 3] == 4)
        assert np.all(label_layer.data[0, :, 3, 3] == 0)
        assert "per-frame labels" in widget._info_text.toPlainText()

    def test_3d_label_that_is_not_per_frame_uses_the_current_z(self, qtbot):
        label_data = np.zeros((4, 5, 6), dtype=np.uint32)
        label_data[2, 1, 1] = 5
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(
            qtbot,
            _image((2, 3, 5, 6)),
            label_layer,
            current_step=(0, 2, 0, 0),
        )

        widget._on_expand_z_changed(2)

        assert label_layer.data.shape == (3, 5, 6)
        assert np.all(label_layer.data[:, 1, 1] == 5)
        # nothing but slice 2's single labelled pixel survived
        assert label_layer.data.sum() == 5 * 3
        assert "Copied slice 2 across Z" in widget._info_text.toPlainText()

    def test_4d_label_only_rewrites_the_current_frame(self, qtbot):
        label_data = np.zeros((2, 3, 5, 6), dtype=np.uint32)
        label_data[1, 2, 4, 4] = 6
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(
            qtbot,
            _image((2, 3, 5, 6)),
            label_layer,
            current_step=(1, 2, 0, 0),
        )

        widget._on_expand_z_changed(2)

        assert label_layer.data.shape == (2, 3, 5, 6)
        assert np.all(label_layer.data[1, :, 4, 4] == 6)
        assert np.all(label_layer.data[0] == 0)
        assert "Copied frame 1 slice 2" in widget._info_text.toPlainText()

    def test_unexpected_label_ndim_for_4d_image_unticks(self, qtbot):
        label_layer = _labels((2, 2, 2, 5, 6))
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        widget._expand_z_checkbox.setChecked(True)
        widget._info_text.setText("")

        widget._on_expand_z_changed(2)

        assert widget._expand_z_checkbox.isChecked() is False
        assert "Unexpected label" in widget._info_text.toPlainText()

    def test_2d_image_cannot_be_z_expanded(self, qtbot):
        widget, _ = self._widget(qtbot, _image((5, 6)), _labels((5, 6)))
        widget._expand_z_checkbox.setChecked(True)

        widget._on_expand_z_changed(2)

        assert widget._expand_z_checkbox.isChecked() is False
        assert "must be 3D or 4D" in widget._info_text.toPlainText()

    def test_expansion_failure_is_reported_not_raised(
        self, qtbot, monkeypatch
    ):
        def boom(*_args, **_kwargs):
            raise RuntimeError("no memory")

        monkeypatch.setattr(widget_mod, "_expand_label_to_3d", boom)
        widget, _ = self._widget(qtbot, _image((3, 5, 6)), _labels((5, 6)))
        widget._expand_z_checkbox.setChecked(True)

        widget._on_expand_z_changed(2)

        assert widget._expand_z_checkbox.isChecked() is False
        assert "no memory" in widget._info_text.toPlainText()


# ====================================================================
# widget: time expansion handler
# ====================================================================
@requires_gui
class TestExpandTimeHandler:
    """``_on_expand_time_changed`` only ever applies to 4D images."""

    def _widget(self, qtbot, image_layer, label_layer, current_step=(0,) * 4):
        viewer = FakeViewer([image_layer, label_layer], current_step)
        widget = LabelBasedCroppingWidget(viewer)
        qtbot.addWidget(widget)
        _select_all(widget)
        return widget, viewer

    def test_unchecking_is_a_no_op(self, qtbot):
        label_layer = _labels((5, 6))
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        before = label_layer.data

        widget._on_expand_time_changed(0)

        assert label_layer.data is before

    def test_non_4d_image_is_refused(self, qtbot):
        widget, _ = self._widget(qtbot, _image((3, 5, 6)), _labels((5, 6)))
        widget._expand_time_checkbox.setChecked(True)

        widget._on_expand_time_changed(2)

        assert widget._expand_time_checkbox.isChecked() is False
        assert "only applies to 4D" in widget._info_text.toPlainText()

    def test_2d_label_requires_z_expansion_first(self, qtbot):
        label_layer = _labels((5, 6))
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        widget._expand_time_checkbox.setChecked(True)
        before = label_layer.data

        widget._on_expand_time_changed(2)

        assert widget._expand_time_checkbox.isChecked() is False
        assert label_layer.data is before
        assert "enable 'Expand across Z'" in widget._info_text.toPlainText()

    def test_2d_label_expands_to_tzyx_when_z_is_ticked(self, qtbot):
        label_layer = _labels((5, 6))
        original = label_layer.data.copy()
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        widget._expand_z_checkbox.blockSignals(True)
        widget._expand_z_checkbox.setChecked(True)
        widget._expand_z_checkbox.blockSignals(False)

        widget._on_expand_time_changed(2)

        assert label_layer.data.shape == (2, 3, 5, 6)
        np.testing.assert_array_equal(label_layer.data[1, 2], original)
        assert "all 2 time frames" in widget._info_text.toPlainText()

    def test_per_frame_label_copies_the_current_frame_to_all(self, qtbot):
        label_data = np.zeros((2, 5, 6), dtype=np.uint32)
        label_data[1, 0, 0] = 3
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(
            qtbot,
            _image((2, 3, 5, 6)),
            label_layer,
            current_step=(1, 0, 0, 0),
        )

        widget._on_expand_time_changed(2)

        assert label_layer.data.shape == (2, 5, 6)
        assert np.all(label_layer.data[:, 0, 0] == 3)
        assert "Copied frame 1" in widget._info_text.toPlainText()

    def test_single_3d_zyx_label_is_stacked_over_time(self, qtbot):
        label_data = np.zeros((3, 5, 6), dtype=np.uint32)
        label_data[2, 1, 1] = 8
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)

        widget._on_expand_time_changed(2)

        assert label_layer.data.shape == (2, 3, 5, 6)
        assert np.all(label_layer.data[:, 2, 1, 1] == 8)
        assert "Copied 3D label" in widget._info_text.toPlainText()

    def test_4d_label_copies_the_current_frame_to_all_frames(self, qtbot):
        label_data = np.zeros((2, 3, 5, 6), dtype=np.uint32)
        label_data[1, 1, 2, 2] = 2
        label_layer = Labels(label_data, name="mask")
        widget, _ = self._widget(
            qtbot,
            _image((2, 3, 5, 6)),
            label_layer,
            current_step=(1, 0, 0, 0),
        )

        widget._on_expand_time_changed(2)

        assert label_layer.data.shape == (2, 3, 5, 6)
        assert np.all(label_layer.data[:, 1, 2, 2] == 2)

    def test_unexpected_label_ndim_unticks(self, qtbot):
        label_layer = _labels((2, 2, 2, 5, 6))
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        widget._expand_time_checkbox.setChecked(True)
        widget._info_text.setText("")

        widget._on_expand_time_changed(2)

        assert widget._expand_time_checkbox.isChecked() is False
        assert "Unexpected label" in widget._info_text.toPlainText()

    def test_expansion_failure_is_reported_not_raised(
        self, qtbot, monkeypatch
    ):
        def boom(*_args, **_kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(widget_mod, "_expand_label_to_time", boom)
        label_layer = _labels((5, 6))
        widget, _ = self._widget(qtbot, _image((2, 3, 5, 6)), label_layer)
        widget._expand_z_checkbox.blockSignals(True)
        widget._expand_z_checkbox.setChecked(True)
        widget._expand_z_checkbox.blockSignals(False)
        widget._expand_time_checkbox.setChecked(True)

        widget._on_expand_time_changed(2)

        assert widget._expand_time_checkbox.isChecked() is False
        assert "kaboom" in widget._info_text.toPlainText()


# ====================================================================
# widget: worker signal guards + entry point
# ====================================================================
@requires_gui
class TestWorkerSignalGuards:
    """Signal emission must never take the worker thread down."""

    def _worker(self):
        return LabelBasedCroppingWorker(
            np.ones((2, 3), np.uint8), np.ones((2, 3), np.uint32)
        )

    class _Boom:
        """A signal stand-in that records the attempt, then explodes."""

        def __init__(self):
            self.attempts = []

        def emit(self, *args):
            self.attempts.append(args)
            raise RuntimeError("receiver gone")

    def test_a_broken_progress_signal_is_swallowed(self, qtbot):
        worker = self._worker()
        worker.progress = self._Boom()
        finished = []
        worker.finished.connect(lambda *a: finished.append(a))

        worker.run()

        # both progress emissions blew up, the run still completed
        assert [a[0] for a in worker.progress.attempts] == [
            "Cropping image...",
            "Cropping completed!",
        ]
        assert len(finished) == 1
        success, message, cropped = finished[0]
        assert success is True
        assert "completed successfully" in message
        np.testing.assert_array_equal(cropped, np.ones((2, 3), np.uint8))

    def test_a_broken_finished_signal_is_swallowed(self, qtbot):
        worker = self._worker()
        worker.finished = self._Boom()

        worker.run()  # must not raise

        # the result was still computed and handed to the dead signal
        assert len(worker.finished.attempts) == 1
        success, message, cropped = worker.finished.attempts[0]
        assert success is True
        assert "completed successfully" in message
        np.testing.assert_array_equal(cropped, np.ones((2, 3), np.uint8))


@requires_gui
class TestNapariEntryPoint:
    """The magicgui callable napari.yaml points at."""

    def test_docks_the_widget_on_the_right(self, qtbot):
        func = getattr(
            widget_mod.label_based_cropping_widget,
            "_function",
            widget_mod.label_based_cropping_widget,
        )
        viewer = FakeViewer([_image((3, 5, 6)), _labels((3, 5, 6))])

        widget = func(viewer)
        qtbot.addWidget(widget)

        assert isinstance(widget, LabelBasedCroppingWidget)
        assert len(viewer.window.docked) == 1
        docked, kwargs = viewer.window.docked[0]
        assert docked is widget
        assert kwargs["name"] == "Label-Based Cropping"
        assert kwargs["area"] == "right"


# ====================================================================
# processing module: _load_image
# ====================================================================
class TestLoadImage:
    """Both readers inside ``_load_image``, plus the hard dependency."""

    def test_missing_tifffile_raises_importerror(self, monkeypatch, tmp_path):
        monkeypatch.setattr(proc_mod, "_HAS_TIFFFILE", False)

        with pytest.raises(ImportError, match="tifffile is required"):
            proc_mod._load_image(str(tmp_path / "nope.tif"))

    def test_series_reader_returns_the_pixels_and_one_letter_per_axis(
        self, tmp_path
    ):
        path = tmp_path / "im.tif"
        written = np.arange(5 * 5 * 6, dtype=np.uint8).reshape(5, 5, 6)
        tifffile.imwrite(str(path), written)

        image, axes = proc_mod._load_image(str(path))

        np.testing.assert_array_equal(image, written)
        # the series reader names every dimension, the last two of which
        # are always the spatial pair the cropping code slices on
        assert len(axes) == written.ndim
        assert axes.endswith("YX")

    def _fake_tifffile(self, image, imagej_metadata):
        class _Tif:
            series = []
            imagej_metadata = None

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def asarray(self_inner):
                return image

        _Tif.imagej_metadata = imagej_metadata
        return types.SimpleNamespace(TiffFile=lambda _p: _Tif())

    def test_page_fallback_without_metadata_reports_no_axes(self, monkeypatch):
        image = np.zeros((3, 5, 6), np.uint8)
        monkeypatch.setattr(
            proc_mod, "tifffile", self._fake_tifffile(image, None)
        )

        loaded, axes = proc_mod._load_image("ignored.tif")

        assert loaded is image
        assert axes == ""

    def test_page_fallback_infers_tyx_from_imagej_frames(self, monkeypatch):
        image = np.zeros((4, 5, 6), np.uint8)
        monkeypatch.setattr(
            proc_mod, "tifffile", self._fake_tifffile(image, {"frames": 4})
        )

        _, axes = proc_mod._load_image("ignored.tif")

        assert axes == "TYX"

    def test_page_fallback_infers_tzyx_for_4d(self, monkeypatch):
        image = np.zeros((2, 3, 5, 6), np.uint8)
        monkeypatch.setattr(
            proc_mod, "tifffile", self._fake_tifffile(image, {"frames": 2})
        )

        _, axes = proc_mod._load_image("ignored.tif")

        assert axes == "TZYX"

    def test_page_fallback_leaves_2d_unlabelled(self, monkeypatch):
        image = np.zeros((5, 6), np.uint8)
        monkeypatch.setattr(
            proc_mod, "tifffile", self._fake_tifffile(image, {"frames": 1})
        )

        _, axes = proc_mod._load_image("ignored.tif")

        assert axes == ""


# ====================================================================
# processing module: small helpers
# ====================================================================
class TestHelperGuards:
    def test_crop_helper_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="does not match label shape"):
            proc_mod._crop_image_with_label(
                np.zeros((3, 5, 6), np.uint8), np.zeros((5, 6), np.uint32)
            )

    def test_crop_helper_does_not_mutate_its_input(self):
        image = np.full((5, 6), 7, np.uint8)
        label = np.zeros((5, 6), np.uint32)
        label[0, 0] = 1

        cropped = proc_mod._crop_image_with_label(image, label)

        assert cropped[0, 0] == 7
        assert cropped[1, 1] == 0
        assert np.all(image == 7), "the source array was modified in place"

    def test_infer_axes_falls_back_to_generic_names_above_4d(self):
        assert (
            proc_mod._infer_axes(np.zeros((2, 2, 2, 2, 2), np.uint8))
            == "D0D1D2D3D4"
        )

    def test_infer_axes_without_metadata_defaults_to_zyx(self):
        assert proc_mod._infer_axes(np.zeros((3, 5, 6), np.uint8)) == "ZYX"

    def test_label_filename_lookup_scans_the_known_suffixes_in_order(
        self, tmp_path
    ):
        intensity = str(tmp_path / "a.tif")
        (tmp_path / "a_other.tif").write_bytes(b"")

        # an unrecognised neighbour is not accepted
        assert proc_mod._get_label_image_filename(intensity) is None

        (tmp_path / "a_mask.tif").write_bytes(b"")
        assert proc_mod._get_label_image_filename(intensity) == str(
            tmp_path / "a_mask.tif"
        )

        # ...and an earlier entry in the suffix list takes precedence
        (tmp_path / "a_labels.tif").write_bytes(b"")
        assert proc_mod._get_label_image_filename(intensity) == str(
            tmp_path / "a_labels.tif"
        )


# ====================================================================
# processing module: label_based_cropping guards
# ====================================================================
def _write(path, array):
    tifffile.imwrite(str(path), array)
    return str(path)


class TestLabelBasedCroppingGuards:
    """The dimension bookkeeping in front of the actual masking."""

    def test_empty_label_path_is_rejected(self):
        with pytest.raises(ValueError, match="must be provided"):
            proc_mod.label_based_cropping(np.zeros((5, 6), np.uint8))

    def test_2d_label_on_3d_image_needs_expand_z(self, tmp_path):
        label = _write(tmp_path / "lab.tif", np.ones((5, 6), np.uint8))

        with pytest.raises(ValueError, match="Enable expand_z"):
            proc_mod.label_based_cropping(
                np.zeros((3, 5, 6), np.uint8), label_image_path=label
            )

    def test_2d_label_on_4d_image_needs_expand_time(self, tmp_path):
        label = _write(tmp_path / "lab.tif", np.ones((5, 6), np.uint8))

        with pytest.raises(ValueError, match="enable expand_time"):
            proc_mod.label_based_cropping(
                np.zeros((2, 3, 5, 6), np.uint8), label_image_path=label
            )

    def test_2d_label_on_4d_image_also_needs_expand_z(self, tmp_path):
        label = _write(tmp_path / "lab.tif", np.ones((5, 6), np.uint8))

        with pytest.raises(ValueError, match="requires expand_z"):
            proc_mod.label_based_cropping(
                np.zeros((2, 3, 5, 6), np.uint8),
                label_image_path=label,
                expand_time=True,
            )

    def test_2d_label_expanded_over_t_and_z(self, tmp_path):
        label_array = np.zeros((5, 6), np.uint8)
        label_array[1:3, 1:4] = 1
        label = _write(tmp_path / "lab.tif", label_array)
        image = np.full((2, 3, 5, 6), 9, np.uint8)

        cropped = proc_mod.label_based_cropping(
            image,
            label_image_path=label,
            expand_time=True,
            expand_z=True,
        )

        assert cropped.shape == (2, 3, 5, 6)
        assert np.all(cropped[:, :, 1:3, 1:4] == 9)
        assert cropped.sum() == 9 * 2 * 3 * 2 * 3

    def test_per_frame_label_wrong_t_is_rejected(self, tmp_path):
        label = _write(tmp_path / "lab.tif", np.ones((5, 5, 6), np.uint8))

        with pytest.raises(ValueError, match=r"must match expected"):
            proc_mod.label_based_cropping(
                np.zeros((2, 3, 5, 6), np.uint8),
                label_image_path=label,
                expand_z=True,
            )

    def test_per_frame_label_needs_expand_z(self, tmp_path):
        label = _write(tmp_path / "lab.tif", np.ones((2, 5, 6), np.uint8))

        with pytest.raises(ValueError, match="Enable expand_z"):
            proc_mod.label_based_cropping(
                np.zeros((2, 3, 5, 6), np.uint8), label_image_path=label
            )

    def test_per_frame_label_is_broadcast_across_z(self, tmp_path):
        label_array = np.zeros((2, 5, 6), np.uint8)
        label_array[1, 0, 0] = 1
        label = _write(tmp_path / "lab.tif", label_array)
        image = np.full((2, 3, 5, 6), 4, np.uint8)

        cropped = proc_mod.label_based_cropping(
            image, label_image_path=label, expand_z=True
        )

        assert cropped.shape == (2, 3, 5, 6)
        assert np.all(cropped[1, :, 0, 0] == 4)
        assert cropped.sum() == 4 * 3

    def test_missing_label_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            proc_mod.label_based_cropping(
                np.zeros((5, 6), np.uint8),
                label_image_path=str(tmp_path / "absent.tif"),
            )

    def test_spatial_mismatch_is_reported(self, tmp_path):
        label = _write(tmp_path / "lab.tif", np.ones((7, 7), np.uint8))

        with pytest.raises(ValueError, match="spatial dimensions"):
            proc_mod.label_based_cropping(
                np.zeros((5, 6), np.uint8), label_image_path=label
            )


# ====================================================================
# processing module: batch driver
# ====================================================================
def _make_batch_inputs(folder):
    """a.tif -> croppable, b.tif -> no label, c.tif -> bad label."""
    image = np.full((5, 5, 6), 5, np.uint8)
    label = np.zeros((5, 6), np.uint8)
    label[1:3, 1:4] = 1
    _write(folder / "a.tif", image)
    _write(folder / "a_labels.tif", label)
    _write(folder / "b.tif", image)
    _write(folder / "c.tif", image)
    _write(folder / "c_labels.tif", np.ones((7, 7), np.uint8))
    return image, label


class TestBatchLabelBasedCropping:
    """The folder-level driver: routing, output naming and formats."""

    def test_routes_files_into_successful_failed_and_skipped(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        image, label = _make_batch_inputs(inp)
        out = tmp_path / "out"

        results = proc_mod.batch_label_based_cropping(
            str(inp), str(out), num_workers=1
        )

        assert out.is_dir()
        assert len(results["successful"]) == 1
        assert os.path.basename(results["successful"][0]) == "a_cropped.tif"
        # b.tif has no label, and the two *_labels.tif files have none
        # either, so they are skipped rather than failed
        assert len(results["skipped"]) == 3
        assert {os.path.basename(p) for p in results["skipped"]} == {
            "b.tif",
            "a_labels.tif",
            "c_labels.tif",
        }
        assert len(results["failed"]) == 1
        assert "c.tif" in results["failed"][0]
        assert "spatial dimensions" in results["failed"][0]

        written = tifffile.imread(str(out / "a_cropped.tif"))
        assert written.shape == image.shape
        expected = np.where(label[None, :, :] > 0, image, 0)
        np.testing.assert_array_equal(written, expected)

    def test_quiet_mode_prints_nothing(self, tmp_path, capsys):
        inp = tmp_path / "in"
        inp.mkdir()
        _make_batch_inputs(inp)

        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(tmp_path / "out"),
            num_workers=1,
            verbose=False,
        )

        assert len(results["successful"]) == 1
        assert capsys.readouterr().out == ""

    def test_verbose_mode_reports_the_summary(self, tmp_path, capsys):
        inp = tmp_path / "in"
        inp.mkdir()
        _make_batch_inputs(inp)

        proc_mod.batch_label_based_cropping(
            str(inp), str(tmp_path / "out"), num_workers=1, verbose=True
        )

        out = capsys.readouterr().out
        assert "BATCH PROCESSING COMPLETE" in out
        assert "Successful: 1" in out
        assert "No label file found" in out

    def test_empty_folder_yields_empty_results(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()

        results = proc_mod.batch_label_based_cropping(
            str(inp), str(tmp_path / "out"), num_workers=1, verbose=False
        )

        assert results == {"successful": [], "failed": [], "skipped": []}

    def test_npy_output_format_writes_a_numpy_array(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        image, label = _make_batch_inputs(inp)
        out = tmp_path / "out"

        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(out),
            output_format="npy",
            num_workers=1,
            verbose=False,
        )

        assert len(results["successful"]) == 1
        saved = np.load(str(out / "a_cropped.npy"))
        np.testing.assert_array_equal(
            saved, np.where(label[None, :, :] > 0, image, 0)
        )

    def test_unknown_output_format_falls_back_to_tifffile(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        _make_batch_inputs(inp)
        out = tmp_path / "out"

        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(out),
            output_format="png",
            num_workers=1,
            verbose=False,
        )

        assert len(results["successful"]) == 1
        assert (out / "a_cropped.tif").exists()

    def test_a_failing_writer_falls_back_to_numpy(self, tmp_path, monkeypatch):
        """
        The last-resort ``except`` in the writer chain: if tifffile
        cannot serialise the array the batch must still leave a file
        behind instead of losing the result.
        """
        inp = tmp_path / "in"
        inp.mkdir()
        _make_batch_inputs(inp)
        out = tmp_path / "out"

        def _boom(*_a, **_k):
            raise RuntimeError("cannot write")

        stub = types.SimpleNamespace(TiffFile=tifffile.TiffFile, imwrite=_boom)
        monkeypatch.setattr(proc_mod, "tifffile", stub)

        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(out),
            output_format="png",
            num_workers=1,
            verbose=False,
        )

        assert len(results["successful"]) == 1
        assert (out / "a_cropped.npy").exists()

    def test_a_custom_suffix_priority_restricts_matching(self, tmp_path):
        """
        ``label_suffix_priority`` must actually be honoured: a file whose
        only label sits under a suffix NOT in the given list must be
        skipped, even though the built-in default list would have found
        it.
        """
        inp = tmp_path / "in"
        inp.mkdir()
        image = np.full((5, 5, 6), 5, np.uint8)
        label = np.zeros((5, 6), np.uint8)
        label[1:3, 1:4] = 1
        _write(inp / "a.tif", image)
        _write(inp / "a_mask.tif", label)  # only under "_mask.tif"
        out = tmp_path / "out"

        # "_mask.tif" is last in the built-in priority list but requested
        # here as the ONLY option, so this also proves the list is used
        # as given rather than merged with the default.
        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(out),
            label_suffix_priority=["_seg.tif"],
            num_workers=1,
            verbose=False,
        )
        assert results["successful"] == []
        assert any(
            "a.tif" in p for p in results["skipped"]
        ), "a_mask.tif must be invisible when only _seg.tif is requested"

        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(out),
            label_suffix_priority=["_mask.tif"],
            num_workers=1,
            verbose=False,
        )
        assert len(results["successful"]) == 1
        np.testing.assert_array_equal(
            tifffile.imread(str(out / "a_cropped.tif")),
            np.where(label[None, :, :] > 0, image, 0),
        )

    def test_auto_detect_labels_false_skips_every_file(self, tmp_path):
        """
        With no explicit label-path mechanism offered by this function,
        turning auto-detection off must mean no file is ever paired with
        a label -- not silently fall back to detecting anyway.
        """
        inp = tmp_path / "in"
        inp.mkdir()
        _make_batch_inputs(inp)
        out = tmp_path / "out"

        results = proc_mod.batch_label_based_cropping(
            str(inp),
            str(out),
            auto_detect_labels=False,
            num_workers=1,
            verbose=False,
        )

        assert results["successful"] == []
        assert results["failed"] == []
        assert {os.path.basename(p) for p in results["skipped"]} == {
            "a.tif",
            "b.tif",
            "a_labels.tif",
            "c.tif",
            "c_labels.tif",
        }


# ====================================================================
# widget: optional-dependency fallbacks
# ====================================================================
@requires_gui
class TestOptionalDependencyFallbacks:
    """
    The widget module is written to import cleanly on an install that
    has no magicgui / qtpy / tifffile, substituting stubs so that
    ``napari_tmidas`` itself still imports.  Executing the file again
    with those names poisoned in ``sys.modules`` is the only way to run
    the fallback definitions; ``_module_copy_without`` keeps that copy
    private so the module every other test holds is untouched.

    ``napari`` is deliberately not poisoned: the entry point's
    ``viewer: napari.Viewer`` annotation is evaluated at def time, so a
    copy built without napari would raise while executing.
    """

    def test_magicgui_stubs_pass_functions_through(self, qtbot):
        probe = _module_copy_without(
            widget_mod, ["magicgui", "magicgui.widgets"]
        )

        assert probe._HAS_MAGICGUI is False
        assert probe.create_widget is None
        assert isinstance(probe.Container, type)

        def sample():
            return 42

        # used bare, the stub decorators hand the function back...
        assert probe.magic_factory(sample) is sample
        assert probe.magicgui(sample) is sample
        # ...and used with options they return a pass-through
        assert probe.magic_factory(call_button="go")(sample) is sample
        assert probe.magicgui(layout="vertical")(sample) is sample
        # so the dock-widget entry point is still a plain callable that
        # builds and docks a widget rather than a magicgui FunctionGui
        assert probe.label_based_cropping_widget is (
            probe.napari_experimental_provide_dock_widget()
        )
        viewer = FakeViewer([_image((3, 5, 6)), _labels((3, 5, 6))])
        docked = probe.label_based_cropping_widget(viewer)
        qtbot.addWidget(docked)
        assert isinstance(docked, probe.LabelBasedCroppingWidget)
        assert viewer.window.docked[0][1]["name"] == "Label-Based Cropping"

    def test_qtpy_stubs_keep_the_worker_importable(self, qtbot):
        probe = _module_copy_without(
            widget_mod, ["qtpy", "qtpy.QtCore", "qtpy.QtWidgets"]
        )

        assert probe._HAS_QTPY is False
        assert probe.QMessageBox is None
        assert probe.Qt is None
        assert probe.Signal("str") is None
        # the stub QThread keeps ``.start()`` callable
        assert probe.QThread().start() is None
        # without Qt the worker is a plain object, not a QThread
        assert probe.LabelBasedCroppingWorker.__bases__ == (object,)
        assert not hasattr(probe.LabelBasedCroppingWorker, "progress")

        worker = probe.LabelBasedCroppingWorker(
            np.ones((2, 3), np.uint8), np.ones((2, 3), np.uint32)
        )
        # run() still executes; the emit helpers become no-ops
        worker.run()
        assert worker._emit_progress("hello") is None
        assert worker._emit_finished(True, "done", None) is None
        # the widget also degrades to a plain object and returns early
        assert probe.LabelBasedCroppingWidget.__bases__ == (object,)
        headless = probe.LabelBasedCroppingWidget(FakeViewer())
        assert headless._worker is None
        assert not hasattr(headless, "_image_layer_combo")

    def test_tifffile_absence_is_recorded(self, qtbot):
        probe = _module_copy_without(widget_mod, ["tifffile"])

        assert probe._HAS_TIFFFILE is False
        assert probe.tifffile is None
        # the rest of the module is unaffected
        assert probe._HAS_QTPY is True
        assert probe._HAS_MAGICGUI is True

    def test_the_shared_module_is_left_alone(self, qtbot):
        """
        A guard against a probe leaking into the rest of the run: the
        classes this file (and every other test module) imported must
        still be the ones living on the real module object.
        """
        assert widget_mod._HAS_QTPY is True
        assert widget_mod._HAS_MAGICGUI is True
        assert widget_mod._HAS_TIFFFILE is True
        assert widget_mod.QMessageBox is not None
        assert widget_mod.LabelBasedCroppingWidget is LabelBasedCroppingWidget
        assert widget_mod.LabelBasedCroppingWorker is LabelBasedCroppingWorker


class TestProcessingModuleWithoutTifffile:
    """
    The processing module degrades to ``_HAS_TIFFFILE = False`` rather
    than failing to import, which is what keeps the registry usable on
    an install without tifffile.

    Executing the file again re-runs the
    ``@BatchProcessingRegistry.register`` decorator, and that writes
    into a process-wide dict, so it is snapshotted and put back
    afterwards -- otherwise every later test in the session would see a
    different entry for "Label-Based Cropping".
    """

    def test_import_degrades_instead_of_failing(self):
        registry = BatchProcessingRegistry._processing_functions
        snapshot = dict(registry)
        try:
            probe = _module_copy_without(proc_mod, ["tifffile"])

            assert probe._HAS_TIFFFILE is False
            assert probe.tifffile is None
            with pytest.raises(ImportError, match="tifffile is required"):
                probe._load_image("anything.tif")
            # executing the copy really did re-run the decorator
            assert (
                registry["Label-Based Cropping"]["func"]
                is probe.label_based_cropping
            )
        finally:
            registry.clear()
            registry.update(snapshot)

        assert proc_mod._HAS_TIFFFILE is True
        assert BatchProcessingRegistry._processing_functions == snapshot
        assert (
            snapshot["Label-Based Cropping"]["func"]
            is proc_mod.label_based_cropping
        )

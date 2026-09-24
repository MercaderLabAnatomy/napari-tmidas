"""
Tests for the batch-processing UI of :mod:`napari_tmidas._file_selector`.

Covers the pieces the other ``test_file_selector*`` modules leave out:

* ``ParameterWidget``: widget construction per parameter type, reading the
  values back, applying saved values, and the channel selector refresh.
* ``FileResultsWidget``: function switching (thread-count locking), the
  cached Cellpose settings loader, ``start_batch_processing`` (thread count
  and output folder resolution), completion, and cancellation.
* A few leftovers: the table's mouse click handler, the TIFF reader fallback
  in ``_load_original_image``, the non-OME ``save_as_zarr`` fallback, and a
  handful of ``ProcessingWorker.process_file`` branches.

``FileResultsWidget`` is built against a throwaway, fully controlled
function registry (the real ``discover_and_load_processing_functions`` is
stubbed out) and a minimal fake viewer.  The worker's ``start`` is stubbed so
no QThread outlives a test; where the end-to-end flow matters the worker's
``run`` is called synchronously on the test thread instead.
"""

import json
import os
import sys
import types

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QSpinBox,
)

import napari_tmidas._file_selector as fs
from napari_tmidas._registry import BatchProcessingRegistry


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------
def _write_tif(path, data, axes=None):
    kwargs = {"photometric": "minisblack"}
    if axes is not None:
        kwargs["ome"] = True
        kwargs["metadata"] = {"axes": axes}
    tifffile.imwrite(str(path), data, **kwargs)
    return str(path)


class _Layer:
    def __init__(self, data, name="layer", **kwargs):
        self.data = data
        self.name = name
        self.kwargs = kwargs


class _Viewer:
    """Just enough of napari.Viewer for the table and results widgets."""

    def __init__(self):
        self.layers = []
        self.status = ""
        self.dims = types.SimpleNamespace(ndisplay=2)
        self.added_images = []

    def add_image(self, data, channel_axis=None, **kwargs):
        layer = _Layer(data, name=kwargs.pop("name", "image"), **kwargs)
        self.added_images.append(layer)
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(data, name=kwargs.pop("name", "labels"), **kwargs)
        self.layers.append(layer)
        return layer

    def reset_view(self):
        pass


class _MessageBoxStub:
    """Replaces QMessageBox so no modal dialog can block the test run."""

    Information = 1
    Ok = 0x400
    Cancel = 0x400000
    Ignore = 0x100000
    warning_answer = Cancel
    instances = []
    warnings = []

    def __init__(self, parent=None):
        self.text = None
        self.informative = None
        self.executed = False
        _MessageBoxStub.instances.append(self)

    def setIcon(self, icon):
        pass

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def setInformativeText(self, text):
        self.informative = text

    def setStandardButtons(self, buttons):
        pass

    def exec_(self):
        self.executed = True
        return self.Ok

    @classmethod
    def warning(cls, *args, **kwargs):
        cls.warnings.append(args)
        return cls.warning_answer


@pytest.fixture
def msgbox(monkeypatch):
    _MessageBoxStub.instances = []
    _MessageBoxStub.warnings = []
    _MessageBoxStub.warning_answer = _MessageBoxStub.Cancel
    monkeypatch.setattr(fs, "QMessageBox", _MessageBoxStub)
    return _MessageBoxStub


@pytest.fixture
def registry(monkeypatch):
    """An empty, isolated function registry restored after the test."""
    saved = dict(BatchProcessingRegistry._processing_functions)
    BatchProcessingRegistry._processing_functions.clear()
    monkeypatch.setattr(fs, "discover_and_load_processing_functions", lambda: None)
    yield BatchProcessingRegistry
    BatchProcessingRegistry._processing_functions.clear()
    BatchProcessingRegistry._processing_functions.update(saved)


@pytest.fixture
def started(monkeypatch):
    """Stub QThread.start on the widget's worker; records started workers."""
    calls = []
    monkeypatch.setattr(
        fs.ProcessingWorker, "start", lambda self: calls.append(self)
    )
    return calls


def _register(registry, name, func=None, **kwargs):
    if func is None:

        def func(image):
            return image

    registry.register(name=name, **kwargs)(func)
    return func


def _results_widget(tmp_path, files=None, viewer=None):
    if files is None:
        files = [
            _write_tif(tmp_path / f"img{i}.tif", np.full((4, 4), i, np.uint16))
            for i in range(2)
        ]
    viewer = viewer or _Viewer()
    widget = fs.FileResultsWidget(viewer, files, str(tmp_path), "")
    return widget, viewer, files


# ---------------------------------------------------------------------------
# ParameterWidget
# ---------------------------------------------------------------------------
class TestParameterWidgetConstruction:
    PARAMS = {
        "count": {
            "type": int,
            "default": 7,
            "min": 2,
            "max": 20,
            "step": 3,
            "description": "an int",
        },
        "sigma": {"type": float, "default": 1.25, "min": 0.5, "max": 4.0},
        "ratio": {"type": float, "default": 0.3, "step": 0.05},
        "enabled": {"type": bool, "default": True},
        "mode": {"type": str, "default": "b", "options": ["a", "b", "c"]},
        "level": {"type": int, "default": 3, "choices": [1, 2, 3]},
        "name": {"type": str, "default": "hello"},
        "output_suffix": {"type": str, "default": None, "description": "d"},
        "channel": {"type": str, "default": "all", "widget_type": "channel_selector"},
    }

    def test_widget_kind_per_parameter(self, qapp):
        pw = fs.ParameterWidget(self.PARAMS, "Some Function")
        w = pw.param_widgets
        assert isinstance(w["count"], QSpinBox)
        assert isinstance(w["sigma"], QDoubleSpinBox)
        assert isinstance(w["enabled"], QCheckBox)
        assert isinstance(w["mode"], QComboBox)
        assert isinstance(w["level"], QComboBox)
        assert isinstance(w["name"], QLineEdit)
        assert isinstance(w["channel"], QComboBox)
        assert pw._channel_selector_widget is w["channel"]

    def test_numeric_bounds_steps_and_defaults(self, qapp):
        pw = fs.ParameterWidget(self.PARAMS)
        w = pw.param_widgets
        assert (w["count"].minimum(), w["count"].maximum()) == (2, 20)
        assert w["count"].singleStep() == 3
        assert w["count"].value() == 7
        assert (w["sigma"].minimum(), w["sigma"].maximum()) == (0.5, 4.0)
        assert w["sigma"].decimals() == 3
        assert w["sigma"].singleStep() == pytest.approx(0.1)
        assert w["sigma"].value() == pytest.approx(1.25)
        assert w["ratio"].singleStep() == pytest.approx(0.05)

    def test_other_defaults(self, qapp):
        pw = fs.ParameterWidget(self.PARAMS)
        w = pw.param_widgets
        assert w["enabled"].isChecked() is True
        assert w["mode"].currentText() == "b"
        assert [w["mode"].itemText(i) for i in range(3)] == ["a", "b", "c"]
        assert w["level"].currentText() == "3"
        assert w["name"].text() == "hello"
        assert w["output_suffix"].text() == ""
        assert w["output_suffix"].minimumWidth() == 300
        assert w["channel"].count() == 1
        assert w["channel"].currentData() == "all"

    def test_option_default_not_in_list_keeps_first(self, qapp):
        pw = fs.ParameterWidget(
            {"mode": {"type": str, "default": "zzz", "options": ["a", "b"]}}
        )
        assert pw.param_widgets["mode"].currentText() == "a"

    def test_use_cpu_checkbox_emits_signal(self, qapp):
        pw = fs.ParameterWidget({"use_cpu": {"type": bool, "default": False}})
        seen = []
        pw.use_cpu_changed.connect(seen.append)
        pw.param_widgets["use_cpu"].setChecked(True)
        pw.param_widgets["use_cpu"].setChecked(False)
        assert seen == [True, False]

    def test_trackastra_label_pattern_triggers_refresh(self, qapp, monkeypatch):
        params = {
            "label_pattern": {"type": str, "default": "_labels.tif"},
            "channel": {"type": str, "widget_type": "channel_selector"},
        }
        pw = fs.ParameterWidget(params, "Trackastra Tracking")
        calls = []
        monkeypatch.setattr(pw, "update_channel_selector", calls.append)

        # No files yet: a pattern edit has nothing to refresh.
        pw.param_widgets["label_pattern"].setText("_seg.tif")
        assert calls == []

        pw.file_list = ["/data/a_seg.tif"]
        pw.param_widgets["label_pattern"].setText("_mask.tif")
        assert calls == [["/data/a_seg.tif"]]

    def test_label_pattern_ignored_for_other_functions(self, qapp, monkeypatch):
        pw = fs.ParameterWidget(
            {"label_pattern": {"type": str, "default": "_labels.tif"}},
            "Something Else",
        )
        pw.file_list = ["/data/a.tif"]
        calls = []
        monkeypatch.setattr(pw, "update_channel_selector", calls.append)
        pw.param_widgets["label_pattern"].setText("_x.tif")
        assert calls == []


class TestParameterWidgetValues:
    def test_get_values_types(self, qapp):
        params = {
            "count": {"type": int, "default": 4},
            "sigma": {"type": float, "default": 2.5},
            "flag": {"type": bool, "default": True},
            "level": {"type": int, "default": 2, "options": [1, 2, 3]},
            "mode": {"type": str, "default": "b", "options": ["a", "b"]},
            "thresh": {"type": float, "default": 0.75},
            "label": {"type": str, "default": "x"},
        }
        pw = fs.ParameterWidget(params)
        # Change a line edit that must be converted to float.
        pw.param_widgets["label"].setText("abc")
        values = pw.get_parameter_values()
        assert values == {
            "count": 4,
            "sigma": 2.5,
            "flag": True,
            "level": 2,
            "mode": "b",
            "thresh": 0.75,
            "label": "abc",
        }
        assert isinstance(values["level"], int)

    def test_unconvertible_values_fall_back_to_text(self, qapp):
        pw = fs.ParameterWidget(
            {
                "level": {"type": int, "default": "auto", "options": ["auto", 1]},
                "size": {"type": None, "default": "big"},
            }
        )
        # int("auto") and None("big") both fail -> the raw text is kept.
        assert pw.get_parameter_values() == {"level": "auto", "size": "big"}

    def test_line_edit_numeric_conversion(self, qapp):
        pw = fs.ParameterWidget({"size": {"type": complex, "default": "1+2j"}})
        assert pw.get_parameter_values() == {"size": 1 + 2j}
        pw.param_widgets["size"].setText("not a number")
        assert pw.get_parameter_values() == {"size": "not a number"}

    def test_channel_selector_values(self, qapp):
        pw = fs.ParameterWidget(
            {"channel": {"type": str, "widget_type": "channel_selector"}}
        )
        combo = pw.param_widgets["channel"]
        assert pw.get_parameter_values() == {"channel": "all"}

        combo.addItem("Channel 1", "1")
        combo.setCurrentIndex(1)
        assert pw.get_parameter_values() == {"channel": 1}

        combo.addItem("Weird", "red")
        combo.setCurrentIndex(2)
        assert pw.get_parameter_values() == {"channel": "red"}

    def test_apply_values_round_trip(self, qapp):
        params = {
            "count": {"type": int, "default": 1, "max": 50},
            "sigma": {"type": float, "default": 1.0},
            "flag": {"type": bool, "default": False},
            "mode": {"type": str, "default": "a", "options": ["a", "b"]},
            "name": {"type": str, "default": "x"},
            "note": {"type": str, "default": "keep"},
            "channel": {"type": str, "widget_type": "channel_selector"},
        }
        pw = fs.ParameterWidget(params)
        combo = pw.param_widgets["channel"]
        for i in range(3):
            combo.addItem(f"Channel {i}", str(i))

        pw.apply_parameter_values(
            {
                "count": "12",
                "sigma": "2.5",
                "flag": 1,
                "mode": "b",
                "name": 42,
                "note": None,
                "channel": 2,
                "not_a_param": 99,
            }
        )
        assert pw.get_parameter_values() == {
            "count": 12,
            "sigma": 2.5,
            "flag": True,
            "mode": "b",
            "name": "42",
            "note": "",
            "channel": 2,
        }

        # None selects "All channels" again.
        pw.apply_parameter_values({"channel": None})
        assert combo.currentData() == "all"

    def test_apply_ignores_invalid_and_unknown(self, qapp):
        params = {
            "count": {"type": int, "default": 5},
            "mode": {"type": str, "default": "a", "options": ["a", "b"]},
            "channel": {"type": str, "widget_type": "channel_selector"},
        }
        pw = fs.ParameterWidget(params)
        pw.apply_parameter_values(
            {"count": "not-an-int", "mode": "zzz", "channel": 7}
        )
        assert pw.get_parameter_values() == {
            "count": 5,
            "mode": "a",
            "channel": "all",
        }
        # None / empty mapping is a no-op.
        pw.apply_parameter_values(None)
        assert pw.get_parameter_values()["count"] == 5


class TestUpdateChannelSelector:
    CHANNEL_PARAMS = {"channel": {"type": str, "widget_type": "channel_selector"}}

    def _items(self, combo):
        return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]

    def test_no_selector_is_noop(self, qapp):
        pw = fs.ParameterWidget({"x": {"type": int, "default": 1}})
        pw.update_channel_selector(["/nope.tif"])  # must not raise
        assert not hasattr(pw, "_channel_selector_widget")

    def test_empty_file_list_resets(self, qapp):
        pw = fs.ParameterWidget(self.CHANNEL_PARAMS)
        pw._channel_selector_widget.addItem("Channel 0", "0")
        pw.update_channel_selector([])
        assert self._items(pw._channel_selector_widget) == [("All channels", "all")]

    def test_multichannel_tiff_populates_and_keeps_selection(self, tmp_path, qapp):
        path = _write_tif(
            tmp_path / "multi.tif", np.zeros((3, 8, 8), np.uint16), axes="CYX"
        )
        pw = fs.ParameterWidget(self.CHANNEL_PARAMS)
        combo = pw._channel_selector_widget
        pw.update_channel_selector([path])
        assert self._items(combo) == [
            ("All channels", "all"),
            ("Channel 0", "0"),
            ("Channel 1", "1"),
            ("Channel 2", "2"),
        ]

        combo.setCurrentIndex(2)
        pw.update_channel_selector([path])
        assert combo.currentData() == "1", "refresh must keep the user's pick"

    def test_single_channel_tiff_has_only_all(self, tmp_path, qapp):
        path = _write_tif(tmp_path / "one.tif", np.zeros((5, 8, 8), np.uint16), axes="ZYX")
        pw = fs.ParameterWidget(self.CHANNEL_PARAMS)
        pw.update_channel_selector([path])
        assert self._items(pw._channel_selector_widget) == [("All channels", "all")]

    def test_non_tiff_goes_through_loader(self, tmp_path, qapp, monkeypatch):
        layers = [
            (np.zeros((4, 4)), {}, "image"),
            (np.zeros((4, 4)), {}, "image"),
            (np.zeros((4, 4), np.uint32), {}, "labels"),
        ]
        loaded = []

        def fake_loader(path, *a, **k):
            loaded.append(path)
            return layers

        monkeypatch.setattr(fs, "load_image_file", fake_loader)
        monkeypatch.setenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", "1")
        pw = fs.ParameterWidget(self.CHANNEL_PARAMS)
        target = str(tmp_path / "picture.png")
        pw.update_channel_selector([target])
        assert loaded == [target]
        # Two image layers -> two channel options.
        assert [d for _, d in self._items(pw._channel_selector_widget)] == [
            "all",
            "0",
            "1",
        ]

    def test_non_tiff_array_shape(self, tmp_path, qapp, monkeypatch):
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, *a, **k: np.zeros((3, 16, 16))
        )
        monkeypatch.setenv("TMIDAS_VERBOSE_CHANNEL_DETECTION", "1")
        pw = fs.ParameterWidget(self.CHANNEL_PARAMS)
        pw.update_channel_selector([str(tmp_path / "x.png")])
        assert pw._channel_selector_widget.count() == 4

    def test_detection_failure_is_reported(self, tmp_path, qapp, monkeypatch, capsys):
        def boom(*a, **k):
            raise RuntimeError("detector broke")

        monkeypatch.setattr(fs, "detect_channels_for_file", boom)
        pw = fs.ParameterWidget(self.CHANNEL_PARAMS)
        pw.update_channel_selector([str(tmp_path / "x.tif")])
        assert self._items(pw._channel_selector_widget) == [("All channels", "all")]
        assert "Channel detection failed: detector broke" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "label_pattern,label_name",
        [
            ("_labels.tif", "cells_labels.tif"),  # configured pattern
            ("_seg.tif", "cells_label.tif"),  # falls back to "_label.tif"
            (None, "cells_labels.tif"),  # no configured pattern
        ],
    )
    def test_trackastra_uses_raw_file_for_label_input(
        self, tmp_path, qapp, label_pattern, label_name
    ):
        _write_tif(tmp_path / "cells.tif", np.zeros((2, 8, 8), np.uint16), axes="CYX")
        labels = _write_tif(
            tmp_path / label_name, np.zeros((8, 8), np.uint32), axes="YX"
        )
        params = dict(self.CHANNEL_PARAMS)
        if label_pattern is not None:
            params["label_pattern"] = {"type": str, "default": label_pattern}
        pw = fs.ParameterWidget(params, "Trackastra Tracking")
        pw.update_channel_selector([labels])
        # The 2-channel raw file was inspected, not the 1-channel label file.
        assert pw._channel_selector_widget.count() == 3

    def test_trackastra_without_raw_file_uses_label_file(self, tmp_path, qapp):
        labels = _write_tif(
            tmp_path / "cells_labels.tif", np.zeros((8, 8), np.uint32), axes="YX"
        )
        params = dict(self.CHANNEL_PARAMS)
        params["label_pattern"] = {"type": str, "default": "_labels.tif"}
        pw = fs.ParameterWidget(params, "Trackastra Tracking")
        pw.update_channel_selector([labels])
        assert pw._channel_selector_widget.count() == 1


# ---------------------------------------------------------------------------
# FileResultsWidget: function switching
# ---------------------------------------------------------------------------
def _thread_locked(widget):
    return (
        widget.thread_count_label.isHidden()
        and widget.thread_count.isHidden()
        and not widget.thread_count.isEnabled()
        and widget.thread_count.value() == 1
    )


class TestUpdateFunctionInfo:
    def test_normal_function_unlocked(self, tmp_path, qapp, registry):
        _register(
            registry,
            "Blur",
            description="Gaussian blur",
            parameters={"sigma": {"type": float, "default": 1.0}},
        )
        widget, _, _ = _results_widget(tmp_path)
        assert widget.processing_selector.currentText() == "Blur"
        assert widget.function_description.text() == "Gaussian blur"
        assert not _thread_locked(widget)
        assert widget.thread_count.isEnabled()
        assert isinstance(widget.param_widget_instance, fs.ParameterWidget)
        assert widget.param_widget_instance.get_parameter_values() == {"sigma": 1.0}

    @pytest.mark.parametrize(
        "name,description,tooltip_part",
        [
            ("Merge Folder", "", "entire folders"),
            ("Split Timepoints", "", "entire folders"),
            ("Denoise", "Runs careamics", "entire folders"),
            ("Ultrack Tracking", "", "memory-intensive"),
        ],
    )
    def test_folder_like_functions_lock_threads(
        self, tmp_path, qapp, registry, name, description, tooltip_part
    ):
        _register(registry, name, description=description)
        widget, _, _ = _results_widget(tmp_path)
        assert _thread_locked(widget)
        assert tooltip_part in widget.thread_count.toolTip()
        assert widget.function_description.text().endswith(
            "This function has to run single-threaded."
        )
        # No parameters -> placeholder label.
        assert isinstance(widget.param_widget_instance, QLabel)
        assert widget.param_widget_instance.text() == "No parameters for this function"

    def test_existing_warning_is_not_duplicated(self, tmp_path, qapp, registry):
        _register(registry, "Merge Folder", description="WARNING: slow")
        widget, _, _ = _results_widget(tmp_path)
        assert widget.function_description.text() == "WARNING: slow"

    def test_gpu_distributed_function_locks_threads(self, tmp_path, qapp, registry):
        def track(image):
            return image

        track.supports_gpu_distribution = True
        _register(registry, "GPU Track", func=track, description="tracks")
        widget, _, _ = _results_widget(tmp_path)
        assert _thread_locked(widget)
        assert "one worker per available GPU" in widget.thread_count.toolTip()
        # GPU-distributed functions are not "single-threaded" folder functions.
        assert widget.function_description.text() == "tracks"

    def test_use_cpu_parameter_drives_gpu_lock(self, tmp_path, qapp, registry):
        _register(
            registry,
            "Segment",
            parameters={"use_cpu": {"type": bool, "default": False}},
        )
        widget, _, _ = _results_widget(tmp_path)
        assert widget._thread_locked_by_gpu is True
        assert _thread_locked(widget)
        assert "GPU processing requires single thread" in widget.thread_count.toolTip()

        widget.param_widget_instance.param_widgets["use_cpu"].setChecked(True)
        assert widget._thread_locked_by_gpu is False
        assert not _thread_locked(widget)

    def test_switching_functions_replaces_parameter_widget(
        self, tmp_path, qapp, registry
    ):
        _register(registry, "A Params", parameters={"k": {"type": int, "default": 2}})
        _register(registry, "B Plain")
        _register(registry, "C Folder")
        widget, _, _ = _results_widget(tmp_path)
        first = widget.param_widget_instance
        assert isinstance(first, fs.ParameterWidget)

        widget.processing_selector.setCurrentText("B Plain")
        assert isinstance(widget.param_widget_instance, QLabel)
        assert widget._param_scroll_area is None

        widget.processing_selector.setCurrentText("C Folder")
        assert isinstance(widget.param_widget_instance, QLabel)
        assert _thread_locked(widget)

        widget.processing_selector.setCurrentText("A Params")
        assert isinstance(widget.param_widget_instance, fs.ParameterWidget)
        assert widget.param_widget_instance is not first
        assert not _thread_locked(widget)

    def test_unknown_function_is_ignored(self, tmp_path, qapp, registry):
        _register(registry, "Blur", description="d")
        widget, _, _ = _results_widget(tmp_path)
        before = widget.param_widget_instance
        widget.update_function_info("does not exist")
        assert widget.param_widget_instance is before
        assert widget.function_description.text() == "d"

    def test_channel_selector_is_filled_from_files(self, tmp_path, qapp, registry):
        files = [
            _write_tif(tmp_path / "rgbish.tif", np.zeros((2, 8, 8), np.uint16), axes="CYX")
        ]
        _register(
            registry,
            "Pick",
            parameters={"channel": {"type": str, "widget_type": "channel_selector"}},
        )
        widget, _, _ = _results_widget(tmp_path, files=files)
        pw = widget.param_widget_instance
        assert pw.file_list == files
        assert pw._channel_selector_widget.count() == 3


class TestCachedCellposeSettings:
    PARAMS = {
        "diameter": {"type": float, "default": 30.0},
        "flow_threshold": {"type": float, "default": 0.4},
        "model": {"type": str, "default": "cyto3", "options": ["cyto3", "nuclei"]},
    }

    def _settings(self, tmp_path, base, tag, signature, sub=()):
        folder = tmp_path.joinpath(
            *sub, "tmp", "cellpose_timepoint_cache", f"{base}_interleaved_ch0_{tag}"
        )
        folder.mkdir(parents=True)
        path = folder / "run_settings.json"
        path.write_text(
            json.dumps({"run_signature": signature})
            if not isinstance(signature, str)
            else signature
        )
        return path

    def test_latest_settings_are_applied(self, tmp_path, qapp, registry, capsys):
        files = [_write_tif(tmp_path / "stack.tif", np.zeros((4, 4), np.uint16))]
        old = self._settings(
            tmp_path, "stack", "old", {"diameter": 10.0, "model": "cyto3"}
        )
        new = self._settings(
            tmp_path,
            "stack",
            "new",
            {"diameter": 55.0, "model": "nuclei", "unrelated_key": 1},
            sub=("tmp", "cellpose_auto_zarr"),
        )
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        _register(registry, "Cellpose Segmentation", parameters=self.PARAMS)

        widget, _, _ = _results_widget(tmp_path, files=files)
        values = widget.param_widget_instance.get_parameter_values()
        assert values == {"diameter": 55.0, "flow_threshold": 0.4, "model": "nuclei"}
        assert str(new) in capsys.readouterr().out

    def test_no_cache_keeps_defaults(self, tmp_path, qapp, registry):
        files = [_write_tif(tmp_path / "stack.tif", np.zeros((4, 4), np.uint16))]
        _register(registry, "Cellpose Segmentation", parameters=self.PARAMS)
        widget, _, _ = _results_widget(tmp_path, files=files)
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 30.0

    def test_other_files_cache_is_not_used(self, tmp_path, qapp, registry):
        files = [_write_tif(tmp_path / "stack.tif", np.zeros((4, 4), np.uint16))]
        self._settings(tmp_path, "different", "x", {"diameter": 99.0})
        _register(registry, "Cellpose Segmentation", parameters=self.PARAMS)
        widget, _, _ = _results_widget(tmp_path, files=files)
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 30.0

    def test_non_dict_signature_is_ignored(self, tmp_path, qapp, registry):
        files = [_write_tif(tmp_path / "stack.tif", np.zeros((4, 4), np.uint16))]
        self._settings(tmp_path, "stack", "x", ["diameter", 12.0])
        _register(registry, "Cellpose Segmentation", parameters=self.PARAMS)
        widget, _, _ = _results_widget(tmp_path, files=files)
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 30.0

    def test_corrupt_json_warns(self, tmp_path, qapp, registry, capsys):
        files = [_write_tif(tmp_path / "stack.tif", np.zeros((4, 4), np.uint16))]
        self._settings(tmp_path, "stack", "x", "{not json")
        _register(registry, "Cellpose Segmentation", parameters=self.PARAMS)
        widget, _, _ = _results_widget(tmp_path, files=files)
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 30.0
        assert "could not load cached Cellpose settings" in capsys.readouterr().out

    def test_guard_clauses(self, tmp_path, qapp, registry):
        files = [_write_tif(tmp_path / "stack.tif", np.zeros((4, 4), np.uint16))]
        self._settings(tmp_path, "stack", "x", {"diameter": 12.0})
        _register(registry, "Plain", parameters=self.PARAMS)
        widget, _, _ = _results_widget(tmp_path, files=files)

        # Not a Cellpose function: cache ignored.
        widget._maybe_apply_cached_cellpose_settings("Plain")
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 30.0

        # No files: nothing to look up.
        widget.file_list = []
        widget._maybe_apply_cached_cellpose_settings("cellpose")
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 30.0

        # Parameter placeholder without apply_parameter_values: no crash.
        widget.file_list = files
        widget.param_widget_instance = QLabel("none")
        widget._maybe_apply_cached_cellpose_settings("cellpose")

        # The positive control: the same cache does apply to a Cellpose name.
        widget.processing_selector.setCurrentText("Plain")
        widget.update_function_info("Plain")
        widget._maybe_apply_cached_cellpose_settings("cellpose")
        assert widget.param_widget_instance.get_parameter_values()["diameter"] == 12.0


# ---------------------------------------------------------------------------
# FileResultsWidget: batch processing
# ---------------------------------------------------------------------------
class TestStartBatchProcessing:
    def test_no_function_selected(self, tmp_path, qapp, registry, started):
        widget, viewer, _ = _results_widget(tmp_path)
        widget.start_batch_processing()
        assert viewer.status == "No processing function selected"
        assert started == []
        assert widget.worker is None

    def test_worker_configuration_and_end_to_end(
        self, tmp_path, qapp, registry, started, msgbox
    ):
        def add_offset(image, offset: int = 0):
            return image + offset

        _register(
            registry,
            "Add Offset",
            func=add_offset,
            suffix="_off",
            parameters={"offset": {"type": int, "default": 5}},
        )
        widget, viewer, files = _results_widget(tmp_path)
        widget.thread_count.setMaximum(8)
        widget.thread_count.setValue(3)
        widget.output_folder.setText("results")

        widget.start_batch_processing()

        assert len(started) == 1
        worker = widget.worker
        assert worker is started[0]
        assert worker.thread_count == 3
        # The 2D TIFFs state their axes, so "YX" was pre-selected and is
        # forwarded alongside the function's own parameters.
        assert widget.dimension_order.currentText() == "YX"
        assert worker.param_values == {"offset": 5, "dimension_order": "YX"}
        assert worker.output_folder == os.path.join(str(tmp_path), "results")
        assert os.path.isdir(worker.output_folder)
        assert worker.output_suffix == "_off"
        assert worker.output_format == "tiff"
        assert not widget.batch_button.isEnabled()
        assert widget.cancel_button.isEnabled()
        assert not widget.progress_bar.isHidden()
        assert viewer.status == "Processing 2 files with Add Offset using 3 threads"

        # Drive the worker synchronously: signals land in the widget.
        worker.run()

        assert widget.worker is None
        assert widget.progress_bar.value() == 100
        assert widget.batch_button.isEnabled()
        assert not widget.cancel_button.isEnabled()
        assert len(widget.processed_files_info) == 2
        assert viewer.status == "Completed processing 2 files"
        for src in files:
            out = widget.table.file_pairs[src]["processed"]
            assert out is not None
            assert os.path.basename(out) == os.path.basename(src).replace(
                ".tif", "_off.tif"
            )
            np.testing.assert_array_equal(
                tifffile.imread(out), tifffile.imread(src) + 5
            )
        assert msgbox.instances == [], "only grid functions show a dialog"

    def test_blank_output_folder_uses_source_dir(
        self, tmp_path, qapp, registry, started
    ):
        _register(registry, "Identity")
        widget, _, files = _results_widget(tmp_path)
        widget.output_format.setCurrentText("Zarr")
        widget.start_batch_processing()
        assert widget.worker.output_folder == os.path.dirname(files[0])
        assert widget.worker.output_format == "zarr"

    def test_dimension_order_is_forwarded(self, tmp_path, qapp, registry, started):
        _register(registry, "Identity")
        widget, _, _ = _results_widget(tmp_path)
        widget.dimension_order.setCurrentText("Auto")
        widget.start_batch_processing()
        assert widget.worker.param_values == {}, "'Auto' is not forwarded"

        widget.dimension_order.setCurrentText("TYX")
        widget.start_batch_processing()
        assert widget.worker.param_values == {"dimension_order": "TYX"}

    def test_dimension_order_prompt_can_cancel(
        self, tmp_path, qapp, registry, started, msgbox
    ):
        def needs_order(image, dimension_order="Auto"):
            return image

        _register(registry, "Needs Order", func=needs_order)
        files = [_write_tif(tmp_path / "vol.tif", np.zeros((3, 4, 4), np.uint16))]
        widget, viewer, _ = _results_widget(tmp_path, files=files)
        assert widget.dimension_order.currentText() == "Auto"

        widget.start_batch_processing()
        assert len(msgbox.warnings) == 1
        assert viewer.status == "Processing cancelled: select a dimension order first"
        assert started == []

        msgbox.warning_answer = msgbox.Ignore
        widget.start_batch_processing()
        assert len(started) == 1

    def test_not_thread_safe_forces_one_thread(self, tmp_path, qapp, registry, started):
        def unsafe(image):
            return image

        unsafe.thread_safe = False
        _register(registry, "Unsafe", func=unsafe)
        widget, _, _ = _results_widget(tmp_path)
        widget.thread_count.setMaximum(8)
        widget.thread_count.setValue(4)
        widget.start_batch_processing()
        assert widget.worker.thread_count == 1

    def test_gpu_mode_forces_one_thread(self, tmp_path, qapp, registry, started):
        _register(
            registry,
            "Segment",
            parameters={"use_cpu": {"type": bool, "default": False}},
        )
        widget, _, _ = _results_widget(tmp_path)
        # Bypass the UI lock to prove start_batch_processing enforces it too.
        widget.thread_count.setMaximum(8)
        widget.thread_count.setEnabled(True)
        widget.thread_count.setValue(4)
        widget.start_batch_processing()
        assert widget.worker.param_values["use_cpu"] is False
        assert widget.worker.thread_count == 1

    @pytest.fixture
    def gpu_module(self, monkeypatch):
        module = types.ModuleType("fake_gpu_module")
        module.requested = []
        monkeypatch.setitem(sys.modules, "fake_gpu_module", module)
        return module

    def _gpu_func(self, module):
        def gpu_track(image, gpus="", workers_per_gpu=1):
            return image

        gpu_track.supports_gpu_distribution = True
        gpu_track.__module__ = module.__name__
        return gpu_track

    @pytest.mark.parametrize(
        "n_gpus,workers,expected",
        [(2, 2, 4), (3, 1, 3), (1, 1, 1), (0, 1, 1), (2, "bad", 2)],
    )
    def test_gpu_distribution_thread_count(
        self, tmp_path, qapp, registry, started, gpu_module, n_gpus, workers, expected
    ):
        def detect(gpus):
            gpu_module.requested.append(gpus)
            return list(range(n_gpus))

        gpu_module._detect_gpu_ids = detect
        _register(
            registry,
            "GPU Track",
            func=self._gpu_func(gpu_module),
            parameters={
                "gpus": {"type": str, "default": "0,1"},
                "workers_per_gpu": {"type": str, "default": str(workers)},
            },
        )
        widget, viewer, _ = _results_widget(tmp_path)
        widget.start_batch_processing()
        assert gpu_module.requested == ["0,1"]
        assert widget.worker.thread_count == expected
        assert viewer.status.endswith(f"using {expected} threads")

    def test_gpu_detection_failure_falls_back_to_one(
        self, tmp_path, qapp, registry, started, gpu_module
    ):
        def detect(gpus):
            raise RuntimeError("no nvidia-smi")

        gpu_module._detect_gpu_ids = detect
        _register(registry, "GPU Track", func=self._gpu_func(gpu_module))
        widget, _, _ = _results_widget(tmp_path)
        widget.start_batch_processing()
        assert widget.worker.thread_count == 1

    def test_gpu_detection_falls_back_to_trackastra_helper(
        self, tmp_path, qapp, registry, started, gpu_module, monkeypatch
    ):
        tracking = pytest.importorskip(
            "napari_tmidas.processing_functions.trackastra_tracking"
        )
        monkeypatch.setattr(tracking, "_detect_gpu_ids", lambda gpus: [0, 1])
        # gpu_module has no _detect_gpu_ids of its own.
        _register(registry, "GPU Track", func=self._gpu_func(gpu_module))
        widget, _, _ = _results_widget(tmp_path)
        widget.start_batch_processing()
        assert widget.worker.thread_count == 2

    def test_grid_overlay_cache_is_reset(
        self, tmp_path, qapp, registry, started, monkeypatch
    ):
        grid = pytest.importorskip(
            "napari_tmidas.processing_functions.grid_view_overlay"
        )
        monkeypatch.setattr(grid, "_grid_output_path", "/stale/grid.tif")
        monkeypatch.setattr(grid, "_grid_saved", True)

        def create_grid_overlay(image):
            return None

        _register(registry, "Grid Overlay", func=create_grid_overlay)
        widget, _, _ = _results_widget(tmp_path)
        widget.start_batch_processing()
        assert grid._grid_output_path is None
        assert grid._grid_saved is False
        assert len(started) == 1


class TestProcessingFinishedAndErrors:
    def test_grid_result_is_shown(
        self, tmp_path, qapp, registry, msgbox, monkeypatch
    ):
        grid = pytest.importorskip(
            "napari_tmidas.processing_functions.grid_view_overlay"
        )
        rgb = np.zeros((6, 6, 3), np.uint8)
        rgb[..., 0] = 200
        grid_path = str(tmp_path / "grid.tif")
        tifffile.imwrite(grid_path, rgb, photometric="rgb")
        monkeypatch.setattr(grid, "_grid_output_path", grid_path)

        _register(registry, "Grid Overlay")
        widget, viewer, _ = _results_widget(tmp_path)
        widget.processing_finished()

        assert len(viewer.added_images) == 1
        layer = viewer.added_images[0]
        np.testing.assert_array_equal(layer.data, rgb)
        assert layer.name == "Grid Overlay (2 pairs)"
        assert layer.kwargs["rgb"] is True
        assert len(msgbox.instances) == 1
        assert msgbox.instances[0].executed
        assert grid_path in msgbox.instances[0].informative
        assert grid._grid_output_path is None, "cache reset for the next run"

    def test_grid_without_output_adds_nothing(
        self, tmp_path, qapp, registry, msgbox, monkeypatch
    ):
        grid = pytest.importorskip(
            "napari_tmidas.processing_functions.grid_view_overlay"
        )
        monkeypatch.setattr(grid, "_grid_output_path", None)
        _register(registry, "Grid Overlay")
        widget, viewer, _ = _results_widget(tmp_path)
        widget.processing_finished()
        assert viewer.added_images == []
        assert msgbox.instances == []
        assert viewer.status == "Completed processing 0 files"

    def test_missing_grid_file_is_reported(
        self, tmp_path, qapp, registry, msgbox, monkeypatch, capsys
    ):
        grid = pytest.importorskip(
            "napari_tmidas.processing_functions.grid_view_overlay"
        )
        monkeypatch.setattr(grid, "_grid_output_path", str(tmp_path / "gone.tif"))
        _register(registry, "Grid Overlay")
        widget, viewer, _ = _results_widget(tmp_path)
        widget.processing_finished()
        assert viewer.added_images == []
        assert "Could not load grid overlay" in capsys.readouterr().out
        assert widget.batch_button.isEnabled()

    def test_processing_error_sets_enhanced_status(self, tmp_path, qapp, registry):
        _register(registry, "Identity")
        widget, viewer, _ = _results_widget(tmp_path)
        widget.processing_error("/x.tif", "Cellpose failed with return code -9")
        assert viewer.status.startswith("Error processing /x.tif: Cellpose failed")
        assert "out-of-memory" in viewer.status

    def test_update_progress(self, tmp_path, qapp, registry):
        _register(registry, "Identity")
        widget, _, _ = _results_widget(tmp_path)
        widget.update_progress(42)
        assert widget.progress_bar.value() == 42


class _FakeWorker:
    def __init__(self, running):
        self.running = running
        self.calls = []

    def isRunning(self):
        return self.running

    def stop(self):
        self.calls.append("stop")

    def wait(self):
        self.calls.append("wait")


class TestCancelProcessing:
    def test_cancel_running_worker(self, tmp_path, qapp, registry, monkeypatch):
        cellpose_cancels = []
        monkeypatch.setattr(
            fs, "cancel_cellpose_processing", lambda: cellpose_cancels.append(1)
        )
        _register(registry, "Identity")
        widget, viewer, _ = _results_widget(tmp_path)
        widget.batch_button.setEnabled(False)
        widget.cancel_button.setEnabled(True)
        worker = _FakeWorker(running=True)
        widget.worker = worker

        widget.cancel_processing()

        assert cellpose_cancels == [1]
        assert worker.calls == ["stop", "wait"]
        assert widget.batch_button.isEnabled()
        assert not widget.cancel_button.isEnabled()
        assert viewer.status == "Processing cancelled"

    def test_cancel_without_running_worker(self, tmp_path, qapp, registry, monkeypatch):
        monkeypatch.setattr(fs, "cancel_cellpose_processing", None)
        _register(registry, "Identity")
        widget, viewer, _ = _results_widget(tmp_path)
        viewer.status = "idle"
        worker = _FakeWorker(running=False)
        widget.worker = worker
        widget.cancel_processing()
        assert worker.calls == []
        assert viewer.status == "idle"

        widget.worker = None
        widget.cancel_processing()  # no worker at all: still a no-op
        assert viewer.status == "idle"

    def test_cancel_real_worker(self, tmp_path, qapp, registry, monkeypatch):
        """A genuinely running QThread is stopped and joined."""
        import threading

        monkeypatch.setattr(fs, "cancel_cellpose_processing", None)
        gate = threading.Event()
        entered = threading.Event()

        def slow(image):
            entered.set()
            gate.wait(10)
            return image

        _register(registry, "Slow", func=slow)
        widget, viewer, _ = _results_widget(tmp_path)
        widget.start_batch_processing()
        worker = widget.worker
        try:
            assert entered.wait(10), "worker never started"
            assert worker.isRunning()
            gate.set()
            widget.cancel_processing()
            assert worker.stop_requested is True
            assert worker.isFinished()
            assert viewer.status == "Processing cancelled"
            assert widget.batch_button.isEnabled()
        finally:
            gate.set()
            worker.wait(10000)


# ---------------------------------------------------------------------------
# ProcessedFilesTableWidget leftovers
# ---------------------------------------------------------------------------
class TestTableMouseAndLoading:
    def _click(self, table, row, column, button=Qt.LeftButton):
        rect = table.visualItemRect(table.item(row, column))
        pos = rect.center()
        event = QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(pos),
            button,
            button,
            Qt.NoModifier,
        )
        table.mousePressEvent(event)

    def _table(self, tmp_path, qtbot):
        src = str(tmp_path / "a.tif")
        table = fs.ProcessedFilesTableWidget(_Viewer())
        qtbot.addWidget(table)
        table.resize(400, 200)
        table.add_initial_files([src])
        table.update_processed_files(
            [{"original_file": src, "processed_file": str(tmp_path / "a_p.tif")}]
        )
        table.show()
        orig, proc = [], []
        table._load_original_image = orig.append
        table._load_processed_image = proc.append
        return table, src, orig, proc

    def test_click_original_and_processed(self, tmp_path, qtbot):
        table, src, orig, proc = self._table(tmp_path, qtbot)
        self._click(table, 0, 0)
        assert orig == [src]
        assert proc == []
        self._click(table, 0, 1)
        assert proc == [str(tmp_path / "a_p.tif")]

    def test_right_click_loads_nothing(self, tmp_path, qtbot):
        table, _, orig, proc = self._table(tmp_path, qtbot)
        self._click(table, 0, 0, button=Qt.RightButton)
        self._click(table, 0, 1, button=Qt.RightButton)
        assert orig == [] and proc == []

    def test_click_unprocessed_cell_loads_nothing(self, tmp_path, qtbot):
        table = fs.ProcessedFilesTableWidget(_Viewer())
        qtbot.addWidget(table)
        table.resize(400, 200)
        table.add_initial_files([str(tmp_path / "a.tif")])
        table.show()
        proc = []
        table._load_processed_image = proc.append
        self._click(table, 0, 1)
        assert proc == []

    def test_tiff_reader_failure_falls_back(self, tmp_path, qapp, monkeypatch, capsys):
        data = np.arange(16, dtype=np.uint16).reshape(4, 4)
        path = _write_tif(tmp_path / "a.tif", data)

        def broken_reader(p):
            raise RuntimeError("reader broke")

        monkeypatch.setattr(fs, "tiff_reader_function", broken_reader)
        viewer = _Viewer()
        table = fs.ProcessedFilesTableWidget(viewer)
        table._load_original_image(path)

        assert "falling back to load_image_file" in capsys.readouterr().out
        assert len(viewer.layers) == 1
        np.testing.assert_array_equal(np.asarray(viewer.layers[0].data), data)
        assert table.current_original_images == viewer.layers

    def test_loader_layer_tuple_shapes(self, tmp_path, qapp, monkeypatch):
        """(data, kwargs) pairs and bare arrays are both accepted as layers."""
        path = tmp_path / "a.png"
        path.write_bytes(b"")  # only has to exist; the loader is stubbed
        first = np.ones((4, 4), np.uint8)
        second = np.full((4, 4), 2, np.uint8)
        labels = np.arange(16, dtype=np.uint32).reshape(4, 4)
        monkeypatch.setattr(
            fs,
            "load_image_file",
            lambda p, *a, **k: [(first, {}), second, labels],
        )
        viewer = _Viewer()
        table = fs.ProcessedFilesTableWidget(viewer)
        table._load_original_image(str(path))

        names = [layer.name for layer in viewer.layers]
        assert names == ["C1: a.png", "C2: a.png", "Labels3: a.png"]
        assert viewer.layers[0].kwargs["colormap"] == "red"
        assert viewer.layers[1].kwargs["colormap"] == "green"
        assert "colormap" not in viewer.layers[2].kwargs
        np.testing.assert_array_equal(viewer.layers[1].data, second)
        np.testing.assert_array_equal(viewer.layers[2].data, labels)
        assert viewer.status == "Loaded 3 channels from a.png"


# ---------------------------------------------------------------------------
# save_as_zarr without ome-zarr
# ---------------------------------------------------------------------------
class TestSaveAsZarrFallback:
    @pytest.mark.parametrize("chunks", ["auto", (2, 3)])
    def test_basic_zarr_is_written_with_chunks_and_zstd(
        self, tmp_path, monkeypatch, chunks
    ):
        # Regression: this path used zarr.save, which in zarr 3 treats the
        # chunks=/compressors= keywords as extra arrays and raises.
        zarr = pytest.importorskip("zarr")
        monkeypatch.setitem(sys.modules, "ome_zarr.io", None)
        data = np.arange(24, dtype=np.uint16).reshape(4, 6)
        target = str(tmp_path / "x.zarr")

        fs.save_as_zarr(data, target, chunks=chunks)

        stored = zarr.open_array(target, mode="r")
        np.testing.assert_array_equal(stored[:], data)
        assert stored.dtype == np.uint16
        if chunks != "auto":
            assert stored.chunks == chunks
        assert type(stored.compressors[0]).__name__ == "ZstdCodec"

    def test_basic_zarr_failure_is_a_value_error(self, tmp_path, monkeypatch):
        zarr = pytest.importorskip("zarr")
        monkeypatch.setitem(sys.modules, "ome_zarr.io", None)

        def broken(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(zarr, "create_array", broken)
        with pytest.raises(ValueError, match="disk full"):
            fs.save_as_zarr(np.zeros((2, 2)), str(tmp_path / "x.zarr"))

    def test_old_zarr_is_rejected(self, tmp_path, monkeypatch):
        zarr = pytest.importorskip("zarr")
        monkeypatch.setitem(sys.modules, "ome_zarr.io", None)
        monkeypatch.setattr(zarr, "__version__", "2.18.0")
        with pytest.raises(RuntimeError, match="Zarr v3"):
            fs.save_as_zarr(np.zeros((2, 2)), str(tmp_path / "x.zarr"))


# ---------------------------------------------------------------------------
# ProcessingWorker.process_file leftovers
# ---------------------------------------------------------------------------
def _worker(tmp_path, func, files, params=None):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return fs.ProcessingWorker(files, func, params or {}, str(out), "", "_proc")


class TestProcessFileBranches:
    def test_single_element_list_image_is_unwrapped(self, tmp_path, monkeypatch):
        arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
        # A multi-layer result with no "image" layer falls back to the first
        # layer's data, which here is a one-level multiscale list.
        monkeypatch.setattr(
            fs, "load_image_file", lambda p, **k: [([arr], {}, "labels")]
        )
        seen = {}

        def record(image):
            seen["image"] = image
            return image * 2

        src = str(tmp_path / "in.tif")
        result = _worker(tmp_path, record, [src]).process_file(src)
        assert isinstance(seen["image"], np.ndarray)
        np.testing.assert_array_equal(seen["image"], arr)
        np.testing.assert_array_equal(tifffile.imread(result["processed_file"]), arr * 2)

    def test_distributed_option_is_reported(self, tmp_path, capsys):
        src = _write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.uint16))
        seen = {}

        def seg(image, use_distributed_segmentation=False, distributed_blocksize_yx=0):
            seen["dist"] = use_distributed_segmentation
            seen["block"] = distributed_blocksize_yx
            return image

        worker = _worker(
            tmp_path,
            seg,
            [src],
            {"use_distributed_segmentation": 1, "distributed_blocksize_yx": 256},
        )
        worker.process_file(src)
        out = capsys.readouterr().out
        assert "Cellpose distributed option: requested=True, blocksize=256" in out
        assert f"source={src}" in out
        assert seen == {"dist": 1, "block": 256}

    def test_uninspectable_signature_drops_private_params(self, tmp_path):
        src = _write_tif(tmp_path / "a.tif", np.ones((4, 4), np.uint16))
        seen = {}

        class Opaque:
            __name__ = "opaque"
            # A non-Signature __signature__ makes inspect.signature raise
            # TypeError, which the worker answers by stripping "_" params.
            __signature__ = "not a signature"

            def __call__(self, image, **kwargs):
                seen.update(kwargs)
                return image

        worker = _worker(tmp_path, Opaque(), [src], {"gain": 2})
        worker.process_file(src)
        assert seen == {"gain": 2}

    @pytest.mark.parametrize(
        "coords,header",
        [
            (np.array([[1.0, 2.0], [3.0, 4.0]]), "y,x"),
            (np.array([[1.0, 2.0, 3.0]]), "z,y,x"),
        ],
    )
    def test_points_csv_without_pandas(self, tmp_path, monkeypatch, coords, header):
        monkeypatch.setitem(sys.modules, "pandas", None)
        src = _write_tif(tmp_path / "spots.tif", np.zeros((4, 4), np.uint16))
        worker = _worker(
            tmp_path, lambda image: coords, [src], {"output_csv": True}
        )
        result = worker.process_file(src)
        assert result["processed_file"].endswith("spots_spots.npy")
        np.testing.assert_array_equal(np.load(result["processed_file"]), coords)
        csv = tmp_path / "out" / "spots_spots.csv"
        lines = csv.read_text().splitlines()
        assert lines[0] == header
        np.testing.assert_allclose(
            np.loadtxt(str(csv), delimiter=",", skiprows=1, ndmin=2), coords
        )

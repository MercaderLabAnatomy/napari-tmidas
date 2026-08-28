"""Additional coverage for :mod:`napari_tmidas._label_inspection`.

Complements ``test_label_inspection.py`` by pinning the branches that
file does not reach: the edit-wrapper's indexing fall-backs and undo
bookkeeping, the single-timepoint delete / relabel / merge paths, the
click-to-split tool, the pick-ray helpers, the save / advance guards
and the raw-slice reads behind them.
"""

import contextlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

import napari_tmidas._label_inspection as mod
from napari_tmidas._label_inspection import (
    LabelInspector,
    _DaskFancyIndexWrapper,
)


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------
class _FakeLabels:
    """Minimal stand-in for a napari ``Labels`` layer."""

    def __init__(self, data, selected_label=1, scale=None):
        self.data = data
        self.selected_label = selected_label
        self.scale = (
            list(scale)
            if scale is not None
            else [1.0] * int(getattr(data, "ndim", 2))
        )
        self.refresh_count = 0
        self.bound_keys = {}
        self.visible = True

    def refresh(self):
        self.refresh_count += 1

    def bind_key(self, key, handler, overwrite=False):
        self.bound_keys[key] = handler


class _FakeViewer:
    """Just enough napari viewer for the inspector's needs."""

    def __init__(self, layers=()):
        self.layers = list(layers)
        self.status = ""
        self.mouse_drag_callbacks = []
        self.bound_keys = {}
        self.added_labels = []
        self.added_images = []
        self.dims = SimpleNamespace(ndisplay=2)

    def bind_key(self, key, handler, overwrite=False):
        self.bound_keys[key] = handler

    def add_labels(self, data, **kwargs):
        layer = _FakeLabels(data, scale=kwargs.get("scale"))
        self.added_labels.append((layer, kwargs))
        self.layers.append(layer)
        return layer

    def add_image(self, data, **kwargs):
        self.added_images.append((data, kwargs))
        return SimpleNamespace(data=data, **kwargs)


class _FakeEvent:
    def __init__(self, **kwargs):
        self.button = 1
        self.type = "mouse_press"
        self.position = (0.0, 0.0, 0.0)
        self.view_direction = None
        self.dims_displayed = [1, 2]
        self.modifiers = ()
        for key, value in kwargs.items():
            setattr(self, key, value)


def _wrap(base, chunks=None):
    """Wrap *base* (numpy) in the module's dask editing wrapper."""
    import dask.array as da

    if chunks is None:
        chunks = (1, *base.shape[1:])
    return _DaskFancyIndexWrapper(da.from_array(base, chunks=chunks))


@pytest.fixture()
def fake_labels_cls(monkeypatch):
    """Make ``isinstance(layer, Labels)`` recognise :class:`_FakeLabels`."""
    monkeypatch.setattr(mod, "Labels", _FakeLabels)
    return _FakeLabels


# ---------------------------------------------------------------------
# _DaskFancyIndexWrapper
# ---------------------------------------------------------------------
class TestWrapperIndexing:
    """Indexing fall-backs of the lazy label-edit wrapper."""

    def test_scalar_in_a_middle_dim_reads_that_hyperslice(self):
        """``w[:, z, :]`` routes through the T-slice cache, not dask."""
        base = np.arange(3 * 4 * 5, dtype=np.uint32).reshape(3, 4, 5)
        wrapper = _wrap(base)

        out = wrapper[:, 1, :]

        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, base[:, 1, :])
        # The constant dim, not dim 0, is what got cached.
        assert (1, 1) in wrapper._cache

    def test_all_slice_index_stays_lazy(self):
        """A pure-slice index must not materialise the whole array."""
        base = np.zeros((3, 4, 4), dtype=np.uint32)
        wrapper = _wrap(base)

        out = wrapper[:, :, :]

        assert hasattr(out, "compute")
        assert out.shape == (3, 4, 4)
        assert wrapper._cache == {}

    def test_multiple_varying_array_dims_read_point_by_point(self):
        """Fancy reads with no constant dim fall back to per-point reads."""
        base = np.arange(3 * 4 * 5, dtype=np.uint32).reshape(3, 4, 5)
        wrapper = _wrap(base)

        rows = np.array([0, 2])
        cols = np.array([1, 3])
        planes = np.array([0, 4])
        out = wrapper[rows, cols, planes]

        assert isinstance(out, np.ndarray)
        assert out.dtype == base.dtype
        assert np.array_equal(out, base[rows, cols, planes])

    def test_setitem_without_a_constant_dim_bakes_and_resets(self):
        """A genuinely nd write materialises and clears all edit state."""
        base = np.zeros((3, 4, 5), dtype=np.uint32)
        base[:, 0, 0] = 5
        wrapper = _wrap(base)
        wrapper.remap_values({5: 6})
        assert wrapper._op_log

        rows = np.array([0, 2])
        cols = np.array([1, 3])
        planes = np.array([0, 4])
        wrapper[rows, cols, planes] = 99

        result = np.asarray(wrapper)
        assert result[0, 1, 0] == 99 and result[2, 3, 4] == 99
        # The remap is baked in, so its undo record is gone for good.
        assert np.all(result[:, 0, 0] == 6)
        assert wrapper._op_log == []
        assert wrapper._lut == {}
        assert wrapper._diffs == {}

    def test_boolean_mask_write_is_promoted_to_a_dense_snapshot(self):
        """Indices that cannot be enumerated become a dense slice record."""
        base = np.zeros((2, 4, 4), dtype=np.uint32)
        wrapper = _wrap(base)

        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        wrapper[0, mask] = 7

        assert wrapper._diffs[(0, 0)][0] == "dense"
        result = np.asarray(wrapper)
        assert result[0, 1, 1] == 7 and result[0, 2, 2] == 7
        assert not np.any(result[1] == 7)

    def test_slice_write_is_enumerated_as_a_sparse_diff(self):
        """Int/slice writes small enough to enumerate stay sparse."""
        base = np.zeros((3, 8, 8), dtype=np.uint32)
        wrapper = _wrap(base)

        wrapper[1, 1:3, 0:2] = 4
        wrapper[1, 2, 0:3] = 6

        entry = wrapper._diffs[(0, 1)]
        assert entry[0] == "sparse"
        assert len(entry[1]) == 2
        result = np.asarray(wrapper)
        assert result[1, 1, 1] == 4
        assert np.array_equal(result[1, 2, 0:3], np.array([6, 6, 6]))

    def test_a_write_too_big_to_enumerate_goes_dense(self):
        """Coordinate storage larger than the slice itself is refused."""
        base = np.zeros((2, 3, 3), dtype=np.uint8)
        wrapper = _wrap(base)

        wrapper[0] = 3

        assert wrapper._diffs[(0, 0)][0] == "dense"
        assert np.all(np.asarray(wrapper)[0] == 3)

    def test_slice_cache_evicts_the_least_recently_used(self):
        """The read cache is bounded by ``_CACHE_MAX_SLICES``."""
        base = np.zeros((6, 2, 2), dtype=np.uint32)
        wrapper = _wrap(base)

        for t in range(6):
            wrapper[t]

        assert len(wrapper._cache) == _DaskFancyIndexWrapper._CACHE_MAX_SLICES
        assert (0, 0) not in wrapper._cache
        assert (0, 5) in wrapper._cache

    def test_identity_mapping_is_not_recorded(self):
        """``remap_values`` ignores no-op mappings entirely."""
        wrapper = _wrap(np.zeros((2, 2, 2), dtype=np.uint32))

        wrapper.remap_values({3: 3})

        assert wrapper._op_log == []
        assert wrapper._lut == {}

    def test_remap_and_undo_round_trip_a_dense_diff(self):
        """A dense snapshot is remapped in place and restored by undo."""
        base = np.zeros((2, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        wrapper = _wrap(base)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        wrapper[0, mask] = 5
        assert wrapper._diffs[(0, 0)][0] == "dense"

        wrapper.remap_values({5: 8})
        assert np.asarray(wrapper)[0, 1, 1] == 8

        assert wrapper.undo_remap() == {5: 8}
        result = np.asarray(wrapper)
        assert result[0, 1, 1] == 5
        assert result[1, 0, 0] == 5

    def test_undo_skips_a_diff_promoted_to_dense_after_the_remap(self):
        """Documented: a restructured diff record keeps the remapped value.

        The base array still reverts, so only the promoted slice differs.
        """
        base = np.zeros((2, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        wrapper = _wrap(base)
        wrapper[np.array([0]), np.array([1]), np.array([1])] = 5
        assert wrapper._diffs[(0, 0)][0] == "sparse"

        wrapper.remap_values({5: 6})
        big = np.zeros((4, 4), dtype=bool)
        big[3, 3] = True
        wrapper[0, big] = 9
        assert wrapper._diffs[(0, 0)][0] == "dense"

        assert wrapper.undo_remap() == {5: 6}
        result = np.asarray(wrapper)
        assert result[1, 0, 0] == 5  # base reverted
        assert result[0, 1, 1] == 6  # promoted snapshot left alone


# ---------------------------------------------------------------------
# Single-timepoint delete / relabel
# ---------------------------------------------------------------------
class TestSingleTimepointEdits:
    """``*_at_timepoint`` edits, their guards and their undo record."""

    def _setup(self, data, selected_label=1):
        layer = _FakeLabels(data, selected_label=selected_label)
        viewer = _FakeViewer([layer])
        return LabelInspector(viewer), viewer, layer

    def test_delete_at_timepoint_without_a_layer(self, fake_labels_cls):
        """No labels layer means nothing to delete."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.delete_label_at_timepoint(5, 0)

        assert viewer.status == "No labels layer found."

    def test_delete_at_timepoint_refuses_background(self, fake_labels_cls):
        """Label 0 is the background and is never deleted."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        inspector, viewer, _ = self._setup(data)

        inspector.delete_label_at_timepoint(0, 1)

        assert viewer.status == "Select a non-background label first."

    def test_delete_at_timepoint_reports_a_missing_label(
        self, fake_labels_cls
    ):
        """A label absent at *t* leaves the data untouched."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[0, 0, 0] = 5
        inspector, viewer, layer = self._setup(data)

        inspector.delete_label_at_timepoint(5, 2)

        assert viewer.status == "Label 5 not present at timepoint 2."
        assert layer.data[0, 0, 0] == 5
        assert inspector._single_t_last is None

    def test_delete_at_timepoint_edits_only_that_frame(self, fake_labels_cls):
        """Numpy labels are written in place and an undo record is kept."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        inspector, viewer, layer = self._setup(data)

        inspector.delete_label_at_timepoint(5, 1)

        assert layer.data[1, 0, 0] == 0
        assert layer.data[0, 0, 0] == 5 and layer.data[2, 0, 0] == 5
        assert inspector._single_t_last["t"] == 1
        assert "removed from timepoint 1" in viewer.status
        assert layer.refresh_count == 1

    def test_delete_at_timepoint_on_a_dask_wrapper(self, fake_labels_cls):
        """A raw dask array is wrapped before the sparse write."""
        import dask.array as da

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        inspector, viewer, layer = self._setup(
            da.from_array(base, chunks=(1, 4, 4))
        )

        inspector.delete_label_at_timepoint(5, 1)

        assert isinstance(layer.data, _DaskFancyIndexWrapper)
        result = np.asarray(layer.data)
        assert result[1, 0, 0] == 0
        assert result[0, 0, 0] == 5

    def test_delete_falls_back_to_the_whole_image_without_a_time_axis(
        self, fake_labels_cls
    ):
        """2-D labels have a single 'timepoint': the whole array."""
        data = np.zeros((4, 4), dtype=np.uint32)
        data[0, 0] = 5
        inspector, viewer, layer = self._setup(data)

        inspector.delete_label_at_timepoint(5, 0)

        assert layer.data[0, 0] == 0
        assert "no time axis" in viewer.status
        assert inspector._single_t_last is None

    def test_relabel_at_timepoint_guards(self, fake_labels_cls):
        """Background source and identity relabel are refused."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        inspector, viewer, layer = self._setup(data)

        inspector.relabel_label_at_timepoint(0, 3, 1)
        assert "Cannot relabel background" in viewer.status
        inspector.relabel_label_at_timepoint(5, 5, 1)
        assert viewer.status == "Label 5 already has this ID."
        assert np.all(layer.data[:, 0, 0] == 5)

    def test_relabel_at_timepoint_without_a_layer(self, fake_labels_cls):
        """No labels layer means nothing to relabel."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.relabel_label_at_timepoint(5, 3, 0)

        assert viewer.status == "No labels layer found."

    def test_relabel_at_timepoint_reports_a_missing_label(
        self, fake_labels_cls
    ):
        """A label absent at *t* is reported, not silently ignored."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[0, 0, 0] = 5
        inspector, viewer, _ = self._setup(data)

        inspector.relabel_label_at_timepoint(5, 3, 2)

        assert viewer.status == "Label 5 not present at timepoint 2."

    def test_relabel_at_timepoint_rewrites_only_that_frame(
        self, fake_labels_cls
    ):
        """The undo record names the reverted transition."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        inspector, viewer, layer = self._setup(data)

        inspector.relabel_label_at_timepoint(5, 3, 2)

        assert layer.data[2, 0, 0] == 3
        assert layer.data[0, 0, 0] == 5
        assert inspector._single_t_last["desc"] == "5→3 reverted"

    def test_relabel_without_a_time_axis_rewrites_everything(
        self, fake_labels_cls
    ):
        """2-D labels relabel across the whole array."""
        data = np.zeros((4, 4), dtype=np.uint32)
        data[0, 0] = 5
        inspector, viewer, layer = self._setup(data)

        inspector.relabel_label_at_timepoint(5, 3, 0)

        assert layer.data[0, 0] == 3
        assert "no time axis" in viewer.status

    def test_ctrl_z_restores_a_single_timepoint_edit(self, fake_labels_cls):
        """The Ctrl+Z handler replays the stored restore record."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        inspector, viewer, layer = self._setup(data)
        inspector.delete_label_at_timepoint(5, 1)
        assert layer.data[1, 0, 0] == 0

        inspector._on_undo_key()

        assert layer.data[1, 0, 0] == 5
        assert inspector._single_t_last is None
        assert "at timepoint 1" in viewer.status

    def test_undo_key_without_a_labels_layer_is_a_noop(self, fake_labels_cls):
        """Ctrl+Z with nothing loaded must not raise."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        assert inspector._on_undo_key() is None
        assert viewer.status == ""


# ---------------------------------------------------------------------
# Click plumbing
# ---------------------------------------------------------------------
class _ClickLabels(_FakeLabels):
    """Labels layer that resolves clicks the way napari does."""

    def __init__(self, data, value=5, t=1.0, **kwargs):
        super().__init__(data, **kwargs)
        self._value = value
        self._t = t

    def get_value(self, position, **kwargs):
        return self._value

    def world_to_data(self, position):
        return np.array([self._t, *np.asarray(position, dtype=float)[1:]])


class _OldNapariLabels(_FakeLabels):
    """Labels layer with napari's pre-kwargs ``get_value`` signature."""

    def get_value(self, position, world=True):
        return 5


class TestClickCallback:
    """``_make_click_callback`` dispatch rules."""

    def _drive(self, callback, viewer, event):
        gen = callback(viewer, event)
        next(gen)
        event.type = "mouse_release"
        with contextlib.suppress(StopIteration):
            next(gen)

    def test_non_left_button_never_fires(self, fake_labels_cls):
        """Right-click is left to napari (context menu / pan)."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        calls = []
        callback = inspector._make_click_callback(
            lambda *a: calls.append(a)
        )

        gen = callback(viewer, _FakeEvent(button=2))
        with pytest.raises(StopIteration):
            next(gen)
        assert calls == []

    def test_no_layer_means_no_dispatch(self, fake_labels_cls):
        """Without a labels layer the click is dropped."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        calls = []
        callback = inspector._make_click_callback(
            lambda *a: calls.append(a)
        )

        self._drive(callback, viewer, _FakeEvent())

        assert calls == []

    def test_older_napari_get_value_signature_is_supported(
        self, fake_labels_cls
    ):
        """A ``TypeError`` from the kwargs call retries the old signature."""
        layer = _OldNapariLabels(np.zeros((2, 2, 2), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        calls = []
        callback = inspector._make_click_callback(
            lambda *a: calls.append(a)
        )

        self._drive(callback, viewer, _FakeEvent())

        assert len(calls) == 1
        assert calls[0][0] is layer
        assert calls[0][1] == 5

    def test_a_drag_does_not_fire_the_tool(self, fake_labels_cls):
        """Pan/zoom drags must never edit labels."""
        layer = _OldNapariLabels(np.zeros((2, 2, 2), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        calls = []
        callback = inspector._make_click_callback(
            lambda *a: calls.append(a)
        )

        event = _FakeEvent()
        gen = callback(viewer, event)
        next(gen)
        event.type = "mouse_move"
        next(gen)
        event.type = "mouse_release"
        with contextlib.suppress(StopIteration):
            next(gen)

        assert calls == []


class TestClickHandlers:
    """Scope routing of the delete / relabel click tools."""

    def test_delete_click_on_background_reports(self, fake_labels_cls):
        """Clicking background says so instead of editing."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector._on_click_delete(None, 0, _FakeEvent())

        assert "background clicked" in viewer.status

    def test_delete_click_routes_to_the_clicked_timepoint(
        self, fake_labels_cls
    ):
        """``delete_scope='current'`` deletes only the clicked frame."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        layer = _ClickLabels(data, value=5, t=1.0)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.delete_scope = "current"

        inspector._on_click_delete(layer, 5, _FakeEvent())

        assert layer.data[1, 0, 0] == 0
        assert layer.data[0, 0, 0] == 5

    def test_delete_click_reports_an_unresolvable_timepoint(
        self, fake_labels_cls
    ):
        """A layer that cannot map the click yields a clear message."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        layer = _FakeLabels(data)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.delete_scope = "current"

        inspector._on_click_delete(layer, 5, _FakeEvent())

        assert "could not resolve the clicked timepoint" in viewer.status

    def test_pipette_on_background_picks_nothing(self, fake_labels_cls):
        """Ctrl+click on background does not change the selected ID."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        layer = _ClickLabels(data, selected_label=2)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector._on_click_relabel(
            layer, 0, _FakeEvent(modifiers=("Control",))
        )

        assert layer.selected_label == 2
        assert "no ID picked" in viewer.status

    def test_relabel_click_on_background_reports(self, fake_labels_cls):
        """A plain click on background relabels nothing."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        layer = _ClickLabels(data)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector._on_click_relabel(layer, 0, _FakeEvent())

        assert "nothing relabeled" in viewer.status

    def test_relabel_click_routes_to_the_clicked_timepoint(
        self, fake_labels_cls
    ):
        """``relabel_scope='current'`` rewrites only the clicked frame."""
        data = np.zeros((3, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        layer = _ClickLabels(data, value=5, t=2.0, selected_label=9)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.relabel_scope = "current"

        inspector._on_click_relabel(layer, 5, _FakeEvent())

        assert layer.data[2, 0, 0] == 9
        assert layer.data[0, 0, 0] == 5

    def test_relabel_click_reports_an_unresolvable_timepoint(
        self, fake_labels_cls
    ):
        """Same guard as delete, but on the relabel path."""
        layer = _FakeLabels(np.zeros((3, 4, 4), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.relabel_scope = "current"

        inspector._on_click_relabel(layer, 5, _FakeEvent())

        assert "could not resolve the clicked timepoint" in viewer.status


# ---------------------------------------------------------------------
# Click-to-split
# ---------------------------------------------------------------------
def _dumbbell(dtype=np.uint32):
    """(2, 8, 8) labels holding one dumbbell-shaped label 1 per frame."""
    data = np.zeros((2, 8, 8), dtype=dtype)
    data[:, 1:4, 1:4] = 1
    data[:, 1:4, 5:8] = 1
    data[:, 2, 4] = 1  # the constriction the watershed should cut
    return data


class _SplitLabels(_FakeLabels):
    """Labels layer whose ``world_to_data`` returns a fixed data coord."""

    def __init__(self, data, coord=(0, 2, 2), **kwargs):
        super().__init__(data, **kwargs)
        self.coord = coord

    def world_to_data(self, position):
        return np.array(self.coord, dtype=float)


class TestSplit:
    """Seeded-watershed split of an under-segmented label."""

    def test_split_without_a_layer(self, fake_labels_cls):
        """No labels layer means nothing to split."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 6)])

        assert viewer.status == "No labels layer found."

    def test_split_refuses_background(self, fake_labels_cls):
        """The background label cannot be split."""
        layer = _FakeLabels(_dumbbell())
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector.split_label_at_timepoint(0, 0, [(2, 2), (2, 6)])

        assert "non-background label" in inspector.viewer.status

    def test_split_needs_two_distinct_seeds(self, fake_labels_cls):
        """Duplicate seeds collapse to one and are refused."""
        layer = _FakeLabels(_dumbbell())
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 2)])

        assert "at least two distinct points" in inspector.viewer.status

    def test_split_refuses_a_seed_off_the_label(self, fake_labels_cls):
        """Every seed must land on the label being split."""
        layer = _FakeLabels(_dumbbell())
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (7, 7)])

        assert "one missed it" in inspector.viewer.status
        assert np.all(np.unique(layer.data) == np.array([0, 1]))

    def test_split_refuses_when_the_dtype_cannot_hold_a_new_id(
        self, fake_labels_cls
    ):
        """A uint8 label image cannot grow past 255."""
        layer = _FakeLabels(_dumbbell(dtype=np.uint8))
        inspector = LabelInspector(_FakeViewer([layer]))
        inspector._next_free_id = 256

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 6)])

        assert "cannot hold" in inspector.viewer.status
        assert set(np.unique(layer.data)) == {0, 1}

    def test_split_cuts_the_label_at_its_constriction(self, fake_labels_cls):
        """The second region gets a fresh globally-unique ID."""
        layer = _FakeLabels(_dumbbell())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 6)])

        frame = layer.data[0]
        assert frame[2, 2] == 1
        assert frame[2, 6] == 2  # global max was 1, so the new ID is 2
        # Only the clicked timepoint changed.
        assert set(np.unique(layer.data[1])) == {0, 1}
        assert inspector._single_t_last["t"] == 0
        assert "new label(s) 2" in viewer.status

    def test_split_undo_merges_the_regions_back(self, fake_labels_cls):
        """Ctrl+Z restores the original single label."""
        layer = _FakeLabels(_dumbbell())
        inspector = LabelInspector(_FakeViewer([layer]))
        before = layer.data.copy()
        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 6)])
        assert 2 in np.unique(layer.data)

        inspector._on_undo_key()

        assert np.array_equal(layer.data, before)

    def test_split_on_a_dask_wrapper(self, fake_labels_cls):
        """A raw dask array is wrapped and written as a sparse diff."""
        import dask.array as da

        base = _dumbbell()
        layer = _FakeLabels(da.from_array(base, chunks=(1, 8, 8)))
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 6)])

        assert isinstance(layer.data, _DaskFancyIndexWrapper)
        result = np.asarray(layer.data)
        assert result[0, 2, 6] == 2
        assert result[1, 2, 6] == 1

    def test_allocated_ids_are_monotonic(self, fake_labels_cls):
        """A second split never reuses the first split's ID."""
        layer = _FakeLabels(_dumbbell())
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector.split_label_at_timepoint(1, 0, [(2, 2), (2, 6)])
        inspector.split_label_at_timepoint(1, 1, [(2, 2), (2, 6)])

        assert layer.data[0, 2, 6] == 2
        assert layer.data[1, 2, 6] == 3
        assert inspector._next_free_id == 4


class TestSplitSeeds:
    """Seed placement, removal and the Apply button."""

    def test_seeds_accumulate_on_one_label_and_timepoint(
        self, fake_labels_cls
    ):
        """Each plain click adds one seed to the active set."""
        layer = _SplitLabels(_dumbbell(), coord=(0, 2, 2))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector._on_click_split(layer, 1, _FakeEvent())
        layer.coord = (0, 2, 6)
        inspector._on_click_split(layer, 1, _FakeEvent())

        assert inspector._split_seeds["label_id"] == 1
        assert inspector._split_seeds["t"] == 0
        assert inspector._split_seeds["coords"] == [(2, 2), (2, 6)]
        assert "2 seed(s)" in viewer.status

    def test_a_click_on_another_label_starts_a_fresh_set(
        self, fake_labels_cls
    ):
        """Seeds never straddle two labels."""
        layer = _SplitLabels(_dumbbell(), coord=(0, 2, 2))
        inspector = LabelInspector(_FakeViewer([layer]))
        inspector._on_click_split(layer, 1, _FakeEvent())

        layer.coord = (1, 2, 6)
        inspector._on_click_split(layer, 4, _FakeEvent())

        assert inspector._split_seeds["label_id"] == 4
        assert inspector._split_seeds["coords"] == [(2, 6)]

    def test_ctrl_click_removes_the_last_seed(self, fake_labels_cls):
        """Ctrl+click pops seeds and finally clears the set."""
        layer = _SplitLabels(_dumbbell(), coord=(0, 2, 2))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector._on_click_split(layer, 1, _FakeEvent())
        layer.coord = (0, 2, 6)
        inspector._on_click_split(layer, 1, _FakeEvent())

        ctrl = _FakeEvent(modifiers=("Control",))
        inspector._on_click_split(layer, 1, ctrl)
        assert inspector._split_seeds["coords"] == [(2, 2)]
        assert "removed last seed" in viewer.status

        inspector._on_click_split(layer, 1, ctrl)
        assert inspector._split_seeds is None
        assert viewer.status == "Split: all seeds removed."

        inspector._on_click_split(layer, 1, ctrl)
        assert viewer.status == "Split: no seed to remove."

    def test_background_click_places_no_seed(self, fake_labels_cls):
        """Clicking background is reported, not recorded."""
        layer = _SplitLabels(_dumbbell())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector._on_click_split(layer, 0, _FakeEvent())

        assert inspector._split_seeds is None
        assert "background clicked" in viewer.status

    def test_unresolvable_click_places_no_seed(self, fake_labels_cls):
        """2-D labels have no timepoint to key the split by."""
        layer = _SplitLabels(np.zeros((8, 8), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector._on_click_split(layer, 1, _FakeEvent())

        assert inspector._split_seeds is None
        assert "could not resolve the clicked voxel" in viewer.status

    def test_a_wrong_length_world_coord_is_refused(self, fake_labels_cls):
        """A coord that does not match the layer's ndim is discarded."""
        layer = _SplitLabels(_dumbbell(), coord=(0, 2, 2))
        inspector = LabelInspector(_FakeViewer([layer]))

        # Contrast: a well-formed coord really is resolved to (t, spatial),
        # so the ``None`` below is the guard firing, not a stub returning.
        assert inspector._click_data_coord(layer, 1, _FakeEvent()) == (
            0,
            (2, 2),
        )

        layer.coord = (0, 2)

        assert inspector._click_data_coord(layer, 1, _FakeEvent()) is None

    def test_a_failing_world_to_data_is_refused(self, fake_labels_cls):
        """An exception from the layer degrades to 'unresolved'."""

        class _Broken(_FakeLabels):
            def world_to_data(self, position):
                raise RuntimeError("no transform")

        layer = _Broken(_dumbbell())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        assert inspector._click_data_coord(layer, 1, _FakeEvent()) is None

        # ...and the caller degrades to a status instead of propagating it.
        inspector._on_click_split(layer, 1, _FakeEvent())

        assert inspector._split_seeds is None
        assert "could not resolve the clicked voxel" in viewer.status

    def test_split_is_blocked_while_a_track_view_is_active(
        self, fake_labels_cls
    ):
        """The projected views cannot supply exact source voxels."""
        layer = _SplitLabels(_dumbbell())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector._split_seeds = {
            "label_id": 1,
            "t": 0,
            "coords": [(2, 2), (2, 6)],
        }
        inspector._track_view_layer = object()

        inspector.commit_split()
        assert "normal frame view" in viewer.status
        assert set(np.unique(layer.data)) == {0, 1}
        assert inspector._split_seeds is not None  # seeds survive the refusal

        viewer.status = ""  # so the next assert cannot read a stale message
        inspector._on_click_split(layer, 1, _FakeEvent())
        assert inspector._split_seeds is None
        assert "normal frame view" in viewer.status

    def test_apply_needs_two_seeds(self, fake_labels_cls):
        """The Apply button explains what is missing."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.commit_split()

        assert "at least two points" in viewer.status

    def test_apply_runs_the_split_and_clears_the_seeds(self, fake_labels_cls):
        """Apply consumes the seeds exactly once."""
        layer = _SplitLabels(_dumbbell(), coord=(0, 2, 2))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector._on_click_split(layer, 1, _FakeEvent())
        layer.coord = (0, 2, 6)
        inspector._on_click_split(layer, 1, _FakeEvent())

        inspector.commit_split()

        assert inspector._split_seeds is None
        assert layer.data[0, 2, 6] == 2

    def test_enable_split_toggles_the_callback_and_undo_key(
        self, fake_labels_cls
    ):
        """Toggling registers/removes the mouse callback and Ctrl+Z."""
        layer = _SplitLabels(_dumbbell())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.enable_click_split(True)
        assert len(viewer.mouse_drag_callbacks) == 1
        assert viewer.bound_keys["Control-Z"] == inspector._on_undo_key
        assert layer.bound_keys["Control-Z"] == inspector._on_undo_key
        assert "Click-to-split ON" in viewer.status

        # Re-enabling is a no-op, not a second callback.
        inspector.enable_click_split(True)
        assert len(viewer.mouse_drag_callbacks) == 1

        inspector._split_seeds = {"label_id": 1, "t": 0, "coords": [(2, 2)]}
        inspector.enable_click_split(False)
        assert viewer.mouse_drag_callbacks == []
        assert inspector._split_seeds is None
        assert viewer.bound_keys["Control-Z"] is None
        assert viewer.status == "Click-to-split OFF."

    def test_enabling_split_turns_the_other_modes_off(self, fake_labels_cls):
        """The click tools are mutually exclusive."""
        layer = _SplitLabels(_dumbbell())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.enable_click_delete(True)
        inspector.enable_click_relabel(False)

        inspector.enable_click_split(True)

        assert inspector._click_delete_cb is None
        assert inspector._click_split_cb is not None
        assert len(viewer.mouse_drag_callbacks) == 1


# ---------------------------------------------------------------------
# Pick-ray helpers
# ---------------------------------------------------------------------
class _RayLabels(_FakeLabels):
    """Labels layer that reports a fixed ray through the volume."""

    def __init__(self, data, start=None, end=None, **kwargs):
        super().__init__(data, **kwargs)
        self.ndim = int(data.ndim)
        self._start = start
        self._end = end

    def world_to_data(self, position):
        return np.asarray(position, dtype=float)

    def get_ray_intersections(
        self, position, view_direction, dims_displayed, world=False
    ):
        if self._start is None:
            return None, None
        return (
            np.asarray(self._start, dtype=float),
            np.asarray(self._end, dtype=float),
        )


class TestRayHelpers:
    """Marching napari's pick ray back to the voxel it hit."""

    def _inspector(self, layer):
        return LabelInspector(_FakeViewer([layer]))

    def test_ray_points_march_the_whole_segment(self, fake_labels_cls):
        """Sampling is ~2 points per voxel, truncated and clipped."""
        data = np.zeros((4, 6, 6), dtype=np.uint32)
        layer = _RayLabels(data, start=(0, 1, 1), end=(3, 1, 1))
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 1.0, 1.0),
            view_direction=(1.0, 0.0, 0.0),
            dims_displayed=[0, 1, 2],
        )

        pts = inspector._ray_points(layer, event, [0, 1, 2])

        assert pts.shape == (6, 3)
        assert np.array_equal(pts[:, 0], np.array([0, 0, 1, 1, 2, 3]))
        assert np.all(pts[:, 1] == 1) and np.all(pts[:, 2] == 1)
        assert pts.max() < data.shape[0]

    def test_ray_that_misses_the_layer_returns_none(self, fake_labels_cls):
        """No intersection means no pick."""
        data = np.zeros((4, 6, 6), dtype=np.uint32)
        layer = _RayLabels(data)
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 1.0, 1.0),
            view_direction=(1.0, 0.0, 0.0),
            dims_displayed=[0, 1, 2],
        )

        assert inspector._ray_points(layer, event, [0, 1, 2]) is None
        assert inspector._ray_hit_voxel(layer, 7, event, [0, 1, 2]) is None
        assert inspector._ray_hit_plane(layer, 7, event, [0, 1, 2]) is None
        assert inspector._ray_signal_voxel(
            layer, event, [0, 1, 2], np.zeros((6, 6))
        ) == (None, 0)

    def test_ray_hit_voxel_finds_the_first_matching_voxel(
        self, fake_labels_cls
    ):
        """The voxel napari's get_value reported is recovered exactly."""
        data = np.zeros((4, 6, 6), dtype=np.uint32)
        data[2, 1, 1] = 7
        data[3, 1, 1] = 7
        layer = _RayLabels(data, start=(0, 1, 1), end=(3, 1, 1))
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 1.0, 1.0),
            view_direction=(1.0, 0.0, 0.0),
            dims_displayed=[0, 1, 2],
        )

        vox = inspector._ray_hit_voxel(layer, 7, event, [0, 1, 2])

        assert np.array_equal(vox, np.array([2, 1, 1]))
        assert inspector._ray_hit_plane(layer, 7, event, [0, 1, 2]) == 2

    def test_ray_hit_voxel_returns_none_when_the_label_is_absent(
        self, fake_labels_cls
    ):
        """A ray that never crosses the label yields no voxel."""
        data = np.zeros((4, 6, 6), dtype=np.uint32)
        layer = _RayLabels(data, start=(0, 1, 1), end=(3, 1, 1))
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 1.0, 1.0),
            view_direction=(1.0, 0.0, 0.0),
            dims_displayed=[0, 1, 2],
        )

        assert inspector._ray_hit_voxel(layer, 7, event, [0, 1, 2]) is None

    def test_clicked_timepoint_uses_the_ray_for_a_3d_layer(
        self, fake_labels_cls
    ):
        """A TYX layer in 3-D display resolves T from the ray."""
        data = np.zeros((4, 6, 6), dtype=np.uint32)
        data[2, 1, 1] = 7
        layer = _RayLabels(data, start=(0, 1, 1), end=(3, 1, 1))
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 1.0, 1.0),
            view_direction=(1.0, 0.0, 0.0),
            dims_displayed=[0, 1, 2],
        )

        assert inspector._clicked_timepoint(layer, 7, event) == 2

    def test_signal_voxel_is_the_brightest_unowned_sample(
        self, fake_labels_cls
    ):
        """The 3-D 'add' seed is the ray's brightest voxel."""
        data = np.zeros((2, 4, 6, 6), dtype=np.uint32)
        layer = _RayLabels(data, start=(0, 0, 1, 1), end=(0, 3, 1, 1))
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 0.0, 1.0, 1.0),
            view_direction=(0.0, 1.0, 0.0, 0.0),
            dims_displayed=[1, 2, 3],
        )
        raw_t = np.zeros((4, 6, 6), dtype=float)
        raw_t[2, 1, 1] = 100.0

        vox, blocker = inspector._ray_signal_voxel(
            layer, event, [1, 2, 3], raw_t
        )

        assert blocker == 0
        assert np.array_equal(vox, np.array([0, 2, 1, 1]))

    def test_signal_voxel_reports_the_label_that_owns_the_peak(
        self, fake_labels_cls
    ):
        """A labeled brightest voxel comes back as a blocker, not a seed."""
        data = np.zeros((2, 4, 6, 6), dtype=np.uint32)
        data[0, 2, 1, 1] = 12
        layer = _RayLabels(data, start=(0, 0, 1, 1), end=(0, 3, 1, 1))
        inspector = self._inspector(layer)
        event = _FakeEvent(
            position=(0.0, 0.0, 1.0, 1.0),
            view_direction=(0.0, 1.0, 0.0, 0.0),
            dims_displayed=[1, 2, 3],
        )
        raw_t = np.zeros((4, 6, 6), dtype=float)
        raw_t[2, 1, 1] = 100.0

        assert inspector._ray_signal_voxel(
            layer, event, [1, 2, 3], raw_t
        ) == (None, 12)


# ---------------------------------------------------------------------
# Merge touching neighbors
# ---------------------------------------------------------------------
def _two_neighbors():
    """(2, 6, 6) labels: 1 and 2 share a border, 5 stands apart."""
    data = np.zeros((2, 6, 6), dtype=np.uint32)
    data[:, 1:3, 1:3] = 1
    data[:, 1:3, 3:5] = 2
    data[:, 5, 5] = 5
    return data


class TestTouchingNeighbors:
    """Adjacency detection and the merge scopes."""

    def test_no_layer(self, fake_labels_cls):
        """Neighbors of nothing is an empty list plus a message."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        assert inspector._touching_neighbors(None, 1, 0) == []
        assert viewer.status == "No labels layer found."

    def test_background_is_refused(self, fake_labels_cls):
        """Label 0 cannot be a merge target."""
        layer = _FakeLabels(_two_neighbors())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        assert inspector._touching_neighbors(layer, 0, 0) == []
        assert "non-background label to merge into" in viewer.status

    def test_a_missing_label_reports_instead_of_merging(
        self, fake_labels_cls
    ):
        """A label absent at the timepoint has no neighbors."""
        data = _two_neighbors()
        data[1] = 0
        layer = _FakeLabels(data)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        assert inspector._touching_neighbors(layer, 1, 1) == []
        assert viewer.status == "Label 1 not present at timepoint 1."

    def test_isolated_label_has_no_neighbors(self, fake_labels_cls):
        """Nothing shares a border with the corner label."""
        layer = _FakeLabels(_two_neighbors())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        assert inspector._touching_neighbors(layer, 5, 0) == []
        assert "no touching neighbors" in viewer.status

    def test_neighbors_of_a_dask_layer_wrap_the_data(self, fake_labels_cls):
        """A raw dask array is wrapped before the slice read."""
        import dask.array as da

        layer = _FakeLabels(
            da.from_array(_two_neighbors(), chunks=(1, 6, 6))
        )
        inspector = LabelInspector(_FakeViewer([layer]))

        assert inspector._touching_neighbors(layer, 1, 0) == [2]
        assert isinstance(layer.data, _DaskFancyIndexWrapper)

    def test_neighbors_without_a_time_axis_use_the_whole_array(
        self, fake_labels_cls
    ):
        """2-D labels have one implicit timepoint."""
        layer = _FakeLabels(_two_neighbors()[0])
        inspector = LabelInspector(_FakeViewer([layer]))

        assert inspector._touching_neighbors(layer, 1, 0) == [2]

    def test_merge_all_timepoints_stops_without_neighbors(
        self, fake_labels_cls
    ):
        """An isolated label leaves the movie untouched."""
        layer = _FakeLabels(_two_neighbors())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        before = layer.data.copy()

        inspector.merge_neighbors_all_timepoints(5, 0)

        assert np.array_equal(layer.data, before)
        assert viewer.status == (
            "Label 5 has no touching neighbors at timepoint 0."
        )
        assert inspector._single_t_last is None
        assert layer.refresh_count == 0

    def test_merge_at_timepoint_without_a_time_axis(self, fake_labels_cls):
        """A 2-D merge reports the whole-image fallback."""
        layer = _FakeLabels(_two_neighbors()[0])
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.merge_neighbors_at_timepoint(1, 0)

        assert not np.any(layer.data == 2)
        assert "no time axis" in viewer.status
        assert inspector._single_t_last is None

    def test_merge_click_on_background(self, fake_labels_cls):
        """A background click merges nothing."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector._on_click_merge(None, 0, _FakeEvent())

        assert "background clicked" in viewer.status

    def test_merge_click_without_a_time_axis_uses_t0(self, fake_labels_cls):
        """2-D labels skip timepoint resolution entirely."""
        layer = _FakeLabels(_two_neighbors()[0])
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector._on_click_merge(layer, 1, _FakeEvent())

        assert not np.any(layer.data == 2)

    def test_merge_click_reports_an_unresolvable_timepoint(
        self, fake_labels_cls
    ):
        """A layer that cannot map the click is reported."""
        layer = _FakeLabels(_two_neighbors())
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector._on_click_merge(layer, 1, _FakeEvent())

        assert "could not resolve the clicked" in viewer.status

    def test_merge_click_routes_to_the_all_timepoints_scope(
        self, fake_labels_cls
    ):
        """``merge_scope='all'`` repairs the whole track in one click."""
        layer = _ClickLabels(_two_neighbors(), value=1, t=0.0)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.merge_scope = "all"

        inspector._on_click_merge(layer, 1, _FakeEvent())

        assert not np.any(layer.data == 2)
        assert np.all(layer.data[:, 1:3, 1:5] == 1)
        assert "on all timepoints" in viewer.status


# ---------------------------------------------------------------------
# All-timepoint guards, low-intensity undo, ID allocation
# ---------------------------------------------------------------------
class TestAllTimepointGuards:
    """Guards shared by the track-wide edit paths."""

    def test_delete_all_timepoints_without_a_layer(self, fake_labels_cls):
        """Nothing loaded, nothing deleted."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.delete_label_all_timepoints(5)

        assert viewer.status == "No labels layer found."

    def test_delete_all_timepoints_refuses_background(self, fake_labels_cls):
        """Label 0 is never deleted."""
        layer = _FakeLabels(np.zeros((2, 4, 4), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.delete_label_all_timepoints(0)

        assert viewer.status == "Select a non-background label first."

    def test_relabel_all_timepoints_without_a_layer(self, fake_labels_cls):
        """Nothing loaded, nothing relabeled."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.relabel_label_all_timepoints(5, 3)

        assert viewer.status == "No labels layer found."

    def test_a_raw_dask_layer_is_wrapped_for_the_lut_path(
        self, fake_labels_cls
    ):
        """An unwrapped dask array gains the edit wrapper on first remap."""
        import dask.array as da

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        layer = _FakeLabels(da.from_array(base, chunks=(1, 4, 4)))
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector.delete_label_all_timepoints(5)

        assert isinstance(layer.data, _DaskFancyIndexWrapper)
        assert not np.any(np.asarray(layer.data) == 5)

    def test_global_max_id_of_a_dask_layer(self, fake_labels_cls):
        """The allocator's seed is read from the lazy array."""
        import dask.array as da

        base = np.zeros((2, 4, 4), dtype=np.uint32)
        base[0, 0, 0] = 12
        layer = _FakeLabels(da.from_array(base, chunks=(1, 4, 4)))
        inspector = LabelInspector(_FakeViewer([layer]))

        assert inspector._global_max_id(layer) == 12

    def test_allocate_new_id_seeds_itself_once(self, fake_labels_cls):
        """The first allocation scans, later ones only count up."""
        data = np.zeros((2, 4, 4), dtype=np.uint32)
        data[0, 0, 0] = 7
        layer = _FakeLabels(data)
        inspector = LabelInspector(_FakeViewer([layer]))

        assert inspector._allocate_new_id(layer) == 8
        assert inspector._allocate_new_id(layer) == 9
        assert inspector._next_free_id == 10


class TestLowIntensityUndo:
    """Preview / restore bookkeeping of the low-intensity tool."""

    def test_apply_then_undo_restores_numpy_labels(self, fake_labels_cls):
        """The snapshot taken on apply is written back on undo."""
        data = np.zeros((2, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        layer = _FakeLabels(data)
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector._apply_low_intensity(layer, {5: 0})
        assert not np.any(layer.data == 5)

        inspector._undo_low_intensity(layer)

        assert np.all(layer.data[:, 0, 0] == 5)
        assert inspector._low_intensity_last is None

    def test_undo_without_a_pending_preview_leaves_the_labels_alone(
        self, fake_labels_cls
    ):
        """Nothing recorded means nothing to reverse."""
        data = np.zeros((2, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        layer = _FakeLabels(data)
        inspector = LabelInspector(_FakeViewer([layer]))
        assert inspector._low_intensity_last is None

        inspector._undo_low_intensity(layer)

        # A body that restored unconditionally would trip both of these.
        assert np.all(layer.data[:, 0, 0] == 5)
        assert layer.refresh_count == 0

    def test_a_later_edit_cancels_the_pending_preview(self, fake_labels_cls):
        """Another tool's edit must make the preview un-undoable.

        The preview's undo is a whole-array snapshot taken *before* the
        newer edit, so replaying it after one would silently resurrect
        what the newer edit removed.
        """
        data = np.zeros((2, 4, 4), dtype=np.uint32)
        data[:, 0, 0] = 5
        data[:, 1, 1] = 7
        layer = _FakeLabels(data)
        inspector = LabelInspector(_FakeViewer([layer]))

        inspector._apply_low_intensity(layer, {5: 0})
        assert inspector._low_intensity_last is not None

        inspector.delete_label_all_timepoints(7)
        assert inspector._low_intensity_last is None

        inspector._undo_low_intensity(layer)

        # 7 stays deleted — the stale snapshot was never replayed.
        assert not np.any(layer.data == 7)
        assert not np.any(layer.data == 5)

    def test_undo_without_a_layer_drops_the_record(self, fake_labels_cls):
        """A vanished layer clears the pending preview instead of raising."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector._low_intensity_last = {"mapping": {5: 0}, "backup": None}

        inspector._undo_low_intensity()

        assert inspector._low_intensity_last is None

    def test_delete_low_intensity_without_a_layer(self, fake_labels_cls):
        """The Apply button reports a missing labels layer."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.delete_low_intensity_tracks(0.5)

        assert viewer.status == "No labels layer found."

    def test_delete_low_intensity_without_a_pair(self, fake_labels_cls):
        """Without a loaded pair there is no raw image to measure."""
        layer = _FakeLabels(np.zeros((2, 4, 4), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.delete_low_intensity_tracks(0.5)

        assert viewer.status == "No image-label pair loaded."


# ---------------------------------------------------------------------
# Track-view plumbing
# ---------------------------------------------------------------------
class _RedrawLayer:
    """Stand-in for a napari Labels layer with a live GPU slice."""

    contour = 0

    def __init__(self, vol):
        self.data = SimpleNamespace(_vol=vol)
        self._slice = SimpleNamespace(
            image=SimpleNamespace(raw=vol, view=np.zeros_like(vol))
        )
        self.colormap = SimpleNamespace(_data_to_texture=lambda arr: arr)
        self._updated_slice = None
        self.partial_refreshes = 0

    def _partial_labels_refresh(self):
        self.partial_refreshes += 1


class TestTrackViewPlumbing:
    """Building, refreshing and partially redrawing the track view."""

    def test_an_unknown_mode_falls_back_to_off(self, fake_labels_cls):
        """Only off / stack / max exist."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.set_track_view_mode("nonsense")

        assert inspector.track_view_mode == "off"
        assert viewer.status == "Track view off."

    def test_track_view_needs_a_3d_or_4d_labels_layer(self, fake_labels_cls):
        """A 2-D labels layer cannot be stacked into tracks."""
        layer = _FakeLabels(np.zeros((4, 4), dtype=np.uint32))
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.set_track_view_mode("stack")

        assert "needs a loaded 3-D" in viewer.status
        assert inspector._track_view_layer is None

    def test_stacked_view_over_a_tyx_movie(self, fake_labels_cls):
        """A TYX movie stacks to one plane per timepoint."""
        data = np.zeros((3, 6, 6), dtype=np.uint32)
        data[1, 2, 2] = 4
        layer = _FakeLabels(data, scale=[1.0, 2.0, 2.0])
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.set_track_view_mode("stack")

        view_layer = inspector._track_view_layer
        assert view_layer is not None
        assert view_layer.data.shape == (3, 6, 6)
        assert view_layer.scale == [1.0, 2.0, 2.0]
        # One plane per timepoint, holding that timepoint's labels.
        assert view_layer.data[1, 2, 2] == 4
        assert view_layer.data[0, 2, 2] == 0
        assert np.array_equal(np.asarray(view_layer.data[:]), data)
        assert layer.visible is False
        assert "Track view ON (T stacked along Z, 3 planes)" in viewer.status

        inspector.set_track_view_mode("off")
        assert inspector._track_view_layer is None
        assert layer.visible is True

    def test_a_broken_layer_scale_degrades_to_unit_spacing(
        self, fake_labels_cls
    ):
        """An unreadable scale must not stop the view from building."""
        data = np.zeros((3, 6, 6), dtype=np.uint32)
        data[1, 2, 2] = 4
        layer = _FakeLabels(data)
        layer.scale = None
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)

        inspector.set_track_view_mode("max")

        view = inspector._track_view_layer
        assert view.scale == [1.0, 1.0, 1.0]
        # A projected view is never paintable.
        assert view.editable is False
        # A TYX movie has an implicit Z of 1, so the projection is a
        # pass-through: one plane per timepoint, IDs unchanged.
        assert view.data.shape == (3, 6, 6)
        assert np.array_equal(np.asarray(view.data[:]), data)

    def test_refresh_only_ever_touches_the_track_view_layer(
        self, fake_labels_cls
    ):
        """Without a view nothing is redrawn; with one, only it is."""
        labels = _FakeLabels(np.zeros((3, 6, 6), dtype=np.uint32))
        viewer = _FakeViewer([labels])
        inspector = LabelInspector(viewer)

        inspector._refresh_track_view({1: 0})

        assert labels.refresh_count == 0

        # A view layer whose data is not a _TrackView cannot be patched
        # in place, so the call falls back to a full layer refresh.
        view = _FakeLabels(np.zeros((3, 6, 6), dtype=np.uint32))
        inspector._track_view_layer = view

        inspector._refresh_track_view({1: 0})

        assert view.refresh_count == 1
        assert labels.refresh_count == 0

    def test_refresh_by_timepoint_and_wholesale(self, fake_labels_cls):
        """Both refresh flavours reach the view's own caches."""
        data = np.zeros((3, 6, 6), dtype=np.uint32)
        data[1, 2, 2] = 4
        layer = _FakeLabels(data)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.set_track_view_mode("stack")
        view_layer = inspector._track_view_layer
        assert view_layer.data[1, 2, 2] == 4

        data[1, 2, 2] = 6
        inspector._refresh_track_view(timepoint=1)
        assert view_layer.data[1, 2, 2] == 6

        data[1, 2, 2] = 8
        inspector._refresh_track_view()
        assert view_layer.data[1, 2, 2] == 8

    def test_partial_redraw_is_skipped_outside_3d_display(
        self, fake_labels_cls
    ):
        """In 2-D slicing a full refresh is already cheap."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        vol = np.zeros((2, 4, 4), dtype=np.uint32)
        region = (slice(0, 1), slice(0, 2), slice(0, 2))

        assert inspector._partial_track_redraw(_RedrawLayer(vol), region) is False

    def test_partial_redraw_is_skipped_for_contours(self, fake_labels_cls):
        """Contour rendering spills outside the changed box."""
        viewer = _FakeViewer([])
        viewer.dims.ndisplay = 3
        inspector = LabelInspector(viewer)
        vol = np.zeros((2, 4, 4), dtype=np.uint32)
        layer = _RedrawLayer(vol)
        layer.contour = 1

        assert (
            inspector._partial_track_redraw(
                layer, (slice(0, 1), slice(0, 2), slice(0, 2))
            )
            is False
        )

    def test_an_empty_region_needs_no_redraw(self, fake_labels_cls):
        """An edit that matched nothing is already up to date."""
        viewer = _FakeViewer([])
        viewer.dims.ndisplay = 3
        inspector = LabelInspector(viewer)
        vol = np.zeros((2, 4, 4), dtype=np.uint32)

        assert (
            inspector._partial_track_redraw(_RedrawLayer(vol), mod._EMPTY_REGION)
            is True
        )

    def test_partial_redraw_needs_the_displayed_buffer(self, fake_labels_cls):
        """A view whose data is not the drawn volume falls back."""
        viewer = _FakeViewer([])
        viewer.dims.ndisplay = 3
        inspector = LabelInspector(viewer)
        vol = np.zeros((2, 4, 4), dtype=np.uint32)
        layer = _RedrawLayer(vol)
        layer.data = SimpleNamespace(_vol=None)

        assert (
            inspector._partial_track_redraw(
                layer, (slice(0, 1), slice(0, 2), slice(0, 2))
            )
            is False
        )

    def test_partial_redraw_pushes_only_the_changed_box(
        self, fake_labels_cls
    ):
        """The texture update is confined to *region*."""
        viewer = _FakeViewer([])
        viewer.dims.ndisplay = 3
        inspector = LabelInspector(viewer)
        vol = np.zeros((2, 4, 4), dtype=np.uint32)
        vol[0, 0:2, 0:2] = 9
        layer = _RedrawLayer(vol)
        region = (slice(0, 1), slice(0, 2), slice(0, 2))

        assert inspector._partial_track_redraw(layer, region) is True

        view = layer._slice.image.view
        assert np.all(view[region] == 9)
        assert view.sum() == 9 * 4  # nothing outside the box was touched
        assert layer._updated_slice == region
        assert layer.partial_refreshes == 1


# ---------------------------------------------------------------------
# Pair discovery, loading, saving and advancing
# ---------------------------------------------------------------------
def _write_pair(folder, stem, image, label):
    """Write one ``<stem>.tif`` / ``<stem>_labels.tif`` pair."""
    import tifffile

    image_path = folder / f"{stem}.tif"
    label_path = folder / f"{stem}_labels.tif"
    # minisblack: a size-3 axis must not be read back as RGB samples.
    tifffile.imwrite(str(image_path), image, photometric="minisblack")
    tifffile.imwrite(str(label_path), label, photometric="minisblack")
    return str(image_path), str(label_path)


class TestPairLoading:
    """Folder scanning and the per-pair viewer setup."""

    def test_a_float_label_file_is_reported_not_loaded(
        self, tmp_path, fake_labels_cls
    ):
        """Only integer label images are accepted."""
        import tifffile

        _write_pair(
            tmp_path,
            "b",
            np.zeros((8, 8), dtype=np.uint16),
            np.zeros((8, 8), dtype=np.float32),
        )
        tifffile.imwrite(
            str(tmp_path / "z_labels.tif"), np.zeros((4, 4), dtype=np.uint32)
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.load_image_label_pairs(str(tmp_path), "_labels.tif")

        assert inspector.image_label_pairs == []
        assert viewer.status == "No valid image-label pairs found."

    def test_loading_scales_the_label_to_the_raw_extent(
        self, tmp_path, fake_labels_cls
    ):
        """A label with more axes than the raw pads its scale with ones."""
        _write_pair(
            tmp_path,
            "a",
            np.zeros((8, 8), dtype=np.uint16),
            np.zeros((2, 4, 4), dtype=np.uint32),
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.load_image_label_pairs(str(tmp_path), "_labels.tif")

        assert len(inspector.image_label_pairs) == 1
        _layer, kwargs = viewer.added_labels[0]
        assert kwargs["scale"] == [1.0, 2.0, 2.0]
        assert "Viewing pair 1 of 1" in viewer.status

    def test_loading_honours_the_channel_axis_and_active_modes(
        self, tmp_path, fake_labels_cls
    ):
        """A forced channel axis splits the raw and rebinds the tools."""
        _write_pair(
            tmp_path,
            "a",
            np.zeros((2, 3, 8, 8), dtype=np.uint16),
            np.zeros((2, 8, 8), dtype=np.uint32),
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.channel_axis_override = "1"
        inspector.enable_click_delete(True)
        inspector.track_view_mode = "stack"

        inspector.load_image_label_pairs(str(tmp_path), "_labels.tif")

        _data, img_kwargs = viewer.added_images[0]
        assert img_kwargs["channel_axis"] == 1
        labels_layer, kwargs = viewer.added_labels[0]
        assert kwargs["scale"] == [1.0, 1.0, 1.0]
        assert labels_layer.bound_keys["Control-Z"] == inspector._on_undo_key
        assert inspector._track_view_layer is not None
        assert "channel_axis=1" in viewer.status

    def test_the_channel_axis_override_is_honoured_and_validated(
        self, fake_labels_cls
    ):
        """A numeric override wins; junk and out-of-range degrade to None."""
        inspector = LabelInspector(_FakeViewer([]))
        image = np.zeros((2, 3, 4, 4))
        label = np.zeros((2, 4, 4))

        def _axis():
            return inspector._resolve_channel_axis(image, label, "x.tif")

        inspector.channel_axis_override = "1"
        assert _axis() == 1
        inspector.channel_axis_override = "0"
        assert _axis() == 0

        inspector.channel_axis_override = "none"
        assert _axis() is None
        inspector.channel_axis_override = "9"  # past the image's ndim
        assert _axis() is None
        inspector.channel_axis_override = "not-a-number"
        assert _axis() is None

    def test_load_current_pair_without_pairs(self, fake_labels_cls):
        """Reloading with an empty list is reported, not an IndexError."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector._load_current_pair()

        assert viewer.status == "No pairs to inspect."


class TestSaveAndAdvance:
    """``save_current_labels`` and the pair cursor."""

    def test_saving_without_pairs(self, fake_labels_cls):
        """Nothing loaded, nothing written."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        inspector.save_current_labels()

        assert viewer.status == "No pairs to save."

    def test_saving_without_a_labels_layer(self, tmp_path, fake_labels_cls):
        """A cleared viewer has nothing to write back."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.image_label_pairs = [("img.tif", str(tmp_path / "l.tif"))]

        inspector.save_current_labels()

        assert viewer.status == "No labels found."

    def test_saving_a_wrapper_writes_the_edited_slices(
        self, tmp_path, fake_labels_cls
    ):
        """Wrapper-backed labels go through the streaming T-by-T writer."""
        import tifffile

        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        image_path, label_path = _write_pair(
            tmp_path, "a", np.zeros((3, 4, 4), dtype=np.uint16), base
        )
        wrapper = _wrap(base.copy())
        wrapper.remap_values({5: 9})
        layer = _FakeLabels(wrapper)
        viewer = _FakeViewer([layer])
        inspector = LabelInspector(viewer)
        inspector.image_label_pairs = [(image_path, label_path)]

        inspector.save_current_labels()

        written = tifffile.imread(label_path)
        expected = np.zeros((3, 4, 4), dtype=np.uint32)
        expected[:, 0, 0] = 9  # the file held 5 before the save
        assert written.dtype == np.uint32
        assert np.array_equal(written, expected)
        assert f"Saved labels to {label_path}" in viewer.status
        # Saving is the undo barrier: the edit state is committed.
        assert wrapper._op_log == []
        assert wrapper._lut == {}
        assert wrapper._diffs == {}
        assert wrapper.undo_remap() is None
        # The atomic write left no temporary behind.
        assert list(tmp_path.glob(".tmp_save_*")) == []

    def test_saving_a_2d_wrapper_uses_the_one_shot_write(self, tmp_path):
        """Small 2-D label images are written in a single call.

        The file starts out holding the *unedited* array, so the assertion
        below can only pass if the writer actually materialised the
        wrapper's LUT and pending diffs.
        """
        import tifffile

        base = np.zeros((4, 4), dtype=np.uint32)
        base[0, 0] = 5
        path = str(tmp_path / "flat.tif")
        tifffile.imwrite(path, base)
        wrapper = _wrap(base.copy())
        wrapper.remap_values({5: 9})  # a LUT edit...
        wrapper[2, 2] = 7  # ...and a pending local diff

        mod._save_label_wrapper(wrapper, path)

        expected = np.zeros((4, 4), dtype=np.uint32)
        expected[0, 0] = 9
        expected[2, 2] = 7
        written = tifffile.imread(path)
        assert written.dtype == np.uint32
        assert np.array_equal(written, expected)
        # ndim < 3 takes the single-shot branch: one page, one 2-D series.
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 1
            assert tif.series[0].shape == (4, 4)
        assert list(tmp_path.glob(".tmp_save_*")) == []
        # The save is the undo barrier: edit state is committed, not kept.
        assert wrapper._op_log == []
        assert wrapper._lut == {}
        assert wrapper._diffs == {}

    def test_a_failed_write_leaves_no_temporary_file(
        self, tmp_path, monkeypatch
    ):
        """The atomic write cleans up and never truncates the source."""
        import tifffile

        base = np.zeros((4, 4), dtype=np.uint32)
        base[0, 0] = 5
        path = str(tmp_path / "flat.tif")
        tifffile.imwrite(path, base)
        wrapper = _wrap(base.copy())
        wrapper.remap_values({5: 9})

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(tifffile, "imwrite", _boom)

        with pytest.raises(OSError, match="disk full"):
            mod._save_label_wrapper(wrapper, path)

        leftovers = list(tmp_path.glob(".tmp_save_*"))
        assert leftovers == []
        # The source file the wrapper still reads from is intact...
        monkeypatch.undo()
        assert np.array_equal(tifffile.imread(path), base)
        # ...and the failed save did NOT commit the edit, so the pending
        # remap is still undoable and still visible through the wrapper.
        assert wrapper._op_log
        assert wrapper._lut == {5: 9}
        assert np.asarray(wrapper)[0, 0] == 9

    def test_proceed_without_pairs(self, fake_labels_cls):
        """Advancing an empty session is reported."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        assert inspector._proceed(save=False) is None
        assert viewer.status == "No pairs to inspect."

    def test_skip_pair_advances_without_saving(
        self, tmp_path, fake_labels_cls
    ):
        """Skip discards in-memory edits and loads the next pair."""
        import tifffile

        _write_pair(
            tmp_path,
            "a",
            np.zeros((4, 4), dtype=np.uint16),
            np.zeros((4, 4), dtype=np.uint32),
        )
        _write_pair(
            tmp_path,
            "b",
            np.zeros((4, 4), dtype=np.uint16),
            np.zeros((4, 4), dtype=np.uint32),
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.load_image_label_pairs(str(tmp_path), "_labels.tif")
        assert len(inspector.image_label_pairs) == 2
        first_label = inspector.image_label_pairs[0][1]
        # An unsaved edit that must NOT reach the file.
        viewer.added_labels[-1][0].data[0, 0] = 7

        assert inspector.skip_pair() is True

        assert inspector.current_index == 1
        assert not np.any(tifffile.imread(first_label) == 7)

        # Already on the last pair: nothing left to advance to.
        assert inspector.skip_pair() is False
        assert "Inspection complete" in viewer.status
        assert viewer.layers == []


class TestRawSliceAt:
    """Single-timepoint raw reads used by the SAM2 tools."""

    def test_without_a_pair(self, fake_labels_cls):
        """No raw image loaded means no slice."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)

        assert inspector._raw_slice_at(np.zeros((2, 4, 4)), 0) is None
        assert viewer.status == "No raw image loaded."

    def test_an_unreadable_raw_is_reported(self, tmp_path, fake_labels_cls):
        """A missing raw file yields a status, not a traceback."""
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.image_label_pairs = [
            (str(tmp_path / "gone.tif"), str(tmp_path / "gone_labels.tif"))
        ]

        assert inspector._raw_slice_at(np.zeros((2, 4, 4)), 0) is None
        assert viewer.status.startswith("Could not load raw image:")

    def test_channel_selection_and_mean(self, tmp_path, fake_labels_cls):
        """'mean' averages the channels; an index picks one."""
        raw = np.zeros((2, 3, 8, 8), dtype=np.uint16)
        raw[:, 1] = 6
        image_path, label_path = _write_pair(
            tmp_path, "a", raw, np.zeros((2, 8, 8), dtype=np.uint32)
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.channel_axis_override = "1"
        inspector.image_label_pairs = [(image_path, label_path)]
        labels = np.zeros((2, 8, 8), dtype=np.uint32)

        mean_slice = inspector._raw_slice_at(labels, 0, channel="mean")
        one_slice = inspector._raw_slice_at(labels, 0, channel="1")

        assert mean_slice.shape == (8, 8)
        assert np.allclose(mean_slice, 2.0)
        assert np.all(one_slice == 6)

    def test_a_misaligned_raw_is_reported(self, tmp_path, fake_labels_cls):
        """Shapes that cannot be squeezed into alignment are refused."""
        image_path, label_path = _write_pair(
            tmp_path,
            "a",
            np.zeros((2, 3, 8, 8), dtype=np.uint16),
            np.zeros((2, 8, 8), dtype=np.uint32),
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.channel_axis_override = "1"
        inspector.image_label_pairs = [(image_path, label_path)]

        out = inspector._raw_slice_at(np.zeros((8, 8)), None)

        assert out is None
        assert "does not align with label shape" in viewer.status

    def test_a_leading_singleton_axis_is_squeezed(
        self, tmp_path, fake_labels_cls
    ):
        """A stray T=1 axis is dropped so 2-D labels still align."""
        raw = np.full((1, 8, 8), 4, dtype=np.uint16)
        image_path, label_path = _write_pair(
            tmp_path, "a", raw, np.zeros((8, 8), dtype=np.uint32)
        )
        viewer = _FakeViewer([])
        inspector = LabelInspector(viewer)
        inspector.channel_axis_override = "none"
        inspector.image_label_pairs = [(image_path, label_path)]

        out = inspector._raw_slice_at(np.zeros((8, 8), dtype=np.uint32), None)

        assert out.shape == (8, 8)
        assert np.all(out == 4)


class TestMessageBoxes:
    """Dialog suppression rules (no modal may ever block a test run)."""

    def test_dialogs_are_suppressed_under_pytest(
        self, monkeypatch, fake_labels_cls
    ):
        """``PYTEST_CURRENT_TEST`` alone disables every dialog."""
        from qtpy.QtWidgets import QApplication

        # Satisfy the other two conditions so only the env var can be
        # what suppresses the dialog.
        assert QApplication.instance() is not None
        viewer = _FakeViewer([])
        viewer.window = object()
        inspector = LabelInspector(viewer)
        assert "PYTEST_CURRENT_TEST" in os.environ

        assert inspector._can_show_message() is False

        # Drop the marker: the very same inspector would now show dialogs,
        # which is what proves the marker is what turned them off.
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        assert inspector._can_show_message() is True

        # A mock viewer (no .window) is still refused.
        del viewer.window
        assert inspector._can_show_message() is False

    def test_show_message_routes_warning_and_info(
        self, monkeypatch, fake_labels_cls
    ):
        """When allowed, the level chooses the QMessageBox method."""
        calls = []

        class _Box:
            @staticmethod
            def warning(parent, title, text):
                calls.append(("warning", title, text))

            @staticmethod
            def information(parent, title, text):
                calls.append(("info", title, text))

        monkeypatch.setattr(mod, "QMessageBox", _Box)
        inspector = LabelInspector(_FakeViewer([]))
        monkeypatch.setattr(inspector, "_can_show_message", lambda: True)

        inspector._show_message("warning", "T1", "body")
        inspector._show_message("info", "T2", "body")

        assert calls == [
            ("warning", "T1", "body"),
            ("info", "T2", "body"),
        ]

    def test_a_gui_failure_never_escapes(self, monkeypatch, fake_labels_cls):
        """RuntimeError from Qt is swallowed rather than crashing a load."""

        calls = []

        class _Box:
            @staticmethod
            def warning(parent, title, text):
                calls.append((title, text))
                raise RuntimeError("no display")

            @staticmethod
            def information(parent, title, text):
                raise AssertionError("a warning must not become an info box")

        monkeypatch.setattr(mod, "QMessageBox", _Box)
        inspector = LabelInspector(_FakeViewer([]))
        monkeypatch.setattr(inspector, "_can_show_message", lambda: True)

        assert inspector._show_message("warning", "T", "body") is None
        # The dialog really was attempted — the failure came from Qt.
        assert calls == [("T", "body")]

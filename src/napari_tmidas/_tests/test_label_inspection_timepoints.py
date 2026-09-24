"""Further coverage for :mod:`napari_tmidas._label_inspection`.

Complements ``test_label_inspection.py`` and
``test_label_inspection_coverage.py`` by pinning branches neither reaches:
the SAM2 grow / add tools on 2-D and dask-backed labels and their refusal
paths, the click-to-add dispatcher, the track views' cache and bbox
bookkeeping, the per-track intensity measurement's edge cases, the loader
fall-backs, and the dock widgets' Skip / mode / split / track-view wiring.
The SAM2 model is always faked.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import napari_tmidas._label_inspection as mod
from napari_tmidas._label_inspection import (
    LabelInspector,
    _bbox_union,
    _DaskFancyIndexWrapper,
    _MaxProjTrackView,
    _pick_track_view_step,
    _StackedTrackView,
    _TrackView,
)
from napari_tmidas._sam2_worker import Sam2Unavailable


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------
class _FakeLabels:
    """Minimal stand-in for a napari ``Labels`` layer."""

    def __init__(self, data):
        self.data = data
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1

    def bind_key(self, *args, **kwargs):
        pass


class _ThresholdWorker:
    """SAM2 stand-in: the bright part of the crop is the winning mask."""

    def __init__(self):
        self.calls = 0
        self.last = None

    def segment(self, image, coords, labels):
        self.calls += 1
        good = image[:, :, 0] >= 128
        tiny = np.zeros_like(good)
        tiny[:2, :2] = True
        self.last = np.stack([tiny, good])
        return self.last, np.array([0.9, 0.8], dtype=np.float32)

    def refine(self, index, coords, labels):
        return self.last[index], 0.95


class _EmptyWorker(_ThresholdWorker):
    """SAM2 stand-in that finds nothing at all."""

    def segment(self, image, coords, labels):
        self.calls += 1
        empty = np.zeros(image.shape[:2], dtype=bool)
        self.last = np.stack([empty, empty])
        return self.last, np.array([0.9, 0.8], dtype=np.float32)


class _BrokenWorker:
    """SAM2 stand-in whose subprocess dies mid-request."""

    def segment(self, image, coords, labels):
        raise Sam2Unavailable("SAM2 worker exited unexpectedly.")

    def refine(self, index, coords, labels):  # pragma: no cover
        raise AssertionError("never reached")


def _disc(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def _viewer():
    return SimpleNamespace(
        layers=[],
        status="",
        mouse_drag_callbacks=[],
        dims=SimpleNamespace(ndisplay=2),
        bind_key=lambda *a, **k: None,
    )


@pytest.fixture()
def sam2(monkeypatch):
    """Wire an inspector to a fake raw image and a fake SAM2 worker.

    Returns ``make(labels, raw, worker)`` → ``(inspector, layer)``.
    """
    monkeypatch.setattr(mod, "Labels", _FakeLabels)

    def make(labels, raw, worker=None):
        viewer = _viewer()
        layer = _FakeLabels(labels)
        viewer.layers = [layer]
        inspector = LabelInspector(viewer)
        inspector.image_label_pairs = [("raw.tif", "raw_labels.tif")]
        monkeypatch.setattr(mod, "_load_image", lambda _p: raw)
        monkeypatch.setattr(
            "napari_tmidas._sam2_worker.Sam2Worker.instance",
            lambda *a, **k: worker or _ThresholdWorker(),
        )
        return inspector, layer

    return make


def _half_disc_pair_2d():
    """A YX raw with a bright disc and a label covering its left half."""
    disc = _disc((21, 21), (10, 10), 6)
    raw = np.full((21, 21), 10, dtype=np.uint16)
    raw[disc] = 200
    labels = np.zeros((21, 21), dtype=np.uint32)
    half = disc.copy()
    half[:, 11:] = False
    labels[half] = 1
    return raw, labels, disc


def _half_disc_pair_tzyx():
    raw2, lab2, disc = _half_disc_pair_2d()
    raw = np.stack([raw2[None], raw2[None]])  # (T=2, Z=1, Y, X)
    labels = np.zeros((2, 1, 21, 21), dtype=np.uint32)
    labels[0, 0] = lab2
    return raw, labels, disc


# ---------------------------------------------------------------------
# grow_label_at_timepoint
# ---------------------------------------------------------------------
class TestGrowEdges:
    def test_no_labels_layer(self, monkeypatch):
        monkeypatch.setattr(mod, "Labels", _FakeLabels)
        inspector = LabelInspector(_viewer())
        inspector.grow_label_at_timepoint(1, 0)
        assert inspector.viewer.status == "No labels layer found."

    def test_background_is_refused(self, sam2):
        raw, labels, _ = _half_disc_pair_2d()
        before = labels.copy()
        inspector, _ = sam2(labels, raw)
        inspector.grow_label_at_timepoint(0, 0)
        assert "non-background label" in inspector.viewer.status
        np.testing.assert_array_equal(labels, before)

    def test_grow_on_2d_labels_without_a_time_axis(self, sam2):
        raw, labels, disc = _half_disc_pair_2d()
        inspector, layer = sam2(labels, raw)

        inspector.grow_label_at_timepoint(1, 7)  # t is ignored without T

        assert labels[10, 14] == 1  # the unlabeled right half joined
        assert (labels[disc] == 1).all()
        assert not labels[~disc].any()
        assert layer.refresh_count == 1
        record = inspector._single_t_last
        assert record["t"] is None
        assert "grow undone" in record["desc"]
        assert "timepoint" not in inspector.viewer.status
        assert inspector.viewer.status.startswith("Grew label 1 by ")

    def test_grow_wraps_a_raw_dask_layer(self, sam2):
        import dask.array as da

        raw, labels, disc = _half_disc_pair_tzyx()
        base = labels.copy()
        inspector, layer = sam2(
            da.from_array(labels, chunks=(1, 1, 21, 21)), raw
        )

        inspector.grow_label_at_timepoint(1, 0)

        assert isinstance(layer.data, _DaskFancyIndexWrapper)
        t0 = np.asarray(layer.data[0])
        assert (t0[0][disc] == 1).all()
        assert not np.asarray(layer.data[1]).any()
        # The edit is staged in the wrapper, never written into the base.
        np.testing.assert_array_equal(labels, base)
        assert "timepoint 0" in inspector.viewer.status

    def test_missing_label_without_time_axis(self, sam2):
        raw, labels, _ = _half_disc_pair_2d()
        inspector, _ = sam2(labels, raw)
        inspector.grow_label_at_timepoint(9, 0)
        assert inspector.viewer.status == "Label 9 is not present."

    def test_missing_label_at_a_timepoint(self, sam2):
        raw, labels, _ = _half_disc_pair_tzyx()
        inspector, _ = sam2(labels, raw)
        inspector.grow_label_at_timepoint(1, 1)
        assert inspector.viewer.status == (
            "Label 1 is not present at timepoint 1."
        )

    def test_unreadable_raw_stops_before_sam2(self, sam2):
        raw, labels, _ = _half_disc_pair_2d()
        before = labels.copy()
        worker = _ThresholdWorker()
        inspector, _ = sam2(labels, raw, worker)
        inspector.image_label_pairs = []

        inspector.grow_label_at_timepoint(1, 0)

        assert inspector.viewer.status == "No raw image loaded."
        assert worker.calls == 0
        np.testing.assert_array_equal(labels, before)

    def test_worker_dying_mid_segment_leaves_labels_alone(self, sam2):
        raw, labels, _ = _half_disc_pair_tzyx()
        before = labels.copy()
        inspector, _ = sam2(labels, raw, _BrokenWorker())

        inspector.grow_label_at_timepoint(1, 0)

        assert inspector.viewer.status == (
            "Grow: SAM2 worker exited unexpectedly."
        )
        np.testing.assert_array_equal(labels, before)
        assert inspector._single_t_last is None

    def test_empty_mask_reports_nothing_added(self, sam2):
        raw, labels, _ = _half_disc_pair_tzyx()
        before = labels.copy()
        inspector, _ = sam2(labels, raw, _EmptyWorker())

        inspector.grow_label_at_timepoint(1, 0)

        assert "nothing added" in inspector.viewer.status
        np.testing.assert_array_equal(labels, before)
        assert inspector._single_t_last is None


class TestOnClickGrow:
    def test_2d_layer_grows_at_t0(self):
        inspector = LabelInspector(_viewer())
        calls = []
        inspector.grow_label_at_timepoint = lambda i, t: calls.append((i, t))
        layer = SimpleNamespace(data=np.zeros((5, 5), dtype=np.uint32))

        inspector._on_click_grow(layer, 3, SimpleNamespace())

        assert calls == [(3, 0)]

    def test_unresolvable_timepoint(self):
        inspector = LabelInspector(_viewer())
        calls = []
        inspector.grow_label_at_timepoint = lambda i, t: calls.append((i, t))
        inspector._clicked_timepoint = lambda *a: None
        layer = SimpleNamespace(data=np.zeros((2, 5, 5), dtype=np.uint32))

        inspector._on_click_grow(layer, 3, SimpleNamespace())

        assert not calls
        assert "could not resolve the clicked timepoint" in (
            inspector.viewer.status
        )


# ---------------------------------------------------------------------
# add_label_at_timepoint
# ---------------------------------------------------------------------
def _missed_cell_2d():
    raw = np.full((21, 21), 10, dtype=np.uint16)
    disc = _disc((21, 21), (10, 10), 5)
    raw[disc] = 200
    return raw, np.zeros((21, 21), dtype=np.uint32), disc


class TestAddEdges:
    def test_no_labels_layer(self, monkeypatch):
        monkeypatch.setattr(mod, "Labels", _FakeLabels)
        inspector = LabelInspector(_viewer())
        inspector.add_label_at_timepoint((1, 1), 0)
        assert inspector.viewer.status == "No labels layer found."

    def test_add_on_2d_labels(self, sam2):
        raw, labels, disc = _missed_cell_2d()
        labels[0, 0] = 4  # an existing label far away sets the next ID
        inspector, layer = sam2(labels, raw)

        inspector.add_label_at_timepoint((10, 10), 0)

        assert (labels[disc] == 5).all()
        assert labels[0, 0] == 4
        assert int((labels == 5).sum()) == int(disc.sum())
        assert layer.refresh_count == 1
        assert inspector._single_t_last["t"] is None
        assert inspector.viewer.status.startswith(
            f"Added label 5: {int(disc.sum())} pixel(s) on 1 plane(s)."
        )

    def test_add_wraps_a_raw_dask_layer(self, sam2):
        import dask.array as da

        raw2, _, disc = _missed_cell_2d()
        raw = np.stack([raw2[None], raw2[None]])
        labels = np.zeros((2, 1, 21, 21), dtype=np.uint32)
        inspector, layer = sam2(
            da.from_array(labels, chunks=(1, 1, 21, 21)), raw
        )

        inspector.add_label_at_timepoint((0, 10, 10), 1)

        assert isinstance(layer.data, _DaskFancyIndexWrapper)
        assert (np.asarray(layer.data[1])[0][disc] == 1).all()
        assert not np.asarray(layer.data[0]).any()
        assert not labels.any()  # staged, not written to the base
        assert "timepoint 1" in inspector.viewer.status

    @pytest.mark.parametrize("coord", [(10,), (21, 3), (-1, 3)])
    def test_click_outside_the_image_is_refused(self, sam2, coord):
        raw, labels, _ = _missed_cell_2d()
        inspector, _ = sam2(labels, raw)
        inspector.add_label_at_timepoint(coord, 0)
        assert "outside the label image" in inspector.viewer.status
        assert not labels.any()

    def test_unreadable_raw_stops_before_sam2(self, sam2):
        raw, labels, _ = _missed_cell_2d()
        worker = _ThresholdWorker()
        inspector, _ = sam2(labels, raw, worker)
        inspector.image_label_pairs = []

        inspector.add_label_at_timepoint((10, 10), 0)

        assert inspector.viewer.status == "No raw image loaded."
        assert worker.calls == 0
        assert not labels.any()

    def test_a_full_dtype_is_refused_before_inference(self, sam2):
        raw, labels, _ = _missed_cell_2d()
        top = np.iinfo(np.uint32).max
        labels[0, 0] = top
        worker = _ThresholdWorker()
        inspector, _ = sam2(labels, raw, worker)

        inspector.add_label_at_timepoint((10, 10), 0)

        assert "cannot hold another ID" in inspector.viewer.status
        assert worker.calls == 0
        assert int((labels != 0).sum()) == 1

    def test_worker_dying_mid_segment_leaves_labels_alone(self, sam2):
        raw, labels, _ = _missed_cell_2d()
        inspector, _ = sam2(labels, raw, _BrokenWorker())

        inspector.add_label_at_timepoint((10, 10), 0)

        assert inspector.viewer.status == (
            "Add: SAM2 worker exited unexpectedly."
        )
        assert not labels.any()
        assert inspector._single_t_last is None


# ---------------------------------------------------------------------
# _on_click_add / _add_from_ray
# ---------------------------------------------------------------------
class _PosLayer:
    def __init__(self, data, world_to_data=None):
        self.data = data
        self.ndim = data.ndim
        self._w2d = world_to_data or (lambda p: np.asarray(p, dtype=float))

    def world_to_data(self, position):
        return self._w2d(position)


def _boom(_p):
    raise RuntimeError("no transform")


class TestOnClickAdd:
    def _inspector(self):
        inspector = LabelInspector(_viewer())
        calls = []
        inspector.add_label_at_timepoint = lambda c, t, raw_t=None: (
            calls.append((c, t))
        )
        return inspector, calls

    def test_refused_while_a_track_view_is_active(self):
        inspector, calls = self._inspector()
        inspector._track_view_layer = object()
        layer = _PosLayer(np.zeros((2, 5, 5), dtype=np.uint32))
        inspector._on_click_add(
            layer,
            0,
            SimpleNamespace(dims_displayed=[1, 2], position=(0, 1, 1)),
        )
        assert not calls
        assert "set Track view to 'Off'" in inspector.viewer.status

    def test_3d_click_on_a_label_is_refused(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((2, 3, 5, 5), dtype=np.uint32))
        inspector._on_click_add(
            layer, 4, SimpleNamespace(dims_displayed=[1, 2, 3])
        )
        assert not calls
        assert "already belongs to label 4" in inspector.viewer.status

    def test_off_canvas_click_is_ignored(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((5, 5), dtype=np.uint32))
        inspector._on_click_add(
            layer, None, SimpleNamespace(dims_displayed=[0, 1])
        )
        assert not calls
        assert inspector.viewer.status == ""

    def test_failing_world_to_data_is_reported(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((5, 5), dtype=np.uint32), _boom)
        inspector._on_click_add(
            layer, 0, SimpleNamespace(dims_displayed=[0, 1], position=(1, 1))
        )
        assert not calls
        assert inspector.viewer.status == (
            "Add: could not resolve the clicked voxel."
        )

    def test_wrong_length_coordinate_is_reported(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((2, 5, 5), dtype=np.uint32))
        inspector._on_click_add(
            layer, 0, SimpleNamespace(dims_displayed=[1, 2], position=(1, 1))
        )
        assert not calls
        assert "could not resolve the clicked voxel" in (
            inspector.viewer.status
        )

    def test_2d_click_is_clipped_and_uses_t0(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((5, 6), dtype=np.uint32))
        inspector._on_click_add(
            layer,
            0,
            SimpleNamespace(dims_displayed=[0, 1], position=(2.4, 99.0)),
        )
        assert calls == [((2, 5), 0)]

    def test_ray_click_with_failing_world_to_data(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((2, 3, 5, 5), dtype=np.uint32), _boom)
        inspector._on_click_add(
            layer, None, SimpleNamespace(dims_displayed=[1, 2, 3], position=0)
        )
        assert not calls
        assert inspector.viewer.status == (
            "Add: could not resolve the clicked frame."
        )

    def test_ray_click_without_a_raw_image(self):
        inspector, calls = self._inspector()
        layer = _PosLayer(np.zeros((2, 3, 5, 5), dtype=np.uint32))
        inspector._on_click_add(
            layer,
            None,
            SimpleNamespace(dims_displayed=[1, 2, 3], position=(1, 0, 2, 2)),
        )
        assert not calls
        assert inspector.viewer.status == "No raw image loaded."

    def test_ray_that_misses_the_volume(self):
        inspector, calls = self._inspector()
        inspector._raw_slice_at = lambda *a: np.zeros((3, 5, 5))
        inspector._ray_points = lambda *a: None
        layer = _PosLayer(np.zeros((2, 3, 5, 5), dtype=np.uint32))
        inspector._on_click_add(
            layer,
            None,
            SimpleNamespace(dims_displayed=[1, 2, 3], position=(9, 0, 2, 2)),
        )
        assert not calls
        assert "view ray missed the volume" in inspector.viewer.status


# ---------------------------------------------------------------------
# SAM2 helpers
# ---------------------------------------------------------------------
class TestAddHelpers:
    def test_plane_contrast_of_an_empty_mask(self):
        raw = np.arange(16, dtype=float).reshape(4, 4)
        lbl = np.zeros((4, 4), dtype=np.uint32)
        empty = np.zeros((4, 4), dtype=bool)
        assert LabelInspector._plane_contrast(raw, lbl, empty) == 0.0

    def test_plane_contrast_without_background(self):
        raw = np.arange(16, dtype=float).reshape(4, 4)
        lbl = np.zeros((4, 4), dtype=np.uint32)
        full = np.ones((4, 4), dtype=bool)
        assert LabelInspector._plane_contrast(raw, lbl, full) == 0.0

    def test_plane_contrast_is_a_median_difference(self):
        raw = np.full((4, 4), 5.0)
        raw[1:3, 1:3] = 50.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        lbl = np.zeros((4, 4), dtype=np.uint32)
        assert LabelInspector._plane_contrast(raw, lbl, mask) == 45.0

    def test_next_plane_stops_where_another_label_owns_the_footprint(self):
        inspector = LabelInspector(_viewer())
        prev = np.zeros((6, 6), dtype=bool)
        prev[2:4, 2:4] = True
        lbl = np.zeros((6, 6), dtype=np.uint32)
        lbl[1:5, 1:5] = 3  # this plane's footprint is taken

        # worker=None: the function must decide without inference.
        out = inspector._next_plane_mask(
            None, prev, lbl, np.zeros((6, 6)), 5, 0, 4.0, 1.0
        )
        assert out is None

    def test_next_plane_stops_where_the_cell_tapers_out(self):
        inspector = LabelInspector(_viewer())
        prev = np.zeros((6, 6), dtype=bool)
        prev[1:5, 1:5] = True
        speck = np.zeros((6, 6), dtype=bool)
        speck[2, 2] = True  # 1 px against a 100 px clicked plane
        prompts = []
        inspector._add_plane_mask = lambda w, prompt, point, *a: (
            prompts.append((prompt.copy(), point.copy())) or speck
        )
        lbl = np.zeros((6, 6), dtype=np.uint32)

        out = inspector._next_plane_mask(
            None, prev, lbl, np.zeros((6, 6)), 5, 0, 100.0, 1.0
        )

        assert out is None
        # The prompt is the previous cross-section, anchored at its centre.
        prompt, point = prompts[0]
        np.testing.assert_array_equal(prompt, prev)
        assert point.sum() == 1 and prev[point].all()


class TestWrapperFancyRead:
    def test_single_index_array_is_read_through_dask(self):
        import dask.array as da

        base = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)
        wrapper = _DaskFancyIndexWrapper(da.from_array(base, chunks=1))
        out = wrapper[np.array([1, 0]), :, 2]
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, base[[1, 0], :, 2])


# ---------------------------------------------------------------------
# Track views
# ---------------------------------------------------------------------
def _tzyx():
    src = np.zeros((2, 2, 4, 4), dtype=np.uint32)
    src[0, 0, 1, 1] = 3
    src[1, 1, 2, 2] = 5
    return src


class TestTrackViews:
    def test_bbox_union_with_one_side_missing(self):
        box = (slice(0, 1), slice(2, 3))
        assert _bbox_union(box, None) is box
        assert _bbox_union(None, box) is box

    def test_needs_a_3d_or_4d_source(self):
        with pytest.raises(ValueError, match="3-D"):
            _StackedTrackView(np.zeros((4, 4), dtype=np.uint32))

    def test_base_class_is_abstract(self):
        view = _TrackView(np.zeros((2, 4, 4), dtype=np.uint32))
        with pytest.raises(NotImplementedError):
            view._plane(0)
        with pytest.raises(NotImplementedError):
            view._planes_of_t(0)

    def test_stacked_view_over_a_plain_dask_source(self):
        import dask.array as da

        src = _tzyx()
        view = _StackedTrackView(da.from_array(src, chunks=(1, 2, 4, 4)))
        np.testing.assert_array_equal(view[3], src[1, 1])
        np.testing.assert_array_equal(view[0], src[0, 0])

    def test_stacked_refresh_timepoint_rereads_only_its_planes(self):
        src = _tzyx()
        view = _StackedTrackView(src)
        vol = view._ensure_vol()
        view._bbox_index()
        src[1, 0, 0, 0] = 9  # edit behind the view's back
        src[0, 0, 3, 3] = 8  # a different timepoint, not refreshed

        region = view.refresh_timepoint(1)

        assert region[0] == slice(2, 4)
        assert vol[2, 0, 0] == 9
        assert vol[0, 3, 3] == 0  # t=0 planes kept their cached content
        assert view._bboxes is None  # index dropped for rebuild

    def test_maxproj_refresh_timepoint_reprojects_exactly(self):
        src = _tzyx()
        view = _MaxProjTrackView(src)
        vol = view._ensure_vol()
        assert vol[1, 2, 2] == 5
        src[1, 0, 0, 0] = 7

        region = view.refresh_timepoint(1)

        assert region[0] == slice(1, 2)
        assert vol[1, 0, 0] == 7
        assert vol[1, 2, 2] == 5
        assert view._planes_of_t(1) == (1,)

    def test_maxproj_plane_cache_is_bounded(self):
        view = _MaxProjTrackView(_tzyx())
        view._CACHE_MAX_PLANES = 1
        view[0]
        view[1]
        assert list(view._cache) == [1]

    def test_bbox_index_needs_a_volume_and_labels(self):
        view = _StackedTrackView(np.zeros((2, 4, 4), dtype=np.uint32))
        assert view._bbox_index() is None  # no volume yet
        view._ensure_vol()
        assert view._bbox_index() is None  # nothing labeled
        assert view._bbox_ok is False

    def test_write_without_an_index_keeps_it_unbuilt(self):
        src = _tzyx()
        view = _StackedTrackView(src)
        view._ensure_vol()
        view[0, 3, 3] = 6
        assert view._bboxes is None
        assert src[0, 0, 3, 3] == 6

    def test_scalar_write_grows_the_index(self):
        src = _tzyx()
        view = _StackedTrackView(src)
        view._ensure_vol()
        view._bbox_index()

        view[3, 0, 1] = 6

        assert view._bboxes[6] == (slice(3, 4), slice(0, 1), slice(1, 2))
        assert src[1, 1, 0, 1] == 6

    def test_slice_write_grows_the_index(self):
        src = _tzyx()
        view = _StackedTrackView(src)
        view._ensure_vol()
        view._bbox_index()

        view[1, 0:2, :] = 7

        assert view._bboxes[7] == (slice(1, 2), slice(0, 2), slice(0, 4))
        assert (src[0, 1, 0:2, :] == 7).all()

    def test_empty_slice_write_records_nothing(self):
        src = _tzyx()
        view = _StackedTrackView(src)
        view._ensure_vol()
        index = view._bbox_index()

        view[1, 2:2, :] = 7

        assert 7 not in index
        assert not (src == 7).any()

    def test_unboundable_write_to_a_tyx_source_drops_the_index(self):
        src = np.zeros((2, 4, 4), dtype=np.uint32)
        src[0, 1, 1] = 2
        view = _StackedTrackView(src)
        view._ensure_vol()
        assert view._bbox_index() is not None
        mask = np.zeros((2, 4, 4), dtype=bool)
        mask[1, 3, 3] = True

        view[mask] = 4  # non-tuple, boolean: no cheap bound

        assert view._bboxes is None
        assert src[1, 3, 3] == 4  # the TYX view aliases the source
        assert view._vol[1, 3, 3] == 4

    def test_non_tuple_scalar_write_hits_the_right_timepoint(self):
        src = _tzyx()
        view = _StackedTrackView(src)
        view[3] = 8  # plane 3 = (t=1, z=1)
        assert (src[1, 1] == 8).all()
        assert not (src[1, 0] == 8).any()

    def test_unsupported_write_index_is_rejected(self):
        view = _StackedTrackView(_tzyx())
        with pytest.raises(NotImplementedError, match="writing"):
            view[np.array([0]), np.array([1])] = 3

    def test_unsupported_read_index_is_rejected(self):
        view = _StackedTrackView(_tzyx())
        with pytest.raises(NotImplementedError, match="index"):
            view[np.array([[0, 1]])]

    def test_array_conversion_honours_dtype(self):
        view = _StackedTrackView(_tzyx())
        out = view.__array__(np.int64)
        assert out.dtype == np.int64
        assert out[0, 1, 1] == 3 and out[3, 2, 2] == 5

    def test_step_search_corrects_a_rounding_shortfall(self):
        # sqrt(9/3) rounds up to 2, but ceil(3/2)**2 = 4 still exceeds 3.
        assert _pick_track_view_step(1, 3, 3, 1, budget=3) == 3


class TestTrackRedraw:
    def test_successful_partial_redraw_skips_the_full_refresh(self):
        inspector = LabelInspector(_viewer())
        view = _StackedTrackView(_tzyx())
        view._ensure_vol()
        layer = MagicMock()
        layer.data = view
        inspector._track_view_layer = layer
        seen = []
        inspector._partial_track_redraw = lambda lay, region: (
            seen.append(region) or True
        )

        inspector._refresh_track_view(timepoint=1)

        assert seen and seen[0][0] == slice(2, 4)
        layer.refresh.assert_not_called()

    def test_partial_redraw_swallows_errors(self):
        inspector = LabelInspector(SimpleNamespace(status=""))  # no dims
        assert inspector._partial_track_redraw(object(), ()) is False


# ---------------------------------------------------------------------
# Intensity measurement edge cases
# ---------------------------------------------------------------------
class TestMeasureIntensities:
    def _inspector(self, monkeypatch, raw, override="none"):
        inspector = LabelInspector(_viewer())
        inspector.image_label_pairs = [("raw.tif", "raw_labels.tif")]
        inspector.channel_axis_override = override
        monkeypatch.setattr(mod, "_load_image", lambda _p: raw)
        return inspector

    def test_track_in_one_timepoint_with_an_empty_frame(self, monkeypatch):
        labels = np.zeros((2, 4, 4), dtype=np.uint32)
        labels[1, :2, :2] = 1  # only in t=1; t=0 has no foreground
        raw = np.zeros((2, 4, 4), dtype=np.uint8)
        raw[1, :2, :2] = [[10, 10], [20, 30]]
        inspector = self._inspector(monkeypatch, raw)

        stats = inspector._measure_track_intensities(_FakeLabels(labels))

        values, freq = stats["hist"][1]
        np.testing.assert_array_equal(values, [10, 20, 30])
        np.testing.assert_array_equal(freq, [2, 1, 1])
        assert freq.dtype == np.int64
        assert stats["counts"] == {1: 4}
        assert stats["sums"] == {1: 70.0}

    def test_unreadable_raw(self, monkeypatch):
        inspector = LabelInspector(_viewer())
        inspector.image_label_pairs = [("raw.tif", "raw_labels.tif")]

        def _fail(_p):
            raise OSError("gone")

        monkeypatch.setattr(mod, "_load_image", _fail)
        labels = np.ones((2, 4, 4), dtype=np.uint32)
        assert (
            inspector._measure_track_intensities(_FakeLabels(labels)) is None
        )
        assert inspector.viewer.status == "Could not load raw image: gone"

    def test_non_integer_channel_falls_back_to_the_mean(self, monkeypatch):
        labels = np.ones((1, 2, 2), dtype=np.uint32)
        raw = np.zeros((1, 2, 2, 2), dtype=np.float32)  # T, C, Y, X
        raw[:, 0] = 2.0
        raw[:, 1] = 4.0
        inspector = self._inspector(monkeypatch, raw, override="1")

        stats = inspector._measure_track_intensities(
            _FakeLabels(labels), channel="green"
        )

        assert stats["hist"] is None  # float raw → no histogram
        assert stats["sums"][1] / stats["counts"][1] == 3.0

    def test_leading_singleton_is_squeezed(self, monkeypatch):
        labels = np.ones((2, 2, 2), dtype=np.uint32)
        raw = np.arange(8, dtype=np.uint16).reshape(1, 2, 2, 2)
        inspector = self._inspector(monkeypatch, raw)

        stats = inspector._measure_track_intensities(_FakeLabels(labels))

        assert stats["counts"] == {1: 8}
        assert stats["raw_min"] == 0 and stats["raw_max"] == 7

    def test_misaligned_raw(self, monkeypatch):
        labels = np.ones((4, 4), dtype=np.uint32)
        raw = np.ones((2, 4, 4), dtype=np.uint16)
        inspector = self._inspector(monkeypatch, raw)

        assert (
            inspector._measure_track_intensities(_FakeLabels(labels)) is None
        )
        assert "does not align with label shape (4, 4)" in (
            inspector.viewer.status
        )

    def test_iter_slices_of_2d_labels(self):
        inspector = LabelInspector(_viewer())
        labels = np.arange(4, dtype=np.uint32).reshape(2, 2)
        raw = np.full((2, 2), 9.0)
        pairs = list(inspector._iter_label_raw_slices(labels, raw))
        assert len(pairs) == 1
        np.testing.assert_array_equal(pairs[0][0], labels)
        np.testing.assert_array_equal(pairs[0][1], raw)

    def test_iter_slices_of_dask_inputs(self):
        import dask.array as da

        inspector = LabelInspector(_viewer())
        labels = np.arange(8, dtype=np.uint32).reshape(2, 2, 2)
        raw = labels.astype(float) * 10
        pairs = list(
            inspector._iter_label_raw_slices(
                da.from_array(labels, chunks=1), da.from_array(raw, chunks=1)
            )
        )
        assert len(pairs) == 2
        for t, (lbl_t, raw_t) in enumerate(pairs):
            assert isinstance(lbl_t, np.ndarray)
            assert isinstance(raw_t, np.ndarray)
            np.testing.assert_array_equal(lbl_t, labels[t])
            np.testing.assert_array_equal(raw_t, raw[t])


class TestDeleteLowIntensityGuards:
    def _inspector(self, monkeypatch, labels, load):
        monkeypatch.setattr(mod, "Labels", _FakeLabels)
        viewer = _viewer()
        viewer.layers = [_FakeLabels(labels)]
        inspector = LabelInspector(viewer)
        inspector.image_label_pairs = [("raw.tif", "raw_labels.tif")]
        inspector.channel_axis_override = "none"
        monkeypatch.setattr(mod, "_load_image", load)
        return inspector

    def test_unreadable_raw_keeps_every_track(self, monkeypatch):
        labels = np.ones((2, 4, 4), dtype=np.uint32)

        def _fail(_p):
            raise OSError("gone")

        inspector = self._inspector(monkeypatch, labels, _fail)
        inspector.delete_low_intensity_tracks(0.5)
        assert inspector.viewer.status == "Could not load raw image: gone"
        assert (labels == 1).all()

    def test_nothing_to_measure(self, monkeypatch):
        labels = np.zeros((2, 4, 4), dtype=np.uint32)
        raw = np.ones((2, 4, 4), dtype=np.uint16)
        inspector = self._inspector(monkeypatch, labels, lambda _p: raw)
        inspector.delete_low_intensity_tracks(0.5)
        assert inspector.viewer.status == "No labels found to measure."


# ---------------------------------------------------------------------
# Loaders and small helpers
# ---------------------------------------------------------------------
class TestLoaders:
    def test_zarr_load_failure_becomes_oserror(self, monkeypatch):
        def _fail(_p):
            raise ValueError("bad store")

        monkeypatch.setattr(
            "napari_tmidas._file_selector.load_zarr_basic", _fail
        )
        with pytest.raises(OSError, match="bad store"):
            mod._load_image("/nowhere/x.zarr")

    def test_image_without_skimage_fallback(self, tmp_path, monkeypatch):
        bogus = tmp_path / "junk.tif"
        bogus.write_bytes(b"not a tiff")
        monkeypatch.setattr(mod, "imread", None)
        with pytest.raises(ImportError, match="non-zarr"):
            mod._load_image(str(bogus))

    def test_label_without_skimage_fallback(self, tmp_path, monkeypatch):
        bogus = tmp_path / "junk_labels.tif"
        bogus.write_bytes(b"not a tiff")
        monkeypatch.setattr(mod, "imread", None)
        with pytest.raises(ImportError, match="label images"):
            mod._load_label(str(bogus))

    def test_unreadable_label_is_not_integer(self, tmp_path, monkeypatch):
        bogus = tmp_path / "junk_labels.tif"
        bogus.write_bytes(b"not a tiff")

        def _fail(_p):
            raise OSError("unreadable")

        monkeypatch.setattr(mod, "imread", _fail)
        assert mod._label_dtype_is_integer(str(bogus)) is False

    def test_global_max_id_of_a_numpy_backed_wrapper(self):
        inspector = LabelInspector(_viewer())
        base = np.zeros((2, 3, 3), dtype=np.uint32)
        base[1, 2, 2] = 17
        layer = _FakeLabels(_DaskFancyIndexWrapper(base))
        assert inspector._global_max_id(layer) == 17


class TestMessageGuard:
    """No QApplication (or a dead one) means no modal dialog.

    The fake ``qtpy.QtWidgets`` is swapped into ``sys.modules`` rather than
    patching the real ``QApplication``, which pytest-qt itself relies on.
    """

    @staticmethod
    def _fake_qt(monkeypatch, instance):
        import sys

        fake = SimpleNamespace(QApplication=SimpleNamespace(instance=instance))
        monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", fake)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    def test_no_qapplication_means_no_dialog(self, monkeypatch):
        self._fake_qt(monkeypatch, lambda: None)
        viewer = SimpleNamespace(status="", window=object())
        assert LabelInspector(viewer)._can_show_message() is False

    def test_a_qt_runtime_error_means_no_dialog(self, monkeypatch):
        def _dead():
            raise RuntimeError("wrapped C/C++ object has been deleted")

        self._fake_qt(monkeypatch, _dead)
        viewer = SimpleNamespace(status="", window=object())
        assert LabelInspector(viewer)._can_show_message() is False

    def test_a_live_qapplication_allows_the_dialog(self, monkeypatch):
        self._fake_qt(monkeypatch, lambda: object())
        viewer = SimpleNamespace(status="", window=object())
        assert LabelInspector(viewer)._can_show_message() is True


class TestPairDiscovery:
    def test_a_label_that_fails_validation_is_reported(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "a.tif").write_bytes(b"")
        (tmp_path / "a_labels.tif").write_bytes(b"")

        def _fail(_p):
            raise OSError("header unreadable")

        monkeypatch.setattr(mod, "_label_dtype_is_integer", _fail)
        inspector = LabelInspector(_viewer())
        messages = []
        inspector._show_message = lambda *a: messages.append(a)

        inspector.load_image_label_pairs(str(tmp_path), "_labels.tif")

        assert inspector.image_label_pairs == []
        assert inspector.viewer.status == "No valid image-label pairs found."
        assert len(messages) == 1
        level, title, text = messages[0]
        assert (level, title) == ("info", "Loading Report")
        assert "a_labels.tif: header unreadable" in text


class TestChannelAxisResolution:
    def test_zarr_detection(self, monkeypatch):
        monkeypatch.setattr(
            "napari_tmidas._file_selector.detect_channels_from_zarr_path",
            lambda _p: (2, 1),
        )
        inspector = LabelInspector(_viewer())
        axis = inspector._resolve_channel_axis(
            np.zeros((3, 2, 4, 4)), np.zeros((3, 4, 4)), "/data/x.zarr"
        )
        assert axis == 1

    def test_detection_failure_means_no_axis(self, monkeypatch):
        def _fail(_p):
            raise KeyError("multiscales")

        monkeypatch.setattr(
            "napari_tmidas._file_selector.detect_channels_from_zarr_path",
            _fail,
        )
        inspector = LabelInspector(_viewer())
        axis = inspector._resolve_channel_axis(
            np.zeros((3, 2, 4, 4)), np.zeros((3, 4, 4)), "/data/x.zarr"
        )
        assert axis is None


# ---------------------------------------------------------------------
# Click / ray resolution
# ---------------------------------------------------------------------
class TestClickResolution:
    def test_ray_points_with_a_broken_layer(self):
        inspector = LabelInspector(_viewer())
        layer = SimpleNamespace(data=np.zeros((2, 3, 3)))  # no transforms
        event = SimpleNamespace(view_direction=(1, 0, 0), position=(0, 0, 0))
        assert inspector._ray_points(layer, event, [0, 1, 2]) is None

    def test_3d_timepoint_when_the_ray_hits_nothing(self):
        inspector = LabelInspector(_viewer())
        inspector._ray_hit_plane = lambda *a: None
        layer = SimpleNamespace(data=np.zeros((2, 3, 3), dtype=np.uint32))
        event = SimpleNamespace(dims_displayed=[0, 1, 2])
        assert inspector._clicked_timepoint(layer, 1, event) is None

    def test_click_data_coord_in_3d_display(self):
        inspector = LabelInspector(_viewer())
        layer = SimpleNamespace(data=np.zeros((3, 4, 4), dtype=np.uint32))
        event = SimpleNamespace(dims_displayed=[0, 1, 2])

        inspector._ray_hit_voxel = lambda *a: None
        assert inspector._click_data_coord(layer, 1, event) is None

        inspector._ray_hit_voxel = lambda *a: np.array([5, 1, -2])
        assert inspector._click_data_coord(layer, 1, event) == (2, (1, 0))


# ---------------------------------------------------------------------
# Dock widgets
# ---------------------------------------------------------------------
def _write_pair(folder, stem, labels):
    import tifffile

    tifffile.imwrite(
        str(folder / f"{stem}.tif"), np.zeros(labels.shape, dtype=np.uint16)
    )
    path = folder / f"{stem}_labels.tif"
    tifffile.imwrite(str(path), labels)
    return path


@pytest.fixture()
def docks(tmp_path, monkeypatch):
    """Build the label_inspector docks on a mock viewer.

    Returns ``(inspector, viewer, [dock widgets in add order])``.
    """
    created = []

    class _Recorder(LabelInspector):
        def __init__(self, viewer):
            super().__init__(viewer)
            created.append(self)

    monkeypatch.setattr(mod, "LabelInspector", _Recorder)

    def build(n_pairs=1):
        for i in range(n_pairs):
            labels = np.zeros((6, 6), dtype=np.uint32)
            labels[1, 1] = i + 1
            _write_pair(tmp_path, f"p{i}", labels)
        viewer = MagicMock()
        viewer.mouse_drag_callbacks = []
        mod.label_inspector(
            folder_path=str(tmp_path),
            label_suffix="_labels.tif",
            viewer=viewer,
        )
        widgets = [c.args[0] for c in viewer.window.add_dock_widget.mock_calls]
        return created[-1], viewer, widgets

    return build


class TestDockWidgets:
    def test_skip_advances_then_disables_at_the_end(self, docks, tmp_path):
        import tifffile

        inspector, viewer, widgets = docks(n_pairs=2)
        save_w, skip_w = widgets[0][0], widgets[0][1]
        on_disk = {
            p: np.asarray(tifffile.imread(str(p)))
            for p in sorted(tmp_path.glob("*_labels.tif"))
        }

        skip_w()
        assert inspector.current_index == 1
        assert skip_w.call_button.enabled is True

        skip_w()
        assert inspector.current_index == 1
        assert "Inspection complete" in viewer.status
        assert skip_w.call_button.enabled is False
        assert save_w.call_button.enabled is False
        for path, before in on_disk.items():
            np.testing.assert_array_equal(tifffile.imread(str(path)), before)

    def test_click_mode_switches_the_tools(self, docks):
        inspector, viewer, widgets = docks()
        click_mode = widgets[1][0]

        click_mode["mode"].value = "Delete label"
        assert inspector._click_delete_cb is not None
        assert viewer.mouse_drag_callbacks == [inspector._click_delete_cb]

        click_mode["mode"].value = "Merge touching neighbors"
        assert inspector._click_delete_cb is None
        assert inspector._click_merge_cb is not None
        assert viewer.mouse_drag_callbacks == [inspector._click_merge_cb]

        click_mode["mode"].value = "Off"
        assert inspector._click_merge_cb is None
        assert viewer.mouse_drag_callbacks == []

    def test_apply_split_runs_the_pending_split(self, docks, monkeypatch):
        inspector, viewer, widgets = docks()
        apply_split = widgets[1]["apply_split"]
        labels = np.zeros((1, 7, 15), dtype=np.uint32)
        labels[0, 1:6, 1:6] = 4
        labels[0, 1:6, 9:14] = 4
        labels[0, 3, 6:9] = 4  # a one-pixel bridge between two blobs
        monkeypatch.setattr(mod, "Labels", _FakeLabels)
        layer = _FakeLabels(labels)
        viewer.layers.__iter__.side_effect = lambda: iter([layer])
        inspector._split_seeds = {
            "label_id": 4,
            "t": 0,
            "coords": [(3, 3), (3, 11)],
        }

        apply_split()

        assert inspector._split_seeds is None
        assert labels[0, 3, 3] == 4
        assert labels[0, 3, 11] == 5
        assert set(np.unique(labels)) == {0, 4, 5}
        assert "Split label 4 at timepoint 0" in viewer.status

    def test_delete_low_intensity_button(self, docks, monkeypatch):
        inspector, viewer, widgets = docks()
        button = widgets[2]["delete_low_intensity"]
        labels = np.zeros((2, 4, 4), dtype=np.uint32)
        labels[:, :2, :2] = 1  # dim track
        labels[:, 2:, 2:] = 2  # bright track
        raw = np.zeros((2, 4, 4), dtype=np.uint16)
        raw[:, :2, :2] = 10
        raw[:, 2:, 2:] = 1000
        monkeypatch.setattr(mod, "Labels", _FakeLabels)
        viewer.layers.__iter__.side_effect = lambda: iter(
            [_FakeLabels(labels)]
        )
        monkeypatch.setattr(mod, "_load_image", lambda _p: raw)
        inspector.channel_axis_override = "none"

        button(threshold=0.5, channel="Mean")

        assert not (labels == 1).any()
        assert (labels[:, 2:, 2:] == 2).all()
        assert "Deleted 1 of 2 track(s)" in viewer.status

    def test_track_view_selector(self, docks):
        inspector, viewer, widgets = docks()
        track_view = widgets[3]

        track_view["mode"].value = "Stack T along Z"

        assert inspector.track_view_mode == "stack"
        # The loaded pair is 2-D, and a mock viewer finds no Labels layer.
        assert "Track view needs" in viewer.status

        track_view["mode"].value = "Off"
        assert inspector.track_view_mode == "off"
        assert viewer.status == "Track view off."


class TestBrowseButton:
    def test_browse_fills_the_folder_field(self, qtbot, monkeypatch):
        from qtpy.QtWidgets import QPushButton

        widget = mod.label_inspector_widget()
        original = widget["folder_path"].value
        layout = widget["folder_path"].native.parent().layout()
        button = layout.itemAt(layout.count() - 1).widget()
        assert isinstance(button, QPushButton)
        assert button.text() == "Browse..."

        try:
            monkeypatch.setattr(
                mod.QFileDialog,
                "getExistingDirectory",
                staticmethod(lambda *a, **k: ""),
            )
            button.click()  # cancelled: field untouched
            assert widget["folder_path"].value == original

            monkeypatch.setattr(
                mod.QFileDialog,
                "getExistingDirectory",
                staticmethod(lambda *a, **k: "/picked/folder"),
            )
            button.click()
            assert widget["folder_path"].value == "/picked/folder"
        finally:
            widget["folder_path"].value = original
            with contextlib.suppress(Exception):
                layout.removeWidget(button)
                button.deleteLater()

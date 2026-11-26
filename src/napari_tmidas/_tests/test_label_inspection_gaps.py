"""
Label Inspection: saving must survive failures in the workflow-log
bookkeeping.

Recording a save in the workspace's workflow log is best-effort: a
read-only share, a malformed edit record or any other logging failure must
never cost the user the label edits they just saved.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import napari_tmidas._label_inspection as mod
from napari_tmidas._label_inspection import (
    LabelInspector,
    _DaskFancyIndexWrapper,
    _wrapper_edit_summary,
)


class _FakeLabels:
    def __init__(self, data):
        self.data = data
        self.selected_label = 1
        self.scale = [1.0] * int(getattr(data, "ndim", 2))
        self.visible = True

    def refresh(self):
        pass

    def bind_key(self, key, handler, overwrite=False):
        pass


class _FakeViewer:
    def __init__(self, layers=()):
        self.layers = list(layers)
        self.status = ""
        self.mouse_drag_callbacks = []
        self.dims = SimpleNamespace(ndisplay=2)

    def bind_key(self, key, handler, overwrite=False):
        pass


@pytest.fixture
def fake_labels_cls(monkeypatch):
    """Make ``isinstance(layer, Labels)`` recognise :class:`_FakeLabels`."""
    monkeypatch.setattr(mod, "Labels", _FakeLabels)


def _wrap(base):
    import dask.array as da

    return _DaskFancyIndexWrapper(
        da.from_array(base, chunks=(1, *base.shape[1:]))
    )


def _write_pair(folder, base):
    image_path = folder / "a.tif"
    label_path = folder / "a_labels.tif"
    tifffile.imwrite(
        str(image_path),
        np.zeros(base.shape, dtype=np.uint16),
        photometric="minisblack",
    )
    tifffile.imwrite(str(label_path), base, photometric="minisblack")
    return str(image_path), str(label_path)


class TestSaveWithABrokenWorkflowLog:
    def test_the_edits_are_still_written(
        self, tmp_path, fake_labels_cls, monkeypatch, capsys
    ):
        class BrokenLog:
            def __init__(self, folder):
                raise PermissionError("read-only share")

        monkeypatch.setattr(mod, "WorkflowLog", BrokenLog)
        base = np.zeros((3, 4, 4), dtype=np.uint32)
        base[:, 0, 0] = 5
        image_path, label_path = _write_pair(tmp_path, base)
        wrapper = _wrap(base.copy())
        wrapper.remap_values({5: 9})
        inspector = LabelInspector(_FakeViewer([_FakeLabels(wrapper)]))
        inspector.image_label_pairs = [(image_path, label_path)]

        inspector.save_current_labels()

        written = tifffile.imread(label_path)
        assert (written[:, 0, 0] == 9).all()
        assert inspector.viewer.status == f"Saved labels to {label_path}."
        out = capsys.readouterr().out
        assert (
            "Workflow log: could not record label save: read-only share"
            in out
        )


class TestEditSummary:
    def test_a_malformed_edit_record_yields_an_empty_summary(self):
        """Bookkeeping must never raise into the save path."""
        broken = SimpleNamespace(_diffs={7: "not a (dim, idx) key"}, _lut={})

        assert _wrapper_edit_summary(broken) == {}

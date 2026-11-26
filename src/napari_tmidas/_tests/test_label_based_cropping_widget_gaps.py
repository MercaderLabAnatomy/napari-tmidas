"""
A workflow log that cannot be written must never break a crop: both halves
of the run record swallow their errors and leave the widget usable.
"""

import os
import sys

import numpy as np
import pytest

napari = pytest.importorskip("napari")

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)

import napari_tmidas._label_based_cropping_widget as mod  # noqa: E402


@pytest.fixture
def widget_on_file(make_napari_viewer, tmp_path, monkeypatch):
    import tifffile
    from napari.layers._source import layer_source

    monkeypatch.delenv("NAPARI_TMIDAS_NO_WORKFLOW_LOG", raising=False)
    image = np.ones((4, 16, 16), dtype=np.uint8)
    tifffile.imwrite(str(tmp_path / "raw.tif"), image)
    viewer = make_napari_viewer()
    with layer_source(path=str(tmp_path / "raw.tif")):
        viewer.add_image(image, name="raw")
    viewer.add_labels(np.ones((4, 16, 16), dtype=np.uint32), name="mask")
    widget = mod.LabelBasedCroppingWidget(viewer)
    widget._crop_name = "cropped"
    widget._source_layer = viewer.layers["raw"]
    return widget, viewer


def test_a_log_that_cannot_be_opened_records_nothing(
    widget_on_file, monkeypatch, capsys
):
    widget, viewer = widget_on_file

    class ReadOnlyFolder:
        def __init__(self, *_args, **_kwargs):
            raise PermissionError("read-only folder")

    monkeypatch.setattr(mod, "WorkflowLog", ReadOnlyFolder)

    widget._record_crop_started(viewer.layers["raw"], viewer.layers["mask"])

    assert widget._log is None
    assert widget._log_run_id is None
    assert "could not record crop" in capsys.readouterr().out


def test_a_log_that_fails_when_finishing_is_swallowed_and_cleared(
    widget_on_file, capsys
):
    widget, viewer = widget_on_file
    widget._record_crop_started(viewer.layers["raw"], viewer.layers["mask"])
    assert widget._log is not None

    def disk_full(*_args, **_kwargs):
        raise OSError("No space left on device")

    widget._log.run_finished = disk_full

    widget._record_crop_finished(True, "ok", np.ones((4, 16, 16)))

    assert widget._log is None
    assert widget._log_run_id is None
    assert "could not record crop" in capsys.readouterr().out

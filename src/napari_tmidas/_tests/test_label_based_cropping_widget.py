# src/napari_tmidas/_tests/test_label_based_cropping_widget.py
"""
Tests for the Label-Based Cropping dock widget.

This module is a registered napari contribution (napari.yaml declares
``_label_based_cropping_widget:napari_experimental_provide_dock_widget``) but
no test had ever imported it, so nothing caught an import error, a broken
signal connection or a constructor that raises -- all of which surface to the
user as a plugin that simply fails to open.

The heavy lifting is tested in test_label_based_cropping.py against the
processing functions; what is checked here is the wiring.
"""
import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari_tmidas._label_based_cropping_widget import (  # noqa: E402
    LabelBasedCroppingWidget,
    LabelBasedCroppingWorker,
    napari_experimental_provide_dock_widget,
)


@pytest.fixture
def viewer_with_layers(make_napari_viewer):
    """A viewer holding one image and one label layer of matching shape."""
    viewer = make_napari_viewer()
    rng = np.random.default_rng(0)
    viewer.add_image(
        (rng.random((8, 32, 32)) * 255).astype(np.uint8), name="raw"
    )
    labels = np.zeros((8, 32, 32), dtype=np.uint32)
    labels[3, 8:16, 8:16] = 1
    viewer.add_labels(labels, name="mask")
    return viewer


class TestWidgetConstruction:
    def test_constructs_without_layers(self, make_napari_viewer):
        """An empty viewer is the state the widget first opens in."""
        widget = LabelBasedCroppingWidget(make_napari_viewer())

        assert widget is not None
        assert widget._image_layer_combo.count() == 0
        assert widget._label_layer_combo.count() == 0

    def test_populates_combos_from_viewer_layers(self, viewer_with_layers):
        widget = LabelBasedCroppingWidget(viewer_with_layers)

        assert widget._image_layer_combo.count() == 1
        assert widget._image_layer_combo.itemText(0) == "raw"
        assert widget._label_layer_combo.count() == 1
        assert widget._label_layer_combo.itemText(0) == "mask"

    def test_image_and_label_layers_do_not_cross_populate(
        self, viewer_with_layers
    ):
        """
        The combos filter by layer type. Offering a Labels layer as the
        intensity image would crop the wrong array.
        """
        widget = LabelBasedCroppingWidget(viewer_with_layers)

        image_items = [
            widget._image_layer_combo.itemText(i)
            for i in range(widget._image_layer_combo.count())
        ]
        label_items = [
            widget._label_layer_combo.itemText(i)
            for i in range(widget._label_layer_combo.count())
        ]

        assert "mask" not in image_items
        assert "raw" not in label_items

    def test_combo_data_indexes_back_into_viewer_layers(
        self, viewer_with_layers
    ):
        """
        Each entry carries the layer's index, which the crop handler uses to
        look the layer back up. An off-by-one here crops a different layer.
        """
        widget = LabelBasedCroppingWidget(viewer_with_layers)

        image_idx = widget._image_layer_combo.itemData(0)
        label_idx = widget._label_layer_combo.itemData(0)

        assert viewer_with_layers.layers[image_idx].name == "raw"
        assert viewer_with_layers.layers[label_idx].name == "mask"

    def test_defaults(self, viewer_with_layers):
        widget = LabelBasedCroppingWidget(viewer_with_layers)

        assert widget._crop_name_input.text() == "cropped"
        assert widget._expand_z_checkbox.isChecked() is False
        assert widget._expand_time_checkbox.isChecked() is False


class TestCropAction:
    """
    The crop runs on a QThread and every guard path opens a modal QMessageBox,
    so the dialog is stubbed out -- unpatched it blocks the test run forever
    waiting for a click that never comes.
    """

    @pytest.fixture
    def no_dialogs(self, monkeypatch):
        """Replace the modal warnings with a recorder."""
        calls = []
        import napari_tmidas._label_based_cropping_widget as mod

        monkeypatch.setattr(
            mod.QMessageBox,
            "warning",
            lambda *a, **k: calls.append(a[2] if len(a) > 2 else ""),
        )
        monkeypatch.setattr(
            mod.QMessageBox,
            "critical",
            lambda *a, **k: calls.append(a[2] if len(a) > 2 else ""),
        )
        return calls

    def test_crop_produces_a_new_layer(
        self, viewer_with_layers, no_dialogs, qtbot
    ):
        widget = LabelBasedCroppingWidget(viewer_with_layers)
        before = len(viewer_with_layers.layers)

        widget._on_crop_clicked()

        # The worker runs on its own thread; waitUntil spins the event loop
        # so the finished signal can be delivered back to the widget.
        qtbot.waitUntil(
            lambda: "cropped" in viewer_with_layers.layers, timeout=10000
        )

        assert len(viewer_with_layers.layers) == before + 1
        assert widget._crop_button.isEnabled(), (
            "the Crop button was left disabled after the worker finished"
        )

    def test_crop_without_layers_warns_instead_of_raising(
        self, make_napari_viewer, no_dialogs
    ):
        """
        Clicking Crop on an empty viewer must surface a warning, not an
        unhandled exception in the Qt event loop.
        """
        widget = LabelBasedCroppingWidget(make_napari_viewer())

        widget._on_crop_clicked()

        assert no_dialogs, "no warning was shown for an empty viewer"
        assert widget._worker is None, "a worker was started with no layers"

    def test_expand_checkboxes_are_safe_without_a_selection(
        self, make_napari_viewer
    ):
        """
        The stateChanged handlers read the combo selection; with nothing
        selected they must decline and untick rather than raise.
        """
        widget = LabelBasedCroppingWidget(make_napari_viewer())

        widget._expand_z_checkbox.setChecked(True)
        widget._expand_time_checkbox.setChecked(True)

        assert widget._expand_z_checkbox.isChecked() is False
        assert widget._expand_time_checkbox.isChecked() is False


class TestCroppingWorker:
    """
    The worker holds the actual behaviour; run() is called directly here so
    the assertions do not depend on thread scheduling.
    """

    def test_reports_success_and_returns_cropped_data(self):
        rng = np.random.default_rng(0)
        image = (rng.random((8, 32, 32)) * 255).astype(np.uint8)
        label = np.zeros((8, 32, 32), dtype=np.uint32)
        label[3:5, 8:16, 8:16] = 1
        results = []

        worker = LabelBasedCroppingWorker(image, label)
        worker.finished.connect(
            lambda ok, msg, data: results.append((ok, msg, data))
        )
        worker.run()

        assert len(results) == 1
        ok, _, data = results[0]
        assert ok is True
        assert data is not None
        # "Cropping" here means masking: the shape is preserved, everything
        # outside the label is zeroed and everything inside is untouched.
        assert data.shape == image.shape
        inside = label > 0
        np.testing.assert_array_equal(data[inside], image[inside])
        assert np.all(data[~inside] == 0)

    def test_shape_mismatch_is_reported_not_raised(self):
        """
        A 2D label against a 3D image is the common mistake the expansion
        checkboxes exist to fix, so it must come back as a message.
        """
        image = np.zeros((8, 32, 32), dtype=np.uint8)
        label = np.zeros((32, 32), dtype=np.uint32)
        results = []

        worker = LabelBasedCroppingWorker(image, label)
        worker.finished.connect(
            lambda ok, msg, data: results.append((ok, msg, data))
        )
        worker.run()

        ok, msg, data = results[0]
        assert ok is False
        assert data is None
        assert "does not match" in msg


class TestInfoText:
    def test_messages_accumulate(self, viewer_with_layers):
        widget = LabelBasedCroppingWidget(viewer_with_layers)

        widget._update_info("first")
        widget._update_info("second")

        text = widget._info_text.toPlainText()
        assert "first" in text and "second" in text


class TestNapariContribution:
    def test_provider_returns_the_registered_callable(self):
        """
        napari.yaml points at this function; if it stops returning a callable
        the plugin fails to load with no other signal.
        """
        provided = napari_experimental_provide_dock_widget()

        assert callable(provided)

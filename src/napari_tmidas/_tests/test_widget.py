import os
import sys

import numpy as np
import pytest

from napari_tmidas._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)

# Check if pytest-qt is available
try:
    import pytest_qt  # noqa: F401

    PYTEST_QT_AVAILABLE = True
except ImportError:
    PYTEST_QT_AVAILABLE = False


def test_threshold_autogenerate_widget():
    # because our "widget" is a pure function, we can call it and
    # test it independently of napari
    im_data = np.random.random((100, 100))
    thresholded = threshold_autogenerate_widget(im_data, 0.5)
    assert thresholded.shape == im_data.shape
    # etc.


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
@pytest.mark.skipif(
    not PYTEST_QT_AVAILABLE
    or (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
    or (sys.platform == "win32" and os.environ.get("CI") == "true"),
    reason="Requires pytest-qt, X11 display in headless *nix CI or full napari install on Windows CI",
)
def test_threshold_magic_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # our widget will be a MagicFactory or FunctionGui instance
    my_widget = threshold_magic_widget()

    # if we "call" this object, it'll execute our function
    thresholded = my_widget(viewer.layers[0], 0.5)
    assert thresholded.shape == layer.data.shape
    # etc.


@pytest.mark.skipif(
    not PYTEST_QT_AVAILABLE
    or (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
    or (sys.platform == "win32" and os.environ.get("CI") == "true"),
    reason="Requires pytest-qt, X11 display in headless *nix CI or full napari install on Windows CI",
)
def test_image_threshold_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))
    my_widget = ImageThreshold(viewer)

    # because we saved our widgets as attributes of the container
    # we can set their values without having to "interact" with the viewer
    my_widget._image_layer_combo.value = layer
    my_widget._threshold_slider.value = 0.5

    # this allows us to run our functions directly and ensure
    # correct results
    my_widget._threshold_im()
    assert len(viewer.layers) == 2


# capsys is a pytest fixture that captures stdout and stderr output streams
@pytest.mark.skipif(
    not PYTEST_QT_AVAILABLE
    or (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
    or (sys.platform == "win32" and os.environ.get("CI") == "true"),
    reason="Requires pytest-qt, X11 display in headless *nix CI or full napari install on Windows CI",
)
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = ExampleQWidget(viewer)

    # call our widget method
    my_widget._on_click()

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "napari has 1 layers\n"

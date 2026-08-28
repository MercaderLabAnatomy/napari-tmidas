"""
Coverage tests for
``napari_tmidas.processing_functions.spotiflow_detection``.

Spotiflow (and torch) are not installed in this environment: the module
is designed to fall back to a dedicated conda environment driven through
a subprocess.  Every test here therefore either exercises pure-numpy
helpers or installs light-weight stand-ins for ``torch`` / ``spotiflow``
into ``sys.modules`` so the real module code runs unchanged.  Nothing
shells out for real.
"""

import importlib
import subprocess
import sys
import tempfile
import types
from importlib import machinery

import numpy as np
import pytest

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import spotiflow_detection as sd

# ---------------------------------------------------------------------
# Test doubles for the absent heavy dependencies
# ---------------------------------------------------------------------


class _FakeTensor:
    """Stand-in for a torch tensor whose ``.cuda()`` may fail."""

    def __init__(self, cuda_ok=True):
        self._cuda_ok = cuda_ok

    def cuda(self):
        if not self._cuda_ok:
            raise RuntimeError("CUDA driver / kernel image mismatch")
        return self


def _make_torch(cuda_available=False, cuda_ok=True):
    """Build a minimal fake ``torch`` module object."""
    mod = types.ModuleType("torch")

    class OutOfMemoryError(RuntimeError):
        pass

    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        OutOfMemoryError=OutOfMemoryError,
    )
    mod.device = lambda name: f"dev:{name}"
    mod.ones = lambda _n: _FakeTensor(cuda_ok)
    return mod


class _FakeModel:
    """Stand-in for a loaded ``Spotiflow`` model."""

    def __init__(
        self,
        points=None,
        details=None,
        is_3d=True,
        to_failures=0,
        predict_errors=None,
    ):
        if points is None:
            points = np.zeros((0, 2), dtype=float)
        self.points = points
        self.details = details
        self.config = types.SimpleNamespace(is_3d=is_3d)
        self._to_failures = to_failures
        self._predict_errors = list(predict_errors or [])
        self.to_calls = []
        self.predict_calls = []

    def to(self, device):
        self.to_calls.append(device)
        if self._to_failures > 0:
            self._to_failures -= 1
            raise RuntimeError("could not move model to GPU")
        return self

    def predict(self, img, **kwargs):
        self.predict_calls.append((img, kwargs))
        if self._predict_errors:
            raise self._predict_errors.pop(0)
        return self.points, self.details


def _install_torch(monkeypatch, **kwargs):
    mod = _make_torch(**kwargs)
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


def _install_spotiflow(monkeypatch, model=None, write_coords=None):
    """Insert a fake ``spotiflow`` package into ``sys.modules``."""
    pkg = types.ModuleType("spotiflow")
    model_mod = types.ModuleType("spotiflow.model")
    utils_mod = types.ModuleType("spotiflow.utils")

    class Spotiflow:
        loaded = []

        @staticmethod
        def from_pretrained(name):
            Spotiflow.loaded.append(("pretrained", name))
            return model

        @staticmethod
        def from_folder(path):
            Spotiflow.loaded.append(("folder", path))
            return model

    model_mod.Spotiflow = Spotiflow
    if write_coords is not None:
        utils_mod.write_coords_csv = write_coords
    pkg.model = model_mod
    pkg.utils = utils_mod
    monkeypatch.setitem(sys.modules, "spotiflow", pkg)
    monkeypatch.setitem(sys.modules, "spotiflow.model", model_mod)
    monkeypatch.setitem(sys.modules, "spotiflow.utils", utils_mod)
    return Spotiflow


def _direct(image, **overrides):
    """Call ``_detect_spots_direct`` with sane defaults."""
    kwargs = {
        "axes": "YX",
        "pretrained_model": "general",
        "model_path": "",
        "subpixel": True,
        "peak_mode": "fast",
        "normalizer": "percentile",
        "normalizer_low": 1.0,
        "normalizer_high": 99.8,
        "prob_thresh": None,
        "n_tiles": "auto",
        "exclude_border": True,
        "scale": "auto",
        "min_distance": 2,
        "force_cpu": True,
    }
    kwargs.update(overrides)
    return sd._detect_spots_direct(image, **kwargs)


# ---------------------------------------------------------------------


class TestAxesHelpers:
    """Pins the pure axes/reshape helpers copied from napari-spotiflow."""

    def test_validate_axes_accepts_matching_length(self):
        # A matching axes string is accepted silently: the validator is a
        # guard, so its contract is "returns None, raises nothing".
        assert sd._validate_axes(np.zeros((3, 4)), "YX") is None
        assert sd._validate_axes(np.zeros((2, 3, 4)), "ZYX") is None

    def test_validate_axes_rejects_mismatch(self):
        with pytest.raises(ValueError, match="2 dimensions"):
            sd._validate_axes(np.zeros((3, 4)), "ZYX")

    @pytest.mark.parametrize(
        ("shape", "axes", "expected"),
        [
            ((5, 6), "YX", (5, 6, 1)),
            ((4, 5, 6), "ZYX", (4, 5, 6, 1)),
            ((4, 5, 6), "TYX", (4, 5, 6, 1)),
            ((3, 4, 5, 6), "TZYX", (3, 4, 5, 6, 1)),
        ],
    )
    def test_prepare_input_appends_channel_axis(self, shape, axes, expected):
        out = sd._prepare_input(np.zeros(shape), axes)
        assert out.shape == expected

    @pytest.mark.parametrize(
        ("shape", "axes"),
        [
            ((5, 6, 2), "YXC"),
            ((4, 5, 6, 2), "ZYXC"),
            ((4, 5, 6, 2), "TYXC"),
            ((3, 4, 5, 6, 2), "TZYXC"),
        ],
    )
    def test_prepare_input_passes_channel_last_through(self, shape, axes):
        img = np.zeros(shape)
        out = sd._prepare_input(img, axes)
        assert out is img

    @pytest.mark.parametrize(
        ("shape", "axes", "expected"),
        [
            ((3, 5, 6), "CYX", (5, 6, 3)),
            ((2, 3, 4, 5), "CZYX", (3, 4, 5, 2)),
            ((3, 2, 4, 5), "ZCYX", (3, 4, 5, 2)),
            ((3, 2, 4, 5), "TCYX", (3, 4, 5, 2)),
            ((2, 3, 4, 5, 6), "TZCYX", (2, 3, 5, 6, 4)),
            ((2, 3, 4, 5, 6), "TCZYX", (2, 4, 5, 6, 3)),
        ],
    )
    def test_prepare_input_moves_channels_last(self, shape, axes, expected):
        out = sd._prepare_input(np.zeros(shape), axes)
        assert out.shape == expected

    def test_prepare_input_rejects_unknown_axes(self):
        with pytest.raises(ValueError, match="Invalid axes: XY"):
            sd._prepare_input(np.zeros((3, 4)), "XY")

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ((5, 6), "YX"),
            ((4, 5, 6), "ZYX"),
            ((4, 5, 6, 3), "ZYXC"),
            ((4, 5, 6, 7), "TZYX"),
            ((2, 3, 4, 5, 6), "TZYXC"),
        ],
    )
    def test_infer_axes(self, shape, expected):
        assert sd._infer_axes(np.zeros(shape)) == expected

    @pytest.mark.parametrize("ndim", [1, 6])
    def test_infer_axes_rejects_unsupported_ndim(self, ndim):
        img = np.zeros((2,) * ndim)
        with pytest.raises(ValueError, match="Cannot infer axes"):
            sd._infer_axes(img)


class TestPointsToLabelMask:
    """Pins the point -> label-mask rasterisation for every shape."""

    def test_empty_points_returns_zero_mask(self):
        mask = sd._points_to_label_mask(
            np.zeros((0, 2)), (10, 12), spot_radius=2
        )
        assert mask.shape == (10, 12)
        assert mask.dtype == np.uint16
        assert not mask.any()

    def test_2d_image_2d_points(self):
        points = np.array([[5.0, 5.0], [15.0, 16.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=2)
        assert mask.shape == (20, 20)
        assert mask.dtype == np.uint16
        assert set(np.unique(mask)) == {0, 1, 2}
        assert mask[5, 5] == 1
        assert mask[15, 16] == 2

    def test_2d_points_outside_bounds_are_skipped(self):
        points = np.array([[50.0, 50.0], [-3.0, 4.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=2)
        assert not mask.any()

    def test_yxc_image_labels_all_channels(self):
        points = np.array([[8.0, 9.0]])
        mask = sd._points_to_label_mask(points, (20, 20, 3), spot_radius=1)
        assert mask.shape == (20, 20, 3)
        assert (mask[8, 9, :] == 1).all()

    def test_pure_3d_image_3d_points(self):
        points = np.array([[4.0, 10.0, 10.0]])
        mask = sd._points_to_label_mask(points, (8, 20, 20), spot_radius=1)
        assert mask.shape == (8, 20, 20)
        assert mask[4, 10, 10] == 1
        # a radius-1 ball is a 6-neighbourhood cross plus the centre
        assert int((mask == 1).sum()) == 7

    def test_3d_image_2d_points_land_in_middle_slice(self):
        points = np.array([[10.0, 10.0]])
        mask = sd._points_to_label_mask(points, (9, 20, 20), spot_radius=1)
        assert mask[4, 10, 10] == 1
        assert not mask[0].any()

    def test_zyxc_image_labels_all_channels(self):
        points = np.array([[2.0, 6.0, 6.0]])
        mask = sd._points_to_label_mask(points, (5, 20, 20, 3), spot_radius=1)
        assert mask.shape == (5, 20, 20, 3)
        assert (mask[2, 6, 6, :] == 1).all()

    def test_tzyx_image_labels_all_timepoints(self):
        points = np.array([[2.0, 6.0, 6.0]])
        mask = sd._points_to_label_mask(points, (3, 5, 20, 20), spot_radius=1)
        assert mask.shape == (3, 5, 20, 20)
        assert (mask[:, 2, 6, 6] == 1).all()

    def test_tzyxc_image_labels_all_t_and_c(self):
        points = np.array([[2.0, 6.0, 6.0]])
        mask = sd._points_to_label_mask(
            points, (2, 5, 20, 20, 3), spot_radius=1
        )
        assert mask.shape == (2, 5, 20, 20, 3)
        assert (mask[:, 2, 6, 6, :] == 1).all()

    def test_3d_points_on_2d_image_drop_leading_axis(self):
        # (z, y, x) - dims 1 and 2 fit the image
        points = np.array([[100.0, 5.0, 6.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=1)
        assert mask[5, 6] == 1

    def test_3d_points_on_2d_image_drop_middle_axis(self):
        # (y, z, x) - dims 0 and 2 fit the image
        points = np.array([[5.0, 50.0, 6.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=1)
        assert mask[5, 6] == 1

    def test_3d_points_on_2d_image_drop_trailing_axis(self):
        # (y, x, z) - dims 0 and 1 fit the image
        points = np.array([[5.0, 6.0, 50.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=1)
        assert mask[5, 6] == 1

    def test_3d_points_on_2d_image_fall_back_to_swap(self):
        # nothing fits: the code swaps the first two columns and the
        # resulting coordinates are out of bounds, so nothing is drawn
        points = np.array([[50.0, 60.0, 70.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=1)
        assert not mask.any()

    def test_unexpected_point_width_raises(self):
        points = np.zeros((2, 4))
        with pytest.raises(ValueError, match="Unexpected points shape"):
            sd._points_to_label_mask(points, (20, 20), spot_radius=1)

    def test_3d_ball_is_clipped_at_the_border(self):
        points = np.array([[0.0, 0.0, 0.0]])
        mask = sd._points_to_label_mask(points, (6, 20, 20), spot_radius=1)
        assert mask[0, 0, 0] == 1
        # only the in-bounds half of the ball survives
        assert int((mask == 1).sum()) == 4


class TestDetectSpotsDirect:
    """Pins device selection, model loading and parameter plumbing."""

    def test_force_cpu_selects_cpu_and_clears_cuda_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        _install_torch(monkeypatch, cuda_available=True)
        model = _FakeModel(points=np.zeros((3, 2)))
        _install_spotiflow(monkeypatch, model)

        out = _direct(np.arange(400).reshape(20, 20), force_cpu=True)

        assert len(out) == 3
        assert model.to_calls == ["dev:cpu"]
        import os

        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    def test_cuda_available_and_working_uses_cuda(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=True, cuda_ok=True)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), force_cpu=False)

        assert model.to_calls == ["dev:cuda"]

    def test_broken_cuda_falls_back_to_cpu(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=True, cuda_ok=False)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), force_cpu=False)

        assert model.to_calls == ["dev:cpu"]

    def test_no_cuda_uses_cpu(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=False)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), force_cpu=False)

        assert model.to_calls == ["dev:cpu"]

    def test_existing_model_path_loads_from_folder(
        self, monkeypatch, tmp_path
    ):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        spot = _install_spotiflow(monkeypatch, model)
        spot.loaded.clear()

        folder = tmp_path / "mymodel"
        folder.mkdir()
        _direct(np.arange(400).reshape(20, 20), model_path=str(folder))

        assert spot.loaded == [("folder", str(folder))]

    def test_missing_model_path_loads_pretrained(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        spot = _install_spotiflow(monkeypatch, model)
        spot.loaded.clear()

        _direct(
            np.arange(400).reshape(20, 20),
            model_path="/does/not/exist",
            pretrained_model="hybiss",
        )

        assert spot.loaded == [("pretrained", "hybiss")]

    def test_model_to_failure_retries_on_cpu(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=True, cuda_ok=True)
        model = _FakeModel(points=np.zeros((1, 2)), to_failures=1)
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), force_cpu=False)

        assert model.to_calls == ["dev:cuda", "dev:cpu"]

    def test_model_to_failure_reraises_when_already_cpu(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)), to_failures=1)
        _install_spotiflow(monkeypatch, model)

        with pytest.raises(RuntimeError, match="could not move model"):
            _direct(np.arange(400).reshape(20, 20), force_cpu=True)

    def test_2d_model_on_3d_data_warns(self, monkeypatch, capsys):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 3)), is_3d=False)
        _install_spotiflow(monkeypatch, model)

        _direct(np.zeros((4, 20, 20)), axes="ZYX")

        assert "2D model on 3D data" in capsys.readouterr().out

    def test_bad_axes_fall_back_to_raw_image(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        image = np.arange(400).reshape(20, 20).astype(float)
        _direct(image, axes="ZYX")

        # _prepare_input raised, so the untouched 2D image was predicted on
        sent = model.predict_calls[0][0]
        assert sent.shape == (20, 20)

    def test_parse_param_handles_auto_tuple_and_garbage(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(
            np.arange(400).reshape(20, 20),
            n_tiles="(2,2)",
            scale="auto",
        )
        kwargs = model.predict_calls[0][1]
        assert kwargs["n_tiles"] == (2, 2)
        assert "scale" not in kwargs

    def test_scale_tuple_is_forwarded(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), scale="(1,1)")
        assert model.predict_calls[0][1]["scale"] == (1, 1)

    def test_non_tuple_string_param_is_passed_verbatim(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), n_tiles="whatever")
        assert model.predict_calls[0][1]["n_tiles"] == "whatever"

    def test_positive_prob_thresh_is_forwarded(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), prob_thresh=0.35)
        assert model.predict_calls[0][1]["prob_thresh"] == 0.35

    @pytest.mark.parametrize("thresh", [None, 0.0])
    def test_automatic_prob_thresh_is_omitted(self, monkeypatch, thresh):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), prob_thresh=thresh)
        assert "prob_thresh" not in model.predict_calls[0][1]

    def test_static_predict_kwargs_are_plumbed_through(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(
            np.arange(400).reshape(20, 20),
            subpixel=False,
            peak_mode="skimage",
            exclude_border=False,
            min_distance=7,
        )
        kwargs = model.predict_calls[0][1]
        assert kwargs["subpix"] is False
        assert kwargs["peak_mode"] == "skimage"
        assert kwargs["exclude_border"] is False
        assert kwargs["min_distance"] == 7
        assert kwargs["normalizer"] is None
        assert kwargs["verbose"] is True

    def test_percentile_normalisation_clips_to_unit_range(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        image = np.arange(400).reshape(20, 20).astype(float)
        _direct(image, normalizer="percentile")
        sent = model.predict_calls[0][0]
        assert sent.min() >= 0.0
        assert sent.max() <= 1.0

    def test_minmax_normalisation_spans_zero_to_one(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        image = np.arange(400).reshape(20, 20).astype(float)
        _direct(image, normalizer="minmax")
        sent = model.predict_calls[0][0]
        assert sent.min() == pytest.approx(0.0)
        assert sent.max() == pytest.approx(1.0)

    def test_minmax_on_constant_image_returns_input(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        image = np.full((20, 20), 7.0)
        _direct(image, normalizer="minmax")
        sent = model.predict_calls[0][0]
        assert (sent == 7.0).all()

    def test_unknown_normalizer_leaves_image_untouched(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        image = np.arange(400).reshape(20, 20).astype(float)
        _direct(image, normalizer="none")
        sent = model.predict_calls[0][0]
        assert sent.max() == 399.0

    def test_cuda_error_during_predict_retries_on_cpu(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        _install_torch(monkeypatch, cuda_available=True, cuda_ok=True)
        model = _FakeModel(
            points=np.zeros((2, 2)),
            predict_errors=[RuntimeError("CUDA out of memory")],
        )
        _install_spotiflow(monkeypatch, model)

        out = _direct(np.arange(400).reshape(20, 20), force_cpu=False)

        assert len(out) == 2
        assert len(model.predict_calls) == 2
        assert model.to_calls == ["dev:cuda", "dev:cpu"]

    def test_non_cuda_runtime_error_propagates(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=True, cuda_ok=True)
        model = _FakeModel(
            points=np.zeros((2, 2)),
            predict_errors=[RuntimeError("shape mismatch")],
        )
        _install_spotiflow(monkeypatch, model)

        with pytest.raises(RuntimeError, match="shape mismatch"):
            _direct(np.arange(400).reshape(20, 20), force_cpu=False)

    def test_many_spots_are_probability_filtered(self, monkeypatch):
        _install_torch(monkeypatch)
        probs = np.linspace(0.0, 1.0, 600)
        details = types.SimpleNamespace(prob=probs)
        model = _FakeModel(points=np.zeros((600, 2)), details=details)
        _install_spotiflow(monkeypatch, model)

        out = _direct(np.arange(400).reshape(20, 20))

        assert len(out) == int((probs > 0.7).sum())

    def test_many_spots_without_prob_attribute_are_kept(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((600, 2)), details=object())
        _install_spotiflow(monkeypatch, model)

        out = _direct(np.arange(400).reshape(20, 20))

        assert len(out) == 600


class TestDetectSpotsEnv:
    """Pins the argument dict handed to the dedicated environment."""

    def test_arguments_are_forwarded_and_points_returned(self, monkeypatch):
        seen = {}
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])

        def fake_run(func_name, args_dict):
            seen["func_name"] = func_name
            seen["args"] = args_dict
            return {"points": expected}

        monkeypatch.setattr(sd, "run_spotiflow_in_env", fake_run)

        image = np.zeros((6, 6), dtype=np.uint16)
        out = sd._detect_spots_env(
            image,
            "YX",
            "hybiss",
            "/tmp/model",
            False,
            "skimage",
            "minmax",
            2.0,
            98.0,
            0.6,
            "(2,2)",
            False,
            "(1,1)",
            5,
            True,
        )

        assert out is expected
        assert seen["func_name"] == "detect_spots"
        args = seen["args"]
        assert args["image"] is image
        assert args["axes"] == "YX"
        assert args["pretrained_model"] == "hybiss"
        assert args["model_path"] == "/tmp/model"
        assert args["subpixel"] is False
        assert args["peak_mode"] == "skimage"
        assert args["normalizer"] == "minmax"
        assert args["normalizer_low"] == 2.0
        assert args["normalizer_high"] == 98.0
        assert args["prob_thresh"] == 0.6
        assert args["n_tiles"] == "(2,2)"
        assert args["exclude_border"] is False
        assert args["scale"] == "(1,1)"
        assert args["min_distance"] == 5
        assert args["force_cpu"] is True


class TestSaveCoordsCsv:
    """Pins CSV naming, dispatch and the pandas fallback writer."""

    def test_empty_input_path_is_a_no_op(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            sd,
            "_save_coords_csv_direct",
            lambda *a: called.append(a),
        )
        sd._save_coords_csv(np.zeros((1, 2)), "")
        assert called == []

    def test_csv_path_is_derived_from_input_file(self, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(
            sd,
            "_save_coords_csv_direct",
            lambda points, path: seen.append(path),
        )
        sd._save_coords_csv(
            np.zeros((1, 2)), str(tmp_path / "stack.tif"), use_env=False
        )
        assert seen == [str(tmp_path / "stack_spots.csv")]

    def test_use_env_dispatches_to_env_writer(self, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(
            sd,
            "_save_coords_csv_env",
            lambda points, path: seen.append(path),
        )
        sd._save_coords_csv(
            np.zeros((1, 2)), str(tmp_path / "stack.tif"), use_env=True
        )
        assert seen == [str(tmp_path / "stack_spots.csv")]

    def test_direct_writer_uses_spotiflow_when_available(
        self, monkeypatch, tmp_path
    ):
        seen = []
        _install_spotiflow(
            monkeypatch,
            write_coords=lambda points, path: seen.append((points, path)),
        )
        points = np.array([[1.0, 2.0]])
        target = str(tmp_path / "a.csv")
        sd._save_coords_csv_direct(points, target)
        assert seen[0][1] == target
        assert not (tmp_path / "a.csv").exists()

    def test_direct_writer_falls_back_to_pandas_2d(self, tmp_path):
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = tmp_path / "spots.csv"
        sd._save_coords_csv_direct(points, str(target))
        text = target.read_text()
        assert text.splitlines()[0] == "y,x"
        assert len(text.strip().splitlines()) == 3

    def test_direct_writer_falls_back_to_pandas_3d(self, tmp_path):
        points = np.array([[1.0, 2.0, 3.0]])
        target = tmp_path / "spots3d.csv"
        sd._save_coords_csv_direct(points, str(target))
        assert target.read_text().splitlines()[0] == "z,y,x"

    def test_env_writer_runs_script_and_cleans_up(self, monkeypatch, tmp_path):
        import os

        from napari_tmidas.processing_functions import (
            spotiflow_env_manager as sem,
        )

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(
            sem, "get_env_python_path", lambda: "/fake/env/bin/python"
        )

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs
            with open(cmd[1]) as handle:
                captured["script"] = handle.read()
            captured["points"] = np.load(
                captured["script"].split("np.load('")[1].split("')")[0]
            )
            return types.SimpleNamespace(stdout="done", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        csv_path = str(tmp_path / "out.csv")
        sd._save_coords_csv_env(points, csv_path)

        assert captured["cmd"][0] == "/fake/env/bin/python"
        assert captured["kwargs"]["check"] is True
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True
        assert "write_coords_csv" in captured["script"]
        assert csv_path in captured["script"]
        np.testing.assert_array_equal(captured["points"], points)
        # both temporary files are removed afterwards
        assert not os.path.exists(captured["cmd"][1])


class TestHeatmapConversion:
    """Pins the heatmap/watershed converter and its fallbacks."""

    def _details(self, heatmap):
        return types.SimpleNamespace(heatmap=heatmap)

    def test_missing_spotiflow_falls_back_to_point_mask(self, capsys):
        points = np.array([[5.0, 6.0]])
        out = sd._convert_points_to_labels_with_heatmap(
            np.zeros((20, 20)), points, 1, "general", "", None, True
        )
        assert out.shape == (20, 20)
        assert out[5, 6] == 1
        assert "Error in heatmap-based conversion" in capsys.readouterr().out

    def test_heatmap_watershed_labels_seeded_points(self, monkeypatch):
        _install_torch(monkeypatch)
        heatmap = np.zeros((20, 20))
        heatmap[4:8, 4:8] = 0.9
        heatmap[14:18, 14:18] = 0.9
        model = _FakeModel(
            points=np.zeros((0, 2)), details=self._details(heatmap)
        )
        _install_spotiflow(monkeypatch, model)

        points = np.array([[5.0, 5.0], [15.0, 15.0]])
        out = sd._convert_points_to_labels_with_heatmap(
            np.zeros((20, 20)), points, 1, "general", "", 0.5, True
        )

        assert out.dtype == np.uint16
        assert out[5, 5] == 1
        assert out[15, 15] == 2
        assert out[0, 0] == 0

    def test_heatmap_without_points_uses_connected_components(
        self, monkeypatch
    ):
        _install_torch(monkeypatch)
        heatmap = np.zeros((20, 20))
        heatmap[2:5, 2:5] = 0.9
        heatmap[12:15, 12:15] = 0.9
        model = _FakeModel(
            points=np.zeros((0, 2)), details=self._details(heatmap)
        )
        _install_spotiflow(monkeypatch, model)

        out = sd._convert_points_to_labels_with_heatmap(
            np.zeros((20, 20)), np.zeros((0, 2)), 1, "general", "", None, True
        )

        assert out.dtype == np.uint16
        assert int(out.max()) == 2

    def test_details_without_heatmap_falls_back(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((0, 2)), details=object())
        _install_spotiflow(monkeypatch, model)

        points = np.array([[5.0, 6.0]])
        out = sd._convert_points_to_labels_with_heatmap(
            np.zeros((20, 20)), points, 1, "general", "", None, True
        )
        assert out[5, 6] == 1

    def test_custom_model_folder_is_used(self, monkeypatch, tmp_path):
        _install_torch(monkeypatch, cuda_available=True)
        model = _FakeModel(points=np.zeros((0, 2)), details=object())
        spot = _install_spotiflow(monkeypatch, model)
        spot.loaded.clear()

        folder = tmp_path / "model"
        folder.mkdir()
        sd._convert_points_to_labels_with_heatmap(
            np.zeros((20, 20)),
            np.zeros((0, 2)),
            1,
            "general",
            str(folder),
            None,
            False,
        )

        assert spot.loaded == [("folder", str(folder))]
        assert model.to_calls == ["dev:cuda"]


class TestSpotiflowDetectSpots:
    """Pins the registered wrapper: axes, dispatch, CSV and mask."""

    def test_dedicated_env_branch_returns_label_mask(self, monkeypatch):
        points = np.array([[5.0, 5.0], [15.0, 15.0]])
        monkeypatch.setattr(sd, "_detect_spots_env", lambda *a, **k: points)
        image = np.zeros((20, 20), dtype=np.uint16)

        out = sd.spotiflow_detect_spots(
            image, force_dedicated_env=True, output_csv=False, spot_radius=2
        )

        assert out.shape == image.shape
        assert out.dtype == np.uint16
        assert set(np.unique(out)) == {0, 1, 2}

    def test_explicit_axes_are_forwarded(self, monkeypatch, capsys):
        seen = {}

        def fake_env(image, axes, *rest):
            seen["axes"] = axes
            return np.zeros((0, 2))

        monkeypatch.setattr(sd, "_detect_spots_env", fake_env)
        sd.spotiflow_detect_spots(
            np.zeros((4, 20, 20)),
            axes="ZYX",
            force_dedicated_env=True,
            output_csv=False,
        )
        assert seen["axes"] == "ZYX"
        assert "Using provided axes: ZYX" in capsys.readouterr().out

    def test_auto_axes_are_inferred(self, monkeypatch):
        seen = {}

        def fake_env(image, axes, *rest):
            seen["axes"] = axes
            return np.zeros((0, 2))

        monkeypatch.setattr(sd, "_detect_spots_env", fake_env)
        sd.spotiflow_detect_spots(
            np.zeros((4, 20, 20)),
            force_dedicated_env=True,
            output_csv=False,
        )
        assert seen["axes"] == "ZYX"

    def test_direct_branch_is_taken_when_spotiflow_available(
        self, monkeypatch
    ):
        monkeypatch.setattr(sd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(sd, "SPOTIFLOW_AVAILABLE", True)
        called = []

        def fake_direct(*args):
            called.append(args)
            return np.array([[3.0, 4.0]])

        monkeypatch.setattr(sd, "_detect_spots_direct", fake_direct)
        monkeypatch.setattr(
            sd,
            "_detect_spots_env",
            lambda *a: pytest.fail("env branch must not run"),
        )

        out = sd.spotiflow_detect_spots(
            np.zeros((20, 20), dtype=np.uint16),
            output_csv=False,
            spot_radius=1,
        )

        assert len(called) == 1
        assert out[3, 4] == 1

    def test_force_dedicated_env_overrides_available_direct_mode(
        self, monkeypatch
    ):
        # Even when Spotiflow is importable (USE_DEDICATED_ENV is False and
        # SPOTIFLOW_AVAILABLE is True, i.e. direct mode is fully eligible),
        # force_dedicated_env=True must still route through the dedicated
        # environment. Every other test that passes force_dedicated_env=True
        # runs with USE_DEDICATED_ENV already True (Spotiflow is genuinely
        # absent in this environment), so "or force_dedicated_env" is never
        # the deciding operand there: `use_env = USE_DEDICATED_ENV` alone
        # would satisfy them just as well. This test isolates the flag.
        monkeypatch.setattr(sd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(sd, "SPOTIFLOW_AVAILABLE", True)
        monkeypatch.setattr(
            sd,
            "_detect_spots_direct",
            lambda *a: pytest.fail("direct branch must not run"),
        )
        called = []

        def fake_env(*args):
            called.append(args)
            return np.zeros((0, 2))

        monkeypatch.setattr(sd, "_detect_spots_env", fake_env)

        sd.spotiflow_detect_spots(
            np.zeros((20, 20), dtype=np.uint16),
            force_dedicated_env=True,
            output_csv=False,
        )

        assert len(called) == 1

    def test_csv_is_written_next_to_the_input_file(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(sd, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(sd, "SPOTIFLOW_AVAILABLE", False)
        monkeypatch.setattr(
            sd,
            "_detect_spots_env",
            lambda *a: np.array([[5.0, 6.0]]),
        )
        image_path = tmp_path / "sample.tif"

        sd.spotiflow_detect_spots(
            np.zeros((20, 20), dtype=np.uint16),
            output_csv=True,
            input_file_path=str(image_path),
        )

        csv_file = tmp_path / "sample_spots.csv"
        assert csv_file.exists()
        assert csv_file.read_text().splitlines()[0] == "y,x"

    def test_csv_requested_without_path_is_skipped(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sd,
            "_detect_spots_env",
            lambda *a: np.zeros((0, 2)),
        )
        sd.spotiflow_detect_spots(
            np.zeros((20, 20), dtype=np.uint16),
            force_dedicated_env=True,
            output_csv=True,
        )
        assert "skipping CSV export" in capsys.readouterr().out

    def test_csv_disabled_does_not_call_the_writer(self, monkeypatch):
        monkeypatch.setattr(
            sd,
            "_detect_spots_env",
            lambda *a: np.zeros((0, 2)),
        )
        monkeypatch.setattr(
            sd,
            "_save_coords_csv",
            lambda *a: pytest.fail("CSV writer must not run"),
        )
        out = sd.spotiflow_detect_spots(
            np.zeros((20, 20), dtype=np.uint16),
            force_dedicated_env=True,
            output_csv=False,
            input_file_path="/nowhere/img.tif",
        )
        assert not out.any()

    def test_3d_stack_returns_matching_label_volume(self, monkeypatch):
        monkeypatch.setattr(
            sd,
            "_detect_spots_env",
            lambda *a: np.array([[2.0, 8.0, 8.0]]),
        )
        image = np.zeros((6, 20, 20), dtype=np.uint16)
        out = sd.spotiflow_detect_spots(
            image, force_dedicated_env=True, output_csv=False, spot_radius=1
        )
        assert out.shape == image.shape
        assert out[2, 8, 8] == 1

    def test_timelapse_stack_labels_every_timepoint(self, monkeypatch):
        monkeypatch.setattr(
            sd,
            "_detect_spots_env",
            lambda *a: np.array([[2.0, 8.0, 8.0]]),
        )
        image = np.zeros((3, 6, 20, 20), dtype=np.uint16)
        out = sd.spotiflow_detect_spots(
            image, force_dedicated_env=True, output_csv=False, spot_radius=1
        )
        assert out.shape == image.shape
        assert (out[:, 2, 8, 8] == 1).all()


class TestRegistration:
    """Pins the registry entry the batch-processing widget consumes."""

    def test_function_is_registered(self):
        info = BatchProcessingRegistry.get_function_info(
            "Spotiflow Spot Detection"
        )
        assert info is not None
        assert info["func"] is sd.spotiflow_detect_spots
        assert info["suffix"] == "_spot_labels"
        assert "Spotiflow" in info["description"]

    def test_registered_parameter_metadata(self):
        info = BatchProcessingRegistry.get_function_info(
            "Spotiflow Spot Detection"
        )
        params = info["parameters"]
        assert params["channel"]["widget_type"] == "channel_selector"
        assert params["pretrained_model"]["default"] == "general"
        assert "smfish_3d" in params["pretrained_model"]["choices"]
        assert params["peak_mode"]["choices"] == ["fast", "skimage"]
        assert params["normalizer"]["choices"] == ["percentile", "minmax"]
        assert params["spot_radius"]["default"] == 3
        assert params["output_csv"]["default"] is True
        assert params["force_dedicated_env"]["default"] is False

    def test_alias_points_at_the_same_function(self):
        assert sd.spotiflow_spot_detection is sd.spotiflow_detect_spots

    def test_module_falls_back_to_dedicated_environment(self):
        # spotiflow is genuinely absent here
        assert sd.SPOTIFLOW_AVAILABLE is False
        assert sd.USE_DEDICATED_ENV is True


class TestDrawFailuresAreContained:
    """A failure while rasterising one spot must not abort the mask."""

    def test_2d_draw_error_is_reported_and_skipped(self, monkeypatch, capsys):
        from skimage import draw

        def boom(*args, **kwargs):
            raise ValueError("bad disk")

        monkeypatch.setattr(draw, "disk", boom)
        points = np.array([[5.0, 5.0]])
        mask = sd._points_to_label_mask(points, (20, 20), spot_radius=2)

        assert not mask.any()
        out = capsys.readouterr().out
        assert "Error drawing spot 0 at (5, 5): bad disk" in out
        assert "Successfully created 0 spots" in out

    def test_3d_draw_error_is_reported_and_skipped(self, monkeypatch, capsys):
        from scipy import ndimage

        def boom(*args, **kwargs):
            raise ValueError("bad ball")

        monkeypatch.setattr(ndimage, "iterate_structure", boom)
        points = np.array([[2.0, 8.0, 8.0]])
        mask = sd._points_to_label_mask(points, (6, 20, 20), spot_radius=1)

        assert not mask.any()
        out = capsys.readouterr().out
        assert "Error drawing 3D spot 0 at (2, 8, 8): bad ball" in out


class TestParseParamFallback:
    """A malformed tuple string falls back to the default value."""

    def test_malformed_tuple_falls_back_to_default(self, monkeypatch):
        _install_torch(monkeypatch)
        model = _FakeModel(points=np.zeros((1, 2)))
        _install_spotiflow(monkeypatch, model)

        _direct(np.arange(400).reshape(20, 20), n_tiles="(2,")

        assert "n_tiles" not in model.predict_calls[0][1]


class TestAvailabilityProbe:
    """Pins the module-level probe choosing direct vs dedicated env."""

    @staticmethod
    def _fake_pkg(name, with_path=False):
        mod = types.ModuleType(name)
        mod.__spec__ = machinery.ModuleSpec(name, None)
        if with_path:
            mod.__path__ = []
        return mod

    def test_importable_spotiflow_enables_direct_mode(self, monkeypatch):
        pkg = self._fake_pkg("spotiflow", with_path=True)
        model_mod = self._fake_pkg("spotiflow.model")
        pkg.model = model_mod
        monkeypatch.setitem(sys.modules, "spotiflow", pkg)
        monkeypatch.setitem(sys.modules, "spotiflow.model", model_mod)
        try:
            importlib.reload(sd)
            assert sd.SPOTIFLOW_AVAILABLE is True
            assert sd.USE_DEDICATED_ENV is False
        finally:
            monkeypatch.undo()
            importlib.reload(sd)
        assert sd.SPOTIFLOW_AVAILABLE is False
        assert sd.USE_DEDICATED_ENV is True

    def test_missing_submodule_raises_and_falls_back(
        self, monkeypatch, capsys
    ):
        # parent package present but ``spotiflow.model`` is not found:
        # find_spec returns None and the module raises its own ImportError
        pkg = self._fake_pkg("spotiflow", with_path=True)
        monkeypatch.setitem(sys.modules, "spotiflow", pkg)
        try:
            importlib.reload(sd)
            assert sd.SPOTIFLOW_AVAILABLE is False
            assert sd.USE_DEDICATED_ENV is True
            assert "will use dedicated environment" in capsys.readouterr().out
        finally:
            monkeypatch.undo()
            importlib.reload(sd)
        assert sd.spotiflow_spot_detection is sd.spotiflow_detect_spots

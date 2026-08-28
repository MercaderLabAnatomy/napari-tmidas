# src/napari_tmidas/_tests/test_convpaint_viscy_prediction_coverage.py
"""
Branch coverage for the convpaint and VisCy prediction wrappers.

Neither ``napari-convpaint`` nor ``viscy`` (nor ``torch``) is installed in
this environment, so both modules normally take only their "dependency
missing" path.  These tests inject fakes for the *collaborators* -- the
environment managers, the vendor SDK modules and ``torch`` -- and then run
the real dispatch, batching, post-processing and error handling code of
``convpaint_prediction`` and ``viscy_virtual_staining``.
"""
import importlib.util
import os
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import convpaint_env_manager as cem
from napari_tmidas.processing_functions import convpaint_prediction as cp
from napari_tmidas.processing_functions import viscy_env_manager as vem
from napari_tmidas.processing_functions import viscy_virtual_staining as vvs

# ---------------------------------------------------------------------------
# Fakes shared by the convpaint and VisCy suites
# ---------------------------------------------------------------------------


class _FakeTensor:
    """The handful of ``torch.Tensor`` methods these modules actually use."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.on_gpu = False

    def __getitem__(self, item):
        return _FakeTensor(self.array[item])

    def cuda(self):
        moved = _FakeTensor(self.array)
        moved.on_gpu = True
        return moved

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _make_fake_torch(cuda_available=False, mps_available=False):
    """Build a stand-in ``torch`` module with a call log attached."""
    module = types.ModuleType("torch")
    log = {"empty_cache": 0, "mps_empty_cache": 0, "devices": []}

    def _device(name):
        log["devices"].append(str(name))
        return f"device:{name}"

    def _empty_cache():
        log["empty_cache"] += 1

    def _mps_empty_cache():
        log["mps_empty_cache"] += 1

    @contextmanager
    def _no_grad():
        yield

    module.device = _device
    module.from_numpy = _FakeTensor
    module.no_grad = _no_grad
    module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available, empty_cache=_empty_cache
    )
    module.mps = types.SimpleNamespace(empty_cache=_mps_empty_cache)
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available)
    )
    module.log = log
    return module


class _FakeInnerModel:
    def __init__(self):
        self.moved_to = []

    def cpu(self):
        self.moved_to.append("cpu")
        return self

    def to(self, device):
        self.moved_to.append(str(device))
        return self


class _FakeFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = _FakeInnerModel()


def _make_fake_convpaint_model(segment=None, fe_device="cuda:0"):
    """Return a ``ConvpaintModel`` stand-in plus a record of its instances."""
    record = {"instances": []}

    class _Param:
        def __init__(self):
            self.fe_use_gpu = True
            self.image_downsample = 1

    class _Model:
        def __init__(self, model_path=None):
            self.model_path = model_path
            self._param = _Param()
            self.classifier = object()
            self.fe_model = _FakeFeatureExtractor(fe_device)
            self.set_params_calls = []
            record["instances"].append(self)

        def set_params(self, **kwargs):
            self.set_params_calls.append(kwargs)
            self._param.image_downsample = kwargs.get("image_downsample", 1)

        def segment(self, image):
            if segment is not None:
                return segment(image)
            return np.where(np.asarray(image) > 0, 2, 1).astype(np.uint32)

    return _Model, record


def _load_module_copy(name, path, registry_keys):
    """
    Execute a fresh copy of a module file, then undo its registration.

    The module-level ``try: import <vendor>`` blocks only run at import
    time, so the "dependency present" branch is unreachable from the
    already-imported module object.  Re-executing the file with a fake in
    ``sys.modules`` runs it; the registry entries it overwrites are put
    back so no other test sees a duplicate function object.
    """
    saved = {
        key: BatchProcessingRegistry._processing_functions.get(key)
        for key in registry_keys
    }
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        with BatchProcessingRegistry._lock:
            for key, value in saved.items():
                if value is None:
                    BatchProcessingRegistry._processing_functions.pop(
                        key, None
                    )
                else:
                    BatchProcessingRegistry._processing_functions[key] = value
    return module


class _LazyArray:
    """Minimal Dask-like array: has ``.compute`` so it must be materialized."""

    def __init__(self, array):
        self._array = array
        self.ndim = array.ndim
        self.shape = array.shape
        self.dtype = array.dtype
        self.computed = 0

    def compute(self):
        self.computed += 1
        return self._array

    def __array__(self, dtype=None, copy=None):
        self.computed += 1
        if dtype is None:
            return self._array
        return self._array.astype(dtype)


@pytest.fixture()
def model_file(tmp_path):
    """A path that passes convpaint's ``os.path.exists`` model check."""
    path = tmp_path / "model.pkl"
    path.write_text("stub")
    return str(path)


@pytest.fixture()
def env_calls(monkeypatch):
    """Replace the out-of-process convpaint call with an in-process stub."""
    calls = []

    def fake_run(
        image,
        model_path,
        image_downsample=2,
        use_cpu=False,
        tmp_dir=None,
        gpu_id=None,
    ):
        array = np.asarray(image)
        calls.append(
            {
                "shape": array.shape,
                "model_path": model_path,
                "image_downsample": image_downsample,
                "use_cpu": use_cpu,
                "tmp_dir": tmp_dir,
                "gpu_id": gpu_id,
            }
        )
        return np.where(array > 0, 2, 1).astype(np.uint32)

    monkeypatch.setattr(cp, "run_convpaint_in_env", fake_run)
    return calls


# ---------------------------------------------------------------------------
# Module-level optional dependency detection
# ---------------------------------------------------------------------------


class TestOptionalDependencyDetection:
    """The import-time flags that pick native vs dedicated-environment."""

    def test_convpaint_present_selects_the_native_backend(
        self, monkeypatch
    ):
        """A resolvable napari_convpaint flips both module flags."""
        sentinel = type("ConvpaintModel", (), {})
        backend = types.ModuleType("napari_convpaint.convpaint_model")
        backend.ConvpaintModel = sentinel
        package = types.ModuleType("napari_convpaint")
        package.convpaint_model = backend
        package.ConvpaintModel = sentinel
        monkeypatch.setitem(sys.modules, "napari_convpaint", package)
        monkeypatch.setitem(
            sys.modules, "napari_convpaint.convpaint_model", backend
        )

        fresh = _load_module_copy(
            "convpaint_prediction_fresh", cp.__file__, ["Convpaint Prediction"]
        )

        assert fresh.CONVPAINT_AVAILABLE is True
        assert fresh.USE_DEDICATED_ENV is False
        assert fresh.ConvpaintModel is sentinel
        # The live registry entry must survive the re-execution.
        info = BatchProcessingRegistry.get_function_info("Convpaint Prediction")
        assert info["func"] is cp.convpaint_predict

    def test_convpaint_absent_selects_the_dedicated_environment(self):
        """With convpaint uninstalled the live module is in fallback mode."""
        assert cp.CONVPAINT_AVAILABLE is False
        assert cp.USE_DEDICATED_ENV is True

    def test_viscy_present_flips_the_availability_flag(self, monkeypatch):
        """An importable viscy makes the module take the native path."""
        monkeypatch.setitem(sys.modules, "viscy", types.ModuleType("viscy"))

        fresh = _load_module_copy(
            "viscy_virtual_staining_fresh",
            vvs.__file__,
            ["VisCy Virtual Staining"],
        )

        assert fresh.VISCY_AVAILABLE is True
        info = BatchProcessingRegistry.get_function_info(
            "VisCy Virtual Staining"
        )
        assert info["func"] is vvs.viscy_virtual_staining


# ---------------------------------------------------------------------------
# convpaint_predict dispatch
# ---------------------------------------------------------------------------


class TestConvpaintPredictDispatch:
    """How convpaint_predict routes each input rank to a segmenter."""

    def test_2d_input_makes_one_call_and_returns_uint32(
        self, model_file, env_calls
    ):
        """YX input is one dedicated-env call; output is uint32 labels."""
        image = np.array([[0, 5], [7, 0]], dtype=np.uint16)

        result = cp.convpaint_predict(
            image,
            model_path=model_file,
            image_downsample=1,
            background_label=1,
        )

        assert len(env_calls) == 1
        assert env_calls[0]["shape"] == (2, 2)
        assert env_calls[0]["model_path"] == model_file
        assert result.dtype == np.uint32
        # background_label 1 is mapped to 0, foreground class 2 survives.
        assert np.array_equal(result, np.array([[0, 2], [2, 0]]))

    def test_3d_zstack_without_batching_makes_one_call(
        self, model_file, env_calls
    ):
        """A short ZYX stack goes through as a single volume."""
        image = np.zeros((3, 2, 2), dtype=np.uint8)
        image[1] = 4

        result = cp.convpaint_predict(
            image,
            model_path=model_file,
            image_downsample=1,
            background_label=0,
            z_batch_size=0,
        )

        assert len(env_calls) == 1
        assert env_calls[0]["shape"] == (3, 2, 2)
        assert result.shape == (3, 2, 2)
        assert set(np.unique(result)) == {1, 2}

    def test_z_batching_splits_the_stack_and_announces_itself(
        self, model_file, env_calls, capsys
    ):
        """z_batch_size>0 slices the stack into ceil(Z/batch) calls."""
        image = np.ones((5, 2, 2), dtype=np.uint8)

        result = cp.convpaint_predict(
            image,
            model_path=model_file,
            image_downsample=1,
            background_label=0,
            z_batch_size=2,
        )

        assert [call["shape"] for call in env_calls] == [
            (2, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
        ]
        assert result.shape == (5, 2, 2)
        out = capsys.readouterr().out
        assert "Z-batching enabled: 2 planes per batch" in out

    def test_tyx_time_series_is_segmented_per_timepoint(
        self, model_file, env_calls
    ):
        """A 3D input with >=100 leading frames is treated as TYX."""
        image = np.zeros((100, 2, 2), dtype=np.uint8)
        image[:, 0, 0] = 9

        result = cp.convpaint_predict(
            image,
            model_path=model_file,
            image_downsample=1,
            background_label=1,
        )

        assert len(env_calls) == 100
        assert env_calls[0]["shape"] == (2, 2)
        assert result.shape == (100, 2, 2)
        assert np.array_equal(result[:, 0, 0], np.full(100, 2))
        assert np.array_equal(result[:, 1, 1], np.zeros(100))

    def test_tzyx_without_output_folder_uses_the_in_memory_path(
        self, model_file, env_calls
    ):
        """Streaming needs an output folder; without one TZYX stays in RAM."""
        image = np.ones((2, 3, 2, 2), dtype=np.uint8)

        result = cp.convpaint_predict(
            image,
            model_path=model_file,
            image_downsample=1,
            background_label=0,
        )

        assert [call["shape"] for call in env_calls] == [
            (3, 2, 2),
            (3, 2, 2),
        ]
        assert result.shape == (2, 3, 2, 2)
        assert result.dtype == np.uint32

    def test_lazy_single_volume_is_materialized_first(
        self, model_file, env_calls, capsys
    ):
        """A non-time-series lazy array is computed before segmenting."""
        lazy = _LazyArray(np.ones((3, 2, 2), dtype=np.uint8))

        result = cp.convpaint_predict(
            lazy,
            model_path=model_file,
            image_downsample=1,
            background_label=0,
        )

        assert lazy.computed >= 1
        assert "Materializing lazy input" in capsys.readouterr().out
        assert result.shape == (3, 2, 2)

    @pytest.mark.parametrize("shape", [(4,), (2, 2, 2, 2, 2)])
    def test_unsupported_rank_raises(self, shape, model_file, env_calls):
        """Only 2D/3D/4D inputs are accepted."""
        image = np.zeros(shape, dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported image dimensions"):
            cp.convpaint_predict(image, model_path=model_file)

        assert env_calls == []

    def test_instance_output_relabels_each_component(
        self, model_file, env_calls
    ):
        """output_type='instance' splits one class into components."""
        image = np.zeros((5, 5), dtype=np.uint8)
        image[0:2, 0:2] = 3
        image[3:5, 3:5] = 3

        result = cp.convpaint_predict(
            image,
            model_path=model_file,
            image_downsample=1,
            background_label=1,
            output_type="instance",
        )

        labels = set(np.unique(result)) - {0}
        assert labels == {1, 2}
        assert result.dtype == np.uint32

    def test_native_backend_is_used_when_convpaint_is_available(
        self, model_file, monkeypatch
    ):
        """With convpaint importable the in-process segmenter is called."""
        seen = []

        def fake_segment(
            image, model_path, image_downsample, use_cpu, gpu_id=None
        ):
            seen.append((np.asarray(image).shape, gpu_id))
            return np.full(np.asarray(image).shape, 2, dtype=np.uint32)

        monkeypatch.setattr(cp, "CONVPAINT_AVAILABLE", True)
        monkeypatch.setattr(cp, "USE_DEDICATED_ENV", False)
        monkeypatch.setattr(cp, "_segment_with_convpaint", fake_segment)

        flat = cp.convpaint_predict(
            np.ones((2, 2), dtype=np.uint8),
            model_path=model_file,
            image_downsample=1,
            background_label=0,
        )
        stack = cp.convpaint_predict(
            np.ones((3, 2, 2), dtype=np.uint8),
            model_path=model_file,
            image_downsample=1,
            background_label=0,
        )

        assert [shape for shape, _ in seen] == [(2, 2), (3, 2, 2)]
        assert flat.shape == (2, 2)
        assert stack.shape == (3, 2, 2)


class TestHelperNativeBranches:
    """The ``use_dedicated=False`` arms of the two loop helpers."""

    def test_time_series_helper_calls_the_in_process_segmenter(
        self, monkeypatch
    ):
        """_process_time_series can run convpaint natively per timepoint."""
        seen = []

        def fake_segment(image, model_path, image_downsample, use_cpu):
            seen.append(np.asarray(image).shape)
            return np.full(np.asarray(image).shape, 3, dtype=np.uint32)

        monkeypatch.setattr(cp, "_segment_with_convpaint", fake_segment)

        result = cp._process_time_series(
            np.ones((2, 2, 2), dtype=np.uint8),
            "model.pkl",
            1,
            use_dedicated=False,
            use_cpu=False,
        )

        assert seen == [(2, 2), (2, 2)]
        assert result.shape == (2, 2, 2)
        assert np.all(result == 3)

    def test_zyx_batch_helper_calls_the_in_process_segmenter(
        self, monkeypatch
    ):
        """_process_zyx_in_batches can run convpaint natively per batch."""
        seen = []

        def fake_segment(
            image, model_path, image_downsample, use_cpu, gpu_id=None
        ):
            seen.append((np.asarray(image).shape, gpu_id))
            return np.full(np.asarray(image).shape, 7, dtype=np.uint32)

        monkeypatch.setattr(cp, "_segment_with_convpaint", fake_segment)

        result = cp._process_zyx_in_batches(
            np.ones((5, 2, 2), dtype=np.uint8),
            "model.pkl",
            1,
            use_dedicated=False,
            use_cpu=False,
            z_batch_size=3,
            gpu_id=1,
        )

        assert seen == [((3, 2, 2), 1), ((2, 2, 2), 1)]
        assert result.shape == (5, 2, 2)
        assert np.all(result == 7)


# ---------------------------------------------------------------------------
# GPU assignment
# ---------------------------------------------------------------------------


class TestResolveGpuAssignment:
    """Worker clamping and round-robin device pinning."""

    @pytest.mark.parametrize("bad", ["two", None, 3.5j])
    def test_non_integer_worker_count_falls_back_to_one(self, bad):
        """A worker count that will not cast to int degrades to 1."""
        n_workers, gpu_ids = cp._resolve_gpu_assignment(
            bad, use_cpu=True, n_tasks=4
        )

        assert n_workers == 1
        assert gpu_ids == [None]

    def test_no_visible_devices_puts_every_worker_on_cpu(
        self, monkeypatch, capsys
    ):
        """An empty device list is not an error, it is CPU mode."""
        monkeypatch.setattr(cem, "detect_gpu_ids", list)

        n_workers, gpu_ids = cp._resolve_gpu_assignment(
            2, use_cpu=False, n_tasks=5
        )

        assert (n_workers, gpu_ids) == (2, [None, None])
        assert "No CUDA devices detected" in capsys.readouterr().out

    def test_devices_are_assigned_round_robin(self, monkeypatch):
        """Worker i gets devices[i % len(devices)]."""
        monkeypatch.setattr(cem, "detect_gpu_ids", lambda: [0, 1])

        n_workers, gpu_ids = cp._resolve_gpu_assignment(
            4, use_cpu=False, n_tasks=8
        )

        assert (n_workers, gpu_ids) == (4, [0, 1, 0, 1])

    def test_oversubscribing_a_device_warns(self, monkeypatch, capsys):
        """More workers than GPUs is allowed but flagged."""
        monkeypatch.setattr(cem, "detect_gpu_ids", lambda: [0])

        n_workers, gpu_ids = cp._resolve_gpu_assignment(
            3, use_cpu=False, n_tasks=5
        )

        assert (n_workers, gpu_ids) == (3, [0, 0, 0])
        out = capsys.readouterr().out
        assert "exceeds the 1" in out
        assert "exhaust GPU memory" in out


# ---------------------------------------------------------------------------
# Per-timepoint post-processing
# ---------------------------------------------------------------------------


class TestPostprocessTimepoint:
    """Background removal plus optional connected components, per frame."""

    def test_semantic_only_drops_the_background_class(self):
        """Semantic output keeps class ids and zeroes the background."""
        labels = np.array([[1, 2], [2, 1]], dtype=np.uint8)

        out = cp._postprocess_timepoint(
            labels, background_label=1, output_type="semantic", is_3d=False
        )

        assert out.dtype == np.uint32
        assert np.array_equal(out, np.array([[0, 2], [2, 0]]))

    def test_instance_2d_splits_components_within_a_class(self):
        """is_3d=False labels the frame with 2D connectivity."""
        labels = np.ones((5, 5), dtype=np.uint8)
        labels[0:2, 0:2] = 2
        labels[3:5, 3:5] = 2

        out = cp._postprocess_timepoint(
            labels, background_label=1, output_type="instance", is_3d=False
        )

        assert out.dtype == np.uint32
        assert set(np.unique(out)) == {0, 1, 2}

    def test_instance_3d_treats_the_volume_as_one_object(self):
        """is_3d=True labels across Z, so a column is a single instance."""
        labels = np.ones((3, 4, 4), dtype=np.uint8)
        labels[:, 0:2, 0:2] = 2

        out = cp._postprocess_timepoint(
            labels, background_label=1, output_type="instance", is_3d=True
        )

        assert set(np.unique(out)) == {0, 1}

    def test_background_label_zero_is_left_alone(self):
        """background_label=0 means the input already uses 0 for background."""
        labels = np.array([[0, 3]], dtype=np.uint8)

        out = cp._postprocess_timepoint(
            labels, background_label=0, output_type="semantic", is_3d=False
        )

        assert np.array_equal(out, np.array([[0, 3]]))


# ---------------------------------------------------------------------------
# Streaming time-series writer
# ---------------------------------------------------------------------------


def _write_source_tif(path, data):
    import tifffile

    tifffile.imwrite(str(path), data)
    return str(path)


class TestStreamingSegmentation:
    """The per-timepoint arms of _segment_time_series_streaming."""

    def test_3d_timepoints_are_z_batched(self, tmp_path, monkeypatch):
        """is_3d + z_batch_size routes each timepoint through the batcher."""
        import tifffile

        calls = []

        def fake_run(
            image,
            model_path,
            image_downsample=2,
            use_cpu=False,
            tmp_dir=None,
            gpu_id=None,
        ):
            array = np.asarray(image)
            calls.append((array.shape, gpu_id))
            return np.full(array.shape, 2, dtype=np.uint32)

        monkeypatch.setattr(cp, "run_convpaint_in_env", fake_run)

        data = np.ones((2, 4, 3, 3), dtype=np.uint8)
        source = _write_source_tif(tmp_path / "src.tif", data)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        written = cp._segment_time_series_streaming(
            data,
            model_path="model.pkl",
            image_downsample=1,
            use_dedicated=True,
            use_cpu=True,
            is_3d=True,
            z_batch_size=2,
            n_workers=1,
            background_label=0,
            output_type="semantic",
            tmp_dir=str(tmp_path / "scratch"),
            source_filepath=source,
            output_folder=str(out_dir),
            output_suffix="_labels",
            output_format="tiff",
        )

        assert calls == [((2, 3, 3), None)] * 4
        assert written == str(out_dir / "src_labels.tif")
        assert os.path.exists(written)
        assert tifffile.imread(written).shape == (2, 4, 3, 3)
        # The scratch buffer must not survive the call.
        assert not [p for p in tmp_path.glob("scratch/.convpaint-*")]

    def test_non_dedicated_stream_uses_the_in_process_segmenter(
        self, tmp_path, monkeypatch
    ):
        """use_dedicated=False streams through _segment_with_convpaint."""
        import tifffile

        calls = []

        def fake_segment(
            image, model_path, image_downsample, use_cpu, gpu_id=None
        ):
            array = np.asarray(image)
            calls.append((array.shape, gpu_id))
            return np.full(array.shape, 5, dtype=np.uint32)

        monkeypatch.setattr(cp, "_segment_with_convpaint", fake_segment)

        data = np.ones((3, 4, 4), dtype=np.uint8)
        source = _write_source_tif(tmp_path / "movie.tif", data)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        written = cp._segment_time_series_streaming(
            data,
            model_path="model.pkl",
            image_downsample=1,
            use_dedicated=False,
            use_cpu=True,
            is_3d=False,
            z_batch_size=0,
            n_workers=1,
            background_label=0,
            output_type="semantic",
            tmp_dir=None,
            source_filepath=source,
            output_folder=str(out_dir),
            output_suffix="_labels",
            output_format="tiff",
        )

        assert calls == [((4, 4), None)] * 3
        result = tifffile.imread(written)
        assert result.shape == (3, 4, 4)
        assert np.all(result == 5)


# ---------------------------------------------------------------------------
# In-process convpaint segmentation
# ---------------------------------------------------------------------------


class TestSegmentWithConvpaint:
    """Device handling and shape bookkeeping around ConvpaintModel."""

    def test_cpu_mode_hides_the_gpu_and_moves_the_extractor(
        self, monkeypatch
    ):
        """use_cpu disables CUDA for the process and demotes the model."""
        monkeypatch.setitem(sys.modules, "torch", _make_fake_torch())
        model_cls, record = _make_fake_convpaint_model(fe_device="cuda:0")
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

        image = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        out = cp._segment_with_convpaint(
            image, "m.pkl", image_downsample=1, use_cpu=True
        )

        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        model = record["instances"][0]
        assert model.model_path == "m.pkl"
        assert model._param.fe_use_gpu is False
        assert model.fe_model.device == "device:cpu"
        assert model.fe_model.model.moved_to == ["cpu"]
        assert np.array_equal(out, np.array([[1, 2], [2, 1]]))

    def test_gpu_id_pins_the_feature_extractor_to_that_device(
        self, monkeypatch
    ):
        """A worker's gpu_id moves the extractor onto cuda:<id>."""
        fake_torch = _make_fake_torch(cuda_available=True)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        model_cls, record = _make_fake_convpaint_model(fe_device="cuda:0")
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)

        cp._segment_with_convpaint(
            np.ones((2, 2), dtype=np.uint8),
            "m.pkl",
            image_downsample=1,
            use_cpu=False,
            gpu_id=1,
        )

        model = record["instances"][0]
        assert model.fe_model.device == "device:cuda:1"
        assert model.fe_model.model.moved_to == ["device:cuda:1"]
        assert fake_torch.log["empty_cache"] == 1

    def test_downsample_above_one_is_pushed_into_the_model(
        self, monkeypatch
    ):
        """image_downsample>1 configures the model, and only then."""
        monkeypatch.setitem(sys.modules, "torch", _make_fake_torch())
        model_cls, record = _make_fake_convpaint_model()
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)

        cp._segment_with_convpaint(
            np.ones((2, 2), dtype=np.uint8), "m.pkl", image_downsample=3
        )

        assert record["instances"][0].set_params_calls == [
            {
                "image_downsample": 3,
                "tile_annotations": False,
                "ignore_warnings": True,
            }
        ]

    def test_singleton_axes_are_squeezed_out(self, monkeypatch):
        """A (1, Y, X) prediction comes back as (Y, X)."""
        monkeypatch.setitem(sys.modules, "torch", _make_fake_torch())
        model_cls, _ = _make_fake_convpaint_model(
            segment=lambda image: np.ones((1,) + np.asarray(image).shape)
        )
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)

        out = cp._segment_with_convpaint(
            np.ones((3, 3), dtype=np.uint8), "m.pkl", image_downsample=1
        )

        assert out.shape == (3, 3)

    def test_shape_mismatch_is_reported_not_raised(self, monkeypatch, capsys):
        """A model that returns the wrong shape warns and returns anyway."""
        monkeypatch.setitem(sys.modules, "torch", _make_fake_torch())
        model_cls, _ = _make_fake_convpaint_model(
            segment=lambda image: np.ones((2, 2), dtype=np.uint32)
        )
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)

        out = cp._segment_with_convpaint(
            np.ones((4, 4), dtype=np.uint8), "m.pkl", image_downsample=1
        )

        assert out.shape == (2, 2)
        assert "Shape mismatch" in capsys.readouterr().out

    def test_mps_cache_is_cleared_when_cuda_is_absent(self, monkeypatch):
        """On Apple silicon the MPS allocator is emptied instead of CUDA."""
        fake_torch = _make_fake_torch(
            cuda_available=False, mps_available=True
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        model_cls, _ = _make_fake_convpaint_model()
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)

        cp._segment_with_convpaint(
            np.ones((2, 2), dtype=np.uint8), "m.pkl", image_downsample=1
        )

        assert fake_torch.log["mps_empty_cache"] == 1
        assert fake_torch.log["empty_cache"] == 0

    def test_without_torch_cpu_forcing_is_a_no_op(self, monkeypatch):
        """torch is optional; use_cpu simply cannot be enforced without it."""
        monkeypatch.setitem(sys.modules, "torch", None)
        model_cls, record = _make_fake_convpaint_model()
        monkeypatch.setattr(cp, "ConvpaintModel", model_cls, raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

        cp._segment_with_convpaint(
            np.ones((2, 2), dtype=np.uint8),
            "m.pkl",
            image_downsample=1,
            use_cpu=True,
        )

        # No torch means the GPU flag stays as the model shipped it.
        assert record["instances"][0]._param.fe_use_gpu is True
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


# ---------------------------------------------------------------------------
# Semantic -> instance conversion
# ---------------------------------------------------------------------------


class TestSemanticToInstance:
    """Rank dispatch and the scikit-image fallback."""

    def test_missing_scikit_image_returns_the_input(self, monkeypatch, capsys):
        """Without scikit-image the semantic labels pass through."""
        monkeypatch.setitem(sys.modules, "skimage", None)
        image = np.array([[0, 2], [2, 0]], dtype=np.uint32)

        out = cp._convert_semantic_to_instance(image)

        assert out is image
        assert "scikit-image not available" in capsys.readouterr().out

    def test_tyx_series_is_labelled_frame_by_frame(self):
        """A >=100 frame 3D input is time, so components never span T."""
        image = np.zeros((100, 4, 4), dtype=np.uint32)
        image[:, 0, 0] = 2
        image[:, 3, 3] = 2

        out = cp._convert_semantic_to_instance(image)

        assert out.shape == (100, 4, 4)
        assert out.dtype == np.uint32
        # Two objects per frame, and the ids restart every frame.
        for frame in out:
            assert set(np.unique(frame)) == {0, 1, 2}

    def test_tzyx_series_is_labelled_volume_by_volume(self):
        """4D input labels each timepoint as its own 3D volume."""
        image = np.zeros((2, 2, 4, 4), dtype=np.uint32)
        image[:, :, 0, 0] = 2

        out = cp._convert_semantic_to_instance(image)

        assert out.shape == (2, 2, 4, 4)
        for volume in out:
            assert set(np.unique(volume)) == {0, 1}

    def test_unsupported_rank_passes_through_with_a_warning(self, capsys):
        """1D input is not something connected components can handle."""
        image = np.array([0, 1, 1], dtype=np.uint32)

        out = cp._convert_semantic_to_instance(image)

        assert out is image
        assert "Unsupported dimensions" in capsys.readouterr().out

    def test_binary_mask_takes_the_single_class_path(self):
        """A mask whose max is 1 skips the per-class loop."""
        from skimage import measure

        mask = np.array(
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8
        )

        out = cp._apply_connected_components(mask, measure, ndim=2)

        assert out.dtype == np.uint32
        assert set(np.unique(out)) == {0, 1, 2}

    def test_multiclass_labels_are_offset_so_ids_stay_unique(self):
        """Class 2 and class 3 components must not collide."""
        from skimage import measure

        image = np.zeros((5, 5), dtype=np.uint8)
        image[0, 0] = 2
        image[0, 4] = 2
        image[4, 0] = 3

        out = cp._apply_connected_components(image, measure, ndim=2)

        assert set(np.unique(out)) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# VisCy: dedicated-environment path
# ---------------------------------------------------------------------------


@pytest.fixture()
def viscy_env(monkeypatch):
    """Stub out the VisCy environment manager entry points."""
    state = {"calls": [], "created": 0, "env_exists": True, "create_error": None}

    def fake_is_env_created():
        return state["env_exists"]

    def fake_create_env():
        state["created"] += 1
        if state["create_error"] is not None:
            raise state["create_error"]
        return True

    def fake_run(image, z_batch_size):
        array = np.asarray(image)
        state["calls"].append({"shape": array.shape, "z": z_batch_size})
        out = np.zeros((array.shape[0], 2) + array.shape[1:], dtype=np.float32)
        out[:, 0] = array
        out[:, 1] = array * 2
        return out

    monkeypatch.setattr(vvs, "is_env_created", fake_is_env_created)
    monkeypatch.setattr(vvs, "create_viscy_env", fake_create_env)
    monkeypatch.setattr(vvs, "run_viscy_in_env", fake_run)
    monkeypatch.setattr(vvs, "VISCY_AVAILABLE", False)
    return state


class TestViscyDedicatedEnvironment:
    """viscy_virtual_staining when viscy runs out of process."""

    def test_z_batch_size_is_forced_to_fifteen(self, viscy_env, capsys):
        """VSCyto3D only accepts 15; anything else is coerced with a note."""
        image = np.zeros((15, 4, 4), dtype=np.float32)

        vvs.viscy_virtual_staining(image, dim_order="ZYX", z_batch_size=8)

        assert viscy_env["calls"][0]["z"] == 15
        assert "requires z_batch_size=15" in capsys.readouterr().out

    def test_single_volume_returns_both_channels(self, viscy_env):
        """output_channel='both' keeps the (Z, C, Y, X) layout."""
        rng = np.random.default_rng(0)
        image = rng.random((15, 4, 4)).astype(np.float32)

        out = vvs.viscy_virtual_staining(image, dim_order="ZYX")

        assert out.shape == (15, 2, 4, 4)
        assert np.allclose(out[:, 0], image)
        assert np.allclose(out[:, 1], image * 2)

    @pytest.mark.parametrize(
        ("channel", "index"), [("nuclei", 0), ("membrane", 1)]
    )
    def test_single_channel_selection_drops_the_channel_axis(
        self, viscy_env, channel, index
    ):
        """Asking for one channel returns (Z, Y, X)."""
        rng = np.random.default_rng(1)
        image = rng.random((15, 4, 4)).astype(np.float32)

        out = vvs.viscy_virtual_staining(
            image, dim_order="ZYX", output_channel=channel
        )

        assert out.shape == (15, 4, 4)
        assert np.allclose(out, image * (1 if index == 0 else 2))

    def test_time_series_is_processed_timepoint_by_timepoint(self, viscy_env):
        """TZYX input yields one call per timepoint, stacked back together."""
        image = np.ones((2, 15, 4, 4), dtype=np.float32)

        out = vvs.viscy_virtual_staining(image, dim_order="TZYX")

        assert [call["shape"] for call in viscy_env["calls"]] == [
            (15, 4, 4),
            (15, 4, 4),
        ]
        assert out.shape == (2, 15, 2, 4, 4)

    def test_input_is_transposed_to_zyx_before_dispatch(self, viscy_env):
        """A YXZ acquisition is reordered, not rejected."""
        image = np.zeros((4, 5, 15), dtype=np.float32)

        vvs.viscy_virtual_staining(image, dim_order="YXZ")

        assert viscy_env["calls"][0]["shape"] == (15, 4, 5)

    def test_missing_environment_is_created_once(self, viscy_env, capsys):
        """A first run builds the dedicated env before predicting."""
        viscy_env["env_exists"] = False
        image = np.zeros((15, 4, 4), dtype=np.float32)

        vvs.viscy_virtual_staining(image, dim_order="ZYX")

        assert viscy_env["created"] == 1
        assert len(viscy_env["calls"]) == 1
        assert "VisCy environment not found" in capsys.readouterr().out

    def test_environment_creation_failure_becomes_a_runtime_error(
        self, viscy_env
    ):
        """A failed env build is re-raised with context, not swallowed."""
        viscy_env["env_exists"] = False
        viscy_env["create_error"] = OSError("no disk space")
        image = np.zeros((15, 4, 4), dtype=np.float32)

        with pytest.raises(
            RuntimeError, match="Failed to create VisCy environment"
        ):
            vvs.viscy_virtual_staining(image, dim_order="ZYX")

        assert viscy_env["calls"] == []


# ---------------------------------------------------------------------------
# VisCy: native in-process path
# ---------------------------------------------------------------------------


def _make_fake_vsunet():
    """A VSUNet stand-in that doubles the input into two channels."""
    record = {}

    class _Model:
        def __init__(self, checkpoint, architecture, model_config):
            self.checkpoint = checkpoint
            self.architecture = architecture
            self.model_config = model_config
            self.eval_calls = 0
            self.cuda_calls = 0
            self.batch_shapes = []

        def eval(self):
            self.eval_calls += 1

        def cuda(self):
            self.cuda_calls += 1
            return self

        def __call__(self, tensor):
            array = tensor.array  # (1, 1, Z, Y, X)
            self.batch_shapes.append(array.shape)
            return _FakeTensor(
                np.concatenate([array, array + 1.0], axis=1)
            )

    class _VSUNet:
        @staticmethod
        def load_from_checkpoint(
            checkpoint, architecture=None, model_config=None
        ):
            record["model"] = _Model(checkpoint, architecture, model_config)
            return record["model"]

    return _VSUNet, record


@pytest.fixture()
def viscy_native(monkeypatch):
    """Install fake ``viscy`` + ``torch`` and point at a fake checkpoint."""

    def _install(cuda_available=False):
        fake_torch = _make_fake_torch(cuda_available=cuda_available)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        vsunet, record = _make_fake_vsunet()
        engine = types.ModuleType("viscy.translation.engine")
        engine.VSUNet = vsunet
        translation = types.ModuleType("viscy.translation")
        translation.engine = engine
        package = types.ModuleType("viscy")
        package.translation = translation
        monkeypatch.setitem(sys.modules, "viscy", package)
        monkeypatch.setitem(sys.modules, "viscy.translation", translation)
        monkeypatch.setitem(
            sys.modules, "viscy.translation.engine", engine
        )

        monkeypatch.setattr(
            vem, "get_model_path", lambda: "/models/VSCyto3D.ckpt"
        )
        monkeypatch.setattr(vvs, "VISCY_AVAILABLE", True)
        return fake_torch, record

    return _install


class TestViscyNativeInference:
    """_run_viscy_native: batching, padding, normalisation, device moves."""

    def test_cpu_inference_returns_z_c_y_x(self, viscy_native):
        """One full batch: (Z, Y, X) in, (Z, 2, Y, X) out."""
        _, record = viscy_native(cuda_available=False)
        rng = np.random.default_rng(0)
        image = rng.random((15, 4, 4)).astype(np.float32)

        out = vvs._run_viscy_native(image, 15)

        assert out.shape == (15, 2, 4, 4)
        model = record["model"]
        assert model.checkpoint == "/models/VSCyto3D.ckpt"
        assert model.architecture == "fcmae"
        assert model.model_config["in_stack_depth"] == 15
        assert model.model_config["out_channels"] == 2
        assert model.eval_calls == 1
        assert model.cuda_calls == 0
        assert model.batch_shapes == [(1, 1, 15, 4, 4)]
        # Channel 1 is the fake model's channel 0 plus one.
        assert np.allclose(out[:, 1], out[:, 0] + 1.0)

    def test_inputs_are_percentile_normalised_before_inference(
        self, viscy_native
    ):
        """The batch is clipped to [0, 1] using the 1st/99th percentiles."""
        viscy_native(cuda_available=False)
        rng = np.random.default_rng(3)
        image = (rng.random((15, 4, 4)) * 1000).astype(np.float32)

        out = vvs._run_viscy_native(image, 15)

        low, high = np.percentile(image, [1, 99])
        expected = np.clip((image - low) / (high - low + 1e-8), 0, 1)
        assert np.allclose(out[:, 0], expected.astype(np.float32))

    def test_short_final_batch_is_edge_padded_then_trimmed(
        self, viscy_native
    ):
        """A 20-slice stack runs as 15 + 15 (padded) and returns 20."""
        _, record = viscy_native(cuda_available=False)
        rng = np.random.default_rng(2)
        image = rng.random((20, 4, 4)).astype(np.float32)

        out = vvs._run_viscy_native(image, 15)

        assert out.shape == (20, 2, 4, 4)
        assert record["model"].batch_shapes == [
            (1, 1, 15, 4, 4),
            (1, 1, 15, 4, 4),
        ]

    def test_gpu_path_moves_the_model_and_frees_the_cache(
        self, viscy_native
    ):
        """With CUDA present the model and every batch move to the device."""
        fake_torch, record = viscy_native(cuda_available=True)
        image = np.ones((30, 3, 3), dtype=np.float32)

        out = vvs._run_viscy_native(image, 15)

        assert out.shape == (30, 2, 3, 3)
        assert record["model"].cuda_calls == 1
        assert fake_torch.log["empty_cache"] == 2

    def test_available_viscy_bypasses_the_dedicated_environment(
        self, viscy_native, monkeypatch
    ):
        """_process_single_volume prefers the native path when it can."""
        viscy_native(cuda_available=False)

        def explode(*args, **kwargs):
            raise AssertionError("dedicated environment must not be used")

        monkeypatch.setattr(vvs, "run_viscy_in_env", explode)
        rng = np.random.default_rng(4)
        image = rng.random((15, 4, 4)).astype(np.float32)

        out = vvs._process_single_volume(image, 15, "membrane")

        assert out.shape == (15, 4, 4)

"""
Coverage for the parts of Batch Crop Anything that never touch SAM2.

``_crop_anything`` is one long Qt widget wrapped around a model that needs
a GPU, a checkpoint download and a package that is not installed here.
Almost everything else in it -- the environment bootstrap dialog, image
loading and dimension sniffing, the label table, the click handlers, the
prompt layers and the whole control panel -- is ordinary Python that can
be driven with a stub predictor and a stub ``torch``.  These tests pin
that behaviour.

Every dialog the module can raise (``QMessageBox``) is patched on the
module object, because the names are imported into it; patching
``qtpy.QtWidgets`` would be too late and a modal dialog would hang the
run.
"""

import importlib.util
import os
import sys
import threading
import types

import numpy as np
import pytest
import tifffile

from napari_tmidas import _crop_anything as ca

pytest.importorskip("pytestqt")

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------
class FakeTensor:
    """Minimal stand-in for the torch tensors SAM2 hands back."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def __gt__(self, other):
        return FakeTensor(self.array > other)

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def __len__(self):
        return len(self.array)

    @property
    def ndim(self):
        return self.array.ndim

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class StubOutOfMemoryError(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError`` in ``except`` tuples."""


def make_torch_stub():
    """A ``torch`` module just complete enough for this module's use."""
    stub = types.SimpleNamespace()
    stub.float32 = "float32"
    stub.inference_mode = lambda *args, **kwargs: NullContext()
    stub.autocast = lambda *args, **kwargs: NullContext()
    stub.cuda = types.SimpleNamespace(
        OutOfMemoryError=StubOutOfMemoryError,
        is_available=lambda: False,
        get_device_name=lambda: "Stub GPU",
    )
    stub.device = lambda name: types.SimpleNamespace(type=name)
    stub.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    return stub


class FakePointsLayer:
    """A points layer with only what the click handlers touch."""

    def __init__(self, ndim=2):
        self.data = np.zeros((0, ndim))
        self.face_color = "green"
        self.name = "Points (Click to Add)"
        self.mouse_drag_callbacks = []


class FakeEvent:
    """A napari mouse event as the handlers consume it."""

    def __init__(
        self,
        position,
        button=1,
        modifiers=(),
        event_type="mouse_press",
    ):
        self.position = position
        self.button = button
        self.modifiers = modifiers
        self.type = event_type


class PredictorWithoutSetImage:
    """A video predictor mistakenly used where an image one is needed."""

    def __init__(self, mask=None, score=0.9):
        self.mask = None if mask is None else np.asarray(mask)
        self.score = score
        self.predict_kwargs = []

    def predict(self, **kwargs):
        self.predict_kwargs.append(kwargs)
        return np.array([self.mask]), np.array([self.score]), None


class StubPredictor(PredictorWithoutSetImage):
    """An image predictor whose masks are chosen by the test."""

    def __init__(self, mask, score=0.9):
        super().__init__(mask, score)
        self.set_image_calls = []

    def set_image(self, image):
        self.set_image_calls.append(image)


class StubVideoPredictor:
    """A video predictor: one frame answer plus a propagation stream."""

    def __init__(self, mask, frames):
        self.mask = np.asarray(mask)
        self.frames = frames
        self.add_calls = []
        self.state = object()

    def init_state(self, path):
        self.init_path = path
        return self.state

    def add_new_points_or_box(self, **kwargs):
        self.add_calls.append(kwargs)
        obj_id = kwargs["obj_id"]
        return None, [obj_id], FakeTensor(self.mask[np.newaxis])

    def propagate_in_video(self, state):
        for frame_idx in range(self.frames):
            yield frame_idx, [1], FakeTensor(self.mask[np.newaxis])


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def no_sam2(monkeypatch):
    """Skip model initialisation; these tests never need a real model."""
    monkeypatch.setattr(
        ca.BatchCropAnything, "_initialize_sam2", lambda self: None
    )


@pytest.fixture
def processor(make_napari_viewer, no_sam2):
    proc = ca.BatchCropAnything(make_napari_viewer())
    proc.predictor = None
    # ``_initialize_sam2`` is what normally sets these; several handlers
    # read ``self.device`` before their own guards run.
    proc.device = types.SimpleNamespace(type="cpu")
    return proc


@pytest.fixture
def torch_stub(monkeypatch):
    stub = make_torch_stub()
    monkeypatch.setattr(ca, "torch", stub)
    return stub


@pytest.fixture
def no_threads(monkeypatch):
    """Run the "remove the progress layer later" thread inline-free.

    The real code sleeps two seconds and then mutates the viewer from
    another thread, which outlives the test and can touch a closed viewer.
    """

    class InstantThread:
        def __init__(self, target=None, **kwargs):
            self.target = target

        def start(self):
            return None

    monkeypatch.setattr(threading, "Thread", InstantThread)


def make_scene(processor, tmp_path, name="scene.tif"):
    """A 2D image on disk plus the segmentation the processor holds."""
    rng = np.random.default_rng(0)
    image = rng.integers(1, 250, size=(16, 16), dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.uint32)
    segmentation[0:4, 0:4] = 1
    segmentation[8:12, 8:12] = 2
    path = tmp_path / name
    tifffile.imwrite(path, image)

    processor.images = [str(path)]
    processor.current_index = 0
    processor.original_image = image
    processor.current_image_for_segmentation = image
    processor.segmentation_result = segmentation
    processor.label_info = {
        1: {"area": 16, "score": 1.0},
        2: {"area": 16, "score": 1.0},
    }
    processor.image_layer = processor.viewer.add_image(image, name="scene")
    processor.label_layer = processor.viewer.add_labels(
        segmentation, name=f"Segmentation ({name})"
    )
    return processor


def make_volume(processor, tmp_path, name="vol.tif"):
    """A 3-frame volume plus its segmentation."""
    image = np.full((3, 8, 8), 7, dtype=np.uint8)
    segmentation = np.zeros((3, 8, 8), dtype=np.uint32)
    segmentation[:, 0:2, 0:2] = 1
    segmentation[1, 5:7, 5:7] = 2
    path = tmp_path / name
    tifffile.imwrite(path, image)

    processor.use_3d = True
    processor.images = [str(path)]
    processor.current_index = 0
    processor.original_image = image
    processor.current_image_for_segmentation = image
    processor.segmentation_result = segmentation
    processor.label_info = {
        1: {"area": 12, "score": 1.0},
        2: {"area": 4, "score": 1.0},
    }
    processor.image_layer = processor.viewer.add_image(image)
    processor.label_layer = processor.viewer.add_labels(
        segmentation, name=f"Segmentation ({name})"
    )
    return processor


@pytest.fixture
def scene(processor, tmp_path):
    return make_scene(processor, tmp_path)


@pytest.fixture
def volume(processor, tmp_path):
    return make_volume(processor, tmp_path)


class TestCheckOrCreateSam2Env:
    """The bootstrap dialog: every branch that can end the call."""

    @pytest.fixture(autouse=True)
    def _quiet_dialogs(self, monkeypatch):
        """No modal dialog may ever open: they would hang the run."""
        self.dialogs = []
        monkeypatch.setattr(
            ca.QMessageBox,
            "information",
            lambda *args, **kwargs: self.dialogs.append(("info", args[1])),
        )
        monkeypatch.setattr(
            ca.QMessageBox,
            "critical",
            lambda *args, **kwargs: self.dialogs.append(("critical", args[1])),
        )
        monkeypatch.setattr(
            ca.sam2_manager, "is_env_created", lambda: False
        )
        monkeypatch.setattr(
            ca.sam2_manager, "is_package_installed", lambda: False
        )

    @pytest.fixture(autouse=True)
    def _restore_sys_path(self):
        before = list(sys.path)
        yield
        sys.path[:] = before

    def _answer(self, monkeypatch, answer):
        monkeypatch.setattr(
            ca.QMessageBox,
            "question",
            lambda *args, **kwargs: answer,
        )

    def _install_fake_sam2(self, monkeypatch):
        module = types.ModuleType("sam2")
        module.__spec__ = importlib.util.spec_from_loader("sam2", loader=None)
        monkeypatch.setitem(sys.modules, "sam2", module)

    def test_existing_sam2_path_is_reused(self, monkeypatch, tmp_path):
        self._install_fake_sam2(monkeypatch)
        monkeypatch.setenv("SAM2_PATH", str(tmp_path))

        assert ca.check_or_create_sam2_env() is True
        assert self.dialogs == [("info", "SAM2 Found")]
        assert str(tmp_path) in sys.path

    def test_unimportable_sam2_path_falls_through(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setenv("SAM2_PATH", str(tmp_path))
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: (_ for _ in ()).throw(ImportError("no sam2")),
        )
        self._answer(monkeypatch, ca.QMessageBox.No)

        assert ca.check_or_create_sam2_env() is False
        # The "SAM2 Found" dialog must not fire, and the warning has to
        # name the path that turned out to be unusable -- otherwise this
        # would also pass on the plain "no SAM2_PATH" route.
        assert self.dialogs == []
        assert f"SAM2_PATH is set to {tmp_path}" in capsys.readouterr().out

    def test_missing_sam2_path_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAM2_PATH", str(tmp_path / "gone"))
        self._answer(monkeypatch, ca.QMessageBox.No)

        assert ca.check_or_create_sam2_env() is False
        assert self.dialogs == []

    def test_declining_the_dialog_creates_nothing(self, monkeypatch):
        created = []
        monkeypatch.setattr(
            ca.sam2_manager, "create_env", lambda: created.append(True)
        )
        self._answer(monkeypatch, ca.QMessageBox.No)

        assert ca.check_or_create_sam2_env() is False
        assert created == []

    def test_accepting_the_dialog_creates_the_env(self, monkeypatch):
        created = []
        monkeypatch.setattr(
            ca.sam2_manager, "create_env", lambda: created.append(True)
        )
        self._answer(monkeypatch, ca.QMessageBox.Yes)

        assert ca.check_or_create_sam2_env() is True
        assert created == [True]
        assert self.dialogs == [("info", "Installation Complete")]

    def test_failed_creation_is_reported(self, monkeypatch):
        def boom():
            raise RuntimeError("no disk space")

        monkeypatch.setattr(ca.sam2_manager, "create_env", boom)
        self._answer(monkeypatch, ca.QMessageBox.Yes)

        assert ca.check_or_create_sam2_env() is False
        assert self.dialogs == [("critical", "Installation Failed")]

    def test_without_qt_no_dialog_is_shown(self, monkeypatch):
        """Headless callers get the console path, not a modal dialog."""
        created = []
        monkeypatch.setattr(ca, "_HAS_QTPY", False)
        monkeypatch.setattr(
            ca.sam2_manager, "create_env", lambda: created.append(True)
        )

        assert ca.check_or_create_sam2_env() is True
        assert created == [True]
        assert self.dialogs == []

    def test_without_qt_a_failure_still_returns_false(self, monkeypatch):
        def boom():
            raise OSError("permission denied")

        monkeypatch.setattr(ca, "_HAS_QTPY", False)
        monkeypatch.setattr(ca.sam2_manager, "create_env", boom)

        assert ca.check_or_create_sam2_env() is False
        assert self.dialogs == []


class TestGetDevice:
    """Accelerator selection, driven entirely by the torch stub."""

    def test_cpu_when_nothing_is_available(self, monkeypatch, torch_stub):
        monkeypatch.setattr(ca.sys, "platform", "linux")

        assert ca.get_device().type == "cpu"

    def test_cuda_is_preferred_on_linux(self, monkeypatch, torch_stub):
        monkeypatch.setattr(ca.sys, "platform", "linux")
        torch_stub.cuda.is_available = lambda: True

        assert ca.get_device().type == "cuda"

    def test_mps_is_used_on_macos(self, monkeypatch, torch_stub):
        monkeypatch.setattr(ca.sys, "platform", "darwin")
        torch_stub.backends.mps.is_available = lambda: True
        # CUDA is never consulted on macOS.
        torch_stub.cuda.is_available = lambda: True

        assert ca.get_device().type == "mps"

    def test_macos_without_mps_falls_back_to_cpu(
        self, monkeypatch, torch_stub
    ):
        monkeypatch.setattr(ca.sys, "platform", "darwin")

        assert ca.get_device().type == "cpu"


class TestInitializeSam2:
    """Model construction, with the checkpoint download stubbed out."""

    CHECKPOINT = "sam2.1_hiera_large.pt"
    CHECKPOINT_DIR = "/opt/sam2/checkpoints/"
    CHECKPOINT_PATH = CHECKPOINT_DIR + CHECKPOINT
    CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    @pytest.fixture
    def viewer(self):
        return types.SimpleNamespace(status="")

    @pytest.fixture(autouse=True)
    def _no_disk_writes(self, monkeypatch, torch_stub):
        """The module downloads into a hard-coded /opt path; block it."""
        monkeypatch.setattr(ca.os, "makedirs", lambda *a, **k: None)

    def _checkpoint_present(self, monkeypatch, present=True):
        real_exists = ca.os.path.exists

        def fake_exists(path):
            if str(path).endswith(self.CHECKPOINT):
                return present
            return real_exists(path)

        monkeypatch.setattr(ca.os.path, "exists", fake_exists)

    def _redirect_checkpoint_writes(self, monkeypatch, tmp_path):
        """Send the hard-coded /opt/sam2 write into *tmp_path*.

        ``_initialize_sam2`` writes the checkpoint to an absolute path
        outside the repository.  Nothing may land there, and whether it
        happens to exist must not decide the outcome of the test.
        """
        import builtins

        real_open = builtins.open
        landing = tmp_path / self.CHECKPOINT

        def guarded_open(file, *args, **kwargs):
            if str(file).startswith(self.CHECKPOINT_DIR):
                assert str(file) == self.CHECKPOINT_PATH
                return real_open(landing, *args, **kwargs)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", guarded_open)
        return landing

    def _install_sam2(self, monkeypatch, built):
        build_sam = types.ModuleType("sam2.build_sam")
        build_sam.build_sam2 = lambda cfg, ckpt: built.append(
            ("build_sam2", cfg, ckpt)
        ) or "raw-model"
        build_sam.build_sam2_video_predictor = (
            lambda cfg, ckpt, device=None: built.append(
                ("video", cfg, ckpt, device)
            )
            or "video-predictor"
        )
        image_pred = types.ModuleType("sam2.sam2_image_predictor")
        image_pred.SAM2ImagePredictor = lambda model: (
            "image-predictor",
            model,
        )
        package = types.ModuleType("sam2")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "sam2", package)
        monkeypatch.setitem(sys.modules, "sam2.build_sam", build_sam)
        monkeypatch.setitem(
            sys.modules, "sam2.sam2_image_predictor", image_pred
        )

    def test_builds_the_image_predictor_in_2d(self, monkeypatch, viewer):
        built = []
        self._checkpoint_present(monkeypatch)
        self._install_sam2(monkeypatch, built)

        proc = ca.BatchCropAnything(viewer, use_3d=False)

        assert proc.predictor == ("image-predictor", "raw-model")
        assert built == [
            (
                "build_sam2",
                self.CONFIG,
                self.CHECKPOINT_PATH,
            )
        ]
        assert "Image Predictor" in viewer.status
        # No video predictor may be built in 2D mode.
        assert proc.use_3d is False

    def test_builds_the_video_predictor_in_3d(self, monkeypatch, viewer):
        built = []
        self._checkpoint_present(monkeypatch)
        self._install_sam2(monkeypatch, built)

        proc = ca.BatchCropAnything(viewer, use_3d=True)

        assert proc.predictor == "video-predictor"
        assert built == [
            (
                "video",
                self.CONFIG,
                self.CHECKPOINT_PATH,
                proc.device,
            )
        ]
        assert "Video Predictor" in viewer.status

    def test_a_missing_checkpoint_is_downloaded(
        self, monkeypatch, viewer, tmp_path
    ):
        """The weights are streamed to the checkpoint path, then loaded."""
        requested = []
        built = []
        self._checkpoint_present(monkeypatch, present=False)
        self._install_sam2(monkeypatch, built)

        class Response:
            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=None):
                assert chunk_size == 8192
                yield b"wei"
                yield b"ghts"

        monkeypatch.setattr(
            ca.requests,
            "get",
            lambda url, **kwargs: requested.append((url, kwargs))
            or Response(),
        )
        landing = self._redirect_checkpoint_writes(monkeypatch, tmp_path)

        proc = ca.BatchCropAnything(viewer, use_3d=False)

        assert len(requested) == 1
        url, kwargs = requested[0]
        assert url.endswith(self.CHECKPOINT)
        assert kwargs == {"stream": True, "timeout": 30}
        # Every streamed chunk is written, in order.
        assert landing.read_bytes() == b"weights"
        # ...and the file just written is the one handed to the builder.
        assert built[0][2] == self.CHECKPOINT_PATH
        assert proc.predictor == ("image-predictor", "raw-model")

    def test_an_existing_checkpoint_is_not_re_downloaded(
        self, monkeypatch, viewer, tmp_path
    ):
        built = []
        self._checkpoint_present(monkeypatch, present=True)
        self._install_sam2(monkeypatch, built)
        monkeypatch.setattr(
            ca.requests,
            "get",
            lambda *a, **k: pytest.fail("checkpoint present; must not fetch"),
        )
        self._redirect_checkpoint_writes(monkeypatch, tmp_path)

        proc = ca.BatchCropAnything(viewer, use_3d=False)

        assert proc.predictor == ("image-predictor", "raw-model")
        assert built[0][2] == self.CHECKPOINT_PATH

    def test_a_download_error_leaves_no_predictor(self, monkeypatch, viewer):
        self._checkpoint_present(monkeypatch, present=False)

        def boom(url, **kwargs):
            raise ca.requests.RequestException("offline")

        monkeypatch.setattr(ca.requests, "get", boom)

        proc = ca.BatchCropAnything(viewer, use_3d=False)

        assert proc.predictor is None
        assert viewer.status == (
            "SAM2 initialization failed: offline"
            " - Images will load without segmentation"
        )

    def test_a_missing_sam2_package_leaves_no_predictor(
        self, monkeypatch, viewer
    ):
        self._checkpoint_present(monkeypatch)
        monkeypatch.setattr(
            ca.requests,
            "get",
            lambda *a, **k: pytest.fail("checkpoint present; must not fetch"),
        )
        monkeypatch.setitem(sys.modules, "sam2", None)

        proc = ca.BatchCropAnything(viewer, use_3d=False)

        assert proc.predictor is None
        # The failure has to be the sam2 import, not some unrelated error.
        assert viewer.status.startswith("SAM2 initialization failed")
        assert "sam2" in viewer.status
        assert "Images will load without segmentation" in viewer.status

    def test_a_device_failure_leaves_no_predictor(self, monkeypatch, viewer):
        def boom():
            raise RuntimeError("CUDA driver missing")

        monkeypatch.setattr(ca, "get_device", boom)

        proc = ca.BatchCropAnything(viewer, use_3d=False)

        assert proc.predictor is None
        assert "CUDA driver missing" in viewer.status


class TestLoadCurrentImage:
    """Reading a file and deciding what its axes mean."""

    def _write(self, tmp_path, array, name="img.tif"):
        path = tmp_path / name
        # Without photometric a leading axis of 3 or 4 round-trips as RGB.
        kwargs = {"photometric": "minisblack"} if array.ndim > 2 else {}
        tifffile.imwrite(path, array, **kwargs)
        return str(path)

    def _labels(self, proc):
        return [
            layer
            for layer in proc.viewer.layers
            if "Segmentation" in layer.name
        ]

    def test_without_images_nothing_is_loaded(self, processor):
        processor.images = []
        processor._load_current_image()
        assert processor.image_layer is None
        assert "no images to process" in processor.viewer.status.lower()

    def test_a_2d_image_gets_an_empty_label_layer(self, processor, tmp_path):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        processor.images = [self._write(tmp_path, image)]

        processor._load_current_image()

        np.testing.assert_array_equal(processor.original_image, image)
        assert processor.segmentation_result.shape == (8, 8)
        assert processor.segmentation_result.dtype == np.uint32
        assert not processor.segmentation_result.any()
        assert "No Segmentation" in self._labels(processor)[0].name

    def test_a_non_uint8_image_is_rescaled_for_display(
        self, processor, tmp_path
    ):
        image = np.zeros((8, 8), dtype=np.uint16)
        image[0, 0] = 1000
        image[1, 1] = 500
        processor.images = [self._write(tmp_path, image)]

        processor._load_current_image()

        shown = processor.image_layer.data
        assert shown.dtype == np.uint8
        assert shown[0, 0] == 255
        assert shown[1, 1] == 127
        # The untouched original is what later crops are cut from.
        assert processor.original_image.dtype == np.uint16

    def test_a_tyx_stack_is_kept_as_time(self, processor, tmp_path):
        volume = np.zeros((10, 8, 8), dtype=np.uint8)
        processor.use_3d = True
        processor.images = [self._write(tmp_path, volume)]

        processor._load_current_image()

        assert processor.time_dim_size == 10
        assert processor.has_z_dim is False
        assert processor.segmentation_result.shape == (10, 8, 8)

    def test_a_short_stack_is_read_as_zyx(self, processor, tmp_path):
        volume = np.zeros((2, 8, 8), dtype=np.uint8)
        processor.use_3d = True
        processor.images = [self._write(tmp_path, volume)]

        processor._load_current_image()

        assert processor.time_dim_size == 1
        assert processor.has_z_dim is True

    def test_a_tzyx_volume_keeps_its_order(self, processor, tmp_path):
        volume = np.zeros((5, 2, 8, 8), dtype=np.uint8)
        processor.use_3d = True
        processor.images = [self._write(tmp_path, volume)]

        processor._load_current_image()

        assert processor.original_image.shape == (5, 2, 8, 8)
        assert processor.time_dim_size == 5
        assert processor.has_z_dim is True

    def test_a_misordered_volume_is_transposed(self, processor, tmp_path):
        """Time is sniffed as the first axis with 4 < size < 400."""
        volume = np.zeros((2, 10, 8, 8), dtype=np.uint8)
        processor.use_3d = True
        processor.images = [self._write(tmp_path, volume)]

        processor._load_current_image()

        assert processor.original_image.shape == (10, 2, 8, 8)
        assert processor.time_dim_size == 10

    def test_a_volume_without_a_time_axis_is_single_frame(
        self, processor, tmp_path
    ):
        volume = np.zeros((2, 3, 4, 4), dtype=np.uint8)
        processor.use_3d = True
        processor.images = [self._write(tmp_path, volume)]

        processor._load_current_image()

        assert processor.original_image.shape == (2, 3, 4, 4)
        assert processor.time_dim_size == 1
        assert processor.has_z_dim is True

    def test_a_five_dimensional_volume_hits_the_fallback(
        self, processor, tmp_path
    ):
        volume = np.zeros((2, 2, 3, 4, 4), dtype=np.uint8)
        processor.use_3d = True
        processor.images = [self._write(tmp_path, volume)]

        processor._load_current_image()

        assert processor.time_dim_size == 1
        assert processor.has_z_dim is False

    def test_an_unreadable_file_leaves_an_error_layer(
        self, processor, tmp_path
    ):
        processor.images = [str(tmp_path / "missing.tif")]
        processor.original_image = np.zeros((4, 4), dtype=np.uint8)

        processor._load_current_image()

        assert "Error processing image" in processor.viewer.status
        assert processor.label_layer.name == "Error: No Segmentation"

    def test_a_predictor_gets_asked_to_segment(self, processor, tmp_path):
        seen = []
        processor.predictor = object()
        processor.images = [
            self._write(tmp_path, np.zeros((8, 8), dtype=np.uint8))
        ]
        processor._generate_segmentation = (
            lambda image, path: seen.append((image.shape, path))
        )

        processor._load_current_image()

        assert seen == [((8, 8), processor.images[0])]


class TestLoadCurrentImageFromZarr:
    """The Zarr branch has to dig the image out of a layer-data list."""

    @pytest.fixture
    def zarr_path(self, processor, tmp_path):
        path = tmp_path / "store.zarr"
        path.mkdir()
        processor.images = [str(path)]
        return str(path)

    def _load(self, processor, monkeypatch, payload):
        monkeypatch.setattr(ca, "load_any_image", lambda path: payload)
        processor._load_current_image()

    def test_a_full_layer_data_tuple(self, processor, zarr_path, monkeypatch):
        image = np.ones((6, 6), dtype=np.uint8)
        self._load(
            processor,
            monkeypatch,
            [(np.zeros((6, 6)), {}, "labels"), (image, {}, "image")],
        )
        np.testing.assert_array_equal(processor.original_image, image)

    def test_a_two_element_tuple(self, processor, zarr_path, monkeypatch):
        image = np.full((6, 6), 3, dtype=np.uint8)
        self._load(processor, monkeypatch, [(image, {})])
        np.testing.assert_array_equal(processor.original_image, image)

    def test_a_bare_array_in_a_list(self, processor, zarr_path, monkeypatch):
        image = np.full((6, 6), 4, dtype=np.uint8)
        self._load(processor, monkeypatch, [image])
        np.testing.assert_array_equal(processor.original_image, image)

    def test_a_lazy_array_is_computed(self, processor, zarr_path, monkeypatch):
        image = np.full((6, 6), 5, dtype=np.uint8)

        class Lazy:
            def compute(self):
                return image

        self._load(processor, monkeypatch, Lazy())
        np.testing.assert_array_equal(processor.original_image, image)

    def test_a_store_without_an_image_is_reported(
        self, processor, zarr_path, monkeypatch
    ):
        self._load(processor, monkeypatch, [(np.zeros((4, 4)), {}, "labels")])
        assert "No image layer found" in processor.viewer.status


def checkbox_at(table, row):
    """The checkbox the table nests inside a container widget."""
    holder = table.cellWidget(row, 0)
    return None if holder is None else holder.findChild(ca.QCheckBox)


class TestLabelTable:
    """Building, filling and re-syncing the label selection table."""

    def test_the_table_lists_every_label(self, scene):
        table = scene.create_label_table(None)

        assert table.columnCount() == 2
        assert table.rowCount() == 2
        assert [table.item(row, 1).text() for row in range(2)] == ["1", "2"]
        assert scene.label_table_widget is table

    def test_checkboxes_start_from_the_selection(self, scene):
        scene.selected_labels = {2}
        table = scene.create_label_table(None)

        assert checkbox_at(table, 0).isChecked() is False
        assert checkbox_at(table, 1).isChecked() is True

    def test_ticking_a_checkbox_selects_that_label(self, scene):
        table = scene.create_label_table(None)

        checkbox_at(table, 0).setChecked(True)
        assert 1 in scene.selected_labels

        checkbox_at(table, 0).setChecked(False)
        assert 1 not in scene.selected_labels

    def test_clicking_the_table_reactivates_the_segmentation(self, scene):
        table = scene.create_label_table(None)
        scene.viewer.layers.selection.active = scene.image_layer

        table.clicked.emit(table.model().index(0, 1))

        assert scene.viewer.layers.selection.active is scene.label_layer

    def test_missing_label_info_is_measured(self, scene):
        scene.label_info = {}
        table = scene.create_label_table(None)

        assert scene.label_info[1]["area"] == 16
        assert scene.label_info[2]["area"] == 16
        assert scene.label_info[1]["score"] == 1.0
        assert table.rowCount() == 2

    def test_float_segmentations_are_counted_without_bincount(self, scene):
        """np.bincount only takes integers, so floats take the other path."""
        scene.segmentation_result = scene.segmentation_result.astype(
            np.float32
        )
        scene.label_info = {}
        table = ca.QTableWidget()

        scene._populate_label_table(table)

        assert table.rowCount() == 2
        assert scene.label_info[np.float32(1)]["area"] == 16

    def test_without_a_segmentation_the_table_empties(self, scene):
        scene.segmentation_result = None
        table = ca.QTableWidget()
        table.setRowCount(3)

        scene._populate_label_table(table)

        assert table.rowCount() == 0
        assert "no segmentation available" in scene.viewer.status.lower()

    def test_an_empty_segmentation_reports_no_objects(self, scene):
        scene.segmentation_result[:] = 0
        table = ca.QTableWidget()

        scene._populate_label_table(table)

        assert table.rowCount() == 0
        assert "no labeled objects" in scene.viewer.status.lower()

    def test_a_broken_label_info_is_reported_not_raised(self, scene):
        scene.label_info = None  # membership test raises TypeError
        table = ca.QTableWidget()
        table.setRowCount(5)

        scene._populate_label_table(table)

        assert table.rowCount() == 0
        assert "error populating table" in scene.viewer.status.lower()

    def test_update_without_a_table_is_a_no_op(self, scene):
        scene.label_table_widget = None
        before = scene.viewer.status

        scene._update_label_table()

        # The guard must return before touching anything: no table is
        # built on the fly and no status message is written.
        assert scene.label_table_widget is None
        assert scene.viewer.status == before

    def test_update_resyncs_the_checkboxes(self, scene):
        table = scene.create_label_table(None)
        scene.selected_labels = {1, 2}

        scene._update_label_table()

        assert checkbox_at(table, 0).isChecked() is True
        assert checkbox_at(table, 1).isChecked() is True

    def test_update_follows_a_changed_segmentation(self, scene):
        scene.create_label_table(None)
        scene.segmentation_result[scene.segmentation_result == 2] = 0

        scene._update_label_table()

        assert scene.label_table_widget.rowCount() == 1

    def test_activating_without_a_label_layer_is_a_no_op(self, processor):
        import numpy as np

        other = processor.viewer.add_image(np.zeros((4, 4), dtype=np.uint8))
        processor.viewer.layers.selection.active = other
        processor.label_layer = None

        processor._ensure_segmentation_layer_active()

        # With nothing to activate the current selection must be left
        # alone rather than cleared.
        assert processor.viewer.layers.selection.active is other


class TestOnLabelClicked2d:
    """Selection, deletion and the guards in front of them (2D)."""

    def test_non_press_events_are_ignored(self, scene):
        scene._on_label_clicked(
            scene.label_layer, FakeEvent((1, 1), event_type="mouse_move")
        )
        assert scene.selected_labels == set()

    def test_other_mouse_buttons_are_ignored(self, scene):
        scene._on_label_clicked(
            scene.label_layer, FakeEvent((1, 1), button=2)
        )
        assert scene.selected_labels == set()

    def test_wrong_coordinate_count_is_reported(self, scene):
        scene._on_label_clicked(scene.label_layer, FakeEvent((0, 1, 1)))
        assert "unexpected coordinate" in scene.viewer.status.lower()

    def test_a_click_outside_the_image_is_reported(self, scene):
        scene._on_label_clicked(scene.label_layer, FakeEvent((99, 1)))
        assert "outside image bounds" in scene.viewer.status.lower()
        assert scene.selected_labels == set()

    def test_clicking_a_label_toggles_it(self, scene):
        scene._on_label_clicked(scene.label_layer, FakeEvent((1, 1)))
        assert scene.selected_labels == {1}

        scene._on_label_clicked(scene.label_layer, FakeEvent((1, 1)))
        assert scene.selected_labels == set()

    def test_selecting_draws_a_preview(self, scene):
        scene._on_label_clicked(scene.label_layer, FakeEvent((9, 9)))

        preview = next(
            layer for layer in scene.viewer.layers if "Preview" in layer.name
        )
        assert preview.name == "Preview (Labels: 2)"
        # Everything outside label 2 is blanked, label 2 keeps its pixels.
        expected = np.where(
            scene.segmentation_result == 2, scene.original_image, 0
        )
        np.testing.assert_array_equal(preview.data, expected)
        assert int((preview.data != 0).sum()) == 16
        # The preview must be a copy: the source image is untouched.
        assert scene.original_image.all()

    def test_ctrl_click_deletes_the_label(self, scene):
        scene._on_label_clicked(
            scene.label_layer, FakeEvent((1, 1), modifiers=("Control",))
        )
        assert not (scene.segmentation_result == 1).any()
        assert "deleted label id: 1" in scene.viewer.status.lower()

    def test_background_clicks_change_nothing(self, scene):
        scene._on_label_clicked(scene.label_layer, FakeEvent((6, 6)))
        assert scene.selected_labels == set()

    def test_shift_click_is_left_to_the_points_layer(self, scene):
        scene._on_label_clicked(
            scene.label_layer, FakeEvent((1, 1), modifiers=("Shift",))
        )
        assert scene.selected_labels == set()

    def test_a_broken_position_is_reported_not_raised(self, scene):
        scene._on_label_clicked(scene.label_layer, FakeEvent(("a", "b")))
        assert "error in click handling" in scene.viewer.status.lower()


class TestOnLabelClicked3d:
    """The same handler over a volume."""

    def test_clicking_a_label_selects_it(self, volume):
        volume._on_label_clicked(volume.label_layer, FakeEvent((1, 5, 5)))
        assert volume.selected_labels == {2}

    def test_two_coordinates_use_the_current_slice(self, volume):
        volume.viewer.dims.set_current_step(0, 1)
        volume._on_label_clicked(volume.label_layer, FakeEvent((5, 5)))
        assert volume.selected_labels == {2}

    def test_wrong_coordinate_count_is_reported(self, volume):
        volume._on_label_clicked(volume.label_layer, FakeEvent((0, 0, 1, 1)))
        assert "unexpected coordinate" in volume.viewer.status.lower()

    def test_a_click_outside_the_volume_is_reported(self, volume):
        volume._on_label_clicked(volume.label_layer, FakeEvent((9, 1, 1)))
        assert "outside volume bounds" in volume.viewer.status.lower()

    def test_ctrl_click_deletes_across_frames(self, volume):
        volume._on_label_clicked(
            volume.label_layer, FakeEvent((0, 0, 0), modifiers=("Control",))
        )
        assert not (volume.segmentation_result == 1).any()

    def test_background_clicks_change_nothing(self, volume):
        volume._on_label_clicked(volume.label_layer, FakeEvent((0, 4, 4)))
        assert volume.selected_labels == set()

    def test_deselecting_removes_the_label_again(self, volume):
        volume._on_label_clicked(volume.label_layer, FakeEvent((0, 0, 0)))
        volume._on_label_clicked(volume.label_layer, FakeEvent((0, 0, 0)))
        assert volume.selected_labels == set()


class TestPromptLayers:
    """Which interaction layer exists depends on the prompt mode."""

    def _names(self, proc):
        return [layer.name for layer in proc.viewer.layers]

    def test_point_mode_creates_a_points_layer(self, scene):
        scene._update_label_layer()

        assert "Points (Click to Add)" in self._names(scene)
        assert not any("Rectangles" in name for name in self._names(scene))

    def test_the_points_layer_is_not_duplicated(self, scene):
        scene._update_label_layer()
        scene._update_label_layer()

        assert self._names(scene).count("Points (Click to Add)") == 1

    def test_the_points_layer_is_made_active(self, scene):
        scene._update_label_layer()
        assert "Points" in scene.viewer.layers.selection.active.name

    def test_box_mode_swaps_in_a_shapes_layer(self, scene):
        scene._update_label_layer()
        scene.prompt_mode = "box"
        scene._update_label_layer()

        assert "Rectangles (Draw to Segment)" in self._names(scene)
        assert not any("Points" in name for name in self._names(scene))
        assert scene.shapes_layer is not None

    def test_going_back_to_points_drops_the_shapes_layer(self, scene):
        scene.prompt_mode = "box"
        scene._update_label_layer()
        scene.prompt_mode = "point"
        scene._update_label_layer()

        assert not any("Rectangles" in name for name in self._names(scene))
        assert scene.shapes_layer is None

    def test_the_status_counts_the_segments(self, scene):
        scene._update_label_layer()
        assert "Found 2 segments" in scene.viewer.status

    def test_drawing_a_rectangle_runs_the_handler(self, scene):
        scene.prompt_mode = "box"
        scene._update_label_layer()

        scene.shapes_layer.data = [
            np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]])
        ]

        # No predictor is loaded, so the handler bails out with a message;
        # what matters is that the shape event reached it at all.
        assert "predictor not initialized" in scene.viewer.status.lower()

    SQUARE = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]])

    def _spy(self, proc, monkeypatch):
        """Record calls into the segmentation handler itself.

        Asserting "the status line did not change" would also pass if the
        event never reached ``on_shape_added`` at all; the spy pins which
        of the two happened.
        """
        seen = []
        monkeypatch.setattr(
            proc, "_on_rectangle_added", lambda coords: seen.append(coords)
        )
        return seen

    def test_a_lone_rectangle_reaches_the_handler(self, scene, monkeypatch):
        """Positive control for the three guards below."""
        scene.prompt_mode = "box"
        scene._update_label_layer()
        seen = self._spy(scene, monkeypatch)

        scene.shapes_layer.data = [self.SQUARE]

        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0], self.SQUARE)
        # The re-entry flag is released again, whatever happened inside.
        assert scene._processing_rectangle is False

    def test_a_second_rectangle_is_skipped(self, scene, monkeypatch):
        """Only a lone, freshly drawn rectangle is segmented."""
        scene.prompt_mode = "box"
        scene._update_label_layer()
        seen = self._spy(scene, monkeypatch)

        scene.shapes_layer.data = [self.SQUARE, self.SQUARE + 5]

        assert seen == []

    def test_an_empty_shapes_layer_is_skipped(self, scene, monkeypatch):
        scene.prompt_mode = "box"
        scene._update_label_layer()
        seen = self._spy(scene, monkeypatch)

        scene.shapes_layer.data = []

        assert seen == []

    def test_deleting_the_rectangle_re_runs_the_handler(
        self, scene, monkeypatch
    ):
        """KNOWN DEFECT, pinned so a fix is noticed.

        napari emits ``events.data`` twice per edit.  On a deletion the
        first emission carries ``action="removing"`` and the shape is
        still in ``layer.data``, so ``on_shape_added`` -- which never
        looks at ``event.action`` -- sees exactly one rectangle and
        segments it again, burning another object id.  Deleting a
        rectangle should segment nothing; if this test starts failing
        with ``len(seen) == 1``, the source was fixed and the assertion
        below should be tightened to that.
        """
        scene.prompt_mode = "box"
        scene._update_label_layer()
        seen = self._spy(scene, monkeypatch)
        scene.shapes_layer.data = [self.SQUARE]
        assert len(seen) == 1

        scene.shapes_layer.data = []

        assert len(seen) == 2
        np.testing.assert_array_equal(seen[1], self.SQUARE)

    def test_a_reentrant_event_is_skipped(self, scene, monkeypatch):
        scene.prompt_mode = "box"
        scene._update_label_layer()
        seen = self._spy(scene, monkeypatch)
        scene._processing_rectangle = True

        scene.shapes_layer.data = [self.SQUARE]

        assert seen == []
        # The guard must not clear the flag it did not set.
        assert scene._processing_rectangle is True


def find_button(widget, text):
    for button in widget.findChildren(ca.QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"no button labelled {text!r}")


def build_panel(proc):
    widget = ca.create_crop_widget(proc)
    status = next(
        label
        for label in widget.findChildren(ca.QLabel)
        if label.text().startswith("Ready to process")
    )
    return types.SimpleNamespace(widget=widget, status=status)


class TestCreateCropWidget:
    """The control panel: its buttons and what each one does."""

    @pytest.fixture
    def panel(self, scene):
        return build_panel(scene)

    def test_every_control_is_present(self, panel):
        for text in [
            "Points",
            "Rectangle",
            "Make Prompt Layer Active",
            "Clear Prompts",
            "Select All",
            "Clear All Labels",
            "Crop with Selected Objects",
            "Previous Image",
            "Next Image",
        ]:
            assert find_button(panel.widget, text) is not None

    def test_the_table_is_embedded(self, panel, scene):
        assert scene.label_table_widget is not None
        assert scene.label_table_widget.rowCount() == 2

    def test_the_2d_instructions_are_shown(self, panel):
        text = panel.widget.findChildren(ca.QLabel)[0].text()
        assert "2D (YX)" in text
        assert "FIRST SLICE" not in text

    def test_the_3d_instructions_warn_about_the_first_slice(self, volume):
        panel = build_panel(volume)
        text = panel.widget.findChildren(ca.QLabel)[0].text()
        assert "3D (TYX/ZYX)" in text
        assert "FIRST SLICE" in text

    def test_switching_to_rectangle_mode(self, panel, scene):
        find_button(panel.widget, "Rectangle").click()

        assert scene.prompt_mode == "box"
        assert find_button(panel.widget, "Rectangle").isChecked() is True
        assert find_button(panel.widget, "Points").isChecked() is False
        assert "Rectangle mode active" in panel.status.text()

    def test_switching_back_to_point_mode(self, panel, scene):
        find_button(panel.widget, "Rectangle").click()
        find_button(panel.widget, "Points").click()

        assert scene.prompt_mode == "point"
        assert find_button(panel.widget, "Points").isChecked() is True
        assert "Point mode active" in panel.status.text()

    def test_activating_the_points_layer(self, panel, scene):
        scene._update_label_layer()
        scene.viewer.layers.selection.active = scene.image_layer

        find_button(panel.widget, "Make Prompt Layer Active").click()

        assert "Points" in scene.viewer.layers.selection.active.name
        assert "Points layer is now active" in panel.status.text()

    def test_activating_without_a_points_layer(self, panel, scene):
        scene._remove_points_layer()

        find_button(panel.widget, "Make Prompt Layer Active").click()

        assert "No points layer found" in panel.status.text()

    def test_activating_the_points_layer_in_3d(self, volume):
        panel = build_panel(volume)
        volume._update_label_layer()

        find_button(panel.widget, "Make Prompt Layer Active").click()

        assert "FIRST SLICE" in panel.status.text()

    def test_activating_the_rectangles_layer(self, panel, scene):
        find_button(panel.widget, "Rectangle").click()

        find_button(panel.widget, "Make Prompt Layer Active").click()

        assert "Rectangles" in scene.viewer.layers.selection.active.name
        assert "Rectangles layer is now active" in panel.status.text()

    def test_activating_without_a_rectangles_layer(self, panel, scene):
        scene.prompt_mode = "box"
        scene._remove_shapes_layer()

        find_button(panel.widget, "Make Prompt Layer Active").click()

        assert "No rectangles layer found" in panel.status.text()

    def test_select_all_button(self, panel, scene):
        find_button(panel.widget, "Select All").click()

        assert scene.selected_labels == {1, 2}
        assert "Selected all 2 objects" in panel.status.text()

    def test_clear_all_labels_button(self, panel, scene):
        scene.selected_labels = {1}

        find_button(panel.widget, "Clear All Labels").click()

        assert not scene.segmentation_result.any()
        assert scene.selected_labels == set()
        assert panel.status.text() == "Selection cleared"

    def test_crop_button_writes_the_files(self, panel, scene, tmp_path):
        scene.selected_labels = {1, 2}

        find_button(panel.widget, "Crop with Selected Objects").click()

        cropped = tifffile.imread(tmp_path / "scene_sam2_cropped.tif")
        labels = tifffile.imread(tmp_path / "scene_sam2_labels.tif")
        # Both selected objects survive, everything else is zeroed.
        np.testing.assert_array_equal(
            cropped,
            np.where(scene.segmentation_result > 0, scene.original_image, 0),
        )
        np.testing.assert_array_equal(labels, scene.segmentation_result)
        assert labels.dtype == np.uint32
        assert cropped.dtype == scene.original_image.dtype
        assert int((cropped != 0).sum()) == 32
        assert "IDs: 1, 2" in panel.status.text()

    def test_crop_keeps_only_the_selected_object(
        self, panel, scene, tmp_path
    ):
        scene.selected_labels = {2}

        find_button(panel.widget, "Crop with Selected Objects").click()

        cropped = tifffile.imread(tmp_path / "scene_sam2_cropped.tif")
        labels = tifffile.imread(tmp_path / "scene_sam2_labels.tif")
        assert int((cropped != 0).sum()) == 16
        assert not cropped[0:4, 0:4].any()
        np.testing.assert_array_equal(
            cropped[8:12, 8:12], scene.original_image[8:12, 8:12]
        )
        # Label 1 is dropped from the mask image too, not just the crop.
        assert set(np.unique(labels).tolist()) == {0, 2}
        assert "IDs: 2" in panel.status.text()

    def test_crop_button_without_a_selection_says_nothing(
        self, panel, scene, tmp_path
    ):
        scene.selected_labels = set()

        find_button(panel.widget, "Crop with Selected Objects").click()

        assert not (tmp_path / "scene_sam2_cropped.tif").exists()
        assert panel.status.text().startswith("Ready to process")

    def test_clear_prompts_empties_the_points_layer(self, panel, scene):
        scene._update_label_layer()
        points = next(
            layer
            for layer in scene.viewer.layers
            if layer.name == "Points (Click to Add)"
        )
        points.data = np.array([[1.0, 1.0], [2.0, 2.0]])

        find_button(panel.widget, "Clear Prompts").click()

        assert len(points.data) == 0
        assert "Cleared all prompts" in panel.status.text()

    def test_clear_prompts_drops_per_object_layers(self, panel, scene):
        scene.viewer.add_points(
            np.array([[1.0, 1.0]]), name="Points for Object 1"
        )

        find_button(panel.widget, "Clear Prompts").click()

        assert not any(
            "Points for Object" in layer.name for layer in scene.viewer.layers
        )

    def test_clear_prompts_resets_2d_tracking(self, panel, scene):
        scene.current_points = [[1, 1]]
        scene.current_labels = [1]
        scene.obj_points = {1: [[1, 1]]}
        scene.obj_labels = {1: [1]}
        scene.obj_boxes = {1: np.zeros(4)}
        scene.current_obj_id = 1

        find_button(panel.widget, "Clear Prompts").click()

        assert scene.current_points == []
        assert scene.obj_points == {}
        assert scene.obj_boxes == {}
        # Numbering resumes above the highest label already on screen.
        assert scene.current_obj_id == 3
        assert scene.next_obj_id == 3

    def test_clear_prompts_without_a_segmentation_restarts_numbering(
        self, panel, scene
    ):
        scene.current_obj_id = 7
        scene.segmentation_result = None

        find_button(panel.widget, "Clear Prompts").click()

        assert scene.current_obj_id == 1
        assert scene.next_obj_id == 1

    def test_clear_prompts_resets_3d_tracking(self, volume):
        panel = build_panel(volume)
        volume.sam2_points_by_obj = {1: [[1, 1]]}
        volume.sam2_labels_by_obj = {1: [1]}
        volume.points_data = {1: [[1, 1]]}
        volume.points_labels = {1: [1]}
        volume.obj_boxes = {1: np.zeros(4)}

        find_button(panel.widget, "Clear Prompts").click()

        assert volume.sam2_points_by_obj == {}
        assert volume.points_data == {}
        assert volume.obj_boxes == {}

    def test_clear_prompts_clears_a_rectangle(self, panel, scene):
        find_button(panel.widget, "Rectangle").click()
        scene.shapes_layer.data = [
            np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]])
        ]

        find_button(panel.widget, "Clear Prompts").click()

        assert len(scene.shapes_layer.data) == 0

    def test_next_at_the_last_image_disables_the_button(self, panel, scene):
        button = find_button(panel.widget, "Next Image")

        button.click()

        assert button.isEnabled() is False
        assert "No more images" in panel.status.text()

    def test_previous_at_the_first_image_disables_the_button(
        self, panel, scene
    ):
        button = find_button(panel.widget, "Previous Image")

        button.click()

        assert button.isEnabled() is False
        assert "Already at the first image" in panel.status.text()


class TestCropWidgetNavigation:
    """Paging through a folder rebuilds the table each time."""

    @pytest.fixture
    def two_images(self, scene, tmp_path):
        second = np.full((16, 16), 5, dtype=np.uint8)
        path = tmp_path / "second.tif"
        tifffile.imwrite(path, second)
        scene.images.append(str(path))
        return scene

    def test_next_loads_the_following_image(self, two_images):
        panel = build_panel(two_images)

        find_button(panel.widget, "Next Image").click()

        assert two_images.current_index == 1
        assert two_images.original_image[0, 0] == 5
        assert find_button(panel.widget, "Previous Image").isEnabled()
        # KNOWN DEFECT, pinned so a fix is noticed.  The panel refreshes
        # the prompt layer last, so its message is what the user is left
        # with -- and without a SAM2 predictor ``_load_current_image``
        # clears every layer and never rebuilds the points layer, so the
        # panel tells the user to load an image immediately after loading
        # one.  A substring check for "points layer" passes on both this
        # message and the correct one, which is why it is pinned exactly.
        assert (
            panel.status.text()
            == "No points layer found. Please load an image first."
        )
        assert not any(
            "Points" in layer.name for layer in two_images.viewer.layers
        )

    def test_the_table_is_rebuilt_for_the_new_image(self, two_images):
        panel = build_panel(two_images)
        first_table = two_images.label_table_widget

        find_button(panel.widget, "Next Image").click()

        assert two_images.label_table_widget is not first_table
        # The fresh image has no segmentation yet.
        assert two_images.label_table_widget.rowCount() == 0

    def test_previous_goes_back(self, two_images):
        panel = build_panel(two_images)
        find_button(panel.widget, "Next Image").click()

        find_button(panel.widget, "Previous Image").click()

        assert two_images.current_index == 0
        assert two_images.original_image[0, 0] != 5
        assert find_button(panel.widget, "Next Image").isEnabled()


class TestGenerate2dSegmentation:
    """The 2D setup pass: image preparation and state reset.

    This is the first point where the module talks to the model, so the
    stub predictor only records what it was handed -- every value checked
    below is computed by ``_generate_2d_segmentation`` itself.
    """

    @pytest.fixture
    def prepared(self, scene, torch_stub):
        scene.predictor = StubPredictor(np.zeros((16, 16), dtype=bool))
        return scene

    def test_the_predictor_receives_a_normalised_rgb_image(self, prepared):
        prepared._generate_2d_segmentation(0.7)

        (sent,) = prepared.predictor.set_image_calls
        assert sent.shape == (16, 16, 3)
        assert sent.dtype == np.float32
        assert sent.max() <= 1.0
        # Grey values are scaled, not clipped or re-ordered, and the
        # single channel is broadcast across all three.
        np.testing.assert_allclose(
            sent[:, :, 0], prepared.original_image / 255.0, atol=1e-6
        )
        np.testing.assert_array_equal(sent[:, :, 0], sent[:, :, 2])
        np.testing.assert_array_equal(sent, prepared.prepared_sam2_image)

    def test_the_interactive_state_is_reset(self, prepared):
        prepared.label_info = {9: {"area": 1, "score": 1.0}}
        prepared.obj_points = {9: [[1, 1]]}

        prepared._generate_2d_segmentation(0.7)

        assert prepared.segmentation_result.shape == (16, 16)
        assert prepared.segmentation_result.dtype == np.uint32
        assert not prepared.segmentation_result.any()
        assert prepared.label_info == {}
        assert prepared.obj_points == {}
        assert prepared.obj_labels == {}
        assert prepared.current_points == []
        assert prepared.current_obj_id == 1
        assert prepared.next_obj_id == 1
        assert prepared._sam2_next_obj_id == 1
        assert prepared.current_scale_factor == 1.0
        assert prepared.viewer.status.startswith("2D Mode:")

    def test_a_large_image_is_downscaled_before_the_model_sees_it(
        self, prepared
    ):
        prepared.current_image_for_segmentation = np.full(
            (1500, 1500), 200, dtype=np.uint8
        )

        prepared._generate_2d_segmentation(0.7)

        (sent,) = prepared.predictor.set_image_calls
        # 2.25 MP over a 2.0 MP budget -> sqrt(2/2.25) on each side.
        assert prepared.current_scale_factor == pytest.approx(
            np.sqrt(2.0 / 2.25)
        )
        assert sent.shape == (1414, 1414, 3)
        # The labels keep the full resolution of the original.
        assert prepared.segmentation_result.shape == (1500, 1500)

    def test_a_video_predictor_is_not_asked_to_set_an_image(
        self, scene, torch_stub
    ):
        """A predictor without ``set_image`` must not break the setup."""
        scene.predictor = PredictorWithoutSetImage()

        scene._generate_2d_segmentation(0.7)

        assert scene.prepared_sam2_image.shape == (16, 16, 3)
        assert scene.predictor.predict_kwargs == []
        assert scene.viewer.status.startswith("2D Mode:")


class TestRectangleSegmentation:
    """A drawn rectangle turned into a label by the image predictor."""

    SQUARE = np.array([[2.0, 3.0], [2.0, 9.0], [7.0, 9.0], [7.0, 3.0]])

    @pytest.fixture
    def boxed(self, scene, torch_stub):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2:8, 3:10] = True
        scene.segmentation_result = np.zeros((16, 16), dtype=np.uint32)
        scene.label_info = {}
        scene.predictor = StubPredictor(mask, score=0.75)
        return scene

    def test_the_mask_becomes_a_new_label(self, boxed):
        boxed._on_rectangle_added(self.SQUARE)

        assert boxed.segmentation_result[2, 3] == 1
        assert boxed.segmentation_result[7, 9] == 1
        assert boxed.segmentation_result[8, 3] == 0
        assert boxed.segmentation_result[2, 2] == 0
        assert int((boxed.segmentation_result == 1).sum()) == 42
        assert boxed.label_info[1] == {
            "area": 42,
            "center_y": pytest.approx(4.5),
            "center_x": pytest.approx(6.0),
            "score": pytest.approx(0.75),
        }

    def test_the_box_is_handed_over_in_xyxy_order(self, boxed):
        boxed._on_rectangle_added(self.SQUARE)

        (call,) = boxed.predictor.predict_kwargs
        # napari gives (y, x) corners; SAM2 wants (x0, y0, x1, y1).
        np.testing.assert_array_equal(call["box"], [3, 2, 9, 7])
        assert call["box"].dtype == np.float32
        assert call["multimask_output"] is False
        np.testing.assert_array_equal(boxed.obj_boxes[1], [3, 2, 9, 7])
        assert boxed.next_obj_id == 2

    def test_the_image_is_handed_over_as_uint8_rgb(self, boxed):
        boxed._on_rectangle_added(self.SQUARE)

        (sent,) = boxed.predictor.set_image_calls
        assert sent.shape == (16, 16, 3)
        assert sent.dtype == np.uint8
        np.testing.assert_array_equal(sent[:, :, 1], boxed.original_image)

    def test_a_second_box_gets_a_new_id_and_spares_the_first(self, boxed):
        boxed._on_rectangle_added(self.SQUARE)
        wider = np.zeros((16, 16), dtype=bool)
        wider[2:8, 3:12] = True  # overlaps everything label 1 owns
        boxed.predictor.mask = wider

        boxed._on_rectangle_added(self.SQUARE)

        assert boxed.segmentation_result[2, 3] == 1  # not overwritten
        assert boxed.segmentation_result[2, 10] == 2  # background filled
        assert int((boxed.segmentation_result == 1).sum()) == 42
        assert int((boxed.segmentation_result == 2).sum()) == 12
        assert boxed.next_obj_id == 3

    def test_a_smaller_mask_is_resized_onto_the_segmentation(self, boxed):
        small = np.zeros((8, 8), dtype=bool)
        small[0:4, 0:4] = True
        boxed.predictor.mask = small

        boxed._on_rectangle_added(self.SQUARE)

        assert boxed.segmentation_result.shape == (16, 16)
        assert int((boxed.segmentation_result == 1).sum()) == 64
        assert boxed.segmentation_result[0, 0] == 1
        assert boxed.segmentation_result[7, 7] == 1
        assert boxed.segmentation_result[8, 8] == 0

    def test_an_empty_prediction_changes_nothing(self, boxed):
        class NoMasks(StubPredictor):
            def predict(self, **kwargs):
                return [], [], None

        boxed.predictor = NoMasks(np.zeros((16, 16), dtype=bool))

        boxed._on_rectangle_added(self.SQUARE)

        assert not boxed.segmentation_result.any()
        assert boxed.label_info == {}

    def test_a_video_predictor_is_rejected_in_box_mode(
        self, scene, torch_stub
    ):
        scene.predictor = PredictorWithoutSetImage()
        scene.segmentation_result = np.zeros((16, 16), dtype=np.uint32)

        scene._on_rectangle_added(self.SQUARE)

        assert scene.viewer.status == (
            "Error: Rectangle mode requires Image Predictor (2D mode)"
        )
        assert scene.predictor.predict_kwargs == []
        assert not scene.segmentation_result.any()

    def test_an_unusable_corner_array_is_reported(self, boxed):
        boxed._on_rectangle_added(np.zeros((4, 5)))

        assert boxed.viewer.status == (
            "Error: Unexpected rectangle dimensions (4, 5). "
            "Expected (4,2) for 2D or (4,3) for 3D."
        )
        assert boxed.predictor.predict_kwargs == []


class TestPointSegmentation:
    """A click on the points layer turned into a label (2D)."""

    @pytest.fixture
    def clicked(self, scene, torch_stub):
        mask = np.zeros((16, 16), dtype=bool)
        mask[10:14, 10:14] = True
        scene.segmentation_result = np.zeros((16, 16), dtype=np.uint32)
        scene.label_info = {}
        scene.predictor = StubPredictor(mask, score=0.5)
        scene.points = FakePointsLayer(ndim=2)
        return scene

    def test_a_positive_click_creates_a_label(self, clicked):
        clicked._on_points_clicked(clicked.points, FakeEvent((11, 12)))

        assert int((clicked.segmentation_result == 1).sum()) == 16
        assert clicked.segmentation_result[10, 10] == 1
        assert clicked.segmentation_result[9, 10] == 0
        # SAM2 is given (x, y); the layer keeps napari's (y, x).
        assert clicked.obj_points[1] == [[12, 11]]
        assert clicked.obj_labels[1] == [1]
        np.testing.assert_array_equal(clicked.points.data, [[11, 12]])
        assert clicked.next_obj_id == 2
        assert clicked.label_info[1] == {
            "area": 16,
            "center_y": pytest.approx(11.5),
            "center_x": pytest.approx(11.5),
            "score": pytest.approx(0.5),
        }
        (call,) = clicked.predictor.predict_kwargs
        np.testing.assert_array_equal(call["point_coords"], [[12, 11]])
        np.testing.assert_array_equal(call["point_labels"], [1])
        assert call["multimask_output"] is True

    def test_a_shift_click_erases_from_that_object_only(self, clicked):
        clicked._on_points_clicked(clicked.points, FakeEvent((11, 12)))
        eraser = np.zeros((16, 16), dtype=bool)
        eraser[10:12, 10:14] = True
        clicked.predictor.mask = eraser

        clicked._on_points_clicked(
            clicked.points, FakeEvent((11, 12), modifiers=("Shift",))
        )

        assert clicked.segmentation_result[10, 10] == 0
        assert clicked.segmentation_result[13, 13] == 1
        assert int((clicked.segmentation_result == 1).sum()) == 8
        # The negative point joins the object it was placed on...
        assert clicked.obj_labels[1] == [1, -1]
        # ...rather than starting a new one.
        assert clicked.next_obj_id == 2

    def test_a_click_outside_the_image_is_undone(self, clicked):
        clicked._on_points_clicked(clicked.points, FakeEvent((99, 1)))

        assert "out of bounds" in clicked.viewer.status
        # The marker added for instant feedback is taken back again.
        assert len(clicked.points.data) == 0
        assert clicked.predictor.set_image_calls == []
        assert not clicked.segmentation_result.any()

    def test_a_click_before_the_segmentation_exists_is_refused(self, clicked):
        clicked.segmentation_result = None

        clicked._on_points_clicked(clicked.points, FakeEvent((11, 12)))

        assert clicked.viewer.status == (
            "Segmentation not ready. Please wait for image to load."
        )
        assert clicked.predictor.predict_kwargs == []
        assert len(clicked.points.data) == 0

    def test_a_video_predictor_is_rejected_in_2d_point_mode(
        self, scene, torch_stub
    ):
        scene.predictor = PredictorWithoutSetImage()
        scene.segmentation_result = np.zeros((16, 16), dtype=np.uint32)

        scene._on_points_clicked(FakePointsLayer(), FakeEvent((1, 1)))

        assert scene.viewer.status == (
            "Error: Point mode in 2D requires Image Predictor"
        )
        assert scene.predictor.predict_kwargs == []

    def test_drags_are_ignored(self, clicked):
        clicked._on_points_clicked(
            clicked.points, FakeEvent((11, 12), event_type="mouse_move")
        )

        assert len(clicked.points.data) == 0
        assert clicked.predictor.predict_kwargs == []
        assert not clicked.segmentation_result.any()


class Test3dPropagation:
    """Video propagation of one object across every frame."""

    @pytest.fixture
    def propagating(self, volume, torch_stub, no_threads):
        volume.segmentation_result = np.zeros((3, 8, 8), dtype=np.uint32)
        volume.label_info = {}
        mask = np.zeros((8, 8), dtype=float)
        mask[1:3, 1:3] = 1.0
        volume.predictor = StubVideoPredictor(mask, frames=3)
        volume._sam2_state = volume.predictor.state
        return volume

    def test_every_frame_receives_the_mask(self, propagating):
        propagating._propagate_mask_for_current_object(1, 0)

        for frame in range(3):
            assert (
                int((propagating.segmentation_result[frame] == 1).sum()) == 4
            )
            assert propagating.segmentation_result[frame][1, 1] == 1
            assert propagating.segmentation_result[frame][0, 0] == 0
        assert propagating.viewer.status == (
            "Propagation of object 1 complete"
        )

    def test_labels_already_in_a_frame_are_not_overwritten(
        self, propagating
    ):
        propagating.segmentation_result[1][1:3, 1:3] = 7

        propagating._propagate_mask_for_current_object(1, 0)

        assert int((propagating.segmentation_result[1] == 7).sum()) == 4
        assert not (propagating.segmentation_result[1] == 1).any()
        # The other frames are still filled in.
        assert int((propagating.segmentation_result[2] == 1).sum()) == 4

    def test_a_progress_overlay_tracks_the_frames(self, propagating):
        propagating._propagate_mask_for_current_object(1, 0)

        progress = next(
            layer
            for layer in propagating.viewer.layers
            if layer.name == "Propagation Progress"
        )
        assert progress.data.shape == (3, 8, 8)
        np.testing.assert_allclose(progress.data[2][1:3, 1:3], 0.8)
        assert progress.data[2][0, 0] == 0.0

    def test_a_mask_for_another_object_is_ignored(self, propagating):
        propagating._propagate_mask_for_current_object(2, 0)

        # The stub only ever reports object 1, so object 2 gets nothing.
        assert not propagating.segmentation_result.any()
        assert propagating.viewer.status == (
            "Propagation of object 2 complete"
        )

    def test_without_a_video_state_nothing_propagates(
        self, volume, torch_stub
    ):
        volume._sam2_state = None
        volume.predictor = StubVideoPredictor(
            np.ones((8, 8), dtype=float), frames=3
        )

        volume._propagate_mask_for_current_object(1, 0)

        assert volume.viewer.status == (
            "SAM2 3D state not initialized for propagation"
        )
        assert not any(
            "Propagation Progress" in layer.name
            for layer in volume.viewer.layers
        )


class Test3dRectangleSegmentation:
    """A rectangle drawn on one slice of a volume, then propagated."""

    RECT = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])

    @pytest.fixture
    def boxed_volume(self, volume, torch_stub, no_threads):
        volume.segmentation_result = np.zeros((3, 8, 8), dtype=np.uint32)
        volume.label_info = {}
        mask = np.zeros((8, 8), dtype=float)
        mask[1:4, 1:4] = 1.0
        volume.predictor = StubVideoPredictor(mask, frames=3)
        volume._sam2_state = volume.predictor.state
        return volume

    def test_the_box_is_sent_for_the_visible_frame(self, boxed_volume):
        boxed_volume.viewer.dims.set_current_step(0, 1)

        boxed_volume._on_rectangle_added(self.RECT)

        (call,) = boxed_volume.predictor.add_calls
        assert call["frame_idx"] == 1
        assert call["obj_id"] == 1
        assert call["inference_state"] is boxed_volume.predictor.state
        np.testing.assert_array_equal(call["box"], [1, 1, 3, 3])
        np.testing.assert_array_equal(boxed_volume.obj_boxes[1], [1, 1, 3, 3])
        assert boxed_volume._sam2_next_obj_id == 2

    def test_the_result_is_written_to_every_frame(self, boxed_volume):
        boxed_volume.viewer.dims.set_current_step(0, 1)

        boxed_volume._on_rectangle_added(self.RECT)

        for frame in range(3):
            assert (
                int((boxed_volume.segmentation_result[frame] == 1).sum()) == 9
            )
        assert boxed_volume.label_layer.name.startswith("Segmentation (")

    def test_without_a_video_state_the_box_is_refused(
        self, boxed_volume
    ):
        boxed_volume._sam2_state = None

        boxed_volume._on_rectangle_added(self.RECT)

        assert boxed_volume.viewer.status == (
            "Error: 3D segmentation state not initialized"
        )
        assert boxed_volume.predictor.add_calls == []
        assert not boxed_volume.segmentation_result.any()

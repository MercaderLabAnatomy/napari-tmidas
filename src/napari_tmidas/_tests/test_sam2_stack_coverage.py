"""Coverage for the SAM2 subprocess stack.

Three modules cooperate to run SAM2 out-of-process:

* ``napari_tmidas._sam2_server`` -- the script that runs *inside* sam2-env
  and answers pickle-framed requests on stdin/stdout.
* ``napari_tmidas._sam2_worker`` -- the napari-side client that owns the
  subprocess and turns every failure into ``Sam2Unavailable``.
* ``napari_tmidas.processing_functions.sam2_mp4`` -- the TIFF -> MP4 helper
  used to feed SAM2's video predictor.

Neither ``torch``/``sam2`` nor ``cv2`` is installed here, so those are
injected as fakes; no real subprocess is ever spawned.

The fakes are deliberately *asymmetric* -- every candidate mask, score and
low-resolution logit carries its own index -- so an assertion on the values
pins which candidate the code selected, not merely how many there were.
"""

import io
import os
import pickle
import struct
import subprocess
import sys
import types

import numpy as np
import pytest
import tifffile

import napari_tmidas._sam2_worker as worker_mod
from napari_tmidas.processing_functions import sam2_mp4 as mp4


def _import_server():
    """Import the server module without leaking its fd-1 redirection.

    Importing it runs ``os.dup2(2, 1)`` and rebinds ``sys.stdout`` at module
    scope, which would otherwise wreck pytest's capture for the whole run.
    """
    import importlib

    saved_fd = os.dup(1)
    saved_stdout = sys.stdout
    try:
        return importlib.import_module("napari_tmidas._sam2_server")
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        sys.stdout = saved_stdout


srv = _import_server()


# ---------------------------------------------------------------- helpers
def _frame(payload):
    blob = pickle.dumps(payload, protocol=4)
    return struct.pack(">I", len(blob)) + blob


def _frames(*payloads):
    return b"".join(_frame(p) for p in payloads)


def _unframe(data):
    out = []
    i = 0
    while i < len(data):
        (size,) = struct.unpack(">I", data[i : i + 4])
        i += 4
        out.append(pickle.loads(data[i : i + size]))
        i += size
    return out


class _ReadQueue:
    """A blocking-read stand-in for ``proc.stdout``."""

    def __init__(self, data=b"", chunk=None):
        self._data = data
        self.chunk = chunk

    def feed(self, data):
        self._data += data

    def read(self, n):
        if self.chunk is not None:
            n = min(n, self.chunk)
        out, self._data = self._data[:n], self._data[n:]
        return out


class _RecordingStdin:
    """A ``proc.stdin`` that keeps every raw byte the worker wrote."""

    def __init__(self):
        self.raw = b""
        self.flushes = 0

    def write(self, data):
        self.raw += bytes(data)
        return len(data)

    def flush(self):
        self.flushes += 1

    def close(self):
        pass


class _LoopbackStdin:
    """A ``proc.stdin`` that answers complete frames via *handler*."""

    def __init__(self, handler, out):
        self._buf = b""
        self._handler = handler
        self._out = out
        self.closed = False
        self.fail_on_write = None

    def write(self, data):
        if self.fail_on_write is not None:
            raise self.fail_on_write
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self._buf += bytes(data)
        return len(data)

    def flush(self):
        while len(self._buf) >= 4:
            (size,) = struct.unpack(">I", self._buf[:4])
            if len(self._buf) < 4 + size:
                return
            request = pickle.loads(self._buf[4 : 4 + size])
            self._buf = self._buf[4 + size :]
            reply = self._handler(request)
            if reply is None:
                continue
            self._out.feed(_frame(reply))

    def close(self):
        self.closed = True


class _FakeProc:
    """Enough of ``subprocess.Popen`` for the worker's protocol."""

    def __init__(self, handler=None, returncode=None):
        self.stdout = _ReadQueue()
        self.stdin = _LoopbackStdin(
            handler or (lambda req: {"ok": True}), self.stdout
        )
        self.returncode = returncode
        self.killed = False
        self.waits = []
        self.wait_timeout = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.wait_timeout:
            raise subprocess.TimeoutExpired("sam2", timeout)
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeManager:
    def __init__(self, root, env_created=True):
        self.env_dir = str(root / "env")
        self.sam2_repo_dir = str(root / "env" / "sam2_repo")
        self.checkpoints_dir = str(root / "env" / "checkpoints")
        self.python = str(root / "env" / "bin" / "python")
        self._env_created = env_created

    def is_env_created(self):
        return self._env_created

    def get_env_python_path(self):
        return self.python


class _FakePredictor:
    """Stands in for ``sam2.sam2_image_predictor.SAM2ImagePredictor``.

    Candidate *i* is identifiable in all three return values: its mask has a
    unique pixel at ``(0, i)``, its score is ``i / 4`` and every element of
    its low-resolution logits is ``i + 1``.  Only because of that can a test
    assert *which* candidate ``refine`` fed back.
    """

    def __init__(self, model):
        self.model = model
        self.image = None
        self.calls = []

    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        n = 3 if kwargs.get("multimask_output") else 1
        masks = np.zeros((n, 4, 4), dtype=np.uint8)
        masks[:, 1, 1] = 1
        for i in range(n):
            masks[i, 0, i] = 1
        scores = np.arange(n, dtype=np.float64) / 4.0
        low_res = np.zeros((n, 2, 2), dtype=np.float32)
        for i in range(n):
            low_res[i] = i + 1
        return masks, scores, low_res


def _expected_masks(n):
    """The bool masks ``_FakePredictor`` produces for *n* candidates."""
    masks = np.zeros((n, 4, 4), dtype=bool)
    masks[:, 1, 1] = True
    for i in range(n):
        masks[i, 0, i] = True
    return masks


def _make_torch(cuda=False, mps=None):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    backends = types.SimpleNamespace()
    if mps is not None:
        backends.mps = types.SimpleNamespace(is_available=lambda: mps)
    torch.backends = backends
    return torch


def _install_fake_sam2(monkeypatch, torch_mod):
    """Put fake ``torch``/``sam2`` modules in ``sys.modules``."""
    build_calls = []

    def build_sam2(model_cfg, checkpoint, device=None):
        build_calls.append((model_cfg, checkpoint, device))
        return {"cfg": model_cfg}

    build_mod = types.ModuleType("sam2.build_sam")
    build_mod.build_sam2 = build_sam2
    pred_mod = types.ModuleType("sam2.sam2_image_predictor")
    pred_mod.SAM2ImagePredictor = _FakePredictor
    pkg = types.ModuleType("sam2")
    pkg.build_sam = build_mod
    pkg.sam2_image_predictor = pred_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "sam2", pkg)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_mod)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", pred_mod)
    return build_calls


def _run_server(monkeypatch, *requests):
    """Drive ``_sam2_server.main`` over a canned stdin, return the replies."""
    out = io.BytesIO()
    stdin = types.SimpleNamespace(buffer=io.BytesIO(_frames(*requests)))
    monkeypatch.setattr(srv, "_PROTO_OUT", out)
    monkeypatch.setattr(srv.sys, "stdin", stdin)
    srv.main()
    return _unframe(out.getvalue())


@pytest.fixture(autouse=True)
def _reset_worker_singleton():
    yield
    worker_mod.Sam2Worker._instance = None


@pytest.fixture
def ready_env(tmp_path, monkeypatch):
    """A fake sam2-env that ``availability()`` accepts."""
    manager = _FakeManager(tmp_path)
    os.makedirs(manager.checkpoints_dir, exist_ok=True)
    ckpt = os.path.join(manager.checkpoints_dir, "sam2.1_hiera_large.pt")
    with open(ckpt, "wb") as fh:
        fh.write(b"not really a checkpoint")
    monkeypatch.setattr(
        worker_mod.Sam2Worker, "_manager", staticmethod(lambda: manager)
    )
    monkeypatch.setattr(worker_mod.atexit, "register", lambda fn: fn)
    return manager


# ================================================== server: framing / send
class TestServerFraming:
    """The 4-byte-length pickle framing used in both directions."""

    def test_read_exact_reassembles_short_reads(self):
        stream = _ReadQueue(b"abcdefgh", chunk=3)
        assert srv._read_exact(stream, 8) == b"abcdefgh"

    def test_read_exact_returns_none_at_clean_eof(self):
        assert srv._read_exact(_ReadQueue(b""), 4) is None

    def test_read_exact_returns_none_on_truncated_stream(self):
        assert srv._read_exact(_ReadQueue(b"ab"), 4) is None

    def test_send_writes_length_prefixed_pickle(self, monkeypatch):
        out = io.BytesIO()
        monkeypatch.setattr(srv, "_PROTO_OUT", out)
        srv._send({"ok": True, "device": "cpu"})
        blob = out.getvalue()
        (size,) = struct.unpack(">I", blob[:4])
        assert size == len(blob) - 4
        assert pickle.loads(blob[4:]) == {"ok": True, "device": "cpu"}

    def test_each_side_parses_the_bytes_the_other_side_wrote(
        self, monkeypatch
    ):
        """No hand-rolled framing here: both directions use the real code.

        The reply is produced by ``_sam2_server._send`` and decoded by
        ``_sam2_worker._read_reply``; the request is produced by
        ``Sam2Worker._request`` and decoded by ``_sam2_server._read_exact``.
        """
        out = io.BytesIO()
        monkeypatch.setattr(srv, "_PROTO_OUT", out)
        srv._send({"ok": True, "device": "cpu"})

        stdin = _RecordingStdin()
        proc = types.SimpleNamespace(
            stdin=stdin,
            stdout=_ReadQueue(out.getvalue()),
            poll=lambda: None,
        )
        worker = worker_mod.Sam2Worker()
        worker._proc = proc
        assert worker._request({"op": "ping", "n": 7}) == {
            "ok": True,
            "device": "cpu",
        }

        stream = io.BytesIO(stdin.raw)
        header = srv._read_exact(stream, 4)
        (size,) = struct.unpack(">I", header)
        assert pickle.loads(srv._read_exact(stream, size)) == {
            "op": "ping",
            "n": 7,
        }
        # Exactly one frame, and it was flushed.
        assert srv._read_exact(stream, 4) is None
        assert stdin.flushes == 1


# ============================================================ server: load
class TestServerSessionLoad:
    """Device selection and predictor construction inside sam2-env."""

    def test_prefers_cuda_when_available(self, monkeypatch):
        calls = _install_fake_sam2(monkeypatch, _make_torch(cuda=True))
        session = srv._Session()
        device = session.load(checkpoint="ckpt.pt", model_cfg="cfg.yaml")
        assert device == "cuda"
        assert session.device == "cuda"
        assert calls == [("cfg.yaml", "ckpt.pt", "cuda")]
        assert isinstance(session.predictor, _FakePredictor)
        assert session.predictor.model == {"cfg": "cfg.yaml"}

    def test_falls_back_to_mps(self, monkeypatch):
        calls = _install_fake_sam2(
            monkeypatch, _make_torch(cuda=False, mps=True)
        )
        session = srv._Session()
        assert session.load("c", "k") == "mps"
        assert session.device == "mps"
        # build_sam2(model_cfg, checkpoint, device=...)
        assert calls == [("k", "c", "mps")]

    def test_cpu_when_mps_present_but_unavailable(self, monkeypatch):
        calls = _install_fake_sam2(
            monkeypatch, _make_torch(cuda=False, mps=False)
        )
        session = srv._Session()
        assert session.load("c", "k") == "cpu"
        assert session.device == "cpu"
        assert calls == [("k", "c", "cpu")]

    def test_cpu_when_torch_has_no_mps_backend(self, monkeypatch):
        calls = _install_fake_sam2(
            monkeypatch, _make_torch(cuda=False, mps=None)
        )
        session = srv._Session()
        assert session.load("c", "k") == "cpu"
        assert calls == [("k", "c", "cpu")]

    def test_a_fresh_session_has_nothing_loaded(self):
        session = srv._Session()
        assert (session.predictor, session.device, session.low_res) == (
            None,
            None,
            None,
        )


# ======================================================= server: inference
class TestServerSessionInference:
    """segment/refine coerce the predictor output into the wire types."""

    def _loaded(self, monkeypatch):
        _install_fake_sam2(monkeypatch, _make_torch())
        session = srv._Session()
        session.load("cfg", "ckpt")
        return session

    def test_segment_returns_bool_masks_and_float32_scores(self, monkeypatch):
        session = self._loaded(monkeypatch)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        coords = [[1, 1]]
        out = session.segment(image, coords, [1])

        assert out["masks"].dtype == np.bool_
        np.testing.assert_array_equal(out["masks"], _expected_masks(3))
        assert out["scores"].dtype == np.float32
        np.testing.assert_allclose(out["scores"], [0.0, 0.25, 0.5])

        # The prompt reached the predictor untouched, on the encoded image.
        assert session.predictor.image is image
        call = session.predictor.calls[-1]
        assert call["point_coords"] is coords
        assert call["point_labels"] == [1]
        assert call["multimask_output"] is True
        assert "mask_input" not in call

        # The logits of every candidate are cached for a later refine.
        np.testing.assert_array_equal(
            session.low_res, np.array([1.0, 2.0, 3.0])[:, None, None]
            * np.ones((1, 2, 2), np.float32)
        )

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_refine_feeds_the_chosen_candidate_back_as_mask_input(
        self, monkeypatch, index
    ):
        session = self._loaded(monkeypatch)
        session.segment(np.zeros((4, 4, 3), np.uint8), [[1, 1]], [1])
        out = session.refine(index, [[2, 2]], [1])

        call = session.predictor.calls[-1]
        # Candidate *index* -- and no other -- was fed back: the fake's
        # logits are all-``index + 1`` for candidate ``index``.
        assert call["mask_input"].shape == (1, 2, 2)
        np.testing.assert_array_equal(
            call["mask_input"], np.full((1, 2, 2), index + 1, np.float32)
        )
        assert call["point_coords"] == [[2, 2]]
        assert call["multimask_output"] is False

        np.testing.assert_array_equal(out["masks"], _expected_masks(1))
        np.testing.assert_allclose(out["scores"], [0.0])
        # The cache now holds the single refined candidate.
        np.testing.assert_array_equal(
            session.low_res, np.ones((1, 2, 2), np.float32)
        )

    def test_refine_before_segment_is_a_runtime_error(self, monkeypatch):
        session = self._loaded(monkeypatch)
        with pytest.raises(RuntimeError, match="refine called before"):
            session.refine(0, [[1, 1]], [1])
        # It failed before touching the predictor.
        assert session.predictor.calls == []


# ======================================================== server: dispatch
class TestServerDispatch:
    """``main()``'s command table, including the unknown-op branch."""

    def test_empty_stdin_exits_without_replying(self, monkeypatch):
        assert _run_server(monkeypatch) == []

    def test_unknown_op_becomes_an_error_reply(self, monkeypatch):
        (reply,) = _run_server(monkeypatch, {"op": "bogus"})
        assert reply == {"error": "unknown op 'bogus'"}

    def test_missing_op_key_is_reported_as_none(self, monkeypatch):
        (reply,) = _run_server(monkeypatch, {"nothing": 1})
        assert reply == {"error": "unknown op None"}

    def test_ping_reports_the_current_device(self, monkeypatch):
        _install_fake_sam2(monkeypatch, _make_torch(cuda=True))
        replies = _run_server(
            monkeypatch,
            {"op": "ping"},
            {"op": "load", "checkpoint": "k", "model_cfg": "c"},
            {"op": "ping"},
        )
        assert replies[0] == {"ok": True, "device": None}
        assert replies[1] == {"ok": True, "device": "cuda"}
        assert replies[2] == {"ok": True, "device": "cuda"}

    def test_shutdown_replies_once_and_stops_reading(self, monkeypatch):
        replies = _run_server(monkeypatch, {"op": "shutdown"}, {"op": "ping"})
        assert replies == [{"ok": True}]

    def test_segment_and_refine_round_trip(self, monkeypatch):
        _install_fake_sam2(monkeypatch, _make_torch())
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        replies = _run_server(
            monkeypatch,
            {"op": "load", "checkpoint": "k", "model_cfg": "c"},
            {
                "op": "segment",
                "image": image,
                "point_coords": np.array([[1, 1]], np.float32),
                "point_labels": np.array([1], np.int32),
            },
            {
                "op": "refine",
                "index": 2,
                "point_coords": np.array([[1, 1]], np.float32),
                "point_labels": np.array([1], np.int32),
            },
        )
        assert replies[0] == {"ok": True, "device": "cpu"}
        # Values survive the pickle round trip, not just the shapes.
        np.testing.assert_array_equal(replies[1]["masks"], _expected_masks(3))
        assert replies[1]["masks"].dtype == np.bool_
        np.testing.assert_allclose(replies[1]["scores"], [0.0, 0.25, 0.5])
        np.testing.assert_array_equal(replies[2]["masks"], _expected_masks(1))
        assert set(replies[2]) == {"masks", "scores"}

    def test_a_failing_request_returns_an_error_with_a_traceback(
        self, monkeypatch
    ):
        # segment before load -> predictor is None -> AttributeError
        (reply,) = _run_server(
            monkeypatch, {"op": "segment", "image": np.zeros((2, 2))}
        )
        assert reply["error"].startswith("AttributeError:")
        assert "set_image" in reply["error"]
        assert "Traceback" in reply["traceback"]
        assert "_sam2_server.py" in reply["traceback"]

    def test_a_failing_request_does_not_kill_the_server(self, monkeypatch):
        replies = _run_server(monkeypatch, {"op": "load"}, {"op": "ping"})
        assert replies[0]["error"] == "KeyError: 'checkpoint'"
        assert replies[1] == {"ok": True, "device": None}


# ================================================== worker: availability
class TestWorkerAvailability:
    """``availability()`` is a filesystem-only precheck for the widgets."""

    def test_missing_env_is_reported_with_the_env_path(
        self, tmp_path, monkeypatch
    ):
        manager = _FakeManager(tmp_path, env_created=False)
        monkeypatch.setattr(
            worker_mod.Sam2Worker, "_manager", staticmethod(lambda: manager)
        )
        ok, message = worker_mod.Sam2Worker.availability()
        assert ok is False
        assert "SAM2 is not installed" in message
        assert manager.env_dir in message

    def test_missing_checkpoint_is_reported(self, tmp_path, monkeypatch):
        manager = _FakeManager(tmp_path)
        monkeypatch.setattr(
            worker_mod.Sam2Worker, "_manager", staticmethod(lambda: manager)
        )
        ok, message = worker_mod.Sam2Worker.availability()
        assert ok is False
        assert "checkpoint is missing" in message
        assert manager.checkpoints_dir in message

    def test_missing_server_script_is_reported(self, ready_env, monkeypatch):
        monkeypatch.setattr(
            worker_mod, "_server_script", lambda: "/no/such/script.py"
        )
        ok, message = worker_mod.Sam2Worker.availability()
        assert (ok, message) == (
            False,
            "The SAM2 server script is missing from the plugin.",
        )

    def test_all_present_is_available(self, ready_env):
        assert worker_mod.Sam2Worker.availability() == (
            True,
            "SAM2 is available.",
        )

    def test_checkpoint_path_lives_under_the_managers_checkpoints_dir(
        self, ready_env
    ):
        path = worker_mod.Sam2Worker.checkpoint_path()
        assert path == os.path.join(
            ready_env.checkpoints_dir, "sam2.1_hiera_large.pt"
        )

    def test_server_script_is_shipped_next_to_the_worker(self):
        path = worker_mod._server_script()
        assert path == os.path.join(
            os.path.dirname(worker_mod.__file__), "_sam2_server.py"
        )
        assert os.path.exists(path)
        assert os.path.samefile(path, srv.__file__)

    def test_manager_is_the_shared_sam2_env_manager(self):
        from napari_tmidas.processing_functions import sam2_env_manager

        assert worker_mod.Sam2Worker._manager() is sam2_env_manager.manager


# ====================================================== worker: lifecycle
class TestWorkerLifecycle:
    """start/stop/restart, including the dead-process paths."""

    def test_start_refuses_when_sam2_is_unavailable(
        self, tmp_path, monkeypatch
    ):
        manager = _FakeManager(tmp_path, env_created=False)
        monkeypatch.setattr(
            worker_mod.Sam2Worker, "_manager", staticmethod(lambda: manager)
        )
        spawned = []
        monkeypatch.setattr(
            worker_mod.subprocess,
            "Popen",
            lambda *a, **k: spawned.append(a) or _FakeProc(),
        )
        worker = worker_mod.Sam2Worker()
        with pytest.raises(worker_mod.Sam2Unavailable, match="not installed"):
            worker.start()
        assert spawned == []
        assert worker.is_running() is False

    def test_start_spawns_the_server_and_records_the_device(
        self, ready_env, monkeypatch
    ):
        os.makedirs(ready_env.sam2_repo_dir, exist_ok=True)
        seen = []

        def handler(request):
            seen.append(request)
            return {"ok": True, "device": "cuda"}

        recorded = {}

        def fake_popen(argv, **kwargs):
            recorded["argv"] = argv
            recorded["kwargs"] = kwargs
            return _FakeProc(handler)

        monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
        worker = worker_mod.Sam2Worker()
        worker.start()

        assert recorded["argv"] == [
            ready_env.python,
            "-u",
            worker_mod._server_script(),
        ]
        assert recorded["kwargs"]["stdin"] is subprocess.PIPE
        assert recorded["kwargs"]["stdout"] is subprocess.PIPE
        assert recorded["kwargs"]["stderr"] is None
        assert recorded["kwargs"]["cwd"] == ready_env.sam2_repo_dir
        assert seen == [
            {
                "op": "load",
                "checkpoint": worker_mod.Sam2Worker.checkpoint_path(),
                "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
            }
        ]
        assert worker.device == "cuda"
        assert worker.is_running() is True

    def test_start_registers_an_atexit_stop(self, ready_env, monkeypatch):
        registered = []
        monkeypatch.setattr(
            worker_mod.atexit, "register", registered.append
        )
        monkeypatch.setattr(
            worker_mod.subprocess,
            "Popen",
            lambda *a, **k: _FakeProc(lambda r: {"ok": True, "device": "cpu"}),
        )
        worker = worker_mod.Sam2Worker()
        worker.start()
        assert registered == [worker.stop]

    def test_start_without_a_repo_checkout_runs_with_no_cwd(
        self, ready_env, monkeypatch
    ):
        recorded = {}

        def fake_popen(argv, **kwargs):
            recorded.update(kwargs)
            return _FakeProc(lambda r: {"ok": True, "device": "cpu"})

        monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
        assert not os.path.isdir(ready_env.sam2_repo_dir)
        worker = worker_mod.Sam2Worker()
        worker.start()
        assert recorded["cwd"] is None

    def test_start_is_idempotent_while_running(self, ready_env, monkeypatch):
        spawns = []
        loads = []

        def fake_popen(argv, **kwargs):
            spawns.append(argv)
            return _FakeProc(
                lambda r: loads.append(r) or {"ok": True, "device": "cpu"}
            )

        monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
        worker = worker_mod.Sam2Worker()
        worker.start()
        worker.start()
        assert len(spawns) == 1
        # The second start() did not re-load the model either.
        assert len(loads) == 1
        assert worker.device == "cpu"

    def test_a_failing_load_stops_the_process_and_raises(
        self, ready_env, monkeypatch
    ):
        proc = _FakeProc(lambda r: {"error": "no checkpoint"})
        monkeypatch.setattr(
            worker_mod.subprocess, "Popen", lambda *a, **k: proc
        )
        worker = worker_mod.Sam2Worker()
        with pytest.raises(worker_mod.Sam2Unavailable, match="no checkpoint"):
            worker.start()
        assert worker.is_running() is False
        assert worker._proc is None
        assert worker.device is None
        assert proc.stdin.closed is True

    def test_stop_is_a_no_op_without_a_process(self):
        worker = worker_mod.Sam2Worker()
        worker.stop()
        assert worker._proc is None

    def test_stop_on_an_already_exited_process_clears_the_device(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc(returncode=1)
        worker._proc = proc
        worker._device = "cuda"
        worker.stop()
        assert worker._proc is None
        assert worker.device is None
        # An exited process is not waited on or killed again.
        assert proc.waits == []
        assert proc.killed is False
        assert proc.stdin.closed is False

    def test_stop_closes_stdin_and_waits(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc()
        worker._proc = proc
        worker.stop()
        assert proc.stdin.closed is True
        assert proc.waits == [5]
        assert proc.killed is False

    def test_stop_kills_a_process_that_will_not_exit(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc()
        proc.wait_timeout = True
        worker._proc = proc
        worker.stop()
        assert proc.waits == [5]
        assert proc.killed is True

    def test_stop_survives_a_closed_stdin(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc()

        def boom():
            raise OSError("already closed")

        proc.stdin.close = boom
        worker._proc = proc
        worker.stop()
        assert proc.waits == [5]
        assert worker._proc is None

    def test_instance_is_a_singleton_that_shutdown_clears(
        self, ready_env, monkeypatch
    ):
        procs = []

        def fake_popen(argv, **kwargs):
            proc = _FakeProc(lambda r: {"ok": True, "device": "cpu"})
            procs.append(proc)
            return proc

        monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
        first = worker_mod.Sam2Worker.instance()
        second = worker_mod.Sam2Worker.instance()
        assert first is second
        assert len(procs) == 1
        assert first.device == "cpu"

        worker_mod.Sam2Worker.shutdown_instance()
        assert worker_mod.Sam2Worker._instance is None
        assert procs[0].stdin.closed is True
        assert first.device is None

        third = worker_mod.Sam2Worker.instance()
        assert third is not first
        assert len(procs) == 2
        assert third.is_running() is True
        worker_mod.Sam2Worker.shutdown_instance()

    def test_shutdown_instance_without_an_instance_is_harmless(self):
        worker_mod.Sam2Worker._instance = None
        worker_mod.Sam2Worker.shutdown_instance()
        assert worker_mod.Sam2Worker._instance is None

    def test_a_dead_process_restarts_on_the_next_start(
        self, ready_env, monkeypatch
    ):
        procs = []

        def fake_popen(argv, **kwargs):
            proc = _FakeProc(lambda r: {"ok": True, "device": "cpu"})
            procs.append(proc)
            return proc

        monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
        worker = worker_mod.Sam2Worker()
        worker.start()
        procs[0].returncode = 137  # the server crashed
        assert worker.is_running() is False
        worker.start()
        assert len(procs) == 2
        assert worker._proc is procs[1]
        assert worker.is_running() is True


# ======================================================== worker: protocol
class TestWorkerProtocol:
    """``_request`` turns every transport failure into Sam2Unavailable."""

    def test_request_without_a_process_reports_the_worker_is_down(self):
        worker = worker_mod.Sam2Worker()
        with pytest.raises(worker_mod.Sam2Unavailable, match="is not running"):
            worker._request({"op": "ping"})

    def test_request_on_an_exited_process_reports_the_worker_is_down(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc(returncode=2)
        worker._proc = proc
        with pytest.raises(worker_mod.Sam2Unavailable, match="is not running"):
            worker._request({"op": "ping"})
        # Nothing was written to a dead process.
        assert proc.stdin._buf == b""

    def test_a_broken_pipe_stops_the_worker(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc()
        proc.stdin.fail_on_write = BrokenPipeError("pipe gone")
        worker._proc = proc
        with pytest.raises(
            worker_mod.Sam2Unavailable, match="Lost contact.*pipe gone"
        ):
            worker._request({"op": "ping"})
        assert worker._proc is None

    def test_an_oserror_on_write_stops_the_worker(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc()
        proc.stdin.fail_on_write = OSError("bad fd")
        worker._proc = proc
        with pytest.raises(
            worker_mod.Sam2Unavailable, match="Lost contact.*bad fd"
        ) as excinfo:
            worker._request({"op": "ping"})
        assert isinstance(excinfo.value.__cause__, OSError)
        assert worker._proc is None

    def test_a_silent_death_mid_request_is_reported(self):
        worker = worker_mod.Sam2Worker()
        # The handler answers nothing, so the reply read hits EOF.
        worker._proc = _FakeProc(lambda request: None)
        with pytest.raises(
            worker_mod.Sam2Unavailable, match="exited while handling"
        ):
            worker._request({"op": "ping"})
        assert worker._proc is None

    def test_an_error_reply_becomes_sam2_failed(self):
        worker = worker_mod.Sam2Worker()
        proc = _FakeProc(lambda request: {"error": "CUDA OOM"})
        worker._proc = proc
        with pytest.raises(
            worker_mod.Sam2Unavailable, match="^SAM2 failed: CUDA OOM$"
        ):
            worker._request({"op": "ping"})
        # An application-level error keeps the process alive.
        assert worker._proc is proc

    def test_a_good_reply_is_returned_verbatim(self):
        worker = worker_mod.Sam2Worker()
        worker._proc = _FakeProc(lambda request: {"echo": request})
        reply = worker._request({"op": "ping", "n": 3})
        assert reply == {"echo": {"op": "ping", "n": 3}}

    def test_read_reply_returns_none_on_a_missing_header(self):
        proc = types.SimpleNamespace(stdout=_ReadQueue(b""))
        assert worker_mod._read_reply(proc) is None

    def test_read_reply_returns_none_on_a_truncated_body(self):
        blob = _frame({"ok": True})
        proc = types.SimpleNamespace(stdout=_ReadQueue(blob[:-3]))
        assert worker_mod._read_reply(proc) is None

    def test_read_reply_reassembles_a_chunked_stream(self):
        payload = {"masks": np.eye(2, dtype=bool), "n": 5}
        proc = types.SimpleNamespace(
            stdout=_ReadQueue(_frame(payload), chunk=3)
        )
        reply = worker_mod._read_reply(proc)
        assert reply["n"] == 5
        np.testing.assert_array_equal(reply["masks"], np.eye(2, dtype=bool))

    def test_read_reply_stops_at_the_frame_boundary(self):
        """A second frame is left in the stream for the next call."""
        stream = _ReadQueue(_frames({"first": 1}, {"second": 2}))
        proc = types.SimpleNamespace(stdout=stream)
        assert worker_mod._read_reply(proc) == {"first": 1}
        assert worker_mod._read_reply(proc) == {"second": 2}
        assert worker_mod._read_reply(proc) is None


# ======================================================= worker: inference
class TestWorkerInference:
    """segment/refine build the request and unpack the reply."""

    def test_segment_coerces_the_prompt_dtypes(self):
        seen = []

        def handler(request):
            seen.append(request)
            return {
                "masks": np.ones((2, 3, 3), dtype=bool),
                "scores": np.array([0.9, 0.4], dtype=np.float32),
            }

        worker = worker_mod.Sam2Worker()
        worker._proc = _FakeProc(handler)
        image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        masks, scores = worker.segment(image, [[1, 2], [0, 0]], [1, 0])

        request = seen[0]
        assert request["op"] == "segment"
        assert set(request) == {
            "op",
            "image",
            "point_coords",
            "point_labels",
        }
        assert request["point_coords"].dtype == np.float32
        assert request["point_labels"].dtype == np.int32
        np.testing.assert_array_equal(request["image"], image)
        np.testing.assert_array_equal(
            request["point_coords"], [[1.0, 2.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(request["point_labels"], [1, 0])
        assert masks.dtype == np.bool_
        np.testing.assert_array_equal(masks, np.ones((2, 3, 3), bool))
        np.testing.assert_allclose(scores, [0.9, 0.4], rtol=1e-6)

    def test_segment_sends_a_contiguous_copy_of_a_strided_view(
        self, monkeypatch
    ):
        """Inspected at the pickle seam, which is the only honest one.

        Asserting on the array the *handler* receives proves nothing here:
        unpickling any ndarray yields a contiguous one, so the received
        image looks identical whether or not ``segment`` copied.  Intercept
        ``pickle.dumps`` instead and look at the payload the worker built.
        """
        payloads = []
        real_dumps = pickle.dumps

        def recording_dumps(obj, protocol=None):
            payloads.append(obj)
            return real_dumps(obj, protocol=protocol)

        monkeypatch.setattr(
            worker_mod,
            "pickle",
            types.SimpleNamespace(
                dumps=recording_dumps, loads=pickle.loads
            ),
        )
        worker = worker_mod.Sam2Worker()
        worker._proc = _FakeProc(
            lambda request: {
                "masks": np.zeros((1, 2, 2), bool),
                "scores": np.zeros(1),
            }
        )
        base = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        view = base[::2]
        assert not view.flags["C_CONTIGUOUS"]

        worker.segment(view, [[0, 0]], [1])

        sent = payloads[0]["image"]
        assert sent.flags["C_CONTIGUOUS"] is True
        assert sent is not view
        assert not np.shares_memory(sent, base)
        np.testing.assert_array_equal(sent, view)

    def test_refine_returns_a_single_mask_and_a_python_float(self):
        seen = []
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 2] = True

        def handler(request):
            seen.append(request)
            return {
                "masks": mask[None],
                "scores": np.array([0.75], dtype=np.float32),
            }

        worker = worker_mod.Sam2Worker()
        worker._proc = _FakeProc(handler)
        got, score = worker.refine(np.int64(2), [[1, 1]], [1])

        assert seen[0]["op"] == "refine"
        assert seen[0]["index"] == 2
        # A numpy integer would not survive pickling into sam2-env cleanly.
        assert type(seen[0]["index"]) is int
        assert seen[0]["point_coords"].dtype == np.float32
        np.testing.assert_array_equal(got, mask)
        assert type(score) is float
        assert score == pytest.approx(0.75)

    def test_worker_talks_to_the_real_server_session(self, monkeypatch):
        """End-to-end: the worker's frames drive the server's dispatch."""
        _install_fake_sam2(monkeypatch, _make_torch())
        session = srv._Session()

        def handler(request):
            out = io.BytesIO()
            monkeypatch.setattr(srv, "_PROTO_OUT", out)
            monkeypatch.setattr(
                srv.sys,
                "stdin",
                types.SimpleNamespace(buffer=io.BytesIO(_frame(request))),
            )
            monkeypatch.setattr(srv, "_Session", lambda: session)
            srv.main()
            return _unframe(out.getvalue())[0]

        worker = worker_mod.Sam2Worker()
        worker._proc = _FakeProc(handler)
        assert worker._request(
            {"op": "load", "checkpoint": "k", "model_cfg": "c"}
        ) == {"ok": True, "device": "cpu"}

        masks, scores = worker.segment(
            np.zeros((4, 4, 3), np.uint8), [[1, 1]], [1]
        )
        assert masks.dtype == np.bool_
        np.testing.assert_array_equal(masks, _expected_masks(3))
        np.testing.assert_allclose(scores, [0.0, 0.25, 0.5])

        mask, score = worker.refine(2, [[1, 1]], [1])
        # Candidate 2's logits made it all the way to the predictor.
        np.testing.assert_array_equal(
            session.predictor.calls[-1]["mask_input"],
            np.full((1, 2, 2), 3.0, np.float32),
        )
        np.testing.assert_array_equal(mask, _expected_masks(1)[0])
        assert type(score) is float
        assert score == 0.0


# =============================================== sam2_mp4: fake OpenCV
class _FakeWriter:
    def __init__(self, path, fourcc, fps, size, opened):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self._opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):  # noqa: N802 - mirrors the cv2 API
        return self._opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class _FakeCapture:
    def __init__(self, opened, frames):
        self._opened = opened
        self._frames = list(frames)
        self.released = False

    def isOpened(self):  # noqa: N802 - mirrors the cv2 API
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


class _FakeCv2:
    COLOR_GRAY2BGR = 8

    def __init__(self, writers_open=(True,), capture=None):
        self._writers_open = list(writers_open)
        self.writers = []
        self.imwrites = []
        self.capture = capture

    def VideoWriter_fourcc(self, *codec):  # noqa: N802 - cv2 API
        return "".join(codec)

    def VideoWriter(self, path, fourcc, fps, size):  # noqa: N802 - cv2 API
        opened = self._writers_open.pop(0) if self._writers_open else False
        writer = _FakeWriter(path, fourcc, fps, size, opened)
        self.writers.append(writer)
        return writer

    def VideoCapture(self, path):  # noqa: N802 - cv2 API
        return self.capture

    def cvtColor(self, frame, code):  # noqa: N802 - cv2 API
        assert code == self.COLOR_GRAY2BGR
        return np.repeat(frame[:, :, None], 3, axis=2)

    def resize(self, frame, size):
        width, height = size
        out = np.zeros((height, width) + frame.shape[2:], frame.dtype)
        rows = min(height, frame.shape[0])
        cols = min(width, frame.shape[1])
        out[:rows, :cols] = frame[:rows, :cols]
        return out

    def imwrite(self, path, frame):
        self.imwrites.append((path, frame.copy()))
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG")
        return True


@pytest.fixture
def fake_cv2(monkeypatch):
    fake = _FakeCv2()
    monkeypatch.setattr(mp4, "cv2", fake)
    return fake


@pytest.fixture
def fake_mkdtemp(monkeypatch, tmp_path):
    made = []

    def mkdtemp(prefix="tmp", **kwargs):
        path = tmp_path / f"{prefix}{len(made)}"
        path.mkdir()
        made.append(path)
        return str(path)

    monkeypatch.setattr(mp4.tempfile, "mkdtemp", mkdtemp)
    return made


def _write_tiff(path, array, **kwargs):
    # Without an explicit photometric, tifffile stores a 3-or-4 long trailing
    # axis as planar RGB in a single page, which is not the layout under test.
    kwargs.setdefault("photometric", "minisblack")
    tifffile.imwrite(str(path), array, **kwargs)
    return path


def _bgr(frame):
    """What the fake cv2 makes of a grayscale frame."""
    return np.repeat(frame[:, :, None], 3, axis=2)


# ============================================== sam2_mp4: normalisation
class TestNormalizeFrame:
    """``_normalize_frame_to_uint8`` maps any dtype onto 0..255."""

    def test_uint8_is_passed_through_untouched(self):
        frame = np.arange(6, dtype=np.uint8).reshape(2, 3)
        assert mp4._normalize_frame_to_uint8(frame) is frame

    def test_uint16_ramp_is_stretched_but_never_reaches_255(self):
        frame = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
        out = mp4._normalize_frame_to_uint8(frame)
        assert out.dtype == np.uint8
        # The 1e-10 added to the denominator makes the scale factor slightly
        # too small, and the truncating .astype(uint8) then costs the top
        # code: the brightest pixel of an integer frame lands on 254, never
        # 255. Pinned so a change to the formula is noticed (see the report).
        np.testing.assert_array_equal(out, [[0, 63], [127, 254]])

    def test_float32_frame_is_stretched_to_the_full_range(self):
        frame = np.linspace(-1.0, 1.0, 9, dtype=np.float32).reshape(3, 3)
        out = mp4._normalize_frame_to_uint8(frame)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(
            out, [[0, 31, 63], [95, 127, 159], [191, 223, 255]]
        )

    def test_float64_loses_the_top_code_where_float32_does_not(self):
        """The epsilon is invisible in float32 and visible in float64.

        Same data, same range, different ceiling -- 255 above, 254 here.
        """
        frame = np.linspace(-1.0, 1.0, 9, dtype=np.float64).reshape(3, 3)
        out = mp4._normalize_frame_to_uint8(frame)
        assert out.max() == 254

    def test_a_constant_integer_frame_becomes_mid_gray(self):
        frame = np.full((2, 2), 7, dtype=np.uint16)
        out = mp4._normalize_frame_to_uint8(frame)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, np.full((2, 2), 128))

    def test_a_constant_float_frame_becomes_black_not_mid_gray(self):
        """A float frame never takes the constant-image branch.

        ``np.issubdtype(frame.dtype, np.floating) or min_val < max_val``
        short-circuits on the dtype, so a constant float image is scaled by
        ``0 / 1e-10`` and comes out solid black while the integer frame
        above comes out mid-gray. Pinned as-is; reported as a defect.
        """
        frame = np.full((2, 2), 3.5, dtype=np.float32)
        out = mp4._normalize_frame_to_uint8(frame)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, np.zeros((2, 2), np.uint8))


# ================================================= sam2_mp4: TIFF loading
class TestLoadTiffStack:
    """``_load_tiff_stack`` normalises every supported TIFF layout."""

    def test_multipage_3d_stack(self, tmp_path):
        data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
        path = _write_tiff(tmp_path / "zyx.tif", data)
        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert is_grayscale is True
        assert frames.dtype == np.uint16
        np.testing.assert_array_equal(frames, data)

    def test_multipage_color_stack_is_not_grayscale(self, tmp_path):
        data = np.zeros((2, 4, 5, 3), dtype=np.uint8)
        data[0, ..., 0] = 255
        data[1, ..., 2] = 128
        path = _write_tiff(tmp_path / "tyxc.tif", data, photometric="rgb")
        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert is_grayscale is False
        np.testing.assert_array_equal(frames, data)

    def test_four_dimensional_stack_is_flattened(self, tmp_path):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        path = _write_tiff(tmp_path / "tzyx.tif", data)
        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert is_grayscale is True
        # (T, Z, Y, X) -> (T*Z, Y, X) in T-major order.
        np.testing.assert_array_equal(frames, data.reshape(6, 4, 5))
        np.testing.assert_array_equal(frames[3], data[1, 0])

    def test_single_page_grayscale_gains_a_leading_axis(self, tmp_path):
        data = np.arange(20, dtype=np.uint8).reshape(4, 5)
        path = _write_tiff(tmp_path / "yx.tif", data)
        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert is_grayscale is True
        assert frames.shape == (1, 4, 5)
        np.testing.assert_array_equal(frames[0], data)

    def test_single_page_rgb(self, tmp_path):
        data = np.zeros((4, 5, 3), dtype=np.uint8)
        data[..., 1] = np.arange(5, dtype=np.uint8)
        path = _write_tiff(tmp_path / "yxc.tif", data, photometric="rgb")
        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert is_grayscale is False
        assert frames.shape == (1, 4, 5, 3)
        np.testing.assert_array_equal(frames[0], data)

    def test_an_unreadable_file_falls_back_to_opencv(
        self, tmp_path, monkeypatch
    ):
        frames_in = [
            np.full((4, 5, 3), 10, np.uint8),
            np.full((4, 5, 3), 20, np.uint8),
        ]
        fake = _FakeCv2(capture=_FakeCapture(True, frames_in))
        monkeypatch.setattr(mp4, "cv2", fake)
        path = tmp_path / "not-a-tiff.tif"
        path.write_bytes(b"definitely not a tiff")

        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert is_grayscale is False
        # Every frame the capture yielded, in order, and nothing more.
        np.testing.assert_array_equal(frames, np.array(frames_in))
        assert fake.capture.released is True

    def test_the_opencv_fallback_calls_single_channel_frames_grayscale(
        self, tmp_path, monkeypatch
    ):
        frames_in = [np.full((4, 5, 1), 7, np.uint8)] * 2
        fake = _FakeCv2(capture=_FakeCapture(True, frames_in))
        monkeypatch.setattr(mp4, "cv2", fake)
        path = tmp_path / "not-a-tiff.tif"
        path.write_bytes(b"definitely not a tiff")

        frames, is_grayscale = mp4._load_tiff_stack(path)
        assert frames.shape == (2, 4, 5, 1)
        assert is_grayscale is True

    def test_an_unreadable_file_with_no_opencv_fallback_raises(
        self, tmp_path, monkeypatch
    ):
        fake = _FakeCv2(capture=_FakeCapture(False, []))
        monkeypatch.setattr(mp4, "cv2", fake)
        path = tmp_path / "broken.tif"
        path.write_bytes(b"nope")
        with pytest.raises(ValueError, match="either tifffile or OpenCV"):
            mp4._load_tiff_stack(path)

    def test_an_unsupported_stack_rank_reaches_the_fallback(
        self, tmp_path, monkeypatch
    ):
        # 5D is neither the 3D nor the 4D layout the loader knows about.
        fake = _FakeCv2(capture=_FakeCapture(False, []))
        monkeypatch.setattr(mp4, "cv2", fake)
        data = np.zeros((2, 2, 3, 4, 5), dtype=np.uint8)
        path = _write_tiff(tmp_path / "five.tif", data)
        with pytest.raises(ValueError, match="either tifffile or OpenCV") as e:
            mp4._load_tiff_stack(path)
        # The real reason is only reachable through __cause__: the loader
        # reports "could not open" for a file it opened fine (see report).
        assert isinstance(e.value.__cause__, ValueError)
        assert "Unsupported TIFF shape" in str(e.value.__cause__)

    def test_an_unsupported_single_frame_shape_reaches_the_fallback(
        self, tmp_path, monkeypatch
    ):
        # RGBA is neither (Y, X) nor (Y, X, 3): the ValueError raised in the
        # tifffile branch is swallowed by the module's own except clause and
        # the loader retries with OpenCV.
        fake = _FakeCv2(capture=_FakeCapture(False, []))
        monkeypatch.setattr(mp4, "cv2", fake)
        data = np.zeros((4, 5, 4), dtype=np.uint8)
        path = _write_tiff(
            tmp_path / "rgba.tif",
            data,
            photometric="rgb",
            extrasamples=["unassalpha"],
        )
        with pytest.raises(ValueError, match="either tifffile or OpenCV") as e:
            mp4._load_tiff_stack(path)
        assert "Unsupported frame shape" in str(e.value.__cause__)


# ================================================ sam2_mp4: OpenCV writer
class TestTifToMp4Opencv:
    """The fast path writes straight into a cv2.VideoWriter."""

    def test_grayscale_stack_is_expanded_to_bgr(self, tmp_path, fake_cv2):
        data = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6)
        src = _write_tiff(tmp_path / "stack.tif", data)
        out = tmp_path / "stack.mp4"
        result = mp4._tif_to_mp4_opencv(src, out, fps=12)

        assert result == str(out)
        writer = fake_cv2.writers[0]
        assert writer.path == str(out)
        assert writer.fourcc == "mp4v"
        assert writer.fps == 12
        assert writer.size == (6, 4)
        # Every source frame, in order, replicated across three channels --
        # writing frame 0 three times would pass a length-only assertion.
        assert len(writer.frames) == 3
        for written, source in zip(writer.frames, data, strict=True):
            np.testing.assert_array_equal(written, _bgr(source))
        assert writer.released is True

    def test_odd_dimensions_are_rounded_up_and_frames_resized(
        self, tmp_path, fake_cv2
    ):
        data = np.arange(2 * 5 * 7, dtype=np.uint8).reshape(2, 5, 7)
        src = _write_tiff(tmp_path / "odd.tif", data)
        mp4._tif_to_mp4_opencv(src, tmp_path / "odd.mp4")
        writer = fake_cv2.writers[0]
        assert writer.size == (8, 6)
        for written, source in zip(writer.frames, data, strict=True):
            assert written.shape == (6, 8, 3)
            # The picture is preserved; only the pad row/column is new.
            np.testing.assert_array_equal(written[:5, :7], _bgr(source))
            assert not written[5, :].any()
            assert not written[:, 7].any()

    def test_it_falls_through_to_the_second_codec(self, tmp_path, monkeypatch):
        fake = _FakeCv2(writers_open=(False, True))
        monkeypatch.setattr(mp4, "cv2", fake)
        data = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
        src = _write_tiff(tmp_path / "s.tif", data)
        mp4._tif_to_mp4_opencv(src, tmp_path / "s.mp4")
        assert [w.fourcc for w in fake.writers] == ["mp4v", "avc1"]
        # The rejected writer is released and never written to.
        assert fake.writers[0].released is True
        assert fake.writers[0].frames == []
        assert len(fake.writers[1].frames) == 2
        np.testing.assert_array_equal(fake.writers[1].frames[0], _bgr(data[0]))

    def test_no_working_codec_raises(self, tmp_path, monkeypatch):
        fake = _FakeCv2(writers_open=(False, False))
        monkeypatch.setattr(mp4, "cv2", fake)
        data = np.zeros((2, 4, 4), dtype=np.uint8)
        src = _write_tiff(tmp_path / "s.tif", data)
        with pytest.raises(RuntimeError, match="Could not initialize"):
            mp4._tif_to_mp4_opencv(src, tmp_path / "s.mp4")
        assert all(w.released for w in fake.writers)

    def test_a_color_stack_is_written_without_conversion(
        self, tmp_path, fake_cv2
    ):
        data = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        data[..., 1] = 200
        data[1, ..., 0] = 5
        src = _write_tiff(tmp_path / "rgb.tif", data, photometric="rgb")
        mp4._tif_to_mp4_opencv(src, tmp_path / "rgb.mp4")
        writer = fake_cv2.writers[0]
        # No cvtColor: the channels arrive exactly as stored.
        for written, source in zip(writer.frames, data, strict=True):
            np.testing.assert_array_equal(written, source)

    def test_a_uint16_stack_is_normalized_per_frame(self, tmp_path, fake_cv2):
        data = np.zeros((2, 4, 4), dtype=np.uint16)
        data[0] = 1000
        data[0, 0, 0] = 0
        data[1] = 30000
        data[1, 0, 0] = 0
        src = _write_tiff(tmp_path / "u16.tif", data)
        mp4._tif_to_mp4_opencv(src, tmp_path / "u16.mp4")
        # Each frame is stretched against its *own* min/max, so both very
        # different frames come out with the same 0..254 span.
        for written in fake_cv2.writers[0].frames:
            assert written.dtype == np.uint8
            assert (written.min(), written.max()) == (0, 254)


# ================================================ sam2_mp4: FFmpeg writer
class TestTifToMp4Ffmpeg:
    """The quality path shells out to ffmpeg over a PNG sequence."""

    def _stack(self, tmp_path, shape=(2, 4, 6)):
        data = np.arange(int(np.prod(shape)), dtype=np.uint8).reshape(shape)
        return _write_tiff(tmp_path / "src.tif", data)

    def test_a_missing_ffmpeg_raises_and_still_cleans_up(
        self, tmp_path, fake_cv2, fake_mkdtemp, monkeypatch
    ):
        monkeypatch.setattr(mp4.shutil, "which", lambda name: None)
        src = self._stack(tmp_path)
        with pytest.raises(RuntimeError, match="FFmpeg is required"):
            mp4._tif_to_mp4_ffmpeg(src, tmp_path / "o.mp4")
        assert len(fake_mkdtemp) == 1
        assert not fake_mkdtemp[0].exists()
        # The check happens before any frame is encoded.
        assert fake_cv2.imwrites == []

    def test_the_ffmpeg_command_line(
        self, tmp_path, fake_cv2, fake_mkdtemp, monkeypatch
    ):
        monkeypatch.setattr(
            mp4.shutil, "which", lambda name: "/usr/bin/ffmpeg"
        )
        runs = []
        monkeypatch.setattr(
            mp4.subprocess,
            "run",
            lambda cmd, **kw: runs.append((cmd, kw)),
        )
        data = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6)
        src = _write_tiff(tmp_path / "src.tif", data)
        out = tmp_path / "movie.mp4"
        result = mp4._tif_to_mp4_ffmpeg(src, out, fps=7, crf=21)

        assert result == str(out)
        temp_dir = fake_mkdtemp[0]
        cmd, kwargs = runs[0]
        assert cmd == [
            "ffmpeg",
            "-framerate",
            "7",
            "-i",
            str(temp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(out),
        ]
        assert kwargs == {"check": True, "capture_output": True}
        assert [p for p, _ in fake_cv2.imwrites] == [
            str(temp_dir / f"frame_{i:06d}.png") for i in range(3)
        ]
        # Frame i of the stack was written to frame_00000i.png, in order.
        for (_, written), source in zip(
            fake_cv2.imwrites, data, strict=True
        ):
            np.testing.assert_array_equal(written, _bgr(source))
        assert not temp_dir.exists()

    def test_keeping_the_temp_dir(
        self, tmp_path, fake_cv2, fake_mkdtemp, monkeypatch
    ):
        monkeypatch.setattr(mp4.shutil, "which", lambda name: "ffmpeg")
        monkeypatch.setattr(mp4.subprocess, "run", lambda cmd, **kw: None)
        src = self._stack(tmp_path)
        mp4._tif_to_mp4_ffmpeg(src, tmp_path / "o.mp4", cleanup_temp=False)
        temp_dir = fake_mkdtemp[0]
        assert temp_dir.exists()
        assert sorted(p.name for p in temp_dir.glob("frame_*.png")) == [
            "frame_000000.png",
            "frame_000001.png",
        ]

    def test_an_ffmpeg_failure_is_reraised_with_its_stderr(
        self, tmp_path, fake_cv2, fake_mkdtemp, monkeypatch, capsys
    ):
        monkeypatch.setattr(mp4.shutil, "which", lambda name: "ffmpeg")

        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(
                1, cmd, stderr=b"x264 exploded"
            )

        monkeypatch.setattr(mp4.subprocess, "run", boom)
        src = self._stack(tmp_path)
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            mp4._tif_to_mp4_ffmpeg(src, tmp_path / "o.mp4")
        assert excinfo.value.returncode == 1
        assert excinfo.value.cmd[0] == "ffmpeg"
        assert "x264 exploded" in capsys.readouterr().out
        assert not fake_mkdtemp[0].exists()

    def test_an_ffmpeg_failure_without_stderr(
        self, tmp_path, fake_cv2, fake_mkdtemp, monkeypatch, capsys
    ):
        monkeypatch.setattr(mp4.shutil, "which", lambda name: "ffmpeg")

        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(mp4.subprocess, "run", boom)
        src = self._stack(tmp_path)
        with pytest.raises(subprocess.CalledProcessError):
            mp4._tif_to_mp4_ffmpeg(src, tmp_path / "o.mp4")
        assert "FFmpeg MP4 creation error: Unknown error" in (
            capsys.readouterr().out
        )


# ================================================== sam2_mp4: entry point
class TestTifToMp4Entrypoint:
    """``tif_to_mp4`` picks a backend and derives the output path."""

    def test_default_uses_opencv_with_the_sibling_mp4_path(
        self, tmp_path, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            mp4,
            "_tif_to_mp4_opencv",
            lambda *args: calls.append(args) or "done",
        )
        monkeypatch.setattr(
            mp4,
            "_tif_to_mp4_ffmpeg",
            lambda *a, **k: pytest.fail("ffmpeg backend must not be used"),
        )
        result = mp4.tif_to_mp4(str(tmp_path / "movie.tif"), fps=25)
        assert result == "done"
        # The str path is turned into a Path and the suffix swapped.
        assert calls == [(tmp_path / "movie.tif", tmp_path / "movie.mp4", 25)]

    def test_use_ffmpeg_routes_to_the_ffmpeg_backend(
        self, tmp_path, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            mp4,
            "_tif_to_mp4_ffmpeg",
            lambda *args: calls.append(args) or "ff",
        )
        monkeypatch.setattr(
            mp4,
            "_tif_to_mp4_opencv",
            lambda *a, **k: pytest.fail("opencv backend must not be used"),
        )
        result = mp4.tif_to_mp4(
            tmp_path / "m.tiff",
            fps=5,
            use_ffmpeg=True,
            crf=30,
            cleanup_temp=False,
        )
        assert result == "ff"
        # .tiff -> .mp4, and crf/cleanup_temp arrive in that order.
        assert calls == [
            (tmp_path / "m.tiff", tmp_path / "m.mp4", 5, 30, False)
        ]

    def test_the_documented_defaults_reach_the_ffmpeg_backend(
        self, tmp_path, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            mp4,
            "_tif_to_mp4_ffmpeg",
            lambda *args: calls.append(args) or "ff",
        )
        mp4.tif_to_mp4(tmp_path / "m.tif", use_ffmpeg=True)
        assert calls == [(tmp_path / "m.tif", tmp_path / "m.mp4", 10, 17, True)]

    def test_end_to_end_with_a_real_tiff_and_a_fake_writer(
        self, tmp_path, fake_cv2
    ):
        data = np.random.default_rng(0).integers(
            0, 4000, size=(4, 6, 8), dtype=np.uint16
        )
        src = _write_tiff(tmp_path / "noise.tif", data)
        result = mp4.tif_to_mp4(src, fps=3)
        assert result == str(tmp_path / "noise.mp4")

        writer = fake_cv2.writers[0]
        assert writer.fps == 3
        assert len(writer.frames) == 4
        brightest = []
        for written, source in zip(writer.frames, data, strict=True):
            assert written.dtype == np.uint8
            assert written.shape == (6, 8, 3)
            # Grayscale expanded to BGR: all three channels are equal.
            np.testing.assert_array_equal(written[..., 0], written[..., 1])
            np.testing.assert_array_equal(written[..., 0], written[..., 2])
            # Each frame is stretched against its own min/max.
            assert (written.min(), written.max()) == (0, 254)
            # The brightest source pixel is the brightest written pixel.
            assert written[..., 0][np.unravel_index(
                np.argmax(source), source.shape
            )] == 254
            brightest.append(int(np.argmax(source)))
        # The four frames really are four different pictures.
        assert len({f[..., 0].tobytes() for f in writer.frames}) == 4

"""
Client for the SAM2 inference server.

SAM2 is installed in its own environment (``~/.napari-tmidas/envs/sam2-env``,
see :mod:`napari_tmidas.processing_functions.sam2_env_manager`) because the
napari environment carries no torch.  This module owns the long-lived
subprocess running :mod:`napari_tmidas._sam2_server` in that environment and
exposes a small synchronous request API to the widgets.

The worker is a process-wide singleton: loading the model costs a few seconds,
so it is started on first use and kept resident for the rest of the session,
after which each prompt costs milliseconds and each new image ~0.3 s.
"""

import atexit
import contextlib
import os
import pickle
import struct
import subprocess
import threading

import numpy as np

# The checkpoint the environment manager downloads, and its matching config.
_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class Sam2Unavailable(RuntimeError):
    """Raised when SAM2 cannot be used, with a message fit for the status bar."""


class Sam2Worker:
    """A resident SAM2 process, driven over a pickle-framed pipe."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._proc = None
        self._device = None
        self._io_lock = threading.Lock()

    # -- discovery ----------------------------------------------------
    @staticmethod
    def _manager():
        from napari_tmidas.processing_functions.sam2_env_manager import (
            manager,
        )

        return manager

    @classmethod
    def checkpoint_path(cls) -> str:
        return os.path.join(cls._manager().checkpoints_dir, _CHECKPOINT_NAME)

    @classmethod
    def availability(cls):
        """Return ``(ok, message)`` describing whether SAM2 can run.

        Checks only the filesystem, so it is cheap enough to call from a
        widget callback before committing to a click.
        """
        manager = cls._manager()
        if not manager.is_env_created():
            return False, (
                "SAM2 is not installed. Open the 'Batch Crop Anything' widget "
                "once to create the SAM2 environment "
                f"({manager.env_dir}), then try again."
            )
        if not os.path.exists(cls.checkpoint_path()):
            return False, (
                "The SAM2 model checkpoint is missing from "
                f"{manager.checkpoints_dir}. Open the 'Batch Crop Anything' "
                "widget once to download it (~850 MB)."
            )
        if not os.path.exists(_server_script()):
            return False, "The SAM2 server script is missing from the plugin."
        return True, "SAM2 is available."

    # -- lifecycle ----------------------------------------------------
    @classmethod
    def instance(cls) -> "Sam2Worker":
        """The shared worker, started on first use."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            cls._instance.start()
            return cls._instance

    @classmethod
    def shutdown_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    @property
    def device(self):
        return self._device

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn the server and load the model, unless already running."""
        if self.is_running():
            return
        ok, message = self.availability()
        if not ok:
            raise Sam2Unavailable(message)

        env_python = self._manager().get_env_python_path()
        # Run from the SAM2 checkout when it is there, so hydra resolves the
        # model config the same way the repo's own scripts do — but never
        # fail to launch over a missing directory (a legacy SAM2_PATH
        # install has no checkout), since the installed package can resolve
        # the config on its own.
        repo_dir = self._manager().sam2_repo_dir
        # Line buffering is irrelevant for the binary pipe but -u keeps the
        # server's stderr log flowing while the model loads.
        self._proc = subprocess.Popen(
            [env_python, "-u", _server_script()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit: torch/hydra chatter goes to the console
            cwd=repo_dir if os.path.isdir(repo_dir) else None,
        )
        atexit.register(self.stop)
        try:
            reply = self._request(
                {
                    "op": "load",
                    "checkpoint": self.checkpoint_path(),
                    "model_cfg": _MODEL_CFG,
                }
            )
        except Exception:
            self.stop()
            raise
        self._device = reply.get("device")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        self._device = None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # -- protocol -----------------------------------------------------
    def _request(self, payload):
        """Send one request and return the response dict.

        A dead server is turned into :class:`Sam2Unavailable` and the
        process is dropped, so the next call starts a fresh one rather than
        inheriting a broken pipe.
        """
        with self._io_lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise Sam2Unavailable(
                    "The SAM2 worker is not running (it exited; see the "
                    "console for its error output)."
                )
            blob = pickle.dumps(payload, protocol=4)
            try:
                proc.stdin.write(struct.pack(">I", len(blob)))
                proc.stdin.write(blob)
                proc.stdin.flush()
                reply = _read_reply(proc)
            except (BrokenPipeError, OSError) as exc:
                self.stop()
                raise Sam2Unavailable(
                    f"Lost contact with the SAM2 worker: {exc}"
                ) from exc
        if reply is None:
            self.stop()
            raise Sam2Unavailable(
                "The SAM2 worker exited while handling a request (see the "
                "console for its error output)."
            )
        if "error" in reply:
            raise Sam2Unavailable(f"SAM2 failed: {reply['error']}")
        return reply

    # -- inference ----------------------------------------------------
    def segment(self, image, point_coords, point_labels):
        """Candidate masks for a point prompt on *image* (uint8 H×W×3).

        *point_coords* are (x, y) pixel positions and *point_labels* 1 for
        "inside the object", 0 for "outside" — the negative points are what
        keep the mask off a touching neighbor.  Returns ``(masks, scores)``
        with one mask per candidate, ordered as SAM2 returns them.
        """
        reply = self._request(
            {
                "op": "segment",
                "image": np.ascontiguousarray(image),
                "point_coords": np.asarray(point_coords, dtype=np.float32),
                "point_labels": np.asarray(point_labels, dtype=np.int32),
            }
        )
        return reply["masks"], reply["scores"]

    def refine(self, index, point_coords, point_labels):
        """Sharpen candidate *index* by feeding it back as a mask prompt."""
        reply = self._request(
            {
                "op": "refine",
                "index": int(index),
                "point_coords": np.asarray(point_coords, dtype=np.float32),
                "point_labels": np.asarray(point_labels, dtype=np.int32),
            }
        )
        return reply["masks"][0], float(reply["scores"][0])


def _server_script() -> str:
    return os.path.join(os.path.dirname(__file__), "_sam2_server.py")


def _read_reply(proc):
    """Read one framed reply, or None if the server closed the pipe.

    The reads block. Polling the *file descriptor* with ``select`` would be
    wrong here: ``proc.stdout`` is buffered, so a short reply is drained into
    Python's buffer in one go and the fd then looks idle while the bytes are
    already in hand — the wait would spin until it timed out. A blocking read
    on a framed protocol has no such gap, and the failure that actually
    happens in practice (the server dying) closes the pipe, which surfaces
    immediately as a short read.
    """
    stream = proc.stdout

    def _read_exact(n):
        chunks = []
        while n > 0:
            chunk = stream.read(n)
            if not chunk:
                return None
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    header = _read_exact(4)
    if header is None:
        return None
    (size,) = struct.unpack(">I", header)
    body = _read_exact(size)
    if body is None:
        return None
    return pickle.loads(body)

"""
SAM2 inference server — runs *inside* the dedicated ``sam2-env``.

The napari environment has no torch, so SAM2 cannot be called in-process.
This script is launched by :mod:`napari_tmidas._sam2_worker` as a long-lived
subprocess using the sam2-env interpreter: the model is loaded once (~3 s) and
then answers prompt requests over stdin/stdout in a few milliseconds, which is
what makes a click-driven tool usable.

It deliberately imports nothing from ``napari_tmidas`` — that package is not
installed in sam2-env.

Protocol: 4-byte big-endian length, then a pickled dict, in both directions.
Requests carry an ``"op"``; every response is a dict with either the op's
result or an ``"error"`` key.
"""

import os
import pickle
import struct
import sys
import traceback

# torch, hydra and sam2 all chatter on stdout, which would corrupt the binary
# protocol. Keep a private duplicate of the real stdout for the protocol and
# point fd 1 at stderr, so any library print lands in the log instead.
_PROTO_OUT = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402


def _read_exact(stream, n):
    """Read exactly *n* bytes, or return None at a clean end of stream."""
    chunks = []
    while n > 0:
        chunk = stream.read(n)
        if not chunk:
            return None
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _send(payload):
    blob = pickle.dumps(payload, protocol=4)
    _PROTO_OUT.write(struct.pack(">I", len(blob)))
    _PROTO_OUT.write(blob)
    _PROTO_OUT.flush()


class _Session:
    """Holds the loaded predictor and the state of the current image."""

    def __init__(self):
        self.predictor = None
        self.device = None
        # low-resolution logits of the last multi-mask prediction, kept so a
        # follow-up "refine" can feed the chosen candidate back as a mask
        # prompt without re-encoding the image.
        self.low_res = None

    def load(self, checkpoint, model_cfg):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and (
            torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"
        self.device = device
        self.predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device=device)
        )
        return device

    def segment(self, image, point_coords, point_labels):
        """Encode *image* and return every candidate mask for the prompt."""
        self.predictor.set_image(image)
        masks, scores, low_res = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        self.low_res = low_res
        return {
            "masks": np.asarray(masks, dtype=bool),
            "scores": np.asarray(scores, dtype=np.float32),
        }

    def refine(self, index, point_coords, point_labels):
        """Re-run the decoder with candidate *index* fed back as a mask prompt.

        The image embedding is already cached by the predictor, so this is a
        decoder-only pass: a few milliseconds, and it sharpens the boundary
        that a single-shot point prompt tends to leave slightly inside the
        true edge.
        """
        if self.low_res is None:
            raise RuntimeError("refine called before segment")
        masks, scores, low_res = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=self.low_res[index : index + 1],
            multimask_output=False,
        )
        self.low_res = low_res
        return {
            "masks": np.asarray(masks, dtype=bool),
            "scores": np.asarray(scores, dtype=np.float32),
        }


def main():
    session = _Session()
    stdin = sys.stdin.buffer
    while True:
        header = _read_exact(stdin, 4)
        if header is None:
            break
        (size,) = struct.unpack(">I", header)
        request = pickle.loads(_read_exact(stdin, size))
        op = request.get("op")
        try:
            if op == "load":
                device = session.load(
                    request["checkpoint"], request["model_cfg"]
                )
                response = {"ok": True, "device": device}
            elif op == "segment":
                response = session.segment(
                    request["image"],
                    request.get("point_coords"),
                    request.get("point_labels"),
                )
            elif op == "refine":
                response = session.refine(
                    request["index"],
                    request.get("point_coords"),
                    request.get("point_labels"),
                )
            elif op == "ping":
                response = {"ok": True, "device": session.device}
            elif op == "shutdown":
                _send({"ok": True})
                break
            else:
                response = {"error": f"unknown op {op!r}"}
        # noqa: BLE001 — deliberate: any failure of a single request must
        # become an error reply, never kill a server holding a loaded model.
        except Exception as exc:  # noqa: BLE001
            response = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        _send(response)


if __name__ == "__main__":
    main()

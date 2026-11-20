"""Streaming wrapper for the pretrained Silero VAD, fed with the challenge's 20 ms / 320-sample frames.

Model: Silero VAD v6 (silero-vad 6.2.3 on PyPI, MIT licence), file models/silero_vad.onnx (2.3 MB), run with
onnxruntime on one CPU thread. torch is not needed. The file's SHA-256 is checked when it is loaded.

Buffering and delay (16 kHz). The model works on 512-sample (32 ms) windows plus 64 samples of context and keeps a
recurrent state. 20 ms frames are queued and the model runs whenever 512 new samples are available (5 frames =
3.125 windows), so the probability returned for a frame is the newest window's output: it covers 32 ms of audio and
can trail the newest sample by up to 31 ms. No future audio is used. reset() starts a new call.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


class SileroStream:
    name, licence = "silero-vad 6.2.3 (ONNX)", "MIT"
    WINDOW, CONTEXT = 512, 64
    _session = None                 # loaded once per process, shared by every instance (the state is per instance)

    def __init__(self):
        if SileroStream._session is None:
            import onnxruntime as ort
            if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
                raise RuntimeError(f"{MODEL} is not the pinned Silero VAD model")
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = opts.intra_op_num_threads = 1
            # no busy-wait spinning after each run: on a laptop it steals the CPU from the real-time server loop
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            SileroStream._session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
        self.session = SileroStream._session
        self.sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.prob = 0.0

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability after one more 20 ms frame (int16 or float samples in the int16 range)."""
        self.pending = np.concatenate([self.pending, np.asarray(frame, dtype=np.float32) / 32768.0])
        while len(self.pending) >= self.WINDOW:
            chunk, self.pending = self.pending[:self.WINDOW], self.pending[self.WINDOW:]
            x = np.concatenate([self.context, chunk])[None, :]
            out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
            self.context = x[0, -self.CONTEXT:]
            self.prob = float(out[0, 0])
        return self.prob


def load(name: str):
    if name != "silero":
        raise ValueError(f"unknown VAD {name!r}; this build ships Silero only")
    return SileroStream()

"""Optional streaming speech recognition for the word layer, run OFF the audio path.

on_frame must never wait for a recogniser. `StreamingTranscriber` copies each caller frame into a bounded queue; a
worker thread feeds the engine and publishes the latest text. `poll()` only reads a string. If the worker falls
behind, old audio is dropped (bounded memory) and results from before the last `reset()` are discarded (no stale
words from a previous burst or scenario).

Engines:
  * Vosk (offline, CPU):   pip install vosk ; download an Arabic model (e.g. vosk-model-ar-mgb2-0.4)
                           set TT_ASR=vosk and TT_VOSK_MODEL=/path/to/model before starting my_system.py
  * anything else: write a class with accept(pcm_bytes) -> None, text() -> str, restart() -> None.

Declare whichever engine and model you use in WRITEUP.md (the rules require it). Measure its delay with
`python tools/offline_eval.py --oracle-asr <ms>`: on dev the word layer helps up to roughly 700 ms of recogniser
delay and starts to cost beyond ~1 s.
"""
from __future__ import annotations

import os
import queue
import threading

import numpy as np


class StreamingTranscriber:
    MAX_QUEUE = 100                     # 2 s of audio; beyond that we drop the oldest frames

    def __init__(self, engine):
        self.engine = engine
        self.q: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE)
        self.gen = 0                    # bumped by reset(); results carry the generation they belong to
        self._text = ""
        self._lock = threading.Lock()
        self.dropped = 0
        self.alive = True
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    # --- called from on_frame: all O(1), never blocks
    def feed(self, frame: np.ndarray) -> None:
        item = (self.gen, frame.astype("<i2").tobytes())
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def reset(self) -> None:
        with self._lock:
            self.gen += 1
            self._text = ""
        try:
            self.q.put_nowait((self.gen, None))      # tells the worker to restart the engine
        except queue.Full:
            pass

    def poll(self) -> str:
        with self._lock:
            return self._text

    def close(self) -> None:
        self.alive = False

    # --- worker thread
    def _work(self):
        cur = -1
        while self.alive:
            try:
                gen, pcm = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self.gen:
                continue                              # audio from before the last reset: stale
            if pcm is None or gen != cur:
                try:
                    self.engine.restart()
                except Exception:
                    pass
                cur = gen
                if pcm is None:
                    continue
            try:
                self.engine.accept(pcm)
                text = self.engine.text()
            except Exception:
                continue
            with self._lock:
                if gen == self.gen:                   # still the current burst
                    self._text = text


_MODELS: dict = {}


def load_vosk_model(model_path: str):
    """Load once per process (call it before serving), share between connections."""
    if model_path not in _MODELS:
        from vosk import Model, SetLogLevel   # optional dependency
        SetLogLevel(-1)
        _MODELS[model_path] = Model(model_path)
    return _MODELS[model_path]


class VoskEngine:
    """Vosk / Kaldi streaming recogniser. Text = finished segments + the current partial hypothesis."""

    def __init__(self, model_path: str, sample_rate: int = 16000):
        import json

        from vosk import KaldiRecognizer
        self._json = json
        self.model = load_vosk_model(model_path)
        self.sr = sample_rate
        self.KaldiRecognizer = KaldiRecognizer
        self.restart()

    def restart(self):
        self.rec = self.KaldiRecognizer(self.model, self.sr)
        self.done = ""

    def accept(self, pcm: bytes):
        if self.rec.AcceptWaveform(pcm):
            self.done = (self.done + " " + self._json.loads(self.rec.Result()).get("text", "")).strip()
            self.partial = ""
        else:
            self.partial = self._json.loads(self.rec.PartialResult()).get("partial", "")

    def text(self) -> str:
        return (self.done + " " + getattr(self, "partial", "")).strip()


def from_environment():
    """Build the transcriber named by TT_ASR (or None). Loads the model BEFORE the server accepts connections."""
    kind = os.environ.get("TT_ASR", "").lower()
    if not kind:
        return None
    if kind == "vosk":
        path = os.environ.get("TT_VOSK_MODEL")
        if not path:
            raise SystemExit("TT_ASR=vosk needs TT_VOSK_MODEL=/path/to/vosk-model")
        return StreamingTranscriber(VoskEngine(path))
    raise SystemExit(f"unknown TT_ASR={kind!r} (supported: vosk)")

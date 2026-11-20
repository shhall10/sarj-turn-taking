"""Causal acoustic front-end for the caller channel. Everything here sees only audio that has already arrived.

Per 20 ms frame it produces a `FrameFeatures`:
    level_db      loudness of the caller channel AFTER removing the agent's echo
    snr_db        how far that is above the tracked background floor
    voicing       0..1, periodicity in the human pitch range (80-400 Hz); coughs, clicks, hum and noise stay low
    echo_share    0..1, how much of the raw caller energy the agent's echo explains
    speech        a cautious yes/no: loud enough above the floor AND voiced (or very loud)

Three ideas, each cheap enough for a 20 ms budget on one CPU core:
  1. Echo removal from the clean agent reference: we estimate the delay and gain of the agent's voice inside the caller
     microphone by cross-correlation, then subtract it. A speakerphone echo then no longer looks like the caller.
  2. A background floor that follows stationary noise (car, AC hum, street) up slowly and down fast, so steady noise
     never counts as speech, while a human onset jumps clearly above it.
  3. Voicing: speech is periodic in the pitch range; most noises are not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME = 320
FRAME_MS = 20


def db(x: float) -> float:
    return 10.0 * np.log10(x + 1e-12)


@dataclass
class FrameFeatures:
    level_db: float = -120.0
    raw_db: float = -120.0
    floor_db: float = -120.0
    snr_db: float = 0.0
    voicing: float = 0.0
    pitch_hz: float = 0.0
    echo_share: float = 0.0
    centroid_hz: float = 0.0
    hf_ratio: float = 0.0
    speech: bool = False

    @property
    def nasal(self) -> bool:
        """A hummed sound ("mmm", "اممم"): energy almost only below ~400 Hz, nothing above 1.5 kHz."""
        return self.speech and self.centroid_hz < 300 and self.hf_ratio < 0.01


class EchoCanceller:
    """Delay + gain echo model, re-estimated by cross-correlation while the agent talks.

    The agent channel is a clean reference. If the caller mic contains a delayed, attenuated copy of it, the normalised
    cross-correlation over the last ~0.6 s peaks at that delay. We only trust a peak that is clearly correlated.
    """

    HIST = SR                     # 1 s of agent history
    WIN = 9600                    # 0.6 s analysis window
    MAX_LAG = 8000                # up to 500 ms echo delay
    EVERY = 10                    # re-estimate every 200 ms

    def __init__(self):
        self.reset()

    def reset(self):
        self.agent = np.zeros(self.HIST + self.WIN, dtype=np.float64)
        self.caller = np.zeros(self.WIN, dtype=np.float64)
        self.delay = None
        self.gain = 0.0
        self.conf = 0.0
        self.n = 0

    def _estimate(self):
        c = self.caller
        a = self.agent
        if np.dot(c, c) < 1e3 * len(c) or np.dot(a[-self.WIN - self.MAX_LAG:], a[-self.WIN - self.MAX_LAG:]) < 1e4 * self.WIN:
            return
        seg = a[-(self.WIN + self.MAX_LAG):]                 # agent samples that could have produced c
        nfft = 1 << int(np.ceil(np.log2(len(seg) + len(c))))
        xc = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(c, nfft)), nfft)
        # xc[k] = sum_t seg[t+k] c[t]; lag (delay) d means c[t] ~ seg[t + MAX_LAG - d]
        k = np.arange(self.MAX_LAG + 1)
        vals = xc[self.MAX_LAG - k]
        # energy of each shifted agent window, via cumulative sums
        cs = np.concatenate([[0.0], np.cumsum(seg ** 2)])
        starts = self.MAX_LAG - k
        e_a = cs[starts + self.WIN] - cs[starts]
        norm = vals / np.sqrt(e_a * np.dot(c, c) + 1e-9)
        d = int(np.argmax(norm))
        r = float(norm[d])
        if r > 0.5:
            g = float(vals[d] / (e_a[d] + 1e-9))
            if self.delay is None or abs(d - self.delay) > 32:
                self.delay, self.gain = d, g
            else:
                self.delay, self.gain = d, 0.7 * self.gain + 0.3 * g
            self.conf = r
        else:
            self.conf *= 0.9

    def process(self, agent: np.ndarray, caller: np.ndarray) -> np.ndarray:
        a = agent.astype(np.float64)
        c = caller.astype(np.float64)
        self.agent = np.concatenate([self.agent[len(a):], a])
        self.caller = np.concatenate([self.caller[len(c):], c])
        self.n += 1
        if self.n % self.EVERY == 0:
            self._estimate()
        if self.delay is None or self.conf < 0.2:
            return c
        d = self.delay
        end = len(self.agent) - d
        ref = self.agent[end - len(c):end]
        return c - self.gain * ref


class NoiseFloor:
    """Minimum-statistics floor: min over the last ~2.4 s of smoothed frame levels, rising at most 4 dB/s."""

    def __init__(self, init_db: float = -75.0):
        self.blocks: list[float] = []
        self.cur = 999.0
        self.count = 0
        self.floor = init_db
        self.smooth = None

    def update(self, level_db: float) -> float:
        self.smooth = level_db if self.smooth is None else 0.6 * self.smooth + 0.4 * level_db
        self.cur = min(self.cur, self.smooth)
        self.count += 1
        if self.count == 10:              # 200 ms block
            first = not self.blocks
            self.blocks = (self.blocks + [self.cur])[-12:]
            self.cur, self.count = 999.0, 0
            target = min(self.blocks)
            if first or target < self.floor:
                self.floor = target                      # follow down fast
            else:
                self.floor = min(target, self.floor + 0.8)   # 0.8 dB per 200 ms
        return self.floor


def _bandpass_kernel(n=63, lo=250.0, hi=1200.0):
    t = np.arange(n) - (n - 1) / 2
    h = (2 * hi / SR * np.sinc(2 * hi / SR * t) - 2 * lo / SR * np.sinc(2 * lo / SR * t)) * np.hamming(n)
    return h


_BP = _bandpass_kernel()
_FREQS = np.fft.rfftfreq(FRAME, 1 / SR)
_SPEECH_BAND = (_FREQS >= 250) & (_FREQS <= 3800)


_ALL = (_FREQS > 80) & (_FREQS < 4000)
_HF = (_FREQS > 1500) & (_FREQS < 4000)


def spectral_shape(x: np.ndarray) -> tuple[float, float]:
    """(centroid in Hz, share of power above 1.5 kHz) over 80-4000 Hz."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    tot = X[_ALL].sum() + 1e-9
    return float((X[_ALL] * _FREQS[_ALL]).sum() / tot), float(X[_HF].sum() / tot)


def band_power(x: np.ndarray) -> float:
    """Mean power in the 250-3800 Hz speech band (rumble, engine and mains hum below 250 Hz are ignored)."""
    X = np.fft.rfft(x * np.hanning(len(x)))
    return float(np.sum(np.abs(X[_SPEECH_BAND]) ** 2) / (len(x) ** 2) * 2 * 2.67)


def voicing(x: np.ndarray) -> tuple[float, float]:
    """Normalised autocorrelation peak in the 80-400 Hz lag range of a 40 ms window; returns (strength, pitch)."""
    y = np.convolve(x, _BP, mode="same")
    y = y - y.mean()
    e = np.dot(y, y)
    if e < 1e-3:
        return 0.0, 0.0
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    lo, hi = SR // 400, SR // 80
    # unbiased normalisation so long lags are not penalised
    ac_n = ac[lo:hi] / (ac[0] + 1e-9) * (n / (n - np.arange(lo, hi)))
    i = int(np.argmax(ac_n))
    return float(np.clip(ac_n[i], 0.0, 1.0)), SR / (lo + i)


class Frontend:
    """Runs echo removal, floor tracking and voicing per frame. Call reset() at each new scenario."""

    SPEECH_SNR = 9.0
    VOICED = 0.45

    def __init__(self):
        self.echo = EchoCanceller()
        self.reset()

    def reset(self):
        self.echo.reset()
        self.floor = NoiseFloor()
        self.prev = np.zeros(FRAME)

    def process(self, agent: np.ndarray, caller: np.ndarray) -> FrameFeatures:
        raw = caller.astype(np.float64)
        res = self.echo.process(agent, caller)
        raw_e = band_power(raw)
        res_e = band_power(res)
        if res_e > raw_e:              # the echo model made things worse: it is wrong for this frame, ignore it
            res, res_e = raw, raw_e
        f = FrameFeatures()
        f.raw_db = db(raw_e / 32768 ** 2)
        f.level_db = db(res_e / 32768 ** 2)
        f.echo_share = float(np.clip(1 - res_e / (raw_e + 1e-9), 0, 1))
        f.floor_db = self.floor.update(f.level_db)
        f.snr_db = f.level_db - f.floor_db
        f.voicing, f.pitch_hz = voicing(np.concatenate([self.prev, res]))
        f.centroid_hz, f.hf_ratio = spectral_shape(res)
        self.prev = res
        self.last_res = res
        f.speech = f.snr_db > self.SPEECH_SNR and (f.voicing > self.VOICED or f.snr_db > 25)
        return f

"""Team system: speaker-aware acoustic turn-taking with an optional word layer.

    python -m ttbench.play                       # 8 example calls, then the score (needs `websockets`)
    python tools/offline_eval.py                 # all 24 dev calls, offline, exact timing (no websockets needed)

How it decides while the AGENT is talking (Core track):
  1. Front-end (turntaking/acoustic.py): remove the agent's own echo using the clean agent channel, track the
     background floor, measure voicing. Steady noise, hum, coughs and the echo stop looking like speech.
  2. Speaker enrolment: the caller always answers the greeting while the agent is silent. We learn how loud THIS
     caller's voice is in THIS call. Later, speech that is much quieter than the caller (a TV, someone else in the
     room, the agent's echo, a soft "mm-hm") is not treated as the caller claiming the floor. Relative level is
     gain-invariant: it works on a quiet phone line and a loud headset alike.
  3. Evidence accumulation: each caller-speech frame adds evidence weighted by how assertive it is (level relative
     to the enrolled voice). Short, soft bursts never reach the threshold; a real interruption gets there in a few
     hundred ms. Tonal sounds with a frozen pitch (car horn, beeps) are rejected.
  4. Word layer (turntaking/words.py), when a streaming recogniser is plugged in as `self.transcriber`:
     "لحظة", "لا لا", "اسمع", "بس…", a question -> STOP at once; words that are only acknowledgements
     ("إيه", "أها", "صح", "زين") -> raise the acoustic threshold so a long "إيه والله صادق، صح" does not stop us.
  5. RESUME: if we stopped on a burst that turns out to be short and soft, we carry on within the 500 ms hiccup window.

While the agent is SILENT (end-of-turn track): TAKE_TURN after a pause whose length adapts to what came before it
(fillers, connectors, digits being read out -> wait longer; a clear falling end -> answer sooner).
"""
from __future__ import annotations

import numpy as np

from ttbench.server import System, run
from turntaking import asr
from turntaking.acoustic import FRAME_MS, Frontend
from turntaking.words import classify

TEAM = "LSR"                 # your team name, shown on the scoreboard

# --- Core track parameters (all in ms or dB; chosen from feature analysis, see tools/analyze_events.py) ------------
REL_MIN_DB = -6.0           # speech quieter than (enrolled voice - 6 dB) is not the caller claiming the floor
REL_FULL_DB = -1.0          # at this relative level a frame counts fully; louder frames count more (up to W_MAX)
W_MAX = 2.0
STOP_EVIDENCE_MS = 350      # weighted caller speech needed for STOP (v2: 400 -> 350, the VAD gate removes the noise that 400 guarded against)
GAP_RESET_MS = 1200         # a silence this long ends the burst (v2: 900 -> 1200)
DECAY_PER_FRAME = 0.997     # evidence fades during short pauses inside a burst ("لا لا … لحظة") (v2: 0.985 -> 0.997)
ST_FRAMES = 15              # short-term loudness window: speech frames of the last 300 ms
PRE_ENROL_EVIDENCE_MS = 700 # before we know the caller's voice (very first utterance) be more careful
BG_ACTIVITY = 0.45          # speech in >45 % of the 4 s before a burst = a competing talker (TV, radio, room)
BG_MARGIN_DB = 6.0          # ... then the burst must be this much louder than that talker
BG_EXTRA_MS = 400           # ... and needs this much more evidence
VOICING_LO, VOICING_HI = 0.25, 0.55   # evidence weight ramps with voicing: coughs and clicks count little
BC_EXTRA_MS = 400           # extra evidence needed while the words so far are only acknowledgements ...
KEEP_WEIGHT = 0.3           # ... and their frames count this much (a long "إيه والله صادق، صح" never gets there)
WAIT_WORDS_MS = 300         # with a recogniser and no verdict yet: a little more evidence, so the words can arrive
WORDS_MIN_EVIDENCE_MS = 100 # words only count when the acoustic layer says it is the caller (not TV / side talk)
GRACE_MS = 120              # ignore the first frames of an agent utterance (onset transients of the echo path)
RESUME_QUIET_MS = 10 ** 9  # RESUME off: on dev it cost more (a real "خلاص خلاص" resumed over) than it saved
RESUME_MAX_BURST_MS = 700   # ... and only if the whole burst was shorter than this
TONAL_PITCH_SPREAD = 0.012

# --- Neural speech check (Silero VAD v6, MIT, ONNX on CPU; skipped automatically if onnxruntime is missing) -------
VAD_MODE = 1                # 0 = off (acoustic only), 1 = a frame counts as caller speech only if the VAD agrees, 2 = soft weight
VAD_GATE = 0.3              # mode 1: speech probability needed (plateau 0.25-0.35 on dev + xdev)
VAD_LO, VAD_HI = 0.2, 0.7   # mode 2: weight ramps between these probabilities  # pitch varying less than 1.2 % over 160 ms = a horn or a beep, not a voice

def _vad_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        from pathlib import Path
        return (Path(__file__).resolve().parent / "models" / "silero_vad.onnx").exists()
    except Exception:
        return False


if VAD_MODE and not _vad_available():
    # Without the neural VAD the v2 settings are not validated: fall back to the exact v1 (BASE 90.1) behaviour.
    VAD_MODE, STOP_EVIDENCE_MS, GAP_RESET_MS, DECAY_PER_FRAME = 0, 400, 900, 0.985
    print("TurnSense: onnxruntime / Silero model not found -> running the v1 BASE settings (acoustic only)", flush=True)

# --- End-of-turn parameters -------------------------------------------------------------------------------------
EOT_QUIET_MS = 800          # default pause before TAKE_TURN
EOT_HOLD_MS = 1800          # pause needed when the words say the caller is mid-thought
EOT_SHORT_TURN_MS = 1400    # a long answer that just stopped needs a slightly longer wait than a one-word answer
ELONG_MS = 260              # a final sound held this long on a steady pitch ("اممم…", "المغرب…") = still thinking
ELONG_SPREAD = 0.07         # steady = pitch within +-7 %
EOT_ELONG_HOLD_MS = 2400    # pause allowed after such a hesitation before we answer
EOT_RESUMED_PAUSE_MS = 500  # the caller came back after a pause this long: they think in chunks...
EOT_PATIENCE_STEP_MS = 250  # ... so each such comeback makes us this much more patient in this turn
EOT_PATIENCE_MAX_MS = 1500


class Enrollment:
    """Running estimate of the caller's own voice level, learned while the agent is silent."""

    def __init__(self):
        self.levels: list[float] = []
        self.level = None

    def add(self, level_db: float):
        self.levels.append(level_db)
        if len(self.levels) >= 8:
            arr = np.array(self.levels[-400:])
            self.level = float(np.median(arr))

    @property
    def ready(self) -> bool:
        return self.level is not None


class MySystem(System):
    def __init__(self):
        self.fe = Frontend()
        self.recent = []                 # linear speech-band power of recent speech frames (short-term loudness)
        self.history = []                # (speech?, short-term loudness) for the last 4 s while the agent talks
        # optional recogniser (TT_ASR=vosk ...): object with reset(), feed(int16 frame), poll() -> str
        self.transcriber = asr.from_environment()
        self.vad = None
        if VAD_MODE:
            try:
                import neural_vad
                self.vad = neural_vad.load("silero")
            except Exception as e:      # no onnxruntime / model: run acoustic-only, never fail the call
                print(f"neural VAD unavailable ({e}); acoustic only", flush=True)
        self.vprob = 1.0
        self.reset_call()

    # --- call / utterance state --------------------------------------------------------------------------------
    def reset_call(self):
        self.fe.reset()
        if getattr(self, 'vad', None) is not None:
            self.vad.reset()
        self.enroll = Enrollment()
        self.agent_talking = False
        self.agent_text = ""
        self.history = []
        self._reset_utterance()
        self._reset_eot()

    def _reset_utterance(self):
        self.stopped = False
        self.utt_ms = 0
        self._reset_burst()

    def _reset_burst(self):
        self.evidence = 0.0
        self.burst_ms = 0
        self.burst_speech_ms = 0
        self.burst_max_rel = -99.0
        self.gap_ms = 10 ** 6
        self.pitches: list[float] = []
        self.contrib: list[float] = []      # evidence added by the last few frames
        self.words_verdict = "NONE"
        self.stop_burst = False
        hist = getattr(self, "history", [])
        act = [st for sp, st in hist if sp and st is not None]
        self.bg_level = float(np.median(act)) if hist and len(act) / len(hist) > BG_ACTIVITY else None

    def _reset_eot(self):
        self.caller_spoke = False
        self.quiet_ms = 0
        self.turn_ms = 0
        self.eot_sent = False
        self.tail_pitch: list[float] = []   # pitch of the most recent voiced frames of the caller
        self.tail_nasal: list[bool] = []    # were the most recent caller frames hummed ("اممم")?
        self.hesitating = False
        self.comebacks = 0

    def on_scenario_start(self, context):
        self.reset_call()

    def on_agent_start(self, text, duration_ms):
        self.agent_talking, self.agent_text = True, text
        self._reset_utterance()

    def on_agent_end(self):
        self.agent_talking = False
        self._reset_eot()
        if self.transcriber is not None:
            try:
                self.transcriber.reset()      # the caller's answer starts now: fresh words for end-of-turn
            except Exception:
                pass

    # --- helpers -----------------------------------------------------------------------------------------------
    def _short_term(self, f) -> float | None:
        """Power-average loudness of the speech frames in the last 300 ms (frame levels jump by 20 dB within a word)."""
        self.recent.append(10 ** (f.level_db / 10) if f.speech else None)
        self.recent = self.recent[-ST_FRAMES:]
        vals = [v for v in self.recent if v is not None]
        return 10 * np.log10(np.mean(vals)) if vals else None

    def _rel(self, st: float | None) -> float:
        if st is None:
            return -99.0
        if not self.enroll.ready:
            return 0.0 if st > -35 else -10.0           # before enrolment: plain loudness
        return st - self.enroll.level

    def _tonal(self) -> bool:
        """Frozen pitch over the last 6 frames (octave jumps of the tracker folded away): a horn, a beep, a ringtone."""
        p = self.pitches[-6:]
        if len(p) < 6 or min(p) <= 0:
            return False
        d = np.abs(np.log2(np.array(p) / p[-1]))
        d = np.minimum(d % 1.0, 1.0 - d % 1.0)
        return bool(d.max() < np.log2(1 + TONAL_PITCH_SPREAD))

    def _elongated(self) -> bool:
        """Did the caller's last sound hold a steady pitch for a while (a filled pause or a lengthened syllable)?"""
        tail = [p for p in self.tail_pitch if p > 0]
        if len(tail) * FRAME_MS < ELONG_MS:
            return False
        ref = tail[-1]
        run, misses = 0, 0
        for p in reversed(tail):
            d = abs(np.log2(p / ref))
            d = min(d % 1.0, 1.0 - d % 1.0)        # octave errors of the pitch tracker are not pitch changes
            if d <= np.log2(1 + ELONG_SPREAD):
                run += 1
            else:
                misses += 1
                if misses > 2:
                    break
        return run * FRAME_MS >= ELONG_MS

    def _words(self, agent_talking: bool) -> str:
        if self.transcriber is None:
            return "NONE"
        try:
            return classify(self.transcriber.poll(), agent_talking)
        except Exception:                               # a recogniser problem must never break the call
            return "NONE"

    # --- every 20 ms -------------------------------------------------------------------------------------------
    def on_frame(self, agent, caller):
        f = self.fe.process(agent, caller)
        self._last = f
        if self.vad is not None and VAD_MODE:
            self.vprob = self.vad(self.fe.last_res)
        self._st = self._short_term(f)
        if self.transcriber is not None:
            try:
                self.transcriber.feed(caller)
            except Exception:
                pass
        if self.agent_talking:
            return self._while_agent_talks(f)
        return self._while_agent_silent(f)

    def _while_agent_talks(self, f):
        self.utt_ms += FRAME_MS
        rel = self._rel(self._st)
        self.history = (self.history + [(f.speech, self._st)])[-200:]
        is_caller = f.speech and rel >= REL_MIN_DB and self.utt_ms > GRACE_MS
        if is_caller and self.vad is not None and VAD_MODE == 1:
            is_caller = self.vprob >= VAD_GATE
        if is_caller and self.gap_ms < GAP_RESET_MS and self.bg_level is not None:
            is_caller = self._st >= self.bg_level + BG_MARGIN_DB
        if is_caller:
            if self.gap_ms >= GAP_RESET_MS:              # a new burst begins
                self._reset_burst()
                if self.transcriber is not None:
                    try:
                        self.transcriber.reset()
                    except Exception:
                        pass
            self.gap_ms = 0
            self.pitches.append(f.pitch_hz if f.voicing > 0.6 else 0.0)
            if self._tonal():
                # retract what the tone added before we recognised it
                self.evidence = max(0.0, self.evidence - sum(self.contrib[-6:]))
                self.contrib = self.contrib[:-6] + [0.0] * min(6, len(self.contrib))
            else:
                w = float(np.clip((rel - REL_MIN_DB) / (REL_FULL_DB - REL_MIN_DB), 0.0, W_MAX))
                w *= float(np.clip((f.voicing - VOICING_LO) / (VOICING_HI - VOICING_LO), 0.2, 1.0))
                if self.vad is not None and VAD_MODE == 2:
                    w *= float(np.clip((self.vprob - VAD_LO) / (VAD_HI - VAD_LO), 0.0, 1.0))
                if self.words_verdict == "KEEP":
                    w *= KEEP_WEIGHT
                self.evidence += w * FRAME_MS
                self.contrib = (self.contrib + [w * FRAME_MS])[-12:]
                self.burst_speech_ms += FRAME_MS
                self.burst_max_rel = max(self.burst_max_rel, rel)
        else:
            self.gap_ms += FRAME_MS
            self.evidence *= DECAY_PER_FRAME
            if not f.speech:
                self.pitches.append(0.0)
        if self.gap_ms < GAP_RESET_MS:
            self.burst_ms += FRAME_MS

        if self.burst_ms and self.transcriber is not None:
            self.words_verdict = self._words(True)     # latest reading of the burst's words (not sticky)

        if not self.stopped:
            need = STOP_EVIDENCE_MS if self.enroll.ready else PRE_ENROL_EVIDENCE_MS
            need += BC_EXTRA_MS if self.words_verdict == "KEEP" else 0
            if self.transcriber is not None and self.words_verdict == "NONE":
                need += WAIT_WORDS_MS
            need += BG_EXTRA_MS if self.bg_level is not None else 0
            if (self.words_verdict == "STOP" and self.burst_ms > 0 and self.gap_ms < GAP_RESET_MS
                    and self.evidence >= WORDS_MIN_EVIDENCE_MS):
                return self._stop("interruption", 0.9, "words")
            if self.evidence >= need:
                conf = float(np.clip(0.5 + (self.burst_max_rel + 3) / 12, 0.3, 0.95))
                return self._stop("interruption", conf, "acoustic")
            return None

        # we have stopped: take it back quickly if the burst was a short, soft one and the caller went quiet
        if (self.stop_burst and self.gap_ms >= RESUME_QUIET_MS and self.burst_speech_ms <= RESUME_MAX_BURST_MS
                and self.burst_max_rel < 0.0 and self.words_verdict != "STOP" and self.resume_ok):
            self.stopped = False
            self.stop_burst = False
            self._reset_burst()
            return {"decision": "RESUME", "label": "backchannel", "confidence": 0.5}
        if self.stop_burst and self.gap_ms >= RESUME_QUIET_MS:
            self.stop_burst = False                     # window passed: stay silent, the floor is theirs
        return None

    def _stop(self, label, conf, why):
        self.stopped = True
        self.stop_burst = True
        self.resume_ok = True
        self.stop_gap_ms = 0
        return {"decision": "STOP", "label": label, "confidence": round(conf, 2)}

    def _while_agent_silent(self, f):
        # the caller's own voice: learn its level (only clearly-voiced frames well above the floor)
        if f.speech and f.snr_db > 15 and self._st is not None:
            self.enroll.add(self._st)
        voice = f.speech and (not self.enroll.ready or f.level_db - self.enroll.level >= -14.0)
        if voice:
            self.tail_nasal = (self.tail_nasal + [f.nasal])[-15:]
        if voice and f.voicing > 0.45:
            self.tail_pitch = (self.tail_pitch + [f.pitch_hz])[-40:]
        elif voice:
            self.tail_pitch = []
        if voice:
            if self.caller_spoke and self.quiet_ms >= EOT_RESUMED_PAUSE_MS:
                self.comebacks += 1
            self.caller_spoke = True
            self.turn_ms += FRAME_MS + self.quiet_ms if self.quiet_ms < 400 else FRAME_MS
            self.quiet_ms = 0
            self.eot_sent = False
            return None
        if not self.caller_spoke or self.eot_sent:
            return None
        if self.quiet_ms == 0:
            hum = len(self.tail_nasal) >= 10 and np.mean(self.tail_nasal) >= 0.6
            self.hesitating = self._elongated() or hum
        self.quiet_ms += FRAME_MS
        need = EOT_QUIET_MS + (200 if self.turn_ms > EOT_SHORT_TURN_MS else 0)
        need = min(EOT_PATIENCE_MAX_MS, need + EOT_PATIENCE_STEP_MS * self.comebacks)
        if self.hesitating:
            need = EOT_ELONG_HOLD_MS
        if self._words(False) == "HOLD_EOT":
            need = max(need, EOT_HOLD_MS)
        need = min(need, max(EOT_ELONG_HOLD_MS, EOT_HOLD_MS))
        if self.quiet_ms >= need:
            self.eot_sent = True
            self.caller_spoke = False
            self.turn_ms = 0
            self.comebacks = 0
            return {"decision": "TAKE_TURN", "label": "end_of_turn"}
        return None


if __name__ == "__main__":
    print(f"TurnSense v2: neural VAD {'ON' if VAD_MODE else 'OFF (v1 base settings)'}", flush=True)
    MySystem()                      # warm-up: loads any recogniser model BEFORE we accept connections
    run(MySystem, TEAM, __file__)

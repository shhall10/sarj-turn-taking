"""Offline replay: same stream layout and official scorer as ttbench.harness, but frames go straight to the system
(no WebSocket, zero network delay). Decisions are stamped at the end of the frame that produced them.
    python offline_replay.py [--data data/dev] [--out runs/offline] [--set NAME=VALUE ...]
"""
import argparse, importlib, json, sys, time, types
from pathlib import Path

sys.modules.setdefault("websockets", types.ModuleType("websockets"))   # harness imports it; not used here
for m in ("websockets.asyncio", "websockets.asyncio.client", "websockets.exceptions", "websockets.asyncio.server"):
    mod = sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["websockets.asyncio.client"].connect = None
sys.modules["websockets.exceptions"].ConnectionClosed = Exception
sys.modules["websockets.asyncio.server"].serve = None

from ttbench.dataset import GAP_MS, lay_out_stream, load_split, sha256_file   # noqa: E402
from ttbench.harness import build_stream                                    # noqa: E402
from ttbench.protocol import FRAME_MS, FRAME_SAMPLES, FRAME_BYTES           # noqa: E402
from ttbench.score import score_run_dir                                     # noqa: E402
from ttbench import __version__                                             # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/dev", type=Path)
p.add_argument("--out", default="runs/offline", type=Path)
p.add_argument("--set", action="append", default=[])
p.add_argument("--gain", type=float, default=0.0)
p.add_argument("--json", action="store_true")
a = p.parse_args()

ms = importlib.import_module("my_system")
for kv in a.set:
    k, v = kv.split("=")
    setattr(ms, k, type(getattr(ms, k))(float(v)) if not isinstance(getattr(ms, k), int) else int(float(v)))

scen = lay_out_stream(load_split(a.data))
stream, controls = build_stream(scen)
if a.gain:
    import numpy as _np
    g = 10 ** (a.gain / 20)
    stream[:, 1] = _np.clip(stream[:, 1].astype(_np.float64) * g, -32768, 32767).astype(_np.int16)
n_frames = len(stream) // FRAME_SAMPLES
a.out.mkdir(parents=True, exist_ok=True)
log = open(a.out / "log.jsonl", "w", encoding="utf-8")
seq = 0
def L(d, msg, sm):
    global seq; seq += 1
    log.write(json.dumps({"seq": seq, "dir": d, "t_mono_ns": time.monotonic_ns(), "stream_ms": sm, "msg": msg}, ensure_ascii=False) + "\n")
hello = {"type": "hello", "team": ms.TEAM, "system_version": "offline", "tracks": ["core", "eot"]}
L("rx", hello, None); L("tx", {"type": "session_start"}, 0.0)
sysobj = ms.MySystem()
dec = 0
for n in range(n_frames):
    for msg in controls.get(n, []):
        L("tx", msg, n * FRAME_MS)
        t = msg["type"]
        if t == "scenario_start": sysobj.on_scenario_start(msg.get("context", {}))
        elif t == "agent_utterance_start": sysobj.on_agent_start(msg.get("text", ""), msg.get("duration_ms", 0))
        elif t == "agent_utterance_end": sysobj.on_agent_end()
    fr = stream[n * FRAME_SAMPLES:(n + 1) * FRAME_SAMPLES]
    out = sysobj.on_frame(fr[:, 0], fr[:, 1])
    if out is not None:
        for d in (out if isinstance(out, list) else [out]):
            m = {"decision": d} if isinstance(d, str) else dict(d)
            dec += 1
            L("rx", {"type": "decision", **m}, n * FRAME_MS + FRAME_MS + 1.0)
L("tx", {"type": "session_end"}, n_frames * FRAME_MS)
log.close()
meta = {"harness_version": __version__, "url": "offline", "speed": 1.0, "session_id": "dev", "official": False,
        "frames_planned": n_frames, "errors": [], "rtt_ms": [], "hello": hello, "completed": True,
        "frames_sent": n_frames, "max_send_lag_ms": 0, "decisions_received": dec, "frame_bytes": FRAME_BYTES,
        "gap_ms": GAP_MS, "scenarios": [{"scenario_id": s.scenario_id, "opaque_id": s.opaque_id, "start_ms": s.start_ms,
                                         "duration_ms": s.duration_ms} for s in scen],
        "dataset": {"path": str(a.data), "manifest_sha256": sha256_file(a.data / "manifest.json")}}
(a.out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
r = score_run_dir(a.data, a.out)
c = r["core"]
if a.json:
    print(json.dumps({"core": c["core_score"], "R": c["R"], "S": c["S"], "eot": r.get("eot", {}).get("eot_score")}))
    sys.exit(0)
print(json.dumps({k: c.get(k) for k in ("core_score", "R", "S", "A", "false_stops", "misses")}),
      "| EOT:", json.dumps({k: v for k, v in r.get("eot", {}).items() if not isinstance(v, (dict, list))}))

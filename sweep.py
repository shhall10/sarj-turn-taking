"""Evaluate a parameter setting on dev + xdev at 0 / -12 / +6 dB caller gain. Usage: python3 sweep.py NAME=VAL ..."""
import json, subprocess, sys, statistics
from concurrent.futures import ThreadPoolExecutor
sets = sys.argv[1:]
jobs = [(d, g) for d in ("data/dev", "data/extra/xdev") for g in (0, -12, 6)]
def run(j):
    d, g = j
    cmd = ["python3", "offline_replay.py", "--data", d, "--gain", str(g), "--json", "--out", f"runs/sw/{abs(hash((d,g,tuple(sets))))}"]
    for s in sets: cmd += ["--set", s]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return j, json.loads(out)
with ThreadPoolExecutor(6) as ex: res = dict(ex.map(run, jobs))
dev0, x0 = res[("data/dev", 0)], res[("data/extra/xdev", 0)]
worst = min(r["core"] for r in res.values())
obj = (dev0["core"] + x0["core"]) / 2
print(f"{' '.join(sets) or 'BASE':55} obj {obj:5.1f} | dev {dev0['core']:5.1f} (R{dev0['R']:.2f} S{dev0['S']:.2f} eot{dev0['eot']:.0f}) "
      f"| xdev {x0['core']:5.1f} (R{x0['R']:.2f} S{x0['S']:.2f}) | gains dev {res[('data/dev',-12)]['core']:.1f}/{res[('data/dev',6)]['core']:.1f} "
      f"xdev {res[('data/extra/xdev',-12)]['core']:.1f}/{res[('data/extra/xdev',6)]['core']:.1f} | worst {worst:.1f}")

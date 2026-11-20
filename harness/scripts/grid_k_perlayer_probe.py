"""Per-hop K against the best single K, offline and without training (row rve-grid-k-learned, stage b).

PAMS says a learned per-layer truncated scale beats a fixed one. The cheap falsifier is the per-layer choice made
by SEARCH instead of training: if a per-hop K picked greedily on the same frames cannot beat the best single K,
a learned one will not either. Each candidate is a real build (the grid is folded into the weights), so this is
64 builds per model, not a simulation.

Usage:
  RVE_TAIL_YUV=0 RVE_INT8_NO_EVENTS=1 python3 harness/scripts/grid_k_perlayer_probe.py --model <pth> --out-file x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402


def ft(p, h, w):
    raw = np.fromfile(p, dtype=np.uint8).reshape(h, w, 3)
    return torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0)[None].contiguous().cuda().half()


def build(model, calib, k: str) -> Int8Body:
    os.environ["RVE_INT8_GRID_K"] = str(k)
    b = Int8Body(model)
    with torch.no_grad():
        b.calibrate(calib)
        b._build()
    return b


def score(body, model, evals) -> float:
    tot = 0.0
    with torch.no_grad():
        for x in evals:
            ref = model(x).float()
            out = body(x).float()
            d = (out - ref).abs() * 255.0
            tot += float((d > 8).float().mean() * 100)
    return tot / len(evals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", default="/tmp/opencode/calib-frames")
    ap.add_argument("--frames", default="/tmp/opencode/eval3")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--ks", default="2,3,4")
    ap.add_argument("--out-file", required=True)
    a = ap.parse_args(argv)

    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    calib = [ft(p, a.height, a.width) for p in sorted(glob.glob(os.path.join(a.calib, "*.rgb")))]
    evals = [ft(p, a.height, a.width) for p in sorted(glob.glob(os.path.join(a.frames, "*.rgb")))]
    ks = [float(v) for v in a.ks.split(",")]

    single = {}
    for k in ks:
        b = build(model, calib, k)
        single[str(k)] = round(score(b, model, evals), 6)
        print(f"K={k:g}: {single[str(k)]:.5f}%", flush=True)
        del b
        torch.cuda.empty_cache()
    best_k = min(single, key=single.get)

    # greedy per-hop: start from the best single K, sweep one hop at a time
    cur = [float(best_k)] * len(build(model, calib, best_k)._launches)
    cur_score = single[best_k]
    for hop in range(len(cur)):
        trials = {}
        for k in ks:
            cand = list(cur)
            cand[hop] = k
            b = build(model, calib, ",".join(str(v) for v in cand))
            trials[str(k)] = score(b, model, evals)
            del b
            torch.cuda.empty_cache()
        pick = min(trials, key=trials.get)
        if trials[pick] < cur_score - 1e-9:
            cur[hop] = float(pick)
            cur_score = trials[pick]
        print(f"hop {hop:2d}: best {pick} ({trials[pick]:.5f}%)  running {cur_score:.5f}%", flush=True)

    res = {"model": os.path.basename(a.model), "scale": scale, "single": single, "best_single": best_k,
           "per_hop": cur, "per_hop_score": round(cur_score, 6),
           "gain_x": round(single[best_k] / cur_score, 3) if cur_score > 0 else None}
    print(f"\nbest single K={best_k}: {single[best_k]:.5f}%   per-hop: {cur_score:.5f}%   "
          f"gain {res['gain_x']}x")
    json.dump(res, open(a.out_file, "w"), indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Can the best grid clamp K be predicted from the WEIGHTS, without building and scoring candidates?

The ladder measured the best K for 22 models (`logs/grid-k/ladder.jsonl`): K=2 for some, K=3/4 for others, by
2.5-7x. The start-up pick costs about 3 s, and this asks whether that can be avoided.

Two facts set the bar for a predictor:
  * the cheap one already tried - the calibration grid's dynamic range - does NOT separate them (rescued spreads
    2.71-6.93 against damaged 4.25-4.55);
  * the two first-order terms both move the same way for every model (input error 2.7-3.9x better, weight error
    1.20-1.40x worse), so the sign is decided by how each model amplifies the perturbation, not by either term.

So this probe computes weight-only statistics (no forward pass) and prints them against the ladder's target, so
the question is answered by looking rather than by hoping. What it would take to pass: a statistic whose ordering
puts every model with "another K beats K=2 by >=2x" above every model without, i.e. a clean separation.

Usage: python scripts/grid_k_predict_probe.py [--ladder ...] [--inventory ...] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import read_shape  # noqa: E402

ROOT = "/home/jorj/Загрузки/Upscaling"


def find(name: str) -> str:
    for dirpath, _dirs, files in os.walk(ROOT):
        if name in files:
            return os.path.join(dirpath, name)
    raise SystemExit(f"not found: {name}")


def stats(path: str) -> dict:
    """Weight-only statistics, one set per model. No activations, no forward pass."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    shape = read_shape(path)
    convs = [k for k in sd if k.endswith(".weight") and getattr(sd[k], "ndim", 0) == 4]
    convs.sort(key=lambda s: int(s.split(".")[1]))
    body = convs[1:-1]                       # the inner convs; the head and the tail are not hops
    row_cv, row_spread, out_spread, cond = [], [], [], []
    for k in body:
        w = sd[k].float().reshape(sd[k].shape[0], -1)          # (out, in*k*k)
        row = w.abs().sum(dim=1)                               # per-output-channel energy
        col = w.abs().sum(dim=0)                               # per-input-position energy
        row_cv.append(float(row.std() / row.mean().clamp_min(1e-9)))
        row_spread.append(float(row.max() / row.median().clamp_min(1e-9)))
        out_spread.append(float(col.max() / col.median().clamp_min(1e-9)))
        sv = torch.linalg.svdvals(w)
        cond.append(float(sv[0] / sv[-1].clamp_min(1e-9)))
    med = statistics.median
    return {
        "feat": shape["feat"], "inner": shape["inner"], "scale": shape["scale"],
        "row_cv": round(med(row_cv), 3), "row_spread": round(med(row_spread), 3),
        "out_spread": round(med(out_spread), 3), "log_cond": round(float(torch.log10(torch.tensor(med(cond)))), 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="docs/wsp/verified/logs/grid-k/ladder.jsonl")
    ap.add_argument("--inventory", default="docs/wsp/verified/logs/model-fit/inventory.json")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    ladder = {}
    for line in open(args.ladder):
        if line.strip():
            row = json.loads(line)
            s = row["shares"]
            cand = {k: v for k, v in s.items() if k in ("K=2", "K=3", "K=4")}
            best_k = min(cand, key=cand.get)
            gain = cand["K=2"] / cand[best_k] if cand[best_k] else float("inf")
            ladder[row["file"]] = {"best_k": best_k, "gain_vs_k2": gain,
                                   "k2": cand["K=2"], "best": cand[best_k]}

    rows = []
    for name, target in ladder.items():
        try:
            st = stats(find(name))
        except Exception as error:                          # noqa: BLE001
            print(f"# skip {name}: {error}", file=sys.stderr)
            continue
        rows.append({"file": name, **st, **target})

    rows.sort(key=lambda r: -r["gain_vs_k2"])
    print(f"{'model':40s} {'bestK':>5s} {'gain':>6s} {'row_cv':>7s} {'rowSpr':>7s} {'outSpr':>7s} {'logCond':>8s}")
    for r in rows:
        print(f"{r['file'][:40]:40s} {r['best_k']:>5s} {r['gain_vs_k2']:6.2f} {r['row_cv']:7.3f} "
              f"{r['row_spread']:7.2f} {r['out_spread']:7.2f} {r['log_cond']:8.2f}")

    movers = [r for r in rows if r["gain_vs_k2"] >= 2.0]
    stayers = [r for r in rows if r["gain_vs_k2"] < 2.0]
    print(f"\nmovers (another K beats K=2 by >=2x): {len(movers)}; stayers: {len(stayers)}")
    for key in ("row_cv", "row_spread", "out_spread", "log_cond"):
        mv = sorted(r[key] for r in movers)
        sv = sorted(r[key] for r in stayers)
        overlap = (min(mv) < max(sv)) if mv and sv else False
        print(f"  {key:10s} movers {mv}  stayers {sv}  -> {'OVERLAP' if overlap else 'SEPARATED'}")
    print("\nscope: weight-only statistics; no forward pass, no calibration maxima. A predictor would have to "
          "separate the two groups above, and the ladder's own grid_spread already failed to.")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"rows": rows}, open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

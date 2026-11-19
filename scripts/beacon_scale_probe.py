"""Beacon's closed-form scale against the swept grid clamp K, on one model and held-out frames.

Beacon (arXiv:2508.20293) says the scale does not have to be searched: choose the integer codes on an unscaled
grid, then take the least-squares optimal scale, c = <Xw,Xq>/||Xq||^2 per channel. Our K sweep is a scale search -
it builds two candidates and picks one by rendering. This probe tests whether the closed form can replace it:

  policy A  K swept (K=2, 3, 4) - the shipped decision, measured here on the same frames
  policy B  per-hop, per-channel scales fitted in closed form to the fp16 activations, injected into the build
            (RVE_INT8_GRID_FIT), so no candidate is built to choose from

The statistic is the lane's own: share of pixels whose channel-max |delta| against the fp16 model exceeds 8
levels, on frames held out from calibration. The fit uses ONE frame; the measurement uses the others.

Usage:
  RVE_TAIL_YUV=0 RVE_INT8_NO_EVENTS=1 python3 scripts/beacon_scale_probe.py --frames /tmp/opencode/eval-frames \
      --calib /tmp/opencode/calib-frames --out-file /tmp/opencode/beacon.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402

MODEL = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"


def frame_tensor(path: str, h: int, w: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8).reshape(h, w, 3)
    return torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0)[None].contiguous().cuda().half()


def build(model, frames, k: str | None = None, fit: str | None = None) -> Int8Body:
    os.environ.pop("RVE_INT8_GRID_FIT", None)
    if k is not None:
        os.environ["RVE_INT8_GRID_K"] = str(k)
    if fit:
        os.environ["RVE_INT8_GRID_FIT"] = fit
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate(frames)
        body._build()
    return body


def share(body, model, x) -> dict:
    with torch.no_grad():
        ref = model(x).float()
        out = body(x).float()
    if out.shape != ref.shape:
        raise SystemExit(f"body output {tuple(out.shape)} against model {tuple(ref.shape)}")
    d = (out - ref).abs() * 255.0
    return {"share_gt8_pct": round(float((d > 8).float().mean() * 100), 6),
            "max_codes": round(float(d.max()), 3), "mean_codes": round(float(d.mean()), 5)}


def fit_grids(body, model, x) -> dict:
    acts = []
    hooks = [pl.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
             for pl in body.inner_prelus]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    grids = {}
    for i, (y, grid) in enumerate(zip(acts, body._hop_grids)):
        g = grid.reshape(1, -1, 1, 1).float()
        codes = torch.round(y.float() / g).clamp(-127, 127)
        num = (y.float() * codes).sum(dim=(0, 2, 3))
        den = (codes * codes).sum(dim=(0, 2, 3)).clamp_min(1.0)
        grids[f"grid_{i}"] = (num / den).cpu().numpy().astype(np.float32)
    return grids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calib", default="/tmp/opencode/calib-frames")
    ap.add_argument("--frames", default="/tmp/opencode/eval-frames")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out-file", default="/tmp/opencode/beacon.json")
    ap.add_argument("--fit-file", default="/tmp/opencode/beacon-fit.npz")
    a = ap.parse_args(argv)

    calib = sorted(glob.glob(os.path.join(a.calib, "*.rgb")))
    evals = sorted(glob.glob(os.path.join(a.frames, "*.rgb")))
    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    calib_t = [frame_tensor(p, a.height, a.width) for p in calib]
    eval_t = [(os.path.basename(p), frame_tensor(p, a.height, a.width)) for p in evals]

    res = {"model": os.path.basename(a.model), "scale": scale, "policies": {}, "frames": [n for n, _ in eval_t]}

    for k in ("2", "3", "4"):
        body = build(model, calib_t, k=k)
        rows = {name: share(body, model, x) for name, x in eval_t}
        res["policies"][f"K={k}"] = rows
        print(f"K={k}: " + "  ".join(f"{n} {r['share_gt8_pct']:.5f}%" for n, r in rows.items()), flush=True)
        del body
        torch.cuda.empty_cache()

    # Beacon: build once at the finest clamp, fit on ONE held-out frame, then rebuild with the fitted grids
    body = build(model, calib_t, k="4")
    grids = fit_grids(body, model, eval_t[0][1])
    np.savez(a.fit_file, **grids)
    del body
    torch.cuda.empty_cache()
    body = build(model, calib_t, fit=a.fit_file)
    rows = {name: share(body, model, x) for name, x in eval_t}
    res["policies"]["fitted"] = rows
    res["fit_frame"] = eval_t[0][0]
    print("fitted: " + "  ".join(f"{n} {r['share_gt8_pct']:.5f}%" for n, r in rows.items()), flush=True)

    best_k = min((p for p in res["policies"] if p != "fitted"),
                 key=lambda p: np.mean([r["share_gt8_pct"] for r in res["policies"][p].values()]))
    fitted_mean = float(np.mean([r["share_gt8_pct"] for r in res["policies"]["fitted"].values()]))
    best_mean = float(np.mean([r["share_gt8_pct"] for r in res["policies"][best_k].values()]))
    res["best_swept"] = best_k
    res["fitted_mean_pct"] = fitted_mean
    res["best_swept_mean_pct"] = best_mean
    res["verdict"] = "fitted wins" if fitted_mean <= best_mean else "sweep wins"
    print(f"\nbest swept {best_k}: {best_mean:.5f}%   fitted: {fitted_mean:.5f}%   -> {res['verdict']}")
    with open(a.out_file, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

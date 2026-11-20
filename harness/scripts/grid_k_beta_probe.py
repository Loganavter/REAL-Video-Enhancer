"""Predict the best grid clamp K from ARITHMETIC, not from a forward pass per candidate.

The earlier predictor attempt asked the wrong question: whether a weight statistic separates the 14 models where
some K beats K=2 from the 8 where it does not. The pick solves something else - `argmin_K` inside ONE model - and
the fact that the optimum is interior and model-dependent (2, 3 or 4) is evidence that the balance of the two
terms works; only the transfer coefficients failed.

So: `input_err(K)` and `weight_err(K)` are computed as pure functions of K from the calibration activations and
the weights (the grid `s[c] = max(s_in[c], s_t/K)`, the codes and the fold are all exact arithmetic), and the
prediction is

    argmin_K [ input_err(K) + beta * weight_err(K) ]

with `beta` calibrated across models. No forward pass per candidate - the only forward is the fp16 pass that
produces the activations, which the calibration does anyway.

Usage: python harness/scripts/grid_k_beta_probe.py [--frames 4] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from model_loader import load_compact  # noqa: E402

ROOT = "/home/jorj/Загрузки/Upscaling"
APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
sys.path.insert(0, os.path.join(APP, "backend", "src", "pytorch"))

KS = (1.0, 2.0, 3.0, 4.0)
BETAS = (1.0, 1.5, 2.0, 2.7, 3.0, 4.0, 6.0)


def find(name: str) -> str:
    for dirpath, _dirs, files in os.walk(ROOT):
        if name in files:
            return os.path.join(dirpath, name)
    raise SystemExit(f"not found: {name}")


def errors_for(model, frames, margin: float = 1.0):
    """input_err[K] and weight_err[K] over the body, from the activations and the weights alone.

    Two passes and no per-candidate forward: pass 1 collects the per-channel activation maxima per layer (which
    the calibration already computes), pass 2 walks each frame once and accumulates both errors for every K as
    pure arithmetic on the grid `s[c] = max(s_in[c], s_t/K)`.
    """
    convs = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
    head = list(model.body)[0]
    prels = [m for m in model.body if isinstance(m, torch.nn.PReLU)]
    rest = convs[1:] if isinstance(head, torch.nn.Conv2d) else convs
    inner = rest[:-1]

    # pass 1: per-channel maxima, exactly as the body's calibration measures them
    maxima = None
    with torch.no_grad():
        for f in frames:
            a = F.prelu(head(f.clamp(0.0, 1.0)), prels[0].weight)
            cur = [a.abs().amax(dim=(0, 2, 3))]
            for idx, conv in enumerate(inner):
                a = F.prelu(conv(a), prels[idx + 1].weight)
                cur.append(a.abs().amax(dim=(0, 2, 3)))
            maxima = cur if maxima is None else [torch.maximum(m, c) for m, c in zip(maxima, cur)]
    s_in_list = [(m / 127.0).float() for m in maxima[:-1]]      # the last one is never an input
    grids = []
    for s_in in s_in_list:
        s_t = float(s_in.max()) * margin
        grids.append({k: torch.maximum(s_in, torch.full_like(s_in, s_t / k)) for k in KS})

    # weight error: per layer, per K, on the folded weights (the domain the kernel multiplies codes in)
    w_num = {k: 0.0 for k in KS}
    w_ref = 0.0
    for idx, conv in enumerate(inner):
        w = conv.weight.detach().float()
        for k in KS:
            wf = w * grids[idx][k].view(1, -1, 1, 1)
            w_sc = wf.abs().amax(dim=(1, 2, 3)) / 127.0
            wq = torch.round(wf / w_sc.view(-1, 1, 1, 1)).clamp(-127, 127)
            w_num[k] += float(((wf - wq * w_sc.view(-1, 1, 1, 1)) ** 2).sum())
            if k == KS[0]:
                w_ref += float((wf ** 2).sum())

    # input error: one walk per frame, both errors accumulated where the activation is at hand
    in_num = {k: 0.0 for k in KS}
    in_ref = 0.0
    with torch.no_grad():
        for f in frames:
            a = F.prelu(head(f.clamp(0.0, 1.0)), prels[0].weight)
            for idx, conv in enumerate(inner):
                x = a.float()
                for k in KS:
                    g = grids[idx][k].view(1, -1, 1, 1)
                    dq = torch.round(x / g).clamp(-127, 127) * g
                    in_num[k] += float(((x - dq) ** 2).sum())
                in_ref += float((x ** 2).sum())
                a = F.prelu(conv(a), prels[idx + 1].weight)
    return ({k: (in_num[k] / max(in_ref, 1e-30)) ** 0.5 for k in KS},
            {k: (w_num[k] / max(w_ref, 1e-30)) ** 0.5 for k in KS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="docs/wsp/verified/logs/grid-k/ladder.jsonl")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    ladder = {}
    for line in open(args.ladder):
        if line.strip():
            row = json.loads(line)
            s = {k: v for k, v in row["shares"].items() if k.startswith("K=")}
            if s:
                ladder[row["file"]] = min(s, key=s.get)

    from src.pytorch.TorchUtils import TorchUtils
    from tail_int8_emulation import CALIB, H, W, decode_yuv10

    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream()
    frames = [tu.yuv420p10_to_tensor(decode_yuv10(i), st, torch.device("cuda", 0), torch.float16)
              for i in CALIB[: args.frames]]

    rows = []
    for name, target in ladder.items():
        try:
            model, _scale = load_compact(find(name))
            model = model.cuda().half()
            ie, we = errors_for(model, frames)
        except Exception as error:                     # noqa: BLE001
            print(f"# skip {name}: {type(error).__name__}: {str(error)[:70]}", file=sys.stderr)
            continue
        rows.append({"file": name, "target": target, "input_err": ie, "weight_err": we})
        print(f"{name[:38]:38s} target {target:>4s}  in " +
              " ".join(f"{ie[k]:.3f}" for k in KS) + "  w " + " ".join(f"{we[k]:.3f}" for k in KS),
              file=sys.stderr)

    print(f"\n{len(rows)} models; agreement of argmin[input + beta*weight] with the ladder:")
    best = (0, None)
    for beta in BETAS:
        hits = 0
        for r in rows:
            score = {k: r["input_err"][k] + beta * r["weight_err"][k] for k in KS}
            pred = f"K={min(score, key=score.get):g}"
            hits += int(pred == r["target"])
        print(f"  beta {beta:4.1f}: {hits:2d}/{len(rows)}")
        if hits > best[0]:
            best = (hits, beta)
    print(f"  best beta {best[1]} with {best[0]}/{len(rows)}")
    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"rows": rows, "betas": BETAS}, open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

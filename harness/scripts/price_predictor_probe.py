"""A curvature-priced predictor for K (row rve-price-curvature-predictor, stage a).

Predicting Quantization Price (arXiv:2609.28270): the correct pre-deployment score is the candidate's layer-output
error covariance weighted by the reference model's downstream curvature; reconstruction and diagonal scores fail to
rank configurations because they drop that curvature. Our linear sum of the two measured errors ranked the ladder
4/22. This probe builds the priced score and ranks K with it:

  price(K) = sum_l  mse_l(K) * c_l          mse_l = error of layer l's quantised output
                                            c_l   = mean squared change of the FINAL output per unit perturbation
                                                    of layer l (finite difference, one random direction)

If price ranks K better than 4/22, the class is alive; if not, the refutation is stronger than before.

Usage: RVE_TAIL_YUV=0 python3 harness/scripts/price_predictor_probe.py --model <pth> --out-file x.json
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", default="/tmp/opencode/calib-frames")
    ap.add_argument("--frames", default="/tmp/opencode/eval3")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--ks", default="2,3,4")
    ap.add_argument("--ladder", default="harness/docs/wsp/verified/logs/grid-k/ladder.jsonl")
    ap.add_argument("--out-file", required=True)
    a = ap.parse_args(argv)

    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    calib = [ft(p, a.height, a.width) for p in sorted(glob.glob(os.path.join(a.calib, "*.rgb")))]
    x = ft(sorted(glob.glob(os.path.join(a.frames, "*.rgb")))[0], a.height, a.width)
    ks = [float(v) for v in a.ks.split(",")]

    # the model's activations, once
    acts = []
    hooks = [pl.register_forward_hook(lambda m, i, o: acts.append(o.detach())) for pl in
             [m for m in model.body if isinstance(m, torch.nn.PReLU)]]
    with torch.no_grad():
        ref = model(x).float()
    for h in hooks:
        h.remove()

    # curvature per layer: one random direction, finite difference on the FINAL output
    curv = []
    for li, y in enumerate(acts):
        v = torch.randn_like(y.float()).half()
        v = (v / v.float().norm() * y.numel() ** 0.5 * 0.05 * float(y.float().std())).half()  # unit RMS, 5% spread
        prelu = [m for m in model.body if isinstance(m, torch.nn.PReLU)][li]

        def hook(_m, _i, out, v=v):
            return out + v

        hd = prelu.register_forward_hook(hook)
        with torch.no_grad():
            pert = model(x).float()
        hd.remove()
        den = float((v.float() ** 2).mean())
        curv.append(float(((pert - ref) ** 2).mean()) / max(den, 1e-12))

    price = {}
    for k in ks:
        os.environ["RVE_INT8_GRID_K"] = str(k)
        body = Int8Body(model)
        with torch.no_grad():
            body.calibrate(calib)
            body._build()
        mse = []
        for y, g in zip(acts, body._hop_grids):
            gg = g.reshape(1, -1, 1, 1).float()
            codes = torch.round(y.float() / gg).clamp(-127, 127)
            mse.append(float(((codes * gg - y.float()) ** 2).mean()))
        price[str(k)] = float(sum(m * c for m, c in zip(mse, curv)))
        del body
        torch.cuda.empty_cache()

    ladder = {}
    for line in open(a.ladder):
        r = json.loads(line)
        if r["file"] == os.path.basename(a.model):
            ladder = r["shares"]
    actual = {k: ladder.get(f"K={int(k)}") for k in ks}
    pred_best = min(price, key=price.get)
    act_best = min((k for k in actual if actual[k] is not None), key=lambda k: actual[k])
    act_best = str(float(act_best))
    res = {"model": os.path.basename(a.model), "scale": scale, "curvature": curv,
           "price": price, "predicted_best": pred_best,
           "ladder": actual, "ladder_best": act_best, "agree": pred_best == act_best}
    print(f"{os.path.basename(a.model)}: price {json.dumps({k: round(v, 8) for k, v in price.items()})}")
    print(f"  predicted K={pred_best}   ladder K={act_best}   agree={res['agree']}")
    json.dump(res, open(a.out_file, "w"), indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

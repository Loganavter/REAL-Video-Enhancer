"""Stage (a) of `rve-tail-as-int8-hop`: does folding the dequant scale into the tail's weights cost picture?

The shipped tail is two steps: the last int8 hop writes codes on a per-channel grid, those codes are
DEQUANTISED to fp16 (1.58 ms) and fed to a cuDNN fp16 convolution (2.33-2.40 ms). The fold is the identity the
body already uses BETWEEN hops (`Int8Body._build`: `weight * (s_in * margin)`), applied to the boundary:

    x_k = q_k * s_k           (the dequant)
    out_o = sum_k x_k * w[k,o] = sum_k q_k * (s_k * w[k,o]) = conv_int(q, w') * sigma_o

so `w' = s_k * w` folded offline, quantised per OUTPUT channel to int8, and the dequant pass disappears. The
output channels pad 12 -> 16 with zeros, which contributes nothing and therefore preserves the arithmetic.

Three arms on the four held-out frames (29, 44, 274, 400 - never calibrated on):
  today   the shipped path (body(x) itself)
  exact   the same fold with NO weight quantisation (the algebra alone; must be ~0 against `today`)
  int8w   the candidate: w' quantised per output channel to int8 (the actual error of the rewrite)

The emulation is the corpus's own recipe (`harness/scripts/int8_kernel_emulation.py`): integer accumulation is exact
in float32 here, since 64*9 terms of 127*127 is 9.3e6, far below 2^24.

Usage: python harness/scripts/tail_int8_emulation.py [--frames 29,44,274,400] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, SHIFT  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
FFMPEG = f"{APP}/bin/ffmpeg"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
CLIP = "/tmp/opencode/bd_violet20s.mkv"
W, H = 1920, 1080
CALIB = (20, 60, 100, 140, 180, 220, 260, 300)
HELD_OUT = (29, 44, 274, 400)


def decode_yuv10(frame: int) -> bytes:
    out = subprocess.run([FFMPEG, "-loglevel", "error", "-i", CLIP, "-vf", f"select=eq(n\\,{frame})",
                          "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return out[:W * H * 3]


@torch.no_grad()
def body_codes(body: Int8Body, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The last int8 hop's codes and their per-channel grid, the body's arithmetic in torch.

    Same recipe as `int8_kernel_emulation.emulate` (per-tensor grid per hop, symmetric +-127, requant into
    the next hop's grid), stopped at the last hop instead of dequantising.
    """
    s_act, outs = body._grids
    head = body.head_conv(x.clamp(0.0, 1.0).half())
    act = TF.prelu(head, body.head_prelu.weight)
    s_t1 = body._s_t1.view(1, -1, 1, 1)
    codes = torch.round(act / s_t1).clamp(-127, 127)
    s_in = body._s_t1
    for idx, conv in enumerate(body.inner):
        weight = conv.weight.detach().float()
        s_t = float(s_in.max())
        w_sc = weight.abs().amax(dim=(1, 2, 3)) / 127.0
        w_q = torch.round(weight / w_sc.view(-1, 1, 1, 1)).clamp(-127, 127)
        acc = TF.conv2d(codes.float(), w_q, None, padding=1)
        v = acc * (w_sc * s_t).view(1, -1, 1, 1) + conv.bias.detach().float().view(1, -1, 1, 1)
        v = TF.prelu(v, body.inner_prelus[idx].weight.detach().float())
        s_out = outs[idx].view(1, -1, 1, 1)
        q = torch.floor(v / s_out + 0.5).clamp(-127, 127)
        if idx + 1 < len(body.inner):
            s_next = outs[idx].max()
            m = torch.round(outs[idx] / s_next * (1 << SHIFT)).to(torch.float32)
            q = torch.clamp(torch.floor((q * m.view(1, -1, 1, 1) + (1 << (SHIFT - 1))) / (1 << SHIFT)),
                            -127, 127)
        codes = q
        s_in = outs[idx]
    return codes, outs[-1]


@torch.no_grad()
def finish(model, tail_out_fp32: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The rest of the caller: upsampler and the nearest residual, as Int8Body.__call__ runs them."""
    out = model.upsampler(tail_out_fp32.half())
    return (out + TF.interpolate(x.clamp(0.0, 1.0), scale_factor=model.upscale, mode="nearest")).float()


def stat(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    d = (a - b).abs().amax(dim=1) * 255.0
    return float((d > 8).float().mean() * 100), float(d.max()), float(d.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=",".join(str(f) for f in HELD_OUT))
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    stream = torch.cuda.Stream(device=dev)

    def frame(idx: int) -> torch.Tensor:
        return tu.yuv420p10_to_tensor(decode_yuv10(idx), stream, dev, torch.float16)

    model, scale, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()

    w_fp = body.tail_conv.weight.detach().float()          # (12, 64, 3, 3)
    b_fp = body.tail_conv.bias.detach().float()            # (12,)
    assert w_fp.shape[0] == 12 and w_fp.shape[1] == 64, w_fp.shape

    rows = []
    print(f"{'frame':>6} {'today-vs-model':>16} {'exact':>18} {'int8w':>18}")
    print(f"{'':>6} {'share/peak':>16} {'share/peak/dmean':>18} {'share/peak/dmean':>18}")
    for f in [int(s) for s in args.frames.split(",")]:
        x = frame(f)
        with torch.no_grad():
            today = body(x).float()                        # the shipped path, dequant + fp16 cuDNN tail
            model_out = model(x).float()
            codes, s_k = body_codes(body, x)
            # the fold, in float64-free float32: w'[o,k,i,j] = s_k * w[o,k,i,j]
            w_fold = w_fp * s_k.view(1, -1, 1, 1)
            acc = TF.conv2d(codes.float(), w_fold, None, padding=1) + b_fp.view(1, -1, 1, 1)
            exact = finish(model, acc, x)
            # the candidate: per-output-channel weight scales, int8 weights, zero-padded 12 -> 16 outputs
            sigma = w_fold.abs().amax(dim=(1, 2, 3)) / 127.0
            w_q = torch.round(w_fold / sigma.view(-1, 1, 1, 1)).clamp(-127, 127)
            w_q16 = torch.zeros((16, 64, 3, 3), device=dev, dtype=torch.float32)
            w_q16[:12] = w_q
            acc_q = TF.conv2d(codes.float(), w_q16, None, padding=1)[:, :12]
            int8w = finish(model, acc_q * sigma.view(1, -1, 1, 1) + b_fp.view(1, -1, 1, 1), x)
            # exactness of the integer accumulation: the largest |acc| must stay under 2^24
            max_acc = float(acc_q.abs().max())

        se, pe, me = stat(exact, today)
        sq, pq, mq = stat(int8w, today)
        sm, pm, _ = stat(today, model_out)
        rows.append(dict(frame=f, exact_share=se, exact_peak=pe, int8_share=sq, int8_peak=pq,
                         int8_mean=mq, today_share=sm, today_peak=pm, max_acc=max_acc))
        print(f"{f:>6} {sm:8.5f}/{pm:5.1f} {se:8.5f}/{pe:5.1f}/{me:6.3f} "
              f"{sq:8.5f}/{pq:5.1f}/{mq:6.3f}")

    worst_share = max(r["int8_share"] for r in rows)
    worst_peak = max(r["int8_peak"] for r in rows)
    exact_share = max(r["exact_share"] for r in rows)
    exact_peak = max(r["exact_peak"] for r in rows)
    print(f"\nworst of {len(rows)} held-out frames: int8w share {worst_share:.5f}% peak {worst_peak:.0f}; "
          f"exact share {exact_share:.5f}% peak {exact_peak:.0f}")
    print(f"bars: share {0.1}%, peak {20.0}; accumulation max |acc| "
          f"{max(r['max_acc'] for r in rows):.3e} against 2^24 = {1 << 24}")
    ok = worst_share <= 0.1 and worst_peak <= 20.0
    print(f"verdict: {'PASS - the fold is inside both bars, go to stage (b)' if ok else 'FAIL - stop here'}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "calib": list(CALIB),
                       "held_out": [int(s) for s in args.frames.split(",")],
                       "rows": rows, "worst_share": worst_share, "worst_peak": worst_peak,
                       "exact_worst_share": exact_share, "exact_worst_peak": exact_peak,
                       "pass": ok}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

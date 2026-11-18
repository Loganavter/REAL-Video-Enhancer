"""Do 6-bit codes between the hops survive the picture bar? Emulation only, no kernel.

The hop moves 265 MB per launch at 157 GB/s, so it sits at ~52% of BOTH roofs and the traffic one is what
stops it (row `rve-hop-load-schedule`). Packing the codes between hops to 6 bits cuts that traffic by 25%,
which matters only if the picture survives. Two ways to spend the two bits, because they cost different
things:

  rescale   keep the RANGE, coarsen the step: s' = s * 127/31, codes = round(x / s'), i.e. 4.1x the
            quantisation step and the same headroom - what a packed 6-bit code does
  clip      keep the STEP, lose the range: codes = clamp(round(x/s), -31, 31) - cheaper, and it clips
            whatever the calibration said would fit in +-127

The chain is the corpus's own emulation recipe (`scripts/int8_kernel_emulation.py`): per-hop per-tensor grid,
symmetric +-127, requant into the next hop's grid. Integer accumulation is exact in float32 (9.3e6 < 2^24).

Reported against three references, and the distinction matters:
  vs 8-bit body   what the 6-bit packing costs ON ITS OWN (a config change)
  vs model        what the body costs with 6 bits (the acceptance bar's own reference)

Usage: python scripts/sub8_codes_probe.py [--frames 29,44,274,400] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, SHIFT  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, HELD_OUT, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"


@torch.no_grad()
def chain(body: Int8Body, x: torch.Tensor, bits: int = 8, mode: str = "rescale"):
    """The body's sixteen hops in torch, optionally with `bits`-bit codes between them.

    Returns the final activation on the LAST hop's grid, dequantised to fp16 - i.e. exactly what the tail
    consumes - so the comparison measures the codes and nothing else.
    """
    s_act, outs = body._grids
    head = body.head_conv(x.clamp(0.0, 1.0).half())
    act = TF.prelu(head, body.head_prelu.weight)
    s_t1 = body._s_t1.view(1, -1, 1, 1)
    codes = torch.round(act / s_t1).clamp(-127, 127)
    s_in = body._s_t1
    lo = -(1 << (bits - 1)) + 1              # -31 for six bits, -127 for eight
    hi = (1 << (bits - 1)) - 1
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
        if bits < 8:
            if mode == "rescale":
                # same range, coarser step: the code keeps its VALUE and spends the two bits on range
                q = torch.clamp(torch.round(q * (hi / 127.0)), lo, hi)
            elif mode == "clip":
                q = torch.clamp(q, lo, hi)
            else:
                raise SystemExit(mode)
            # dequantise on the coarser grid for the next hop's convolution
            step = 127.0 / hi if mode == "rescale" else 1.0
            deq = q * step
        else:
            deq = q
        if idx + 1 < len(body.inner):
            s_next = outs[idx].max()
            m = torch.round(outs[idx] / s_next * (1 << SHIFT)).to(torch.float32)
            r = torch.clamp(torch.floor((deq * m.view(1, -1, 1, 1) + (1 << (SHIFT - 1))) / (1 << SHIFT)),
                            lo, hi)
            codes = r
        else:
            codes = deq
        s_in = outs[idx]
    # `codes` is already on the last hop's per-channel grid (the rescale variant scales inside the loop),
    # so the dequant is one multiply - adding `step` here would count the coarser step twice
    return (codes * outs[-1].view(1, -1, 1, 1)).half()


@torch.no_grad()
def image(model, act: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = model.upsampler(model.body[-1](act.half()))
    return (out + TF.interpolate(x.clamp(0.0, 1.0), scale_factor=model.upscale, mode="nearest")).float()


def stat(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs().amax(dim=1) * 255.0
    return float((d > 8).float().mean() * 100), float(d.max())


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

    model, _, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()

    rows = []
    print(f"{'frame':>6} {'8bit vs model':>15} {'6rescale vs 8b':>16} {'6rescale vs M':>15} "
          f"{'6clip vs 8b':>14} {'6clip vs M':>13}")
    for f in [int(s) for s in args.frames.split(",")]:
        x = frame(f)
        with torch.no_grad():
            ref_model = model(x).float()
            a8 = image(model, chain(body, x, 8), x)
            r6 = image(model, chain(body, x, 6, "rescale"), x)
            c6 = image(model, chain(body, x, 6, "clip"), x)
        s8, p8 = stat(a8, ref_model)
        sr8, pr8 = stat(r6, a8)
        sr_m, pr_m = stat(r6, ref_model)
        sc8, pc8 = stat(c6, a8)
        sc_m, pc_m = stat(c6, ref_model)
        rows.append(dict(frame=f, eight_share=s8, eight_peak=p8,
                         rescale_share=sr8, rescale_peak=pr8, rescale_model_share=sr_m,
                         rescale_model_peak=pr_m, clip_share=sc8, clip_peak=pc8,
                         clip_model_share=sc_m, clip_model_peak=pc_m))
        print(f"{f:>6} {s8:8.5f}/{p8:4.0f} {sr8:8.5f}/{pr8:5.1f} {sr_m:8.5f}/{pr_m:4.0f} "
              f"{sc8:7.5f}/{pc8:4.0f} {sc_m:7.5f}/{pc_m:4.0f}")

    worst_r8 = max(r["rescale_share"] for r in rows)
    worst_rp = max(r["rescale_peak"] for r in rows)
    worst_c8 = max(r["clip_share"] for r in rows)
    worst_cp = max(r["clip_peak"] for r in rows)
    worst_rm = max(r["rescale_model_share"] for r in rows)
    worst_cm = max(r["clip_model_share"] for r in rows)
    print(f"\nworst of {len(rows)} held-out frames:")
    print(f"  6-bit rescale vs 8-bit: share {worst_r8:.5f}%  peak {worst_rp:.1f}")
    print(f"  6-bit clip    vs 8-bit: share {worst_c8:.5f}%  peak {worst_cp:.1f}")
    print(f"  6-bit rescale vs MODEL: share {worst_rm:.5f}%   (8-bit body vs model "
          f"{max(r['eight_share'] for r in rows):.5f}%)")
    print(f"  6-bit clip    vs MODEL: share {worst_cm:.5f}%")
    ok = worst_rm <= 0.1 and worst_r8 <= 0.1
    print(f"bars: share 0.1%; verdict for the packed variant: "
          f"{'PASS - worth a kernel' if ok else 'FAIL - close before any kernel'}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "calib": list(CALIB),
                       "held_out": [int(s) for s in args.frames.split(",")], "rows": rows,
                       "worst": {"rescale_vs_8": worst_r8, "rescale_peak": worst_rp,
                                 "clip_vs_8": worst_c8, "clip_peak": worst_cp,
                                 "rescale_vs_model": worst_rm, "clip_vs_model": worst_cm},
                       "pass": ok}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

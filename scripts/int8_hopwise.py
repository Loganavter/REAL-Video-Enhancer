"""Where do the simulation and the kernel diverge? NOT YET ANSWERED - this comparison is invalid as written.

Two attempts on 2026-11-18 both produced nonsense (mean |d| 100-290% of the activation scale, an activation
scale of 0.021 at hop 16 against 2.7 elsewhere), and the reason is a design error, not a finding: the app's
body quantises SYMMETRICALLY on +-127 while `int8_scales_probe` quantises ASYMMETRICALLY over 0..255, so a
hop-by-hop diff measures the difference between two recipes rather than an implementation bug. The first
attempt also dequantised hop i with hop i's per-channel scale when the epilogue had already requanted the
buffer onto hop i+1's per-tensor grid.

The corrected experiment is in the queue row `rve-int8-sim-vs-kernel-2026-11-18`: emulate the KERNEL's own
recipe in torch (per-tensor symmetric +-127, per-channel weights folded, requant with shift 20), check that
the emulation matches the kernel to ~0, and only then vary one term at a time. Kept in the tree so the
mistake is not repeated rather than deleted.

Original intent:

The lane has a 6-14x disagreement between `scripts/int8_accuracy_sim.py` (0.065% on frame 29) and the app's
Triton body (0.42-0.91% over the clip). Every lever the simulation proposed was then worth ~0 in the kernel,
which means the simulation is not modelling what the kernel does. This script localises that:

  * runs the app's own `Int8Body` on one frame and dequantises the activation after hops 1, 2, 4, 8, 16;
  * runs the simulation's quantiser on the same frame and captures its activation at the same hops;
  * reports, per hop, mean |d| and the share above 8 levels between the two - in the SAME units (the
    activation, 64 channels at 1080p), so the number says where the two implementations part company.

Reading it: divergence at hop 1 = the head quantise or the weight folding; growing with hop index =
the requant chain (shift, rounding, clamp); divergence only at the end = accumulation.

Usage: python scripts/int8_hopwise.py --frame 169 [--body-calib 20,60,100,...]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as TF

REPO = "/home/jorj/Загрузки/projects/REAL-Video-Enhancer"
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, SHIFT, QB  # noqa: E402
from src.pytorch.Int8Kernels import body_prologue_kernel, dequantise_nhwc_to_nchw, \
    quantise_nchw_to_nhwc_prelu  # noqa: E402

W, H = 1920, 1080
HOPS = (1, 2, 4, 8, 16)


def load_frame(frames_dir: str, idx: int) -> torch.Tensor:
    raw = np.fromfile(os.path.join(frames_dir, f"v{idx}.raw"), dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(H, W, 3).astype(np.float32) / 255.0)
    return img.permute(2, 0, 1)[None].cuda().half()


@torch.no_grad()
def kernel_hops(body: Int8Body, x: torch.Tensor) -> dict[int, torch.Tensor]:
    """The body's own path, stopping after the selected hops to read the activation back."""
    h, w = x.shape[2], x.shape[1]
    # x arrives NCHW (1, 3, H, W); the body expects the same layout it gets from the app
    xin = x
    height, width = xin.shape[2], xin.shape[3]
    n_pix = height * width
    head = body.head_conv(xin.clamp(0.0, 1.0))
    buf = body._int8_buffers(xin, height, width)[0]
    qgrid = ((n_pix + QB - 1) // QB,)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, buf, n_pix,
                                       BLOCK=QB, C=body.feat, num_warps=4)
    grid = ((n_pix + body.block_m - 1) // body.block_m,)
    _, a, b = body._int8_buffers(xin, height, width)
    src, nxt = buf, a
    out: dict[int, torch.Tensor] = {}
    for i, launch in enumerate(body._launches, start=1):
        dst = nxt
        body_prologue_kernel[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"],
                                   launch["bias"], launch["slope"], launch["os"], dst,
                                   height, width, BLOCK_M=body.block_m, C=body.feat,
                                   num_warps=body.warps)
        if i in HOPS:
            # hop i's samples sit on hop i+1's per-tensor grid (the epilogue requants into it); the last
            # hop keeps its own per-channel grid. Using the wrong one here is what made the first run
            # report mean |d| 217 - a unit error, not a finding.
            _, outs = body._grids
            if i < len(body._launches):
                scale = outs[i].max().expand(body.feat).contiguous().float()
            else:
                scale = launch["os"].float()
            deq = torch.empty((1, body.feat, height, width), device=xin.device, dtype=torch.float16)
            dequantise_nhwc_to_nchw[((n_pix + 255) // 256,)](dst, scale, deq, n_pix,
                                                             BLOCK=256, C=body.feat)
            out[i] = deq.clone()
        src, nxt = dst, (b if dst is a else a)
    return out


@torch.no_grad()
def sim_hops(model, x: torch.Tensor, frames: list[torch.Tensor]) -> dict[int, torch.Tensor]:
    """The simulation's quantiser, per-tensor asymmetric over max, capturing the same hops."""
    from int8_scales_probe import calibrate, _span, _quant, quantise_weights
    model.load_state_dict(quantise_weights(model.state_dict()))
    cal = calibrate(model, frames)
    ins_range, ins_samp, out_range, out_samp = cal
    out = x.clamp(0.0, 1.0)
    res: dict[int, torch.Tensor] = {}
    hop = 0
    for name, module in model.body.named_children():
        if isinstance(module, torch.nn.Conv2d):
            lo, hi = _span(ins_range[f"body.{name}"], ins_samp[f"body.{name}"], False, 100.0)
            out = _quant(out, lo, hi, out.dtype)
            out = module(out)
        elif isinstance(module, torch.nn.PReLU):
            out = module(out)
            lo, hi = _span(out_range[f"body.{name}"], out_samp[f"body.{name}"], True, 100.0)
            out = _quant(out, lo, hi, out.dtype)
            hop += 1
            if hop in HOPS:
                res[hop] = out.clone()
        else:
            out = module(out)
    return res


def compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Relative, because activations are not 0..1 pictures: |d| as a share of the activation's own scale."""
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    ref = af.abs().mean().clamp_min(1e-6)
    return float(d.mean() / ref * 100), float(ref), float((d > 0.1 * ref).float().mean() * 100)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--frames", default="/var/tmp/rve-rem/frames")
    ap.add_argument("--calib", default="20,60,100,140,180,220,260,300")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    args = ap.parse_args(argv)

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    kframes = [load_frame(args.frames, int(i)) for i in args.calib.split(",")]
    test = load_frame(args.frames, args.frame)

    body = Int8Body(model)
    body.calibrate(kframes)
    body._build()
    kh = kernel_hops(body, test)
    sh = sim_hops(model, test, kframes)

    print(f"frame {args.frame}: kernel against the simulation, per hop (activation units, 255 = 1.0)")
    print(f"{'hop':>4}{'|d| % of scale':>15}{'act scale':>11}{'share >10%':>12}")
    for hop in HOPS:
        if hop in kh and hop in sh:
            m, mx, s = compare(kh[hop], sh[hop])
            print(f"{hop:>4}{m:>14.2f}%{mx:>11.4f}{s:>11.2f}%")
    print("\nDiverging at hop 1 = head quantise or weight folding; growing with the hop index = the requant "
          "chain (shift 20, rounding, clamp); only at the end = accumulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

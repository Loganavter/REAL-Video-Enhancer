"""The tail's last three steps in one kernel: conv + pixel shuffle + residual, and whether it is identical.

Arms (row `rve-tail-epilogue-shuffle-residual-2026-11-20`):
  today   int8 tail conv -> nn.PixelShuffle(2) -> + nearest(F.interpolate(x))
  conv    the int8 tail conv alone
  fused   tail_int8_shuffle_kernel: one launch writing the 4x-upsampled, residual-added output

The fused version performs the SAME two additions on the SAME 12 values in the same order - pixel_shuffle is
a permutation and the residual adds one LR pixel to all four subpixels - so byte-identity is available and is
checked here, element by element, not assumed.

Usage: python scripts/tail_shuffle_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB  # noqa: E402
from src.pytorch.Int8Kernels import tail_int8_kernel, tail_int8_shuffle_kernel  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, body_codes, decode_yuv10  # noqa: E402
from tail_hop_probe import COUT, COUT_PAD, CIN  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--warps", type=int, default=4)
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
        x = frame(args.frame)
        codes, s_k = body_codes(body, x)
    assert isinstance(model.upsampler, torch.nn.PixelShuffle), type(model.upsampler)
    assert model.upsampler.upscale_factor == 2, model.upsampler.upscale_factor

    w_fp = body.tail_conv.weight.detach().float()
    b_fp = body.tail_conv.bias.detach().float()
    w_fold = w_fp * s_k.view(1, -1, 1, 1)
    sigma = (w_fold.abs().amax(dim=(1, 2, 3)) / 127.0).clamp_min(1e-12)
    w_q = torch.round(w_fold / sigma.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
    taps = torch.zeros((9, CIN, COUT_PAD), device=dev, dtype=torch.int8)
    for i in range(3):
        for j in range(3):
            taps[i * 3 + j, :, :COUT] = w_q[:, :, i, j].t()
    sigma16 = torch.zeros(COUT_PAD, device=dev, dtype=torch.float32)
    sigma16[:COUT] = sigma
    bias16 = torch.zeros(COUT_PAD, device=dev, dtype=torch.float32)
    bias16[:COUT] = b_fp

    src = codes.permute(0, 2, 3, 1).contiguous().to(torch.int8)
    mid = torch.empty((1, H, W, COUT), device=dev, dtype=torch.float16)
    out = torch.empty((1, H * 2, W * 2, 3), device=dev, dtype=torch.float16)
    lr = x.clamp(0.0, 1.0).half()
    grid = ((H * W + args.block_m - 1) // args.block_m,)

    def conv_only():
        tail_int8_kernel[grid](src, taps, sigma16, bias16, mid, H, W, BLOCK_M=args.block_m, CIN=CIN,
                               COUT=COUT, COUT_PAD=COUT_PAD, num_warps=args.warps)
        return mid

    def today():
        l = conv_only().permute(0, 3, 1, 2)
        o = model.upsampler(l)
        return o + TF.interpolate(lr, scale_factor=2, mode="nearest")

    def fused():
        tail_int8_shuffle_kernel[grid](src, taps, sigma16, bias16, lr, out, H, W, BLOCK_M=args.block_m,
                                       CIN=CIN, COUT=COUT, COUT_PAD=COUT_PAD, SCALE=2, num_warps=args.warps)
        return out.permute(0, 3, 1, 2)

    with torch.no_grad():
        ref = today().clone().float()
        got = fused().clone().float()
        torch.cuda.synchronize()
        d = (ref - got).abs()
        n_diff = int((d > 0).sum())
        print(f"fused vs today: {n_diff} of {d.numel()} values differ "
              f"({100.0 * n_diff / d.numel():.6f}%), max |d| {float(d.max()):.6g}  "
              f"-> {'BYTE-IDENTICAL' if n_diff == 0 else 'NOT identical'}")

    def timeit(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / args.iters

    arms = {"today": today, "conv": conv_only, "fused": fused}
    draws = {k: [] for k in arms}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>9}" for k in arms) + "   (ms)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:9.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print("\nmedians of rounds 2+:")
    for k in arms:
        print(f"  {k:>6}: {med[k]:7.3f} ms")
    save = med["today"] - med["fused"]
    print(f"\nsaving: {save:+.3f} ms per frame ({100.0 * save / med['today']:+.1f}%) "
          f"- the shuffle + residual pass was {med['today'] - med['conv']:.3f} ms")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters, "block_m": args.block_m,
                       "warps": args.warps, "draws": draws, "median": med, "saving_ms": save,
                       "identity": {"differing": n_diff, "total": d.numel(),
                                    "max_abs": float(d.max())}}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

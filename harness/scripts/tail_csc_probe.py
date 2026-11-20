"""Does the tail writing yuv420p directly beat writing fp16 RGB and converting in the pack?

Arms, same tail weights and same codes:
  today   tail_int8_shuffle_kernel -> fp16 RGB (49.77 MB) -> tensor_to_yuv420p (the pack's conversion)
  fused   tail_int8_yuv_kernel -> yuv420p (12.44 MB) directly

Both payloads are compared byte for byte (the offline gate measured max 1 level), and both are timed with the
conversion inside the arm so the comparison covers the whole job, not just the kernel.

Usage: python harness/scripts/tail_csc_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time
import numpy as np
import torch, torch.nn.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend")); sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB  # noqa: E402
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu, tail_int8_shuffle_kernel, tail_int8_yuv_kernel  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, body_codes, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream(device=dev)
    def frame(i): return tu.yuv420p10_to_tensor(decode_yuv10(i), st, dev, torch.float16)
    model, _, _ = load(MODEL); model = model.eval().cuda().half()
    body = Int8Body(model); body.calibrate([frame(i) for i in CALIB]); body._build()
    x = frame(args.frame)
    codes, sk = body_codes(body, x)
    codes = codes.permute(0, 2, 3, 1).contiguous().to(torch.int8)
    lr = x.clamp(0.0, 1.0).half()
    n = H * W
    out_fp16 = torch.empty((1, 2 * H, 2 * W, 3), device=dev, dtype=torch.float16)
    out_yuv = torch.empty(6 * n, device=dev, dtype=torch.uint8)
    grid = ((n + 127) // 128,)

    def today():
        tail_int8_shuffle_kernel[grid](codes, body._tail_taps, body._tail_sigma, body._tail_bias, lr, out_fp16,
                                       H, W, BLOCK_M=128, CIN=64, COUT=12, COUT_PAD=16, SCALE=2, num_warps=4)
        return tu.tensor_to_yuv420p(out_fp16.permute(0, 3, 1, 2))

    def fused():
        tail_int8_yuv_kernel[grid](codes, body._tail_taps, body._tail_sigma, body._tail_bias, lr, out_yuv,
                                   H, W, BLOCK_M=128, CIN=64, COUT=12, COUT_PAD=16, SCALE=2, num_warps=4)
        return out_yuv

    a = torch.frombuffer(bytearray(today()), dtype=torch.uint8).cuda().to(torch.int16)
    b = fused().to(torch.int16)
    d = (a - b).abs()
    print(f"payload: {a.numel()} bytes, max |d| {int(d.max())}, mean {float(d.float().mean()):.4f}, "
          f"share >0 {(d>0).float().mean()*100:.3f}%, >1 {(d>1).float().mean()*100:.5f}%")

    def timeit(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / args.iters

    draws = {"today (rgb + pack csc)": [], "fused (yuv from tail)": []}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>24}" for k in draws))
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in (("today (rgb + pack csc)", today), ("fused (yuv from tail)", fused)):
            ms = timeit(fn); draws[k].append(ms); cells.append(f"{ms:24.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>24} {'ms':>8}")
    for k in med:
        print(f"{k:>24} {med[k]:8.3f}")
    g = med["today (rgb + pack csc)"] - med["fused (yuv from tail)"]
    print(f"\nsaving: {g:+.3f} ms per frame ({100.0*g/med['today (rgb + pack csc)']:+.1f}%)")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "clip": os.path.basename(CLIP), "frame": args.frame, "rounds": args.rounds,
                   "iters": args.iters, "draws": draws, "median": med, "saving_ms": g,
                   "max_code_diff": int(d.max()),
                   "share_gt1_pct": float((d > 1).float().mean() * 100)},
                  open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Amortise the per-launch constant over two frames: does M x 2 batching move the hop?

The measured account (int8-hop-occupancy.txt): one tap costs 0.59 ms, three cost the same, nine cost 1.55 - the
FIRST gather's round trip dominates, the SM has only 3.96 of 12 warps to hide it, and every launch-shaped knob is
either flat or worse. But that constant is paid per LAUNCH, not per pixel: a launch that carries two frames of
rows pays it once for twice the work.

  batch 1   16 launches per frame, 0.59 ms of constant each   (today)
  batch 2   16 launches per TWO frames - the constant halves per frame
  batch 3/4 as above

Cost: the pipeline must hold N frames, and the buffers grow N-fold (133 MB per frame of codes).

The kernel here is the shipped one with a batch index: rows are a flat run over B*H*W, and
`b = rows // (H*W)`, `h = (rows // W) % H` select the image. Codes must match the unbatched kernel exactly.

Usage: python harness/scripts/hop_batch_probe.py [--batches 1,2,3] [--rounds 3] [--iters 20] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB, SHIFT  # noqa: E402
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
GFLOP = 152.9


@triton.jit
def hop_batched(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
                H, W, B, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """The shipped kernel over a flat run of B*H*W rows."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    per_img = H * W
    b = rows // per_img
    rem = rows % per_img
    h = rem // W
    w = rem % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (b < B) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (b * per_img + yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        bb = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, bb, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    osc = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / osc + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8), mask=(b < B)[:, None])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="169,170,171,172")
    ap.add_argument("--batches", default="1,2,3")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
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
        heads = []
        for f in [int(s) for s in args.frames.split(",")]:
            x = frame(f)
            heads.append(body.head_conv(x.clamp(0.0, 1.0)))
    n_pix = H * W
    qgrid = ((n_pix + QB - 1) // QB,)
    codes = []
    for hd in heads:
        buf = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
        quantise_nchw_to_nhwc_prelu[qgrid](hd, body._s_t1, body._slope_half, buf, n_pix, BLOCK=QB,
                                          C=body.feat, num_warps=4)
        codes.append(buf)
    torch.cuda.synchronize()

    L = body._launches[0]
    bm, warps = body.block_m, body.warps

    def make(batch: int):
        src = torch.cat([codes[i] for i in range(batch)], dim=0) if batch > 1 else codes[0]
        dst = torch.empty_like(src)
        grid = ((batch * n_pix + bm - 1) // bm,)

        def run():
            hop_batched[grid](src, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                              H, W, batch, BLOCK_M=bm, C=body.feat, num_warps=warps)
            return dst
        return run, dst

    arms = {}
    for b in [int(s) for s in args.batches.split(",")]:
        fn, out = make(b)
        arms[f"batch {b}"] = (fn, out, b)

    # codes: each image inside the batched output must equal the unbatched result
    with torch.no_grad():
        ref_fn, ref = make(1)
        ref_fn()
        torch.cuda.synchronize()
        ref = ref.clone()
        for k, (fn, out, b) in arms.items():
            out.zero_()
            fn()
            torch.cuda.synchronize()
            bad = 0
            for i in range(b):
                seg = out.view(b, H * W, 64)[i].view(1, H, W, 64)
                d = (ref.view(1, H * W, 64).view(1, H, W, 64).to(torch.int16) -
                     seg.to(torch.int16)).abs()
                bad += int((d > 0).sum())
            print(f"codes {k}: {bad} of {b * ref.numel()} differ")

    def timeit(fn, b) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / args.iters / b      # per FRAME

    draws = {k: [] for k in arms}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>12}" for k in arms) + "   (ms per FRAME)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, (fn, _, b) in arms.items():
            ms = timeit(fn, b)
            draws[k].append(ms)
            cells.append(f"{ms:12.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>10} {'ms/frame':>10} {'16 hops':>9} {'vs batch 1':>12}")
    for k in arms:
        rel = 100.0 * (med["batch 1"] - med[k]) / med["batch 1"]
        print(f"{k:>10} {med[k]:10.3f} {16 * med[k]:9.2f} {rel:+11.1f}%")
    best = min(med, key=med.get)
    print(f"\nbest: {best} ({100.0 * (med['batch 1'] - med[best]) / med['batch 1']:+.1f}% per frame)")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frames": args.frames,
                       "rounds": args.rounds, "iters": args.iters,
                       "draws": draws, "median_per_frame": med,
                       "hops16_ms": {k: 16 * v for k, v in med.items()}, "best": best}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

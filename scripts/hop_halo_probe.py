"""Does a zero halo make the hop's nine gathers cheaper? The mask-free half of Im2win, in Triton.

Im2win (arXiv:2306.14316) argues direct convolution loses to GEMM-based convolution because of DISCONTINUOUS
memory access, and fixes it with window-ordered contiguous loads plus shared memory. The shared-memory half is
not expressible in Triton (see int8-hop-smem.txt); the contiguity half is: a (H+2, W+2, C) buffer with a
one-pixel zero border means the kernel needs no bounds mask at all - three compares and a select per tap per
row disappear, and every tap's row segment is a fixed-stride read from a padded array.

  today   (H, W, C) buffer, `inside = (yy>=0)&(yy<H)&(xx>=0)&(xx<W)` and a masked load
  halo    (H+2, W+2, C) buffer, zeros on the border, unmasked load

Codes must be identical: a halo changes no arithmetic. Any difference is an indexing bug.

Usage: python scripts/hop_halo_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB, SHIFT  # noqa: E402
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
GFLOP = 152.9
HP, WP = H + 2, W + 2


@triton.jit
def hop_masked(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
               H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        a = tl.load(x_ptr + ((yy * W + xx)[:, None] * C + lanes[None, :]), mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes, C)


@triton.jit
def hop_halo(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
             H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """The same kernel against a zero-haloed buffer: no mask, no bounds arithmetic."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        # +1 shifts into the padded array; no bounds test is needed because the border is zero
        src = ((h + 1 + dy) * (W + 2) + (w + 1 + dx))[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes, C)


@triton.jit
def _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes,
         C: tl.constexpr):
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    osc = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / osc + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
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
        head = body.head_conv(x.clamp(0.0, 1.0))

    n_pix = H * W
    qgrid = ((n_pix + QB - 1) // QB,)
    src_buf = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, src_buf, n_pix, BLOCK=QB,
                                       C=body.feat, num_warps=4)
    # the halo copy: zeros on a one-pixel border, codes in the interior. Cost is reported, not hidden.
    halo = torch.zeros((HP, WP, body.feat), device=dev, dtype=torch.int8)
    halo[1:H + 1, 1:W + 1, :] = src_buf.view(H, W, body.feat)
    dst = torch.empty_like(src_buf)
    torch.cuda.synchronize()

    big = halo.view(1, HP, WP, body.feat)
    L = body._launches[0]
    bm, warps = body.block_m, body.warps
    grid = ((n_pix + bm - 1) // bm,)

    def masked():
        hop_masked[grid](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                         H, W, BLOCK_M=bm, C=body.feat, num_warps=warps)
        return dst

    def haloed():
        hop_halo[grid](big, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                       H, W, BLOCK_M=bm, C=body.feat, num_warps=warps)
        return dst

    with torch.no_grad():
        ref = masked().clone()
        got = haloed().clone()
        torch.cuda.synchronize()
        d = (ref.to(torch.int16) - got.to(torch.int16)).abs()
        n = int((d > 0).sum())
        print(f"codes masked vs halo: {n} of {d.numel()} differ, max {int(d.max())} "
              f"-> {'IDENTICAL' if n == 0 else 'INDEXING BUG'}")

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

    def time_copy():
        def run():
            halo.zero_()
            halo[1:H + 1, 1:W + 1, :] = src_buf.view(H, W, body.feat)
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(10):
            run()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 10

    arms = {"masked (today)": masked, "halo (mask-free)": haloed}
    draws = {k: [] for k in arms}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>18}" for k in arms))
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:18.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    gain = 100.0 * (med["masked (today)"] - med["halo (mask-free)"]) / med["masked (today)"]
    copy_ms = time_copy()
    print(f"\n{'arm':>18} {'ms':>8} {'TOPS':>7}")
    for k in arms:
        print(f"{k:>18} {med[k]:8.3f} {GFLOP / med[k]:7.1f}")
    print(f"\nthe halo costs {gain:+.1f}% on the hop "
          f"({'worth it' if gain >= 5 else 'NOT worth it'}), and a separate "
          f"{copy_ms:.3f} ms pass to build it - {copy_ms * 16:.1f} ms per frame for sixteen hops")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters, "draws": draws, "median": med,
                       "gain_pct": gain, "halo_copy_ms": copy_ms, "codes_differ": n}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

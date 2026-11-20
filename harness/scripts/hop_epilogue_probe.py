"""How much of the hop is the EPILOGUE, and how much is one tap's latency?

Same kernel, three epilogues, all at the shipped BLOCK_M 128 / 4 warps:
  bare store      q = acc.to(int8)                       - no dequant, no PReLU, no requant
  dequant only    v = acc*w_scale + bias, cast
  shipped         the full epilogue (dequant, PReLU, floor, requant, clamp)

Together with `hop_roof_probe.py` (1 tap 1.073, 3 taps 1.072, 9 taps 1.551, store-only 0.485) this separates
three costs that were previously one number: the write, the epilogue, and the FIRST tap's round trip.

Usage: python harness/scripts/hop_epilogue_probe.py [--out-file x.json]
"""
import os, sys, statistics
import torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.getcwd(), "backend")); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from cut_channels_naive import load
from src.pytorch.Int8Body import Int8Body, QB, SHIFT
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu
from src.pytorch.TorchUtils import TorchUtils
from tail_int8_emulation import CALIB, H, W, decode_yuv10

@triton.jit
def hop(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
        H: tl.constexpr, W: tl.constexpr, BLOCK_M: tl.constexpr, C: tl.constexpr, EPI: tl.constexpr):
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
    if EPI == 0:
        q = acc.to(tl.int8)                       # bare store: no dequant, no PReLU, no requant
    elif EPI == 1:
        v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
        q = (v.to(tl.int8))                        # dequant only
    else:
        v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
        slope = tl.load(slope_ptr + lanes)[None, :]
        v = tl.where(v >= 0, v, v * slope)
        osc = tl.load(out_scale_ptr + lanes)[None, :]
        q = tl.floor(v / osc + 0.5)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
        q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        q = q.to(tl.int8)                          # the shipped epilogue
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q)

import argparse, json, time
_ap = argparse.ArgumentParser(); _ap.add_argument("--out-file", default=""); _args = _ap.parse_args()

dev = torch.device("cuda", 0)
tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
st = torch.cuda.Stream(device=dev)
def frame(i): return tu.yuv420p10_to_tensor(decode_yuv10(i), st, dev, torch.float16)
model, _, _ = load("/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth")
model = model.eval().cuda().half()
b = Int8Body(model); b.calibrate([frame(i) for i in CALIB]); b._build()
x = frame(169); head = b.head_conv(x.clamp(0.0, 1.0))
n = H * W; qg = ((n + QB - 1) // QB,)
src = torch.empty((1, H, W, 64), device=dev, dtype=torch.int8)
quantise_nchw_to_nhwc_prelu[qg](head, b._s_t1, b._slope_half, src, n, BLOCK=QB, C=64, num_warps=4)
dst = torch.empty_like(src); torch.cuda.synchronize()
L = b._launches[0]; bm = 128
grid = ((n + bm - 1) // bm,)
def run(epi): hop[grid](src, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst, H, W,
                        BLOCK_M=bm, C=64, EPI=epi, num_warps=4)
def t(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(20): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 20
arms = {"bare store": lambda: run(0), "dequant only": lambda: run(1), "shipped epilogue": lambda: run(2)}
res = {k: [] for k in arms}
for r in range(3):
    for k, fn in arms.items(): res[k].append(t(fn))
print(f"{'arm':>18} {'median r2+':>11}   draws")
for k, v in res.items():
    print(f"{k:>18} {statistics.median(v[1:]):11.3f}   {[round(x,3) for x in v]}")
base = statistics.median(res["bare store"][1:]); full = statistics.median(res["shipped epilogue"][1:])
print(f"\nthe epilogue costs {full - base:+.3f} ms of {full:.3f} ({100*(full-base)/full:.1f}%)")
if _args.out_file:
    os.makedirs(os.path.dirname(_args.out_file), exist_ok=True)
    json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "clip": "bd_violet20s.mkv",
               "frame": 169, "draws": res, "median": {k: statistics.median(v[1:]) for k, v in res.items()},
               "epilogue_ms": full - base}, open(_args.out_file, "w"), indent=1)
    print(f"# -> {_args.out_file}", file=sys.stderr)

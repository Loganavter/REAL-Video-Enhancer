"""A/A control for the epilogue probe: the kernel at HEAD and the current kernel with FOLD=0 must agree.

The row's rewrite added a compile-time branch to `body_prologue_kernel`. A branch that is compiled away
should leave the shipped path untouched, and "should" is not evidence: this loads the HEAD source as a
separate module and compares int8 codes on the same input, hop by hop.

Expected: 0 differing codes on every hop. A non-zero count means the specialization changed the shipped
arithmetic and the FOLD knob is not neutral.

Usage: python scripts/epilogue_control.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from hop_epilogue_probe import CALIB, MODEL, decode_yuv10  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB, SHIFT  # noqa: E402
from src.pytorch import Int8Kernels as cur  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402

HEAD = "/tmp/opencode/epilogue-head/Int8Kernels_head.py"


def load_head():
    spec = importlib.util.spec_from_file_location("Int8Kernels_head", HEAD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    head_mod = load_head()
    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=1920, height=1080, device_type="cuda", hdr_mode=True, gpu_id=0,
                    high_bitdepth=True)
    stream = torch.cuda.Stream(device=dev)
    clip = "/tmp/opencode/bd_violet20s.mkv"

    def frame(idx):
        return tu.yuv420p10_to_tensor(decode_yuv10(clip, idx, 1920, 1080), stream, dev, torch.float16)

    model, _, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()
        x = frame(169)
        hd = body.head_conv(x.clamp(0.0, 1.0))
    n_pix = 1080 * 1920
    qgrid = ((n_pix + QB - 1) // QB,)
    bufs = [torch.empty((1, 1080, 1920, 64), device=dev, dtype=torch.int8) for _ in range(3)]
    if hd.is_contiguous(memory_format=torch.channels_last):
        cur.quantise_nhwc_prelu[qgrid](hd, body._s_t1, body._slope_half, bufs[0], n_pix, BLOCK=QB,
                                      C=64, num_warps=4)
    else:
        cur.quantise_nchw_to_nhwc_prelu[qgrid](hd, body._s_t1, body._slope_half, bufs[0], n_pix,
                                              BLOCK=QB, C=64, num_warps=4)
    torch.cuda.synchronize()

    grid = ((n_pix + body.block_m - 1) // body.block_m,)

    def run(mod, fold, src, dst, L):
        mod.body_prologue_kernel[grid](src, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"],
                                       L["os"], dst, 1080, 1920, BLOCK_M=body.block_m, C=64,
                                       num_warps=body.warps, **({"FOLD": fold} if fold is not None else {}))

    # A/A: HEAD vs current FOLD=0, hop 1
    a, b = bufs[0], bufs[1]
    run(head_mod, None, a, b, body._launches[0])
    run(cur, 0, a, bufs[2], body._launches[0])
    torch.cuda.synchronize()
    d = (b.to(torch.int16) - bufs[2].to(torch.int16)).abs()
    n = int((d > 0).sum())
    print(f"A/A hop 1 (HEAD vs current FOLD=0): {n} of {d.numel()} codes differ, max {int(d.max())}")

    # the whole chain: both kernels always read the SAME input, which is the HEAD chain's output, so a
    # difference can only come from the epilogue and never from a different input
    x_in = bufs[0].clone()
    y_head = torch.empty_like(x_in)
    y_cur = torch.empty_like(x_in)
    worst, total = 0, 0
    for L in body._launches:
        run(head_mod, None, x_in, y_head, L)
        run(cur, 0, x_in, y_cur, L)
        torch.cuda.synchronize()
        dd = (y_head.to(torch.int16) - y_cur.to(torch.int16)).abs()
        worst = max(worst, int(dd.max()))
        total += int((dd > 0).sum())
        x_in = y_head.clone()
    print(f"A/A over sixteen hops: {total} differing codes in total, worst per-hop max {worst}")
    return 0 if n == 0 and total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

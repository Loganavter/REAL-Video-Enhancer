"""Compile and launch the two minimal hop data-movement kernels (`hop_staging_sass.cu`) for counting.

Why separate minimal kernels instead of the real one: the falsifier the staging family needed is "how many
load/store INSTRUCTIONS does staging add", and Triton has no explicit shared memory - a staged variant cannot
be written in it at all, which is the whole reason that route means CUDA. So these two kernels move the same
data (C=64 int8 codes, 3x3 taps, 16-byte vectors) with today's nine direct reads and with a staged 3-row band;
the arithmetic is a dp4a accumulate in both, so only the load/store class is comparable - and that is exactly
the resource the claim is about.

Usage:
  <python> scripts/hop_staging_launch.py --compile-only
  <python> scripts/hop_staging_launch.py --iters 1
  <ncu> --metrics smsp__inst_executed_op_ldg.sum,smsp__inst_executed_op_lds.sum,\
smsp__inst_executed_op_sts.sum <python> scripts/hop_staging_launch.py --iters 1
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess

NVCC = "/opt/cuda/bin/nvcc"
CUDA = ctypes.CDLL("libcuda.so.1")
CUDA.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "scripts", "hop_staging_sass.cu")
CUBIN = "/tmp/opencode/sass/staged.cubin"
TILE = 128


def check(res: int, what: str) -> None:
    if res != 0:
        raise RuntimeError(f"{what} failed: {res}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(CUBIN), exist_ok=True)
    subprocess.run([NVCC, "-arch=sm_89", "-O3", "-cubin", "-o", CUBIN, SRC], check=True)
    print(f"# compiled {os.path.basename(SRC)} -> {CUBIN}")
    if args.compile_only:
        return 0

    import torch
    torch.zeros(1, device="cuda")              # establish the context, then share it
    check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(CUDA.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")

    blob = open(CUBIN, "rb").read()
    module = ctypes.c_void_p()
    check(CUDA.cuModuleLoadData(ctypes.byref(module), blob), "cuModuleLoadData")

    h, w, c = args.h, args.w, 64
    x = torch.zeros((h + 2) * (w + 2) * c, dtype=torch.int8, device="cuda")
    wt = torch.zeros(9 * c * c, dtype=torch.int8, device="cuda")
    out = torch.zeros(h * w, dtype=torch.int32, device="cuda")
    grid = ((h * w + TILE - 1) // TILE, 1, 1)

    for name in ("today", "staged"):
        fn = ctypes.c_void_p()
        check(CUDA.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()),
              f"cuModuleGetFunction {name}")
        values = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(wt.data_ptr()),
                  ctypes.c_void_p(out.data_ptr()), ctypes.c_int(h), ctypes.c_int(w)]
        params = (ctypes.c_void_p * len(values))()
        for i, v in enumerate(values):
            params[i] = ctypes.cast(ctypes.byref(v), ctypes.c_void_p)   # keep v alive below
        for _ in range(args.iters):
            check(CUDA.cuLaunchKernel(fn, grid[0], 1, 1, TILE, 1, 1, 0, None, params, None),
                  f"launch {name}")
        check(CUDA.cuCtxSynchronize(), "cuCtxSynchronize")
        print(f"# launched {name} {args.iters}x on {h}x{w} ({grid[0]} blocks x {TILE} threads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

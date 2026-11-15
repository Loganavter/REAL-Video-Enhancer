"""Launch the capped cubin directly, bypassing Triton's launcher, and check it bit-exactly first.

Four silent contracts, every one of them a wrong picture rather than a fault:
  sharedMemBytes   12288 (12 KB of dynamic staging) - 0 reads outside the buffer
  block size       128 threads - the cubin no longer declares .reqntid
  grid             16200 blocks at BM 128 (8100 belongs to BM 256) - else half a frame
  scratch          the PTX signature has two trailing pointers (Triton's global_scratch and profile
                   scratch); passing NULL may be fine, so they are allocated instead of guessed

Usage: launch_cubin.py <cubin> <ref.py-derived tensors>  - implemented as a self-check below.
"""
import ctypes
import sys

import torch

CUDA = ctypes.CDLL("libcuda.so.1")
CUDA.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]


def check(res, what):
    if res != 0:
        raise RuntimeError(f"{what} failed: {res}")


def main() -> int:
    cubin_path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/opencode/k_s1.cubin"
    smem = 12288
    block = 128
    H = W = 64
    C = 64

    check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    torch.zeros(1, device="cuda")          # let torch establish the context, then share it
    ctx = ctypes.c_void_p()
    check(CUDA.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    if not ctx.value:
        check(CUDA.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")

    blob = open(cubin_path, "rb").read()
    module = ctypes.c_void_p()
    check(CUDA.cuModuleLoadData(ctypes.byref(module), blob), "cuModuleLoadData")
    fn = ctypes.c_void_p()
    check(CUDA.cuModuleGetFunction(ctypes.byref(fn), module, b"conv3x3_int8_prelu_kernel"),
          "cuModuleGetFunction")

    torch.manual_seed(0)
    x = torch.randint(-127, 127, (1, H, W, C), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 127, (9, C, C), device="cuda", dtype=torch.int8)
    slope = torch.rand(C, device="cuda") * 2.4 - 0.7
    oscale = torch.rand(C, device="cuda") * 0.05 + 0.004
    y = torch.zeros_like(x)
    scratch = torch.zeros(1024, device="cuda", dtype=torch.uint8)
    prof = torch.zeros(1024, device="cuda", dtype=torch.uint8)

    def ptr(t):
        return ctypes.c_void_p(t.data_ptr())

    params = (ctypes.c_void_p * 11)()
    values = [
        (ptr(x), "ptr"), (ptr(w), "ptr"), (ptr(slope), "ptr"), (ptr(oscale), "ptr"), (ptr(y), "ptr"),
        (ctypes.c_uint(H), "u32"), (ctypes.c_uint(W), "u32"),
        (ctypes.c_float(1.0), "f32"), (ctypes.c_float(0.01), "f32"),
        (ptr(scratch), "ptr"), (ptr(prof), "ptr"),
    ]
    keep = []
    for i, (v, _) in enumerate(values):
        params[i] = ctypes.cast(ctypes.byref(v), ctypes.c_void_p)
        keep.append(v)

    grid = (H * W + block - 1) // block
    check(CUDA.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, ctypes.c_uint(smem), None, params, None),
          "cuLaunchKernel")
    check(CUDA.cuCtxSynchronize(), "cuCtxSynchronize")

    # reference: the same maths in torch, int8 exact
    xf = x.float()
    wf = w.float() * 0.01
    acc = torch.zeros(H * W, C, device="cuda", dtype=torch.float32)
    for tap in range(9):
        dy, dx = tap // 3 - 1, tap % 3 - 1
        sh = torch.zeros_like(x)
        ys, yt = slice(max(0, dy), H + min(0, dy)), slice(max(0, -dy), H + min(0, -dy))
        xs, xt = slice(max(0, dx), W + min(0, dx)), slice(max(0, -dx), W + min(0, -dx))
        sh[:, yt, xt, :] = x[:, ys, xs, :]
        acc += torch._int_mm(sh.reshape(H * W, C).contiguous(), w[tap].contiguous()).float()
    v = acc * 1.0
    v = torch.where(v >= 0, v, v * slope[None, :])
    want = torch.floor(v / oscale[None, :] + 0.5).clamp(-127, 127).to(torch.int8).reshape(1, H, W, C)

    diff = (y.int() - want.int()).abs()
    print(f"grid {grid} x {block} threads, smem {smem} B: max |diff| {int(diff.max())}, "
          f"share != 0 {float((diff != 0).float().mean()) * 100:.2f}%")
    return 0 if int(diff.max()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

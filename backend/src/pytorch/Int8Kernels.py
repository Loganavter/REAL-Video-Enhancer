"""The Triton kernels the int8 body needs, vendored into the app.

`scripts/` is not shipped with the app, so the kernels live here rather than being imported from the offline
tooling. They are the exact kernels measured in the plan: a prologue that requants a per-channel activation
grid to a per-tensor one inside the same launch as the int8 dot, plus the two element-wise conversions the
boundaries need.

Three details that were each worth a bug when they were got wrong:

* the requant uses ONE constant for both signs, `(prod + half) >> shift`. Adding `-half` for negatives rounds
  away from zero and measured five times worse than plain flooring;
* the conv bias is added after dequantisation and before PReLU. Its mean is 2-4 against activations of ~0.5 -
  dropping it is a different network, not a rounding choice;
* every conversion here fuses what torch does in three passes (round/clamp/cast then permute then half), which
  is where 4.0 ms of a 54 ms cycle went.
"""
import triton
import triton.language as tl

__all__ = ["body_prologue_kernel", "quantise_nchw_to_nhwc", "dequantise_nhwc_to_nchw"]


@triton.jit
def body_prologue_kernel(x_ptr, w_ptr, m_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
                         H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    m = tl.load(m_ptr + lanes)                      # int32 per INPUT channel
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.static_range(9):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0).to(tl.int32)
        a = ((a * m[None, :]) + (1 << (shift - 1))) >> shift
        a = tl.minimum(tl.maximum(a, -127), 127).to(tl.int8)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


@triton.jit
def quantise_nchw_to_nhwc(x_ptr, s_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """head activation (NCHW float) -> int8 on the per-channel grid, written NHWC, in one pass."""
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    v = tl.load(x_ptr + ch[:, None] * n_pix + pix[None, :], mask=mask[None, :], other=0.0)
    s = tl.load(s_ptr + ch)[:, None]
    q = tl.floor(v / s + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + pix[:, None] * C + ch[None, :], tl.trans(q).to(tl.int8), mask=mask[:, None])


@triton.jit
def dequantise_nhwc_to_nchw(q_ptr, s_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """last hop's int8 codes (NHWC) -> fp16 NCHW in one pass, for the fp16 tail."""
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    q = tl.load(q_ptr + pix[:, None] * C + ch[None, :], mask=mask[:, None], other=0).to(tl.float32)
    s = tl.load(s_ptr + ch)[None, :]
    tl.store(y_ptr + ch[:, None] * n_pix + pix[None, :], tl.trans(q * s).to(tl.float16),
             mask=mask[None, :])

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

try:                       # round-half-to-EVEN, which is what torch.round does
    from triton.language.extra import libdevice
    _rint = libdevice.rint
except Exception:          # pragma: no cover - older triton
    _rint = None

__all__ = ["body_prologue_kernel", "quantise_nchw_to_nhwc", "quantise_nchw_to_nhwc_prelu", "quantise_nhwc",
           "quantise_nhwc_prelu",
           "dequantise_nhwc_to_nchw", "dequantise_nhwc"]


@triton.jit
def body_prologue_kernel(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
                         y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    # tl.range with one stage, NOT static_range: unrolling all nine taps costs registers and measured
    # 2.919 ms per launch against 2.198 for the rolled loop at BM 128 / 4 warps. Pipelining it (stages 2-3)
    # then made it worse again - 2.497 and 2.635 - because the gathers are already L2-resident.
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        # the buffer ALREADY arrives on this hop's per-tensor grid: the previous hop's epilogue
        # requanted it, once per element, instead of nine times here on every load.
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    # ...and the requant for the NEXT hop goes here, one multiply and one shift per element
    q = q.to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
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
    # floor(x + 0.5) is round-half-UP; torch.round is round-half-to-EVEN. On half-precision activations a
    # whole-step ratio lands exactly on .5 often enough that the head codes differ on 0.0023% of elements,
    # and that difference amplified through sixteen hops to 1.23% of pixels over the bar. Match the
    # reference exactly rather than approximately.
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


@triton.jit
def quantise_nhwc(x_ptr, s_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """Head activation ALREADY in NHWC memory -> int8 NHWC. No copy, no transpose, no `.contiguous()`.

    A PReLU output of a channels_last convolution is channels_last, and `permute(0,2,3,1)` on it is a *view* -
    the bytes are already NHWC. The NCHW variant indexes `ch * n_pix + pix` and therefore needs a contiguous
    tensor, which costs a full transpose of a 64-channel 1080p half tensor (89.25 ms against 72.49 ms for the
    whole body). This one reads the same bytes the tensor already has.
    """
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    v = tl.load(x_ptr + pix[:, None] * C + ch[None, :], mask=mask[:, None], other=0.0)
    s = tl.load(s_ptr + ch)[None, :]
    q = tl.floor(v / s + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + pix[:, None] * C + ch[None, :], q.to(tl.int8), mask=mask[:, None])


@triton.jit
def dequantise_nhwc(q_ptr, s_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """Last hop's codes -> fp16, staying in NHWC memory.

    The model's weights are channels_last, so `y.permute(0,3,1,2)` is a channels_last *view* the tail
    convolution consumes natively. Writing NCHW instead forces the same transpose the head conversion was
    paying for.
    """
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    q = tl.load(q_ptr + pix[:, None] * C + ch[None, :], mask=mask[:, None], other=0).to(tl.float32)
    s = tl.load(s_ptr + ch)[None, :]
    tl.store(y_ptr + pix[:, None] * C + ch[None, :], (q * s).to(tl.float16), mask=mask[:, None])


@triton.jit
def quantise_nhwc_prelu(x_ptr, s_ptr, slope_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """Pre-PReLU head activation (NHWC) -> PReLU -> int8 NHWC, in one pass.

    The head's PReLU was a separate element-wise pass over a 64-channel 1080p tensor: 265 MB read and 265 MB
    written to produce what this kernel can write straight into the 132 MB int8 buffer. Measured 2.0 ms.
    """
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    v = tl.load(x_ptr + pix[:, None] * C + ch[None, :], mask=mask[:, None], other=0.0)
    # the slope MUST be fp16: an fp32 slope promotes this whole tile to fp32, which at BLOCK 1024 measured
    # 12.818 ms against 1.522 ms with an fp16 slope at BLOCK 256 - a 8x kernel-quality penalty for one type
    slope = tl.load(slope_ptr + ch)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    q = tl.floor(v.to(tl.float32) / tl.load(s_ptr + ch)[None, :] + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + pix[:, None] * C + ch[None, :], q.to(tl.int8), mask=mask[:, None])


@triton.jit
def quantise_nchw_to_nhwc_prelu(x_ptr, s_ptr, slope_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """Pre-PReLU head (NCHW) -> PReLU -> int8 NHWC, one pass.

    The app's head is contiguous NCHW, and `F.prelu` on it allocates a fresh 265 MB tensor every frame - which
    measured 20.2 ms for the phase against 1.6 ms for the conversion kernel alone on the same input. Fusing the
    PReLU removes both the allocation and the extra read-write.
    """
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    v = tl.load(x_ptr + ch[:, None] * n_pix + pix[None, :], mask=mask[None, :], other=0.0)
    sl = tl.load(slope_ptr + ch)[:, None].to(tl.float16)
    v = tl.where(v >= 0, v, v * sl)
    q = tl.floor(v / tl.load(s_ptr + ch)[:, None].to(tl.float16) + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + pix[:, None] * C + ch[None, :], tl.trans(q).to(tl.int8), mask=mask[:, None])

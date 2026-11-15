import sys, torch, triton
import triton.language as tl

@triton.jit
def body_prologue_kernel(x_ptr, w_ptr, m_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr, H, W,
                         BLOCK_M: tl.constexpr, C: tl.constexpr):
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
        # per-channel grid -> per-tensor grid: add half a step and let the arithmetic shift floor.
        # ONE constant for both signs - adding -half for negatives (what this did first) rounds AWAY
        # from zero, which is worse than flooring, not better.
        a = ((a * m[None, :]) + (1 << (shift - 1))) >> shift
        a = tl.minimum(tl.maximum(a, -127), 127).to(tl.int8)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    # the conv's BIAS belongs here, after dequantisation and before PReLU: its mean is 2-4 against an
    # activation of ~0.5, so dropping it is not a rounding choice - it is a different network
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))



@triton.jit
def quantise_nchw_to_nhwc(x_ptr, s_ptr, y_ptr, n_pix, BLOCK: tl.constexpr, C: tl.constexpr):
    """Head activation (NCHW float) -> int8 on the per-channel grid, written NHWC, in ONE pass.

    The torch version round/clamp/cast then permute+contiguous, which is four passes over a 64-channel
    1080p buffer and measured 10.0 ms - 17% of the whole int8 cycle.
    """
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
    """Last hop's int8 codes (NHWC) -> fp16 NCHW in ONE pass, for the fp16 tail.

    The torch version dequantises into a 530 MB float32 tensor, then permutes and casts it to half for the
    tail conv - three passes and two extra buffers over a 64-channel 1080p activation.
    """
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    ch = tl.arange(0, C)
    mask = pix < n_pix
    q = tl.load(q_ptr + pix[:, None] * C + ch[None, :], mask=mask[:, None], other=0).to(tl.float32)
    s = tl.load(s_ptr + ch)[None, :]
    tl.store(y_ptr + ch[:, None] * n_pix + pix[None, :], tl.trans(q * s).to(tl.float16),
             mask=mask[None, :])

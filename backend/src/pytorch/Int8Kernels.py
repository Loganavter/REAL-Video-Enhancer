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

__all__ = ["body_prologue_kernel", "hop_padded_kernel", "head_int8_kernel", "tail_int8_kernel",
           "tail_int8_shuffle_kernel", "tail_int8_yuv_kernel",
           "quantise_nchw_to_nhwc", "quantise_nchw_to_nhwc_prelu",
           "quantise_nhwc", "quantise_nhwc_prelu",
           "dequantise_nhwc_to_nchw", "dequantise_nhwc"]


@triton.jit
def hop_padded_kernel(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
                      y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr, READ_PADDED: tl.constexpr,
                      WRITE_PADDED: tl.constexpr = 1):
    """The hop against a zero-haloed buffer: no bounds mask, and the store lands in the next interior.

    Measured (docs/wsp/verified/int8-hop-tiles.txt, int8-hop-halo-chain.txt): the mask-free form is 2.3% faster
    with identical codes, and the halo costs nothing when every hop writes into the next hop's padded buffer at a
    (+1, +1) offset - the border is zeroed once per frame (~0.08 ms) instead of a 134 MB copy per hop.

    READ_PADDED=0 is used for the FIRST hop, whose input is the quantiser's plain (H, W, C) buffer: the read is
    masked there and the store is already padded, so no extra pass exists anywhere in the chain.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        if READ_PADDED:
            src = ((h + 1 + dy) * (W + 2) + (w + 1 + dx))[:, None] * C + lanes[None, :]
            a = tl.load(x_ptr + src)
        else:
            yy = h + dy
            xx = w + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            src = (yy * W + xx)[:, None] * C + lanes[None, :]
            a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    osc = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / osc + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    if WRITE_PADDED:
        dst = ((h + 1) * (W + 2) + (w + 1))[:, None] * C + lanes[None, :]
    else:
        # the LAST hop writes a plain (H, W, C) buffer: the tail kernel indexes pix*C linearly, so it cannot
        # read a padded interior, and copying the interior out would cost more than the halo saves
        dst = rows[:, None] * C + lanes[None, :]
    tl.store(y_ptr + dst, q.to(tl.int8))


@triton.jit
def head_int8_kernel(x_ptr, w_ptr, bias_ptr, slope_ptr, s_ptr, y_ptr, H, W, BLOCK_M: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, CIN: tl.constexpr):
    """3 -> C fp16 head convolution with the bias, the PReLU and the int8 requant in one kernel.

    The head is 3 -> 64 3x3, so its whole 3x3x3 window is K = 27 terms - ONE dot per pixel, padded to 32 and
    masked, instead of nine taps. Fused with the epilogue it writes 133 MB of codes where the shipped path
    writes 265 MB of fp16 and then reads it back in a separate quantise pass; that pass is what the fusion
    removes, and the write it removes is 0.92 ms of traffic on its own.

    The input is the app's contiguous NCHW fp16 frame, so lane k = c*9 + t indexes plane `c` at the shifted
    coordinate. Same epilogue ORDER as the shipped path (conv, then bias, then PReLU, then round to the
    per-channel grid), so the codes are comparable rather than merely plausible.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    ki = tl.arange(0, K)
    c = ki // 9
    t = ki % 9
    dy = t // 3 - 1
    dx = t % 3 - 1
    yy = h[:, None] + dy[None, :]
    xx = w[:, None] + dx[None, :]
    inside = (c[None, :] < CIN) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    src = c[None, :] * (H * W) + yy * W + xx
    a = tl.load(x_ptr + src, mask=inside, other=0.0)
    lo = tl.arange(0, C)
    b = tl.load(w_ptr + ki[:, None] * C + lo[None, :], mask=(c[:, None] < CIN), other=0.0)
    acc = tl.dot(a, b)
    v = acc + tl.load(bias_ptr + lo)[None, :]
    slope = tl.load(slope_ptr + lo)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    s = tl.load(s_ptr + lo)[None, :]
    q = tl.floor(v / s + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lo[None, :], q.to(tl.int8))


@triton.jit
def tail_int8_kernel(x_ptr, w_ptr, sigma_ptr, bias_ptr, y_ptr, H, W, BLOCK_M: tl.constexpr,
                     CIN: tl.constexpr, COUT: tl.constexpr, COUT_PAD: tl.constexpr):
    """The tail as one int8 convolution whose weights already carry the dequant scale.

    The shipped tail dequantises the last hop's codes to fp16 (a full pass) and feeds them to a cuDNN fp16
    convolution. The dequant scale is per INPUT channel, and a convolution is linear in its input, so
    `w'[k,o] = s_k * w[k,o]` folded offline makes the codes themselves the input: the dequant pass disappears,
    the output scale is one multiply in the epilogue, and the integer accumulation is exact in fp32
    (64*9 terms of 127*127 = 9.3e6, far below 2^24).

    Outputs pad 12 -> 16 for the tensor-core tile; the padding contributes nothing (its weights are zero) and
    is not written, so the caller gets a compact 12-channel NHWC buffer - a channels_last view the upsampler
    already accepts.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    li = tl.arange(0, CIN)
    lo = tl.arange(0, COUT_PAD)
    acc = tl.zeros((BLOCK_M, COUT_PAD), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * CIN + li[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * CIN * COUT_PAD + li[:, None] * COUT_PAD + lo[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(sigma_ptr + lo)[None, :] + tl.load(bias_ptr + lo)[None, :]
    tl.store(y_ptr + rows[:, None] * COUT + lo[None, :], v.to(tl.float16), mask=lo[None, :] < COUT)


@triton.jit
def body_prologue_kernel(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
                         y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr, FOLD: tl.constexpr = 0):
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
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    if FOLD:
        # The rescale to the output grid is taken ONCE PER CHANNEL per CTA instead of once per element.
        # PReLU is homogeneous of degree one, so a POSITIVE rescale commutes with it:
        #     prelu(acc*w + b) / s + 0.5  ==  prelu(acc*(w/s) + b/s) + 0.5
        # and the only per-element work left is one FMA and one add. The 0.5 cannot move inside for the
        # same reason the division can: the shift would move the PReLU sign test.
        inv = 1.0 / out_scale
        u = acc.to(tl.float32) * (tl.load(w_scale_ptr + lanes)[None, :] * inv) \
            + tl.load(bias_ptr + lanes)[None, :] * inv
        u = tl.where(u >= 0, u, u * slope)
        q = tl.floor(u + 0.5)
    else:
        v = tl.where(v >= 0, v, v * slope)
        q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    # ...and the requant for the NEXT hop goes here, one multiply and one shift per element
    q = q.to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


@triton.jit
def tail_int8_shuffle_kernel(x_ptr, w_ptr, sigma_ptr, bias_ptr, lr_ptr, y_ptr, H, W, BLOCK_M: tl.constexpr,
                             CIN: tl.constexpr, COUT: tl.constexpr, COUT_PAD: tl.constexpr,
                             SCALE: tl.constexpr):
    """The tail, the pixel shuffle and the residual in one kernel - the body's last three steps.

    The shipped path computes the 64 -> 12 tail convolution, then `nn.PixelShuffle(2)` rearranges those 12
    channels into 3 at twice the size, then the residual adds the nearest-upsampled input. All three are
    element-wise on the same 12 values: pixel_shuffle is a PERMUTATION, and the residual adds the SAME LR
    pixel to all four subpixels. So folding them here is not an approximation - it computes the same
    additions in the same order and the render is expected to be byte-identical.

    Lane k of the 16-wide epilogue is input channel `c = k // (SCALE*SCALE)`, subpixel `i = (k % 4) // 2`,
    `j = k % 2`, which is torch's own `channel = c*r^2 + i*r + j`. Each input pixel writes 4 subpixels of 3
    contiguous channels into two output rows, so neighbouring pixels still write contiguous runs.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    li = tl.arange(0, CIN)
    lo = tl.arange(0, COUT_PAD)
    acc = tl.zeros((BLOCK_M, COUT_PAD), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * CIN + li[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * CIN * COUT_PAD + li[:, None] * COUT_PAD + lo[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(sigma_ptr + lo)[None, :] + tl.load(bias_ptr + lo)[None, :]
    chan = lo // (SCALE * SCALE)
    sub = lo % (SCALE * SCALE)
    sy = sub // SCALE
    sx = sub % SCALE
    lm = lo[None, :] < COUT
    # the LR pixel this output pixel's nearest upsample lands on: (h, w), plane `chan`
    lr = tl.load(lr_ptr + chan[None, :] * (H * W) + rows[:, None], mask=lm, other=0.0)
    # Round the conv result to fp16 BEFORE the residual, exactly as the shipped path does (it stores fp16 and
    # then adds an fp16 residual). Adding in fp32 and rounding once would be a different arithmetic - one
    # rounding instead of two - and would show up as 1-ulp differences on a handful of pixels.
    val = v.to(tl.float16) + lr
    out_pos = ((h[:, None] * SCALE + sy[None, :]) * (W * SCALE) + (w[:, None] * SCALE + sx[None, :])) * 3 \
        + chan[None, :]
    tl.store(y_ptr + out_pos, val, mask=lm)


@triton.jit
def tail_int8_yuv_kernel(x_ptr, w_ptr, sigma_ptr, bias_ptr, lr_ptr, y_ptr, H, W, BLOCK_M: tl.constexpr,
                         CIN: tl.constexpr, COUT: tl.constexpr, COUT_PAD: tl.constexpr,
                         SCALE: tl.constexpr):
    """Tail conv + shuffle + residual + colour conversion, writing packed yuv420p directly.

    PixelShuffle(2) turns the 12 tail channels into a 2x2 output block per input pixel, and 4:2:0 chroma averages
    over exactly that block - so the four Y and the single U and V of a position are all produced from one input
    pixel, with no cross-lane work. Y, Cb and Cr are each LINEAR in (R, G, B), which is why each is a weighted sum
    over the channel axis of the reshaped tile. Weights, with BT.709 and limited range:

        Y  =  0.2126 R + 0.7152 G + 0.0722 B          -> Yq = round(219 Y + 16),    clamp 16..235
        Cb = (B - Y) * 0.5/(1-0.0722) = -0.11459 R - 0.38547 G + 0.50000 B
        Cr = (R - Y) * 0.5/(1-0.2126) =  0.50000 R - 0.45419 G - 0.04585 B
        Uq = round(224 mean(Cb) + 128), Vq likewise,  clamp 16..240

    Output: 6 bytes per input pixel - four Y in the 2H x 2W plane, then U and V in two H x W planes.

    The arithmetic order differs from the shipped path (fp16 arithmetic, rounding before the chroma average), so
    this is a DIFFERENT-ARITHMETIC change: the offline gate measured max 1 level over the whole payload.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    li = tl.arange(0, CIN)
    lo = tl.arange(0, COUT_PAD)
    acc = tl.zeros((BLOCK_M, COUT_PAD), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        a = tl.load(x_ptr + ((yy * W + xx)[:, None] * CIN + li[None, :]), mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * CIN * COUT_PAD + li[:, None] * COUT_PAD + lo[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(sigma_ptr + lo)[None, :] + tl.load(bias_ptr + lo)[None, :]
    v3 = tl.reshape(v, (BLOCK_M, 4, SCALE * SCALE))     # [channel][subpixel], torch's pixel_shuffle order
    ch = tl.arange(0, 4)
    wy = tl.where(ch == 0, 0.2126, tl.where(ch == 1, 0.7152, tl.where(ch == 2, 0.0722, 0.0)))
    wu = tl.where(ch == 0, -0.11459, tl.where(ch == 1, -0.38547, tl.where(ch == 2, 0.50000, 0.0)))
    wv = tl.where(ch == 0, 0.50000, tl.where(ch == 1, -0.45419, tl.where(ch == 2, -0.04585, 0.0)))
    y4 = tl.sum(v3 * wy[None, :, None], 1)              # (BLOCK_M, 4)
    u4 = tl.sum(v3 * wu[None, :, None], 1)
    w4 = tl.sum(v3 * wv[None, :, None], 1)
    # the residual adds the SAME LR pixel to all four subpixels: one value per position, per output channel
    lr_r = tl.load(lr_ptr + rows)
    lr_g = tl.load(lr_ptr + (H * W) + rows)
    lr_b = tl.load(lr_ptr + 2 * (H * W) + rows)
    y_res = lr_r * 0.2126 + lr_g * 0.7152 + lr_b * 0.0722
    u_res = lr_r * -0.11459 + lr_g * -0.38547 + lr_b * 0.5
    v_res = lr_r * 0.5 + lr_g * -0.45419 + lr_b * -0.04585
    yq = tl.minimum(tl.maximum(tl.floor((y4 + y_res[:, None]) * 219.0 + 16.0 + 0.5), 16.0), 235.0)
    uq = tl.minimum(tl.maximum(tl.floor((tl.sum(u4, 1) / 4.0 + u_res) * 224.0 + 128.0 + 0.5), 16.0), 240.0)
    vq = tl.minimum(tl.maximum(tl.floor((tl.sum(w4, 1) / 4.0 + v_res) * 224.0 + 128.0 + 0.5), 16.0), 240.0)
    sub = tl.arange(0, SCALE * SCALE)
    sy = sub // SCALE
    sx = sub % SCALE
    y_idx = (h[:, None] * SCALE + sy[None, :]) * (W * SCALE) + (w[:, None] * SCALE + sx[None, :])
    tl.store(y_ptr + y_idx, yq.to(tl.uint8))
    n = H * W
    tl.store(y_ptr + 4 * n + rows, uq.to(tl.uint8))
    tl.store(y_ptr + 5 * n + rows, vq.to(tl.uint8))


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

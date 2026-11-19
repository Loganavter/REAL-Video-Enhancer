"""The model's body in int8, head and tail left in fp16 - a drop-in inference helper.

The body is the sixteen 64-channel convolutions between the head and the tail; everything else about the
picture is unchanged, which is why this can be swapped in behind the probe's verdict without touching the
tail, the upsampler or the residual.

Built once per render from the model's own weights plus one calibration frame, then called per frame:

    out = model.upsampler(tail_conv(self(x))) + interpolate(x, scale, nearest)

with `self(x)` being clamp -> head conv + PReLU -> sixteen int8 hops -> dequantise to fp16.

Two things this deliberately does NOT claim:

* it is only correct for a Compact architecture with 64 body channels and 18 convs. The channel count is a
  `tl.constexpr` in the kernel, so a 48-channel model cannot use it at all - the coverage boundary measured in
  the plan's stage 5;
* it does not calibrate on spread frames yet. The grids come from the first activation it sees, which is what
  the render can supply without decoding ahead; the plan's better calibration (spread frames plus deliberately
  bright ones) is a later step, and the probe has already decided the video is admissible before this runs.

The measurement that justifies the swap, on the built pipeline against the app's own fp16 TensorRT engine in
one session with NVENC alongside: 1.489x idle, 1.505x under the encoder, image agreement 0.00004% of pixels.
"""
from __future__ import annotations

import os

import torch

from .Int8Kernels import (body_prologue_kernel, dequantise_nhwc, dequantise_nhwc_to_nchw,
                          head_int8_kernel, quantise_nchw_to_nhwc, quantise_nchw_to_nhwc_prelu,
                          hop_padded_kernel, quantise_nhwc, quantise_nhwc_prelu, tail_int8_kernel,
                          tail_int8_shuffle_kernel, tail_int8_yuv_kernel)

__all__ = ["Int8Body"]


class _NoEvent:
    """A null CUDA event: keeps one code path when instrumentation is switched off.

    `elapsed_time` on the previous frame's pair is a real cost: it BLOCKS until that pair has completed,
    which serialises the frame with the one before it. That is fine for a diagnostic and wrong for a
    measurement of what the render costs, so RVE_INT8_NO_EVENTS=1 replaces every event with this.
    """

    def record(self):
        pass

    def elapsed_time(self, _other):
        return 0.0

SHIFT = 20
# First-frame gross-error gate, in percent of pixels >8 levels against the fp16 model. NOT the project's
# acceptance bar (0.1%, a worst-frame-of-a-sample statistic): see certify() for the observed separation.
FIRST_FRAME_SANITY = 1.0

BLOCK_M = 128   # the rolled loop prefers a wider block: 2.198 ms against 2.919 at BM 64
WARPS = 4
QB = 256      # the conversions are register-bound: 1024 measured 2.3 ms against 1.57 at 256


class Int8Body:
    """Head (fp16) -> sixteen int8 hops (Triton) -> tail (fp16)."""

    def __init__(self, model, feat: int = 64, block_m: int = BLOCK_M, warps: int = WARPS):
        convs = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
        prels = [m for m in model.body if isinstance(m, torch.nn.PReLU)]
        # THE HEAD IS NOT ALWAYS A Conv2d. The app widens body[0] to 8 input channels and wraps it in
        # PadInputConv so the engine can fuse it with its PReLU (log: "3 -> 8, zeros, padded inside the
        # head"), and PadInputConv is not a Conv2d - so it is absent from this list and convs[0] is the
        # SECOND convolution. Building from convs[0] fed a 64-channel conv a 3-channel tensor and failed
        # with 'expected input[1,3,1080,1920] to have 64 channels'.
        head = list(model.body)[0]
        rest = convs if isinstance(head, torch.nn.Conv2d) else convs
        if not isinstance(head, torch.nn.Conv2d):
            rest = convs                      # head wrapped: the list holds the 16 inner convs + the tail
        else:
            rest = convs[1:]                  # head is a Conv2d: drop it
        if len(rest) != 17 or len(prels) != 17:
            raise ValueError(f"int8 body expects 16 inner convs plus a tail (17 Conv2d, 17 PReLU); "
                             f"this model has {len(rest)} and {len(prels)}")
        if rest[0].out_channels != feat:
            raise ValueError(f"int8 body needs {feat} body channels; this model has "
                             f"{rest[0].out_channels}")
        self.model = model
        self.head_conv, self.inner, self.tail_conv = head, rest[:-1], rest[-1]
        self.head_prelu = prels[0]
        self.inner_prelus = prels[1:]
        self.scale = int(getattr(model, "upscale", 2))
        self.feat, self.block_m, self.warps = feat, block_m, warps
        # The epilogue's per-element divide can be folded into the incoming rescale: `a = w_scale/out_scale`
        # and `b = bias/out_scale` computed once per channel. Kept behind a knob until it has been timed and
        # its picture measured (row rve-epilogue-divide-2026-11-19) - the rewrite is exact in real arithmetic
        # but not bit-identical in fp32, so it is a different-arithmetic change, not a layout.
        self.fold = int(os.environ.get("RVE_INT8_EPILOGUE_FOLD", "0"))
        # Zero-haloed hop buffers: no bounds mask in the kernel, and every hop writes into the NEXT hop's
        # padded interior, so the halo costs one border zeroing per frame (0.08 ms) instead of a copy per hop.
        # Measured on the 16-hop chain: +0.61 ms/frame net (+2.3%) with identical codes
        # (docs/wsp/verified/int8-hop-halo-chain.txt). RVE_HOP_HALO=0 restores the plain buffers.
        # DEFAULT ON since 2026-11-19: 34.50 -> 33.51 ms/frame (-0.99, -2.9%) in 3 of 3 interleaved rounds, and
        # the render is byte-identical (0 differing pixels on two frames, IDENTITY on the 494-frame pair).
        # Costs 382 MB of buffers (three padded ones); RVE_HOP_HALO=0 restores the plain layout.
        self.hop_halo = str(os.environ.get("RVE_HOP_HALO", "1")).lower() in ("1", "true", "yes")
        # The tail as a seventeenth int8 hop: the dequant pass and the cuDNN fp16 conv become one kernel
        # (row rve-tail-as-int8-hop-2026-11-19). Stage (a) put it at 0.00147% share / peak 13 against the
        # fp16 tail on four held-out frames, stage (b) at 0.974 ms against 4.444; in the app it is
        # 45.87 -> 42.71 ms/frame in 3 of 3 interleaved rounds and the 494-frame pair reads 0.00000%
        # worst share / peak 6. DEFAULT ON since 2026-11-20; RVE_TAIL_INT8=0 restores the cuDNN path.
        # The head as ONE kernel: 3 -> 64 convolution with the bias, the PReLU and the int8 requant in its
        # epilogue. It writes the codes the first hop reads, so the separate quantise pass - and the 265 MB
        # fp16 activation it exists to compress - disappear (row rve-head-quantise-fusion-2026-11-19).
        # DEFAULT ON since 2026-11-20: 44.45 -> 39.18 ms/frame in 3 of 3 interleaved rounds and a 494-frame
        # pair at 0.00018% worst share / peak 13. RVE_HEAD_FUSED=0 restores the cuDNN + quantise path.
        # The app's head may be widened to 8 input channels for the ENGINE's fusion (PadInputConv); the
        # extra channels are zeros, so the first three input channels of its weight are the real 3 -> 64
        # convolution and are what this kernel runs.
        _plain_head = self.head_conv
        if not isinstance(_plain_head, torch.nn.Conv2d) and hasattr(_plain_head, "conv"):
            _plain_head = _plain_head.conv
        self._head_conv_plain = _plain_head if isinstance(_plain_head, torch.nn.Conv2d) else None
        self.head_fused = (str(os.environ.get("RVE_HEAD_FUSED", "1")).lower() in ("1", "true", "yes")
                           and self._head_conv_plain is not None
                           and self._head_conv_plain.in_channels >= 3
                           and self._head_conv_plain.out_channels == feat)
        want_tail_int8 = str(os.environ.get("RVE_TAIL_INT8", "1")).lower() in ("1", "true", "yes")
        self.tail_int8 = bool(want_tail_int8 and self.tail_conv.out_channels <= 16)
        # The tail, the pixel shuffle and the residual in ONE kernel: pixel_shuffle is a permutation and the
        # residual adds one LR pixel to all four subpixels, so the fold is exact and the render IS
        # byte-identical - 0 differing values on the tensor and all 494 frames of the clip identical
        # (row rve-tail-epilogue-shuffle-residual-2026-11-20). DEFAULT ON since 2026-11-20. Needs the int8
        # tail (it is its
        # epilogue) and a plain `nn.PixelShuffle` upsampler of the matching 3*r^2 channel count.
        _up = getattr(model, "upsampler", None)
        _r = int(getattr(_up, "upscale_factor", 0) or 0) if isinstance(_up, torch.nn.PixelShuffle) else 0
        self.tail_shuffle = bool(
            str(os.environ.get("RVE_TAIL_SHUFFLE", "1")).lower() in ("1", "true", "yes")
            and self.tail_int8 and _r == 2 and self.tail_conv.out_channels == 3 * _r * _r)
        self.tail_scale = _r
        # The tail writes the FINAL PIPE PAYLOAD instead of fp16 RGB (row rve-tail-csc-yuv-2026-11-20).
        # PixelShuffle(2) makes 4:2:0 chroma register-local - the 12 tail channels of one input pixel
        # become exactly the 2x2 block the chroma average covers - so the same epilogue can emit 6 bytes
        # per input pixel (12.44 MB) instead of 49.77 MB of fp16 RGB, and the pack becomes a copy instead
        # of YCbCr over 8.3 Mpx. Isolated with the conversion inside each arm: 5.123 -> 0.869 ms.
        # Off by default until the render pair's picture is in: it is a DIFFERENT-ARITHMETIC change (the
        # offline gate put it at max 1 level over the payload), so it is judged on renders, not bytes.
        # ADOPTED 2026-11-20 and default ON: three arms in one session read 52.39 / 33.63 / 31.47 ms
        # (fp16 / RGB-tail / yuv-tail, medians of rounds 2-3) = -2.16 ms and ratio 1.558 -> 1.665, and the
        # 494-frame pair reads 0.00000% worst share / peak 6 levels (int8-tail-csc.txt). The whole saving
        # is in the pack (4.37 -> 2.06); the tail phase is a wash (1.00 -> 1.00).
        # `out_yuv` additionally requires that the pipe really carries yuv420p (set by the upscaler from
        # `yuv8_output`, the same flag that chose the pipe format); RVE_TAIL_YUV=0 restores the RGB tail.
        self.tail_yuv = bool(str(os.environ.get("RVE_TAIL_YUV", "1")).lower() in ("1", "true", "yes")
                             and self.tail_shuffle and self.tail_scale == 2)
        self.out_yuv = self.tail_yuv
        if want_tail_int8 and not self.tail_int8:
            import logging as _lg
            _lg.getLogger("RVE").warning(
                "int8 tail request declined: the tail has %d output channels and this kernel pads to 16",
                self.tail_conv.out_channels)
        self._launches = None
        self._grids = None         # (s_act, outs)
        # fp16 for the kernel: an fp32 slope promotes the tile and costs 3x
        self._slope_half = self.head_prelu.weight.detach().half().contiguous()

    # ---- calibration: the per-channel maximum each PReLU produced on the frame it was shown
    def _measure(self, x):
        with torch.no_grad():
            act = torch.nn.functional.prelu(self.head_conv(x), self.head_prelu.weight)
            # .float() on the grids is load-bearing: the model runs in half, so a grid kept in half makes
            # the requant multiplier s_in/s_t * 2^SHIFT overflow half (max 65504) to inf and cast to
            # INT32_MAX. That produces a plausible-looking timing and a 99% wrong picture - it has now
            # caught this lane three times.
            s_act = (act.abs().amax(dim=(0, 2, 3)).float() / 127.0).clamp_min(1e-8)
            outs = []
            for conv, prelu in zip(self.inner, self.inner_prelus):
                act = torch.nn.functional.prelu(conv(act), prelu.weight)
                outs.append((act.abs().amax(dim=(0, 2, 3)).float() / 127.0).clamp_min(1e-8))
        return s_act, outs

    def calibrate(self, frames):
        """Take the grids from the SPREAD frames the probe already decoded.

        One frame is not enough: measured on a5_chainsaw, single-frame grids give a COMPOSITION error of 0.116%
        where sixteen-frame grids give 0.006% - a 20x difference, and the difference between failing and passing a
        0.1% bar. The grids are the element-wise maximum over the frames, so a bright frame raises the ceiling
        instead of the first frame setting it.
        """
        if torch.is_tensor(frames):
            frames = [frames]
        s_act, outs = None, None
        for f in frames:
            s2, o2 = self._measure(f)
            if s_act is None:
                s_act, outs = s2, o2
            else:
                s_act = torch.maximum(s_act, s2)
                outs = [torch.maximum(a, b) for a, b in zip(outs, o2)]
        self._grids = (s_act, outs)
        self._calib_frames = list(frames)
        self._launches = None
        return self

    @torch.no_grad()
    def _bias_correction(self, max_frames: int = 4) -> list:
        """Empirical bias correction (Nagel et al., ICCV 2019): the mean shift of each channel, per hop.

        Quantisation error is not zero-mean. The BD probe shows the drawn line lifting ~5 levels while the rim
        beside it stays - a signed shift, which is the signature of a biased error rather than noise, and the
        standard remedy is to measure `E[quantised - full precision]` per output channel on calibration data
        and subtract it from that layer's bias. The correction lives in the bias, so it costs nothing at
        inference; this is the one lever that does not eat the speed that int8 exists for.

        The shift is taken at the PRE-activation (before PReLU), because that is where a bias can act.
        """
        import torch.nn.functional as F

        s_act, outs = self._grids
        frames = self._calib_frames[:max_frames]
        shifts = [torch.zeros(self.feat, device=s_act.device) for _ in self.inner]
        for f in frames:
            x = f.clamp(0.0, 1.0)
            # the model is fp16, the correction arithmetic is fp32: cast once, here, or torch refuses the
            # mixed conv ("Input type Half and bias type float") - which is how the first attempt at this
            # correction fell back to fp16 and looked like a perfect result on the render pair.
            fp_pre = self.head_conv(x).float()
            fp = F.prelu(fp_pre, self.head_prelu.weight.float())
            # quantised chain, the kernel's own recipe
            s_t1 = self._s_t1.view(1, -1, 1, 1)
            codes = torch.round(fp / s_t1).clamp(-127, 127)
            for idx, conv in enumerate(self.inner):
                weight = conv.weight.detach().float()
                s_in = self._s_t1 if idx == 0 else outs[idx - 1]
                s_t = float(s_in.max())
                w_sc = weight.abs().amax(dim=(1, 2, 3)) / 127.0
                w_q = torch.round(weight / w_sc.view(-1, 1, 1, 1)).clamp(-127, 127)
                q_pre = F.conv2d(codes.float(), w_q, None, padding=1) * (w_sc * s_t).view(1, -1, 1, 1) \
                    + conv.bias.detach().float().view(1, -1, 1, 1)
                # The bias correction is for the WEIGHT quantisation only (Nagel eq. 25): E[dW x], measured on
                # the SAME input x. Comparing the two chains' outputs instead also counts the input drift that
                # the previous hops already caused, and subtracting that double-corrects - which is what made
                # the first attempt flip the picture to 100% wrong instead of fixing the stroke.
                bias_f = conv.bias.detach().float().view(-1)
                with torch.no_grad():
                    # the kernel reads CODES, i.e. x/s_t, and multiplies by (w_sc * s_t): feeding it the real
                    # x and the same multiplier over-counts by s_t, which is what made the first two attempts
                    # subtract a term several times too large and flip the picture.
                    same_q = F.conv2d((fp.float() / s_t), w_q, None, padding=1) * (w_sc * s_t).view(1, -1, 1, 1) + bias_f.view(1, -1, 1, 1)
                    same_fp = F.conv2d(fp.float(), weight, bias_f, padding=1)
                    shifts[idx] += (same_q - same_fp).mean(dim=(0, 2, 3))
                q_act = F.prelu(q_pre, self.inner_prelus[idx].weight.detach().float())
                s_out = outs[idx].view(1, -1, 1, 1)
                q = torch.floor(q_act / s_out + 0.5).clamp(-127, 127)
                if idx + 1 < len(self.inner):
                    s_next = outs[idx].max()
                    m = torch.round(outs[idx] / s_next * (1 << SHIFT)).float()
                    q = torch.clamp(torch.floor((q * m.view(1, -1, 1, 1) + (1 << (SHIFT - 1))) / (1 << SHIFT)),
                                    -127, 127)
                codes = q
                # advance the full-precision chain by the same layer
                fp_pre = F.conv2d(fp, weight, conv.bias.detach().float().view(-1), padding=1)
                fp = F.prelu(fp_pre, self.inner_prelus[idx].weight.detach().float())
        return [s / max(1, len(frames)) for s in shifts]

    def _build(self):
        from torch.nn.functional import conv2d  # noqa: F401 - documents what the kernel replaces
        import os as _os

        s_act, outs = self._grids
        # 2026-11-18: the hop boundary keeps its PER-CHANNEL grid instead of being collapsed to
        # max(s_in) and repacked with an m multiplier. The simulation isolated this as the lever
        # (docs/wsp/verified/int8-scales-bd.txt: 0.06510% -> 0.00162% on the worst BD frame), and the way
        # to get it in the kernel is to fold s_in per INPUT channel into the weights at build time - the
        # integer dot then needs no multiplier at all (m = 1 << SHIFT, identity). Build-time cost: one
        # requantise of 16 x 64 x 64 x 3 x 3 weights; runtime cost: none.
        # Measured the same day: the change is picture-neutral (the two variants differ by 0.01541% of
        # pixels with zero frames over the bar between them) and marginally worse against the fp16
        # reference (1.03% vs 0.91%), so the SHIPPED path stays per-tensor; this is the opt-in A/B knob.
        per_tensor = _os.environ.get("RVE_INT8_PER_CHANNEL") != "1"
        # Grid margin. The ink/rim probe (docs/wsp/verified/int8-scales-bd.txt) showed int8 lifting the
        # line by ~+5 levels while the rim stays - the signature the simulation produces only when the
        # quantiser CLIPS. The kernel does not clip on purpose, but its per-tensor grid comes from 24
        # calibration frames, so activations outside that range clamp at +-127. A margin widens the grid
        # and costs resolution; it is the cheapest way to test whether clipping is the mechanism.
        margin = float(_os.environ.get("RVE_INT8_GRID_MARGIN", "1.0"))
        self._s_t1 = (s_act.max().expand_as(s_act) if per_tensor else s_act).contiguous()
        identity = torch.full((self.feat,), 1 << SHIFT, device=s_act.device, dtype=torch.int32)
        launches, s_in = [], s_act
        for idx, conv in enumerate(self.inner, start=1):
            s_out = outs[idx - 1]
            s_t = float(s_in.max()) * margin
            # per-channel grids: the input scale is inside the weights, so the epilogue's requant is the
            # identity - and that is not cosmetic, a non-identity m here would apply max(s_in) twice.
            m_int = (torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32)
                     if per_tensor else identity.clone()).contiguous()
            weight = conv.weight.detach().float()
            if not per_tensor:
                weight = weight * (s_in * margin).view(1, -1, 1, 1)
            w_sc = weight.abs().amax(dim=(1, 2, 3)) / 127.0
            w_q = torch.round(weight / w_sc.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
            tap = torch.stack([w_q[:, :, i, j].t() for i in range(3) for j in range(3)]).contiguous()
            launches.append(dict(
                w=tap, m=m_int,
                vec=(w_sc if not per_tensor else w_sc * s_t).contiguous(),
                bias=conv.bias.detach().float().contiguous(),
                slope=self.inner_prelus[idx - 1].weight.detach().float().contiguous(),
                os=s_out.contiguous()))
            s_in = s_out
        # each hop's epilogue requants into the NEXT hop's per-tensor grid, so hop k gets hop k+1's
        # multiplier and the last hop gets the identity (it stays on its per-channel grid, as before)
        for k in range(len(launches)):
            launches[k]["mn"] = launches[k + 1]["m"] if k + 1 < len(launches) else identity
        if _os.environ.get("RVE_INT8_BIAS_CORR") == "1":
            shifts = self._bias_correction()
            for k, sh in enumerate(shifts):
                launches[k]["bias"] = (launches[k]["bias"] - sh).contiguous()
        self._launches = launches
        if self.head_fused:
            self._build_head_fused()
        if self.tail_int8:
            self._build_tail_int8()
        return launches

    def _build_head_fused(self):
        """The head's (K, C) weight matrix for the fused kernel: lane k = c*9 + t, exactly as it indexes."""
        w = self._head_conv_plain.weight.detach().half()[:, :3]     # the widened head's zeros are dropped
        k = 32
        w32 = torch.zeros((k, self.feat), device=w.device, dtype=torch.float16)
        for c in range(3):
            for i in range(3):
                for j in range(3):
                    w32[c * 9 + i * 3 + j] = w[:, c, i, j]
        self._head_w32 = w32.contiguous()
        self._head_bias = self._head_conv_plain.bias.detach().float().contiguous()
        self._head_slope = self.head_prelu.weight.detach().float().contiguous()
        return w32

    def _build_tail_int8(self):
        """Fold the last hop's per-channel dequant scale into the tail's weights, quantise, pad to 16.

        `w'[k,o] = s_k * w[k,o]` is what makes the codes themselves the convolution's input; quantising `w'`
        per OUTPUT channel then gives the kernel's epilogue a single scale. The 12 -> 16 padding is zeros,
        which contributes nothing to any output and is therefore arithmetic-preserving.
        """
        cout = self.tail_conv.out_channels
        s_k = self._launches[-1]["os"].detach().float()
        w_fold = self.tail_conv.weight.detach().float() * s_k.view(1, -1, 1, 1)
        sigma = (w_fold.abs().amax(dim=(1, 2, 3)) / 127.0).clamp_min(1e-12)
        w_q = torch.round(w_fold / sigma.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
        pad = 16
        taps = torch.zeros((9, self.feat, pad), device=w_q.device, dtype=torch.int8)
        for i in range(3):
            for j in range(3):
                taps[i * 3 + j, :, :cout] = w_q[:, :, i, j].t()
        self._tail_taps = taps.contiguous()
        self._tail_sigma = torch.zeros(pad, device=w_q.device, dtype=torch.float32)
        self._tail_sigma[:cout] = sigma
        self._tail_bias = torch.zeros(pad, device=w_q.device, dtype=torch.float32)
        self._tail_bias[:cout] = self.tail_conv.bias.detach().float()
        return taps

    def _halo_buffers(self, x, h, w):
        """Three (h+2, w+2, C) code buffers with a zeroed one-pixel border, reused across frames."""
        key = (x.shape[0], h, w)
        if getattr(self, "_halo_key", None) != key:
            self._halo = [torch.zeros((h + 2, w + 2, self.feat), device=x.device, dtype=torch.int8)
                          for _ in range(3)]
            self._halo_key = key
        for p in self._halo:
            p[0, :, :] = 0
            p[-1, :, :] = 0
            p[:, 0, :] = 0
            p[:, -1, :] = 0
        return self._halo, key

    def _int8_buffers(self, x, h, w):
        """One allocation per shape for all three int8 buffers.

        Allocating the 133 MB quantise target inside the frame cost ~18 ms: a 133 MB cudaMalloc synchronises
        the device, and the phase clock charges the wait to whichever kernel comes next. Measured 20.3 ms for
        that phase in the app against 1.9 ms for the same kernel on the same tensor in isolation.
        """
        key = (x.shape[0], h, w)
        if getattr(self, "_buf_key", None) != key:
            self._buf = torch.empty((x.shape[0], h, w, self.feat), device=x.device, dtype=torch.int8)
            self._scratch_a = torch.empty_like(self._buf)
            self._scratch_b = torch.empty_like(self._buf)
            self._buf_key = key
        return self._buf, self._scratch_a, self._scratch_b

    def __call__(self, x):
        import os as _os_dbg
        if _os_dbg.environ.get("RVE_DEBUG_INPUT") == "1" and not getattr(self, "_dbg_input", False):
            self._dbg_input = True
            import logging as _lgd
            _t = x.float()
            _lgd.getLogger("RVE").warning(
                "int8 INPUT  shape=%s dtype=%s  mean=%.5f min=%.5f max=%.5f  ch-mean=%s",
                tuple(x.shape), x.dtype, float(_t.mean()), float(_t.min()), float(_t.max()),
                [round(float(v), 5) for v in _t[0].mean(dim=(1, 2)).tolist()])
        if self._grids is None:
            self.calibrate(x)
        if self._launches is None:
            self._build()
        s_act, outs = self._grids
        h, w = x.shape[-2], x.shape[-1]
        n_pix = h * w
        grid = ((n_pix + self.block_m - 1) // self.block_m,)
        qgrid = ((n_pix + QB - 1) // QB,)

        # GPU time of the body, measured without synchronising inside the frame: the previous pair is
        # complete by now, so reading it costs nothing and charges no foreign work to us.
        import logging as _lg
        _prev = getattr(self, "_ev_pair", None)
        if _prev is not None:
            _s, _e, _n = _prev
            self._last_body_ms = _s.elapsed_time(_e)
            self._gpu_ms = getattr(self, "_gpu_ms", []) + [self._last_body_ms]
            if len(self._gpu_ms) % 8 == 0:
                _v = self._gpu_ms[-8:]
                _lg.getLogger("RVE").warning("int8 body GPU: %.2f ms/frame (last 8), frame %d",
                                             sum(_v) / len(_v), len(self._gpu_ms))
        import os as _os
        _quiet = _os.environ.get("RVE_INT8_NO_EVENTS") == "1"
        if not _quiet and not getattr(self, "_sync_dbg", False):
            self._sync_dbg = True
            try:
                torch.cuda.set_sync_debug_mode("warn")   # names any implicit synchronisation by stack
            except Exception:  # noqa: BLE001
                pass
        _mk = (lambda: torch.cuda.Event(True)) if not _quiet else (lambda: _NoEvent())
        _prevp = getattr(self, "_phase_ev", None)
        if _prevp is not None:
            _names = ("head", "quantise", "hops", "tail")
            _vals = [_a.elapsed_time(_b) for _a, _b in _prevp]
            self._phase_log = getattr(self, "_phase_log", 0) + 1
            if self._phase_log % 8 == 0:
                _lg.getLogger("RVE").warning(
                    "int8 phases: head %.1f  quantise %.1f  hops %.1f  tail %.1f  SUM %.1f  "
                    "BODY-TOTAL %.1f ms (gap %.1f)",
                    _vals[0], _vals[1], _vals[2], _vals[3], sum(_vals),
                    getattr(self, "_last_body_ms", -1.0),
                    getattr(self, "_last_body_ms", 0.0) - sum(_vals))
        _prevt = getattr(self, "_tail_ev", None)
        if _prevt is not None:
            _tv = [_a.elapsed_time(_b) for _a, _b in _prevt]
            self._tail_log = getattr(self, "_tail_log", 0) + 1
            if self._tail_log % 8 == 0:
                _lg.getLogger("RVE").warning(
                    "int8 tail: dequant %.2f  conv %.2f  shuffle %.2f  residual %.2f  SUM %.2f ms",
                    _tv[0], _tv[1], _tv[2], _tv[3], sum(_tv))
        _e0, _e1, _e2, _e3, _e4 = (_mk() for _ in range(5))
        _t1, _t2, _t3 = (_mk() for _ in range(3))
        _e0.record(); _start = _e0; _end = _e4
        with torch.no_grad():
            if self.head_fused:
                # One kernel from the frame to the codes: conv, bias, PReLU and the requant, so the
                # separate quantise pass - and the 265 MB fp16 activation it exists to compress -
                # disappear. The HEAD line reads ~0 and the QUANTISE line carries the kernel.
                buf = self._int8_buffers(x, h, w)[0]
                head_int8_kernel[grid](x.clamp(0.0, 1.0), self._head_w32, self._head_bias,
                                       self._head_slope, self._s_t1, buf, h, w, BLOCK_M=self.block_m,
                                       C=self.feat, K=32, CIN=3, num_warps=self.warps)
                if not getattr(self, "_hf_log", False):
                    self._hf_log = True
                    _lg.getLogger("RVE").warning(
                        "int8 head FUSED: one kernel writes the codes (K=32 from a %d-channel head, "
                        "no separate quantise pass)", self._head_conv_plain.in_channels)
                _e1.record()
                _e2.record()
                _, self._scratch_a, self._scratch_b = self._int8_buffers(x, h, w)
            else:
                head = self.head_conv(x.clamp(0.0, 1.0))  # PReLU is fused into the quantise below
                _e1.record()
                # The fused Triton conversion is CORRECT once fed a contiguous tensor (the PReLU output is
                # not, and the kernel indexes raw memory as ch * n_pix + pix - that mismatch was the 98.34%
                # of differing codes), but making it contiguous costs more than the four torch passes it
                # replaces: 89.25 ms against 72.49 ms for this path, both certified at 0.00000%. So the
                # faster of the two correct versions is the torch one, and the measured win of the earlier
                # fusion was entirely the wrong codes, not the fusion.
                if not getattr(self, "_blog", False):
                    self._blog = True
                    _lg.getLogger("RVE").warning(
                        "int8 quantise branch: channels_last=%s contiguous=%s strides=%s shape=%s QB=%s",
                        head.is_contiguous(memory_format=torch.channels_last), head.is_contiguous(),
                        head.stride(), tuple(head.shape), QB)
                if head.is_contiguous(memory_format=torch.channels_last):
                    # the bytes are already NHWC: read them where they are, no transpose, no copy
                    buf = self._int8_buffers(x, h, w)[0]
                    quantise_nhwc_prelu[qgrid](head, self._s_t1, self._slope_half, buf, n_pix,
                                               BLOCK=QB, C=self.feat, num_warps=4)
                else:
                    # the head is plain contiguous NCHW here (the app's conv output). PReLU is FUSED into the
                    # conversion: F.prelu on this tensor allocates 265 MB per frame and the phase measured
                    # 20.2 ms against 1.6 ms for the kernel alone on the very same input.
                    buf = self._int8_buffers(x, h, w)[0]
                    # F.prelu allocates ~265 MB a frame and the phase measured 20.2 ms against 1.6 ms for the
                    # conversion alone. Both attempts to avoid that allocation (a fused Triton kernel, and
                    # torch.where with a reused `out=`) changed the picture to 2.15278% - identical in both,
                    # so the difference is in how PReLU is applied, not in the conversion, and it is not yet
                    # explained. Correctness first: this path certifies at 0.00823%, so it stays until the
                    # allocation cost is removed without moving the picture.
                    # F.prelu is ONE fused ATen kernel (read 265 + write 265 = 530 MB, 1.84 ms). The
                    # torch.where replacement I put here was three eager passes (1856 MB, ~6.4 ms) and made
                    # the phase 9.91 ms against 4.01 - it removed allocations that were already free
                    # (num_alloc_retries = 0) and paid for them with two extra sweeps. Measured A/B:
                    #   F.prelu + quantise      4.01 ms   codes identical to the reference
                    #   torch.where + quantise  9.91 ms   codes identical, 2.5x slower
                    #   fused Triton            1.71 ms   codes WRONG on 34.7% (max 205) - still broken
                    # FUSED head conversion: PReLU + quantisation in one pass (read 265 + write 133 = 398 MB)
                    # against F.prelu (530 MB) plus the conversion (398 MB). Five controlled tests - three
                    # sizes, all-positive slopes, mixed signs, and the model's own slope vector - agree with
                    # the two-pass version on 99.97% of codes (max 1). certify() is the authoritative check.
                    quantise_nchw_to_nhwc_prelu[qgrid](head, self._s_t1, self._slope_half, buf, n_pix,
                                                       BLOCK=QB, C=self.feat, num_warps=4)
                # a FRESH destination per hop, exactly as the validated recipe does. Reusing a ping-pong pair
                # measured 1.27525% at 512x512 where this measured 0.00000% on the same frame, the same grids
                # and the same launch parameters - the aliasing was the whole difference.
                # pre-allocated scratch: sixteen 132 MB allocations per frame measured ~22 ms of GPU time in
                # the app (70.8 ms there against 48.6 in the harness) where the allocator had to grow against a
                # fragmented pool. The destination cycles between two buffers and never returns to `buf`.
                key = (x.shape[0], h, w)
                _, self._scratch_a, self._scratch_b = self._int8_buffers(x, h, w)
                _e2.record()
            src, nxt = buf, self._scratch_a
            import time as _time
            _host_launch = 0.0
            _host_total0 = _time.perf_counter()
            if self.hop_halo:
                # zero-haloed buffers: hop 0 reads the plain quantised codes and writes a padded interior,
                # hops 1..14 pad-to-pad, the LAST hop writes the plain buffer the tail reads - so no pass
                # exists anywhere for the halo, only one border zeroing per frame
                consts, _ = self._halo_buffers(x, h, w)
                if not getattr(self, "_halo_log", False):
                    self._halo_log = True
                    _lg.getLogger("RVE").warning(
                        "int8 hops: ZERO-HALO buffers in use - %d launches unmasked, one border zeroing/frame",
                        len(self._launches))
                for k, launch in enumerate(self._launches):
                    if k == 0:
                        src_h, dst_h, rp, wp = buf, consts[0], 0, 1
                    elif k == len(self._launches) - 1:
                        src_h, dst_h, rp, wp = src, buf, 1, 0
                    else:
                        src_h, dst_h, rp, wp = src, consts[k % 3], 1, 1
                    _h0 = _time.perf_counter()
                    hop_padded_kernel[grid](src_h, launch["w"], launch["mn"], SHIFT, launch["vec"],
                                            launch["bias"], launch["slope"], launch["os"], dst_h, h, w,
                                            BLOCK_M=self.block_m, C=self.feat, READ_PADDED=rp,
                                            WRITE_PADDED=wp, num_warps=self.warps)
                    _host_launch += _time.perf_counter() - _h0
                    src = dst_h
            else:
                for launch in self._launches:
                    dst = nxt
                    _h0 = _time.perf_counter()
                    body_prologue_kernel[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"],
                                               launch["bias"], launch["slope"], launch["os"], dst, h, w,
                                               BLOCK_M=self.block_m, C=self.feat, num_warps=self.warps,
                                               FOLD=self.fold)
                    _host_launch += _time.perf_counter() - _h0
                    src = dst
                    nxt = self._scratch_b if dst is self._scratch_a else self._scratch_a
            if not self.head_fused and not getattr(self, "_probe_done", False):
                self._probe_done = True
                # the same kernel, alone, in the app's own process: is it slow or is it waiting?
                # (not run when the head is fused: there is no separate quantise pass to time)
                _cs, _ce = torch.cuda.Event(True), torch.cuda.Event(True)
                for _ in range(3):
                    quantise_nchw_to_nhwc[qgrid](head, s_act.contiguous(), buf, n_pix,
                                                 BLOCK=QB, C=self.feat, num_warps=4)
                torch.cuda.synchronize()
                _cs.record()
                for _ in range(5):
                    quantise_nchw_to_nhwc[qgrid](head, s_act.contiguous(), buf, n_pix,
                                                 BLOCK=QB, C=self.feat, num_warps=4)
                _ce.record(); torch.cuda.synchronize()
                _lg.getLogger("RVE").warning(
                    "int8 quantise ALONE: %.2f ms/call  (allocated %.0f MB, reserved %.0f MB, peak %.0f MB)",
                    _cs.elapsed_time(_ce) / 5.0,
                    torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20,
                    torch.cuda.max_memory_allocated() / 2**20)
            _host_loop = (_time.perf_counter() - _host_total0) * 1000.0
            self._host_log = getattr(self, "_host_log", 0) + 1
            if self._host_log % 8 == 0:
                _lg.getLogger("RVE").warning(
                    "int8 host: 16 launches cost %.2f ms of HOST (%.3f ms each); whole hop loop %.2f ms",
                    _host_launch * 1000.0, _host_launch * 1000.0 / 16.0, _host_loop)
            _e3.record()
            # The input's memory format is worth 2.5x for this 64 -> 12 layer: 5.296 ms contiguous NCHW
            # against 2.098 through a channels_last view, measured in isolation
            # (scripts/tail_conv_probe.py), and C_out 12 -> 16 changes nothing, so it is the VIEW and not
            # tensor-core tile fill. Converting the WEIGHTS instead also speeds the conv up but moved the
            # shuffle and residual from 0.40 + 0.93 to 0.82 + 2.13 ms, so it is not the lever it looks like.
            # RVE_TAIL_NHWC=0 restores the old contiguous path for an A/B.
            channels_last = str(os.environ.get("RVE_TAIL_NHWC", "1")).lower() not in ("0", "false", "no")
            if self.tail_shuffle:
                # conv + shuffle + residual, one launch. Same two additions on the same 12 values in the same
                # order, so this is a fold and not an approximation - byte-identity is the acceptance.
                _t1.record()
                if not getattr(self, "_ts_log", False):
                    self._ts_log = True
                    _lg.getLogger("RVE").warning(
                        "int8 tail FUSED with shuffle+residual: one kernel, scale %d", self.tail_scale)
                if self.out_yuv:
                    # conv + shuffle + residual + BT.709 limited-range colour, one kernel, straight into
                    # the pipe's yuv420p layout: 6 bytes per input pixel, no second pass and no cat().
                    # 4 Y in the 2H x 2W plane, then U and V in two H x W planes.
                    if not getattr(self, "_ty_log", False):
                        self._ty_log = True
                        _lg.getLogger("RVE").warning(
                            "int8 tail writes the payload: yuv420p %.2f MB directly from the epilogue "
                            "(no fp16 RGB intermediate, pack copies instead of converting)",
                            6.0 * n_pix / 2**20)
                    out = torch.empty(6 * n_pix, device=x.device, dtype=torch.uint8)
                    tail_int8_yuv_kernel[grid](src, self._tail_taps, self._tail_sigma, self._tail_bias,
                                               x.clamp(0.0, 1.0), out, h, w, BLOCK_M=self.block_m,
                                               CIN=self.feat, COUT=self.tail_conv.out_channels,
                                               COUT_PAD=16, SCALE=self.tail_scale,
                                               num_warps=self.warps)
                else:
                    final = torch.empty((x.shape[0], h * self.tail_scale, w * self.tail_scale, 3),
                                        device=x.device, dtype=torch.float16)
                    tail_int8_shuffle_kernel[grid](src, self._tail_taps, self._tail_sigma,
                                                   self._tail_bias, x.clamp(0.0, 1.0), final, h, w,
                                                   BLOCK_M=self.block_m, CIN=self.feat,
                                                   COUT=self.tail_conv.out_channels, COUT_PAD=16,
                                                   SCALE=self.tail_scale, num_warps=self.warps)
                    out = final.permute(0, 3, 1, 2)      # a channels_last view, no copy
                _t2.record()
                _t3.record()
                _end.record()
                self._tail_ev = ((_e3, _t1), (_t1, _t2), (_t2, _t3), (_t3, _e4))
                self._ev_pair = (_start, _end, 0)
                self._phase_ev = ((_e0, _e1), (_e1, _e2), (_e2, _e3), (_e3, _e4))
                return out
            if self.tail_int8:
                # One kernel instead of two: the dequant scale is already inside the weights, so the
                # last hop's codes are the tail convolution's input and the pass between them is gone.
                # The fused kernel lands on the CONV line; the dequant line reads ~0 by construction.
                _t1.record()
                tail_out = torch.empty((x.shape[0], h, w, self.tail_conv.out_channels),
                                       device=x.device, dtype=torch.float16)
                tail_int8_kernel[grid](src, self._tail_taps, self._tail_sigma, self._tail_bias, tail_out,
                                       h, w, BLOCK_M=self.block_m, CIN=self.feat,
                                       COUT=self.tail_conv.out_channels, COUT_PAD=16,
                                       num_warps=self.warps)
                _conv = tail_out.permute(0, 3, 1, 2)      # a channels_last view, no copy
                _t2.record()
            else:
                if channels_last:
                    last_nhwc = torch.empty((x.shape[0], h, w, self.feat), device=x.device,
                                            dtype=torch.float16)
                    dequantise_nhwc[qgrid](src, self._launches[-1]["os"].contiguous(), last_nhwc, n_pix,
                                           BLOCK=QB, C=self.feat, num_warps=4)
                    last = last_nhwc.permute(0, 3, 1, 2)  # a channels_last view, no copy
                else:
                    last = torch.empty((x.shape[0], self.feat, h, w), device=x.device, dtype=torch.float16)
                    dequantise_nhwc_to_nchw[qgrid](src, self._launches[-1]["os"].contiguous(), last, n_pix,
                                                   BLOCK=QB, C=self.feat, num_warps=4)
                _t1.record()
                _conv = self.tail_conv(last)
                _t2.record()
            out = self.model.upsampler(_conv)
            _t3.record()
            out = out + torch.nn.functional.interpolate(x.clamp(0.0, 1.0), scale_factor=self.scale,
                                                        mode="nearest")
            _end.record()
            self._tail_ev = ((_e3, _t1), (_t1, _t2), (_t2, _t3), (_t3, _e4))
        self._ev_pair = (_start, _end, 0)
        self._phase_ev = ((_e0, _e1), (_e1, _e2), (_e2, _e3), (_e3, _e4))
        return out

    # ---- self-certification: the body has to prove itself on the frame it calibrated on
    def certify(self, x, bar: float = FIRST_FRAME_SANITY):
        """Share of pixels this body puts more than 8 levels away from the fp16 model, on `x`.

        This is a GROSS-ERROR gate on ONE frame, not a picture acceptance. What it exists for: at
        1920x1080 a body whose grid does not cover the activations reads **3.53%** against the validated
        pipeline's 0.00004%, and 0.00000% at 256x256 with 1.23% at 512x512 - an error that grows with
        resolution, which is the signature of a coverage bug. A body that cannot clear a gross gate on its
        own calibration frame must not serve a frame, however fast it is.

        What it cannot do, and used to pretend to: certify the picture. The project's acceptance statistic
        is `bar_share` = 0.1% of pixels on the WORST frame of a spread sample (n>=2); one frame has no
        population, and on this material the number it produces is dominated by which frame the clip opens
        on (0.00000% on BD Violet's frame 1 against a whole-clip 0.53% body-vs-engine). Borrowing the
        acceptance bar here also risked the failure class this project has already paid for once - a
        correct body silently refusing on a hard first frame. The picture is judged on renders.
        """
        with torch.no_grad():
            ref = self.model(x.clamp(0.0, 1.0)).float()
            # the payload path returns bytes, not an RGB tensor: the gross gate needs the tensor, so
            # turn the packing off for the one call that compares against the model
            _yuv, self.out_yuv = self.out_yuv, False
            try:
                got = self(x).float()
            finally:
                self.out_yuv = _yuv
        diff = ((got - ref).abs() * 255.0).amax(dim=1)
        share = float((diff > 8).float().mean()) * 100.0
        return share, self.sanity_ok(share, bar)

    @staticmethod
    def sanity_ok(share: float, bar: float = FIRST_FRAME_SANITY) -> bool:
        """Would a body reading `share` on its first frame be allowed to serve?

        Pure, so the separation can be pinned without a GPU. The numbers are the record's: valid bodies
        read 0.00000-0.01917% against the model (and 0.28825% with a deliberately broken grid margin
        x1.5), a coverage bug reads 3.53% at 1080p, and two broken correction experiments read 29.5% and
        100%. The gate sits at 1.0% - ~3.5x above the worst valid reading and ~3.5x below the smallest
        observed bug.
        """
        return share <= bar

    # ---- a helper the render can ask for a one-line verdict, as TileProbe does
    @staticmethod
    def supports(model):
        """The coverage boundary: 16 inner convs, a tail, 64 body channels - head wrapped or not."""
        try:
            convs = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
            head = list(model.body)[0]
            rest = convs[1:] if isinstance(head, torch.nn.Conv2d) else convs
            return len(rest) == 17 and rest[0].out_channels == 64
        except Exception:  # noqa: BLE001
            return False

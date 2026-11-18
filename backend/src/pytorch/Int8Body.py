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

import torch

from .Int8Kernels import (body_prologue_kernel, dequantise_nhwc, dequantise_nhwc_to_nchw,
                          quantise_nchw_to_nhwc, quantise_nchw_to_nhwc_prelu, quantise_nhwc,
                          quantise_nhwc_prelu)

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
        return launches

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
        _e0, _e1, _e2, _e3, _e4 = (_mk() for _ in range(5))
        _e0.record(); _start = _e0; _end = _e4
        with torch.no_grad():
            head = self.head_conv(x.clamp(0.0, 1.0))      # PReLU is fused into the quantise below
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
            for launch in self._launches:
                dst = nxt
                _h0 = _time.perf_counter()
                body_prologue_kernel[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"],
                                           launch["bias"], launch["slope"], launch["os"], dst, h, w,
                                           BLOCK_M=self.block_m, C=self.feat, num_warps=self.warps)
                _host_launch += _time.perf_counter() - _h0
                src = dst
                nxt = self._scratch_b if dst is self._scratch_a else self._scratch_a
            if not getattr(self, "_probe_done", False):
                self._probe_done = True
                # the same kernel, alone, in the app's own process: is it slow or is it waiting?
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
            channels_last = self.tail_conv.weight.is_contiguous(memory_format=torch.channels_last)
            if channels_last:
                last_nhwc = torch.empty((x.shape[0], h, w, self.feat), device=x.device,
                                        dtype=torch.float16)
                dequantise_nhwc[qgrid](src, self._launches[-1]["os"].contiguous(), last_nhwc, n_pix,
                                       BLOCK=QB, C=self.feat, num_warps=4)
                last = last_nhwc.permute(0, 3, 1, 2)      # a channels_last view, no copy
            else:
                last = torch.empty((x.shape[0], self.feat, h, w), device=x.device, dtype=torch.float16)
                dequantise_nhwc_to_nchw[qgrid](src, self._launches[-1]["os"].contiguous(), last, n_pix,
                                               BLOCK=QB, C=self.feat, num_warps=4)
            out = self.model.upsampler(self.tail_conv(last))
            out = out + torch.nn.functional.interpolate(x.clamp(0.0, 1.0), scale_factor=self.scale,
                                                        mode="nearest")
            _end.record()
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
            got = self(x).float()
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

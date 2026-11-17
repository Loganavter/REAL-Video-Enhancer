"""Decide, before the first frame, whether the int8 body may serve this video.

The body is the sixteen 64-channel convolutions between the head and the tail. In int8 they run 1.49x
faster than the fp16 TensorRT engine (measured, one session, NVENC alongside), but the acceptance bar is
about pictures, and the picture depends on the material: on title-card-like content the body's own error
is 0.009% of pixels, on the Violet pair it is 0.057% while *composition* - what the temporal gate
actually shows when a frame is served from an int8 reference one frame old - is 0.170%.

So the probe answers the composition question, not the engine's own defect. Certifying on the engine's
own number would have shipped both Violet titles: measured, their own defect passes the bar and their
composition does not.

Two things this module deliberately is not:

* it is not a runtime switch. The verdict is taken once, here, and the render guards on it - an engine
  swapped mid-render would mix two error histories inside one gate window, which is the one thing the
  gate cannot survive;
* it is not dependent on the offline `scripts/` tooling. The calibration and the quantised forward are
  vendored below, because `scripts/` is not shipped with the app and a probe that imports its measuring
  stick from a directory that is not there fails in the field, not in a test.

A probe failure must never cost a render: every exception path returns a verdict of "no".

Usage from the render path: ``probe(clip, model_path, width, height)`` -> ``Int8ProbeResult``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

try:                      # the probe needs torch; without it the verdict is simply "no"
    import torch
except ImportError:       # pragma: no cover - only in a torch-less cell
    torch = None  # type: ignore[assignment]

__all__ = ["Int8ProbeResult", "probe"]

FF = "ffmpeg"
GATE_THRESHOLD = 5.0   # the render's temporalReuseThreshold default, in 8-bit levels


@dataclass
class Int8ProbeResult:
    use_int8: bool
    engine: float          # worst share of pixels >8 levels off, int8(t) vs fp16(t)
    composition: float     # worst share, int8(t-1) vs fp16(t) - the number the bar sees
    peak: float
    pairs: int
    bar: float
    reason: str
    frames_used: int = 0
    error: str = ""
    frames: object = None      # the decoded calibration frames, so the body reuses them instead of re-decoding
    # the depth-2 pair of numbers and the control that owns them. `composition_d2` is what the verdict is
    # actually taken on when the gate serves depth 2; `stale_d1`/`stale_d2` are the same statistic with
    # fp16 on both sides, i.e. the gate's own reference without the quantiser. Kept as fields, not only in
    # the reason string: a number that exists only in prose cannot be recorded.
    composition_d2: float = float("nan")
    stale_d1: float = float("nan")
    stale_d2: float = float("nan")

    def stats(self) -> str:
        if self.error:
            return f"int8 probe unavailable ({self.error}); int8 stays off"
        return (f"int8 probe: engine {self.engine:.3f}%  composition {self.composition:.3f}%  "
                f"peak {self.peak:.0f}  ({self.pairs} pairs, bar {self.bar}%) -> "
                f"{'int8' if self.use_int8 else 'fp16'}")


def _read_pairs(clip, count, w, h):
    """Adjacent pairs, scaled to w x h.

    The scale filter is not cosmetic: these sources are 3840x2160, and a reader that assumes 1920x1080
    takes the first w*h*3 bytes of a 4K frame and reinterprets them as a 1080p picture. That read 4.1%
    engine error on every title where the same clip measures 0.0086% - it would have returned a pass
    rate of zero and disabled the path everywhere.
    """
    import numpy as np
    import torch

    probe = subprocess.run([FF, "-hide_banner", "-i", clip], capture_output=True, text=True)
    line = [ln for ln in probe.stderr.splitlines() if "Duration" in ln]
    if not line:
        raise RuntimeError("cannot read duration")
    hh, mm, ss = line[0].split("Duration:")[1].split(",")[0].strip().split(":")
    duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
    size = w * h * 3
    out = []
    for i in range(count):
        t = duration * (i + 0.5) / count
        argv = [FF, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", clip,
                "-frames:v", "3", "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        raw = subprocess.run(argv, capture_output=True).stdout
        if len(raw) < 3 * size:
            continue
        pair = []
        for k in range(3):
            img = np.frombuffer(raw[k * size:(k + 1) * size], dtype=np.uint8)
            img = img.reshape(h, w, 3).astype("float32") / 255.0
            pair.append(torch.from_numpy(img).permute(2, 0, 1)[None])
        out.append(pair)
    return out


def _quantise_weights(state):
    out, scales = {}, {}
    for name, tensor in state.items():
        if tensor.ndim == 4:
            scale = (tensor.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0).clamp_min(1e-8)
            out[name] = torch.round(tensor / scale).clamp(-127, 127) * scale
            scales[name] = scale
        else:
            out[name] = tensor
    return out, scales


def _calibrate(model, frames):
    """Per-tensor input ranges per conv, per-channel output ranges per PReLU, on the quantised model."""
    qstate, _ = _quantise_weights(model.state_dict())
    model.load_state_dict(qstate)
    inputs, outputs = {}, {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def pre_hook(_m, inp, key=name):
                lo, hi = float(inp[0].min()), float(inp[0].max())
                prev = inputs.get(key)
                inputs[key] = (min(prev[0], lo), max(prev[1], hi)) if prev else (lo, hi)
            handles.append(module.register_forward_pre_hook(pre_hook))
        if isinstance(module, torch.nn.PReLU):
            def post_hook(_m, _i, out, key=name):
                pair = (out.amin(dim=(0, 2, 3)), out.amax(dim=(0, 2, 3)))
                prev = outputs.get(key)
                outputs[key] = pair if prev is None else (torch.minimum(prev[0], pair[0]),
                                                          torch.maximum(prev[1], pair[1]))
            handles.append(module.register_forward_hook(post_hook))
    with torch.no_grad():
        for frame in frames:
            model(frame)
    for handle in handles:
        handle.remove()
    return inputs, outputs


def _quantised_forward(model, x, inputs, outputs):
    import torch.nn.functional as F

    x = x.clamp(0.0, 1.0)
    out = x
    for name, module in model.body.named_children():
        if isinstance(module, torch.nn.Conv2d):
            lo, hi = inputs[f"body.{name}"]
            s_in = max(abs(lo), abs(hi)) / 127.0
            out = torch.round(out / s_in).clamp(-127, 127) * s_in
            out = module(out)
        elif isinstance(module, torch.nn.PReLU):
            out = module(out)
            lo, hi = outputs[f"body.{name}"]
            s_out = (torch.maximum(lo.abs(), hi.abs()) / 127.0).view(1, -1, 1, 1).clamp_min(1e-8)
            out = torch.round(out / s_out).clamp(-127, 127) * s_out
        else:
            out = module(out)
    up = model.upsampler(out)
    return up + F.interpolate(x, scale_factor=model.upscale, mode="nearest")


def _statistic(a, b, bar=8.0):
    diff = (a.float() - b.float()).abs() * 255.0
    per_px = diff.amax(dim=1)
    return float((per_px > bar).float().mean()) * 100.0, float(per_px.max())


def probe(clip: str, model_path: str, width: int, height: int, frames: int = 16,
          bar: float = 0.1) -> Int8ProbeResult:
    """Would int8 fit this video? Composition first, own defect second, bar = 0.1% of pixels."""
    try:
        import torch
        from ..pytorch.spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact

        pairs = _read_pairs(clip, frames, width, height)
        if len(pairs) < 3:
            return Int8ProbeResult(False, 0.0, 0.0, 0.0, 0, bar, "not enough frames", error="too few pairs")

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        state = state.get("params", state)
        tail = [k for k in state if k.endswith("34.weight")][0]
        scale = int(round((state[tail].shape[0] / 3) ** 0.5))
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale)
        model.load_state_dict(state)
        model = model.eval().cuda().half()

        calib_idx = [i for i in range(len(pairs)) if i % 2 == 0]
        test_idx = [i for i in range(len(pairs)) if i % 2 == 1]
        frames_all = [[f.cuda().half() for f in pair] for pair in pairs]

        # a pair only reaches the composition if the GATE would reuse it: its own block-mean statistic
        # has to clear the threshold. Without this filter the probe measures scene cuts, where nothing is
        # reused - it read 99.7% composition (peak 255) on a clip whose real composition is 0.02%.
        # Taken BEFORE calibration: it reads the frames, not the weights, and the fp16 control below
        # needs the pair sets to compute its own statistic one pair at a time.
        def gate_delta(fa, fb):
            def blocks(f):
                import torch.nn.functional as F
                return F.avg_pool2d(f.float().mean(dim=1, keepdim=True), 32, ceil_mode=True) * 255.0
            return float((blocks(fb) - blocks(fa)).abs().max())

        reused = [i for i in test_idx if gate_delta(frames_all[i][0], frames_all[i][1]) <= GATE_THRESHOLD]
        # depth 2 only over pairs the gate would reuse at BOTH steps: checking one delta and then
        # comparing frames two apart measures motion, not quantisation (it read 10.7% on Violet).
        reused2 = [i for i in reused if len(frames_all[i]) > 2
                   and gate_delta(frames_all[i][1], frames_all[i][2]) <= GATE_THRESHOLD]

        # The discriminating control for the depth-2 blocker needs no int8: on the SAME pairs, at the
        # SAME separation, compare fp16 against fp16. The render is index 0 and the references are 1 and
        # 2, exactly as in `comp` and `comp2`, so only the renderer changes. If the gate's own reference
        # already reads ~0.2% two steps out, the blocker is the gate's certification (E25's depth model),
        # not the quantisation - and no int8 work moves it.
        # It is also the population the gate stores: fp16 arithmetic on the ORIGINAL weights. `_calibrate`
        # loads int8-rounded weights into `model`, so `ref` below is fp16 on quantised weights.
        # One pair at a time: a full copy of every output is ~200 MB per 4K triple, and holding it for
        # the whole sample is what made the spread sample unreachable (see the note on the loop below).
        stale1, stale2 = [], []
        with torch.no_grad():
            for i in sorted(set(reused) | set(reused2)):
                outs = [model(f).float() for f in frames_all[i]]
                if i in reused:
                    stale1.append(_statistic(outs[1], outs[0])[0])
                if i in reused2:
                    stale2.append(_statistic(outs[2], outs[0])[0])
                del outs
        stale1_worst = max(stale1, default=float("nan"))
        stale2_worst = max(stale2, default=float("nan"))

        inputs, outputs = _calibrate(model, [f for i in calib_idx for f in frames_all[i]])
        inputs["body.0"] = (0.0, 1.0)      # the head's input range is the clamp's own, not a measurement
        # One pair at a time, and the three statistics are taken inside the loop: holding `ref` and `got`
        # for every pair cost ~200 MB per decoded triple at 4K, so a 16-triple sample already sat at
        # 11.2 GiB and a 32-triple one OOMed. The protocol's spread sample (13-19) was unreachable for
        # that reason alone, not for want of a GPU.
        engine, comp, comp2 = [], [], []
        with torch.no_grad():
            for i, pair in enumerate(frames_all):
                outs_ref = [model(f).float() for f in pair]
                outs_got = [_quantised_forward(model, f, inputs, outputs).float() for f in pair]
                if i in test_idx:
                    engine.append(_statistic(outs_ref[1], outs_got[1]))
                if i in reused:
                    comp.append(_statistic(outs_ref[1], outs_got[0]))
                if i in reused2:
                    comp2.append(_statistic(outs_ref[2], outs_got[0]))
                del outs_ref, outs_got
        if not engine:
            return Int8ProbeResult(False, 0.0, 0.0, 0.0, 0, bar, "no test pairs", error="no test pairs")
        engine_worst = max(s for s, _ in engine)
        if not comp:
            return Int8ProbeResult(False, engine_worst, 0.0, 0.0, 0, bar,
                                   "no pair the gate would reuse", frames_used=len(pairs))
        comp_worst = max(s for s, _ in comp)
        peak = max(p for _, p in comp)
        c2 = max((s for s, _ in comp2), default=float("nan"))
        _n2 = len(reused2)
        # The gate serves references up to its chain depth, so the verdict is taken at the WORST depth it may
        # serve - certifying at depth 1 and serving at depth 2 is the blind spot this project already hit once.
        worst_depth = comp_worst if not comp2 else max(comp_worst, c2)
        verdict = worst_depth <= bar
        reason = (f"composition d1 {comp_worst:.3f}% / d2 {c2:.3f}% -> worst {worst_depth:.3f}% "
                  f"{'<=' if verdict else '>'} bar {bar}% (own defect {engine_worst:.3f}%, "
                  f"{len(comp)} of {len(test_idx)} pairs at d1, {_n2} at d2); "
                  f"fp16 reference on the same pairs: d1 {stale1_worst:.3f}% / d2 {stale2_worst:.3f}%")
        return Int8ProbeResult(verdict, engine_worst, comp_worst, peak, len(comp), bar, reason,
                               frames_used=len(pairs),
                               frames=[f for i in calib_idx for f in frames_all[i]],
                               composition_d2=c2, stale_d1=stale1_worst, stale_d2=stale2_worst)
    except Exception as error:  # noqa: BLE001 - a probe failure must never cost a render
        return Int8ProbeResult(False, 0.0, 0.0, 0.0, 0, bar, "probe failed", error=str(error))

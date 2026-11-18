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

import os

import subprocess
from dataclasses import dataclass

try:                      # the probe needs torch; without it the verdict is simply "no"
    import torch
except ImportError:       # pragma: no cover - only in a torch-less cell
    torch = None  # type: ignore[assignment]

__all__ = ["Int8ProbeResult", "probe"]

FF = "ffmpeg"
GATE_THRESHOLD = 5.0   # the render's temporalReuseThreshold default, in 8-bit levels.
# NOTE: the auto tuner clamps its own choice to (2.0, 4.0), so a probe filtered at 5.0 certifies on pairs
# the shipped gate can never serve - a superset. Callers that know the gate's threshold should pass it.


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
    # adversarial selection: the worst composition over the top-k reuse events of the whole clip,
    # ranked by the gate's own statistic. The spread sample cannot reach these by construction.
    adversarial: float = float("nan")
    adversarial_control: float = float("nan")
    adversarial_frames: int = 0

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


def _read_indices(clip, indices, w, h):
    """The wanted frames, by index, in ONE streaming pass.

    `select=eq(n,index)` is not usable here: with B-frames the filter's counter does not agree with
    the pipe's frame order, and 12 of 13 pairs came back as different frames (their measured share
    was above the gate's threshold, which is how it was caught). Counting the stream is exact.
    """
    import subprocess

    import numpy as np
    import torch

    wanted = set(int(i) for i in indices)
    size = w * h * 3
    argv = [FF, "-hide_banner", "-loglevel", "error", "-i", clip, "-vf", f"scale={w}:{h}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE)
    out, index = {}, -1
    while wanted:
        raw = proc.stdout.read(size)
        if len(raw) < size:
            break
        index += 1
        if index in wanted:
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).astype("float32") / 255.0
            out[index] = torch.from_numpy(img).permute(2, 0, 1)[None]
            wanted.discard(index)
    proc.stdout.close()
    proc.wait()
    return out


def _scan_reuse_events(clip, w, h, gate_threshold, max_chain, limit=64):
    """Every frame the gate would serve from a reference, ranked by the gate's own statistic.

    The whole clip, low resolution, no model: the gate is cheap and the ranking is what the bar's
    worst frame lives on. Returns [(frame, reference, share)] worst first.
    """
    import subprocess

    import numpy as np
    import torch

    from .TemporalReuse import ReferenceFrameGate

    gate = ReferenceFrameGate(threshold=gate_threshold, max_chain=max_chain,
                              statistic="channel_max_share")
    size = w * h * 3
    argv = [FF, "-hide_banner", "-loglevel", "error", "-i", clip, "-vf", f"scale={w}:{h}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE)
    events, kept, index = [], None, -1
    while True:
        raw = proc.stdout.read(size)
        if len(raw) < size:
            break
        index += 1
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        duplicate = gate.is_duplicate(t)
        if duplicate:
            events.append((index, kept, float(gate.last_delta)))
        else:
            kept = index
    proc.stdout.close()
    proc.wait()
    events.sort(key=lambda e: -e[2])
    return events[:limit]


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
          bar: float = 0.1, gate_threshold: float = GATE_THRESHOLD,
          filter_statistic: str = "block_mean", adversarial: int = 0,
          max_chain: int = 3, engine_ref=None) -> Int8ProbeResult:
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
            # The pair filter has to use the same variable the gate decides on, or the probe
            # certifies a population the render never serves. Three statistics have lived here:
            # the gate's (1,2,1)/4 luma block mean, a plain mean(R,G,B) block mean (neither the
            # gate's nor the bar's), and now the bar's own quantity.
            if filter_statistic == "channel_max_share":
                cm = (fb.float() - fa.float()).abs().amax(dim=1)
                return float((cm > 8.0 / 255.0).float().mean()) * 100.0
            def blocks(f):
                import torch.nn.functional as F
                return F.avg_pool2d(f.float().mean(dim=1, keepdim=True), 32, ceil_mode=True) * 255.0
            return float((blocks(fb) - blocks(fa)).abs().max())

        reused = [i for i in test_idx
                  if gate_delta(frames_all[i][0], frames_all[i][1]) <= gate_threshold]
        # depth 2 only over pairs the gate would reuse at BOTH steps: checking one delta and then
        # comparing frames two apart measures motion, not quantisation (it read 10.7% on Violet).
        reused2 = [i for i in reused if len(frames_all[i]) > 2
                   and gate_delta(frames_all[i][1], frames_all[i][2]) <= gate_threshold]

        # The discriminating control for the depth-2 blocker needs no int8: on the SAME pairs, at the
        # SAME separation, compare fp16 against fp16. The render is index 0 and the references are 1 and
        # 2, exactly as in `comp` and `comp2`, so only the renderer changes. If the gate's own reference
        # already reads ~0.2% two steps out, the blocker is the gate's certification (E25's depth model),
        # not the quantisation - and no int8 work moves it.
        #
        # `ref` below is the fp16 model, and `_calibrate` MUTATES `model` into the quantised one, so the
        # original weights are kept aside and the quantised copy goes into a second module. Computing
        # `ref` after `_calibrate` (the previous order) made the reference fp16 arithmetic on
        # int8-rounded weights: peak 4 levels, 0.00000% above 8, not bit-identical - below the bar, but
        # the reference was not what it is labelled, and a 4-level error against defects clustered at
        # 8-16 levels is exactly the sensitive regime for a threshold of 8.
        import copy as _copy

        orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        inputs, outputs = _calibrate(model, [f for i in calib_idx for f in frames_all[i]])
        inputs["body.0"] = (0.0, 1.0)      # the head's input range is the clamp's own, not a measurement
        model_q = _copy.deepcopy(model)    # the quantised twin: `got` must use these weights
        model.load_state_dict(orig_state)  # and `ref` must be the fp16 model, not fp16 on int8 weights
        def reference(x):
            """fp16 reference for one LR frame: the app's own engine when one is passed.

            The probe measures against torch fp16 by default, while the app falls back to a TensorRT
            plan. Measured on one frame the two differ by 0.0116% - small, but the reference should be
            the engine the render actually uses. The engine takes a Frame, so the tensor is wrapped.
            """
            if engine_ref is None:
                return model(x).float()
            from .Frame import Frame
            frame = Frame("tensorrt", width, height, "cuda", 0, False, "float16",
                          high_bitdepth=False)
            frame.set_frame_tensor(x, copy=True)
            return engine_ref(frame).get_frame_tensor().detach().float()

        # One pair at a time, and every statistic - including the fp16 control - is taken inside the
        # loop: holding `ref` and `got` for every pair cost ~200 MB per decoded triple at 4K, so a
        # 16-triple sample already sat at 11.2 GiB and a 32-triple one OOMed. The protocol's spread
        # sample (13-19) was unreachable for that reason alone, not for want of a GPU.
        engine, comp, comp2, stale1, stale2 = [], [], [], [], []
        with torch.no_grad():
            for i, pair in enumerate(frames_all):
                outs_ref = [reference(f) for f in pair]
                outs_got = [_quantised_forward(model_q, f, inputs, outputs).float() for f in pair]
                if i in test_idx:
                    engine.append(_statistic(outs_ref[1], outs_got[1]))
                # MIRRORED, as the gate does it: the frame that is served is the LATER one, and the
                # reference is EARLIER. The previous order rendered index 0 against references 1 and 2,
                # i.e. rendered the earliest frame against later ones - a direction the gate never
                # serves. That is why the fp16 control on these pairs (0.083% on Chainsaw) exceeded the
                # worst of the full fp16 sweep (0.07546%): a max over a subset cannot exceed the max
                # over the set, so the pairs were not a subset of the render's population.
                if i in reused:
                    comp.append(_statistic(outs_ref[0], outs_got[1]))
                    stale1.append(_statistic(outs_ref[0], outs_ref[1])[0])
                if i in reused2:
                    comp2.append(_statistic(outs_ref[0], outs_got[2]))
                    stale2.append(_statistic(outs_ref[0], outs_ref[2])[0])
                del outs_ref, outs_got
        stale1_worst = max(stale1, default=float("nan"))
        stale2_worst = max(stale2, default=float("nan"))
        # Adversarial selection: the spread sample can only find the damage it happens to land on.
        # Scan the whole clip with the gate's own statistic (no model), rank the reuses, and render
        # the top ones. This is the only path that can reach the frame the bar's worst lives on.
        adv, adv_ctl, adv_n = float("nan"), float("nan"), 0
        if adversarial and filter_statistic == "channel_max_share":
            events = _scan_reuse_events(clip, width, height, gate_threshold, max_chain,
                                        limit=int(adversarial))
            adv_vals, ctl_vals, discarded = [], [], 0
            need = [i for f, r, _s in events if r is not None for i in (f, r)]
            cache = _read_indices(clip, need, width, height) if need else {}
            with torch.no_grad():
                for frame_idx, ref_idx, _share in events:
                    if ref_idx is None:
                        continue
                    x_t = cache.get(frame_idx)
                    x_r = cache.get(ref_idx)
                    if x_t is None or x_r is None:
                        continue
                    x_t = x_t.cuda().half()
                    x_r = x_r.cuda().half()
                    # self-check: the scan's bookkeeping must agree with the pixels. A pair whose
                    # measured share is above the gate's threshold is not a reuse the gate would make,
                    # and rendering it would report a scene change as a composition error.
                    lr_share = float((x_r - x_t).abs().amax(dim=1).gt(8.0 / 255.0).float().mean()) * 100.0
                    if lr_share > gate_threshold:
                        discarded += 1
                        continue
                    ref_r = reference(x_r)
                    got_t = _quantised_forward(model_q, x_t, inputs, outputs).float()
                    fp16_t = reference(x_t)
                    adv_vals.append(_statistic(ref_r, got_t)[0])
                    ctl_vals.append(_statistic(ref_r, fp16_t)[0])
            adv_n = len(adv_vals)
            adv = max(adv_vals, default=float("nan"))
            adv_ctl = max(ctl_vals, default=float("nan"))

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
        if adv_n:
            worst_depth = max(worst_depth, adv)
        verdict = worst_depth <= bar
        reason = (f"composition d1 {comp_worst:.3f}% / d2 {c2:.3f}% -> worst {worst_depth:.3f}% "
                  f"{'<=' if verdict else '>'} bar {bar}% (own defect {engine_worst:.3f}%, "
                  f"{len(comp)} of {len(test_idx)} pairs at d1, {_n2} at d2); "
                  f"fp16 reference on the same pairs: d1 {stale1_worst:.3f}% / d2 {stale2_worst:.3f}%")
        if adv_n:
            # A run that had to throw most of its pairs away is not a smaller sample, it is an
            # invalid one: the discard rate is the share of pairs where the scan's bookkeeping and
            # the pixels disagree.
            rate = 100.0 * discarded / max(1, adv_n + discarded)
            reason += (f"; adversarial top-{adv_n} of the clip: {adv:.3f}% "
                       f"(fp16 control {adv_ctl:.3f}%, discarded {discarded} of "
                       f"{adv_n + discarded} = {rate:.0f}%{' - INVALID' if rate > 20 else ''})")
        return Int8ProbeResult(verdict, engine_worst, comp_worst, peak, len(comp), bar, reason,
                               frames_used=len(pairs),
                               frames=[f for i in calib_idx for f in frames_all[i]],
                               composition_d2=c2, stale_d1=stale1_worst, stale_d2=stale2_worst,
                               adversarial=adv, adversarial_control=adv_ctl,
                               adversarial_frames=adv_n)
    except Exception as error:  # noqa: BLE001 - a probe failure must never cost a render
        import traceback as _tb
        if os.environ.get("RVE_PROBE_TRACE"):
            _tb.print_exc()
        return Int8ProbeResult(False, 0.0, 0.0, 0.0, 0, bar, "probe failed", error=str(error))

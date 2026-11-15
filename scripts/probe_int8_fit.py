"""Would int8 fit THIS video? The probe the lane ended up needing, in one script.

The verdict is per video, not per title: one pair in four crossed the bar, and it was the pair whose own
frame was the engine's worst. So the product question is not "is int8 good" but "does it pass here", and
that is cheaper to answer than to argue.

The probe reads N frames spread over the clip, calibrates the int8 scales on a disjoint subset, then
measures the two statistics that matter separately:

    engine       int8(t)   against fp16(t)      - the engine's own error
    composition  int8(t-1) against fp16(t)      - what the bar sees when a frame is reused

The bar is the share of pixels whose worst channel differs by more than 8 levels, under 0.1%. If the
composition passes, int8 runs; if not, the caller falls back to the fp16 engine - which is the default, so
a failed probe costs only its own runtime.

Cost is ~1.5 s of forward passes per video and pays for itself only when the render is longer than
roughly 4 s: the engine win is 1.42x and the probe is 0.8 s, so 0.19*T > 0.8 => T > 4.2 s.

Usage: probe_int8_fit.py <clip> [--frames 16] [--bar 0.1]
"""
import argparse
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/scripts")
from cut_channels_naive import load, statistic  # noqa: E402
from int8_accuracy_sim import calibrate, quantised_forward  # noqa: E402

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
GATE_THRESHOLD = 4.0


def read_pairs(clip, count, w, h, step=0.0417):
    """N ADJACENT pairs spread over the clip. The composition is defined on a pair 1/24 s apart; the
    first version spread single frames over the clip and paired those, which measures a scene change."""
    probe = subprocess.run([FF, "-hide_banner", "-i", clip], capture_output=True, text=True)
    line = [l for l in probe.stderr.splitlines() if "Duration" in l]
    if not line:
        raise SystemExit("cannot read duration")
    hh, mm, ss = line[0].split("Duration:")[1].split(",")[0].strip().split(":")
    duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
    out = []
    size = w * h * 3
    for i in range(count):
        t = duration * (i + 0.5) / count
        # scale to the requested size: the clips here are 3840x2160, and without this the reader takes the
        # first w*h*3 bytes of a 4K frame and reinterprets them as 1080p - a mangled picture that read
        # 4.1% engine error where the same clip measured 0.0086% when decoded properly
        argv = [FF, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", clip,
                "-frames:v", "3", "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        raw = subprocess.run(argv, capture_output=True).stdout
        if len(raw) < 2 * size:
            continue
        pair = []
        for k in range(2):
            chunk = raw[k * size:(k + 1) * size]
            img = np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
            pair.append(torch.from_numpy(img).permute(2, 0, 1)[None])
        out.append(pair)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--bar", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    pairs = read_pairs(args.clip, args.frames, args.width, args.height)
    if len(pairs) < 3:
        raise SystemExit("not enough pairs decoded")
    print(f"{args.clip}: {len(pairs)} adjacent pairs")

    model, scale, _ = load(args.model)
    model = model.eval().cuda().half()
    x = [[f.cuda().half() for f in pair] for pair in pairs]
    with torch.no_grad():
        ref = [[model(f).float() for f in pair] for pair in x]

    # calibrate on every other PAIR, measure on the rest: the split has to be disjoint
    calib_idx = list(range(0, len(x), 2))
    test_idx = [i for i in range(1, len(x), 2)]
    inputs, outputs, _ = calibrate(model, [f for i in calib_idx for f in x[i]])
    inputs["body.0"] = (0.0, 1.0)           # the head's input range is the clamp's own

    with torch.no_grad():
        q = {i: [quantised_forward(model, f, inputs, outputs, {}, asym=False).float() for f in x[i]]
             for i in test_idx}

    # a pair only reaches the composition if the GATE would reuse it: its own block-mean statistic has to
    # clear the threshold. Without this filter the probe measures scene cuts, where nothing is reused.
    def gate_delta(fa, fb):
        """The gate's own statistic: max over 32x32 block means of the luma difference, in 8-bit levels."""
        def blocks(f):
            return torch.nn.functional.avg_pool2d(f.float().mean(dim=1, keepdim=True), 32,
                                                  ceil_mode=True) * 255.0
        return float((blocks(fb) - blocks(fa)).abs().max())

    reused = [i for i in test_idx if gate_delta(x[i][0], x[i][1]) <= GATE_THRESHOLD]
    engine = [statistic(ref[i][1], q[i][1]) for i in test_idx]
    comp = [statistic(ref[i][1], q[i][0]) for i in reused]
    print(f"pairs the gate would reuse: {len(reused)} of {len(test_idx)}")
    e_share = max(s for s, _, _ in engine)
    c_share = max(s for s, _, _ in comp) if comp else float("nan")
    e_peak = max(p for _, p, _ in engine)
    c_peak = max(p for _, p, _ in comp) if comp else float("nan")
    print(f"engine       worst share {e_share:6.3f}%  peak {e_peak:4.0f}")
    print(f"composition  worst share {c_share:6.3f}%  peak {c_peak:4.0f}   (bar {args.bar}%, "
          f"over {len(comp)} gated pairs)")
    verdict = "int8" if c_share <= args.bar else "fp16 fallback"
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "int8" else 2


if __name__ == "__main__":
    sys.exit(main())

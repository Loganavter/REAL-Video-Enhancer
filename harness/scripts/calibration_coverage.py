"""Does calibration need FULL frames? Measure it instead of quoting the field's habit.

The probe calibrates the int8 body on 24 spread 1080p frames - 21.1 s of the 29.3 s it costs, and 49.8
megapixels of material to measure 64 per-channel ranges. Post-training quantisation in the field calibrates
on a small unlabelled set (a few forward passes over crops), and if that holds here the probe's price falls
into the noise and int8 stops being a long-clip-only decision.

The risk this script exists to expose: the grids are per-channel MAXIMA, and a maximum over fewer samples
is biased low. 24 full frames give 49.8 M values per channel; 96 crops of 256x256 give 6.3 M. A lower
ceiling means clipping, which is not a softer error but a different one.

Arms:
  full   24 spread 1080p frames (today's recipe)
  crop   96 crops of 256x256 taken from the same 24 frames (4 per frame, fixed offsets)
  crop-p 96 crops with a p99.99 ceiling instead of the maximum, on the same samples

Scoring: the SAME held-out full frames for every arm, against the model's own fp16 output, share of pixels
>8 levels and peak - the project's own statistic, so the arms are comparable to the record.

Usage:
  python harness/harness/scripts/calibration_coverage.py --clip /tmp/opencode/bd_violet20s.mkv \
      --model "<install>/custom_models/2x-Futsuu-Anime(1).pth" --out docs/wsp/verified/records/int8-calibration-crops.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import torch  # noqa: E402

from harness.core import Population, Record, Store  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend"))
from src.pytorch.Int8Body import Int8Body  # noqa: E402
import src.utils.Int8Probe as _probe  # noqa: E402

FFMPEG = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
_probe.FF = FFMPEG if os.path.exists(FFMPEG) else _probe.FF
from src.utils.Int8Probe import _read_indices  # noqa: E402


def _frame_count(clip: str) -> int:
    """Frames in the clip, from ffprobe - the sample is spread over the whole file."""
    import re as _re
    import subprocess as _sp
    import shutil as _sh
    # the install's bin/ carries ffmpeg only; ffprobe is the host's
    exe = _sh.which("ffprobe") or "/usr/bin/ffprobe"
    out = _sp.run([exe,
                   "-v", "error", "-select_streams", "v:0", "-count_packets",
                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", clip],
                  capture_output=True, text=True)
    m = _re.search(r"\d+", out.stdout or "")
    if m:
        return int(m.group(0))
    raise SystemExit(f"cannot count frames in {clip}: {out.stderr.strip()[:120]}")


def load_model(path: str, scale: int):
    from src.pytorch.spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale)
    model.load_state_dict(state)
    return model.eval().cuda().half()


def crops_of(frames, size: int, per_frame: int):
    """Fixed-offset crops, so the sample is deterministic and a rerun is the same measurement."""
    out = []
    for f in frames:
        h, w = f.shape[-2], f.shape[-1]
        for k in range(per_frame):
            y = ((h - size) * (k + 1)) // (per_frame + 1)
            x = ((w - size) * (k + 1)) // (per_frame + 1)
            out.append(f[..., y:y + size, x:x + size].contiguous())
    return out


def timeit(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--per-frame", type=int, default=4)
    ap.add_argument("--test", default="29,44,274,400")
    # a percentile arm is NOT implemented in Int8Body yet (the grids are maxima); it is the follow-up if
    # the crop arm fails, not something this run can pretend to measure

    ap.add_argument("--out", default="docs/wsp/verified/records/int8-calibration-crops.jsonl")
    ap.add_argument("--log", default="docs/wsp/verified/logs/int8-calibration-crops.log")
    ap.add_argument("--session", default="")
    ap.add_argument("--frames-total", type=int, default=0,
                    help="frame count to spread the sample over; 0 = ask ffprobe")
    args = ap.parse_args(argv)

    session = args.session or time.strftime("s-%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # The sample must be spread over the CLIP, not clustered at its start. This used to be a hardcoded
    # `total = 48`, i.e. every calibration frame came from the first two seconds - a clustered sample wearing
    # a spread label (found 2026-11-19 while pointing the script at a 6-minute source).
    total = args.frames_total or _frame_count(args.clip)
    calib_idx = [round(i * (total - 1) / (args.calib - 1)) for i in range(args.calib)]
    test_idx = [int(t) for t in args.test.split(",")]
    log_lines = [f"# calibration coverage {session} clip={args.clip} crop={args.crop} "
                 f"per_frame={args.per_frame} calib={calib_idx} test={test_idx}"]

    model = load_model(args.model, args.scale)
    cache = _read_indices(args.clip, sorted(set(calib_idx + test_idx)), args.width, args.height)
    # a container can report one more packet than it decodes (495 vs 494 on the BD cuts), so keep only the
    # indices that actually came back instead of indexing past the decoded set
    calib_idx = [i for i in calib_idx if i in cache]
    test_idx = [i for i in test_idx if i in cache]
    if len(calib_idx) < 3 or len(test_idx) < 2:
        raise SystemExit(f"not enough frames decoded: calib {len(calib_idx)}, test {len(test_idx)}")
    frames = [cache[i].cuda().half() for i in calib_idx]
    tests = [cache[i].cuda().half() for i in test_idx]
    crops = [c.cuda().half() for c in crops_of(frames, args.crop, args.per_frame)]
    log_lines.append(f"# pixels: full arm {len(frames)} x {args.width*args.height} = "
                     f"{len(frames)*args.width*args.height/1e6:.1f} Mpx; crop arm {len(crops)} x "
                     f"{args.crop**2} = {len(crops)*args.crop**2/1e6:.1f} Mpx")

    arms = {}
    for name, samples in (("full", frames), ("crop", crops)):
        body = Int8Body(model)
        _, seconds = timeit(lambda b=body, s=samples: b.calibrate(s))
        shares, peaks = [], []
        for t in tests:
            with torch.no_grad():
                ref = model(t.clamp(0.0, 1.0)).float()
                got = body(t).float()
            diff = ((got - ref).abs() * 255.0).amax(dim=1)
            shares.append(float((diff > 8).float().mean()) * 100.0)
            peaks.append(float(diff.max()))
        arms[name] = dict(seconds=seconds, share=max(shares), shares=shares, peak=max(peaks))
        log_lines.append(f"# arm={name} calibrate={seconds:.2f}s worst_share={max(shares):.5f}% "
                         f"peak={max(peaks):.0f} per_frame={[round(s, 5) for s in shares]}")
        print(f"{name:7} calibrate {seconds:6.2f} s   worst share {max(shares):8.5f}%   peak {max(peaks):3.0f}")

    # how far the ranges moved, per channel, per layer: the mechanism behind any difference
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), args.log), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), args.out)
    digest = hashlib.sha256(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                              args.log), "rb").read()).hexdigest()
    store = Store(session=session)
    pop = Population.of("clips", [os.path.basename(args.clip)],
                        f"{os.path.basename(args.clip)}, {args.height}p, test frames {test_idx}, "
                        f"body vs fp16 model")
    ids = {}
    for name, d in arms.items():
        rec = store.emit(Record(
            value=round(d["share"], 5), unit="%", population=pop,
            instrument="the body's certify() share, worst of the held-out frames, against the model",
            config=f"calibration arm {name}: {len(frames) if name == 'full' else len(crops)} samples",
            session=session, statistic="worst_of", tier="measured",
            run=f"harness/harness/scripts/calibration_coverage.py --arm {name}", artifact=args.log,
            extra={"artifact_sha256": digest, "of": "share", "arm": name}))
        ids[name] = rec.id
        rec2 = store.emit(Record(
            value=round(d["seconds"], 2), unit="s", population=pop,
            instrument="wall clock around Int8Body.calibrate, one GPU job, synchronised",
            config=f"calibration arm {name}: {len(frames) if name == 'full' else len(crops)} samples",
            session=session, statistic="single", tier="measured",
            run=f"harness/harness/scripts/calibration_coverage.py --arm {name}", artifact=args.log,
            extra={"artifact_sha256": digest, "of": "frame_time", "arm": name}))
        ids[name + "-cost"] = rec2.id
        print(rec.claims_line())
        print(rec2.claims_line())
    with open(out, "w") as fh:
        for rec in store.records.values():
            fh.write(json.dumps(rec.to_dict()) + "\n")
    print(f"# {len(store.records)} records -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the APP's own int8 fit probe on a clip, at full precision.

This is the runner for `backend/src/utils/Int8Probe.py` - the probe the product decides with - not for
`scripts/probe_int8_fit.py`, which is the offline simulation and a different instrument. The difference
is the one that cost this lane a retraction: the offline simulation and the shipped path agreed on the
engine term and diverged on composition.

It prints the reason string and every number at full precision, because the reason rounds to three
decimals and a value that exists only rounded is not a record.

The `fp16 reference` half of the reason is the depth-2 control: on the SAME pairs and the SAME
separation the composition is read on, fp16 against fp16. Only the renderer changes, so the gap between
the two is the quantiser's share of the depth-2 error and the rest is the gate's own reference.

Usage:
    probe_int8_app.py [--frames N] [--model PATH] [--width W] [--height H] <clip>...

Run it under the GPU lock, one job at a time:
    flock -n /var/tmp/opencode/gpu.lock <bundled python> scripts/probe_int8_app.py /clip.mkv
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "backend"))

from src.utils.Int8Probe import probe  # noqa: E402

DEFAULT_MODEL = os.path.join(
    os.path.expanduser("~"), "Загрузки", "REAL-2", "REAL-Video-Enhancer-RVE-2.4.2",
    "custom_models", "2x-Futsuu-Anime(1).pth")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=16,
                    help="decoded triples; the product calls the probe with 16 (8 test pairs)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bar", type=float, default=0.1)
    ap.add_argument("--gate-threshold", type=float, default=None,
                    help="the gate's own threshold for the pair filter; the auto tuner clamps to "
                         "(2.0, 4.0), so the probe's 5.0 default certifies on a superset")
    args = ap.parse_args()

    for clip in args.clips:
        kw = {} if args.gate_threshold is None else {"gate_threshold": args.gate_threshold}
        r = probe(clip, args.model, args.width, args.height, frames=args.frames, bar=args.bar, **kw)
        print(f"{clip} (frames={args.frames})")
        print(f"  {r.reason}")
        print(f"  verdict int8={r.use_int8} error={r.error}")
        print(f"  RAW engine={r.engine!r} d1={r.composition!r} d2={r.composition_d2!r} "
              f"stale_d1={r.stale_d1!r} stale_d2={r.stale_d2!r} peak={r.peak!r} pairs={r.pairs}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

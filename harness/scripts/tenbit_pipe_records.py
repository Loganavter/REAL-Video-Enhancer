"""Records for the 10-bit proof: the reader/packer round trip, by code and on real frames.

Two instruments, two populations, deliberately not merged:

* the CODE populations are synthetic and legal - every luma code 64..940, and every chroma code that keeps
  RGB inside the gamut - so a delta there is depth or rounding and nothing else;
* the FRAME population is real material (5 frames of the native 10-bit clip), where the delta also carries
  the reader's own RGB conversion and its clamping at saturated pixels. The share of clipped pixels is
  emitted next to it, because that is the part of the residual that is not depth.

Usage: python harness/scripts/tenbit_pipe_records.py [--probe ...] [--out ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)

from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/tenbit-pipe/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/tenbit-pipe.jsonl")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    dig = hashlib.sha256(open(args.probe, "rb").read()).hexdigest()
    clip = pr["clip"]
    store = Store(session=f"s-{pr['take']}")
    y, u, v = pr["ramps"]
    frames = pr["frames"]
    summary = pr["summary"]

    def code_pop(plane: str, lo: int, hi: int, label: str) -> Population:
        return Population.of("codes", [f"{plane}#{c}" for c in range(lo, hi + 1)], label)

    pop_y = code_pop("Y", 64, 940, "every legal 10-bit luma code, chroma neutral (synthetic, in gamut)")
    def swept_pop(plane: str, label: str) -> Population:
        # the codes the probe ACTUALLY swept - not the nominal range: the population has to be the measured
        # item set, or the record's label would describe a different sweep than the number
        codes = pr["ramps"][1 if plane == "U" else 2]["swept_codes"]
        return Population.of("codes", [f"{plane}#{c}" for c in codes], label)

    pop_u = swept_pop("U", "every U code that keeps RGB in gamut at co-varied luma (synthetic)")
    pop_v = swept_pop("V", "every V code that keeps RGB in gamut at co-varied luma (synthetic)")
    pop_f = Population.of("frames", [f"{clip}#frame{r['frame']}" for r in frames],
                          f"{clip}, 5 frames spread over the clip (the app's own reader and packer)")

    def emit(value, unit, pop, instrument, config, statistic, rid, extra=None) -> None:
        store.emit(Record(value=value, unit=unit, population=pop, instrument=instrument, config=config,
                          session=store.session, statistic=statistic, tier="measured", id=rid,
                          run="scripts/tenbit_pipe_probe.py", artifact=args.probe,
                          extra={"artifact_sha256": dig, "of": "yuv420p10le_roundtrip", **(extra or {})}))

    emit(float(y["max_delta"]), "levels", pop_y,
         "code-by-code |delta| between the clip's own yuv420p10le values and reader->packer output",
         "TorchUtils.yuv420p10_to_tensor -> tensor_to_yuv420p10, 10-bit limited range BT.709, fp16 tensor",
         "max", "TB001")
    emit(float(u["max_delta"]), "levels", pop_u,
         "same round trip, one uniform chroma plane per code (bilinear upsample + box filter are the identity)",
         "10-bit U codes in gamut at co-varied luma, V neutral", "max", "TB002")
    emit(float(v["max_delta"]), "levels", pop_v,
         "same round trip, one uniform chroma plane per code",
         "10-bit V codes in gamut at co-varied luma, U neutral", "max", "TB003")
    emit(float(summary["y_max_delta_in_gamut"]), "levels", pop_f,
         "|delta| on Y in 10-bit codes, REAL frames, pixels whose RGB is more than 1% inside the gamut",
         f"{clip}, reader->packer only (no model)", "max", "TB004")
    emit(float(summary["y_p99_delta_worst"]), "levels", pop_f,
         "|delta| on Y in 10-bit codes, REAL frames, 99th percentile of the worst frame",
         f"{clip}, reader->packer only (no model)", "max", "TB005")
    emit(float(summary["in_levels_min"]), "count", pop_f,
         "distinct Y codes in the frame as decoded from the clip", f"{clip}, the material's own count",
         "single", "TB006")
    emit(float(summary["out_levels_min"]), "count", pop_f,
         "distinct Y codes after the reader->packer round trip (an 8-bit collapse cannot exceed 256)",
         f"{clip}, reader->packer only (no model)", "single", "TB007")
    emit(round(float(summary["share_le1_pct_min"]), 3), "%", pop_f,
         "share of Y samples within one 10-bit code of the source, worst frame", f"{clip}", "max", "TB008")
    emit(round(float(summary["clipped_pct_max"]), 3), "%", pop_f,
         "share of pixels whose RGB lands within 1% of the gamut edge (the reader clamps there, so those "
         "cannot round-trip through RGB - this is the non-depth part of the residual)",
         f"{clip}", "max", "TB009")

    store.dump(args.out)
    print(store.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

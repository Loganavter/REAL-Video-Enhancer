"""Records for the int8 tail: the emulation's picture (stage a) and the kernel's timing (stage b).

Two files, two instruments, one row. The picture half is the corpus's own emulation recipe on four held-out
frames; the timing half is one Triton kernel against the shipped two-step tail on the same codes.

Usage: python scripts/tail_hop_records.py [--emul .../emulation.json] [--kernel .../kernel.json]
                                          [--out docs/wsp/verified/records/int8-tail-hop.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.core import Population, Record, Store  # noqa: E402


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emul", default="docs/wsp/verified/logs/tail-hop/emulation.json")
    ap.add_argument("--kernel", default="docs/wsp/verified/logs/tail-hop/kernel.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-tail-hop.jsonl")
    args = ap.parse_args(argv)

    em = json.load(open(args.emul))
    km = json.load(open(args.kernel))
    edig, kdig = sha256(args.emul), sha256(args.kernel)
    esess = f"s-{em['take']}"
    ksess = f"s-{km['take']}"
    clip = em["clip"]

    frames = [f"{clip}#frame{f}" for f in em["held_out"]]
    pop = Population.of("frames", frames,
                        f"{clip} frames {em['held_out']} (held out: calibrated on {em['calib']}), "
                        f"folded-scale int8 tail against the shipped fp16 tail")
    st = Store(session=esess)
    for i, r in enumerate(em["rows"], start=1):
        st.emit(Record(value=round(r["int8_share"], 5), unit="%", population=pop,
                       instrument="the body's arithmetic in torch, integer accumulation exact "
                                  "in float32 (|acc| max 3.4e4 against 2^24)",
                       config=f"folded-scale int8 tail, weights per-output-channel int8, outputs "
                              f"padded 12->16, frame {r['frame']}",
                       session=esess, statistic="single", tier="measured", id=f"A{i:03d}",
                       run="scripts/tail_int8_emulation.py", artifact=args.emul,
                       extra={"artifact_sha256": edig, "of": "share", "frame": r["frame"],
                              "peak": r["int8_peak"], "mean_abs_levels": round(r["int8_mean"], 3)}))
    st.emit(Record(value=round(em["worst_share"], 5), unit="%", population=pop,
                   instrument="the body's arithmetic in torch, integer accumulation exact in float32",
                   config="folded-scale int8 tail, weights per-output-channel int8, outputs padded 12->16",
                   session=esess, statistic="worst_of", tier="measured", id="A005",
                   run="scripts/tail_int8_emulation.py", artifact=args.emul,
                   extra={"artifact_sha256": edig, "of": "share", "bar": "bar_int8_share"}))
    st.emit(Record(value=float(em["worst_peak"]), unit="levels", population=pop,
                   instrument="the body's arithmetic in torch, channel-max |delta| in RGB levels",
                   config="folded-scale int8 tail, weights per-output-channel int8, outputs padded 12->16",
                   session=esess, statistic="max", tier="measured", id="A006",
                   run="scripts/tail_int8_emulation.py", artifact=args.emul,
                   extra={"artifact_sha256": edig, "of": "peak", "bar": "bar_int8_peak",
                          "note": "the DELTA between the two configurations, not the int8 path's "
                                  "deviation from the model"}))
    st.emit(Record(value=round(em["exact_worst_share"], 5), unit="%", population=pop,
                   instrument="the same emulation with NO weight quantisation - the algebra alone",
                   config="folded-scale tail, weights kept in float32 (fold only, no int8 weights)",
                   session=esess, statistic="worst_of", tier="measured", id="A007",
                   run="scripts/tail_int8_emulation.py", artifact=args.emul,
                   extra={"artifact_sha256": edig, "of": "share",
                          "note": "isolates the fold's algebra from the weight quantisation"}))

    kpop = Population.of("frames", (f"{km['clip']}#frame{km['frame']}",),
                         f"{km['clip']} frame {km['frame']}, int8 tail kernel against the shipped "
                         f"two-step tail on the same codes")
    kinst = (f"torch.cuda.Event around {km['iters']} launches behind 3 warm-ups, {km['rounds']} "
             f"interleaved rounds, medians of rounds 2+")
    kcfg = f"tail 64->12 3x3 at 1920x1080, BLOCK_M {km['block_m']}, {km['warps']} warps"
    for rid, arm, label in (("B001", "tail today", "dequantise_nhwc + cuDNN fp16 conv"),
                            ("B002", "dequant", "dequantise_nhwc alone"),
                            ("B003", "conv", "cuDNN fp16 conv alone"),
                            ("B004", "tail int8", "one int8 Triton conv, scale folded into the weights, "
                                                  "epilogue fused, outputs padded 12->16")):
        st.emit(Record(value=round(km["median"][arm], 3), unit="ms",
                       population=Population(kind=kpop.kind, items=kpop.items,
                                             label=f"{kpop.label}, arm {arm}: {label}"),
                       instrument=kinst, config=kcfg, session=ksess, statistic="median",
                       tier="measured", id=rid, run="scripts/tail_hop_probe.py", artifact=args.kernel,
                       extra={"artifact_sha256": kdig, "arm": arm, "of": "tail_time"}))
    st.delta("B004", "B001", note="the candidate's saving on the tail, one frame")

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

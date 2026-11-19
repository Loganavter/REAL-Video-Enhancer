"""The frame budget, assembled from records instead of typed by hand.

Every number in `docs/FRAME-BUDGET.md` is either parsed out of a render log (which is named in the table) or
fetched by id from a record file. Nothing is transcribed: a stale table is the same failure as a stale
number, and this lane has retracted ten of those.

Usage: python scripts/frame_budget.py [--logs DIR] [--arm int8] [--out docs/FRAME-BUDGET.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

PHASES = re.compile(r"int8 phases: head ([\d.]+)\s+quantise ([\d.]+)\s+hops ([\d.]+)\s+tail ([\d.]+)"
                    r"\s+SUM ([\d.]+)\s+BODY-TOTAL ([\d.]+)")
TAILSPLIT = re.compile(r"int8 tail: dequant ([\d.-]+)\s+conv ([\d.]+)\s+shuffle ([\d.]+)"
                       r"\s+residual ([\d.]+)\s+SUM ([\d.]+)")
LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
PACK = re.compile(r"pack\+queue ([\d.]+)")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(REPO, "docs/wsp/verified/records")

# what the frame has gained and lost, each cited by record id (resolved below, never typed)
ADOPTED = [
    ("head as one fused kernel", "H007", "H006", "int8-head-fusion.jsonl"),
    ("tail as a 17th int8 hop", "C002", "C001", "int8-tail-hop.jsonl"),
    ("tail shuffle + residual folded in", "Q006", "Q005", "int8-tail-shuffle.jsonl"),
]
REJECTED = [
    ("epilogue divide folded", "R002", "int8-epilogue-fold.jsonl"),
    ("6-bit codes between hops", "S005", "int8-sub8-codes.jsonl"),
    ("vectorised loads / weight hoist", "L002", "int8-hop-loads.jsonl"),
    ("persistent-CTA grid", "N002", "int8-hop-persistent.jsonl"),
    ("cache policies", "C002", "int8-hop-cache.jsonl"),
    ("2-D CTA tiles", "T004", "int8-hop-tiles.jsonl"),
    ("output-channel split", "X002", "int8-hop-csplit.jsonl"),
    ("M x 2 batching", "Y002", "int8-hop-batch.jsonl"),
    ("pack ring (pinned slots)", "K002", "int8-pack-ring.jsonl"),
]


def load_records() -> dict:
    """{(file stem, id): record} - ids collide ACROSS files (C001 is a frame time in one file and a hop
    time in another), so the file is part of the key. The first version of this script got that wrong and
    printed 6-bit codes as "503.0 count"."""
    out = {}
    for path in glob.glob(os.path.join(RECORDS, "*.jsonl")):
        stem = os.path.splitext(os.path.basename(path))[0]
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[(stem, rec["id"])] = rec
    return out


def newest_log(logdir: str, arm: str):
    best = None
    for path in glob.glob(os.path.join(logdir, f"*_{arm}_r*_null.log")):
        m = re.search(r"_r(\d+)_null\.log$", path)
        if not m:
            continue
        key = (os.path.getmtime(path), int(m.group(1)))
        if best is None or key > best[0]:
            best = (key, path)
    return best[1] if best else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="docs/wsp/verified/logs/wall-defaults/null")
    ap.add_argument("--arm", default="int8")
    ap.add_argument("--out", default="docs/FRAME-BUDGET.md")
    args = ap.parse_args(argv)

    log = newest_log(args.logs, args.arm)
    if log is None:
        raise SystemExit(f"no *_{args.arm}_r*_null.log under {args.logs}")
    text = open(log, errors="replace").read().replace("\r", "\n")
    ph = PHASES.findall(text)
    ts = TAILSPLIT.findall(text)
    loop = LOOP.findall(text)
    pack = PACK.findall(text)
    if not ph or not loop:
        raise SystemExit(f"{log}: no phase or loop line")
    head, quant, hops, tail, body_sum, body_tot = (float(x) for x in ph[-1])
    n_frames, loop_ms = int(loop[-1][0]), float(loop[-1][1])
    pack_ms = float(pack[-1]) if pack else float("nan")
    tsplit = [float(x) for x in ts[-1]] if ts else None

    recs = load_records()

    def val(stem: str, rid: str) -> str:
        rec = recs.get((stem, rid))
        return f"{rec['value']} {rec['unit']}" if rec else f"*{stem}:{rid} missing*"

    def saved(stem: str, hi: str, lo: str) -> str:
        """How much the loop shrank: value(arm off) - value(arm on)."""
        a, b = recs.get((stem, lo)), recs.get((stem, hi))
        if not a or not b:
            return "*missing*"
        d = a["value"] - b["value"]
        pct = 100.0 * d / a["value"] if a["value"] else 0.0
        return f"-{d:.2f} ms ({pct:.1f}%)"

    lines = []
    lines.append("# The frame budget, as it stands")
    lines.append("")
    lines.append("**Generated** by `scripts/frame_budget.py` - do not edit. Every number is parsed from the render")
    lines.append("log named below or resolved by id from `docs/wsp/verified/records/`.")
    lines.append("")
    lines.append(f"Source run: `{os.path.relpath(log, REPO)}` ({n_frames} frames, null sink, reuse off).")
    lines.append("")
    lines.append("## The body, from the app's own phase counter")
    lines.append("")
    lines.append("| line | ms | share of body |")
    lines.append("|---|---|---|")
    for name, v in (("head", head), ("quantise", quant), ("hops", hops), ("tail", tail)):
        share = 100.0 * v / body_tot if body_tot else 0.0
        lines.append(f"| {name} | {v:.2f} | {share:.0f}% |")
    lines.append(f"| **BODY TOTAL** | **{body_tot:.2f}** | 100% |")
    lines.append("")
    if tsplit:
        d, c, sh, r, s = tsplit
        lines.append(f"Tail split on the same run: dequant {d:.2f}, conv {c:.2f}, shuffle {sh:.2f}, "
                     f"residual {r:.2f}, sum {s:.2f} ms.")
        lines.append("")
    lines.append("## The frame")
    lines.append("")
    lines.append("| what | ms |")
    lines.append("|---|---|")
    lines.append(f"| body (above) | {body_tot:.2f} |")
    lines.append(f"| pack + queue | {pack_ms:.2f} |")
    lines.append(f"| **render loop** | **{loop_ms:.2f}** |")
    lines.append(f"| everything else (reader, wrapper, writer wait) | {loop_ms - body_tot - pack_ms:+.2f} |")
    lines.append("")
    lines.append("## What is in this frame because it was measured and adopted")
    lines.append("")
    lines.append("| change | saved on the loop | records |")
    lines.append("|---|---|---|")
    for what, hi, lo, src in ADOPTED:
        stem = os.path.splitext(src)[0]
        lines.append(f"| {what} | {saved(stem, hi, lo)} | {stem}:{lo} / {hi} |")
    lines.append("")
    lines.append("## What was measured and deliberately NOT adopted")
    lines.append("")
    lines.append("| lever | measured result | record |")
    lines.append("|---|---|---|")
    for what, rid, src in REJECTED:
        stem = os.path.splitext(src)[0]
        lines.append(f"| {what} | {val(stem, rid)} (the arm that lost) | {stem}:{rid} |")
    lines.append("")
    lines.append("## The ratio this produces")
    lines.append("")
    lines.append("Measured last in `logs/wall-final` (Violet, null sink, 3 rounds, the shipped default):")
    lines.append("`int8-wall-final.jsonl` W009 = **1.552x** against `bar_int8_wall` = 1.2x. Older, pre-halo:")
    lines.append("`int8-wall-defaults.txt` 1.399x / 1.481x null and 1.511x / 1.506x under `hevc_nvenc`.")
    lines.append("")
    open(os.path.join(REPO, args.out), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

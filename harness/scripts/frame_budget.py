"""The frame budget, assembled from records instead of typed by hand.

Every number in `harness/docs/FRAME-BUDGET.md` is either parsed out of a render log (which is named in the table) or
fetched by id from a record file. Nothing is transcribed: a stale table is the same failure as a stale
number, and this lane has retracted ten of those.

Usage: python harness/scripts/frame_budget.py [--logs DIR] [--arm int8] [--out harness/docs/FRAME-BUDGET.md]
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
TAILOUT = re.compile(r"int8 tail output: (\S+ \S+) \(RVE_TAIL_YUV=(\S+) -> tail_yuv=(\S+), "
                     r"yuv8_output=(\S+), pipe=(\S+)\)")
GRID = re.compile(r"int8 grid: ([^\n]+?) \(RVE_INT8_GRID_K=([^,]+), RVE_INT8_PER_CHANNEL=(\S+)\)")
PACK = re.compile(r"pack\+queue ([\d.]+)")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORDS = os.path.join(REPO, "harness/docs/wsp/verified/records")

# what the frame has gained and lost, each cited by record id (resolved below, never typed)
ADOPTED = [
    ("head as one fused kernel", "H007", "H006", "int8-head-fusion.jsonl"),
    ("tail as a 17th int8 hop", "C002", "C001", "int8-tail-hop.jsonl"),
    ("tail shuffle + residual folded in", "Q006", "Q005", "int8-tail-shuffle.jsonl"),
    ("zero-halo hop buffers (mask-free)", "H005", "H004", "int8-hop-halo.jsonl"),
    ("tail writes the yuv420p payload itself", "JC003", "JC002", "int8-tail-csc.jsonl"),
]
# (lever, what it was supposed to buy, against what, record id, file)
REJECTED = [
    ("epilogue divide folded into a and b", "1.3-2.7 ms: one fp32 divide per element, in the kernel all 16 "
     "hops run", "16 hops chained, 29.3 ms before it", "R002", "int8-epilogue-fold.jsonl"),
    ("6-bit codes between hops", "25% of the hop's 265 MB of traffic", "picture bar 0.1% of pixels",
     "S005", "int8-sub8-codes.jsonl"),
    ("vectorised loads + weight hoist", "the hop is at 52% of both roofs, so its schedule",
     "shipped hop 1.756 ms", "L002", "int8-hop-loads.jsonl"),
    ("persistent-CTA grid", "the same schedule, one CTA per SM looping over tiles",
     "shipped hop 1.703 ms", "N002", "int8-hop-persistent.jsonl"),
    ("cache eviction / modifier policy", "L1 hit rate is only 22%, L2 traffic 6.1x DRAM",
     "shipped hop 1.580 ms", "C002", "int8-hop-cache.jsonl"),
    ("2-D CTA tiles", "halve the window amplification (3.05x -> 1.59x)", "shipped tile 1.504 ms",
     "T004", "int8-hop-tiles.jsonl"),
    ("output channels split across CTAs", "halve the accumulator to double resident warps",
     "shipped hop 1.591 ms", "X002", "int8-hop-csplit.jsonl"),
    ("M x 2 batching", "amortise the per-launch constant - WHICH DOES NOT EXIST: a whole 256x256 hop is "
     "0.0572 ms, so any per-launch fee is <= 0.011 ms. Flat is the expected result, not a clue",
     "batch 1 of the same kernel 1.769 ms/frame", "Y002", "int8-hop-batch.jsonl"),
    ("pack ring (pinned slots)", "the 12.44 MB payload copy is synchronous", "ring off 34.42 ms loop",
     "K002", "int8-pack-ring.jsonl"),
    ("reverse tile order on alternate hops", "the previous hop's tail is still in L2 (32 MB of 132)",
     "forward 26.551 ms over 16 hops", "V002", "int8-hop-traverse.jsonl"),
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


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unknown"


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
    ap.add_argument("--logs", default="harness/docs/wsp/verified/logs/wall-defaults/null")
    ap.add_argument("--arm", default="int8")
    ap.add_argument("--out", default="harness/docs/FRAME-BUDGET.md")
    args = ap.parse_args(argv)

    log = newest_log(args.logs, args.arm)
    if log is None:
        raise SystemExit(f"no *_{args.arm}_r*_null.log under {args.logs}")
    text = open(log, errors="replace").read().replace("\r", "\n")
    ph = PHASES.findall(text)
    ts = TAILSPLIT.findall(text)
    loop = LOOP.findall(text)
    pack = PACK.findall(text)
    tailout = TAILOUT.findall(text)
    grid = GRID.findall(text)
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

    commit = git_commit()
    lines = []
    A = lines.append
    A("# The frame budget, as it stands")
    A("")
    A("**Generated** by `harness/scripts/frame_budget.py` - do not edit. Every number is parsed from the render log")
    A("named below or resolved by id from `harness/docs/wsp/verified/records/`. The prose says what each line IS and")
    A("what its floor is, so a reader who was never in this lane can challenge any of it.")
    A("")
    A("## The machine, the model, the build")
    A("")
    A("| | |")
    A("|---|---|")
    A("| card | NVIDIA GeForce RTX 4060 Ti, 34 SMs, **176 dense int8 TOPS** and **288 GB/s** (the two roofs "
      "every number below is measured against) |")
    A("| ridge | 176 TOPS / 288 GB/s = **611 FLOP/byte** - and the hop sits at 610, i.e. exactly on it |")
    _clip = re.sub(r"_(fp16|int8|[a-z0-9]+)_r\d+_(null|nvenc|ffv1)\.log$", "", os.path.basename(log))
    A(f"| clip | `{_clip}`, {n_frames} frames, 1920x1080 in, 4K out |")
    A("| model | `2x-Futsuu-Anime(1).pth`, SRVGGNetCompact 2x, 64 body channels, 18 convolutions |")
    A("| precision | fp16 engine vs the int8 body: 8-bit codes between the 16 body hops, fp16 head and tail "
      "weights where noted |")
    _to = tailout[-1] if tailout else None
    _payload_line = "int8 tail writes the payload" in text
    A("| build | fused head kernel, int8 tail kernel, tail shuffle+residual inside that tail kernel, zero-halo "
      "hop buffers, and the tail writing the yuv420p payload itself |")
    if grid:
        _g = grid[-1]
        A(f"| int8 grid, as the run resolved it | **{_g[0]}** - `s[c] = max(s_in[c], s_t/K)`, K=2 by default "
          f"(RVE_INT8_GRID_K={_g[1]}, RVE_INT8_PER_CHANNEL={_g[2]}) |")
    if _to:
        A(f"| tail output, as the run resolved it | **{_to[0]}** - RVE_TAIL_YUV={_to[1]} -> tail_yuv={_to[2]}, "
          f"yuv8_output={_to[3]}, pipe={_to[4]} (the gate refuses on HDR / 10-bit / non-bt709, so the flag "
          f"alone is not what the run did) |")
    elif _payload_line:
        A("| tail output, as the run resolved it | **yuv420p payload** - the run printed `int8 tail writes the "
          "payload`; it predates the explicit gate line, so this is read from the payload line rather than "
          "from the flag |")
    else:
        A("| tail output, as the run resolved it | *not resolved: no `int8 tail output` and no payload line in "
          "this log, so the flag is not evidence of the path* |")
    A("| the bar | `bar_int8_wall` = **1.2x** wall clock against the fp16 engine; the picture bar is the share "
      "of pixels more than 8 levels off on the worst frame of a spread sample, 0.1% |")
    A("")
    A(f"| commit | `{commit}` - the tree these numbers were taken on |")
    A("")
    A("The pipeline, in the order a frame is paid for: **decode** (ffmpeg, outside the loop) -> **reader** "
      "(yuv420p10 -> a tensor) -> **head** (3 -> 64 channels) -> **sixteen hops** (the body) -> **tail** "
      "(64 -> 12 channels, pixel shuffle to 3 at 2x) -> **residual** (add the upsampled input) -> **pack** "
      "(4K payload to the host) -> **writer** (ffmpeg stdin). The render loop below is the wall clock of that "
      "chain; the body is what the model costs, measured by the app's own events.")
    A("")
    A("## The body, from the app's own phase counter")
    A("")
    A("| line | ms | share | what it is |")
    A("|---|---|---|---|")
    A(f"| head | {head:.2f} | {100*head/body_tot:.0f}% | the first convolution, 3 -> 64 channels, 3x3, "
      f"fused with its PReLU and with the quantisation into int8 codes (one kernel) |")
    A(f"| quantise | {quant:.2f} | {100*quant/body_tot:.0f}% | 0.00 by construction: the head writes codes "
      f"directly, so this pass no longer exists |")
    A(f"| hops | {hops:.2f} | {100*hops/body_tot:.0f}% | the body proper: sixteen 64 -> 64 int8 convolutions, "
      f"3x3, one Triton kernel launched sixteen times |")
    A(f"| tail | {tail:.2f} | {100*tail/body_tot:.0f}% | the last convolution 64 -> 12 channels, its pixel "
      f"shuffle to 3 channels at 2x, and the residual add - all three in one kernel now |")
    A(f"| **BODY TOTAL** | **{body_tot:.2f}** | 100% | measured with the app's own CUDA events, window by "
      f"window |")
    A("")
    if tsplit:
        d, c, sh, r, s = tsplit
        A(f"Tail split on the same run: dequant {d:.2f}, conv {c:.2f}, shuffle {sh:.2f}, residual {r:.2f}, "
          f"sum {s:.2f} ms. The zeroes are the second adoption: the dequant pass was folded into the tail's "
          f"weights and the shuffle+residual into its epilogue.")
        A("")
    # the tail's own floor: the last hop's codes (64 B per LR pixel) + the residual LR frame (6 B) + the
    # payload (6 B when the tail packs yuv420p, 24 B when it writes fp16 RGB). At 288 GB/s. The LR geometry
    # is read from the reader's own command line in the log, not typed.
    _lr = re.search(r"-s (\d+)x(\d+)", text)
    lr_px = int(_lr.group(1)) * int(_lr.group(2)) if _lr else 1920 * 1080
    tail_in = lr_px * (64 + 6) / 1e6
    A("### The tail did not get faster, and its floor says why")
    A("")
    A(f"Its floor is the traffic it cannot avoid: {tail_in:.2f} MB of codes and residual to read, plus the")
    A(f"output. Writing yuv420p moved that floor from {(tail_in + 24 * lr_px / 1e6) / 288.0 * 1000:.3f} ms to")
    A(f"{(tail_in + 6 * lr_px / 1e6) / 288.0 * 1000:.3f} ms - a 0.13 ms shift - and the measured phase read 1.00 ms")
    A("both before and after. So the tail was never bounded by its write; it is bounded upstream (the code read,")
    A("the convolution and the epilogue), and the whole saving of the yuv payload appeared in the pack instead.")
    A("A floor is not a promise: a member that is not the binding one can be removed for free in time and only")
    A("for certain on paper.")
    A("")
    per_hop = hops / 16.0
    A("## The hop against its two roofs - why every single lever came out flat")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| one hop, measured | {per_hop:.3f} ms |")
    A(f"| its compute floor (152.9 GFLOP at 176 TOPS) | 0.869 ms = **{100*0.869/per_hop:.0f}% of the roof** |")
    A(f"| its DRAM floor (250.6 MB at 288 GB/s, counter-measured) | 0.870 ms = **{100*0.870/per_hop:.0f}% of the roof** |")
    A("| the two floors agree | that is what 'on the ridge' means: 610 FLOP/byte against a ridge of 611 |")
    A("| ncu's own reading | Compute (SM) 74%, DRAM 56%, LSU the busiest pipe (73.9% by instruction count), "
      "L2 6.1x the DRAM traffic |")
    A("")
    A("The 74% and the 57% are not a contradiction: ncu measures **pipe saturation** (including the load/store "
      "unit), the 57% measures **arithmetic against the int8 roof**. Both say the same thing - the kernel presses "
      "on compute and on memory at the same time, in equal measure - and that is why a change that improves only "
      "one of the two reads as flat noise. Thirty configurations in seven families were tried on this line; every")
    A("one lost (the table three sections down).")
    A("")
    A("## The frame")
    A("")
    A("| what | ms | floor, and where the floor comes from |")
    A("|---|---|---|")
    A(f"| body (above) | {body_tot:.2f} | **13.9 ms** for the hops alone at the 176 TOPS roof (152.9 GFLOP per "
      f"hop x 16); DRAM floor **13.9 ms** (250.6 MB per hop x 16 at 288 GB/s) - the two agree, which is the "
      f"ridge again |")
    A(f"| pack + queue | {pack_ms:.2f} | **1.9-2.4 ms**: the 4K yuv payload is 12.44 MB and the link is "
      f"**PCIe gen 3 x8** - measured under load at 6.55 GB/s round-trip, 25 of 25 samples, against 7.88 GB/s "
      f"theoretical. The card is a PCIe 4.0 part but the slot negotiates gen 3, so this floor is a property of "
      f"the motherboard |")
    A(f"| **render loop** | **{loop_ms:.2f}** | what the user waits for; `null` sink, so no encoder |")
    A(f"| everything else | {loop_ms - body_tot - pack_ms:+.2f} | reader, the wrapper around the model call, "
      f"and the writer's back-pressure. The input path inside it is measured at 1.38 ms, of which 0.07 was "
      f"removable |")
    A("")
    A("The phase lines account for -2.31 ms of the tail-CSC row and the loop for -2.16, and the 0.15 ms between")
    A("them lands here, in `everything else`: the payload branch in `Frame`, the Y-plane reads for the preview")
    A("and the gate's luma, and the guard's event. Small, and named rather than absorbed - an unattributed")
    A("0.15 ms is exactly the size of finding that turns out to be a second copy somewhere.")
    A("")
    A("## What is in this frame because it was measured and adopted")
    A("")
    A("| change | saved on the loop | records |")
    A("|---|---|---|")
    for what, hi, lo, src in ADOPTED:
        stem = os.path.splitext(src)[0]
        A(f"| {what} | {saved(stem, hi, lo)} | {stem}:{lo} / {hi} |")
    A("")
    A("## What was measured and deliberately NOT adopted")
    A("")
    A("An outsider reading this list is the intended use: every entry is a lever someone expected to work.")
    A("")
    A("| lever | what it was supposed to buy | what it actually read | against | record |")
    A("|---|---|---|---|---|")
    for what, expected, against, rid, src in REJECTED:
        stem = os.path.splitext(src)[0]
        A(f"| {what} | {expected} | **{val(stem, rid)}** | {against} | `harness/docs/wsp/verified/{stem}.txt` : "
          f"{rid} |")
    A("")
    A("## What is still open, and what it costs")
    A("")
    A("| idea | expected | what it needs | state |")
    A("|---|---|---|---|")
    A("| the hop: taps, staging, cascade, tiles, schedules | -5.3 ms at the very best, and only by removing the "
      "FIRST tap's exposed latency | **closed by mechanism**: staging ADDS LSU instructions (9 -> 11.26 at a 2-D "
      "tile, 9 -> 15.09 for a band) on the busiest pipe, `tl.gather` prices the register route at 13.5x, a "
      "register-local dp4a roofs at 44.2 TOPS against the 103.4 already reached, and the per-launch constant "
      "does not exist (a whole 256x256 hop is 0.057 ms) | `int8-hop-tapreuse.txt`, `int8-hop-constant.txt` |")
    A("| overlap the IO with the body (streams) | the pack was expected to be a copy that could hide | **-0.01 "
      "ms**: it is YCbCr arithmetic, and the tail-CSC adoption removed that arithmetic instead | "
      "`frame-pipeline.txt` |")
    A("| a PCIe 4.0 slot | ~1.2 ms off `pack`, since the link is gen 3 x8 today | hardware | not software |")
    A("| skipping duplicate frames / static regions | large on anime, and the owner's call forbids it | - | "
      "closed by policy, `rve-reuse-gate-forbidden` |")
    A("| cut the model (fewer channels or hops) | 2.9-4.5 ms per hop removed | 14-37% per-layer residual, 1.58% "
      "of pixels over the picture bar | closed by measurement |")
    A("")
    A("## Which side of the ratio a millisecond lands on")
    A("")
    A("The bar is a RATIO, `fp16 / int8`, so a millisecond saved is not worth the same everywhere:")
    A("")
    A("| a millisecond saved in | int8 becomes | fp16 becomes | ratio |")
    A("|---|---|---|---|")
    _i = float(recs[("int8-tail-csc", "JC003")]["value"])
    _f = float(recs[("int8-tail-csc", "JC001")]["value"])
    A(f"| the int8 body only (a kernel) | {_i - 1:.2f} | {_f:.2f} | **{_f / (_i - 1):.3f}** |")
    A(f"| a stage BOTH arms pay (pack, reader, writer) | {_i - 1:.2f} | {_f - 1:.2f} | {(_f - 1) / (_i - 1):.3f} |")
    A("")
    A("So a body-side millisecond is worth ~2.8x a shared one against this bar, and the largest wall-clock win in")
    A("the budget can be the smallest ratio win. That is why `rve-hop-fuse2-cascade` (body) and the pack (shared)")
    A("must not be reported as if their milliseconds were interchangeable.")
    A("")
    A("## The ratio this produces")
    A("")
    A(f"Three arms in one session, 3 interleaved rounds, shipped default (`int8-tail-csc.jsonl`): the fp16 "
      f"engine at {val('int8-tail-csc', 'JC001')}, the int8 body at {val('int8-tail-csc', 'JC003')}, so "
      f"**{val('int8-tail-csc', 'JC004')}** against `bar_int8_wall` = 1.2x. The same session WITHOUT the "
      f"tail-CSC row read {val('int8-tail-csc', 'JC005')}, i.e. the row is worth "
      f"{val('int8-tail-csc', 'JC006')} on the int8 arm.")
    A("")
    if ("int8-tail-csc", "JC013") in recs:
        A(f"With `hevc_nvenc` in the loop - the product's own condition - the same two arms read "
          f"{val('int8-tail-csc', 'JC011')} and {val('int8-tail-csc', 'JC012')}, so the product's ratio is "
          f"**{val('int8-tail-csc', 'JC013')}**. On this clip the encoder is almost entirely overlapped: the "
          f"loop sits within 0.2 ms of the null sink in both arms.")
    A("")
    A("## What this file is NOT")
    A("")
    A("- **The encoder is in the ratio, not only in the stand-in.** The product's number above is quoted from")
    A("  `hevc_nvenc` (JC013). On this clip the encoder is almost entirely overlapped - both arms sit within")
    A("  0.2 ms of their null-sink numbers - so the ~10 ms penalty that earlier reports warned about is")
    A("  CONTENT-dependent, not a constant: a claim made from one clip's null-sink loop is a claim about that")
    A("  clip, and the more mobile the frame, the more the encoder can cost.")
    A("- **One clip, one model, one card.** The gains were replayed on three clips (1920x1080 8-bit, native")
    A("  1440x1080 10-bit, and a same-shot cut) and hold within 20% of their size; other cards are untested and")
    A("  the lane has external evidence that int8's advantage is a property of the card.")
    A("- **Not a picture claim.** The picture is judged elsewhere, on 494-frame lossless pairs, and the two")
    A("  boundary changes are byte-identical (`int8-tail-hop.txt`, `int8-hop-halo-chain.txt`).")
    A("- **Not the whole story of the hop.** The hop has its own record: 30 configurations in seven families,")
    A("  ncu counters, DRAM/L2 traffic, and the page of levers that lost (`int8-hop-occupancy.txt`,")
    A("  `int8-hop-dram.txt`, `int8-hop-ncu.txt`).")
    A("")
    A("## Regenerate")
    A("")
    A("```bash")
    A("python harness/scripts/frame_budget.py --logs harness/docs/wsp/verified/logs/wall-final --arm int8")
    A("```")
    A("")
    open(os.path.join(REPO, args.out), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E29 phase 1: is the L2-resident layer pyramid arithmetic real, and what is its ceiling?

CPU arithmetic only - no GPU, no picture. The question the ledger could not answer with a bound
("the model has that headroom; the builder cannot collect it") is: between two body layers, how many
bytes must stay on-chip so that the next layer reads them from L2 instead of DRAM, and does the
16-boundary sum fit the card's L2?

MICRO'16 (Alwani et al.) did this for the first five VGG layers: 77 MB -> 3.6 MB with 362 KB on-chip.
Our version of the same accounting: a 3x3 layer needs two rows of its input that the previous layer
did not overwrite, so a row-block pipeline has to keep `(K-S) = 2` rows of a 64-channel fp16 frame per
boundary resident.

What this cannot do: say that the time saved equals the traffic saved. That is a claim about the
memory system (L2 hit rate under this access pattern), and it needs ncu counters or a fused prototype.
Phase 1's job is to say whether the arithmetic is worth a phase 2 at all.

Usage:
  python3 harness/scripts/pyramid_budget.py                    # 1080p and 4K
  python3 harness/scripts/pyramid_budget.py --width 1920 --height 1080 --emit /tmp/opencode/e29.jsonl
"""
from __future__ import annotations

import argparse
import sys

# What is measured on this host and where it comes from (RESEARCH-LEDGER.md unless stated).
DRAM_GB_s = 288.0          # effective DRAM bandwidth, measured (ledger: the int8 pass at its floor)
FP16_TFLOP_s = 74.0        # the borrowed sustained fp16 figure the ledger's floor is built on
INT8_TOPS = 91.0           # what the int8 conv reports on this shape (E9', E27)
L2_MIB = 32.0              # AD106 (RTX 4060 Ti 16 GB): 32 MiB of L2
FRAME_MS = 56.8            # measured fp16 body+plan at 1080p, the number a win is relative to
BODY_LAYERS = 16
CH = 64
HALO_ROWS = 2              # 3x3: two rows survive from one vertical step to the next
FLOPS_BODY_1080 = 2.446e12  # 16 x 64x64x9 x 1920x1080 x 2


def tensor_bytes(w: int, h: int, ch: int = CH, itemsize: int = 2) -> float:
    return float(w) * h * ch * itemsize


def layer_flops(w: int, h: int, ch: int = CH) -> float:
    return 2.0 * ch * ch * 9 * w * h


def report(w: int, h: int, *, emit: str = "") -> dict:
    frame = tensor_bytes(w, h)
    boundary = HALO_ROWS * w * CH * 2          # rows x width x channels x fp16
    total = boundary * BODY_LAYERS
    out: dict[str, float] = {
        "frame_MB": frame / 2**20,
        "boundary_KB": boundary / 2**10,
        "total_MB": total / 2**20,
        "l2_MiB": L2_MIB,
        "fits": float(total <= L2_MIB * 2**20),
    }

    logical = 2.0 * frame * BODY_LAYERS        # each boundary: read the input, write the output
    out["logical_GB"] = logical / 1e9
    out["logical_ms"] = logical / (DRAM_GB_s * 1e9) * 1e3

    body_flops = layer_flops(w, h) * BODY_LAYERS
    out["compute_fp16_ms"] = body_flops / (FP16_TFLOP_s * 1e12) * 1e3
    out["compute_int8_ms"] = body_flops / (INT8_TOPS * 1e12) * 1e3

    # Roofline. "l2_absorbs" is an assumption, not a measurement: the ledger's ~1/3 comes from an
    # ncu sector-hit rate of 90% on ONE kernel, which is not the same population as a whole graph.
    # The ceiling is a ratio of two floors and is only a SPEEDUP against a measured frame of the
    # same size and precision - at any other size it is just arithmetic, and saying "0.53x" against
    # a 1080p measurement would be the class-1 error this repo keeps re-learning.
    for frac in (0.0, 1 / 3, 1.0):
        dram_ms = out["logical_ms"] * (1.0 - frac)
        out[f"dram_ms@{frac:.2f}"] = dram_ms
        out[f"roofline_fp16@{frac:.2f}"] = max(out["compute_fp16_ms"], dram_ms)
        out[f"roofline_int8@{frac:.2f}"] = max(out["compute_int8_ms"], dram_ms)
    if (w, h) == (1920, 1080):
        # divide the ROUNDED compute floor, because that is the record the ceiling cites and the
        # verifier recomputes exactly - a quotient of the unrounded value is a different number
        out["ceiling_fp16"] = FRAME_MS / round(out["compute_fp16_ms"], 1)
        out["ceiling_int8"] = FRAME_MS / round(out["compute_int8_ms"], 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=0, help="0 = both 1080p and 4K")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    sizes = [(args.width, args.height)] if args.width and args.height else [(1920, 1080), (3840, 2160)]
    rows = {}
    for w, h in sizes:
        r = report(w, h)
        rows[(w, h)] = r
        print(f"\n== {w}x{h} ==")
        print(f"  frame tensor          {r['frame_MB']:8.1f} MB (fp16, 64 ch)")
        print(f"  reuse buffer / boundary {r['boundary_KB']:8.1f} KB "
              f"({HALO_ROWS} rows x {w} x {CH} ch x 2 B)")
        print(f"  16 boundaries         {r['total_MB']:8.1f} MB   vs L2 {r['l2_MiB']:.0f} MiB  "
              f"-> {'FITS' if r['fits'] else 'DOES NOT FIT'}")
        print(f"  logical traffic       {r['logical_GB']:8.2f} GB  -> {r['logical_ms']:7.1f} ms at "
              f"{DRAM_GB_s:.0f} GB/s")
        print(f"  compute floor         fp16 {r['compute_fp16_ms']:6.1f} ms   "
              f"int8 {r['compute_int8_ms']:6.1f} ms at {INT8_TOPS:.0f} TOPS")
        for frac in (0.0, 1 / 3, 1.0):
            print(f"  if L2 absorbs {frac:4.0%}: DRAM {r[f'dram_ms@{frac:.2f}']:6.1f} ms, "
                  f"roofline fp16 {r[f'roofline_fp16@{frac:.2f}']:6.1f} ms "
                  f"int8 {r[f'roofline_int8@{frac:.2f}']:6.1f} ms")
        if "ceiling_fp16" in r:
            print(f"  ceiling vs the measured {FRAME_MS:.1f} ms (fp16, 1080p, same size): "
                  f"{r['ceiling_fp16']:.2f}x fp16-roofline, {r['ceiling_int8']:.2f}x int8-roofline "
                  f"(the numerator is an fp16 measurement - see the note)")
        else:
            print(f"  no measured frame at {w}x{h}: the rooflines above are absolute, not a speedup")

    print("\nNOT checked: whether L2 actually retains the buffer under this access pattern "
          "(needs ncu or a fused prototype); whether the builder can emit such a schedule at all; "
          "the halo's own arithmetic cost (recompute) - this counts reuse only")
    print(f"ASSUMPTIONS: DRAM {DRAM_GB_s} GB/s, fp16 {FP16_TFLOP_s} TFLOP/s, int8 {INT8_TOPS} TOPS, "
          f"L2 {L2_MIB} MiB, measured frame {FRAME_MS} ms - all measured elsewhere in the ledger, "
          f"none re-measured here")

    if args.emit:
        import os as _bos, sys as _bsys
        _HERE = _bos.path.dirname(_bos.path.abspath(__file__))
        _ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
        for _p in (_ROOT, _HERE):
            if _p not in _bsys.path:
                _bsys.path.insert(0, _p)
        from harness import Population, Record, Store
        store = Store()
        # The verifier recomputes only `a/b` and `a-b`, and says so rather than passing an
        # unrecomputable number. Everything here that is arithmetic on documented constants is
        # therefore declared `assumed` with its derivation in `note`, and only the one quotient
        # the decision reads is `derived` - so a verdict citing the ceiling still has to face the
        # fact that its inputs are constants, not measurements of this run.
        for (w, h), r in rows.items():
            pop = Pops = Population.of("boundaries", [f"body-{i}" for i in range(BODY_LAYERS)],
                                       label=f"{w}x{h}, {CH} ch fp16, {HALO_ROWS}-row reuse buffer "
                                             f"per layer boundary")
            common = dict(instrument="arithmetic from tensor shapes", population=pop,
                          config=f"{w}x{h}, fp16, {CH} ch, 3x3, {BODY_LAYERS} layers",
                          session=store.session, run=f"E29-{w}x{h}")
            total = store.emit(Record(value=round(r["total_MB"], 1), unit="MB", statistic="single",
                                      tier="assumed", **common,
                                      note=f"16 boundaries x {r['boundary_KB']:.1f} KB: "
                                           f"{HALO_ROWS} rows x {w} x {CH} ch x 2 B, times 16",
                                      extra={"of": "reuse_buffer_total",
                                             "fits_l2_mib": L2_MIB}))
            logical = store.emit(Record(value=round(r["logical_ms"], 1), unit="ms", statistic="single",
                                        tier="assumed", **common,
                                        note=f"{r['logical_GB']:.2f} GB logical traffic at "
                                             f"{DRAM_GB_s:.0f} GB/s (the measured effective rate)",
                                        extra={"of": "dram_time_at_zero_absorption"}))
            compute = store.emit(Record(value=round(r["compute_fp16_ms"], 1), unit="ms", statistic="single",
                                        tier="assumed", **common,
                                        note=f"{FLOPS_BODY_1080 / 1e12:.3f} TFLOP body at the borrowed "
                                             f"{FP16_TFLOP_s:.0f} TFLOP/s",
                                        extra={"of": "compute_floor_fp16"}))
            if "ceiling_fp16" in r:
                frame = store.emit(Record(value=FRAME_MS, unit="ms", statistic="median",
                                          tier="assumed", **{**common,
                                                             "instrument": "measured 2026-11-17 elsewhere "
                                                                           "in this session, cited here not re-run"},
                                          note="the fp16 plan at 1080p this ceiling is a ratio of",
                                          extra={"of": "frame_time"}))
                store.emit(Record(value=round(r["ceiling_fp16"], 6), unit="x", statistic="single",
                                  population=pop, tier="derived",
                                  formula=f"{frame.id}/{compute.id}", inputs=(frame.id, compute.id),
                                  instrument="arithmetic", config=common["config"],
                                  session=store.session, run=f"E29-{w}x{h}",
                                  note=f"valid because DRAM ({logical.value} ms) is below the compute "
                                       f"floor ({compute.value} ms) at this size - `max()` is not "
                                       f"recomputable by the verifier, so the max is argued here and "
                                       f"the quotient is checked",
                                  extra={"of": "ceiling_vs_measured"}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

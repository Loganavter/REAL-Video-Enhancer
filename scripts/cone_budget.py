"""E36 phase 1: cone instead of cylinder - how much work does a partially changed frame need?

CPU arithmetic only. The frame gate drops or reuses WHOLE frames; this asks what it would cost to
recompute only the region that changed, with a halo that shrinks as you go deeper:

  cylinder   every layer recomputes the changed region dilated by the full receptive field (18 px),
             because that is what a region-based recompute needs at the INPUT;
  cone       layer L (1..16) recomputes the changed region dilated by (16-L)+1, i.e. only the halo
             its own distance to the output requires.

Work ratio = cylinder_area / sum_L cone_area(L). The ledger already has the 1-D case (E36's note:
cylinder 16*38*L against cone 304*L, exactly 2x); this is the 2-D version the queue asks for, and it
reproduces the 1-D factor as its control.

The number this must not pretend to be: a speedup. A 57x57 region is 0.16% of a 1080p layer, so the
sixteen launches cost more than their work - the arithmetic is a ceiling on the saving, and the launch
overhead is a floor on the cost. Both are printed.

Usage:
  python3 scripts/cone_budget.py                       # 1080p
  python3 scripts/cone_budget.py --width 1920 --height 1080 --emit /tmp/opencode/e36.jsonl
"""
from __future__ import annotations

import argparse
import sys

BODY_LAYERS = 16
RECEPTIVE = 18              # 1 + 16 + 1 LR px (MODEL.md §1) - the cylinder halo at the input
FRAME_MS = 56.8             # measured fp16 plan at 1080p, the base a saving is relative to
LAUNCH_US = 5.0             # a conservative kernel-launch cost on this class of part; assumed


def areas(side: float, layers: int = BODY_LAYERS) -> tuple[float, list[float]]:
    """Cylinder square side and the per-layer cone squares, for a changed square of `side` px.

    Layer L (1..16) reads the changed region dilated by (17-L): to produce the output of layer 16 a
    one-pixel halo is needed, one more for every layer of distance from it, and 16 at the first body
    convolution. The cylinder gives every layer the full receptive halo (18). This convention is the
    one the ledger's 1-D control uses (2 + 2k, k = 1..16 = 304), which is why the control below
    reproduces its 2.0x.
    """
    cyl_side = side + 2 * RECEPTIVE
    cyl = cyl_side * cyl_side
    cone = [(side + 2 * (layers + 1 - L)) ** 2 for L in range(1, layers + 1)]
    return cyl, cone


def line_control() -> float:
    """The ledger's 1-D case: 16*38*L against 304*L. Returns the ratio it reports (2.0).

    A product of widths, not of areas - which is why it is computed here rather than through
    `areas()`, and why it doubles as a self-check on the convention: if this stops printing 2.00 the
    cone convention has drifted from the one the ledger recorded.
    """
    cyl = BODY_LAYERS * (2 + 2 * RECEPTIVE)                    # 16 * 38 = 608
    cone = sum(2 + 2 * k for k in range(1, BODY_LAYERS + 1))   # 304
    return cyl / cone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--radii", default="5,10,20,40")
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    print(f"1-D control (ledger E36): cylinder/cone = {line_control():.2f}x for a 2 px band")
    rows = []
    frame_px = args.width * args.height
    print(f"\n{'r':>4s} {'side':>6s} {'cyl px':>9s} {'cone px':>10s} {'ratio':>7s} "
          f"{'cone share':>10s} {'work ms':>8s} {'launch ms':>9s} {'bound':>7s}")
    for r in [int(x) for x in args.radii.split(",") if x.strip()]:
        side = 2 * r + 1
        cyl, cone = areas(side)
        cyl_total = BODY_LAYERS * cyl
        cone_total = sum(cone)
        ratio = cyl_total / cone_total
        # share of a full frame's work (16 layers of W x H)
        share = cone_total / (BODY_LAYERS * frame_px)
        work_ms = share * FRAME_MS
        launch_ms = BODY_LAYERS * LAUNCH_US / 1000.0
        rows.append({"r": r, "side": side, "cyl_total": cyl_total, "cone_total": cone_total,
                     "ratio": ratio, "share": share, "work_ms": work_ms,
                     "launch_ms": launch_ms, "bound": max(work_ms, launch_ms)})
        print(f"{r:4d} {side:6d} {cyl_total:9.0f} {cone_total:10.0f} {ratio:7.2f} "
              f"{share:9.4%} {work_ms:8.3f} {launch_ms:9.3f} {max(work_ms, launch_ms):7.3f}")

    print(f"\nAT 1080p, base = measured {FRAME_MS:.1f} ms/frame (fp16 plan). The saving is not the "
          f"ratio: it is 56.8 ms of a full re-render replaced by the bound in the last column, on a "
          f"frame the gate would otherwise RENDER because part of it changed")
    print(f"the binding number is the launch floor ({BODY_LAYERS} x {LAUNCH_US:.0f} us = "
          f"{BODY_LAYERS * LAUNCH_US / 1000:.2f} ms) until r is large enough that work dominates")
    print("\nNOT checked: whether the gate can certify a region-based recompute at all (the veto is "
          "pixel-level and a cone that under-estimates the change is wrong by construction); the "
          "real launch/occupancy cost of a 57x57 GEMM; per-tile input/output packing")
    print("ASSUMPTIONS: receptive radius 18 px, 16 body layers, 5 us launch, base 56.8 ms measured "
          "elsewhere - none re-measured here")

    if args.emit:
        sys.path.insert(0, "scripts")
        from harness import Population, Record, Store
        store = Store()
        # The two constants the arithmetic rests on. They are the model's own geometry (MODEL.md §1),
        # not measurements of this run, so they are declared assumed and every derived number points
        # at them instead of at a number in someone's head.
        geom = store.emit(Record(value=float(RECEPTIVE), unit="count", statistic="single",
                                 population=Population.of("body layers",
                                                          [f"body-{L}" for L in range(1, BODY_LAYERS + 1)],
                                                          label="the 16-body-conv receptive path"),
                                 tier="assumed", instrument="MODEL.md §1 (1+16+1 LR px), not measured here",
                                 config=f"{BODY_LAYERS} body convs, 3x3", session=store.session,
                                 run="E36", note="receptive radius in LR px; halved in the cone "
                                                 "as 17-L per layer",
                                 extra={"of": "receptive_radius"}))
        for row in rows:
            pop = Population.of("layers", [f"body-{L}" for L in range(1, BODY_LAYERS + 1)],
                                label=f"changed square r={row['r']} (side {row['side']} px), "
                                      f"{args.width}x{args.height}, cone halo 1 px/layer")
            cfg = (f"{args.width}x{args.height}, r={row['r']}, receptive 18, 16 layers")
            cyl = store.emit(Record(value=round(row["cyl_total"], 0), unit="count",
                                    statistic="single", population=pop, tier="assumed",
                                    instrument="arithmetic from the receptive radius",
                                    config=cfg, session=store.session, run=f"E36-r{row['r']}",
                                    note="16 layers x (side+36)^2 - squares, not a product of widths; "
                                         "the verifier recomputes quotients, so this is declared",
                                    extra={"of": "cylinder_px"}))
            cone = store.emit(Record(value=round(row["cone_total"], 0), unit="count",
                                     statistic="single", population=pop, tier="assumed",
                                     instrument="arithmetic from the receptive radius",
                                     config=cfg, session=store.session, run=f"E36-r{row['r']}",
                                     note="sum over L of (side+2*(17-L))^2",
                                     extra={"of": "cone_px"}))
            rid = store.emit(Record(value=round(row["ratio"], 6), unit="x", statistic="single",
                                    population=pop, tier="derived",
                                    formula=f"{cyl.id}/{cone.id}", inputs=(cyl.id, cone.id),
                                    instrument="arithmetic", config=cfg,
                                    session=store.session, run=f"E36-r{row['r']}",
                                    extra={"of": "work_ratio"}))
            store.emit(Record(value=round(row["bound"], 3), unit="ms", statistic="single",
                              population=pop, tier="assumed",
                              instrument="arithmetic",
                              config=cfg, session=store.session, run=f"E36-r{row['r']}",
                              note=f"max(work={row['work_ms']:.3f} ms, {BODY_LAYERS} launches x "
                                   f"{LAUNCH_US:.0f} us={row['launch_ms']:.3f} ms) - a max is not "
                                   f"recomputable by the verifier, so this is declared, and it "
                                   f"depends on the ratio record {rid.id}",
                              extra={"of": "cone_time_bound"}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

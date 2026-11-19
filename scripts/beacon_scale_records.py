"""Records for the Beacon closed-form scale against the swept grid clamp K.

Two inputs, both committed under `docs/wsp/verified/logs/beacon/`:
  beacon.json            per policy, per held-out frame: share of pixels >8 levels, max, mean
  beacon-clipping.json   per hop: saturation share at |code| = 127 and the grid ratio (K=2 / fitted)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs/beacon")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-beacon-scale.jsonl")
    a = ap.parse_args(argv)

    def path(name: str) -> str:
        return os.path.join(a.dir, name)

    def sha(p: str) -> str:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    acc = json.load(open(path("beacon.json")))
    clip = json.load(open(path("beacon-clipping.json")))
    st = Store(session="s-2026-11-19T14:45:00+03:00")
    instr = "channel-max |delta| in 8-bit codes against the fp16 model, share above 8 levels"
    cfg = (f"{acc['model']} at {acc['scale']}x, 1920x1080, calibrated on 8 frames, measured on 3 held-out "
           f"frames from the same two bluray cuts")

    n = 1
    for policy, rows in acc["policies"].items():
        for frame, row in rows.items():
            pop = Population.of("pixels of one frame", (f"{frame}:{policy}",), f"{frame}, policy {policy}")
            st.emit(Record(value=float(row["share_gt8_pct"]), unit="%", population=pop, instrument=instr,
                           config=cfg, session=st.session, statistic="single", tier="measured",
                           id=f"BK{n:03d}", run="scripts/beacon_scale_probe.py", artifact=path("beacon.json"),
                           extra={"artifact_sha256": sha(path("beacon.json")), "policy": policy,
                                  "frame": frame}))
            n += 1

    # the mechanism: the fitted grid is tighter, so the late hops saturate
    pop = Population.of("layers", tuple(f"hop-{r['hop']}" for r in clip["hops"]),
                        "the sixteen hops, frame " + clip["frame"])
    st.emit(Record(value=float(clip["worst_sat_fit_pct"]), unit="%", population=pop, instrument=instr,
                   config=cfg + "; share of codes at the int8 rail under the FITTED grids",
                   session=st.session, statistic="max", tier="measured", id=f"BK{n:03d}",
                   run="scripts/beacon_scale_probe.py (clipping analysis)", artifact=path("beacon-clipping.json"),
                   extra={"artifact_sha256": sha(path("beacon-clipping.json")), "of": "saturation_fit"}))
    n += 1
    st.emit(Record(value=float(clip["worst_ratio"]), unit="x", population=pop, instrument=instr,
                   config=cfg + "; grid ratio K=2 mean / fitted mean",
                   session=st.session, statistic="max", tier="measured", id=f"BK{n:03d}",
                   run="scripts/beacon_scale_probe.py (clipping analysis)", artifact=path("beacon-clipping.json"),
                   extra={"artifact_sha256": sha(path("beacon-clipping.json")), "of": "grid_ratio"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

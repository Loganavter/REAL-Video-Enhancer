"""Records for the ink-lift localisation: body RGB, the colour pipe, and the decoded file."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/wsp/verified/logs/ink-lift.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-ink-lift.jsonl")
    a = ap.parse_args(argv)
    d = json.load(open(a.json))
    dig = hashlib.sha256(open(a.json, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{d['frame']}#ink-core",), f"{d['frame']}, the dark line core (luma < 0.2)")
    st = Store(session="s-2026-11-19T15:10:00+03:00")
    instr = "signed mean |delta| in 8-bit levels against the fp16 model on the ink core"
    cfg = f"{d['model']} at {d['scale']}x, 1920x1080 in, 4K out, crop calibration"
    for i, key in enumerate(("body_rgb", "yuv_roundtrip", "decoded_file", "decoded_vs_roundtrip"), start=1):
        st.emit(Record(value=float(d[key]["ink_mean_signed"]), unit="levels", population=pop,
                       instrument=instr + (" (decoded_vs_roundtrip: against the pipe round-trip)" if "roundtrip" in key and key.startswith("decoded") else ""),
                       config=cfg, session=st.session, statistic="mean", tier="measured", id=f"IK{i:03d}",
                       run="harness/scripts/ink_lift_probe.py", artifact=a.json,
                       extra={"artifact_sha256": dig, "stage": key}))
    st.emit(Record(value=float(d["decoded_file"]["share_gt8_pct"]), unit="%", population=pop,
                   instrument="share of pixels above 8 levels, decoded file against the model",
                   config=cfg, session=st.session, statistic="single", tier="measured", id="IK005",
                   run="harness/scripts/ink_lift_probe.py", artifact=a.json, extra={"artifact_sha256": dig}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

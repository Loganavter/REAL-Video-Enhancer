"""Records for the wired crop calibration: cost, start-up, and the 494-frame pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs/calib-crops")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-calib-crops.jsonl")
    a = ap.parse_args(argv)

    def path(n: str) -> str:
        return os.path.join(a.dir, n)

    def sha(p: str) -> str:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    body = json.load(open(path("calib-crops.json")))
    probe = json.load(open(path("calib-probe-times.json")))
    pair = json.load(open(path("pair.json")))["summary"]

    st = Store(session="s-2026-11-19T14:50:00+03:00")
    pop = Population.of("runs", ("calib-full", "calib-crops"),
                        "the same model, the same eight frames, two calibration populations")
    cfg = "2x-Futsuu-Anime(1).pth, 1920x1080, K=2, the same frames; only the crop count differs"
    n = 1
    for arm in body["arms"]:
        st.emit(Record(value=float(arm["calibrate_s"]), unit="s", population=pop,
                       instrument="wall clock around Int8Body.calibrate with the calibration cache disabled",
                       config=cfg + f", crops={arm['crops']}", session=st.session, statistic="single",
                       tier="measured", id=f"CC{n:03d}", run="harness/scripts/calib_crops_probe.py",
                       artifact=path("calib-crops.json"),
                       extra={"artifact_sha256": sha(path("calib-crops.json")), "crops": arm["crops"]}))
        n += 1
    for crops, seconds in probe["arms"].items():
        st.emit(Record(value=float(seconds), unit="s", population=pop,
                       instrument="wall clock around Int8Probe._calibrate, 16 frames",
                       config=cfg + f", probe path, crops={crops}", session=st.session, statistic="single",
                       tier="measured", id=f"CC{n:03d}", run="probe timing (inline)", artifact=path(
                           "calib-probe-times.json"),
                       extra={"artifact_sha256": sha(path("calib-probe-times.json")), "crops": int(crops)}))
        n += 1

    frames = Population.of("frames", tuple(f"bd_violet20s.mkv#frame{i}" for i in range(494)),
                           "BD Violet, 494 frames, ffv1 lossless, crops against full-frame calibration")
    st.emit(Record(value=float(pair["worst_share"]), unit="%", population=frames,
                   instrument="channel-max |delta| in RGB over 64px blocks (gate_sweep --reference-kind config)",
                   config="calibration on 4 crops per frame against full frames, everything else held",
                   session=st.session, statistic="worst_of", tier="measured", id=f"CC{n:03d}",
                   run="harness/harness/scripts/gate_sweep.py", artifact=path("pair.json"),
                   extra={"artifact_sha256": sha(path("pair.json")), "of": "worst_share",
                          "frames_above_bar": pair["frames_above_bar"]}))
    n += 1
    st.emit(Record(value=float(pair["worst_peak"]), unit="levels", population=frames,
                   instrument="peak channel-max |delta| in 8-bit codes over the same pair",
                   config="calibration on 4 crops per frame against full frames, everything else held",
                   session=st.session, statistic="max", tier="measured", id=f"CC{n:03d}",
                   run="harness/harness/scripts/gate_sweep.py", artifact=path("pair.json"),
                   extra={"artifact_sha256": sha(path("pair.json")), "of": "peak"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

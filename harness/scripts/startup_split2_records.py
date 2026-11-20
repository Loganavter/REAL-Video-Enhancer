"""Records for the split body-build block and for the certificate's picture effect, separated by mechanism."""
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
    ap.add_argument("--split", default="harness/docs/wsp/verified/logs/body-build-split.json")
    ap.add_argument("--certdir", default="harness/docs/wsp/verified/logs/cert-picture")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-startup-split2.jsonl")
    a = ap.parse_args(argv)

    def sha(p: str) -> str:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    warm = json.load(open("harness/docs/wsp/verified/logs/body-build-warm.json"))["arms"]["warm"]
    cold = json.load(open(a.split))["arms"]["warm"]        # the second run, TRITON_CACHE_DIR empty
    st = Store(session="s-2026-11-19T15:30:00+03:00")
    pop = Population.of("process starts", ("body-build#warm-triton", "body-build#cold-triton"),
                        "one body build per process, Futsuu at 1920x1080, 8 calibration frames")
    cfg = "2x-Futsuu-Anime(1).pth, 1920x1080, K=2, calibration on 4 crops per frame"
    n = 1
    for value, note, art in (
        (warm["build_s"], "build (grids -> int8 weights)", "harness/docs/wsp/verified/logs/body-build-warm.json"),
        (warm["first_call_s"], "first body call, Triton cache WARM", "harness/docs/wsp/verified/logs/body-build-warm.json"),
        (warm["steady_call_s"], "steady-state call (median of 5)", "harness/docs/wsp/verified/logs/body-build-warm.json"),
        (cold["first_call_s"], "first body call, Triton cache COLD (compiled from scratch)", a.split),
    ):
        st.emit(Record(value=float(value), unit="s", population=pop,
                       instrument="host wall clock around each phase, torch.cuda.synchronize between",
                       config=cfg, session=st.session, statistic="single", tier="measured",
                       id=f"BB{n:03d}", run="harness/scripts/body_build_split.py", artifact=art,
                       extra={"artifact_sha256": sha(art), "phase": note}))
        n += 1

    frames = Population.of("frames", tuple(f"bd_violet20s.mkv#frame{i}" for i in range(494)),
                           "BD Violet, 494 frames, ffv1 lossless")
    for arm, label in (("cert", "certificate ON against the shipped certificate-off path"),
                       ("alloc", "2 GB allocated and released, no certificate"),
                       ("warm", "fp16 engine warmed on 6 frames, no certificate")):
        j = json.load(open(os.path.join(a.certdir, f"{arm}.json")))["summary"]
        st.emit(Record(value=float(j["worst_share"]), unit="%", population=frames,
                       instrument="channel-max |delta| in RGB over 64px blocks (gate_sweep --reference-kind config)",
                       config=label, session=st.session, statistic="worst_of", tier="measured",
                       id=f"CP{n:03d}", run="harness/scripts/wall_ab.sh + harness/scripts/gate_sweep.py",
                       artifact=os.path.join(a.certdir, f"{arm}.json"),
                       extra={"artifact_sha256": sha(os.path.join(a.certdir, f"{arm}.json")),
                              "arm": arm, "peak": j["worst_peak"], "frames_above_bar": j["frames_above_bar"]}))
        n += 1
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

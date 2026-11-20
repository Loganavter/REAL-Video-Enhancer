"""Records for the fade row: pure sequential fails on a black-head clip, the hybrid does not."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="harness/docs/wsp/verified/logs/calib-fade")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-calib-fade.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    st = Store(session="s-2026-11-19T16:00:00+03:00")
    n = 1
    for fname, label, frames_n in (("fade-pair.json", "Interstellar cut that opens on black, hybrid against spread", 266),
                                   ("violet-pair.json", "BD Violet, hybrid against spread", 494)):
        j = json.load(open(os.path.join(a.dir, fname)))["summary"]
        pop = Population.of("frames", tuple(f"{label}#{i}" for i in range(frames_n)), label)
        st.emit(Record(value=float(j["worst_share"]), unit="%", population=pop,
                       instrument="channel-max |delta| over 64px blocks (gate_sweep --reference-kind config)",
                       config=label, session=st.session, statistic="worst_of", tier="measured",
                       id=f"FD{n:03d}", run="harness/scripts/wall_ab.sh + harness/scripts/gate_sweep.py",
                       artifact=os.path.join(a.dir, fname),
                       extra={"artifact_sha256": sha(os.path.join(a.dir, fname)),
                              "frames_above_bar": j["frames_above_bar"], "peak": j["worst_peak"]}))
        n += 1
    # the failure the row exists for: pure sequential on the same black-head cut
    log = "harness/docs/wsp/verified/logs/calib-fade/inter25s_seq_r1_ffv1.log"
    pop = Population.of("frames", tuple(f"inter25s#seq#{i}" for i in range(266)), "Interstellar cut, pure sequential")
    j = json.load(open(os.path.join(a.dir, "fade-seq.json")))["summary"]
    st.emit(Record(value=float(j["worst_share"]), unit="%", population=pop,
                   instrument="channel-max |delta| over 64px blocks (gate_sweep --reference-kind config)",
                   config="pure sequential first-48-frame calibration against the spread decode, clip opens on black",
                   session=st.session, statistic="worst_of", tier="measured", id=f"FD{n:03d}",
                   run="harness/scripts/wall_ab.sh + harness/scripts/gate_sweep.py",
                   artifact=os.path.join(a.dir, "fade-seq.json"),
                   extra={"artifact_sha256": sha(os.path.join(a.dir, "fade-seq.json")),
                          "frames_above_bar": j["frames_above_bar"], "peak": j["worst_peak"]}))
    n += 1
    log = "harness/docs/wsp/verified/logs/startup-timed-hybrid.log"
    total = None
    for line in open(log):
        m = re.search(r"startup: first frame enqueued ([\d.]+) s", line)
        if m: total = float(m.group(1))
    if total is not None:
        st.emit(Record(value=total, unit="s", population=Population.of(
            "process starts", ("bd_violet20s#hybrid",), "one cold backend process, hybrid calibration"),
            instrument="the app's own stdout, seconds since process start",
            config="2x-Futsuu-Anime, 1920x1080, crops + hybrid decode, CACHE MISS", session=st.session,
            statistic="single", tier="measured", id=f"FD{n:03d}", run="/tmp/opencode/timed_run.py",
            artifact=log, extra={"artifact_sha256": sha(log)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

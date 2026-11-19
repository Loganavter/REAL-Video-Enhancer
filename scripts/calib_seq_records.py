"""Records for the sequential-vs-spread calibration decode."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs/calib-seq")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-calib-seq.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    d = json.load(open(os.path.join(a.dir, "decode-split.json")))
    pair = json.load(open(os.path.join(a.dir, "pair.json")))["summary"]
    st = Store(session="s-2026-11-19T15:45:00+03:00")
    pop = Population.of("runs", ("decode-spread", "decode-sequential"),
                        "bd_violet20s, 16 triples, 1920x1080, the probe's own decode path")
    for i, (key, note) in enumerate((("spread_s", "16 seeks, 3 frames each"), ("sequential_s", "first 48 frames, one pass")), start=1):
        st.emit(Record(value=float(d[key]), unit="s", population=pop,
                       instrument="wall clock around the probe's decode", config=note, session=st.session,
                       statistic="single", tier="measured", id=f"SQ{i:03d}", run="Int8Probe decode (inline)",
                       artifact=os.path.join(a.dir, "decode-split.json"),
                       extra={"artifact_sha256": sha(os.path.join(a.dir, "decode-split.json"))}))
    frames = Population.of("frames", tuple(f"bd_violet20s.mkv#frame{i}" for i in range(494)),
                           "BD Violet, 494 frames, ffv1 lossless")
    st.emit(Record(value=float(pair["worst_share"]), unit="%", population=frames,
                   instrument="channel-max |delta| over 64px blocks (gate_sweep --reference-kind config)",
                   config="sequential first-48-frame calibration against the spread decode, crops on both",
                   session=st.session, statistic="worst_of", tier="measured", id="SQ003",
                   run="scripts/wall_ab.sh + scripts/gate_sweep.py", artifact=os.path.join(a.dir, "pair.json"),
                   extra={"artifact_sha256": sha(os.path.join(a.dir, "pair.json")),
                          "peak": pair["worst_peak"], "frames_above_bar": pair["frames_above_bar"]}))
    # the end-to-end start-up with both changes on (crops + sequential), same instrument as the split page
    import re
    log = "docs/wsp/verified/logs/startup-timed-seq.log"
    total = None
    for line in open(log):
        m = re.search(r"startup: first frame enqueued ([\d.]+) s", line)
        if m:
            total = float(m.group(1))
    if total is not None:
        st.emit(Record(value=total, unit="s", population=Population.of(
            "process starts", ("bd_violet20s#crops+sequential",), "one cold backend process, both changes on"),
            instrument="the app's own stdout, seconds since process start", config="2x-Futsuu-Anime(1).pth, 1920x1080, crops + sequential, CACHE MISS",
            session=st.session, statistic="single", tier="measured", id="SQ004",
            run="/tmp/opencode/timed_run.py", artifact=log,
            extra={"artifact_sha256": sha(log)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

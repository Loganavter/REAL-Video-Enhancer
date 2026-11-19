"""Records for the IO-overlap experiment: the stream A/B, the guard A/B, and the guard's picture."""
from __future__ import annotations
import argparse, hashlib, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402

LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
PACK = re.compile(r"pack\+queue ([\d.]+)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)


def read_arm(logdir: str, tag: str, arms: tuple[str, str]):
    loops: dict[str, list[float]] = {}
    packs: dict[str, list[float]] = {}
    session = "unknown"
    for name in sorted(os.listdir(logdir)):
        m = re.match(rf"^(.+)_({'|'.join(arms)})_r(\d+)_null\.log$", name)
        if not m:
            continue
        text = open(os.path.join(logdir, name), errors="replace").read().replace("\r", "\n")
        if session == "unknown" and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        lm, pm = LOOP.search(text), PACK.search(text)
        if lm and int(m.group(3)) >= 2:
            loops.setdefault(m.group(2), []).append(float(lm.group(2)))
            if pm:
                packs.setdefault(m.group(2), []).append(float(pm.group(1)))
    return loops, packs, session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/wsp/verified/records/frame-pipeline.jsonl")
    ap.add_argument("--pair", default="docs/wsp/verified/records/payload-guard-pair.jsonl")
    a = ap.parse_args(argv)
    st = Store()
    i = 0
    clip = "bd_violet20s.mkv"
    for logdir, tag, arms, what in (
            ("docs/wsp/verified/logs/frame-pipeline/ab", "pipe", ("pipe0", "pipe1"),
             "the payload copy moved to its own CUDA stream (RVE_PIPE)"),
            ("docs/wsp/verified/logs/frame-pipeline/guard-ab", "guard", ("guard0", "guard1"),
             "an event guard before the writer reads the pinned slot (RVE_PAYLOAD_GUARD)")):
        loops, packs, session = read_arm(logdir, tag, arms)
        if len(loops) != 2:
            continue
        pop = Population.of("frames", (f"{clip}#first494",), f"{clip}, 494 frames, null sink, {what}")
        ids = {}
        for arm in arms:
            for kind, src in (("loop", loops), ("pack", packs)):
                if arm not in src:
                    continue
                i += 1
                art = f"{logdir}/{clip.removesuffix('.mkv')}_{arm}_r2_null.log"
                st.emit(Record(value=round(statistics.median(src[arm]), 2), unit="ms",
                               population=Population(kind=pop.kind, items=pop.items,
                                                     label=f"{pop.label}, arm {arm}"),
                               instrument=("the render loop line" if kind == "loop"
                                           else "the profile's pack+queue line"),
                               config="int8 fit, null sink, arms alternating in one session",
                               session=session, statistic="median", tier="measured", id=f"F{i:03d}",
                               run=f"rve-backend.py, logs {logdir}", artifact=art,
                               extra={"artifact_sha256": hashlib.sha256(open(art, "rb").read()).hexdigest(),
                                      "of": kind, "arm": arm, "per_round": src[arm]}))
                ids[(kind, arm)] = f"F{i:03d}"
        if ("loop", arms[1]) in ids:
            st.delta(ids[("loop", arms[1])], ids[("loop", arms[0])], note=f"{what}: the frame")
        if ("pack", arms[1]) in ids:
            st.delta(ids[("pack", arms[1])], ids[("pack", arms[0])], note=f"{what}: the pack line")
    if os.path.exists(a.pair):
        for rid, rec in Store.load(a.pair).records.items():
            st.emit(rec)
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

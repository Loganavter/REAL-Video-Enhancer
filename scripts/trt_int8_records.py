"""Records for TensorRT's own int8 (calibrated, implicit) against the fp16 engine of the same model.

Inputs, all committed under `docs/wsp/verified/logs/trt-int8/`:
  prof_fp16.log, prof_int8_entropy.log, prof_int8_minmax.log   `trtexec --loadEngine --dumpProfile`
  trt-int8-accuracy.json                                        `scripts/trt_int8_accuracy.py`
  trt-int8-manifest.json                                        `scripts/trt_int8_calibrate.py` (frames by sha256)

The two engines have different layer sets by construction (25 fused vs 60 split), so the fp16/int8 quotient is a
CROSS-POPULATION ratio and the harness marks it `assumed` - the arithmetic is checked, the population claim is
not pretended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402

ROW = re.compile(r"\[I\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$")


def profile(path: str) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    inside = False
    for line in open(path):
        if "=== Profile" in line:
            inside = True
            continue
        if "=== Performance summary" in line:
            inside = False
        if not inside:
            continue
        m = ROW.search(line)
        if m and m.group(5).strip() != "Total" and "Time(ms)" not in line:
            rows.append((float(m.group(2)), m.group(5).strip()))
    if not rows:
        raise SystemExit(f"{path}: no profile rows")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs/trt-int8")
    ap.add_argument("--out", default="docs/wsp/verified/records/trt-int8-calibrated.jsonl")
    ap.add_argument("--probe", default="docs/wsp/verified/logs/trt-int8/trt-vs-body.json")
    a = ap.parse_args(argv)

    def path(name: str) -> str:
        return os.path.join(a.dir, name)

    def sha(p: str) -> str:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    prof = {"fp16": profile(path("prof_fp16.log")),
            "int8-entropy": profile(path("prof_int8_entropy.log")),
            "int8-minmax": profile(path("prof_int8_minmax.log"))}
    acc = json.load(open(path("trt-int8-accuracy.json")))
    man = json.load(open(path("trt-int8-manifest.json")))
    frames = man["frames"]
    build_s = {("fp16" if not arm["int8"] else "int8-" + arm["int8"]): arm["build_s"] for arm in man["arms"]}

    st = Store(session="s-2026-11-19T14:10:00+03:00")
    instrument = ("trtexec 10.12.0.36 --loadEngine --dumpProfile: per-layer Avg(ms) summed over the table, "
                  "each Avg the mean of 20 runs (--avgRuns=20)")
    config = ("2x-Futsuu-Anime(1).pth -> ONNX opset 17 (sha %s), 1x3x1080x1920, builder level 5, whole frame "
              "(not the app's tiled engine)" % man["onnx_sha256"])
    digests = {name: sha(path(name)) for name in
               ("prof_fp16.log", "prof_int8_entropy.log", "prof_int8_minmax.log",
                "trt-int8-accuracy.json", "trt-int8-manifest.json")}

    def emit(value, unit, pop, *, rid, artifact, statistic="mean", n=20, note="", extra=None, tier="measured",
             formula="", inputs=()):
        st.emit(Record(value=round(float(value), 4), unit=unit, population=pop, instrument=instrument,
                       config=config, session=st.session, statistic=statistic, tier=tier, id=rid,
                       formula=formula, inputs=inputs, run="trtexec --loadEngine", artifact=artifact,
                       note=note, extra={"artifact_sha256": digests[os.path.basename(artifact)], **(extra or {})}))

    def layers(rows, label, tag):
        # the arm is part of the item: the fp16 and int8 builds share layer names where they agree
        return Population.of("layers", tuple(f"{tag}/{n}" for _, n in rows), label)

    emit(sum(ms for ms, _ in prof["fp16"]), "ms", layers(prof["fp16"], "the 25 fused layers of the fp16 engine", "fp16"),
         rid="TI001", artifact=path("prof_fp16.log"), note="sum of 25 layer Avg(ms)")
    emit(sum(ms for ms, _ in prof["int8-entropy"]), "ms",
         layers(prof["int8-entropy"], "the 60 layers of the calibrated int8 engine", "int8-entropy"),
         rid="TI002", artifact=path("prof_int8_entropy.log"),
         note="sum of 60 layer Avg(ms); the split from 25 to 60 is PReLU leaving the int8 epilogue")
    emit(sum(ms for ms, _ in prof["int8-minmax"]), "ms",
         layers(prof["int8-minmax"], "the 60 layers of the min-max calibrated int8 engine", "int8-minmax"),
         rid="TI003", artifact=path("prof_int8_minmax.log"))
    emit(len(prof["int8-entropy"]), "count", layers(prof["int8-entropy"], "the 60 layers of the int8 engine", "int8-entropy"),
         rid="TI004", artifact=path("prof_int8_entropy.log"), statistic="single", n=1)
    emit(len(prof["fp16"]), "count", layers(prof["fp16"], "the 25 layers of the fp16 engine", "fp16"),
         rid="TI005", artifact=path("prof_fp16.log"), statistic="single", n=1)

    r1 = st.ratio("TI001", "TI002", cross_population=True, note="fp16 / int8-entropy, whole frame")
    r1.id = "TI006"
    r2 = st.ratio("TI001", "TI003", cross_population=True, note="fp16 / int8-minmax, whole frame")
    r2.id = "TI007"
    print(f"# ratios: {r1.value} {r2.value}")

    for name in ("fp16", "int8-entropy", "int8-minmax"):
        emit(build_s[name], "s", layers(prof[name], f"the build of the {name} engine", name), rid=(
            "TI008" if name == "fp16" else "TI009" if name == "int8-entropy" else "TI010"),
            artifact=path("trt-int8-manifest.json"), statistic="single", n=1)

    # accuracy: one population per held-out frame, its sha256 in the item so the label cannot drift
    ids = 11
    for frame, arms in acc["frames"].items():
        key = frame.replace(".rgb", "")
        pop = Population.of("pixels of one frame", (f"{frame}:{frames.get(key, 'eval')}",),
                            f"{frame} (sha {frames.get(key, 'held-out')}), held out from calibration")
        for arm, row in arms.items():
            arm_name = "int8-entropy" if "entropy" in arm else "int8-minmax"
            for field, unit, stat, note in (
                ("max_codes", "levels", "max", "worst pixel, channel-max |delta| in 8-bit codes"),
                ("share_gt8_pct", "%", "single", "share of pixels off by more than 8 codes - the body's own bar"),
                ("psnr_db", "dB", "single", "PSNR against the fp16 engine"),
                ("mean_codes", "levels", "mean", "mean |delta| in 8-bit codes"),
            ):
                emit(row[field], unit, pop, rid=f"TI{ids:03d}", artifact=path("trt-int8-accuracy.json"),
                     statistic=stat, n=1 if stat != "mean" else row.get("n", 1),
                     note=f"{arm_name}, {note}", extra={"of": field, "arm": arm_name})
                ids += 1

    # -- the same-instrument run: the body, TRT's int8 and the fp16 engine in one process -------------
    if os.path.exists(a.probe):
        pr = json.load(open(a.probe))
        frame = pr["frames"][0]
        n_rounds = len(next(iter(pr["rounds"].values()))[1:])
        items = tuple(f"{frame}#round-{i}" for i in range(2, 2 + n_rounds))
        pop = Population.of("rounds", items, f"{frame}, interleaved rounds 2+ in one process, three arms")
        cfg = (f"2x-Futsuu-Anime(1).pth, {pr['shape'][2]}x{pr['shape'][3]}, one input frame, three arms "
               f"interleaved in one process")
        instr = ("torch.cuda.Event bracket over 10 iterations per round, 4 interleaved rounds, median of rounds 2+")
        dprobe = sha(a.probe)
        order = {"fp16-engine": "TV001", "trt-int8": "TV002", "triton-body": "TV003"}
        for arm, rid in order.items():
            st.emit(Record(value=float(pr["median_ms"][arm]), unit="ms", population=pop, instrument=instr,
                           config=cfg, session=st.session, statistic="median", tier="measured", id=rid,
                           run="scripts/trt_vs_body_probe.py", artifact=a.probe,
                           extra={"artifact_sha256": dprobe, "arm": arm,
                                  "rounds": pr["rounds"][arm]}))
        rb = st.ratio("TV002", "TV003", note="TRT int8 / Triton body: how much faster the body is")
        rb.id = "TV004"
        rf = st.ratio("TV001", "TV003", note="fp16 engine / Triton body")
        rf.id = "TV005"
        rt = st.ratio("TV001", "TV002", note="fp16 engine / TRT int8")
        rt.id = "TV006"
        print(f"# same-instrument: body vs TRT int8 {rb.value}x, body vs fp16 {rf.value}x, "
              f"TRT int8 vs fp16 {rt.value}x")
        n = 7
        for fname, arms in pr["accuracy"].items():
            p_pop = Population.of("pixels of one frame", (f"{fname}:{frames.get(fname.replace('.rgb', ''), 'eval')}",),
                                  f"{fname}, held out from calibration")
            for arm, row in arms.items():
                for field, unit, stat in (("max_codes", "levels", "max"), ("share_gt8_pct", "%", "single"),
                                          ("psnr_db", "dB", "single")):
                    st.emit(Record(value=float(row[field]), unit=unit, population=p_pop, instrument=instr,
                                   config=cfg, session=st.session, statistic=stat, tier="measured",
                                   id=f"TV{n:03d}", run="scripts/trt_vs_body_probe.py", artifact=a.probe,
                                   extra={"artifact_sha256": dprobe, "arm": arm, "of": field}))
                    n += 1

    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

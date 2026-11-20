"""Records for the fp16 engine's per-layer budget, read from two `trtexec --dumpProfile` logs.

Two builds of the same model, same shape, same precision, different builder optimisation level:
the default (3) and the level the app builds with (5). The profile table is the only place where
"what else is in the fp16 arm" can be read, so the numbers on the page come from here.

Each record's population is the layer set the statistic was summed over - the total is the whole
table, a group record is that group's layers. A group is a SUM of per-layer Avg(ms) over 20 runs,
so it is `mean` over n=20, not a single sample.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402

ROW = re.compile(r"\[I\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$")
BUILT = re.compile(r"Engine built in ([\d.]+) sec\.")
VERSION = re.compile(r"TensorRT version: ([\d.]+)")


def profile(path: str) -> tuple[list[tuple[float, str]], float, str]:
    """Rows of (Avg(ms), layer name), the engine build time in seconds, and the TRT version."""
    rows: list[tuple[float, str]] = []
    built, version = 0.0, ""
    in_profile = False
    for line in open(path):
        if "=== Profile" in line:
            in_profile = True
            continue
        if "=== Performance summary" in line:
            in_profile = False
        m = BUILT.search(line)
        if m:
            built = float(m.group(1))
        m = VERSION.search(line)
        if m:
            version = m.group(1)
        if not in_profile:
            continue
        m = ROW.search(line)
        if m and m.group(5).strip() != "Total" and "Time(ms)" not in line:
            rows.append((float(m.group(2)), m.group(5).strip()))
    if not rows:
        raise SystemExit(f"{path}: no profile rows - the log is not a --dumpProfile run")
    return rows, built, version


def group(rows, pred) -> list[tuple[float, str]]:
    return [(ms, n) for ms, n in rows if pred(n)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt3", default="harness/docs/wsp/verified/logs/fp16-engine/opt3-default.log")
    ap.add_argument("--opt5", default="harness/docs/wsp/verified/logs/fp16-engine/opt5.log")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/fp16-engine.jsonl")
    a = ap.parse_args(argv)

    r3, built3, ver3 = profile(a.opt3)
    r5, built5, ver5 = profile(a.opt5)
    if ver3 != ver5:
        raise SystemExit(f"the two logs come from different TensorRT versions: {ver3} vs {ver5}")

    def sha(path: str) -> str:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    d3, d5 = sha(a.opt3), sha(a.opt5)
    instrument = ("trtexec %s --dumpProfile: per-layer Avg(ms) summed over the table, each Avg the mean "
                  "of 20 runs (--avgRuns=20)" % ver3)
    model = "2x-Futsuu-Anime(1).pth, ONNX opset 17, static 1x3x1080x1920"
    s3, s5 = "s-2026-11-19T13:49:23+03:00", "s-2026-11-19T13:52:03+03:00"

    def pop(rows, label: str, tag: str) -> Population:
        # the build tag is part of the item: two builds of the same model share every layer NAME
        return Population.of("layers", tuple(f"{tag}/{n}" for _, n in rows), label)

    st = Store(session=s5)

    def emit(value: float, unit: str, rows, label: str, *, rid: str, config: str, session: str,
             artifact: str, digest: str, statistic: str = "mean", n: int = 20, note: str = "",
             extra: dict | None = None, tier: str = "measured", formula: str = "",
             inputs: tuple = ()) -> None:
        tag = "opt5" if "level 5" in config else "opt3"
        st.emit(Record(value=round(value, 3), unit=unit, population=pop(rows, label, tag),
                       instrument=instrument, config=config, session=session, statistic=statistic,
                       tier=tier, id=rid, formula=formula, inputs=inputs, run="trtexec --dumpProfile",
                       artifact=artifact, note=note,
                       extra={"artifact_sha256": digest, "engine_build_s": round(
                           built3 if artifact == a.opt3 else built5, 3), **(extra or {})}, ))

    # -- the default build (opt level 3) ------------------------------------------------------------
    c3 = f"{model}; --fp16, builderOptimizationLevel default (3)"
    emit(sum(ms for ms, _ in r3), "ms", r3, "every layer of the default-opt fp16 build", rid="FE001",
         config=c3, session=s3, artifact=a.opt3, digest=d3, note="sum of 60 layer Avg(ms)")
    prelu3 = group(r3, lambda n: "PRelu" in n and n.startswith("PWN("))
    emit(sum(ms for ms, _ in prelu3), "ms", prelu3, "the seventeen PReLU layers, default-opt build",
         rid="FE002", config=c3, session=s3, artifact=a.opt3, digest=d3,
         note="one separate full pass per conv (body.1..body.33, odd)")
    conv3 = group(r3, lambda n: re.search(r"/body\.\d+/Conv$", n) is not None
                  and "/body.0/" not in n and "/body.34/" not in n)
    emit(sum(ms for ms, _ in conv3), "ms", conv3, "the sixteen body convolutions, default-opt build",
         rid="FE003", config=c3, session=s3, artifact=a.opt3, digest=d3)
    copy3 = group(r3, lambda n: "opy" in n or "Reformat" in n)
    emit(sum(ms for ms, _ in copy3), "ms", copy3, "layout copies and reformats, default-opt build",
         rid="FE004", config=c3, session=s3, artifact=a.opt3, digest=d3)
    emit(len(r3), "count", r3, "every layer of the default-opt fp16 build", rid="FE005",
         config=c3, session=s3, artifact=a.opt3, digest=d3, statistic="single", n=1)
    emit(built3, "s", r3, "the default-opt build of this engine", rid="FE017", config=c3, session=s3,
         artifact=a.opt3, digest=d3, statistic="single", n=1)

    # -- the app's build (opt level 5) --------------------------------------------------------------
    c5 = f"{model}; --fp16, builderOptimizationLevel 5 (the app's opt level)"
    emit(sum(ms for ms, _ in r5), "ms", r5, "every layer of the opt-5 fp16 build", rid="FE006",
         config=c5, session=s5, artifact=a.opt5, digest=d5, note="sum of 25 layer Avg(ms)")
    hops = group(r5, lambda n: "CorrMulLtSele" in n)
    emit(sum(ms for ms, _ in hops), "ms", hops, "the sixteen hop kernels (conv+bias+PReLU+select fused)",
         rid="FE007", config=c5, session=s5, artifact=a.opt5, digest=d5)
    first = group(r5, lambda n: "MulLtSele" in n and "Corr" not in n)
    emit(sum(ms for ms, _ in first), "ms", first, "the first body conv at its own shape (head side)",
         rid="FE008", config=c5, session=s5, artifact=a.opt5, digest=d5)
    tail = group(r5, lambda n: "/body_34/Conv" in n)
    emit(sum(ms for ms, _ in tail), "ms", tail, "the tail conv 64 -> 12", rid="FE009",
         config=c5, session=s5, artifact=a.opt5, digest=d5)
    head = group(r5, lambda n: "/body_0/Conv" in n)
    emit(sum(ms for ms, _ in head), "ms", head, "the head conv 3 -> 64", rid="FE010",
         config=c5, session=s5, artifact=a.opt5, digest=d5)
    shuf = group(r5, lambda n: "TranReshMoveAdd" in n)
    emit(sum(ms for ms, _ in shuf), "ms", shuf, "pixel shuffle + residual add", rid="FE011",
         config=c5, session=s5, artifact=a.opt5, digest=d5, note="already one kernel")
    copy5 = group(r5, lambda n: "opy" in n or "Reformat" in n)
    emit(sum(ms for ms, _ in copy5), "ms", copy5, "layout copies and reformats, opt-5 build",
         rid="FE012", config=c5, session=s5, artifact=a.opt5, digest=d5,
         note="0.548 PReLU-output reformat + 0.144 Clip-input reformat; Move (0.066) and MaxMin (0.027) "
              "do not match this group")
    resize = group(r5, lambda n: "/Resize" in n)
    emit(sum(ms for ms, _ in resize), "ms", resize, "nearest-neighbour upsample of the LR input",
         rid="FE013", config=c5, session=s5, artifact=a.opt5, digest=d5)
    prelu5 = group(r5, lambda n: "PRelu" in n)
    emit(sum(ms for ms, _ in prelu5), "ms", prelu5, "the one reformat whose name mentions PRelu",
         rid="FE014", config=c5, session=s5, artifact=a.opt5, digest=d5,
         note="a layout copy, not a PReLU pass: at opt 5 the sixteen PReLU passes are inside the convs")
    emit(len(r5), "count", r5, "every layer of the opt-5 fp16 build", rid="FE015",
         config=c5, session=s5, artifact=a.opt5, digest=d5, statistic="single", n=1)
    emit(built5, "s", r5, "the opt-5 build of this engine", rid="FE016", config=c5, session=s5,
         artifact=a.opt5, digest=d5, statistic="single", n=1,
         note=f"the default-opt build of the same model took {built3:.1f} s (FE017)")

    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

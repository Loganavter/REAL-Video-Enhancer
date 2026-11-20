"""Records for the RRDBNet scope: arithmetic per frame, and the build that did not finish."""
from __future__ import annotations
import argparse, glob, hashlib, os, sys, torch
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402

MODELS = ["2x-sudo-RealESRGAN(1).pth", "2x_Pooh_RPLKSR_Candidate_1_387k.pth",
          "2x-UniScale-CartoonRestore-lite.pth", "2x-AnimeSharpV3.pth"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/jorj/Загрузки/Upscaling/Models/All")
    ap.add_argument("--log", default="docs/wsp/verified/logs/rrdb/build-killed.log")
    ap.add_argument("--out", default="docs/wsp/verified/records/rrdb-scope.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    st = Store(session="s-2026-11-19T17:35:00+03:00")
    n = 1
    for name in MODELS:
        p = os.path.join(a.dir, name)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        sd = sd.get("params", sd)
        convs = [v for k, v in sd.items() if k.endswith("weight") and v.dim() == 4]
        fl = sum(v.shape[1] * v.shape[0] * v.shape[2] * v.shape[3] * 2 * 1920 * 1080 for v in convs) / 1e12
        pop = Population.of("models", (name,), f"{name}, one 1920x1080 frame, LR-resolution body")
        st.emit(Record(value=round(fl, 2), unit="TFLOP", population=pop,
                       instrument="sum over the checkpoint's conv shapes of Cin*Cout*kh*kw*2*H*W",
                       config=f"{name}, {len(convs)} convolutions, 1920x1080 in", session=st.session,
                       statistic="single", tier="measured", id=f"RB{n:03d}",
                       run="harness/harness/scripts/rrdb_scope_records.py", artifact=p,
                       extra={"artifact_sha256": sha(p), "convs": len(convs)}))
        n += 1
    # the build that was killed: a LOWER BOUND on the engine build, and the reason the row closes
    log = a.log
    st.emit(Record(value=14.2, unit="min", population=Population.of(
        "process starts", ("rrdb#build-killed",), "one trtexec build of 2x-sudo-RealESRGAN at opt-5"),
        instrument="wall clock until the build was killed; the log shows it still compiling",
        config="trtexec --fp16 --builderOptimizationLevel=5, ONNX 16 MB, 95 convolutions", session=st.session,
        statistic="single", tier="measured", id=f"RB{n:03d}", run="~/.local/bin/trtexec", artifact=log,
        extra={"artifact_sha256": sha(log), "lower_bound": True}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

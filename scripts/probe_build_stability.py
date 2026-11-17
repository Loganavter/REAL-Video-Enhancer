"""Is a build repeatable? Rebuild one ONNX N times and record the tactic, the census and the time.

This exists because a five-way disagreement about C=32 was resolved by it: the ledger's "22.4-24.9 ms,
16 fused layers", E30a's "1.23x, 0 fused", B1's "0 myelin, 40.6 ms" and B3's "19 fused, 23.2 ms" were
not four graphs or two builders - they were **draws from one bimodal build**. Three rebuilds of one file
gave 39.23 / 39.29 / 20.95 ms with 0/0/18 fused layers.

Consequence for every synthesized-graph number in this repo: build more than once, print the conv tactic
and the tile, and never read `__myl_` alone - two plans with 19 fused layers were 1.76x apart.

Usage (one GPU job at a time):
  F=~/.cache/opencode/venvs/trt310/bin/python
  $F scripts/make_width_probe.py /tmp/opencode/p32.onnx --width-ch 32
  flock -n /var/tmp/opencode/gpu.lock $F scripts/probe_build_stability.py /tmp/opencode/p32.onnx \
      --trials 3 --emit /tmp/opencode/stability.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

sys.path.insert(0, "scripts")
from probe_prelu_forms import build, graph, load_engine, time_plan   # noqa: E402


def conv_tactics(plan: str) -> list[str]:
    import tensorrt as trt
    eng = load_engine(plan)
    insp = eng.create_engine_inspector()
    rows = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    out: list[str] = []
    for r in rows:
        t = r.get("TacticName", "")
        if t and ("xmma" in t or "scudnn" in t or "winograd" in t) and t not in out:
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tag", default="stability")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--workspace-gib", type=int, default=8)
    ap.add_argument("--out", default="/tmp/opencode")
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    rows = []
    for i in range(args.trials):
        plan = f"{args.out}/{args.tag}_trial{i}.plan"
        info = build(args.onnx, plan, args.height, args.width, fp16=True,
                     workspace_gib=args.workspace_gib)
        if "error" in info:
            rows.append({"trial": i, "error": info["error"][0][:160]})
            print(f"trial {i}: BUILD FAILED {info['error'][:1]}")
            continue
        eng = load_engine(plan)
        g = graph(eng)
        ms = time_plan(eng, args.height, args.width, iters=20)
        tactics = conv_tactics(plan)
        rows.append({"trial": i, "ms": round(ms, 2), "graph": g, "tactics": tactics[:3],
                     "bytes": info["bytes"], "sha256": info["sha256"]})
        print(f"trial {i}: {ms:7.2f} ms  myelin {g['fused_mylin']}  rf {g['reformat']}  "
              f"pw {g['pointwise']}  plan {info['bytes']} B")
        for t in tactics[:2]:
            print(f"    {t[:105]}")

    ok = [r for r in rows if "ms" in r]
    if ok:
        times = [r["ms"] for r in ok]
        print(f"\nspread: {min(times):.2f}-{max(times):.2f} ms  "
              f"({max(times) / min(times):.2f}x between draws of one file)")
        print(f"fused-layer counts: {sorted({r['graph']['fused_mylin'] for r in ok})}")
    print(f"\nNOT checked: whether a fast draw is reproducible under a timing cache (that is B2's "
          f"question); the picture (random weights)")

    if args.emit:
        import tensorrt as trt
        from harness import Population, Record, Store
        store = Store()
        for r in ok:
            pop = Population.of("rebuilds", [f"trial-{r['trial']}"],
                                label=f"{args.onnx}: {args.width}x{args.height}, fp16, built by "
                                      f"scripts/probe_prelu_forms.build with 8 GiB workspace")
            store.emit(Record(value=r["ms"], unit="ms", statistic="single", population=pop,
                              instrument="torch.cuda.Event around 20 executions after 6 warmups, "
                                         "one GPU job at a time",
                              config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}, "
                                     f"opt 5, avg_timing_iters 8, workspace "
                                     f"{args.workspace_gib} GiB",
                              session=store.session, run=f"stability-{args.tag}",
                              artifact=r["sha256"],
                              extra={"of": "frame_time", "quantity": "speed",
                                     "artifact_sha256": r["sha256"], "graph": r["graph"],
                                     "conv_tactics": r["tactics"]}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""What is inside a built plan: per-layer time, kernel/tactic name, and whether PReLU fused.

Written because two reports needed it and neither had a tool: the channel-width probe's "body" column
mixed fused myelin blocks with separate PReLU passes, and its rows did not add up to the total by 2 ms.
The engine inspector answers both - it reports the tactic name per layer, which is the only way to tell
"myelin fused the PReLU" from "myelin refused this shape".

Usage: inspect_plan.py <plan> [--height 1080] [--width 1920] [--all]
"""
import argparse
import json
import sys

import torch
import tensorrt as trt

LOG = trt.Logger(trt.Logger.ERROR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="print every layer, not just the top ones")
    args = ap.parse_args()

    engine = trt.Runtime(LOG).deserialize_cuda_engine(open(args.plan, "rb").read())
    inspector = engine.create_engine_inspector()
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    i_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    o_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    x = torch.rand(1, 3, args.height, args.width, device="cuda", dtype=torch.float32)
    y = torch.empty(tuple(engine.get_tensor_shape(o_name)), device="cuda", dtype=torch.float32)
    ctx.set_tensor_address(i_name, x.data_ptr())
    ctx.set_tensor_address(o_name, y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(5):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()

    class P(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.per = {}

        def report_layer_time(self, name, ms):
            if ms > 0:
                self.per[name] = self.per.get(name, 0.0) + ms

    prof = P()
    ctx.profiler = prof
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(args.iters):
        ctx.execute_async_v3(stream)
    e.record()
    torch.cuda.synchronize()
    total = s.elapsed_time(e) / args.iters
    per = {k: v / args.iters for k, v in prof.per.items()}

    rows = []
    for i in range(engine.num_layers):
        info = json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        name = info.get("Name", f"layer{i}")
        rows.append((name, per.get(name, 0.0), info.get("TacticName") or info.get("LayerType") or ""))

    measured = sum(r[1] for r in rows)
    print(f"total {total:.2f} ms | sum of layer times {measured:.2f} ms | "
          f"layers {engine.num_layers} | unattributed {total - measured:+.2f} ms")
    print(f"{'ms':>6}  {'name':<34} tactic")
    for name, ms, tactic in sorted(rows, key=lambda r: -r[1])[: (None if args.all else 12)]:
        print(f"{ms:6.2f}  {name[:34]:<34} {tactic[:70]}")

    # the two questions this tool exists for
    fused = [n for n, _, t in rows if "myl" in t.lower() or "myl" in n.lower()]
    prelu_pass = [n for n, _, _ in rows if "PRelu" in n and n not in fused]
    print(f"\nfused myelin kernels: {len(fused)} | separate PReLU passes: {len(prelu_pass)}")
    bodies = [n for n, _, _ in rows if "/body." in n and "PRelu" not in n]
    print(f"body conv layers visible by name: {len(bodies)}; "
          f"sum {sum(ms for n, ms, _ in rows if '/body.' in n and 'PRelu' not in n):.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

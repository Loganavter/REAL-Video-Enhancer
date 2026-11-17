"""Re-derive the residual-scale claim from the ledger, without spandrel.

The ledger says (2026-11-15, "a real 1080p frame"): correction |out - nearest(x)| median 0.9 levels,
p90 2.1, p99 10.1, p99.9 19.4, max 44.5, 98.43% of pixels under the 8-level bar. The population is
"a real 1080p frame" - no frame named, no hash, n=1 - so the number cannot be matched to another
run. This script measures it on a NAMED frame (sha256 of the raw pixels) and emits records, so the
result is checkable even though it is a different frame from the original.

Run: <torch-python> measure_residual.py <frame.raw> <W> <H> <model.pth> [out.jsonl]
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/scripts")


class Compact(nn.Module):
    """spandrel's SRVGGNetCompact, rebuilt: the checkpoint's own keys decide the layers."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        self.body = nn.ModuleList()
        i = 0
        while (key := f"body.{i}.weight") in state:
            w = state[key]
            if w.dim() == 4:
                self.body.append(nn.Conv2d(w.shape[1], w.shape[0], 3, 1, 1))
            elif w.dim() == 1:
                self.body.append(nn.PReLU(num_parameters=w.shape[0]))
            else:
                raise SystemExit(f"unexpected tensor {key} {tuple(w.shape)}")
            i += 1
        tail = state[f"body.{i - 1}.weight"]
        self.scale = int(round((tail.shape[0] / 3) ** 0.5))
        self.upsampler = nn.PixelShuffle(self.scale)
        # Load here, not in the caller. A Compact built and not loaded is a RANDOM network: on
        # 2026-11-17 exactly that produced a 43% "difference" between two references that should have
        # been identical. A class that silently yields a random model is a trap.
        self.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(0.0, 1.0)
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def main() -> int:
    frame, w, h, model_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out_path = sys.argv[5] if len(sys.argv) > 5 else ""

    raw = np.fromfile(frame, dtype=np.uint8)
    if raw.size != w * h * 3:
        raise SystemExit(f"{frame}: {raw.size} bytes is not {w}x{h}x3")
    sha = hashlib.sha256(raw.tobytes()).hexdigest()

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    model = Compact(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    scale = model.scale

    x = torch.from_numpy(raw.reshape(h, w, 3).transpose(2, 0, 1)).float().div_(255.0)[None]
    with torch.no_grad():
        out = model(x)

    corr = (out - F.interpolate(x, scale_factor=scale, mode="nearest")).abs().mul(255.0).flatten().numpy()
    q = {name: float(np.percentile(corr, p)) for name, p in
         (("median", 50), ("p90", 90), ("p99", 99), ("p99.9", 99.9))}
    mx = float(corr.max())
    under = float((corr <= 8.0).mean() * 100.0)

    print(f"frame sha256 {sha[:16]}  {w}x{h}  scale {scale}x  shape {tuple(out.shape)}")
    print(f"median {q['median']:.2f}  p90 {q['p90']:.2f}  p99 {q['p99']:.2f}  "
          f"p99.9 {q['p99.9']:.2f}  max {mx:.2f}  under 8 levels {under:.2f}%")

    if out_path:
        from harness import Population, Record, Store
        s = Store()
        pop = Population.of("pixels of one 1080p frame", [sha[:16]],
                            label=f"{frame.split('/')[-1]} {w}x{h}, correction |out-nearest|")
        common = dict(population=pop, instrument="torch 2.14.1+cpu, SRVGGNetCompact forward",
                      config=f"ckpt-{hashlib.sha256(open(model_path,'rb').read()).hexdigest()[:16]}",
                      session=s.session, run="run-residual-2026-11-16",
                      extra={"artifact_sha256": sha, "of": "levels", "quantity": "accuracy"})
        made = []
        for name, stat in (("median", "median"), ("p90", "p90"), ("p99", "p99"), ("p99.9", "p99.9")):
            made.append(s.emit(Record(value=q[name], unit="levels", statistic=stat, **common)))
        made.append(s.emit(Record(value=mx, unit="levels", statistic="max", **common)))
        made.append(s.emit(Record(value=under, unit="%", statistic="single", **common)))
        s.dump(out_path)
        print("records:", " ".join(r.id for r in made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

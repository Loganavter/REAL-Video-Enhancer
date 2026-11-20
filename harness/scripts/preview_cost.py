"""What the preview cost before it became a thumbnail, on the same frame, measured both ways.

The removed implementation built the preview from the whole frame: cast the packed yuv420p
payload to float, run the BT.709 matrix over all 8.3 Mpx, clamp, pack to RGB and transfer a
full 4K picture to the host - on every frame, for a viewer whose reader polls at 2-5 Hz. The
replacement samples one row and one column in eight and transfers 0.69 MB.

Both bodies are kept side by side here, so the difference stays a measurement instead of a
memory: the "old" body is a faithful copy of `Frame.raw_payload_preview` as it stood at
`61f0cbb2` (removed in the preview refactor), the "new" one is `Frame.preview_small`.

Not an acceptance statistic: one frame, one card, one session. It prices the two
implementations; the loop's own `preview ... ms` line is what an acceptance run reads.

Usage:
  python harness/scripts/preview_cost.py [--size 3840x2160] [--preview 640x360] [--iters 10]
"""

from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _bos.path.join(_ROOT, "backend"), _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import statistics
import time

import torch

from src.utils.Frame import Frame


def parse_size(value):
    width, height = (int(part) for part in str(value).lower().split("x", 1))
    return width, height


def make_payload_frame(width, height, device):
    """A frame holding a packed yuv420p payload, exactly what the int8 tail emits."""
    frame = Frame(
        backend="pytorch", width=width, height=height, device=device, gpu_id=0,
        hdr_mode=False, dtype="float32",
    )
    n = width * height
    c = (height // 2) * (width // 2)
    torch.manual_seed(0)
    frame._tensor = torch.randint(0, 256, (n + 2 * c,), dtype=torch.uint8, device=device)
    return frame


def old_preview(frame):
    """`Frame.raw_payload_preview` at 61f0cbb2: whole frame in float, full-size RGB D2H."""
    flat = frame._tensor.reshape(-1).float()
    h, w = int(frame.height), int(frame.width)
    n = h * w
    c = (h // 2) * (w // 2)
    y = flat[:n].view(h, w) - 16.0
    u = (flat[n:n + c].view(h // 2, w // 2) - 128.0).repeat_interleave(2, 0).repeat_interleave(2, 1)
    v = (flat[n + c:n + 2 * c].view(h // 2, w // 2) - 128.0).repeat_interleave(2, 0).repeat_interleave(2, 1)
    u, v = u[:h, :w], v[:h, :w]
    r = 1.164 * y + 1.793 * v
    g = 1.164 * y - 0.213 * u - 0.533 * v
    b = 1.164 * y + 2.112 * u
    rgb = torch.stack((r, g, b), dim=-1).clamp_(0.0, 255.0).to(torch.uint8)
    return rgb.cpu().numpy().tobytes()


def time_it(fn, frame, iters):
    samples = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        payload = fn(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
        del payload
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="3840x2160", help="output frame WxH (the payload's size)")
    parser.add_argument("--preview", default="640x360", help="thumbnail WxH")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    width, height = parse_size(args.size)
    pw, ph = parse_size(args.preview)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = make_payload_frame(width, height, device)
    old_preview(frame)  # warm-up: the first call pays module init, not the work
    frame.preview_small(pw, ph)

    def new_preview(f):
        return f.preview_small(pw, ph)

    old = statistics.median(time_it(old_preview, frame, args.iters))
    new = statistics.median(time_it(new_preview, frame, args.iters))
    new_bytes = len(new_preview(frame)[0])
    old_bytes = width * height * 3
    print(f"frame {width}x{height} on {device}, thumbnail {pw}x{ph}, median of {args.iters}")
    print(f"  old (whole-frame float + full D2H): {old:.2f} ms, {old_bytes / 1e6:.1f} MB to the host")
    print(f"  new (sampled planes, thumbnail D2H): {new:.2f} ms, {new_bytes / 1e6:.2f} MB to the host")
    print(f"  ratio: {old / new:.1f}x")
    print("uncovered scope: one frame, one card, one session; this prices the two bodies, "
          "it is not the loop's per-frame cost (see the render's `preview ... ms` line).")
    return 0


if __name__ == "__main__":
    _bsys.exit(main())

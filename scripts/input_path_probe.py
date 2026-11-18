"""What the input path costs per frame: the clone in get_frame_tensor and the trailing sync_stream.

`Frame.get_frame_tensor` returns `self._tensor.clone()` on every call ("have to test later") and
`yuv420p10_to_tensor` ends with an explicit `sync_stream`. Both sit inside the wrapper the frame
accounting could not attribute, and neither has ever been timed on its own (row
`rve-input-clone-sync-2026-11-19`, from the researcher's C9).

Arms, round-robin inside one process so a warm-up trend lands on every arm:

  conv          yuv420p10_to_tensor as shipped (trailing sync_stream included)
  conv+clone    the same plus the `.clone()` get_frame_tensor performs          <- shipped path
  conv-nosync   as shipped, with sync_stream patched to a no-op
  nosync+clone  both removals together
  clone-only    `.clone()` of a frame already converted: the clone's cost alone

Every arm keeps its last two results alive, as the app does while the frame is in the queue, so the
allocator sees the same reuse pattern on all four.

Reported per arm: CUDA-event ms and wall ms per call, medians of rounds 2+ (the first draw carries the
warm-up) and the paired differences between arms, which is what attributes the cost.

Usage: python scripts/input_path_probe.py [--width 1920] [--height 1080]
                                          [--rounds 5] [--iters 40] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))

from src.pytorch.TorchUtils import TorchUtils  # noqa: E402

ARMS = ("conv", "conv+clone", "conv-nosync", "nosync+clone", "clone-only")


def build_utils(width: int, height: int, hdr: bool) -> TorchUtils:
    """The shipped converter, without the app's device-detection bootstrap.

    `TorchUtils.__init__` and `yuv420p10_to_tensor` only need width/height and the stream helpers;
    `handle_device` additionally calls `BackendDetect.get_gpus_torch()`, which needs `torchvision`
    (absent from this offline venv) and is irrelevant to what is being timed, so the stream is made
    here instead of through `init_stream`.
    """
    return TorchUtils(width=width, height=height, device_type="cuda", hdr_mode=hdr, gpu_id=0,
                      high_bitdepth=hdr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = build_utils(args.width, args.height, hdr=True)
    stream = torch.cuda.Stream(device=dev)

    n = args.width * args.height
    # a yuv420p10le payload: n luma + 2 * (h/2 * w/2) chroma uint16 = 3 bytes/pixel - the input pipe
    # width the app actually feeds this function
    payload = np.empty(n + n // 2, dtype=np.uint16)
    payload[:n] = np.random.randint(64, 960, size=n, dtype=np.uint16)
    payload[n:] = np.random.randint(64, 960, size=n // 2, dtype=np.uint16)
    frame_bytes = payload.tobytes()
    print(f"payload {len(frame_bytes) / 1e6:.2f} MB per frame at {args.width}x{args.height}, "
          f"{args.rounds} rounds x {args.iters} calls")

    def conv(sync: bool):
        if sync:
            return tu.yuv420p10_to_tensor(frame_bytes, stream, dev, torch.float16)
        old = tu.sync_stream
        tu.sync_stream = lambda *_a, **_k: None
        try:
            return tu.yuv420p10_to_tensor(frame_bytes, stream, dev, torch.float16)
        finally:
            tu.sync_stream = old

    static = tu.yuv420p10_to_tensor(frame_bytes, stream, dev, torch.float16)
    torch.cuda.synchronize()
    print(f"frame tensor {tuple(static.shape)} {static.dtype}, {static.numel() * 2 / 1e6:.2f} MB")

    hold: list = []

    def run(arm: str):
        if arm == "conv":
            out = conv(True)
        elif arm == "conv+clone":
            out = conv(True).clone()
        elif arm == "conv-nosync":
            out = conv(False)
        elif arm == "nosync+clone":
            out = conv(False).clone()
        elif arm == "clone-only":
            out = static.clone()
        else:
            raise SystemExit(arm)
        hold.append(out)
        del hold[:-2]
        return out

    def time_arm(arm: str) -> tuple[float, float]:
        for _ in range(3):
            run(arm)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        w0 = time.perf_counter()
        ev0.record()
        for _ in range(args.iters):
            run(arm)
        ev1.record()
        torch.cuda.synchronize()
        w1 = time.perf_counter()
        # perf_counter is seconds; every number in this file is milliseconds
        return ev0.elapsed_time(ev1) / args.iters, (w1 - w0) * 1000.0 / args.iters

    draws = {arm: {"event": [], "wall": []} for arm in ARMS}
    print(f"{'round':>5} " + "  ".join(f"{a:>15}" for a in ARMS) + "   (event / wall, ms/call)")
    for r in range(1, args.rounds + 1):
        cells = []
        for arm in ARMS:
            ev_ms, wall_ms = time_arm(arm)
            draws[arm]["event"].append(ev_ms)
            draws[arm]["wall"].append(wall_ms)
            cells.append(f"{ev_ms:6.3f}/{wall_ms:6.3f}")
        print(f"{r:>5} " + "  ".join(f"{c:>15}" for c in cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {arm: {"event": statistics.median(draws[arm]["event"][tail]),
                 "wall": statistics.median(draws[arm]["wall"][tail]),
                 "event_min": min(draws[arm]["event"]),
                 "wall_min": min(draws[arm]["wall"])} for arm in ARMS}
    print("\nmedians of rounds 2+ (ms/call), and the min draw as the least-disturbed sample:")
    for arm in ARMS:
        print(f"  {arm:>13}: event {med[arm]['event']:7.3f} (min {med[arm]['event_min']:6.3f})"
              f"   wall {med[arm]['wall']:7.3f} (min {med[arm]['wall_min']:6.3f})")

    def diff(a: str, b: str) -> tuple[float, float]:
        """paired per-round differences b - a, median and range; pairing removes the round's drift."""
        d = [draws[b]["event"][i] - draws[a]["event"][i] for i in range(len(draws[a]["event"]))]
        dw = [draws[b]["wall"][i] - draws[a]["wall"][i] for i in range(len(draws[a]["wall"]))]
        return statistics.median(d[len(d) // 2:]), statistics.median(dw[len(dw) // 2:])

    print("\npaired differences (later rounds only), median:")
    for a, b, why in (("conv", "conv+clone", "clone cost"),
                      ("conv-nosync", "nosync+clone", "clone cost, no sync"),
                      ("conv-nosync", "conv", "sync cost"),
                      ("conv", "nosync+clone", "both, on the shipped arm"),
                      ("clone-only", "clone-only", "clone alone (absolute)")):
        de, dw = diff(a, b)
        if a == b:
            print(f"  {why:>26}: event {med[a]['event']:7.3f}  wall {med[a]['wall']:7.3f}")
        else:
            print(f"  {why:>26}: event {de:+7.3f}  wall {dw:+7.3f}   ({b} - {a})")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "width": args.width, "height": args.height, "hdr": True,
                       "rounds": args.rounds, "iters": args.iters,
                       "payload_mb": round(len(frame_bytes) / 1e6, 2),
                       "frame_mb": round(static.numel() * 2 / 1e6, 2),
                       "draws": draws, "median": med}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

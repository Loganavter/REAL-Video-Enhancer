"""A preview consumer without the GUI, so the preview's cost can be measured headless.

The preview channel's producer lives in the render loop and its reader lives in the Qt
window, which meant the cost of a preview could only ever be observed by a human watching
a GUI. Every log in harness/docs/wsp/verified/logs reads `preview 0.00 ms` for exactly that
reason: those runs had no consumer, so the renderer built nothing.

This instrument is that consumer. It allocates the same block the GUI allocates
(header + two slots, layout from `backend/src/utils/PreviewChannel.py`), asks for frames
at a viewer's cadence, copies what arrives, and prints what it saw. Run it first, then
start the render with the same `--preview_shared_memory_id` and `--preview_size`; the
render's own profile line then carries the per-frame cost next to `preview`.

Usage:
  python harness/scripts/preview_sink.py --id /preview_bench --size 640x360 --seconds 30
  # in another shell:
  backend/py -W ignore backend/rve-backend.py -i CLIP -o OUT -b tensorrt \
      --preview_shared_memory_id /preview_bench --preview_size 640x360

What it prints on exit: frames consumed, bytes copied, the wall clock it was attached
for, and the per-consume copy cost - the reader's share, as distinct from the producer's.
It also prints the uncovered scope: it validates geometry and format, never the picture.
"""

from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import struct
import time
from multiprocessing import shared_memory

HEADER = struct.Struct("<IHHBBBI")   # mirrors backend/src/utils/PreviewChannel.py
FMT_RGB888 = 1
FMT_YUV420 = 2
OFF_WANTED = 9
POLL_SECONDS = 0.1                   # the GUI's UpdateGUIThread polls at 10 Hz


def parse_size(value):
    width, height = (int(part) for part in str(value).lower().split("x", 1))
    return width, height


def payload_length(fmt, width, height):
    if fmt == FMT_RGB888:
        return width * height * 3
    return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="shared memory name, e.g. /preview_bench")
    parser.add_argument("--size", default="640x360", help="WxH, must match --preview_size")
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to stay attached")
    parser.add_argument("--poll", type=float, default=POLL_SECONDS, help="reader poll interval")
    parser.add_argument("--leave", action="store_true",
                        help="do not unlink the block on exit (a second consumer will attach)")
    args = parser.parse_args()

    width, height = parse_size(args.size)
    slot_bytes = width * height * 3
    size = HEADER.size + 2 * slot_bytes
    shm = shared_memory.SharedMemory(name=args.id, create=True, size=size)
    shm.buf[OFF_WANTED] = 1
    frames = 0
    copied = 0
    seq_seen = 0
    gaps = []
    last_arrival = time.time()
    started = last_arrival
    copy_ms = []
    formats = {FMT_RGB888: 0, FMT_YUV420: 0}
    try:
        while time.time() - started < args.seconds:
            seq, w, h, fmt, _wanted, _flags, _reserved = HEADER.unpack_from(shm.buf, 0)
            if seq != seq_seen and w and h and fmt in formats:
                seq_seen = seq
                # the slot stride is 3 bytes/px whatever the format carries
                offset = HEADER.size + (seq & 1) * (w * h * 3)
                length = payload_length(fmt, w, h)
                t0 = time.perf_counter()
                pixels = bytes(shm.buf[offset:offset + length])
                copy_ms.append((time.perf_counter() - t0) * 1000.0)
                frames += 1
                formats[fmt] = formats.get(fmt, 0) + 1
                copied += len(pixels)
                now = time.time()
                gaps.append(now - last_arrival)
                last_arrival = now
                # a viewer consumed a frame: ask for the next one
                shm.buf[OFF_WANTED] = 1
            time.sleep(args.poll)
    finally:
        attached = time.time() - started
        shm.close()
        if not args.leave:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass  # the producer closed it first; the block is already gone

    copy_ms.sort()
    print(f"preview sink: {args.id} {width}x{height} ({size} bytes, {size / 1e6:.2f} MB)")
    print(f"attached {attached:.1f} s, consumed {frames} frames "
          f"({frames / attached if attached else 0:.2f} fps), {copied / 1e6:.2f} MB copied")
    print(f"formats: rgb888 {formats[FMT_RGB888]}, sampled yuv420 {formats[FMT_YUV420]}")
    if copy_ms:
        median = copy_ms[len(copy_ms) // 2]
        print(f"reader copy: median {median:.3f} ms, max {copy_ms[-1]:.3f} ms per frame")
    if len(gaps) > 1:
        gaps_sorted = sorted(gaps[1:])
        print(f"interval between frames: median {gaps_sorted[len(gaps_sorted) // 2] * 1000:.0f} ms, "
              f"max {gaps_sorted[-1] * 1000:.0f} ms")
    print("uncovered scope: geometry/format only - this instrument does not judge the picture, "
          "and a zero frame count means the producer never saw `wanted`.")
    return 0


if __name__ == "__main__":
    _bsys.exit(main())

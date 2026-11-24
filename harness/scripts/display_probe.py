"""Watch the display channel of a running render: the pane's own viewer, as a probe.

The app allocates the block and reads it; this does the same from a shell, so the channel and the seek request can be
verified without a window (the GUI wiring is another lane). It is also the instrument for "does the export pay for a
pane nobody watches": run it with --wanted 0 and the renderer must not publish at all.

    python3 harness/scripts/display_probe.py --run -- python3 ... backend/rve-backend.py ... --display_state_id NAME
    python3 harness/scripts/display_probe.py --attach NAME [--seconds 8] [--seek 3.5]

The header is the one `backend/src/utils/DisplayChannel.py` writes: `<IHHBBBI d` = seq, width, height, fmt, wanted, flags,
reserved (15 bytes, little-endian, no padding; `wanted` at byte 9), then two slots of box_w * box_h * 3.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time

#: Keep in step with `backend/src/utils/DisplayChannel.py` (and the app's `src/ui/video/channel.py`): 23 bytes, the
#: trailing double is the picture's media time (-1 = the writer could not name it). A 15-byte copy here made the probe
#: read the slots 8 bytes early and show a torn picture.
HEADER = struct.Struct("<IHHBBBI d")
FMT_RGB888 = 1
OFF_WANTED = 9


def read_header(buf):
    return HEADER.unpack_from(buf, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attach", default="", help="an existing block (the app allocated it)")
    ap.add_argument("--run", action="store_true", help="allocate the block, then run the command after --")
    ap.add_argument("--name", default="", help="block name; defaults to /rve_display_probe<pid>")
    ap.add_argument("--box", default="960x540")
    ap.add_argument("--wanted", type=int, default=1, help="1 asks the renderer to publish, 0 asks it not to")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seek", type=float, default=None, help="request this time (seconds) after 2 s")
    args, rest = ap.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    width, height = (int(v) for v in args.box.lower().split("x", 1))

    from multiprocessing import shared_memory

    name = args.attach or args.name or f"/rve_display_probe{os.getpid()}"
    owner = False
    block = None
    if args.attach:
        for attempt in range(100):
            try:
                block = shared_memory.SharedMemory(name=name)
                break
            except FileNotFoundError:
                time.sleep(0.05)
        if block is None:
            raise SystemExit(f"no block named {name}")
    else:
        size = HEADER.size + 2 * width * height * 3
        try:
            block = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
            block = shared_memory.SharedMemory(name=name, create=True, size=size)
        owner = True
        block.buf[OFF_WANTED] = 1 if args.wanted else 0

    seek_block = None
    seek_name = name + "_seek"
    if not args.attach:
        seek_block = shared_memory.SharedMemory(name=seek_name, create=True, size=16)
        struct.pack_into("<dI", seek_block.buf, 0, 0.0, 0)

    process = None
    if args.run and rest:
        if seek_block is None:
            seek_block = shared_memory.SharedMemory(name=seek_name, create=True, size=16)
            struct.pack_into("<dI", seek_block.buf, 0, 0.0, 0)
        command = list(rest) + ["--display_state_id", name, "--display_size", f"{width}x{height}",
                                "--seek_state_id", seek_name]
        print("running:", " ".join(command[:6]), "...")
        process = subprocess.Popen(command)

    frames = 0
    last_seq = 0
    first = None
    seeked_at = None
    started = time.time()
    try:
        while time.time() - started < args.seconds:
            # A request written before the backend attaches is part of the generation it starts from, i.e. it is
            # treated as already acted on (the baseline is read at attach). So ask only once the renderer is alive -
            # the first published frame is the proof that it is.
            if (args.seek is not None and seeked_at is None and first is not None
                    and time.time() - started > first + 0.5):
                generation = struct.unpack_from("<dI", seek_block.buf, 0)[1]
                struct.pack_into("<dI", seek_block.buf, 0, float(args.seek), generation + 1)
                seeked_at = time.time() - started
                print(f"seek requested: t={args.seek}s (generation {generation + 1})")
            seq, w, h, fmt, wanted, _flags, _reserved, _stamp = read_header(block.buf)
            if seq != last_seq:
                last_seq = seq
                frames += 1
                if first is None:
                    first = time.time() - started
                if seeked_at is not None and time.time() - started > seeked_at:
                    print(f"  frame after seek: seq={seq} {w}x{h} fmt={fmt} at t={time.time() - started:.2f}s")
                    seeked_at = float("inf")
            time.sleep(0.05)
    finally:
        seq, w, h, fmt, wanted, _flags, _reserved, _stamp = read_header(block.buf)
        print(f"\nframes seen: {frames}  last seq: {seq}  size {w}x{h}  fmt {fmt}  wanted {wanted}")
        print(f"first frame after {first if first is not None else 'never'} s"
              f"{'' if frames else '  <- the renderer published nothing'}")
        if process is not None:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
        block.close()
        if owner:
            block.unlink()
        if seek_block is not None:
            seek_block.close()
            if not args.attach:
                try:
                    seek_block.unlink()
                except Exception:
                    pass
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())

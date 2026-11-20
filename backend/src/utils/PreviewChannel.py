"""The preview channel: a small, fixed-size thumbnail of the frame being rendered.

The preview used to be the output frame itself - a 4K RGB buffer copied to the host on
every frame (`Frame.raw_payload_preview` built it with a whole-frame float cast and a
full D2H) and handed to a GUI that immediately shrank it to a ~600 px label with a
nearest-neighbour scale. Against the measured pack (12.44 MB in 2.06 ms, FRAME-BUDGET)
that is ~2 ms of host transfer per frame for a picture nobody sees at more than 5 Hz.

This module carries a thumbnail instead, and three things make it cheap:

* the pixels are sampled in the tensor (one row/column in eight), never read whole;
* the block is `width*height*3`, so the copy is 0.69 MB at 640x360 instead of 24.9 MB;
* a `wanted` byte means the renderer builds one only after the viewer consumed the last.

Layout (little-endian); the GUI side mirrors it in `src/ui/QTcustom.py`:

    0   u32  seq      incremented after a complete write; bit 0 = slot
    4   u16  width
    6   u16  height
    8   u8   fmt      1 = RGB888
    9   u8   wanted   1 = the viewer wants a frame; the renderer clears it on publish
    10  u8   flags    bit 0 = the frame came from a high-bitdepth source
    11  u8   pad
    12  u32  reserved
    16  slot 0: width*height*3 bytes
        slot 1: width*height*3 bytes

Two slots plus the sequence number are what stop a reader from seeing half of one frame
and half of the next. The old channel had no lock and no sequence at all, and its reader
padded short reads with NULs, so a torn frame was drawn as valid.

The block is created by the GUI before the backend starts. A backend that cannot find it
within `wait_seconds` keeps rendering without a preview - the old code spun forever in
that case, so a GUI that died before `create` left the render hung at frame zero.
"""

import struct
import time
from multiprocessing import shared_memory

HEADER = struct.Struct("<IHHBBBI")
FMT_RGB888 = 1
FMT_YUV420 = 2          # sampled planes: Y (w*h), then U and V at ((w+1)//2)x((h+1)//2)
FLAG_HIGH_BITDEPTH = 1

OFF_SEQ = 0
OFF_WIDTH = 4
OFF_HEIGHT = 6
OFF_FMT = 8
OFF_WANTED = 9
OFF_FLAGS = 10


def payload_size(fmt, width, height):
    """Bytes a thumbnail of this format occupies (the slot stride stays 3 bytes/px)."""
    if fmt == FMT_RGB888:
        return width * height * 3
    if fmt == FMT_YUV420:
        return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
    raise ValueError(f"unknown preview format {fmt!r}")


def slot_stride(width, height):
    """The block reserves 3 bytes/px per slot whatever the format, so a swap of format
    between two frames cannot move a slot under the reader."""
    return width * height * 3


class PreviewChannel:
    """Writer side of the preview block. One per render; cheap to ask, cheap to fill."""

    def __init__(self, shm_id, wait_seconds=10.0):
        self.name = shm_id
        self.shm = None
        if not shm_id:
            return
        deadline = time.time() + float(wait_seconds)
        announced = False
        while True:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_id)
                break
            except FileNotFoundError:
                if time.time() >= deadline:
                    return
                if not announced:
                    print(f"preview: waiting for {shm_id} to be created (up to {wait_seconds:.0f} s)")
                    announced = True
                time.sleep(0.05)

    @property
    def attached(self):
        return self.shm is not None

    def wanted(self):
        """True when the viewer has consumed the last frame and wants the next one."""
        if self.shm is None:
            return False
        return self.shm.buf[OFF_WANTED] == 1

    def publish(self, payload, width, height, fmt=FMT_RGB888, high_bitdepth=False):
        """Write one thumbnail into the inactive slot, then bump the sequence.

        The header is written last and in one `pack_into`, so a reader that sees the new
        `seq` also sees the matching geometry and format. A length mismatch is refused
        loudly: a preview is not a measurement, but a half-written block is a bug.
        """
        if self.shm is None:
            return False
        width, height = int(width), int(height)
        expected = payload_size(fmt, width, height)
        if len(payload) != expected:
            raise ValueError(f"preview payload is {len(payload)} bytes, {fmt} at "
                             f"{width}x{height} is {expected}")
        seq = struct.unpack_from("<I", self.shm.buf, OFF_SEQ)[0]
        seq = (seq + 1) & 0xFFFFFFFF
        slot = seq & 1
        offset = HEADER.size + slot * slot_stride(width, height)
        self.shm.buf[offset:offset + expected] = payload
        HEADER.pack_into(
            self.shm.buf, OFF_SEQ,
            seq, width, height, fmt, 0,
            FLAG_HIGH_BITDEPTH if high_bitdepth else 0, 0,
        )
        return True

    def close(self):
        if self.shm is not None:
            try:
                self.shm.close()
            finally:
                self.shm = None

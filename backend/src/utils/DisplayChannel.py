"""The composited-frame display channel: one writer (the renderer), one reader (the window pane).

This is *not* the deleted 640x360 thumbnail channel. The pane sizes this block to its own box before the
render starts, and the renderer publishes into it only while the pane says it wants frames
(``wanted != 0``) - a run with no viewer attached pays one byte read per frame and nothing else.

Layout (little-endian, the invariant both sides copy verbatim; the reader's copy lives in
``src/ui/QTcustom.py``, the writer's here)::

    offset  size  field
    0       4     seq      uint32   bumped AFTER the payload is in place; the newest complete
                                    slot is ``(seq - 1) & 1``
    4       2     width    uint16   width of the payload in the newest slot (box fit, aspect kept)
    6       2     height   uint16
    8       1     fmt      uint8    1 = interleaved RGB888
    9       1     wanted   uint8    0 = the viewer is not asking; the writer publishes nothing
    10      1     flags    uint8    reserved for the writer, 0 today
    11      4     reserved uint32   0
    15      8     time     double   media time of the picture in the newest slot, seconds;
                                    ``TIME_UNSET`` (-1) when the writer cannot name it
    23      ...   two payload slots, each ``box_w * box_h * 3`` bytes

(``struct.Struct("<IHHBBBI")``: four-byte seq, two shorts, three bytes, then the reserved word - 15
bytes. ``wanted`` is the byte at 9, which is the one offset both sides must agree on by name.)

The writer fills slot ``seq & 1``, then writes the header with ``seq + 1``. Two slots exist so the
reader can read the slot the writer is not filling; the reader validates by reading ``seq`` before and
after the copy and retrying a torn read.

The seek request block (16 bytes, `SeekState`) rides on the same machinery: ``<dI`` =
``requested_seconds`` (double) + ``generation`` (uint32), newest wins, one slot, 4 trailing reserved
bytes so the block is 16.
"""

from __future__ import annotations

import struct

#: The header. `<` = little-endian, standard sizes, no padding - 23 bytes. The `I` is the reserved word
#: (4 bytes) and the trailing `d` is the frame's media time, so a reader that assumed a 15-byte header
#: would read the slots 8 bytes early; both sides must derive `HEADER_SIZE` from this struct, never
#: hard-code it. The time exists because a sequence number alone cannot answer "is this the frame the
#: seek asked for" (owner, 2026-11-24: the pane's log could name the publication but not its content).
HEADER = struct.Struct("<IHHBBBI d")
HEADER_SIZE = HEADER.size

OFFSET_SEQ = 0
OFFSET_WIDTH = 4
OFFSET_HEIGHT = 6
OFFSET_FMT = 8
OFFSET_WANTED = 9
OFFSET_FLAGS = 10
OFFSET_RESERVED = 11
OFFSET_TIME = 15

#: `time` value meaning "the writer cannot name this picture". Media times are >= 0, so -1 is free.
TIME_UNSET = -1.0

#: `fmt` value of the payload: three interleaved 8-bit channels - the one format a pane can blit.
FMT_RGB888 = 1
RGB_BYTES_PER_PIXEL = 3

#: The seek request: `requested_seconds` then `generation`. 12 bytes of struct inside a 16-byte block.
SEEK_STRUCT = struct.Struct("<dI")
SEEK_BLOCK_SIZE = 16
UINT32_MAX = 0xFFFFFFFF


def _open_block(name: str, size: int):
    """Attach to a GUI-owned segment, or create it when the GUI has not got there yet.

    Returns `(segment, created)`. The owner of a name is the GUI (see `utils/shm.py`); the backend only
    creates as a fallback so a render started before the window has attached still has a block, and it
    remembers that it created it so `close()` can clean up a segment nobody else will.
    """
    from .shm import attach, create

    try:
        return attach(name), False
    except FileNotFoundError:
        return create(name, size), True


class DisplayChannel:
    """The writer's end of the composited-frame block.

    `name` is the segment the GUI created (`/image_display<pid>` style); `box` is the pane size in
    pixels, which is what the two payload slots are sized for. With no name the channel is inert:
    `publish` answers False and the renderer never touches it.
    """

    def __init__(self, name: str | None, box=(640, 360)):
        self.name = name
        self.box = (int(box[0]), int(box[1]))
        self.created = False
        self._shm = None
        self._size_warned = False
        if name:
            try:
                self._shm, self.created = _open_block(
                    name, HEADER_SIZE + 2 * (self.box[0] * self.box[1] * RGB_BYTES_PER_PIXEL))
            except Exception:
                # A display that cannot be attached must never take a render down: the export is the
                # job, the pane is the courtesy.
                self._shm = None
        self.attached = self._shm is not None and len(self._shm.buf) >= HEADER_SIZE + RGB_BYTES_PER_PIXEL
        #: The bytes each slot really has (the GUI's segment is the authority, not our box).
        self.slot_size = (len(self._shm.buf) - HEADER_SIZE) // 2 if self.attached else 0

    # -- reader side of the same block (used by the pane, kept here as the layout's single source) ---
    def wanted(self) -> bool:
        """Whether the pane is asking for frames right now. False for an inert channel."""
        if not self.attached:
            return False
        return self._shm.buf[OFFSET_WANTED] != 0

    @property
    def seq(self) -> int:
        if not self.attached:
            return 0
        return HEADER.unpack_from(self._shm.buf, 0)[0]

    def published_time(self) -> float | None:
        """The media time of the newest published picture, or None when the writer did not name it."""
        if not self.attached:
            return None
        stamp = HEADER.unpack_from(self._shm.buf, 0)[7]
        return None if stamp < 0 else float(stamp)

    def publish(self, payload, width: int, height: int, fmt: int = FMT_RGB888,
                time_s: float | None = None) -> bool:
        """Copy one composited frame into the free slot. False when nobody asked or it does not fit.

        `time_s` is the media time of THIS picture: the seek path knows it exactly (it decoded that frame),
        a streaming frame often does not, and the header says which case it is instead of leaving the
        reader to guess.

        The payload is `width * height * 3` bytes of interleaved RGB888 (so it may be smaller than the
        slot: a box fit keeps the picture's aspect, and the header says what the payload really is).
        """
        if not self.attached:
            return False
        if not self.wanted():
            return False
        width, height = int(width), int(height)
        if fmt != FMT_RGB888 or width <= 0 or height <= 0:
            return False
        needed = width * height * RGB_BYTES_PER_PIXEL
        if needed > self.slot_size:
            if not self._size_warned:
                self._size_warned = True
                from .Util import log
                log(f"display: {width}x{height} RGB888 is {needed} bytes but the block's slot is "
                    f"{self.slot_size} - not publishing (the pane's box is smaller than its own frame)")
            return False
        view = memoryview(payload).cast("B")
        if len(view) != needed:
            return False
        buf = self._shm.buf
        seq = HEADER.unpack_from(buf, 0)[0]
        slot = seq & 1
        start = HEADER_SIZE + slot * self.slot_size
        buf[start:start + needed] = view
        # Read `wanted` again rather than reusing the check above: the pane may have gone away while the
        # payload was copied, and its byte must survive this header write untouched either way.
        wanted = buf[OFFSET_WANTED]
        stamp = TIME_UNSET if time_s is None else float(time_s)
        HEADER.pack_into(buf, 0, (seq + 1) & UINT32_MAX, width, height, FMT_RGB888,
                         wanted, 0, 0, stamp)
        return True

    def close(self) -> None:
        """Release the block. Unlink only when this process created it (the GUI owns its own names)."""
        segment, created = self._shm, self.created
        self._shm = None
        self.attached = False
        if segment is None:
            return
        try:
            segment.close()
        except Exception:
            pass
        if created:
            try:
                segment.unlink()
            except Exception:
                pass


class SeekState:
    """The reader of the GUI's newest-wins seek request (16 bytes, `SEEK_STRUCT` at offset 0).

    The GUI writes `(seconds, generation + 1)` into one slot; the backend renders that frame once the
    generation it sees differs from the last one it acted on. One slot and no queue on purpose: a newer
    request overwrites an older one, and a render loop that never looks pays for one 12-byte read at the
    start of an iteration.
    """

    def __init__(self, name: str | None):
        self.name = name
        self.created = False
        self._shm = None
        if name:
            try:
                self._shm, self.created = _open_block(name, SEEK_BLOCK_SIZE)
            except Exception:
                self._shm = None
        self.attached = self._shm is not None and len(self._shm.buf) >= SEEK_STRUCT.size
        #: The generation this process has already acted on (the value at attach is the baseline).
        self.last_generation = self.generation()

    def generation(self) -> int:
        if not self.attached:
            return 0
        return int(SEEK_STRUCT.unpack_from(self._shm.buf, 0)[1])

    def pending(self) -> float | None:
        """The requested time when a request newer than the last acted one is waiting, else None."""
        if not self.attached:
            return None
        seconds, generation = SEEK_STRUCT.unpack_from(self._shm.buf, 0)
        generation = int(generation)
        if generation == self.last_generation:
            return None
        self.last_generation = generation
        return float(seconds)

    def close(self) -> None:
        segment, created = self._shm, self.created
        self._shm = None
        self.attached = False
        if segment is None:
            return
        try:
            segment.close()
        except Exception:
            pass
        if created:
            try:
                segment.unlink()
            except Exception:
                pass


def write_seek(segment, seconds: float, generation: int) -> None:
    """Write one seek request into `segment` (a `SharedMemory` or a `SeekState`'s block).

    The GUI writes these same bytes itself; this helper exists so the writer and the tests share one
    definition of the layout instead of two `struct.pack_into` calls that can drift.
    """
    target = getattr(segment, "_shm", None)
    if target is None:
        target = segment
    SEEK_STRUCT.pack_into(target.buf, 0, float(seconds), int(generation) & UINT32_MAX)

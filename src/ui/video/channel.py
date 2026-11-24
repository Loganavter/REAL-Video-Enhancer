"""The pane's end of the composited-frame display channel, and the newest-wins seek request.

**This layout is a contract with ``backend/src/utils/DisplayChannel.py``.** Both sides copy it verbatim,
field for field and offset for offset, because a field at the wrong offset does not raise: it draws a
torn or colour-shifted picture, and the seek block hands the renderer the wrong second. A change here
without the writer is a torn picture, and there is no test that can see it from one side alone - which
is exactly why the reader's half lives in its own module, next to the frame type, instead of inside the
window module that also owns comboboxes and popups.

What moved here (verbatim, from ``src/ui/QTcustom.py``, 2026-11-24):

* the header struct and its offsets, the format and the unset-time sentinel;
* the pid-scoped block names (``/image_display<pid>``, ``/seek_state<pid>``) and the default box;
* the seek request struct and block size;
* :class:`DisplayReader`, the reader the pane polls on its 100 ms tick.

Nothing else. The window module keeps its popups, its combobox helpers and its clock.
"""

from __future__ import annotations

import os
import struct

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

#: The pane's block, as the reader sees it. The writer's copy is `backend/src/utils/DisplayChannel.py`; both sides
#: derive every offset from the struct, never from a literal, because a field at the wrong offset does not raise -
#: it draws a torn or colour-shifted picture. 23 bytes: the seven fields above plus the picture's media time.
#: `<IHHBBBI d`: seq, width, height, fmt, wanted, flags, reserved, then the picture's media time (double, -1 when the
#: writer cannot name it). The time is what makes "is this the frame the seek asked for" answerable - a sequence number
#: alone can only say "a new publication arrived".
DISPLAY_HEADER = struct.Struct("<IHHBBBI d")
DISPLAY_HEADER_SIZE = DISPLAY_HEADER.size
DISPLAY_OFFSET_SEQ = 0
DISPLAY_OFFSET_WIDTH = 4
DISPLAY_OFFSET_HEIGHT = 6
DISPLAY_OFFSET_FMT = 8
DISPLAY_OFFSET_TIME = 15
#: The `time` value the writer packs when it cannot name the picture.
DISPLAY_TIME_UNSET = -1.0
DISPLAY_OFFSET_WANTED = 9
#: The writer's byte at 10 (0 today) and its reserved word at 11; named so neither side spells the gap by hand.
DISPLAY_OFFSET_FLAGS = 10
DISPLAY_OFFSET_RESERVED = 11
DISPLAY_FMT_RGB888 = 1
DISPLAY_RGB_BYTES_PER_PIXEL = 3

#: The newest-wins seek request: `requested_seconds` (double) + `generation` (uint32) in a 16-byte block, the same
#: layout `DisplayChannel.SEEK_STRUCT` writes and the renderer's `SeekState` reads.
SEEK_REQUEST = struct.Struct("<dI")
SEEK_BLOCK_SIZE = 16

#: The block's size when the pane has not been laid out yet (or reports no size). The probe measured its channel at
#: this box; the live pane sizes its own block from its device pixels where that is available.
DEFAULT_DISPLAY_BOX = (960, 540)


def display_state_id(pid: int | None = None) -> str:
    """The frame block's name: one per GUI process, exactly like `PAUSED_STATE_SHARED_MEMORY_ID`."""
    return "/image_display" + str(os.getpid() if pid is None else int(pid))


def seek_state_id(pid: int | None = None) -> str:
    """The seek request block's name: same process-scoped convention."""
    return "/seek_state" + str(os.getpid() if pid is None else int(pid))


DISPLAY_STATE_SHARED_MEMORY_ID = display_state_id()
SEEK_STATE_SHARED_MEMORY_ID = seek_state_id()


class DisplayReader(QObject):
    """The pane's end of the display channel (`backend/src/utils/DisplayChannel.py` is the writer's).

    The GUI allocates the block before the backend starts (`ProcessTab._prepare_display_state`), because the pane is
    what knows its own box and the renderer must only ever attach. On the page's 100 ms tick `poll` reads the header,
    skips an unchanged `seq`, copies the newest complete slot and hands the picture on as a `QImage`.

    Two invariants live here and nowhere else:

    * **The newest complete slot is `(seq - 1) & 1`.** The writer fills slot `seq & 1` and only *then* writes
      `seq + 1`, so after a publish the payload is one slot behind the header's counter. (The probe reads only the
      header; this offset is the writer's own contract, pinned by `test_display_channel.py`.)
    * **One `seq` per tick.** A frame that arrived while the previous one was being copied is not chased in a loop:
      the next tick shows the newest one, which is the point of the two slots.
    """

    #: One composited frame, already a Qt image owned by the GUI (the shared buffer is copied, not aliased).
    display_frame = Signal(QImage)

    @property
    def media_time(self) -> float | None:
        """The media time of the picture this reader last handed over, seconds; None when the writer did not name it.

        A sequence number proves a NEW picture, not WHICH one: a seek log needs this number to say whether the pane is
        showing the frame that was asked for (owner, 2026-11-24).
        """
        return getattr(self, "_media_time", None)

    @property
    def seq(self) -> int:
        """The sequence number of the last picture this reader accepted (the publisher's own counter).

        The publisher writes `seq` BEFORE the payload is complete, so the picture a reader holds is the slot one behind
        it - the test for "did the pane get a NEW picture" is this counter moving, not the clock. Exposed because the
        pane's own identity for a composited frame is this number: the renderer's frame counter is a different fact and
        they do not always agree (owner, 2026-11-24: a seek log named the frame the render was on, not the one on
        screen).
        """
        return int(getattr(self, "_last_seq", 0) or 0)

    def __init__(self, name: str, box=DEFAULT_DISPLAY_BOX, seek_name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.name = str(name)
        self.seek_name = str(seek_name) if seek_name else f"{self.name}_seek"
        width, height = int(box[0]), int(box[1])
        self.box = (max(1, width), max(1, height))
        self._slot_size = 0
        self._last_seq = 0
        self._want = False
        self._shm = None
        self._seek_shm = None
        self._attached = False
        self._open()

    # -- attach / release ---------------------------------------------------------------------------

    def _open(self) -> None:
        from .. import shared_flag

        wanted_size = DISPLAY_HEADER_SIZE + 2 * (self.box[0] * self.box[1] * DISPLAY_RGB_BYTES_PER_PIXEL)
        segment = None
        created = False
        try:
            try:
                segment = shared_flag.attach(self.name)
            except FileNotFoundError:
                segment = None
            if segment is None or len(segment.buf) < wanted_size:
                if segment is not None:
                    # A previous process with this pid left a smaller block: the name is ours (it carries the pid), so
                    # replacing it cannot take a live pane's block away.
                    shared_flag.release(segment)
                segment = shared_flag.create(self.name, wanted_size)
                created = True
        except Exception:
            segment = None
        self._shm = segment
        self._attached = segment is not None and len(segment.buf) >= DISPLAY_HEADER_SIZE + 3
        if self._attached:
            # The segment is the authority on the slot size: a larger stale block still fits this pane's frames,
            # and its own size is what the writer validated against.
            self._slot_size = (len(segment.buf) - DISPLAY_HEADER_SIZE) // 2
            if created:
                # A freshly owned block starts at `seq` 0: an attached one (the pane reopening, a hand-written test
                # block) keeps whatever its writer already put there.
                # A zeroed block must say "no picture, no time": `0.0` would claim second zero, so the time
                # field starts at the unset sentinel and the pane's reader reports None.
                DISPLAY_HEADER.pack_into(segment.buf, 0, 0, 0, 0, 0, 0, 0, 0, DISPLAY_TIME_UNSET)
            self._write_wanted(0)
        try:
            self._seek_shm = shared_flag.create(self.seek_name, SEEK_BLOCK_SIZE)
        except Exception:
            self._seek_shm = None
        # `/dev/shm` must not keep these after a quit that skips every close handler (the names carry the pid, so
        # nothing else would ever find them) - the same lifetime rule as the paused flag.
        shared_flag.register(lambda: self._shm)
        shared_flag.register(lambda: self._seek_shm)

    @property
    def attached(self) -> bool:
        return bool(self._attached and self._shm is not None)

    @property
    def slot_size(self) -> int:
        return int(self._slot_size)

    @property
    def seq(self) -> int:
        if not self.attached:
            return 0
        return int(DISPLAY_HEADER.unpack_from(self._shm.buf, 0)[0])

    def release(self) -> None:
        """Close and unlink both blocks; safe twice and never raises."""
        from .. import shared_flag

        self._want = False
        for attribute in ("_seek_shm", "_shm"):
            segment = getattr(self, attribute, None)
            setattr(self, attribute, None)
            if segment is not None:
                shared_flag.release(segment)
        self._attached = False
        self._slot_size = 0
        self._last_seq = 0

    # -- asking for frames --------------------------------------------------------------------------

    def _write_wanted(self, value: int) -> None:
        if self._shm is None:
            return
        try:
            self._shm.buf[DISPLAY_OFFSET_WANTED] = int(value) & 0xFF
        except Exception:
            pass

    def set_wanted(self, wanted: bool) -> None:
        """Tell the renderer whether a picture is wanted right now (one byte; the renderer reads it per frame)."""
        self._want = bool(wanted)
        self._write_wanted(1 if self._want else 0)

    def wanted(self) -> bool:
        return bool(self._want)

    def block_wanted(self) -> int:
        """The byte the renderer actually reads (for tests and for "is the backend asking?")."""
        if not self.attached:
            return 0
        return int(self._shm.buf[DISPLAY_OFFSET_WANTED])

    # -- the tick -----------------------------------------------------------------------------------

    def poll(self) -> QImage | None:
        """Read at most one new frame and emit it; `None` when nothing new (or nothing wanted) is there."""
        if not self.attached or not self._want:
            return None
        buf = self._shm.buf
        seq, width, height, fmt, _wanted, _flags, _reserved, stamp = DISPLAY_HEADER.unpack_from(buf, 0)
        seq = int(seq)
        # The picture about to be handed over carries its own time; -1 means the writer could not name it.
        self._media_time = float(stamp) if stamp >= 0 else None
        if seq == self._last_seq:
            return None
        width, height = int(width), int(height)
        if fmt != DISPLAY_FMT_RGB888 or width <= 0 or height <= 0:
            return None
        needed = width * height * DISPLAY_RGB_BYTES_PER_PIXEL
        slot = (seq - 1) & 1
        start = DISPLAY_HEADER_SIZE + slot * self._slot_size
        if self._slot_size <= 0 or start + needed > len(buf):
            return None
        payload = bytes(buf[start:start + needed])
        # A frame published while the copy ran would make the picture a mix of two: drop this tick rather than draw it.
        if int(DISPLAY_HEADER.unpack_from(buf, 0)[0]) != seq:
            return None
        self._last_seq = seq
        # Re-assert the request: the writer preserves this byte, and this is where the pane proves it is still here.
        self._write_wanted(1)
        image = QImage(payload, width, height, width * DISPLAY_RGB_BYTES_PER_PIXEL,
                       QImage.Format.Format_RGB888).copy()
        self.display_frame.emit(image)
        return image

    # -- the newest-wins seek -----------------------------------------------------------------------

    def seek_generation(self) -> int:
        if self._seek_shm is None:
            return 0
        return int(SEEK_REQUEST.unpack_from(self._seek_shm.buf, 0)[1])

    def request_seek(self, seconds: float) -> int:
        """Write `(seconds, generation + 1)`; returns the generation written (0 when the block is missing)."""
        if self._seek_shm is None:
            return 0
        generation = (self.seek_generation() + 1) & 0xFFFFFFFF
        SEEK_REQUEST.pack_into(self._seek_shm.buf, 0, float(seconds), generation)
        return generation

    def seek_request(self) -> tuple[float, int]:
        """The bytes a renderer would read: `(seconds, generation)`."""
        if self._seek_shm is None:
            return 0.0, 0
        seconds, generation = SEEK_REQUEST.unpack_from(self._seek_shm.buf, 0)
        return float(seconds), int(generation)

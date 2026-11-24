"""The pane's end of the composited-frame display channel, the newest-wins seek request, and the render status.

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

The render status (``STATUS``, ``/status_state<pid>``) is a **second, pid-scoped segment**, not a block
appended to the frame header: the header's offsets are a pixel contract and its payload slots start at
``DISPLAY_HEADER_SIZE``, so growing it moves every picture. A status record is a handful of counters
written once per frame - it earns its own segment instead of that blast radius. The pane reads it with
:meth:`DisplayReader.status` (and :meth:`PreviewFeed.status`), and the record is a seqlock: the writer
turns ``generation`` odd before it writes the fields and even (last) after, so a reader that sees an odd
value or a generation that moved between its two reads answers `None` instead of a mixed record.
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

#: The render-status record (24 bytes, its own pid-scoped segment): the seqlock `generation`, the two frame
#: counters, the ETA and FPS the backend prints, and the phase/engine codes. Byte for byte the backend's `STATUS`.
STATUS = struct.Struct("<IIIffBB2x")
STATUS_SIZE = STATUS.size
#: The record's fields WITHOUT `generation`: what the writer packs between its odd and even marker.
STATUS_FIELDS = struct.Struct("<IIffBB2x")
STATUS_FIELDS_OFFSET = 4
STATUS_OFFSET_GENERATION = 0
STATUS_OFFSET_FRAMES_RENDERED = 4
STATUS_OFFSET_FRAMES_TOTAL = 8
STATUS_OFFSET_ETA = 12
STATUS_OFFSET_FPS = 16
STATUS_OFFSET_PHASE = 20
STATUS_OFFSET_ENGINE = 21
STATUS_OFFSET_RESERVED = 22

UINT32_MAX = 0xFFFFFFFF

#: `eta_seconds` meaning "the renderer cannot estimate it (yet)".
ETA_UNSET = -1.0

#: What the render is doing, as a byte; `PHASE_NAMES` is the word the pane shows for it.
PHASE_IDLE = 0
PHASE_PREPARING = 1
PHASE_BUILDING_ENGINE = 2
PHASE_RENDERING = 3
PHASE_FINALISING = 4
PHASE_DONE = 5
PHASE_FAILED = 6
PHASE_PAUSED = 7

PHASE_NAMES = {
    PHASE_IDLE: "Idle",
    PHASE_PREPARING: "Preparing",
    PHASE_BUILDING_ENGINE: "Building engine…",
    PHASE_RENDERING: "Rendering",
    PHASE_FINALISING: "Finalising",
    PHASE_DONE: "Idle",
    PHASE_FAILED: "Failed",
    PHASE_PAUSED: "Paused",
}

#: Which upscaler produced the picture, as a byte; `ENGINE_NAMES` is the backend's own `--backend` token.
ENGINE_UNKNOWN = 0
ENGINE_PYTORCH = 1
ENGINE_TENSORRT = 2
ENGINE_NCNN = 3
ENGINE_ONNX = 4
ENGINE_DIRECTML = 5

ENGINE_NAMES = {
    ENGINE_UNKNOWN: "unknown",
    ENGINE_PYTORCH: "pytorch",
    ENGINE_TENSORRT: "tensorrt",
    ENGINE_NCNN: "ncnn",
    ENGINE_ONNX: "onnx",
    ENGINE_DIRECTML: "directml",
}

#: The block's size when the pane has not been laid out yet (or reports no size). The probe measured its channel at
#: this box; the live pane sizes its own block from its device pixels where that is available.
DEFAULT_DISPLAY_BOX = (960, 540)


def display_state_id(pid: int | None = None) -> str:
    """The frame block's name: one per GUI process, exactly like `PAUSED_STATE_SHARED_MEMORY_ID`."""
    return "/image_display" + str(os.getpid() if pid is None else int(pid))


def seek_state_id(pid: int | None = None) -> str:
    """The seek request block's name: same process-scoped convention."""
    return "/seek_state" + str(os.getpid() if pid is None else int(pid))


def status_state_id(pid: int | None = None) -> str:
    """The render-status block's name: same process-scoped convention, one record per GUI process."""
    return "/status_state" + str(os.getpid() if pid is None else int(pid))


def format_eta(seconds: float | None) -> str:
    """An ETA in seconds as the string the backend prints (`h:mm:ss`, minutes and seconds zero-padded).

    The same arithmetic as `backend/src/FFmpeg.py: convertTime`, so the pane's `ETA:` reads the value
    the log line reads, not a rounded-off twin of it.
    """
    total = int(max(0.0, float(seconds if seconds is not None else 0.0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


DISPLAY_STATE_SHARED_MEMORY_ID = display_state_id()
SEEK_STATE_SHARED_MEMORY_ID = seek_state_id()
STATUS_STATE_SHARED_MEMORY_ID = status_state_id()


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

    def __init__(self, name: str, box=DEFAULT_DISPLAY_BOX, seek_name: str | None = None,
                 status_name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.name = str(name)
        self.seek_name = str(seek_name) if seek_name else f"{self.name}_seek"
        self.status_name = str(status_name) if status_name else status_state_id()
        width, height = int(box[0]), int(box[1])
        self.box = (max(1, width), max(1, height))
        self._slot_size = 0
        self._last_seq = 0
        self._want = False
        self._shm = None
        self._seek_shm = None
        self._status_shm = None
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
        # The render-status record is its own pid-scoped block (see the module docstring): the pane owns it and
        # the renderer only ever attaches, like the frame and seek blocks.
        try:
            segment = shared_flag.create(self.status_name, STATUS_SIZE)
            if segment is not None and len(segment.buf) >= STATUS_SIZE:
                # A fresh block must say "no render yet": a zeroed ETA would claim zero seconds left. The
                # display header's time field starts at its own unset sentinel for the same reason.
                STATUS.pack_into(segment.buf, 0, 0, 0, 0, ETA_UNSET, 0.0,
                                 PHASE_IDLE, ENGINE_UNKNOWN)
                self._status_shm = segment
            else:
                if segment is not None:
                    shared_flag.release(segment)   # a stale block of the wrong size is not ours to keep
                self._status_shm = None
        except Exception:
            self._status_shm = None
        # `/dev/shm` must not keep these after a quit that skips every close handler (the names carry the pid, so
        # nothing else would ever find them) - the same lifetime rule as the paused flag.
        shared_flag.register(lambda: self._shm)
        shared_flag.register(lambda: self._seek_shm)
        shared_flag.register(lambda: self._status_shm)

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
        for attribute in ("_status_shm", "_seek_shm", "_shm"):
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

    # -- the render status --------------------------------------------------------------------------

    def status(self) -> dict | None:
        """The render's own numbers, or `None` when there is no complete record to read.

        The record is a seqlock (`STATUS`): the writer turns `generation` odd while it fills the fields and
        writes the next even value LAST. So a read is accepted only when the generation was even before the
        fields and still the same after them; an odd value, or one that moved, means the writer was inside
        the record and the answer is `None` - never a mixed record. Retried a few times so a tick that lands
        during a write usually still reports the previous one instead of losing a beat.

        Keys: `generation`, `frames_rendered`, `frames_total`, `eta_seconds` (None when unset), `fps`,
        `phase`, `phase_name`, `engine`, `engine_name`.
        """
        block = getattr(self, "_status_shm", None)
        if not self.attached or block is None:
            return None
        buf = block.buf
        for _ in range(4):
            before = int(struct.unpack_from("<I", buf, 0)[0])
            if before & 1:                     # the writer is inside the record
                continue
            frames, total, eta, fps, phase, engine = STATUS_FIELDS.unpack_from(
                buf, STATUS_FIELDS_OFFSET)
            # A SECOND read of the generation, after the fields: a writer that started while they were
            # copied has already turned it odd (or moved it by two), and the record is refused.
            after = int(struct.unpack_from("<I", buf, 0)[0])
            if after != before:
                continue
            phase = int(phase)
            engine = int(engine)
            return {
                "generation": after,
                "frames_rendered": int(frames),
                "frames_total": int(total),
                "eta_seconds": None if float(eta) < 0 else float(eta),
                "fps": float(fps),
                "phase": phase,
                "phase_name": PHASE_NAMES.get(phase, "Idle"),
                "engine": engine,
                "engine_name": ENGINE_NAMES.get(engine, "unknown"),
            }
        return None

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

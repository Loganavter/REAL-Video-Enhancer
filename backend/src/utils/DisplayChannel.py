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
    10      1     flags    uint8    bit 0 = this picture ANSWERS a seek request (see `DISPLAY_FLAG_SEEK_ANSWER`)
    11      4     reserved uint32   the seek generation this picture answers, 0 when it answers none
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

The render status (``STATUS``, `StatusChannel`) lives in a **second, pid-scoped segment**
(``/status_state<pid>``) instead of a block appended to this header. Why: the header above is a pixel
contract whose offsets are pinned byte for byte by `harness/tests/test_display_channel.py` and whose
payload slots start at ``HEADER_SIZE`` - growing it moves every payload, and one wrong offset draws a
torn picture instead of raising. A status record is a handful of counters written once per frame; it
does not deserve that blast radius. One extra attach per process buys an untouched frame contract and
a block that can be read (and torn-read-checked) on its own.

The status record is a seqlock: ``generation`` is bumped to an ODD value before the fields are written
and to the next EVEN value after they are in place, and the even value is written LAST. A reader takes
``generation`` before and after the fields and only accepts an even generation that did not move; a
writer that starts or finishes between those two reads makes the value move (or turns it odd), so a
mixed record is refused rather than reported.
"""

from __future__ import annotations

import os
import math
import struct
import threading

#: `flags` bit 0: the picture in the newest slot is the answer to a seek request, and `reserved` holds that
#: request's generation. The pane needs the pair because a sequence number proves a NEW picture, not the ONE it asked
#: for: the streaming path names every frame it draws, so "the first picture with a media time" stopped meaning "the
#: seek's answer" the day streaming frames got names (measured 2026-11-24: a press at 1.888 s closed on the streaming
#: frame at 4.046 s, +2.158 s off, because that frame carried a time too).
DISPLAY_FLAG_SEEK_ANSWER = 0x01

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

#: The pane's announced box: `gen u32, req_w u16, req_h u16, real_w u16, real_h u16, state u8, flags u8, pad 2`.
#: The pane WRITES a request (what resolution it wants the writer to fit into) and this side ANSWERS with what it
#: really fits - a slot can be smaller than the request. Request and fact are separate states on purpose: a single
#: "make it bigger" flag cannot say "requested but not ready", and that is the state a stale picture hides in.
BOX_STRUCT = struct.Struct("<IHHHHBBxx")
BOX_IDLE, BOX_REQUESTED, BOX_READY, BOX_FAILED = 0, 1, 2, 3
SEEK_BLOCK_SIZE = 16
UINT32_MAX = 0xFFFFFFFF

#: The render-status record (24 bytes, its own pid-scoped segment - see the module docstring): the seqlock
#: `generation`, the render loop's two frame counters, the ETA the backend prints, the FPS it prints, and the
#: phase/engine codes a reader turns into the words the pane shows. `<` keeps standard sizes and no padding;
#: the two trailing bytes are reserved so a later field does not move this record's size.
STATUS = struct.Struct("<IIIffBB2x")
STATUS_SIZE = STATUS.size
#: The record's fields WITHOUT `generation`: what a writer packs between its odd and even marker.
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

#: `eta_seconds` meaning "the renderer cannot estimate it (yet)". A remaining time is never negative.
ETA_UNSET = -1.0

#: What the render is doing, as a byte. The pane maps a code to the word it shows (`PHASE_NAMES`).
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

#: Which upscaler produced the picture, as a byte. The name is the backend's own `--backend` token.
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

_ENGINE_CODES = {
    "pytorch": ENGINE_PYTORCH,
    "tensorrt": ENGINE_TENSORRT,
    "ncnn": ENGINE_NCNN,
    "onnx": ENGINE_ONNX,
    "directml": ENGINE_DIRECTML,
}


def engine_code(backend: object) -> int:
    """The `engine` byte for a backend name (`--backend`/`self.backend`); unknown names stay `ENGINE_UNKNOWN`."""
    return _ENGINE_CODES.get(str(backend or "").strip().lower(), ENGINE_UNKNOWN)


def status_state_id(pid: int | None = None) -> str:
    """The status block's name: one per GUI process, the same pid-scoped convention as the paused flag."""
    return "/status_state" + str(os.getpid() if pid is None else int(pid))


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

    def fit_announced_box(self, width: int, height: int) -> tuple[int, int]:
        """The largest box with the REQUESTED aspect that this block's slot can hold.

        A pane can ask for more than the block was allocated for (it grew past the maximum the GUI chose). The honest
        answer is the size actually fitted - libplacebo's `pl_swapchain_resize` returns "what width/height the swapchain
        was actually set to", xpra clamps with `min(client, root)` and replies with the server's size. Silently writing a
        smaller payload while the pane believes otherwise is the failure mode all of them avoid.
        """
        want_w = max(2, int(width))
        want_h = max(2, int(height))
        area = (self.slot_size // 3) if self.slot_size else 0
        if area <= 0 or want_w * want_h <= area:
            return want_w - (want_w & 1), want_h - (want_h & 1)
        scale = math.sqrt(area / float(want_w * want_h))
        real_w = max(2, int(want_w * scale)) & ~1
        real_h = max(2, int(want_h * scale)) & ~1
        return real_w, real_h

    def published_time(self) -> float | None:
        """The media time of the newest published picture, or None when the writer did not name it."""
        if not self.attached:
            return None
        stamp = HEADER.unpack_from(self._shm.buf, 0)[7]
        return None if stamp < 0 else float(stamp)

    def publish(self, payload, width: int, height: int, fmt: int = FMT_RGB888,
                time_s: float | None = None, seek_generation: int | None = None) -> bool:
        """Copy one composited frame into the free slot. False when nobody asked or it does not fit.

        `time_s` is the media time of THIS picture: the seek path knows it exactly (it decoded that frame), a
        streaming frame derives it from the export's counter (or cannot name it at all), and the header says which
        case it is instead of leaving the reader to guess.

        `seek_generation` is set by the ONE publish that answers a seek request - the renderer passes the generation
        it acted on. The pane closes a seek only on a picture that carries the generation it asked for; without it the
        only available rule was "the first named picture", which the streaming path now satisfies first.

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
        answers = int(seek_generation or 0) & UINT32_MAX
        HEADER.pack_into(buf, 0, (seq + 1) & UINT32_MAX, width, height, FMT_RGB888,
                         wanted, DISPLAY_FLAG_SEEK_ANSWER if answers else 0, answers, stamp)
        return True

    def published_seek_generation(self) -> int:
        """The seek generation the newest picture answers, or 0 when it answers none."""
        if not self.attached:
            return 0
        flags = HEADER.unpack_from(self._shm.buf, 0)[5]
        if not int(flags) & DISPLAY_FLAG_SEEK_ANSWER:
            return 0
        return int(HEADER.unpack_from(self._shm.buf, 0)[6])

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


class DisplayBoxState:
    """The writer's side of the announced box: act on a generation change, answer with what it really fits.

    Read once per loop iteration - one 16-byte unpack at the frame boundary, where the professional sources put a
    geometry change (mpv refuses to reconfigure while a frame is displaying; libplacebo retires the old target only
    after its fences signal). Here nothing is retired, because the payload always fits a PREALLOCATED slot and each
    publication carries its own `width`/`height` in the frame header: a frame is either the old box or the new one,
    never torn.
    """

    def __init__(self, name: str | None):
        self.name = name
        self.created = False
        self._shm = None
        if name:
            try:
                self._shm, self.created = _open_block(name, BOX_STRUCT.size)
            except Exception:
                self._shm = None
        self.attached = self._shm is not None and len(self._shm.buf) >= BOX_STRUCT.size
        #: The generation acted on. None means "the first record is still a request" (the pane announces before the
        #: renderer starts, so the first iteration must apply it).
        self.last_generation = None

    def pending(self):
        """`(generation, want_w, want_h)` when the pane announced something this process has not acted on."""
        if not self.attached:
            return None
        generation, want_w, want_h, _real_w, _real_h, _state, _flags = BOX_STRUCT.unpack_from(self._shm.buf, 0)
        # The writer's own torn-read guard: a record written while this unpack ran is refused, not half-applied.
        if int(BOX_STRUCT.unpack_from(self._shm.buf, 0)[0]) != int(generation):
            return None
        generation = int(generation)
        if generation == 0 or generation == self.last_generation:
            return None
        self.last_generation = generation
        return generation, int(want_w), int(want_h)

    def answer(self, generation: int, want_w: int, want_h: int, real_w: int, real_h: int) -> None:
        """Publish what was really fitted; `state` says whether the request was honoured exactly."""
        if not self.attached:
            return
        honoured = (int(real_w), int(real_h)) == (int(want_w), int(want_h))
        BOX_STRUCT.pack_into(self._shm.buf, 0, int(generation), int(want_w), int(want_h),
                             int(real_w), int(real_h), BOX_READY if honoured else BOX_FAILED, 0)

    def close(self) -> None:
        segment, created, self._shm = self._shm, self.created, None
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


class StatusChannel:
    """The writer's end of the render-status block (its own pid-scoped segment, `STATUS`).

    Frame data is the display channel's job; this block carries the numbers the pane used to scrape out
    of the backend's stdout: how far the render is, how much is left, how fast it is going and which
    phase it is in. It is written once per rendered frame and on every phase change, never instead of
    the log line - the log stays the record a person reads.

    `set()` is a partial update: the fields passed replace the stored ones and the rest are rewritten
    unchanged, so a frame tick never has to know the ETA and an ETA tick never has to know the frame
    count. The two writer threads (the render loop and the information printer) share one lock; the
    seqlock marker makes the record safe to read from the other process without one.
    """

    def __init__(self, name: str | None, engine: int = ENGINE_UNKNOWN):
        self.name = name
        self.created = False
        self._shm = None
        self._lock = threading.Lock()
        self._state = {
            "generation": 0,
            "frames_rendered": 0,
            "frames_total": 0,
            "eta_seconds": ETA_UNSET,
            "fps": 0.0,
            "phase": PHASE_IDLE,
            "engine": int(engine) & 0xFF,
        }
        if name:
            try:
                self._shm, self.created = _open_block(name, STATUS_SIZE)
            except Exception:
                # A status block that cannot be attached must never take a render down.
                self._shm = None
        self.attached = self._shm is not None and len(self._shm.buf) >= STATUS_SIZE
        if self.attached:
            # Continue the sequence of a block a previous writer left: the reader rejects a generation that
            # moves, so starting over at 0 under a block already at 40 would look like a torn write forever.
            self._state["generation"] = int(struct.unpack_from("<I", self._shm.buf, 0)[0]) & ~1

    def set(self, **changes) -> int:
        """Merge `changes` into the record, write it, and return the new (even) generation; 0 when detached."""
        if not self.attached or self._shm is None:
            return 0
        with self._lock:
            state = dict(self._state)
            state.update(changes)
            generation = (int(state["generation"]) + 2) & UINT32_MAX
            buf = self._shm.buf
            # Odd: a write is in progress. A reader that lands anywhere inside the record sees this and refuses.
            struct.pack_into("<I", buf, 0, (generation - 1) & UINT32_MAX)
            STATUS_FIELDS.pack_into(
                buf,
                STATUS_FIELDS_OFFSET,
                int(state["frames_rendered"]) & UINT32_MAX,
                int(state["frames_total"]) & UINT32_MAX,
                float(state["eta_seconds"]),
                float(state["fps"]),
                int(state["phase"]) & 0xFF,
                int(state["engine"]) & 0xFF,
            )
            # Even, written LAST: everything before this byte is already in place when a reader accepts it.
            struct.pack_into("<I", buf, 0, generation)
            state["generation"] = generation
            self._state = state
            return generation

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


def write_seek(segment, seconds: float, generation: int) -> None:
    """Write one seek request into `segment` (a `SharedMemory` or a `SeekState`'s block).

    The GUI writes these same bytes itself; this helper exists so the writer and the tests share one
    definition of the layout instead of two `struct.pack_into` calls that can drift.
    """
    target = getattr(segment, "_shm", None)
    if target is None:
        target = segment
    SEEK_STRUCT.pack_into(target.buf, 0, float(seconds), int(generation) & UINT32_MAX)

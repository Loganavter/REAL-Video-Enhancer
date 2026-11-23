"""The file half of the preview player: PyAV on the CPU, driven one tick at a time.

Route (plan-rve-video-preview-widget-2026-11-23, "Decode"): PyAV, hardware decode never requested, no
`QMediaPlayer`. `QMediaPlayer` hands frames over as `QVideoFrame` and owns a native surface; we need
the PTS of a *source* frame for "hold LMB to see the original" and a seek we control, and neither is
ours with Qt Multimedia. PyAV costs one wheel and gives a PTS per frame, a keyframe seek and a
decoder we can pace.

Shape of the object: it is a `QObject` with signals, never a widget, and it owns no thread. The
caller moves it to a `QThread` (started with its default `run()`, so the thread has an event loop) and
starts that thread; every tick then runs there, and frames reach the surface as `frame_ready`
signal deliveries. The class itself never blocks, never sleeps and never touches a widget.

Pacing: no audio, so the PTS clock owns time (`src/ui/video/clock.py`, reused rather than duplicated).
A tick decodes at most ``max_frames_per_tick`` frames: frames already due are published at once, the
first frame still in the future is held and the next tick is scheduled for the moment it falls due.
A decoder that cannot keep up therefore falls behind the picture, not the GUI - the consumer's
`FrameQueue` drops the late ones.

Seek policy: `container.seek(..., backward=True)` to the previous keyframe in microseconds, decode
forward, and publish the last frame at or before the requested PTS. While a seek is in flight the
surface keeps its previous picture; this class publishes only when it has the frame to publish.

``import av`` happens inside `open()`, not at module scope: an interpreter without PyAV can still
import this module (the surface may be showing the live stream, which needs no decoder at all) and
gets a clear `error` signal if it asks for a file.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QMetaObject, QObject, Qt, QTimer, Signal, Slot

from .clock import PlaybackClock
from .frames import FORMATS, FrameRef, frame_pts, pack_frame

#: Microsecond seek unit: `av.time_base` is 1/1000000 and `container.seek()` without `stream=` reads
#: its offset in that unit.
_MICROS = 1_000_000

#: Text used by the surface to tell "no decoder here" from "this file is broken".
NO_PYAV = "PyAV is not installed: the file player needs `pip install av`"


class VideoDecoder(QObject):
    """One video file, opened once and played/seeked from its own thread."""

    #: One `FrameRef`, published in presentation order.
    frame_ready = Signal(object)
    #: Emitted once per successful `open`: duration_s, fps, frames_total (0 where unknown).
    opened = Signal(float, float, float)
    #: Emitted when a seek finished and the frame at the target was published.
    seeked = Signal(float)
    #: A human-readable failure; a decode error stops playback, it is never swallowed.
    error = Signal(str)

    def __init__(self, parent: QObject | None = None, *, pixel_format: str = "rgb24",
                 max_frames_per_tick: int = 4, codec_threads: int = 2) -> None:
        super().__init__(parent)
        if pixel_format not in FORMATS:
            raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
        self._pixel_format = pixel_format
        self._max_frames_per_tick = max(1, int(max_frames_per_tick))
        self._codec_threads = max(1, int(codec_threads))

        self._container = None
        self._stream = None
        self._demux = None
        self._frame_iter = None
        self._duration = 0.0
        self._fps = 0.0
        self._interval = 0.04      # 25 fps until the stream says otherwise
        self._frames_total = 0.0
        self._decode_pts = 0.0     # last decoded frame's PTS: the fallback for one that has none
        self._packed = 0           # frames packed so far: the index of last resort
        self._at_eof = False

        self._playing = False
        self._held: FrameRef | None = None
        self._pending_seek: float | None = None
        #: A one-off decode request ("the original at this PTS"), independent of the play position.
        self._request: float | None = None
        self._pump_pending = False

        self._clock = PlaybackClock()

    # -- lifecycle ---------------------------------------------------------------------------------

    def open(self, path: str) -> bool:
        """Open the container and report the stream's shape through `opened`."""
        try:
            import av
        except ImportError:
            self.error.emit(NO_PYAV)
            return False

        self.close()
        try:
            container = av.open(path, mode="r")
        except Exception as exc:  # a missing file, a container without a header, a broken codec
            self.error.emit(f"cannot open {path!r}: {exc}")
            return False

        if not container.streams.video:
            container.close()
            self.error.emit(f"{path!r} has no video stream")
            return False

        stream = container.streams.video[0]
        # The plan's budget: decode may not compete with inference, so the codec thread pool is
        # capped. `thread_type` stays libavcodec's default; only the count is ours.
        stream.codec_context.thread_count = self._codec_threads

        duration = 0.0
        if container.duration:
            duration = float(container.duration) / _MICROS
        elif stream.duration:
            duration = float(stream.duration * stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames_total = float(stream.frames or (round(duration * fps) if fps else 0))

        self._container = container
        self._stream = stream
        self._duration = duration
        self._fps = fps
        self._interval = 1.0 / fps if fps > 0 else 0.04
        self._frames_total = frames_total
        self._decode_pts = 0.0
        self._packed = 0
        self._at_eof = False
        self._reset_stream()

        now = time.monotonic()
        self._clock.start(now, 0.0)
        self._clock.pause(now)
        self.opened.emit(duration, fps, frames_total)
        return True

    def close(self) -> None:
        """Stop, release the container and forget every pending action."""
        self._playing = False
        self._pump_pending = False
        self._held = None
        self._pending_seek = None
        self._request = None
        self._demux = None
        self._frame_iter = None
        self._stream = None
        container, self._container = self._container, None
        if container is not None:
            try:
                container.close()
            except Exception as exc:  # report it: a container that will not close is a bug
                self.error.emit(f"closing the container failed: {exc}")

    # -- transport ---------------------------------------------------------------------------------

    def play(self) -> None:
        """Start (or resume) playback; after the end of the file, restart from the beginning."""
        if self._container is None:
            self.error.emit("play() called before open()")
            return
        if self._playing:
            return
        if self._at_eof:
            # Play after the end starts the file again; the surface keeps its last picture until the
            # first frame of the new pass arrives.
            self._pending_seek = 0.0
            self._at_eof = False
        self._playing = True
        self._clock.resume(time.monotonic())
        self._kick()

    def pause(self) -> None:
        """Stop presenting; the position is frozen and the next `play` resumes from it."""
        if not self._playing:
            return
        self._clock.pause(time.monotonic())
        self._playing = False

    def seek(self, seconds: float) -> None:
        """Jump to `seconds`: keyframe rewind, decode forward, publish, then continue if playing."""
        if self._container is None:
            self.error.emit("seek() called before open()")
            return
        target = self._clamp(seconds)
        self._held = None
        self._at_eof = False
        self._pending_seek = target
        now = time.monotonic()
        self._clock.start(now, target)
        if not self._playing:
            self._clock.pause(now)
        self._kick()

    def request_frame(self, seconds: float) -> None:
        """Decode and publish one frame at `seconds` without moving the play position.

        This is the LMB "show the original" path. A second decoder instance on the input file is the
        cheaper answer when the request repeats; a request against *this* decoder costs two seeks
        (out and back) because one container has one read position.
        """
        if self._container is None:
            self.error.emit("request_frame() called before open()")
            return
        self._request = self._clamp(seconds)
        self._kick()

    def is_playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def fps(self) -> float:
        return self._fps

    # -- the pump ----------------------------------------------------------------------------------

    @Slot()
    def _pump(self) -> None:
        """One tick, in the decoder's own thread: honour a request/seek, then advance playback."""
        self._pump_pending = False
        if self._container is None:
            return

        # A request and a seek are both "decode to this PTS now"; each may chain the other (a request
        # while playing has to put the play position back afterwards).
        while True:
            if self._request is not None:
                target, self._request = self._request, None
                resume_at = self._clock.position(time.monotonic())
                self._seek_and_publish(target, announce=False)
                if self._playing:
                    self._pending_seek = resume_at
                continue
            if self._pending_seek is not None:
                target, self._pending_seek = self._pending_seek, None
                self._seek_and_publish(target)
                continue
            break

        if not self._playing:
            return
        try:
            delay_ms = self._advance()
        except Exception as exc:  # a decode error stops the picture loudly; it is not retried
            self._playing = False
            self._clock.pause(time.monotonic())
            self.error.emit(f"decode failed: {exc}")
            return
        if delay_ms is not None and self._playing:
            self._schedule(delay_ms)

    def _advance(self) -> int | None:
        """Publish the frames that are due; return ms until the next tick, or None at end of file."""
        now = time.monotonic()
        position = self._clock.position(now)
        for _ in range(self._max_frames_per_tick):
            frame = self._take()
            if frame is None:
                self._end_of_file(now)
                return None
            if frame.pts <= position:
                self.frame_ready.emit(frame)
                continue
            # Not due yet: hold it and come back when its PTS arrives. Holding one frame keeps the
            # decode one frame ahead of the picture without a busy loop.
            self._held = frame
            return self._delay_until(frame.pts, position)
        return 0

    def _take(self) -> FrameRef | None:
        if self._held is not None:
            frame, self._held = self._held, None
            return frame
        if self._frame_iter is None:
            return None
        try:
            decoded = next(self._frame_iter)
        except StopIteration:
            return None
        return self._pack(decoded)

    def _pack(self, decoded) -> FrameRef:
        pts = frame_pts(decoded)
        if getattr(decoded, "time", None) is None:
            pts = self._decode_pts + self._interval
        self._decode_pts = pts
        self._packed += 1
        index = int(round(pts / self._interval)) if self._fps > 0 else self._packed
        return pack_frame(decoded, index, self._pixel_format, pts=pts)

    def _delay_until(self, pts: float, position: float) -> int:
        delay_ms = (pts - position) * 1000.0
        # Never sleep longer than a second: a stream with a gap (or a clock that jumped) re-evaluates
        # instead of parking the thread on a stale estimate.
        return int(min(max(delay_ms, 0.0), 1000.0))

    def _end_of_file(self, now: float) -> None:
        self._playing = False
        self._at_eof = True
        self._clock.pause(now)
        self._held = None

    # -- seeking -----------------------------------------------------------------------------------

    def _seek_and_publish(self, target: float, *, announce: bool = True) -> None:
        """Keyframe rewind, decode forward, publish the last frame at or before `target`.

        ``announce=False`` is the one-off request path: the frame is published but the playhead did
        not move, so the surface must not be told that it did.
        """
        self._rewind(target)
        published: FrameRef | None = None
        while True:
            frame = self._take()
            if frame is None:
                break
            if frame.pts > target:
                # First frame after the target: it belongs to the next playback tick, not to the
                # seek result, so it is held rather than published.
                self._held = frame
                break
            published = frame
        if published is None:
            published, self._held = self._held, None
        if published is not None:
            self.frame_ready.emit(published)
        if announce:
            self.seeked.emit(float(target))

    def _rewind(self, target: float) -> None:
        # A frame held from the old position belongs to that position, not to this one.
        self._held = None
        self._container.seek(int(max(0.0, target) * _MICROS), backward=True)
        self._reset_stream()

    def _reset_stream(self) -> None:
        """Point the demuxer at the current container position and flush the decoder state."""
        self._stream.codec_context.flush_buffers()
        self._demux = self._container.demux(self._stream)
        self._frame_iter = self._frame_stream()

    def _frame_stream(self):
        """Decoded frames from the current demux position, one packet at a time.

        Kept as one generator across ticks on purpose: abandoning the inner `packet.decode()` loop
        between ticks would throw away the frames of a packet that is already in the decoder.
        """
        while True:
            try:
                packet = next(self._demux)
            except StopIteration:
                return
            for frame in packet.decode():
                yield frame

    # -- scheduling --------------------------------------------------------------------------------

    def _clamp(self, seconds: float) -> float:
        target = max(0.0, float(seconds))
        if self._duration > 0:
            target = min(target, self._duration)
        return target

    def _kick(self) -> None:
        """Ask for a tick in the decoder's own thread; the caller may be the GUI thread."""
        if self._pump_pending:
            return
        self._pump_pending = True
        QMetaObject.invokeMethod(self, "_pump", Qt.ConnectionType.QueuedConnection)

    def _schedule(self, delay_ms: int) -> None:
        """Re-arm the pump from inside the decoder's thread, where a timer belongs to that thread."""
        if self._pump_pending:
            return
        self._pump_pending = True
        QTimer.singleShot(int(delay_ms), self, self._pump)

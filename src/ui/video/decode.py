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
from .frames import FORMATS, FrameRef, frame_pts, pack_frame, stream_color_tags

#: Microsecond seek unit: `av.time_base` is 1/1000000 and `container.seek()` without `stream=` reads
#: its offset in that unit.
_MICROS = 1_000_000

#: Text used by the surface to tell "no decoder here" from "this file is broken".
NO_PYAV = "PyAV is not installed: the file player needs `pip install av`"


#: Containers store PTS rounded to whole milliseconds; the names this decoder is asked to find are exact (`n/fps`, or
#: a request's seconds). Half a millisecond of slack is the exact inverse of that rounding: without it the frame whose
#: stored PTS rounded UP reads as "after the target" and the PREVIOUS frame is published. Measured on the owner's clip
#: (2026-11-25): the name 3.21154s is stored as 3.212s, so `pts <= name` published 3.170s - one frame early, on half of
#: all presses, while the render's own path (`-ss`, ceil) returned 3.212s. That one frame is the compare's desync.
PTS_QUANTISATION_S = 0.0005


def frame_belongs_to(frame_pts: float, target: float) -> bool:
    """Whether the frame at `frame_pts` is the last one at or before the target (playback: the frame it lands in)."""
    try:
        return float(frame_pts) <= float(target) + PTS_QUANTISATION_S
    except (TypeError, ValueError):
        return False


def frame_at_or_after(frame_pts: float, target: float, *, tolerance: float = 0.0) -> bool:
    """Whether the frame at `frame_pts` is the FIRST one at or after the target - the frame `-ss target` returns.

    This is the rule the render's own seek path uses (`SeekReader._index_at` is a `ceil`), so a ONE-OFF request (the
    LMB compare) must use it too: a `floor` here and a `ceil` there put the source one frame behind the result on every
    press whose target sits between two frames. Measured 2026-11-25: ask 8.105s, frames at 8.091/8.133, the result was
    the 8.133 frame and the compare showed 8.091.

    `tolerance` is for the caller whose target is a NAME rather than an exact boundary: the pane names a picture with
    the backend's grid time, and that grid is fitted from millisecond-quantised probe samples, so the name sits a few
    milliseconds off the frame it belongs to (measured: name 8.136s, the frame's own PTS 8.133s, and a plain `ceil`
    then picked the NEXT frame at 8.175). The caller passes half a frame interval - the exact inverse of "a name is a
    frame's time" - and the frame the picture came from is chosen again.
    """
    try:
        return float(frame_pts) >= float(target) - PTS_QUANTISATION_S - float(tolerance)
    except (TypeError, ValueError):
        return False


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
                 max_frames_per_tick: int = 4, codec_threads: int = 2,
                 max_size: tuple[int, int] | None = None, content_time: bool = False,
                 color_space: str | None = None, color_range: str | None = None) -> None:
        super().__init__(parent)
        if pixel_format not in FORMATS:
            raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
        self._pixel_format = pixel_format
        #: The matrix/range the shader converts the planes with. `None` means "read the stream's own
        #: tags at open, and let the renderer fall back to bt709/tv when the file has none" - which
        #: is what this app's writer declares. A caller that produced the file itself (the playback
        #: host knows the job's settings) may pass them and they win.
        self._color_space = color_space
        self._color_range = color_range
        #: True when the CALLER stated them: an auto-read pair must be re-read on the next file.
        self._color_explicit = color_space is not None or color_range is not None
        self._max_frames_per_tick = max(1, int(max_frames_per_tick))
        self._codec_threads = max(1, int(codec_threads))
        #: The largest picture this decoder will hand over, or ``None`` for the file's own resolution.
        #:
        #: Playback cost is decode + conversion + upload per frame, and all three grow with the pixel count. Packed
        #: RGB24 is ``width * height * 3`` bytes: 24.9 MB for a 3840x2160 frame, 6.2 MB at 1080p, 1.56 MB at 960x540.
        #: The queue pane shows one held picture in a box that is a few hundred pixels wide, so the extra pixels are
        #: paid for by the decoder, copied and uploaded only to be thrown away by the surface. `_fit` asks PyAV for the
        #: smaller picture instead (swscale, in the decode already happening) - a 4K item on a 960x540 pane packs 16x
        #: less per frame. The file's timestamps and the seek path are untouched.
        self._max_size = None if max_size is None else (max(2, int(max_size[0])), max(2, int(max_size[1])))

        #: Name frames in CONTENT time: the file's PTS minus the container's own start. A rendered output is written
        #: with `-avoid_negative_ts make_zero`, which shifts the whole timeline by the encoder's B-frame delay
        #: (measured 2026-11-25: 0.083 s, two frames at 23.976) while the first frame is still the source's first.
        #: A SOURCE file must keep its own absolute times - the renderer seeks by them - so this is opt-in and only the
        #: playback of an output turns it on.
        self._content_time = bool(content_time)
        self._media_origin: float | None = None
        self._origin = 0.0

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

        try:
            start = stream.start_time
            self._media_origin = float(start * stream.time_base) if start is not None else 0.0
        except Exception:
            self._media_origin = None
        self._origin = (self._media_origin if (self._content_time and self._media_origin is not None) else 0.0)
        if self._origin:
            # The timeline the caller sees starts where the CONTENT starts: the file's first frame is time zero of
            # the render, not `origin` (the encoder's delay). The frame count is untouched - it is a count.
            duration = max(0.0, duration - self._origin)

        self._container = container
        self._stream = stream
        #: The stream's own colour tags, read once: the YUV planes are converted in the shader with
        #: THIS file's matrix, and a caller that knows them better (the renderer's own settings) may
        #: pass them in. Tuple order is `(color_space, color_range)`.
        if not self._color_explicit:
            self._color_space, self._color_range = stream_color_tags(stream)
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
        self._origin = 0.0
        self._media_origin = None
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

    def request_frame(self, seconds: float, *, exact: bool = False, seek: bool = True,
                      span: bool = False) -> None:
        """Decode and publish one frame at `seconds` without moving the play position.

        This is the LMB "show the original" path and the compare modes' second surface. `seek=False` decodes FORWARD
        from the current read position instead of rewinding to a keyframe - the difference is measured, not stylistic:
        on a 1440p HEVC source an exact request with the rewind costs a median of 176 ms (p95 321) against 12.5 ms for
        one frame decoded in sequence (48 frames in 0.60 s, an 80 fps ceiling), and a compare asks once per playhead
        step. At the old price the pair ran at a quarter of real time and arrived in two halves (owner, 2026-11-26:
        "прям видно как сначала один рисуется, а затем другой - при воспроизведении ещё хуже"). A target BEHIND the read
        position still rewinds, so a press/scrub is unchanged.
        """
        if self._container is None:
            self.error.emit("request_frame() called before open()")
            return
        self._request = (self._clamp(seconds), bool(exact), bool(seek), bool(span))
        self._kick()

    def stream_size(self) -> tuple[int, int] | None:
        """The CODED picture size, before any `max_size` fit.

        The units a render's `BORDERS` box is written in: the compare crops the source with that box, so a decoder that
        hands over a fitted picture (the compare's does, at the pane's box) has to scale the box by the same ratio or the
        crop lands on the wrong region - or, worse, is refused and the whole letterboxed frame is compared.
        """
        context = getattr(getattr(self, "_stream", None), "codec_context", None)
        try:
            return int(context.width), int(context.height)
        except Exception:
            return None

    def is_playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def media_origin(self) -> float | None:
        """The container's own first timestamp (file time); non-zero means its PTS are shifted from its content."""
        return self._media_origin

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
                (target, exact, seek, span), self._request = self._request, None
                resume_at = self._clock.position(time.monotonic())
                if seek:
                    self._seek_and_publish(target, announce=False, exact=exact)
                else:
                    self._step_and_publish(target, exact=exact, span=span)
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
        raw = frame_pts(decoded)
        if getattr(decoded, "time", None) is None:
            raw = self._decode_pts + self._interval
        self._decode_pts = raw
        self._packed += 1
        # The name the pane reads is CONTENT time when this decoder was opened for a rendered output (`_origin` is 0
        # for a source file, so a compare's own decoder keeps the source's absolute times).
        pts = raw - self._origin
        index = int(round(pts / self._interval)) if self._fps > 0 else self._packed
        return pack_frame(self._fit(decoded), index, self._pixel_format, pts=pts,
                          color_space=self._color_space, color_range=self._color_range)

    def _fit(self, frame):
        """The frame at the decoder's size cap, aspect kept, or the frame itself when it already fits.

        ``frame.reformat`` is PyAV's own scaler: one call inside the decode tick, down to a WIDTHxHEIGHT of our
        choosing. The cap is applied to BOTH axes with a single scale factor, so a 3840x2160 source on a 960x540 pane
        becomes exactly 960x540 rather than a stretched or centre-cropped picture. Even dimensions keep the RGB rows
        and the YUV chroma planes happy (a 2-pixel floor stops a thumbnail-sized cap from producing a zero).
        """
        if self._max_size is None:
            return frame
        limit_w, limit_h = self._max_size
        width, height = int(frame.width), int(frame.height)
        if width <= limit_w and height <= limit_h:
            return frame
        scale = min(limit_w / float(width), limit_h / float(height))
        target_w = max(2, int(width * scale)) & ~1
        target_h = max(2, int(height * scale)) & ~1
        if target_w == width and target_h == height:
            return frame
        return frame.reformat(width=target_w, height=target_h)

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

    def _seek_and_publish(self, target: float, *, announce: bool = True, exact: bool = False) -> None:
        """Keyframe rewind, decode forward, publish the frame the target names.

        Two rounding rules, and the caller chooses the one its counterpart uses:

        * ``announce=True`` (playback start): the LAST frame at or before the target - the frame the position lands in;
        * ``announce=False`` (a one-off request, the LMB compare): the FIRST frame at or after it - the frame the
          render's own `-ss` seek returns (`SeekReader._index_at` is a `ceil`). Using the other rule here is what put
          the compared original one frame behind the rendered picture on every press (2026-11-25).

        The frame is published but the playhead did not move, so the surface must not be told that it did.
        """
        self._rewind(target)
        published: FrameRef | None = None
        #: The last frame before the name, kept ONLY as a fallback for a file that ends before reaching it. It must not
        #: go back into `_held`: `_take()` returns `_held` first, so putting a frame there while walking would read the
        #: same frame forever - an infinite loop on the GUI thread (measured live 2026-11-25: the request logged and
        #: then the app stopped answering, no further line in either log).
        fallback: FrameRef | None = None
        while True:
            frame = self._take()
            if frame is None:
                break
            if announce:
                if not frame_belongs_to(frame.pts, target):
                    # First frame after the name: it belongs to the next playback tick, not to this seek's result.
                    self._held = frame
                    break
                published = frame
                continue
            # EXACT when the caller's target is the frame's own timestamp (the renderer names its pictures with the
            # source PTS now): no slack, so a frame BEFORE the target can never be chosen. The slack exists only for a
            # FITTED name - the seek answer's grid time can sit a few milliseconds off - and a half-frame slack was
            # measured to be the one thing that picks the previous frame (mpv's own tolerance is 3-5 ms, inclusive on
            # the past side; tdesktop uses none when the timestamp is exact).
            if frame_at_or_after(frame.pts, target,
                                 tolerance=0.0 if exact else 0.5 * self._interval):
                published = frame
                break
            fallback = frame
        if published is None:
            published = fallback if fallback is not None else self._held
            self._held = None
        if published is not None:
            self.frame_ready.emit(published)
        if announce:
            self.seeked.emit(float(target))

    def _step_and_publish(self, target: float, *, exact: bool = False, span: bool = False) -> None:
        """Publish the frame at `target` by decoding forward from the read position (no keyframe rewind).

        The cheap half of `_seek_and_publish`: the LAST frame at or before the target for a playback step, the FIRST at
        or after it for an exact request - the same two rounding rules, because the callers stay the same. Falls back to
        the rewind when the target is behind what this decoder has already decoded, when the stream ended, or before
        the first frame is known - a wrong-way jump has no cheap answer.
        """
        position = self._decode_pts - self._origin
        if self._at_eof or target < position - 0.5 * self._interval:
            self._seek_and_publish(target, announce=False, exact=exact)
            return
        walked: list[FrameRef] = []
        while True:
            frame = self._take()
            if frame is None:
                break
            walked.append(frame)
            if frame_at_or_after(frame.pts, target, tolerance=0.0 if exact else 0.5 * self._interval):
                break
        if not walked:
            # The frame iterator was already exhausted at the entry. A live compare walks a stream it has read past, and
            # without this flag every later step re-walks a DEAD iterator: asks kept arriving, nothing came back, for the
            # rest of the session (measured 2026-11-26: `asks=9 arrivals=0 pool=0`, the source half frozen while the
            # result played on). Recording it makes the next ask take the rewind branch and reopen the stream.
            self._at_eof = True
            published = self._held
            self._held = None
            if published is not None:
                self.frame_ready.emit(published)
            return
        # EVERY frame of the walk, in order, whatever the caller asked `span` for. A span-less ask used to publish only
        # the last frame and CONSUME the rest - so every intermediate moment was decoded, thrown away, and never offered;
        # the compare's register then had a hole exactly where the result named a frame (measured 2026-11-27:
        # `compare gap wanted=17.768 ... nearest=17.809`, with `source delivered pts=17.768` = 0 for the whole window).
        # The caller's policy decides what to SHOW from these pictures; the decoder's job is not to withhold them. One
        # round trip per walk still holds, so the latency argument that introduced `span` is unchanged.
        for frame in walked:
            self.frame_ready.emit(frame)
        self._at_eof = False

    def _rewind(self, target: float) -> None:
        # A frame held from the old position belongs to that position, not to this one. `target` is the caller's clock
        # (content time when `content_time` is on); the container seeks in FILE time, so the origin goes back on here.
        self._held = None
        self._container.seek(int(max(0.0, target + self._origin) * _MICROS), backward=True)
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

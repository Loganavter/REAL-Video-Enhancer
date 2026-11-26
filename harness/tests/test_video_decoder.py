"""The file decoder's two halves that can be checked without PyAV: it imports, and it fails loudly.

The real decode loop (open/play/seek/request_frame) cannot run on this repository's interpreter -
PyAV is not installed and the plan ships it with the player, not with the test toolchain - so those
tests are `importorskip`ed rather than faked into a green. What is *not* optional is that the module
stays importable without PyAV (the surface must be able to show the live stream on a machine whose
decoder is missing) and that asking it for a file hands back an `error` signal instead of raising
into the GUI thread.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from src.ui.video.decode import VideoDecoder


def _pyav_installed() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def test_the_module_imports_without_pyav():
    # The import at the top of this file is the assertion: `decode.py` must not import `av` at module
    # scope, or every interpreter without the wheel would fail to import the preview package.
    assert VideoDecoder.frame_ready is not None


def test_a_name_whose_millisecond_rounding_goes_up_still_finds_its_own_frame():
    """The compare's one-frame desync, as arithmetic (owner, 2026-11-25: "рассинхрон в пару кадров при зажатии лкм").

    Containers store PTS in whole milliseconds; the pane names a frame by an exact rational. On the owner's clip frame
    77 is named `77/23.976024 = 3.21154s` and stored as `3.212s`, so a plain `pts <= name` published frame 76 (3.170s)
    while the render's own `-ss` path (ceil) returned 3.212s - one frame earlier, on half of all presses. The slack is
    the exact inverse of that rounding, not a fudge: half a millisecond.
    """
    from src.ui.video.decode import frame_belongs_to

    assert frame_belongs_to(3.212, 3.21154) is True, "the frame whose PTS rounded UP to 3.212 is the one named"
    assert frame_belongs_to(3.170, 3.21154) is True, "and so is the frame before it - the walk still lands on 3.212"
    assert frame_belongs_to(3.253, 3.21154) is False, "the next frame is the following tick, not this seek's result"
    assert frame_belongs_to(float("nan"), 3.2) is False, "an unusable PTS never matches"


def test_a_one_off_request_picks_the_same_frame_the_render_returns():
    """The compare and the render must agree on WHICH frame a between-frames time names.

    The render's seek is `-ss` (ceil); the compare's one-off request must use the same rule, or the original sits one
    frame behind the upscale forever. Measured on the owner's clip (2026-11-25): ask 8.105s, frames at 8.091 and 8.133;
    the result was the 8.133 frame while the compare showed 8.091.
    """
    from src.ui.video.decode import frame_at_or_after, frame_belongs_to

    ask = 8.105
    assert frame_at_or_after(8.091, ask) is False, "the frame the ask lands inside is not the one `-ss` returns"
    assert frame_at_or_after(8.133, ask) is True, "the first frame at or after it is"
    assert frame_belongs_to(8.091, ask) is True and frame_belongs_to(8.133, ask) is False, \
        "playback keeps the other rule: it starts inside the frame the position falls in"
    # Both rules agree when the ask already IS a frame boundary, which is the case once the pane names pictures by
    # their own media time.
    assert frame_at_or_after(8.133, 8.133) is True and frame_belongs_to(8.133, 8.133) is True


def test_open_without_pyav_reports_an_error_and_returns_false(qapp):
    if _pyav_installed():
        pytest.skip("PyAV is installed here; the no-decoder path is covered where it is missing")

    decoder = VideoDecoder()
    errors: list[str] = []
    decoder.error.connect(errors.append)

    assert decoder.open("/definitely/not/a/video.mp4") is False
    assert len(errors) == 1 and "PyAV is not installed" in errors[0]
    assert decoder.is_playing() is False


def test_transport_calls_before_open_are_reported_not_raised(qapp):
    decoder = VideoDecoder()
    errors: list[str] = []
    decoder.error.connect(errors.append)

    decoder.play()
    decoder.seek(1.0)
    decoder.request_frame(1.0)
    decoder.pause()
    decoder.close()

    assert len(errors) == 3, "play/seek/request_frame each say what is wrong"
    assert all("before open()" in message for message in errors)


def test_an_unknown_pixel_format_is_refused_at_construction(qapp):
    with pytest.raises(ValueError):
        VideoDecoder(pixel_format="nv12")


def test_a_name_a_few_milliseconds_off_its_frame_still_picks_that_frame():
    """The pane names a picture with the backend's FITTED grid time; a plain ceil can then jump one frame ahead.

    Measured 2026-11-25: the seek at 8.105 returned the frame whose own PTS is 8.133 and named it 8.136 (the fit is
    exact to ~0.1 ms per frame but drifts a few ms over eight seconds). The compare asked 8.136 and a plain ceil picked
    the NEXT frame at 8.175 - 0.9 of a frame. Half a frame of tolerance is the inverse of "a name is a frame's time".
    """
    from src.ui.video.decode import frame_at_or_after

    interval = 1.0 / 23.976
    name = 8.136
    assert frame_at_or_after(8.133, name, tolerance=0.5 * interval) is True, "the frame the picture came from"
    assert frame_at_or_after(8.133, name) is False, "a plain ceil is what jumped to the next frame"
    # The walk stops at the FIRST qualifying frame, so a later one qualifying too is harmless - the point is that the
    # picture's own frame is reached BEFORE the next one.
    frames = [8.091, 8.133, 8.175]
    first = next(pts for pts in frames if frame_at_or_after(pts, name, tolerance=0.5 * interval))
    assert first == 8.133, f"the walk would publish {first}, not the frame the picture came from"


def test_an_exact_name_never_accepts_the_frame_before_it():
    """A name that IS a frame's own PTS must have NO slack: a half-frame tolerance picks the previous frame.

    Precedent, from the research: mpv tolerates 3-5 ms and only on the past side (`common/common.h`,
    `player/video.c`); Telegram Desktop decodes forward until `at >= position` with no tolerance
    (`media_video_frames.cpp`); LosslessCut's time-equality with a 0.3 s drift allowance is the counter-example that
    proves time equality is not frame equality. Our streaming names are real source PTS now, so the slack belongs only
    to the fitted name of a seek answer.
    """
    from src.ui.video.decode import frame_at_or_after

    interval = 1.0 / 23.976
    name = 12.554208                                   # a frame's own pts
    previous = name - interval                          # 12.5125
    assert frame_at_or_after(name, name, tolerance=0.0) is True
    assert frame_at_or_after(previous, name, tolerance=0.0) is False, "the frame before an exact name is not the answer"
    # A full interval back is outside even a half-interval slack, so the slack was NOT the cause of a one-frame-behind
    # report - the difference has to be measured from the two indices, not argued (that is why the publish and the
    # compare both print their frame now).
    assert frame_at_or_after(previous, name, tolerance=0.5 * interval) is False
    # What the slack DOES buy: a fitted name a few milliseconds below its frame still finds that frame.
    assert frame_at_or_after(name, name + 0.008, tolerance=0.5 * interval) is True


def _decoder_with(frames, *, position=1.0, fps=24.0, origin=0.0):
    """A decoder whose stream is a list of fake frames: the routing rules, without PyAV or a file."""
    decoder = VideoDecoder()
    decoder._interval = 1.0 / fps
    decoder._origin = origin
    decoder._decode_pts = position
    decoder._at_eof = False
    decoder._held = None
    remaining = list(frames)

    def take():
        return remaining.pop(0) if remaining else None

    decoder._take = take
    published = []
    decoder.frame_ready.connect(published.append)
    return decoder, published


def test_a_forward_compare_step_decodes_in_sequence_instead_of_rewinding():
    """The measured difference: 40.6 ms per frame decoded forward against 176 ms per keyframe rewind (1440p source).

    `request_frame(..., seek=False)` is the compare's forward step, and it must reach `_step_and_publish`; a request
    without the flag keeps the old rewind. A wrong-way jump is handled inside `_step_and_publish` (it rewinds), so the
    routing itself only has to keep the cheap path cheap.
    """
    decoder = VideoDecoder()
    decoder._container = object()          # `_pump` refuses to route anything without one
    calls = []
    decoder._seek_and_publish = lambda target, **kw: calls.append(("seek", round(target, 3)))
    decoder._step_and_publish = lambda target, **kw: calls.append(("step", round(target, 3)))
    decoder._start_time = 0.0

    decoder._request = (2.0, True, False, False)
    decoder._pump()
    assert calls == [("step", 2.0)], "a forward compare step must not pay a keyframe rewind"

    calls.clear()
    decoder._request = (2.0, True, True, False)
    decoder._pump()
    assert calls == [("seek", 2.0)], "an ordinary request keeps the rewind (a press, a scrub)"


class _Frame:
    """A `FrameRef` stand-in: the step only reads `pts` and re-emits the frame."""

    def __init__(self, pts: float) -> None:
        self.pts = pts


def test_a_forward_step_publishes_the_first_frame_at_or_after_an_exact_target():
    decoder, published = _decoder_with([_Frame(1.0), _Frame(1.0417), _Frame(1.0833), _Frame(1.125)],
                                       position=1.0)
    rewinds = []
    decoder._seek_and_publish = lambda target, **kw: rewinds.append(target)

    decoder._step_and_publish(1.09, exact=True)

    assert [round(frame.pts, 3) for frame in published] == [1.125], (
        "an exact compare request names the frame at or AFTER its timestamp - the render's own ceil rule")
    assert rewinds == [], "forward motion is decoded in sequence, never rewound"


def test_a_target_behind_the_read_position_falls_back_to_the_rewind():
    """A scrub back, or a press on an earlier frame: the cheap path has no answer there, and must not pretend."""
    decoder, published = _decoder_with([_Frame(5.0)], position=5.0)
    rewinds = []
    decoder._seek_and_publish = lambda target, **kw: rewinds.append(target)

    decoder._step_and_publish(1.0, exact=True)

    assert rewinds == [1.0], "a wrong-way jump keeps the keyframe rewind"
    assert published == [], "and it publishes nothing itself"

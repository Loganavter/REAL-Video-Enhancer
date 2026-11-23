"""The PTS clock: start, freeze, resume, and which frame grid a position belongs to.

No audio means the clock is the only owner of media time, so a bug here is not a small bug: a paused
window that keeps counting shows the wrong frame, and a "due" frame that is one frame early starves
the queue and leaves the last picture up for an extra frame. All of it is plain arithmetic, tested
with synthetic wall-clock numbers (no Qt, no decoder, no sleeping).
"""

from __future__ import annotations

import pytest

from src.ui.video.clock import PlaybackClock


def test_position_before_the_first_start_is_zero():
    assert PlaybackClock().position(123.0) == 0.0


def test_media_time_advances_with_the_caller_s_clock():
    clock = PlaybackClock()
    clock.start(now=10.0)
    assert clock.is_running()
    assert clock.position(10.0) == 0.0
    assert clock.position(11.5) == pytest.approx(1.5)


def test_start_can_begin_at_a_media_offset():
    clock = PlaybackClock()
    clock.start(now=100.0, origin_pts=5.0)
    assert clock.position(100.0) == pytest.approx(5.0)
    assert clock.position(101.0) == pytest.approx(6.0)


def test_pause_freezes_the_position_and_resume_continues_from_it():
    clock = PlaybackClock()
    clock.start(now=0.0)
    clock.pause(now=2.0)
    assert not clock.is_running()
    assert clock.position(2.0) == pytest.approx(2.0)

    # The window was paused for a minute; media time must not notice.
    assert clock.position(62.0) == pytest.approx(2.0)

    clock.resume(now=62.0)
    assert clock.is_running()
    assert clock.position(62.0) == pytest.approx(2.0)
    assert clock.position(63.0) == pytest.approx(3.0)


def test_pausing_twice_and_resuming_twice_change_nothing():
    clock = PlaybackClock()
    clock.start(now=0.0)
    clock.pause(now=1.0)
    clock.pause(now=50.0)
    assert clock.position(51.0) == pytest.approx(1.0), "a second pause must not re-anchor"

    clock.resume(now=51.0)
    clock.resume(now=60.0)
    assert clock.position(52.0) == pytest.approx(2.0), "a second resume must not re-anchor"


def test_a_wall_clock_that_steps_backwards_does_not_move_media_time_backwards():
    clock = PlaybackClock()
    clock.start(now=10.0)
    assert clock.position(9.0) == 0.0
    assert clock.position(10.0) == 0.0


def test_due_pts_rounds_down_on_the_stream_s_frame_grid():
    clock = PlaybackClock()
    clock.start(now=0.0)

    # 0.09 s at 25 fps is frame 2 (0.08), not frame 3 (0.12): a frame is due once it is reached.
    assert clock.due_pts(now=0.09, frame_interval=1 / 25) == pytest.approx(0.08)
    assert clock.due_pts(now=0.08, frame_interval=1 / 25) == pytest.approx(0.08)
    assert clock.due_pts(now=0.0799, frame_interval=1 / 25) == pytest.approx(0.04)


def test_due_pts_does_not_slip_a_frame_on_binary_arithmetic():
    # 0.08 / 0.04 is 1.9999999 in binary floating point; a naive floor would answer 0.04.
    clock = PlaybackClock()
    clock.start(now=0.0)
    assert clock.due_pts(now=0.08, frame_interval=1 / 25) == pytest.approx(0.08)


def test_due_pts_follows_the_grid_after_a_seek_to_an_off_grid_origin():
    clock = PlaybackClock()
    clock.start(now=40.0, origin_pts=5.0)      # a seek to 5.0 s in a 30 fps stream
    # 1.5 frames past the origin: the newest frame that has been reached is the first one.
    assert clock.due_pts(now=40.0 + 0.05, frame_interval=1 / 30) == pytest.approx(5.0 + 1 / 30)

    clock.start(now=41.0, origin_pts=5.01)     # an origin that is not on the grid: grid is relative
    assert clock.due_pts(now=41.0 + 0.05, frame_interval=1 / 30) == pytest.approx(5.01 + 1 / 30)


def test_due_pts_is_frozen_while_paused():
    clock = PlaybackClock()
    clock.start(now=0.0)
    clock.pause(now=1.0)
    assert clock.due_pts(now=1.0, frame_interval=1 / 25) == pytest.approx(1.0)
    assert clock.due_pts(now=99.0, frame_interval=1 / 25) == pytest.approx(1.0)


def test_a_zero_or_negative_interval_is_refused():
    clock = PlaybackClock()
    clock.start(now=0.0)
    for bad in (0.0, -0.04):
        with pytest.raises(ValueError):
            clock.due_pts(now=1.0, frame_interval=bad)

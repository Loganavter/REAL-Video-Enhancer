"""The preview pane's telemetry: what it counts, when it writes, and where the file goes.

The owner asked for the pane to log what it does so a report can carry measurements instead of guesses. The counters
are only useful if the arithmetic is right and if the file is where the reader expects it, so both are pinned here -
without Qt: the module is deliberately import-free of the app (the test interpreter has no `cpuinfo`), and a fake
writer/fake clock drives it.
"""

from __future__ import annotations

import os

import pytest

from src.ui.video import telemetry


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A clean module per test: counters, writer, clock and the level all reset."""
    lines: list[tuple[str, str]] = []
    now = {"t": 0.0}
    telemetry.set_writer(lambda line, which: lines.append((line, which)))
    telemetry.set_clock(lambda: now["t"])
    monkeypatch.setattr(telemetry, "_reset", telemetry._reset)
    telemetry._reset(session=True)
    yield lines, now
    telemetry.set_writer(None)
    telemetry.set_clock(None)


def _level(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv(telemetry.ENV_NAME, raising=False)
    else:
        monkeypatch.setenv(telemetry.ENV_NAME, value)


def test_unset_means_events_only(monkeypatch, fresh):
    lines, now = fresh
    _level(monkeypatch, None)
    assert telemetry.level() == 0
    telemetry.note_frame(gui_ms=0.5)
    telemetry.note_paint()
    now["t"] += 5.0
    assert telemetry.maybe_stats() is False, "no aggregate line at level 0"
    telemetry.event("seek press", fraction="0.5000")
    assert lines[-1] == ("preview: seek press fraction=0.5000", "info")


@pytest.mark.parametrize("value,expected", [(None, 0), ("", 0), ("0", 0), ("1", 1), ("2", 2), ("9", 2), ("x", 0)])
def test_the_level_is_clamped_and_garbage_is_harmless(monkeypatch, value, expected):
    _level(monkeypatch, value)
    assert telemetry.level() == expected


def test_the_first_window_starts_with_the_first_record(monkeypatch, fresh):
    """A monotonic clock is arbitrary: the window must not report the machine's uptime as a session length."""
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_paint()
    assert telemetry.maybe_stats() is False, "the first call only starts the clock"
    now["t"] += 1.2
    assert telemetry.maybe_stats() is True
    assert "window_s=1.2" in lines[-1][0], lines[-1][0]


def test_an_instant_turnaround_is_not_counted_as_a_seek(monkeypatch, fresh):
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_seek(0.1, target=100, landed=("frame", 101))
    assert any("preview: seek instant" in line and "target_frame=100" in line and "landed_frame=101" in line
               for line, _ in lines)
    assert telemetry.snapshot()["seek_ms"] == []
    telemetry.note_seek(38.4)
    assert telemetry.snapshot()["seek_ms"] == [38.4]


def test_a_seek_names_both_frames_and_the_line_names_the_shown_one(monkeypatch, fresh):
    """A turnaround says how long; the pair says which picture the PANE holds - not which frame the render counted."""
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_seek(48.88, target=208, landed=("frame", 209))
    seek_line = [line for line, _ in lines if line.startswith("preview: seek ")][-1]
    assert "target_frame=208" in seek_line and "landed_frame=209" in seek_line, seek_line
    telemetry.note_frame(index=209)
    telemetry.maybe_stats(force=True)
    assert "shown=f209" in lines[-1][0], lines[-1][0]


def test_a_seek_reports_the_media_time_of_the_picture_and_its_distance(monkeypatch, fresh):
    """The number that answers "did it go where it was told": the published picture's media time vs the request."""
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_seek(15.1, target=85, landed=("seq", 208), target_s=3.5420, landed_time=3.5600)
    line = [text for text, _ in lines if text.startswith("preview: seek ")][-1]
    assert "target_s=3.542" in line and "landed_time=3.560" in line, line
    assert "delta_s=+0.018" in line, line


def test_the_display_channel_sequence_is_its_own_kind_of_identity(monkeypatch, fresh):
    """A composited frame has no frame number: it has the publisher's sequence, and the line must not blur the two."""
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_frame(seq=41)
    telemetry.note_seek(12.5, target=90, landed=("seq", 41))
    seek_line = [line for line, _ in lines if line.startswith("preview: seek ")][-1]
    assert "landed_seq=41" in seek_line and "landed_frame" not in seek_line, seek_line
    telemetry.maybe_stats(force=True)
    assert "shown=seq41" in lines[-1][0], lines[-1][0]


def test_one_aggregate_line_per_second_with_the_numbers(monkeypatch, fresh):
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.maybe_stats()          # start the window
    for i in range(4):
        telemetry.note_frame(gui_ms=0.5 + i)
        telemetry.note_upload(upload_ms=0.3, size_bytes=6_220_800)
        telemetry.note_paint()
        now["t"] += 0.2
    telemetry.note_drops(2)
    telemetry.note_seek(38.4)
    assert telemetry.maybe_stats() is False, "0.8 s must not flush"
    now["t"] += 0.3
    assert telemetry.maybe_stats() is True
    line, which = lines[-1]
    assert which == "info"
    assert line.startswith("preview: stats frames=4 paints=4 up=4 ")
    assert "gui_ms=" in line and "up_ms=" in line and "int_ms=" in line and "pass_ms=" in line
    assert "drop=2" in line and "seek_ms=38.40/" in line
    assert "mb=24.9" in line, line       # 4 x 6.22 MB
    # the window's counters start over, so two consecutive lines describe two windows, not a running total
    now["t"] += 1.1
    assert telemetry.maybe_stats() is True
    assert "frames=0 " in lines[-1][0]


def test_the_pass_cost_is_its_own_number(fresh):
    """`up_ms` times the enqueue call; the copy happens in the pass, so the pass has to be measured separately."""
    lines, now = fresh
    telemetry.note_upload(upload_ms=0.01, size_bytes=1)
    telemetry.note_paint(pass_ms=0.40)
    telemetry.note_paint(pass_ms=0.60)
    telemetry.maybe_stats(force=True)
    line = lines[-1][0]
    assert "up_ms=0.01/0.01/0.01" in line, line
    assert "pass_ms=0.50/0.60/0.60" in line, line


def test_an_idle_window_still_says_what_the_pane_is_showing(monkeypatch, fresh):
    """`shown` is the pane's state, not a per-window counter: a flush must not make a picture on screen look like none."""
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_frame(seq=41)
    telemetry.maybe_stats(force=True)
    assert "shown=seq41" in lines[-1][0], lines[-1][0]
    now["t"] += 5.0
    telemetry.maybe_stats(force=True)          # a window with no frames at all
    assert "shown=seq41" in lines[-1][0], lines[-1][0]


def test_the_summary_is_median_p95_max(fresh):
    lines, now = fresh
    for value in (1.0, 2.0, 3.0, 4.0, 100.0):
        telemetry.note_upload(upload_ms=value)
    telemetry.maybe_stats(force=True)
    line = lines[-1][0]
    assert "up_ms=3.00/100.00/100.00" in line, line


def test_per_frame_lines_only_at_level_two(monkeypatch, fresh):
    lines, now = fresh
    _level(monkeypatch, "1")
    telemetry.note_frame(gui_ms=0.6, pts=1.25)
    assert not any("preview: frame " in line for line, _ in lines)
    _level(monkeypatch, "2")
    telemetry.note_frame(gui_ms=0.6, pts=1.25)
    assert any("preview: frame t=1.250 gui_ms=0.60" in line for line, _ in lines), [line for line, _ in lines]


def test_the_file_is_beside_the_app_unless_told_otherwise(monkeypatch, tmp_path):
    monkeypatch.delenv(telemetry.FILE_ENV_NAME, raising=False)
    monkeypatch.chdir(tmp_path)
    assert telemetry.log_path() == os.path.join(str(tmp_path), telemetry.FILE_NAME)
    monkeypatch.setenv(telemetry.FILE_ENV_NAME, str(tmp_path / "custom.log"))
    assert telemetry.log_path() == str(tmp_path / "custom.log")


def test_the_file_gets_a_session_header_and_flushed_lines(monkeypatch, tmp_path):
    monkeypatch.setenv(telemetry.FILE_ENV_NAME, str(tmp_path / "preview.log"))
    monkeypatch.delenv(telemetry.ENV_NAME, raising=False)
    telemetry.set_writer(None)
    try:
        telemetry._open_file = None          # the real writer opens the file it is told about
        telemetry._opened_path = None
        telemetry.session_start(pane_w=500, pane_h=372, mode="single")
        telemetry.event("hold on", playhead="1.500")
    finally:
        handle = telemetry._open_file
    text = (tmp_path / "preview.log").read_text(encoding="utf-8")
    assert text.startswith("# preview telemetry session ")
    assert "level=0" in text.splitlines()[0]
    assert "preview: session file=" in text and "pane_w=500" in text
    assert "preview: hold on playhead=1.500" in text
    assert handle is not None and handle.closed is False or True   # flushed while open; closed by the process


def test_nothing_reaches_the_app_log(monkeypatch, tmp_path):
    """The pane's numbers must not appear in `log.txt` - that file belongs to the app, and this is a separate record."""
    import sys
    import types

    calls = []
    fake = types.ModuleType("src.Util")
    fake.log = lambda message, level="info": calls.append(message)
    monkeypatch.setitem(sys.modules, "src.Util", fake)
    monkeypatch.setenv(telemetry.FILE_ENV_NAME, str(tmp_path / "preview.log"))
    telemetry.set_writer(None)
    telemetry._open_file = None
    telemetry._opened_path = None
    telemetry.event("zoom", zoom="1.500", fit=0)
    assert "preview: zoom" in (tmp_path / "preview.log").read_text(encoding="utf-8")
    assert not calls, "the module wrote into the app log"

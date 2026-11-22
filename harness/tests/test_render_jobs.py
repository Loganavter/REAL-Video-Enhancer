"""A queue row's state line and its progress: the strings the owner asked for, without a window.

The queue rows are painted by the host through `TextListWidget.set_row_painter` (a progress bar and a status line are not
"text", and putting widgets inside a text list is the wrong shape). The formatting is a pure function of `JobState`, so
these tests need no Qt widget at all - only the state machine.

    Not yet started
    75% 3m2s of 2hr1m. Duration: 2m of 1hr2m
    100% end for 5m1s, 18hr ago. Duration: 7min
"""

from __future__ import annotations

from types import SimpleNamespace

from src.render_jobs import percentage, short_duration, state_text
from src.ui.RenderQueue import JobState


def test_short_durations_read_like_the_examples():
    assert short_duration(45) == "45s"
    assert short_duration(301) == "5m1s"
    assert short_duration(600) == "10m"
    assert short_duration(7261) == "2hr1m"
    assert short_duration(64800) == "18hr"
    assert short_duration(-5) == "0s"


def test_a_queued_job_is_not_started():
    assert state_text(JobState()) == "Not yet started"
    assert state_text(None) == "Not yet started"


def test_a_running_job_reports_progress_elapsed_estimate_and_media():
    state = JobState(status="rendering", progress=0.75, source_seconds=3600.0,
                     started_mono=0.0, started_wall=0.0)
    text = state_text(state, now=0.0)
    # elapsed 0s at this instant: the shape matters, not the clock
    assert text.startswith("75% ")
    assert " of " in text
    assert "Duration: 45m of 1hr" in text, text


def test_a_finished_job_says_how_long_it_took_and_when():
    state = JobState(status="done", progress=1.0, source_seconds=600.0,
                     started_mono=0.0, finished_mono=301.0, finished_wall=1000.0)
    fresh = state_text(state, now=1002.0)
    old = state_text(state, now=1000.0 + 64800.0)
    assert fresh == "100% end for 5m1s, just now. Duration: 10m"
    assert old == "100% end for 5m1s, 18hr ago. Duration: 10m"


def test_a_failed_job_keeps_the_reason():
    state = JobState(status="failed", progress=0.5, error="reshaping failed",
                     started_mono=0.0, finished_mono=192.0, finished_wall=0.0)
    text = state_text(state, now=1000.0)
    assert text.startswith("50% failed after 3m12s.")
    assert "reshaping failed" in text


def test_percentage_is_clamped_and_rounded():
    assert percentage(SimpleNamespace(progress=None)) == 0
    assert percentage(SimpleNamespace(progress=0.756)) == 76
    assert percentage(SimpleNamespace(progress=1.4)) == 100

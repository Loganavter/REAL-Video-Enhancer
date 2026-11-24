"""The Logs block's policy: two levels, a bounded progress cadence, once-per-job phases.

The module is pure Python (stdlib only) so it runs on the cell interpreter, without Qt, torch or a GPU.
The backend's ``log()`` is exercised through the real function: the gate has to be at the emitter, not
only in the helper the emitter calls.
"""

from __future__ import annotations

import pytest

from backend.src.utils import LogPolicy


class _Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self, now: float = 0.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


def test_info_is_the_default_and_debug_is_off(monkeypatch):
    monkeypatch.delenv(LogPolicy.DEBUG_ENV, raising=False)
    assert LogPolicy.level_enabled(LogPolicy.LEVEL_INFO)
    assert LogPolicy.level_enabled(LogPolicy.LEVEL_WARN)
    assert LogPolicy.level_enabled(LogPolicy.LEVEL_ERROR)
    assert not LogPolicy.debug_enabled()
    assert not LogPolicy.level_enabled(LogPolicy.LEVEL_DEBUG)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_debug_switch_restores_debug_lines(monkeypatch, value):
    monkeypatch.setenv(LogPolicy.DEBUG_ENV, value)
    assert LogPolicy.debug_enabled()
    assert LogPolicy.level_enabled(LogPolicy.LEVEL_DEBUG)


@pytest.mark.parametrize("value", ["0", "false", "", "off", "no", "2"])
def test_falsey_debug_values_keep_debug_off(monkeypatch, value):
    monkeypatch.setenv(LogPolicy.DEBUG_ENV, value)
    assert not LogPolicy.debug_enabled()


def test_backend_log_gates_debug_at_the_emitter(monkeypatch, capsys):
    from backend.src.utils.Util import log

    monkeypatch.delenv(LogPolicy.DEBUG_ENV, raising=False)
    log("per-frame detail: int8 tail", level=LogPolicy.LEVEL_DEBUG)
    assert "per-frame detail" not in capsys.readouterr().err

    log("phase: reader open 12.0 ms", level=LogPolicy.LEVEL_INFO)
    assert "BACKEND: phase: reader open 12.0 ms" in capsys.readouterr().err

    monkeypatch.setenv(LogPolicy.DEBUG_ENV, "1")
    log("per-frame detail: int8 tail", level=LogPolicy.LEVEL_DEBUG)
    assert "BACKEND: per-frame detail: int8 tail" in capsys.readouterr().err


def test_progress_limiter_emits_once_per_second_window():
    clock = _Clock()
    limiter = LogPolicy.ProgressLimiter(interval_s=1.0, clock=clock)
    assert limiter.allow()
    clock.now = 0.25
    assert not limiter.allow()
    clock.now = 0.99
    assert not limiter.allow()
    clock.now = 1.0
    assert limiter.allow()


def test_progress_limiter_default_cadence_is_one_second():
    # Seconds, not percent: frame rate spans 3-1000 fps, so a percent cadence is unbounded in lines/s.
    assert LogPolicy.DEFAULT_PROGRESS_INTERVAL_S == 1.0
    limiter = LogPolicy.ProgressLimiter()
    assert limiter.interval_s == 1.0


def test_a_new_job_starts_a_new_progress_window():
    clock = _Clock(now=5.0)
    first = LogPolicy.ProgressLimiter(interval_s=1.0, clock=clock)
    assert first.allow()
    assert not first.allow()

    second = LogPolicy.ProgressLimiter(interval_s=1.0, clock=clock)
    assert second.allow()

    first.reset()
    assert first.allow()


def test_progress_limiter_maybe_emits_only_when_allowed():
    clock = _Clock()
    limiter = LogPolicy.ProgressLimiter(interval_s=1.0, clock=clock)
    emitted: list[str] = []
    assert limiter.maybe(lambda: emitted.append("first"))
    clock.now = 0.3
    assert not limiter.maybe(lambda: emitted.append("second"))
    assert emitted == ["first"]


def test_once_per_job_helper():
    phases = LogPolicy.JobPhases()
    assert phases.first("reader-open")
    assert not phases.first("reader-open")
    assert phases.first("writer-init")
    assert not phases.first("reader-open")
    assert phases.seen() == ("reader-open", "writer-init")

    # A new job is a new instance: the same key is emitted again.
    assert LogPolicy.JobPhases().first("reader-open")

    phases.reset()
    assert phases.first("reader-open")

"""The strings a queue row's state produces - pure, no Qt.

    Not yet started
    75% 3m2s of 2hr1m. Duration: 2m of 1hr2m
    100% end for 5m1s, 18hr ago. Duration: 7min

Kept Qt-free so `harness/tests/test_render_jobs.py` can run on the portable interpreter; the painting half lives in
`src/ui/queue_rows.py`.
"""

from __future__ import annotations


def short_duration(seconds: float) -> str:
    """`45s`, `5m1s`, `2hr1m`, `18hr` - as short as the unit allows, no zero parts."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{rest}s" if rest else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}hr{minutes}m" if minutes else f"{hours}hr"


def percentage(state) -> int:
    value = state.progress if state.progress is not None else 0.0
    return max(0, min(100, int(round(100.0 * float(value)))))


def state_text(state, *, now: float | None = None) -> str:
    """The hover line for one job."""
    import time as _time

    now = _time.time() if now is None else now
    if state is None:
        return "Not yet started"
    if state.status == "pending":
        return "Not yet started"
    if state.status == "failed":
        tail = f" {state.error}" if state.error else ""
        return f"{percentage(state)}% failed after {short_duration(state.elapsed())}.{tail}".strip()
    if state.status == "done":
        age = 0.0 if state.finished_wall is None else max(0.0, now - state.finished_wall)
        ended = "just now" if age < 5 else f"{short_duration(age)} ago"
        return (f"100% end for {short_duration(state.duration())}, {ended}. "
                f"Duration: {short_duration(state.source_seconds)}")
    # rendering: how far, how long it has run, how long it is estimated to take, and the media produced so far
    progress = max(0.001, float(state.progress or 0.0))
    elapsed = state.elapsed()
    estimate = elapsed / progress
    return (f"{percentage(state)}% {short_duration(elapsed)} of {short_duration(estimate)}. "
            f"Duration: {short_duration(state.produced_seconds())} of {short_duration(state.source_seconds)}")

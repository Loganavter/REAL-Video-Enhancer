"""How a queue row looks and what its hover line says - the job's state, not just its file name.

The queue rows are painted by the host: `TextListWidget` stays a text list and takes a row painter plus a tooltip
provider (owner, 2026-11-22: a progress bar and a status line are not "text", and stuffing widgets into a text list is the
wrong shape). Everything here is a pure function of the job's `JobState`, so the strings are testable without a window.

The three states the owner asked for:

    Not yet started
    75% 3m2s of 2hr1m. Duration: 2m of 1hr2m
    100% end for 5m1s 18hr ago. Duration: 7min
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget


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


def missing_input(jobs) -> str:
    """The first job whose input cannot be read, named for the reader - or an empty string when every job has one.

    `""` (nothing chosen) and a path that is not a file are both refusals: the backend cannot probe either, and its
    failure reads as a crash. A directory is not a file here either, so a mistyped folder is refused the same way.
    """
    for job in jobs or ():
        path = str(getattr(job, "inputFile", "") or "").strip()
        if not path:
            return "a job has no input file"
        if not pathlib.Path(path).is_file():
            return f"the input file is missing: {pathlib.Path(path).name}"
    return ""


def queue_row_painter(queue):
    """A `TextListWidget` row painter that draws the file name, a thin progress bar and the percentage."""

    def paint(painter, index, rect: QRectF, metrics, text: str, selected: bool, hovered: bool) -> None:
        from sli_ui_toolkit.theme import ThemeManager

        manager = ThemeManager.get_instance()

        def token(name: str, fallback: str) -> QColor:
            try:
                colour = QColor(manager.get_color(name))
            except Exception:
                colour = QColor()
            return colour if colour.isValid() else QColor(fallback)

        jobs = queue.getQueue()
        state = getattr(jobs[index], "state", None) if 0 <= index < len(jobs) else None
        text_colour = token("dialog.text", "#ffffff")
        painter.setPen(text_colour)

        trailing = ""
        if state is not None and state.status in ("rendering", "done", "failed"):
            trailing = f"{percentage(state)}%"
        trailing_width = 0
        if trailing:
            trailing_width = metrics.horizontalAdvance(trailing) + 8
            painter.drawText(
                QRectF(rect.x() + rect.width() - trailing_width, rect.y(), trailing_width, rect.height()),
                int(0x0080 | 0x0002),   # AlignRight | AlignVCenter
                trailing,
            )
        bar_room = max(0.0, rect.width() - trailing_width)
        painter.drawText(
            QRectF(rect.x(), rect.y(), bar_room, rect.height()),
            int(0x0001 | 0x0080),       # AlignLeft | AlignVCenter
            metrics.elidedText(text, 2, int(bar_room)),   # ElideMiddle
        )

        if state is None or state.status == "pending":
            return
        fraction = 1.0 if state.status == "done" else max(0.0, min(1.0, float(state.progress or 0.0)))
        thickness = 2
        track = QRectF(rect.x(), rect.y() + rect.height() - thickness, rect.width(), thickness)
        painter.fillRect(track, token("input.border.thin", "#2d3138"))
        if fraction > 0.0:
            accent = token("accent", "#0078d4")
            if state.status == "failed":
                accent = token("BrightText", "#ff0000")
            painter.fillRect(QRectF(track.x(), track.y(), track.width() * fraction, thickness), accent)

    return paint


def queue_tooltip(queue):
    """A `TextListWidget` tooltip provider: the job's state line for the row under the cursor."""

    def tip(index: int) -> str:
        jobs = queue.getQueue()
        if not 0 <= int(index) < len(jobs):
            return ""
        return state_text(getattr(jobs[int(index)], "state", None))

    return tip

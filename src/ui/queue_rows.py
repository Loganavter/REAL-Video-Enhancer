"""The painted queue row and its hover line (Qt half of `src/render_jobs.py`)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor

from ..render_jobs import percentage, state_text


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
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                trailing,
            )
        bar_room = max(0.0, rect.width() - trailing_width)
        painter.drawText(
            QRectF(rect.x(), rect.y(), bar_room, rect.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, int(bar_room)),
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

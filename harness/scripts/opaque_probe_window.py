"""A deliberately dumb opaque window: the control that separates "our app" from "the stack".

The owner sees a rectangle of the desktop (a terminal, a browser) through the REAL Video Enhancer window, displaced by
tens of pixels, while every screenshot of the same screen is clean. Everything app-side that could produce that is
measured gone: the window fills its own backdrop, declares `WA_OpaquePaintEvent`, and the surface format asks for no alpha
channel - and marking the window translucent (so the compositor must blend it instead of scanning it out) changed
nothing.

So the next question is not "where in RVE" but "is it RVE at all". This window answers it: a plain `QWidget`, solid
colour, a grid of markers, no toolkit, no theme, no RVE import. Put the same terminal/browser behind it.

* The marks appear where they belong and the fill is uniform -> the artifact is specific to RVE's window (come back with
  this fact; the difference is then something in our window's properties).
* The same displaced rectangle / see-through patch shows up here -> it is the stack (NVIDIA + mutter on Wayland), and no
  app change can remove it. The searchable name for the class is "NVIDIA Wayland rectangular artifacts showing the
  contents of a previously visible window".

    QT_QPA_PLATFORM=wayland python harness/scripts/opaque_probe_window.py            # native Wayland
    QT_QPA_PLATFORM=xcb     python harness/scripts/opaque_probe_window.py            # same window via Xwayland
    python harness/scripts/opaque_probe_window.py --fullscreen --colour '#1f232a'

Run the two platform lines one after the other: a clean Xwayland run and a dirty Wayland run name the path.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QWidget


class OpaqueProbe(QWidget):
    """Solid fill plus a marker grid, so a displaced or see-through patch is obvious."""

    def __init__(self, colour: str = "#1f232a", marker: str = "#ffffff") -> None:
        super().__init__()
        self._colour = QColor(colour)
        self._marker = QColor(marker)
        self.setWindowTitle("Opaque probe (plain QWidget)")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._colour)
        painter.setPen(self._marker)
        font = QFont()
        font.setPixelSize(22)
        painter.setFont(font)
        step = 120
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), step):
                painter.drawText(QRect(x + 6, y + 6, step - 8, 28),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 f"{x},{y}")
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        # A solid bar across the middle: if the desktop shows through anywhere, this bar is broken.
        painter.fillRect(QRect(0, self.height() // 2 - 40, self.width(), 80), self._marker)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="A plain opaque Qt window, to separate the app from the stack.")
    parser.add_argument("--colour", default="#1f232a")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args(argv)

    fmt = QSurfaceFormat.defaultFormat()
    fmt.setAlphaBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv[:1])
    window = OpaqueProbe(args.colour)
    window.resize(1100, 800)
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    print(f"platform: {app.platformName()} | default alphaBufferSize: {QSurfaceFormat.defaultFormat().alphaBufferSize()}")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

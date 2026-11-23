"""Opt-in proof that a `QRhiWidget` renders *inside this app's own window*.

Why this exists: the preview surface (plan-rve-video-preview-widget-2026-11-23) is the first `QRhiWidget` this window
would ever contain, and the knowledge plane records what the wrong kind of extra surface does on this Wayland/NVIDIA
stack ("quality" not assumed: BlinkTable's `inv-2026-10-06-ql-rhi-menu-first.md` - a `Qt.Popup` broke compositing and
"existing QRhiWidgets slow down"). The question "does RHI initialize and present in *this* window" can only be answered
on the owner's live display, so this module gives that display a one-verb experiment: the capture server's `rhi:` verb
creates a probe widget inside the running window, renders a few frames, saves a readback and reports what happened.

Nothing here is product code: the probe is created on demand, removed immediately after, and never installed by startup.
"""

from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QRhiWidget


class _Probe(QRhiWidget):
    """A plain cleared surface: enough to prove initialize + present + readback, no shader needed."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.initialized = False
        self.frames = 0
        self.last_api = None
        # A fixed size keeps the experiment independent of the pane's layout; the widget is deleted right after.
        self.setFixedSize(320, 180)
        self.setWindowFlags(Qt.WindowType.Widget)

    def initialize(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        self.initialized = True
        try:
            self.last_api = str(self.api())
        except Exception:
            self.last_api = None

    def render(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        target = self.renderTarget()
        if target is None:
            return
        command_buffer.beginPass(target, QColor("#2f6f4f"), (1.0, 0))
        command_buffer.endPass()
        self.frames += 1


def probe(window, out_dir: str, *, settle: int = 12, delay_s: float = 0.02) -> dict:
    """Run the experiment in `window`, save a readback PNG into `out_dir`, return a JSON-able report."""
    from PySide6.QtWidgets import QApplication

    failures: list[str] = []
    started = time.time()
    widget = _Probe(window)
    try:
        widget.renderFailed.connect(lambda: failures.append("renderFailed"))
    except Exception:
        pass
    widget.move(40, 40)
    widget.show()
    for _ in range(settle):
        QApplication.processEvents()
        time.sleep(delay_s)

    image = None
    try:
        image = widget.grabFramebuffer()
    except Exception as exc:  # a failed readback is a result, not a crash
        failures.append(f"grabFramebuffer: {type(exc).__name__}: {exc}")

    saved_path = os.path.join(out_dir, f"rhi-probe-{time.strftime('%Y%m%d-%H%M%S')}.png")
    saved = False
    size = None
    if image is not None and not image.isNull():
        saved = bool(image.save(saved_path))
        size = [image.width(), image.height()]
    else:
        saved_path = None

    report = {
        "initialized": widget.initialized,
        "frames": widget.frames,
        "api": widget.last_api,
        "visible": widget.isVisible(),
        "grab_saved": saved,
        "grab_path": saved_path,
        "grab_size": size,
        "failures": failures,
        "seconds": round(time.time() - started, 3),
    }
    widget.hide()
    widget.deleteLater()
    QApplication.processEvents()
    return report

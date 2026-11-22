"""A full repaint after a structural change, on Wayland only.

Measured, not guessed (owner's session, 2026-11-22, `Ctrl+Shift+P` over the queue list):

    chain: _Rows < qt_scrollarea_viewport < OverlayScrollArea < TextListWidget#renderQueueListWidget …
    RENDERED pixel at the cursor: #1f232a          (the list's own plate)
    platform=wayland, SCREEN: n/a                  (a Wayland client cannot read the screen)

The app paints that pixel correctly, the screen shows the previous frame's selection band there, and a window resize
clears it - which is a full surface commit. So the frame the compositor kept is stale, not our painting.

This is the documented remedy on the reference host, not an invention: "Deferred relayouts heal geometry, not pixels …
on Wayland translucent windows the stale frame may never be repainted — force synchronous repaints after the heal"
(BlinkTable `docs/dev/rendering/investigations/csd-scale-relayout.md:253-256`). RVE's window keeps translucent pages and
panels (BlinkTable's case exactly), so the same rule applies.

Scope is deliberately narrow: only on the Wayland platform (X11 and offscreen already repaint correctly - every
screenshot here was clean), and only at the structural moments that leave an area uncovered: a page or sub-tab switch,
a list rebuild, and the render start/finish toggles.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget


def is_wayland() -> bool:
    app = QApplication.instance()
    try:
        return app is not None and app.platformName() == "wayland"
    except Exception:
        return False


def force_full_repaint(widget: QWidget | None, *, reason: str = "") -> bool:
    """Repaint the whole window NOW (synchronously), Wayland only; returns whether it ran.

    `repaint()` and not `update()`: the point is to commit the surface before the compositor presents the frame, and an
    `update()` coalesces into exactly the partial damage that left the stale strip behind.
    """
    if widget is None or not is_wayland():
        return False
    window = widget.window()
    if window is None:
        return False
    try:
        window.repaint()
    except Exception:
        return False
    if reason:
        try:
            from src.Util import log

            log(f"wayland repaint: {reason}")
        except Exception:
            pass
    return True

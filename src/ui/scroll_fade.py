"""Gradient fade at the clipped edges of every scroll area in the window.

A scroll area cuts its content with a hard line, which reads as "the text just ends". Google's panes fade the last few
pixels instead, so the eye knows there is more (owner, 2026-11-22, asking for that look).

The behaviour lives in the toolkit (`OverlayScrollArea.set_edge_fade_enabled`); this is the host's decision, stated once
per window like the type sizes and the wheel policy, instead of at each of the ~20 scroll-area construction sites. The
fade colour comes from the area itself - its surface token when it has one, otherwise the palette Window role, which the
shell pins to `surface.page`.
"""

from __future__ import annotations

from sli_ui_toolkit.ui.widgets.atomic.minimalist_scrollbar import OverlayScrollArea


def apply_edge_fades(root) -> int:
    """Enable the edge fade on every scroll area under ``root``; returns how many were enabled.

    Idempotent: an area that already has it stays as it is, so calling it again (a dialog built later, another window)
    costs nothing.
    """
    enabled = 0
    for area in root.findChildren(OverlayScrollArea):
        if area.edge_fade_enabled():
            continue
        area.set_edge_fade_enabled(True)
        enabled += 1
    return enabled

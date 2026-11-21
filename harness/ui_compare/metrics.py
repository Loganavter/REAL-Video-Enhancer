"""Measuring a widget against itself: absolute rects and the share of its own pixels that differ.

The page table divides by the page (~800 000 px), where a 200x35 button is under 1% and a visibly wrong button reads as
noise. Every measurement here divides by the widget, which is the population the question is about.
"""

from __future__ import annotations

import numpy as np


def abs_rect(node: dict, ox: int = 0, oy: int = 0) -> tuple[int, int, int, int]:
    """A node's rectangle in screen space (dump geometry is parent-local)."""
    x, y, w, h = node.get("geometry", [0, 0, 0, 0])
    return ox + x, oy + y, w, h


def ink_share(a, b) -> float:
    """Share of pixels more than 8 levels off, in % of the compared box (`a`, `b` are BGR crops of equal size)."""
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return 100.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return float((diff > 8).mean()) * 100.0


def box_of(image, rect: tuple[int, int, int, int]):
    """Crop a rect out of an image, or None when it does not fit (scrolled out of view)."""
    x, y, w, h = rect
    height, width = image.shape[0], image.shape[1]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return None
    return image[y:y + h, x:x + w]

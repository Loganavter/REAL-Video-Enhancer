"""The fonts RVE ships with itself, registered once per process.

Why this exists: the Home page's 25 pt display label asks for the family ``Fira Mono``
(``src/ui/typography.py::DISPLAY_PT``, ``src/ui/pages/home_page.py``). A host that has Fira **Code** installed but
not Fira **Mono** answers ``fc-match "Fira Mono"`` with ``NotoSans-Regular``, and Qt then paints the heading in the
wrong face without a word. The owner decided to bundle the face rather than accept the fallback.

``ensure()`` is the one entry point. It is idempotent: the first call loads the bundled TTFs with
``QFontDatabase.addApplicationFont`` and caches the family names Qt reports; every later call returns the same tuple
and registers nothing. A missing or unreadable file is reported through :mod:`logging` and skipped - a broken font is
a cosmetic loss, never a reason for the window not to open.

Paths are resolved next to this module (``src/ui/fonts/``), never from the working directory or an absolute build
path, because a packaged app is not started from its source tree.
"""

from __future__ import annotations

import logging
import os
import threading

#: Every file in ``fonts/`` that carries the face, in the order it should be loaded (regular first).
FONT_FILES: tuple[str, ...] = (
    "FiraMono-Regular.ttf",
    "FiraMono-Medium.ttf",
    "FiraMono-Bold.ttf",
)

#: The directory the bundled files live in, next to this module.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_LOG = logging.getLogger(__name__)

_FAMILIES: tuple[str, ...] | None = None
_LOCK = threading.Lock()


def families() -> tuple[str, ...] | None:
    """The families the bundled files resolved to, or ``None`` when ``ensure()`` has not registered them yet."""
    return _FAMILIES


def ensure() -> tuple[str, ...]:
    """Register the bundled fonts with Qt and return the family names Qt reports.

    Idempotent: the first successful call does the work, later calls return the cached tuple without touching
    ``QFontDatabase``. Calling it before a ``QGuiApplication`` exists is reported and returns ``()`` **without
    caching**, so the real startup (which always has an application by then) can still register the files.
    """
    global _FAMILIES

    with _LOCK:
        if _FAMILIES is not None:
            return _FAMILIES

        from PySide6.QtGui import QFontDatabase, QGuiApplication

        if QGuiApplication.instance() is None:
            _LOG.warning("bundled fonts: no QGuiApplication yet - skipping registration")
            return ()

        found: list[str] = []
        try:
            for name in FONT_FILES:
                path = os.path.join(FONT_DIR, name)
                if not os.path.isfile(path):
                    _LOG.warning("bundled fonts: %s is missing from %s", name, FONT_DIR)
                    continue
                font_id = QFontDatabase.addApplicationFont(path)
                if font_id < 0:
                    _LOG.warning("bundled fonts: Qt refused to load %s", path)
                    continue
                for family in QFontDatabase.applicationFontFamilies(font_id):
                    if family and family not in found:
                        found.append(family)
        except Exception as exc:  # never let a font file cost the application its window
            _LOG.warning("bundled fonts: registration failed: %s", exc)
            return ()

        _FAMILIES = tuple(found)
        return _FAMILIES

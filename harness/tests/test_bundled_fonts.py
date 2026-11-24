"""The app ships Fira Mono, so its 25 pt display face stops falling back to Noto Sans.

Why this file exists (measured 2026-11-24): the Home page's display label is built with
``apply(self.label, DISPLAY_PT, family="Fira Mono")`` (``src/ui/pages/home_page.py:106``), but this host has Fira
**Code** installed and not Fira **Mono**. Qt then resolved ``QFont("Fira Mono")`` to ``Noto Sans`` and painted the
heading in the wrong face without an error. The red state this file was written against is exactly that value:

    QFontInfo(QFont("Fira Mono")).family() == "Noto Sans"

The fix is the bundle, not another literal: ``src/ui/bundled_fonts.py`` registers the three TTFs in ``src/ui/fonts/``
with ``QFontDatabase.addApplicationFont``, and ``src/ui/toolkit_boot.py::ensure()`` calls it before the theme and
before any page is built. The last test reads the resolved font off the real window, because the requested string was
always ``Fira Mono`` - a test on the request would have been green before the fix.

All of it is offscreen; nothing here needs a window.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QFont, QFontDatabase, QFontInfo  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FONT_DIR = ROOT / "src" / "ui" / "fonts"
FAMILY = "Fira Mono"
TTFS = ("FiraMono-Regular.ttf", "FiraMono-Medium.ttf", "FiraMono-Bold.ttf")


def _ensure():
    from src.ui import bundled_fonts

    return bundled_fonts.ensure()


# --- the bundle on disk ---------------------------------------------------------------------------


def test_the_face_ships_with_its_license_next_to_the_loader():
    """The TTFs and the OFL travel together: the license is a condition of redistributing the face."""
    for name in TTFS:
        path = FONT_DIR / name
        assert path.is_file(), f"{path} is missing"
        assert path.stat().st_size > 100_000, f"{path} looks truncated ({path.stat().st_size} bytes)"

    license_path = FONT_DIR / "OFL.txt"
    assert license_path.is_file(), "the SIL OFL must ship next to the fonts"
    text = license_path.read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in text.upper()
    assert "Version 1.1" in text

    readme = (FONT_DIR / "README.md").read_text(encoding="utf-8")
    assert "mozilla/Fira" in readme, "the README must name where the files came from"
    assert "FiraMono-Bold.ttf" in readme and "559,175" in readme, "the README must state the size cost"


# --- registration ---------------------------------------------------------------------------------


def test_ensure_registers_a_family_qt_actually_resolves(qapp):
    """Red before the fix: Qt reported ``Noto Sans`` for the requested family."""
    registered = _ensure()

    assert FAMILY in registered, f"ensure() reported {registered!r}, not {FAMILY!r}"
    assert FAMILY in QFontDatabase.families(), "Qt does not list the bundled family"

    requested = QFont(FAMILY)
    resolved = QFontInfo(requested).family()
    assert resolved.startswith(FAMILY), f"QFont({FAMILY!r}) resolved to {resolved!r}"
    assert "Noto" not in resolved


def test_ensure_is_idempotent(qapp):
    """A second call returns the same families and registers nothing again."""
    first = _ensure()
    count_after_first = len(QFontDatabase.families())

    second = _ensure()
    assert second == first
    assert len(QFontDatabase.families()) == count_after_first, "the second call added a font again"


def test_the_bundled_face_is_bold_capable(qapp):
    """The display heading is styled from the same face; a Regular-only bundle would silently fake bold."""
    _ensure()

    styles = [style.lower() for style in QFontDatabase.styles(FAMILY)]
    assert "bold" in styles, f"no bold style in {styles}"

    bold = QFont(FAMILY)
    bold.setBold(True)
    info = QFontInfo(bold)
    assert info.family().startswith(FAMILY)
    assert info.bold() or info.weight() >= QFont.Weight.DemiBold


def test_a_missing_file_is_reported_and_does_not_raise(qapp, monkeypatch, caplog):
    """A damaged install costs the face, never the window: report the file and carry on."""
    from src.ui import bundled_fonts

    saved = bundled_fonts._FAMILIES
    monkeypatch.setattr(bundled_fonts, "_FAMILIES", None)
    monkeypatch.setattr(bundled_fonts, "FONT_FILES", ("FiraMono-DoesNotExist.ttf",))
    try:
        with caplog.at_level(logging.WARNING, logger=bundled_fonts.__name__):
            got = bundled_fonts.ensure()
        assert got == ()
        assert any("FiraMono-DoesNotExist.ttf" in record.getMessage() for record in caplog.records)
    finally:
        bundled_fonts._FAMILIES = saved


# --- the real window ------------------------------------------------------------------------------


def _build_shell(qapp):
    """The real ``MainWindowShell`` offscreen, with the resource module stubbed (no icon binaries needed)."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from PySide6.QtWidgets import QMainWindow

    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    return shell, window


def test_the_home_display_label_resolves_to_the_bundled_face(qapp):
    """The point of the feature, read off the widget: 25 pt on the Home page is painted in Fira Mono, not Noto Sans."""
    from sli_ui_toolkit.widgets import Text

    _, window = _build_shell(qapp)
    label = window.findChild(Text, "label")
    assert label is not None, "the Home page display label was not found in the real shell"

    font = label.font()
    info = QFontInfo(font)
    assert info.family().startswith(FAMILY), f"the display label resolved to {info.family()!r}"
    assert "Noto" not in info.family()
    assert info.pointSizeF() == pytest.approx(25.0, abs=0.5), f"read the wrong label: {info.pointSizeF()} pt"

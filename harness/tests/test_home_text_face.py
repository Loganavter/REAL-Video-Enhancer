"""The two rich-text panes on the home page read with ONE face.

Owner, 2026-11-28 (screenshot of the home page): the changelog pane on the left did not follow the app's typography at
all, and the system-information pane beside it was a different size. Measured cause: both are `RichTextView`, whose
body is a `Text(mode="document")` sized through `setTextPixelSize`/`setFontPointSize` - `apply()` (the app's one way to
set a font) pins a toolkit `Text`'s OWN size and never reaches that body, so `apply(systemInfoText, GROUP_TITLE_PT)`
was a no-op AND the wrong size if it had worked, while `changeLogText` had no call at all.

Both panes now state the same BODY size through the body's own channel; the family and the interface scale keep coming
from the toolkit's metrics bus (`RichTextView._on_ui_metrics_changed`), so a font or scale change moves neither pane
out of step with the other.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

WINDOW = (1163, 900)


def _settle(qapp, turns: int = 10) -> None:
    for _ in range(turns):
        qapp.processEvents()


@pytest.fixture()
def home(qapp, factor_state):
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*WINDOW)
    window.show()
    _settle(qapp)
    try:
        yield shell, window
    finally:
        window.close()


@pytest.fixture()
def factor_state():
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    from src.ui import i18n

    scale = UiScale.get_instance()
    before_factor, before_language = scale.factor(), i18n.current_language()
    try:
        yield scale
    finally:
        scale.set_factor(before_factor)
        i18n.configure_i18n(before_language)


def _body_fonts(shell):
    page = shell.homePage_page
    return page.changeLogText._body_font(), page.systemInfoText._body_font()


def test_both_panes_read_with_the_same_size_and_family(home):
    shell, _window = home
    changelog, system = _body_fonts(shell)
    assert changelog.pixelSize() == system.pixelSize(), (
        f"the changelog body is {changelog.pixelSize()} px and the system pane {system.pixelSize()} px")
    assert changelog.family() == system.family(), (
        f"the two panes resolve different families: {changelog.family()!r} vs {system.family()!r}")


def test_the_scale_moves_both_panes_together(home, factor_state, qapp):
    """A raised factor re-resolves both bodies from the same bus - neither may drift from the other."""
    shell, window = home
    before = _body_fonts(shell)[0].pixelSize()
    factor_state.set_factor(1.5)
    window.resize(window.width(), window.height())
    _settle(qapp)
    changelog, system = _body_fonts(shell)
    assert changelog.pixelSize() == system.pixelSize(), (changelog.pixelSize(), system.pixelSize())
    assert changelog.pixelSize() > before, (
        f"the body did not follow the scale: {before} px at 1.0 and {changelog.pixelSize()} px at 1.5")

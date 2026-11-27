"""The transport row is its own rounded plate, sized to its controls - not a square band with dead space.

Owner, 2026-11-28: "сделай подложку скругленной и убери снизу лишний пад". Before: the visible backing was the
COLUMN BODY's `dialog.background` (`bottomMenuSubContainer`), a full-height square panel that showed a band under the
row, plus an empty `progressDetail` caption line that reserved a whole line of height while the app was idle. After: no
fill on the column body, the row (`widget_13`) wears the rounded plate itself, it cannot stretch - and the caption line
is GONE, not hidden: the run card already states those numbers (owner, 2026-11-28: "подписи надо убрать было").

The numbers below are measured on the real shell, not chosen: the row's plate radius is the same 10 design px the
video-info card uses, and the row's height is its layout hint (9 + 55 + 9 = 73 design px at 1.0).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# The app and the backend both own a `src` namespace package; the suite's conftest puts the backend first. `ProcessTab`
# needs the APP's `src.constants`, so the app root goes first for this module (the same pattern as
# `test_process_readouts_language.py`).
_ROOT = str(Path(__file__).resolve().parents[2])
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
try:  # pragma: no cover - the branch taken depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtWidgets import QMainWindow, QSizePolicy  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.pages.proc_page import TRANSPORT_PLATE_RADIUS_PX  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _settle(qapp, rounds: int = 3) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def _shell(qapp, *, show: bool = True):
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    UiScale.get_instance().set_factor(1.0)
    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    if show:
        window.resize(1163, 900)
        window.show()
        _settle(qapp)
    else:
        _settle(qapp)
    return shell, window


def test_the_row_wears_the_rounded_plate_and_the_column_body_paints_nothing(qapp):
    """One plate, on the row, with the card radius; the body behind it is transparent."""
    from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

    shell, window = _shell(qapp)
    try:
        page = shell.procPage_page
        row = page.widget_13
        body = page.bottomMenuSubContainer

        assert body.background_token() is None, (
            "the column body must not paint: a square fill under the row was the band the owner asked to remove")
        assert row.background_token() == "dialog.background", "the row is the plate now"
        assert row.radius() == scaled_px(TRANSPORT_PLATE_RADIUS_PX) == scaled_px(10), (
            f"the plate radius must be the design 10 resolved through UiScale, got {row.radius()}")
    finally:
        window.close()


def test_the_row_cannot_stretch_to_swallow_the_columns_leftover_height(qapp):
    """A stretched row is dead padding under a control; the trailing stretch takes that space instead."""
    shell, window = _shell(qapp)
    try:
        row = shell.procPage_page.widget_13
        assert row.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed, (
            "the row's height is its controls' height, not the column's leftover space")
        # The measured box: the layout's 9 px margins around the 55 px designer button.
        assert row.sizeHint().height() <= 90, (
            f"the row hint should hug its 55 px controls, got {row.sizeHint().height()}")
    finally:
        window.close()

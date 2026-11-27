"""The window remembers its size, position and state across restarts.

Owner, 2026-11-28: "надо добавить запоминание размера и состояния окна при перезапуске". The behaviours are
BlinkTable's (`src/utils/geometry.py`), not its code:

* the NORMAL rect is what persists - closing maximized writes only the flag, so the size the user chose is not
  replaced by the screen-sized geometry the maximized window happens to show;
* a saved rect is clamped INTO the screen's available rect, so a window saved on a monitor that is now gone lands on
  a remaining screen instead of off its edge;
* garbage in the store falls back to the design size and says so - a preference must never stop the window opening.

The real `MainWindowShell` is built offscreen over a temp `settings.txt` (the `test_settings_store.py` pattern). All
five behaviour tests fail on the pre-change tree: `Settings` had no `writeWindowGeometry`, `build()` took no
`available` argument, `src/ui/shell/window.py` had no restore/save helpers and nothing was written on close, so no
saved rect could exist at all. The two AST pins additionally catch dropping the wiring while keeping the helpers.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, QSize  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: A screen big enough for any rect these tests save, with a non-zero origin to catch a clamp that forgets `left()`.
SCREEN = QRect(0, 0, 1920, 1080)

#: What `window_default_size()` states at 1.0 (and the two geometry tests never scale): the design box.
DEFAULT_SIZE = QSize(1163, 700)


def _settings(tmp_path):
    """A settings owner over a temp file, exactly the app's store with the machine's paths replaced."""
    from src.ui.settings_store import Settings

    return Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder=str(tmp_path / "Videos"),
    )


def _build(qapp, settings=None, available=None):
    """The real shell in a bare `QMainWindow`, offscreen, with the compiled resource blob stubbed."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    if settings is not None:
        # The app sets this before `MainWindowShell.build`; the shell reads the geometry from the window's owner.
        window.settings = settings
    MainWindowShell().build(window, available=available)
    return window


def _settle(qapp, turns=4):
    for _ in range(turns):
        qapp.processEvents()


# --- restore ------------------------------------------------------------------------------------------------------


def test_a_saved_rect_is_restored_exactly(qapp, tmp_path):
    """The rect the user closed with is the rect the next launch opens with - position included."""
    from src.ui.settings_store import window_rect

    settings = _settings(tmp_path)
    settings.writeWindowGeometry((120, 90, 1300, 820), False)

    window = _build(qapp, settings, available=SCREEN)

    assert window.geometry().getRect() == (120, 90, 1300, 820)
    assert window_rect(settings.settings) == (120, 90, 1300, 820)


def test_an_offscreen_or_oversized_rect_is_clamped_on_screen(qapp, tmp_path):
    """A rect saved on a monitor that is gone must land ON screen: size through `fit_size`, position pulled back in."""
    settings = _settings(tmp_path)
    settings.writeWindowGeometry((4000, 3000, 4000, 3000), False)
    fake_screen = QRect(0, 0, 1600, 900)

    window = _build(qapp, settings, available=fake_screen)

    rect = window.geometry()
    assert fake_screen.contains(rect), f"{rect.getRect()} is not inside the fake screen {fake_screen.getRect()}"
    assert rect.width() <= fake_screen.width() and rect.height() <= fake_screen.height()
    assert rect.getRect() == (0, 0, 1600, 900)


def test_a_rect_left_of_the_primary_screen_is_pulled_onto_it(qapp, tmp_path):
    """Negative coordinates are legitimate on a multi-monitor desktop, but not when that monitor is gone."""
    settings = _settings(tmp_path)
    settings.writeWindowGeometry((-1500, 200, 1200, 800), False)

    window = _build(qapp, settings, available=SCREEN)

    rect = window.geometry()
    assert SCREEN.contains(rect), f"{rect.getRect()} is not inside {SCREEN.getRect()}"
    assert rect.x() == SCREEN.left()


def test_the_maximized_flag_is_applied(qapp, tmp_path):
    """A window closed maximized opens maximized, and its NORMAL rect is still the one that was saved."""
    settings = _settings(tmp_path)
    settings.writeWindowGeometry((120, 90, 1300, 820), True)

    window = _build(qapp, settings, available=SCREEN)
    assert window.isMaximized(), "the saved maximized state was not applied at build time"

    window.show()
    _settle(qapp)
    assert window.isMaximized(), "show() dropped the restored maximized state"
    assert window.normalGeometry().getRect() == (120, 90, 1300, 820), (
        "the saved normal rect must be what unmaximizing returns to")


# --- save on close ------------------------------------------------------------------------------------------------


def test_closing_a_normal_window_writes_its_current_geometry(qapp, tmp_path):
    """Read from the LIVE window at close time, not from a copy taken at build."""
    from src.ui.settings_store import window_rect, window_was_maximized
    from src.ui.shell.window import save_window_geometry

    settings = _settings(tmp_path)
    window = _build(qapp, settings, available=SCREEN)
    window.setGeometry(QRect(70, 80, 1280, 860))

    save_window_geometry(window, window.settings)

    reread = _settings(tmp_path)
    assert window_rect(reread.settings) == (70, 80, 1280, 860)
    assert window_was_maximized(reread.settings) is False


def test_closing_maximized_keeps_the_saved_normal_rect(qapp, tmp_path):
    """The maximized geometry is the screen's; writing it as "normal" is how an app forgets the user's size."""
    from src.ui.settings_store import window_rect, window_was_maximized
    from src.ui.shell.window import save_window_geometry

    settings = _settings(tmp_path)
    settings.writeWindowGeometry((110, 70, 1250, 810), False)
    window = _build(qapp, settings, available=SCREEN)
    window.showMaximized()
    _settle(qapp)
    assert window.isMaximized()
    assert window.geometry().width() != 1250, "the maximized geometry must differ, or this test proves nothing"

    save_window_geometry(window, window.settings)

    reread = _settings(tmp_path)
    assert window_rect(reread.settings) == (110, 70, 1250, 810), "closing maximized replaced the normal rect"
    assert window_was_maximized(reread.settings) is True


# --- garbage ------------------------------------------------------------------------------------------------------


def test_garbage_geometry_falls_back_to_the_default_size_and_says_so(qapp, tmp_path, monkeypatch):
    """A hand-edited/foreign value is a preference to ignore, never an exception at start."""
    from src.ui import settings_store

    messages: list[str] = []
    monkeypatch.setattr(settings_store, "log", messages.append)
    settings = _settings(tmp_path)
    settings.writeSetting("window_x", "abc")
    settings.writeSetting("window_y", "def")
    settings.writeSetting("window_width", "wide")
    settings.writeSetting("window_height", "tall")

    window = _build(qapp, settings, available=QRect(0, 0, 1920, 1080))

    assert (window.width(), window.height()) == (DEFAULT_SIZE.width(), DEFAULT_SIZE.height())
    assert any("window geometry" in message for message in messages), (
        "a garbage rect must be reported, not silently ignored")


# --- wiring -------------------------------------------------------------------------------------------------------


def _class_method(path: Path, class_name: str, method: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next((node for node in ast.walk(tree)
                  if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    assert owner is not None, f"{path.name} has no {class_name} class"
    found = next((node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == method), None)
    assert found is not None, f"{class_name} has no {method}()"
    return found


def _plain_calls(node: ast.AST) -> set[str]:
    return {call.func.id for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}


def test_the_window_build_restores_the_saved_geometry():
    """The helper existing is not the feature; `build()` must call it instead of the unconditional `resize()`."""
    from src.ui.shell import window as shell_window

    assert "restore_window_geometry" in _plain_calls(
        _class_method(ROOT / "src" / "ui" / "shell" / "window.py", "MainWindowShell", "build"))
    assert shell_window.window_default_size() == DEFAULT_SIZE, (
        "the default size must stay the design size (not scaled) - persistence does not change that rule")


def test_the_window_owner_saves_on_close():
    """`closeEvent` is where the user's choice is captured; dropping the call is a silent regression."""
    close = _class_method(ROOT / "REAL-Video-Enhancer.py", "MainWindow", "closeEvent")
    assert "save_window_geometry" in _plain_calls(close), "MainWindow.closeEvent no longer saves the window geometry"

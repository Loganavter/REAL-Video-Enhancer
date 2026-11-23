"""The capture walk is an instrument: it must hand the interface back the way it took it.

Defect this file pins (measured live 2026-11-24 on a freshly booted instance): the control channel's warm-up walk
(`src/ui/capture_server.py`) visits every page and every tab of every page and then writes `ready` - leaving the window
on the LAST page and the LAST tab. On the settings-nested page that was `state:processSettingsContainer ->
currentIndex: 4` (the Logs tab, an empty console) and `state:tabWidget -> currentIndex: 2`, which reads exactly like a
blank page in a screenshot. One click on any tab fixed the view, and the page rendered correctly: the walk, not the
interface, was the thing being photographed.

The invariant, stated once: **when the walk finishes, the page stack's current index and every tab host's current index
that the walk touched are what they were before step 0, and that is true before `ready` exists** - a client that waits
for `ready` can never observe a half-walked interface. A capture ON REQUEST (`capture()` verb) keeps its own behaviour:
it ends on the last requested stem.

The first two tests need no toolkit (a fake window: a `QStackedWidget` of pages, one carrying a tab-host double of the
toolkit's shape); the shell test skips when the toolkit is not importable and is re-run on the shared tree.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QStackedWidget, QTabWidget, QWidget  # noqa: E402

from src.ui import capture_server, page_capture  # noqa: E402


class TopTabHost(QWidget):
    """The toolkit tab host's SHAPE, not the toolkit class: the walk finds a host by class name (`_tab_host`).

    Three tabs is enough to catch "restored to the first tab" and "not restored at all" as different answers.
    """

    def __init__(self, parent: QWidget | None = None, tabs: int = 3) -> None:
        super().__init__(parent)
        self._tabs = tabs
        self._current = 0

    def count(self) -> int:  # noqa: N802 - Qt's spelling
        return self._tabs

    def currentIndex(self) -> int:  # noqa: N802 - Qt's spelling
        return self._current

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt's spelling
        self._current = index


class _FakeWindow(QWidget):
    """Just enough window: a named page stack, and the third page carrying a tab host."""

    def __init__(self, host_kind: str = "toolkit", pages: int = 3) -> None:
        super().__init__()
        self.setObjectName("fakeWindow")
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("stackedWidget")
        for index in range(pages):
            page = QWidget(self.stack)
            page.setObjectName(f"page{index}")
            self.stack.addWidget(page)
        page = self.stack.widget(pages - 1)
        self.host = TopTabHost(page) if host_kind == "toolkit" else QTabWidget(page)
        if host_kind != "toolkit":
            self.host.addTab(QWidget(self.host), "one")
            self.host.addTab(QWidget(self.host), "two")
            self.host.addTab(QWidget(self.host), "three")


def _walk(window) -> list[tuple[str, object]]:
    """Execute every action `steps()` names, in order - the same loop the capture server runs."""
    actions = page_capture.steps(window)
    assert actions, "the fake window has no pages to walk - this test would prove nothing"
    for _name, action in actions:
        action()
    return actions


@pytest.fixture()
def app():
    return QApplication.instance() or QApplication([])


# -- the walk hands back what it took ----------------------------------------------------------------


@pytest.mark.parametrize("host_kind", ["toolkit", "stock"])
def test_the_walk_hands_back_the_page_and_the_tab(app, host_kind):
    """Both discovery paths (`TopTabHost` by name, stock `QTabWidget`) are pinned: only one of them was migrated."""
    window = _FakeWindow(host_kind)
    try:
        window.stack.setCurrentIndex(0)
        window.host.setCurrentIndex(1)          # the "boot" tab: neither the first nor the last
        before = page_capture.walk_state(window)
        assert before == {"page": 0, "tabs": {id(window.host): 1}}, (
            f"walk_state does not describe the boot state it must restore: {before}")

        names = [name for name, _action in _walk(window)]
        assert names[:3] == ["nav0", "nav1", "nav2"], f"the steps() stems changed: {names}"
        assert names[3:] == ["nav2-tab0", "nav2-tab1", "nav2-tab2"], f"the steps() stems changed: {names}"

        # The walk really moved the interface - without this the restore assertion below passes on a no-op walk.
        assert window.stack.currentIndex() == 2, "the walk did not reach the last page"
        assert window.host.currentIndex() == 2, "the walk did not reach the last tab"

        page_capture.restore_walk_state(window, before)
        assert window.stack.currentIndex() == before["page"], (
            "the walk left the window on the last page - a probe that waits for `ready` measures the instrument")
        assert window.host.currentIndex() == before["tabs"][id(window.host)], (
            "the walk left the page on the last tab - this is the empty Logs tab that read as a blank page")
    finally:
        window.close()


def test_restoring_is_a_no_op_for_a_window_with_nothing_recorded(app):
    window = _FakeWindow("toolkit")
    try:
        window.stack.setCurrentIndex(1)
        window.host.setCurrentIndex(2)
        page_capture.restore_walk_state(window, None)
        page_capture.restore_walk_state(window, {"page": None, "tabs": {}})
        assert window.stack.currentIndex() == 1 and window.host.currentIndex() == 2
    finally:
        window.close()


def test_a_page_with_a_single_tab_is_not_part_of_the_walk(app):
    """`steps()` only walks hosts with more than one tab; the saved state follows the same rule."""
    window = _FakeWindow("toolkit")
    try:
        window.host._tabs = 1                   # a host nobody can switch: the walk never touches it
        before = page_capture.walk_state(window)
        assert before["tabs"] == {}, f"a single-tab host cannot be switched, yet it is recorded: {before}"
    finally:
        window.close()


# -- the server restores BEFORE `ready` exists ------------------------------------------------------


def test_the_server_restores_the_walk_before_it_writes_ready(app, tmp_path, monkeypatch):
    """The ordering is the invariant: a client that waits for `ready` must never see the walk's leftovers."""
    window = _FakeWindow("toolkit")
    try:
        window.stack.setCurrentIndex(0)
        window.host.setCurrentIndex(1)
        monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", str(tmp_path))
        monkeypatch.setenv("RVE_UI_CAPTURE_DELAY_MS", "0")
        monkeypatch.setenv("RVE_UI_CAPTURE_STEP_MS", "0")
        capture_server.install(window)

        ready = tmp_path / "ready"
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert ready.exists(), "the warm-up walk never finished - cannot judge the state a client sees"
        assert window.stack.currentIndex() == 0, (
            "`ready` exists while the window still shows the walk's last page")
        assert window.host.currentIndex() == 1, (
            "`ready` exists while the page still shows the walk's last tab (the empty Logs tab)")

        (tmp_path / "quit").write_text("", encoding="utf-8")
        for _ in range(5):
            QTest.qWait(30)
    finally:
        window.close()


# -- the real shell (the supervisor runs this one on the shared tree) -------------------------------


def test_the_boot_page_and_tab_survive_a_walk_in_the_real_shell(app):
    """The measured case: the settings-nested host and the process page's tab host both end where they began."""
    pytest.importorskip("sli_ui_toolkit")
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from PySide6.QtWidgets import QMainWindow

    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    toolkit_boot.ensure()
    toolkit_theme.install(app)
    window = QMainWindow()
    try:
        MainWindowShell().build(window)
        window.resize(1163, 700)
        window.show()
        for _ in range(8):
            app.processEvents()

        stack = window.page_handles.menu.stackedWidget
        stack.setCurrentIndex(0)                # the boot page
        hosts = {}
        for index in range(stack.count()):
            host = page_capture._tab_host(stack.widget(index))
            if host is not None and host.count() > 1:
                host.setCurrentIndex(1 if host.count() > 1 else 0)   # the second tab: not the walk's last one
                hosts[type(host).__name__] = host
        for _ in range(4):
            app.processEvents()
        assert hosts, "the shell built no multi-tab host - this test would prove nothing"

        before = page_capture.walk_state(window)
        _walk(window)
        assert stack.currentIndex() == stack.count() - 1, "the walk did not reach the last page"

        page_capture.restore_walk_state(window, before)
        for _ in range(4):
            app.processEvents()
        assert stack.currentIndex() == before["page"], (
            f"the shell was left on page {stack.currentIndex()} instead of the boot page {before['page']}")
        for host in hosts.values():
            assert host.currentIndex() == before["tabs"][id(host)], (
                f"{type(host).__name__} was left on tab {host.currentIndex()}, "
                f"not the boot tab {before['tabs'][id(host)]}")
    finally:
        window.close()

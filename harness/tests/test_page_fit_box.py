"""A page is laid out for the box it gets, before its first frame.

The shell insets its page stack (`PAGE_INSET_H`/`PAGE_INSET_V`, the number the deleted window stylesheet's
`QStackedWidget{padding:5px 10px}` used to supply), and the fit pass exists so a page is laid out for its real size
even while a `QStackedLayout` in StackOne mode only lays out the page it shows.

The pass used the stack's OUTER rect: 2*PAGE_INSET_H wider and 2*PAGE_INSET_V taller than the box a page actually
gets. Every page was therefore laid out for a box it never has, Qt's own layout pass corrected it afterwards, and the
correction showed as content moving by the inset the first time the page was touched (owner, 2026-11-23: "the content
of every new page jumps up 5-10 px to the right place").

These pins state the invariant instead: after a show and after a resize, every page's geometry IS the stack's
contents rect, and no later event-loop turn changes it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget  # noqa: E402


def _shell():
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    app = QApplication.instance() or QApplication([])
    toolkit_boot.ensure()
    toolkit_theme.install(app)
    window = QMainWindow()
    MainWindowShell().build(window)
    window.resize(1163, 900)
    window.show()
    for _ in range(5):
        app.processEvents()
    return app, window


def _assert_pages_fill_the_contents_rect(window) -> None:
    stack = window.page_handles.menu.stackedWidget
    box = stack.contentsRect()
    assert box != stack.rect(), "the stack must keep its page inset, or this pin measures nothing"
    for index in range(stack.count()):
        page = stack.widget(index)
        assert page.geometry() == box, (
            f"page {index} ({type(page).__name__}) is laid out for {page.geometry().getRect()}, but the box it gets is "
            f"{box.getRect()} - the first frame is laid out for a size the page never has and the correction moves it"
        )


def test_every_page_keeps_the_box_it_gets_after_show():
    _app, window = _shell()
    _assert_pages_fill_the_contents_rect(window)
    window.close()


def test_a_resize_does_not_leave_a_correction_for_the_next_turn():
    """The fit pass must land the final geometry itself; a value that still changes on the next turn is a visible move."""
    app, window = _shell()
    stack = window.page_handles.menu.stackedWidget
    window.resize(980, 760)
    before = [stack.widget(index).geometry().getRect() for index in range(stack.count())]
    for _ in range(6):
        app.processEvents()
    after = [stack.widget(index).geometry().getRect() for index in range(stack.count())]
    assert before == after, "a page's geometry changed after the resize settled - that is the jump the owner sees"
    _assert_pages_fill_the_contents_rect(window)
    window.close()


def test_a_page_switch_leaves_nothing_to_correct_on_the_next_turn():
    """Switching pages is a layout pass of its own: the shown page's geometry is right immediately, but the queued
    LayoutRequest that would lay its contents out for that geometry is not - one frame of the previous box, which reads
    as content moving by the page inset the moment the page is touched. The shell settles the shown page in the same
    turn, deeply, because the scroll area's content widget can already be the final size and then never re-runs its own
    layout (measured on the settings page: a top tab host 1000x766 -> 1008x777 on the next turn).
    """
    app, window = _shell()
    stack = window.page_handles.menu.stackedWidget

    def snapshot(page):
        return [(type(child).__name__, child.objectName(), child.geometry().getRect())
                for child in page.findChildren(QWidget)]

    for index in range(stack.count()):
        stack.setCurrentIndex(index)
        page = stack.widget(index)
        before = snapshot(page)
        for _ in range(4):
            app.processEvents()
        after = snapshot(page)
        assert before == after, (
            f"page {index} ({type(page).__name__}) still changed its contents after the switch settled - that is the "
            "one-frame geometry the owner sees as content moving"
        )
    window.close()

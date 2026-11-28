"""The draggable boundary between the preview and the tab column (owner, 2026-11-28).

*"а и было бы неплохо добавить передвигаемый сплиттер между превью и вкладками"* - a splitter between the preview and
the tabs. Before this wave the boundary was a grid: `gridLayout_3` held `processInfoContainer` (the preview) and
`processContainer` (the column) as two cells, and the column's width was stated as a FIXED number at three sites
(`setFixedWidth(scaled_px(PROCESS_COLUMN_PX))` on the column and on its two inner strips). There was nothing to drag.

What each pin proves, and what the old page answered:

* **the design geometry is the DEFAULT** - at 1.0 the split's children are the shipped 500 px column and the 541 px
  preview, with one `PANE_GAP_PX` handle between them. Old page: `page.preview_splitter` did not exist (the test dies
  on the attribute), and the widths were the fixed statements;
* **a drag moves the boundary** - `moveSplitter` changes both pane sizes and never flips the mode. Old page: nothing
  to drag, so the assertion has no subject;
* **the floors stop a drag** - the preview cannot go under `PANE_FLOOR_PX` and the column cannot go under its own
  content floor; the splitter's children carry those minimums, so the stop is Qt's, not a clamp after the fact. Old
  page: the column was fixed, the preview's floor was only the layout policy's arithmetic;
* **the split is remembered per mode** - a drag writes the mode's key through the settings owner, and the next build
  restores it. Old page: no key existed;
* **a garbage value falls back** - a hand edit or another build's line reads as "never dragged" and is logged. Old
  page: no key to be garbage;
* **each mode keeps its own split** - SIDE -> STACK -> SIDE restores the side split, not the stack one's numbers;
* **one splitter, of the mode's orientation** - horizontal in `side`, vertical in `stack`, and never both. Old page:
  no splitter at all.

Everything is offscreen and builds the real `MainWindowShell`; the compiled resource blob is stubbed because a test
tree has no build step for it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QMainWindow, QSplitter  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.layout_policy import BODY_INSET_PX, PANE_FLOOR_PX, PANE_GAP_PX, SIDE, STACK  # noqa: E402
from src.ui.pages.proc_page import PROCESS_COLUMN_PX  # noqa: E402
from src.ui.settings_store import (  # noqa: E402
    PREVIEW_SPLIT_SIDE_KEY,
    PREVIEW_SPLIT_STACK_KEY,
    Settings,
)
from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px  # noqa: E402

#: The size every other layout gate measures the shell at (`test_page_overflow.py`, `test_page_fit_box.py`).
WINDOW_SIZE = (1163, 900)

#: The settings file every build in this file uses, so a value one test writes cannot reach another.
SETTINGS_NAME = "settings.txt"

#: One write per drag is debounced by `SPLIT_SAVE_DELAY_MS`; wait past it rather than call the flush directly, so the
#: path under test is the shipped one.
SAVE_WAIT_MS = 700


@pytest.fixture()
def process_state():
    """Leave the process-wide factor and language where they were: the singleton outlives a test."""
    from src.ui import i18n
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _settle(qapp, turns: int = 10) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _settings(tmp_path) -> Settings:
    """The real settings owner on a file of this test's own (the app's `Settings()` would touch the user's file)."""
    return Settings(
        settings_file=str(tmp_path / SETTINGS_NAME),
        default_output_folder="/fixed/Videos",
    )


def _shell(qapp, factor: float, *, size=WINDOW_SIZE, settings=None):
    """The real window at `factor`, shown and settled; the page has applied its policy by then."""
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(factor)
    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    if settings is not None:
        # Exactly where the app puts it (`self.settings = Settings()`), and before the shell builds the pages.
        window.settings = settings
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*size)
    window.show()
    _settle(qapp)
    # A show can land its own geometry first; state the measured box once more and let the page's resize hook run.
    window.resize(*size)
    _settle(qapp)
    return shell, window


def _drag(splitter: QSplitter, position: int) -> None:
    """Move the one handle to `position` in the splitter's own coordinates - the same call Qt makes while a drag runs."""
    splitter.moveSplitter(position, 1)


# --- the design geometry is the default ----------------------------------------------------------------------------


def test_the_side_split_is_the_shipped_geometry_at_one_point_zero(qapp, process_state):
    """At 1.0 the split's children ARE today's numbers - stated through the constants, not a captured snapshot.

    Old page: there was no splitter (`page.preview_splitter` did not exist), and the column's width was the fixed
    statement `setFixedWidth(scaled_px(PROCESS_COLUMN_PX))`. The preview was the grid's remainder. This pin makes the
    boundary answer for the same page: the column is `scaled_px(PROCESS_COLUMN_PX)`, the preview is the body's content
    width less that column and one `PANE_GAP_PX` handle.
    """
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        splitter = page.preview_splitter
        assert isinstance(splitter, QSplitter), "the preview/column boundary is not a splitter"
        assert splitter.orientation() == Qt.Orientation.Horizontal
        assert page.layout_mode == SIDE

        column = scaled_px(PROCESS_COLUMN_PX)
        body = page.pageScroll.viewport().width() - BODY_INSET_PX
        preview = body - PANE_GAP_PX - column
        # The COLUMN is the splitter's first child - it keeps the left-hand cell the grid gave it before this wave.
        assert splitter.widget(0) is page.processContainer
        assert splitter.widget(1) is page.processInfoContainer
        assert splitter.handleWidth() == PANE_GAP_PX, (
            f"the handle is {splitter.handleWidth()} px; the body frame's own gap is {PANE_GAP_PX}")
        assert splitter.sizes() == [column, preview], (
            f"the split is {splitter.sizes()}, not the shipped [{column}, {preview}]")
        assert page.processContainer.width() == column
        assert page.processInfoContainer.width() == preview
        # The numbers themselves, so a constant edited by mistake fails the identity the owner ships at 1.0.
        assert (column, preview) == (500, 541), f"the 1.0 split is {column}/{preview}, not 500/541"
        assert page.pageScroll.horizontalScrollBar().maximum() == 0
    finally:
        window.close()


def test_the_stacked_split_is_the_band_above_the_column(qapp, process_state):
    """In `stack` the boundary is vertical and the split is the 16:9 band pane above the column at its content height.

    Old page: the two panes were two grid rows whose heights the layout inferred; there was no splitter, and the
    band's 16:9 box was the only stated number. The default split has to reproduce that geometry exactly - the
    preview pane at its own content height (the band plus its controls) and the column at its content height.
    """
    shell, window = _shell(qapp, 2.0)
    try:
        page = shell.procPage_page
        splitter = page.preview_splitter
        assert page.layout_mode == STACK
        assert splitter.orientation() == Qt.Orientation.Vertical, "a stacked page splits vertically"

        band = page.previewBand
        assert band.height() == round(band.width() * 9 / 16), "the stacked band is still 16:9 of its width"
        # The BAND is the splitter's first child: a vertical split with the column on top would pass a height check
        # and put the page's picture under its tabs.
        assert splitter.widget(0) is page.processInfoContainer
        assert splitter.widget(1) is page.processContainer
        assert splitter.sizes() == [page.processInfoContainer.height(), page.processContainer.height()]
        assert splitter.sizes()[0] >= band.height(), "the preview pane cannot be shorter than the band it carries"
        # The band is above the column, not beside it: the boundary's two panes share the full body width.
        assert page.processInfoContainer.width() == page.processContainer.width() == (
            page.pageScroll.viewport().width() - BODY_INSET_PX)
    finally:
        window.close()


def test_one_splitter_of_the_modes_orientation_is_live_at_a_time(qapp, process_state):
    """One live splitter, of the current mode's axis: a switch must not leave the stale one owning the layout.

    Old page: no splitter at all. With one, the risk is the opposite - Qt cannot re-orient a `QSplitter`, so a switch
    that built a second one without retiring the first would leave two boundaries in the tree.
    """
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        assert page.layout_mode == SIDE
        assert page.preview_splitter.orientation() == Qt.Orientation.Horizontal

        process_state.set_factor(2.0)
        _settle(qapp)
        assert page.layout_mode == STACK
        live = [s for s in page.pageScrollBody.findChildren(QSplitter) if s.isVisible()]
        assert len(live) == 1, f"{len(live)} visible splitters are laid out at once"
        assert live[0] is page.preview_splitter
        assert live[0].orientation() == Qt.Orientation.Vertical
        # The grid holds the splitter and nothing else: the two panes are no longer its items.
        assert page.gridLayout_3.count() == 1
        assert page.gridLayout_3.itemAt(0).widget() is page.preview_splitter

        process_state.set_factor(1.0)
        _settle(qapp)
        live = [s for s in page.pageScrollBody.findChildren(QSplitter) if s.isVisible()]
        assert len(live) == 1 and live[0].orientation() == Qt.Orientation.Horizontal
    finally:
        window.close()


# --- a drag moves the boundary -------------------------------------------------------------------------------------


def test_a_drag_moves_the_boundary_and_never_changes_the_mode(qapp, process_state):
    """A horizontal drag changes both pane widths, keeps the mode, and leaves one handle's width between them.

    Old page: the column was `setFixedWidth`, so there was no handle to move and nothing whose size could follow.
    """
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        splitter = page.preview_splitter
        body = page.pageScroll.viewport().width() - BODY_INSET_PX
        assert page.processContainer.width() == 500

        _drag(splitter, 620)
        _settle(qapp)
        assert page.layout_mode == SIDE, "dragging is a split, not a mode decision"
        assert page.processContainer.width() == 620
        assert page.processInfoContainer.width() == body - PANE_GAP_PX - 620
        assert page.policy_column == 620
        assert page.pageScroll.horizontalScrollBar().maximum() == 0

        _drag(splitter, 520)
        _settle(qapp)
        assert page.processContainer.width() == 520
        assert page.processInfoContainer.width() == body - PANE_GAP_PX - 520
        # The column's two inner strips follow the column: one number, stated at every site.
        assert page.bottomMenuSubContainer.width() == 520
        assert page.renderRow.width() == 520
    finally:
        window.close()


def test_a_drag_past_a_floor_stops_at_the_floor(qapp, process_state):
    """Neither side can be collapsed: the preview stops at `PANE_FLOOR_PX`, the column at its own content floor.

    The floors are the splitter's children's minimum sizes - Qt clamps the drag there, so there is no post-hoc
    correction to get wrong. Old page: the column was fixed (no drag at all), and the preview's floor was only the
    policy's arithmetic in `layout_decision`.
    """
    from src.ui.layout_policy import column_floor_for_factor

    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        splitter = page.preview_splitter
        body = page.pageScroll.viewport().width() - BODY_INSET_PX
        column_floor = column_floor_for_factor(1.0)

        # Far past the preview's floor: the preview keeps 320 px and the column takes the rest.
        _drag(splitter, body)
        _settle(qapp)
        assert page.processInfoContainer.width() == PANE_FLOOR_PX, (
            f"the preview collapsed to {page.processInfoContainer.width()} px, under its {PANE_FLOOR_PX} floor")
        assert page.processContainer.width() == body - PANE_GAP_PX - PANE_FLOOR_PX
        assert page.pageScroll.horizontalScrollBar().maximum() == 0

        # Far past the column's floor: the column keeps its content floor.
        _drag(splitter, -body)
        _settle(qapp)
        assert page.processContainer.width() == column_floor, (
            f"the column collapsed to {page.processContainer.width()} px, under its {column_floor} content floor")
        assert page.processInfoContainer.width() == body - PANE_GAP_PX - column_floor
    finally:
        window.close()


def test_a_stacked_drag_stops_at_the_column_content_floor(qapp, process_state):
    """Vertically the same rule: the preview grows to the column's own content floor, no further.

    Old page: the stacked heights were the layout's inference, and there was no boundary to drag.
    """
    shell, window = _shell(qapp, 2.0)
    try:
        page = shell.procPage_page
        splitter = page.preview_splitter
        column_floor = page.processContainer.minimumSizeHint().height()
        _drag(splitter, 10_000)
        _settle(qapp)
        assert page.processContainer.height() == column_floor, (
            f"the stacked column collapsed to {page.processContainer.height()} px, "
            f"under its {column_floor} px content floor")
        assert page.processInfoContainer.height() >= page.previewBand.height(), (
            "the preview pane ate the band it carries - a 16:9 box that cannot be squeezed")
        # The other side: the preview's floor is the band plus its controls, so it cannot be dragged shut either.
        _drag(splitter, -10_000)
        _settle(qapp)
        assert page.previewBand.height() == round(page.previewBand.width() * 9 / 16)
        assert page.processContainer.height() >= column_floor
    finally:
        window.close()


def test_collapsing_the_band_does_not_reserve_its_height(qapp, process_state):
    """A collapsed band is out of the page's height, not just out of sight.

    The splitter's minimum height is the natural stack; a hidden pane must not keep its share of it, or collapsing in
    side mode would leave the 1.0 page 194 px taller than its window and give it a vertical scroll for a band nobody
    can see. Old page: the band was hidden and the grid row collapsed with it - this pin keeps that behaviour.
    """
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        fitted = page.pageScrollBody.height()
        assert page.pageScroll.verticalScrollBar().maximum() == 0

        page.set_preview_collapsed(True)
        _settle(qapp)
        assert page.pageScrollBody.height() == fitted, (
            f"the collapsed page is {page.pageScrollBody.height()} px tall, not the {fitted} it fits its window in")
        assert page.pageScroll.verticalScrollBar().maximum() == 0
        assert page.processContainer.width() == page.pageScroll.viewport().width() - BODY_INSET_PX

        page.set_preview_collapsed(False)
        _settle(qapp)
        assert page.pageScrollBody.height() == fitted
        assert page.processContainer.width() == scaled_px(PROCESS_COLUMN_PX)
    finally:
        window.close()


# --- persistence: one key per mode ---------------------------------------------------------------------------------


def test_a_saved_side_split_is_restored_on_the_next_build(qapp, process_state, tmp_path):
    """A drag writes `preview_split_side`; the next window opens with that split. Old page: the key did not exist.

    The stored value is DESIGN px, so the same split survives a UI factor change: this pin writes one by hand (500 ->
    620 at 1.0) and asks the freshly built page for it.
    """
    settings = _settings(tmp_path)
    settings.writeSetting(PREVIEW_SPLIT_SIDE_KEY, "620")
    shell, window = _shell(qapp, 1.0, settings=settings)
    try:
        page = shell.procPage_page
        body = page.pageScroll.viewport().width() - BODY_INSET_PX
        assert page.processContainer.width() == 620, (
            "the saved split was not restored: the column is the policy's width, not the user's")
        assert page.processInfoContainer.width() == body - PANE_GAP_PX - 620
        assert page.policy_column == 620
    finally:
        window.close()


def test_a_saved_stack_split_is_restored_on_the_next_build(qapp, process_state, tmp_path):
    """The other key, the other axis: the stacked preview pane keeps the height the user dragged it to.

    Old page: no key, and no way to state the stacked heights at all - they were the layout's inference. The stored
    value is DESIGN px: the page resolves it through `scaled_px` for the factor in force, exactly like the column's
    own 500 design px.
    """
    settings = _settings(tmp_path)
    # The natural split is what the page answers with nothing saved; the saved value is 20 design px taller.
    natural_shell, natural_window = _shell(qapp, 2.0)
    try:
        natural_preview = natural_shell.procPage_page.processInfoContainer.height()
        natural_column = natural_shell.procPage_page.processContainer.height()
    finally:
        natural_window.close()
    natural_design = int(round(natural_preview / UiScale.get_instance().factor()))
    settings.writeSetting(PREVIEW_SPLIT_STACK_KEY, str(natural_design + 20))

    shell, window = _shell(qapp, 2.0, settings=settings)
    try:
        page = shell.procPage_page
        assert page.layout_mode == STACK
        expected = scaled_px(natural_design + 20)
        assert expected > natural_preview, "this pin needs a saved split taller than the default one"
        assert page.processInfoContainer.height() == expected, (
            f"the stacked preview is {page.processInfoContainer.height()} px, not the saved {expected}")
        # The split moved inside the same box: the page is exactly as tall as the natural stack, so what the preview
        # gained the column gave (and the page's scroll range did not change).
        assert page.processInfoContainer.height() + page.processContainer.height() == (
            natural_preview + natural_column)
        assert page.preview_splitter.height() == natural_preview + natural_column + page.preview_splitter.handleWidth()
    finally:
        window.close()


def test_a_garbage_saved_split_falls_back_to_the_design_split(qapp, process_state, tmp_path):
    """A hand-edited value is not an error to raise on: the default split is applied and the value is reported.

    Old page: there was no key, so this test had nothing to corrupt; with the splitter, a bad byte must not cost the
    page its boundary or the window its start.
    """
    settings = _settings(tmp_path)
    settings.writeSetting(PREVIEW_SPLIT_SIDE_KEY, "banana")
    shell, window = _shell(qapp, 1.0, settings=settings)
    try:
        page = shell.procPage_page
        body = page.pageScroll.viewport().width() - BODY_INSET_PX
        assert page.processContainer.width() == PROCESS_COLUMN_PX, (
            "a garbage split must read as the default one")
        assert page.processInfoContainer.width() == body - PANE_GAP_PX - PROCESS_COLUMN_PX
    finally:
        window.close()

    # The reader's half: a value that came from the FILE is blanked once, so the page never sees it again.
    reread = Settings(settings_file=str(tmp_path / SETTINGS_NAME), default_output_folder="/fixed/Videos")
    assert reread.settings[PREVIEW_SPLIT_SIDE_KEY] == "", "the reader kept a value it could not parse"


def test_a_drag_is_stored_once_per_gesture_and_restored_after_a_rebuild(qapp, process_state, tmp_path):
    """The shipped path: drag -> the settings owner's file -> the next window. Old page: no key, no write, no restore."""
    settings = _settings(tmp_path)
    shell, window = _shell(qapp, 1.0, settings=settings)
    try:
        page = shell.procPage_page
        _drag(page.preview_splitter, 640)
        _settle(qapp)
        assert page.processContainer.width() == 640
        assert settings.settings[PREVIEW_SPLIT_SIDE_KEY] == "", (
            "the write must wait for the gesture to settle, not fire on every pixel")
        QTest.qWait(SAVE_WAIT_MS)
        _settle(qapp)
        assert settings.settings[PREVIEW_SPLIT_SIDE_KEY] == "640"
    finally:
        window.close()

    shell, window = _shell(qapp, 1.0, settings=settings)
    try:
        assert shell.procPage_page.processContainer.width() == 640, "the drag did not survive the rebuild"
    finally:
        window.close()


# --- the two modes keep their own splits ---------------------------------------------------------------------------


def test_switching_modes_keeps_each_modes_own_split(qapp, process_state, tmp_path):
    """SIDE -> STACK -> SIDE restores the SIDE split, and the stack drag survives the same round trip.

    Old page: the modes were grid placements of two fixed/flexible cells, so there was no per-mode split to keep or
    lose. This is the pin that catches one key being reused for both axes.
    """
    settings = _settings(tmp_path)
    shell, window = _shell(qapp, 1.0, settings=settings)
    try:
        page = shell.procPage_page
        assert page.layout_mode == SIDE
        _drag(page.preview_splitter, 620)
        _settle(qapp)
        QTest.qWait(SAVE_WAIT_MS)
        _settle(qapp)
        assert settings.settings[PREVIEW_SPLIT_SIDE_KEY] == "620"

        # 2.0 at this window width stacks; the vertical drag is its own choice.
        process_state.set_factor(2.0)
        _settle(qapp)
        assert page.layout_mode == STACK
        assert settings.settings[PREVIEW_SPLIT_STACK_KEY] == "", (
            "a mode switch is not a drag: it must not store a split the user never chose")
        stacked = page.processInfoContainer.height()
        _drag(page.preview_splitter, stacked + 60)
        _settle(qapp)
        QTest.qWait(SAVE_WAIT_MS)
        _settle(qapp)
        assert page.processInfoContainer.height() == stacked + 60
        stack_design = settings.settings[PREVIEW_SPLIT_STACK_KEY]
        assert stack_design and stack_design != settings.settings[PREVIEW_SPLIT_SIDE_KEY], (
            "the stack drag overwrote the side key, or was not stored at all")

        # Back to side: the SIDE split, not the stack one's number.
        process_state.set_factor(1.0)
        _settle(qapp)
        assert page.layout_mode == SIDE
        assert page.processContainer.width() == 620, (
            f"the side split came back as {page.processContainer.width()} px, not 620")

        # And back to stack: the stack split, not the side one.
        process_state.set_factor(2.0)
        _settle(qapp)
        assert page.layout_mode == STACK
        assert abs(page.processInfoContainer.height() - (stacked + 60)) <= 1, (
            f"the stack split came back as {page.processInfoContainer.height()} px, "
            f"not the {stacked + 60} px it was dragged to")
    finally:
        window.close()


def test_the_column_never_clips_its_settings_at_any_split(qapp):
    """A DRAGGED split must not make the General tab overflow its viewport - the "splitter over the content" look.

    Owner, 2026-11-28: "если влево максимально прижать, то снова перекрывает сплиттер". The gate that existed
    (`test_general_tab_fit`) measured the default split only; a column at its floor or at the far right is a different
    measurement, and the settings scroll area hides horizontal overflow (`ScrollBarAlwaysOff`), so an overflow there is
    silently clipped at the splitter's edge.
    """
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        general = page.generalSettings_page
        splitter = page.preview_splitter
        assert splitter is not None

        # A long path and a filled pane: the two things a real session carries and the empty page does not.
        general.outputFileText.setText(
            "/home/jorj/native1440_20s_2x-futsuu-anime(2).pth_24.0fps_2880x2160_(12).mkv")
        general.set_video_info({
            "Resolution:": ("2880x2160", "2880x2160"),
            "FPS:": ("23.976", "23.976"),
            "Frame Count:": ("493", "493"),
            "Encoder:": ("libx264", "libx264"),
            "Container:": ("mkv", "mkv"),
            "Color Space:": ("bt709", "bt709"),
            "Pixel Format:": ("yuv420p10le", "yuv420p10le"),
            "Bit Depth:": ("10", "10"),
            "Is HDR:": ("False", "False"),
        })
        for _ in range(4):
            qapp.processEvents()

        for position in (0, 120, 260, 2000):
            splitter.moveSplitter(position, 1)
            for _ in range(4):
                qapp.processEvents()
            viewport = general.scrollArea_4.viewport().width()
            content = general.scrollArea_4.widget().width()
            assert content <= viewport, (
                f"split position {position}: the settings content is {content} px in a {viewport} px viewport - "
                f"the splitter would cut it (the area hides horizontal overflow)")
    finally:
        window.close()

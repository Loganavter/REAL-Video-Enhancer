"""The pipeline ribbon reflows into rows when the column narrows, instead of eliding five stages into slivers.

Owner report, 2026-11-28 (with a screenshot of the ribbon cut at the plate's edge): *"а нельзя эту вкладку сделать
более динамичной? Ну чтобы как минимум этапы складывались друг над другом, когда по ширине ужимают"*. The ribbon was
ONE `QGridLayout` row - five stage columns with the design's 13/13/9/9/14 shares and four 12 px chevron columns - so a
narrow column gave every stage a sliver and "Upscale" / `2x-Futsuu-Anime...` were cut at the card's edge.

The contracts pinned here, and why each is a behaviour rather than a shape:

* **the shipped column is still one row of five, whole.** At the shipped 485 px card nothing moves: five stages, one
  band, every name painted in full (not elided). A reflow that fires one step too early is a regression of the look the
  owner already approved.
* **a narrow column takes another row.** At 380 px the five stages become 3+2 (`INFO_STAGE_MIN_PX` per stage), every
  stage is still painted, and the ribbon's HEIGHT grows by the second band - the numbers below, not "looks wrapped".
* **each band owns its paint rows.** A stage on the second band has its name under its own rail (grid row 10), never
  under the first band's (row 2). Chevrons separate neighbours on the SAME row only, so a row end has none and the
  first stage of a row starts flush at column 0.
* **shrinking and restoring leaves no residue.** Back at the shipped width it is one row again with the same
  arrangement a fresh build has: no orphan chevron, no stale cell, no dead height left behind by the band that went.
* **a splitter drag is not a per-pixel rebuild.** The rebuild counter moves once per column boundary crossed, not once
  per pixel, and the final arrangement is exactly what a fresh build at that width produces.
* **a language switch repaints TEXT.** The stage order and the number of bands are unchanged, the values stay
  byte-identical - and the grid is not rebuilt (the counter does not move).

The minimum per stage is stated once (`INFO_STAGE_MIN_PX = 82`, measured against the RIBBON HOST and its own
margins - ~428 px usable on the shipped window, where five whole names would need 422 px plus rounding): the host gives five stages 410 px after
four 12 px chevrons, i.e. 82 px each - so the shipped window can never reflow, and the narrow-column case is measured
against the same number (`test_the_column_arithmetic_reflows_at_the_stated_minimum`).

Red-before: on the pre-change tree the ribbon was a single fixed row (`rowCount() == 7`, five stage columns, no
reflow path at all), so every reflow assertion here fails there - the narrow case stays one row (the height does not
move), the second band's rows do not exist, and `_ribbon_rebuilds` / `_reflow_ribbon` do not exist. The idle and
narrow cases were verified by reverting `_reflow_ribbon`/`_rebuild_ribbon` to the one-row placement and watching these
tests fail (`SELF-REVIEW` in `DONE.md`).

Runs offscreen only. What needs a window: the LOOK of a 3+2 band (the chevron between the two stages of a short last
row, the gutter between bands) and the splitter drag on a live instance - both are `LIVE-PROOF WANTED` in `DONE.md`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

_ROOT = str(Path(__file__).resolve().parents[2])
for _name, _attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                      ("pypresence", {"Presence": object})):
    try:  # pragma: no cover - the branch depends on the interpreter
        __import__(_name)
    except ModuleNotFoundError:  # pragma: no cover
        _module = types.ModuleType(_name)
        for _attr, _value in _attrs.items():
            setattr(_module, _attr, _value)
        sys.modules[_name] = _module

from src.ui import i18n  # noqa: E402
from src.ui.pages.general_settings import (  # noqa: E402
    INFO_ARROW_COLUMN_PX,
    INFO_RIBBON_BAND_ROWS,
    INFO_ROW_GAP_PX,
    INFO_STAGE_MIN_PX,
    INFO_STAGE_ORDER,
    GeneralSettingsPage,
    PipelineFacts,
    PipelineStage,
    ribbon_columns_for,
    ribbon_row_spans,
)

#: The owner's own pipeline: a 2x interpolation, a denoiser and a 2x upscale model - the screenshot's ribbon.
CONFIGURED = PipelineFacts(
    stages=(
        PipelineStage("Interpolate", True, "", 2),
        PipelineStage("Decompress", False),
        PipelineStage("Denoise", True, "drunet_color.pth"),
        PipelineStage("Deblur", False),
        PipelineStage("Upscale", True, "2x-Futsuu-Anime(1).pth", 2),
    ),
    backend="tensorrt", preset="fast", reuse=True, by_cel=True,
)

#: The card's width in the shipped column (the owner's screenshot): five stages, one band.
SHIPPED_PX = 485
#: The narrowed column the owner drags the splitter to: three stages on the first band, two on the second.
NARROW_PX = 380

#: Every stage's four painted parts, in paint order.
PARTS = ("rail", "name", "value", "sub")


@pytest.fixture()
def language(qapp):
    """The real i18n module, configured in English and restored afterwards."""
    before = i18n.current_language()
    i18n.configure_i18n("en")
    yield i18n
    i18n.emit_language_changed(before)
    i18n.configure_i18n(before)


def _page(qapp) -> GeneralSettingsPage:
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(1.0)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    page = GeneralSettingsPage()
    root = page.build()
    i18n.install(root)
    page.set_pipeline(CONFIGURED)
    root.resize(900, 1200)
    root.show()
    _settle(qapp)
    page._test_root = root  # keep the root alive for the duration of the test
    return page


def _settle(qapp, rounds: int = 12) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def _at(page, qapp, width: int) -> None:
    """Put the ribbon's card at ``width`` - the pane is the ribbon's column, so this is the splitter's own input.

    The pane is pinned rather than the page root: the General tab's other rows have their own minimum width (measured:
    the scroll content refuses to go below ~504 px at 1.0), so a narrow pane is what a narrow COLUMN gives the card,
    whether the column narrowed because a splitter was dragged or because the factor rose.
    """
    page.videoInfoPane.setFixedWidth(width)
    _settle(qapp)


def _release(page, qapp) -> None:
    """Let the layout size the pane again (the page's own column width)."""
    page.videoInfoPane.setMinimumWidth(0)
    page.videoInfoPane.setMaximumWidth(16777215)
    _settle(qapp)


def _text(widget) -> str:
    """The string that was SET (a toolkit `Text` answers its ELIDED copy from `text()`)."""
    original = getattr(widget, "get_original_text", None)
    return original() if callable(original) else widget.text()


def _cell(page, key: str, part: str = "name") -> tuple[int, int, int, int]:
    """`(row, column, row_span, column_span)` of one painted part in the ribbon's grid."""
    layout = page.infoRibbonLayout
    index = layout.indexOf(page.infoStages[key][part])
    assert index >= 0, f"{key}'s {part} is not in the grid at all"
    return layout.getItemPosition(index)


def _arrow_cells(page) -> dict[int, tuple[int, int, int, int] | None]:
    """`{chevron index: cell}` for the chevrons that are IN the grid (`None` for a hidden one)."""
    layout = page.infoRibbonLayout
    cells = {}
    for index, arrow in enumerate(page.infoArrows):
        position = layout.indexOf(arrow)
        cells[index] = layout.getItemPosition(position) if position >= 0 else None
    return cells


def _bands(page) -> list[list[str]]:
    """The stages grouped by the grid row their NAME sits on, in stage order - the ribbon's rows as painted."""
    by_row: dict[int, list[str]] = {}
    for key, _share in INFO_STAGE_ORDER:
        by_row.setdefault(_cell(page, key)[0], []).append(key)
    return [by_row[row] for row in sorted(by_row)]


def _arrangement(page) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """Where every painted part sits - the whole arrangement, for comparing two builds."""
    return {(key, part): _cell(page, key, part) for key, _share in INFO_STAGE_ORDER for part in PARTS}


# --- 1. the shipped column: one row of five, every name whole -------------------------------------


def test_the_shipped_column_keeps_one_row_of_five_with_every_name_whole(language, qapp):
    """At 485 px the ribbon must not move: one band of five, and no name elided.

    Red before: the pre-change tree placed five stage columns on ONE fixed row of 7 grid rows; there was no
    arrangement to ask, so `_cell`/`_bands` did not exist and the grid's own row count was 7 (a test that could not
    even run). This asserts the shipped look, which a reflow that fires one step early would break.
    """
    page = _page(qapp)
    _at(page, qapp, SHIPPED_PX)
    layout = page.infoRibbonLayout

    assert _bands(page) == [["Interpolate", "Decompress", "Denoise", "Deblur", "Upscale"]], "not one row of five"
    # 8 grid rows for one band: rail, gap, name, gap, value, gap, sub, gutter - the four paint rows every stage has.
    assert layout.rowCount() == INFO_RIBBON_BAND_ROWS, f"one band occupies {INFO_RIBBON_BAND_ROWS} grid rows"
    assert page._ribbon_columns == len(INFO_STAGE_ORDER)

    for index, (key, _share) in enumerate(INFO_STAGE_ORDER):
        row, column, row_span, column_span = _cell(page, key)
        assert (row, column, row_span) == (2, 2 * index, 1), f"{key} is not in its own column on the one band"
        assert column_span == 1, f"{key} spans {column_span} slots - the shipped row gives one per stage"
        name = page.infoStages[key]["name"]
        assert name.width() >= name.sizeHint().width(), (
            f"{key}'s caption needs {name.sizeHint().width()} px and its cell is {name.width()} px")
        assert _text(name) == name.text(), f"{key} is painted elided at the shipped width: {name.text()!r}"

    # The chevrons sit between neighbours on the one row, and only there: four of them, columns 1/3/5/7, row 2.
    assert _arrow_cells(page) == {index: (2, 2 * index + 1, 1, 1) for index in range(4)}


# --- 2. a narrow column: 3+2, every stage painted, the ribbon taller --------------------------------


def test_a_narrow_column_stacks_the_stages_and_grows_the_ribbon(language, qapp):
    """At 380 px the ribbon becomes 3+2 and its height grows by the second band - numbers, not "looks wrapped".

    Red before: the one-row ribbon stayed one row at any width (the five columns just got narrower), so the number of
    bands was 1, the second band's rows did not exist and the height did not move.
    """
    page = _page(qapp)
    _at(page, qapp, SHIPPED_PX)
    shipped_height = page.infoRibbonLayout.geometry().height()
    pane_shipped = page.videoInfoPane.sizeHint().height()

    _at(page, qapp, NARROW_PX)

    assert _bands(page) == [["Interpolate", "Decompress", "Denoise"], ["Deblur", "Upscale"]], "not 3+2"
    assert page._ribbon_columns == 3
    # The second band owns its own paint rows: names at row 2 and row 2 + 8, rails 2 rows above each.
    assert _cell(page, "Deblur")[0] == 2 + INFO_RIBBON_BAND_ROWS
    assert _cell(page, "Deblur", "rail")[0] == INFO_RIBBON_BAND_ROWS
    assert _cell(page, "Upscale", "value")[0] == 4 + INFO_RIBBON_BAND_ROWS

    # Both bands start flush at column 0 and end on the ribbon's right edge (no dead space, no sliver at the left).
    assert _cell(page, "Interpolate")[1] == 0 and _cell(page, "Deblur")[1] == 0
    last_band_end = _cell(page, "Upscale")[1] + _cell(page, "Upscale")[3] - 1
    assert last_band_end == 2 * 3 - 2, f"the last band ends at column {last_band_end}, not on the ribbon's edge"

    # Every stage is still painted, with a caption that fits the cell it was given at this width.
    for key, _share in INFO_STAGE_ORDER:
        name = page.infoStages[key]["name"]
        assert not name.isHidden(), f"{key} was dropped instead of stacked"
        assert name.width() >= name.sizeHint().width(), (
            f"{key}'s caption needs {name.sizeHint().width()} px and its narrowed cell is {name.width()} px")

    # The chevrons separate same-row neighbours only: two on the first band, one on the second, never at a row end.
    assert _arrow_cells(page) == {0: (2, 1, 1, 1), 1: (2, 3, 1, 1), 2: None,
                                  3: (2 + INFO_RIBBON_BAND_ROWS, 1, 1, 1)}

    # The height is the extra band, not a repainted row: the second band's rail, its gaps, its three text rows and
    # the gutter between the bands.
    narrow_height = page.infoRibbonLayout.geometry().height()
    name_height = page.infoStages["Interpolate"]["name"].height()
    expected_growth = name_height + INFO_ROW_GAP_PX
    assert narrow_height - shipped_height >= expected_growth, (
        f"the ribbon grew {narrow_height - shipped_height} px for a second band; its name line alone is "
        f"{name_height} px and the gutter {INFO_ROW_GAP_PX}")
    assert page.videoInfoPane.sizeHint().height() > pane_shipped, "the card did not grow with its bands"


# --- 3. shrink then restore: one row again, no residue --------------------------------------------


def test_shrinking_then_restoring_returns_to_one_row_without_residue(language, qapp):
    """Back at the shipped width the ribbon is the one row a fresh build makes - no orphan chevron, no stale cell.

    Red before: there was nothing to restore to (a reflow did not exist), and the assertion is against a SECOND page
    built at the shipped width, so a reflow that leaves the extra band's chevron hidden-but-in-the-grid or a leftover
    grid cell fails here.
    """
    page = _page(qapp)
    _at(page, qapp, SHIPPED_PX)
    _at(page, qapp, NARROW_PX)
    assert len(_bands(page)) == 2, "the narrow step did not stack, so the restore proves nothing"
    _at(page, qapp, SHIPPED_PX)

    fresh = _page(qapp)
    _at(fresh, qapp, SHIPPED_PX)

    assert len(_bands(page)) == 1 and _bands(page)[0] == [key for key, _share in INFO_STAGE_ORDER]
    assert _arrangement(page) == _arrangement(fresh), "the restored ribbon is not the arrangement a fresh build makes"
    assert _arrow_cells(page) == _arrow_cells(fresh)
    # 5 stages x 4 painted parts + 4 chevrons: nothing else is in the grid (no residue from the second band).
    assert page.infoRibbonLayout.count() == len(INFO_STAGE_ORDER) * len(PARTS) + 4
    assert page.infoRibbonLayout.geometry().height() == fresh.infoRibbonLayout.geometry().height(), (
        "the second band's rows still measure: the restored ribbon is taller than a fresh one-row build")


# --- 4. a drag: one rebuild per column boundary, not one per pixel ---------------------------------


def test_a_splitter_drag_rebuilds_once_per_column_boundary(language, qapp):
    """Drag the splitter from 700 px to 380 px: two column boundaries, two rebuilds, and the same end state.

    Red before: there was no rebuild path to count and no reflow, so the ribbon stayed at five per row however narrow
    the column got - the end state differs from a fresh build at 380 px (which is what fails on the old tree).
    """
    page = _page(qapp)
    _at(page, qapp, 700)
    assert page._ribbon_columns == 5
    before = page._ribbon_rebuilds

    steps = 0
    for width in range(700, NARROW_PX - 1, -4):
        page.videoInfoPane.setFixedWidth(width)
        qapp.processEvents()
        steps += 1
    _settle(qapp)

    crossed = 2   # 5 stages/row -> 4 -> 3 over this drag; the counter must not track the 80 width steps
    assert page._ribbon_rebuilds - before == crossed, (
        f"{page._ribbon_rebuilds - before} rebuilds for {steps} width steps - the drag paid per pixel")
    assert len(_bands(page)) == 2

    # The end state is what a fresh page built at that width makes, not a running total of the drag.
    fresh = _page(qapp)
    _at(fresh, qapp, page.videoInfoPane.width())
    assert _arrangement(page) == _arrangement(fresh)
    assert _arrow_cells(page) == _arrow_cells(fresh)


# --- 5. a language switch repaints the text, not the grid ------------------------------------------


def test_a_language_switch_keeps_the_stage_order_and_the_bands(language, qapp):
    """Switch to Russian at a narrow column: same stages, same bands, same cells - only the words turn.

    Red before: the row structure did not exist (one row at every width, so `_bands` could only answer one row), and
    the ru paint of a narrow column kept five slivers instead of the stack.
    """
    page = _page(qapp)
    _at(page, qapp, NARROW_PX)
    bands_before = _bands(page)
    arrangement_before = _arrangement(page)
    rebuilds_before = page._ribbon_rebuilds
    values_before = {key: _text(page.infoStages[key]["value"]) for key, _share in INFO_STAGE_ORDER}

    language.emit_language_changed("ru")
    _settle(qapp)

    assert _bands(page) == bands_before, "the language switch changed which stages share a row"
    assert _arrangement(page) == arrangement_before, "the language switch moved a stage in the grid"
    assert page._ribbon_rebuilds == rebuilds_before, "the language switch rebuilt the grid instead of the text"
    assert [key for key, _share in INFO_STAGE_ORDER] == list(page.infoStages), "the stage order moved"
    assert _text(page.infoStages["Interpolate"]["name"]) == "Интерполяция"
    assert _text(page.infoStages["Upscale"]["name"]) == "Увеличение"
    # A parameter is DATA: the switch must not touch a model name or a multiplier (`off` is a catalogue word, so the
    # disabled stages are compared by their state only).
    for key in ("Interpolate", "Denoise", "Upscale"):
        assert _text(page.infoStages[key]["value"]) == values_before[key], f"{key}'s parameter was rewritten"
    assert _text(page.infoStages["Upscale"]["sub"]) == "масштаб ×2"

    language.emit_language_changed("en")
    _settle(qapp)
    assert _bands(page) == bands_before and _arrangement(page) == arrangement_before
    assert page._ribbon_rebuilds == rebuilds_before
    assert _text(page.infoStages["Interpolate"]["name"]) == "Interpolate"


# --- 6. the arithmetic that decides a reflow --------------------------------------------------------


def test_the_column_arithmetic_reflows_at_the_stated_minimum():
    """The decision is pure arithmetic, and 82 is the shipped card's own per-stage share - not a tuned guess."""
    arrow = INFO_ARROW_COLUMN_PX
    stages = len(INFO_STAGE_ORDER)

    # The shipped 485 px card: 12 px of inset each side, four 12 px chevrons - five stages get >= the minimum.
    shipped_inner = SHIPPED_PX - 2 * 12
    assert shipped_inner >= stages * INFO_STAGE_MIN_PX + (stages - 1) * arrow
    assert ribbon_columns_for(shipped_inner, INFO_STAGE_MIN_PX, arrow, stages) == 5

    # The narrowed column: 3 per row (3+2), then stacked.
    narrow_inner = NARROW_PX - 2 * 12
    assert ribbon_columns_for(narrow_inner, INFO_STAGE_MIN_PX, arrow, stages) == 3
    assert ribbon_columns_for(3 * INFO_STAGE_MIN_PX + 2 * arrow, INFO_STAGE_MIN_PX, arrow, stages) == 3
    assert ribbon_columns_for(stages * INFO_STAGE_MIN_PX + (stages - 1) * arrow - 1,
                              INFO_STAGE_MIN_PX, arrow, stages) == 4

    # No width resolved yet (the first layout pass) or no room at all: the shipped one-row design, never zero.
    assert ribbon_columns_for(0, INFO_STAGE_MIN_PX, arrow, stages) == stages
    assert ribbon_columns_for(1, INFO_STAGE_MIN_PX, arrow, stages) == 1

    # Spans fill a row exactly: the first stage at column 0, the last ending on the ribbon's right edge.
    for count in range(1, stages + 1):
        spans = ribbon_row_spans(count, count)
        assert spans[0][0] == 0
        assert spans[-1][1] == 2 * count - 2
        assert all(span[1] >= span[0] for span in spans)
    # A short last band lets its tail stage take the leftover slots (nothing sits flush-left with dead space right).
    assert ribbon_row_spans(2, 3) == ((0, 0), (2, 4))
    assert ribbon_row_spans(1, 5) == ((0, 8),)

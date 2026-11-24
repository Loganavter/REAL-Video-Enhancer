"""The settings pages follow the UI scale: rows, controls, gaps and the inline rich text.

Why this file exists (2026-11-24): wave 2 made the process column grow and the page body scroll, but the settings
pages still stated the Designer's pixel geometry. At factor 1.5 a 30 px row held a 12 pt label's 24 px line and clipped
it, and the General tab's information pane stated ``font-size:10pt`` on every span - Qt's rich-text engine treats an
inline size as ABSOLUTE, so that body stayed at 10 pt while every label around it grew (the same body measured the same
block height with a 12 px and a 24 px document font).

The pins below are the wave's contract:

* every design number is resolved at construction and again on ``UiScale.scale_changed``: rows 30 -> 45 -> 60,
  the output path's floor 25 -> 38 -> 50, the value fields 75x30 -> 112x45 -> 150x60, the separator 3 -> 4 -> 6,
  and the design gaps 11/10/9/6 -> 16/15/14/9 -> 22/20/18/12;
* 1.0 is unchanged - the same numbers today, so the look does not move when the feature is off;
* the General tab's rich body is built at DESIGN x factor (11/10 pt -> 16.5/15 pt -> 22/20 pt), which keeps 1.0
  pixel-identical and makes the body grow with everything else.

Everything is offscreen and builds the real pages; the compiled resource blob is stubbed because a test tree has no
build step for it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from sli_ui_toolkit.ui.managers.ui_scale import UiScale  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.general_settings import GeneralSettingsPage  # noqa: E402
from src.ui.pages.tab_2 import Tab2Page  # noqa: E402
from src.ui.pages.tab_3 import Tab3Page  # noqa: E402

FACTORS = (1.0, 1.5, 2.0)

#: A settings row's design height: 30 px, the value the Designer stated as ``(..., 30)``.
ROW_HEIGHT = {1.0: 30, 1.5: 45, 2.0: 60}

#: The output path field's floor: the Designer's ``(0, 25)``.
ROW_MIN_HEIGHT = {1.0: 25, 1.5: 38, 2.0: 50}

#: A value spin box's design width; its height is ``ROW_HEIGHT``.
FIELD_WIDTH = {1.0: 75, 1.5: 112, 2.0: 150}

#: The separator's design height (a painted line, so it stays one pixel at 1.0 and grows with the factor).
SEPARATOR_HEIGHT = {1.0: 3, 1.5: 4, 2.0: 6}

#: The design gaps the pages state: the page body (11), a row (10), a section inset (9), a tight section gap (6).
PAGE_GAP = {1.0: 11, 1.5: 16, 2.0: 22}
ROW_GAP = {1.0: 10, 1.5: 15, 2.0: 20}
SECTION_INSET = {1.0: 9, 1.5: 14, 2.0: 18}
TIGHT_GAP = {1.0: 6, 1.5: 9, 2.0: 12}

#: The rich body's design point sizes, before the factor: the Designer's 11 pt body and 10 pt spans.
INFO_BODY_PT = 11
INFO_TEXT_PT = 10

#: What the nine-line body measures at each factor (the 10 px rich inset plus one line per row: 18 px at 10 pt,
#: 28 at 15 pt, 37 at 20 pt). Measured off the real pane, not derived.
INFO_BODY_HEIGHT = {1.0: 172, 1.5: 262, 2.0: 343}


@pytest.fixture(autouse=True)
def _restore_ui_scale():
    """Leave the process-wide factor where it was: the singleton outlives every later test."""
    scale = UiScale.get_instance()
    previous = scale.factor()
    yield
    scale.set_factor(previous)


@pytest.fixture(autouse=True)
def _settle(qapp):
    """The pages are built on the one QApplication; give the toolkit's deferred passes a turn to run."""
    for _ in range(4):
        qapp.processEvents()


def _prepare(qapp, factor: float):
    from src.ui import toolkit_boot, toolkit_theme

    UiScale.get_instance().set_factor(factor)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)


def _show(qapp, root):
    root.resize(900, 900)
    root.show()
    for _ in range(6):
        qapp.processEvents()


def _general(qapp, factor: float) -> GeneralSettingsPage:
    _prepare(qapp, factor)
    page = GeneralSettingsPage()
    _show(qapp, page.build())
    return page


def _tab2(qapp, factor: float) -> Tab2Page:
    _prepare(qapp, factor)
    page = Tab2Page()
    _show(qapp, page.build())
    return page


def _tab3(qapp, factor: float) -> Tab3Page:
    _prepare(qapp, factor)
    page = Tab3Page()
    _show(qapp, page.build())
    return page


# --- 1.0 identity: the gate the whole wave answers to ------------------------------------------------------------


def test_at_factor_one_the_pages_state_todays_numbers(qapp):
    """1.0 unchanged: 30 -> 30, 25 -> 25, 75x30 unchanged, the gaps and the 10/11 pt body as they are today."""
    general = _general(qapp, 1.0)
    tab2 = _tab2(qapp, 1.0)
    tab3 = _tab3(qapp, 1.0)

    for button in (general.inputFileSelectButton, general.batchSelectButton,
                   general.outputFileSelectButton, general.openOutputFolderButton):
        assert (button.minimumHeight(), button.maximumHeight()) == (30, 30)
    assert (general.inputFileText.minimumHeight(), general.inputFileText.maximumHeight()) == (30, 30)
    assert (general.outputFileText.minimumHeight(), general.outputFileText.maximumHeight()) == (30, 30)
    assert general.backendComboBox.maximumHeight() == 30
    for spin in (general.interpolationMultiplierSpinBox, general.upscaleScaleSpinBox):
        assert (spin.minimumWidth(), spin.maximumWidth()) == (75, 75)
        assert (spin.minimumHeight(), spin.maximumHeight()) == (30, 30)
    assert general.line.height() == 3
    assert general.verticalLayout_32.spacing() == 11
    assert general.verticalLayout_32.contentsMargins().left() == 11
    assert general.verticalLayout_38.contentsMargins().left() == 9
    assert general.verticalLayout_39.spacing() == 6
    assert general.horizontalLayout_50.spacing() == 6

    assert tab2.output_folder_location.minimumHeight() == 25
    assert (tab2.scene_change_detection_threshold.minimumWidth(),
            tab2.scene_change_detection_threshold.minimumHeight()) == (75, 30)
    assert tab2.horizontalLayout_19.spacing() == 10
    assert tab2.horizontalLayout_25.spacing() == 6

    assert tab3.precisionSettingsContainer.spacing() == 10
    assert tab3.precisionSettingsContainer_3.spacing() == 10

    markup = general.inputVideoInfoTextEdit._markup
    assert "font-size:11pt" in markup, "the 1.0 body must keep the Designer's 11 pt"
    assert "font-size:10pt" in markup, "the 1.0 spans must keep the Designer's 10 pt"


# --- the rows and controls ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", FACTORS)
def test_general_settings_rows_and_controls_follow_the_factor(qapp, factor):
    general = _general(qapp, factor)
    row = ROW_HEIGHT[factor]

    for button in (general.inputFileSelectButton, general.batchSelectButton,
                   general.outputFileSelectButton, general.openOutputFolderButton):
        assert (button.minimumHeight(), button.maximumHeight()) == (row, row), (
            f"at factor {factor} {button.objectName()} is {button.minimumHeight()}x{button.maximumHeight()} px, "
            f"not {row} px tall: the row keeps the Designer's absolute 30")
    for edit in (general.inputFileText, general.outputFileText):
        assert edit.minimumHeight() == row and edit.maximumHeight() == row
    for combo in (general.backendComboBox, general.interpolateModelComboBox, general.decompressModelComboBox,
                  general.denoiseModelComboBox, general.deblurModelComboBox, general.upscaleModelComboBox):
        assert combo.maximumHeight() == row, (
            f"at factor {factor} {combo.objectName()} caps its height at {combo.maximumHeight()}, not {row}")

    for spin in (general.interpolationMultiplierSpinBox, general.upscaleScaleSpinBox):
        assert (spin.minimumWidth(), spin.maximumWidth()) == (FIELD_WIDTH[factor], FIELD_WIDTH[factor])
        assert (spin.minimumHeight(), spin.maximumHeight()) == (row, row)
    assert general.line.height() == SEPARATOR_HEIGHT[factor]


@pytest.mark.parametrize("factor", FACTORS)
def test_general_settings_gaps_follow_the_factor(qapp, factor):
    general = _general(qapp, factor)

    assert general.verticalLayout_32.spacing() == PAGE_GAP[factor]
    assert general.verticalLayout_32.contentsMargins().left() == PAGE_GAP[factor]
    assert general.verticalLayout_38.contentsMargins().left() == SECTION_INSET[factor]
    assert general.horizontalLayout_60.contentsMargins().left() == SECTION_INSET[factor]
    assert general.verticalLayout_39.spacing() == TIGHT_GAP[factor]
    assert general.verticalLayout_39.contentsMargins().left() == SECTION_INSET[factor]
    assert general.horizontalLayout_50.spacing() == TIGHT_GAP[factor]


@pytest.mark.parametrize("factor", FACTORS)
def test_tab_2_rows_and_gaps_follow_the_factor(qapp, factor):
    tab2 = _tab2(qapp, factor)

    assert tab2.output_folder_location.minimumHeight() == ROW_MIN_HEIGHT[factor]
    assert (tab2.scene_change_detection_threshold.minimumWidth(),
            tab2.scene_change_detection_threshold.minimumHeight()) == (FIELD_WIDTH[factor], ROW_HEIGHT[factor])
    for layout in (tab2.horizontalLayout_19, tab2.horizontalLayout_17, tab2.horizontalLayout_27,
                   tab2.horizontalLayout_14):
        assert layout.spacing() == ROW_GAP[factor]
    assert tab2.horizontalLayout_25.spacing() == TIGHT_GAP[factor]


@pytest.mark.parametrize("factor", FACTORS)
def test_tab_3_gaps_follow_the_factor(qapp, factor):
    tab3 = _tab3(qapp, factor)

    for layout in (tab3.precisionSettingsContainer, tab3.tensorRTOptLevelSettingsContainer,
                   tab3.horizontalLayout_67, tab3.tensorRTOptLevelSettingsContainer_3,
                   tab3.precisionSettingsContainer_2, tab3.precisionSettingsContainer_4,
                   tab3.precisionSettingsContainer_3):
        assert layout.spacing() == ROW_GAP[factor]


# --- the rich-text body ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", FACTORS)
def test_the_info_body_is_built_at_design_times_factor(qapp, factor):
    """An inline ``font-size`` is absolute, so the page must state the scaled point size itself."""
    markup = _general(qapp, factor).inputVideoInfoTextEdit._markup

    assert f"font-size:{INFO_TEXT_PT * factor:g}pt" in markup, (
        f"at factor {factor} the spans must state {INFO_TEXT_PT * factor:g}pt; an unscaled 10 pt span is absolute "
        "and keeps the body at the design size")
    assert f"font-size:{INFO_BODY_PT * factor:g}pt" in markup


@pytest.mark.parametrize("factor", FACTORS)
def test_the_info_body_grows_with_the_factor(qapp, factor):
    """The nine-line body's rendered height tracks the factor - the pin is the ink, not the markup string."""
    pane = _general(qapp, factor).inputVideoInfoTextEdit
    height = pane.label.height()

    assert height == INFO_BODY_HEIGHT[factor], (
        f"at factor {factor} the nine-line body measures {height} px, not {INFO_BODY_HEIGHT[factor]}: "
        "the inline font-size must follow the factor")


# --- a live change -----------------------------------------------------------------------------------------------


def test_a_live_factor_change_reaches_every_page(qapp):
    """The capture channel's ``scale:<factor>`` moves the running pages - no restart, no rebuild."""
    general = _general(qapp, 1.0)
    tab2 = _tab2(qapp, 1.0)
    tab3 = _tab3(qapp, 1.0)
    assert general.inputFileText.maximumHeight() == 30
    assert general.line.height() == 3
    assert general.verticalLayout_32.spacing() == 11
    assert tab2.scene_change_detection_threshold.minimumWidth() == 75
    assert tab3.precisionSettingsContainer.spacing() == 10
    assert "font-size:10pt" in general.inputVideoInfoTextEdit._markup

    UiScale.get_instance().set_factor(1.5)
    for _ in range(6):
        qapp.processEvents()

    assert general.inputFileText.maximumHeight() == 45
    assert general.interpolationMultiplierSpinBox.minimumWidth() == 112
    assert general.line.height() == 4
    assert general.verticalLayout_32.spacing() == 16
    assert tab2.scene_change_detection_threshold.minimumWidth() == 112
    assert tab2.scene_change_detection_threshold.minimumHeight() == 45
    assert tab3.precisionSettingsContainer.spacing() == 15
    assert "font-size:15pt" in general.inputVideoInfoTextEdit._markup
    assert general.inputVideoInfoTextEdit.label.height() == 262

    UiScale.get_instance().set_factor(2.0)
    for _ in range(6):
        qapp.processEvents()

    assert general.inputFileText.maximumHeight() == 60
    assert general.interpolationMultiplierSpinBox.minimumWidth() == 150
    assert general.line.height() == 6
    assert general.verticalLayout_32.spacing() == 22
    assert tab2.scene_change_detection_threshold.minimumWidth() == 150
    assert tab3.precisionSettingsContainer.spacing() == 20
    assert "font-size:20pt" in general.inputVideoInfoTextEdit._markup
    assert general.inputVideoInfoTextEdit.label.height() == 343

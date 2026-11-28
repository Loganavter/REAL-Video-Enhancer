"""The settings pages follow the UI scale: rows, controls, gaps and the video-info pane's blocks.

Why this file exists (2026-11-24): wave 2 made the process column grow and the page body scroll, but the settings
pages still stated the Designer's pixel geometry. At factor 1.5 a 30 px row held a 12 pt label's 24 px line and clipped
it, and the General tab's information pane stated ``font-size:10pt`` on every span - Qt's rich-text engine treats an
inline size as ABSOLUTE, so that body stayed at 10 pt while every label around it grew.

The pins below are the wave's contract:

* every design number is resolved at construction and again on ``UiScale.scale_changed``: rows 33 -> 50 -> 66,
  the output path's floor 25 -> 38 -> 50, the value fields 75x33 -> 112x50 -> 150x66, the separator 3 -> 4 -> 6,
  and the design gaps 11/10/9/6 -> 16/15/14/9 -> 22/20/18/12. The row is the toolkit's combobox height (33, not the
  Designer's 30) since 2026-11-28: a row shorter than a combo squashed six fields below their own minimum;
* 1.0 states those numbers today, so nothing else moves when the feature is off;
* the video-info pane's toolkit labels keep design pixel sizes (the toolkit scales them once) and its hairlines/insets
  are stated through ``scaled_px`` at apply time.

The pane used to be one `RichTextView` document whose ``_markup``/``label.height()`` this file pinned. Variant A
replaced it with toolkit `Text`/`SurfacePanel`/`ProgressBar` blocks (`harness/tests/test_general_settings_info.py`
owns their content contract), so the rich-body pins here became the block-scale pins at the bottom of the file.

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

#: A settings row's design height: the toolkit's combobox height (33 px), which the Designer's 30 could not hold.
ROW_HEIGHT = {1.0: 33, 1.5: 50, 2.0: 66}

#: The output path field's floor: the Designer's ``(0, 25)``.
ROW_MIN_HEIGHT = {1.0: 25, 1.5: 38, 2.0: 50}

#: A value spin box's design width; its height is ``ROW_HEIGHT`` on the General tab.
FIELD_WIDTH = {1.0: 75, 1.5: 112, 2.0: 150}

#: `tab_2`'s scene-change threshold keeps its own value-field row: the Designer's 75x30. Only the General tab's row
#: carries the toolkit's combobox height (see ``ROW_HEIGHT``).
TAB2_THRESHOLD_ROW = {1.0: 30, 1.5: 45, 2.0: 60}

#: The separator's design height (a painted line, so it stays one pixel at 1.0 and grows with the factor).
SEPARATOR_HEIGHT = {1.0: 3, 1.5: 4, 2.0: 6}

#: The design gaps the pages state: the page body (11), a row (10), a section inset (9), a tight section gap (6).
PAGE_GAP = {1.0: 11, 1.5: 16, 2.0: 22}
ROW_GAP = {1.0: 10, 1.5: 15, 2.0: 20}
SECTION_INSET = {1.0: 9, 1.5: 14, 2.0: 18}
TIGHT_GAP = {1.0: 6, 1.5: 9, 2.0: 12}

#: The video-info pane's design geometry, before the factor: its 1 px hairline and its 12 px card inset.
INFO_HAIRLINE_PX = 1
INFO_CARD_INSET_PX = 12


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
    """1.0 states the row at the toolkit's combobox height: 33 -> 33, 25 -> 25, 75x33, the gaps and the body as today."""
    general = _general(qapp, 1.0)
    tab2 = _tab2(qapp, 1.0)
    tab3 = _tab3(qapp, 1.0)

    for button in (general.inputFileSelectButton, general.batchSelectButton,
                   general.outputFileSelectButton, general.openOutputFolderButton):
        assert (button.minimumHeight(), button.maximumHeight()) == (33, 33)
    assert (general.inputFileText.minimumHeight(), general.inputFileText.maximumHeight()) == (33, 33)
    assert (general.outputFileText.minimumHeight(), general.outputFileText.maximumHeight()) == (33, 33)
    assert (general.backendComboBox.minimumHeight(), general.backendComboBox.maximumHeight()) == (33, 33)
    for spin in (general.interpolationMultiplierSpinBox, general.upscaleScaleSpinBox):
        assert (spin.minimumWidth(), spin.maximumWidth()) == (75, 75)
        assert (spin.minimumHeight(), spin.maximumHeight()) == (33, 33)
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

    # The video-info pane's 1.0 geometry: a 1 px hairline and the card's 12 px inset, resolved design px.
    assert general.infoHairlines[0].height() == INFO_HAIRLINE_PX
    assert general.infoCellLayouts[0].contentsMargins().left() == INFO_CARD_INSET_PX


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
        assert (combo.minimumHeight(), combo.maximumHeight()) == (row, row), (
            f"at factor {factor} {combo.objectName()} states {combo.minimumHeight()}..{combo.maximumHeight()}, "
            f"not the row's {row}")

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
            tab2.scene_change_detection_threshold.minimumHeight()) == (FIELD_WIDTH[factor], TAB2_THRESHOLD_ROW[factor])
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


# --- the video-info pane's blocks --------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", FACTORS)
def test_the_info_pane_blocks_follow_the_factor(qapp, factor):
    """The pane's toolkit labels scale once (design px -> the toolkit's resolution) and its hairlines/insets scale."""
    from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

    from src.ui.typography import CAPTION_PT, GROUP_TITLE_PT, label_px

    general = _general(qapp, factor)

    assert general.infoSizeIn.font().pixelSize() == scaled_px(label_px(GROUP_TITLE_PT)), (
        f"at factor {factor} the headline is {general.infoSizeIn.font().pixelSize()} px, not the design title size "
        "resolved once by the toolkit")
    assert general.infoFactValues["Encoder:"].font().pixelSize() == scaled_px(label_px(CAPTION_PT))
    assert general.infoHairlines[0].height() == max(INFO_HAIRLINE_PX, scaled_px(INFO_HAIRLINE_PX))
    assert general.infoCellLayouts[0].contentsMargins().left() == scaled_px(INFO_CARD_INSET_PX)


# --- a live change -----------------------------------------------------------------------------------------------


def test_a_live_factor_change_reaches_every_page(qapp):
    """The capture channel's ``scale:<factor>`` moves the running pages - no restart, no rebuild."""
    general = _general(qapp, 1.0)
    tab2 = _tab2(qapp, 1.0)
    tab3 = _tab3(qapp, 1.0)
    assert general.inputFileText.maximumHeight() == 33
    assert general.line.height() == 3
    assert general.verticalLayout_32.spacing() == 11
    assert tab2.scene_change_detection_threshold.minimumWidth() == 75
    assert tab3.precisionSettingsContainer.spacing() == 10
    assert general.infoHairlines[0].height() == 1

    UiScale.get_instance().set_factor(1.5)
    for _ in range(6):
        qapp.processEvents()

    assert general.inputFileText.maximumHeight() == 50
    assert general.interpolationMultiplierSpinBox.minimumWidth() == 112
    assert general.line.height() == 4
    assert general.verticalLayout_32.spacing() == 16
    assert tab2.scene_change_detection_threshold.minimumWidth() == 112
    assert tab2.scene_change_detection_threshold.minimumHeight() == 45
    assert tab3.precisionSettingsContainer.spacing() == 15
    assert general.infoHairlines[0].height() == 2
    assert general.infoCellLayouts[0].contentsMargins().left() == 18

    UiScale.get_instance().set_factor(2.0)
    for _ in range(6):
        qapp.processEvents()

    assert general.inputFileText.maximumHeight() == 66
    assert general.interpolationMultiplierSpinBox.minimumWidth() == 150
    assert general.line.height() == 6
    assert general.verticalLayout_32.spacing() == 22
    assert tab2.scene_change_detection_threshold.minimumWidth() == 150
    assert tab3.precisionSettingsContainer.spacing() == 20
    assert general.infoHairlines[0].height() == 2
    assert general.infoCellLayouts[0].contentsMargins().left() == 24

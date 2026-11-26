"""The General page's video-info pane: variant A (transition headline, facts grid, run card).

The pane used to be one `RichTextView` document. It is now three toolkit blocks in the page's own column, and the
contract that matters did not change with the shape:

* `GeneralSettingsPage.set_video_info(values)` fills it from the nine `INFO_LABELS` keys, `None` restores the
  nine-row placeholder;
* `video_info_text()` is what the capture probe reads (`FPS: 24.0 -> 24.0` per line, `""` when empty), so the probe
  stops depending on a control;
* a language switch rebuilds the block from the STORED values - names follow the catalogue, values stay byte-identical
  (the regression the old line-rewriting relabel guarded);
* the pane's height is its content's: no 190 px floor, and a filled pane is taller than the empty placeholder.

This module supersedes `test_info_pane_language.py` and `test_info_pane_fit.py`: both pinned the internals of the
removed `RichTextView` (`._markup`, `.label`, `._scroll`) and had nothing left to measure once the widget was gone.

Runs offscreen only. The look at 1.5/2.0 and the live language switch on a real window are `LIVE-PROOF WANTED` in
`DONE.md`.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui import i18n  # noqa: E402
from src.ui.pages.general_settings import INFO_LABELS, GeneralSettingsPage  # noqa: E402

#: A loaded clip in the app's own shape, keyed by the catalogue names the page lists. The values are formatted by
#: `MainWindow.updateVideoGUIText` (input, output); the pane treats them as opaque strings.
VALUES: dict[str, tuple[str, str]] = {
    "FPS:": ("24.0", "48.0"),
    "Resolution:": ("1440x1080", "2880x2160"),
    "Frame Count:": ("493", "986"),
    "Encoder:": ("hevc", "libx264"),
    "Container:": ("mkv", "mkv"),
    "Color Space:": ("bt709", "bt709"),
    "Pixel Format:": ("yuv420p10le", "yuv420p10le"),
    "Is HDR:": ("False", "False"),
    "Bit Depth:": ("10 bit", "10 bit"),
}

#: The facts grid's six names in Russian (the catalogue's own lines, minus the trailing colon the cells drop).
RU_FACT_NAMES = {
    "Encoder:": "Кодировщик",
    "Container:": "Контейнер",
    "Color Space:": "Цветовое пространство",
    "Pixel Format:": "Формат пикселей",
    "Bit Depth:": "Глубина цвета",
    "Is HDR:": "Поддержка HDR",
}


@pytest.fixture()
def language(qapp):
    """The real i18n module, configured in English and restored afterwards."""
    before = i18n.current_language()
    i18n.configure_i18n("en")
    yield i18n
    i18n.emit_language_changed(before)
    i18n.configure_i18n(before)


def _page(qapp) -> GeneralSettingsPage:
    """A real General Settings page, shown at factor 1.0 with the translator installed on its root."""
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(1.0)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    page = GeneralSettingsPage()
    root = page.build()
    i18n.install(root)
    root.resize(900, 1200)
    root.show()
    _settle(qapp)
    page._test_root = root  # keep the root alive for the duration of the test
    return page


def _settle(qapp, rounds: int = 8) -> None:
    for _ in range(rounds):
        qapp.processEvents()


# --- 1. the values, and the placeholder they replace ----------------------------------------------


def test_set_video_info_fills_every_block_and_none_restores_the_placeholder(language, qapp):
    page = _page(qapp)
    assert page.infoPlaceholder.isVisible() and not page.infoFilled.isVisible()

    page.set_video_info(VALUES)
    _settle(qapp)

    assert page.infoFilled.isVisible() and not page.infoPlaceholder.isVisible()
    # transition headline: both sizes, each with its own muted fps/frames line, and the upscale tag
    assert page.infoSizeIn.text() == "1440x1080"
    assert page.infoSizeOut.text() == "2880x2160"
    assert "24.0" in page.infoMetaIn.text() and "493" in page.infoMetaIn.text()
    assert "48.0" in page.infoMetaOut.text() and "986" in page.infoMetaOut.text()
    assert page.infoTag.text() == "2x upscale"
    assert page.infoTagPlate.isVisible()
    # facts grid: the OUTPUT value of each row
    assert page.infoFactValues["Encoder:"].text() == "libx264"
    assert page.infoFactValues["Container:"].text() == "mkv"
    assert page.infoFactValues["Color Space:"].text() == "bt709"
    assert page.infoFactValues["Pixel Format:"].text() == "yuv420p10le"
    assert page.infoFactValues["Bit Depth:"].text() == "10 bit"
    assert page.infoFactValues["Is HDR:"].text() == "False"
    # run card: the frame counter, the bar's range, the footer
    assert page.infoFrameLine.text() == "frame 0 / 986"
    assert (page.infoProgress.minimum(), page.infoProgress.maximum()) == (0, 986)
    assert page.infoProgress.value() == 0
    assert "fps" in page.infoFps.text()
    assert "elapsed" in page.infoElapsed.text()

    page.set_video_info(None)
    _settle(qapp)
    assert page.infoPlaceholder.isVisible() and not page.infoFilled.isVisible()
    assert [widget.text() for widget in page.infoPlaceholderLabels] == list(INFO_LABELS)


# --- 2. the text the capture probe reads ----------------------------------------------------------


def test_video_info_text_round_trips_and_is_empty_for_the_placeholder(language, qapp):
    page = _page(qapp)
    assert page.video_info_text() == ""

    page.set_video_info(VALUES)
    lines = page.video_info_text().split("\n")
    assert len(lines) == len(INFO_LABELS) == 9
    for key, line in zip(INFO_LABELS, lines):
        value_in, value_out = VALUES[key]
        assert line == f"{key} {value_in} -> {value_out}", line
    assert lines[0] == "FPS: 24.0 -> 48.0"
    assert lines[1] == "Resolution: 1440x1080 -> 2880x2160"

    page.set_video_info(None)
    assert page.video_info_text() == ""


# --- 3. a language switch: names turn, values do not ----------------------------------------------


def test_a_language_switch_changes_row_names_but_keeps_values(language, qapp):
    page = _page(qapp)
    page.set_video_info(VALUES)
    _settle(qapp)
    values_before = {key: widget.text() for key, widget in page.infoFactValues.items()}
    sizes_before = (page.infoSizeIn.text(), page.infoSizeOut.text())
    probe_before = page.video_info_text()

    language.emit_language_changed("ru")
    _settle(qapp)

    for key, russian in RU_FACT_NAMES.items():
        assert page.infoFactLabels[key].text() == russian
    assert {key: widget.text() for key, widget in page.infoFactValues.items()} == values_before
    assert (page.infoSizeIn.text(), page.infoSizeOut.text()) == sizes_before
    # the probe's plain text is language-independent: it is a value record, not a painted line
    assert page.video_info_text() == probe_before

    language.emit_language_changed("en")
    _settle(qapp)
    assert page.infoFactLabels["Encoder:"].text() == "Encoder"
    assert page.infoFactValues["Encoder:"].text() == "libx264"

    # the placeholder follows the catalogue too
    page.set_video_info(None)
    language.emit_language_changed("ru")
    _settle(qapp)
    assert page.infoPlaceholderLabels[0].text() == "К/с:"
    assert page.infoPlaceholderLabels[3].text() == "Кодировщик:"


# --- 4. the height is the content's ---------------------------------------------------------------


def test_the_pane_height_is_its_content(language, qapp):
    page = _page(qapp)
    empty_height = page.videoInfoPane.height()
    empty_hint = page.videoInfoPane.sizeHint().height()
    assert empty_height == empty_hint, (
        f"the empty pane is {empty_height} px against a {empty_hint} px hint - something pins its height")

    page.set_video_info(VALUES)
    _settle(qapp)
    filled_height = page.videoInfoPane.height()
    filled_hint = page.videoInfoPane.sizeHint().height()

    assert filled_height == filled_hint, (
        f"the filled pane is {filled_height} px against a {filled_hint} px hint")
    assert filled_height > empty_height, (
        f"the filled pane ({filled_height} px) is not taller than the placeholder ({empty_height} px)")
    assert empty_height != 190 and filled_height != 190, (
        "the retired 190 px floor is back: the pane is always that tall whatever its content")

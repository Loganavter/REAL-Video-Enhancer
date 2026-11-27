"""The settings tabs' tooltips follow a live language switch, on the same page objects.

The measured defect (2026-11-28): the process page's preview chrome and the General tab had their `setToolTip(...)`
calls moved into `retranslate()` in the previous slice, but the five settings tabs still built every tooltip once,
inside `build()`. After a switch the labels turned Russian while the tooltip beside them stayed English - a user read
two languages in one row.

Before this slice, measured on the real built window (`harness/tests` offscreen, `QT_QPA_PLATFORM=offscreen`):
20 catalogue-document tooltips were still the language the page was built in. Two plain-text tooltips already
followed (the encoder command's help, rebuilt through `_encoder_command_tooltip.set_help`, and the toolkit combo's
own elision tooltip, which mirrors the current item's text), so the honest remainder was 20 of 20, and the tab total
is 22 of 22 after the fix.

The mapping below is the guard the task asked for: each page names every tooltip it carries and the English title it
must show, so a future page that adds a tooltip in `build()` (or renames one) fails here with the widget named,
instead of silently staying English. The count is explicit for the same reason.

Runs offscreen only: opening a tooltip on a real pointer and switching while it is open is `LIVE-PROOF WANTED` in
`DONE.md`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

_HERE = Path(__file__).resolve()

#: The catalogue-document tooltip on each settings page, by the widget's object name, with the English title it
#: carries. 4 (Output Settings) + 7 (Render Settings) + 1 (RVE Settings) on the settings page, 4 (Encoder Options) +
#: 4 (Advanced) on the process page = 20 documents; plus the two plain-text tooltips asserted below = 22 tooltips.
CATALOGUE_TOOLTIPS = {
    "tab_2": {
        "scene_change_detection_enabled": "Scene Change Detection Enabled:",
        "scene_change_detection_method": "Scene Change Detection Method",
        "scene_change_detection_threshold": "Scene Change Detection Threshold",
        "auto_border_cropping": "Audo Border Cropping:",
    },
    "tab_3": {
        "precision": "Precision:",
        "tensorrt_optimization_level": "TensorRT Optimization Level:",
        "auto_hdr_mode": "Auto HDR Mode",
        "dynamic_tensorrt_engine": "TensorRT Dynamic Engine:",
        "uhd_mode": "Saves VRAM and speeds up inference",
        "ncnn_gpu_id": "NCNN GPU ID -",
        "pytorch_gpu_id": "PyTorch GPU ID -",
    },
    "tab_4": {
        "discord_rich_presence": "Show REAL Video Enhancer in discord.",
    },
    "tab_6": {
        "encoder": "Encoder:",
        "video_encoder_speed": "Encoder Speed:",
        "video_container": "Video Container",
        "video_pixel_format": "Pixel Format",
    },
    "advancedSettings": {
        "benchmarkModeCheckBox": "Perform processing without outputing new video.",
        "tilingCheckBox": "Split up processing upscaled frames into chunks.",
        "dynamicScaledOpticalFlowCheckBox": "Scales optical flow based on the difference between frames.",
        "ensembleCheckBox": "Ensembles flow, can produce better results.",
    },
}

#: Every tooltip this file expects on the five pages, including the two plain-text ones.
EXPECTED_TOTAL = 22


@pytest.fixture()
def language(qapp):
    """The real i18n module, configured in English, restored to the language it found afterwards."""
    from src.ui import i18n

    before = i18n.current_language()
    i18n.configure_i18n("en")
    yield i18n
    i18n.emit_language_changed(before)


def _build_shell(qapp):
    """The real shell in a bare `QMainWindow`, `resources_rc` stubbed (offscreen, no icons)."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from PySide6.QtWidgets import QMainWindow

    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    return shell, window


def _pages(shell):
    """The five page objects (the builders, captured by identity) and their widget roots."""
    proc = shell.procPage_page
    settings = shell.settingsPage_page
    builders = {
        "tab_2": settings.tab_2_page,
        "tab_3": settings.tab_3_page,
        "tab_4": settings.tab_4_page,
        "tab_6": proc.tab_6_page,
        "advancedSettings": proc.advancedSettings_page,
    }
    roots = {
        "tab_2": settings.tab_2,
        "tab_3": settings.tab_3,
        "tab_4": settings.tab_4,
        "tab_6": proc.tab_6,
        "advancedSettings": proc.advancedSettings,
    }
    return builders, roots


def _catalogue_tooltips(roots):
    """Every `i18n.tooltip(...)` document under the roots, as `{page: {objectName: text}}`."""
    from PySide6.QtWidgets import QWidget

    found: dict[str, dict[str, str]] = {}
    for page, root in roots.items():
        documents = {}
        for widget in root.findChildren(QWidget):
            text = widget.toolTip()
            # `src/ui/i18n.py::tooltip` is the one builder of these documents; the fixed prologue is its shape.
            if text.startswith("<html><head/><body>"):
                documents[widget.objectName()] = text
        found[page] = documents
    return found


def test_every_catalogue_tooltip_on_the_settings_tabs_follows_the_switch(language, qapp):
    """Build in English, switch to `ru`, and read the real widgets - no restart, no rebuild."""
    shell, window = _build_shell(qapp)
    try:
        builders, roots = _pages(shell)
        before = _catalogue_tooltips(roots)

        # The measured shape, page by page: a page that adds a tooltip in `build()` or renames a widget fails here
        # with the page and the widget named, rather than passing while one row stays in the old language.
        assert {page: sorted(documents) for page, documents in before.items()} == {
            page: sorted(expected) for page, expected in CATALOGUE_TOOLTIPS.items()
        }
        assert sum(len(documents) for documents in before.values()) == 20, (
            "expected 20 catalogue-document tooltips on the five settings pages")
        for page, expected in CATALOGUE_TOOLTIPS.items():
            for name, english in expected.items():
                assert english in before[page][name], (
                    f"{page}.{name} does not carry its English source in the tooltip: {before[page][name]!r}")

        language.emit_language_changed("ru")

        after = _catalogue_tooltips(roots)
        unchanged = [
            f"{page}.{name}"
            for page, documents in before.items()
            for name, text in documents.items()
            if after[page].get(name) == text
        ]
        assert not unchanged, (
            "these tooltips kept the language the page was built in - move their `setToolTip(i18n.tooltip(...))` "
            f"call from build() into retranslate(): {unchanged}")
        # A sample in the target language, so a table edit that blanks an entry cannot pass as "changed".
        assert "Определение смены сцен:" in after["tab_2"]["scene_change_detection_enabled"]
        assert "Формат пикселей" in after["tab_6"]["video_pixel_format"]

        # The seam repaints the live pages; a switch that rebuilt them would hand the user new widget identities.
        builders_after, roots_after = _pages(shell)
        assert builders_after == builders and roots_after == roots, (
            "the switch must repaint the live pages, not rebuild them")

        language.emit_language_changed("en")
        restored = _catalogue_tooltips(roots)
        assert restored == before, "switching back must restate the English source, byte for byte"
    finally:
        language.emit_language_changed("en")
        window.close()


def test_the_plain_text_tooltips_on_the_settings_tabs_follow_the_switch(language, qapp):
    """The two tooltips that are not `i18n.tooltip(...)` documents; together with the 20 above that is 22."""
    from PySide6.QtWidgets import QWidget

    shell, window = _build_shell(qapp)
    try:
        builders, roots = _pages(shell)
        command_field = roots["tab_6"].findChild(QWidget, "EncoderCommand")
        assert command_field is not None, "Tab6Page no longer names its command field `EncoderCommand`"
        quality = builders["tab_6"].video_quality

        english_command = command_field.toolTip()
        english_quality = quality.toolTip()
        assert english_command.startswith("Encoder command: ")
        # The toolkit combo shows its current item's text as a tooltip when the field elides it; that text is a
        # catalogue value (`Lossless`).
        assert english_quality == "Lossless"

        language.emit_language_changed("ru")

        assert command_field.toolTip() != english_command
        assert "Команда кодека" in command_field.toolTip()
        assert quality.toolTip() == "Без потерь"
        assert english_command not in command_field.toolTip()

        documents = _catalogue_tooltips(roots)
        # 20 catalogue documents (the test above) + these two = the 22 tooltips measured on the settings tabs.
        assert sum(len(page) for page in documents.values()) + 2 == EXPECTED_TOTAL
    finally:
        language.emit_language_changed("en")
        window.close()


def test_a_gpu_row_keeps_its_reason_across_a_switch(language, qapp):
    """The GPU-id tooltip is state, not a fixed document: the switch re-words it without replacing the reason.

    `apply_gpu_counts` writes a different line for 0 / 1 / 2+ devices. `retranslate()` remembers the counts, so after
    a switch the row still says "no GPU was detected" (in Russian), not the generic "sets what GPU is used".
    """
    shell, window = _build_shell(qapp)
    try:
        builders, _roots = _pages(shell)
        page = builders["tab_3"]
        page.apply_gpu_counts(ncnn_gpu_count=0, pytorch_gpu_count=1)
        assert "no GPU" in page.ncnn_gpu_id.toolTip()
        assert "one GPU" in page.pytorch_gpu_id.toolTip()

        language.emit_language_changed("ru")

        ncnn = page.ncnn_gpu_id.toolTip()
        pytorch = page.pytorch_gpu_id.toolTip()
        assert "GPU NCNN —" in ncnn and "GPU PyTorch —" in pytorch
        assert "no GPU" in ncnn, "the reason line must survive the switch (it is state, not a label)"
        assert "one GPU" in pytorch
        # The labels carry the same document once the glue has reported counts.
        assert page.label_45.toolTip() == ncnn
        assert page.label_43.toolTip() == pytorch
    finally:
        language.emit_language_changed("en")
        window.close()

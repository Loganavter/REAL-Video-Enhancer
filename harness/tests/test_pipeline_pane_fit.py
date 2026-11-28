"""The video-info pane fits the column with a REAL pipeline in it - at every factor.

The gap this closes (found by `rve-split-research`, 2026-11-28): `test_general_tab_fit` opened the upscale section but
never fed the pane a `PipelineFacts`, so it measured the EMPTY pane (382 px) and passed, while a real plan - five stage
labels plus a model name - made the ribbon grid's hint the column's minimum (measured 744 px at 1.0 ru inside a 470 px
viewport: 296 px clipped). The cause was two size policies without `ShrinkFlag` (`widget_26`'s `Minimum` and
`videoInfoPane`'s `MinimumExpanding`): `qSmartMinSize` takes the SIZE HINT as the minimum, so the pane voted on the
column's width and lost the vote by being cut at the splitter.

The invariant pinned here is the same one the page has for its fields: **the settings content is never wider than the
viewport it is shown in**, with the longest labels (ru), a long model name and a long output path - at 1.0, 1.25, 1.5
and 2.0.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

WINDOW = (1163, 900)

#: The longest model name the app ships with, so the ribbon's value column is realistic.
MODEL = "2x-Futsuu-Anime(1).pth"

#: A long auto-generated output path: the other thing a real session carries.
OUTPUT = "/home/jorj/native1440_20s_2x-futsuu-anime(1).pth_24.0fps_2880x2160_(12).mkv"


def _settle(qapp, turns: int = 10) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _shell(qapp, factor: float):
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.pages.general_settings import PipelineFacts, PipelineStage
    from src.ui.shell.window import MainWindowShell
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    i18n.configure_i18n("ru")           # the longest labels
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    UiScale.get_instance().set_factor(factor)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*WINDOW)
    window.show()
    _settle(qapp)

    page = shell.procPage_page
    page.generalSettings_page.generalUpscaleContainer.setVisible(True)
    page.generalSettings_page.outputFileText.setText(OUTPUT)
    page.generalSettings_page.set_pipeline(PipelineFacts(
        stages=(
            PipelineStage("Interpolate", True, "", 2),
            PipelineStage("Decompress"),
            PipelineStage("Denoise", True),
            PipelineStage("Deblur"),
            PipelineStage("Upscale", True, MODEL, 2),
        ),
        backend="tensorrt", preset="Fast", reuse=True, by_cel=True,
    ))
    _settle(qapp)
    return shell, window, page


@pytest.fixture()
def factor_state():
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    from src.ui import i18n

    scale = UiScale.get_instance()
    before_factor, before_language = scale.factor(), i18n.current_language()
    try:
        yield scale
    finally:
        scale.set_factor(before_factor)
        i18n.configure_i18n(before_language)


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0])
def test_the_pane_fits_the_column_with_a_real_plan(qapp, factor_state, factor):
    """A filled plan must not widen the column; the pane takes the width it is given and its content fits."""
    shell, window, page = _shell(qapp, factor)
    try:
        general = page.generalSettings_page
        viewport = general.scrollArea_4.viewport().width()
        content = general.scrollArea_4.widget().width()
        assert content <= viewport, (
            f"at {factor} the settings content is {content} px in a {viewport} px viewport - the plan would be cut "
            f"at the column's edge ({content - viewport} px)")
    finally:
        window.close()


def test_the_pane_does_not_vote_on_the_columns_width(qapp, factor_state):
    """The property behind the fix, measured: with a full plan the pane still has a SMALL minimum width.

    The two policies without `ShrinkFlag` made `qSmartMinSize` take the ribbon grid's hint (744 px at 1.0 ru) as the
    pane's - and therefore the column's - minimum. A pane that takes the width it is given reports a small minimum even
    when its cards carry long labels.
    """
    shell, window, page = _shell(qapp, 1.0)
    try:
        general = page.generalSettings_page
        pane = general.videoInfoPane
        host = pane.parentWidget()
        viewport = general.scrollArea_4.viewport().width()
        assert pane.minimumSizeHint().width() <= viewport, (
            f"the pane demands {pane.minimumSizeHint().width()} px of a {viewport} px viewport - its plan's text is "
            f"voting on the column")
        assert host.minimumSizeHint().width() <= viewport, (
            f"the host demands {host.minimumSizeHint().width()} px of a {viewport} px viewport")
    finally:
        window.close()

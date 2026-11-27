"""The video-info pane's pipeline ribbon (variant A): the PLAN the pane used to know nothing about.

The owner's report (2026-11-28): the General tab has a dozen render options and the pane knew exactly one of them (the
`2x upscale` tag). The chosen mockup states the pipeline as a left-to-right chain - Interpolate, Decompress, Denoise,
Deblur, Upscale - each stage carrying its own switch and its own parameter, plus a footer naming the backend, the
quality preset and the frame-reuse facts.

The contracts pinned here, and why each one is a behaviour rather than a shape:

* `GeneralSettingsPage.set_pipeline(facts)` paints the ribbon the facts describe - the enabled stages in render order
  with their parameters, the rest muted as `off` so the reader sees the whole path. Nothing here parses a label.
* switching a stage off removes its parameter from the screen in the same call. A stage that keeps its old model under
  `off` is the bug this test exists for: it is how a plan starts lying about what the render will do.
* the footer is the SETTINGS' answer (`settings['quality_preset']`, `settings['temporal_reuse']`,
  `settings['temporal_reuse_statistic']`), which is the same store `build_command` reads; a fact that is not
  on/selected is not painted at all.
* the idle state (no clip loaded) still shows the plan: it is a configuration, not a measurement. The three filled
  blocks stay hidden exactly as before, so the owner's "no labelled skeleton" rule is intact.
* a language switch repaints the stage names and the footer labels from the catalogue, while the values (model names,
  multipliers, the backend token) stay byte-identical.

`ProcessTab` is the one feeder: `pipeline_facts()` reads the widgets `getCurrentRenderOptions` reads and the `Settings`
`build_command` reads, and the tab's own 100 ms tick hands the result over (`_update_pipeline_ribbon`). `set_pipeline`
is asserted directly in most cases so a widget-level regression and a wiring-level one fail apart.

This file could not even be imported against the pre-change tree: `PipelineFacts`/`PipelineStage`/`set_pipeline` did not
exist, and neither did `ProcessTab.pipeline_facts`. Runs offscreen only; the look at 1.5/2.0 and the live switch on a
real window are `LIVE-PROOF WANTED` in `DONE.md`.
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

#: The app checkout root: `ProcessTab` pulls `src.constants`, and the suite's conftest puts the backend's `src` first
#: (the same seam `test_general_settings_info.py` resolves). The app's optional runtime imports are stubbed only when
#: this interpreter lacks them.
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
    INFO_STAGE_ORDER,
    GeneralSettingsPage,
    PipelineFacts,
    PipelineStage,
)

#: A configured pipeline: interpolation, denoise and a 2x upscale model, with decompress and deblur off. The model
#: names are the combo's own selections - the pane never invents a file name from them.
CONFIGURED = PipelineFacts(
    stages=(
        PipelineStage("Interpolate", True, "", 2),
        PipelineStage("Decompress", False),
        PipelineStage("Denoise", True, "drunet_color.pth"),
        PipelineStage("Deblur", False),
        PipelineStage("Upscale", True, "2x-Futsuu-Anime(1).pth", 2),
    ),
    backend="tensorrt",
    preset="fast",
    reuse=True,
    by_cel=True,
)


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


def _text(widget) -> str:
    """A toolkit `Text`'s stored string, not its elided display copy.

    The ribbon's values elide (a stage is ~80 px wide and a model name is long), and `Text.text()` answers the ELIDED
    string; `get_original_text()` is what was set and what the pane would still state at a wider column.
    """
    original = getattr(widget, "get_original_text", None)
    return original() if callable(original) else widget.text()


def _ribbon(page) -> dict[str, tuple[str, str, str]]:
    """`{stage key: (name, value, sub)}` as painted - the ribbon's whole visible statement per stage."""
    return {key: (_text(widgets["name"]), _text(widgets["value"]), _text(widgets["sub"]))
            for key, widgets in page.infoStages.items()}


def _footer(page) -> tuple[dict[str, str], dict[str, bool]]:
    """The footer's painted `(label, value)` pairs and which pairs are visible."""
    values = {key: (_text(label), _text(value))
              for key, (label, value, _host) in page.infoFooterValues.items()}
    visible = {key: host.isVisible() for key, (_label, _value, host) in page.infoFooterValues.items()}
    return values, visible


# --- 1. the stages, in render order, with their parameters ---------------------------------------


def test_the_ribbon_names_the_enabled_stages_in_render_order(language, qapp):
    page = _page(qapp)
    page.set_pipeline(CONFIGURED)
    _settle(qapp)

    assert list(page.infoStages) == [key for key, _share in INFO_STAGE_ORDER]
    assert _ribbon(page) == {
        "Interpolate": ("Interpolate", "×2", ""),
        "Decompress": ("Decompress", "off", ""),
        "Denoise": ("Denoise", "drunet_color.pth", ""),
        "Deblur": ("Deblur", "off", ""),
        "Upscale": ("Upscale", "2x-Futsuu-Anime(1).pth", "scale ×2"),
    }
    # The rail is the mockup's on/off statement: accent for a stage that runs, the hairline token for one that does not.
    assert page.infoStages["Interpolate"]["rail"].background_token() == "accent"
    assert page.infoStages["Deblur"]["rail"].background_token() == "separator.color"


def test_switching_a_stage_off_drops_its_parameter(language, qapp):
    """Turn deblur and upscale off: their models and the upscale factor must leave the ribbon with them."""
    page = _page(qapp)
    page.set_pipeline(PipelineFacts(
        stages=(
            PipelineStage("Interpolate", False),
            PipelineStage("Decompress", False),
            PipelineStage("Denoise", False),
            PipelineStage("Deblur", True, "nafnet_gopro1x.pth"),
            PipelineStage("Upscale", True, "2x-Futsuu-Anime(1).pth", 4),
        ),
        backend="pytorch", preset="best", reuse=False, by_cel=False,
    ))
    _settle(qapp)
    assert _ribbon(page)["Deblur"] == ("Deblur", "nafnet_gopro1x.pth", "")
    assert _ribbon(page)["Upscale"] == ("Upscale", "2x-Futsuu-Anime(1).pth", "scale ×4")

    page.set_pipeline(PipelineFacts(
        stages=(
            PipelineStage("Interpolate", False),
            PipelineStage("Decompress", False),
            PipelineStage("Denoise", False),
            PipelineStage("Deblur", False),
            PipelineStage("Upscale", False),
        ),
        backend="pytorch", preset="best", reuse=False, by_cel=False,
    ))
    _settle(qapp)
    # no model, no factor, no stale sub line: the stage is simply off
    assert _ribbon(page)["Deblur"] == ("Deblur", "off", "")
    assert _ribbon(page)["Upscale"] == ("Upscale", "off", "")
    assert not page.infoStages["Upscale"]["sub"].isVisible()
    assert "scale" not in page.infoStages["Upscale"]["sub"].text()
    assert page.infoStages["Upscale"]["rail"].background_token() == "separator.color"


# --- 2. the footer: backend, preset, reuse, by cel ------------------------------------------------


def test_the_footer_states_backend_preset_reuse_and_by_cel(language, qapp):
    page = _page(qapp)
    page.set_pipeline(CONFIGURED)
    _settle(qapp)

    values, visible = _footer(page)
    assert values == {
        "Backend": ("Backend", "tensorrt"),
        "Preset": ("Preset", "Fast"),
        "Reuse": ("Reuse", "on · by cel"),
    }
    assert all(visible.values())
    # The preset is the SETTINGS value painted through the catalogue the Quality Preset row already uses.
    page.set_pipeline(PipelineFacts(backend="pytorch", preset="best", reuse=True))
    _settle(qapp)
    assert _footer(page)[0]["Preset"] == ("Preset", "Best quality")
    assert _footer(page)[0]["Reuse"] == ("Reuse", "on")

    # A fact that is not on is not painted: reuse off drops the whole pair (and `by cel` with it, as the command does -
    # it passes no statistic at all once reuse is off).
    page.set_pipeline(PipelineFacts(backend="pytorch", preset="best", reuse=False, by_cel=False))
    _settle(qapp)
    _values, visible = _footer(page)
    assert visible["Reuse"] is False
    assert visible["Backend"] is True and visible["Preset"] is True


# --- 3. idle: the plan is the page, the measurement is not ----------------------------------------


def test_the_idle_pane_shows_the_plan_and_its_note(language, qapp):
    """No clip loaded: the plan card and its one sentence, and none of the three filled blocks."""
    page = _page(qapp)
    assert not page.infoPipelineCard.isVisible(), "nothing states a plan before the feeder has run"
    assert not page.infoFilled.isVisible()

    page.set_pipeline(CONFIGURED)
    _settle(qapp)

    assert page.infoPipelineCard.isVisible()
    assert page.infoPipelineCard.height() > 0
    assert page.infoIdleNote.isVisible()
    assert page.infoIdleNote.text().startswith("No video loaded.")
    # The owner's rule stands: a clip-less pane is not a labelled skeleton - only the plan is painted.
    assert not page.infoFilled.isVisible()
    assert page.video_info_text() == "", "the plan must not become a tenth row of the probe's record"

    # A loaded clip hides the note and brings the filled blocks back, with the plan still above them.
    page.set_video_info({"FPS:": ("24.0", "48.0"), "Resolution:": ("1440x1080", "2880x2160"),
                         "Frame Count:": ("493", "986")})
    _settle(qapp)
    assert page.infoFilled.isVisible()
    assert not page.infoIdleNote.isVisible()
    assert page.infoPipelineCard.isVisible()
    assert page.videoInfoPane.height() == page.videoInfoPane.sizeHint().height()


# --- 4. a language switch repaints the plan, not its values ---------------------------------------


def test_a_language_switch_repaints_the_ribbon(language, qapp):
    page = _page(qapp)
    page.set_pipeline(CONFIGURED)
    page.set_video_info({"FPS:": ("24.0", "48.0"), "Resolution:": ("1440x1080", "2880x2160"),
                         "Frame Count:": ("493", "986")})
    _settle(qapp)
    probe_before = page.video_info_text()

    language.emit_language_changed("ru")
    _settle(qapp)

    ribbon = _ribbon(page)
    assert ribbon["Interpolate"][0] == "Интерполяция"
    assert ribbon["Decompress"][0] == "Декомпрессия"
    assert ribbon["Denoise"][0] == "Шумоподавление"
    assert ribbon["Upscale"][0] == "Увеличение"
    assert ribbon["Deblur"][1] == "выкл"
    assert ribbon["Upscale"][2] == "масштаб ×2"
    footer, _visible = _footer(page)
    assert footer["Backend"] == ("Бэкенд", "tensorrt")
    assert footer["Preset"] == ("Пресет", "Быстрое")
    assert footer["Reuse"] == ("Переиспользование", "вкл · по целлу")
    assert page.infoIdleNote.text().startswith("Видео не загружено"), "the note follows the catalogue too"

    # The pipeline itself did not move: a parameter is a DATA value (a model file name, a multiplier), so it survives
    # the switch byte-identical; only the catalogue words - the stage names, the `off`/`on` state, `scale` - turn.
    assert ribbon["Interpolate"][1] == "×2"
    assert ribbon["Denoise"][1] == "drunet_color.pth"
    assert ribbon["Upscale"][1] == "2x-Futsuu-Anime(1).pth"
    assert page.video_info_text() == probe_before

    language.emit_language_changed("en")
    _settle(qapp)
    assert _ribbon(page)["Interpolate"][0] == "Interpolate"
    assert _footer(page)[0]["Preset"] == ("Preset", "Fast")


# --- 5. the feed: ProcessTab reads the settings the command reads ---------------------------------


class _Switch:
    """A checkbox stand-in: `isChecked()` is the only read the feed makes of a stage switch."""

    def __init__(self, checked: bool = False) -> None:
        self.checked = checked

    def isChecked(self) -> bool:
        return self.checked


class _Choice:
    """A combo stand-in: `currentText()` is the selection (a model name, "None", or the backend token)."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def currentText(self) -> str:
        return self.text


class _Spin:
    """A spin-box stand-in."""

    def __init__(self, value: float = 0) -> None:
        self.value_ = value

    def value(self) -> float:
        return self.value_


def _general(**overrides):
    """The General tab's widget surface as the feed reads it: switches, selections and the two multipliers."""
    surface = types.SimpleNamespace(
        interpolateCheckBox=_Switch(True), interpolateModelComboBox=_Choice("rife-v4.6.pth"),
        decompressCheckBox=_Switch(False), decompressModelComboBox=_Choice("deH264_RTMoSR.pth"),
        denoiseCheckBox=_Switch(True), denoiseModelComboBox=_Choice("drunet_color.pth"),
        deblurCheckBox=_Switch(False), deblurModelComboBox=_Choice("nafnet_gopro1x.pth"),
        upscaleCheckBox=_Switch(True), upscaleModelComboBox=_Choice("2x-Futsuu-Anime(1).pth"),
        interpolationMultiplierSpinBox=_Spin(3), upscaleScaleSpinBox=_Spin(2),
        backendComboBox=_Choice("tensorrt"),
    )
    for name, value in overrides.items():
        setattr(surface, name, value)
    return surface


def _settings(**values):
    stored = {"quality_preset": "fast", "temporal_reuse": "True", "temporal_reuse_statistic": "coarse"}
    stored.update(values)
    return types.SimpleNamespace(
        settings=stored,
        allowedSettings={"quality_preset": ("best", "fast"),
                         "temporal_reuse": ("True", "False"),
                         "temporal_reuse_statistic": ("mean_abs", "coarse")},
    )


def test_the_feed_reads_the_same_settings_the_command_reads():
    _path_before = list(sys.path)
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)
    sys.modules.pop("src.constants", None)
    try:
        from src.ui.ProcessTab import ProcessTab
    finally:
        sys.path[:] = _path_before

    tab = object.__new__(ProcessTab)
    facts = tab.pipeline_facts(general=_general(), settings=_settings())

    assert [stage.key for stage in facts.stages] == ["Interpolate", "Decompress", "Denoise", "Deblur", "Upscale"]
    by_key = {stage.key: stage for stage in facts.stages}
    assert (by_key["Interpolate"].enabled, by_key["Interpolate"].factor) == (True, 3)
    assert (by_key["Denoise"].enabled, by_key["Denoise"].model) == (True, "drunet_color.pth")
    assert by_key["Deblur"].enabled is False and by_key["Deblur"].model == ""
    assert (by_key["Upscale"].enabled, by_key["Upscale"].model, by_key["Upscale"].factor) == (
        True, "2x-Futsuu-Anime(1).pth", 2)
    assert (facts.backend, facts.preset, facts.reuse, facts.by_cel) == ("tensorrt", "fast", True, True)

    # The switches are the truth: an unchecked stage is off AND carries no parameter, whatever its combo still shows.
    off = tab.pipeline_facts(general=_general(deblurCheckBox=_Switch(True), upscaleCheckBox=_Switch(False)),
                             settings=_settings())
    off_by_key = {stage.key: stage for stage in off.stages}
    assert (off_by_key["Deblur"].enabled, off_by_key["Deblur"].model) == (True, "nafnet_gopro1x.pth")
    assert (off_by_key["Upscale"].enabled, off_by_key["Upscale"].model, off_by_key["Upscale"].factor) == (
        False, "", None)

    # The store's own rules, the same ones `build_command` applies: an unknown preset is "best", and `by cel` is the
    # coarse statistic only while reuse is on (reuse off means the command passes no statistic at all).
    odd = tab.pipeline_facts(general=_general(), settings=_settings(quality_preset="placebo"))
    assert odd.preset == "best"
    no_reuse = tab.pipeline_facts(general=_general(),
                                  settings=_settings(temporal_reuse="False", temporal_reuse_statistic="coarse"))
    assert (no_reuse.reuse, no_reuse.by_cel) == (False, False)


# --- 6. the tick hands the plan over (the seam, not a hand-picked setter) --------------------------


def test_the_tick_hands_the_plan_to_the_pane(language, qapp):
    """`updateProcessTab` is the app's own 100 ms clock; the plan must ride it, not a test-only setter."""
    _path_before = list(sys.path)
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)
    sys.modules.pop("src.constants", None)
    try:
        from src.ui.ProcessTab import ProcessTab
    finally:
        sys.path[:] = _path_before

    page = _page(qapp)
    tab = object.__new__(ProcessTab)
    tab.parent = types.SimpleNamespace(shell=types.SimpleNamespace(
        procPage_page=types.SimpleNamespace(generalSettings_page=page)))
    tab.page_handles = types.SimpleNamespace(general=_general())
    tab.settings = _settings()
    tab._pipeline_state = None
    tab._pipeline_refresh_pending = False
    tab._run_card_state = None
    tab._render_started_mono = None
    for name in ("_ensure_live_tail", "_tick_press_deadline",
                 "_sync_preview_transport", "_refresh_held_progress", "_push_preview_state",
                 "_ensure_picture_timer"):
        setattr(tab, name, lambda: None)
    tab.renderTextOutputList = None

    tab.updateProcessTab()
    _settle(qapp)

    ribbon = _ribbon(page)
    assert ribbon["Interpolate"] == ("Interpolate", "×3", "")
    assert ribbon["Denoise"] == ("Denoise", "drunet_color.pth", "")
    assert ribbon["Upscale"] == ("Upscale", "2x-Futsuu-Anime(1).pth", "scale ×2")
    assert _footer(page)[0]["Backend"] == ("Backend", "tensorrt")

    # The gate: an unchanged configuration repaints nothing (the pane's setter is not called a second time).
    calls = []
    page.set_pipeline = lambda facts: calls.append(facts)
    tab.updateProcessTab()
    assert calls == [], f"an unchanged plan was repainted: {calls}"

    # A changed switch does reach the pane on the next tick.
    tab.page_handles.general.upscaleCheckBox.checked = False
    tab.updateProcessTab()
    assert calls and calls[-1].stages[-1].enabled is False

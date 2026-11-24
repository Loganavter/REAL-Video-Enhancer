"""The playback strip follows the toolkit's design language, and it says where the playhead is.

The owner's report (2026-11-24): *"bring the playback strip with the seeker to the toolkit design ... and for some reason
there is no basic time display there."* The strip was painted inside the QRhi HUD pass with bespoke numbers - a 6 px
track, a 7 px handle - and literal RGBA tuples. The HUD is one rect-only pass, so a text readout cannot live in it: the
pane puts a real toolkit `Text` widget beside the picture instead.

What is pinned here, offscreen and without a GPU:

* the strip's track height and handle radius are the toolkit's own `Slider.TRACK_HEIGHT` / `Slider.RADIUS`, read at
  runtime through the same `scaled_px()` the toolkit uses - not a second copy of the numbers;
* every colour the strip paints resolves through `ThemeManager`: the module publishes the token names, the resolver
  returns exactly the theme's colour for each, and the strip path carries no literal 4-float colour tuple;
* `format_time` prints `mm:ss.mmm`, and the readout prints `playhead / duration` with em dashes before a frame exists;
* a `set_progress` call moves the readout, and the readout is a real child widget the pane positions above the strip
  (a child of a QRhiWidget composites on top; text baked into the GPU pass cannot work, the pass is rects only);
* both the strip and the readout follow `UiScale`.

The strip is a timeline, not a slider: it keeps its produced band, hover marker and blocked marker, and the HUD can only
draw rectangles - the toolkit's rounded slider ends are the one thing the pass cannot reproduce.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager  # noqa: E402
from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px  # noqa: E402
from sli_ui_toolkit.ui.widgets.atomic.slider import Slider  # noqa: E402
from sli_ui_toolkit.ui.widgets.text import Text  # noqa: E402

from src.ui.widgets import playback_strip as strip  # noqa: E402
from src.ui.widgets.video_preview import VideoPreviewSurface  # noqa: E402

PANE_SIZE = (800, 600)
FRAME_SIZE = (1600, 900)
PLAYHEAD = 6.194
DURATION = 20.646


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _picture(width: int, height: int, colour: str = "#204060") -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(colour))
    return image


@pytest.fixture()
def pane():
    _app()
    widget = VideoPreviewSurface()
    widget.resize(*PANE_SIZE)
    return widget


@pytest.fixture(autouse=True)
def _toolkit_palette():
    """The app registers the toolkit palettes at start-up; a bare `ThemeManager` has none.

    Register them here and put the previous dicts back, so this module can assert real token resolution without
    leaking a palette into the rest of the suite (or depending on which test ran first).
    """
    from sli_ui_toolkit.palettes import FLUENT_DARK, FLUENT_LIGHT

    theme = ThemeManager.get_instance()
    previous = (dict(theme._light_palette), dict(theme._dark_palette))
    theme.register_palettes(FLUENT_LIGHT, FLUENT_DARK)
    yield theme
    theme._light_palette, theme._dark_palette = previous


def _with_frame(pane: VideoPreviewSurface) -> VideoPreviewSurface:
    pane.set_frame(_picture(*FRAME_SIZE))
    return pane


# --- the design language's metrics -------------------------------------------------------------------

def test_track_height_and_handle_radius_are_the_toolkit_constants(pane):
    """The numbers come from `Slider`, not from this file: a toolkit bump must move the strip."""
    assert pane._strip_height == scaled_px(Slider.TRACK_HEIGHT) != 6
    assert pane._handle_radius == scaled_px(Slider.RADIUS) != 7


def test_the_toolkit_playhead_constant_is_read_not_copied():
    """The timeline's playhead handle is the strip's halo: its design size comes from the toolkit widget itself."""
    from sli_ui_toolkit.ui.widgets.composite.timeline_widget.widget import TimelineWidget

    size = strip.timeline_handle_size()
    assert size == TimelineWidget().HANDLE_SIZE, "the strip reads the toolkit widget, it does not copy the number"
    assert pane_halo(size) == max(scaled_px(Slider.RADIUS) + 1.0, scaled_px(size) / 2.0)


def pane_halo(size: int) -> float:
    """What the pane derives from the timeline handle for the playhead halo."""
    return max(scaled_px(Slider.RADIUS) + 1.0, scaled_px(size) / 2.0)


def test_the_pane_uses_the_same_halo_formula():
    _app()
    widget = VideoPreviewSurface()
    assert widget._playhead_halo_radius == pane_halo(strip.timeline_handle_size())


# --- the colours come from tokens ----------------------------------------------------------------------

def test_every_strip_token_resolves_through_the_theme_manager():
    """The published map is the whole vocabulary: each name resolves to the theme's own colour."""
    theme = ThemeManager.get_instance()
    colours = strip.strip_colours()
    assert set(colours) == set(strip.STRIP_TOKENS)
    assert len(strip.STRIP_TOKENS) >= 6, "track, produced, playhead, outline, hover, blocked"
    for name, token in strip.STRIP_TOKENS.items():
        assert theme.try_get_color(token) is not None, f"{token} is not a toolkit token"
        assert colours[name] == theme.get_color(token), f"{name} did not come from {token}"


def test_the_strip_path_has_no_literal_four_float_colour():
    """A literal RGBA tuple in the strip path bypasses the theme; prose and the vertex pairs are not colours."""
    for function in (VideoPreviewSurface._hud_vertices, strip.strip_colours, strip.rgba):
        hits = _four_float_tuples(function)
        assert hits == [], f"{function.__qualname__} still carries literal colours: {hits}"
    # `rgba()` returns a 4-tuple, but it builds it from a QColor, and no other function in the module does.
    assert _four_float_tuples(strip) == []


def _four_float_tuples(target) -> list[str]:
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):  # pragma: no cover - a builtin/C function never carries a literal
        return []
    tree = ast.parse(textwrap.dedent(source))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 4:
            continue
        if all(isinstance(elt, ast.Constant) and isinstance(elt.value, float) for elt in node.elts):
            hits.append(ast.unparse(node))
    return hits


def test_the_produced_band_paints_the_progress_fill_token(pane):
    """The vertices really carry the resolved colour (not a constant that merely looks like it)."""
    _with_frame(pane).set_progress(10.0, 20.0, PLAYHEAD, DURATION)
    vertices = pane._hud_vertices()
    colours = {tuple(round(value, 5) for value in vertices[index + 2:index + 6])
               for index in range(0, len(vertices), 6)}
    expected = tuple(round(value, 5) for value in
                     strip.rgba(ThemeManager.get_instance().get_color(strip.STRIP_TOKENS["produced"])))
    assert expected in colours


# --- the time readout ------------------------------------------------------------------------------------

def test_the_formatter_prints_minutes_seconds_and_milliseconds():
    assert strip.format_time(PLAYHEAD) == "00:06.194"
    assert strip.format_time(DURATION) == "00:20.646"
    assert strip.format_time(0.0) == "00:00.000"
    assert strip.format_time(61.5) == "01:01.500"
    assert strip.format_time(None) == strip.TIME_PLACEHOLDER
    assert strip.format_time(float("nan")) == strip.TIME_PLACEHOLDER
    assert strip.format_time(float("inf")) == strip.TIME_PLACEHOLDER
    assert strip.format_time(-1.0) == strip.TIME_PLACEHOLDER


def test_em_dashes_before_a_frame_exists(pane):
    assert "—" in pane._time_readout.text()
    assert pane._time_readout.text() == strip.format_readout(None, None)


def test_the_readout_shows_playhead_over_duration_after_a_frame(pane):
    _with_frame(pane)
    pane.set_progress(rendered=0.0, total=DURATION, playhead=PLAYHEAD, duration=DURATION)
    assert pane._time_readout.text() == "00:06.194 / 00:20.646"


def test_set_progress_moves_the_readout(pane):
    _with_frame(pane)
    pane.set_progress(0.0, DURATION, PLAYHEAD, DURATION)
    before = pane._time_readout.text()
    pane.set_progress(0.0, DURATION, 7.5, DURATION)
    assert pane._time_readout.text() == "00:07.500 / 00:20.646"
    assert pane._time_readout.text() != before


def test_the_readout_is_a_real_toolkit_text_child_above_the_strip(pane):
    """Not baked into the HUD: a `Text` widget the pane owns, positioned over the picture and clear of the strip."""
    _with_frame(pane).set_progress(0.0, DURATION, PLAYHEAD, DURATION)
    readout = pane._time_readout
    assert isinstance(readout, Text)
    assert readout.parent() is pane
    assert readout.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents), (
        "the readout must not swallow presses meant for the picture")
    geometry = readout.geometry()
    assert pane.rect().contains(geometry), "the readout stays inside the pane"
    assert geometry.bottom() <= pane._strip_rect().top(), "the readout sits above the strip"


def test_the_readout_goes_back_to_dashes_when_the_frame_clears(pane):
    _with_frame(pane).set_progress(0.0, DURATION, PLAYHEAD, DURATION)
    pane.set_frame(None)
    assert "—" in pane._time_readout.text()


# --- scale ------------------------------------------------------------------------------------------------

def test_the_strip_and_the_readout_follow_ui_scale():
    _app()
    scale = UiScale.get_instance()
    try:
        scale.set_factor(1.0)
        expected_small_height = scaled_px(Slider.TRACK_HEIGHT)
        expected_small_radius = scaled_px(Slider.RADIUS)
        small = _sized_pane()
        small_height = small._strip_height
        small_radius = small._handle_radius
        small_font = small._time_readout.font().pixelSize()
        small_width = small._time_readout.width()

        scale.set_factor(1.5)
        expected_large_height = scaled_px(Slider.TRACK_HEIGHT)
        large = _sized_pane()

        assert small_height == expected_small_height
        assert small_radius == expected_small_radius
        assert large._strip_height == expected_large_height > small_height
        assert large._handle_radius == scaled_px(Slider.RADIUS) > small_radius
        assert large._time_readout.font().pixelSize() > small_font
        assert large._time_readout.width() > small_width
        # The strip already on screen follows the factor too (it subscribes to `scale_changed`).
        assert small._strip_height == expected_large_height
    finally:
        scale.set_factor(1.0)


def _sized_pane() -> VideoPreviewSurface:
    widget = VideoPreviewSurface()
    widget.resize(*PANE_SIZE)
    widget.set_frame(_picture(*FRAME_SIZE))
    widget.set_progress(0.0, DURATION, PLAYHEAD, DURATION)
    return widget

"""The compare boundary's colours: the toolkit's own tokens, resolved at paint time.

The owner's report (2026-11-24): *"bring the playback strip with the seeker to the toolkit design ... and for some
reason there is no basic time display there."* The answer shipped in two parts, and only ONE of them still lives in
the pane:

* the transport bar (`src/ui/preview/controls.py` + `preview/timeline.py`) is the strip and the time display now - its
  readout, its formatter and its produced band are pinned in `test_preview_controls.py`;
* the pane keeps only the compare boundary (the `sbs` seam and the `split` divider), whose colours still come from
  `playback_strip.strip_colours()` through `ThemeManager` rather than from literal RGBA tuples.

The pane's OWN painted strip - track, produced band, playhead handle, hover and blocked markers, and the readout widget
that sat above it - is deleted (audit 2026-11-27 §6.6: two bar implementations, two seek semantics, and a guard that
could never fire in the shipped window). What is pinned here is what survives: the token vocabulary resolves through
the theme, and no function on the surviving paint path carries a literal 4-float colour.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager  # noqa: E402

from src.ui.widgets import playback_strip as strip  # noqa: E402
from src.ui.widgets.video_preview import DIVIDER_REST_ALPHA, VideoPreviewSurface  # noqa: E402

PANE_SIZE = (800, 600)
FRAME_SIZE = (1600, 900)


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


# --- the design language's metrics -------------------------------------------------------------------

def test_the_toolkit_playhead_constant_is_read_not_copied():
    """The handle size is the toolkit timeline widget's own design constant, not a copy in this tree.

    The PANE no longer derives a halo from it (its strip is deleted); the module's answer stays pinned because it is
    still the one place the number is read (`playback_strip.timeline_handle_size`).
    """
    from sli_ui_toolkit.ui.widgets.composite.timeline_widget.widget import TimelineWidget

    _app()                                   # a toolkit widget needs the application object
    size = strip.timeline_handle_size()
    assert size == TimelineWidget().HANDLE_SIZE, "the strip reads the toolkit widget, it does not copy the number"
    source = textwrap.dedent(inspect.getsource(strip.timeline_handle_size))
    assert any(isinstance(node, ast.Constant) and node.value == "HANDLE_SIZE"
               for node in ast.walk(ast.parse(source))), "the constant is read, not spelled as a number"


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
    for function in (VideoPreviewSurface._hud_vertices, VideoPreviewSurface._compare_boundary_rects,
                     strip.strip_colours, strip.rgba):
        hits = _four_float_tuples(function)
        assert hits == [], f"{function.__qualname__} still carries literal colours: {hits}"
    # `rgba()` returns a 4-tuple, but it builds it from a QColor, and no other function in the module does.
    assert _four_float_tuples(strip) == []


def test_the_compare_boundary_paints_the_divider_token(pane):
    """The vertices really carry the resolved colour (not a constant that merely looks like it).

    This test used to read the strip's produced BAND; with the strip deleted the pane's one painted boundary is the
    `split` divider, so the same claim - "the colour is the token's, resolved at paint time" - is pinned on it.
    """
    pane.set_frame(_picture(*FRAME_SIZE))
    pane.set_original(_picture(*FRAME_SIZE))
    pane.set_view_mode("split")
    pane._divider_left_at = None                       # full alpha: the rule and its grip are both painted
    vertices = pane._hud_vertices()
    assert vertices, "split without a boundary would mean the divider is gone"
    colours = {tuple(round(value, 5) for value in vertices[index + 2:index + 6])
               for index in range(0, len(vertices), 6)}
    divider = ThemeManager.get_instance().get_color(strip.STRIP_TOKENS["divider"])
    outline = ThemeManager.get_instance().get_color(strip.STRIP_TOKENS["divider_outline"])
    grip = DIVIDER_REST_ALPHA + (0.55 - DIVIDER_REST_ALPHA) * 1.0      # the grip's own alpha at full pointer
    for colour, alpha in ((divider, 0.38), (divider, grip), (outline, 0.78)):
        expected = tuple(round(value, 5) for value in strip.rgba(colour, alpha))
        assert expected in colours, f"no rect carries {colour} at alpha {alpha}: {sorted(colours)}"


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

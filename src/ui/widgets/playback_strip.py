"""The playback strip's shared vocabulary: the toolkit's metrics, its tokens, and the time readout.

The strip itself is painted inside the pane's QRhi HUD pass (`VideoPreviewSurface._hud_vertices`), which is a rect-only
pass and therefore cannot draw text. This module holds the two things that must not be guessed in that pass:

* **where the numbers come from** - the track height and thumb radius are ``Slider.TRACK_HEIGHT`` / ``Slider.RADIUS``
  (design px) through the toolkit's own ``scaled_px()``; the playhead halo is the timeline widget's playhead handle
  (``TimelineWidget.HANDLE_SIZE``, an instance attribute in the toolkit, so it is read from a widget once and cached).
  A toolkit bump moves the strip with it, with no second copy of the values here.
* **where the colours come from** - ``STRIP_TOKENS`` names every colour the strip paints, and ``strip_colours()``
  resolves each one through ``ThemeManager`` at paint time, so a theme switch (light/dark) recolours the strip and the
  readout together.

The readout is a real widget because the HUD cannot carry text: ``PlaybackTimeReadout`` is a toolkit ``Text`` the pane
owns as a child, positions over the picture, and feeds from ``set_progress``. A sibling/child widget composites above a
``QRhiWidget`` on this platform (framework precedent: a QRhiWidget overwrites its own rect and cannot alpha-blend, but
its children are composed by Qt afterwards), while text baked into the GPU pass would need a whole glyph atlas this
pane does not have.
"""

from __future__ import annotations

import functools

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.ui.widgets.atomic.slider import Slider
from sli_ui_toolkit.ui.widgets.text import Text

#: One em dash per missing field group: `mm:ss.mmm` becomes `—:—.———`.
EM_DASH = "\u2014"
TIME_PLACEHOLDER = f"{EM_DASH}:{EM_DASH}.{EM_DASH}{EM_DASH}{EM_DASH}"
#: Playhead / duration, the two time bases the pane keeps apart.
READOUT_SEPARATOR = " / "
#: Readout type: the design language's caption size (11 px), scaled by `UiFont`.
READOUT_PIXEL_SIZE = 11

#: Every colour the strip paints, by role. The strip is a timeline, not a slider, but it borrows the slider's groove
#: and thumb language and the timeline's progress tokens; nothing here is a literal RGBA.
STRIP_TOKENS = {
    # the groove the renderer has not reached yet - the slider's own unfilled track
    "track": "slider.track.unfilled",
    # the produced band - the toolkit's progress fill (the render head is the fill's own edge)
    "produced": "toast.progress.fill",
    # the playhead handle - the slider's thumb, outer fill (its accent inner square is the "produced" token)
    "playhead": "slider.thumb.outer",
    # the handle's outline/halo and the playhead line's backdrop - the slider groove's background token
    "playhead_outline": "slider.track.background",
    # the hover marker - the thumb colour, dimmed at paint time
    "hover": "slider.thumb.outer",
    # the blocked marker (past the produced edge) - Qt's `BrightText` is the palette's red in both themes
    "blocked": "BrightText",
    # the compare modes' divider - a separator, not part of the strip's bands
    "divider": "separator.color",
}


# -- metrics ------------------------------------------------------------------------------------------

def strip_track_height() -> int:
    """The track's height in widget px: the toolkit slider's track, through the toolkit's scale."""
    return scaled_px(Slider.TRACK_HEIGHT)


def strip_handle_radius() -> int:
    """The playhead handle's radius in widget px: the toolkit slider's thumb radius."""
    return scaled_px(Slider.RADIUS)


def strip_margin() -> int:
    """The strip's inset from the pane edge: the toolkit slider's own margin."""
    return scaled_px(Slider.MARGIN)


@functools.lru_cache(maxsize=1)
def timeline_handle_size() -> int:
    """The timeline widget's playhead handle (18 design px), read from the toolkit rather than copied.

    The toolkit assigns ``HANDLE_SIZE`` in ``TimelineWidget.__init__`` (it is not a class attribute), so the honest
    source is a widget. The design value is cached; scaling stays at the call site so a `UiScale` change still moves it.
    Constructing a timeline is cheap (a few ms offscreen), and it happens once per process.
    """
    from sli_ui_toolkit.ui.widgets.composite.timeline_widget.widget import TimelineWidget

    size = getattr(TimelineWidget, "HANDLE_SIZE", None)
    if size is not None:
        return int(size)
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:  # pragma: no cover - a widget always exists in the app and the tests
        return 2 * Slider.RADIUS
    return int(TimelineWidget().HANDLE_SIZE)


def playhead_halo_radius(handle_radius: int | None = None) -> float:
    """The playhead handle's halo: at least one px wider than the thumb, and the timeline handle's full size.

    At the default scale that is 8 px thumb radius -> 16 px handle, inside the timeline's 18 px handle, i.e. a 1 px
    halo on each side. Both numbers are the toolkit's.
    """
    radius = strip_handle_radius() if handle_radius is None else int(handle_radius)
    return max(float(radius) + 1.0, scaled_px(timeline_handle_size()) / 2.0)


# -- colours ------------------------------------------------------------------------------------------

def strip_colours() -> dict[str, QColor]:
    """Resolve every strip colour from the active theme, by role (see ``STRIP_TOKENS``)."""
    theme = ThemeManager.get_instance()
    return {name: QColor(theme.get_color(token)) for name, token in STRIP_TOKENS.items()}


def rgba(colour: QColor, alpha: float | None = None) -> tuple[float, float, float, float]:
    """A theme colour as the HUD's straight-alpha floats; `alpha` (0..1) dims a marker without a new literal."""
    value = QColor(colour)
    if alpha is not None:
        value.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return (value.redF(), value.greenF(), value.blueF(), value.alphaF())


# -- the time readout -----------------------------------------------------------------------------------

def format_time(seconds: float | None) -> str:
    """`mm:ss.mmm`, or the em-dash placeholder when there is no time yet."""
    if seconds is None:
        return TIME_PLACEHOLDER
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return TIME_PLACEHOLDER
    if value != value or value < 0:  # NaN or negative: not a time this pane can stand on
        return TIME_PLACEHOLDER
    total_ms = int(round(value * 1000.0))
    minutes, rest = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(rest, 1_000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def format_readout(playhead: float | None, duration: float | None) -> str:
    """The readout line: `playhead / duration`, em dashes before a frame/duration exists."""
    return f"{format_time(playhead)}{READOUT_SEPARATOR}{format_time(duration)}"


class PlaybackTimeReadout(Text):
    """The pane's time display: a toolkit `Text` the pane positions above the strip and feeds from `set_progress`.

    It is deliberately a widget, not part of the HUD: the HUD pass draws rectangles only. `panel_fill` gives the caption
    a theme plate so it stays readable over any picture, in either theme, and the small padding keeps the glyphs off the
    plate's edge without a QSS sheet (the design language forbids styling toolkit widgets with QSS).
    """

    #: Design-px padding of the caption plate, scaled by `UiScale` in `sizeHint`.
    PAD_X = 6
    PAD_Y = 2

    def __init__(self, parent=None) -> None:
        super().__init__("", parent, pixel_size=READOUT_PIXEL_SIZE, color_token="dialog.text",
                         alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.setObjectName("playbackTimeReadout")
        self.set_panel_fill(True)
        # The caption is not a control: a press must reach the pane (the picture / the strip below it).
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.set_times(None, None)

    def set_times(self, playhead: float | None, duration: float | None) -> None:
        text = format_readout(playhead, duration)
        if text != self.text():
            self.setText(text)
        self.adjustSize()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt spelling
        hint = super().sizeHint()
        return QSize(hint.width() + 2 * scaled_px(self.PAD_X), hint.height() + 2 * scaled_px(self.PAD_Y))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt spelling
        return self.sizeHint()


__all__ = [
    "EM_DASH",
    "READOUT_PIXEL_SIZE",
    "READOUT_SEPARATOR",
    "STRIP_TOKENS",
    "TIME_PLACEHOLDER",
    "PlaybackTimeReadout",
    "format_readout",
    "format_time",
    "playhead_halo_radius",
    "rgba",
    "strip_colours",
    "strip_handle_radius",
    "strip_margin",
    "strip_track_height",
    "timeline_handle_size",
]

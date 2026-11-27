"""The action catalog: one spec per control, and nothing else decides how a control is wired.

Owner direction (2026-11-25): *"заранее продумай архитектуру всех этих кнопок чтобы они не через магию работали, а
через профессиональный паттерн."* The pattern is a registry. Every control is one `ActionSpec`, and adding a control
means adding a spec here plus (when it is a new verb) one dispatch case in `PreviewController` - never a lambda in a
third file, never a `setEnabled` in a signal handler.

An `ActionSpec` carries what a widget needs (`text`, `tip`, `icon`, `checkable`, `group`, `hold`, `bar`) and what the
state decides (`visible_when`, `enabled_when`, `checked_when`, `icon_when`). The view builds the widgets from these
specs once; `PreviewControls.set_state` walks the same list to apply the predicates in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtGui import QIcon

from sli_ui_toolkit.widgets import Button, Layer

from .state import PreviewCommand, PreviewState

#: The app's own SVG set (compiled into `resources_rc`): the same icons the page's buttons carry.
ICON_PREFIX = ":/icons/icons/"

TOP = "top"
BOTTOM = "bottom"


def icon(name: str) -> QIcon:
    return QIcon(f"{ICON_PREFIX}{name}.svg")


def _picture(state: PreviewState) -> bool:
    return state.has_picture()


def _timeline(state: PreviewState) -> bool:
    return state.has_picture() and state.duration > 0.0


@dataclass(frozen=True)
class ActionSpec:
    """One control: its identity, its verb, and the state predicates that decide how it paints."""

    id: str
    command: PreviewCommand
    value: object = None
    text: str = ""
    tip: str = ""
    icon: QIcon | None = None
    icon_size: int = 16
    #: toggle buttons carry `value` to the command; a plain button emits its command on release.
    checkable: bool = False
    #: exclusive visual group (the toolkit's `ButtonGroup` is a container; the exclusivity is a `QButtonGroup`).
    group: str | None = None
    #: the keyboard route for the verb (the pane's `WidgetShortcut`; the slider keeps its own arrows).
    shortcut: str | None = None
    variant: str = "surface"
    #: DESIGN px, and `icon_size` below too: `Button` resolves both through `scaled_px()` itself, so this file must
    #: never pre-scale them (the same rule as the shell's `NAV_ICON_PX`, `shell/window.py:73-80`). Width 0 lets the
    #: toolkit size the button from its content (text buttons).
    size: tuple[int, int] = (0, 30)
    bar: str = TOP
    #: overflow order: the HIGHEST priority survives a narrow pane; lower ones collapse into the "…" menu.
    priority: int = 0
    visible_when: Callable[[PreviewState], bool] | None = None
    enabled_when: Callable[[PreviewState], bool] | None = None
    checked_when: Callable[[PreviewState], bool] | None = None
    #: the glyph follows the state (a player's play/pause button: the icon is the ACTION a press performs).
    icon_when: Callable[[PreviewState], QIcon | None] | None = None


#: The catalog. One list for both bars: `bar` puts a control in the picture's top row or the transport below it.
ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        id="mode_single", command=PreviewCommand.SET_MODE, value="single", tip="Single picture",
        icon=icon("square"), checkable=True, group="view_mode", variant="ghost", bar=TOP, size=(30, 30), priority=70,
        enabled_when=_picture, checked_when=lambda s: s.mode == "single",
    ),
    ActionSpec(
        id="mode_sbs", command=PreviewCommand.SET_MODE, value="sbs", tip="Side by side: source | result",
        icon=icon("columns"), checkable=True, group="view_mode", variant="ghost", bar=TOP, size=(30, 30), priority=70,
        enabled_when=_picture, checked_when=lambda s: s.mode == "sbs",
    ),
    ActionSpec(
        id="mode_split", command=PreviewCommand.SET_MODE, value="split", tip="Split: source left of the divider",
        icon=icon("divide-square"), checkable=True, group="view_mode", variant="ghost", bar=TOP, size=(30, 30), priority=70,
        enabled_when=_picture, checked_when=lambda s: s.mode == "split",
    ),
    ActionSpec(
        id="zoom_out", command=PreviewCommand.ZOOM_OUT, tip="Zoom out", icon=icon("zoom-out"),
        variant="ghost", group="zoom", bar=TOP, size=(28, 30), priority=30,
        enabled_when=_picture,
    ),
    ActionSpec(
        id="zoom_fit", command=PreviewCommand.ZOOM_FIT, text="Fit", tip="Fit the picture to the pane",
        checkable=True, group="zoom", variant="ghost", bar=TOP, size=(0, 30), priority=40,
        enabled_when=_picture, checked_when=lambda s: s.zoom_fit,
    ),
    ActionSpec(
        id="zoom_one", command=PreviewCommand.ZOOM_ONE, text="1:1", tip="One source pixel per screen pixel",
        checkable=True, group="zoom", variant="ghost", bar=TOP, size=(0, 30), priority=40,
        enabled_when=_picture,
        checked_when=lambda s: (not s.zoom_fit) and abs(float(s.zoom) - 1.0) < 1e-3,
    ),
    ActionSpec(
        id="zoom_in", command=PreviewCommand.ZOOM_IN, tip="Zoom in", icon=icon("zoom-in"),
        variant="ghost", group="zoom", bar=TOP, size=(28, 30), priority=30,
        enabled_when=_picture,
    ),
    ActionSpec(
        id="step_back", command=PreviewCommand.STEP_BACK, tip="One frame back", icon=icon("skip-back"),
        variant="ghost", bar=BOTTOM, size=(28, 32), priority=50, enabled_when=_timeline, shortcut="Left",
    ),
    ActionSpec(
        id="play", command=PreviewCommand.PLAY_PAUSE, tip="Play / pause",
        icon=icon("play"), icon_size=18, bar=BOTTOM, size=(36, 36), priority=100,
        enabled_when=_timeline, shortcut="Space",
        icon_when=lambda s: icon("pause") if s.playing() else icon("play"),
    ),
    ActionSpec(
        id="step_forward", command=PreviewCommand.STEP_FORWARD, tip="One frame forward", icon=icon("skip-forward"),
        variant="ghost", bar=BOTTOM, size=(28, 32), priority=50, enabled_when=_timeline, shortcut="Right",
    ),
)


def actions_for(bar: str) -> tuple[ActionSpec, ...]:
    """The specs of one bar, in catalog order."""
    return tuple(spec for spec in ACTIONS if spec.bar == bar)


def build_button(spec: ActionSpec, parent=None, *,
                 corner_radii: tuple[int, int, int, int] | None = None,
                 extra_layers: list[Layer] | None = None) -> Button:
    """One toolkit Button from one spec. No QSS, no stock widget: the toolkit owns the visuals.

    The spec's `size` and `icon_size` are DESIGN px and are handed to `Button` untouched: the toolkit multiplies them
    (`setFixedSize(scaled_px(w), scaled_px(h))`, `scaled_px(ctx.region_icon_size_px)`) and re-applies the design values
    on every `scale_changed`. Pre-scaling here made a 30 px mode button 120 device px at factor 2.0 instead of 60
    (measured 2026-11-28).

    `corner_radii` (design px, tl/tr/br/bl) nests the button into its group capsule; `extra_layers` are the toolkit's
    own paint hooks, used for the checked ring.

    A control emits exactly ONE verb: a toggle emits its command when it is checked (the family unchecks the rest), and
    a plain button on release. There is no hold/press-and-release pair any more - the compare is the pane's LMB gesture
    (`VideoPreviewSurface.mousePressEvent`), not a bar button (owner, 2026-11-25: "Зачем нужна кнопка compare?").
    """
    width, height = spec.size
    button = Button(
        icon=spec.icon,
        text=spec.text,
        toggle=spec.checkable,
        variant=spec.variant,
        icon_size=spec.icon_size,
        size=(width, height),
        corner_radii=corner_radii,
        extra_layers=extra_layers,
        parent=parent,
    )
    if spec.tip:
        # The app installs the toolkit's tooltip filter at start-up (`install_application_tooltips`), so this string is
        # rendered by the toolkit's own tooltip system - no hand-rolled tooltip widget.
        button.setToolTip(spec.tip)
    # A stable name for the control channel and the tests: `preview_<id>` addresses the button without coordinates.
    button.setObjectName(f"preview_{spec.id}")
    if spec.id == "play":
        # The transport's one accent control: an exact opaque base (a tint seed would wash out over the picture).
        # The toolkit's API is `set_background_color` (`set_override_bg_color` does not exist; the old call raised and
        # was swallowed, leaving the surface white under a white glyph - measured live 2026-11-25).
        from sli_ui_toolkit.theme import ThemeManager

        try:
            button.set_background_color(ThemeManager.get_instance().get_color("accent"))
        except Exception:
            pass
    return button


def checked_for(spec: ActionSpec, state: PreviewState) -> bool:
    return bool(spec.checked_when(state)) if spec.checked_when is not None else False


def enabled_for(spec: ActionSpec, state: PreviewState) -> bool:
    if state.phase.value == "closed":
        return False
    return bool(spec.enabled_when(state)) if spec.enabled_when is not None else True


def visible_for(spec: ActionSpec, state: PreviewState) -> bool:
    return bool(spec.visible_when(state)) if spec.visible_when is not None else True

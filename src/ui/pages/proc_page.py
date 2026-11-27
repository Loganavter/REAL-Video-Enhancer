"""Page `procPage` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from PySide6.QtCore import (QCoreApplication, QEvent, QObject, QPoint, QSize, Qt,
    QTimer, Signal)
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
from src.ui.layout_policy import (BODY_INSET_PX, COLUMN_DESIGN_PX, PANE_FLOOR_PX, PANE_GAP_PX,
    SIDE, STACK, column_floor_for_factor, layout_decision)
from src.ui.settings_store import PREVIEW_SPLIT_SIDE_KEY, PREVIEW_SPLIT_STACK_KEY, preview_split
from src.ui.pages.general_settings import GeneralSettingsPage
from src.ui.pages.tab_6 import Tab6Page
from src.ui.pages.render_queue_tab import RenderQueueTabPage
from src.ui.pages.advanced_settings import AdvancedSettingsPage
from src.ui.pages.tab_5 import Tab5Page
from src.ui.pages.handles import ProcPageHandles
from src.ui import i18n
from src.ui.typography import apply, CAPTION_PT
from src.ui.scale_follower import ScaleFollower
from src.ui.widgets.tab_strip import TAB_HEIGHT, tab_button, tab_style

# Layer B (2026-11-21): the three widget classes this page no longer builds stock. `TopTabHost` is the tab host, built
# with the stock `QTabWidget`'s strip geometry (the Designer's 22 px strip, no pane radius or inset, a 4 px tab corner);
# `ImageLabel` is the pixmap counterpart of `QLabel` (parent first, and the alignment is stated because Qt's default is
# left+vcenter, not the centre the preview label asks for).
#
# Layer E (page migration): `Text` where the Designer had a `QLabel`, `Button` where it had a `QPushButton`,
# `DoubleSpinBox` for a `QDoubleSpinBox` and `ToastProgressBar` for a `QProgressBar`. Nothing here needs a Qt-shaped
# compatibility adapter.
from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px
from sli_ui_toolkit.widgets import (Button, DoubleSpinBox, ImageLabel, MinimalistScrollBar, OverlayScrollArea,
                                    Splitter, SurfacePanel, TabStyle, Text, ToastProgressBar, TopTabHost)

#: Qt's spin-box text alignment (`QAbstractSpinBox().alignment()` is AlignLeft|AlignVCenter); the toolkit field
#: defaults to centre, which would move every value to the middle of its box.
SPIN_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

#: The process column's width in DESIGN px (owner, 2026-11-24): the desired width grows with the UI factor - 500 ->
#: 750 at 1.5 -> 1000 at 2.0 - and `layout_policy` decides whether it fits beside the preview or the page stacks. It is
#: a design value, so every resolve goes through `scaled_px()`/`layout_decision()` at apply time, never at import.
PROCESS_COLUMN_PX = COLUMN_DESIGN_PX

#: The transport row's own plate radius, in DESIGN px - the same 10 the video-info card uses, so the column reads as
#: one family of surfaces instead of cards plus a square band.
TRANSPORT_PLATE_RADIUS_PX = 10

#: The floating collapse toggle's inset from the page viewport's top-right corner, in DESIGN px.
OVERLAY_INSET_PX = 8

#: The transport buttons' ceiling in DESIGN px - the Designer's 59x55, and the same box the toolkit `Button` is built
#: with (`size=(59, 55)` -> `setFixedSize(scaled_px(...))`). The ceiling has to be resolved at the SAME factor, or a
#: raw 59x55 maximum against an 88x82 minimum leaves the layout centring a 55 px box in an 82 px cell.
TRANSPORT_BUTTON_MAX_WIDTH_PX = 59
TRANSPORT_BUTTON_MAX_HEIGHT_PX = 55

#: The transport glyphs' DESIGN box (the Designer's 40x45). A toolkit design value: `Button.setIconSize` stores it and
#: the style bridge multiplies it once at paint time, so the page passes it UNSCALED (30 design px paints 45 at 1.5 and
#: 60 at 2.0 - the same contract `download_page.ICON_BUTTON_ICON_PX` states).
TRANSPORT_GLYPH_WIDTH_PX = 40
TRANSPORT_GLYPH_HEIGHT_PX = 45

#: The preview band's aspect ratio (owner, 2026-11-24): in stacked mode the preview's height is 16:9 of its width,
#: never a share of the window. 16/9 is width/height, so the height is width * 9 / 16.
PREVIEW_BAND_ASPECT_W = 16
PREVIEW_BAND_ASPECT_H = 9

#: The splitter's handle as a DESIGN length. `layout_policy` pays `PANE_GAP_PX` of gap between the two panes at every
#: factor (its `side_cost`), so the handle is stated as the design value that resolves to that gap and never wider:
#: 6 px at 1.0/1.5/2.0, 5 px at 1.25/1.75. A handle that grew with the UI would take the preview under its 320 px
#: floor in the capped cells the policy measured (at 1.5 in the design window the column yields so the preview gets
#: exactly 320, which leaves no pixel to spare).
SPLIT_HANDLE_GAP_PX = PANE_GAP_PX

#: How long a finished drag waits before its split reaches `settings.txt`: one write per drag, not one per pixel.
SPLIT_SAVE_DELAY_MS = 400


class PageLayoutState(QObject):
    """The page's layout mode, as a signal its glue can listen to.

    `ProcPagePage` is a builder, not a `QObject`, so the one thing the glue needs from it - "the mode and the column
    changed, re-apply your half" - lives on this object, parented to the page root (Qt drops the connection with the
    widget). `toggleRequested` is the typed spelling of the band's toggle press: the page owns the state change, and
    `ProcessTab` only wires the button to this signal.
    """

    #: `(mode, column_px, preview_collapsed)` on every policy apply, including a resize that keeps the same mode.
    changed = Signal(str, int, bool)
    #: A request for one more policy pass. The page does not need it for its own widgets; the glue emits it after its
    #: construction changed the render row's hint (the idle state hides pause/kill), which is what sizes the footer.
    reapplyRequested = Signal()
    toggleRequested = Signal()


class _PageLayoutFilter(QObject):
    """Re-run the policy when the page viewport changes size, and keep the overlays on its bottom edge.

    The factor is not the only input: a user drag-resizes the window without touching the scale, and at a raised
    factor that changes which side of the 2.0 stack boundary the page is on. A `ScaleFollower` alone would leave the
    page in the mode it happened to build with. The splitter is watched too: its extent is the body's content width,
    and a pass that ran before the body was laid out would otherwise leave the panes at the proportion Qt chose for a
    size they no longer have.
    """

    def __init__(self, page: "ProcPagePage") -> None:
        super().__init__(page.procPage)
        self._page = page

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._page._apply_layout_policy()
        return False


# --- the tab strip's geometry policy (host side) ------------------------------------------------
#
# Owner, 2026-11-21: "the Log label is clipped and scroller arrows appear". Both halves of that are the strip's
# geometry, and the strip is a host component here: the toolkit's `TopTabBar` fixes each tab to its own `sizeHint`
# and the strip to the sum, so a bar narrower than that sum draws its last tabs past its own edge - they are not
# reachable at all, because the overflow bar is a HOVER-only affordance (it cannot be found by a user who does not
# already know it is there). The host's job is the answer, not the drawing: measure the labels, say whether they
# fit, and if they do not, expose the overflow where a pointer and a wheel can reach it and keep the SELECTED tab
# on screen (a selection you cannot see is the same defect as a label you cannot read).
#
# The arithmetic is a pure function with no Qt in it, so the answer for a strip can be pinned without a display.


@dataclass(frozen=True)
class TabStripFit:
    """Whether a strip of ``widths`` fits in ``available`` pixels, and by how much it does not."""

    widths: tuple[int, ...]
    available: int
    #: Everything in the strip that is not a label: the bar's own side margins plus the gaps between tabs.
    gap: int

    @property
    def needed(self) -> int:
        """The width the whole strip occupies at its natural tab widths."""
        return sum(self.widths) + self.gap

    @property
    def overflow(self) -> int:
        """Pixels of strip that the bar cannot show; 0 when every tab fits."""
        return max(0, self.needed - max(0, int(self.available)))

    @property
    def fits(self) -> bool:
        return self.overflow == 0


def fit_tab_strip(label_widths: Iterable[int], available: int, *, gap: int = 0) -> TabStripFit:
    """The strip's geometry: every tab at its natural width, plus how much of the strip does not fit.

    ``label_widths`` are the tab buttons' own hints (label + padding + any floor the style asks for), and ``gap`` is
    the strip's non-label width, so the answer is exactly "do the buttons fit in the bar" - no guessed constants
    that could drift away from the toolkit's own numbers. Callers that can widen the strip use ``fits``; a caller
    that cannot (this page's column is 500 design px, 750/1000 at 1.5/2.0) uses ``overflow`` to size and expose its
    affordance.
    """
    widths = tuple(max(0, int(width)) for width in label_widths)
    return TabStripFit(widths=widths, available=max(0, int(available)), gap=max(0, int(gap)))


def overflow_style(style):
    """The shared strip look with the overflow bar ALWAYS visible instead of hover-only.

    The toolkit reserves the bar's row (`reserve_scroll_row=True`) so its appearance cannot reflow the strip; what it
    does not do by default is SHOW the bar while the tabs overflow, and a bar that appears only under the pointer is
    not an affordance - it is a discovery puzzle. Everything else here is the shared style, untouched.
    """
    return replace(style, scroll_policy=Qt.ScrollBarPolicy.ScrollBarAlwaysOn, reserve_scroll_row=True)


class TabButtonFactory:
    """The shared tab-button look, remembering the buttons it built.

    `TopTabBar` exposes no per-tab button (its `TabItem` carries text and data only), and the host's own factory is
    the one place that sees what it builds - so the host keeps them, in tab order, for the geometry answer and for
    the selected plate. A strip is built once; nothing here rebuilds tabs.
    """

    def __init__(self) -> None:
        self.buttons: list = []

    def __call__(self, text: str, tab_height: int, style):
        button = tab_button(text, tab_height, style)
        self.buttons.append(button)
        return button


class TabStripHost(QObject):
    """The host half of a `TopTabHost`'s strip: measure it, expose its overflow, paint the selection from a token.

    Three things the toolkit leaves to the host, done in one place so the process and settings strips cannot drift:

    * **the answer** - `strip_fit()` reports whether the labels fit and how much does not (`fit_tab_strip`);
    * **reach** - any resize re-measures; a strip that overflows keeps the SELECTED tab scrolled into view, and a
      strip that fits drops any stale offset (a widened window must not leave the first tabs pushed out);
    * **the selection's colour** - the plate is the host's `TabStyle.selected_token`, resolved at paint time, so a
      theme restyles the selected tab without touching this page. The toolkit paints a host-built button from its
      own variant token (and recolours only the tab's TEXT from `accent`), so the plate is the host's to state.
    """

    def __init__(self, host, style, factory: TabButtonFactory) -> None:
        super().__init__(host)
        self._host = host
        self._style = style
        self._factory = factory
        self._bar = host.tabBar()
        self._bar.installEventFilter(self)
        host.currentChanged.connect(self._on_current_changed)
        try:
            from sli_ui_toolkit.theme import ThemeManager

            ThemeManager.get_instance().theme_changed.connect(self.refresh)
        except Exception:                     # a host without a theme manager still gets geometry
            pass
        self.refresh()

    # -- the answer -------------------------------------------------------------------------------

    def strip_fit(self) -> TabStripFit:
        """This strip, measured now: the buttons' natural hints and everything around them."""
        hints = [button.sizeHint().width() for button in self._factory.buttons]
        # The gap is what the bar needs on top of those buttons - its own margins and the spacing between tabs,
        # reported by the bar itself so no number is duplicated here.
        gap = max(0, self._bar.required_width() - sum(hints))
        return fit_tab_strip(hints, self._bar.width(), gap=gap)

    # -- reach ------------------------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Re-measure after anything that changes widths: resize, theme, a new selection."""
        fit = self.strip_fit()
        if fit.fits:
            self._bar.set_scroll_offset(0)
        else:
            self.reveal(self._host.currentIndex())
        self._paint_selection()

    def reveal(self, index: int) -> None:
        """Scroll the strip so tab ``index`` is whole inside the bar, if the strip is offset at all."""
        if not 0 <= index < len(self._factory.buttons):
            return
        button = self._factory.buttons[index]
        left = button.x()
        right = left + button.width()
        offset = self._bar.scroll_offset()
        width = self._bar.width()
        if left < offset:
            self._bar.set_scroll_offset(left)
        elif right > offset + width:
            self._bar.set_scroll_offset(right - width)

    def _on_current_changed(self, index: int) -> None:
        self.reveal(index)
        self._paint_selection()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if watched is self._bar and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.refresh()
        return super().eventFilter(watched, event)

    # -- the selection's colour -------------------------------------------------------------------

    def _paint_selection(self) -> None:
        """The selected tab's plate is the host's token, read now (not a colour frozen at build time)."""
        colour = None
        try:
            from sli_ui_toolkit.theme import ThemeManager

            colour = ThemeManager.get_instance().try_get_color(self._style.selected_token)
        except Exception:
            colour = None
        current = self._host.currentIndex()
        for index, button in enumerate(self._factory.buttons):
            selected = index == current
            # `set_override_bg_color` is the toolkit's exact-base-fill spelling: a checked button with an override
            # keeps the override as its plate (the variant's checked layer is skipped) and still answers hover.
            button.set_override_bg_color(QColor(colour) if (selected and colour is not None) else None)


class ProcPagePage:
    """`procPage` - placed by its parent with `addWidget`."""

    page_id = "procPage"

    def build(self, settings=None):
        """Build the widget tree and return the page root.

        ``settings`` is the settings owner (`src/ui/settings_store.Settings`), handed over by the window that builds
        the pages - it is where the preview splitter's two per-mode splits are read and written. A page built on its
        own (a test, the capture walker) passes nothing and keeps the design split without persisting anything.
        """
        self._settings = settings
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        # splitter: skipped 'sizePolicy1.setHeightForWidth(self.mainWindowContainer.sizePolicy().hasHeightForWidth())' (['mainWindowContainer'] live outside this page)
        # splitter: skipped 'sizePolicy7.setHeightForWidth(self.inputFileText.sizePolicy().hasHeightForWidth())' (['inputFileText'] live outside this page)
        # splitter: skipped 'sizePolicy7.setHeightForWidth(self.scrollArea_6.sizePolicy().hasHeightForWidth())' (['scrollArea_6'] live outside this page)
        # splitter: skipped 'sizePolicy1.setHeightForWidth(self.renderQueueTab.sizePolicy().hasHeightForWidth())' (['renderQueueTab'] live outside this page)
        self.procPage = SurfacePanel(background_token="surface.page")
        self.procPage.setObjectName(u"procPage")
        # The page body scrolls (owner, 2026-11-24). At a raised factor the content is taller than the window and the
        # process column wider than the page, so the alternative to a scroll surface is squeezing (which is what the
        # owner reported) or clipping. The surface is the toolkit's own `OverlayScrollArea`, not a stock `QScrollArea`:
        # this page paints from tokens and a stock area brings the widget style back. `content_inset=0` because there is
        # no stock frame here to reproduce (any inset would move the 1.0 page by a pixel), and
        # `reserve_scrollbar_space=False` because the bar is an overlay - it must not narrow the columns when it
        # appears. `corner_radius=0`: the page is a square rect and the bar's corner follows it.
        self.pageScroll = OverlayScrollArea(self.procPage, content_inset=0, reserve_scrollbar_space=False,
                                            corner_radius=0)
        self.pageScroll.setObjectName(u"procPageScroll")
        self.pageScroll.setWidgetResizable(True)
        # The content is a toolkit container, not a bare `QWidget` wrapper: it paints nothing (`background_token=None`)
        # so the page's own `surface.page` shows through, and the tree reads page -> scroll -> body -> columns.
        self.pageScrollBody = SurfacePanel(self.pageScroll, background_token=None)
        self.pageScrollBody.setObjectName(u"procPageScrollBody")
        # Layer C: the page body paints itself (`surface.page`); every control rule this sheet carried styled a widget
        # that is now a toolkit control painting from tokens. The one rule kept is the read-only rich-text panel on
        # the General tab, which is still a stock `QTextEdit` (a toolkit row: no rich-text pane with a scroll host).
        # No page stylesheet: every control on this page (and its tabs) paints itself from tokens. The read-only
        # rich-text panes are the toolkit's `RichTextView`, which carries its own plate.
        self.gridLayout_3 = QGridLayout(self.pageScrollBody)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.processInfoContainer = SurfacePanel(self.pageScrollBody, background_token="dialog.background", radius=10)
        self.processInfoContainer.setObjectName(u"processInfoContainer")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.processInfoContainer.sizePolicy().hasHeightForWidth())
        self.processInfoContainer.setSizePolicy(sizePolicy3)
        # The pane's own surface (a token, not an id rule): `dialog.background` is the #1f232a the rule carried.
        self.verticalLayout_5 = QVBoxLayout(self.processInfoContainer)
        self.verticalLayout_5.setSpacing(9)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(9, 9, 9, 9)
        self.previewSubContainer = SurfacePanel(self.processInfoContainer, background_token="dialog.background")
        self.previewSubContainer.setObjectName(u"previewSubContainer")
        #: The typed name for the thing wave 3 treats as the stacked band (the preview's own container, not the whole
        #: info pane): its height is what 16:9 applies to.
        self.previewBand = self.previewSubContainer
        self.verticalLayout_8 = QVBoxLayout(self.previewSubContainer)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        # The stock label's own sheet was `QLabel{color:#fff; background-color:#1f232a}`: the colour had no subject (a
        # pixmap label has no text channel) and the background is `#previewSubContainer`'s own plate, painted by the
        # container's QSS, so the label keeps no surface of its own and the parent shows through.
        # The pane is the GPU surface (plan-rve-video-preview-widget-2026-11-23): a full-resolution frame with zoom,
        # the strip and its markers, all in one QRhi pass. `RVE_UI_RASTER_PREVIEW=1` keeps the old pixmap label for one
        # A/B - the fallback exists because a first QRhiWidget changes the window's compositing path on this stack.
        import os as _preview_env

        from src.ui.rhi_surface import gpu_preview_available

        # The raster pane is not only an A/B switch any more: when the process has no usable GL/EGL context every
        # QRhiWidget renders nothing (`No QRhi`), so the page falls back on its own instead of showing an empty plate.
        if not gpu_preview_available():
            self.previewLabel = ImageLabel(self.previewSubContainer,
                alignment=Qt.AlignmentFlag.AlignCenter)
            self.previewLabel.setObjectName(u"previewLabel")
            self.previewLabel.setScaledContents(False)
            self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # `RVE_UI_PREVIEW_LABEL=default|opaque|fill|hide|small` cuts THIS label down one property at a time.
            from src.ui.devtools import label_bisect

            label_bisect.cut(self.previewLabel)
            self.preview_label_is_surface = False
        else:
            from src.ui.widgets.video_preview import VideoPreviewSurface

            self.previewLabel = VideoPreviewSurface(self.previewSubContainer)
            self.previewLabel.setObjectName(u"previewLabel")
            # Ask for the configured backend explicitly and make a failure name itself (`RVE_UI_RHI_BACKEND`).
            from src.ui.rhi_surface import configure_rhi_widget

            configure_rhi_widget(self.previewLabel)
            # The pane is opaque (an RHI widget cannot alpha-blend with its siblings) and shows this colour wherever no
            # frame is drawn: the same token the plate under it uses, so an empty pane reads as part of the page.
            from sli_ui_toolkit.theme import ThemeManager

            self.previewLabel.set_background_colour(
                ThemeManager.get_instance().try_get_color("dialog.background") or "#24252a")
            self.preview_label_is_surface = True
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.previewLabel.sizePolicy().hasHeightForWidth())
        self.previewLabel.setSizePolicy(sizePolicy4)
        self.verticalLayout_8.addWidget(self.previewLabel)
        # The preview's transport: one toggle, and the preview stands PAUSED until it is pressed (owner, 2026-11-23:
        # "добавь паузу и чтобы изначально на ней и стояло"). It lives next to the picture rather than in
        # `RenderedPreviewControlsContainer`, which is hidden at start-up and therefore cannot carry a control a live
        # preview needs. Checked = paused; the glyph is the ACTION the press performs, like a player's own button.
        self.previewPauseButton = Button(
            icon=(QIcon(u":/icons/icons/pause-circle.svg"), QIcon(u":/icons/icons/play-circle.svg")),
            toggle=True, variant="surface",
            content_align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(8, 5, 8, 5), corner_radius=10, size=(40, 40), icon_size=26)
        self.previewPauseButton.setObjectName(u"previewPauseButton")
        self.previewPauseButton.setChecked(True)
        self.previewPauseButton.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Straight into the pane's own column, aligned left: a wrapper `QWidget` around one button would be a bare
        # container (the toolkit-native ratchet counts them, and proc_page has no budget for a new one).
        self.verticalLayout_8.addWidget(self.previewPauseButton, 0, Qt.AlignmentFlag.AlignLeft)
        self.VideoPreview = QWidget(self.previewSubContainer)
        self.VideoPreview.setObjectName(u"VideoPreview")
        self.VideoPreview.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.VideoPreview.sizePolicy().hasHeightForWidth())
        self.VideoPreview.setSizePolicy(sizePolicy1)
        self.horizontalLayout_26 = QHBoxLayout(self.VideoPreview)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.verticalLayout_8.addWidget(self.VideoPreview)
        self.verticalLayout_5.addWidget(self.previewSubContainer)
        self.RenderedPreviewControlsContainer = QWidget(self.processInfoContainer)
        self.RenderedPreviewControlsContainer.setObjectName(u"RenderedPreviewControlsContainer")
        self.RenderedPreviewControlsContainer.setEnabled(True)

        self.horizontalLayout_47 = QHBoxLayout(self.RenderedPreviewControlsContainer)
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.widget_30 = QWidget(self.RenderedPreviewControlsContainer)
        self.widget_30.setObjectName(u"widget_30")
        self.verticalLayout_37 = QVBoxLayout(self.widget_30)
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.label_9 = Text(parent=self.widget_30)
        self.label_9.setObjectName(u"label_9")
        self.verticalLayout_37.addWidget(self.label_9)
        # `single_step=0.1` is the toolkit field's "clamp and round to the display precision" semantic (its `setValue`
        # SNAPS to `single_step`, which at the default 1.0 would turn 3.5 into 3.0); `decimals=2` is what the Designer's
        # stock `QDoubleSpinBox` displayed and `alignment` is Qt's own (the toolkit field defaults to centre).
        self.startTimeSpinBox = DoubleSpinBox(self.widget_30, single_step=0.1, decimals=2, show_step_buttons=True,
                                              step_button_width=20, field_height=29, alignment=SPIN_ALIGNMENT)
        self.startTimeSpinBox.setObjectName(u"startTimeSpinBox")
        self.verticalLayout_37.addWidget(self.startTimeSpinBox)
        self.label_52 = Text(parent=self.widget_30)
        self.label_52.setObjectName(u"label_52")
        self.verticalLayout_37.addWidget(self.label_52)
        self.endTimeSpinBox = DoubleSpinBox(self.widget_30, single_step=0.1, decimals=2, show_step_buttons=True,
                                            step_button_width=20, field_height=29, alignment=SPIN_ALIGNMENT)
        self.endTimeSpinBox.setObjectName(u"endTimeSpinBox")
        # Qt's `setMinimum` is the one range spelling the toolkit field does not answer (`DoubleSpinBox` carries
        # `setRange` only), so the Designer's minimum plus Qt's own default maximum (99.99) are one call.
        self.endTimeSpinBox.setRange(1.000000000000000, 99.990000000000000)
        self.endTimeSpinBox.setValue(1.000000000000000)
        self.verticalLayout_37.addWidget(self.endTimeSpinBox)
        self.widget_14 = QWidget(self.widget_30)
        self.widget_14.setObjectName(u"widget_14")
        self.horizontalLayout_48 = QHBoxLayout(self.widget_14)
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.horizontalLayout_48.setContentsMargins(0, 0, 0, 0)
        self.renderPreviewBtn = Button(parent=self.widget_14, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.renderPreviewBtn.setObjectName(u"renderPreviewBtn")
        apply(self.renderPreviewBtn, None)
        # QPushButton's own policy (as wide as it needs, exactly as tall as its hint): the toolkit button is happy to
        # fill the cell, which turns a row button into a slab when the column is taller than its contents.
        self.renderPreviewBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.renderPreviewBtn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.horizontalLayout_48.addWidget(self.renderPreviewBtn)
        self.verticalLayout_37.addWidget(self.widget_14)
        # The scrub bar is the toolkit's scrollbar, not a stock `QScrollBar`: same value API (`value`/`setValue`/
        # `setRange`/`setMaximum`/`valueChanged`/`setEnabled` - `MinimalistScrollBar` subclasses `QScrollBar`), but the
        # thumb is painted from the theme (`accent` while dragging, a translucent white handle otherwise) instead of
        # from the window's deleted `QScrollBar::handle` rule. Orientation is the constructor's first argument, which
        # replaces the Designer's `setOrientation(Horizontal)` statement.
        self.timeInVideoScrollBar = MinimalistScrollBar(Qt.Orientation.Horizontal, self.widget_30)
        self.timeInVideoScrollBar.setObjectName(u"timeInVideoScrollBar")
        self.verticalLayout_37.addWidget(self.timeInVideoScrollBar)
        self.horizontalLayout_47.addWidget(self.widget_30)
        self.verticalLayout_5.addWidget(self.RenderedPreviewControlsContainer)
        self.processContainer = QWidget(self.pageScrollBody)
        self.processContainer.setObjectName(u"processContainer")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.processContainer.sizePolicy().hasHeightForWidth())
        self.processContainer.setSizePolicy(sizePolicy5)
        # The column is STATED at the factor in force, not merely capped: the owner asked for 500/750/1000 exactly, and
        # the content's own hint at 1.5 (668 px) is narrower than 750, so a maximum alone would leave the column short.
        # A narrower window is carried by the page scroll area above, not by squeezing this column.
        # The width is now the SPLIT's to state (`_apply_split_sizes`): the design 500 is the default split, and a drag
        # moves it between the pane floors. Only the two strips INSIDE the column keep an explicit width, because their
        # content hints (the tab strip's five labels) are wider than the column at 1.0.
        # No stylesheet: a disabled toolkit control takes its colour from its token. The rule only tinted the two stock
        # queue lists, which carry their own look until the toolkit grows a compact list.
        self.verticalLayout_7 = QVBoxLayout(self.processContainer)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 10, 0, 0)
        # NO fill on the column body: the only thing that ever showed of it was the band behind the transport row
        # (owner, 2026-11-28: "сделай подложку скругленной и убери снизу лишний пад"). The row carries its own rounded
        # plate now (`widget_13`), so a full-height `dialog.background` under it would paint a square band with dead
        # space below the controls.
        self.bottomMenuSubContainer = SurfacePanel(self.processContainer, background_token=None)
        self.bottomMenuSubContainer.setObjectName(u"bottomMenuSubContainer")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.bottomMenuSubContainer.sizePolicy().hasHeightForWidth())
        self.bottomMenuSubContainer.setSizePolicy(sizePolicy6)
        self.bottomMenuSubContainer.setMinimumSize(QSize(0, 0))
        # The width is stated by `_apply_column_widths` from the live split (this strip was fixed to the design column
        # before the splitter existed, and its content hint alone - the five tab labels - would win over a narrower
        # column).
        self.bottomMenuSubContainer.setFixedWidth(scaled_px(PROCESS_COLUMN_PX))

        self.verticalLayout_21 = QVBoxLayout(self.bottomMenuSubContainer)
        self.verticalLayout_21.setSpacing(0)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.widget_2 = SurfacePanel(self.bottomMenuSubContainer, background_token="surface.page")
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_24 = QVBoxLayout(self.widget_2)
        self.verticalLayout_24.setSpacing(0)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(0, 0, 0, 0)
        # The original interface's tabs are BUTTONS standing apart from the content: no folder outline, no join cover,
        # a plate on the selected one and a gap under the strip. `TabStyle` is the toolkit's styling surface for that.
        # The only difference from the shared style is the overflow row's policy (`overflow_style`): this column is 500
        # design px, so five labels do not fit at a raised factor (and at 1.0 in a long translation), and an affordance
        # that waits for the pointer is not one.
        strip_style = overflow_style(tab_style())
        strip_buttons = TabButtonFactory()
        self.processSettingsContainer = TopTabHost(self.widget_2, tab_height=TAB_HEIGHT, pane_radius=0, pane_inset=(10, 5, 10, 5),
            tab_factory=strip_buttons,
            tab_style=strip_style,
            chrome_inset=0, tab_radius=4, pane_fill_token=None)
        self.processSettingsContainer.setObjectName(u"processSettingsContainer")
        sizePolicy3.setHeightForWidth(self.processSettingsContainer.sizePolicy().hasHeightForWidth())
        self.processSettingsContainer.setSizePolicy(sizePolicy3)
        self.processSettingsContainer.setMinimumSize(QSize(0, 0))
        self.processSettingsContainer.setMaximumSize(QSize(16777215, 16777215))
        self.generalSettings_page = GeneralSettingsPage()
        self.generalSettings = self.generalSettings_page.build()
        self.processSettingsContainer.addTab(self.generalSettings, "")
        self.tab_6_page = Tab6Page()
        self.tab_6 = self.tab_6_page.build()
        self.processSettingsContainer.addTab(self.tab_6, "")
        self.renderQueueTab_page = RenderQueueTabPage()
        self.renderQueueTab = self.renderQueueTab_page.build()
        self.processSettingsContainer.addTab(self.renderQueueTab, "")
        self.advancedSettings_page = AdvancedSettingsPage()
        self.advancedSettings = self.advancedSettings_page.build()
        self.processSettingsContainer.addTab(self.advancedSettings, "")
        self.tab_5_page = Tab5Page()
        self.tab_5 = self.tab_5_page.build()
        self.processSettingsContainer.addTab(self.tab_5, "")
        self.verticalLayout_24.addWidget(self.processSettingsContainer)
        # The settings block TAKES the column's leftover height (stretch 1), and the transport row keeps its own height
        # at the bottom: the content area reaches the row instead of stopping at its hint and leaving a gap
        # (owner, 2026-11-28: "почему-то окно контента не уехало вслед за виджетом вниз ... не растянулось").
        self.verticalLayout_21.addWidget(self.widget_2, 1)
        # NO panel behind the transport row (owner, 2026-11-27: "нужно ещё линию под контролами убрать - она инфу
        # дублирует"). The row is a control strip, not a card: its `surface.page` fill showed as a rounded band under the
        # buttons and the progress bar, carrying nothing.
        # The transport row IS the plate: rounded like every other card on this page (owner, 2026-11-28), sized to its
        # controls - it must not stretch to swallow the column's leftover height, which is what put dead padding under
        # the buttons (`verticalLayout_21`'s trailing stretch takes that space instead).
        self.widget_13 = SurfacePanel(self.bottomMenuSubContainer, background_token="dialog.background",
                                      radius=scaled_px(TRANSPORT_PLATE_RADIUS_PX))
        self.widget_13.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.widget_13.setObjectName(u"widget_13")
        self.widget_13.setFixedWidth(scaled_px(PROCESS_COLUMN_PX))
        #: The typed name for the render row (Start/progress row): in stacked mode it moves into the sticky
        #: footer, in side mode it stays in the column's flow.
        self.renderRow = self.widget_13
        self.horizontalLayout_32 = QHBoxLayout(self.widget_13)
        self.horizontalLayout_32.setSpacing(6)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.horizontalLayout_32.setContentsMargins(9, 9, 9, 9)
        self.widget_24 = QWidget(self.widget_13)
        self.widget_24.setObjectName(u"widget_24")
        self.horizontalLayout_39 = QHBoxLayout(self.widget_24)
        self.horizontalLayout_39.setSpacing(0)
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.horizontalLayout_39.setContentsMargins(0, 0, 0, 0)
        # `toggle=True`: the transport is one button whose glyph IS the state (unchecked play, checked pause), so it
        # has to be checkable - without it `setChecked` is a silent no-op and the icon never swaps.
        self.startRenderButton = Button(parent=self.widget_24, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(59, 55))
        self.startRenderButton.setObjectName(u"startRenderButton")
        self.startRenderButton.setEnabled(True)
        self.startRenderButton.setMaximumSize(
            QSize(scaled_px(TRANSPORT_BUTTON_MAX_WIDTH_PX), scaled_px(TRANSPORT_BUTTON_MAX_HEIGHT_PX)))
        # The Designer pinned this button at 59x55 DESIGN px. The toolkit scales that box itself in the constructor
        # (`size=(59, 55)` -> `setFixedSize(scaled_px(...))`), so the ceiling must be stated at the same factor: a raw
        # 59x55 maximum against a 88x82 minimum left the layout centring a 55 px box in an 82 px cell and the button
        # hung 13 px (27 px at 2.0) below its row. Both bounds now resolve through `scaled_px`, like the toolkit's own.

        # QPushButton's own policy: as wide as it needs, exactly as tall as its hint - the toolkit button would
        # otherwise fill the cell and grow past the Designer's 55 px ceiling in a taller row.
        self.startRenderButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # ONE transport button (owner, 2026-11-24): play and pause are the same control seen twice - the glyph is the
        # ACTION the press performs, exactly like the preview's own transport (`icon=(play, pause)` = unchecked,
        # checked). Checked means "a job is running"; `ProcessTab._set_render_controls` is the one writer of that state.
        self.startRenderButton.setIcon((QIcon(u":/icons/icons/play-circle.svg"),
                                        QIcon(u":/icons/icons/pause-circle.svg")))
        self.startRenderButton.setIconSize(QSize(TRANSPORT_GLYPH_WIDTH_PX, TRANSPORT_GLYPH_HEIGHT_PX))
        self.startRenderButton.setChecked(False)
        self.startRenderButton.setAutoRepeat(False)
        self.startRenderButton.setAutoExclusive(False)
        self.horizontalLayout_39.addWidget(self.startRenderButton)
        self.horizontalLayout_32.addWidget(self.widget_24)
        self.onRenderButtonsContiainer = QHBoxLayout()
        self.onRenderButtonsContiainer.setObjectName(u"onRenderButtonsContiainer")
        self.killRenderButton = Button(parent=self.widget_13, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.killRenderButton.setObjectName(u"killRenderButton")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy11.setHorizontalStretch(0)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.killRenderButton.sizePolicy().hasHeightForWidth())
        self.killRenderButton.setSizePolicy(sizePolicy11)
        self.killRenderButton.setMaximumSize(
            QSize(scaled_px(TRANSPORT_BUTTON_MAX_WIDTH_PX), scaled_px(TRANSPORT_BUTTON_MAX_HEIGHT_PX)))
        icon5 = QIcon()
        icon5.addFile(u":/icons/icons/x-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.killRenderButton.setIcon(icon5)
        self.killRenderButton.setIconSize(QSize(TRANSPORT_GLYPH_WIDTH_PX, TRANSPORT_GLYPH_HEIGHT_PX))
        self.onRenderButtonsContiainer.addWidget(self.killRenderButton)
        self.horizontalLayout_32.addLayout(self.onRenderButtonsContiainer)
        # NO retained boxes (owner, 2026-11-24, asked directly): keeping the boxes of the hidden pause/kill pair made
        # the row stable across states, but it also reserved their 84 px between the start button and the bar - the bar
        # sat at x=158 for a 333 px width, while it gets 417 px when the reservation is gone. The owner prefers the
        # longer idle bar and accepts the one re-flow at the moment a job starts. The gate that pinned the stable row
        # now pins this choice instead (`harness/tests/test_render_row_geometry.py`).
        # `height=25` is the Designer's bar row; the toolkit toast bar's own default is 6 px. Its track paints from
        # the toolkit's `toast.progress.background` token (the adapter's call-site token was `progress.track.background`).
        self.progressBar = ToastProgressBar(self.widget_13, height=25, track_token="progress.track.background")
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy7.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy7)
        self.progressBar.setValue(0)
        # `setTextVisible(False)` is gone with the stock bar: `ToastProgressBar` paints no label at all, so that state
        # is the one it is always in.
        self.horizontalLayout_32.addWidget(self.progressBar)
        self.verticalLayout_21.addWidget(self.widget_13)
        # There is no status line here. A `progressDetail` caption lived under the row (2026-11-22) and then under the
        # render card, and the owner has now removed it for good (2026-11-28: "подписи надо убрать было"): the run
        # card already states the frame counter, the percent, the remaining time, fps and the elapsed clock, so the
        # line was a duplicate under the controls. `ProcessTab` no longer produces it.
        #
        # (The column's leftover height is already carried by the stretch added above the row.)
        self.verticalLayout_7.addWidget(self.bottomMenuSubContainer)
        # The two panes are NOT placed in the grid here: the splitter that owns the boundary is built by the first
        # policy pass (`_split_for`), which adds one item spanning the body's two content columns. Placing them first
        # and then moving them into the splitter would be a second, transient placement of the same two widgets.
        # The body is complete, so it becomes the scroll surface's widget and the surface fills the page. The page
        # itself keeps its rect (`MainWindowShell` lays pages out for the stack's contents rect); scrolling happens
        # inside it.
        self.pageScroll.setWidget(self.pageScrollBody)
        self.verticalLayout_page = QVBoxLayout(self.procPage)
        self.verticalLayout_page.setObjectName(u"verticalLayout_page")
        self.verticalLayout_page.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_page.addWidget(self.pageScroll)
        # The sticky footer and the band's collapse toggle are children of the PAGE, not of the scrolling body: they
        # must stay on the viewport's edges while the body scrolls. Both are hidden in side mode, so the 1.0 page keeps
        # its flow geometry exactly. `widget_13` moves between the column's flow and this footer (`_apply_render_row`).
        self.renderRowFooter = SurfacePanel(self.procPage, background_token="dialog.background", radius=0)
        self.renderRowFooter.setObjectName(u"renderRowFooter")
        self.renderRowFooterLayout = QHBoxLayout(self.renderRowFooter)
        self.renderRowFooterLayout.setObjectName(u"renderRowFooterLayout")
        self.renderRowFooterLayout.setContentsMargins(0, 0, 0, 0)
        self.renderRowFooterLayout.setSpacing(0)
        self.renderRowFooter.hide()
        # The collapse chevron floats at the viewport's top-right in stacked mode. It lives outside the pane on purpose:
        # collapsing hides the band, and a toggle inside the hidden band could never expand it again. The glyph is the
        # action the press performs, like the preview's own transport button.
        self.previewBandToggle = Button(
            parent=self.procPage, text=u"\u25be", variant="surface",
            content_align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(8, 2, 8, 2), corner_radius=8, size=(0, 0))
        self.previewBandToggle.setObjectName(u"previewBandToggle")
        self.previewBandToggle.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.previewBandToggle.hide()
        # The policy's state: the mode, the column it asked for, and whether the band is collapsed. `ProcPageHandles`
        # exposes the state object so `ProcessTab` can wire the toggle without reaching into the builder.
        self._layout_mode = SIDE
        self._policy_column = scaled_px(PROCESS_COLUMN_PX)
        self._preview_collapsed = False
        self._applying_layout = False
        self._base_grid_bottom = self.gridLayout_3.contentsMargins().bottom()
        # The boundary between the preview and the column: ONE live splitter, rebuilt when a mode needs the other
        # orientation (Qt cannot re-orient a `QSplitter`). Its sizes are re-applied on every policy pass, so a resize
        # and a scale change both keep the split. `_settings` is the owner the window hands over; a page built without
        # one keeps the design split and stores nothing.
        self._split = None
        self._split_orientation = None
        self._applying_split = False
        self._pending_split: dict[str, str] = {}
        self._split_cache: dict[str, tuple[str, int | None]] = {}
        self._split_flush = QTimer(self.procPage)
        self._split_flush.setSingleShot(True)
        self._split_flush.setInterval(SPLIT_SAVE_DELAY_MS)
        self._split_flush.timeout.connect(self._flush_split)
        self.layoutState = PageLayoutState(self.procPage)
        self.previewBandToggle.clicked.connect(self.layoutState.toggleRequested.emit)
        self.layoutState.toggleRequested.connect(self.toggle_preview_band)
        self.layoutState.reapplyRequested.connect(self._apply_layout_policy)
        # The geometry is re-resolved at construction and re-applied on every factor change AND on every page resize (a
        # window drag changes the viewport without a factor change), so `scale:<factor>` and a resize both move it
        # without a restart. The filter is created first: the first policy pass builds the splitter, and the splitter
        # is one of the widgets whose resize re-runs the pass.
        self._layout_filter = _PageLayoutFilter(self)
        self.procPage.installEventFilter(self._layout_filter)
        self.pageScroll.viewport().installEventFilter(self._layout_filter)
        self._column_scale = ScaleFollower(self.procPage, self._apply_layout_policy)
        self.processSettingsContainer.setCurrentIndex(0)
        # The strip's host half (see `TabStripHost`): measures the labels, exposes an overflow the user can reach,
        # and paints the selected plate from the host's token. It must come after the labels are set, or it would
        # measure placeholders; `retranslate()` below re-runs `refresh()` for the final texts.
        self._tab_strip = TabStripHost(self.processSettingsContainer, strip_style, strip_buttons)
        # The glue's typed edge over this page's own chrome (the tabs have their own groups). Required fields: a
        # missing widget is a TypeError here, not a silent `None` at first render.
        self._handles = ProcPageHandles(
            startRenderButton=self.startRenderButton,
            killRenderButton=self.killRenderButton,
            onRenderButtonsContiainer=self.onRenderButtonsContiainer,
            previewLabel=self.previewLabel,
            previewPauseButton=self.previewPauseButton,
            processSettingsContainer=self.processSettingsContainer,
            renderQueueTab=self.renderQueueTab,
            progressBar=self.progressBar,
            VideoPreview=self.VideoPreview,
            RenderedPreviewControlsContainer=self.RenderedPreviewControlsContainer,
            renderPreviewBtn=self.renderPreviewBtn,
            startTimeSpinBox=self.startTimeSpinBox,
            endTimeSpinBox=self.endTimeSpinBox,
            timeInVideoScrollBar=self.timeInVideoScrollBar,
            previewBand=self.previewBand,
            previewBandToggle=self.previewBandToggle,
            renderRow=self.renderRow,
            renderRowFooter=self.renderRowFooter,
            layoutState=self.layoutState,
        )
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.procPage)
        self.retranslate()
        return self.procPage

    # -- the layout policy: cascade, then stack ---------------------------------------------------------------

    @property
    def layout_mode(self) -> str:
        """The mode the policy last applied: `"side"` or `"stack"` (`SIDE`/`STACK` in `layout_policy`)."""
        return self._layout_mode

    @property
    def policy_column(self) -> int:
        """The column's applied width (the same number on all three column widgets).

        It is the policy's answer unless the user dragged the boundary: a stored/saved split wins inside the pane
        floors, and this property then reports the width actually in force.
        """
        return self._policy_column

    @property
    def preview_collapsed(self) -> bool:
        """Whether the stacked preview band is currently collapsed."""
        return self._preview_collapsed

    @property
    def preview_splitter(self):
        """The live splitter between the preview and the column (`None` before the first policy pass)."""
        return self._split

    def _column_floor_px(self) -> int:
        """The column's content floor, from the factor - never from the live content hint.

        The General tab's hint moves with the width it is measuring (618 px at a 1000 px column, 603 px at the 647 px
        column the same content would get), so feeding it back would let a single resize flip the 2.0 decision. The
        factor form is the measured 650 at 2.0/575 at 1.75 resolved through the same scale as the column.
        """
        return column_floor_for_factor(UiScale.get_instance().factor())

    def _column_min_height_px(self) -> int:
        """The stacked column's own content floor in HEIGHT: what its tab host and its row need to be whole.

        The width floor above is a different question (it is the General tab's reflow); in `stack` the column is full
        width and the boundary moves vertically, so the floor that stops a drag is this one. A column at it scrolls its
        tab pane inside itself, which is the tab host's own designed behaviour.
        """
        return max(1, self.processContainer.minimumSizeHint().height())

    def _apply_layout_policy(self) -> None:
        """Decide `(mode, column)` from the live viewport and apply it to both panes and the page chrome.

        Called at build, on every `UiScale.scale_changed`, and on every viewport resize (the factor is not the only
        input: a window drag changes the viewport too). The guard keeps the layout writes from re-entering through
        the resize events they cause.
        """
        if self._applying_layout:
            return
        self._applying_layout = True
        try:
            viewport = self.pageScroll.viewport().width()
            mode, column = layout_decision(
                UiScale.get_instance().factor(),
                viewport,
                max(PANE_FLOOR_PX, self.processInfoContainer.minimumSizeHint().width()),
                self._column_floor_px(),
            )
            # A collapsed band is the same cascade with the preview out of the sum: the column takes the body's whole
            # content width. In side mode this is the candidate the owner asked for (the column widens); in stack mode
            # it was already full width, and collapsing reclaims the vertical space.
            if self._preview_collapsed:
                mode = STACK
                column = max(0, viewport - BODY_INSET_PX)
            # The splitter first: it owns the boundary, and the sizes applied below are the ones the panes then have.
            self._split_for(mode)
            self._apply_preview_band(mode, column)
            self._apply_split_floors(mode)
            column = self._apply_split_sizes(mode, column)
            self._apply_render_row(mode)
            self._layout_mode = mode
            self._policy_column = column
            self.previewBandToggle.setVisible(mode == STACK)
            self.layoutState.changed.emit(mode, column, self._preview_collapsed)
            self._position_overlays()
        finally:
            self._applying_layout = False

    # -- the boundary: one splitter, one owner ----------------------------------------------------------------

    def _split_handle_design_px(self) -> int:
        """The handle's DESIGN width: the gap the policy pays, resolved so the handle is never wider than it.

        `layout_policy` charges `PANE_GAP_PX` between the panes at every factor (its `side_cost`), so the handle must
        not grow with the UI: 6 px at 1.0/1.5/2.0 and 5 px at 1.25/1.75. The toolkit re-resolves a design width through
        `scaled_px`, so the design value is the gap divided by the factor.
        """
        factor = max(1e-9, UiScale.get_instance().factor())
        return max(1, int(SPLIT_HANDLE_GAP_PX / factor))

    def _split_for(self, mode: str):
        """The live splitter for `mode`, rebuilt when the mode needs the other orientation.

        Qt cannot re-orient a `QSplitter`, so a mode change that needs the other axis detaches both panes, removes and
        destroys the old splitter, and builds the right one. The page therefore never holds two splitters, and never
        holds one whose orientation disagrees with the mode it was last applied for.
        """
        orientation = Qt.Orientation.Horizontal if mode == SIDE else Qt.Orientation.Vertical
        splitter = self._split
        if splitter is not None and self._split_orientation == orientation:
            splitter.set_handle_width(self._split_handle_design_px())
            return splitter
        if splitter is not None:
            # The panes leave first (a new parent takes them out of the splitter's own list), then the stale splitter
            # goes: it is removed from the grid it no longer occupies and deleted on the next turn.
            self.gridLayout_3.removeWidget(splitter)
            for pane in (self.processContainer, self.processInfoContainer):
                pane.setParent(self.pageScrollBody)
            splitter.hide()
            splitter.deleteLater()
            self._split = None
            self._split_orientation = None
        splitter = Splitter(orientation, self.pageScrollBody,
                            handle_width=self._split_handle_design_px(), children_collapsible=False)
        splitter.setObjectName(u"previewSplitter")
        # The child order IS the layout order. `side` keeps the column on the LEFT - the cell the grid gave it before
        # this wave - with the preview taking the remainder; `stack` puts the preview band above the column.
        panes = ((self.processContainer, self.processInfoContainer) if orientation == Qt.Orientation.Horizontal
                 else (self.processInfoContainer, self.processContainer))
        for pane in panes:
            splitter.addWidget(pane)
        splitter.splitterMoved.connect(self._on_split_moved)
        # The splitter's extent is the body's content width/height: when the grid hands it a new box (a mode switch,
        # a resize) the sizes are re-stated for that box instead of left at the proportion Qt scaled them to.
        splitter.installEventFilter(self._layout_filter)
        self.gridLayout_3.addWidget(splitter, 0, 1, 1, 2)
        self._split = splitter
        self._split_orientation = orientation
        return splitter

    def _apply_split_floors(self, mode: str) -> None:
        """State each pane's floor and policy for `mode`; the splitter clamps a drag at them.

        Nothing is clamped after the fact: `QSplitter` honours its children's minimum sizes, so a drag that would
        collapse either side simply stops there. In `side` the floors are widths - the preview's 320 px usability floor
        and the column's own content floor (the width the General tab needs to stop scrolling inside itself) - and the
        panes keep the column's mode-dependent VERTICAL policy (owner, 2026-11-24: the render row and the status line
        must always be visible). `Minimum` has no ShrinkFlag, so `qSmartMinSize` takes the column's size hint as its
        minimum: in `side` the column shares its row with the preview and can yield to the tab pane's own scroll
        (`Preferred`, which has ShrinkFlag), while in `stack` it IS the page's height and keeps `Minimum`. In `stack`
        the widths are the body's, so the floors are heights: the preview's own content height (a shorter pane would
        clip the fixed 16:9 band and its controls) and the column's own content floor.
        """
        preview = self.processInfoContainer
        column = self.processContainer
        policy = column.sizePolicy()
        wanted = QSizePolicy.Policy.Preferred if mode == SIDE else QSizePolicy.Policy.Minimum
        if policy.verticalPolicy() != wanted:
            policy.setVerticalPolicy(wanted)
            column.setSizePolicy(policy)
            column.updateGeometry()
        unbounded = 16777215
        if mode == SIDE:
            preview.setMinimumSize(QSize(PANE_FLOOR_PX, 0))
            column.setMinimumSize(QSize(self._column_floor_px(), 0))
        else:
            preview.setMinimumSize(QSize(0, self._stack_preview_floor_px()))
            column.setMinimumSize(QSize(0, self._column_min_height_px()))
        preview.setMaximumSize(QSize(unbounded, unbounded))
        column.setMaximumSize(QSize(unbounded, unbounded))
        column.show()
        preview.setVisible(not self._preview_collapsed)

    def _stack_preview_floor_px(self) -> int:
        """The stacked preview pane's height floor: its own content (the 16:9 band plus its controls).

        `PANE_FLOOR_PX` is a WIDTH floor and does not describe this axis: a stacked pane shorter than its content would
        clip the band, whose height is fixed at 16:9 of the width (`_apply_preview_band`). The floor is therefore the
        pane's own minimum hint, never under the width floor.
        """
        return max(PANE_FLOOR_PX, self.processInfoContainer.minimumSizeHint().height())

    def _stack_natural_split(self) -> tuple[int, int]:
        """Today's stacked geometry: the band's pane above, the column at its own content height.

        The preview pane's content height is what the 16:9 band plus the controls need (`_apply_preview_band` has just
        fixed the band); the column's is its content at the body's width. This is the default split AND the splitter's
        minimum height, so a stacked page is exactly as tall as it was before the boundary existed and the page scrolls
        instead of squeezing.
        """
        preview = self._stack_preview_floor_px()
        column = max(self._column_min_height_px(), self.processContainer.sizeHint().height())
        return preview, column

    def _side_column(self, policy_column: int, extent: int, gap: int) -> int:
        """The column's width in `side`: the policy's answer (or the saved split), clamped by both panes' floors."""
        floor = self._column_floor_px()
        ceiling = max(floor, extent - gap - PANE_FLOOR_PX)
        saved = self._saved_split(SIDE)
        wanted = policy_column if saved is None else scaled_px(saved)
        return max(floor, min(int(wanted), ceiling))

    def _stack_preview_height(self, extent: int) -> int:
        """The preview pane's height in `stack`: the default band height, or the saved split, between the floors."""
        floor = self._stack_preview_floor_px()
        ceiling = max(floor, extent - self._split.handleWidth() - self._column_min_height_px())
        saved = self._saved_split(STACK)
        wanted = floor if saved is None else scaled_px(saved)
        return max(floor, min(int(wanted), ceiling))

    def _apply_split_sizes(self, mode: str, column: int) -> int:
        """State the split: the policy's geometry by default, the mode's saved split when one is stored.

        Returns the column's applied width, which is what `policy_column` reports (all three column widgets carry it).
        In `side` the splitter's extent is the body's content width and the preview takes the remainder; in `stack`
        the widths are the body's, so the column's width IS the extent and only the heights are split.
        """
        splitter = self._split
        if splitter is None:                     # the page has not been laid out yet: nothing to size
            return column
        self._applying_split = True
        try:
            if mode == SIDE:
                # The body IS the viewport (the policy's whole purpose: no horizontal scroll), so the split's extent
                # is the viewport less the body's 18 px frame - a number this pass knows, unlike the splitter's own
                # width, which is only right after the layout that this pass may still be inside.
                extent = self.pageScroll.viewport().width() - BODY_INSET_PX
                if extent <= 0:
                    extent = splitter.width()
                gap = splitter.handleWidth()
                applied = self._side_column(column, extent, gap)
                self._apply_column_widths(applied)
                splitter.setSizes([applied, max(1, extent - gap - applied)])
                return applied
            self._apply_column_widths(column)
            natural_preview, natural_column = self._stack_natural_split()
            # A collapsed band is not part of the page's height: it is hidden, its handle is hidden with it, and the
            # column takes the whole split (this is the "collapsing reclaims the vertical space" half of the owner's
            # rule). Reserving the hidden pane's height would make a collapsed 1.0 page scroll for a band nobody sees.
            if self._preview_collapsed:
                natural_total = natural_column
            else:
                natural_total = natural_preview + splitter.handleWidth() + natural_column
            splitter.setMinimumHeight(natural_total)
            extent = max(splitter.height(), natural_total)
            applied = self._stack_preview_height(extent)
            splitter.setSizes([applied, max(1, extent - splitter.handleWidth() - applied)])
            return column
        finally:
            self._applying_split = False

    def _apply_column_widths(self, column: int) -> None:
        """State the column's width on the two strips inside it.

        The column itself is sized by the splitter - that is what a drag moves - but its two inner strips state their
        width explicitly: their content hints (the tab strip's five labels, the render row's controls) are wider than
        the column at 1.0, and a `Minimum`/`MinimumExpanding` policy would answer that hint instead of the column.
        """
        column = max(0, int(column))
        for container in (self.bottomMenuSubContainer, self.widget_13):
            container.setFixedWidth(column)

    # -- the split's persistence ------------------------------------------------------------------------------

    def _saved_split(self, mode: str) -> int | None:
        """The saved split for `mode` in DESIGN px, or `None` when nothing usable is stored (the default split).

        A drag that has not reached the file yet is the current choice, so it is answered from the pending value first
        (a resize inside the save delay must not snap the boundary back). The parse is memoised per stored string: this
        runs on every layout pass, and a garbage value must be reported once instead of on each resize (the store's
        reader already reports and blanks one that came from the file).
        """
        key = PREVIEW_SPLIT_SIDE_KEY if mode == SIDE else PREVIEW_SPLIT_STACK_KEY
        pending = self._pending_split.get(key)
        if pending is not None:
            return int(pending)
        values = getattr(self._settings, "settings", None) if self._settings is not None else None
        if not values:
            return None
        raw = str(values.get(key, "") or "")
        cached = self._split_cache.get(key)
        if cached is not None and cached[0] == raw:
            return cached[1]
        parsed = preview_split(raw, key)
        self._split_cache[key] = (raw, parsed)
        return parsed

    def _on_split_moved(self, position: int, index: int) -> None:
        """A drag of the live boundary: follow it with the column's strips and remember the split for `mode`.

        A programmatic `setSizes` emits this signal too, so the policy's own passes are filtered out - only a move the
        user made is a choice worth storing.
        """
        if self._applying_split or self._split is None:
            return
        if self._layout_mode == SIDE:
            column = self.processContainer.width()
            self._apply_column_widths(column)
            # The property states the width IN FORCE, and a drag is the one writer that can move it off the policy's
            # answer without a new policy pass.
            self._policy_column = column
            self._remember_split(PREVIEW_SPLIT_SIDE_KEY, column)
        else:
            self._remember_split(PREVIEW_SPLIT_STACK_KEY, self.processInfoContainer.height())

    def _remember_split(self, key: str, actual_px: int) -> None:
        """Store `actual_px` as `key`'s DESIGN length; the file write coalesces with the rest of the drag."""
        if actual_px <= 0:
            return
        factor = max(1e-9, UiScale.get_instance().factor())
        self._pending_split[key] = str(max(1, int(round(actual_px / factor))))
        self._split_flush.start()

    def _flush_split(self) -> None:
        """Write the drag's final split(s) through the settings owner - one write per drag, not one per pixel."""
        settings = self._settings
        pending = list(self._pending_split.items())
        if settings is None:
            return                       # no owner: the drag still holds for this session, through `_pending_split`
        for key, value in pending:
            settings.writeSetting(key, value)
            self._pending_split.pop(key, None)

    def _apply_preview_band(self, mode: str, column: int) -> None:
        """In stacked mode the preview's height is 16:9 of its width; in side mode it is the pane's own hint."""
        if mode == STACK and not self._preview_collapsed:
            margins = self.verticalLayout_5.contentsMargins()
            band_width = max(0, column - margins.left() - margins.right())
            height = int(round(band_width * PREVIEW_BAND_ASPECT_H / PREVIEW_BAND_ASPECT_W))
            self.previewBand.setMinimumHeight(height)
            self.previewBand.setMaximumHeight(height)
        else:
            # Release the fixed box: `setFixedHeight` cannot be undone by `setFixedHeight(0)` (that is a zero box), and
            # the side page's 1.0 geometry must be exactly the flow geometry it had before this wave.
            self.previewBand.setMinimumHeight(0)
            self.previewBand.setMaximumHeight(16777215)

    def _apply_render_row(self, mode: str) -> None:
        """Move the render row into the sticky footer in stacked mode and back into the column's flow in side mode."""
        footer = self.renderRowFooter
        if mode == STACK:
            self.verticalLayout_21.removeWidget(self.widget_13)
            if self.widget_13.parentWidget() is not footer:
                self.renderRowFooterLayout.addWidget(self.widget_13)
            row_height = max(self.widget_13.sizeHint().height(), self.widget_13.minimumSizeHint().height())
            footer.setFixedHeight(row_height)
            footer.show()
            footer.raise_()
            margins = self.gridLayout_3.contentsMargins()
            self.gridLayout_3.setContentsMargins(
                margins.left(), margins.top(), margins.right(), self._base_grid_bottom + row_height)
        else:
            self.renderRowFooterLayout.removeWidget(self.widget_13)
            if self.widget_13.parentWidget() is not self.bottomMenuSubContainer:
                # Appended, not inserted at index 1: the stretch above the row owns index 1, and the row must stay the
                # column's LAST item so it keeps sitting on the bottom edge.
                self.verticalLayout_21.addWidget(self.widget_13)
            footer.hide()
            margins = self.gridLayout_3.contentsMargins()
            self.gridLayout_3.setContentsMargins(
                margins.left(), margins.top(), margins.right(), self._base_grid_bottom)

    def _position_overlays(self) -> None:
        """Pin the sticky footer to the page viewport's bottom and the collapse toggle to its top-right."""
        viewport = self.pageScroll.viewport()
        origin = viewport.mapTo(self.procPage, QPoint(0, 0))
        if self.renderRowFooter.isVisible():
            height = self.renderRowFooter.height()
            self.renderRowFooter.setGeometry(
                origin.x(), origin.y() + viewport.height() - height, viewport.width(), height)
            self.renderRowFooter.raise_()
        if self.previewBandToggle.isVisible():
            toggle = self.previewBandToggle
            toggle.adjustSize()
            toggle.setGeometry(
                origin.x() + viewport.width() - toggle.width() - OVERLAY_INSET_PX, origin.y() + OVERLAY_INSET_PX,
                toggle.width(), toggle.height())
            toggle.raise_()

    def toggle_preview_band(self) -> None:
        """The collapse toggle's one action: flip the band and re-run the policy."""
        self.set_preview_collapsed(not self._preview_collapsed)

    def _apply_preview_band_tooltip(self) -> None:
        """State the chevron's tooltip from the band's state and the language in force.

        The glyph and its tooltip name the ACTION the press performs, so both are (state, language) - one writer for
        the two callers that can change either (`set_preview_collapsed`, `retranslate`).
        """
        self.previewBandToggle.setToolTip(QCoreApplication.translate(
            "MainWindow",
            u"Expand the preview band" if self._preview_collapsed else u"Collapse the preview band", None))

    def set_preview_collapsed(self, collapsed: bool) -> None:
        """Collapse or expand the preview band; a no-op when the state already matches."""
        collapsed = bool(collapsed)
        if collapsed == self._preview_collapsed:
            return
        self._preview_collapsed = collapsed
        # The glyph is the ACTION the press performs, like the preview's own pause/play button.
        self.previewBandToggle.setText(u"\u25b8" if collapsed else u"\u25be")
        self._apply_preview_band_tooltip()
        self._apply_layout_policy()

    def retranslate(self, language: str | None = None) -> None:
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Start Time (seconds)", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"End Time (seconds)", None))
        self.renderPreviewBtn.setText(QCoreApplication.translate("MainWindow", u"Render Preview", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.generalSettings), QCoreApplication.translate("MainWindow", u"General", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.tab_6), QCoreApplication.translate("MainWindow", u"Encoder Options", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.renderQueueTab), QCoreApplication.translate("MainWindow", u"Render Queue", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.advancedSettings), QCoreApplication.translate("MainWindow", u"Advanced", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.tab_5), QCoreApplication.translate("MainWindow", u"Logs", None))
        # The preview chrome's tooltips are catalogue documents too, and `setToolTip` at build time only covers the
        # language in force then: a switch left them English (measured 2026-11-27: 27 of 30 tooltips stayed English
        # while their labels followed). They are rebuilt from their parts here, like every label above.
        self.renderPreviewBtn.setToolTip(i18n.tooltip(
            'Render Preview:',
            '- Renders a small portion of the video to show the results (most useful for upscaling)'
        ))
        # The chevron states the ACTION the press performs, so its text is the band's state, not a fixed string. The
        # legacy pause button is hidden by `ProcessTab` but still owned by this page (kept for an A/B), so its tooltip
        # follows the same pass.
        self._apply_preview_band_tooltip()
        self.previewPauseButton.setToolTip(QCoreApplication.translate(
            "MainWindow", u"Paused: the preview holds this frame. Press to follow the render.", None))
        # The labels are what the strip measures: re-run the answer once they are the real ones.
        self._tab_strip.refresh()
        self.startRenderButton.setText("")
        self.killRenderButton.setText("")

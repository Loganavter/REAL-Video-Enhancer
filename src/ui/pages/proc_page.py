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

from PySide6.QtCore import (QCoreApplication, QEvent, QObject, QSize, Qt)
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
from src.ui.pages.general_settings import GeneralSettingsPage
from src.ui.pages.tab_6 import Tab6Page
from src.ui.pages.render_queue_tab import RenderQueueTabPage
from src.ui.pages.advanced_settings import AdvancedSettingsPage
from src.ui.pages.tab_5 import Tab5Page
from src.ui.pages.handles import ProcPageHandles
from src.ui.typography import apply, CAPTION_PT
from src.ui.widgets.tab_strip import TAB_HEIGHT, tab_button, tab_style

# Layer B (2026-11-21): the three widget classes this page no longer builds stock. `TopTabHost` is the tab host, built
# with the stock `QTabWidget`'s strip geometry (the Designer's 22 px strip, no pane radius or inset, a 4 px tab corner);
# `ImageLabel` is the pixmap counterpart of `QLabel` (parent first, and the alignment is stated because Qt's default is
# left+vcenter, not the centre the preview label asks for).
#
# Layer E (page migration): `Text` where the Designer had a `QLabel`, `Button` where it had a `QPushButton`,
# `DoubleSpinBox` for a `QDoubleSpinBox` and `ToastProgressBar` for a `QProgressBar`. Nothing here needs a Qt-shaped
# compatibility adapter.
from sli_ui_toolkit.widgets import (Button, DoubleSpinBox, ImageLabel, MinimalistScrollBar, SurfacePanel, TabStyle,
                                    Text, ToastProgressBar, TopTabHost)

#: Qt's spin-box text alignment (`QAbstractSpinBox().alignment()` is AlignLeft|AlignVCenter); the toolkit field
#: defaults to centre, which would move every value to the middle of its box.
SPIN_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


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
    that cannot (this page's column is capped at 500 px) uses ``overflow`` to size and expose its affordance.
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

    def build(self):
        """Build the widget tree and return the page root."""
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
        # Layer C: the page body paints itself (`surface.page`); every control rule this sheet carried styled a widget
        # that is now a toolkit control painting from tokens. The one rule kept is the read-only rich-text panel on
        # the General tab, which is still a stock `QTextEdit` (a toolkit row: no rich-text pane with a scroll host).
        # No page stylesheet: every control on this page (and its tabs) paints itself from tokens. The read-only
        # rich-text panes are the toolkit's `RichTextView`, which carries its own plate.
        self.gridLayout_3 = QGridLayout(self.procPage)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.processInfoContainer = SurfacePanel(self.procPage, background_token="dialog.background", radius=10)
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

        if _preview_env.environ.get("RVE_UI_RASTER_PREVIEW", "0").strip().lower() in ("1", "true", "yes", "on"):
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
        self.renderPreviewBtn.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Render Preview:</span></p><p>     - Renders a small portion of the video to show the results (most useful for upscaling)</p></body></html>", None))
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
        self.verticalLayout_34 = QVBoxLayout()
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.ETA = Text(parent=self.processInfoContainer)
        self.ETA.setObjectName(u"ETA")
        apply(self.ETA, CAPTION_PT)

        self.verticalLayout_34.addWidget(self.ETA)
        self.FPS = Text(parent=self.processInfoContainer)
        self.FPS.setObjectName(u"FPS")
        apply(self.FPS, CAPTION_PT)

        self.verticalLayout_34.addWidget(self.FPS)
        self.STATUS = Text(parent=self.processInfoContainer)
        self.STATUS.setObjectName(u"STATUS")
        apply(self.STATUS, CAPTION_PT)

        self.verticalLayout_34.addWidget(self.STATUS)
        self.verticalLayout_5.addLayout(self.verticalLayout_34)
        self.gridLayout_3.addWidget(self.processInfoContainer, 0, 2, 1, 1)
        self.processContainer = QWidget(self.procPage)
        self.processContainer.setObjectName(u"processContainer")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.processContainer.sizePolicy().hasHeightForWidth())
        self.processContainer.setSizePolicy(sizePolicy5)
        self.processContainer.setMaximumSize(QSize(500, 16777215))
        # No stylesheet: a disabled toolkit control takes its colour from its token. The rule only tinted the two stock
        # queue lists, which carry their own look until the toolkit grows a compact list.
        self.verticalLayout_7 = QVBoxLayout(self.processContainer)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 10, 0, 0)
        self.bottomMenuSubContainer = SurfacePanel(self.processContainer, background_token="dialog.background")
        self.bottomMenuSubContainer.setObjectName(u"bottomMenuSubContainer")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.bottomMenuSubContainer.sizePolicy().hasHeightForWidth())
        self.bottomMenuSubContainer.setSizePolicy(sizePolicy6)
        self.bottomMenuSubContainer.setMinimumSize(QSize(0, 0))
        self.bottomMenuSubContainer.setMaximumSize(QSize(500, 16777215))

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
        # The only difference from the shared style is the overflow row's policy (`overflow_style`): this column is
        # capped at 500 px, so five labels do not always fit, and an affordance that waits for the pointer is not one.
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
        self.verticalLayout_21.addWidget(self.widget_2)
        self.widget_13 = SurfacePanel(self.bottomMenuSubContainer, background_token="surface.page")
        self.widget_13.setObjectName(u"widget_13")
        self.widget_13.setMaximumSize(QSize(500, 16777215))
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
        self.startRenderButton = Button(parent=self.widget_24, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(59, 55))
        self.startRenderButton.setObjectName(u"startRenderButton")
        self.startRenderButton.setEnabled(True)
        self.startRenderButton.setMaximumSize(QSize(59, 55))
        # The Designer pinned this button at 59x55. The toolkit's hint is 60x50 (the icon is 40x45 and its box is square:
        # 45 + 5 padding), so the ceiling alone left it 5 px short and 1 px wide. A minimum states the Designer's number,
        # and with the ceiling above it the button is exactly 59x55 in any row.

        # QPushButton's own policy: as wide as it needs, exactly as tall as its hint - the toolkit button would
        # otherwise fill the cell and grow past the Designer's 55 px ceiling in a taller row.
        self.startRenderButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon3 = QIcon()
        icon3.addFile(u":/icons/icons/play-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.startRenderButton.setIcon(icon3)
        self.startRenderButton.setIconSize(QSize(40, 45))
        # `setCheckable(False)` states the button's own default and `Button` has no such setter (toggle mode is the
        # constructor's `toggle=`), so the Designer's line is a no-op here; the two state resets below are kept.
        self.startRenderButton.setChecked(False)
        self.startRenderButton.setAutoRepeat(False)
        self.startRenderButton.setAutoExclusive(False)
        self.horizontalLayout_39.addWidget(self.startRenderButton)
        self.horizontalLayout_32.addWidget(self.widget_24)
        self.onRenderButtonsContiainer = QHBoxLayout()
        self.onRenderButtonsContiainer.setObjectName(u"onRenderButtonsContiainer")
        self.pauseRenderButton = Button(parent=self.widget_13, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.pauseRenderButton.setObjectName(u"pauseRenderButton")
        self.pauseRenderButton.setEnabled(True)
        self.pauseRenderButton.setMaximumSize(QSize(59, 55))
        self.pauseRenderButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon4 = QIcon()
        icon4.addFile(u":/icons/icons/pause-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pauseRenderButton.setIcon(icon4)
        self.pauseRenderButton.setIconSize(QSize(40, 45))
        self.pauseRenderButton.setChecked(False)
        self.onRenderButtonsContiainer.addWidget(self.pauseRenderButton)
        self.killRenderButton = Button(parent=self.widget_13, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.killRenderButton.setObjectName(u"killRenderButton")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy11.setHorizontalStretch(0)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.killRenderButton.sizePolicy().hasHeightForWidth())
        self.killRenderButton.setSizePolicy(sizePolicy11)
        self.killRenderButton.setMaximumSize(QSize(59, 55))
        icon5 = QIcon()
        icon5.addFile(u":/icons/icons/x-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.killRenderButton.setIcon(icon5)
        self.killRenderButton.setIconSize(QSize(40, 45))
        self.onRenderButtonsContiainer.addWidget(self.killRenderButton)
        self.horizontalLayout_32.addLayout(self.onRenderButtonsContiainer)
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
        # The bar itself paints no label, so the numbers live on their own line right under it: percentage, elapsed,
        # remaining and the frame counter (owner, 2026-11-22: "under the huge progress bar there is no way to learn the
        # progress in percent or minutes, how much has passed").
        self.progressDetail = Text(parent=self.bottomMenuSubContainer)
        self.progressDetail.setObjectName(u"progressDetail")
        apply(self.progressDetail, CAPTION_PT)
        self.verticalLayout_21.addWidget(self.progressDetail)
        self.verticalLayout_7.addWidget(self.bottomMenuSubContainer)
        self.gridLayout_3.addWidget(self.processContainer, 0, 1, 1, 1)
        self.processSettingsContainer.setCurrentIndex(0)
        # The strip's host half (see `TabStripHost`): measures the labels, exposes an overflow the user can reach,
        # and paints the selected plate from the host's token. It must come after the labels are set, or it would
        # measure placeholders; `retranslate()` below re-runs `refresh()` for the final texts.
        self._tab_strip = TabStripHost(self.processSettingsContainer, strip_style, strip_buttons)
        # The glue's typed edge over this page's own chrome (the tabs have their own groups). Required fields: a
        # missing widget is a TypeError here, not a silent `None` at first render.
        self._handles = ProcPageHandles(
            startRenderButton=self.startRenderButton,
            pauseRenderButton=self.pauseRenderButton,
            killRenderButton=self.killRenderButton,
            onRenderButtonsContiainer=self.onRenderButtonsContiainer,
            previewLabel=self.previewLabel,
            progressBar=self.progressBar,
            progressDetail=self.progressDetail,
            FPS=self.FPS,
            ETA=self.ETA,
            STATUS=self.STATUS,
        )
        self.retranslate()
        return self.procPage

    def retranslate(self) -> None:
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Start Time (seconds)", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"End Time (seconds)", None))
        self.renderPreviewBtn.setText(QCoreApplication.translate("MainWindow", u"Render Preview", None))
        self.ETA.setText(QCoreApplication.translate("MainWindow", u"ETA: ", None))
        self.FPS.setText(QCoreApplication.translate("MainWindow", u"FPS:", None))
        self.STATUS.setText(QCoreApplication.translate("MainWindow", u"STATUS:", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.generalSettings), QCoreApplication.translate("MainWindow", u"General", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.tab_6), QCoreApplication.translate("MainWindow", u"Encoder Options", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.renderQueueTab), QCoreApplication.translate("MainWindow", u"Render Queue", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.advancedSettings), QCoreApplication.translate("MainWindow", u"Advanced", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.tab_5), QCoreApplication.translate("MainWindow", u"Logs", None))
        # The labels are what the strip measures: re-run the answer once they are the real ones.
        self._tab_strip.refresh()
        self.startRenderButton.setText("")
        self.pauseRenderButton.setText("")
        self.killRenderButton.setText("")

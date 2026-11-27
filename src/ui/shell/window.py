"""The window shell: pages built by their own modules, chrome built here.

Hand-maintained since 2026-11-23: the tree started as the Designer's compiled `mainwindow.py`, and the splitter plus its
fixture were removed together with the parity gates. What remains is this file's own: the central widget, the nav rail,
the window metrics and the page stack the five `src/ui/pages/` pages land in. The Designer is not a reference here any
more - the toolkit and `harness/tests/test_toolkit_native.py` are.

No stylesheet lives here any more. The window-wide QSS block (`QMainWindow`, `#centralwidget`, the nav rail's plate,
`QPushButton` and its `:checked`/`:hover`/`:disabled` rules, the tab bar and the scrollbars) was the Designer's way of
naming colours, and every one of those selectors now has an owner that paints instead: the nav rail is a
`SurfacePanel` on `surface.page`, its four buttons are toolkit `Button`s whose plate, hover, checked and disabled
colours are `button.dialog.default.*` tokens, and the pages bring their own surfaces. A QSS rule left behind would be a
no-op on those widgets - the painter owns their pixels - which is exactly what `qss-on-adapter` in
`harness/scripts/ui_gates.py` now catches in this directory too.

The window copies no page widget onto itself any more. RVE's glue used to reach every page widget as an attribute of
the window (`self.output_folder_location`, `self.label_20`, ...), which a recursive page-attribute copy produced by
putting a page's (and its descendants') attributes on the window. That is gone (2026-11-23): the names were never
declared anywhere, so a widget that stopped being built failed at first click, and 16 of the 459 copied names were
reachable only through `getattr(..., None)` and failed silently. The window now composes `page_handles`
(`src/ui/pages/handles.py`) once and copies nothing; its own glue and the page glue read
`self.page_handles.<group>.<name>`, a required dataclass field that the owning page builds.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt
from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSpacerItem,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)
import resources_rc

from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.widgets import Button, SurfacePanel

from src.ui import toolkit_theme
from src.ui import i18n
from src.ui.scale_follower import ScaleFollower
from src.ui.settings_store import window_rect, window_was_maximized
from src.ui.pages.home_page import HomePagePage
from src.ui.pages.more_page import MorePagePage
from src.ui.pages.proc_page import ProcPagePage
from src.ui.pages.settings_page import SettingsPagePage
from src.ui.pages.download_page import DownloadPagePage
from src.ui.pages.handles import MenuHandles, PAGE_NAMES, PageSelector, WindowHandles

#: The page gutter the deleted window stylesheet applied (`QStackedWidget{padding:5px 10px}`). It was the only
#: geometry in that sheet; deleting the sheet without moving the number moved every page 10 px left, 5 px up and made
#: it 20 px wider (measured against a capture of the same tree before the sheet went).
PAGE_INSET_H = 10
PAGE_INSET_V = 5

#: The window's default size in DESIGN px - 1163x700 is what the Designer's window opened at and what every layout gate
#: measures at 1.0. It is a design value like any other: `window_default_size()` resolves it through `scaled_px()` where
#: the window is built and again on every `UiScale.scale_changed`, never at import (an import-time number would freeze
#: the factor the process happened to start with).
WINDOW_DEFAULT_PX = (1163, 700)

#: The window's minimum in DESIGN px: 1050x700 at 1.0, exactly as the Designer stated it.
WINDOW_MIN_PX = (1050, 700)

#: The ceiling the minimum stops at: a 1366x768 laptop's client area, the narrowest panel class this app supports (its
#: 1.0 minimum is already 1050 px wide, so it cannot go below that). The minimum is a USABILITY floor, not a CONTENT
#: floor: the process page body scrolls (wave 2), so content wider or taller than the window is reachable instead of
#: clipped - which is what lets this number stop growing. Scaling the whole 1050 design px to 2.0 would state a
#: 2100x1400 minimum, a window no laptop can fit, and a minimum larger than the screen is a window the user cannot make
#: usable at all. At the cap the rail (110 design px at 2.0), the page insets (40) and the 1000 px process column still
#: fit inside 1366 px, so the cap does not force a horizontal scroll on the column itself; the squeezed preview at 2.0
#: is the owner's accepted trade. The HEIGHT stops at its 1.0 value for the same reason: a 768 px panel leaves about
#: 700 px of client area after decorations, and rows that do not fit scroll.
WINDOW_MIN_CEILING_PX = (1366, 700)

#: The nav rail's width in DESIGN px - the four buttons' shared maximum, because the rail is as wide as its buttons -
#: and their icon box. The rail width is resolved through `scaled_px()` (55 -> 82 -> 110): it is a geometry the shell
#: owns, so it scales here. The icon box is passed to `Button.setIconSize` in DESIGN px instead, because the toolkit's
#: own `sizeHint()` and painter multiply that field (`scaled_px(self._icon_size_px)`, and the region spec reaches
#: `content.py`'s `scaled_px(ctx.region_icon_size_px)`) - scaling it here as well would paint a 140 px glyph at 2.0.
NAV_RAIL_PX = 55
NAV_ICON_PX = 35

#: `homeBtn`'s own floor (the Designer's `setMinimumSize(31, 0)`), design px.
NAV_BUTTON_MIN_PX = 31

#: `QWIDGETSIZE_MAX`: "this axis has no maximum", for the axis a `setMaximumSize` does not constrain.
UNBOUNDED_PX = 16777215


def window_default_size() -> QSize:
    """The window's default size: the design size, NOT multiplied by the UI scale.

    Measured against BlinkTable on 2026-11-24 (its `src/ui/layout_geometry.py:16` + `shared_toolkit/ui/layout_sizing.py:445`):
    a scaled window default is how an app ends up opening a 2326x1400 window on a 2000x1400 screen. BlinkTable keeps its
    window size design-space and lets the CONTENT scale inside it; overflow is carried by scroll areas and an expanding
    canvas, so the failure mode at 2.0 is cramped content, not an unreachable window. RVE now does the same: the type,
    the rail, the column and every page's boxes scale, the window does not.
    """
    return QSize(*WINDOW_DEFAULT_PX)


def window_minimum_size() -> QSize:
    """The window's minimum: the design size, in DESIGN space, never multiplied by the UI scale.

    A scaled floor is a trap and BlinkTable's contract test names it (`tests/contracts/test_window_minimum_size_design_space.py`):
    a 700 px floor at 1.5 states 1050 px, which can exceed the height a user's screen lets them use - and a minimum
    larger than the screen is a window they cannot shrink. The pages scroll (the process page since wave 2, settings
    since before), so the minimum is a usability floor, not a content floor.
    """
    return QSize(*WINDOW_MIN_PX)


def _available_window_rect(window):
    """The available rect of the window's screen (or the primary one), or ``None`` when there is no screen.

    Read at the moment of use: the window may be unplaced at build time, and on a multi-monitor desktop `screen()` can
    answer with a small panel. `None` (offscreen/headless) leaves the size alone rather than inventing a bound.
    """
    try:
        from PySide6.QtWidgets import QApplication

        screen = window.screen() if window is not None else None
        if screen is None and QApplication.instance() is not None:
            screen = QApplication.primaryScreen()
        return None if screen is None else screen.availableGeometry()
    except Exception:
        return None


def fit_size(size: QSize, available) -> QSize:
    """``size`` clamped into ``available`` (a ``QRect`` or ``None``); never larger than the screen.

    Pure on purpose: the caller passes the available rect it read at that moment, so a test can prove the clamp with a
    fake screen instead of depending on the machine. ``None`` means "no screen information" and returns ``size``
    unchanged - a headless or offscreen run must not be silently shrunk by a made-up number.
    """
    if available is None:
        return QSize(size)
    return QSize(min(size.width(), available.width()), min(size.height(), available.height()))


def fit_window_rect(rect: QRect, available, minimum: QSize | None = None) -> QRect:
    """``rect`` clamped INTO ``available``: its size through `fit_size`, its position back on the screen.

    The size half is `fit_size()` itself - one clamp, not a second copy of it. The position half is what it cannot
    state: a rect saved on a monitor that is now gone must land ON a remaining screen instead of off its edge
    (BlinkTable's `fit_into`, `src/utils/geometry.py`). ``available=None`` keeps the position and applies only the
    minimum, the same "no screen information" convention as `fit_size()`.

    `minimum` is `window_minimum_size()` by default and is applied AFTER the fit, because a usability floor outranks
    a screen bound. When the screen is smaller than that floor the window cannot fit at all: it is then pinned to the
    available rect's top-left corner rather than moved further off it.
    """
    if minimum is None:
        minimum = window_minimum_size()
    size = fit_size(rect.size(), available)
    width = max(size.width(), minimum.width())
    height = max(size.height(), minimum.height())
    x, y = rect.x(), rect.y()
    if available is not None:
        x = max(available.left(), min(x, available.left() + available.width() - width))
        y = max(available.top(), min(y, available.top() + available.height() - height))
    return QRect(x, y, width, height)


def restore_window_geometry(window, settings=None, available=None) -> bool:
    """Apply the saved size/position/state to `window`; return whether the last close was maximized.

    `settings` defaults to `window.settings` (the app sets it before `MainWindowShell.build`), `available` to the
    window's screen rect. Nothing saved means EXACTLY the pre-persistence behaviour - the design size resolved through
    `window_default_size()`, clamped by `fit_size()` and left for the window manager to place - which is also the path
    every test that builds the shell without a store takes. A saved rect is applied with `setGeometry` so the POSITION
    survives, after `fit_window_rect` has clamped it on screen and raised it to `window_minimum_size()`.

    The maximized state is set ON the window here, never by showing it: the caller shows the window once, after the
    warm pass. Qt keeps the geometry set before the state as `normalGeometry()`, so unmaximizing returns to the saved
    rect rather than to the maximized one.
    """
    if available is None:
        available = _available_window_rect(window)
    if settings is None:
        settings = getattr(window, "settings", None)
    values = getattr(settings, "settings", None) if settings is not None else None
    stored = window_rect(values) if isinstance(values, dict) else None
    if stored is None:
        window.resize(fit_size(window_default_size(), available))
    else:
        window.setGeometry(fit_window_rect(QRect(*stored), available))
    maximized = isinstance(values, dict) and window_was_maximized(values)
    if maximized:
        window.setWindowState(window.windowState() | Qt.WindowState.WindowMaximized)
    return maximized


def save_window_geometry(window, settings=None) -> None:
    """Persist the window's NORMAL rect and maximized state to the settings owner, at close time.

    Read from the LIVE window, never from a copy taken at build: the user may have resized or maximized it. Closed
    maximized/fullscreen, only the flag is written and the saved normal rect stays untouched - writing the
    screen-sized geometry as "normal" is exactly how an app forgets the size the user chose. `window.geometry()` is
    the space `restore_window_geometry` writes back.
    """
    if settings is None:
        settings = getattr(window, "settings", None)
    if settings is None:
        return
    maximized = window.isMaximized() or window.isFullScreen()
    rect = None if maximized else window.geometry().getRect()
    settings.writeWindowGeometry(rect, maximized)


# The page-attribute copy lived here until 2026-11-23: it copied every public attribute of each page onto the window,
# recursively. The typed handles in `src/ui/pages/handles.py` replace it and no shim re-creates the old surface - a
# name that comes back through a compatibility copy is a name nobody declares again.


class MainWindowShell:
    """Builds the main window's chrome and the five nav pages into a `QMainWindow`."""

    def build(self, window, available=None):
        """Build the chrome and pages into `window`; `available` overrides the screen rect (tests)."""
        # The interface language is configured HERE, by the window's owner, not only in `main()`: three entry points
        # build this window (the app's `main`, the live harness, the tests) and they used to differ - a live instance
        # ran in English whatever `settings.txt` said, and the first switch answered "i18n root not configured"
        # (measured on a lease, 2026-11-24). `main()` still gets there first; this is the guarantee for the rest.
        try:
            from src.ui import i18n

            i18n.ensure_configured()
        except Exception as _i18n_exc:  # a language is a preference, not a precondition
            print(f"[window] interface language failed: {type(_i18n_exc).__name__}: {_i18n_exc}", flush=True)
        # The saved interface face, applied before a single page exists: every widget resolves its font through
        # `UiFont`, so this is the restart half of the switch (the live half is the RVE Settings row wired further
        # down). It is also the ONE apply path - `typography.apply_font_family` validates against this machine's
        # families and refuses an unreadable value by name instead of asking Qt for a face nobody has.
        font_family = self._apply_saved_font_family(window)
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(0)
        icon7 = QIcon()
        icon7.addFile(u":/icons/icons/download.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        # splitter: skipped 'sizePolicy12.setHeightForWidth(self.auto_hdr_mode.sizePolicy().hasHeightForWidth())' (['auto_hdr_mode'] live outside this page)
        # splitter: skipped 'sizePolicy12.setHeightForWidth(self.dynamic_tensorrt_engine.sizePolicy().hasHeightForWidth())' (['dynamic_tensorrt_engine'] live outside this page)
        if not window.objectName():
            window.setObjectName(u"MainWindow")
        # The size/position the user left the window in, when one was saved; otherwise exactly the design size clamped
        # to the screen, which is what every launch did before persistence (see `restore_window_geometry`).
        restore_window_geometry(window, available=available)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(window.sizePolicy().hasHeightForWidth())
        window.setSizePolicy(sizePolicy)
        window.setMinimumSize(window_minimum_size())
        window.setMaximumSize(QSize(16777215, 16777215))
        window.setSizeIncrement(QSize(1, 1))
        window.setToolTipDuration(0)
        self.centralwidget = QWidget(window)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.centralWidget = QWidget(self.centralwidget)
        self.centralWidget.setObjectName(u"centralWidget")
        self.centralWidget.setMinimumSize(QSize(0, 0))
        self.verticalLayout_2 = QVBoxLayout(self.centralWidget)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.mainWindowContainer = QWidget(self.centralWidget)
        self.mainWindowContainer.setObjectName(u"mainWindowContainer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.mainWindowContainer.sizePolicy().hasHeightForWidth())
        self.mainWindowContainer.setSizePolicy(sizePolicy1)
        self.mainWindowContainer.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_3 = QHBoxLayout(self.mainWindowContainer)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.mainWindowContainer)
        self.stackedWidget.setObjectName(u"stackedWidget")
        # The gutter around every page used to come from the window stylesheet (`QStackedWidget{padding:5px 10px}`),
        # which styled COLOUR and INSET in one place; deleting the sheet without moving this moved every page 10 px
        # left and made it 20 px wider (measured against a capture of the same tree before the sheet went). The inset
        # belongs to the container, so it is stated here and nowhere else - in design px, re-resolved on `scale_changed`
        # below with the rest of the window's own geometry.
        self.stackedWidget.setContentsMargins(
            scaled_px(PAGE_INSET_H), scaled_px(PAGE_INSET_V),
            scaled_px(PAGE_INSET_H), scaled_px(PAGE_INSET_V),
        )
        # The gutter has to be the page colour, or it reads as a 10 px lighter frame: the deleted sheet also painted the
        # stack (`QStackedWidget{background-color:#16191d}`), and without it the palette's Window role shows through
        # (measured: a 10x670 and a 1044x5 band, 2.9 % of the page, all of it that frame).
        _gutter = toolkit_theme.token_color("surface.page")
        if _gutter is not None:
            self.stackedWidget.setAutoFillBackground(True)
            _gutter_palette = self.stackedWidget.palette()
            _gutter_palette.setColor(QPalette.ColorRole.Window, _gutter)
            self.stackedWidget.setPalette(_gutter_palette)
        self.homePage_page = HomePagePage()
        self.homePage = self.homePage_page.build()
        self.stackedWidget.addWidget(self.homePage)
        self.morePage_page = MorePagePage()
        self.morePage = self.morePage_page.build()
        self.stackedWidget.addWidget(self.morePage)
        self.procPage_page = ProcPagePage()
        self.procPage = self.procPage_page.build()
        self.stackedWidget.addWidget(self.procPage)
        self.settingsPage_page = SettingsPagePage()
        self.settingsPage = self.settingsPage_page.build()
        self.stackedWidget.addWidget(self.settingsPage)
        self.downloadPage_page = DownloadPagePage()
        self.downloadPage = self.downloadPage_page.build()
        self.stackedWidget.addWidget(self.downloadPage)
        self.horizontalLayout_3.addWidget(self.stackedWidget)
        self.verticalLayout_2.addWidget(self.mainWindowContainer)
        self.gridLayout.addWidget(self.centralWidget, 0, 1, 1, 1)
        self.leftMenuContainer = QWidget(self.centralwidget)
        self.leftMenuContainer.setObjectName(u"leftMenuContainer")
        self.verticalLayout = QVBoxLayout(self.leftMenuContainer)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        # The rail's plate. `#leftMenuSubContainer{background-color:#16191d; border-radius:30px}` was the only rule that
        # gave the nav its darker body, and a stock QWidget cannot paint a token - `SurfacePanel` can, and
        # `surface.page` is the colour it names (#16191d).
        self.leftMenuSubContainer = SurfacePanel(self.leftMenuContainer, background_token="surface.page", radius=30)
        self.leftMenuSubContainer.setObjectName(u"leftMenuSubContainer")
        self.verticalLayout_3 = QVBoxLayout(self.leftMenuSubContainer)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalWidget_2 = QWidget(self.leftMenuSubContainer)
        self.verticalWidget_2.setObjectName(u"verticalWidget_2")
        self.verticalLayout_6 = QVBoxLayout(self.verticalWidget_2)
        self.verticalLayout_6.setSpacing(9)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(9, 9, 9, 9)
        # The four nav buttons are toolkit `Button`s with the toggle state machine built in (`toggle=True` - the
        # toolkit's toggle mode is constructor-only). The look they carry is the deleted QSS's `QPushButton` rule
        # expressed as widget properties: the `surface` variant's plate reads `button.dialog.default.background`
        # (#2c313c), hover `...background.hover` (#343b47) and disabled `...background.disabled` (#676e7b). The surface
        # family has no `checked` token, so the painter falls back to `button.dialog.default.background.pressed`, which
        # RVE maps to #676e7b - the QSS's `:checked` colour; radius 10, `padding:5px 10px` and left-aligned content are
        # the toolkit button's own defaults. `size=(0, 0)` drops the standalone 36x36 control size, and
        # Minimum/Fixed is QPushButton's own policy (as wide as it needs, exactly as tall as its hint) - the toolkit
        # button would otherwise split the rail's spare height between the two top buttons. The Designer's
        # `setCheckable(True)` is the constructor's `toggle=True`, since `Button` has no such setter.
        #
        # Keyboard: a `QPushButton` gave the rail Tab focus and Space/Return for free, and a bare custom `QWidget`
        # would not have - the toolkit `Button` sets StrongFocus and its own `keyPressEvent` activates on
        # Space/Return/Enter, so changing the class did not cost the rail its keyboard path. Measured offscreen over
        # the four built buttons (2026-11-23): toolkit `Button` (not a `QPushButton`), focusPolicy StrongFocus, one
        # `clicked` per Space, objectNames `homeBtn`/`processBtn`/`settingsBtn`/`downloadBtn` intact.
        self.homeBtn = Button(parent=self.verticalWidget_2, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.homeBtn.setObjectName(u"homeBtn")
        self.homeBtn.setMinimumSize(QSize(scaled_px(NAV_BUTTON_MIN_PX), 0))
        self.homeBtn.setMaximumSize(QSize(scaled_px(NAV_RAIL_PX), UNBOUNDED_PX))
        self.homeBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon8 = QIcon()
        icon8.addFile(u":/icons/icons/home.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.homeBtn.setIcon(icon8)
        self.homeBtn.setIconSize(QSize(NAV_ICON_PX, NAV_ICON_PX))
        self.homeBtn.setChecked(True)
        self.verticalLayout_6.addWidget(self.homeBtn)
        self.processBtn = Button(parent=self.verticalWidget_2, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.processBtn.setObjectName(u"processBtn")
        self.processBtn.setMaximumSize(QSize(scaled_px(NAV_RAIL_PX), UNBOUNDED_PX))
        self.processBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon9 = QIcon()
        icon9.addFile(u":/icons/icons/cpu.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.processBtn.setIcon(icon9)
        self.processBtn.setIconSize(QSize(NAV_ICON_PX, NAV_ICON_PX))
        self.verticalLayout_6.addWidget(self.processBtn)
        self.verticalLayout_3.addWidget(self.verticalWidget_2)
        self.verticalWidget = QWidget(self.leftMenuSubContainer)
        self.verticalWidget.setObjectName(u"verticalWidget")
        self.verticalLayout_4 = QVBoxLayout(self.verticalWidget)
        self.verticalLayout_4.setSpacing(9)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, -1, 9, -1)
        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_4.addItem(self.verticalSpacer_5)
        self.settingsBtn = Button(parent=self.verticalWidget, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.settingsBtn.setObjectName(u"settingsBtn")
        sizePolicy12.setHeightForWidth(self.settingsBtn.sizePolicy().hasHeightForWidth())
        self.settingsBtn.setSizePolicy(sizePolicy12)
        self.settingsBtn.setMaximumSize(QSize(scaled_px(NAV_RAIL_PX), UNBOUNDED_PX))
        self.settingsBtn.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        icon10 = QIcon()
        icon10.addFile(u":/icons/icons/settings.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.settingsBtn.setIcon(icon10)
        self.settingsBtn.setIconSize(QSize(NAV_ICON_PX, NAV_ICON_PX))
        self.verticalLayout_4.addWidget(self.settingsBtn)
        self.downloadBtn = Button(parent=self.verticalWidget, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadBtn.setObjectName(u"downloadBtn")
        self.downloadBtn.setMaximumSize(QSize(scaled_px(NAV_RAIL_PX), UNBOUNDED_PX))
        self.downloadBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.downloadBtn.setIcon(icon7)
        self.downloadBtn.setIconSize(QSize(NAV_ICON_PX, NAV_ICON_PX))
        self.verticalLayout_4.addWidget(self.downloadBtn)
        self.verticalLayout_3.addWidget(self.verticalWidget)
        self.verticalLayout.addWidget(self.leftMenuSubContainer)
        self.gridLayout.addWidget(self.leftMenuContainer, 0, 0, 1, 1)
        window.setCentralWidget(self.centralwidget)
        # The glue's typed edge over the pages and the nav stack. Each page built its own group in `build()`; this
        # composes them once, on the WINDOW, so glue reads `window.page_handles.<group>.<name>` and a missing widget is a
        # `TypeError` at page build. Nothing is copied onto the window beside this object (the deleted page-attribute
        # copy did). The selector keeps the stack's index contract behind a page NAME.
        self.page_handles = WindowHandles(
            general=self.procPage_page.generalSettings_page._handles,
            proc=self.procPage_page._handles,
            queue=self.procPage_page.renderQueueTab_page._handles,
            settings=self.settingsPage_page._handles,
            home=self.homePage_page._handles,
            output=self.settingsPage_page.tab_2_page._handles,
            render_settings=self.settingsPage_page.tab_3_page._handles,
            app_settings=self.settingsPage_page.tab_4_page._handles,
            encoder=self.procPage_page.tab_6_page._handles,
            logs=self.procPage_page.tab_5_page._handles,
            advanced=self.procPage_page.advancedSettings_page._handles,
            download=self.downloadPage_page._handles,
            menu=MenuHandles(
                stackedWidget=self.stackedWidget,
                pages=PageSelector(stack=self.stackedWidget, names=PAGE_NAMES),
                homeBtn=self.homeBtn,
                processBtn=self.processBtn,
                settingsBtn=self.settingsBtn,
                downloadBtn=self.downloadBtn,
            ),
        )
        self.page_handles.menu.pages.show("proc")
        window.page_handles = self.page_handles
        # The RVE Settings row shows the face actually in force, and a live pick goes through the SAME apply path the
        # startup restore used. The index is set BEFORE the connection: restoring a saved value must not look like a
        # user choice (and must not rewrite `settings.txt` while the window opens).
        self._install_font_family_row(window, font_family)
        QMetaObject.connectSlotsByName(window)
        # The window paints the page colour itself. Every surface inside is token-painted, and a surface that fails to
        # paint (a missing token, a widget built before the theme) then shows the DESKTOP through a compositing WM -
        # observed 2026-11-22 on the owner's screen as a ghost window. An opaque backdrop makes that impossible.
        _backdrop = toolkit_theme.token_color("surface.page")
        if _backdrop is None:
            # Loud, not silent: with no backdrop the window is only opaque where its children happen to paint, and a
            # compositing WM shows whatever is behind through every surface that does not (a "ghost window" on the
            # owner's screen, 2026-11-22). The theme is mandatory before the window is built, so a missing token here
            # means the startup order broke - fail with the reason instead of shipping a see-through window.
            raise RuntimeError(
                "window backdrop: the theme has no `surface.page` token, so the window would have no opaque "
                "background and a compositing window manager would show the desktop through it. Install the toolkit "
                "theme before MainWindowShell.build().")
        # Paint the backdrop ourselves - the reference host's `OpaqueFillHost` pattern: an explicit `paintEvent` fill
        # with `WA_TranslucentBackground` off and `autoFillBackground` off, and NO `WA_OpaquePaintEvent` (that flag is
        # for leaf widgets that fill their own rect; on a container whose children skip painting it leaves their areas
        # unrepainted, which is the stale-fragment bug this comment used to prescribe). The see-through the owner saw
        # through the preview pane turned out to be environment-level - it reproduces in BlinkTable and in Telegram
        # Desktop on the same mutter/NVIDIA session and never in a screenshot (session note §20) - so it must not be
        # bought with a flag that breaks repainting.
        window._backdrop_colour = _backdrop
        window.setAutoFillBackground(False)
        # `RVE_UI_FORCE_COMPOSITE=1` withholds the opaque DECLARATION only: the window still paints its own backdrop, so
        # the picture is identical, but the compositor has to blend it instead of trusting the opaque region. That is the
        # scanout test - and it must not touch the decorations. (Marking the window translucent did force compositing,
        # but Qt gives a translucent window NO frame, so the title bar - the only place to grab the window - vanished;
        # owner, 2026-11-22.)
        import os as _os

        if _os.environ.get("RVE_UI_NO_BACKDROP_PAINT"):
            # The paintEvent-based fill and the opacity declaration go together: with one gone the other would leave the
            # window unpainted. Both are switched off for the bisect of the window's own properties.
            window.setAutoFillBackground(True)
        elif _os.environ.get("RVE_UI_OPAQUE_SURFACE"):
            # Opt-in only, and not the default: `WA_OpaquePaintEvent` on a window whose children are all translucent is
            # the anti-pattern the reference host spells out - "that flag plus a skipped/disabled update punches
            # see-through holes through the window chrome" (BlinkTable `ui/widgets/shelf/widget.py`, OpaqueFillHost).
            # Here it cost stale pixels: a sub-tab switch or a render start left fragments of the previous page because
            # nothing repainted the area the new content did not cover (owner, 2026-11-22, healed by a window resize).
            # The canopy below is the reference host's pattern instead: the widget paints its own rect, the flag that
            # tells Qt "I paint everything, do not repaint for me" is NOT set.
            window.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        _window_palette = window.palette()
        _window_palette.setColor(QPalette.ColorRole.Window, _backdrop)
        window.setPalette(_window_palette)

        # The nav stack needs the same guarantee `TopTabHost` gives its sub-tabs: `QStackedLayout` in StackOne mode lays
        # out only the page it shows, so `procPage`/`settingsPage`/`downloadPage` sat at Qt's 640x480 while hidden and
        # the audit caught exactly that (2026-11-22). Showing one then clips its content at the old width until a window
        # resize - the owner's "the right 15 % never repaints".
        from PySide6.QtCore import QEvent, QObject

        # Bisect switches: the "fit every page + settle" pass is the newest geometry work in the shell and a Resize ->
        # layout -> resize feedback loop would look exactly like the owner's staircase after a resize. Each switch
        # removes one half so the loop can be attributed instead of guessed.
        import os as _fit_env

        _FIT_DISABLED = bool(_fit_env.environ.get("RVE_UI_NO_FIT_PAGES"))
        _SETTLE_DISABLED = bool(_fit_env.environ.get("RVE_UI_NO_SETTLE"))
        _stack = self.stackedWidget

        class _PageStackFit(QObject):
            def eventFilter(self, watched, event):
                if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
                    self.fit()
                return False

            def fit(self, only=None) -> None:
                """Give every page the box it gets and lay it out for that box, synchronously.

                ``only`` limits the settle pass to one page (the one a switch just showed): its geometry was set by the
                stacked layout, but the LayoutRequest pass that would lay its children out for that geometry is still
                queued - one frame with children sized for the box the page had *before* the switch. Measured on the
                settings page before this hook: its top tab host sat at 1000x766 for one turn and became 1008x777 only
                on the next event-loop pass. The settle is ``deep`` because the scroll area's content widget was already
                the final size, so nothing resized it and its own layout never re-ran (2026-11-23).
                """
                if _FIT_DISABLED:
                    return
                stack = _stack
                # The page's box is the stack MINUS its page inset (`setContentsMargins(PAGE_INSET_H, PAGE_INSET_V, ...)`),
                # not the stack's outer rect: the outer size is 2*PAGE_INSET_H wider and 2*PAGE_INSET_V taller than the
                # box `QStackedLayout` actually gives a page. Laying every page out for the outer size (and leaving it
                # where it was) meant the first painted frame of a page carried content laid out for a box 20 px wider
                # and 10 px taller than the page it sits in, and Qt's own layout pass then corrected it - the owner's
                # "content of every new page jumps up 5-10 px to the right place as soon as I touch it" (2026-11-23).
                # `contentsRect()` is exactly the rect the stacked layout will apply, so the settle pass below lays the
                # page out for its final box and the first frame is already right.
                box = stack.contentsRect()
                if box.isEmpty():
                    return
                for index in range(stack.count()):
                    page = stack.widget(index)
                    if page is None:
                        continue
                    if page.geometry() != box:
                        page.setGeometry(box)
                    if _SETTLE_DISABLED or page is not (only if only is not None else stack.currentWidget()):
                        continue
                    try:
                        from sli_ui_toolkit.ui.layout_settle import settle_layout
                    except Exception:
                        continue
                    try:
                        settle_layout(page, deep=True)
                    except Exception:
                        pass

        _page_fit = _PageStackFit(window)
        _page_fit.fit()
        _stack.installEventFilter(_page_fit)
        # A page switch is not a Resize/Show of the stack, so the filter above never sees it: the newly current page
        # would keep its children laid out for the box it had while hidden until the queued layout request ran. Settle
        # the page that was just shown, in the same turn the switch happens.
        _stack.currentChanged.connect(
            lambda index: _page_fit.fit(only=_stack.widget(index))
        )

        window._page_stack_fit = _page_fit

        # --- the window's own geometry, stated once --------------------------------------------------------------
        # Everything below is a design value from this module's constants. `scaled_px` runs HERE - at build, and again
        # on every `UiScale.scale_changed` through the follower installed after this - never at import, where it would
        # freeze whatever factor the process started with. The icon box is the one deliberate exception: it stays in
        # design px because the toolkit's `Button` multiplies that field itself (see `NAV_ICON_PX`), and a scaled value
        # here would be scaled a second time inside `sizeHint`/the painter.
        #
        # A scale change re-applies the DEFAULT size, so a window the user had resized is re-stated at the new factor
        # rather than kept at a px box that means a different thing at 1.5 or 2.0. Changing the UI scale is a global
        # geometry change; the alternative (keep the box, leave the chrome cramped or huge) is what the migration fixed.
        def _apply_window_geometry() -> None:
            # The minimum is design-space and therefore constant across factors; a sealed change of it is still the
            # right order (a resize below a minimum in force is silently clamped away).
            window.setMinimumSize(window_minimum_size())
            # NO `resize()` here: the window does not follow the factor (see `window_default_size`), and resizing on a
            # scale change would undo a size the user chose by hand. BlinkTable's main-window policy returns before
            # `resize()` for exactly this reason (`shared_toolkit/ui/layout_sizing.py:445`, `resize_when_hidden=False`).
            inset_h = scaled_px(PAGE_INSET_H)
            inset_v = scaled_px(PAGE_INSET_V)
            self.stackedWidget.setContentsMargins(inset_h, inset_v, inset_h, inset_v)
            for button in (self.homeBtn, self.processBtn, self.settingsBtn, self.downloadBtn):
                button.setMaximumSize(QSize(scaled_px(NAV_RAIL_PX), UNBOUNDED_PX))
                button.setIconSize(QSize(NAV_ICON_PX, NAV_ICON_PX))
            self.homeBtn.setMinimumSize(QSize(scaled_px(NAV_BUTTON_MIN_PX), 0))
            # A page's box is the stack MINUS its inset, and the pages were laid out for the old one. Re-fit them in the
            # same turn instead of leaving every page's children at the previous gutter until the next window resize.
            fit = getattr(self, "_page_stack_fit", None)
            if fit is not None:
                fit.fit()

        # The follower calls the statement once now and again on `UiScale.scale_changed`; parented to the window, so Qt
        # drops the connection when the window goes away (a process-wide signal would fire into deleted widgets).
        window._ui_scale_follower = ScaleFollower(window, _apply_window_geometry)
        # One place for the control type size: a page's 12 pt container font would otherwise cascade 16 px into every
        # toolkit control (the deleted adapter stated 12 px per control; this states it per window).
        from src.ui.typography import apply_control_fonts

        apply_control_fonts(window)
        # And one place for the wheel: a value control answers it only when it already has focus, so a wheel over a
        # combo scrolls the page instead of stepping the value under the cursor (owner: "tons of misclicks").
        from src.ui.wheel_policy import apply_wheel_requires_focus

        apply_wheel_requires_focus(window)
        # The clipped-edge fade: content that continues past a scroll area's edge fades out by alpha instead of being cut
        # by a hard line (owner's ongoing requirement). It was switched off while the render-time freeze was being
        # diagnosed; the freeze turned out to be the render log re-wrapping itself (§24), and the fade is a cheap painted
        # overlay again (24 px fill per edge, no QGraphicsEffect - an effect renders the whole widget per paint). So it is
        # ON by default; `RVE_UI_EDGE_FADE=0` turns it off for an A/B.
        import os as _os

        if _os.environ.get("RVE_UI_EDGE_FADE", "1").strip().lower() not in ("0", "false", "no", "off"):
            from src.ui.scroll_fade import apply_edge_fades

            apply_edge_fades(window)

        # The language pass, once for the whole window: every page bound a relay under its own root in `build()`
        # (`i18n.bind_page`), and this one install is what makes the toolkit's `language_changed` reach them - current
        # pages and any widget added later (the toolkit installs a `ChildAdded` watcher). The shell's own title is the
        # last relay, so a switch repaints it too instead of leaving the one string that was dead code (the method had
        # no caller before this).
        self._window = window
        i18n.bind_page(self, window)
        i18n.install(window)

    def _apply_saved_font_family(self, window) -> str:
        """Apply the stored interface family (or the app's default face) and return what is in force.

        The value lives in the settings owner the app puts on `window` before building; a bare `QMainWindow` (the
        tests, the live harness) has none and opens in the default. Validation is against the families this machine
        reports, because a `settings.txt` can come from another host: an unknown face is refused and named in the log
        by `typography.apply_font_family`, never handed to Qt and never a reason for the window not to open.
        """
        settings = getattr(window, "settings", None)
        values = getattr(settings, "settings", None)
        stored = values.get("font_family") if isinstance(values, dict) else None
        from src.ui.typography import apply_font_family, available_families

        return apply_font_family(stored, available_families())

    def _install_font_family_row(self, window, family: str) -> None:
        """Point the RVE Settings combo at the face in force and wire a live choice to the one apply path.

        Reached through the page object rather than `page_handles`, the same way the render-settings page is reached:
        `page_handles` names controls the glue reads, and this row's reader is the shell itself. A live change asks
        `UiFont.set_family`, whose `font_changed` repaints the built tree - no page is rebuilt.
        """
        combo = self.settingsPage_page.tab_4_page.font_family
        combo.setCurrentIndex(max(0, combo.findData(family)))
        combo.currentIndexChanged.connect(lambda _index: self._font_family_chosen(window))

    def _font_family_chosen(self, window) -> None:
        """Apply and persist the family the user picked in the RVE Settings row.

        The first entry's data is ``""`` (the app's own face), which is a real choice and must be applied and saved
        exactly like a named family - only a missing item (``None``) is a no-op.
        """
        combo = self.settingsPage_page.tab_4_page.font_family
        family = combo.itemData(combo.currentIndex())
        if family is None:
            return
        from src.ui.typography import apply_font_family, available_families

        applied = apply_font_family(family, available_families())
        settings = getattr(window, "settings", None)
        if settings is not None and settings.settings.get("font_family") != applied:
            settings.writeSetting("font_family", applied)

    def retranslate(self, language: str | None = None) -> None:
        # The product name, not the Qt context: the field used to read "MainWindow" in the title bar of BOTH languages
        # (the source string doubled as the context). A product name is not translated, so no catalogue key is added.
        self._window.setWindowTitle("REAL Video Enhancer")

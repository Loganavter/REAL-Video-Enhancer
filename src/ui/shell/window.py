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

from PySide6.QtCore import QCoreApplication, QMetaObject, QSize, Qt
from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSpacerItem,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)
import resources_rc

from sli_ui_toolkit.widgets import Button, SurfacePanel

from src.ui import toolkit_theme
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


# The page-attribute copy lived here until 2026-11-23: it copied every public attribute of each page onto the window,
# recursively. The typed handles in `src/ui/pages/handles.py` replace it and no shim re-creates the old surface - a
# name that comes back through a compatibility copy is a name nobody declares again.


class MainWindowShell:
    """Builds the main window's chrome and the five nav pages into a `QMainWindow`."""

    def build(self, window):
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(0)
        icon7 = QIcon()
        icon7.addFile(u":/icons/icons/download.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        # splitter: skipped 'sizePolicy12.setHeightForWidth(self.auto_hdr_mode.sizePolicy().hasHeightForWidth())' (['auto_hdr_mode'] live outside this page)
        # splitter: skipped 'sizePolicy12.setHeightForWidth(self.dynamic_tensorrt_engine.sizePolicy().hasHeightForWidth())' (['dynamic_tensorrt_engine'] live outside this page)
        if not window.objectName():
            window.setObjectName(u"MainWindow")
        window.resize(1163, 700)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(window.sizePolicy().hasHeightForWidth())
        window.setSizePolicy(sizePolicy)
        window.setMinimumSize(QSize(1050, 700))
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
        # belongs to the container, so it is stated here and nowhere else.
        self.stackedWidget.setContentsMargins(PAGE_INSET_H, PAGE_INSET_V, PAGE_INSET_H, PAGE_INSET_V)
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
        self.homeBtn.setMinimumSize(QSize(31, 0))
        self.homeBtn.setMaximumSize(QSize(55, 16777215))
        self.homeBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon8 = QIcon()
        icon8.addFile(u":/icons/icons/home.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.homeBtn.setIcon(icon8)
        self.homeBtn.setIconSize(QSize(35, 35))
        self.homeBtn.setChecked(True)
        self.verticalLayout_6.addWidget(self.homeBtn)
        self.processBtn = Button(parent=self.verticalWidget_2, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.processBtn.setObjectName(u"processBtn")
        self.processBtn.setMaximumSize(QSize(55, 16777215))
        self.processBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon9 = QIcon()
        icon9.addFile(u":/icons/icons/cpu.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.processBtn.setIcon(icon9)
        self.processBtn.setIconSize(QSize(35, 35))
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
        self.settingsBtn.setMaximumSize(QSize(55, 16777215))
        self.settingsBtn.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        icon10 = QIcon()
        icon10.addFile(u":/icons/icons/settings.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.settingsBtn.setIcon(icon10)
        self.settingsBtn.setIconSize(QSize(35, 35))
        self.verticalLayout_4.addWidget(self.settingsBtn)
        self.downloadBtn = Button(parent=self.verticalWidget, text="", variant="surface", toggle=True,
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadBtn.setObjectName(u"downloadBtn")
        self.downloadBtn.setMaximumSize(QSize(55, 16777215))
        self.downloadBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.downloadBtn.setIcon(icon7)
        self.downloadBtn.setIconSize(QSize(35, 35))
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

    def retranslate(self, window) -> None:
        window.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))

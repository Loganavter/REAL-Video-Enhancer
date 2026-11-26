"""The toolkit is the interface's only reference - these pins keep it that way.

Why this file exists (2026-11-23): the migration to the toolkit used the Designer's compiled output as the source of
truth. Every gate whose verdict was "this matches the Designer" fought the toolkit-ward changes, so the fixture, the
splitter and the parity modules were removed and the invariants worth keeping were moved here, where nothing reads the
Designer:

* pages may not construct fonts - the type scale in ``src/ui/typography.py`` is the only source (formerly a parity check
  against the Designer's ``QFont`` temporaries);
* every page ``DoubleSpinBox`` keeps its step arrows - the Designer's ``QSpinBox`` always had them, and the toolkit
  spells that ``show_step_buttons=True``;
* an attribute the shell exposes on the window may not shadow a ``QMainWindow`` member - and since 2026-11-23 the
  declared surface is the typed handle set, not a copy of the pages' attributes;
* the three shapes the Designer left behind may not grow: generated ``name_<N>`` object names, bare ``QWidget``/``QFrame``
  wrappers that exist only to hold a layout, and ``QSpacerItem`` spacers. Each is a per-file count that may only shrink;
  removing one means dropping the same count here, which is the point (a baseline that does not shrink is rot).

The three counts are a ratchet, not a wall: they are accepted debt on the pages the migration has not rewritten yet
(same reasoning as ``harness/scripts/ui_gates.py``'s ``KNOWN``). A new page that introduces one fails immediately.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

from src.ui.pages import handles

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "src" / "ui" / "pages"
SHELL = ROOT / "src" / "ui" / "shell" / "window.py"

#: The Designer disambiguates duplicate object names by appending ``_N`` (``widget_19``, ``verticalLayout_23``,
#: ``scrollAreaWidgetContents_5``). A name a page gives on purpose never needs a bare number.
GENERATED_NAME = re.compile(r"^[A-Za-z][A-Za-z_0-9]*_[0-9]+$")

#: Generated names per page, today. A count may only go down; a page at 0 drops its entry.
KNOWN_GENERATED_NAMES: dict[str, int] = {
    "advanced_settings.py": 24,
    "download_page.py": 31,
    "general_settings.py": 55,
    "home_page.py": 10,
    "more_page.py": 1,
    "proc_page.py": 19,
    "render_queue_tab.py": 3,
    "settings_page.py": 4,
    "tab_2.py": 19,
    "tab_3.py": 15,
    "tab_4.py": 9,
    "tab_5.py": 2,
    "tab_6.py": 32,
}

#: ``QWidget``/``QFrame`` containers whose only call is ``setObjectName`` (a layout inside, nothing else): the wrapper
#: level that makes the tree read as ``page -> container -> rows``. The page's own root and a real widget are not one.
KNOWN_BARE_WRAPPERS: dict[str, int] = {
    "advanced_settings.py": 6,
    "download_page.py": 7,
    "general_settings.py": 24,
    "home_page.py": 2,
    "proc_page.py": 3,
    "render_queue_tab.py": 1,
    "settings_page.py": 1,
    "tab_2.py": 6,
    "tab_3.py": 1,
    "tab_4.py": 4,
    "tab_6.py": 8,
}

#: Designer spacers. The toolkit shape is ``layout.addStretch()``; a ``QSpacerItem`` is a widget-less item the page has
#: to size by hand and it fights the layout on resize.
KNOWN_SPACERS: dict[str, int] = {
    "advanced_settings.py": 7,
    "download_page.py": 2,
    "general_settings.py": 8,
    "home_page.py": 1,
    "settings_page.py": 1,
    "tab_2.py": 6,
    "tab_3.py": 4,
    "tab_4.py": 3,
    "tab_6.py": 6,
}

#: Window attributes that collide with a `QMainWindow` member (a later call to the Qt method/property would hit the
#: widget instead). Empty since 2026-11-23: `centralWidget` was the one record, and it was a collision the deleted
#: page-attribute copy created (`window.centralWidget` was a QWidget, so `window.centralWidget()` was a TypeError).
#: The dict stays as the place a deliberate collision has to be recorded, not deleted.
SHADOWED_QT_ATTRIBUTES: dict[str, str] = {}#: `dir(QMainWindow)` public members, as data: generated once from PySide6 6.11 (the app's Qt) so this
#: file can assert the shadow direction without importing Qt. Inherited QWidget/QObject names included.
QT_MAINWINDOW_ATTRIBUTES = frozenset("""
DockOption PaintDeviceMetric RenderFlag acceptDrops accessibleDescription accessibleIdentifier accessibleName
actionEvent actions activateWindow addAction addActions addDockWidget addToolBar addToolBarBreak adjustSize
autoFillBackground backgroundRole backingStore baseSize blockSignals centralWidget changeEvent childAt
childEvent children childrenRect childrenRegion clearFocus clearMask close closeEvent colorCount connect
connectNotify contentsMargins contentsRect contextMenuEvent contextMenuPolicy corner create createPopupMenu
createWinId createWindowContainer cursor customContextMenuRequested customEvent deleteLater depth destroy
destroyed devType devicePixelRatio devicePixelRatioF devicePixelRatioFScale disconnect disconnectNotify
dockOptions dockWidgetArea documentMode dragEnterEvent dragLeaveEvent dragMoveEvent dropEvent dumpObjectInfo
dumpObjectTree dynamicPropertyNames effectiveWinId emit encodeMetricF ensurePolished enterEvent event
eventFilter find findChild findChildren focusInEvent focusNextChild focusNextPrevChild focusOutEvent
focusPolicy focusPreviousChild focusProxy focusWidget font fontInfo fontMetrics foregroundRole frameGeometry
frameSize geometry grab grabGesture grabKeyboard grabMouse grabShortcut graphicsEffect graphicsProxyWidget
hasFocus hasHeightForWidth hasMouseTracking hasTabletTracking height heightForWidth heightMM hide hideEvent
iconSize iconSizeChanged inherits initPainter inputMethodEvent inputMethodHints inputMethodQuery insertAction
insertActions insertToolBar insertToolBarBreak installEventFilter internalWinId isActiveWindow isAncestorOf
isAnimated isDockNestingEnabled isEnabled isEnabledTo isFullScreen isHidden isLeftToRight isMaximized
isMinimized isModal isQmlExposed isQuickItemType isRightToLeft isSeparator isSignalConnected isTopLevel
isVisible isVisibleTo isWidgetType isWindow isWindowModified isWindowType keyPressEvent keyReleaseEvent
keyboardGrabber killTimer layout layoutDirection leaveEvent locale logicalDpiX logicalDpiY lower mapFrom
mapFromGlobal mapFromParent mapTo mapToGlobal mapToParent mask maximumHeight maximumSize maximumWidth menuBar
menuWidget metaObject metric minimumHeight minimumSize minimumSizeHint minimumWidth mouseDoubleClickEvent
mouseGrabber mouseMoveEvent mousePressEvent mouseReleaseEvent move moveEvent moveToThread nativeEvent
nativeParentWidget nextInFocusChain normalGeometry objectName objectNameChanged overrideWindowFlags
overrideWindowState paintEngine paintEvent painters paintingActive palette parent parentWidget physicalDpiX
physicalDpiY pos previousInFocusChain property raise_ receivers rect redirected releaseKeyboard releaseMouse
releaseShortcut removeAction removeDockWidget removeEventFilter removeToolBar removeToolBarBreak render
repaint resize resizeDocks resizeEvent restoreDockWidget restoreGeometry restoreState saveGeometry saveState
screen scroll sender senderSignalIndex setAcceptDrops setAccessibleDescription setAccessibleIdentifier
setAccessibleName setAnimated setAttribute setAutoFillBackground setBackgroundRole setBaseSize
setCentralWidget setContentsMargins setContextMenuPolicy setCorner setCursor setDisabled setDockNestingEnabled
setDockOptions setDocumentMode setEnabled setFixedHeight setFixedSize setFixedWidth setFocus setFocusPolicy
setFocusProxy setFont setForegroundRole setGeometry setGraphicsEffect setHidden setIconSize
setInputMethodHints setLayout setLayoutDirection setLocale setMask setMaximumHeight setMaximumSize
setMaximumWidth setMenuBar setMenuWidget setMinimumHeight setMinimumSize setMinimumWidth setMouseTracking
setObjectName setPalette setParent setProperty setScreen setShortcutAutoRepeat setShortcutEnabled
setSizeIncrement setSizePolicy setStatusBar setStatusTip setStyle setStyleSheet setTabOrder setTabPosition
setTabShape setTabletTracking setToolButtonStyle setToolTip setToolTipDuration setUnifiedTitleAndToolBarOnMac
setUpdatesEnabled setVisible setWhatsThis setWindowFilePath setWindowFlag setWindowFlags setWindowIcon
setWindowIconText setWindowModality setWindowModified setWindowOpacity setWindowRole setWindowState
setWindowTitle sharedPainter show showEvent showFullScreen showMaximized showMinimized showNormal
signalsBlocked size sizeHint sizeIncrement sizePolicy splitDockWidget stackUnder startTimer staticMetaObject
statusBar statusTip style styleSheet tabPosition tabShape tabifiedDockWidgetActivated tabifiedDockWidgets
tabifyDockWidget tabletEvent takeCentralWidget testAttribute thread timerEvent toolBarArea toolBarBreak
toolButtonStyle toolButtonStyleChanged toolTip toolTipDuration topLevelWidget tr underMouse ungrabGesture
unifiedTitleAndToolBarOnMac unsetCursor unsetLayoutDirection unsetLocale update updateGeometry
updateMicroFocus updatesEnabled visibleRegion whatsThis wheelEvent width widthMM winId window windowFilePath
windowFlags windowHandle windowIcon windowIconChanged windowIconText windowIconTextChanged windowModality
windowOpacity windowRole windowState windowTitle windowTitleChanged windowType x y
""".split())

# -- source helpers -----------------------------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _page_trees() -> dict[str, ast.Module]:
    return {path.name: _parse(path) for path in sorted(PAGES.glob("*.py"))}


# -- kept invariants ----------------------------------------------------------------------------------------------


def test_pages_do_not_construct_fonts() -> None:
    """`src/ui/typography.py` is the only place a font is stated; a page names a scale entry."""
    offenders: list[str] = []
    for name, tree in _page_trees().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if called in {"QFont", "setPointSize", "setPointSizeF"}:
                    offenders.append(f"{name}:{node.lineno}: {ast.unparse(node)}")
    assert not offenders, "page code must go through src/ui/typography.py:\n" + "\n".join(offenders)


def test_every_page_spin_box_keeps_its_step_buttons() -> None:
    """A stepped value keeps Qt's ▲/▼ gutter: the toolkit's `DoubleSpinBox` defaults it OFF, so the flag is required."""
    offenders = []
    for name, tree in _page_trees().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if called != "DoubleSpinBox":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            value = keywords.get("show_step_buttons")
            enabled = isinstance(value, ast.Constant) and value.value is True
            if not enabled:
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, "these DoubleSpinBox fields render without step buttons - pass show_step_buttons=True: " + (
        ", ".join(offenders))


def _handle_names() -> set[str]:
    """Every name the typed edge declares: the handle groups' fields.

    Until 2026-11-23 the shell copied each page's public attributes onto the window, and THIS was the surface a
    `QMainWindow` member could collide with: `centralWidget` was one of the copied names, so `window.centralWidget()`
    was a `TypeError`. The copy is gone; the declared surface is `WindowHandles`, so the names come from there - the
    same set `harness/tests/test_page_handles.py` requires the glue to read through.
    """
    names: set[str] = set()
    for field in dataclasses.fields(handles.WindowHandles):
        annotation = field.type
        group = getattr(handles, annotation if isinstance(annotation, str) else annotation.__name__)
        names |= {member.name for member in dataclasses.fields(group)}
    assert names, "WindowHandles declares no field - this pin would measure an empty surface"
    return names


def test_no_handle_name_shadows_a_qmainwindow_attribute() -> None:
    shadows = sorted(_handle_names() & QT_MAINWINDOW_ATTRIBUTES)
    unrecorded = [name for name in shadows if name not in SHADOWED_QT_ATTRIBUTES]
    stale = sorted(set(SHADOWED_QT_ATTRIBUTES) - set(shadows))
    assert not unrecorded, (
        f"these handle names are also QMainWindow methods or properties: {unrecorded} - rename the widget, or record "
        "the deliberate collision in SHADOWED_QT_ATTRIBUTES with the reason")
    assert not stale, f"SHADOWED_QT_ATTRIBUTES records collisions that no longer exist: {stale} - drop them"


# -- shape ratchet ------------------------------------------------------------------------------------------------


def _generated_names(tree: ast.Module) -> list[str]:
    found = []
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "setObjectName" and call.args:
            argument = call.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if GENERATED_NAME.match(argument.value):
                    found.append(argument.value)
    return found


def _bare_wrappers(tree: ast.Module) -> list[str]:
    """`self.x = QWidget(...)`/`QFrame(...)` where nothing is ever called on `self.x` but `setObjectName`."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not (isinstance(target, ast.Attribute) and isinstance(value, ast.Call)):
            continue
        called = getattr(value.func, "id", getattr(value.func, "attr", ""))
        if called not in ("QWidget", "QFrame"):
            continue
        used = {sub.func.attr for sub in ast.walk(tree)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Attribute) and sub.func.value.attr == target.attr}
        if not (used - {"setObjectName"}):
            found.append(target.attr)
    return found


def _spacers(tree: ast.Module) -> int:
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "QSpacerItem")


def _ratchet(kind: str, measured: dict[str, int], known: dict[str, int]) -> None:
    grown = {name: count for name, count in measured.items() if count > known.get(name, 0)}
    assert not grown, (
        f"{kind} grew past the recorded baseline: {grown}. The Designer shape may not come back: use the toolkit "
        "container the page already has, `layout.addStretch()` for a spacer, and a name that says what the widget is")
    stale = {name: count for name, count in known.items() if measured.get(name, 0) < count}
    assert not stale, (
        f"{kind} dropped below its baseline on {stale} - lower the entry in the same change, a baseline that does not "
        "shrink is rot")


def test_no_new_generated_designer_names() -> None:
    measured = {name: len(_generated_names(tree)) for name, tree in _page_trees().items()}
    measured = {name: count for name, count in measured.items() if count}
    _ratchet("generated object names", measured, KNOWN_GENERATED_NAMES)


def test_no_new_bare_wrapper_widgets() -> None:
    measured = {name: len(_bare_wrappers(tree)) for name, tree in _page_trees().items()}
    measured = {name: count for name, count in measured.items() if count}
    _ratchet("bare wrapper widgets", measured, KNOWN_BARE_WRAPPERS)


def test_no_new_spacer_items() -> None:
    measured = {name: count for name, tree in _page_trees().items() if (count := _spacers(tree))}
    _ratchet("QSpacerItem spacers", measured, KNOWN_SPACERS)

"""The window's exposed attribute surface, checked statically (AST only, no Qt import).

`src/ui/shell/window.py:expose()` copies every page widget onto the `QMainWindow`, and RVE's glue
reads ~120 of those names afterwards. That name set *is* the app's API: a page widget renamed or
deleted without touching the glue is a startup crash, not a layout tweak. The live check
(`harness/scripts/ui_shell_identity.py:128-133`) builds both trees with PySide6 and diffs the
windows' `vars()`; this file is the static half of the same contract, so it also runs on the
portable harness interpreter.

The model mirrors `expose()` exactly: the shell's own (`self.<name> = ...` in `build()`) attributes
are exposed, and an attribute whose name ends in `_page` is a page object, recursed into instead of
exposed. Three directions are asserted:

* every attribute the frozen Designer fixture builds is exposed by the shell, or listed in
  ``ACCEPTED_MISSING`` with the reason it is gone (and the dict is exact - a name that comes back
  has to leave the dict);
* every exposed name is a Designer name, or listed in ``EXPOSED_NOT_DESIGNER``;
* no exposed name shadows a ``QMainWindow`` member, except the recorded ones - today exactly one,
  ``centralWidget``, which ``window.py:267`` assigns as a ``QWidget`` and therefore makes
  ``window.centralWidget()`` raise ``TypeError``. Recorded, not renamed: renaming the Designer name
  is the glue's problem, not this pin's.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHELL = ROOT / "src" / "ui" / "shell" / "window.py"
PAGES = ROOT / "src" / "ui" / "pages"
DESIGNER = ROOT / "harness" / "fixtures" / "designer_mainwindow.py"

_ICON_REASON = ("question-mark icon label, deleted with the rest in 54b36567; the frozen Designer "
                "fixture still builds it")

#: Designer attributes the app deliberately no longer exposes. Keyed by name: a name that reappears on
#: the surface must be removed from here (the test asserts both directions), so this cannot rot silently.
_FINISHED_LIST_REASON = (
    "the queue page's second list ('Finished Output Files') was removed on purpose: a finished job stays in the queue as "
    "a row with its final state, so the page is one list that takes the whole height (owner, 2026-11-22). The frozen "
    "Designer fixture still builds it"
)

_QUEUE_BUTTONS_REASON = (
    "the queue's Remove / Move Up / Move Down buttons and their column were deleted on purpose: the actions live in "
    "the list's right-click menu and apply to the whole selection (owner, 2026-11-22). The frozen Designer fixture "
    "still builds them"
)

ACCEPTED_MISSING: dict[str, str] = {
    "RemoveFromRenderQueue": _QUEUE_BUTTONS_REASON,
    "MoveUpRenderQueue": _QUEUE_BUTTONS_REASON,
    "MoveDownRenderQueue": _QUEUE_BUTTONS_REASON,
    "widget_23": _QUEUE_BUTTONS_REASON,
    "OutputFilesListWidget": _FINISHED_LIST_REASON,
    "gridLayout_4": _FINISHED_LIST_REASON,
    "label_51": _FINISHED_LIST_REASON,
    "verticalLayout_28": _QUEUE_BUTTONS_REASON,
    "label_17": _ICON_REASON,
    "label_21": _ICON_REASON,
    "label_24": _ICON_REASON,
    "label_29": _ICON_REASON,
    "label_31": _ICON_REASON,
    "label_34": _ICON_REASON,
    "label_35": _ICON_REASON,
    "label_38": _ICON_REASON,
    "label_39": _ICON_REASON,
    "label_44": _ICON_REASON,
    "label_46": _ICON_REASON,
    "label_47": _ICON_REASON,
    "label_5": _ICON_REASON,
    "label_50": _ICON_REASON,
    "label_54": _ICON_REASON,
    "label_55": _ICON_REASON,
    "label_56": _ICON_REASON,
    "label_57": _ICON_REASON,
    "label_58": _ICON_REASON,
    "label_59": _ICON_REASON,
    "label_63": _ICON_REASON,
    "label_64": _ICON_REASON,
    "label_66": _ICON_REASON,
    "label_69": _ICON_REASON,
    "label_71": _ICON_REASON,
    "label_73": _ICON_REASON,
    "label_75": _ICON_REASON,
    "label_85": _ICON_REASON,
    "label_89": _ICON_REASON,
    "qualityPresetInfoLabel": _ICON_REASON,
}

#: Exposed names the Designer never builds. Empty today: the 13 page objects (`<name>_page`) are
#: `expose()`'s recursion seam - they are walked, never set on the window - so every exposed name is a
#: Designer name. An entry here is an attribute the glue reads that the Designer knows nothing about.
EXPOSED_NOT_DESIGNER: dict[str, str] = {
    "progressDetail": ("added on purpose: the progress bar paints no label, so the percentage, elapsed, remaining and "
                       "frame counter need a line of their own under it (owner, 2026-11-22)"),
}

#: Exposed names that collide with a `QMainWindow` member (a later call to the Qt method/property would
#: hit the widget instead).
SHADOWED_QT_ATTRIBUTES: dict[str, str] = {
    "centralWidget": ("window.py:267 assigns a QWidget under QMainWindow.centralWidget's own name, so "
                      "window.centralWidget() is a TypeError; reachable through expose()"),
}

#: `dir(QMainWindow)` public members, as data: generated once from PySide6 6.11 (the app's Qt) so this
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


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_self(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self")


def _function(node: ast.AST, name: str) -> ast.FunctionDef | None:
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef) and child.name == name:
            return child
    return None


def _self_assignments(node: ast.AST) -> dict[str, ast.expr]:
    """`self.<name> = <value>` inside `node`; a later assignment wins, as it would at runtime."""
    found: dict[str, ast.expr] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if _is_self(target):
                    found[target.attr] = child.value
    return found


def _shell_build() -> ast.FunctionDef:
    shell_class = next((node for node in _parse(SHELL).body if isinstance(node, ast.ClassDef)), None)
    assert shell_class is not None, "window.py has no shell class"
    build = _function(shell_class, "build")
    assert build is not None, f"{shell_class.name} has no build()"
    return build


def _designer() -> tuple[dict[str, ast.expr], set[str]]:
    """The fixture's `setupUi` attributes, plus the names `retranslateUi` touches (a set, not assignments)."""
    tree = _parse(DESIGNER)
    setup = _function(tree, "setupUi")
    retranslate = _function(tree, "retranslateUi")
    assert setup is not None, "the Designer fixture has no setupUi"
    assert retranslate is not None, "the Designer fixture has no retranslateUi"
    touched = {child.attr for child in ast.walk(retranslate) if _is_self(child)}
    return _self_assignments(setup), touched


def _page_classes() -> dict[str, dict[str, ast.expr]]:
    """class name -> its `self.<name> = ...` assignments, for every class in `src/ui/pages/`."""
    classes: dict[str, dict[str, ast.expr]] = {}
    for path in sorted(PAGES.glob("*.py")):
        for node in _parse(path).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = _self_assignments(node)
    return classes


def _exposed() -> tuple[set[str], list[str]]:
    """The attributes `MainWindowShell.build` puts on the window, modelling `expose()` line for line."""
    classes = _page_classes()
    exposed: set[str] = set()
    roots: list[str] = []

    def visit(assignments: dict[str, ast.expr]) -> None:
        for name, value in assignments.items():
            if name.startswith("_"):
                continue
            if not name.endswith("_page"):
                exposed.add(name)
                continue
            roots.append(name)
            called = value.func.id if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) else None
            assert called in classes, (
                f"page object {name} builds {ast.unparse(value)}; the walk only follows classes declared in "
                f"{PAGES.name}/ - teach this walk the new page module")
            visit(classes[called])

    visit(_self_assignments(_shell_build()))
    return exposed, roots


def test_the_walk_follows_the_same_seam_as_expose() -> None:
    build = _shell_build()
    calls = [node for node in ast.walk(build) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "expose"]
    assert calls, "MainWindowShell.build no longer calls expose(window, self) - the surface this test pins is gone"
    exposed, roots = _exposed()
    assert roots, "the walk found no page objects - expose()'s `_page` recursion (or the naming) changed"
    assert exposed, "the walk produced an empty surface - nothing was parsed"


def test_designer_retranslate_only_names_attributes_it_builds() -> None:
    attributes, touched = _designer()
    orphaned = sorted(touched - set(attributes))
    assert not orphaned, f"retranslateUi touches attributes setupUi never assigns: {orphaned}"


def test_every_designer_attribute_is_exposed_or_recorded() -> None:
    attributes, _touched = _designer()
    exposed, _roots = _exposed()
    missing = sorted(set(attributes) - exposed)
    unexpected = [name for name in missing if name not in ACCEPTED_MISSING]
    stale = [name for name in ACCEPTED_MISSING if name not in missing]
    empty = sorted(name for name, reason in ACCEPTED_MISSING.items() if not reason.strip())
    assert not unexpected, (
        "the Designer builds these attributes and the window no longer exposes them - expose them from the page, "
        f"or record each in ACCEPTED_MISSING with a reason: {unexpected}")
    assert not stale, (
        f"ACCEPTED_MISSING records attributes that are exposed again (or never existed): {stale} - drop them")
    assert not empty, f"ACCEPTED_MISSING entries need a one-line reason: {empty}"


def test_every_extra_exposed_name_is_recorded() -> None:
    attributes, _touched = _designer()
    exposed, _roots = _exposed()
    extra = sorted(exposed - set(attributes))
    unrecorded = [name for name in extra if name not in EXPOSED_NOT_DESIGNER]
    stale = sorted(set(EXPOSED_NOT_DESIGNER) - set(extra))
    empty = sorted(name for name, reason in EXPOSED_NOT_DESIGNER.items() if not reason.strip())
    assert not unrecorded, (
        f"the window exposes attributes the Designer never builds: {unrecorded} - record each in "
        "EXPOSED_NOT_DESIGNER with a reason, or stop exposing it")
    assert not stale, f"EXPOSED_NOT_DESIGNER records names that are Designer names (or absent): {stale}"
    assert not empty, f"EXPOSED_NOT_DESIGNER entries need a one-line reason: {empty}"


def test_no_exposed_name_shadows_a_qmainwindow_attribute() -> None:
    exposed, _roots = _exposed()
    shadows = sorted(exposed & QT_MAINWINDOW_ATTRIBUTES)
    unrecorded = [name for name in shadows if name not in SHADOWED_QT_ATTRIBUTES]
    stale = sorted(set(SHADOWED_QT_ATTRIBUTES) - set(shadows))
    assert not unrecorded, (
        f"these exposed attributes shadow a QMainWindow method or property of the same name: {unrecorded} - "
        "rename the widget, or record the deliberate collision in SHADOWED_QT_ATTRIBUTES with the reason")
    assert not stale, f"SHADOWED_QT_ATTRIBUTES records collisions that no longer exist: {stale} - drop them"

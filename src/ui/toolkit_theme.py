"""RVE's look as toolkit tokens: the only place a colour is mapped.

Why this file exists: RVE's look was not a stylesheet in one place. It was 22 inline `setStyleSheet` blocks the
Designer wrote into `mainwindow.py`, one `styleSheet()` shared by the seven popups (`src/ui/QTstyle.py`, deleted when
the shell's last QSS went), and a QPalette applied in `main()`. The toolkit's look is named tokens. Under the owner's
decision of 2026-11-21 ("the theme owns the whole look"), this module is the single translation: each entry names the
place in RVE that requires the colour, the toolkit token that replaces it, and the value.

Two measured facts shape it:

* a ticket that names a token the palette does not define is a **silent no-op**. The previous table asked for
  `dialog.button.background.hover` and `dialog.button.background.checked`, which FLUENT_DARK does not have - so hover
  and checked were never applied and the page looked "almost right" for a reason nobody could see in the log;
* importing the toolkit subtree (`sli_ui_toolkit.theme`, `sli_ui_toolkit.widgets`) runs
  `scrollbar_policy._register_builtin_policies()` at import time, and `register_widget_policy` immediately sweeps
  every widget that already exists. Imported after `setupUi`, it retrofits 68 of RVE's 69 stock scrollbars with
  `MinimalistScrollBar`; imported before the window exists, it touches nothing (measured both ways, 2026-11-21). So
  the import order is part of the look, which is why `install()` is called from `main()` before `MainWindow` is
  constructed and why the structural harness is the judge of it rather than an eyeball.

`TOKENS` is built from `LOOK` so a token cannot be added to the palette without saying which rule asked for it, and
`install()` refuses unknown token names instead of skipping them.
"""

from __future__ import annotations

import re

from PySide6.QtGui import QColor

# provenance -> (token, hex). The provenance is either a literal QSS selector RVE used to carry, or a QPalette role from
# the `QPalette` this module builds (`qt_palette()`, the theme-off fallback that used to live in `QTstyle.py`). Keep the
# spelling exact: the audit in `harness/scripts` matches these keys against that palette, so a rule that changes there
# fails instead of drifting.
LOOK: dict[str, tuple[str, str]] = {
    # --- QPalette roles: what stock widgets paint from when QSS says nothing -------------------------------
    "Palette().Window": ("Window", "#1f232a"),
    "Palette().WindowText": ("WindowText", "#ffffff"),
    "Palette().Base": ("Base", "#191919"),
    "Palette().AlternateBase": ("AlternateBase", "#353535"),
    "Palette().ToolTipBase": ("ToolTipBase", "#000000"),
    "Palette().ToolTipText": ("ToolTipText", "#ffffff"),
    "Palette().Text": ("Text", "#ffffff"),
    "Palette().Button": ("Button", "#353535"),
    "Palette().ButtonText": ("ButtonText", "#ffffff"),
    "Palette().BrightText": ("BrightText", "#ff0000"),
    "Palette().Highlight": ("Highlight", "#2a82da"),
    "Palette().HighlightedText": ("HighlightedText", "#000000"),
    # --- surfaces ----------------------------------------------------------------------------------------
    "MainWindow / #centralwidget{background-color}": ("dialog.background", "#1f232a"),
    # The page/nav body. It was QSS-owned ("stock until W6") until the toolkit gained a container that paints a token
    # (`SurfacePanel`); now a page is that container and the colour is a token with one meaning.
    "QStackedWidget page surfaces{background-color}": ("surface.page", "#16191d"),
    "QPushButton{background-color}": ("dialog.button.background", "#2c313c"),
    # The toolkit button paints its plate from `button.dialog.default.*` (variant "surface"), so RVE's rule is stated
    # on those tokens; mapping only `dialog.button.background` left the plate at FLUENT's #3c3c3c (measured: the
    # migrated button had no visible plate against the page).
    "QPushButton{background-color} (surface variant)": ("button.dialog.default.background", "#2c313c"),
    "QPushButton:hover{background-color} (surface variant)": ("button.dialog.default.background.hover", "#343b47"),
    "QPushButton:pressed{background-color} (surface variant)": ("button.dialog.default.background.pressed", "#676e7b"),
    "QPushButton:disabled{background-color} (surface variant)": ("button.dialog.default.background.disabled", "#676e7b"),
    "QPushButton:hover{background-color}": ("dialog.button.hover", "#343b47"),
    "QPushButton:checked / :disabled{background-color}": ("button.toggle.background.checked", "#676e7b"),
    "QPlainTextEdit / QTextEdit{background-color}": ("surface.list", "#2c313c"),
    # The toolkit's combo field paints from `surface.background` (no per-widget override exists), and RVE's control
    # plate is #2c313c, so that token is RVE's CONTROL surface. The page/nav background (#16191d) stays with the page
    # QSS until W6 rebuilds those containers; mapping it here as well would give one token two meanings, which this
    # table refuses by design. A field-specific token is a toolkit row for later, not a second colour here.
    "QComboBox{background-color} (toolkit field token)": ("surface.background", "#2c313c"),
    "QLineEdit / dialog inputs{background-color}": ("dialog.input.background", "#2c313c"),
    # --- text ----------------------------------------------------------------------------------------------
    "QLabel / QLineEdit / QPushButton{color}": ("dialog.text", "#ffffff"),
    # --- controls -----------------------------------------------------------------------------------------
    # The stock checkbox is a white 1px outline with a white fill when checked (the app's own stylesheet, now deleted).
    # The toolkit's indicator paints its outline from `dialog.border` and its checked fill from `accent`, so RVE's
    # monochrome look lands on those two tokens - and `accent` is also what a toolkit progress/toast fill uses, which
    # is consistent with RVE's white `QProgressBar::chunk`.
    "QCheckBox::indicator{border}": ("dialog.border", "#ffffff"),
    # RVE's checked box is the `check-square.svg` icon: a white outline and a white tick on the page background, i.e.
    # NO fill. The toolkit now takes those two colours from tokens (`checkbox.check`, `checkbox.background.checked`);
    # a fully transparent fill is how an outline-only box is asked for.
    "QCheckBox::indicator::checked{background-color}": ("checkbox.background.checked", "#00ffffff"),
    "QCheckBox::indicator::checked{image: check-square.svg}": ("checkbox.check", "#ffffff"),
    "QProgressBar::chunk{background-color}": ("accent", "#ffffff"),
    "QProgressBar{background-color}": ("slider.track.background", "#2c313c"),
    # The Process page's own rule (`#progressBar{background-color:#343b47}`) - a PER-WIDGET override of the global bar
    # colour. It was deleted with the page stylesheet and is stated as its own token so the widget can carry it.
    "#progressBar{background-color}": ("progress.track.background", "#343b47"),
    "QPushButton::toolTip / menus{background-color}": ("tooltip.background", "#676e7b"),
    "QToolTip{color}": ("tooltip.text", "#ffffff"),
}


#: Colours RVE's stylesheet still owns because the widget that paints them is still stock: the page/nav surfaces,
#: the scroll areas, the tab widget, the list and the text edits. They become tokens when those widgets migrate (W6),
#: and naming them here is the honest alternative to dropping the audit - a rule that vanishes from the table without a
#: reason is exactly how a colour gets lost.
QSS_OWNED: dict[str, str] = {
    "#16191d": "page/nav surfaces (#leftMenuSubContainer, #bottomMenuSubContainer, QStackedWidget) - stock until W6",
    "#cccccc": "tooltip/disabled text in rich-text labels - stock until W4 finishes",
}


#: `QPalette` roles the token table cannot map: the toolkit palette has no token for "a disabled control" or "a link",
#: so these are facts of the fallback look rather than colours a toolkit painter reads. They are the half of the old
#: `QTstyle.Palette()` that `LOOK` does not cover - kept here because this module now owns that palette.
FALLBACK_ROLES: dict[str, str] = {
    "Link": "#2a82da",
}
FALLBACK_DISABLED_ROLES: dict[str, str] = {
    "Base": "#313131",
    "Text": "#5a5a5a",
    "Button": "#2a2a2a",
    "ButtonText": "#5a5a5a",
    "Window": "#313131",
    "WindowText": "#5a5a5a",
}


def qt_palette():
    """The stock `QPalette` RVE applies when the toolkit theme is off - `main()`'s fallback.

    The enabled roles are read from :data:`LOOK` (so the table and the palette cannot disagree about a colour), the
    disabled group and the link colour from the two tables above. This replaced `QTstyle.Palette()`: the audit used to
    compare the token table against that file, and it now compares it against this function, which is what the app
    actually applies.
    """
    from PySide6.QtGui import QPalette

    palette = QPalette()
    for rule, (_token, hex_value) in LOOK.items():
        match = re.fullmatch(r"Palette\(\)\.(\w+)", rule)
        if match:
            palette.setColor(getattr(QPalette.ColorRole, match.group(1)), QColor(hex_value))
    for name, hex_value in FALLBACK_ROLES.items():
        palette.setColor(getattr(QPalette.ColorRole, name), QColor(hex_value))
    for name, hex_value in FALLBACK_DISABLED_ROLES.items():
        palette.setColor(QPalette.ColorGroup.Disabled, getattr(QPalette.ColorRole, name), QColor(hex_value))
    return palette


def _build_tokens() -> dict[str, str]:
    tokens: dict[str, str] = {}
    for rule, (token, hex_value) in LOOK.items():
        previous = tokens.get(token)
        if previous is not None and previous.lower() != hex_value.lower():
            # Two rules asking for different colours under one token is a decision the table cannot make silently:
            # either the token is wrong for one of them, or RVE's look has two colours here and the toolkit needs a
            # second token name.
            raise ValueError(f"token {token!r} mapped to both {previous} and {hex_value} ({rule})")
        tokens[token] = hex_value
    return tokens


#: token -> hex, derived from :data:`LOOK`.
TOKENS: dict[str, str] = _build_tokens()


def enabled() -> bool:
    """The theme is always on - toolkit widgets paint from tokens and RVE's pages are built with them.

    `RVE_TOOLKIT_THEME=off` is the escape hatch for an A/B against the stock look (and for a bisect when a colour is
    questioned): with the theme off, the app builds the pages but toolkit widgets have no palette to read.
    """
    import os
    return os.environ.get("RVE_TOOLKIT_THEME", "").lower() not in {"0", "off", "false", "no"}


def _hexargb(value: str) -> str:
    """`#rrggbb`, `#aarrggbb` and Qt's `.name()` spellings compared in one space (alpha included)."""
    color = QColor(value)
    return color.name(QColor.NameFormat.HexArgb).lower()


def token_color(name: str) -> QColor | None:
    """A toolkit token as a `QColor`, or `None` when the palette does not define it.

    For the few containers that are plain `QWidget`s with no toolkit counterpart (a `QStackedWidget` painting the gutter
    around its pages): reading the token keeps the colour theme-owned instead of a hex in the shell.
    """
    from sli_ui_toolkit.theme import ThemeManager

    try:
        value = ThemeManager.get_instance().try_get_color(name)
    except Exception:
        return None
    return QColor(value) if value is not None and value.isValid() else None


def palette() -> dict[str, QColor]:
    """FLUENT_DARK with RVE's values on top - complete on purpose, so no toolkit token is left unpainted."""
    from sli_ui_toolkit.palettes import FLUENT_DARK

    resolved = {name: QColor(value) for name, value in FLUENT_DARK.items()}
    unknown = sorted(set(TOKENS) - set(resolved))
    if unknown:
        raise ValueError(f"tokens not in the toolkit palette: {unknown}")
    for token, hex_value in TOKENS.items():
        resolved[token] = QColor(hex_value)
    return resolved


def install(app) -> list[str]:
    """Register RVE's palette, apply it, and return the applied lines (provenance -> token = value)."""
    from sli_ui_toolkit.palettes import FLUENT_LIGHT
    from sli_ui_toolkit.theme import ThemeManager

    theme = ThemeManager.get_instance()
    theme.register_palettes(FLUENT_LIGHT, palette())
    theme.set_theme("dark", app, await_ripples=False)
    return [f"{rule} -> {token} = {hex_value}" for rule, (token, hex_value) in LOOK.items()]

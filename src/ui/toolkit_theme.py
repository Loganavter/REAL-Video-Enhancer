"""RVE's look as toolkit tokens: the only place a colour is mapped.

Why this file exists: RVE's look is not a stylesheet in one place. It is 22 inline `setStyleSheet` blocks the Designer
wrote into `mainwindow.py`, one `styleSheet()` shared by the seven popups (`src/ui/QTstyle.py`), and a QPalette applied
in `main()` (`REAL-Video-Enhancer.py:994`). The toolkit's look is named tokens. Under the owner's decision of
2026-11-21 ("the theme owns the whole look"), this module is the single translation: each entry names the place in RVE
that requires the colour, the toolkit token that replaces it, and the value.

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

from PySide6.QtGui import QColor

# provenance -> (token, hex). The provenance is either a literal QSS selector from `QTstyle.py` / `mainwindow.py`, or
# a QPalette role from `QTstyle.Palette()`. Keep the spelling exact: the audit test in `harness/tests` matches these
# keys against the stylesheet and the palette, so a rule that changes there fails here instead of drifting.
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
    "QPushButton{background-color}": ("dialog.button.background", "#2c313c"),
    # The toolkit button paints its plate from `button.dialog.default.*` (variant "surface"), so RVE's rule is stated
    # on those tokens; mapping only `dialog.button.background` left the plate at FLUENT's #3c3c3c (measured: the
    # migrated button had no visible plate against the page).
    "QPushButton{background-color} (surface variant)": ("button.dialog.default.background", "#2c313c"),
    "QPushButton:hover{background-color} (surface variant)": ("button.dialog.default.background.hover", "#343b47"),
    "QPushButton:pressed{background-color} (surface variant)": ("button.dialog.default.background.pressed", "#676e7b"),
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
    # The stock checkbox is a white 1px outline with a white fill when checked (QTstyle.py). The toolkit's
    # indicator paints its outline from `dialog.border` and its checked fill from `accent`, so RVE's monochrome
    # look lands on those two tokens - and `accent` is also what a toolkit progress/toast fill uses, which is
    # consistent with RVE's white `QProgressBar::chunk`.
    "QCheckBox::indicator{border}": ("dialog.border", "#ffffff"),
    # RVE's checked box is the `check-square.svg` icon: a white outline and a white tick on the page background, i.e.
    # NO fill. The toolkit now takes those two colours from tokens (`checkbox.check`, `checkbox.background.checked`);
    # a fully transparent fill is how an outline-only box is asked for.
    "QCheckBox::indicator::checked{background-color}": ("checkbox.background.checked", "#00ffffff"),
    "QCheckBox::indicator::checked{image: check-square.svg}": ("checkbox.check", "#ffffff"),
    "QProgressBar::chunk{background-color}": ("accent", "#ffffff"),
    "QProgressBar{background-color}": ("slider.track.background", "#2c313c"),
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

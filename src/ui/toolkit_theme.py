"""Install `sli-ui-toolkit`'s theme in RVE and map RVE's QSS onto its tokens.

Why this file exists: the P0 spike showed that a toolkit widget paints **nothing** underneath its content until the
theme is installed, and that the *look* comes from the theme, not from the widget. RVE's look lives in QSS hex
(`src/ui/QTstyle.py` plus inline strings in the generated `mainwindow.py`); the toolkit's lives in named tokens. So
the minimal-deviation migration is a map, and this module is its single home.

The map is filled in as phases migrate screens: an entry is only added after the screen it affects was captured and
compared against the stock build, so nothing here is a guess about a colour nobody looked at.

Source of every value below: `src/ui/QTstyle.py` (`styleSheet()`), read 2026-11-21.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

# RVE QSS rule -> (token, hex). The left column is the literal selector in QTstyle.py so the mapping can be audited
# against the stylesheet without guessing which rule a token came from.
QSS_TO_TOKENS: dict[str, tuple[str, str]] = {
    "QPushButton{background-color}": ("dialog.button.background", "#2c313c"),
    "QPushButton:hover{background-color}": ("dialog.button.background.hover", "#343b47"),
    "QPushButton:checked{background-color}": ("dialog.button.background.checked", "#676e7b"),
    "QPushButton{color}": ("dialog.text", "#ffffff"),
    "QLabel{color}": ("dialog.text", "#ffffff"),
    "QPlainTextEdit{background-color}": ("surface.list", "#2c313c"),
    "#centralwidget{background-color}": ("dialog.background", "#1f232a"),
}

# The tokens that actually exist in the toolkit's dark palette. Overriding a key the palette does not define is a
# silent no-op (set_color writes into the dict, but no painter reads it), which is exactly the kind of false fix this
# project keeps paying for - so unknown keys are reported instead of applied.
KNOWN_TOKENS = frozenset({
    "dialog.button.background",
    "dialog.text",
    "surface.list",
})


def install(app) -> list[str]:
    """Register the palettes, set the dark theme, apply the token map. Returns the lines applied/rejected."""
    from sli_ui_toolkit.palettes import FLUENT_DARK, FLUENT_LIGHT
    from sli_ui_toolkit.theme import ThemeManager

    theme = ThemeManager.get_instance()
    theme.register_palettes(FLUENT_LIGHT, FLUENT_DARK)
    theme.set_theme("dark", app, await_ripples=False)

    applied: list[str] = []
    for rule, (token, hex_value) in QSS_TO_TOKENS.items():
        if token in KNOWN_TOKENS:
            theme.set_color(token, QColor(hex_value))
            applied.append(f"{rule} -> {token} = {hex_value}")
        else:
            applied.append(f"{rule} -> {token}: SKIPPED (token not in the toolkit palette)")
    return applied

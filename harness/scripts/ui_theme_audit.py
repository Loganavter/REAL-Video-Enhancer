"""Audit the look's token table against the two places RVE's colours actually live.

`src/ui/toolkit_theme.py` claims to be the only place a colour is mapped. That claim is only worth something if the
table cannot drift from its sources, and PySide6 lives in the app's venv (the harness suite runs on the portable
interpreter without Qt), so the audit is a script first and a test second:

    <app venv>/bin/python harness/scripts/ui_theme_audit.py      # this run
    pytest harness/tests/test_ui_theme.py                        # wherever PySide6 is installed

Problems it reports: a QPalette role mapped to a colour `QTstyle.Palette()` does not have; a hex in
`QTstyle.styleSheet()` that no token covers; a token the toolkit palette does not define (a silent no-op); a token
missing from the built palette.
"""

from __future__ import annotations

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ROLE_NAMES = {
    "Window", "WindowText", "Base", "AlternateBase", "ToolTipBase", "ToolTipText",
    "Text", "Button", "ButtonText", "BrightText", "Highlight", "HighlightedText",
}


def audit() -> list[str]:
    """Every disagreement between the token table and RVE's own palette/stylesheet, as readable lines."""
    from PySide6.QtGui import QColor, QPalette

    from src.ui import QTstyle, toolkit_theme

    problems: list[str] = []
    palette = QTstyle.Palette()

    roles: dict[str, str] = {}
    for rule, (_token, hex_value) in toolkit_theme.LOOK.items():
        match = re.fullmatch(r"Palette\(\)\.(\w+)", rule)
        if match:
            roles[match.group(1)] = hex_value
    if not roles:
        problems.append("the token table stopped mapping QPalette roles")
    for name, hex_value in roles.items():
        if name not in ROLE_NAMES:
            problems.append(f"Palette().{name}: not a QPalette role")
            continue
        role = getattr(QPalette.ColorRole, name)
        actual = palette.color(role).name()
        if actual.lower() != hex_value.lower():
            problems.append(f"Palette().{name}: table says {hex_value}, QTstyle.Palette() says {actual}")

    from src.ui.toolkit_theme import _hexargb

    tokens = {_hexargb(value) for value in toolkit_theme.TOKENS.values()}
    styled = {_hexargb(value) for value in re.findall(r"#[0-9a-fA-F]{6}", QTstyle.styleSheet())}
    owned = {_hexargb(value) for value in toolkit_theme.QSS_OWNED}
    for missing in sorted(styled - tokens - owned, key=lambda value: value.lower()):
        problems.append(f"QTstyle.styleSheet() uses {missing}, which no token covers and no QSS_OWNED entry explains")

    try:
        from sli_ui_toolkit.palettes import FLUENT_DARK
    except ImportError as exc:                     # the toolkit is not installed in this interpreter
        problems.append(f"cannot import the toolkit palette: {exc}")
        return problems

    for unknown in sorted(set(toolkit_theme.TOKENS) - set(FLUENT_DARK)):
        problems.append(f"token {unknown!r} is not in FLUENT_DARK - set_color would be a silent no-op")

    resolved = toolkit_theme.palette()
    for token in FLUENT_DARK:
        if token not in resolved:
            problems.append(f"palette() drops toolkit token {token!r}")
    for token, hex_value in toolkit_theme.TOKENS.items():
        if token in resolved and _hexargb(resolved[token].name(QColor.NameFormat.HexArgb)) != _hexargb(hex_value):
            problems.append(f"palette()[{token!r}] is {resolved[token].name()}, table says {hex_value}")
    return problems


def main() -> int:
    from src.ui import toolkit_theme

    problems = audit()
    for line in problems:
        print(f"FAIL  {line}")
    print(f"{len(toolkit_theme.LOOK)} mappings audited: "
          + ("clean" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

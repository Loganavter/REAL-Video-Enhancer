"""Audit the look's token table against the palette RVE actually applies.

`src/ui/toolkit_theme.py` claims to be the only place a colour is mapped. That claim is only worth something if the
table cannot drift from what the app paints, and PySide6 lives in the app's venv (the harness suite runs on the
portable interpreter without Qt), so the audit is a script first and a test second:

    <app venv>/bin/python harness/scripts/ui_theme_audit.py      # this run
    pytest harness/tests/test_ui_theme.py                        # wherever PySide6 is installed

Problems it reports: a QPalette role mapped to a colour `toolkit_theme.qt_palette()` does not have; a token the toolkit
palette does not define (a silent no-op); a token missing from the built palette, or resolved to a different colour.

The check this audit used to run - "every hex in `QTstyle.styleSheet()` is covered by a token" - retired with the
stylesheet: the shell's QSS and `QTstyle.styleSheet()` are gone, so the loop that reached into RVE's last stylesheet
had no second source left. The table itself is now the record of every colour the app paints, and `qss-on-adapter` in
`harness/scripts/ui_gates.py` is what keeps a new stylesheet from appearing beside it.
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
    "Link", "LinkVisited",
}


def audit() -> list[str]:
    """Every disagreement between the token table and the palette RVE applies, as readable lines."""
    from PySide6.QtGui import QColor, QPalette

    from src.ui import toolkit_theme

    problems: list[str] = []
    palette = toolkit_theme.qt_palette()

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
            problems.append(f"Palette().{name}: table says {hex_value}, qt_palette() says {actual}")
    for name in toolkit_theme.FALLBACK_ROLES:
        if name not in ROLE_NAMES:
            problems.append(f"FALLBACK_ROLES.{name}: not a QPalette role")
    for name in toolkit_theme.FALLBACK_DISABLED_ROLES:
        if name not in ROLE_NAMES:
            problems.append(f"FALLBACK_DISABLED_ROLES.{name}: not a QPalette role")

    from src.ui.toolkit_theme import _hexargb

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

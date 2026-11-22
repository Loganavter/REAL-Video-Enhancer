"""The debug keys stay environment-gated, and they keep both halves of the diagnosis.

`Ctrl+Shift+R` (repaint the window) and `Ctrl+Shift+H` (hide the widget under the cursor, log its chain) answer the two
opposite questions a stray rectangle can raise: "is it a stale frame?" and "is it painted by this widget?". They are
instruments for a bug report, not a user feature - so the app must arm them only under `RVE_UI_DEBUG_KEYS=1`, exactly
like the inspector.

Pure AST: the portable harness interpreter has no PySide6.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "REAL-Video-Enhancer.py"
MODULE = ROOT / "src" / "ui" / "devtools" / "debug_keys.py"


def test_the_debug_keys_module_keeps_its_flag_and_both_shortcuts() -> None:
    assert MODULE.is_file(), "src/ui/devtools/debug_keys.py is gone - the app imports it behind the flag"
    source = MODULE.read_text(encoding="utf-8")
    for needle in ("RVE_UI_DEBUG_KEYS", "Ctrl+Shift+R", "Ctrl+Shift+H", "def install", "def enabled"):
        assert needle in source, f"the debug keys lost {needle!r}"


def test_the_app_arms_the_debug_keys_only_when_asked() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    installs = [node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "install"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "debug_keys"]
    assert installs, "the app no longer installs the debug keys - the flag would do nothing"

    def guarded_by_debug_keys(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        return any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                   and call.func.attr == "enabled"
                   and isinstance(call.func.value, ast.Name) and call.func.value.id == "debug_keys"
                   for call in ast.walk(node.test))

    guards = [node for node in ast.walk(tree) if guarded_by_debug_keys(node)]
    assert guards, "the debug keys are installed without an enabled() guard - they would arm for every user"
    for node in guards:
        body_ids = {id(inner) for statement in node.body for inner in ast.walk(statement)}
        assert any(id(install) in body_ids for install in installs), (
            "the enabled() guard does not wrap the install call")

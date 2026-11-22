"""The UI inspector is wired behind an explicit opt-in, and it stays that way.

The toolkit's inspector (widget tree, family, config, state, painted regions, theme tokens, constructor site) is the only
instrument that can name "what is that rectangle on my screen" without a screenshot. Wiring it into the app is fine;
wiring it unconditionally is not - its controller installs an application-wide event filter and opens a second top-level
dialog, which is wrong for a render session. `RVE_UI_INSPECTOR=1` is the switch, and these two statements keep it honest:

* the module exists with the env gate and an install entry point;
* the app calls it only inside an `enabled()` guard (a bare `install()` would arm the filter for every user).

Pure AST: the portable harness interpreter has no PySide6.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "REAL-Video-Enhancer.py"
MODULE = ROOT / "src" / "ui" / "devtools" / "ui_inspector.py"


def test_the_inspector_module_states_the_flag_and_the_entry_point() -> None:
    assert MODULE.is_file(), "src/ui/devtools/ui_inspector.py is gone - the app imports it behind the flag"
    source = MODULE.read_text(encoding="utf-8")
    assert "RVE_UI_INSPECTOR" in source, "the opt-in flag is no longer spelled out"
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"enabled", "install"} <= functions, f"the inspector module lost an entry point: {sorted(functions)}"


def test_the_app_arms_the_inspector_only_when_asked() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    installs = [node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "install"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "ui_inspector"]
    assert installs, "the app no longer installs the inspector - the flag would do nothing"

    def guards_the_inspector(node: ast.AST) -> bool:
        """`if ui_inspector.enabled():` - the capture server and the page walker have their own `enabled()` calls."""
        if not isinstance(node, ast.If):
            return False
        for call in ast.walk(node.test):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "enabled"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "ui_inspector"):
                return True
        return False

    guarded = [node for node in ast.walk(tree) if guards_the_inspector(node)]
    assert guarded, "the inspector is installed without an enabled() guard - it would arm for every user"
    for node in guarded:
        body_ids = {id(inner) for statement in node.body for inner in ast.walk(statement)}
        assert any(id(install) in body_ids for install in installs), (
            "the enabled() guard does not wrap the install call")

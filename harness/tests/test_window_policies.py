"""The window's per-window policies are stated ONCE, and a deletion must fail here.

Two things are decided for a whole window instead of at every control site: the control type size (`typography.
apply_control_fonts`) and the wheel policy (`wheel_policy.apply_wheel_requires_focus`). Both are single calls in
`MainWindowShell.build`, so a refactor that drops one is silent in every other test - the pages still build, the suite
still passes, and the behaviour only shows up in the running app (a 12 pt cascade into every control; a wheel that steps
whatever combo the pointer crosses while the page is being scrolled).

Pure AST: the portable harness interpreter has no PySide6.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOW = ROOT / "src" / "ui" / "shell" / "window.py"
POLICY = ROOT / "src" / "ui" / "wheel_policy.py"


def _calls_in(path: Path, function: str) -> set[str]:
    """Every plain function name called anywhere inside `function` of `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {n.func.id for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    raise AssertionError(f"{path.name} has no {function}() - the pin is looking at nothing")


def test_the_window_build_applies_both_window_wide_policies() -> None:
    calls = _calls_in(WINDOW, "build")
    assert "apply_control_fonts" in calls, "the 12 px control type is no longer stated per window"
    assert "apply_wheel_requires_focus" in calls, (
        "the wheel policy is gone: a wheel over a combo steps its value while scrolling the page (owner report "
        "2026-11-22)")


def test_the_wheel_policy_turns_the_toolkit_flag_on_for_every_control_that_has_one() -> None:
    """The policy must SET the flag, and it must not filter by class name - a new control family with the mixin is
    covered by the same walk."""
    source = POLICY.read_text(encoding="utf-8")
    # The toolkit spells both as Qt-style methods on the mixin; the policy reaches them through `getattr`, so they are
    # string literals here rather than call targets (a call-node check would look at `getattr` and prove nothing).
    assert "setWheelRequiresFocus" in source, "the policy no longer sets the toolkit's flag"
    assert "wheelRequiresFocus" in source, "the policy must stay idempotent (skip a control that already states it)"
    assert "__name__" not in source, "class-name filtering would silently skip a new control family"
    tree = ast.parse(source)
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setter"
               for node in ast.walk(tree)), "the policy never calls the setter it looked up"


def test_the_policy_is_applied_to_the_window_not_to_selected_pages() -> None:
    """One walk over the window covers every page; a per-page call is the shape that drifts."""
    calls = _calls_in(WINDOW, "build")
    assert "apply_wheel_requires_focus" in calls
    tree = ast.parse(WINDOW.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "apply_wheel_requires_focus":
            assert node.args, "apply_wheel_requires_focus() must receive the root it walks"
            assert isinstance(node.args[0], ast.Name) and node.args[0].id == "window", (
                "the policy is applied to something other than the whole window")

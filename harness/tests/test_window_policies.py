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
APP = ROOT / "REAL-Video-Enhancer.py"


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


def test_the_window_backdrop_cannot_be_silently_skipped() -> None:
    """No `surface.page` token must stop the build, not leave a window that paints nothing.

    Measured 2026-11-22: building the window without the theme installed left `autoFillBackground=False` and the palette
    Window role at Qt's light default (`#efefef`) - every page/panel is `WA_TranslucentBackground`, so a compositing WM
    shows the desktop through any surface that fails to paint. The branch used to `if _backdrop is not None:` and skip
    everything else, which is exactly the silent version of that failure.
    """
    tree = ast.parse(WINDOW.read_text(encoding="utf-8"))
    raises = [node for node in ast.walk(tree)
              if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
              and isinstance(node.exc.func, ast.Name) and node.exc.func.id == "RuntimeError"]
    messages = " ".join(
        ast.literal_eval(arg) for node in raises for arg in node.exc.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    assert "surface.page" in messages, "the backdrop guard no longer explains the missing token"


def test_the_window_declares_its_surface_opaque_and_paints_every_pixel() -> None:
    """The fill alone is not enough: the surface must be declared opaque, or the compositor blends per pixel.

    Every page and panel inside the window is `WA_TranslucentBackground`, and an `ImageLabel` with no pixmap paints
    nothing - so a pixel the app leaves alone carries alpha 0 in the window's buffer and the compositing WM shows the
    window behind (owner's terminal through the preview pane, 2026-11-22). `WA_OpaquePaintEvent` puts the whole window
    rect into the surface's opaque region; the contract it leans on - "every pixel is painted here" - is the
    `MainWindow.paintEvent` fill below, and the backdrop colour it uses comes from the shell.
    """
    shell = WINDOW.read_text(encoding="utf-8")
    assert "WA_OpaquePaintEvent, True" in shell, (
        "the window no longer declares its surface opaque - per-pixel alpha can bleed the desktop through")
    assert "_backdrop_colour" in shell, "the shell no longer hands the backdrop colour to the window's paintEvent"

    app = ast.parse(APP.read_text(encoding="utf-8"))
    main_window = next((node for node in ast.walk(app)
                        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"), None)
    assert main_window is not None, "MainWindow class is gone"
    paint = next((node for node in main_window.body
                  if isinstance(node, ast.FunctionDef) and node.name == "paintEvent"), None)
    assert paint is not None, (
        "MainWindow has no paintEvent: `WA_OpaquePaintEvent` would then promise an opaque surface nobody fills")
    body = ast.dump(paint)
    assert "fillRect" in body, "the window paintEvent no longer fills the rect"
    assert "_backdrop_colour" in body, "the window paintEvent no longer uses the theme backdrop"

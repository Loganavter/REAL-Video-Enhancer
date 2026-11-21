"""A missing toolkit must fail with a remedy, not with `ModuleNotFoundError` three imports deep.

That exact failure happened: W4 made the toolkit an import-time dependency of the app, every measurement ran with
`PYTHONPATH` set, and `./run.sh` for the owner died on the first page import. This pins the bootstrap's contract:
it resolves the toolkit or explains what to install, and a bogus override does not shadow a working checkout.
"""

from __future__ import annotations

import os
import sys

from src.ui import toolkit_boot


def test_source_resolves_or_is_already_importable() -> None:
    found = toolkit_boot.source()
    if found is None:
        # Already importable in this interpreter - the normal case once the toolkit is installed.
        import sli_ui_toolkit  # noqa: F401
        return
    assert os.path.isfile(os.path.join(found, "sli_ui_toolkit", "__init__.py")), found
    assert found in sys.path


def test_bogus_override_falls_back_to_a_real_checkout(monkeypatch) -> None:
    monkeypatch.setenv("RVE_TOOLKIT_PATH", "/nonexistent/toolkit")
    try:
        found = toolkit_boot.source()
    except ImportError:
        return          # nothing installed and no sibling checkout: the message is what matters, tested below
    assert found is None or os.path.isfile(os.path.join(found, "sli_ui_toolkit", "__init__.py"))


def test_error_names_both_remedies(monkeypatch, tmp_path) -> None:
    """No install, no sibling: the message must say what to do (install it, or point RVE_TOOLKIT_PATH at it)."""
    monkeypatch.setattr(toolkit_boot, "_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("RVE_TOOLKIT_PATH", str(tmp_path / "missing"))
    monkeypatch.setitem(sys.modules, "sli_ui_toolkit", None)   # make `import sli_ui_toolkit` fail deterministically
    try:
        toolkit_boot.source()
    except ImportError as exc:
        message = str(exc)
        assert "RVE_TOOLKIT_PATH" in message
        assert "pip install" in message
    else:
        raise AssertionError("a missing toolkit must raise ImportError with a remedy")
    finally:
        sys.modules.pop("sli_ui_toolkit", None)

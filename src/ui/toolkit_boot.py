"""Make `sli_ui_toolkit` importable before any page imports it.

Why this exists: W4 moved RVE's pages onto toolkit widgets, so the toolkit is now an **import-time dependency of the
application** - not a measurement-only extra. The runtime venv does not carry it (the toolkit is a separate checkout),
and every measurement until now ran with `PYTHONPATH=<toolkit>/src`, which hid the failure: `./run.sh` without that
variable died on `ModuleNotFoundError: No module named 'sli_ui_toolkit'`.

Resolution order, explicit and loud:

1. **already importable** - the normal case once the toolkit is installed into the runtime (nothing to do);
2. `RVE_TOOLKIT_PATH` - an explicit override, e.g. a different checkout or a wheel's location;
3. a **sibling checkout** next to this repository (`../sli-ui-toolkit/src`, then `../sli_ui_toolkit/src`) - the layout
   this machine has;
4. otherwise: an `ImportError` that says what to install or which variable to set, instead of a bare
   `ModuleNotFoundError` three imports deep.

The sibling fallback is a development convenience, not the shipping answer: a packaged build should have the toolkit
installed, and then step 1 returns immediately.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_toolkit(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "sli_ui_toolkit", "__init__.py"))


def source() -> str | None:
    """Return the directory that provided the toolkit, or `None` when it was already importable."""
    try:
        import sli_ui_toolkit  # noqa: F401
        return None
    except ModuleNotFoundError:
        pass

    candidates = []
    override = os.environ.get("RVE_TOOLKIT_PATH", "").strip()
    if override:
        candidates.append(override)
        candidates.append(os.path.join(override, "src"))
    siblings = os.path.dirname(_REPO)
    for name in ("sli-ui-toolkit", "sli_ui_toolkit"):
        candidates.append(os.path.join(siblings, name, "src"))

    for candidate in candidates:
        if candidate and _is_toolkit(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate

    raise ImportError(
        "sli_ui_toolkit is not importable and no checkout was found.\n"
        "  fix: install it into this interpreter (pip install <toolkit checkout>), or\n"
        "       set RVE_TOOLKIT_PATH=/path/to/sli-ui-toolkit/src, or\n"
        f"       keep the toolkit next to this repository (looked in {siblings}/sli-ui-toolkit/src)"
    )


def ensure() -> str | None:
    """Idempotent alias: call it before the first page import."""
    return source()

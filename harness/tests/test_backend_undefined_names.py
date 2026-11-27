"""No undefined name in the backend: the offscreen suite cannot run the torch path, and a NameError there is a dead render.

Evidence (2026-11-27): the pts carry landed as `carry_identity_from(image)` AFTER `del image` in
`UpscaleTorch._call_single`. pyflakes named it; the offscreen suite stayed green (no torch, no GPU in the test
interpreter); the first real render died with `NameError: name 'image' is not defined` five seconds in, and only the
live proof saw it. `TemporalReuse.should_reuse`'s debug branch had the same shape - a `delta` that did not exist in its
scope - for as long as the flag existed, so the one flag meant to explain a reuse raised instead.

The rule is one check, not a style gate: an UNDEFINED NAME outside the vendored model code fails. Unused imports,
f-strings without placeholders and the vendored architectures are not this test's business.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src"

#: Vendored model code: no writer of this tree owns it, and its `from ... import *` helpers are unresolvable to a linter.
VENDORED = ("spandrel", "DRBA", "InterpolateArchs")


def test_no_undefined_name_in_the_backend():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", str(BACKEND)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:                     # pragma: no cover - the interpreter always has a module runner
        pytest.skip("pyflakes is not installed for this interpreter")
    if "No module named pyflakes" in result.stderr:
        pytest.skip("pyflakes is not installed for this interpreter")

    offenders = [
        line for line in result.stdout.splitlines()
        if "undefined name" in line and not any(f"/{vendor}/" in line for vendor in VENDORED)
    ]
    assert not offenders, "a name used before it exists is a dead render, not a lint nit:\n" + "\n".join(offenders)

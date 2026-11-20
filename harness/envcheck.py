"""Check environment claims in the docs against the machine.

Why this exists: on 2026-11-17 the audit found three places in this repo asserting things about the
host that were no longer true - MODEL.md said `ncu` was blocked in three passages (the counters had
been readable since 2026-11-15), harness/docs/wsp/README.md said no TensorRT runtime existed (10.12.0.36 had
been installed that afternoon), and DELEGATION.md told workers to use a bundled interpreter that does
not exist. Each was a number's worth of damage: a crew that follows a stale environment note burns its
budget on `ModuleNotFoundError` and blames the code.

A doc claim about the machine is checkable, so it should be checked the same way a number is: read the
fact, compare, and fail loudly.

Usage:
  python -m harness.envcheck [--root .] [docs ...]
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

VENV_TRT = os.path.expanduser("~/.cache/opencode/venvs/trt310")

# (regex over a line, fact key, what the fact is, the human statement printed on a hit)
CLAIMS = [
    (re.compile(r"\bno\s+TensorRT\b|TensorRT\s+is\s+not\s+installed|no\s+`?libnvinfer`?", re.I),
     "trt", "TensorRT {value} is installed at " + VENV_TRT),
    (re.compile(r"ncu\s+is\s+blocked|ERR_NVGPUCTRPERM|RmProfilingAdminOnly:\s*1\b", re.I),
     "ncu", "RmProfilingAdminOnly is {value} - the counters are readable"),
    (re.compile(r"bundled interpreter|python/\.\.\.` is outside", re.I),
     "bundled_python", "no bundled interpreter exists on this host (checked)"),
    (re.compile(r"system `?python3`? is 3\.14 with \*\*no torch\*\*", re.I),
     "torch", "torch is importable: {value}"),
    # `lean` is not on PATH here, so a doc that says "no toolchain" is the stale claim to catch; the
    # toolchain itself is host state in ~/.elan and was once lost with the tarball in tmpfs.
    (re.compile(r"no Lean toolchain|`?lean`?[^.]{0,20}\bare\s+absent\b", re.I),
     "lean", "Lean is installed: {value}"),
]


def facts() -> dict[str, str]:
    out: dict[str, str] = {}
    params = "/proc/driver/nvidia/params"
    try:
        text = open(params).read()
        m = re.search(r"RmProfilingAdminOnly:\s*(\d+)", text)
        out["ncu"] = m.group(1) if m else "unknown"
    except OSError:
        out["ncu"] = "unreadable"
    trt = os.path.join(VENV_TRT, "bin", "python")
    if os.path.exists(trt):
        try:
            r = subprocess.run([trt, "-c", "import tensorrt as t; print(t.__version__)"],
                               capture_output=True, text=True, timeout=60)
            out["trt"] = r.stdout.strip() or "import failed"
        except Exception as exc:                                   # noqa: BLE001
            out["trt"] = f"probe failed: {exc}"
    else:
        out["trt"] = "not installed (no venv at " + VENV_TRT + ")"
    py = shutil.which("python3")
    try:
        r = subprocess.run([py, "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True, timeout=60)
        out["torch"] = r.stdout.strip() or "absent in python3"
    except Exception:                                              # noqa: BLE001
        out["torch"] = "absent in python3"
    out["bundled_python"] = "absent"
    lean = os.path.expanduser("~/.elan/bin/lean")
    try:
        r = subprocess.run([lean, "--version"], capture_output=True, text=True, timeout=60)
        text = (r.stdout or "") + (r.stderr or "")
        out["lean"] = text.strip().splitlines()[0] if text.strip() else "no output"
    except Exception as exc:                                   # noqa: BLE001
        out["lean"] = f"not runnable ({exc.__class__.__name__})"
    return out


def stale(fact: dict[str, str], path: str, text: str) -> list[tuple[int, str, str]]:
    """Which claims in ``text`` the machine contradicts."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        # a line that is QUOTING an old claim as evidence is not making one: mark it `env-ok`, and
        # audit reports are skipped wholesale (their job is to quote the stale text)
        if "env-ok" in line:
            continue
        for rx, key, statement in CLAIMS:
            if not rx.search(line):
                continue
            value = fact.get(key, "unknown")
            if key == "ncu" and value == "0":
                hits.append((lineno, line.strip()[:110], statement.format(value=value)))
            elif key == "trt" and value and not value.startswith(("not installed", "probe failed")):
                hits.append((lineno, line.strip()[:110], statement.format(value=value)))
            elif key == "torch" and "absent" not in value:
                hits.append((lineno, line.strip()[:110], statement.format(value=value)))
            elif key == "lean" and value and not value.startswith(("not runnable", "no output")):
                hits.append((lineno, line.strip()[:110], statement.format(value=value)))
    return hits


def check(root: str = ".", paths: list[str] | None = None) -> tuple[list[tuple[str, int, str, str]], dict]:
    fact = facts()
    files = paths or sorted(glob.glob(os.path.join(root, "harness", "docs", "**", "*.md"), recursive=True))
    out = []
    for f in files:
        if os.path.basename(f).startswith("AUDIT-"):
            continue
        try:
            text = open(f).read()
        except OSError:
            continue
        for lineno, line, statement in stale(fact, f, text):
            out.append((f, lineno, line, statement))
    return out, fact


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = "."
    if args and args[0] == "--root":
        root, args = args[1], args[2:]
    hits, fact = check(root, args or None)
    print("facts:", ", ".join(f"{k}={v}" for k, v in sorted(fact.items())))
    if hits:
        print(f"STALE CLAIMS ({len(hits)}):")
        for f, lineno, line, statement in hits:
            print(f"  {f}:{lineno}  {line}")
            print(f"      -> {statement}")
    else:
        print("no stale environment claims found")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

"""What a number was measured on: the commit, the card, the library versions, and when.

Why this exists, in one sentence with the evidence: a recorded wall ratio of 1.09-1.14x (W009-W044,
`int8-wall-ab.jsonl`, 2026-11-18) was quoted for a day as if it described the product, while the same
int8 body in that recording ran **44.6-47.9 ms/frame** and the current tree runs it at **26.5 ms/frame** -
so the record described a body that no longer exists, and nothing in the record said so. A number without
its subject is not evidence, it is a rumour with a decimal point.

Nothing here raises: a missing GPU, a missing git, a missing torch - each field degrades to None and the
record says "unknown" rather than refusing to be written at all (a measurement that cannot be recorded is
a measurement that will be re-taken and re-quoted).
"""
from __future__ import annotations

# One implementation, in the product: the backend writes the same stamp into its run report, and two provenance
# functions that disagree would be worse than none. The harness may import this; the backend must never import the
# harness. The import is lazy so the harness still works in a tree where the backend was not installed.
try:
    from backend.src.utils.Provenance import (  # noqa: F401
        device, git_state, model_digest, stamp, versions,
    )
except Exception:  # pragma: no cover - a fallback for a tree without the backend package
    import datetime as _dt
    import hashlib
    import os
    import subprocess

    def _run(cmd, timeout=5.0):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    def git_state(root="."):
        return _run(["git", "-C", str(root), "rev-parse", "HEAD"]), None

    def device():
        return {}

    def versions():
        return {}

    def model_digest(path=None):
        if not path or not os.path.isfile(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def stamp(root=".", model=None):
        commit, dirty = git_state(root)
        return {"measured_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "commit": commit, "dirty": dirty, "model_sha": model_digest(model)}

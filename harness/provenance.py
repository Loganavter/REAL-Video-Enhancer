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

import datetime as _dt
import hashlib
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_state(root: str | Path = ".") -> tuple[str | None, bool | None]:
    """(commit, dirty) of the tree the measurement ran in. `dirty` matters: a result taken on an edited
    working tree cannot be reproduced from the commit alone."""
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(root), "status", "--porcelain"])
    return commit, (bool(status) if status is not None else None)


def device() -> dict:
    """Name and compute capability of GPU 0, from nvidia-smi (no torch import)."""
    info: dict = {}
    row = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader"], timeout=5)
    if row:
        name, cap, driver = (part.strip() for part in row.splitlines()[0].split(","))
        info.update(device=name, capability=cap, driver=driver)
    return info


def versions() -> dict:
    """Library versions that change what the kernels compile to."""
    found: dict = {}
    for module, key in (("torch", "torch"), ("triton", "triton"), ("tensorrt", "tensorrt")):
        try:
            mod = __import__(module)
            found[key] = getattr(mod, "__version__", None)
        except Exception:
            continue
    return found


def model_digest(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def stamp(root: str | Path = ".", model: str | None = None) -> dict:
    """The full provenance tuple, with every field best-effort."""
    commit, dirty = git_state(root)
    return {
        "measured_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model_sha": model_digest(model),
        **device(),
        **versions(),
    }

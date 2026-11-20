"""What a render ran on: the commit, the card, the library versions, the model.

The backend writes this into its run report; `harness/provenance.py` imports it from here rather than keeping a
second copy, because two provenance implementations that disagree are worse than none (the harness may depend on
the product, never the other way round).

Everything degrades to None instead of raising: a measurement that cannot be recorded is a measurement that will be
taken again and quoted from memory.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import subprocess


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_state(root: str = ".") -> tuple[str | None, bool | None]:
    commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    status = _run(["git", "-C", root, "status", "--porcelain"])
    return commit, (bool(status) if status is not None else None)


def device() -> dict:
    row = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"], timeout=5)
    if not row:
        return {}
    name, capability, driver = (part.strip() for part in row.splitlines()[0].split(","))
    return {"device": name, "capability": capability, "driver": driver}


def versions() -> dict:
    found: dict = {}
    for module, key in (("torch", "torch"), ("triton", "triton"), ("tensorrt", "tensorrt")):
        try:
            found[key] = getattr(__import__(module), "__version__", None)
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


def stamp(root: str = ".", model: str | None = None) -> dict:
    commit, dirty = git_state(root)
    return {
        "measured_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model_sha": model_digest(model),
        **device(),
        **versions(),
    }

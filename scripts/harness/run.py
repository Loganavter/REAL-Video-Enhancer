"""Process lifecycle: a run returns a tuple, and a probe has three states, not two.

Class 4 of the retraction list was "intent as completion": a call was started, the harness did not
own it, and a NameError inside the probe was read as a verdict about fp16. Two rules follow and are
enforced here:

* ``run()`` waits. It never detaches, and on timeout it kills the process *group* (a lone kill leaves
  the child that actually holds the GPU). A detached job is not a run, it is a hope.
* ``Probe.status`` is ``pass | fail | error``, and ``error`` is not a result. ``ResultsTable.add``
  refuses it, so a crash cannot appear in a table of populations - the only place it used to land.

A run is claimable only when it exited 0, did not time out, and its parser produced a schema.
``claimable()`` is the single door: a claim may cite nothing else.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

GPU_LOCK = "/var/tmp/opencode/gpu.lock"


class RunError(RuntimeError):
    pass


@dataclass
class Run:
    cmd: Sequence[str]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    wall_s: float = 0.0
    timed_out: bool = False
    error: str = ""
    parsed: Any = None
    artifacts: dict = field(default_factory=dict)
    run_id: str = ""
    cwd: str = ""
    gpu: bool = False

    @property
    def status(self) -> str:
        if self.timed_out:
            return "error"
        if self.error:
            return "error"
        if self.exit_code != 0:
            return "error"
        return "ok"

    @property
    def error_kind(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.error.startswith("unparsed"):
            return "unparsed"
        if self.error:
            return "exception"
        if self.exit_code not in (0, None):
            return "exit"
        return ""

    def summary(self) -> str:
        return (f"[{self.status.upper()}] {self.run_id or 'run'} exit={self.exit_code} "
                f"{self.wall_s:.1f}s{(' ' + self.error) if self.error else ''}")


def run(cmd: Sequence[str], *, cwd: str | None = None, timeout: float = 600.0,
        parse: Callable[[str], Any] | None = None, gpu: bool = False,
        env: dict | None = None, run_id: str = "") -> Run:
    """Run a command to completion and return the tuple. Never backgrounded, never partially read."""
    if gpu:
        # One GPU job at a time: a blocked job exits instead of queueing silently.
        cmd = ["flock", "-n", GPU_LOCK, *cmd]
    t0 = time.time()
    proc = subprocess.Popen(list(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True,
                            env={**os.environ, **(env or {})})
    r = Run(cmd=cmd, run_id=run_id or f"run-{int(t0)}", cwd=cwd or os.getcwd(), gpu=gpu)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with _suppress():
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        r.timed_out = True
        r.error = f"timeout after {timeout:g}s (process group killed)"
    r.stdout, r.stderr = out or "", err or ""
    r.exit_code = proc.returncode
    r.wall_s = time.time() - t0
    if parse is not None and not r.timed_out:
        try:
            r.parsed = parse(r.stdout)
        except Exception as exc:                       # noqa: BLE001 - the point is to catch everything
            r.error = f"unparsed: {type(exc).__name__}: {exc}"
    return r


class _suppress:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return True


def claimable(r: Run) -> bool:
    """The only evidence a claim may cite."""
    return r.status == "ok" and r.parsed is not None


# --------------------------------------------------------------------------------------------------
# three-state probe
# --------------------------------------------------------------------------------------------------
@dataclass
class Probe:
    name: str
    status: str                     # pass | fail | error
    detail: str = ""
    run: Run | None = None
    data: dict = field(default_factory=dict)
    synthetic: bool = False        # an in-process check that deliberately ran no command

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "error"):
            raise RunError(f"probe {self.name}: unknown status {self.status!r}")
        if self.status in ("pass", "fail"):
            if self.run is not None and not claimable(self.run):
                raise RunError(f"probe {self.name}: a pass/fail needs a claimable run, got {self.run.summary()}")
            if self.run is None and not self.synthetic:
                raise RunError(f"probe {self.name}: no run and not declared synthetic - a probe that "
                               f"never ran is not a result")

    @property
    def is_result(self) -> bool:
        return self.status in ("pass", "fail")


def probe(name: str, check: Callable[[], bool], *, run: Run | None = None,
          prelude: str = "", synthetic: bool = False) -> Probe:
    """Three states. An exception is ``error``, never ``fail``: a broken probe measures nothing."""
    if run is not None and run.status != "ok":
        return Probe(name=name, status="error", run=run, synthetic=synthetic,
                     detail=f"run not ok: {run.summary()} {run.error}".strip())
    try:
        ok = bool(check())
    except Exception as exc:                       # noqa: BLE001
        return Probe(name=name, status="error", run=run, synthetic=synthetic,
                     detail=f"{prelude}{type(exc).__name__}: {exc}")
    return Probe(name=name, status="pass" if ok else "fail", run=run, detail=prelude,
                 synthetic=synthetic)


@dataclass
class ResultsTable:
    """A table of populations. An ``error`` row never enters it - that was the whole bug."""
    rows: list[Probe] = field(default_factory=list)
    skipped: list[Probe] = field(default_factory=list)

    def add(self, p: Probe) -> None:
        if not p.is_result:
            self.skipped.append(p)
            raise RunError(f"refusing to add probe {p.name!r} with status {p.status!r} to a results "
                           f"table: an error is not a result ({p.detail})")
        self.rows.append(p)

    def render(self) -> str:
        lines = [f"{'probe':<32} {'status':<6} detail"]
        for p in self.rows:
            lines.append(f"{p.name:<32} {p.status:<6} {p.detail}")
        if self.skipped:
            lines.append(f"\n{len(self.skipped)} probe(s) not in the table:")
            for p in self.skipped:
                lines.append(f"  {p.name}: {p.status} - {p.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# artifact side: nothing is read by eye
# --------------------------------------------------------------------------------------------------
@dataclass
class Schema:
    """A parsed artifact. ``require`` is the only accessor callers should use - it fails loudly on a
    field the artifact did not carry, instead of substituting a default from memory."""
    kind: str
    fields: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    source: str = ""

    def __getitem__(self, key: str) -> Any:
        if key in self.fields:
            return self.fields[key]
        raise RunError(f"{self.kind}: field {key!r} is not in the artifact "
                       f"(have {sorted(self.fields)}, missing {self.missing})")

    def require(self, *keys: str) -> "Schema":
        for k in keys:
            _ = self[k]
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)


_PTXAS_REG = re.compile(r"Used\s+(\d+)\s+registers")
_PTXAS_SMEM = re.compile(r"(\d+)\s+bytes\s+(?:smem|stack frame|spill stores|spill loads)")
_PTXAS_SPILL = re.compile(r"(\d+)\s+bytes\s+spill\s+(stores|loads)")


def parse_ptxas(text: str) -> Schema:
    """Static smem is what ptxas prints. Dynamic smem is a *launch* parameter and is deliberately
    absent: ``parse_ptxas(...).require('dynamic_smem')`` must fail unless the caller supplied it,
    because "smem 0" read as "no staging" is class 5."""
    f: dict[str, Any] = {}
    if m := _PTXAS_REG.search(text):
        f["registers"] = int(m.group(1))
    if m := re.search(r"(\d+)\s+bytes\s+smem", text):
        f["static_smem"] = int(m.group(1))
    for m in _PTXAS_SPILL.finditer(text):
        f[f"spill_{m.group(2)}"] = int(m.group(1))
    missing = [k for k in ("registers", "static_smem") if k not in f]
    missing.append("dynamic_smem")
    return Schema(kind="ptxas", fields=f, missing=missing, source="ptxas stderr")


def parse_engine_inspector(text: str) -> Schema:
    """TensorRT IEngineInspector JSON -> layer counts. A local graph edit changes global
    partitioning, so a timing claim after a graph edit has to cite these counts, not just ms."""
    layers = json.loads(text)
    if isinstance(layers, dict):
        layers = layers.get("Layers", layers.get("layers", []))
    names = [str(l.get("Name", l.get("name", ""))) for l in layers]
    fused = sum(1 for n in names if n.startswith("__myl_"))
    reformat = sum(1 for n in names if "Reformat" in n)
    pointwise = sum(1 for n in names if "PointWise" in n or n.startswith("PWN"))
    conv = sum(1 for n in names if "Conv" in n or "Gemm" in n or "xmma" in n.lower())
    f = {"total": len(names), "fused_mylin": fused, "reformat": reformat,
         "pointwise": pointwise, "convolutions": conv, "names": names}
    missing = []
    if not names:
        missing = ["total", "fused_mylin", "reformat", "pointwise", "convolutions"]
    return Schema(kind="engine_inspector", fields=f, missing=missing, source="IEngineInspector JSON")


def parse_checkpoint(path: str) -> Schema:
    """Checkpoint shapes without torch: a .pth is a zip holding a pickle and raw storage.

    Class 5 in the list was ``convs[0]`` read as the head while it was the second convolution, and
    a 3-channel head weight that was really ``(64, 8, 3, 3)`` with five zero planes. Both are
    answered by reading the tensor names and shapes out of the file instead of recalling them.
    """
    import pickle
    import zipfile

    with zipfile.ZipFile(path) as z:
        pkl = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not pkl:
            raise RunError(f"{path}: not a torch checkpoint (no data.pkl in the zip)")
        shapes: dict[str, tuple] = {}

        class _Unpickler(pickle.Unpickler):
            def persistent_load(self, pid):       # noqa: N802 - pickle API
                return ("S", pid[2])

            def find_class(self, mod, name):      # noqa: N802 - pickle API
                if name == "_rebuild_tensor_v2":
                    def rebuild(storage, offset, size, stride, *_):
                        return ("T", tuple(size))
                    return rebuild
                if name == "_rebuild_parameter":
                    return lambda data, *a, **k: data
                import collections
                if name == "OrderedDict":
                    return collections.OrderedDict
                return type(name, (), {"__init__": lambda self, *a, **k: None,
                                       "__setstate__": lambda self, st: None})
        sd = _Unpickler(z.open(pkl[0])).load()
    if isinstance(sd, dict) and "params" in sd and isinstance(sd["params"], dict):
        sd = sd["params"]
    for k, v in sd.items():
        if isinstance(v, tuple) and v and v[0] == "T":
            shapes[k] = v[1]
    n = 1
    total = 0
    for s in shapes.values():
        n = 1
        for d in s:
            n *= d
        total += n
    convs = sorted(k for k in shapes if k.endswith(".weight") and len(shapes[k]) == 4)
    slopes = sorted(k for k in shapes if k.endswith(".weight") and len(shapes[k]) == 1)
    biases = sorted(k for k in shapes if k.endswith(".bias"))
    return Schema(kind="checkpoint", fields={
        "tensors": len(shapes), "params": total, "bytes_fp16": total * 2,
        "shapes": shapes, "convs": convs, "slope_tensors": slopes, "bias_tensors": biases,
        "body_layers": len([k for k in convs if k.startswith("body")]),
    }, missing=[] if shapes else ["tensors"], source=path)

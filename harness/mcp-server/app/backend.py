"""Handlers for the RVE live MCP tools: the app's control channel, a leased display, and screen truth.

Two layers, kept apart on purpose:

* **the instance** — a private Xvfb display plus the real window, started and stopped through `harness/scripts/live.sh`
  (it encodes the traps: never the user's display, kill by PID, software GL, emptied GTK modules). This module only
  reads its state files;
* **the channel** — the app's always-on control directory (`request` file in, `answers/<id>.json` out). One request per
  call, an id per request, so two clients never eat each other's reply.

Nothing here guesses: an answer that is missing is reported as a timeout with the instance status attached, not retried
forever.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 15.0


class ToolError(RuntimeError):
    """A tool failure with a hint the caller can act on."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


def _repo_root() -> Path:
    override = os.environ.get("RVE_REPO_ROOT")
    if override:
        return Path(override).resolve()
    # <repo>/harness/mcp-server/app/backend.py -> <repo>
    return Path(__file__).resolve().parents[3]


class RveBackend:
    """The live instance's state files and the app's control channel."""

    def __init__(self, *, repo_root: Path | None = None, state_dir: Path | None = None,
                 control_dir: Path | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.repo_root = (repo_root or _repo_root()).resolve()
        self.state_dir = Path(state_dir or os.environ.get("RVE_LIVE_STATE", "/tmp/opencode/live"))
        self._control_override = control_dir or (Path(os.environ["RVE_CONTROL_DIR"])
                                                 if os.environ.get("RVE_CONTROL_DIR") else None)
        self.timeout_s = float(os.environ.get("RVE_MCP_TIMEOUT", timeout_s))
        self._live = self.repo_root / "harness" / "scripts" / "live.sh"

    # -- instance ------------------------------------------------------

    def _read(self, name: str) -> str | None:
        path = self.state_dir / name
        try:
            return path.read_text().strip()
        except OSError:
            return None

    def _app_pid(self) -> int | None:
        raw = self._read("app.pid")
        if not raw or not raw.isdigit():
            return None
        return int(raw)

    def _running(self) -> bool:
        pid = self._app_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def control_dir(self) -> Path:
        if self._control_override is not None:
            return self._control_override
        recorded = self._read("control_dir")
        if recorded:
            return Path(recorded)
        candidates = [
            Path("/tmp/opencode/.rve-control"),
            self.repo_root / ".rve-control",
            Path(os.environ.get("TMPDIR", "/tmp")) / "opencode" / ".rve-control",
        ]
        ready = [c for c in candidates if (c / "ready").exists()]
        if ready:
            return ready[0]
        existing = [c for c in candidates if c.is_dir()]
        return existing[0] if existing else candidates[0]

    def _status(self) -> dict:
        control = self.control_dir()
        log = self.state_dir / "app.log"
        tail = ""
        if log.exists():
            try:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
            except OSError:
                tail = ""
        return {
            "running": self._running(),
            "display": self._read("display"),
            "control_dir": str(control),
            "app_pid": self._app_pid(),
            "log_tail": tail,
        }

    def _live_command(self, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
        if not self._live.exists():
            raise ToolError(f"live harness missing at {self._live}",
                            "restore harness/scripts/live.sh from the repo; every instance tool depends on it")
        completed = subprocess.run(["bash", str(self._live), *args], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            raise ToolError(f"live.sh {' '.join(args)} failed",
                            (completed.stderr or completed.stdout).strip()[-800:] or "no output")
        return completed

    # -- control channel -----------------------------------------------

    def _request(self, verb: str, timeout: float | None = None) -> dict:
        root = self.control_dir()
        if not self._running():
            raise ToolError("no live instance is running",
                            "call instance_boot first (it leases a private display and starts the window)")
        answers = root / "answers"
        answers.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex[:10]
        target = answers / f"{request_id}.json"
        try:
            (root / "request").write_text(f"{verb}#{request_id}", encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"cannot write the app's request file in {root}", str(exc)) from exc
        deadline = time.time() + float(timeout or self.timeout_s)
        while time.time() < deadline:
            if target.exists():
                for _ in range(5):  # the app may still be writing it
                    try:
                        return json.loads(target.read_text(encoding="utf-8"))
                    except ValueError:
                        time.sleep(0.02)
                raise ToolError(f"the app's answer file {target} is not valid JSON",
                                "the control server wrote a partial answer; retry once")
            time.sleep(0.05)
        raise ToolError(f"the app did not answer {verb!r} within {timeout or self.timeout_s:.0f}s",
                        "check instance_status: a crashed window leaves its pid file behind")

    # -- tools ----------------------------------------------------------

    def instance_status(self) -> dict:
        return self._status()

    def instance_boot(self, *, width: int = 1163, height: int = 900, timeout_s: float = 60) -> dict:
        if self._running():
            return {**self._status(), "already_running": True}
        # live_boot.py reads the window size from the environment of the boot; set it for this call only.
        previous_w, previous_h = os.environ.get("RVE_LIVE_WIDTH"), os.environ.get("RVE_LIVE_HEIGHT")
        os.environ["RVE_LIVE_WIDTH"], os.environ["RVE_LIVE_HEIGHT"] = str(int(width)), str(int(height))
        try:
            self._live_command("up", timeout=max(60.0, timeout_s + 30))
        finally:
            if previous_w is None:
                os.environ.pop("RVE_LIVE_WIDTH", None)
            else:
                os.environ["RVE_LIVE_WIDTH"] = previous_w
            if previous_h is None:
                os.environ.pop("RVE_LIVE_HEIGHT", None)
            else:
                os.environ["RVE_LIVE_HEIGHT"] = previous_h
        return {**self._status(), "already_running": False}

    def instance_stop(self) -> dict:
        if self._running() or self._app_pid() is not None:
            self._live_command("down", timeout=60)
        return self._status()

    def window_screenshot(self, *, label: str | None = None) -> dict:
        display = self._read("display")
        if not display:
            raise ToolError("no display recorded for the live instance", "call instance_boot first")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = f"-{label}" if label else ""
        target = self.state_dir / f"shot-{stamp}{suffix}.png"
        completed = subprocess.run(["import", "-window", "root", str(target)],
                                   env={**os.environ, "DISPLAY": display}, capture_output=True, text=True, timeout=60)
        if completed.returncode != 0 or not target.exists():
            raise ToolError("screen capture failed", (completed.stderr or "import wrote nothing").strip()[-500:])
        width = height = None
        try:  # PNG IHDR: bytes 16..24 are width and height, big-endian - no imaging dependency on this server
            import struct

            with open(target, "rb") as handle:
                header = handle.read(24)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                width, height = struct.unpack(">II", header[16:24])
        except OSError:
            pass
        return {"path": str(target), "width": width, "height": height}

    def window_grab(self, *, path: str | None = None) -> dict:
        verb = f"grab:{path}" if path else "grab:"
        answer = self._request(verb, timeout=30)
        target = answer.get("path")
        if not target:
            raise ToolError("the app did not report a grab path", str(answer)[:400])
        size = answer.get("size") or []
        return {"path": target,
                "width": int(size[0]) if len(size) > 0 else None,
                "height": int(size[1]) if len(size) > 1 else None,
                "answer": answer}

    def where(self) -> dict:
        return {"verb": "where:", "answer": self._request("where:")}

    def state(self) -> dict:
        return {"verb": "state:", "answer": self._request("state:")}

    def control_inventory(self) -> dict:
        return {"verb": "probe:", "answer": self._request("probe:")}

    def widget_at(self, *, x: int, y: int) -> dict:
        return {"verb": f"probe_at:{int(x)},{int(y)}", "answer": self._request(f"probe_at:{int(x)},{int(y)}")}

    def rhi_probe(self, *, hold_ms: int = 0, x: int | None = None, y: int | None = None) -> dict:
        verb = f"rhi:{int(hold_ms)}"
        if x is not None and y is not None:
            verb = f"rhi:{int(hold_ms)}@{int(x)},{int(y)}"
        return {"verb": verb, "answer": self._request(verb, timeout=45)}

    def repaint(self) -> dict:
        return {"verb": "repaint:", "answer": self._request("repaint:")}

    def _xdo(self, *args: str) -> dict:
        display = self._read("display")
        if not display:
            raise ToolError("no display recorded for the live instance", "call instance_boot first")
        completed = subprocess.run(["xdotool", *args], env={**os.environ, "DISPLAY": display},
                                   capture_output=True, text=True, timeout=60)
        if completed.returncode != 0:
            raise ToolError(f"xdotool {' '.join(args)} failed", (completed.stderr or "").strip()[-400:])
        return {"ok": True, "stdout": completed.stdout.strip()[:400]}

    def click_at(self, *, x: int, y: int, button: int = 1) -> dict:
        return self._xdo("mousemove", str(int(x)), str(int(y)), "click", str(int(button)))

    def drag_from_to(self, *, x1: int, y1: int, x2: int, y2: int, steps: int = 8) -> dict:
        steps = max(1, int(steps))
        self._xdo("mousemove", str(int(x1)), str(int(y1)))
        self._xdo("mousedown", "1")
        try:
            for step in range(1, steps + 1):
                x = int(x1 + (x2 - x1) * step / steps)
                y = int(y1 + (y2 - y1) * step / steps)
                self._xdo("mousemove", str(x), str(y))
                time.sleep(0.01)
        finally:
            self._xdo("mouseup", "1")
        return {"ok": True}

    def key_press(self, *, key: str) -> dict:
        return self._xdo("key", key)

    def type_text(self, *, text: str) -> dict:
        return self._xdo("type", text)

    def control_read(self, *, verb: str, timeout_s: float | None = None) -> dict:
        from . import catalog

        if not any(verb.startswith(prefix) for prefix in catalog.READ_VERBS):
            raise ToolError(f"{verb.split(':')[0]!r} is not a read verb",
                            f"read verbs: {', '.join(catalog.READ_VERBS)}; use control_write for the rest")
        return {"verb": verb, "answer": self._request(verb, timeout=timeout_s)}

    def control_write(self, *, verb: str, timeout_s: float | None = None) -> dict:
        from . import catalog

        if not any(verb.startswith(prefix) for prefix in catalog.WRITE_VERBS):
            raise ToolError(f"{verb.split(':')[0]!r} is not a write verb",
                            f"write verbs: {', '.join(catalog.WRITE_VERBS)}; use control_read for the rest")
        return {"verb": verb, "answer": self._request(verb, timeout=timeout_s)}

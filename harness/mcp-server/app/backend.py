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
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 15.0

#: How long a SIGTERM gets before the stop path escalates to SIGKILL. A Qt window closes in well under a second; the
#: grace exists for a render that is inside a long native call, not for a hung process.
STOP_GRACE_S = 10.0


def _pid_alive(pid: int) -> bool:
    """Whether `pid` is a live process.

    `os.kill(pid, 0)` also succeeds for a zombie - a process that exited but whose parent has not reaped it yet - and
    that is exactly what a just-killed child looks like to the caller that killed it. Reading the state letter from
    `/proc/<pid>/stat` keeps `running` honest there; hosts without /proc fall back to the signal probe.

    0 and negative values are process GROUPS, not processes: `kill(0, 0)` probes this server's own group and answers
    "alive", and signalling 0 would hit every process the server was started with. A pid file holding 0 is garbage.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        # "pid (comm) state ..." - comm may itself contain ')' and spaces, so split on the LAST ')'.
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


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
        base_state = Path(state_dir or os.environ.get("RVE_LIVE_STATE", "/tmp/opencode/live"))
        # live.sh writes the instance's state dir (a lease moves it under the lease's runtime dir) into
        # <base>/state_dir; the server follows the same pointer, so `live.sh up` and the MCP tools agree on the
        # instance without an explicit env var.
        pointer = base_state / "state_dir"
        try:
            recorded = pointer.read_text().strip()
        except OSError:
            recorded = ""
        self.state_dir = Path(recorded) if recorded else base_state
        #: An explicit state dir is the caller's (the tests pass one); otherwise the pointer is re-read on every use.
        #: It used to be resolved ONCE, at construction: the server booted an instance, `live.sh` moved the pointer to
        #: the new tree, and every later `instance_status` still read the directory it had cached at startup - so a
        #: warm instance reported "no live instance", `instance_stop` did not stop it, and an agent concluded the MCP
        #: was not wired at all (measured 2026-11-24: pid 987708 alive while `running: false`).
        self._state_base = base_state
        self._state_explicit = state_dir is not None
        self._control_override = control_dir or (Path(os.environ["RVE_CONTROL_DIR"])
                                                 if os.environ.get("RVE_CONTROL_DIR") else None)
        self.timeout_s = float(os.environ.get("RVE_MCP_TIMEOUT", timeout_s))
        self._live = self.repo_root / "harness" / "scripts" / "live.sh"

    # -- instance ------------------------------------------------------

    def _refresh_state_dir(self) -> None:
        """Follow `live.sh`'s pointer again: it moves whenever a boot leases a new tree."""
        if self._state_explicit:
            return
        try:
            recorded = (self._state_base / "state_dir").read_text().strip()
        except OSError:
            return
        if recorded:
            self.state_dir = Path(recorded)

    def _state_candidates(self) -> list[Path]:
        """Every tree that may hold this instance's records: the pointer's tree and the base tree.

        `live.sh up` writes pid/display/control into ONE tree and a pointer to it into the base dir. The two disagree in
        the cases that matter: a boot without a lease writes into the BASE tree and leaves an older pointer in place,
        and a crash leaves the pointer at a tree whose pid is gone. Reading only the pointer is what made a live app
        report `running: false` (measured 2026-11-24).
        """
        self._refresh_state_dir()
        trees = [self.state_dir]
        if not self._state_explicit and self._state_base != self.state_dir:
            trees.append(self._state_base)
        return trees

    @staticmethod
    def _read_from(state: Path, name: str) -> str | None:
        try:
            return (state / name).read_text().strip()
        except OSError:
            return None

    @classmethod
    def _pid_from(cls, state: Path) -> int | None:
        raw = cls._read_from(state, "app.pid")
        if not raw or not raw.isdigit():
            return None
        return int(raw)

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def _state(self) -> Path:
        """The state tree that actually describes the instance, and the one every read must use.

        Order: a tree whose recorded `app.pid` is alive (the newest one if two exist), then a tree that still holds a pid
        record (the stale record `instance_stop` acts on and its reason names), then the pointer's tree. The pointer is
        followed, not trusted.
        """
        trees = self._state_candidates()
        alive = [tree for tree in trees if (pid := self._pid_from(tree)) is not None and _pid_alive(pid)]
        pool = alive or [tree for tree in trees if (tree / "app.pid").exists()]
        chosen = max(pool, key=lambda tree: self._mtime(tree / "app.pid")) if pool else trees[0]
        self.state_dir = chosen
        return chosen

    def _read(self, name: str) -> str | None:
        return self._read_from(self._state(), name)

    def _app_pid(self) -> int | None:
        return self._pid_from(self._state())

    def _running(self) -> bool:
        pid = self._app_pid()
        return pid is not None and _pid_alive(pid)

    def _lease(self, state: Path) -> str | None:
        return self._read_from(state, "lease") or None

    def control_dir(self) -> Path:
        if self._control_override is not None:
            return self._control_override
        state = self._state()
        recorded = self._read_from(state, "control_dir")
        if recorded:
            return Path(recorded)
        # `live.sh`'s own default for a tree that records no path: `<state>/control`. Nothing else is guessed - the
        # checkout's `.rve-control` belongs to a window this server did not boot, and following it is how `status`
        # used to point at a stale lease while the real app answered under the lease tree (measured 2026-11-24).
        return state / "control"

    def _status(self) -> dict:
        state = self._state()
        pid = self._pid_from(state)
        running = pid is not None and _pid_alive(pid)
        control = self.control_dir()
        log = state / "app.log"
        tail = ""
        if log.exists():
            try:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
            except OSError:
                tail = ""
        lease = self._lease(state)
        stale = None
        if pid is not None and not running:
            stale = (f"app.pid records {pid}, which is not running; lease {lease or 'none'} in {state} is stale "
                     "(instance_stop clears it, instance_boot reclaims it)")
        return {
            "running": running,
            "display": self._read_from(state, "display") if running else None,
            "control_dir": str(control),
            "app_pid": pid,
            # A lease is reported only for the instance that is actually up: a dead pid's lease is a leftover, and
            # reporting it as if it owned a display is what sent an agent to a lease nobody holds.
            "lease": lease if running else None,
            "state_dir": str(state),
            "stale": stale,
            "log_tail": tail,
        }

    def _terminate(self, pid: int, grace: float = STOP_GRACE_S) -> bool:
        """SIGTERM `pid`, wait up to `grace`, then SIGKILL. True when the process is gone afterwards."""
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        deadline = time.time() + grace
        while time.time() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(0.05)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return not _pid_alive(pid)
        for _ in range(40):
            if not _pid_alive(pid):
                return True
            time.sleep(0.05)
        return not _pid_alive(pid)

    def _release_lease(self, lease_id: str) -> bool:
        """Hand a leased display back to the machine's pool (`live-lease release <id>`).

        `live.sh down` does this when it runs in the same tree, but the server cannot rely on that: it may hold an
        explicit tree, and the pointer is removed first. A host without the tool has no lease to release.
        """
        if not lease_id:
            return False
        try:
            completed = subprocess.run(["live-lease", "release", str(lease_id)],
                                       capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0

    def _stop_owned_xvfb(self, state: Path) -> int | None:
        """Kill the Xvfb this instance started (`xvfb_owned` 1), by the pid in its own file. A leased display is the
        lease's business, not ours."""
        if (self._read_from(state, "xvfb_owned") or "") != "1":
            return None
        raw = self._read_from(state, "xvfb.pid") or ""
        pid = int(raw) if raw.isdigit() else None
        if pid is not None and _pid_alive(pid):
            self._terminate(pid)
        self._unlink(state / "xvfb.pid")
        self._unlink(state / "xvfb_owned")
        return pid

    def _forget_state_pointer(self, state: Path) -> None:
        """Drop `<base>/state_dir` when it names the tree just cleared, or a tree that has no live app: the next boot
        must resolve the base tree instead of following a record nobody owns."""
        if self._state_explicit:
            return
        pointer = self._state_base / "state_dir"
        try:
            recorded = pointer.read_text().strip()
        except OSError:
            return
        if not recorded:
            return
        named = Path(recorded)
        pid = self._pid_from(named)
        if recorded == str(state) or pid is None or not _pid_alive(pid):
            try:
                pointer.unlink()
            except OSError:
                pass

    def _clear_stale_records(self, state: Path) -> dict | None:
        """Drop the records of an app that is not running in `state`, releasing the lease it left behind.

        The pid is checked against liveness first: a tree whose app IS running is never touched here.
        """
        pid = self._pid_from(state)
        if pid is not None and _pid_alive(pid):
            return None
        lease = self._lease(state)
        if pid is None and lease is None:
            return None
        released = self._release_lease(lease) if lease else False
        for name in ("app.pid", "lease", "display", "control_dir", "xvfb_owned", "xvfb.pid"):
            self._unlink(state / name)
        self._forget_state_pointer(state)
        return {"state_dir": str(state), "stale_pid": pid, "stale_lease": lease, "lease_released": released}

    def _reclaim_stale(self) -> dict | None:
        """Clear the records of an app that is no longer running, before a boot: a stale pid must not be mistaken for an
        instance and its lease must go back to the pool instead of being overwritten by the next boot.

        Every candidate tree is checked: the leftovers can be in the pointer's tree, in the base tree, or in both.
        """
        cleared = [record for tree in self._state_candidates()
                   if (record := self._clear_stale_records(tree)) is not None]
        if not cleared:
            return None
        return {"cleared": cleared, "lease_released": any(record["lease_released"] for record in cleared)}

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
        reclaimed = self._reclaim_stale()
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
        answer = {**self._status(), "already_running": False, "reclaimed": reclaimed}
        if not answer["running"]:
            # `live.sh up` returning 0 is not the same as a window: a boot that lost the race for the display or died
            # in startup leaves its records behind. Say so with the pid and the log, instead of answering
            # `running: false, already_running: false` over a live process (the 2026-11-24 defect).
            raise ToolError(
                f"live.sh up returned but {answer['state_dir']} records no running app (app.pid {answer['app_pid']})",
                f"the boot produced no window; last log lines: {answer['log_tail'][-500:] or '(no log)'}")
        return answer

    def instance_stop(self) -> dict:
        """Stop the app this server's state files name, by PID, and release the lease that boot recorded.

        Implemented here rather than by delegating to `live.sh down`: that script reads the pointer tree itself, so a
        server holding an explicit tree, or a tree the pointer no longer names, stopped nothing and left the pid and the
        lease alive (measured 2026-11-24). The pid is the only process touched - never a pattern, which would match the
        shell running the stop.
        """
        state = self._state()
        pid = self._pid_from(state)
        stopped_pid: int | None = None
        terminated = False
        if pid is not None:
            if _pid_alive(pid):
                terminated = self._terminate(pid)
                if terminated:
                    stopped_pid = pid
            self._unlink(state / "app.pid")
        xvfb_pid = self._stop_owned_xvfb(state)
        lease = self._lease(state)
        released = self._release_lease(lease) if lease else False
        if lease:
            self._unlink(state / "lease")
        control = self.control_dir()
        self._unlink(control / "ready")
        self._forget_state_pointer(state)
        # Leftovers in the other candidate tree (a dead pid, its lease) are cleared too: after the stop there is no
        # instance at all, and a stale record would make the next `status` name a lease nobody holds.
        for other in self._state_candidates():
            if other != state:
                self._clear_stale_records(other)
        # The answer describes the tree that was stopped, not whatever stale tree the pointer left behind.
        self.state_dir = state
        return {**self._status(), "stopped_pid": stopped_pid, "cleared_pid": pid, "terminated": terminated,
                "xvfb_pid": xvfb_pid, "lease": None, "lease_released": released}

    def window_screenshot(self, *, label: str | None = None) -> dict:
        state = self._state()
        if not self._running():
            raise ToolError("no live instance is running", "call instance_boot first")
        display = self._read_from(state, "display")
        if not display:
            raise ToolError("no display recorded for the live instance", "call instance_boot first")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = f"-{label}" if label else ""
        target = state / f"shot-{stamp}{suffix}.png"
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
        state = self._state()
        if not self._running():
            # A stale `display` record names an X server that may since belong to another lane: typing into it is the
            # input-device version of acting on a lease nobody holds.
            raise ToolError("no live instance is running", "call instance_boot first")
        display = self._read_from(state, "display")
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

        # The CURRENT set, re-derived from the app's channel (cached by mtime): a long-running server must not answer from
        # a list that was true when it started.
        allowed = catalog.read_verbs()
        if not any(verb.startswith(prefix) for prefix in allowed):
            raise ToolError(f"{verb.split(':')[0]!r} is not a read verb",
                            f"read verbs: {', '.join(allowed)}; use control_write for the rest")
        return {"verb": verb, "answer": self._request(verb, timeout=timeout_s)}

    def control_write(self, *, verb: str, timeout_s: float | None = None) -> dict:
        from . import catalog

        allowed = catalog.write_verbs()
        if not any(verb.startswith(prefix) for prefix in allowed):
            raise ToolError(f"{verb.split(':')[0]!r} is not a write verb",
                            f"write verbs: {', '.join(allowed)}; use control_read for the rest")
        return {"verb": verb, "answer": self._request(verb, timeout=timeout_s)}

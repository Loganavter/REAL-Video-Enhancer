"""The RVE live MCP server: catalog shape and a protocol round-trip, without a display.

Two layers are worth pinning, and both are testable on the portable interpreter:

* the **catalog** — names, descriptions, schemas, annotations, handlers that actually exist on the backend. A catalog
  that drifts from the backend fails at runtime with "no handler", which is the one failure an agent cannot debug from
  the outside;
* the **transport** — `initialize`, `tools/list`, `tools/call` (success and the two error paths: unknown tool, bad
  arguments) over the same in-process subclass BlinkTable's suite uses. The live behaviour (boot a window, screen truth)
  is exercised by `harness/scripts/live.sh` and the session record, not here: a test that needs Xvfb is not a test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT / "harness" / "mcp-server"
APP_DIR = SERVER_DIR / "app"


def _load(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app_modules():
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    catalog = _load("app.catalog", APP_DIR / "catalog.py", "app")
    backend = _load("app.backend", APP_DIR / "backend.py", "app")
    protocol = _load("app.protocol", APP_DIR / "protocol.py", "app")
    return catalog, backend, protocol


def test_every_tool_is_named_and_described_for_an_agent(app_modules):
    catalog, backend, _protocol = app_modules
    names = [tool["name"] for tool in catalog.TOOLS]
    assert len(names) == len(set(names)), "tool names must be unique"
    for tool in catalog.TOOLS:
        name = tool["name"]
        assert name == name.lower() and " " not in name and "-" not in name, (
            f"{name!r} must be snake_case: a client turns it into a tool id")
        assert 80 <= len(tool["description"]) <= 700, f"{name!r}: a description that says what + when NOT"
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"].get("additionalProperties") is False
        assert tool.get("outputSchema"), f"{name!r} needs an outputSchema"
        assert set(tool["annotations"]) == {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
        assert callable(getattr(backend.RveBackend, tool["handler"], None)), (
            f"{name!r} names handler {tool['handler']!r}, which the backend does not implement")


def test_reads_and_writes_never_share_a_tool(app_modules):
    catalog, _backend, _protocol = app_modules
    for tool in catalog.TOOLS:
        if tool["name"] in {"control_read", "control_write"}:
            continue
        read_only = tool["annotations"]["readOnlyHint"]
        assert tool["name"] != "control_read" or read_only
    read_tools = [t["name"] for t in catalog.TOOLS if t["annotations"]["readOnlyHint"]]
    write_tools = [t["name"] for t in catalog.TOOLS if not t["annotations"]["readOnlyHint"]]
    assert read_tools and write_tools
    assert not set(read_tools) & set(write_tools)


class _StubBackend:
    """The smallest surface the transport needs: one read tool, one write tool, one failure."""

    def where(self):
        return {"verb": "where:", "answer": {"kind": "where"}}

    def instance_status(self):
        return {"running": False, "control_dir": "/tmp/none", "display": None, "app_pid": None, "log_tail": ""}

    def click_at(self, *, x, y, button=1):
        from app.backend import ToolError

        raise ToolError("no live instance is running", "call instance_boot first")


def test_the_transport_answers_initialize_and_the_catalog(app_modules):
    _catalog, _backend, protocol = app_modules
    mcp = protocol.MCPProtocol(_StubBackend())
    init = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    assert init["result"]["serverInfo"]["name"] == "rve-live"
    listed = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [tool["name"] for tool in listed["result"]["tools"]]
    assert "where" in names and "click_at" in names
    assert all("handler" not in tool for tool in listed["result"]["tools"]), "handlers stay server-side"


def test_the_transport_runs_a_tool_and_reports_failures_with_a_hint(app_modules):
    _catalog, _backend, protocol = app_modules
    mcp = protocol.MCPProtocol(_StubBackend())

    ok = mcp.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "where", "arguments": {}}})
    assert ok["result"]["isError"] is False
    assert ok["result"]["structuredContent"]["answer"]["kind"] == "where"

    failed = mcp.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                         "params": {"name": "click_at", "arguments": {"x": 1, "y": 2}}})
    assert failed["result"]["isError"] is True
    assert "instance_boot" in failed["result"]["content"][0]["text"]

    unknown = mcp.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                          "params": {"name": "no_such_tool", "arguments": {}}})
    assert unknown["error"]["code"] == protocol.INVALID_PARAMS

    bad_args = mcp.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                           "params": {"name": "click_at", "arguments": {}}})
    assert bad_args["error"]["code"] == protocol.INVALID_PARAMS
    assert "x" in bad_args["error"]["data"]["hint"] or "required" in bad_args["error"]["message"]


def test_a_client_that_still_handshakes_may_send_tracing_meta(app_modules):
    """opencode sends `params._meta` with trace fields and no `protocolVersion` on every call.

    Treating any `_meta` as the modern stateless envelope rejected those calls ("missing the required envelope key(s)")
    while `tools/list` worked - i.e. a connected-looking server whose every tool failed. Measured against the real
    client; the fix is the detection, and this pin keeps it.
    """
    _catalog, _backend, protocol = app_modules
    mcp = protocol.MCPProtocol(_StubBackend())
    init = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    assert "result" in init
    answer = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                         "params": {"name": "where", "arguments": {},
                                    "_meta": {"progressToken": "abc", "traceparent": "00-xyz"}}})
    assert answer.get("result", {}).get("isError") is False, answer


# --- the instance's state files -------------------------------------------------
#
# `instance_boot` starts the app through `live.sh`, which writes pid/display/control/lease into one state tree and a
# pointer to it into the base dir - and the two disagree exactly when it matters (a boot without a lease writes into the
# base tree and leaves an older pointer; a crash leaves the pointer at a tree whose pid is gone). Measured 2026-11-24:
# `instance_status` answered `running: false` over pid 987708, pointed at the stale lease, and `instance_stop` left the
# pid alive. So these tests build the state files themselves (a `sleep` process stands in for the app), which needs no
# display and pins the answers the tools give: which tree, which pid, which control dir, which lease.
#
# Red-before: each test names the record it would not survive if the backend went back to reading only the pointer and
# delegating the stop to `live.sh down`.

import os
import signal
import subprocess
import time


def _sleep_proc() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "60"])


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=15)


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait(timeout=15)
    return proc.pid


def _write_instance(state: Path, pid: int, *, display: str = ":97", control: Path | None = None,
                    lease: str | None = None) -> Path:
    """The files `live.sh up` leaves for one instance."""
    state.mkdir(parents=True, exist_ok=True)
    (state / "app.pid").write_text(f"{pid}\n")
    (state / "display").write_text(f"{display}\n")
    control_dir = control or state / "control"
    (state / "control_dir").write_text(f"{control_dir}\n")
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / "ready").write_text("")
    if lease:
        (state / "lease").write_text(f"{lease}\n")
    (state / "app.log").write_text("booted\n")
    return control_dir


@pytest.fixture
def fake_base(tmp_path, monkeypatch):
    """A base state dir `live.sh` would own, pointed at by the environment the server reads."""
    base = tmp_path / "live"
    base.mkdir()
    monkeypatch.setenv("RVE_LIVE_STATE", str(base))
    monkeypatch.delenv("RVE_CONTROL_DIR", raising=False)
    return base


@pytest.fixture
def lease_releases(app_modules, monkeypatch):
    """The lease ids a stop/boot handed back, without running the real `live-lease`."""
    _catalog, backend, _protocol = app_modules
    calls: list[str] = []
    monkeypatch.setattr(backend.RveBackend, "_release_lease",
                        lambda self, lease_id: calls.append(lease_id) or True)
    return calls


def test_status_reads_the_tree_the_app_runs_in_not_the_stale_pointer(app_modules, fake_base, lease_releases):
    """The 2026-11-24 defect: a stale pointer over a live app answered `running: false`."""
    _catalog, backend, _protocol = app_modules
    stale = fake_base / "old-lease-tree"
    stale.mkdir()
    (fake_base / "state_dir").write_text(f"{stale}\n")
    proc = _sleep_proc()
    try:
        control = _write_instance(fake_base, proc.pid, display=":94", lease="lane-77")
        answer = backend.RveBackend(repo_root=ROOT).instance_status()
        assert answer["running"] is True
        assert answer["app_pid"] == proc.pid
        assert answer["display"] == ":94"
        assert answer["control_dir"] == str(control), "the control dir must be the running tree's, not a guess"
        assert answer["lease"] == "lane-77"
        assert answer["state_dir"] == str(fake_base), "the answer names the tree it was read from"
        assert answer["stale"] is None
    finally:
        _reap(proc)


def test_a_dead_pid_is_not_a_running_instance_and_claims_no_lease(app_modules, fake_base, lease_releases):
    _catalog, backend, _protocol = app_modules
    dead = _dead_pid()
    _write_instance(fake_base, dead, lease="lane-dead")
    answer = backend.RveBackend(repo_root=ROOT).instance_status()
    assert answer["running"] is False
    assert answer["app_pid"] == dead, "the stale record is named, not hidden"
    assert answer["lease"] is None, "a dead pid's lease is a leftover, not an owned display"
    assert answer["display"] is None
    assert str(dead) in answer["stale"]
    assert lease_releases == [], "status does not touch anything"


def test_pid_zero_is_not_a_process(app_modules, fake_base, lease_releases):
    """`kill(0, 0)` probes the caller's own process group: a pid file with 0 must never read as an app (nor be signalled)."""
    _catalog, backend, _protocol = app_modules
    (fake_base / "app.pid").write_text("0\n")
    (fake_base / "lease").write_text("lane-zero\n")
    answer = backend.RveBackend(repo_root=ROOT).instance_status()
    assert answer["running"] is False
    assert str(0) in answer["stale"]


def test_an_unreaped_child_is_not_a_running_instance(app_modules, fake_base, lease_releases):
    """`kill(pid, 0)` succeeds for a zombie; a just-killed app must not read as running."""
    _catalog, backend, _protocol = app_modules
    proc = _sleep_proc()
    proc.send_signal(signal.SIGKILL)
    time.sleep(0.2)  # not reaped on purpose
    try:
        _write_instance(fake_base, proc.pid)
        assert backend.RveBackend(repo_root=ROOT).instance_status()["running"] is False
    finally:
        proc.wait(timeout=15)


def test_stop_kills_the_recorded_pid_only_and_releases_the_lease(app_modules, fake_base, lease_releases):
    _catalog, backend, _protocol = app_modules
    app_proc, bystander = _sleep_proc(), _sleep_proc()
    try:
        control = _write_instance(fake_base, app_proc.pid, display=":95", lease="lane-95")
        answer = backend.RveBackend(repo_root=ROOT).instance_stop()
        app_proc.wait(timeout=15)
        assert app_proc.returncode is not None, "the recorded pid is gone"
        assert bystander.poll() is None, "a pid that is not in the state files is never touched"
        assert answer["stopped_pid"] == app_proc.pid
        assert answer["terminated"] is True
        assert answer["running"] is False
        assert answer["lease"] is None
        assert answer["lease_released"] is True
        assert lease_releases == ["lane-95"]
        assert not (fake_base / "app.pid").exists()
        assert not (fake_base / "lease").exists()
        assert not (control / "ready").exists()
        assert not (fake_base / "state_dir").exists(), "the next boot resolves the base tree again"
    finally:
        _reap(app_proc)
        _reap(bystander)


def test_stop_stops_the_base_tree_instance_when_the_pointer_is_stale(app_modules, fake_base, lease_releases):
    """The same 2026-11-24 shape on the stop path: `live.sh down` reads the pointer and stopped nothing."""
    _catalog, backend, _protocol = app_modules
    stale = fake_base / "old-lease-tree"
    stale.mkdir()
    (fake_base / "state_dir").write_text(f"{stale}\n")
    (stale / "app.pid").write_text("0\n")  # a leftover record in the tree the pointer names
    proc = _sleep_proc()
    try:
        _write_instance(fake_base, proc.pid, display=":92", lease="lane-92")
        answer = backend.RveBackend(repo_root=ROOT).instance_stop()
        proc.wait(timeout=15)
        assert answer["stopped_pid"] == proc.pid
        assert answer["state_dir"] == str(fake_base), "the answer names the tree that was stopped"
        assert lease_releases == ["lane-92"]
    finally:
        _reap(proc)


def test_stop_clears_a_stale_record_and_releases_its_lease(app_modules, fake_base, lease_releases):
    _catalog, backend, _protocol = app_modules
    dead = _dead_pid()
    _write_instance(fake_base, dead, lease="lane-stale")
    answer = backend.RveBackend(repo_root=ROOT).instance_stop()
    assert answer["stopped_pid"] is None, "nothing was killed"
    assert answer["cleared_pid"] == dead
    assert answer["lease_released"] is True
    assert lease_releases == ["lane-stale"]
    assert not (fake_base / "app.pid").exists()
    assert not (fake_base / "lease").exists()


def test_boot_reports_an_already_running_instance_without_running_live_sh(app_modules, fake_base,
                                                                         lease_releases, monkeypatch):
    _catalog, backend, _protocol = app_modules
    proc = _sleep_proc()
    called: list[tuple] = []
    monkeypatch.setattr(backend.RveBackend, "_live_command",
                        lambda self, *args, **kwargs: called.append(args) or None)
    try:
        control = _write_instance(fake_base, proc.pid, display=":96", lease="lane-96")
        answer = backend.RveBackend(repo_root=ROOT).instance_boot()
        assert called == [], "an already running instance is reported, not duplicated"
        assert answer["already_running"] is True
        assert answer["running"] is True
        assert answer["control_dir"] == str(control)
    finally:
        _reap(proc)


def test_boot_reclaims_a_stale_lease_then_reports_the_instance_it_started(app_modules, fake_base,
                                                                         lease_releases, monkeypatch):
    _catalog, backend, _protocol = app_modules
    dead = _dead_pid()
    _write_instance(fake_base, dead, lease="lane-old")
    started = _sleep_proc()

    def fake_up(self, *args, **kwargs):
        assert lease_releases == ["lane-old"], "the stale lease goes back to the pool before the new boot"
        assert not (fake_base / "app.pid").exists(), "the stale pid record is cleared before the new boot"
        _write_instance(fake_base, started.pid, display=":93", lease="lane-new")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend.RveBackend, "_live_command", fake_up)
    try:
        answer = backend.RveBackend(repo_root=ROOT).instance_boot()
        assert answer["already_running"] is False
        assert answer["running"] is True, "the pid the boot wrote is the answer, not a fresh guess"
        assert answer["app_pid"] == started.pid
        assert answer["display"] == ":93"
        assert answer["lease"] == "lane-new"
        reclaimed = answer["reclaimed"]["cleared"][0]
        assert reclaimed["stale_pid"] == dead
        assert reclaimed["stale_lease"] == "lane-old"
    finally:
        _reap(started)


def test_boot_names_the_reason_when_live_sh_returns_without_an_app(app_modules, fake_base, lease_releases,
                                                                  monkeypatch):
    """`live.sh up` exiting 0 with no window must not answer `running: false, already_running: false`."""
    _catalog, backend, _protocol = app_modules
    monkeypatch.setattr(backend.RveBackend, "_live_command", lambda self, *args, **kwargs: None)
    with pytest.raises(backend.ToolError) as caught:
        backend.RveBackend(repo_root=ROOT).instance_boot()
    assert "no running app" in caught.value.message
    assert "log" in (caught.value.hint or "")


def test_actions_refuse_a_stale_instance_instead_of_aiming_at_its_display(app_modules, fake_base, lease_releases):
    _catalog, backend, _protocol = app_modules
    dead = _dead_pid()
    _write_instance(fake_base, dead, display=":97")
    instance = backend.RveBackend(repo_root=ROOT)
    with pytest.raises(backend.ToolError) as typed:
        instance.key_press(key="Return")
    assert "no live instance" in typed.value.message
    with pytest.raises(backend.ToolError) as shot:
        instance.window_screenshot()
    assert "no live instance" in shot.value.message


def test_the_control_dir_falls_back_to_the_state_tree_not_the_checkout(app_modules, fake_base, lease_releases):
    _catalog, backend, _protocol = app_modules
    dead = _dead_pid()
    _write_instance(fake_base, dead)
    (fake_base / "control_dir").unlink()
    assert backend.RveBackend(repo_root=ROOT).control_dir() == fake_base / "control"

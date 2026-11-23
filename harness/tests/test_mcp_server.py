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

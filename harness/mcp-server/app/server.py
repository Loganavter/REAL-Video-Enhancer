#!/usr/bin/env python3
"""Stdio entry point for the RVE live MCP server.

The client launches this as a subprocess and speaks newline-delimited JSON-RPC 2.0 over stdin/stdout (logs go to
stderr). No third-party dependency: the transport is `app/protocol.py` (adopted from BlinkTable's hardened one), the
surface is `app/catalog.py`, the handlers are `app/backend.py`.

    python3 harness/mcp-server/app/server.py            # serve on stdio
    python3 harness/mcp-server/app/server.py --check    # print the catalog and the protocol revision

Wiring (opencode.jsonc)::

    "mcpServers": {
      "rve": {
        "type": "local",
        "command": "python3",
        "args": ["<repo>/harness/mcp-server/app/server.py"],
        "environment": {"RVE_REPO_ROOT": "<repo>"}
      }
    }

Environment: ``RVE_REPO_ROOT`` (repo root), ``RVE_LIVE_STATE`` (instance state dir, default /tmp/opencode/live),
``RVE_CONTROL_DIR`` (the app's control dir), ``RVE_MCP_TIMEOUT`` (seconds per control request).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PACKAGE_PARENT = _HERE.parents[1]  # harness/mcp-server
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from app import catalog  # noqa: E402
from app.backend import RveBackend  # noqa: E402
from app.protocol import MCPProtocol, serve_stdio  # noqa: E402

_DEFAULT_REPO_ROOT = _HERE.parents[3]  # harness/mcp-server/app/server.py -> the repo


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="RVE live MCP server (stdio)")
    parser.add_argument("--check", action="store_true", help="print the catalog and exit")
    parser.add_argument("--repo-root", default=None, help="override the RVE checkout root")
    args = parser.parse_args(argv[1:])

    if args.check:
        print(json.dumps({
            "server": catalog.SERVER_NAME,
            "version": catalog.SERVER_VERSION,
            "protocol": catalog.PROTOCOL_VERSION,
            "tools": [tool["name"] for tool in catalog.TOOLS],
            "resources": len(catalog.RESOURCES),
            "prompts": len(catalog.PROMPTS),
        }, indent=2, ensure_ascii=False))
        return 0

    repo = Path(args.repo_root).resolve() if args.repo_root else _DEFAULT_REPO_ROOT
    backend = RveBackend(repo_root=repo)
    return serve_stdio(MCPProtocol(backend))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

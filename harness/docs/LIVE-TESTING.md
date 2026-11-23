# Live testing without the owner (the RVE control surface)

The offscreen platform cannot answer the questions that matter for painting work: it has **no QRhi**, and `QWidget::grab()`
**re-renders** the tree, so a stale frame or a compositing bug simply disappears in the screenshot. Until this existed, the
owner had to sit at the machine and be the instrument.

`harness/scripts/live.sh` boots the real app on a **private** X display, drives it through its own always-on control
channel and takes real framebuffer screenshots. It is the RVE analogue of BlinkTable's MCP surface: one command per action,
no human in the loop.

```bash
harness/scripts/live.sh up            # lease a free display (:90+), start Xvfb if needed, boot the app, wait for ready
harness/scripts/live.sh shot          # root-window screenshot -> prints the PNG path (the real framebuffer)
harness/scripts/live.sh ask "where:"  # one control-channel request; JSON answer on stdout
harness/scripts/live.sh click 420 300 # a real click through X (hit-testing included)
harness/scripts/live.sh move 420 300
harness/scripts/live.sh key Return
harness/scripts/live.sh down          # stop the app and the Xvfb it started (by PID)
```

## Control-channel verbs (`ask "…"`)

`where:` (which build/toolkit is loaded), `state:` (render/queue state), `probe:` (static control inventory),
`probe_at:X,Y` (what is under a point, rendered colour + widget chain), `grab:[path]` (re-rendered window PNG — not
screen truth), `repaint:` (force a full repaint, the way a window resize would), `click:X,Y`, `drag:…`, `queue:…`,
`select:…`, `move:…`, `job:…`, `audit:…`, and `rhi:[hold_ms]` — the QRhi probe described below.

Answers are JSON files written under the control directory (`/tmp/opencode/.rve-control` by default), so two clients never
eat each other's reply.

## The `rhi:` probe — is a `QRhiWidget` safe in this window?

The preview surface will be the first `QRhiWidget` in RVE's window. `ask "rhi:2500"` creates a plain one inside the
running window, renders a few frames, keeps it on screen for the hold (so `import -window root` can photograph it), writes
a `grabFramebuffer()` readback plus a whole-window tree grab, and answers:

```
{initialized, frames, api, visible, grab_saved, grab_path, grab_size, window_saved, window_path, failures}
```

Measured 2026-11-23 on Xvfb + llvmpipe (this host, `xcb`): `initialized: true`, `frames: 2`, `api: Api.OpenGL`,
readback 320×180 solid `#2f6f4f`. So RHI initialises and presents inside this window on this path; the owner's GPU/Wayland
run is still the final word for the shipped platform.

## Traps this harness encodes (each one cost a session)

1. **Never the owner's display** (`:0`, `:1` or their Wayland socket). `live.sh` leases from `:90` up and skips a display
   that is already in use.
2. **Kill by PID**, from the pid files. `pkill -f live_boot.py` matches the shell running the command and SIGTERMs it —
   observed, mid-experiment, twice in one session.
3. **`xcb` needs software GL here**: `LIBGL_ALWAYS_SOFTWARE=1` (llvmpipe) is what makes QRhi/OpenGL initialise when the
   host has no GPU-backed GL for Xvfb.
4. **Empty the GTK appmenu module and the platform theme** under xcb: the module aborts the process at boot on this host
   (`Settings schema 'org.appmenu.gtk-module' does not contain a key named 'backends-allowed'`).
5. **`grab:` is not screen truth.** It re-renders; `import -window root` (what `shot` does) is the framebuffer. Use the
   first for "what should be there", the second for "what is actually there".
6. The app's control directory is the server's choice, not the repo's: read it from the boot log (`control channel: ready
   in …`) or set `RVE_CONTROL_DIR`.

## The same surface as an MCP server (`harness/mcp-server/app`)

`live.sh` is the shell handle; the MCP server is the typed one, and it is what an agent should call.

```
python3 harness/mcp-server/app/server.py --check   # catalog + protocol revision
python3 harness/mcp-server/app/server.py           # serve MCP over stdio (no third-party deps)
```

17 tools: `instance_boot|stop|status`, `window_screenshot` (root framebuffer, screen truth), `window_grab` (re-render),
`where`, `state`, `control_inventory`, `widget_at`, `rhi_probe`, `click_at`, `drag_from_to`, `key_press`, `type_text`,
`repaint`, `control_read`, `control_write`. Reads never share a tool with writes; `control_*` accept only the app's own
read/write verb sets. The transport is BlinkTable's hardened one (MCP `2026-07-28` plus the legacy handshake).

Wiring (opencode config — the key is `mcp`, not BlinkTable's doc's `mcpServers`; verified against the SDK's
`McpLocalConfig`):

```jsonc
"mcp": {
  "rve": {
    "type": "local",
    "command": ["python3", "<repo>/harness/mcp-server/app/server.py"],
    "cwd": "<repo>",
    "environment": {"RVE_REPO_ROOT": "<repo>"},
    "enabled": true
  }
}
```

Measured against the real client: opencode sends `params._meta` with trace fields and **no** `protocolVersion` on every
call. Treating any `_meta` as the modern stateless envelope rejected every tool with "missing the required envelope
key(s)" while `tools/list` worked — a connected-looking server whose every call failed. The transport now detects the
envelope by `protocolVersion`, and `test_mcp_server.py` pins that case. A running server does not reload code: after an
edit, the session must be reopened (the catalog is cached at start).

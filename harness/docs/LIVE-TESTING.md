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

### Capturing the 13 pages (A/A and regression tables)

`harness/scripts/live.sh ask "all" 600` walks every page stem (`nav0..nav4`, the tab variants) and writes a
`<stamp>/` directory under the control dir with a `.png` and a `.json` per page. The whole walk takes about four
minutes on this host — a client timeout below that gives up while the server is still working, and the files are the
answer. `capture:` prefixed names are **not** a valid spec (the dispatcher passes the raw text through), the bare word
`all` is.

Compare two captures with `python3 harness/scripts/ui_diff.py compare --stock-dumps A --new-dumps B --stock-png A
--new-png B`. Two captures of the same tree are the instrument's A/A: 2026-11-23 it read 0.00 % on all 13 pages,
142/142 widgets inside budget, 409/409 nodes identical — that is the floor the 0.05 % page budget is measured against.

### Preview renders and the LMB compare

`harness/scripts/live.sh ask "preview:/path/clip.mkv" 420` loads the clip, switches to the processing page and presses
**Render Preview** — the one path that sets `RenderOptions.isPreview`, writes the streamable preview file and feeds the
pane (the plain `render:` verb leaves the pane empty). The answer names the preview file, its size, and what the pane
holds: `class`, `visible`, `has_frame`, `frame` and `progress` (rendered, total, playhead, duration).

Three answers this verb learned to give the hard way (2026-11-23):

- **A refusal is named, with the app's own words.** A click that leaves no backend and an enabled button is a refusal,
  not a render that finished in 0.9 s: `error` carries the toast text (`Video is not loaded!`, `Not a valid input!`,
  `End time must be greater than start time!`). Nothing about a refusal reaches the app log.
- **The backend child lives on the ProcessTab** (`processTab.renderProcess`); `window.renderProcess` does not exist,
  and reading it made every real render look refused.
- **The answer is bound to its own request id**, because the verb answers from timers seconds later: without that, a
  `state:` probe sent meanwhile claims the preview's answer and the caller sees `widget not found`.

`state:<objectName>` on a pane also reports `progress` and `strip` (`[x, y, w, h]` of the seek strip), which is what a
live seek is measured with: press the strip with `drag:previewLabel@x1,y1:x2,y2` and watch `playhead` fall behind
`rendered` instead of being dragged back to the edge. On a checkable widget `state:` also reports `checked`, which is how
the preview's transport toggle is read.

The preview **stands paused**: `preview:` starts it holding the first frame it decodes, and it does not advance until
`click:previewPauseButton@20,20` is sent (checked = paused; the glyph is the action a press performs - a play symbol
while paused). A second press holds it again, and a finished render hands over to the file decoder *paused on the frame
the pane already shows* rather than auto-playing. Measured 2026-11-23: at 52/300 produced frames the playhead stood at
0.12 s, after the press it advanced 0.64 -> 2.84 s while `rendered` went 166 -> 230, and after the second press it held
3.48 s through the end of the render.

A `click:` on a widget that is not a painted list used to raise (`_rows` on a toolkit `Button` is a list of content
rows, not the rows widget) and that exception **killed the poll loop** - every later request timed out. The loop now
answers the error and keeps polling; the geometry chain guards both attributes.

With a frame in the pane, the LMB compare is testable: hold the button with `xdotool mousedown 1` (a click is too fast
to capture), take a root screenshot while it is held, then `mouseup 1`. 2026-11-23, on a letterboxed 160x120 clip with
auto border cropping on: the render cropped to 160x64 (pane frame 320x128), and the held-LMB frame was the **source
decoded with the same crop** — no bar rows in either picture, 26.5 % of the pane's pixels differ by more than 8 levels
(source vs the upscale), framing identical.

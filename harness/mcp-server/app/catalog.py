"""Declarative MCP surface for the RVE live server: tools only, no magic.

Why this shape: RVE's app already exposes a full control channel (`src/ui/capture_server.py`: where/state/probe/
probe_at/grab/repaint/rhi/click/drag/queue/select/move/job/audit/render). This catalog turns that channel into *typed
tools* so an agent acts structurally instead of moving a cursor: every tool has a JSON Schema, an output shape and
explicit annotations, and the names say whether a call reads or writes.

Rules kept from the reference implementation (BlinkTable's catalog): `verb_noun` snake_case names, descriptions that say
what + when NOT, flat `inputSchema` with `additionalProperties: false`, `outputSchema` on every tool, all four
annotation hints stated, and reads never share a tool with writes. The generic verb set stays reachable through
`control_read` / `control_write`, restricted to the app's own read and write verbs respectively.
"""

from __future__ import annotations

PROTOCOL_VERSION = "2026-07-28"
MODERN_PROTOCOL_VERSIONS = ("2026-07-28",)
HANDSHAKE_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
SUPPORTED_PROTOCOL_VERSIONS = (*HANDSHAKE_PROTOCOL_VERSIONS, *MODERN_PROTOCOL_VERSIONS)

SERVER_NAME = "rve-live"
SERVER_VERSION = "0.1.0"

#: Freshness hints on cacheable list results (the transport stamps them): a static catalog may be cached for a minute,
#: a live answer may not be cached at all.
STATIC_TTL_MS = 60_000
DYNAMIC_TTL_MS = 0

INSTRUCTIONS = (
    "Drives a local REAL-Video-Enhancer window for verification work. Start it with instance_boot (a private Xvfb "
    "display, never the user's screen), then act with the read tools (where, state, control_inventory, widget_at, "
    "window_screenshot, window_grab, rhi_probe) and the write tools (click_at, drag_from_to, key_press, type_text, "
    "repaint, control_write). window_screenshot is the real framebuffer; window_grab re-renders the tree and hides "
    "stale-frame bugs. Stop the instance with instance_stop when done."
)

#: What each channel verb prefix IS, for the read/write split. The SET of verbs is no longer written here: it is read from
#: the app's own dispatch table (see `channel_verbs`). A hardcoded copy lags `src/ui/capture_server.py`, and then a probe
#: cannot drive a verb the app answers - measured 2026-11-26, `preview:` refused with "is not a write verb" while
#: `hold:`/`release:`/`wheel:` were unreachable, which is what pushed live checks of the preview's gestures into
#: hand-driven `xdotool`. A verb this table does not classify is exposed by BOTH lists (reachable, never silently lost),
#: and `harness/tests/test_control_wheel_verb.py` fails until it is classified here.
VERB_KINDS = {
    "where": "read", "state": "read", "probe": "read", "probe_at": "read", "audit": "read", "grab": "read",
    "tail": "read", "stack": "read",
    "repaint": "write", "click": "write", "drag": "write", "wheel": "write", "hold": "write", "release": "write",
    "queue": "write", "select": "write", "move": "write", "job": "write", "render": "write", "rhi": "write",
    "preview": "write", "input": "write", "scale": "write", "tab": "write", "start": "write", "page": "write",
}

#: The channel source, relative to the checkout.
CHANNEL_SOURCE = ("src", "ui", "capture_server.py")

_verb_cache: dict[str, object] = {"mtime": None, "verbs": ()}


def _repo_root():
    import os
    import pathlib

    declared = os.environ.get("RVE_REPO_ROOT")
    if declared:
        return pathlib.Path(declared)
    return pathlib.Path(__file__).resolve().parents[3]


def channel_verbs() -> tuple[str, ...]:
    """Every verb prefix `capture_server` dispatches, read from its source and cached until that file changes.

    A regex over the module, not an import: the server must not pull Qt into its own process to learn a list of words.
    The cache is keyed on mtime because the alternative - reading once at startup - is exactly how the server ended up
    refusing verbs the app answered, with "reopen the session" as its only remedy. An unreadable source answers the
    empty tuple, so a checkout without the app falls back to `VERB_KINDS` instead of failing at import.
    """
    import re

    path = _repo_root().joinpath(*CHANNEL_SOURCE)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ()
    if _verb_cache["mtime"] == mtime:
        return tuple(_verb_cache["verbs"])
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    verbs = tuple(sorted(set(re.findall(r'wanted\.startswith\("([a-z_]+):"\)', source))))
    _verb_cache.update({"mtime": mtime, "verbs": verbs})
    return verbs


def _verbs(kind: str) -> tuple[str, ...]:
    known = set(channel_verbs()) or set(VERB_KINDS)
    chosen = {verb for verb in known if VERB_KINDS.get(verb, kind) == kind}
    chosen |= {verb for verb in known if verb not in VERB_KINDS}
    return tuple(f"{verb}:" for verb in sorted(chosen))


def read_verbs() -> tuple[str, ...]:
    """The app's READ verbs, as the channel states them now."""
    return _verbs("read")


def write_verbs() -> tuple[str, ...]:
    """The app's WRITE verbs, as the channel states them now."""
    return _verbs("write")


#: Import-time snapshots, kept for callers that only print them. Anything that GATES a request calls the functions.
READ_VERBS = read_verbs()
WRITE_VERBS = write_verbs()

_ANNOTATIONS_READ = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
_ANNOTATIONS_WRITE = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
_ANNOTATIONS_START = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

_OUTPUT_CONTROL = {
    "type": "object",
    "properties": {
        "answer": {"type": "object", "description": "The app's JSON answer to the control request."},
        "verb": {"type": "string"},
    },
    "required": ["answer", "verb"],
    "additionalProperties": True,
}

_OUTPUT_PATH = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
    },
    "required": ["path"],
    "additionalProperties": True,
}

_OUTPUT_STATUS = {
    "type": "object",
    "properties": {
        "running": {"type": "boolean"},
        "display": {"type": ["string", "null"]},
        "control_dir": {"type": "string"},
        "app_pid": {"type": ["integer", "null"]},
        "log_tail": {"type": "string"},
    },
    "required": ["running", "control_dir"],
    "additionalProperties": True,
}


def _xy(name: str, what: str) -> dict:
    return {"type": "integer", "description": f"{what} ({name})"}


TOOLS: list[dict] = [
    {
        "name": "instance_status",
        "handler": "instance_status",
        "title": "Live instance status",
        "description": (
            "Reports whether a private RVE instance is running, which X display it leases, where its control channel "
            "lives and the last lines of its log. Use it first: every other tool acts on that instance, and a call "
            "without one fails with a hint to boot it. Does not start or stop anything."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _OUTPUT_STATUS,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "instance_boot",
        "handler": "instance_boot",
        "title": "Boot a private instance",
        "description": (
            "Starts Xvfb on a free display (:90 and up, never the user's :0) and the real RVE window, then waits for "
            "its control channel. Safe to call twice: an already running instance is reported instead of duplicated. "
            "Do not use it to test packaging or startup timing - it bypasses backend probing on purpose."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "default": 1163, "description": "window width in logical pixels"},
                "height": {"type": "integer", "default": 900, "description": "window height in logical pixels"},
                "timeout_s": {"type": "number", "default": 60, "description": "how long to wait for the control channel"},
            },
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_STATUS,
        "annotations": _ANNOTATIONS_START,
    },
    {
        "name": "instance_stop",
        "handler": "instance_stop",
        "title": "Stop the private instance",
        "description": (
            "Stops the app and (if this server started it) its Xvfb, by PID from the state files - never by pattern, "
            "because a pattern kill matches the shell that runs it. Leaves the app's own files untouched."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _OUTPUT_STATUS,
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "window_screenshot",
        "handler": "window_screenshot",
        "title": "Screen-truth screenshot",
        "description": (
            "Captures the X root window - the real framebuffer - and returns the PNG path. This is the instrument for "
            "'what is actually on screen'. It does NOT re-render the widget tree, which is why it is the only picture "
            "that can show a stale frame or a compositing fault."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "a short name put into the file name"}},
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_PATH,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "window_grab",
        "handler": "window_grab",
        "title": "Window tree grab",
        "description": (
            "Asks the app to render its widget tree into a PNG (the `grab:` verb). Use it for layout and for 'what "
            "should be there'. Do not use it to judge stale pixels: a tree grab re-renders and hides exactly those."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "absolute PNG path; the app defaults to its control dir"}},
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_PATH,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "where",
        "handler": "where",
        "title": "Which build is running",
        "description": (
            "Answers the app's identity: interpreter, loaded toolkit path, toolkit git head and dirty count, whether "
            "the page-fit glue is present. Use it before trusting any measurement - a stale toolkit checkout has "
            "produced believable but false numbers on this project."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "state",
        "handler": "state",
        "title": "Render and queue state",
        "description": (
            "Reports the app's current state (page, render progress, queue) through the `state:` verb. Use it to check "
            "what the window believes before clicking something that depends on it."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "control_inventory",
        "handler": "control_inventory",
        "title": "Static control inventory",
        "description": (
            "Lists the interactive controls the app knows about (the `probe:` verb): names, classes, states. Use it to "
            "find a target before `widget_at` or `click_at`; it does not touch the window."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "widget_at",
        "handler": "widget_at",
        "title": "What is under a point",
        "description": (
            "Reports the rendered colour, the widget chain and any popup at a window point (the `probe_at:` verb). Use "
            "it to prove hit-testing and clipping (a widget 'inside' a pane that paints outside it shows up here). "
            "Coordinates are window coordinates, not screen."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": _xy("x", "window x"),
                "y": _xy("y", "window y"),
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "rhi_probe",
        "handler": "rhi_probe",
        "title": "QRhi in-window probe",
        "description": (
            "Creates a plain QRhiWidget inside the running window, renders, keeps it for `hold_ms` (so a screenshot can "
            "photograph it), writes a readback PNG and a window-tree PNG, then removes it. Use it before building any "
            "GPU-painted widget; the answer separates 'RHI initialises' from 'its pixels are composited'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "hold_ms": {"type": "integer", "default": 0, "description": "keep the probe visible this long"},
                "x": {"type": "integer", "description": "probe left, window coordinates (default 40)"},
                "y": {"type": "integer", "description": "probe top, window coordinates (default 40)"},
            },
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "click_at",
        "handler": "click_at",
        "title": "Click a window point",
        "description": (
            "A real pointer click at window coordinates through X (xdotool), so hit-testing, focus and hover behave as "
            "for a user. Use it for interactions the app's `click:` verb does not cover; prefer `click_at` over typing "
            "coordinates into a shell."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"x": _xy("x", "window x"), "y": _xy("y", "window y"),
                           "button": {"type": "integer", "default": 1, "description": "1 left, 2 middle, 3 right"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                         "additionalProperties": True},
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "drag_from_to",
        "handler": "drag_from_to",
        "title": "Drag between two points",
        "description": (
            "Presses at (x1,y1), moves in steps and releases at (x2,y2) - a real drag, not a synthetic event. Use it "
            "for selection bands, split handles and scrollbars."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"x1": _xy("x1", "start x"), "y1": _xy("y1", "start y"),
                           "x2": _xy("x2", "end x"), "y2": _xy("y2", "end y"),
                           "steps": {"type": "integer", "default": 8, "description": "intermediate moves"}},
            "required": ["x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
        "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                         "additionalProperties": True},
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "key_press",
        "handler": "key_press",
        "title": "Press a key",
        "description": (
            "Sends one key or chord to the focused widget (xdotool key syntax: `Return`, `ctrl+s`, `Escape`). Use it "
            "for keyboard affordances (Space to play, Left/Right to step) rather than inventing a special tool."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "xdotool key name or chord"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                         "additionalProperties": True},
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "type_text",
        "handler": "type_text",
        "title": "Type text",
        "description": (
            "Types a string into the focused widget (xdotool type). Use it to fill a field; a value that only arrives "
            "through a paste path must be tested elsewhere, not here."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "text to type"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                         "additionalProperties": True},
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "repaint",
        "handler": "repaint",
        "title": "Force a full repaint",
        "description": (
            "Asks the app for a full repaint - the same thing a window resize does. Use it to tell 'this region never "
            "repaints' from 'the compositor showed a stale frame': if a repaint heals the region, the paint path is "
            "fine and the frame was stale."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_WRITE,
    },
    {
        "name": "control_read",
        "handler": "control_read",
        "title": "A read verb from the app",
        "description": (
            "Sends one of the app's READ control verbs and returns its JSON answer. Use it for verbs without a "
            "dedicated tool (audit, grab and friends). Writes are refused here on purpose - use control_write."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "verb": {"type": "string", "description": "a read verb from READ_VERBS, with its argument, e.g. 'audit:'"},
                "timeout_s": {"type": "number", "default": 15},
            },
            "required": ["verb"],
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_READ,
    },
    {
        "name": "control_write",
        "handler": "control_write",
        "title": "A write verb from the app",
        "description": (
            "Sends one of the app's WRITE control verbs (queue, select, move, job, render, click, drag, rhi). Use it for "
            "actions without a dedicated tool. Read verbs are refused here; prefer the named tools when one exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "verb": {"type": "string", "description": "a write verb with its argument, e.g. 'queue:clear'"},
                "timeout_s": {"type": "number", "default": 15},
            },
            "required": ["verb"],
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_CONTROL,
        "annotations": _ANNOTATIONS_WRITE,
    },
]

RESOURCES: list[dict] = []
RESOURCE_TEMPLATES: list[dict] = []
PROMPTS: list[dict] = []

"""The control channel is opt-in: an ordinary launch must not create files or walk the pages.

Owner, 2026-11-24: "why does this thing switch itself on - it walks every page". The channel is a supervisor
instrument (`src/ui/capture_server.py`), and it used to be ON when `RVE_UI_CAPTURE_SERVER` was absent; worse, an
EMPTY-but-exported value (a leftover from a copied recipe) was read as "an explicit directory", so the app walked all
13 pages into the repo's `.rve-control` and printed `walk restored to page 2` - the app doing work nobody asked for.

The matrix pinned here is the whole contract, and `mode()` is the single place it is parsed:

| `RVE_UI_CAPTURE_SERVER` | channel | drop box | page walk |
|---|---|---|---|
| unset | off | - | - |
| `""` / spaces | off | - | - |
| `0` / `false` / `no` / `off` | off | - | - |
| `1` (or `true`/`yes`/`on`) | on | `<app root>/.rve-control` | no |
| a directory | on | that directory | yes (a capture run) |

The harness passes the directory explicitly (`harness/scripts/live.sh`, `harness/scripts/ui_live.py`), so opt-in costs
the automation nothing - which is why the switch flips without a fallback.
"""

from __future__ import annotations

import sys
import types

import pytest

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")


@pytest.fixture()
def switch(monkeypatch):
    """Set/clear the variable the way a shell does; `None` means unset."""
    from src.ui import capture_server

    def apply(value):
        if value is None:
            monkeypatch.delenv("RVE_UI_CAPTURE_SERVER", raising=False)
        else:
            monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", value)
        return capture_server.mode()

    return apply


@pytest.mark.parametrize("value", [None, "", "   ", "0", "false", "no", "off", "OFF"])
def test_an_unset_or_off_switch_installs_nothing(switch, value):
    channel = switch(value)
    assert channel.on is False
    assert channel.directory is None
    assert channel.walk is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON"])
def test_a_bare_on_flag_uses_the_default_box_and_does_not_walk(switch, value):
    """`=1` is the diagnosis channel: it must not switch pages under the owner's hands."""
    channel = switch(value)
    assert channel.on is True
    assert channel.directory is None
    assert channel.walk is False


def test_the_default_box_sits_next_to_the_running_script(switch):
    """`<dirname of argv[0]>/.rve-control`: its documented derivation, not the repo path (under pytest argv[0] is pytest)."""
    import os
    import sys

    from src.ui.capture_server import CONTROL_DIR_NAME, control_root

    switch("1")
    expected = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), CONTROL_DIR_NAME)
    assert control_root() == expected


@pytest.mark.parametrize("value", ["/tmp/rve-capture", "capture/out"])
def test_a_directory_is_a_capture_run_and_does_walk(switch, value):
    from src.ui import capture_server

    channel = switch(value)
    assert channel.on is True
    assert channel.directory == value
    assert channel.walk is True
    assert capture_server.control_root() == value
    assert capture_server.enabled() is True


def test_enabled_agrees_with_mode(switch):
    from src.ui import capture_server

    for value in (None, "", "0", "1", "/tmp/x"):
        channel = switch(value)
        assert capture_server.enabled() is channel.on

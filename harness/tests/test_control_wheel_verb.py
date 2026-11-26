"""The `wheel:` verb: the one gesture whose live checks used to be hand-driven `xdotool click 4` at screen coordinates.

Why it exists: the preview's zoom can only be proved by turning the wheel over a specific half of the pane, and neither
`click:` nor `drag:` can express that. A hand-driven run makes every subsequent reader re-derive the coordinates and the
button number, and it cannot say whether the widget CONSUMED the notch - which is exactly the question an "the wheel does
nothing over the original" report asks. The answer carries `accepted` for that reason.

These tests cover the spec parser (the part that has a contract without a window); the gesture itself is proved on a
lease and the catalog sync keeps `control_write` able to drive it.
"""

from __future__ import annotations

import inspect
import re

import pytest

from src.ui import capture_server


def test_a_wheel_names_a_widget_a_point_and_a_direction():
    target, point, delta, error = capture_server.wheel_spec("previewLabel@100,240:-120")

    assert (target, point, delta, error) == ("previewLabel", (100, 240), -120, None)


def test_the_direction_defaults_to_one_notch_up():
    assert capture_server.wheel_spec("previewLabel@10,20")[2] == 120


def test_a_missing_point_is_answered_not_guessed():
    target, point, _delta, error = capture_server.wheel_spec("previewLabel")

    assert target == "previewLabel" and point is None
    assert error and "x,y" in error


def test_a_broken_point_says_so():
    assert "x,y" in capture_server.wheel_spec("previewLabel@nope:nope")[3]


def test_a_zero_delta_is_not_a_notch():
    assert capture_server.wheel_spec("previewLabel@10,20:0")[3] == "a zero delta is not a notch"


def test_a_missing_widget_name_is_an_error_not_a_wildcard():
    assert capture_server.wheel_spec("@10,20")[3]


def test_the_mcp_catalog_can_drive_every_verb_the_channel_dispatches():
    """A verb the app answers but the control surface refuses is invisible to every probe that uses the surface.

    Measured 2026-11-26: `hold:`, `release:` and `preview:` were all answered by the app and refused by `control_write`,
    which is what pushed a live check of the compare's gestures into hand-driven `xdotool`.
    """
    import importlib.util  # noqa: PLC0415 - the sync guard is the test's whole point
    import pathlib

    # `harness/mcp-server/app` is not an importable package (the directory has a hyphen), so load the module by path:
    # the guard has to read the REAL catalog, not a copy of its lists.
    catalog_path = pathlib.Path(__file__).resolve().parents[1] / "mcp-server" / "app" / "catalog.py"
    spec = importlib.util.spec_from_file_location("rve_mcp_catalog", catalog_path)
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)

    source = inspect.getsource(capture_server)
    channel = set(re.findall(r'wanted\.startswith\("([a-z_]+):"\)', source))
    assert channel, "the dispatch table must be readable; a silent empty set would pass this test"
    exposed = {prefix.rstrip(":") for prefix in catalog.read_verbs() + catalog.write_verbs()}

    assert not (channel - exposed), f"the channel answers {sorted(channel - exposed)} but the catalog refuses them"


def test_every_channel_verb_is_classified_as_read_or_write():
    """The sets are derived from the channel, so an unclassified verb would silently live in BOTH lists.

    Reachable is better than refused, but "a write exposed as a read" defeats the split's purpose: the annotation on
    `control_read` says the call changes nothing. This test is the reminder to classify a new verb when it is added.
    """
    import importlib.util
    import pathlib

    catalog_path = pathlib.Path(__file__).resolve().parents[1] / "mcp-server" / "app" / "catalog.py"
    spec = importlib.util.spec_from_file_location("rve_mcp_catalog_kinds", catalog_path)
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)

    unclassified = [verb for verb in catalog.channel_verbs() if verb not in catalog.VERB_KINDS]

    assert not unclassified, (
        f"{unclassified} are answered by the channel and not classified in VERB_KINDS; they would appear in both the "
        f"read and the write list")

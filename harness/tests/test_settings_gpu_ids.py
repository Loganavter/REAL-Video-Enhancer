"""The backend's GPU report is a COUNT, and each count gets an honest control.

Owner report, 2026-11-26: the "NCNN GPU ID" and "PyTorch GPU ID" steppers looked dead while "Upscale Scale" right
below worked. On a one-GPU machine (`PyTorch GPU 0: NVIDIA GeForce RTX 4060 Ti`) the report read 0 - the counter
started at -1 and clamped - so the field's range was `0..0`: nothing to step. NCNN prints no device line (the
backend was pytorch) and `max(0, -1)` was ALSO 0, so "no GPU of this backend" and "the only GPU" were the same
number.

The contract this file pins:

* `parse_backend_report` counts DEVICE LINES from 0: 0, 1 and 3 `PyTorch GPU n` lines are 0, 1 and 3 devices;
* the Render Settings page turns each count into the honest control - `0` disables the whole row and says which
  backend reported nothing, `1` disables the stepper at the only id, `2+` gives it the live range `0 .. count - 1`;
* the glue hands COUNTS to the page and never a maximum to a widget, so the page's own rows stay the page's.

Runs offscreen only. The two rows on a real one-GPU window (disabled with a readable reason, not an inert stepper)
are `LIVE-PROOF WANTED` in `DONE.md`.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.tab_3 import Tab3Page  # noqa: E402
from src.ui.use_cases import startup  # noqa: E402

#: The two rows as the page builds them: (label attribute, stepper attribute, the backend name the reason names).
ROWS = (
    ("label_45", "ncnn_gpu_id", "NCNN"),
    ("label_43", "pytorch_gpu_id", "PyTorch"),
)


# --- 1. the report is a count ----------------------------------------------------------------------


@pytest.mark.parametrize("devices", [0, 1, 3])
def test_the_report_counts_device_lines_from_zero(devices):
    """One line per device: N `PyTorch GPU n` lines are N devices, never N-1 (one line used to read as zero)."""
    text = "".join(f"PyTorch GPU {index}: GPU {index}\n" for index in range(devices))
    report = startup.parse_backend_report(text)
    assert report.pytorch_gpu_count == devices
    assert report.ncnn_gpu_count == 0
    assert report.half_precision is False


def test_a_device_named_line_counts_once():
    """The probe prints `PyTorch GPU 0: <name>`; the name is not a second line and the id is not a count."""
    report = startup.parse_backend_report(
        "PyTorch GPU 0: NVIDIA GeForce RTX 4060 Ti\n"
        "PyTorch GPU 1: NVIDIA GeForce RTX 3060\n"
        "Half precision support: True\n"
    )
    assert report.pytorch_gpu_count == 2
    assert report.half_precision is True


# --- 2. a count of zero disables the row, and says why ---------------------------------------------


@pytest.fixture()
def page(qapp):
    """A real Render Settings tab, shown at factor 1.0 offscreen (`test_general_settings_info.py`'s recipe)."""
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(1.0)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    tab = Tab3Page()
    root = tab.build()
    root.resize(900, 700)
    root.show()
    for _ in range(8):
        qapp.processEvents()
    tab._test_root = root  # keep the root alive for the duration of the test
    return tab


def test_no_devices_disables_the_whole_row_and_names_the_backend(page):
    page.apply_gpu_counts(ncnn_gpu_count=0, pytorch_gpu_count=0)
    for label_name, stepper_name, backend in ROWS:
        label = getattr(page, label_name)
        stepper = getattr(page, stepper_name)
        assert not label.isEnabled(), f"{backend}: the label of an empty row must read as unavailable"
        assert not stepper.isEnabled(), f"{backend}: no device means no stepper to move"
        assert stepper.maximum() == 0
        assert "reports no GPU" in stepper.toolTip()
        assert backend in stepper.toolTip(), f"the reason must name {backend}"


def test_one_device_forces_the_only_id_and_disables_the_stepper(page):
    page.apply_gpu_counts(ncnn_gpu_count=1, pytorch_gpu_count=1)
    for label_name, stepper_name, backend in ROWS:
        label = getattr(page, label_name)
        stepper = getattr(page, stepper_name)
        assert label.isEnabled(), f"{backend}: one device is a real choice of row, just not of id"
        assert not stepper.isEnabled(), f"{backend}: there is nothing to choose with one device"
        assert stepper.maximum() == 0
        assert stepper.value() == 0
        assert stepper.text() == "0", "a fixed value, not a dead arrow"
        assert "one GPU" in stepper.toolTip()
        assert backend in stepper.toolTip()


def test_two_or_more_devices_step_over_zero_to_count_minus_one(page):
    page.apply_gpu_counts(ncnn_gpu_count=3, pytorch_gpu_count=2)
    expected = (("label_45", "ncnn_gpu_id", "NCNN", 2), ("label_43", "pytorch_gpu_id", "PyTorch", 1))
    for label_name, stepper_name, backend, maximum in expected:
        label = getattr(page, label_name)
        stepper = getattr(page, stepper_name)
        assert label.isEnabled()
        assert stepper.isEnabled(), f"{backend}: {maximum + 1} devices are a live range 0..{maximum}"
        assert stepper.maximum() == maximum
        assert stepper.value() == 0
        stepper.setValue(maximum)
        assert stepper.value() == maximum
        assert stepper.text() == str(maximum)


# --- 3. the stored value stays readable, and a stale one cannot survive ------------------------------


def test_a_stale_stored_id_is_clamped_not_shown_on_a_disabled_stepper(page):
    """The restore path (`SettingsTab.connectSettingText`) calls `setValue` on whatever the file held."""
    page.apply_gpu_counts(ncnn_gpu_count=1, pytorch_gpu_count=1)
    page.pytorch_gpu_id.setValue(4)
    assert page.pytorch_gpu_id.value() == 0
    assert page.pytorch_gpu_id.text() == "0"
    assert isinstance(page.pytorch_gpu_id.text(), str), "the setting key keeps its string/int shape"


def test_re_applying_a_count_keeps_its_statement(page):
    """A settings reset re-reads the values but not the probe; the range must not drift back to the default 0..100."""
    page.apply_gpu_counts(ncnn_gpu_count=4, pytorch_gpu_count=1)
    assert page.ncnn_gpu_id.maximum() == 3
    assert page.pytorch_gpu_id.maximum() == 0
    assert not page.pytorch_gpu_id.isEnabled()


# --- 4. the glue passes counts, the page owns the widgets -------------------------------------------


def test_the_glue_hands_the_page_counts_and_never_a_maximum_to_a_stepper():
    source = Path(__file__).resolve().parents[2] / "src" / "ui" / "SettingsTab.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    called = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_gpu_counts"
    ]
    assert called, "SettingsTab no longer states the GPU rows' controls from the probe's counts"
    assert {keyword.arg for keyword in called[0].keywords} == {"ncnn_gpu_count", "pytorch_gpu_count"}
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setMaximum"
    ]
    assert not offenders, (
        "SettingsTab sets a widget's maximum again (line(s) "
        + ", ".join(str(line) for line in offenders)
        + "): the Render Settings page owns its rows"
    )

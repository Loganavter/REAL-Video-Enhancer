"""The render-failure sentence: a popup has to say WHAT happened, not "check the logs".

Written from the owner's own failure (2026-11-27): the only thing on screen was
`Rendering failed! Please check the logs tab!`, while the fact - `earlyoom` sent SIGTERM to the render's ffmpeg and
then to the backend, because the machine was out of memory - was in `journalctl` and in `log.txt`, where a user does
not look for a diagnosis.

Every row here pins a sentence a user reads: the signal's NAME and its usual cause, the backend's own last error line,
and the multi-job case (a queue pass can fail in more than one place).
"""

from __future__ import annotations

import pytest

from src.ui.render_failure import describe_exit, describe_failure, failed_codes


def test_a_signal_death_is_named_and_explained_not_printed_as_a_number():
    text = describe_exit(-15)
    assert "SIGTERM" in text and "-15" not in text.replace("SIGTERM", ""), text
    assert "earlyoom" in text or "memory killer" in text, (
        "the owner's own failure was the memory killer, and the sentence has to say so")


def test_the_other_signals_the_app_has_met_say_what_they_mean():
    assert "SIGKILL" in describe_exit(-9) and "memory" in describe_exit(-9)
    assert "SIGSEGV" in describe_exit(-11) and "crashed" in describe_exit(-11)
    assert "signal 7" in describe_exit(-7), "an unnamed signal still prints its number, never a bare code"


def test_a_plain_exit_code_carries_the_backends_own_words():
    text = describe_exit(1, "FFmpeg failed to render the video.")
    assert text.startswith("the backend exited with code 1"), text
    assert "FFmpeg failed to render the video." in text


def test_the_detail_is_one_line_and_bounded_so_a_popup_stays_a_popup():
    text = describe_exit(1, "line one\n   line two\t" + "x" * 400)
    assert "\n" not in text and "\t" not in text
    assert text.endswith("...") and len(text) < 260


def test_success_codes_never_produce_a_sentence():
    assert failed_codes([0, 3221225477]) == [], "3221225477 is the Windows ncnn quirk, not a failure"
    assert describe_failure([0]) == ""
    assert describe_failure([0, 3221225477]) == ""
    assert describe_failure([]) == ""


def test_one_failed_job_of_one_reads_as_its_own_reason():
    assert describe_failure([-15]) == describe_exit(-15)
    assert describe_failure([1], reasons=["RENDER FAILED: the engine could not load"]) == (
        "RENDER FAILED: the engine could not load"), (
        "the backend's own summary beats a code, because it says what broke rather than how the process ended")


def test_a_queue_pass_that_fails_more_than_once_says_how_many_and_which_codes():
    text = describe_failure([0, -15], reasons=["the render process was killed by SIGTERM (earlyoom)"])
    assert text.startswith("1 of 2 render processes failed"), text
    assert "codes: -15" in text
    assert "SIGTERM" in text


def test_the_message_popup_grows_to_its_message_instead_of_clipping_it():
    """The second half of the owner's report: a real sentence was CUT OFF inside the dialog.

    `RegularQTPopup` was `setFixedSize(400, 100)`, so anything longer than two short lines was invisible - which is why
    the failure text had been shortened to "check the logs tab" in the first place. The window now sizes to its text,
    bounded in width so a traceback cannot cover the screen.
    """
    from PySide6.QtWidgets import QApplication

    from src.ui.dialogs.toolkit_dialogs import RegularQTPopup

    QApplication.instance() or QApplication([])
    short = RegularQTPopup("ok", blocking=False)
    # The exact sentence the live proof showed clipped ("...): [out#0/rawvideo @" with the rest invisible), and it has
    # to survive the TYPOGRAPHY step too: `showEvent` applies the app's sizes after the dialog was built, which made
    # text that fitted at construction time three lines tall later.
    long = RegularQTPopup(
        "Rendering failed: " + describe_exit(
            -15, "FFmpeg failed to render the video. [out#0/rawvideo @ 0x55c430f1bfc0] Error muxing a packet"),
        blocking=False)
    long.show()
    QApplication.processEvents()

    assert short.width() == 400, "a short message keeps the old width"
    assert 400 <= long.width() <= RegularQTPopup.MAX_WIDTH
    assert long.height() > short.height(), "a longer message needs more room, or its tail is invisible"
    assert long.label.height() >= long.label.heightForWidth(long.label.width()) - 1, (
        f"the label must have room for every wrapped line it was given: "
        f"{long.label.height()} < {long.label.heightForWidth(long.label.width())}")
    assert long.label.height() > 30, "three wrapped lines are taller than one"

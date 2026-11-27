"""What a render's exit code MEANS, in words a person can act on.

The owner's report (2026-11-27): a render failed and the only thing on screen was

    Rendering failed! Please check the logs tab!

while the actual fact was in the journal, not in the app: `earlyoom` had sent SIGTERM to the render's ffmpeg (3.1 GB
RSS) and then to the backend, because the machine was out of memory. `-15` is not a number a user should have to
recognise, and "check the logs" is not a diagnosis.

This module is one pure function plus a table: no Qt, no tab state, so the sentence can be tested on every code the
app can see (0, 1, 3221225477 - the Windows ncnn quirk - and a signal death, which arrives NEGATIVE because
`subprocess.wait()` reports `-signum`).

Where the text goes: the queue row's hover line (`render_jobs.state_text` reads `state.error`), the completion popup,
and the log. One string, three readers, no second copy.
"""

from __future__ import annotations

__all__ = ["SIGNAL_NAMES", "describe_exit", "describe_failure", "failed_codes"]

#: Exit codes the app treats as SUCCESS, whatever else happened in the log.
#: 3221225477 (0xC0000005) appears when using ncnn on Windows - the render is fine, the process's exit status is not.
OK_CODES = (0, 3221225477)

#: The signals a render is realistically killed by, named so the message does not print a bare number.
SIGNAL_NAMES = {
    1: "SIGHUP",
    2: "SIGINT",
    6: "SIGABRT",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    15: "SIGTERM",
}

#: What each of those usually MEANS here. Only the ones the app has actually seen get a sentence: a table of guesses
#: for every signal would read as knowledge the app does not have.
_SIGNAL_WHY = {
    9: "a forced stop or the system's out-of-memory killer",
    11: "the backend crashed",
    15: "outside the app - on this machine usually the system's memory killer (earlyoom)",
}


def failed_codes(codes) -> list[int]:
    """The exit codes that mean a FAILED job, in the order they arrived."""
    return [int(code) for code in codes if int(code) not in OK_CODES]


def describe_exit(code: int, detail: str = "") -> str:
    """One sentence for one exit code, with the backend's own last words appended when there are any.

    `detail` is the backend line that names the failure (`RENDER FAILED: ...`, a traceback's last line, ffmpeg's error);
    it is trimmed to one line so a popup stays a popup. Never returns an empty string.
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = -1
    text = ""
    if code < 0:
        signal = -code
        name = SIGNAL_NAMES.get(signal, f"signal {signal}")
        text = f"the render process was killed by {name}"
        why = _SIGNAL_WHY.get(signal)
        if why:
            text += f" ({why})"
    else:
        text = f"the backend exited with code {code}"
    line = " ".join(str(detail or "").split())
    if line:
        if len(line) > 180:
            line = line[:177] + "..."
        text += f": {line}"
    return text


def describe_failure(codes, reasons=(), detail: str = "") -> str:
    """The sentence a popup and the queue row show for a finished run.

    `codes` is every job's exit code from this pass, `reasons` the per-job messages the reader already built (the
    backend's own `RENDER FAILED` line wins over a code, because it says what broke rather than how the process ended),
    `detail` the last backend error line for the job that has no reason of its own.
    """
    failed = failed_codes(codes)
    if not failed:
        return ""
    total = len(list(codes))
    first_reason = next((str(r).strip() for r in reasons if str(r or "").strip()), "")
    if first_reason:
        text = first_reason
    else:
        text = describe_exit(failed[0], detail)
    if len(failed) > 1 or total > 1:
        distinct = sorted(set(failed))
        text = f"{len(failed)} of {total} render processes failed (codes: " \
               f"{', '.join(str(c) for c in distinct)}); first: {text}"
    return text

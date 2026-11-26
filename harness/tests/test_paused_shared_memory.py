"""The paused-state shared flag: no leak warnings on exit, no stale segment in /dev/shm.

The owner's quit printed `1 leaked shared_memory objects to clean up` and then `No such file or directory:
'//paused_state<pid>'` (2026-11-22). The cause is CPython's `SharedMemory.unlink()`: it raises on a segment that is
already gone BEFORE it unregisters the name, so the resource tracker keeps reporting it. The flag now owns its lifetime
(`src/ui/shared_flag.py`, `track=False`).

These are subprocess tests: the warnings are printed by the tracker at interpreter shutdown, so they only exist in a
process that has exited.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Runs a create/release sequence in a child, prints the segment's presence, and exits - the tracker speaks at exit.
_SCRIPT = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {root!r})
    from multiprocessing import shared_memory

    mode = {mode!r}
    name = "/rve-flag-test-" + str(os.getpid())
    path = "/dev/shm/" + name.lstrip("/")

    if mode == "shipped":
        from src.ui.shared_flag import create, release
        flag = create(name)
        created = os.path.exists(path)
        os.unlink(path)          # the second owner's cleanup: the name is gone before ours runs
        release(flag)
    else:
        flag = shared_memory.SharedMemory(name=name, create=True, size=1)
        created = os.path.exists(path)
        os.unlink(path)
        try:
            flag.unlink()
        except FileNotFoundError:
            pass

    print("created:", created)
    print("left behind:", os.path.exists(path))
    """
)


def _run(mode: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(root=ROOT, mode=mode)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_the_shipped_lifecycle_releases_the_flag_without_a_tracker_warning():
    result = _run("shipped")
    assert result.returncode == 0, result.stderr
    assert "created: True" in result.stdout
    assert "left behind: False" in result.stdout
    assert "leaked shared_memory" not in result.stderr
    assert "No such file or directory" not in result.stderr


def test_an_attach_to_an_existing_flag_reuses_it_and_releases_it():
    name = "/rve-flag-attach-" + str(os.getpid())
    path = "/dev/shm/" + name.lstrip("/")
    from multiprocessing import shared_memory

    from src.ui.shared_flag import create, release

    first = create(name)
    assert os.path.exists(path)
    second = create(name)
    assert first.buf[0] == second.buf[0] == 0, "the second call attaches, it does not wipe the flag"
    release(second)
    assert not os.path.exists(path), "releasing detaches and removes the name"
    release(first), "releasing again must not raise"
    assert not os.path.exists(path)


def test_the_old_shape_is_what_printed_the_warning():
    # A discriminating control: the same sequence with the default `track=True` DOES warn. Without this, the test above
    # would pass even if the tracker had simply stopped complaining.
    result = _run("old")
    assert "created: True" in result.stdout
    assert "leaked shared_memory" in result.stderr


def test_releasing_a_block_whose_segment_is_already_gone_does_not_talk_to_the_tracker(monkeypatch):
    """Owner, 2026-11-26: the terminal printed `KeyError: '//status_state<pid>'` from the resource tracker at exit.

    Every block here is created with `track=False` on a modern interpreter - it was never registered - and `release`'s
    "already gone" path unregistered it anyway. The tracker answers that with a KeyError in its own process, which is the
    traceback nobody could place.
    """
    import multiprocessing.resource_tracker as tracker

    from src.ui import shared_flag

    calls = []
    monkeypatch.setattr(tracker, "unregister", lambda name, rtype: calls.append((name, rtype)))

    block = shared_flag.create("/test_release_untracked", 8)
    try:
        # Unlink behind the flag's back: this is the FileNotFoundError path, the one that used to call the tracker.
        import os
        from multiprocessing import shared_memory

        name = block._name
        # `track=False` here too: this helper is not what the test is about, and a tracked attach would register (and
        # then unregister) the name itself, filling `calls` with the test's own noise.
        shared_memory.SharedMemory(name=name, track=False).unlink()
        shared_flag.release(block)
    finally:
        pass

    assert calls == [], "a block this process never registered must not be unregistered"

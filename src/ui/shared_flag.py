"""The one-byte shared flag the backend polls (the paused state), with an explicit lifetime.

The flag is a POSIX shared-memory segment named after this process (`/paused_state<pid>`), so nothing else can own it
and nothing else can clean it up. Two things went wrong with the plain `SharedMemory(..., create=True)` version (owner,
2026-11-22, on quit):

* the segment was registered with `multiprocessing`'s resource tracker, and CPython's `SharedMemory.unlink()` raises
  BEFORE it unregisters - so whenever the name was already gone (a previous run's cleanup, a manual `rm`, a second
  unlink) the registration stayed and the tracker printed "There appear to be 1 leaked shared_memory objects to clean
  up" at exit, followed by "No such file or directory";
* an exit during a render never reached the cleanup at all, leaving the segment in `/dev/shm`.

`track=False` makes this module the single owner: create here, release here, and the tracker has nothing to say. Kept
Qt-free on purpose - the harness interpreter can exercise the lifecycle without PySide6.
"""

from __future__ import annotations

from multiprocessing import shared_memory


#: Objects to release at interpreter exit. A window can be quit without its close handler ever running (`app.quit()`
#: from the shutdown path, a crash, a kill during a render), and the segments are per-pid, so nothing else would ever
#: find them - `/dev/shm` kept both the paused flag and the preview block after a real quit (measured 2026-11-22).
_CLEANUPS: list = []


def _run_cleanups() -> None:
    for getter in list(_CLEANUPS):
        try:
            release(getter())
        except Exception:
            pass


def register(getter) -> None:
    """Release whatever ``getter()`` returns when the interpreter exits (idempotent per getter)."""
    if any(existing is getter for existing in _CLEANUPS):
        return
    _CLEANUPS.append(getter)
    if len(_CLEANUPS) == 1:
        import atexit

        atexit.register(_run_cleanups)


def _open(name: str, create: bool, size: int = 1):
    """`SharedMemory` without the resource tracker where the interpreter supports it.

    `track=` arrived in 3.13; the front end runs 3.13 while the bundled backend interpreter is 3.12, and both import
    this module, so the fallback keeps the old signature working there (`release` then drops the registration itself).
    """
    kwargs = {"create": True, "size": int(size)} if create else {}
    try:
        return shared_memory.SharedMemory(name=name, track=False, **kwargs)
    except TypeError:  # Python < 3.13: no `track` parameter
        return shared_memory.SharedMemory(name=name, **kwargs)


def create(name: str, size: int = 1):
    """Create the segment, or attach to the one a previous run left behind (the name carries the pid)."""
    try:
        return _open(name, create=True, size=size)
    except FileExistsError:
        return _open(name, create=False)


def attach(name: str):
    """Attach to an existing segment without creating it (used to drop a stale one)."""
    return _open(name, create=False)


#: The preview block is created by its own widget, so it keeps the older spelling.
create_many = create


def release(flag) -> None:
    """Close and unlink the flag; safe twice, safe when it is already gone, never raises."""
    if flag is None:
        return
    try:
        flag.close()
    except Exception:
        pass
    try:
        flag.unlink()
        return
    except FileNotFoundError:
        pass  # the flag is released either way
    except Exception:
        pass
    # The name was already gone. On an interpreter without `track=` the creation registered it, and CPython's `unlink`
    # raises BEFORE it unregisters - drop the registration here so the tracker does not report a leak at exit.
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(getattr(flag, "_name", ""), "shared_memory")
    except Exception:
        pass

"""Shared-memory segments this process does not own, attached without the resource tracker.

Every segment the app uses is named after the **GUI** process (`/paused_state<pid>`, `/image_preview<pid>`), and the
backend only ever *attaches* to them. `multiprocessing.shared_memory.SharedMemory(name=...)` registers any segment it
touches with this process's `resource_tracker`, which then unlinks it at interpreter exit - so when the GUI (the owner)
has already unlinked it, or the backend is killed mid-render, the tracker prints

    Traceback (most recent call last):
      File ".../multiprocessing/resource_tracker.py", line 457, in main
        cache[rtype].remove(name)
    KeyError: '//paused_state<pid>'

and, worse than the noise, a surviving backend would unlink the GUI's segments on its way out. Attach through this
module instead: `track=False` on 3.13+, and an explicit `unregister` on the 3.12 backend interpreter (which has no
`track=` parameter). The owner (`src/ui/shared_flag.py`) does the same from the other side.
"""

from __future__ import annotations

from multiprocessing import resource_tracker, shared_memory


def _untrack(segment) -> None:
    """Drop this process's registration; a segment we attached to is not ours to unlink."""
    try:
        resource_tracker.unregister(getattr(segment, "_name", ""), "shared_memory")
    except Exception:
        pass


def attach(name: str):
    """Attach to an existing segment owned by another process, never registering it for cleanup here."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:                      # Python < 3.13: no `track` parameter
        segment = shared_memory.SharedMemory(name=name)
        _untrack(segment)
        return segment


def create(name: str, size: int = 1):
    """Create or attach (`create=True` semantics), still without taking ownership of someone else's name."""
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=int(size), track=False)
    except FileExistsError:
        return attach(name)
    except TypeError:                      # Python < 3.13
        try:
            segment = shared_memory.SharedMemory(name=name, create=True, size=int(size))
        except FileExistsError:
            return attach(name)
        _untrack(segment)
        return segment

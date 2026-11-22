"""Dump every thread's Python stack on SIGUSR1 - the only way to sample a hung window without ptrace.

The GUI thread froze after a render (owner, 2026-11-22) and there was no way to look inside it: `py-spy`/`gdb` need ptrace,
and this host runs `kernel.yama.ptrace_scope=1`, so nothing but a parent (root) may attach. A signal handler is allowed
from the same user, and `faulthandler` writes the stacks from *inside* the process.

    kill -USR1 <pid>          # dumps to <repo>/hang-stack-<stamp>.txt and to the log
    cat hang-stack-*.txt

Always installed: registering a handler costs nothing until the signal arrives, and a hang is exactly when a build has no
other instrument. The file goes next to the app's `log.txt` so it is found the same way.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import time

INSTALLED_ATTR = "_fault_stack_installed"


def stack_path(directory: str | None = None) -> str:
    directory = directory or os.getcwd()
    return os.path.join(directory, f"hang-stack-{time.strftime('%Y%m%d-%H%M%S')}.txt")


def install(directory: str | None = None) -> str | None:
    """Register the SIGUSR1 dump; returns the file it will write (``None`` when the signal is unavailable)."""
    if getattr(sys, INSTALLED_ATTR, False):
        return None
    if not hasattr(signal, "SIGUSR1"):
        return None
    path = stack_path(directory)
    try:
        handle = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - kept open by faulthandler
    except OSError:
        return None
    faulthandler.register(signal.SIGUSR1, file=handle, all_threads=True, chain=False)
    setattr(sys, INSTALLED_ATTR, True)
    print(f"[stack] kill -USR1 {os.getpid()} dumps every thread's Python stack to {path}", flush=True)
    return path

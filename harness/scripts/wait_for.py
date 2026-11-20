"""Wait for an explicit condition, print one verdict line, exit. Then the harness notifies you.

Why this exists: a supervisor turn that ends in `sleep 300` burns a whole model turn doing nothing,
and a turn that re-runs `ls` every minute does the same with less patience. The two things a wait
actually needs are (a) a condition stated once, (b) a verdict that travels with the notification - so
the next turn starts from "done, here is the table" instead of "let me look again".

The harness notifies on a BACKGROUND COMMAND's completion, so the intended use is:

    # launch the job and, separately, launch the waiter - both as background commands
    harness/scripts/wait_for.py --pid 175560 --file /tmp/opencode/b3/c32_leaky.plan \\
        --timeout 3600 --tail 20 --log /tmp/opencode/b3.log

Exit codes: 0 condition met, 2 timeout (argparse usage errors also exit 2), 4 the watched
process died without the condition, 130 interrupted.

Conditions (all by default, `--any` for any):
  --file PATH          PATH exists
  --gone PATH          PATH does not exist
  --pid N              process N exists
  --pid-gone N         process N is gone
  --log-match TSV      "PATH::REGEX" appears in PATH (repeatable)
  --log-idle "PATH::N" PATH has not changed for N seconds (repeatable)
  --gpu-free           /var/tmp/opencode/gpu.lock is not held (the one-GPU-job rule)
  --min-bytes "PATH::N" PATH is at least N bytes (a build that writes its output at the end)

Examples:
  # a TensorRT build: wait for the plan, keep the log tail in the verdict
  harness/scripts/wait_for.py --file /tmp/opencode/b3/c32_leaky.plan --log /tmp/opencode/b3.log --tail 15

  # a render with a known end marker
  harness/scripts/wait_for.py --log-match "/tmp/opencode/render.log::^DONE " --timeout 7200

  # is the card free right now?
  harness/scripts/wait_for.py --gpu-free --timeout 5
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
import time
from dataclasses import dataclass, field

GPU_LOCK = "/var/tmp/opencode/gpu.lock"


@dataclass
class Check:
    name: str
    test: object                      # callable() -> (bool, str)
    detail: str = ""
    met: bool = field(default=False, init=False)


def _exists(path: str):
    def t() -> tuple[bool, str]:
        try:
            st = os.stat(path)
            return True, f"{st.st_size} B"
        except OSError:
            return False, "absent"
    return t


def _gone(path: str):
    def t() -> tuple[bool, str]:
        return (not os.path.exists(path)), ("absent" if not os.path.exists(path) else "present")
    return t


def _pid_alive(pid: int):
    def t() -> tuple[bool, str]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False, "gone"
        except PermissionError:
            return True, "alive (other uid)"
        return True, "alive"
    return t


def _pid_gone(pid: int):
    def t() -> tuple[bool, str]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True, "gone"
        return False, "alive"
    return t


def _log_match(path: str, pattern: str):
    rx = re.compile(pattern, re.M)

    def t() -> tuple[bool, str]:
        try:
            with open(path, "r", errors="replace") as fh:
                if rx.search(fh.read()):
                    return True, "matched"
            return False, "no match yet"
        except OSError:
            return False, "log missing"
    return t


def _log_idle(path: str, seconds: float):
    def t() -> tuple[bool, str]:
        try:
            age = time.time() - os.stat(path).st_mtime
        except OSError:
            return False, "log missing"
        return age >= seconds, f"idle {age:.0f}s"
    return t


def _min_bytes(path: str, n: int):
    def t() -> tuple[bool, str]:
        try:
            size = os.stat(path).st_size
        except OSError:
            return False, "absent"
        return size >= n, f"{size} B"
    return t


def _gpu_free():
    def t() -> tuple[bool, str]:
        if not os.path.exists(GPU_LOCK):
            return True, "no lock file"
        with open(GPU_LOCK, "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False, "held"
            else:
                fcntl.flock(fh, fcntl.LOCK_UN)
                return True, "free"
    return t


def parse_tsv(value: str) -> tuple[str, str]:
    if "::" not in value:
        raise argparse.ArgumentTypeError(f"expected PATH::VALUE, got {value!r}")
    path, rest = value.rsplit("::", 1)
    return path, rest


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[], help="path that must exist")
    ap.add_argument("--gone", action="append", default=[], help="path that must be gone")
    ap.add_argument("--pid", type=int, action="append", default=[], help="process that must be alive")
    ap.add_argument("--pid-gone", type=int, action="append", default=[],
                    help="process that must be gone")
    ap.add_argument("--log-match", action="append", default=[], type=parse_tsv,
                    help="PATH::REGEX must appear")
    ap.add_argument("--log-idle", action="append", default=[], type=parse_tsv,
                    help="PATH::SECONDS unchanged")
    ap.add_argument("--min-bytes", action="append", default=[], type=parse_tsv,
                    help="PATH::BYTES reached")
    ap.add_argument("--gpu-free", action="store_true", help=f"{GPU_LOCK} not held")
    ap.add_argument("--any", action="store_true", help="succeed when ANY condition holds")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds (default 3600)")
    ap.add_argument("--interval", type=float, default=2.0, help="poll seconds (default 2)")
    ap.add_argument("--report", type=float, default=60.0, help="progress line seconds (default 60)")
    ap.add_argument("--log", default="", help="file whose tail goes into the verdict")
    ap.add_argument("--tail", type=int, default=10, help="lines of --log in the verdict")
    ap.add_argument("--label", default="", help="what is being waited for, for the verdict line")
    args = ap.parse_args()

    checks: list[Check] = []
    for p in args.file:
        checks.append(Check(f"file:{p}", _exists(p)))
    for p in args.gone:
        checks.append(Check(f"gone:{p}", _gone(p)))
    for pid in args.pid:
        checks.append(Check(f"pid:{pid}", _pid_alive(pid)))
    for pid in args.pid_gone:
        checks.append(Check(f"pid-gone:{pid}", _pid_gone(pid)))
    for path, pattern in args.log_match:
        checks.append(Check(f"log-match:{path}", _log_match(path, pattern)))
    for path, secs in args.log_idle:
        checks.append(Check(f"log-idle:{path}", _log_idle(path, float(secs))))
    for path, n in args.min_bytes:
        checks.append(Check(f"min-bytes:{path}", _min_bytes(path, int(n))))
    if args.gpu_free:
        checks.append(Check(f"gpu-free:{GPU_LOCK}", _gpu_free()))

    if not checks:
        ap.error("no conditions given - one of --file/--gone/--pid/--log-match/--gpu-free is required")

    label = args.label or ",".join(c.name for c in checks)
    start = time.monotonic()
    last_report = 0.0
    verdict = "timeout"
    while True:
        states = []
        for c in checks:
            met, detail = c.test()          # type: ignore[operator]
            c.met = met
            states.append(f"{c.name.split(':')[0]}={detail}")
        done = any(c.met for c in checks) if args.any else all(c.met for c in checks)
        elapsed = time.monotonic() - start
        if done:
            verdict = "met"
            break
        # A watched process that died without satisfying the condition is not "still waiting":
        # a build that crashed must not look like a slow build.
        for c in checks:
            if c.name.startswith("pid-gone:") and not c.met:
                continue
            if c.name.startswith("pid:") and not c.met:
                verdict = "died"
        if verdict == "died":
            break
        if elapsed > args.timeout:
            break
        if elapsed - last_report >= args.report:
            last_report = elapsed
            print(f"[{elapsed:7.0f}s] waiting: {' '.join(states)}", flush=True)
        time.sleep(args.interval)

    elapsed = time.monotonic() - start
    print(f"WAIT {verdict} after {elapsed:.0f}s  {label}")
    print("  " + "  ".join(f"{c.name}={('yes' if c.met else 'no')}" for c in checks))
    if args.log:
        try:
            with open(args.log, "r", errors="replace") as fh:
                lines = fh.readlines()[-args.tail:]
            print(f"  --- {args.log} (last {len(lines)}) ---")
            for line in lines:
                print("  " + line.rstrip())
        except OSError as exc:
            print(f"  (--log unreadable: {exc})")
    return {"met": 0, "timeout": 2, "died": 4}[verdict]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("WAIT interrupted")
        sys.exit(130)

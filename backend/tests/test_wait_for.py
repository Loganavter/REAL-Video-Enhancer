"""The waiter's contract: a condition, a verdict, and a code that distinguishes stuck from dead.

CPU only and no GPU: the tool exists so that a supervisor does not spend a turn on `sleep`, and a tool
that is only exercised by hand is exactly the kind that silently waits on a crashed job.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WAIT = Path(__file__).resolve().parents[2] / "scripts" / "wait_for.py"


def run(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(WAIT), *args], capture_output=True, text=True,
                          timeout=timeout)


def test_met_file(tmp_path: Path) -> None:
    target = tmp_path / "plan.bin"
    target.write_bytes(b"x" * 10)
    p = run("--file", str(target), "--timeout", "5")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "WAIT met" in p.stdout


def test_timeout_is_two(tmp_path: Path) -> None:
    p = run("--file", str(tmp_path / "never"), "--timeout", "0.3", "--interval", "0.1")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "WAIT timeout" in p.stdout


def test_dead_process_is_not_a_timeout() -> None:
    # A pid that cannot exist: the waiter must report `died` (4), not wait out its timeout (2).
    p = run("--pid", "999999", "--timeout", "30", "--interval", "0.1")
    assert p.returncode == 4, p.stdout + p.stderr
    assert "WAIT died" in p.stdout


def test_gone_and_any(tmp_path: Path) -> None:
    p = run("--gone", str(tmp_path / "absent"), "--any", "--timeout", "5")
    assert p.returncode == 0


def test_log_match(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    log.write_text("building\nplan -> /tmp/x.plan  1 MB  397.2s  fp16=True\n")
    p = run("--log-match", f"{log}::^plan ->", "--tail", "1", "--log", str(log), "--timeout", "5")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "plan -> /tmp/x.plan" in p.stdout


def test_no_conditions_is_an_error() -> None:
    p = run("--timeout", "1")
    assert p.returncode != 0
    assert "no conditions given" in (p.stderr + p.stdout)

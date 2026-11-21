"""Shared machinery for the stage wrappers: subprocess running and timing.

Every stage is an external program. This module is what they have in common:
run it, time it, capture both streams, and raise with the tail of stderr
attached when it fails. No stage catches its own failure -- a stage that fails
fails its design, which the runner records as `errored`.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


class StageError(RuntimeError):
    """A stage failed. Always fatal to the design, never to the run."""


@dataclass
class StageResult:
    """What a stage call did, for the log and for hardware_notes.md."""

    stage: str
    command: list[str]
    returncode: int
    wall_seconds: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    peak_vram_mb: float | None = None
    extra: dict = field(default_factory=dict)

    def as_log_dict(self) -> dict:
        return {
            "stage": self.stage,
            "command": " ".join(self.command),
            "returncode": self.returncode,
            "wall_seconds": round(self.wall_seconds, 2),
            "peak_vram_mb": self.peak_vram_mb,
            **self.extra,
        }


def run_command(
    command: list[str],
    stage: str,
    cwd: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout: float | None = None,
    tail_lines: int = 40,
) -> StageResult:
    """Run `command`, timing it, and raise StageError on a non-zero exit.

    The tail of both streams goes into the error message. At hour 20 the thing
    you need is the last thing the tool said, not a pointer to a log file.
    """
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise StageError(
            f"{stage}: executable not found: {command[0]!r}. Check the path in "
            "config/pipeline.yaml."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise StageError(
            f"{stage}: timed out after {timeout}s. Command: {' '.join(command)}"
        ) from exc
    elapsed = time.monotonic() - started

    out_tail = _tail(proc.stdout, tail_lines)
    err_tail = _tail(proc.stderr, tail_lines)

    if proc.returncode != 0:
        raise StageError(
            f"{stage}: exited {proc.returncode} after {elapsed:.1f}s\n"
            f"command: {' '.join(command)}\n"
            f"--- stderr tail ---\n{err_tail or '(empty)'}\n"
            f"--- stdout tail ---\n{out_tail or '(empty)'}"
        )

    return StageResult(
        stage=stage,
        command=command,
        returncode=proc.returncode,
        wall_seconds=elapsed,
        stdout_tail=out_tail,
        stderr_tail=err_tail,
        peak_vram_mb=query_peak_vram(),
    )


def _tail(text: str | None, lines: int) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def query_peak_vram() -> float | None:
    """Current VRAM use in MB from nvidia-smi, or None where it is unavailable.

    Phase 0 records peak VRAM and wall time per stage into
    runs/hardware_notes.md; this is where the number comes from. Returning None
    rather than 0.0 where there is no GPU keeps "not measured" distinct from
    "measured as nothing".
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return max(float(v) for v in proc.stdout.split() if v.strip().isdigit())
    except ValueError:
        return None

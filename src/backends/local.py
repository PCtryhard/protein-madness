"""Local backend: run the command as a subprocess on this machine.

    from src.backends import get_backend
    from src.backends.base import Resources

    be = get_backend("local", log_dir="results/logs")
    jid = be.submit(["python", "-m", "src.score.tier1", "--in", ..., "--out", ...],
                    Resources(cpus=4, time_limit="00:30:00"))
    be.wait(jid).raise_for_status()

This is the reference implementation of the interface and the one the smoke test
uses: no queue, no credentials, no network. stdout and stderr go to files under
``log_dir`` so a failure at hour 20 leaves evidence instead of a scrollback.

Job arrays run here too, serially by default (``max_parallel``), with
``SLURM_ARRAY_TASK_ID`` set exactly as on Euler. That is not emulation for its
own sake: it is how the array path gets tested on a laptop before 500 tasks are
handed to a shared cluster.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .base import (Backend, BackendError, Command, Job, JobResult, Resources,
                   command_text, normalise_command)

log = logging.getLogger("pipeline.backends.local")

DEFAULT_LOG_DIR = Path("results") / "logs"


@dataclass
class _Task:
    index: int | None                  # array index, or None for a plain job
    proc: subprocess.Popen | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    returncode: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    _handles: list = field(default_factory=list, repr=False)

    def close(self) -> None:
        for h in self._handles:
            try:
                h.close()
            except OSError:
                pass
        self._handles.clear()


@dataclass
class _LocalJob:
    job: Job
    pending: list[int | None]
    running: dict[int | None, _Task] = field(default_factory=dict)
    done: dict[int | None, _Task] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def finished(self) -> bool:
        return not self.pending and not self.running


class LocalBackend(Backend):
    """Subprocesses on this machine.

    ``max_parallel`` only affects array jobs; a plain job is always one process.
    Keep it at 1 unless the stage is known to be single-threaded, because the
    tools in this repo already use ``compute.threads``.
    """

    name = "local"

    def __init__(self, log_dir: str | Path | None = None, cwd: str | Path | None = None,
                 env: dict | None = None, max_parallel: int = 1) -> None:
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.cwd = Path(cwd) if cwd else None
        self.env_overrides = dict(env or {})
        self.max_parallel = max(1, int(max_parallel))
        self._jobs: dict[str, _LocalJob] = {}

    # ------------------------------------------------------------ submit

    def submit(self, cmd: Command, resources: Resources) -> str:
        """Start the job. Array tasks beyond ``max_parallel`` start during :meth:`wait`."""
        command, shell = normalise_command(cmd)
        job_id = f"local-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, backend=self.name, command=cmd, resources=resources,
                  workdir=self.cwd, log_dir=self.log_dir,
                  extra={"shell": shell, "command_text": command_text(cmd)})
        indices: list[int | None]
        if resources.array_size is None:
            indices = [None]
        else:
            indices = list(range(1, resources.array_size + 1))
        lj = _LocalJob(job=job, pending=indices)
        self._jobs[job_id] = lj
        self._fill(lj, command, shell)
        log.info("local: submitted %s (%d task%s) -> %s", job_id, job.n_tasks,
                 "" if job.n_tasks == 1 else "s", self.log_dir)
        return job_id

    def _fill(self, lj: _LocalJob, command, shell: bool) -> None:
        """Start pending tasks up to ``max_parallel``."""
        limit = 1 if lj.job.resources.array_size is None else self.max_parallel
        while lj.pending and len(lj.running) < limit and not lj.cancelled:
            index = lj.pending.pop(0)
            lj.running[index] = self._spawn(lj.job, index, command, shell)

    def _spawn(self, job: Job, index: int | None, command, shell: bool) -> _Task:
        suffix = "" if index is None else f".{index}"
        out_path = self.log_dir / f"{job.job_id}{suffix}.out"
        err_path = self.log_dir / f"{job.job_id}{suffix}.err"
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.env_overrides.items()})
        if index is not None:
            # Same variables the sbatch array path exports, so one script works
            # unchanged on the laptop and on Euler.
            env["SLURM_ARRAY_TASK_ID"] = str(index)
            env["SLURM_ARRAY_TASK_COUNT"] = str(job.resources.array_size)
            env["SLURM_ARRAY_JOB_ID"] = job.job_id
        out_f = out_path.open("w", encoding="utf-8")
        err_f = err_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(command, shell=shell, stdout=out_f, stderr=err_f,
                                    cwd=str(self.cwd) if self.cwd else None, env=env,
                                    text=True)
        except OSError as e:
            out_f.close()
            err_f.close()
            raise BackendError(
                f"local: cannot start {command_text(job.command)}: {e}"
            ) from e
        task = _Task(index=index, proc=proc, stdout_path=out_path, stderr_path=err_path,
                     started_at=time.time(), _handles=[out_f, err_f])
        log.debug("local: %s task %s pid %d", job.job_id, index, proc.pid)
        return task

    # ------------------------------------------------------------ wait

    def wait(self, job_id: str, timeout: float | None = None,
             poll_interval: float = 0.2) -> JobResult:
        """Block until every task exits. ``timeout`` kills the job and reports 'timeout'."""
        lj = self._job(job_id)
        command, shell = normalise_command(lj.job.command)
        deadline = None if timeout is None else time.time() + timeout
        while not lj.finished:
            self._reap(lj)
            self._fill(lj, command, shell)
            if lj.finished:
                break
            if deadline is not None and time.time() > deadline:
                self.cancel(job_id)
                self._reap(lj)
                return self._result(lj, state="timeout",
                                    message=f"killed after {timeout:.0f}s wall-clock timeout")
            time.sleep(max(0.01, poll_interval))
        if lj.cancelled:
            return self._result(lj, state="cancelled", message="cancelled by caller")
        rcs = [t.returncode for t in lj.done.values()]
        bad = [t for t in lj.done.values() if t.returncode not in (0, None)]
        state = "completed" if not bad else "failed"
        msg = ""
        if bad:
            first = sorted(bad, key=lambda t: (t.index is None, t.index or 0))[0]
            msg = (f"{len(bad)}/{len(rcs)} task(s) failed; first failure "
                   f"task={first.index} exit={first.returncode}, stderr: {first.stderr_path}")
        return self._result(lj, state=state, message=msg)

    def _reap(self, lj: _LocalJob) -> None:
        for index, task in list(lj.running.items()):
            rc = task.proc.poll()
            if rc is None:
                continue
            task.returncode = rc
            task.ended_at = time.time()
            task.close()
            lj.done[index] = lj.running.pop(index)

    def _result(self, lj: _LocalJob, state: str, message: str = "") -> JobResult:
        tasks = list(lj.done.values())
        failures = [t.returncode for t in tasks if t.returncode not in (0, None)]
        worst = failures[0] if failures else 0       # first non-zero exit code
        first = tasks[0] if tasks else None
        ends = [t.ended_at for t in tasks if t.ended_at]
        wall = (max(ends) - lj.job.submitted_at) if ends else None
        task_states = {
            t.index: ("completed" if t.returncode == 0 else "failed")
            for t in tasks if t.index is not None
        }
        res = JobResult(
            job_id=lj.job.job_id, backend=self.name, state=state,
            returncode=None if state == "cancelled" else worst,
            stdout_path=first.stdout_path if first else None,
            stderr_path=first.stderr_path if first else None,
            wall_seconds=None if wall is None else round(wall, 3),
            task_states=task_states, message=message,
        )
        log.info("local: %s %s (exit %s) in %ss", lj.job.job_id, state, res.returncode,
                 res.wall_seconds)
        return res

    # ------------------------------------------------------------ misc

    def cancel(self, job_id: str) -> None:
        lj = self._job(job_id)
        lj.cancelled = True
        lj.pending.clear()
        for task in lj.running.values():
            if task.proc and task.proc.poll() is None:
                try:
                    task.proc.send_signal(getattr(signal, "SIGTERM", signal.SIGINT))
                    task.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    task.proc.kill()
                except OSError as e:                              # pragma: no cover
                    log.warning("local: cannot signal pid %s: %s", task.proc.pid, e)
        log.info("local: cancelled %s", job_id)

    def job(self, job_id: str) -> Job:
        """The submitted :class:`~src.backends.base.Job` record."""
        return self._job(job_id).job

    def _job(self, job_id: str) -> _LocalJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise BackendError(
                f"local: unknown job id '{job_id}' (this backend keeps job state "
                "in memory, so ids do not survive the process that submitted them)"
            ) from None

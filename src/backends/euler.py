"""Euler backend: write an sbatch script, submit it, poll squeue/sacct.

    from src.backends import get_backend
    from src.backends.base import Resources

    be = get_backend("euler", compute=cfg["compute"], log_dir="results/logs")

    # one job
    jid = be.submit(["python", "-m", "src.fold.esmfold", "--in", "d.fasta",
                     "--out", "results/fold"],
                    Resources(gpus=1, gpu_mem_gb=10.0, cpus=4, memory_gb=16,
                              time_limit="04:00:00"))

    # 500 designs as ONE array job, 20 tasks at a time
    jid = be.submit_array(cmds, Resources(gpus=1, gpu_mem_gb=10.0,
                                          time_limit="00:20:00"), throttle=20)
    be.wait(jid, poll_interval=30).raise_for_status()

Why the array path matters: ~32 RTX 2080Ti are shared with ~70 participants.
500 designs submitted as 500 jobs will sit in the queue all weekend; 500 designs
as one array with a ``%N`` throttle will not. :meth:`submit_array` writes one
command per line to a task file and each array task runs its own line, which is
the standard pattern and keeps the per-task command identical to what we ran on
the laptop.

Euler specifics baked in here (verified against https://docs.hpc.ethz.ch/):

* GPUs: ``--gpus=[type:]<number>``; GPU memory: ``--gres=gpumem:<size>[units]``.
* Memory is requested PER CORE (``--mem-per-cpu``); there is no documented
  ``--mem``. :meth:`Resources.mem_per_cpu_mb` does the division.
* ``--time`` defaults to 1 hour, so always set it.
* Arrays are ``--array=1-N``; the index arrives as ``$SLURM_ARRAY_TASK_ID``.
* ``$HOME`` is 50 GB / 500k files, ``$SCRATCH`` is 2.5 TB and purged after
  ~2 weeks. Model weights go in ``$SCRATCH/weights``; see MANIFEST.md.
* No root: the environment is a venv in ``$HOME`` created by
  ``env/euler_setup.sh``, which this backend sources by default.
* Compute nodes have no outbound internet unless ``module load eth_proxy``.

This module imports and renders scripts anywhere. It refuses to SUBMIT when
``sbatch`` is not on PATH, which is the normal state of a dev laptop.
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

from .base import (Backend, BackendError, BackendUnavailable, Command, Job,
                   JobResult, Resources, command_text, check_fits)

log = logging.getLogger("pipeline.backends.euler")

DEFAULT_LOG_DIR = Path("results") / "logs"

#: Default lines run before the command. ``env/euler_setup.sh`` writes this
#: activate script; keeping the module loads in one file means a change there
#: does not need a code change here.
DEFAULT_SETUP = ('source "$HOME/enzyme-hack-env/activate.sh"',)

#: Slurm state -> our normalised state. Slurm appends reasons ("CANCELLED by
#: 1234"), so we match on the leading word.
STATE_MAP = {
    "COMPLETED": "completed",
    "RUNNING": "running",
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "REQUEUED": "pending",
    "RESIZING": "running",
    "SUSPENDED": "running",
    "COMPLETING": "running",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "BOOT_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "DEADLINE": "failed",
    "PREEMPTED": "failed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "timeout",
}
ACTIVE_STATES = frozenset({"pending", "running"})

_SUBMIT_RE = re.compile(r"(\d+)")


def _normalise_state(raw: str) -> str:
    return STATE_MAP.get((raw or "").strip().split()[0].upper() if raw.strip() else "", "unknown")


class EulerBackend(Backend):
    """Slurm on ETH Euler.

    ``compute`` is the ``compute:`` block of ``configs/pipeline.yaml``. It is
    required for GPU jobs: the 11 GB ceiling is enforced through
    :func:`~src.backends.base.check_fits` before anything is submitted.

    ``gpumem_units`` is the suffix appended to ``--gres=gpumem:``. The ETH docs
    document the option as ``gpumem:<size>[units]`` but give no example value,
    so the default is ``"g"`` and this knob exists in case the cluster wants a
    different spelling. If a submission is rejected with a GRES parse error,
    this is the first thing to change.
    """

    name = "euler"

    def __init__(self, compute: dict | None = None, log_dir: str | Path | None = None,
                 chdir: str | Path | None = None, account: str | None = None,
                 setup_commands: Sequence[str] | None = None,
                 job_name: str = "enzyme", gpumem_units: str = "g",
                 sbatch_extra: Sequence[str] = ()) -> None:
        self.compute = compute
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.chdir = Path(chdir).resolve() if chdir else Path.cwd()
        self.account = account or os.environ.get("SBATCH_ACCOUNT") or None
        self.setup_commands = list(DEFAULT_SETUP if setup_commands is None else setup_commands)
        self.job_name = job_name
        self.gpumem_units = gpumem_units
        self.sbatch_extra = list(sbatch_extra)
        self._jobs: dict[str, Job] = {}

    # ------------------------------------------------------------ availability

    def available(self) -> bool:
        return shutil.which("sbatch") is not None

    def _require_sbatch(self) -> str:
        path = shutil.which("sbatch")
        if path is None:
            raise BackendUnavailable(
                "sbatch is not on PATH: this is not a Slurm machine. Run this on "
                "a Euler login node (ssh <user>@euler.ethz.ch), or use "
                "--backend local / --backend modal. The sbatch script itself can "
                "still be rendered here with EulerBackend.render_script() for "
                "inspection; nothing was submitted."
            )
        return path

    # ------------------------------------------------------------ script

    def directives(self, resources: Resources, job_name: str, array_spec: str | None) -> list[str]:
        """The ``#SBATCH`` lines for one job. Pure function; no cluster needed."""
        out = self.log_dir
        pattern = "%x-%A_%a" if array_spec else "%x-%j"
        d = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={out.as_posix()}/{pattern}.out",
            f"#SBATCH --error={out.as_posix()}/{pattern}.err",
            f"#SBATCH --time={resources.time_limit}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={resources.cpus}",
            f"#SBATCH --mem-per-cpu={resources.mem_per_cpu_mb()}M",
        ]
        if resources.gpus > 0:
            gpu_spec = (f"{resources.gpu_model}:{resources.gpus}" if resources.gpu_model
                        else str(resources.gpus))
            d.append(f"#SBATCH --gpus={gpu_spec}")
            if resources.gpu_mem_gb > 0:
                size = int(math.ceil(resources.gpu_mem_gb))
                d.append(f"#SBATCH --gres=gpumem:{size}{self.gpumem_units}")
        if array_spec:
            d.append(f"#SBATCH --array={array_spec}")
        if self.chdir:
            d.append(f"#SBATCH --chdir={self.chdir.as_posix()}")
        if self.account:
            d.append(f"#SBATCH --account={self.account}")
        d.extend(self.sbatch_extra)
        return d

    def render_script(self, resources: Resources, job_name: str,
                      command: Command | None = None, task_file: Path | None = None,
                      array_spec: str | None = None) -> str:
        """The full sbatch script text.

        Exactly one of ``command`` (a single job, or an array whose command uses
        ``$SLURM_ARRAY_TASK_ID`` itself) or ``task_file`` (one command per line,
        task N runs line N) must be given.
        """
        if (command is None) == (task_file is None):
            raise BackendError("render_script: pass exactly one of command= or task_file=")
        lines = ["#!/bin/bash", *self.directives(resources, job_name, array_spec), ""]
        lines += [
            "set -euo pipefail",
            "",
            'echo "host      : $(hostname)"',
            'echo "slurm job : ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"',
            'echo "started   : $(date -Is)"',
            "",
            "# Environment: module loads + the venv in $HOME, from env/euler_setup.sh.",
            "# Weights live in $SCRATCH/weights ($HOME quota is 50 GB / 500k files).",
            *self.setup_commands,
            'export ENZYME_WEIGHTS="${ENZYME_WEIGHTS:-$SCRATCH/weights}"',
            "",
        ]
        if resources.gpus > 0:
            lines += [
                "if command -v nvidia-smi >/dev/null 2>&1; then",
                '  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',
                "fi",
                "",
            ]
        if task_file is not None:
            lines += [
                f'TASK_FILE="{task_file.as_posix()}"',
                'CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")',
                'if [ -z "$CMD" ]; then',
                '  echo "no command on line ${SLURM_ARRAY_TASK_ID} of $TASK_FILE" >&2',
                "  exit 2",
                "fi",
                'echo "command   : $CMD"',
                'eval "$CMD"',
            ]
        else:
            lines += [f"echo 'command   : {command_text(command)}'", command_text(command)]
        lines += ["", 'echo "finished  : $(date -Is)"', ""]
        return "\n".join(lines)

    # ------------------------------------------------------------ submit

    def submit(self, cmd: Command, resources: Resources) -> str:
        """Submit one job (or one array running the same command per task)."""
        return self._submit(resources, command=cmd)

    def submit_array(self, commands: Sequence[Command], resources: Resources,
                     throttle: int | None = None) -> str:
        """Submit N commands as ONE array job, task i running ``commands[i-1]``.

        ``throttle`` (Slurm ``%N``) caps simultaneous tasks. Set it on a shared
        cluster; ``resources.array_throttle`` is the fallback.
        """
        cmds = [command_text(c) for c in commands]
        if not cmds:
            raise BackendError("submit_array: no commands")
        if any("\n" in c for c in cmds):
            raise BackendError("submit_array: a command contains a newline; one line per task")
        res = resources.with_array(len(cmds), throttle)
        return self._submit(res, commands=cmds)

    def _submit(self, resources: Resources, command: Command | None = None,
                commands: Sequence[str] | None = None) -> str:
        sbatch = self._require_sbatch()
        if resources.gpus > 0:
            if self.compute is None:
                raise BackendError(
                    "EulerBackend was built without the compute config, so a GPU "
                    "request cannot be checked against compute.max_vram_gb. Pass "
                    "get_backend('euler', compute=cfg['compute'])."
                )
            ceiling = check_fits(resources, self.compute)
            log.info("euler: %.1f GB VRAM requested, ceiling %.1f GB",
                     resources.gpu_mem_gb, ceiling)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        job_name = f"{self.job_name}-{stamp}"
        array_spec = None
        if resources.array_size:
            array_spec = f"1-{resources.array_size}"
            if resources.array_throttle:
                array_spec += f"%{resources.array_throttle}"

        task_file = None
        if commands is not None:
            task_file = self.log_dir / f"{job_name}.tasks"
            task_file.write_text("\n".join(commands) + "\n", encoding="utf-8")
        script = self.render_script(resources, job_name, command=command,
                                    task_file=task_file, array_spec=array_spec)
        script_path = self.log_dir / f"{job_name}.sbatch"
        script_path.write_text(script, encoding="utf-8")

        proc = subprocess.run([sbatch, "--parsable", str(script_path)],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise BackendError(
                f"sbatch rejected {script_path} (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        m = _SUBMIT_RE.search(proc.stdout)
        if not m:
            raise BackendError(f"cannot read a job id out of sbatch output: {proc.stdout!r}")
        job_id = m.group(1)
        self._jobs[job_id] = Job(
            job_id=job_id, backend=self.name,
            command=command if command is not None else list(commands or []),
            resources=resources, workdir=self.chdir, log_dir=self.log_dir,
            script_path=script_path,
            extra={"task_file": str(task_file) if task_file else "", "job_name": job_name,
                   "array_spec": array_spec or ""},
        )
        log.info("euler: submitted job %s (%s, %d task%s) script=%s", job_id, job_name,
                 resources.array_size or 1, "" if (resources.array_size or 1) == 1 else "s",
                 script_path)
        return job_id

    # ------------------------------------------------------------ wait

    def wait(self, job_id: str, timeout: float | None = None,
             poll_interval: float = 30.0) -> JobResult:
        """Poll ``squeue`` until the job leaves the queue, then read ``sacct``.

        ``squeue`` only knows about active jobs, so "not in squeue" means
        finished, not missing. The final states come from ``sacct``, which is
        also the only source for per-array-task outcomes.
        """
        self._require_sbatch()
        t0 = time.time()
        deadline = None if timeout is None else t0 + timeout
        while True:
            states = self._squeue(job_id)
            if not states:
                break
            if deadline is not None and time.time() > deadline:
                return JobResult(job_id=job_id, backend=self.name, state="timeout",
                                 wall_seconds=round(time.time() - t0, 1),
                                 task_states=states,
                                 message=(f"still in the queue after {timeout:.0f}s; "
                                          f"job NOT cancelled. States: {sorted(set(states.values()))}"))
            log.debug("euler: %s %s", job_id, sorted(set(states.values())))
            time.sleep(max(1.0, poll_interval))
        return self.status(job_id, wall_seconds=time.time() - t0)

    def _squeue(self, job_id: str) -> dict[int, str]:
        """Active tasks only. ``{task_index_or_0: state}``; empty when finished."""
        proc = subprocess.run(["squeue", "--job", str(job_id), "--noheader", "--array",
                               "--format=%i|%T"],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            # An unknown job id is how squeue reports a finished job on some
            # configurations; treat it as "no longer active" and let sacct decide.
            if "Invalid job id" in proc.stderr or "invalid job id" in proc.stderr:
                return {}
            raise BackendError(f"squeue failed for job {job_id}: {proc.stderr.strip()}")
        out: dict[int, str] = {}
        for line in proc.stdout.splitlines():
            if "|" not in line:
                continue
            jid, raw = line.split("|", 1)
            state = _normalise_state(raw)
            if state not in ACTIVE_STATES:
                continue
            out[self._task_index(jid)] = state
        return out

    @staticmethod
    def _task_index(slurm_job_id: str) -> int:
        """``'12345_7' -> 7``; a non-array id and un-expanded ``12345_[1-9]`` -> 0."""
        if "_" not in slurm_job_id:
            return 0
        tail = slurm_job_id.rsplit("_", 1)[1].strip()
        return int(tail) if tail.isdigit() else 0

    def status(self, job_id: str, wall_seconds: float | None = None) -> JobResult:
        """Final outcome from ``sacct``. Safe to call on a job this process did not submit."""
        if shutil.which("sacct") is None:
            raise BackendUnavailable(
                f"sacct is not on PATH, so the outcome of job {job_id} cannot be "
                "read. Check it by hand on a login node: "
                f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed"
            )
        proc = subprocess.run(["sacct", "--jobs", str(job_id), "--noheader", "--parsable2",
                               "--allocations",
                               "--format=JobID,State,ExitCode,Elapsed"],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise BackendError(f"sacct failed for job {job_id}: {proc.stderr.strip()}")
        rows = []
        for line in proc.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 4 or "." in parts[0]:      # skip .batch / .extern steps
                continue
            rows.append((parts[0], _normalise_state(parts[1]), parts[2], parts[3]))
        if not rows:
            return JobResult(job_id=job_id, backend=self.name, state="unknown",
                             wall_seconds=wall_seconds,
                             message=(f"sacct has no record of job {job_id} yet. Slurm "
                                      "accounting can lag by a few seconds; retry, or "
                                      f"check 'sacct -j {job_id}' by hand."))

        task_states = {self._task_index(jid): st for jid, st, _, _ in rows}
        exits = []
        for _, _, exitcode, _ in rows:
            head = exitcode.split(":")[0]
            exits.append(int(head) if head.isdigit() else 0)
        failures = [(jid, st, rc) for (jid, st, _, _), rc in zip(rows, exits)
                    if st != "completed" or rc != 0]
        job = self._jobs.get(job_id)
        pattern = "%x-%A_%a" if (job and job.resources.array_size) else "%x-%j"
        if failures:
            state = "timeout" if all(f[1] == "timeout" for f in failures) else (
                "cancelled" if all(f[1] == "cancelled" for f in failures) else "failed")
            msg = (f"{len(failures)}/{len(rows)} task(s) not COMPLETED; first: "
                   f"{failures[0][0]} {failures[0][1]} exit={failures[0][2]}. "
                   f"Logs: {self.log_dir.as_posix()}/{pattern}.err")
            rc = next((f[2] for f in failures if f[2] != 0), 1)
        else:
            state, msg, rc = "completed", "", 0
        res = JobResult(job_id=job_id, backend=self.name, state=state, returncode=rc,
                        wall_seconds=None if wall_seconds is None else round(wall_seconds, 1),
                        task_states=task_states, message=msg)
        log.info("euler: %s %s (exit %s, %d task records)", job_id, state, rc, len(rows))
        return res

    # ------------------------------------------------------------ misc

    def cancel(self, job_id: str) -> None:
        if shutil.which("scancel") is None:
            raise BackendUnavailable(f"scancel is not on PATH; cancel job {job_id} by hand")
        proc = subprocess.run(["scancel", str(job_id)], capture_output=True, text=True,
                              timeout=120)
        if proc.returncode != 0:
            raise BackendError(f"scancel failed for job {job_id}: {proc.stderr.strip()}")
        log.info("euler: cancelled %s", job_id)

    def job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise BackendError(
                f"euler: job {job_id} was not submitted by this process. "
                "wait()/status() still work; the Job record does not survive."
            ) from None

"""Backend dispatch: one interface over the laptop, Euler, Modal and Tamarind.

Every backend exposes the same two calls::

    job_id = backend.submit(cmd, resources)
    result = backend.wait(job_id)

``cmd`` is a command line in the CLI contract of :mod:`src.cli`, e.g.
``["python", "-m", "src.fold.esmfold", "--in", ..., "--out", ...]``. Backends
do not know what the command does; they only know how to get it onto a machine
and how to tell when it finished. That is what makes a stage swappable between
the laptop and the cluster at hour 20 without touching the tool itself.

The 11 GB ceiling
-----------------
The reserved Euler nodes are RTX 2080Ti: ``sm_75``, 11 GB, no bf16. That number
lives in ``configs/pipeline.yaml`` under ``compute.max_vram_gb``, never in code.
:func:`check_fits` is the one place that compares a request against it, and it
refuses rather than letting the job OOM an hour into the queue. Modal is the
documented escape hatch for anything that does not fit; see
:mod:`src.backends.modal`.
"""
from __future__ import annotations

import abc
import importlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

log = logging.getLogger("pipeline.backends")

Command = str | Sequence[str]

#: Backend name -> "module:class". Imported lazily so that importing this
#: module never pulls in ``modal`` or needs ``sbatch`` on PATH.
BACKENDS: dict[str, str] = {
    "local": "src.backends.local:LocalBackend",
    "euler": "src.backends.euler:EulerBackend",
    "modal": "src.backends.modal:ModalBackend",
    "tamarind": "src.backends.tamarind:TamarindBackend",
}

#: Terminal job states, normalised across backends.
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "timeout", "unknown"})

_TIME_RE = re.compile(r"^(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?$")


class BackendError(RuntimeError):
    """Anything a backend refuses to do, with the reason in the message."""


class BackendUnavailable(BackendError):
    """The backend cannot be used from here (no ``sbatch``, no ``modal``, no API)."""


class ResourcesTooLarge(BackendError):
    """The request exceeds a ceiling in ``compute:`` in the pipeline config."""


# ---------------------------------------------------------------- resources

@dataclass(frozen=True)
class Resources:
    """What one job needs. Backend-neutral; each backend translates it.

    ``time_limit`` is Slurm's spelling (``"HH:MM:SS"``, ``"D-HH:MM:SS"`` or
    ``"MM:SS"``) because that is the most restrictive of the three backends and
    the only one that will kill us for getting it wrong.

    ``memory_gb`` is the job's TOTAL host RAM. Euler asks for memory per core,
    so :mod:`src.backends.euler` divides by ``cpus``; do not pre-divide here.

    ``array_size`` is the number of tasks in a job array, or None for a single
    job. Running 500 designs as 500 separate submissions does not finish inside
    a hackathon, so the array path is the normal path for generation and folding.
    ``array_throttle`` caps how many array tasks run at once (Slurm ``%N``),
    which is how we stay a good citizen on 32 GPUs shared with ~70 people.

    ``gpu_model`` pins a specific GPU where the backend supports it (Euler:
    ``rtx_2080``, ``rtx_3090``, ``a100``...; Modal: see its own docs). Leave it
    None unless the job genuinely depends on it.
    """

    gpus: int = 0
    gpu_mem_gb: float = 0.0
    cpus: int = 1
    time_limit: str = "01:00:00"
    memory_gb: float = 4.0
    array_size: int | None = None
    array_throttle: int | None = None
    gpu_model: str | None = None

    def __post_init__(self) -> None:
        if self.gpus < 0 or self.cpus < 1:
            raise ValueError(f"Resources: gpus must be >=0 and cpus >=1, got {self.gpus}/{self.cpus}")
        if self.gpu_mem_gb < 0 or self.memory_gb <= 0:
            raise ValueError(f"Resources: bad memory request {self.gpu_mem_gb}/{self.memory_gb} GB")
        if self.array_size is not None and self.array_size < 1:
            raise ValueError(f"Resources: array_size must be >=1 or None, got {self.array_size}")
        if self.array_throttle is not None and self.array_throttle < 1:
            raise ValueError(f"Resources: array_throttle must be >=1 or None, got {self.array_throttle}")
        self.time_limit_seconds()  # validate the spelling now, not in the queue

    def time_limit_seconds(self) -> int:
        m = _TIME_RE.match(self.time_limit.strip())
        if not m:
            raise ValueError(
                f"Resources: time_limit '{self.time_limit}' is not 'HH:MM:SS', "
                "'D-HH:MM:SS' or 'MM:SS'"
            )
        days, a, b, c = m.groups()
        d = int(days or 0)
        if c is None:                      # MM:SS
            h, mi, s = 0, int(a), int(b)
        else:                              # HH:MM:SS
            h, mi, s = int(a), int(b), int(c)
        return d * 86400 + h * 3600 + mi * 60 + s

    def mem_per_cpu_mb(self) -> int:
        """Total RAM spread over the cores, rounded up. Euler's ``--mem-per-cpu``."""
        return int(math.ceil(self.memory_gb * 1024 / self.cpus))

    def with_array(self, size: int, throttle: int | None = None) -> "Resources":
        """Copy with the array fields set. ``Resources`` is frozen on purpose."""
        return Resources(
            gpus=self.gpus, gpu_mem_gb=self.gpu_mem_gb, cpus=self.cpus,
            time_limit=self.time_limit, memory_gb=self.memory_gb,
            array_size=size, array_throttle=throttle if throttle is not None else self.array_throttle,
            gpu_model=self.gpu_model,
        )


def check_fits(resources: Resources, compute_config: dict) -> float:
    """Refuse a request that cannot run under the configured GPU ceiling.

    ``compute_config`` is the ``compute:`` block of ``configs/pipeline.yaml``.
    ``max_vram_gb`` must be present: a missing ceiling is a config bug, not a
    licence to submit whatever we like. Returns the ceiling in GB so callers
    can log it.

    A GPU job must state ``gpu_mem_gb``. An unstated requirement cannot be
    checked, and "unchecked" is how a 500-task array dies at task 1 on a 2080Ti.
    """
    if not isinstance(compute_config, dict) or compute_config.get("max_vram_gb") is None:
        raise BackendError(
            "compute.max_vram_gb is missing from the pipeline config: pass the "
            "'compute:' block of configs/pipeline.yaml to check_fits(). The GPU "
            "ceiling is a config value, not a default hidden in code."
        )
    ceiling = float(compute_config["max_vram_gb"])
    if resources.gpus <= 0:
        return ceiling
    if resources.gpu_mem_gb <= 0:
        raise ResourcesTooLarge(
            f"this job asks for {resources.gpus} GPU(s) but does not state "
            "gpu_mem_gb, so it cannot be checked against the "
            f"compute.max_vram_gb={ceiling} GB ceiling. Set Resources(gpu_mem_gb=...) "
            "to the real peak requirement."
        )
    if resources.gpu_mem_gb > ceiling:
        raise ResourcesTooLarge(
            f"job needs {resources.gpu_mem_gb} GB of VRAM but compute.max_vram_gb "
            f"is {ceiling} GB (ETH Euler reserved nodes are RTX 2080Ti: 11 GB, "
            "sm_75, no bf16). Either make it fit (fp16 + attention chunking; see "
            "compute.precision and the fold.* chunk_size settings), or run it on "
            "the modal backend, which is the documented escape hatch for >11 GB. "
            "Raise compute.max_vram_gb only if the hardware actually changed."
        )
    return ceiling


# ---------------------------------------------------------------- jobs

@dataclass
class Job:
    """A submitted job, as the submitting process remembers it."""

    job_id: str
    backend: str
    command: Command
    resources: Resources
    workdir: Path | None = None
    log_dir: Path | None = None
    script_path: Path | None = None
    submitted_at: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return self.resources.array_size or 1


@dataclass
class JobResult:
    """How a job ended. ``state`` is one of :data:`TERMINAL_STATES`.

    ``returncode`` is the worst exit code over all array tasks, so a 500-task
    array that lost three tasks is not reported as a success. ``task_states``
    maps array index -> normalised state and is empty for non-array jobs.
    ``payload`` carries a backend-specific return value (Modal functions return
    Python objects rather than exit codes); it is None for process backends.
    """

    job_id: str
    backend: str
    state: str
    returncode: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    wall_seconds: float | None = None
    task_states: dict[int, str] = field(default_factory=dict)
    payload: object = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "completed" and (self.returncode in (0, None))

    def raise_for_status(self) -> "JobResult":
        """Convenience for callers that want a failure to stop the pipeline."""
        if not self.ok:
            raise BackendError(
                f"{self.backend} job {self.job_id} ended '{self.state}' "
                f"(exit {self.returncode}): {self.message or 'see ' + str(self.stderr_path)}"
            )
        return self


# ---------------------------------------------------------------- interface

class Backend(abc.ABC):
    """The dispatch interface. Two required calls: :meth:`submit`, :meth:`wait`."""

    name: str = "base"

    @abc.abstractmethod
    def submit(self, cmd: Command, resources: Resources) -> str:
        """Start the command and return a job id. Does not block."""

    @abc.abstractmethod
    def wait(self, job_id: str, timeout: float | None = None,
             poll_interval: float = 5.0) -> JobResult:
        """Block until the job reaches a terminal state, or ``timeout`` seconds pass."""

    def cancel(self, job_id: str) -> None:
        """Best-effort cancellation. Default: not supported by this backend."""
        raise BackendUnavailable(f"{self.name}: cancel() is not implemented for this backend")

    def available(self) -> bool:
        """True when this backend can actually be used from this machine."""
        return True

    def run(self, cmd: Command, resources: Resources, timeout: float | None = None,
            poll_interval: float = 5.0) -> JobResult:
        """submit + wait. The common case for a single stage."""
        return self.wait(self.submit(cmd, resources), timeout=timeout,
                         poll_interval=poll_interval)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


def get_backend(name: str, **kwargs) -> Backend:
    """Factory. ``get_backend("euler", account="es_hackathon")``.

    Imports lazily: asking for ``local`` on the laptop must not require
    ``modal`` to be installed, and importing this module must not require
    ``sbatch`` to be on PATH.
    """
    key = (name or "").strip().lower()
    if key not in BACKENDS:
        raise BackendError(
            f"unknown backend '{name}'; known backends are {sorted(BACKENDS)}"
        )
    mod_name, cls_name = BACKENDS[key].split(":")
    try:
        module = importlib.import_module(mod_name)
    except ImportError as e:                                   # pragma: no cover
        raise BackendUnavailable(f"backend '{key}' could not be imported: {e}") from e
    return getattr(module, cls_name)(**kwargs)


def normalise_command(cmd: Command) -> tuple[list[str] | str, bool]:
    """Return ``(command, shell)``.

    A list is executed without a shell (the safe, portable form). A string is
    handed to the shell, because that is the only way ``module load ... && python
    -m ...`` can work in an sbatch script. Tools in this repo should pass lists.
    """
    if isinstance(cmd, str):
        if not cmd.strip():
            raise BackendError("empty command")
        return cmd, True
    parts = [str(c) for c in cmd]
    if not parts:
        raise BackendError("empty command")
    return parts, False


def command_text(cmd: Command) -> str:
    """One-line human/shell rendering of a command, for scripts and logs."""
    if isinstance(cmd, str):
        return cmd
    import shlex
    return " ".join(shlex.quote(str(c)) for c in cmd)

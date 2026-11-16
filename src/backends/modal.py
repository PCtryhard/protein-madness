"""Modal backend: invoke GPU functions deployed from ``env/modal_image.py``.

    from src.backends import get_backend
    be = get_backend("modal")
    print(be.gpu_check())                 # torch.cuda.get_device_name() on an A100

    jid = be.spawn("gpu_check")           # non-blocking
    be.wait(jid, timeout=600).raise_for_status()

This is the escape hatch: $30/month of free credits, A100/H100 on demand, no
event code. It is the answer to "this needs more than 11 GB" and to "Euler is
contended". :func:`~src.backends.base.check_fits` is deliberately NOT applied
here — exceeding the 2080Ti ceiling is the whole reason to come to Modal.

What is real in this module and what is not
-------------------------------------------
Real: looking up and calling a function of a DEPLOYED Modal app, spawning it in
the background, polling it, cancelling it. That is enough to run the GPU
function in ``env/modal_image.py``.

Not real: :meth:`submit`, the generic ``submit(cmd, resources)`` half of the
backend interface. Running an arbitrary ``python -m src....`` command on Modal
needs three things this repo has not got yet: the repo inside the image, a
remote command-runner function, and a decision about where weights live on
Modal (a ``modal.Volume`` vs baking them into the image). It raises
NotImplementedError naming exactly that rather than pretending. See TODO.md.

One API fact worth knowing before you plan around it: on Modal, resources are
fixed at function-definition time (``@app.function(gpu="A100-80GB")``), not at
call time. A ``Resources`` object cannot change the GPU of an already deployed
function; it can only be checked against what that function was deployed with.

Verified against the current Modal docs (2026-11):
* ``modal.Function.from_name(app_name, name)`` then ``.remote()`` / ``.spawn()``
  -- https://modal.com/docs/guide/trigger-deployed-functions
* ``FunctionCall.get(timeout=0)`` polls and raises ``TimeoutError`` while the
  call is still running; ``FunctionCall.from_id(id)`` rebuilds a handle
  -- https://modal.com/docs/reference/modal.FunctionCall
* lookups only resolve for DEPLOYED apps (``modal deploy``), not ephemeral ones.
"""
from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

from .base import (Backend, BackendError, BackendUnavailable, Command, Job,
                   JobResult, Resources)

log = logging.getLogger("pipeline.backends.modal")

#: Must match ``APP_NAME`` in ``env/modal_image.py``. ``env/`` is not an
#: importable package, so the constant is written twice on purpose; changing one
#: without the other makes every lookup fail with "app not found".
APP_NAME = "enzyme-hack"

SETUP_HINT = (
    "pip install modal && modal setup   (or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET), "
    "then deploy the app once: modal deploy env/modal_image.py"
)


class ModalBackend(Backend):
    """Calls functions of a deployed Modal app.

    ``app_name`` defaults to :data:`APP_NAME`. ``environment_name`` is Modal's
    workspace environment; leave it None for the default.
    """

    name = "modal"

    def __init__(self, app_name: str = APP_NAME, environment_name: str | None = None) -> None:
        self.app_name = app_name
        self.environment_name = environment_name
        self._modal = None
        self._calls: dict[str, Job] = {}

    # ------------------------------------------------------------ availability

    @property
    def modal(self):
        """The ``modal`` module, imported lazily with a useful error if absent."""
        if self._modal is None:
            try:
                self._modal = importlib.import_module("modal")
            except ImportError as e:
                raise BackendUnavailable(
                    "the 'modal' package is not installed in this environment "
                    "(it is deliberately not in env/requirements.txt: the core "
                    f"env is CPU-only). {SETUP_HINT}"
                ) from e
        return self._modal

    def available(self) -> bool:
        try:
            self.modal
        except BackendUnavailable:
            return False
        return True

    def function(self, function_name: str):
        """Look up a deployed function. Raises with the deploy command on failure."""
        modal = self.modal
        kwargs = {}
        if self.environment_name:
            kwargs["environment_name"] = self.environment_name
        try:
            return modal.Function.from_name(self.app_name, function_name, **kwargs)
        except Exception as e:
            raise BackendUnavailable(
                f"cannot look up modal function '{function_name}' in app "
                f"'{self.app_name}': {type(e).__name__}: {e}\n"
                "Lookups only resolve for DEPLOYED apps. Deploy it once (this "
                "builds the image, which takes tens of minutes the first time):\n"
                "  modal deploy env/modal_image.py"
            ) from e

    # ------------------------------------------------------------ real calls

    def call(self, function_name: str, *args, **kwargs):
        """Blocking remote call; returns the function's return value."""
        t0 = time.perf_counter()
        value = self.function(function_name).remote(*args, **kwargs)
        log.info("modal: %s.%s returned in %.1fs", self.app_name, function_name,
                 time.perf_counter() - t0)
        return value

    def spawn(self, function_name: str, *args, **kwargs) -> str:
        """Non-blocking remote call; returns a job id usable with :meth:`wait`."""
        fc = self.function(function_name).spawn(*args, **kwargs)
        job_id = str(fc.object_id)
        self._calls[job_id] = Job(
            job_id=job_id, backend=self.name,
            command=f"modal:{self.app_name}::{function_name}",
            resources=Resources(),
            extra={"function": function_name, "app": self.app_name},
        )
        log.info("modal: spawned %s::%s -> %s", self.app_name, function_name, job_id)
        return job_id

    def wait(self, job_id: str, timeout: float | None = None,
             poll_interval: float = 5.0) -> JobResult:
        """Poll the function call until it returns, raises, or ``timeout`` passes.

        The remote return value lands in ``JobResult.payload``; Modal functions
        return Python objects, not exit codes, so ``returncode`` is 0 or 1.
        """
        modal = self.modal
        try:
            fc = modal.FunctionCall.from_id(job_id)
        except Exception as e:
            raise BackendError(f"modal: unknown function call id '{job_id}': {e}") from e
        meta = self._calls.get(job_id)
        t0 = time.time()
        deadline = None if timeout is None else t0 + timeout
        while True:
            try:
                value = fc.get(timeout=0)
            except TimeoutError:
                # Documented: get(timeout=0) polls and raises TimeoutError while
                # the call is still running.
                if deadline is not None and time.time() > deadline:
                    return JobResult(job_id=job_id, backend=self.name, state="timeout",
                                     wall_seconds=round(time.time() - t0, 1),
                                     message=(f"still running after {timeout:.0f}s; the call was "
                                              "NOT cancelled (use cancel()). Logs: "
                                              "modal app logs / the Modal dashboard."))
                time.sleep(max(0.5, poll_interval))
                continue
            except Exception as e:
                # .get() re-raises whatever the remote function raised.
                return JobResult(job_id=job_id, backend=self.name, state="failed",
                                 returncode=1, wall_seconds=round(time.time() - t0, 1),
                                 message=f"{type(e).__name__}: {e}")
            wall = round(time.time() - t0, 1)
            log.info("modal: %s completed in %ss", job_id, wall)
            return JobResult(job_id=job_id, backend=self.name, state="completed",
                             returncode=0, wall_seconds=wall, payload=value,
                             message="" if meta is None else f"{meta.extra.get('function', '')}")

    def cancel(self, job_id: str) -> None:
        self.modal.FunctionCall.from_id(job_id).cancel()
        log.info("modal: cancelled %s", job_id)

    def gpu_check(self) -> dict:
        """Call the trivial GPU function in ``env/modal_image.py``.

        This is the "is Modal usable at all" probe: it returns the device name,
        the torch/CUDA versions and the compute capability. Run it on Friday,
        not on Saturday, because the first call also builds the image.
        """
        return self.call("gpu_check")

    # ------------------------------------------------------------ not real yet

    def submit(self, cmd: Command, resources: Resources) -> str:
        """NOT IMPLEMENTED. Use :meth:`spawn` with a deployed function name.

        Running an arbitrary repo command on Modal is missing three things:
        the repo inside the image (``Image.add_local_dir(..., copy=True)``), a
        remote command-runner function to exec it, and a decision on where
        weights live on Modal (``modal.Volume`` vs baked into the image).
        """
        raise NotImplementedError(
            "modal backend cannot run an arbitrary command yet: env/modal_image.py "
            "contains no remote command-runner, the repo is not added to the image, "
            "and the weights location on Modal (modal.Volume vs baked-in) is not "
            f"decided. Use ModalBackend.spawn('<function>') / .call('<function>') "
            f"against a deployed function instead. Command was: {cmd!r}. See TODO.md."
        )

    def upload_weights(self, local_dir: str | Path, volume_name: str) -> None:
        """NOT IMPLEMENTED. Needs the weights-storage decision first."""
        raise NotImplementedError(
            "weights on Modal are not set up: we have not decided between a "
            "modal.Volume populated once and an image that bakes the weights in. "
            "See MANIFEST.md and TODO.md."
        )

"""Modal image definition + the one trivial GPU function, for src/backends/modal.py.

    # ONE-OFF, DO THIS BEFORE SATURDAY. The first build takes TENS OF MINUTES
    # (a ~2.5 GB CUDA torch wheel). Paying that at 09:00 on Saturday with the
    # challenge on the screen is exactly the cost this file exists to avoid.
    pip install modal          # not in env/requirements.txt: the core env is CPU-only
    modal setup                # one-time auth, writes ~/.modal.toml
    modal deploy env/modal_image.py            # builds the image AND deploys the app
    modal run env/modal_image.py::gpu_check    # smoke test: prints the device name

After ``modal deploy`` the image layers are cached in the workspace, so later
deploys are seconds unless a pinned version below changes. Only the deployed
app is reachable by name from other processes:
``modal.Function.from_name("enzyme-hack", "gpu_check")`` — an app started with
``modal run``/``modal serve`` is ephemeral and will NOT resolve.

Why Modal exists in this repo: the reserved Euler GPUs are RTX 2080Ti, 11 GB,
sm_75, no bf16. Anything that needs more than that (Boltz-2, Chai-1, a big
batch) comes here instead of being squeezed until it OOMs. Modal is therefore
the one backend that does NOT apply ``compute.max_vram_gb``.

Verified against the current Modal docs (2026-11):
* ``modal.App(name=...)`` -- ``modal.Stub`` is an error since Modal 1.0
  (https://modal.com/docs/guide/apps)
* ``Image.debian_slim(python_version=...)``, ``.apt_install``, ``.pip_install``
  with ``extra_index_url``, ``force_build`` (https://modal.com/docs/guide/images)
* GPU strings accepted by ``gpu=``: "T4", "L4", "A10", "L40S", "A100",
  "A100-40GB", "A100-80GB", "H100", "H200", "B200", and "<type>:<n>" for
  multi-GPU (https://modal.com/docs/guide/gpu)
* ``modal deploy APP_REF`` / ``modal run {file}::{function}``
  (https://modal.com/docs/reference/cli/deploy, .../cli/run)

Pins: bump them deliberately, in a commit of their own, and re-run the deploy.
A changed pin invalidates that image layer and everything after it, i.e. it
buys back the tens of minutes.
"""
from __future__ import annotations

import modal

#: Must match ``APP_NAME`` in ``src/backends/modal.py``.
APP_NAME = "enzyme-hack"

PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.5.1"
NUMPY_VERSION = "1.26.4"
#: CUDA 12.4 wheels. Modal ships the lower half of the CUDA stack in the
#: container, so a cu12x torch wheel runs without installing the CUDA toolkit.
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

#: Deliberately minimal: torch + numpy and nothing else. This image exists to
#: prove the GPU path works end to end. Model stacks (esmfold, boltz, ...) get
#: their own images when the fold backends land, so that a change to one does
#: not rebuild the others. Each layer is cached independently.
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install(f"torch=={TORCH_VERSION}", extra_index_url=TORCH_INDEX)
    .pip_install(f"numpy=={NUMPY_VERSION}")
)

app = modal.App(name=APP_NAME, image=image)


@app.function(gpu="A100-40GB", timeout=600)
def gpu_check() -> dict:
    """Print and return what GPU we actually got. The whole point of this file.

    Returns a plain dict so ``src.backends.modal.ModalBackend.gpu_check()`` can
    put it straight into a log or a meta.json. ``capability`` is the thing to
    compare against Euler: the 2080Ti is sm_75 and has no bf16, so a kernel that
    works here may still not work there.
    """
    import torch

    available = torch.cuda.is_available()
    info = {
        "cuda_available": available,
        "device_name": torch.cuda.get_device_name() if available else None,
        "device_count": torch.cuda.device_count(),
        "capability": ".".join(map(str, torch.cuda.get_device_capability())) if available else None,
        "total_memory_gb": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                            if available else None),
        "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    print(torch.cuda.get_device_name() if available else "NO CUDA DEVICE")
    for k, v in info.items():
        print(f"{k:18s} {v}")
    return info


@app.local_entrypoint()
def main() -> None:
    """``modal run env/modal_image.py`` -- runs :func:`gpu_check` remotely."""
    import json

    print(json.dumps(gpu_check.remote(), indent=2))

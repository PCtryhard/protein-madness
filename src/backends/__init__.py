"""Compute backends: local, euler, modal, tamarind.

    from src.backends import get_backend
    from src.backends.base import Resources, check_fits

Submodules are imported lazily by :func:`~src.backends.base.get_backend`, so
importing this package never requires ``modal`` to be installed or ``sbatch``
to be on PATH.
"""
from __future__ import annotations

from .base import (BACKENDS, Backend, BackendError, BackendUnavailable, Job,
                   JobResult, Resources, ResourcesTooLarge, check_fits,
                   get_backend)

__all__ = [
    "BACKENDS", "Backend", "BackendError", "BackendUnavailable", "Job",
    "JobResult", "Resources", "ResourcesTooLarge", "check_fits", "get_backend",
]

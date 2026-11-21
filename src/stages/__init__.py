"""The five stage wrappers, each one external program.

One forward pass is: seed PDB with fixed motif residues, then RFdiffusion
partial diffusion, then LigandMPNN, then Boltz-2, then hydrogen placement,
then scoring.

The mutation branch returns straight to Boltz-2, because a point mutation needs
no new backbone and Boltz-2 repacks the neighbourhood implicitly when it
predicts the complex from sequence.
"""

from __future__ import annotations

from . import boltz, ligandmpnn, protonate, rfdiffusion
from .base import StageError, StageResult, run_command

__all__ = [
    "StageError",
    "StageResult",
    "boltz",
    "ligandmpnn",
    "protonate",
    "rfdiffusion",
    "run_command",
]

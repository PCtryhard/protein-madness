"""The five stage wrappers, each one external program.

One forward pass is: seed PDB with fixed motif residues, then RFD3 (motif
scaffolding, or partial diffusion on the wiggle branch), then LigandMPNN, then
Boltz-2, then hydrogen placement, then scoring.

RFD3 rather than RFdiffusion: that is what the clusters have, and the two are
not interchangeable. The RFdiffusion wrapper is in outdated/src/stages/ for a
cluster that has the other tool.

The mutation branch returns straight to Boltz-2, because a point mutation needs
no new backbone and Boltz-2 repacks the neighbourhood implicitly when it
predicts the complex from sequence.
"""

from __future__ import annotations

from . import boltz, ligandmpnn, protonate, rfd3
from .base import StageError, StageResult, run_command

__all__ = [
    "StageError",
    "StageResult",
    "boltz",
    "ligandmpnn",
    "protonate",
    "rfd3",
    "run_command",
]

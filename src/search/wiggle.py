"""The wiggle branch: re-noise a parent backbone, re-denoise, redesign, refold.

One correction on this branch, from PIPELINE.md. Re-noising the Boltz-2
prediction feeds RFdiffusion a backbone from a different distribution than the
crystal-like structures it was trained on. Re-noise the RFdiffusion backbone
that produced the parent, carrying the parent's sequence forward.
`search.wiggle.source` defaults to `rfdiffusion_backbone` for this reason; the
other setting exists so the assumption can be tested rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import WiggleCfg
from .tree import Candidate


class WiggleError(RuntimeError):
    """The parent backbone for a wiggle step could not be determined."""


@dataclass(frozen=True)
class WiggleJob:
    """One parent to re-noise, with the backbone the branch will act on."""

    parent: Candidate
    backbone_path: str
    carry_sequence: str

    @property
    def parent_id(self) -> str:
        return self.parent.design_id


def backbone_for(parent: Candidate, cfg: WiggleCfg) -> str:
    """Which structure file this branch re-noises, per `search.wiggle.source`.

    Raises rather than falling back to the other source. A silent fallback
    would mean half a run used one distribution and half the other, and the
    comparison the `boltz_prediction` setting exists to enable would be
    meaningless.
    """
    if cfg.source == "rfdiffusion_backbone":
        path = parent.backbone_path
        if not path:
            raise WiggleError(
                f"design {parent.design_id} has no recorded RFdiffusion backbone, "
                "which search.wiggle.source='rfdiffusion_backbone' requires. A "
                "mutation-branch design has no backbone of its own: it inherits "
                "its parent's, which the runner records when the design is stored."
            )
    elif cfg.source == "boltz_prediction":
        path = parent.structure_path
        if not path:
            raise WiggleError(
                f"design {parent.design_id} has no recorded Boltz-2 prediction, "
                "which search.wiggle.source='boltz_prediction' requires."
            )
    else:  # pragma: no cover - config.Literal forbids anything else
        raise WiggleError(f"unknown wiggle source {cfg.source!r}")

    if not Path(path).exists():
        raise WiggleError(
            f"backbone for design {parent.design_id} is recorded as {path}, which "
            "does not exist. The run directory may have been moved or pruned."
        )
    return path


def jobs_for(parents: list[Candidate], cfg: WiggleCfg) -> list[WiggleJob]:
    """Build one job per parent, carrying each parent's sequence forward."""
    return [
        WiggleJob(
            parent=parent,
            backbone_path=backbone_for(parent, cfg),
            carry_sequence=parent.sequence,
        )
        for parent in parents
    ]

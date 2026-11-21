"""Scoring: confidence, hydrogen-bond geometry, and the weighted combination.

The one entry point most callers want is `score_design`, which takes a
protonated structure plus Boltz-2's confidence sidecar and returns a `Score`
with every component filled in.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..tracking import Provenance, Structure
from .combine import Score, average_replicates, combine
from .confidence import (
    ConfidenceError,
    logged_metrics,
    motif_pae,
    normalise,
    read_confidence_json,
)
from .hbond import (
    BondMeasurement,
    HBondGeometryError,
    angle_term,
    distance_term,
    measure_all,
    measure_bond,
)

__all__ = [
    "BondMeasurement",
    "ConfidenceError",
    "HBondGeometryError",
    "Score",
    "ScoredDesign",
    "angle_term",
    "average_replicates",
    "combine",
    "distance_term",
    "logged_metrics",
    "measure_all",
    "measure_bond",
    "motif_pae",
    "normalise",
    "read_confidence_json",
    "score_design",
]


class ScoredDesign:
    """A score plus everything worth logging alongside it.

    The extra metrics are logged for every design without being scored on, so
    the weighting can be revisited offline without recomputing anything.
    """

    __slots__ = ("score", "measurements", "extra_metrics")

    def __init__(
        self,
        score: Score,
        measurements: list[BondMeasurement],
        extra_metrics: dict[str, float],
    ) -> None:
        self.score = score
        self.measurements = measurements
        self.extra_metrics = extra_metrics

    def hbond_raw(self) -> list[dict]:
        return [m.as_log_dict() for m in self.measurements]


def score_design(
    cfg: Config,
    structure: Structure,
    provenance: Provenance,
    confidence_json: str | Path,
    motif_pae_value: float | None = None,
) -> ScoredDesign:
    """Score one predicted, protonated design.

    Raises rather than returning a partial score. A design whose hydrogen-bond
    atoms could not be resolved is never scored: it is recorded as errored and
    the search moves on.
    """
    raw = read_confidence_json(confidence_json)
    confidence = normalise(
        raw, cfg.pipeline.scoring.confidence, motif_pae=motif_pae_value
    )

    measurements = measure_all(
        cfg.target.hbonds,
        structure,
        provenance,
        cfg.pipeline.scoring.angle.rule,
    )
    score = combine(confidence, measurements, cfg.pipeline.scoring)

    extra = logged_metrics(raw)
    if motif_pae_value is not None:
        extra["motif_pae"] = motif_pae_value

    return ScoredDesign(score=score, measurements=measurements, extra_metrics=extra)

"""The combined score.

    S = w_c * c + w_g * ( w_d * agg_i(d_i) + w_a * agg_i(a_i) )

with the aggregation over bonds set by `scoring.aggregate`. Use `mean` while
exploring and `min` when every named bond must form, since `min` makes one bad
bond dominate.

The gate is separate from the weighting: any bond with `required: true` beyond
`distance.fail` sets the total to zero and the design is recorded as gated,
never deleted, so the search can see what it rejected.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..config import ScoringCfg
from .hbond import BondMeasurement, angle_term, distance_term, is_gated


@dataclass(frozen=True)
class Score:
    """A design's score and every component that went into it."""

    total: float
    confidence: float
    distance: float
    angle: float
    geometry: float
    gated: bool
    gated_bonds: list[str] = field(default_factory=list)
    per_bond: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_log_dict(self) -> dict:
        return {
            "total": self.total,
            "term_confidence": self.confidence,
            "term_distance": self.distance,
            "term_angle": self.angle,
            "term_geometry": self.geometry,
            "gated": self.gated,
            "gated_bonds": list(self.gated_bonds),
            "per_bond": self.per_bond,
        }


def _aggregate(values: list[float], how: str) -> float:
    if not values:
        # config.Target rejects an empty hbond list, so this is unreachable from
        # a loaded config; it is here so a programmatic caller gets an error
        # rather than a silent 0.0 that would read as "terrible geometry".
        raise ValueError("cannot aggregate an empty list of bond terms")
    if how == "mean":
        return float(statistics.fmean(values))
    if how == "min":
        return float(min(values))
    raise ValueError(f"unknown aggregation {how!r}")


def combine(
    confidence: float,
    measurements: list[BondMeasurement],
    cfg: ScoringCfg,
) -> Score:
    """Normalise every term, aggregate over bonds, and apply the gate."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"confidence term {confidence} is outside [0, 1]; it must be "
            "normalised before combining, or the weights are meaningless"
        )

    per_bond: dict[str, dict[str, float]] = {}
    d_terms: list[float] = []
    a_terms: list[float] = []

    for m in measurements:
        d = distance_term(m.distance, cfg.distance)
        a = angle_term(m.angle, m.angle_ideal, cfg.angle)
        d_terms.append(d)
        a_terms.append(a)
        per_bond[m.bond_id] = {
            "distance_raw": m.distance,
            "angle_raw": m.angle,
            "angle_ideal": m.angle_ideal,
            "distance_term": d,
            "angle_term": a,
        }

    d_agg = _aggregate(d_terms, cfg.aggregate)
    a_agg = _aggregate(a_terms, cfg.aggregate)
    geometry = (
        cfg.geometry_weights.distance * d_agg
        + cfg.geometry_weights.angle * a_agg
    )
    total = cfg.weights.confidence * confidence + cfg.weights.geometry * geometry

    failed = is_gated(measurements, cfg.distance)
    if failed:
        total = 0.0

    return Score(
        total=total,
        confidence=confidence,
        distance=d_agg,
        angle=a_agg,
        geometry=geometry,
        gated=bool(failed),
        gated_bonds=failed,
        per_bond=per_bond,
    )


def average_replicates(scores: list[Score]) -> Score:
    """Mean of several Boltz-2 seeds for the same sequence.

    Phase 0 decides whether this is needed: if the standard deviation of the
    geometry sub-score across seeds is more than about half the spread between
    genuinely different designs, the geometry terms are mostly prediction noise
    and `run.eval_replicates` should go to 3.

    A design gated in ANY replicate is gated. Averaging a zero from a gated
    replicate with good scores from others would let a design that sometimes
    loses a required bond outrank one that always keeps it.
    """
    if not scores:
        raise ValueError("cannot average an empty list of scores")
    if len(scores) == 1:
        return scores[0]

    gated_bonds = sorted({b for s in scores for b in s.gated_bonds})
    any_gated = any(s.gated for s in scores)

    per_bond: dict[str, dict[str, float]] = {}
    for bond_id in scores[0].per_bond:
        keys = scores[0].per_bond[bond_id]
        per_bond[bond_id] = {
            k: float(statistics.fmean(
                s.per_bond[bond_id][k] for s in scores if bond_id in s.per_bond
            ))
            for k in keys
        }

    mean = lambda attr: float(statistics.fmean(getattr(s, attr) for s in scores))

    return Score(
        total=0.0 if any_gated else mean("total"),
        confidence=mean("confidence"),
        distance=mean("distance"),
        angle=mean("angle"),
        geometry=mean("geometry"),
        gated=any_gated,
        gated_bonds=gated_bonds,
        per_bond=per_bond,
    )

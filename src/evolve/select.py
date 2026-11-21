"""Fitness, hard gates, and who becomes a parent.

The danger here is specific and worth stating before any code: **a single-seed
score is noisy, and selecting on a noisy score evolves the population towards
lucky seeds rather than good designs.** Two generations of that and the run is
worthless. So selection is two-stage - triage every child cheaply, then
re-measure the leaders across several seeds, and *only confirmed scores may
become parents*. The consistency metric therefore does double duty: it is the
judges' metric 4 and it is our noise control.

**Fitness is a cost: lower is better.** It is dominated by geometry, because
geometry is what the reference design fails at and where the dynamic range is
(BRIEF.md 1: the catalytic contact went 2.53 A in the design to 11.18 A in the
re-prediction). iPDE and burial are supporting terms. Everything is weighted in
*interpretable units* rather than z-scores, because z-scores need a population
and we want a design's cost to mean the same thing in generation 1 and
generation 20 - otherwise "did we improve?" has no answer.

The weights below are a starting point, not a result. BRIEF.md 6 sets them from
the calibration controls. Whatever they end up as, they get written down once
and are not tuned per-design: tuning a selection constant mid-run is how you
fool yourself.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("pipeline.evolve.select")

#: Target distance for a productive contact. Below this, no penalty: the
#: geometry is already as good as the metric can see.
IDEAL_CONTACT_A = 3.5


@dataclass(frozen=True)
class FitnessWeights:
    """Weights for the cost function. Written down once, not tuned per-design."""

    #: Angstroms of base-to-ligand error. The dominant term, by design.
    base_geometry: float = 1.0
    #: The donor matters, but the base is the catalyst.
    donor_geometry: float = 0.5
    #: iPDE is roughly 0-30; 0.5 puts a 2-unit iPDE change on par with 1 A.
    ipde: float = 0.5
    #: Burial shortfall below the target, per CA atom.
    burial: float = 0.05
    #: Target burial. The reference design scores 30; aim above it.
    burial_target: float = 35.0
    #: k in "mean - k * spread". Penalises designs that only work on one seed.
    spread: float = 1.0


def contact_penalty(distance: float, ideal: float = IDEAL_CONTACT_A) -> float:
    """Angstroms beyond the ideal approach distance. Zero if close enough.

    One-sided on purpose: a contact at 3.0 A is not worse than one at 3.5 A for
    any reason the scoring metric can see, and penalising it would push designs
    towards a precision we have no evidence the judge rewards.
    """
    if not math.isfinite(distance):
        return math.inf
    return max(0.0, float(distance) - ideal)


def fitness(aggregate: dict, weights: FitnessWeights = FitnessWeights()) -> float:
    """Cost of one design, from :func:`src.score.official.aggregate_seeds` output.

    Lower is better. ``inf`` means the design could not be measured, which is
    not the same as being bad and must not be silently treated as zero.
    """
    base = aggregate.get("base_to_ligand_a_mean", math.nan)
    donor = aggregate.get("donor_to_ligand_a_mean", math.nan)
    ipde = aggregate.get("ipde_mean", math.nan)
    burial = aggregate.get("base_burial_ca_count_mean", math.nan)

    if not math.isfinite(base) or not math.isfinite(ipde):
        return math.inf

    cost = weights.base_geometry * contact_penalty(base)
    cost += weights.donor_geometry * contact_penalty(donor) if math.isfinite(donor) else 0.0
    cost += weights.ipde * float(ipde)
    if math.isfinite(burial):
        cost += weights.burial * max(0.0, weights.burial_target - float(burial))

    # Metric 4: penalise designs whose numbers move between seeds. A mediocre
    # mean with near-zero spread is a better bet than a brilliant outlier.
    spread = 0.0
    for key, scale in (
        ("base_to_ligand_a_std", weights.base_geometry),
        ("ipde_std", weights.ipde),
    ):
        value = aggregate.get(key, math.nan)
        if math.isfinite(value):
            spread += scale * float(value)
    cost += weights.spread * spread
    return float(cost)


# ------------------------------------------------------------------- gates

@dataclass(frozen=True)
class HardGates:
    """Conditions a design must meet to stay in the population at all.

    Without these the loop will cheerfully discover that it can minimise one
    term by abandoning the active site.
    """

    max_length: int = 260
    #: Beyond this the ligand has left the pocket; the design is not an enzyme.
    max_base_contact_a: float = 12.0
    #: Below the reference design's burial we are going backwards.
    min_burial: float = 15.0
    #: Severe clashes mean a strained pose the model will relieve by ejecting
    #: the ligand - the exact failure we are trying to fix.
    max_clash_pairs: int = 8


def passes_gates(aggregate: dict, sequence: str, gates: HardGates = HardGates()) -> tuple[bool, list[str]]:
    """(survives, reasons it did not). Reasons are recorded, never discarded."""
    reasons = []
    if len(sequence) > gates.max_length:
        reasons.append(f"length {len(sequence)} > {gates.max_length}")
    base = aggregate.get("base_to_ligand_a_mean", math.nan)
    if not math.isfinite(base):
        reasons.append("base contact not measurable")
    elif base > gates.max_base_contact_a:
        reasons.append(f"base contact {base:.2f} A > {gates.max_base_contact_a} A")
    burial = aggregate.get("base_burial_ca_count_mean", math.nan)
    if math.isfinite(burial) and burial < gates.min_burial:
        reasons.append(f"burial {burial:.0f} < {gates.min_burial}")
    return (not reasons), reasons


# --------------------------------------------------------------- selection

def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("hamming needs equal-length sequences")
    return sum(1 for x, y in zip(a, b) if x != y)


def select_parents(
    candidates: list[tuple[str, float]],
    n_parents: int,
    min_hamming: int = 2,
) -> list[str]:
    """Pick parents by cost, refusing to fill the slate with near-identical designs.

    ``candidates`` is [(sequence, cost)]. Greedy: take the best, then the best
    remaining that differs from everything already chosen by at least
    ``min_hamming`` positions. If diversity runs out we return fewer parents
    rather than duplicating a lineage - the driver treats a short slate as the
    signal to inject fresh blood (see :mod:`src.evolve.diversity`).
    """
    if n_parents < 1:
        raise ValueError("n_parents must be >= 1")
    ranked = sorted(
        (c for c in candidates if math.isfinite(c[1])), key=lambda item: item[1]
    )
    chosen: list[str] = []
    for sequence, _cost in ranked:
        if len(chosen) >= n_parents:
            break
        if all(hamming(sequence, other) >= min_hamming for other in chosen):
            chosen.append(sequence)
    if len(chosen) < n_parents:
        log.warning(
            "only %d of %d requested parents met the diversity floor (min_hamming=%d)",
            len(chosen), n_parents, min_hamming,
        )
    return chosen


def triage_cut(costs: list[float], keep: int) -> list[int]:
    """Indices of the ``keep`` cheapest finite costs, best first.

    Stage one of two-stage selection: cheap single-seed scores decide who is
    worth measuring properly, and nothing here may become a parent.
    """
    finite = [(i, c) for i, c in enumerate(costs) if math.isfinite(c)]
    finite.sort(key=lambda item: item[1])
    return [i for i, _ in finite[:keep]]


def improved(current_best: float, previous_best: float, tolerance: float = 1e-6) -> bool:
    """Did this generation actually beat the last one?

    Drives the stagnation trigger in :meth:`src.evolve.mutate.MutationPolicy.stagnant`,
    and is the measurement behind BRIEF.md's Gate 5.
    """
    if not math.isfinite(current_best):
        return False
    if not math.isfinite(previous_best):
        return True
    return current_best < previous_best - tolerance

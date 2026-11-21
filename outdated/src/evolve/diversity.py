"""Guards against the loop quietly converging on one design.

Four separate failure modes, because they fail differently and the fixes differ:

1. **Re-folding what we already folded.** A content-addressed cache. Free, and
   it saves a surprising amount once mutations start reverting.
2. **Population collapse** - everyone descended from one lineage, every sequence
   a near-copy. Measured by mean pairwise Hamming distance, per-position
   entropy and surviving lineage count.
3. **Structural convergence** - children fold to the same structure as their
   parent and only the *score* jitters. That means we are sampling seed noise,
   not exploring, and the generation taught us nothing. Measured by parent-child
   CA-RMSD.
4. **Eligibility drift** - evolution moves us away from HG3.17 (good) but could
   in principle move us towards some natural protein (disqualifying). Checked
   against the sequence database every few generations, not only at submission.

Checked every generation; acted on every second or third (BRIEF.md 4.5).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("pipeline.evolve.diversity")


def sequence_id(sequence: str) -> str:
    """Stable short id for a sequence. The cache key, and the design id."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:12]


class SequenceCache:
    """Never fold the same sequence twice.

    Backed by a JSON file so it survives the driver exiting between
    generations, which it does by design (one generation = one Slurm job).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.warning("cache at %s is corrupt; starting empty", self.path)
                self._data = {}

    def __contains__(self, sequence: str) -> bool:
        return sequence_id(sequence) in self._data

    def get(self, sequence: str) -> dict | None:
        return self._data.get(sequence_id(sequence))

    def put(self, sequence: str, record: dict) -> None:
        self._data[sequence_id(sequence)] = record

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._data)


# ------------------------------------------------------------- measurement

def mean_pairwise_hamming(sequences: list[str]) -> float:
    """Average number of differing positions across all pairs.

    The headline diversity number. Falls towards zero as the population
    collapses onto one design.
    """
    unique = [s for s in sequences if s]
    if len(unique) < 2:
        return 0.0
    lengths = {len(s) for s in unique}
    if len(lengths) != 1:
        raise ValueError("cannot compare sequences of different lengths")
    total = 0
    pairs = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            total += sum(1 for a, b in zip(unique[i], unique[j]) if a != b)
            pairs += 1
    return total / pairs if pairs else 0.0


def positional_entropy(sequences: list[str]) -> np.ndarray:
    """Shannon entropy (bits) at each position across the population.

    Shows *where* diversity has been lost, which tells you whether the
    population has converged on a genuinely good motif or merely inbred.
    """
    if not sequences:
        raise ValueError("no sequences")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError("cannot compare sequences of different lengths")
    length = lengths.pop()
    out = np.zeros(length, dtype=float)
    for position in range(length):
        counts = Counter(s[position] for s in sequences)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        out[position] = entropy
    return out


def lineage_counts(records: list[dict]) -> dict[str, int]:
    """How many survivors each founding lineage has.

    ``records`` need a ``lineage`` key; the driver stamps it at generation 0 and
    carries it through every child.
    """
    return dict(Counter(r.get("lineage", "unknown") for r in records))


def parent_child_rmsd(parent_coords: np.ndarray, child_coords: np.ndarray) -> float:
    """CA-RMSD after optimal superposition. Kabsch, no external dependency.

    Near-zero RMSD across a whole generation while scores move means the score
    movement is seed noise - the signature of failure mode 3.
    """
    if parent_coords.shape != child_coords.shape:
        raise ValueError("coordinate arrays must have the same shape")
    if parent_coords.ndim != 2 or parent_coords.shape[1] != 3:
        raise ValueError("expected an (N, 3) coordinate array")
    if not (np.isfinite(parent_coords).all() and np.isfinite(child_coords).all()):
        return math.nan
    p = parent_coords - parent_coords.mean(axis=0)
    c = child_coords - child_coords.mean(axis=0)
    u, _s, vt = np.linalg.svd(c.T @ p)
    rotation = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    aligned = c @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - p) ** 2, axis=1))))


# ---------------------------------------------------------------- verdict

@dataclass(frozen=True)
class DiversityThresholds:
    """When to intervene. Deliberately generous: we would rather inject fresh
    sequences one generation early than discover inbreeding five late."""

    min_mean_hamming: float = 3.0
    min_lineages: int = 2
    #: Below this, children are structurally indistinguishable from parents.
    min_parent_child_rmsd: float = 0.5


@dataclass
class DiversityReport:
    mean_hamming: float
    n_lineages: int
    median_parent_child_rmsd: float
    collapsed: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return {
            "mean_hamming": self.mean_hamming,
            "n_lineages": self.n_lineages,
            "median_parent_child_rmsd": self.median_parent_child_rmsd,
            "collapsed": self.collapsed,
            "reasons": self.reasons,
        }


def assess(
    sequences: list[str],
    records: list[dict] | None = None,
    rmsds: list[float] | None = None,
    thresholds: DiversityThresholds = DiversityThresholds(),
) -> DiversityReport:
    """Has the population collapsed, and why?

    A ``collapsed`` verdict is the driver's trigger to widen the mutation
    schedule and inject fresh sequences - not to stop.
    """
    reasons: list[str] = []
    mean_h = mean_pairwise_hamming(sequences)
    if len(sequences) >= 2 and mean_h < thresholds.min_mean_hamming:
        reasons.append(f"mean pairwise Hamming {mean_h:.1f} < {thresholds.min_mean_hamming}")

    n_lineages = len(lineage_counts(records)) if records else 1
    if records and n_lineages < thresholds.min_lineages:
        reasons.append(f"only {n_lineages} surviving lineage(s)")

    median_rmsd = math.nan
    if rmsds:
        finite = [r for r in rmsds if math.isfinite(r)]
        if finite:
            median_rmsd = float(np.median(finite))
            if median_rmsd < thresholds.min_parent_child_rmsd:
                reasons.append(
                    f"median parent-child RMSD {median_rmsd:.2f} A < "
                    f"{thresholds.min_parent_child_rmsd} A: children fold like their "
                    f"parents, so score movement is seed noise"
                )

    report = DiversityReport(mean_h, n_lineages, median_rmsd, bool(reasons), reasons)
    if report.collapsed:
        log.warning("population collapse: %s", "; ".join(reasons))
    return report

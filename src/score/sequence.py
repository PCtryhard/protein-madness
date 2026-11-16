"""Tier-1 metrics: sequence only. CPU, no model weights, no databases.

Every function here takes a sequence (and sometimes the parent or the resolved
reference) and returns a scalar or a small dict. Each is explainable to a judge
in one sentence; that sentence is the first line of its docstring.

Numbering: positions are 1-indexed sequence positions, which equal normalised
structure positions (see src/io/structure.py).
"""
from __future__ import annotations

from collections import Counter

from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

from ..io.structure import ONE_TO_THREE

# --- physicochemical constants -------------------------------------------------
# Kyte & Doolittle (1982) hydropathy, for GRAVY.
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# EMBOSS pKa set, as used by Biopython's IsoelectricPoint.
PKA_SIDE_CHAIN = {"C": 8.5, "D": 3.9, "E": 4.1, "H": 6.5, "K": 10.8, "R": 12.5, "Y": 10.1}
POSITIVE = {"K", "R", "H"}
NEGATIVE = {"D", "E"}
PKA_N_TERM, PKA_C_TERM = 8.6, 3.6
HYDROPHOBIC = set("AVILMFWCY")
# Conservative substitution sets, used by the cheap mutational scan.
CHEMICAL_GROUPS = {
    "hydrophobic_small": "AVILM",
    "aromatic": "FWY",
    "polar": "STNQ",
    "acidic": "DE",
    "basic": "KRH",
    "special": "GPC",
}


def _counts(seq: str) -> Counter:
    return Counter(seq)


def length(seq: str) -> int:
    """Number of residues in the design."""
    return len(seq)


def sequence_identity(seq: str, parent: str) -> float:
    """Fraction of positions identical to the wild-type/parent sequence.

    Equal-length sequences are compared position-wise. Sequences of different
    length are not aligned here: that is a different metric and pretending
    otherwise would silently report nonsense.
    """
    if len(seq) != len(parent):
        raise ValueError(
            f"sequence_identity needs equal lengths, got {len(seq)} and {len(parent)}; "
            "align first or use a length-aware metric"
        )
    if not parent:
        raise ValueError("sequence_identity: empty parent")
    return sum(a == b for a, b in zip(seq, parent)) / len(parent)


def n_mutations(seq: str, parent: str) -> int:
    """How many positions differ from the parent."""
    return sum(a != b for a, b in zip(seq, parent)) if len(seq) == len(parent) else -1


def charge_at_ph(seq: str, ph: float = 7.0) -> float:
    """Net charge at the given pH, by Henderson-Hasselbalch over the ionisable groups.

    Delegates to Biopython's IsoelectricPoint (EMBOSS pKa set) rather than
    carrying a second implementation, so our numbers reproduce exactly in a
    tool a judge already trusts.
    """
    return float(IsoelectricPoint(seq).charge_at_pH(ph))


def isoelectric_point(seq: str) -> float:
    """pH at which the net charge is zero (theoretical pI). Biopython, EMBOSS pKa set."""
    return float(IsoelectricPoint(seq).pi())


def gravy(seq: str) -> float:
    """Mean Kyte-Doolittle hydropathy: positive is overall hydrophobic."""
    vals = [KD_HYDROPATHY[c] for c in seq if c in KD_HYDROPATHY]
    if not vals:
        raise ValueError("gravy: no canonical residues in sequence")
    return sum(vals) / len(vals)


def fraction_charged(seq: str) -> float:
    """Fraction of residues that are D, E, K or R."""
    counts = _counts(seq)
    return sum(counts.get(a, 0) for a in "DEKR") / len(seq)


def fraction_hydrophobic(seq: str) -> float:
    """Fraction of residues with hydrophobic side chains (AVILMFWCY)."""
    return sum(1 for c in seq if c in HYDROPHOBIC) / len(seq)


def max_homopolymer_run(seq: str) -> int:
    """Length of the longest run of one repeated residue (expression/synthesis risk)."""
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if seq else 0


def homopolymer_runs(seq: str, min_run: int = 4) -> list[tuple[int, str, int]]:
    """All runs of at least ``min_run`` identical residues, as (start_pos, residue, length)."""
    out = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            if i - start >= min_run:
                out.append((start + 1, seq[start], i - start))
            start = i
    return out


def low_complexity_fraction(seq: str, window: int = 10) -> float:
    """Highest single-residue fraction in any window: 1.0 means a pure repeat."""
    if len(seq) <= window:
        return max(_counts(seq).values()) / len(seq) if seq else 0.0
    worst = 0.0
    for i in range(len(seq) - window + 1):
        w = seq[i : i + window]
        worst = max(worst, max(_counts(w).values()) / window)
    return worst


def shannon_entropy(seq: str) -> float:
    """Residue-composition entropy in bits; low means a compositionally biased design."""
    import math

    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in _counts(seq).values())


def cysteine_count(seq: str) -> int:
    """Number of cysteines in the sequence."""
    return _counts(seq).get("C", 0)


def unpaired_cysteines(seq: str) -> int:
    """1 if the cysteine count is odd (a free thiol is a misfolding/aggregation risk), else 0.

    Sequence-only metric: it cannot see which cysteines actually form disulfides.
    The structure-aware count lives in score/structure.py.
    """
    return cysteine_count(seq) % 2


def catalytic_integrity(seq: str, catalytic: list[dict]) -> dict:
    """Are the expected catalytic residues still present at their positions?

    ``catalytic`` is the resolved list from the target config:
    ``[{pos, resn, role}]`` with ``pos`` a 1-indexed sequence position.
    Returns the fraction intact, the count, and which ones were lost — a design
    that loses a catalytic residue is dead regardless of every other metric.
    """
    if not catalytic:
        return {"catalytic_integrity": float("nan"), "n_catalytic": 0,
                "catalytic_lost": "", "catalytic_observed": ""}
    lost, observed = [], []
    for c in catalytic:
        pos, want3 = int(c["pos"]), str(c["resn"]).upper()
        if pos > len(seq):
            lost.append(f"{want3}{pos}>len")
            observed.append("-")
            continue
        have1 = seq[pos - 1]
        observed.append(have1)
        if ONE_TO_THREE.get(have1) != want3:
            lost.append(f"{want3}{pos}{have1}")
    n = len(catalytic)
    return {
        "catalytic_integrity": (n - len(lost)) / n,
        "n_catalytic": n,
        "catalytic_lost": "+".join(lost),
        "catalytic_observed": "".join(observed),
    }


def composition(seq: str) -> dict:
    """Per-residue-type counts, as ``count_A`` ... ``count_Y``."""
    counts = _counts(seq)
    return {f"count_{aa}": counts.get(aa, 0) for aa in sorted(KD_HYDROPATHY)}


def all_sequence_metrics(seq: str, parent: str | None = None, catalytic: list[dict] | None = None,
                         ph: float = 7.0, window: int = 10) -> dict:
    """Every Tier-1 sequence metric for one design, as a flat dict."""
    row = {
        "length": length(seq),
        "net_charge_ph7": charge_at_ph(seq, ph),
        "isoelectric_point": isoelectric_point(seq),
        "gravy": gravy(seq),
        "fraction_charged": fraction_charged(seq),
        "fraction_hydrophobic": fraction_hydrophobic(seq),
        "max_homopolymer_run": max_homopolymer_run(seq),
        "low_complexity_fraction": low_complexity_fraction(seq, window),
        "shannon_entropy": shannon_entropy(seq),
        "cysteine_count": cysteine_count(seq),
        "unpaired_cys": unpaired_cysteines(seq),
    }
    if parent is not None and len(parent) == len(seq):
        row["sequence_identity"] = sequence_identity(seq, parent)
        row["n_mutations"] = n_mutations(seq, parent)
        row["unpaired_cys_delta"] = unpaired_cysteines(seq) - unpaired_cysteines(parent)
        row["net_charge_delta"] = row["net_charge_ph7"] - charge_at_ph(parent, ph)
    row.update(catalytic_integrity(seq, catalytic or []))
    return row

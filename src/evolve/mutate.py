"""The mutation operator: what changes in a child, and by how much.

Three things are sampled for every child: **how many** positions change, **where**
they are, and **what** they become.

**How many** follows an adaptive schedule. Mostly single mutations, with a tail,
and the tail widens automatically when the population stops improving. That is a
mutator phenotype, and it is the standard answer to a search stuck in a local
optimum. It switches on a measured trigger, never on a judgement call at 03:00.

**Where** is weighted, because positions are not equal:

* the catalytic base and H-bond donor are **never** mutable. A child that
  touches one is discarded, not repaired - the target config says which
  positions we may change, and a design that quietly changed others is not the
  thing we claim to have made.
* the second shell (anything near the ligand) gets most of the weight: it sets
  pocket shape, and therefore iPDE and geometry, which is what we are failing on.
* the core gets some: it sets whether Boltz folds the thing reproducibly, which
  is metric 4.
* the surface gets a little. Neutral drift is not useless - it is how real
  directed evolution escapes local optima.

**What** mixes three proposal distributions: conservative (BLOSUM62), model-
guided (LigandMPNN's own posterior, when we have it - it already knows what
belongs at that position given the ligand), and radical (uniform). Without the
radical tail the search cannot cross a chemical gap; with too much of it the
search is random. See BRIEF.md 4.2 for the weights and the reasoning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

log = logging.getLogger("pipeline.evolve.mutate")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Positions whose class we know. Weights are relative, not probabilities.
POSITION_CLASSES = ("second_shell", "core", "surface")


@dataclass(frozen=True)
class MutationPolicy:
    """How aggressive the operator is. One policy per generation, logged."""

    #: P(k mutations) for k = 1, 2, 3, 4, 5. Normalised on use.
    k_weights: tuple[float, ...] = (0.70, 0.22, 0.05, 0.02, 0.01)
    #: P(conservative), P(model-guided), P(radical). Normalised on use.
    mode_weights: tuple[float, float, float] = (0.55, 0.30, 0.15)
    #: Relative sampling weight per position class.
    class_weights: dict[str, float] = field(
        default_factory=lambda: {"second_shell": 0.60, "core": 0.30, "surface": 0.10}
    )
    #: Never introduce a new cysteine: unpaired Cys is a liability we cannot see.
    exclude_cysteine: bool = True

    def stagnant(self) -> "MutationPolicy":
        """A widened policy, for when the elites have stopped improving.

        More multiple mutations and more radical substitutions. BRIEF.md 4.2
        gives the trigger: no elite improvement for two generations.
        """
        return replace(
            self,
            k_weights=(0.40, 0.35, 0.15, 0.07, 0.03),
            mode_weights=(0.40, 0.30, 0.30),
        )


@dataclass(frozen=True)
class Mutation:
    """One substitution, recorded so a design's lineage can be reconstructed."""

    position: int  # 1-based
    from_aa: str
    to_aa: str
    mode: str

    def __str__(self) -> str:
        return f"{self.from_aa}{self.position}{self.to_aa}"


def _normalise(weights) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if np.any(array < 0):
        raise ValueError("weights must be non-negative")
    total = array.sum()
    if total <= 0:
        raise ValueError("weights must not sum to zero")
    return array / total


def blosum_row(aa: str) -> dict[str, float]:
    """P(substitution | parent residue) from BLOSUM62, as a usable distribution.

    BLOSUM scores are log-odds, so we exponentiate to get something samplable,
    drop the identity substitution (a child identical to its parent is not a
    child), and renormalise. Biotite ships the matrix, so we do not embed a copy
    that could drift from the standard one.
    """
    from biotite.sequence.align import SubstitutionMatrix

    matrix = SubstitutionMatrix.std_protein_matrix()
    alphabet = matrix.get_alphabet1()
    scores = {}
    for other in AMINO_ACIDS:
        if other == aa:
            continue
        try:
            code_a = alphabet.encode(aa)
            code_b = alphabet.encode(other)
        except KeyError:  # pragma: no cover - non-standard residue
            continue
        scores[other] = float(matrix.get_score_by_code(code_a, code_b))
    if not scores:
        raise ValueError(f"no BLOSUM substitutions available for {aa!r}")
    values = np.array(list(scores.values()), dtype=float)
    # Temperature 1.0 on the log-odds: conservative swaps dominate but the
    # tail stays reachable.
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return dict(zip(scores.keys(), probabilities))


def classify_positions(
    length: int,
    second_shell: set[int],
    core: set[int] | None = None,
) -> dict[int, str]:
    """Label every 1-based position as second_shell / core / surface.

    ``second_shell`` normally comes from :func:`shell_positions_from_structure`.
    Anything not in ``second_shell`` or ``core`` is treated as surface, which is
    the conservative assumption: it gets the least weight.
    """
    core = core or set()
    labels = {}
    for position in range(1, length + 1):
        if position in second_shell:
            labels[position] = "second_shell"
        elif position in core:
            labels[position] = "core"
        else:
            labels[position] = "surface"
    return labels


def shell_positions_from_structure(atoms, ligand_resname: str = "6NT", radius: float = 10.0) -> set[int]:
    """1-based positions with any heavy atom within ``radius`` of the ligand.

    These are the positions that actually shape the pocket, and therefore the
    ones most likely to move iPDE and geometry - which is what we are failing
    on (BRIEF.md 1). Positions are indices into the protein CA array, matching
    :mod:`src.score.official`.
    """
    import biotite.structure as struc  # noqa: F401  (import kept local, biotite is heavy)

    ligand = atoms[(atoms.res_name == ligand_resname) & (atoms.element != "H")]
    if ligand.array_length() == 0:
        raise ValueError(f"no {ligand_resname} ligand in the structure")
    protein = atoms[(~atoms.hetero) & (atoms.element != "H")]
    ca = atoms[(~atoms.hetero) & (atoms.atom_name == "CA")]
    if ca.array_length() == 0:
        raise ValueError("structure has no protein CA atoms")
    # map (chain, res_id, ins_code) -> 1-based position, in CA file order
    index = {
        (ca.chain_id[i], ca.res_id[i], ca.ins_code[i]): i + 1
        for i in range(ca.array_length())
    }
    distances = np.linalg.norm(
        protein.coord[:, None] - ligand.coord[None, :], axis=-1
    ).min(axis=1)
    near = distances <= radius
    positions = set()
    for i in np.flatnonzero(near):
        key = (protein.chain_id[i], protein.res_id[i], protein.ins_code[i])
        if key in index:
            positions.add(index[key])
    return positions


def sample_count(policy: MutationPolicy, rng: np.random.Generator) -> int:
    weights = _normalise(policy.k_weights)
    return int(rng.choice(np.arange(1, len(weights) + 1), p=weights))


def sample_positions(
    count: int,
    labels: dict[int, str],
    policy: MutationPolicy,
    rng: np.random.Generator,
    excluded: set[int],
) -> list[int]:
    """Pick ``count`` distinct mutable positions, weighted by class."""
    candidates = [p for p in sorted(labels) if p not in excluded]
    if not candidates:
        raise ValueError("no mutable positions: every position is excluded")

    # class_weights is a BUDGET PER CLASS, not a weight per position. The
    # surface is ~90% of a protein, so weighting each position equally by class
    # would hand the surface most of the probability mass and defeat the point
    # of targeting the second shell. Divide each class's share among its members.
    members: dict[str, int] = {}
    for position in candidates:
        members[labels[position]] = members.get(labels[position], 0) + 1
    weights = np.array(
        [
            policy.class_weights.get(labels[p], 0.0) / members[labels[p]]
            for p in candidates
        ],
        dtype=float,
    )
    if weights.sum() <= 0:
        weights = np.ones(len(candidates), dtype=float)
    weights = weights / weights.sum()
    count = min(count, len(candidates))
    chosen = rng.choice(len(candidates), size=count, replace=False, p=weights)
    return [candidates[i] for i in sorted(chosen)]


def sample_substitution(
    parent_aa: str,
    position: int,
    policy: MutationPolicy,
    rng: np.random.Generator,
    posterior: dict[int, dict[str, float]] | None = None,
) -> tuple[str, str]:
    """Return (new residue, which mode produced it)."""
    modes = ("conservative", "model", "radical")
    weights = list(_normalise(policy.mode_weights))
    has_posterior = bool(posterior and position in posterior)
    if not has_posterior:
        # Fold the model-guided share back into conservative rather than
        # silently sampling from a distribution we do not have.
        weights[0] += weights[1]
        weights[1] = 0.0
    mode = str(rng.choice(modes, p=_normalise(weights)))

    if mode == "model":
        distribution = dict(posterior[position])
    elif mode == "conservative":
        distribution = blosum_row(parent_aa)
    else:
        distribution = {aa: 1.0 for aa in AMINO_ACIDS}

    distribution.pop(parent_aa, None)
    if policy.exclude_cysteine and parent_aa != "C":
        distribution.pop("C", None)
    if not distribution:
        raise ValueError(f"no substitution available at position {position}")
    letters = list(distribution)
    probabilities = _normalise([distribution[a] for a in letters])
    return str(rng.choice(letters, p=probabilities)), mode


def mutate(
    sequence: str,
    labels: dict[int, str],
    policy: MutationPolicy,
    rng: np.random.Generator,
    excluded: set[int],
    posterior: dict[int, dict[str, float]] | None = None,
) -> tuple[str, list[Mutation]]:
    """One child: ``sequence`` with k sampled point mutations.

    ``excluded`` must contain the catalytic positions. This function will not
    mutate them, and :func:`assert_catalytic_intact` re-checks afterwards -
    belt and braces, because a silent change there invalidates every number we
    report about the design.
    """
    if not sequence:
        raise ValueError("cannot mutate an empty sequence")
    count = sample_count(policy, rng)
    positions = sample_positions(count, labels, policy, rng, excluded)
    letters = list(sequence)
    mutations = []
    for position in positions:
        parent_aa = letters[position - 1]
        if parent_aa not in AMINO_ACIDS:
            log.warning("position %d is %r, not a standard residue; skipped", position, parent_aa)
            continue
        new_aa, mode = sample_substitution(parent_aa, position, policy, rng, posterior)
        letters[position - 1] = new_aa
        mutations.append(Mutation(position, parent_aa, new_aa, mode))
    return "".join(letters), mutations


def assert_catalytic_intact(child: str, parent: str, catalytic: dict[int, str]) -> None:
    """Raise unless every catalytic position still holds its required residue.

    ``catalytic`` maps 1-based position -> expected one-letter code.
    """
    if len(child) != len(parent):
        raise ValueError(f"child length {len(child)} != parent length {len(parent)}")
    for position, expected in catalytic.items():
        if not 1 <= position <= len(child):
            raise ValueError(f"catalytic position {position} outside the sequence")
        actual = child[position - 1]
        if actual != expected:
            raise ValueError(
                f"catalytic position {position} is {actual!r}, must stay {expected!r}"
            )


def crossover(
    parent_a: str, parent_b: str, rng: np.random.Generator, excluded: set[int]
) -> str:
    """Single-point recombination between two lineages.

    The classic way to combine independently discovered improvements, and the
    direct analogue of DNA shuffling. Only useful once two distinct lineages
    exist, so the driver turns it on after the population has split.
    """
    if len(parent_a) != len(parent_b):
        raise ValueError("crossover needs equal-length parents")
    if len(parent_a) < 2:
        raise ValueError("sequence too short to recombine")
    point = int(rng.integers(1, len(parent_a)))
    child = parent_a[:point] + parent_b[point:]
    # Catalytic positions must come from whichever parent is authoritative;
    # both parents agree on them by construction, so just assert it.
    for position in excluded:
        if 1 <= position <= len(child) and parent_a[position - 1] != parent_b[position - 1]:
            raise ValueError(
                f"parents disagree at protected position {position}; refusing to recombine"
            )
    return child

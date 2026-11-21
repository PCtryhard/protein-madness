"""The mutation branch: single mutants of a parent sequence, straight to Boltz-2.

The branch skips RFdiffusion and LigandMPNN entirely. That skip is sound
because a point mutation needs no new backbone, and Boltz-2 predicts the whole
complex from sequence, so it repacks the neighbourhood implicitly.

Two constraints from the hardware, both in config rather than in code here:
the mutable set is restricted to a shell around the ligand, and the alphabet is
reduced. Full saturation of 12 positions is 228 mutants, which is several hours
for one cycle of one scaffold on two 12 GB cards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..config import MotifCfg, MutateCfg
from ..tracking import Provenance, Structure


class MutationError(RuntimeError):
    """The mutable set could not be determined."""


@dataclass(frozen=True)
class Mutant:
    """One single-point mutant of a parent sequence."""

    position: int          # 1-based position in the parent sequence
    wild_type: str         # one-letter
    mutant: str            # one-letter
    sequence: str

    @property
    def label(self) -> str:
        """e.g. A72V -- the form recorded in the `mutation` database column."""
        return f"{self.wild_type}{self.position}{self.mutant}"


def shell_positions(
    structure: Structure,
    provenance: Provenance,
    motif: MotifCfg,
    cfg: MutateCfg,
    chain: str,
) -> list[int]:
    """Sequence positions within `shell_radius` of any ligand atom.

    Excludes every residue in `motif.fixed`: those are never mutated and never
    redesigned, and a mutation branch that touched them could destroy the very
    hydrogen bonds the score rewards.

    Returns 1-based positions in the Boltz-2 numbering of `chain`, which is the
    numbering the sequence string uses.
    """
    if not structure.ligand:
        raise MutationError(
            f"{structure.source}: no ligand atoms, so the shell around the ligand "
            "is undefined. The mutation branch cannot choose positions."
        )

    ligand_xyz = np.array(list(structure.ligand.values()), dtype=float)

    # The motif residues to exclude, in the numbering this structure uses.
    excluded: set[int] = set()
    for chain_id, resnum in motif.as_keys():
        try:
            mapped_chain, mapped_num = provenance.seed_residue_in_boltz(chain_id, resnum)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise MutationError(
                f"motif residue {chain_id}{resnum} could not be mapped into the "
                f"predicted structure, so the mutation branch cannot exclude it: {exc}"
            ) from exc
        if mapped_chain == chain:
            excluded.add(mapped_num)

    within: list[int] = []
    for resnum in structure.residue_numbers(chain):
        if resnum in excluded:
            continue
        side_chain = [
            xyz
            for (c, r, name), xyz in structure.atoms.items()
            if c == chain and r == resnum and not _is_hydrogen(name)
        ]
        if not side_chain:
            continue
        coords = np.array(side_chain, dtype=float)
        # Heavy-atom distance from any atom of the residue to any ligand atom.
        dmin = float(
            np.sqrt(
                ((coords[:, None, :] - ligand_xyz[None, :, :]) ** 2).sum(axis=2)
            ).min()
        )
        if dmin <= cfg.shell_radius:
            within.append(resnum)

    if not within:
        raise MutationError(
            f"no mutable residue within {cfg.shell_radius} A of the ligand in "
            f"{structure.source} after excluding the motif. Either the ligand "
            "drifted out of the pocket or shell_radius is too small."
        )
    return sorted(within)


def _is_hydrogen(atom_name: str) -> bool:
    stripped = atom_name.strip()
    for ch in stripped:
        if ch.isalpha():
            return ch.upper() == "H"
    return False


def enumerate_mutants(
    sequence: str,
    positions: list[int],
    cfg: MutateCfg,
    rng: random.Random,
) -> list[Mutant]:
    """All single mutants at `positions` over the configured alphabet, sampled.

    If the count exceeds `max_mutants_per_parent`, sample without replacement
    rather than truncating in sequence order -- truncation would bias towards
    whichever end of the chain comes first.
    """
    candidates: list[Mutant] = []
    for position in positions:
        if not 1 <= position <= len(sequence):
            raise MutationError(
                f"position {position} is outside the parent sequence "
                f"(length {len(sequence)})"
            )
        wild_type = sequence[position - 1]
        for mutant in cfg.alphabet:
            if mutant == wild_type:
                continue
            mutated = sequence[: position - 1] + mutant + sequence[position:]
            candidates.append(
                Mutant(
                    position=position,
                    wild_type=wild_type,
                    mutant=mutant,
                    sequence=mutated,
                )
            )

    if not candidates:
        raise MutationError(
            "no mutants generated: every mutable position already holds the only "
            f"residue the alphabet {cfg.alphabet!r} offers"
        )

    if len(candidates) <= cfg.max_mutants_per_parent:
        return candidates
    return rng.sample(candidates, cfg.max_mutants_per_parent)


def mutants_for_parent(
    sequence: str,
    structure: Structure,
    provenance: Provenance,
    motif: MotifCfg,
    cfg: MutateCfg,
    chain: str,
    rng: random.Random,
) -> list[Mutant]:
    """The whole branch for one parent: shell, then enumerate, then sample."""
    positions = shell_positions(structure, provenance, motif, cfg, chain)
    expected = structure.sequence(chain)
    if len(expected) != len(sequence):
        raise MutationError(
            f"parent sequence is {len(sequence)} residues but chain {chain} of "
            f"{structure.source} has {len(expected)}. The shell positions would "
            "point at the wrong residues."
        )
    return enumerate_mutants(sequence, positions, cfg, rng)

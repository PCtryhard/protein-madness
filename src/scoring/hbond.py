"""Hydrogen-bond geometry: the distance and angle terms, per named bond.

Two things the brief corrects, and which this module implements accordingly:

1. The distance is always heavy-atom donor to acceptor. Unaffected by
   protonation, so it is the one term that never depends on tooling.

2. The angle depends on the rule in `scoring.angle.rule`.

   `acceptor_side_always` (the conservative default) measures directionality at
   the acceptor, D_ant(acceptor side)-A...D, against that bond's own
   `acceptor_ideal_angle`. It needs no hydrogens and cannot be gamed by the
   protonator.

   `split_by_donor_type` measures D-H...A against 180 deg for donors whose
   hydrogen position is determined by chemistry, and falls back to the
   acceptor-side angle for rotatable donors, whose hydrogen the protonator
   places to optimise hydrogen bonding -- measuring D-H...A there would partly
   measure what PDBFixer decided rather than what Boltz-2 predicted.

Applying a 180 deg target to heavy atoms -- which is what happens if hydrogens
are simply skipped -- penalises good geometry and rewards a nearly linear
antecedent arrangement that is physically wrong. That is the single easiest way
to get a plausible-looking pipeline that optimises in the wrong direction, so
nothing in this module has a default angle target: every value comes from
config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AngleCfg, AngleRule, AtomSpec, DistanceCfg, HBond
from ..tracking import AtomRef, Provenance, Structure

LINEAR_DH_A = 180.0  # the geometric definition of the D-H...A ideal, not a tunable


class HBondGeometryError(RuntimeError):
    """The geometry of a named bond could not be measured."""


@dataclass(frozen=True)
class BondMeasurement:
    """What was measured for one named bond, before normalisation."""

    bond_id: str
    distance: float
    angle: float
    angle_kind: str           # "donor_h_acceptor" | "acceptor_side"
    angle_ideal: float
    required: bool
    atoms: dict[str, str]     # role -> resolved atom label, for the log
    # Which geometry weight the distance term falls under. Defaults True so a
    # caller that builds a measurement by hand keeps the old behaviour of
    # everything counting as catalytic.
    catalytic: bool = True

    def as_log_dict(self) -> dict:
        return {
            "bond_id": self.bond_id,
            "distance": self.distance,
            "angle": self.angle,
            "angle_kind": self.angle_kind,
            "angle_ideal": self.angle_ideal,
            "required": self.required,
            "catalytic": self.catalytic,
            "atoms": self.atoms,
        }


def angle_between(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Angle a-vertex-c in degrees."""
    v1 = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(vertex, dtype=float)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        raise HBondGeometryError(
            "cannot measure an angle: two of the three atoms are coincident"
        )
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _ref(spec: AtomSpec) -> AtomRef:
    return AtomRef.from_spec(spec)


def find_donor_hydrogen(
    structure: Structure,
    donor_xyz: np.ndarray,
    acceptor_xyz: np.ndarray,
    max_bond_length: float = 1.3,
) -> np.ndarray:
    """The hydrogen on the donor heavy atom, for the D-H...A angle.

    Picks the hydrogen covalently bound to the donor (within `max_bond_length`)
    whose D-H...A angle is largest, i.e. the one actually pointing at the
    acceptor. On a donor with several equivalent hydrogens -- an Arg NH2, a Lys
    ammonium -- that is the one forming the bond, which is the one we mean.

    Raises if the donor carries no hydrogen at all. That is not a geometry of
    zero quality, it is an unmeasurable quantity: the protonation stage either
    did not run or did not protonate this atom.
    """
    candidates = []
    for (_, _, name), xyz in structure.atoms.items():
        if not _is_hydrogen(name):
            continue
        if float(np.linalg.norm(xyz - donor_xyz)) <= max_bond_length:
            candidates.append(xyz)

    if not candidates:
        raise HBondGeometryError(
            "no hydrogen found within "
            f"{max_bond_length} A of the donor heavy atom. The D-H...A angle is "
            "unmeasurable. Either the protonation stage did not run, or this "
            "donor was not protonated; with scoring.angle.rule="
            "'acceptor_side_always' no hydrogens are needed at all."
        )
    return max(candidates, key=lambda h: angle_between(donor_xyz, h, acceptor_xyz))


def _is_hydrogen(atom_name: str) -> bool:
    # PDB atom names put the element in columns 13-14, so a hydrogen may be
    # written " HA ", "1HB " or "HD21". Anything whose first alphabetic
    # character is H and which is not a known heavy-atom name.
    stripped = atom_name.strip()
    if not stripped:
        return False
    for ch in stripped:
        if ch.isalpha():
            return ch.upper() == "H"
    return False


def measure_bond(
    bond: HBond,
    structure: Structure,
    provenance: Provenance,
    rule: AngleRule,
) -> BondMeasurement:
    """Resolve one named bond's atoms and measure its distance and angle.

    Every atom goes through `Provenance.resolve`, which raises on a miss. There
    is no proximity-based hydrogen-bond detection anywhere: the spec names the
    atoms, and we find those exact atoms or fail.
    """
    donor = provenance.resolve(_ref(bond.donor), structure)
    acceptor = provenance.resolve(_ref(bond.acceptor), structure)

    distance = float(np.linalg.norm(donor - acceptor))
    atoms = {
        "donor": str(_ref(bond.donor)),
        "acceptor": str(_ref(bond.acceptor)),
    }

    if bond.angle_is_acceptor_side(rule):
        if bond.acceptor_antecedent is None or bond.acceptor_ideal_angle is None:
            # config.load() validates this up front; reaching here means the
            # bond was built in code rather than loaded.
            raise HBondGeometryError(
                f"bond {bond.id!r} is measured at the acceptor but has no "
                "acceptor_antecedent or no acceptor_ideal_angle"
            )
        antecedent = provenance.resolve(_ref(bond.acceptor_antecedent), structure)
        angle = angle_between(antecedent, acceptor, donor)
        atoms["acceptor_antecedent"] = str(_ref(bond.acceptor_antecedent))
        return BondMeasurement(
            bond_id=bond.id,
            distance=distance,
            angle=angle,
            angle_kind="acceptor_side",
            angle_ideal=float(bond.acceptor_ideal_angle),
            required=bond.required,
            catalytic=bond.is_catalytic,
            atoms=atoms,
        )

    hydrogen = find_donor_hydrogen(structure, donor, acceptor)
    angle = angle_between(donor, hydrogen, acceptor)
    atoms["hydrogen"] = "placed by the protonation stage"
    return BondMeasurement(
        bond_id=bond.id,
        distance=distance,
        angle=angle,
        angle_kind="donor_h_acceptor",
        angle_ideal=LINEAR_DH_A,
        required=bond.required,
        catalytic=bond.is_catalytic,
        atoms=atoms,
    )


def measure_all(
    bonds: list[HBond],
    structure: Structure,
    provenance: Provenance,
    rule: AngleRule,
) -> list[BondMeasurement]:
    """Measure every named bond, or raise.

    Deliberately not wrapped in a try/except per bond: a design with one
    unmeasurable bond is an errored design, not a design with a partial score.
    """
    return [measure_bond(b, structure, provenance, rule) for b in bonds]


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def distance_term(distance: float, cfg: DistanceCfg) -> float:
    """Per-bond distance score, 1 best.

        d = clip(1 - |r - ideal| / (tol - ideal), 0, 1)

    Symmetric about the ideal, so a 2.4 A clash is penalised as well as a 3.4 A
    stretch.
    """
    span = cfg.tolerated - cfg.ideal
    return max(0.0, min(1.0, 1.0 - abs(distance - cfg.ideal) / span))


def angle_term(angle: float, ideal: float, cfg: AngleCfg) -> float:
    """Per-bond angle score, 1 best.

        a = clip(1 - |theta - theta_ideal| / theta_tol, 0, 1)

    `ideal` is the bond's own target: 180 deg for a measured D-H...A, or the
    bond's `acceptor_ideal_angle` when measured at the acceptor.
    """
    return max(0.0, min(1.0, 1.0 - abs(angle - ideal) / cfg.tolerance))


def is_gated(measurements: list[BondMeasurement], cfg: DistanceCfg) -> list[str]:
    """Ids of required bonds that are impossible: too long, or a clash.

    Any one of them zeroes the design's total. The design is recorded as gated
    rather than deleted, so the search can see what it rejected.

    The spec names only the long-distance gate (`distance.fail`). The short
    end was added because the gate was asymmetric in a way that mattered: the
    distance TERM is symmetric about the ideal, so a 1.7 A contact scores 0
    exactly like a 4.0 A one, but only the long one gated. A design with an
    atom driven through the ligand could therefore sit in the frontier on its
    confidence term alone. Three of the seven shipped targets start from motif
    files with sub-2 A modelled contacts, so this is not hypothetical.
    `distance.clash` is the threshold; set it very small to recover the
    spec's one-sided behaviour.
    """
    return [
        m.bond_id
        for m in measurements
        if m.required and (m.distance > cfg.fail or m.distance < cfg.clash)
    ]

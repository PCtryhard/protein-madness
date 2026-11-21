"""The four metrics the judges actually score, re-implemented exactly.

The Perlin/D-BSSE scoring slide names four things and nothing else:

1. **iPDE**     - Boltz-2's estimated error on protein<->ligand token distances.
                  Is the ligand confidently placed?
2. **Geometry** - distances/angles vs. the ideal approach geometry for the
                  catalytic base and the H-bond donor.
3. **Burial**   - how many CA atoms sit within 12 A of the catalytic base.
                  A solvent-exposed base loses the pKa that makes it a base.
4. **Consistency** - how well 1-3 reproduce across 5 Boltz-2 seeds. A proxy for
                  preorganisation, and our only defence against ranking noise.

Why this module exists, and why it is the first thing in the repo to get built:
we submit *sequences*. The organisers re-fold them themselves with Boltz-2 and
score the result. So our own ranking is only worth something if it reproduces
theirs. Everything downstream - selection, the evolution loop, the leaderboard,
the pitch - is built on these numbers being right.

**The atom names are not ours to choose.** The organisers' own
``example_e2e_kemp/src/verify_boltz.py`` measures exactly two contacts:

    ASP(base)  OD2 -> 6NT N3
    GLN(donor) NE2 -> 6NT N1

We measure the same two, with the same atom names, from the same file. Where
their script makes a convention choice we copy the choice rather than improve on
it; a better metric that disagrees with the judge is worse than a faithful one.

**Residue indexing follows theirs exactly.** ``position`` is a 1-based index
into the protein's CA atoms in file order - ``ca[position - 1]`` - not a PDB
residue id. That is what their bridge file records and what their verifier uses.
Three numbering systems are in play in this project (HG3.17 reference, RFD3
output, Boltz output) and this is the one that matters at scoring time.

Missing inputs give ``nan`` and a note, never 0.0: a plausible-looking number is
worse than no number, because a filter will act on it.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

log = logging.getLogger("pipeline.score.official")

#: Radius for the catalytic-base burial count, from the scoring slide.
BURIAL_RADIUS_A = 12.0
#: Heavy-atom count of 6-nitrobenzotriazole; a sanity check on the ligand.
LIGAND_HEAVY_ATOMS = 12
#: Pairs closer than this are clashes. Matches the organisers' verifier.
CLASH_DISTANCE_A = 2.2


@dataclass(frozen=True)
class CatalyticSite:
    """Where the catalytic machinery is, in *Boltz output* numbering.

    ``base_position`` and ``donor_position`` are 1-based indices into the
    protein CA atoms in file order, which for a design we folded ourselves is
    simply the position in the sequence we submitted.

    The atom names default to the Kemp eliminase case and are overridable so
    this module is not welded to one challenge.
    """

    base_position: int
    donor_position: int
    base_atom: str = "OD2"
    base_ligand_atom: str = "N3"
    donor_atom: str = "NE2"
    donor_ligand_atom: str = "N1"
    ligand_resname: str = "6NT"
    #: Third atom for the approach ANGLE at each contact: the atom the
    #: contacting atom hangs off. For the aspartate carboxylate that is CG, for
    #: the glutamine amide it is CD. The angle CG-OD2...N3 says whether the
    #: carboxylate is *pointing at* the ligand or merely near it -- two designs
    #: can share a 3 A distance and only one of them be productive.
    base_root_atom: str = "CG"
    donor_root_atom: str = "CD"

    def __post_init__(self) -> None:
        for name in ("base_position", "donor_position"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a 1-based integer position, got {value!r}")


@dataclass
class OfficialScore:
    """One Boltz-2 prediction, scored. One row per seed."""

    source: str = ""
    seed: int | None = None
    n_residues: int = 0

    # metric 1
    ipde: float = math.nan
    # supporting confidence values, cheap to carry and useful for triage
    iptm: float = math.nan
    ligand_iptm: float = math.nan
    complex_plddt: float = math.nan
    complex_iplddt: float = math.nan
    confidence_score: float = math.nan

    # metric 2 - distances first, they carry most of the signal
    base_to_ligand_a: float = math.nan
    donor_to_ligand_a: float = math.nan
    # metric 2 - approach angles. The scoring slide says the geometry term is
    # "an arrangement of distance, angle and dihedral measurements", so a
    # distance-only score is an incomplete copy of it.
    base_angle_deg: float = math.nan
    donor_angle_deg: float = math.nan

    # "how much did it change" - RMSD between the structure we designed and the
    # structure Boltz predicts from its sequence. A design the model reproduces
    # is a design whose active site survives, which is the whole problem.
    design_rmsd_a: float = math.nan

    # metric 3
    base_burial_ca_count: int = -1

    # pocket sanity, not scored directly but explains failures
    min_ligand_protein_a: float = math.nan
    clash_pairs: int = -1

    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        log.warning("%s: %s", self.source or "prediction", message)
        self.notes.append(message)

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------- loading

def read_confidence(path: Path) -> dict:
    """Boltz-2's ``confidence_*.json``. Raises if absent or unparseable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no Boltz confidence file at {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def read_complex(path: Path) -> struc.AtomArray:
    """Read a Boltz-2 mmCIF prediction, first model, author numbering.

    Deliberately *not* passed through ``src.io.structure.normalise``: that
    function drops heteroatoms, and the ligand is the entire point here.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no structure at {path}")
    with path.open() as stream:
        atoms = pdbx.get_structure(pdbx.CIFFile.read(stream), model=1, use_author_fields=True)
    if not np.isfinite(atoms.coord).all():
        raise ValueError(f"{path} contains non-finite coordinates")
    return atoms


def find_prediction(run_dir: Path) -> tuple[Path, Path]:
    """Locate (structure, confidence) inside a Boltz ``--out_dir``.

    Boltz nests output as ``boltz_results_<stem>/predictions/<stem>/``. We search
    rather than hardcode the stem, and raise if the pair is ambiguous so a
    half-written run cannot be scored by accident.
    """
    run_dir = Path(run_dir)
    structures = sorted(p for p in run_dir.rglob("*_model_0.cif"))
    if not structures:
        structures = sorted(p for p in run_dir.rglob("*.cif") if "model" in p.name)
    confidences = sorted(run_dir.rglob("confidence_*_model_0.json"))
    if not confidences:
        confidences = sorted(run_dir.rglob("confidence_*.json"))
    if len(structures) != 1 or len(confidences) != 1:
        raise ValueError(
            f"expected exactly one prediction under {run_dir}, found "
            f"{len(structures)} structures and {len(confidences)} confidence files"
        )
    return structures[0], confidences[0]


# ------------------------------------------------------------------- metrics

def _protein_ca(atoms: struc.AtomArray) -> struc.AtomArray:
    ca = atoms[(~atoms.hetero) & (atoms.atom_name == "CA")]
    if ca.array_length() == 0:
        raise ValueError("structure contains no protein CA atoms")
    return ca


def _ligand(atoms: struc.AtomArray, resname: str) -> struc.AtomArray:
    lig = atoms[(atoms.res_name == resname) & (atoms.element != "H")]
    return lig


def contact_distance(
    atoms: struc.AtomArray,
    position: int,
    protein_atom: str,
    ligand: struc.AtomArray,
    ligand_atom: str,
) -> float:
    """Distance from one named side-chain atom to one named ligand atom.

    ``position`` indexes the protein CA array, 1-based, exactly as the
    organisers' verifier does. Returns ``nan`` when either atom is missing or
    ambiguous rather than guessing a substitute atom.
    """
    ca = _protein_ca(atoms)
    if not 1 <= position <= ca.array_length():
        raise IndexError(
            f"position {position} is outside the predicted structure "
            f"(1..{ca.array_length()})"
        )
    residue = ca[position - 1]
    selected = atoms[
        (~atoms.hetero)
        & (atoms.chain_id == residue.chain_id)
        & (atoms.res_id == residue.res_id)
        & (atoms.ins_code == residue.ins_code)
        & (atoms.atom_name == protein_atom)
    ]
    point = ligand[ligand.atom_name == ligand_atom]
    if selected.array_length() != 1 or point.array_length() != 1:
        return math.nan
    return float(np.linalg.norm(selected.coord[0] - point.coord[0]))


def _residue_atom(atoms: struc.AtomArray, position: int, atom_name: str):
    """One named atom of the residue at a 1-based CA-array position, or None."""
    ca = _protein_ca(atoms)
    if not 1 <= position <= ca.array_length():
        raise IndexError(
            f"position {position} is outside the predicted structure (1..{ca.array_length()})"
        )
    residue = ca[position - 1]
    selected = atoms[
        (~atoms.hetero)
        & (atoms.chain_id == residue.chain_id)
        & (atoms.res_id == residue.res_id)
        & (atoms.ins_code == residue.ins_code)
        & (atoms.atom_name == atom_name)
    ]
    return selected.coord[0] if selected.array_length() == 1 else None


def approach_angle(
    atoms: struc.AtomArray,
    position: int,
    root_atom: str,
    contact_atom: str,
    ligand: struc.AtomArray,
    ligand_atom: str,
) -> float:
    """Angle root-contact...ligand, in degrees. NaN if any atom is missing.

    Distance alone cannot tell a productive contact from an accidental one: a
    carboxylate oxygen 3 A from the ligand but pointing away is not a catalytic
    base. This is the cheap second half of the geometry term.
    """
    root = _residue_atom(atoms, position, root_atom)
    contact = _residue_atom(atoms, position, contact_atom)
    point = ligand[ligand.atom_name == ligand_atom]
    if root is None or contact is None or point.array_length() != 1:
        return math.nan
    a = root - contact
    b = point.coord[0] - contact
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return math.nan
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def backbone_rmsd(predicted: struc.AtomArray, reference: struc.AtomArray) -> float:
    """CA-RMSD after optimal superposition: how much the fold *changed*.

    The organisers compute this themselves (``backbone_ca_rmsd_from_rfd3`` in
    their summary: 2.16 A for the reference design). Low change means Boltz
    agrees with the structure we designed, which is the thing that decides
    whether the active site survives re-prediction.
    """
    p = _protein_ca(predicted).coord
    r = _protein_ca(reference).coord
    if p.shape != r.shape or p.size == 0:
        return math.nan
    if not (np.isfinite(p).all() and np.isfinite(r).all()):
        return math.nan
    pc, rc = p - p.mean(axis=0), r - r.mean(axis=0)
    u, _s, vt = np.linalg.svd(pc.T @ rc)
    rotation = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    aligned = pc @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - rc) ** 2, axis=1))))


def base_burial(atoms: struc.AtomArray, base_position: int, radius: float = BURIAL_RADIUS_A) -> int:
    """Number of CA atoms within ``radius`` of the catalytic base's CA.

    The scoring slide says "number of CA atoms close to the catalytic base (up
    to 12 A)". It does not say *which* base atom the distance is measured from.
    We use the base residue's own CA, because it is the one atom every residue
    has and the one choice that cannot depend on side-chain rotamer. The base's
    own CA is excluded from the count.

    If the organisers turn out to measure from the carboxylate instead, this is
    a one-line change - but the *ranking* is near-identical either way, since
    the two reference points are under 3 A apart and the shell is 12 A.
    """
    ca = _protein_ca(atoms)
    if not 1 <= base_position <= ca.array_length():
        raise IndexError(
            f"base_position {base_position} is outside the predicted structure "
            f"(1..{ca.array_length()})"
        )
    origin = ca.coord[base_position - 1]
    distances = np.linalg.norm(ca.coord - origin, axis=-1)
    return int(np.sum(distances <= radius) - 1)


def ligand_packing(atoms: struc.AtomArray, ligand: struc.AtomArray) -> tuple[float, int]:
    """(minimum protein-ligand heavy-atom distance, count of clashing pairs).

    Not one of the four scored metrics, but it is how a failure explains itself:
    a strained pose with sub-2.2 A contacts is one the model will relieve by
    pushing the ligand out of the pocket, which is what destroys the geometry
    term. Cheap to carry and worth having in the leaderboard.
    """
    heavy = atoms[(~atoms.hetero) & (atoms.element != "H")]
    if heavy.array_length() == 0 or ligand.array_length() == 0:
        return math.nan, -1
    distances = np.linalg.norm(heavy.coord[:, None] - ligand.coord[None, :], axis=-1)
    if not np.isfinite(distances).all():
        return math.nan, -1
    return float(distances.min()), int(np.sum(distances < CLASH_DISTANCE_A))


# ------------------------------------------------------------------- scoring

def _confidence_value(conf: dict, key: str, score: OfficialScore) -> float:
    value = conf.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        score.note(f"confidence field {key!r} missing or non-finite")
        return math.nan
    return float(value)


def score_prediction(
    structure_path: Path,
    confidence_path: Path,
    site: CatalyticSite,
    *,
    seed: int | None = None,
    reference_structure: Path | None = None,
) -> OfficialScore:
    """Score one Boltz-2 prediction against the four official metrics.

    ``reference_structure`` is the structure we *designed* (the RFD3 backbone).
    Supplying it adds ``design_rmsd_a`` - how much the fold changed when the
    model rebuilt it from sequence alone.
    """
    score = OfficialScore(source=str(structure_path), seed=seed)
    atoms = read_complex(structure_path)
    conf = read_confidence(confidence_path)

    ca = _protein_ca(atoms)
    score.n_residues = ca.array_length()
    if score.n_residues > 260:
        score.note(f"{score.n_residues} residues, above the competition limit of 260")

    # ---- metric 1: iPDE, plus supporting confidence values
    score.ipde = _confidence_value(conf, "complex_ipde", score)
    score.iptm = _confidence_value(conf, "iptm", score)
    score.ligand_iptm = _confidence_value(conf, "ligand_iptm", score)
    score.complex_plddt = _confidence_value(conf, "complex_plddt", score)
    score.complex_iplddt = _confidence_value(conf, "complex_iplddt", score)
    score.confidence_score = _confidence_value(conf, "confidence_score", score)

    # ---- ligand
    ligand = _ligand(atoms, site.ligand_resname)
    if ligand.array_length() == 0:
        score.note(f"no {site.ligand_resname} ligand in the prediction; geometry unavailable")
        return score
    if ligand.array_length() != LIGAND_HEAVY_ATOMS:
        score.note(
            f"{ligand.array_length()} {site.ligand_resname} heavy atoms, expected "
            f"{LIGAND_HEAVY_ATOMS}"
        )
    copies = len(set(zip(ligand.chain_id, ligand.res_id)))
    if copies != 1:
        score.note(f"{copies} copies of {site.ligand_resname}; expected exactly one")

    # ---- metric 2: geometry
    score.base_to_ligand_a = contact_distance(
        atoms, site.base_position, site.base_atom, ligand, site.base_ligand_atom
    )
    score.donor_to_ligand_a = contact_distance(
        atoms, site.donor_position, site.donor_atom, ligand, site.donor_ligand_atom
    )
    if math.isnan(score.base_to_ligand_a):
        score.note(
            f"cannot measure base contact {site.base_atom}->{site.base_ligand_atom} "
            f"at position {site.base_position}; atom missing or ambiguous"
        )
    if math.isnan(score.donor_to_ligand_a):
        score.note(
            f"cannot measure donor contact {site.donor_atom}->{site.donor_ligand_atom} "
            f"at position {site.donor_position}; atom missing or ambiguous"
        )

    # ---- metric 2b: approach angles (secondary to the distances)
    score.base_angle_deg = approach_angle(
        atoms, site.base_position, site.base_root_atom, site.base_atom,
        ligand, site.base_ligand_atom,
    )
    score.donor_angle_deg = approach_angle(
        atoms, site.donor_position, site.donor_root_atom, site.donor_atom,
        ligand, site.donor_ligand_atom,
    )

    # ---- metric 3: burial
    score.base_burial_ca_count = base_burial(atoms, site.base_position)

    # ---- how much the fold changed from the design we handed over
    if reference_structure is not None:
        try:
            score.design_rmsd_a = backbone_rmsd(atoms, read_complex(reference_structure))
        except (ValueError, OSError, FileNotFoundError) as exc:
            score.note(f"cannot compare with the designed structure: {exc}")

    # ---- pocket sanity
    score.min_ligand_protein_a, score.clash_pairs = ligand_packing(atoms, ligand)
    return score


def score_run(run_dir: Path, site: CatalyticSite, *, seed: int | None = None) -> OfficialScore:
    """Score a Boltz ``--out_dir`` directly."""
    structure_path, confidence_path = find_prediction(run_dir)
    return score_prediction(structure_path, confidence_path, site, seed=seed)


# --------------------------------------------------------------- consistency

#: Metrics whose reproducibility across seeds is metric 4.
CONSISTENCY_METRICS = (
    "ipde",
    "base_to_ligand_a", "donor_to_ligand_a",
    "base_angle_deg", "donor_angle_deg",
    "base_burial_ca_count",
    "complex_plddt",        # stability
    "design_rmsd_a",        # how much the fold changed
)


def aggregate_seeds(scores: list[OfficialScore]) -> dict:
    """Metric 4: collapse several seeds of ONE sequence into one row.

    Reports mean and standard deviation for each metric, so selection can use
    ``mean - k * spread`` (see BRIEF.md 4.4) rather than a lucky seed. A design
    with a mediocre mean and near-zero spread is a better bet - mechanistically
    and for the score - than one with a brilliant outlier.

    ``n_seeds`` is carried so a design is never silently rewarded for having
    been measured fewer times.
    """
    if not scores:
        raise ValueError("aggregate_seeds needs at least one score")
    out: dict = {"n_seeds": len(scores)}
    if len({s.n_residues for s in scores}) != 1:
        raise ValueError("cannot aggregate seeds of different sequences (residue counts differ)")
    for metric in CONSISTENCY_METRICS:
        values = np.array(
            [float(getattr(s, metric)) for s in scores if getattr(s, metric) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[f"{metric}_mean"] = math.nan
            out[f"{metric}_std"] = math.nan
            continue
        out[f"{metric}_mean"] = float(values.mean())
        # population std: we want the spread of the seeds we actually ran, not
        # an estimate of a wider population we are not sampling.
        out[f"{metric}_std"] = float(values.std())
        out[f"{metric}_n"] = int(values.size)
    notes = sorted({n for s in scores for n in s.notes})
    out["notes"] = notes
    out["metrics_missing"] = [
        m for m in CONSISTENCY_METRICS if not math.isfinite(out.get(f"{m}_mean", math.nan))
    ]
    return out


# --------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Score Boltz-2 output against the four official metrics.",
    )
    parser.add_argument("run_dir", type=Path, help="a Boltz --out_dir, or a directory of them")
    parser.add_argument("--base", type=int, required=True, help="catalytic base position (1-based)")
    parser.add_argument("--donor", type=int, required=True, help="H-bond donor position (1-based)")
    parser.add_argument("--ligand", default="6NT")
    parser.add_argument("--aggregate", action="store_true", help="treat subdirectories as seeds")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    site = CatalyticSite(
        base_position=args.base, donor_position=args.donor, ligand_resname=args.ligand
    )
    if args.aggregate:
        subdirs = sorted(p for p in args.run_dir.iterdir() if p.is_dir())
        scores = [score_run(d, site) for d in subdirs]
        print(json.dumps(aggregate_seeds(scores), indent=2))
    else:
        print(json.dumps(score_run(args.run_dir, site).as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Target config loading and the resolved reference context.

``configs/target*.yaml`` is written in the REFERENCE numbering (what the PDB
file and the challenge brief use). Everything downstream works in NORMALISED
numbering (1..N == sequence position). :class:`Reference` is the one place
where that translation happens.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
import biotite.structure as struc

from .structure import NumberingMap, normalise, read_structure, sequence_of, get_atom

REPO_ROOT = Path(__file__).resolve().parents[2]

GEOMETRY_KINDS = {2: "distance", 3: "angle", 4: "dihedral"}


def resolve_path(p: str | Path | None) -> Path | None:
    """Config paths are relative to the repo root unless absolute."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


@dataclass
class Target:
    path: Path
    name: str
    reference_structure: Path
    reference_sequence: Path | None
    chain: str | None
    catalytic_residues: list[dict]
    catalytic_geometry: list[dict]
    substrate: dict
    design_region: dict
    objective: str
    success_metrics: list
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def hash(self) -> str:
        return file_hash(self.path)


def load_target(path: str | Path) -> Target:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: target config must be a mapping")
    for key in ("name", "reference_structure"):
        if not raw.get(key):
            raise ValueError(f"{path}: missing required key '{key}'")
    return Target(
        path=path,
        name=str(raw["name"]),
        reference_structure=resolve_path(raw["reference_structure"]),
        reference_sequence=resolve_path(raw.get("reference_sequence")),
        chain=raw.get("chain"),
        catalytic_residues=list(raw.get("catalytic_residues") or []),
        catalytic_geometry=list(raw.get("catalytic_geometry") or []),
        substrate=dict(raw.get("substrate") or {}),
        design_region=dict(raw.get("design_region") or {"mode": "all"}),
        objective=str(raw.get("objective") or ""),
        success_metrics=list(raw.get("success_metrics") or []),
        raw=raw,
    )


def parse_atom_spec(spec: str) -> tuple[int, str]:
    """'35:OE1' -> (35, 'OE1'). Residue number is in reference numbering."""
    try:
        resi, atom = str(spec).split(":")
        return int(resi), atom.strip()
    except ValueError as e:
        raise ValueError(f"bad atom spec '{spec}', expected '<resi>:<ATOM>'") from e


def measure(coords: list[np.ndarray]) -> float:
    """Distance (A), angle (deg) or dihedral (deg) for 2, 3 or 4 points."""
    if len(coords) == 2:
        return float(struc.distance(coords[0], coords[1]))
    if len(coords) == 3:
        return float(np.degrees(struc.angle(*coords)))
    if len(coords) == 4:
        return float(np.degrees(struc.dihedral(*coords)))
    raise ValueError(f"geometry needs 2, 3 or 4 atoms, got {len(coords)}")


def deviation(kind: str, value: float, ideal: float) -> float:
    """Absolute deviation; dihedrals wrap at 360 degrees."""
    d = abs(value - ideal)
    if kind == "dihedral":
        d = min(d, 360.0 - d)
    return d


@dataclass
class Reference:
    """The target's reference structure, normalised, with everything resolved."""

    target: Target
    atoms: struc.AtomArray          # normalised
    numbering: NumberingMap
    sequence: str
    catalytic: list[dict]           # [{pos, resn, role, orig}] pos = normalised
    geometry: list[dict]            # [{name, kind, atoms:[(pos, atom)], ideal, tol, reference_value}]
    design_positions: list[int]     # normalised, sorted

    @property
    def catalytic_positions(self) -> list[int]:
        return [c["pos"] for c in self.catalytic]


def load_reference(target: Target) -> Reference:
    """Normalise the reference and resolve all config indices. Raises on any mismatch."""
    raw_atoms = read_structure(target.reference_structure)
    atoms, numbering = normalise(raw_atoms, chain=target.chain)
    sequence = sequence_of(atoms)

    catalytic = []
    for c in target.catalytic_residues:
        pos = numbering.to_new(int(c["resi"]), chain=target.chain)
        found = numbering.resn(pos)
        expected = str(c["resn"]).upper()
        if found != expected:
            raise ValueError(
                f"catalytic residue {c['resi']}: config says {expected}, structure has {found}"
            )
        catalytic.append({"pos": pos, "resn": expected, "role": c.get("role", ""),
                          "orig": int(c["resi"])})

    geometry = []
    for g in target.catalytic_geometry:
        specs = [parse_atom_spec(s) for s in g["atoms"]]
        if len(specs) not in GEOMETRY_KINDS:
            raise ValueError(f"geometry {g['atoms']}: need 2, 3 or 4 atoms")
        kind = GEOMETRY_KINDS[len(specs)]
        resolved = [(numbering.to_new(r, chain=target.chain), a) for r, a in specs]
        ref_value = measure([get_atom(atoms, p, a) for p, a in resolved])
        ideal = g.get(f"ideal_{kind}", g.get("ideal"))
        geometry.append({
            "name": g.get("name") or kind + "_" + "_".join(str(s).replace(":", "") for s in g["atoms"]),
            "kind": kind,
            "atoms": resolved,
            "ideal": float(ideal) if ideal is not None else ref_value,
            "ideal_from_config": ideal is not None,
            "tol": float(g["tol"]) if g.get("tol") is not None else None,
            "reference_value": ref_value,
        })

    design_positions = resolve_design_region(target, atoms, numbering, [c["pos"] for c in catalytic])
    return Reference(target, atoms, numbering, sequence, catalytic, geometry, design_positions)


def resolve_design_region(
    target: Target, atoms: struc.AtomArray, numbering: NumberingMap, catalytic_pos: list[int]
) -> list[int]:
    """Positions free to change, normalised numbering.

    modes: ``all`` | ``explicit`` (``positions: [...]``) | ``shell`` (any heavy
    atom within ``shell_radius`` A of any catalytic-residue heavy atom).
    ``exclude`` (reference numbering) is always removed.
    """
    region = target.design_region
    mode = region.get("mode", "all")
    n = len(numbering)
    if mode == "all":
        positions = set(range(1, n + 1))
    elif mode == "explicit":
        positions = {numbering.to_new(int(r), chain=target.chain) for r in region.get("positions") or []}
    elif mode == "shell":
        if not catalytic_pos:
            raise ValueError("design_region.mode=shell needs catalytic_residues to centre on")
        radius = float(region["shell_radius"])
        prot = atoms[~atoms.hetero & (atoms.element != "H")]
        centre = prot[np.isin(prot.res_id, catalytic_pos)]
        d = np.linalg.norm(prot.coord[:, None, :] - centre.coord[None, :, :], axis=2).min(axis=1)
        positions = {int(r) for r in np.unique(prot.res_id[d <= radius])}
    else:
        raise ValueError(f"unknown design_region.mode '{mode}' (all|explicit|shell)")
    exclude = {numbering.to_new(int(r), chain=target.chain) for r in region.get("exclude") or []}
    return sorted(positions - exclude)

"""Structure reading, writing and THE normalisation function.

Every entry point that touches a structure calls :func:`normalise`. After it:

* one protein chain (unless ``chains`` lists several), no waters/heteroatoms
  unless ``keep_hetero`` is set
* residues renumbered 1..N in file order, insertion codes stripped
* a numbering map (original chain/resi/insertion code -> new resi) is returned
  and can be written as a JSON sidecar, so catalytic residue indices given in
  the reference numbering can always be mapped to sequence positions and back

Position ``i`` in the normalised numbering is index ``i-1`` of the sequence
returned by :func:`sequence_of`. Every metric in ``src/score`` relies on that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

STRUCT_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
# Modified residues we are willing to read as their parent amino acid.
# Anything else non-standard becomes 'X' and is reported, never guessed.
MODIFIED_TO_PARENT = {"MSE": "MET", "SEC": "CYS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}


@dataclass(frozen=True)
class ResidueKey:
    """A residue in the ORIGINAL file numbering."""

    chain: str
    resi: int
    ins: str = ""

    def __str__(self) -> str:
        return f"{self.chain}:{self.resi}{self.ins}"


@dataclass
class NumberingMap:
    """Maps normalised positions (1..N) <-> original residue identifiers."""

    entries: list[dict]  # [{new_resi, chain, resi, ins, resn}], ordered by new_resi

    def to_new(self, resi: int, chain: str | None = None, ins: str = "") -> int:
        hits = [
            e for e in self.entries
            if e["resi"] == resi and e["ins"] == ins and (chain is None or e["chain"] == chain)
        ]
        if not hits:
            raise KeyError(f"original residue {chain or '*'}:{resi}{ins} not in structure")
        if len(hits) > 1:
            raise KeyError(
                f"original residue {resi}{ins} is ambiguous across chains "
                f"{sorted({h['chain'] for h in hits})}; specify the chain"
            )
        return hits[0]["new_resi"]

    def to_orig(self, new_resi: int) -> ResidueKey:
        if not 1 <= new_resi <= len(self.entries):
            raise KeyError(f"normalised position {new_resi} out of range 1..{len(self.entries)}")
        e = self.entries[new_resi - 1]
        return ResidueKey(e["chain"], e["resi"], e["ins"])

    def resn(self, new_resi: int) -> str:
        return self.entries[new_resi - 1]["resn"]

    def __len__(self) -> int:
        return len(self.entries)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"numbering": self.entries}, indent=1), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "NumberingMap":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["numbering"])


def read_structure(path: str | Path, model: int = 1) -> struc.AtomArray:
    """Read PDB or mmCIF (author numbering), first altloc, with B-factors."""
    path = Path(path)
    suffix = path.suffix.lower()
    extra = ["b_factor", "occupancy"]
    if suffix in {".pdb", ".ent"}:
        f = pdb.PDBFile.read(str(path))
        return pdb.get_structure(f, model=model, extra_fields=extra)
    if suffix in {".cif", ".mmcif"}:
        f = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(f, model=model, extra_fields=extra, use_author_fields=True)
    raise ValueError(f"unsupported structure format: {path}")


def write_structure(atoms: struc.AtomArray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        f = pdb.PDBFile()
        f.set_structure(atoms)
        f.write(str(path))
    elif suffix in {".cif", ".mmcif"}:
        f = pdbx.CIFFile()
        pdbx.set_structure(f, atoms)
        f.write(str(path))
    else:
        raise ValueError(f"unsupported structure format: {path}")


def _is_polymer_residue(atoms: struc.AtomArray) -> np.ndarray:
    known = set(THREE_TO_ONE) | set(MODIFIED_TO_PARENT)
    return np.isin(atoms.res_name, list(known)) | (~atoms.hetero & (atoms.res_name != "HOH"))


def normalise(
    atoms: struc.AtomArray,
    chain: str | None = None,
    chains: list[str] | None = None,
    keep_hetero: bool = False,
) -> tuple[struc.AtomArray, NumberingMap]:
    """Normalise a structure. See module docstring.

    ``chain``: the single chain to keep. If None and the file has exactly one
    protein chain that chain is used; if it has several this raises rather
    than guessing. ``chains`` keeps several (numbered consecutively, in the
    order given). ``keep_hetero`` retains non-water heteroatoms of the kept
    chains (ligands) after the protein, numbered after the last residue; they
    are not part of the NumberingMap.
    """
    polymer = _is_polymer_residue(atoms)
    available = list(dict.fromkeys(atoms.chain_id[polymer].tolist()))
    if chains is None:
        if chain is None:
            if len(available) != 1:
                raise ValueError(
                    f"structure has protein chains {available}; specify which one to keep"
                )
            chain = available[0]
        chains = [chain]
    missing = [c for c in chains if c not in available]
    if missing:
        raise ValueError(f"chain(s) {missing} not found; structure has {available}")

    pieces: list[struc.AtomArray] = []
    entries: list[dict] = []
    new_resi = 0
    for c in chains:
        sel = atoms[polymer & (atoms.chain_id == c)]
        starts = struc.get_residue_starts(sel, add_exclusive_stop=True)
        seen: set[tuple[int, str]] = set()
        new_ids = np.zeros(sel.array_length(), dtype=int)
        for a, b in zip(starts[:-1], starts[1:]):
            key = (int(sel.res_id[a]), str(sel.ins_code[a]))
            if key in seen:
                raise ValueError(f"chain {c}: residue {key[0]}{key[1]} appears twice")
            seen.add(key)
            new_resi += 1
            new_ids[a:b] = new_resi
            entries.append(
                {"new_resi": new_resi, "chain": c, "resi": key[0], "ins": key[1],
                 "resn": str(sel.res_name[a])}
            )
        sel = sel.copy()
        sel.res_id = new_ids
        sel.ins_code = np.full(sel.array_length(), "", dtype=sel.ins_code.dtype)
        pieces.append(sel)

    if new_resi == 0:
        raise ValueError("no protein residues left after normalisation")

    if keep_hetero:
        het = atoms[~polymer & (atoms.res_name != "HOH") & np.isin(atoms.chain_id, chains)]
        if het.array_length():
            het = het.copy()
            starts = struc.get_residue_starts(het, add_exclusive_stop=True)
            ids = np.zeros(het.array_length(), dtype=int)
            for k, (a, b) in enumerate(zip(starts[:-1], starts[1:]), start=1):
                ids[a:b] = new_resi + k
            het.res_id = ids
            het.ins_code = np.full(het.array_length(), "", dtype=het.ins_code.dtype)
            pieces.append(het)

    out = pieces[0]
    for p in pieces[1:]:
        out = out + p
    return out, NumberingMap(entries)


def sequence_of(atoms: struc.AtomArray) -> str:
    """One-letter sequence of the protein residues, in file order. Unknown -> 'X'."""
    prot = atoms[_is_polymer_residue(atoms)]
    starts = struc.get_residue_starts(prot)
    out = []
    for name in prot.res_name[starts]:
        name = MODIFIED_TO_PARENT.get(name, name)
        out.append(THREE_TO_ONE.get(name, "X"))
    return "".join(out)


def chain_breaks(atoms: struc.AtomArray, max_ca_ca: float = 4.2) -> list[int]:
    """Normalised positions i where the CA(i)-CA(i+1) distance says residues are missing."""
    ca = atoms[(atoms.atom_name == "CA") & _is_polymer_residue(atoms)]
    d = np.linalg.norm(ca.coord[1:] - ca.coord[:-1], axis=1)
    return [int(r) for r in ca.res_id[:-1][d > max_ca_ca]]


def get_atom(atoms: struc.AtomArray, resi: int, atom_name: str) -> np.ndarray:
    """Coordinates of one atom by normalised residue number and atom name."""
    hit = atoms[(atoms.res_id == resi) & (atoms.atom_name == atom_name) & ~atoms.hetero]
    if hit.array_length() == 0:
        present = sorted(set(atoms.atom_name[(atoms.res_id == resi) & ~atoms.hetero].tolist()))
        raise KeyError(f"atom {resi}:{atom_name} not found (residue {resi} has {present})")
    if hit.array_length() > 1:
        raise KeyError(f"atom {resi}:{atom_name} matches {hit.array_length()} atoms")
    return hit.coord[0]


def list_structures(directory: str | Path) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in STRUCT_SUFFIXES)

"""Identity tracking: the chain of index maps that follows a hydrogen bond
from seed numbering to Boltz-2 output, and the resolver that turns a named
atom into coordinates.

Nothing here infers an identity by proximity or alignment. A hydrogen bond is
declared once against seed numbering and keeps its string id forever; what
changes is where its atoms live, and each stage contributes one explicit map.

The one rule above all others: `resolve` raises on a miss and never returns a
default. A missing atom scored as zero distance looks like a perfect bond; a
missing atom scored as infinite distance looks like a failed design. Both
corrupt the search silently, which is worse than a crash.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .config import AtomSpec, MotifCfg

ResKey = tuple[str, int]


class TrackingError(RuntimeError):
    """Base for every failure in this module. All of them stop a design."""


class AtomResolutionError(TrackingError):
    """A named atom could not be found. Never recoverable, never defaulted."""


class MotifTraceError(TrackingError):
    """A motif residue was lost or changed between two stages."""


# --------------------------------------------------------------------------
# a parsed structure, independent of file format
# --------------------------------------------------------------------------


@dataclass
class Structure:
    """Minimal structure: per-atom coordinates keyed by chain/resnum/atom name.

    Deliberately not a full structure object. The scorer needs three things --
    a coordinate by name, a residue's type, and the ligand's heavy atoms -- and
    a dict of those is easier to assert on than a library's object graph.
    """

    # (chain, resnum, atom_name) -> xyz
    atoms: dict[tuple[str, int, str], np.ndarray]
    # (chain, resnum) -> three-letter residue name
    residues: dict[ResKey, str]
    # ligand atoms, keyed by atom name, in whatever naming the file used
    ligand: dict[str, np.ndarray] = field(default_factory=dict)
    ligand_resname: str | None = None
    source: str | None = None

    def sequence(self, chain: str) -> str:
        """One-letter sequence of `chain`, in ascending residue number."""
        nums = sorted(n for c, n in self.residues if c == chain)
        return "".join(THREE_TO_ONE.get(self.residues[(chain, n)], "X") for n in nums)

    def residue_numbers(self, chain: str) -> list[int]:
        return sorted(n for c, n in self.residues if c == chain)


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # protonation-state variants that pdbfixer/reduce may write
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "MSE": "M",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}
# ambiguous reverse entries resolved to the canonical name
ONE_TO_THREE.update({"H": "HIS", "C": "CYS", "M": "MET"})

_SOLVENT = {"HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "ACT", "CL", "NA",
            "MG", "ZN", "CA", "K", "MN", "PEG", "DMS", "TRS", "EPE"}


def read_pdb(path: str | Path, ligand_resname: str | None = None) -> Structure:
    """Parse a PDB file.

    Only the first altloc of any atom is kept (blank or 'A'), because a
    disordered sidechain with two positions would otherwise give two different
    answers for the same bond depending on dict insertion order.
    """
    path = Path(path)
    atoms: dict[tuple[str, int, str], np.ndarray] = {}
    residues: dict[ResKey, str] = {}
    ligand: dict[str, np.ndarray] = {}
    found_ligand_resname: str | None = None

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            chain = line[21].strip() or "A"
            try:
                resnum = int(line[22:26])
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                )
            except ValueError as exc:
                raise TrackingError(f"{path}: unparseable coordinate line: {line!r}") from exc

            is_ligand = (
                ligand_resname is not None and resname == ligand_resname
            ) or (record == "HETATM" and ligand_resname is None and resname not in _SOLVENT)

            if is_ligand:
                if atom_name in ligand:
                    continue  # first altloc wins, as above
                ligand[atom_name] = xyz
                found_ligand_resname = resname
                continue

            if record == "HETATM" and resname in _SOLVENT:
                continue

            key = (chain, resnum, atom_name)
            if key in atoms:
                continue
            atoms[key] = xyz
            residues[(chain, resnum)] = resname

    if not atoms:
        raise TrackingError(f"{path}: no protein atoms parsed")

    return Structure(
        atoms=atoms,
        residues=residues,
        ligand=ligand,
        ligand_resname=found_ligand_resname or ligand_resname,
        source=str(path),
    )


def read_cif(path: str | Path, ligand_resname: str | None = None) -> Structure:
    """Parse an mmCIF `atom_site` loop. Boltz-2 writes CIF, not PDB.

    A deliberately small reader: it reads the atom_site loop by column name, so
    it does not care about column order, and it ignores everything else in the
    file. Confidence metrics come from the sidecar JSON, not from here.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    columns, rows = _read_atom_site_loop(lines)

    if not columns or not rows:
        raise TrackingError(f"{path}: no _atom_site loop found")

    idx = {name: i for i, name in enumerate(columns)}
    required = ["label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    missing = [c for c in required if c not in idx]
    if missing:
        raise TrackingError(f"{path}: _atom_site loop lacks columns {missing}")

    chain_col = next(
        (c for c in ("auth_asym_id", "label_asym_id") if c in idx), None
    )
    seq_col = next(
        (c for c in ("auth_seq_id", "label_seq_id") if c in idx), None
    )
    if chain_col is None or seq_col is None:
        raise TrackingError(f"{path}: _atom_site loop lacks chain or residue id columns")

    atoms: dict[tuple[str, int, str], np.ndarray] = {}
    residues: dict[ResKey, str] = {}
    ligand: dict[str, np.ndarray] = {}
    found_ligand_resname: str | None = None

    for row in rows:
        if len(row) < len(columns):
            continue
        altloc = row[idx["label_alt_id"]] if "label_alt_id" in idx else "."
        if altloc not in (".", "?", "A", ""):
            continue
        resname = row[idx["label_comp_id"]].strip().upper()
        atom_name = row[idx["label_atom_id"]].strip().strip('"')
        chain = row[idx[chain_col]].strip() or "A"
        raw_seq = row[idx[seq_col]].strip()
        try:
            xyz = np.array(
                [float(row[idx["Cartn_x"]]), float(row[idx["Cartn_y"]]),
                 float(row[idx["Cartn_z"]])],
                dtype=float,
            )
        except ValueError:
            continue

        is_polymer = resname in THREE_TO_ONE
        if ligand_resname is not None:
            is_ligand = resname == ligand_resname
        else:
            is_ligand = not is_polymer and resname not in _SOLVENT

        if is_ligand:
            if atom_name not in ligand:
                ligand[atom_name] = xyz
                found_ligand_resname = resname
            continue
        if not is_polymer:
            continue
        if raw_seq in (".", "?", ""):
            continue
        resnum = int(raw_seq)
        key = (chain, resnum, atom_name)
        if key not in atoms:
            atoms[key] = xyz
            residues[(chain, resnum)] = resname

    if not atoms:
        raise TrackingError(f"{path}: no polymer atoms parsed")

    return Structure(
        atoms=atoms, residues=residues, ligand=ligand,
        ligand_resname=found_ligand_resname or ligand_resname, source=str(path),
    )


def _read_atom_site_loop(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Find the `_atom_site` loop and return its column names and data rows.

    Written as an explicit three-state scan -- looking for the loop, reading its
    column names, reading its rows -- because the compressed version of this
    had a real bug: a `loop_` appearing after the atom_site data reset the
    column list, and the rows already collected were then discarded. CIF files
    put their loops in no guaranteed order, so that would have depended on
    which writer produced the file.
    """
    columns: list[str] = []
    rows: list[list[str]] = []
    state = "seeking"      # seeking -> header -> data

    for line in lines:
        stripped = line.strip()

        if state == "seeking":
            if stripped == "loop_":
                state = "header"
                columns = []
            continue

        if state == "header":
            if stripped.startswith("_atom_site."):
                columns.append(stripped.split(".", 1)[1])
                continue
            if stripped.startswith("_"):
                # Some other category's loop. Go back to looking.
                state = "seeking"
                columns = []
                continue
            if not columns:
                # A loop we do not care about; its data starts here.
                state = "seeking"
                continue
            state = "data"
            # fall through: this line is the first data row

        if state == "data":
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("_")
                or stripped == "loop_"
                or stripped.startswith("data_")
            ):
                # The loop has ended. We have what we came for.
                break
            rows.append(_split_cif_row(stripped))

    return columns, rows


def _split_cif_row(line: str) -> list[str]:
    out: list[str] = []
    token = ""
    quote: str | None = None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
            else:
                token += ch
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch.isspace():
            if token:
                out.append(token)
                token = ""
            continue
        token += ch
    if token:
        out.append(token)
    return out


def read_structure(path: str | Path, ligand_resname: str | None = None) -> Structure:
    """Dispatch on suffix. Boltz writes .cif, RFdiffusion and pdbfixer .pdb."""
    suffix = Path(path).suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return read_cif(path, ligand_resname)
    if suffix in (".pdb", ".ent"):
        return read_pdb(path, ligand_resname)
    raise TrackingError(f"unsupported structure format: {path}")


# --------------------------------------------------------------------------
# stage 1: seed -> RFdiffusion output, from the .trb pickle
# --------------------------------------------------------------------------


def load_trb(trb_path: str | Path) -> dict:
    with Path(trb_path).open("rb") as fh:
        return pickle.load(fh)


def seed_to_hal_map(trb_path: str | Path, motif: MotifCfg) -> dict[ResKey, ResKey]:
    """Build the seed -> RFdiffusion-output residue map from the `.trb` pickle.

    RFdiffusion writes a `.trb` beside every output PDB holding
    `con_ref_pdb_idx` and `con_hal_pdb_idx` as (chain, resnum) pairs for every
    motif residue. We read those directly rather than the zero-indexed
    `con_ref_idx0` / `con_hal_idx0`, because the pdb_idx form is already in the
    numbering the rest of the pipeline speaks.

    A motif residue missing from the map means the contig string and the motif
    list disagree. That stops the run: the alternative is a design whose motif
    silently moved.
    """
    trb = load_trb(trb_path)

    for key in ("con_ref_pdb_idx", "con_hal_pdb_idx"):
        if key not in trb:
            raise MotifTraceError(
                f"{trb_path} has no {key!r}. Keys present: {sorted(trb)}. "
                "Without it the motif cannot be traced and the run must stop."
            )

    ref = [(str(c), int(n)) for c, n in trb["con_ref_pdb_idx"]]
    hal = [(str(c), int(n)) for c, n in trb["con_hal_pdb_idx"]]
    if len(ref) != len(hal):
        raise MotifTraceError(
            f"{trb_path}: con_ref_pdb_idx has {len(ref)} entries but "
            f"con_hal_pdb_idx has {len(hal)}"
        )

    mapping = dict(zip(ref, hal))

    missing = sorted(motif.as_keys() - set(mapping))
    if missing:
        raise MotifTraceError(
            f"{trb_path}: motif residues {missing} are absent from the .trb "
            "constraint map. The contig string and motif.fixed disagree. "
            f"Mapped residues were: {sorted(mapping)}"
        )
    return mapping


# --------------------------------------------------------------------------
# stage 2: RFdiffusion -> LigandMPNN (identity, but asserted)
# --------------------------------------------------------------------------


def assert_motif_unchanged(
    structure: Structure,
    expected: dict[ResKey, str],
    stage: str,
) -> None:
    """Assert the residue type at every motif position is what we expect.

    LigandMPNN redesigning a fixed residue is a configuration error that is
    otherwise invisible, so this runs after every stage rather than only at the
    end.
    """
    problems = []
    for key, want in sorted(expected.items()):
        got = structure.residues.get(key)
        if got is None:
            problems.append(f"{key[0]}{key[1]}: absent (expected {want})")
        elif THREE_TO_ONE.get(got, "X") != THREE_TO_ONE.get(want, "X"):
            problems.append(f"{key[0]}{key[1]}: {got} (expected {want})")
    if problems:
        raise MotifTraceError(
            f"motif identity changed after {stage}:\n  " + "\n  ".join(problems)
        )


# --------------------------------------------------------------------------
# stage 3: LigandMPNN -> Boltz-2
# --------------------------------------------------------------------------


def hal_to_boltz_map(
    submitted_sequences: dict[str, str],
    hal_residue_numbers: dict[str, list[int]],
) -> dict[ResKey, ResKey]:
    """Map RFdiffusion-output numbering to Boltz-2 numbering.

    Boltz-2 takes sequences, not a PDB, and numbers its output 1..N per chain
    in submitted order. Do not assume an offset of zero: compute the index of
    each residue in the exact string that was submitted.

    `hal_residue_numbers` is the ascending residue numbers of each chain in the
    structure the sequence was read from; `submitted_sequences` is what went
    into the Boltz YAML. Their lengths must agree, or the sequence was trimmed
    or reordered somewhere and the mapping would be a lie.
    """
    mapping: dict[ResKey, ResKey] = {}
    for chain, seq in submitted_sequences.items():
        numbers = hal_residue_numbers.get(chain)
        if numbers is None:
            raise MotifTraceError(
                f"chain {chain!r} was submitted to Boltz-2 but is absent from the "
                f"source structure (which has {sorted(hal_residue_numbers)})"
            )
        if len(numbers) != len(seq):
            raise MotifTraceError(
                f"chain {chain!r}: submitted {len(seq)} residues but the source "
                f"structure has {len(numbers)}. The sequence was trimmed or "
                "reordered; the Boltz numbering cannot be derived."
            )
        for position, resnum in enumerate(numbers, start=1):
            mapping[(chain, resnum)] = (chain, position)
    return mapping


# --------------------------------------------------------------------------
# stage 4: ligand atom names
# --------------------------------------------------------------------------


def ligand_atom_map_ccd(
    reference_names: list[str], output_ligand: dict[str, np.ndarray]
) -> dict[str, str]:
    """On the CCD path Boltz-2 uses CCD atom names, so names resolve directly.

    Still checked rather than assumed: a name in the target spec that is absent
    from the output is an error here, not a KeyError in the scorer.
    """
    missing = [n for n in reference_names if n not in output_ligand]
    if missing:
        raise AtomResolutionError(
            f"ligand atoms {missing} named in the target spec are absent from the "
            f"predicted ligand. Atoms present: {sorted(output_ligand)}. "
            "On the CCD path names should match exactly; check the CCD code."
        )
    return {n: n for n in reference_names}


def ligand_atom_map_graph(
    reference_sdf: str | Path, output_block: str | Path
) -> dict[str, str]:
    """On the SMILES path, map reference atom names onto Boltz's own naming.

    Boltz assigns its own atom names when given SMILES, so the names in the
    target spec do not survive. Build an RDKit molecule from the reference SDF,
    which carries the intended names, build one from the output, and match the
    graphs. The map is identical for every design, so callers cache it.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AtomResolutionError(
            "the SMILES ligand path needs RDKit for graph matching, and it is "
            "not installed. Install rdkit, or give the ligand a CCD code."
        ) from exc

    ref = Chem.MolFromMolFile(str(reference_sdf), removeHs=True, sanitize=True)
    if ref is None:
        raise AtomResolutionError(f"RDKit could not read {reference_sdf}")

    out_path = Path(output_block)
    if out_path.suffix.lower() in (".cif", ".mmcif"):
        out = Chem.MolFromMmcifFile(str(out_path), removeHs=True, sanitize=True) \
            if hasattr(Chem, "MolFromMmcifFile") else None
    else:
        out = Chem.MolFromPDBFile(str(out_path), removeHs=True, sanitize=True)
    if out is None:
        raise AtomResolutionError(
            f"RDKit could not read the ligand from {output_block} for graph matching"
        )

    match = out.GetSubstructMatch(ref)
    if not match or len(match) != ref.GetNumAtoms():
        raise AtomResolutionError(
            f"no one-to-one substructure match between {reference_sdf} and "
            f"{output_block}; the predicted ligand is not the reference molecule"
        )

    mapping: dict[str, str] = {}
    for ref_idx, out_idx in enumerate(match):
        ref_atom = ref.GetAtomWithIdx(ref_idx)
        out_atom = out.GetAtomWithIdx(int(out_idx))
        ref_name = _rdkit_atom_name(ref_atom, ref_idx)
        out_name = _rdkit_atom_name(out_atom, int(out_idx))
        mapping[ref_name] = out_name
    return mapping


def _rdkit_atom_name(atom, index: int) -> str:
    info = atom.GetPDBResidueInfo()
    if info is not None and info.GetName().strip():
        return info.GetName().strip()
    if atom.HasProp("molFileAlias"):
        return atom.GetProp("molFileAlias").strip()
    return f"{atom.GetSymbol()}{index + 1}"


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomRef:
    """An atom to resolve. The scorer's unit of reference."""

    site: Literal["protein", "ligand"]
    atom: str
    chain: str | None = None
    resnum: int | None = None

    @classmethod
    def from_spec(cls, spec: AtomSpec) -> AtomRef:
        return cls(site=spec.site, atom=spec.atom, chain=spec.chain, resnum=spec.resnum)

    def __str__(self) -> str:
        if self.site == "ligand":
            return f"ligand:{self.atom}"
        return f"{self.chain}{self.resnum}:{self.atom}"


@dataclass
class Provenance:
    """Everything needed to resolve a declared atom in a predicted structure.

    Serialised whole into the design's database row. A design whose maps are
    lost is not re-derivable, and the lineage is worthless without them.
    """

    design_id: str
    scaffold_id: str
    parent_id: str | None = None
    seed_to_hal: dict[ResKey, ResKey] = field(default_factory=dict)
    hal_to_boltz: dict[ResKey, ResKey] = field(default_factory=dict)
    ligand_atom_map: dict[str, str] = field(default_factory=dict)
    motif_expected: dict[ResKey, str] = field(default_factory=dict)

    # ---- resolution ----

    def seed_residue_in_boltz(self, chain: str, resnum: int) -> ResKey:
        """Follow one protein residue through both maps."""
        seed_key = (chain, resnum)
        if seed_key not in self.seed_to_hal:
            raise AtomResolutionError(
                f"residue {chain}{resnum} is not in seed_to_hal for design "
                f"{self.design_id}. Mapped: {sorted(self.seed_to_hal)}"
            )
        hal_key = self.seed_to_hal[seed_key]
        if hal_key not in self.hal_to_boltz:
            raise AtomResolutionError(
                f"residue {chain}{resnum} maps to {hal_key} in the RFdiffusion "
                f"output, which is not in hal_to_boltz for design {self.design_id}"
            )
        return self.hal_to_boltz[hal_key]

    def resolve(self, ref: AtomRef, structure: Structure) -> np.ndarray:
        """Return xyz. Raise AtomResolutionError on any miss.

        Never returns a default. See the module docstring for why.
        """
        if ref.site == "ligand":
            name = self.ligand_atom_map.get(ref.atom)
            if name is None:
                raise AtomResolutionError(
                    f"ligand atom {ref.atom!r} has no entry in ligand_atom_map for "
                    f"design {self.design_id}. Mapped: {sorted(self.ligand_atom_map)}"
                )
            if name not in structure.ligand:
                raise AtomResolutionError(
                    f"ligand atom {ref.atom!r} maps to {name!r}, absent from "
                    f"{structure.source}. Present: {sorted(structure.ligand)}"
                )
            return structure.ligand[name]

        if ref.chain is None or ref.resnum is None:
            raise AtomResolutionError(f"protein AtomRef {ref} has no chain/resnum")

        chain, resnum = self.seed_residue_in_boltz(ref.chain, ref.resnum)
        key = (chain, resnum, ref.atom)
        if key not in structure.atoms:
            present = sorted(
                n for c, r, n in structure.atoms if (c, r) == (chain, resnum)
            )
            raise AtomResolutionError(
                f"atom {ref} resolves to {chain}{resnum}:{ref.atom} in "
                f"{structure.source}, which is absent. Atoms at that residue: "
                f"{present or 'none - the residue itself is missing'}"
            )
        return structure.atoms[key]

    # ---- serialisation ----

    def to_json(self) -> str:
        return json.dumps(
            {
                "design_id": self.design_id,
                "scaffold_id": self.scaffold_id,
                "parent_id": self.parent_id,
                "seed_to_hal": [[list(k), list(v)] for k, v in self.seed_to_hal.items()],
                "hal_to_boltz": [[list(k), list(v)] for k, v in self.hal_to_boltz.items()],
                "ligand_atom_map": self.ligand_atom_map,
                "motif_expected": [[list(k), v] for k, v in self.motif_expected.items()],
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, blob: str) -> Provenance:
        raw = json.loads(blob)
        return cls(
            design_id=raw["design_id"],
            scaffold_id=raw["scaffold_id"],
            parent_id=raw.get("parent_id"),
            seed_to_hal={(k[0], int(k[1])): (v[0], int(v[1]))
                         for k, v in raw.get("seed_to_hal", [])},
            hal_to_boltz={(k[0], int(k[1])): (v[0], int(v[1]))
                          for k, v in raw.get("hal_to_boltz", [])},
            ligand_atom_map=raw.get("ligand_atom_map", {}),
            motif_expected={(k[0], int(k[1])): v
                            for k, v in raw.get("motif_expected", [])},
        )


def identity_provenance(
    design_id: str,
    scaffold_id: str,
    structure: Structure,
    motif: MotifCfg,
    ligand_atoms: list[str],
    parent_id: str | None = None,
) -> Provenance:
    """Provenance for a structure already in its own numbering.

    Used for the seed itself, for hand-built test fixtures, and by the
    `measure` subcommand: all three cases where no stage has renumbered
    anything, so both residue maps are the identity. It still goes through the
    same resolver, so the fixtures exercise the real code path.
    """
    keys = motif.as_keys()
    missing = sorted(k for k in keys if k not in structure.residues)
    if missing:
        raise MotifTraceError(
            f"{structure.source}: motif residues {missing} are absent"
        )
    return Provenance(
        design_id=design_id,
        scaffold_id=scaffold_id,
        parent_id=parent_id,
        seed_to_hal={k: k for k in keys},
        hal_to_boltz={k: k for k in keys},
        ligand_atom_map={n: n for n in ligand_atoms},
        motif_expected={k: structure.residues[k] for k in keys},
    )

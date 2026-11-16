"""Shared fixtures. All expected values in the tests are for the committed toy
enzyme (hen egg-white lysozyme, PDB 1AKI chain A) and are hardcoded on purpose."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.io.structure import read_structure, write_structure
from src.io.target import load_reference, load_target

ROOT = Path(__file__).resolve().parents[1]
TOY_PDB = ROOT / "data" / "toy" / "ref.pdb"
TOY_TARGET = ROOT / "configs" / "target.example.yaml"
PIPELINE = ROOT / "configs" / "pipeline.yaml"

TOY_SEQ = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIP"
    "CSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)


@pytest.fixture(scope="session")
def toy_target():
    return load_target(TOY_TARGET)


@pytest.fixture(scope="session")
def toy_ref(toy_target):
    return load_reference(toy_target)


@pytest.fixture()
def nasty_pdb(tmp_path):
    """The toy enzyme re-numbered the way real PDB files hurt you.

    * chain A -> chain 'E', numbering shifted by +100 (so Glu35 is E:135)
    * original residues 48,49,50 become 148, 148A, 148B (insertion codes), and
      everything after is shifted back by 2 (so Asp52 is E:150)
    * a decoy chain 'B' (copy of residues 1-20) placed BEFORE the real chain
    * waters kept
    Returns (path, glu_resi, asp_resi) in the nasty numbering.
    """
    atoms = read_structure(TOY_PDB)
    prot = atoms[~atoms.hetero].copy()
    water = atoms[atoms.hetero].copy()

    orig = prot.res_id.copy()
    new_id = orig + 100
    ins = np.full(prot.array_length(), "", dtype="U1")
    new_id[orig == 49] = 148
    ins[orig == 49] = "A"
    new_id[orig == 50] = 148
    ins[orig == 50] = "B"
    new_id[orig > 50] = orig[orig > 50] + 100 - 2
    prot.res_id = new_id
    prot.ins_code = ins
    prot.chain_id = np.full(prot.array_length(), "E", dtype="U4")

    decoy = atoms[~atoms.hetero & (atoms.res_id <= 20)].copy()
    decoy.chain_id = np.full(decoy.array_length(), "B", dtype="U4")
    decoy.coord = decoy.coord + 60.0

    path = tmp_path / "nasty.pdb"
    write_structure(decoy + prot + water, path)
    return path, 135, 150


@pytest.fixture()
def nasty_target(nasty_pdb, tmp_path):
    path, glu, asp = nasty_pdb
    cfg = {
        "name": "nasty",
        "reference_structure": str(path),
        "chain": "E",
        "catalytic_residues": [
            {"resi": glu, "resn": "GLU", "role": "general_acid"},
            {"resi": asp, "resn": "ASP", "role": "nucleophile"},
        ],
        "catalytic_geometry": [
            {"atoms": [f"{glu}:OE1", f"{asp}:OD1"], "ideal_distance": 7.05, "tol": 1.5}
        ],
        "design_region": {"mode": "shell", "shell_radius": 8.0, "exclude": [glu, asp]},
    }
    p = tmp_path / "target.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p

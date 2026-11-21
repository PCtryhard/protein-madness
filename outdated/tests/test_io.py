"""Numbering is the classic way to silently poison every metric. These tests
exist to catch that at hour 3 instead of hour 28."""
import numpy as np
import pytest
import yaml

from src.io.fasta import SeqRecord, read_fasta, write_fasta
from src.io.structure import (NumberingMap, chain_breaks, get_atom, normalise, read_structure,
                              sequence_of, write_structure)
from src.io.target import load_reference, load_target
from src.io.validate_target import validate

from .conftest import TOY_PDB, TOY_SEQ, TOY_TARGET


# ---------- fasta ----------

def test_fasta_roundtrip(tmp_path):
    recs = [SeqRecord("d_abc", "ACDEFGHIK" * 20, {"method": "mutate", "mutations": "A1C+D3E"}),
            SeqRecord("d_def", "MKV")]
    write_fasta(recs, tmp_path / "x.fasta")
    back = read_fasta(tmp_path / "x.fasta")
    assert [(r.id, r.seq, r.fields) for r in back] == [(r.id, r.seq, r.fields) for r in recs]


def test_fasta_rejects_duplicate_ids(tmp_path):
    (tmp_path / "x.fasta").write_text(">a\nAC\n>a\nAC\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_fasta(tmp_path / "x.fasta")


# ---------- toy enzyme, clean numbering ----------

def test_toy_sequence_and_numbering():
    atoms, numbering = normalise(read_structure(TOY_PDB), chain="A")
    assert sequence_of(atoms) == TOY_SEQ
    assert len(numbering) == 129
    assert numbering.to_new(35) == 35 and numbering.resn(35) == "GLU"
    assert numbering.to_new(52) == 52 and numbering.resn(52) == "ASP"
    assert not atoms.hetero.any()          # 78 waters dropped
    assert atoms.array_length() == 1001
    assert chain_breaks(atoms) == []


def test_position_i_is_sequence_index_i_minus_1():
    atoms, _ = normalise(read_structure(TOY_PDB), chain="A")
    seq = sequence_of(atoms)
    assert seq[35 - 1] == "E" and seq[52 - 1] == "D"


# ---------- the nasty file: offset + insertion codes + decoy chain ----------

def test_normalise_nasty_numbering(nasty_pdb):
    path, glu, asp = nasty_pdb
    raw = read_structure(path)
    atoms, numbering = normalise(raw, chain="E")
    # same protein, same sequence, renumbered 1..129, insertion codes gone
    assert sequence_of(atoms) == TOY_SEQ
    assert list(np.unique(atoms.res_id)) == list(range(1, 130))
    assert set(atoms.ins_code.tolist()) == {""}
    assert set(atoms.chain_id.tolist()) == {"E"}
    # catalytic residues given in the nasty numbering land on the right positions
    assert numbering.to_new(glu, chain="E") == 35
    assert numbering.to_new(asp, chain="E") == 52
    # insertion codes map to distinct consecutive positions
    assert numbering.to_new(148, chain="E") == 48
    assert numbering.to_new(148, chain="E", ins="A") == 49
    assert numbering.to_new(148, chain="E", ins="B") == 50
    # and back again
    assert str(numbering.to_orig(35)) == "E:135"
    assert str(numbering.to_orig(49)) == "E:148A"
    assert str(numbering.to_orig(52)) == "E:150"


def test_coordinates_survive_normalisation(nasty_pdb):
    path, _, _ = nasty_pdb
    clean, _ = normalise(read_structure(TOY_PDB), chain="A")
    nasty, _ = normalise(read_structure(path), chain="E")
    for resi, atom in [(35, "OE1"), (52, "OD1"), (129, "CA")]:
        assert np.allclose(get_atom(clean, resi, atom), get_atom(nasty, resi, atom), atol=1e-3)


def test_multichain_requires_explicit_chain(nasty_pdb):
    path, _, _ = nasty_pdb
    with pytest.raises(ValueError, match="specify which one"):
        normalise(read_structure(path))
    with pytest.raises(ValueError, match="not found"):
        normalise(read_structure(path), chain="Z")


def test_unknown_original_residue_raises(nasty_pdb):
    path, _, _ = nasty_pdb
    _, numbering = normalise(read_structure(path), chain="E")
    with pytest.raises(KeyError):
        numbering.to_new(35, chain="E")    # 35 only exists in the clean numbering


def test_sidecar_roundtrip(tmp_path, nasty_pdb):
    path, _, _ = nasty_pdb
    _, numbering = normalise(read_structure(path), chain="E")
    numbering.write(tmp_path / "n.json")
    assert NumberingMap.read(tmp_path / "n.json").entries == numbering.entries


def test_pdb_cif_roundtrip_agree(tmp_path):
    atoms, _ = normalise(read_structure(TOY_PDB), chain="A")
    write_structure(atoms, tmp_path / "a.pdb")
    write_structure(atoms, tmp_path / "a.cif")
    a = read_structure(tmp_path / "a.pdb")
    b = read_structure(tmp_path / "a.cif")
    assert sequence_of(a) == sequence_of(b) == TOY_SEQ
    assert np.allclose(a.coord, b.coord, atol=1e-3)
    assert (a.res_id == b.res_id).all()


def test_chain_break_detected():
    atoms, _ = normalise(read_structure(TOY_PDB), chain="A")
    gapped = atoms[~np.isin(atoms.res_id, [70, 71, 72])]
    gapped, numbering = normalise(gapped, chain="A")
    assert chain_breaks(gapped) == [69]
    assert numbering.to_new(73) == 70      # numbering closes the gap; the map remembers


# ---------- target config ----------

def test_reference_resolution_toy(toy_ref):
    assert toy_ref.sequence == TOY_SEQ
    assert toy_ref.catalytic_positions == [35, 52]
    g = toy_ref.geometry[0]
    assert g["kind"] == "distance" and g["atoms"] == [(35, "OE1"), (52, "OD1")]
    assert g["reference_value"] == pytest.approx(7.050, abs=0.005)
    assert toy_ref.geometry[1]["reference_value"] == pytest.approx(134.41, abs=0.05)
    assert len(toy_ref.design_positions) == 48
    assert 35 not in toy_ref.design_positions and 52 not in toy_ref.design_positions


def test_reference_resolution_is_numbering_invariant(toy_ref, nasty_target):
    """Same protein under hostile numbering -> identical resolved positions and geometry."""
    nasty = load_reference(load_target(nasty_target))
    assert nasty.catalytic_positions == toy_ref.catalytic_positions
    assert nasty.design_positions == toy_ref.design_positions
    assert nasty.geometry[0]["reference_value"] == pytest.approx(toy_ref.geometry[0]["reference_value"], abs=1e-3)


def test_validator_passes_toy():
    errors, _, _ = validate(TOY_TARGET)
    assert errors == []


def test_validator_catches_off_by_one(tmp_path):
    cfg = yaml.safe_load(TOY_TARGET.read_text())
    cfg["catalytic_residues"][0]["resi"] = 36      # the classic
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    errors, _, _ = validate(p)
    assert any("config says GLU, structure has SER" in e and "off-by-N" in e for e in errors)


def test_validator_catches_bad_atom_and_missing_file(tmp_path):
    cfg = yaml.safe_load(TOY_TARGET.read_text())
    cfg["catalytic_geometry"][0]["atoms"] = ["35:OE9", "52:OD1"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    errors, _, _ = validate(p)
    assert any("35:OE9 not found" in e for e in errors)

    cfg["reference_structure"] = "data/toy/nope.pdb"
    p.write_text(yaml.safe_dump(cfg))
    errors, _, _ = validate(p)
    assert any("not found" in e for e in errors)


def test_validator_warns_when_reference_violates_its_own_ideal(tmp_path):
    cfg = yaml.safe_load(TOY_TARGET.read_text())
    cfg["catalytic_geometry"][0]["ideal_distance"] = 9.1   # value from the brief; 1AKI has 7.05
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(cfg))
    errors, warnings, _ = validate(p)
    assert errors == []
    assert any("REFERENCE itself is outside tol" in w for w in warnings)

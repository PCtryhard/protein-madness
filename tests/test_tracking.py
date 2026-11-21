"""Identity tracking: nothing is inferred by proximity, and a miss raises.

Includes the phase-1 acceptance check against a real structure: every motif
residue traced from seed numbering, every named bond's atoms resolved, and the
measured geometry matching what the target file claims it measured.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.config import MotifCfg, ResidueRef, load
from src.tracking import (
    AtomRef,
    AtomResolutionError,
    MotifTraceError,
    Provenance,
    TrackingError,
    assert_motif_unchanged,
    hal_to_boltz_map,
    identity_provenance,
    ligand_atom_map_ccd,
    read_pdb,
    read_structure,
    seed_to_hal_map,
)

ROOT = Path(__file__).resolve().parents[1]
from tests.paths import BOLTZ_CIF as _BOLTZ_CIF, REAL_SEED, REAL_TARGET

SEED_PDB = REAL_SEED


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_read_fixture_pdb():
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    assert structure.residues[("A", 229)] == "ASP"
    assert structure.residues[("A", 291)] == "HIS"
    assert sorted(structure.ligand) == ["N1", "N2", "N3"]
    assert structure.ligand_resname == "6NT"
    np.testing.assert_allclose(structure.atoms[("A", 229, "OD2")], [0.0, 0.0, 0.0])


def test_sequence_from_structure():
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    assert structure.sequence("A") == "DH"      # ASP 229, HIS 291, in order
    assert structure.residue_numbers("A") == [229, 291]


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "thing.xyz"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(TrackingError, match="unsupported structure format"):
        read_structure(path)


def test_empty_pdb_raises(tmp_path):
    path = tmp_path / "empty.pdb"
    path.write_text("END\n", encoding="utf-8")
    with pytest.raises(TrackingError, match="no protein atoms"):
        read_pdb(path)


@pytest.mark.skipif(not SEED_PDB.exists(), reason="seed PDB not in this checkout")
def test_altloc_is_resolved_deterministically():
    """A disordered sidechain with two positions would otherwise give two
    different answers for the same bond depending on dict ordering."""
    structure = read_pdb(SEED_PDB, ligand_resname="6NT")
    # 5FQJ has altlocs at several residues; every atom must appear exactly once.
    assert len(structure.atoms) == len(set(structure.atoms))
    first = read_pdb(SEED_PDB, ligand_resname="6NT")
    for key, xyz in structure.atoms.items():
        np.testing.assert_allclose(first.atoms[key], xyz)


# --------------------------------------------------------------------------
# the .trb map
# --------------------------------------------------------------------------


def _write_trb(path: Path, ref: list, hal: list) -> Path:
    with path.open("wb") as fh:
        pickle.dump({"con_ref_pdb_idx": ref, "con_hal_pdb_idx": hal}, fh)
    return path


def test_seed_to_hal_map_from_trb(tmp_path):
    motif = MotifCfg(
        fixed=[ResidueRef(chain="A", resnum=229), ResidueRef(chain="A", resnum=291)]
    )
    trb = _write_trb(
        tmp_path / "x.trb",
        ref=[("A", 229), ("A", 291)],
        hal=[("A", 12), ("A", 74)],
    )
    mapping = seed_to_hal_map(trb, motif)
    assert mapping[("A", 229)] == ("A", 12)
    assert mapping[("A", 291)] == ("A", 74)


def test_missing_motif_residue_in_trb_stops_the_run(tmp_path):
    """A missing one means the contig string and the motif list disagree, which
    must stop the run rather than produce a design whose motif silently moved."""
    motif = MotifCfg(
        fixed=[ResidueRef(chain="A", resnum=229), ResidueRef(chain="A", resnum=291)]
    )
    trb = _write_trb(
        tmp_path / "x.trb", ref=[("A", 229)], hal=[("A", 12)]
    )
    with pytest.raises(MotifTraceError, match="absent from the .trb"):
        seed_to_hal_map(trb, motif)


def test_trb_without_the_constraint_keys_raises(tmp_path):
    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=229)])
    path = tmp_path / "bad.trb"
    with path.open("wb") as fh:
        pickle.dump({"something_else": []}, fh)
    with pytest.raises(MotifTraceError, match="con_ref_pdb_idx"):
        seed_to_hal_map(path, motif)


def test_trb_with_mismatched_lengths_raises(tmp_path):
    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=229)])
    trb = _write_trb(
        tmp_path / "x.trb", ref=[("A", 229), ("A", 291)], hal=[("A", 12)]
    )
    with pytest.raises(MotifTraceError, match="entries"):
        seed_to_hal_map(trb, motif)


# --------------------------------------------------------------------------
# the hal -> boltz map
# --------------------------------------------------------------------------


def test_hal_to_boltz_does_not_assume_offset_zero():
    """Boltz numbers 1..N per chain in submitted order; the backbone it came
    from may be numbered anything."""
    mapping = hal_to_boltz_map(
        submitted_sequences={"A": "MKV"},
        hal_residue_numbers={"A": [101, 102, 103]},
    )
    assert mapping == {("A", 101): ("A", 1), ("A", 102): ("A", 2), ("A", 103): ("A", 3)}


def test_hal_to_boltz_length_mismatch_raises():
    """A trimmed or reordered sequence makes the mapping a lie."""
    with pytest.raises(MotifTraceError, match="trimmed or reordered"):
        hal_to_boltz_map({"A": "MKV"}, {"A": [1, 2]})


def test_hal_to_boltz_unknown_chain_raises():
    with pytest.raises(MotifTraceError, match="absent"):
        hal_to_boltz_map({"B": "MKV"}, {"A": [1, 2, 3]})


# --------------------------------------------------------------------------
# motif identity assertions
# --------------------------------------------------------------------------


def test_assert_motif_unchanged_passes_when_unchanged():
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    assert_motif_unchanged(
        structure, {("A", 229): "ASP", ("A", 291): "HIS"}, stage="test"
    )


def test_assert_motif_unchanged_catches_a_redesigned_residue():
    """LigandMPNN redesigning a fixed residue is a configuration error that is
    otherwise invisible."""
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    with pytest.raises(MotifTraceError, match="A229: ASP"):
        assert_motif_unchanged(
            structure, {("A", 229): "GLU", ("A", 291): "HIS"}, stage="ligandmpnn"
        )


def test_assert_motif_unchanged_catches_a_missing_residue():
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    with pytest.raises(MotifTraceError, match="absent"):
        assert_motif_unchanged(structure, {("A", 999): "ASP"}, stage="boltz")


def test_histidine_protonation_variants_count_as_histidine():
    """pdbfixer may write HID/HIE/HIP. Those are the same residue and must not
    trip the motif assertion."""
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    structure.residues[("A", 291)] = "HIE"
    assert_motif_unchanged(structure, {("A", 291): "HIS"}, stage="protonate")


# --------------------------------------------------------------------------
# the ligand atom map
# --------------------------------------------------------------------------


def test_ccd_atom_map_is_the_identity_but_still_checked():
    mapping = ligand_atom_map_ccd(["N1", "N3"], {"N1": None, "N2": None, "N3": None})
    assert mapping == {"N1": "N1", "N3": "N3"}


def test_ccd_atom_map_raises_on_a_name_not_in_the_output():
    with pytest.raises(AtomResolutionError, match="N9"):
        ligand_atom_map_ccd(["N9"], {"N1": None})


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------


def test_resolve_follows_both_maps():
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    provenance = Provenance(
        design_id="d",
        scaffold_id="s",
        seed_to_hal={("A", 45): ("A", 1229)},
        hal_to_boltz={("A", 1229): ("A", 229)},
        ligand_atom_map={"N3": "N3"},
    )
    xyz = provenance.resolve(AtomRef("protein", "OD2", "A", 45), structure)
    np.testing.assert_allclose(xyz, [0.0, 0.0, 0.0])


def test_resolve_never_returns_a_default_for_a_missing_atom():
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    provenance = Provenance(
        design_id="d",
        scaffold_id="s",
        seed_to_hal={("A", 229): ("A", 229)},
        hal_to_boltz={("A", 229): ("A", 229)},
    )
    with pytest.raises(AtomResolutionError, match="OXT"):
        provenance.resolve(AtomRef("protein", "OXT", "A", 229), structure)


def test_resolve_error_names_the_atoms_that_are_there():
    """At hour 20 the useful error says what it found, not just what it missed."""
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    provenance = Provenance(
        design_id="d", scaffold_id="s",
        seed_to_hal={("A", 229): ("A", 229)},
        hal_to_boltz={("A", 229): ("A", 229)},
    )
    with pytest.raises(AtomResolutionError, match="OD1"):
        provenance.resolve(AtomRef("protein", "NOPE", "A", 229), structure)


def test_atomref_str_is_readable():
    assert str(AtomRef("ligand", "N3")) == "ligand:N3"
    assert str(AtomRef("protein", "OD2", "A", 229)) == "A229:OD2"


# --------------------------------------------------------------------------
# serialisation: a design whose maps are lost is not re-derivable
# --------------------------------------------------------------------------


def test_provenance_round_trips_through_json():
    original = Provenance(
        design_id="d1",
        scaffold_id="s1",
        parent_id="d0",
        seed_to_hal={("A", 229): ("A", 12)},
        hal_to_boltz={("A", 12): ("A", 5)},
        ligand_atom_map={"N3": "N3"},
        motif_expected={("A", 229): "ASP"},
    )
    restored = Provenance.from_json(original.to_json())
    assert restored.design_id == "d1"
    assert restored.parent_id == "d0"
    assert restored.seed_to_hal == original.seed_to_hal
    assert restored.hal_to_boltz == original.hal_to_boltz
    assert restored.ligand_atom_map == original.ligand_atom_map
    assert restored.motif_expected == original.motif_expected


def test_provenance_json_is_stable():
    """Stable output means the stored blob diffs cleanly between designs."""
    prov = Provenance(design_id="d", scaffold_id="s", seed_to_hal={("A", 1): ("A", 2)})
    assert prov.to_json() == prov.to_json()


# --------------------------------------------------------------------------
# phase-1 acceptance, against the real seed structure
# --------------------------------------------------------------------------


@pytest.mark.skipif(not SEED_PDB.exists(), reason="seed PDB not in this checkout")
def test_real_target_bonds_resolve_and_measure_as_documented(pipeline_path):
    """The numbers in the variant-01 target came from this structure.

    This is the check that they still do -- that the residue identities, atom
    names and measured geometry in the target file describe the seed PDB in the
    repo, and that every motif residue traces end to end.
    """
    from src.scoring.hbond import measure_all

    cfg = load(pipeline_path, ROOT / REAL_TARGET)
    structure = read_structure(SEED_PDB, cfg.target.ligand.ccd)

    # The motif residues exist and are the types the comments claim.
    assert structure.residues[("A", 229)] == "ASP"
    assert structure.residues[("A", 291)] == "HIS"

    ligand_atoms = sorted(
        {
            spec.atom
            for bond in cfg.target.hbonds
            for spec in (bond.donor, bond.acceptor,
                         bond.donor_antecedent, bond.acceptor_antecedent)
            if spec is not None and spec.site == "ligand"
        }
    )
    provenance = identity_provenance(
        design_id="seed", scaffold_id="5fqj", structure=structure,
        motif=cfg.target.motif, ligand_atoms=ligand_atoms,
    )
    measurements = {
        m.bond_id: m
        for m in measure_all(
            cfg.target.hbonds, structure, provenance,
            cfg.pipeline.scoring.angle.rule,
        )
    }

    base = measurements["lig_n3_to_asp229_od2"]
    assert base.distance == pytest.approx(2.734, abs=0.005)
    assert base.angle == pytest.approx(106.6, abs=0.1)

    donor = measurements["his291_ne2_to_lig_n1"]
    assert donor.distance == pytest.approx(3.084, abs=0.005)
    assert donor.angle == pytest.approx(126.6, abs=0.1)

    # Each bond's documented ideal is the value measured here, so the seed
    # itself scores 1.0 on the angle term by construction.
    for bond in cfg.target.hbonds:
        assert bond.acceptor_ideal_angle == pytest.approx(
            measurements[bond.id].angle, abs=0.1
        )


@pytest.mark.skipif(not SEED_PDB.exists(), reason="seed PDB not in this checkout")
def test_identity_provenance_rejects_a_motif_absent_from_the_structure():
    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=9999)])
    structure = read_structure(SEED_PDB, "6NT")
    with pytest.raises(MotifTraceError, match="absent"):
        identity_provenance(
            design_id="x", scaffold_id="s", structure=structure,
            motif=motif, ligand_atoms=[],
        )


# --------------------------------------------------------------------------
# the CIF reader, against a real Boltz-2 prediction
# --------------------------------------------------------------------------

BOLTZ_CIF = _BOLTZ_CIF


@pytest.mark.skipif(not BOLTZ_CIF.exists(), reason="run data not in this checkout")
def test_reads_a_real_boltz_prediction():
    """Boltz-2 writes mmCIF, not PDB, so this is the format the scorer sees."""
    structure = read_structure(BOLTZ_CIF, "6NT")
    assert len(structure.residues) == 260
    assert sorted({c for c, _ in structure.residues}) == ["A"]
    assert len(structure.sequence("A")) == 260


@pytest.mark.skipif(not BOLTZ_CIF.exists(), reason="run data not in this checkout")
def test_boltz_numbers_its_output_from_one():
    """The reason hal_to_boltz_map exists: Boltz renumbers 1..N per chain, so
    an offset of zero must never be assumed."""
    structure = read_structure(BOLTZ_CIF, "6NT")
    numbers = structure.residue_numbers("A")
    assert numbers[0] == 1
    assert numbers[-1] == 260


@pytest.mark.skipif(not BOLTZ_CIF.exists(), reason="run data not in this checkout")
def test_ccd_ligand_atom_names_survive_into_the_prediction():
    """The CCD path is preferred precisely because the names resolve directly;
    this is the check that they do."""
    structure = read_structure(BOLTZ_CIF, "6NT")
    for atom in ("N1", "N2", "N3", "O11", "O21"):
        assert atom in structure.ligand, f"{atom} missing from the predicted ligand"
    assert structure.ligand_resname == "6NT"


@pytest.mark.skipif(not BOLTZ_CIF.exists(), reason="run data not in this checkout")
def test_the_predicted_ligand_has_the_heavy_atom_count_the_token_guard_uses():
    """boltz.KNOWN_LIGAND_ATOM_COUNTS['6NT'] is 12. If that is wrong the token
    budget is wrong, so it is checked against a real prediction."""
    from src.stages.boltz import KNOWN_LIGAND_ATOM_COUNTS

    structure = read_structure(BOLTZ_CIF, "6NT")
    assert len(structure.ligand) == KNOWN_LIGAND_ATOM_COUNTS["6NT"]


@pytest.mark.skipif(not SEED_PDB.exists(), reason="seed PDB not in this checkout")
def test_the_seed_ligand_has_the_same_heavy_atom_count():
    from src.stages.boltz import KNOWN_LIGAND_ATOM_COUNTS

    structure = read_structure(SEED_PDB, "6NT")
    assert len(structure.ligand) == KNOWN_LIGAND_ATOM_COUNTS["6NT"]


@pytest.mark.skipif(not BOLTZ_CIF.exists(), reason="run data not in this checkout")
def test_resolving_seed_numbering_against_a_boltz_output_needs_the_map():
    """Without the real map from the .trb, a seed-numbered atom must not
    silently resolve to whatever sits at that index in the prediction."""
    structure = read_structure(BOLTZ_CIF, "6NT")
    provenance = Provenance(design_id="d", scaffold_id="s")   # no maps at all
    with pytest.raises(AtomResolutionError, match="seed_to_hal"):
        provenance.resolve(AtomRef("protein", "OD2", "A", 229), structure)


def test_cif_reader_survives_a_loop_after_the_atom_site_loop(tmp_path):
    """Regression: a `loop_` following the atom_site data used to reset the
    column list, and the rows already collected were discarded. CIF files put
    their loops in no guaranteed order."""
    path = tmp_path / "trailing.cif"
    path.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "CA ALA A 1 1.000 2.000 3.000\n"
        "CB ALA A 1 2.000 3.000 4.000\n"
        "#\n"
        "loop_\n"
        "_something_else.key\n"
        "_something_else.value\n"
        "a b\n",
        encoding="utf-8",
    )
    structure = read_structure(path)
    assert structure.residues == {("A", 1): "ALA"}
    np.testing.assert_allclose(structure.atoms[("A", 1, "CA")], [1.0, 2.0, 3.0])


def test_cif_reader_skips_an_unrelated_loop_before_atom_site(tmp_path):
    path = tmp_path / "leading.cif"
    path.write_text(
        "data_test\n"
        "loop_\n"
        "_entity.id\n"
        "_entity.type\n"
        "1 polymer\n"
        "2 non-polymer\n"
        "#\n"
        "loop_\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "CA GLY A 7 0.000 0.000 0.000\n",
        encoding="utf-8",
    )
    structure = read_structure(path)
    assert structure.residues == {("A", 7): "GLY"}


def test_cif_without_an_atom_site_loop_raises(tmp_path):
    path = tmp_path / "none.cif"
    path.write_text("data_test\nloop_\n_entity.id\n1\n", encoding="utf-8")
    with pytest.raises(TrackingError, match="no _atom_site loop"):
        read_structure(path)


def test_cif_missing_required_columns_raises(tmp_path):
    path = tmp_path / "partial.cif"
    path.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "CA ALA\n",
        encoding="utf-8",
    )
    with pytest.raises(TrackingError, match="lacks"):
        read_structure(path)

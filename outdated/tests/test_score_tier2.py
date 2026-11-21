"""Tier-2 metric tests on the toy enzyme.

The reference structure scored against itself is the ground truth: RMSD 0,
TM-score 1, every catalytic geometry exactly at its reference value. A
deliberately perturbed copy must move those numbers in the right direction.
"""
import numpy as np
import pytest
import biotite.structure as struc

from src.io.structure import normalise, read_structure, write_structure
from src.score import structure as st

from .conftest import TOY_PDB


@pytest.fixture(scope="module")
def toy_atoms():
    atoms, _ = normalise(read_structure(TOY_PDB), chain="A")
    return atoms


@pytest.fixture()
def plddt_pdb(tmp_path, toy_atoms):
    """The toy structure with a synthetic pLDDT profile in the B-factor column.

    Residues 1-20 get 50.0, the rest 90.0, so active-site and global means are
    hand-checkable.
    """
    a = toy_atoms.copy()
    a.b_factor = np.where(a.res_id <= 20, 50.0, 90.0)
    p = tmp_path / "plddt.pdb"
    write_structure(a, p)
    return p


# ---------- confidence ----------

def test_plddt_means_are_hand_checkable(plddt_pdb):
    atoms, _ = normalise(read_structure(plddt_pdb), chain="A")
    resi, vals = st.per_residue_plddt(atoms)
    assert len(resi) == 129
    assert vals[:20].tolist() == [50.0] * 20
    expected = (20 * 50.0 + 109 * 90.0) / 129
    assert st.mean_plddt(atoms) == pytest.approx(expected)
    assert st.plddt_at(atoms, [35, 52]) == pytest.approx(90.0)
    assert st.plddt_at(atoms, [1, 2, 3]) == pytest.approx(50.0)
    assert np.isnan(st.plddt_at(atoms, [9999]))


def test_plddt_rescales_0_1_scale(tmp_path, toy_atoms):
    a = toy_atoms.copy()
    a.b_factor = np.full(a.array_length(), 0.85)
    p = tmp_path / "frac.pdb"
    write_structure(a, p)
    atoms, _ = normalise(read_structure(p), chain="A")
    assert st.mean_plddt(atoms) == pytest.approx(85.0, abs=0.5)


def test_plddt_refuses_empty_bfactors(tmp_path, toy_atoms):
    a = toy_atoms.copy()
    a.b_factor = np.zeros(a.array_length())
    p = tmp_path / "zero.pdb"
    write_structure(a, p)
    atoms, _ = normalise(read_structure(p), chain="A")
    with pytest.raises(ValueError, match="no pLDDT"):
        st.mean_plddt(atoms)


def test_active_site_shell(toy_atoms):
    site = st.active_site_positions(toy_atoms, [35, 52], shell=8.0)
    assert 35 in site and 52 in site
    assert 10 < len(site) < 129
    assert st.active_site_positions(toy_atoms, [], 8.0) == []
    assert len(st.active_site_positions(toy_atoms, [35, 52], 4.0)) < len(site)


# ---------- PAE ----------

def test_pae_between_picks_worst_direction():
    pae = np.zeros((129, 129))
    pae[34, 51] = 3.0
    pae[51, 34] = 11.0          # asymmetric: the worse direction must win
    out = st.pae_between(pae, [35, 52])
    assert out["catalytic_pae_max"] == pytest.approx(11.0)
    assert out["catalytic_pae_worst_pair"] == "35-52"


def test_pae_absent_is_nan_not_zero():
    out = st.pae_between(None, [35, 52])
    assert np.isnan(out["catalytic_pae_max"])


def test_pae_rejects_out_of_range_positions():
    with pytest.raises(ValueError, match="outside PAE matrix"):
        st.pae_between(np.zeros((40, 40)), [35, 52])


def test_pae_roundtrip_npz(tmp_path):
    pae = np.random.default_rng(0).random((20, 20))
    np.savez(tmp_path / "x.pae.npz", pae=pae)
    assert np.allclose(st.load_pae(tmp_path / "x.pae.npz"), pae)
    assert st.load_pae(tmp_path / "missing.pae.npz") is None


# ---------- catalytic geometry: the enzyme-specific metric ----------

def test_geometry_of_reference_against_itself(toy_ref):
    """Measured values reproduce the reference exactly.

    The deviation is NOT zero: it is measured against the *configured* ideal
    (7.05 in target.example.yaml), while the structure has 7.0455. That 0.0045
    gap is the rounding in the config, and the metric reporting it is correct —
    deviation is from the ideal we claim to want, not from whatever we measured.
    """
    out = st.catalytic_geometry(toy_ref.atoms, toy_ref)
    g = toy_ref.geometry[0]
    assert out["geom_acid_nucleophile"] == pytest.approx(g["reference_value"], abs=1e-6)
    assert out["geom_acid_nucleophile_dev"] == pytest.approx(
        abs(g["reference_value"] - g["ideal"]), abs=1e-9)
    assert out["catalytic_geometry_worst_dev"] < 0.01
    assert out["catalytic_geometry_fraction_pass"] == 1.0
    assert out["catalytic_geometry_missing"] == ""


def test_geometry_detects_a_pulled_apart_active_site(toy_ref):
    """Pull Asp52 4 A directly away from Glu35: the distance must catch it."""
    from src.io.structure import get_atom

    axis = get_atom(toy_ref.atoms, 52, "OD1") - get_atom(toy_ref.atoms, 35, "OE1")
    axis = axis / np.linalg.norm(axis)
    moved = toy_ref.atoms.copy()
    sel = moved.res_id == 52
    moved.coord[sel] = moved.coord[sel] + 4.0 * axis
    out = st.catalytic_geometry(moved, toy_ref)
    assert out["geom_acid_nucleophile"] == pytest.approx(
        toy_ref.geometry[0]["reference_value"] + 4.0, abs=0.05)
    assert out["geom_acid_nucleophile_dev"] > 1.5          # beyond tol=1.5
    assert out["geom_acid_nucleophile_pass"] == 0
    assert out["catalytic_geometry_fraction_pass"] < 1.0


def test_geometry_angle_term_catches_a_sideways_shift(toy_ref):
    """A shift that barely changes the distance still breaks the carboxylate angle.

    This is why the config carries more than one measurement: a single distance
    is easy to satisfy with a geometrically wrong active site.
    """
    moved = toy_ref.atoms.copy()
    sel = moved.res_id == 52
    moved.coord[sel] = moved.coord[sel] + np.array([4.0, 0.0, 0.0])
    out = st.catalytic_geometry(moved, toy_ref)
    assert out["geom_acid_nucleophile"] < toy_ref.geometry[0]["reference_value"] + 0.5
    assert out["geom_carboxylate_angle_dev"] > 5.0
    assert out["catalytic_geometry_worst_dev"] > 5.0


def test_geometry_missing_atoms_are_nan_and_counted_failed(toy_ref):
    stripped = toy_ref.atoms[~((toy_ref.atoms.res_id == 52) & (toy_ref.atoms.atom_name == "OD1"))]
    out = st.catalytic_geometry(stripped, toy_ref)
    assert np.isnan(out["geom_acid_nucleophile"])
    assert "acid_nucleophile" in out["catalytic_geometry_missing"]
    assert out["catalytic_geometry_fraction_pass"] < 1.0   # counted, not skipped


def test_geometry_tolerance_scale(toy_ref):
    moved = toy_ref.atoms.copy()
    sel = moved.res_id == 52
    moved.coord[sel] = moved.coord[sel] + np.array([1.8, 0.0, 0.0])
    strict = st.catalytic_geometry(moved, toy_ref, tolerance_scale=1.0)
    loose = st.catalytic_geometry(moved, toy_ref, tolerance_scale=3.0)
    assert loose["catalytic_geometry_n_pass"] >= strict["catalytic_geometry_n_pass"]


# ---------- self-consistency ----------

def test_self_consistency_identity(toy_ref, toy_atoms):
    out = st.self_consistency(toy_atoms, toy_ref.atoms)
    assert out["self_consistency_rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert out["self_consistency_tmscore"] == pytest.approx(1.0, abs=1e-3)


def test_self_consistency_invariant_to_rigid_motion(toy_ref, toy_atoms):
    moved = toy_atoms.copy()
    angle = np.pi / 3
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    moved.coord = moved.coord @ rot.T + np.array([12.0, -5.0, 3.0])
    out = st.self_consistency(moved, toy_ref.atoms)
    assert out["self_consistency_rmsd"] == pytest.approx(0.0, abs=1e-2)
    assert out["self_consistency_tmscore"] == pytest.approx(1.0, abs=1e-2)


def test_self_consistency_detects_distortion(toy_ref, toy_atoms):
    rng = np.random.default_rng(0)
    noisy = toy_atoms.copy()
    noisy.coord = noisy.coord + rng.normal(0, 2.0, noisy.coord.shape)
    out = st.self_consistency(noisy, toy_ref.atoms)
    assert out["self_consistency_rmsd"] > 1.0
    assert out["self_consistency_tmscore"] < 1.0


def test_active_site_rmsd_zero_for_identity(toy_ref, toy_atoms):
    site = st.active_site_positions(toy_atoms, [35, 52], 8.0)
    assert st.active_site_rmsd(toy_atoms, toy_ref.atoms, site) == pytest.approx(0.0, abs=1e-3)


# ---------- surface and cysteines ----------

def test_sasa_toy(toy_atoms):
    out = st.sasa_metrics(toy_atoms)
    # Hen lysozyme (129 aa) has roughly 6000-7500 A^2 of accessible surface.
    assert 5000 < out["sasa_total"] < 9000
    assert out["sasa_hydrophobic_exposed"] < out["sasa_total"]
    assert out["aggregation_flag"] in (0, 1)
    assert out["hydrophobic_patch_area"] >= 0


def test_structural_cysteines_toy(toy_atoms):
    out = st.unpaired_cysteines_structural(toy_atoms)
    assert out["n_cys_structural"] == 8
    assert out["n_disulfides"] == 4          # hen lysozyme has four disulfide bonds
    assert out["n_unpaired_cys_structural"] == 0


# ---------- ligand ----------

def test_ligand_absent_is_reported_not_invented(toy_ref, toy_atoms):
    out = st.ligand_pose_metrics(toy_atoms, toy_ref.atoms, [35, 52])
    assert out["ligand_present"] == 0
    assert np.isnan(out["ligand_rmsd"])
    assert out["ligand_catalytic_contacts"] == 0


def test_ligand_contacts_counted(toy_ref, toy_atoms):
    """Place a synthetic 2-atom ligand next to Glu35 and check it is detected."""
    # Place the ligand along the outward direction from the Glu35 side-chain
    # centroid, so it sits in front of OE1 rather than inside the residue.
    glu = toy_atoms[toy_atoms.res_id == 35]
    oe1 = toy_atoms[(toy_atoms.res_id == 35) & (toy_atoms.atom_name == "OE1")].coord[0]
    outward = oe1 - glu.coord.mean(axis=0)
    outward = outward / np.linalg.norm(outward)
    lig = struc.AtomArray(2)
    lig.coord = np.array([oe1 + 2.5 * outward, oe1 + 3.5 * outward])
    lig.chain_id = np.array(["A", "A"])
    lig.res_id = np.array([200, 200])
    lig.res_name = np.array(["LIG", "LIG"])
    lig.atom_name = np.array(["C1", "C2"])
    lig.element = np.array(["C", "C"])
    lig.hetero = np.array([True, True])
    lig.b_factor = np.array([0.0, 0.0])
    lig.occupancy = np.array([1.0, 1.0])
    lig.ins_code = np.array(["", ""])
    combined = toy_atoms + lig
    out = st.ligand_pose_metrics(combined, None, [35, 52], ligand_resn="LIG")
    assert out["ligand_present"] == 1
    # the nearest catalytic atom is OE1 itself, 2.5 A from the first ligand atom
    assert out["ligand_min_catalytic_distance"] == pytest.approx(2.5, abs=0.2)
    assert out["ligand_catalytic_contacts"] >= 1
    assert out["ligand_buried_fraction"] > 0


# ---------- foldseek refuses to fake it ----------

def test_structural_novelty_unavailable_without_db(tmp_path):
    with pytest.raises((st.FoldseekUnavailable, RuntimeError)):
        st.structural_novelty(tmp_path, None)


# ---------- the aggregate row ----------

def test_all_structure_metrics_on_reference(plddt_pdb, toy_ref):
    row = st.all_structure_metrics(plddt_pdb, toy_ref)
    assert row["self_consistency_rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert row["catalytic_geometry_worst_dev"] < 0.01      # config rounding only, see above
    assert row["mean_plddt"] > 80
    assert row["active_site_plddt"] == pytest.approx(90.0)
    assert row["ligand_present"] == 0
    assert "pae:not produced" in row["tier2_notes"]    # honest about what is missing
    assert np.isnan(row["catalytic_pae_max"])

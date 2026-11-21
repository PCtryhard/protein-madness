"""The official scorer must reproduce the ORGANISERS' numbers, not ours.

The fixture in ``tests/data/boltz_reference/`` is the real output of the
organisers' own verification run on the AWS cluster (job 1, Kemp example,
260 residues, 6NT bound), together with the ``summary.json`` their
``verify_boltz.py`` wrote from it. Every expected value below is read out of
that file rather than typed in, so this test asserts agreement with *them* and
cannot drift into asserting agreement with ourselves.

If this test fails, our ranking is fiction and nothing downstream - selection,
the evolution loop, the leaderboard - means anything. It is the gate.

Residue positions come from the same summary: in that 260-residue design the
catalytic base ASP (reference A127) is at position 177 and the H-bond donor GLN
(reference A50) is at position 99.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.score.official import (
    BURIAL_RADIUS_A,
    CatalyticSite,
    OfficialScore,
    aggregate_seeds,
    base_burial,
    find_prediction,
    read_complex,
    score_run,
)

REFERENCE = Path(__file__).parent / "data" / "boltz_reference"
BASE_POSITION = 177
DONOR_POSITION = 99


@pytest.fixture(scope="module")
def organiser_summary() -> dict:
    return json.loads((REFERENCE / "summary.json").read_text())


@pytest.fixture(scope="module")
def site() -> CatalyticSite:
    return CatalyticSite(base_position=BASE_POSITION, donor_position=DONOR_POSITION)


@pytest.fixture(scope="module")
def scored(site) -> OfficialScore:
    return score_run(REFERENCE, site)


def test_positions_match_the_organisers_bridge(organiser_summary):
    """Our hardcoded positions are the ones their pipeline actually mapped."""
    fixed = {entry["original"]: entry["position"] for entry in organiser_summary["rfd3"]["fixed"]}
    assert fixed["A127"] == BASE_POSITION
    assert fixed["A50"] == DONOR_POSITION


def test_ipde_matches(scored, organiser_summary):
    """Metric 1. iPDE is Boltz's complex_ipde and nothing else."""
    assert scored.ipde == pytest.approx(
        organiser_summary["boltz"]["confidence"]["complex_ipde"], abs=1e-9
    )


def test_geometry_matches_both_contacts(scored, organiser_summary):
    """Metric 2, measured with THEIR atom names: OD2->N3 and NE2->N1."""
    contacts = organiser_summary["boltz"]["contacts_angstrom"]
    assert scored.base_to_ligand_a == pytest.approx(
        contacts["A127_OD2_to_6NT_N3"], abs=1e-6
    )
    assert scored.donor_to_ligand_a == pytest.approx(
        contacts["A50_NE2_to_6NT_N1"], abs=1e-6
    )


def test_packing_matches(scored, organiser_summary):
    boltz = organiser_summary["boltz"]
    assert scored.min_ligand_protein_a == pytest.approx(
        boltz["minimum_ligand_protein_distance_angstrom"], abs=1e-6
    )
    assert scored.clash_pairs == boltz["ligand_protein_pairs_below_2_2_angstrom"]


def test_supporting_confidence_matches(scored, organiser_summary):
    conf = organiser_summary["boltz"]["confidence"]
    assert scored.iptm == pytest.approx(conf["iptm"], abs=1e-9)
    assert scored.complex_plddt == pytest.approx(conf["complex_plddt"], abs=1e-9)
    assert scored.confidence_score == pytest.approx(conf["confidence_score"], abs=1e-9)


def test_length_and_no_warnings(scored):
    assert scored.n_residues == 260
    assert scored.notes == [], f"unexpected notes: {scored.notes}"


# ------------------------------------------------------------------ burial

def test_burial_is_a_plausible_count(scored):
    """Metric 3. The organisers' summary does not expose it, so we assert the
    properties that must hold rather than a number we cannot cross-check."""
    assert scored.base_burial_ca_count >= 0
    assert scored.base_burial_ca_count < scored.n_residues


def test_burial_excludes_the_base_itself(site):
    """A residue is not in its own neighbourhood count."""
    atoms = read_complex(find_prediction(REFERENCE)[0])
    tiny = base_burial(atoms, BASE_POSITION, radius=0.1)
    assert tiny == 0


def test_burial_grows_with_radius(site):
    atoms = read_complex(find_prediction(REFERENCE)[0])
    small = base_burial(atoms, BASE_POSITION, radius=6.0)
    large = base_burial(atoms, BASE_POSITION, radius=BURIAL_RADIUS_A)
    assert 0 <= small < large


def test_out_of_range_position_raises():
    atoms = read_complex(find_prediction(REFERENCE)[0])
    with pytest.raises(IndexError):
        base_burial(atoms, 10_000)


# ------------------------------------------------------------- consistency

def _fake(ipde: float, base: float) -> OfficialScore:
    return OfficialScore(
        n_residues=260, ipde=ipde, base_to_ligand_a=base,
        donor_to_ligand_a=3.0, base_burial_ca_count=30,
    )


def test_aggregate_reports_mean_and_spread():
    """Metric 4. Selection uses mean - k*spread, so both must be present."""
    out = aggregate_seeds([_fake(1.0, 4.0), _fake(3.0, 6.0)])
    assert out["n_seeds"] == 2
    assert out["ipde_mean"] == pytest.approx(2.0)
    assert out["ipde_std"] == pytest.approx(1.0)
    assert out["base_to_ligand_a_mean"] == pytest.approx(5.0)
    # The angle/stability/change terms were never measured on these fakes, and
    # must be REPORTED as missing rather than quietly treated as fine: a design
    # is never rewarded for having been measured less.
    assert set(out["metrics_missing"]) == {
        "base_angle_deg", "donor_angle_deg", "complex_plddt", "design_rmsd_a",
    }


def test_aggregate_ignores_nan_but_records_the_count():
    """A design is never rewarded for having been measured fewer times."""
    out = aggregate_seeds([_fake(1.0, 4.0), _fake(math.nan, 6.0)])
    assert out["ipde_mean"] == pytest.approx(1.0)
    assert out["ipde_n"] == 1
    assert out["n_seeds"] == 2


def test_aggregate_refuses_mismatched_sequences():
    a, b = _fake(1.0, 4.0), _fake(1.0, 4.0)
    b.n_residues = 200
    with pytest.raises(ValueError, match="different sequences"):
        aggregate_seeds([a, b])


def test_aggregate_needs_at_least_one_score():
    with pytest.raises(ValueError):
        aggregate_seeds([])


def test_catalytic_site_rejects_zero_based_positions():
    with pytest.raises(ValueError, match="1-based"):
        CatalyticSite(base_position=0, donor_position=99)


# ------------------------------- angles, stability, and amount of change

def test_approach_angle_is_measured_on_the_reference(scored):
    """The scoring slide says geometry is distance, angle AND dihedral. A
    distance-only score is an incomplete copy of the judges' metric."""
    assert 0.0 <= scored.base_angle_deg <= 180.0
    assert 0.0 <= scored.donor_angle_deg <= 180.0


def test_approach_angle_geometry_is_correct():
    """A right angle must read 90 degrees, whatever the chemistry."""
    import numpy as np
    from src.score.official import approach_angle

    class _Lig:
        atom_name = np.array(["N3"])
        coord = np.array([[0.0, 1.0, 0.0]])
        def __getitem__(self, mask):
            return self
        def array_length(self):
            return 1

    atoms = read_complex(find_prediction(REFERENCE)[0])
    # geometry is exercised directly: root at +x, contact at origin,
    # ligand at +y  ->  90 degrees
    from src.score import official
    saved = official._residue_atom
    official._residue_atom = lambda a, p, n: (
        np.array([1.0, 0.0, 0.0]) if n == "CG" else np.array([0.0, 0.0, 0.0])
    )
    try:
        angle = approach_angle(atoms, 1, "CG", "OD2", _Lig(), "N3")
    finally:
        official._residue_atom = saved
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_backbone_rmsd_is_zero_for_the_same_structure():
    from src.score.official import backbone_rmsd
    atoms = read_complex(find_prediction(REFERENCE)[0])
    # 1e-5, not 1e-9: the Kabsch superposition goes through an SVD, so an
    # identical pair lands on float noise rather than exact zero.
    assert backbone_rmsd(atoms, atoms) == pytest.approx(0.0, abs=1e-5)


def test_consistency_covers_angles_stability_and_change():
    from src.score.official import CONSISTENCY_METRICS
    for metric in ("base_angle_deg", "complex_plddt", "design_rmsd_a"):
        assert metric in CONSISTENCY_METRICS


def test_fitness_weights_distance_above_angle():
    """The mentor's framing, encoded: distance dominates, angle tie-breaks."""
    from src.evolve.select import FitnessWeights
    w = FitnessWeights()
    # 1 A of contact error must cost more than 10 degrees of angle error
    assert w.base_geometry > w.angle * 1.0


def test_fitness_penalises_a_badly_pointed_contact():
    from src.evolve.select import fitness
    good = {"base_to_ligand_a_mean": 3.0, "ipde_mean": 2.0,
            "base_burial_ca_count_mean": 40.0, "base_angle_deg_mean": 120.0}
    bad = dict(good, base_angle_deg_mean=30.0)
    assert fitness(bad) > fitness(good)


def test_fitness_penalises_an_unstable_fold():
    from src.evolve.select import fitness
    stable = {"base_to_ligand_a_mean": 3.0, "ipde_mean": 2.0,
              "base_burial_ca_count_mean": 40.0, "complex_plddt_mean": 0.85}
    floppy = dict(stable, complex_plddt_mean=0.45)
    assert fitness(floppy) > fitness(stable)


def test_fitness_penalises_a_fold_that_changed_a_lot():
    from src.evolve.select import fitness
    faithful = {"base_to_ligand_a_mean": 3.0, "ipde_mean": 2.0,
                "base_burial_ca_count_mean": 40.0, "design_rmsd_a_mean": 1.0}
    drifted = dict(faithful, design_rmsd_a_mean=6.0)
    assert fitness(drifted) > fitness(faithful)


def test_missing_new_terms_do_not_break_scoring():
    """Older aggregates without angles/stability must still rank."""
    from src.evolve.select import fitness
    minimal = {"base_to_ligand_a_mean": 3.0, "ipde_mean": 2.0,
               "base_burial_ca_count_mean": 40.0}
    assert math.isfinite(fitness(minimal))


# ------------------------------------------------- calibration against reality

#: Measured on PDB 4BS0 (HG3.17 + 6NT, 1.09 A) with our own metrics.
#: If a real, highly active Kemp eliminase does not score well here, our
#: ranking of de novo designs is meaningless.
HG317 = {
    "base_to_ligand_a_mean": 2.53, "donor_to_ligand_a_mean": 3.10,
    "base_angle_deg_mean": 130.9, "donor_angle_deg_mean": 133.6,
    "base_burial_ca_count_mean": 47.0, "ipde_mean": 2.0,
}
#: The organisers' own de novo reference design, measured in job 1.
REFERENCE_DESIGN = {
    "base_to_ligand_a_mean": 11.178812, "donor_to_ligand_a_mean": 7.478555,
    "base_burial_ca_count_mean": 30.0, "ipde_mean": 2.5525,
}


def test_a_real_kemp_eliminase_beats_the_reference_design():
    """The sanity check the whole ranking rests on."""
    from src.evolve.select import fitness
    assert fitness(HG317) < fitness(REFERENCE_DESIGN)


def test_the_real_enzyme_pays_no_geometry_penalty_at_all():
    """HG3.17's contacts and angles should sit in the free region. If they do
    not, our 'ideal' is wrong, not the enzyme."""
    from src.evolve.select import FitnessWeights, contact_penalty
    w = FitnessWeights()
    assert contact_penalty(HG317["base_to_ligand_a_mean"]) == 0.0
    assert contact_penalty(HG317["donor_to_ligand_a_mean"]) == 0.0
    for key in ("base_angle_deg_mean", "donor_angle_deg_mean"):
        assert abs(HG317[key] - w.ideal_angle_deg) <= w.angle_tolerance_deg


def test_the_real_enzyme_passes_the_hard_gates():
    from src.evolve.select import passes_gates
    ok, reasons = passes_gates(HG317, "A" * 260)
    assert ok, f"HG3.17 was rejected by our own gates: {reasons}"


def test_burial_target_is_set_from_the_real_enzyme_not_the_failing_design():
    from src.evolve.select import FitnessWeights
    w = FitnessWeights()
    assert w.burial_target >= 45.0
    assert w.burial_target > REFERENCE_DESIGN["base_burial_ca_count_mean"]


def test_knocking_out_the_catalytic_base_is_punished():
    """Move the base out of contact and the design must get clearly worse."""
    from src.evolve.select import fitness
    knockout = dict(HG317, base_to_ligand_a_mean=9.0)
    assert fitness(knockout) > fitness(HG317) + 3.0

"""Phase 2 acceptance: the scorer returns the values computed by hand.

The expected numbers here are derived in tests/fixtures/build_fixtures.py, not
read off a previous run of this code. With the shipped defaults
(distance ideal 2.9, tolerated 3.5, fail 4.5; angle tolerance 60) and the
fixture target's ideal angle of 180:

    2.900 A, 180 deg -> distance term 1.0,       angle term 1.0
    3.400 A, 130 deg -> distance term 1/6,       angle term 1/6
    5.000 A           -> beyond fail, so GATED and the total is 0
"""

from __future__ import annotations

import pytest

from src.scoring.combine import average_replicates, combine
from src.scoring.hbond import (
    HBondGeometryError,
    angle_between,
    angle_term,
    distance_term,
    measure_all,
)
from src.tracking import AtomResolutionError

SIXTH = 1.0 / 6.0

# A PDB file stores coordinates to three decimal places, so a fixture can only
# encode its intended geometry to about 1e-3 A. That propagates: the distance
# term divides by a 0.6 A span, and the angle recovered from rounded
# coordinates drifts by ~0.01 deg. These tolerances are what the file format
# allows, and they are still two orders of magnitude tighter than any genuine
# scoring error, which would move a term by 0.1 or more.
ANGLE_TOL = 0.05        # degrees
DIST_TOL = 1e-3         # angstrom
TERM_TOL = 1e-3         # normalised term


def measure(cfg, loaded):
    structure, provenance = loaded
    return measure_all(
        cfg.target.hbonds, structure, provenance, cfg.pipeline.scoring.angle.rule
    )


# --------------------------------------------------------------------------
# the raw geometry comes out as built
# --------------------------------------------------------------------------


def test_ideal_fixture_has_the_geometry_it_claims(cfg, ideal):
    measurements = measure(cfg, ideal)
    assert len(measurements) == 2
    for m in measurements:
        assert m.distance == pytest.approx(2.900, abs=DIST_TOL)
        assert m.angle == pytest.approx(180.0, abs=ANGLE_TOL)
        assert m.angle_kind == "acceptor_side"
        assert m.angle_ideal == 180.0


def test_stretched_fixture_has_the_geometry_it_claims(cfg, stretched):
    for m in measure(cfg, stretched):
        assert m.distance == pytest.approx(3.400, abs=DIST_TOL)
        assert m.angle == pytest.approx(130.0, abs=ANGLE_TOL)


def test_mixed_fixture_differs_between_its_two_bonds(cfg, mixed):
    by_id = {m.bond_id: m for m in measure(cfg, mixed)}
    assert by_id["lig_n3_to_asp229_od2"].distance == pytest.approx(2.900, abs=DIST_TOL)
    assert by_id["lig_n3_to_asp229_od2"].angle == pytest.approx(180.0, abs=ANGLE_TOL)
    assert by_id["his291_ne2_to_lig_n1"].distance == pytest.approx(3.400, abs=DIST_TOL)
    assert by_id["his291_ne2_to_lig_n1"].angle == pytest.approx(130.0, abs=ANGLE_TOL)


# --------------------------------------------------------------------------
# the normalised terms match the hand calculation
# --------------------------------------------------------------------------


def test_distance_term_hand_values(pipeline):
    cfg = pipeline.scoring.distance
    assert distance_term(2.900, cfg) == pytest.approx(1.0)
    assert distance_term(3.400, cfg) == pytest.approx(SIXTH)
    # symmetric about the ideal: a 2.4 A clash scores the same as a 3.4 A stretch
    assert distance_term(2.400, cfg) == pytest.approx(SIXTH)
    # at and beyond the tolerance it is zero, never negative
    assert distance_term(3.500, cfg) == pytest.approx(0.0)
    assert distance_term(9.000, cfg) == 0.0
    assert distance_term(0.100, cfg) == 0.0


def test_distance_term_is_symmetric(pipeline):
    cfg = pipeline.scoring.distance
    for delta in (0.0, 0.1, 0.25, 0.4, 0.6, 1.0):
        assert distance_term(cfg.ideal + delta, cfg) == pytest.approx(
            distance_term(cfg.ideal - delta, cfg)
        )


def test_angle_term_hand_values(pipeline):
    cfg = pipeline.scoring.angle
    assert angle_term(180.0, 180.0, cfg) == pytest.approx(1.0)
    assert angle_term(130.0, 180.0, cfg) == pytest.approx(SIXTH)
    assert angle_term(120.0, 180.0, cfg) == pytest.approx(0.0)
    assert angle_term(90.0, 180.0, cfg) == 0.0
    # and against a non-180 ideal, which is the acceptor-side case
    assert angle_term(106.6, 106.6, cfg) == pytest.approx(1.0)
    assert angle_term(126.6, 106.6, cfg) == pytest.approx(1.0 - 20.0 / 60.0)


# --------------------------------------------------------------------------
# the combination
# --------------------------------------------------------------------------


def test_ideal_fixture_scores_one_on_geometry(cfg, ideal):
    score = combine(1.0, measure(cfg, ideal), cfg.pipeline.scoring)
    assert score.distance == pytest.approx(1.0)
    assert score.angle == pytest.approx(1.0)
    assert score.geometry == pytest.approx(1.0)
    assert not score.gated
    # S = 0.7*1 + 0.3*(0.6*1 + 0.4*1) = 1.0
    assert score.total == pytest.approx(1.0)


def test_total_is_the_weighted_sum_from_the_spec(cfg, stretched):
    scoring = cfg.pipeline.scoring
    score = combine(0.5, measure(cfg, stretched), scoring)

    geometry = (
        scoring.geometry_weights.distance * SIXTH
        + scoring.geometry_weights.angle * SIXTH
    )
    expected = (
        scoring.weights.confidence * 0.5 + scoring.weights.geometry * geometry
    )
    assert score.geometry == pytest.approx(geometry, abs=TERM_TOL)
    assert score.total == pytest.approx(expected, abs=TERM_TOL)
    # with the shipped 0.7/0.3 and 0.6/0.4: 0.7*0.5 + 0.3*(1/6) = 0.4
    assert score.total == pytest.approx(0.4, abs=TERM_TOL)


def test_confidence_outside_unit_interval_is_rejected(cfg, ideal):
    measurements = measure(cfg, ideal)
    with pytest.raises(ValueError, match="outside"):
        combine(1.4, measurements, cfg.pipeline.scoring)
    with pytest.raises(ValueError, match="outside"):
        combine(-0.1, measurements, cfg.pipeline.scoring)


# --------------------------------------------------------------------------
# both aggregation modes
# --------------------------------------------------------------------------


def test_mean_aggregation(cfg, mixed):
    scoring = cfg.pipeline.scoring.model_copy(update={"aggregate": "mean"})
    score = combine(1.0, measure(cfg, mixed), scoring)
    expected = (1.0 + SIXTH) / 2
    assert score.distance == pytest.approx(expected, abs=TERM_TOL)
    assert score.angle == pytest.approx(expected, abs=TERM_TOL)


def test_min_aggregation_lets_one_bad_bond_dominate(cfg, mixed):
    scoring = cfg.pipeline.scoring.model_copy(update={"aggregate": "min"})
    score = combine(1.0, measure(cfg, mixed), scoring)
    assert score.distance == pytest.approx(SIXTH, abs=TERM_TOL)
    assert score.angle == pytest.approx(SIXTH, abs=TERM_TOL)


def test_min_is_never_above_mean(cfg, mixed):
    measurements = measure(cfg, mixed)
    mean = combine(
        1.0, measurements, cfg.pipeline.scoring.model_copy(update={"aggregate": "mean"})
    )
    minimum = combine(
        1.0, measurements, cfg.pipeline.scoring.model_copy(update={"aggregate": "min"})
    )
    assert minimum.total <= mean.total


def test_unknown_aggregation_raises(cfg, ideal):
    scoring = cfg.pipeline.scoring.model_copy(update={"aggregate": "median"})
    with pytest.raises(ValueError, match="unknown aggregation"):
        combine(1.0, measure(cfg, ideal), scoring)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_gate_fires_on_a_five_angstrom_bond(cfg, gated):
    score = combine(1.0, measure(cfg, gated), cfg.pipeline.scoring)
    assert score.gated
    assert score.gated_bonds == ["lig_n3_to_asp229_od2"]
    assert score.total == 0.0


def test_gated_design_keeps_its_component_terms_for_the_record(cfg, gated):
    """The design is recorded as gated, not deleted, so the search can see
    what it rejected. The components stay readable even though the total is 0."""
    score = combine(1.0, measure(cfg, gated), cfg.pipeline.scoring)
    assert score.total == 0.0
    assert score.confidence == pytest.approx(1.0)
    assert score.per_bond["his291_ne2_to_lig_n1"]["distance_term"] == pytest.approx(1.0)
    assert score.per_bond["lig_n3_to_asp229_od2"]["distance_raw"] == pytest.approx(
        5.0, abs=DIST_TOL
    )


def test_gate_does_not_fire_below_the_threshold(cfg, stretched):
    score = combine(1.0, measure(cfg, stretched), cfg.pipeline.scoring)
    assert not score.gated
    assert score.total > 0.0


def test_a_bond_marked_not_required_cannot_gate(cfg, gated):
    """`required: false` contributes to the score but cannot zero a design."""
    bonds = [
        b.model_copy(update={"required": False}) if b.id == "lig_n3_to_asp229_od2" else b
        for b in cfg.target.hbonds
    ]
    structure, provenance = gated
    measurements = measure_all(
        bonds, structure, provenance, cfg.pipeline.scoring.angle.rule
    )
    score = combine(1.0, measurements, cfg.pipeline.scoring)
    assert not score.gated
    assert score.total > 0.0


# --------------------------------------------------------------------------
# replicate averaging
# --------------------------------------------------------------------------


def test_average_replicates_means_each_term(cfg, ideal, stretched):
    good = combine(1.0, measure(cfg, ideal), cfg.pipeline.scoring)
    poor = combine(0.0, measure(cfg, stretched), cfg.pipeline.scoring)
    averaged = average_replicates([good, poor])
    assert averaged.total == pytest.approx((good.total + poor.total) / 2, abs=TERM_TOL)
    assert averaged.confidence == pytest.approx(0.5)
    assert not averaged.gated


def test_a_design_gated_in_any_replicate_is_gated(cfg, ideal, gated):
    """Averaging a zero from a gated replicate with good scores from others
    would let a design that sometimes loses a required bond outrank one that
    always keeps it."""
    good = combine(1.0, measure(cfg, ideal), cfg.pipeline.scoring)
    bad = combine(1.0, measure(cfg, gated), cfg.pipeline.scoring)
    averaged = average_replicates([good, good, bad])
    assert averaged.gated
    assert averaged.total == 0.0
    assert "lig_n3_to_asp229_od2" in averaged.gated_bonds


def test_average_of_one_is_itself(cfg, ideal):
    score = combine(0.75, measure(cfg, ideal), cfg.pipeline.scoring)
    assert average_replicates([score]) is score


def test_average_of_nothing_raises():
    with pytest.raises(ValueError):
        average_replicates([])


# --------------------------------------------------------------------------
# the angle primitive
# --------------------------------------------------------------------------


def test_angle_between_known_values():
    import numpy as np

    origin = np.array([0.0, 0.0, 0.0])
    assert angle_between(
        np.array([1.0, 0.0, 0.0]), origin, np.array([-1.0, 0.0, 0.0])
    ) == pytest.approx(180.0)
    assert angle_between(
        np.array([1.0, 0.0, 0.0]), origin, np.array([0.0, 1.0, 0.0])
    ) == pytest.approx(90.0)
    assert angle_between(
        np.array([1.0, 0.0, 0.0]), origin, np.array([1.0, 1.0, 0.0])
    ) == pytest.approx(45.0)


def test_angle_between_coincident_atoms_raises():
    import numpy as np

    origin = np.array([0.0, 0.0, 0.0])
    with pytest.raises(HBondGeometryError, match="coincident"):
        angle_between(origin, origin, np.array([1.0, 0.0, 0.0]))


# --------------------------------------------------------------------------
# resolution failures raise, never default
# --------------------------------------------------------------------------


def test_a_missing_ligand_atom_raises_rather_than_defaulting(cfg, ideal):
    """A missing atom scored as zero distance looks like a perfect bond, and as
    infinite distance looks like a failed design. Both corrupt the search."""
    structure, provenance = ideal
    structure.ligand.pop("N3")
    with pytest.raises(AtomResolutionError, match="N3"):
        measure(cfg, (structure, provenance))


def test_a_missing_protein_atom_raises(cfg, ideal):
    structure, provenance = ideal
    del structure.atoms[("A", 229, "OD2")]
    with pytest.raises(AtomResolutionError, match="OD2"):
        measure(cfg, (structure, provenance))


def test_an_unmapped_ligand_atom_raises(cfg, ideal):
    structure, provenance = ideal
    provenance.ligand_atom_map.pop("N1")
    with pytest.raises(AtomResolutionError, match="ligand_atom_map"):
        measure(cfg, (structure, provenance))


def test_an_unmapped_residue_raises(cfg, ideal):
    structure, provenance = ideal
    provenance.seed_to_hal.pop(("A", 229))
    with pytest.raises(AtomResolutionError, match="seed_to_hal"):
        measure(cfg, (structure, provenance))

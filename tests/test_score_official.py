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
    assert out["metrics_missing"] == []


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

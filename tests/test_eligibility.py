"""Eligibility: the rules that zero a design regardless of its score.

The property that matters most here is that **"not checked" never looks like
"passed"**. A novelty metric that quietly returns 0.0 when it could not search
would tell us a design is novel exactly when we failed to look.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.score.eligibility import (
    MAX_LENGTH,
    MAX_SEQUENCE_IDENTITY,
    MAX_TM_SCORE,
    ToolUnavailable,
    assess,
    check_length,
    max_sequence_identity,
    max_tm_score,
)

SEQ = "ACDEFGHIKL" * 20  # 200 aa


def test_length_within_limit():
    ok, reason = check_length(SEQ)
    assert ok and reason == ""


def test_length_over_limit_is_rejected():
    ok, reason = check_length("A" * (MAX_LENGTH + 1))
    assert not ok and "length" in reason


def test_exactly_the_limit_is_allowed():
    """260 is allowed; 261 is not. Off-by-one here would bin every design."""
    assert check_length("A" * MAX_LENGTH)[0]
    assert not check_length("A" * (MAX_LENGTH + 1))[0]


def test_eligible_design_passes_everything():
    report = assess("d1", SEQ, identity=0.12, tm=0.35)
    assert report.eligible
    assert report.reasons == []
    assert report.unchecked == []


def test_high_identity_disqualifies():
    report = assess("d1", SEQ, identity=0.85, tm=0.3)
    assert not report.eligible
    assert any("identity" in r for r in report.reasons)


def test_high_tm_disqualifies():
    report = assess("d1", SEQ, identity=0.1, tm=0.92)
    assert not report.eligible
    assert any("TM-score" in r for r in report.reasons)


def test_thresholds_are_exclusive_not_inclusive():
    """'more than 70%' does not qualify - exactly 70% still does."""
    assert assess("d", SEQ, identity=MAX_SEQUENCE_IDENTITY, tm=MAX_TM_SCORE).eligible
    assert not assess("d", SEQ, identity=MAX_SEQUENCE_IDENTITY + 0.01, tm=0.1).eligible
    assert not assess("d", SEQ, identity=0.1, tm=MAX_TM_SCORE + 0.01).eligible


def test_unchecked_rules_are_recorded_not_assumed_passed():
    """The whole point: missing a check is not the same as passing it."""
    report = assess("d1", SEQ)
    assert report.unchecked == ["sequence identity", "TM-score"]
    assert math.isnan(report.sequence_identity)
    assert math.isnan(report.tm_score)


def test_too_long_and_unchecked_still_fails_on_length():
    report = assess("d1", "A" * 300)
    assert not report.eligible
    assert report.unchecked  # and we still know what we did not check


def test_report_serialises_for_the_leaderboard():
    row = assess("d1", SEQ, identity=0.2, tm=0.4).as_dict()
    assert row["design_id"] == "d1"
    assert row["length"] == len(SEQ)
    assert row["eligible"] is True


# ------------------------------------------------------- missing tooling

def test_missing_binary_raises_rather_than_returning_zero():
    """A fabricated 0.0 would read as 'perfectly novel'. Raise instead."""
    with pytest.raises(ToolUnavailable, match="mmseqs"):
        max_sequence_identity({"d": SEQ}, mmseqs="/nonexistent/mmseqs")


def test_missing_foldseek_raises(tmp_path):
    structure = tmp_path / "x.pdb"
    structure.write_text("ATOM\n")
    with pytest.raises(ToolUnavailable, match="foldseek"):
        max_tm_score(structure, foldseek="/nonexistent/foldseek")


def test_empty_query_set_is_not_an_error():
    assert max_sequence_identity({}) == {}


# ------------------------------------------------- TM-score as a hard gate

def test_tm_gate_kills_a_design_in_selection():
    """Measured: 2 of our first 8 backbones failed at 0.703 / 0.705."""
    from src.evolve.select import HardGates, passes_gates

    aggregate = {
        "base_to_ligand_a_mean": 4.0, "ipde_mean": 2.0,
        "base_burial_ca_count_mean": 59.0,     # the BEST burial we measured...
        "tm_score": 0.703, "tm_closest": "3I1C",
    }
    ok, reasons = passes_gates(aggregate, SEQ)
    assert not ok, "a TM-score of 0.703 must not survive selection"
    assert any("TM-score" in r and "3I1C" in r for r in reasons)
    assert HardGates().max_tm_score == 0.70


def test_tm_gate_opposes_burial_and_wins():
    """The two best-buried backbones were the two ineligible ones, so the gate
    must not be tradeable against a good burial score."""
    from src.evolve.select import passes_gates

    great_but_ineligible = {"base_to_ligand_a_mean": 3.0, "ipde_mean": 1.0,
                            "base_burial_ca_count_mean": 59.0, "tm_score": 0.705}
    worse_but_eligible = {"base_to_ligand_a_mean": 5.0, "ipde_mean": 2.5,
                          "base_burial_ca_count_mean": 40.0, "tm_score": 0.499}
    assert not passes_gates(great_but_ineligible, SEQ)[0]
    assert passes_gates(worse_but_eligible, SEQ)[0]


def test_unmeasured_tm_does_not_block_the_loop_but_is_not_a_pass():
    """During evolution an unmeasured TM-score cannot kill a design -- we would
    never fold anything. It must still be visible as unchecked at submission."""
    from src.evolve.select import passes_gates

    aggregate = {"base_to_ligand_a_mean": 4.0, "ipde_mean": 2.0,
                 "base_burial_ca_count_mean": 40.0}
    assert passes_gates(aggregate, SEQ)[0]
    assert assess("d1", SEQ, identity=0.0).unchecked == ["TM-score"]


def test_tm_at_exactly_the_limit_still_qualifies():
    from src.evolve.select import passes_gates
    aggregate = {"base_to_ligand_a_mean": 4.0, "ipde_mean": 2.0,
                 "base_burial_ca_count_mean": 40.0, "tm_score": 0.70}
    assert passes_gates(aggregate, SEQ)[0]


def test_reference_set_is_defined_by_search_not_hardcoded_ids():
    """Auditable: anyone can re-run the queries and get the same set."""
    from src.score.eligibility import KEMP_SEARCH_TERMS
    assert "Kemp eliminase" in KEMP_SEARCH_TERMS
    assert all(isinstance(t, str) and t for t in KEMP_SEARCH_TERMS)


def test_max_tm_vs_empty_set_raises():
    from src.score.eligibility import max_tm_vs_set
    with pytest.raises(ToolUnavailable, match="empty reference set"):
        max_tm_vs_set(Path("x.pdb"), [])

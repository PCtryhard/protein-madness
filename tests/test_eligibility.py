"""Eligibility: the rules that zero a design regardless of its score.

The property that matters most here is that **"not checked" never looks like
"passed"**. A novelty metric that quietly returns 0.0 when it could not search
would tell us a design is novel exactly when we failed to look.
"""
from __future__ import annotations

import math

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

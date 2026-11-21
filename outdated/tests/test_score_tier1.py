"""Tier-1 metric tests. Expected values are hardcoded for the toy enzyme or
for hand-checkable synthetic sequences."""
import math

import pytest

from src.ids import apply_mutations, config_hash, design_id, mutation_string
from src.score import novelty as nov
from src.score.sequence import (all_sequence_metrics, catalytic_integrity, charge_at_ph,
                                cysteine_count, fraction_charged, gravy, homopolymer_runs,
                                isoelectric_point, low_complexity_fraction, max_homopolymer_run,
                                n_mutations, sequence_identity, shannon_entropy,
                                unpaired_cysteines)

from .conftest import TOY_SEQ


# ---------- hand-checkable values ----------

def test_gravy_known_values():
    assert gravy("I") == pytest.approx(4.5)
    assert gravy("IR") == pytest.approx((4.5 - 4.5) / 2)
    # Toy enzyme, cross-checked against Bio.SeqUtils.ProtParam.ProteinAnalysis.gravy()
    assert gravy(TOY_SEQ) == pytest.approx(-0.472093, abs=1e-6)


def test_charge_and_pi_toy():
    # Cross-checked against Bio.SeqUtils.IsoelectricPoint (EMBOSS pKa set).
    assert charge_at_ph(TOY_SEQ, 7.0) == pytest.approx(7.7724, abs=1e-3)
    assert isoelectric_point(TOY_SEQ) == pytest.approx(9.3238, abs=1e-3)
    # Lysozyme is a basic protein: pI well above 7 and positive at pH 7.
    assert isoelectric_point(TOY_SEQ) > 7.0


def test_charge_monotonic_in_ph():
    vals = [charge_at_ph(TOY_SEQ, ph) for ph in (3, 5, 7, 9, 11, 13)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_charge_of_extremes():
    assert charge_at_ph("KKKKK", 7.0) > 4.0
    assert charge_at_ph("DDDDD", 7.0) < -4.0
    assert isoelectric_point("KKKKK") > 10.0
    assert isoelectric_point("DDDDD") < 4.5


def test_identity_and_mutations():
    assert sequence_identity("ACDE", "ACDE") == 1.0
    assert sequence_identity("ACDE", "ACDF") == pytest.approx(0.75)
    assert n_mutations("ACDE", "ACDF") == 1
    with pytest.raises(ValueError, match="equal lengths"):
        sequence_identity("ACD", "ACDE")


def test_homopolymer_and_complexity():
    assert max_homopolymer_run("AAABBBBCA") == 4
    assert max_homopolymer_run("ABCDE") == 1
    assert max_homopolymer_run("") == 0
    assert homopolymer_runs("AAABBBBCA", min_run=3) == [(1, "A", 3), (4, "B", 4)]
    assert low_complexity_fraction("AAAAAAAAAA", window=10) == 1.0
    assert low_complexity_fraction("ACDEFGHIKL", window=10) == pytest.approx(0.1)
    # a pure window anywhere in an otherwise diverse sequence is caught
    assert low_complexity_fraction("ACDEFGHIKL" + "Q" * 10 + "ACDEFGHIKL", window=10) == 1.0


def test_entropy_bounds():
    assert shannon_entropy("AAAA") == pytest.approx(0.0)
    assert shannon_entropy("ACGT") == pytest.approx(2.0)
    assert 3.5 < shannon_entropy(TOY_SEQ) < math.log2(20)


def test_cysteines():
    assert cysteine_count(TOY_SEQ) == 8            # 4 disulfides in hen lysozyme
    assert unpaired_cysteines(TOY_SEQ) == 0
    assert unpaired_cysteines("CCC") == 1
    assert fraction_charged("DEKR") == 1.0
    assert fraction_charged("AAAA") == 0.0


# ---------- catalytic integrity: the metric that kills a design ----------

def test_catalytic_integrity_toy(toy_ref):
    intact = catalytic_integrity(TOY_SEQ, toy_ref.catalytic)
    assert intact["catalytic_integrity"] == 1.0
    assert intact["n_catalytic"] == 2
    assert intact["catalytic_lost"] == ""
    assert intact["catalytic_observed"] == "ED"


def test_catalytic_integrity_detects_loss(toy_ref):
    broken = apply_mutations(TOY_SEQ, "E35A")
    out = catalytic_integrity(broken, toy_ref.catalytic)
    assert out["catalytic_integrity"] == pytest.approx(0.5)
    assert out["catalytic_lost"] == "GLU35A"

    both = apply_mutations(TOY_SEQ, "E35A+D52N")
    assert catalytic_integrity(both, toy_ref.catalytic)["catalytic_integrity"] == 0.0


def test_catalytic_integrity_truncated_design(toy_ref):
    out = catalytic_integrity(TOY_SEQ[:40], toy_ref.catalytic)
    assert out["catalytic_integrity"] == pytest.approx(0.5)
    assert "ASP52>len" in out["catalytic_lost"]


def test_catalytic_integrity_without_config_is_nan():
    out = catalytic_integrity(TOY_SEQ, [])
    assert math.isnan(out["catalytic_integrity"])    # never silently 1.0


# ---------- the aggregate row ----------

def test_all_sequence_metrics_toy(toy_ref):
    row = all_sequence_metrics(TOY_SEQ, parent=TOY_SEQ, catalytic=toy_ref.catalytic)
    assert row["length"] == 129
    assert row["sequence_identity"] == 1.0
    assert row["n_mutations"] == 0
    assert row["catalytic_integrity"] == 1.0
    assert row["unpaired_cys_delta"] == 0
    assert row["net_charge_delta"] == pytest.approx(0.0)
    assert row["gravy"] == pytest.approx(-0.472093, abs=1e-6)
    # no NaNs anywhere in a fully specified row
    assert not [k for k, v in row.items() if isinstance(v, float) and math.isnan(v)]


def test_all_sequence_metrics_tracks_a_mutant(toy_ref):
    mutant = apply_mutations(TOY_SEQ, "K1D")
    row = all_sequence_metrics(mutant, parent=TOY_SEQ, catalytic=toy_ref.catalytic)
    assert row["n_mutations"] == 1
    assert row["sequence_identity"] == pytest.approx(128 / 129)
    assert row["net_charge_delta"] < -1.0          # K->D removes + and adds -
    assert row["catalytic_integrity"] == 1.0


# ---------- ids and provenance ----------

def test_design_id_deterministic_and_sensitive():
    h = config_hash({"a": 1})
    assert design_id(TOY_SEQ, "mutate", h) == design_id(TOY_SEQ, "mutate", h)
    assert design_id(TOY_SEQ, "mutate", h) != design_id(TOY_SEQ, "proteinmpnn", h)
    assert design_id(TOY_SEQ, "mutate", h) != design_id(TOY_SEQ, "mutate", config_hash({"a": 2}))
    assert design_id(TOY_SEQ, "mutate", h) != design_id(TOY_SEQ[:-1], "mutate", h)
    assert design_id(TOY_SEQ, "mutate", h).startswith("d_")
    with pytest.raises(ValueError):
        design_id("", "mutate", h)


def test_config_hash_insensitive_to_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_mutation_string_roundtrip():
    child = apply_mutations(TOY_SEQ, "K1D+E35A")
    assert mutation_string(TOY_SEQ, child) == "K1D+E35A"
    assert apply_mutations(TOY_SEQ, "") == TOY_SEQ
    with pytest.raises(ValueError, match="parent has"):
        apply_mutations(TOY_SEQ, "A1C")      # position 1 is K, not A
    with pytest.raises(ValueError, match="outside"):
        apply_mutations(TOY_SEQ, "K999D")


# ---------- novelty refuses to fake it ----------

def test_novelty_unavailable_without_db():
    with pytest.raises(nov.NoveltyUnavailable):
        nov.check_available(None)


def test_novelty_vs_parent_is_labelled_separately():
    assert nov.novelty_vs_parent(TOY_SEQ, TOY_SEQ) == 0.0
    assert nov.novelty_vs_parent(apply_mutations(TOY_SEQ, "K1D"), TOY_SEQ) == pytest.approx(1 / 129)

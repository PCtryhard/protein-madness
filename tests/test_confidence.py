"""The confidence term, checked against a real Boltz-2 output.

`data/No_mutation_run/` holds confidence sidecars from the earlier runs on this
target. Parsing a real file rather than a hand-written one is the point: it is
the only way to know the reader matches what Boltz actually writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import ConfidenceCfg
from src.scoring.confidence import (
    ConfidenceError,
    logged_metrics,
    motif_pae,
    normalise,
    read_confidence_json,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = (
    ROOT / "data" / "No_mutation_run" / "boltz" / "seed-1" / "5A_01"
    / "confidence_5A_01_model_0.json"
)


# --------------------------------------------------------------------------
# against the real file
# --------------------------------------------------------------------------


@pytest.mark.skipif(not REAL.exists(), reason="run data not in this checkout")
def test_reads_a_real_boltz_confidence_file():
    raw = read_confidence_json(REAL)
    assert raw["complex_ipde"] == pytest.approx(2.7573986, abs=1e-6)
    assert raw["iptm"] == pytest.approx(0.70962006, abs=1e-6)
    assert raw["ligand_iptm"] == pytest.approx(0.70962006, abs=1e-6)
    assert raw["complex_plddt"] == pytest.approx(0.78489292, abs=1e-6)


@pytest.mark.skipif(not REAL.exists(), reason="run data not in this checkout")
def test_nested_chain_metrics_are_flattened():
    raw = read_confidence_json(REAL)
    assert raw["chains_ptm.0"] == pytest.approx(0.8372982, abs=1e-6)


@pytest.mark.skipif(not REAL.exists(), reason="run data not in this checkout")
def test_logged_metrics_are_the_documented_set():
    """These are logged for every design without being scored on, so the
    weighting can be revisited offline without recomputing anything."""
    logged = logged_metrics(read_confidence_json(REAL))
    for key in ("iptm", "ptm", "ligand_iptm", "complex_plddt", "complex_ipde"):
        assert key in logged


@pytest.mark.skipif(not REAL.exists(), reason="run data not in this checkout")
def test_real_file_normalises_to_the_expected_term():
    """complex_ipde 2.757 against the shipped ramp (best 0.5, worst 5.0):
    (5.0 - 2.7574) / (5.0 - 0.5) = 0.49835..."""
    raw = read_confidence_json(REAL)
    cfg = ConfidenceCfg()
    assert normalise(raw, cfg) == pytest.approx((5.0 - 2.7573986) / 4.5, abs=1e-6)


# --------------------------------------------------------------------------
# the ramp
# --------------------------------------------------------------------------


def test_ipde_ramp_endpoints():
    cfg = ConfidenceCfg(metric="complex_ipde", ipde_best=0.5, ipde_worst=5.0)
    assert normalise({"complex_ipde": 0.5}, cfg) == pytest.approx(1.0)
    assert normalise({"complex_ipde": 5.0}, cfg) == pytest.approx(0.0)
    assert normalise({"complex_ipde": 2.75}, cfg) == pytest.approx(0.5)


def test_ipde_ramp_clips_rather_than_going_negative():
    cfg = ConfidenceCfg(metric="complex_ipde")
    assert normalise({"complex_ipde": 99.0}, cfg) == 0.0
    assert normalise({"complex_ipde": 0.0}, cfg) == 1.0


def test_iptm_passes_through_unchanged():
    """Already 0..1 with higher better; rescaling would invent a range the
    model does not have."""
    cfg = ConfidenceCfg(metric="iptm")
    assert normalise({"iptm": 0.71}, cfg) == pytest.approx(0.71)


def test_ligand_iptm_passes_through():
    cfg = ConfidenceCfg(metric="ligand_iptm")
    assert normalise({"ligand_iptm": 0.42}, cfg) == pytest.approx(0.42)


def test_an_iptm_outside_the_unit_interval_is_rejected():
    cfg = ConfidenceCfg(metric="iptm")
    with pytest.raises(ConfidenceError, match="cannot be a confidence probability"):
        normalise({"iptm": 1.4}, cfg)


# --------------------------------------------------------------------------
# missing metrics raise rather than defaulting
# --------------------------------------------------------------------------


def test_a_missing_metric_raises():
    """Scoring a design on a metric that was not produced would be a
    fabricated number."""
    cfg = ConfidenceCfg(metric="complex_ipde")
    with pytest.raises(ConfidenceError, match="absent from the Boltz-2 output"):
        normalise({"iptm": 0.5}, cfg)


def test_the_error_lists_what_was_there():
    cfg = ConfidenceCfg(metric="complex_ipde")
    with pytest.raises(ConfidenceError, match="iptm"):
        normalise({"iptm": 0.5}, cfg)


def test_motif_pae_metric_without_a_value_raises():
    cfg = ConfidenceCfg(metric="motif_pae")
    with pytest.raises(ConfidenceError, match="no motif PAE"):
        normalise({"complex_ipde": 1.0}, cfg)


def test_motif_pae_metric_uses_the_supplied_value():
    cfg = ConfidenceCfg(metric="motif_pae", ipde_best=0.5, ipde_worst=5.0)
    assert normalise({}, cfg, motif_pae=0.5) == pytest.approx(1.0)
    assert normalise({}, cfg, motif_pae=5.0) == pytest.approx(0.0)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(ConfidenceError, match="not found"):
        read_confidence_json(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfidenceError, match="not valid JSON"):
        read_confidence_json(path)


def test_a_file_with_no_numbers_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"note": "nothing numeric"}), encoding="utf-8")
    with pytest.raises(ConfidenceError, match="no numeric"):
        read_confidence_json(path)


def test_booleans_are_not_mistaken_for_numbers(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"flag": True, "iptm": 0.5}), encoding="utf-8")
    raw = read_confidence_json(path)
    assert "flag" not in raw
    assert raw["iptm"] == 0.5


# --------------------------------------------------------------------------
# motif PAE
# --------------------------------------------------------------------------


def test_motif_pae_averages_both_directions():
    """PAE is not symmetric, so both blocks are averaged."""
    pae = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [5.0, 6.0, 0.0],
        ]
    )
    # motif = token 0, ligand = token 2: pae[0,2]=2, pae[2,0]=5 -> 3.5
    assert motif_pae(pae, [0], [2]) == pytest.approx(3.5)


def test_motif_pae_over_several_tokens():
    pae = np.full((4, 4), 2.0)
    assert motif_pae(pae, [0, 1], [2, 3]) == pytest.approx(2.0)


def test_motif_pae_rejects_a_non_square_matrix():
    with pytest.raises(ConfidenceError, match="not square"):
        motif_pae(np.zeros((2, 3)), [0], [1])


def test_motif_pae_rejects_out_of_range_indices():
    with pytest.raises(ConfidenceError, match="outside the PAE matrix"):
        motif_pae(np.zeros((3, 3)), [0], [9])


def test_motif_pae_rejects_empty_index_lists():
    with pytest.raises(ConfidenceError, match="at least one"):
        motif_pae(np.zeros((3, 3)), [], [1])

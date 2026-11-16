"""The CLI contract and the leaderboard join.

These are the two things every module depends on: if the contract breaks, six
people's stages stop composing, and if the join breaks, the one table we
present from is wrong.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import cli
from src.report import fake_data
from src.report.leaderboard import apply_filters, join_stages, markdown_table, rank

from .conftest import PIPELINE, ROOT, TOY_TARGET

PY = sys.executable


def run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", module, *args], cwd=ROOT, capture_output=True, text=True)


# ---------- inputs ----------

def test_load_inputs_accepts_fasta_structure_and_directory(tmp_path):
    from src.io.fasta import SeqRecord, write_fasta
    from src.io.structure import normalise, read_structure, write_structure

    fasta = tmp_path / "x.fasta"
    write_fasta([SeqRecord("d_1", "ACDEF"), SeqRecord("d_2", "GHIKL")], fasta)
    inp = cli.load_inputs(fasta)
    assert inp.kind == "fasta" and inp.design_ids == ["d_1", "d_2"] and len(inp) == 2
    assert cli.load_inputs(fasta, limit=1).design_ids == ["d_1"]

    atoms, _ = normalise(read_structure(ROOT / "data" / "toy" / "ref.pdb"), chain="A")
    sdir = tmp_path / "structs"
    sdir.mkdir()
    write_structure(atoms, sdir / "d_9.pdb")
    write_structure(atoms, sdir / "d_8.cif")
    dir_in = cli.load_inputs(sdir)
    assert dir_in.kind == "structures" and sorted(dir_in.design_ids) == ["d_8", "d_9"]
    assert cli.load_inputs(sdir / "d_9.pdb").design_ids == ["d_9"]


def test_load_inputs_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        cli.load_inputs(tmp_path / "nope.fasta")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .* files"):
        cli.load_inputs(empty)


# ---------- the stage contract, end to end through a real subprocess ----------

def test_stage_writes_csv_and_meta_and_caches(tmp_path):
    out = tmp_path / "mutate"
    r = run_module("src.generate.mutate", "--in", "data/toy/ref.fasta", "--out", str(out),
                   "--target", str(TOY_TARGET), "--max-designs", "12")
    assert r.returncode == 0, r.stderr

    csv_path = out / "generate.mutate.csv"
    meta_path = out / "generate.mutate.meta.json"
    assert csv_path.exists() and meta_path.exists()

    df = pd.read_csv(csv_path)
    assert df.columns[0] == "design_id"                 # contract: first column
    assert not df["design_id"].duplicated().any()
    assert len(df) == 12

    meta = json.loads(meta_path.read_text())
    for key in ("tool", "version", "git_sha", "config_hash", "backend", "wall_seconds",
                "command", "argv", "timestamp_utc", "status"):
        assert key in meta, key
    assert meta["tool"] == "generate.mutate"
    assert meta["status"] == "ok"
    assert meta["n_rows"] == 12
    assert "src.generate.mutate" in meta["command"]

    # second run with the same config is a cached no-op
    r2 = run_module("src.generate.mutate", "--in", "data/toy/ref.fasta", "--out", str(out),
                    "--target", str(TOY_TARGET), "--max-designs", "12")
    assert r2.returncode == 0
    assert "cached" in (r2.stderr + r2.stdout)
    assert json.loads(meta_path.read_text())["wall_seconds"] == meta["wall_seconds"]

    # --force recomputes
    r3 = run_module("src.generate.mutate", "--in", "data/toy/ref.fasta", "--out", str(out),
                    "--target", str(TOY_TARGET), "--max-designs", "12", "--force")
    assert r3.returncode == 0 and "cached" not in (r3.stderr + r3.stdout)


def test_changing_config_invalidates_the_cache(tmp_path):
    out = tmp_path / "m"
    a = run_module("src.generate.mutate", "--in", "data/toy/ref.fasta", "--out", str(out),
                   "--target", str(TOY_TARGET), "--max-designs", "8")
    assert a.returncode == 0
    h1 = json.loads((out / "generate.mutate.meta.json").read_text())["config_hash"]
    b = run_module("src.generate.mutate", "--in", "data/toy/ref.fasta", "--out", str(out),
                   "--target", str(TOY_TARGET), "--max-designs", "9")
    assert b.returncode == 0 and "cached" not in (b.stderr + b.stdout)
    assert json.loads((out / "generate.mutate.meta.json").read_text())["config_hash"] != h1


def test_failed_stage_records_failure_in_meta(tmp_path):
    """A stage that dies must leave evidence, not a stale 'ok' meta."""
    out = tmp_path / "t1"
    r = run_module("src.score.tier1", "--in", "data/toy/ref.pdb", "--out", str(out),
                   "--target", str(TOY_TARGET))
    assert r.returncode != 0
    assert "must be a FASTA file" in (r.stderr + r.stdout)


def test_tool_config_merges_defaults():
    cfg = cli.load_pipeline_config(PIPELINE)
    merged = cli.tool_config(cfg, "generate.mutate")
    assert merged["seed"] == 0                      # from defaults
    assert merged["mode"] == "single"               # from the tool section
    assert cli.tool_config(cfg, "nonexistent.tool")["device"] == "cpu"


def test_missing_target_message_is_actionable(tmp_path):
    r = run_module("src.score.tier1", "--in", "data/toy/ref.fasta", "--out", str(tmp_path / "o"))
    assert "--target" in (r.stderr + r.stdout)


# ---------- the leaderboard join ----------

@pytest.fixture()
def stage_csvs(tmp_path):
    a = tmp_path / "stage_a"
    b = tmp_path / "stage_b"
    a.mkdir()
    b.mkdir()
    pd.DataFrame({"design_id": ["d_1", "d_2", "d_3"], "method": ["m", "m", "m"],
                  "catalytic_integrity": [1.0, 1.0, 0.5],
                  "max_homopolymer_run": [2, 3, 2]}).to_csv(a / "score.tier1.csv", index=False)
    # d_3 was never folded; d_4 exists only here (folded but not tier-1 scored)
    pd.DataFrame({"design_id": ["d_1", "d_2", "d_4"],
                  "mean_plddt": [92.0, 61.0, 88.0],
                  "active_site_plddt": [95.0, 55.0, 80.0]}).to_csv(b / "score.tier2.csv",
                                                                   index=False)
    return tmp_path


def test_join_is_outer_and_keeps_every_design(stage_csvs):
    from src.report.leaderboard import find_stage_csvs

    df = join_stages(find_stage_csvs(stage_csvs))
    assert sorted(df["design_id"]) == ["d_1", "d_2", "d_3", "d_4"]
    assert pd.isna(df.set_index("design_id").loc["d_3", "mean_plddt"])     # not folded -> NaN
    assert pd.isna(df.set_index("design_id").loc["d_4", "catalytic_integrity"])
    assert set(df.set_index("design_id").loc["d_1", "stages"].split(",")) == {
        "score.tier1", "score.tier2"}


def test_conflicting_columns_are_kept_not_silently_overwritten(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pd.DataFrame({"design_id": ["d_1"], "mean_plddt": [90.0]}).to_csv(a / "fold.esmfold.csv",
                                                                     index=False)
    pd.DataFrame({"design_id": ["d_1"], "mean_plddt": [40.0]}).to_csv(b / "fold.boltz.csv",
                                                                     index=False)
    from src.report.leaderboard import find_stage_csvs

    df = join_stages(find_stage_csvs(tmp_path))
    cols = [c for c in df.columns if c.startswith("mean_plddt")]
    assert len(cols) == 2, f"a disagreement between stages must stay visible, got {cols}"


def test_hard_filter_kills_a_design_and_soft_does_not(stage_csvs):
    from src.report.leaderboard import find_stage_csvs

    df = join_stages(find_stage_csvs(stage_csvs))
    df = apply_filters(df, [
        {"metric": "catalytic_integrity", "op": "==", "value": 1.0, "hard": True},
        {"metric": "mean_plddt", "op": ">=", "value": 70.0, "hard": False},
    ])
    s = df.set_index("design_id")
    assert s.loc["d_1", "pass_hard"] and s.loc["d_1", "pass_soft"]
    assert s.loc["d_2", "pass_hard"] and not s.loc["d_2", "pass_soft"]   # low pLDDT, soft only
    assert not s.loc["d_3", "pass_hard"]                                  # lost a catalytic residue
    assert "catalytic_integrity==1.0" in s.loc["d_3", "filters_failed"]
    # d_4 has no catalytic_integrity value: missing must fail a hard filter, not pass it
    assert not s.loc["d_4", "pass_hard"]


def test_filter_on_absent_metric_is_skipped_not_fatal(stage_csvs):
    from src.report.leaderboard import find_stage_csvs

    df = join_stages(find_stage_csvs(stage_csvs))
    out = apply_filters(df, [{"metric": "not_a_real_metric", "op": ">=", "value": 1, "hard": True}])
    assert out["pass_hard"].all()


def test_ranking_prefers_better_designs_and_flags_missing_metrics(stage_csvs):
    from src.report.leaderboard import find_stage_csvs

    df = join_stages(find_stage_csvs(stage_csvs))
    df = apply_filters(df, [{"metric": "catalytic_integrity", "op": "==", "value": 1.0,
                             "hard": True}])
    df = rank(df, [{"metric": "mean_plddt", "weight": 1.0, "direction": "higher"},
                   {"metric": "active_site_plddt", "weight": 1.5, "direction": "higher"}])
    s = df.set_index("design_id")
    assert s.loc["d_1", "score"] > s.loc["d_2", "score"]
    assert s.loc["d_1", "rank"] == 1
    assert s.loc["d_3", "rank"] > s.loc["d_2", "rank"]       # hard-failed designs sort last
    assert s.loc["d_3", "score_metrics_missing"] == 2


def test_ranking_direction_lower_is_better():
    df = pd.DataFrame({"design_id": ["a", "b"], "rmsd": [0.5, 3.0], "pass_hard": [True, True]})
    out = rank(df, [{"metric": "rmsd", "weight": 1.0, "direction": "lower"}])
    assert out.set_index("design_id").loc["a", "rank"] == 1


def test_ranking_with_no_available_terms_is_flagged(stage_csvs, caplog):
    from src.report.leaderboard import find_stage_csvs

    df = join_stages(find_stage_csvs(stage_csvs))
    df = apply_filters(df, [])
    out = rank(df, [{"metric": "seq_novelty", "weight": 1.0, "direction": "higher"}])
    assert (out["score"] == 0).all()
    assert any("no ranking terms available" in r.message for r in caplog.records)


def test_markdown_table_renders_present_columns_only(stage_csvs):
    from src.report.leaderboard import find_stage_csvs

    df = rank(apply_filters(join_stages(find_stage_csvs(stage_csvs)), []), [])
    md = markdown_table(df)
    assert md.startswith("| rank | design_id")
    assert "mean_plddt" in md
    assert "self_consistency_rmsd" not in md       # column absent -> column omitted


# ---------- fake data is unmistakable ----------

def test_fake_data_is_labelled_and_refused_by_pitch_assets(tmp_path):
    df = fake_data.make_fake_leaderboard(n=20, seed=1)
    assert df["is_fake"].all()
    assert (df["method"] == "wildtype").sum() == 1
    assert df["mean_plddt"].isna().any()          # realistic gaps, not a perfect table

    lb = tmp_path / "leaderboard.csv"
    df.to_csv(lb, index=False)
    r = run_module("src.report.pitch_assets", "--in", str(lb), "--out", str(tmp_path / "p"),
                   "--pitch", str(tmp_path / "PITCH.md"))
    assert r.returncode != 0
    assert "Refusing to generate pitch assets from fabricated data" in (r.stdout + r.stderr)

    r2 = run_module("src.report.pitch_assets", "--in", str(lb), "--out", str(tmp_path / "p"),
                    "--pitch", str(tmp_path / "PITCH.md"), "--allow-fake")
    assert r2.returncode == 0


def test_fake_data_is_deterministic():
    a = fake_data.make_fake_leaderboard(n=30, seed=7)
    b = fake_data.make_fake_leaderboard(n=30, seed=7)
    assert list(a["design_id"]) == list(b["design_id"])
    assert np.allclose(a["mean_plddt"].fillna(-1), b["mean_plddt"].fillna(-1))

"""The sqlite store: lineage, the sequence-hash cache, and the leaderboard.

The cache is the one thing in the build whose absence would be felt most, so
the hit/miss counter the phase-3 acceptance test asks for is verified here.
"""

from __future__ import annotations

import json

import pytest

from src.config import BoltzCfg, Ligand
from src.store import (
    DesignRow,
    DesignStore,
    candidates_from_rows,
    new_design_id,
    sequence_hash,
)


@pytest.fixture
def store(tmp_path):
    with DesignStore(tmp_path / "designs.db") as s:
        yield s


def row(design_id: str, **kwargs) -> DesignRow:
    defaults = dict(
        scaffold_id="t.seed_a",
        cycle=0,
        branch="seed",
        sequence="MKVL",
        seq_hash="hash-" + design_id,
        status="scored",
        provenance=json.dumps({"design_id": design_id}),
        score=0.5,
        term_confidence=0.6,
        term_distance=0.4,
        term_angle=0.3,
    )
    defaults.update(kwargs)
    return DesignRow(design_id=design_id, **defaults)


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------


def test_insert_and_get(store):
    store.insert(row("d1"))
    got = store.get("d1")
    assert got is not None
    assert got.design_id == "d1"
    assert got.score == pytest.approx(0.5)
    assert got.created_at  # stamped automatically


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_duplicate_design_id_is_an_error_not_an_overwrite(store):
    store.insert(row("d1"))
    with pytest.raises(ValueError, match="already in"):
        store.insert(row("d1", score=0.99))
    assert store.get("d1").score == pytest.approx(0.5)


def test_store_reopens(tmp_path):
    path = tmp_path / "designs.db"
    with DesignStore(path) as s:
        s.insert(row("d1"))
    with DesignStore(path) as s:
        assert s.get("d1") is not None


def test_counts_by_status(store):
    store.insert(row("a"))
    store.insert(row("b", status="gated", gated=True, score=0.0))
    store.insert(row("c", status="errored", score=None, error="boom"))
    assert store.counts_by_status() == {"scored": 1, "gated": 1, "errored": 1}


# --------------------------------------------------------------------------
# lineage
# --------------------------------------------------------------------------


def test_ancestry_returns_the_full_line(store):
    store.insert(row("c0", cycle=0))
    store.insert(row("c1", cycle=1, parent_id="c0", branch="wiggle"))
    store.insert(row("c2", cycle=2, parent_id="c1", branch="mutate",
                     mutation="A2V"))
    line = store.ancestry("c2")
    assert [r.design_id for r in line] == ["c0", "c1", "c2"]


def test_ancestry_of_a_root_is_itself(store):
    store.insert(row("root"))
    assert [r.design_id for r in store.ancestry("root")] == ["root"]


def test_ancestry_ignores_unrelated_designs(store):
    store.insert(row("a", cycle=0))
    store.insert(row("b", cycle=0))
    store.insert(row("a1", cycle=1, parent_id="a"))
    assert [r.design_id for r in store.ancestry("a1")] == ["a", "a1"]


def test_the_tree_is_never_pruned(store):
    """Every cycle's best design and its full ancestry are kept; only the
    frontier is pruned."""
    store.insert(row("c0", cycle=0, score=0.1))
    for cycle in range(1, 6):
        store.insert(
            row(f"c{cycle}", cycle=cycle, parent_id=f"c{cycle - 1}", score=0.1 * cycle)
        )
    assert len(store.by_scaffold("t.seed_a")) == 6
    assert len(store.ancestry("c5")) == 6


# --------------------------------------------------------------------------
# the leaderboard and the frontier rebuild
# --------------------------------------------------------------------------


def test_leaderboard_orders_by_score(store):
    store.insert(row("low", score=0.1))
    store.insert(row("high", score=0.9))
    store.insert(row("mid", score=0.5))
    assert [r.design_id for r in store.leaderboard("t.seed_a")] == [
        "high", "mid", "low"
    ]


def test_leaderboard_excludes_gated_and_errored(store):
    store.insert(row("ok", score=0.5))
    store.insert(row("gated", status="gated", gated=True, score=0.0))
    store.insert(row("errored", status="errored", score=None))
    assert [r.design_id for r in store.leaderboard("t.seed_a")] == ["ok"]


def test_leaderboard_is_per_scaffold(store):
    """No cross-scaffold competition and no shared frontier."""
    store.insert(row("a", scaffold_id="t.seed_a", score=0.9))
    store.insert(row("b", scaffold_id="t.seed_b", score=0.1))
    assert [r.design_id for r in store.leaderboard("t.seed_a")] == ["a"]
    assert [r.design_id for r in store.leaderboard("t.seed_b")] == ["b"]
    assert store.scaffolds() == ["t.seed_a", "t.seed_b"]


def test_frontier_at_rebuilds_the_best_n(store):
    store.insert(row("a", cycle=0, score=0.3))
    store.insert(row("b", cycle=1, score=0.9))
    store.insert(row("c", cycle=2, score=0.5))
    assert [r.design_id for r in store.frontier_at("t.seed_a", 1, 2)] == ["b", "a"]
    assert store.max_cycle("t.seed_a") == 2


def test_max_cycle_of_an_unknown_scaffold(store):
    assert store.max_cycle("nope") == -1


def test_candidates_from_rows_skips_unscored(store):
    rows = [row("a", score=0.5), row("b", score=None, status="errored")]
    candidates = candidates_from_rows(rows)
    assert [c.design_id for c in candidates] == ["a"]


def test_to_candidate_on_an_errored_row_raises():
    with pytest.raises(ValueError, match="cannot enter the search"):
        row("x", score=None, status="errored").to_candidate()


# --------------------------------------------------------------------------
# the sequence hash
# --------------------------------------------------------------------------


def ligand() -> Ligand:
    return Ligand(id="6NT", ccd="6NT")


def test_hash_is_stable_for_the_same_inputs():
    cfg = BoltzCfg()
    assert sequence_hash("MKVL", ligand(), cfg) == sequence_hash("MKVL", ligand(), cfg)


def test_hash_changes_with_the_sequence():
    cfg = BoltzCfg()
    assert sequence_hash("MKVL", ligand(), cfg) != sequence_hash("MKVM", ligand(), cfg)


def test_hash_changes_with_the_ligand():
    cfg = BoltzCfg()
    other = Ligand(id="LIG", ccd="ATP")
    assert sequence_hash("MKVL", ligand(), cfg) != sequence_hash("MKVL", other, cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("recycling_steps", 5),
        ("diffusion_samples", 3),
        ("sampling_steps", 50),
        ("use_potentials", True),
        ("precision", "16"),
        ("no_kernels", False),
    ],
)
def test_hash_changes_with_any_boltz_setting_that_changes_the_numbers(field, value):
    """A cache hit must mean 'this exact computation was already done'."""
    base = BoltzCfg()
    changed = base.model_copy(update={field: value})
    assert sequence_hash("MKVL", ligand(), base) != sequence_hash(
        "MKVL", ligand(), changed
    )


def test_hash_ignores_settings_that_do_not_change_the_prediction():
    base = BoltzCfg()
    changed = base.model_copy(update={"max_tokens": 999, "cache_dir": "/elsewhere"})
    assert sequence_hash("MKVL", ligand(), base) == sequence_hash(
        "MKVL", ligand(), changed
    )


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


def test_cache_miss_then_hit(store):
    assert store.cache_lookup("h1") is None
    store.insert(row("d1", seq_hash="h1"))
    hit = store.cache_lookup("h1")
    assert hit is not None and hit.design_id == "d1"
    assert store.stats() == {"cache_miss": 1, "cache_hit": 1}


def test_cache_counter_is_what_the_phase_3_acceptance_test_reads(store):
    store.insert(row("d1", seq_hash="h1"))
    for _ in range(3):
        store.cache_lookup("h1")
    store.cache_lookup("unseen")
    stats = store.stats()
    assert stats["cache_hit"] == 3
    assert stats["cache_miss"] == 1


def test_a_gated_result_is_a_cache_hit(store):
    """Gated is a real, reproducible answer for that sequence."""
    store.insert(row("g", seq_hash="h1", status="gated", gated=True, score=0.0))
    hit = store.cache_lookup("h1")
    assert hit is not None and hit.gated


def test_an_errored_result_is_not_a_cache_hit(store):
    """The error may have been a transient OOM; reusing it would make one
    failure permanent for every design that shares the sequence."""
    store.insert(row("e", seq_hash="h1", status="errored", score=None, error="OOM"))
    assert store.cache_lookup("h1") is None


def test_cache_returns_the_oldest_matching_row(store):
    store.insert(row("first", seq_hash="h1", score=0.4,
                     created_at="2026-11-21T10:00:00+00:00"))
    store.insert(row("second", seq_hash="h1", score=0.4,
                     created_at="2026-11-21T11:00:00+00:00"))
    assert store.cache_lookup("h1").design_id == "first"


# --------------------------------------------------------------------------
# design ids
# --------------------------------------------------------------------------


def test_design_id_is_readable_and_sortable():
    design_id = new_design_id("5fqj.seed_5a", 3, "mutate", "parent:A72V")
    assert design_id.startswith("5fqj.seed_5a.c03.mutate.")
    assert len(design_id.rsplit(".", 1)[1]) == 6


def test_design_id_is_stable_for_the_same_discriminator():
    a = new_design_id("s", 1, "wiggle", "x")
    b = new_design_id("s", 1, "wiggle", "x")
    assert a == b


def test_design_id_differs_for_different_discriminators():
    a = new_design_id("s", 1, "mutate", "parent:A1V")
    b = new_design_id("s", 1, "mutate", "parent:A1G")
    assert a != b


def test_cycles_sort_lexicographically():
    ids = [new_design_id("s", c, "seed", "x") for c in (1, 2, 10, 11)]
    assert ids == sorted(ids)

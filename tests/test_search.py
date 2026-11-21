"""The search: elitism, monotonicity, patience, and the two branches.

The property that matters most is that the best score never regresses, because
the previous frontier is always a candidate for the next one. It is asserted
directly, and also as an invariant over a randomised sequence of cycles.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.config import MotifCfg, ResidueRef, RunCfg, SearchCfg, WiggleCfg
from src.search.mutate import (
    MutationError,
    enumerate_mutants,
    shell_positions,
)
from src.search.tree import Candidate, ScaffoldSearch
from src.search.wiggle import WiggleError, backbone_for, jobs_for
from src.tracking import Provenance, read_pdb

ROOT = Path(__file__).resolve().parents[1]


def candidate(design_id: str, score: float, **kwargs) -> Candidate:
    defaults = dict(
        scaffold_id="s", cycle=0, branch="seed", sequence="AAAA",
    )
    defaults.update(kwargs)
    return Candidate(design_id=design_id, score=score, **defaults)


def make_search(**overrides) -> ScaffoldSearch:
    run = RunCfg(**{"patience": 2, "min_delta": 0.005, "max_cycles": 12,
                    **overrides.pop("run", {})})
    search = SearchCfg(**{"keep_per_cycle": 2, **overrides.pop("search", {})})
    return ScaffoldSearch(scaffold_id="s", run=run, search=search)


# --------------------------------------------------------------------------
# the frontier
# --------------------------------------------------------------------------


def test_frontier_keeps_the_best_n():
    s = make_search()
    s.seed_frontier([candidate("a", 0.1), candidate("b", 0.9), candidate("c", 0.5)])
    assert [c.design_id for c in s.frontier] == ["b", "c"]


def test_empty_seed_frontier_raises():
    s = make_search()
    with pytest.raises(ValueError, match="nothing to search from"):
        s.seed_frontier([])


def test_ties_break_deterministically():
    """A reproducible run cannot have ties resolved by dict ordering."""
    s = make_search()
    s.seed_frontier([candidate("z", 0.5), candidate("a", 0.5), candidate("m", 0.5)])
    assert [c.design_id for c in s.frontier] == ["a", "m"]


def test_gated_designs_never_enter_the_frontier():
    s = make_search()
    s.seed_frontier(
        [candidate("good", 0.3), candidate("gated", 0.0, gated=True)]
    )
    assert [c.design_id for c in s.frontier] == ["good"]


def test_previous_frontier_is_always_a_candidate():
    """This is what guarantees the best score never regresses."""
    s = make_search()
    s.seed_frontier([candidate("parent", 0.9)])
    s.advance([candidate("worse", 0.1, cycle=1), candidate("worse2", 0.2, cycle=1)])
    assert "parent" in [c.design_id for c in s.frontier]
    assert s.best_score == pytest.approx(0.9)


def test_best_score_is_monotonic_over_random_cycles():
    rng = random.Random(0)
    s = make_search(run={"max_cycles": 100, "patience": 1000})
    s.seed_frontier([candidate("seed", 0.5)])
    previous = s.best_score
    for cycle in range(1, 30):
        produced = [
            candidate(f"d{cycle}_{i}", rng.uniform(0.0, 1.0), cycle=cycle)
            for i in range(4)
        ]
        s.advance(produced)
        assert s.best_score >= previous
        previous = s.best_score


def test_frontier_dedupes_by_design_id():
    s = make_search()
    parent = candidate("parent", 0.9)
    s.seed_frontier([parent])
    s.advance([parent, candidate("new", 0.95, cycle=1)])
    ids = [c.design_id for c in s.frontier]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# stopping
# --------------------------------------------------------------------------


def test_patience_stops_after_two_flat_cycles():
    s = make_search(run={"patience": 2})
    s.seed_frontier([candidate("seed", 0.5)])
    s.advance([candidate("a", 0.50001, cycle=1)])     # below min_delta
    assert not s.stopped
    s.advance([candidate("b", 0.50002, cycle=2)])
    assert s.stopped
    assert "no improvement" in s.stop_reason


def test_improvement_above_min_delta_resets_patience():
    s = make_search(run={"patience": 2})
    s.seed_frontier([candidate("seed", 0.5)])
    s.advance([candidate("a", 0.5001, cycle=1)])      # noise
    assert s.cycles_without_improvement == 1
    s.advance([candidate("b", 0.6, cycle=2)])         # real gain
    assert s.cycles_without_improvement == 0
    assert not s.stopped


def test_min_delta_means_float_noise_does_not_count_as_improvement():
    s = make_search(run={"patience": 2, "min_delta": 0.005})
    s.seed_frontier([candidate("seed", 0.5)])
    s.advance([candidate("a", 0.5 + 1e-9, cycle=1)])
    s.advance([candidate("b", 0.5 + 2e-9, cycle=2)])
    assert s.stopped


def test_max_cycles_stops_the_search():
    s = make_search(run={"max_cycles": 3, "patience": 99})
    s.seed_frontier([candidate("seed", 0.1)])
    for cycle in range(1, 4):
        s.advance([candidate(f"d{cycle}", 0.1 + cycle, cycle=cycle)])
    assert s.stopped
    assert "max_cycles" in s.stop_reason


def test_advance_after_stopping_raises():
    s = make_search(run={"max_cycles": 1, "patience": 99})
    s.seed_frontier([candidate("seed", 0.1)])
    s.advance([candidate("a", 0.5, cycle=1)])
    assert s.stopped
    with pytest.raises(RuntimeError, match="has stopped"):
        s.advance([candidate("b", 0.9, cycle=2)])


def test_all_gated_stops_the_scaffold():
    s = make_search()
    s.seed_frontier([candidate("seed", 0.4)])
    s.frontier = []
    s.advance([candidate("g", 0.0, cycle=1, gated=True)])
    assert s.stopped


def test_history_records_every_cycle():
    s = make_search(run={"patience": 99})
    s.seed_frontier([candidate("seed", 0.1)])
    s.advance([candidate("a", 0.2, cycle=1)])
    s.advance([candidate("b", 0.3, cycle=2)])
    assert [h["cycle"] for h in s.history] == [1, 2]
    assert s.history[-1]["best_design_id"] == "b"


# --------------------------------------------------------------------------
# parent selection
# --------------------------------------------------------------------------


def test_branch_parents_cycle_a_short_frontier():
    """A frontier of 2 can still feed a branch that wants 4 parents."""
    s = make_search(search={"keep_per_cycle": 2, "wiggle": {"parents_per_cycle": 4}})
    s.seed_frontier([candidate("a", 0.9), candidate("b", 0.8)])
    parents = s.wiggle_parents()
    assert len(parents) == 4
    assert [p.design_id for p in parents] == ["a", "b", "a", "b"]


def test_disabled_branch_asks_for_no_parents():
    s = make_search(search={"wiggle": {"enabled": False}})
    s.seed_frontier([candidate("a", 0.9)])
    assert s.wiggle_parents() == []
    assert s.mutate_parents()


# --------------------------------------------------------------------------
# the wiggle branch
# --------------------------------------------------------------------------


def test_wiggle_uses_the_rfdiffusion_backbone_by_default(tmp_path):
    backbone = tmp_path / "bb.pdb"
    backbone.write_text("ATOM\n", encoding="utf-8")
    parent = candidate("p", 0.5, backbone_path=str(backbone),
                       structure_path="/nonexistent.cif")
    cfg = WiggleCfg(source="rfdiffusion_backbone")
    assert backbone_for(parent, cfg) == str(backbone)


def test_wiggle_can_be_pointed_at_the_boltz_prediction(tmp_path):
    prediction = tmp_path / "pred.cif"
    prediction.write_text("data_x\n", encoding="utf-8")
    parent = candidate("p", 0.5, backbone_path="/nonexistent.pdb",
                       structure_path=str(prediction))
    cfg = WiggleCfg(source="boltz_prediction")
    assert backbone_for(parent, cfg) == str(prediction)


def test_wiggle_never_falls_back_to_the_other_source(tmp_path):
    """A silent fallback would mean half a run used one distribution and half
    the other, making the comparison meaningless."""
    prediction = tmp_path / "pred.cif"
    prediction.write_text("data_x\n", encoding="utf-8")
    parent = candidate("p", 0.5, backbone_path=None,
                       structure_path=str(prediction))
    with pytest.raises(WiggleError, match="no recorded RFdiffusion backbone"):
        backbone_for(parent, WiggleCfg(source="rfdiffusion_backbone"))


def test_wiggle_raises_when_the_recorded_backbone_is_gone():
    parent = candidate("p", 0.5, backbone_path="/definitely/not/here.pdb")
    with pytest.raises(WiggleError, match="does not exist"):
        backbone_for(parent, WiggleCfg(source="rfdiffusion_backbone"))


def test_wiggle_jobs_carry_the_parent_sequence(tmp_path):
    backbone = tmp_path / "bb.pdb"
    backbone.write_text("ATOM\n", encoding="utf-8")
    parent = candidate("p", 0.5, backbone_path=str(backbone), sequence="MKVL")
    jobs = jobs_for([parent], WiggleCfg())
    assert jobs[0].carry_sequence == "MKVL"
    assert jobs[0].parent_id == "p"


# --------------------------------------------------------------------------
# the mutation branch
# --------------------------------------------------------------------------


def test_enumerate_mutants_covers_the_alphabet(pipeline):
    cfg = pipeline.search.mutate.model_copy(
        update={"alphabet": "AG", "max_mutants_per_parent": 100}
    )
    mutants = enumerate_mutants("AAAA", [2, 3], cfg, random.Random(0))
    # 2 positions x (2 letters - the wild type A) = 2 mutants
    assert len(mutants) == 2
    assert {m.label for m in mutants} == {"A2G", "A3G"}
    assert all(len(m.sequence) == 4 for m in mutants)


def test_mutant_sequence_differs_at_exactly_one_position(pipeline):
    cfg = pipeline.search.mutate.model_copy(update={"max_mutants_per_parent": 500})
    parent = "MKVLASTYG"
    for m in enumerate_mutants(parent, [3, 5], cfg, random.Random(0)):
        differences = [
            i for i, (a, b) in enumerate(zip(parent, m.sequence)) if a != b
        ]
        assert differences == [m.position - 1]


def test_sampling_is_without_replacement_and_capped(pipeline):
    cfg = pipeline.search.mutate.model_copy(update={"max_mutants_per_parent": 7})
    mutants = enumerate_mutants("AAAAAAAAAA", list(range(1, 11)), cfg, random.Random(1))
    assert len(mutants) == 7
    labels = [m.label for m in mutants]
    assert len(set(labels)) == 7


def test_sampling_is_not_biased_to_the_start_of_the_chain(pipeline):
    """Truncating in sequence order would bias towards whichever end of the
    chain comes first."""
    cfg = pipeline.search.mutate.model_copy(update={"max_mutants_per_parent": 20})
    positions = list(range(1, 41))
    mutants = enumerate_mutants("A" * 40, positions, cfg, random.Random(3))
    chosen = {m.position for m in mutants}
    assert max(chosen) > 20, "sample never reached the back half of the chain"


def test_position_outside_the_sequence_raises(pipeline):
    with pytest.raises(MutationError, match="outside the parent sequence"):
        enumerate_mutants("AAAA", [9], pipeline.search.mutate, random.Random(0))


def test_sampling_is_reproducible_from_the_seed(pipeline):
    cfg = pipeline.search.mutate.model_copy(update={"max_mutants_per_parent": 5})
    first = enumerate_mutants("A" * 20, list(range(1, 21)), cfg, random.Random(42))
    second = enumerate_mutants("A" * 20, list(range(1, 21)), cfg, random.Random(42))
    assert [m.label for m in first] == [m.label for m in second]


# --------------------------------------------------------------------------
# the shell, on a real structure
# --------------------------------------------------------------------------


def test_shell_excludes_motif_residues(pipeline):
    """Mutable positions are those within shell_radius of any ligand atom,
    excluding every residue in motif.fixed."""
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    motif = MotifCfg(
        fixed=[ResidueRef(chain="A", resnum=229), ResidueRef(chain="A", resnum=291)]
    )
    provenance = Provenance(
        design_id="d", scaffold_id="s",
        seed_to_hal={("A", 229): ("A", 229), ("A", 291): ("A", 291)},
        hal_to_boltz={("A", 229): ("A", 229), ("A", 291): ("A", 291)},
    )
    cfg = pipeline.search.mutate.model_copy(update={"shell_radius": 50.0})
    # Both residues are in the structure and both are motif, so nothing is left.
    with pytest.raises(MutationError, match="no mutable residue"):
        shell_positions(structure, provenance, motif, cfg, chain="A")


def test_shell_finds_a_non_motif_residue_in_range(pipeline):
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=229)])
    provenance = Provenance(
        design_id="d", scaffold_id="s",
        seed_to_hal={("A", 229): ("A", 229)},
        hal_to_boltz={("A", 229): ("A", 229)},
    )
    cfg = pipeline.search.mutate.model_copy(update={"shell_radius": 50.0})
    assert shell_positions(structure, provenance, motif, cfg, chain="A") == [291]


def test_shell_without_a_ligand_raises(pipeline):
    """No ligand means the shell is undefined; guessing would be worse."""
    structure = read_pdb(ROOT / "tests/fixtures/ideal.pdb", ligand_resname="6NT")
    structure.ligand.clear()
    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=229)])
    provenance = Provenance(design_id="d", scaffold_id="s")
    with pytest.raises(MutationError, match="no ligand atoms"):
        shell_positions(structure, provenance, motif, pipeline.search.mutate, "A")


# --------------------------------------------------------------------------
# seed-only runs
# --------------------------------------------------------------------------


def test_max_cycles_zero_stops_after_seeding():
    """The first run you want: score cycle 0, get a leaderboard, expand
    nothing. The runner's loop would otherwise run one whole cycle before the
    stopping condition was consulted."""
    s = make_search(run={"max_cycles": 0})
    s.seed_frontier([candidate("a", 0.4), candidate("b", 0.6)])
    assert s.stopped
    assert "seed-only" in s.stop_reason
    assert s.cycle == 0
    # the frontier is still built, so the leaderboard and report work
    assert [c.design_id for c in s.frontier] == ["b", "a"]
    assert s.best_score == pytest.approx(0.6)


def test_seed_only_refuses_to_advance():
    s = make_search(run={"max_cycles": 0})
    s.seed_frontier([candidate("a", 0.4)])
    with pytest.raises(RuntimeError, match="has stopped"):
        s.advance([candidate("b", 0.9, cycle=1)])


def test_max_cycles_one_still_runs_a_cycle():
    s = make_search(run={"max_cycles": 1, "patience": 99})
    s.seed_frontier([candidate("a", 0.4)])
    assert not s.stopped
    s.advance([candidate("b", 0.9, cycle=1)])
    assert s.stopped
    assert "max_cycles=1" in s.stop_reason

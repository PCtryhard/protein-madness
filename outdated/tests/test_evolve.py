"""The evolution loop, tested without a GPU.

Every test here runs on CPU in milliseconds. That is deliberate: the loop had to
be buildable and verifiable while the clusters were busy, and it must stay
testable when they are unavailable.

The properties that matter most are the ones that protect the science rather
than the code: the catalytic residues are never mutated, a design is never
folded twice, selection never promotes on a single noisy seed, and the Boltz
input always carries the ligand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evolve import diversity as div
from src.evolve import fold_batch
from src.evolve.driver import Individual, RunConfig, init_run, load_population
from src.evolve.mutate import (
    AMINO_ACIDS,
    Mutation,
    MutationPolicy,
    assert_catalytic_intact,
    blosum_row,
    classify_positions,
    crossover,
    mutate,
    sample_count,
    sample_positions,
)
from src.evolve.select import (
    FitnessWeights,
    HardGates,
    contact_penalty,
    fitness,
    hamming,
    improved,
    passes_gates,
    select_parents,
    triage_cut,
)

SEQ = "MVTAEVLAVRKVGPCTEVTVRVTVGGEEATVRVTTFEVAPGVVYVLLPSIG" * 2  # 100 aa, deterministic
CATALYTIC = {10: SEQ[9], 40: SEQ[39]}


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


@pytest.fixture()
def labels():
    return classify_positions(len(SEQ), second_shell={5, 6, 7, 8, 20, 21}, core={30, 31, 32})


# ------------------------------------------------------------------ mutate

def test_mutation_never_touches_catalytic_positions(labels, rng):
    """The single most important property in this module."""
    excluded = set(CATALYTIC)
    for _ in range(300):
        child, mutations = mutate(SEQ, labels, MutationPolicy(), rng, excluded)
        assert all(m.position not in excluded for m in mutations)
        assert_catalytic_intact(child, SEQ, CATALYTIC)


def test_mutation_actually_changes_something(labels, rng):
    child, mutations = mutate(SEQ, labels, MutationPolicy(), rng, set(CATALYTIC))
    assert mutations
    assert child != SEQ
    assert len(child) == len(SEQ)


def test_recorded_mutations_match_the_child(labels, rng):
    """A lineage that does not reconstruct the sequence is a lie in the pitch."""
    child, mutations = mutate(SEQ, labels, MutationPolicy(), rng, set(CATALYTIC))
    rebuilt = list(SEQ)
    for m in mutations:
        assert rebuilt[m.position - 1] == m.from_aa
        rebuilt[m.position - 1] = m.to_aa
    assert "".join(rebuilt) == child


def test_k_schedule_mostly_single_mutations(rng):
    counts = [sample_count(MutationPolicy(), rng) for _ in range(2000)]
    assert 0.6 < sum(c == 1 for c in counts) / len(counts) < 0.8
    assert max(counts) <= 5


def test_stagnant_policy_is_more_aggressive(rng):
    normal = [sample_count(MutationPolicy(), rng) for _ in range(2000)]
    stagnant = [sample_count(MutationPolicy().stagnant(), rng) for _ in range(2000)]
    assert np.mean(stagnant) > np.mean(normal)


def test_position_weighting_prefers_the_second_shell(labels, rng):
    picks = []
    for _ in range(400):
        picks.extend(sample_positions(1, labels, MutationPolicy(), rng, set(CATALYTIC)))
    shell = {5, 6, 7, 8, 20, 21}
    fraction = sum(1 for p in picks if p in shell) / len(picks)
    # class_weights is a per-CLASS budget: the second shell carries 60% of the
    # probability mass however few positions it contains. Without that, the
    # surface (~90% of the protein) would swallow most mutations.
    assert 0.5 < fraction < 0.7


def test_cysteine_is_not_introduced(labels, rng):
    for _ in range(300):
        child, mutations = mutate(SEQ, labels, MutationPolicy(), rng, set(CATALYTIC))
        for m in mutations:
            if m.from_aa != "C":
                assert m.to_aa != "C"


def test_blosum_row_is_a_distribution_without_the_parent():
    row = blosum_row("D")
    assert "D" not in row
    assert set(row) <= set(AMINO_ACIDS)
    assert row  # non-empty
    assert math.isclose(sum(row.values()), 1.0, rel_tol=1e-9)
    # Glutamate is the conservative swap for aspartate and must beat tryptophan.
    assert row["E"] > row["W"]


def test_model_posterior_is_used_when_supplied(labels):
    """Model-guided mode samples LigandMPNN's own distribution, not BLOSUM."""
    rng = np.random.default_rng(3)
    posterior = {5: {"W": 1.0}}
    policy = MutationPolicy(k_weights=(1.0,), mode_weights=(0.0, 1.0, 0.0))
    only_position_5 = {p for p in range(1, len(SEQ) + 1) if p != 5}
    seen = set()
    for _ in range(40):
        _child, mutations = mutate(SEQ, labels, policy, rng, only_position_5, posterior=posterior)
        seen.update((m.position, m.to_aa, m.mode) for m in mutations)
    assert seen, "no mutation was produced"
    assert all(aa == "W" and mode == "model" for _p, aa, mode in seen)


def test_model_mode_falls_back_when_no_posterior_exists(labels):
    """A position we have no model distribution for must not be sampled from a
    distribution we do not have: the model share folds back into conservative."""
    rng = np.random.default_rng(5)
    policy = MutationPolicy(k_weights=(1.0,), mode_weights=(0.0, 1.0, 0.0))
    only_position_3 = {p for p in range(1, len(SEQ) + 1) if p != 3}
    _child, mutations = mutate(SEQ, labels, policy, rng, only_position_3, posterior={5: {"W": 1.0}})
    assert mutations and all(m.mode == "conservative" for m in mutations)


def test_assert_catalytic_intact_rejects_a_changed_base():
    broken = list(SEQ)
    broken[9] = "A" if SEQ[9] != "A" else "G"
    with pytest.raises(ValueError, match="catalytic position"):
        assert_catalytic_intact("".join(broken), SEQ, CATALYTIC)


def test_crossover_refuses_parents_that_disagree_at_protected_positions(rng):
    other = list(SEQ)
    other[9] = "A" if SEQ[9] != "A" else "G"
    with pytest.raises(ValueError, match="protected position"):
        crossover(SEQ, "".join(other), rng, set(CATALYTIC))


# ------------------------------------------------------------------ select

def _aggregate(base, ipde=2.0, burial=35.0, base_std=0.0, ipde_std=0.0, donor=3.0):
    return {
        "n_seeds": 5,
        "base_to_ligand_a_mean": base, "base_to_ligand_a_std": base_std,
        "donor_to_ligand_a_mean": donor, "donor_to_ligand_a_std": 0.0,
        "ipde_mean": ipde, "ipde_std": ipde_std,
        "base_burial_ca_count_mean": burial, "base_burial_ca_count_std": 0.0,
    }


def test_closer_catalytic_contact_is_cheaper():
    """Geometry dominates: this is the metric the reference design fails."""
    assert fitness(_aggregate(4.0)) < fitness(_aggregate(11.18))


def test_reference_design_is_beatable_and_we_can_measure_it():
    """The organisers' own design scores 11.18 A / iPDE 2.55. Anything that
    holds the contact at 4 A must come out clearly better."""
    reference = fitness(_aggregate(11.178812, ipde=2.5525, burial=30))
    better = fitness(_aggregate(4.0, ipde=2.5525, burial=30))
    assert better < reference
    assert reference - better > 5.0


def test_contact_penalty_is_one_sided():
    assert contact_penalty(3.0) == 0.0
    assert contact_penalty(3.5) == 0.0
    assert contact_penalty(5.5) == pytest.approx(2.0)
    assert math.isinf(contact_penalty(math.nan))


def test_spread_is_penalised():
    """A design that only works on one seed must lose to a consistent one."""
    steady = fitness(_aggregate(5.0, base_std=0.0))
    jittery = fitness(_aggregate(5.0, base_std=3.0))
    assert jittery > steady


def test_unmeasurable_design_is_infinite_not_zero():
    assert math.isinf(fitness(_aggregate(math.nan)))


def test_gates_reject_an_ejected_ligand():
    ok, reasons = passes_gates(_aggregate(30.0), SEQ)
    assert not ok and any("base contact" in r for r in reasons)


def test_gates_reject_an_over_long_design():
    ok, reasons = passes_gates(_aggregate(4.0), "A" * 300)
    assert not ok and any("length" in r for r in reasons)


def test_gates_accept_a_good_design():
    ok, reasons = passes_gates(_aggregate(4.0), SEQ)
    assert ok and reasons == []


def test_select_parents_enforces_diversity():
    a = "AAAACCCC"
    near = "AAAACCCG"      # 1 apart
    far = "GGGGTTTT"
    chosen = select_parents([(a, 1.0), (near, 1.1), (far, 5.0)], 2, min_hamming=2)
    assert chosen == [a, far]


def test_select_parents_returns_fewer_rather_than_duplicating():
    a = "AAAACCCC"
    chosen = select_parents([(a, 1.0), ("AAAACCCG", 1.1)], 3, min_hamming=4)
    assert chosen == [a]


def test_triage_keeps_the_cheapest():
    assert triage_cut([5.0, 1.0, math.inf, 3.0], 2) == [1, 3]


def test_improved_requires_a_real_decrease():
    assert improved(1.0, 2.0)
    assert not improved(2.0, 2.0)
    assert not improved(math.inf, 2.0)
    assert improved(2.0, math.inf)


def test_hamming_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        hamming("AA", "AAA")


# --------------------------------------------------------------- diversity

def test_sequence_cache_round_trips(tmp_path):
    cache = div.SequenceCache(tmp_path / "cache.json")
    assert SEQ not in cache
    cache.put(SEQ, {"cost": 1.0})
    cache.save()
    assert SEQ in div.SequenceCache(tmp_path / "cache.json")


def test_cache_survives_corruption(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    assert len(div.SequenceCache(path)) == 0


def test_mean_pairwise_hamming():
    assert div.mean_pairwise_hamming(["AAAA", "AAAA"]) == 0.0
    assert div.mean_pairwise_hamming(["AAAA", "AAAC"]) == 1.0
    assert div.mean_pairwise_hamming(["AAAA"]) == 0.0


def test_positional_entropy_flags_the_variable_column():
    entropy = div.positional_entropy(["AAAA", "AAAC", "AAAG", "AAAT"])
    assert entropy[0] == pytest.approx(0.0)
    assert entropy[3] == pytest.approx(2.0)


def test_assess_flags_a_collapsed_population():
    report = div.assess(["AAAAAAAA", "AAAAAAAC"], records=[{"lineage": "L0"}] * 2)
    assert report.collapsed
    assert any("Hamming" in r for r in report.reasons)


def test_assess_flags_structural_convergence():
    """Children folding exactly like their parents means we are sampling noise."""
    report = div.assess(
        ["AAAACCCC", "GGGGTTTT"],
        records=[{"lineage": "L0"}, {"lineage": "L1"}],
        rmsds=[0.01, 0.02, 0.03],
    )
    assert report.collapsed
    assert any("seed noise" in r for r in report.reasons)


def test_assess_passes_a_healthy_population():
    report = div.assess(
        ["AAAACCCC", "GGGGTTTT"],
        records=[{"lineage": "L0"}, {"lineage": "L1"}],
        rmsds=[1.5, 2.0],
    )
    assert not report.collapsed


def test_parent_child_rmsd_is_zero_for_identical_coordinates():
    coords = np.random.default_rng(0).normal(size=(20, 3))
    assert div.parent_child_rmsd(coords, coords.copy()) == pytest.approx(0.0, abs=1e-9)


def test_parent_child_rmsd_ignores_rigid_motion():
    coords = np.random.default_rng(1).normal(size=(20, 3))
    moved = coords + np.array([10.0, -5.0, 3.0])
    assert div.parent_child_rmsd(coords, moved) == pytest.approx(0.0, abs=1e-6)


def test_sequence_id_is_stable_and_short():
    assert div.sequence_id(SEQ) == div.sequence_id(SEQ)
    assert len(div.sequence_id(SEQ)) == 12


# -------------------------------------------------------------- fold_batch

def test_design_yaml_always_carries_the_ligand_and_no_msa():
    spec = fold_batch.design_yaml("ACDEF")
    protein, ligand = spec["sequences"]
    assert protein["protein"]["msa"] == "empty"
    assert ligand["ligand"]["ccd"] == "6NT"


def test_write_generation_writes_one_yaml_per_design(tmp_path):
    out = fold_batch.write_generation(tmp_path / "designs", {"d1": "ACDEF", "d2": "GHIKL"})
    files = sorted(p.name for p in out.glob("*.yaml"))
    assert files == ["d1.yaml", "d2.yaml"]
    spec = yaml.safe_load((out / "d1.yaml").read_text())
    assert spec["sequences"][0]["protein"]["sequence"] == "ACDEF"


def test_write_generation_rejects_unsafe_ids(tmp_path):
    with pytest.raises(ValueError, match="safe filename"):
        fold_batch.write_generation(tmp_path / "d", {"../evil": "ACDEF"})


def test_euler_profile_uses_mem_per_cpu_not_mem():
    """Euler's cli_filter rejects --mem outright; AWS accepts it."""
    header = "\n".join(fold_batch.SiteProfile.euler().sbatch_header("j", "log"))
    assert "--mem-per-cpu=6G" in header
    assert "--mem=" not in header
    assert "--account=es_workshop" in header
    assert "--reservation=ws_49734_gpu" in header


def test_aws_profile_avoids_the_big_node():
    """g6.2xlarge takes the whole allocation for one GPU; l4big never finishes."""
    header = "\n".join(fold_batch.SiteProfile().sbatch_header("j", "log"))
    assert "--constraint=g6.xlarge" in header
    assert "l4big" not in header


def test_fold_command_matches_the_organisers_flags():
    command = fold_batch.fold_command(fold_batch.SiteProfile(), "designs", "out", 7)
    for flag in ("--sampling_steps 200", "--recycling_steps 3", "--diffusion_samples 1",
                 "--no_kernels", "--output_format mmcif", "--seed 7"):
        assert flag in command


def test_written_files_use_unix_line_endings(tmp_path):
    """sbatch refuses a script with CRLF, and we author these on Windows.

    Regression test: the first real submission failed with "Batch script
    contains DOS line breaks".
    """
    script = fold_batch.write_fold_script(
        tmp_path / "fold.sbatch", fold_batch.SiteProfile.euler(), "d", "out", [1]
    )
    assert b"\r\n" not in script.read_bytes()
    designs = fold_batch.write_generation(tmp_path / "designs", {"d1": "ACDEF"})
    assert b"\r\n" not in (designs / "d1.yaml").read_bytes()


def test_write_fold_script_covers_every_seed(tmp_path):
    path = fold_batch.write_fold_script(
        tmp_path / "fold.sbatch", fold_batch.SiteProfile.euler(), "d", "out", [1, 2, 3]
    )
    text = path.read_text()
    assert text.count("boltz") >= 3
    for seed in (1, 2, 3):
        assert f"--seed {seed}" in text
    assert "HF_HUB_OFFLINE=1" in text


# ------------------------------------------------------------------ driver

def _config(founder=SEQ):
    return RunConfig(
        founders={"L0": founder},
        base_position=10,
        donor_position=40,
        catalytic=dict(CATALYTIC),
        population=6,
        second_shell=(5, 6, 7, 8),
    )


def test_init_run_builds_a_population_not_a_point(tmp_path):
    init_run(tmp_path / "run", _config())
    people = load_population(tmp_path / "run", 0)
    assert len(people) == 6
    assert people[0].lineage == "L0"
    assert people[0].sequence == SEQ          # the founder itself is kept
    assert len({p.sequence for p in people}) == 6   # the rest are distinct


def test_init_run_keeps_every_catalytic_residue(tmp_path):
    init_run(tmp_path / "run", _config())
    for person in load_population(tmp_path / "run", 0):
        assert_catalytic_intact(person.sequence, SEQ, CATALYTIC)


def test_init_run_writes_designs_for_folding(tmp_path):
    init_run(tmp_path / "run", _config())
    yamls = list((tmp_path / "run" / "gen_000" / "designs").glob("*.yaml"))
    assert len(yamls) == 6


def test_init_run_refuses_to_overwrite(tmp_path):
    init_run(tmp_path / "run", _config())
    with pytest.raises(FileExistsError):
        init_run(tmp_path / "run", _config())


def test_config_round_trips(tmp_path):
    config = _config()
    restored = RunConfig.from_yaml(config.to_yaml())
    assert restored.catalytic == config.catalytic
    assert restored.base_position == config.base_position
    assert restored.site().base_position == 10


def test_fold_script_refuses_output_inside_the_input_directory(tmp_path):
    """Regression: Boltz scans the input dir and refuses a subdirectory in it.
    This exact mistake wasted a 48-design batch on the first real run."""
    with pytest.raises(ValueError, match="inside designs_dir"):
        fold_batch.write_fold_script(
            tmp_path / "f.sbatch", fold_batch.SiteProfile.euler(),
            "/work/chunk_a", "/work/chunk_a/boltz_seed1", [1],
        )
    with pytest.raises(ValueError, match="inside designs_dir"):
        fold_batch.write_fold_script(
            tmp_path / "f.sbatch", fold_batch.SiteProfile.euler(),
            "/work/chunk_a", "/work/chunk_a", [1],
        )


def test_fold_script_accepts_a_sibling_output_directory(tmp_path):
    path = fold_batch.write_fold_script(
        tmp_path / "f.sbatch", fold_batch.SiteProfile.euler(),
        "/work/chunk_a", "/work/out_a", [1],
    )
    assert "/work/out_a" in path.read_text()

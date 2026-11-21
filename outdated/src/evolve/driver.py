"""Generations, state on disk, and resumability.

**The driver is not a long-running process.** It is three short commands, each
of which reads the state directory, does one step, and exits:

    init     create the run and write generation 0's designs
    ingest   score a generation's Boltz output and record it
    propose  select parents, mutate, and write the next generation

A daemon would die with the SSH session and take the run with it; a sequence of
short commands survives anything, and re-running one is safe because a
generation that already has ``scores.csv`` is a no-op. That is also why one
generation maps cleanly onto one Slurm job.

Layout (BRIEF.md 4.6)::

    results/evo/<run_id>/
      config.yaml          frozen parameters for this run
      cache.json           sequence -> score, so nothing is ever folded twice
      lineage.csv          the whole family tree, flat
      gen_000/
        population.jsonl   one row per individual: id, parent, mutations, sequence
        designs/           the YAMLs handed to Boltz
        fold.sbatch        the job that folds them
        boltz/seed_N/      raw model output
        scores.csv         our four metrics + spread, per individual
        elites.json        who became a parent, and why

Every individual, including the dead ones, gets a row. The lineage of a
submitted design - "backbone 7, picked up D93E in generation 4 and A118W in
generation 9" - is a pitch slide no one-shot team can produce.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..score.official import CatalyticSite, OfficialScore, aggregate_seeds
from . import diversity as div
from . import fold_batch
from .mutate import (
    MutationPolicy,
    assert_catalytic_intact,
    classify_positions,
    mutate,
)
from .select import (
    FitnessWeights,
    HardGates,
    fitness,
    improved,
    passes_gates,
    select_parents,
)

log = logging.getLogger("pipeline.evolve.driver")


@dataclass
class Individual:
    design_id: str
    sequence: str
    generation: int
    parent_id: str | None = None
    lineage: str = "founder"
    mutations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunConfig:
    """Frozen for the life of a run. Changing it mid-run invalidates comparisons."""

    founders: dict[str, str]           # lineage name -> sequence
    base_position: int
    donor_position: int
    catalytic: dict[int, str]          # position -> required one-letter code
    population: int = 24
    n_parents: int = 4
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    triage_seeds: tuple[int, ...] = (1,)
    second_shell: tuple[int, ...] = ()
    core: tuple[int, ...] = ()
    ligand_ccd: str = fold_batch.LIGAND_CCD
    rng_seed: int = 0

    def site(self) -> CatalyticSite:
        return CatalyticSite(
            base_position=self.base_position,
            donor_position=self.donor_position,
            ligand_resname=self.ligand_ccd,
        )

    def to_yaml(self) -> str:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        data["triage_seeds"] = list(self.triage_seeds)
        data["second_shell"] = list(self.second_shell)
        data["core"] = list(self.core)
        data["catalytic"] = {int(k): v for k, v in self.catalytic.items()}
        return yaml.safe_dump(data, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "RunConfig":
        data = yaml.safe_load(text)
        data["seeds"] = tuple(data.get("seeds", (1,)))
        data["triage_seeds"] = tuple(data.get("triage_seeds", (1,)))
        data["second_shell"] = tuple(data.get("second_shell", ()))
        data["core"] = tuple(data.get("core", ()))
        data["catalytic"] = {int(k): v for k, v in (data.get("catalytic") or {}).items()}
        return cls(**data)


# ------------------------------------------------------------------- paths

def gen_dir(root: Path, generation: int) -> Path:
    return Path(root) / f"gen_{generation:03d}"


def load_config(root: Path) -> RunConfig:
    return RunConfig.from_yaml((Path(root) / "config.yaml").read_text())


def load_population(root: Path, generation: int) -> list[Individual]:
    path = gen_dir(root, generation) / "population.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no population at {path}")
    return [Individual(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def write_population(root: Path, generation: int, people: list[Individual]) -> None:
    directory = gen_dir(root, generation)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "population.jsonl").write_text(
        "\n".join(json.dumps(p.as_dict()) for p in people) + "\n"
    )


def latest_generation(root: Path) -> int:
    generations = sorted(int(p.name.split("_")[1]) for p in Path(root).glob("gen_*") if p.is_dir())
    if not generations:
        raise FileNotFoundError(f"no generations under {root}")
    return generations[-1]


def is_scored(root: Path, generation: int) -> bool:
    return (gen_dir(root, generation) / "scores.csv").is_file()


# -------------------------------------------------------------------- init

def init_run(root: Path, config: RunConfig) -> Path:
    """Create the run directory and generation 0 from the founder sequences."""
    root = Path(root)
    if (root / "config.yaml").exists():
        raise FileExistsError(f"{root} already holds a run; pick a new run id")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(config.to_yaml())

    people: list[Individual] = []
    rng = np.random.default_rng(config.rng_seed)
    labels = classify_positions(
        len(next(iter(config.founders.values()))),
        set(config.second_shell),
        set(config.core),
    )
    policy = MutationPolicy()
    excluded = set(config.catalytic)

    for lineage, sequence in config.founders.items():
        assert_catalytic_intact(sequence, sequence, config.catalytic)
        people.append(
            Individual(
                design_id=div.sequence_id(sequence),
                sequence=sequence,
                generation=0,
                lineage=lineage,
            )
        )
    # Fill the rest of generation 0 with mutated founders, so generation 0 is
    # already a population rather than a single point.
    founders = list(config.founders.items())
    while len(people) < config.population:
        lineage, parent_sequence = founders[len(people) % len(founders)]
        child, mutations = mutate(parent_sequence, labels, policy, rng, excluded)
        assert_catalytic_intact(child, parent_sequence, config.catalytic)
        design_id = div.sequence_id(child)
        if any(p.design_id == design_id for p in people):
            continue
        people.append(
            Individual(
                design_id=design_id,
                sequence=child,
                generation=0,
                parent_id=div.sequence_id(parent_sequence),
                lineage=lineage,
                mutations=[str(m) for m in mutations],
            )
        )
    write_population(root, 0, people)
    _write_designs(root, 0, people, config)
    log.info("initialised run at %s with %d individuals", root, len(people))
    return root


def _write_designs(root: Path, generation: int, people: list[Individual], config: RunConfig) -> None:
    directory = gen_dir(root, generation)
    fold_batch.write_generation(
        directory / "designs",
        {p.design_id: p.sequence for p in people},
        config.ligand_ccd,
    )


# ------------------------------------------------------------------ ingest

SCORE_COLUMNS = [
    "design_id", "generation", "lineage", "parent_id", "mutations", "n_seeds",
    "ipde_mean", "ipde_std",
    "base_to_ligand_a_mean", "base_to_ligand_a_std",
    "donor_to_ligand_a_mean", "donor_to_ligand_a_std",
    "base_burial_ca_count_mean", "base_burial_ca_count_std",
    "cost", "passes_gates", "gate_reasons", "notes",
]


def ingest_generation(root: Path, generation: int, weights: FitnessWeights = FitnessWeights(),
                      gates: HardGates = HardGates()) -> Path:
    """Score a generation's Boltz output and write ``scores.csv``.

    Re-running is safe and will simply recompute from the same raw output.
    """
    root = Path(root)
    config = load_config(root)
    people = load_population(root, generation)
    directory = gen_dir(root, generation)
    boltz_root = directory / "boltz"
    if not boltz_root.is_dir():
        raise FileNotFoundError(f"no Boltz output under {boltz_root}; has the fold job run?")

    seed_dirs = {
        int(p.name.split("_")[1]): p
        for p in sorted(boltz_root.glob("seed_*")) if p.is_dir()
    }
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* directories under {boltz_root}")

    site = config.site()
    design_ids = [p.design_id for p in people]
    per_design = fold_batch.collect_seeds(seed_dirs, design_ids, site)

    cache = div.SequenceCache(root / "cache.json")
    rows = []
    for person in people:
        scores: list[OfficialScore] = per_design.get(person.design_id, [])
        if not scores:
            rows.append({
                "design_id": person.design_id, "generation": generation,
                "lineage": person.lineage, "parent_id": person.parent_id or "",
                "mutations": ";".join(person.mutations), "n_seeds": 0,
                "cost": math.inf, "passes_gates": False,
                "gate_reasons": "no prediction returned", "notes": "",
            })
            continue
        aggregate = aggregate_seeds(scores)
        cost = fitness(aggregate, weights)
        ok, reasons = passes_gates(aggregate, person.sequence, gates)
        row = {
            "design_id": person.design_id, "generation": generation,
            "lineage": person.lineage, "parent_id": person.parent_id or "",
            "mutations": ";".join(person.mutations),
            "n_seeds": aggregate["n_seeds"], "cost": cost,
            "passes_gates": ok, "gate_reasons": "; ".join(reasons),
            "notes": "; ".join(aggregate.get("notes", [])),
        }
        for column in SCORE_COLUMNS:
            if column.endswith(("_mean", "_std")) and column in aggregate:
                row[column] = aggregate[column]
        rows.append(row)
        cache.put(person.sequence, {"design_id": person.design_id, "cost": cost,
                                    "generation": generation})
    cache.save()

    path = directory / "scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("scored generation %d: %d rows -> %s", generation, len(rows), path)
    _append_lineage(root, rows)
    return path


def _append_lineage(root: Path, rows: list[dict]) -> None:
    path = Path(root) / "lineage.csv"
    exists = path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_scores(root: Path, generation: int) -> list[dict]:
    path = gen_dir(root, generation) / "scores.csv"
    if not path.is_file():
        raise FileNotFoundError(f"generation {generation} has not been ingested ({path})")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------- propose

def propose_next(root: Path, weights: FitnessWeights = FitnessWeights()) -> int:
    """Select parents from the last scored generation and write the next one.

    Returns the new generation number. Applies the diversity guard and the
    stagnation trigger, both automatically and both logged - no judgement calls
    at hour 25.
    """
    root = Path(root)
    config = load_config(root)
    generation = latest_generation(root)
    if not is_scored(root, generation):
        raise RuntimeError(
            f"generation {generation} has no scores.csv; run `ingest` before `propose`"
        )
    rows = read_scores(root, generation)
    people = {p.design_id: p for p in load_population(root, generation)}

    survivors = [
        (people[r["design_id"]].sequence, float(r["cost"]))
        for r in rows
        if r.get("passes_gates") in ("True", True) and r.get("cost") not in ("", None)
        and math.isfinite(float(r["cost"]))
    ]
    if not survivors:
        raise RuntimeError(
            f"generation {generation}: no design passed the hard gates. "
            "Widen the gates deliberately or go back to generation 0 - do not "
            "quietly lower a threshold to make a run continue."
        )

    # --- stagnation: did we actually beat the previous generation?
    best_now = min(cost for _s, cost in survivors)
    best_before = math.inf
    if generation > 0 and is_scored(root, generation - 1):
        previous = [float(r["cost"]) for r in read_scores(root, generation - 1)
                    if r.get("cost") not in ("", None) and math.isfinite(float(r["cost"]))]
        best_before = min(previous) if previous else math.inf
    policy = MutationPolicy()
    if not improved(best_now, best_before):
        policy = policy.stagnant()
        log.warning(
            "generation %d did not improve on %d (best %.3f vs %.3f): widening mutations",
            generation, generation - 1, best_now, best_before,
        )

    parents = select_parents(survivors, config.n_parents)
    if not parents:
        raise RuntimeError("no parents survived the diversity floor")

    # --- diversity guard
    report = div.assess(
        [people[r["design_id"]].sequence for r in rows if r["design_id"] in people],
        records=[{"lineage": r.get("lineage", "unknown")} for r in rows],
    )
    if report.collapsed:
        policy = policy.stagnant()

    (gen_dir(root, generation) / "elites.json").write_text(json.dumps({
        "best_cost": best_now,
        "previous_best_cost": None if math.isinf(best_before) else best_before,
        "improved": improved(best_now, best_before),
        "policy": "stagnant" if policy != MutationPolicy() else "normal",
        "diversity": report.as_dict(),
        "parents": [div.sequence_id(s) for s in parents],
    }, indent=2))

    # --- breed
    rng = np.random.default_rng(config.rng_seed + generation + 1)
    labels = classify_positions(len(parents[0]), set(config.second_shell), set(config.core))
    excluded = set(config.catalytic)
    cache = div.SequenceCache(root / "cache.json")
    lineage_of = {people[r["design_id"]].sequence: r.get("lineage", "unknown") for r in rows
                  if r["design_id"] in people}

    children: list[Individual] = []
    attempts = 0
    while len(children) < config.population and attempts < config.population * 50:
        attempts += 1
        parent_sequence = parents[len(children) % len(parents)]
        child, mutations = mutate(parent_sequence, labels, policy, rng, excluded)
        try:
            assert_catalytic_intact(child, parent_sequence, config.catalytic)
        except ValueError as exc:  # pragma: no cover - the operator should prevent this
            log.error("discarding child: %s", exc)
            continue
        if child in cache or any(c.sequence == child for c in children):
            continue  # never fold the same sequence twice
        children.append(Individual(
            design_id=div.sequence_id(child),
            sequence=child,
            generation=generation + 1,
            parent_id=div.sequence_id(parent_sequence),
            lineage=lineage_of.get(parent_sequence, "unknown"),
            mutations=[str(m) for m in mutations],
        ))
    if len(children) < config.population:
        log.warning(
            "only bred %d/%d children in %d attempts; the search space near these "
            "parents is close to exhausted", len(children), config.population, attempts,
        )
    if not children:
        raise RuntimeError("could not breed any new sequence; inject fresh founders")

    next_generation = generation + 1
    write_population(root, next_generation, children)
    _write_designs(root, next_generation, children, config)
    log.info("generation %d: %d children from %d parents", next_generation, len(children), len(parents))
    return next_generation


# --------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Directed evolution driver.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a run from a founder sequence")
    p_init.add_argument("root", type=Path)
    p_init.add_argument("--founder", required=True, help="sequence, or a path to a FASTA")
    p_init.add_argument("--base", type=int, required=True)
    p_init.add_argument("--donor", type=int, required=True)
    p_init.add_argument("--population", type=int, default=24)
    p_init.add_argument("--shell", type=Path, help="JSON list of second-shell positions")

    p_ing = sub.add_parser("ingest", help="score a generation's Boltz output")
    p_ing.add_argument("root", type=Path)
    p_ing.add_argument("--generation", type=int, default=None)

    p_pro = sub.add_parser("propose", help="select parents and write the next generation")
    p_pro.add_argument("root", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "init":
        founder = args.founder
        candidate = Path(founder)
        if candidate.is_file():
            founder = "".join(
                line.strip() for line in candidate.read_text().splitlines()
                if not line.startswith(">")
            )
        shell = tuple(json.loads(args.shell.read_text())) if args.shell else ()
        config = RunConfig(
            founders={"L0": founder},
            base_position=args.base,
            donor_position=args.donor,
            catalytic={args.base: founder[args.base - 1], args.donor: founder[args.donor - 1]},
            population=args.population,
            second_shell=shell,
        )
        init_run(args.root, config)
        print(f"run initialised at {args.root}")
    elif args.command == "ingest":
        generation = args.generation if args.generation is not None else latest_generation(args.root)
        print(ingest_generation(args.root, generation))
    else:
        print(f"wrote generation {propose_next(args.root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""The two-worker GPU queue and the per-scaffold search loop.

Two cards at 12 GB is not 24 GB. Every model instance must fit inside one card,
so the parallelism available is two independent designs at a time, not one
larger job split across both. There is no sharding of a single model across
both GPUs, by design.

Each GPU gets one worker thread which owns it for the length of the run. Every
stage runs as a subprocess with `CUDA_VISIBLE_DEVICES` pinned to that worker's
card, so two models never share a card: the queue, not a lock inside a model,
is what enforces it.

Each seed scaffold is an independent search with its own frontier, stopping
condition and leaderboard. There is no competition between scaffolds, so
results stay comparable across seeds and the number of seeds can change
without altering behaviour.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dataclasses import replace

from .config import Config, Seed
from .live import LiveResults
from .scoring import ScoredDesign, score_design
from .search import Candidate, ScaffoldSearch, jobs_for, mutants_for_parent
from .search.wiggle import WiggleJob
from .stages import boltz as boltz_stage
from .stages import ligandmpnn as mpnn_stage
from .stages import protonate as protonate_stage
from .stages import rfd3 as rfd_stage
from .stages.base import StageError
from .store import (
    DesignRow,
    DesignStore,
    new_design_id,
    sequence_hash,
)
from .tracking import (
    ResKey,
    write_pdb,
    Provenance,
    TrackingError,
    assert_motif_unchanged,
    hal_to_boltz_map,
    ligand_atom_map_ccd,
    ligand_atom_map_graph,
    read_structure,
    seed_to_hal_map_rfd3,
)

log = logging.getLogger("pipeline.runner")

DESIGN_CHAIN = "A"


class GpuPool:
    """One worker thread per card, each pinned to its own device.

    Submitted work is a callable taking the gpu index. The pool is the only
    thing that decides which card a stage runs on, which is what keeps the
    "never two models on one card" rule in one place.
    """

    def __init__(self, gpus: list[int]) -> None:
        if not gpus:
            raise ValueError("GpuPool needs at least one device")
        self.gpus = list(gpus)
        self._free: list[int] = list(gpus)
        self._lock = threading.Lock()
        self._available = threading.Semaphore(len(gpus))
        self._pool = ThreadPoolExecutor(
            max_workers=len(gpus), thread_name_prefix="gpu"
        )

    def submit(self, fn: Callable[[int], object]) -> Future:
        return self._pool.submit(self._with_gpu, fn)

    def _with_gpu(self, fn: Callable[[int], object]) -> object:
        self._available.acquire()
        with self._lock:
            gpu = self._free.pop()
        try:
            return fn(gpu)
        finally:
            with self._lock:
                self._free.append(gpu)
            self._available.release()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> GpuPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


@dataclass
class RunPaths:
    """Where a run writes. One directory per run, one subdir per design."""

    root: Path

    @classmethod
    def create(cls, runs_dir: str | Path, run_name: str) -> RunPaths:
        root = Path(runs_dir) / run_name
        (root / "structures").mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    @property
    def db(self) -> Path:
        return self.root / "designs.db"

    @property
    def hardware_notes(self) -> Path:
        return self.root / "hardware_notes.md"

    @property
    def stage_log(self) -> Path:
        return self.root / "stages.jsonl"

    @property
    def designs_csv(self) -> Path:
        return self.root / "designs.csv"

    @property
    def sequences_fasta(self) -> Path:
        return self.root / "sequences.fasta"

    def design_dir(self, design_id: str) -> Path:
        path = self.root / "structures" / design_id
        path.mkdir(parents=True, exist_ok=True)
        return path


class Runner:
    """Executes a run: cycle 0 from each seed, then the search loop."""

    def __init__(
        self,
        cfg: Config,
        paths: RunPaths,
        store: DesignStore,
        repo_root: Path | None = None,
        live: LiveResults | None = None,
    ) -> None:
        self.cfg = cfg
        self.paths = paths
        self.store = store
        self.repo_root = repo_root or Path.cwd()
        self.rng = random.Random(cfg.pipeline.run.seed)
        self._ligand_map_cache: dict[str, str] | None = None
        self._log_lock = threading.Lock()
        # Appends a CSV row and a FASTA record the moment each design finishes,
        # so the run directory is readable while the run is still going. Shared
        # between both workers; it takes its own lock.
        self.live = live if live is not None else LiveResults(paths.root, cfg)

    # ------------------------------------------------------------------
    # stage logging
    # ------------------------------------------------------------------

    def _log_stage(self, design_id: str, payload: dict) -> None:
        with self._log_lock:
            with self.paths.stage_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"design_id": design_id, **payload}) + "\n")

    def _stage_begin(self, scaffold_id: str, stage: str, detail: str = "") -> None:
        """Announce a stage BEFORE it runs.

        The stage log only records a stage once it finishes, so without this
        the console goes quiet for the minutes RFD3 or Boltz takes and there is
        no way to tell work from a hang.
        """
        log.info("[%s] %s ... %s", scaffold_id, stage, detail)
        with self._log_lock:
            with self.paths.stage_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "scaffold_id": scaffold_id,
                    "stage": stage,
                    "event": "begin",
                    "detail": detail,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }) + "\n")

    # ------------------------------------------------------------------
    # the ligand atom map, cached once per run
    # ------------------------------------------------------------------

    def ligand_atom_map(self, predicted_structure_path: Path) -> dict[str, str]:
        """Map target-spec ligand names onto the predicted structure's names.

        Cached per run: it is identical for every design, and on the SMILES
        path it costs an RDKit substructure match.
        """
        if self._ligand_map_cache is not None:
            return self._ligand_map_cache

        ligand = self.cfg.target.ligand
        wanted = sorted(
            {
                spec.atom
                for bond in self.cfg.target.hbonds
                for spec in (
                    bond.donor, bond.acceptor,
                    bond.donor_antecedent, bond.acceptor_antecedent,
                )
                if spec is not None and spec.site == "ligand"
            }
        )
        structure = read_structure(predicted_structure_path, ligand.ccd or ligand.id)

        if ligand.uses_ccd:
            mapping = ligand_atom_map_ccd(wanted, structure.ligand)
        else:
            assert ligand.reference_sdf is not None  # config.Ligand guarantees it
            full = ligand_atom_map_graph(
                self.repo_root / ligand.reference_sdf, predicted_structure_path
            )
            missing = [n for n in wanted if n not in full]
            if missing:
                raise TrackingError(
                    f"ligand atoms {missing} named in the target spec were not "
                    "matched by the graph mapping"
                )
            mapping = {n: full[n] for n in wanted}

        self._ligand_map_cache = mapping
        return mapping

    # ------------------------------------------------------------------
    # one Boltz-2 + protonate + score pass, with the cache in front of it
    # ------------------------------------------------------------------

    def predict_and_score(
        self,
        design_id: str,
        sequence: str,
        seed_to_hal: dict,
        motif_expected: dict,
        scaffold_id: str,
        gpu: int,
    ) -> tuple[ScoredDesign, Path, Provenance]:
        """Boltz-2, hydrogens if the rule needs them, then the score.

        Averages `run.eval_replicates` Boltz-2 seeds. Phase 0 decides whether
        more than one is needed: if the geometry sub-score's spread across
        seeds rivals the spread between genuinely different designs, the
        geometry terms are mostly prediction noise.
        """
        pipeline = self.cfg.pipeline
        design_dir = self.paths.design_dir(design_id)
        sequences = {DESIGN_CHAIN: sequence}

        replicate_scores = []
        structure_path: Path | None = None
        provenance: Provenance | None = None

        for replicate in range(pipeline.run.eval_replicates):
            seed_value = pipeline.run.seed + replicate
            name = f"{design_id}_r{replicate}" if pipeline.run.eval_replicates > 1 else design_id
            self._stage_begin(
                scaffold_id, "boltz-2 (fold + score)",
                f"{design_id}, replicate {replicate}, {len(sequence)} residues",
            )
            out, stage_result = boltz_stage.predict(
                sequences=sequences,
                ligand=self.cfg.target.ligand,
                cfg=pipeline.boltz,
                out_dir=design_dir / f"boltz_r{replicate}",
                name=name,
                gpu=gpu,
                seed=seed_value,
            )
            self._log_stage(design_id, stage_result.as_log_dict())

            # Boltz numbers its output 1..N per chain in submitted order. The
            # map is computed from the exact string submitted, not assumed.
            submitted = boltz_stage.read_sequences_from_yaml(
                design_dir / f"boltz_r{replicate}" / f"{name}.yaml"
            )
            # The residue numbers of the structure the sequence came from.
            # LigandMPNN preserves its input numbering, and RFdiffusion writes
            # its output renumbered from 1, so for a design that came off a
            # backbone these are 1..len(sequence). The map is still built and
            # length-checked by hal_to_boltz_map rather than assumed to be an
            # offset of zero.
            hal_numbers = {DESIGN_CHAIN: list(range(1, len(sequence) + 1))}
            hal_to_boltz = hal_to_boltz_map(submitted, hal_numbers)

            prov = Provenance(
                design_id=design_id,
                scaffold_id=scaffold_id,
                seed_to_hal=seed_to_hal,
                hal_to_boltz=hal_to_boltz,
                ligand_atom_map=self.ligand_atom_map(out.structure),
                motif_expected=motif_expected,
            )

            scoring_path = out.structure
            if protonate_stage.needed(pipeline.scoring.angle.rule):
                scoring_path, _ = protonate_stage.run(
                    pipeline.protonation,
                    out.structure,
                    design_dir / f"protonated_r{replicate}.pdb",
                )

            structure = read_structure(
                scoring_path, self.cfg.target.ligand.ccd or self.cfg.target.ligand.id
            )
            # Assert the motif survived the whole pass, not only at the end.
            boltz_expected = {
                prov.seed_residue_in_boltz(c, n): resname
                for (c, n), resname in motif_expected.items()
            }
            assert_motif_unchanged(structure, boltz_expected, stage="boltz-2")

            scored = score_design(self.cfg, structure, prov, out.confidence)
            replicate_scores.append(scored)
            if structure_path is None:
                structure_path = out.structure
                provenance = prov

        assert structure_path is not None and provenance is not None

        if len(replicate_scores) == 1:
            combined = replicate_scores[0]
        else:
            from .scoring.combine import average_replicates

            combined = ScoredDesign(
                score=average_replicates([s.score for s in replicate_scores]),
                measurements=replicate_scores[0].measurements,
                extra_metrics=replicate_scores[0].extra_metrics,
            )
        return combined, structure_path, provenance

    def _record(self, row: DesignRow) -> None:
        """Store a finished design and publish it to the live files.

        The single place both happen, so a design can never reach designs.db
        without reaching designs.csv -- which is what someone watching the run
        is actually reading.
        """
        self.store.insert(row)
        try:
            self.live.append(row)
        except OSError as exc:
            # A failure to write the convenience CSV must not lose a design
            # that is already safely in the database.
            log.warning("could not append %s to designs.csv: %s", row.design_id, exc)

    def submit_or_cache(
        self,
        design_id: str,
        sequence: str,
        scaffold_id: str,
        cycle: int,
        branch: str,
        parent_id: str | None,
        seed_to_hal: dict,
        motif_expected: dict,
        backbone_path: str | None,
        mutation: str | None,
        gpu: int,
    ) -> Candidate:
        """One design, end to end, with the sequence-hash cache in front.

        On a cache hit nothing touches a GPU: the stored score is reused and a
        new row records the lineage. A mutation search regenerates identical
        sequences constantly, and on two cards this cache is worth more than
        any other optimisation in the build.
        """
        seq_hash = sequence_hash(
            sequence, self.cfg.target.ligand, self.cfg.pipeline.boltz
        )
        hit = self.store.cache_lookup(seq_hash)
        if hit is not None:
            log.info("cache hit for %s (reusing %s)", design_id, hit.design_id)
            row = DesignRow(
                design_id=design_id,
                parent_id=parent_id,
                scaffold_id=scaffold_id,
                cycle=cycle,
                branch=branch,
                mutation=mutation,
                sequence=sequence,
                seq_hash=seq_hash,
                score=hit.score,
                term_confidence=hit.term_confidence,
                term_distance=hit.term_distance,
                term_angle=hit.term_angle,
                gated=hit.gated,
                status=hit.status,
                hbond_raw=hit.hbond_raw,
                extra_metrics=hit.extra_metrics,
                provenance=hit.provenance,
                structure_path=hit.structure_path,
                backbone_path=backbone_path or hit.backbone_path,
            )
            self._record(row)
            return row.to_candidate()

        try:
            scored, structure_path, provenance = self.predict_and_score(
                design_id=design_id,
                sequence=sequence,
                seed_to_hal=seed_to_hal,
                motif_expected=motif_expected,
                scaffold_id=scaffold_id,
                gpu=gpu,
            )
        except (StageError, TrackingError, ValueError, RuntimeError) as exc:
            # Recorded, not swallowed: the design is errored and the run goes
            # on, but nothing partial is ever scored.
            log.warning("design %s errored: %s", design_id, exc)
            self._record(
                DesignRow(
                    design_id=design_id,
                    parent_id=parent_id,
                    scaffold_id=scaffold_id,
                    cycle=cycle,
                    branch=branch,
                    mutation=mutation,
                    sequence=sequence,
                    seq_hash=seq_hash,
                    status="errored",
                    error=f"{type(exc).__name__}: {exc}",
                    provenance=json.dumps({"design_id": design_id}),
                    backbone_path=backbone_path,
                )
            )
            raise

        score = scored.score
        row = DesignRow(
            design_id=design_id,
            parent_id=parent_id,
            scaffold_id=scaffold_id,
            cycle=cycle,
            branch=branch,
            mutation=mutation,
            sequence=sequence,
            seq_hash=seq_hash,
            score=score.total,
            term_confidence=score.confidence,
            term_distance=score.distance,
            term_angle=score.angle,
            gated=score.gated,
            status="gated" if score.gated else "scored",
            hbond_raw=json.dumps(scored.hbond_raw()),
            extra_metrics=json.dumps(scored.extra_metrics),
            provenance=provenance.to_json(),
            structure_path=str(structure_path),
            backbone_path=backbone_path,
        )
        self._record(row)
        return row.to_candidate()

    # ------------------------------------------------------------------
    # cycle 0: the seed
    # ------------------------------------------------------------------

    def seed_cycle(self, seed: Seed, gpu_pool: GpuPool) -> list[Candidate]:
        """Motif-scaffold from the seed, design sequences, predict, score."""
        scaffold_id = f"{self.cfg.target.name}.{seed.id}"
        design_dir = self.paths.design_dir(f"{scaffold_id}.c00.seed")

        def generate(gpu: int) -> list[Candidate]:
            spec = rfd_stage.spec_from_seed(
                seed=seed,
                motif=self.cfg.target.motif,
                ligand_id=self.cfg.target.ligand.id,
                repo_root=self.repo_root,
                key=f"{self.cfg.target.name}_{seed.id}",
            )
            n = self.cfg.pipeline.rfd3.n_batches
            self._stage_begin(
                scaffold_id, "rfd3 (cycle 0: motif scaffolding)",
                f"{n} backbone(s), contig {seed.contig}, pinning "
                f"{spec.unindex}",
            )
            backbones, stage_result = rfd_stage.run(
                self.cfg.pipeline.rfd3,
                specs=[spec],
                out_dir=design_dir / "rfd3",
                n_designs=n,
                partial=False,     # cycle 0 scaffolds the motif; it does not wiggle
                gpu=gpu,
                seed=self.cfg.pipeline.run.seed,
            )
            self._log_stage(scaffold_id, stage_result.as_log_dict())
            log.info("[%s] rfd3 produced %d backbone(s) in %.0fs",
                     scaffold_id, len(backbones), stage_result.wall_seconds)
            return self._design_and_score_backbones(
                backbones, scaffold_id, cycle=0, branch="seed",
                parent_id=None, gpu=gpu,
            )

        return gpu_pool.submit(generate).result()

    def _design_and_score_backbones(
        self,
        backbones: list[rfd_stage.RFD3Output],
        scaffold_id: str,
        cycle: int,
        branch: str,
        parent_id: str | None,
        gpu: int,
        seed_to_input: dict[ResKey, ResKey] | None = None,
    ) -> list[Candidate]:
        """LigandMPNN then Boltz-2 then score, for each backbone."""
        out: list[Candidate] = []
        motif = self.cfg.target.motif

        for backbone in backbones:
            seed_to_hal = seed_to_hal_map_rfd3(
                backbone.metadata, motif, seed_to_input=seed_to_input
            )
            backbone_structure = read_structure(
                backbone.structure, self.cfg.target.ligand.ccd
            )
            fidelity = backbone.motif_fidelity()
            if fidelity:
                log.info(
                    "[%s] backbone %d motif RMSD: %s",
                    scaffold_id, backbone.index,
                    ", ".join(f"{k} {v:.2f}A" for k, v in sorted(fidelity.items())),
                )
            motif_expected = {
                hal: backbone_structure.residues[hal]
                for hal in seed_to_hal.values()
                if hal in backbone_structure.residues
            }
            # Re-key to seed numbering, which is what Provenance expects.
            motif_expected = {
                seed_key: motif_expected[hal_key]
                for seed_key, hal_key in seed_to_hal.items()
                if hal_key in motif_expected
            }

            # LigandMPNN reads a PDB; RFD3 writes a gzipped mmCIF. Convert
            # once, next to the backbone, so the file is inspectable too.
            backbone_pdb = backbone.structure.parent / (
                backbone.structure.name.split(".cif")[0] + ".pdb"
            )
            if not backbone_pdb.exists():
                write_pdb(backbone_structure, backbone_pdb)

            self._stage_begin(
                scaffold_id, "ligandmpnn",
                f"backbone {backbone.index}, "
                f"{self.cfg.pipeline.ligandmpnn.sequences_per_backbone} sequence(s)",
            )
            mpnn_out, stage_result = mpnn_stage.run(
                self.cfg.pipeline.ligandmpnn,
                input_pdb=backbone_pdb,
                out_dir=backbone.structure.parent / f"mpnn_{backbone.index}",
                motif=motif,
                seed_to_hal=seed_to_hal,
                gpu=gpu,
                seed=self.cfg.pipeline.run.seed,
            )
            self._log_stage(scaffold_id, stage_result.as_log_dict())
            designs = mpnn_stage.designed_sequences(mpnn_out)
            log.info("[%s] ligandmpnn gave %d sequence(s) for backbone %d",
                     scaffold_id, len(designs), backbone.index)

            for header, sequence in designs:
                design_id = new_design_id(
                    scaffold_id, cycle, branch, f"{backbone.structure}:{header}"
                )
                try:
                    out.append(
                        self.submit_or_cache(
                            design_id=design_id,
                            sequence=sequence,
                            scaffold_id=scaffold_id,
                            cycle=cycle,
                            branch=branch,
                            parent_id=parent_id,
                            seed_to_hal=seed_to_hal,
                            motif_expected=motif_expected,
                            backbone_path=str(backbone_pdb),
                            mutation=None,
                            gpu=gpu,
                        )
                    )
                except (StageError, TrackingError, ValueError, RuntimeError):
                    continue   # already recorded as errored
        return out

    # ------------------------------------------------------------------
    # the two branches
    # ------------------------------------------------------------------

    def wiggle_branch(
        self, search: ScaffoldSearch, jobs: list[WiggleJob], gpu_pool: GpuPool
    ) -> list[Candidate]:
        """Re-noise each parent's backbone, redesign, refold, rescore."""
        futures = [gpu_pool.submit(self._one_wiggle(search, job)) for job in jobs]
        out: list[Candidate] = []
        for future in futures:
            try:
                out.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - per-design, already recorded
                log.warning("wiggle job failed: %s", exc)
        return out

    def _one_wiggle(
        self, search: ScaffoldSearch, job: WiggleJob
    ) -> Callable[[int], list[Candidate]]:
        def work(gpu: int) -> list[Candidate]:
            design_dir = self.paths.design_dir(
                f"{search.scaffold_id}.c{search.cycle + 1:02d}.wiggle.{job.parent_id[-6:]}"
            )
            seed = self._seed_for(search.scaffold_id)
            spec = rfd_stage.spec_from_seed(
                seed=seed,
                motif=self.cfg.target.motif,
                ligand_id=self.cfg.target.ligand.id,
                repo_root=self.repo_root,
                key=f"{self.cfg.target.name}_{seed.id}_wiggle",
            )
            row = self.store.get(job.parent_id)
            if row is None:
                raise TrackingError(f"no stored provenance for parent {job.parent_id}")
            provenance = Provenance.from_json(row.provenance)
            seed_to_input = {}
            fixed = {spec.ligand: "ALL"} if spec.ligand else {}
            labels = []
            for seed_key in sorted(self.cfg.target.motif.as_keys()):
                if self.cfg.pipeline.search.wiggle.source == "boltz_prediction":
                    source_key = provenance.seed_residue_in_boltz(*seed_key)
                else:
                    source_key = provenance.seed_to_hal[seed_key]
                seed_to_input[seed_key] = source_key
                source_label = f"{source_key[0]}{source_key[1]}"
                labels.append(source_label)
                seed_label = f"{seed_key[0]}{seed_key[1]}"
                fixed[source_label] = spec.select_fixed_atoms[seed_label]
            spec = replace(spec, input_pdb=Path(job.backbone_path),
                           unindex=",".join(labels), select_fixed_atoms=fixed)
            n = (self.cfg.pipeline.search.wiggle.designs_per_parent
                 or self.cfg.pipeline.rfd3.n_batches)
            self._stage_begin(
                search.scaffold_id, "rfd3 (wiggle: partial diffusion)",
                f"parent {job.parent_id}, partial_t="
                f"{self.cfg.pipeline.rfd3.partial_t} A, {n} backbone(s)",
            )
            backbones, stage_result = rfd_stage.run(
                self.cfg.pipeline.rfd3,
                specs=[spec],
                out_dir=design_dir / "rfd3",
                n_designs=n,
                partial=True,       # RFD3 partial noise in angstroms
                gpu=gpu,
                seed=self.cfg.pipeline.run.seed + search.cycle,
            )
            self._log_stage(job.parent_id, stage_result.as_log_dict())
            log.info("[%s] wiggle produced %d backbone(s) from %s",
                     search.scaffold_id, len(backbones), job.parent_id)
            return self._design_and_score_backbones(
                backbones,
                search.scaffold_id,
                cycle=search.cycle + 1,
                branch="wiggle",
                parent_id=job.parent_id,
                gpu=gpu,
                seed_to_input=seed_to_input,
            )

        return work

    def mutate_branch(
        self, search: ScaffoldSearch, parents: list[Candidate], gpu_pool: GpuPool
    ) -> list[Candidate]:
        """Enumerate single mutants of each parent and send them to Boltz-2.

        Skips diffusion and LigandMPNN entirely: a point mutation needs no new
        backbone, and Boltz-2 repacks the neighbourhood implicitly.
        """
        out: list[Candidate] = []
        futures: list[Future] = []

        for parent in parents:
            row = self.store.get(parent.design_id)
            if row is None or not row.structure_path:
                log.warning(
                    "mutate: parent %s has no stored structure; skipping",
                    parent.design_id,
                )
                continue
            provenance = Provenance.from_json(row.provenance)
            structure = read_structure(
                row.structure_path,
                self.cfg.target.ligand.ccd or self.cfg.target.ligand.id,
            )
            try:
                mutants = mutants_for_parent(
                    sequence=parent.sequence,
                    structure=structure,
                    provenance=provenance,
                    motif=self.cfg.target.motif,
                    cfg=self.cfg.pipeline.search.mutate,
                    chain=DESIGN_CHAIN,
                    rng=self.rng,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("mutate: no mutants for %s: %s", parent.design_id, exc)
                continue

            for mutant in mutants:
                design_id = new_design_id(
                    search.scaffold_id,
                    search.cycle + 1,
                    "mutate",
                    f"{parent.design_id}:{mutant.label}",
                )
                futures.append(
                    gpu_pool.submit(
                        self._one_mutant(
                            search, parent, provenance, row, mutant, design_id
                        )
                    )
                )

        for future in futures:
            try:
                out.append(future.result())
            except Exception as exc:  # noqa: BLE001 - already recorded as errored
                log.debug("mutant failed: %s", exc)
        return out

    def _one_mutant(
        self, search, parent, provenance, row, mutant, design_id
    ) -> Callable[[int], Candidate]:
        def work(gpu: int) -> Candidate:
            return self.submit_or_cache(
                design_id=design_id,
                sequence=mutant.sequence,
                scaffold_id=search.scaffold_id,
                cycle=search.cycle + 1,
                branch="mutate",
                parent_id=parent.design_id,
                seed_to_hal=provenance.seed_to_hal,
                motif_expected=provenance.motif_expected,
                # A mutant has no backbone of its own; it inherits its parent's,
                # which is what the wiggle branch needs if it later expands it.
                backbone_path=row.backbone_path,
                mutation=mutant.label,
                gpu=gpu,
            )

        return work

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def _seed_for(self, scaffold_id: str) -> Seed:
        seed_id = scaffold_id.split(".", 1)[1]
        for seed in self.cfg.target.seeds:
            if seed.id == seed_id:
                return seed
        raise KeyError(f"no seed named {seed_id!r} in target {self.cfg.target.name!r}")

    def run_scaffold(self, seed: Seed, gpu_pool: GpuPool) -> ScaffoldSearch:
        """One scaffold's whole search, from cycle 0 to its stopping condition."""
        scaffold_id = f"{self.cfg.target.name}.{seed.id}"
        search = ScaffoldSearch(
            scaffold_id=scaffold_id,
            run=self.cfg.pipeline.run,
            search=self.cfg.pipeline.search,
        )

        log.info("[%s] cycle 0: seeding", scaffold_id)
        search.seed_frontier(self.seed_cycle(seed, gpu_pool))
        log.info(
            "[%s] cycle 0 done: best=%.4f frontier=%s",
            scaffold_id, search.best_score, [c.design_id for c in search.frontier],
        )

        while not search.stopped:
            produced: list[Candidate] = []
            if self.cfg.pipeline.search.wiggle.enabled:
                jobs = jobs_for(
                    search.wiggle_parents(), self.cfg.pipeline.search.wiggle
                )
                produced += self.wiggle_branch(search, jobs, gpu_pool)
            if self.cfg.pipeline.search.mutate.enabled:
                produced += self.mutate_branch(
                    search, search.mutate_parents(), gpu_pool
                )

            if not produced:
                search.stopped = True
                search.stop_reason = "no branch produced a scorable design"
                break

            search.advance(produced)
            log.info(
                "[%s] cycle %d: %d designs, best=%.4f, %d cycle(s) without improvement",
                scaffold_id, search.cycle, len(produced), search.best_score,
                search.cycles_without_improvement,
            )

        log.info("[%s] stopped: %s", scaffold_id, search.stop_reason)
        return search

    def run_all(self) -> dict[str, ScaffoldSearch]:
        """Every scaffold, independently. Adding a seed adds a search."""
        results: dict[str, ScaffoldSearch] = {}
        with GpuPool(self.cfg.pipeline.run.gpus) as pool:
            for seed in self.cfg.target.seeds:
                results[seed.id] = self.run_scaffold(seed, pool)
        return results

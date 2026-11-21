"""Pydantic schema for the two config files, and the loader for both.

Nothing downstream reads YAML. Every stage, scorer and search module takes a
typed object from here, so a missing key is one error at startup rather than a
KeyError at hour 20.

Validation is deliberately strict: `extra="forbid"` everywhere, so a typo in a
config key is an error instead of a silently ignored setting that leaves you
debugging the wrong thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[float, Field(gt=0.0)]


class Strict(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}


# --------------------------------------------------------------------------
# pipeline.yaml
# --------------------------------------------------------------------------


class RunCfg(Strict):
    name: str | None = None
    seed: int = 0
    gpus: list[int] = [0, 1]
    # 0 means seed-only: generate and score cycle 0, then stop without
    # expanding either branch. That is the run you want first -- it exercises
    # the whole forward pass at scale and gives a leaderboard, without
    # committing to the several hours a full search takes.
    max_cycles: Annotated[int, Field(ge=0)] = 12
    patience: Annotated[int, Field(ge=1)] = 2
    min_delta: Annotated[float, Field(gt=0.0)] = 0.005
    eval_replicates: Annotated[int, Field(ge=1)] = 1

    @field_validator("gpus")
    @classmethod
    def _unique_gpus(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("run.gpus must name at least one device")
        if len(set(v)) != len(v):
            raise ValueError(f"run.gpus has duplicates: {v}")
        return v

    @field_validator("min_delta")
    @classmethod
    def _min_delta_above_noise(cls, v: float) -> float:
        # min_delta exists so float noise in the seventh decimal does not read
        # as improvement. A value that small defeats the point.
        if v < 1e-6:
            raise ValueError(
                f"run.min_delta={v} is at the level of floating-point noise; "
                "the search would never stop. Use something like 0.005."
            )
        return v


class RFdiffusionCfg(Strict):
    run_inference: str
    ckpt: str = "Base_ckpt.pt"
    partial_T: Annotated[int, Field(ge=1)] = 10
    diffuser_T: Annotated[int, Field(ge=1)] = 50
    designs_per_parent: Annotated[int, Field(ge=1)] = 4
    extra_args: list[str] = []

    @model_validator(mode="after")
    def _partial_below_full(self) -> RFdiffusionCfg:
        if self.partial_T >= self.diffuser_T:
            raise ValueError(
                f"rfdiffusion.partial_T={self.partial_T} must be below "
                f"diffuser_T={self.diffuser_T}: partial diffusion re-noises part "
                "of the trajectory, not all of it."
            )
        return self


class RFD3Cfg(Strict):
    """RFD3, which is what the hackathon clusters actually have.

    Separate from RFdiffusionCfg because the tools are not interchangeable:
    RFD3 takes a JSON design spec and supports atom-level freezing, while
    RFdiffusion takes a contig string and holds whole residues. The runner
    uses RFD3; src/stages/rfdiffusion.py is kept for a cluster that has the
    other one.
    """

    binary: str = "rfd3"
    checkpoint: str
    n_batches: Annotated[int, Field(ge=1)] = 5
    diffusion_batch_size: Annotated[int, Field(ge=1)] = 1
    partial_T: Annotated[int, Field(ge=1)] = 10
    extra_args: list[str] = []


class LigandMPNNCfg(Strict):
    run_script: str
    checkpoint: str = "ligandmpnn_v_32_010_25.pt"
    temperature: Annotated[float, Field(gt=0.0)] = 0.1
    sequences_per_backbone: Annotated[int, Field(ge=1)] = 4
    omit_AA: str = "CX"
    pack_side_chains: bool = False
    # The cluster's own launcher passes both of these; without the atom context
    # LigandMPNN is not actually ligand-aware, which is the whole point of
    # using it rather than ProteinMPNN.
    use_atom_context: bool = True
    use_side_chain_context: bool = True


class BoltzCfg(Strict):
    binary: str = "boltz"
    cache_dir: str = "~/.boltz"
    recycling_steps: Annotated[int, Field(ge=1)] = 3
    diffusion_samples: Annotated[int, Field(ge=1)] = 1
    sampling_steps: Annotated[int, Field(ge=1)] = 200
    use_potentials: bool = False
    no_kernels: bool = True
    precision: Literal["32", "16", "bf16"] = "32"
    max_tokens: Annotated[int, Field(ge=1)] = 400
    accelerator: Literal["gpu", "cpu"] = "gpu"
    devices: Annotated[int, Field(ge=1)] = 1
    # Where the pre-downloaded weights live. The wrapper symlinks them into the
    # run's own cache dir rather than copying 6 GB per run, and rather than
    # letting Boltz try to download them -- which fails on a compute node with
    # no outbound internet.
    assets_dir: str | None = None

    @field_validator("precision")
    @classmethod
    def _no_bf16_silently(cls, v: str) -> str:
        # Turing (sm_75) has no native bf16. This is not enforced as an error
        # because the config may be used on newer cards; phase 0 settles it and
        # writes the answer into runs/hardware_notes.md.
        return v


class ProtonationCfg(Strict):
    engine: Literal["pdbfixer", "reduce", "none"] = "pdbfixer"
    ph: float = 7.0


class ScoreWeights(Strict):
    confidence: Fraction = 0.70
    geometry: Fraction = 0.30

    @model_validator(mode="after")
    def _sums_to_one(self) -> ScoreWeights:
        total = self.confidence + self.geometry
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"scoring.weights must sum to 1.0, got {total} "
                f"(confidence={self.confidence}, geometry={self.geometry})"
            )
        return self


class GeometryWeights(Strict):
    distance: Fraction = 0.60
    angle: Fraction = 0.40

    @model_validator(mode="after")
    def _sums_to_one(self) -> GeometryWeights:
        total = self.distance + self.angle
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"scoring.geometry_weights must sum to 1.0, got {total}"
            )
        return self


ConfidenceMetric = Literal["complex_ipde", "iptm", "ligand_iptm", "motif_pae"]


class ConfidenceCfg(Strict):
    metric: ConfidenceMetric = "complex_ipde"
    ipde_best: float = 0.5
    ipde_worst: float = 5.0

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceCfg:
        if self.ipde_worst <= self.ipde_best:
            raise ValueError(
                f"scoring.confidence.ipde_worst ({self.ipde_worst}) must exceed "
                f"ipde_best ({self.ipde_best}); lower iPDE is better."
            )
        return self


class DistanceCfg(Strict):
    ideal: Positive = 2.9
    tolerated: Positive = 3.5
    fail: Positive = 4.5
    clash: Positive = 2.2

    @model_validator(mode="after")
    def _ordered(self) -> DistanceCfg:
        if not self.clash < self.ideal < self.tolerated < self.fail:
            raise ValueError(
                "scoring.distance must satisfy clash < ideal < tolerated < fail, got "
                f"clash={self.clash}, ideal={self.ideal}, "
                f"tolerated={self.tolerated}, fail={self.fail}"
            )
        return self


AngleRule = Literal["split_by_donor_type", "acceptor_side_always"]


class AngleCfg(Strict):
    tolerance: Positive = 60.0
    rule: AngleRule = "acceptor_side_always"


class ScoringCfg(Strict):
    weights: ScoreWeights = ScoreWeights()
    geometry_weights: GeometryWeights = GeometryWeights()
    confidence: ConfidenceCfg = ConfidenceCfg()
    distance: DistanceCfg = DistanceCfg()
    angle: AngleCfg = AngleCfg()
    aggregate: Literal["mean", "min"] = "mean"


class WiggleCfg(Strict):
    enabled: bool = True
    parents_per_cycle: Annotated[int, Field(ge=1)] = 4
    source: Literal["rfdiffusion_backbone", "boltz_prediction"] = "rfdiffusion_backbone"
    # How many re-noised backbones to make from each parent. null means "use
    # rfdiffusion.designs_per_parent", which is what cycle 0 uses. They are
    # separate because the two jobs are different sizes: cycle 0 wants breadth
    # from one seed, while a wiggle wants a few neighbours of something already
    # good, and tying them together forces one number to serve both.
    designs_per_parent: Annotated[int, Field(ge=1)] | None = None


class MutateCfg(Strict):
    enabled: bool = True
    parents_per_cycle: Annotated[int, Field(ge=1)] = 2
    shell_radius: Positive = 5.0
    alphabet: str = "ADEFGHIKLMNQRSTVWY"
    max_mutants_per_parent: Annotated[int, Field(ge=1)] = 80

    @field_validator("alphabet")
    @classmethod
    def _valid_alphabet(cls, v: str) -> str:
        legal = set("ACDEFGHIKLMNPQRSTVWY")
        bad = sorted(set(v) - legal)
        if bad:
            raise ValueError(
                f"search.mutate.alphabet has non-amino-acid letters: {bad}"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"search.mutate.alphabet has duplicates: {v}")
        if not v:
            raise ValueError("search.mutate.alphabet is empty")
        return v


class SearchCfg(Strict):
    keep_per_cycle: Annotated[int, Field(ge=1)] = 2
    wiggle: WiggleCfg = WiggleCfg()
    mutate: MutateCfg = MutateCfg()

    @model_validator(mode="after")
    def _at_least_one_branch(self) -> SearchCfg:
        if not self.wiggle.enabled and not self.mutate.enabled:
            raise ValueError(
                "both search branches are disabled; the search cannot expand"
            )
        return self


class PipelineCfg(Strict):
    run: RunCfg = RunCfg()
    rfd3: RFD3Cfg
    # Optional: only needed on a cluster that has RFdiffusion instead of RFD3.
    rfdiffusion: RFdiffusionCfg | None = None
    ligandmpnn: LigandMPNNCfg
    boltz: BoltzCfg = BoltzCfg()
    protonation: ProtonationCfg = ProtonationCfg()
    scoring: ScoringCfg = ScoringCfg()
    search: SearchCfg = SearchCfg()

    @model_validator(mode="after")
    def _protonation_matches_angle_rule(self) -> PipelineCfg:
        # The split rule measures D-H...A on fixed-H donors, which needs
        # hydrogens. Without a protonator that is unmeasurable, and silently
        # falling back to the heavy-atom angle would apply a 180 deg target to
        # a quantity whose ideal is ~109-120 deg -- optimising in the wrong
        # direction. See PIPELINE.md.
        rule = self.scoring.angle.rule
        if rule == "split_by_donor_type" and self.protonation.engine == "none":
            raise ValueError(
                "scoring.angle.rule='split_by_donor_type' needs hydrogens, but "
                "protonation.engine='none'. Either set an engine (pdbfixer or "
                "reduce) or use rule='acceptor_side_always', which needs none."
            )
        return self


# --------------------------------------------------------------------------
# targets/<name>.yaml
# --------------------------------------------------------------------------


class Seed(Strict):
    id: str
    pdb: str
    # RFD3's contig is a total-length range such as "260-260", not
    # RFdiffusion's segment specification. The residues to keep are named by
    # motif.fixed, and which of their atoms to pin by motif.fixed_atoms, so
    # they are not repeated here.
    contig: str

    @field_validator("contig")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("seed.contig is empty")
        return v

    @property
    def max_length(self) -> int:
        """The longest protein this contig can produce, for the token budget."""
        try:
            parts = [int(x) for x in self.contig.split("-")]
        except ValueError as exc:
            raise ValueError(
                f"seed {self.id}: cannot read a length range from contig "
                f"{self.contig!r}; RFD3 expects something like '260-260'"
            ) from exc
        return max(parts)


class Ligand(Strict):
    id: str
    ccd: str | None = None
    smiles: str | None = None
    reference_sdf: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Ligand:
        if self.ccd is None and self.smiles is None:
            raise ValueError(
                f"ligand {self.id!r} needs either a ccd code or a smiles string"
            )
        if self.ccd is None and self.reference_sdf is None:
            # On the SMILES path Boltz assigns its own atom names and the names
            # in the target spec do not survive. The SDF is what the graph match
            # in tracking.py maps against; without it, H-bond atoms cannot be
            # resolved at all.
            raise ValueError(
                f"ligand {self.id!r} uses smiles with no reference_sdf. Boltz "
                "assigns its own atom names on the SMILES path, so the names in "
                "hbonds cannot be resolved. Supply reference_sdf, or use a CCD code."
            )
        return self

    @property
    def uses_ccd(self) -> bool:
        return self.ccd is not None


class ResidueRef(Strict):
    chain: str
    resnum: int


class AtomSpec(Strict):
    """An atom named in the target spec, before it is resolved to coordinates."""

    site: Literal["protein", "ligand"]
    atom: str
    chain: str | None = None
    resnum: int | None = None

    @model_validator(mode="after")
    def _protein_needs_residue(self) -> AtomSpec:
        if self.site == "protein" and (self.chain is None or self.resnum is None):
            raise ValueError(
                f"protein atom {self.atom!r} needs both chain and resnum"
            )
        if self.site == "ligand" and (self.chain is not None or self.resnum is not None):
            raise ValueError(
                f"ligand atom {self.atom!r} must not carry chain/resnum; it is "
                "identified by atom name within the ligand"
            )
        return self


class HBond(Strict):
    id: str
    donor: AtomSpec
    acceptor: AtomSpec
    donor_antecedent: AtomSpec | None = None
    acceptor_antecedent: AtomSpec | None = None
    rotatable_donor: bool = False
    required: bool = True
    acceptor_ideal_angle: float | None = None

    @model_validator(mode="after")
    def _antecedent_present_for_measurable_angle(self) -> HBond:
        if self.donor.site == self.acceptor.site == "ligand":
            raise ValueError(
                f"hbond {self.id!r} has both atoms on the ligand; that is an "
                "intramolecular contact, not an interface bond"
            )
        return self

    def angle_is_acceptor_side(self, rule: AngleRule) -> bool:
        """Which angle this bond's score is measured on, given the config rule."""
        if rule == "acceptor_side_always":
            return True
        return self.rotatable_donor

    def validate_for_rule(self, rule: AngleRule) -> None:
        """Raise unless this bond can actually be measured under `rule`.

        Called at load time so a bond that cannot be scored is a startup error,
        not a NaN discovered after a GPU-hour.
        """
        if self.angle_is_acceptor_side(rule):
            if self.acceptor_antecedent is None:
                raise ValueError(
                    f"hbond {self.id!r} is measured at the acceptor under "
                    f"rule={rule!r} but has no acceptor_antecedent"
                )
            if self.acceptor_ideal_angle is None:
                raise ValueError(
                    f"hbond {self.id!r} is measured at the acceptor under "
                    f"rule={rule!r} but has no acceptor_ideal_angle"
                )


class Target(Strict):
    name: str
    seeds: list[Seed]
    ligand: Ligand
    motif: MotifCfg
    hbonds: list[HBond]

    @field_validator("seeds")
    @classmethod
    def _seeds_unique(cls, v: list[Seed]) -> list[Seed]:
        if not v:
            raise ValueError("target.seeds is empty; there is nothing to search")
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate seed ids: {ids}")
        return v

    @field_validator("hbonds")
    @classmethod
    def _hbonds_unique(cls, v: list[HBond]) -> list[HBond]:
        if not v:
            raise ValueError(
                "target.hbonds is empty; the geometry half of the score would "
                "be undefined"
            )
        ids = [b.id for b in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate hbond ids: {ids}")
        return v

    @model_validator(mode="after")
    def _hbond_residues_are_fixed(self) -> Target:
        # A bond whose protein partner is free to mutate can be destroyed by the
        # mutation branch, which would then be optimising against its own score.
        fixed = {(r.chain, r.resnum) for r in self.motif.fixed}
        for bond in self.hbonds:
            for role, spec in (("donor", bond.donor), ("acceptor", bond.acceptor)):
                if spec.site != "protein":
                    continue
                key = (spec.chain, spec.resnum)
                if key not in fixed:
                    raise ValueError(
                        f"hbond {bond.id!r} has its {role} on {spec.chain}{spec.resnum}, "
                        "which is not in motif.fixed. The mutation branch could "
                        "mutate the bond away. Add it to motif.fixed."
                    )
        return self


class MotifCfg(Strict):
    fixed: list[ResidueRef]
    # Which atoms of each motif residue RFD3 pins, keyed by "<chain><resnum>".
    # A residue absent from this map is pinned whole ("ALL"), the conservative
    # reading. Atom-level freezing is an RFD3 capability; RFdiffusion cannot do
    # it and ignores this.
    fixed_atoms: dict[str, str] = {}

    @field_validator("fixed")
    @classmethod
    def _nonempty_unique(cls, v: list[ResidueRef]) -> list[ResidueRef]:
        if not v:
            raise ValueError("motif.fixed is empty; this is not a motif-anchored run")
        keys = [(r.chain, r.resnum) for r in v]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate residues in motif.fixed: {keys}")
        return v

    def as_keys(self) -> set[tuple[str, int]]:
        return {(r.chain, r.resnum) for r in self.fixed}

    def labels(self) -> set[str]:
        return {f"{r.chain}{r.resnum}" for r in self.fixed}

    @model_validator(mode="after")
    def _fixed_atoms_match_fixed(self) -> MotifCfg:
        unknown = sorted(set(self.fixed_atoms) - self.labels())
        if unknown:
            raise ValueError(
                f"motif.fixed_atoms names {unknown}, which are not in "
                f"motif.fixed ({sorted(self.labels())}). An atom selection for "
                "a residue that is not held would be silently ignored."
            )
        return self


Target.model_rebuild()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


class Config(Strict):
    """Both files, validated together."""

    pipeline: PipelineCfg
    target: Target
    pipeline_path: str
    target_path: str


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def load_pipeline(path: str | Path) -> PipelineCfg:
    return PipelineCfg.model_validate(_read_yaml(Path(path)))


def load_target(path: str | Path) -> Target:
    return Target.model_validate(_read_yaml(Path(path)))


def load(pipeline_path: str | Path, target_path: str | Path) -> Config:
    """Load and cross-validate both config files.

    Cross-validation is the point of doing it here: a target whose bonds cannot
    be measured under the pipeline's angle rule is rejected now rather than
    after the first Boltz-2 call.
    """
    pipeline = load_pipeline(pipeline_path)
    target = load_target(target_path)

    rule = pipeline.scoring.angle.rule
    for bond in target.hbonds:
        bond.validate_for_rule(rule)

    return Config(
        pipeline=pipeline,
        target=target,
        pipeline_path=str(pipeline_path),
        target_path=str(target_path),
    )

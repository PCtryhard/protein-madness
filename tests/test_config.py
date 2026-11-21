"""Config validation: a bad config is one error at startup, not a crash later.

Also the rule the brief states and this repo greps for: every number used in a
computation comes from config. `test_no_numeric_literals_in_scoring` is that
grep, as a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import PipelineCfg, Target, load, load_pipeline, load_target

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the shipped files load
# --------------------------------------------------------------------------


def test_shipped_pipeline_loads(pipeline_path):
    cfg = load_pipeline(pipeline_path)
    assert cfg.scoring.weights.confidence + cfg.scoring.weights.geometry == 1.0
    assert cfg.boltz.max_tokens >= 1


@pytest.mark.parametrize(
    "target", ["config/targets/5fqj.yaml", "config/targets/example_target.yaml"]
)
def test_shipped_targets_load(target):
    loaded = load_target(ROOT / target)
    assert loaded.seeds
    assert loaded.hbonds
    assert loaded.motif.fixed


@pytest.mark.parametrize(
    "target", ["config/targets/5fqj.yaml", "config/targets/example_target.yaml"]
)
def test_shipped_targets_cross_validate_against_the_shipped_pipeline(
    pipeline_path, target
):
    """Every shipped target must load under the shipped angle rule.

    This is the check that caught the example target: the brief's example omits
    `acceptor_ideal_angle` on its fixed-hydrogen bond, which is fine under
    split_by_donor_type and invalid under the acceptor_side_always default. A
    documented example that cannot load is worse than no example.
    """
    cfg = load(pipeline_path, ROOT / target)
    for bond in cfg.target.hbonds:
        bond.validate_for_rule(cfg.pipeline.scoring.angle.rule)


@pytest.mark.parametrize(
    "target", ["config/targets/5fqj.yaml", "config/targets/example_target.yaml"]
)
def test_shipped_targets_also_load_under_the_other_angle_rule(pipeline_path, target):
    """Flipping scoring.angle.rule must stay a one-line config change."""
    import yaml as _yaml

    raw = _yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["angle"]["rule"] = "split_by_donor_type"
    pipeline = PipelineCfg.model_validate(raw)
    loaded = load_target(ROOT / target)
    for bond in loaded.hbonds:
        bond.validate_for_rule(pipeline.scoring.angle.rule)


def test_real_target_cross_validates_against_the_real_pipeline(pipeline_path):
    """The pairing that actually runs: 5fqj under the shipped angle rule."""
    cfg = load(pipeline_path, ROOT / "config/targets/5fqj.yaml")
    assert cfg.target.name == "5fqj"
    for bond in cfg.target.hbonds:
        bond.validate_for_rule(cfg.pipeline.scoring.angle.rule)


def test_every_hbond_id_is_unique_across_shipped_targets():
    for target in ("config/targets/5fqj.yaml", "config/targets/example_target.yaml"):
        loaded = load_target(ROOT / target)
        ids = [b.id for b in loaded.hbonds]
        assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------
# a typo is an error, not a silently ignored setting
# --------------------------------------------------------------------------


def test_unknown_key_is_rejected(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["weigths"] = {"confidence": 0.5}   # deliberate typo
    with pytest.raises(ValidationError, match="weigths"):
        PipelineCfg.model_validate(raw)


def test_weights_must_sum_to_one(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["weights"] = {"confidence": 0.8, "geometry": 0.3}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        PipelineCfg.model_validate(raw)


def test_geometry_weights_must_sum_to_one(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["geometry_weights"] = {"distance": 0.5, "angle": 0.4}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        PipelineCfg.model_validate(raw)


def test_distance_thresholds_must_be_ordered(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["distance"] = {"ideal": 3.6, "tolerated": 3.5, "fail": 4.5}
    with pytest.raises(ValidationError, match="ideal < tolerated < fail"):
        PipelineCfg.model_validate(raw)


def test_ipde_best_must_be_below_worst(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["confidence"]["ipde_worst"] = 0.1
    with pytest.raises(ValidationError, match="must exceed"):
        PipelineCfg.model_validate(raw)


def test_min_delta_at_noise_level_is_rejected(pipeline_path):
    """min_delta has to exist, otherwise floating-point noise in the seventh
    decimal registers as improvement and the search never stops."""
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["run"]["min_delta"] = 1e-12
    with pytest.raises(ValidationError, match="floating-point noise"):
        PipelineCfg.model_validate(raw)


def test_partial_T_must_be_below_diffuser_T(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["rfdiffusion"]["partial_T"] = 50
    raw["rfdiffusion"]["diffuser_T"] = 50
    with pytest.raises(ValidationError, match="must be below"):
        PipelineCfg.model_validate(raw)


def test_both_branches_disabled_is_rejected(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["search"]["wiggle"]["enabled"] = False
    raw["search"]["mutate"]["enabled"] = False
    with pytest.raises(ValidationError, match="cannot expand"):
        PipelineCfg.model_validate(raw)


def test_bad_alphabet_is_rejected(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["search"]["mutate"]["alphabet"] = "ABZ"
    with pytest.raises(ValidationError, match="non-amino-acid"):
        PipelineCfg.model_validate(raw)


def test_duplicate_gpus_rejected(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["run"]["gpus"] = [0, 0]
    with pytest.raises(ValidationError, match="duplicates"):
        PipelineCfg.model_validate(raw)


# --------------------------------------------------------------------------
# the angle rule / protonation pairing
# --------------------------------------------------------------------------


def test_split_rule_with_no_protonator_is_rejected(pipeline_path):
    """The split rule needs hydrogens. Falling back to the heavy-atom angle
    would apply a 180 deg target to a quantity whose ideal is ~109-120 deg."""
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["angle"]["rule"] = "split_by_donor_type"
    raw["protonation"]["engine"] = "none"
    with pytest.raises(ValidationError, match="needs hydrogens"):
        PipelineCfg.model_validate(raw)


def test_acceptor_side_rule_needs_no_protonator(pipeline_path):
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    raw["scoring"]["angle"]["rule"] = "acceptor_side_always"
    raw["protonation"]["engine"] = "none"
    cfg = PipelineCfg.model_validate(raw)
    assert cfg.scoring.angle.rule == "acceptor_side_always"


def test_acceptor_side_rule_requires_an_ideal_angle_on_every_bond(
    pipeline_path, tmp_path
):
    """A bond that cannot be scored is a startup error, not a NaN found after
    a GPU-hour."""
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    del raw["hbonds"][0]["acceptor_ideal_angle"]
    target_path = tmp_path / "broken.yaml"
    target_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="acceptor_ideal_angle"):
        load(pipeline_path, target_path)


# --------------------------------------------------------------------------
# target-spec invariants
# --------------------------------------------------------------------------


def test_hbond_residue_must_be_in_the_motif(tmp_path):
    """A bond whose protein partner is free to mutate could be destroyed by
    the mutation branch, which would then optimise against its own score."""
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["motif"]["fixed"] = [{"chain": "A", "resnum": 229}]   # drop HIS291
    with pytest.raises(ValidationError, match="not in motif.fixed"):
        Target.model_validate(raw)


def test_empty_seeds_rejected(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["seeds"] = []
    with pytest.raises(ValidationError, match="nothing to search"):
        Target.model_validate(raw)


def test_empty_hbonds_rejected():
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["hbonds"] = []
    with pytest.raises(ValidationError, match="geometry half"):
        Target.model_validate(raw)


def test_duplicate_seed_ids_rejected():
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["seeds"].append(dict(raw["seeds"][0]))
    with pytest.raises(ValidationError, match="duplicate seed ids"):
        Target.model_validate(raw)


def test_smiles_ligand_without_reference_sdf_rejected():
    """On the SMILES path Boltz assigns its own atom names, so the names in
    hbonds cannot be resolved without the reference SDF."""
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["ligand"] = {
        "id": "6NT", "ccd": None, "smiles": "c1ccccc1", "reference_sdf": None
    }
    with pytest.raises(ValidationError, match="reference_sdf"):
        Target.model_validate(raw)


def test_ligand_needs_ccd_or_smiles():
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["ligand"] = {"id": "X", "ccd": None, "smiles": None, "reference_sdf": None}
    with pytest.raises(ValidationError, match="ccd code or a smiles"):
        Target.model_validate(raw)


def test_protein_atom_spec_needs_chain_and_resnum():
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["hbonds"][0]["acceptor"] = {"site": "protein", "atom": "OD2"}
    with pytest.raises(ValidationError, match="chain and resnum"):
        Target.model_validate(raw)


def test_ligand_atom_spec_must_not_carry_a_residue():
    raw = yaml.safe_load(
        (ROOT / "config/targets/5fqj.yaml").read_text(encoding="utf-8")
    )
    raw["hbonds"][0]["donor"] = {
        "site": "ligand", "atom": "N3", "chain": "A", "resnum": 400
    }
    with pytest.raises(ValidationError, match="must not carry"):
        Target.model_validate(raw)


# --------------------------------------------------------------------------
# the literal grep, as a test
# --------------------------------------------------------------------------

_SCORING_FILES = [
    "src/scoring/combine.py",
    "src/scoring/hbond.py",
    "src/scoring/confidence.py",
]

# Numbers that are definitions rather than choices, and so are allowed to be
# literals in scoring code:
#   0, 1      the clip bounds, and the identity/zero in a comparison
#   2         the divisor of a two-way mean (PAE is not symmetric)
#   180.0     the geometric definition of D-H...A linearity, not a tunable
#   1.3       the longest covalent X-H bond, for finding a donor's hydrogen.
#             Arguably a tunable; it is a default argument, overridable by the
#             caller, and it selects an atom rather than scoring anything.
_ALLOWED_NUMBERS = {0, 1, 2, 0.0, 1.0, 180.0, 1.3}


def _numeric_literals(path: Path) -> list[tuple[int, object]]:
    """Every numeric constant in executable code, via the AST.

    Using the AST rather than a regex over lines is what makes this a usable
    guard: docstrings, comments and strings like "utf-8" are not code, and a
    grep that flags them gets switched off within a day.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            found.append((node.lineno, node.value))
    return found


def test_no_numeric_literals_in_scoring():
    """Every number used in a computation comes from config.

    A literal float inside a scoring function is a bug, and each phase review
    should grep for one. This is that grep, run as a test so it cannot be
    forgotten. Anything genuinely a definition goes in `_ALLOWED_NUMBERS` with
    a comment saying why it is not a choice.
    """
    offenders: list[str] = []
    for relative in _SCORING_FILES:
        path = ROOT / relative
        for lineno, value in _numeric_literals(path):
            if value in _ALLOWED_NUMBERS:
                continue
            offenders.append(f"{relative}:{lineno}: {value!r}")
    assert not offenders, (
        "numeric literals in scoring code -- move them to config/pipeline.yaml "
        "or justify them in _ALLOWED_NUMBERS:\n" + "\n".join(offenders)
    )

"""The seven shipped variants: every target resolves against its own seed.

This is the phase-1 acceptance check run across all seven motif folders rather
than just the reference one. What it really guards is the atom names: ASP has
OD2/CG, GLU has OE2/CD, HIS has NE2/CD2, ASN has ND2/CG, GLN has NE2/CD.
Getting one wrong gives an AtomResolutionError at the first real design, hours
into a run; these tests move that to a second here.

The measured values are deliberately asserted as they ARE, clashes included,
because the motif files carry unrelaxed mutations. See scripts/build_targets.py
for why that is expected and why the ideal angles are not read out of them.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.config import load
from src.scoring.combine import combine
from src.scoring.hbond import measure_all
from src.stages.boltz import ligand_atom_count
from src.tracking import identity_provenance, read_structure
from tests.paths import ROOT, seed_for, variant_targets


def _load_variant(pipeline_path, target_rel: str):
    cfg = load(pipeline_path, ROOT / target_rel)
    seed = cfg.target.seeds[0]
    structure = read_structure(ROOT / seed.pdb, cfg.target.ligand.ccd)
    ligand_atoms = sorted(
        {
            spec.atom
            for bond in cfg.target.hbonds
            for spec in (bond.donor, bond.acceptor,
                         bond.donor_antecedent, bond.acceptor_antecedent)
            if spec is not None and spec.site == "ligand"
        }
    )
    provenance = identity_provenance(
        design_id=f"seed:{cfg.target.name}",
        scaffold_id=cfg.target.name,
        structure=structure,
        motif=cfg.target.motif,
        ligand_atoms=ligand_atoms,
    )
    return cfg, structure, provenance


def _measure(pipeline_path, target_rel: str):
    cfg, structure, provenance = _load_variant(pipeline_path, target_rel)
    measurements = measure_all(
        cfg.target.hbonds, structure, provenance, cfg.pipeline.scoring.angle.rule
    )
    return cfg, structure, {m.bond_id: m for m in measurements}


def test_at_least_one_variant_has_a_target():
    """Not a fixed count. A variant only gets a target file when its seed's
    motif actually touches the ligand, so the number tracks the data: right
    now only 01 has such a seed (via Yazan's .pre-boltz file), because the
    refolded pockets for the others put the ligand 25-41 A away."""
    targets = variant_targets()
    assert targets, "no variant targets generated at all"
    assert "config/targets/01_his_asp.yaml" in targets


@pytest.mark.parametrize("target", variant_targets())
def test_every_variant_resolves_all_its_bonds(pipeline_path, target):
    """Every named atom in every variant exists in that variant's seed."""
    cfg, _, by_id = _measure(pipeline_path, target)
    assert len(by_id) == len(cfg.target.hbonds)
    for measurement in by_id.values():
        assert measurement.distance > 0.0


@pytest.mark.parametrize("target", variant_targets())
def test_every_variant_fixes_every_hbond_residue(pipeline_path, target):
    """Config enforces this, but assert it per variant: a bond on a residue
    the mutation branch could change would let the search destroy its own
    score."""
    cfg, _, _ = _load_variant(pipeline_path, target)
    fixed = cfg.target.motif.as_keys()
    for bond in cfg.target.hbonds:
        for spec in (bond.donor, bond.acceptor):
            if spec.site == "protein":
                assert (spec.chain, spec.resnum) in fixed


@pytest.mark.parametrize("target", variant_targets())
def test_every_variant_fits_the_token_budget(pipeline_path, target):
    """The longest contig expansion plus the ligand must stay inside
    boltz.max_tokens, or the run refuses every design it generates."""
    cfg, _, _ = _load_variant(pipeline_path, target)
    atoms = ligand_atom_count(cfg.target.ligand)
    # RFD3's contig is a total length range, so the longest protein it can
    # produce is simply its upper bound.
    longest = cfg.target.seeds[0].max_length
    assert longest + atoms <= cfg.pipeline.boltz.max_tokens, (
        f"{target}: longest expansion is {longest} + {atoms} ligand = "
        f"{longest + atoms} tokens, over max_tokens {cfg.pipeline.boltz.max_tokens}"
    )



def test_every_seed_is_usable(pipeline_path):
    """Every variant's motif residues must actually touch its ligand.

    THIS IS A DATA CHECK, NOT A CODE CHECK. It goes red when a seed file is
    replaced with one whose ligand is not in the pocket, which is exactly what
    happened to variants 01-03: their refolded pockets put the motif 25-39 A
    from the ligand. Fix the seed, not the test -- and see the refusal that
    `python scripts/build_targets.py` prints for the two likely causes.
    """
    module = _build_targets_module()
    broken = {}
    for target in variant_targets():
        cfg, _, by_id = _measure(pipeline_path, target)
        far = [
            (m.bond_id, m.distance)
            for m in by_id.values()
            if m.required and m.distance > module.SEED_UNUSABLE_DISTANCE
        ]
        if far:
            broken[cfg.target.name] = far
    assert not broken, (
        "these seeds have required contacts beyond "
        f"{module.SEED_UNUSABLE_DISTANCE} A, so the motif and the ligand are "
        "not interacting:\n"
        + "\n".join(
            f"  {name}: "
            + ", ".join(f"{bond} at {dist:.1f} A" for bond, dist in far)
            for name, far in sorted(broken.items())
        )
    )


def test_the_base_contact_is_crystal_geometry(pipeline_path):
    """Variant 01's seed is the un-refolded motif, so its two contacts are the
    real 5FQJ values and its ideals match them."""
    _, _, by_id = _measure(pipeline_path, "config/targets/01_his_asp.yaml")
    base = by_id["lig_n3_to_asp229_od2"]
    assert base.distance == pytest.approx(2.734, abs=0.005)
    assert base.angle == pytest.approx(106.6, abs=0.1)





def test_generated_targets_are_not_stale():
    """No target file may differ from what the generator would write now.

    Deliberately tolerant of the generator REFUSING a variant: a refused seed
    is a data problem that test_every_seed_is_usable reports, and failing here
    for it too would just make one problem look like two. What this catches is
    the different problem of an edit to the generator that was never re-run.
    """
    result = subprocess.run(
        [sys.executable, "scripts/build_targets.py", "--check"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    combined = result.stdout + result.stderr
    assert "out of date" not in combined, (
        "config/targets/ is stale; run `python scripts/build_targets.py`\n"
        + combined
    )


# --------------------------------------------------------------------------
# the generator's ideal-angle provenance, and the refold handoff
# --------------------------------------------------------------------------


def _build_targets_module():
    """Import scripts/build_targets.py as a module."""
    import importlib.util

    if "build_targets" in sys.modules:
        return sys.modules["build_targets"]
    path = ROOT / "scripts" / "build_targets.py"
    spec = importlib.util.spec_from_file_location("build_targets", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because @dataclass resolves its own module out of
    # sys.modules while the class body runs.
    sys.modules["build_targets"] = module
    spec.loader.exec_module(module)
    return module


def test_plausible_window_is_the_scoring_window(pipeline):
    """"Plausible" must mean "would score above zero", i.e. bounded by
    `tolerated`, not by the looser `fail` gate. A 3.9 A contact sits inside the
    4.5 A fail gate and still scores zero, so its angle is not a target."""
    from src.scoring.hbond import distance_term

    distance = pipeline.scoring.distance
    assert distance_term(distance.tolerated, distance) == pytest.approx(0.0)
    assert distance_term(distance.tolerated - 0.1, distance) > 0.0




def test_an_unknown_residue_type_is_a_hard_error(tmp_path, pipeline):
    """A residue the chemistry table does not know must stop the script, not
    be guessed at."""
    module = _build_targets_module()
    residues = {229: "ASP", 291: "TRP"}      # TRP is not a known donor here
    with pytest.raises(SystemExit, match="chemistry table"):
        module.plan_bonds(residues)



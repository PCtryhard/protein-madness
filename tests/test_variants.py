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


def test_there_are_seven_variants():
    assert len(variant_targets()) == 7


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
    longest = 0
    for segment in cfg.target.seeds[0].contigs.split("/"):
        if segment.startswith("A"):
            longest += 1                      # a single fixed motif residue
        else:
            longest += int(segment.split("-")[1])
    assert longest + atoms <= cfg.pipeline.boltz.max_tokens, (
        f"{target}: longest expansion is {longest} + {atoms} ligand = "
        f"{longest + atoms} tokens, over max_tokens {cfg.pipeline.boltz.max_tokens}"
    )


def test_the_base_is_asp_or_glu_and_the_donor_varies(pipeline_path):
    """The seven are the cross of two bases and three donors, plus variant 01
    which has no residue at 261."""
    bases, donors = set(), set()
    for target in variant_targets():
        _, structure, _ = _load_variant(pipeline_path, target)
        bases.add(structure.residues[("A", 229)])
        donors.add(structure.residues[("A", 291)])
    assert bases == {"ASP", "GLU"}
    assert donors == {"HIS", "ASN", "GLN"}


def test_only_variant_01_lacks_the_nitro_histidine(pipeline_path):
    without = []
    for target in variant_targets():
        cfg, structure, _ = _load_variant(pipeline_path, target)
        if ("A", 261) not in structure.residues:
            without.append(cfg.target.name)
    assert without == ["01_his_asp"]


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


def test_the_unmutated_base_contact_is_crystal_geometry(pipeline_path):
    """Wherever ASP229 has not been refolded, its contact is the real 5FQJ
    value. Variant 04 is the reference; 01-03 were replaced by refolds whose
    ligand is 25 A away, which test_every_seed_is_usable covers."""
    _, _, by_id = _measure(pipeline_path, "config/targets/04_gln_asp_his.yaml")
    base = by_id["lig_n3_to_asp229_od2"]
    assert base.distance == pytest.approx(2.734, abs=0.005)
    assert base.angle == pytest.approx(106.6, abs=0.1)


def test_the_modelled_mutations_really_do_clash(pipeline_path):
    """Not a wish, a record. These are the values in the shipped motif files,
    and they are why build_targets.py does not read ideals out of them.

    If this test ever fails, someone has repacked the motif files -- which
    would be good news, and the ideal angles should then be re-derived from
    the repacked files rather than from chemistry defaults.
    """
    _, _, by_id = _measure(pipeline_path, "config/targets/05_his_glu_his.yaml")
    assert by_id["lig_n3_to_glu229_oe2"].distance == pytest.approx(1.677, abs=0.005)
    assert by_id["his261_ne2_to_lig_o11"].distance == pytest.approx(1.820, abs=0.005)


def test_the_asn_variant_points_the_donor_away(pipeline_path):
    """ASN291 ND2 sits 4.823 A from the ligand N1 in the un-refolded seed --
    past the 4.5 A fail threshold. The seed is never scored, but it says
    plainly that this variant starts with no donor contact at all."""
    _, _, by_id = _measure(pipeline_path, "config/targets/06_asn_glu_his.yaml")
    assert by_id["asn291_nd2_to_lig_n1"].distance == pytest.approx(4.823, abs=0.005)


def test_a_clashing_seed_would_gate_if_it_were_a_design(pipeline_path):
    """The seeds are inputs and are never scored. But if a DESIGN came out of
    the pipeline with variant 05's geometry, the clash gate must catch it --
    which is the whole reason that gate exists."""
    cfg, _, by_id = _measure(pipeline_path, "config/targets/05_his_glu_his.yaml")
    score = combine(1.0, list(by_id.values()), cfg.pipeline.scoring)
    assert score.gated
    assert "lig_n3_to_glu229_oe2" in score.gated_bonds
    assert score.total == 0.0


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


def test_generator_prefers_measured_when_the_seed_is_clean(tmp_path, pipeline):
    """The refold handoff, in one test: a seed whose contacts are real H-bonds
    yields `measured` ideals taken from that seed."""
    module = _build_targets_module()
    source = seed_for("config/targets/04_"
                       + [t for t in variant_targets()
                          if "/04_" in t][0].split("/04_")[1])
    clean = tmp_path / "clean.pdb"
    clean.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    residues = module.read_motif(clean)
    plans = module.plan_bonds(residues)
    module.measure(plans, clean, pipeline.scoring.distance.clash,
                   pipeline.scoring.distance.tolerated)

    # Wherever a contact is a plausible H-bond, the ideal is what was
    # measured in the seed rather than a stand-in.
    measured = [p for p in plans if p.source == "measured"]
    assert measured, "variant 04's base contact is crystal geometry"
    for plan in measured:
        assert plan.ideal == pytest.approx(plan.measured_angle, abs=0.05)


def test_generator_falls_back_and_says_so_when_the_seed_clashes(tmp_path, pipeline):
    module = _build_targets_module()
    source = seed_for("config/targets/05_"
                       + [t for t in variant_targets()
                          if "/05_" in t][0].split("/05_")[1])
    clashing = tmp_path / "clashing.pdb"
    clashing.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    residues = module.read_motif(clashing)
    plans = module.plan_bonds(residues)
    module.measure(plans, clashing, pipeline.scoring.distance.clash,
                   pipeline.scoring.distance.tolerated)

    by_id = {p.bond_id: p for p in plans}
    # GLU229 clashes at 1.677 A: no crystal value for a GLU there, so chemistry.
    glu = by_id["lig_n3_to_glu229_oe2"]
    assert glu.source == "chemistry"
    assert glu.note, "a fallback must explain itself"
    # HIS291 is a real crystal contact and survives unmutated.
    assert by_id["his291_ne2_to_lig_n1"].source == "measured"


def test_generator_flags_a_plausible_distance_with_an_implausible_angle(
    tmp_path, pipeline
):
    """A sidechain translated but not repacked gives a fine distance and a
    nonsense direction. That must not install a nonsense ideal silently."""
    import numpy as np

    module = _build_targets_module()
    source = seed_for("config/targets/04_"
                       + [t for t in variant_targets()
                          if "/04_" in t][0].split("/04_")[1])
    lines = source.read_text(encoding="utf-8").splitlines()

    def xyz(line):
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    ligand = {
        line[12:16].strip(): xyz(line)
        for line in lines
        if line.startswith("HETATM") and line[17:20].strip() == "6NT"
    }
    out = []
    for line in lines:
        if line.startswith("ATOM") and int(line[22:26]) == 229 \
                and line[12:16].strip() == "OD2":
            # Keep a good distance, wreck the direction: place OD2 along the
            # N2->N3 bond axis, which drives the acceptor-side angle far down.
            direction = ligand["N3"] - ligand["N2"]
            target = ligand["N3"] + direction / np.linalg.norm(direction) * 2.85
            line = line[:30] + f"{target[0]:8.3f}{target[1]:8.3f}{target[2]:8.3f}" \
                   + line[54:]
        out.append(line)
    bad = tmp_path / "bad.pdb"
    bad.write_text("\n".join(out) + "\n", encoding="utf-8")

    residues = module.read_motif(bad)
    plans = module.plan_bonds(residues)
    module.measure(plans, bad, pipeline.scoring.distance.clash,
                   pipeline.scoring.distance.tolerated)

    base = {p.bond_id: p for p in plans}["lig_n3_to_asp229_od2"]
    assert base.source == "measured", "the distance is fine, so it is measured"
    assert "SUSPICIOUS" in base.note, (
        f"angle {base.measured_angle:.1f} deg at {base.distance:.3f} A should "
        "have been flagged"
    )


def test_an_unknown_residue_type_is_a_hard_error(tmp_path, pipeline):
    """A residue the chemistry table does not know must stop the script, not
    be guessed at."""
    module = _build_targets_module()
    residues = {229: "ASP", 291: "TRP"}      # TRP is not a known donor here
    with pytest.raises(SystemExit, match="chemistry table"):
        module.plan_bonds(residues)


def test_a_seed_whose_ligand_is_far_away_is_refused(tmp_path, pipeline):
    """A required contact at 25 A is not a bad contact, it is not a contact.
    No stand-in ideal can rescue it, so the target must not be generated.

    This fired for real: the first refolded pockets for variants 01-03 had
    their motif residues 25-39 A from the ligand, and without this they would
    have produced seven runnable-looking targets, three of which asked
    RFdiffusion to scaffold a constellation in which nothing touches.
    """
    import numpy as np

    module = _build_targets_module()
    source = seed_for("config/targets/04_"
                       + [t for t in variant_targets()
                          if "/04_" in t][0].split("/04_")[1])
    lines = source.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith("HETATM") and line[17:20].strip() == "6NT":
            # translate the ligand 30 A away, leaving the protein where it is
            xyz = np.array([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])]) + np.array([30.0, 0.0, 0.0])
            line = line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:]
        out.append(line)
    far = tmp_path / "far.pdb"
    far.write_text("\n".join(out) + "\n", encoding="utf-8")

    residues = module.read_motif(far)
    plans = module.plan_bonds(residues)
    module.measure(plans, far, pipeline.scoring.distance.clash,
                   pipeline.scoring.distance.tolerated)

    broken = module.unusable_bonds(plans)
    assert broken, "a 30 A displacement should make the seed unusable"
    assert all(p.required for p in broken), "only required bonds can make it unusable"


def test_a_merely_bad_seed_is_not_refused(tmp_path, pipeline):
    """The distinction that matters: a 1.7 A clash or a 3.9 A stretch is a bad
    contact RFdiffusion can be asked to improve, and it still generates."""
    module = _build_targets_module()
    source = seed_for("config/targets/05_"
                       + [t for t in variant_targets()
                          if "/05_" in t][0].split("/05_")[1])
    residues = module.read_motif(source)
    plans = module.plan_bonds(residues)
    module.measure(plans, source, pipeline.scoring.distance.clash,
                   pipeline.scoring.distance.tolerated)
    assert module.unusable_bonds(plans) == []

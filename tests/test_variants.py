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
from tests.paths import ROOT, variant_targets


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


def test_variant_01_carries_the_crystal_geometry(pipeline_path):
    """01 is the only variant with no modelled mutation, so its two contacts
    are the real 5FQJ values and its ideals match them exactly."""
    _, _, by_id = _measure(pipeline_path, "config/targets/01_his_asp.yaml")
    base = by_id["lig_n3_to_asp229_od2"]
    donor = by_id["his291_ne2_to_lig_n1"]
    assert base.distance == pytest.approx(2.734, abs=0.005)
    assert base.angle == pytest.approx(106.6, abs=0.1)
    assert donor.distance == pytest.approx(3.084, abs=0.005)
    assert donor.angle == pytest.approx(126.6, abs=0.1)


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


def test_the_asn_variants_point_the_donor_away(pipeline_path):
    """ASN291 ND2 sits 4.823 A from the ligand N1 in the seed -- past the 4.5 A
    fail threshold. The seed is never scored, but it says plainly that these
    two variants start with no donor contact at all."""
    _, _, by_id = _measure(pipeline_path, "config/targets/03_asn_asp_his.yaml")
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


def test_generated_targets_are_up_to_date():
    """`build_targets.py --check` must pass, so an edit to the generator that
    was never re-run cannot slip through."""
    result = subprocess.run(
        [sys.executable, "scripts/build_targets.py", "--check"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        "config/targets/ is out of date; run `python scripts/build_targets.py`\n"
        + result.stdout
        + result.stderr
    )

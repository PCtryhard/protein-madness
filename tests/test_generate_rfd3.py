"""RFD3 input generation: de novo and partial diffusion.

The property that matters: **the catalytic motif is pinned in every mode.**
Everything else about a design is negotiable; that is not.
"""
from __future__ import annotations

import json

import pytest

from src.generate.rfd3 import (
    KEMP_BASE,
    KEMP_DONOR,
    KEMP_FIXED_ATOMS,
    KEMP_LIGAND,
    KEMP_STACK,
    MAX_PARTIAL_T,
    BackboneSpec,
    denovo,
    partial,
    rfd3_command,
    write_input_json,
)


def test_denovo_pins_base_donor_and_ligand():
    entry = denovo("d1").as_entry()
    fixed = entry["select_fixed_atoms"]
    assert fixed[KEMP_BASE] == "OD1,CG,OD2"
    assert fixed[KEMP_DONOR] == "OE1,CD,NE2"
    assert fixed[KEMP_LIGAND] == "ALL"
    assert entry["ligand"] == KEMP_LIGAND
    assert "partial_t" not in entry


def test_denovo_uses_the_full_length_budget():
    """Burial counts CA atoms near the base, so more protein is better."""
    assert denovo("d1").as_entry()["contig"] == "260-260"


def test_denovo_rejects_over_length():
    with pytest.raises(ValueError, match="competition limit"):
        denovo("d1", length=300)


def test_denovo_with_stack_adds_the_tryptophan():
    entry = denovo("d1", with_stack=True).as_entry()
    assert KEMP_STACK in entry["select_fixed_atoms"]
    assert entry["select_fixed_atoms"][KEMP_STACK] == "CD1,NE1,CE2"
    assert KEMP_STACK in entry["unindex"]


def test_partial_sets_partial_t_and_keeps_the_motif():
    entry = partial("p1", "parent.pdb", partial_t=5.0).as_entry()
    assert entry["partial_t"] == 5.0
    assert entry["input"] == "parent.pdb"
    assert entry["select_fixed_atoms"][KEMP_BASE] == KEMP_FIXED_ATOMS[KEMP_BASE]
    assert entry["select_fixed_atoms"][KEMP_LIGAND] == "ALL"


def test_partial_t_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        partial("p1", "parent.pdb", partial_t=0.0)


def test_large_partial_t_warns_but_is_allowed(caplog):
    """RFD3 recommends t <= 15 A; beyond that it is effectively de novo."""
    with caplog.at_level("WARNING"):
        spec = partial("p1", "parent.pdb", partial_t=MAX_PARTIAL_T + 5)
    assert spec.partial_t == MAX_PARTIAL_T + 5
    assert any("exceeds the recommended maximum" in r.message for r in caplog.records)


def test_unknown_motif_residue_is_refused():
    """Better to fail than to let RFD3 pick which atoms to hold."""
    with pytest.raises(ValueError, match="no fixed-atom specification"):
        BackboneSpec(name="x", input_pdb="in.pdb", motif_residues=("A999",))


def test_write_input_json_round_trips(tmp_path):
    path = write_input_json(
        tmp_path / "Kemp.json",
        [denovo("denovo_1"), partial("partial_1", "parent.pdb", partial_t=4.0)],
    )
    data = json.loads(path.read_text())
    assert set(data) == {"_comment", "denovo_1", "partial_1"}
    assert "partial_t" not in data["denovo_1"]
    assert data["partial_1"]["partial_t"] == 4.0


def test_write_input_json_uses_unix_line_endings(tmp_path):
    path = write_input_json(tmp_path / "Kemp.json", [denovo("d1")])
    assert b"\r\n" not in path.read_bytes()


def test_write_input_json_refuses_duplicate_names(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_input_json(tmp_path / "k.json", [denovo("same"), denovo("same")])


def test_comment_warns_about_engine_arguments(tmp_path):
    """RFD3 treats every key in this file as a design field, so engine
    arguments must go on the command line. The organisers hit this; so will we."""
    path = write_input_json(tmp_path / "Kemp.json", [denovo("d1")])
    comment = json.loads(path.read_text())["_comment"]
    assert "command line" in comment


def test_rfd3_command_matches_the_verified_invocation():
    command = rfd3_command("Kemp.json", ["denovo_1"], "out", seed=7, n_batches=4)
    for fragment in ("json_keys_subset=[denovo_1]", "seed=7", "n_batches=4",
                     "prevalidate_inputs=True", "cleanup_guideposts=True"):
        assert fragment in command


def test_rfd3_command_refuses_an_empty_key_list():
    """Without json_keys_subset RFD3 runs every entry in the file."""
    with pytest.raises(ValueError, match="json_keys_subset"):
        rfd3_command("Kemp.json", [], "out")


# ------------------------------------------- extra-contact conditioning arm

def test_extra_contacts_leaves_the_scored_geometry_alone():
    """The judges measure ASP OD2->N3 and GLN NE2->N1. Adding conditioning must
    not disturb either, or we trade a known metric for an unproven one."""
    from src.generate.rfd3 import denovo_extra_contacts

    plain = denovo("a").as_entry()
    extra = denovo_extra_contacts("b").as_entry()
    assert extra["select_fixed_atoms"][KEMP_BASE] == plain["select_fixed_atoms"][KEMP_BASE]
    assert extra["select_fixed_atoms"][KEMP_DONOR] == plain["select_fixed_atoms"][KEMP_DONOR]
    assert extra["select_fixed_atoms"][KEMP_LIGAND] == "ALL"
    assert extra["contig"] == plain["contig"]


def test_extra_contacts_targets_the_unengaged_nitro_oxygens():
    from src.generate.rfd3 import NITRO_OXYGENS, denovo_extra_contacts

    entry = denovo_extra_contacts("b").as_entry()
    assert entry["select_hbond_acceptor"] == {KEMP_LIGAND: ",".join(NITRO_OXYGENS)}
    assert entry["select_buried"] == KEMP_BASE


def test_extra_contacts_rejects_atoms_that_are_not_on_the_ligand():
    """A typo'd atom name would silently condition on nothing."""
    from src.generate.rfd3 import denovo_extra_contacts

    with pytest.raises(ValueError, match="not 6NT atoms"):
        denovo_extra_contacts("b", acceptors=("OXX",))


def test_extra_contacts_can_be_switched_off():
    from src.generate.rfd3 import denovo_extra_contacts

    entry = denovo_extra_contacts("b", acceptors=(), bury_base=False).as_entry()
    assert "select_hbond_acceptor" not in entry
    assert "select_buried" not in entry

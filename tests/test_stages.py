"""Stage wrappers: the parts that can be tested without a GPU or weights.

What is covered here is command construction, the Boltz-2 token refusal, input
and output file handling, and the protonation/angle-rule pairing. What is NOT
covered is any actual model call: that needs the cluster, and phase 0 is where
it gets verified. See README.md for which gates have passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import Ligand, MotifCfg, ResidueRef
from src.stages import boltz, ligandmpnn, protonate, rfd3
from src.stages.base import StageError, run_command

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the token budget: never silently clamped
# --------------------------------------------------------------------------


def test_token_count_is_protein_plus_ligand_heavy_atoms():
    assert boltz.token_count([260], 12) == 272
    assert boltz.token_count([100, 50], 8) == 158


def test_known_ligand_atom_count():
    assert boltz.ligand_atom_count(Ligand(id="6NT", ccd="6NT")) == 12


def test_unknown_ccd_refuses_rather_than_guessing(pipeline):
    """The token budget is the one guard against an OOM that kills a worker,
    and a guessed input makes it worthless."""
    with pytest.raises(boltz.BoltzTooLargeError, match="no recorded heavy-atom count"):
        boltz.ligand_atom_count(Ligand(id="X", ccd="ZZZ"))


def test_a_system_within_budget_is_accepted(pipeline):
    tokens = boltz.refuse_if_too_large(
        {"A": "M" * 260}, Ligand(id="6NT", ccd="6NT"), pipeline.boltz
    )
    assert tokens == 272


def test_the_real_5fqj_system_fits(pipeline):
    """260 protein + 12 ligand = 272 tokens, inside the 300 preference."""
    tokens = boltz.refuse_if_too_large(
        {"A": "M" * 260}, Ligand(id="6NT", ccd="6NT"), pipeline.boltz
    )
    assert tokens < 300


def test_an_oversized_system_is_refused_not_clamped(pipeline):
    with pytest.raises(boltz.BoltzTooLargeError, match="Refusing to submit"):
        boltz.refuse_if_too_large(
            {"A": "M" * 500}, Ligand(id="6NT", ccd="6NT"), pipeline.boltz
        )


def test_the_refusal_names_the_actual_counts(pipeline):
    with pytest.raises(boltz.BoltzTooLargeError) as exc:
        boltz.refuse_if_too_large(
            {"A": "M" * 500}, Ligand(id="6NT", ccd="6NT"), pipeline.boltz
        )
    message = str(exc.value)
    assert "512 tokens" in message
    assert "500 protein" in message
    assert "12 ligand" in message
    assert "max_tokens=400" in message


def test_exactly_at_the_limit_is_allowed(pipeline):
    cfg = pipeline.boltz.model_copy(update={"max_tokens": 272})
    assert boltz.refuse_if_too_large({"A": "M" * 260}, Ligand(id="6NT", ccd="6NT"), cfg) == 272


def test_one_token_over_the_limit_is_refused(pipeline):
    cfg = pipeline.boltz.model_copy(update={"max_tokens": 271})
    with pytest.raises(boltz.BoltzTooLargeError):
        boltz.refuse_if_too_large({"A": "M" * 260}, Ligand(id="6NT", ccd="6NT"), cfg)


# --------------------------------------------------------------------------
# the Boltz input file
# --------------------------------------------------------------------------


def test_input_yaml_has_the_shape_boltz_expects(tmp_path):
    path = boltz.write_input_yaml(
        tmp_path / "x.yaml", {"A": "MKVL"}, Ligand(id="6NT", ccd="6NT")
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["sequences"][0]["protein"]["sequence"] == "MKVL"
    assert raw["sequences"][0]["protein"]["msa"] == "empty"
    assert raw["sequences"][1]["ligand"]["ccd"] == "6NT"
    # the ligand gets a free chain id, not the protein's
    assert raw["sequences"][1]["ligand"]["id"] == "B"


def test_input_yaml_uses_smiles_when_there_is_no_ccd(tmp_path):
    ligand = Ligand(id="LIG", smiles="c1ccccc1", reference_sdf="x.sdf")
    path = boltz.write_input_yaml(tmp_path / "x.yaml", {"A": "MK"}, ligand)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["sequences"][1]["ligand"]["smiles"] == "c1ccccc1"
    assert "ccd" not in raw["sequences"][1]["ligand"]


def test_sequences_are_read_back_exactly_as_submitted(tmp_path):
    """The hal->boltz map must be computed from the exact string submitted."""
    boltz.write_input_yaml(
        tmp_path / "x.yaml", {"A": "MKVLAST"}, Ligand(id="6NT", ccd="6NT")
    )
    assert boltz.read_sequences_from_yaml(tmp_path / "x.yaml") == {"A": "MKVLAST"}


def test_reading_a_yaml_with_no_protein_raises(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "sequences": []}), encoding="utf-8")
    with pytest.raises(StageError, match="no protein chains"):
        boltz.read_sequences_from_yaml(path)


# --------------------------------------------------------------------------
# the Boltz command line
# --------------------------------------------------------------------------


def test_boltz_command_carries_every_config_knob(pipeline, tmp_path):
    cmd = boltz.build_command(tmp_path / "in.yaml", tmp_path / "out", pipeline.boltz)
    joined = " ".join(cmd)
    assert cmd[0] == pipeline.boltz.binary
    assert "predict" in cmd
    assert f"--recycling_steps {pipeline.boltz.recycling_steps}" in joined
    assert f"--diffusion_samples {pipeline.boltz.diffusion_samples}" in joined
    assert f"--sampling_steps {pipeline.boltz.sampling_steps}" in joined
    assert all(isinstance(part, str) for part in cmd)


def test_no_kernels_is_passed_when_configured(pipeline, tmp_path):
    on = pipeline.boltz.model_copy(update={"no_kernels": True})
    off = pipeline.boltz.model_copy(update={"no_kernels": False})
    assert "--no_kernels" in boltz.build_command(tmp_path / "i", tmp_path / "o", on)
    assert "--no_kernels" not in boltz.build_command(tmp_path / "i", tmp_path / "o", off)


def test_potentials_flag_follows_config(pipeline, tmp_path):
    on = pipeline.boltz.model_copy(update={"use_potentials": True})
    assert "--use_potentials" in boltz.build_command(tmp_path / "i", tmp_path / "o", on)


def test_locate_output_raises_with_a_listing_when_nothing_was_written(tmp_path):
    (tmp_path / "junk.txt").write_text("x", encoding="utf-8")
    with pytest.raises(StageError, match="no model_0.cif"):
        boltz.locate_output(tmp_path, "design1")


def test_locate_output_requires_the_confidence_json(tmp_path):
    nested = tmp_path / "boltz_results_d" / "predictions" / "d"
    nested.mkdir(parents=True)
    (nested / "d_model_0.cif").write_text("data_d\n", encoding="utf-8")
    with pytest.raises(StageError, match="no confidence JSON"):
        boltz.locate_output(tmp_path, "d")


def test_locate_output_finds_nested_results(tmp_path):
    nested = tmp_path / "boltz_results_d" / "predictions" / "d"
    nested.mkdir(parents=True)
    (nested / "d_model_0.cif").write_text("data_d\n", encoding="utf-8")
    (nested / "confidence_d_model_0.json").write_text("{}", encoding="utf-8")
    out = boltz.locate_output(tmp_path, "d")
    assert out.structure.name == "d_model_0.cif"
    assert out.confidence.name == "confidence_d_model_0.json"
    assert out.pae is None


# --------------------------------------------------------------------------
# RFdiffusion
# --------------------------------------------------------------------------


def test_rfd3_partial_diffusion_sets_noise_in_design_spec(
    pipeline, tmp_path, monkeypatch
):
    """partial_t travels in the input JSON, and the contig is dropped.

    RFD3 forbids `length` during partial diffusion and ignores the contig,
    since the parent's own length is the target.
    """
    import json

    from src.stages import rfd3

    source = tmp_path / "parent.pdb"
    source.touch()
    out = tmp_path / "out"
    spec = rfd3.DesignSpec("k", source, "260-260", "A42", {"A42": "ALL"})
    commands = []
    monkeypatch.setattr(
        rfd3, "run_command", lambda cmd, **kwargs: commands.append(cmd)
    )
    monkeypatch.setattr(rfd3, "collect_outputs", lambda *args, **kwargs: [])

    rfd3.run(pipeline.rfd3, [spec], out, n_designs=4, partial=True)

    payload = json.loads((out / "rfd3_inputs.json").read_text())["k"]
    assert payload["partial_t"] == pipeline.rfd3.partial_t
    assert "contig" not in payload
    joined = " ".join(commands[0])
    assert "partial_T" not in joined
    assert "json_keys_subset=[k]" in joined
    assert "n_batches=4" in joined
    assert f"ckpt_path={pipeline.rfd3.checkpoint}" in joined


def test_rfd3_command_has_no_partial_diffusion_override(pipeline, tmp_path):
    """Partial diffusion must NOT appear on the command line.

    `rfd3 partial_T=10` fails Hydra composition outright, and `+partial_T=10`
    then fails with `RFD3InferenceConfig.__init__() got an unexpected keyword
    argument 'partial_T'`. There is no such CLI parameter. This test exists so
    nobody puts one back.
    """
    from src.stages import rfd3

    cmd = rfd3.build_command(
        pipeline.rfd3, tmp_path / "in.json", ["k"], tmp_path / "out",
        n_designs=4,
    )
    joined = " ".join(cmd)
    assert "partial_T" not in joined
    assert "partial_t" not in joined


def test_rfd3_partial_diffusion_travels_in_the_design_spec(tmp_path):
    """RFD3 reads partial diffusion as `partial_t` on the per-example spec.

    In ANGSTROMS of noise, not timesteps -- see RFD3's own field description.
    """
    import json

    from src.stages import rfd3

    spec = rfd3.DesignSpec(
        key="k",
        input_pdb=tmp_path / "parent.pdb",
        contig=None,
        unindex="A142,A179",
        select_fixed_atoms={"A142": "OD1,CG,OD2", "A179": "CD2,CE1,NE2"},
        ligand="6NT",
        partial_t=2.0,
    )
    value = spec.as_json_value()
    assert value["partial_t"] == 2.0

    written = rfd3.write_input_json([spec], tmp_path / "in.json")
    assert json.loads(written.read_text())["k"]["partial_t"] == 2.0


def test_rfd3_backbone_index_is_the_batch_not_the_model(tmp_path):
    """The batch index is the digit before "_model_", not the last digit.

    Scanning the name backwards finds model_0's "0" first, which made every
    backbone report index 0. That index feeds the LigandMPNN output directory
    (`mpnn_{index}`), so all five backbones shared one directory -- it only
    worked because the FASTA is looked up by stem.
    """
    from src.stages import rfd3

    out = tmp_path / "rfd3"
    out.mkdir()
    for i in range(3):
        stem = f"rfd3_inputs_01_his_asp_01_{i}_model_0"
        (out / f"{stem}.cif.gz").write_bytes(b"")
        (out / f"{stem}.json").write_text("{}", encoding="utf-8")

    outputs = rfd3.collect_outputs(out, ["01_his_asp_01"])
    assert sorted(o.index for o in outputs) == [0, 1, 2]


def test_rfd3_fresh_scaffolding_omits_partial_t(tmp_path):
    from src.stages import rfd3

    spec = rfd3.DesignSpec(
        key="k",
        input_pdb=tmp_path / "seed.pdb",
        contig="260-260",
        unindex="A229,A291",
        select_fixed_atoms={"A229": "ALL"},
        ligand="6NT",
    )
    assert "partial_t" not in spec.as_json_value()



def test_wiggle_metadata_maps_back_to_original_seed_ids(tmp_path):
    from src.tracking import seed_to_hal_map_rfd3

    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=229),
                            ResidueRef(chain="A", resnum=291)])
    metadata = tmp_path / "output.json"
    metadata.write_text('{"diffused_index_map": {"A42": "A44", "A77": "A78"}}')
    parent = {("A", 229): ("A", 42), ("A", 291): ("A", 77)}
    assert seed_to_hal_map_rfd3(metadata, motif, seed_to_input=parent) == {
        ("A", 229): ("A", 44), ("A", 291): ("A", 78),
    }


def test_fixed_residues_are_passed_in_backbone_numbering():
    """Passing seed numbers to a renumbered backbone would fix the wrong
    residues -- the silent failure the chain of maps exists to prevent."""
    motif = MotifCfg(
        fixed=[ResidueRef(chain="A", resnum=229), ResidueRef(chain="A", resnum=291)]
    )
    seed_to_hal = {("A", 229): ("A", 12), ("A", 291): ("A", 74)}
    assert ligandmpnn.fixed_residues_argument(motif, seed_to_hal) == "A12 A74"


def test_an_unmapped_motif_residue_raises():
    motif = MotifCfg(fixed=[ResidueRef(chain="A", resnum=229)])
    with pytest.raises(StageError, match="not in the seed->backbone map"):
        ligandmpnn.fixed_residues_argument(motif, {})


def test_ligandmpnn_command_carries_the_knobs(pipeline, tmp_path):
    cmd = ligandmpnn.build_command(
        pipeline.ligandmpnn, tmp_path / "bb.pdb", tmp_path / "out", "A12 A74", 4
    )
    joined = " ".join(cmd)
    assert f"--temperature {pipeline.ligandmpnn.temperature}" in joined
    assert "--fixed_residues A12 A74" in joined
    assert f"--omit_AA {pipeline.ligandmpnn.omit_AA}" in joined
    assert "--number_of_batches 4" in joined


def test_fasta_parsing(tmp_path):
    path = tmp_path / "x.fa"
    path.write_text(
        ">input\nMKVL\n>design, T=0.1, sample=1\nMKVM\n", encoding="utf-8"
    )
    parsed = ligandmpnn.read_fasta(path)
    assert parsed["input"] == "MKVL"
    assert parsed["design, T=0.1, sample=1"] == "MKVM"


def test_designed_sequences_drops_the_input_record(tmp_path):
    path = tmp_path / "x.fa"
    path.write_text(
        ">input\nMKVL\n>design, T=0.1, sample=1\nMKVM\n", encoding="utf-8"
    )
    output = ligandmpnn.LigandMPNNOutput(
        fasta=path,
        sequences=ligandmpnn.read_fasta(path),
        packed_pdbs=[],
        out_dir=tmp_path,
    )
    designs = ligandmpnn.designed_sequences(output)
    assert [seq for _, seq in designs] == ["MKVM"]


def test_a_fasta_with_only_the_input_raises(tmp_path):
    path = tmp_path / "x.fa"
    path.write_text(">input\nMKVL\n", encoding="utf-8")
    output = ligandmpnn.LigandMPNNOutput(
        fasta=path, sequences=ligandmpnn.read_fasta(path),
        packed_pdbs=[], out_dir=tmp_path,
    )
    with pytest.raises(StageError, match="no.*designs were produced"):
        ligandmpnn.designed_sequences(output)


# --------------------------------------------------------------------------
# protonation
# --------------------------------------------------------------------------


def test_protonation_is_only_needed_by_the_split_rule():
    assert protonate.needed("split_by_donor_type")
    assert not protonate.needed("acceptor_side_always")


def test_engine_none_returns_the_input_unchanged(pipeline, tmp_path):
    from src.config import ProtonationCfg

    source = tmp_path / "in.cif"
    source.write_text("data_x\n", encoding="utf-8")
    path, result = protonate.run(
        ProtonationCfg(engine="none"), source, tmp_path / "out.pdb"
    )
    assert path == source
    assert result is None


def test_reduce_refuses_rather_than_truncating(tmp_path):
    from src.config import ProtonationCfg

    source = tmp_path / "in.cif"
    source.write_text("data_x\n", encoding="utf-8")
    with pytest.raises(StageError, match="not implemented"):
        protonate.run(ProtonationCfg(engine="reduce"), source, tmp_path / "out.pdb")


def test_missing_input_raises(tmp_path):
    from src.config import ProtonationCfg

    with pytest.raises(StageError, match="input not found"):
        protonate.run(
            ProtonationCfg(engine="none"), tmp_path / "nope.cif", tmp_path / "o.pdb"
        )


# --------------------------------------------------------------------------
# run_command
# --------------------------------------------------------------------------


def test_run_command_succeeds_and_times():
    result = run_command(
        ["python", "-c", "print('hello')"], stage="test"
    )
    assert result.returncode == 0
    assert "hello" in result.stdout_tail
    assert result.wall_seconds >= 0


def test_run_command_raises_with_the_stderr_tail():
    with pytest.raises(StageError, match="exited 1"):
        run_command(
            ["python", "-c", "import sys; sys.stderr.write('BOOM'); sys.exit(1)"],
            stage="test",
        )


def test_run_command_attaches_the_error_output():
    with pytest.raises(StageError) as exc:
        run_command(
            ["python", "-c", "import sys; sys.stderr.write('BOOM'); sys.exit(2)"],
            stage="test",
        )
    assert "BOOM" in str(exc.value)


def test_run_command_names_a_missing_executable():
    with pytest.raises(StageError, match="executable not found"):
        run_command(["definitely-not-a-real-binary-xyz"], stage="test")


def test_ligandmpnn_defaults_to_the_running_interpreter(pipeline, tmp_path):
    """A bare "python" resolves to whatever is on PATH, which on a cluster is
    usually a module-system interpreter with no torch. That failure cost a GPU
    allocation, so the default is now the interpreter running the pipeline."""
    import sys

    cfg = pipeline.ligandmpnn.model_copy(update={"python": None})
    cmd = ligandmpnn.build_command(
        cfg, tmp_path / "bb.pdb", tmp_path / "out", "A12", 4
    )
    assert cmd[0] == sys.executable


def test_ligandmpnn_interpreter_can_be_set_explicitly(pipeline, tmp_path):
    cfg = pipeline.ligandmpnn.model_copy(update={"python": "/envs/x/bin/python"})
    cmd = ligandmpnn.build_command(
        cfg, tmp_path / "bb.pdb", tmp_path / "out", "A12", 4
    )
    assert cmd[0] == "/envs/x/bin/python"


def test_rfd3_normalises_an_oddly_suffixed_seed(tmp_path, pipeline):
    """Every external tool dispatches on the file extension, not content. A
    real seed is named 01_his_asp_.pdb.pre-boltz, and ProDy duly parses it as
    mmCIF and reports "no atoms". The fix writes a properly-suffixed copy into
    the RUN directory, never renaming the file in data/."""
    from src.stages.rfd3 import _normalised_input
    from tests.paths import REAL_SEED

    assert REAL_SEED.exists(), REAL_SEED
    assert not REAL_SEED.name.endswith(".pdb"), "this test needs an odd suffix"

    out = _normalised_input(REAL_SEED, tmp_path)
    assert out.suffix == ".pdb"
    assert out.parent == tmp_path, "the copy goes in the run dir, not data/"
    assert REAL_SEED.exists(), "the original must be left alone"

    # and it round-trips: same motif residues, same ligand
    from src.tracking import read_structure

    before = read_structure(REAL_SEED, "6NT")
    after = read_structure(out, "6NT")
    assert set(after.residues) == set(before.residues)
    assert set(after.ligand) == set(before.ligand)


def test_rfd3_leaves_a_normal_pdb_alone(tmp_path, pipeline):
    from src.stages.rfd3 import _normalised_input

    normal = tmp_path / "already.pdb"
    normal.write_text("ATOM\n", encoding="utf-8")
    assert _normalised_input(normal, tmp_path / "run") == normal

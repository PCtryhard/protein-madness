"""Select the top 60 geometry-ranked designs and extend each on Euler."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess

import yaml

from scripts.build_targets import acceptor_ideal
from src.cli import main as pipeline_cli
from src.config import load
from src.runner import GpuPool, Runner, RunPaths
from src.scoring import score_design
from src.store import DesignRow, DesignStore
from src.tracking import Provenance, assert_motif_unchanged, hal_to_boltz_map
from src.tracking import read_structure, seed_to_hal_map_rfd3

MANIFEST = Path("data/rescored/manifest.csv")
OUTPUT = Path("data/final_run_60")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select_parents():
    rows = sorted(read_csv(MANIFEST), key=lambda row: int(row["rerank_rank"]))[:60]
    if [int(row["rerank_rank"]) for row in rows] != list(range(1, 61)):
        raise ValueError("the manifest must have one entry for each rank 1–60")
    if len({row["design_id"] for row in rows}) != 60:
        raise ValueError("the top 60 entries must have distinct design IDs")
    return rows


def source_files(entry):
    variant = Path("data") / entry["variant"]
    run = entry["run"]
    designs = {row["design_id"]: row
               for row in read_csv(variant / "scores" / run / "designs.csv")}
    design = designs[entry["design_id"]]
    archive = variant / "rfd3" / run
    backbone = archived_backbone(archive, design)
    structure = Path(entry["source_cif"])
    confidence = structure.with_name(f"confidence_{structure.stem}.json")
    target = variant / "inputs" / run / "target.yaml"
    for file in (backbone, backbone.with_suffix(".json"), structure, confidence, target):
        if not file.is_file():
            raise FileNotFoundError(file)
    return design, backbone, structure, confidence, target, designs, archive


def archived_backbone(archive, design):
    return archive / design["backbone_path"].split("/structures/", 1)[1]


def trace_motif(designs, archive, design, motif):
    branch = design["branch"]
    if branch == "mutate":
        return trace_motif(designs, archive, designs[design["parent_id"]], motif)
    backbone = archived_backbone(archive, design)
    seed_to_input = None
    if branch == "wiggle":
        # Single-chain archived backbones and Boltz predictions use contiguous
        # numbering, but explicitly map by ordinal for historical wiggle inputs.
        parent = designs[design["parent_id"]]
        parent_map = trace_motif(designs, archive, parent, motif)
        metadata = json.loads(backbone.with_suffix(".json").read_text())
        raw = metadata["diffused_index_map"]
        if all(f"{ch}{num}" in raw for ch, num in parent_map.values()):
            seed_to_input = parent_map
        else:
            parent_structure = read_structure(archived_backbone(archive, parent), "6NT")
            renumbered = hal_to_boltz_map(
                {"A": parent["sequence"]}, {"A": parent_structure.residue_numbers("A")}
            )
            seed_to_input = {key: renumbered[value] for key, value in parent_map.items()}
    return seed_to_hal_map_rfd3(backbone.with_suffix(".json"), motif, seed_to_input=seed_to_input)


def setup_config(output, source_target, backbone, use_euler):
    pipeline = yaml.safe_load(Path("config/pipeline.yaml").read_text())
    pipeline["run"].update(max_cycles=4, gpus=[0])
    if use_euler:
        rfd3 = os.environ["PM_RFD3_BIN"]
        help_text = subprocess.run([rfd3, "--help"], check=True, text=True,
                                   capture_output=True).stdout
        if any(line.split()[:1] == ["design"] for line in help_text.splitlines()):
            wrapper = output / "rfd3-design.sh"
            wrapper.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(rfd3)} design \"$@\"\n")
            wrapper.chmod(0o755)
            rfd3 = str(wrapper.resolve())
        pipeline["rfd3"].update(binary=rfd3, checkpoint=os.environ["PM_RFD3_CKPT"])
        pipeline["ligandmpnn"].update(
            python=os.sys.executable, run_script=os.environ["PM_MPNN_SCRIPT"],
            checkpoint=os.environ["PM_MPNN_CKPT"],
        )
        pipeline["boltz"].update(binary=os.environ["PM_BOLTZ_BIN"],
                                 cache_dir=os.environ["PM_BOLTZ_CACHE"], assets_dir=None)
    target = yaml.safe_load(source_target.read_text())
    if len(target["seeds"]) != 1:
        raise ValueError(f"expected one source scaffold in {source_target}")
    target["seeds"][0]["pdb"] = str(backbone.resolve())
    for bond in target["hbonds"]:
        atom = bond["acceptor"]["atom"]
        key = f"lig_{atom.lower()}" if bond["acceptor"]["site"] == "ligand" else ""
        geometry = acceptor_ideal(key, atom)
        if geometry is None:
            raise ValueError(f"cannot resolve angle ideal for {bond['id']}")
        bond["acceptor_ideal_angle"] = geometry[0]
    for name, config in (("pipeline.yaml", pipeline), ("target.yaml", target)):
        (output / name).write_text(yaml.safe_dump(config, sort_keys=False))
    return load(output / "pipeline.yaml", output / "target.yaml")


def import_parent(entry, cfg, design, backbone, structure, confidence, designs, archive):
    motif = cfg.target.motif
    ccd = cfg.target.ligand.ccd
    seed_to_hal = trace_motif(designs, archive, design, motif)
    scaffold = read_structure(backbone, ccd)
    prediction = read_structure(structure, ccd)
    sequence = design["sequence"]
    if prediction.sequence("A") != sequence:
        raise ValueError(f"{structure}: prediction sequence differs from its design row")
    provenance = Provenance(
        design_id=entry["design_id"], scaffold_id=f"{cfg.target.name}.{cfg.target.seeds[0].id}",
        seed_to_hal=seed_to_hal,
        hal_to_boltz=hal_to_boltz_map({"A": sequence}, {"A": scaffold.residue_numbers("A")}),
        ligand_atom_map={atom: atom for atom in prediction.ligand},
        motif_expected={key: scaffold.residues[seed_to_hal[key]] for key in motif.as_keys()},
    )
    assert_motif_unchanged(
        prediction,
        {provenance.seed_residue_in_boltz(*key): aa for key, aa in provenance.motif_expected.items()},
        stage="imported Boltz-2 prediction",
    )
    scored = score_design(cfg, prediction, provenance, confidence)
    score = scored.score
    if score.gated:
        raise ValueError(f"manifest rank {entry['rerank_rank']} fails current gates: {score.gated_bonds}")
    return DesignRow(
        design_id=entry["design_id"], scaffold_id=provenance.scaffold_id, cycle=0,
        branch="seed", sequence=sequence, seq_hash=f"imported:{entry['design_id']}",
        status="scored", provenance=provenance.to_json(), score=score.total,
        term_confidence=score.confidence, term_distance=score.distance,
        term_catalytic=score.catalytic, term_non_catalytic=score.non_catalytic,
        term_angle=score.angle, hbond_raw=json.dumps(scored.hbond_raw()),
        extra_metrics=json.dumps(scored.extra_metrics),
        structure_path=str(structure.resolve()), backbone_path=str(backbone.resolve()),
    )


def write_selection(entries):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "selected_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="validate all parents on CPU")
    mode.add_argument("--rank", type=int, choices=range(1, 61), metavar="1–60")
    args = parser.parse_args()

    if args.prepare:
        from tempfile import TemporaryDirectory

        entries = select_parents()
        with TemporaryDirectory() as directory:
            for entry in entries:
                design, backbone, structure, confidence, target, designs, archive = source_files(entry)
                cfg = setup_config(Path(directory), target, backbone, use_euler=False)
                import_parent(entry, cfg, design, backbone, structure, confidence, designs, archive)
        write_selection(entries)
        print(f"Validated {len(entries)} parents: {OUTPUT / 'selected_manifest.csv'}")
        return

    entries = read_csv(OUTPUT / "selected_manifest.csv")
    entry, = [row for row in entries if int(row["rerank_rank"]) == args.rank]
    design, backbone, structure, confidence, target, designs, archive = source_files(entry)
    output = OUTPUT / f"{args.rank:02d}-{entry['variant']}-{entry['hex_id']}"
    output.mkdir(exist_ok=False)
    cfg = setup_config(output, target, backbone, use_euler=True)
    row = import_parent(entry, cfg, design, backbone, structure, confidence, designs, archive)
    paths = RunPaths.create(output.parent, output.name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    with DesignStore(paths.db) as store, GpuPool(cfg.pipeline.run.gpus) as pool:
        runner = Runner(cfg, paths, store)
        runner._record(row)
        search = runner.run_scaffold(cfg.target.seeds[0], pool, initial=[row.to_candidate()])
        print(f"Best {search.best_design_id}: {search.best_score:.4f}; {search.stop_reason}")
    pipeline_cli(["report", "--run", str(output), "--write"])


if __name__ == "__main__":
    main()

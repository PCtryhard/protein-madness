"""ProteinMPNN: fixed-backbone sequence design (no ligand).

    python -m src.generate.proteinmpnn --in data/toy/ref.pdb --out results/run/proteinmpnn \
        --config configs/pipeline.yaml --target configs/target.yaml

Use LigandMPNN (``src/generate/ligandmpnn.py``) instead where a substrate is
defined: for enzyme design, conditioning on the ligand in the pocket is the
whole point. This module is the fallback when the challenge has no usable
substrate definition, or for comparison.

Shells out to the official ProteinMPNN repository's ``protein_mpnn_run.py``,
cloned and fetched by ``scripts/fetch_weights.sh proteinmpnn``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import cli
from ..io.fasta import read_fasta
from ..io.structure import normalise, read_structure, write_structure
from ..io.target import load_reference
from . import common

log = logging.getLogger("pipeline.generate.proteinmpnn")
TOOL = "generate.proteinmpnn"
VERSION = "0.1.0"


def find_repo(explicit: str | None = None) -> Path:
    for cand in [explicit, os.environ.get("PROTEINMPNN_DIR"),
                 str(Path.home() / "ProteinMPNN")]:
        if not cand:
            continue
        p = Path(os.path.expandvars(cand))
        if (p / "protein_mpnn_run.py").exists():
            return p
    raise common.ModelUnavailable(
        "ProteinMPNN checkout not found. Run `bash scripts/fetch_weights.sh proteinmpnn`, "
        "then set $PROTEINMPNN_DIR or pass --repo."
    )


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--repo", default=None, help="ProteinMPNN checkout (default: $PROTEINMPNN_DIR)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--fix-catalytic", dest="fix_catalytic", action="store_true", default=True)
    ap.add_argument("--no-fix-catalytic", dest="fix_catalytic", action="store_false")
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    device = cli.resolve_device(args, tcfg)
    target = cli.require_target(args)
    ref = load_reference(target)
    n_per = args.n or int(tcfg.get("n_per_backbone", 8))
    temperature = args.temperature if args.temperature is not None \
        else float(tcfg.get("temperature", 0.1))

    with cli.stage(TOOL, args, tcfg, VERSION,
                   hash_extra={"target": target.hash, "n": n_per, "temp": temperature,
                               "fix_catalytic": args.fix_catalytic}) as stage:
        if stage is None:
            return 0
        out_dir = stage.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        repo = find_repo(args.repo)
        work = Path(tempfile.mkdtemp(prefix="proteinmpnn_"))

        backbone = out_dir / "input_backbone.pdb"
        atoms, _ = normalise(read_structure(target.reference_structure), chain=target.chain)
        write_structure(atoms, backbone)
        chain = ref.numbering.entries[0]["chain"]

        cmd = [sys.executable, str(repo / "protein_mpnn_run.py"),
               "--pdb_path", str(backbone),
               "--pdb_path_chains", chain,
               "--out_folder", str(work),
               "--num_seq_per_target", str(n_per),
               "--sampling_temp", str(temperature),
               "--seed", str(int(cfg.get("defaults", {}).get("seed", 0))),
               "--batch_size", "1"]
        if device.startswith("cpu"):
            cmd.append("--device")
            cmd.append("cpu")
        if args.fix_catalytic and ref.catalytic_positions:
            fixed = out_dir / "fixed_positions.jsonl"
            import json
            stem = backbone.stem
            fixed.write_text(json.dumps({stem: {chain: ref.catalytic_positions}}) + "\n",
                             encoding="utf-8")
            cmd += ["--fixed_positions_jsonl", str(fixed)]
            log.info("fixing catalytic residues: %s", ref.catalytic_positions)

        log.info("running ProteinMPNN: %d designs", n_per)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"ProteinMPNN failed (exit {proc.returncode}):\n"
                             f"{proc.stderr[-2000:]}")

        produced = sorted((work / "seqs").glob("*.fa"))
        if not produced:
            raise SystemExit(f"ProteinMPNN produced no sequences in {work / 'seqs'}")

        collector = common.DesignCollector(TOOL, stage.cfg_hash, parent_id="reference",
                                           parent_sequence=ref.sequence)
        allowed = set(ref.design_positions)
        for fa in produced:
            records = read_fasta(fa)
            # ProteinMPNN's first record is the input sequence, not a design.
            for rec in records[1:] if len(records) > 1 else records:
                seq = rec.seq
                if len(seq) == len(ref.sequence) and allowed:
                    seq = "".join(c if (i + 1) in allowed else ref.sequence[i]
                                  for i, c in enumerate(seq))
                collector.add(seq, notes=f"src={fa.name}")
        if not collector.rows:
            raise SystemExit("ProteinMPNN produced sequences but none survived validation")
        collector.write(out_dir)
        stage.write_csv(collector.rows)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

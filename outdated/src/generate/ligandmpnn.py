"""LigandMPNN: ligand-aware fixed-backbone sequence design.

    python -m src.generate.ligandmpnn --in data/toy/ref.pdb --out results/run/ligandmpnn \
        --config configs/pipeline.yaml --target configs/target.yaml

For enzyme design this matters more than ProteinMPNN: it conditions on the
substrate in the pocket, so the residues it picks around the active site are
chosen in the presence of the thing they have to act on.

Shells out to the official LigandMPNN repository's ``run.py``, which is cloned
and whose weights are fetched by ``scripts/fetch_weights.sh ligandmpnn``. The
path is taken from ``$LIGANDMPNN_DIR`` or ``--repo``; this module does not
vendor a copy of the model.

Catalytic residues from the target config are passed as FIXED positions, so a
design that destroys the active site is never generated in the first place.
"""
from __future__ import annotations

import json
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

log = logging.getLogger("pipeline.generate.ligandmpnn")
TOOL = "generate.ligandmpnn"
VERSION = "0.1.0"


def find_repo(explicit: str | None = None) -> Path:
    """Locate the LigandMPNN checkout, or explain exactly how to get it."""
    for cand in [explicit, os.environ.get("LIGANDMPNN_DIR"),
                 str(Path.home() / "LigandMPNN"), "/cluster/scratch/$USER/LigandMPNN"]:
        if not cand:
            continue
        p = Path(os.path.expandvars(cand))
        if (p / "run.py").exists():
            return p
    raise common.ModelUnavailable(
        "LigandMPNN checkout not found. Run `bash scripts/fetch_weights.sh ligandmpnn` "
        "(clones the repo and downloads weights), then set $LIGANDMPNN_DIR or pass --repo. "
        "Weights must NOT live in this repo; on Euler put them in $SCRATCH."
    )


def prepare_structure(target, out_dir: Path) -> tuple[Path, list[int]]:
    """Write the normalised backbone LigandMPNN will design on.

    Returns the path and the catalytic positions in NORMALISED numbering, which
    is the numbering LigandMPNN sees because it is reading this file.
    """
    ref = load_reference(target)
    path = out_dir / "input_backbone.pdb"
    # keep_hetero: the ligand in the pocket is the entire point of this tool
    atoms, _ = normalise(read_structure(target.reference_structure),
                         chain=target.chain, keep_hetero=True)
    write_structure(atoms, path)
    return path, ref.catalytic_positions


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--repo", default=None, help="LigandMPNN checkout (default: $LIGANDMPNN_DIR)")
    ap.add_argument("--n", type=int, default=None, help="designs per backbone")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--fix-catalytic", dest="fix_catalytic", action="store_true", default=True,
                    help="keep catalytic residues fixed (default)")
    ap.add_argument("--no-fix-catalytic", dest="fix_catalytic", action="store_false")
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    device = cli.resolve_device(args, tcfg)
    target = cli.require_target(args)
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

        backbone, catalytic = prepare_structure(target, out_dir)
        ref = load_reference(target)
        work = Path(tempfile.mkdtemp(prefix="ligandmpnn_"))

        cmd = [sys.executable, str(repo / "run.py"),
               "--model_type", "ligand_mpnn",
               "--checkpoint_ligand_mpnn",
               str(repo / "model_params" / f"{tcfg.get('checkpoint', 'ligandmpnn_v_32_010_25')}.pt"),
               "--pdb_path", str(backbone),
               "--out_folder", str(work),
               "--batch_size", "1",
               "--number_of_batches", str(n_per),
               "--temperature", str(temperature),
               "--seed", str(int(cfg.get("defaults", {}).get("seed", 0))),
               "--device", "cpu" if device.startswith("cpu") else device.replace("cuda:", "")]
        if args.fix_catalytic and catalytic:
            chain = ref.numbering.entries[0]["chain"]
            cmd += ["--fixed_residues", " ".join(f"{chain}{p}" for p in catalytic)]
            log.info("fixing catalytic residues: %s", catalytic)
        if ref.design_positions:
            log.info("design region: %d positions (LigandMPNN redesigns everything not fixed; "
                     "positions outside the region are reverted below)", len(ref.design_positions))

        log.info("running LigandMPNN: %d designs from %s", n_per, backbone.name)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"LigandMPNN failed (exit {proc.returncode}):\n"
                             f"{proc.stderr[-2000:]}")

        produced = sorted((work / "seqs").glob("*.fa")) + sorted((work / "seqs").glob("*.fasta"))
        if not produced:
            raise SystemExit(f"LigandMPNN produced no sequences in {work / 'seqs'}; "
                             f"stdout:\n{proc.stdout[-1500:]}")

        collector = common.DesignCollector(TOOL, stage.cfg_hash, parent_id="reference",
                                           parent_sequence=ref.sequence)
        # Keep positions outside the design region at the parent residue: the
        # target config says which positions we are allowed to change, and a
        # design that quietly changed others would not be the thing we claim.
        allowed = set(ref.design_positions)
        for fa in produced:
            for rec in read_fasta(fa):
                seq = rec.seq
                if len(seq) == len(ref.sequence) and allowed:
                    seq = "".join(c if (i + 1) in allowed else ref.sequence[i]
                                  for i, c in enumerate(seq))
                collector.add(seq, notes=f"src={fa.name}")
        if not collector.rows:
            raise SystemExit("LigandMPNN produced sequences but none survived validation")
        collector.write(out_dir)
        stage.write_csv(collector.rows)
        (out_dir / "ligandmpnn.cmd.json").write_text(json.dumps({"cmd": cmd}, indent=1),
                                                     encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""RFdiffusion: backbone generation and motif scaffolding.

    python -m src.generate.rfdiffusion --out results/run/rfdiff \
        --config configs/pipeline.yaml --target configs/target.yaml \
        --contigs "10-40/A35-35/5-20/A52-52/10-40"

"Scaffold this catalytic motif" is the most likely shape of an enzyme-design
challenge, so the motif-holding path is the one that matters here. ``--contigs``
and motif-holding are exposed directly rather than buried in config.

Shells out to the official RFdiffusion repository, cloned and fetched by
``scripts/fetch_weights.sh rfdiffusion``.

Output: backbones only. RFdiffusion designs shapes, not sequences — feed its
PDBs to LigandMPNN to get sequences, then fold and score those. The
``designs.fasta`` this stage writes therefore contains no sequences; its CSV
rows describe backbones and carry ``structure_path``.

``contigs`` has no default and is not guessed: it encodes which residues of
which chain are held fixed, and a wrong contig string silently scaffolds the
wrong motif. It comes from the challenge brief Saturday, via the target config
or ``--contigs``.
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
from ..io.structure import normalise, read_structure, write_structure
from ..io.target import load_reference
from . import common

log = logging.getLogger("pipeline.generate.rfdiffusion")
TOOL = "generate.rfdiffusion"
VERSION = "0.1.0"


def find_repo(explicit: str | None = None) -> Path:
    for cand in [explicit, os.environ.get("RFDIFFUSION_DIR"),
                 str(Path.home() / "RFdiffusion")]:
        if not cand:
            continue
        p = Path(os.path.expandvars(cand))
        if (p / "scripts" / "run_inference.py").exists():
            return p
    raise common.ModelUnavailable(
        "RFdiffusion checkout not found. Run `bash scripts/fetch_weights.sh rfdiffusion`, "
        "then set $RFDIFFUSION_DIR or pass --repo."
    )


def motif_contigs(reference, flank: str = "10-40", linker: str = "5-20") -> str:
    """Suggest a contig string that holds the catalytic residues of the reference.

    This is a STARTING POINT to paste into the target config after reading it,
    not something to run blind: it holds each catalytic residue as a single
    fixed position with flexible linkers between, in the reference's own
    chain/numbering. Whether that is the right scaffold problem is a judgement
    the challenge brief decides.
    """
    chain = reference.numbering.entries[0]["chain"]
    parts = [flank]
    for i, c in enumerate(reference.catalytic):
        orig = reference.numbering.to_orig(c["pos"])
        parts.append(f"{chain}{orig.resi}-{orig.resi}")
        parts.append(linker if i < len(reference.catalytic) - 1 else flank)
    return "/".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--repo", default=None, help="RFdiffusion checkout (default: $RFDIFFUSION_DIR)")
    ap.add_argument("--contigs", default=None,
                    help="contig string, e.g. '10-40/A35-35/5-20/A52-52/10-40'")
    ap.add_argument("--n", type=int, default=None, help="number of backbones")
    ap.add_argument("--suggest-contigs", action="store_true",
                    help="print a contig string that holds the catalytic residues, then exit")
    args = ap.parse_args(argv)
    # --in is not meaningful here: the backbone comes from the target config.
    if args.inp in (None, ""):
        args.inp = "unused"
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    device = cli.resolve_device(args, tcfg)
    target = cli.require_target(args)
    ref = load_reference(target)

    if args.suggest_contigs:
        print(motif_contigs(ref))
        print("# starting point only: check it against the challenge brief before running")
        return 0

    contigs = args.contigs or tcfg.get("contigs")
    if not contigs:
        raise SystemExit(
            "rfdiffusion needs --contigs (or tools.generate.rfdiffusion.contigs in "
            "configs/pipeline.yaml). No default is guessed: a wrong contig string scaffolds "
            "the wrong motif and nothing downstream would notice.\n"
            "Run with --suggest-contigs for a starting point built from the catalytic residues."
        )
    n_designs = args.n or int(tcfg.get("n_designs", 8))

    with cli.stage(TOOL, args, tcfg, VERSION,
                   hash_extra={"target": target.hash, "contigs": contigs, "n": n_designs}) as stage:
        if stage is None:
            return 0
        out_dir = stage.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        repo = find_repo(args.repo)
        work = Path(tempfile.mkdtemp(prefix="rfdiffusion_"))

        # RFdiffusion reads the motif from the reference in ITS OWN numbering,
        # which is why we write the un-renumbered reference here.
        motif_pdb = out_dir / "motif_input.pdb"
        shutil.copyfile(target.reference_structure, motif_pdb)

        cmd = [sys.executable, str(repo / "scripts" / "run_inference.py"),
               f"inference.input_pdb={motif_pdb}",
               f"inference.output_prefix={work / 'design'}",
               f"inference.num_designs={n_designs}",
               f"contigmap.contigs=[{contigs}]",
               f"diffuser.T={int(tcfg.get('diffuser_T', 50))}"]
        if device.startswith("cpu"):
            log.warning("RFdiffusion on CPU is impractically slow; use a GPU backend")
        log.info("running RFdiffusion: %d backbones, contigs=%s", n_designs, contigs)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        if proc.returncode != 0:
            raise SystemExit(f"RFdiffusion failed (exit {proc.returncode}):\n"
                             f"{proc.stderr[-2000:]}")

        produced = sorted(work.glob("design_*.pdb"))
        if not produced:
            raise SystemExit(f"RFdiffusion produced no backbones in {work}")

        rows = []
        for i, pdb in enumerate(produced):
            did = f"bb_{stage.cfg_hash[:6]}_{i:04d}"
            dest = out_dir / f"{did}.pdb"
            atoms, _ = normalise(read_structure(pdb))
            write_structure(atoms, dest)
            rows.append({
                "design_id": did, "method": TOOL, "parent_id": target.name,
                "structure_path": str(dest), "contigs": contigs,
                "n_residues": len(set(atoms.res_id.tolist())),
                "notes": "backbone only; run LigandMPNN on it to get a sequence",
            })
        stage.write_csv(rows)
        log.info("%d backbones -> %s. Next: python -m src.generate.ligandmpnn on these.",
                 len(rows), out_dir)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

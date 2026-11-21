"""Boltz-2: co-folding with the substrate. The ligand-aware escape hatch.

    python -m src.fold.boltz --in designs.fasta --out results/run/fold_boltz \
        --config configs/pipeline.yaml --target configs/target.yaml --device cuda

Why this is NOT the default (DECISIONS.md, 2026-11-16): Boltz-2 generally wants
more than the 11 GB on Euler's RTX 2080Ti, so it runs on Modal's A100s. It
earns its place because it places the substrate in the pocket, which is the
enzyme-relevant signal ESMFold cannot give us.

Shells out to the real ``boltz`` CLI. Everything below follows the documented
behaviour of the current release:
  * ``boltz predict <input.yaml> --out_dir ... --recycling_steps N
    --diffusion_samples N --accelerator gpu|cpu --output_format pdb``
  * YAML: ``version: 1`` + ``sequences:`` with ``protein`` entries (id,
    sequence, msa) and ``ligand`` entries (id, and either smiles or ccd, never
    both). ``msa: empty`` selects single-sequence mode.
  * outputs land in ``<out_dir>/predictions/<stem>/`` as
    ``<stem>_model_0.pdb`` (or .cif), ``plddt_<stem>_model_0.npz`` and
    ``pae_<stem>_model_0.npz``; confidence scores are on a 0-1 scale.

Two things the Boltz docs do NOT state, so this module does not assume them:
the array KEYS inside those .npz files (we read them defensively and raise if
the contents are not what we expect), and whether pLDDT is written into the
B-factor column (we write our own .plddt.csv from the npz regardless).
See TODO.md.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from .. import cli
from ..io.target import load_target
from . import common

log = logging.getLogger("pipeline.fold.boltz")
TOOL = "fold.boltz"
VERSION = "0.1.0"
BACKEND = "boltz"


def boltz_binary() -> str:
    exe = shutil.which("boltz")
    if exe is None:
        raise common.ModelUnavailable(
            "boltz not on PATH. Install it (`pip install boltz==2.2.0`) and fetch weights with "
            "`bash scripts/fetch_weights.sh boltz`. On Euler this generally exceeds 11 GB: "
            "use the modal backend."
        )
    return exe


def build_input_yaml(sequence: str, chain: str = "A", msa: str = "empty",
                     ligand_smiles: str | None = None, ligand_ccd: str | None = None,
                     ligand_chain: str = "B") -> dict:
    """The documented Boltz YAML schema. ``msa='empty'`` is single-sequence mode.

    Single-sequence mode is the default here because Euler compute nodes have
    no outbound internet and so cannot reach the MSA server. The Boltz docs
    note single-sequence mode reduces accuracy; when an MSA is available pass
    its .a3m path as ``msa``.
    """
    if ligand_smiles and ligand_ccd:
        raise ValueError("Boltz accepts either smiles or ccd for a ligand, not both")
    entries: list[dict] = [{"protein": {"id": chain, "sequence": sequence, "msa": msa}}]
    if ligand_smiles:
        entries.append({"ligand": {"id": ligand_chain, "smiles": ligand_smiles}})
    elif ligand_ccd:
        entries.append({"ligand": {"id": ligand_chain, "ccd": ligand_ccd}})
    return {"version": 1, "sequences": entries}


def _read_npz_matrix(path: Path, expected_ndim: int, what: str) -> np.ndarray:
    """Read a Boltz .npz whose array key is not documented.

    Takes the single array when there is exactly one, otherwise a key that
    matches the quantity by name. Raises rather than guessing between several
    unnamed arrays: a silently mis-read PAE matrix would poison every
    active-site confidence number we report.
    """
    with np.load(path) as z:
        keys = list(z.keys())
        if len(keys) == 1:
            arr = z[keys[0]]
        else:
            named = [k for k in keys if what in k.lower()]
            if len(named) != 1:
                raise ValueError(
                    f"{path}: expected one array for {what}, found keys {keys}. The Boltz docs "
                    "do not specify these keys; inspect the file and update src/fold/boltz.py "
                    "rather than guessing."
                )
            arr = z[named[0]]
    arr = np.asarray(arr, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != expected_ndim:
        raise ValueError(f"{path}: {what} has shape {arr.shape}, expected {expected_ndim}-D")
    return arr


def collect_outputs(pred_dir: Path, stem: str) -> tuple[Path, np.ndarray, np.ndarray | None]:
    """Find the structure, per-token pLDDT and PAE Boltz wrote for one input."""
    struct = next((p for p in [pred_dir / f"{stem}_model_0.pdb",
                               pred_dir / f"{stem}_model_0.cif"] if p.exists()), None)
    if struct is None:
        found = sorted(p.name for p in pred_dir.glob("*"))[:10] if pred_dir.exists() else []
        raise FileNotFoundError(
            f"boltz produced no structure for {stem} in {pred_dir} (found: {found})")

    plddt_npz = pred_dir / f"plddt_{stem}_model_0.npz"
    if not plddt_npz.exists():
        raise FileNotFoundError(f"boltz produced no pLDDT file: {plddt_npz}")
    plddt = _read_npz_matrix(plddt_npz, 1, "plddt")
    # Boltz confidence scores are documented as being on a 0-1 scale.
    if np.nanmax(plddt) <= 1.0:
        plddt = plddt * 100.0

    pae_npz = pred_dir / f"pae_{stem}_model_0.npz"
    pae = None
    if pae_npz.exists():
        pae = _read_npz_matrix(pae_npz, 2, "pae")
    return struct, plddt, pae


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--msa", default="empty",
                    help="'empty' for single-sequence mode (default; required on Euler), "
                         "or a path to a .a3m")
    ap.add_argument("--smiles", default=None, help="substrate SMILES (default: target.substrate)")
    ap.add_argument("--ccd", default=None, help="substrate CCD code instead of SMILES")
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    compute = cfg.get("compute") or {}
    device = cli.resolve_device(args, tcfg)

    smiles, ccd = args.smiles, args.ccd
    if args.target and not (smiles or ccd):
        substrate = load_target(args.target).substrate or {}
        smiles = substrate.get("smiles")
        ccd = substrate.get("reference_ligand") if not smiles else None
    if not (smiles or ccd):
        log.warning("no substrate given: folding the protein alone, which wastes the one thing "
                    "this backend is here for. Set substrate.smiles in the target config.")

    inputs = cli.load_inputs(args.inp, args.limit)
    if inputs.kind != "fasta":
        raise SystemExit(f"{TOOL} folds sequences: --in must be a FASTA file, got {inputs.path}")
    common.check_fits(max(len(r.seq) for r in inputs.sequences), compute, BACKEND)

    with cli.stage(TOOL, args, tcfg, VERSION,
                   hash_extra={"n_inputs": len(inputs), "msa": args.msa,
                               "smiles": smiles, "ccd": ccd}) as stage:
        if stage is None:
            return 0
        out_dir = stage.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [r for r in inputs.sequences
                if args.force or not common.already_folded(out_dir, r.id)]
        if todo:
            exe = boltz_binary()
        log.info("boltz: %d sequences, %d to do (device=%s, ligand=%s)",
                 len(inputs), len(todo), device, smiles or ccd or "none")

        rows, failed = [], []
        for rec in inputs.sequences:
            struct_out = out_dir / f"{rec.id}.pdb"
            if not args.force and common.already_folded(out_dir, rec.id):
                import pandas as pd
                pl = pd.read_csv(out_dir / f"{rec.id}.plddt.csv")["plddt"].to_numpy()
                rows.append(common.summary_row(rec.id, struct_out, pl, BACKEND, 0.0,
                                               (out_dir / f"{rec.id}.pae.npz").exists(),
                                               fold_cached=1))
                continue

            work = Path(tempfile.mkdtemp(prefix=f"boltz_{rec.id}_"))
            yaml_path = work / f"{rec.id}.yaml"
            yaml_path.write_text(yaml.safe_dump(
                build_input_yaml(rec.seq, msa=args.msa, ligand_smiles=smiles, ligand_ccd=ccd),
                sort_keys=False), encoding="utf-8")
            cmd = [exe, "predict", str(yaml_path), "--out_dir", str(work),
                   "--output_format", "pdb",
                   "--recycling_steps", str(int(tcfg.get("recycling_steps", 3))),
                   "--diffusion_samples", str(int(tcfg.get("diffusion_samples", 1))),
                   "--accelerator", "cpu" if device.startswith("cpu") else "gpu",
                   "--write_full_pae"]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            if proc.returncode != 0:
                log.error("%s: boltz exited %d:\n%s", rec.id, proc.returncode,
                          proc.stderr[-1500:])
                failed.append(rec.id)
                continue
            try:
                struct, plddt, pae = collect_outputs(work / "predictions" / rec.id, rec.id)
            except (FileNotFoundError, ValueError) as e:
                log.error("%s: %s", rec.id, e)
                failed.append(rec.id)
                continue
            shutil.copyfile(struct, struct_out)
            common.write_plddt(out_dir, rec.id, np.arange(1, len(plddt) + 1), plddt)
            if pae is not None:
                common.write_pae(out_dir, rec.id, pae)
            rows.append(common.summary_row(rec.id, struct_out, plddt, BACKEND, wall,
                                           pae is not None, fold_cached=0,
                                           fold_ligand=smiles or ccd or ""))
            log.info("%s: mean pLDDT %.1f, %.1fs", rec.id, float(np.nanmean(plddt)), wall)

        if failed:
            log.warning("%d design(s) failed: %s", len(failed), ", ".join(failed[:8]))
        if not rows:
            raise SystemExit(f"{TOOL}: nothing folded successfully")
        stage.write_csv(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

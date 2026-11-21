"""ESMFold: single-sequence structure prediction. The pipeline default.

    python -m src.fold.esmfold --in designs.fasta --out results/run/fold \
        --config configs/pipeline.yaml --device cuda

Why this is the default (DECISIONS.md, 2026-11-16): ESMFold needs no MSA, so it
works on Euler compute nodes, which have restricted outbound internet and would
fail any MMseqs2-web-API backend. It fits Euler's 11 GB RTX 2080Ti in fp16 with
chunked attention — fp32 does not fit — and it is still tight.

Weights are NOT in this repo: ``scripts/fetch_weights.sh esmfold`` downloads
them and records the location in MANIFEST.md.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from .. import cli
from . import common

log = logging.getLogger("pipeline.fold.esmfold")
TOOL = "fold.esmfold"
VERSION = "0.1.0"
BACKEND = "esmfold"


def load_model(device: str, precision: str, chunk_size: int, weights_dir: str | None = None):
    """Load ESMFold via HuggingFace transformers.

    Raises :class:`common.ModelUnavailable` with the exact command to run when
    the package or the weights are missing. It does not fall back to a smaller
    model or to random coordinates.
    """
    try:
        import torch
        from transformers import EsmForProteinFolding
    except ImportError as e:
        raise common.ModelUnavailable(
            f"ESMFold needs torch + transformers ({e.name} missing). On Euler: "
            "`bash env/euler_setup.sh` then `pip install transformers==4.46.3 accelerate==1.1.1`. "
            "On Modal the image in env/modal_image.py already has them."
        ) from e

    kwargs = {}
    if weights_dir:
        kwargs["cache_dir"] = weights_dir
    try:
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", **kwargs)
    except OSError as e:
        raise common.ModelUnavailable(
            f"ESMFold weights not available locally ({e}). Run "
            "`bash scripts/fetch_weights.sh esmfold` on this machine first. Euler compute nodes "
            "have no outbound internet, so fetch on the login node into $SCRATCH."
        ) from e

    model = model.eval()
    if device.startswith("cuda"):
        model = model.cuda() if device == "cuda" else model.to(device)
        if precision == "fp16":
            # fp16 + chunked attention is what makes this fit in 11 GB.
            model.esm = model.esm.half()
            model.trunk.set_chunk_size(chunk_size)
            log.info("ESMFold: fp16 ESM trunk, chunk_size=%d (required to fit 11 GB)", chunk_size)
        else:
            log.warning("ESMFold in %s on GPU: this does NOT fit an 11 GB card", precision)
    else:
        log.warning("ESMFold on CPU: minutes per sequence. Use --device cuda where possible.")
    return model, torch


def fold_one(model, torch, sequence: str, num_recycles: int = 1) -> tuple[str, np.ndarray, np.ndarray]:
    """Fold one sequence -> (PDB text, per-residue pLDDT 0-100, PAE matrix).

    ESMFold writes pLDDT into the B-factor column of its PDB output, which is
    exactly where Tier-2 reads it from.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, num_recycles=num_recycles)
    pdb_text = model.output_to_pdb(out)[0]
    # ESMFold's plddt is per-atom (L, 37); take the CA column (atom index 1).
    plddt = out["plddt"][0, :, 1].detach().float().cpu().numpy()
    if np.nanmax(plddt) <= 1.0:
        plddt = plddt * 100.0
    pae = out.get("predicted_aligned_error")
    pae = pae[0].detach().float().cpu().numpy() if pae is not None else None
    return pdb_text, plddt, pae


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--weights-dir", default=None, help="model cache (default: HF cache / $SCRATCH)")
    ap.add_argument("--num-recycles", type=int, default=None)
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    compute = cfg.get("compute") or {}
    device = cli.resolve_device(args, tcfg)
    precision = common.resolve_precision(device, compute)
    chunk = int(tcfg.get("chunk_size", 64))
    recycles = args.num_recycles if args.num_recycles is not None \
        else int(tcfg.get("num_recycles", 1))

    inputs = cli.load_inputs(args.inp, args.limit)
    if inputs.kind != "fasta":
        raise SystemExit(f"{TOOL} folds sequences: --in must be a FASTA file, got {inputs.path}")
    longest = max(len(r.seq) for r in inputs.sequences)
    common.check_fits(longest, compute, BACKEND)

    with cli.stage(TOOL, args, tcfg, VERSION,
                   hash_extra={"n_inputs": len(inputs), "precision": precision,
                               "recycles": recycles, "chunk": chunk}) as stage:
        if stage is None:
            return 0
        out_dir = stage.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        todo = [r for r in inputs.sequences
                if args.force or not common.already_folded(out_dir, r.id)]
        log.info("esmfold: %d sequences, %d already folded, %d to do (device=%s, %s)",
                 len(inputs), len(inputs) - len(todo), len(todo), device, precision)

        model = torch = None
        if todo:
            model, torch = load_model(device, precision, chunk, args.weights_dir)

        rows, failed = [], []
        for rec in inputs.sequences:
            struct_path = out_dir / f"{rec.id}.pdb"
            if not args.force and common.already_folded(out_dir, rec.id):
                import pandas as pd
                pl = pd.read_csv(out_dir / f"{rec.id}.plddt.csv")["plddt"].to_numpy()
                rows.append(common.summary_row(rec.id, struct_path, pl, BACKEND, 0.0,
                                               (out_dir / f"{rec.id}.pae.npz").exists(),
                                               fold_cached=1))
                continue
            t0 = time.perf_counter()
            try:
                pdb_text, plddt, pae = fold_one(model, torch, rec.seq, recycles)
            except Exception as e:
                # One design failing must not cost the batch.
                log.error("%s failed: %s: %s", rec.id, type(e).__name__, e)
                failed.append(rec.id)
                if "out of memory" in str(e).lower():
                    log.error("CUDA OOM: lower fold.esmfold.chunk_size (now %d) or use the "
                              "modal backend", chunk)
                continue
            wall = time.perf_counter() - t0
            struct_path.write_text(pdb_text, encoding="utf-8")
            common.write_plddt(out_dir, rec.id, np.arange(1, len(plddt) + 1), plddt)
            if pae is not None:
                common.write_pae(out_dir, rec.id, pae)
            rows.append(common.summary_row(rec.id, struct_path, plddt, BACKEND, wall,
                                           pae is not None, fold_cached=0))
            log.info("%s: %d aa, mean pLDDT %.1f, %.1fs", rec.id, len(rec.seq),
                     float(np.nanmean(plddt)), wall)

        if failed:
            log.warning("%d design(s) failed to fold: %s", len(failed), ", ".join(failed[:8]))
        if not rows:
            raise SystemExit(f"{TOOL}: nothing folded successfully")
        stage.write_csv(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

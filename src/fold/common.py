"""Shared machinery for fold backends, so each wrapper is thin and identical.

Every fold backend, whatever the model, writes the same things into ``--out``:

    <design_id>.pdb          predicted structure, pLDDT in the B-factor column
    <design_id>.plddt.csv    per-residue pLDDT (resi, plddt)
    <design_id>.pae.npz      PAE matrix, when the backend produces one
    fold.<backend>.csv       one row per design
    fold.<backend>.meta.json the stage contract

Tier-2 scoring reads exactly those, so swapping backends when Euler is
contended changes nothing downstream.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("pipeline.fold")


class ModelUnavailable(RuntimeError):
    """Model weights or the package implementing a backend are missing.

    Carries the exact install/fetch command, because at hour 20 nobody wants to
    reconstruct it.
    """


def check_fits(n_residues: int, compute_cfg: dict, backend: str) -> None:
    """Refuse oversized jobs up front rather than OOMing on a shared GPU.

    ``compute.max_residues`` is a config value (11 GB on Euler's 2080Ti is the
    reason it exists), never an assumption buried in code.
    """
    max_res = int(compute_cfg.get("max_residues", 0) or 0)
    if max_res and n_residues > max_res:
        raise SystemExit(
            f"{backend}: {n_residues} residues exceeds compute.max_residues={max_res} in "
            f"configs/pipeline.yaml. Euler's RTX 2080Ti has {compute_cfg.get('max_vram_gb')} GB "
            "and no bf16. Raise the limit deliberately, use the modal backend, or split the job."
        )


def resolve_precision(device: str, compute_cfg: dict) -> str:
    """fp16 on GPU by default; fp32 on CPU. bf16 only if the config allows it.

    Euler's sm_75 cards have no bf16 support at all, so ``allow_bf16: false`` in
    the pipeline config is a hardware fact, not a preference.
    """
    want = str(compute_cfg.get("precision", "fp16")).lower()
    if want == "bf16" and not compute_cfg.get("allow_bf16", False):
        raise SystemExit(
            "compute.precision=bf16 but compute.allow_bf16=false. Euler's RTX 2080Ti (sm_75) "
            "has no bf16 support; set precision to fp16 or run on the modal backend."
        )
    if device.startswith("cpu"):
        if want != "fp32":
            log.info("device is CPU: using fp32 regardless of compute.precision=%s", want)
        return "fp32"
    return want


def write_plddt(out_dir: Path, design_id: str, resi: np.ndarray, plddt: np.ndarray) -> Path:
    """Per-residue pLDDT as ``<design_id>.plddt.csv``. Figure 3 reads this file."""
    path = Path(out_dir) / f"{design_id}.plddt.csv"
    lines = ["resi,plddt"] + [f"{int(r)},{float(v):.4f}" for r, v in zip(resi, plddt)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_pae(out_dir: Path, design_id: str, pae: np.ndarray) -> Path:
    """PAE matrix as ``<design_id>.pae.npz``, which Tier-2 picks up automatically."""
    path = Path(out_dir) / f"{design_id}.pae.npz"
    np.savez_compressed(path, pae=np.asarray(pae, dtype=np.float32))
    return path


def summary_row(design_id: str, structure_path: Path, plddt: np.ndarray,
                backend: str, wall_s: float, has_pae: bool, **extra) -> dict:
    """The per-design row every fold backend writes. Keep these columns identical."""
    return {
        "design_id": design_id,
        "fold_backend": backend,
        "structure_path": str(structure_path),
        "fold_mean_plddt": float(np.nanmean(plddt)),
        "fold_min_plddt": float(np.nanmin(plddt)),
        "fold_n_residues": int(len(plddt)),
        "fold_has_pae": int(has_pae),
        "fold_seconds": round(wall_s, 2),
        **extra,
    }


def already_folded(out_dir: Path, design_id: str) -> bool:
    """A design is folded if its structure and per-residue pLDDT both exist.

    Lets a crashed fold stage resume design-by-design instead of restarting a
    batch that cost an hour of shared GPU time.
    """
    out_dir = Path(out_dir)
    return (out_dir / f"{design_id}.pdb").exists() and \
           (out_dir / f"{design_id}.plddt.csv").exists()

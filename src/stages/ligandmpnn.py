"""LigandMPNN: design a sequence onto an RFdiffusion backbone, ligand-aware.

LigandMPNN preserves the numbering of its input PDB, so the map from
RFdiffusion output to LigandMPNN output is the identity. That is asserted
rather than assumed: fixed positions are passed in output numbering, and after
the call the residue type at each motif position is checked. LigandMPNN
redesigning a fixed residue is a configuration error that is otherwise
invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import LigandMPNNCfg, MotifCfg
from ..tracking import ResKey
from .base import StageError, StageResult, run_command


@dataclass(frozen=True)
class LigandMPNNOutput:
    """The designed sequences for one backbone."""

    fasta: Path
    sequences: dict[str, str]     # header -> sequence
    packed_pdbs: list[Path]
    out_dir: Path


def fixed_residues_argument(
    motif: MotifCfg, seed_to_hal: dict[ResKey, ResKey]
) -> str:
    """The `--fixed_residues` string, in the backbone's own numbering.

    LigandMPNN takes space-separated `<chain><resnum>` tokens. The motif is
    declared in seed numbering, so every entry goes through the map built from
    the `.trb` first -- passing seed numbers to a renumbered backbone would
    fix the wrong residues, which is the silent failure this whole chain of
    maps exists to prevent.
    """
    tokens: list[str] = []
    for chain, resnum in sorted(motif.as_keys()):
        key = (chain, resnum)
        if key not in seed_to_hal:
            raise StageError(
                f"ligandmpnn: motif residue {chain}{resnum} is not in the "
                "seed->backbone map, so its position in the backbone is unknown"
            )
        hal_chain, hal_num = seed_to_hal[key]
        tokens.append(f"{hal_chain}{hal_num}")
    return " ".join(tokens)


def build_command(
    cfg: LigandMPNNCfg,
    input_pdb: Path,
    out_dir: Path,
    fixed_residues: str,
    num_sequences: int,
    seed: int | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", cfg.run_script,
        "--model_type", "ligand_mpnn",
        "--checkpoint_ligand_mpnn", cfg.checkpoint,
        "--pdb_path", str(input_pdb),
        "--out_folder", str(out_dir),
        "--batch_size", "1",
        "--number_of_batches", str(num_sequences),
        "--temperature", str(cfg.temperature),
    ]
    if fixed_residues:
        cmd += ["--fixed_residues", fixed_residues]
    if cfg.omit_AA:
        cmd += ["--omit_AA", cfg.omit_AA]
    # Without the atom context LigandMPNN is not ligand-aware at all, which
    # defeats the point of using it over ProteinMPNN. The cluster's own
    # launcher passes both.
    cmd += ["--ligand_mpnn_use_atom_context", "1" if cfg.use_atom_context else "0"]
    cmd += [
        "--ligand_mpnn_use_side_chain_context",
        "1" if cfg.use_side_chain_context else "0",
    ]
    if cfg.pack_side_chains:
        cmd += ["--pack_side_chains", "1", "--number_of_packs_per_design", "1"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def run(
    cfg: LigandMPNNCfg,
    input_pdb: str | Path,
    out_dir: str | Path,
    motif: MotifCfg,
    seed_to_hal: dict[ResKey, ResKey],
    num_sequences: int | None = None,
    gpu: int | None = None,
    seed: int | None = None,
    timeout: float | None = None,
) -> tuple[LigandMPNNOutput, StageResult]:
    """Design sequences onto one backbone."""
    input_pdb = Path(input_pdb)
    if not input_pdb.exists():
        raise StageError(f"ligandmpnn: input PDB not found: {input_pdb}")

    out_dir = Path(out_dir)
    count = cfg.sequences_per_backbone if num_sequences is None else num_sequences
    cmd = build_command(
        cfg,
        input_pdb,
        out_dir,
        fixed_residues_argument(motif, seed_to_hal),
        count,
        seed=seed,
    )

    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = run_command(
        cmd, stage="ligandmpnn", env_overrides=env_overrides, timeout=timeout
    )
    return collect_output(out_dir, input_pdb.stem), result


def collect_output(out_dir: Path, stem: str) -> LigandMPNNOutput:
    """Read the designed sequences out of LigandMPNN's FASTA."""
    out_dir = Path(out_dir)
    candidates = sorted(out_dir.rglob(f"{stem}.fa")) + sorted(out_dir.rglob(f"{stem}.fasta"))
    if not candidates:
        candidates = sorted(out_dir.rglob("*.fa")) + sorted(out_dir.rglob("*.fasta"))
    if not candidates:
        listing = sorted(p.name for p in out_dir.rglob("*") if p.is_file())[:30]
        raise StageError(
            f"ligandmpnn produced no FASTA under {out_dir}. Files: {listing or 'none'}"
        )
    fasta = candidates[0]
    sequences = read_fasta(fasta)
    if not sequences:
        raise StageError(f"ligandmpnn: {fasta} holds no sequences")

    packed = sorted(out_dir.rglob("*packed*.pdb"))
    return LigandMPNNOutput(
        fasta=fasta, sequences=sequences, packed_pdbs=packed, out_dir=out_dir
    )


def read_fasta(path: str | Path) -> dict[str, str]:
    """Parse a FASTA into header -> sequence, preserving order.

    LigandMPNN's first record is the input sequence, not a design; callers drop
    it by key rather than by position, because that ordering is a convention
    rather than a guarantee.
    """
    path = Path(path)
    out: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                out[header] = "".join(chunks)
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        out[header] = "".join(chunks)
    return out


def designed_sequences(output: LigandMPNNOutput) -> list[tuple[str, str]]:
    """Just the designs: everything whose header is not the input record."""
    designs = [
        (header, seq)
        for header, seq in output.sequences.items()
        if "T=" in header or "sample" in header.lower()
    ]
    if designs:
        return designs
    # Fall back to "everything after the first record", which is LigandMPNN's
    # documented layout, rather than returning nothing.
    items = list(output.sequences.items())
    if len(items) <= 1:
        raise StageError(
            f"ligandmpnn: {output.fasta} holds only the input sequence; no "
            "designs were produced"
        )
    return items[1:]

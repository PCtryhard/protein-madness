"""Boltz-2: predict the complex from sequence plus ligand.

Boltz-2 is the memory bottleneck on two 12 GB cards. NVIDIA's packaged Boltz-2
NIM states a 48 GB minimum; the open-source `boltz` package is much lighter but
reports put consumer 12 GB cards at roughly 300 to 500 tokens before
exhausting VRAM. On Linux there is no CPU fallback, so the run fails rather
than slowing down.

Hence `refuse_if_too_large`, which is called before every submission. Building
the refusal in is the whole point: the alternative is discovering the limit
mid-search, at which point a worker dies and takes its design with it.

Memory knobs, in the order to cut them: `diffusion_samples`, then
`recycling_steps`, then system size. Dropping `diffusion_samples` to 1 costs
the least accuracy per GB recovered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import BoltzCfg, Ligand
from .base import StageError, StageResult, run_command

# Heavy-atom counts for the CCD codes this project uses, for the token
# estimate. A ligand not listed here must declare its atom count, because
# guessing it would defeat the refusal.
KNOWN_LIGAND_ATOM_COUNTS = {
    # 5-nitrobenzotriazole. Counted from the motif PDBs in data/01..07, and
    # cross-checked against a real Boltz-2 prediction by a test.
    "6NT": 12,
}


class BoltzTooLargeError(StageError):
    """The system exceeds `boltz.max_tokens`. Never clamped, always refused."""


@dataclass(frozen=True)
class BoltzOutput:
    """Where a finished Boltz-2 call put its results."""

    structure: Path          # the predicted complex, .cif
    confidence: Path         # confidence_<name>_model_0.json
    pae: Path | None         # pae_<name>_model_0.npz, when written
    out_dir: Path
    cached: bool = False


def token_count(sequence_lengths: list[int], ligand_atoms: int) -> int:
    """Protein tokens plus ligand heavy-atom tokens.

    Boltz tokenises a polymer one token per residue and a ligand one token per
    heavy atom, so this is the figure the VRAM limit applies to.
    """
    return sum(sequence_lengths) + ligand_atoms


def ligand_atom_count(ligand: Ligand) -> int:
    """Heavy-atom count for the ligand, or raise.

    Never estimated. The token budget is the one guard against an OOM that
    kills a worker, and a guessed input makes it worthless.
    """
    if ligand.ccd:
        count = KNOWN_LIGAND_ATOM_COUNTS.get(ligand.ccd.upper())
        if count is None:
            raise BoltzTooLargeError(
                f"ligand CCD {ligand.ccd!r} has no recorded heavy-atom count, so "
                "the token budget cannot be checked. Add it to "
                "KNOWN_LIGAND_ATOM_COUNTS in src/stages/boltz.py (count the "
                "heavy atoms in the seed PDB) rather than letting the run guess."
            )
        return count

    if ligand.reference_sdf:
        return _sdf_heavy_atom_count(ligand.reference_sdf)

    raise BoltzTooLargeError(
        f"ligand {ligand.id!r} has neither a CCD code nor a reference SDF; its "
        "atom count is unknown and the token budget cannot be checked"
    )


def _sdf_heavy_atom_count(path: str | Path) -> int:
    """Heavy atoms in an SDF, read from the V2000 counts line."""
    path = Path(path)
    if not path.exists():
        raise BoltzTooLargeError(f"ligand reference_sdf not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise BoltzTooLargeError(f"{path} is too short to be an SDF")
    try:
        total = int(lines[3][:3])
    except ValueError as exc:
        raise BoltzTooLargeError(
            f"{path}: cannot read the atom count from the SDF counts line"
        ) from exc
    heavy = 0
    for line in lines[4 : 4 + total]:
        if len(line) >= 34 and line[31:34].strip().upper() != "H":
            heavy += 1
    if heavy == 0:
        raise BoltzTooLargeError(f"{path}: no heavy atoms found")
    return heavy


def refuse_if_too_large(
    sequences: dict[str, str], ligand: Ligand, cfg: BoltzCfg
) -> int:
    """Check the token budget and return the count, or raise.

    Never silently clamps. A refusal is logged and the design is recorded as
    errored; the run continues with the designs that fit.
    """
    atoms = ligand_atom_count(ligand)
    tokens = token_count([len(s) for s in sequences.values()], atoms)
    if tokens > cfg.max_tokens:
        raise BoltzTooLargeError(
            f"system is {tokens} tokens "
            f"({sum(len(s) for s in sequences.values())} protein + {atoms} ligand), "
            f"over boltz.max_tokens={cfg.max_tokens}. Refusing to submit: on "
            "Linux there is no CPU fallback, so this would fail rather than slow "
            "down. Shorten the design or raise max_tokens deliberately."
        )
    return tokens


def write_input_yaml(
    path: Path, sequences: dict[str, str], ligand: Ligand
) -> Path:
    """Write the Boltz-2 input YAML for one design.

    `msa: empty` runs single-sequence: these are de novo designs, so there is
    no MSA to find, and letting Boltz search for one would waste minutes per
    call for nothing.
    """
    entries: list[dict] = []
    for chain_id, sequence in sequences.items():
        entries.append(
            {"protein": {"id": chain_id, "sequence": sequence, "msa": "empty"}}
        )

    ligand_chain = _next_chain_id(list(sequences))
    if ligand.ccd:
        entries.append({"ligand": {"id": ligand_chain, "ccd": ligand.ccd}})
    else:
        entries.append({"ligand": {"id": ligand_chain, "smiles": ligand.smiles}})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"version": 1, "sequences": entries}, fh, sort_keys=False)
    return path


def _next_chain_id(used: list[str]) -> str:
    for letter in "BCDEFGHIJKLMNOPQRSTUVWXYZA":
        if letter not in used:
            return letter
    raise StageError(f"no free chain id left; chains in use: {used}")


def build_command(
    input_yaml: Path, out_dir: Path, cfg: BoltzCfg, gpu: int | None = None
) -> list[str]:
    """The `boltz predict` command line, entirely from config."""
    cmd = [
        cfg.binary, "predict", str(input_yaml),
        "--out_dir", str(out_dir),
        "--cache", str(Path(cfg.cache_dir).expanduser()),
        "--recycling_steps", str(cfg.recycling_steps),
        "--diffusion_samples", str(cfg.diffusion_samples),
        "--sampling_steps", str(cfg.sampling_steps),
        "--output_format", "mmcif",
        "--write_full_pae",
        "--accelerator", cfg.accelerator,
        "--devices", str(cfg.devices),
    ]
    if cfg.no_kernels:
        cmd.append("--no_kernels")
    if cfg.use_potentials:
        cmd.append("--use_potentials")
    if gpu is not None:
        cmd += ["--devices", "1"]
    return cmd


def prepare_cache(cfg: BoltzCfg, cache_dir: Path) -> Path:
    """Symlink the pre-downloaded weights into this run's cache directory.

    Boltz fetches its weights on first use. That fails on a compute node with
    no outbound internet, and even where it works it is 6 GB per cache. The
    clusters ship the assets read-only, so the wrapper links them in:
    boltz2_conf.ckpt, boltz2_aff.ckpt, mols.tar and the mols/ directory.

    Does nothing when `assets_dir` is unset, which is the right behaviour on a
    machine that can download.
    """
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.assets_dir:
        return cache_dir

    assets = Path(cfg.assets_dir).expanduser()
    if not assets.is_dir():
        raise StageError(
            f"boltz.assets_dir is set to {assets}, which is not a directory. "
            "Unset it to let Boltz download its own weights, or point it at the "
            "staged assets."
        )

    wanted = ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar", "mols")
    missing = [n for n in wanted if not (assets / n).exists()]
    if missing:
        raise StageError(
            f"boltz.assets_dir {assets} is missing {missing}. Boltz would try "
            "to download them, which fails on a node with no internet."
        )
    for name in wanted:
        link = cache_dir / name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(assets / name)
    return cache_dir


def predict(
    sequences: dict[str, str],
    ligand: Ligand,
    cfg: BoltzCfg,
    out_dir: Path,
    name: str,
    gpu: int | None = None,
    seed: int | None = None,
    timeout: float | None = None,
) -> tuple[BoltzOutput, StageResult]:
    """Run one Boltz-2 prediction. Raises on refusal, failure or missing output."""
    refuse_if_too_large(sequences, ligand, cfg)

    out_dir = Path(out_dir)
    input_yaml = write_input_yaml(out_dir / f"{name}.yaml", sequences, ligand)
    prepare_cache(cfg, Path(cfg.cache_dir))
    cmd = build_command(input_yaml, out_dir, cfg, gpu=gpu)
    if seed is not None:
        cmd += ["--seed", str(seed)]

    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = run_command(
        cmd, stage="boltz", env_overrides=env_overrides, timeout=timeout
    )
    return locate_output(out_dir, name), result


def locate_output(out_dir: Path, name: str) -> BoltzOutput:
    """Find the files a finished prediction wrote.

    Boltz nests its output as
    `<out_dir>/boltz_results_<name>/predictions/<name>/`, but the exact layout
    has moved between versions, so the structure and confidence files are
    searched for by pattern and their absence is an error with the directory
    listing attached.
    """
    out_dir = Path(out_dir)
    structures = sorted(out_dir.rglob(f"{name}_model_0.cif"))
    if not structures:
        structures = sorted(out_dir.rglob(f"{name}*model_0.cif"))
    if not structures:
        listing = sorted(p.name for p in out_dir.rglob("*") if p.is_file())[:30]
        raise StageError(
            f"Boltz-2 produced no model_0.cif for {name!r} under {out_dir}. "
            f"Files present: {listing or 'none'}"
        )
    structure = structures[0]

    confidences = sorted(structure.parent.glob(f"confidence_{name}*model_0.json"))
    if not confidences:
        confidences = sorted(out_dir.rglob(f"confidence_{name}*model_0.json"))
    if not confidences:
        raise StageError(
            f"Boltz-2 wrote {structure} but no confidence JSON for {name!r}. "
            "The design cannot be scored without it."
        )

    paes = sorted(structure.parent.glob(f"pae_{name}*model_0.npz"))
    return BoltzOutput(
        structure=structure,
        confidence=confidences[0],
        pae=paes[0] if paes else None,
        out_dir=out_dir,
    )


def read_sequences_from_yaml(path: str | Path) -> dict[str, str]:
    """The exact sequences that were submitted, read back from the input file.

    Used to build the hal->boltz map: the index of a motif residue must be
    computed in the exact string that was submitted, not in a string rebuilt
    from the structure afterwards.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in raw.get("sequences", []):
        if "protein" in entry:
            out[str(entry["protein"]["id"])] = str(entry["protein"]["sequence"])
    if not out:
        raise StageError(f"{path} declares no protein chains")
    return out

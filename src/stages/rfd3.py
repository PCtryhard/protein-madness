"""RFD3: motif scaffolding and partial diffusion, as installed on the cluster.

This replaces the RFdiffusion wrapper for the hackathon clusters. The two tools
are not interchangeable and the differences all matter:

* **Input.** RFdiffusion takes a contig string on the command line. RFD3 takes
  a JSON file holding one entry per design, with `contig`, `unindex`,
  `select_fixed_atoms` and `ligand`, and is told which entries to build with
  `json_keys_subset`.

* **Atom-level freezing exists here.** `select_fixed_atoms` pins named atoms
  rather than whole residues -- `A229: OD1,CG,OD2` holds a carboxylate without
  holding the backbone. PIPELINE.md says this is impossible; that is true of
  RFdiffusion and false of RFD3, and the motif specs rely on it.

* **Output.** A gzipped mmCIF plus a metadata JSON, not a PDB plus a `.trb`
  pickle. The motif mapping lives in that JSON as `diffused_index_map`:

      {"diffused_index_map": {"A229": "A56", "A291": "A179"}}

  which is the seed -> output residue map `tracking.py` needs. The same file
  also carries `join_point_rmsd_by_token`, RFD3's own per-residue motif
  fidelity, which is worth logging: it says how well the motif was actually
  held, independently of anything we measure later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import MotifCfg, RFD3Cfg, Seed
from .base import StageError, StageResult, run_command


@dataclass(frozen=True)
class RFD3Output:
    """One generated backbone and the metadata that maps its motif."""

    structure: Path          # *.cif.gz
    metadata: Path           # *.json, holds diffused_index_map
    index: int
    design_key: str

    def read_metadata(self) -> dict:
        try:
            return json.loads(self.metadata.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StageError(
                f"{self.metadata} is not valid JSON, so the motif mapping "
                f"cannot be read: {exc}"
            ) from exc

    def motif_fidelity(self) -> dict[str, float]:
        """RFD3's own per-residue motif RMSD, if it reported any.

        Logged rather than scored: it measures whether RFD3 held the motif,
        which is a different question from whether Boltz-2 reproduces it.
        """
        metrics = self.read_metadata().get("metrics", {})
        by_token = metrics.get("join_point_rmsd_by_token", {})
        return {k: float(v) for k, v in by_token.items()} if isinstance(by_token, dict) else {}


@dataclass
class DesignSpec:
    """One entry in the RFD3 input JSON."""

    key: str
    input_pdb: Path
    contig: str | None
    unindex: str
    select_fixed_atoms: dict[str, str]
    ligand: str | None = None
    # Angstroms of noise for partial diffusion; None turns it off. RFD3 reads
    # this from the design spec, NOT from a command-line override.
    partial_t: float | None = None

    def as_json_value(self) -> dict:
        value: dict[str, object] = {
            # Absolute, because RFD3 resolves it relative to its own cwd and
            # the job's cwd is not the inputs directory.
            "input": str(self.input_pdb.resolve()),
            "unindex": self.unindex,
            "select_fixed_atoms": dict(self.select_fixed_atoms),
        }
        if self.contig is not None:
            value["contig"] = self.contig
        if self.ligand:
            value["ligand"] = self.ligand
        if self.partial_t is not None:
            value["partial_t"] = float(self.partial_t)
        return value


def spec_from_seed(seed: Seed, motif: MotifCfg, ligand_id: str,
                   repo_root: Path, key: str) -> DesignSpec:
    """Build an RFD3 design spec from a target's seed and motif.

    `unindex` names the residues RFD3 must keep from the input, and
    `select_fixed_atoms` says which of their atoms to pin. Both are derived
    from the target spec so there is one source of truth: a residue in
    `motif.fixed` without a `fixed_atoms` entry is pinned whole ("ALL"), which
    is the conservative reading.
    """
    residues = sorted(motif.as_keys())
    unindex = ",".join(f"{chain}{resnum}" for chain, resnum in residues)

    fixed: dict[str, str] = {}
    for chain, resnum in residues:
        label = f"{chain}{resnum}"
        fixed[label] = motif.fixed_atoms.get(label, "ALL")
    if ligand_id:
        # The ligand is always held whole: its pose is the thing the motif is
        # defined against.
        fixed[ligand_id] = "ALL"

    return DesignSpec(
        key=key,
        input_pdb=repo_root / seed.pdb,
        contig=seed.contig,
        unindex=unindex,
        select_fixed_atoms=fixed,
        ligand=ligand_id or None,
    )


def _normalised_input(seed: Path, out_dir: Path) -> Path:
    """A copy of `seed` with a suffix the external tools understand.

    Returns `seed` unchanged when it is already .pdb or .cif. Otherwise parses
    it with our own content-sniffing reader and writes a .pdb into `out_dir`,
    leaving the original untouched.
    """
    from ..tracking import read_structure, write_pdb

    name = seed.name.lower()
    if name.endswith((".pdb", ".ent", ".cif", ".mmcif")):
        return seed

    out_dir.mkdir(parents=True, exist_ok=True)
    base = seed.name.split(".pdb")[0] if ".pdb" in seed.name else seed.stem
    target = out_dir / f"{base}.pdb"
    if not target.exists():
        write_pdb(read_structure(seed), target)
    return target


def write_input_json(specs: list[DesignSpec], path: Path) -> Path:
    """Write the RFD3 input file.

    Every key in this file is a design specification field as far as RFD3 is
    concerned, so engine arguments (ckpt_path, out_dir, batch sizes) must go on
    the command line and must NOT be added here.
    """
    if not specs:
        raise StageError("no RFD3 design specs to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {spec.key: spec.as_json_value() for spec in specs}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_command(
    cfg: RFD3Cfg,
    input_json: Path,
    design_keys: list[str],
    out_dir: Path,
    n_designs: int,
) -> list[str]:
    """The RFD3 command line: a binary plus Hydra-style engine overrides.

    Partial diffusion is NOT set here. `rfd3 partial_T=10` fails Hydra
    composition, and `+partial_T=10` then fails with
    `RFD3InferenceConfig.__init__() got an unexpected keyword argument`.
    RFD3 takes it as `partial_t` on the per-example design spec instead, so it
    travels in the input JSON -- see DesignSpec.partial_t.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ",".join(design_keys)
    cmd = [
        cfg.binary,
        f"inputs={input_json}",
        f"json_keys_subset=[{keys}]",
        f"ckpt_path={cfg.checkpoint}",
        f"out_dir={out_dir}",
        f"diffusion_batch_size={cfg.diffusion_batch_size}",
        f"n_batches={n_designs}",
        "skip_existing=False",
        "prevalidate_inputs=True",
        "cleanup_guideposts=True",
        "output_full_json=True",
        "dump_prediction_metadata_json=True",
    ]
    cmd.extend(cfg.extra_args)
    return cmd


def run(
    cfg: RFD3Cfg,
    specs: list[DesignSpec],
    out_dir: Path,
    n_designs: int | None = None,
    partial: bool = False,
    gpu: int | None = None,
    seed: int | None = None,
    timeout: float | None = None,
) -> tuple[list[RFD3Output], StageResult]:
    """Generate backbones with RFD3 and pair each with its metadata."""
    out_dir = Path(out_dir)
    for spec in specs:
        if not spec.input_pdb.exists():
            raise StageError(f"rfd3: input PDB not found: {spec.input_pdb}")

    # Every external tool here dispatches on the FILE EXTENSION, not on
    # content. A real seed is named 01_his_asp_.pdb.pre-boltz, and ProDy (via
    # LigandMPNN) duly tries to parse it as mmCIF and reports "no atoms". Our
    # own readers sniff the content, but the tools do not.
    #
    # So normalise at the boundary: write a properly-suffixed copy into THIS
    # RUN's directory and hand the tools that. The alternative -- renaming the
    # file in data/ -- edits input that belongs to whoever produced it.
    specs = [replace(spec, input_pdb=_normalised_input(spec.input_pdb, out_dir))
             for spec in specs]

    if partial:
        # contig is dropped: RFD3 forbids `length` during partial diffusion
        # and ignores the contig, since the parent's own length is the target.
        specs = [replace(spec, contig=None, partial_t=float(cfg.partial_t))
                 for spec in specs]
    input_json = write_input_json(specs, out_dir / "rfd3_inputs.json")
    count = cfg.n_batches if n_designs is None else n_designs
    cmd = build_command(
        cfg, input_json, [s.key for s in specs], out_dir, count
    )
    if seed is not None:
        cmd.append(f"seed={seed}")

    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = run_command(
        cmd, stage="rfd3", env_overrides=env_overrides, timeout=timeout
    )
    return collect_outputs(out_dir, [s.key for s in specs]), result


def collect_outputs(out_dir: Path, design_keys: list[str]) -> list[RFD3Output]:
    """Pair every produced structure with its metadata JSON, or raise.

    A structure without its metadata is unusable: the motif mapping lives only
    there, and inferring it from coordinates is the proximity guess the rules
    forbid.
    """
    out_dir = Path(out_dir)
    outputs: list[RFD3Output] = []

    for structure in sorted(out_dir.rglob("*.cif.gz")):
        # RFD3 names outputs <task>_<index>_model_0.cif.gz and writes
        # <task>_<index>_model_0.json beside them.
        metadata = structure.with_suffix("").with_suffix(".json")
        if not metadata.is_file():
            raise StageError(
                f"rfd3 wrote {structure} but no {metadata.name}. The motif map "
                "lives in that file, so the design cannot be traced. Check that "
                "dump_prediction_metadata_json was passed."
            )
        # The BATCH index is the digit before "_model_", not the last digit
        # in the name: scanning backwards finds model_0's "0" first and so
        # reported every backbone as index 0. That fed `mpnn_{index}`, so all
        # five backbones shared one LigandMPNN output directory.
        stem = structure.name.split(".cif")[0]
        parts = stem.split("_")
        index = 0
        if "model" in parts:
            before = parts[: parts.index("model")]
            index = next((int(p) for p in reversed(before) if p.isdigit()), 0)
        else:
            index = next((int(p) for p in reversed(parts) if p.isdigit()), 0)
        key = next((k for k in design_keys if k in stem), stem)
        outputs.append(
            RFD3Output(structure=structure, metadata=metadata,
                       index=index, design_key=key)
        )

    if not outputs:
        listing = sorted(p.name for p in out_dir.rglob("*") if p.is_file())[:30]
        raise StageError(
            f"rfd3 produced no *.cif.gz under {out_dir}. "
            f"Files present: {listing or 'none'}"
        )
    return outputs


def assert_motif_in_metadata(output: RFD3Output, motif: MotifCfg) -> None:
    """Fail fast if RFD3's map is missing a motif residue.

    Thin wrapper over tracking.seed_to_hal_map_rfd3, kept here so a stage can
    check its own output before the design travels further.
    """
    from ..tracking import seed_to_hal_map_rfd3

    seed_to_hal_map_rfd3(output.metadata, motif)

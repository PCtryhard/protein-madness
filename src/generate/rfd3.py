"""Backbone generation with RFdiffusion3: fresh de novo, or partial diffusion.

Two modes, and the team is using both:

**De novo** (``mode="denovo"``) - scaffold a brand-new backbone around nothing
but the catalytic motif and the ligand. ``contig: "260-260"`` asks for exactly
260 residues; the base, the donor and 6NT are pinned by ``select_fixed_atoms``
and everything else is invented. This is the arm that is clear of both
eligibility filters, because the result is a homologue of nothing.

**Partial diffusion** (``mode="partial"``) - take an existing structure, add
``partial_t`` angstroms of noise, and re-diffuse. RFD3 exposes this as:

    partial_t: Angstroms of noise to add for partial diffusion
               (None turns off partial diffusion), t <= 15 recommended.

That is a *structural* mutation operator, and it composes with the sequence-level
evolution in :mod:`src.evolve`: point mutations explore sequence space around a
fixed backbone, partial diffusion explores backbone space around a fixed design.
Small ``partial_t`` stays near the parent; large ``partial_t`` approaches de novo.

Both write the same ``Kemp.json``-shaped input the organisers' example uses, so
anything built on top of their pipeline keeps working.

**The catalytic motif is pinned in both modes.** It is the one thing we never
let the model reconsider, because a design that lost it is not an enzyme
regardless of every other number.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("pipeline.generate.rfd3")

#: The HG3.17-derived Kemp motif, as the organisers ship it.
#: Base, H-bond donor, and the pi-stacking tryptophan (optional third residue).
KEMP_BASE = "A127"
KEMP_DONOR = "A50"
KEMP_STACK = "A44"
KEMP_LIGAND = "6NT"

#: Which atoms of each motif residue are held in 3D. Copied from the
#: organisers' inputs/Kemp.json -- these are not ours to invent.
KEMP_FIXED_ATOMS: dict[str, str] = {
    KEMP_BASE: "OD1,CG,OD2",     # aspartate carboxylate: the catalytic base
    KEMP_DONOR: "OE1,CD,NE2",    # glutamine amide: stabilises the oxyanion
    KEMP_STACK: "CD1,NE1,CE2",   # tryptophan indole: pi-stacks the substrate
    KEMP_LIGAND: "ALL",
}

#: RFD3 recommends partial_t <= 15 A; beyond that it is effectively de novo.
MAX_PARTIAL_T = 15.0


@dataclass
class BackboneSpec:
    """One RFD3 design request, in the organisers' Kemp.json dialect."""

    name: str
    input_pdb: str
    contig: str = "260-260"
    ligand: str = KEMP_LIGAND
    motif_residues: tuple[str, ...] = (KEMP_BASE, KEMP_DONOR)
    partial_t: float | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.partial_t is not None:
            if self.partial_t <= 0:
                raise ValueError("partial_t must be positive; use None to disable")
            if self.partial_t > MAX_PARTIAL_T:
                log.warning(
                    "partial_t=%.1f A exceeds the recommended maximum of %.0f A; "
                    "this is closer to de novo generation than to refinement",
                    self.partial_t, MAX_PARTIAL_T,
                )
        unknown = [r for r in self.motif_residues if r not in KEMP_FIXED_ATOMS]
        if unknown:
            raise ValueError(
                f"no fixed-atom specification for {unknown}; add it to "
                f"KEMP_FIXED_ATOMS rather than letting RFD3 choose"
            )

    def as_entry(self) -> dict:
        """The JSON entry RFD3 consumes.

        Note the organisers' warning in their own Kemp.json: RFD3's native JSON
        treats *every* key as a DesignInputSpecification field, so engine
        arguments (out_dir, ckpt_path, seed, n_batches) must go on the command
        line, never in this file.
        """
        fixed = {r: KEMP_FIXED_ATOMS[r] for r in self.motif_residues}
        fixed[self.ligand] = KEMP_FIXED_ATOMS[self.ligand]
        entry: dict = {
            "input": self.input_pdb,
            "ligand": self.ligand,
            "contig": self.contig,
            "unindex": ",".join(self.motif_residues),
            "select_fixed_atoms": fixed,
        }
        if self.partial_t is not None:
            entry["partial_t"] = self.partial_t
        entry.update(self.extra)
        return entry


def denovo(
    name: str,
    input_pdb: str = "HG3_17_kemp_motif.pdb",
    length: int = 260,
    with_stack: bool = False,
) -> BackboneSpec:
    """A fresh backbone built around the catalytic motif and nothing else.

    ``with_stack=True`` adds the Trp44 pi-stack as a third motif residue. Harder
    to scaffold, but a pi-stacked ligand is a more confidently placed ligand -
    and the measured failure mode is precisely that the ligand drifts away from
    the base (BRIEF.md 1). Worth running as a parallel arm.

    ``length`` is held exactly (``"260-260"``) rather than sampled, because the
    burial metric counts CA atoms near the base and more protein means more
    atoms available to be close. 260 is the competition maximum.
    """
    if not 1 <= length <= 260:
        raise ValueError(f"length must be between 1 and 260 (competition limit), got {length}")
    residues = (KEMP_BASE, KEMP_DONOR, KEMP_STACK) if with_stack else (KEMP_BASE, KEMP_DONOR)
    return BackboneSpec(
        name=name, input_pdb=input_pdb, contig=f"{length}-{length}", motif_residues=residues
    )


def partial(
    name: str,
    parent_pdb: str,
    partial_t: float = 5.0,
    length: int = 260,
    with_stack: bool = False,
) -> BackboneSpec:
    """Re-noise and re-diffuse an existing design: a structural mutation.

    This is the knob for the partial-diffusion arm. Guidance from RFD3 itself is
    ``t <= 15`` angstroms; in practice treat it as a dial:

    * ``2-4 A``  - gentle refinement, stays recognisably the parent
    * ``5-8 A``  - meaningful backbone change, same overall fold
    * ``10-15 A``- substantial rearrangement, approaching de novo

    Pair a small ``partial_t`` with the sequence-level loop for local search, and
    a large one when the sequence loop has stopped improving (the same
    stagnation logic as :meth:`src.evolve.mutate.MutationPolicy.stagnant`).
    """
    residues = (KEMP_BASE, KEMP_DONOR, KEMP_STACK) if with_stack else (KEMP_BASE, KEMP_DONOR)
    return BackboneSpec(
        name=name,
        input_pdb=parent_pdb,
        contig=f"{length}-{length}",
        motif_residues=residues,
        partial_t=partial_t,
    )


def write_input_json(path: Path, specs: list[BackboneSpec]) -> Path:
    """Write a Kemp.json-shaped file holding one entry per design."""
    if not specs:
        raise ValueError("no specs to write")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate spec names: {names}")
    payload = {
        "_comment": (
            "Generated by src.generate.rfd3. Engine arguments (out_dir, ckpt_path, "
            "seed, n_batches, diffusion_batch_size) go on the rfd3 command line, "
            "never in this file: RFD3 treats every key here as a "
            "DesignInputSpecification field."
        ),
    }
    for spec in specs:
        payload[spec.name] = spec.as_entry()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    log.info("wrote %d RFD3 specs to %s", len(specs), path)
    return path


def rfd3_command(
    input_json: str,
    keys: list[str],
    out_dir: str,
    checkpoint: str = "$RFD3_CHECKPOINT",
    seed: int = 1,
    n_batches: int = 1,
    batch_size: int = 1,
    rfd3: str = "$ENV_ROOT/env-rfd3/bin/rfd3",
) -> str:
    """The rfd3 invocation, matching the organisers' verified kemp-e2e.sbatch.

    ``n_batches`` multiplies designs per key, which is how a de novo arm gets
    many backbones from one spec: the seed advances per batch.
    """
    if not keys:
        raise ValueError("no json_keys_subset given; RFD3 would run every entry")
    key_list = ",".join(keys)
    return (
        f'"{rfd3}" inputs="{input_json}" '
        f"'json_keys_subset=[{key_list}]' "
        f'ckpt_path="{checkpoint}" out_dir="{out_dir}" '
        f"diffusion_batch_size={batch_size} n_batches={n_batches} seed={seed} "
        f"skip_existing=False prevalidate_inputs=True "
        f"cleanup_guideposts=True output_full_json=True dump_prediction_metadata_json=True"
    )

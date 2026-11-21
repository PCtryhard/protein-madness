"""Generate config/targets/0N_*.yaml from the motif PDBs in data/01..07.

    python scripts/build_targets.py            # write the seven target files
    python scripts/build_targets.py --check    # fail if they are out of date

Seven variants of the same catalytic constellation, one per data folder, each
an independent search. The naming in the filenames is
`<residue 291>_<residue 229>_<residue 261>`:

    01  his_asp_       ASP229             HIS291     (no 261)
    02  his_asp_his    ASP229   HIS261    HIS291
    03  asn_asp_his    ASP229   HIS261    ASN291
    04  gln_asp_his    ASP229   HIS261    GLN291
    05  his_glu_his    GLU229   HIS261    HIS291
    06  asn_glu_his    GLU229   HIS261    ASN291
    07  gln_glu_his    GLU229   HIS261    GLN291

Position 229 is the catalytic base, which accepts from the benzotriazole N3-H.
Position 291 is the donor, which donates to the ligand N1. Position 261 is an
added histidine donating to a nitro oxygen.

WHY THE IDEAL ANGLES ARE NOT READ OUT OF THESE FILES
----------------------------------------------------

Only variant 01's two contacts are crystal geometry. Every mutated or added
residue in 02-07 was modelled in without repacking, and its new atoms land on
top of the ligand. Measured in the files as they stand:

    ASP229 OD2...N3    2.734 A    crystal, real
    HIS291 NE2...N1    3.084 A    crystal, real
    GLU229 OE2...N3    1.677 A    IMPOSSIBLE -- modelled, clashes
    HIS261 NE2...O11   1.820 A    IMPOSSIBLE -- modelled, clashes
    ASN291 ND2...N1    4.823 A    too far -- modelled, points away
    GLN291 NE2...N1    3.860 A    too far -- modelled

That is normal and harmless for an RFdiffusion motif input, because the
scaffold and sidechains get rebuilt. But it means the seed's own geometry is
not a target to aim at. So:

* Angles at an acceptor that exists unmutated in the 5FQJ crystal use the
  measured crystal value.
* Angles at a mutated or added acceptor use the hybridisation value for that
  chemistry, and say so in a comment.

The ligand-side ideal is the same in every variant by construction: the
acceptor is a ligand atom, so its lone-pair direction is a property of the
ligand and does not depend on which residue donates to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGETS = ROOT / "config" / "targets"

# ---------------------------------------------------------------------------
# chemistry per residue type: the donor/acceptor atom and its antecedent
# ---------------------------------------------------------------------------

BASE_ATOMS = {              # position 229, the catalytic base (an acceptor)
    "ASP": ("OD2", "CG"),
    "GLU": ("OE2", "CD"),
}
DONOR_ATOMS = {             # position 291, donates to ligand N1
    "HIS": ("NE2", "CD2"),
    "ASN": ("ND2", "CG"),
    "GLN": ("NE2", "CD"),
}
NITRO_ATOMS = {             # position 261, donates to a nitro oxygen
    "HIS": ("NE2", "CD2"),
}

# Measured in the 5FQJ crystal; see tests/test_tracking.py, which asserts them.
CRYSTAL_BASE_ANGLE_ASP = 106.6      # CG-OD2...N3
CRYSTAL_LIGAND_N1_ANGLE = 126.6     # N2-N1...NE2

# Hybridisation values, used only where the crystal has nothing to offer.
SP2_CARBOXYLATE = 120.0             # a modelled GLU229
SP2_NITRO_OXYGEN = 120.0            # the ligand nitro O11

LIGAND_NAME = "6NT"


def read_motif(pdb: Path) -> dict[int, str]:
    """Residue number -> three-letter name, for the protein residues present."""
    residues: dict[int, str] = {}
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        residues[int(line[22:26])] = line[17:20].strip()
    return residues


def contigs_for(positions: list[int]) -> str:
    """An RFdiffusion motif-scaffolding contig string.

    Scaffold segments between and around the fixed motif residues. The ranges
    give RFdiffusion freedom over total length while keeping the whole system
    comfortably inside the Boltz-2 token budget: the longest expansion here is
    143 residues + 12 ligand atoms = 155 tokens, against max_tokens 400 and the
    300 that PIPELINE.md prefers.

    UNVERIFIED. This string is this script's construction, not something taken
    from a run that worked. The earlier runs in data/No_mutation_run used RFD3,
    whose contig syntax is different. Confirm it against the actual
    RFdiffusion build in phase 1 before trusting a production run.
    """
    parts = ["20-40"]
    for index, position in enumerate(sorted(positions)):
        if index:
            parts.append("10-30")
        parts.append(f"A{position}-{position}")
    parts.append("20-40")
    return "/".join(parts)


def build(folder: Path, pdb: Path) -> str:
    variant = folder.name                       # "01"
    stem = pdb.stem                             # "01_his_asp_"
    label = stem[3:].strip("_") or "motif"      # "his_asp_"  -> "his_asp"
    residues = read_motif(pdb)

    if 229 not in residues or 291 not in residues:
        raise SystemExit(
            f"{pdb}: expected residues 229 and 291, found {sorted(residues)}"
        )
    base_res = residues[229]
    donor_res = residues[291]
    has_nitro = 261 in residues

    if base_res not in BASE_ATOMS:
        raise SystemExit(f"{pdb}: unexpected base residue {base_res}229")
    if donor_res not in DONOR_ATOMS:
        raise SystemExit(f"{pdb}: unexpected donor residue {donor_res}291")

    base_atom, base_ante = BASE_ATOMS[base_res]
    donor_atom, donor_ante = DONOR_ATOMS[donor_res]

    if base_res == "ASP":
        base_ideal, base_note = (
            CRYSTAL_BASE_ANGLE_ASP,
            "measured in the 5FQJ crystal (CG-OD2...N3)",
        )
    else:
        base_ideal, base_note = (
            SP2_CARBOXYLATE,
            "sp2 carboxylate; GLU229 is a modelled mutation and its "
            "coordinate in the seed clashes with the ligand at 1.677 A, "
            "so the seed's own angle is NOT a target",
        )

    positions = sorted(residues)
    lines: list[str] = [
        f"# Variant {variant}: {label}",
        "#",
        f"#   base      A229  {base_res}   accepts from the benzotriazole N3-H",
        f"#   donor     A291  {donor_res}   donates to the ligand N1",
    ]
    if has_nitro:
        lines.append(
            f"#   nitro     A261  {residues[261]}   donates to the ligand nitro O11"
        )
    else:
        lines.append("#   (no residue at 261 in this variant)")
    lines += [
        "#",
        "# GENERATED by scripts/build_targets.py -- edit that script, not this file.",
        "# Run `python scripts/build_targets.py --check` to verify it is current.",
        "#",
        "# The ideal angles below are NOT read out of the seed PDB. Only variant 01's",
        "# two contacts are crystal geometry; every mutated or added residue was",
        "# modelled in without repacking and clashes with the ligand. See the",
        "# docstring of scripts/build_targets.py for the measured values and why",
        "# they are not targets.",
        "",
        f"name: \"{variant}_{label}\"",
        "",
        "seeds:",
        "  # A motif-only PDB: the fixed residues plus the ligand, in 5FQJ numbering.",
        "  # Cycle 0 is motif scaffolding (no partial_T); the wiggle branch then",
        "  # re-noises the backbones that come out of it.",
        "  #",
        "  # The id is quoted because YAML reads a bare 01 as the integer 1, and the",
        "  # id ends up in design ids, filenames and the scaffold column.",
        f"  - id: \"{variant}\"",
        f"    pdb: {pdb.relative_to(ROOT).as_posix()}",
        f"    contigs: \"{contigs_for(positions)}\"   # UNVERIFIED, see build_targets.py",
        "",
        "ligand:",
        f"  id: {LIGAND_NAME}",
        f"  ccd: {LIGAND_NAME}                   # a real CCD code, so atom names resolve directly",
        "  smiles: null",
        "  reference_sdf: null",
        "",
        "motif:",
        "  fixed:                       # whole residues, never mutated, never redesigned",
    ]
    for position in positions:
        lines.append(
            f"    - {{chain: A, resnum: {position}}}   # {residues[position]}"
        )

    lines += [
        "",
        "hbonds:",
        f"  - id: lig_n3_to_{base_res.lower()}229_{base_atom.lower()}",
        "    # The benzotriazole N3-H donates to the carboxylate. The donor hydrogen",
        "    # is on the ligand and its position is fixed by the aromatic ring.",
        "    donor:    {site: ligand,  atom: N3}",
        f"    acceptor: {{site: protein, chain: A, resnum: 229, atom: {base_atom}}}",
        f"    acceptor_antecedent: {{site: protein, chain: A, resnum: 229, atom: {base_ante}}}",
        "    rotatable_donor: false",
        "    required: true",
        f"    acceptor_ideal_angle: {base_ideal}   # {base_note}",
        "",
        f"  - id: {donor_res.lower()}291_{donor_atom.lower()}_to_lig_n1",
        f"    # {donor_res}291 donates to the ligand N1. The acceptor is a ligand atom, so",
        "    # its ideal direction is a property of the ligand and is the same in",
        "    # every variant regardless of which residue donates.",
        f"    donor:    {{site: protein, chain: A, resnum: 291, atom: {donor_atom}}}",
        f"    donor_antecedent: {{site: protein, chain: A, resnum: 291, atom: {donor_ante}}}",
        "    acceptor: {site: ligand,  atom: N1}",
        "    acceptor_antecedent: {site: ligand, atom: N2}",
        "    rotatable_donor: false",
        "    required: true",
        f"    acceptor_ideal_angle: {CRYSTAL_LIGAND_N1_ANGLE}   "
        "# measured in the 5FQJ crystal (N2-N1...NE2)",
    ]

    if has_nitro:
        nitro_atom, nitro_ante = NITRO_ATOMS[residues[261]]
        lines += [
            "",
            f"  - id: {residues[261].lower()}261_{nitro_atom.lower()}_to_lig_o11",
            f"    # {residues[261]}261 donates to a nitro oxygen, stabilising the developing",
            "    # negative charge. Scored but NOT gated: this contact is the thing the",
            "    # 02-07 variants exist to test, so a design that fails to form it should",
            "    # rank low rather than be thrown away. Change `required` to true to gate",
            "    # on it instead.",
            f"    donor:    {{site: protein, chain: A, resnum: 261, atom: {nitro_atom}}}",
            f"    donor_antecedent: {{site: protein, chain: A, resnum: 261, atom: {nitro_ante}}}",
            "    acceptor: {site: ligand,  atom: O11}",
            "    acceptor_antecedent: {site: ligand, atom: NO1}",
            "    rotatable_donor: false",
            "    required: false",
            f"    acceptor_ideal_angle: {SP2_NITRO_OXYGEN}   "
            "# sp2 nitro oxygen; the modelled",
            "    # seed coordinate is a 1.820 A clash at 94.2 deg and is not a target",
        ]

    return "\n".join(lines) + "\n"


def discover() -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for folder in sorted(DATA.glob("0[1-9]")):
        pdbs = sorted(folder.glob("inputs/*/*.pdb"))
        if not pdbs:
            print(f"warning: {folder} holds no inputs/*/*.pdb, skipping", file=sys.stderr)
            continue
        if len(pdbs) > 1:
            raise SystemExit(
                f"{folder} holds {len(pdbs)} PDBs; one motif per folder is expected: "
                f"{[p.name for p in pdbs]}"
            )
        found.append((folder, pdbs[0]))
    if not found:
        raise SystemExit(f"no variant folders found under {DATA}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if any target file is missing or out of date",
    )
    args = parser.parse_args()

    TARGETS.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []

    for folder, pdb in discover():
        label = pdb.stem[3:].strip("_") or "motif"
        out = TARGETS / f"{folder.name}_{label}.yaml"
        text = build(folder, pdb)
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != text:
                stale.append(out.name)
            continue
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")

    if args.check:
        if stale:
            print(
                "out of date (run `python scripts/build_targets.py`): "
                + ", ".join(stale),
                file=sys.stderr,
            )
            return 1
        print("all target files are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

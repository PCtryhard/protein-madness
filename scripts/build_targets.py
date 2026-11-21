"""Generate config/targets/0N_*.yaml from the motif PDBs in data/01..07.

    python scripts/build_targets.py            # write the seven target files
    python scripts/build_targets.py --check    # fail if they are out of date
    python scripts/build_targets.py --verbose  # show every measurement

THE HANDOFF THIS SCRIPT IS BUILT FOR
------------------------------------

The motif files ship with deliberate steric clashes: every mutated or added
residue in variants 02-07 was modelled in without repacking, so its new atoms
land on top of the ligand. When refolded, non-clashing replacements arrive, the
whole update is:

    1. drop the new PDBs into data/0N/inputs/<anything>/
    2. python scripts/build_targets.py
    3. read the table it prints

Nothing else changes. This script decides per bond where the ideal angle comes
from, and it prefers the seed as soon as the seed is believable:

  measured   the seed's own geometry is plausible (an H-bond distance between
             scoring.distance.clash and .fail), so the angle in the seed IS the
             design target -- that is the geometry RFdiffusion will be asked to
             reproduce. Preferred whenever available.
  crystal    the seed's geometry is not plausible, but this acceptor exists
             unmutated in the 5FQJ crystal, so the crystal value stands in.
  chemistry  neither: fall back to the hybridisation value for that acceptor
             and WARN, because nothing measured supports it.

So today the script reports mostly `crystal` and `chemistry` with warnings.
Once the refolded files land it should report `measured` for everything, and it
says so per bond. If it still warns, the new files still clash -- which is
worth knowing before a GPU is booked.

NAMING
------

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
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_pipeline                      # noqa: E402
from src.scoring.hbond import angle_between               # noqa: E402
from src.tracking import read_structure                   # noqa: E402

DATA = ROOT / "data"
TARGETS = ROOT / "config" / "targets"
PIPELINE = ROOT / "config" / "pipeline.yaml"

# ---------------------------------------------------------------------------
# chemistry per residue type: the functional atom and its antecedent
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
    "ASN": ("ND2", "CG"),
    "GLN": ("NE2", "CD"),
}

# Measured in the 5FQJ crystal; tests/test_variants.py asserts them.
CRYSTAL_ANGLES = {
    # (acceptor description) -> degrees
    "asp229_od2": 106.6,    # CG-OD2...N3, ASP229 is unmutated in 5FQJ
    "lig_n1": 126.6,        # N2-N1...NE2; the acceptor is a ligand atom, so
                            # this holds whichever residue donates to it
}
# Hybridisation values, used only where nothing measured supports a number.
CHEMISTRY_ANGLES = {
    "glu229_oe2": 120.0,    # sp2 carboxylate
    "lig_o11": 120.0,       # sp2 nitro oxygen
}

# A directional hydrogen bond measured at the acceptor lies in this range.
# Outside it, a "measured" ideal is more likely a modelling artefact than a
# target, so the script says so loudly rather than installing it silently.
ANGLE_SANE_MIN = 90.0
ANGLE_SANE_MAX = 180.0

# A seed whose required contact is further than this is not a motif at all --
# the residues and the ligand are not interacting. Scaffolding from it would
# ask RFdiffusion to reproduce a geometry in which nothing touches, so the
# target is refused rather than generated with a stand-in ideal. Far beyond any
# hydrogen bond, so it cannot fire on a merely poor seed.
SEED_UNUSABLE_DISTANCE = 10.0

# Which atoms RFD3 pins per motif residue: three per functional group, enough
# to fix its position and orientation without holding the backbone. That is the
# capability RFdiffusion lacks and RFD3 has, and the motif specs rely on it.
RFD3_FIXED_ATOMS = {
    "ASP": "OD1,CG,OD2",
    "GLU": "OE1,CD,OE2",
    "HIS": "CD2,CE1,NE2",
    "ASN": "OD1,CG,ND2",
    "GLN": "OE1,CD,NE2",
}

LIGAND_NAME = "6NT"
LIGAND_RESNAME = "6NT"


@dataclass
class BondPlan:
    """One hydrogen bond to emit, with its ideal angle and where that came from."""

    bond_id: str
    donor_site: str              # "ligand" | "protein"
    donor_atom: str
    donor_chain: str | None
    donor_resnum: int | None
    donor_antecedent: str | None
    acceptor_site: str
    acceptor_atom: str
    acceptor_chain: str | None
    acceptor_resnum: int | None
    acceptor_antecedent: str
    acceptor_key: str            # which entry in CRYSTAL/CHEMISTRY applies
    required: bool
    comment: str
    # filled in by measure()
    distance: float | None = None
    measured_angle: float | None = None
    ideal: float = 0.0
    source: str = "?"            # measured | crystal | chemistry
    note: str = ""


def read_motif(pdb: Path) -> dict[int, str]:
    residues: dict[int, str] = {}
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        residues[int(line[22:26])] = line[17:20].strip()
    return residues


def contig_for(positions: list[int]) -> str:
    """RFD3's contig: a total length range, not RFdiffusion segment specs.

    260 is the competition limit, and what both the organisers' example and the
    team's earlier runs used, so designs stay comparable. WHICH residues are
    kept is said by unindex / select_fixed_atoms, which src/stages/rfd3.py
    derives from motif.fixed and motif.fixed_atoms rather than from here.
    """
    return "260-260"


def _rfdiffusion_contigs_unused(positions: list[int]) -> str:
    """An RFdiffusion motif-scaffolding contig string.

    Scaffold segments between and around the fixed motif residues. The ranges
    give RFdiffusion freedom over total length while keeping the whole system
    inside the Boltz-2 token budget: the longest expansion is 143 residues + 12
    ligand atoms = 155 tokens, against max_tokens 400.

    UNVERIFIED. This is the script's construction, not a string taken from a
    run that worked -- the earlier runs used RFD3, whose contig syntax differs.
    Confirm it against the actual RFdiffusion build before a production run.
    """
    parts = ["20-40"]
    for index, position in enumerate(sorted(positions)):
        if index:
            parts.append("10-30")
        parts.append(f"A{position}-{position}")
    parts.append("20-40")
    return "/".join(parts)


def validate_residues(residues: dict[int, str], where: str = "motif") -> None:
    """Every position must carry a residue this script knows the chemistry of.

    Raises rather than guessing: a residue whose functional atom we cannot name
    would otherwise reach the scorer as an AtomResolutionError hours into a run.
    Lives here, not only in main(), so the error is the same from any caller.
    """
    if 229 not in residues or 291 not in residues:
        raise SystemExit(
            f"{where}: expected residues 229 and 291, found {sorted(residues)}"
        )
    tables = {229: BASE_ATOMS, 291: DONOR_ATOMS, 261: NITRO_ATOMS}
    for position, name in sorted(residues.items()):
        table = tables.get(position)
        if table is None:
            raise SystemExit(
                f"{where}: residue {name}{position} is at a position this script "
                f"does not handle (it knows {sorted(tables)}). Add it to the "
                "chemistry table in scripts/build_targets.py with its functional "
                "atom and antecedent."
            )
        if name not in table:
            raise SystemExit(
                f"{where}: residue {name}{position} is not in this script's "
                f"chemistry table for that position ({sorted(table)}). Add it "
                "with its functional atom and antecedent."
            )


def plan_bonds(residues: dict[int, str]) -> list[BondPlan]:
    validate_residues(residues)
    base_res = residues[229]
    donor_res = residues[291]
    base_atom, base_ante = BASE_ATOMS[base_res]
    donor_atom, donor_ante = DONOR_ATOMS[donor_res]

    plans = [
        BondPlan(
            bond_id=f"lig_n3_to_{base_res.lower()}229_{base_atom.lower()}",
            donor_site="ligand", donor_atom="N3",
            donor_chain=None, donor_resnum=None, donor_antecedent=None,
            acceptor_site="protein", acceptor_atom=base_atom,
            acceptor_chain="A", acceptor_resnum=229,
            acceptor_antecedent=base_ante,
            acceptor_key=f"{base_res.lower()}229_{base_atom.lower()}",
            required=True,
            comment=("The benzotriazole N3-H donates to the carboxylate. The donor\n"
                     "    # hydrogen is on the ligand and its position is fixed by the\n"
                     "    # aromatic ring."),
        ),
        BondPlan(
            bond_id=f"{donor_res.lower()}291_{donor_atom.lower()}_to_lig_n1",
            donor_site="protein", donor_atom=donor_atom,
            donor_chain="A", donor_resnum=291, donor_antecedent=donor_ante,
            acceptor_site="ligand", acceptor_atom="N1",
            acceptor_chain=None, acceptor_resnum=None,
            acceptor_antecedent="N2",
            acceptor_key="lig_n1",
            required=True,
            comment=(f"{donor_res}291 donates to the ligand N1. The acceptor is a ligand\n"
                     "    # atom, so its ideal direction is a property of the ligand and does\n"
                     "    # not depend on which residue donates to it."),
        ),
    ]

    if 261 in residues:
        nitro_res = residues[261]
        nitro_atom, nitro_ante = NITRO_ATOMS[nitro_res]
        plans.append(
            BondPlan(
                bond_id=f"{nitro_res.lower()}261_{nitro_atom.lower()}_to_lig_o11",
                donor_site="protein", donor_atom=nitro_atom,
                donor_chain="A", donor_resnum=261, donor_antecedent=nitro_ante,
                acceptor_site="ligand", acceptor_atom="O11",
                acceptor_chain=None, acceptor_resnum=None,
                acceptor_antecedent="NO1",
                acceptor_key="lig_o11",
                required=False,
                comment=(f"{nitro_res}261 donates to a nitro oxygen, stabilising the developing\n"
                         "    # negative charge. Scored but NOT gated: this contact is the thing\n"
                         "    # the 02-07 variants exist to test, so a design that fails to form\n"
                         "    # it should rank low rather than be thrown away. Set required: true\n"
                         "    # to gate on it instead."),
            )
        )
    return plans


def measure(plans: list[BondPlan], pdb: Path, clash: float, tolerated: float) -> None:
    """Fill in each plan's distance, measured angle, ideal and source.

    Uses the pipeline's own structure reader and angle function, so what is
    measured here is exactly what the scorer will measure later.
    """
    structure = read_structure(pdb, LIGAND_RESNAME)

    def coord(site: str, atom: str, chain: str | None, resnum: int | None):
        if site == "ligand":
            return structure.ligand.get(atom)
        return structure.atoms.get((chain, resnum, atom))

    for plan in plans:
        donor = coord(plan.donor_site, plan.donor_atom,
                      plan.donor_chain, plan.donor_resnum)
        acceptor = coord(plan.acceptor_site, plan.acceptor_atom,
                         plan.acceptor_chain, plan.acceptor_resnum)
        antecedent = coord(plan.acceptor_site, plan.acceptor_antecedent,
                           plan.acceptor_chain, plan.acceptor_resnum)

        if donor is None or acceptor is None or antecedent is None:
            missing = [
                name for name, value in (
                    (plan.donor_atom, donor),
                    (plan.acceptor_atom, acceptor),
                    (plan.acceptor_antecedent, antecedent),
                ) if value is None
            ]
            raise SystemExit(
                f"{pdb}: bond {plan.bond_id} names atom(s) {missing} that are not "
                "in the file. Either the PDB is missing a sidechain atom or the "
                "chemistry table in this script is wrong for that residue."
            )

        import numpy as np

        plan.distance = float(np.linalg.norm(donor - acceptor))
        plan.measured_angle = angle_between(antecedent, acceptor, donor)

        # "Plausible" means the seed's own contact would actually score
        # above zero: between the clash gate and `tolerated`, which is where
        # the distance term reaches 0. A 3.9 A contact sits inside the 4.5 A
        # fail gate but scores zero, so its angle is not a target either.
        plausible = clash < plan.distance < tolerated
        if plausible:
            plan.ideal = round(plan.measured_angle, 1)
            plan.source = "measured"
            notes = []
            # A directional H-bond at the acceptor sits well above 90 deg. Below
            # that the donor is approaching edge-on to the acceptor's
            # antecedent bond, which is not a geometry to aim for -- it usually
            # means the sidechain was moved without being repacked.
            if not ANGLE_SANE_MIN <= plan.measured_angle <= ANGLE_SANE_MAX:
                notes.append(
                    f"SUSPICIOUS: {plan.measured_angle:.1f} deg is outside "
                    f"{ANGLE_SANE_MIN}-{ANGLE_SANE_MAX} deg, which is not a "
                    "plausible acceptor-side H-bond direction. The distance is "
                    "fine, so the sidechain was probably moved without being "
                    "repacked. Check this refold before running on it"
                )
            crystal = CRYSTAL_ANGLES.get(plan.acceptor_key)
            if crystal is not None and abs(plan.ideal - crystal) > 5.0:
                notes.append(
                    f"differs from the 5FQJ crystal value {crystal} deg by "
                    f"{abs(plan.ideal - crystal):.1f} deg -- check the refold"
                )
            plan.note = "; ".join(notes)
        elif plan.acceptor_key in CRYSTAL_ANGLES:
            plan.ideal = CRYSTAL_ANGLES[plan.acceptor_key]
            plan.source = "crystal"
            plan.note = (
                f"seed distance {plan.distance:.3f} A is outside "
                f"{clash}-{tolerated} A, so the seed's own angle is not a target"
            )
        else:
            plan.ideal = CHEMISTRY_ANGLES[plan.acceptor_key]
            plan.source = "chemistry"
            plan.note = (
                f"seed distance {plan.distance:.3f} A is outside "
                f"{clash}-{tolerated} A and there is no crystal value for this "
                "acceptor, so nothing measured supports this ideal"
            )


def unusable_bonds(plans: list[BondPlan]) -> list[BondPlan]:
    """Required bonds whose seed geometry means the motif is not a motif.

    Distinct from a fallback ideal: a 1.7 A clash or a 3.9 A stretch is a bad
    contact that RFdiffusion can still be asked to improve, and the crystal or
    the chemistry supplies a target. A 25 A separation is not a contact, and
    nothing downstream can rescue it.
    """
    return [
        p for p in plans
        if p.required
        and p.distance is not None
        and p.distance > SEED_UNUSABLE_DISTANCE
    ]


def render(variant: str, label: str, pdb: Path, residues: dict[int, str],
           plans: list[BondPlan]) -> str:
    positions = sorted(residues)
    sources = {p.source for p in plans}
    if sources == {"measured"}:
        provenance = "Every ideal angle below was MEASURED in this seed."
    else:
        provenance = (
            "Some ideal angles below are NOT measured in this seed, because the "
            "seed's\n# own geometry for those bonds is not a plausible hydrogen "
            "bond. The `from:`\n# note on each one says where its value came "
            "from. Re-run\n# scripts/build_targets.py after replacing the seed "
            "with a repacked one."
        )

    lines = [
        f"# Variant {variant}: {label}",
        "#",
        f"#   base      A229  {residues[229]}   accepts from the benzotriazole N3-H",
        f"#   donor     A291  {residues[291]}   donates to the ligand N1",
    ]
    if 261 in residues:
        lines.append(f"#   nitro     A261  {residues[261]}   donates to the ligand nitro O11")
    else:
        lines.append("#   (no residue at 261 in this variant)")
    lines += [
        "#",
        "# GENERATED by scripts/build_targets.py -- edit that script, not this file.",
        "# Verify with: python scripts/build_targets.py --check",
        "#",
        f"# {provenance}",
        "",
        f'name: "{variant}_{label}"',
        "",
        "seeds:",
        "  # A motif-only PDB: the fixed residues plus the ligand, in 5FQJ numbering.",
        "  # Cycle 0 is motif scaffolding (no partial_T); the wiggle branch then",
        "  # re-noises the backbones that come out of it.",
        "  #",
        "  # The id is quoted because YAML reads a bare 01 as the integer 1, and the",
        "  # id ends up in design ids, filenames and the scaffold column.",
        f'  - id: "{variant}"',
        f"    pdb: {pdb.relative_to(ROOT).as_posix()}",
        f'    contig: "{contig_for(positions)}"   # RFD3 total length; 260 is the limit',
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
        lines.append(f"    - {{chain: A, resnum: {position}}}   # {residues[position]}")

    lines.append("  fixed_atoms:                 # atoms RFD3 pins, per residue")
    for position in positions:
        name = residues[position]
        atoms = RFD3_FIXED_ATOMS.get(name)
        if atoms:
            lines.append(f'    A{position}: "{atoms}"   # {name}')
        else:
            lines.append(f"    # A{position} ({name}): no selection; pinned whole")

    lines += ["", "hbonds:"]
    for plan in plans:
        lines.append(f"  - id: {plan.bond_id}")
        lines.append(f"    # {plan.comment}")
        if plan.donor_site == "ligand":
            lines.append(f"    donor:    {{site: ligand,  atom: {plan.donor_atom}}}")
        else:
            lines.append(
                f"    donor:    {{site: protein, chain: {plan.donor_chain}, "
                f"resnum: {plan.donor_resnum}, atom: {plan.donor_atom}}}"
            )
            lines.append(
                f"    donor_antecedent: {{site: protein, chain: {plan.donor_chain}, "
                f"resnum: {plan.donor_resnum}, atom: {plan.donor_antecedent}}}"
            )
        if plan.acceptor_site == "ligand":
            lines.append(f"    acceptor: {{site: ligand,  atom: {plan.acceptor_atom}}}")
            lines.append(
                f"    acceptor_antecedent: {{site: ligand, atom: {plan.acceptor_antecedent}}}"
            )
        else:
            lines.append(
                f"    acceptor: {{site: protein, chain: {plan.acceptor_chain}, "
                f"resnum: {plan.acceptor_resnum}, atom: {plan.acceptor_atom}}}"
            )
            lines.append(
                f"    acceptor_antecedent: {{site: protein, chain: {plan.acceptor_chain}, "
                f"resnum: {plan.acceptor_resnum}, atom: {plan.acceptor_antecedent}}}"
            )
        lines.append("    rotatable_donor: false")
        lines.append(f"    required: {str(plan.required).lower()}")
        lines.append(f"    acceptor_ideal_angle: {plan.ideal}")
        lines.append(
            f"    # from: {plan.source}; seed geometry is "
            f"{plan.distance:.3f} A at {plan.measured_angle:.1f} deg"
        )
        if plan.note:
            lines.append(f"    # note: {plan.note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def discover(inputs_name: str | None = None) -> list[tuple[Path, Path]]:
    """One motif PDB per variant folder.

    A folder accumulates several `inputs/<subdir>/` as new versions arrive, and
    nothing is deleted. The MOST RECENTLY MODIFIED subdirectory wins, because
    "the one I just dropped in" is what anyone adding a folder means --
    lexicographic order would quietly prefer `5fqj-6cf4761` over a new
    `01_refold`. The choice is always printed, never silent.

    `--inputs NAME` overrides it and must then exist in every variant folder,
    which is the predictable option when the new files land in a consistently
    named directory.
    """
    found: list[tuple[Path, Path]] = []
    # Two or more digits, so a tenth variant is picked up rather than silently
    # skipped. `0[1-9]` would have missed data/10 onwards, and a commit on main
    # already carried the message "created 08-14".
    folders = sorted(
        d for d in DATA.glob("*")
        if d.is_dir() and d.name.isdigit() and len(d.name) >= 2
    )
    for folder in folders:
        candidates = [d for d in (folder / "inputs").glob("*") if d.is_dir()]

        # A PDB sitting directly in inputs/, with no subdirectory, is a valid
        # layout and is what one push actually used. Treat inputs/ itself as a
        # candidate so the layout can change without this script caring.
        if list((folder / "inputs").glob("*.pdb")):
            candidates.append(folder / "inputs")

        if not candidates:
            print(f"warning: {folder}/inputs holds no *.pdb and no "
                  "subdirectories, skipping", file=sys.stderr)
            continue

        if inputs_name:
            if inputs_name == ".":
                chosen_dir = folder / "inputs"
                if not list(chosen_dir.glob("*.pdb")):
                    raise SystemExit(
                        f"{chosen_dir} holds no *.pdb directly (--inputs .)"
                    )
            else:
                chosen = [d for d in candidates if d.name == inputs_name]
                if not chosen:
                    raise SystemExit(
                        f"{folder}/inputs has no subdirectory named "
                        f"{inputs_name!r}; it has "
                        f"{sorted(d.name for d in candidates if d.is_dir())}"
                    )
                chosen_dir = chosen[0]
        elif len(candidates) == 1:
            chosen_dir = candidates[0]
        else:
            # REFUSE to guess. This used to pick the newest by mtime, which
            # silently chose the wrong seed the first time it mattered: a git
            # checkout gives every file nearly the same mtime, so some variants
            # got the crystal motif and others got a full Boltz prediction with
            # the ligand in the wrong pocket. Nothing in the output said so.
            #
            # A wrong seed is not a recoverable error -- every design in the run
            # is scaffolded around it -- so an explicit choice is worth the
            # inconvenience.
            names = sorted(
                "." if d == folder / "inputs" else d.name for d in candidates
            )
            listing = "\n".join(f"    {n}" for n in names)
            raise SystemExit(
                f"{folder}/inputs has {len(candidates)} candidate seed "
                "locations and this script will not guess between them:\n"
                f"{listing}\n"
                "\nPass one explicitly, e.g.:\n"
                f"    python scripts/build_targets.py --inputs {names[-1]}\n"
                "\nUse --inputs . for a PDB sitting directly in inputs/.\n"
                "Every variant folder must then contain that name."
            )

        pdbs = sorted(chosen_dir.glob("*.pdb"))
        if not pdbs:
            raise SystemExit(
                f"{chosen_dir} holds no *.pdb. It is the newest inputs "
                f"subdirectory in {folder}, so this is probably the half-copied "
                "new input rather than something to skip. Pass --inputs to "
                "choose a different one explicitly."
            )
        if len(pdbs) > 1:
            raise SystemExit(
                f"{chosen_dir} holds {len(pdbs)} PDBs and one motif per variant "
                f"is expected: {[p.name for p in pdbs]}. Keep one, or put the "
                "others in their own inputs subdirectory."
            )
        if len(candidates) > 1:
            others = ", ".join(
                sorted(
                    "." if d == folder / "inputs" else d.name
                    for d in candidates if d != chosen_dir
                )
            )
            label = "." if chosen_dir == folder / "inputs" else chosen_dir.name
            print(f"  {folder.name}: using inputs/{label}  (not: {others})")
        found.append((folder, pdbs[0]))
    if not found:
        raise SystemExit(f"no variant folders found under {DATA}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any target file is out of date")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print every measurement, not just the summary")
    parser.add_argument("--inputs", metavar="NAME",
                        help="use data/0N/inputs/NAME for every variant instead "
                             "of the most recently modified subdirectory")
    args = parser.parse_args()

    distance = load_pipeline(PIPELINE).scoring.distance
    TARGETS.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    rows: list[tuple[str, BondPlan]] = []
    unusable: list[tuple[str, Path, list[BondPlan]]] = []

    for folder, pdb in discover(args.inputs):
        label = pdb.stem[3:].strip("_") or "motif"
        residues = read_motif(pdb)
        validate_residues(residues, where=str(pdb))
        plans = plan_bonds(residues)
        measure(plans, pdb, distance.clash, distance.tolerated)
        rows.extend((folder.name, p) for p in plans)

        broken = unusable_bonds(plans)
        if broken:
            unusable.append((folder.name, pdb, broken))
            continue

        out = TARGETS / f"{folder.name}_{label}.yaml"
        text = render(folder.name, label, pdb, residues, plans)
        if args.check:
            if (out.read_text(encoding="utf-8") if out.exists() else "") != text:
                stale.append(out.name)
        else:
            out.write_text(text, encoding="utf-8")

    # ---- the summary table ----
    if not args.check:
        print()
        header = (f"{'variant':<8} {'bond':<30} {'dist A':>7} {'angle':>7} "
                  f"{'ideal':>7}  {'from':<10} ")
        print(header)
        print("-" * (len(header) + 20))
        for variant, plan in rows:
            flag = "" if plan.source == "measured" else "  <-- not measured"
            print(f"{variant:<8} {plan.bond_id:<30} {plan.distance:>7.3f} "
                  f"{plan.measured_angle:>7.1f} {plan.ideal:>7.1f}  "
                  f"{plan.source:<10}{flag}")

        suspicious = [(v, p) for v, p in rows if "SUSPICIOUS" in p.note]
        if suspicious:
            print()
            print(f"  !! {len(suspicious)} measured angle(s) look wrong:")
            for variant, plan in suspicious:
                print(f"     {variant} {plan.bond_id}: {plan.measured_angle:.1f} deg "
                      f"at {plan.distance:.3f} A")
            print("     A plausible distance with an implausible direction "
                  "usually means the")
            print("     sidechain was moved without being repacked. Check these "
                  "before running.")

        counts: dict[str, int] = {}
        for _, plan in rows:
            counts[plan.source] = counts.get(plan.source, 0) + 1
        total = len(rows)
        print()
        print(f"  {total} bonds across {len({v for v, _ in rows})} variants: "
              + ", ".join(f"{n} {source}" for source, n in sorted(counts.items())))

        not_measured = total - counts.get("measured", 0)
        if not_measured:
            print()
            print(f"  {not_measured} of {total} ideal angles are NOT measured in "
                  "their seed, because")
            print("  the seed's geometry for those bonds is not a plausible "
                  f"hydrogen bond ({distance.clash}-{distance.tolerated} A).")
            print("  That is expected for the shipped motif files, whose mutations "
                  "were modelled")
            print("  in without repacking. Replace a seed with a repacked one and "
                  "re-run this")
            print("  script: the affected bonds will switch to `measured` on their own.")
            if args.verbose:
                print()
                for variant, plan in rows:
                    if plan.source != "measured":
                        print(f"    {variant} {plan.bond_id}: {plan.note}")
        else:
            print()
            print("  All ideal angles are measured in their seeds. Nothing is "
                  "standing in for")
            print("  a number nobody checked.")
        print()

    if unusable:
        print()
        print("=" * 72, file=sys.stderr)
        print(f"REFUSED {len(unusable)} variant(s): the seed is not a motif",
              file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        for variant, pdb, broken in unusable:
            print(file=sys.stderr)
            print(f"  {variant}  {pdb}", file=sys.stderr)
            for plan in broken:
                print(f"    {plan.bond_id}: {plan.distance:.1f} A "
                      f"between {plan.donor_atom} and {plan.acceptor_atom}",
                      file=sys.stderr)
        print(f"""
A required contact further than {SEED_UNUSABLE_DISTANCE} A means the motif
residues and the ligand are not interacting in that file. No ideal angle can
fix it: RFdiffusion would be asked to scaffold a constellation in which nothing
touches, and every design from it would be meaningless.

No target file was written for those variants, so the previous one -- if any --
is still in place and still points at this seed. Check the seed before running.

Two things that produce exactly this, worth ruling out in order:

  1. The protein and the ligand are in different coordinate frames. If the
     protein coordinates came from a prediction and the ligand was carried over
     from the crystal (or vice versa) without superposing the two, the file is
     internally inconsistent. The B-factor columns are a clue: a per-residue
     uniform value is a predicted pLDDT, a per-atom varying value is usually
     crystallographic.

  2. The prediction genuinely lost the ligand. This is the failure RUNLOG.md
     section 1 records -- a motif held to 0.013 A at design time and the ligand
     11.18 A away after re-prediction. If that is what happened, the file is a
     faithful extraction of a bad prediction, and the news is about the
     approach rather than about the file.

Either way the fix is upstream of this repo.""", file=sys.stderr)

    if args.check:
        if stale:
            print("out of date (run `python scripts/build_targets.py`): "
                  + ", ".join(stale), file=sys.stderr)
            return 1
        if unusable:
            return 1
        print("all target files are current")
    return 1 if unusable else 0


if __name__ == "__main__":
    raise SystemExit(main())

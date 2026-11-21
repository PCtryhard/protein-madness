"""Relax a full variant complex and export its mapped 5FQJ motif with 6NT."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CONTACTS = {
    203: (229, {"D": "OD2", "E": "OE2"}, "N3"),
    233: (261, {"H": "NE2", "N": "ND2"}, "O11"),
    263: (291, {"H": "NE2", "N": "ND2", "Q": "NE2"}, "N1"),
}


def unique_file(paths, label):
    files = sorted(set(paths))
    if len(files) != 1:
        raise ValueError(f"expected one {label}, found {len(files)}: {files}")
    return files[0]


def map_motif(model, chain, fasta):
    import gemmi
    from Bio import SeqIO
    from Bio.Align import PairwiseAligner

    reference = str(SeqIO.read(fasta, "fasta").seq).upper()
    if len(reference) != 268:
        raise ValueError(f"expected a 268-residue 5FQJ variant FASTA: {fasta}")
    residues = [res for res in model[chain] if res.find_atom("CA", "*")]
    query = "".join(gemmi.find_tabulated_residue(res.name).one_letter_code
                    for res in residues)
    aligner = PairwiseAligner(match_score=2, mismatch_score=-1,
                             open_gap_score=-8, extend_gap_score=-0.5)
    alignments = iter(aligner.align(reference, query))
    positions = [p for p in CONTACTS if p != 233 or reference[p - 1] != "V"]

    def indices(alignment):
        return {int(a) + 1: int(b) for a, b in zip(*alignment.indices)
                if int(a) + 1 in positions and b >= 0}

    mapping = indices(next(alignments))
    for i, alternative in enumerate(alignments, 1):
        if indices(alternative) != mapping or i >= 100:
            raise ValueError("ambiguous motif alignment; check the CIF and variant FASTA")

    motif = []
    for position in positions:
        author_id, atom_names, ligand_atom = CONTACTS[position]
        aa = reference[position - 1]
        if position not in mapping or query[mapping[position]] != aa or aa not in atom_names:
            raise ValueError(f"CIF does not match the expected motif at FASTA position {position} ({aa})")
        residue = residues[mapping[position]]
        if residue.seqid.icode.strip():
            raise ValueError(f"motif residue {residue.seqid} has an insertion code")
        motif.append((residue.seqid.num, author_id, residue.name, atom_names[aa], ligand_atom))
    return motif


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", help="variant number (01–13) or path to its CIF/PDB")
    parser.add_argument("--cif", type=Path, help="override structure discovery for a numbered run")
    parser.add_argument("--fasta", type=Path, help="override variant FASTA discovery")
    parser.add_argument("--params", type=Path, default=ROOT / "data/ligands/6NT/6NT.params")
    parser.add_argument("--out", type=Path, help="override motif output path")
    parser.add_argument("--chain", default="A", help="protein chain in the input structure")
    parser.add_argument("--map-only", action="store_true", help="check motif mapping without PyRosetta")
    args = parser.parse_args()
    try:
        if args.run.isdecimal() and 1 <= int(args.run) <= 13:
            run = ROOT / "data" / f"{int(args.run):02d}"
            folders = (run, run / "inputs")
            structures = [p for folder in folders for p in folder.glob("*.cif")]
            if not structures:
                structures = [p for folder in folders for p in folder.glob(f"{run.name}_*.pdb")
                              if not p.name.endswith("_fastrelax_motif.pdb")]
            cif = args.cif or unique_file(structures, "input structure")
        else:
            cif = Path(args.run).resolve()
            run = next((p for p in cif.parents if p.parent == ROOT / "data"), None)
            if run is None and (args.fasta is None or args.out is None):
                raise ValueError("provide --fasta and --out for a CIF outside a run folder")
        fasta = args.fasta or unique_file(
            (p for p in run.rglob(f"{run.name}_*") if p.suffix.lower() in {".fa", ".fasta"}),
            "numbered variant FASTA",
        )
        out = args.out or run / "inputs" / f"{run.name}_fastrelax_motif.pdb"
        import gemmi

        structure = gemmi.read_structure(str(cif))
        if len(structure) != 1:
            raise ValueError("expected exactly one CIF model")
        motif = map_motif(structure[0], args.chain, fasta)
        ligands = [(chain.name, res.seqid.num) for chain in structure[0]
                   for res in chain if res.name == "6NT"]
        if len(ligands) != 1:
            raise ValueError("expected exactly one 6NT residue in the CIF")
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Structure: {cif.resolve()}")
    for source, target, aa, _, _ in motif:
        print(f"Input {args.chain}{source} {aa} -> motif A{target}")
    if args.map_only:
        return
    if not args.params.is_file():
        parser.error(f"missing 6NT params: {args.params}")
    if out.suffix.lower() != ".pdb":
        parser.error(f"--out must end in .pdb: {out}")
    if out.exists():
        parser.error(f"output already exists: {out}; pass --out with a new filename")
    try:
        import pyrosetta
    except ImportError as exc:
        parser.error(f"PyRosetta is required for relaxation: {exc}")
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from pyrosetta.rosetta.protocols.relax import FastRelax

    pyrosetta.init(extra_options=f"-in:file:extra_res_fa {args.params.resolve()} "
                                 "-constant_seed -jran 1")
    with TemporaryDirectory() as tmp:
        temporary = Path(tmp) / "complex.pdb"
        structure.write_pdb(str(temporary))
        pose = pyrosetta.pose_from_pdb(str(temporary))
    info = pose.pdb_info()
    mapped = [(info.pdb2pose(args.chain, source), aa, atom, ligand_atom)
              for source, _, aa, atom, ligand_atom in motif]
    ligand = info.pdb2pose(*ligands[0])
    for index, expected in [(i, aa) for i, aa, _, _ in mapped] + [(ligand, "6NT")]:
        if not index or pose.residue(index).name3() != expected:
            parser.error(f"Rosetta did not load mapped residue {expected} (pose index {index})")

    ligand_atoms = pose.residue(ligand)
    if ligand_atoms.nheavyatoms() != 12 or not all(
        ligand_atoms.has(atom) for _, _, _, atom in mapped
    ):
        parser.error("check that the 6NT params use the CIF's CCD atom names")
    for index, aa, atom, _ in mapped:
        if not pose.residue(index).has(atom):
            parser.error(f"missing {atom} on mapped {aa} at pose index {index}")

    def contacts():
        return [(pose.residue(i).xyz(atom) - pose.residue(ligand).xyz(lig_atom)).norm()
                for i, _, atom, lig_atom in mapped]

    before = contacts()
    movable = {index for index, _, _, _ in mapped}
    for index in range(1, pose.total_residue() + 1):
        residue = pose.residue(index)
        if residue.is_protein() and any(
            (residue.xyz(i) - ligand_atoms.xyz(j)).norm_squared() <= 64
            for i in range(1, residue.nheavyatoms() + 1)
            for j in range(1, ligand_atoms.nheavyatoms() + 1)
        ):
            movable.add(index)

    moves = MoveMap()
    moves.set_bb(False)
    moves.set_jump(False)
    moves.set_chi(False)
    for index in movable:
        moves.set_chi(index, True)

    frozen = ",".join(str(i) for i in range(1, pose.total_residue() + 1)
                      if i not in movable)
    task = TaskFactory()
    task.push_back(operation.RestrictToRepacking())
    task.push_back(operation.OperateOnResidueSubset(
        operation.PreventRepackingRLT(), ResidueIndexSelector(frozen)))
    relax = FastRelax(pyrosetta.get_fa_scorefxn(), 1)
    relax.set_movemap(moves)
    relax.set_task_factory(task)
    relax.apply(pose)
    after = contacts()

    ids = {(args.chain, source): ("A", target) for source, target, _, _, _ in motif}
    ids[ligands[0]] = ("B", 1295)
    lines = []
    with TemporaryDirectory() as tmp:
        relaxed = Path(tmp) / "relaxed.pdb"
        pose.dump_pdb(str(relaxed))
        records = relaxed.read_text().splitlines()
    for line in records:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key = (line[21], int(line[22:26]))
        if key in ids and line[76:78].strip() not in {"H", "D"}:
            chain, number = ids[key]
            lines.append(line[:21] + f"{chain}{number:4d} " + line[27:])
    if sum(line.startswith("ATOM  ") and line[12:16].strip() == "CA" for line in lines) != len(motif):
        raise RuntimeError("output is missing mapped protein residues")
    if sum(line.startswith("HETATM") for line in lines) != 12:
        raise RuntimeError("expected exactly 12 ligand heavy atoms in output")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write("\n".join(lines + ["END"]) + "\n")
    for (_, target, aa, atom, lig_atom), old, new in zip(motif, before, after):
        print(f"{aa} A{target} {atom}–6NT {lig_atom}: {old:.2f} Å → {new:.2f} Å")
    print(f"Wrote {out} ({len(motif)} motif residues and 6NT B1295)")


if __name__ == "__main__":
    main()

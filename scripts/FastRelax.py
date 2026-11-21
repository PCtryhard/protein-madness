"""Relax the variant 01 complex and export its Asp/His motif with 6NT."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cif", type=Path, help="full protein–6NT mmCIF")
    parser.add_argument("--params", required=True, type=Path, help="Rosetta 6NT.params")
    parser.add_argument("--out", required=True, type=Path, help="output motif PDB")
    parser.add_argument("--chain", default="A", help="protein chain in the CIF")
    parser.add_argument("--asp", type=int, default=202, help="Asp residue ID in the CIF")
    parser.add_argument("--his", type=int, default=262, help="His residue ID in the CIF")
    args = parser.parse_args()
    for path in (args.cif, args.params):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.out.exists() or args.out.suffix.lower() != ".pdb":
        parser.error("--out must name a new .pdb file")

    try:
        import gemmi
        import pyrosetta
    except ImportError as exc:
        parser.error(f"run in an environment with Gemmi and PyRosetta: {exc}")
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from pyrosetta.rosetta.protocols.relax import FastRelax

    structure = gemmi.read_structure(str(args.cif))
    if len(structure) != 1:
        parser.error("expected exactly one CIF model")
    ligands = [(chain.name, res.seqid.num) for chain in structure[0]
               for res in chain if res.name == "6NT"]
    if len(ligands) != 1:
        parser.error("expected exactly one 6NT residue in the CIF")

    pyrosetta.init(extra_options=f"-in:file:extra_res_fa {args.params.resolve()} "
                                 "-constant_seed -jran 1")
    with TemporaryDirectory() as tmp:
        temporary = Path(tmp) / "complex.pdb"
        structure.write_pdb(str(temporary))
        pose = pyrosetta.pose_from_pdb(str(temporary))
    info = pose.pdb_info()
    asp = info.pdb2pose(args.chain, args.asp)
    his = info.pdb2pose(args.chain, args.his)
    ligand = info.pdb2pose(*ligands[0])
    for index, expected in ((asp, "ASP"), (his, "HIS"), (ligand, "6NT")):
        if not index or pose.residue(index).name3() != expected:
            parser.error(f"could not load {expected} at the specified residue ID")

    ligand_atoms = pose.residue(ligand)
    if ligand_atoms.nheavyatoms() != 12 or not all(ligand_atoms.has(n) for n in ("N1", "N3")):
        parser.error("check that the 6NT params use the CIF's CCD atom names")

    def contacts():
        return (
            (pose.residue(asp).xyz("OD2") - pose.residue(ligand).xyz("N3")).norm(),
            (pose.residue(his).xyz("NE2") - pose.residue(ligand).xyz("N1")).norm(),
        )

    before = contacts()
    movable = {asp, his}
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

    ids = {(args.chain, args.asp): ("A", 229), (args.chain, args.his): ("A", 291),
           ligands[0]: ("B", 1295)}
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
    if sum(line.startswith("ATOM  ") and line[12:16].strip() == "CA" for line in lines) != 2:
        raise RuntimeError("expected two protein residues in output")
    if sum(line.startswith("HETATM") for line in lines) != 12:
        raise RuntimeError("expected exactly 12 ligand heavy atoms in output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write("\n".join(lines + ["END"]) + "\n")
    for label, old, new in zip(("Asp OD2–6NT N3", "His NE2–6NT N1"), before, after):
        print(f"{label}: {old:.2f} Å → {new:.2f} Å")
    print(f"Wrote {args.out} (A229, A291, and 6NT B1295)")


if __name__ == "__main__":
    main()

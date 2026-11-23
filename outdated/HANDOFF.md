# When the refolded motif files arrive

The shipped motif files have deliberate steric clashes: every mutated or added
residue in variants 02–07 was modelled in without repacking, so its atoms land
on top of the ligand. The refolded replacements change three commands' worth of
work and nothing else.

## The whole procedure

**1. Drop the new PDBs in, one per variant, in a new subdirectory.**

```
data/01/inputs/refold/01_his_asp_.pdb
data/02/inputs/refold/02_his_asp_his.pdb
...
data/07/inputs/refold/07_gln_glu_his.pdb
```

The subdirectory name does not matter — the generator takes the **most recently
modified** one in each `data/0N/inputs/`, and prints which it chose. Nothing is
deleted, so the originals stay for comparison. If you want to be explicit:

```powershell
python scripts/build_targets.py --inputs refold
```

Requirements on the files themselves: chain A, 5FQJ residue numbering (229,
261, 291), the `6NT` ligand present as `HETATM`, and the sidechain atoms the
chemistry needs — `OD2`/`CG` for ASP, `OE2`/`CD` for GLU, `NE2`/`CD2` for HIS,
`ND2`/`CG` for ASN, `NE2`/`CD` for GLN. One PDB per variant folder. A missing
atom is a hard error naming the atom, not a silent skip.

**2. Regenerate the targets.**

```powershell
python scripts/build_targets.py
```

This is where the work happens. For each bond it measures the seed's own
geometry and decides where the ideal angle comes from:

| it prints | meaning |
| --- | --- |
| `measured` | the seed's contact is a plausible H-bond (2.2–3.5 Å), so **its own angle becomes the target** — that is the geometry RFdiffusion will be asked to reproduce |
| `crystal` | the seed's contact is not plausible, but this acceptor exists unmutated in 5FQJ, so the crystal value stands in |
| `chemistry` | neither, so a hybridisation value stands in, and it warns — nothing measured supports that number |

Right now it reports **9 measured, 4 crystal, 7 chemistry**. After a good
refold it should report 20 measured and print:

> All ideal angles are measured in their seeds. Nothing is standing in for a
> number nobody checked.

If it still warns, the new files still clash — worth knowing before booking a
GPU.

It also flags a plausible distance with an implausible direction:

```
  !! 1 measured angle(s) look wrong:
     05 lig_n3_to_glu229_oe2: 77.5 deg at 2.800 A
```

An acceptor-side angle below 90° means the donor approaches edge-on. That
usually means a sidechain was translated without being repacked, and it would
otherwise install a nonsense target silently.

**3. Check it before spending GPU time.**

```powershell
python -m pytest -q
```

```powershell
python -m src.cli measure --target config/targets/05_his_glu_his.yaml --pdb data/05/inputs/refold/05_his_glu_his.pdb
```

`measure` prints every bond with its resolved atoms, distance, angle and
normalised terms. A perfect seed scores 1.000 on both terms by construction,
since its own geometry became the target.

One test will fail on purpose: `test_the_modelled_mutations_really_do_clash`
asserts the *old* clash distances (1.677 Å, 1.820 Å). That failure is the
signal that the refold landed. Update those numbers to the new ones — the test
exists to make the change visible, not to block it.

## Then run it

Seed-only is the default, 10 backbones × 8 sequences per variant:

```powershell
bash cluster/aws/run_all.sh 01 02
```

That is roughly an hour per variant. It prints which profile it is using and
whether the run is seed-only or a full search before it starts anything. See
[cluster/README.md](cluster/README.md) for Euler, and the timing table in
[config/pipeline.yaml](config/pipeline.yaml) for where the estimate comes from.

## What does not need touching

- `config/pipeline.yaml` — the only pipeline config, and nothing in it is
  target-specific
- `config/targets/*.yaml` — generated; edit `scripts/build_targets.py` instead
- any code in `src/` — the chemistry table lives in `scripts/build_targets.py`,
  and a residue type it does not know is a hard error telling you to add it
- the contig strings — still `20-40/A229-229/…`, still **unverified against the
  real RFdiffusion build**. That check is independent of the refold and is
  still the highest-risk item before a production run.

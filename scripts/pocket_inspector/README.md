# Kemp Pocket Inspector

A browsable 3D view of the rescored designs. One command builds it from
`data/rescored/`; you open the result in a browser.

```bash
python scripts/pocket_inspector/build_inspector.py
open inspector/pocket_inspector.html
```

No dependencies beyond the Python standard library, no GPU, no network while
building. Nothing is refolded or re-scored — this reads committed data and
draws it.

## What you get

One entry per rescored design, 232 of them. The viewport shows the Boltz
prediction with the ligand in gold, the catalytic residues in red, the rest of
the pocket within 5 Å in cyan, and **every catalytic bond as a dashed line
labelled with its distance in angstroms**.

Those distances are measured at build time from the deposited coordinates. The
atom pair is taken from the re-rank's own `catalytic_residues` mapping, so
`lig_n3_to_asp229_od2=ASP232` is measured on residue 232 of that design rather
than on a guessed position. All 671 bonds across the 232 designs resolve.

Alongside the viewport:

- **Rank by** — the geometry re-rank or the shipped pipeline benchmark. Every
  design shows both scores, both ranks, and how far it moved.
- **Adjustments** — filter by how many `mutate`/`wiggle` steps are actually in
  a design's ancestry. Not the cycle number: the search re-expands the seed
  every cycle, so 40 of the 232 have a shorter lineage than their cycle.
- Per-bond out-of-plane offset and group angle, the five re-rank terms, the
  catalytic and site residues, and the original `design_id` and run.

## Two output shapes

| flag | result | how to open |
| --- | --- | --- |
| `--single-file` (default) | one `.html`, about 40 MB, every structure embedded | double click it |
| `--split` | `index.html` + `designs.json` + one bundle per variant | needs a server, see below |

Browsers refuse `fetch()` from `file://`, so the split build needs one:

```bash
cd inspector && python3 -m http.server 8000
# then open http://localhost:8000/
```

The single file is the one to send someone. The split build is better if you
want to edit the page, since you are not reloading 40 MB each time.

Either way the page loads 3Dmol.js from a CDN, so the browser needs network
access the first time. Everything else is local.

## Other options

```bash
# a checkout somewhere else
python scripts/pocket_inspector/build_inspector.py --rescored /path/to/data/rescored

# write it elsewhere
python scripts/pocket_inspector/build_inspector.py --out ~/Desktop/inspector
```

## Do not commit the output

`inspector/` is a copy of structures that are already in `data/rescored/cif/`.
Write it outside the repo, or add it to `.gitignore`. A 40 MB generated page in
git history is 40 MB nobody can remove later.

## Files here

| file | what it is |
| --- | --- |
| `build_inspector.py` | the builder: reads the tables, measures the bonds, writes the page |
| `template.html` | the page itself, with a single hook where the data is inlined |
| `README.md` | this |

`template.html` is a working page on its own — it falls back to fetching
`designs.json` and `b/v*.json` when no data has been inlined, which is exactly
what `--split` produces.

## What the page cannot tell you

Every number comes from a single Boltz prediction. `scripts/noise_control.py`
measured seed-to-seed spread on one fixed sequence at ±4 Å and ±28°, and both
production batches ran with `eval_replicates: 1`. Differences smaller than that
are not resolvable, and sidechain placement in a designed pocket is where a
structure predictor is least reliable. Treat the ranking as a way to find
designs worth looking at, not as a measurement.

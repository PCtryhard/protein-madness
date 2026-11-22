# Kemp Pocket Inspector

A browsable 3D view of the rescored designs. One command builds it from
`data/rescored/`; you open the result in a browser.

```bash
python3 scripts/pocket_inspector/build_inspector.py --online
open inspector/pocket_inspector_online.html
```

That gives a 28 kB page that pulls the structures from this repository as you
browse. Send it to someone and it works on their machine with no checkout, no
build and no copy of the data. Everything else below is for working offline.

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

## Three output shapes

| flag | result | how to open |
| --- | --- | --- |
| `--online` | one `.html`, 28 kB, fetches this repo's files as you browse | double click it; needs internet |
| `--single-file` (default) | one `.html`, about 40 MB, every structure embedded | double click it; works offline |
| `--split` | `index.html` + `designs.json` + one bundle per variant | needs a server, see below |

**`--online` is the one to share.** It reads `manifest.csv`,
`rerank_geometry.csv` and `cif/*.cif` straight from
`raw.githubusercontent.com`, which serves `access-control-allow-origin: *`, so
any browser may fetch them. The bond distances are then measured in the
browser from those coordinates, the same arithmetic the offline builds do at
build time. It also follows the repository: re-run the re-rank, push, and
every copy of the page shows the new numbers without being rebuilt.

Two limits worth stating plainly. It needs the viewer to be online. And **it
only works while this repository is public** — a private repo's raw URLs
require a token, and a token must never be embedded in a page other people
open. If the repo goes private, hand people the `--single-file` build instead.

Point it somewhere else with `--base`, for any static host that allows
cross-origin reads:

```bash
python3 scripts/pocket_inspector/build_inspector.py --online \
  --base https://example.org/rescored/
```

Browsers refuse `fetch()` from `file://`, so the split build needs one:

```bash
cd inspector && python3 -m http.server 8000
# then open http://localhost:8000/
```

The single file is the one to send someone who has no internet, or to keep as
a frozen snapshot of today's numbers. The split build is better if you want to
edit the page, since you are not reloading 40 MB each time.

Every build loads 3Dmol.js from a CDN, so the browser needs network access the
first time even for the offline ones.

## Other options

```bash
# a checkout somewhere else
python3 scripts/pocket_inspector/build_inspector.py --rescored /path/to/data/rescored

# write it elsewhere
python3 scripts/pocket_inspector/build_inspector.py --out ~/Desktop/inspector
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

`template.html` is a working page on its own. It has two hooks the builder
fills in, `/*__DATA__*/` and `/*__REMOTE__*/`, and with neither filled it falls
back to fetching sibling `designs.json` and `b/v*.json` — which is exactly what
`--split` produces.

## What the page cannot tell you

Every number comes from a single Boltz prediction. `scripts/noise_control.py`
measured seed-to-seed spread on one fixed sequence at ±4 Å and ±28°, and both
production batches ran with `eval_replicates: 1`. Differences smaller than that
are not resolvable, and sidechain placement in a designed pocket is where a
structure predictor is least reliable. Treat the ranking as a way to find
designs worth looking at, not as a measurement.

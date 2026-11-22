# Rescored designs (round 2, geometry re-rank)

All 232 scored designs that have a committed structure, re-scored on catalytic
geometry, with their predicted complexes copied here under a traceable name.

Produced from `scripts/rerank_geometry.py` at commit `5177aeb`, patched as
described below, run with `--min-score 0 --include-legacy`. Nothing was
refolded: this is a pure re-score of structures already in the repo, so it
costs no GPU and is reproducible.

## Contents

| path | what it is |
| --- | --- |
| `cif/` | 232 predicted complexes, renamed (scheme below) |
| `manifest.csv` | one row per file: new name, original `design_id`, run, lineage, scores |
| `rerank_geometry.csv` | the re-rank table, 232 rows |
| `rerank_geometry.json` | same, as JSON |
| `rerank_round2.html` | standalone browsable leaderboard, opens offline |
| `rerank_legacy_layout.patch` | the one-function fix the run depended on, NOT applied to `scripts/` |

## Naming scheme

    XX_yyyyy_zzzzzz.cif

- `XX` — seed variant, `01` to `13`.
- `yyyyy` — one character per adjustment in the design's ancestry, oldest
  first: `m` for mutate, `w` for wiggle. So `mwm` is a design that was
  mutated, then wiggled, then mutated again. A cycle-0 seed had no
  adjustment and is written `s`.
- `zzzzzz` — the six hex digits the pipeline puts at the end of every
  `design_id`.

Examples:

    01_s_95cdcf.cif     variant 01, unadjusted seed
    01_w_865ea3.cif     variant 01, one wiggle
    02_mwm_cce984.cif   variant 02, mutate then wiggle then mutate
    08_wmm_25a996.cif   variant 08, wiggle then mutate then mutate

All 232 names are unique, and the hex ids are globally unique across the whole
set, so `zzzzzz` alone identifies a design.

## The lineage is the parent chain, not the cycle number

Reconstructed by walking `parent_id` through each run's `designs.csv` and
recording the `branch` of every non-seed ancestor.

**The number of adjustments is not the cycle number.** The search re-expands
the seed at every cycle, so a design labelled `c03` may descend from the seed
through only one or two adjustments. Of the 232 designs here, 40 have a
lineage shorter than their cycle number:

| cycle | adjustments | designs |
| --- | --- | --- |
| 0 | 0 | 90 |
| 1 | 1 | 76 |
| 2 | 1 | 14 |
| 2 | 2 | 19 |
| 3 | 2 | 26 |
| 3 | 3 | 7 |

The filename records the adjustments actually performed. Naming from the cycle
number instead would claim adjustments that never happened for those 40.
`search_cycle` and `adjustments` are both columns in `manifest.csv` if you need
to tell them apart.

## The patch this run depended on

The first re-rank pass skipped every `fastrelax-*` run, which removed 29
designs including all 12 of variant 12's hits. The reason was real but the
remedy was wrong.

The two run families write different columns:

| | `fastrelax-*` | `overnight-*`, `depth-*` |
| --- | --- | --- |
| weights | conf 0.70 / geom 0.30 | conf 0.50 / geom 0.50 |
| geometry split | distance 0.60 / angle 0.40 | catalytic 0.50 / non_catalytic 0.20 / angle 0.30 |
| columns | `term_distance`, `term_angle` | `term_catalytic`, `term_non_catalytic`, `term_angle` |

`old_geometry()` looked only for the newer column names. On a `fastrelax` row
it matched `term_angle` alone, renormalised over that single weight, and
returned the angle as the entire geometry composite, silently discarding the
distance. Running `--include-legacy` without the patch would therefore have
inflated those designs: variant 12's best would have taken `hbond = 0.934`
instead of `0.732`, scoring 0.778 instead of 0.717 and placing about third.

`rerank_legacy_layout.patch` adds a branch that rebuilds the composite as
`0.6 * distance + 0.4 * angle` at that run's own weights. It is recorded here
for reproducibility and is **not** applied to `scripts/rerank_geometry.py` —
apply it there before re-running if you want these numbers back.

## Known limits

- **The hbond slice is not perfectly uniform.** It is each row's own pipeline
  definition of the geometry composite. The confidence term is identical in
  both configs (`complex_ipde` ramped 0.5 to 5.0 A), and planarity,
  cat_polarity and apolar_match are recomputed from the structures the same
  way for every design. So roughly 70 % of the score is strictly comparable
  and 30 % approximately so. Recomputing the hbond composite from the
  structures with one definition would close this.
- **Two scored designs are absent.** `02.c01.wiggle.4b0d78` and
  `02.c01.wiggle.6d5166` have no committed structure: Boltz served them from
  cache as duplicates of `4642b0`, so nothing was written to re-score.
- **Round 1 scores mix two scales.** `old_score` and `old_rank` in the
  re-rank table come from each run's own config, 29 of them on the
  confidence-heavy 0.70/0.30 formula. Use `new_score` for ranking; treat
  `old_score` as a per-run gate check.
- **Variant 01 is flattered by having only two bonds.** Its designs average
  their terms over two contacts rather than three, and the missing one is the
  nitro contact, the weakest across every variant. Designs with two measured
  contacts average 0.633; those with three average 0.581.

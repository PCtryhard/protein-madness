# outdated/ — everything superseded, in two layers

**Nothing in this folder is maintained. Do not follow it.** It is kept because
it records what we believed and why, which is worth more at hour 25 than a tidy
repo.

Read **[../PIPELINE.md](../PIPELINE.md)** for the current specification and
**[../README.md](../README.md)** for how to run it.

---

## Layer 2: the first full pipeline (superseded 21 Nov 2026)

The repo's second design: multi-backend generation and folding, a three-tier
scoring stack, an evolutionary driver, and a report/pitch generator. It was
built against the announced Kemp eliminase challenge and it ran — the results
in `../data/No_mutation_run/` and the measurements in `../RUNLOG.md` came out
of it, and they are still the empirical baseline.

It was replaced because it was too complex to build on, not because it was
wrong.

| Here | Replaced by |
|---|---|
| `src/` — `backends/ evolve/ fold/ generate/ io/ report/ score/` | **[../src/](../src/)** — one path: RFdiffusion → LigandMPNN → Boltz-2 → protonate → score |
| `src/score/tier1,tier2,tier3,official,eligibility,novelty` | **[../src/scoring/](../src/scoring/)** — one weighted score, Boltz-2 confidence + H-bond geometry |
| `src/evolve/driver.py` + `select.py` + `diversity.py` | **[../src/search/](../src/search/)** + **[../src/runner.py](../src/runner.py)** — per-scaffold elitist hill-climb, no cross-scaffold competition |
| `src/backends/` — Euler, Modal, Tamarind, local | gone. Two local GPUs, one stage per card, no sharding |
| `src/fold/` — ESMFold, Chai, ColabFold, Boltz | **[../src/stages/boltz.py](../src/stages/boltz.py)** — Boltz-2 only, because Boltz-2 *is* the scoring function |
| `src/io/` — normalisation, `NumberingMap`, validator | **[../src/tracking.py](../src/tracking.py)** — four explicit index maps, nothing inferred by proximity |
| `src/report/` — leaderboard, figures, pitch assets | `python -m src.cli report` — one markdown leaderboard per scaffold |
| `configs/pipeline.yaml` + `configs/target.example.yaml` | **[../config/pipeline.yaml](../config/pipeline.yaml)** + **[../config/targets/](../config/targets/)** |
| `scripts/run_pipeline.py`, `smoke_test.sh`, `tm_screen.py`, … | `python -m src.cli` — `run \| resume \| report \| smoke \| measure` |
| `tests/` — 166 tests against the old modules | **[../tests/](../tests/)** — 220 tests against the new ones |
| `BRIEF.md` — the hackathon strategy | **[../PIPELINE.md](../PIPELINE.md)** |
| `EVOLVE.md` — how to run the old design loop | **[../README.md](../README.md)** §Running it |
| `DESIGN_RATIONALE.md` — decisions behind the old scorer | **[../PIPELINE.md](../PIPELINE.md)** and its appendix |
| `env/euler_setup.sh`, `env/modal_image.py`, `env/environment.yml` | gone with the backends. **[../env/requirements.txt](../env/requirements.txt)** is CPU-only; GPU stacks are installed per-tool on the cluster |
| `notebooks/figures.ipynb` | nothing yet. No figures in this version |

### Worth keeping in mind from this layer

Three things it established that the new pipeline is built to respect:

* **Ligand drift after re-prediction is the whole problem.** RFD3 held the
  motif to 0.013 Å and Boltz-2 then put the ligand 11.18 Å from the catalytic
  base. `../RUNLOG.md` §1. This is why the new scorer measures Boltz-2 output
  and nothing else.
* **The angle convention.** The old scorer measured `CG-OD2···N3`, i.e.
  acceptor-side directionality. The new one defaults to the same thing, so the
  numbers stay comparable. `../PIPELINE.md` appendix says why.
* **A metric whose inputs are missing must not return a number.** Carried over
  verbatim as a rule: `resolve` raises, never defaults.

---

## Layer 1: the pre-announcement repo

Written **before the challenge was announced**, when the plan was to be
target-agnostic and the compute was assumed to be ETH Euler plus Modal. All
three assumptions turned out wrong:

| We assumed | It was actually |
|---|---|
| Unknown target, decided Saturday morning | A **de novo Kemp eliminase**, known from the start |
| Euler + Modal, ESMFold as the default folder | A provided **AWS cluster**, and **Boltz-2** |
| We would invent our own scoring | Four published metrics, re-implemented exactly |

| Here | Replaced by, at the time |
|---|---|
| `PLAN.md` | `BRIEF.md` §9–10 (itself now outdated) |
| `CLAUDE_CODE_BRIEF.md` | `BRIEF.md` |
| `ONBOARDING.md` | `../README.md` for setup |
| `MANIFEST.md` | `BRIEF.md` §3, §8 — tool and weight locations |
| `TODO.md` | `BRIEF.md` §11 |
| `PITCH.md` | `BRIEF.md` §12 |

---

## Still live, deliberately not moved

- **[../RUNLOG.md](../RUNLOG.md)** — the evidence record. Produced by the layer-2
  pipeline, but it measures the *target*, not the tooling, so it still applies.
- **[../DECISIONS.md](../DECISIONS.md)** — append-only and still in use. Some
  entries are superseded; the file says which.
- **[../data/](../data/)** — the 5FQJ seeds and the no-mutation baseline run.
  Inputs and results, not pipeline. The new `config/targets/5fqj.yaml` reads
  its geometry from `../data/5A_5FQJ.pdb`.

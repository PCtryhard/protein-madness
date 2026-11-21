# outdated/ — the pre-announcement repo

Everything in this folder was written **before the challenge was announced**, when
the plan was to be target-agnostic and the compute was assumed to be ETH Euler
plus Modal. All three assumptions turned out wrong:

| We assumed | It is actually |
|---|---|
| Unknown target, decided Saturday morning | A **de novo Kemp eliminase**, known from the start |
| Euler + Modal, ESMFold as the default folder | A provided **AWS cluster**, and **Boltz-2** — which is not a choice, because Boltz-2 *is* the scoring function |
| We would invent our own scoring | The score is **four published metrics**, which we re-implemented exactly |

**Nothing here is maintained. Do not follow it.** It is kept because it records
what we believed and why, which is worth more at hour 25 than a tidy repo.

## What replaced it

| Outdated file | Read instead |
|---|---|
| `PLAN.md` | **[../BRIEF.md](../BRIEF.md)** §9–10 — phases, gates and the Sunday freeze |
| `CLAUDE_CODE_BRIEF.md` | **[../BRIEF.md](../BRIEF.md)** — the whole strategy |
| `ONBOARDING.md` | **[../README.md](../README.md)** for setup, **[../EVOLVE.md](../EVOLVE.md)** for how to run things |
| `MANIFEST.md` | **[../BRIEF.md](../BRIEF.md)** §3, §8 — the real tool and weight locations on both clusters |
| `TODO.md` | **[../BRIEF.md](../BRIEF.md)** §11 — the build list. Roughly half of the old TODO became moot rather than done |
| `PITCH.md` | **[../BRIEF.md](../BRIEF.md)** §12 — the pitch outline |

## Still live, deliberately not moved

- **`../DECISIONS.md`** — append-only and still in use. Some entries are
  superseded (ESMFold as default folder, the 11 GB VRAM ceiling); the file's own
  rule is that a superseded entry is never edited, only followed by a new one
  saying so. That convention is why it stays.
- The code these documents describe — `src/backends/euler.py`, `modal.py`,
  `tamarind.py`, `src/fold/esmfold.py`, `colabfold.py`, `chai.py`,
  `src/generate/rfdiffusion.py`, `src/score/tier3.py` — is **parked, not
  deleted**, and still passes its tests. It is not part of the Kemp pipeline and
  should not be debugged. `BRIEF.md` §11 lists it.
- `scripts/fetch_weights.sh` still appends its download log to
  `outdated/MANIFEST.md`. It belongs to the Euler/Modal world and is not used by
  the Kemp pipeline, which reads weights straight from `/mnt/ref` on the
  provided cluster.

## The one thing from here that is still true

> *"The differentiator here is evaluation, not generation. Every team will
> produce sequences; few will be able to say anything defensible about whether
> their designs are good."*

That was written before we knew the target, and it turned out to be the whole
game. `src/score/official.py` is the direct descendant of it.

"""Tamarind.bio backend: a documented MANUAL fallback, not an automated one.

There is no public, documented Tamarind.bio API that we have seen, and the
event code that unlocks the 200 free jobs per participant only arrives during
the hackathon. So this module does NOT pretend to submit anything. Every
automated entry point raises NotImplementedError; what it provides instead is:

* :func:`instructions` — exactly what a human uploads and exactly where the
  result file has to land for the rest of the pipeline to pick it up.
* :func:`missing_outputs` — checks which designs the operator has not brought
  back yet. This part is real: it only looks at the filesystem.

The contract with the rest of the repo is the fold stage's output contract, so
a hand-run Tamarind job is indistinguishable downstream from an ESMFold run:

    <fold_out_dir>/<design_id>.pdb        one structure per design, that name

Nothing else. If pLDDT/PAE are wanted from a Tamarind job they have to be
parsed out of its own output archive, and no parser exists here because we have
not seen the archive layout. Scoring that reads the B-factor column of the PDB
(mean pLDDT) will work; PAE-based Tier-2 metrics will not.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from .base import Backend, BackendUnavailable, Command, JobResult, Resources

log = logging.getLogger("pipeline.backends.tamarind")

WEB_UI = "https://app.tamarind.bio/"

UNAVAILABLE = (
    "tamarind is a MANUAL web-UI fallback: there is no documented public API we "
    "have seen, and the event code arrives during the hackathon. Nothing was "
    "submitted. Run src.backends.tamarind.instructions(...) and do it by hand, "
    "or use --backend euler / --backend modal."
)


def instructions(fasta_path: str | Path, fold_out_dir: str | Path,
                 design_ids: Sequence[str] | None = None) -> str:
    """The operator runbook for one manual Tamarind fold batch.

    ``fasta_path`` is the FASTA to upload (its record ids ARE the design_ids;
    that is what keeps the leaderboard join intact). ``fold_out_dir`` is the
    ``--out`` directory of the fold stage this stands in for.
    """
    fasta_path = Path(fasta_path)
    fold_out_dir = Path(fold_out_dir)
    ids = list(design_ids) if design_ids is not None else []
    sample = ", ".join(ids[:3]) + (" ..." if len(ids) > 3 else "")
    return f"""\
MANUAL TAMARIND RUN -- there is no API, a human does this.

1. Open {WEB_UI} and sign in. Enter the hackathon event code to get the 200
   free jobs (the code is handed out during the event; without it you are on
   the free tier and will run out fast).

2. UPLOAD exactly this file:
       {fasta_path.resolve()}
   {f'It contains {len(ids)} record(s): {sample}' if ids else 'One record per design.'}
   Do NOT rename the FASTA records. The record id is the design_id and it is
   the only thing joining this batch back to results/leaderboard.csv.

3. Pick a structure-prediction job type (ESMFold / Boltz / Chai, whatever the
   UI offers that matches what the rest of the run used). Note the exact tool
   and version in DECISIONS.md -- a judge will ask, and a batch folded with a
   different predictor is not comparable to the rest of the table.

4. DOWNLOAD the results and put ONE PDB PER DESIGN in this directory, named
   after the design_id and nothing else:
       {fold_out_dir.resolve().as_posix()}/<design_id>.pdb
   Example:
       {(fold_out_dir.resolve() / ((ids[0] if ids else "d_1a2b3c4d") + ".pdb")).as_posix()}

5. Verify every design came back before scoring:
       python -c "from src.io.fasta import read_fasta; \\
           from src.backends.tamarind import missing_outputs; \\
           print(missing_outputs([r.id for r in read_fasta('{fasta_path.as_posix()}')], \\
                                 '{fold_out_dir.as_posix()}'))"
   An empty list means the batch is complete.

6. Record it in MANIFEST.md / DECISIONS.md: which designs were folded where.
   A mixed-provenance leaderboard with no record of the mix is unpitchable.
"""


def missing_outputs(design_ids: Iterable[str], fold_out_dir: str | Path) -> list[str]:
    """Design ids that still have no ``<design_id>.pdb`` in ``fold_out_dir``.

    The only automated thing this module can honestly do. Empty list means the
    manual batch is complete and the fold stage's outputs are ready to score.
    """
    out = Path(fold_out_dir)
    return [d for d in design_ids if not (out / f"{d}.pdb").is_file()]


class TamarindBackend(Backend):
    """Honest stub. Every automated call raises; :func:`instructions` is the API."""

    name = "tamarind"

    def available(self) -> bool:
        return False

    def submit(self, cmd: Command, resources: Resources) -> str:
        raise NotImplementedError(UNAVAILABLE)

    def wait(self, job_id: str, timeout: float | None = None,
             poll_interval: float = 5.0) -> JobResult:
        raise NotImplementedError(UNAVAILABLE)

    def cancel(self, job_id: str) -> None:
        raise BackendUnavailable(
            "tamarind jobs are started by hand in the web UI and must be "
            f"cancelled there: {WEB_UI}"
        )

    @staticmethod
    def instructions(fasta_path: str | Path, fold_out_dir: str | Path,
                     design_ids: Sequence[str] | None = None) -> str:
        """See :func:`instructions`."""
        return instructions(fasta_path, fold_out_dir, design_ids)

    @staticmethod
    def missing_outputs(design_ids: Iterable[str], fold_out_dir: str | Path) -> list[str]:
        """See :func:`missing_outputs`."""
        return missing_outputs(design_ids, fold_out_dir)

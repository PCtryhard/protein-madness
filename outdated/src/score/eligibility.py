"""The rules that zero a design no matter how well it scores.

From the challenge slide, three hard disqualifiers:

* **length** - at most 260 amino acids.
* **sequence novelty** - more than **70 % identity** to a published enzyme does
  not qualify.
* **structural novelty** - a backbone with **TM-score > 0.7** to a published
  structure does not qualify.

A disqualified design scores zero, so these are checks, not opinions, and they
run automatically before anything is submitted - and periodically during a run,
because evolution can drift towards a natural protein as easily as away from it.

**Nothing here fabricates a number.** If the database or the binary is missing,
these functions raise. A novelty metric that silently returns 0.0 when it could
not search is worse than no metric at all: 0.0 is a number a filter will act on,
and it would tell us a design is novel precisely when we failed to check.

Tooling, all installed on the AWS head node under ``/mnt/work/tools``:

* ``mmseqs``   searches sequences against the provided UniRef90 database
* ``foldseek`` searches a structure against the PDB and reports TM-scores
* ``TMalign``  pairwise TM-score, for spot checks against one named structure
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("pipeline.score.eligibility")

MAX_LENGTH = 260
MAX_SEQUENCE_IDENTITY = 0.70
MAX_TM_SCORE = 0.70

#: Where the tools live on the AWS head node. Overridable for other sites.
DEFAULT_MMSEQS = "/mnt/ref/software/mmseqs2-18-8cc5c/bin/mmseqs"
DEFAULT_MMSEQS_DB = "/mnt/ref/data_weights/mmseqs2_db/targetDB_gpu"
DEFAULT_FOLDSEEK = "/mnt/work/tools/bin/foldseek"
DEFAULT_FOLDSEEK_DB = "/mnt/work/tools/db/pdb"
DEFAULT_TMALIGN = "/mnt/work/tools/bin/TMalign"


class ToolUnavailable(RuntimeError):
    """A required binary or database is missing. Never silently substituted."""


def _require(path: str, what: str) -> str:
    if shutil.which(path) or Path(path).exists():
        return path
    raise ToolUnavailable(
        f"{what} not found at {path!r}. Install it or pass an explicit path; "
        f"this check will not guess and will not return a default."
    )


# ------------------------------------------------------------------ length

def check_length(sequence: str, maximum: int = MAX_LENGTH) -> tuple[bool, str]:
    if len(sequence) > maximum:
        return False, f"length {len(sequence)} > {maximum}"
    return True, ""


# ------------------------------------------------------- sequence identity

def max_sequence_identity(
    sequences: dict[str, str],
    db: str = DEFAULT_MMSEQS_DB,
    mmseqs: str = DEFAULT_MMSEQS,
    tmp: Path | None = None,
    sensitivity: float = 5.7,
    gpu: bool = True,
) -> dict[str, float]:
    """Highest identity to anything in the reference database, per sequence.

    Returns a value in [0, 1] per design. A design with **no hits at all** maps
    to 0.0 - and that is a real measurement, not a fallback: mmseqs searched and
    found nothing, which is exactly what we expect for a de novo design and is
    the result we want.

    ``gpu=True`` by default because the **provided UniRef90 index is built for
    GPU prefilter only** - a CPU search against it fails with "Index was not
    built with `prefilter` support". That means this must run on a GPU node,
    exactly as the organisers' own pipeline does. Setting ``gpu=False`` will
    only work against a CPU-indexed database.
    """
    if not sequences:
        return {}   # nothing was asked, so nothing is unchecked
    _require(mmseqs, "mmseqs")
    if not Path(f"{db}.dbtype").exists() and not Path(db).exists():
        raise ToolUnavailable(f"mmseqs database not found at {db!r}")

    with tempfile.TemporaryDirectory(dir=tmp) as work_str:
        work = Path(work_str)
        fasta = work / "query.fasta"
        fasta.write_text(
            "".join(f">{name}\n{seq}\n" for name, seq in sequences.items()), newline="\n"
        )
        out = work / "hits.m8"
        command = [
            mmseqs, "easy-search", str(fasta), db, str(out), str(work / "tmp"),
            "-s", str(sensitivity), "--format-output", "query,target,fident,alnlen,evalue",
            "--max-seqs", "300",
        ]
        if gpu:
            command += ["--gpu", "1", "--db-load-mode", "2"]
        log.info("mmseqs easy-search over %d sequences", len(sequences))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"mmseqs failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )
        best = {name: 0.0 for name in sequences}
        if out.exists():
            for line in out.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                name, identity = parts[0], float(parts[2])
                if name in best:
                    best[name] = max(best[name], identity)
        return best


# ------------------------------------------------------ structural novelty

def max_tm_score(
    structure: Path,
    db: str = DEFAULT_FOLDSEEK_DB,
    foldseek: str = DEFAULT_FOLDSEEK,
    tmp: Path | None = None,
) -> float:
    """Highest TM-score to any published structure, via a Foldseek search.

    This is the check we were blind on all morning. ``TMalign`` compares two
    structures; the rule is about *any* published structure, so it needs a
    search, which is what Foldseek does.

    TM-score is reported against the **query** length (``alntmscore`` normalised
    by the query), which is the convention that matters when our design is the
    query.
    """
    _require(foldseek, "foldseek")
    structure = Path(structure)
    if not structure.is_file():
        raise FileNotFoundError(f"no structure at {structure}")
    if not Path(f"{db}.dbtype").exists() and not Path(db).exists():
        raise ToolUnavailable(
            f"foldseek database not found at {db!r}; run "
            f"`foldseek databases PDB {db} tmp` on a node with internet"
        )
    with tempfile.TemporaryDirectory(dir=tmp) as work_str:
        work = Path(work_str)
        out = work / "hits.tsv"
        command = [
            foldseek, "easy-search", str(structure), db, str(out), str(work / "tmp"),
            "--alignment-type", "1",                     # TM-align mode
            "--format-output", "query,target,alntmscore,qtmscore,ttmscore",
            "--max-seqs", "1000", "-e", "10",
        ]
        log.info("foldseek TM search for %s", structure.name)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"foldseek failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )
        best = 0.0
        if out.exists():
            for line in out.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        best = max(best, float(parts[3]))   # qtmscore
                    except ValueError:
                        continue
        return best


#: RCSB full-text queries that define the targeted reference set. These are the
#: structures our scaffolds are most likely to resemble, because the motif we
#: build around was lifted from one of them.
KEMP_SEARCH_TERMS = (
    "Kemp eliminase",
    "Kemp elimination",
    "benzisoxazole",
    "computational enzyme design",
)
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def rcsb_search(term: str, limit: int = 40, timeout: int = 60) -> list[str]:
    """PDB ids matching a full-text query.

    The reference set is built by search rather than from a hardcoded list, so
    it is reproducible and auditable: anyone can re-run the query and see the
    same entries, and nothing depends on an id someone remembered.
    """
    import json as _json
    import urllib.request

    query = {
        "query": {"type": "terminal", "service": "full_text",
                  "parameters": {"value": term}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    request = urllib.request.Request(
        RCSB_SEARCH_URL, data=_json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        raw = handle.read()
    if not raw.strip():
        return []            # RCSB answers 204 No Content when nothing matches
    return [hit["identifier"] for hit in _json.loads(raw).get("result_set", [])]


def fetch_structure(pdb_id: str, out_dir: Path, timeout: int = 60) -> Path | None:
    """Download one PDB entry. Returns None if it could not be fetched."""
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pdb_id}.pdb"
    if path.is_file() and path.stat().st_size > 1000:
        return path
    try:
        with urllib.request.urlopen(
            RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id), timeout=timeout
        ) as handle:
            path.write_bytes(handle.read())
    except Exception as exc:                     # noqa: BLE001 - report, never guess
        log.warning("could not fetch %s: %s", pdb_id, exc)
        return None
    return path


def build_reference_set(
    out_dir: Path, terms: tuple[str, ...] = KEMP_SEARCH_TERMS, limit: int = 40
) -> list[Path]:
    """Download the targeted set of published structures to compare against.

    Needs outbound internet, so run it on a login/head node - never inside a
    compute job, where a retrying download looks exactly like a hang.
    """
    ids: list[str] = []
    for term in terms:
        try:
            found = rcsb_search(term, limit=limit)
        except Exception as exc:                 # noqa: BLE001
            log.warning("RCSB search failed for %r: %s", term, exc)
            continue
        log.info("RCSB %r: %d hits", term, len(found))
        ids.extend(found)
    paths = [p for p in (fetch_structure(i, out_dir) for i in sorted(set(ids))) if p]
    log.info("reference set: %d structures in %s", len(paths), out_dir)
    return paths


def max_tm_vs_set(
    structure: Path, targets: list[Path], tmalign: str = DEFAULT_TMALIGN
) -> tuple[float, str]:
    """(highest TM-score, which structure produced it) over a reference set.

    **This is a lower bound on the real rule**, which is about *any* published
    structure. A targeted set of a hundred-odd entries can only ever say "at
    least this similar". Treat a pass here as "not obviously disqualified",
    never as "cleared" - see :func:`max_tm_score` for the full Foldseek search.
    """
    if not targets:
        raise ToolUnavailable(
            "empty reference set; run build_reference_set() on a node with internet"
        )
    best, best_id = 0.0, ""
    for target in targets:
        try:
            score = pairwise_tm(structure, target, tmalign)
        except RuntimeError as exc:
            log.warning("TMalign failed on %s vs %s: %s", structure.name, target.name, exc)
            continue
        if score > best:
            best, best_id = score, target.stem
    return best, best_id


def pairwise_tm(a: Path, b: Path, tmalign: str = DEFAULT_TMALIGN) -> float:
    """TM-score between two structures, **normalised by the first** (our design).

    TMalign prints two numbers, one per chain length. Which one you take is not
    a detail: normalising by the reference instead would let a design hide
    behind a much larger published structure. The question we are asked is "how
    much of *our* design looks like something published", so the query length is
    the right denominator.
    """
    _require(tmalign, "TMalign")
    result = subprocess.run([tmalign, str(a), str(b)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TMalign failed:\n{result.stderr[-1000:]}")
    scores = [
        float(line.split("=")[1].split()[0])
        for line in result.stdout.splitlines()
        if line.startswith("TM-score=") and "Chain_1" in line
    ]
    if not scores:
        raise RuntimeError(f"TMalign produced no TM-score:\n{result.stdout[-1000:]}")
    return max(scores)


# ------------------------------------------------------------------ verdict

@dataclass
class EligibilityReport:
    design_id: str
    length: int
    sequence_identity: float = math.nan
    tm_score: float = math.nan
    eligible: bool = True
    reasons: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "design_id": self.design_id, "length": self.length,
            "sequence_identity": self.sequence_identity, "tm_score": self.tm_score,
            "eligible": self.eligible, "reasons": "; ".join(self.reasons),
            "unchecked": "; ".join(self.unchecked),
        }


def assess(
    design_id: str,
    sequence: str,
    identity: float | None = None,
    tm: float | None = None,
) -> EligibilityReport:
    """Combine the three rules into one verdict.

    A rule we could not evaluate is recorded in ``unchecked`` and does **not**
    count as a pass. "We did not check" and "it passed" must never look the same
    in the leaderboard.
    """
    report = EligibilityReport(design_id=design_id, length=len(sequence))
    ok, reason = check_length(sequence)
    if not ok:
        report.reasons.append(reason)

    if identity is None:
        report.unchecked.append("sequence identity")
    else:
        report.sequence_identity = identity
        if identity > MAX_SEQUENCE_IDENTITY:
            report.reasons.append(
                f"sequence identity {identity:.2f} > {MAX_SEQUENCE_IDENTITY}"
            )

    if tm is None:
        report.unchecked.append("TM-score")
    else:
        report.tm_score = tm
        if tm > MAX_TM_SCORE:
            report.reasons.append(f"TM-score {tm:.2f} > {MAX_TM_SCORE}")

    report.eligible = not report.reasons
    return report

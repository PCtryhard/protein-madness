"""The CLI contract every tool in generate/, fold/ and score/ obeys.

    python -m src.<module>.<tool> --in <path> --out <path> --config configs/pipeline.yaml [--device ...]

Guarantees provided here so no tool has to reimplement them:

* ``--in`` accepts a FASTA file, a single structure, or a directory of
  structures; :func:`load_inputs` normalises all three into one object.
* every tool writes ``<out>/<tool>.csv``, one row per design, ``design_id``
  first column.
* every tool writes ``<out>/<tool>.meta.json`` with tool name, version, git
  SHA, config hash, backend, wall time and the exact command line.
* a tool whose outputs exist with a matching config hash exits 0 logging
  "cached" unless ``--force``.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import yaml

from .ids import config_hash, git_sha
from .io.fasta import SeqRecord, read_fasta
from .io.structure import STRUCT_SUFFIXES, list_structures
from .io.target import REPO_ROOT, load_target

log = logging.getLogger("pipeline")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------- arguments

def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="FASTA, a structure, or a directory of structures")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "pipeline.yaml"),
                    help="pipeline config (default: configs/pipeline.yaml)")
    ap.add_argument("--target", default=None,
                    help="target config; required by tools that need the reference enzyme")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: from config)")
    ap.add_argument("--backend", default=None, help="override the tool's backend")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N inputs")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def load_pipeline_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: pipeline config must be a mapping")
    return cfg


def tool_config(cfg: dict, tool: str) -> dict:
    """Merge ``defaults`` with the tool's own section. Tool keys win."""
    merged = dict(cfg.get("defaults") or {})
    merged.update(cfg.get("tools", {}).get(tool) or {})
    return merged


def resolve_device(args, tcfg: dict) -> str:
    return args.device or tcfg.get("device") or "cpu"


# ---------------------------------------------------------------- inputs

@dataclass
class Inputs:
    """Whatever ``--in`` pointed at, in one shape.

    ``sequences``: FASTA records (empty for structure input).
    ``structures``: structure paths (empty for FASTA input).
    ``design_ids``: ids in input order. For structures the id is the file stem,
    which is how the fold stage keeps the join to the generate stage intact.
    """

    path: Path
    kind: str                                # "fasta" | "structures"
    sequences: list[SeqRecord] = field(default_factory=list)
    structures: list[Path] = field(default_factory=list)

    @property
    def design_ids(self) -> list[str]:
        if self.kind == "fasta":
            return [r.id for r in self.sequences]
        return [p.stem for p in self.structures]

    def __len__(self) -> int:
        return len(self.sequences) if self.kind == "fasta" else len(self.structures)


def load_inputs(path: str | Path, limit: int | None = None) -> Inputs:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--in path does not exist: {path}")
    if path.is_dir():
        structures = list_structures(path)
        if not structures:
            raise FileNotFoundError(f"--in directory has no {sorted(STRUCT_SUFFIXES)} files: {path}")
        return Inputs(path, "structures", structures=structures[:limit])
    if path.suffix.lower() in STRUCT_SUFFIXES:
        return Inputs(path, "structures", structures=[path])
    records = read_fasta(path)
    return Inputs(path, "fasta", sequences=records[:limit])


# ---------------------------------------------------------------- outputs, caching, meta

@dataclass
class Stage:
    """One tool invocation: paths, caching decision, meta.json."""

    tool: str
    out_dir: Path
    cfg_hash: str
    version: str = "0.1.0"
    backend: str = "local"
    extra_meta: dict = field(default_factory=dict)

    @property
    def csv_path(self) -> Path:
        return self.out_dir / f"{self.tool}.csv"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / f"{self.tool}.meta.json"

    def cached(self) -> bool:
        """True if this exact stage already ran with this config hash."""
        if not (self.csv_path.exists() and self.meta_path.exists()):
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return meta.get("config_hash") == self.cfg_hash and meta.get("status") == "ok"

    def write_csv(self, rows: Sequence[dict]) -> pd.DataFrame:
        """One row per design, ``design_id`` first. Raises if ids are missing or duplicated."""
        df = pd.DataFrame(list(rows))
        if df.empty:
            raise ValueError(f"{self.tool}: produced no rows")
        if "design_id" not in df.columns:
            raise ValueError(f"{self.tool}: rows have no design_id column")
        if df["design_id"].duplicated().any():
            dup = df.loc[df["design_id"].duplicated(), "design_id"].unique()[:5]
            raise ValueError(f"{self.tool}: duplicate design_ids {list(dup)}")
        cols = ["design_id"] + [c for c in df.columns if c != "design_id"]
        df = df[cols]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        return df

    def write_meta(self, status: str, wall_s: float, n_rows: int, **extra) -> None:
        meta = {
            "tool": self.tool,
            "version": self.version,
            "status": status,
            "git_sha": git_sha(),
            "config_hash": self.cfg_hash,
            "backend": self.backend,
            "device": self.extra_meta.get("device", ""),
            "wall_seconds": round(wall_s, 3),
            "n_rows": n_rows,
            "command": " ".join([Path(sys.executable).name, "-m", f"src.{self.tool}", *sys.argv[1:]])
            if sys.argv else "",
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self.extra_meta,
            **extra,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


@contextmanager
def stage(tool: str, args, tcfg: dict, version: str = "0.1.0",
          hash_extra: object = None) -> Iterator[Stage | None]:
    """Run a stage, or yield None when the cached outputs are still valid.

    Usage::

        with stage("score.tier1", args, tcfg) as st:
            if st is None:
                return 0
            st.write_csv(rows)
    """
    out_dir = Path(args.out)
    cfg_h = config_hash(tcfg, extra={"tool": tool, "version": version, "extra": hash_extra})
    st = Stage(tool=tool, out_dir=out_dir, cfg_hash=cfg_h, version=version,
               backend=args.backend or tcfg.get("backend", "local"),
               extra_meta={"device": resolve_device(args, tcfg), "input": str(args.inp)})
    if st.cached() and not args.force:
        log.info("cached: %s (config %s) -> %s", tool, cfg_h, st.csv_path)
        yield None
        return
    t0 = time.perf_counter()
    try:
        yield st
    except BaseException as e:
        st.write_meta("failed", time.perf_counter() - t0, 0,
                      error=f"{type(e).__name__}: {e}")
        raise
    wall = time.perf_counter() - t0
    n = 0
    if st.csv_path.exists():
        n = max(0, sum(1 for _ in st.csv_path.open(encoding="utf-8")) - 1)
    st.write_meta("ok", wall, n)
    log.info("%s: %d rows in %.1fs -> %s", tool, n, wall, st.csv_path)


def require_target(args):
    """Load the target config, with a clear error when a tool needs it and it is missing."""
    if not args.target:
        raise SystemExit(
            "this tool needs the target enzyme: pass --target configs/target.yaml "
            "(copy configs/target.example.yaml)"
        )
    return load_target(args.target)

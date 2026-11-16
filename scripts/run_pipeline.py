"""One entry point: take a target config, run the stages, write the leaderboard.

    python scripts/run_pipeline.py --target configs/target.yaml --out results/run1
    python scripts/run_pipeline.py --target configs/target.yaml --out results/run1 \
        --stages mutate,tier1,leaderboard,figures

Each stage is one subprocess obeying the CLI contract, so a crash at hour 20
costs one stage and not the run: re-running skips what is already cached.
Stages that need a GPU or model weights are opt-in by name; the default chain
is the CPU path that must always work.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cli import load_pipeline_config, setup_logging  # noqa: E402
from src.io.validate_target import validate              # noqa: E402

DEFAULT_STAGES = ["validate", "mutate", "tier1", "leaderboard", "figures"]
GPU_STAGES = {"fold", "tier2", "proteinmpnn", "ligandmpnn", "rfdiffusion"}


def run(cmd: list[str], label: str, dry: bool = False) -> tuple[int, float]:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    if dry:
        return 0, 0.0
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="configs/target.yaml")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "run"),
                    help="run directory; each stage writes a subdirectory")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma-separated (default: {','.join(DEFAULT_STAGES)})")
    ap.add_argument("--parent", default=None, help="parent FASTA (default: target reference)")
    ap.add_argument("--fold-backend", default=None, help="esmfold|colabfold|boltz|chai|tamarind")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-designs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    py = [sys.executable]
    common = ["--config", args.config, "--target", args.target]
    if args.force:
        common.append("--force")
    if args.device:
        common += ["--device", args.device]
    if args.verbose:
        common.append("-v")

    parent = args.parent
    if parent is None:
        from src.io.target import load_target
        t = load_target(args.target)
        if t.reference_sequence and t.reference_sequence.exists():
            parent = str(t.reference_sequence)
        else:
            raise SystemExit("no parent sequence: set reference_sequence in the target config "
                             "or pass --parent")

    timings: dict[str, float] = {}
    designs_fasta = out / "mutate" / "designs.fasta"
    fold_dir = out / "fold"

    for stage in stages:
        if stage == "validate":
            errors, warnings, info = validate(args.target)
            for m in info:
                print(f"[ ok ] {m}")
            for m in warnings:
                print(f"[WARN] {m}")
            for m in errors:
                print(f"[FAIL] {m}")
            if errors:
                print("\nTarget config is invalid: every downstream number would be wrong. Stopping.")
                return 1
            continue

        if stage == "mutate":
            cmd = py + ["-m", "src.generate.mutate", "--in", parent,
                        "--out", str(out / "mutate")] + common
            if args.max_designs:
                cmd += ["--max-designs", str(args.max_designs)]
        elif stage == "tier1":
            if not designs_fasta.exists():
                print(f"[FAIL] tier1 needs {designs_fasta}: run the mutate stage first")
                return 1
            cmd = py + ["-m", "src.score.tier1", "--in", str(designs_fasta),
                        "--out", str(out / "tier1")] + common
        elif stage == "fold":
            backend = args.fold_backend
            if not backend:
                print("[FAIL] --fold-backend is required for the fold stage "
                      "(esmfold|colabfold|boltz|chai|tamarind). No default is chosen for you.")
                return 1
            cmd = py + ["-m", f"src.fold.{backend}", "--in", str(designs_fasta),
                        "--out", str(fold_dir)] + common
        elif stage == "tier2":
            cmd = py + ["-m", "src.score.tier2", "--in", str(fold_dir),
                        "--out", str(out / "tier2")] + common
        elif stage == "leaderboard":
            cmd = py + ["-m", "src.report.leaderboard", "--in", str(out),
                        "--out", str(out), "--config", args.config]
        elif stage == "figures":
            cmd = py + ["-m", "src.report.figures", "--in", str(out / "leaderboard.csv"),
                        "--out", str(out / "figures"), "--target", args.target,
                        "--plddt-dir", str(fold_dir), "--structure-dir", str(fold_dir)]
        elif stage == "pitch":
            cmd = py + ["-m", "src.report.pitch_assets", "--in", str(out / "leaderboard.csv"),
                        "--out", str(out / "pitch"), "--target", args.target,
                        "--plddt-dir", str(fold_dir), "--structure-dir", str(fold_dir)]
        else:
            print(f"[FAIL] unknown stage '{stage}'. Known: validate, mutate, tier1, fold, "
                  "tier2, leaderboard, figures, pitch")
            return 1

        code, secs = run(cmd, stage, args.dry_run)
        timings[stage] = secs
        if code != 0:
            print(f"\n[FAIL] stage '{stage}' exited {code}. Earlier stages are cached: "
                  f"fix and re-run with the same --out to resume.")
            return code

    if not args.dry_run:
        (out / "run.meta.json").write_text(json.dumps({
            "target": args.target, "stages": stages,
            "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
            "total_seconds": round(sum(timings.values()), 2),
        }, indent=1), encoding="utf-8")
    print(f"\nPipeline finished: {', '.join(stages)} in "
          f"{sum(timings.values()):.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

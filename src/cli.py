"""The command-line interface. No web interface, by design.

    python -m src.cli smoke    --target config/targets/01_his_asp.yaml
    python -m src.cli measure  --target config/targets/01_his_asp.yaml \n                               --pdb data/01/inputs/5fqj-6cf4761/01_his_asp_.pdb
    python -m src.cli run      --target config/targets/01_his_asp.yaml --name run1
    python -m src.cli watch     --run runs/run1 --follow
    python -m src.cli resume    --run runs/run1
    python -m src.cli report    --run runs/run1

`smoke` and `measure` need no GPU and no model weights: they are the phase-1
acceptance test and the way to check a target spec against a real structure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load
from .scoring.combine import combine
from .scoring.hbond import measure_all
from .store import DesignStore
from .tracking import identity_provenance, read_structure

DEFAULT_PIPELINE = "config/pipeline.yaml"
DEFAULT_RUNS_DIR = "runs"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------
# smoke: config loads, every tunable resolves, nothing touches a GPU
# --------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = load(args.pipeline, args.target)
    print(f"pipeline : {args.pipeline}")
    print(f"target   : {args.target}  ({cfg.target.name})")
    print()
    print(f"seeds           : {len(cfg.target.seeds)} "
          f"({', '.join(s.id for s in cfg.target.seeds)})")
    print(f"motif residues  : "
          f"{', '.join(f'{r.chain}{r.resnum}' for r in cfg.target.motif.fixed)}")
    print(f"ligand          : {cfg.target.ligand.id} "
          f"({'CCD ' + cfg.target.ligand.ccd if cfg.target.ligand.ccd else 'SMILES'})")
    print()
    print("hydrogen bonds:")
    rule = cfg.pipeline.scoring.angle.rule
    for bond in cfg.target.hbonds:
        side = "acceptor" if bond.angle_is_acceptor_side(rule) else "D-H...A"
        ideal = bond.acceptor_ideal_angle if bond.angle_is_acceptor_side(rule) else 180.0
        gate = "required" if bond.required else "scored only"
        print(f"  {bond.id}")
        print(f"      measured at : {side}  (ideal {ideal} deg)")
        print(f"      gate        : {gate}")
    print()
    print(f"angle rule      : {rule}")
    print(f"protonation     : {cfg.pipeline.protonation.engine}"
          f"{' (skipped: not needed by this rule)' if rule == 'acceptor_side_always' else ''}")
    gw = cfg.pipeline.scoring.geometry_weights
    print(f"score weights   : confidence {cfg.pipeline.scoring.weights.confidence}, "
          f"geometry {cfg.pipeline.scoring.weights.geometry} "
          f"(catalytic {gw.catalytic}, non-catalytic {gw.non_catalytic}, "
          f"angle {gw.angle})")
    print(f"aggregate       : {cfg.pipeline.scoring.aggregate}")
    dist = cfg.pipeline.scoring.distance
    print(f"gate            : required bond beyond {dist.fail} A, or closer "
          f"than {dist.clash} A, zeroes the design")
    print(f"distance term   : 1.0 at {dist.ideal} A, falling to 0 at "
          f"{dist.clash} A below and {dist.tolerated} A above")
    print(f"max tokens      : {cfg.pipeline.boltz.max_tokens}")
    print(f"gpus            : {cfg.pipeline.run.gpus}")
    print()

    missing = [s.pdb for s in cfg.target.seeds if not Path(s.pdb).exists()]
    if missing:
        print(f"NOTE: seed PDBs not found: {missing}")
    print("SMOKE TEST PASSED")
    return 0


# --------------------------------------------------------------------------
# measure: resolve and measure the named bonds in a real structure
# --------------------------------------------------------------------------


def cmd_measure(args: argparse.Namespace) -> int:
    """The phase-1 acceptance table, on any structure in seed numbering.

    Prints each named hydrogen bond with its resolved atoms and the resulting
    distance and angle. Any unresolvable atom raises rather than defaulting,
    which is the behaviour this subcommand exists to demonstrate.
    """
    cfg = load(args.pipeline, args.target)
    ligand = cfg.target.ligand
    structure = read_structure(args.pdb, ligand.ccd or ligand.id)

    ligand_atoms = sorted(
        {
            spec.atom
            for bond in cfg.target.hbonds
            for spec in (bond.donor, bond.acceptor,
                         bond.donor_antecedent, bond.acceptor_antecedent)
            if spec is not None and spec.site == "ligand"
        }
    )
    provenance = identity_provenance(
        design_id=f"measure:{Path(args.pdb).name}",
        scaffold_id=cfg.target.name,
        structure=structure,
        motif=cfg.target.motif,
        ligand_atoms=ligand_atoms,
    )

    rule = cfg.pipeline.scoring.angle.rule
    measurements = measure_all(cfg.target.hbonds, structure, provenance, rule)

    print(f"structure : {args.pdb}")
    print(f"target    : {cfg.target.name}   angle rule: {rule}")
    print()
    header = f"{'bond':<28} {'distance':>9} {'angle':>8} {'ideal':>7} {'d term':>7} {'a term':>7}"
    print(header)
    print("-" * len(header))
    for m in measurements:
        from .scoring.hbond import angle_term, distance_term

        d = distance_term(m.distance, cfg.pipeline.scoring.distance)
        a = angle_term(m.angle, m.angle_ideal, cfg.pipeline.scoring.angle)
        print(f"{m.bond_id:<28} {m.distance:>8.3f}A {m.angle:>7.1f}d "
              f"{m.angle_ideal:>6.1f}d {d:>7.3f} {a:>7.3f}")
    print()
    print("resolved atoms:")
    for m in measurements:
        print(f"  {m.bond_id}: " + ", ".join(f"{k}={v}" for k, v in m.atoms.items()))
    print()

    if args.confidence:
        from .scoring.confidence import normalise, read_confidence_json

        raw = read_confidence_json(args.confidence)
        confidence = normalise(raw, cfg.pipeline.scoring.confidence)
        score = combine(confidence, measurements, cfg.pipeline.scoring)
        print(f"confidence ({cfg.pipeline.scoring.confidence.metric}) : {confidence:.4f}")
        print(f"geometry sub-score                      : {score.geometry:.4f}")
        print(f"TOTAL                                   : {score.total:.4f}"
              + (f"   GATED on {score.gated_bonds}" if score.gated else ""))
    else:
        print("No --confidence given, so only the geometry terms are shown.")
        print("The total score needs a Boltz-2 confidence JSON.")
    return 0


# --------------------------------------------------------------------------
# run / resume
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import RunPaths, Runner

    cfg = load(args.pipeline, args.target)
    run_name = args.name or cfg.pipeline.run.name
    if not run_name:
        print("error: a run needs a name (--name, or run.name in pipeline.yaml)",
              file=sys.stderr)
        return 2

    paths = RunPaths.create(args.runs_dir, run_name)
    print(f"run       : {run_name}")
    print(f"directory : {paths.root}")
    print(f"scaffolds : {len(cfg.target.seeds)}")
    print(f"cycles    : {cfg.pipeline.run.max_cycles}"
          + ("  (seed-only)" if cfg.pipeline.run.max_cycles == 0 else ""))
    print()
    print("These are written as each design finishes, not at the end:")
    print(f"  {paths.designs_csv}")
    print(f"  {paths.sequences_fasta}")
    print(f"  {paths.root / 'progress.txt'}")
    print(f"  {paths.root / 'structures'}/<design_id>/   (backbones and predictions)")
    print()
    print("Watch it from another terminal with:")
    print(f"  python -m src.cli watch --run {paths.root}")
    print()

    with DesignStore(paths.db) as store:
        runner = Runner(cfg, paths, store)
        searches = runner.run_all()
        _print_run_summary(searches, store)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume is deliberately not implemented yet. Phase 3 wires it up."""
    run_dir = Path(args.run)
    db = run_dir / "designs.db"
    if not db.exists():
        print(f"error: no designs.db in {run_dir}", file=sys.stderr)
        return 2
    print(
        "resume is not implemented. The pieces it needs are in place -- "
        "DesignStore.frontier_at() and max_cycle() rebuild a scaffold's state "
        "from the database -- but rebuilding a ScaffoldSearch's patience "
        "counter from stored rows is untested, and a resume that silently "
        "restarts the patience clock would change where the search stops.\n"
        "\nUse `report` to see what the run reached.",
        file=sys.stderr,
    )
    return 1


def cmd_watch(args: argparse.Namespace) -> int:
    """Print a live view of a run from its CSV. Safe while the run is going.

    Reads designs.csv and progress.txt, which the runner appends to and
    rewrites as designs finish. Deliberately reads the CSV rather than
    designs.db: a plain file read cannot block a worker mid-write, and it works
    just as well on a copy rsynced to another machine.
    """
    import time

    run_dir = Path(args.run)
    csv_path = run_dir / "designs.csv"
    progress_path = run_dir / "progress.txt"

    if not run_dir.exists():
        print(f"error: {run_dir} does not exist", file=sys.stderr)
        return 2

    while True:
        if progress_path.exists():
            print(progress_path.read_text(encoding="utf-8").rstrip())
        else:
            print(f"(no progress.txt in {run_dir} yet)")

        rows = _read_designs_csv(csv_path)
        scored = [r for r in rows if r.get("status") == "scored"]
        if scored:
            scored.sort(key=lambda r: _as_float(r.get("score")), reverse=True)
            print()
            print(f"top {min(args.limit, len(scored))} of {len(scored)} scored:")
            print(f"  {'score':>7}  {'cycle':>5}  {'branch':<7}  design")
            for row in scored[: args.limit]:
                print(f"  {_as_float(row.get('score')):>7.4f}  "
                      f"{row.get('cycle', ''):>5}  {row.get('branch', ''):<7}  "
                      f"{row.get('design_id', '')}")

        failed = [r for r in rows if r.get("status") == "errored"]
        if failed:
            print()
            print(f"{len(failed)} errored; most recent:")
            for row in failed[-args.errors:]:
                print(f"  {row.get('design_id', '')}: {row.get('error', '')[:120]}")

        if not args.follow:
            return 0
        time.sleep(args.interval)
        print()
        print("-" * 70)
        print()


def _read_designs_csv(path: Path) -> list[dict]:
    """Read the live CSV, tolerating a row being appended as we read.

    The writer flushes and fsyncs whole rows, so a torn final line is
    unlikely -- but it costs nothing to drop one rather than crash a watcher.
    """
    if not path.exists():
        return []
    import csv as _csv

    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = _csv.DictReader(fh)
        rows = []
        try:
            for row in reader:
                rows.append(row)
        except _csv.Error:
            pass
    return rows


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")


def _print_run_summary(searches: dict, store: DesignStore) -> None:
    print()
    print("=" * 72)
    for seed_id, search in searches.items():
        print(f"{seed_id}: best {search.best_score:.4f} "
              f"({search.best_design_id}) after {search.cycle} cycle(s)")
        print(f"    stopped: {search.stop_reason}")
    stats = store.stats()
    counts = store.counts_by_status()
    print()
    print(f"designs: {counts}")
    print(f"boltz cache: {stats.get('cache_hit', 0)} hits, "
          f"{stats.get('cache_miss', 0)} misses")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    """A markdown leaderboard per scaffold, plus the best design's lineage."""
    run_dir = Path(args.run)
    db = run_dir / "designs.db"
    if not db.exists():
        print(f"error: no designs.db in {run_dir}", file=sys.stderr)
        return 2

    lines: list[str] = [f"# Run report: {run_dir.name}", ""]
    with DesignStore(db) as store:
        counts = store.counts_by_status()
        stats = store.stats()
        lines += [
            f"Designs by status: {counts}",
            "",
            f"Boltz-2 cache: {stats.get('cache_hit', 0)} hits, "
            f"{stats.get('cache_miss', 0)} misses",
            "",
        ]

        for scaffold_id in store.scaffolds():
            board = store.leaderboard(scaffold_id, limit=args.limit)
            lines += [f"## {scaffold_id}", ""]
            if not board:
                lines += ["No scored designs.", ""]
                continue
            lines += [
                "| rank | design | cycle | branch | mutation | score | conf | dist | angle |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for rank, row in enumerate(board, start=1):
                lines.append(
                    f"| {rank} | `{row.design_id}` | {row.cycle} | {row.branch} | "
                    f"{row.mutation or '-'} | {_f(row.score)} | "
                    f"{_f(row.term_confidence)} | {_f(row.term_distance)} | "
                    f"{_f(row.term_angle)} |"
                )
            lines.append("")

            best = board[0]
            lines += [f"### Lineage of `{best.design_id}`", ""]
            for row in store.ancestry(best.design_id):
                lines.append(
                    f"- cycle {row.cycle} `{row.design_id}` "
                    f"({row.branch}{', ' + row.mutation if row.mutation else ''}) "
                    f"score {_f(row.score)}"
                )
            lines.append("")

            if best.hbond_raw:
                lines += ["### Per-bond geometry of the best design", "",
                          "| bond | distance | angle | ideal |",
                          "| --- | --- | --- | --- |"]
                for bond in json.loads(best.hbond_raw):
                    lines.append(
                        f"| {bond['bond_id']} | {bond['distance']:.3f} A | "
                        f"{bond['angle']:.1f} deg | {bond['angle_ideal']:.1f} deg |"
                    )
                lines.append("")

    text = "\n".join(lines)
    out = run_dir / "report.md" if args.write else None
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Motif-anchored protein design with score-guided search.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    # -v accepted after the subcommand as well as before it. argparse puts
    # top-level flags before the subcommand, which is not where anyone types
    # them: `run --name x -v` is the natural order and used to be an error.
    verbose_parent = argparse.ArgumentParser(add_help=False)
    # default=SUPPRESS so that when -v is NOT given after the subcommand, the
    # subparser leaves the namespace alone. With a normal False default it
    # overwrites the top-level flag, and `-v smoke` silently stops working --
    # which is the other half of this bug.
    verbose_parent.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS
    )

    sub = parser.add_subparsers(dest="command", required=True,
                                parser_class=lambda **kw: argparse.ArgumentParser(
                                    parents=[verbose_parent], **kw))

    def add_config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--pipeline", default=DEFAULT_PIPELINE)
        p.add_argument("--target", required=True)

    smoke = sub.add_parser("smoke", help="validate both config files, no GPU")
    add_config_args(smoke)
    smoke.set_defaults(func=cmd_smoke)

    measure = sub.add_parser(
        "measure", help="measure the named H-bonds in a structure, no GPU"
    )
    add_config_args(measure)
    measure.add_argument("--pdb", required=True, help=".pdb or .cif in seed numbering")
    measure.add_argument("--confidence", help="Boltz-2 confidence JSON, for a total")
    measure.set_defaults(func=cmd_measure)

    run = sub.add_parser("run", help="run the search")
    add_config_args(run)
    run.add_argument("--name", help="run name; overrides run.name")
    run.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="not implemented; see `report`")
    resume.add_argument("--run", required=True)
    resume.set_defaults(func=cmd_resume)

    watch = sub.add_parser(
        "watch", help="live view of a running (or finished) run, from its CSV"
    )
    watch.add_argument("--run", required=True)
    watch.add_argument("--limit", type=int, default=10,
                       help="how many top designs to show (default 10)")
    watch.add_argument("--errors", type=int, default=3,
                       help="how many recent errors to show (default 3)")
    watch.add_argument("-f", "--follow", action="store_true",
                       help="keep refreshing instead of printing once")
    watch.add_argument("--interval", type=float, default=30.0,
                       help="seconds between refreshes with --follow (default 30)")
    watch.set_defaults(func=cmd_watch)

    report = sub.add_parser("report", help="markdown leaderboard per scaffold")
    report.add_argument("--run", required=True)
    report.add_argument("--limit", type=int, default=20)
    report.add_argument("--write", action="store_true", help="write <run>/report.md")
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

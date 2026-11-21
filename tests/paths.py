"""Where the test suite looks for real repo data.

One module so a data reshuffle is a one-line change rather than fifteen. The
`data/` layout has already moved once -- from 5A/10A/15A full-length seeds to
the 01..07 motif folders -- and the tests that hard-coded the old paths went
quietly to SKIPPED rather than failing, which is the worst outcome.

Kept out of conftest.py because pytest treats conftest specially; a plain
module is importable from test files without surprises.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TARGETS_DIR = ROOT / "config" / "targets"

# The reference real target and its seed. Variant 01 is the one whose two
# contacts are genuine 5FQJ crystal geometry -- 02..07 carry modelled
# mutations that clash with the ligand -- so it is the variant the geometry
# assertions are written against.
REAL_TARGET = "config/targets/01_his_asp.yaml"
REAL_SEED = ROOT / "data" / "01" / "inputs" / "5fqj-6cf4761" / "01_his_asp_.pdb"

EXAMPLE_TARGET = "config/targets/example_target.yaml"

# A real Boltz-2 prediction kept in the repo, for the CIF reader.
BOLTZ_CIF = (
    ROOT / "data" / "No_mutation_run" / "boltz" / "seed-1" / "5A_01"
    / "5A_01_model_0.cif"
)
BOLTZ_CONFIDENCE = (
    ROOT / "data" / "No_mutation_run" / "boltz" / "seed-1" / "5A_01"
    / "confidence_5A_01_model_0.json"
)


def all_shipped_targets() -> list[str]:
    """Every target file in config/targets/, repo-relative.

    Globbed rather than listed so a new variant is covered by the
    loads-under-both-rules tests the moment it is generated.
    """
    return sorted(
        p.relative_to(ROOT).as_posix() for p in TARGETS_DIR.glob("*.yaml")
    )


def variant_targets() -> list[str]:
    """Just the seven numbered variants, without the schema example."""
    return [t for t in all_shipped_targets() if not t.endswith("example_target.yaml")]

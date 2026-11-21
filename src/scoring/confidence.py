"""The confidence term: Boltz-2 interface confidence, normalised to 0..1.

Every term in the score is normalised to [0, 1] with 1 best, because the
weights are otherwise meaningless -- mixing a raw iPDE in angstroms with an
angular deviation in degrees would let the units decide the weighting.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ConfidenceCfg

# Boltz-2 writes these into confidence_<name>_model_N.json. The keys we log for
# every design without scoring on them, so the weighting can be revisited
# offline without recomputing anything.
LOGGED_METRICS = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


class ConfidenceError(RuntimeError):
    """A confidence metric was missing or unusable. Stops the design."""


def read_confidence_json(path: str | Path) -> dict[str, float]:
    """Read Boltz-2's confidence sidecar, flattened to a float dict."""
    path = Path(path)
    if not path.exists():
        raise ConfidenceError(f"Boltz-2 confidence file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfidenceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfidenceError(f"{path} does not contain a JSON object")

    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, dict):
            # chains_ptm / pair_chains_iptm are nested one level
            for sub, subval in value.items():
                if isinstance(subval, (int, float)) and not isinstance(subval, bool):
                    out[f"{key}.{sub}"] = float(subval)
    if not out:
        raise ConfidenceError(f"{path} contained no numeric confidence values")
    return out


def logged_metrics(raw: dict[str, float]) -> dict[str, float]:
    """The subset we record for every design. Absent keys are simply absent.

    Not an error: which keys Boltz writes depends on its version and on whether
    a ligand was present. The scoring metric itself is checked separately and
    strictly, in `normalise`.
    """
    return {k: raw[k] for k in LOGGED_METRICS if k in raw}


def normalise(
    raw: dict[str, float],
    cfg: ConfidenceCfg,
    motif_pae: float | None = None,
) -> float:
    """Map the configured confidence metric onto [0, 1], 1 best.

    For iPDE and PAE, lower is better and the mapping is the linear ramp from
    PIPELINE.md:

        c = clip((worst - x) / (worst - best), 0, 1)

    For the iPTM metrics, higher is already better and already lives on [0, 1],
    so they pass through unchanged. Rescaling them would invent a range the
    model does not have.
    """
    metric = cfg.metric

    if metric == "motif_pae":
        if motif_pae is None:
            raise ConfidenceError(
                "scoring.confidence.metric is 'motif_pae' but no motif PAE was "
                "computed. It needs the PAE matrix from Boltz-2 plus the motif "
                "and ligand indices; see scoring/confidence.motif_pae()."
            )
        return _ramp(motif_pae, cfg.ipde_best, cfg.ipde_worst)

    if metric == "complex_ipde":
        value = _require(raw, "complex_ipde")
        return _ramp(value, cfg.ipde_best, cfg.ipde_worst)

    # iptm / ligand_iptm: already 0..1, higher better
    value = _require(raw, metric)
    if not 0.0 <= value <= 1.0:
        raise ConfidenceError(
            f"{metric}={value} is outside [0, 1]; it cannot be a confidence "
            "probability. Check the Boltz-2 output format."
        )
    return value


def _require(raw: dict[str, float], key: str) -> float:
    if key not in raw:
        raise ConfidenceError(
            f"confidence metric {key!r} is absent from the Boltz-2 output. "
            f"Keys present: {sorted(raw)}. Scoring a design on a metric that was "
            "not produced would be a fabricated number."
        )
    return raw[key]


def _ramp(value: float, best: float, worst: float) -> float:
    """Linear ramp, lower-is-better, clipped to [0, 1]."""
    return max(0.0, min(1.0, (worst - value) / (worst - best)))


def motif_pae(
    pae_matrix,
    motif_boltz_indices: list[int],
    ligand_token_indices: list[int],
) -> float:
    """Mean PAE restricted to motif-residue / ligand-atom pairs.

    `complex_ipde` is a global interface figure and averages over the whole
    interface, including regions we do not care about. This is the more
    sensitive signal for a motif-anchored objective, so it is logged from the
    start even while `complex_ipde` is the scoring metric.

    Indices are zero-based positions in the Boltz-2 token ordering. Both
    directions of each pair are averaged, since PAE is not symmetric.
    """
    import numpy as np

    pae = np.asarray(pae_matrix, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ConfidenceError(f"PAE matrix is not square: shape {pae.shape}")
    if not motif_boltz_indices or not ligand_token_indices:
        raise ConfidenceError(
            "motif PAE needs at least one motif index and one ligand index; got "
            f"{len(motif_boltz_indices)} and {len(ligand_token_indices)}"
        )
    n = pae.shape[0]
    for label, idx in (("motif", motif_boltz_indices), ("ligand", ligand_token_indices)):
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            raise ConfidenceError(
                f"{label} token indices {bad} are outside the PAE matrix (size {n})"
            )

    rows = np.asarray(motif_boltz_indices, dtype=int)
    cols = np.asarray(ligand_token_indices, dtype=int)
    block = pae[np.ix_(rows, cols)]
    block_t = pae[np.ix_(cols, rows)]
    return float((block.mean() + block_t.mean()) / 2.0)

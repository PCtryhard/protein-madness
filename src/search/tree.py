"""Per-scaffold elitist hill-climb: the frontier, the stopping rule, lineage.

Each seed scaffold is an independent search with its own frontier, stopping
condition and leaderboard. There is no competition between scaffolds, so
results stay comparable across seeds and the number of seeds can change
without altering behaviour.

A cycle expands the current frontier through both branches, scores everything,
and forms the next frontier from the best designs across the branches AND the
previous frontier together. Including the previous frontier is what guarantees
the best score never regresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RunCfg, SearchCfg


@dataclass(frozen=True)
class Candidate:
    """One scored design, as the search sees it.

    The search needs only an id, a score and enough to expand from; everything
    else lives in the database row.
    """

    design_id: str
    scaffold_id: str
    cycle: int
    branch: str                      # seed | wiggle | mutate
    sequence: str
    score: float
    gated: bool = False
    parent_id: str | None = None
    backbone_path: str | None = None   # RFdiffusion backbone, for the wiggle branch
    structure_path: str | None = None  # Boltz-2 prediction
    mutation: str | None = None

    def sort_key(self) -> tuple:
        # Deterministic: score descending, then design_id so ties never depend
        # on dict or filesystem ordering. A reproducible run needs this.
        return (-self.score, self.design_id)


@dataclass
class ScaffoldSearch:
    """The state of one scaffold's search. One of these per seed."""

    scaffold_id: str
    run: RunCfg
    search: SearchCfg
    cycle: int = 0
    frontier: list[Candidate] = field(default_factory=list)
    best_score: float = float("-inf")
    best_design_id: str | None = None
    cycles_without_improvement: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    history: list[dict] = field(default_factory=list)

    # ---- frontier ----

    def seed_frontier(self, candidates: list[Candidate]) -> None:
        """Install cycle 0's designs as the first frontier."""
        if not candidates:
            raise ValueError(
                f"scaffold {self.scaffold_id}: no cycle-0 designs survived; there "
                "is nothing to search from. Check the seed and the contig string."
            )
        self._update_best(candidates)
        self.frontier = self._select(candidates)

    def advance(self, produced: list[Candidate]) -> list[Candidate]:
        """Fold a cycle's output into the frontier and return the new frontier.

        `produced` is everything the branches made this cycle, already scored.
        The previous frontier is always a candidate for the next one, which is
        what makes the best score monotonic.
        """
        if self.stopped:
            raise RuntimeError(
                f"scaffold {self.scaffold_id} has stopped ({self.stop_reason}); "
                "advance() must not be called again"
            )

        pool = self._dedupe(produced + self.frontier)
        improvement = self._update_best(produced)
        self.frontier = self._select(pool)
        self.cycle += 1

        if improvement > self.run.min_delta:
            self.cycles_without_improvement = 0
        else:
            self.cycles_without_improvement += 1

        self.history.append(
            {
                "cycle": self.cycle,
                "produced": len(produced),
                "frontier": [c.design_id for c in self.frontier],
                "best_score": self.best_score,
                "best_design_id": self.best_design_id,
                "improvement": improvement,
                "cycles_without_improvement": self.cycles_without_improvement,
            }
        )

        self._check_stop()
        return list(self.frontier)

    def _select(self, pool: list[Candidate]) -> list[Candidate]:
        # Gated designs are kept in the database but never carried forward:
        # their score is zero by definition and they have no gradient to climb.
        viable = [c for c in pool if not c.gated]
        if not viable:
            self.stopped = True
            self.stop_reason = "every design in the pool was gated"
            return list(self.frontier)
        return sorted(viable, key=Candidate.sort_key)[: self.search.keep_per_cycle]

    @staticmethod
    def _dedupe(pool: list[Candidate]) -> list[Candidate]:
        seen: set[str] = set()
        out: list[Candidate] = []
        for c in pool:
            if c.design_id in seen:
                continue
            seen.add(c.design_id)
            out.append(c)
        return out

    def _update_best(self, candidates: list[Candidate]) -> float:
        """Record the best score seen and return this cycle's improvement."""
        viable = [c for c in candidates if not c.gated]
        if not viable:
            return 0.0
        best = max(viable, key=lambda c: (c.score, c.design_id))
        if best.score > self.best_score:
            improvement = (
                best.score - self.best_score
                if self.best_score != float("-inf")
                else float("inf")
            )
            self.best_score = best.score
            self.best_design_id = best.design_id
            return improvement
        return 0.0

    # ---- stopping ----

    def _check_stop(self) -> None:
        if self.cycles_without_improvement >= self.run.patience:
            self.stopped = True
            self.stop_reason = (
                f"no improvement above min_delta={self.run.min_delta} for "
                f"{self.cycles_without_improvement} consecutive cycles"
            )
            return
        if self.cycle >= self.run.max_cycles:
            self.stopped = True
            self.stop_reason = f"reached max_cycles={self.run.max_cycles}"

    # ---- parent selection for the branches ----

    def wiggle_parents(self) -> list[Candidate]:
        if not self.search.wiggle.enabled:
            return []
        return self._parents(self.search.wiggle.parents_per_cycle)

    def mutate_parents(self) -> list[Candidate]:
        if not self.search.mutate.enabled:
            return []
        return self._parents(self.search.mutate.parents_per_cycle)

    def _parents(self, count: int) -> list[Candidate]:
        """The frontier, cycled if the branch wants more parents than it holds.

        Cycling rather than truncating means a frontier of 2 can still feed a
        branch that wants 4 parents: each frontier member is expanded twice.
        With stochastic stages that is two different children, not a duplicate.
        """
        if not self.frontier:
            return []
        out: list[Candidate] = []
        while len(out) < count:
            out.extend(self.frontier)
        return out[:count]

"""High-level planning interface."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from .datasets import load_processed_routes, prepare_processed_data
from .formula import infer_target_class, parse_formula
from .mcts import MonteCarloTreeSearch
from .retrieval import RetrievalIndex
from .schema import PlannedRoute, PlanningProblem, PlanningState


class SynthesisPlanner:
    def __init__(self, data_dir: str | Path = "data/raw", processed_dir: str | Path = "data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)

    def ensure_processed_data(self) -> None:
        solid = self.processed_dir / "solid_state_routes.jsonl"
        solution = self.processed_dir / "solution_routes.jsonl"
        if not solid.exists() or not solution.exists():
            prepare_processed_data(self.data_dir, self.processed_dir)

    def plan(
        self,
        problem: PlanningProblem,
        iterations: int = 250,
        top_k: int = 5,
        exploration_constant: float = 1.4,
        rollout_count: int = 8,
        seed: int | None = None,
    ) -> list[PlannedRoute]:
        self.ensure_processed_data()
        routes = load_processed_routes(self.processed_dir, problem.modality)
        retrieval = RetrievalIndex(routes)
        analogs = retrieval.retrieve(problem.target_formula, top_k=12)
        candidate_precursor_sets = retrieval.candidate_precursor_sets(problem.target_formula, analogs, max_sets=12)

        root_state = PlanningState(
            problem=problem,
            target_elements=tuple(sorted(parse_formula(problem.target_formula))),
            target_class=infer_target_class(problem.target_formula),
        )
        mcts = MonteCarloTreeSearch(
            exploration_constant=exploration_constant,
            rollout_count=rollout_count,
            seed=seed,
        )
        root = mcts.run(root_state, analogs, candidate_precursor_sets, iterations=iterations)
        return _select_portfolio(root.terminal_routes, top_k)

    def save_routes(self, routes: list[PlannedRoute], output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = routes[0].target_formula if routes else "empty"
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{target}_{timestamp}.json"
        with path.open("w") as handle:
            json.dump([route.to_dict() for route in routes], handle, indent=2)
        return path


def _select_portfolio(routes: list[PlannedRoute], top_k: int) -> list[PlannedRoute]:
    ranked = sorted(routes, key=lambda route: route.score.total, reverse=True)
    portfolio = []
    seen_signatures = set()
    for route in ranked:
        signature = (
            tuple(sorted(precursor.formula for precursor in route.precursors)),
            tuple(operation.verb for operation in route.operations),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        portfolio.append(route)
        if len(portfolio) >= top_k:
            break
    return portfolio

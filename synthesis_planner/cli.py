"""CLI for downloading data, preparing indices, and planning routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .datasets import download_public_datasets, prepare_processed_data
from .planner import SynthesisPlanner
from .schema import PlanningProblem


def load_config(config_path: str = "config.json") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    config = config or {}
    parser = argparse.ArgumentParser(description="Target-conditioned MCTS synthesis planning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download the public synthesis datasets")
    download.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))

    prepare = subparsers.add_parser("prepare-data", help="Normalize the raw datasets into route records")
    prepare.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))
    prepare.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))

    plan = subparsers.add_parser("plan", help="Plan solid-state synthesis routes for a target")
    plan.add_argument("--target", required=True, help="Target formula, for example BaTiO3")
    plan.add_argument("--modality", choices=["solid_state"], default=config.get("modality", "solid_state"))
    plan.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))
    plan.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
    plan.add_argument("--iterations", type=int, default=config.get("iterations", 250))
    plan.add_argument("--top-k", type=int, default=config.get("top_k", 5))
    plan.add_argument("--exploration-constant", type=float, default=config.get("exploration_constant", 1.4))
    plan.add_argument("--rollout-count", type=int, default=config.get("rollout_count", 8))
    plan.add_argument("--seed", type=int, default=config.get("seed"))
    plan.add_argument("--output-dir", default="planning_results")

    return parser


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    args = build_parser(config).parse_args(argv)

    if args.command == "download-data":
        destinations = download_public_datasets(args.data_dir)
        for key, path in destinations.items():
            print(f"{key}: {path}")
        return 0

    if args.command == "prepare-data":
        outputs = prepare_processed_data(args.data_dir, args.processed_dir)
        for key, path in outputs.items():
            print(f"{key}: {path}")
        return 0

    if args.command == "plan":
        planner = SynthesisPlanner(args.data_dir, args.processed_dir)
        routes = planner.plan(
            PlanningProblem(target_formula=args.target, modality=args.modality),
            iterations=args.iterations,
            top_k=args.top_k,
            exploration_constant=args.exploration_constant,
            rollout_count=args.rollout_count,
            seed=args.seed,
        )
        if not routes:
            print("No routes generated.")
            return 1

        output_path = planner.save_routes(routes, args.output_dir)
        for idx, route in enumerate(routes, start=1):
            precursors = ", ".join(precursor.formula for precursor in route.precursors)
            operations = " -> ".join(operation.source_label or operation.verb for operation in route.operations)
            print(f"Rank {idx}: score={route.score.total:.3f}")
            print(f"  Precursors: {precursors}")
            print(f"  Operations: {operations}")
            print(f"  Evidence: {', '.join(route.evidence_dois[:3]) if route.evidence_dois else 'n/a'}")
            if route.judge.notes:
                print(f"  Notes: {route.judge.notes[0]}")
        print(f"Saved: {output_path}")
        return 0

    return 1

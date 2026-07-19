"""
Tests for post-run analysis (metrics, ranking, text report).

Pure logic; uses the same toy integer-line material space as the MCTS tests.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Dict, List

import pytest

from mcts_framework.core.material import Material
from mcts_framework.core.move_generator import MoveGenerator
from mcts_framework.core.evaluator import PropertyEvaluator
from mcts_framework.core.reward import RewardFunction
from mcts_framework.core.selection import UCB1
from mcts_framework.core.mcts import MCTS
from mcts_framework.viz import analysis


class IntMaterial(Material):
    def __init__(self, value: int):
        self.value = value

    def get_identifier(self) -> str:
        return str(self.value)

    def copy(self):
        return IntMaterial(self.value)


class LineMoves(MoveGenerator):
    def generate_moves(self, material):
        return [IntMaterial(material.value + 1), IntMaterial(material.value - 1)]


class DistEval(PropertyEvaluator):
    def __init__(self, target: int):
        super().__init__()
        self.target = target

    async def _compute(self, material) -> Dict[str, float]:
        return {"distance": float(abs(material.value - self.target))}


class NegDist(RewardFunction):
    def compute_reward(self, properties: Dict[str, float]) -> float:
        return -properties["distance"]

    def get_property_names(self) -> List[str]:
        return ["distance"]


async def _run(iterations=100, target=5):
    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(),
        property_evaluator=DistEval(target),
        reward_function=NegDist(),
        selection_strategy=UCB1(),
        n_rollout=1,
        rollout_depth=0,
        seed=0,
    )
    await mcts.run(iterations=iterations)
    return mcts


# --- compute_metrics -----------------------------------------------------


@pytest.mark.asyncio
async def test_compute_metrics_fields():
    mcts = await _run()
    m = analysis.compute_metrics(mcts)
    assert set(m) == {
        "iterations", "unique_materials", "tree_size", "evaluated_nodes",
        "max_depth", "best_reward", "best_material", "efficiency",
    }


@pytest.mark.asyncio
async def test_compute_metrics_values():
    mcts = await _run(iterations=100, target=5)
    m = analysis.compute_metrics(mcts)
    assert m["best_material"] == "5"
    assert m["best_reward"] == 0.0
    assert m["max_depth"] >= 5           # must descend at least to reach "5"
    assert 0.0 <= m["efficiency"] <= 1.0
    assert m["evaluated_nodes"] > 0


# --- rank_materials ------------------------------------------------------


@pytest.mark.asyncio
async def test_rank_materials_sorted_and_includes_properties():
    mcts = await _run()
    ranked = analysis.rank_materials(mcts, n=5)
    assert len(ranked) <= 5
    rewards = [r["own_reward"] for r in ranked]
    assert rewards == sorted(rewards, reverse=True)
    # Properties (distance) are merged into each row.
    assert "distance" in ranked[0]
    assert ranked[0]["identifier"] == "5"


# --- generate_report -----------------------------------------------------


@pytest.mark.asyncio
async def test_generate_report_contains_key_sections():
    mcts = await _run()
    report = analysis.generate_report(mcts, top_n=5)
    assert "MCTS Materials Search Report" in report
    assert "Search metrics" in report
    assert "Best material" in report
    assert "5" in report  # the optimum shows up


@pytest.mark.asyncio
async def test_generate_report_handles_no_evaluations():
    """A never-run search still produces a report without crashing."""
    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(),
        property_evaluator=DistEval(5),
        reward_function=NegDist(),
        selection_strategy=UCB1(),
    )
    # No run() call -> no evaluated nodes.
    report = analysis.generate_report(mcts)
    assert "no materials evaluated" in report
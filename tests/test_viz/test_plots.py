"""
Tests for plotting functions.

Requires matplotlib (skipped if absent); uses the non-interactive Agg backend
so no display is needed. networkx is required for the tree plot. Tests check
that each function returns a Figure and can save to disk, not pixel output.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Dict, List

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless backend; must be set before pyplot import

from mcts_framework.core.material import Material  # noqa: E402
from mcts_framework.core.move_generator import MoveGenerator  # noqa: E402
from mcts_framework.core.evaluator import PropertyEvaluator  # noqa: E402
from mcts_framework.core.reward import RewardFunction  # noqa: E402
from mcts_framework.core.selection import UCB1  # noqa: E402
from mcts_framework.core.mcts import MCTS  # noqa: E402
from mcts_framework.viz import plots  # noqa: E402


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
    async def _compute(self, material) -> Dict[str, float]:
        return {"distance": float(abs(material.value - 5))}


class NegDist(RewardFunction):
    def compute_reward(self, properties: Dict[str, float]) -> float:
        return -properties["distance"]

    def get_property_names(self) -> List[str]:
        return ["distance"]


async def _run(iterations=80):
    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(),
        property_evaluator=DistEval(),
        reward_function=NegDist(),
        selection_strategy=UCB1(),
        n_rollout=1,
        rollout_depth=0,
        seed=0,
    )
    await mcts.run(iterations=iterations)
    return mcts


@pytest.mark.asyncio
async def test_plot_convergence_returns_figure(tmp_path):
    mcts = await _run()
    out = tmp_path / "conv.png"
    fig = plots.plot_convergence(mcts, save_path=str(out))
    assert fig is not None
    assert out.exists()


@pytest.mark.asyncio
async def test_plot_property_distribution(tmp_path):
    mcts = await _run()
    out = tmp_path / "dist.png"
    fig = plots.plot_property_distribution(mcts, "distance", save_path=str(out))
    assert fig is not None
    assert out.exists()


@pytest.mark.asyncio
async def test_plot_property_distribution_missing_property():
    mcts = await _run()
    with pytest.raises(ValueError):
        plots.plot_property_distribution(mcts, "nonexistent_property")


@pytest.mark.asyncio
async def test_plot_search_tree_returns_figure(tmp_path):
    pytest.importorskip("networkx")
    mcts = await _run()
    out = tmp_path / "tree.png"
    fig = plots.plot_search_tree(mcts, save_path=str(out))
    assert fig is not None
    assert out.exists()


@pytest.mark.asyncio
async def test_plot_search_tree_truncation(tmp_path):
    """max_nodes cap is honored and reflected without crashing."""
    pytest.importorskip("networkx")
    mcts = await _run(iterations=120)
    out = tmp_path / "tree_trunc.png"
    fig = plots.plot_search_tree(mcts, save_path=str(out), max_nodes=10)
    assert fig is not None
    assert out.exists()
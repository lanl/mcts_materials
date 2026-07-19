"""
Unit tests for the core MCTS algorithm.

Uses a tiny integer-line material space so behavior is fully predictable:
a material is an integer; moves are +1/-1; reward peaks at a target value.

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


# --- Tiny toy material space ---------------------------------------------


class IntMaterial(Material):
    """Material wrapping a single integer on a line."""

    def __init__(self, value: int):
        self.value = value

    def get_identifier(self) -> str:
        return str(self.value)

    def copy(self) -> "IntMaterial":
        return IntMaterial(self.value)


class LineMoves(MoveGenerator[IntMaterial]):
    """Moves: step +1 or -1, clamped to [lo, hi]."""

    def __init__(self, lo: int = -20, hi: int = 20):
        self.lo = lo
        self.hi = hi

    def generate_moves(self, material: IntMaterial) -> List[IntMaterial]:
        out = []
        if material.value + 1 <= self.hi:
            out.append(IntMaterial(material.value + 1))
        if material.value - 1 >= self.lo:
            out.append(IntMaterial(material.value - 1))
        return out


class DistanceEvaluator(PropertyEvaluator):
    """Property = distance to a target integer."""

    def __init__(self, target: int):
        super().__init__()
        self.target = target
        self.compute_calls = 0

    async def _compute(self, material: IntMaterial) -> Dict[str, float]:
        self.compute_calls += 1
        return {"distance": abs(material.value - self.target)}


class NegDistanceReward(RewardFunction):
    """Reward = -distance (maximized at the target)."""

    def compute_reward(self, properties: Dict[str, float]) -> float:
        return -properties["distance"]

    def get_property_names(self) -> List[str]:
        return ["distance"]


def make_mcts(target: int = 5, start: int = 0, **kwargs) -> MCTS:
    return MCTS(
        root_material=IntMaterial(start),
        move_generator=LineMoves(),
        property_evaluator=DistanceEvaluator(target),
        reward_function=NegDistanceReward(),
        selection_strategy=UCB1(),
        seed=kwargs.pop("seed", 0),
        **kwargs,
    )


# --- Basic smoke / convergence -------------------------------------------


@pytest.mark.asyncio
async def test_runs_without_error():
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=50)
    assert mcts.iteration >= 0
    assert len(mcts.reward_history) > 0


@pytest.mark.asyncio
async def test_finds_target():
    """MCTS should discover the target integer given enough iterations."""
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=200)
    # Best reward is 0 (distance 0) at the target.
    assert mcts.best_reward == 0.0
    assert mcts.best_node is not None
    assert mcts.best_node.material.get_identifier() == "5"


@pytest.mark.asyncio
async def test_history_lengths_match_iterations():
    mcts = make_mcts(n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=30)
    assert len(mcts.reward_history) == len(mcts.unique_materials_history)


# --- Deduplication: reserve-on-attach ------------------------------------


@pytest.mark.asyncio
async def test_no_duplicate_materials_in_tree():
    """Every material identifier appears at most once across the whole tree."""
    mcts = make_mcts(target=8, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=300)

    ids = [nd.material.get_identifier() for nd in mcts.all_nodes()]
    assert len(ids) == len(set(ids)), "Duplicate material found in tree"


@pytest.mark.asyncio
async def test_visited_matches_tree_nodes():
    """visited_materials should equal the set of identifiers in the tree."""
    mcts = make_mcts(target=8, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=300)

    tree_ids = {nd.material.get_identifier() for nd in mcts.all_nodes()}
    assert tree_ids == mcts.visited_materials


@pytest.mark.asyncio
async def test_expand_skips_already_claimed_material():
    """
    A candidate generated earlier but attached elsewhere since must be
    skipped rather than duplicated.

    From root 0, both +1 and -1 are candidates. We prime the situation by
    manually reserving "1" (as if another path attached it), then expand and
    confirm root never attaches a second "1".
    """
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0)

    # Force generation of root's pending list.
    child_a = await mcts._expand(mcts.root)
    assert child_a is not None

    # Simulate another path claiming the remaining pending candidate.
    for pending in list(mcts.root.pending_children):
        mcts.visited_materials.add(pending.get_identifier())

    # Next expand should skip the now-claimed candidate(s) and attach nothing.
    child_b = await mcts._expand(mcts.root)
    assert child_b is None

    ids = [nd.material.get_identifier() for nd in mcts.all_nodes()]
    assert len(ids) == len(set(ids))


# --- Caching -------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_caches_repeated_materials():
    """The evaluator should not recompute the same material twice."""
    evaluator = DistanceEvaluator(target=5)
    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(),
        property_evaluator=evaluator,
        reward_function=NegDistanceReward(),
        selection_strategy=UCB1(),
        n_rollout=1,
        rollout_depth=0,
        seed=0,
    )
    await mcts.run(iterations=200)

    # Unique materials evaluated should be <= number of distinct integers,
    # and compute_calls should equal the number of unique cache entries.
    assert evaluator.compute_calls == len(evaluator.cache)


# --- Rollout discount ----------------------------------------------------


@pytest.mark.asyncio
async def test_rollout_depth_zero_is_pure_node_value():
    """With rollout_depth=0, reward equals the node's own value."""
    mcts = make_mcts(target=0, start=3, n_rollout=1, rollout_depth=0)
    reward = await mcts._simulate(mcts.root)
    # distance from 3 to target 0 is 3 -> reward -3
    assert reward == -3.0


# --- Results helpers -----------------------------------------------------


@pytest.mark.asyncio
async def test_get_best_materials_sorted():
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=200)

    top = mcts.get_best_materials(n=5)
    assert len(top) <= 5
    # get_best_materials ranks by own_reward (the node's own material value),
    # not subtree_best (a backprop-accumulated subtree maximum).
    rewards = [nd.own_reward for nd in top]
    assert rewards == sorted(rewards, reverse=True)


@pytest.mark.asyncio
async def test_best_material_is_not_the_root():
    """
    Regression: internal nodes accumulate a subtree-best score `subtree_best` during
    backprop, so ranking by subtree_best would float the root (and other
    ancestors of the optimum) to the top. get_best_materials must rank by
    own_reward, so the true optimum - not the root - comes first.
    """
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=200)

    top = mcts.get_best_materials(n=1)
    assert top[0].material.get_identifier() == "5"
    assert top[0].own_reward == 0.0

    # The root reached the optimum in its subtree (subtree_best == 0.0 via
    # backprop), but the root is the fixed starting point - it is never
    # simulated itself, so own_reward stays None and it is excluded from the
    # ranking entirely. Ranking by subtree_best would instead have floated it
    # to the top.
    assert mcts.root.subtree_best == 0.0
    assert mcts.root.own_reward is None
    assert mcts.root not in top


@pytest.mark.asyncio
async def test_summary_fields():
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0)
    await mcts.run(iterations=100)
    s = mcts.summary()
    assert set(s) == {
        "iterations",
        "best_reward",
        "best_material",
        "unique_materials",
        "tree_size",
    }
    assert s["best_material"] == "5"


@pytest.mark.asyncio
async def test_best_node_and_reward_are_consistent():
    """
    Regression (bug #1): the global best_reward must equal best_node's OWN
    reward, never a discounted rollout sample of a different material. Use
    rollout_depth>0 and n_rollout>1 so rollout samples are actually drawn -
    the old code could set best_reward from one of those.
    """
    mcts = make_mcts(target=8, start=0, n_rollout=4, rollout_depth=3)
    await mcts.run(iterations=200)

    assert mcts.best_node is not None
    # best_reward is exactly best_node's own evaluated reward (self-consistent).
    assert mcts.best_reward == mcts.best_node.own_reward
    # And that reward corresponds to the reported best material.
    assert mcts.summary()["best_material"] == mcts.best_node.material.get_identifier()

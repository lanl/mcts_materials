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


# --- Rollout depth -------------------------------------------------------


@pytest.mark.asyncio
async def test_rollout_depth_zero_is_pure_node_value():
    """With rollout_depth=0, reward equals the node's own value."""
    mcts = make_mcts(target=0, start=3, n_rollout=1, rollout_depth=0)
    reward = await mcts._simulate(mcts.root)
    # distance from 3 to target 0 is 3 -> reward -3
    assert reward == -3.0


@pytest.mark.asyncio
async def test_n_rollout_counts_walks_not_off_by_one():
    """
    Regression: n_rollout is the number of rollout WALKS drawn, additional to
    the mandatory depth-0 self-evaluation. n_rollout=1 must draw one walk (the
    old code ran the loop n_rollout-1 times, so n_rollout=1 drew ZERO walks and
    rollout_depth was silently ignored). We count _rollout_sample calls.
    """
    for n in (0, 1, 3):
        mcts = make_mcts(target=0, start=5, n_rollout=n, rollout_depth=4, seed=0)
        walks = {"n": 0}
        orig = mcts._rollout_sample

        async def wrapped(material, _orig=orig, _c=walks):
            _c["n"] += 1
            return await _orig(material)

        mcts._rollout_sample = wrapped
        await mcts._simulate(mcts.root)
        assert walks["n"] == n, f"n_rollout={n} should draw {n} walks, drew {walks['n']}"


@pytest.mark.asyncio
async def test_rollout_depth_zero_skips_walks():
    """
    With rollout_depth == 0 a walk can take no moves and only re-evaluates the
    node itself, so it cannot change the aggregate. _simulate must skip rollout
    sampling entirely (no _rollout_sample calls) even when n_rollout > 0, and
    the value is just the node's own reward.
    """
    mcts = make_mcts(target=0, start=3, n_rollout=5, rollout_depth=0, seed=0)
    walks = {"n": 0}
    orig = mcts._rollout_sample

    async def wrapped(material, _orig=orig, _c=walks):
        _c["n"] += 1
        return await _orig(material)

    mcts._rollout_sample = wrapped
    reward = await mcts._simulate(mcts.root)
    assert walks["n"] == 0  # no walks drawn despite n_rollout=5
    assert reward == -3.0   # value is just own_reward (distance 3 -> -3)


# --- rollout_aggregation + max-along-walk --------------------------------


def test_invalid_rollout_aggregation_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        make_mcts(rollout_aggregation="median")


@pytest.mark.asyncio
async def test_own_reward_is_depth0_regardless_of_aggregation():
    """own_reward is always the undiscounted depth-0 value, for max and mean."""
    for agg in ("max", "mean"):
        mcts = make_mcts(target=0, start=3, n_rollout=5, rollout_depth=2,
                         rollout_aggregation=agg)
        await mcts._simulate(mcts.root)
        # start=3, target=0 -> own distance 3 -> own_reward -3
        assert mcts.root.own_reward == -3.0


@pytest.mark.asyncio
async def test_max_along_walk_returns_best_step():
    """
    A depth>0 rollout scores every step and returns the max. From start=3
    toward target=0, a walk can reach 0 (reward 0) within a couple of steps;
    max-along-walk should surface a step reward >= the endpoint's.
    """
    mcts = make_mcts(target=0, start=3, n_rollout=1, rollout_depth=5,
                     rollout_aggregation="max", seed=1)
    # _rollout_sample returns the max reward along one random walk.
    r = await mcts._rollout_sample(mcts.root.material)
    # Rewards are -distance <= 0; best possible along a walk that can reach 0.
    assert r <= 0.0
    # Over several walks, at least one should hit the target (reward 0).
    best = float("-inf")
    for _ in range(20):
        best = max(best, await mcts._rollout_sample(mcts.root.material))
    assert best == 0.0


@pytest.mark.asyncio
async def test_mean_aggregation_differs_from_max():
    """
    Under 'mean', _simulate returns the plain average of all samples (own +
    undiscounted extras); under 'max' it returns the maximum. On the same
    seed/state the mean must be <= the max, and typically strictly less when
    the samples aren't all equal.
    """
    common = dict(target=0, start=5, n_rollout=6, rollout_depth=2, seed=0)

    mcts_mean = make_mcts(rollout_aggregation="mean", **common)
    mean_val = await mcts_mean._simulate(mcts_mean.root)

    mcts_max = make_mcts(rollout_aggregation="max", **common)
    max_val = await mcts_max._simulate(mcts_max.root)

    assert mean_val <= max_val
    # own_reward (depth-0) is identical regardless of aggregation.
    assert mcts_mean.root.own_reward == mcts_max.root.own_reward == -5.0


# --- search_mode (fast vs thorough) --------------------------------------


def test_invalid_search_mode_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        make_mcts(search_mode="turbo")


@pytest.mark.asyncio
async def test_fast_mode_stops_when_root_converges():
    """In 'fast' mode the run stops once the root self-terminates (its
    no-improvement countdown fires), before the full iteration budget."""
    mcts = make_mcts(target=5, start=0, n_rollout=1, rollout_depth=0,
                     termination_limit=10, search_mode="fast")
    await mcts.run(iterations=2000)
    assert mcts.root.terminated
    assert mcts.iteration + 1 < 2000  # stopped early


@pytest.mark.asyncio
async def test_thorough_mode_ignores_root_convergence():
    """In 'thorough' mode root self-termination does NOT stop the run; it uses
    the full budget (unless the space is genuinely exhausted). With the same
    settings that make 'fast' stop early, 'thorough' runs longer and visits
    at least as many compounds."""
    common = dict(target=5, start=0, n_rollout=1, rollout_depth=0,
                  termination_limit=10)
    fast = make_mcts(search_mode="fast", **common)
    await fast.run(iterations=500)
    thorough = make_mcts(search_mode="thorough", **common)
    await thorough.run(iterations=500)

    assert thorough.iteration + 1 > fast.iteration + 1
    assert len(thorough.visited_materials) >= len(fast.visited_materials)


@pytest.mark.asyncio
async def test_thorough_mode_still_stops_on_true_exhaustion():
    """A tiny bounded space is fully reachable; 'thorough' must still stop once
    every branch is exhausted rather than spin forever."""
    # Line clamped to [0, 3]: only 4 compounds exist.
    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(lo=0, hi=3),
        property_evaluator=DistanceEvaluator(target=2),
        reward_function=NegDistanceReward(),
        selection_strategy=UCB1(),
        termination_limit=5,
        n_rollout=1, rollout_depth=0,
        search_mode="thorough", seed=0,
    )
    await mcts.run(iterations=100000)
    # Stopped on exhaustion well before the huge budget, having found all 4.
    assert mcts.terminated
    assert mcts.iteration + 1 < 100000
    assert len(mcts.visited_materials) == 4


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
    reward, never a rollout sample of a different material. Use rollout_depth>0
    and n_rollout>=1 so rollout samples are actually drawn - the old code could
    set best_reward from one of those.
    """
    mcts = make_mcts(target=8, start=0, n_rollout=4, rollout_depth=3)
    await mcts.run(iterations=200)

    assert mcts.best_node is not None
    # best_reward is exactly best_node's own evaluated reward (self-consistent).
    assert mcts.best_reward == mcts.best_node.own_reward
    # And that reward corresponds to the reported best material.
    assert mcts.summary()["best_material"] == mcts.best_node.material.get_identifier()

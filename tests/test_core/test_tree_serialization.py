"""
Tests for MCTS tree serialization (to_tree_dict / save_tree_json).

The serialized tree must capture structure (parent links, root id 0) plus each
node's identifier, stats, and properties, and round-trip through JSON - it is
what the radial search-tree figure reads offline.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
from typing import Dict, List

import pytest

from mcts_framework.core.evaluator import PropertyEvaluator
from mcts_framework.core.material import Material
from mcts_framework.core.mcts import MCTS
from mcts_framework.core.move_generator import MoveGenerator
from mcts_framework.core.reward import RewardFunction
from mcts_framework.core.selection import UCB1


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


async def _run(iterations=60):
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
async def test_tree_dict_structure_and_root():
    mcts = await _run()
    tree = mcts.to_tree_dict()

    assert tree["root_id"] == 0
    assert len(tree["nodes"]) == len(mcts.all_nodes())
    root = tree["nodes"][0]
    assert root["parent"] is None
    assert root["identifier"] == "0"
    # Every non-root node references a valid parent id; ids are unique.
    ids = {rec["id"] for rec in tree["nodes"]}
    assert len(ids) == len(tree["nodes"])
    for rec in tree["nodes"][1:]:
        assert rec["parent"] in ids


@pytest.mark.asyncio
async def test_tree_dict_is_json_safe_and_carries_stats():
    mcts = await _run()
    tree = mcts.to_tree_dict()

    # Round-trips through JSON (no inf/NaN/objects).
    reloaded = json.loads(json.dumps(tree))
    assert reloaded == tree

    # subtree_best of -inf is normalized to None (JSON-safe).
    for rec in tree["nodes"]:
        assert rec["subtree_best"] is None or isinstance(rec["subtree_best"], float)
    # An evaluated node carries its properties.
    evaluated = [r for r in tree["nodes"] if r["own_reward"] is not None]
    assert evaluated and "distance" in evaluated[0]["properties"]


@pytest.mark.asyncio
async def test_save_tree_json_writes_file(tmp_path):
    mcts = await _run()
    path = tmp_path / "tree.json"
    mcts.save_tree_json(str(path))
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["nodes"] and data["root_id"] == 0

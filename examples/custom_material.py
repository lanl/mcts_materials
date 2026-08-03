"""
Example: define a brand-new material type and search it with the framework.

This uses NO heavy dependencies - it demonstrates the four interfaces you
implement to plug any material into the MCTS core:

    Material          - what a candidate is + a unique identifier
    MoveGenerator     - how to enumerate neighbors (expansion)
    PropertyEvaluator - how to compute properties (simulation)
    RewardFunction    - how to score properties

Here the "material" is just an integer on a line, and the goal is to find the
integer closest to a target. Swap these four classes for crystal/molecule/
alloy implementations and the same MCTS core drives the search unchanged.

Run:  python examples/custom_material.py

© 2026. Triad National Security, LLC. All rights reserved.
"""

import asyncio
from typing import Dict, List

from mcts_framework import (
    Material,
    MoveGenerator,
    PropertyEvaluator,
    RewardFunction,
    MCTS,
    UCB1,
)


class IntPoint(Material):
    """A candidate is a single integer on the number line."""

    def __init__(self, value: int):
        self.value = value

    def get_identifier(self) -> str:
        return str(self.value)

    def copy(self) -> "IntPoint":
        return IntPoint(self.value)


class StepMoves(MoveGenerator[IntPoint]):
    """Neighbors are +1 and -1 (clamped to a range)."""

    def __init__(self, lo: int = -50, hi: int = 50):
        self.lo, self.hi = lo, hi

    def generate_moves(self, material: IntPoint) -> List[IntPoint]:
        out = []
        if material.value + 1 <= self.hi:
            out.append(IntPoint(material.value + 1))
        if material.value - 1 >= self.lo:
            out.append(IntPoint(material.value - 1))
        return out


class DistanceEvaluator(PropertyEvaluator):
    """Property = absolute distance to the target integer."""

    def __init__(self, target: int):
        super().__init__()
        self.target = target

    async def _compute(self, material: IntPoint) -> Dict[str, float]:
        return {"distance": float(abs(material.value - self.target))}


class ClosenessReward(RewardFunction):
    """Reward = -distance (maximized when we hit the target)."""

    def compute_reward(self, properties: Dict[str, float]) -> float:
        return -properties["distance"]

    def get_property_names(self) -> List[str]:
        return ["distance"]


async def main() -> None:
    target = 17
    mcts = MCTS(
        root_material=IntPoint(0),
        move_generator=StepMoves(),
        property_evaluator=DistanceEvaluator(target),
        reward_function=ClosenessReward(),
        selection_strategy=UCB1(),
        n_rollout=1,
        rollout_depth=0,
        seed=0,
    )

    await mcts.run(iterations=300)

    print("Target:", target)
    print("Summary:", mcts.summary())
    print("\nTop 5 discovered:")
    for node in mcts.get_best_materials(n=5):
        print(f"  {node.material.get_identifier():>4}  "
              f"reward={node.own_reward:.1f}  visits={node.visits}")


if __name__ == "__main__":
    asyncio.run(main())
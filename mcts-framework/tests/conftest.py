"""
Shared pytest fixtures for MCTS framework tests.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from typing import Dict, List

import pytest

from mcts_framework.core.evaluator import PropertyEvaluator
from mcts_framework.core.material import Material
from mcts_framework.core.move_generator import MoveGenerator
from mcts_framework.core.reward import RewardFunction


class SimpleMaterial(Material):
    """Simple material implementation for testing."""

    def __init__(self, name: str):
        self.name = name

    def get_identifier(self) -> str:
        return self.name

    def copy(self) -> 'SimpleMaterial':
        return SimpleMaterial(self.name)


# --- Shared toy MCTS pieces (integer-line search) ------------------------
#
# A minimal, dependency-free MCTS problem reused across test modules (core,
# CLI). Defined in conftest so every module can import them without depending
# on `tests` being an importable package (which is env/rootdir-fragile).


class IntMaterial(Material):
    """Material wrapping a single integer on a line."""

    def __init__(self, value: int):
        self.value = value

    def get_identifier(self) -> str:
        return str(self.value)

    def copy(self) -> "IntMaterial":
        return IntMaterial(self.value)


class LineMoves(MoveGenerator["IntMaterial"]):
    """Moves: step +1 or -1, clamped to [lo, hi]."""

    def __init__(self, lo: int = -20, hi: int = 20):
        self.lo = lo
        self.hi = hi

    def generate_moves(self, material: "IntMaterial") -> List["IntMaterial"]:
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

    async def _compute(self, material: "IntMaterial") -> Dict[str, float]:
        self.compute_calls += 1
        return {"distance": abs(material.value - self.target)}


class NegDistanceReward(RewardFunction):
    """Reward = -distance (maximized at the target)."""

    def compute_reward(self, properties: Dict[str, float]) -> float:
        return -properties["distance"]

    def get_property_names(self) -> List[str]:
        return ["distance"]


@pytest.fixture
def simple_material():
    """Create a simple test material."""
    return SimpleMaterial("test_material")


@pytest.fixture
def simple_materials():
    """Create multiple simple test materials."""
    return [SimpleMaterial(f"material_{i}") for i in range(5)]

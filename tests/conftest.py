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


# --- Superhydride template builder ---------------------------------------
#
# Structures for the superhydride tests are built programmatically so that
# suite needs no data files. ASE is imported inside the builder, not at module
# scope, so the tests that need no structures still run without it.
#
# These live here rather than in a tests/test_superhydride/conftest.py because
# `tests` is not an importable package: a second conftest.py would be bound to
# the same module name and shadow this one (which test_cli imports by name).


def build_superhydride_template(host_a: str = "La", host_b: str = "Be", a: float = 5.0):
    """
    Return an ASE Atoms for a synthetic cubic XYH8 cell.

    Host A sits at the corner, host B at the body centre, and eight hydrogens
    at the (1/4, 3/4)^3 positions around them. This is the shape of a ternary
    superhydride template - two distinct host sites and a hydrogen sublattice -
    not a physical claim about any particular compound.
    """
    from ase import Atoms

    positions = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    symbols = [host_a, host_b]
    for x in (0.25, 0.75):
        for y in (0.25, 0.75):
            for z in (0.25, 0.75):
                positions.append((x, y, z))
                symbols.append("H")
    return Atoms(symbols=symbols, scaled_positions=positions, cell=[a, a, a], pbc=True)


@pytest.fixture
def make_superhydride_template():
    """The template builder itself, for tests that need several compositions."""
    return build_superhydride_template


@pytest.fixture
def make_superhydride_structure():
    """Factory returning a SuperhydrideStructure for a given host pair."""
    from mcts_framework.superhydride import SuperhydrideStructure

    def _make(host_a: str = "La", host_b: str = "Be", a: float = 5.0):
        return SuperhydrideStructure(build_superhydride_template(host_a, host_b, a))

    return _make


@pytest.fixture
def superhydride_template(make_superhydride_structure):
    """The default LaBeH8-shaped template."""
    return make_superhydride_structure()

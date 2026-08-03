"""
MCTS Framework for Materials Discovery.

A clean, modular Monte Carlo Tree Search implementation for discovering
optimal materials (crystals, molecules, alloys, etc.).

© 2026. Triad National Security, LLC. All rights reserved.
"""

__version__ = "0.1.0"

# Core abstractions and algorithm
from .core.material import Material
from .core.move_generator import MoveGenerator
from .core.evaluator import PropertyEvaluator
from .core.reward import RewardFunction
from .core.search_node import SearchNode
from .core.selection import (
    SelectionStrategy,
    UCB1,
    PUCT,
    EpsilonGreedy,
    Boltzmann,
    create_selection_strategy,
)
from .core.mcts import MCTS

__all__ = [
    "Material",
    "MoveGenerator",
    "PropertyEvaluator",
    "RewardFunction",
    "SearchNode",
    "SelectionStrategy",
    "UCB1",
    "PUCT",
    "EpsilonGreedy",
    "Boltzmann",
    "create_selection_strategy",
    "MCTS",
]

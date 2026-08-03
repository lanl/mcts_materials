"""
Intermetallic crystal structure support for MCTS materials search.

Ports the validated periodic-table substitution logic from the original
mcts_crystal codebase into the modular framework interfaces.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .structure import IntermetallicStructure
from .moves import PeriodicTableMoves
from .doscar import DoscarRewardLookup
from .rewards import (
    EhullReward,
    RdosReward,
    EhullRdosReward,
    EhullRdosProductReward,
    ehull_reward,
    create_intermetallic_reward,
)
from .evaluator import MaceEvaluator, UnstablePenalty

__all__ = [
    "IntermetallicStructure",
    "PeriodicTableMoves",
    "DoscarRewardLookup",
    "EhullReward",
    "RdosReward",
    "EhullRdosReward",
    "EhullRdosProductReward",
    "ehull_reward",
    "create_intermetallic_reward",
    "MaceEvaluator",
    "UnstablePenalty",
]
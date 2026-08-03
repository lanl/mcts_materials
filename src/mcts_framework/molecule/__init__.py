"""
Molecular structure support for MCTS materials search.

Integrates the molecule-modifier package for functional-group expansion and
ML property prediction (melting point, H2 capacity, synthesizability).

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .structure import MolecularStructure
from .moves import FunctionalGroupMoves
from .evaluator import MoleculeEvaluator
from .functional_groups import DEFAULT_FUNCTIONAL_GROUPS, default_functional_groups
from .rewards import (
    MeltingPointReward,
    H2CapacityReward,
    SynthesizabilityReward,
    MultiObjectiveReward,
    create_molecule_reward,
)

__all__ = [
    "MolecularStructure",
    "FunctionalGroupMoves",
    "MoleculeEvaluator",
    "DEFAULT_FUNCTIONAL_GROUPS",
    "default_functional_groups",
    "MeltingPointReward",
    "H2CapacityReward",
    "SynthesizabilityReward",
    "MultiObjectiveReward",
    "create_molecule_reward",
]
"""
Abstract base class for generating child materials.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

from .material import Material

M = TypeVar('M', bound=Material)


class MoveGenerator(ABC, Generic[M]):
    """
    Abstract base class for generating child materials.

    Given a parent material, generates a list of "neighboring" materials
    that can be reached in one step (e.g., element substitution, functional
    group addition, composition change).

    Type Parameters:
        M: Material type (IntermetallicStructure, MolecularStructure, etc.)

    Examples:
        - Crystals: PeriodicTableMoves - substitute one element (Ti→V, Si→Ge)
        - Molecules: FunctionalGroupMoves - add one functional group
        - Alloys: CompositionMoves - change composition by ±5%
    """

    @abstractmethod
    def generate_moves(self, material: M) -> List[M]:
        """
        Generate all possible single-step moves from the given material.

        Args:
            material: Parent material

        Returns:
            List of child materials. May include duplicates (will be
            deduplicated by MCTS using material.get_identifier()).

        Notes:
            - This method should be deterministic for reproducibility
            - Invalid materials can be filtered via filter_invalid()
            - Empty list indicates no moves possible (dead end)
        """
        pass

    def filter_invalid(self, materials: List[M]) -> List[M]:
        """
        Optional: filter out invalid materials.

        Override this method to apply chemical/physical constraints.
        Default implementation returns all materials unchanged.

        Args:
            materials: List of candidate materials

        Returns:
            Filtered list of valid materials

        Examples:
            - Remove unstable structures
            - Remove chemically impossible combinations
            - Remove materials violating constraints
        """
        return materials

"""
Abstract base class for materials being searched.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Any


class Material(ABC):
    """
    Abstract base class for materials being searched.

    A Material represents a candidate structure (crystal, molecule, alloy, etc.)
    and provides a unique identifier for deduplication and caching.

    Examples:
        - Crystals: IntermetallicStructure wrapping ASE Atoms
        - Molecules: MolecularStructure wrapping RDKit Mol
        - Alloys: AlloyComposition with element fractions
    """

    @abstractmethod
    def get_identifier(self) -> str:
        """
        Return unique identifier for this material.

        Used for:
        - Deduplication (same identifier = same material)
        - Caching property calculations
        - Tracking discovered materials

        Returns:
            Unique string identifier

        Examples:
            - Crystals: chemical formula (e.g., "Pb6U1W6")
            - Molecules: canonical SMILES (e.g., "CCO")
            - Alloys: composition string (e.g., "Fe50Ni50")
        """
        pass

    @abstractmethod
    def copy(self) -> 'Material':
        """
        Return deep copy of this material.

        Returns:
            Independent copy of this material
        """
        pass

    def __hash__(self) -> int:
        """Materials are hashable by their identifier."""
        return hash(self.get_identifier())

    def __eq__(self, other: Any) -> bool:
        """Materials are equal if they have the same identifier."""
        if not isinstance(other, Material):
            return False
        return self.get_identifier() == other.get_identifier()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_identifier()})"

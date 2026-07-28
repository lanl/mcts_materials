"""
Abstract base class for computing material properties.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import asyncio

from .material import Material


class PropertyEvaluator(ABC):
    """
    Abstract base class for computing material properties.

    Given a material, computes one or more properties (e.g., formation energy,
    melting point, bandgap). Results are automatically cached.

    This class provides:
    - Automatic caching by material identifier
    - Async interface for parallel evaluation
    - Cache persistence methods

    Subclasses implement _compute() with actual calculation logic.
    """

    def __init__(self):
        """Initialize evaluator with empty cache."""
        self.cache: Dict[str, Dict[str, float]] = {}

    async def evaluate(self, material: Material) -> Dict[str, float]:
        """
        Evaluate properties for the given material.

        This method checks the cache first, then calls _compute() if needed.
        Results are automatically cached by material identifier.

        Args:
            material: Material to evaluate

        Returns:
            Dictionary of property_name -> value

        Examples:
            - Crystals: {"e_form": -1.2, "e_above_hull": 0.03}
            - Molecules: {"melting_point": 350.5, "h2_capacity": 5.2}
        """
        identifier = material.get_identifier()

        # Check cache
        if identifier in self.cache:
            return self.cache[identifier]

        # Compute properties
        properties = await self._compute(material)

        # Cache result
        self.cache[identifier] = properties
        return properties

    @abstractmethod
    async def _compute(self, material: Material) -> Dict[str, float]:
        """
        Compute properties for the material (implemented by subclasses).

        This is where expensive calculations happen (DFT, ML models, etc.).
        Use 'async' to allow parallel evaluation of multiple materials.

        For CPU-bound work, use asyncio.get_event_loop().run_in_executor()
        to run synchronous code without blocking the event loop.

        Args:
            material: Material to evaluate

        Returns:
            Dictionary of property_name -> value

        Raises:
            Any exception - will be propagated to caller
        """
        pass

    def get_cached_result(self, identifier: str) -> Optional[Dict[str, float]]:
        """
        Check if result is cached without triggering computation.

        Args:
            identifier: Material identifier

        Returns:
            Cached properties dict, or None if not cached
        """
        return self.cache.get(identifier)

    def save_cache(self, filepath: str) -> None:
        """
        Save cache to disk.

        Override this method to implement cache persistence.
        Default implementation does nothing.

        Args:
            filepath: Path to save cache file
        """
        pass

    def load_cache(self, filepath: str) -> None:
        """
        Load cache from disk.

        Override this method to implement cache persistence.
        Default implementation does nothing.

        Args:
            filepath: Path to cache file
        """
        pass

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()

    def __len__(self) -> int:
        """Return number of cached materials."""
        return len(self.cache)

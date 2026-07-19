"""
Shared pytest fixtures for MCTS framework tests.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import pytest
from mcts_framework.core.material import Material


class SimpleMaterial(Material):
    """Simple material implementation for testing."""

    def __init__(self, name: str):
        self.name = name

    def get_identifier(self) -> str:
        return self.name

    def copy(self) -> 'SimpleMaterial':
        return SimpleMaterial(self.name)


@pytest.fixture
def simple_material():
    """Create a simple test material."""
    return SimpleMaterial("test_material")


@pytest.fixture
def simple_materials():
    """Create multiple simple test materials."""
    return [SimpleMaterial(f"material_{i}") for i in range(5)]

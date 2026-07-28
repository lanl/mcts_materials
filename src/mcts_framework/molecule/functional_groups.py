"""
Default functional groups for molecular substitution.

Functional groups are given as SMILES fragment strings that molecule-modifier
attaches at a substitution site. Users override these via config
(MoleculeConfig.functional_groups); the defaults here are a small, chemically
reasonable starter set.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import List

# Small, common functional groups as SMILES fragments.
DEFAULT_FUNCTIONAL_GROUPS: List[str] = [
    "C",         # methyl   -CH3
    "CC",        # ethyl    -C2H5
    "O",         # hydroxyl -OH
    "N",         # amino    -NH2
    "C(=O)O",    # carboxyl -COOH
    "F",         # fluoro   -F
]


def default_functional_groups() -> List[str]:
    """Return a copy of the default functional-group SMILES list."""
    return list(DEFAULT_FUNCTIONAL_GROUPS)
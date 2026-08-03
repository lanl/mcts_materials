"""
Functional-group substitution move generator for molecules.

Generates child molecules by attaching each configured functional group at the
substitution sites found on the parent, using the molecule-modifier package.
RDKit and molecule-modifier are imported lazily so the core framework stays
dependency-light.

molecule-modifier API used (see its README):
    find_substitution_sites(mol, mode=..., target=...) -> list of sites
    substitute(mol, mode=..., sites=..., new_group=Mol) -> list of Mol
    deduplicate_smiles(list[str]) -> list[str]

© 2025. Triad National Security, LLC. All rights reserved.
"""

import logging
from typing import List, TYPE_CHECKING

from ..core.move_generator import MoveGenerator
from .structure import MolecularStructure

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


class FunctionalGroupMoves(MoveGenerator[MolecularStructure]):
    """
    Move generator that substitutes functional groups onto the parent molecule.

    For each functional-group SMILES, molecule-modifier finds the parent's
    substitution sites and produces one substituted molecule per site; the
    combined set is deduplicated by canonical SMILES before being wrapped as
    child MolecularStructures.
    """

    def __init__(
        self,
        functional_groups: List[str],
        mode: str = "hydrogen",
        target: str = "C",
    ):
        """
        Args:
            functional_groups: SMILES fragments to attach (e.g. ['C', 'O', 'N']).
            mode: molecule-modifier substitution mode ('hydrogen' or 'aromatic').
            target: Target atom type for site detection (e.g. 'C').
        """
        if not functional_groups:
            raise ValueError("functional_groups must be a non-empty list")
        self.functional_groups = functional_groups
        self.mode = mode
        self.target = target

    def generate_moves(
        self, material: MolecularStructure
    ) -> List[MolecularStructure]:
        from rdkit import Chem
        from molecule_modifier import (
            find_substitution_sites,
            substitute,
            deduplicate_smiles,
        )

        mol = material.mol
        sites = find_substitution_sites(mol, mode=self.mode, target=self.target)
        if not sites:
            return []

        all_smiles: List[str] = []
        for fg_smiles in self.functional_groups:
            fg_mol = Chem.MolFromSmiles(fg_smiles)
            if fg_mol is None:
                logger.warning("Skipping unparseable functional group: %r", fg_smiles)
                continue
            new_mols = substitute(
                mol, mode=self.mode, sites=sites, new_group=fg_mol
            )
            all_smiles.extend(Chem.MolToSmiles(m) for m in new_mols)

        # Deduplicate across all functional groups by canonical SMILES.
        unique_smiles = deduplicate_smiles(all_smiles)

        children: List[MolecularStructure] = []
        for smiles in unique_smiles:
            child_mol = Chem.MolFromSmiles(smiles)
            if child_mol is not None:
                children.append(MolecularStructure(child_mol))
        return children

"""
Molecular structure material.

Wraps an RDKit Mol. RDKit is imported lazily so the core framework and other
material types don't require it.

Identity
--------
The identifier is the RDKit **canonical SMILES**, which is order-independent
and unique per molecular graph - so two Mol objects describing the same
molecule (regardless of atom ordering or input SMILES spelling) share an
identifier, and distinct molecules never collide.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import TYPE_CHECKING

from ..core.material import Material

if TYPE_CHECKING:  # avoid importing rdkit at module load time
    from rdkit.Chem import Mol


class MolecularStructure(Material):
    """A molecule candidate, backed by an RDKit Mol object."""

    def __init__(self, mol: "Mol"):
        self.mol = mol

    def get_identifier(self) -> str:
        """Return the RDKit canonical SMILES."""
        from rdkit import Chem
        return Chem.MolToSmiles(self.mol)

    def get_smiles(self) -> str:
        """Alias for get_identifier(); returns canonical SMILES."""
        return self.get_identifier()

    def copy(self) -> "MolecularStructure":
        """Return a structure wrapping an independent copy of the Mol."""
        from rdkit import Chem
        return MolecularStructure(Chem.Mol(self.mol))

    @classmethod
    def from_smiles(cls, smiles: str) -> "MolecularStructure":
        """
        Build a MolecularStructure from a SMILES string.

        Raises:
            ValueError: if RDKit cannot parse the SMILES.
        """
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        return cls(mol)
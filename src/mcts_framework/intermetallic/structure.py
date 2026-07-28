"""
Intermetallic crystal structure material.

Wraps an ASE Atoms object. ASE and spglib are imported lazily (only when this
module is used) so the core framework and other material types don't require
them.

Identity
--------
The identifier is a **crystallographic descriptor** of the form::

    "<formula>|SG<number>|<wyckoff-decoration>"

where:

    * ``formula``  - ASE Hill-notation chemical formula (composition).
    * ``SG<number>`` - space group number (1-230) from spglib.
    * ``wyckoff-decoration`` - the sorted set of ``<element><multiplicity><letter>``
      terms, i.e. which element occupies which symmetry-distinct Wyckoff site
      (e.g. ``Pb6c-U1a-W6d``).

This distinguishes structures by *symmetry and site decoration* rather than raw
coordinates: two candidates that place U on different Wyckoff sites of the same
space group get different identifiers even at identical composition, while the
same structure re-listed in a different atom order (or slightly perturbed within
``symprec``) maps to the same identifier.

For the common substitution-only search (every candidate derived from one CIF
template) composition already determines the structure; the space-group/Wyckoff
descriptor is a principled guard so mixing templates or adding structural moves
later cannot silently alias two distinct materials. Use :meth:`get_formula` to
recover just the composition (e.g. for a formula-keyed energy cache).

© 2025. Triad National Security, LLC. All rights reserved.
"""

from collections import Counter
from typing import TYPE_CHECKING, Any

from ..core.material import Material

if TYPE_CHECKING:  # avoid importing ase at module load time
    from ase import Atoms

# Default symmetry tolerance (Angstrom) handed to spglib. Exact for
# substitution on a fixed template; absorbs small distortions if relaxed.
_DEFAULT_SYMPREC = 1e-3


def _dataset_attr(dataset: Any, key: str) -> Any:
    """
    Read a field from an spglib symmetry dataset.

    spglib >= 2.5 returns a dataclass (attribute access); older versions
    return a dict (item access). Support both.
    """
    if hasattr(dataset, key):
        return getattr(dataset, key)
    return dataset[key]


class IntermetallicStructure(Material):
    """A crystal structure candidate, backed by an ASE Atoms object."""

    def __init__(self, atoms: "Atoms", symprec: float = _DEFAULT_SYMPREC):
        """
        Args:
            atoms: ASE Atoms object for this candidate structure.
            symprec: Symmetry tolerance (Angstrom) passed to spglib when
                deriving the space group and Wyckoff positions.
        """
        self.atoms = atoms
        self.symprec = symprec

    def get_formula(self) -> str:
        """Return the composition-only ASE Hill-notation formula (e.g. 'Pb6UW6')."""
        return self.atoms.get_chemical_formula()

    def _symmetry_descriptor(self) -> str:
        """
        Build 'SG<number>|<wyckoff-decoration>' via spglib.

        The decoration aggregates atoms into (element, Wyckoff letter) groups
        and renders them as sorted '<element><count><letter>' terms joined by
        '-', e.g. 'Pb6c-U1a-W6d'. Sorting makes it independent of atom order.
        """
        import spglib
        from ase.data import chemical_symbols

        cell = (
            self.atoms.get_cell().array,
            self.atoms.get_scaled_positions(),
            self.atoms.get_atomic_numbers(),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=self.symprec)
        if dataset is None:
            raise ValueError(
                "spglib could not determine symmetry for this structure "
                f"(symprec={self.symprec})"
            )

        number = int(_dataset_attr(dataset, "number"))
        wyckoffs = list(_dataset_attr(dataset, "wyckoffs"))
        numbers = self.atoms.get_atomic_numbers()

        # Count (element_symbol, wyckoff_letter) occurrences.
        groups: Counter = Counter()
        for z, letter in zip(numbers, wyckoffs):
            groups[(chemical_symbols[z], letter)] += 1

        terms = [
            f"{sym}{count}{letter}"
            for (sym, letter), count in groups.items()
        ]
        decoration = "-".join(sorted(terms))
        return f"SG{number}|{decoration}"

    def get_identifier(self) -> str:
        """Return '<formula>|SG<number>|<wyckoff-decoration>'."""
        return f"{self.get_formula()}|{self._symmetry_descriptor()}"

    def copy(self) -> "IntermetallicStructure":
        """Return a structure wrapping an independent copy of the Atoms."""
        return IntermetallicStructure(self.atoms.copy(), symprec=self.symprec)

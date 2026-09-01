"""
Superhydride crystal structure material.

Wraps an ASE Atoms object. ASE and spglib are imported lazily (only when this
module is used) so the core framework and other material types don't require
them.

Identity
--------
The identifier is a crystallographic descriptor of the form::

    "<formula>|SG<number>|<wyckoff-decoration>"

matching the scheme used for intermetallics: composition, space-group number
from spglib, and the sorted set of ``<element><multiplicity><letter>`` terms
saying which element occupies which symmetry-distinct Wyckoff site.

The Wyckoff decoration is not decoration here. A ternary superhydride template
such as XYH8 has two crystallographically distinct host sites, and swapping
which host occupies which site gives a genuinely different material at
identical composition and space group. Composition alone would alias the two.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from collections import Counter
from typing import TYPE_CHECKING, Any

from ..core.material import Material
from .elements import HYDROGEN

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


class SuperhydrideStructure(Material):
    """A hydride crystal candidate, backed by an ASE Atoms object."""

    def __init__(self, atoms: "Atoms", symprec: float = _DEFAULT_SYMPREC):
        """
        Args:
            atoms: ASE Atoms object for this candidate structure.
            symprec: Symmetry tolerance (Angstrom) passed to spglib when
                deriving the space group and Wyckoff positions.
        """
        self.atoms = atoms
        self.symprec = symprec

    # --- Composition ------------------------------------------------------

    def get_formula(self) -> str:
        """Return the composition-only formula (e.g. 'LaBeH8')."""
        return self.atoms.get_chemical_formula(mode="metal")

    def get_hydrogen_fraction(self) -> float:
        """
        Return H_f = N_H / N_total, the hydrogen fraction.

        One of the four inputs to the Belli Tc fit, and the only one that comes
        free from the composition - no DFT needed.
        """
        numbers = self.atoms.get_atomic_numbers()
        if len(numbers) == 0:
            raise ValueError("Structure has no atoms; hydrogen fraction is undefined")
        n_hydrogen = int(sum(1 for z in numbers if int(z) == HYDROGEN))
        return n_hydrogen / len(numbers)

    def get_host_elements(self) -> list:
        """Return the sorted atomic numbers of the non-hydrogen (host) species."""
        return sorted({int(z) for z in self.atoms.get_atomic_numbers() if int(z) != HYDROGEN})

    # --- Identity ---------------------------------------------------------

    def _symmetry_descriptor(self) -> str:
        """
        Build 'SG<number>|<wyckoff-decoration>' via spglib.

        The decoration aggregates atoms into (element, Wyckoff letter) groups
        and renders them as sorted '<element><count><letter>' terms joined by
        '-', e.g. 'Be4b-H32f-La4a'. Sorting makes it independent of atom order.
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

        groups: Counter = Counter()
        for z, letter in zip(numbers, wyckoffs):
            groups[(chemical_symbols[z], letter)] += 1

        terms = [f"{sym}{count}{letter}" for (sym, letter), count in groups.items()]
        return f"SG{number}|{'-'.join(sorted(terms))}"

    def get_identifier(self) -> str:
        """Return '<formula>|SG<number>|<wyckoff-decoration>'."""
        return f"{self.get_formula()}|{self._symmetry_descriptor()}"

    def copy(self) -> "SuperhydrideStructure":
        """Return a structure wrapping an independent copy of the Atoms."""
        return SuperhydrideStructure(self.atoms.copy(), symprec=self.symprec)

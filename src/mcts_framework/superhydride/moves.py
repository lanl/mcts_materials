"""
Host-substitution move generator for superhydride structures.

Expansion rule
--------------
One move substitutes **one** host species for a chemically adjacent element,
everywhere it occurs, leaving the hydrogen sublattice and the structural
template untouched.

This is the generative process the source paper describes for the ternary
hydride datasets themselves: "once an optimal template is discovered, new
structures are generated via atomic substitution of elements with similar
chemistries in a given supercell" (Belli et al., Ann. Phys. 2025, 537, e00280,
Sec. 2.1). Searching the host sublattice of a template is therefore not a
convenience - it is the space those datasets live in.

Two consequences worth stating plainly:

    * H_f is fixed by the template. Of the four inputs to the Tc fit, the
      search moves phi, phi* and H_DOS; the hydrogen fraction changes only if
      the template does. Since Tc scales as H_f^3, choosing the template is
      choosing the ceiling.
    * Branching is the SUM over host sites of their allowed substitutions, not
      the product. A ternary with ~4 neighbors per host expands to ~8 children
      rather than ~25, so tree depth measures chemical distance from the
      starting composition.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import TYPE_CHECKING, List

from ..core.move_generator import MoveGenerator
from . import elements
from .structure import SuperhydrideStructure

if TYPE_CHECKING:
    from ase import Atoms


class HostSubstitutionMoves(MoveGenerator[SuperhydrideStructure]):
    """
    Single-host element-substitution move generator.

    For each distinct non-hydrogen species in the parent, emits one child per
    allowed substitution of that species. Hydrogen is never substituted.
    """

    def __init__(
        self,
        palette: str = "high_tc",
        preserve_distinct_hosts: bool = True,
    ):
        """
        Args:
            palette: Host element palette - 'electropositive', 'covalent',
                'high_tc' (the union of the two high-Tc classes, default) or
                'all' (adds late transition metals). See
                :mod:`~mcts_framework.superhydride.elements`.
            preserve_distinct_hosts: Drop substitutions that would make two
                host species identical. With the default True a ternary stays
                ternary; set False to let the search collapse a ternary onto
                the binary hydride it contains (the Tc fit is valid for both,
                but the binary space is largely exhausted in the literature).

        Raises:
            ValueError: on an unknown palette name.
        """
        elements.host_palette(palette)  # fail fast on a bad name
        self.palette = palette
        self.preserve_distinct_hosts = preserve_distinct_hosts

    def generate_moves(
        self, material: SuperhydrideStructure
    ) -> List[SuperhydrideStructure]:
        hosts = material.get_host_elements()
        host_set = set(hosts)

        children: List[SuperhydrideStructure] = []
        for host in hosts:
            for target in elements.host_moves(host, self.palette):
                if self.preserve_distinct_hosts and target in host_set:
                    continue
                children.append(
                    SuperhydrideStructure(
                        self._substitute(material.atoms, host, target),
                        symprec=material.symprec,
                    )
                )
        return children

    @staticmethod
    def _substitute(atoms: "Atoms", host: int, target: int) -> "Atoms":
        """Return a copy of ``atoms`` with every ``host`` atom replaced by ``target``."""
        new_atoms = atoms.copy()
        numbers = [
            target if int(z) == host else int(z)
            for z in new_atoms.get_atomic_numbers()
        ]
        new_atoms.set_atomic_numbers(numbers)
        return new_atoms

"""
Periodic-table move generator for intermetallic structures.

Generates child structures by substituting elements on the transition-metal,
Group IV, and f-block sites of the parent, using the validated move rules in
``elements.py``. Behavior mirrors mcts_crystal's expand(): the full set of
children is the Cartesian product of the allowed metal, Group IV, and f-block
moves.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from typing import List, Optional, TYPE_CHECKING

from ..core.move_generator import MoveGenerator
from .structure import IntermetallicStructure
from . import elements

if TYPE_CHECKING:
    from ase import Atoms


class PeriodicTableMoves(MoveGenerator[IntermetallicStructure]):
    """
    Element-substitution move generator.

    For the parent structure, determines the current metal / Group IV /
    f-block occupant(s) and the allowed moves for each, then emits one child
    per (metal, group_iv, f_block) combination. The f-block axis is omitted
    when the structure has no f-block site.
    """

    def __init__(
        self,
        f_block_mode: str = "u_only",
        move_step: int = 1,
        u_bridge: str = "narrow",
    ):
        """
        Args:
            f_block_mode: One of the canonical f-block modes understood by
                elements.f_block_moves (u_only, lanthanides_u,
                lanthanides_u_no_wrap, full_f_block).
            move_step: Max positions a single substitution may jump along the
                transition-metal / Group IV / lanthanide axes (default 1 =
                adjacent only; e.g. 3 gives extended-range exploration). Fixed
                cross-period metal jumps and full_f_block are unaffected.
            u_bridge: which lanthanides U(92) connects to - 'narrow' (Nd only,
                default) or 'wide' (Nd/Gd/Er). Only affects the lanthanide/U
                modes; orthogonal to move_step.
        """
        self.f_block_mode = f_block_mode
        self.move_step = move_step
        self.u_bridge = u_bridge

    def generate_moves(
        self, material: IntermetallicStructure
    ) -> List[IntermetallicStructure]:
        atoms = material.atoms
        present = set(atoms.get_atomic_numbers())

        # Determine the allowed moves on each site axis from the current
        # occupants. Absent axes contribute a single "no-op" so the product
        # still works.
        metal_axis: List[int] = [0]
        giv_axis: List[int] = [0]
        fblock_axis: List[Optional[int]] = [None]

        for z in present:
            site = elements.classify_site(int(z))
            if site == "metal":
                metal_axis = elements.metal_moves(int(z), self.move_step)
            elif site == "group_iv":
                giv_axis = elements.group_iv_moves(int(z), self.move_step)
            elif site == "f_block":
                fblock_axis = list(
                    elements.f_block_moves(
                        int(z), self.f_block_mode, self.move_step, self.u_bridge
                    )
                )

        children: List[IntermetallicStructure] = []
        for metal in metal_axis:
            for giv in giv_axis:
                for fblock in fblock_axis:
                    new_atoms = self._substitute(atoms, metal, giv, fblock)
                    children.append(
                        IntermetallicStructure(new_atoms, symprec=material.symprec)
                    )
        return children

    @staticmethod
    def _substitute(
        atoms: "Atoms",
        metal: int,
        g_iv: int,
        f_block: Optional[int],
    ) -> "Atoms":
        """
        Return a copy of ``atoms`` with sites substituted.

        Each atom is remapped by site type:
            f-block site -> f_block (if provided)
            Group IV site -> g_iv
            everything else (metal) -> metal

        A target of 0 (the no-op sentinel) leaves that atom unchanged. Ported
        from mcts_crystal.node.substitute().
        """
        op = []
        for z in atoms.get_atomic_numbers():
            z = int(z)
            if z in elements.F_BLOCK_ELEMENTS and f_block is not None:
                op.append(f_block - z)
            elif z in elements.GROUP_IV_CHAIN:
                op.append((g_iv - z) if g_iv else 0)
            else:
                op.append((metal - z) if metal else 0)

        new_atoms = atoms.copy()
        new_atoms.set_atomic_numbers(new_atoms.get_atomic_numbers() + op)
        return new_atoms

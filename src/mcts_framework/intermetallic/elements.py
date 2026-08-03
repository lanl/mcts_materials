"""
Periodic-table element groups and move rules for intermetallic search.

All logic here is ported verbatim (in behavior) from the validated
mcts_crystal implementation so that results reproduce exactly. The functions
are pure - they take atomic numbers and return lists of atomic numbers - so
they are easy to unit-test independently of ASE.

Element groups
--------------
    Group IV chain : Si(14) - Ge(32) - Sn(50) - Pb(82)
    3d metals      : Ti(22) - Zn(30)
    4d metals      : Zr(40) - Cd(48)
    5d metals      : Hf(72) - Hg(80)
    f-block        : lanthanides La(57)-Lu(71) + actinides Ac(89)-Pu(94)

© 2025. Triad National Security, LLC. All rights reserved.
"""

from typing import List, Optional

# --- Element group definitions -------------------------------------------

GROUP_IV_CHAIN: List[int] = [14, 32, 50, 82]  # Si, Ge, Sn, Pb (chain order)

# f-block atomic numbers that participate in substitution (lanthanides +
# allowed actinides). Used by substitute() to recognize the f-block site.
F_BLOCK_ELEMENTS: List[int] = list(range(57, 72)) + list(range(89, 95))

# Transition-metal ranges by row.
TM_3D = range(22, 31)  # Ti..Zn
TM_4D = range(40, 49)  # Zr..Cd
TM_5D = range(72, 81)  # Hf..Hg


# --- Move-rule functions --------------------------------------------------


def group_iv_moves(atomic_num: int, move_step: int = 1) -> List[int]:
    """
    Group IV moves: current element plus chain neighbors within move_step.

    No wrap-around (Si and Pb are chain ends). At move_step=1 this is just the
    immediate neighbors (e.g. Ge -> {Si, Ge, Sn}); larger move_step reaches
    farther along the Si-Ge-Sn-Pb chain. Returns a sorted list.
    """
    if atomic_num not in GROUP_IV_CHAIN:
        return [atomic_num]
    idx = GROUP_IV_CHAIN.index(atomic_num)
    moves = {atomic_num}
    for delta in range(1, move_step + 1):
        if idx - delta >= 0:
            moves.add(GROUP_IV_CHAIN[idx - delta])
        if idx + delta < len(GROUP_IV_CHAIN):
            moves.add(GROUP_IV_CHAIN[idx + delta])
    return sorted(moves)


def metal_moves(atomic_num: int, move_step: int = 1) -> List[int]:
    """
    Transition-metal moves: within-row steps up to move_step, plus fixed
    single-period cross-row jumps.

    Within a d-row (3d 22-30, 4d 40-48, 5d 72-80) the element may move +/-1..
    move_step positions (clamped to the row). Cross-period jumps to the
    adjacent d-row(s) are always a single period step (+/-18 between 3d/4d,
    +/-32 between 4d/5d), independent of move_step. At move_step=1 this
    reproduces the original hand-tuned edge cases exactly (e.g. Ti->[22,23,40],
    Cd->[30,47,48,80]). Returns a sorted, de-duplicated list.
    """
    mv = {atomic_num}
    if 22 <= atomic_num <= 30:  # 3d
        for s in range(1, move_step + 1):
            if atomic_num - s >= 22:
                mv.add(atomic_num - s)
            if atomic_num + s <= 30:
                mv.add(atomic_num + s)
        mv.add(atomic_num + 18)  # cross-period up to 4d
    elif 40 <= atomic_num <= 48:  # 4d
        for s in range(1, move_step + 1):
            if atomic_num - s >= 40:
                mv.add(atomic_num - s)
            if atomic_num + s <= 48:
                mv.add(atomic_num + s)
        mv.add(atomic_num - 18)  # cross-period down to 3d
        if atomic_num + 32 <= 80:
            mv.add(atomic_num + 32)  # cross-period up to 5d
    elif 72 <= atomic_num <= 80:  # 5d
        for s in range(1, move_step + 1):
            if atomic_num - s >= 72:
                mv.add(atomic_num - s)
            if atomic_num + s <= 80:
                mv.add(atomic_num + s)
        mv.add(atomic_num - 32)  # cross-period down to 4d
    else:
        # Not a recognized transition metal - no move.
        return [atomic_num]
    return sorted(mv)


# U(92) bridge widths: which lanthanides U connects to (and which connect back
# to U). 'narrow' = Nd only; 'wide' = Nd/Gd/Er (light/mid/heavy). This is
# orthogonal to move_step (jump distance) and to wrap-around (mode).
_U_BRIDGE_LANTHANIDES = {
    "narrow": (60,),        # Nd
    "wide": (60, 64, 68),   # Nd, Gd, Er
}


def f_block_moves(
    atomic_num: int,
    mode: str,
    move_step: int = 1,
    u_bridge: str = "narrow",
) -> List[int]:
    """
    F-block moves under the given mode.

    Modes (canonical names; see core.config for the alias handling):
        u_only                 : [92] only.
        lanthanides_u          : +/-move_step with wrap-around (Ce<->Lu); U bridge.
        lanthanides_u_no_wrap  : +/-move_step, no wrap; U bridge.
        full_f_block           : Ce-Lu + Th-Pu, +/-1 plus vertical Ln<->An.

    Two orthogonal knobs control the lanthanide/U modes:
      - move_step: lanthanide jump range (default 1 = adjacent; larger = farther).
      - u_bridge : which lanthanides U(92) connects to - 'narrow' (Nd only) or
        'wide' (Nd/Gd/Er). Independent of jump distance and wrap-around.
    full_f_block ignores both (its +/-1 neighbor + vertical analog structure is
    fixed and has no U bridge).

    Returns a sorted, de-duplicated list of atomic numbers.
    """
    lanthanides = list(range(58, 72))  # Ce(58)..Lu(71)
    bridge = _U_BRIDGE_LANTHANIDES.get(u_bridge, _U_BRIDGE_LANTHANIDES["narrow"])

    def _add_u_bridge(possible: List[int]) -> None:
        """Connect U(92) to its bridge lanthanides, and those back to U."""
        if atomic_num == 92:
            possible.extend(bridge)
        elif atomic_num in bridge:
            possible.append(92)

    if mode == "u_only":
        possible = [92]

    elif mode == "lanthanides_u":
        possible = [atomic_num]
        if atomic_num in lanthanides:
            idx = lanthanides.index(atomic_num)
            for delta in range(-move_step, move_step + 1):
                if delta != 0:
                    possible.append(lanthanides[(idx + delta) % len(lanthanides)])
        _add_u_bridge(possible)

    elif mode == "lanthanides_u_no_wrap":
        # +/-move_step neighbors, NO wrap-around (Ce/Lu are chain ends).
        possible = [atomic_num]
        if atomic_num in lanthanides:
            idx = lanthanides.index(atomic_num)
            for delta in range(1, move_step + 1):
                if idx - delta >= 0:
                    possible.append(lanthanides[idx - delta])
                if idx + delta < len(lanthanides):
                    possible.append(lanthanides[idx + delta])
        _add_u_bridge(possible)

    else:  # full_f_block
        actinides = list(range(90, 95))  # Th(90)..Pu(94)
        all_f = lanthanides + actinides
        possible = [atomic_num]
        for delta in (-1, 1):
            neighbor = atomic_num + delta
            if neighbor in all_f:
                possible.append(neighbor)
        # Vertical Ln <-> An analog moves (+/-32).
        if 58 <= atomic_num <= 62:  # Ce..Sm -> Th..Pu
            analog = atomic_num + 32
            if analog in all_f:
                possible.append(analog)
        if 90 <= atomic_num <= 94:  # Th..Pu -> Ce..Sm
            analog = atomic_num - 32
            if analog in all_f:
                possible.append(analog)

    return sorted(set(possible))


def classify_site(atomic_num: int) -> Optional[str]:
    """
    Classify an atomic number into its substitution site type.

    Returns one of 'group_iv', 'metal', 'f_block', or None if the element is
    not part of any recognized substitution group.
    """
    if atomic_num in GROUP_IV_CHAIN:
        return "group_iv"
    if atomic_num in F_BLOCK_ELEMENTS:
        return "f_block"
    if 22 <= atomic_num <= 30 or 40 <= atomic_num <= 48 or 72 <= atomic_num <= 80:
        return "metal"
    return None

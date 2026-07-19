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


def group_iv_moves(atomic_num: int) -> List[int]:
    """
    Group IV moves: current element plus immediate chain neighbors only.

    No direct jumps (e.g. Sn->Si must pass through Ge). Returns sorted list.
    """
    if atomic_num not in GROUP_IV_CHAIN:
        return [atomic_num]
    idx = GROUP_IV_CHAIN.index(atomic_num)
    moves = [atomic_num]
    if idx > 0:
        moves.append(GROUP_IV_CHAIN[idx - 1])
    if idx < len(GROUP_IV_CHAIN) - 1:
        moves.append(GROUP_IV_CHAIN[idx + 1])
    return sorted(moves)


def metal_moves(atomic_num: int) -> List[int]:
    """
    Transition-metal moves: up/down a group and left/right a period.

    Edge elements (Ti, Zn, Zr, Cd, Hf, Hg) have hand-tuned move sets matching
    the validated mcts_crystal behavior. Returns the raw list (unsorted, as in
    the original); callers that dedupe should sort as needed.
    """
    if 22 <= atomic_num <= 30:  # 3d
        if atomic_num == 22:  # Ti
            return [22, 23, 40]  # Ti, V, Zr
        if atomic_num == 30:  # Zn
            return [29, 30, 48]  # Cu, Zn, Cd
        return [atomic_num - 1, atomic_num, atomic_num + 1, atomic_num + 18]
    if 40 <= atomic_num <= 48:  # 4d
        if atomic_num == 40:  # Zr
            return [40, 41, 72, 22]  # Zr, Nb, Hf, Ti
        if atomic_num == 48:  # Cd
            return [47, 48, 80, 30]  # Ag, Cd, Hg, Zn
        return [atomic_num - 1, atomic_num, atomic_num + 1, atomic_num + 32, atomic_num - 18]
    if 72 <= atomic_num <= 80:  # 5d
        if atomic_num == 72:  # Hf
            return [72, 73, 40]  # Hf, Ta, Zr
        if atomic_num == 80:  # Hg
            return [79, 80, 48]  # Au, Hg, Cd
        return [atomic_num - 1, atomic_num, atomic_num + 1, atomic_num - 32]
    # Not a recognized transition metal - no move.
    return [atomic_num]


def f_block_moves(atomic_num: int, mode: str) -> List[int]:
    """
    F-block moves under the given mode.

    Modes (canonical names; see core.config for the alias handling):
        u_only                 : [92] only.
        lanthanides_u          : +/-1 with wrap-around (Ce<->Lu); U<->Nd.
        lanthanides_u_extended : +/-1,2,3 with wrap-around; U<->Nd/Gd/Er.
        lanthanides_u_no_wrap  : +/-1 no wrap; U<->Nd. (formerly 'experimental')
        full_f_block           : Ce-Lu + Th-Pu, +/-1 plus vertical Ln<->An.

    Returns a sorted, de-duplicated list of atomic numbers.
    """
    lanthanides = list(range(58, 72))  # Ce(58)..Lu(71)

    if mode == "u_only":
        possible = [92]

    elif mode == "lanthanides_u_extended":
        possible = [atomic_num]
        if atomic_num in lanthanides:
            idx = lanthanides.index(atomic_num)
            for delta in (-3, -2, -1, 1, 2, 3):
                possible.append(lanthanides[(idx + delta) % len(lanthanides)])
        if atomic_num == 92:
            possible.extend([60, 64, 68])  # Nd, Gd, Er
        elif atomic_num in (60, 64, 68):
            possible.append(92)

    elif mode == "lanthanides_u":
        possible = [atomic_num]
        if atomic_num in lanthanides:
            idx = lanthanides.index(atomic_num)
            possible.append(lanthanides[(idx - 1) % len(lanthanides)])  # wrap Ce->Lu
            possible.append(lanthanides[(idx + 1) % len(lanthanides)])  # wrap Lu->Ce
        if atomic_num == 92:
            possible.append(60)  # U -> Nd
        elif atomic_num == 60:
            possible.append(92)  # Nd -> U

    elif mode == "lanthanides_u_no_wrap":
        # +/-1 neighbors, NO wrap-around (Ce/Lu are chain ends). Formerly
        # 'experimental'; its old comment mislabeled the set as actinides.
        possible = [atomic_num]
        for delta in (-1, 1):
            neighbor = atomic_num + delta
            if neighbor in lanthanides:
                possible.append(neighbor)
        if atomic_num == 92:
            possible.append(60)  # U -> Nd
        elif atomic_num == 60:
            possible.append(92)  # Nd -> U

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

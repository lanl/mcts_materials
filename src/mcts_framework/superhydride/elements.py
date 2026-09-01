"""
Host-element palettes and move rules for the ternary superhydride search.

The search substitutes the *non-hydrogen* (host) sublattice of a fixed
structural template. This module defines which elements may occupy a host
site, and which elements are one "chemical step" from a given host. The
functions are pure - atomic number in, atomic numbers out - so they unit-test
without ASE.

Chemical classes
----------------
Belli et al. (Ann. Phys. 2025, 537, e00280, Sec. 2.1) classify hydride
superconductors by the dominant bonding mechanism of hydrogen, and only two of
the three classes reach high Tc:

    electropositive : every host is an electropositive element (Pauling
        electronegativity ~<1.6) - alkali, alkaline earth, rare earth, plus
        early/post-transition metals such as Al, Zr and Ta. Hosts act as
        spacers and electron donors; high Tc needs H_f >~ 0.8 and, usually,
        >100 GPa.
    covalent : hydrogen forms (polar) covalent bonds to a p-block element with
        electronegativity ~>1.6 (B, Si, P, S, Se, ...), often with an
        electropositive countercation. Reaches moderate-to-high Tc at markedly
        lower pressure.
    interstitial : late transition-metal hosts with hydrogen in octahedral or
        tetrahedral holes. The Fermi level is dominated by metal d-states and
        the measured/computed Tcs are low, so these are excluded from the
        default palette.

Two class memberships follow the paper's chemistry rather than a hard
electronegativity cut: Al (1.61) and V (1.63) sit marginally above 1.6 but are
grouped with their electropositive congeners - Al because the paper names AlH3
as an example of the class, V because it is the 3d member of the Nb/Ta group.

Move rules
----------
Two elements are one chemical step apart when they are adjacent in the
periodic table as it is drawn:

    horizontal : consecutive atomic numbers within one period. In the long
        form this also walks the f-block chain and its ends
        (Ba-La-Ce-...-Lu-Hf), so no special-casing is needed for lanthanides.
    vertical   : same group, adjacent period (Ca->Mg/Sr, B->Al, Y->Sc/La).
    Ln<->An analog : the vertical relationship the long form cannot draw,
        implemented as +/-32 in atomic number (Ce<->Th, Nd<->U).

Moves are then intersected with the active palette, so an element outside the
palette is simply not reachable.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

# --- Periodic-table geometry ---------------------------------------------

# Groups occupied in the short periods (no d-block).
_SHORT_PERIOD_GROUPS = (1, 2, 13, 14, 15, 16, 17, 18)
_LONG_PERIOD_GROUPS = tuple(range(1, 19))


def _row(groups: Iterable[int], first_z: int) -> Dict[int, int]:
    """Map consecutive atomic numbers from ``first_z`` onto ``groups``."""
    return {group: first_z + offset for offset, group in enumerate(groups)}


# (period -> {group -> atomic number}) for everything outside the f-block.
# Period 6 runs Cs, Ba, La, then jumps to Hf..Rn; the lanthanides in between
# carry no group and are handled by _F_BLOCK_PERIOD below.
_PERIOD_ROWS: Dict[int, Dict[int, int]] = {
    1: {1: 1, 18: 2},
    2: _row(_SHORT_PERIOD_GROUPS, 3),    # Li..Ne
    3: _row(_SHORT_PERIOD_GROUPS, 11),   # Na..Ar
    4: _row(_LONG_PERIOD_GROUPS, 19),    # K..Kr
    5: _row(_LONG_PERIOD_GROUPS, 37),    # Rb..Xe
    6: {1: 55, 2: 56, 3: 57, **_row(range(4, 19), 72)},  # Cs, Ba, La, Hf..Rn
    7: {1: 87, 2: 88, 3: 89},            # Fr, Ra, Ac
}

#: Lanthanides La(57)-Lu(71) and actinides Ac(89)-Pu(94). La and Ac also sit in
#: the group-3 column of _PERIOD_ROWS; listing them here as well is harmless
#: (moves are de-duplicated) and keeps the f-block chains contiguous.
LANTHANIDES: FrozenSet[int] = frozenset(range(57, 72))
ACTINIDES: FrozenSet[int] = frozenset(range(89, 95))

#: Which period each f-block element belongs to, so the horizontal rule can
#: walk Ba-La-Ce-...-Lu-Hf and Ra-Ac-Th-Pa-U as continuous rows.
_F_BLOCK_PERIOD: Dict[int, int] = {
    **{z: 6 for z in LANTHANIDES},
    **{z: 7 for z in ACTINIDES},
}

#: Atomic-number offset between a lanthanide and its actinide analog.
_LN_AN_OFFSET = 32


def _build_indices() -> Tuple[Dict[int, int], Dict[int, int], Dict[Tuple[int, int], int]]:
    """Build the (Z -> period), (Z -> group) and ((period, group) -> Z) maps."""
    period_of: Dict[int, int] = {}
    group_of: Dict[int, int] = {}
    grid: Dict[Tuple[int, int], int] = {}
    for period, row in _PERIOD_ROWS.items():
        for group, z in row.items():
            period_of[z] = period
            group_of[z] = group
            grid[(period, group)] = z
    for z, period in _F_BLOCK_PERIOD.items():
        period_of.setdefault(z, period)
    return period_of, group_of, grid


_PERIOD_OF, _GROUP_OF, _GRID = _build_indices()


# --- Host palettes --------------------------------------------------------

#: Electropositive hosts: alkali, alkaline earth, group 3 + lanthanides,
#: early transition metals (groups 4-5), Al, and the two actinides with a
#: substantial hydride literature (Th, U). See the module docstring on Al/V.
ELECTROPOSITIVE: FrozenSet[int] = frozenset(
    {3, 11, 19, 37, 55}                     # Li, Na, K, Rb, Cs
    | {4, 12, 20, 38, 56}                   # Be, Mg, Ca, Sr, Ba
    | {21, 39}                              # Sc, Y
    | set(LANTHANIDES)                      # La-Lu
    | {22, 40, 72}                          # Ti, Zr, Hf
    | {23, 41, 73}                          # V, Nb, Ta
    | {13}                                  # Al
    | {90, 92}                              # Th, U
)

#: Covalent hosts: p-block elements that form directional X-H bonds. Halogens
#: and oxygen are excluded - they give molecular HX or oxides rather than the
#: hypervalent/multi-centre motifs the covalent class is built on.
COVALENT: FrozenSet[int] = frozenset(
    {5, 6, 7}                               # B, C, N
    | {14, 15, 16}                          # Si, P, S
    | {31, 32, 33, 34}                      # Ga, Ge, As, Se
    | {49, 50, 51, 52}                      # In, Sn, Sb, Te
    | {81, 82, 83}                          # Tl, Pb, Bi
)

#: Late transition metals, which give interstitial hydrides with low Tc.
TRANSITION: FrozenSet[int] = frozenset(
    set(range(24, 31))                      # Cr..Zn
    | set(range(42, 49))                    # Mo..Cd
    | set(range(74, 81))                    # W..Hg
)

#: Named palettes. 'high_tc' (the default) is the union of the two classes the
#: paper shows can reach high Tc.
_PALETTES: Dict[str, FrozenSet[int]] = {
    "electropositive": ELECTROPOSITIVE,
    "covalent": COVALENT,
    "high_tc": ELECTROPOSITIVE | COVALENT,
    "all": ELECTROPOSITIVE | COVALENT | TRANSITION,
}

PALETTE_NAMES = tuple(_PALETTES)

#: Hydrogen is the anion sublattice, never a host.
HYDROGEN = 1


def host_palette(name: str) -> FrozenSet[int]:
    """
    Return the set of atomic numbers a host site may take under ``name``.

    Raises:
        ValueError: on an unknown palette name.
    """
    try:
        return _PALETTES[name]
    except KeyError:
        raise ValueError(
            f"Unknown host palette {name!r}; expected one of {PALETTE_NAMES}"
        ) from None


def classify_host(atomic_num: int) -> Optional[str]:
    """
    Classify a host element as 'electropositive', 'covalent' or 'transition'.

    Returns None for elements in none of the three classes (including
    hydrogen, which is not a host).
    """
    if atomic_num in ELECTROPOSITIVE:
        return "electropositive"
    if atomic_num in COVALENT:
        return "covalent"
    if atomic_num in TRANSITION:
        return "transition"
    return None


# --- Move rules -----------------------------------------------------------


def chemical_neighbors(atomic_num: int) -> List[int]:
    """
    Return the elements one chemical step from ``atomic_num``, palette-free.

    A step is a horizontal move (adjacent atomic number in the same period,
    which walks the f-block chain in the long form), a vertical move (same
    group, adjacent period), or a lanthanide/actinide analog move (+/-32).
    The element itself is never included.

    Returns:
        Sorted list of atomic numbers.
    """
    period = _PERIOD_OF.get(atomic_num)
    if period is None:
        return []

    neighbors = set()

    # Horizontal: the next/previous element, provided it is in the same period.
    for step in (-1, 1):
        if _PERIOD_OF.get(atomic_num + step) == period:
            neighbors.add(atomic_num + step)

    # Vertical: same group, one period up or down.
    group = _GROUP_OF.get(atomic_num)
    if group is not None:
        for step in (-1, 1):
            analog = _GRID.get((period + step, group))
            if analog is not None:
                neighbors.add(analog)

    # The vertical relationship the long form cannot draw: Ce<->Th, Nd<->U.
    if atomic_num in LANTHANIDES and atomic_num + _LN_AN_OFFSET in ACTINIDES:
        neighbors.add(atomic_num + _LN_AN_OFFSET)
    elif atomic_num in ACTINIDES and atomic_num - _LN_AN_OFFSET in LANTHANIDES:
        neighbors.add(atomic_num - _LN_AN_OFFSET)

    neighbors.discard(atomic_num)
    return sorted(neighbors)


def host_moves(atomic_num: int, palette: str = "high_tc") -> List[int]:
    """
    Return the substitutions allowed for a host site currently holding
    ``atomic_num``, under the given palette.

    Chemical neighbors outside the palette are dropped, so an element the
    palette excludes is unreachable rather than a dead end mid-search. An
    element that is not itself in the palette has no moves at all - its site
    stays frozen (see :func:`validate_hosts`).

    Returns:
        Sorted list of atomic numbers, never containing ``atomic_num``.
    """
    allowed = host_palette(palette)
    if atomic_num not in allowed:
        return []
    return [z for z in chemical_neighbors(atomic_num) if z in allowed]


def validate_hosts(atomic_numbers: Iterable[int], palette: str) -> List[str]:
    """
    Check a starting structure against a palette; return warning messages.

    Pure over an iterable of atomic numbers (extracted from the loaded
    structure by the caller), so config validation stays free of file I/O and
    of ASE. Raises on a structure that cannot be searched at all; softer
    problems come back as warning strings for the caller to emit.

    Raises:
        ValueError: if the structure contains no hydrogen (not a hydride), or
            no host element the palette can move.
    """
    numbers = [int(z) for z in atomic_numbers]
    if HYDROGEN not in numbers:
        raise ValueError(
            "The starting structure contains no hydrogen, so it is not a "
            "hydride. Provide a structure whose formula includes H."
        )

    hosts = sorted({z for z in numbers if z != HYDROGEN})
    if not hosts:
        raise ValueError(
            "The starting structure is elemental hydrogen: there is no host "
            "sublattice to substitute."
        )

    allowed = host_palette(palette)
    frozen = [z for z in hosts if z not in allowed]
    movable = [z for z in hosts if z in allowed and host_moves(z, palette)]

    if not movable:
        raise ValueError(
            f"No host site in the starting structure can move under "
            f"palette={palette!r} (hosts: {hosts}). Choose a wider palette or "
            f"a different starting structure."
        )

    warnings_out: List[str] = []
    if frozen:
        warnings_out.append(
            f"Host element(s) {frozen} are outside palette={palette!r}; those "
            f"sites will stay fixed for the whole search."
        )
    if len(hosts) < 2:
        warnings_out.append(
            f"The starting structure has a single host element {hosts}, so it "
            f"is a binary hydride. The search will explore binaries unless the "
            f"template provides two distinct host sites."
        )
    return warnings_out

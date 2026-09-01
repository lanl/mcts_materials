"""
Unit tests for superhydride host palettes and move rules.

Pure integer-in/integer-out functions (no ASE), verifying that the
periodic-table navigation makes the chemical steps it claims to.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import pytest

from mcts_framework.superhydride import elements

# Atomic numbers used throughout, spelled out so the assertions read chemically.
H, LI, BE, B, C, N = 1, 3, 4, 5, 6, 7
NA, MG, AL, SI, P, S = 11, 12, 13, 14, 15, 16
K, CA, SC, TI, V, CR, FE, CU, ZN = 19, 20, 21, 22, 23, 24, 26, 29, 30
GA, GE, AS, SE, BR = 31, 32, 33, 34, 35
RB, SR, Y, ZR, NB, PD, TE = 37, 38, 39, 40, 41, 46, 52
CS, BA, LA, CE, PR, ND, PM, YB, LU = 55, 56, 57, 58, 59, 60, 61, 70, 71
HF, TA, W, PT, PB = 72, 73, 74, 78, 82
AC, TH, PA, U = 89, 90, 91, 92
OXYGEN, FLUORINE, CHLORINE = 8, 9, 17


# --- Palettes -------------------------------------------------------------


def test_palette_membership_follows_the_papers_classes():
    assert {LI, CA, BA, LA, LU, TI, TA, AL, TH, U} <= elements.ELECTROPOSITIVE
    assert {B, SI, S, SE, TE, PB} <= elements.COVALENT
    assert {CR, FE, CU, PD, PT, W} <= elements.TRANSITION
    # The three classes are disjoint, so classify_host is unambiguous.
    assert not (elements.ELECTROPOSITIVE & elements.COVALENT)
    assert not (elements.ELECTROPOSITIVE & elements.TRANSITION)
    assert not (elements.COVALENT & elements.TRANSITION)


def test_hydrogen_is_never_a_host():
    for palette in elements.PALETTE_NAMES:
        assert H not in elements.host_palette(palette)
    assert elements.classify_host(H) is None


def test_halogens_and_oxygen_excluded():
    # They give molecular HX or oxides rather than the hydridic motifs the
    # covalent class is built on.
    for z in (OXYGEN, FLUORINE, CHLORINE, BR):
        assert z not in elements.host_palette("all")


def test_high_tc_is_the_union_of_the_two_high_tc_classes():
    assert elements.host_palette("high_tc") == elements.ELECTROPOSITIVE | elements.COVALENT
    # ...and excludes the interstitial class.
    assert not (elements.host_palette("high_tc") & elements.TRANSITION)
    assert elements.host_palette("all") == elements.host_palette("high_tc") | elements.TRANSITION


def test_unknown_palette_is_rejected():
    with pytest.raises(ValueError, match="Unknown host palette"):
        elements.host_palette("noble_gases")


def test_classify_host():
    assert elements.classify_host(LA) == "electropositive"
    assert elements.classify_host(B) == "covalent"
    assert elements.classify_host(PD) == "transition"
    assert elements.classify_host(2) is None  # He


# --- Chemical neighbours --------------------------------------------------


def test_vertical_moves_walk_a_group():
    assert MG in elements.chemical_neighbors(CA)   # Ca -> Mg, one period up
    assert SR in elements.chemical_neighbors(CA)   # Ca -> Sr, one period down
    assert AL in elements.chemical_neighbors(B)    # B -> Al
    assert {SC, LA} <= set(elements.chemical_neighbors(Y))  # group 3


def test_horizontal_moves_bridge_the_short_period_gap():
    # Be (group 2) and B (group 13) are not adjacent columns in the standard
    # layout, but they are consecutive elements in period 2 - and chemically
    # adjacent, which is what a substitution move means.
    assert B in elements.chemical_neighbors(BE)
    assert BE in elements.chemical_neighbors(B)


def test_horizontal_moves_walk_the_lanthanide_chain():
    # In the long form period 6 runs Ba - La - Ce - ... - Lu - Hf, so the
    # f-block chain and both its ends come out of the horizontal rule.
    assert {BA, CE} <= set(elements.chemical_neighbors(LA))
    assert {LA, PR} <= set(elements.chemical_neighbors(CE))
    assert {YB, HF} <= set(elements.chemical_neighbors(LU))
    assert LU in elements.chemical_neighbors(HF)


def test_lanthanide_actinide_analog_moves():
    # The vertical relationship the long form cannot draw: +/-32.
    assert TH in elements.chemical_neighbors(CE)
    assert CE in elements.chemical_neighbors(TH)
    assert U in elements.chemical_neighbors(ND)
    assert ND in elements.chemical_neighbors(U)


def test_neighbors_are_symmetric_and_exclude_self():
    for z in sorted(elements.host_palette("all")):
        neighbors = elements.chemical_neighbors(z)
        assert z not in neighbors
        assert neighbors == sorted(set(neighbors))
        for other in neighbors:
            assert z in elements.chemical_neighbors(other), f"{z}->{other} not symmetric"


def test_unknown_element_has_no_neighbors():
    assert elements.chemical_neighbors(999) == []


# --- Palette-restricted moves --------------------------------------------


def test_host_moves_match_the_documented_expansion():
    # The example from the expansion rule: LaBeH8's two host sites.
    assert elements.host_moves(LA, "high_tc") == sorted([Y, BA, CE])
    assert elements.host_moves(BE, "high_tc") == sorted([LI, B, MG])


def test_host_moves_drop_elements_outside_the_palette():
    # Be -> B is a covalent target, so it disappears under 'electropositive'.
    assert B in elements.host_moves(BE, "high_tc")
    assert B not in elements.host_moves(BE, "electropositive")
    assert elements.host_moves(BE, "electropositive") == sorted([LI, MG])


def test_host_moves_of_an_element_outside_the_palette_are_empty():
    # B is not electropositive, so under that palette its site is frozen.
    assert elements.host_moves(B, "electropositive") == []
    assert elements.host_moves(PD, "high_tc") == []
    assert elements.host_moves(PD, "all") != []


def test_host_moves_never_return_self_or_hydrogen():
    for palette in elements.PALETTE_NAMES:
        for z in sorted(elements.host_palette(palette)):
            moves = elements.host_moves(z, palette)
            assert z not in moves
            assert H not in moves


def test_transition_metals_only_reachable_under_all():
    assert CR not in elements.host_moves(V, "high_tc")
    assert CR in elements.host_moves(V, "all")


def test_palette_is_connected_enough_to_search():
    """Every electropositive element can move somewhere within its palette."""
    stranded = [
        z for z in sorted(elements.ELECTROPOSITIVE) if not elements.host_moves(z, "electropositive")
    ]
    assert stranded == []


# --- Starting-structure validation ---------------------------------------


def test_validate_hosts_accepts_a_ternary_hydride():
    assert elements.validate_hosts([LA] + [BE] + [H] * 8, "high_tc") == []


def test_validate_hosts_rejects_a_structure_without_hydrogen():
    with pytest.raises(ValueError, match="no hydrogen"):
        elements.validate_hosts([LA, BE], "high_tc")


def test_validate_hosts_rejects_elemental_hydrogen():
    with pytest.raises(ValueError, match="elemental hydrogen"):
        elements.validate_hosts([H, H, H], "high_tc")


def test_validate_hosts_rejects_a_template_with_no_movable_host():
    # Pd is outside high_tc, so nothing in PdH2 can move.
    with pytest.raises(ValueError, match="No host site"):
        elements.validate_hosts([PD, H, H], "high_tc")


def test_validate_hosts_warns_about_a_frozen_host():
    messages = elements.validate_hosts([LA, PD] + [H] * 8, "high_tc")
    assert any("stay fixed" in m for m in messages)


def test_validate_hosts_warns_about_a_binary_template():
    messages = elements.validate_hosts([CA] + [H] * 6, "high_tc")
    assert any("binary hydride" in m for m in messages)

"""
Unit tests for SuperhydrideStructure and the host-substitution expansion rule.

Structures are built programmatically so the suite needs no data files. The
XYH8 template used here is a synthetic cubic cell with two distinct host sites
and eight hydrogens - the shape of a ternary superhydride template, not a
physical claim about any particular compound.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import pytest

ase = pytest.importorskip("ase", reason="superhydride structures need ASE")
pytest.importorskip("spglib", reason="structure identifiers need spglib")

from mcts_framework.superhydride import (  # noqa: E402
    HostSubstitutionMoves,
    SuperhydrideStructure,
    elements,  # noqa: E402
)

LA, BE, H = 57, 4, 1


# Local aliases for the shared superhydride fixtures in tests/conftest.py.
# (They carry a `superhydride_` prefix there because that conftest is global.)


@pytest.fixture
def template(superhydride_template):
    return superhydride_template


@pytest.fixture
def make_structure(make_superhydride_structure):
    return make_superhydride_structure


# --- The material ---------------------------------------------------------


def test_formula_and_hydrogen_fraction(template):
    # ASE 'metal' mode lists the metals alphabetically, then the non-metals.
    assert template.get_formula() == "BeLaH8"
    assert template.get_hydrogen_fraction() == pytest.approx(0.8)


def test_host_elements_exclude_hydrogen(template):
    assert template.get_host_elements() == sorted([LA, BE])


def test_identifier_carries_formula_spacegroup_and_wyckoff(template):
    identifier = template.get_identifier()
    formula, spacegroup, decoration = identifier.split("|")
    assert formula == "BeLaH8"
    assert spacegroup.startswith("SG")
    assert 1 <= int(spacegroup[2:]) <= 230
    # One term per (element, Wyckoff letter) group, sorted.
    assert decoration.split("-") == sorted(decoration.split("-"))
    assert decoration.count("-") == 2  # La, Be, H


def test_identifier_is_independent_of_atom_order(template):
    shuffled = template.atoms[[9, 0, 3, 1, 7, 2, 8, 4, 6, 5]]
    assert SuperhydrideStructure(shuffled).get_identifier() == template.get_identifier()


def test_identifier_distinguishes_which_host_sits_on_which_site(make_structure):
    """
    Swapping the two hosts between sites gives the same composition and space
    group but a different material. Composition alone would alias them; the
    Wyckoff decoration is what keeps them apart.
    """
    normal = make_structure("La", "Be")
    swapped = make_structure("Be", "La")
    assert normal.get_formula() == swapped.get_formula()
    assert normal.get_identifier() != swapped.get_identifier()


def test_copy_is_independent(template):
    duplicate = template.copy()
    assert duplicate.get_identifier() == template.get_identifier()
    duplicate.atoms.set_atomic_numbers([39] + list(duplicate.atoms.get_atomic_numbers()[1:]))
    assert template.get_formula() == "BeLaH8"


def test_materials_are_deduplicated_by_identifier(template):
    assert template == template.copy()
    assert len({template, template.copy()}) == 1


# --- The expansion rule ---------------------------------------------------


def test_expansion_changes_exactly_one_host_per_move(template):
    children = HostSubstitutionMoves().generate_moves(template)
    parent_hosts = set(template.get_host_elements())
    assert children
    for child in children:
        changed = set(child.get_host_elements()) - parent_hosts
        assert len(changed) == 1, child.get_formula()


def test_expansion_reproduces_the_documented_children(template):
    """
    La -> {Y, Ba, Ce} and Be -> {Li, B, Mg}, one child each, and nothing else.
    """
    formulas = {c.get_formula() for c in HostSubstitutionMoves().generate_moves(template)}
    assert formulas == {
        "BeYH8", "BaBeH8", "BeCeH8",   # La site -> Y, Ba, Ce
        "LaLiH8", "LaBH8", "LaMgH8",   # Be site -> Li, B, Mg
    }


def test_branching_is_the_sum_over_hosts_not_the_product(template):
    """
    One host per move: |children| = sum of per-site options, so a ternary
    expands linearly rather than quadratically.
    """
    moves = HostSubstitutionMoves()
    children = moves.generate_moves(template)
    expected = sum(len(elements.host_moves(z, "high_tc")) for z in template.get_host_elements())
    assert len(children) == expected == 6


def test_hydrogen_is_never_substituted(template):
    for child in HostSubstitutionMoves().generate_moves(template):
        assert child.get_hydrogen_fraction() == pytest.approx(0.8)
        assert list(child.atoms.get_atomic_numbers()).count(H) == 8


def test_the_template_is_structurally_untouched(template):
    for child in HostSubstitutionMoves().generate_moves(template):
        assert child.atoms.get_cell()[:] == pytest.approx(template.atoms.get_cell()[:])
        assert child.atoms.get_scaled_positions() == pytest.approx(
            template.atoms.get_scaled_positions()
        )


def test_parent_is_not_mutated_by_expansion(template):
    before = template.get_formula()
    HostSubstitutionMoves().generate_moves(template)
    assert template.get_formula() == before


def test_children_are_distinct(template):
    children = HostSubstitutionMoves().generate_moves(template)
    identifiers = [c.get_identifier() for c in children]
    assert len(set(identifiers)) == len(identifiers)
    assert template.get_identifier() not in identifiers


def test_palette_narrows_the_expansion(template):
    wide = HostSubstitutionMoves(palette="high_tc").generate_moves(template)
    narrow = HostSubstitutionMoves(palette="electropositive").generate_moves(template)
    assert {c.get_formula() for c in narrow} < {c.get_formula() for c in wide}
    # B is covalent, so LaBH8 is unreachable under the electropositive palette.
    assert "LaBH8" not in {c.get_formula() for c in narrow}


def test_preserve_distinct_hosts_blocks_ternary_to_binary_collapse(make_structure):
    """
    In CeLaH8 the two hosts are lanthanide-chain neighbours, so La -> Ce would
    leave the binary Ce2H8. The default refuses that; opting out allows it.
    """
    parent = make_structure("La", "Ce")

    kept = {c.get_formula() for c in HostSubstitutionMoves().generate_moves(parent)}
    assert not {"Ce2H8", "La2H8"} & kept
    assert all(len(c.get_host_elements()) == 2 for c in
               HostSubstitutionMoves().generate_moves(parent))

    collapsed = {
        c.get_formula()
        for c in HostSubstitutionMoves(preserve_distinct_hosts=False).generate_moves(parent)
    }
    assert {"Ce2H8", "La2H8"} <= collapsed


def test_a_host_outside_the_palette_is_frozen(make_structure):
    """Pd is not in high_tc, so only the La site expands in LaPdH8."""
    parent = make_structure("La", "Pd")
    children = HostSubstitutionMoves(palette="high_tc").generate_moves(parent)
    assert children
    assert all("Pd" in c.get_formula() for c in children)


def test_unknown_palette_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown host palette"):
        HostSubstitutionMoves(palette="noble_gases")


def test_expansion_is_deterministic(template):
    moves = HostSubstitutionMoves()
    first = [c.get_formula() for c in moves.generate_moves(template)]
    second = [c.get_formula() for c in moves.generate_moves(template)]
    assert first == second


def test_expansion_walks_the_space_over_several_steps(template):
    """
    Single-site moves still reach compositions two host changes away, just one
    level deeper - which is the point of the rule: depth measures chemical
    distance from the starting composition.
    """
    moves = HostSubstitutionMoves()
    reachable = set()
    for child in moves.generate_moves(template):
        reachable.update(g.get_formula() for g in moves.generate_moves(child))
    assert "MgYH8" in reachable  # La->Y and Be->Mg, two steps

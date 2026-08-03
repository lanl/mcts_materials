"""
Tests for FunctionalGroupMoves.

Requires RDKit for real Mol objects; molecule-modifier is mocked via
sys.modules so we can verify our integration logic (how we call the API and
process its results) without the actual package or its model files.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import sys
import types

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mcts_framework.molecule.moves import FunctionalGroupMoves  # noqa: E402
from mcts_framework.molecule.structure import MolecularStructure  # noqa: E402


@pytest.fixture
def fake_molecule_modifier(monkeypatch):
    """
    Install a fake 'molecule_modifier' module whose functions exercise the
    real code paths of FunctionalGroupMoves.generate_moves.

    - find_substitution_sites: returns two dummy sites.
    - substitute: ignores sites, returns one molecule that is the parent SMILES
      with the functional group appended (so different groups yield different
      SMILES), producing deterministic, checkable children.
    - deduplicate_smiles: canonical-SMILES dedup via RDKit.
    """
    mod = types.ModuleType("molecule_modifier")

    def find_substitution_sites(mol, mode="hydrogen", target="C"):
        return [0, 1]  # two dummy sites

    def substitute(mol, mode="hydrogen", sites=None, new_group=None):
        # Build a new molecule = parent + new_group as disconnected fragment,
        # which gives a distinct, valid SMILES per functional group.
        combo = Chem.CombineMols(mol, new_group)
        return [combo]

    def deduplicate_smiles(smiles_list):
        seen = {}
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                seen[Chem.MolToSmiles(m)] = None
        return list(seen)

    mod.find_substitution_sites = find_substitution_sites
    mod.substitute = substitute
    mod.deduplicate_smiles = deduplicate_smiles
    monkeypatch.setitem(sys.modules, "molecule_modifier", mod)
    return mod


def test_generate_moves_produces_children(fake_molecule_modifier):
    parent = MolecularStructure.from_smiles("CCO")  # ethanol
    gen = FunctionalGroupMoves(functional_groups=["C", "N"])
    children = gen.generate_moves(parent)

    # Two functional groups -> two distinct children (methyl vs amino fragment).
    assert len(children) == 2
    for child in children:
        assert isinstance(child, MolecularStructure)


def test_generate_moves_deduplicates(fake_molecule_modifier):
    """Identical functional groups should collapse to one child."""
    parent = MolecularStructure.from_smiles("CCO")
    gen = FunctionalGroupMoves(functional_groups=["C", "C"])
    children = gen.generate_moves(parent)
    assert len(children) == 1


def test_generate_moves_no_sites_returns_empty(monkeypatch, fake_molecule_modifier):
    """If no substitution sites are found, no children are generated."""
    fake_molecule_modifier.find_substitution_sites = lambda *a, **k: []
    parent = MolecularStructure.from_smiles("CCO")
    gen = FunctionalGroupMoves(functional_groups=["C"])
    assert gen.generate_moves(parent) == []


def test_generate_moves_skips_bad_functional_group(fake_molecule_modifier):
    """An unparseable functional-group SMILES is skipped, not fatal."""
    parent = MolecularStructure.from_smiles("CCO")
    gen = FunctionalGroupMoves(functional_groups=["C", "not_a_smiles"])
    children = gen.generate_moves(parent)
    # Only the valid 'C' group yields a child.
    assert len(children) == 1


def test_empty_functional_groups_rejected():
    with pytest.raises(ValueError):
        FunctionalGroupMoves(functional_groups=[])
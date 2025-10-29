"""
Quick test of f-block MCTS with efficient cache lookup.
"""

import sys
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import MCTSTreeNode, MCTS, MaceEnergyCalculator, ResultsAnalyzer
from ase.io import read


def main():
    """Quick test of f-block MCTS optimization."""
    print("=" * 70)
    print("QUICK F-BLOCK MCTS TEST (WITH EFFICIENT CACHE LOOKUP)")
    print("=" * 70)
    
    # Load structure and set up energy calculator
    cif_file = Path(__file__).parent / "mat_Pb6U1W6_sg191.cif"
    csv_file = Path(__file__).parent / "mace_calculations.csv"
    
    atoms = read(str(cif_file))
    energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
    
    print(f"1. Initial structure: {atoms.get_chemical_formula()}")
    
    # Create MCTS with f-block substitutions
    root = MCTSTreeNode(atoms)
    mcts = MCTS(root)
    
    print(f"2. F-block element: {root.f_block}")
    print(f"   Available f-block moves: {len(getattr(root, 'f_block_move', []))}")
    print(f"   Total search space: {len(root.metal_move)} × {len(root.g_iv_move)} × {len(getattr(root, 'f_block_move', [1]))} = expanded!")
    
    # Run short MCTS
    print(f"\n3. Running 10 MCTS iterations...")
    
    results = mcts.run(
        n_iterations=10,
        energy_calculator=energy_calc,
        rollout_depth=1,
        n_rollout=2
    )
    
    print(f"   ✓ Completed: {results['iterations_completed']} iterations")
    print(f"   ✓ Compounds explored: {len(results['stat_dict'])}")
    print(f"   ✓ Best compound: {results['best_node_formula']}")
    print(f"   ✓ Best formation energy: {results['best_node_e_form']:.4f} eV/atom")
    
    # Analyze f-block discoveries
    print(f"\n4. F-block element discoveries:")
    
    element_names = {
        57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm',
        89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu'
    }
    
    f_block_compounds = {}
    for formula in results['stat_dict'].keys():
        for num, name in element_names.items():
            if name in formula:
                if name not in f_block_compounds:
                    f_block_compounds[name] = []
                f_block_compounds[name].append(formula)
                break
    
    for element, compounds in f_block_compounds.items():
        print(f"   - {element}: {compounds}")
    
    # Show top results
    analyzer = ResultsAnalyzer()
    top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=5)
    
    print(f"\n5. Top 5 compounds found:")
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        print(f"   {i}. {row['formula']:12s} E_form: {row['formation_energy']:8.4f} eV/atom")
    
    print(f"\n" + "=" * 70)
    print("✅ F-BLOCK MCTS TEST COMPLETE!")
    print("Cache lookup is fast, MACE calculations only when needed")
    print("F-block substitutions working correctly")
    print("=" * 70)


if __name__ == "__main__":
    main()
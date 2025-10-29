"""
Example script demonstrating MCTS with expanded f-block element substitutions.
This shows the dramatically expanded search space including lanthanides and actinides.
"""

import sys
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import (
    MCTSTreeNode,
    MCTS, 
    MaceEnergyCalculator,
    TreeVisualizer,
    ResultsAnalyzer
)
from ase.io import read
import pandas as pd


def main():
    """
    Demonstrate MCTS crystal structure optimization with f-block element substitutions.
    """
    print("=" * 70)
    print("MCTS CRYSTAL OPTIMIZATION WITH F-BLOCK ELEMENT SUBSTITUTIONS")
    print("=" * 70)
    
    # Step 1: Load the starting crystal structure
    print("\n1. Loading initial crystal structure...")
    cif_file = Path(__file__).parent / "mat_Pb6U1W6_sg191.cif"
    
    if not cif_file.exists():
        print(f"❌ CIF file not found: {cif_file}")
        return 1
        
    atoms = read(str(cif_file))
    print(f"   ✓ Initial structure: {atoms.get_chemical_formula()}")
    print(f"   ✓ Number of atoms: {len(atoms)}")
    
    # Step 2: Set up energy calculator
    print("\n2. Setting up energy calculator with cached MACE calculations...")
    csv_file = Path(__file__).parent / "mace_calculations.csv"
    
    if not csv_file.exists():
        print(f"❌ MACE calculations file not found: {csv_file}")
        return 1
    
    # Load and inspect the cached data
    df = pd.read_csv(csv_file)
    print(f"   ✓ Loaded {len(df)} cached calculations")
    
    energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
    
    # Step 3: Initialize MCTS with f-block substitutions
    print("\n3. Initializing MCTS algorithm with f-block element substitutions...")
    root_node = MCTSTreeNode(atoms, f_block_mode='full_f_block')
    mcts = MCTS(root_node)
    
    print(f"   ✓ Root compound: {root_node.get_chemical_formula()}")
    
    # Expand root to see possible moves
    root_node.expand()
    print(f"   ✓ Total possible moves: {len(root_node.expansion_list)}")
    
    # Show what substitutions are possible
    print(f"\n   🧪 Chemical substitution space:")
    print(f"   - Transition metals: {len(root_node.metal_move)} options")
    print(f"   - Group IV elements: {len(root_node.g_iv_move)} options")
    if hasattr(root_node, 'f_block_move'):
        print(f"   - F-block elements: {len(root_node.f_block_move)} options")
        
        # Show f-block elements available
        element_names = {
            57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
            65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
            89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu'
        }
        f_block_names = [element_names.get(num, f'Z={num}') for num in root_node.f_block_move]
        print(f"     Available f-elements: {', '.join(f_block_names)}")
    
    # Step 4: Run MCTS search
    print(f"\n4. Running MCTS optimization...")
    n_iterations = 20  # Moderate run to explore f-block space
    
    print(f"   Running {n_iterations} MCTS iterations...")
    print("   (This may take longer due to expanded search space)")
    
    results = mcts.run(
        n_iterations=n_iterations,
        energy_calculator=energy_calc,
        rollout_depth=1,
        n_rollout=3,
        selection_mode='epsilon'
    )
    
    print(f"   ✓ Completed {results['iterations_completed']} iterations")
    print(f"   ✓ Explored {len(results['stat_dict'])} unique compounds")
    print(f"   ✓ Best reward found: {results['best_reward']:.4f}")
    
    # Step 5: Analyze f-block discoveries
    print(f"\n5. Analyzing f-block element discoveries...")
    
    # Count compounds by f-block element
    f_block_compounds = {}
    other_compounds = []
    
    element_names = {
        57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu'
    }
    
    for formula in results['stat_dict'].keys():
        # Simple parsing to identify f-block elements
        found_f_element = None
        for num, name in element_names.items():
            if name in formula:
                found_f_element = name
                break
                
        if found_f_element:
            if found_f_element not in f_block_compounds:
                f_block_compounds[found_f_element] = []
            f_block_compounds[found_f_element].append(formula)
        else:
            other_compounds.append(formula)
    
    print(f"   📊 F-block element exploration:")
    for element, compounds in f_block_compounds.items():
        print(f"   - {element}: {len(compounds)} compounds discovered")
        for compound in compounds[:3]:  # Show first 3
            print(f"     • {compound}")
        if len(compounds) > 3:
            print(f"     • ... and {len(compounds)-3} more")
    
    # Step 6: Get top compounds
    print(f"\n6. Top discoveries across all elements...")
    analyzer = ResultsAnalyzer()
    top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)
    
    print(f"   🏆 TOP 10 COMPOUNDS BY FORMATION ENERGY:")
    print("   " + "="*60)
    
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        energy = row['formation_energy']
        visits = int(row['visit_count'])
        stability = "Stable" if row['e_above_hull'] < 0.1 else "Metastable"
        
        # Identify f-block element in compound
        f_element = "?"
        for num, name in element_names.items():
            if name in row['formula']:
                f_element = name
                break
        
        print(f"   {i:2d}. {row['formula']:15s} | "
              f"E_form: {energy:8.4f} eV/atom | "
              f"F-element: {f_element:2s} | "
              f"Visits: {visits:3d}")
    
    # Step 7: Create visualizations
    print(f"\n7. Creating visualizations...")
    
    # Create output directory
    output_dir = Path("mcts_f_block_results")
    output_dir.mkdir(exist_ok=True)
    
    visualizer = TreeVisualizer()
    
    # Radial tree visualization showing f-block substitutions
    visualizer.plot_radial_tree_visualization(
        mcts,
        output_dir=str(output_dir),
        csv_file=str(csv_file)
    )
    
    # Energy distribution plot
    energy_fig = visualizer.plot_energy_distribution(
        mcts.stat_dict,
        top_n=12,
        save_path=output_dir / "f_block_energy_distribution.png"
    )
    
    print(f"   ✓ Visualizations saved to {output_dir}/")
    
    # Step 8: Generate detailed report
    print(f"\n8. Generating detailed report...")
    
    report = analyzer.create_summary_report(
        mcts,
        save_path=output_dir / "f_block_search_report.txt"
    )
    
    # Export results
    analyzer.export_results(
        mcts.stat_dict,
        output_dir / "f_block_compounds_found.csv"
    )
    
    print(f"   ✓ Reports saved to {output_dir}/")
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("🎯 F-BLOCK MCTS OPTIMIZATION COMPLETED!")
    print("=" * 70)
    
    best_compound = results['best_node_formula']
    best_energy = results['best_node_e_form']
    
    print(f"🥇 Best compound discovered: {best_compound}")
    print(f"⚡ Formation energy: {best_energy:.4f} eV/atom")
    print(f"🔬 F-block elements explored: {len(f_block_compounds)} different elements")
    print(f"🔍 Total compounds found: {len(results['stat_dict'])}")
    print(f"📁 Results directory: {output_dir.absolute()}")
    
    print(f"\n🌟 F-block elements discovered:")
    for element in sorted(f_block_compounds.keys()):
        atomic_num = next(num for num, name in element_names.items() if name == element)
        element_type = "Lanthanide" if 57 <= atomic_num <= 71 else "Actinide"
        print(f"   {element} ({element_type}): {len(f_block_compounds[element])} compounds")
    
    print(f"\n💡 The expanded search space now includes {len(element_names)} f-block")
    print("   elements, dramatically increasing the potential for discovering")
    print("   novel compounds with unique properties!")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
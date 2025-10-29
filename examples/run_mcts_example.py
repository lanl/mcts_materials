"""
Example script demonstrating MCTS crystal structure optimization.

This script shows how to:
1. Load a crystal structure
2. Set up MCTS optimization
3. Run the search
4. Analyze results
5. Create visualizations
"""

import os
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
from ase import Atoms


def create_example_structure():
    """
    Create an example crystal structure for demonstration.
    This creates a simple 1:6:6 structure similar to the DyFe6Sn6 prototype.
    """
    atoms = Atoms('UFe6Sn6', positions=[
        [0.0, 0.0, 0.0],     # U at origin
        [2.0, 0.0, 1.0],     # Fe 1
        [0.0, 2.0, 1.0],     # Fe 2
        [-2.0, 0.0, 1.0],    # Fe 3
        [0.0, -2.0, 1.0],    # Fe 4
        [2.0, 2.0, -1.0],    # Fe 5
        [-2.0, -2.0, -1.0],  # Fe 6
        [1.0, 1.0, 0.0],     # Sn 1
        [-1.0, 1.0, 0.0],    # Sn 2
        [1.0, -1.0, 0.0],    # Sn 3
        [-1.0, -1.0, 0.0],   # Sn 4
        [0.0, 0.0, 2.0],     # Sn 5
        [0.0, 0.0, -2.0],    # Sn 6
    ])
    
    # Set hexagonal-like cell
    atoms.set_cell([6, 6, 4])
    atoms.set_pbc(True)
    
    return atoms


def load_structure_from_cif():
    """
    Load structure from CIF file if available.
    """
    cif_file = Path(__file__).parent / "DyFe6Sn6_ICSD655986.cif"
    
    if cif_file.exists():
        print(f"Loading structure from {cif_file}")
        atoms = read(str(cif_file))
        
        # Replace Dy with U for our search (since we're looking at U-based compounds)
        symbols = atoms.get_chemical_symbols()
        for i, symbol in enumerate(symbols):
            if symbol == 'Dy':
                symbols[i] = 'U'
        atoms.set_chemical_symbols(symbols)
        
        return atoms
    else:
        print("CIF file not found, using example structure")
        return create_example_structure()


def main():
    """
    Main example workflow.
    """
    print("=" * 70)
    print("MCTS CRYSTAL STRUCTURE OPTIMIZATION EXAMPLE")
    print("=" * 70)
    
    # Step 1: Load or create initial structure
    print("\n1. Loading initial crystal structure...")
    initial_atoms = load_structure_from_cif()
    print(f"   Initial formula: {initial_atoms.get_chemical_formula()}")
    print(f"   Number of atoms: {len(initial_atoms)}")
    
    # Step 2: Set up energy calculator
    print("\n2. Setting up energy calculator...")
    csv_file = Path(__file__).parent.parent / "mace_calculations.csv"
    
    if csv_file.exists():
        print(f"   Using cached calculations from: {csv_file}")
        energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
    else:
        print("   Warning: mace_calculations.csv not found, using minimal calculator")
        # Create minimal dummy data for demonstration
        import tempfile
        import pandas as pd
        
        dummy_data = {
            'name': ['Fe6Sn6U', 'Co6Sn6U', 'Ni6Sn6U', 'Mn6Sn6U', 'Ti6Sn6U'],
            'e_form': [-0.1, -0.15, -0.12, -0.08, -0.05],
            'e_above_hull': [0.05, 0.02, 0.04, 0.08, 0.12],
            'e_decomp': [0.15, 0.17, 0.16, 0.16, 0.17]
        }
        df = pd.DataFrame(dummy_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_csv = f.name
            
        energy_calc = MaceEnergyCalculator(csv_file=temp_csv)
    
    # Step 3: Initialize MCTS
    print("\n3. Initializing MCTS...")
    root_node = MCTSTreeNode(initial_atoms)
    mcts = MCTS(root_node)
    
    print(f"   Root compound: {root_node.get_chemical_formula()}")
    print(f"   Possible metal substitutions: {len(root_node.metal_move)}")
    print(f"   Possible Group IV substitutions: {len(root_node.g_iv_move)}")
    print(f"   Total possible moves: {len(root_node.expansion_list)}")
    
    # Step 4: Run MCTS search
    print("\n4. Running MCTS search...")
    n_iterations = 20  # Small number for demonstration
    print(f"   Running {n_iterations} iterations...")
    
    results = mcts.run(
        n_iterations=n_iterations,
        energy_calculator=energy_calc,
        rollout_depth=1,
        n_rollout=5,
        selection_mode='epsilon'
    )
    
    print(f"   ✓ Completed {results['iterations_completed']} iterations")
    print(f"   ✓ Explored {len(results['stat_dict'])} unique compounds")
    print(f"   ✓ Best reward: {results['best_reward']:.4f}")
    if results['best_node_formula']:
        print(f"   ✓ Best compound: {results['best_node_formula']}")
        print(f"   ✓ Formation energy: {results['best_node_e_form']:.4f} eV/atom")
    
    # Step 5: Analyze results
    print("\n5. Analyzing results...")
    analyzer = ResultsAnalyzer()
    
    # Get top 10 compounds
    top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)
    
    # Get efficiency metrics
    efficiency = analyzer.analyze_search_efficiency(mcts.stat_dict)
    
    # Print summary
    print(f"   Search efficiency: {efficiency['search_diversity']:.4f}")
    print(f"   Compounds near hull (100 meV): {efficiency['compounds_near_hull_100meV']}")
    print(f"   Best formation energy found: {efficiency['best_formation_energy']:.4f} eV/atom")
    
    # Print top compounds
    print("\n   Top 10 compounds by formation energy:")
    print("   " + "-" * 60)
    for i, (_, row) in enumerate(top_compounds.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['formula']:15s} | "
              f"E_form: {row['formation_energy']:8.4f} eV/atom | "
              f"Visits: {row['visit_count']:3.0f}")
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = TreeVisualizer()
    
    # Create output directory
    output_dir = Path("mcts_results")
    output_dir.mkdir(exist_ok=True)
    
    # Tree expansion plot
    tree_fig = visualizer.plot_tree_expansion(
        mcts, 
        max_depth=3,
        save_path=output_dir / "tree_expansion.png"
    )
    
    # Energy distribution plot
    energy_fig = visualizer.plot_energy_distribution(
        mcts.stat_dict,
        top_n=15,
        save_path=output_dir / "energy_distribution.png"
    )
    
    # Summary plot
    summary_fig = visualizer.create_summary_plot(
        mcts,
        save_path=output_dir / "summary.png"
    )
    
    print(f"   ✓ Visualizations saved to {output_dir}/")
    
    # Step 7: Generate reports
    print("\n7. Generating reports...")
    
    # Summary report
    report = analyzer.create_summary_report(
        mcts,
        save_path=output_dir / "mcts_summary_report.txt"
    )
    
    # Export detailed results
    analyzer.export_results(
        mcts.stat_dict,
        output_dir / "detailed_results.csv"
    )
    
    # Save MCTS statistics
    mcts.save_statistics(output_dir / "mcts_statistics.csv")
    
    print(f"   ✓ Reports saved to {output_dir}/")
    
    # Step 8: Display summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"🎯 Best compound found: {results['best_node_formula']}")
    print(f"⚡ Formation energy: {results['best_node_e_form']:.4f} eV/atom")
    print(f"🔍 Total compounds explored: {len(results['stat_dict'])}")
    print(f"📊 Results saved in: {output_dir.absolute()}")
    print("=" * 70)
    
    # Clean up temporary file if created
    if 'temp_csv' in locals():
        os.unlink(temp_csv)


if __name__ == "__main__":
    main()
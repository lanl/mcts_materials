"""
Example script demonstrating MCTS with the cached mace_calculations.csv data.
This shows how the algorithm uses cached calculations to efficiently explore the chemical space.
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
    Demonstrate MCTS crystal structure optimization using cached MACE calculations.
    """
    print("=" * 70)
    print("MCTS CRYSTAL OPTIMIZATION WITH CACHED MACE CALCULATIONS")
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
    print(f"   ✓ Space group: P6/mmm (191)")
    
    # Step 2: Set up energy calculator with cached MACE data
    print("\n2. Setting up energy calculator with cached MACE calculations...")
    csv_file = Path(__file__).parent / "mace_calculations.csv"
    
    if not csv_file.exists():
        print(f"❌ MACE calculations file not found: {csv_file}")
        return 1
    
    # Load and inspect the cached data
    df = pd.read_csv(csv_file)
    print(f"   ✓ Loaded {len(df)} cached calculations")
    print(f"   ✓ Energy range: {df['e_form'].min():.4f} to {df['e_form'].max():.4f} eV/atom")
    
    # Show a few examples
    print("\n   Sample cached calculations:")
    for _, row in df.head(5).iterrows():
        print(f"   - {row['name']:12s}: E_form = {row['e_form']:7.4f} eV/atom")
    
    energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
    
    # Step 3: Initialize MCTS
    print("\n3. Initializing MCTS algorithm...")
    root_node = MCTSTreeNode(atoms)
    mcts = MCTS(root_node)
    
    print(f"   ✓ Root compound: {root_node.get_chemical_formula()}")
    
    # Expand root to see possible moves
    root_node.expand()
    print(f"   ✓ Possible transitions: {len(root_node.expansion_list)}")
    
    # Show what transitions are possible
    print("\n   Available chemical substitutions:")
    print(f"   - Transition metals: {root_node.metal_move}")
    print(f"   - Group IV elements: {root_node.g_iv_move}")
    
    # Step 4: Run MCTS search (short run for demonstration)
    print("\n4. Running MCTS optimization...")
    n_iterations = 15  # Short run to demonstrate
    
    print(f"   Running {n_iterations} MCTS iterations...")
    
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
    
    # Step 5: Analyze results
    print("\n5. Analyzing results...")
    analyzer = ResultsAnalyzer()
    
    # Get top compounds
    top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=8)
    
    print(f"   Search completed! Found {len(top_compounds)} compounds.")
    print("\n   🏆 TOP COMPOUNDS BY FORMATION ENERGY:")
    print("   " + "="*55)
    
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        energy = row['formation_energy']
        visits = int(row['visit_count'])
        stability = "Stable" if row['e_above_hull'] < 0.1 else "Metastable"
        
        print(f"   {i:2d}. {row['formula']:12s} | "
              f"E_form: {energy:8.4f} eV/atom | "
              f"Visits: {visits:3d} | {stability}")
    
    # Check how many were found in cache vs calculated
    cached_count = 0
    calculated_count = 0
    
    for formula in mcts.stat_dict.keys():
        if formula in df['name'].values:
            cached_count += 1
        else:
            calculated_count += 1
    
    print(f"\n   📊 Results breakdown:")
    print(f"   - Compounds found in cache: {cached_count}")
    print(f"   - New calculations needed: {calculated_count}")
    print(f"   - Cache hit rate: {cached_count/(cached_count+calculated_count)*100:.1f}%")
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    
    # Create output directory
    output_dir = Path("mcts_results_cached")
    output_dir.mkdir(exist_ok=True)
    
    visualizer = TreeVisualizer()
    
    # Energy distribution plot
    energy_fig = visualizer.plot_energy_distribution(
        mcts.stat_dict,
        top_n=12,
        save_path=output_dir / "energy_distribution.png"
    )
    
    # Radial tree visualization (new format)
    visualizer.plot_radial_tree_visualization(
        mcts,
        output_dir=str(output_dir),
        csv_file=str(csv_file)
    )
    
    # Summary plot
    summary_fig = visualizer.create_summary_plot(
        mcts,
        save_path=output_dir / "mcts_summary.png"
    )
    
    print(f"   ✓ Visualizations saved to {output_dir}/")
    
    # Step 7: Generate detailed report
    print("\n7. Generating detailed report...")
    
    report = analyzer.create_summary_report(
        mcts,
        save_path=output_dir / "detailed_report.txt"
    )
    
    # Export results
    analyzer.export_results(
        mcts.stat_dict,
        output_dir / "all_compounds_found.csv"
    )
    
    print(f"   ✓ Reports saved to {output_dir}/")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 MCTS OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    best_compound = results['best_node_formula']
    best_energy = results['best_node_e_form']
    
    print(f"🥇 Best compound discovered: {best_compound}")
    print(f"⚡ Formation energy: {best_energy:.4f} eV/atom")
    print(f"🔍 Total compounds explored: {len(results['stat_dict'])}")
    print(f"📈 Search efficiency: {cached_count/(cached_count+calculated_count)*100:.1f}% cache hit rate")
    print(f"📁 Results directory: {output_dir.absolute()}")
    
    print("\n💡 The MCTS algorithm successfully used cached MACE calculations")
    print("   to efficiently explore the chemical space and identify")
    print("   promising compounds with low formation energies!")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
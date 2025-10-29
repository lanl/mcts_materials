"""
Simple test for the radial tree visualization feature.
"""

import sys
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import MCTSTreeNode, MCTS, MaceEnergyCalculator, TreeVisualizer
from ase.io import read


def main():
    """Test the radial tree visualization."""
    print("Testing Radial Tree Visualization...")
    
    # Load structure
    cif_file = Path(__file__).parent / "mat_Pb6U1W6_sg191.cif"
    atoms = read(str(cif_file))
    
    # Set up MCTS with cached energy data
    csv_file = Path(__file__).parent / "mace_calculations.csv"
    energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
    
    # Run a short MCTS to create some tree structure
    root = MCTSTreeNode(atoms)
    mcts = MCTS(root)
    
    print("Running short MCTS to build tree structure...")
    results = mcts.run(
        n_iterations=8,
        energy_calculator=energy_calc,
        rollout_depth=1,
        n_rollout=2
    )
    
    print(f"Created tree with {len(results['stat_dict'])} nodes")
    
    # Create radial tree visualization
    output_dir = Path("radial_tree_test")
    output_dir.mkdir(exist_ok=True)
    
    visualizer = TreeVisualizer()
    
    print("Creating radial tree visualization...")
    visualizer.plot_radial_tree_visualization(
        mcts,
        output_dir=str(output_dir),
        csv_file=str(csv_file)
    )
    
    print(f"✅ Radial tree visualization saved to {output_dir}/radial_tree_visualization.png")
    print(f"📊 Tree shows formation energies from cached MACE calculations")
    print(f"🌈 Node colors indicate stability (green=stable, yellow=metastable, red=unstable)")
    

if __name__ == "__main__":
    main()
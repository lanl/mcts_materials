#!/usr/bin/env python3
"""
Simple MCTS Crystal Structure Optimization Runner

Usage:
    python run_mcts.py [OPTIONS]

Examples:
    python run_mcts.py                                    # Default: 1000 iterations, weighted rollout
    python run_mcts.py --iterations 100                  # 100 iterations
    python run_mcts.py --structure my_structure.cif      # Custom structure
    python run_mcts.py --rollout-method fe               # Formation energy rollouts only
    python run_mcts.py --rollout-method eh               # Energy above hull rollouts only
    python run_mcts.py --rollout-method dos              # DOSCAR rewards only
    python run_mcts.py --rollout-method weighted --alpha 2.0 --beta 3.0 --gamma 1.0  # Custom weights with DOSCAR
    python run_mcts.py --f-block-mode lanthanides_u      # Lanthanides + Uranium mode
    python run_mcts.py --exploration-constant 0.2        # Higher exploration (default: 0.1)
    python run_mcts.py --no-labels                       # Turn off labels on radial tree visualization
    python run_mcts.py --iterations 200 --structure my_structure.cif --rollout-method fe
"""

import sys
import argparse
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcts_crystal import (
    MCTSTreeNode,
    MCTS,
    MaceEnergyCalculator,
    TreeVisualizer,
    ResultsAnalyzer,
    DoscarRewardLookup
)
from ase.io import read
import pandas as pd


def main():
    """Main MCTS runner function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MCTS crystal structure optimization')
    parser.add_argument('--iterations', '-n', type=int, default=1000,
                       help='Number of MCTS iterations (default: 1000)')
    parser.add_argument('--structure', '-s', type=str, 
                       default='examples/mat_Pb6U1W6_sg191.cif',
                       help='Path to starting crystal structure CIF file')
    parser.add_argument('--output', '-o', type=str, default='mcts_results',
                       help='Output directory name (default: mcts_results)')
    parser.add_argument('--f-block-mode', type=str, default='u_only',
                       choices=['u_only', 'full_f_block', 'experimental', 'lanthanides_u'],
                       help='F-block substitution mode: u_only (default), full_f_block, experimental (actinides except Ac), or lanthanides_u (lanthanides + U)')
    parser.add_argument('--exploration-constant', '-c', type=float, default=0.1,
                       help='Exploration constant for UCB calculation (default: 0.1)')
    parser.add_argument('--rollout-method', type=str, default='weighted',
                       choices=['fe', 'eh', 'both', 'weighted', 'dos'],
                       help='Rollout method: fe (formation energy), eh (energy above hull), both (mix of fe and eh), weighted (tunable combination, default), or dos (DOSCAR rewards only)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for formation energy in weighted rollout method (default: 1.0). Reward = alpha*(-e_form) + beta*(-e_above_hull) + gamma*(doscar_reward)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for energy above hull in weighted rollout method (default: 1.0). Reward = alpha*(-e_form) + beta*(-e_above_hull) + gamma*(doscar_reward)')
    parser.add_argument('--gamma', type=float, default=0.0,
                       help='Weight for DOSCAR reward in weighted rollout method (default: 0.0). Reward = alpha*(-e_form) + beta*(-e_above_hull) + gamma*(doscar_reward)')
    parser.add_argument('--termination-limit', type=int, default=60,
                       help='Number of visits before terminating a node without improvement (default: 60)')
    parser.add_argument('--mp-api-key', type=str, default=None,
                       help='Materials Project API key (required for rollout methods: eh, both, weighted)')
    parser.add_argument('--no-labels', action='store_true',
                       help='Turn off labels on radial tree visualization (default: labels shown)')

    args = parser.parse_args()

    # Validate that MP API key is provided when needed
    methods_requiring_api_key = ['eh', 'both', 'weighted']
    if args.rollout_method in methods_requiring_api_key and args.mp_api_key is None:
        print(f"❌ Error: --mp-api-key is required when using rollout method '{args.rollout_method}'")
        print(f"   Energy above hull calculations require Materials Project API access")
        print(f"   Get your API key from: https://materialsproject.org/api")
        print(f"   Then run with: --mp-api-key YOUR_KEY")
        return 1

    # Check if DOSCAR rewards file exists for 'dos' method
    if args.rollout_method == 'dos':
        doscar_file = Path("doscar_rewards.csv")
        if not doscar_file.exists():
            print(f"❌ Error: doscar_rewards.csv not found for rollout method 'dos'")
            print(f"   Please ensure doscar_rewards.csv is in the working directory")
            print(f"   Run: python utils/calculate_doscar_rewards.py")
            return 1
    
    print("=" * 80)
    print("MCTS CRYSTAL STRUCTURE OPTIMIZATION")
    print("=" * 80)
    
    # Step 1: Load starting crystal structure
    print(f"\n1. Loading starting crystal structure...")
    structure_path = Path(args.structure)
    
    if not structure_path.exists():
        print(f"❌ Error: Structure file not found: {structure_path}")
        print(f"   Please check the file path or use the default structure")
        return 1
        
    try:
        atoms = read(str(structure_path))
        print(f"   ✓ Loaded: {atoms.get_chemical_formula()}")
        print(f"   ✓ File: {structure_path}")
        print(f"   ✓ Atoms: {len(atoms)}")
    except Exception as e:
        print(f"❌ Error loading structure: {e}")
        return 1
    
    # Step 2: Set up energy calculator
    print(f"\n2. Setting up energy calculator...")
    csv_file = Path("high_throughput_mace_results.full.csv")

    if not csv_file.exists():
        print(f"❌ Error: MACE calculations file not found: {csv_file}")
        print(f"   Please ensure high_throughput_mace_results.full.csv is in the working directory")
        return 1
    
    try:
        df = pd.read_csv(csv_file)
        energy_calc = MaceEnergyCalculator(csv_file=str(csv_file), mp_api_key=args.mp_api_key)
        print(f"   ✓ Cached calculations: {len(df)} entries")
        print(f"   ✓ Energy range: {df['e_form'].min():.3f} to {df['e_form'].max():.3f} eV/atom")
        if args.mp_api_key:
            print(f"   ✓ Materials Project API key provided")
        else:
            print(f"   ⚠ No MP API key - energy above hull will be approximate (e_above_hull = e_form)")
    except Exception as e:
        print(f"❌ Error setting up energy calculator: {e}")
        return 1

    # Load DOSCAR rewards if gamma > 0 or using 'dos' rollout method
    doscar_lookup = None
    if args.gamma > 0 or args.rollout_method == 'dos':
        print(f"\n2.5. Loading DOSCAR rewards...")
        try:
            doscar_lookup = DoscarRewardLookup()
        except Exception as e:
            print(f"❌ Error loading DOSCAR rewards: {e}")
            if args.rollout_method == 'dos':
                print(f"   Cannot continue without DOSCAR rewards for 'dos' method")
                return 1
            else:
                print(f"   Continuing without DOSCAR rewards (gamma will be ignored)")
                doscar_lookup = None
    
    # Step 3: Initialize MCTS
    print(f"\n3. Initializing MCTS algorithm...")
    try:
        root_node = MCTSTreeNode(atoms, f_block_mode=args.f_block_mode,
                                exploration_constant=args.exploration_constant,
                                termination_limit=args.termination_limit)
        mcts = MCTS(root_node)

        print(f"   ✓ Root compound: {root_node.get_chemical_formula()}")
        print(f"   ✓ F-block mode: {args.f_block_mode}")
        print(f"   ✓ Exploration constant: {args.exploration_constant}")
        print(f"   ✓ Termination limit: {args.termination_limit}")
        
        # Show search space
        root_node.expand()
        print(f"   ✓ Search space: {len(root_node.expansion_list)} possible moves")
        print(f"   ✓ Transition metals: {len(root_node.metal_move)} options")
        print(f"   ✓ Group IV elements: {len(root_node.g_iv_move)} options")
        if hasattr(root_node, 'f_block_move'):
            if args.f_block_mode == 'u_only':
                print(f"   ✓ F-block elements: {len(root_node.f_block_move)} options (U-only mode)")
            else:
                print(f"   ✓ F-block elements: {len(root_node.f_block_move)} options (full f-block)")
            
    except Exception as e:
        print(f"❌ Error initializing MCTS: {e}")
        return 1
    
    # Step 4: Run MCTS optimization
    print(f"\n4. Running MCTS optimization...")
    print(f"   Iterations: {args.iterations}")
    print(f"   Rollout method: {args.rollout_method}")
    if args.rollout_method == 'weighted':
        print(f"   Alpha (e_form weight): {args.alpha}")
        print(f"   Beta (e_above_hull weight): {args.beta}")
        print(f"   Gamma (DOSCAR reward weight): {args.gamma}")
    print(f"   This may take several minutes depending on cache hit rate...")

    try:
        results = mcts.run(
            n_iterations=args.iterations,
            energy_calculator=energy_calc,
            rollout_depth=1,
            n_rollout=5,
            selection_mode='epsilon',
            rollout_method=args.rollout_method,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            doscar_lookup=doscar_lookup
        )
        
        print(f"   ✓ Completed: {results['iterations_completed']} iterations")
        print(f"   ✓ Compounds explored: {len(results['stat_dict'])}")
        print(f"   ✓ Search terminated: {results['terminated']}")
        
    except Exception as e:
        print(f"❌ Error during MCTS run: {e}")
        return 1
    
    # Step 4.5: Reconcile MCTS tree with MACE calculations CSV
    print(f"\n4.5. Reconciling MCTS tree with MACE calculations CSV...")
    try:
        # Load the existing calculations
        mace_df = pd.read_csv(csv_file)
        existing_formulas = set(mace_df['name'])
        
        # Get all compounds from the MCTS tree
        all_tree_compounds = mcts.get_statistics_dataframe()
        
        new_records = []
        for formula, row in all_tree_compounds.iterrows():
            if formula not in existing_formulas:
                new_records.append({
                    'name': formula,
                    'e_form': row['e_form'],
                    'e_above_hull': row['e_above_hull'],
                    'e_decomp': 0  # No decomposition info available
                })
        
        if new_records:
            new_records_df = pd.DataFrame(new_records)
            updated_df = pd.concat([mace_df, new_records_df], ignore_index=True)
            updated_df.to_csv(csv_file, index=False)
            print(f"   ✓ Added {len(new_records)} new compounds to {csv_file}")
        else:
            print("   ✓ MACE calculations CSV is already up-to-date.")

    except Exception as e:
        print(f"❌ Error during MCTS reconciliation: {e}")
        return 1

    # Step 5: Analyze results
    print(f"\n5. Analyzing results...")
    try:
        analyzer = ResultsAnalyzer(csv_file=str(csv_file))
        
        # Get efficiency metrics
        efficiency = analyzer.analyze_search_efficiency(mcts.stat_dict)
        
        print(f"   ✓ Best formation energy: {efficiency['best_formation_energy']:.4f} eV/atom")
        print(f"   ✓ Best compound: {results['best_node_formula']}")
        print(f"   ✓ Compounds within 100 meV of hull: {efficiency['compounds_near_hull_100meV']}")
        print(f"   ✓ Search efficiency: {efficiency['search_diversity']:.4f}")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return 1
    
    # Step 6: Save results
    print(f"\n6. Saving results...")
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create visualizations
        visualizer = TreeVisualizer()
        
        # Radial tree visualization
        visualizer.plot_radial_tree_visualization(
            mcts,
            output_dir=str(output_dir),
            csv_file=str(csv_file),
            show_labels=not args.no_labels
        )
        
        # Energy distribution plot
        visualizer.plot_energy_distribution(
            mcts.stat_dict,
            top_n=15,
            save_path=output_dir / "energy_distribution.png",
            csv_file=str(csv_file)
        )
        
        # Iteration progress plot
        visualizer.plot_iteration_progress(
            mcts,
            save_path=output_dir / "iteration_progress.png",
            csv_file=str(csv_file)
        )
        
        # Energy above hull distribution plot
        visualizer.plot_energy_above_hull_distribution(
            mcts.stat_dict,
            top_n=15,
            save_path=output_dir / "energy_above_hull_distribution.png",
            csv_file=str(csv_file)
        )
        
        # Energy above hull iteration progress plot
        visualizer.plot_energy_above_hull_progress(
            mcts,
            save_path=output_dir / "energy_above_hull_progress.png",
            csv_file=str(csv_file)
        )
        
        # Formation energy by elements plot
        visualizer.plot_formation_energy_by_elements(
            mcts.stat_dict,
            csv_file=str(csv_file),
            save_path=output_dir / "formation_energy_by_elements.png"
        )
        
        # Energy above hull by elements plot
        visualizer.plot_energy_above_hull_by_elements(
            mcts.stat_dict,
            csv_file=str(csv_file),
            save_path=output_dir / "energy_above_hull_by_elements.png"
        )
        
        # Generate reports
        analyzer.create_summary_report(
            mcts,
            save_path=output_dir / "mcts_report.txt"
        )
        
        # Export data
        analyzer.export_results(
            mcts.stat_dict,
            output_dir / "all_compounds.csv"
        )
        
        # Get and display top compounds
        top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)
        
        print(f"   ✓ Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return 1
    
    # Step 7: Display summary
    print(f"\n" + "=" * 80)
    print("🎯 MCTS OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"🥇 Best compound: {results['best_node_formula']}")
    print(f"⚡ Formation energy: {results['best_node_e_form']:.4f} eV/atom")
    print(f"🔍 Total compounds explored: {len(results['stat_dict'])}")
    print(f"📁 Results directory: {output_dir.absolute()}")
    
    print(f"\n🏆 TOP 10 COMPOUNDS DISCOVERED:")
    print("-" * 60)
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        stability = "Stable" if row['e_above_hull'] < 0.1 else "Metastable"
        print(f"{i:2d}. {row['formula']:15s} | "
              f"E_form: {row['formation_energy']:8.4f} eV/atom | "
              f"{stability}")
    
    print(f"\n📊 FILES CREATED:")
    print(f"   • radial_tree_visualization.png - Tree structure with formation energies")
    print(f"   • energy_distribution.png - Formation energy distribution")
    print(f"   • iteration_progress.png - Search progress over iterations")
    print(f"   • energy_above_hull_distribution.png - Energy above hull distribution")
    print(f"   • energy_above_hull_progress.png - Energy above hull search progress")
    print(f"   • formation_energy_by_elements.png - Formation energy by transition metal/Group IV")
    print(f"   • energy_above_hull_by_elements.png - Energy above hull by transition metal/Group IV")
    print(f"   • mcts_report.txt - Detailed text report")
    print(f"   • all_compounds.csv - All discovered compounds data")
    
    print(f"\n💡 To run again:")
    print(f"   python run_mcts.py --iterations {args.iterations}")
    print(f"   python run_mcts.py --iterations 1000  # Longer search")
    print(f"   python run_mcts.py --structure my_file.cif  # Different starting material")
    print(f"   python run_mcts.py --f-block-mode lanthanides_u  # Lanthanides + U substitution")
    print(f"   python run_mcts.py --alpha 2.0 --beta 3.0 --gamma 1.5  # Custom reward weights (include DOSCAR)")
    print(f"   python run_mcts.py -c 0.2  # Higher exploration constant")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
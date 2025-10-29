#!/usr/bin/env python3
"""
Starting Material Sensitivity Study for MCTS Crystal Structure Optimization

This script analyzes how different starting materials affect MCTS performance,
examining the top 10 most stable materials discovered and best rewards achieved.

Usage:
    python starting_material_sensitivity_study.py [--iterations N] [--repeats N] [--output DIR]

Examples:
    python starting_material_sensitivity_study.py                    # Default: 100 iterations, 3 repeats
    python starting_material_sensitivity_study.py --iterations 200   # 200 iterations per run
    python starting_material_sensitivity_study.py --repeats 5        # 5 repeats per starting material

NOTE: If using rollout methods that require energy above hull (eh, both, weighted),
      you must provide your Materials Project API key below.
"""

# CONFIGURATION: Add your Materials Project API key here if needed
MP_API_KEY = None  # Get your key from: https://materialsproject.org/api

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import (
    MCTSTreeNode,
    MCTS, 
    MaceEnergyCalculator,
    ResultsAnalyzer
)
from ase.io import read


def run_mcts_with_starting_material(
    starting_material_path: str,
    energy_calculator,
    n_iterations: int = 100,
    f_block_mode: str = 'u_only',
    exploration_constant: float = 0.1,
    n_rollout: int = 5,
    rollout_method: str = 'fe'
) -> Dict:
    """
    Run MCTS with a specific starting material.
    
    Args:
        starting_material_path: Path to the starting crystal structure
        energy_calculator: Energy calculator instance
        n_iterations: Number of MCTS iterations
        f_block_mode: F-block substitution mode
        exploration_constant: UCB exploration constant
        n_rollout: Number of rollout simulations
        rollout_method: Rollout evaluation method
        
    Returns:
        Dictionary containing results and iteration history
    """
    print(f"  Running with starting material: {Path(starting_material_path).name}")
    
    # Load starting structure
    atoms = read(starting_material_path)
    formula = atoms.get_chemical_formula()
    print(f"    Formula: {formula}")
    
    # Initialize MCTS
    root_node = MCTSTreeNode(
        atoms, 
        f_block_mode=f_block_mode,
        exploration_constant=exploration_constant
    )
    mcts = MCTS(root_node)
    
    # Track iteration-by-iteration progress
    iteration_history = {
        'iterations': [],
        'best_rewards': [],
        'compounds_explored': [],
        'best_formulas': []
    }

    # Track best reward to ensure monotonic increase
    best_reward_so_far = -float('inf')

    # Run MCTS with detailed tracking
    for i in range(n_iterations):
        if mcts.terminated:
            break
            
        # Run single iteration
        select_chain = mcts.select_node(mode='epsilon')
        
        if mcts.terminated:
            break
            
        mcts.stat_node_visited()
        
        reward, renew_t_to_terminate = mcts.expansion_simulation(
            rollout_depth=1,
            n_rollout=n_rollout,
            energy_calculator=energy_calculator,
            rollout_method=rollout_method
        )
        
        mcts.back_propagation(reward, select_chain, renew_t_to_terminate)
        mcts.stat_node_visited()
        
        # Record progress with monotonic best reward tracking
        best_reward_so_far = max(best_reward_so_far, mcts.max_reward)
        iteration_history['iterations'].append(i + 1)
        iteration_history['best_rewards'].append(best_reward_so_far)
        iteration_history['compounds_explored'].append(len(mcts.stat_dict))
        best_formula = mcts.best_node.get_chemical_formula() if mcts.best_node else formula
        iteration_history['best_formulas'].append(best_formula)
    
    # Get top 10 compounds by formation energy
    stat_df = mcts.get_statistics_dataframe()
    if not stat_df.empty:
        # Convert numeric columns to proper dtypes
        stat_df['e_form'] = pd.to_numeric(stat_df['e_form'], errors='coerce')
        stat_df['e_above_hull'] = pd.to_numeric(stat_df['e_above_hull'], errors='coerce')

        # Remove rows with NaN values in the energy columns
        stat_df = stat_df.dropna(subset=['e_form', 'e_above_hull'])

        if not stat_df.empty:
            top_10_fe = stat_df.nsmallest(10, 'e_form')
            top_10_eh = stat_df.nsmallest(10, 'e_above_hull')
        else:
            top_10_fe = pd.DataFrame()
            top_10_eh = pd.DataFrame()
    else:
        top_10_fe = pd.DataFrame()
        top_10_eh = pd.DataFrame()
    
    # Final results
    results = {
        'starting_material': starting_material_path,
        'starting_formula': formula,
        'iterations_completed': i + 1 if not mcts.terminated else i,
        'best_reward': mcts.max_reward,
        'best_node_formula': mcts.best_node.get_chemical_formula() if mcts.best_node else None,
        'best_node_e_form': mcts.best_node.e_form if mcts.best_node else None,
        'best_node_e_above_hull': mcts.best_node.e_above_hull if mcts.best_node else None,
        'compounds_explored': len(mcts.stat_dict),
        'terminated': mcts.terminated,
        'iteration_history': iteration_history,
        'top_10_formation_energy': top_10_fe.to_dict('index') if not top_10_fe.empty else {},
        'top_10_energy_above_hull': top_10_eh.to_dict('index') if not top_10_eh.empty else {},
        'stat_dict': mcts.stat_dict.copy()
    }
    
    return results


def analyze_starting_material_results(results_list: List[Dict]) -> Dict:
    """
    Analyze results from multiple starting material runs.
    
    Args:
        results_list: List of results from different starting materials
        
    Returns:
        Dictionary containing summary statistics
    """
    # Group results by starting material
    grouped_results = {}
    for result in results_list:
        material = Path(result['starting_material']).stem
        if material not in grouped_results:
            grouped_results[material] = []
        grouped_results[material].append(result)
    
    summary = {}
    
    for material, material_results in grouped_results.items():
        # Extract metrics across replicates
        best_rewards = [r['best_reward'] for r in material_results]
        best_e_forms = [r['best_node_e_form'] for r in material_results if r['best_node_e_form'] is not None]
        best_e_hulls = [r['best_node_e_above_hull'] for r in material_results if r['best_node_e_above_hull'] is not None]
        compounds_explored = [r['compounds_explored'] for r in material_results]
        
        # Get most common top compounds across replicates
        all_top_compounds_fe = {}
        all_top_compounds_eh = {}
        
        for result in material_results:
            for formula, data in result['top_10_formation_energy'].items():
                if formula not in all_top_compounds_fe:
                    all_top_compounds_fe[formula] = []
                all_top_compounds_fe[formula].append(data['e_form'])
                
            for formula, data in result['top_10_energy_above_hull'].items():
                if formula not in all_top_compounds_eh:
                    all_top_compounds_eh[formula] = []
                all_top_compounds_eh[formula].append(data['e_above_hull'])
        
        # Calculate statistics
        summary[material] = {
            'starting_formula': material_results[0]['starting_formula'],
            'n_replicates': len(material_results),
            'best_reward_mean': np.mean(best_rewards),
            'best_reward_std': np.std(best_rewards),
            'best_reward_min': np.min(best_rewards),
            'best_reward_max': np.max(best_rewards),
            'best_e_form_mean': np.mean(best_e_forms) if best_e_forms else None,
            'best_e_form_std': np.std(best_e_forms) if best_e_forms else None,
            'best_e_hull_mean': np.mean(best_e_hulls) if best_e_hulls else None,
            'best_e_hull_std': np.std(best_e_hulls) if best_e_hulls else None,
            'compounds_explored_mean': np.mean(compounds_explored),
            'compounds_explored_std': np.std(compounds_explored),
            'unique_top_compounds_fe': len(all_top_compounds_fe),
            'unique_top_compounds_eh': len(all_top_compounds_eh),
            'consistent_top_compounds_fe': [
                formula for formula, energies in all_top_compounds_fe.items() 
                if len(energies) >= len(material_results) * 0.5
            ],
            'consistent_top_compounds_eh': [
                formula for formula, energies in all_top_compounds_eh.items() 
                if len(energies) >= len(material_results) * 0.5
            ]
        }
    
    return summary


def plot_starting_material_comparison(results_list: List[Dict], output_dir: Path):
    """
    Create comprehensive plots comparing starting materials.
    
    Args:
        results_list: List of results from different starting materials
        output_dir: Directory to save plots
    """
    # Group results by starting material
    grouped_results = {}
    for result in results_list:
        material = Path(result['starting_material']).stem
        if material not in grouped_results:
            grouped_results[material] = []
        grouped_results[material].append(result)
    
    # Set up the plotting style
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, len(grouped_results)))

    # Track highest mean value across all curves for ylim setting
    highest_mean_value = -float('inf')

    # Figure 1: Best reward convergence over iterations (similar to sensitivity studies)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (material, material_results) in enumerate(grouped_results.items()):
        # Collect iteration histories from all replicates
        all_iterations = []
        all_rewards = []
        
        for result in material_results:
            hist = result['iteration_history']
            all_iterations.extend(hist['iterations'])
            all_rewards.extend(hist['best_rewards'])
            
            # # Plot individual replicate as thin line
            # ax.plot(hist['iterations'], hist['best_rewards'], 
            #        color=colors[i], alpha=0.3, linewidth=1)
        
        # Calculate mean trajectory with standard deviation
        if material_results:
            max_iterations = max(len(r['iteration_history']['iterations']) for r in material_results)
            mean_trajectory = []
            std_trajectory = []

            for iter_idx in range(max_iterations):
                rewards_at_iter = []
                for result in material_results:
                    if iter_idx < len(result['iteration_history']['best_rewards']):
                        rewards_at_iter.append(result['iteration_history']['best_rewards'][iter_idx])

                if rewards_at_iter:
                    mean_trajectory.append(np.mean(rewards_at_iter))
                    std_trajectory.append(np.std(rewards_at_iter))
            
            iterations_range = range(1, len(mean_trajectory) + 1)
            
            # Plot mean with error bars
            ax.plot(iterations_range, mean_trajectory, 
                   color=colors[i], linewidth=3, 
                   label=f'{material} (n={len(material_results)})')
            
            # Add standard deviation as shaded area
            if len(std_trajectory) > 0:
                mean_array = np.array(mean_trajectory)
                std_array = np.array(std_trajectory)
                ax.fill_between(iterations_range,
                               mean_array - std_array,
                               mean_array + std_array,
                               color=colors[i], alpha=0.2)

        # Update highest mean value for ylim setting
        if material_results and mean_trajectory:
            highest_mean_value = max(highest_mean_value, max(mean_trajectory))
    
    ax.set_xlabel('MCTS Iteration')
    ax.set_ylabel('Best Reward')
    ax.set_title('Starting Material Sensitivity: Best Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis upper limit to highest mean value to prevent std bands from going above realistic values
    if highest_mean_value > -float('inf'):
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], highest_mean_value * 1.05)  # Add 5% padding above the highest mean

    plt.tight_layout()
    plt.savefig(output_dir / 'starting_material_reward_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Summary statistics comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    materials = list(grouped_results.keys())
    
    # Best reward statistics
    best_rewards_mean = []
    best_rewards_std = []
    best_e_forms_mean = []
    best_e_forms_std = []
    compounds_explored_mean = []
    compounds_explored_std = []
    
    for material in materials:
        material_results = grouped_results[material]
        
        best_rewards = [r['best_reward'] for r in material_results]
        best_e_forms = [r['best_node_e_form'] for r in material_results if r['best_node_e_form'] is not None]
        compounds_explored = [r['compounds_explored'] for r in material_results]
        
        best_rewards_mean.append(np.mean(best_rewards))
        best_rewards_std.append(np.std(best_rewards))
        best_e_forms_mean.append(np.mean(best_e_forms) if best_e_forms else 0)
        best_e_forms_std.append(np.std(best_e_forms) if best_e_forms else 0)
        compounds_explored_mean.append(np.mean(compounds_explored))
        compounds_explored_std.append(np.std(compounds_explored))
    
    # Plot 1: Best rewards
    x_pos = np.arange(len(materials))
    ax1.bar(x_pos, best_rewards_mean, yerr=best_rewards_std, 
           color=colors[:len(materials)], alpha=0.7, capsize=5)
    ax1.set_xlabel('Starting Material')
    ax1.set_ylabel('Best Reward')
    ax1.set_title('Best Reward by Starting Material')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(materials, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best formation energies
    ax2.bar(x_pos, best_e_forms_mean, yerr=best_e_forms_std, 
           color=colors[:len(materials)], alpha=0.7, capsize=5)
    ax2.set_xlabel('Starting Material')
    ax2.set_ylabel('Best Formation Energy (eV/atom)')
    ax2.set_title('Best Formation Energy by Starting Material')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(materials, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Compounds explored
    ax3.bar(x_pos, compounds_explored_mean, yerr=compounds_explored_std, 
           color=colors[:len(materials)], alpha=0.7, capsize=5)
    ax3.set_xlabel('Starting Material')
    ax3.set_ylabel('Compounds Explored')
    ax3.set_title('Search Diversity by Starting Material')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(materials, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of best rewards
    all_best_rewards = []
    material_labels = []
    for material in materials:
        material_results = grouped_results[material]
        best_rewards = [r['best_reward'] for r in material_results]
        all_best_rewards.extend(best_rewards)
        material_labels.extend([material] * len(best_rewards))
    
    reward_df = pd.DataFrame({
        'Material': material_labels,
        'Best_Reward': all_best_rewards
    })
    
    sns.boxplot(data=reward_df, x='Material', y='Best_Reward', ax=ax4)
    ax4.set_xlabel('Starting Material')
    ax4.set_ylabel('Best Reward')
    ax4.set_title('Best Reward Distribution by Starting Material')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'starting_material_summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for starting material sensitivity study."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run starting material sensitivity study')
    parser.add_argument('--iterations', '-n', type=int, default=100,
                       help='Number of MCTS iterations per run (default: 100)')
    parser.add_argument('--repeats', '-r', type=int, default=3,
                       help='Number of replicate runs per starting material (default: 3)')
    parser.add_argument('--output', '-o', type=str, default='starting_material_sensitivity_results',
                       help='Output directory name (default: starting_material_sensitivity_results)')
    parser.add_argument('--f-block-mode', type=str, default='u_only',
                       choices=['u_only', 'full_f_block'],
                       help='F-block substitution mode (default: u_only)')
    parser.add_argument('--exploration-constant', type=float, default=0.1,
                       help='Exploration constant for UCB calculation (default: 0.1)')
    parser.add_argument('--n-rollout', type=int, default=5,
                       help='Number of rollout simulations (default: 5)')
    parser.add_argument('--rollout-method', type=str, default='fe',
                       choices=['fe', 'eh', 'both'],
                       help='Rollout evaluation method (default: fe)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARTING MATERIAL SENSITIVITY STUDY")
    print("=" * 80)
    
    # Find starting materials
    data_dir = Path("../data")
    starting_materials = list(data_dir.glob("*.cif"))
    
    if not starting_materials:
        print("❌ Error: No CIF files found in data/ directory")
        return 1
    
    print(f"\n📁 Configuration:")
    print(f"   Iterations per run: {args.iterations}")
    print(f"   F-block mode: {args.f_block_mode}")
    print(f"   Exploration constant: {args.exploration_constant}")
    print(f"   N rollout: {args.n_rollout}")
    print(f"   Rollout method: {args.rollout_method}")
    print(f"   Starting materials: {[p.name for p in starting_materials]}")
    print(f"   Repeats per material: {args.repeats}")
    print(f"   Total runs: {len(starting_materials) * args.repeats}")
    
    # Step 1: Load energy calculator
    print(f"\n1. Setting up energy calculator...")
    csv_file = Path("../examples/mace_calculations.csv")
    
    if not csv_file.exists():
        print(f"❌ Error: MACE calculations file not found: {csv_file}")
        return 1
    
    try:
        energy_calc = MaceEnergyCalculator(csv_file=str(csv_file), mp_api_key=MP_API_KEY)
        df = pd.read_csv(csv_file)
        print(f"   ✓ Cached calculations: {len(df)} entries")
    except Exception as e:
        print(f"❌ Error setting up energy calculator: {e}")
        return 1
    
    # Step 2: Run sensitivity study
    print(f"\n2. Running starting material sensitivity study...")
    print(f"   This will run {len(starting_materials) * args.repeats} separate MCTS searches...")
    
    results_list = []
    
    for material_idx, starting_material in enumerate(starting_materials):
        material_name = starting_material.stem
        print(f"\n📊 Starting Material {material_idx + 1}/{len(starting_materials)}: {material_name}")
        
        for repeat in range(args.repeats):
            print(f"  🔄 Repeat {repeat + 1}/{args.repeats}")
            
            try:
                result = run_mcts_with_starting_material(
                    starting_material_path=str(starting_material),
                    energy_calculator=energy_calc,
                    n_iterations=args.iterations,
                    f_block_mode=args.f_block_mode,
                    exploration_constant=args.exploration_constant,
                    n_rollout=args.n_rollout,
                    rollout_method=args.rollout_method
                )
                
                result['repeat'] = repeat + 1
                results_list.append(result)
                
                print(f"     ✅ Best reward: {result['best_reward']:.4f}")
                print(f"     ✅ Best compound: {result['best_node_formula']}")
                print(f"     ✅ Compounds explored: {result['compounds_explored']}")
                
            except Exception as e:
                print(f"     ❌ Error in repeat {repeat + 1}: {e}")
                continue
    
    if not results_list:
        print("❌ No successful runs completed!")
        return 1
    
    # Step 3: Analyze results
    print(f"\n3. Analyzing results...")
    try:
        summary = analyze_starting_material_results(results_list)
        
        print(f"   ✓ Processed {len(results_list)} total runs")
        print(f"   ✓ Analyzed {len(summary)} starting materials")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return 1
    
    # Step 4: Save results and create visualizations
    print(f"\n4. Saving results...")
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Save detailed results
        with open(output_dir / "starting_material_detailed_results.json", 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = []
            for result in results_list:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_result[key] = value.item()
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)
            json.dump(serializable_results, f, indent=2)
        
        # Save summary
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(output_dir / "starting_material_summary.csv")
        
        # Create visualizations
        plot_starting_material_comparison(results_list, output_dir)
        
        print(f"   ✓ Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return 1
    
    # Step 5: Display summary
    print(f"\n" + "=" * 80)
    print("🎯 STARTING MATERIAL SENSITIVITY STUDY COMPLETED!")
    print("=" * 80)
    
    print(f"\n📊 SUMMARY BY STARTING MATERIAL:")
    print("-" * 80)
    
    for material, stats in summary.items():
        print(f"\n🧪 {material} ({stats['starting_formula']}):")
        print(f"   Best reward: {stats['best_reward_mean']:.4f} ± {stats['best_reward_std']:.4f}")
        if stats['best_e_form_mean'] is not None:
            print(f"   Best formation energy: {stats['best_e_form_mean']:.4f} ± {stats['best_e_form_std']:.4f} eV/atom")
        print(f"   Compounds explored: {stats['compounds_explored_mean']:.1f} ± {stats['compounds_explored_std']:.1f}")
        print(f"   Consistent top compounds (FE): {len(stats['consistent_top_compounds_fe'])}")
        print(f"   Consistent top compounds (EH): {len(stats['consistent_top_compounds_eh'])}")
    
    # Find best performing starting material
    best_material = max(summary.keys(), key=lambda k: summary[k]['best_reward_mean'])
    best_stats = summary[best_material]
    
    print(f"\n🏆 BEST PERFORMING STARTING MATERIAL:")
    print(f"   Material: {best_material}")
    print(f"   Formula: {best_stats['starting_formula']}")
    print(f"   Average best reward: {best_stats['best_reward_mean']:.4f}")
    print(f"   Standard deviation: {best_stats['best_reward_std']:.4f}")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   • starting_material_reward_convergence.png - Reward convergence comparison")
    print(f"   • starting_material_summary_statistics.png - Summary statistics comparison")
    print(f"   • starting_material_detailed_results.json - Detailed results data")
    print(f"   • starting_material_summary.csv - Summary statistics table")
    
    print(f"\n💡 To run again:")
    print(f"   python starting_material_sensitivity_study.py --iterations {args.iterations}")
    print(f"   python starting_material_sensitivity_study.py --repeats 5  # More replicates")
    print(f"   python starting_material_sensitivity_study.py --iterations 200 --repeats 5  # Longer, more robust")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
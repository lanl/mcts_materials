#!/usr/bin/env python3
"""
Exploration Parameter Sensitivity Study for MCTS Crystal Structure Optimization

This script performs a sensitivity analysis of the exploration constant in the UCB
calculation, running multiple MCTS searches with different exploration values and
comparing their performance.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
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


def run_mcts_with_exploration_constant(
    atoms, 
    energy_calculator, 
    exploration_constant: float,
    n_iterations: int = 100,
    f_block_mode: str = 'u_only'
) -> Dict:
    """
    Run MCTS with a specific exploration constant.
    
    Args:
        atoms: Initial crystal structure
        energy_calculator: Energy calculator instance
        exploration_constant: UCB exploration constant
        n_iterations: Number of MCTS iterations
        f_block_mode: F-block substitution mode
        
    Returns:
        Dictionary containing results and iteration history
    """
    print(f"  Running with exploration constant = {exploration_constant:.3f}")
    
    # Initialize MCTS
    root_node = MCTSTreeNode(
        atoms, 
        f_block_mode=f_block_mode,
        exploration_constant=exploration_constant
    )
    mcts = MCTS(root_node)
    
    # Track best reward over iterations
    iteration_history = []
    best_reward_so_far = -float('inf')
    
    # Run MCTS iteration by iteration to track progress
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
            n_rollout=5,
            energy_calculator=energy_calculator
        )
        
        mcts.back_propagation(reward, select_chain, renew_t_to_terminate)
        mcts.stat_node_visited()
        
        # Update best reward tracking - ensure monotonic increase
        best_reward_so_far = max(best_reward_so_far, mcts.max_reward)

        iteration_history.append({
            'iteration': i + 1,
            'best_reward': best_reward_so_far,
            'compounds_explored': len(mcts.stat_dict),
            'current_reward': reward
        })
    
    # Final results
    results = {
        'exploration_constant': exploration_constant,
        'iterations_completed': i + 1 if not mcts.terminated else i,
        'best_reward': mcts.max_reward,
        'best_compound': mcts.best_node.get_chemical_formula() if mcts.best_node else None,
        'best_e_form': mcts.best_node.e_form if mcts.best_node else None,
        'compounds_explored': len(mcts.stat_dict),
        'stat_dict': mcts.stat_dict.copy(),
        'iteration_history': iteration_history,
        'terminated': mcts.terminated
    }
    
    return results


def create_sensitivity_plot(
    results_list: List[Dict], 
    save_path: Path,
    title: str = "Exploration Constant Sensitivity Analysis"
):
    """
    Create a multi-curve plot showing best reward vs discovery order for different exploration constants
    with error bars showing variance across repeats.
    
    Args:
        results_list: List of results dictionaries from different exploration constants
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(14, 8))

    # Group results by exploration constant
    grouped_results = {}
    for result in results_list:
        exp_const = result['exploration_constant']
        if exp_const not in grouped_results:
            grouped_results[exp_const] = []
        grouped_results[exp_const].append(result)

    # Define colors for different curves
    exploration_constants = sorted(grouped_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(exploration_constants)))

    # Track highest mean value across all curves for ylim setting
    highest_mean_value = -float('inf')
    
    for i, exp_const in enumerate(exploration_constants):
        repeats = grouped_results[exp_const]
        
        if not repeats:
            continue
        
        # Find maximum iteration count across all repeats
        max_iterations = max(len(r['iteration_history']) for r in repeats if r['iteration_history'])
        
        if max_iterations == 0:
            continue
        
        # Calculate mean and std for each iteration
        iterations = list(range(1, max_iterations + 1))
        mean_rewards = []
        std_rewards = []

        for iter_num in iterations:
            iter_rewards = []
            for repeat in repeats:
                if repeat['iteration_history'] and len(repeat['iteration_history']) >= iter_num:
                    iter_rewards.append(repeat['iteration_history'][iter_num - 1]['best_reward'])

            if iter_rewards:
                mean_rewards.append(np.mean(iter_rewards))
                std_rewards.append(np.std(iter_rewards))
            else:
                mean_rewards.append(np.nan)
                std_rewards.append(np.nan)
        
        # Remove NaN values
        valid_indices = [i for i, (m, s) in enumerate(zip(mean_rewards, std_rewards))
                        if not (np.isnan(m) or np.isnan(s))]

        if not valid_indices:
            continue

        valid_iterations = [iterations[i] for i in valid_indices]
        valid_means = [mean_rewards[i] for i in valid_indices]
        valid_stds = [std_rewards[i] for i in valid_indices]
        
        # Plot mean line
        plt.plot(
            valid_iterations, 
            valid_means,
            color=colors[i],
            linewidth=2.5,
            label=f'C = {exp_const:.3f}',
            marker='o' if len(valid_iterations) < 50 else None,
            markersize=4 if len(valid_iterations) < 50 else 0,
            alpha=0.9,
            zorder=3
        )
        
        # Add confidence bands for standard deviation
        if len(valid_iterations) > 1:
            upper_bound = [m + s for m, s in zip(valid_means, valid_stds)]
            lower_bound = [m - s for m, s in zip(valid_means, valid_stds)]

            # Plot upper and lower bound curves
            plt.plot(valid_iterations, upper_bound,
                    color=colors[i], linewidth=1, alpha=0.6, linestyle='--', zorder=1)
            plt.plot(valid_iterations, lower_bound,
                    color=colors[i], linewidth=1, alpha=0.6, linestyle='--', zorder=1)

            # Fill between with faded color
            plt.fill_between(valid_iterations, lower_bound, upper_bound,
                           color=colors[i], alpha=0.15, zorder=2)

        # Update highest mean value for ylim setting
        if valid_means:
            highest_mean_value = max(highest_mean_value, max(valid_means))
    
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Best Reward Found', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title='Exploration Constant', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Set y-axis upper limit to highest mean value to prevent std bands from going above realistic values
    if highest_mean_value > -float('inf'):
        current_ylim = plt.ylim()
        plt.ylim(current_ylim[0], highest_mean_value * 1.05)  # Add 5% padding above the highest mean

    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Sensitivity plot saved to: {save_path}")
    
    return plt.gcf()




def main():
    """Main sensitivity study function."""
    parser = argparse.ArgumentParser(
        description='Run exploration constant sensitivity study for MCTS crystal optimization'
    )
    parser.add_argument('--iterations', '-n', type=int, default=50,
                       help='Number of MCTS iterations per run (default: 50)')
    parser.add_argument('--structure', '-s', type=str,
                       default='../examples/mat_Pb6U1W6_sg191.cif',
                       help='Path to starting crystal structure CIF file')
    parser.add_argument('--output', '-o', type=str, default='exploration_sensitivity_results',
                       help='Output directory name (default: exploration_sensitivity_results)')
    parser.add_argument('--f-block-mode', type=str, default='u_only',
                       choices=['u_only', 'full_f_block'],
                       help='F-block substitution mode (default: u_only)')
    parser.add_argument('--exploration-range', type=str, default='0.01,0.05,0.1,0.2,0.5,1.0',
                       help='Comma-separated exploration constants to test (default: 0.01,0.05,0.1,0.2,0.5,1.0)')
    parser.add_argument('--repeats', '-r', type=int, default=3,
                       help='Number of repeated runs per exploration constant (default: 3)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPLORATION CONSTANT SENSITIVITY STUDY")
    print("=" * 80)
    
    # Parse exploration constants
    try:
        exploration_constants = [float(x.strip()) for x in args.exploration_range.split(',')]
        exploration_constants = sorted(exploration_constants)
    except ValueError:
        print("❌ Error: Invalid exploration range format. Use comma-separated numbers.")
        return 1
    
    print(f"\n📊 Study Parameters:")
    print(f"   Structure: {args.structure}")
    print(f"   Iterations per run: {args.iterations}")
    print(f"   F-block mode: {args.f_block_mode}")
    print(f"   Exploration constants: {exploration_constants}")
    print(f"   Repeats per constant: {args.repeats}")
    print(f"   Total runs: {len(exploration_constants) * args.repeats}")
    
    # Step 1: Load crystal structure
    print(f"\n1. Loading crystal structure...")
    structure_path = Path(args.structure)
    
    if not structure_path.exists():
        print(f"❌ Error: Structure file not found: {structure_path}")
        return 1
        
    try:
        atoms = read(str(structure_path))
        print(f"   ✓ Loaded: {atoms.get_chemical_formula()}")
    except Exception as e:
        print(f"❌ Error loading structure: {e}")
        return 1
    
    # Step 2: Set up energy calculator
    print(f"\n2. Setting up energy calculator...")
    csv_file = Path("../examples/mace_calculations.csv")
    
    if not csv_file.exists():
        print(f"❌ Error: MACE calculations file not found: {csv_file}")
        return 1
    
    try:
        energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
        df = pd.read_csv(csv_file)
        print(f"   ✓ Cached calculations: {len(df)} entries")
    except Exception as e:
        print(f"❌ Error setting up energy calculator: {e}")
        return 1
    
    # Step 3: Run sensitivity study
    print(f"\n3. Running sensitivity study...")
    print(f"   This will run {len(exploration_constants) * args.repeats} separate MCTS searches...")
    
    results_list = []
    
    for i, exploration_const in enumerate(exploration_constants, 1):
        print(f"\n   [{i}/{len(exploration_constants)}] Testing exploration constant = {exploration_const}")
        
        repeat_results = []
        
        for repeat in range(args.repeats):
            print(f"     Repeat {repeat + 1}/{args.repeats}...")
            
            try:
                results = run_mcts_with_exploration_constant(
                    atoms=atoms,
                    energy_calculator=energy_calc,
                    exploration_constant=exploration_const,
                    n_iterations=args.iterations,
                    f_block_mode=args.f_block_mode
                )
                repeat_results.append(results)
                
                print(f"       ✓ Completed: {results['iterations_completed']} iterations")
                print(f"       ✓ Best reward: {results['best_reward']:.4f}")
                print(f"       ✓ Compounds explored: {results['compounds_explored']}")
                
            except Exception as e:
                print(f"       ❌ Error in repeat {repeat + 1}: {e}")
                continue
        
        if repeat_results:
            # Store all repeats for this exploration constant
            for result in repeat_results:
                result['repeat_id'] = len([r for r in repeat_results[:repeat_results.index(result) + 1]])
            results_list.extend(repeat_results)
            
            # Print summary statistics
            rewards = [r['best_reward'] for r in repeat_results]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print(f"     📊 Summary: {mean_reward:.4f} ± {std_reward:.4f} (mean ± std)")
    
    if not results_list:
        print("❌ No successful runs completed!")
        return 1
    
    # Step 4: Create output directory and save results
    print(f"\n4. Analyzing results and creating plots...")
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Save raw results as JSON
    results_file = output_dir / "sensitivity_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = []
        for r in results_list:
            json_r = r.copy()
            # Remove stat_dict as it's too large for JSON
            json_r.pop('stat_dict', None)
            json_results.append(json_r)
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"   ✓ Raw results saved to: {results_file}")
    
    # Create sensitivity plot
    sensitivity_plot_path = output_dir / "exploration_sensitivity_curves.png"
    create_sensitivity_plot(
        results_list, 
        sensitivity_plot_path,
        title=f"Exploration Constant Sensitivity Analysis\\n{atoms.get_chemical_formula()} - {args.iterations} iterations"
    )
    
    # Step 6: Display results summary
    print(f"\n" + "=" * 80)
    print("🎯 SENSITIVITY STUDY COMPLETED!")
    print("=" * 80)
    
    print(f"\\n📊 EXPLORATION CONSTANT PERFORMANCE SUMMARY:")
    print("-" * 80)
    print(f"{'Constant':>8} | {'Mean Reward':>12} | {'Std Reward':>11} | {'Mean Compounds':>14} | {'Efficiency':>10}")
    print("-" * 80)
    
    # Group results by exploration constant and calculate statistics
    grouped_results = {}
    for result in results_list:
        exp_const = result['exploration_constant']
        if exp_const not in grouped_results:
            grouped_results[exp_const] = []
        grouped_results[exp_const].append(result)
    
    best_mean_reward = -float('inf')
    best_exp_const = None
    best_results_group = None
    
    for exp_const in sorted(grouped_results.keys()):
        repeats = grouped_results[exp_const]
        rewards = [r['best_reward'] for r in repeats]
        compounds = [r['compounds_explored'] for r in repeats]
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_compounds = np.mean(compounds)
        efficiency = mean_reward / mean_compounds if mean_compounds > 0 else 0
        
        print(f"C = {exp_const:5.3f} | {mean_reward:8.4f} ± {std_reward:4.3f} | {mean_compounds:8.1f} | {efficiency:8.4f}")
        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_exp_const = exp_const
            best_results_group = repeats
    
    print(f"\\n🏆 BEST EXPLORATION CONSTANT:")
    print(f"   Value: {best_exp_const:.3f}")
    print(f"   Mean Reward: {best_mean_reward:.4f}")
    if best_results_group:
        best_single_run = max(best_results_group, key=lambda x: x['best_reward'])
        print(f"   Best Single Run: {best_single_run['best_reward']:.4f}")
        print(f"   Best Compound: {best_single_run['best_compound']}")
        print(f"   Formation Energy: {best_single_run['best_e_form']:.4f} eV/atom")
    
    print(f"\\n📁 Results saved to: {output_dir.absolute()}")
    print(f"📈 Key files:")
    print(f"   • exploration_sensitivity_curves.png - Multi-curve progress plot")
    print(f"   • sensitivity_results.json - Raw numerical results")
    
    print(f"\\n💡 Recommended usage:")
    print(f"   python run_mcts.py --exploration-constant {best_exp_const:.3f}")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
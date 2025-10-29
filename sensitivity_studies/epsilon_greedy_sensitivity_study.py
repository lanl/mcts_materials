#!/usr/bin/env python3
"""
Epsilon-Greedy Sensitivity Study for MCTS Crystal Structure Optimization

This script performs a sensitivity analysis of the epsilon parameter in the epsilon-greedy
selection strategy, running multiple MCTS searches with different exploration/exploitation
ratios and comparing their performance.
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
import random

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import (
    MCTSTreeNode,
    MCTS, 
    MaceEnergyCalculator,
    ResultsAnalyzer
)
from ase.io import read


def run_mcts_with_epsilon_greedy(
    atoms, 
    energy_calculator, 
    epsilon: float,
    n_iterations: int = 100,
    f_block_mode: str = 'u_only',
    exploration_constant: float = 0.1
) -> Dict:
    """
    Run MCTS with a specific epsilon value for epsilon-greedy selection.
    
    Args:
        atoms: Initial crystal structure
        energy_calculator: Energy calculator instance
        epsilon: Exploration probability (0.0 = pure exploitation, 1.0 = pure exploration)
        n_iterations: Number of MCTS iterations
        f_block_mode: F-block substitution mode
        exploration_constant: UCB exploration constant
        
    Returns:
        Dictionary containing results and iteration history
    """
    print(f"  Running with epsilon = {epsilon:.1%} exploration, {(1-epsilon):.1%} exploitation")
    
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
            
        # Custom epsilon-greedy node selection
        select_chain = _custom_epsilon_greedy_select(mcts, epsilon)
        
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
        
        # Update best reward tracking
        if mcts.max_reward > best_reward_so_far:
            best_reward_so_far = mcts.max_reward
            
        iteration_history.append({
            'iteration': i + 1,
            'best_reward': best_reward_so_far,
            'compounds_explored': len(mcts.stat_dict),
            'current_reward': reward
        })
    
    # Final results
    results = {
        'epsilon': epsilon,
        'exploration_pct': epsilon * 100,
        'exploitation_pct': (1 - epsilon) * 100,
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


def _custom_epsilon_greedy_select(mcts: MCTS, epsilon: float) -> List[MCTSTreeNode]:
    """
    Custom epsilon-greedy node selection with configurable epsilon.
    
    Args:
        mcts: MCTS instance
        epsilon: Exploration probability
        
    Returns:
        List of selected nodes for back-propagation
    """
    select_chain = [mcts.root]
    current = mcts.root
    
    while not current.expandable:
        ucb_values = []
        
        for child_node in current.children:
            if child_node.terminated:
                ucb_values.append(-1e4)
                if child_node.get_chemical_formula() in mcts.stat_dict:
                    mcts.stat_dict[child_node.get_chemical_formula()][2] = True
            else:
                ucb_values.append(child_node.get_ucb())
        
        # Check if all children are terminated
        if set(ucb_values) == {-1e4}:
            mcts.terminated = True
            break
            
        # Epsilon-greedy selection with configurable epsilon
        if random.random() < epsilon:
            # EXPLORE: Use probability-weighted selection
            current = current.children[mcts._probability_selector(ucb_values)]
        else:
            # EXPLOIT: Choose node with highest UCB value
            current = current.children[np.argmax(ucb_values)]
            
        select_chain.append(current)
        
    mcts.current_node = current
    return select_chain


def create_sensitivity_plot(
    results_list: List[Dict], 
    save_path: Path,
    title: str = "Epsilon-Greedy Sensitivity Analysis"
):
    """
    Create a multi-curve plot showing best reward vs discovery order for different epsilon values
    with error bars showing variance across repeats.
    
    Args:
        results_list: List of results dictionaries from different epsilon values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    # Group results by epsilon value
    grouped_results = {}
    for result in results_list:
        epsilon = result['epsilon']
        if epsilon not in grouped_results:
            grouped_results[epsilon] = []
        grouped_results[epsilon].append(result)
    
    # Define colors for different curves
    epsilon_values = sorted(grouped_results.keys())
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(epsilon_values)))  # Red (exploit) to Blue (explore)
    
    for i, epsilon in enumerate(epsilon_values):
        repeats = grouped_results[epsilon]
        
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
            label=f'ε = {epsilon:.1f} ({epsilon:.0%} explore)',
            marker='o' if len(valid_iterations) < 50 else None,
            markersize=4 if len(valid_iterations) < 50 else 0,
            alpha=0.9,
            zorder=3
        )
        
        # Add smooth confidence bands for standard deviation
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
    
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Best Reward Found', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title='Epsilon Value (Exploration %)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Sensitivity plot saved to: {save_path}")
    
    return plt.gcf()


def main():
    """Main sensitivity study function."""
    parser = argparse.ArgumentParser(
        description='Run epsilon-greedy sensitivity study for MCTS crystal optimization'
    )
    parser.add_argument('--iterations', '-n', type=int, default=50,
                       help='Number of MCTS iterations per run (default: 50)')
    parser.add_argument('--structure', '-s', type=str,
                       default='../examples/mat_Pb6U1W6_sg191.cif',
                       help='Path to starting crystal structure CIF file')
    parser.add_argument('--output', '-o', type=str, default='epsilon_greedy_sensitivity_results',
                       help='Output directory name (default: epsilon_greedy_sensitivity_results)')
    parser.add_argument('--f-block-mode', type=str, default='u_only',
                       choices=['u_only', 'full_f_block'],
                       help='F-block substitution mode (default: u_only)')
    parser.add_argument('--exploration-constant', type=float, default=0.1,
                       help='UCB exploration constant (default: 0.1)')
    parser.add_argument('--epsilon-range', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                       help='Comma-separated epsilon values to test (default: 0.1 to 0.9 in 0.1 steps)')
    parser.add_argument('--repeats', '-r', type=int, default=3,
                       help='Number of repeated runs per epsilon value (default: 3)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EPSILON-GREEDY SENSITIVITY STUDY")
    print("=" * 80)
    
    # Parse epsilon values
    try:
        epsilon_values = [float(x.strip()) for x in args.epsilon_range.split(',')]
        epsilon_values = sorted([e for e in epsilon_values if 0.0 <= e <= 1.0])
    except ValueError:
        print("❌ Error: Invalid epsilon range format. Use comma-separated values between 0.0 and 1.0.")
        return 1
    
    print(f"\n📊 Study Parameters:")
    print(f"   Structure: {args.structure}")
    print(f"   Iterations per run: {args.iterations}")
    print(f"   F-block mode: {args.f_block_mode}")
    print(f"   Exploration constant: {args.exploration_constant}")
    print(f"   Epsilon values: {epsilon_values}")
    print(f"   Repeats per epsilon: {args.repeats}")
    print(f"   Total runs: {len(epsilon_values) * args.repeats}")
    
    print(f"\n🎯 Exploration vs Exploitation Breakdown:")
    for eps in epsilon_values:
        print(f"   ε = {eps:.1f}: {eps:.0%} exploration, {(1-eps):.0%} exploitation")
    
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
    print(f"   This will run {len(epsilon_values) * args.repeats} separate MCTS searches...")
    
    results_list = []
    
    for i, epsilon in enumerate(epsilon_values, 1):
        print(f"\n   [{i}/{len(epsilon_values)}] Testing epsilon = {epsilon:.1f} ({epsilon:.0%} exploration)")
        
        repeat_results = []
        
        for repeat in range(args.repeats):
            print(f"     Repeat {repeat + 1}/{args.repeats}...")
            
            try:
                results = run_mcts_with_epsilon_greedy(
                    atoms=atoms,
                    energy_calculator=energy_calc,
                    epsilon=epsilon,
                    n_iterations=args.iterations,
                    f_block_mode=args.f_block_mode,
                    exploration_constant=args.exploration_constant
                )
                repeat_results.append(results)
                
                print(f"       ✓ Completed: {results['iterations_completed']} iterations")
                print(f"       ✓ Best reward: {results['best_reward']:.4f}")
                print(f"       ✓ Compounds explored: {results['compounds_explored']}")
                
            except Exception as e:
                print(f"       ❌ Error in repeat {repeat + 1}: {e}")
                continue
        
        if repeat_results:
            # Store all repeats for this epsilon value
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
    sensitivity_plot_path = output_dir / "epsilon_greedy_sensitivity_curves.png"
    create_sensitivity_plot(
        results_list, 
        sensitivity_plot_path,
        title=f"Epsilon-Greedy Sensitivity Analysis\\n{atoms.get_chemical_formula()} - {args.iterations} iterations"
    )
    
    # Step 5: Display results summary
    print(f"\n" + "=" * 80)
    print("🎯 SENSITIVITY STUDY COMPLETED!")
    print("=" * 80)
    
    print(f"\\n📊 EPSILON-GREEDY PERFORMANCE SUMMARY:")
    print("-" * 95)
    print(f"{'Epsilon':>7} | {'Explore%':>8} | {'Exploit%':>8} | {'Mean Reward':>12} | {'Std Reward':>11} | {'Efficiency':>10}")
    print("-" * 95)
    
    # Group results by epsilon value and calculate statistics
    grouped_results = {}
    for result in results_list:
        epsilon = result['epsilon']
        if epsilon not in grouped_results:
            grouped_results[epsilon] = []
        grouped_results[epsilon].append(result)
    
    best_mean_reward = -float('inf')
    best_epsilon = None
    best_results_group = None
    
    for epsilon in sorted(grouped_results.keys()):
        repeats = grouped_results[epsilon]
        rewards = [r['best_reward'] for r in repeats]
        compounds = [r['compounds_explored'] for r in repeats]
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_compounds = np.mean(compounds)
        efficiency = mean_reward / mean_compounds if mean_compounds > 0 else 0
        
        print(f"{epsilon:7.1f} | {epsilon*100:6.0f}% | {(1-epsilon)*100:6.0f}% | {mean_reward:8.4f} ± {std_reward:4.3f} | {efficiency:8.4f}")
        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_epsilon = epsilon
            best_results_group = repeats
    
    print(f"\\n🏆 OPTIMAL EPSILON-GREEDY BALANCE:")
    print(f"   Epsilon: {best_epsilon:.1f}")
    print(f"   Strategy: {best_epsilon:.0%} exploration, {(1-best_epsilon):.0%} exploitation")
    print(f"   Mean Reward: {best_mean_reward:.4f}")
    if best_results_group:
        best_single_run = max(best_results_group, key=lambda x: x['best_reward'])
        print(f"   Best Single Run: {best_single_run['best_reward']:.4f}")
        print(f"   Best Compound: {best_single_run['best_compound']}")
        print(f"   Formation Energy: {best_single_run['best_e_form']:.4f} eV/atom")
    
    print(f"\\n📁 Results saved to: {output_dir.absolute()}")
    print(f"📈 Key files:")
    print(f"   • epsilon_greedy_sensitivity_curves.png - Multi-curve progress plot")
    print(f"   • sensitivity_results.json - Raw numerical results")
    
    print(f"\\n💡 Recommended usage:")
    print(f"   Modify MCTS epsilon-greedy to use {best_epsilon:.0%} exploration")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
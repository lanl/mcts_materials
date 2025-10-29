#!/usr/bin/env python3
"""
Rollout Method Comparison Study

Runs MCTS with different rollout methods (fe, eh, both) for multiple replicates
and generates comparison visualizations.

Usage:
    python rollout_method_comparison.py

NOTE: If using rollout methods that require energy above hull (eh, both, weighted),
      you must provide your Materials Project API key below.
"""

# CONFIGURATION: Add your Materials Project API key here if needed
MP_API_KEY = None  # Get your key from: https://materialsproject.org/api

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil

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


def run_single_mcts_replicate(rollout_method, replicate_id, n_iterations=1000):
    """
    Run a single MCTS replicate with the specified rollout method.

    Args:
        rollout_method: 'fe', 'eh', or 'both'
        replicate_id: Replicate number (1-10)
        n_iterations: Number of MCTS iterations

    Returns:
        dict: Results dictionary with statistics
    """
    print(f"\n{'='*80}")
    print(f"Running replicate {replicate_id} with rollout_method='{rollout_method}'")
    print(f"{'='*80}")

    # Load starting structure
    structure_path = Path(__file__).parent.parent / "examples" / "mat_Pb6U1W6_sg191.cif"
    atoms = read(str(structure_path))

    # Set up energy calculator
    csv_file = Path(__file__).parent.parent / "examples" / "mace_calculations.csv"
    energy_calc = MaceEnergyCalculator(
        csv_file=str(csv_file),
        mp_api_key=MP_API_KEY
    )

    # Initialize MCTS
    root_node = MCTSTreeNode(atoms, f_block_mode='u_only', exploration_constant=0.1)
    mcts = MCTS(root_node)

    # Track iteration history manually
    iteration_history = []
    cumulative_best_e_form = float('inf')  # Track the actual best formation energy
    cumulative_best_e_above_hull = float('inf')  # Track the actual best energy above hull

    # Run MCTS iteration by iteration to track progress
    for iter_num in range(n_iterations):
        if mcts.terminated:
            break

        # Selection
        select_chain = mcts.select_node(mode='epsilon')

        if mcts.terminated:
            break

        # Record statistics
        mcts.stat_node_visited()

        # Expansion and simulation
        reward, renew_t_to_terminate = mcts.expansion_simulation(
            rollout_depth=1,
            n_rollout=5,
            energy_calculator=energy_calc,
            rollout_method=rollout_method
        )

        # Back-propagation
        mcts.back_propagation(reward, select_chain, renew_t_to_terminate)

        # Update statistics
        mcts.stat_node_visited()

        # Update cumulative best values by checking ALL discovered compounds in stat_dict
        for formula, stats in mcts.stat_dict.items():
            # stats format: [best_reward, visit_count, terminated, e_above_hull, e_form]
            e_form = stats[4]
            e_above_hull = stats[3]
            cumulative_best_e_form = min(cumulative_best_e_form, e_form)
            cumulative_best_e_above_hull = min(cumulative_best_e_above_hull, e_above_hull)

        # Track history every 10 iterations
        if iter_num % 10 == 0 or iter_num == n_iterations - 1:
            iteration_history.append({
                'iteration': iter_num,
                'best_e_form': cumulative_best_e_form,
                'best_e_above_hull': cumulative_best_e_above_hull,
                'compounds_explored': len(mcts.stat_dict)
            })

    # Final result
    iterations_completed = iter_num + 1 if not mcts.terminated else iter_num

    # Collect statistics
    analyzer = ResultsAnalyzer(csv_file=str(csv_file))
    efficiency = analyzer.analyze_search_efficiency(mcts.stat_dict)

    # Get top compounds
    top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)

    # Package results
    result_data = {
        'replicate_id': replicate_id,
        'rollout_method': rollout_method,
        'iterations_completed': iterations_completed,
        'compounds_explored': len(mcts.stat_dict),
        'best_compound': mcts.best_node.get_chemical_formula() if mcts.best_node else None,
        'best_formation_energy': mcts.best_node.e_form if mcts.best_node else None,
        'best_energy_above_hull': mcts.best_node.e_above_hull if mcts.best_node else None,
        'compounds_near_hull_100meV': efficiency['compounds_near_hull_100meV'],
        'search_diversity': efficiency['search_diversity'],
        'top_10_compounds': top_compounds.to_dict('records'),
        'iteration_history': iteration_history,
        'stat_dict': {k: v for k, v in mcts.stat_dict.items()}
    }

    print(f"✓ Replicate {replicate_id} completed")
    print(f"  Best compound: {mcts.best_node.get_chemical_formula() if mcts.best_node else 'None'}")
    if mcts.best_node:
        print(f"  Formation energy: {mcts.best_node.e_form:.4f} eV/atom")
    print(f"  Compounds explored: {len(mcts.stat_dict)}")

    return result_data


def run_comparison_study(rollout_methods=['fe', 'eh', 'both'], n_replicates=10, n_iterations=1000):
    """
    Run comparison study across different rollout methods.

    Args:
        rollout_methods: List of rollout methods to test
        n_replicates: Number of replicates per method
        n_iterations: Number of MCTS iterations per replicate

    Returns:
        list: All results data
    """
    all_results = []

    for method in rollout_methods:
        print(f"\n{'#'*80}")
        print(f"# TESTING ROLLOUT METHOD: {method.upper()}")
        print(f"{'#'*80}")

        for rep in range(1, n_replicates + 1):
            try:
                result = run_single_mcts_replicate(method, rep, n_iterations)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in replicate {rep} for method {method}: {e}")
                continue

    return all_results


def create_comparison_visualizations(all_results, output_dir):
    """
    Create comprehensive comparison visualizations.

    Args:
        all_results: List of result dictionaries
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Convert to DataFrame for easier analysis
    summary_data = []
    for result in all_results:
        summary_data.append({
            'rollout_method': result['rollout_method'],
            'replicate_id': result['replicate_id'],
            'best_formation_energy': result['best_formation_energy'],
            'best_energy_above_hull': result['best_energy_above_hull'],
            'compounds_explored': result['compounds_explored'],
            'compounds_near_hull_100meV': result['compounds_near_hull_100meV'],
            'search_diversity': result['search_diversity']
        })

    df = pd.DataFrame(summary_data)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 1. Convergence Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for method in df['rollout_method'].unique():
        method_results = [r for r in all_results if r['rollout_method'] == method]

        # Formation energy convergence
        for result in method_results:
            history = result['iteration_history']
            iterations = [h['iteration'] for h in history]
            fe_history = [h['best_e_form'] for h in history]
            axes[0].plot(iterations, fe_history, alpha=0.3,
                        color={'fe': 'blue', 'eh': 'red', 'both': 'green'}[method])

        # Energy above hull convergence
        for result in method_results:
            history = result['iteration_history']
            iterations = [h['iteration'] for h in history]
            eh_history = [h['best_e_above_hull'] for h in history]
            axes[1].plot(iterations, eh_history, alpha=0.3,
                        color={'fe': 'blue', 'eh': 'red', 'both': 'green'}[method])

    # Add mean curves
    for method in df['rollout_method'].unique():
        method_results = [r for r in all_results if r['rollout_method'] == method]

        # Get common iteration points (use first result as reference)
        if method_results:
            ref_iterations = [h['iteration'] for h in method_results[0]['iteration_history']]

            # Formation energy mean
            fe_at_iterations = []
            for iter_point in ref_iterations:
                fe_values = []
                for result in method_results:
                    history = result['iteration_history']
                    # Find closest iteration
                    for h in history:
                        if h['iteration'] >= iter_point:
                            fe_values.append(h['best_e_form'])
                            break
                if fe_values:
                    fe_at_iterations.append(np.mean(fe_values))

            axes[0].plot(ref_iterations[:len(fe_at_iterations)], fe_at_iterations,
                        linewidth=3, label=method,
                        color={'fe': 'blue', 'eh': 'red', 'both': 'green'}[method])

            # Energy above hull mean
            eh_at_iterations = []
            for iter_point in ref_iterations:
                eh_values = []
                for result in method_results:
                    history = result['iteration_history']
                    # Find closest iteration
                    for h in history:
                        if h['iteration'] >= iter_point:
                            eh_values.append(h['best_e_above_hull'])
                            break
                if eh_values:
                    eh_at_iterations.append(np.mean(eh_values))

            axes[1].plot(ref_iterations[:len(eh_at_iterations)], eh_at_iterations,
                        linewidth=3, label=method,
                        color={'fe': 'blue', 'eh': 'red', 'both': 'green'}[method])

    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Best Formation Energy (eV/atom)', fontsize=12)
    axes[0].set_title('Formation Energy Convergence', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Best Energy Above Hull (eV/atom)', fontsize=12)
    axes[1].set_title('Energy Above Hull Convergence', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Summary statistics table
    summary_stats = df.groupby('rollout_method').agg({
        'best_formation_energy': ['mean', 'std', 'min'],
        'best_energy_above_hull': ['mean', 'std', 'min'],
        'compounds_explored': ['mean', 'std'],
        'compounds_near_hull_100meV': ['mean', 'std', 'max']
    }).round(4)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_stats)

    # Save summary statistics
    summary_stats.to_csv(output_path / 'summary_statistics.csv')

    print(f"\n✓ All visualizations saved to: {output_path.absolute()}")


def main():
    """Main execution function."""
    print("="*80)
    print("ROLLOUT METHOD COMPARISON STUDY")
    print("="*80)
    print("\nThis study will:")
    print("  • Test 3 rollout methods: 'fe', 'eh', 'both'")
    print("  • Run 10 replicates for each method")
    print("  • Generate comparison visualizations")
    print("  • Analyze search efficiency and convergence")
    print("\nTotal runs: 30 (3 methods × 10 replicates)")
    print("Estimated time: ~30-60 minutes (depending on cache hit rate)")

    output_dir = Path(__file__).parent.parent / "rollout_method_comparison_results"

    # Run comparison study
    all_results = run_comparison_study(
        rollout_methods=['fe', 'eh', 'both'],
        n_replicates=10,
        n_iterations=1000
    )

    # Save raw results
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'raw_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n✓ Raw results saved to: {output_dir / 'raw_results.json'}")

    # Create visualizations
    print("\nGenerating comparison visualizations...")
    create_comparison_visualizations(all_results, output_dir)

    print("\n" + "="*80)
    print("ROLLOUT METHOD COMPARISON STUDY COMPLETE!")
    print("="*80)
    print(f"\n📁 Results directory: {output_dir.absolute()}")
    print("\n📊 Files created:")
    print("   • convergence_comparison.png")
    print("   • summary_statistics.csv")
    print("   • raw_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())

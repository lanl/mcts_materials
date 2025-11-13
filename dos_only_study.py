#!/usr/bin/env python3
"""
DOS-only study: Varying gamma with alpha=0, beta=0 to understand
pure spectroscopic optimization behavior.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcts_crystal import (
    MCTSTreeNode,
    MCTS,
    MaceEnergyCalculator,
    DoscarRewardLookup
)
from ase.io import read


def run_mcts_dos_only(atoms, energy_calc, doscar_lookup, gamma, n_iterations=1000,
                      seed=None, epsilon=0.5, termination_limit=500):
    """Run MCTS with only DOS rewards (alpha=0, beta=0).

    Args:
        atoms: Starting structure
        energy_calc: Energy calculator
        doscar_lookup: DOS reward lookup
        gamma: DOS reward weight
        n_iterations: Number of MCTS iterations
        seed: Random seed for reproducibility (if None, uses non-deterministic seed)
        epsilon: Exploration rate for epsilon-greedy (default: 0.5 = 50% exploration)
        termination_limit: Node visit limit before termination (default: 500)
    """
    import random

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"\nRunning MCTS with α=0.0, β=0.0, γ={gamma}, seed={seed}, ε={epsilon}")
    else:
        print(f"\nRunning MCTS with α=0.0, β=0.0, γ={gamma}, ε={epsilon} (no seed)")

    root_node = MCTSTreeNode(atoms, f_block_mode='lanthanides_u_extended',
                            exploration_constant=2*np.sqrt(2),
                            termination_limit=termination_limit)
    mcts = MCTS(root_node, epsilon=epsilon)

    results = mcts.run(
        n_iterations=n_iterations,
        energy_calculator=energy_calc,
        rollout_depth=20,
        n_rollout=20,
        selection_mode='epsilon',
        rollout_method='weighted',
        alpha=0.0,
        beta=0.0,
        gamma=gamma,
        doscar_lookup=doscar_lookup
    )

    stat_df = mcts.get_statistics_dataframe()
    results['gamma'] = gamma
    results['stat_df'] = stat_df

    print(f"✓ Best: {results['best_node_formula']} (E_form={results['best_node_e_form']:.4f})")
    return results


def main():
    """DOS-only study main function."""
    print("="*80)
    print("DOS-ONLY STUDY: Pure Spectroscopic Optimization")
    print("="*80)

    # Load resources
    print("\nLoading resources...")
    cif_file = Path("examples/mat_Pb6U1W6_sg191.cif")
    atoms = read(str(cif_file))
    csv_file = Path("high_throughput_mace_results.full.csv")
    energy_calc = MaceEnergyCalculator(csv_file=str(csv_file), mp_api_key=None)
    doscar_lookup = DoscarRewardLookup()
    print("✓ Resources loaded")

    # Test different gamma values with alpha=0, beta=0
    # Run multiple seeds to avoid getting stuck in local optima
    gamma_values = [1.0]
    n_seeds = 1  # Run 5 different random seeds

    print(f"\nRunning {len(gamma_values)} gamma configurations × {n_seeds} seeds = {len(gamma_values) * n_seeds} total runs")
    print(f"Each run: 5000 iterations, ε=0.5, termination_limit=500")

    all_results = []
    for gamma in gamma_values:
        for seed in range(n_seeds):
            print(f"\n[γ={gamma}, seed={seed}]")
            try:
                results = run_mcts_dos_only(
                    atoms, energy_calc, doscar_lookup,
                    gamma=gamma,
                    n_iterations=10000,
                    seed=seed,
                    epsilon=0.5,
                    termination_limit=250
                )
                results['label'] = f'DOS-only (γ={gamma}, seed={seed})'
                results['seed'] = seed
                all_results.append(results)
            except Exception as e:
                print(f"❌ Error: {e}")

    # Save results
    print("\nSaving results...")
    output_dir = Path("dos_only_study_results")
    output_dir.mkdir(exist_ok=True)

    summary_data = []
    for res in all_results:
        stat_df = res['stat_df']

        # Convert dos_reward to numeric if needed
        stat_df['dos_reward'] = pd.to_numeric(stat_df['dos_reward'], errors='coerce').fillna(0.0)

        # Get top 10 compounds by DOS reward
        top_dos = stat_df.nlargest(10, 'dos_reward')

        # Get statistics
        summary_data.append({
            'label': res['label'],
            'gamma': res['gamma'],
            'seed': res.get('seed', 0),
            'iterations': res['iterations_completed'],
            'compounds_explored': len(res['stat_dict']),
            'best_compound': res['best_node_formula'],
            'best_e_form': res['best_node_e_form'],
            'best_e_above_hull': res['best_node_e_above_hull'],
            'avg_dos_reward': stat_df['dos_reward'].mean(),
            'max_dos_reward': stat_df['dos_reward'].max(),
            'top_dos_compound': top_dos.index[0] if len(top_dos) > 0 else 'N/A',
            'top_dos_reward': top_dos['dos_reward'].iloc[0] if len(top_dos) > 0 else 0.0,
            'top_dos_e_form': top_dos['e_form'].iloc[0] if len(top_dos) > 0 else 0.0
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "dos_only_summary.csv", index=False)

    # Save detailed results for each run
    for res in all_results:
        label_safe = res['label'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        filename = f"results_{label_safe}.csv"
        res['stat_df'].to_csv(output_dir / filename)

    # Create analysis report
    print("\n" + "="*80)
    print("DOS-ONLY STUDY RESULTS")
    print("="*80)

    # Find overall best across all seeds
    best_overall = summary_df.loc[summary_df['max_dos_reward'].idxmax()]
    print("\n🏆 BEST RESULT ACROSS ALL SEEDS:")
    print(f"   Seed: {best_overall['seed']}")
    print(f"   Best compound: {best_overall['top_dos_compound']}")
    print(f"   DOS reward: {best_overall['max_dos_reward']:.4f}")
    print(f"   E_form: {best_overall['top_dos_e_form']:.4f} eV/atom")
    print(f"   Iterations: {best_overall['iterations']}")
    print(f"   Compounds explored: {best_overall['compounds_explored']}")

    print("\n1. Results by seed:")
    for _, row in summary_df.iterrows():
        print(f"\n   Seed {row['seed']}:")
        print(f"      Top compound: {row['top_dos_compound']}")
        print(f"      DOS reward: {row['top_dos_reward']:.4f}")
        print(f"      E_form: {row['top_dos_e_form']:.4f} eV/atom")
        print(f"      Compounds explored: {row['compounds_explored']}")
        print(f"      Iterations completed: {row['iterations']}")

    print("\n2. DOS reward statistics across seeds:")
    print(f"   Best DOS reward found: {summary_df['max_dos_reward'].max():.4f}")
    print(f"   Worst DOS reward found: {summary_df['max_dos_reward'].min():.4f}")
    print(f"   Mean DOS reward: {summary_df['max_dos_reward'].mean():.4f}")
    print(f"   Std DOS reward: {summary_df['max_dos_reward'].std():.4f}")
    print(f"   Mean compounds explored: {summary_df['compounds_explored'].mean():.1f}")
    print(f"   Mean iterations completed: {summary_df['iterations'].mean():.1f}")

    # Save metadata
    metadata = {
        'n_iterations': 10000,
        'n_seeds': n_seeds,
        'epsilon': 0.5,
        'f_block_mode': 'lanthanides_u_extended',
        'exploration_constant': float(2 * np.sqrt(2)),
        'termination_limit': 250,
        'alpha': 0.0,
        'beta': 0.0,
        'gamma_values': gamma_values,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ DOS-only study completed! Results in: {output_dir}")

    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find the gamma that gives best balance
    summary_df['stability_penalty'] = summary_df['best_e_form'].clip(lower=0) * 10
    summary_df['dos_score'] = summary_df['max_dos_reward'] - summary_df['stability_penalty']
    best_gamma = summary_df.loc[summary_df['dos_score'].idxmax()]

    print("\n1. For pure spectroscopic optimization:")
    print(f"   Recommended γ: {best_gamma['gamma']}")
    print(f"   This found: {best_gamma['top_dos_compound']} with DOS reward = {best_gamma['top_dos_reward']:.4f}")

    print("\n2. Stability considerations:")
    stable_compounds = []
    for res in all_results:
        stable = res['stat_df'][res['stat_df']['e_form'] < 0]
        if len(stable) > 0:
            stable_compounds.append({
                'gamma': res['gamma'],
                'count': len(stable),
                'best': stable['e_form'].min()
            })

    if stable_compounds:
        print(f"   Number of thermodynamically favorable compounds (E_form < 0) found:")
        for sc in stable_compounds:
            print(f"      γ={sc['gamma']}: {sc['count']} compounds, best E_form={sc['best']:.4f}")
    else:
        print("   ⚠️  No thermodynamically favorable compounds (E_form < 0) found with pure DOS optimization")
        print("   Recommendation: Use balanced approach (α=1.0, β=1.0, γ=1.0-2.0) to balance stability and spectroscopy")

    print("\n3. High DOS reward compounds to investigate further:")
    # Collect all high DOS compounds across all runs
    high_dos = []
    for res in all_results:
        top = res['stat_df'].nlargest(3, 'dos_reward')
        for idx, row in top.iterrows():
            high_dos.append({
                'compound': idx,
                'dos_reward': row['dos_reward'],
                'e_form': row['e_form'],
                'e_above_hull': row['e_above_hull'],
                'gamma': res['gamma']
            })

    # Sort by DOS reward and show top 5 unique
    high_dos_df = pd.DataFrame(high_dos).drop_duplicates('compound').nlargest(5, 'dos_reward')
    for _, row in high_dos_df.iterrows():
        print(f"\n      {row['compound']}:")
        print(f"         DOS reward: {row['dos_reward']:.4f}")
        print(f"         E_form: {row['e_form']:.4f} eV/atom")
        print(f"         E_above_hull: {row['e_above_hull']:.4f} eV/atom")
        stability = "Stable" if row['e_form'] < 0 and row['e_above_hull'] < 0.1 else "Metastable/Unstable"
        print(f"         Status: {stability}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

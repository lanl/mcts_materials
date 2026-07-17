"""Shared setup/helpers for the product-mode MCTS sensitivity studies.

Identical structure to ../common.py but with a BASELINE that reflects the
published product-reward formulation:
  - rollout_method='ehull_rdos_product'  (reward = r_Ehull × r_DOS)
  - gamma=1.0                            (no r_DOS normalisation)
  - n_rollout=1                          (single depth-0 evaluation per expansion)
  - move_step=1

The starting material is Pd6Ge6U (d=6 from UTi6Sn6 in the 108-compound U-only
move graph), identical to the additive sensitivity study baseline, so results
are directly comparable.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcts_crystal.node import MCTSTreeNode
from mcts_crystal.mcts import MCTS
from mcts_crystal.energy_calculator import MaceEnergyCalculator
from mcts_crystal.doscar_utils import DoscarRewardLookup
from mcts_crystal.cli import override_composition

BASELINE = dict(
    transition_metal='Pd',
    group_iv='Ge',
    f_block_mode='u_only',
    selection_mode='ucb1',
    exploration_constant=0.5,
    epsilon=0.2,
    temperature=1.0,
    termination_limit=25,
    rollout_depth=3,
    n_rollout=1,
    rollout_method='ehull_rdos_product',
    beta=1.0,
    gamma=1.0,
    rollout_aggregation='max',
    rollout_discount=1.0,
    move_step=1,
)

N_SEEDS = 10
N_ITERATIONS = 500
SEEDS = list(range(N_SEEDS))

RESULTS_DIR = REPO_ROOT / 'sensitivity_studies' / 'results' / 'product_mode'


def load_shared_resources():
    atoms_template = read(str(REPO_ROOT / 'examples' / 'mat_Pb6U1W6_sg191.cif'))
    energy_calc = MaceEnergyCalculator(
        csv_file=str(REPO_ROOT / 'high_throughput_mace_results.full.csv'),
        mp_api_key=None,
    )
    doscar_lookup = DoscarRewardLookup(
        peaks_file=str(REPO_ROOT / 'doscar_peaks_data_with_U.csv')
    )
    return atoms_template, energy_calc, doscar_lookup


def run_replicate(atoms_template, energy_calc, doscar_lookup, seed, params):
    cfg = {**BASELINE, **params}

    random.seed(seed)
    np.random.seed(seed)

    atoms = override_composition(
        atoms_template.copy(),
        transition_metal=cfg['transition_metal'],
        group_iv=cfg['group_iv'],
    )
    root = MCTSTreeNode(
        atoms, f_block_mode=cfg['f_block_mode'],
        exploration_constant=cfg['exploration_constant'],
        termination_limit=cfg['termination_limit'],
        move_step=cfg.get('move_step', 1),
    )
    root.e_form, root.e_above_hull = energy_calc.calculate_energies(atoms)

    mcts = MCTS(root, epsilon=cfg['epsilon'], temperature=cfg['temperature'])
    mcts.run(
        n_iterations=N_ITERATIONS,
        energy_calculator=energy_calc,
        rollout_depth=cfg['rollout_depth'],
        n_rollout=cfg['n_rollout'],
        selection_mode=cfg['selection_mode'],
        rollout_method=cfg['rollout_method'],
        beta=cfg['beta'],
        gamma=cfg['gamma'],
        doscar_lookup=doscar_lookup,
        rollout_aggregation=cfg['rollout_aggregation'],
        rollout_discount=cfg['rollout_discount'],
    )
    return np.array(mcts.max_reward_history), np.array(mcts.n_unique_compounds_history)


def sweep_result_path(sweep_name, filename='convergence_data.csv'):
    out_dir = RESULTS_DIR / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def _resolved_config_fingerprint(param_overrides_by_value):
    resolved = {label: {**BASELINE, **overrides}
                for label, overrides in param_overrides_by_value.items()}
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def run_sweep(sweep_name, param_overrides_by_value, checkpoint_path=None):
    atoms_template, energy_calc, doscar_lookup = load_shared_resources()
    fingerprint = _resolved_config_fingerprint(param_overrides_by_value)
    meta_path = Path(checkpoint_path).with_suffix('.meta.json') if checkpoint_path is not None else None

    COLS = ['value', 'seed', 'iteration', 'n_unique_compounds', 'best_reward']
    rows = []
    done_pairs = set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        saved_fingerprint = None
        if meta_path is not None and meta_path.exists():
            try:
                saved_fingerprint = json.loads(meta_path.read_text()).get('fingerprint')
            except Exception:
                saved_fingerprint = None
        if saved_fingerprint != fingerprint:
            print(f"[{sweep_name}] checkpoint fingerprint mismatch — starting over")
        else:
            existing = pd.read_csv(checkpoint_path)
            if 'n_unique_compounds' in existing.columns:
                rows = list(existing[COLS].itertuples(index=False, name=None))
            else:
                print(f"[{sweep_name}] checkpoint missing n_unique_compounds — re-running all")
                existing = pd.DataFrame()
            if len(rows):
                done_pairs = set(zip(existing['value'], existing['seed']))
                print(f"[{sweep_name}] resuming: {len(done_pairs)} (value, seed) pairs done")

    def _checkpoint():
        if checkpoint_path is not None:
            pd.DataFrame(rows, columns=COLS).to_csv(checkpoint_path, index=False)
            meta_path.write_text(json.dumps({'fingerprint': fingerprint}))

    t0 = time.time()
    for value_label, overrides in param_overrides_by_value.items():
        for seed in SEEDS:
            if (value_label, seed) in done_pairs:
                continue
            ts = time.time()
            reward_hist, unique_hist = run_replicate(atoms_template, energy_calc, doscar_lookup, seed, overrides)
            print(f"[{sweep_name}] value={value_label!r} seed={seed} "
                  f"-> final_best={reward_hist[-1]:.4f} ({time.time()-ts:.1f}s)")
            for it, (reward, n_uniq) in enumerate(zip(reward_hist, unique_hist)):
                rows.append((value_label, seed, it, int(n_uniq), reward))
            _checkpoint()
    print(f"[{sweep_name}] total runtime: {time.time()-t0:.1f}s")

    return pd.DataFrame(rows, columns=COLS)


def save_sweep_results(df, sweep_name, filename='convergence_data.csv'):
    out_path = sweep_result_path(sweep_name, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path

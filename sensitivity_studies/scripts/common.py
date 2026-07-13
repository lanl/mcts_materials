"""Shared setup/helpers for the MCTS sensitivity studies.

Each sweep script varies exactly one hyperparameter (or a small group of
closely related ones) away from the calibrated baseline below, runs N_SEEDS
replicate MCTS searches per value, and writes a long-format CSV of
best_reward (the composite reward, beta*ehull_reward + gamma*r_DOS) vs.
iteration to results/<sweep_name>/. plot_sweep.py then turns that CSV into a
convergence figure.

beta, gamma, the ehull_reward sharpness constant (120), the E_hull stability
threshold (0.05), and the rDOS Gaussian width sigma (0.5) are physics-informed
and are NOT swept here - only MCTS-algorithm hyperparameters are.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[2]

from mcts_crystal.node import MCTSTreeNode
from mcts_crystal.mcts import MCTS
from mcts_crystal.energy_calculator import MaceEnergyCalculator
from mcts_crystal.doscar_utils import DoscarRewardLookup
from mcts_crystal.cli import override_composition

# Calibrated baseline. Matches the main published study's settings
# (config.json + analysis/ehull_rdos_u_only_study/run_study.sh's CLI flags -
# termination_limit/rollout_depth/n_rollout aren't in config.json itself,
# they're hardcoded on run_study.sh's run_mcts.py invocation) except for the
# starting material: these sweeps intentionally start from Pd6Ge6U (move-
# graph distance d=6 from the true global-best U-only compound, see
# compute_global_u_only_ranks in analysis/ehull_rdos_u_only_study/
# generate_figures.py - rank 54/108, i.e. an unremarkable, non-cherry-picked
# point) rather than the main study's Cr6Sn6U (d=3), so that hyperparameter
# effects aren't masked by starting so close to the optimum that the search
# converges before any hyperparameter can matter.
#
# n_rollout=2/rollout_depth=2 means: every expansion draws two reward samples.
# The first (depth=0) evaluates the new node's own composition and records its
# e_form/e_above_hull. The second (depth=2) performs a max-along-walk: two
# successive random substitutions are applied, the composite reward is scored
# at *each* intermediate composition, and the maximum of those two step-rewards
# is returned. rollout_discount=1.0 means neither sample is discounted before
# taking the max. With n_rollout=1 the depth>0 walk never runs, making
# rollout_depth a no-op. Each sweep overrides only the key(s) it is studying.
# gamma = 1/max(r_DOS over 108 U-only compounds) = 1/2516.1664410449775,
# matching analysis/ehull_rdos_u_only_study_max_undiscounted/run_study.sh.
BASELINE = dict(
    transition_metal='Pd',
    group_iv='Ge',
    f_block_mode='u_only',
    selection_mode='ucb1',
    exploration_constant=0.5,
    epsilon=0.2,
    temperature=1.0,
    termination_limit=25,
    rollout_depth=2,
    n_rollout=2,
    rollout_method='ehull_rdos',
    beta=1.0,
    gamma=0.00039742998860786596,
    rollout_aggregation='max',
    rollout_discount=1.0,
    move_step=1,
)

N_SEEDS = 10
N_ITERATIONS = 500
SEEDS = list(range(N_SEEDS))


def load_shared_resources():
    """Load the starting-structure template and the (seed-independent,
    reusable) energy/rDOS lookups once, to be shared across every replicate
    run in a sweep."""
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
    """Run a single fresh MCTS search (params overrides BASELINE) and return
    its best_reward-vs-iteration array (length N_ITERATIONS+1, including the
    iteration-0 pre-search sentinel)."""
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
    return np.array(mcts.max_reward_history)


def sweep_result_path(sweep_name, filename='convergence_data.csv'):
    """Path where a sweep's checkpointed/final results CSV lives, creating
    the containing directory if needed."""
    out_dir = REPO_ROOT / 'sensitivity_studies' / 'results' / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def _resolved_config_fingerprint(param_overrides_by_value):
    """Hash of the fully-resolved (BASELINE + per-value overrides) config.

    Guards against a real bug: matching a checkpoint purely by (value_label,
    seed) is unsafe across runs where BASELINE itself changed (e.g. a global
    n_rollout bump) but a sweep's value *labels* happened to stay the same
    (e.g. rollout_depth's labels '1'/'3 (calibrated)'/'5' didn't change even
    though their fixed n_rollout override did, and termination_limit's
    labels never override n_rollout at all, so they silently inherit
    whatever BASELINE says) - stale, wrong-baseline rows would otherwise look
    "already done" and get skipped instead of recomputed.
    """
    resolved = {label: {**BASELINE, **overrides}
                for label, overrides in param_overrides_by_value.items()}
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def run_sweep(sweep_name, param_overrides_by_value, checkpoint_path=None):
    """Run every (value, seed) replicate for a sweep and return a tidy
    long-format DataFrame: columns = [value, seed, iteration, best_reward].

    Args:
        sweep_name: used only for the progress log
        param_overrides_by_value: dict mapping a human-readable value label
            (e.g. "0.5" or "Cr6Sn6U") to the dict of BASELINE overrides that
            produce it (e.g. {'exploration_constant': 0.5})
        checkpoint_path: if given (use sweep_result_path() to compute it),
            results are written to this CSV after every single replicate,
            not just once at the end - and any (value, seed) pairs already
            present there on startup are skipped, *provided* the resolved
            BASELINE+overrides config fingerprint (see
            _resolved_config_fingerprint) still matches what produced that
            checkpoint - otherwise the whole checkpoint is treated as stale
            and the sweep starts over from scratch. So a killed/interrupted
            sweep can be resumed by just rerunning the same script instead of
            losing all progress, without silently reusing rows computed
            under a since-changed baseline.
    """
    atoms_template, energy_calc, doscar_lookup = load_shared_resources()
    fingerprint = _resolved_config_fingerprint(param_overrides_by_value)
    meta_path = Path(checkpoint_path).with_suffix('.meta.json') if checkpoint_path is not None else None

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
            print(f"[{sweep_name}] checkpoint exists but its config fingerprint doesn't "
                  f"match (BASELINE or this sweep's overrides changed since it was "
                  f"written) - ignoring it and starting this sweep over from scratch")
        else:
            existing = pd.read_csv(checkpoint_path)
            rows = list(existing[['value', 'seed', 'iteration', 'best_reward']]
                        .itertuples(index=False, name=None))
            done_pairs = set(zip(existing['value'], existing['seed']))
            if done_pairs:
                print(f"[{sweep_name}] resuming from checkpoint: "
                      f"{len(done_pairs)} (value, seed) pairs already done")

    def _checkpoint():
        if checkpoint_path is not None:
            pd.DataFrame(rows, columns=['value', 'seed', 'iteration', 'best_reward']) \
                .to_csv(checkpoint_path, index=False)
            meta_path.write_text(json.dumps({'fingerprint': fingerprint}))

    t0 = time.time()
    for value_label, overrides in param_overrides_by_value.items():
        for seed in SEEDS:
            if (value_label, seed) in done_pairs:
                continue
            ts = time.time()
            history = run_replicate(atoms_template, energy_calc, doscar_lookup, seed, overrides)
            print(f"[{sweep_name}] value={value_label!r} seed={seed} "
                  f"-> {len(history)} pts, final_best={history[-1]:.4f} ({time.time()-ts:.1f}s)")
            for it, reward in enumerate(history):
                rows.append((value_label, seed, it, reward))
            _checkpoint()
    print(f"[{sweep_name}] total runtime: {time.time()-t0:.1f}s")

    return pd.DataFrame(rows, columns=['value', 'seed', 'iteration', 'best_reward'])


def save_sweep_results(df, sweep_name, filename='convergence_data.csv'):
    out_path = sweep_result_path(sweep_name, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path

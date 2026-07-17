"""Sensitivity sweep: UCB1 exploration constant c (--exploration-constant).
Product-reward formulation (r_Ehull × r_DOS, gamma=1, n_rollout=1).
"""

from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'c_sweep'

VALUES = {
    '0.05': dict(exploration_constant=0.05),
    '0.1': dict(exploration_constant=0.1),
    '0.2': dict(exploration_constant=0.2),
    '0.5 (calibrated)': dict(exploration_constant=0.5),
    '1.0': dict(exploration_constant=1.0),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES, checkpoint_path=sweep_result_path(SWEEP_NAME))
    save_sweep_results(df, SWEEP_NAME)

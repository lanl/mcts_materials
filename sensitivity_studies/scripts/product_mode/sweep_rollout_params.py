"""Sensitivity sweep: termination_limit and rollout_depth.
Product-reward formulation (r_Ehull × r_DOS, gamma=1).

n_rollout is fixed at 1 (BASELINE) for termination_limit.
For rollout_depth, n_rollout=2 is used so that the extra rollout samples
actually execute (with n_rollout=1 rollout_depth is a no-op).
"""

from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'rollout_params_sweep'

ROLLOUT_DEPTH_VALUES = {
    '1': dict(rollout_depth=1, n_rollout=2),
    '3 (calibrated)': dict(rollout_depth=3, n_rollout=2),
    '5': dict(rollout_depth=5, n_rollout=2),
}

TERMINATION_LIMIT_VALUES = {
    '10': dict(termination_limit=10),
    '25 (calibrated)': dict(termination_limit=25),
    '50': dict(termination_limit=50),
}

if __name__ == '__main__':
    path = sweep_result_path(SWEEP_NAME, filename='convergence_data_rollout_depth.csv')
    df = run_sweep(f'{SWEEP_NAME}/rollout_depth', ROLLOUT_DEPTH_VALUES, checkpoint_path=path)
    save_sweep_results(df, SWEEP_NAME, filename='convergence_data_rollout_depth.csv')

    path = sweep_result_path(SWEEP_NAME, filename='convergence_data_termination_limit.csv')
    df = run_sweep(f'{SWEEP_NAME}/termination_limit', TERMINATION_LIMIT_VALUES, checkpoint_path=path)
    save_sweep_results(df, SWEEP_NAME, filename='convergence_data_termination_limit.csv')

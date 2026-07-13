"""Sensitivity sweep: child-selection strategy (--selection-mode).

Compares all five available modes head-to-head at the calibrated
exploration_constant/epsilon/temperature. See README.md "Child Selection
Methods" for what each mode does.
"""

from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'selection_mode_sweep'

VALUES = {
    'Ucb1 (Calibrated)': dict(selection_mode='ucb1'),
    'Epsilon Greedy': dict(selection_mode='epsilon_greedy'),
    'Boltzmann': dict(selection_mode='boltzmann'),
    'Puct': dict(selection_mode='puct'),
    'Hybrid': dict(selection_mode='hybrid'),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES, checkpoint_path=sweep_result_path(SWEEP_NAME))
    save_sweep_results(df, SWEEP_NAME)

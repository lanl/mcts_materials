"""Generic convergence-sensitivity plot: mean best_reward vs. iteration, one
overlaid curve per swept value with a shaded 10th-90th percentile band across
replicate seeds (not mean +/- std: std is a parametric estimate that can
extend past the best value any seed actually reached, which would misleadingly
suggest the search found something better than it did - a percentile band is
computed directly from the observed seeds, so it can never exceed what was
actually seen). Single 3in x 3in panel, 10pt font (no subplots).

Usage: python plot_sweep.py <results_dir>/convergence_data.csv "title" [out.png] [--label-prefix="Depth = "]

title='' omits the title entirely. --label-prefix prepends each legend
entry's swept value (e.g. "Depth = 3 (calibrated)" instead of just
"3 (calibrated)").
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7,
    'figure.figsize': (3, 3),
})


def plot_convergence(csv_path, title, out_path, label_prefix=''):
    df = pd.read_csv(csv_path)
    # Iteration 0 is a pre-search sentinel (max_reward = -10.0, not a real
    # composite score), so the convergence curve starts at iteration 1.
    df = df[df['iteration'] > 0]

    def _pctl_agg(s):
        return pd.Series({'mean': s.mean(), 'p10': np.percentile(s, 10), 'p90': np.percentile(s, 90)})

    stats = df.groupby(['value', 'iteration'])['best_reward'].apply(_pctl_agg).unstack().reset_index()

    fig, ax = plt.subplots()
    # Preserve the order values were defined in the sweep script, not
    # alphabetical (pandas groupby sorts by default).
    value_order = list(dict.fromkeys(df['value']))
    for value in value_order:
        sub = stats[stats['value'] == value]
        ax.plot(sub['iteration'], sub['mean'], label=f"{label_prefix}{value}", linewidth=1)
        ax.fill_between(sub['iteration'], sub['p10'], sub['p90'], alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best composite reward')
    if title:
        ax.set_title(title)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    positional = [a for a in sys.argv[1:] if not a.startswith('--')]
    label_prefix = ''
    for a in sys.argv[1:]:
        if a.startswith('--label-prefix='):
            label_prefix = a.split('=', 1)[1]

    csv_path = Path(positional[0])
    title = positional[1] if len(positional) > 1 else ''
    out_path = Path(positional[2]) if len(positional) > 2 else csv_path.with_suffix('.png')
    plot_convergence(csv_path, title, out_path, label_prefix=label_prefix)

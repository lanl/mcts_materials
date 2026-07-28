"""Supplementary figure for the termination_limit sweep: mean +/- std
iterations completed before the search halts (or exhausts the 500-iteration
budget), per termination_limit value. The reward-vs-iteration convergence
curve can't show this parameter's effect, since the optimum is found well
before termination ever triggers - this is the actual quantity it controls.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (3, 3),
})


def plot_iterations_completed(csv_path, title, out_path):
    df = pd.read_csv(csv_path)
    completed = df.groupby(['value', 'seed'])['iteration'].max().reset_index()
    value_order = list(dict.fromkeys(df['value']))
    stats = completed.groupby('value')['iteration'].agg(['mean', 'std']).reindex(value_order)

    fig, ax = plt.subplots()
    ax.bar(range(len(stats)), stats['mean'], yerr=stats['std'], capsize=3,
           color='#4C72B0', width=0.6)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats.index, rotation=30, ha='right')
    ax.set_ylabel('Iterations completed')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    csv_path = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else ''
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else csv_path.with_suffix('.png')
    plot_iterations_completed(csv_path, title, out_path)

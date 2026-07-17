#!/usr/bin/env python3
"""Combined ms1 vs ms3 discovery curve figure (6x3 inches).

Left panel : move-step 1  (product_mode_g1_ms1) — no colorbar
Right panel: move-step 3  (product_mode_g1)      — no y-label, shared colorbar
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STUDY_DIR = Path(__file__).parent
REPO_ROOT  = STUDY_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.discovery_efficiency_study.generate_figures import (
    EXTENDED_MCTS_RUNS, N_EXTENDED_SEEDS, X_MAX, N_RANDOM_REPS,
    edit_distance, interpolate_curve, load_product_seeds,
    load_design_space, ehull_reward,
)
from mcts_crystal.doscar_utils import DoscarRewardLookup


def load_dos_rewards():
    peaks = REPO_ROOT / 'doscar_peaks_data_with_U.csv'
    return DoscarRewardLookup(peaks_file=str(peaks)).rewards_dict


def build_panel_data(subdir):
    """Return run_records, rand_p25, rand_p50, rand_p75, max_dist."""
    dos_rewards = load_dos_rewards()
    design_space = load_design_space(REPO_ROOT, dos_rewards)

    x_grid = np.arange(1, X_MAX + 1)
    run_records = []
    for elem, tm in EXTENDED_MCTS_RUNS:
        seed_curves = load_product_seeds(STUDY_DIR, elem, tm, N_EXTENDED_SEEDS, subdir)
        if not seed_curves:
            continue
        dist = edit_distance(elem, tm)
        interps = []
        for x_m, y_m in seed_curves:
            interp = interpolate_curve(x_m, y_m, x_grid)
            n = int(x_m.max()) if len(x_m) else 0
            interp[n:] = np.nan
            interps.append(interp)
        mat = np.array(interps)
        with np.errstate(all='ignore'):
            med = np.nanmedian(mat, axis=0)
            p25 = np.nanpercentile(mat, 25, axis=0)
            p75 = np.nanpercentile(mat, 75, axis=0)
        run_records.append({'dist': dist, 'med': med, 'p25': p25, 'p75': p75})

    r_ehull_arr = design_space['e_above_hull'].apply(ehull_reward).values
    r_dos_arr   = design_space['r_DOS'].values
    product_scores = r_ehull_arr * r_dos_arr
    rng = np.random.default_rng(0)
    N   = len(product_scores)
    rand_mat = np.empty((N_RANDOM_REPS, X_MAX))
    for rep in range(N_RANDOM_REPS):
        perm = np.arange(N); rng.shuffle(perm)
        rand_mat[rep] = np.maximum.accumulate(product_scores[perm[:X_MAX]])
    rand_p50 = np.percentile(rand_mat, 50, axis=0)
    rand_p25 = np.percentile(rand_mat, 25, axis=0)
    rand_p75 = np.percentile(rand_mat, 75, axis=0)
    max_dist = max(r['dist'] for r in run_records) if run_records else 1
    return run_records, rand_p25, rand_p50, rand_p75, max_dist, x_grid


def draw_panel(ax, run_records, rand_p25, rand_p50, rand_p75,
               max_dist, x_grid, cmap, show_ylabel, show_colorbar):
    ax.fill_between(x_grid, rand_p25, rand_p75,
                    color='#888888', alpha=0.20, label='Random')
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.8, linestyle='--')

    for rec in sorted(run_records, key=lambda r: -r['dist']):
        c     = cmap(rec['dist'] / max(max_dist, 1))
        valid = ~np.isnan(rec['med'])
        if not valid.any():
            continue
        ax.fill_between(x_grid[valid], rec['p25'][valid], rec['p75'][valid],
                        color=c, alpha=0.18, linewidth=0)
        ax.plot(x_grid[valid], rec['med'][valid], color=c, lw=1.0, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    if show_colorbar:
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label('Edit distance to\nglobal best', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Unique compounds evaluated', fontsize=9)
    if show_ylabel:
        ax.set_ylabel(r'Best product reward ($r_{E_\mathrm{Hull}} \times r_\mathrm{DOS}$)',
                      fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    print('Loading ms1 data...')
    ms1 = build_panel_data('product_mode_g1_ms1')
    print('Loading ms3 data...')
    ms3 = build_panel_data('product_mode_g1')

    max_dist = max(ms1[4], ms3[4])
    cmap = plt.colormaps['RdBu_r']

    # 3-column gridspec: two equal plot panels + thin colorbar column.
    fig = plt.figure(figsize=(6, 3))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    cax  = fig.add_subplot(gs[0, 2])

    draw_panel(ax1, *ms1[:4], max_dist, ms1[5], cmap,
               show_ylabel=True, show_colorbar=False)
    ax1.set_title('Move Step 1', fontsize=9)

    draw_panel(ax2, *ms3[:4], max_dist, ms3[5], cmap,
               show_ylabel=False, show_colorbar=False)
    ax2.set_ylabel(' ', fontsize=9)
    ax2.set_title('Move Step 3', fontsize=9)

    # Equalize x and y limits across both panels
    xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1]))
    ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1]))
    for ax in (ax1, ax2):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Shared colorbar in its own column
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Edit distance to\nglobal best', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    # Force both plot axes to identical physical width
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    w = min(pos1.width, pos2.width)
    ax1.set_position([pos1.x0, pos1.y0, w, pos1.height])
    ax2.set_position([pos2.x0, pos2.y0, w, pos2.height])
    out = STUDY_DIR / 'figures' / 'discovery_curve_ms1_vs_ms3_product.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()

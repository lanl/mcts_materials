#!/usr/bin/env python3
"""Figures and table for the U-only product-mode study.

Reward:  r_Ehull × (gamma × r_DOS)  with gamma = 1.0 (raw DOS reward)
Seeds:   5 (seed_0 … seed_4) in product_mode/
Output:  product_mode/figures/
"""
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.data import atomic_numbers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcts_crystal.doscar_utils import DoscarRewardLookup
from synthesized_compounds import SYNTHESIZED_COMPOUNDS
from create_composite_radial_tree import TRANSITION_METALS, GROUP_IV, F_BLOCK

STUDY_DIR   = Path(__file__).parent
REPO_ROOT   = STUDY_DIR.parents[1]
PRODUCT_DIR = STUDY_DIR / 'product_mode'
GAMMA       = 1.0          # raw r_DOS, not normalised
N_SEEDS     = 5
SENTINEL_EHULL = 9.9

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ehull_reward(x: float) -> float:
    return -np.tanh(120.0 * (float(x) - 0.05))


def _parse_elements(name: str) -> list:
    return re.findall(r'[A-Z][a-z]?', str(name))


def _formula_key(name: str) -> tuple:
    """Non-f-block element set (TM + group-IV) for cross-dataset matching."""
    parts = _parse_elements(name)
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


def _load_dos_rewards() -> dict:
    peaks = REPO_ROOT / 'doscar_peaks_data_with_U.csv'
    return DoscarRewardLookup(peaks_file=str(peaks)).rewards_dict


def _r_dos(formula: str, dos_rewards: dict) -> float:
    """r_DOS keyed by 'U-GIV-TM' for a U-containing compound."""
    elems = _parse_elements(formula)
    u_elem = next((e for e in elems if e == 'U'), None)
    tm     = next((e for e in elems if e in TRANSITION_METALS), None)
    giv    = next((e for e in elems if e in GROUP_IV), None)
    if u_elem and tm and giv:
        return float(dos_rewards.get(f"U-{giv}-{tm}", 0.0))
    return 0.0


def _format_name(name: str) -> str:
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(name))
    re_e, tm_p, giv_p = None, None, None
    for elem, cnt in matches:
        if not elem: continue
        if elem in F_BLOCK:            re_e  = elem
        elif elem in TRANSITION_METALS: tm_p  = (elem, cnt)
        elif elem in GROUP_IV:          giv_p = (elem, cnt)
    if re_e and tm_p and giv_p:
        return (f"{re_e}{tm_p[0]}$_{{{tm_p[1]}}}${giv_p[0]}$_{{{giv_p[1]}}}$")
    return str(name)


def _elem_set(name: str) -> frozenset:
    return frozenset(_parse_elements(str(name)))


# ---------------------------------------------------------------------------
# Global U-only ranking by product reward
# ---------------------------------------------------------------------------

def compute_global_u_only_ranks_product(dos_rewards: dict) -> dict:
    """Rank the U-only design space by r_Ehull × (GAMMA × r_DOS).

    no_mp_data compounds are penalised (e_above_hull → 10) so they rank last.
    Returns {formula_key: rank}, rank 1 = best.
    """
    mace_csv = REPO_ROOT / 'high_throughput_mace_results.full.csv'
    df = pd.read_csv(mace_csv)
    if 'name' not in df.columns:
        df = df.rename(columns={'formula': 'name'})

    # U-only: contains U, no other f-block element
    df['elems'] = df['name'].apply(lambda n: set(_parse_elements(n)))
    df = df[df['elems'].apply(lambda s: 'U' in s and not (s & (F_BLOCK - {'U'})))].copy()

    # Penalise missing MP data
    if 'data_quality' in df.columns:
        mask = df['data_quality'].isin(['no_mp_data', 'error'])
        df.loc[mask, 'e_above_hull'] = 10.0

    df['r_DOS'] = df['name'].apply(lambda n: _r_dos(n, dos_rewards))
    df['product'] = df['e_above_hull'].apply(ehull_reward) * (GAMMA * df['r_DOS'])
    df = df.sort_values('product', ascending=False).reset_index(drop=True)

    ranks = {}
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        key = _formula_key(row['name'])
        if key and key not in ranks:
            ranks[key] = rank
    print(f"U-only design space: {len(df)} compounds, "
          f"best product reward = {df['product'].iloc[0]:.2f} ({df['name'].iloc[0]})")
    return ranks


# ---------------------------------------------------------------------------
# Pool discovered compounds across all seeds
# ---------------------------------------------------------------------------

def load_pooled_compounds(dos_rewards: dict) -> pd.DataFrame:
    """Pool all_compounds.csv across 5 seeds; deduplicate; add product reward."""
    frames = []
    for seed in range(N_SEEDS):
        csv = PRODUCT_DIR / f'seed_{seed}' / 'all_compounds.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if 'name' not in df.columns:
            df = df.rename(columns={'formula': 'name'})
        frames.append(df[['name', 'e_above_hull']].copy())
    if not frames:
        raise FileNotFoundError("No all_compounds.csv found in product_mode/seed_*/")
    pooled = pd.concat(frames, ignore_index=True)
    pooled = (pooled.sort_values('e_above_hull')
                    .drop_duplicates(subset=['name'], keep='first')
                    .reset_index(drop=True))
    pooled['r_DOS'] = pooled['name'].apply(lambda n: _r_dos(n, dos_rewards))
    pooled['r_ehull'] = pooled['e_above_hull'].apply(ehull_reward)
    pooled['product'] = pooled['r_ehull'] * (GAMMA * pooled['r_DOS'])
    return pooled.sort_values('product', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1: ehull vs r_DOS scatter
# ---------------------------------------------------------------------------

def plot_ehull_vs_rdos_product(pooled: pd.DataFrame, dos_rewards: dict,
                                figures_dir: Path) -> None:
    """Full U-only design space (gray) + product-mode top-15 overlay (blue)."""
    # Build U-only design space backdrop
    mace_csv = REPO_ROOT / 'high_throughput_mace_results.full.csv'
    df_all = pd.read_csv(mace_csv)
    if 'name' not in df_all.columns:
        df_all = df_all.rename(columns={'formula': 'name'})
    df_all['elems'] = df_all['name'].apply(lambda n: set(_parse_elements(n)))
    df_u = df_all[df_all['elems'].apply(
        lambda s: 'U' in s and not (s & (F_BLOCK - {'U'})))].copy()
    df_u['r_DOS'] = df_u['name'].apply(lambda n: _r_dos(n, dos_rewards))

    # Penalise no_mp_data in backdrop
    if 'data_quality' in df_u.columns:
        mask = df_u['data_quality'].isin(['no_mp_data', 'error'])
        df_u.loc[mask, 'e_above_hull'] = 10.0

    top15 = pooled.head(15)

    # Synthesised compounds overlay
    synth_sets = [_elem_set(n) for n in SYNTHESIZED_COMPOUNDS]
    compounds_dat = STUDY_DIR / 'compounds_filtered.dat'
    attempted_sets = []
    if compounds_dat.exists():
        try:
            df_exp = pd.read_csv(compounds_dat, sep=r'\s+', comment='#',
                                 header=None, names=['name', 'e_form', 'e_hull'])
            attempted_sets = [_elem_set(n) for n in df_exp['name']]
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(3, 3))

    # All U-only compounds (backdrop, sentinel-filtered for display)
    bg = df_u[df_u['e_above_hull'] < SENTINEL_EHULL]
    ax.scatter(bg['r_DOS'], bg['e_above_hull'], s=5, color='#D0D0D0',
               linewidths=0, label='All Compounds')

    # Synthesized / unsuccessful overlays from attempted list
    succ_x, succ_y = [], []
    unsucc_x, unsucc_y = [], []
    bg_lookup = {_formula_key(r['name']): (float(r['r_DOS']), float(r['e_above_hull']))
                 for _, r in bg.iterrows()}
    for s in attempted_sets:
        key = tuple(sorted(e for e in s if e not in F_BLOCK))
        if key not in bg_lookup:
            continue
        x, y = bg_lookup[key]
        if any(s == ss for ss in synth_sets):
            succ_x.append(x); succ_y.append(y)
        else:
            unsucc_x.append(x); unsucc_y.append(y)
    if unsucc_x:
        ax.scatter(unsucc_x, unsucc_y, s=80, marker='s', facecolors='none',
                   edgecolors='#9467bd', linewidths=1.2, label='Unsuccessful Synthesis')
    if succ_x:
        ax.scatter(succ_x, succ_y, s=100, marker='s', facecolors='#9467bd',
                   edgecolors='#9467bd', linewidths=0.8, label='Successful Synthesis')

    # Top-15 MCTS overlay
    top15_x = GAMMA * top15['r_DOS'].values
    top15_y = top15['e_above_hull'].values
    # Use raw r_DOS on x (gamma=1 so gamma*r_DOS == r_DOS)
    ax.scatter(top15['r_DOS'], top15_y, s=45, color='#5BC0EB', marker='^',
               edgecolors='none', alpha=0.55, label='Top 15 (MCTS)')

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylim(top=1.5)
    ax.set_xlabel(r'$r_{\mathrm{DOS}}$', fontsize=9)
    ax.set_ylabel(r'$E_{\mathrm{Hull}}$ (eV/atom)', fontsize=9)
    ax.tick_params(labelsize=8)

    handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='#D0D0D0',
               markeredgecolor='#D0D0D0', markersize=5, label='All Compounds'),
        Line2D([0],[0], marker='^', linestyle='None', markerfacecolor='#5BC0EB',
               markeredgecolor='none', markersize=7, alpha=0.55, label='Top 15 (MCTS)'),
        Line2D([0],[0], marker='s', linestyle='None', markerfacecolor='none',
               markeredgecolor='#9467bd', markersize=7, label='Unsuccessful Synthesis'),
        Line2D([0],[0], marker='s', linestyle='None', markerfacecolor='#9467bd',
               markeredgecolor='#9467bd', markersize=7, label='Successful Synthesis'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_size_inches(3, 3)
    plt.tight_layout()

    out = figures_dir / 'ehull_vs_rdos_product.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'Saved: {out}')


def plot_rehull_vs_rdos_product(pooled: pd.DataFrame, dos_rewards: dict,
                                 figures_dir: Path) -> None:
    """Same as plot_ehull_vs_rdos_product but y-axis shows r_Ehull instead of E_Hull."""
    mace_csv = REPO_ROOT / 'high_throughput_mace_results.full.csv'
    df_all = pd.read_csv(mace_csv)
    if 'name' not in df_all.columns:
        df_all = df_all.rename(columns={'formula': 'name'})
    df_all['elems'] = df_all['name'].apply(lambda n: set(_parse_elements(n)))
    df_u = df_all[df_all['elems'].apply(
        lambda s: 'U' in s and not (s & (F_BLOCK - {'U'})))].copy()
    df_u['r_DOS'] = df_u['name'].apply(lambda n: _r_dos(n, dos_rewards))

    if 'data_quality' in df_u.columns:
        mask = df_u['data_quality'].isin(['no_mp_data', 'error'])
        df_u.loc[mask, 'e_above_hull'] = 10.0

    df_u['r_ehull'] = df_u['e_above_hull'].apply(ehull_reward)

    top15 = pooled.head(15)

    synth_sets = [_elem_set(n) for n in SYNTHESIZED_COMPOUNDS]
    compounds_dat = STUDY_DIR / 'compounds_filtered.dat'
    attempted_sets = []
    if compounds_dat.exists():
        try:
            df_exp = pd.read_csv(compounds_dat, sep=r'\s+', comment='#',
                                 header=None, names=['name', 'e_form', 'e_hull'])
            attempted_sets = [_elem_set(n) for n in df_exp['name']]
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(3, 3))

    bg = df_u[df_u['e_above_hull'] < SENTINEL_EHULL]
    ax.scatter(bg['r_DOS'], bg['r_ehull'], s=5, color='#D0D0D0',
               linewidths=0, label='All Compounds')

    bg_lookup = {_formula_key(r['name']): (float(r['r_DOS']), float(r['r_ehull']))
                 for _, r in bg.iterrows()}
    succ_x, succ_y = [], []
    unsucc_x, unsucc_y = [], []
    for s in attempted_sets:
        key = tuple(sorted(e for e in s if e not in F_BLOCK))
        if key not in bg_lookup:
            continue
        x, y = bg_lookup[key]
        if any(s == ss for ss in synth_sets):
            succ_x.append(x); succ_y.append(y)
        else:
            unsucc_x.append(x); unsucc_y.append(y)
    if unsucc_x:
        ax.scatter(unsucc_x, unsucc_y, s=80, marker='s', facecolors='none',
                   edgecolors='#9467bd', linewidths=1.2, label='Unsuccessful Synthesis')
    if succ_x:
        ax.scatter(succ_x, succ_y, s=100, marker='s', facecolors='#9467bd',
                   edgecolors='#9467bd', linewidths=0.8, label='Successful Synthesis')

    ax.scatter(top15['r_DOS'], top15['r_ehull'], s=45, color='#5BC0EB', marker='^',
               edgecolors='none', alpha=0.55, label='Top 15 (MCTS)')

    # y=0 is the stability boundary (r_Ehull=0 ↔ E_Hull=0.05 eV/atom)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$r_{\mathrm{DOS}}$', fontsize=9)
    ax.set_ylabel(r'$r_{E_{\mathrm{Hull}}}$', fontsize=9)
    ax.tick_params(labelsize=8)

    handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='#D0D0D0',
               markeredgecolor='#D0D0D0', markersize=5, label='All Compounds'),
        Line2D([0],[0], marker='^', linestyle='None', markerfacecolor='#5BC0EB',
               markeredgecolor='none', markersize=7, alpha=0.55, label='Top 15 (MCTS)'),
        Line2D([0],[0], marker='s', linestyle='None', markerfacecolor='none',
               markeredgecolor='#9467bd', markersize=7, label='Unsuccessful Synthesis'),
        Line2D([0],[0], marker='s', linestyle='None', markerfacecolor='#9467bd',
               markeredgecolor='#9467bd', markersize=7, label='Successful Synthesis'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = figures_dir / 'rehull_vs_rdos_product.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 2: convergence curve (median ± IQR across 5 seeds)
# ---------------------------------------------------------------------------

def plot_convergence_product(figures_dir: Path) -> None:
    """Running-max product reward vs iterations, median ± IQR across 5 seeds."""
    curves = []
    for seed in range(N_SEEDS):
        csv = PRODUCT_DIR / f'seed_{seed}' / 'convergence_history.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        # best_reward is the running-max product reward recorded by the MCTS
        y = df['best_reward'].values.astype(float)
        y = np.maximum.accumulate(y)
        curves.append(y)

    if not curves:
        print('plot_convergence_product: no data; skipping.')
        return

    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        mat[i, :len(c)] = c

    iters = np.arange(max_len)
    with np.errstate(all='ignore'):
        med = np.nanmedian(mat, axis=0)
        p25 = np.nanpercentile(mat, 25, axis=0)
        p75 = np.nanpercentile(mat, 75, axis=0)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.fill_between(iters, p25, p75, alpha=0.25, color='#2166AC', linewidth=0)
    ax.plot(iters, med, color='#2166AC', lw=1.8, label='Product mode (median)')
    for c in curves:
        ax.plot(np.arange(len(c)), c, color='#2166AC', lw=0.5, alpha=0.3)

    # Ignore sentinel value at iter 0 for y-axis
    valid = med[med > -5]
    if valid.size:
        ax.set_ylim(bottom=valid.min() * 0.95)

    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel(r'Best $r_{E_\mathrm{Hull}} \times r_{\mathrm{DOS}}$', fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = figures_dir / 'convergence_product.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Table: top-15 by product reward
# ---------------------------------------------------------------------------

def write_top15_table_product(pooled: pd.DataFrame, global_ranks: dict,
                               figures_dir: Path) -> None:
    tables_dir = figures_dir.parent / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    synth_sets    = [_elem_set(n) for n in SYNTHESIZED_COMPOUNDS]
    attempted_dat = STUDY_DIR / 'compounds_filtered.dat'
    attempted_sets = []
    if attempted_dat.exists():
        try:
            df_exp = pd.read_csv(attempted_dat, sep=r'\s+', comment='#',
                                 header=None, names=['name', 'e_form', 'e_hull'])
            attempted_sets = [_elem_set(n) for n in df_exp['name']]
        except Exception:
            pass

    top15 = pooled.head(15)
    eol = ' \\\\\n'
    tex_path = tables_dir / 'top15_u_only_product.tex'
    n_space = len(global_ranks)

    with open(tex_path, 'w') as f:
        f.write(f'% Top-15 U-only compounds, product-mode study (gamma={GAMMA}, 5 seeds pooled).\n')
        f.write(f'% Global rank within U-only design space by product reward r_Ehull x r_DOS.\n')
        f.write('\\begin{tabular}{rrlrrrrc}\n\\toprule\n')
        f.write('MCTS & True & Compound & $E_{\\mathrm{Hull}}$ & '
                '$r_{E_{\\mathrm{Hull}}}$ & $r_{\\mathrm{DOS}}$ & '
                'Product Reward & Synth' + eol)
        f.write('\\midrule\n')
        for mcts_rank, (_, row) in enumerate(top15.iterrows(), start=1):
            name = row['name']
            key  = _formula_key(name)
            true_rank = global_ranks.get(key, '--')
            es = _elem_set(name)
            if any(es == s for s in synth_sets):
                synth = 'Yes'
            elif any(es == s for s in attempted_sets):
                synth = 'No'
            else:
                synth = '--'
            f.write(
                f"{mcts_rank} & {true_rank} & {_format_name(name)} & "
                f"{row['e_above_hull']:.4f} & {row['r_ehull']:.4f} & "
                f"{row['r_DOS']:.1f} & {row['product']:.2f} & {synth}" + eol)
        f.write('\\bottomrule\n\\end{tabular}\n')

    print(f'Wrote: {tex_path}')


# ---------------------------------------------------------------------------
# Figure: convergence by starting material (product mode)
# ---------------------------------------------------------------------------

def _metal_move_neighbors(z):
    if 22 <= z <= 30:
        if z == 22: return [22, 23, 40]
        if z == 30: return [29, 30, 48]
        return [z - 1, z, z + 1, z + 18]
    if 40 <= z <= 48:
        if z == 40: return [40, 41, 72, 22]
        if z == 48: return [47, 48, 80, 30]
        return [z - 1, z, z + 1, z + 32, z - 18]
    if 72 <= z <= 80:
        if z == 72: return [72, 73, 40]
        if z == 80: return [79, 80, 48]
        return [z - 1, z, z + 1, z - 32]
    return [z]


_GIV_CHAIN = [14, 32, 50, 82]


def _giv_move_neighbors(z):
    idx = _GIV_CHAIN.index(z)
    neighbors = [z]
    if idx > 0: neighbors.append(_GIV_CHAIN[idx - 1])
    if idx < len(_GIV_CHAIN) - 1: neighbors.append(_GIV_CHAIN[idx + 1])
    return neighbors


def _bfs_distances(start, neighbor_fn):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        n = queue.popleft()
        for nb in neighbor_fn(n):
            if nb not in dist:
                dist[nb] = dist[n] + 1
                queue.append(nb)
    return dist


def _edit_distance_product(tm_sym, giv_sym, target_tm, target_giv):
    """MCTS move-graph distance from (tm_sym, giv_sym) to (target_tm, target_giv)."""
    metal_dist = _bfs_distances(atomic_numbers[target_tm], _metal_move_neighbors)
    giv_dist   = _bfs_distances(atomic_numbers[target_giv], _giv_move_neighbors)
    return max(metal_dist.get(atomic_numbers[tm_sym], 99),
               giv_dist.get(atomic_numbers[giv_sym], 99))


def _set_plateau_xlim(ax, curves, buffer_frac=0.25, tol_frac=0.01, ignore_first=1):
    """Trim x-axis so the last-settling curve sits at (1-buffer_frac) width."""
    plateau_starts = []
    n_iter = None
    for arr in curves:
        arr = np.asarray(arr, dtype=float)
        if arr.size <= ignore_first:
            continue
        n_iter = len(arr) if n_iter is None else max(n_iter, len(arr))
        body = arr[ignore_first:]
        final_val = body[-1]
        rng = np.nanmax(body) - np.nanmin(body)
        tol = tol_frac * rng if rng > 0 else 1e-9 * max(1.0, abs(final_val))
        changed = np.where(np.abs(body - final_val) > tol)[0]
        plateau_starts.append((changed[-1] + 1 + ignore_first)
                               if changed.size > 0 else ignore_first)
    if not plateau_starts or n_iter is None:
        return
    plateau_start = min(max(plateau_starts), n_iter - 1)
    xmax = plateau_start + buffer_frac * (n_iter - plateau_start)
    ax.set_xlim(left=0, right=max(xmax, 10))


def plot_convergence_by_starting_material_product(figures_dir: Path) -> None:
    """Convergence curves across 4 starting materials (product mode sweep)."""
    sweep_csv = PRODUCT_DIR / 'starting_material_sweep' / 'convergence_data.csv'
    if not sweep_csv.exists():
        print(f'Skipping convergence_by_starting_material: missing {sweep_csv}')
        return

    # Identify the global best compound (rank 1) for edit-distance labels
    # Under product reward, global best is UTi6Sn6 (Ti + Sn)
    target_tm, target_giv = 'Ti', 'Sn'

    df = pd.read_csv(sweep_csv)
    value_order = ['Cr6Sn6U (d=2)', 'Fe6Sn6U (d=4)', 'Ni6Sn6U (d=6)', 'Pt6Sn6U (d=8)']
    value_order = [v for v in value_order if v in df['value'].unique()]

    def _pctl_agg(s):
        return pd.Series({'mean': s.mean(),
                          'p10': np.percentile(s, 10),
                          'p90': np.percentile(s, 90)})

    x_col = 'n_unique_compounds' if 'n_unique_compounds' in df.columns else 'iteration'

    # Forward-fill each seed's best_reward after early termination so that
    # terminated seeds continue contributing their final best to the mean.
    x_max_global = df.groupby('value')[x_col].max()
    filled_frames = []
    for (val, seed), grp in df.groupby(['value', 'seed']):
        # Keep max best_reward per unique x value (duplicates occur when MCTS
        # revisits already-counted compounds within one iteration)
        grp = (grp.sort_values(x_col)
                  .groupby(x_col, as_index=False)['best_reward'].max()
                  .set_index(x_col))
        full_idx = pd.RangeIndex(0, x_max_global[val] + 1)
        grp = grp.reindex(full_idx)['best_reward'].ffill().bfill()
        filled_frames.append(pd.DataFrame({
            'value': val, 'seed': seed,
            x_col: grp.index, 'best_reward': grp.values,
        }))
    df_filled = pd.concat(filled_frames, ignore_index=True)

    stats = (df_filled.groupby(['value', x_col])['best_reward']
               .apply(_pctl_agg).unstack().reset_index())

    fig, ax = plt.subplots(figsize=(3, 3))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    curves_for_xlim = []
    for color, value in zip(colors, value_order):
        g = stats[stats['value'] == value].sort_values(x_col).reset_index(drop=True)
        curves_for_xlim.append(g['mean'].to_numpy())
        g_plot = g[g[x_col] >= 1]
        formula = value.split(' (')[0]
        # Compute edit distance to target
        parts = re.findall(r'[A-Z][a-z]?', formula)
        tm  = next((e for e in parts if e in TRANSITION_METALS), None)
        giv = next((e for e in parts if e in GROUP_IV), None)
        if tm and giv:
            d = _edit_distance_product(tm, giv, target_tm, target_giv)
            legend_label = f"d={d}"
        else:
            legend_label = value.split('(')[1].rstrip(')') if '(' in value else value
        ax.plot(g_plot[x_col], g_plot['mean'], lw=1.8, color=color,
                label=legend_label)
        ax.fill_between(g_plot[x_col], g_plot['p10'], g_plot['p90'],
                        color=color, alpha=0.2, linewidth=0)

    _set_plateau_xlim(ax, curves_for_xlim)
    x_label = 'Unique compounds evaluated' if x_col == 'n_unique_compounds' else 'Iteration'
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel('Best Composite Score', fontsize=9)
    ax.legend(fontsize=7, title='Edit dist. to global best',
              title_fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = figures_dir / 'convergence_by_starting_material_product.png'
    fig.set_size_inches(3, 3)
    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    figures_dir = PRODUCT_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print('Loading DOS rewards...')
    dos_rewards = _load_dos_rewards()

    print('Computing global U-only ranks (product reward)...')
    global_ranks = compute_global_u_only_ranks_product(dos_rewards)

    print('Pooling discovered compounds across 5 seeds...')
    pooled = load_pooled_compounds(dos_rewards)
    top5 = pooled.head(5)
    print(f"Top-5 by product reward:")
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        gr = global_ranks.get(_formula_key(r['name']), '?')
        print(f"  {i}. {r['name']:<20} ehull={r['e_above_hull']:.4f}  "
              f"r_DOS={r['r_DOS']:.1f}  product={r['product']:.2f}  "
              f"true_rank={gr}")

    plot_ehull_vs_rdos_product(pooled, dos_rewards, figures_dir)
    plot_rehull_vs_rdos_product(pooled, dos_rewards, figures_dir)
    plot_convergence_product(figures_dir)
    write_top15_table_product(pooled, global_ranks, figures_dir)

    # Convergence by starting material (requires the sweep to have been run first)
    plot_convergence_by_starting_material_product(figures_dir)

    # Radial tree (product mode) — run the existing script with --product flag
    print('Generating radial tree (product mode)...')
    _script = Path(__file__).parent / 'create_composite_radial_tree.py'
    _venv_py = Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'
    _ret = subprocess.call([str(_venv_py), str(_script), '--product'])
    if _ret != 0:
        print(f'  Warning: radial tree script exited with code {_ret}')

    print(f'\nAll outputs in: {figures_dir}')


if __name__ == '__main__':
    main()

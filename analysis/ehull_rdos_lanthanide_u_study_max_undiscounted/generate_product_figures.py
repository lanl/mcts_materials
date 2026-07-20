#!/usr/bin/env python3
"""Product-mode figures and table for the lanthanide+U max-undiscounted study.

Reward: r_Ehull × (NORMALIZED_GAMMA × r_DOS)  (multiplicative)
Source: discovery_efficiency_study/product_mode/Tb_Cr_s{0..4}/
        (Tb-Cr = 50th-percentile starting material by edit distance, best avg top-15 rank)

Figures (saved to figures/):
  ehull_vs_rdos_product.png   — 1620-compound design space + top-15 overlay, y=E_Hull
  rehull_vs_rdos_product.png  — same but y=r_Ehull
Table (saved to tables/):
  top15_lanthanide_u_product.tex — top-15 by product reward, Tb-Cr start (5 seeds)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcts_crystal.doscar_utils import DoscarRewardLookup

STUDY_DIR   = Path(__file__).parent
REPO_ROOT   = STUDY_DIR.parents[1]
DISC_DIR    = REPO_ROOT / 'analysis' / 'discovery_efficiency_study'
TB_CR_SEEDS = 5
TM_START    = 'Tb'  # 50th-percentile starting material
TM_TM       = 'Cr'

NORMALIZED_GAMMA = 1.0 / 2516.1664410449775
SENTINEL_EHULL   = 9.9

LANTHANIDES_U = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'U',
}
TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}
GROUP_IV = {'Si', 'Ge', 'Sn', 'Pb'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ehull_reward(e: float) -> float:
    return -np.tanh(120.0 * (float(e) - 0.05))


def _parse_elements(name: str) -> list:
    return re.findall(r'[A-Z][a-z]?', str(name))


def _decompose(name: str):
    elems = _parse_elements(name)
    re_e  = next((e for e in elems if e in LANTHANIDES_U), None)
    tm_e  = next((e for e in elems if e in TRANSITION_METALS), None)
    giv_e = next((e for e in elems if e in GROUP_IV), None)
    return re_e, tm_e, giv_e


def _formula_key(name: str) -> tuple:
    re_e, tm_e, giv_e = _decompose(name)
    if re_e and tm_e and giv_e:
        return (re_e, tm_e, giv_e)
    return (None, None, None)


def _lookup_dos(name: str, dos_rewards: dict) -> float:
    re_e, tm_e, giv_e = _decompose(name)
    if re_e and giv_e and tm_e:
        return float(dos_rewards.get(f"{re_e}-{giv_e}-{tm_e}", 0.0))
    return 0.0


def _format_name(name: str) -> str:
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(name))
    re_e, tm_p, giv_p = None, None, None
    for elem, cnt in matches:
        if not elem: continue
        if elem in LANTHANIDES_U:       re_e  = elem
        elif elem in TRANSITION_METALS: tm_p  = (elem, cnt)
        elif elem in GROUP_IV:          giv_p = (elem, cnt)
    if re_e and tm_p and giv_p:
        return f"{re_e}{tm_p[0]}$_{{{tm_p[1]}}}${giv_p[0]}$_{{{giv_p[1]}}}$"
    return str(name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dos_rewards() -> dict:
    peaks = REPO_ROOT / 'doscar_peaks_data_with_U.csv'
    return DoscarRewardLookup(peaks_file=str(peaks)).rewards_dict


def _apply_no_mp_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """Set e_above_hull=10 for compounds with missing MP data."""
    if 'data_quality' in df.columns:
        mask = df['data_quality'].isin(['no_mp_data', 'error'])
        df.loc[mask, 'e_above_hull'] = 10.0
    return df


def load_design_space(dos_rewards: dict) -> pd.DataFrame:
    """Full 1620-compound lanthanide+U design space (La excluded, no_mp_data penalised).

    e_above_hull_raw preserves the original MACE value for scatter-plot display;
    e_above_hull has no_mp_data compounds set to 10 for reward/ranking computation.
    """
    df = pd.read_csv(REPO_ROOT / 'high_throughput_mace_results.full.csv')
    if 'name' not in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['re'] = df['name'].apply(
        lambda n: next((e for e in _parse_elements(n) if e in LANTHANIDES_U), None))
    df = df[df['re'].notna() & (df['re'] != 'La')].copy()
    df['e_above_hull_raw'] = df['e_above_hull'].copy()  # preserve for plotting
    df = _apply_no_mp_penalty(df)
    df['r_DOS']       = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['g_r_DOS']     = NORMALIZED_GAMMA * df['r_DOS']   # gamma-weighted
    df['r_ehull']     = df['e_above_hull'].apply(ehull_reward)
    df['r_ehull_raw'] = df['e_above_hull_raw'].apply(ehull_reward)  # preserve for plotting
    df['product']     = df['r_ehull'] * df['g_r_DOS']
    return df


def compute_global_ranks(design_space: pd.DataFrame) -> dict:
    """Rank by multiplicative product reward; return {(re, tm, giv): rank}."""
    ranked = design_space.sort_values('product', ascending=False).reset_index(drop=True)
    ranks = {}
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        key = _formula_key(row['name'])
        if key != (None, None, None) and key not in ranks:
            ranks[key] = rank
    return ranks


def load_pooled_tb_cr(dos_rewards: dict) -> pd.DataFrame:
    """Pool all_compounds.csv from Tb_Cr_s{0..4}; deduplicate; compute product reward."""
    frames = []
    for seed in range(TB_CR_SEEDS):
        csv = DISC_DIR / 'product_mode' / f'{TM_START}_{TM_TM}_s{seed}' / 'all_compounds.csv'
        if not csv.exists():
            print(f'  Missing: {csv}')
            continue
        df = pd.read_csv(csv)
        if 'name' not in df.columns:
            df = df.rename(columns={'formula': 'name'})
        frames.append(df[['name', 'e_above_hull']].copy())
    if not frames:
        raise FileNotFoundError('No Tb_Cr all_compounds.csv found.')
    pooled = (pd.concat(frames, ignore_index=True)
                .sort_values('e_above_hull')
                .drop_duplicates(subset=['name'], keep='first')
                .reset_index(drop=True))
    # Exclude sentinel compounds (missing MP data → unreliable reward)
    pooled = pooled[pooled['e_above_hull'] < SENTINEL_EHULL].copy()
    pooled['r_DOS']   = pooled['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    pooled['g_r_DOS'] = NORMALIZED_GAMMA * pooled['r_DOS']
    pooled['r_ehull'] = pooled['e_above_hull'].apply(ehull_reward)
    pooled['product'] = pooled['r_ehull'] * pooled['g_r_DOS']
    return pooled.sort_values('product', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _scatter_base(design_space: pd.DataFrame, pooled: pd.DataFrame,
                  y_col: str, y_label: str, out_path: Path) -> None:
    """Shared logic for ehull-vs-rdos and rehull-vs-rdos plots."""
    fig, ax = plt.subplots(figsize=(3, 3))

    # Background scatter: valid-MP-data compounds only (no_mp_data excluded —
    # their raw MACE e_hull equals e_form with no competing-phase reference,
    # making those values unreliable for display).
    bg = design_space[design_space['data_quality'] == 'valid'] if 'data_quality' in design_space.columns else design_space
    y_bg_vals = bg[y_col]

    # Axis limits from the valid-only range.
    x_pad = (bg['r_DOS'].max() - bg['r_DOS'].min()) * 0.05
    y_pad = (y_bg_vals.max() - y_bg_vals.min()) * 0.05
    xlim = (bg['r_DOS'].min() - x_pad, bg['r_DOS'].max() + x_pad)
    ylim = (y_bg_vals.min() - y_pad, y_bg_vals.max() + y_pad)

    # Gray backdrop: 1512 valid design-space compounds.
    ax.scatter(bg['r_DOS'], y_bg_vals, s=4, color='#D0D0D0', linewidths=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Top-15 MCTS overlay
    top15 = pooled.head(15)
    ax.scatter(top15['r_DOS'], top15[y_col], s=45, color='#5BC0EB',
               marker='^', edgecolors='none', alpha=0.55)

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$r_{\mathrm{DOS}}$', fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(labelsize=8)

    handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='#D0D0D0',
               markeredgecolor='#D0D0D0', markersize=5, label='Design space'),
        Line2D([0],[0], marker='^', linestyle='None', markerfacecolor='#5BC0EB',
               markeredgecolor='none', markersize=7, alpha=0.55, label='Top 15 (MCTS)'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_size_inches(3, 3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_ehull_vs_rdos_product(design_space: pd.DataFrame, pooled: pd.DataFrame,
                                figures_dir: Path) -> None:
    _scatter_base(design_space, pooled,
                  y_col='e_above_hull',
                  y_label=r'$E_{\mathrm{Hull}}$ (eV/atom)',
                  out_path=figures_dir / 'ehull_vs_rdos_product.png')


def plot_rehull_vs_rdos_product(design_space: pd.DataFrame, pooled: pd.DataFrame,
                                 figures_dir: Path) -> None:
    _scatter_base(design_space, pooled,
                  y_col='r_ehull',
                  y_label=r'$r_{E_{\mathrm{Hull}}}$',
                  out_path=figures_dir / 'rehull_vs_rdos_product.png')


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def write_top15_table(pooled: pd.DataFrame, global_ranks: dict,
                      n_design_space: int, tables_dir: Path) -> None:
    """Top-15 LaTeX table for Tb-Cr start (product reward)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    top15 = pooled.head(15)
    eol = ' \\\\\n'
    tex_path = tables_dir / 'top15_lanthanide_u_product.tex'

    with open(tex_path, 'w') as f:
        f.write(f'% Top-15 lanthanide+U compounds, product-mode study '
                f'(gamma=1.0, Tb-Cr start, 5 seeds pooled).\n')
        f.write(f'% Global rank within {n_design_space}-compound design space '
                f'(La excluded, no_mp_data penalised) by product reward.\n')
        f.write('\\begin{tabular}{rrlrrrrr}\n\\toprule\n')
        f.write('MCTS Rank & True Rank & Compound & '
                '$E_{\\mathrm{Hull}}$ (eV/atom) & $r_{E_{\\mathrm{Hull}}}$ & '
                '$r_{\\mathrm{DOS}}$ & Product Reward' + eol)
        f.write('\\midrule\n')
        for mcts_rank, (_, row) in enumerate(top15.iterrows(), start=1):
            key  = _formula_key(row['name'])
            true_rank = global_ranks.get(key, '--')
            product_g1 = row['r_ehull'] * row['r_DOS']
            f.write(
                f"{mcts_rank} & {true_rank} & {_format_name(row['name'])} & "
                f"{row['e_above_hull']:.4f} & {row['r_ehull']:.4f} & "
                f"{row['r_DOS']:.2f} & {product_g1:.2f}" + eol)
        f.write('\\bottomrule\n\\end{tabular}\n')

    print(f'Wrote: {tex_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    figures_dir = STUDY_DIR / 'figures'
    tables_dir  = STUDY_DIR / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print('Loading DOS rewards...')
    dos_rewards = load_dos_rewards()

    print('Building 1620-compound lanthanide+U design space...')
    design_space = load_design_space(dos_rewards)
    n_valid = len(design_space[design_space['e_above_hull'] < SENTINEL_EHULL])
    print(f'  Total (incl. penalised): {len(design_space)}, '
          f'display-filtered (ehull<{SENTINEL_EHULL}): {n_valid}')

    print('Computing global ranks (product reward)...')
    global_ranks = compute_global_ranks(design_space)
    best_key = min(global_ranks, key=global_ranks.get)
    print(f'  Global best: {best_key}  (rank 1)')

    print(f'Pooling Tb-Cr product-mode runs ({TB_CR_SEEDS} seeds)...')
    pooled = load_pooled_tb_cr(dos_rewards)
    top3 = pooled.head(3)
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        gr = global_ranks.get(_formula_key(r['name']), '?')
        print(f'  {i}. {r["name"]:<25} ehull={r["e_above_hull"]:.4f}  '
              f'g_r_DOS={r["g_r_DOS"]:.4f}  product={r["product"]:.4f}  '
              f'true_rank={gr}')

    plot_ehull_vs_rdos_product(design_space, pooled, figures_dir)
    plot_rehull_vs_rdos_product(design_space, pooled, figures_dir)
    write_top15_table(pooled, global_ranks, len(design_space), tables_dir)

    print(f'\nAll outputs in: {figures_dir}  and  {tables_dir}')


if __name__ == '__main__':
    main()

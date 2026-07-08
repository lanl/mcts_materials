#!/usr/bin/env python3
"""Generate figures and the top-15 table for the lanthanide+U max-undiscounted study.

Parallels analysis/ehull_rdos_u_only_study_max_undiscounted/generate_figures.py but
uses the full 1,702-compound lanthanide+U design space instead of 108 U-only compounds.

Figures produced:
  figures/ehull_vs_rdos.png   — scatter of all 1,702 compounds with MCTS top-15 overlay
Tables produced:
  tables/top15_lanthanide_u.tex  — top-15 from Yb_start run with global ranks

MCTS source run: Yb_start/  (CrSn6Yb starting material, rank 511/1702, above-average
start that is not the global optimum; found global ranks 1, 3, 4, 5, 12 in top 15).

Hyperparameters:
  gamma = 1/(max raw r_DOS across the 108 U-only compounds) = 1/2516.1664410449775
  rollout_aggregation = max, rollout_discount = 1.0, f-block-mode = lanthanides_u
  DOS rewards: per-compound lookup (R-GIV-TM key), not cross-lanthanide max.
  True Rank: within the full 1,702-compound lanthanide+U exhaustive design space.
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

NORMALIZED_GAMMA = 1.0 / 2516.1664410449775

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


def ehull_reward(e_hull: float) -> float:
    return -np.tanh(120.0 * (float(e_hull) - 0.05))


def _parse_elements(name: str) -> list:
    return re.findall(r'[A-Z][a-z]?', str(name))


def _decompose(name: str):
    """Return (f_block_elem, tm_elem, giv_elem) or (None, None, None)."""
    elems = _parse_elements(name)
    re_e = next((e for e in elems if e in LANTHANIDES_U), None)
    tm_e = next((e for e in elems if e in TRANSITION_METALS), None)
    giv_e = next((e for e in elems if e in GROUP_IV), None)
    return re_e, tm_e, giv_e


def load_dos_rewards(repo_root: Path) -> dict:
    from mcts_crystal.doscar_utils import DoscarRewardLookup
    peaks_csv = repo_root / 'doscar_peaks_data_with_U.csv'
    return DoscarRewardLookup(peaks_file=str(peaks_csv)).rewards_dict


def _lookup_dos(name: str, dos_rewards: dict) -> float:
    """Per-compound r_DOS using the actual R element (R-GIV-TM key)."""
    re_e, tm_e, giv_e = _decompose(name)
    if re_e and giv_e and tm_e:
        key = f"{re_e}-{giv_e}-{tm_e}"
        if key in dos_rewards:
            return float(dos_rewards[key])
    return 0.0


def load_design_space(repo_root: Path, dos_rewards: dict) -> pd.DataFrame:
    """Load all 1,702 lanthanide+U compounds with composite scores."""
    mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
    df = pd.read_csv(mace_csv)
    if 'name' not in df.columns and 'formula' in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['re_elem'] = df['name'].apply(
        lambda n: next((e for e in _parse_elements(n) if e in LANTHANIDES_U), None))
    df = df[df['re_elem'].notna()].copy()
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
    df['weighted_r_DOS'] = NORMALIZED_GAMMA * df['r_DOS']
    df['composite_score'] = df['ehull_reward'] + df['weighted_r_DOS']
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['global_rank'] = range(1, len(df) + 1)
    return df


def compute_global_ranks(design_space_df: pd.DataFrame) -> dict:
    """Return {(re, tm, giv) -> global_rank} from the pre-ranked design space."""
    ranks = {}
    for _, row in design_space_df.iterrows():
        key = _decompose(row['name'])
        if key != (None, None, None) and key not in ranks:
            ranks[key] = int(row['global_rank'])
    return ranks


def compute_composite(df: pd.DataFrame, dos_rewards: dict) -> pd.DataFrame:
    df = df.copy()
    if 'name' not in df.columns and 'formula' in df.columns:
        df['name'] = df['formula']
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
    df['weighted_r_DOS'] = NORMALIZED_GAMMA * df['r_DOS']
    df['composite_score'] = df['ehull_reward'] + df['weighted_r_DOS']
    return df.sort_values('composite_score', ascending=False).reset_index(drop=True)


def format_name(name: str) -> str:
    """Format RM6X6 formula as RTM$_6$GIV$_6$ for LaTeX."""
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(name))
    re_elem, tm_part, giv_part = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in LANTHANIDES_U:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_part = (elem, count)
        elif elem in GROUP_IV:
            giv_part = (elem, count)
    if re_elem and tm_part and giv_part:
        tm_e, tm_n = tm_part
        giv_e, giv_n = giv_part
        return f"{re_elem}{tm_e}$_{{{tm_n}}}${giv_e}$_{{{giv_n}}}$"
    return str(name)


def plot_ehull_vs_rdos(design_space_df: pd.DataFrame, mcts_run_dir: Path,
                       out_dir: Path) -> None:
    """Scatter of all 1,702 lanthanide+U compounds with MCTS top-15 overlay.

    Background: every compound in the lanthanide+U design space (gray dots).
    Overlay: top 15 compounds discovered by the MCTS run in mcts_run_dir (blue triangles).
    No experimental synthesis overlay — no U-only experimental data is applicable here.
    """
    fig, ax = plt.subplots(figsize=(3, 3))

    x_all = design_space_df['weighted_r_DOS'].astype(float)
    y_all = design_space_df['e_above_hull'].astype(float)
    ax.scatter(x_all, y_all, s=4, color='#D0D0D0', linewidths=0,
               label=f'All Compounds (n={len(design_space_df):,})')

    # Build lookup: decomposed key -> (weighted_r_DOS, e_above_hull) from full space
    space_lookup = {}
    for _, row in design_space_df.iterrows():
        key = _decompose(row['name'])
        if key != (None, None, None):
            space_lookup[key] = (float(row['weighted_r_DOS']), float(row['e_above_hull']))

    # Top-15 MCTS overlay from the chosen run
    composite_csv = mcts_run_dir / 'all_compounds_by_composite_score.csv'
    if composite_csv.exists():
        df_top = pd.read_csv(composite_csv).head(15)
        xs, ys = [], []
        for _, row in df_top.iterrows():
            name = row.get('name', row.get('formula', ''))
            key = _decompose(str(name))
            if key in space_lookup:
                xi, yi = space_lookup[key]
                xs.append(xi); ys.append(yi)
        if xs:
            ax.scatter(xs, ys, s=45, color='#5BC0EB', marker='^',
                       edgecolors='none', alpha=0.55, label='Top 15 (MCTS)')

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylim(top=2)
    ax.set_xlabel(r"$\alpha_{\mathrm{DOS}} \cdot r_{\mathrm{DOS}}$", fontsize=9)
    ax.set_ylabel(r"$E_{\mathrm{Hull}}$ (eV/atom)", fontsize=9)
    ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='#D0D0D0', markeredgecolor='#D0D0D0',
               markersize=5, label='All Compounds'),
        Line2D([0], [0], marker='^', linestyle='None',
               markerfacecolor='#5BC0EB', markeredgecolor='none',
               markersize=7, alpha=0.55, label='Top 15 (MCTS)'),
    ]
    ax.legend(handles=legend_handles, fontsize=7)
    plt.tight_layout()

    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / 'ehull_vs_rdos.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def write_top15_table(df_sorted: pd.DataFrame, global_ranks: dict,
                      out_dir: Path, n_space: int) -> None:
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    top15 = df_sorted.head(15).copy()
    eol = ' \\\\\n'
    tex_path = tables_dir / 'top15_lanthanide_u.tex'

    with open(tex_path, 'w') as f:
        f.write(f'% Top 15 compounds (lanthanide+U study, Yb start). gamma={NORMALIZED_GAMMA:g}.\n')
        f.write(f'% True Rank = rank within the full {n_space}-compound lanthanide+U design space.\n')
        f.write('\\begin{tabular}{rrlrrrr}\n')
        f.write('\\toprule\n')
        f.write('MCTS Rank & True Rank & Compound & $E_{\\mathrm{Hull}}$ (eV/atom) & '
                '$r_{E_{\\mathrm{Hull}}}$ & $\\alpha_{\\mathrm{DOS}} \\cdot r_{\\mathrm{DOS}}$ '
                '& Composite' + eol)
        f.write('\\midrule\n')
        for mcts_rank, (_, row) in enumerate(top15.iterrows(), start=1):
            name = row.get('name', row.get('formula', ''))
            key = _decompose(str(name))
            true_rank = global_ranks.get(key, '--')
            ehull = float(row['e_above_hull']) if pd.notna(row.get('e_above_hull')) else float('nan')
            ehull_r = float(row.get('ehull_reward', ehull_reward(ehull)))
            rdos = float(row.get('weighted_r_DOS', 0.0)) if pd.notna(row.get('weighted_r_DOS')) else 0.0
            comp = float(row.get('composite_score', 0.0)) if pd.notna(row.get('composite_score')) else 0.0
            f.write(f"{mcts_rank} & {true_rank} & {format_name(name)} & "
                    f"{ehull:.4f} & {ehull_r:.4f} & {rdos:.4f} & {comp:.4f}" + eol)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f'Wrote LaTeX table: {tex_path}')


def main():
    study_dir = Path(__file__).parent
    repo_root = study_dir.parents[1]

    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print('Loading DOS rewards...')
    dos_rewards = load_dos_rewards(repo_root)

    print('Loading full 1,702-compound lanthanide+U design space...')
    design_space = load_design_space(repo_root, dos_rewards)
    print(f'  Design space: {len(design_space)} compounds')
    print(f'  Global best: {design_space.iloc[0]["name"]} (composite={design_space.iloc[0]["composite_score"]:.4f})')

    global_ranks = compute_global_ranks(design_space)

    # Source run: Yb_start_step3 (move_step=3) preferred; falls back to Yb_start (move_step=1)
    yb_run_dir = study_dir / 'Yb_start_step3'
    if not (yb_run_dir / 'all_compounds.csv').exists():
        yb_run_dir = study_dir / 'Yb_start'
    if not (yb_run_dir / 'all_compounds.csv').exists():
        print(f'ERROR: neither Yb_start_step3 nor Yb_start found')
        sys.exit(1)
    print(f'Using run: {yb_run_dir.name}')

    print('Computing composite scores for Yb_start run...')
    df_yb = pd.read_csv(yb_run_dir / 'all_compounds.csv')
    df_yb_sorted = compute_composite(df_yb, dos_rewards)
    df_yb_sorted.to_csv(yb_run_dir / 'all_compounds_by_composite_score.csv', index=False)
    print(f'  Yb_start: {len(df_yb_sorted)} unique compounds visited')
    print(f'  Top 5 global ranks: {[global_ranks.get(_decompose(r["name"]), "?") for _, r in df_yb_sorted.head(5).iterrows()]}')

    print('Generating ehull_vs_rdos figure...')
    plot_ehull_vs_rdos(design_space, yb_run_dir, study_dir)

    print('Writing top-15 table...')
    write_top15_table(df_yb_sorted, global_ranks, study_dir, len(design_space))

    # Also save full pooled CSV for reference
    start_dirs = sorted(p for p in study_dir.iterdir()
                        if p.is_dir() and p.name.endswith('_start'))
    frames = [df_yb_sorted]
    for d in start_dirs:
        if d == yb_run_dir:
            continue
        csv = d / 'all_compounds.csv'
        if csv.exists():
            sub = compute_composite(pd.read_csv(csv), dos_rewards)
            frames.append(sub)
    if (study_dir / 'all_compounds.csv').exists():
        frames.append(compute_composite(pd.read_csv(study_dir / 'all_compounds.csv'), dos_rewards))

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.drop_duplicates(subset='name').sort_values('composite_score', ascending=False)
    pooled.to_csv(study_dir / 'all_compounds_by_composite_score.csv', index=False)
    pooled.head(15).to_csv(study_dir / 'top15_compounds_by_composite_score.csv', index=False)
    print(f'Pooled CSV: {len(pooled)} unique compounds across all runs')


if __name__ == '__main__':
    main()

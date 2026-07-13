#!/usr/bin/env python3
"""Discovery efficiency study: MCTS vs. random search.

Generates:
  figures/discovery_curve_edit_distance.png
      30 MCTS curves (coloured by edit distance to global best) +
      random baseline (30 independent shuffles of the full design space).
  tables/top15_best_run.tex
      Top-15 table from the single best MCTS run.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mcts_crystal'))

NORMALIZED_GAMMA = 1.0 / 2516.1664410449775

LANTHANIDES_U = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'U',
]
LANTHANIDES_U_SET = set(LANTHANIDES_U)

TM_3D_SEQ = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}
GROUP_IV = {'Si', 'Ge', 'Sn', 'Pb'}

# Global-best compound: EuCu6Sn6 (composite ≈ 7.40)
GLOBAL_BEST_LAN = 'Eu'
GLOBAL_BEST_TM = 'Cu'
GLOBAL_BEST_GIV = 'Sn'

# X-axis clip (unique compounds evaluated)
X_MAX = 200
# Random baseline replicates
N_RANDOM_REPS = 30

# ---------------------------------------------------------------------------
# All 30 MCTS run definitions: (elem, tm, subdir_in_study_dir)
# subdir=None → run lives directly in study_dir/{elem}_start
# ---------------------------------------------------------------------------
MCTS_RUNS = [
    # Cu-TM starts (edit dist 0–7)
    ('Eu', 'Cu', 'cu_tm'), ('Sm', 'Cu', 'cu_tm'), ('Gd', 'Cu', 'cu_tm'),
    ('Tb', 'Cu', 'cu_tm'), ('Nd', 'Cu', 'cu_tm'), ('Pr', 'Cu', 'cu_tm'),
    ('Ce', 'Cu', 'cu_tm'), ('Er', 'Cu', 'cu_tm'), ('Yb', 'Cu', 'cu_tm'),
    # Fe-TM starts (edit dist 3–10)
    ('Eu', 'Fe', 'fe_tm'), ('Sm', 'Fe', 'fe_tm'), ('Gd', 'Fe', 'fe_tm'),
    ('Tb', 'Fe', 'fe_tm'), ('Nd', 'Fe', 'fe_tm'), ('Pr', 'Fe', 'fe_tm'),
    ('Ce', 'Fe', 'fe_tm'), ('Er', 'Fe', 'fe_tm'), ('Yb', 'Fe', 'fe_tm'),
    # Cr-TM starts (edit dist 5–12)
    ('Eu', 'Cr', None),    ('Sm', 'Cr', None),    ('Gd', 'Cr', None),
    ('Tb', 'Cr', None),    ('Nd', 'Cr', None),    ('Pr', 'Cr', None),
    ('Ce', 'Cr', None),    ('Er', 'Cr', None),    ('Yb', 'Cr', None),
    # Gap-filling starts (edit dist 11, 13, 14)
    ('Yb', 'Mn', 'mn_tm'), ('La', 'Ti', 'ti_tm'), ('Yb', 'Ti', 'ti_tm'),
]


def _parse_elements(name):
    return re.findall(r'[A-Z][a-z]?', str(name))


def _decompose(name):
    elems = _parse_elements(name)
    re_e = next((e for e in elems if e in LANTHANIDES_U_SET), None)
    tm_e = next((e for e in elems if e in TRANSITION_METALS), None)
    giv_e = next((e for e in elems if e in GROUP_IV), None)
    return re_e, tm_e, giv_e


def ehull_reward(x):
    return -np.tanh(120.0 * (float(x) - 0.05))


def _lookup_dos(name, dos_rewards):
    re_e, tm_e, giv_e = _decompose(name)
    if re_e and giv_e and tm_e:
        key = f"{re_e}-{giv_e}-{tm_e}"
        return float(dos_rewards.get(key, 0.0))
    return 0.0


def edit_distance(elem, tm):
    """Minimum moves from {elem}{tm}6Sn6 to EuCu6Sn6 (group-IV always Sn)."""
    lan_dist = abs(LANTHANIDES_U.index(elem) - LANTHANIDES_U.index(GLOBAL_BEST_LAN))
    if tm in TM_3D_SEQ and GLOBAL_BEST_TM in TM_3D_SEQ:
        tm_dist = abs(TM_3D_SEQ.index(tm) - TM_3D_SEQ.index(GLOBAL_BEST_TM))
    else:
        tm_dist = 99  # unknown / non-3d
    return lan_dist + tm_dist


def load_run(study_dir, elem, tm, subdir, dos_rewards):
    """Load one MCTS run. Returns (compounds_df, conv_df) or (None, None)."""
    base = study_dir / subdir if subdir else study_dir
    run_dir = base / f'{elem}_start'
    all_csv = run_dir / 'all_compounds.csv'
    conv_csv = run_dir / 'convergence_history.csv'
    if not all_csv.exists() or not conv_csv.exists():
        return None, None

    df = pd.read_csv(all_csv)
    if 'name' not in df.columns and 'formula' in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['composite'] = df['e_above_hull'].apply(ehull_reward) + NORMALIZED_GAMMA * df['r_DOS']

    conv_df = pd.read_csv(conv_csv)
    return df, conv_df


def load_design_space(repo_root, dos_rewards):
    """Full 1702-compound lanthanide+U design space."""
    mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
    df = pd.read_csv(mace_csv)
    if 'name' not in df.columns and 'formula' in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['re'] = df['name'].apply(
        lambda n: next((e for e in _parse_elements(n) if e in LANTHANIDES_U_SET), None))
    df = df[df['re'].notna()].copy()
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['composite'] = df['e_above_hull'].apply(ehull_reward) + NORMALIZED_GAMMA * df['r_DOS']
    return df.reset_index(drop=True)


def mcts_curve_from_conv(conv_df):
    """(x, y) curve: unique compounds evaluated vs. best reward (monotone)."""
    valid = conv_df[conv_df['best_reward'] > -5.0].copy()
    if valid.empty:
        valid = conv_df.iloc[1:].copy()
    x = valid['n_unique_compounds'].values.astype(float)
    y = valid['best_reward'].values.astype(float)
    y = np.maximum.accumulate(y)
    return x, y


def interpolate_curve(x, y, grid):
    """Step-interpolate (x, y) onto grid; NaN before first observation."""
    result = np.full(len(grid), np.nan)
    for i, gx in enumerate(grid):
        past = y[x <= gx]
        if len(past) > 0:
            result[i] = past[-1]
    if not np.isnan(result).all():
        first = np.where(~np.isnan(result))[0]
        if len(first):
            result[:first[0]] = result[first[0]]
    return result


def random_curves_independent(design_space_df, n_reps, base_seed, max_evals):
    """30 independent random shuffles of the full design space (no fixed start).

    Each trial draws a uniformly random ordering of all 1702 compounds and
    tracks the running-maximum composite score.
    """
    rng = np.random.default_rng(base_seed)
    scores = design_space_df['composite'].values.copy()
    N = len(scores)
    curves = np.empty((n_reps, max_evals))
    for r in range(n_reps):
        perm = np.arange(N)
        rng.shuffle(perm)
        curves[r] = np.maximum.accumulate(scores[perm[:max_evals]])
    return curves


def format_name(name):
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(name))
    re_elem, tm_part, giv_part = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in LANTHANIDES_U_SET:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_part = (elem, count)
        elif elem in GROUP_IV:
            giv_part = (elem, count)
    if re_elem and tm_part and giv_part:
        return f"{re_elem}{tm_part[0]}$_{{{tm_part[1]}}}${giv_part[0]}$_{{{giv_part[1]}}}$"
    return str(name)


def compute_global_ranks(repo_root, dos_rewards):
    mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
    if not mace_csv.exists():
        return {}
    df = pd.read_csv(mace_csv)
    if 'name' not in df.columns and 'formula' in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['re'] = df['name'].apply(
        lambda n: next((e for e in _parse_elements(n) if e in LANTHANIDES_U_SET), None))
    df = df[df['re'].notna()].copy()
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['composite'] = df['e_above_hull'].apply(ehull_reward) + NORMALIZED_GAMMA * df['r_DOS']
    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    ranks = {}
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        key = _decompose(row['name'])
        if key != (None, None, None):
            ranks[key] = rank
    return ranks


def write_top15_table(df_sorted, global_ranks, dos_rewards, out_dir, run_label):
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    top15 = df_sorted.head(15).copy()
    eol = ' \\\\\n'
    tex_path = tables_dir / 'top15_best_run.tex'
    with open(tex_path, 'w') as f:
        f.write(f'% Top 15 compounds from the best single MCTS run ({run_label}).\n')
        f.write(f'% gamma={NORMALIZED_GAMMA:g}. True Rank = rank within full '
                f'lanthanide+U design space by composite score.\n')
        f.write('\\begin{tabular}{rrlrrrr}\n\\toprule\n')
        f.write('MCTS Rank & True Rank & Name & $E_{\\mathrm{Hull}}$ (eV/atom) & '
                '$r_{E_{\\mathrm{Hull}}}$ & $\\alpha_{\\mathrm{DOS}} \\cdot r_{\\mathrm{DOS}}$ '
                '& Composite' + eol)
        f.write('\\midrule\n')
        for mcts_rank, (_, row) in enumerate(top15.iterrows(), start=1):
            name = str(row.get('name', row.get('formula', '')))
            true_rank = global_ranks.get(_decompose(name), '--')
            ehull = float(row['e_above_hull'])
            ehull_r = ehull_reward(ehull)
            rdos_w = NORMALIZED_GAMMA * float(row.get('r_DOS', _lookup_dos(name, dos_rewards)))
            comp = ehull_r + rdos_w
            f.write(f"{mcts_rank} & {true_rank} & {format_name(name)} & "
                    f"{ehull:.4f} & {ehull_r:.4f} & {rdos_w:.4f} & {comp:.4f}" + eol)
        f.write('\\bottomrule\n\\end{tabular}\n')
    print(f'Wrote table: {tex_path}')


# ---------------------------------------------------------------------------
# Extended-mode figure (lanthanides_u_extended + N seeds per starting material)
# ---------------------------------------------------------------------------

# Same 30 starting materials; include La-Ti so the distant-start data appears
EXTENDED_MCTS_RUNS = [
    ('Eu', 'Cu'), ('Sm', 'Cu'), ('Gd', 'Cu'), ('Tb', 'Cu'), ('Nd', 'Cu'),
    ('Pr', 'Cu'), ('Ce', 'Cu'), ('Er', 'Cu'), ('Yb', 'Cu'),
    ('Eu', 'Fe'), ('Sm', 'Fe'), ('Gd', 'Fe'), ('Tb', 'Fe'), ('Nd', 'Fe'),
    ('Pr', 'Fe'), ('Ce', 'Fe'), ('Er', 'Fe'), ('Yb', 'Fe'),
    ('Eu', 'Cr'), ('Sm', 'Cr'), ('Gd', 'Cr'), ('Tb', 'Cr'), ('Nd', 'Cr'),
    ('Pr', 'Cr'), ('Ce', 'Cr'), ('Er', 'Cr'), ('Yb', 'Cr'),
    ('Yb', 'Mn'), ('La', 'Ti'), ('Yb', 'Ti'),
]
N_EXTENDED_SEEDS = 5


def load_extended_seeds(study_dir, elem, tm, n_seeds, dos_rewards):
    """Load up to n_seeds convergence curves for one (elem, tm) starting material.
    Returns list of (x_array, y_array) tuples (running-max best composite vs.
    unique compounds evaluated).
    """
    ext_dir = study_dir / 'extended_mode'
    curves = []
    for seed in range(n_seeds):
        run_dir = ext_dir / f'{elem}_{tm}_s{seed}'
        conv_csv = run_dir / 'convergence_history.csv'
        if not conv_csv.exists():
            continue
        conv_df = pd.read_csv(conv_csv)
        x_m, y_m = mcts_curve_from_conv(conv_df)
        curves.append((x_m, y_m))
    return curves


def plot_discovery_curve_extended(study_dir, design_space, dos_rewards):
    """Generate discovery_curve_edit_distance_extended.png.

    Same layout as discovery_curve_edit_distance.png but:
      - MCTS runs use lanthanides_u_extended + 5 seeds per starting material
      - Each starting material is shown as a median line + IQR band coloured by
        edit distance (gives a 'whisker' view instead of spaghetti)
    """
    x_grid = np.arange(1, X_MAX + 1)

    # --- Load runs, aggregate across seeds ---
    run_records = []
    for elem, tm in EXTENDED_MCTS_RUNS:
        seed_curves = load_extended_seeds(study_dir, elem, tm, N_EXTENDED_SEEDS, dos_rewards)
        if not seed_curves:
            continue
        dist = edit_distance(elem, tm)
        # Interpolate each seed onto x_grid and stack
        interps = []
        for x_m, y_m in seed_curves:
            interp = interpolate_curve(x_m, y_m, x_grid)
            # clip beyond the max unique compounds this seed evaluated
            n = int(x_m.max()) if len(x_m) else 0
            interp[n:] = np.nan
            interps.append(interp)
        mat = np.array(interps)  # (n_seeds, X_MAX)
        with np.errstate(all='ignore'):
            med = np.nanmedian(mat, axis=0)
            p25 = np.nanpercentile(mat, 25, axis=0)
            p75 = np.nanpercentile(mat, 75, axis=0)
        best = float(np.nanmax(mat))
        n_seeds_loaded = len(seed_curves)
        run_records.append({
            'label': f'{elem}-{tm}',
            'elem': elem, 'tm': tm,
            'dist': dist,
            'med': med, 'p25': p25, 'p75': p75,
            'n_seeds': n_seeds_loaded,
            'best': best,
        })

    if not run_records:
        print('plot_discovery_curve_extended: no extended-mode data found; skipping.')
        return

    n_runs = len(run_records)
    print(f'Extended mode: loaded {n_runs} starting materials '
          f'(total seeds: {sum(r["n_seeds"] for r in run_records)}).')

    # --- Random baseline (same as original figure) ---
    rand_curves = random_curves_independent(design_space, N_RANDOM_REPS,
                                            base_seed=0, max_evals=X_MAX)
    rand_p50 = np.percentile(rand_curves, 50, axis=0)
    rand_p25 = np.percentile(rand_curves, 25, axis=0)
    rand_p75 = np.percentile(rand_curves, 75, axis=0)

    # --- Figure ---
    max_dist = max(r['dist'] for r in run_records)
    cmap = plt.colormaps['RdYlGn_r']

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20,
                    label='Random — full space (median ± IQR)')
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.8, linestyle='--',
            label='_nolegend_')

    # Draw farthest starts first (closest on top)
    for rec in sorted(run_records, key=lambda r: -r['dist']):
        c = cmap(rec['dist'] / max(max_dist, 1))
        valid = ~np.isnan(rec['med'])
        if not valid.any():
            continue
        ax.fill_between(x_grid[valid], rec['p25'][valid], rec['p75'][valid],
                        color=c, alpha=0.18, linewidth=0)
        ax.plot(x_grid[valid], rec['med'][valid], color=c, lw=1.0, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Edit distance to\nglobal best', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Unique compounds evaluated', fontsize=10)
    ax.set_ylabel('Best composite score', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    note = (f'lanthanides_u_extended, {N_EXTENDED_SEEDS} seeds/start')
    ax.set_title(note, fontsize=7, color='#555555')

    fig.tight_layout()
    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'discovery_curve_edit_distance_extended.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')


def main():
    study_dir = Path(__file__).parent
    repo_root = study_dir.parents[1]

    from mcts_crystal.doscar_utils import DoscarRewardLookup
    dos_rewards = DoscarRewardLookup(
        peaks_file=str(repo_root / 'doscar_peaks_data_with_U.csv')).rewards_dict

    global_ranks = compute_global_ranks(repo_root, dos_rewards)
    design_space = load_design_space(repo_root, dos_rewards)
    print(f'Full design space: {len(design_space)} lanthanide+U compounds.')

    x_grid = np.arange(1, X_MAX + 1)

    # --- Load MCTS runs ---
    run_records = []   # list of dicts
    for elem, tm, subdir in MCTS_RUNS:
        compounds_df, conv_df = load_run(study_dir, elem, tm, subdir, dos_rewards)
        if compounds_df is None:
            print(f'  Warning: missing {elem}-{tm} ({subdir}) — skipping.')
            continue
        dist = edit_distance(elem, tm)
        x_m, y_m = mcts_curve_from_conv(conv_df)
        n = min(len(compounds_df), X_MAX)
        interp = interpolate_curve(x_m, y_m, x_grid)
        interp[n:] = np.nan
        best = float(np.nanmax(y_m)) if len(y_m) else -np.inf
        run_records.append({
            'label': f'{elem}-{tm}',
            'elem': elem, 'tm': tm,
            'dist': dist,
            'curve': interp,
            'compounds_df': compounds_df,
            'best': best,
        })

    n_runs = len(run_records)
    print(f'Loaded {n_runs} / {len(MCTS_RUNS)} MCTS runs.')
    print(f'Edit distances: {sorted(r["dist"] for r in run_records)}')

    # --- Random baseline: 30 independent shuffles, no fixed start ---
    rand_curves = random_curves_independent(design_space, N_RANDOM_REPS,
                                            base_seed=0, max_evals=X_MAX)
    rand_p50 = np.percentile(rand_curves, 50, axis=0)
    rand_p25 = np.percentile(rand_curves, 25, axis=0)
    rand_p75 = np.percentile(rand_curves, 75, axis=0)

    # --- U-only MCTS curves (from existing u_only study runs) ---
    analysis_dir = study_dir.parent
    u_study_dirs = [
        analysis_dir / 'ehull_rdos_u_only_study_max_undiscounted',
        analysis_dir / 'ehull_rdos_u_only_study_max_undiscounted' / 'd5_start_run',
        analysis_dir / 'ehull_rdos_u_only_study_final',
        analysis_dir / 'ehull_rdos_u_only_study_normalized',
        analysis_dir / 'ehull_rdos_u_only_study_normalized' / 'd5_start_run',
    ]
    u_space = design_space[design_space['re'] == 'U'].copy().reset_index(drop=True)
    N_u = len(u_space)
    u_x_grid = np.arange(1, N_u + 1)
    u_mcts_curves = []
    for d in u_study_dirs:
        csv = d / 'convergence_history.csv'
        if not csv.exists():
            continue
        conv = pd.read_csv(csv)
        x_m, y_m = mcts_curve_from_conv(conv)
        n = int(conv['n_unique_compounds'].max())
        interp = interpolate_curve(x_m, y_m, u_x_grid)
        interp[n:] = np.nan
        u_mcts_curves.append(interp)
    u_mcts_matrix = np.array(u_mcts_curves)
    with np.errstate(all='ignore'):
        u_mcts_p50 = np.nanpercentile(u_mcts_matrix, 50, axis=0)
        u_mcts_p25 = np.nanpercentile(u_mcts_matrix, 25, axis=0)
        u_mcts_p75 = np.nanpercentile(u_mcts_matrix, 75, axis=0)
    print(f'U-only MCTS: {len(u_mcts_curves)} runs loaded, best={u_space["composite"].max():.4f}')

    # --- Global-best run for table ---
    best_rec = max(run_records, key=lambda r: r['best'])
    print(f'\nGlobal best: {best_rec["label"]} (composite={best_rec["best"]:.4f})')
    best_df_sorted = (best_rec['compounds_df']
                      .sort_values('composite', ascending=False)
                      .reset_index(drop=True))
    write_top15_table(best_df_sorted, global_ranks, dos_rewards, study_dir,
                      best_rec['label'])

    # --- Figure: edit-distance coloured MCTS curves + random band ---
    max_dist = max(r['dist'] for r in run_records)
    cmap = plt.colormaps['RdYlGn_r']   # red=far, green=close

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Full-space random band (background, drawn first)
    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20,
                    label='Random — full space (median ± IQR)')
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.8, linestyle='--',
            label='_nolegend_')

    # MCTS curves, coloured by edit distance (farthest drawn first, closest on top)
    run_records_sorted = sorted(run_records, key=lambda r: -r['dist'])
    for rec in run_records_sorted:
        c = cmap(rec['dist'] / max(max_dist, 1))
        valid = ~np.isnan(rec['curve'])
        if valid.any():
            ax.plot(x_grid[valid], rec['curve'][valid],
                    color=c, lw=0.9, alpha=0.75)

    # U-only MCTS curve (median ± IQR) — distinct style
    u_valid = ~np.isnan(u_mcts_p50)
    ax.fill_between(u_x_grid[u_valid], u_mcts_p25[u_valid], u_mcts_p75[u_valid],
                    color='#7b3294', alpha=0.22)
    ax.plot(u_x_grid[u_valid], u_mcts_p50[u_valid],
            color='#7b3294', lw=2.0, linestyle='-.',
            label='MCTS — U only (median ± IQR)')

    # Colourbar for edit distance
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Edit distance to\nglobal best', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Unique compounds evaluated', fontsize=10)
    ax.set_ylabel('Best composite score', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'discovery_curve_edit_distance.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')

    # --- Summary ---
    print(f'\n--- {n_runs} runs summary (sorted by edit distance) ---')
    for rec in sorted(run_records, key=lambda r: r['dist']):
        print(f"  {rec['label']:12s}  dist={rec['dist']:2d}  "
              f"best={rec['best']:.4f}  n={len(rec['compounds_df'])}")

    # --- Extended-mode figure (generated if data exists) ---
    plot_discovery_curve_extended(study_dir, design_space, dos_rewards)


if __name__ == '__main__':
    main()

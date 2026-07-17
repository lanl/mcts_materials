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


def _apply_no_mp_penalty(df):
    """Mirror energy_calculator: set e_above_hull=10.0 for no_mp_data/error rows."""
    if 'data_quality' in df.columns:
        mask = df['data_quality'].isin(['no_mp_data', 'error'])
        df.loc[mask, 'e_above_hull'] = 10.0
    return df


def load_design_space(repo_root, dos_rewards):
    """Lanthanide+U design space excluding La. no_mp_data compounds kept but penalised
    (e_above_hull set to 10.0 → r_Ehull ≈ -1) to match energy_calculator behaviour."""
    mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
    df = pd.read_csv(mace_csv)
    if 'name' not in df.columns and 'formula' in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['re'] = df['name'].apply(
        lambda n: next((e for e in _parse_elements(n) if e in LANTHANIDES_U_SET), None))
    df = df[df['re'].notna() & (df['re'] != 'La')].copy()
    df = _apply_no_mp_penalty(df)
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
    """Rank all compounds by multiplicative reward: r_Ehull × (gamma × r_DOS).

    La excluded (not in expansion). no_mp_data compounds kept but penalised to
    e_above_hull=10 → r_Ehull≈-1 → negative product → naturally rank last.
    """
    mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
    if not mace_csv.exists():
        return {}
    df = pd.read_csv(mace_csv)
    if 'name' not in df.columns and 'formula' in df.columns:
        df = df.rename(columns={'formula': 'name'})
    df['re'] = df['name'].apply(
        lambda n: next((e for e in _parse_elements(n) if e in LANTHANIDES_U_SET), None))
    df = df[df['re'].notna() & (df['re'] != 'La')].copy()
    df = _apply_no_mp_penalty(df)
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['composite'] = (df['e_above_hull'].apply(ehull_reward)
                       * (NORMALIZED_GAMMA * df['r_DOS']))
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
    cmap = plt.colormaps['RdBu_r']

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20,
                    label='Random')
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
    ax.set_xlabel('Iterations', fontsize=10)
    ax.set_ylabel('Best composite score', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    fig.tight_layout()
    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'discovery_curve_edit_distance_extended.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')


def load_product_seeds(study_dir, elem, tm, n_seeds, subdir='product_mode_g1'):
    """Load up to n_seeds convergence curves from a product-mode run directory.

    Returns list of (x_array, y_array) tuples using best_reward (product metric).
    """
    prod_dir = study_dir / subdir
    curves = []
    for seed in range(n_seeds):
        run_dir = prod_dir / f'{elem}_{tm}_s{seed}'
        conv_csv = run_dir / 'convergence_history.csv'
        if not conv_csv.exists():
            continue
        conv_df = pd.read_csv(conv_csv)
        x_m, y_m = mcts_curve_from_conv(conv_df)
        curves.append((x_m, y_m))
    return curves


def plot_discovery_curve_product(study_dir, design_space, dos_rewards,
                                  subdir='product_mode_g1', suffix='ms3'):
    """Generate discovery_curve_edit_distance_product_{suffix}.png.

    Same layout as the extended figure but using the multiplicative reward
    (r_Ehull × r_DOS).  Random baseline is recomputed with the product
    metric so the comparison is on the same scale.
    """
    prod_dir = study_dir / subdir
    if not prod_dir.exists():
        print(f'plot_discovery_curve_product: {subdir}/ not found; skipping.')
        return

    x_grid = np.arange(1, X_MAX + 1)

    # --- Load product-mode runs, aggregate across seeds ---
    run_records = []
    for elem, tm in EXTENDED_MCTS_RUNS:
        seed_curves = load_product_seeds(study_dir, elem, tm, N_EXTENDED_SEEDS, subdir)
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
        best = float(np.nanmax(mat))
        run_records.append({
            'label': f'{elem}-{tm}',
            'elem': elem, 'tm': tm,
            'dist': dist,
            'med': med, 'p25': p25, 'p75': p75,
            'n_seeds': len(seed_curves),
            'best': best,
        })

    if not run_records:
        print(f'plot_discovery_curve_product ({suffix}): no data found; skipping.')
        return

    print(f'Product mode ({suffix}): loaded {len(run_records)} starting materials '
          f'(total seeds: {sum(r["n_seeds"] for r in run_records)}).')

    # --- Random baseline using product reward metric (gamma=1: r_Ehull × r_DOS) ---
    r_ehull_arr = design_space['e_above_hull'].apply(ehull_reward).values
    r_dos_arr = design_space['r_DOS'].values
    product_scores = r_ehull_arr * r_dos_arr

    rng = np.random.default_rng(0)
    N = len(product_scores)
    rand_mat = np.empty((N_RANDOM_REPS, X_MAX))
    for rep in range(N_RANDOM_REPS):
        perm = np.arange(N)
        rng.shuffle(perm)
        rand_mat[rep] = np.maximum.accumulate(product_scores[perm[:X_MAX]])
    rand_p50 = np.percentile(rand_mat, 50, axis=0)
    rand_p25 = np.percentile(rand_mat, 25, axis=0)
    rand_p75 = np.percentile(rand_mat, 75, axis=0)

    # --- Figure ---
    max_dist = max(r['dist'] for r in run_records)
    cmap = plt.colormaps['RdBu_r']

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20,
                    label='Random')
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.8, linestyle='--',
            label='_nolegend_')

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
    ax.set_ylabel(r'Best product reward ($r_{E_\mathrm{Hull}} \times r_\mathrm{DOS}$)',
                  fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / f'discovery_curve_edit_distance_product_{suffix}.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')


def plot_discovery_curve_ms1_vs_ms3(study_dir, design_space, dos_rewards):
    """Compare +1 move vs +3 move product-reward discovery curves.

    Pools all starting materials and shows grand-median best product reward
    vs unique compounds evaluated for product_mode_g1_ms1 (+1 move) and
    product_mode_g1 (+3 move), plus the random baseline.
    """
    ms3_dir = study_dir / 'product_mode_g1'
    ms1_dir = study_dir / 'product_mode_g1_ms1'
    if not ms3_dir.exists() and not ms1_dir.exists():
        print('plot_discovery_curve_ms1_vs_ms3: neither ms1 nor ms3 data found; skipping.')
        return

    x_grid = np.arange(1, X_MAX + 1)

    def _pool_curves(subdir):
        """Collect and interpolate all seed curves from one move-step directory."""
        all_interps = []
        for elem, tm in EXTENDED_MCTS_RUNS:
            seed_curves = load_product_seeds(study_dir, elem, tm, N_EXTENDED_SEEDS, subdir)
            for x_m, y_m in seed_curves:
                interp = interpolate_curve(x_m, y_m, x_grid)
                n = int(x_m.max()) if len(x_m) else 0
                interp[n:] = np.nan
                all_interps.append(interp)
        return np.array(all_interps) if all_interps else None

    mat_ms3 = _pool_curves('product_mode_g1')
    mat_ms1 = _pool_curves('product_mode_g1_ms1')

    # Random baseline (product reward)
    r_ehull_arr = design_space['e_above_hull'].apply(ehull_reward).values
    r_dos_arr = design_space['r_DOS'].values
    product_scores = r_ehull_arr * r_dos_arr
    rng = np.random.default_rng(0)
    N = len(product_scores)
    rand_mat = np.empty((N_RANDOM_REPS, X_MAX))
    for rep in range(N_RANDOM_REPS):
        perm = np.arange(N)
        rng.shuffle(perm)
        rand_mat[rep] = np.maximum.accumulate(product_scores[perm[:X_MAX]])
    rand_p50 = np.percentile(rand_mat, 50, axis=0)
    rand_p25 = np.percentile(rand_mat, 25, axis=0)
    rand_p75 = np.percentile(rand_mat, 75, axis=0)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20)
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.5, linestyle='--', label='Random')

    series = [
        (mat_ms3, '#2166AC', '+3 move (product)'),
        (mat_ms1, '#4DAC26', '+1 move (product)'),
    ]
    for mat, color, label in series:
        if mat is None:
            continue
        # Only show percentiles where ≥10 seeds have data to avoid end-of-run IQR spikes.
        coverage = np.sum(~np.isnan(mat), axis=0)
        with np.errstate(all='ignore'):
            med = np.where(coverage >= 10, np.nanmedian(mat, axis=0), np.nan)
            p25 = np.where(coverage >= 10, np.nanpercentile(mat, 25, axis=0), np.nan)
            p75 = np.where(coverage >= 10, np.nanpercentile(mat, 75, axis=0), np.nan)
        valid = ~np.isnan(med)
        if not valid.any():
            continue
        ax.fill_between(x_grid[valid], p25[valid], p75[valid],
                        color=color, alpha=0.20, linewidth=0)
        ax.plot(x_grid[valid], med[valid], color=color, lw=1.8, label=label)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Unique compounds evaluated', fontsize=10)
    ax.set_ylabel(r'Best product reward ($r_{E_\mathrm{Hull}} \times r_\mathrm{DOS}$)',
                  fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'discovery_curve_product_ms1_vs_ms3.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')


def plot_ehull_vs_rdos_product(study_dir, design_space, dos_rewards):
    """Scatter of full design space with top compounds from product-mode runs overlaid.

    Background: all lanthanide+U compounds (gray).
    Overlay: top 20 unique compounds by additive composite score discovered across
    all product-mode seeds (blue triangles) — same axes as the existing ehull_vs_rdos
    figure so the two are directly comparable.
    """
    prod_dir = study_dir / 'product_mode_g1'
    if not prod_dir.exists():
        print('plot_ehull_vs_rdos_product: product_mode_g1/ not found; skipping.')
        return

    # Pool all unique compounds discovered across every product-mode seed
    all_rows = []
    for elem, tm in EXTENDED_MCTS_RUNS:
        for seed in range(N_EXTENDED_SEEDS):
            csv = prod_dir / f'{elem}_{tm}_s{seed}' / 'all_compounds.csv'
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if 'name' not in df.columns and 'formula' in df.columns:
                df = df.rename(columns={'formula': 'name'})
            all_rows.append(df)

    if not all_rows:
        print('plot_ehull_vs_rdos_product: no compound data found; skipping.')
        return

    pooled = pd.concat(all_rows, ignore_index=True)
    # Deduplicate by formula; keep row with lowest e_above_hull per compound
    pooled = (pooled.sort_values('e_above_hull')
              .drop_duplicates(subset=['name'], keep='first')
              .reset_index(drop=True))
    pooled = pooled[pooled['e_above_hull'] < SENTINEL_EHULL].reset_index(drop=True)
    # Rank by product reward (same metric used during search)
    pooled['r_DOS_val'] = pooled['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    pooled['r_ehull'] = pooled['e_above_hull'].apply(ehull_reward)
    pooled['product_reward'] = pooled['r_ehull'] * pooled['r_DOS_val']
    pooled = pooled.sort_values('product_reward', ascending=False).reset_index(drop=True)
    top15 = pooled.head(15)
    print(f'Product-mode ehull_vs_rdos: {len(pooled)} unique compounds; '
          f'top product_reward={pooled["product_reward"].iloc[0]:.4f}')

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(3, 3))

    x_all = design_space['r_DOS'].values
    y_all = design_space['e_above_hull'].values
    ax.scatter(x_all, y_all, s=4, color='#D0D0D0', linewidths=0,
               label=f'All Compounds (n={len(design_space):,})')

    xs = top15['r_DOS_val'].values
    ys = top15['e_above_hull'].values
    ax.scatter(xs, ys, s=45, color='#5BC0EB', marker='^',
               edgecolors='none', alpha=0.55, label='Top 15 (MCTS)')

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylim(top=2)
    ax.set_xlabel(r'$r_{\mathrm{DOS}}$', fontsize=9)
    ax.set_ylabel(r'$E_{\mathrm{Hull}}$ (eV/atom)', fontsize=9)
    ax.tick_params(labelsize=8)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='#D0D0D0', markeredgecolor='#D0D0D0',
               markersize=5, label='All Compounds'),
        Line2D([0], [0], marker='^', linestyle='None',
               markerfacecolor='#5BC0EB', markeredgecolor='none',
               markersize=7, alpha=0.55, label='Top 15 (MCTS)'),
    ]
    ax.legend(handles=legend_handles, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / 'ehull_vs_rdos_product.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {out_path}')


SENTINEL_EHULL = 9.9  # compounds at or above this have missing MP data


def _top15_avg_rank(run_dir_list, dos_rewards, global_ranks, design_space_size):
    """Pool all_compounds.csv files from run_dir_list, deduplicate, return avg global
    rank of the top-15 compounds by additive composite score.  Compounds missing from
    global_ranks (no MP data) are assigned rank design_space_size + 1.
    """
    frames = []
    for d in run_dir_list:
        csv = d / 'all_compounds.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if 'name' not in df.columns and 'formula' in df.columns:
            df = df.rename(columns={'formula': 'name'})
        frames.append(df[['name', 'e_above_hull']].copy())
    if not frames:
        return None
    pooled = pd.concat(frames, ignore_index=True)
    pooled = (pooled.sort_values('e_above_hull')
              .drop_duplicates(subset=['name'], keep='first')
              .reset_index(drop=True))
    pooled['r_DOS'] = pooled['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    pooled['composite'] = (pooled['e_above_hull'].apply(ehull_reward)
                           * (NORMALIZED_GAMMA * pooled['r_DOS']))
    pooled = pooled.sort_values('composite', ascending=False).reset_index(drop=True)
    top15 = pooled.head(15)
    ranks = [global_ranks.get(_decompose(row['name']), design_space_size + 1)
             for _, row in top15.iterrows()]
    return float(np.mean(ranks))


def plot_edit_distance_vs_rank(study_dir, design_space, dos_rewards, global_ranks):
    """Edit distance vs. average global rank of top-15 compounds found by MCTS.

    One point per (elem, tm) starting material, pooled across all seeds.
    Shown for both extended-mode (additive) and product-mode (multiplicative) runs.
    Lower rank = better discovery.
    """
    prod_dir = study_dir / 'product_mode_g1'
    ext_dir = study_dir / 'extended_mode'
    n_space = len(design_space)

    series = {}  # label -> list of (dist, avg_rank)
    for mode_label, base_dir in [('Additive', ext_dir), ('Multiplicative', prod_dir)]:
        if not base_dir.exists():
            continue
        points = []
        for elem, tm in EXTENDED_MCTS_RUNS:
            run_dirs = [base_dir / f'{elem}_{tm}_s{seed}'
                        for seed in range(N_EXTENDED_SEEDS)]
            avg_rank = _top15_avg_rank(run_dirs, dos_rewards, global_ranks, n_space)
            if avg_rank is None:
                continue
            points.append((edit_distance(elem, tm), avg_rank))
        if points:
            series[mode_label] = points

    if not series:
        print('plot_edit_distance_vs_rank: no data found; skipping.')
        return

    fig, ax = plt.subplots(figsize=(3.5, 3))
    colors = {'Additive': '#2166AC', 'Multiplicative': '#D6604D'}
    markers = {'Additive': 'o', 'Multiplicative': 's'}

    for label, points in series.items():
        dists = np.array([p[0] for p in points])
        ranks = np.array([p[1] for p in points])
        # scatter individual starting materials
        ax.scatter(dists, ranks, color=colors[label], marker=markers[label],
                   s=25, alpha=0.6, linewidths=0, label=f'_{label}')
        # median line grouped by unique edit distance
        unique_dists = sorted(set(dists))
        med_ranks = [np.median(ranks[dists == d]) for d in unique_dists]
        ax.plot(unique_dists, med_ranks, color=colors[label], lw=1.5,
                marker=markers[label], markersize=5, label=label)

    ax.invert_yaxis()  # rank 1 at top
    ax.set_xlabel('Edit distance to global best', fontsize=10)
    ax.set_ylabel('Avg. global rank (top-15)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'edit_distance_vs_rank.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')


def plot_discovery_curve_product_single(study_dir, design_space, dos_rewards):
    """Non-extended product-mode discovery curve: one line per starting material (seed 0).

    Mirrors discovery_curve_edit_distance.png format but for the multiplicative reward
    runs.  Each starting material shown as a single curve coloured by edit distance.
    """
    x_grid = np.arange(1, X_MAX + 1)
    prod_dir = study_dir / 'product_mode_g1'
    if not prod_dir.exists():
        print('plot_discovery_curve_product_single: product_mode_g1/ not found; skipping.')
        return

    run_records = []
    for elem, tm in EXTENDED_MCTS_RUNS:
        conv_csv = prod_dir / f'{elem}_{tm}_s0' / 'convergence_history.csv'
        if not conv_csv.exists():
            continue
        conv_df = pd.read_csv(conv_csv)
        x_m, y_m = mcts_curve_from_conv(conv_df)
        if len(x_m) == 0:
            continue
        dist = edit_distance(elem, tm)
        interp = interpolate_curve(x_m, y_m, x_grid)
        n = int(x_m.max())
        interp[n:] = np.nan
        best = float(np.nanmax(y_m))
        if best < 0:
            continue
        run_records.append({'elem': elem, 'tm': tm, 'dist': dist,
                            'curve': interp, 'best': best})

    if not run_records:
        print('plot_discovery_curve_product_single: no data found; skipping.')
        return

    print(f'Product mode (single seed): {len(run_records)} starting materials.')

    # Random baseline using product reward (gamma=1: r_Ehull × r_DOS)
    r_ehull_arr = design_space['e_above_hull'].apply(ehull_reward).values
    r_dos_arr = design_space['r_DOS'].values
    product_scores = r_ehull_arr * r_dos_arr
    rng = np.random.default_rng(0)
    N = len(product_scores)
    rand_mat = np.empty((N_RANDOM_REPS, X_MAX))
    for rep in range(N_RANDOM_REPS):
        perm = np.arange(N)
        rng.shuffle(perm)
        rand_mat[rep] = np.maximum.accumulate(product_scores[perm[:X_MAX]])
    rand_p50 = np.percentile(rand_mat, 50, axis=0)
    rand_p25 = np.percentile(rand_mat, 25, axis=0)
    rand_p75 = np.percentile(rand_mat, 75, axis=0)

    max_dist = max(r['dist'] for r in run_records)
    cmap = plt.colormaps['RdBu_r']

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20,
                    label='Random')
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.8, linestyle='--',
            label='_nolegend_')

    for rec in sorted(run_records, key=lambda r: -r['dist']):
        c = cmap(rec['dist'] / max(max_dist, 1))
        valid = ~np.isnan(rec['curve'])
        if not valid.any():
            continue
        ax.plot(x_grid[valid], rec['curve'][valid], color=c, lw=1.0, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Edit distance to\nglobal best', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Iterations', fontsize=10)
    ax.set_ylabel(r'Best product reward ($r_{E_\mathrm{Hull}} \times r_\mathrm{DOS}$)',
                  fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'discovery_curve_edit_distance_product_single_ms3.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {fig_path}')


def plot_move_step_comparison(study_dir, design_space, dos_rewards, global_ranks):
    """Compare +1 move (additive search) vs +3 move (multiplicative search) on the
    multiplicative metric: avg global rank of top-15 compounds, per starting material.

    For +1 move runs: evaluate all_compounds.csv retroactively on the product reward.
    For +3 move runs: use product_mode (which searched with multiplicative reward).
    """
    n_space = len(design_space)

    def top15_mult_rank(csv_path):
        """Return avg global rank of top-15 by multiplicative reward from one run."""
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        if 'name' not in df.columns and 'formula' in df.columns:
            df = df.rename(columns={'formula': 'name'})
        df = df.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)
        df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
        df['product'] = (df['e_above_hull'].apply(ehull_reward)
                         * (NORMALIZED_GAMMA * df['r_DOS']))
        df = df.sort_values('product', ascending=False).reset_index(drop=True)
        top15 = df.head(15)
        ranks = [global_ranks.get(_decompose(r['name']), n_space + 1)
                 for _, r in top15.iterrows()]
        return float(np.mean(ranks))

    # Map each (elem, tm) to its +1 move run directory
    move1_lookup = {(e, t): d for e, t, d in MCTS_RUNS}

    points_1 = []    # (dist, avg_rank) for +1 move additive (retroactive product eval)
    points_3 = []    # (dist, avg_rank) for +3 move product search
    points_ms1 = []  # (dist, avg_rank) for +1 move product search (product_mode_g1_ms1)

    for elem, tm in EXTENDED_MCTS_RUNS:
        dist = edit_distance(elem, tm)

        # +1 move additive: single run from original MCTS_RUNS, retroactive product eval
        # Skip La — it is excluded from the lanthanide+U design space so its rank is invalid.
        if elem == 'La':
            pass
        else:
            subdir = move1_lookup.get((elem, tm))
            if subdir is not None:
                base = study_dir / subdir if subdir else study_dir
                csv1 = base / f'{elem}_start' / 'all_compounds.csv'
            else:
                csv1 = Path('/nonexistent')
            rank1 = top15_mult_rank(csv1)
            if rank1 is not None:
                points_1.append((dist, rank1))

        # +3 move product: pool all product_mode_g1 seeds (move-step=3)
        run_dirs = [study_dir / 'product_mode_g1' / f'{elem}_{tm}_s{s}'
                    for s in range(N_EXTENDED_SEEDS)]
        rank3 = _top15_avg_rank(run_dirs, dos_rewards, global_ranks, n_space)
        if rank3 is not None:
            points_3.append((dist, rank3))

        # +1 move product: pool all product_mode_g1_ms1 seeds (move-step=1)
        run_dirs_ms1 = [study_dir / 'product_mode_g1_ms1' / f'{elem}_{tm}_s{s}'
                        for s in range(N_EXTENDED_SEEDS)]
        rank_ms1 = _top15_avg_rank(run_dirs_ms1, dos_rewards, global_ranks, n_space)
        if rank_ms1 is not None:
            points_ms1.append((dist, rank_ms1))

    if not points_1 and not points_3 and not points_ms1:
        print('plot_move_step_comparison: no data; skipping.')
        return

    # Print summary stats
    r1 = [p[1] for p in points_1]
    r3 = [p[1] for p in points_3]
    rms1 = [p[1] for p in points_ms1]
    if r1:
        print(f'\n+1 move (additive, retroactive mult eval): n={len(r1)}, '
              f'mean rank={np.mean(r1):.1f}, median={np.median(r1):.1f}')
    if r3:
        print(f'+3 move (product search):                  n={len(r3)}, '
              f'mean rank={np.mean(r3):.1f}, median={np.median(r3):.1f}')
    if rms1:
        print(f'+1 move (product search):                  n={len(rms1)}, '
              f'mean rank={np.mean(rms1):.1f}, median={np.median(rms1):.1f}')

    fig, ax = plt.subplots(figsize=(3.5, 3))

    for label, points, color, marker in [
        ('+1 move (additive)', points_1, '#D6604D', 'o'),
        ('+3 move (product)', points_3, '#2166AC', 's'),
        ('+1 move (product)', points_ms1, '#4DAC26', '^'),
    ]:
        if not points:
            continue
        dists = np.array([p[0] for p in points])
        ranks = np.array([p[1] for p in points])
        ax.scatter(dists, ranks, color=color, marker=marker,
                   s=25, alpha=0.6, linewidths=0, label=f'_{label}')
        unique_dists = sorted(set(dists))
        med_ranks = [np.median(ranks[dists == d]) for d in unique_dists]
        ax.plot(unique_dists, med_ranks, color=color, lw=1.5,
                marker=marker, markersize=5, label=label)

    ax.invert_yaxis()
    # Cap y-axis at the design-space size so outliers don't compress the interesting range.
    all_ranks = ([p[1] for p in points_1] + [p[1] for p in points_3]
                 + [p[1] for p in points_ms1])
    y_max = min(n_space, max(all_ranks) * 1.1) if all_ranks else n_space
    ax.set_ylim(bottom=y_max, top=0)
    ax.set_xlabel('Edit distance to global best', fontsize=10)
    ax.set_ylabel('Avg. global rank (top-15)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    figures_dir = study_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'move_step_comparison.png'
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

    # --- Figure: edit-distance coloured bands from the +1-move runs ---
    # Pool the original single-seed runs with any additional seeds from
    # move1_seeds/{elem}_{tm}_s{seed}/ (generated by run_move1_seeds.sh).
    # Distances with multiple curves (from different starts or seeds) get real
    # IQR bands; single-curve distances show a line only.
    # The failed La-Ti run (best ≈ -1) is excluded.
    from collections import defaultdict
    _dist_curves_orig = defaultdict(list)
    for _rec in run_records:
        if _rec['best'] < 0:
            print(f"  Excluding failed run {_rec['label']} (best={_rec['best']:.4f})")
            continue
        _dist_curves_orig[_rec['dist']].append(_rec['curve'])

    # Supplement with extra seeds from move1_seeds/ if available
    _move1_dir = study_dir / 'move1_seeds'
    if _move1_dir.exists():
        _n_extra = 0
        for _elem, _tm, _subdir in MCTS_RUNS:
            _dist_m1 = edit_distance(_elem, _tm)
            for _seed in range(1, 5):
                _conv = _move1_dir / f'{_elem}_{_tm}_s{_seed}' / 'convergence_history.csv'
                if not _conv.exists():
                    continue
                _cdf = pd.read_csv(_conv)
                _xm, _ym = mcts_curve_from_conv(_cdf)
                _n = min(int(_xm.max()) if len(_xm) else 0, X_MAX)
                _interp_m1 = interpolate_curve(_xm, _ym, x_grid)
                _interp_m1[_n:] = np.nan
                _dist_curves_orig[_dist_m1].append(_interp_m1)
                _n_extra += 1
        if _n_extra:
            print(f'  Loaded {_n_extra} extra move1_seeds curves.')

    _dist_stats = {}
    for _dist, _curves in _dist_curves_orig.items():
        _mat = np.array(_curves)
        with np.errstate(all='ignore'):
            _dist_stats[_dist] = {
                'med': np.nanmedian(_mat, axis=0),
                'p25': np.nanpercentile(_mat, 25, axis=0),
                'p75': np.nanpercentile(_mat, 75, axis=0),
                'n': len(_curves),
            }
    print(f'Edit-distance figure: {len(_dist_stats)} distance groups, '
          f'{sum(v["n"] for v in _dist_stats.values())} total curves '
          f'(+1-move runs).')

    _max_dist = max(_dist_stats.keys())
    cmap = plt.colormaps['RdBu_r']

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.fill_between(x_grid, rand_p25, rand_p75, color='#888888', alpha=0.20,
                    label='Random')
    ax.plot(x_grid, rand_p50, color='#555555', lw=1.8, linestyle='--',
            label='_nolegend_')

    for _dist in sorted(_dist_stats.keys(), reverse=True):
        _rec = _dist_stats[_dist]
        _c = cmap(_dist / max(_max_dist, 1))
        _valid = ~np.isnan(_rec['med'])
        if not _valid.any():
            continue
        if _rec['n'] > 1:
            ax.fill_between(x_grid[_valid], _rec['p25'][_valid], _rec['p75'][_valid],
                            color=_c, alpha=0.18, linewidth=0)
        ax.plot(x_grid[_valid], _rec['med'][_valid], color=_c, lw=1.0, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=_max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Edit distance to\nglobal best', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Iterations', fontsize=10)
    ax.set_ylabel('Best composite score', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
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

    # --- Product-mode figures (generated if data exists) ---
    plot_discovery_curve_product(study_dir, design_space, dos_rewards,
                                  subdir='product_mode_g1', suffix='ms3')
    plot_discovery_curve_product(study_dir, design_space, dos_rewards,
                                  subdir='product_mode_g1_ms1', suffix='ms1')
    plot_discovery_curve_product_single(study_dir, design_space, dos_rewards)
    plot_discovery_curve_ms1_vs_ms3(study_dir, design_space, dos_rewards)
    plot_ehull_vs_rdos_product(study_dir, design_space, dos_rewards)
    plot_edit_distance_vs_rank(study_dir, design_space, dos_rewards, global_ranks)
    plot_move_step_comparison(study_dir, design_space, dos_rewards, global_ranks)


if __name__ == '__main__':
    main()

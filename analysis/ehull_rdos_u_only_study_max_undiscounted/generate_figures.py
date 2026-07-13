#!/usr/bin/env python3
"""
Generate publication figures as PNGs directly from MCTS outputs and high-throughput data.

Max-undiscounted study:
  - gamma = 1/(max raw r_DOS across the 108 U-only compounds) = 1/2516.1664410449775
    (normalizes gamma*r_DOS to top out at 1.0, same scale as ehull_reward's ~[-1,1] range;
    see NORMALIZED_GAMMA below - NOT read from config.json)
  - rollout_aggregation = max with rollout_discount=1.0 (no 0.9^depth decay on extra
    rollout samples — all samples compared on equal footing before taking the max)

Contrasts with ehull_rdos_u_only_study_final/ (same gamma, mean aggregation) and
ehull_rdos_u_only_study_normalized/ (same gamma, max+0.9^depth discount).

This script replaces the prior gnuplot/prepare pipeline and writes PNG files:
- ehull_vs_rdos.png
- convergence_by_starting_material.png (reuses sweep_starting_material.py's own
  starting_material_sweep_max_undiscounted data here)
- radial_tree_composite.png (delegates to create_composite_radial_tree.py)

It expects to be run from this directory where the MCTS run outputs
(`all_compounds.csv`, `convergence_history.csv`, `mcts_object.pkl`) are present
(the included `run_study.sh` places them there). It also expects the
high-throughput cache `high_throughput_mace_results.full.csv` and
`doscar_peaks_data_with_U.csv` to be available at the repository root for the
`ehull_vs_rdos` figure if the analysis CSV is not present (rDOS is always
computed in real time from the raw peaks file - no precomputed rewards cache).
"""

from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import sys
import shutil
import re
import json
import pickle
from ase.data import atomic_numbers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesized_compounds import SYNTHESIZED_COMPOUNDS
from create_composite_radial_tree import F_BLOCK, TRANSITION_METALS, GROUP_IV

# 1 / (max raw r_DOS across the 108 U-only compounds, U-Pb-Mn / Mn6Pb6U =
# 2516.1664410449775) - normalizes gamma*r_DOS to top out at 1.0, the same
# scale as ehull_reward's ~[-1,1] range. Deliberately NOT read from
# config.json: that file is shared with the calibrated (gamma=0.0001) study
# and must stay there.
NORMALIZED_GAMMA = 1.0 / 2516.1664410449775
DEFAULT_GAMMA = NORMALIZED_GAMMA


def load_gamma(default: float = DEFAULT_GAMMA) -> float:
    """Return the fixed normalized gamma for this study (ignores config.json)."""
    return NORMALIZED_GAMMA


def compute_composite(df, beta=1.0, gamma=None):
    if gamma is None:
        gamma = load_gamma()
    try:
        from mcts_crystal.node import ehull_reward
    except Exception:
        # Fallback: define local ehull_reward matching mcts_crystal.node implementation
        def ehull_reward(e_hull: float) -> float:
            import numpy as _np
            return -_np.tanh(120.0 * (e_hull - 0.05))
    df = df.copy()
    df['name'] = df.get('formula', df.get('name'))
    # ensure r_DOS is computed from canonical doscar rewards if not present
    # Do NOT store gamma-weighted values in r_DOS; r_DOS should be the raw [0,1] reward.
    if 'r_DOS' not in df.columns or df['r_DOS'].isnull().all() or (df['r_DOS'].astype(float) == 0.0).all():
        # load master mapping from repo root
        repo_root = Path(__file__).parents[2]
        df_master, dos_by_key = load_master_mace(repo_root)

        # element-set key function (ignore f-block differences)
        F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}
        def parse_elems(s):
            if pd.isna(s):
                return []
            if '-' in str(s):
                parts = [p for p in re.split('[^A-Za-z]', str(s)) if p]
                return parts
            return re.findall(r'[A-Z][a-z]?', str(s))

        def key_from_name(name):
            elems = parse_elems(name)
            return tuple(sorted([e for e in elems if e not in F_BLOCK]))

        # default to 0.0
        df['r_DOS'] = 0.0
        for idx, row in df.iterrows():
            key = key_from_name(row.get('name', row.get('formula', '')))
            if key in dos_by_key:
                df.at[idx, 'r_DOS'] = float(dos_by_key[key])
            else:
                # fallback: try normalized name match against master if available
                if df_master is not None:
                    nm = str(row.get('name', row.get('formula', '')))
                    # try exact match on name/formula
                    try:
                        match = df_master[(df_master.get('name', df_master.get('formula')) == nm)].iloc[0]
                        # if master has r_dos or similar, use it
                        if 'r_dos' in df_master.columns and not pd.isna(match.get('r_dos')):
                            df.at[idx, 'r_DOS'] = float(match.get('r_dos'))
                    except Exception:
                        pass
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
    # Keep raw r_DOS as canonical; composite uses gamma weighting when computing score
    df['weighted_r_DOS'] = gamma * df['r_DOS']
    df['composite_score'] = beta * df['ehull_reward'] + df['weighted_r_DOS']
    df_sorted = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return df_sorted


def load_master_mace(repo_root: Path):
    """Load canonical high-throughput MACE CSV and DOS rewards (computed in real
    time from raw peak data - there is no precomputed rewards cache).
    Returns (df_mace, dos_by_key) where dos_by_key maps element-set keys (the
    TM/group-IV pair, f-block element stripped) to a DOS reward.

    doscar_peaks_data_with_U.csv has a separate row - and a genuinely
    different reward - per f-block element (e.g. 'U-Ge-Fe' vs 'Er-Ge-Fe' are
    different compounds with different DOS peaks), so for each (TM, group-IV)
    pair we prefer the exact uranium-row reward where one exists. We only
    fall back to the max across other f-block substitutions (a same-framework
    proxy) when uranium's own row is missing - never as a default, since that
    would substitute a different compound's reward for uranium's.
    """
    # Prefer the file in this repo; only search parent/sibling folders as a fallback
    # (this repo may sit next to other mcts_materials checkouts with stale copies).
    MAX_PARENT_DEPTH = 4
    search_roots = [repo_root] + list(repo_root.parents)[:MAX_PARENT_DEPTH]

    def find_canonical(filename):
        local = repo_root / filename
        if local.exists():
            return local
        for r in search_roots:
            try:
                candidates = list(Path(r).rglob(filename))
            except OSError:
                continue
            if candidates:
                return candidates[0]
        return repo_root / filename

    mace_csv = find_canonical('high_throughput_mace_results.full.csv')
    df_mace = None
    dos_by_key = {}
    if mace_csv.exists():
        try:
            df_mace = pd.read_csv(mace_csv)
        except Exception:
            df_mace = None

    # Compute DOS rewards in real time from the raw peaks file (prefer local repo copy)
    from mcts_crystal.doscar_utils import DoscarRewardLookup
    peaks_csv = find_canonical('doscar_peaks_data_with_U.csv')
    dos_dict = DoscarRewardLookup(peaks_file=str(peaks_csv)).rewards_dict

    F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}

    def parse_elems(s):
        if pd.isna(s):
            return []
        if '-' in str(s):
            parts = [p for p in re.split('[^A-Za-z]', str(s)) if p]
            return parts
        return re.findall(r'[A-Z][a-z]?', str(s))

    # collapse compound_name -> reward into element-set keys (TM/group-IV pair,
    # f-block element stripped). dos_by_key_fallback is the max across whatever
    # f-block substitutions exist for that pair; dos_by_key_uranium is the
    # exact 'U-...' row only. Uranium's own data wins wherever it exists.
    dos_by_key_fallback = {}
    dos_by_key_uranium = {}
    for name, val in dos_dict.items():
        try:
            v = float(val)
        except Exception:
            continue
        elems = parse_elems(name)
        key = tuple(sorted([e for e in elems if e not in F_BLOCK]))
        if not key:
            continue
        dos_by_key_fallback[key] = max(dos_by_key_fallback.get(key, float('-inf')), v)
        if 'U' in elems:
            dos_by_key_uranium[key] = v

    dos_by_key = dos_by_key_fallback
    dos_by_key.update(dos_by_key_uranium)

    return df_mace, dos_by_key


def describe_mcts_run_starting_material(run_dir: Path, mcts_materials_root: Path):
    """Return (formatted_label, edit_distance) describing the starting
    material of the MCTS run pickled in run_dir/mcts_object.pkl: its
    U-Tm-GroupIV-formatted formula, and its move-graph edit distance to the
    true global-best U-only compound (per compute_global_u_only_ranks).
    Returns (None, None) if the pickle or the global-best lookup is unavailable.
    """
    pkl_path = run_dir / 'mcts_object.pkl'
    if not pkl_path.exists():
        return None, None
    with open(pkl_path, 'rb') as f:
        mcts = pickle.load(f)
    root_formula = mcts.root.get_chemical_formula()

    global_ranks = compute_global_u_only_ranks(mcts_materials_root)
    best_key = next((k for k, r in global_ranks.items() if r == 1), None)
    if best_key is None:
        return format_name_u_tm_giv(root_formula), None
    target_tm = next((e for e in best_key if e in TRANSITION_METALS), None)
    target_giv = next((e for e in best_key if e in GROUP_IV), None)
    tm, giv = _parse_tm_giv(root_formula)
    if not (target_tm and target_giv and tm and giv):
        return format_name_u_tm_giv(root_formula), None

    distance = _edit_distance_to_target(tm, giv, target_tm, target_giv)
    return format_name_u_tm_giv(root_formula), distance


def plot_ehull_vs_rdos(repo_root: Path, out_dir: Path, mcts_run_dir: Path = None,
                        start_label: str = None, start_dist: int = None):
    """mcts_run_dir: directory whose all_compounds_by_composite_score.csv and
    mcts_object.pkl drive the 'Top 10 (MCTS)' overlay and its starting-material
    annotation. Defaults to out_dir (this study's main run). The 'All
    Compounds' backdrop and synthesized-compound overlays are always the full
    exhaustive design space, independent of mcts_run_dir.

    start_label/start_dist: pre-computed describe_mcts_run_starting_material()
    result for mcts_run_dir; computed internally if not passed (passing it in
    lets main() share one computation with the radial-tree figure)."""
    if mcts_run_dir is None:
        mcts_run_dir = out_dir
    comp_csv = out_dir / 'all_compounds_by_composite_score.csv'
    df_u = None
    # x-axis for this plot is gamma * r_DOS (the actual composite-score component),
    # applied consistently to the backdrop and every overlay below.
    gamma = load_gamma()

    # Delegates to load_master_mace() for both the high-throughput CSV and the
    # DOS-reward lookup, so this backdrop always uses the same (uranium-row-
    # preferring, not cross-lanthanide-max) r_DOS as compute_composite() and
    # the top15 table - see load_master_mace()'s docstring for why that matters.
    df_mace, dos_by_key = load_master_mace(repo_root)

    def _make_key_from_formula(formula):
        if pd.isna(formula):
            return ()
        s = str(formula)
        if '-' in s:
            parts = [p for p in re.split('[^A-Za-z]', s) if p]
        else:
            parts = re.findall(r'[A-Z][a-z]?', s)
        F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}
        return tuple(sorted([e for e in parts if e not in F_BLOCK]))

    if df_mace is not None:
        # attach r_dos by element-set key (TM/group-IV pair, f-block element stripped)
        df_mace['name'] = df_mace.get('name', df_mace.get('formula'))
        df_mace['r_dos'] = 0.0
        for idx, row in df_mace.iterrows():
            key = _make_key_from_formula(row['name'])
            if key in dos_by_key:
                df_mace.at[idx, 'r_dos'] = dos_by_key[key]
        if 'name' not in df_mace.columns and 'formula' in df_mace.columns:
            df_mace['name'] = df_mace['formula']

        # Filter to U-containing compounds (exclude Ce)
        def elem_set(name):
            if pd.isna(name):
                return set()
            s = str(name)
            if '-' in s:
                parts = [p for p in re.split('[^A-Za-z]', s) if p]
                return set([p.capitalize() for p in parts])
            parts = re.findall(r'[A-Z][a-z]?', s)
            return set(parts)

        df_mace['elem_set'] = df_mace['name'].apply(elem_set)
        df_u = df_mace[df_mace['elem_set'].apply(lambda s: 'U' in s and 'Ce' not in s)].copy()

    # If we still don't have a mace-derived U list, fall back to the analysis CSV (MCTS outputs)
    if df_u is None and comp_csv.exists():
        df_all = pd.read_csv(comp_csv)
        # Filter U-containing (no Ce) by formula/name field
        def has_element(name, elem):
            import re
            pattern = r'(?<![a-z])' + elem + r'(?![a-z])'
            return bool(re.search(pattern, str(name)))

        if 'name' not in df_all.columns and 'formula' in df_all.columns:
            df_all['name'] = df_all['formula']

        df_u = df_all[df_all['name'].apply(lambda x: has_element(x, 'U'))].copy()
        df_u = df_u[~df_u['name'].apply(lambda x: has_element(x, 'Ce'))].copy()
        # Ensure r_DOS column exists (case-insensitive fallback)
        if 'r_DOS' not in df_u.columns and 'dos_reward' in df_u.columns:
            df_u['r_DOS'] = df_u['dos_reward']
        df_u['r_dos'] = df_u['r_DOS'].fillna(0.0)

    if df_u is None or df_u.empty:
        print('Skipping ehull_vs_rdos: no U-containing data found')
        return

    # Some entries may not map; attempt conversion using same heuristic as DoscarRewardLookup
    # For robustness, keep 0.0 where not found.

    # attempt to load experimental compounds list (if provided elsewhere in repo)
    exp_src = repo_root / 'redo_mcts_materials' / 'experimental_comparison' / 'compounds_filtered.dat'
    exp_dst = out_dir / 'compounds_filtered.dat'
    df_exp = None
    if exp_src.exists():
        try:
            df_exp = pd.read_csv(exp_src, sep=r"\s+", comment='#', header=None,
                                names=['name', 'e_form', 'e_hull'])
            # copy into analysis folder for consistency and ignore in git
            if not exp_dst.exists():
                shutil.copy(exp_src, exp_dst)
        except Exception:
            df_exp = None

    fig, ax = plt.subplots(figsize=(4, 4))
    # Use r_dos (already present) or fall back to r_DOS column; plot gamma * r_DOS
    rdos_vals = df_u.get('r_dos', df_u.get('r_DOS', pd.Series(0.0))).astype(float) * gamma
    ax.scatter(rdos_vals, df_u['e_above_hull'].astype(float), s=6, color='#D0D0D0', label='All Compounds (U-containing)')

    # Normalize helper
    def _norm(s):
        if pd.isna(s):
            return ''
        return ''.join([c for c in str(s).lower() if c.isalnum()])

    # prepare df_u name_norm
    if 'name' in df_u.columns:
        df_u['name_norm'] = df_u['name'].apply(_norm)
    else:
        df_u['name_norm'] = df_u.get('formula', '').apply(_norm)

    # synthesized names (successful) - see synthesized_compounds.py
    synthesized_names = SYNTHESIZED_COMPOUNDS
    def elem_set(name):
        if pd.isna(name):
            return set()
        s = str(name)
        if '-' in s:
            parts = [p for p in re.split('[^A-Za-z]', s) if p]
            return set([p.capitalize() for p in parts])
        parts = re.findall(r'[A-Z][a-z]?', s)
        return set(parts)

    synth_sets = [elem_set(n) for n in synthesized_names]

    # experimental attempted compounds (from df_exp) and classify as successful/unsuccessful
    exp_unsucc_x, exp_unsucc_y = [], []
    exp_succ_x, exp_succ_y = [], []
    if df_exp is not None and not df_exp.empty:
        df_exp['name_norm'] = df_exp['name'].apply(_norm)
        name_map = {n: r for n, r in zip(df_u['name_norm'], df_u.to_dict('records'))}
        for _, r in df_exp.iterrows():
            nrm = r['name_norm']
            if nrm in name_map:
                row = name_map[nrm]
                x = float(row.get('r_dos', row.get('r_DOS', 0.0))) * gamma
                y = float(row.get('e_above_hull', r['e_hull']))
                # classify by element set equality
                if any(elem_set(r['name']) == s for s in synth_sets):
                    exp_succ_x.append(x); exp_succ_y.append(y)
                else:
                    exp_unsucc_x.append(x); exp_unsucc_y.append(y)

    # plot experimental attempted: unsuccessful = purple unfilled squares
    if exp_unsucc_x:
        ax.scatter(exp_unsucc_x, exp_unsucc_y, s=80, marker='s', facecolors='none', edgecolors='#9467bd', linewidths=1.2, label='Unsuccessful Synthesis')
    # plot experimental successful: purple filled squares
    if exp_succ_x:
        ax.scatter(exp_succ_x, exp_succ_y, s=100, marker='s', facecolors='#9467bd', edgecolors='#9467bd', linewidths=0.8, label='Successful Synthesis')

    # Top10 overlay if available (from mcts_run_dir) - light-blue triangles for
    # MCTS predictions. Smaller + semi-transparent so the several entries that
    # sit very close together (e.g. similar r_DOS/e_hull) visibly darken where
    # they overlap, instead of looking like fewer than 10 markers. Matched by
    # element-set key (not exact name string) since the MCTS run's formula
    # ordering (e.g. "Sn6Ti6U") doesn't always match the master dataset's
    # (e.g. "Ti6Sn6U") - an exact-string lookup here silently dropped points.
    composite_csv = mcts_run_dir / 'all_compounds_by_composite_score.csv'
    if composite_csv.exists():
        df_comp = pd.read_csv(composite_csv)
        top15 = df_comp.head(15)
        mace_lookup = {}
        for _, r in df_u.iterrows():
            mace_lookup[_formula_key(r['name'])] = (r['e_above_hull'], r.get('r_dos', r.get('r_DOS', 0.0)))
        xs, ys = [], []
        for _, row in top15.iterrows():
            name = row['name'] if 'name' in row else row.get('formula')
            key = _formula_key(name)
            if key in mace_lookup:
                e_hull, rdos = mace_lookup[key]
                xs.append(float(rdos) * gamma)
                ys.append(e_hull)
        if xs:
            ax.scatter(xs, ys, s=45, color='#5BC0EB', marker='^', edgecolors='none',
                       alpha=0.45, label='Top 15 (MCTS)')

    # Annotate which starting material produced this Top-10 overlay, and how
    # far (in MCTS move-graph hops) it is from the true global-best compound.
    if start_label is None:
        start_label, start_dist = describe_mcts_run_starting_material(mcts_run_dir, repo_root)

    ax.set_xlabel(r"$\alpha_{\mathrm{DOS}} \cdot r_{\mathrm{DOS}}$")
    ax.set_ylabel(r"$E_{\mathrm{Hull}}$ (eV/atom)")
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    # Create a clean legend with specific marker styles so Top10 appears as a triangle
    from matplotlib.lines import Line2D
    legend_handles = []
    # All compounds: small light-gray circle (no line)
    legend_handles.append(Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='#D0D0D0', markeredgecolor='#D0D0D0', markersize=6, label='All Compounds'))
    # Top10 (MCTS): light blue filled triangle (no line)
    legend_handles.append(Line2D([0], [0], marker='^', linestyle='None', markerfacecolor='#5BC0EB', markeredgecolor='none', markersize=8, alpha=0.45, label='Top 15 (MCTS)'))
    # Unsuccessful: purple unfilled square (no line)
    legend_handles.append(Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='none', markeredgecolor='#9467bd', markersize=8, label='Unsuccessful Synthesis'))
    # Successful: purple filled square with purple edge (no line)
    legend_handles.append(Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='#9467bd', markeredgecolor='#9467bd', markersize=9, label='Successful Synthesis'))
    ax.legend(handles=legend_handles, fontsize=8)
    plt.tight_layout()
    if start_label is not None:
        dist_str = f", d={start_dist} to global best" if start_dist is not None else ""
        fig.text(0.5, -0.02, f"Top 15 (MCTS) from {start_label} start{dist_str}",
                  ha='center', va='top', fontsize=7)
    out = out_dir / 'ehull_vs_rdos.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def format_name_u_tm_giv(name):
    """Reorder a 1-6-6 formula as U, transition metal, group IV with the 6's
    rendered as LaTeX subscripts, e.g. 'Fe6Ge6U' -> 'UFe$_6$Ge$_6$'.

    Reuses the same element categorization as create_composite_radial_tree.py
    so the ordering convention stays consistent across figures and tables.
    """
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(name))
    re_elem, tm_part, giv_part = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in F_BLOCK:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_part = (elem, count)
        elif elem in GROUP_IV:
            giv_part = (elem, count)
    if re_elem and tm_part and giv_part:
        tm_elem, tm_n = tm_part
        giv_elem, giv_n = giv_part
        tm_sub = f"$_{{{tm_n}}}$" if tm_n else ''
        giv_sub = f"$_{{{giv_n}}}$" if giv_n else ''
        return f"{re_elem}{tm_elem}{tm_sub}{giv_elem}{giv_sub}"
    return str(name)


def _formula_key(name):
    """Element-set key (sorted, non-f-block) used to match a formula across
    datasets regardless of element ordering, e.g. 'Fe6Ge6U' -> ('Fe', 'Ge')."""
    if pd.isna(name):
        return ()
    s = str(name)
    if '-' in s:
        parts = [p for p in re.split('[^A-Za-z]', s) if p]
    else:
        parts = re.findall(r'[A-Z][a-z]?', s)
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


def compute_global_u_only_ranks(repo_root: Path):
    """Rank every compound in the full high-throughput U-only design space
    (U present, no other f-block element) by the same beta=1.0/gamma composite
    score used everywhere else, so MCTS top-15 results can be checked against
    the true best-in-space compounds (search coverage, not just local quality).

    Returns {element_key: global_rank} with rank 1 = best composite score.
    """
    df_mace, dos_by_key = load_master_mace(repo_root)
    if df_mace is None or not len(df_mace):
        return {}

    try:
        from mcts_crystal.node import ehull_reward
    except Exception:
        def ehull_reward(x):
            return -np.tanh(120.0 * (x - 0.05))

    gamma = load_gamma()
    df = df_mace.copy()
    df['name'] = df.get('name', df.get('formula'))
    df['elem_set'] = df['name'].apply(lambda n: set(re.findall(r'[A-Z][a-z]?', str(n))))
    # U-only design space: contains U, and no other f-block/rare-earth element
    df = df[df['elem_set'].apply(lambda s: 'U' in s and not (s & (F_BLOCK - {'U'})))].copy()
    df['key'] = df['name'].apply(_formula_key)
    df['r_dos'] = df['key'].map(lambda k: dos_by_key.get(k, 0.0))
    df['composite_score'] = df['e_above_hull'].apply(ehull_reward) + gamma * df['r_dos']
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return {key: rank for rank, key in enumerate(df['key'], start=1)}


def write_top15_table(df_sorted: pd.DataFrame, out_dir: Path, repo_root: Path):
    """Write a LaTeX table of the top 15 compounds with requested columns."""
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    top15 = df_sorted.head(15).copy()
    gamma = load_gamma()
    global_ranks = compute_global_u_only_ranks(repo_root)
    # ensure columns exist
    if 'r_DOS' not in top15.columns and 'dos_reward' in top15.columns:
        top15['r_DOS'] = top15['dos_reward']
    if 'weighted_r_DOS' not in top15.columns:
        top15['weighted_r_DOS'] = top15['r_DOS'] * gamma
    if 'ehull_reward' not in top15.columns and 'e_above_hull' in top15.columns:
        # recompute if possible
        try:
            from mcts_crystal.node import ehull_reward
        except Exception:
            def ehull_reward(x):
                import numpy as _np
                return -_np.tanh(120.0 * (x - 0.05))
        top15['ehull_reward'] = top15['e_above_hull'].apply(ehull_reward)

    # synthesized detection by element sets - see synthesized_compounds.py
    synthesized_names = SYNTHESIZED_COMPOUNDS
    import re
    def elem_set(name):
        if pd.isna(name):
            return set()
        s = str(name)
        if '-' in s:
            parts = [p for p in re.split('[^A-Za-z]', s) if p]
            return set([p.capitalize() for p in parts])
        parts = re.findall(r'[A-Z][a-z]?', s)
        return set(parts)

    synth_sets = [elem_set(n) for n in synthesized_names]

    # attempted-but-possibly-unsuccessful compounds - see compounds_filtered.dat
    # (same experimental-attempts file used for the Successful/Unsuccessful
    # Synthesis overlay in plot_ehull_vs_rdos). Anything not in this list was
    # never attempted, so the table should show '-' rather than 'No'.
    attempted_path = out_dir / 'compounds_filtered.dat'
    attempted_sets = []
    if attempted_path.exists():
        try:
            df_exp = pd.read_csv(attempted_path, sep=r"\s+", comment='#', header=None,
                                names=['name', 'e_form', 'e_hull'])
            attempted_sets = [elem_set(n) for n in df_exp['name']]
        except Exception:
            attempted_sets = []

    rows = []
    for rank, (_, r) in enumerate(top15.iterrows(), start=1):
        name = r.get('name', r.get('formula', ''))
        rdos = float(r.get('weighted_r_DOS', 0.0)) if pd.notna(r.get('weighted_r_DOS', None)) else 0.0
        ehull_r = float(r.get('ehull_reward', 0.0)) if pd.notna(r.get('ehull_reward', None)) else 0.0
        comp = float(r.get('composite_score', 0.0)) if pd.notna(r.get('composite_score', None)) else 0.0
        ehull = float(r.get('e_above_hull', np.nan)) if pd.notna(r.get('e_above_hull', None)) else np.nan
        is_synth = any(elem_set(name) == s for s in synth_sets)
        is_attempted = any(elem_set(name) == s for s in attempted_sets)
        if is_synth:
            synth_status = 'Yes'
        elif is_attempted:
            synth_status = 'No'
        else:
            synth_status = '-'
        display_name = format_name_u_tm_giv(name)
        global_rank = global_ranks.get(_formula_key(name))
        rows.append((rank, global_rank, display_name, rdos, ehull_r, comp, ehull, synth_status))

    eol = ' \\\\\n'  # LaTeX row terminator (literal "\\") followed by a real newline
    tex_path = tables_dir / 'top15_u_only.tex'
    with open(tex_path, 'w') as f:
        f.write(f'% Top 15 compounds (U-only study). gamma={gamma:g}. True Rank = rank within the\n')
        f.write(f'% full {len(global_ranks)}-compound exhaustive U-only design space (search coverage check).\n')
        f.write('\\begin{tabular}{rrlrrrrc}\n')
        f.write('\\toprule\n')
        f.write('MCTS Rank & True Rank & Name & $\\gamma \\cdot r_{\\mathrm{DOS}}$ ($\\gamma$='
                f'{gamma:g}) & $r_{{E_{{\\mathrm{{Hull}}}}}}$ & Composite & E\\_hull & Synth' + eol)
        f.write('\\midrule\n')
        for rank, global_rank, name, rdos, ehull_r, comp, ehull, synth in rows:
            gr = str(global_rank) if global_rank is not None else '--'
            f.write(f"{rank} & {gr} & {name} & {rdos:.4f} & {ehull_r:.4f} & {comp:.4f} & {ehull:.4f} & {synth}" + eol)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    print('Wrote LaTeX table:', tex_path)


def _metal_move_neighbors(z):
    """Transition-metal neighbor atomic numbers (same-period +-1, same-group
    +-18/+-32). Mirrors the move rules in mcts_crystal/node.py's
    _determine_possible_moves - kept in sync manually since that method
    operates on an Atoms object's atomic numbers, not bare integers."""
    if 22 <= z <= 30:
        if z == 22:
            return [22, 23, 40]
        if z == 30:
            return [29, 30, 48]
        return [z - 1, z, z + 1, z + 18]
    if 40 <= z <= 48:
        if z == 40:
            return [40, 41, 72, 22]
        if z == 48:
            return [47, 48, 80, 30]
        return [z - 1, z, z + 1, z + 32, z - 18]
    if 72 <= z <= 80:
        if z == 72:
            return [72, 73, 40]
        if z == 80:
            return [79, 80, 48]
        return [z - 1, z, z + 1, z - 32]
    return [z]


_GIV_CHAIN = [14, 32, 50, 82]  # Si, Ge, Sn, Pb


def _giv_move_neighbors(z):
    """Group-IV neighbor atomic numbers: chain-restricted Si<->Ge<->Sn<->Pb.
    Mirrors mcts_crystal/node.py's _determine_possible_moves."""
    idx = _GIV_CHAIN.index(z)
    neighbors = [z]
    if idx > 0:
        neighbors.append(_GIV_CHAIN[idx - 1])
    if idx < len(_GIV_CHAIN) - 1:
        neighbors.append(_GIV_CHAIN[idx + 1])
    return neighbors


def _bfs_distances(start, neighbor_fn):
    """Shortest-path distance from `start` to every node reachable via
    neighbor_fn(node) -> list of neighbor nodes."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        n = queue.popleft()
        for nb in neighbor_fn(n):
            if nb not in dist:
                dist[nb] = dist[n] + 1
                queue.append(nb)
    return dist


def _edit_distance_to_target(tm_symbol, giv_symbol, target_tm, target_giv):
    """Number of MCTS moves from (tm_symbol, giv_symbol) to (target_tm,
    target_giv). node.py's expand() changes the metal AND group-IV element
    simultaneously every move (cartesian product of their neighbor sets), so
    the two attributes advance in parallel - the distance is the max of
    their independent graph distances, not the sum."""
    metal_dist = _bfs_distances(atomic_numbers[target_tm], _metal_move_neighbors)
    giv_dist = _bfs_distances(atomic_numbers[target_giv], _giv_move_neighbors)
    return max(metal_dist[atomic_numbers[tm_symbol]], giv_dist[atomic_numbers[giv_symbol]])


def _parse_tm_giv(formula):
    """Pull out the transition-metal and group-IV element symbols from a
    1-6-6 formula string (f-block element, if any, is ignored)."""
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    tm = next((e for e, _ in matches if e in TRANSITION_METALS), None)
    giv = next((e for e, _ in matches if e in GROUP_IV), None)
    return tm, giv


def _set_plateau_xlim(ax, curves, buffer_frac=0.25, tol_frac=0.01, ignore_first=1):
    """Truncate the x-axis around the transient of one or more same-length
    curves: find the last index at which *any* curve still differs from its
    own final value by more than tol_frac of its settled range, then size the
    axis so that point sits at (1 - buffer_frac) of the width, leaving the
    rest as flat-plateau context.

    A relative tol_frac (rather than exact equality) matters once curves are
    averaged across seeds: a single seed nudging the mean by a negligible,
    sub-percent amount very late in the run would otherwise count as "still
    changing" and drag the whole x-axis back out to the full run length.

    ignore_first excludes a leading pre-search sentinel (e.g. best_reward =
    -10.0 at iteration 0) from both the range and "still changing" checks, so
    it doesn't swamp the real post-search dynamic range.

    Using the slowest-to-settle curve (max plateau-start across `curves`)
    keeps every curve's full transient visible when comparing several at once.
    """
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
        plateau_starts.append((changed[-1] + 1 + ignore_first) if changed.size > 0 else ignore_first)
    if not plateau_starts or n_iter is None:
        return
    plateau_start = min(max(plateau_starts), n_iter - 1)
    if plateau_start > 0:
        xmax = min(plateau_start / (1.0 - buffer_frac), n_iter - 1)
        ax.set_xlim(0, xmax)


def plot_convergence_by_starting_material(mcts_materials_root: Path, out_dir: Path):
    """Compare convergence dynamics across different starting materials.

    Reuses sensitivity_studies/results/starting_material_sweep_max_undiscounted/convergence_data.csv
    (generated by this directory's own sweep_starting_material.py, which adds
    gamma=NORMALIZED_GAMMA to every replicate - the calibrated study's
    starting_material_sweep/ uses gamma=0.0001 and is NOT reused here) instead
    of re-running MCTS inline: that sweep uses this study's exact composite
    reward (rollout_method='ehull_rdos', beta=1.0, gamma=NORMALIZED_GAMMA),
    varying only transition_metal/group_iv, with 5 seeds per starting
    material. Each line below is the mean best-composite-so-far across those
    5 seeds, with a shaded 10th-90th percentile band (not mean +/- std: std is
    a parametric estimate that can extend past the best value any seed
    actually reached, which would misleadingly suggest the search found
    something better than it did. A percentile band is computed directly from
    the observed seeds, so it can never exceed what was actually seen).

    Each legend entry also reports the move-graph edit distance from that
    starting material to the true global-best U-only compound under
    NORMALIZED_GAMMA (per compute_global_u_only_ranks - this is UTi6Sn6, not
    the calibrated study's UZr6Pb6, so the ladder of starting materials here
    is its own Cr/Fe/Ni/Pt6Sn6U set chosen to land on d=2/4/6/8 to THIS
    target, not the calibrated study's V/Ru/Pd/Cu6Ge6U set), to show that
    starting further away from the optimum is *why* that material converges
    slower.
    """
    sweep_csv = (mcts_materials_root / 'sensitivity_studies' / 'results'
                 / 'starting_material_sweep_max_undiscounted' / 'convergence_data.csv')
    if not sweep_csv.exists():
        print('Skipping convergence_by_starting_material: missing', sweep_csv)
        return

    global_ranks = compute_global_u_only_ranks(mcts_materials_root)
    best_key = next((k for k, r in global_ranks.items() if r == 1), None)
    target_tm, target_giv = (None, None)
    if best_key is not None:
        target_tm = next((e for e in best_key if e in TRANSITION_METALS), None)
        target_giv = next((e for e in best_key if e in GROUP_IV), None)

    df = pd.read_csv(sweep_csv)
    value_order = list(df['value'].unique())  # preserve the sweep's own (graph-distance) ordering

    def _pctl_agg(s):
        return pd.Series({'mean': s.mean(), 'p10': np.percentile(s, 10), 'p90': np.percentile(s, 90)})

    stats = df.groupby(['value', 'iteration'])['best_reward'].apply(_pctl_agg).unstack().reset_index()

    fig, ax = plt.subplots(figsize=(4, 3.2))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    curves_for_xlim = []
    for color, value in zip(colors, value_order):
        g = stats[stats['value'] == value].sort_values('iteration').reset_index(drop=True)
        # Keep iteration 0 in the array used for x-axis truncation below, so
        # its index stays aligned with real iteration numbers; the sentinel
        # best_reward=-10.0 at iteration 0 is excluded only from the drawn
        # line/band so it doesn't dominate the y-axis.
        curves_for_xlim.append(g['mean'].to_numpy())

        g_plot = g[g['iteration'] >= 1]
        formula = value.split(' (')[0]
        label = value.split('(')[1].rstrip(')') if '(' in value else value
        if target_tm and target_giv:
            tm, giv = _parse_tm_giv(formula)
            if tm and giv:
                d = _edit_distance_to_target(tm, giv, target_tm, target_giv)
                label = f"d={d}"
        ax.plot(g_plot['iteration'], g_plot['mean'], lw=1.8, color=color, label=label)
        ax.fill_between(g_plot['iteration'], g_plot['p10'], g_plot['p90'],
                         color=color, alpha=0.2, linewidth=0)

    _set_plateau_xlim(ax, curves_for_xlim)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Composite Score')
    ax.legend(fontsize=7, title='Edit distance to global best', title_fontsize=7, loc='lower left')
    plt.tight_layout()
    out = out_dir / 'convergence_by_starting_material.png'
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parents[2]
    out_dir = script_dir

    # Ensure a subsidiary figures directory exists; we'll move all generated PNGs here
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: compute composite scores and save CSVs
    all_compounds_csv = out_dir / 'all_compounds.csv'
    if not all_compounds_csv.exists():
        print('Error: all_compounds.csv not found in analysis directory')
        sys.exit(1)

    df = pd.read_csv(all_compounds_csv)
    df_sorted = compute_composite(df)
    df_sorted.to_csv(out_dir / 'all_compounds_by_composite_score.csv', index=False)
    df_sorted.head(10).to_csv(out_dir / 'top10_compounds_by_composite_score.csv', index=False)
    print('Saved composite CSVs')

    # Supplementary run from a starting material d=5 (out of max 9) from the
    # true global best - far enough that search quality/coverage is visibly
    # imperfect, without being the single most extreme case (see
    # describe_mcts_run_starting_material / sensitivity_studies discussion).
    # If present, keep its composite CSV in sync too.
    d5_compounds_csv = out_dir / 'd5_start_run' / 'all_compounds.csv'
    if d5_compounds_csv.exists():
        df_d5 = pd.read_csv(d5_compounds_csv)
        compute_composite(df_d5).to_csv(
            d5_compounds_csv.parent / 'all_compounds_by_composite_score.csv', index=False)
        print('Saved d5_start_run composite CSV')

    # write LaTeX table of top 15 - sourced from the same d=5-starting-
    # material run (d5_start_run/) as the ehull_vs_rdos Top-10 overlay
    # and the radial tree, so all three reflect one consistent MCTS search
    # rather than mixing in the separate main Cr6Sn6U run.
    table_df_sorted = df_sorted
    if d5_compounds_csv.exists():
        table_df_sorted = compute_composite(pd.read_csv(d5_compounds_csv))
    try:
        write_top15_table(table_df_sorted, out_dir, repo_root)
    except Exception:
        pass

    # Generate figures
    # ehull_vs_rdos's "Top 10 (MCTS)" overlay comes from the d=5-starting-
    # material supplementary run (see d5_start_run/), not this study's main
    # Cr6Sn6U run, so the figure shows search quality from a starting point
    # far enough from the optimum that coverage gaps are visible. Falls back
    # to the main run if that supplementary run hasn't been generated yet.
    d5_start_dir = out_dir / 'd5_start_run'
    mcts_run_dir = d5_start_dir if (d5_start_dir / 'mcts_object.pkl').exists() else out_dir
    start_label, start_dist = describe_mcts_run_starting_material(mcts_run_dir, script_dir.parents[1])
    # Shared with create_composite_radial_tree.py (a separate subprocess, so it
    # can't just call describe_mcts_run_starting_material() itself without a
    # circular import) via this small sidecar file.
    if start_label is not None:
        with open(mcts_run_dir / 'starting_material_info.json', 'w') as f:
            json.dump({'label': start_label, 'distance': start_dist}, f)
    plot_ehull_vs_rdos(repo_root, out_dir, mcts_run_dir=mcts_run_dir,
                       start_label=start_label, start_dist=start_dist)
    plot_convergence_by_starting_material(script_dir.parents[1], out_dir)

    # Generate radial tree via existing script (it writes PNG). Uses the same
    # d=5-starting-material run as the ehull_vs_rdos Top-10 overlay above.
    subprocess.run([sys.executable, str(out_dir / 'create_composite_radial_tree.py'),
                     '--run-dir', str(mcts_run_dir)], check=False)

    # Move any PNGs generated into the subsidiary figures folder
    for p in out_dir.glob('*.png'):
        try:
            target = figures_dir / p.name
            # Overwrite if exists
            if target.exists():
                target.unlink()
            p.rename(target)
        except Exception:
            pass

    print('\nAll figures written to:', figures_dir)


if __name__ == '__main__':
    main()

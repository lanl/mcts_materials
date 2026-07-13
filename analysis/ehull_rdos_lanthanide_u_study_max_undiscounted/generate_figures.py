#!/usr/bin/env python3
"""Generate the top-15 table for the lanthanide+U max-undiscounted study.

Uses the same NORMALIZED_GAMMA and reward formula as the U-only
max_undiscounted study. DOS rewards are looked up per f-block element
(e.g. Ce-Ge-Fe, not just the U row), matching how the MCTS itself scores
compounds during the search. True Rank is within the full lanthanide+U
exhaustive design space (lanthanides La-Lu + U only; other actinides excluded).

Only produces the LaTeX table — no figures (no experimental validation is
available for the lanthanide+U expanded design space).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mcts_crystal'))

NORMALIZED_GAMMA = 1.0 / 2516.1664410449775

# Lanthanides (La-Lu) + U — the lanthanides_u f-block mode.
# Other actinides (Th, Pa, Np, Pu, Ac) are excluded from True Rank computation.
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
    """Return (f_block_elem, tm_elem, giv_elem) for an RM6X6 formula, or
    (None, None, None) if the formula cannot be parsed."""
    elems = _parse_elements(name)
    re_e = next((e for e in elems if e in LANTHANIDES_U), None)
    tm_e = next((e for e in elems if e in TRANSITION_METALS), None)
    giv_e = next((e for e in elems if e in GROUP_IV), None)
    return re_e, tm_e, giv_e


def _tm_giv_key(name: str) -> tuple:
    """(TM, GroupIV) element-set key — used for global-rank deduplication."""
    _, tm, giv = _decompose(name)
    if tm and giv:
        return tuple(sorted([tm, giv]))
    return ()


def load_dos_rewards(repo_root: Path) -> dict:
    """Return {(f_block, giv, tm) -> raw r_DOS} from doscar peaks data.
    Keys match the 'R-GIV-TM' format in DoscarRewardLookup.rewards_dict.
    """
    from mcts_crystal.doscar_utils import DoscarRewardLookup
    peaks_csv = repo_root / 'doscar_peaks_data_with_U.csv'
    return DoscarRewardLookup(peaks_file=str(peaks_csv)).rewards_dict


def _lookup_dos(name: str, dos_rewards: dict) -> float:
    """Look up the element-specific r_DOS for a compound by its formula."""
    re_e, tm_e, giv_e = _decompose(name)
    if re_e and giv_e and tm_e:
        key = f"{re_e}-{giv_e}-{tm_e}"
        if key in dos_rewards:
            return float(dos_rewards[key])
    return 0.0


def compute_composite(df: pd.DataFrame, dos_rewards: dict) -> pd.DataFrame:
    df = df.copy()
    if 'name' not in df.columns and 'formula' in df.columns:
        df['name'] = df['formula']
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
    df['weighted_r_DOS'] = NORMALIZED_GAMMA * df['r_DOS']
    df['composite_score'] = df['ehull_reward'] + df['weighted_r_DOS']
    return df.sort_values('composite_score', ascending=False).reset_index(drop=True)


def compute_global_ranks(repo_root: Path, dos_rewards: dict) -> dict:
    """Rank all lanthanide+U compounds in the full high-throughput CSV.
    Returns {(f_block, tm, giv) -> rank} with rank 1 = best composite score.
    Keyed by decomposed elements so formula-ordering differences don't matter."""
    mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
    if not mace_csv.exists():
        return {}
    df = pd.read_csv(mace_csv)
    df['re_elem'] = df['name'].apply(lambda n: next(
        (e for e in _parse_elements(n) if e in LANTHANIDES_U), None))
    df = df[df['re_elem'].notna()].copy()
    df['r_DOS'] = df['name'].apply(lambda n: _lookup_dos(n, dos_rewards))
    df['composite_score'] = df['e_above_hull'].apply(ehull_reward) + NORMALIZED_GAMMA * df['r_DOS']
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    ranks = {}
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        key = _decompose(row['name'])
        if key != (None, None, None):
            ranks[key] = rank
    return ranks


def format_name(name: str) -> str:
    """Format RM6X6 name as RTM$_6$GIV$_6$ for LaTeX."""
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


def write_top15_table(df_sorted: pd.DataFrame, global_ranks: dict, out_dir: Path,
                      multistart: bool = False) -> None:
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    top15 = df_sorted.head(15).copy()
    eol = ' \\\\\n'

    stem = 'top15_lanthanide_u_multistart' if multistart else 'top15_lanthanide_u'
    tex_path = tables_dir / f'{stem}.tex'
    with open(tex_path, 'w') as f:
        if multistart:
            f.write(f'% Top 15 compounds pooled across all start directories. gamma={NORMALIZED_GAMMA:g}.\n')
        else:
            f.write(f'% Top 15 compounds (lanthanide+U study). gamma={NORMALIZED_GAMMA:g}.\n')
        f.write('% True Rank = rank within the full 1620-compound lanthanide+U exhaustive design space.\n')
        f.write('\\begin{tabular}{rrlrrrr}\n')
        f.write('\\toprule\n')
        f.write('MCTS Rank & True Rank & Name & $E_{\\mathrm{Hull}}$ (eV/atom) & '
                '$r_{E_{\\mathrm{Hull}}}$ & $\\alpha_{\\mathrm{DOS}} \\cdot r_{\\mathrm{DOS}}$ '
                '& Composite' + eol)
        f.write('\\midrule\n')
        for mcts_rank, (_, row) in enumerate(top15.iterrows(), start=1):
            name = row.get('name', row.get('formula', ''))
            true_rank = global_ranks.get(_decompose(str(name)), '--')
            ehull = float(row['e_above_hull']) if pd.notna(row.get('e_above_hull')) else float('nan')
            ehull_r = float(row['ehull_reward']) if pd.notna(row.get('ehull_reward')) else 0.0
            rdos = float(row['weighted_r_DOS']) if pd.notna(row.get('weighted_r_DOS')) else 0.0
            comp = float(row['composite_score']) if pd.notna(row.get('composite_score')) else 0.0
            f.write(f"{mcts_rank} & {true_rank} & {format_name(name)} & "
                    f"{ehull:.4f} & {ehull_r:.4f} & {rdos:.4f} & {comp:.4f}" + eol)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    print('Wrote LaTeX table:', tex_path)


def load_all_compounds(study_dir: Path) -> tuple[pd.DataFrame, bool]:
    """Load compounds from all available start directories and the root CSV.

    Returns (combined_df, multistart_flag). If subdirectory CSVs exist, pools
    them together with the root CSV (if present) and returns multistart=True.
    """
    start_dirs = sorted(p for p in study_dir.iterdir()
                        if p.is_dir() and p.name.endswith('_start'))
    frames = []
    for src in [study_dir] + start_dirs:
        csv = src / 'all_compounds.csv'
        if csv.exists():
            sub = pd.read_csv(csv)
            if 'name' not in sub.columns and 'formula' in sub.columns:
                sub['name'] = sub['formula']
            frames.append(sub)

    if not frames:
        print('Error: no all_compounds.csv found — run run_study.sh first.')
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset='name').reset_index(drop=True)
    return combined, len(start_dirs) > 0


def main():
    study_dir = Path(__file__).parent
    repo_root = study_dir.parents[1]

    dos_rewards = load_dos_rewards(repo_root)
    df, multistart = load_all_compounds(study_dir)
    df_sorted = compute_composite(df, dos_rewards)
    df_sorted.to_csv(study_dir / 'all_compounds_by_composite_score.csv', index=False)
    df_sorted.head(15).to_csv(study_dir / 'top15_compounds_by_composite_score.csv', index=False)
    sources = 'pooled multi-start' if multistart else 'single-start'
    print(f'Ranked {len(df_sorted)} unique compounds ({sources}).')

    global_ranks = compute_global_ranks(repo_root, dos_rewards)
    print(f'Global design space: {len(global_ranks)} lanthanide+U compounds.')

    write_top15_table(df_sorted, global_ranks, study_dir, multistart=multistart)


if __name__ == '__main__':
    main()

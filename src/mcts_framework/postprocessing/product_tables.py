"""
Generate top-N recommendation tables for product-mode studies.

Creates both LaTeX (.tex) and plain text (.txt) versions of the top-15 table
with proper compound name formatting (e.g., UTi$_{6}$Sn$_{6}$).

© 2026. Triad National Security, LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import Callable, Hashable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.config import Config
from ..intermetallic import ehull_reward
from .design_space import full_formula_key, rank_design_space, score_by_method


def _elem_set(name) -> set:
    """Set of element symbols in a formula/compound name."""
    if pd.isna(name):
        return set()
    s = str(name)
    if "-" in s:
        return {p.capitalize() for p in re.split(r"[^A-Za-z]", s) if p}
    return set(re.findall(r"[A-Z][a-z]?", s))


def _load_attempted_sets(attempted_path: Optional[str]) -> List[set]:
    """Load element sets of experimentally-attempted compounds."""
    if not attempted_path:
        return []
    p = Path(attempted_path)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, sep=r"\s+", comment="#", header=None,
                         names=["name", "e_form", "e_hull"])
        return [_elem_set(n) for n in df["name"]]
    except Exception:
        return []


def _latex_formula(name: str) -> str:
    """
    Convert compound name to LaTeX format with subscripts.

    Examples:
        Sn6Ti6U -> UTi$_{6}$Sn$_{6}$
        Mn6Sn6U -> UMn$_{6}$Sn$_{6}$
    """
    # Extract formula from full identifier if needed
    formula = name.split("|")[0] if "|" in name else name

    # Parse element-count pairs
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)

    # Group by element
    elem_counts = {}
    for elem, count in matches:
        if not elem:
            continue
        count = int(count) if count else 1
        elem_counts[elem] = elem_counts.get(elem, 0) + count

    # Order: U first, then alphabetically
    sorted_elems = []
    if 'U' in elem_counts:
        sorted_elems.append(('U', elem_counts['U']))
        del elem_counts['U']
    sorted_elems.extend(sorted(elem_counts.items()))

    # Build LaTeX string
    parts = []
    for elem, count in sorted_elems:
        if count == 1:
            parts.append(elem)
        else:
            parts.append(f"{elem}$_{{{count}}}$")

    return "".join(parts)


def write_product_mode_table(
    df: pd.DataFrame,
    out_path: str,
    config: Config,
    n: int = 15,
    synthesized: Optional[Sequence[str]] = None,
    attempted_path: Optional[str] = None,
    study_label: str = "",
    key_fn: Callable[[str], Hashable] = full_formula_key,
    space_filter: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Write top-N product-mode table in LaTeX format (.tex).

    Similar to write_top_n_table but with product-mode specific formatting.
    """
    ic = config.intermetallic
    if ic is None:
        raise ValueError("write_product_mode_table requires intermetallic Config")

    rollout_method = ic.rollout_method
    gamma = ic.gamma
    beta = ic.beta
    mace_cache = ic.cache_path
    doscar_peaks = ic.doscar_data_path

    if mace_cache is None or doscar_peaks is None:
        raise ValueError("config.intermetallic must set cache_path and doscar_data_path")

    df = df.copy()
    if "name" not in df.columns:
        df["name"] = df.get("formula", df.get("identifier"))

    # Load design space to get r_DOS values
    from .design_space import load_design_space
    _, doscar_lookup = load_design_space(mace_cache, doscar_peaks)

    # Compute r_DOS from doscar lookup if not present
    if "r_DOS" not in df.columns:
        def get_rdos(name):
            # Extract formula from full identifier
            formula = name.split("|")[0] if "|" in name else name
            return doscar_lookup.get_reward(formula)
        df["r_DOS"] = df["name"].apply(get_rdos)

    df["ehull_reward"] = df["e_above_hull"].apply(ehull_reward)
    df["reward"] = df.apply(
        lambda r: score_by_method(
            rollout_method, r["e_above_hull"], r["r_DOS"], beta, gamma
        ),
        axis=1,
    )

    df_sorted = df.sort_values("reward", ascending=False).reset_index(drop=True)
    top = df_sorted.head(n).copy()

    global_ranks = rank_design_space(
        mace_cache, doscar_peaks, rollout_method, gamma, beta,
        key_fn=key_fn, space_filter=space_filter,
    )

    synth_sets = [_elem_set(s) for s in (synthesized or [])]
    attempted_sets = _load_attempted_sets(attempted_path)

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        name = r.get("name", r.get("formula", ""))
        es = _elem_set(name)

        if any(es == s for s in synth_sets):
            synth = "Yes"
        elif any(es == s for s in attempted_sets):
            synth = "No"
        else:
            synth = "--"

        rows.append({
            'mcts_rank': rank,
            'true_rank': global_ranks.get(key_fn(name)),
            'name': str(name),
            'latex_name': _latex_formula(name),
            'e_hull': float(r.get("e_above_hull", np.nan)),
            'r_ehull': float(r.get("ehull_reward", 0.0) or 0.0),
            'r_dos': float(r.get("r_DOS", 0.0) or 0.0),
            'product': float(r.get("reward", 0.0) or 0.0),
            'synth': synth,
        })

    # Write LaTeX table
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write(f"% Top-{n} {study_label} compounds, product-mode study "
                f"(gamma={gamma:g}, beta={beta:g}).\n")
        f.write(f"% Global rank within design space by product reward r_Ehull x r_DOS.\n")
        f.write("\\begin{tabular}{rrlrrrrc}\n")
        f.write("\\toprule\n")
        f.write("MCTS & True & Compound & $E_{\\mathrm{Hull}}$ & "
                "$r_{E_{\\mathrm{Hull}}}$ & $r_{\\mathrm{DOS}}$ & "
                "Product Reward & Synth \\\\\n")
        f.write("\\midrule\n")

        for row in rows:
            tr = str(row['true_rank']) if row['true_rank'] is not None else "--"
            f.write(f"{row['mcts_rank']} & {tr} & {row['latex_name']} & "
                    f"{row['e_hull']:.4f} & {row['r_ehull']:.4f} & "
                    f"{row['r_dos']:.1f} & {row['product']:.2f} & "
                    f"{row['synth']} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    return str(out)


def write_product_mode_txt_table(
    df: pd.DataFrame,
    out_path: str,
    config: Config,
    n: int = 15,
    synthesized: Optional[Sequence[str]] = None,
    attempted_path: Optional[str] = None,
    study_label: str = "",
    key_fn: Callable[[str], Hashable] = full_formula_key,
    space_filter: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Write top-N product-mode table in plain text format (.txt).

    Creates a human-readable version of the LaTeX table.
    """
    ic = config.intermetallic
    if ic is None:
        raise ValueError("write_product_mode_txt_table requires intermetallic Config")

    rollout_method = ic.rollout_method
    gamma = ic.gamma
    beta = ic.beta
    mace_cache = ic.cache_path
    doscar_peaks = ic.doscar_data_path

    if mace_cache is None or doscar_peaks is None:
        raise ValueError("config.intermetallic must set cache_path and doscar_data_path")

    df = df.copy()
    if "name" not in df.columns:
        df["name"] = df.get("formula", df.get("identifier"))

    # Load design space to get r_DOS values
    from .design_space import load_design_space
    _, doscar_lookup = load_design_space(mace_cache, doscar_peaks)

    # Compute r_DOS from doscar lookup if not present
    if "r_DOS" not in df.columns:
        def get_rdos(name):
            # Extract formula from full identifier
            formula = name.split("|")[0] if "|" in name else name
            return doscar_lookup.get_reward(formula)
        df["r_DOS"] = df["name"].apply(get_rdos)

    df["ehull_reward"] = df["e_above_hull"].apply(ehull_reward)
    df["reward"] = df.apply(
        lambda r: score_by_method(
            rollout_method, r["e_above_hull"], r["r_DOS"], beta, gamma
        ),
        axis=1,
    )

    df_sorted = df.sort_values("reward", ascending=False).reset_index(drop=True)
    top = df_sorted.head(n).copy()

    global_ranks = rank_design_space(
        mace_cache, doscar_peaks, rollout_method, gamma, beta,
        key_fn=key_fn, space_filter=space_filter,
    )

    synth_sets = [_elem_set(s) for s in (synthesized or [])]
    attempted_sets = _load_attempted_sets(attempted_path)

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        name = r.get("name", r.get("formula", ""))
        es = _elem_set(name)

        if any(es == s for s in synth_sets):
            synth = "Yes"
        elif any(es == s for s in attempted_sets):
            synth = "No"
        else:
            synth = "--"

        # Get plain formula
        formula = name.split("|")[0] if "|" in name else name

        rows.append({
            'mcts_rank': rank,
            'true_rank': global_ranks.get(key_fn(name)),
            'formula': formula,
            'e_hull': float(r.get("e_above_hull", np.nan)),
            'r_ehull': float(r.get("ehull_reward", 0.0) or 0.0),
            'r_dos': float(r.get("r_DOS", 0.0) or 0.0),
            'product': float(r.get("reward", 0.0) or 0.0),
            'synth': synth,
        })

    # Write plain text table
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write(f"Top-{n} {study_label} Compounds - Product-Mode Study\n")
        f.write(f"=" * 100 + "\n")
        f.write(f"Parameters: gamma={gamma:g}, beta={beta:g}, "
                f"rollout_method={rollout_method}\n")
        f.write(f"Global rank within design space by product reward (r_Ehull × r_DOS)\n")
        f.write("\n")

        # Header
        f.write(f"{'MCTS':>4} {'True':>4} {'Compound':>15} {'E_Hull':>10} "
                f"{'r_Ehull':>10} {'r_DOS':>10} {'Product':>10} {'Synth':>6}\n")
        f.write(f"{'Rank':>4} {'Rank':>4} {'':>15} {'(eV/atom)':>10} "
                f"{'':>10} {'':>10} {'Reward':>10} {'':>6}\n")
        f.write("-" * 100 + "\n")

        # Data rows
        for row in rows:
            tr = str(row['true_rank']) if row['true_rank'] is not None else "--"
            f.write(f"{row['mcts_rank']:>4} {tr:>4} {row['formula']:>15} "
                    f"{row['e_hull']:>10.4f} {row['r_ehull']:>10.4f} "
                    f"{row['r_dos']:>10.1f} {row['product']:>10.2f} "
                    f"{row['synth']:>6}\n")

        f.write("-" * 100 + "\n")
        f.write(f"\nSynthesis status:\n")
        f.write(f"  Yes = Successfully synthesized\n")
        f.write(f"  No  = Attempted but not synthesized\n")
        f.write(f"  --  = Not attempted\n")

    return str(out)

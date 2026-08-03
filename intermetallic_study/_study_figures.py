"""
Study-specific figure/table helpers for the intermetallic product-mode studies.

These live OUTSIDE the mcts_framework library because they encode
chemistry-specific choices particular to the U-only / lanthanide+U studies:
the synthesized-compound list, the experimental-literature overlay and its data
format, the U-only / lanthanide+U design-space filters, and the f-block-stripped
matching key used to overlay TM/Group-IV experimental literature onto a
lanthanide backdrop. The generic library (mcts_framework.postprocessing) stays
chemistry-agnostic; the study drivers pass these in as arguments (space_filter,
key_fn, synthesized, overlay points).

Both study drivers (u_only/, lanthanide_u/) import from here.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from mcts_framework.intermetallic import ehull_reward
from mcts_framework.postprocessing import (
    full_formula_key,
    load_design_space,
    rank_design_space,
    score_by_method,
)

# --- Study element categories (symbol space) -----------------------------

F_BLOCK = {
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu",
}
_LANTHANIDES = {
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
}

# Experimentally-synthesized U-only compounds (dash form), for the U-only figure.
SYNTHESIZED_COMPOUNDS = ["U-Sn-V", "U-Sn-Nb", "U-Ge-Cr", "U-Ge-Co"]


# --- Study matching keys / filters (kept out of the library) -------------

def formula_key_non_f(name: str) -> Tuple[str, ...]:
    """
    Sorted tuple of NON-f-block elements, e.g. 'Fe6Ge6Er' -> ('Fe', 'Ge').

    This f-block-stripped key deliberately collapses all lanthanide/U variants
    of a TM/Group-IV pair onto one key - used ONLY to overlay experimental
    TM/Group-IV literature onto a lanthanide backdrop. It is study-specific and
    intentionally not part of the library (which keys by full composition).
    """
    parts = re.findall(r"[A-Z][a-z]?", str(name).split("|")[0])
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


def u_only_filter(name: str) -> bool:
    """True if the formula contains U and no other f-block element."""
    elems = set(re.findall(r"[A-Z][a-z]?", str(name)))
    return "U" in elems and not (elems & (F_BLOCK - {"U"}))


def lanthanide_u_filter(name: str) -> bool:
    """True if the formula contains U or any lanthanide."""
    elems = set(re.findall(r"[A-Z][a-z]?", str(name)))
    return "U" in elems or bool(elems & _LANTHANIDES)


def _elem_set(name) -> set:
    """Element symbols in a formula/compound name (plain or dash form)."""
    if pd.isna(name):
        return set()
    s = str(name)
    if "-" in s:
        return {p.capitalize() for p in re.split(r"[^A-Za-z]", s) if p}
    return set(re.findall(r"[A-Z][a-z]?", s))


# --- Experimental-literature overlay (lanthanide+U study) ----------------

def experimental_overlay_points(
    experimental_path: str, cache_path: str, doscar_path: str
) -> Tuple[List[float], List[float]]:
    """
    (r_DOS, e_hull) coordinates for experimental-literature compounds.

    Reads a whitespace file of 'TM GIV R' triples, resolves each to a specific
    lanthanide compound's r_DOS (trying common TM6GIV6R orderings) and its
    e_above_hull from the design space; falls back to the f-block-stripped key.
    Returns ([], []) if the file is absent/unparseable. Study-specific.
    """
    p = Path(experimental_path)
    if not p.exists():
        return [], []

    df_mace, doscar_lookup = load_design_space(cache_path, doscar_path)
    if df_mace is None:
        return [], []
    df_mace = df_mace.assign(name=df_mace.get("name", df_mace.get("formula")))
    if "data_quality" in df_mace.columns:
        df_mace = df_mace[df_mace["data_quality"] == "valid"]

    backdrop = {}
    for _, r in df_mace.iterrows():
        key = formula_key_non_f(r["name"])
        rdos = doscar_lookup.get_reward(str(r["name"]).split("|")[0])
        if key not in backdrop or rdos > backdrop[key][0]:
            backdrop[key] = (rdos, float(r["e_above_hull"]))

    xs, ys = [], []
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            tm, giv, r = parts
            rdos = ehull = None
            for test in (f"{tm}6{giv}6{r}", f"{r}{tm}6{giv}6",
                         f"{tm}{giv}{r}", f"{r}{giv}{tm}"):
                tr = doscar_lookup.get_reward(test)
                if tr > 0:
                    m = df_mace[df_mace["name"] == test]
                    if len(m):
                        rdos, ehull = tr, float(m.iloc[0]["e_above_hull"])
                        break
            if rdos is None or ehull is None:
                hit = backdrop.get(tuple(sorted(p for p in (r, tm, giv) if p not in F_BLOCK)))
                if hit is not None:
                    rdos = hit[0] if rdos is None else rdos
                    ehull = hit[1] if ehull is None else ehull
            if rdos is not None and ehull is not None:
                xs.append(rdos)
                ys.append(ehull)
    return xs, ys


# --- Plain-text top-N table (study convenience) --------------------------

def write_txt_table(
    df: pd.DataFrame,
    out_path: str,
    config,
    n: int = 15,
    study_label: str = "",
    space_filter=None,
) -> str:
    """
    Human-readable plain-text top-N table (the study's .txt recommendation list).

    Scores by the run's reward via the library's score_by_method, and ranks the
    design space via rank_design_space - same numbers as the LaTeX table, plain
    layout. Kept here (not in the library) as a study convenience.
    """
    ic = config.intermetallic
    rollout_method, gamma, beta = ic.rollout_method, ic.gamma, ic.beta
    _, doscar_lookup = load_design_space(ic.cache_path, ic.doscar_data_path)

    df = df.copy()
    if "name" not in df.columns:
        df["name"] = df.get("formula", df.get("identifier"))
    if "r_DOS" not in df.columns:
        df["r_DOS"] = df["name"].apply(
            lambda x: doscar_lookup.get_reward(str(x).split("|")[0])
        )
    df["reward"] = df.apply(
        lambda r: score_by_method(rollout_method, r["e_above_hull"], r["r_DOS"], beta, gamma),
        axis=1,
    )
    top = df.sort_values("reward", ascending=False).head(n).reset_index(drop=True)
    ranks = rank_design_space(ic.cache_path, ic.doscar_data_path, rollout_method,
                              gamma, beta, space_filter=space_filter)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"Top-{n} {study_label} Compounds - Product-Mode Study\n")
        f.write("=" * 100 + "\n")
        f.write(f"Parameters: gamma={gamma:g}, beta={beta:g}, "
                f"rollout_method={rollout_method}\n\n")
        f.write(f"{'MCTS':>4} {'True':>4} {'Compound':>15} {'E_Hull':>10} "
                f"{'r_Ehull':>10} {'r_DOS':>10} {'Reward':>10}\n")
        f.write("-" * 100 + "\n")
        for rank, (_, r) in enumerate(top.iterrows(), start=1):
            name = str(r["name"]).split("|")[0]
            tr = ranks.get(full_formula_key(name))
            f.write(f"{rank:>4} {str(tr) if tr else '--':>4} {name:>15} "
                    f"{float(r['e_above_hull']):>10.4f} "
                    f"{ehull_reward(float(r['e_above_hull'])):>10.4f} "
                    f"{float(r['r_DOS']):>10.1f} {float(r['reward']):>10.2f}\n")
        f.write("-" * 100 + "\n")
    return str(out)

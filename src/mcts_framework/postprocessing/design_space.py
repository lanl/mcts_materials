"""
Design-space loading, composite scoring, and ranking for post-run analysis.

Operates on FINISHED run outputs plus the high-throughput MACE cache and DOSCAR
peak data. Everything is parameterized (gamma, beta, paths), and nothing here
assumes a particular chemistry (no U-only, f-block, or ternary special-casing) -
so one implementation serves every study variant and extends to molecules.

Reuses the framework's own physics: ehull_reward and DoscarRewardLookup from
mcts_framework.intermetallic (identical to the values the search itself uses).

© 2026. Triad National Security, LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import Callable, Dict, Hashable, Optional, Tuple

import pandas as pd

from ..intermetallic import DoscarRewardLookup, ehull_reward


def full_formula_key(name) -> str:
    """
    Composition key that fully distinguishes compounds: a normalized formula
    with element counts, order-independent. e.g. 'Fe6Ge6U' and 'UFe6Ge6' both
    map to 'Fe6Ge6U1'. Distinct compositions never collide.

    This is the DEFAULT ranking key. For design spaces where composition alone
    does not distinguish materials (multiple structure templates, or molecules),
    pass a custom key_fn to the ranking functions - e.g. one returning
    IntermetallicStructure.get_identifier() (formula|SG|Wyckoff) or a molecule's
    canonical SMILES. The ranking logic is agnostic to what the key is.
    """
    counts: Dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(name)):
        if el:
            counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def score_by_method(
    rollout_method: str,
    e_above_hull: float,
    r_dos: float,
    beta: float = 1.0,
    gamma: float = 0.0001,
) -> float:
    """
    Reproduce a run's reward for a single compound, dispatching on the rollout
    method the run used, so post-run ranking scores compounds by the same
    formula the search optimized.

    Mirrors the intermetallic reward classes exactly:
        ehull              -> ehull_reward(e_hull)
        rdos               -> r_DOS                       (raw, unweighted)
        ehull_rdos         -> beta * ehull_reward + gamma * r_DOS
        ehull_rdos_product -> ehull_reward * r_DOS        (no gamma)

    A test pins this to the reward classes so the two cannot drift.
    """
    ehull_term = ehull_reward(e_above_hull)
    if rollout_method == "ehull":
        return ehull_term
    if rollout_method == "rdos":
        return r_dos
    if rollout_method == "ehull_rdos":
        return beta * ehull_term + gamma * r_dos
    if rollout_method == "ehull_rdos_product":
        return ehull_term * r_dos
    raise ValueError(f"Unknown rollout_method: {rollout_method!r}")


def load_design_space(
    mace_cache: str,
    doscar_peaks: str,
) -> Tuple[Optional[pd.DataFrame], DoscarRewardLookup]:
    """
    Load the high-throughput MACE cache and the DOSCAR rDOS lookup.

    Returns (df_mace, doscar_lookup):
      - df_mace: the MACE results CSV (columns include name, e_above_hull), or
        None if the file is missing/unreadable.
      - doscar_lookup: a DoscarRewardLookup that resolves each compound's own
        rDOS directly from its full formula via get_reward(formula) - the same
        object and lookup path the search itself uses. rDOS is per-compound;
        there is no collapsing across f-block substitutions or fallback between
        different compounds.
    """
    df_mace = None
    mace_path = Path(mace_cache)
    if mace_path.exists():
        try:
            df_mace = pd.read_csv(mace_path)
        except Exception:
            df_mace = None

    return df_mace, DoscarRewardLookup(peaks_file=str(doscar_peaks))


def rank_design_space(
    mace_cache: str,
    doscar_peaks: str,
    rollout_method: str,
    gamma: float = 0.0001,
    beta: float = 1.0,
    key_fn: Callable[[str], Hashable] = full_formula_key,
    space_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[Hashable, int]:
    """
    Rank a design space by the run's reward so MCTS results can be checked
    against the true best-in-space compounds (search-coverage check).

    Args:
        mace_cache, doscar_peaks: data-file paths.
        rollout_method: the reward the run optimized ('ehull', 'rdos',
            'ehull_rdos', 'ehull_rdos_product'). Compounds are scored by that
            same formula (via score_by_method) so the ranking cannot disagree
            with the run it describes.
        gamma, beta: composite-score weights (used only by 'ehull_rdos').
        key_fn: maps a compound's formula/name to the key used to identify it
            in the returned ranking. Defaults to full_formula_key (full
            composition). Pass a structure- or molecule-aware key (e.g.
            Material.get_identifier) for design spaces where composition alone
            does not distinguish materials.
        space_filter: optional predicate on the formula string; only compounds
            for which it returns True are ranked. Default None = rank every
            compound in the MACE cache.

    Returns:
        {key: rank} with rank 1 = best score. Ties keep first-seen order (stable
        sort). rDOS is attached per-compound via the full formula.
    """
    df_mace, doscar_lookup = load_design_space(mace_cache, doscar_peaks)
    if df_mace is None or not len(df_mace):
        return {}

    df = df_mace.copy()
    df["name"] = df.get("name", df.get("formula"))
    if space_filter is not None:
        df = df[df["name"].apply(space_filter)].copy()
    # rDOS is resolved per-compound from the full formula - identical to the
    # lookup the search uses; no collapsing across f-block substitutions.
    df["r_dos"] = df["name"].apply(doscar_lookup.get_reward)
    df["score"] = df.apply(
        lambda r: score_by_method(
            rollout_method, r["e_above_hull"], r["r_dos"], beta, gamma
        ),
        axis=1,
    )
    df = df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    ranks: Dict[Hashable, int] = {}
    for rank, name in enumerate(df["name"], start=1):
        k = key_fn(name)
        ranks.setdefault(k, rank)  # first (best) occurrence wins on collision
    return ranks

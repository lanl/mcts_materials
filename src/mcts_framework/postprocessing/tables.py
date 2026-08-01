"""
LaTeX top-N candidate tables for post-run analysis.

The table length N is a parameter (top-10 / top-15 / top-20 / ...), and the
composite-score weights (gamma, beta) and data paths are read from the run's own
Config rather than re-specified by the caller. One implementation serves every
study; nothing here is chemistry-specific.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import Callable, Hashable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.config import Config
from ..intermetallic import ehull_reward
from .design_space import (
    full_formula_key,
    load_design_space,
    rank_design_space,
    score_by_method,
)


def _elem_set(name) -> set:
    """Set of element symbols in a formula/compound name (both plain and dash form)."""
    if pd.isna(name):
        return set()
    s = str(name)
    if "-" in s:
        return {p.capitalize() for p in re.split(r"[^A-Za-z]", s) if p}
    return set(re.findall(r"[A-Z][a-z]?", s))


def _load_attempted_sets(attempted_path: Optional[str]) -> List[set]:
    """Element sets of experimentally-attempted compounds (compounds_filtered.dat),
    used to distinguish 'No' (attempted, not synthesized) from '-' (never attempted)."""
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
    Render a compound name with LaTeX count subscripts, U first then alphabetical.

    e.g. 'Sn6Ti6U' -> 'UTi$_{6}$Sn$_{6}$'. Strips any '|SG|Wyckoff' identifier
    suffix first. Order (U-first) matches the original product-mode tables.
    """
    formula = str(name).split("|")[0]
    counts: dict = {}
    for elem, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem:
            counts[elem] = counts.get(elem, 0) + (int(count) if count else 1)
    ordered = []
    if "U" in counts:
        ordered.append(("U", counts.pop("U")))
    ordered.extend(sorted(counts.items()))
    return "".join(e if c == 1 else f"{e}$_{{{c}}}$" for e, c in ordered)


def write_top_n_table(
    df: pd.DataFrame,
    out_path: str,
    config: Config,
    n: int = 15,
    synthesized: Optional[Sequence[str]] = None,
    attempted_path: Optional[str] = None,
    study_label: str = "",
    key_fn: Callable[[str], Hashable] = full_formula_key,
    space_filter: Optional[Callable[[str], bool]] = None,
    latex_names: bool = False,
) -> str:
    """
    Write a LaTeX table of the top-`n` compounds from a run's results.

    The composite-score weights (gamma, beta) and the design-space data paths
    (MACE cache, DOSCAR peaks) are taken from the run's own Config - the same
    values the search used - so the analysis cannot silently disagree with the
    run it describes.

    Args:
        df: A run's compounds DataFrame. Must have a name/formula column and
            e_above_hull; r_DOS / dos_reward and composite columns are used if
            present, otherwise recomputed from the config's gamma/beta.
        out_path: Where to write the .tex file.
        config: the run's Config. gamma, beta, cache_path (MACE cache) and
            doscar_data_path (DOSCAR peaks) are read from config.intermetallic.
        n: table length (10, 15, 20, ...). Configurable per call.
        synthesized: dash-form compound names known to be synthesized ('Yes').
        attempted_path: path to compounds_filtered.dat (attempted -> 'No',
            else '-').
        study_label: short label written into the table's comment header.
        key_fn: how to identify a compound for the True-Rank column. Defaults to
            full_formula_key (full composition), so distinct compounds never
            collide. Pass a structure/molecule identifier for design spaces
            where composition alone doesn't distinguish materials.
        space_filter: predicate selecting which compounds form the ranking
            design space. Default None ranks against every compound in the MACE
            cache; pass a predicate (e.g. a U-only filter) to restrict it.
        latex_names: if True, render the Compound column with LaTeX count
            subscripts (U first), e.g. 'UTi$_6$Sn$_6$'; default False writes the
            plain formula.

    Returns:
        The path written (str).
    """
    ic = config.intermetallic
    if ic is None:
        raise ValueError("write_top_n_table requires an intermetallic Config section")
    rollout_method = ic.rollout_method
    gamma = ic.gamma
    beta = ic.beta
    mace_cache = ic.cache_path
    doscar_peaks = ic.doscar_data_path
    if mace_cache is None or doscar_peaks is None:
        raise ValueError(
            "config.intermetallic must set cache_path (MACE cache) and "
            "doscar_data_path (DOSCAR peaks) for the True-Rank column"
        )

    df = df.copy()
    if "name" not in df.columns:
        df["name"] = df.get("formula")

    # Recompute r_DOS, r_ehull, and the run's reward from the config's method so
    # the table always agrees with the run (rather than trusting a pre-existing
    # score column that may have used a different formula).
    #
    # r_DOS is recovered self-sufficiently: prefer an existing r_DOS/dos_reward
    # column, else look it up per-compound from the run's DOSCAR data (the same
    # lookup the search uses). Without this fallback, calling this function on a
    # tree-derived df with no r_DOS column would silently score every compound
    # with r_DOS=0 - zeroing any rDOS-dependent reward. The name may be a full
    # "formula|SG|Wyckoff" identifier, so the lookup splits on "|".
    if "r_DOS" not in df.columns and "dos_reward" in df.columns:
        df["r_DOS"] = df["dos_reward"]
    if "r_DOS" not in df.columns:
        _, doscar_lookup = load_design_space(mace_cache, doscar_peaks)
        df["r_DOS"] = df["name"].apply(
            lambda n: doscar_lookup.get_reward(str(n).split("|")[0])
        )
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
        # Strip any "|SG|Wyckoff" identifier suffix before element-set matching
        # so 'SG'/Wyckoff letters aren't read as spurious elements (which would
        # break the synthesized/attempted match). full_formula_key strips it too.
        formula = str(name).split("|")[0]
        es = _elem_set(formula)
        if any(es == s for s in synth_sets):
            synth = "Yes"
        elif any(es == s for s in attempted_sets):
            synth = "No"
        else:
            synth = "--"
        rows.append((
            rank,
            global_ranks.get(key_fn(name)),
            _latex_formula(name) if latex_names else formula,
            float(r.get("e_above_hull", np.nan)),
            float(r.get("ehull_reward", 0.0) or 0.0),
            float(r.get("r_DOS", 0.0) or 0.0),
            float(r.get("reward", 0.0) or 0.0),
            synth,
        ))

    space_desc = "filtered" if space_filter is not None else "full"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    eol = " \\\\\n"
    with open(out, "w") as f:
        f.write(f"% Top {n} compounds{(' (' + study_label + ')') if study_label else ''}. "
                f"rollout_method={rollout_method}, gamma={gamma:g}, beta={beta:g}. "
                f"Reward is the run's reward; True Rank = rank within the "
                f"{len(global_ranks)}-compound {space_desc} design space.\n")
        f.write("\\begin{tabular}{rrlrrrrc}\n")
        f.write("\\toprule\n")
        f.write("MCTS Rank & True Rank & Compound & $E_{\\mathrm{Hull}}$ & "
                "$r_{E_{\\mathrm{Hull}}}$ & $r_{\\mathrm{DOS}}$ & Reward & "
                "Synthesized?" + eol)
        f.write("\\midrule\n")
        for rank, gr, name, ehull, ehull_r, rdos, reward, synth in rows:
            gr_s = str(gr) if gr is not None else "--"
            f.write(f"{rank} & {gr_s} & {name} & {ehull:.4f} & {ehull_r:.4f} & "
                    f"{rdos:.1f} & {reward:.2f} & {synth}" + eol)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return str(out)

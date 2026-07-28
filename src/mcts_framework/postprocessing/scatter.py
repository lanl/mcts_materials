"""
E_hull vs weighted-rDOS scatter for post-run analysis.

Plots the full design space (E_hull against r_DOS) as a grey backdrop and
overlays a run's top-N compounds, so you can see where the search's picks land
relative to every candidate. The data paths and reward settings
(rollout_method / beta / gamma) are read from the run's own Config; the top-N
overlay is ranked by the SAME reward the run optimized (via score_by_method),
so it agrees with the run and with the table. Those settings only pick which
compounds are highlighted, not the axes. The ranking key and any design-space
filter are pluggable, so nothing here is chemistry-specific. Optional
synthesized / attempted overlays highlight experimentally-known compounds when
a list is supplied.

matplotlib is imported lazily so importing this module does not require the
[viz] extra.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import Callable, Hashable, Optional, Sequence

import pandas as pd

from ..core.config import Config
from .design_space import full_formula_key, load_design_space, score_by_method


def _elem_set(name) -> set:
    """Set of element symbols in a formula/compound name (plain or dash form)."""
    if pd.isna(name):
        return set()
    s = str(name)
    if "-" in s:
        return {p.capitalize() for p in re.split(r"[^A-Za-z]", s) if p}
    return set(re.findall(r"[A-Z][a-z]?", s))


def _load_attempted_sets(attempted_path: Optional[str]) -> list:
    """Element sets of experimentally-attempted compounds (compounds_filtered.dat)."""
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


def plot_ehull_vs_rdos(
    run_df: pd.DataFrame,
    out_path: str,
    config: Config,
    top_n: int = 10,
    key_fn: Callable[[str], Hashable] = full_formula_key,
    space_filter: Optional[Callable[[str], bool]] = None,
    synthesized: Optional[Sequence[str]] = None,
    attempted_path: Optional[str] = None,
):
    """
    Scatter E_hull vs r_DOS: full design space (backdrop) + run top-N.

    The design-space backdrop comes from the run's MACE cache and DOSCAR peaks
    (per-compound r_DOS via the full formula). The top-N overlay is ranked by
    the run's own reward (rollout_method/beta/gamma from Config, via
    score_by_method), not by the axes.

    Args:
        run_df: the run's compounds DataFrame (needs a name/formula column). The
            top-N overlay is chosen by scoring each named compound with the
            run's reward method; the r_DOS/e_hull used both for scoring and for
            plotting come from the design-space backdrop.
        out_path: where to write the .png.
        config: the run's Config. rollout_method, beta, gamma, cache_path (MACE
            cache) and doscar_data_path (DOSCAR peaks) are read from
            config.intermetallic.
        top_n: how many of the run's best compounds to overlay.
        key_fn: identifies a compound for matching the overlay against the
            backdrop (default full_formula_key, order-independent composition).
        space_filter: optional predicate restricting the backdrop (e.g. U-only).
            Default None plots every compound in the MACE cache.
        synthesized: dash- or plain-form names known synthesized (filled square).
        attempted_path: compounds_filtered.dat path; attempted-but-not-
            synthesized compounds get an open square.

    Returns:
        The matplotlib Figure, or None if there is no design-space data to plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ic = config.intermetallic
    if ic is None:
        raise ValueError("plot_ehull_vs_rdos requires an intermetallic Config section")
    rollout_method = ic.rollout_method
    beta = ic.beta
    gamma = ic.gamma
    if ic.cache_path is None or ic.doscar_data_path is None:
        raise ValueError(
            "config.intermetallic must set cache_path (MACE cache) and "
            "doscar_data_path (DOSCAR peaks) for the design-space backdrop"
        )

    df_mace, doscar_lookup = load_design_space(ic.cache_path, ic.doscar_data_path)
    if df_mace is None or not len(df_mace):
        return None

    space = df_mace.copy()
    space["name"] = space.get("name", space.get("formula"))
    if space_filter is not None:
        space = space[space["name"].apply(space_filter)].copy()
    if space.empty:
        return None
    space["r_dos"] = space["name"].apply(doscar_lookup.get_reward)
    space["x"] = space["r_dos"].astype(float)
    space["y"] = space["e_above_hull"].astype(float)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(space["x"], space["y"], s=6, color="#D0D0D0", label="All compounds")

    # Backdrop lookup: key -> (x, y), for matching overlays across name orderings.
    backdrop = {}
    for _, r in space.iterrows():
        backdrop[key_fn(r["name"])] = (float(r["x"]), float(r["y"]))

    # Experimentally-attempted overlays (optional).
    synth_sets = [_elem_set(s) for s in (synthesized or [])]
    attempted_sets = _load_attempted_sets(attempted_path)
    succ_x, succ_y, unsucc_x, unsucc_y = [], [], [], []
    for _, r in space.iterrows():
        es = _elem_set(r["name"])
        if any(es == s for s in synth_sets):
            succ_x.append(r["x"]); succ_y.append(r["y"])
        elif any(es == s for s in attempted_sets):
            unsucc_x.append(r["x"]); unsucc_y.append(r["y"])
    if unsucc_x:
        ax.scatter(unsucc_x, unsucc_y, s=80, marker="s", facecolors="none",
                   edgecolors="#9467bd", linewidths=1.2, label="Unsuccessful synthesis")
    if succ_x:
        ax.scatter(succ_x, succ_y, s=100, marker="s", facecolors="#9467bd",
                   edgecolors="#9467bd", linewidths=0.8, label="Successful synthesis")

    # Run top-N overlay, matched to the backdrop by key so element ordering in
    # the run's formulas doesn't silently drop points. Rank by the SAME reward
    # the run optimized (via score_by_method dispatching on rollout_method), so
    # the highlighted picks agree with the run and with the table - rather than
    # a fixed additive composite or a pre-existing score column. r_DOS and
    # e_hull are taken from the backdrop lookup (same source as the axes).
    df = run_df.copy()
    if "name" not in df.columns:
        df["name"] = df.get("formula", df.get("identifier"))

    def _run_score(name):
        hit = backdrop.get(key_fn(name))
        if hit is None:
            return None
        r_dos, e_hull = hit  # backdrop stores (x=r_DOS, y=e_hull)
        return score_by_method(rollout_method, e_hull, r_dos, beta, gamma)

    df["_score"] = df["name"].apply(_run_score)
    df = df[df["_score"].notna()].sort_values("_score", ascending=False)
    top = df.head(top_n)
    xs, ys = [], []
    for _, row in top.iterrows():
        hit = backdrop.get(key_fn(row["name"]))
        if hit is not None:
            xs.append(hit[0]); ys.append(hit[1])
    if xs:
        ax.scatter(xs, ys, s=45, color="#5BC0EB", marker="^", edgecolors="none",
                   alpha=0.65, label=f"Top {top_n} (MCTS)")

    ax.set_xlabel(r"$r_{\mathrm{DOS}}$")
    ax.set_ylabel(r"$E_{\mathrm{Hull}}$ (eV/atom)")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#D0D0D0",
               markeredgecolor="#D0D0D0", markersize=6, label="All compounds"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#5BC0EB",
               markeredgecolor="none", markersize=8, label=f"Top {top_n} (MCTS)"),
    ]
    if unsucc_x:
        handles.append(Line2D([0], [0], marker="s", linestyle="None",
                       markerfacecolor="none", markeredgecolor="#9467bd",
                       markersize=8, label="Unsuccessful synthesis"))
    if succ_x:
        handles.append(Line2D([0], [0], marker="s", linestyle="None",
                       markerfacecolor="#9467bd", markeredgecolor="#9467bd",
                       markersize=9, label="Successful synthesis"))
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig

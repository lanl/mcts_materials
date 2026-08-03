"""
Product-mode publication figures for U-only and lanthanide+U studies.

Replicates the three key figures from the original analysis scripts:
1. ehull_vs_rdos_product.png (U-only scatter with synthesized overlays)
2. ehull_vs_rdos_product_with_experimental.png (lanthanide+U with literature)
3. radial_tree_composite_product.png (3-panel radial tree)

These figures use the exact styling, colors, and layout from the original
mcts_crystal analysis scripts to ensure visual consistency with published work.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import Callable, Hashable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.config import Config
from .design_space import full_formula_key, load_design_space, score_by_method


# Element categories (from create_composite_radial_tree.py)
F_BLOCK = {
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu'
}
TRANSITION_METALS = {
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
}
GROUP_IV = {'Si', 'Ge', 'Sn', 'Pb'}

# Synthesized compounds (from synthesized_compounds.py)
SYNTHESIZED_COMPOUNDS = ['U-Sn-V', 'U-Sn-Nb', 'U-Ge-Cr', 'U-Ge-Co']


def _elem_set(name) -> set:
    """Set of element symbols in a formula/compound name (plain or dash form)."""
    if pd.isna(name):
        return set()
    s = str(name)
    if "-" in s:
        return {p.capitalize() for p in re.split(r"[^A-Za-z]", s) if p}
    return set(re.findall(r"[A-Z][a-z]?", s))


def _parse_elements(name: str) -> list:
    """Parse element symbols from formula."""
    return re.findall(r'[A-Z][a-z]?', str(name))


def _formula_key_non_f(name: str) -> Tuple:
    """Sorted tuple of non-f-block elements for cross-dataset matching."""
    parts = _parse_elements(name)
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


def _u_only_filter(name: str) -> bool:
    """True if compound contains U but no other f-block elements."""
    elems = set(_parse_elements(str(name)))
    return 'U' in elems and not (elems & (F_BLOCK - {'U'}))


def _lanthanide_u_filter(name: str) -> bool:
    """True if compound contains U or any lanthanide."""
    elems = set(_parse_elements(str(name)))
    lanthanides = {'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
    return 'U' in elems or bool(elems & lanthanides)


def plot_ehull_vs_rdos_product_u_only(
    out_path: str,
    config: Config,
    top_n: int = 15,
    attempted_path: Optional[str] = None,
    run_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    E_hull vs r_DOS scatter for U-only product-mode study.

    Replicates ehull_vs_rdos_product.png from the original analysis:
    - Full U-only design space as gray backdrop
    - Top-15 MCTS compounds as blue triangles
    - Synthesized compounds as filled purple squares
    - Unsuccessful synthesis attempts as open purple squares
    - Figure size: 3" × 3" at 300 DPI
    - Colors match legacy: #D0D0D0 (gray), #5BC0EB (blue), #9467bd (purple)

    Args:
        out_path: Output PNG file path
        config: Run config (must have intermetallic section with cache/doscar paths)
        top_n: Number of top compounds to overlay (default 15)
        attempted_path: Path to compounds_filtered.dat (attempted syntheses)
        run_df: Optional run compounds DataFrame. If None, ranks design space directly.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ic = config.intermetallic
    if ic is None:
        raise ValueError("plot_ehull_vs_rdos_product_u_only requires intermetallic config")
    if ic.cache_path is None or ic.doscar_data_path is None:
        raise ValueError("config.intermetallic must set cache_path and doscar_data_path")

    # Load design space and filter to U-only
    df_mace, doscar_lookup = load_design_space(ic.cache_path, ic.doscar_data_path)
    if df_mace is None or df_mace.empty:
        raise ValueError("No design space data loaded from cache_path")

    space = df_mace.copy()
    space["name"] = space.get("name", space.get("formula"))
    space = space[space["name"].apply(_u_only_filter)].copy()
    if space.empty:
        raise ValueError("No U-only compounds in design space")

    space["r_dos"] = space["name"].apply(doscar_lookup.get_reward)
    space["e_hull"] = space["e_above_hull"].astype(float)

    # Penalize no_mp_data compounds (set e_hull to 10)
    if "data_quality" in space.columns:
        mask = space["data_quality"].isin(["no_mp_data", "error"])
        space.loc[mask, "e_hull"] = 10.0

    # Filter sentinel values for display
    SENTINEL_EHULL = 9.9
    bg = space[space["e_hull"] < SENTINEL_EHULL].copy()

    # Build backdrop lookup: formula_key -> (r_dos, e_hull)
    backdrop = {}
    for _, r in bg.iterrows():
        key = _formula_key_non_f(r["name"])
        backdrop[key] = (float(r["r_dos"]), float(r["e_hull"]))

    # Get top-N compounds from run or design space ranking
    if run_df is not None:
        # Rank run compounds by product reward
        df = run_df.copy()
        if "name" not in df.columns:
            df["name"] = df.get("formula", df.get("identifier"))

        def _run_score(name):
            key = _formula_key_non_f(name)
            hit = backdrop.get(key)
            if hit is None:
                return None
            r_dos, e_hull = hit
            return score_by_method("ehull_rdos_product", e_hull, r_dos, 1.0, 1.0)

        df["_score"] = df["name"].apply(_run_score)
        df = df[df["_score"].notna()].sort_values("_score", ascending=False)
        top_compounds = df.head(top_n)
    else:
        # Rank full design space by product reward
        from ..intermetallic import ehull_reward
        space["product"] = space.apply(
            lambda r: ehull_reward(r["e_hull"]) * r["r_dos"], axis=1
        )
        space = space.sort_values("product", ascending=False)
        top_compounds = space.head(top_n)

    # Synthesized/attempted overlays
    synth_sets = [_elem_set(s) for s in SYNTHESIZED_COMPOUNDS]
    succ_x, succ_y, unsucc_x, unsucc_y = [], [], [], []

    if attempted_path and Path(attempted_path).exists():
        try:
            df_att = pd.read_csv(attempted_path, sep=r"\s+", comment="#", header=None,
                                 names=["name", "e_form", "e_hull"])
            # Iterate through attempted compounds and look up their coordinates in backdrop
            for _, row in df_att.iterrows():
                name = row["name"]
                key = _formula_key_non_f(name)
                if key not in backdrop:
                    continue

                r_dos, e_hull = backdrop[key]
                es = _elem_set(name)

                # Check if successfully synthesized
                if any(es == s for s in synth_sets):
                    succ_x.append(r_dos)
                    succ_y.append(e_hull)
                else:
                    unsucc_x.append(r_dos)
                    unsucc_y.append(e_hull)
        except Exception as e:
            print(f"Warning: Could not load attempted syntheses: {e}")

    # Create figure (3" × 3", exact legacy size)
    fig, ax = plt.subplots(figsize=(3, 3))

    # Background: all U-only compounds
    ax.scatter(bg["r_dos"], bg["e_hull"], s=5, color="#D0D0D0",
               linewidths=0, label="All Compounds")

    # Unsuccessful synthesis (open purple squares)
    if unsucc_x:
        ax.scatter(unsucc_x, unsucc_y, s=80, marker="s", facecolors="none",
                   edgecolors="#9467bd", linewidths=1.2, label="Unsuccessful Synthesis")

    # Successful synthesis (filled purple squares)
    if succ_x:
        ax.scatter(succ_x, succ_y, s=100, marker="s", facecolors="#9467bd",
                   edgecolors="#9467bd", linewidths=0.8, label="Successful Synthesis")

    # Top-15 MCTS overlay (blue triangles)
    top_x, top_y = [], []
    for _, row in top_compounds.iterrows():
        key = _formula_key_non_f(row["name"])
        hit = backdrop.get(key)
        if hit is not None:
            top_x.append(hit[0])
            top_y.append(hit[1])
    if top_x:
        ax.scatter(top_x, top_y, s=45, color="#5BC0EB", marker="^",
                   edgecolors="none", alpha=0.55, label="Top 15 (MCTS)")

    # Styling (exact legacy format)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylim(top=1.5)
    ax.set_xlabel(r"$r_{\mathrm{DOS}}$", fontsize=9)
    ax.set_ylabel(r"$E_{\mathrm{Hull}}$ (eV/atom)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Legend (exact legacy order and styling)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#D0D0D0",
               markeredgecolor="#D0D0D0", markersize=5, label="All Compounds"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#5BC0EB",
               markeredgecolor="none", markersize=7, alpha=0.55, label="Top 15 (MCTS)"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none",
               markeredgecolor="#9467bd", markersize=7, label="Unsuccessful Synthesis"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#9467bd",
               markeredgecolor="#9467bd", markersize=7, label="Successful Synthesis"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.set_size_inches(3, 3)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ehull_vs_rdos_product_lanthanide_u(
    out_path: str,
    config: Config,
    experimental_path: Optional[str] = None,
    top_n: int = 15,
    run_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    E_hull vs r_DOS scatter for lanthanide+U product-mode study.

    Replicates ehull_vs_rdos_product_with_experimental.png from the original analysis:
    - Full lanthanide+U design space as gray backdrop
    - Top-15 MCTS compounds as blue triangles
    - Experimental literature compounds as red diamonds
    - Figure size: 3" × 3" at 300 DPI
    - Colors match legacy: #D0D0D0 (gray), #5BC0EB (blue), #E84855 (red)

    Args:
        out_path: Output PNG file path
        config: Run config (must have intermetallic section with cache/doscar paths)
        experimental_path: Path to experimental_citation_compounds file
        top_n: Number of top compounds to overlay (default 15)
        run_df: Optional run compounds DataFrame. If None, ranks design space directly.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ic = config.intermetallic
    if ic is None:
        raise ValueError("plot_ehull_vs_rdos_product_lanthanide_u requires intermetallic config")
    if ic.cache_path is None or ic.doscar_data_path is None:
        raise ValueError("config.intermetallic must set cache_path and doscar_data_path")

    # Load full design space (don't filter yet - need all compounds for experimental overlay)
    df_mace, doscar_lookup = load_design_space(ic.cache_path, ic.doscar_data_path)
    if df_mace is None or df_mace.empty:
        raise ValueError("No design space data loaded from cache_path")

    space = df_mace.copy()
    space["name"] = space.get("name", space.get("formula"))
    space["r_dos"] = space["name"].apply(doscar_lookup.get_reward)
    space["e_hull"] = space["e_above_hull"].astype(float)

    # Filter to valid compounds only
    if "data_quality" in space.columns:
        space = space[space["data_quality"] == "valid"].copy()

    # Use full space as backdrop (for experimental compound lookups)
    bg = space.copy()

    # Filter to lanthanide+U for ranking MCTS compounds
    space_filtered = space[space["name"].apply(_lanthanide_u_filter)].copy()
    if space_filtered.empty:
        raise ValueError("No lanthanide+U compounds in design space")

    # Build backdrop lookup: for each non-f-block key, keep the compound with MAX rdos
    # (multiple lanthanide substitutions collapse to same key; we want the best one)
    backdrop = {}
    for _, r in bg.iterrows():
        key = _formula_key_non_f(r["name"])
        rdos = float(r["r_dos"])
        ehull = float(r["e_hull"])
        if key not in backdrop or rdos > backdrop[key][0]:
            backdrop[key] = (rdos, ehull)

    # Get top-N compounds from MCTS results
    # Plot ALL top-N at their lanthanide-specific positions (not collapsed by non-f-block key)
    if run_df is not None:
        # Simply take the top N from run results as-is (already sorted by reward)
        top_compounds = run_df.head(top_n).copy()
        if "name" not in top_compounds.columns:
            top_compounds["name"] = top_compounds.get("formula", top_compounds.get("identifier"))
    else:
        from ..intermetallic import ehull_reward
        space_filtered["product"] = space_filtered.apply(
            lambda r: ehull_reward(r["e_hull"]) * r["r_dos"], axis=1
        )
        space_filtered = space_filtered.sort_values("product", ascending=False)
        top_compounds = space_filtered.head(top_n)

    # Parse experimental compounds from file - look up SPECIFIC lanthanide compound
    exp_x, exp_y = [], []
    if experimental_path and Path(experimental_path).exists():
        try:
            with open(experimental_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        tm, giv, r = parts

                        # Look up the SPECIFIC lanthanide compound (not collapsed by non-f-block key)
                        # Try common stoichiometries: TM6GIV6R format (most common in design space)
                        test_formulas = [
                            f"{tm}6{giv}6{r}",    # e.g., Fe6Ge6Lu
                            f"{r}{tm}6{giv}6",    # e.g., LuFe6Ge6
                            f"{tm}{giv}{r}",      # e.g., FeGeLu
                            f"{r}{giv}{tm}",      # e.g., LuGeFe
                        ]

                        rdos = None
                        ehull = None
                        for test_formula in test_formulas:
                            test_rdos = doscar_lookup.get_reward(test_formula)
                            if test_rdos > 0:
                                # Found a match - get its ehull from design space
                                match_rows = bg[bg["name"] == test_formula]
                                if len(match_rows) > 0:
                                    rdos = test_rdos
                                    ehull = float(match_rows.iloc[0]["e_hull"])
                                    break

                        # Fallback to backdrop if no specific match
                        if rdos is None or ehull is None:
                            key = tuple(sorted([p for p in [r, tm, giv] if p not in F_BLOCK]))
                            hit = backdrop.get(key)
                            if hit is not None:
                                if rdos is None:
                                    rdos = hit[0]
                                if ehull is None:
                                    ehull = hit[1]

                        if rdos is not None and ehull is not None:
                            exp_x.append(rdos)
                            exp_y.append(ehull)
        except Exception as e:
            print(f"Warning: Could not parse experimental compounds: {e}")

    # Use lanthanide+U filtered space for background display
    bg_display = space_filtered.copy()

    # Calculate axis limits with padding
    y_bg = bg_display["e_hull"]
    x_pad = (bg_display["r_dos"].max() - bg_display["r_dos"].min()) * 0.05
    y_pad = (y_bg.max() - y_bg.min()) * 0.05
    xlim = (bg_display["r_dos"].min() - x_pad, bg_display["r_dos"].max() + x_pad)
    ylim = (y_bg.min() - y_pad, y_bg.max() + y_pad)

    # Create figure
    fig, ax = plt.subplots(figsize=(3, 3))

    # Background: all lanthanide+U compounds
    ax.scatter(bg_display["r_dos"], bg_display["e_hull"], s=4, color="#D0D0D0",
               linewidths=0, zorder=1)

    # Experimental literature (red diamonds, behind MCTS so MCTS is visible)
    if exp_x:
        ax.scatter(exp_x, exp_y, s=28, color="#E84855", marker="D",
                   edgecolors="none", alpha=0.8, zorder=3)

    # Top-N MCTS overlay (blue triangles, on top)
    # Use MCTS compounds' own rdos and ehull values
    top_x, top_y = [], []
    for _, row in top_compounds.iterrows():
        formula = row["name"]

        # Get rdos - lanthanide-specific from doscar lookup
        rdos = doscar_lookup.get_reward(formula)

        # Get ehull - use the value from MCTS results (these may be NEW compounds not in design space!)
        ehull = None
        if "e_above_hull" in row:
            ehull = float(row["e_above_hull"])

        if rdos > 0 and ehull is not None:
            top_x.append(rdos)
            top_y.append(ehull)
        else:
            # Fallback to backdrop if no ehull in results
            key = _formula_key_non_f(formula)
            hit = backdrop.get(key)
            if hit is not None:
                top_x.append(hit[0])
                top_y.append(hit[1])

    if top_x:
        ax.scatter(top_x, top_y, s=45, color="#5BC0EB", marker="^",
                   edgecolors="none", alpha=0.55, zorder=4)

    # Styling
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$r_{\mathrm{DOS}}$", fontsize=9)
    ax.set_ylabel(r"$E_{\mathrm{Hull}}$ (eV/atom)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Legend
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="#D0D0D0", markeredgecolor="#D0D0D0",
               markersize=5, label="Design space"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor="#5BC0EB", markeredgecolor="none",
               markersize=7, alpha=0.55, label="Top 15 (MCTS)"),
        Line2D([0], [0], marker="D", linestyle="None",
               markerfacecolor="#E84855", markeredgecolor="none",
               markersize=5, alpha=0.8, label="Experimental literature"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.set_size_inches(3, 3)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_radial_tree_product(
    tree_path: str,
    out_path: str,
    config: Config,
    table_path: Optional[str] = None,
    max_nodes: int = 60,
) -> None:
    """
    3-panel radial tree visualization for product-mode studies.

    Replicates radial_tree_composite_product.png from the original analysis:
    - Panel (a): Product reward (r_Ehull × r_DOS) with rank labels
    - Panel (b): r_Ehull component
    - Panel (c): r_DOS component
    - Layout: (a) takes left 2/3 width at full height, (b) and (c) stack vertically on right 1/3
    - Root node marked with gold star
    - Top-15 compounds labeled with MCTS rank (white text)
    - Diverging RdBu colormap for product/r_ehull, Blues for r_DOS
    - BFS spanning tree edges as bold arrows, cross-links omitted from main panel
    - Figure size: 6.5" × 4" at 300 DPI

    Args:
        tree_path: Path to tree.json from a completed run
        out_path: Output PNG file path
        config: Run config (must have intermetallic section)
        table_path: Optional path to top15 LaTeX table for rank labels
        max_nodes: Maximum nodes to display (default 60)
    """
    import json
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import networkx as nx

    from ..intermetallic import ehull_reward
    from .radial_tree import _radial_layout

    ic = config.intermetallic
    if ic is None:
        raise ValueError("plot_radial_tree_product requires intermetallic config")
    if ic.cache_path is None or ic.doscar_data_path is None:
        raise ValueError("config.intermetallic must set cache_path and doscar_data_path")

    # Load tree
    with open(tree_path) as f:
        tree = json.load(f)

    # Load design space for r_DOS lookup
    _, doscar_lookup = load_design_space(ic.cache_path, ic.doscar_data_path)

    # Build collapsed graph (one node per unique material)
    by_id = {rec["id"]: rec for rec in tree["nodes"]}
    root_id = tree.get("root_id", 0)

    unique = {}
    edges = set()
    root_key = None

    for rec in tree["nodes"]:
        key = str(rec["identifier"])
        u = unique.setdefault(key, {"visits": 0, "total_reward": 0.0, "e_hulls": [], "r_doss": []})
        u["visits"] += int(rec.get("visits", 0) or 0)
        u["total_reward"] += float(rec.get("total_reward", 0.0) or 0.0)

        props = rec.get("properties") or {}
        if "e_above_hull" in props:
            u["e_hulls"].append(float(props["e_above_hull"]))

        # Get r_DOS from doscar lookup
        # Extract formula from full identifier (format: "Formula|SG191|sites...")
        formula = key.split("|")[0] if "|" in key else key
        r_dos = doscar_lookup.get_reward(formula)
        u["r_doss"].append(r_dos)

        parent = rec.get("parent")
        if parent is None:
            root_key = key
        else:
            pkey = str(by_id[parent]["identifier"])
            if pkey != key:
                edges.add((pkey, key))

    # Parse MCTS ranks from table
    mcts_ranks = {}
    if table_path and Path(table_path).exists():
        try:
            with open(table_path) as f:
                for line in f:
                    m = re.match(
                        r'\s*(\d+)\s*&\s*\d+\s*&\s*(.*?)\s*&',
                        line
                    )
                    if m:
                        rank = int(m.group(1))
                        elems = re.findall(r'[A-Z][a-z]?', m.group(2))
                        key = tuple(sorted(e for e in elems if e not in F_BLOCK))
                        mcts_ranks[key] = rank
        except Exception as e:
            print(f"Warning: Could not parse table ranks: {e}")

    # Keep all nodes (no filtering for legibility)
    # Apply max_nodes limit if there are too many
    if len(unique) > max_nodes:
        # Prioritize high-visit nodes and top-15 table compounds
        sorted_nodes = sorted(unique.items(), key=lambda kv: kv[1]["visits"], reverse=True)
        force_keep = {k for k in unique if _formula_key_non_f(k) in mcts_ranks}
        keep_set = force_keep | {k for k, _ in sorted_nodes[:max_nodes]}
    else:
        # Show all nodes
        keep_set = set(unique.keys())
    if root_key is not None:
        keep_set.add(root_key)

    # Build networkx graph
    graph = nx.DiGraph()
    for key in keep_set:
        v = unique[key]
        e_hull = v["e_hulls"][0] if v["e_hulls"] else None
        r_dos = max(v["r_doss"]) if v["r_doss"] else 0.0

        # Product reward: r_Ehull × r_DOS (gamma=1.0 for raw r_DOS)
        if e_hull is not None:
            r_ehull = ehull_reward(e_hull)
            product = r_ehull * r_dos
        else:
            r_ehull = None
            product = None

        q_per_n = v["total_reward"] / v["visits"] if v["visits"] > 0 else None

        graph.add_node(
            key,
            identifier=key,
            visits=v["visits"],
            e_hull=e_hull,
            r_dos=r_dos,
            r_ehull=r_ehull,
            product=product,
            q_per_n=q_per_n,
        )

    for a, b in edges:
        if a in keep_set and b in keep_set:
            graph.add_edge(a, b)

    if root_key is None or root_key not in graph:
        root_key = next(iter(graph.nodes()), None)

    if not graph.nodes():
        print("Warning: No nodes to plot")
        return

    # Compute layout, flip 180 degrees
    pos = _radial_layout(graph, root_key, radius_step=10.0) if root_key is not None else {}
    pos = {nid: (-x, -y) for nid, (x, y) in pos.items()} if pos else {}

    # Spread overlapping nodes (simplified version)
    if pos and root_key is not None:
        pos = _spread_nodes_simple(pos, root_key, min_dist=5.5)

    # Extract coloring data
    products = {}
    r_ehulls = {}
    r_doss_vals = {}

    for n in graph.nodes():
        products[n] = graph.nodes[n].get("product")
        r_ehulls[n] = graph.nodes[n].get("r_ehull")
        r_doss_vals[n] = graph.nodes[n].get("r_dos", 0.0)

    # Colormaps (matching legacy)
    _rdbu_base = matplotlib.colormaps['RdBu']
    _rdbu_bright = mcolors.LinearSegmentedColormap.from_list(
        'RdBu_bright', _rdbu_base(np.linspace(0.1, 0.9, 256))
    )
    cmap_product = _rdbu_bright
    cmap_rehull = _rdbu_bright
    cmap_rdos = matplotlib.colormaps['Blues']

    # Normalization
    prod_vals = [v for v in products.values() if v is not None and not pd.isna(v)]
    norm_product = mcolors.Normalize(vmin=-2250, vmax=2250) if prod_vals else None

    rehull_vals = [v for v in r_ehulls.values() if v is not None and not pd.isna(v)]
    norm_rehull = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0) if rehull_vals else None

    rdos_vals = [v for v in r_doss_vals.values() if v > 0]
    norm_rdos = mcolors.Normalize(vmin=0.0, vmax=max(rdos_vals)) if rdos_vals else None

    # Node colors for each panel
    node_list = list(graph.nodes())
    node_colors_product = [
        cmap_product(norm_product(products[n]))
        if (products.get(n) is not None and not pd.isna(products[n]) and norm_product is not None)
        else 'lightgray'
        for n in node_list
    ]
    node_colors_rehull = [
        cmap_rehull(norm_rehull(r_ehulls[n]))
        if (r_ehulls.get(n) is not None and not pd.isna(r_ehulls[n]) and norm_rehull is not None)
        else 'lightgray'
        for n in node_list
    ]
    node_colors_rdos = [
        cmap_rdos(norm_rdos(r_doss_vals[n]))
        if (r_doss_vals.get(n) > 0 and norm_rdos is not None)
        else 'lightgray'
        for n in node_list
    ]

    # Split edges: BFS spanning tree vs cross-links
    if root_key is not None and root_key in graph:
        bfs_tree_edges = set(nx.bfs_tree(graph, root_key).edges())
    else:
        bfs_tree_edges = set()
    tree_edges = [e for e in graph.edges() if e in bfs_tree_edges]

    # Layout: large main panel (a) on left (2/3 width, full height),
    # r_ehull (b) and r_dos (c) stacked vertically on right (1/3 width each)
    fig = plt.figure(figsize=(6.5, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], wspace=0.0)
    ax_main = fig.add_subplot(gs[:, 0])   # combined product + rank labels
    ax_rehull = fig.add_subplot(gs[0, 1])  # r_ehull
    ax_rdos = fig.add_subplot(gs[1, 1])    # r_dos

    panels = [
        (ax_main, node_colors_product, cmap_product, norm_product, 'Product Reward', '(a)'),
        (ax_rehull, node_colors_rehull, cmap_rehull, norm_rehull, r"$r_{E_{\mathrm{Hull}}}$", '(b)'),
        (ax_rdos, node_colors_rdos, cmap_rdos, norm_rdos, r"$r_{\mathrm{DOS}}$", '(c)'),
    ]

    NODE_SIZE_MAIN = 120
    NODE_SIZE_SMALL = 24

    for ax, ncols, cmap_m, norm_m, label, abc in panels:
        is_main = ax is ax_main
        ns = NODE_SIZE_MAIN if is_main else NODE_SIZE_SMALL

        # Draw edges
        arrow_sz = 7 if is_main else 4
        edge_w = 1.2 if is_main else 0.7
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=tree_edges,
                               edge_color='dimgray', width=edge_w, arrows=True,
                               arrowsize=arrow_sz, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.08',
                               node_size=ns, min_source_margin=2, min_target_margin=2)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=node_list,
                               node_color=ncols, edgecolors='black',
                               linewidths=0.5, node_size=ns)

        # Rank labels (main panel only)
        if is_main:
            for node in graph.nodes():
                key = _formula_key_non_f(node)
                rank = mcts_ranks.get(key)
                if rank is None:
                    continue
                x, y = pos.get(node, (None, None))
                if x is None:
                    continue
                ax.text(x, y, str(rank), ha='center', va='center', fontsize=8,
                        fontweight='normal', color='white', zorder=10)

        # Gold star on root
        if root_key is not None and root_key in pos:
            rx, ry = pos[root_key]
            ax.scatter([rx], [ry], s=ns * 0.55, marker='*', facecolor='gold',
                       edgecolors='black', linewidths=0.5, zorder=5)

        # Panel label
        ax.text(0.02, 0.98, abc, transform=ax.transAxes, va='top', ha='left',
                fontsize=10, weight='bold')

        # Set limits
        if pos:
            _xs = [pos[n][0] for n in graph.nodes() if n in pos]
            _ys = [pos[n][1] for n in graph.nodes() if n in pos]
            _cx = (max(_xs) + min(_xs)) / 2
            _cy = (max(_ys) + min(_ys)) / 2
            _x_ext = max(_xs) - min(_xs)
            _y_ext = max(_ys) - min(_ys)
            if is_main:
                _half = max(_x_ext, _y_ext) / 2 * 1.15
                ax.set_xlim(_cx - _half, _cx + _half)
                ax.set_ylim(_cy - _half, _cy + _half)
            else:
                _half_x = max(_x_ext, _y_ext) / 2 * 1.55
                _half_y = _y_ext / 2 * 1.12
                ax.set_xlim(_cx - _half_x, _cx + _half_x)
                ax.set_ylim(_cy - _half_y, _cy + _half_y)
        ax.set_axis_off()

        # Colorbar
        sm = cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
        sm.set_array([])
        if is_main:
            cax = ax.inset_axes([0.20, 0.01, 0.55, 0.022])
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(label, fontsize=8)
            cbar.ax.xaxis.set_label_position('bottom')
            cbar.ax.xaxis.tick_bottom()
        else:
            cax = ax.inset_axes([0.92, 0.10, 0.05, 0.75])
            cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(label, fontsize=8, labelpad=4)
            cbar.ax.yaxis.set_label_position('right')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _spread_nodes_simple(pos, root, min_dist=5.5, max_iters=600):
    """
    Push overlapping nodes apart while preserving radius.

    Simplified version of the legacy _spread_nodes function.
    """
    import math
    pos = {k: list(v) for k, v in pos.items()}
    original_radii = {}
    for k, v in pos.items():
        if k == root:
            continue
        r = math.sqrt(v[0] ** 2 + v[1] ** 2)
        original_radii[k] = r if r > 1e-9 else min_dist

    movable = [n for n in pos if n != root]

    for _ in range(max_iters):
        moved = False
        for i in range(len(movable)):
            for j in range(i + 1, len(movable)):
                n1, n2 = movable[i], movable[j]
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]
                dx, dy = x2 - x1, y2 - y1
                d = math.sqrt(dx * dx + dy * dy)
                if d >= min_dist:
                    continue
                if d < 1e-9:
                    # Break degeneracy
                    dx, dy, d = 1e-4, 0.0, 1e-4
                push = (min_dist - d) * 0.5
                ux, uy = dx / d, dy / d
                pos[n1][0] -= push * ux
                pos[n1][1] -= push * uy
                pos[n2][0] += push * ux
                pos[n2][1] += push * uy
                moved = True

        # Renormalize to original radii
        for n in movable:
            r_orig = original_radii[n]
            x, y = pos[n]
            r_curr = math.sqrt(x * x + y * y)
            if r_curr > 1e-9:
                pos[n][0] = x * r_orig / r_curr
                pos[n][1] = y * r_orig / r_curr

        if not moved:
            break

    return {k: tuple(v) for k, v in pos.items()}

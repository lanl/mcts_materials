"""
Radial search-tree figure for post-run analysis.

Reads a run's persisted tree (tree.json, written by save_results) and draws the
explored tree radially, collapsed to one node per unique material and colored by
the run's reward and its components. Ports create_composite_radial_tree.py from
the original analysis scripts, but reads the tree from JSON (the framework does
not pickle live MCTS objects) and derives r_DOS / reward from the run's own
Config, so it stays chemistry-agnostic and consistent with the tables/scatter.

matplotlib and networkx are imported lazily so importing this module does not
require the [viz] extra.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Optional

import numpy as np

from ..core.config import Config
from ..intermetallic import ehull_reward
from .design_space import full_formula_key, load_design_space, score_by_method


def _radial_layout(graph, root, radius_step: float = 4.0) -> Dict[Any, tuple]:
    """
    Radial tree layout that keeps each subtree within an angular wedge.

    Radius = shortest-path depth from root; angle is assigned recursively over a
    BFS spanning tree, each branch's wedge sized by its leaf count, so subtrees
    stay clustered and edges stay short. Ported from the original analysis
    script's radial_layout.
    """
    import networkx as nx

    pos: Dict[Any, tuple] = {}
    if root is None or root not in graph:
        return pos

    try:
        depths = nx.single_source_shortest_path_length(graph, root)
    except Exception:
        nodes = list(graph.nodes())
        for i, nkey in enumerate(nodes):
            theta = 2.0 * math.pi * (i / max(1, len(nodes)))
            pos[nkey] = (radius_step * math.cos(theta), radius_step * math.sin(theta))
        return pos

    bfs_tree = nx.bfs_tree(graph, root)
    children = {n: list(bfs_tree.successors(n)) for n in bfs_tree.nodes()}
    leaf_count: Dict[Any, int] = {}

    def count_leaves(n):
        kids = children.get(n, [])
        leaf_count[n] = 1 if not kids else sum(count_leaves(c) for c in kids)
        return leaf_count[n]

    count_leaves(root)
    angle: Dict[Any, float] = {}

    def assign_angles(n, theta_start, theta_end):
        angle[n] = 0.5 * (theta_start + theta_end)
        kids = children.get(n, [])
        if not kids:
            return
        total = sum(leaf_count[c] for c in kids) or len(kids)
        cursor, span = theta_start, theta_end - theta_start
        for c in kids:
            child_span = span * (leaf_count.get(c, 1) / total)
            assign_angles(c, cursor, cursor + child_span)
            cursor += child_span

    assign_angles(root, 0.0, 2.0 * math.pi)

    fallback = [n for n in graph.nodes() if n not in angle]
    for i, n in enumerate(fallback):
        angle[n] = 2.0 * math.pi * (i / max(1, len(fallback)))

    for n in graph.nodes():
        d = depths.get(n, 0)
        th = angle.get(n, 0.0)
        pos[n] = (d * radius_step * math.cos(th), d * radius_step * math.sin(th))
    pos[root] = (0.0, 0.0)
    return pos


def _build_graph(
    tree: Dict[str, Any],
    doscar_lookup,
    rollout_method: str,
    beta: float,
    gamma: float,
    max_nodes: int,
):
    """
    Collapse the serialized tree to one node per identifier and build a graph.

    Each unique material aggregates visit counts (sum) and reward (max); r_DOS
    comes from the run's DOSCAR lookup and the node reward from score_by_method,
    so coloring matches the run. Low-visit nodes are trimmed to keep the figure
    legible (mirrors the original's visit>=2 / top-max-nodes rule).
    """
    import networkx as nx

    by_id = {rec["id"]: rec for rec in tree["nodes"]}
    root_id = tree.get("root_id", 0)

    unique: Dict[str, Dict[str, Any]] = {}
    edges = set()
    root_key = None
    for rec in tree["nodes"]:
        key = str(rec["identifier"])
        u = unique.setdefault(
            key, {"visits": 0, "total_reward": 0.0, "e_hulls": []}
        )
        u["visits"] += int(rec.get("visits", 0) or 0)
        u["total_reward"] += float(rec.get("total_reward", 0.0) or 0.0)
        props = rec.get("properties") or {}
        if "e_above_hull" in props:
            u["e_hulls"].append(float(props["e_above_hull"]))

        parent = rec.get("parent")
        if parent is None:
            root_key = key
        else:
            pkey = str(by_id[parent]["identifier"])
            if pkey != key:
                edges.add((pkey, key))

    # Trim for legibility: keep visits >= 2, else top-`max_nodes` by visits.
    keep = [k for k, v in unique.items() if v["visits"] >= 2]
    if len(keep) > max_nodes or len(keep) < 10:
        ordered = sorted(unique.items(), key=lambda kv: kv[1]["visits"], reverse=True)
        keep = [k for k, _ in ordered[:max_nodes]]
    keep_set = set(keep)
    if root_key is not None:
        keep_set.add(root_key)

    graph = nx.DiGraph()
    for key in keep_set:
        v = unique[key]
        e_hull = v["e_hulls"][0] if v["e_hulls"] else None
        r_dos = doscar_lookup.get_reward(key)
        reward = (
            score_by_method(rollout_method, e_hull, r_dos, beta, gamma)
            if e_hull is not None else None
        )
        q_per_n = v["total_reward"] / v["visits"] if v["visits"] > 0 else None
        graph.add_node(
            key, identifier=key, visits=v["visits"], e_hull=e_hull,
            r_dos=r_dos, reward=reward, q_per_n=q_per_n,
            r_ehull=(ehull_reward(e_hull) if e_hull is not None else None),
        )
    for a, b in edges:
        if a in keep_set and b in keep_set:
            graph.add_edge(a, b)

    if root_key is None or root_key not in graph:
        root_key = next(iter(graph.nodes()), None)
    return graph, root_key


def plot_radial_tree(
    tree_path: str,
    out_path: str,
    config: Config,
    max_nodes: int = 60,
    key_fn: Callable[[str], Hashable] = full_formula_key,
):
    """
    Draw the explored search tree radially, in four colored panels.

    Panels color the same layout by: the run's reward, r_ehull, weighted r_DOS
    (gamma * r_DOS), and Q/N (mean reward = total_reward/visits). The MCTS root
    is starred; the BFS spanning tree is drawn as bold arrows and revisit
    cross-links as faint threads (so the figure shows branch structure, not a
    tangle).

    Args:
        tree_path: path to a run's tree.json (written by save_results).
        out_path: where to write the .png.
        config: the run's Config; rollout_method, gamma, beta, cache_path and
            doscar_data_path drive the per-node reward/coloring.
        max_nodes: cap on drawn unique-material nodes (low-visit nodes trimmed).
        key_fn: reserved for identifier normalization; currently identifiers are
            used verbatim as node keys.

    Returns:
        The matplotlib Figure, or None if the tree has no drawable nodes.
    """
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import networkx as nx

    ic = config.intermetallic
    if ic is None:
        raise ValueError("plot_radial_tree requires an intermetallic Config section")
    if ic.doscar_data_path is None:
        raise ValueError("config.intermetallic must set doscar_data_path for rDOS coloring")

    with open(tree_path) as f:
        tree = json.load(f)

    _, doscar_lookup = load_design_space(ic.cache_path or "", ic.doscar_data_path)
    graph, root_key = _build_graph(
        tree, doscar_lookup, ic.rollout_method, ic.beta, ic.gamma, max_nodes
    )
    if graph.number_of_nodes() == 0 or root_key is None:
        return None

    pos = _radial_layout(graph, root_key, radius_step=4.0)
    pos = {k: (-x, -y) for k, (x, y) in pos.items()}  # flip 180 deg for aesthetics

    # Spanning tree vs revisit cross-links.
    tree_edges_set = set(nx.bfs_tree(graph, root_key).edges())
    tree_edges = [e for e in graph.edges() if e in tree_edges_set]
    cross_edges = [e for e in graph.edges() if e not in tree_edges_set]

    def _panel_values(attr, scale=1.0):
        vals = {}
        for n in graph.nodes():
            v = graph.nodes[n].get(attr)
            vals[n] = None if v is None else float(v) * scale
        return vals

    panels_spec = [
        ("reward", 1.0, "viridis", "Reward"),
        ("r_ehull", 1.0, "Oranges", r"$r_{E_{\mathrm{Hull}}}$"),
        ("r_dos", ic.gamma, "Greens", r"$\gamma \cdot r_{\mathrm{DOS}}$"),
        ("q_per_n", 1.0, "Purples", r"$Q/N$"),
    ]

    plt.rcParams.update({"font.size": 10})
    NODE_SIZE = 70
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.75), constrained_layout=True)
    labels_abc = ["(a)", "(b)", "(c)", "(d)"]

    for i, (attr, scale, cmap_name, label) in enumerate(panels_spec):
        ax = axes[i]
        cmap = matplotlib.colormaps[cmap_name]
        values = _panel_values(attr, scale)
        finite = [v for v in values.values() if v is not None and not np.isnan(v)]
        norm = mcolors.Normalize(vmin=min(finite), vmax=max(finite)) if finite else None
        colors = [
            "lightgray" if (values[n] is None or norm is None) else cmap(norm(values[n]))
            for n in graph.nodes()
        ]

        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=cross_edges, edge_color="#DDDDDD",
            width=0.4, arrows=False,
        )
        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=tree_edges, edge_color="dimgray",
            width=0.7, arrows=True, arrowsize=5, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08", node_size=NODE_SIZE,
            min_source_margin=2, min_target_margin=2,
        )
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_color=colors, edgecolors="black",
            linewidths=0.5, node_size=NODE_SIZE,
        )
        if root_key in pos:
            rx, ry = pos[root_key]
            ax.scatter([rx], [ry], s=NODE_SIZE * 0.55, marker="*", facecolor="gold",
                       edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(0.02, 0.98, labels_abc[i], transform=ax.transAxes, va="top",
                ha="left", fontsize=10, weight="bold")
        ax.axis("equal")
        ax.axis("off")
        if norm is not None:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                                fraction=0.08, pad=0.08, aspect=20)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(label, fontsize=9)

    fig.text(0.5, -0.03, "★ starting node (MCTS root)    bold arrow = expansion step",
             ha="center", va="top", fontsize=7)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig

"""
Radial search-tree figure for post-run analysis.

Reads a run's persisted tree (tree.json, written by save_results) and draws the
explored tree radially, collapsed to one node per unique material and colored by
the run's reward and its components. Reads the tree from JSON (the framework does
not pickle live MCTS objects) and derives r_DOS / reward from the run's own
Config, so it stays chemistry-agnostic and consistent with the tables/scatter.

Three panels: (a) the run's reward (large, with top-N rank labels), (b) the
E_hull reward component, (c) r_DOS. The node layout is identical across panels;
only the coloring differs. Node ordering is sorted by identifier so the figure
is deterministic across runs (independent of set-iteration / hash-seed order).

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
        nodes = sorted(graph.nodes())
        for i, nkey in enumerate(nodes):
            theta = 2.0 * math.pi * (i / max(1, len(nodes)))
            pos[nkey] = (radius_step * math.cos(theta), radius_step * math.sin(theta))
        return pos

    bfs_tree = nx.bfs_tree(graph, root)
    # Sort children so wedge assignment (and thus the whole layout) is
    # deterministic regardless of graph insertion / set-iteration order.
    children = {n: sorted(bfs_tree.successors(n)) for n in bfs_tree.nodes()}
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

    fallback = [n for n in sorted(graph.nodes()) if n not in angle]
    for i, n in enumerate(fallback):
        angle[n] = 2.0 * math.pi * (i / max(1, len(fallback)))

    for n in graph.nodes():
        d = depths.get(n, 0)
        th = angle.get(n, 0.0)
        pos[n] = (d * radius_step * math.cos(th), d * radius_step * math.sin(th))
    pos[root] = (0.0, 0.0)
    return pos


def _spread_nodes(pos, root, min_dist: float = 5.5, max_iters: int = 600):
    """
    Push overlapping nodes apart while preserving each node's radius from root.

    Iterative relaxation: nodes closer than min_dist repel along their
    connecting axis, then every node is renormalized back to its original
    radius so the radial/depth structure is preserved. Deterministic (fixed
    iteration order over sorted nodes).
    """
    pos = {k: list(v) for k, v in pos.items()}
    original_radii = {}
    for k, v in pos.items():
        if k == root:
            continue
        r = math.sqrt(v[0] ** 2 + v[1] ** 2)
        original_radii[k] = r if r > 1e-9 else min_dist

    movable = sorted(n for n in pos if n != root)

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
                    dx, dy, d = 1e-4, 0.0, 1e-4  # break exact degeneracy
                push = (min_dist - d) * 0.5
                ux, uy = dx / d, dy / d
                pos[n1][0] -= push * ux
                pos[n1][1] -= push * uy
                pos[n2][0] += push * ux
                pos[n2][1] += push * uy
                moved = True

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

    Each unique material aggregates visit counts (sum) and total_reward (sum);
    r_DOS comes from the run's DOSCAR lookup (on the plain formula) and the node
    reward from score_by_method, so coloring matches the run. Keeps up to
    max_nodes by visit count. Node/edge insertion is sorted by identifier so the
    resulting layout is deterministic.
    """
    import networkx as nx

    by_id = {rec["id"]: rec for rec in tree["nodes"]}

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

    # Trim for legibility: keep the highest-visit unique materials. Sort by
    # (visits desc, identifier) so ties break deterministically.
    ordered = sorted(unique.items(), key=lambda kv: (-kv[1]["visits"], kv[0]))
    keep = [k for k, _ in ordered[:max_nodes]]
    keep_set = set(keep)
    if root_key is not None:
        keep_set.add(root_key)

    graph = nx.DiGraph()
    for key in sorted(keep_set):  # sorted insertion -> deterministic node order
        v = unique[key]
        e_hull = v["e_hulls"][0] if v["e_hulls"] else None
        r_dos = doscar_lookup.get_reward(str(key).split("|")[0])
        reward = (
            score_by_method(rollout_method, e_hull, r_dos, beta, gamma)
            if e_hull is not None else None
        )
        graph.add_node(
            key, identifier=key, visits=v["visits"], e_hull=e_hull,
            r_dos=r_dos, reward=reward,
            r_ehull=(ehull_reward(e_hull) if e_hull is not None else None),
        )
    for a, b in sorted(edges):
        if a in keep_set and b in keep_set:
            graph.add_edge(a, b)

    if root_key is None or root_key not in graph:
        root_key = next(iter(sorted(graph.nodes())), None)
    return graph, root_key


def plot_radial_tree(
    tree_path: str,
    out_path: str,
    config: Config,
    max_nodes: int = 60,
    top_n: int = 15,
    key_fn: Callable[[str], Hashable] = full_formula_key,
):
    """
    Draw the explored search tree radially in three panels.

    Panel (a) (large, left) colors nodes by the run's reward (via score_by_method
    on the config's rollout_method/beta/gamma) and labels the top-`top_n`
    compounds with their MCTS rank; panels (b) and (c) (stacked, right) recolor
    the same layout by the E_hull reward component and r_DOS. The MCTS root is
    starred and only the BFS spanning-tree edges are drawn (as arrows). The
    layout is deterministic (nodes sorted by identifier).

    Args:
        tree_path: path to a run's tree.json (written by save_results).
        out_path: where to write the .png.
        config: the run's Config; rollout_method, beta, gamma, cache_path and
            doscar_data_path drive the per-node reward/coloring.
        max_nodes: cap on drawn unique-material nodes (low-visit nodes trimmed).
        top_n: how many top-reward compounds get an MCTS-rank label on panel (a).
        key_fn: identifies a compound for the rank labels (default
            full_formula_key). The rank is derived from the run's own reward
            ordering, so no external table is needed.

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

    # Top-N MCTS ranks by the run's reward, keyed via key_fn (self-contained,
    # no external table). Nodes with no reward (unevaluated) are skipped.
    scored = [
        (graph.nodes[n]["reward"], n)
        for n in graph.nodes()
        if graph.nodes[n].get("reward") is not None
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    mcts_ranks = {key_fn(n): rank for rank, (_, n) in enumerate(scored[:top_n], start=1)}

    pos = _radial_layout(graph, root_key, radius_step=10.0)
    pos = {k: (-x, -y) for k, (x, y) in pos.items()}  # flip 180 deg for aesthetics
    pos = _spread_nodes(pos, root_key, min_dist=5.5)

    # Only the BFS spanning tree is drawn (revisit cross-links omitted for
    # legibility, matching the original product figure).
    tree_edges = list(nx.bfs_tree(graph, root_key).edges())

    node_list = sorted(graph.nodes())

    def _colors(attr, cmap, norm):
        out = []
        for n in node_list:
            v = graph.nodes[n].get(attr)
            if v is None or (isinstance(v, float) and np.isnan(v)) or norm is None:
                out.append("lightgray")
            else:
                out.append(cmap(norm(float(v))))
        return out

    def _norm(attr, positive_only=False):
        vals = [
            graph.nodes[n].get(attr) for n in node_list
            if graph.nodes[n].get(attr) is not None
            and not (isinstance(graph.nodes[n].get(attr), float) and np.isnan(graph.nodes[n][attr]))
        ]
        if positive_only:
            vals = [v for v in vals if v > 0]
        if not vals:
            return None
        return mcolors.Normalize(vmin=min(vals), vmax=max(vals))

    _rdbu = matplotlib.colormaps["RdBu"]
    cmap_reward = mcolors.LinearSegmentedColormap.from_list(
        "RdBu_bright", _rdbu(np.linspace(0.1, 0.9, 256))
    )
    cmap_rehull = cmap_reward
    cmap_rdos = matplotlib.colormaps["Blues"]

    norm_reward = _norm("reward")
    norm_rehull = _norm("r_ehull")
    norm_rdos = _norm("r_dos", positive_only=True)

    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(6.5, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], wspace=0.0)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_rehull = fig.add_subplot(gs[0, 1])
    ax_rdos = fig.add_subplot(gs[1, 1])

    panels = [
        (ax_main, _colors("reward", cmap_reward, norm_reward), cmap_reward,
         norm_reward, "Reward", "(a)"),
        (ax_rehull, _colors("r_ehull", cmap_rehull, norm_rehull), cmap_rehull,
         norm_rehull, r"$r_{E_{\mathrm{Hull}}}$", "(b)"),
        (ax_rdos, _colors("r_dos", cmap_rdos, norm_rdos), cmap_rdos,
         norm_rdos, r"$r_{\mathrm{DOS}}$", "(c)"),
    ]

    NODE_SIZE_MAIN = 120
    NODE_SIZE_SMALL = 24

    for ax, ncols, cmap_m, norm_m, label, abc in panels:
        is_main = ax is ax_main
        ns = NODE_SIZE_MAIN if is_main else NODE_SIZE_SMALL

        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=tree_edges, edge_color="dimgray",
            width=1.2 if is_main else 0.7, arrows=True,
            arrowsize=7 if is_main else 4, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08", node_size=ns,
            min_source_margin=2, min_target_margin=2,
        )
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=node_list, node_color=ncols,
            edgecolors="black", linewidths=0.5, node_size=ns,
        )

        if is_main:
            for n in node_list:
                rank = mcts_ranks.get(key_fn(n))
                if rank is None or n not in pos:
                    continue
                x, y = pos[n]
                ax.text(x, y, str(rank), ha="center", va="center", fontsize=8,
                        color="white", zorder=10)

        if root_key in pos:
            rx, ry = pos[root_key]
            ax.scatter([rx], [ry], s=ns * 0.55, marker="*", facecolor="gold",
                       edgecolors="black", linewidths=0.5, zorder=5)

        ax.text(0.02, 0.98, abc, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, weight="bold")

        if pos:
            xs = [pos[n][0] for n in graph.nodes() if n in pos]
            ys = [pos[n][1] for n in graph.nodes() if n in pos]
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            x_ext, y_ext = max(xs) - min(xs), max(ys) - min(ys)
            if is_main:
                half = max(x_ext, y_ext) / 2 * 1.15
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
            else:
                half_x = max(x_ext, y_ext) / 2 * 1.55
                half_y = y_ext / 2 * 1.12
                ax.set_xlim(cx - half_x, cx + half_x)
                ax.set_ylim(cy - half_y, cy + half_y)
        ax.set_axis_off()

        if norm_m is not None:
            sm = cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
            sm.set_array([])
            if is_main:
                cax = ax.inset_axes([0.20, 0.01, 0.55, 0.022])
                cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
                cbar.ax.xaxis.set_label_position("bottom")
                cbar.ax.xaxis.tick_bottom()
            else:
                cax = ax.inset_axes([0.92, 0.10, 0.05, 0.75])
                cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
                cbar.ax.yaxis.set_label_position("right")
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(label, fontsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig

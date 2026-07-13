#!/usr/bin/env python3
"""
Create radial tree visualization colored by composite reward.

Composite = beta*ehull_reward(e_hull) + gamma*r_DOS, beta=1.0. This is the
gamma-normalized variant of the study: gamma is fixed to 1/(max raw r_DOS
across the 108 U-only compounds) instead of being loaded from config.json
(config.json stays at the calibrated gamma=0.0001 for that other study).

Blue-red color scheme: blue = higher composite (better), red = lower composite (worse).
"""

import pickle
import math
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import pandas as pd
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcts_crystal.node import ehull_reward

# 1 / (max raw r_DOS across the 108 U-only compounds, U-Pb-Mn / Mn6Pb6U =
# 2516.1664410449775) - see generate_figures.py's NORMALIZED_GAMMA.
DEFAULT_GAMMA = 1.0 / 2516.1664410449775


# Element categories
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


def reorder_formula_unicode(formula):
    """Reorder formula to RE TM₆ GIV₆ with unicode subscripts."""
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    re_elem, tm_elem, giv_elem = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in F_BLOCK:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_elem = elem
        elif elem in GROUP_IV:
            giv_elem = elem

    if re_elem and tm_elem and giv_elem:
        return f"{re_elem}{tm_elem}₆{giv_elem}₆"
    return formula


def compute_composite(e_hull, r_dos, beta=1.0, gamma=DEFAULT_GAMMA):
    """Compute composite reward (E_form is not part of the reward)."""
    return beta * ehull_reward(e_hull) + gamma * r_dos


def count_descendants(G, node):
    """Count descendants including self using NetworkX reachability to avoid recursion/cycles."""
    try:
        desc = nx.descendants(G, node)
        return 1 + len(desc)
    except Exception:
        return 1


def radial_layout(G, root, radius_step=3.0):
    """Compute a radial tree layout that keeps each subtree within an angular wedge.

    Depth (radius) comes from shortest-path distance from the root. Angle is
    assigned recursively: each node's children split its angular span,
    weighted by subtree size, so a branch's descendants stay clustered near
    that branch instead of being scattered around the full circle. This is
    what keeps edges short and non-crossing rather than radiating across the
    whole diagram.
    """
    pos = {}
    if root is None or root not in G:
        return pos

    try:
        depths = nx.single_source_shortest_path_length(G, root)
    except Exception:
        nodes = list(G.nodes())
        n = len(nodes)
        for i, nkey in enumerate(nodes):
            theta = 2.0 * math.pi * (i / max(1, n))
            pos[nkey] = (radius_step * math.cos(theta), radius_step * math.sin(theta))
        return pos

    # Build a BFS spanning tree so every node has exactly one parent for the
    # purposes of layout, even if the underlying graph has cross-links.
    bfs_tree = nx.bfs_tree(G, root)
    children = {n: list(bfs_tree.successors(n)) for n in bfs_tree.nodes()}

    # Subtree leaf counts, used to proportionally size each branch's wedge.
    leaf_count = {}

    def count_leaves(n):
        kids = children.get(n, [])
        if not kids:
            leaf_count[n] = 1
        else:
            leaf_count[n] = sum(count_leaves(c) for c in kids)
        return leaf_count[n]

    count_leaves(root)

    angle = {}

    def assign_angles(n, theta_start, theta_end):
        angle[n] = 0.5 * (theta_start + theta_end)
        kids = children.get(n, [])
        if not kids:
            return
        total = sum(leaf_count[c] for c in kids) or len(kids)
        cursor = theta_start
        span = theta_end - theta_start
        for c in kids:
            child_span = span * (leaf_count.get(c, 1) / total)
            assign_angles(c, cursor, cursor + child_span)
            cursor += child_span

    assign_angles(root, 0.0, 2.0 * math.pi)

    # Any node unreachable in the BFS tree (shouldn't normally happen) gets a
    # fallback angle so layout never silently drops it.
    fallback_nodes = [n for n in G.nodes() if n not in angle]
    for i, n in enumerate(fallback_nodes):
        angle[n] = 2.0 * math.pi * (i / max(1, len(fallback_nodes)))

    for n in G.nodes():
        d = depths.get(n, 0)
        th = angle.get(n, 0.0)
        r = d * radius_step
        pos[n] = (r * math.cos(th), r * math.sin(th))
    pos[root] = (0.0, 0.0)

    return pos


def build_tree_data(mcts):
    """Build tree data from MCTS, computing composite for each node."""
    tree_data = {}

    # Load doscar rewards for r_DOS lookup
    doscar_dict = {}
    if hasattr(mcts, 'stat_dict'):
        for formula, stats in mcts.stat_dict.items():
            if len(stats) >= 6:
                doscar_dict[formula] = stats[5]  # r_dos at index 5

    def traverse(node, node_id=0, parent_id=None):
        formula = node.get_chemical_formula()
        e_form = node.e_form if hasattr(node, 'e_form') else None
        e_hull = node.e_above_hull if hasattr(node, 'e_above_hull') else None
        r_dos = doscar_dict.get(formula, 0.0)

        composite = None
        if e_hull is not None:
            composite = compute_composite(e_hull, r_dos)

        tree_data[node_id] = {
            "formula": formula,
            "e_form": e_form,
            "e_hull": e_hull,
            "r_dos": r_dos,
            "composite": composite,
            "parent_id": parent_id,
            "visit_count": node.t_of_visit,
            "total_reward": node.total_reward if hasattr(node, 'total_reward') else 0.0,
        }

        for i, child in enumerate(node.children):
            child_id = node_id * 100 + i + 1
            traverse(child, child_id, node_id)

    traverse(mcts.root, node_id=0)
    return tree_data


def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None,
                         help='Directory containing mcts_object.pkl to visualize '
                              '(default: this script\'s own directory, i.e. the '
                              'study\'s main run). Figures still always get saved '
                              'into this script\'s figures/ directory.')
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else script_dir

    # Load MCTS pickle
    pkl_path = run_dir / 'mcts_object.pkl'
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        return 1

    print("Loading MCTS pickle...")
    with open(pkl_path, 'rb') as f:
        mcts = pickle.load(f)

    # Optional starting-material annotation, written by generate_figures.py's
    # describe_mcts_run_starting_material() (avoids needing to import that
    # module here, which would create a circular import the other way).
    start_info = {}
    start_info_path = run_dir / 'starting_material_info.json'
    if start_info_path.exists():
        try:
            with open(start_info_path) as f:
                start_info = json.load(f)
        except Exception:
            start_info = {}

    # Build tree data with composite scores
    print("Building tree data...")
    tree_data = build_tree_data(mcts)
    print(f"  {len(tree_data)} nodes in tree")

    # Build a condensed networkx graph collapsing nodes by unique formula
    # Aggregate composite (max), r_dos (max), and sum visit_counts for unique formulas.
    unique_map = {}
    edges = set()
    root_formula = None

    for nid, info in tree_data.items():
        formula = info.get('formula') or f'UNK_{nid}'
        # normalize formula key as string
        key = str(formula)
        if key not in unique_map:
            unique_map[key] = {
                'formulas': [formula],
                'composites': [],
                'e_hulls': [],
                'r_doss': [],
                'visit_count': 0,
                'total_reward': 0.0,
            }
        if info.get('composite') is not None:
            unique_map[key]['composites'].append(info['composite'])
        if info.get('e_hull') is not None:
            unique_map[key]['e_hulls'].append(info['e_hull'])
        unique_map[key]['r_doss'].append(info.get('r_dos', 0.0))
        unique_map[key]['visit_count'] += int(info.get('visit_count', 0) or 0)
        unique_map[key]['total_reward'] += float(info.get('total_reward', 0.0) or 0.0)

        parent = info.get('parent_id')
        if parent is None:
            root_formula = key
        else:
            parent_formula = tree_data.get(parent, {}).get('formula')
            if parent_formula is None:
                parent_key = f'UNK_{parent}'
            else:
                parent_key = str(parent_formula)
            if parent_key != key:
                edges.add((parent_key, key))

    # Decide which unique nodes to keep (trim low-visit nodes for clarity)
    all_nodes = list(unique_map.items())
    # Keep nodes with at least 2 visits by default
    nodes_keep = [k for k, v in unique_map.items() if v['visit_count'] >= 2]
    # If too many nodes remain, keep top 60 by visit_count
    if len(nodes_keep) > 60:
        sorted_nodes = sorted(unique_map.items(), key=lambda kv: kv[1]['visit_count'], reverse=True)
        nodes_keep = [k for k, _ in sorted_nodes[:60]]
    # If too few nodes kept (e.g., most have 1 visit), instead keep top 40 by visit_count
    if len(nodes_keep) < 10:
        sorted_nodes = sorted(unique_map.items(), key=lambda kv: kv[1]['visit_count'], reverse=True)
        nodes_keep = [k for k, _ in sorted_nodes[:40]]

    nodes_keep_set = set(nodes_keep)

    # Build networkx graph from filtered unique_map
    G = nx.DiGraph()
    for key in nodes_keep:
        info = unique_map[key]
        comp_vals = [v for v in info['composites'] if v is not None]
        comp_agg = max(comp_vals) if comp_vals else None
        rdos_vals = [float(v) for v in info['r_doss'] if v is not None]
        rdos_agg = max(rdos_vals) if rdos_vals else 0.0
        eh_vals = [v for v in info['e_hulls'] if v is not None]
        eh_agg = eh_vals[0] if eh_vals else None
        vc = info['visit_count']
        q_per_n = info['total_reward'] / vc if vc > 0 else None
        G.add_node(key, formula=key, composite=comp_agg, e_hull=eh_agg, r_dos=rdos_agg,
                   visit_count=vc, q_per_n=q_per_n)

    for a, b in edges:
        if a in nodes_keep_set and b in nodes_keep_set:
            if a not in G:
                G.add_node(a, formula=a, composite=None, e_hull=None, r_dos=0.0, visit_count=0)
            if b not in G:
                G.add_node(b, formula=b, composite=None, e_hull=None, r_dos=0.0, visit_count=0)
            G.add_edge(a, b)

    if root_formula is None or root_formula not in G:
        # fallback to arbitrary kept node
        root_formula = next(iter(G.nodes())) if len(G.nodes()) else None

    print(f"  {len(G.nodes())} nodes after trimming (from {len(unique_map)} unique)")

    # Compute layout and flip 180 degrees
    # Increase radius_step to spread points more visibly
    pos = radial_layout(G, root_formula, radius_step=4.0) if root_formula is not None else {}
    pos = {nid: (-x, -y) for nid, (x, y) in pos.items()} if pos else {}

    # Extract metrics from graph node attributes for coloring
    composites = {n: G.nodes[n].get('composite', None) for n in G.nodes()}
    ehull_rewards = {}
    r_doss = {}
    q_per_ns = {}
    for n in G.nodes():
        eh = G.nodes[n].get('e_hull', None)
        try:
            ehull_rewards[n] = ehull_reward(eh) if eh is not None else np.nan
        except Exception:
            ehull_rewards[n] = np.nan
        r_doss[n] = float(G.nodes[n].get('r_dos', 0.0) or 0.0)
        qn = G.nodes[n].get('q_per_n', None)
        q_per_ns[n] = float(qn) if qn is not None else np.nan

    # Choose readable sequential colormaps
    cmap_comp = matplotlib.colormaps['viridis']
    arr_comp = [v for v in composites.values() if v is not None and not pd.isna(v)]
    norm_comp = mcolors.Normalize(vmin=min(arr_comp), vmax=max(arr_comp)) if arr_comp else None

    cmap_ehull = matplotlib.colormaps['Oranges']
    arr_eh = [v for v in ehull_rewards.values() if v is not None and not pd.isna(v)]
    norm_ehull = mcolors.Normalize(vmin=min(arr_eh), vmax=max(arr_eh)) if arr_eh else None

    cmap_rdos = matplotlib.colormaps['Greens']
    # Color by gamma * r_DOS (the actual composite-score component), not raw r_DOS
    arr_r = [v * DEFAULT_GAMMA for v in r_doss.values() if v is not None and not pd.isna(v)]
    norm_rdos = mcolors.Normalize(vmin=min(arr_r), vmax=max(arr_r)) if arr_r else None

    cmap_qpern = matplotlib.colormaps['Purples']
    arr_qn = [v for v in q_per_ns.values() if not pd.isna(v)]
    norm_qpern = mcolors.Normalize(vmin=min(arr_qn), vmax=max(arr_qn)) if arr_qn else None

    if norm_comp is None:
        print("No composite scores available")
        return 1

    def node_colors_for_from_graph(attr_name, cmap, norm, scale=1.0):
        cols = []
        for nid in G.nodes():
            val = G.nodes[nid].get(attr_name, None)
            if val is None or (isinstance(val, float) and pd.isna(val)) or norm is None:
                cols.append('lightgray')
            else:
                cols.append(cmap(norm(float(val) * scale)))
        return cols

    node_colors_comp = node_colors_for_from_graph('composite', cmap_comp, norm_comp)
    node_colors_ehull = node_colors_for_from_graph('e_hull', cmap_ehull, norm_ehull)
    # Color by gamma * r_dos, explicitly labeled below
    node_colors_rdos = node_colors_for_from_graph('r_dos', cmap_rdos, norm_rdos, scale=DEFAULT_GAMMA)
    node_colors_qpern = node_colors_for_from_graph('q_per_n', cmap_qpern, norm_qpern)

    # Set global font size to 10pt for consistent publication text
    plt.rcParams.update({'font.size': 10})

    # Smaller node markers leave room between circles/arrows at this node density
    NODE_SIZE = 70

    # Most edges in this graph are "revisit" links: the same composition reached
    # from a second, later parent after already being placed via its first
    # parent. Drawing every one of those as a bold arrow is what produced the
    # tangled web in the original figure. Split edges into the BFS spanning
    # tree (the structure the radial layout is actually built from) and the
    # remaining cross-links, then render the cross-links as faint background
    # threads and the spanning tree as the bold, arrowed structure on top.
    if root_formula is not None and root_formula in G:
        bfs_tree_edges = set(nx.bfs_tree(G, root_formula).edges())
    else:
        bfs_tree_edges = set()
    tree_edges = [e for e in G.edges() if e in bfs_tree_edges]
    cross_edges = [e for e in G.edges() if e not in bfs_tree_edges]

    # Create a wide figure: 8in wide x 2.75in tall with 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.75), constrained_layout=True)

    panels = [
        (axes[0], node_colors_comp, cmap_comp, norm_comp, 'Composite Score'),
        (axes[1], node_colors_ehull, cmap_ehull, norm_ehull, r"$r_{E_{\mathrm{Hull}}}$"),
        (axes[2], node_colors_rdos, cmap_rdos, norm_rdos,
         r"$\alpha_{\mathrm{DOS}} \cdot r_{\mathrm{DOS}}$"),
        (axes[3], node_colors_qpern, cmap_qpern, norm_qpern, r"$Q/N$"),
    ]

    labels_abc = ['(a)', '(b)', '(c)', '(d)']
    for i, (ax, ncols, cmap_m, norm_m, label) in enumerate(panels):
        # Draw edges first so smaller nodes sit cleanly on top of arrow tips.
        # Spanning-tree edges: the actual branch structure, drawn bold with arrows.
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=tree_edges,
                               edge_color='dimgray', width=0.7, arrows=True, arrowsize=5,
                               arrowstyle='-|>', connectionstyle='arc3,rad=0.08',
                               node_size=NODE_SIZE, min_source_margin=2, min_target_margin=2)
        # Draw nodes (shrunk so adjacent circles/arrows no longer overlap)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncols, edgecolors='black',
                               linewidths=0.5, node_size=NODE_SIZE)
        # Highlight the MCTS root as the starting node without changing its color encoding
        if root_formula is not None and root_formula in pos:
            rx, ry = pos[root_formula]
            ax.scatter([rx], [ry], s=NODE_SIZE * 0.55, marker='*', facecolor='gold',
                       edgecolors='black', linewidths=0.5, zorder=5)
        # add panel letter in top-left
        try:
            abc = labels_abc[i]
        except Exception:
            abc = ''
        ax.text(0.02, 0.98, abc, transform=ax.transAxes, va='top', ha='left', fontsize=10, weight='bold')
        ax.axis('equal')
        # colorbar for each panel; place metric label below the colorbar
        sm = cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.08, pad=0.08, aspect=20)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label, fontsize=10)
        try:
            cbar.ax.xaxis.set_label_position('bottom')
            cbar.ax.xaxis.tick_bottom()
        except Exception:
            pass

    caption = '★ starting node (MCTS root)    bold arrow = expansion step'
    if start_info.get('label'):
        dist_str = f", d={start_info['distance']} to global best" if start_info.get('distance') is not None else ''
        caption += f"\nStarting material: {start_info['label']}{dist_str}"
    fig.text(0.5, -0.03, caption, ha='center', va='top', fontsize=7)

    # Ensure figures directory exists and save into it
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / 'radial_tree_composite.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())

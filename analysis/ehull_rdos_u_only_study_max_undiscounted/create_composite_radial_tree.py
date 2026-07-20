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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcts_crystal.node import ehull_reward
from synthesized_compounds import SYNTHESIZED_COMPOUNDS

# 1 / (max raw r_DOS across the 108 U-only compounds, U-Pb-Mn / Mn6Pb6U =
# 2516.1664410449775) - see generate_figures.py's NORMALIZED_GAMMA.
DEFAULT_GAMMA = 1.0 / 2516.1664410449775
# Product-mode gamma (raw r_DOS, not normalised) and its composite ceiling.
PRODUCT_GAMMA = 1.0
PRODUCT_VMAX = 1686.75  # UTi6Sn6 product reward — anchors colorbar at 1.0


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


def _formula_key(name):
    """Sorted tuple of non-f-block elements — matches generate_figures.py's key."""
    s = str(name)
    if '-' in s:
        parts = [p for p in re.split('[^A-Za-z]', s) if p]
    else:
        parts = re.findall(r'[A-Z][a-z]?', s)
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


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
    """Additive composite: beta*r_Ehull + gamma*r_DOS."""
    return beta * ehull_reward(e_hull) + gamma * r_dos


def compute_composite_product(e_hull, r_dos, **_):
    """Multiplicative composite: r_Ehull × (PRODUCT_GAMMA × r_DOS)."""
    return ehull_reward(e_hull) * (PRODUCT_GAMMA * float(r_dos))


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


def _spread_nodes(pos, root, min_dist=2.0, max_iters=600):
    """Push overlapping nodes apart while preserving each node's original radius.

    After radial_layout, siblings in narrow wedges can land on top of each
    other.  This iteratively resolves collisions by pushing node pairs apart
    along their connecting vector, then renormalising each node back to its
    original radius so the concentric-ring depth hierarchy is preserved.
    """
    pos = {k: list(v) for k, v in pos.items()}
    # Store original radii so we can renormalise after every push sweep.
    original_radii = {}
    for k, v in pos.items():
        if k == root:
            continue
        r = math.sqrt(v[0] ** 2 + v[1] ** 2)
        original_radii[k] = r if r > 1e-9 else min_dist

    movable = [n for n in pos if n != root]
    lcg = 12345  # deterministic LCG state for tie-breaking

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
                    # Break exact degeneracy with a deterministic nudge.
                    lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
                    a = lcg * (2 * math.pi / 0x7FFFFFFF)
                    dx, dy, d = math.cos(a) * 1e-4, math.sin(a) * 1e-4, 1e-4
                push = (min_dist - d) * 0.5
                ux, uy = dx / d, dy / d
                pos[n1][0] -= push * ux
                pos[n1][1] -= push * uy
                pos[n2][0] += push * ux
                pos[n2][1] += push * uy
                moved = True

        # Renormalise every movable node back to its original radius so the
        # concentric-ring structure is maintained.
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


def build_tree_data(mcts, comp_fn=compute_composite):
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
            composite = comp_fn(e_hull, r_dos)

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
    import random as _random
    _random.seed(42)
    np.random.seed(42)

    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None,
                         help='Directory containing mcts_object.pkl to visualize '
                              '(default: this script\'s own directory, i.e. the '
                              'study\'s main run). Figures still always get saved '
                              'into this script\'s figures/ directory.')
    parser.add_argument('--product', action='store_true',
                        help='Product-mode: use multiplicative composite r_Ehull×r_DOS '
                             'with gamma=1.0 (raw DOS). Defaults run-dir to product_mode/seed_0 '
                             'and saves to product_mode/figures/.')
    args = parser.parse_args()
    product_mode = args.product

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif product_mode:
        run_dir = script_dir / 'product_mode' / 'seed_0'
    else:
        run_dir = script_dir

    comp_fn = compute_composite_product if product_mode else compute_composite

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
    tree_data = build_tree_data(mcts, comp_fn=comp_fn)
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

    # Parse top-15 table now so we can force-include those nodes regardless of
    # visit count — if a compound appears in the table it was explored by MCTS
    # and must be visible in the tree.
    if product_mode:
        _table_path = Path(__file__).parent / 'product_mode' / 'tables' / 'top15_u_only_product.tex'
    else:
        _table_path = Path(__file__).parent / 'tables' / 'top15_u_only.tex'
    _table_keys = set()
    if _table_path.exists():
        with open(_table_path) as _tf:
            for _tline in _tf:
                _m2 = re.match(r'\s*\d+\s*&\s*\d+\s*&\s*(.*?)\s*&', _tline)
                if _m2:
                    _elems = re.findall(r'[A-Z][a-z]?', _m2.group(1))
                    _tkey = tuple(sorted(e for e in _elems if e not in F_BLOCK))
                    if _tkey:
                        _table_keys.add(_tkey)

    # Decide which unique nodes to keep (trim low-visit nodes for clarity)
    # Keep nodes with at least 2 visits by default
    nodes_keep = [k for k, v in unique_map.items() if v['visit_count'] >= 2]
    # Always include top-15 table compounds even if visit_count == 1
    for k in unique_map:
        if _formula_key(k) in _table_keys and k not in nodes_keep:
            nodes_keep.append(k)
    # If too many nodes remain, keep top 60 by visit_count (but keep table nodes)
    if len(nodes_keep) > 60:
        sorted_nodes = sorted(unique_map.items(), key=lambda kv: kv[1]['visit_count'], reverse=True)
        force_keep = {k for k in unique_map if _formula_key(k) in _table_keys}
        nodes_keep = list(force_keep | {k for k, _ in sorted_nodes[:60]})
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

    # Inject top-15 table compounds discovered only in secondary runs (e.g.
    # d5_start_run).  Each is connected to its nearest-neighbour in the main
    # tree by edit distance along the TM / GIV design-space axes so that the
    # edge represents the approximate rollout origin.
    TM_ORDER = ['Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
                'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg']
    GIV_ORDER = ['Si','Ge','Sn','Pb']
    _TM_IDX = {e: i for i, e in enumerate(TM_ORDER)}
    _GIV_IDX = {e: i for i, e in enumerate(GIV_ORDER)}

    def _design_dist(fa, fb):
        def _dec(f):
            els = re.findall(r'[A-Z][a-z]?', str(f))
            tm = next((e for e in els if e in _TM_IDX), None)
            giv = next((e for e in els if e in _GIV_IDX), None)
            return tm, giv
        tm_a, giv_a = _dec(fa); tm_b, giv_b = _dec(fb)
        d = abs(_TM_IDX.get(tm_a, -99) - _TM_IDX.get(tm_b, -99)) if (tm_a and tm_b) else 99
        d += abs(_GIV_IDX.get(giv_a, -99) - _GIV_IDX.get(giv_b, -99)) if (giv_a and giv_b) else 99
        return d

    secondary_edges = set()
    if product_mode:
        # Inject top-15 compounds found in other seeds (seed_1..4) that are
        # missing from the primary seed_0 tree.
        existing_fkeys = {_formula_key(n) for n in G.nodes()}
        for _seed in range(1, 5):
            _seed_csv = script_dir / 'product_mode' / f'seed_{_seed}' / 'all_compounds.csv'
            if not _seed_csv.exists():
                continue
            _df_s = pd.read_csv(_seed_csv)
            for _, row in _df_s.iterrows():
                formula = str(row.get('formula', row.get('name', '')))
                fkey = _formula_key(formula)
                if fkey not in _table_keys or fkey in existing_fkeys:
                    continue
                e_hull = float(row['e_above_hull'])
                r_dos = float(row.get('dos_reward', row.get('r_DOS', 0.0)))
                comp = compute_composite_product(e_hull, r_dos)
                vc = int(row.get('visit_count', 1))
                G.add_node(formula, formula=formula, composite=comp, e_hull=e_hull,
                           r_dos=r_dos, visit_count=vc, q_per_n=comp)
                existing_fkeys.add(fkey)
                nearest = min((n for n in G.nodes() if n != formula),
                              key=lambda n: _design_dist(formula, n),
                              default=root_formula)
                secondary_edges.add((nearest, formula))
                G.add_edge(nearest, formula)
                print(f"  Added from seed_{_seed}: {formula} -> nearest={nearest}")
    else:
        d5_csv = script_dir / 'd5_start_run' / 'all_compounds.csv'
        if d5_csv.exists():
            df_d5 = pd.read_csv(d5_csv)
            existing_fkeys = {_formula_key(n) for n in G.nodes()}
            for _, row in df_d5.iterrows():
                formula = str(row['formula'])
                fkey = _formula_key(formula)
                if fkey not in _table_keys or fkey in existing_fkeys:
                    continue
                e_hull = float(row['e_above_hull'])
                r_dos = float(row['dos_reward'])
                comp = compute_composite(e_hull, r_dos)
                vc = int(row['visit_count'])
                G.add_node(formula, formula=formula, composite=comp, e_hull=e_hull,
                           r_dos=r_dos, visit_count=vc, q_per_n=float(row['best_reward']))
                existing_fkeys.add(fkey)
                nearest = min((n for n in G.nodes() if n != formula),
                              key=lambda n: _design_dist(formula, n),
                              default=root_formula)
                secondary_edges.add((nearest, formula))
                G.add_edge(nearest, formula)
                print(f"  Added from d5_start_run: {formula} -> nearest={nearest}")

    print(f"  {len(G.nodes())} nodes after trimming (from {len(unique_map)} unique)")

    # Override node attributes with ground-truth values from the study's output CSVs.
    # build_tree_data recomputes composite using the current doscar file, which may have
    # drifted since the study was run.  The CSVs store e_above_hull and dos_reward as
    # they were at run time, so we use those as the authoritative source for colors.
    if product_mode:
        _gt_csvs = sorted((script_dir / 'product_mode').glob('seed_*/all_compounds.csv'))
    else:
        _gt_csvs = [script_dir / 'all_compounds.csv',
                    script_dir / 'd5_start_run' / 'all_compounds.csv']
    for _gt_csv in _gt_csvs:
        if _gt_csv.exists():
            _df_gt = pd.read_csv(_gt_csv)
            for _, _row in _df_gt.iterrows():
                _f = str(_row['formula'])
                if _f not in G:
                    continue
                _eh = float(_row['e_above_hull'])
                _rdos = float(_row['dos_reward'])
                G.nodes[_f]['e_hull'] = _eh
                G.nodes[_f]['r_dos'] = _rdos
                G.nodes[_f]['composite'] = comp_fn(_eh, _rdos)

    # MCTS ranks and synthesized set for panel (b) annotations.
    # Read both MCTS rank and True Rank from the committed table.
    # Always use product-mode table so coloring and rank labels are consistent.
    table_path = script_dir / 'product_mode' / 'tables' / 'top15_u_only_product.tex'
    mcts_ranks_from_table = {}   # formula_key -> mcts_rank (label shown on node)
    true_ranks_from_table = {}   # formula_key -> true_rank (kept for reference)
    table_composites = {}        # formula_key -> committed composite value (ground truth)
    if table_path.exists():
        with open(table_path) as _f:
            for _line in _f:
                # Columns: MCTS & True & Compound & E_Hull & r_Ehull & r_DOS & Product Reward & Synth
                # E_Hull can be negative, so use [\d.-]+ for that column too.
                _m = re.match(
                    r'\s*(\d+)\s*&\s*(\d+)\s*&\s*(.*?)\s*&'            # rank, true_rank, name
                    r'\s*[\d.-]+\s*&\s*[\d.-]+\s*&\s*[\d.]+\s*&\s*([\d.]+)\s*&',  # skip 3, capture Product
                    _line,
                )
                if _m:
                    _mcts_rank = int(_m.group(1))
                    _true_rank = int(_m.group(2))
                    _elems = re.findall(r'[A-Z][a-z]?', _m.group(3))
                    _key = tuple(sorted(e for e in _elems if e not in F_BLOCK))
                    if _key:
                        mcts_ranks_from_table[_key] = _mcts_rank
                        true_ranks_from_table[_key] = _true_rank
                        try:
                            table_composites[_key] = float(_m.group(4))
                        except ValueError:
                            pass
    synth_keys = {_formula_key(s) for s in SYNTHESIZED_COMPOUNDS}

    # Compute layout, flip 180 degrees, then resolve any node overlaps.
    pos = radial_layout(G, root_formula, radius_step=10.0) if root_formula is not None else {}
    pos = {nid: (-x, -y) for nid, (x, y) in pos.items()} if pos else {}
    if pos and root_formula is not None:
        pos = _spread_nodes(pos, root_formula, min_dist=5.5)

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

    # All panels use the same white→blue sequential colormap.
    # Truncate RdBu to [0.1, 0.9] so the endpoints are bright red/blue rather than near-black.
    _rdbu_base = matplotlib.colormaps['RdBu']
    _rdbu_bright = mcolors.LinearSegmentedColormap.from_list(
        'RdBu_bright', _rdbu_base(np.linspace(0.1, 0.9, 256))
    )
    # Panel (a): diverging red-to-blue colormap centred at 0 for product reward.
    cmap_comp = _rdbu_bright

    # Color all nodes by product reward (r_Ehull × r_DOS, gamma=1); diverging norm centred at 0.
    composite_for_color = {}
    for _n in G.nodes():
        _eh = G.nodes[_n].get('e_hull', None)
        _rdos = float(G.nodes[_n].get('r_dos', 0.0) or 0.0)
        if _eh is not None:
            composite_for_color[_n] = ehull_reward(_eh) * _rdos
        else:
            composite_for_color[_n] = np.nan

    _comp_vals = [v for v in composite_for_color.values() if not pd.isna(v)]
    if _comp_vals:
        norm_comp = mcolors.Normalize(vmin=-2250, vmax=2250)
    else:
        norm_comp = None

    # r_EHull: same bright RdBu, fixed range [-1, 1].
    cmap_ehull = _rdbu_bright
    arr_eh = [v for v in ehull_rewards.values() if v is not None and not pd.isna(v)]
    norm_ehull = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0) if arr_eh else None

    cmap_rdos = matplotlib.colormaps['Blues']
    # Always use raw r_DOS (no gamma scaling); auto-scale to data range.
    arr_r = [v for v in r_doss.values() if v is not None and not pd.isna(v) and v > 0]
    norm_rdos = mcolors.Normalize(vmin=0.0, vmax=max(arr_r)) if arr_r else None

    cmap_qpern = matplotlib.colormaps['Blues']
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

    node_list_for_color = list(G.nodes())
    node_colors_comp = [
        cmap_comp(norm_comp(composite_for_color[n]))
        if (n in composite_for_color and not pd.isna(composite_for_color[n]) and norm_comp is not None)
        else 'lightgray'
        for n in node_list_for_color
    ]
    # Color by ehull_reward (transformed) so the diverging norm centred at 0 is correct.
    node_colors_ehull = [
        cmap_ehull(norm_ehull(float(ehull_rewards[nid])))
        if (ehull_rewards.get(nid) is not None and not pd.isna(ehull_rewards[nid]) and norm_ehull is not None)
        else 'lightgray'
        for nid in G.nodes()
    ]
    # Raw r_DOS (scale=1.0) — no gamma normalisation.
    node_colors_rdos = node_colors_for_from_graph('r_dos', cmap_rdos, norm_rdos, scale=1.0)

    # Set global font size to 10pt for consistent publication text
    plt.rcParams.update({'font.size': 10})

    # Smaller node markers leave room between circles/arrows at this node density
    NODE_SIZE = 60

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

    # Layout: large combined panel (a) on the left (2/3 width, full height),
    # r_EHull (b) and r_DOS (c) stacked vertically on the right (1/3 width each).
    fig = plt.figure(figsize=(6.5, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], wspace=0.0)
    ax_main  = fig.add_subplot(gs[:, 0])   # combined composite + rank labels
    ax_ehull = fig.add_subplot(gs[0, 1])   # r_EHull
    ax_rdos  = fig.add_subplot(gs[1, 1])   # r_DOS

    rdos_label = r"$r_{\mathrm{DOS}}$"
    comp_label = 'Product Reward'

    ranked_keys = set(mcts_ranks_from_table.keys())

    panels = [
        (ax_main,  node_colors_comp,  cmap_comp,  norm_comp,  comp_label,                    '(a)'),
        (ax_ehull, node_colors_ehull, cmap_ehull, norm_ehull, r"$r_{E_{\mathrm{Hull}}}$",    '(b)'),
        (ax_rdos,  node_colors_rdos,  cmap_rdos,  norm_rdos,  rdos_label,                    '(c)'),
    ]

    # Larger nodes for the spacious main panel; smaller for the compact right panels.
    NODE_SIZE_MAIN  = 120
    NODE_SIZE_SMALL = 24

    for ax, ncols, cmap_m, norm_m, label, abc in panels:
        is_main = ax is ax_main
        ns = NODE_SIZE_MAIN if is_main else NODE_SIZE_SMALL

        # Dashed secondary edges omitted from the main panel for clarity.
        if not is_main:
            sec_edgelist = [e for e in secondary_edges if e[0] in pos and e[1] in pos]
            if sec_edgelist:
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=sec_edgelist,
                                       edge_color='#AAAAAA', width=0.5, arrows=True,
                                       arrowsize=4, arrowstyle='-|>',
                                       connectionstyle='arc3,rad=0.15',
                                       style='dashed',
                                       node_size=ns, min_source_margin=2, min_target_margin=2)

        arrow_sz = 7 if is_main else 4
        edge_w = 1.2 if is_main else 0.7
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=tree_edges,
                               edge_color='dimgray', width=edge_w, arrows=True,
                               arrowsize=arrow_sz, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.08',
                               node_size=ns, min_source_margin=2, min_target_margin=2)

        # All nodes at full opacity.
        node_list = list(G.nodes())
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                               node_color=ncols, edgecolors='black',
                               linewidths=0.5, node_size=ns)

        # Rank labels on the main panel only; scale font with node size.
        if is_main:
            for node in G.nodes():
                key = _formula_key(node)
                rank = mcts_ranks_from_table.get(key)
                if rank is None:
                    continue
                x, y = pos.get(node, (None, None))
                if x is None:
                    continue
                ax.text(x, y, str(rank), ha='center', va='center', fontsize=8,
                        fontweight='normal', color='white', zorder=10)

        # Gold star on MCTS root.
        if root_formula is not None and root_formula in pos:
            rx, ry = pos[root_formula]
            ax.scatter([rx], [ry], s=ns * 0.55, marker='*', facecolor='gold',
                       edgecolors='black', linewidths=0.5, zorder=5)

        ax.text(0.02, 0.98, abc, transform=ax.transAxes, va='top', ha='left',
                fontsize=10, weight='bold')
        # Set equal, centered limits so the graph fills the axes box and all
        # panels align at their edges rather than shrinking to maintain aspect.
        if pos:
            _xs = [pos[n][0] for n in G.nodes() if n in pos]
            _ys = [pos[n][1] for n in G.nodes() if n in pos]
            _cx = (max(_xs) + min(_xs)) / 2
            _cy = (max(_ys) + min(_ys)) / 2
            _x_ext = max(_xs) - min(_xs)
            _y_ext = max(_ys) - min(_ys)
            if is_main:
                _half = max(_x_ext, _y_ext) / 2 * 1.15
                ax.set_xlim(_cx - _half, _cx + _half)
                ax.set_ylim(_cy - _half, _cy + _half)
            else:
                # x: pad based on full extent; y: pad based on actual y extent
                # so the tree stretches vertically to fill the panel.
                _half_x = max(_x_ext, _y_ext) / 2 * 1.55
                _half_y = _y_ext / 2 * 1.12
                ax.set_xlim(_cx - _half_x, _cx + _half_x)
                ax.set_ylim(_cy - _half_y, _cy + _half_y)
        ax.set_axis_off()

        # Colorbar inset inside each panel (bottom-centre of the axes).
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
            cax = ax.inset_axes([0.92, 0.10, 0.05, 0.75])  # vertical, right side
            cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(label, fontsize=8, labelpad=4)
            cbar.ax.yaxis.set_label_position('right')

    # Ensure figures directory exists and save into it
    if product_mode:
        figures_dir = script_dir / 'product_mode' / 'figures'
        output_filename = 'radial_tree_composite_product.png'
    else:
        figures_dir = script_dir / 'figures'
        output_filename = 'radial_tree_composite.png'
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())

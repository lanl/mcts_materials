"""
Plotting for completed MCTS runs.

Three material-agnostic plots:
    plot_convergence          : best reward and unique-material count vs iteration
    plot_property_distribution : histogram of one property across evaluated nodes
    plot_search_tree           : radial tree, nodes colored by own_reward,
                                 sized by visits

matplotlib (and networkx for the tree) are imported lazily inside each
function, so importing this module does not require the [viz] extra. Each
function saves to `save_path` if given and returns the matplotlib Figure.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Optional

from ..core.mcts import MCTS


def plot_convergence(mcts: MCTS, save_path: Optional[str] = None):
    """
    Plot best-reward-so-far and cumulative unique-material count per iteration.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    iterations = range(len(mcts.reward_history))

    fig, ax_reward = plt.subplots(figsize=(8, 5))
    ax_reward.plot(
        iterations, mcts.reward_history,
        color="tab:blue", label="Best reward",
    )
    ax_reward.set_xlabel("Iteration")
    ax_reward.set_ylabel("Best reward", color="tab:blue")
    ax_reward.tick_params(axis="y", labelcolor="tab:blue")

    # Overlay unique-material count on a twin axis.
    ax_count = ax_reward.twinx()
    ax_count.plot(
        iterations, mcts.unique_materials_history,
        color="tab:orange", label="Unique materials", alpha=0.7,
    )
    ax_count.set_ylabel("Unique materials", color="tab:orange")
    ax_count.tick_params(axis="y", labelcolor="tab:orange")

    ax_reward.set_title("MCTS convergence")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_property_distribution(
    mcts: MCTS,
    property_name: str,
    save_path: Optional[str] = None,
    bins: int = 30,
):
    """
    Histogram of `property_name` across all evaluated nodes that recorded it.

    Raises:
        ValueError: if no evaluated node has the requested property.
    """
    import matplotlib.pyplot as plt

    values = [
        node.properties[property_name]
        for node in mcts.all_nodes()
        if node.own_reward is not None and property_name in node.properties
    ]
    if not values:
        raise ValueError(f"No evaluated nodes recorded property {property_name!r}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="tab:green", edgecolor="black", alpha=0.75)
    ax.set_xlabel(property_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {property_name} "
                 f"({len(values)} evaluated materials)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_search_tree(
    mcts: MCTS,
    save_path: Optional[str] = None,
    max_nodes: int = 500,
):
    """
    Draw the explored search tree radially.

    Node color encodes own_reward (viridis; unevaluated nodes are grey) and
    node size encodes visit count. For readability the drawing is capped at
    `max_nodes` most-visited nodes; if the tree is larger, the cap is logged
    via a note in the title (silent truncation is avoided).

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    all_nodes = mcts.all_nodes()
    truncated = len(all_nodes) > max_nodes
    if truncated:
        # Keep the most-visited nodes, but always include the root.
        kept = sorted(all_nodes, key=lambda n: n.visits, reverse=True)[:max_nodes]
        kept_set = set(id(n) for n in kept)
        kept_set.add(id(mcts.root))
    else:
        kept_set = set(id(n) for n in all_nodes)

    # Build a directed graph over the kept nodes.
    graph = nx.DiGraph()
    node_ids = {}
    for i, node in enumerate(all_nodes):
        if id(node) not in kept_set:
            continue
        node_ids[id(node)] = i
        graph.add_node(i, own_reward=node.own_reward, visits=node.visits)
    for node in all_nodes:
        if id(node) not in kept_set or node.parent is None:
            continue
        if id(node.parent) in kept_set:
            graph.add_edge(node_ids[id(node.parent)], node_ids[id(node)])

    # Radial layout: prefer graphviz twopi, fall back to spring layout.
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="twopi")
    except Exception:
        pos = nx.spring_layout(graph, seed=0)

    rewards = [graph.nodes[n]["own_reward"] for n in graph.nodes]
    visits = np.array([graph.nodes[n]["visits"] for n in graph.nodes], dtype=float)

    # Map own_reward -> RGBA via viridis; unevaluated (None) nodes are grey.
    # We resolve colors to RGBA tuples ourselves (rather than passing a mix of
    # a grey string and numeric values to nx.draw, which matplotlib rejects).
    finite = [r for r in rewards if r is not None]
    vmin = min(finite) if finite else 0.0
    vmax = max(finite) if finite else 1.0
    span = (vmax - vmin) or 1.0
    cmap = plt.get_cmap("viridis")
    grey = (0.8, 0.8, 0.8, 1.0)
    node_color = [
        grey if r is None else cmap((r - vmin) / span)
        for r in rewards
    ]
    sizes = 20 + 80 * (visits / visits.max() if visits.max() > 0 else visits)

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw(
        graph, pos, ax=ax, node_size=sizes, node_color=node_color,
        edge_color="#999999", width=0.5, with_labels=False,
    )
    title = "MCTS search tree (color=own reward, size=visits)"
    if truncated:
        title += f"\n(showing {max_nodes} most-visited of {len(all_nodes)} nodes)"
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

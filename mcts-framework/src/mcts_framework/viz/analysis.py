"""
Post-run analysis: search-efficiency metrics and a text report.

Pure-Python (no matplotlib), operating on a finished MCTS object via its
public surface (all_nodes, get_best_materials, history lists, summary). Safe to
import without the [viz] extra.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Any, Dict, List

from ..core.mcts import MCTS


def compute_metrics(mcts: MCTS) -> Dict[str, Any]:
    """
    Compute search-efficiency metrics for a completed run.

    Returns a dict with:
        iterations         : iterations actually run
        unique_materials   : distinct materials attached to the tree
        tree_size          : total nodes in the tree
        evaluated_nodes    : nodes with an own_reward (actually simulated)
        max_depth          : deepest node depth (root = 0)
        best_reward        : best composite reward found
        best_material      : identifier of the best material
        efficiency         : evaluated_nodes / iterations (0 if no iterations)
    """
    nodes = mcts.all_nodes()
    evaluated = [n for n in nodes if n.own_reward is not None]
    max_depth = max((n.get_depth() for n in nodes), default=0)

    iterations = mcts.iteration + 1 if mcts.reward_history else 0
    efficiency = (len(evaluated) / iterations) if iterations else 0.0

    return {
        "iterations": iterations,
        "unique_materials": len(mcts.visited_materials),
        "tree_size": mcts.root.get_subtree_size(),
        "evaluated_nodes": len(evaluated),
        "max_depth": max_depth,
        "best_reward": mcts.best_reward,
        "best_material": (
            mcts.best_node.material.get_identifier()
            if mcts.best_node is not None
            else None
        ),
        "efficiency": efficiency,
    }


def rank_materials(mcts: MCTS, n: int = 20) -> List[Dict[str, Any]]:
    """
    Return the top-n evaluated materials as plain dicts (identifier,
    own_reward, visits, and any recorded properties), ranked by own_reward.
    """
    rows: List[Dict[str, Any]] = []
    for node in mcts.get_best_materials(n=n):
        row: Dict[str, Any] = {
            "identifier": node.material.get_identifier(),
            "own_reward": node.own_reward,
            "visits": node.visits,
        }
        row.update(node.properties)
        rows.append(row)
    return rows


def generate_report(mcts: MCTS, top_n: int = 10) -> str:
    """
    Build a human-readable text report of the completed search.

    Includes the efficiency metrics and a ranked list of the top materials
    with their properties. Returned as a string so callers decide whether to
    print it or write it to disk.
    """
    metrics = compute_metrics(mcts)
    lines: List[str] = []

    lines.append("=" * 60)
    lines.append("MCTS Materials Search Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Search metrics")
    lines.append("-" * 60)
    lines.append(f"  Iterations run    : {metrics['iterations']}")
    lines.append(f"  Unique materials  : {metrics['unique_materials']}")
    lines.append(f"  Tree size (nodes) : {metrics['tree_size']}")
    lines.append(f"  Evaluated nodes   : {metrics['evaluated_nodes']}")
    lines.append(f"  Max tree depth    : {metrics['max_depth']}")
    lines.append(f"  Efficiency        : {metrics['efficiency']:.3f} "
                 f"(evaluated / iteration)")
    lines.append("")
    lines.append(f"  Best material     : {metrics['best_material']}")
    lines.append(f"  Best reward       : {metrics['best_reward']:.4f}")
    lines.append("")

    lines.append(f"Top {top_n} materials (by own reward)")
    lines.append("-" * 60)
    ranked = rank_materials(mcts, n=top_n)
    if not ranked:
        lines.append("  (no materials evaluated)")
    else:
        for i, row in enumerate(ranked, 1):
            reward = row["own_reward"]
            reward_str = f"{reward:.4f}" if reward is not None else "n/a"
            lines.append(
                f"  {i:>2}. {row['identifier']:<28} "
                f"reward={reward_str}  visits={row['visits']}"
            )
            # Property detail line (skip the fields already shown).
            props = {
                k: v for k, v in row.items()
                if k not in ("identifier", "own_reward", "visits")
            }
            if props:
                prop_str = ", ".join(
                    f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in props.items()
                )
                lines.append(f"      {prop_str}")

    lines.append("=" * 60)
    return "\n".join(lines)
"""
Persist MCTS search results to an output directory.

Writes four files:
    summary.json          - run summary (best material, counts, tree size)
    best_materials.csv    - top-N materials ranked by own_reward, with props
    convergence.csv       - per-iteration best_reward and unique-material count
    report.txt            - human-readable analysis report (viz.analysis)

All tabular output is plain JSON/CSV so it is trivial to load for downstream
analysis or plotting.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..core.config import Config
from ..core.mcts import MCTS

logger = logging.getLogger(__name__)


def save_results(
    mcts: MCTS,
    output_dir: str,
    top_n: int = 20,
    config: Optional[Config] = None,
) -> Dict[str, str]:
    """
    Write summary, best-materials, and convergence files to output_dir.

    Args:
        mcts: A completed MCTS run.
        output_dir: Directory to create/write into.
        top_n: How many top materials to record in best_materials.csv.
        config: The run's Config. When given, it is persisted (secrets redacted)
            as config.yaml so post-run analysis can read the run's own gamma,
            beta, and data paths instead of re-specifying them.

    Returns:
        Mapping of logical name -> written file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": str(out / "summary.json"),
        "best_materials": str(out / "best_materials.csv"),
        "convergence": str(out / "convergence.csv"),
        "report": str(out / "report.txt"),
    }

    _write_summary(mcts, paths["summary"])
    _write_best_materials(mcts, paths["best_materials"], top_n)
    _write_convergence(mcts, paths["convergence"])
    _write_report(mcts, paths["report"], top_n)

    if config is not None:
        paths["config"] = str(out / "config.yaml")
        config.dump_yaml(paths["config"])

    logger.info("Saved results to %s", out)
    return paths


def _write_summary(mcts: MCTS, path: str) -> None:
    summary: Dict[str, Any] = mcts.summary()
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _write_best_materials(mcts: MCTS, path: str, top_n: int) -> None:
    """Top-N nodes by own_reward, one row each with flattened properties."""
    rows = []
    for node in mcts.get_best_materials(n=top_n):
        row: Dict[str, Any] = {
            "identifier": node.material.get_identifier(),
            "own_reward": node.own_reward,
            "visits": node.visits,
        }
        # Flatten evaluated properties (e_form, e_above_hull, melting_point, ...).
        for key, value in node.properties.items():
            row[key] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_convergence(mcts: MCTS, path: str) -> None:
    """Per-iteration running-best reward and cumulative unique-material count."""
    df = pd.DataFrame({
        "iteration": range(len(mcts.reward_history)),
        "best_reward": mcts.reward_history,
        "unique_materials": mcts.unique_materials_history,
    })
    df.to_csv(path, index=False)


def _write_report(mcts: MCTS, path: str, top_n: int) -> None:
    """Human-readable text report via the pure-Python analysis module."""
    from ..viz.analysis import generate_report

    with open(path, "w") as f:
        f.write(generate_report(mcts, top_n=top_n))

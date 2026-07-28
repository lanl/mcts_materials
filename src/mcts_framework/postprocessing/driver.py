"""
One generic driver that regenerates a study's outputs from a finished run.

A "study" in the framework is just an MCTS run directory: `mcts-run run` writes
config.yaml (the run's own parameters) and tree.json (the explored tree) into it,
alongside the summary/CSV files. This driver reads those two artifacts back and
produces the publication outputs - the top-N LaTeX table, the E_hull-vs-rDOS
scatter, and the radial search tree - all keyed off the run's own config, so
there is no per-study script and no duplicated parameters. Point it at any run
directory and it reproduces that run's figures.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Sequence

import pandas as pd

from ..core.config import Config
from .design_space import full_formula_key, load_design_space
from .radial_tree import plot_radial_tree
from .scatter import plot_ehull_vs_rdos
from .tables import write_top_n_table


def load_run_dataframe(tree_path: str) -> pd.DataFrame:
    """
    Reconstruct a run's evaluated-compounds DataFrame from its tree.json.

    Collapses the tree to one row per unique material that was actually
    evaluated (own_reward set), taking that node's own_reward and properties
    (e_above_hull, e_form, formula, ...). This is the run-results table the
    table/scatter consume; it comes from the persisted tree so no separate CSV
    is required.

    The 'name' column is the plain chemical formula (from the node's 'formula'
    property), NOT the full identifier - the identifier carries the
    '|SG|Wyckoff' suffix, which the composition-based ranking keys cannot parse.
    Dedup is still on the full identifier, so distinct site decorations at equal
    composition are kept as separate rows. The identifier is preserved in its
    own column.
    """
    with open(tree_path) as f:
        tree = json.load(f)

    rows: Dict[str, dict] = {}
    for rec in tree.get("nodes", []):
        if rec.get("own_reward") is None:
            continue
        identifier = str(rec["identifier"])
        if identifier in rows:
            continue  # first (root-ward) evaluation wins; identical material
        props = rec.get("properties") or {}
        # Prefer the plain 'formula' property as the display/ranking name;
        # fall back to the identifier if a material type doesn't record one.
        name = str(props.get("formula", identifier))
        row = {
            "name": name,
            "identifier": identifier,
            "own_reward": rec["own_reward"],
            "visits": rec.get("visits", 0),
        }
        row.update(props)
        rows[identifier] = row

    return pd.DataFrame(list(rows.values()))


def load_run_config(run_dir: str) -> Config:
    """Load the config.yaml a run persisted into its output directory."""
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.yaml in {run_dir!r}; re-run with save_results(config=...) "
            f"so the run persists its own parameters for analysis."
        )
    return Config.from_yaml(str(cfg_path))


def generate_study_outputs(
    run_dir: str,
    out_dir: Optional[str] = None,
    top_n: int = 15,
    key_fn: Callable[[str], Hashable] = full_formula_key,
    space_filter: Optional[Callable[[str], bool]] = None,
    synthesized: Optional[Sequence[str]] = None,
    attempted_path: Optional[str] = None,
    study_label: str = "",
) -> Dict[str, str]:
    """
    Regenerate a study's table + figures from a finished run directory.

    Reads run_dir/config.yaml and run_dir/tree.json and writes, into out_dir
    (default run_dir/figures):
        top{N}_table.tex     - top-N LaTeX candidate table
        ehull_vs_rdos.png    - E_hull vs r_DOS scatter (design space + top-N)
        radial_tree.png      - 4-panel radial search tree

    Only intermetallic runs produce the design-space table/scatter/tree (they
    need the MACE cache + DOSCAR data referenced by the config). key_fn,
    space_filter, synthesized, and attempted_path are forwarded to the table and
    scatter for study-specific ranking/overlays.

    Returns a mapping of logical name -> written path (only for outputs that
    were actually produced).
    """
    config = load_run_config(run_dir)
    if config.material_type != "intermetallic":
        raise ValueError(
            f"generate_study_outputs currently supports intermetallic runs; "
            f"got material_type={config.material_type!r}"
        )

    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run / "figures"
    out.mkdir(parents=True, exist_ok=True)

    tree_path = run / "tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(
            f"No tree.json in {run_dir!r}; re-run with save_tree=True so the "
            f"explored tree is persisted for analysis."
        )
    df = load_run_dataframe(str(tree_path))

    # The search only records e_form/e_above_hull/formula on each node, not the
    # rDOS - so recompute r_DOS per compound from the run's DOSCAR data (the same
    # lookup rank_design_space uses). Without this the table/scatter would treat
    # r_DOS as 0, silently zeroing any rDOS-dependent reward (e.g. the product).
    ic = config.intermetallic
    if ic is not None and ic.doscar_data_path and "name" in df.columns and len(df):
        _, doscar_lookup = load_design_space(ic.cache_path or "", ic.doscar_data_path)
        df["r_DOS"] = df["name"].apply(doscar_lookup.get_reward)

    produced: Dict[str, str] = {}

    table_path = out / f"top{top_n}_table.tex"
    write_top_n_table(
        df, str(table_path), config, n=top_n, key_fn=key_fn,
        space_filter=space_filter, synthesized=synthesized,
        attempted_path=attempted_path, study_label=study_label,
    )
    produced["table"] = str(table_path)

    scatter_path = out / "ehull_vs_rdos.png"
    fig = plot_ehull_vs_rdos(
        df, str(scatter_path), config, top_n=min(top_n, 10),
        key_fn=key_fn, space_filter=space_filter,
        synthesized=synthesized, attempted_path=attempted_path,
    )
    if fig is not None:
        produced["scatter"] = str(scatter_path)

    radial_path = out / "radial_tree.png"
    fig = plot_radial_tree(str(tree_path), str(radial_path), config, key_fn=key_fn)
    if fig is not None:
        produced["radial_tree"] = str(radial_path)

    return produced

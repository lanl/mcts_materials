#!/usr/bin/env python3
"""
Generate product-mode figures and tables for lanthanide+U study (single seed).

Produces:
1. ehull_vs_rdos_product_with_experimental.png - Scatter with experimental literature
2. top15_lanthanide_u_product.tex - LaTeX table of top 15 compounds
3. top15_recommendations.txt - Plain text table of top 15 compounds

Usage:
    python generate_figures.py
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from mcts_framework.postprocessing import (
    load_run_config,
    plot_ehull_vs_rdos_product_lanthanide_u,
    write_product_mode_table,
    write_product_mode_txt_table,
)


def load_seed_results(study_dir: Path, seed: int = 0) -> pd.DataFrame:
    """Load compounds from a single seed."""
    csv = study_dir / "results" / f"seed_{seed}" / "best_materials.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Results not found at {csv}")

    df = pd.read_csv(csv)
    df["seed"] = seed

    print(f"Loaded {len(df)} compounds from seed {seed}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate lanthanide+U study figures"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top compounds to overlay (default: 15)",
    )
    args = parser.parse_args()

    # Paths
    study_dir = Path(__file__).parent
    repo_root = study_dir.parent.parent
    results_dir = study_dir / "results"
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Optional: experimental literature compounds overlay
    experimental_path = (
        repo_root
        / "mcts_materials"
        / "analysis"
        / "ehull_rdos_lanthanide_u_study_max_undiscounted"
        / "experimental_citation_compounds"
    )

    print("=" * 60)
    print("Lanthanide+U Study Figure Generation")
    print("=" * 60)
    print(f"Study dir: {study_dir}")
    print(f"Figures will be saved to: {figures_dir}")
    print()

    # Check seed results exist
    seed_dir = results_dir / "seed_0"
    if not seed_dir.exists():
        print(f"ERROR: Seed 0 results not found at {seed_dir}")
        print(f"Run the study first: bash run_all_seeds.sh")
        sys.exit(1)

    # Load config from seed_0
    config_dir = results_dir / "seed_0"
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    config = load_run_config(str(config_dir))
    print(f"Loaded config from: {config_path}")
    print()

    # Load results from seed 0
    print("Loading results from seed 0...")
    try:
        results_df = load_seed_results(study_dir, seed=0)
        reward_col = "own_reward" if "own_reward" in results_df.columns else "reward"
        print(f"Top 5 compounds by product reward:")
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            print(f"  {i}. {row['formula']}: reward={row.get(reward_col, 'N/A')}")
        print()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Figure: Scatter plot with experimental
    print("Generating ehull_vs_rdos_product_with_experimental.png...")
    scatter_path = figures_dir / "ehull_vs_rdos_product_with_experimental.png"

    # Resolve config paths relative to repo root (config stores relative paths)
    if config.intermetallic:
        if config.intermetallic.cache_path and not Path(config.intermetallic.cache_path).is_absolute():
            config.intermetallic.cache_path = str(repo_root / config.intermetallic.cache_path)
        if config.intermetallic.doscar_data_path and not Path(config.intermetallic.doscar_data_path).is_absolute():
            config.intermetallic.doscar_data_path = str(repo_root / config.intermetallic.doscar_data_path)

    try:
        plot_ehull_vs_rdos_product_lanthanide_u(
            out_path=str(scatter_path),
            config=config,
            experimental_path=str(experimental_path) if experimental_path.exists() else None,
            top_n=args.top_n,
            run_df=results_df,
        )
    except Exception as e:
        print(f"ERROR generating scatter plot: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()

    # Helper function for lanthanide+U filter
    def _lanthanide_u_filter(name: str) -> bool:
        """True if compound contains U or any lanthanide."""
        lanthanides = {'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
        elems = set(re.findall(r'[A-Z][a-z]?', str(name)))
        return 'U' in elems or bool(elems & lanthanides)

    # Generate top-15 tables
    print("Generating top-15 tables...")
    tables_dir = figures_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # No synthesized compounds list for lanthanide+U (experimental data used instead)
    synthesized = []

    # LaTeX table
    tex_path = tables_dir / "top15_lanthanide_u_product.tex"
    try:
        write_product_mode_table(
            df=results_df,
            out_path=str(tex_path),
            config=config,
            n=15,
            synthesized=synthesized,
            attempted_path=None,
            study_label="lanthanide+U",
            space_filter=_lanthanide_u_filter,
        )
        print(f"  Saved: {tex_path.name}")
    except Exception as e:
        print(f"  ERROR generating LaTeX table: {e}")
        import traceback
        traceback.print_exc()

    # Plain text table
    txt_path = study_dir / "top15_recommendations.txt"
    try:
        write_product_mode_txt_table(
            df=results_df,
            out_path=str(txt_path),
            config=config,
            n=15,
            synthesized=synthesized,
            attempted_path=None,
            study_label="lanthanide+U",
            space_filter=_lanthanide_u_filter,
        )
        print(f"  Saved: {txt_path.name}")
    except Exception as e:
        print(f"  ERROR generating text table: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("Figure and table generation complete!")
    print("=" * 60)
    print(f"Figures saved to: {figures_dir}")
    print(f"  - {scatter_path.name}")
    print(f"Tables saved to:")
    if tex_path.exists():
        print(f"  - {tex_path.relative_to(study_dir)}")
    if txt_path.exists():
        print(f"  - {txt_path.relative_to(study_dir)}")
    print()


if __name__ == "__main__":
    main()

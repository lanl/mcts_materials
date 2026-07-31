#!/usr/bin/env python3
"""
Generate product-mode figures and tables for U-only study (single seed).

Produces:
1. ehull_vs_rdos_product.png - Scatter plot with synthesized compounds
2. radial_tree_composite_product.png - 3-panel radial tree from seed_0
3. top15_u_only_product.tex - LaTeX table of top 15 compounds
4. top15_recommendations.txt - Plain text table of top 15 compounds

Usage:
    python generate_figures.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mcts_framework.postprocessing import (
    load_run_config,
    plot_ehull_vs_rdos,
    plot_radial_tree,
    write_top_n_table,
)

# Study-specific helpers live alongside the drivers (not in the library).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _study_figures import (  # noqa: E402
    SYNTHESIZED_COMPOUNDS,
    u_only_filter,
    write_txt_table,
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
    parser = argparse.ArgumentParser(description="Generate U-only study figures")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to use for radial tree (default: 0)",
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

    # Optional: attempted syntheses overlay
    attempted_path = (
        repo_root
        / "mcts_materials"
        / "analysis"
        / "ehull_rdos_u_only_study_max_undiscounted"
        / "compounds_filtered.dat"
    )

    print("=" * 60)
    print("U-Only Study Figure Generation")
    print("=" * 60)
    print(f"Study dir: {study_dir}")
    print(f"Figures will be saved to: {figures_dir}")
    print()

    # Check seed results exist
    seed_dir = results_dir / f"seed_{args.seed}"
    if not seed_dir.exists():
        print(f"ERROR: Seed {args.seed} results not found at {seed_dir}")
        print(f"Run the study first: bash run_all_seeds.sh")
        sys.exit(1)

    # Load config from seed_0 (all seeds use same params except seed value)
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

    # Figure 1: Scatter plot
    print("Generating ehull_vs_rdos_product.png...")
    scatter_path = figures_dir / "ehull_vs_rdos_product.png"
    try:
        plot_ehull_vs_rdos(
            run_df=results_df,
            out_path=str(scatter_path),
            config=config,
            top_n=args.top_n,
            space_filter=u_only_filter,
            synthesized=SYNTHESIZED_COMPOUNDS,
            attempted_path=str(attempted_path) if attempted_path.exists() else None,
            ymax=1.5,
        )
    except Exception as e:
        print(f"ERROR generating scatter plot: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()

    # Figure 2: Radial tree (from specified seed)
    print(f"Generating radial_tree_composite_product.png (from seed {args.seed})...")
    tree_path = results_dir / f"seed_{args.seed}" / "tree.json"
    radial_path = figures_dir / "radial_tree_composite_product.png"

    if not tree_path.exists():
        print(f"WARNING: tree.json not found at {tree_path}")
        print("Skipping radial tree generation.")
    else:
        try:
            plot_radial_tree(
                tree_path=str(tree_path),
                out_path=str(radial_path),
                config=config,
                max_nodes=60,
                top_n=args.top_n,
            )
        except Exception as e:
            print(f"ERROR generating radial tree: {e}")
            import traceback

            traceback.print_exc()

    print()

    # Generate top-15 tables
    print("Generating top-15 tables...")
    tables_dir = figures_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # LaTeX table (generic library, LaTeX subscript names, U-only design space).
    tex_path = tables_dir / "top15_u_only_product.tex"
    try:
        write_top_n_table(
            df=results_df,
            out_path=str(tex_path),
            config=config,
            n=15,
            synthesized=SYNTHESIZED_COMPOUNDS,
            attempted_path=str(attempted_path) if attempted_path.exists() else None,
            study_label="U-only",
            space_filter=u_only_filter,
            latex_names=True,
        )
        print(f"  Saved: {tex_path.name}")
    except Exception as e:
        print(f"  ERROR generating LaTeX table: {e}")
        import traceback
        traceback.print_exc()

    # Plain text table (study-local helper).
    txt_path = study_dir / "top15_recommendations.txt"
    try:
        write_txt_table(
            df=results_df,
            out_path=str(txt_path),
            config=config,
            n=15,
            study_label="U-only",
            space_filter=u_only_filter,
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
    if radial_path.exists():
        print(f"  - {radial_path.name}")
    print(f"Tables saved to:")
    if tex_path.exists():
        print(f"  - {tex_path.relative_to(study_dir)}")
    if txt_path.exists():
        print(f"  - {txt_path.relative_to(study_dir)}")
    print()


if __name__ == "__main__":
    main()

"""
Compute the "true" / reference ranking of the intermetallic design space.

Ranks the intermetallic design space by a reward function (default:
ehull_rdos_product = ehull_reward(E_hull) * r_DOS) and writes the sorted table
to CSV. This is the ground-truth ordering an MCTS run's results can be checked
against: how highly did the search's picks actually rank, and how much of the
top of the space did it recover?

The design space defaults to the lanthanide series (La-Lu) plus U (see
_F_BLOCK). The high-throughput MACE cache also contains compounds with other
actinides (Th, Pa, Np, Pu); these are OUTSIDE the default design space and are
excluded (pass --space all to rank the raw cache instead).

It reuses the framework's own scoring - load_design_space (the same MACE cache +
per-compound DOSCAR rDOS the search uses) and score_by_method (the exact reward
formulas the rollout methods use) - so the reference ranking cannot drift from
what MCTS optimizes.

Requires the [intermetallic] extra (ASE, spglib, ...) for the DOSCAR lookup:
    pip install -e ".[intermetallic]"        # or: uv sync --extra intermetallic

Examples
--------
    # Rank the default design space (lanthanides + U) by the product reward:
    python examples/reference_ranking.py \
        --mace-cache examples/high_throughput_mace_results.full.csv \
        --doscar examples/doscar_peaks_data_with_U.csv

    # Restrict the design space to U-only compounds, keep the top 25:
    python examples/reference_ranking.py --space u_only --top 25

    # Rank the raw MACE cache including other actinides (Th/Pa/Np/Pu):
    python examples/reference_ranking.py --space all

    # Score by a different reward the search might have used:
    python examples/reference_ranking.py --method ehull_rdos --gamma 0.0001

© 2026. Triad National Security, LLC. All rights reserved.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from mcts_framework.postprocessing import (
    load_design_space,
    score_by_method,
)

# The f-block elements that define the design space: the full lanthanide
# series (La-Lu) plus U. A compound belongs to the design space iff its
# f-block element is one of these. Actinides beyond U (Th, Pa, Np, Pu, ...)
# present in the MACE cache are OUTSIDE this space and are excluded by default.
_F_BLOCK = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "U",
}

# All lanthanides + actinides, used to detect f-block elements that fall
# OUTSIDE _F_BLOCK (so such compounds can be filtered out of the design space).
_ALL_F_BLOCK = {
    # lanthanides
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    # actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
}


def _in_fblock(name: str) -> bool:
    """True if the formula's f-block element(s) all lie within _F_BLOCK.

    Compounds with an f-block element outside _F_BLOCK (e.g. the actinides
    Th/Pa/Np/Pu that appear in the MACE cache) are excluded from the design
    space. A compound with no f-block element at all is also excluded.
    """
    elems = set(re.findall(r"[A-Z][a-z]?", str(name)))
    f_present = elems & _ALL_F_BLOCK
    return bool(f_present) and f_present <= _F_BLOCK


def _u_only(name: str) -> bool:
    """True if the formula contains U and no other f-block element."""
    elems = set(re.findall(r"[A-Z][a-z]?", str(name)))
    return "U" in elems and not (elems & (_F_BLOCK - {"U"}))


def rank_reference(
    mace_cache: str,
    doscar_peaks: str,
    method: str = "ehull_rdos_product",
    beta: float = 1.0,
    gamma: float = 0.0001,
    space: str = "fblock",
) -> pd.DataFrame:
    """
    Return the full design space ranked by `method`, best first.

    Columns: rank, name, e_above_hull, r_DOS, reward. rDOS is resolved
    per-compound from the DOSCAR data via the same lookup the search uses;
    reward is score_by_method(method, ...) - identical to the run's reward.
    Ties keep first-seen order (stable sort).
    """
    df_mace, doscar_lookup = load_design_space(mace_cache, doscar_peaks)
    if df_mace is None or not len(df_mace):
        raise SystemExit(f"No usable MACE cache at {mace_cache!r}")

    df = df_mace.copy()
    df["name"] = df.get("name", df.get("formula"))
    if space == "fblock":
        df = df[df["name"].apply(_in_fblock)].copy()
    elif space == "u_only":
        df = df[df["name"].apply(_u_only)].copy()
    elif space != "all":
        raise SystemExit(f"Unknown --space {space!r} (use 'fblock', 'u_only', or 'all')")
    if df.empty:
        raise SystemExit(f"No compounds left after --space {space!r} filter")

    df["r_DOS"] = df["name"].apply(doscar_lookup.get_reward).astype(float)
    df["e_above_hull"] = df["e_above_hull"].astype(float)
    df["reward"] = df.apply(
        lambda r: score_by_method(method, r["e_above_hull"], r["r_DOS"], beta, gamma),
        axis=1,
    )

    df = df.sort_values("reward", ascending=False, kind="stable").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "name", "e_above_hull", "r_DOS", "reward"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Reference ranking of the intermetallic design space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mace-cache", default="examples/high_throughput_mace_results.full.csv",
                   help="High-throughput MACE results CSV (name, e_above_hull, ...).")
    p.add_argument("--doscar", default="examples/doscar_peaks_data_with_U.csv",
                   help="DOSCAR peaks CSV for the per-compound rDOS lookup.")
    p.add_argument("--method", default="ehull_rdos_product",
                   choices=["ehull", "rdos", "ehull_rdos", "ehull_rdos_product"],
                   help="Reward function to rank by (match your MCTS run).")
    p.add_argument("--beta", type=float, default=1.0,
                   help="E_hull weight (only used by ehull_rdos).")
    p.add_argument("--gamma", type=float, default=0.0001,
                   help="rDOS weight (only used by ehull_rdos).")
    p.add_argument("--space", default="fblock", choices=["fblock", "u_only", "all"],
                   help="Design space to rank: 'fblock' (lanthanides La-Lu + U, "
                   "the default; excludes cache compounds with other actinides), "
                   "'u_only' (U as the sole f-block element), or 'all' (every "
                   "compound in the MACE cache, including Th/Pa/Np/Pu).")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: reference_ranking_<method>.csv).")
    p.add_argument("--top", type=int, default=15,
                   help="How many rows to print to the console.")
    args = p.parse_args(argv)

    ranked = rank_reference(
        args.mace_cache, args.doscar, method=args.method,
        beta=args.beta, gamma=args.gamma, space=args.space,
    )

    out = args.out or f"reference_ranking_{args.method}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)

    print(f"Ranked {len(ranked)} compounds by '{args.method}' "
          f"(space={args.space}). Wrote {out}\n")
    with pd.option_context("display.max_rows", None, "display.width", 100):
        print(ranked.head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

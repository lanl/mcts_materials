#!/usr/bin/env python3
"""
Generate LaTeX table for experimental literature compounds.

Outputs a table similar to top15_lanthanide_u_product.tex but for the
experimental compounds from the citation file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import re
from mcts_framework.intermetallic import DoscarRewardLookup, ehull_reward

F_BLOCK = {'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu'}
LANTHANIDES = {'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}


def _formula_key_non_f(name: str):
    parts = re.findall(r'[A-Z][a-z]?', str(name))
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


def format_formula_latex(tm, giv, r):
    """Format experimental compound as LaTeX formula."""
    # Determine which element is the lanthanide
    lanthanide = None
    if r in LANTHANIDES:
        lanthanide = r

    if lanthanide:
        # Format as TM-GIV-Ln (e.g., Co-Ge-Gd)
        return f"{tm}--{giv}--{lanthanide}"
    else:
        # No lanthanide, just show TM-GIV-R
        return f"{tm}--{giv}--{r}"


def main():
    # Paths
    study_dir = Path(__file__).parent
    repo_root = study_dir.parent.parent
    tables_dir = study_dir / "figures" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    experimental_path = (
        repo_root
        / "mcts_materials"
        / "analysis"
        / "ehull_rdos_lanthanide_u_study_max_undiscounted"
        / "experimental_citation_compounds"
    )

    cache_path = repo_root / "high_throughput_mace_results.full.csv"
    doscar_path = repo_root / "doscar_peaks_data_with_U.csv"

    print("=" * 60)
    print("Experimental Compounds Table Generation")
    print("=" * 60)
    print(f"Study dir: {study_dir}")
    print(f"Experimental file: {experimental_path}")
    print()

    if not experimental_path.exists():
        print(f"ERROR: Experimental compounds file not found at {experimental_path}")
        sys.exit(1)

    # Load design space
    df_all = pd.read_csv(cache_path)
    df_all["name"] = df_all.get("name", df_all.get("formula"))

    doscar_lookup = DoscarRewardLookup(peaks_file=str(doscar_path))

    # Filter to valid compounds for backdrop (fallback)
    df = df_all.copy()
    if "data_quality" in df.columns:
        df = df[df["data_quality"] == "valid"].copy()

    df["r_dos"] = df["name"].apply(doscar_lookup.get_reward)
    df["e_hull"] = df["e_above_hull"].astype(float)

    # Build backdrop with MAX rdos for each non-f-block key (valid compounds only)
    backdrop = {}
    best_compound = {}
    for _, r in df.iterrows():
        key = _formula_key_non_f(r["name"])
        rdos = float(r["r_dos"])
        ehull = float(r["e_hull"])
        if key not in backdrop or rdos > backdrop[key][0]:
            backdrop[key] = (rdos, ehull)
            best_compound[key] = r["name"]

    # Parse experimental compounds - match plotting logic EXACTLY
    exp_data = []
    with open(experimental_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                tm, giv, r = parts
                key = tuple(sorted([p for p in [r, tm, giv] if p not in F_BLOCK]))

                # EXACT SAME LOGIC AS PLOTTING: Look up SPECIFIC lanthanide compound
                # Try common stoichiometries: TM6GIV6R format (most common in design space)
                test_formulas = [
                    f"{tm}6{giv}6{r}",    # e.g., Fe6Ge6Lu
                    f"{r}{tm}6{giv}6",    # e.g., LuFe6Ge6
                    f"{tm}{giv}{r}",      # e.g., FeGeLu
                    f"{r}{giv}{tm}",      # e.g., LuGeFe
                ]

                rdos = None
                ehull = None
                matched_formula = None

                for test_formula in test_formulas:
                    test_rdos = doscar_lookup.get_reward(test_formula)
                    if test_rdos > 0:
                        # Found a match - get its ehull from design space (valid compounds)
                        match_rows = df[df["name"] == test_formula]
                        if len(match_rows) > 0:
                            rdos = test_rdos
                            ehull = float(match_rows.iloc[0]["e_hull"])
                            matched_formula = test_formula
                            break

                # Fallback to backdrop if no specific match
                if rdos is None or ehull is None:
                    hit = backdrop.get(key)
                    if hit is None:
                        continue
                    if rdos is None:
                        rdos = hit[0]
                    if ehull is None:
                        ehull = hit[1]
                    if matched_formula is None:
                        matched_formula = best_compound.get(key, "Unknown")

                # Skip if still no valid values
                if rdos is None or ehull is None:
                    continue

                r_ehull = ehull_reward(ehull)
                product = r_ehull * rdos
                exp_data.append({
                    'tm': tm,
                    'giv': giv,
                    'r': r,
                    'key': key,
                    'best_formula': matched_formula,
                    'e_hull': ehull,
                    'r_ehull': r_ehull,
                    'r_dos': rdos,
                    'product': product,
                })

    # Sort by product reward (descending)
    exp_data.sort(key=lambda x: x['product'], reverse=True)

    print(f"Matched {len(exp_data)} experimental compounds to design space")
    print(f"Generating LaTeX table...")

    # Generate LaTeX table
    out_path = tables_dir / "experimental_compounds.tex"
    with open(out_path, 'w') as f:
        f.write("% Experimental literature compounds matched to design space.\n")
        f.write("% Sorted by product reward r_Ehull x r_DOS.\n")
        f.write("% Values taken from best-matching compound in design space (max r_DOS).\n")
        f.write("\\begin{tabular}{rlrrrr}\n")
        f.write("\\toprule\n")
        f.write("Rank & Compound & $E_{\\mathrm{Hull}}$ & $r_{E_{\\mathrm{Hull}}}$ & $r_{\\mathrm{DOS}}$ & Product Reward \\\\\n")
        f.write("\\midrule\n")

        for i, entry in enumerate(exp_data, 1):
            formula = format_formula_latex(entry['tm'], entry['giv'], entry['r'])
            e_hull = entry['e_hull']
            r_ehull = entry['r_ehull']
            r_dos = entry['r_dos']
            product = entry['product']

            f.write(f"{i} & {formula} & {e_hull:.4f} & {r_ehull:.4f} & {r_dos:.1f} & {product:.2f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Saved: {out_path}")

    # Also generate a text version for easy viewing
    txt_path = study_dir / "experimental_compounds.txt"
    with open(txt_path, 'w') as f:
        f.write("Experimental Literature Compounds (matched to design space)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Compound':<12} {'Best Match':<20} {'E_Hull':>9} {'r_Ehull':>9} {'r_DOS':>9} {'Product':>10}\n")
        f.write("-" * 80 + "\n")

        for i, entry in enumerate(exp_data, 1):
            compound = f"{entry['tm']}-{entry['giv']}-{entry['r']}"
            best = entry['best_formula'][:20]
            f.write(f"{i:<6} {compound:<12} {best:<20} {entry['e_hull']:>9.4f} {entry['r_ehull']:>9.4f} "
                   f"{entry['r_dos']:>9.1f} {entry['product']:>10.2f}\n")

    print(f"Saved: {txt_path}")
    print()
    print("=" * 60)
    print("Table generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
High-throughput MACE calculations for all combinations of:
- Full f-block elements (lanthanides Ce-Lu + actinides Th-Pu)
- Full transition metals (3d: Ti-Zn, 4d: Zr-Cd, 5d: Hf-Hg)
- Full group IVA (Si, Ge, Sn, Pb)

Uses stoichiometry M6X6F (where M=metal, X=group IVA, F=f-block)
based on the starting structure mat_Pb6U1W6_sg191.cif

NOTE: This script requires energy above hull calculations.
      You must provide your Materials Project API key below.
"""

# CONFIGURATION: Add your Materials Project API key here (REQUIRED for this script)
MP_API_KEY = None  # Get your key from: https://materialsproject.org/api

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from ase.io import read
from ase import Atoms
from tqdm import tqdm
import itertools

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import MaceEnergyCalculator


def get_element_symbol(atomic_number):
    """Convert atomic number to element symbol."""
    periodic_table = {
        14: 'Si', 32: 'Ge', 50: 'Sn', 82: 'Pb',
        22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu'
    }
    return periodic_table.get(atomic_number, f'Z{atomic_number}')


def create_substituted_structure(template_atoms, metal_z, group_iv_z, f_block_z):
    """
    Create a substituted structure from the template.

    Args:
        template_atoms: Template ASE Atoms object (Pb6U1W6)
        metal_z: Atomic number for transition metal substitution
        group_iv_z: Atomic number for group IV substitution
        f_block_z: Atomic number for f-block substitution

    Returns:
        New ASE Atoms object with substitutions
    """
    # Template structure is Pb6U1W6, where:
    # - Pb (82) is group IVA
    # - U (92) is f-block
    # - W (74) is transition metal

    new_atoms = template_atoms.copy()
    atomic_numbers = new_atoms.get_atomic_numbers()

    # Group IVA elements
    group_iv_list = [14, 32, 50, 82]  # Si, Ge, Sn, Pb

    # F-block elements
    f_block_list = list(range(58, 72)) + list(range(90, 95))  # Ce-Lu + Th-Pu

    # Transition metals (3d, 4d, 5d)
    transition_metals = (
        list(range(22, 31)) +   # Ti-Zn
        list(range(40, 49)) +   # Zr-Cd
        list(range(72, 81))     # Hf-Hg
    )

    # Perform substitutions
    new_atomic_numbers = []
    for z in atomic_numbers:
        if z in f_block_list:
            new_atomic_numbers.append(f_block_z)
        elif z in group_iv_list:
            new_atomic_numbers.append(group_iv_z)
        elif z in transition_metals:
            new_atomic_numbers.append(metal_z)
        else:
            new_atomic_numbers.append(z)

    new_atoms.set_atomic_numbers(new_atomic_numbers)
    return new_atoms


def get_formula_string(metal_z, group_iv_z, f_block_z):
    """
    Generate formula string in the format used by the code (e.g., 'Fe6Sn6U').
    Based on stoichiometry M6X6F1.
    """
    metal_sym = get_element_symbol(metal_z)
    group_iv_sym = get_element_symbol(group_iv_z)
    f_block_sym = get_element_symbol(f_block_z)

    # Format: Metal6GroupIV6FBlock
    # Sort to match the typical chemical formula ordering
    return f"{metal_sym}6{group_iv_sym}6{f_block_sym}"


def normalize_formula(formula):
    """
    Normalize formula for comparison with cached results.
    Sorts elements alphabetically.
    """
    import re

    # Find all element-count pairs
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    # Convert to dictionary
    element_counts = {}
    for element, count in matches:
        if not element:
            continue
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count

    # Sort elements alphabetically and reconstruct formula
    sorted_elements = sorted(element_counts.keys())
    normalized_parts = []
    for element in sorted_elements:
        count = element_counts[element]
        if count == 1:
            normalized_parts.append(element)
        else:
            normalized_parts.append(f"{element}{count}")

    return ''.join(normalized_parts)


def main():
    """
    Run high-throughput MACE calculations for all material combinations.
    """
    print("=" * 80)
    print("HIGH-THROUGHPUT MACE CALCULATIONS")
    print("All combinations: Full f-block × Full transition metals × Full group IVA")
    print("=" * 80)

    # Step 1: Load template structure
    print("\n1. Loading template crystal structure...")
    cif_file = Path(__file__).parent.parent / "examples" / "mat_Pb6U1W6_sg191.cif"

    if not cif_file.exists():
        print(f"ERROR: Template CIF file not found: {cif_file}")
        return 1

    template_atoms = read(str(cif_file))
    print(f"   ✓ Template structure: {template_atoms.get_chemical_formula()}")
    print(f"   ✓ Stoichiometry: M6X6F1")

    # Step 2: Define element sets
    print("\n2. Defining element sets...")

    # Group IVA: Si, Ge, Sn, Pb
    group_iv_elements = [14, 32, 50, 82]

    # Transition metals: 3d, 4d, 5d
    transition_metals = (
        list(range(22, 31)) +   # Ti(22) to Zn(30)
        list(range(40, 49)) +   # Zr(40) to Cd(48)
        list(range(72, 81))     # Hf(72) to Hg(80)
    )

    # F-block: Lanthanides (Ce-Lu) + Actinides (Th-Pu)
    f_block_elements = (
        list(range(58, 72)) +   # Ce(58) to Lu(71)
        list(range(90, 95))     # Th(90) to Pu(94)
    )

    print(f"   ✓ Group IVA elements: {len(group_iv_elements)} elements")
    print(f"     {', '.join([get_element_symbol(z) for z in group_iv_elements])}")
    print(f"   ✓ Transition metals: {len(transition_metals)} elements")
    print(f"     3d: {', '.join([get_element_symbol(z) for z in range(22, 31)])}")
    print(f"     4d: {', '.join([get_element_symbol(z) for z in range(40, 49)])}")
    print(f"     5d: {', '.join([get_element_symbol(z) for z in range(72, 81)])}")
    print(f"   ✓ F-block elements: {len(f_block_elements)} elements")
    print(f"     Lanthanides: {', '.join([get_element_symbol(z) for z in range(58, 72)])}")
    print(f"     Actinides: {', '.join([get_element_symbol(z) for z in range(90, 95)])}")

    # Step 3: Generate all combinations
    print("\n3. Generating all combinations...")
    all_combinations = list(itertools.product(transition_metals, group_iv_elements, f_block_elements))
    total_combinations = len(all_combinations)

    print(f"   ✓ Total combinations: {total_combinations:,}")
    print(f"     ({len(transition_metals)} metals × {len(group_iv_elements)} group IV × {len(f_block_elements)} f-block)")

    # Step 4: Define output file path (fixed filename, no timestamp)
    print("\n4. Setting up output file...")
    output_file = Path(__file__).parent.parent / "high_throughput_mace_results.csv"

    # Load existing calculations cache
    print("\n   Loading existing calculations...")
    cache_sources = []

    # Load from examples cache
    examples_cache_file = Path(__file__).parent.parent / "examples" / "mace_calculations.csv"
    if examples_cache_file.exists():
        examples_df = pd.read_csv(examples_cache_file)
        cache_sources.append(('examples cache', examples_df))
        print(f"   ✓ Loaded {len(examples_df)} entries from {examples_cache_file.name}")

    # Load from existing output file
    if output_file.exists():
        output_df = pd.read_csv(output_file)
        cache_sources.append(('existing results', output_df))
        print(f"   ✓ Loaded {len(output_df)} entries from {output_file.name}")
    else:
        print(f"   ✓ No existing results file found (will create new)")

    # Create normalized lookup for faster matching
    cache_lookup = {}
    for source_name, df in cache_sources:
        for _, row in df.iterrows():
            normalized = normalize_formula(row['name'])
            # Only add if not already present (first source wins)
            if normalized not in cache_lookup:
                cache_lookup[normalized] = {
                    'name': row['name'],
                    'e_form': row['e_form'],
                    'e_above_hull': row['e_above_hull'],
                    'e_decomp': row['e_decomp'],
                    'source': source_name
                }

    print(f"   ✓ Created normalized cache lookup with {len(cache_lookup)} unique entries")

    # Step 5: Set up energy calculator
    print("\n5. Setting up MACE energy calculator...")
    mp_api_key = MP_API_KEY

    # Initialize energy calculator with output file as cache
    energy_calc = MaceEnergyCalculator(
        csv_file=str(output_file),
        mp_api_key=mp_api_key
    )

    print(f"   ✓ MACE calculator initialized")
    print(f"   ✓ Materials Project API key configured")
    print(f"   ✓ Output file: {output_file.name}")

    # Step 6: Process all combinations
    print("\n6. Processing all combinations...")
    print(f"   This will take a significant amount of time for {total_combinations:,} compounds.")
    print(f"   Progress will be saved incrementally to: {output_file}")

    results = []
    cache_hits = 0
    new_calculations = 0
    errors = 0

    # Create progress bar
    pbar = tqdm(all_combinations, desc="Processing compounds", unit="compound")

    for metal_z, group_iv_z, f_block_z in pbar:
        # Generate formula
        formula = get_formula_string(metal_z, group_iv_z, f_block_z)
        normalized = normalize_formula(formula)

        # Update progress bar description
        pbar.set_description(f"Processing {formula}")

        try:
            # Check if already in cache or existing results
            if normalized in cache_lookup:
                cached = cache_lookup[normalized]
                results.append({
                    'name': formula,
                    'e_form': cached['e_form'],
                    'e_above_hull': cached['e_above_hull'],
                    'e_decomp': cached['e_decomp'],
                    'source': cached.get('source', 'cache')
                })
                cache_hits += 1
                pbar.set_postfix({
                    'cache': cache_hits,
                    'new': new_calculations,
                    'errors': errors
                })
            else:
                # Perform new MACE calculation
                new_atoms = create_substituted_structure(
                    template_atoms, metal_z, group_iv_z, f_block_z
                )

                # Calculate energies
                e_form, e_above_hull = energy_calc.calculate_energies(new_atoms)

                # Get e_decomp from the calculator's last calculation
                # (it's stored as an instance variable after calculate_energies)
                e_decomp = energy_calc.last_e_decomp

                results.append({
                    'name': formula,
                    'e_form': e_form,
                    'e_above_hull': e_above_hull,
                    'e_decomp': e_decomp,
                    'source': 'calculated'
                })
                new_calculations += 1

                pbar.set_postfix({
                    'cache': cache_hits,
                    'new': new_calculations,
                    'errors': errors
                })

                # Save intermediate results every 10 calculations
                if new_calculations % 10 == 0:
                    # Save only the new calculations (not cached ones)
                    new_results = [r for r in results if r['source'] == 'calculated']
                    if new_results:
                        temp_df = pd.DataFrame(new_results)
                        # Append to existing file or create new one
                        if output_file.exists():
                            temp_df[['name', 'e_form', 'e_above_hull', 'e_decomp']].to_csv(
                                output_file, mode='a', header=False, index=False
                            )
                        else:
                            temp_df[['name', 'e_form', 'e_above_hull', 'e_decomp']].to_csv(
                                output_file, index=False
                            )
                        # Clear new results from memory after saving
                        results = [r for r in results if r['source'] != 'calculated']

        except Exception as e:
            print(f"\n   ERROR processing {formula}: {e}")
            results.append({
                'name': formula,
                'e_form': 0.0,
                'e_above_hull': 0.0,
                'e_decomp': 0.0,
                'source': 'error'
            })
            errors += 1
            pbar.set_postfix({
                'cache': cache_hits,
                'new': new_calculations,
                'errors': errors
            })

    pbar.close()

    # Step 7: Save final results
    print("\n7. Saving final results...")
    results_df = pd.DataFrame(results)

    # Save only the new calculations (append mode)
    new_results = [r for r in results if r['source'] in ['calculated', 'error']]
    if new_results:
        new_results_df = pd.DataFrame(new_results)

        # Append to existing output file
        if output_file.exists():
            new_results_df[['name', 'e_form', 'e_above_hull', 'e_decomp']].to_csv(
                output_file, mode='a', header=False, index=False
            )
            print(f"   ✓ Appended {len(new_results)} new results to: {output_file}")
        else:
            new_results_df[['name', 'e_form', 'e_above_hull', 'e_decomp']].to_csv(
                output_file, index=False
            )
            print(f"   ✓ Created new results file: {output_file}")
    else:
        print(f"   ✓ No new results to save (all from cache)")

    # Save full results with source column for debugging
    results_df.to_csv(output_file.with_suffix('.full.csv'), index=False)
    print(f"   ✓ Full session results saved to: {output_file.with_suffix('.full.csv')}")

    # Step 8: Print summary statistics
    print("\n8. Summary Statistics")
    print("=" * 80)
    print(f"Total combinations processed: {len(results):,}")
    print(f"  - From cache: {cache_hits:,} ({100*cache_hits/len(results):.1f}%)")
    print(f"  - New calculations: {new_calculations:,} ({100*new_calculations/len(results):.1f}%)")
    print(f"  - Errors: {errors:,} ({100*errors/len(results):.1f}%)")

    # Filter successful results only
    successful_df = results_df[results_df['source'] != 'error']

    if len(successful_df) > 0:
        print(f"\nEnergy Statistics (successful calculations only):")
        print(f"  Formation Energy (e_form):")
        print(f"    Mean: {successful_df['e_form'].mean():.4f} eV/atom")
        print(f"    Min:  {successful_df['e_form'].min():.4f} eV/atom")
        print(f"    Max:  {successful_df['e_form'].max():.4f} eV/atom")
        print(f"    Std:  {successful_df['e_form'].std():.4f} eV/atom")

        print(f"\n  Energy Above Hull (e_above_hull):")
        print(f"    Mean: {successful_df['e_above_hull'].mean():.4f} eV/atom")
        print(f"    Min:  {successful_df['e_above_hull'].min():.4f} eV/atom")
        print(f"    Max:  {successful_df['e_above_hull'].max():.4f} eV/atom")
        print(f"    Std:  {successful_df['e_above_hull'].std():.4f} eV/atom")

        # Find most stable compounds
        print(f"\nTop 10 Most Stable Compounds (by e_above_hull):")
        top_stable = successful_df.nsmallest(10, 'e_above_hull')
        for i, (_, row) in enumerate(top_stable.iterrows(), 1):
            print(f"  {i:2d}. {row['name']:20s}  "
                  f"E_form={row['e_form']:8.4f}  "
                  f"E_hull={row['e_above_hull']:8.4f} eV/atom")

        # Count thermodynamically stable compounds
        stable_count = len(successful_df[successful_df['e_above_hull'] <= 0.0])
        print(f"\nThermodynamically stable compounds (e_above_hull ≤ 0): {stable_count}")

        metastable_count = len(successful_df[
            (successful_df['e_above_hull'] > 0.0) &
            (successful_df['e_above_hull'] <= 0.1)
        ])
        print(f"Metastable compounds (0 < e_above_hull ≤ 0.1): {metastable_count}")

    print("\n" + "=" * 80)
    print("HIGH-THROUGHPUT CALCULATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

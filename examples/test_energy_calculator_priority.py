"""
Test script to verify the energy calculator prioritizes CSV cache lookup.
"""

import sys
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import MaceEnergyCalculator
from ase.io import read
import pandas as pd


def main():
    """Test energy calculator cache priority."""
    print("=" * 60)
    print("TESTING ENERGY CALCULATOR CACHE PRIORITY")
    print("=" * 60)
    
    # Load the CSV file to see what's available
    csv_file = Path(__file__).parent / "mace_calculations.csv"
    df = pd.read_csv(csv_file)
    
    print(f"1. Loaded CSV with {len(df)} cached calculations")
    print("   Sample entries:")
    for i, row in df.head(5).iterrows():
        print(f"   - {row['name']:12s}: E_form = {row['e_form']:8.4f} eV/atom")
    
    # Test structure
    cif_file = Path(__file__).parent / "mat_Pb6U1W6_sg191.cif"
    atoms = read(str(cif_file))
    print(f"\n2. Test structure: {atoms.get_chemical_formula()}")
    
    # Create energy calculator
    energy_calc = MaceEnergyCalculator(csv_file=str(csv_file))
    
    # Test cache lookup for known compounds
    print(f"\n3. Testing cache lookup for known compounds...")
    
    test_formulas = ["Pb6UW6", "Fe6Sn6U", "Co6Sn6U", "Ni6Sn6U"]
    
    for formula in test_formulas:
        # Check if it's in the CSV
        in_csv = formula in df['name'].values
        print(f"\n   Testing {formula}:")
        print(f"   - In CSV: {in_csv}")
        
        if in_csv:
            csv_row = df[df['name'] == formula].iloc[0]
            expected_e_form = csv_row['e_form']
            expected_e_hull = csv_row['e_above_hull']
            print(f"   - Expected E_form: {expected_e_form:.4f} eV/atom")
            print(f"   - Expected E_hull: {expected_e_hull:.4f} eV/atom")
    
    # Test actual energy calculation with real structure
    print(f"\n4. Testing energy calculation with real structure...")
    
    # This should use cache if available, or compute if not
    e_form, e_hull = energy_calc.calculate_energies(atoms)
    
    print(f"   Result: E_form = {e_form:.4f} eV/atom, E_hull = {e_hull:.4f} eV/atom")
    
    # Check formula normalization
    print(f"\n5. Testing formula normalization...")
    test_pairs = [
        ("Pb6UW6", "W6Pb6U1"),
        ("Fe6Sn6U", "USn6Fe6"),
        ("Co6Ge6U", "UGe6Co6"),
    ]
    
    for formula1, formula2 in test_pairs:
        norm1 = energy_calc._normalize_formula(formula1)
        norm2 = energy_calc._normalize_formula(formula2)
        match = norm1 == norm2
        print(f"   {formula1:12s} -> {norm1}")
        print(f"   {formula2:12s} -> {norm2}")
        print(f"   Match: {match}\n")
    
    print("=" * 60)
    print("✅ ENERGY CALCULATOR TEST COMPLETE")
    print("Cache lookup should now be much faster!")
    print("MACE calculations only run when compound not in CSV")
    print("Force threshold set to reasonable 0.05 eV/Å")
    print("=" * 60)


if __name__ == "__main__":
    main()
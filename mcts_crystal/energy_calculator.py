"""
Energy calculator using MACE with CSV lookup for cached results.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from ase import Atoms
from ase.optimize import FIRE
from ase.filters import ExpCellFilter
from mace.calculators import mace_mp
from matbench_discovery.energy import get_e_form_per_atom
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram


class MaceEnergyCalculator:
    """
    Energy calculator that uses MACE-MP with CSV lookup for cached results.
    """
    
    def __init__(self, csv_file: Optional[str] = None, mp_api_key: Optional[str] = None):
        """
        Initialize the energy calculator.

        Args:
            csv_file: Path to CSV file with cached calculations
            mp_api_key: Materials Project API key for phase diagram calculations
        """
        self.csv_file = csv_file
        self.mp_api_key = mp_api_key
        self.cache_df = None
        self.calculator = None
        self.last_e_decomp = 0.0  # Store last calculated decomposition energy
        
        # Load cached data if available
        if csv_file and Path(csv_file).exists():
            self.cache_df = pd.read_csv(csv_file)
            print(f"Loaded {len(self.cache_df)} cached calculations from {csv_file}")
        else:
            self.cache_df = pd.DataFrame(columns=['name', 'e_form', 'e_above_hull', 'e_decomp'])
            
        # Initialize MACE calculator
        self._init_calculator()
        
    def _init_calculator(self):
        """Initialize MACE-MP calculator."""
        try:
            self.calculator = mace_mp(
                model="large", 
                dispersion=False, 
                default_dtype="float64", 
                device='cpu'
            )
        except Exception as e:
            print(f"Warning: Could not initialize MACE calculator: {e}")
            self.calculator = None
            
    def _get_cached_result(self, formula: str) -> Optional[Tuple[float, float]]:
        """
        Get cached results for a chemical formula with flexible matching.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Tuple of (e_form, e_above_hull) if found, None otherwise
        """
        if self.cache_df is None or self.cache_df.empty:
            return None
            
        # Try exact match first
        matches = self.cache_df[self.cache_df['name'] == formula]
        
        if not matches.empty:
            row = matches.iloc[0]
            return float(row['e_form']), float(row['e_above_hull'])
        
        # Try flexible matching - normalize both formulas for comparison
        normalized_input = self._normalize_formula(formula)
        
        for _, row in self.cache_df.iterrows():
            cached_formula = str(row['name'])
            if self._normalize_formula(cached_formula) == normalized_input:
                return float(row['e_form']), float(row['e_above_hull'])
            
        return None
        
    def _normalize_formula(self, formula: str) -> str:
        """
        Normalize chemical formula for matching (sort elements alphabetically).
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Normalized formula string
        """
        try:
            # Extract elements and counts using simple regex
            import re
            
            # Find all element-count pairs
            pattern = r'([A-Z][a-z]?)(\d*)'
            matches = re.findall(pattern, formula)
            
            # Convert to dictionary
            element_counts = {}
            for element, count in matches:
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
            
        except Exception:
            # Fallback to original formula if normalization fails
            return formula
        
    def _cache_result(self, formula: str, e_form: float, e_above_hull: float, e_decomp: float = 0.0):
        """
        Cache calculation result.
        
        Args:
            formula: Chemical formula
            e_form: Formation energy
            e_above_hull: Energy above hull
            e_decomp: Decomposition energy
        """
        new_row = pd.DataFrame([{
            'name': formula,
            'e_form': e_form,
            'e_above_hull': e_above_hull,
            'e_decomp': e_decomp
        }])
        
        self.cache_df = pd.concat([self.cache_df, new_row], ignore_index=True)
        
        # Save to file if specified
        if self.csv_file:
            self.cache_df.to_csv(self.csv_file, index=False)
            
    def _get_decomposition_energy(self, atoms: Atoms) -> float:
        """
        Calculate decomposition energy using Materials Project phase diagram.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Decomposition energy (formation energy of decomposition products)
        """
        if not self.mp_api_key:
            print(f"   No MP API key provided - using e_decomp = 0.0")
            return 0.0
            
        chemical_formula = atoms.get_chemical_formula()
        element_set = set(atoms.get_chemical_symbols())
        
        try:
            print(f"   Calculating decomposition energy for {chemical_formula} using MP API...")
            
            with MPRester(self.mp_api_key) as mpr:
                # Use the updated MP API method
                try:
                    entries = mpr.get_entries_in_chemsys(
                        elements=element_set,
                        additional_criteria={"thermo_types": ["GGA_GGA+U"]}
                    )
                except TypeError:
                    # Fallback for newer MP API versions
                    print(f"   Using fallback MP API call (no additional criteria)")
                    entries = mpr.get_entries_in_chemsys(elements=element_set)
                
            if not entries:
                print(f"   Warning: No MP entries found for elements {element_set}")
                return 0.0
                
            print(f"   Found {len(entries)} MP entries for elements {element_set}")
            
            # Create phase diagram
            pd_obj = PhaseDiagram(entries)
            comp = Composition(chemical_formula)
            decomp = pd_obj.get_decomposition(comp)
            
            print(f"   Decomposition products: {len(decomp)} phases")
            
            total_e_decomp = 0.0
            for entry, fraction in decomp.items():
                try:
                    structure = entry.structure
                    decomp_atoms = AseAtomsAdaptor.get_atoms(structure, msonable=False)
                    
                    # Use MACE to calculate energy of decomposition product
                    if self.calculator is None:
                        print(f"   Warning: No MACE calculator available, using entry energy")
                        e_form = entry.energy_per_atom
                    else:
                        decomp_atoms.calc = self.calculator
                        e_form = get_e_form_per_atom(dict(
                            energy=decomp_atoms.get_total_energy(),
                            composition=decomp_atoms.get_chemical_formula()
                        ))
                    
                    total_e_decomp += e_form * fraction
                    print(f"     {entry.composition}: e_form={e_form:.4f} eV/atom (fraction={fraction:.4f})")
                    
                except Exception as phase_error:
                    print(f"   Warning: Error processing decomposition phase {entry.composition}: {phase_error}")
                    # Use the MP energy as fallback
                    e_form = entry.energy_per_atom
                    total_e_decomp += e_form * fraction
                    print(f"     {entry.composition}: using MP energy={e_form:.4f} eV/atom (fraction={fraction:.4f})")
                
            print(f"   ✓ Calculated decomposition energy: {total_e_decomp:.4f} eV/atom")
            return total_e_decomp
            
        except Exception as e:
            print(f"   ❌ Error calculating decomposition energy for {chemical_formula}: {e}")
            print(f"      This may be due to: network issues, MP API limits, or missing MP data")
            print(f"      Returning 0.0 as fallback (this will make e_above_hull = e_form)")
            import traceback
            print(f"      Full error: {traceback.format_exc()}")
            return 0.0
            
    def calculate_energies(self, atoms: Atoms) -> Tuple[float, float]:
        """
        Calculate formation energy and energy above hull for given atoms.
        
        Priority order:
        1. Check CSV cache first (mace_calculations.csv)
        2. If not found, perform MACE calculation with reasonable force threshold
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Tuple of (formation_energy, energy_above_hull)
        """
        formula = atoms.get_chemical_formula(mode='metal')
        
        # ALWAYS check cache first
        cached_result = self._get_cached_result(formula)
        if cached_result is not None:
            print(f"   Using cached result for {formula}")
            # Also load e_decomp for cached results
            cached_row = self.cache_df[self.cache_df['name'] == formula]
            if not cached_row.empty and 'e_decomp' in cached_row.columns:
                self.last_e_decomp = float(cached_row.iloc[0]['e_decomp'])
            return cached_result
            
        # Only perform calculation if not in cache
        print(f"   Computing new MACE calculation for {formula} (not in cache)")
        
        if self.calculator is None:
            print(f"   Warning: No calculator available, returning zero energies for {formula}")
            return 0.0, 0.0
            
        try:
            # Set calculator and optimize structure
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.calculator
            
            # Structural optimization with reasonable force threshold (0.05 eV/Å)
            atoms_filtered = ExpCellFilter(atoms_copy)
            optimizer = FIRE(atoms_filtered)
            optimizer.run(fmax=0.05)  # Much more reasonable force threshold
            
            optimized_atoms = atoms_filtered.atoms
            
            # Calculate formation energy
            e_form = get_e_form_per_atom(dict(
                energy=optimized_atoms.get_total_energy(),
                composition=optimized_atoms.get_chemical_formula()
            ))
            
            # Calculate energy above hull
            e_decomp = self._get_decomposition_energy(optimized_atoms)
            e_above_hull = e_form - e_decomp

            # Store e_decomp for later access
            self.last_e_decomp = e_decomp

            # Cache the result for future use
            self._cache_result(formula, e_form, e_above_hull, e_decomp)
            print(f"   ✓ Cached new result: {formula} E_form={e_form:.4f} eV/atom")

            return e_form, e_above_hull
            
        except Exception as e:
            print(f"   Error calculating energies for {formula}: {e}")
            # Cache zero values to avoid repeated failures
            self._cache_result(formula, 0.0, 0.0, 0.0)
            return 0.0, 0.0
            
    def get_cached_dataframe(self) -> pd.DataFrame:
        """
        Get the cached results as a DataFrame.
        
        Returns:
            DataFrame with cached results
        """
        return self.cache_df.copy() if self.cache_df is not None else pd.DataFrame()
        
    def save_cache(self, filename: Optional[str] = None):
        """
        Save cache to CSV file.
        
        Args:
            filename: Output filename (uses self.csv_file if not provided)
        """
        output_file = filename or self.csv_file
        if output_file and self.cache_df is not None:
            self.cache_df.to_csv(output_file, index=False)
            print(f"Saved cache to {output_file}")
        else:
            print("No filename specified or no cache to save")
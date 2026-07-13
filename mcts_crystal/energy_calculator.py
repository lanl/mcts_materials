"""
Energy calculator using MACE with CSV lookup for cached results.
"""

import logging
import threading
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from ase import Atoms
from ase.optimize import FIRE
from ase.filters import ExpCellFilter

# mace-torch, pymatgen, and matbench-discovery are only needed for live MACE/Materials
# Project calls (install with `pip install -e .[full]`). They are imported lazily inside
# the methods that use them so the rest of this package stays importable/testable without
# them - cache-only lookups work fine either way.

logger = logging.getLogger(__name__)


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

        # A MACE calculator instance is stateful (results/atoms live on the
        # calculator itself), so it can't be shared across threads safely.
        # self.calculator is the eagerly-loaded instance used by the main
        # thread; other threads lazily load and cache their own instance here.
        self._thread_local = threading.local()
        # Guards reads/writes of cache_df and the CSV file it's persisted to,
        # so concurrent rollouts can't corrupt the shared cache.
        self._cache_lock = threading.Lock()

        # Load cached data if available
        if csv_file and Path(csv_file).exists():
            self.cache_df = pd.read_csv(csv_file)
            # Add data_quality column if it doesn't exist (for backward compatibility)
            if 'data_quality' not in self.cache_df.columns:
                # Infer data quality from E_hull vs E_form comparison
                # Missing MP data: E_hull = E_form (only elemental references used)
                # Valid MP data: E_hull != E_form (competing phases found)
                def infer_quality(row):
                    # Check if E_hull equals E_form (within numerical precision)
                    if abs(row['e_above_hull'] - row['e_form']) < 1e-9:
                        return 'no_mp_data'  # Missing MP phase diagram data
                    else:
                        return 'valid'  # Has competing phase data
                self.cache_df['data_quality'] = self.cache_df.apply(infer_quality, axis=1)

                # Also fix any E_decomp errors: E_decomp should equal E_form - E_hull
                self.cache_df['e_decomp'] = self.cache_df['e_form'] - self.cache_df['e_above_hull']
                logger.info(f"Added data_quality column and corrected E_decomp values")
            logger.info(f"Loaded {len(self.cache_df)} cached calculations from {csv_file}")
        else:
            self.cache_df = pd.DataFrame(columns=['name', 'e_form', 'e_above_hull', 'e_decomp', 'data_quality'])
            
        # Initialize MACE calculator (eagerly, on the main thread)
        self.calculator = self._init_calculator()
        
    def _init_calculator(self):
        """Build a new MACE-MP calculator instance, or None if loading fails."""
        try:
            from mace.calculators import mace_mp
            return mace_mp(
                model="large",
                dispersion=False,
                default_dtype="float64",
                device='cpu'
            )
        except Exception as e:
            logger.warning(f"Could not initialize MACE calculator: {e}")
            return None

    def _get_calculator(self):
        """
        Get the MACE calculator instance to use on the current thread.

        A MACE/ASE calculator stores its results on the instance itself, so
        sharing one instance across threads would let concurrent rollouts
        clobber each other's results. The main thread uses self.calculator
        (loaded eagerly in __init__, unchanged from prior behavior); any other
        thread (i.e. a rollout running in the thread pool) lazily loads and
        reuses its own instance for the lifetime of that worker thread.
        """
        if threading.current_thread() is threading.main_thread():
            return self.calculator

        if not hasattr(self._thread_local, 'calculator'):
            self._thread_local.calculator = self._init_calculator()
        return self._thread_local.calculator

    def _get_cached_result(self, formula: str) -> Optional[Tuple[float, float]]:
        """
        Get cached results for a chemical formula with flexible matching.

        Args:
            formula: Chemical formula string

        Returns:
            Tuple of (e_form, e_above_hull) if found, None otherwise
            Note: If data_quality is 'no_mp_data' or 'error', e_above_hull is set to 10.0
        """
        if self.cache_df is None or self.cache_df.empty:
            return None

        # Try exact match first
        matches = self.cache_df[self.cache_df['name'] == formula]

        if not matches.empty:
            row = matches.iloc[0]
            e_form = float(row['e_form'])
            e_above_hull = float(row['e_above_hull'])

            # Check data quality and apply penalty if needed
            if 'data_quality' in row and row['data_quality'] in ['no_mp_data', 'error']:
                logger.info(f"      Cached compound {formula} has data_quality={row['data_quality']}")
                logger.info(f"      Applying penalty: setting e_above_hull = 10.0 eV/atom")
                e_above_hull = 10.0

            return e_form, e_above_hull

        # Try flexible matching - normalize both formulas for comparison
        normalized_input = self._normalize_formula(formula)

        for _, row in self.cache_df.iterrows():
            cached_formula = str(row['name'])
            if self._normalize_formula(cached_formula) == normalized_input:
                e_form = float(row['e_form'])
                e_above_hull = float(row['e_above_hull'])

                # Check data quality and apply penalty if needed
                if 'data_quality' in row and row['data_quality'] in ['no_mp_data', 'error']:
                    logger.warning(f"   Cached compound {cached_formula} has data_quality={row['data_quality']}")
                    logger.info(f"      Applying penalty: setting e_above_hull = 10.0 eV/atom")
                    e_above_hull = 10.0

                return e_form, e_above_hull

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
        
    def _cache_result(self, formula: str, e_form: float, e_above_hull: float, e_decomp: float = 0.0, data_quality: str = 'valid'):
        """
        Cache calculation result.

        Args:
            formula: Chemical formula
            e_form: Formation energy
            e_above_hull: Energy above hull
            e_decomp: Decomposition energy
            data_quality: Data quality flag ('valid', 'no_mp_data', 'no_api_key', 'error')
        """
        new_row = pd.DataFrame([{
            'name': formula,
            'e_form': e_form,
            'e_above_hull': e_above_hull,
            'e_decomp': e_decomp,
            'data_quality': data_quality
        }])

        with self._cache_lock:
            self.cache_df = pd.concat([self.cache_df, new_row], ignore_index=True)

            # Save to file if specified
            if self.csv_file:
                self.cache_df.to_csv(self.csv_file, index=False)
            
    def _get_decomposition_energy(self, atoms: Atoms) -> Tuple[float, str]:
        """
        Calculate decomposition energy using Materials Project phase diagram.

        Args:
            atoms: ASE Atoms object

        Returns:
            Tuple of (decomposition energy, data_quality flag)
            data_quality can be: 'valid', 'no_mp_data', 'no_api_key', 'error'
        """
        if not self.mp_api_key:
            logger.info(f"   No MP API key provided - using e_decomp = 0.0")
            return 0.0, 'no_api_key'
            
        chemical_formula = atoms.get_chemical_formula()
        element_set = set(atoms.get_chemical_symbols())
        
        try:
            from pymatgen.core import Composition
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.ext.matproj import MPRester
            from pymatgen.analysis.phase_diagram import PhaseDiagram
            from matbench_discovery.energy import get_e_form_per_atom

            logger.info(f"   Calculating decomposition energy for {chemical_formula} using MP API...")

            with MPRester(self.mp_api_key) as mpr:
                # Use the updated MP API method
                try:
                    entries = mpr.get_entries_in_chemsys(
                        elements=element_set,
                        additional_criteria={"thermo_types": ["GGA_GGA+U"]}
                    )
                except TypeError:
                    # Fallback for newer MP API versions
                    logger.info(f"   Using fallback MP API call (no additional criteria)")
                    entries = mpr.get_entries_in_chemsys(elements=element_set)
                
            if not entries:
                logger.warning(f"     No MP entries found for elements {element_set}")
                logger.info(f"      Compound {chemical_formula} will be flagged as missing MP data")
                return 0.0, 'no_mp_data'

            logger.info(f"   Found {len(entries)} MP entries for elements {element_set}")

            # Create phase diagram
            pd_obj = PhaseDiagram(entries)
            comp = Composition(chemical_formula)
            decomp = pd_obj.get_decomposition(comp)

            logger.info(f"   Decomposition products: {len(decomp)} phases")

            total_e_decomp = 0.0
            for entry, fraction in decomp.items():
                try:
                    structure = entry.structure
                    decomp_atoms = AseAtomsAdaptor.get_atoms(structure, msonable=False)

                    # Use MACE to calculate energy of decomposition product
                    calculator = self._get_calculator()
                    if calculator is None:
                        logger.warning(f"   No MACE calculator available, using entry energy")
                        e_form = entry.energy_per_atom
                    else:
                        decomp_atoms.calc = calculator
                        e_form = get_e_form_per_atom(dict(
                            energy=decomp_atoms.get_total_energy(),
                            composition=decomp_atoms.get_chemical_formula()
                        ))

                    total_e_decomp += e_form * fraction
                    logger.info(f"     {entry.composition}: e_form={e_form:.4f} eV/atom (fraction={fraction:.4f})")

                except Exception as phase_error:
                    logger.error(f"   Error processing decomposition phase {entry.composition}: {phase_error}")
                    # Use the MP energy as fallback
                    e_form = entry.energy_per_atom
                    total_e_decomp += e_form * fraction
                    logger.info(f"     {entry.composition}: using MP energy={e_form:.4f} eV/atom (fraction={fraction:.4f})")

            logger.info(f"   Calculated decomposition energy: {total_e_decomp:.4f} eV/atom")
            return total_e_decomp, 'valid'

        except Exception as e:
            logger.error(f"      Error calculating decomposition energy for {chemical_formula}: {e}")
            logger.info(f"      This may be due to: network issues, MP API limits, or missing MP data")
            logger.info(f"      Compound will be flagged with low reward to avoid selection")
            import traceback
            logger.error(f"      Full error: {traceback.format_exc()}")
            return 0.0, 'error'
            
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
        with self._cache_lock:
            cached_result = self._get_cached_result(formula)
            if cached_result is not None:
                logger.info(f"   Using cached result for {formula}")
                # Also load e_decomp for cached results
                cached_row = self.cache_df[self.cache_df['name'] == formula]
                if not cached_row.empty and 'e_decomp' in cached_row.columns:
                    self.last_e_decomp = float(cached_row.iloc[0]['e_decomp'])
                return cached_result

        # Only perform calculation if not in cache
        logger.info(f"   Computing new MACE calculation for {formula} (not in cache)")

        calculator = self._get_calculator()
        if calculator is None:
            logger.warning(f"   No calculator available, returning zero energies for {formula}")
            return 0.0, 0.0

        try:
            from matbench_discovery.energy import get_e_form_per_atom

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
            e_decomp, data_quality = self._get_decomposition_energy(optimized_atoms)

            # Check data quality - if MP data is missing, penalize heavily
            if data_quality in ['no_mp_data', 'error']:
                # Set e_above_hull to a very large positive value (unstable)
                e_above_hull = 10.0  # 10 eV/atom above hull = extremely unstable
                logger.warning(f"      WARNING: Missing MP data for {formula}")
                logger.info(f"      Setting e_above_hull = {e_above_hull} eV/atom (flagged as unstable)")
                logger.info(f"      This compound will have very low reward and be avoided by MCTS")
            else:
                e_above_hull = e_form - e_decomp

            # Store e_decomp for later access
            self.last_e_decomp = e_decomp

            # Cache the result for future use with data quality flag
            self._cache_result(formula, e_form, e_above_hull, e_decomp, data_quality)
            logger.info(f"   Cached new result: {formula} E_form={e_form:.4f} eV/atom, data_quality={data_quality}")

            return e_form, e_above_hull
            
        except Exception as e:
            logger.error(f"   Error calculating energies for {formula}: {e}")
            # Cache with error flag to avoid repeated failures
            self._cache_result(formula, 0.0, 10.0, 0.0, 'error')
            return 0.0, 10.0
            
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
            logger.info(f"Saved cache to {output_file}")
        else:
            logger.info("No filename specified or no cache to save")

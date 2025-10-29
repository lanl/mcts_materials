"""
Results analysis tools for MCTS crystal structure optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .mcts import MCTS


class ResultsAnalyzer:
    """
    Analyze and report results from MCTS crystal structure search.
    """
    
    def __init__(self, csv_file: str = None):
        """Initialize results analyzer."""
        self.csv_file = csv_file
        self.formation_energy_lookup = {}
        if csv_file:
            self._load_formation_energies(csv_file)
    
    def _load_formation_energies(self, csv_file: str):
        """Load formation energies from CSV file for accurate lookup."""
        import os
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    formula = row['name']
                    e_form = float(row['e_form'])
                    self.formation_energy_lookup[formula] = e_form
                print(f"Loaded formation energies from: {csv_file}")
        except Exception as e:
            print(f"Warning: Could not load formation energies: {e}")
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize chemical formula for matching."""
        try:
            import re
            pattern = r'([A-Z][a-z]?)(\d*)'
            matches = re.findall(pattern, formula)
            element_counts = {}
            for element, count in matches:
                count = int(count) if count else 1
                element_counts[element] = element_counts.get(element, 0) + count
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
            return formula
    
    def _get_formation_energy(self, formula: str) -> float:
        """Get formation energy for a formula, with flexible matching."""
        # Try exact match first
        if formula in self.formation_energy_lookup:
            return self.formation_energy_lookup[formula]
        
        # Try normalized matching
        normalized_input = self._normalize_formula(formula)
        for cached_formula, e_form in self.formation_energy_lookup.items():
            if self._normalize_formula(cached_formula) == normalized_input:
                return e_form
        
        # Fallback: return None if not found
        return None
        
    def get_top_compounds(self, stat_dict: Dict, n_top: int = 10, 
                         metric: str = 'formation_energy') -> pd.DataFrame:
        """
        Get top N compounds based on specified metric.
        
        Args:
            stat_dict: Statistics dictionary from MCTS run
            n_top: Number of top compounds to return
            metric: Metric to rank by ('formation_energy' or 'energy_above_hull')
            
        Returns:
            DataFrame with top compounds
        """
        if not stat_dict:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(stat_dict).T
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df = df.reset_index()
        df.columns = ['formula'] + list(df.columns[1:])
        
        # The e_form is now directly available in the stat_dict
        df['formation_energy'] = df['e_form']
        
        # Sort by specified metric
        if metric == 'formation_energy':
            df_sorted = df.sort_values('formation_energy')
        elif metric == 'energy_above_hull':
            df_sorted = df.sort_values('e_above_hull')
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        return df_sorted.head(n_top)
        
    def analyze_search_efficiency(self, stat_dict: Dict) -> Dict:
        """
        Analyze the efficiency of the MCTS search.
        
        Args:
            stat_dict: Statistics dictionary from MCTS run
            
        Returns:
            Dictionary with efficiency metrics
        """
        if not stat_dict:
            return {}
            
        df = pd.DataFrame(stat_dict).T
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        
        # The e_form is now directly available in the stat_dict
        df['formation_energy'] = df['e_form']
        
        total_visits = df['visit_count'].sum()
        unique_compounds = len(df)
        
        # Find compounds with good energetics
        stable_compounds = df[df['e_above_hull'] < 0.1]  # Within 100 meV of hull
        very_stable = df[df['e_above_hull'] < 0.05]  # Within 50 meV of hull
        
        efficiency_metrics = {
            'total_compounds_explored': unique_compounds,
            'total_visits': int(total_visits),
            'average_visits_per_compound': total_visits / unique_compounds if unique_compounds > 0 else 0,
            'compounds_near_hull_100meV': len(stable_compounds),
            'compounds_near_hull_50meV': len(very_stable),
            'best_formation_energy': df['formation_energy'].min(),
            'best_energy_above_hull': df['e_above_hull'].min(),
            'search_diversity': unique_compounds / total_visits if total_visits > 0 else 0
        }
        
        return efficiency_metrics
        
    def get_chemical_trends(self, stat_dict: Dict) -> Dict:
        """
        Analyze chemical trends in the discovered compounds.
        
        Args:
            stat_dict: Statistics dictionary from MCTS run
            
        Returns:
            Dictionary with chemical trend analysis
        """
        if not stat_dict:
            return {}
            
        df = pd.DataFrame(stat_dict).T
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df = df.reset_index()
        df.columns = ['formula'] + list(df.columns[1:])
        
        # The e_form is now directly available in the stat_dict
        df['formation_energy'] = df['e_form']
        
        # Extract elements from formulas
        transition_metals = []
        group_iv_elements = []
        
        for formula in df['formula']:
            # Simple parsing - assumes format like "Fe6Sn6U"
            # This is a simplified approach, could be improved with proper formula parsing
            if 'Ti' in formula or 'V' in formula or 'Cr' in formula or 'Mn' in formula:
                if 'Ti' in formula:
                    transition_metals.append('Ti')
                elif 'V' in formula:
                    transition_metals.append('V')
                elif 'Cr' in formula:
                    transition_metals.append('Cr')
                elif 'Mn' in formula:
                    transition_metals.append('Mn')
            elif 'Fe' in formula or 'Co' in formula or 'Ni' in formula or 'Cu' in formula or 'Zn' in formula:
                if 'Fe' in formula:
                    transition_metals.append('Fe')
                elif 'Co' in formula:
                    transition_metals.append('Co')
                elif 'Ni' in formula:
                    transition_metals.append('Ni')
                elif 'Cu' in formula:
                    transition_metals.append('Cu')
                elif 'Zn' in formula:
                    transition_metals.append('Zn')
                    
            if 'Si' in formula:
                group_iv_elements.append('Si')
            elif 'Ge' in formula:
                group_iv_elements.append('Ge')
            elif 'Sn' in formula:
                group_iv_elements.append('Sn')
            elif 'Pb' in formula:
                group_iv_elements.append('Pb')
        
        trends = {
            'most_common_transition_metal': max(set(transition_metals), key=transition_metals.count) if transition_metals else 'None',
            'most_common_group_iv': max(set(group_iv_elements), key=group_iv_elements.count) if group_iv_elements else 'None',
            'transition_metal_distribution': dict(pd.Series(transition_metals).value_counts()) if transition_metals else {},
            'group_iv_distribution': dict(pd.Series(group_iv_elements).value_counts()) if group_iv_elements else {}
        }
        
        return trends
        
    def create_summary_report(self, mcts: MCTS, save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive summary report of the MCTS run.
        
        Args:
            mcts: MCTS instance
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        # Get analysis results
        top_compounds = self.get_top_compounds(mcts.stat_dict, n_top=10)
        efficiency = self.analyze_search_efficiency(mcts.stat_dict)
        trends = self.get_chemical_trends(mcts.stat_dict)
        
        report_lines = [
            "=" * 80,
            "MCTS CRYSTAL STRUCTURE OPTIMIZATION REPORT",
            "=" * 80,
            "",
            "SEARCH SUMMARY:",
            f"- Root compound: {mcts.root.get_chemical_formula()}",
            f"- Total iterations: {efficiency.get('total_visits', 0)}",
            f"- Unique compounds explored: {efficiency.get('total_compounds_explored', 0)}",
            f"- Search terminated: {mcts.terminated}",
            "",
            "BEST RESULTS:",
            f"- Best formation energy: {efficiency.get('best_formation_energy', 0):.4f} eV/atom",
            f"- Best energy above hull: {efficiency.get('best_energy_above_hull', 0):.4f} eV/atom",
            f"- Best compound: {mcts.best_node.get_chemical_formula() if mcts.best_node else 'None'}",
            "",
            "SEARCH EFFICIENCY:",
            f"- Compounds within 100 meV of hull: {efficiency.get('compounds_near_hull_100meV', 0)}",
            f"- Compounds within 50 meV of hull: {efficiency.get('compounds_near_hull_50meV', 0)}",
            f"- Average visits per compound: {efficiency.get('average_visits_per_compound', 0):.2f}",
            f"- Search diversity: {efficiency.get('search_diversity', 0):.4f}",
            "",
            "CHEMICAL TRENDS:",
            f"- Most common transition metal: {trends.get('most_common_transition_metal', 'None')}",
            f"- Most common Group IV element: {trends.get('most_common_group_iv', 'None')}",
            "",
            "TOP 10 COMPOUNDS BY FORMATION ENERGY:",
            "-" * 50,
        ]
        
        # Add top compounds table
        if not top_compounds.empty:
            for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
                report_lines.append(
                    f"{i:2d}. {row['formula']:15s} | "
                    f"E_form: {row['formation_energy']:8.4f} eV/atom | "
                    f"E_hull: {row['e_above_hull']:8.4f} eV/atom | "
                    f"Visits: {row['visit_count']:4.0f}"
                )
        else:
            report_lines.append("No compounds found.")
            
        report_lines.extend([
            "",
            "=" * 80,
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Summary report saved to {save_path}")
            
        return report
        
    def export_results(self, stat_dict: Dict, filename: str):
        """
        Export complete results to CSV file.
        
        Args:
            stat_dict: Statistics dictionary from MCTS run
            filename: Output filename
        """
        df = pd.DataFrame(stat_dict).T
        if not df.empty:
            df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
            df = df.reset_index()
            df.columns = ['formula'] + list(df.columns[1:])
            
            # The e_form is now directly available in the stat_dict
            df['formation_energy'] = df['e_form']
            
            # Sort by formation energy
            df = df.sort_values('formation_energy')
            
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        else:
            print("No results to export")
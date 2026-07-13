"""
Simple MCTS Crystal Structure Optimization Runner.

This is the implementation behind `python run_mcts.py [OPTIONS]` (a thin wrapper
at the repo root, kept for backward compatibility) and the `mcts-run` console
command installed by `pip install -e .`.

Usage:
    python run_mcts.py [OPTIONS]

Local secrets/config:
    Instead of passing --mp-api-key on the command line, copy config.example.json
    to config.json and fill in your Materials Project API key. config.json is
    gitignored and is never pushed to the repo; any value loaded from it is used
    as the default and can still be overridden by a CLI flag.

Examples:
    python run_mcts.py                                    # Default: 1000 iterations, ehull rollout
    python run_mcts.py --iterations 100                   # 100 iterations
    python run_mcts.py --structure my_structure.cif        # Custom structure
    python run_mcts.py --rollout-method ehull --mp-api-key YOUR_KEY       # E_hull only (MACE + Materials Project, no DFT/DOSCAR data needed)
    python run_mcts.py --rollout-method ehull_rdos --beta 1.0 --gamma 0.0001 --mp-api-key YOUR_KEY  # E_hull + rDOS (requires doscar_peaks_data_with_U.csv)
    python run_mcts.py --rollout-method rdos               # rDOS only (requires doscar_peaks_data_with_U.csv, no MACE/MP needed)
    python run_mcts.py --f-block-mode lanthanides_u_extended  # Lanthanides + U, extended moves
    python run_mcts.py --exploration-constant 0.2          # Higher exploration (default: 0.1)
    python run_mcts.py --selection-mode boltzmann --temperature 0.5  # Softmax exploration instead of classic UCB1
    python run_mcts.py --selection-mode epsilon_greedy --epsilon 0.2  # argmax UCB1, uniform-random epsilon fraction of the time
    python run_mcts.py --no-labels                         # Turn off labels on radial tree visualization
    python run_mcts.py --iterations 200 --structure my_structure.cif --rollout-method ehull
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from .node import MCTSTreeNode
from .mcts import MCTS
from .energy_calculator import MaceEnergyCalculator
from .visualization import TreeVisualizer
from .analysis import ResultsAnalyzer
from .doscar_utils import DoscarRewardLookup
from ase.io import read
from ase import Atoms
import pandas as pd

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.json') -> dict:
    """
    Load local run configuration (e.g. Materials Project API key) if present.

    config.json is gitignored - it never gets pushed to the repo. See
    config.example.json for the expected keys. Values loaded here become
    argparse defaults, so any CLI flag still overrides them.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            config = json.load(f)
        logger.info(f"Loaded local config from {path}")
        return config
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return {}


def override_composition(atoms: Atoms, transition_metal: str = None, group_iv: str = None,
                         f_block_element: str = None) -> Atoms:
    """
    Override the composition of a structure while maintaining the crystal structure.

    Args:
        atoms: ASE Atoms object with template structure
        transition_metal: Symbol for transition metal (e.g., 'Rh', 'Pt', 'W')
        group_iv: Symbol for Group IV element (e.g., 'Si', 'Ge', 'Sn', 'Pb')
        f_block_element: Symbol for f-block element (e.g., 'Ce', 'Eu', 'Yb'). Replaces
            whichever f-block element is present in the CIF (default: unchanged).

    Returns:
        New Atoms object with substituted elements
    """
    from ase.data import atomic_numbers as ase_atomic_numbers

    # Define element groups
    transition_metals = list(range(22, 31)) + list(range(40, 49)) + list(range(72, 81))
    group_iv_elements = [14, 32, 50, 82]
    # f-block: lanthanides (57-71) + actinides (89-103)
    f_block_elements = list(range(57, 72)) + list(range(89, 104))

    element_to_z = {
        'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Si': 14, 'Ge': 32, 'Sn': 50, 'Pb': 82,
    }

    target_tm_z = element_to_z.get(transition_metal) if transition_metal else None
    target_giv_z = element_to_z.get(group_iv) if group_iv else None
    target_fb_z = ase_atomic_numbers.get(f_block_element) if f_block_element else None

    if target_tm_z is not None and target_tm_z not in transition_metals:
        raise ValueError(f"Invalid transition metal: {transition_metal}")
    if target_giv_z is not None and target_giv_z not in group_iv_elements:
        raise ValueError(f"Invalid Group IV element: {group_iv}")
    if target_fb_z is not None and target_fb_z not in f_block_elements:
        raise ValueError(f"Invalid f-block element: {f_block_element}")

    new_atoms = atoms.copy()
    atomic_numbers = new_atoms.get_atomic_numbers().copy()

    for i, z in enumerate(atomic_numbers):
        if z in transition_metals and target_tm_z is not None:
            atomic_numbers[i] = target_tm_z
        elif z in group_iv_elements and target_giv_z is not None:
            atomic_numbers[i] = target_giv_z
        elif z in f_block_elements and target_fb_z is not None:
            atomic_numbers[i] = target_fb_z

    new_atoms.set_atomic_numbers(atomic_numbers)
    return new_atoms


def build_parser(config: Optional[dict] = None) -> argparse.ArgumentParser:
    """
    Build the run_mcts.py argument parser.

    Args:
        config: Values loaded from config.json (or {}/None). These become argparse
            defaults; an explicit CLI flag always overrides them.
    """
    parser = argparse.ArgumentParser(description='Run MCTS crystal structure optimization')
    parser.add_argument('--iterations', '-n', type=int, default=1000,
                       help='Number of MCTS iterations (default: 1000)')
    parser.add_argument('--structure', '-s', type=str,
                       default='examples/mat_Pb6U1W6_sg191.cif',
                       help='Path to starting crystal structure CIF file')
    parser.add_argument('--output', '-o', type=str, default='mcts_results',
                       help='Output directory name (default: mcts_results)')
    parser.add_argument('--f-block-mode', type=str, default='u_only',
                       choices=['u_only', 'full_f_block', 'experimental', 'lanthanides_u', 'lanthanides_u_extended'],
                       help='F-block substitution mode: u_only (default), full_f_block, experimental (actinides except Ac), lanthanides_u (lanthanides + U, ±1 moves), or lanthanides_u_extended (lanthanides + U, ±3 moves for faster exploration of heavy lanthanides)')
    parser.add_argument('--exploration-constant', '-c', type=float, default=0.1,
                       help='Exploration constant for UCB calculation (default: 0.1)')
    parser.add_argument('--rollout-method', type=str, default='ehull',
                       choices=['ehull', 'ehull_rdos', 'rdos'],
                       help='Rollout method: ehull (tanh-transformed energy above hull only; MACE + Materials Project, no DFT/DOSCAR data needed), '
                            'ehull_rdos (ehull + rDOS, weighted by --beta/--gamma; requires doscar_peaks_data_with_U.csv), '
                            'or rdos (rDOS only, computed in real time from doscar_peaks_data_with_U.csv; no MACE/Materials Project needed). '
                            'Reward: ehull -> ehull_reward(e_above_hull); ehull_rdos -> beta*ehull_reward(e_above_hull) + gamma*r_DOS; rdos -> r_DOS')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for the E_hull reward when using ehull_rdos (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.0001,
                       help='Weight for the rDOS reward when using ehull_rdos (default: 0.0001)')
    parser.add_argument('--termination-limit', type=int, default=60,
                       help='Number of visits before terminating a node without improvement (default: 60)')
    parser.add_argument('--move-step', type=int, default=1,
                       help='Max positions to move per substitution step for TM, group-IV, and '
                            'lanthanide axes (default: 1 = adjacent only; 3 = extended mode)')
    parser.add_argument('--selection-mode', type=str, default='ucb1',
                       choices=['ucb1', 'epsilon_greedy', 'boltzmann', 'puct', 'hybrid'],
                       help='Child-selection strategy for the MCTS selection phase: '
                            'ucb1 (default; classic UCB1, deterministic argmax of Q/N + c*sqrt(ln(N_parent)/N)), '
                            'epsilon_greedy (argmax UCB1, but pick uniformly at random with probability --epsilon), '
                            'boltzmann (stochastic; P(child) proportional to exp(UCB1 / --temperature)), '
                            'puct (AlphaZero-style Q + c*prior*sqrt(N_parent)/(1+N) with a uniform prior, since there is no learned policy network), '
                            'hybrid (this codebase\'s original default before --selection-mode existed: epsilon-greedy outer loop, '
                            'but the exploratory draw is weighted by UCB1-squared rather than uniform).')
    parser.add_argument('--epsilon', '-e', type=float, default=0.2,
                       help='Exploration rate for epsilon_greedy/hybrid selection modes (default: 0.2, range: 0.0-1.0). Unused by ucb1/boltzmann/puct.')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for the boltzmann selection mode (default: 1.0). Higher = more uniform/exploratory, lower = closer to greedy argmax. Unused by other selection modes.')
    parser.add_argument('--rollout-depth', type=int, default=2,
                       help='Number of random substitution steps per rollout walk (default: 2). Each step evaluates a new candidate; the max reward along the walk is returned (max-along-walk behavior).')
    parser.add_argument('--n-rollout', type=int, default=2,
                       help='Number of independent rollout walks per expansion (default: 2). The first walk has depth 0 (evaluates the node itself); remaining walks use --rollout-depth steps.')
    parser.add_argument('--rollout-aggregation', type=str, default='max',
                       choices=['max', 'mean'],
                       help="How to combine a node's --n-rollout reward samples into its reward: "
                            "max (default; optimistic, biased upward by however many samples are drawn - "
                            "the original behavior; the n_rollout-1 extra samples are discounted by "
                            "--rollout-discount**rollout_depth before comparison) or mean (plain average across "
                            "UNDISCOUNTED samples, an unbiased estimate of the node's expected reward - "
                            "the depth discount is dropped here since discounting-then-averaging-by-"
                            "unweighted-n would just drag the mean toward zero rather than reflect "
                            "lower confidence in farther samples).")
    parser.add_argument('--rollout-discount', type=float, default=0.9,
                       help="Base of the depth-dependent discount applied to extra rollout samples under "
                            "'max' aggregation: extra samples are scaled by rollout_discount**rollout_depth "
                            "before taking the max (default: 0.9). Set to 1.0 to compare all samples on "
                            "equal footing with no discount. Unused under 'mean' aggregation.")
    parser.add_argument('--n-workers', type=int, default=1,
                       help='Number of worker threads to evaluate each expansion\'s rollout simulations concurrently (default: 1, i.e. sequential). Useful for ehull/ehull_rdos, where each rollout is an independent MACE relaxation + Materials Project call. For a fixed --n-workers value, --seed still reproduces identical results run-to-run (results will differ from a different --n-workers value at the same seed, since the RNG is consumed differently).')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--mp-api-key', type=str, default=None,
                       help='Materials Project API key (required for rollout methods: ehull, ehull_rdos). Prefer setting this in config.json instead of passing it on the command line.')
    parser.add_argument('--no-labels', action='store_true',
                       help='Turn off labels on radial tree visualization (default: labels shown)')
    parser.add_argument('--transition-metal', type=str, default=None,
                       help='Override transition metal in starting structure (e.g., Rh, Pt, W). If specified, substitutes TM in loaded CIF file.')
    parser.add_argument('--group-iv', type=str, default=None,
                       help='Override Group IV element in starting structure (e.g., Si, Ge, Sn, Pb). If specified, substitutes Group IV in loaded CIF file.')
    parser.add_argument('--f-block-element', type=str, default=None,
                       help='Override f-block (rare-earth) element in starting structure (e.g., Ce, Eu, Yb). Replaces whichever f-block element is present in the CIF. Default: unchanged (U when using the standard CIF).')

    if config:
        # Local config supplies defaults; explicit CLI flags still take precedence.
        known_dests = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in config.items() if k in known_dests})

    return parser


def main():
    """Main MCTS runner function."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    config = load_config()
    parser = build_parser(config)
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # Validate that MP API key is provided when needed
    methods_requiring_api_key = ['ehull', 'ehull_rdos']
    if args.rollout_method in methods_requiring_api_key and args.mp_api_key is None:
        logger.error(f"--mp-api-key is required when using rollout method '{args.rollout_method}'")
        logger.info(f"   Energy above hull calculations require Materials Project API access")
        logger.info(f"   Get your API key from: https://materialsproject.org/api")
        logger.info(f"   Then run with: --mp-api-key YOUR_KEY, or set \"mp_api_key\" in config.json")
        return 1

    # Check if DOSCAR peaks file exists for methods that need rDOS (rewards are
    # always computed in real time from raw peak data - no precomputed cache)
    if args.rollout_method in ['ehull_rdos', 'rdos']:
        peaks_file = Path("doscar_peaks_data_with_U.csv")
        if not peaks_file.exists():
            logger.error(f"DOSCAR peaks file (doscar_peaks_data_with_U.csv) not found for rollout method {args.rollout_method}")
            logger.info(f"   Provide doscar_peaks_data_with_U.csv (raw peaks) in the repo root")
            logger.info(f"   See the README's Data Availability section for the expected schema.")
            return 1

    logger.info("=" * 80)
    logger.info("MCTS CRYSTAL STRUCTURE OPTIMIZATION")
    logger.info("=" * 80)

    # Step 1: Load starting crystal structure
    logger.info(f"\n1. Loading starting crystal structure...")
    structure_path = Path(args.structure)

    if not structure_path.exists():
        logger.error(f"Structure file not found: {structure_path}")
        logger.info(f"   Please check the file path or use the default structure")
        return 1

    try:
        atoms = read(str(structure_path))
        logger.info(f"   Loaded: {atoms.get_chemical_formula()}")
        logger.info(f"   File: {structure_path}")
        logger.info(f"   Atoms: {len(atoms)}")

        # Override composition if requested
        if args.transition_metal is not None or args.group_iv is not None or args.f_block_element is not None:
            logger.info(f"\n1.5. Overriding starting composition...")
            atoms = override_composition(atoms, args.transition_metal, args.group_iv, args.f_block_element)
            logger.info(f"   New composition: {atoms.get_chemical_formula()}")
    except Exception as e:
        logger.error(f"Error loading structure: {e}")
        return 1

    # Step 2: Set up energy calculator
    logger.info(f"\n2. Setting up energy calculator...")
    csv_file = Path("high_throughput_mace_results.full.csv")

    if not csv_file.exists():
        logger.error(f"MACE calculations file not found: {csv_file}")
        logger.info(f"   Please ensure high_throughput_mace_results.full.csv is in the working directory")
        logger.info(f"   See the README's Data Availability section for the expected schema.")
        return 1

    try:
        df = pd.read_csv(csv_file)
        energy_calc = MaceEnergyCalculator(csv_file=str(csv_file), mp_api_key=args.mp_api_key)
        logger.info(f"   Cached calculations: {len(df)} entries")
        logger.info(f"   Energy range: {df['e_form'].min():.3f} to {df['e_form'].max():.3f} eV/atom")
        if args.mp_api_key:
            logger.info(f"   Materials Project API key provided")
        else:
            logger.warning(f"   No MP API key - energy above hull will be approximate (e_above_hull = e_form)")
    except Exception as e:
        logger.error(f"Error setting up energy calculator: {e}")
        return 1

    # Load DOSCAR rewards if this rollout method uses rDOS
    doscar_lookup = None
    if args.rollout_method in ['ehull_rdos', 'rdos']:
        logger.info(f"\n2.5. Loading DOSCAR rewards...")
        try:
            doscar_lookup = DoscarRewardLookup()
        except Exception as e:
            logger.error(f"Error loading DOSCAR rewards: {e}")
            logger.error(f"   Cannot continue without DOSCAR rewards for '{args.rollout_method}' method")
            return 1

    # Step 3: Initialize MCTS
    logger.info(f"\n3. Initializing MCTS algorithm...")
    try:
        root_node = MCTSTreeNode(atoms, f_block_mode=args.f_block_mode,
                                exploration_constant=args.exploration_constant,
                                termination_limit=args.termination_limit,
                                move_step=args.move_step)

        # Calculate energies for root node (tracked for reference even when not part of the reward)
        root_e_form, root_e_hull = energy_calc.calculate_energies(atoms)
        root_node.e_form = root_e_form
        root_node.e_above_hull = root_e_hull

        mcts = MCTS(root_node, epsilon=args.epsilon, temperature=args.temperature)

        logger.info(f"   Root compound: {root_node.get_chemical_formula()}")
        logger.info(f"   Root E_form: {root_e_form:.4f} eV/atom")
        logger.info(f"   Root E_hull: {root_e_hull:.4f} eV/atom")
        logger.info(f"   F-block mode: {args.f_block_mode}")
        logger.info(f"   Exploration constant: {args.exploration_constant}")
        logger.info(f"   Selection mode: {args.selection_mode}")
        if args.selection_mode in ('epsilon_greedy', 'hybrid'):
            logger.info(f"   Epsilon (exploration rate): {args.epsilon}")
        if args.selection_mode == 'boltzmann':
            logger.info(f"   Temperature: {args.temperature}")
        logger.info(f"   Termination limit: {args.termination_limit}")

        # Show search space
        root_node.expand()
        logger.info(f"   Search space: {len(root_node.expansion_list)} possible moves")
        logger.info(f"   Transition metals: {len(root_node.metal_move)} options")
        logger.info(f"   Group IV elements: {len(root_node.g_iv_move)} options")
        if hasattr(root_node, 'f_block_move'):
            if args.f_block_mode == 'u_only':
                logger.info(f"   F-block elements: {len(root_node.f_block_move)} options (U-only mode)")
            else:
                logger.info(f"   F-block elements: {len(root_node.f_block_move)} options (full f-block)")

    except Exception as e:
        logger.error(f"Error initializing MCTS: {e}")
        return 1

    # Step 4: Run MCTS optimization
    logger.info(f"\n4. Running MCTS optimization...")
    logger.info(f"   Iterations: {args.iterations}")
    logger.info(f"   Rollout method: {args.rollout_method}")
    logger.info(f"   Rollout depth: {args.rollout_depth}")
    logger.info(f"   Number of rollouts: {args.n_rollout}")
    logger.info(f"   Rollout aggregation: {args.rollout_aggregation}")
    logger.info(f"   Rollout discount: {args.rollout_discount}")
    logger.info(f"   Worker threads per expansion: {args.n_workers}")
    if args.rollout_method == 'ehull_rdos':
        logger.info(f"   Beta (E_hull weight, ehull_reward = -tanh(120*(E_hull-0.05))): {args.beta}")
        logger.info(f"   Gamma (rDOS weight): {args.gamma}")
    logger.info(f"   This may take several minutes depending on cache hit rate...")

    try:
        results = mcts.run(
            n_iterations=args.iterations,
            energy_calculator=energy_calc,
            rollout_depth=args.rollout_depth,
            n_rollout=args.n_rollout,
            selection_mode=args.selection_mode,
            rollout_method=args.rollout_method,
            beta=args.beta,
            gamma=args.gamma,
            doscar_lookup=doscar_lookup,
            n_workers=args.n_workers,
            rollout_aggregation=args.rollout_aggregation,
            rollout_discount=args.rollout_discount
        )

        logger.info(f"   Completed: {results['iterations_completed']} iterations")
        logger.info(f"   Compounds explored: {len(results['stat_dict'])}")
        logger.info(f"   Search terminated: {results['terminated']}")

    except Exception as e:
        logger.error(f"Error during MCTS run: {e}")
        return 1

    # Step 5: Analyze results
    logger.info(f"\n5. Analyzing results...")
    try:
        analyzer = ResultsAnalyzer(csv_file=str(csv_file))

        # Get efficiency metrics
        efficiency = analyzer.analyze_search_efficiency(mcts.stat_dict)

        logger.info(f"   Best formation energy: {efficiency['best_formation_energy']:.4f} eV/atom")
        logger.info(f"   Best compound: {results['best_node_formula']}")
        logger.info(f"   Compounds within 100 meV of hull: {efficiency['compounds_near_hull_100meV']}")
        logger.info(f"   Search efficiency: {efficiency['search_diversity']:.4f}")

    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        return 1

    # Step 6: Save results
    logger.info(f"\n6. Saving results...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save convergence history first (before visualizations that might fail)
    try:
        mcts.save_convergence_history(str(output_dir / "convergence_history.csv"))
    except Exception as e:
        logger.warning(f"Could not save convergence history: {e}")

    try:
        # Create visualizations
        visualizer = TreeVisualizer()

        # Radial tree visualization generation disabled (use composite radial tree instead)

        # Skipping some auxiliary visualizations per publication configuration
        # (energy_distribution, iteration_progress, energy_above_hull_*, formation_energy_by_elements)
        # Only essential visualizations are generated to avoid duplicate/unused figures.

        # Export data
        analyzer.export_results(
            mcts.stat_dict,
            output_dir / "all_compounds.csv"
        )

        # Get and display top compounds
        top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)

        logger.info(f"   Results saved to: {output_dir.absolute()}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Don't return early - still save MCTS pickle below

    # Save MCTS object for later visualization (outside try-except to ensure it always runs)
    try:
        import pickle
        mcts_pickle_path = output_dir / "mcts_object.pkl"
        with open(mcts_pickle_path, 'wb') as f:
            pickle.dump(mcts, f)
        logger.info(f"   MCTS object saved to: {mcts_pickle_path}")
    except Exception as e:
        logger.error(f"Error saving MCTS pickle: {e}")

    # Step 7: Display summary
    logger.info(f"\n" + "=" * 80)
    logger.info("MCTS OPTIMIZATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Best compound: {results['best_node_formula']}")
    logger.info(f"Formation energy: {results['best_node_e_form']:.4f} eV/atom")
    logger.info(f"Total compounds explored: {len(results['stat_dict'])}")
    logger.info(f"Results directory: {output_dir.absolute()}")

    logger.info(f"\nTOP 10 COMPOUNDS DISCOVERED:")
    logger.info("-" * 60)
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        stability = "Stable" if row['e_above_hull'] < 0.1 else "Metastable"
        logger.info(f"{i:2d}. {row['formula']:15s} | "
                    f"E_form: {row['formation_energy']:8.4f} eV/atom | "
                    f"{stability}")

    logger.info(f"\nFILES CREATED:")
    logger.info(f"   - radial_tree_visualization.png - Tree structure with formation energies")
    logger.info(f"   - energy_distribution.png - Formation energy distribution")
    logger.info(f"   - iteration_progress.png - Search progress over iterations")
    logger.info(f"   - energy_above_hull_distribution.png - Energy above hull distribution")
    logger.info(f"   - energy_above_hull_progress.png - Energy above hull search progress")
    logger.info(f"   - formation_energy_by_elements.png - Formation energy by transition metal/Group IV")
    logger.info(f"   - energy_above_hull_by_elements.png - Energy above hull by transition metal/Group IV")
    logger.info(f"   - all_compounds.csv - All discovered compounds data")

    logger.info(f"\nTo run again:")
    logger.info(f"   python run_mcts.py --iterations {args.iterations}")
    logger.info(f"   python run_mcts.py --iterations 1000  # Longer search")
    logger.info(f"   python run_mcts.py --structure my_file.cif  # Different starting material")
    logger.info(f"   python run_mcts.py --f-block-mode lanthanides_u_extended  # Fast exploration of heavy lanthanides")
    logger.info(f"   python run_mcts.py --rollout-method ehull_rdos --beta 1.0 --gamma 0.0001  # E_hull + rDOS")
    logger.info(f"   python run_mcts.py -c 0.2  # Higher exploration constant")

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

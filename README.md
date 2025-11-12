# MCTS Materials

A Monte Carlo Tree Search (MCTS) implementation for discovering and optimizing stable intermetallic crystal structures containing uranium and f-block elements by iteratively exploring chemical space guided by formation energies and thermodynamic stability metrics from MACE energy calculations.

## Overview

This project applies MCTS, a reinforcement learning algorithm traditionally used in game playing, to the problem of materials discovery. The algorithm intelligently explores the vast chemical space of possible crystal structures by:

1. **Selection**: Choosing promising compounds to explore using Upper Confidence Bound (UCB) criteria
2. **Expansion**: Generating new candidate structures through element substitution
3. **Simulation**: Evaluating structures using formation energy and energy above hull calculations
4. **Backpropagation**: Updating the search tree based on discovered energies

The search focuses on intermetallic compounds with transition metals, Group IV elements (Si, Ge, Sn, Pb), and f-block elements (lanthanides and actinides), aiming to discover thermodynamically stable or metastable structures.

## Key Features

- **Intelligent exploration** of chemical space using MCTS with UCB-based selection
- **Multiple rollout methods** to balance formation energy and thermodynamic stability
- **Flexible f-block substitution modes** (U-only, full f-block, or experimental)
- **High-throughput energy calculations** using cached MACE results
- **Comprehensive visualization** including tree structures, energy distributions, and iteration progress
- **Automated analysis** with efficiency metrics and compound ranking

## Installation

### Requirements

```bash
pip install ase pandas numpy matplotlib scipy
```

You'll also need:
- Python 3.8+
- Materials Project API key (required only for energy above hull calculations)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mcts_materials
```

2. Ensure you have the high-throughput energy database:
   - The code expects `high_throughput_mace_results.full.csv` in the working directory
   - This file contains pre-computed MACE formation energies and energy above hull values

3. Get a Materials Project API key (if using energy above hull):
   - Register at https://materialsproject.org/
   - Navigate to your dashboard and copy your API key
   - **Note**: API key is only required for rollout methods: `eh`, `both`, or `weighted`
   - Not needed for `rollout-method='fe'` (formation energy only)

## Usage

### Basic Usage

Run MCTS with default parameters:

```bash
python run_mcts.py --mp-api-key YOUR_API_KEY
```

**Note**: The default rollout method is `weighted`, which requires an API key. If you don't have an API key, use `--rollout-method fe` instead (formation energy only).

```bash
# Without API key (formation energy only)
python run_mcts.py --rollout-method fe
```

This will:
- Use the default starting structure (`examples/mat_Pb6U1W6_sg191.cif`)
- Run 1000 iterations
- Save results to `mcts_results/` directory
- Generate visualizations and analysis reports

### Custom Parameters

```bash
# Custom number of iterations
python run_mcts.py --iterations 500 --mp-api-key YOUR_API_KEY

# Custom starting structure
python run_mcts.py --structure my_structure.cif --mp-api-key YOUR_API_KEY

# Different rollout method (requires API key)
python run_mcts.py --rollout-method eh --mp-api-key YOUR_API_KEY

# Full f-block substitution mode
python run_mcts.py --f-block-mode full_f_block --mp-api-key YOUR_API_KEY

# Adjust energy above hull weighting
python run_mcts.py --eh-weight 10.0 --mp-api-key YOUR_API_KEY

# Higher exploration
python run_mcts.py --exploration-constant 0.2 --mp-api-key YOUR_API_KEY

# Custom output directory
python run_mcts.py --output my_results --mp-api-key YOUR_API_KEY
```

### Example Commands

```bash
# Quick test run (formation energy only, no API key needed)
python run_mcts.py --iterations 100 --rollout-method fe

# Formation energy optimization only (no API key needed)
python run_mcts.py --iterations 1000 --rollout-method fe

# Hull stability optimization only (requires API key)
python run_mcts.py --iterations 1000 --rollout-method eh --mp-api-key YOUR_API_KEY

# Balanced optimization (default, requires API key)
python run_mcts.py --iterations 1000 --rollout-method weighted --eh-weight 5.0 --mp-api-key YOUR_API_KEY

# Full f-block exploration (requires API key)
python run_mcts.py --iterations 1000 --f-block-mode full_f_block --rollout-method weighted --mp-api-key YOUR_API_KEY
```

## Hyperparameters

### Core MCTS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 1000 | Number of MCTS iterations to perform |
| `--structure` | `examples/mat_Pb6U1W6_sg191.cif` | Path to starting crystal structure (CIF format) |
| `--output` | `mcts_results` | Output directory for results and visualizations |

### Search Strategy Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--rollout-method` | `weighted` | `fe`, `eh`, `both`, `weighted` | Rollout evaluation method |
| `--eh-weight` | 5.0 | float | Weight for energy above hull in weighted mode (higher = prioritize stability) |
| `--mp-api-key` | None | string | Materials Project API key (required for `eh`, `both`, `weighted` methods) |
| `--exploration-constant` | 0.1 | float | UCB exploration constant (higher = more exploration vs exploitation) |
| `--f-block-mode` | `u_only` | `u_only`, `full_f_block`, `experimental` | F-block element substitution strategy |
| `--no-labels` | False | flag | Turn off labels in radial tree visualization |

### Rollout Method Details

- **`fe` (Formation Energy)**: Optimizes for lowest formation energy only
  - Best for finding thermodynamically stable compounds
  - Reward = -e_form
  - **No API key required**

- **`eh` (Energy Above Hull)**: Optimizes for lowest energy above hull only
  - Best for finding compounds stable against decomposition
  - Reward = -e_above_hull
  - **Requires Materials Project API key**

- **`both`**: Simple mixture of both metrics
  - Uses fe for first rollout, eh for subsequent rollouts
  - Reward = -e_form - e_above_hull (unweighted)
  - **Requires Materials Project API key**

- **`weighted` (Recommended)**: Tunable weighted combination
  - Balances both metrics with adjustable weighting
  - Reward = -e_form - α × e_above_hull (where α = eh_weight)
  - Default α = 5.0 provides balanced optimization
  - Increase α to prioritize hull stability
  - Decrease α to prioritize formation energy
  - **Requires Materials Project API key**

### Materials Project API Key

The Materials Project API is used to calculate **energy above hull**, which measures thermodynamic stability against decomposition. This requires querying the Materials Project database for phase diagram information.

**When is the API key required?**
- Required for rollout methods: `eh`, `both`, `weighted`
- Not required for: `fe` (formation energy only)

**How to provide your API key:**
```bash
python run_mcts.py --mp-api-key YOUR_API_KEY --rollout-method weighted
```

**What happens without an API key?**
- If you try to use `eh`, `both`, or `weighted` without an API key, the script will exit with an error
- Use `--rollout-method fe` to run without an API key

### F-Block Substitution Modes

- **`u_only`** (Default): Only uranium (U) substitutions allowed
  - Fastest, focused search
  - Ideal for uranium-containing intermetallics

- **`full_f_block`**: Full lanthanide and actinide series
  - Explores lanthanides (Ce-Lu) and actinides (Th-Pu)
  - Allows "vertical" moves between analogous elements
  - Larger search space

- **`experimental`**: Lanthanides (minus La) plus uranium
  - Focuses on experimentally accessible actinides
  - Excludes La, includes Ce-Lu and U
  - Good balance of search space and practicality

### Internal Parameters (Fixed)

These are set in the code and optimized for typical use:
- `rollout_depth`: 1 (depth of random substitutions during rollout)
- `n_rollout`: 5 (number of rollout simulations per expansion)
- `selection_mode`: 'epsilon' (ε-greedy with 20% random selection)

## Output Files

After running MCTS, the output directory contains:

### Visualizations

- **`radial_tree_visualization.png`**: Tree structure showing explored compounds and their relationships
- **`energy_distribution.png`**: Formation energy distribution for top compounds
- **`iteration_progress.png`**: Best formation energy found over iterations
- **`energy_above_hull_distribution.png`**: Energy above hull distribution
- **`energy_above_hull_progress.png`**: Best energy above hull over iterations
- **`formation_energy_by_elements.png`**: Heatmap showing formation energies by element combination
- **`energy_above_hull_by_elements.png`**: Heatmap showing hull energies by element combination

### Data Files

- **`all_compounds.csv`**: Complete list of all explored compounds with energies and statistics
- **`mcts_report.txt`**: Detailed text report with search efficiency metrics

### Report Contents

The text report includes:
- Best compounds discovered (by formation energy and hull stability)
- Search efficiency metrics
- Number of compounds within 100 meV of convex hull
- Diversity of explored chemical space

## Understanding the Results

### Key Metrics

- **Formation Energy (e_form)**: Energy per atom relative to elemental references
  - Negative values indicate exothermic formation (stable)
  - More negative = more stable

- **Energy Above Hull (e_above_hull)**: Energy above the convex hull of stable phases
  - Zero or negative = thermodynamically stable
  - 0-0.1 eV/atom = potentially synthesizable metastable phase
  - \>0.1 eV/atom = likely unstable against decomposition

### Interpreting Visualizations

- **Tree visualization**: Shows parent-child relationships and exploration paths
  - Node size indicates visit frequency
  - Color indicates formation energy (cooler = more stable)

- **Energy distributions**: Show the landscape of discovered compounds
  - Look for clusters of low-energy compounds

- **Progress plots**: Show learning efficiency
  - Steeper drops indicate effective exploration
  - Plateaus suggest converged search

## Project Structure

```
mcts_materials/
├── run_mcts.py                    # Main runner script
├── mcts_crystal/                  # Core MCTS package
│   ├── __init__.py
│   ├── mcts.py                    # MCTS algorithm implementation
│   ├── node.py                    # Tree node and substitution logic
│   ├── energy_calculator.py       # Energy calculation interface
│   ├── visualization.py           # Plotting and visualization
│   └── analysis.py                # Results analysis tools
├── high_throughput_mace_results.full.csv  # Pre-computed energy database
├── examples/                      # Example structures and data
│   └── mat_Pb6U1W6_sg191.cif     # Default starting structure
└── sensitivity_studies/           # Parameter sensitivity analysis scripts
    ├── rollout_method_comparison.py
    ├── eh_weight_sensitivity.py
    ├── exploration_sensitivity.py
    └── ...
```

## Sensitivity Studies

The `sensitivity_studies/` directory contains scripts for analyzing the effect of various hyperparameters:

- **`rollout_method_comparison.py`**: Compare fe, eh, both, and weighted methods
- **`eh_weight_sensitivity.py`**: Test different eh_weight values (1.0 to 10.0)
- **`exploration_sensitivity.py`**: Test exploration constants (0.05 to 0.5)
- **`rollout_depth_sensitivity.py`**: Compare rollout depths (0 to 3)
- **`selection_mode_sensitivity.py`**: Compare selection strategies
- **`starting_material_sensitivity.py`**: Test different starting structures

These help understand optimal parameter settings for different materials discovery goals.

**Note on API keys for sensitivity studies:**
If a sensitivity study uses rollout methods that require energy above hull (`eh`, `both`, `weighted`), you must configure your Materials Project API key:

1. Open the sensitivity study script
2. Find the configuration section at the top: `MP_API_KEY = None`
3. Replace with your key: `MP_API_KEY = "YOUR_API_KEY"`
4. Run the script

Example:
```python
# At the top of the sensitivity study file
MP_API_KEY = "your_actual_key_here"  # Replace this
```

## Algorithm Details

### MCTS Loop

1. **Selection Phase**: Start at root, traverse tree selecting children with highest UCB values
   - UCB = (total_reward / visits) + c × √(ln(parent_visits) / visits)
   - Balances exploitation (high reward) and exploration (low visits)

2. **Expansion Phase**: When reaching a leaf node, create child nodes by:
   - Substituting transition metals (move ±1 period or ±1 group)
   - Substituting Group IV elements (Si → Ge → Sn → Pb)
   - Substituting f-block elements (based on f-block mode)

3. **Simulation Phase**: Perform rollout simulations:
   - Evaluate current node (depth=0)
   - Perform random substitutions for additional rollouts (depth=1)
   - Calculate reward based on rollout method

4. **Backpropagation Phase**: Update all nodes in selection chain:
   - Add reward to total_reward
   - Increment visit count
   - Update best_reward if improved

### Termination Criteria

Search terminates when:
- All iterations completed, OR
- All leaf nodes marked as terminated (visited 30 times without improvement)

## Tips for Effective Use

1. **Get a Materials Project API key** if you want to use energy above hull optimization (`eh`, `both`, or `weighted` methods)
   - Or use `--rollout-method fe` if you don't have an API key
2. **Start with default parameters** to understand baseline behavior
3. **Use weighted rollout method** (default) for balanced optimization when you have an API key
4. **Increase eh_weight** (e.g., 10.0) if prioritizing thermodynamic stability
5. **Increase iterations** (e.g., 2000-5000) for more thorough exploration
6. **Use u_only mode** for focused uranium materials discovery
7. **Use full_f_block mode** for broader lanthanide/actinide exploration
8. **Check energy_above_hull values** - aim for < 0.1 eV/atom for synthesizability
9. **Monitor iteration progress plots** to assess convergence

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mcts_materials,
  title = {PACEHOLDER},
  author = {PLACEHOLDER},
  year = {2025},
  url = {https://github.com/lanl/mcts_materials}
}
```

## Contact

[placeholder]

## Acknowledgments

- MACE (Machine Learning Aided Chemical Equilibrium) for energy calculations
- Materials Project for thermodynamic data
- ASE (Atomic Simulation Environment) for structure manipulation

## Copyright

© 2025. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.(Copyright request O5871).

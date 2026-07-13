# MCTS Materials

A Monte Carlo Tree Search (MCTS) implementation for discovering and optimizing stable intermetallic crystal structures containing uranium and f-block elements by iteratively exploring chemical space guided by thermodynamic stability (energy above hull) and electronic density-of-states (rDOS) metrics.

## Overview

This project applies MCTS, a reinforcement learning algorithm traditionally used in game playing, to the problem of materials discovery. The algorithm intelligently explores the vast chemical space of possible crystal structures by:

1. **Selection**: Choosing promising compounds to explore using Upper Confidence Bound (UCB) criteria
2. **Expansion**: Generating new candidate structures through element substitution
3. **Simulation**: Evaluating structures using a sharp tanh-transformed energy-above-hull reward and/or a DOSCAR-derived electronic structure reward (rDOS)
4. **Backpropagation**: Updating the search tree based on discovered rewards

The search focuses on intermetallic compounds with transition metals, Group IV elements (Si, Ge, Sn, Pb), and f-block elements (lanthanides and actinides), aiming to discover thermodynamically stable or metastable structures with favorable electronic structure near the Fermi level.

## Key Features

- **Intelligent exploration** of chemical space using MCTS, with a choice of five child-selection strategies (`ucb1`, `epsilon_greedy`, `boltzmann`, `puct`, `hybrid`) via `--selection-mode`
- **Three rollout methods** (`ehull`, `ehull_rdos`, `rdos`) for stability- and/or electronic-structure-guided search
- **Flexible f-block substitution modes** (U-only, full f-block, experimental, lanthanides+U, or extended lanthanides+U)
- **High-throughput energy calculations** using cached MACE results
- **Comprehensive visualization** including tree structures, energy distributions, and iteration progress
- **Automated analysis** with efficiency metrics and compound ranking
- **Hyperparameter sensitivity studies** (replicate-run convergence sweeps) for calibrating MCTS-algorithm parameters - see [Sensitivity Studies](#sensitivity-studies)

## Installation

### Requirements

Python 3.9+. The package is installed via `pip` using `pyproject.toml`, in one of three sizes depending on what you need:

```bash
pip install ase pandas numpy matplotlib scipy mace-torch matbench-discovery
```

`pip install -r requirements.txt` is equivalent to `pip install -e .[full]` and still works if that's the habit you're in.

The core install (no `[full]`) is intentionally lightweight: `rollout-method rdos` and the test suite don't need MACE, Materials Project, or pymatgen at all, so you can work on the search algorithm itself without installing the heavier ML/DFT stack.

Once installed, you get a console command in addition to the script entry point:

```bash
mcts-run --rollout-method ehull_rdos --beta 1.0 --gamma 0.0001   # equivalent to: python run_mcts.py --rollout-method ehull_rdos ...
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mcts_materials
```

2. Get a Materials Project API key (if using energy above hull):
   - Register at https://materialsproject.org/
   - Navigate to your dashboard and copy your API key
   - **Note**: API key is required for rollout methods `ehull` and `ehull_rdos`
   - Not needed for `--rollout-method rdos` (rDOS only)

## Usage

### Basic Usage

With `config.json` set up (see above):

```bash
python run_mcts.py
```

Without `config.json`, pass the key explicitly:

```bash
python run_mcts.py --mp-api-key YOUR_API_KEY
```

The default rollout method is `ehull`, which needs the Materials Project API key but no DOSCAR data. To run with no API key at all, use `--rollout-method rdos` (requires `doscar_peaks_data_with_U.csv`).

This will:
- Use the default starting structure (`examples/mat_Pb6U1W6_sg191.cif`)
- Run 1000 iterations
- Save results to `mcts_results/` directory
- Generate visualizations and analysis reports

### Example Commands

```bash
# E_hull only - MACE + Materials Project, no DFT/DOSCAR data needed
python run_mcts.py --iterations 1000 --rollout-method ehull

# E_hull + rDOS (the published study's reward)
python run_mcts.py --iterations 1000 --rollout-method ehull_rdos --beta 1.0 --gamma 0.0001

# rDOS only
python run_mcts.py --iterations 1000 --rollout-method rdos

# Full f-block exploration
python run_mcts.py --iterations 1000 --f-block-mode full_f_block --rollout-method ehull_rdos
```

To reproduce the published U-only `ehull_rdos` study and its figures end to end, see [analysis/ehull_rdos_u_only_study/](analysis/ehull_rdos_u_only_study/run_study.sh).

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
| `--rollout-method` | `ehull` | `ehull`, `ehull_rdos`, `rdos` | Rollout evaluation method |
| `--beta` | 1.0 | float | Weight for the E_hull reward in `ehull_rdos` |
| `--gamma` | 0.0001 | float | Weight for the rDOS reward in `ehull_rdos` |
| `--mp-api-key` | None | string | Materials Project API key (required for `ehull`, `ehull_rdos`; prefer `config.json`) |
| `--exploration-constant` | 0.1 | float | UCB1/PUCT exploration constant `c` (higher = more exploration vs exploitation) |
| `--selection-mode` | `ucb1` | `ucb1`, `epsilon_greedy`, `boltzmann`, `puct`, `hybrid` | Child-selection strategy for the selection phase - see [Child Selection Methods](#child-selection-methods) |
| `--epsilon` | 0.2 | float | Exploration rate, used by `epsilon_greedy`/`hybrid` selection modes only |
| `--temperature` | 1.0 | float | Softmax temperature, used by `boltzmann` selection mode only |
| `--f-block-mode` | `u_only` | `u_only`, `full_f_block`, `experimental`, `lanthanides_u`, `lanthanides_u_extended` | F-block element substitution strategy |
| `--transition-metal` | None | element symbol | Override the transition metal in the starting structure |
| `--group-iv` | None | element symbol | Override the Group IV element in the starting structure |
| `--no-labels` | False | flag | Turn off labels in radial tree visualization |

### Rollout Method Details

- **`ehull` (default)**: Sharp tanh-transformed energy above hull
  - `ehull_reward(E_hull) = -tanh(120 * (E_hull - 0.05))` — a sharp transition around the 0.05 eV/atom stability threshold (≈+1 for stable compounds, ≈-1 for unstable ones)
  - Reward = `ehull_reward(E_hull)`
  - **Requires a Materials Project API key. Does not require DOSCAR/DFT data.**

- **`ehull_rdos`**: E_hull + electronic density-of-states reward, the formulation used in the published study
  - Reward = `beta * ehull_reward(E_hull) + gamma * r_DOS`
  - Default `beta=1.0`, `gamma=0.0001`
  - **Requires a Materials Project API key and `doscar_peaks_data_with_U.csv`.**

- **`rdos`**: DOSCAR-derived electronic structure reward only
  - Reward = `r_DOS`, computed in real time from `doscar_peaks_data_with_U.csv`
  - **Requires `doscar_peaks_data_with_U.csv`. Does not require MACE or a Materials Project API key.**

Formation energy (`e_form`) is always computed and logged on every node and in every output CSV for reference, but it is not part of any of the three rewards above.

### Materials Project API Key

The Materials Project API is used to calculate **energy above hull**, which measures thermodynamic stability against decomposition. This requires querying the Materials Project database for phase diagram information.

**When is the API key required?**
- Required for rollout methods: `ehull`, `ehull_rdos`
- Not required for: `rdos`

**How to provide your API key:**
- Preferred: copy `config.example.json` to `config.json` and set `mp_api_key` there (gitignored, read locally, never pushed)
- Or: `python run_mcts.py --mp-api-key YOUR_API_KEY --rollout-method ehull_rdos`

**What happens without an API key?**
- If you try to use `ehull` or `ehull_rdos` without an API key, the script will exit with an error
- Use `--rollout-method rdos` to run without an API key (still requires `doscar_peaks_data_with_U.csv`)

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

- **`lanthanides_u`**: All lanthanides (Ce-Lu) plus U, ±1 nearest-neighbor moves

- **`lanthanides_u_extended`**: All lanthanides (Ce-Lu) plus U, ±3 moves
  - Faster exploration of heavy lanthanides (Tm, Yb, Lu) from any starting point

### Internal Parameters (Fixed defaults, overridable on the CLI)

- `--rollout-depth`: 1 (depth of random substitutions during rollout)
- `--n-rollout`: 5 (number of rollout simulations per expansion)
- `--rollout-aggregation`: `max` (how the `--n-rollout` reward samples are combined into a node's reward)
  - `max` (default): take the best of the samples - optimistic, biased upward by however many samples are drawn (a node sampled more times looks better on average even with no real difference in quality). The `n_rollout - 1` extra (depth > 0) samples are discounted by `0.9 ** rollout_depth` before the comparison, since a speculative sample wandered further from the node is less informative about the node's own quality.
  - `mean`: plain average across the samples - an unbiased estimate of the node's expected reward, at the cost of being more pessimistic about a node whose extra speculative samples happened to land on worse compositions. The extra samples are **not** discounted here: multiplying them by `0.9 ** rollout_depth` and then averaging with an unweighted `n` would just drag the mean toward zero by an amount that grows with `n_rollout`/`rollout_depth`, which isn't a meaningful confidence weighting (zero isn't a neutral reward in this scheme) - so `mean` uses the raw, undiscounted samples instead.

See [Child Selection Methods](#child-selection-methods) for `--selection-mode`, `--epsilon`, and `--temperature`.

## Data Availability

This repository ships **no proprietary DFT/DOSCAR data**. The high-throughput energy/DOS database underlying this work has not been publicly released yet, so the following files are gitignored and must be supplied locally — they are never committed or pushed:

| File (repo root) | Required by | Schema |
|---|---|---|
| `high_throughput_mace_results.full.csv` | all rollout methods | CSV with columns `name` (chemical formula, e.g. `Ti6Si6Ce`), `e_form` (eV/atom), `e_above_hull` (eV/atom), `e_decomp` (eV/atom), `source` (free text) |
| `doscar_peaks_data_with_U.csv` | `ehull_rdos`, `rdos` | Raw DOSCAR peak data (`COMPOUND_NAME`, `PEAK_ENERGY`, `PEAK_WIDTH`, `PEAK_HEIGHT`). rDOS is always computed in real time from this file (Gaussian-weighted sum of peak intensity near the Fermi level — see `mcts_crystal/doscar_utils.py:DoscarRewardLookup`); there is no precomputed rewards cache |

If you don't have these files, `run_mcts.py` will exit with a clear error naming the missing file rather than silently degrading. Once the underlying high-throughput study is released, these files will be published alongside it — check the paper / repo announcements for the data DOI.

`high_throughput_mace_results.full.csv` also acts as a cache: any new compound MACE evaluates during a run is appended to it, so subsequent runs reuse prior calculations.

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
- **`convergence_history.csv`**: Best composite reward (`best_reward`), and best E_form/E_hull/rDOS compound found, per iteration
- **`mcts_report.txt`**: Detailed text report with search efficiency metrics
- **`mcts_object.pkl`**: Pickled `MCTS` object, for offline re-analysis/plotting (e.g. `create_composite_radial_tree.py`)

### Report Contents

The text report includes:
- Best compounds discovered (by formation energy and hull stability)
- Search efficiency metrics
- Number of compounds within 100 meV of convex hull
- Diversity of explored chemical space

## Understanding the Results

### Key Metrics

- **Formation Energy (e_form)**: Energy per atom relative to elemental references (reference metric only - not part of the reward)
  - Negative values indicate exothermic formation (stable)
  - More negative = more stable

- **Energy Above Hull (e_above_hull)**: Energy above the convex hull of stable phases
  - Zero or negative = thermodynamically stable
  - 0-0.1 eV/atom = potentially synthesizable metastable phase
  - \>0.1 eV/atom = likely unstable against decomposition

- **rDOS**: Gaussian-weighted sum of DOS peak height/intensity near the Fermi level
  - Higher = sharper, more intense electronic structure features near E_F (a proxy for heavy-fermion/correlated-electron character)

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
├── run_mcts.py                    # Thin compatibility wrapper - delegates to mcts_crystal/cli.py
├── pyproject.toml                  # Package metadata, dependencies/extras, mcts-run entry point
├── config.example.json            # Local config template (copy to config.json, gitignored)
├── requirements.txt                # Equivalent to `pip install -e .[full]`
├── .gitignore                      # Excludes config.json, data files, caches, run outputs
├── mcts_crystal/                  # Core MCTS package
│   ├── __init__.py
│   ├── cli.py                     # run_mcts.py/mcts-run implementation (argument parsing, config loading)
│   ├── mcts.py                    # MCTS algorithm implementation
│   ├── node.py                    # Tree node, substitution logic, reward functions
│   ├── energy_calculator.py       # MACE + Materials Project energy interface (lazy-imported, optional)
│   ├── doscar_utils.py            # DOSCAR/rDOS reward lookup
│   ├── visualization.py           # Plotting and visualization
│   └── analysis.py                # Results analysis tools
├── tests/                          # pytest suite (mocks MACE/Materials Project; no [full] install needed)
├── .github/workflows/tests.yml     # CI: runs the test suite on push/PR
├── examples/
│   └── mat_Pb6U1W6_sg191.cif      # Default starting structure
├── analysis/
│   ├── ehull_rdos_u_only_study/              # Scripts to reproduce the published U-only ehull_rdos study and figures
│   ├── ehull_rdos_u_only_study_normalized/   # Same study, gamma normalized to 1/(max raw r_DOS) instead of the calibrated 0.0001
│   ├── ehull_rdos_u_only_study_mean_rollout/ # Same study, --rollout-aggregation mean instead of the default max
│   └── ehull_rdos_u_only_study_final/        # Final parameters: normalized gamma + mean rollout aggregation
├── sensitivity_studies/           # Replicate-run hyperparameter sensitivity sweeps (see Sensitivity Studies)
│   ├── scripts/                   # Generation scripts (run_all.sh reproduces everything)
│   └── results/                   # Per-sweep convergence_data.csv + figures
├── high_throughput_mace_results.full.csv  # NOT bundled - see Data Availability
└── doscar_peaks_data_with_U.csv            # NOT bundled - see Data Availability
```

## Reproducing the Published Study

`analysis/ehull_rdos_u_only_study/` contains the scripts used to run and analyze the U-only `ehull_rdos` study (U-only f-block mode; `--rollout-method ehull_rdos`, plus `iterations`/`selection_mode`/`exploration_constant`/`beta`/`gamma`/`transition_metal`/`group_iv` all read from `config.json` rather than hardcoded in the script - see `config.json`/`config.example.json` for current values):

- `run_study.sh`: runs `run_mcts.py` with the published settings, then calls `generate_plots.sh`
- `generate_plots.sh`: regenerates all figures (E_hull-vs-rDOS scatter, convergence-by-starting-material plot, composite-colored radial tree) and the top-15 LaTeX table via `generate_figures.py`
- `sweep_starting_material.py`: replicate-run sweep (5 seeds each) across starting materials at increasing move-graph distance from the global-best compound, feeding `convergence_by_starting_material.png`

This requires `high_throughput_mace_results.full.csv` and `doscar_peaks_data_with_U.csv` locally (see [Data Availability](#data-availability)), and a Materials Project API key via `config.json` or `MP_API_KEY`.

`analysis/ehull_rdos_u_only_study_normalized/` is the same study with gamma fixed to `1 / (max raw r_DOS across the 108 U-only compounds)` = `1/2516.1664410449775` ≈ `0.0003974`, instead of the calibrated `0.0001` - this normalizes the `gamma*r_DOS` term to top out at 1.0, the same scale as `ehull_reward`'s ~[-1,1] range. Unlike the calibrated study, gamma here is hardcoded in `generate_figures.py`/`create_composite_radial_tree.py` and passed explicitly via `--gamma` in `run_study.sh` rather than read from `config.json` (which stays at the calibrated 0.0001). It has its own `sweep_starting_material.py`/`convergence_by_starting_material.png` (`starting_material_sweep_normalized/`), since the calibrated study's sweep uses gamma=0.0001 and the normalized gamma also shifts which compound is the true global-best target (UTi6Sn6 here vs. UZr6Pb6 for the calibrated study), so the starting-material ladder had to be rechosen to land on clean edit distances to the new target.

`analysis/ehull_rdos_u_only_study_final/` combines both parameter changes: `gamma = 1/2516.1664410449775` (normalized) and `--rollout-aggregation mean` — the final settled parameters for this study. It has its own `sweep_starting_material.py`/`convergence_by_starting_material.png` (`starting_material_sweep_final/`, using the same Cr/Fe/Ni/Pt6Sn6U d=2/4/6/8 ladder as the normalized study since gamma determines the global-best target).

`analysis/ehull_rdos_u_only_study_mean_rollout/` is the same study (calibrated gamma=0.0001 unchanged) with `--rollout-aggregation mean` instead of the default `max` (see [Hyperparameters](#hyperparameters) below) - a node's reward is the plain average of its rollout samples rather than the optimistic max, and the depth discount is dropped for the extra samples (see `mcts_crystal/mcts.py`'s `_run_rollout_samples`). It also has its own `sweep_starting_material.py`/`convergence_by_starting_material.png` (`starting_material_sweep_mean_rollout/`, same starting-material ladder as the calibrated study since gamma - and therefore the global-best target - is unchanged here, only `rollout_aggregation='mean'` is added to each replicate).

## Sensitivity Studies

`sensitivity_studies/` holds replicate-run sensitivity sweeps (5 seeds x 500 iterations per value tested) used to calibrate MCTS-algorithm hyperparameters - `exploration_constant`, starting material, `selection_mode`, and `n_rollout`/`rollout_depth`/`termination_limit` - against the calibrated U-only `ehull_rdos` baseline. The reward formula's physics-informed constants (ehull_reward sharpness=120, E_hull threshold=0.05, rDOS sigma=0.5) are intentionally not swept. Run `bash sensitivity_studies/scripts/run_all.sh` to reproduce everything; see `sensitivity_studies/README.md` for full methodology and findings (e.g. starting material is the dominant factor in this search space, while `exploration_constant`/`selection_mode` show no measurable effect for a specific, explained structural reason).

## Algorithm Details

### MCTS Loop

1. **Selection Phase**: Start at root, traverse tree picking one child at each level according to `--selection-mode` (default `ucb1`) - see [Child Selection Methods](#child-selection-methods)

2. **Expansion Phase**: When reaching a leaf node, create child nodes by:
   - Substituting transition metals (move ±1 period or ±1 group)
   - Substituting Group IV elements (Si → Ge → Sn → Pb)
   - Substituting f-block elements (based on f-block mode)

3. **Simulation Phase**: Perform rollout simulations:
   - Evaluate current node (depth=0)
   - Perform random substitutions for additional rollouts (depth>0)
   - Calculate reward based on rollout method (`ehull`, `ehull_rdos`, or `rdos`)

4. **Backpropagation Phase**: Update all nodes in selection chain:
   - Add reward to total_reward
   - Increment visit count
   - Update best_reward if improved

### Termination Criteria

Search terminates when:
- All iterations completed, OR
- All leaf nodes marked as terminated (visited `--termination-limit` times without improvement, default 60)

### Child Selection Methods

`--selection-mode` controls how a child is picked at each level of the tree during the selection phase. In all modes, `N` is a node's visit count, `Q` is its mean reward (`total_reward / N`), and a terminated child is never selected. `c` is `--exploration-constant`.

- **`ucb1` (default)**: Classic UCB1 - deterministic, always picks the child maximizing
  ```
  UCB1 = Q + c * sqrt(ln(N_parent) / N)
  ```
  An unvisited child (`N=0`) has `UCB1 = +inf`, so unvisited children are always tried before any visited one. This is the standard MCTS/UCT selection rule: there is no extra randomization beyond what the formula itself provides - the exploration/exploitation tradeoff comes entirely from how the bonus term shrinks as `N` grows.

- **`epsilon_greedy`**: Textbook epsilon-greedy. With probability `1 - epsilon`, picks `argmax(UCB1)` (identical to `ucb1`); with probability `epsilon` (`--epsilon`, default 0.2), picks a child **uniformly at random**, ignoring UCB1 value entirely.

- **`boltzmann`**: Softmax/Boltzmann exploration. Always stochastic - picks child `i` with probability
  ```
  P(i) ∝ exp(UCB1_i / T)
  ```
  where `T` is `--temperature` (default 1.0). Lower `T` biases toward greedy `argmax`; higher `T` biases toward uniform random. Unvisited children (`UCB1 = +inf`) are still always explored first.

- **`puct`**: AlphaZero-style PUCT (Predictor + UCB applied to Trees) - deterministic, always picks the child maximizing
  ```
  PUCT = Q + c * P(i) * sqrt(N_parent) / (1 + N)
  ```
  Since this codebase has no learned policy network to predict `P(i)`, a uniform prior `P(i) = 1 / num_children` is used instead. Unlike `ucb1`, an unvisited child has `Q = 0` rather than `+inf` - its exploration bonus is largest when `N=0` (via the `1/(1+N)` term), which is what drives initial exploration instead of an infinite sentinel value.

- **`hybrid`**: This codebase's original default (previously the hardcoded, unnamed behavior before `--selection-mode` existed). An epsilon-greedy outer loop like `epsilon_greedy`, but the exploratory branch (probability `epsilon`) draws from a roulette wheel weighted by **UCB1-squared** rather than picking uniformly - a child with twice the UCB1 of another gets four times the selection probability in that branch. This is not a standard technique from the MCTS literature; it's kept for backward compatibility and comparison against the four methods above.

## Tips for Effective Use

1. **Get a Materials Project API key** for `ehull` or `ehull_rdos`, and put it in `config.json` (gitignored) rather than passing it on the command line repeatedly
2. **Use `rdos`** if you don't have a Materials Project API key but do have `doscar_peaks_data_with_U.csv`
3. **Start with default parameters** to understand baseline behavior
4. **Use `ehull_rdos`** (the published study's method) for balanced stability + electronic-structure optimization
5. **Increase `--gamma`** to prioritize electronic structure (rDOS) over hull stability, or `--beta` for the reverse
6. **Increase iterations** (e.g., 2000-5000) for more thorough exploration
7. **Use `u_only` mode** for focused uranium materials discovery
8. **Use `lanthanides_u_extended` mode** for broader, faster lanthanide exploration
9. **Check energy_above_hull values** - aim for < 0.1 eV/atom for synthesizability
10. **Monitor iteration progress plots** to assess convergence

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

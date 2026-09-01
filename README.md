# MCTS Framework for Materials Discovery

A clean, modular Monte Carlo Tree Search (MCTS) implementation for discovering
optimal materials through intelligent exploration of chemical space. The search
core is completely material-agnostic; crystals, molecules, and any custom
material type plug in through four small interfaces.

## Features

- **Material-agnostic core** — the MCTS algorithm knows nothing about
  chemistry; it operates on abstract `Material` / `MoveGenerator` /
  `PropertyEvaluator` / `RewardFunction` interfaces.
- **Three built-in material types**
  - **Intermetallic crystals** — periodic-table element substitution, MACE +
    Materials Project energies, DOSCAR rDOS rewards (ported from the validated
    `mcts_crystal` codebase).
  - **Ternary superhydrides** — host-sublattice substitution on a hydride
    template, scored by the ELF-based Tc estimator (networking value φ,
    molecularity index φ\*, hydrogen fraction and H-projected DOS). Descriptors
    come from a precomputed table or from Quantum ESPRESSO on demand.
  - **Molecules** — functional-group substitution via `molecule-modifier`,
    ML property prediction (melting point, H₂ capacity, synthesizability).
- **Four selection strategies** — UCB1, PUCT, ε-greedy, Boltzmann.
- **Type-safe config** — Pydantic models load/validate YAML or JSON.
- **Async-first** — property evaluation runs in a thread pool.
- **No-duplicate tree** — reserve-on-attach dedup guarantees each material
  appears at most once, without ever losing an unclaimed candidate.
- **Visualization & analysis** — convergence plots, search-tree diagrams,
  property histograms, efficiency metrics, and a text report.
- **Lightweight core** — heavy deps (ASE/MACE/pymatgen, RDKit,
  matplotlib/networkx) are optional extras, lazily imported.

## Architecture

The four interfaces you implement (or reuse) to search any material space:

```python
from mcts_framework import Material, MoveGenerator, PropertyEvaluator, RewardFunction

class MyMaterial(Material):
    def get_identifier(self) -> str: ...   # unique key (dedup + caching)
    def copy(self) -> "MyMaterial": ...

class MyMoves(MoveGenerator[MyMaterial]):
    def generate_moves(self, material): ...    # enumerate neighbors

class MyEvaluator(PropertyEvaluator):
    async def _compute(self, material) -> dict: ...   # compute properties

class MyReward(RewardFunction):
    def compute_reward(self, properties) -> float: ...  # score them
```

The `MCTS` class composes these with a `SelectionStrategy` and runs the
selection → expansion → simulation → backpropagation loop. See
`examples/custom_material.py` for a complete, dependency-free example.

## Installation

Pick whichever environment manager you prefer. The optional-dependency groups
are the same either way:

| Extra | Pulls in | Needed for |
| --- | --- | --- |
| _(none)_ | numpy, pandas, pydantic, pyyaml, tqdm, typer | core + `rdos`-only runs + tests |
| `intermetallic` | ASE, spglib, MACE, pymatgen, matbench-discovery | `ehull*` rollout methods |
| `superhydride` | ASE, spglib, scipy | superhydride search |
| `molecule` | RDKit (+ molecule-modifier, installed separately) | molecule search |
| `viz` | matplotlib, seaborn, networkx | plots + `mcts-run figures` |
| `dev` | pytest, ruff, black, mypy | running the test suite |
| `all` | everything above | — |

### Option 1 — pip

```bash
pip install -e .                      # core only
pip install -e ".[intermetallic]"     # + intermetallic support
pip install -e ".[superhydride]"      # + superhydride support
pip install -e ".[molecule]"          # + molecule support
pip install -e ".[viz]"               # + visualization
pip install -e ".[all]"               # everything + dev tools
```

### Option 2 — uv

[uv](https://docs.astral.sh/uv/) creates a project-local `.venv` from this
`pyproject.toml`, independent of conda. `uv sync` also writes a reproducible
`uv.lock`.

```bash
uv sync --python 3.11 --extra intermetallic --extra viz --extra dev
# or everything:
uv sync --python 3.11 --extra all
```

With uv, prefix the commands below with `uv run` (e.g. `uv run mcts-run …`,
`uv run pytest`) — no manual environment activation needed. The rest of this
README shows the bare commands; they assume an activated env (pip) or an
implicit `uv run` prefix (uv).

## Usage

### Command line

```bash
# Validate a config without running (fast check before a long job):
mcts-run validate --config examples/config_intermetallic.yaml

# Run a search:
mcts-run run --config examples/config_intermetallic.yaml
mcts-run run --config examples/config_superhydride.yaml
mcts-run run --config examples/config_molecule.yaml
```

Each run writes to the configured `output_dir`:
- `summary.json` — run summary (best material, reward, tree size)
- `best_materials.csv` — top materials ranked by own reward, with properties
- `convergence.csv` — per-iteration best reward and unique-material count
- `report.txt` — human-readable analysis report
- `config.yaml` — the exact config the run used (Materials Project key redacted)
- `tree.json` — the explored search tree (structure + per-node stats/properties)

### Post-run figures & tables

`config.yaml` and `tree.json` make each run self-describing, so publication
figures are regenerated straight from a finished run directory — no per-study
script and no parameters to re-specify (gamma, reward method, and data paths all
come from the run's own `config.yaml`):

```bash
mcts-run run     --config study.yaml       # writes results + config.yaml + tree.json
mcts-run figures --run-dir mcts_results    # regenerates table + both figures
```

`mcts-run figures` writes into `<run-dir>/figures/` (override with `--out-dir`):

- `top{N}_table.tex` — top-N candidate LaTeX table (`--top-n`, default 15), with
  a "True Rank" column ranking each pick against the full design space
- `ehull_vs_rdos.png` — E_hull vs r_DOS scatter: full design space (backdrop)
  plus the run's top-N overlay
- `radial_tree.png` — 4-panel radial search tree (reward / r_ehull / γ·r_DOS /
  Q·N⁻¹), root starred, expansion edges bold

Needs the `[viz]` extra (and, for intermetallics, the MACE cache + DOSCAR data
referenced by the config). The same outputs are available programmatically via
`mcts_framework.postprocessing.generate_study_outputs(run_dir)`.

### Python API

```python
import asyncio
from mcts_framework import MCTS, UCB1
from mcts_framework.intermetallic import (
    IntermetallicStructure, PeriodicTableMoves, MaceEvaluator,
    DoscarRewardLookup, create_intermetallic_reward,
)
from ase.io import read

atoms = read("structure.cif")
doscar = DoscarRewardLookup("doscar_peaks_data_with_U.csv")

mcts = MCTS(
    root_material=IntermetallicStructure(atoms),
    move_generator=PeriodicTableMoves(f_block_mode="u_only"),
    property_evaluator=MaceEvaluator(mp_api_key="..."),
    reward_function=create_intermetallic_reward("ehull_rdos", doscar,
                                                beta=1.0, gamma=0.0001),
    selection_strategy=UCB1(),
    exploration_constant=0.1,
)

asyncio.run(mcts.run(iterations=1000))
for node in mcts.get_best_materials(n=10):
    print(node.material.get_identifier(), node.own_reward)
```

Runnable example scripts:
- `examples/custom_material.py` — dependency-free toy material (great starting point)
- `examples/run_intermetallic.py` — crystal search (rDOS reward; needs `[intermetallic]`)
- `examples/run_molecule.py` — molecule search (needs `[molecule]` + molecule-modifier)

**Production studies**: The `intermetallic_study/` directory contains complete configurations for:
- **U-only** and **Lanthanide+U** product-mode studies (5 seeds each, 1000/500 iterations)
- **Sensitivity analyses** across 4 hyperparameters (starting material, termination limit, rollout depth, move step)
  - 18 systematic runs varying one parameter while holding others constant
  - Publication-quality 3"×3" learning curves showing exploration vs. reward trade-offs

See [`intermetallic_study/README.md`](intermetallic_study/README.md) for details.

Migrating from the original `mcts_crystal` code? See [MIGRATION.md](MIGRATION.md).

## Project structure

```
mcts_materials/          # repo root
├── src/mcts_framework/
│   ├── core/           # Material-agnostic: interfaces, SearchNode, selection, MCTS, config
│   ├── intermetallic/  # Crystal structures, periodic-table moves, MACE/MP evaluator, rewards
│   ├── superhydride/   # Hydride templates, host substitution, ELF descriptors, Tc reward
│   │   └── qe/         # Quantum ESPRESSO ground-state funnel for those descriptors
│   ├── molecule/       # RDKit structures, functional-group moves, ML evaluator, rewards
│   ├── viz/            # Analysis (metrics/report) and plots (convergence/tree/distribution)
│   ├── postprocessing/ # Regenerate study outputs from a run: tables, scatter, radial tree, driver
│   └── cli/            # `mcts-run` entry point (Typer): main, builders, results
├── tests/              # pytest suite (core is dependency-free; material tests skip if deps absent)
├── examples/           # Config templates, run_intermetallic.py, run_molecule.py, custom_material.py
├── intermetallic_study/  # Production studies: u_only, lanthanide_u, sensitivity analyses
└── reference/          # Kept-aside material (legacy sensitivity_studies/ for reference)
```

## Design notes

### Identity / deduplication
- **Intermetallics**: identifier is `<formula>|SG<number>|<Wyckoff-decoration>`
  (via spglib), so different site decorations at equal composition never
  collide, and atom reordering is irrelevant.
- **Superhydrides**: same `<formula>|SG<number>|<Wyckoff-decoration>` scheme.
  The decoration matters here: an XYH8 template has two distinct host sites, so
  swapping which host occupies which site is a different material at identical
  composition and space group.
- **Molecules**: identifier is the RDKit canonical SMILES.
- The tree reserves a material's identifier **only when it is attached** as a
  child; a candidate generated but not yet attached stays available to whoever
  attaches it first — no duplicates, nothing lost.

### Ranking
`get_best_materials()` ranks by each node's **own** evaluated reward
(`own_reward`), not the backprop-accumulated subtree maximum (`subtree_best`),
so internal nodes don't inherit their best descendant's score. Likewise the
run-level `best_node` / `best_reward` track `own_reward`, so the reported best
material and its reward are always self-consistent.

### Preserved physics
The intermetallic rewards preserve the validated constants from `mcts_crystal`:
`ehull_reward = -tanh(120·(E_hull − 0.05))` and rDOS Gaussian width σ = 0.5 eV.

### Superhydride reward and expansion rule
The reward is Equation 2 of Belli, Torres, Contreras-García and Zurek,
*Ann. Phys. (Berlin)* **2025**, 537, e00280 — a symbolic-regression fit over
244 binary and ternary hydrides (RMSE 41 K, MAE 31 K, max deviation 108 K):

```
Tc = 422.2 · (27/4) · (φ*² − φ*³) · H_f³ · (φ·H_DOS)^⅓ + 5.5   [K]
```

- `φ` **networking value** — highest ELF isovalue whose isosurface spans the
  crystal in all three directions.
- `φ*` **molecularity index** — highest ELF isovalue at which two hydrogen
  atoms connect.
- `H_f` hydrogen fraction; `H_DOS` hydrogen share of the DOS at E_F.

The `27/4` normalises `(φ*² − φ*³)` to 1 at its peak `φ* = 2/3` (the paper
quotes the empirical optimum at 0.68), so 422.2 K is the entire dynamic range
and the fit is **bounded to [5.5, 427.7] K**. `normalize_reward` (default) maps
that onto (0, 1] by dividing by 427.7 — a monotone rescaling that leaves the
ranking alone but keeps rewards O(1), which `exploration_constant` assumes.
The fit is monotone in φ, H_f and H_DOS but **not** in φ*: pushing φ* past 2/3
towards 1 *lowers* Tc, because intact H₂ molecules put their states away from
E_F. Candidates whose descriptors are missing score 0.0, which is below any
real estimate since the fit cannot return less than 5.5 K.

**Stability is deliberately not scored.** Screening the survivors for
thermodynamic and dynamic stability is a separate, later step.

**Where the descriptors come from.** `evaluator: table` reads φ, φ\* and H_DOS
from a CSV (`formula,phi,phi_star,h_dos`), so the search runs with no DFT stack
at all; compositions absent from it score 0.0, which makes a table-less run a
cheap enumeration of what is worth computing. `evaluator: quantum_espresso`
computes them per candidate by running

```
[vc-relax ×2] → scf → nscf → pp.x (ELF cube) → projwfc.x (projected DOS)
```

Three things that subpackage refuses to leave to chance, because each produces
numbers rather than errors: `JOB DONE.` gates every step (pw.x exits 0 on
several genuine failures); each candidate gets its own working directory *and*
`outdir` (concurrent runs sharing one read each other's wavefunctions); and
`vc-relax` runs twice, because the plane-wave basis is defined on the cell the
run *started* from, so a single pass can declare convergence while its stress
is 100+ kbar out. Its CSV cache is written in the descriptor-table schema, so a
finished campaign replays as a `descriptor_table_path`.

**Expansion**: one move substitutes *one* host species for a chemically
adjacent element (same group ±1 period, same period ±1 atomic number — which
walks the lanthanide chain and its Ba/Hf ends — plus the ±32 Ln↔An analog
moves), leaving the hydrogen sublattice and the template untouched. This is the
process the paper describes for the ternary datasets themselves: "once an
optimal template is discovered, new structures are generated via atomic
substitution of elements with similar chemistries in a given supercell."
Branching is the **sum** over host sites rather than the product, so tree depth
measures chemical distance from the starting composition. Because H_f is fixed
by the template and Tc scales as H_f³, **choosing the template chooses the
ceiling**.

### Search behavior (matches current `mcts_crystal`)
- **`move_step`** (intermetallic): max positions a substitution may jump along
  the transition-metal / Group IV / lanthanide axes (default 1 = adjacent;
  3 = extended-range). This is the sole knob for jump distance.
- **`u_bridge`** (intermetallic): which lanthanides U(92) connects to in the
  lanthanide/U modes — `narrow` (Nd only, default) or `wide` (Nd/Gd/Er).
  `move_step` (jump distance) and `u_bridge` (U connectivity) are orthogonal;
  together they replace the old conflated `lanthanides_u_extended` mode.
- **`n_rollout`** (core): number of random lookahead walks drawn per expanded
  node, *in addition* to the node's own depth-0 evaluation (so a node is scored
  over `n_rollout + 1` samples). `n_rollout=1` draws one walk; `n_rollout=0`
  disables lookahead (value is just the node's own reward).
- **`rollout_aggregation`** (core): how a node's samples combine — `max`
  (default; best reward reachable within `rollout_depth` steps) or `mean`
  (unbiased average). Samples are undiscounted (evaluations are deterministic).
- **max-along-walk rollouts**: a depth>0 rollout scores every composition along
  the random walk and returns the max, extracting up to `rollout_depth`
  candidate evaluations per walk instead of only the endpoint.

### `search_mode` — efficiency vs. breadth
Controls when the run stops, trading evaluation cost (DFT/MACE calls) against
top-N coverage:
- **`fast`** (default): stop as soon as the root converges (its
  no-improvement countdown fires). Fewest evaluations; finds the single
  optimum quickly.
- **`thorough`**: ignore root convergence and use the full `iterations`
  budget (stopping early only on true exhaustion — every branch terminated).
  Explores more compounds for a better ranked top-N candidate list, at the
  cost of more evaluations. Pair with a larger `exploration_constant` for
  wider coverage.

## Development

```bash
# pip:
pip install -e ".[dev]"
pytest                    # run the suite

# uv:
uv sync --extra dev
uv run pytest
```

**Testing note (molecule integration):** the molecule unit tests mock the
`molecule-modifier` API (its real package + model files aren't required for the
suite). Those mocks assume the documented API shape; a one-time validation pass
against the real installed `molecule-modifier` is still recommended before
production use, to confirm argument names and prediction DataFrame columns match.

## Copyright

© 2026. Triad National Security, LLC. All rights reserved. Produced under U.S.
Government contract 89233218CNA000001 for Los Alamos National Laboratory.

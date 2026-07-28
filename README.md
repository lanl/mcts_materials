# MCTS Framework for Materials Discovery

A clean, modular Monte Carlo Tree Search (MCTS) implementation for discovering
optimal materials through intelligent exploration of chemical space. The search
core is completely material-agnostic; crystals, molecules, and any custom
material type plug in through four small interfaces.

## Features

- **Material-agnostic core** — the MCTS algorithm knows nothing about
  chemistry; it operates on abstract `Material` / `MoveGenerator` /
  `PropertyEvaluator` / `RewardFunction` interfaces.
- **Two built-in material types**
  - **Intermetallic crystals** — periodic-table element substitution, MACE +
    Materials Project energies, DOSCAR rDOS rewards (ported from the validated
    `mcts_crystal` codebase).
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
| `molecule` | RDKit (+ molecule-modifier, installed separately) | molecule search |
| `viz` | matplotlib, seaborn, networkx | plots + `mcts-run figures` |
| `dev` | pytest, ruff, black, mypy | running the test suite |
| `all` | everything above | — |

### Option 1 — pip

```bash
pip install -e .                      # core only
pip install -e ".[intermetallic]"     # + intermetallic support
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

Migrating from the original `mcts_crystal` code? See [MIGRATION.md](MIGRATION.md).

## Project structure

```
mcts_materials/          # repo root
├── src/mcts_framework/
│   ├── core/           # Material-agnostic: interfaces, SearchNode, selection, MCTS, config
│   ├── intermetallic/  # Crystal structures, periodic-table moves, MACE/MP evaluator, rewards
│   ├── molecule/       # RDKit structures, functional-group moves, ML evaluator, rewards
│   ├── viz/            # Analysis (metrics/report) and plots (convergence/tree/distribution)
│   ├── postprocessing/ # Regenerate study outputs from a run: tables, scatter, radial tree, driver
│   └── cli/            # `mcts-run` entry point (Typer): main, builders, results
├── tests/              # pytest suite (core is dependency-free; material tests skip if deps absent)
├── examples/           # Config templates, run_intermetallic.py, run_molecule.py, custom_material.py
└── reference/          # Kept-aside material (e.g. sensitivity_studies/ for the future sweep harness)
```

## Design notes

### Identity / deduplication
- **Intermetallics**: identifier is `<formula>|SG<number>|<Wyckoff-decoration>`
  (via spglib), so different site decorations at equal composition never
  collide, and atom reordering is irrelevant.
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

### Search behavior (matches current `mcts_crystal`)
- **`move_step`** (intermetallic): max positions a substitution may jump along
  the transition-metal / Group IV / lanthanide axes (default 1 = adjacent;
  3 = extended-range). This is the sole knob for jump distance.
- **`u_bridge`** (intermetallic): which lanthanides U(92) connects to in the
  lanthanide/U modes — `narrow` (Nd only, default) or `wide` (Nd/Gd/Er).
  `move_step` (jump distance) and `u_bridge` (U connectivity) are orthogonal;
  together they replace the old conflated `lanthanides_u_extended` mode.
- **`rollout_aggregation`** (core): how a node's `n_rollout` samples combine —
  `max` (default; extra samples discounted by `0.9**rollout_depth`) or `mean`
  (unbiased average of undiscounted samples).
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

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

```bash
# Core only (numpy, pandas, pydantic, pyyaml, tqdm)
pip install -e .

# With intermetallic support (ASE, spglib, MACE, pymatgen, matbench-discovery)
pip install -e ".[intermetallic]"

# With molecule support (RDKit; plus the molecule-modifier package separately)
pip install -e ".[molecule]"

# With visualization (matplotlib, seaborn, networkx)
pip install -e ".[viz]"

# Everything + dev tools
pip install -e ".[all]"
```

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
mcts-framework/
├── src/mcts_framework/
│   ├── core/           # Material-agnostic: interfaces, SearchNode, selection, MCTS, config
│   ├── intermetallic/  # Crystal structures, periodic-table moves, MACE/MP evaluator, rewards
│   ├── molecule/       # RDKit structures, functional-group moves, ML evaluator, rewards
│   ├── viz/            # Analysis (metrics/report) and plots (convergence/tree/distribution)
│   └── cli/            # `mcts-run` entry point (Typer): main, builders, results
├── tests/              # pytest suite (core is dependency-free; material tests skip if deps absent)
├── examples/           # Config templates, run_intermetallic.py, run_molecule.py, custom_material.py
└── MIGRATION.md        # Mapping from the original mcts_crystal codebase
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
  3 = extended-range). Unifies the old `lanthanides_u` / `lanthanides_u_extended`
  distinction.
- **`rollout_aggregation`** (core): how a node's `n_rollout` samples combine —
  `max` (default; extra samples discounted by `0.9**rollout_depth`) or `mean`
  (unbiased average of undiscounted samples).
- **max-along-walk rollouts**: a depth>0 rollout scores every composition along
  the random walk and returns the max, extracting up to `rollout_depth`
  candidate evaluations per walk instead of only the endpoint.

## Development

```bash
pip install -e ".[dev]"
pytest                    # run the suite
```

**Testing note (molecule integration):** the molecule unit tests mock the
`molecule-modifier` API (its real package + model files aren't required for the
suite). Those mocks assume the documented API shape; a one-time validation pass
against the real installed `molecule-modifier` is still recommended before
production use, to confirm argument names and prediction DataFrame columns match.

## Copyright

© 2026. Triad National Security, LLC. All rights reserved. Produced under U.S.
Government contract 89233218CNA000001 for Los Alamos National Laboratory.

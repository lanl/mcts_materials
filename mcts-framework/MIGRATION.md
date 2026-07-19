# Migrating from `mcts_crystal` to `mcts-framework`

`mcts-framework` is a ground-up redesign of the original `mcts_crystal`
codebase. It is a **separate, standalone package** — the two can coexist, and
nothing in `mcts_crystal` is modified. This guide maps the old concepts to the
new ones.

## Why the rewrite

The original code fused everything into one `MCTSTreeNode` class: tree
bookkeeping, ASE crystal representation, periodic-table substitution, MACE/
Materials Project energy calls, and reward math. That made it hard to add new
material types (molecules, alloys). The new framework separates concerns into
four small interfaces that any material type implements, driving one generic
MCTS core:

| Interface | Responsibility |
|---|---|
| `Material` | what a candidate *is* + a unique identifier |
| `MoveGenerator` | how to enumerate neighbors (expansion) |
| `PropertyEvaluator` | how to compute properties (simulation) |
| `RewardFunction` | how to score properties |

## Concept mapping

| `mcts_crystal` | `mcts-framework` |
|---|---|
| `MCTSTreeNode` (crystal + tree + rewards) | `SearchNode` (tree only) wrapping an `IntermetallicStructure` (crystal) |
| `MCTS` | `mcts_framework.core.mcts.MCTS` (generic over material type) |
| `MaceEnergyCalculator` | `mcts_framework.intermetallic.MaceEvaluator` |
| `DoscarRewardLookup` | `mcts_framework.intermetallic.DoscarRewardLookup` (unchanged behavior, one bug fixed*) |
| `rollout_method="ehull"` (string) | `EhullReward()` object |
| `rollout_method="ehull_rdos"` | `EhullRdosReward(doscar, beta, gamma)` |
| `rollout_method="rdos"` | `RdosReward(doscar)` |
| `node.e_form`, `node.e_above_hull` | `node.properties["e_form"]`, `node.properties["e_above_hull"]` |
| `f_block_mode="experimental"` | `f_block_mode="lanthanides_u_no_wrap"` (old name still accepted as an alias) |
| `python run_mcts.py --rollout-method ehull ...` | `mcts-run run --config config.yaml` |

\* The DOSCAR valence-suffix lookup bug is fixed in the new code; results for
the published U-only study are unaffected (all its compounds had core entries).

## Programmatic API: before / after

**Old (`mcts_crystal`):**

```python
from mcts_crystal.node import MCTSTreeNode
from mcts_crystal.mcts import MCTS
from ase.io import read

atoms = read("structure.cif")
root = MCTSTreeNode(atoms, f_block_mode="u_only")
mcts = MCTS(root, epsilon=0.2)
mcts.run(iterations=1000, rollout_method="ehull", ...)
```

**New (`mcts-framework`):**

```python
import asyncio
from ase.io import read
from mcts_framework import MCTS, UCB1
from mcts_framework.intermetallic import (
    IntermetallicStructure, PeriodicTableMoves, MaceEvaluator, EhullReward,
)

atoms = read("structure.cif")
mcts = MCTS(
    root_material=IntermetallicStructure(atoms),
    move_generator=PeriodicTableMoves(f_block_mode="u_only"),
    property_evaluator=MaceEvaluator(mp_api_key="...", cache_path="cache.csv"),
    reward_function=EhullReward(),
    selection_strategy=UCB1(),
    exploration_constant=0.1,
    n_rollout=5,
    rollout_depth=1,
    seed=0,
)
asyncio.run(mcts.run(iterations=1000))

for node in mcts.get_best_materials(n=10):
    print(node.material.get_identifier(), node.own_reward)
```

Key differences:

- **`run()` is async** — wrap it in `asyncio.run(...)`.
- **Components are injected**, not baked into the node. Swap
  `IntermetallicStructure`+`PeriodicTableMoves`+`MaceEvaluator` for the
  `molecule.*` equivalents to search molecules with the same core.
- **Selection is a strategy object** (`UCB1()`, `PUCT()`, `EpsilonGreedy()`,
  `Boltzmann()`) instead of a mode string.
- **Rewards are objects**, not `rollout_method` strings.

## CLI: before / after

**Old:** `python run_mcts.py --iterations 1000 --rollout-method ehull_rdos --beta 1.0 --gamma 0.0001`

**New:** write a config file and run `mcts-run run --config my_run.yaml`. See
`examples/config_intermetallic.yaml`. Validate a config without running via
`mcts-run validate --config my_run.yaml`.

## Identifiers changed

`mcts_crystal` keyed compounds by chemical formula (e.g. `Pb6U1W6`). The new
`IntermetallicStructure.get_identifier()` returns a **crystallographic**
identifier — `"<formula>|SG<number>|<Wyckoff-decoration>"` — so structures that
share a composition but differ in space group / site decoration never collide.
Use `IntermetallicStructure.get_formula()` to recover the plain formula (e.g.
for a formula-keyed energy cache; `MaceEvaluator`'s CSV cache still keys by
formula, matching the old cache files).

## What carried over unchanged

- The `ehull_reward` formula `-tanh(120 * (e_hull - 0.05))` and the rDOS
  Gaussian width `sigma = 0.5` (physics-informed constants).
- The periodic-table move rules (transition-metal / Group IV / f-block
  substitutions), ported verbatim including all edge cases.
- The MACE + Materials Project energy pipeline and its CSV cache schema
  (`name, e_form, e_above_hull, e_decomp, data_quality`).

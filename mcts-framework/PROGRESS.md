# MCTS Framework — Progress & Handoff Notes

Living status doc for the from-scratch MCTS framework. Read this first when
resuming work. The full design rationale lives in the approved plan file:
`~/.claude/plans/i-would-like-to-generic-orbit.md`.

## How to resume the Claude Code session

Session transcripts are saved automatically under
`~/.claude/projects/-Users-ywl-Research-mcts-materials/<session-id>.jsonl`.

- `claude --continue` (or `-c`) — reopen the most recent session with context.
- `claude --resume` (or `-r`) — pick from a list of past sessions.

Transcripts are auto-cleaned after `cleanupPeriodDays` (default 30) — bump that
in `~/.claude/settings.json` or copy the `.jsonl` elsewhere to keep this long
term. The code, tests, this file, and the plan file all persist in the repo
regardless.

## Current status

Test suite: **155 passed, 2 skipped** (`pytest` from the `mcts-framework/` dir).
The 2 skips are RDKit-dependent molecule tests (RDKit not installed in the
current env — see caveats).

### Upstream sync (adopted latest mcts_crystal development)

After rebasing `refactor` onto `origin/main` (17 upstream commits), the
framework was updated to match the *current* mcts_crystal search behavior:

- **`move_step`** (intermetallic): generalizes the metal/Group IV/lanthanide
  move range (default 1 = adjacent; larger = extended). Unifies
  `lanthanides_u` (step 1) and `lanthanides_u_extended` (step 3). Lives in
  `intermetallic/elements.py` move fns + `PeriodicTableMoves`; move fns now
  return SORTED lists (same element sets as before, matching upstream).
- **`rollout_aggregation`** (core): `max` (default; extra samples discounted
  by `0.9**rollout_depth`) or `mean` (unbiased average, undiscounted). In
  `core/mcts.py` + `MCTSConfig`.
- **max-along-walk rollouts** (core): depth>0 rollouts now score every step of
  the random walk and return the max, not just the endpoint.
- Defaults chosen to match mcts_crystal exactly (move_step=1,
  rollout_aggregation='max', max-along-walk). `node.own_reward` remains the
  undiscounted depth-0 value regardless of aggregation (preserves the
  global-best-consistency fix).

### Second upstream catch-up: ehull_rdos_product reward

`origin/main` had NOT moved (still 168b220); the new work lived on
`origin/develop` (#29 "multiple reward"), whose sole functional addition was a
multiplicative reward method. Per user decision, we did NOT rebase (main
unchanged) and only ported the reward:

- **`EhullRdosProductReward`** (`intermetallic/rewards.py`) — new rollout
  method `ehull_rdos_product`. Plumbed through the config `rollout_method`
  Literal (+ needs mp_api_key + doscar_data_path), builders, and factory.
- **Deliberate divergence from mcts_crystal**: upstream computes
  `ehull_reward(e_hull) * (gamma * r_DOS)`; the framework drops gamma
  (`ehull_reward(e_hull) * r_DOS`) because a global scalar cannot change a
  purely multiplicative ranking. See MIGRATION.md for the rationale.

### Done
- **Core** (`src/mcts_framework/core/`): Material, MoveGenerator,
  PropertyEvaluator, RewardFunction abstractions; SearchNode; 4 selection
  strategies (UCB1/PUCT/EpsilonGreedy/Boltzmann); MCTS algorithm; Pydantic
  config models.
- **Intermetallic** (`src/mcts_framework/intermetallic/`): structure (spglib
  Wyckoff-decorated identifier), periodic-table moves (ported verbatim from
  mcts_crystal), MACE+Materials Project evaluator (CSV cache), DOSCAR rDOS
  lookup, and ehull / rdos / ehull_rdos rewards.
- **Molecule** (`src/mcts_framework/molecule/`): SMILES-identity structure,
  functional-group moves (molecule-modifier), property evaluator
  (Chemprop/XGBoost/H2/synthesizability), and rewards.

- **CLI** (`src/mcts_framework/cli/`): Typer app with `run` and `validate`
  commands. `builders.build_mcts(config)` dispatches on material_type and
  assembles components (lazy heavy imports); `results.save_results` writes
  summary.json + best_materials.csv + convergence.csv. Entry point
  `mcts-run = "mcts_framework.cli.main:app"`. Verified installed & working
  (`mcts-run validate --config ...`).

- **Viz + analysis** (`src/mcts_framework/viz/`): `plots.py` (plot_convergence,
  plot_property_distribution, plot_search_tree — matplotlib/networkx lazy) and
  `analysis.py` (compute_metrics, rank_materials, generate_report — pure
  Python). The CLI's save_results now also writes `report.txt` via
  generate_report. (This subpackage was implemented in a prior session; verified
  passing, 10 viz tests.)

- **Docs & examples** (Phase 7): README updated (CLI `run`/`validate`
  subcommands, correct output files, cli/ package, subtree_best rename);
  `MIGRATION.md` written (mcts_crystal -> mcts-framework mapping);
  `examples/` has config_intermetallic.yaml (defaults to rdos, validates
  out-of-the-box) + config_molecule.yaml, and runnable scripts
  custom_material.py, run_intermetallic.py, run_molecule.py. The Pb6U1W6 CIF
  is bundled in examples/. run_intermetallic.py verified end-to-end (rDOS
  degrades to 0.0 with a warning if the DOSCAR CSV is absent) - this was the
  first real exercise of the intermetallic pipeline (caveat #2 partially
  retired: expansion + spglib identity + rdos reward path now confirmed;
  live MACE/MP energy path still unexercised).

### Project status: feature-complete
All planned phases (1-7) done. 138 passed, 2 skipped.

(General reminder for future sessions: prior sessions have added files ahead of
the todo; always `ls`/read before writing a "new" file to avoid duplicating
work, as happened with the CLI test, the viz module, and the examples.)

## Cleanup pass (post Phase 6)

A holistic walkthrough was done before Phase 7. Changes:

- **Deleted dead code**: orphaned `src/mcts_framework/cli.py` (old argparse
  version, shadowed by the `cli/` package, referenced by nothing).
- **Bug fix - global best consistency**: `MCTS.best_node`/`best_reward` (and
  the `reward_history` convergence curve) now track each node's `own_reward`,
  not the rollout-max returned by `_simulate`. Previously the reported best
  reward could come from a discounted rollout of a *different* material than
  `best_node`, so the summary/report could show a mismatched material/reward
  pair. Locked in by `test_best_node_and_reward_are_consistent`.
- **Rename for clarity**: `SearchNode.best_reward` -> `SearchNode.subtree_best`
  (it is a backprop-accumulated subtree maximum used only by the termination
  heuristic; the old name collided conceptually with the global
  `MCTS.best_reward`). `own_reward` remains the node's own evaluated value.
- **Recursion -> loop**: `MCTS._select` no longer recurses when a branch is
  fully terminated (could hit the recursion limit on deep dead trees); it
  restarts the walk from the root via an outer loop.
- **De-duplicated**: extracted module-level `_resolve_rdos()` in
  intermetallic/rewards.py (was copy-pasted in RdosReward and EhullRdosReward).
- **Trimmed**: removed unused `SearchNode.get_path_to_root` (+ its test).
- Tests: 138 passed, 2 skipped after the pass.

## Key design decisions (locked in with the user)

- **Standalone package** `mcts-framework`, coexists with old `mcts_crystal`.
  Core deps only; material deps are optional extras (`[intermetallic]`,
  `[molecule]`, `[viz]`). Heavy libs imported lazily.
- **Dedup = "reserve only on attach"** (MCTS): a material identifier is
  recorded in `visited_materials` only when a node is actually attached, and
  `_expand` re-checks candidates at attach time. Guarantees no duplicate ever
  enters the tree and never loses a generated-but-unattached candidate.
- **Node reward fields**: `own_reward` = this node's own simulated reward (use
  for ranking real candidates); `best_reward` = subtree max accumulated during
  backprop (do NOT rank materials by this — it floats the root). See
  `get_best_materials`.
- **Intermetallic identity** = `<formula>|SG<number>|<Wyckoff-decoration>` via
  spglib (order-independent; different site decorations never collide). Use
  `get_formula()` for the composition-only string (energy cache key).
- **f_block_mode rename**: `experimental` -> `lanthanides_u_no_wrap`; old name
  still accepted as an alias in config.
- **DOSCAR valence bug FIXED**: original mcts_crystal kept the `_valence`
  suffix in reward-dict keys, making the valence fallback unreachable. The port
  strips the suffix. Published U-only results unaffected (all had core entries).
- **Melting-point reward**: favors HIGH melting points, **linear**, normalized
  to `[min_temp, max_temp]`, **unclamped** (values outside the window extend
  past [0,1]). Not tanh.
- **Physics constants preserved**: ehull reward `-tanh(120*(e_hull-0.05))`,
  rDOS Gaussian sigma=0.5.

## Open caveats / risks

1. **Molecule integration is unit-tested only against MOCKS.** The moves and
   evaluator tests mock `molecule_modifier`, so they verify our call/return
   handling but assume the real API's signatures and DataFrame columns
   (`melting_temp`, `h2_capacity`, `synthesizability`). Not yet validated
   against the real package. **Needs a real-dependency integration pass.**
2. **Live MACE / Materials Project path is untested.** The intermetallic
   evaluator's cache logic and reward math are tested, but the actual MACE
   relaxation + MP phase-diagram calls have not been exercised end-to-end here.
3. **2 skipped tests** require RDKit (`test_molecule/test_moves.py`,
   `test_molecule/test_structure_evaluator.py`).

## Installing optional deps for full testing

```bash
# From the mcts-framework/ directory:
pip install -e ".[dev]"                 # core + pytest (always)
pip install -e ".[intermetallic]"       # ase, spglib, mace-torch, matbench-discovery, pymatgen
                                        # (spglib already installed in current env)

# RDKit (canonical channel is conda-forge; pip wheels also work):
conda install -c conda-forge rdkit      # or: pip install rdkit

# molecule-modifier is a LOCAL editable package (not on PyPI), from its repo:
cd /path/to/molecule-modifier
pip install -e ".[prediction]"          # chemprop 2.2.1, torch, lightning
# Point at external model files (needed for melting_point):
export CHEMPROP_MODEL_DIR=".../external_models/chemprop_model/models"
export XGBOOST_MODEL_DIR=".../external_models/Balu/251212_1254_br5000"
# (or place external_models/ as a sibling dir for auto-detection)
```

### Recommended next testing step
Add real-dependency **integration tests** (separate from the mocked unit
tests), e.g. marked `@pytest.mark.integration` with a skip guard when RDKit /
molecule-modifier / model files are absent. They should: parse a known SMILES,
run one real functional-group substitution, and run one real property
prediction, asserting sane ranges — catching any drift between the mock's
assumed API and molecule-modifier's actual API.

## Layout

```
mcts-framework/
  pyproject.toml          # deps, extras, mcts-run entry point, pytest config
  README.md
  PROGRESS.md             # this file
  src/mcts_framework/
    core/                 # material, move_generator, evaluator, reward,
                          # search_node, selection, mcts, config
    intermetallic/        # structure, moves, elements, evaluator, doscar, rewards
    molecule/             # structure, moves, evaluator, rewards, functional_groups
    (cli/  — TODO)
  tests/
    test_core/            # config, mcts, search_node, selection
    test_intermetallic/   # elements, structure_moves, doscar_rewards
    test_molecule/        # rewards (always), moves + structure_evaluator (need RDKit)
```

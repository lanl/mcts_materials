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

Test suite: **231 passed** (`pytest` from the repo root). Skips only when
optional deps (RDKit) or example data files are absent; with `[all]` installed
the full suite runs.

**Layout note:** `mcts-framework/` was promoted to the repo root — the package
now lives at `src/mcts_framework/`, tests at `tests/`, examples at `examples/`,
and the old `mcts_crystal` package + analysis/sensitivity scripts were deleted
(`sensitivity_studies/` kept aside under `reference/`). References below to a
`mcts-framework/` subdirectory are historical.

### Postprocessing subpackage (done this session)

Ported the analysis figures/tables from the original `mcts_crystal`
`analysis/*/generate_figures.py` scripts into one reusable, config-driven
`src/mcts_framework/postprocessing/` package (no per-study duplication):

- **`design_space.py`** — `full_formula_key` (default ranking key, full
  composition), `score_by_method` (reproduces all four reward classes exactly;
  pinned to them by a test), `rank_design_space` (dispatches on the run's
  `rollout_method`, attaches rDOS per-compound via the full formula — the old
  f-block-stripping `tm_giv_key` machinery was deleted).
- **`tables.py`** — `write_top_n_table`: N is a parameter; gamma/beta/data
  paths/rollout_method read from the run's own `Config`; default ranks the full
  design space; method-agnostic columns (r_DOS, r_ehull, Reward).
- **`scatter.py`** — `plot_ehull_vs_rdos`: E_hull vs raw r_DOS backdrop + run
  top-N overlay (matched by full-formula key); pluggable `space_filter`,
  synthesized/attempted overlays.
- **`radial_tree.py`** — `plot_radial_tree`: 4-panel radial tree
  (reward / r_ehull / γ·r_DOS / Q·N⁻¹) from a run's persisted `tree.json`.
- **`driver.py`** — `generate_study_outputs(run_dir)`: reads `config.yaml` +
  `tree.json` and regenerates table + both figures into `run_dir/figures/`.
  Exposed as `mcts-run figures --run-dir DIR`.
- **Tree persistence**: `MCTS.to_tree_dict` / `save_tree_json` write the
  explored tree as portable JSON; `save_results` writes `tree.json` +
  `config.yaml` (mp_api_key redacted) by default.

Commits on `refactor`: `9db399b` (tables/ranking), `b9897bf` (scatter + radial
tree + tree persistence), `36bb531` (driver + `mcts-run figures` + README).

### Next up: sweep harness (NOT done — cross-run figures)

The single-run driver above cannot produce the three figures that aggregate
across *many* runs. These are the last piece of the analysis port and require
porting `sensitivity_studies/common.py`'s replicate-run harness (a parameter
grid → many runs → aggregate), which is a different shape from the per-run
driver.

Figures still to port (originals in `analysis/*/generate_figures.py` and
`sensitivity_studies/`):

- **#4 convergence-by-starting-material** — run the search from several
  starting compositions and overlay their convergence curves (best-reward vs
  iteration). Original: `plot_convergence_by_starting_material` in
  `analysis/ehull_rdos_u_only_study/generate_figures.py`.
- **#5 sensitivity sweeps** — sweep a hyperparameter (exploration_constant,
  selection_mode, termination_limit, …), replicate each setting with several
  seeds, and plot the effect on convergence/coverage. Original harness:
  `sensitivity_studies/scripts/common.py` (`BASELINE` dict + `run_replicate`,
  using `override_composition` and `max_reward_history`).
- **#6 iterations-vs-termination** — how final coverage / best reward scales
  with `iterations` and `termination_limit`.

Porting notes / gaps to bridge when we resume:
- `common.py` uses `override_composition` (start-material override); the
  framework equivalent is building the root `IntermetallicStructure` with the
  substituted composition, or the config `transition_metal`/`group_iv`
  overrides on `IntermetallicConfig`.
- `common.py`'s `max_reward_history` maps to the framework's
  `MCTS.reward_history` (already persisted as `convergence.csv`).
- Likely shape: a `postprocessing/sweeps.py` that runs a grid of configs
  (varying one field), collects each run's `convergence.csv`/`summary.json`,
  and emits the overlay/sensitivity plots — plus an `mcts-run sweep` command or
  a thin driver. Decide grid-runner vs. read-existing-runs when we resume.
- `search_mode` interacts with #4/#6: use `thorough` for coverage-sensitive
  sweeps so the root-convergence early stop doesn't cap breadth (see the
  `search_mode` section above).

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

### search_mode knob (fast vs thorough)

Equivalence testing against develop surfaced that the framework consistently
explored fewer compounds than mcts_crystal (e.g. 41 vs 68 for ehull). Root
cause: the framework halts the whole run when the ROOT node self-terminates
(its visits-without-improvement countdown), while mcts_crystal only halts on
true exhaustion. On the near-continuous rollout reward the root's countdown
fires early, leaving much of the reachable space unexplored. (Sweeps confirmed
`termination_limit` barely moves coverage; `exploration_constant` helps but is
capped by the root halt.)

Added `search_mode` on `MCTS` + `MCTSConfig` (plumbed via builders):
- **`fast`** (default): keep the early stop-on-root-convergence — fewest
  evaluations (DFT/MACE), finds the optimum quickly.
- **`thorough`**: ignore root self-termination, use the full iteration budget
  (stop only on true exhaustion) — broader top-N candidate list. Reproduces
  mcts_crystal-style breadth.

Also changed the no-improvement test in `SearchNode.update` from strict `>` to
non-strict `>=` (ties reset the countdown), per-node against `subtree_best`.
See MIGRATION.md "Termination / search_mode" for the full comparison.

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

### Project status

Core framework + all material types + CLI + viz + postprocessing done. The
cross-run **sweep harness** (#4/#5/#6, see "Next up" above) is the remaining
analysis port. 203 passed, 2 skipped.

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

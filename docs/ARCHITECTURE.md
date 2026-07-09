# Architecture

## Scope

The repo currently implements the first executable stage of the proposal:

- ingest public synthesis corpora
- normalize mined records into a shared route schema
- retrieve analogous literature routes for a target
- generate candidate solid-state procedures with a staged grammar
- score completed routes with hard-check style heuristics
- rank routes with Monte Carlo Tree Search

The planner is intentionally a solid-state baseline. Hydrothermal and precipitation records are normalized and stored, but they are not yet expanded into an executable search grammar.

## Active package

All active source code now lives in `synthesis_planner/`.

- `cli.py`
  - entrypoints for `download-data`, `prepare-data`, and `plan`
- `datasets.py`
  - downloads the public datasets
  - normalizes raw records into `RouteRecord` JSONL files
- `formula.py`
  - parses inorganic formulas
  - infers coarse target families
- `schema.py`
  - shared dataclasses for routes, planning state, actions, and scores
- `retrieval.py`
  - finds analogous routes
  - builds precursor priors from literature usage
- `grammar.py`
  - staged solid-state action expansion
- `scoring.py`
  - heuristic route evaluation and deterministic judge output
- `mcts.py`
  - compact PUCT-style tree search
- `planner.py`
  - top-level orchestration and portfolio selection

## Planning pipeline

1. Raw datasets are downloaded into `data/raw/`.
2. `prepare-data` converts them into normalized JSONL route files in `data/processed/`.
3. `plan` loads the processed solid-state routes.
4. Retrieval finds analogous target recipes.
5. Precursor candidates are proposed from exact analog routes and element-level precursor usage.
6. MCTS explores a staged grammar:
   - choose precursor set
   - choose preparation steps
   - choose heating schedule
   - finalize route
7. Rollouts complete partial routes and score them.
8. A small diverse top-k portfolio is written to `planning_results/`.

## Intentional simplifications

- No live LLM calls: the judge is deterministic and rubric-based.
- No live thermodynamics APIs: route scoring is heuristic.
- No multimodal branching yet: only `solid_state` is currently planned end to end.
- No benchmark harness yet beyond the unit suite and smoke-test workflow.

## Repo cleanup

The old crystal-specific package and legacy study directories were removed to keep the repository aligned with the proposal:

- removed `mcts_crystal/`
- removed `analysis/`
- removed `sensitivity_studies/`
- removed `examples/`

That leaves the repo centered on the synthesis-planning package, its tests, and the proposal-driven data workflow.

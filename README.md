# MCTS Materials Synthesis Planner

This repo is now a synthesis-planning project aligned with the proposal in [MCTS_Materials_Synthesis_Project_Proposal.docx](./MCTS_Materials_Synthesis_Project_Proposal.docx). The old crystal-search package and its study directories have been removed so the repository now reflects the new synthesis-first scope directly.

The current codebase focuses on the proposal's first practical milestone:

- public dataset download and normalization
- a canonical route schema for mined synthesis recipes
- a solid-state synthesis grammar over precursor choice, preparation, heating, and termination
- retrieval of analogous literature routes
- heuristic route scoring with hard-check style penalties and a deterministic judge
- Monte Carlo Tree Search over partial synthesis routes

The initial planner is intentionally a strong solid-state baseline, not a claim that the full proposal is finished. The solution-based dataset is downloaded and normalized for future hydrothermal and precipitation support, but the executable planner currently targets `solid_state` routes.

## Public data

The repo now uses the public datasets referenced in the proposal:

- Solid-state dataset: `CederGroupHub/text-mined-synthesis_public`
  - file: `solid-state_dataset_20200713.json.xz`
- Solution dataset: `CederGroupHub/text-mined-solution-synthesis_public`
  - file: `solution-synthesis_dataset_2021-8-5.json.zip`

They are downloaded into `data/raw/` and normalized into JSONL route records in `data/processed/`.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel pytest numpy pandas
.venv/bin/python -m pip install .
```

For development:

```bash
.venv/bin/python -m pip install '.[dev]'
```

## Workflow

Download the public datasets:

```bash
python run_mcts.py download-data
```

Normalize them into route records:

```bash
python run_mcts.py prepare-data
```

Plan routes for a target:

```bash
python run_mcts.py plan --target BaTiO3 --iterations 250 --top-k 5
```

The planner writes ranked routes to `planning_results/` and prints a short summary to the terminal.

## Documentation

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md): package layout, planning pipeline, and current design boundaries
- [docs/DATASETS.md](./docs/DATASETS.md): public dataset sources, local file layout, and normalization outputs

## What the planner does

Given a target such as `BaTiO3`, the current planner:

1. Loads normalized solid-state routes from the mined literature corpus.
2. Retrieves analogous targets by element overlap, target family, and rough composition similarity.
3. Builds precursor candidates from exact analog routes plus element-level precursor usage priors.
4. Expands a staged solid-state grammar:
   - precursor set
   - preparation steps
   - heating schedule
   - finalization
5. Scores completed routes using:
   - element coverage
   - precursor plausibility
   - retrieval support
   - condition plausibility
   - hazard and complexity penalties
   - deterministic judge notes and failure flags
6. Uses MCTS to prioritize promising partial routes and returns a small diverse portfolio.

## Repo layout

- `docs/`
  - architecture and dataset notes for the rewritten planner
- `data/`
  - `raw/`: downloaded public corpora
  - `processed/`: normalized JSONL route records generated locally
- `synthesis_planner/`
  - `datasets.py`: dataset download, loading, normalization
  - `formula.py`: formula parsing and target-family heuristics
  - `retrieval.py`: analog retrieval and precursor prior generation
  - `grammar.py`: staged solid-state grammar
  - `scoring.py`: hard checks, heuristics, deterministic judge
  - `mcts.py`: compact PUCT-style tree search
  - `planner.py`: high-level planning interface
  - `cli.py`: `download-data`, `prepare-data`, `plan`
- `tests/`
  - focused on formula parsing, normalization, retrieval, scoring, planner behavior, and CLI parsing

## Current limitations

- The executable planner is currently `solid_state` only.
- Thermodynamic scoring is heuristic at this stage; the proposal's richer physics integrations are not yet wired in.
- The "LLM judge" is currently a deterministic rubric-based substitute so the repo works offline and remains testable.
- Route scoring is still a baseline heuristic layer rather than a calibrated literature-plus-physics model.

## Verification

Unit tests:

```bash
.venv/bin/python -m pytest
```

The rewritten test suite covers the new synthesis-planning path rather than the legacy crystal-search code.
